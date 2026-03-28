#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
check_trace_ready.py — DDP / TP Trace 驗證與分析工具

檢查 train_rocm_pytorch.py（DDP）與 train_rocm_tensor.py（TP）
產出的 host/device trace 是否符合 Chakra/HTA 的最低需求，
並提供詳細的通訊 pattern 分析。

=== 功能 ===
1. 基本 Smoke Test：host JSON 合法性、ProfilerStep、同步事件、kernel 事件
2. 並行模式辨識：根據通訊 pattern 自動判斷 DDP / TP / 混合模式
3. 通訊統計：每種 collective 的次數、bytes、每步平均
4. 健康檢查：根據模式給出針對性建議
5. 對比分析：可同時分析多個 model-tag 的 trace 做比較

=== 用法 ===
  # 檢查特定模型的 trace
  python ./src/tests/check_trace_ready.py --model-tag qwen15b_tp
  python ./src/tests/check_trace_ready.py --model-tag qwen05b

  # 對比 DDP 與 TP 的 trace
  python ./src/tests/check_trace_ready.py --compare qwen05b qwen15b_tp

  # 檢查所有 trace（自動偵測）
  python ./src/tests/check_trace_ready.py

  # 指定目錄
  python ./src/tests/check_trace_ready.py --traces-dir ./data/chakra/pytorch_traces
"""

import argparse
import json
import os
import re
import sys
from pathlib import Path
from statistics import median
from typing import Dict, List, Tuple, Optional
from collections import defaultdict


# =====================================================================
# Constants
# =====================================================================

KINETO_STEP_PAT = re.compile(r'^ProfilerStep#\d+$')

# Collective communication types to track
COLLECTIVE_TYPES = {
    'all_reduce':      re.compile(r'nccl:all_reduce|nccl_all_reduce|c10d::allreduce', re.I),
    'all_gather':      re.compile(r'nccl:all_gather|nccl_all_gather|c10d::allgather|all_gather', re.I),
    'reduce_scatter':  re.compile(r'nccl:reduce_scatter|nccl_reduce_scatter|c10d::reducescatter|reduce_scatter', re.I),
    'all_to_all':      re.compile(r'nccl:all_to_all|all_to_all|alltoall', re.I),
    'broadcast':       re.compile(r'nccl:broadcast|c10d::broadcast', re.I),
}

# Byte patterns for streaming scan
BYTE_PATTERNS = {
    "profstep":     re.compile(br'"name"\s*:\s*"ProfilerStep#\d+', re.I),
    "hip_rec":      re.compile(br'"name"\s*:\s*"hipEventRecord"', re.I),
    "hip_wait":     re.compile(br'"name"\s*:\s*"hipStreamWaitEvent"', re.I),
    "cuda_rec":     re.compile(br'"name"\s*:\s*"cudaEventRecord"', re.I),
    "cuda_wait":    re.compile(br'"name"\s*:\s*"cudaStreamWaitEvent"', re.I),
    "kernel_like":  re.compile(br'"cat"\s*:\s*"(Kernel|gpu_op)"|"(convolution|GEMM|matmul|hipLaunchKernel)"', re.I),
    # Collective ops (byte-level for streaming)
    "all_reduce":   re.compile(br'nccl:all_reduce|nccl_all_reduce', re.I),
    "all_gather":   re.compile(br'nccl:all_gather|nccl_all_gather', re.I),
    "reduce_scatter": re.compile(br'nccl:reduce_scatter|nccl_reduce_scatter', re.I),
    "all_to_all":   re.compile(br'all_to_all|alltoall', re.I),
    "nccl_generic": re.compile(br'ncclDevKernel_Generic', re.I),
}


# =====================================================================
# Streaming Helpers (memory-efficient for large trace files)
# =====================================================================

def _stream_count(fname: Path, patterns: Dict[str, re.Pattern]) -> Dict[str, int]:
    """Count pattern matches in file using chunked reading."""
    cnt = {k: 0 for k in patterns}
    with fname.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            for key, rgx in patterns.items():
                cnt[key] += len(rgx.findall(chunk))
    return cnt


def _try_parse_json(fname: Path) -> Tuple[bool, str]:
    """Try to parse a JSON file."""
    try:
        with fname.open("r", encoding="utf-8", errors="strict") as f:
            json.load(f)
        return True, "ok"
    except Exception as e:
        return False, str(e)


def _looks_like_concatenated_json(text: str) -> bool:
    return len(re.findall(r'\{\s*"schema"\s*:', text)) >= 2


# =====================================================================
# Detailed Trace Analysis (full JSON parse for smaller files)
# =====================================================================

def parse_trace_detailed(path: Path) -> Dict:
    """
    Parse device trace JSON and extract detailed communication statistics.

    Returns dict with:
      - steps: ProfilerStep count and timing
      - collectives: per-type count, bytes, timing
      - kernels: GPU kernel stats
      - mode: detected parallel mode ('DDP', 'TP', 'TP+DDP', 'unknown')
    """
    obj = json.loads(path.read_text(encoding='utf-8', errors='ignore'))
    unit = str(obj.get('displayTimeUnit', '')).lower()

    def to_ms(d):
        try:
            dv = float(d)
        except Exception:
            dv = 0.0
        return dv if unit == 'ms' else dv / 1000.0

    # Accumulators
    steps = []
    kernel_events = []
    collective_stats = {ctype: {'count': 0, 'total_ms': 0.0, 'bytes_list': []}
                        for ctype in COLLECTIVE_TYPES}
    nccl_generic_count = 0
    bytes_from_tags = defaultdict(int)  # pg_name → total bytes

    for ev in obj.get('traceEvents', []):
        if ev.get('ph') != 'X':
            continue

        name = str(ev.get('name', ''))
        cat = str(ev.get('cat', ''))
        dur = to_ms(ev.get('dur', 0.0))
        lname = name.lower()
        lcat = cat.lower()

        # ProfilerStep
        if KINETO_STEP_PAT.match(name):
            steps.append(dur)

        # Kernel events
        if 'kernel' in lcat or 'nccldevkernel' in lname or 'gpu_op' in lcat:
            kernel_events.append((name, dur))

        # AMD NCCL Generic kernel
        if 'nccldevkernel_generic' in lname:
            nccl_generic_count += 1

        # Collective communication events
        for ctype, regex in COLLECTIVE_TYPES.items():
            if regex.search(lname):
                collective_stats[ctype]['count'] += 1
                collective_stats[ctype]['total_ms'] += dur

                # Extract bytes from tagged name: "nccl:all_reduce|bytes=12345|pg=dp0"
                m = re.search(r'bytes=(\d+)', lname)
                if m:
                    b = int(m.group(1))
                    collective_stats[ctype]['bytes_list'].append(b)

                # Extract process group tag
                pg_m = re.search(r'pg=(\w+)', lname)
                if pg_m and m:
                    bytes_from_tags[pg_m.group(1)] += int(m.group(1))

                break  # Only match first type

    # Compute derived stats
    n_steps = len(steps) if steps else 1

    for ctype in collective_stats:
        cs = collective_stats[ctype]
        cs['per_step'] = cs['count'] / n_steps
        cs['bytes_total'] = sum(cs['bytes_list'])
        cs['bytes_per_step'] = cs['bytes_total'] / n_steps
        if cs['bytes_list']:
            cs['bytes_median'] = int(median(cs['bytes_list']))
            cs['bytes_min'] = min(cs['bytes_list'])
            cs['bytes_max'] = max(cs['bytes_list'])
        else:
            cs['bytes_median'] = 0
            cs['bytes_min'] = 0
            cs['bytes_max'] = 0

    # Detect parallel mode
    ar_count = collective_stats['all_reduce']['count']
    ag_count = collective_stats['all_gather']['count']
    rs_count = collective_stats['reduce_scatter']['count']
    a2a_count = collective_stats['all_to_all']['count']

    if ag_count > 0 and rs_count > 0 and ar_count > 0:
        mode = 'TP+DDP'
    elif ag_count > 0 or rs_count > 0:
        mode = 'TP (with SP)'
    elif ar_count > 0 and (ar_count / n_steps) > 20:
        # TP without SP also uses AllReduce, but many more per step than DDP
        # DDP typically has 5-20 AllReduce per step (gradient buckets)
        # TP has 50+ AllReduce per step (every layer)
        mode = 'TP (AllReduce)'
    elif ar_count > 0:
        mode = 'DDP'
    else:
        mode = 'unknown'

    return {
        'steps_n': len(steps),
        'step_median_ms': median(steps) if steps else None,
        'step_total_ms': sum(steps) if steps else None,
        'kernel_count': len(kernel_events),
        'kernel_total_ms': sum(d for _, d in kernel_events),
        'kernel_per_step_ms': sum(d for _, d in kernel_events) / n_steps,
        'nccl_generic_count': nccl_generic_count,
        'collectives': collective_stats,
        'mode': mode,
        'pg_bytes': dict(bytes_from_tags),
        'top_kernels': sorted(kernel_events, key=lambda x: -x[1])[:10],
    }


# =====================================================================
# Formatting Helpers
# =====================================================================

def _fmt_bytes(b: int) -> str:
    """Format bytes as human-readable string."""
    if b >= 1024**3:
        return f"{b / 1024**3:.2f} GiB"
    elif b >= 1024**2:
        return f"{b / 1024**2:.2f} MiB"
    elif b >= 1024:
        return f"{b / 1024:.1f} KiB"
    return f"{b} B"


def _fmt_ms(ms: float) -> str:
    if ms is None:
        return "N/A"
    if ms >= 1000:
        return f"{ms / 1000:.2f} s"
    return f"{ms:.3f} ms"


# =====================================================================
# Report: Single Model
# =====================================================================

def print_single_report(tag: str, results: Dict[str, Dict]):
    """Print detailed report for a single model tag."""
    ranks = sorted(results.keys())

    # Use first rank as representative (all should be similar)
    rep = results[ranks[0]]

    print(f"\n{'='*70}")
    print(f"  Trace Analysis: {tag}")
    print(f"  Detected Mode: {rep['mode']}")
    print(f"  Ranks: {len(ranks)}")
    print(f"{'='*70}")

    # --- Steps ---
    print(f"\n  Steps per rank: {rep['steps_n']}")
    if rep['step_median_ms']:
        print(f"  Step median:    {_fmt_ms(rep['step_median_ms'])}")

    # --- Kernel ---
    print(f"\n  GPU kernels:    {rep['kernel_count']} events")
    print(f"  Kernel total:   {_fmt_ms(rep['kernel_total_ms'])}")
    print(f"  Kernel/step:    {_fmt_ms(rep['kernel_per_step_ms'])}")

    if rep['nccl_generic_count'] > 0:
        print(f"  AMD Generic NCCL kernels: {rep['nccl_generic_count']}")

    # --- Collective Communication ---
    print(f"\n  {'Collective':<18} {'Count':>7} {'Per Step':>10} {'Bytes Total':>14} {'Bytes/Step':>14} {'Median Size':>14}")
    print(f"  {'-'*18} {'-'*7} {'-'*10} {'-'*14} {'-'*14} {'-'*14}")

    total_comm_count = 0
    total_comm_bytes = 0
    for ctype in ['all_reduce', 'all_gather', 'reduce_scatter', 'all_to_all', 'broadcast']:
        cs = rep['collectives'][ctype]
        if cs['count'] == 0:
            continue
        total_comm_count += cs['count']
        total_comm_bytes += cs['bytes_total']
        print(f"  {ctype:<18} {cs['count']:>7} {cs['per_step']:>10.1f} "
              f"{_fmt_bytes(cs['bytes_total']):>14} {_fmt_bytes(int(cs['bytes_per_step'])):>14} "
              f"{_fmt_bytes(cs['bytes_median']):>14}")

    if total_comm_count == 0:
        print(f"  (no collective communication detected)")

    # --- Process Group Breakdown ---
    if rep['pg_bytes']:
        print(f"\n  Process Group Breakdown:")
        for pg, b in sorted(rep['pg_bytes'].items()):
            print(f"    {pg}: {_fmt_bytes(b)}")

    # --- Summary ---
    n_steps = rep['steps_n'] if rep['steps_n'] > 0 else 1
    print(f"\n  Summary:")
    print(f"    Total COMM ops/step:  {total_comm_count / n_steps:.1f}")
    print(f"    Total COMM bytes/step: {_fmt_bytes(int(total_comm_bytes / n_steps))}")
    if rep['kernel_per_step_ms'] and rep['kernel_per_step_ms'] > 0:
        comm_per_step_ms = sum(
            rep['collectives'][ct]['total_ms'] for ct in rep['collectives']
        ) / n_steps
        ratio = comm_per_step_ms / rep['kernel_per_step_ms'] if rep['kernel_per_step_ms'] > 0 else 0
        print(f"    COMM/Compute ratio:    {ratio:.2%}")
        print(f"    COMM time/step:        {_fmt_ms(comm_per_step_ms)}")
        print(f"    Compute time/step:     {_fmt_ms(rep['kernel_per_step_ms'])}")

    # --- Cross-rank consistency ---
    if len(ranks) > 1:
        print(f"\n  Cross-Rank Consistency:")
        for r_name, r_data in results.items():
            comm_total = sum(r_data['collectives'][ct]['count'] for ct in r_data['collectives'])
            print(f"    {r_name}: {r_data['steps_n']} steps, "
                  f"{comm_total} COMM ops, "
                  f"mode={r_data['mode']}")

    # --- Health Check ---
    print(f"\n  Health Check:")
    issues = []

    if rep['steps_n'] == 0:
        issues.append("❌ No ProfilerStep detected — profiler may not have started")
    if rep['kernel_count'] == 0:
        issues.append("❌ No GPU kernels detected — device trace may be empty")
    if total_comm_count == 0:
        issues.append("❌ No collective communication detected — model may not be distributed")

    if rep['mode'] == 'DDP':
        ar_per_step = rep['collectives']['all_reduce']['per_step']
        if ar_per_step < 1:
            issues.append(f"⚠️ Only {ar_per_step:.1f} AllReduce/step — expected ≥1 for DDP")
        if ar_per_step > 0:
            ar_bytes_med = rep['collectives']['all_reduce']['bytes_median']
            if ar_bytes_med == 0:
                issues.append("⚠️ AllReduce bytes=0 — tagging hook may not be working")

    elif 'TP' in rep['mode']:
        ar_per_step = rep['collectives']['all_reduce']['per_step']
        if ar_per_step < 20:
            issues.append(f"⚠️ Only {ar_per_step:.1f} AllReduce/step — expected ≥50 for TP")

    if not issues:
        print("    ✅ All checks passed")
    else:
        for iss in issues:
            print(f"    {iss}")

    # --- Converter Readiness ---
    print(f"\n  Converter Readiness:")
    if total_comm_count > 0 and rep['steps_n'] > 0 and rep['kernel_count'] > 0:
        print(f"    ✅ Ready for: conver_to_chakra_et.py --model-tag {tag}")
    else:
        print(f"    ❌ Not ready — fix issues above first")

    print()


# =====================================================================
# Report: Comparison
# =====================================================================

def print_comparison_report(all_results: Dict[str, Dict[str, Dict]]):
    """Print side-by-side comparison of multiple model tags."""
    tags = sorted(all_results.keys())
    if len(tags) < 2:
        print("[info] Need at least 2 model tags for comparison")
        return

    print(f"\n{'='*70}")
    print(f"  Comparison Report: {' vs '.join(tags)}")
    print(f"{'='*70}")

    # Header
    col_w = 20
    header = f"  {'Metric':<30}"
    for tag in tags:
        header += f" {tag:>{col_w}}"
    print(header)
    print(f"  {'-'*30}" + f" {'-'*col_w}" * len(tags))

    # Helper to get representative data (rank 0)
    def rep(tag):
        ranks = sorted(all_results[tag].keys())
        return all_results[tag][ranks[0]] if ranks else None

    reps = {t: rep(t) for t in tags}

    # Rows
    def row(label, getter, fmt=str):
        line = f"  {label:<30}"
        for t in tags:
            r = reps[t]
            val = getter(r) if r else "N/A"
            line += f" {fmt(val):>{col_w}}"
        print(line)

    row("Detected Mode", lambda r: r['mode'])
    row("Steps", lambda r: r['steps_n'])
    row("Step median", lambda r: r['step_median_ms'], lambda v: _fmt_ms(v) if v else "N/A")
    print()

    row("AllReduce count", lambda r: r['collectives']['all_reduce']['count'])
    row("AllReduce/step", lambda r: r['collectives']['all_reduce']['per_step'], lambda v: f"{v:.1f}")
    row("AllReduce bytes/step",
        lambda r: r['collectives']['all_reduce']['bytes_per_step'],
        lambda v: _fmt_bytes(int(v)))
    row("AllReduce median size",
        lambda r: r['collectives']['all_reduce']['bytes_median'],
        lambda v: _fmt_bytes(v))
    print()

    row("AllGather count", lambda r: r['collectives']['all_gather']['count'])
    row("ReduceScatter count", lambda r: r['collectives']['reduce_scatter']['count'])
    print()

    row("GPU kernels", lambda r: r['kernel_count'])
    row("Kernel time/step", lambda r: r['kernel_per_step_ms'], lambda v: _fmt_ms(v) if v else "N/A")

    # Total COMM
    def total_comm(r):
        return sum(r['collectives'][ct]['count'] for ct in r['collectives'])

    def total_comm_bytes_per_step(r):
        n = r['steps_n'] if r['steps_n'] > 0 else 1
        return sum(r['collectives'][ct]['bytes_total'] for ct in r['collectives']) / n

    print()
    row("Total COMM ops", total_comm)
    row("Total COMM ops/step", lambda r: total_comm(r) / max(r['steps_n'], 1), lambda v: f"{v:.1f}")
    row("Total COMM bytes/step", total_comm_bytes_per_step, lambda v: _fmt_bytes(int(v)))

    # COMM/Compute ratio
    def comm_compute_ratio(r):
        n = max(r['steps_n'], 1)
        comm_ms = sum(r['collectives'][ct]['total_ms'] for ct in r['collectives']) / n
        comp_ms = r['kernel_per_step_ms']
        if comp_ms and comp_ms > 0:
            return comm_ms / comp_ms
        return 0

    print()
    row("COMM/Compute ratio", comm_compute_ratio, lambda v: f"{v:.2%}" if v else "N/A")

    # Interpretation
    print(f"\n  Interpretation:")
    for t in tags:
        r = reps[t]
        if r is None:
            continue
        ratio = comm_compute_ratio(r)
        ops_per_step = total_comm(r) / max(r['steps_n'], 1)
        if ratio > 0.5:
            verdict = "COMM-heavy → topology matters"
        elif ratio > 0.1:
            verdict = "Moderate COMM → topology may matter"
        else:
            verdict = "COMP-heavy → topology impact minimal"
        print(f"    {t}: {ops_per_step:.0f} COMM ops/step, ratio={ratio:.2%} → {verdict}")

    print()


# =====================================================================
# Smoke Test (streaming, memory-efficient)
# =====================================================================

def smoke_test_rank(traces_dir: Path, rank: int, tag: str = None) -> Tuple[bool, Dict]:
    """Run basic smoke test on a single rank's traces."""
    suffix = f"_{tag}" if tag else ""
    host = traces_dir / f"host_{rank}{suffix}.json"
    dev = traces_dir / f"device_{rank}{suffix}.json"

    result = {'rank': rank, 'host_ok': False, 'device_ok': False, 'issues': []}

    # Host check
    if not host.exists():
        result['issues'].append(f"host file missing: {host.name}")
    else:
        good, why = _try_parse_json(host)
        if good:
            result['host_ok'] = True
        else:
            result['issues'].append(f"host JSON parse failed: {why}")
            try:
                txt = host.read_text(encoding="utf-8", errors="ignore")
                if _looks_like_concatenated_json(txt):
                    result['issues'].append("Likely concatenated JSON — run _repair_host_json()")
            except Exception:
                pass

    # Device check
    if not dev.exists():
        result['issues'].append(f"device file missing: {dev.name}")
    else:
        cnt = _stream_count(dev, BYTE_PATTERNS)
        result['device_counts'] = cnt

        if cnt['profstep'] == 0:
            result['issues'].append("No ProfilerStep# found")
        if cnt['kernel_like'] == 0:
            result['issues'].append("No GPU kernel events found")

        total_sync = cnt['hip_rec'] + cnt['cuda_rec'] + cnt['hip_wait'] + cnt['cuda_wait']
        if total_sync == 0:
            result['issues'].append("No sync events (may affect trace linking quality)")

        total_comm = cnt['all_reduce'] + cnt['all_gather'] + cnt['reduce_scatter']
        if total_comm == 0 and cnt['nccl_generic'] == 0:
            result['issues'].append("No collective communication detected")

        if cnt['profstep'] > 0 and cnt['kernel_like'] > 0:
            result['device_ok'] = True

    return result['host_ok'] and result['device_ok'], result


# =====================================================================
# Discover
# =====================================================================

def discover_ranks(traces_dir: Path, ranks_arg: str, tag: str = None) -> List[int]:
    if ranks_arg != "auto":
        return [int(x) for x in ranks_arg.split(",") if x.strip() != ""]
    ranks = set()
    suffix = f"_{tag}" if tag else ""
    for p in traces_dir.glob(f"host_*{suffix}.json"):
        parts = p.stem.split('_')
        if len(parts) >= 2 and parts[1].isdigit():
            ranks.add(int(parts[1]))
    for p in traces_dir.glob(f"device_*{suffix}.json"):
        parts = p.stem.split('_')
        if len(parts) >= 2 and parts[1].isdigit():
            ranks.add(int(parts[1]))
    return sorted(ranks)


def discover_tags(traces_dir: Path) -> List[str]:
    """Discover all model tags from device trace filenames."""
    tags = set()
    for p in traces_dir.glob("device_*.json"):
        parts = p.stem.split('_')  # device_0_qwen05b -> ['device', '0', 'qwen05b']
        if len(parts) >= 3:
            tags.add('_'.join(parts[2:]))
    return sorted(tags)


# =====================================================================
# Main
# =====================================================================

def main():
    ap = argparse.ArgumentParser(
        description="DDP / TP Trace 驗證與分析工具",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Check specific model trace
  python check_trace_ready.py --model-tag qwen15b_tp

  # Compare DDP vs TP traces
  python check_trace_ready.py --compare qwen05b qwen15b_tp

  # Check all available traces
  python check_trace_ready.py --all

  # Quick smoke test only (no full JSON parse)
  python check_trace_ready.py --model-tag qwen15b_tp --smoke-only
        """
    )
    ap.add_argument("--traces-dir", type=str,
                    default=str(Path(__file__).resolve().parents[2] / "data" / "chakra" / "pytorch_traces"),
                    help="Trace 目錄路徑")
    ap.add_argument("--ranks", type=str, default="auto",
                    help="指定 ranks (e.g. '0,1') 或 'auto'")
    ap.add_argument("--model-tag", type=str, default=None,
                    help="單一模型分析 (e.g. qwen05b, qwen15b_tp, resnet50)")
    ap.add_argument("--compare", nargs='+', metavar='TAG',
                    help="對比多個模型的 trace (e.g. --compare qwen05b qwen15b_tp)")
    ap.add_argument("--all", action="store_true",
                    help="分析目錄中所有可偵測到的 model tags")
    ap.add_argument("--smoke-only", action="store_true",
                    help="只做快速 smoke test (不做完整 JSON 解析)")

    args = ap.parse_args()
    traces_dir = Path(args.traces_dir)

    if not traces_dir.is_dir():
        print(f"[error] Directory not found: {traces_dir}")
        sys.exit(1)

    print(f"[info] Trace directory: {traces_dir}")

    # Determine which tags to analyze
    if args.compare:
        tags_to_analyze = args.compare
    elif args.all:
        tags_to_analyze = discover_tags(traces_dir)
        if not tags_to_analyze:
            print("[error] No model tags found in trace directory")
            sys.exit(1)
        print(f"[info] Discovered tags: {tags_to_analyze}")
    elif args.model_tag:
        tags_to_analyze = [args.model_tag]
    else:
        # Default: discover all tags
        tags_to_analyze = discover_tags(traces_dir)
        if not tags_to_analyze:
            # Fallback: try without tag
            tags_to_analyze = [None]
        print(f"[info] Discovered tags: {[t or '(no tag)' for t in tags_to_analyze]}")

    # Process each tag
    all_results = {}  # tag -> {rank_file -> parsed_data}
    ok_all = True

    for tag in tags_to_analyze:
        tag_label = tag or "(no tag)"
        ranks = discover_ranks(traces_dir, args.ranks, tag=tag)
        if not ranks:
            print(f"\n[warn] No traces found for tag '{tag_label}'")
            continue

        print(f"\n{'─'*70}")
        print(f"  Checking: {tag_label} (ranks: {ranks})")
        print(f"{'─'*70}")

        # Step 1: Smoke test
        for r in ranks:
            passed, smoke_result = smoke_test_rank(traces_dir, r, tag)
            status = "✅ PASS" if passed else "❌ FAIL"
            print(f"  [Rank {r}] {status}")

            if 'device_counts' in smoke_result:
                cnt = smoke_result['device_counts']
                print(f"    ProfilerStep: {cnt['profstep']}, "
                      f"Kernels: {cnt['kernel_like']}, "
                      f"AllReduce: {cnt['all_reduce']}, "
                      f"AllGather: {cnt['all_gather']}, "
                      f"ReduceScatter: {cnt['reduce_scatter']}, "
                      f"NCCL Generic: {cnt['nccl_generic']}")

            for iss in smoke_result.get('issues', []):
                print(f"    ⚠️  {iss}")

            if not passed:
                ok_all = False

        # Step 2: Detailed analysis (unless --smoke-only)
        if not args.smoke_only:
            tag_results = {}
            for r in ranks:
                suffix = f"_{tag}" if tag else ""
                dev_path = traces_dir / f"device_{r}{suffix}.json"
                if dev_path.exists():
                    try:
                        tag_results[dev_path.name] = parse_trace_detailed(dev_path)
                    except Exception as e:
                        print(f"  [warn] Failed to parse {dev_path.name}: {e}")

            if tag_results:
                all_results[tag_label] = tag_results
                print_single_report(tag_label, tag_results)

    # Step 3: Comparison (if multiple tags)
    if len(all_results) >= 2 and not args.smoke_only:
        print_comparison_report(all_results)

    # Final verdict
    print(f"\n{'='*70}")
    if ok_all:
        print("  [READY] All traces passed smoke test.")
        if tags_to_analyze and tags_to_analyze[0]:
            print(f"  Next: python src/conver_to_chakra_et.py --model-tag {tags_to_analyze[0]}")
    else:
        print("  [NOT READY] Some traces have issues — see details above.")
    print(f"{'='*70}\n")

    sys.exit(0 if ok_all else 1)


if __name__ == "__main__":
    main()
