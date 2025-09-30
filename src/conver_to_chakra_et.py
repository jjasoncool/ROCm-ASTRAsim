#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
PyTorch Chrome Trace to Chakra protobuf ET Converter

This module converts PyTorch profiler traces to Chakra execution traces for
simulation-based performance analysis and workload modeling.

Core Features:
- Multi-epoch trace processing with automatic unit detection
- GPU compute and communication pattern extraction
- Configurable performance parameter calibration
- High-performance multi-core processing
- Chakra protobuf format compliance

Input:  ../data/chakra/pytorch_traces/trace_rank_{rank}_epoch_{epoch}.json
Output: ../data/chakra/workload_et/allreduce.{rank}.et

Alpha calibration sources (priority order):
1. CLI specification (--alpha-us)
2. GPU metrics from training
3. Calibration database (runs/calibration_all.csv)
4. GPU core frequency estimation
5. Bootstrap mode

Requirements:
- chakra.schema.protobuf.et_def_pb2
- chakra.src.third_party.utils.protolib
"""

from __future__ import annotations
import os
import re
import csv
import argparse
import bisect
import statistics
import subprocess
import logging
from pathlib import Path
from typing import List, Dict, Tuple, Optional
from concurrent.futures import ProcessPoolExecutor, as_completed

# Logging configuration (--debug enables DEBUG level)
logging.basicConfig(level=logging.WARNING, format="%(levelname)s: %(message)s")
logger = logging.getLogger("et_convert")

# JSON backend: prefer orjson, fallback to standard json
try:
    import orjson as _json
    def _load_json_bytes(b: bytes): return _json.loads(b)
    _JSON_BACKEND = "orjson"
except Exception:
    import json as _json
    def _load_json_bytes(b: bytes): return _json.loads(b.decode("utf-8"))
    _JSON_BACKEND = "json"

# ijson for streaming (optional) - this version focuses on batch processing
try:
    import ijson
    _HAS_IJSON = True
except Exception:
    _HAS_IJSON = False

# ROCm SMI parsing utilities
import json as json_std

# Memory detection (optional)
try:
    import psutil
    _HAS_PSUTIL = True
except Exception:
    _HAS_PSUTIL = False

# Chakra protobuf dependencies
from chakra.src.third_party.utils.protolib import encodeMessage as encode_message
from chakra.schema.protobuf.et_def_pb2 import (
    Node as ChakraNode,
    BoolList,
    GlobalMetadata,
    AttributeProto as ChakraAttr,
    COMM_COLL_NODE,
    ALL_REDUCE,
)

# Compute enum (different names in different versions)
try:
    from chakra.schema.protobuf.et_def_pb2 import COMP_NODE as COMPUTE_ENUM
except Exception:
    try:
        from chakra.schema.protobuf.et_def_pb2 import COMPUTE_NODE as COMPUTE_ENUM
    except Exception:
        COMPUTE_ENUM = None

# Configuration constants
STEP_PATTERN = re.compile(r"^ProfilerStep#\d+$")
RANK_EPOCH_PATTERN = re.compile(r"rank_(\d+)_epoch_(\d+)\.json$", re.IGNORECASE)

# GPU event detection tokens
GPU_CATEGORIES = ("kernel", "gpu", "cuda", "hip", "hcc", "gfx", "memcpy", "memset", "driver", "runtime", "hsa")
GPU_ARG_KEYS = ("Device", "device", "Stream", "stream", "queue", "Queue Id", "Correlation ID")

# Communication detection tokens
COMM_TOKENS = ("nccl", "rccl", "allreduce", "all_reduce", "allgather", "all_gather",
               "reducescatter", "reduce_scatter", "alltoall", "broadcast")

# Utility functions
def rank_epoch_from_name(p: Path) -> Tuple[int, int]:
    """Extract rank and epoch from filename."""
    m = RANK_EPOCH_PATTERN.search(p.name)
    if not m:
        raise ValueError(f"Cannot parse rank/epoch from filename: {p.name}")
    return int(m.group(1)), int(m.group(2))

def get_world_size(trace: Dict) -> int:
    """Extract world size from trace metadata."""
    di = trace.get("distributedInfo", {}) or {}
    return int(di.get("world_size", 1))

def to_ns(us: float) -> int:
    """Convert microseconds to nanoseconds."""
    return int(round(us * 1000.0))

def _infer_us_scale_from_steps(trace: Dict) -> float:
    """Infer time unit scale factor by analyzing ProfilerStep durations.

    Automatically detects whether timestamps are in nanoseconds, microseconds,
    or milliseconds based on median duration values.

    Args:
        trace: Chrome trace data dictionary

    Returns:
        Scale factor to convert to microseconds:
        - 1e-3 for nanoseconds
        - 1.0 for microseconds
        - 1e3 for milliseconds
    """
    raw = []
    events = trace.get("traceEvents", []) or []
    for e in events:
        if not isinstance(e, dict): continue
        if e.get("ph") != "X": continue
        nm = e.get("name", "")
        if isinstance(nm, str) and STEP_PATTERN.match(nm):
            d = e.get("dur", None)
            if d is not None:
                try: raw.append(float(d))
                except Exception: pass
            if len(raw) >= 64:  # 取樣 64 筆即可
                break
    if not raw:
        # 沒找到 step，就用所有事件 dur 略估
        for e in events[:2048]:
            if not isinstance(e, dict): continue
            if e.get("ph") != "X": continue
            d = e.get("dur", None)
            if d is not None:
                try: raw.append(float(d))
                except Exception: pass
    if not raw:
        # 保守視為 μs
        return 1.0
    med = statistics.median(raw)
    if med >= 1e7:
        return 1e-3  # ns → μs
    elif med >= 1e3:
        return 1.0  # 已是 μs
    else:
        return 1e3   # ms → μs

def load_trace(path: Path) -> Dict:
    """Load and parse Chrome trace JSON file.

    Args:
        path: Path to trace file

    Returns:
        Parsed trace data dictionary

    Raises:
        RuntimeError: If file loading or JSON parsing fails
    """
    try:
        data = path.read_bytes()
        return _load_json_bytes(data)
    except Exception as ex:
        raise RuntimeError(f"Failed to load/parse JSON: {path}") from ex

def to_ns(us: float) -> int:
    """Convert microseconds to nanoseconds."""
    return int(round(us * 1000.0))

# Event classification functions
def _is_gpu_event(ev: dict) -> bool:
    """Check if event is GPU-related based on category and arguments."""
    cat = str(ev.get("cat", "")).lower()
    if any(tok in cat for tok in GPU_CATEGORIES):
        return True
    args = ev.get("args", {}) or {}
    return any(k in args for k in GPU_ARG_KEYS)

def _is_comm_event(ev: dict) -> bool:
    """Check if event is communication-related."""
    s = (str(ev.get("name", "")) + " " + str(ev.get("cat", ""))).lower()
    return any(tok in s for tok in COMM_TOKENS)

# Interval processing utilities
def _overlap(a0: float, a1: float, b0: float, b1: float) -> float:
    """Calculate overlap between two intervals."""
    return max(0.0, min(a1, b1) - max(a0, b0))

def _merge_intervals(intervals: List[Tuple[float, float]]) -> List[Tuple[float,float]]:
    """Merge overlapping intervals into union."""
    if not intervals:
        return []
    intervals.sort()
    merged = [intervals[0]]
    for s, e in intervals[1:]:
        ps, pe = merged[-1]
        if s <= pe:
            merged[-1] = (ps, max(pe, e))
        else:
            merged.append((s, e))
    return merged

def _intervals_length(intervals: List[Tuple[float,float]]) -> float:
    """Calculate total length of interval list."""
    return sum(e - s for s, e in intervals)

def _subtract_intervals(a: List[Tuple[float,float]], b: List[Tuple[float,float]]) -> List[Tuple[float,float]]:
    """Return A \\ B interval subtraction (assumes sorted inputs)."""
    if not a or not b:
        return a[:]
    out = []
    i = j = 0
    while i < len(a):
        s, e = a[i]
        while j < len(b) and b[j][1] <= s:
            j += 1
        cur_s = s
        while j < len(b) and b[j][0] < e:
            bs, be = b[j]
            if bs > cur_s:
                out.append((cur_s, bs))
            cur_s = max(cur_s, be)
            j += 1
            if cur_s >= e:
                break
        if cur_s < e:
            out.append((cur_s, e))
        i += 1
    return out

def _extract_steps_us(trace: Dict, scale_us: float) -> List[Tuple[float, float]]:
    """Extract ProfilerStep boundaries converted to microseconds."""
    steps: List[Tuple[float, float]] = []
    events = trace.get("traceEvents", [])
    if not isinstance(events, list):
        return steps
    for e in events:
        if not isinstance(e, dict):
            continue
        if e.get("ph") != "X":
            continue
        name = e.get("name")
        if not isinstance(name, str) or not STEP_PATTERN.match(name):
            continue
        try:
            ts = float(e.get("ts", 0.0)) * scale_us
            dur = float(e.get("dur", 0.0)) * scale_us
        except Exception:
            continue
        if dur > 0:
            steps.append((ts, ts + dur))
    steps.sort()
    return steps

def collect_step_aggregates_union_gpu_only(trace: Dict, steps: List[Tuple[float, float]], scale_us: float
                                           ) -> Tuple[List[Tuple[float, float, float, float]], dict]:
    """Extract GPU compute and communication metrics from trace events.

    Processes only GPU events to avoid CPU overhead contamination.
    Uses interval union operations to eliminate overlapping duration counts.

    Args:
        trace: Chrome trace data
        steps: List of (start_us, end_us) step boundaries
        scale_us: Time unit conversion factor

    Returns:
        Tuple of:
        - List of (start_us, end_us, compute_us, exposed_comm_us) per step
        - Statistics dictionary with event counts
    """
    if not steps:
        return [], {"gpu_event_cnt": 0, "comm_event_cnt": 0}

    starts = [s for s, _ in steps]
    ends   = [e for _, e in steps]

    comp_intervals: List[List[Tuple[float,float]]] = [[] for _ in steps]
    comm_intervals: List[List[Tuple[float,float]]] = [[] for _ in steps]

    gpu_cnt = 0
    comm_cnt = 0

    events = trace.get("traceEvents", [])
    if not isinstance(events, list):
        return [(s, e, 0.0, 0.0) for s, e in steps], {"gpu_event_cnt": 0, "comm_event_cnt": 0}

    for ev in events:
        if not isinstance(ev, dict): continue
        if ev.get("ph") != "X": continue

        if not _is_gpu_event(ev):
            continue
        gpu_cnt += 1

        try:
            ev_s = float(ev.get("ts", 0.0)) * scale_us
            ev_e = ev_s + float(ev.get("dur", 0.0)) * scale_us
        except Exception:
            continue
        if ev_e <= ev_s:
            continue

        # 定位第一個可能相交步
        i = bisect.bisect_right(ends, ev_s)
        i = max(0, i - 1)

        is_comm = _is_comm_event(ev)
        if is_comm:
            comm_cnt += 1

        while i < len(steps) and starts[i] < ev_e:
            inter_s = max(starts[i], ev_s)
            inter_e = min(ends[i], ev_e)
            if inter_e > inter_s:
                if is_comm:
                    comm_intervals[i].append((inter_s, inter_e))
                else:
                    comp_intervals[i].append((inter_s, inter_e))
            i += 1

    out: List[Tuple[float, float, float, float]] = []
    for k, (s, e) in enumerate(steps):
        comp_m = _merge_intervals(comp_intervals[k])
        comm_m = _merge_intervals(comm_intervals[k])
        comp_len = _intervals_length(comp_m)
        # 暴露通訊 = comm_union 減去 compute_union 的覆蓋
        comm_exposed = _intervals_length(_subtract_intervals(comm_m, comp_m))
        out.append((s, e, comp_len, comm_exposed))

    info = {"gpu_event_cnt": gpu_cnt, "comm_event_cnt": comm_cnt}
    return out, info

# ----------------------------- α（引導/讀取） -----------------------------

def _read_gpu_metrics_for_alpha(gpu_metrics_dir: Path, rank: int, epoch: int) -> Optional[float]:
    """
    從對應的 GPU 指標檔案讀取建議的 alpha 值

    Args:
        gpu_metrics_dir: GPU 指標目錄
        rank: rank ID
        epoch: epoch ID

    Returns:
        alpha_us: 建議的 alpha 值（μs/週期），若找不到則回傳 None
    """
    if not gpu_metrics_dir.exists():
        return None

    # 尋找對應的 GPU 指標檔案
    metrics_file = gpu_metrics_dir / f"gpu_metrics_rank_{rank}_epoch_{epoch}.json"
    if not metrics_file.exists():
        return None

    try:
        with metrics_file.open("r", encoding="utf-8") as f:
            data = json_std.load(f)

        # 提取建議的 alpha 值
        alpha_calcs = data.get("alpha_calculations", {})
        if "alpha_us_recommended" in alpha_calcs:
            return float(alpha_calcs["alpha_us_recommended"])

        # 如果沒有建議值，嘗試其他值
        for key in ["alpha_us_eff_0.3", "alpha_us_eff_0.2", "alpha_us_eff_0.5"]:
            if key in alpha_calcs:
                return float(alpha_calcs[key])

        return None
    except Exception as e:
        logger.warning(f"Failed to read GPU metrics {metrics_file}: {e}")
        return None

def _collect_alpha_from_gpu_metrics(gpu_metrics_dir: Path, results_by_rank: Dict) -> Tuple[Optional[float], dict]:
    """
    從所有可用的 GPU 指標檔案收集 alpha 值並統計

    Returns:
        (best_alpha, stats_info)
    """
    if not gpu_metrics_dir or not gpu_metrics_dir.exists():
        return None, {"reason": "GPU metrics directory not found"}

    alpha_values = []
    collected_info = []
    missing_files = []
    total_attempts = 0

    for rank, epochs_aggs in results_by_rank.items():
        for epoch, _ in epochs_aggs:
            total_attempts += 1
            alpha = _read_gpu_metrics_for_alpha(gpu_metrics_dir, rank, epoch)
            if alpha and alpha > 0:
                alpha_values.append(alpha)
                collected_info.append({
                    "rank": rank,
                    "epoch": epoch,
                    "alpha_us": alpha
                })
            else:
                # 記錄缺失的檔案
                metrics_file = gpu_metrics_dir / f"gpu_metrics_rank_{rank}_epoch_{epoch}.json"
                if not metrics_file.exists():
                    missing_files.append(f"rank_{rank}_epoch_{epoch}")

    # 提供詳細統計信息
    stats = {
        "source": "GPU metrics from training",
        "collected_samples": len(alpha_values),
        "total_attempts": total_attempts,
        "missing_files_count": len(missing_files),
        "details": collected_info
    }

    # 如果有缺失檔案，記錄警告
    if missing_files:
        logger.warning(f"Missing {len(missing_files)} GPU metrics files: {missing_files[:5]}{'...' if len(missing_files) > 5 else ''}")
        stats["missing_files"] = missing_files

    if alpha_values:
        best_alpha = statistics.median(alpha_values)
        stats.update({
            "chosen_alpha": best_alpha,
            "alpha_range": (min(alpha_values), max(alpha_values)),
            "reason": f"median of {len(alpha_values)} GPU-measured alpha values"
        })
        return best_alpha, stats
    else:
        stats["reason"] = "no GPU metrics found with valid alpha"
        return None, stats

def _read_alpha_us_from_db(db_path: Path, quality_threshold: float = 5.0) -> Tuple[Optional[float], dict]:
    """Load performance parameter alpha from calibration database.

    Applies quality filtering to select reliable calibration results:
    - Filters out entries with high relative communication error
    - Excludes entries where communication equals wall-clock time

    Args:
        db_path: Path to calibration CSV file
        quality_threshold: Maximum acceptable relative error percentage

    Returns:
        Tuple of (best_alpha_value, statistics_dict)
    """
    if not db_path.exists():
        return None, {"reason": "DB file not found"}

    all_vals: List[float] = []
    good_vals: List[float] = []
    filtered_out = {"high_rel_err": 0, "comm_equals_wall": 0, "invalid_data": 0}

    with db_path.open("r", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            try:
                alpha = float(row.get("alpha_us", ""))
                if alpha <= 0:
                    continue
                all_vals.append(alpha)

                # 檢查相對誤差
                rel_err_comm = float(row.get("rel_err_comm", "0"))
                flags = row.get("flags", "")

                # 過濾條件
                if rel_err_comm > quality_threshold:
                    filtered_out["high_rel_err"] += 1
                    continue

                if "comm_equals_wall" in flags:  # 通訊等於牆鐘時間（compute ≈ 0）
                    filtered_out["comm_equals_wall"] += 1
                    continue

                good_vals.append(alpha)

            except Exception:
                filtered_out["invalid_data"] += 1
                continue

    stats = {
        "total_records": len(all_vals),
        "good_records": len(good_vals),
        "filtered_out": filtered_out,
        "all_alpha_range": (min(all_vals), max(all_vals)) if all_vals else None,
        "good_alpha_range": (min(good_vals), max(good_vals)) if good_vals else None,
        "quality_threshold": quality_threshold
    }

    if good_vals:
        best_alpha = statistics.median(good_vals)
        stats["chosen_alpha"] = best_alpha
        stats["reason"] = f"median of {len(good_vals)} good calibrations"
        return best_alpha, stats
    elif all_vals:
        # 如果沒有「好」的校準，但有數據，回傳最小值（通常較保守）
        fallback_alpha = min(all_vals)
        stats["chosen_alpha"] = fallback_alpha
        stats["reason"] = f"fallback: min of {len(all_vals)} records (all had quality issues)"
        return fallback_alpha, stats
    else:
        stats["reason"] = "no valid alpha_us data found"
        return None, stats

def _alpha_from_core_frequency(mhz: float, scale: float) -> Optional[float]:
    """Calculate alpha from core frequency: alpha_us = scale / MHz (temporary guidance only)."""
    try:
        mhz = float(mhz)
        return (scale / mhz) if mhz > 0 else None
    except Exception:
        return None

def _query_rocm_smi_core_mhz() -> Optional[float]:
    """Query GPU core frequency using ROCm SMI tool.

    Attempts to retrieve shader clock frequency from the first GPU device.
    Handles various ROCm SMI output formats and provides fallback clock sources.

    Returns:
        Core frequency in MHz, or None if query fails or GPU is idle
    """
    try:
        p = subprocess.run(["rocm-smi", "--showclocks", "--json"],
                           check=True, capture_output=True, text=True)
        data = json_std.loads(p.stdout)
        first = next(iter(data.values()))

        def extract_mhz_from_string(text: str) -> Optional[float]:
            """Extract MHz value from string format like '(243Mhz)'."""
            if not isinstance(text, str):
                return None
            match = re.search(r'(\d+(?:\.\d+)?)', text.replace('(', '').replace(')', '').lower().replace('mhz', ''))
            if match:
                try:
                    return float(match.group(1))
                except:
                    return None
            return None

        def parse_clock_value(value) -> Optional[float]:
            """Parse clock value supporting multiple formats."""
            if isinstance(value, (int, float)):
                return float(value) if value > 0 else None
            elif isinstance(value, str):
                mhz = extract_mhz_from_string(value)
                return mhz if mhz and mhz > 0 else None
            elif isinstance(value, dict):
                for key in ("current", "avg", "max", "min"):
                    v = value.get(key)
                    if v is not None:
                        parsed = parse_clock_value(v)
                        if parsed and parsed > 0:
                            return parsed
            return None

        # Priority clock sources (core/shader clock preferred)
        priority_keys = [
            "sclk clock speed:",
            "sclk",
            "gfx_clock",
            "GFX Clock (MHz)",
            "Current Graphics Clock (MHz)"
        ]

        for key in priority_keys:
            if key in first:
                mhz = parse_clock_value(first[key])
                if mhz and mhz > 0:
                    logger.debug(f"ROCm SMI: Found core frequency {mhz} MHz from '{key}'")
                    return mhz

        # Handle idle/power-saving mode
        logger.warning("ROCm SMI: Core clock is 0 or unavailable, GPU may be idle")
        logger.debug(f"ROCm SMI available clocks: {list(first.keys())}")

        # Fallback clocks (not suitable for alpha calculation)
        fallback_keys = ["socclk clock speed:", "dcefclk clock speed:", "fclk clock speed:"]
        for key in fallback_keys:
            if key in first:
                mhz = parse_clock_value(first[key])
                if mhz and mhz > 0:
                    logger.warning(f"ROCm SMI: Core clock unavailable, found {key}={mhz}MHz (reference only)")

        return None
    except subprocess.CalledProcessError as e:
        logger.debug(f"ROCm SMI execution failed: {e}")
        return None
    except Exception as e:
        logger.debug(f"ROCm SMI parsing error: {e}")
        return None

# ----------------------------- 多進程 worker -----------------------------

def _worker_parse_one(args_tuple) -> Tuple[int, int, int, float, List[Tuple[float, float, float, float]], dict, Optional[str]]:
    """Worker process function for parallel trace file processing.

    Parses a single trace file and extracts performance metrics.
    Designed for use with ProcessPoolExecutor for scalable processing.

    Args:
        args_tuple: (path_str, max_steps, keep_every, rebase_ts)

    Returns:
        Tuple of (rank, epoch, world_size, scale_factor, aggregates, info, error_msg)
    """
    (path_str, max_steps, keep_every, rebase_ts) = args_tuple
    p = Path(path_str)
    try:
        r, e = rank_epoch_from_name(p)
    except Exception:
        return 0, 0, 0, 1.0, [], {}, f"檔名解析失敗: {p.name}"
    try:
        trace = load_trace(p)
        world = get_world_size(trace)
        scale_us = _infer_us_scale_from_steps(trace)
        steps = _extract_steps_us(trace, scale_us)
        if not steps:
            return r, e, world, scale_us, [], {}, f"{p.name} 無 ProfilerStep 事件"
        # 聚合（GPU-only + de-overlap + exposed_comm）
        agg, info = collect_step_aggregates_union_gpu_only(trace, steps, scale_us)

        # 篩步數與抽樣
        if max_steps is not None and max_steps > 0:
            agg = agg[:max_steps]
        if keep_every > 1:
            agg = agg[::keep_every]

        # rebase_ts：讓本檔內 ts 以第一步起點為 0（跨檔銜接在主進程處理）
        if rebase_ts and agg:
            base = agg[0][0]
            agg = [(s - base, e - base, cu, mu) for (s, e, cu, mu) in agg]

        return r, e, world, scale_us, agg, info, None
    except Exception as ex:
        return r, e, 0, 1.0, [], {}, f"讀/解/聚合失敗: {p.name} ({ex})"

# ----------------------------- 計算 bytes -----------------------------

def per_gpu_allreduce_bytes(world_size: int, model_param_bytes: int) -> int:
    """Calculate AllReduce bytes per GPU for ring algorithm.

    Uses ring topology formula: 2*(N-1)/N * model_bytes
    For N=2, this approximates to model_bytes per GPU.

    Args:
        world_size: Number of GPUs in distributed training
        model_param_bytes: Total model parameter size in bytes

    Returns:
        Bytes transferred per GPU per AllReduce operation
    """
    return int(2 * (world_size - 1) / world_size * model_param_bytes)

# ----------------------------- jobs/記憶體 -----------------------------

def _estimate_memory_jobs(args, files: List[Path]) -> int:
    """Estimate optimal number of parallel jobs based on memory constraints."""
    jobs = max(1, int(args.jobs))
    sample = files[:8]
    sizes_mb = [max(1, p.stat().st_size // (1024*1024)) for p in sample] or [64]
    median_mb = statistics.median(sizes_mb)
    est_worker_mb = max(64, int(median_mb * args.mem_overhead_factor))

    avail_mb = None
    if args.max_memory_mb:
        avail_mb = max(256, int(args.max_memory_mb))
    elif _HAS_PSUTIL:
        try:
            avail_mb = max(256, int(psutil.virtual_memory().available // (1024*1024)))
        except Exception:
            pass

    if avail_mb:
        cap = max(1, int(avail_mb // est_worker_mb))
        jobs = min(jobs, cap)
    else:
        cpu = os.cpu_count() or 2
        jobs = min(jobs, max(1, cpu // 2))

    return max(1, jobs)

# ----------------------------- 主程式 -----------------------------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--traces-dir", default=None,
                    help="Trace directory (default: script_dir/../data/chakra/pytorch_traces)")
    ap.add_argument("--out-dir", default=None,
                    help="Output .et directory (default: script_dir/../data/chakra/workload_et)")
    ap.add_argument("--gpu-metrics-dir", default=None,
                    help="GPU metrics directory (default: script_dir/../data/chakra/gpu_metrics)")
    ap.add_argument("--trace-pattern", default="trace_rank_*.json",
                    help="Input trace filename pattern (default: trace_rank_*.json, multi-epoch supported)")

    # Legacy parameters
    ap.add_argument("--dtype", choices=["fp32", "fp16"], default="fp32",
                    help="Gradient dtype (affects ALLREDUCE bytes calculation)")
    ap.add_argument("--model-param-count", type=int, default=17159306,
                    help="Model parameter count (default: ~17M for CIFAR10_CNN)")
    ap.add_argument("--override-bytes", type=int, default=10485760,
                    help="Direct specification of ALLREDUCE bytes per step per GPU (10MB/step)")
    ap.add_argument("--meta-version", default="0.0.4",
                    help="GlobalMetadata version (default: 0.0.4)")
    ap.add_argument("--file-prefix", default="allreduce",
                    help="Output filename prefix (default: allreduce → allreduce.{rank}.et)")
    ap.add_argument("--debug", action="store_true", help="Print detailed step metrics for debugging")

    # Large file management
    ap.add_argument("--epoch-min", type=int, default=None, help="Only process epochs >= this value")
    ap.add_argument("--epoch-max", type=int, default=None, help="Only process epochs <= this value")
    ap.add_argument("--max-steps-per-epoch", type=int, default=None, help="Maximum steps to export per epoch")
    ap.add_argument("--keep-every", type=int, default=1, help="Keep every N steps (default: 1 = keep all)")
    ap.add_argument("--no-ts-attrs", action="store_true", help="Omit timestamp attributes to reduce .et file size")

    # Compute node configuration (auto-enabled if alpha available)
    ap.add_argument("--emit-compute", action="store_true",
                    help="Force enable: Insert Compute nodes before each COMM (tries DB/guidance/bootstrap if no alpha)")
    ap.add_argument("--no-emit-compute", action="store_true",
                    help="Force disable: No Compute nodes even if alpha found, maintain legacy behavior")
    ap.add_argument("--alpha-us", type=float, default=None, help="Alpha (microseconds/cycle); will try DB/guidance/bootstrap if not provided")
    ap.add_argument("--alpha-db", type=str, default="runs/calibration_all.csv", help="Alpha database (smart filtering + median selection)")
    ap.add_argument("--alpha-quality-threshold", type=float, default=5.0,
                    help="Calibration quality threshold: filter rel_err_comm > this value (default: 5.0 = 500%%)")
    ap.add_argument("--alpha-analysis-only", action="store_true",
                    help="Only analyze alpha quality from calibration database, no conversion")
    ap.add_argument("--alpha-from-core-mhz", type=float, default=None,
                    help="Estimate alpha from GPU core frequency (MHz): alpha_us = alpha_scale / MHz (temporary guidance)")
    ap.add_argument("--alpha-from-rocm-smi", action="store_true",
                    help="Use rocm-smi to read core MHz, alpha_us = alpha_scale / MHz (temporary guidance)")
    ap.add_argument("--alpha-scale", type=float, default=30.0,
                    help="Scale factor for core frequency (temporary guidance only; default: 30.0)")
    ap.add_argument("--bootstrap-alpha", action="store_true",
                    help="If no alpha, use temporary alpha=1.0 μs/cycle to insert Compute (separate Compute/Comm for later calibration)")

    # Timestamp processing
    ap.add_argument("--rebase-ts", action="store_true", help="Make cross-epoch ts_ns monotonically increasing")

    # Bytes alias
    ap.add_argument("--bytes-per-step", type=int, default=None,
                    help="Alias for --override-bytes; takes priority if both specified")

    # Multi-core/memory management
    ap.add_argument("--jobs", type=int, default=os.cpu_count() or 2,
                    help="Number of parallel processing workers (default: CPU core count)")
    ap.add_argument("--max-memory-mb", type=int, default=None,
                    help="Maximum allowed memory usage (MB). If specified, limits jobs accordingly")
    ap.add_argument("--mem-overhead-factor", type=float, default=2.0,
                    help="Memory overhead factor for worker estimation (default: 2.0 × median file size)")

    args = ap.parse_args()
    if args.debug:
        logger.setLevel(logging.DEBUG)

    # 路徑
    script_dir = Path(__file__).resolve().parent
    traces_dir = Path(args.traces_dir).resolve() if args.traces_dir \
        else (script_dir / "../data/chakra/pytorch_traces").resolve()
    out_dir    = Path(args.out_dir).resolve() if args.out_dir \
        else (script_dir / "../data/chakra/workload_et").resolve()
    gpu_metrics_dir = Path(args.gpu_metrics_dir).resolve() if args.gpu_metrics_dir \
        else (script_dir / "../data/chakra/gpu_metrics").resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"[paths] traces={traces_dir}  out={out_dir}  gpu_metrics={gpu_metrics_dir}  json_backend={_JSON_BACKEND}")

    # File collection and filtering
    trace_files_all = sorted(traces_dir.glob(args.trace_pattern))
    if not trace_files_all:
        raise FileNotFoundError(f"No files matching {args.trace_pattern} found in {traces_dir}")

    trace_files: List[Path] = []
    for p in trace_files_all:
        try:
            _, e = rank_epoch_from_name(p)
        except Exception:
            continue
        if args.epoch_min is not None and e < args.epoch_min: continue
        if args.epoch_max is not None and e > args.epoch_max: continue
        trace_files.append(p)
        if not trace_files:
            raise RuntimeError("No trace files match criteria. Check --epoch-min/max and filename format.")    # Dynamic job estimation
    jobs = _estimate_memory_jobs(args, trace_files)
    print(f"[jobs] requested={args.jobs} → using={jobs}")

    # 多進程解析
    worlds: List[int] = []
    scales: List[float] = []
    results_by_rank: Dict[int, List[Tuple[int, List[Tuple[float, float, float, float]]]]] = {}
    infos_by_rank_epoch: Dict[Tuple[int,int], dict] = {}

    worker_args = [(str(p), args.max_steps_per_epoch, args.keep_every, args.rebase_ts) for p in trace_files]
    with ProcessPoolExecutor(max_workers=jobs) as ex:
        futs = [ex.submit(_worker_parse_one, a) for a in worker_args]
        for fut in as_completed(futs):
            r, e, world, scale_us, agg, info, err = fut.result()
            if err:
                logger.warning(err)
            if world == 0 and not agg:
                # 壞檔或解析失敗，跳過
                continue
            results_by_rank.setdefault(r, []).append((e, agg))
            infos_by_rank_epoch[(r,e)] = info
            if world > 0: worlds.append(world)
            scales.append(scale_us)

    ranks = sorted(results_by_rank.keys())
    if not ranks:
        raise RuntimeError("No valid trace data found (files may be corrupted or filtered out).")

    world_size = max(worlds) if worlds else 1
    scale_median = statistics.median(scales) if scales else 1.0
    print(f"[scale] inferred_time_scale_to_μs ≈ {scale_median:.6g}")

    # dtype → bytes
    dtype_bytes = 4 if args.dtype == "fp32" else 2
    model_param_bytes = args.model_param_count * dtype_bytes
    if args.bytes_per_step is not None:
        bytes_per_step = int(args.bytes_per_step)
    elif args.override_bytes is not None:
        bytes_per_step = int(args.override_bytes)
    else:
        bytes_per_step = per_gpu_allreduce_bytes(world_size, model_param_bytes)
    print(f"[info] world_size={world_size}  bytes_per_step={bytes_per_step}  dtype={args.dtype}")

    # 是否插 Compute 與 α（改進的優先序）
    alpha_gpu, alpha_gpu_stats = _collect_alpha_from_gpu_metrics(gpu_metrics_dir, results_by_rank)
    alpha_db, alpha_db_stats = _read_alpha_us_from_db(Path(args.alpha_db), args.alpha_quality_threshold)

    # Alpha calibration quality analysis mode
    if args.alpha_analysis_only:
        print(f"\n=== Alpha Calibration Quality Analysis ===")
        print(f"GPU metrics directory: {gpu_metrics_dir}")
        print(f"Calibration database: {args.alpha_db}")
        print(f"Quality threshold: rel_err_comm ≤ {args.alpha_quality_threshold}")

        # GPU metrics results
        print(f"\n【GPU Metrics】{alpha_gpu_stats.get('reason', 'N/A')}")
        if alpha_gpu_stats.get("collected_samples", 0) > 0:
            samples = alpha_gpu_stats["collected_samples"]
            alpha_range = alpha_gpu_stats.get("alpha_range")
            print(f"  Collected {samples} GPU-measured alpha values")
            if alpha_range:
                print(f"  Range: {alpha_range[0]:.6f} ~ {alpha_range[1]:.6f} μs/cycle")
            if alpha_gpu_stats.get("chosen_alpha"):
                print(f"  Recommended: {alpha_gpu_stats['chosen_alpha']:.6f} μs/cycle")

        # Calibration database results
        print(f"\n【Calibration Database】{alpha_db_stats.get('reason', 'N/A')}")
        if alpha_db_stats.get("total_records", 0) > 0:
            total = alpha_db_stats["total_records"]
            good = alpha_db_stats["good_records"]
            print(f"  Total calibration records: {total}, good quality: {good}")
            if alpha_db_stats.get("chosen_alpha"):
                print(f"  Recommended: {alpha_db_stats['chosen_alpha']:.6f} μs/cycle")

        # Summary recommendations
        print(f"\n【Recommendations】")
        if alpha_gpu and alpha_gpu > 0:
            print(f"  ✅ Use GPU-measured alpha = {alpha_gpu:.6f} μs/cycle (most accurate)")
        elif alpha_db and alpha_db > 0:
            print(f"  ⚠️  Use calibration DB alpha = {alpha_db:.6f} μs/cycle (check quality)")
        else:
            print(f"  ❌ No available alpha, recommend retraining with --enable-gpu-monitoring")
        return

    alpha_cli = args.alpha_us
    alpha_clock: Optional[float] = None
    if args.alpha_from_core_mhz is not None:
        alpha_clock = _alpha_from_core_mhz(args.alpha_from_core_mhz, args.alpha_scale)
    elif args.alpha_from_rocm_smi:
        mhz = _query_rocm_smi_core_mhz()
        if mhz is not None:
            alpha_clock = _alpha_from_core_mhz(mhz, args.alpha_scale)
        else:
            logger.warning("rocm-smi 無法取得核心 MHz；略過核心頻率引導。")

    # Display alpha source statistics
    if alpha_gpu_stats.get("collected_samples", 0) > 0:
        samples = alpha_gpu_stats["collected_samples"]
        alpha_range = alpha_gpu_stats.get("alpha_range", (0, 0))
        print(f"  └─ Collected {samples} training-measured alpha values, range: {alpha_range[0]:.6f} ~ {alpha_range[1]:.6f}")

    if alpha_db_stats.get("total_records", 0) > 0:
        total = alpha_db_stats["total_records"]
        good = alpha_db_stats["good_records"]
        print(f"  └─ Calibration DB: {total} total records, {good} good quality")

    alpha: Optional[float] = None
    if args.no_emit_compute:
        emit_compute = False
        reason = "forced OFF by --no-emit-compute"
    else:
        # 改進的 alpha 優先序：GPU 實測 > CLI 指定 > 校準 DB > 頻率引導 > Bootstrap
        if alpha_cli and alpha_cli > 0:
            alpha = alpha_cli; emit_compute = True; reason = "ON: --alpha-us provided (manual override)"
        elif alpha_gpu and alpha_gpu > 0:
            alpha = alpha_gpu; emit_compute = True; reason = f"ON: alpha from GPU measurements ({alpha_gpu_stats.get('reason', 'N/A')})"
        elif alpha_db and alpha_db > 0:
            alpha = alpha_db; emit_compute = True; reason = f"ON: alpha from calibration DB ({alpha_db_stats.get('reason', 'N/A')})"
        elif alpha_clock and alpha_clock > 0:
            alpha = alpha_clock; emit_compute = True; reason = f"ON: alpha from GPU core clock (scale={args.alpha_scale})"
            logger.warning("Core frequency-derived alpha is for temporary guidance only. Please calibrate with 2-GPU setup.")
        elif args.emit_compute and args.bootstrap_alpha:
            alpha = 1.0; emit_compute = True; reason = "ON: emit-compute + bootstrap-alpha=1.0"
        elif args.bootstrap_alpha:
            alpha = 1.0; emit_compute = True; reason = "AUTO: bootstrap-alpha=1.0"
        else:
            emit_compute = False; reason = "AUTO: no alpha → COMM-only"
    if emit_compute and COMPUTE_ENUM is None:
        emit_compute = False
        reason = "Compute enum missing in your Chakra build → fallback OFF"

    print(f"[compute] emit_compute={emit_compute}  alpha={alpha if alpha else 'N/A'}  ({reason})")

    # 逐 rank 輸出 .et
    for r in ranks:
        epochs_aggs = sorted(results_by_rank[r], key=lambda x: x[0])

        ts_offset_us = 0.0
        et_path = out_dir / f"{args.file_prefix}.{r}.et"
        total_steps_written = 0

        # 檢查是否觀察到 GPU 事件（用於旗標輸出）
        gpu_seen = any((infos_by_rank_epoch.get((r,e), {}).get("gpu_event_cnt", 0) > 0) for e, _ in epochs_aggs)
        comm_gpu_seen = any((infos_by_rank_epoch.get((r,e), {}).get("comm_event_cnt", 0) > 0) for e, _ in epochs_aggs)

        with et_path.open("wb") as et:
            encode_message(et, GlobalMetadata(version=args.meta_version))
            node_id = 1

            for e, agg in epochs_aggs:
                if not agg:
                    logger.warning(f"Rank{r} epoch{e} has no step data, skipping.")
                    continue

                last_end_in_epoch = 0.0
                for idx, (s_us, e_us, comp_us, comm_exposed_us) in enumerate(agg, 1):
                    step_len = max(0.0, e_us - s_us)

                    # 調整：此時 comp_us + comm_exposed_us ≤ step_len（理論上）
                    if args.debug:
                        print(f"[Debug Rank {r} E{e:02d} Step {idx:04d}] "
                              f"step_len={step_len:.3f}  comp={comp_us:.3f}  comm_exposed={comm_exposed_us:.3f}")

                    ts_ns = to_ns(s_us + (ts_offset_us if args.rebase_ts else 0.0))

                    # （可選）Compute
                    if emit_compute:
                        compute_cycles = max(1, int(round(comp_us / alpha)))
                        comp_node = ChakraNode()
                        comp_node.id = node_id; node_id += 1
                        comp_node.name = "Compute Step"
                        comp_node.type = COMPUTE_ENUM
                        comp_node.attr.append(ChakraAttr(name="is_cpu_op", bool_val=False))
                        comp_node.attr.append(ChakraAttr(name="compute_cycles", int64_val=compute_cycles))
                        comp_node.attr.append(ChakraAttr(name="exec_cycles",     int64_val=compute_cycles))
                        comp_node.attr.append(ChakraAttr(name="cycles",          int64_val=compute_cycles))
                        comp_node.attr.append(ChakraAttr(name="duration_cycles", int64_val=compute_cycles))
                        if not args.no_ts_attrs:
                            comp_node.attr.append(ChakraAttr(name="ts_ns",      int64_val=ts_ns))
                            comp_node.attr.append(ChakraAttr(name="compute_ns", int64_val=to_ns(comp_us)))
                        encode_message(et, comp_node)

                    # COMM（保持官方風格；注意：comm_ns 寫的是 **exposed_comm** 供對照）
                    comm_node = ChakraNode()
                    comm_node.id = node_id; node_id += 1
                    comm_node.name = "All-Reduce Step"
                    comm_node.type = COMM_COLL_NODE
                    comm_node.attr.append(ChakraAttr(name="is_cpu_op", bool_val=False))
                    comm_node.attr.append(ChakraAttr(name="comm_type", int64_val=ALL_REDUCE))
                    comm_node.attr.append(ChakraAttr(name="comm_size", int64_val=bytes_per_step))
                    comm_node.attr.append(ChakraAttr(name="involved_dim", bool_list=BoolList(values=[True])))
                    if not args.no_ts_attrs:
                        comm_node.attr.append(ChakraAttr(name="ts_ns",      int64_val=ts_ns))
                        comm_node.attr.append(ChakraAttr(name="compute_ns", int64_val=to_ns(comp_us)))
                        comm_node.attr.append(ChakraAttr(name="comm_ns",    int64_val=to_ns(comm_exposed_us)))
                    encode_message(et, comm_node)

                    total_steps_written += 1
                    last_end_in_epoch = max(last_end_in_epoch, e_us)

                if args.rebase_ts:
                    ts_offset_us += last_end_in_epoch

        flag = []
        if not gpu_seen:
            flag.append("no_gpu_events_observed")
        if gpu_seen and not comm_gpu_seen:
            flag.append("comm_unobserved_no_gpu_events")
        print(f"[OK] wrote {et_path}  steps={total_steps_written}  "
              f"(emit-compute={'ON' if emit_compute else 'OFF'}, "
              f"rebase-ts={'ON' if args.rebase_ts else 'OFF'}, "
              f"json={_JSON_BACKEND})"
              + (f"  flags={','.join(flag)}" if flag else ""))

if __name__ == "__main__":
    main()
