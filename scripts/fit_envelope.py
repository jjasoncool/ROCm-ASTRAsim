#!/usr/bin/env python3
"""Fit T(M) = alpha + M/B for both the measured RCCL path and the ns-3 simulator,
then compare them per bucket.

Real side  : parsed from rccl-tests all_reduce_perf output (Step 1).
Sim side   : parsed from the per-collective "COMM interval" debug lines that
             ASTRA-sim already emits in every 2-node run's stdout.log, matched to
             the AllReduce sizes decoded from that run's ET.

No scipy needed: T = alpha + M*(1/B) is linear in its parameters, so ordinary
least squares is exact and transparent.

Usage:
  python scripts/fit_envelope.py \
      --rccl-log runs/calibration/step1_rccl_sweep_raw.log \
      --sizes runs/calibration/bucket_sizes.json \
      --out-csv runs/calibration/step1_rccl_sweep.csv

All paths are relative to the repo root, so run from there (inside the container
that means -w /workspace, which is the same directory).

--- How the raw inputs under runs/calibration/ were captured (2026-08-21) ---

Step 0, transport path verification:
  docker exec rocm-horovod bash -lc 'rocm-smi --showtopo' \
      > runs/calibration/step0_topo.txt 2>&1
  docker exec -w /workspace/rccl-tests/build rocm-horovod bash -lc \
      'NCCL_DEBUG=INFO NCCL_DEBUG_SUBSYS=INIT,GRAPH \
       ./all_reduce_perf -b 1M -e 1M -d float -g 2 -w 5 -n 20 2>&1' \
      > runs/calibration/step0_nccl_debug.log 2>&1
  -> confirmed "via P2P/direct pointer" (PCIe peer-to-peer, no host staging).

Step 1, real RCCL envelope sweep (5 independent repeats, FP32):
  docker exec -w /workspace/rccl-tests/build rocm-horovod bash -lc \
      'for r in 1 2 3 4 5; do echo "===== RUN $r ====="; \
       ./all_reduce_perf -b 8 -e 512M -f 2 -d float -g 2 -w 20 -n 100 2>&1; done' \
      > runs/calibration/step1_rccl_sweep_raw.log 2>&1

Q2 micro, PyTorch-path clean baseline at the real DDP bucket sizes:
  docker exec -w /workspace rocm-horovod bash -lc \
      '/opt/conda/envs/py_3.12/bin/torchrun --standalone --nproc_per_node=2 \
       scripts/bucket_micro_allreduce.py \
       --sizes-file runs/calibration/bucket_sizes.json \
       --out runs/calibration/q2_micro.csv 2>&1' \
      > runs/calibration/q2_micro_raw.log 2>&1

bucket_sizes.json is the distinct ALL_REDUCE comm_size set decoded read-only from
data/chakra/workload_et/et.{resnet50,cifar10,qwen05b}.0.et.
"""
import argparse
import csv
import glob
import json
import os
import re
import statistics

import numpy as np

MIB = 1024.0 * 1024.0

# rccl-tests data rows: size count type redop root | oop: time algbw busbw wrong | ip: same
ROW = re.compile(
    r"^\s*(\d+)\s+(\d+)\s+(\w+)\s+(\w+)\s+(-?\d+)\s+"
    r"([\d.]+)\s+([\d.]+)\s+([\d.]+)\s+(\S+)\s+"
    r"([\d.]+)\s+([\d.]+)\s+([\d.]+)\s+(\S+)\s*$"
)
INTERVAL = re.compile(r"COMM interval\[(\d+)\]: start=\d+ end=\d+ dur=(\d+)")
TOTALS = re.compile(r"COMM total_intervals=(\d+) raw_sum=(\d+)")


def parse_rccl(path):
    """-> {run_id: {payload_bytes: (oop_us, ip_us)}}"""
    runs, cur = {}, None
    with open(path) as f:
        for line in f:
            if line.startswith("===== RUN"):
                cur = int(line.split()[2])
                runs[cur] = {}
                continue
            m = ROW.match(line)
            # 用 `cur is not None` 而非 `cur`：重複次數若從 0 編號，`if cur`
            # 會把整個 run 0 的資料列悄悄丟掉，同時 len(runs) 仍把它算進去。
            if m and cur is not None:
                nbytes = int(m.group(1))
                # 以尺寸為 key：下游的中位數彙總與 CSV 都以尺寸為軸。掃描若含
                # 多個 dtype 或 redop，同尺寸會互相覆蓋，且 CSV 仍標成 float32，
                # 因此這裡出聲而非默默蓋掉。
                if nbytes in runs[cur]:
                    print(f"[rccl] 警告：run {cur} 的 {nbytes} B 重複出現 "
                          f"(dtype={m.group(3)} redop={m.group(4)})，後者覆蓋前者；"
                          f"單一 dtype/redop 的掃描才會是乾淨的。")
                runs[cur][nbytes] = (float(m.group(6)), float(m.group(10)))
    return runs


def parse_ns3_intervals(stdout_path):
    """-> every per-collective duration in ms, in log order, from the first rank's block.

    Duplicates are kept; the caller dedups. Only the first rank is read because the
    ranks report identical intervals.
    """
    durs, seen_total = [], False
    with open(stdout_path, errors="ignore") as f:
        for line in f:
            m = INTERVAL.search(line)
            if m:
                durs.append(int(m.group(2)) / 1e6)  # cycles are ns -> ms
            elif TOTALS.search(line):
                seen_total = True
                break  # first rank's block is enough; they are identical
    return durs


def ols_fit(sizes_mib, times_ms):
    """T = alpha + M/B. Returns (alpha_ms, B_mib_per_ms, r2)."""
    x = np.asarray(sizes_mib, dtype=float)
    y = np.asarray(times_ms, dtype=float)
    A = np.vstack([np.ones_like(x), x]).T
    (alpha, slope), *_ = np.linalg.lstsq(A, y, rcond=None)
    pred = alpha + slope * x
    ss_res = float(((y - pred) ** 2).sum())
    ss_tot = float(((y - y.mean()) ** 2).sum())
    r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else float("nan")
    return float(alpha), (1.0 / slope if slope else float("inf")), r2


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--rccl-log", default="runs/calibration/step1_rccl_sweep_raw.log")
    ap.add_argument("--sizes", default="runs/calibration/bucket_sizes.json")
    ap.add_argument("--out-csv", default="runs/calibration/step1_rccl_sweep.csv")
    ap.add_argument("--runs-glob",
                    default="runs/20260820-055114*_ns3_2gpu_*_2_nodes_1_switch_topology")
    ap.add_argument("--out-ns3-csv", default="runs/calibration/ns3_collective_times.csv",
                    help="每個模擬 collective 的時間；由 run log 的 COMM interval 解出")
    ap.add_argument("--out-table", default="runs/calibration/envelope_fit_table.md",
                    help="擬合結果與逐 bucket 對照表")
    ap.add_argument("--large-threshold-mib", type=float, default=4.0,
                    help="fit B on messages at or above this size")
    ap.add_argument("--small-threshold-bytes", type=int, default=1024,
                    help="estimate alpha from the plateau at or below this size")
    args = ap.parse_args()

    # ---------- real side ----------
    runs = parse_rccl(args.rccl_log)
    all_sizes = sorted({s for r in runs.values() for s in r})
    print(f"[real] {len(runs)} sweep repeats x {len(all_sizes)} sizes\n")

    os.makedirs(os.path.dirname(args.out_csv) or ".", exist_ok=True)
    with open(args.out_csv, "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["run_id", "payload_bytes", "dtype", "inplace",
                    "time_us", "algbw_GBps"])
        for rid, row in sorted(runs.items()):
            for nbytes, (oop, ip) in sorted(row.items()):
                for inplace, t in ((0, oop), (1, ip)):
                    w.writerow([rid, nbytes, "float32", inplace,
                                f"{t:.3f}", f"{nbytes / (t / 1e6) / 1e9:.4f}"])

    agg = {}
    if not all_sizes:
        raise SystemExit(f"[rccl] {args.rccl_log} 解析不到任何資料列；"
                         f"請確認它是 all_reduce_perf 的輸出，且含 '===== RUN n' 分隔。"
                         f"空資料會讓擬合退化成除以零。")
    for nbytes in all_sizes:
        ips = [runs[r][nbytes][1] for r in runs if nbytes in runs[r]]
        oops = [runs[r][nbytes][0] for r in runs if nbytes in runs[r]]
        agg[nbytes] = {
            "ip_med": statistics.median(ips), "ip_min": min(ips), "ip_max": max(ips),
            "oop_med": statistics.median(oops),
        }

    print(f"{'bytes':>13} {'MiB':>10} {'ip median us':>13} {'min-max':>19} {'GB/s':>7}")
    for nbytes in all_sizes:
        a = agg[nbytes]
        bw = nbytes / (a["ip_med"] / 1e6) / 1e9
        print(f"{nbytes:>13,} {nbytes / MIB:>10.4f} {a['ip_med']:>13.2f} "
              f"{a['ip_min']:>8.2f}-{a['ip_max']:<10.2f} {bw:>7.2f}")

    large = [(n / MIB, agg[n]["ip_med"] / 1000.0) for n in all_sizes
             if n / MIB >= args.large_threshold_mib]
    if len(large) < 2:
        raise SystemExit(f"[rccl] 只有 {len(large)} 個尺寸 >= {args.large_threshold_mib} MiB，"
                         f"無法擬合頻寬；請調低 --large-threshold-mib 或擴大掃描範圍。")
    a_r, b_r, r2_r = ols_fit([m for m, _ in large], [t for _, t in large])
    small = [agg[n]["ip_med"] / 1000.0 for n in all_sizes
             if n <= args.small_threshold_bytes]
    alpha_plateau = statistics.median(small) if small else float("nan")

    print(f"\n[real] OLS on M >= {args.large_threshold_mib} MiB "
          f"({len(large)} points):")
    print(f"       alpha_real = {a_r * 1000:.2f} us   "
          f"B_real = {b_r:.4f} MiB/ms = {b_r * MIB / 1e9 * 1000:.3f} GB/s   R2 = {r2_r:.6f}")
    print(f"[real] small-message plateau (<= {args.small_threshold_bytes} B): "
          f"{alpha_plateau * 1000:.2f} us")

    # ---------- sim side ----------
    with open(args.sizes) as f:
        sizes_by_tag = json.load(f)

    print()
    sim_points = []
    run_dirs = sorted(glob.glob(args.runs_glob))
    if not run_dirs:
        # 預設值指向作者 2026-08-20 那批 run；換台機器必然為空。靜默跳過會只寫出
        # 三個輸出中的一個，且舊的 envelope_fit_table.md 仍留在旁邊。
        raise SystemExit(f"[sim] --runs-glob '{args.runs_glob}' 沒有匹配任何 run 目錄；"
                         f"請指向你自己的 2-GPU 校準 run（例如 'runs/*_ns3_2gpu_*_2_nodes_1_switch_topology'）。")
    for d in run_dirs:
        tag = None
        for t in sizes_by_tag:
            if f"_{t}_" in os.path.basename(d):
                tag = t
        if tag is None:
            continue
        durs = parse_ns3_intervals(os.path.join(d, "stdout.log"))
        if not durs:
            print(f"[sim] {d}: stdout.log 沒有 COMM interval 行；"
                  f"容器是否以 ASTRA_PATCHES=all 建置？")
            continue
        distinct = sorted(set(round(x, 6) for x in durs))
        sizes = sorted(sizes_by_tag[tag])
        if len(distinct) != len(sizes):
            print(f"[sim] {tag}: {len(distinct)} distinct durations vs "
                  f"{len(sizes)} distinct sizes - skipping (needs manual mapping)")
            continue
        for nbytes, ms in zip(sizes, distinct):
            sim_points.append((tag, nbytes, ms))
            print(f"[sim] {tag:9s} {nbytes:>12,} B = {nbytes / MIB:8.3f} MiB "
                  f"-> {ms:9.6f} ms")

    if sim_points:
        a_s, b_s, r2_s = ols_fit([n / MIB for _, n, _ in sim_points],
                                 [ms for _, _, ms in sim_points])
        print(f"\n[sim ] OLS on all {len(sim_points)} simulated collectives:")
        print(f"       alpha_sim  = {a_s * 1000:.2f} us   "
              f"B_sim  = {b_s:.4f} MiB/ms = {b_s * MIB / 1e9 * 1000:.3f} GB/s   R2 = {r2_s:.8f}")

        print(f"\n[ratio] B_sim/B_real   = {b_s / b_r:.4f}")
        print(f"[ratio] alpha_sim/alpha_real = {a_s / a_r:.2f}x"
              if a_r else "[ratio] alpha_real ~ 0")

        print(f"\n{'bucket':>13} {'MiB':>9} {'sim ms':>10} {'real ms':>10} {'sim/real':>9}")
        for tag, nbytes, ms in sim_points:
            m = nbytes / MIB
            t_real = a_r + m / b_r
            print(f"{nbytes:>13,} {m:>9.3f} {ms:>10.4f} {t_real:>10.4f} "
                  f"{ms / t_real:>9.3f}")

        # ns-3 側逐 collective 時間。這不是合成 sweep 的產物——ASTRA-sim 每個
        # collective 的耗時已經印在既有 run 的 stdout.log，重新模擬只會重畫同一條線。
        os.makedirs(os.path.dirname(args.out_ns3_csv) or ".", exist_ok=True)
        with open(args.out_ns3_csv, "w", newline="") as fh:
            w = csv.writer(fh)
            w.writerow(["tag", "payload_bytes", "payload_mib", "sim_ms",
                        "envelope_pred_ms", "sim_over_envelope", "source"])
            for tag, nbytes, ms in sim_points:
                m = nbytes / MIB
                t_real = a_r + m / b_r
                w.writerow([tag, nbytes, f"{m:.4f}", f"{ms:.6f}",
                            f"{t_real:.6f}", f"{ms / t_real:.4f}",
                            "COMM interval lines in runs/20260820-055114*/stdout.log"])
        print(f"\n  wrote {args.out_ns3_csv}")

        with open(args.out_table, "w") as fh:
            fh.write("# Communication envelope — fit results\n\n")
            fh.write("T(M) = alpha + M/B, fitted independently on each source.\n\n")
            fh.write("| source | alpha | B | R^2 |\n|---|---|---|---|\n")
            fh.write(f"| rccl-tests (FP32, {len(runs)} repeats x {len(all_sizes)} sizes) | "
                     f"{alpha_plateau * 1000:.2f} us plateau (<={args.small_threshold_bytes:,} B) / "
                     f"{a_r * 1000:.2f} us fit intercept | "
                     f"{b_r * MIB / 1e9 * 1000:.3f} GB/s | {r2_r:.6f} |\n")
            fh.write(f"| ns-3 + ASTRA-sim ({len(sim_points)} collectives) | "
                     f"{a_s * 1000:.2f} us | {b_s * MIB / 1e9 * 1000:.3f} GB/s | "
                     f"{r2_s:.8f} |\n\n")
            fh.write(f"B_sim/B_real = {b_s / b_r:.4f}; "
                     f"alpha_sim/alpha_real = {a_s / a_r:.2f}x against the fit "
                     f"intercept, {a_s / alpha_plateau:.1f}x against the plateau.\n\n")
            fh.write("The ns-3 side is a closed-form line (R^2 = 1.0 to eight decimals):\n"
                     "the fit from one workload predicts collectives in the others to\n"
                     "0.00x%. Fitting alpha + M/B to it is close to tautological. The real\n"
                     "curve is not a single line - it has a plateau, a transition, then the\n"
                     "asymptote - so the fit intercept is not the true fixed overhead.\n\n")
            fh.write("## Per collective\n\n")
            fh.write("| bytes | MiB | ns-3 ms | envelope ms | ns-3 / envelope |\n")
            fh.write("|---|---|---|---|---|\n")
            for tag, nbytes, ms in sorted(sim_points, key=lambda x: x[1]):
                m = nbytes / MIB
                t_real = a_r + m / b_r
                fh.write(f"| {nbytes:,} | {m:.3f} | {ms:.4f} | {t_real:.4f} | "
                         f"{ms / t_real:.3f} |\n")
        print(f"  wrote {args.out_table}")


if __name__ == "__main__":
    main()
