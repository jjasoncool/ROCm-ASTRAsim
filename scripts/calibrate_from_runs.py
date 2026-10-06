#!/usr/bin/env python3
"""Compute calibration metrics from existing run directories, without simulating.

Reuses run_ns3.align_and_compare so this and a live run cannot drift apart.
Inputs are all on disk already: the run's stdout.log for simulated cycles, the
Kineto traces, and the source ET.

Usage:
  python3 scripts/calibrate_from_runs.py --out runs/calibration_aligned.csv \
      resnet50=1:<run_dir> cifar10=4:<run_dir> qwen05b=2:<run_dir>

Each argument is tag=et_iters:run_dir. et_iters is how many training iterations
the ET covers (the --trace-steps used when it was generated); it is required
because alpha_us needs a per-step denominator and guessing it produces a
plausible-looking conversion factor that is silently wrong.
"""
import argparse
import csv
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import run_ns3 as R

FIELDS = ["tag", "run_dir", "selected_rank", "trace_kernel_count",
          "et_collective_count", "window_ratio", "et_iterations",
          "trace_iterations", "et_per_iter", "trace_per_iter",
          "sim_cycles_step", "sim_cycles_comm",
          "ns3_comm_ms", "ns3_comm_ms_aligned",
          "real_t_net_comm_ms", "real_t_net_comm_ms_rank0", "real_t_net_comm_ms_rank1",
          "real_t_step_ms", "real_t_kernel_ms",
          "ns3_signed_err_comm", "ns3_abs_err_comm", "alpha_us", "alpha_comm_us", "flags"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("targets", nargs="+", help="tag=et_iters:run_dir")
    ap.add_argument("--workload", default="data/chakra/workload_et")
    ap.add_argument("--trace-dir", default="data/chakra/pytorch_traces")
    ap.add_argument("--out", default="runs/calibration_aligned.csv")
    args = ap.parse_args()

    rows = []
    for t in args.targets:
        spec, run_dir = t.split(":", 1)
        tag, iters = spec.split("=")
        iters = int(iters)
        run_dir = Path(run_dir)
        print(f"\n=== {tag} (et_iters={iters}) {run_dir}")

        step_c, comm_c, gpu_c = R.parse_astra_stdout_cycles(run_dir / "stdout.log")
        real_step, per_rank, epoch = R.extract_real_metrics_from_traces(
            Path(args.trace_dir), tag)
        et_n = R.count_et_collectives(Path(args.workload), tag)
        calib = R.align_and_compare(per_rank, et_n, step_c, comm_c, real_step, iters)

        print(f"  sim_cycles_comm={comm_c:,} " if comm_c is not None else "  sim_cycles_comm=(缺) ",
              end="")
        print(f" et_collectives={et_n}  "
              f"selected_rank={calib['selected_rank']}  window_ratio={calib['window_ratio']}")
        if calib["ns3_signed_err_comm"] is not None:
            print(f"  ns3_aligned={calib['ns3_comm_ms_aligned']:.6f} ms  "
                  f"real={calib['real_t_net_comm_ms']:.6f} ms  "
                  f"signed_err={calib['ns3_signed_err_comm'] * 100:+.2f}%")
        if calib["alpha_us"] is not None:
            print(f"  alpha_us={calib['alpha_us']:.6f}")

        rows.append({
            "tag": tag, "run_dir": str(run_dir),
            "sim_cycles_step": step_c, "sim_cycles_comm": comm_c,
            "et_collective_count": et_n, "real_t_step_ms": real_step,
            **{k: calib.get(k) for k in
               ("selected_rank", "trace_kernel_count", "window_ratio", "et_iterations",
                "trace_iterations", "et_per_iter", "trace_per_iter",
                "ns3_comm_ms", "ns3_comm_ms_aligned", "real_t_net_comm_ms",
                "real_t_net_comm_rank0", "real_t_net_comm_rank1", "real_t_kernel_ms",
                "ns3_signed_err_comm", "ns3_abs_err_comm", "alpha_us", "alpha_comm_us")},
            "real_t_net_comm_ms_rank0": calib.get("real_t_net_comm_rank0"),
            "real_t_net_comm_ms_rank1": calib.get("real_t_net_comm_rank1"),
            "flags": "|".join(calib["flags"]),
        })

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=FIELDS, extrasaction="ignore")
        w.writeheader()
        for r in rows:
            w.writerow({k: ("" if r.get(k) is None else r.get(k)) for k in FIELDS})
    print(f"\nwrote {out}")


if __name__ == "__main__":
    main()
