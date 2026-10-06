#!/usr/bin/env python3
"""Contention-free micro-benchmark of torch.distributed.all_reduce at the exact
DDP bucket sizes recorded in the Chakra ETs.

Purpose: the Kineto traces give in-situ RCCL kernel durations (AllReduce racing a
backward pass). This script measures the same collectives on the same software
path with nothing else running, so the two can be differenced per bucket.

We deliberately use the PyTorch path (not rccl-tests) because that is the path the
traces recorded; rccl-tests covers the raw-RCCL envelope separately.

Timing mirrors how Kineto reports a kernel: one CUDA event pair per individual
all_reduce, enqueued back to back, so each measurement brackets a single kernel
rather than an amortised loop.

Run (inside the rocm-horovod container; /workspace/runs is bind-mounted, so the
output lands directly in the repo with no copy step):
  torchrun --nproc_per_node=2 scripts/bucket_micro_allreduce.py \
      --sizes-file runs/calibration/bucket_sizes.json \
      --out runs/calibration/q2_micro.csv
"""
import argparse
import csv
import json
import os
import statistics

import torch
import torch.distributed as dist


def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sizes-file", default=None,
                    help="JSON: {\"tag\": [byte_size, ...], ...}")
    ap.add_argument("--sizes", default=None,
                    help="Comma-separated byte sizes; overrides --sizes-file")
    ap.add_argument("--dtype", default="float32", choices=["float32", "float16"])
    ap.add_argument("--warmup", type=int, default=20)
    ap.add_argument("--iters", type=int, default=100)
    ap.add_argument("--repeats", type=int, default=5)
    ap.add_argument("--out", default="runs/calibration/q2_micro.csv")
    return ap.parse_args()


def collect_sizes(args):
    """Returns [(tag, byte_size)] deduplicated on byte_size, ascending."""
    if args.sizes:
        # 與 --sizes-file 同樣去重並遞增排序：重複的尺寸只會重複量同一件事。
        uniq = sorted({int(s) for s in args.sizes.split(",") if s.strip()})
        return [("cli", s) for s in uniq]
    with open(args.sizes_file) as f:
        by_tag = json.load(f)
    seen = {}
    for tag, sizes in by_tag.items():
        for s in sizes:
            seen.setdefault(int(s), tag)
    return [(tag, size) for size, tag in sorted(seen.items())]


def time_one_size(tensor, warmup, iters):
    """Per-call CUDA event timing, back-to-back enqueue. Returns list of ms."""
    for _ in range(warmup):
        dist.all_reduce(tensor)
    torch.cuda.synchronize()
    dist.barrier()

    starts = [torch.cuda.Event(enable_timing=True) for _ in range(iters)]
    ends = [torch.cuda.Event(enable_timing=True) for _ in range(iters)]
    for i in range(iters):
        starts[i].record()
        dist.all_reduce(tensor)
        ends[i].record()
    torch.cuda.synchronize()
    return [s.elapsed_time(e) for s, e in zip(starts, ends)]


def main():
    args = parse_args()
    dist.init_process_group(backend="nccl")
    rank = dist.get_rank()
    local_rank = int(os.environ.get("LOCAL_RANK", rank))
    torch.cuda.set_device(local_rank)

    dtype = torch.float32 if args.dtype == "float32" else torch.float16
    itemsize = torch.empty(0, dtype=dtype).element_size()
    sizes = collect_sizes(args)

    if rank == 0:
        os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
        fh = open(args.out, "w", newline="")
        writer = csv.writer(fh)
        writer.writerow([
            "repeat", "tag", "payload_bytes", "numel", "dtype",
            "median_ms", "min_ms", "max_ms", "p10_ms", "p90_ms",
            "algbw_GBps_median", "algbw_GBps_min",
        ])
        print(f"[micro] {len(sizes)} distinct bucket sizes, dtype={args.dtype}, "
              f"warmup={args.warmup} iters={args.iters} repeats={args.repeats}",
              flush=True)

    for repeat in range(1, args.repeats + 1):
        for tag, nbytes in sizes:
            numel = nbytes // itemsize
            if numel == 0:
                continue
            t = torch.ones(numel, dtype=dtype, device=f"cuda:{local_rank}")
            times = time_one_size(t, args.warmup, args.iters)
            del t
            torch.cuda.empty_cache()

            if rank == 0:
                times.sort()
                med = statistics.median(times)
                lo, hi = times[0], times[-1]
                p10 = times[int(0.10 * (len(times) - 1))]
                p90 = times[int(0.90 * (len(times) - 1))]
                # algbw: bytes moved / time, as rccl-tests defines it
                bw_med = nbytes / (med / 1000.0) / 1e9
                bw_min = nbytes / (hi / 1000.0) / 1e9
                writer.writerow([
                    repeat, tag, nbytes, numel, args.dtype,
                    f"{med:.6f}", f"{lo:.6f}", f"{hi:.6f}",
                    f"{p10:.6f}", f"{p90:.6f}",
                    f"{bw_med:.4f}", f"{bw_min:.4f}",
                ])
                fh.flush()
                print(f"[micro r{repeat}] {nbytes:>12,} B  median={med:9.4f} ms  "
                      f"min={lo:9.4f}  max={hi:9.4f}  {bw_med:6.2f} GB/s", flush=True)
        dist.barrier()

    if rank == 0:
        fh.close()
        print(f"[micro] wrote {args.out}", flush=True)
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
