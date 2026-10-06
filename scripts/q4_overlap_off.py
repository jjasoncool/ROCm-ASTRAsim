#!/usr/bin/env python3
"""Time Qwen 0.5B's DDP AllReduce buckets with the overlap against backward removed.

Third column of the Q2 decomposition. q2_micro.csv enqueues the same byte sizes
back to back on a standalone tensor; the Kineto trace records them racing a
backward pass. This takes the buckets from a real DDP step: the comm hook drains
the GPU with torch.cuda.synchronize() and then brackets dist.all_reduce on the
real bucket buffer, so backward is paused and nothing overlaps the transfer.

Only the all_reduce sits between the two CUDA events. DDP's bucketing, gradient
copy-in/out and averaging run outside them, so this does NOT measure DDP framework
overhead. What differs from q2_micro is that each collective is launched alone
onto an idle GPU, which puts per-call launch latency and any wait for the peer
rank inside the window.

One comm hook serves both phases (DDP allows only one per instance): it records
bucket sizes during warmup, then times them. Phase 1 clears its accumulator
before every step, so the recorded set is exactly one step's buckets — deriving
it by dividing the total hook count by --warmup-steps silently truncates, because
DDP's first iteration rebuilds its buckets and fires a different number of hooks.
Compare the printed set against `python src/tests/validate_et.py --prefix et.qwen05b`
to confirm both sides cover the same collectives.

Run:
  torchrun --standalone --nproc_per_node=2 scripts/q4_overlap_off.py \
      --out runs/calibration/q4_overlap_off.csv
"""
import argparse
import csv
import os
import statistics

import torch
import torch.distributed as dist
import torch.nn as nn
from torch.nn.parallel import DistributedDataParallel as DDP


def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model-dir", default="/workspace/data/models")
    ap.add_argument("--seq-len", type=int, default=256)
    ap.add_argument("--batch-size", type=int, default=2)
    ap.add_argument("--warmup-steps", type=int, default=3)
    ap.add_argument("--steps", type=int, default=10)
    ap.add_argument("--out", default="runs/calibration/q4_overlap_off.csv")
    return ap.parse_args()


def main():
    args = parse_args()
    dist.init_process_group(backend="nccl")
    rank = dist.get_rank()
    local_rank = int(os.environ.get("LOCAL_RANK", rank))
    torch.cuda.set_device(local_rank)
    dev = torch.device(f"cuda:{local_rank}")

    from transformers import AutoModelForCausalLM

    if rank == 0:
        print("[q4] loading Qwen2.5-0.5B (FP32)", flush=True)
    model = AutoModelForCausalLM.from_pretrained(
        "Qwen/Qwen2.5-0.5B", dtype=torch.float32, cache_dir=args.model_dir
    ).to(dev)
    ddp = DDP(model, device_ids=[local_rank], broadcast_buffers=False)
    opt = torch.optim.AdamW(ddp.parameters(), lr=1e-4)
    loss_fn = nn.CrossEntropyLoss()

    vocab = model.config.vocab_size
    def batch():
        ids = torch.randint(0, vocab, (args.batch_size, args.seq_len), device=dev)
        return ids, ids.clone()

    def step(sync):
        ids, labels = batch()
        ctx = ddp.no_sync() if not sync else torch.enable_grad()
        with ctx:
            out = ddp(input_ids=ids).logits
            loss = loss_fn(out.view(-1, vocab), labels.view(-1))
            loss.backward()
        return loss

    # 單一 hook 兩種模式：DDP 每個實例只能註冊一次 comm hook。
    #   record -> 只記尺寸，用來還原這一步真正的 bucket 佈局
    #   time   -> 先同步再只計時 all_reduce；DDP 的分桶與複製不在計時區間內
    world = dist.get_world_size()
    state = {"mode": "record", "cur": []}

    def hook(_st, bucket):
        buf = bucket.buffer()
        nbytes = buf.numel() * buf.element_size()
        if state["mode"] == "time":
            # 先同步：本 bucket 之前的 backward 工作全部做完，collective 才開始，
            # 這樣量到的是「無重疊」而非「與 backward 競爭」。
            torch.cuda.synchronize()
            s_ev = torch.cuda.Event(enable_timing=True)
            e_ev = torch.cuda.Event(enable_timing=True)
            s_ev.record()
            dist.all_reduce(buf)
            e_ev.record()
            torch.cuda.synchronize()
            state["cur"].append((nbytes, s_ev.elapsed_time(e_ev)))
        else:
            dist.all_reduce(buf)
            state["cur"].append((nbytes, None))
        buf.div_(world)          # 預設 hook 會除以 world_size，這裡比照辦理
        fut = torch.futures.Future()
        fut.set_result(buf)
        return fut

    ddp.register_comm_hook(state=None, hook=hook)

    # ---- phase 1: learn DDP's real bucket layout ----
    # 每步前清空再收：DDP 第一次迭代會重建 bucket，觸發次數與後續不同，
    # 用「總次數 / warmup_steps」回推每步 bucket 數會默默截斷成員。
    last_step_sizes = []
    for _ in range(args.warmup_steps):
        state["cur"] = []
        opt.zero_grad(set_to_none=True)
        step(sync=True)
        opt.step()
        torch.cuda.synchronize()
        last_step_sizes = [n for n, _ in state["cur"]]
    dist.barrier()

    sizes = sorted(last_step_sizes)
    if rank == 0:
        from collections import Counter
        total = sum(sizes)
        print(f"[q4] DDP buckets/step: {len(sizes)}  總計 {total:,} B "
              f"({total / 2**20:.1f} MiB)", flush=True)
        for s_, n in sorted(Counter(sizes).items()):
            print(f"[q4]   {s_:>12,} B = {s_ / 2**20:8.3f} MiB  x{n}", flush=True)
        print("[q4] 請與 `python src/tests/validate_et.py --prefix et.qwen05b` 的 bucket "
              "清單比對，確認量到的是同一批 collective。", flush=True)

    # ---- phase 2: 同一批 bucket，DDP hook 留在計時區間內、但不與 backward 重疊 ----
    per_bucket = {}
    for it in range(args.steps):
        state["mode"] = "time"
        state["cur"] = []
        opt.zero_grad(set_to_none=True)
        step(sync=True)
        opt.step()
        torch.cuda.synchronize()
        if it >= 2:  # 丟掉前兩步當暖身
            for nbytes, ms in sorted(state["cur"], key=lambda x: x[0]):
                per_bucket.setdefault(nbytes, []).append(ms)

    sizes = sorted(per_bucket) if per_bucket else sizes

    if rank == 0:
        os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
        with open(args.out, "w", newline="") as fh:
            w = csv.writer(fh)
            # 一列一個相異 payload 尺寸；n 是該尺寸累積到的計時樣本數
            # （步數 × 該尺寸的 bucket 個數），相同尺寸的 bucket 合併統計。
            w.writerow(["payload_bytes", "n",
                        "median_ms", "min_ms", "max_ms", "algbw_GBps_median"])
            for n in sizes:
                v = sorted(x for x in per_bucket.get(n, []) if x is not None)
                if not v:
                    continue
                med = statistics.median(v)
                w.writerow([n, len(v), f"{med:.6f}", f"{v[0]:.6f}",
                            f"{v[-1]:.6f}", f"{n / (med / 1000.0) / 1e9:.4f}"])
                print(f"[q4] {n:>12,} B  median={med:9.4f} ms  "
                      f"min={v[0]:9.4f}  max={v[-1]:9.4f}", flush=True)
        print(f"[q4] wrote {args.out}", flush=True)

    dist.destroy_process_group()


if __name__ == "__main__":
    main()
