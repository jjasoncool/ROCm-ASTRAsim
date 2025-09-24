#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
CIFAR-10 on ROCm (AMD GPU) with unified CPU+CUDA traces per rank.
- 單卡:  torchrun --standalone --nproc_per_node=1 ./src/train_rocm_pytorch.py --epochs 2 --batch-size 128
- 雙卡:  torchrun --standalone --nproc_per_node=2 ./src/train_rocm_pytorch.py --epochs 2 --batch-size 128
- 雙卡多事件: torchrun --standalone --nproc_per_node=2 ./src/train_rocm_pytorch.py --epochs=10 --batch-size=512 --omp-threads=4 --debug-epoch-print

輸出 (相對於此檔案):
  ../data/chakra/pytorch_traces/trace_rank_{rank}_epoch_{epoch}.json  # 統一檔案，含 CPU+CUDA events
資料集:
  ../data/cifar10
"""

import os
import argparse
import torch
import torch.nn as nn
import torch.optim as optim
import torch.distributed as dist
from torch.utils.data import DataLoader
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.profiler import profile, ProfilerActivity, schedule, record_function
from tqdm import tqdm
import torchvision
import torchvision.transforms as transforms

# ------------------ 模型 ------------------
class CIFAR10_CNN(nn.Module):
    def __init__(self, num_classes=10):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 64, 3, padding=1), nn.ReLU(True),
            nn.Conv2d(64, 128, 3, padding=1), nn.ReLU(True),
            nn.MaxPool2d(2),                 # 128x16x16
            nn.Conv2d(128, 256, 3, padding=1), nn.ReLU(True),
            nn.MaxPool2d(2),                 # 256x8x8
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(256*8*8, 1024), nn.ReLU(True),
            nn.Linear(1024, num_classes),
        )
    def forward(self, x):
        return self.classifier(self.features(x))

# ------------------ DDP ------------------
def setup_ddp():
    if int(os.environ.get("WORLD_SIZE", "1")) > 1:
        dist.init_process_group(backend="nccl", init_method="env://")
        local_rank = int(os.environ["LOCAL_RANK"])
        torch.cuda.set_device(local_rank)
        return int(os.environ["RANK"]), int(os.environ["WORLD_SIZE"]), True
    return 0, 1, False

def cleanup_ddp(enabled):
    if enabled and dist.is_initialized():
        try:
            dist.barrier()
        except Exception as e:
            print(f"[Cleanup] Barrier error: {e}")
        finally:
            dist.destroy_process_group()

def is_main(rank):
    return rank == 0

# ------------------ DDP Comm Hook：加標籤，讓 Kineto 一定看到通訊事件 ------------------
def make_tagging_allreduce_hook():
    # 使用 PyTorch 內建 allreduce hook，外面套一層 record_function 標籤
    from torch.distributed.algorithms.ddp_comm_hooks import default_hooks as dh
    def hook(state, bucket):
        with record_function("nccl_all_reduce"):
            return dh.allreduce_hook(state, bucket)
    return hook

# ------------------（保留原本）訓練一個 epoch（不含 profiler） ------------------
def train_one_epoch(model, loader, device, optimizer, scaler, loss_fn, epoch, pbar_desc=None):
    model.train()
    total, correct, loss_sum = 0, 0, 0.0
    is_rank0 = is_main(int(os.environ.get("RANK", "0")))
    iterator = tqdm(loader, desc=pbar_desc or f"Epoch {epoch}", dynamic_ncols=True) if is_rank0 else loader

    for x, y in iterator:
        x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)
        optimizer.zero_grad(set_to_none=True)
        with torch.autocast(device_type="cuda", dtype=torch.float16):
            logits = model(x)
            loss = loss_fn(logits, y)
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        loss_sum += loss.item()
        total += y.size(0)
        correct += (logits.argmax(1) == y).sum().item()

        if iterator is not loader:
            iterator.set_postfix(loss=f"{loss_sum/max(1,total):.4f}", acc=f"{correct/max(1,total):.4f}")
    return loss_sum / max(1, len(loader)), correct / max(1, total)

# ------------------ Main ------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--epochs", type=int, default=2, help="總訓練 epoch")
    ap.add_argument("--batch-size", type=int, default=128, help="每個 batch 的樣本數")
    ap.add_argument("--workers", type=int, default=4, help="DataLoader 背景執行緒數")
    ap.add_argument("--omp-threads", type=int, default=2, help="設定 OMP_NUM_THREADS（預設 2）")
    ap.add_argument("--debug-epoch-print", action="store_true", help="列印 epoch 開始/結束 debug 訊息")
    ap.add_argument("--profile-steps", type=int, default=16, help="每個 epoch 要 profile 的步數上限（避免檔案過大）")
    args = ap.parse_args()

    # 以 script 位置為基準
    script_dir = os.path.dirname(os.path.abspath(__file__))

    # OMP 設定
    os.environ["OMP_NUM_THREADS"] = str(args.omp_threads)

    # 路徑
    traces_dir = os.path.abspath(os.path.join(script_dir, "../data/chakra/pytorch_traces"))
    dataset_dir = os.path.abspath(os.path.join(script_dir, "../data/cifar10"))
    os.makedirs(traces_dir, exist_ok=True)
    os.makedirs(dataset_dir, exist_ok=True)

    # DDP
    rank, world, ddp_enabled = setup_ddp()
    device = torch.device(f"cuda:{int(os.environ.get('LOCAL_RANK',0))}" if ddp_enabled else "cuda")

    # 依 rank 逐一列印初始化訊息
    def _print_rank_init_info():
        prefix = f"[Rank {rank}]"
        print(f"{prefix} World size={world} | Device={device}", flush=True)
        print(f"{prefix} OMP_NUM_THREADS = {os.environ.get('OMP_NUM_THREADS')}", flush=True)
        print(f"{prefix} Dataset dir : {dataset_dir}", flush=True)
        print(f"{prefix} Traces dir  : {traces_dir}", flush=True)

    if ddp_enabled:
        for r in range(world):
            if r == rank:
                _print_rank_init_info()
            # 用最保險的 barrier 介面（避免部分版本不支援 device_ids 參數）
            dist.barrier()
    else:
        _print_rank_init_info()

    # 資料集
    tfm = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5,0.5,0.5),(0.5,0.5,0.5))
    ])
    trainset = torchvision.datasets.CIFAR10(root=dataset_dir, train=True, download=True, transform=tfm)
    sampler = torch.utils.data.distributed.DistributedSampler(trainset, num_replicas=world, rank=rank) if ddp_enabled else None
    loader  = DataLoader(trainset, batch_size=args.batch_size, shuffle=(sampler is None),
                         sampler=sampler, num_workers=args.workers, pin_memory=True, drop_last=True)

    # 模型/優化器
    model = CIFAR10_CNN().to(device)
    if ddp_enabled:
        model = DDP(model, device_ids=[device.index], output_device=device.index, broadcast_buffers=False)
        # ★ 安裝 comm hook：用 Kineto 標記 all-reduce，提升 ROCm/RCCL 事件被抓到的機率
        model.register_comm_hook(state=None, hook=make_tagging_allreduce_hook())

    optimizer = optim.AdamW(model.parameters(), lr=1e-3)
    scaler    = torch.amp.GradScaler('cuda')
    loss_fn   = nn.CrossEntropyLoss()

    # Profiler 排程：warmup 1 step + active N steps（重複 1 次），由 prof.step() 推進
    active_steps = max(1, int(args.profile_steps))
    sched = schedule(wait=0, warmup=1, active=active_steps, repeat=1)

    for epoch in range(1, args.epochs + 1):
        if ddp_enabled:
            loader.sampler.set_epoch(epoch)

        # 建立 profiler；on_trace_ready 會輸出到 rank/epoch 專屬檔
        trace_path = os.path.join(traces_dir, f"trace_rank_{rank}_epoch_{epoch}.json")
        prof = profile(
            activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
            schedule=sched,
            record_shapes=True, profile_memory=True, with_stack=True, with_flops=True,
            on_trace_ready=lambda p: p.export_chrome_trace(trace_path)
        )

        if args.debug_epoch_print and is_main(rank):
            print(f"[Debug] Begin Epoch {epoch}")

        # ----- 關鍵：在 batch 迴圈內呼叫 prof.step() -----
        model.train()
        is_rank0 = is_main(rank)
        iterator = tqdm(loader, desc=f"Epoch {epoch} [CPU+CUDA]", dynamic_ncols=True) if is_rank0 else loader

        prof.__enter__()
        loss_sum, total, correct = 0.0, 0, 0
        for step_idx, (x, y) in enumerate(iterator, start=1):
            x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast(device_type="cuda", dtype=torch.float16):
                logits = model(x)
                loss = loss_fn(logits, y)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            # ★ 每個 batch 推進一次 profiler 狀態機（必要）
            prof.step()

            # 更新即時統計
            loss_sum += loss.item()
            total += y.size(0)
            correct += (logits.argmax(1) == y).sum().item()
            if iterator is not loader:
                iterator.set_postfix(loss=f"{loss_sum/max(1,step_idx):.4f}", acc=f"{correct/max(1,total):.4f}")

            # 避免檔案過大：只記錄前 active_steps+1（含 warmup）*重複次數 的步數，也可拿掉這段
            if step_idx >= (active_steps + 1):  # +1 for warmup
                pass  # 繼續訓練，但 profiler 會自動停止收集

        prof.__exit__(None, None, None)

        # epoch summary
        epoch_loss = loss_sum / max(1, len(loader))
        epoch_acc  = correct / max(1, total)
        print(f"[Rank {rank}] [Epoch {epoch:02d}] loss={epoch_loss:.4f} acc={epoch_acc:.4f} traced->{trace_path}", flush=True)

        if args.debug_epoch_print and is_main(rank):
            print(f"[Debug] End   Epoch {epoch}")

    cleanup_ddp(ddp_enabled)

if __name__ == "__main__":
    main()
