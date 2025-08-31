#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
CIFAR-10 training on ROCm (AMD GPU) with automatic Chrome trace export.
- 單卡: torchrun --standalone --nproc_per_node=1 train_rocm_pytorch.py --epochs 2 --batch-size 128
- 多卡: torchrun --standalone --nproc_per_node=2 train_rocm_pytorch.py --epochs 2 --batch-size 128
Trace 會輸出到 (以 script 位置為基準): ../data/chakra/pytorch_traces/trace.json
Dataset 會存放在 (以 script 位置為基準): ../data/cifar10
"""

import os
import argparse
import torch
import torch.nn as nn
import torch.optim as optim
import torch.distributed as dist
from torch.utils.data import DataLoader
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.profiler import profile, ProfilerActivity, schedule
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
        dist.barrier()
        dist.destroy_process_group()

def is_main(rank): return rank == 0

# ------------------ 訓練 ------------------
def train_one_epoch(model, loader, device, optimizer, scaler, loss_fn, epoch, prof=None):
    model.train()
    total, correct, running = 0, 0, 0.0
    pbar = tqdm(loader, desc=f"Epoch {epoch}", dynamic_ncols=True) if is_main(int(os.environ.get("RANK", "0"))) else loader
    for x, y in pbar:
        x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)
        optimizer.zero_grad(set_to_none=True)
        with torch.autocast(device_type="cuda", dtype=torch.float16):
            logits = model(x)
            loss = loss_fn(logits, y)
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        running += loss.item()
        total += y.size(0)
        correct += (logits.argmax(1) == y).sum().item()
        if prof is not None:
            prof.step()  # 告訴 profiler 完成一個「步」
        if pbar is not loader:  # 只有 main rank 有 tqdm
            pbar.set_postfix(loss=f"{running/total:.4f}", acc=f"{correct/total:.4f}")
    return running / max(1, len(loader)), correct / max(1, total)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=10, help="訓練回合數")
    parser.add_argument("--batch-size", type=int, default=128, help="每個 batch 的樣本數")
    parser.add_argument("--workers", type=int, default=4, help="DataLoader 背景執行緒數")
    args = parser.parse_args()

    # 以 script 位置為基準的路徑
    script_dir = os.path.dirname(os.path.abspath(__file__))

    # ---- trace 目錄 ----
    trace_dir = os.path.abspath(os.path.join(script_dir, "../data/chakra/pytorch_traces"))
    os.makedirs(trace_dir, exist_ok=True)
    trace_path = os.path.join(trace_dir, "trace.json")

    # ---- CIFAR-10 資料集目錄 ----
    dataset_dir = os.path.abspath(os.path.join(script_dir, "../data/cifar10"))
    os.makedirs(dataset_dir, exist_ok=True)

    rank, world, ddp_enabled = setup_ddp()
    device = torch.device(f"cuda:{int(os.environ.get('LOCAL_RANK',0))}" if ddp_enabled else "cuda")
    if is_main(rank):
        print(f"[Info] Using device {device}, World size={world}")
        print(f"[Info] Dataset dir: {dataset_dir}")
        print(f"[Info] Trace path : {trace_path}")

    # 資料集
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5,0.5,0.5), (0.5,0.5,0.5))
    ])
    trainset = torchvision.datasets.CIFAR10(root=dataset_dir, train=True, download=True, transform=transform)
    sampler = torch.utils.data.distributed.DistributedSampler(trainset, num_replicas=world, rank=rank) if ddp_enabled else None
    trainloader = DataLoader(trainset, batch_size=args.batch_size, shuffle=(sampler is None),
                             sampler=sampler, num_workers=args.workers, pin_memory=True, drop_last=True)

    # 模型/優化器
    model = CIFAR10_CNN().to(device)
    if ddp_enabled:
        model = DDP(model, device_ids=[device.index], output_device=device.index, broadcast_buffers=False)
    optimizer = optim.AdamW(model.parameters(), lr=1e-3)
    scaler = torch.amp.GradScaler('cuda')  # 新 API
    loss_fn = nn.CrossEntropyLoss()

    # ---- 正確使用 profiler：用 with + schedule + 每步 prof.step() ----
    # 這裡設定：wait=0, warmup=1, active=1, repeat=1 -> 會收集第 1 個 epoch（常用情境足夠）
    prof_sched = schedule(wait=0, warmup=1, active=1, repeat=1)
    with profile(
        activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
        schedule=prof_sched,
        record_shapes=True,
        profile_memory=True,
        with_stack=True,
        on_trace_ready=lambda p: p.export_chrome_trace(trace_path)  # 收集完成時自動輸出
    ) as prof:

        for epoch in range(1, args.epochs + 1):
            if ddp_enabled:
                trainloader.sampler.set_epoch(epoch)
            loss, acc = train_one_epoch(model, trainloader, device, optimizer, scaler, loss_fn, epoch, prof)
            if is_main(rank):
                print(f"[Epoch {epoch:02d}] loss={loss:.4f} acc={acc:.4f}")

    # profiler 在 with 之後已安全關閉，避免 SIGSEGV
    cleanup_ddp(ddp_enabled)

if __name__ == "__main__":
    main()
