
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Demo: Collect Chakra Execution Trace (ET) + Kineto device trace for ResNet-50 (DDP, 2 GPUs).

- Saves traces to:
    /workspace/data/chakra/pytorch_traces/host_[rank].json
    /workspace/data/chakra/pytorch_traces/device_[rank].json

- Also drops lightweight GPU metric snapshots to:
    /workspace/data/chakra/gpu_metrics/gpu_metrics_rank_[rank]_chakra_epoch_1.json

- Logs:
    /workspace/data/chakra/log/log.log
    /workspace/data/chakra/log/err.log

Notes for ROCm/AMD:
- Use PyTorch 2.5.0+ (ROCm wheels). The 'nccl' backend maps to RCCL on AMD.
- torch.profiler.ProfilerActivity.CUDA covers ROCm GPUs as well; if unavailable, we fall back to CPU-only ET.

Run:
  python /workspace/src/offical_demo/demo_chakra_trace.py --world-size 2 --use-gpu

"""
import argparse
import json
import logging
import os
import sys
from pathlib import Path
from datetime import datetime

import torch
import torch.distributed as dist
import torch.multiprocessing as mp
import torch.nn as nn
import torch.optim
import torch.utils.data
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.profiler import (
    ExecutionTraceObserver,
    ProfilerActivity,
    profile,
    kineto_available,
    supported_activities,
)
from torch._C._profiler import _ExperimentalConfig

import torchvision
import torchvision.transforms as T
from torchvision import models

# ------------------------------
# Globals & paths
# ------------------------------
DATA_ROOT = Path("/workspace/data")
CHAKRA_BASE = DATA_ROOT / "chakra"
PYTORCH_TRACES_DIR = CHAKRA_BASE / "pytorch_traces"
GPU_METRICS_DIR = CHAKRA_BASE / "gpu_metrics"
LOG_DIR = CHAKRA_BASE / "log"
WORKLOAD_ET_DIR = CHAKRA_BASE / "workload_et"  # (created for later ASTRA-sim use)
CIFAR_ROOT = DATA_ROOT / "cifar10"

for d in [PYTORCH_TRACES_DIR, GPU_METRICS_DIR, LOG_DIR, WORKLOAD_ET_DIR, CIFAR_ROOT]:
    d.mkdir(parents=True, exist_ok=True)

# logging
LOG_FILE = LOG_DIR / "log.log"
ERR_FILE = LOG_DIR / "err.log"
_logger = logging.getLogger("demo_chakra_trace")
_logger.setLevel(logging.INFO)
_fmt = logging.Formatter("[%(asctime)s][%(levelname)s][rank=%(rank)s] %(message)s")
class RankFilter(logging.Filter):
    def __init__(self, rank):
        super().__init__()
        self.rank = rank
    def filter(self, record):
        record.rank = self.rank
        return True

# ------------------------------
# Utilities
# ------------------------------
def _build_transforms():
    # Keep it simple & deterministic
    return T.Compose([
        T.Resize(256),
        T.CenterCrop(224),
        T.ToTensor(),
    ])

def _make_dataloader(rank, world_size, batch_size=32, num_workers=4):
    transform = _build_transforms()
    # Put CIFAR under /workspace/data/cifar10 as requested
    trainset = torchvision.datasets.CIFAR10(
        root=str(CIFAR_ROOT), train=True, download=True, transform=transform
    )
    sampler = torch.utils.data.distributed.DistributedSampler(
        trainset, num_replicas=world_size, rank=rank, shuffle=False, drop_last=False
    )
    loader = torch.utils.data.DataLoader(
        trainset, batch_size=batch_size, sampler=sampler,
        shuffle=False, num_workers=num_workers, pin_memory=False, persistent_workers=False
    )
    return loader

def _select_activities():
    # Try to include GPU activity if available/kineto-ready; otherwise CPU only
    acts = set([ProfilerActivity.CPU])
    try:
        # In ROCm builds, CUDA activity name is reused
        if kineto_available():
            acts |= {ProfilerActivity.CUDA}
    except Exception:
        pass
    return list(acts)

def _write_gpu_metrics(rank, device):
    try:
        metrics = {
            "timestamp": datetime.utcnow().isoformat() + "Z",
            "rank": rank,
            "device": str(device),
            "cuda_is_available": torch.cuda.is_available(),
            "memory_allocated": int(torch.cuda.memory_allocated(rank)) if torch.cuda.is_available() else 0,
            "max_memory_allocated": int(torch.cuda.max_memory_allocated(rank)) if torch.cuda.is_available() else 0,
        }
    except Exception as e:
        metrics = {"timestamp": datetime.utcnow().isoformat()+"Z", "rank": rank, "error": str(e)}

    outp = GPU_METRICS_DIR / f"gpu_metrics_rank_{rank}_chakra_epoch_1.json"
    with open(outp, "w", encoding="utf-8") as f:
        json.dump(metrics, f, ensure_ascii=False, indent=2)

# ------------------------------
# Core training + tracing
# ------------------------------
def run_worker(rank, world_size, use_gpu=True, batch_size=32):
    # Set up per-rank logging
    fh = logging.FileHandler(LOG_FILE, mode="a", encoding="utf-8")
    eh = logging.FileHandler(ERR_FILE, mode="a", encoding="utf-8")
    fh.setFormatter(_fmt); eh.setFormatter(_fmt)
    fh.addFilter(RankFilter(rank)); eh.addFilter(RankFilter(rank))
    _logger.addHandler(fh); _logger.addHandler(eh)

    _logger.info(f"Starting rank {rank}/{world_size} | use_gpu={use_gpu}")
    device = None
    if use_gpu and torch.cuda.is_available():
        device = torch.device(f"cuda:{rank}")
        torch.cuda.set_device(rank)
    else:
        device = torch.device("cpu")
        use_gpu = False

    # Model
    try:
        model = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V1)
    except Exception:
        # Fallback to random init if weights can't be fetched
        model = models.resnet50(weights=None)

    if use_gpu:
        model = model.to(device)
    model.train()

    # DDP wrap
    model = DDP(model, device_ids=[rank]) if use_gpu else DDP(model)

    # Data
    loader = _make_dataloader(rank, world_size, batch_size=batch_size, num_workers=4)

    # Loss/opt
    criterion = nn.CrossEntropyLoss().to(device) if use_gpu else nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=0.001, momentum=0.9)

    # Trace file paths
    host_trace = PYTORCH_TRACES_DIR / f"host_{rank}.json"
    device_trace = PYTORCH_TRACES_DIR / f"device_{rank}.json"

    # Profiler setup
    acts = _select_activities()
    _logger.info(f"supported activities (selected): {acts}")
    eg_obs = ExecutionTraceObserver().register_callback(str(host_trace))

    try:
        with profile(
            activities=acts,
            schedule=torch.profiler.schedule(wait=0, warmup=0, active=1, repeat=1),
            execution_trace_observer=eg_obs,
            record_shapes=True,
            experimental_config=_ExperimentalConfig(enable_cuda_sync_events=True),
        ) as prof:
            # one short step as in the example
            for step, (inputs, labels) in enumerate(loader):
                _logger.info(f"step={step}")
                if use_gpu:
                    inputs = inputs.to(device, non_blocking=True)
                    labels = labels.to(device, non_blocking=True)

                outputs = model(inputs)
                loss = criterion(outputs, labels)
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                optimizer.step()

                prof.step()
                # keep exactly one active step
                break
        # Export device-side Kineto/Chrome trace
        try:
            prof.export_chrome_trace(str(device_trace))
        except Exception as e:
            _logger.error(f"export_chrome_trace failed on rank {rank}: {e}")

    except Exception as e:
        _logger.exception(f"Profiler failed on rank {rank}: {e}")

    # GPU metrics snapshot
    _write_gpu_metrics(rank, device)
    _logger.info(f"Rank {rank} done. host={host_trace}, device={device_trace}")

def init_and_run(rank, world_size, use_gpu):
    # Minimal DDP init (nccl maps to RCCL on ROCm/AMD)
    backend = "nccl" if use_gpu and torch.cuda.is_available() else "gloo"
    os.environ.setdefault("MASTER_ADDR", "127.0.0.1")
    os.environ.setdefault("MASTER_PORT", "29501")
    dist.init_process_group(backend=backend, rank=rank, world_size=world_size)
    try:
        run_worker(rank, world_size, use_gpu=use_gpu, batch_size=32)
    finally:
        dist.destroy_process_group()

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--world-size", type=int, default=2, help="number of processes/ranks")
    ap.add_argument("--use-gpu", action="store_true", default=torch.cuda.is_available(), help="use GPUs (ROCm/CUDA) if available")
    args = ap.parse_args()

    # Spawn ranks
    mp.set_start_method("spawn", force=True)
    procs = []
    for r in range(args.world_size):
        p = mp.Process(target=init_and_run, args=(r, args.world_size, args.use_gpu))
        p.start()
        procs.append(p)
    for p in procs:
        p.join()

if __name__ == "__main__":
    main()
