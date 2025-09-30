#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
CIFAR-10 Training Framework for ROCm (AMD GPU) with Distributed Profiling

This module provides a comprehensive training framework for CIFAR-10 classification
using AMD ROCm GPUs with integrated performance monitoring and distributed training support.

Usage Examples:
  Single GPU:    torchrun --standalone --nproc_per_node=1 ./src/train_rocm_pytorch.py --epochs 2 --batch-size 128
  Dual GPU:      torchrun --standalone --nproc_per_node=2 ./src/train_rocm_pytorch.py --epochs 2 --batch-size 128
  Full Monitor:  torchrun --standalone --nproc_per_node=2 ./src/train_rocm_pytorch.py --epochs=10 --batch-size=512 --omp-threads=4 --debug-epoch-print
  Disable GPU:   torchrun --standalone --nproc_per_node=2 ./src/train_rocm_pytorch.py --epochs=10 --disable-gpu-monitoring

Output Files:
  - ../data/chakra/pytorch_traces/trace_rank_{rank}_epoch_{epoch}.json    # CPU+CUDA event traces
  - ../data/chakra/gpu_metrics/gpu_metrics_rank_{rank}_epoch_{epoch}.json # GPU frequency and alpha metrics

Dataset Location: ../data/cifar10

Key Features:
- Profiler lifecycle management with proper schedule handling
- GPU frequency monitoring with alpha calculation (enabled by default)
- Distributed training support with communication hooks
- Comprehensive performance metrics collection
- ROCTracer integration for detailed trace analysis
"""
import os
import argparse
import torch
import torch.nn as nn
import torch.optim as optim
import torch.distributed as dist
from torch.utils.data import DataLoader
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.profiler import profile, ProfilerActivity, schedule, record_function, ProfilerAction
from tqdm import tqdm
import torchvision
import torchvision.transforms as transforms
import time
import json
import subprocess
import threading
import statistics
from pathlib import Path

# ==================== GPU Performance Monitoring and Alpha Calculation ====================
class GPUMonitor:
    def __init__(self, device_id=0, sample_interval=0.5):
        """
        GPU frequency and performance monitoring system.

        Args:
            device_id: GPU device ID for monitoring
            sample_interval: Sampling interval in seconds
        """
        self.device_id = device_id
        self.sample_interval = sample_interval
        self.monitoring = False
        self.samples = []
        self.monitor_thread = None

    def _query_gpu_freq(self):
        """Query current GPU frequency information."""
        try:
            result = subprocess.run(
                ["rocm-smi", "--showclocks", "--json"],
                capture_output=True, text=True, check=True
            )
            data = json.loads(result.stdout)

            # Extract frequency information for corresponding GPU
            gpu_key = f"card{self.device_id}"
            if gpu_key not in data:
                return None

            gpu_info = data[gpu_key]

            # Parse frequency data (handle "(2500Mhz)" format)
            def parse_freq(freq_str):
                if isinstance(freq_str, str):
                    import re
                    match = re.search(r'(\d+(?:\.\d+)?)', freq_str.replace('(', '').replace(')', '').lower())
                    if match:
                        return float(match.group(1))
                elif isinstance(freq_str, (int, float)):
                    return float(freq_str)
                return None

            sclk = parse_freq(gpu_info.get("sclk clock speed:", "0"))
            mclk = parse_freq(gpu_info.get("mclk clock speed:", "0"))

            return {
                "timestamp": time.time(),
                "sclk_mhz": sclk,
                "mclk_mhz": mclk,
                "gpu_id": self.device_id
            }
        except Exception as e:
            print(f"[GPUMonitor] Failed to query GPU frequency: {e}")
            return None

    def _monitor_loop(self):
        """Monitoring loop running in background thread."""
        while self.monitoring:
            sample = self._query_gpu_freq()
            if sample and sample["sclk_mhz"] and sample["sclk_mhz"] > 0:
                self.samples.append(sample)
            time.sleep(self.sample_interval)

    def start_monitoring(self):
        """Start GPU monitoring."""
        if self.monitoring:
            return
        self.monitoring = True
        self.samples.clear()
        self.monitor_thread = threading.Thread(target=self._monitor_loop, daemon=True)
        self.monitor_thread.start()
        print(f"[GPUMonitor] Started monitoring GPU {self.device_id}")

    def stop_monitoring(self):
        """Stop GPU monitoring."""
        if not self.monitoring:
            return
        self.monitoring = False
        if self.monitor_thread:
            self.monitor_thread.join(timeout=2.0)
        print(f"[GPUMonitor] Stopped monitoring GPU {self.device_id}, collected {len(self.samples)} samples")

    def get_freq_stats(self):
        """Get frequency statistics from collected samples."""
        if not self.samples:
            return None

        sclk_values = [s["sclk_mhz"] for s in self.samples if s["sclk_mhz"] and s["sclk_mhz"] > 0]
        mclk_values = [s["mclk_mhz"] for s in self.samples if s["mclk_mhz"] and s["mclk_mhz"] > 0]

        if not sclk_values:
            return None

        return {
            "sclk_mhz": {
                "mean": statistics.mean(sclk_values),
                "median": statistics.median(sclk_values),
                "min": min(sclk_values),
                "max": max(sclk_values),
                "samples": len(sclk_values)
            },
            "mclk_mhz": {
                "mean": statistics.mean(mclk_values) if mclk_values else 0,
                "median": statistics.median(mclk_values) if mclk_values else 0,
                "min": min(mclk_values) if mclk_values else 0,
                "max": max(mclk_values) if mclk_values else 0,
                "samples": len(mclk_values)
            },
            "total_samples": len(self.samples),
            "duration_seconds": (self.samples[-1]["timestamp"] - self.samples[0]["timestamp"]) if len(self.samples) > 1 else 0
        }

class AlphaCalculator:
    """
    Performance alpha calculator based on measured step times and GPU frequency.

    This class provides static methods to estimate alpha values (microseconds per cycle)
    from actual training measurements, enabling accurate performance modeling.
    """

    @staticmethod
    def estimate_alpha(step_duration_ms, gpu_freq_mhz, efficiency_factor=0.3):
        """
        Estimate alpha value from step duration and GPU frequency.

        Args:
            step_duration_ms: Training step duration in milliseconds
            gpu_freq_mhz: GPU core frequency in MHz
            efficiency_factor: GPU utilization efficiency (default: 30%)

        Returns:
            float: Alpha value in microseconds per cycle, or None if invalid input
        """
        if not gpu_freq_mhz or gpu_freq_mhz <= 0:
            return None

        # Calculate theoretical cycles: frequency(MHz) * time(ms) * 1000
        theoretical_cycles = gpu_freq_mhz * step_duration_ms * 1000

        # Apply efficiency factor to get effective cycles
        effective_cycles = theoretical_cycles * efficiency_factor

        # Calculate alpha: time(microseconds) / cycles
        step_duration_us = step_duration_ms * 1000
        alpha_us = step_duration_us / effective_cycles if effective_cycles > 0 else None

        return alpha_us

    @staticmethod
    def calculate_alpha_from_measurements(step_times_ms, freq_stats, efficiency_range=(0.2, 0.5)):
        """
        Calculate comprehensive alpha metrics from measurement data.

        Args:
            step_times_ms: List of step execution times in milliseconds
            freq_stats: GPU frequency statistics dictionary
            efficiency_range: Tuple of (min, max) efficiency factors

        Returns:
            dict: Alpha values using different estimation methods, or None if invalid data
        """
        if not step_times_ms or not freq_stats or not freq_stats["sclk_mhz"]:
            return None

        median_step_ms = statistics.median(step_times_ms)
        median_freq_mhz = freq_stats["sclk_mhz"]["median"]

        results = {}

        for eff in efficiency_range:
            alpha = AlphaCalculator.estimate_alpha(median_step_ms, median_freq_mhz, eff)
            if alpha:
                results[f"alpha_us_eff_{eff:.1f}"] = alpha

        # Calculate estimations based on mean frequency
        mean_freq_mhz = freq_stats["sclk_mhz"]["mean"]
        for eff in efficiency_range:
            alpha = AlphaCalculator.estimate_alpha(median_step_ms, mean_freq_mhz, eff)
            if alpha:
                results[f"alpha_us_eff_{eff:.1f}_mean_freq"] = alpha

        # Calculate recommended value using median frequency and middle efficiency
        mid_eff = sum(efficiency_range) / 2
        recommended_alpha = AlphaCalculator.estimate_alpha(median_step_ms, median_freq_mhz, mid_eff)
        if recommended_alpha:
            results["alpha_us_recommended"] = recommended_alpha

        return results

# ==================== Neural Network Model Definition ====================
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

# ==================== Distributed Data Parallel Setup ====================
def setup_ddp():
    """Initialize distributed data parallel training if multiple GPUs are available."""
    if int(os.environ.get("WORLD_SIZE", "1")) > 1:
        # ROCm backend uses "nccl" (implemented as RCCL)
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
    """Create an allreduce communication hook with profiling tags."""
    # Use PyTorch built-in allreduce hook wrapped with record_function for tracing
    from torch.distributed.algorithms.ddp_comm_hooks import default_hooks as dh
    def hook(state, bucket):
        with record_function("nccl_all_reduce"):  # This creates identifiable comm markers in trace
            return dh.allreduce_hook(state, bucket)
    return hook

# ==================== Training Functions ====================
def train_one_epoch(model, loader, device, optimizer, scaler, loss_fn, epoch, pbar_desc=None):
    """Train the model for one epoch without profiling.

    Args:
        model: Model to train
        loader: DataLoader for training data
        device: Device to run training on
        optimizer: Optimizer for parameter updates
        scaler: Gradient scaler for mixed precision
        loss_fn: Loss function
        epoch: Current epoch number
        pbar_desc: Description for progress bar

    Returns:
        Tuple of (average_loss, accuracy)
    """
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

# ==================== Main Function ====================
def main():
    """Main training function with comprehensive profiling and monitoring capabilities."""
    ap = argparse.ArgumentParser()
    ap.add_argument("--epochs", type=int, default=2, help="Total number of training epochs")
    ap.add_argument("--batch-size", type=int, default=128, help="Number of samples per batch")
    ap.add_argument("--workers", type=int, default=4, help="Number of DataLoader background threads")
    ap.add_argument("--omp-threads", type=int, default=2, help="Set OMP_NUM_THREADS (default: 2)")
    ap.add_argument("--debug-epoch-print", action="store_true", help="Print epoch start/end debug messages")
    ap.add_argument("--profile-steps", type=int, default=16, help="Maximum steps to profile per epoch (to avoid large files)")
    ap.add_argument("--disable-gpu-monitoring", action="store_true", help="Disable GPU frequency monitoring and alpha calculation")
    ap.add_argument("--gpu-sample-interval", type=float, default=0.1, help="GPU monitoring sampling interval in seconds (default: 0.1)")
    ap.add_argument("--efficiency-min", type=float, default=0.2, help="Minimum efficiency factor for alpha calculation (default: 0.2)")
    ap.add_argument("--efficiency-max", type=float, default=0.5, help="Maximum efficiency factor for alpha calculation (default: 0.5)")
    args = ap.parse_args()

    # Use script directory as base path
    script_dir = os.path.dirname(os.path.abspath(__file__))

    # OMP configuration
    os.environ["OMP_NUM_THREADS"] = str(args.omp_threads)

    # Directory paths
    traces_dir = os.path.abspath(os.path.join(script_dir, "../data/chakra/pytorch_traces"))
    dataset_dir = os.path.abspath(os.path.join(script_dir, "../data/cifar10"))
    gpu_metrics_dir = os.path.abspath(os.path.join(script_dir, "../data/chakra/gpu_metrics"))
    os.makedirs(traces_dir, exist_ok=True)
    os.makedirs(dataset_dir, exist_ok=True)
    os.makedirs(gpu_metrics_dir, exist_ok=True)

    # Initialize distributed data parallel
    rank, world, ddp_enabled = setup_ddp()
    device = torch.device(f"cuda:{int(os.environ.get('LOCAL_RANK',0))}" if ddp_enabled else "cuda")

    # GPU monitoring setup (enabled by default unless explicitly disabled)
    gpu_monitor = None
    enable_gpu_monitoring = not args.disable_gpu_monitoring
    if enable_gpu_monitoring:
        # Get actual GPU device ID
        if ddp_enabled:
            device_id = int(os.environ.get('LOCAL_RANK', 0))
        else:
            device_id = 0
        gpu_monitor = GPUMonitor(device_id, args.gpu_sample_interval)

    # Print initialization information by rank
    def _print_rank_init_info():
        prefix = f"[Rank {rank}]"
        print(f"{prefix} World size={world} | Device={device}", flush=True)
        print(f"{prefix} OMP_NUM_THREADS = {os.environ.get('OMP_NUM_THREADS')}", flush=True)
        print(f"{prefix} Dataset dir : {dataset_dir}", flush=True)
        print(f"{prefix} Traces dir  : {traces_dir}", flush=True)
        print(f"{prefix} GPU Metrics : {gpu_metrics_dir}", flush=True)
        print(f"{prefix} GPU Monitor : {'Enabled' if enable_gpu_monitoring else 'Disabled'}", flush=True)

    if ddp_enabled:
        for r in range(world):
            if r == rank:
                _print_rank_init_info()
            # Use the most reliable barrier interface (avoiding device_ids parameter for compatibility)
            dist.barrier()
    else:
        _print_rank_init_info()

    # Dataset setup
    tfm = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5,0.5,0.5),(0.5,0.5,0.5))
    ])
    trainset = torchvision.datasets.CIFAR10(root=dataset_dir, train=True, download=True, transform=tfm)
    sampler = torch.utils.data.distributed.DistributedSampler(trainset, num_replicas=world, rank=rank) if ddp_enabled else None
    loader  = DataLoader(trainset, batch_size=args.batch_size, shuffle=(sampler is None),
                         sampler=sampler, num_workers=args.workers, pin_memory=True, drop_last=True)

    # Model and optimizer setup
    model = CIFAR10_CNN().to(device)
    if ddp_enabled:
        model = DDP(model, device_ids=[device.index], output_device=device.index, broadcast_buffers=False)
        # Install communication hook: Use Kineto tags for all-reduce, improving ROCm/RCCL event capture
        model.register_comm_hook(state=None, hook=make_tagging_allreduce_hook())

    optimizer = optim.AdamW(model.parameters(), lr=1e-3)
    scaler    = torch.amp.GradScaler('cuda')
    loss_fn   = nn.CrossEntropyLoss()

    # Important: Don't create schedule here to avoid cross-epoch reuse of the same schedule object

    for epoch in range(1, args.epochs + 1):
        if ddp_enabled:
            loader.sampler.set_epoch(epoch)

        # Create new schedule for each epoch to avoid residual state
        active_steps = max(1, min(int(args.profile_steps), len(loader)))
        sched = schedule(wait=0, warmup=1, active=active_steps, repeat=1)

        # Setup profiler output paths for rank/epoch specific files
        trace_path = os.path.join(traces_dir, f"trace_rank_{rank}_epoch_{epoch}.json")
        gpu_metrics_path = os.path.join(gpu_metrics_dir, f"gpu_metrics_rank_{rank}_epoch_{epoch}.json")

        if args.debug_epoch_print and is_main(rank):
            print(f"[Debug] Begin Epoch {epoch}")

        # Start GPU monitoring
        if gpu_monitor:
            gpu_monitor.start_monitoring()

        # Record step timing
        step_times_ms = []
        epoch_start_time = time.time()

        # Progress bar: For complete balance, consider not showing for any rank, or redirect to files
        model.train()
        is_rank0 = is_main(rank)
        # Option 1: Complete balance (no progress bars for any rank)
        # iterator = loader
        # Option 2: Current setup (only rank 0 shows progress, minimal CPU overhead difference)
        iterator = tqdm(loader, desc=f"Epoch {epoch} [CPU+CUDA]", dynamic_ncols=True) if is_rank0 else loader

        # Use context manager for profiler lifecycle and only step when profiler is active
        with profile(
            activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
            schedule=sched,
            record_shapes=True, profile_memory=True, with_stack=True, with_flops=True,
            on_trace_ready=lambda p: p.export_chrome_trace(trace_path)
        ) as prof:

            loss_sum, total, correct = 0.0, 0, 0
            for step_idx, (x, y) in enumerate(iterator, start=1):
                step_start_time = time.time()

                x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)
                optimizer.zero_grad(set_to_none=True)
                with torch.autocast(device_type="cuda", dtype=torch.float16):
                    logits = model(x)
                    loss = loss_fn(logits, y)
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()

                # Only call profiler step when profiler is still active (avoid STOP then __exit__)
                if getattr(prof, "current_action", ProfilerAction.NONE) != ProfilerAction.NONE:
                    prof.step()

                # Record step timing
                step_end_time = time.time()
                step_duration_ms = (step_end_time - step_start_time) * 1000
                step_times_ms.append(step_duration_ms)

                # Update real-time statistics (show average loss/accuracy)
                loss_sum += loss.item()
                total += y.size(0)
                correct += (logits.argmax(1) == y).sum().item()
                if iterator is not loader:
                    iterator.set_postfix(loss=f"{loss_sum/max(1,step_idx):.4f}",
                                         acc=f"{correct/max(1,total):.4f}",
                                         step_ms=f"{step_duration_ms:.1f}")

                # No manual limit: schedule will automatically stop collection; subsequent steps blocked above

        # No manual prof.__exit__(); with block will safely close on exit

        # Stop GPU monitoring and calculate alpha
        if gpu_monitor:
            gpu_monitor.stop_monitoring()
            freq_stats = gpu_monitor.get_freq_stats()

            # Calculate alpha values
            alpha_results = None
            if freq_stats and step_times_ms:
                alpha_results = AlphaCalculator.calculate_alpha_from_measurements(
                    step_times_ms, freq_stats, (args.efficiency_min, args.efficiency_max)
                )

            # Save GPU metrics and alpha results
            gpu_metrics_data = {
                "epoch": epoch,
                "rank": rank,
                "device_id": device_id if gpu_monitor else (int(os.environ.get('LOCAL_RANK', 0)) if ddp_enabled else 0),
                "timestamp": time.time(),
                "epoch_duration_seconds": time.time() - epoch_start_time,
                "step_count": len(step_times_ms),
                "step_times_ms": {
                    "mean": statistics.mean(step_times_ms) if step_times_ms else 0,
                    "median": statistics.median(step_times_ms) if step_times_ms else 0,
                    "min": min(step_times_ms) if step_times_ms else 0,
                    "max": max(step_times_ms) if step_times_ms else 0,
                    "samples": len(step_times_ms)
                },
                "gpu_frequency_stats": freq_stats,
                "alpha_calculations": alpha_results,
                "training_config": {
                    "batch_size": args.batch_size,
                    "world_size": world,
                    "efficiency_range": [args.efficiency_min, args.efficiency_max]
                }
            }

            # Save to file
            try:
                with open(gpu_metrics_path, 'w') as f:
                    json.dump(gpu_metrics_data, f, indent=2)
                print(f"[Rank {rank}] GPU metrics saved to: {gpu_metrics_path}")

                # Print recommended alpha value
                if alpha_results and "alpha_us_recommended" in alpha_results:
                    recommended_alpha = alpha_results["alpha_us_recommended"]
                    print(f"[Rank {rank}] Recommended Alpha: {recommended_alpha:.6f} μs/cycle")
                    if freq_stats:
                        median_freq = freq_stats["sclk_mhz"]["median"]
                        print(f"[Rank {rank}] Based on GPU frequency: {median_freq:.1f} MHz")

            except Exception as e:
                print(f"[Rank {rank}] Failed to save GPU metrics: {e}")

        if ddp_enabled:
            dist.barrier()  # Synchronize all ranks to avoid interleaved progress/output

        # Epoch summary
        epoch_loss = loss_sum / max(1, len(loader))
        epoch_acc  = correct / max(1, total)
        print(f"[Rank {rank}] [Epoch {epoch:02d}] loss={epoch_loss:.4f} acc={epoch_acc:.4f} traced->{trace_path}", flush=True)

        if args.debug_epoch_print and is_main(rank):
            print(f"[Debug] End   Epoch {epoch}")

    cleanup_ddp(ddp_enabled)

if __name__ == "__main__":
    main()
