#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Qwen2.5-1.5B Tensor Parallel (TP=2) Training with Kineto Profiler
for Trace Collection on AMD ROCm

=== 目的 ===
收集 Transformer 模型的真實 Tensor Parallel 通訊 trace。
使用 PyTorch 原生 parallelize_module API 對 Qwen2.5-1.5B 做 TP=2，
每個 Transformer Block 的 Linear 層被 ColwiseParallel / RowwiseParallel 切開，
產生真實的 AllGather 和 ReduceScatter RCCL 呼叫。

=== 為什麼用 Qwen 1.5B ===
- 單卡 FP32 訓練需要 ~24 GB → 超過 RX 9070 XT 的 16 GB → 必須用 TP
- TP=2 後每卡 ~12-14 GB → 剛好塞進 16 GB
- 這不是人為製造的場景，而是硬體限制下的必然選擇

=== TP 通訊 Pattern（Megatron-style）===
每個 Transformer Block (共 28 層):
  Attention:
    q/k/v_proj (ColwiseParallel) → o_proj (RowwiseParallel) → AllReduce/ReduceScatter
  MLP:
    gate/up_proj (ColwiseParallel) → down_proj (RowwiseParallel) → AllReduce/ReduceScatter
  = 2 次 collective per block

  Forward:  28 blocks × 2 = 56 collectives
  Backward: 56 collectives
  Total:    ~112 COMM ops per step

=== 記憶體估算 (FP16 loading, TP=2, 每卡) ===
  FP16 weights:     750M × 2 bytes = 1.5 GB
  FP32 AdamW master + m + v: 750M × 4 × 3 = 9 GB
  FP16 gradients:   750M × 2 bytes = 1.5 GB
  Activations + overhead:    ~2-3 GB
  Total: ~14-15 GB (fits in 16 GB)

=== 與 DDP 腳本的差異 ===
  - 模型: Qwen2.5-1.5B (vs DDP 的 0.5B)
  - 並行: TP via parallelize_module (vs DDP 的 gradient AllReduce)
  - 資料: 無 DistributedSampler (TP 下兩張 GPU 看相同資料)
  - DDP AllReduce 在後續 add_ddp_to_tp_et.py 中手動加入 ET

=== 輸出 ===
  ./data/chakra/pytorch_traces/host_{rank}_qwen15b_tp.json
  ./data/chakra/pytorch_traces/device_{rank}_qwen15b_tp.json
  ./data/chakra/gpu_metrics/gpu_metrics_{rank}_qwen15b_tp.json

=== 用法 ===
  torchrun --standalone --nproc_per_node=2 ./src/train_rocm_tensor.py \
    --epochs 3 --batch-size 1 --workers 0 \
    --seq-len 256 \
    --trace-wait 10 --trace-steps 2 \
    --inject-sync-hack

=== 後續步驟 ===
  1. python src/conver_to_chakra_et.py --model-tag qwen15b_tp --default-gpu-freq 2935
  2. python src/add_ddp_to_tp_et.py   (將 DDP AllReduce 加入 ET)
  3. python src/run_ns3.py ...         (ASTRA-sim 模擬)
"""

import os
import sys
import argparse
import time
import json
import statistics
import re
from pathlib import Path
import threading

import torch
import torch.nn as nn
import torch.optim as optim
import torch.distributed as dist

from torch.utils.data import DataLoader
from torch.profiler import (
    profile, ProfilerActivity, record_function,
    schedule, supported_activities,
)
from torch.profiler import ExecutionTraceObserver
from torch._C._profiler import _ExperimentalConfig

from rocm_compat import ROCmCompat

# --- Transformers ---
try:
    from transformers import AutoModelForCausalLM, AutoConfig
    HAS_TRANSFORMERS = True
except ImportError:
    HAS_TRANSFORMERS = False
    print("[FATAL] transformers not installed.")
    print("  pip install transformers accelerate sentencepiece --break-system-packages")

# --- PyTorch TP API ---
try:
    from torch.distributed.tensor.parallel import (
        parallelize_module,
        ColwiseParallel,
        RowwiseParallel,
    )
    from torch.distributed.device_mesh import init_device_mesh
    HAS_TP_API = True
except ImportError:
    HAS_TP_API = False
    print("[FATAL] PyTorch TP API not available. Requires PyTorch >= 2.1")


# =====================================================================
# Log (same as DDP version)
# =====================================================================

class Logger(object):
    """將輸出同時導向終端機與檔案 (Tee)"""
    def __init__(self, filename):
        self.terminal = sys.stdout
        self.log = open(filename, "w", encoding="utf-8")
    def write(self, message):
        self.terminal.write(message)
        self.log.write(message)
        self.log.flush()
    def flush(self):
        self.terminal.flush()
        self.log.flush()


# =====================================================================
# Shared Utilities (from DDP version)
# =====================================================================

def _wait_file_stable(p: Path, tries: int = 10, sleep_s: float = 0.05) -> None:
    last = -1
    for _ in range(tries):
        sz = p.stat().st_size if p.exists() else -1
        if sz == last and sz > 0:
            return
        last = sz
        time.sleep(sleep_s)


def _repair_host_json(host_path: Path) -> bool:
    try:
        txt = host_path.read_text(encoding="utf-8", errors="ignore")
        try:
            json.loads(txt)
            return False
        except Exception:
            pass
        iters = list(re.finditer(r'\{\s*"schema"\s*:', txt))
        if not iters:
            return False
        candidates = []
        for i, m in enumerate(iters):
            s = m.start()
            e = iters[i+1].start() if i+1 < len(iters) else len(txt)
            chunk = txt[s:e].lstrip()
            try:
                obj = json.loads(chunk)
                nodes = obj.get("nodes", [])
                n_cnt = len(nodes) if isinstance(nodes, list) else 0
                st_ts = obj.get("start_ts", -1)
                candidates.append((n_cnt, st_ts, chunk))
            except Exception:
                continue
        if not candidates:
            return False
        candidates.sort(key=lambda x: (x[0], x[1]))
        best = candidates[-1][2]
        host_path.write_text(best, encoding="utf-8")
        print(f"[repair] host trace fixed -> {host_path}")
        return True
    except Exception as e:
        print(f"[repair] host trace repair failed: {e}")
        return False


def _check_device_sync_and_steps(device_json: Path) -> None:
    try:
        txt = device_json.read_text(encoding="utf-8", errors="ignore").lower()
        has_rec  = bool(re.search(r'(hip|cuda)eventrecord', txt))
        has_wait = bool(re.search(r'(hip|cuda)streamwaitevent', txt))
        has_step = bool(re.search(r'profilerstep#', txt))
        has_ag   = bool(re.search(r'nccl:all_gather|all_gather', txt))
        has_rs   = bool(re.search(r'nccl:reduce_scatter|reduce_scatter', txt))
        has_ar   = bool(re.search(r'nccl:all_reduce|all_reduce', txt))
        print(
            f"[check] {device_json.name}: "
            f"sync=({'OK' if (has_rec and has_wait) else 'MISS'}) "
            f"steps={'OK' if has_step else 'MISS'} "
            f"AG={'OK' if has_ag else 'MISS'} "
            f"RS={'OK' if has_rs else 'MISS'} "
            f"AR={'OK' if has_ar else 'MISS'}"
        )
    except Exception as e:
        print(f"[check] skip: {e}")


# =====================================================================
# GPU Monitor (from DDP version)
# =====================================================================

class GPUMonitor:
    def __init__(self, device_id=0, sample_interval=0.1,
                 output_dir=None, rank=0, model_name="unknown"):
        self.device_id = device_id
        self.sample_interval = sample_interval
        self.output_dir = output_dir
        self.rank = rank
        self.model_name = model_name
        self.monitoring = False
        self.samples = []
        self._th = None
        self._compat = ROCmCompat()
        if self._compat.is_available():
            vi = self._compat.get_version_info()
            print(f"[GPUMonitor] Using {vi.tool_type.value} (version: {vi.tool_version})")
        else:
            print("[GPUMonitor] Warning: No ROCm monitoring tools available")

    def _poll(self):
        while self.monitoring:
            try:
                fi = self._compat.get_gpu_frequency(self.device_id)
                if fi and fi.sclk_mhz is not None and fi.sclk_mhz > 0:
                    self.samples.append(fi.sclk_mhz)
            except Exception:
                pass
            time.sleep(self.sample_interval)

    def start(self):
        if self.monitoring:
            return
        self.monitoring = True
        self.samples.clear()
        self._th = threading.Thread(target=self._poll, daemon=True)
        self._th.start()

    def stop(self):
        if not self.monitoring:
            return
        self.monitoring = False
        if self._th:
            self._th.join(timeout=2.0)

        if self.samples:
            avg_freq = statistics.mean(self.samples)
            median_freq = statistics.median(self.samples)
            max_freq = max(self.samples)

            metrics = {
                "rank": self.rank,
                "device_id": self.device_id,
                "sclk_avg_mhz": avg_freq,
                "sclk_median_mhz": median_freq,
                "sclk_max_mhz": max_freq,
                "samples_count": len(self.samples),
            }
            print(f"[GPUMonitor] Rank {self.rank}: Median={median_freq:.1f} MHz, Max={max_freq:.1f} MHz")

            if self.output_dir:
                out_path = Path(self.output_dir) / f"gpu_metrics_{self.rank}_{self.model_name}.json"
                try:
                    with out_path.open("w", encoding="utf-8") as f:
                        json.dump(metrics, f, indent=2)
                    print(f"[GPUMonitor] Saved to {out_path}")
                except Exception as e:
                    print(f"[GPUMonitor] Save failed: {e}")
        else:
            print(f"[GPUMonitor] No samples collected for Rank {self.rank}")


# =====================================================================
# Dummy Dataset (same as DDP version)
# =====================================================================

class DummyLLMDataset(torch.utils.data.Dataset):
    """隨機 token dataset — trace collection 不需要真實文本。"""
    def __init__(self, vocab_size=32000, seq_len=256, num_samples=1000):
        self.vocab_size = vocab_size
        self.seq_len = seq_len
        self.num_samples = num_samples

    def __len__(self):
        return self.num_samples

    def __getitem__(self, idx):
        input_ids = torch.randint(0, self.vocab_size, (self.seq_len,))
        return input_ids, input_ids.clone()


# =====================================================================
# TP Plan for Qwen2.5
# =====================================================================

def build_qwen_tp_plan(num_layers: int) -> dict:
    """
    Build Megatron-style TP plan for Qwen2.5 architecture.

    Each Transformer Block has:
      Attention: q/k/v_proj (ColwiseParallel) → o_proj (RowwiseParallel)
      MLP:       gate/up_proj (ColwiseParallel) → down_proj (RowwiseParallel)

    Communication pattern per block (TP=2):
      Forward:  2 collectives (ReduceScatter or AllReduce at each RowwiseParallel)
      Backward: 2 collectives (AllGather at each RowwiseParallel)

    Total for 28 layers: 56 collectives in forward + 56 in backward = 112 per step
    """
    plan = {}
    for i in range(num_layers):
        prefix = f"model.layers.{i}"

        # Attention: split heads across GPUs
        plan[f"{prefix}.self_attn.q_proj"] = ColwiseParallel()
        plan[f"{prefix}.self_attn.k_proj"] = ColwiseParallel()
        plan[f"{prefix}.self_attn.v_proj"] = ColwiseParallel()
        plan[f"{prefix}.self_attn.o_proj"] = RowwiseParallel()

        # MLP: split intermediate dim across GPUs
        plan[f"{prefix}.mlp.gate_proj"] = ColwiseParallel()
        plan[f"{prefix}.mlp.up_proj"]   = ColwiseParallel()
        plan[f"{prefix}.mlp.down_proj"] = RowwiseParallel()

    print(f"[TP Plan] {num_layers} layers × 7 Linear = {num_layers * 7} parallelized layers")
    print(f"[TP Plan] Expected COMM per step: ~{num_layers * 2 * 2} "
          f"({num_layers * 2} fwd + {num_layers * 2} bwd)")

    return plan


# =====================================================================
# Process Group Setup
# =====================================================================

def setup_tp():
    """Initialize process group for Tensor Parallel (TP=2)."""
    world = int(os.environ.get("WORLD_SIZE", "1"))
    if world < 2:
        raise RuntimeError(
            "TP requires at least 2 GPUs.\n"
            "Usage: torchrun --standalone --nproc_per_node=2 ./src/train_rocm_tensor.py"
        )

    local_rank = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(local_rank)
    dev = torch.device(f"cuda:{local_rank}")

    try:
        dist.init_process_group(backend="nccl", init_method="env://", device_id=dev)
    except TypeError:
        dist.init_process_group(backend="nccl", init_method="env://")

    rank = int(os.environ["RANK"])
    return rank, world


def cleanup_tp():
    if dist.is_initialized():
        try:
            dist.barrier()
        finally:
            dist.destroy_process_group()


# =====================================================================
# Training Step
# =====================================================================

def train_step(model, x, y, optimiz, device):
    """Single forward + backward + optimizer step (no GradScaler for FP16 model)."""
    x = x.to(device, non_blocking=True)
    y = y.to(device, non_blocking=True)
    optimiz.zero_grad(set_to_none=True)
    with torch.autocast(device_type="cuda", dtype=torch.float16):
        outputs = model(input_ids=x, labels=y)
        loss = outputs.loss
    loss.backward()
    optimiz.step()
    return loss.item()


# =====================================================================
# Main
# =====================================================================

def main():
    assert HAS_TRANSFORMERS, "pip install transformers accelerate sentencepiece --break-system-packages"
    assert HAS_TP_API, "PyTorch >= 2.1 required for TP API"

    ap = argparse.ArgumentParser(
        description="Qwen2.5-1.5B TP=2 Training for Trace Collection (AMD ROCm)"
    )
    # Training
    ap.add_argument("--epochs", type=int, default=3)
    ap.add_argument("--batch-size", type=int, default=1,
                    help="Batch size per GPU (default 1 to fit in 16 GB)")
    ap.add_argument("--seq-len", type=int, default=256,
                    help="Sequence length (default 256)")
    ap.add_argument("--workers", type=int, default=0)
    ap.add_argument("--omp-threads", type=int, default=2)
    ap.add_argument("--debug-epoch-print", action="store_true")

    # Model
    ap.add_argument("--model-name", type=str, default="Qwen/Qwen2.5-1.5B",
                    help="HuggingFace model name (default: Qwen/Qwen2.5-1.5B)")
    ap.add_argument("--model-dtype", type=str, default="float16",
                    choices=["float16", "float32"],
                    help="Model loading dtype (float16 recommended to fit 16 GB)")

    # Profiler
    ap.add_argument("--profile-epoch", type=int, default=2)
    ap.add_argument("--trace-wait", type=int, default=10,
                    help="Skip N steps before profiling (default 10, LLM is slower)")
    ap.add_argument("--trace-steps", type=int, default=2)
    ap.add_argument("--trace-shapes", action="store_true")
    ap.add_argument("--trace-stack", action="store_true")
    ap.add_argument("--trace-mem", action="store_true")

    # Monitoring & engineering
    ap.add_argument("--disable-gpu-monitoring", action="store_true")
    ap.add_argument("--gpu-sample-interval", type=float, default=0.01)
    ap.add_argument("--no-cleanup", action="store_true")
    ap.add_argument("--inject-sync-hack", action="store_true",
                    help="Inject sync events for trace linker stability")

    args = ap.parse_args()
    os.environ["OMP_NUM_THREADS"] = str(args.omp_threads)

    MODEL_TAG = "qwen15b_tp"

    # ---- Paths ----
    script_dir = os.path.dirname(os.path.abspath(__file__))
    base_dir = os.path.abspath(os.path.join(script_dir, "../data/chakra"))
    traces_dir = Path(base_dir) / "pytorch_traces"
    metrics_dir = Path(base_dir) / "gpu_metrics"
    log_dir = Path(base_dir) / "log"
    traces_dir.mkdir(parents=True, exist_ok=True)
    metrics_dir.mkdir(parents=True, exist_ok=True)
    log_dir.mkdir(parents=True, exist_ok=True)

    # ---- TP Setup ----
    rank, world = setup_tp()
    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    device = torch.device(f"cuda:{local_rank}")

    # ---- Log ----
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    log_file = log_dir / f"train_{MODEL_TAG}_rank{local_rank}_{timestamp}.log"
    sys.stdout = Logger(log_file)
    sys.stderr = sys.stdout
    print(f"[Log] Log started: {log_file}")

    dist.barrier()
    print(f"[Rank {rank}] World={world} | Device={device} | Mode=TP (no DDP)")
    print(f"[Rank {rank}] Traces: {traces_dir}")

    # ===== Model Loading =====
    load_dtype = torch.float16 if args.model_dtype == "float16" else torch.float32
    print(f"[Rank {rank}] Loading {args.model_name} ({args.model_dtype})...")
    print(f"[Rank {rank}] ⚠️ This will download ~3 GB on first run")

    t_load_start = time.time()
    model = AutoModelForCausalLM.from_pretrained(
        args.model_name,
        dtype=load_dtype,
    ).to(device)
    t_load = time.time() - t_load_start
    print(f"[Rank {rank}] Model loaded in {t_load:.1f}s")

    # ---- Print model info before TP ----
    n_params = sum(p.numel() for p in model.parameters())
    n_layers = model.config.num_hidden_layers
    hidden_size = model.config.hidden_size
    if rank == 0:
        print(f"\n{'='*60}")
        print(f"[Model] {args.model_name}")
        print(f"[Model] Parameters: {n_params:,} ({n_params/1e9:.2f}B)")
        print(f"[Model] Layers: {n_layers}, Hidden: {hidden_size}")
        print(f"[Model] Pre-TP memory per GPU: {n_params * (2 if load_dtype == torch.float16 else 4) / 1e9:.2f} GB (weights only)")
        print(f"{'='*60}\n")

        # Save model info for downstream tools (add_ddp_to_et.py, etc.)
        model_info = {
            "model_name": args.model_name,
            "model_tag": MODEL_TAG,
            "n_params": n_params,
            "n_layers": n_layers,
            "hidden_size": hidden_size,
            "param_dtype_bytes": 2 if load_dtype == torch.float16 else 4,
            "tp_degree": world,
        }
        model_info_path = metrics_dir / f"model_info_{MODEL_TAG}.json"
        with model_info_path.open("w") as f:
            json.dump(model_info, f, indent=2)
        print(f"[Model] Info saved to {model_info_path}")

    # ===== Apply Tensor Parallelism =====
    print(f"[Rank {rank}] Applying TP=2 parallelization...")

    tp_mesh = init_device_mesh("cuda", (world,), mesh_dim_names=("tp",))
    tp_plan = build_qwen_tp_plan(n_layers)

    try:
        model = parallelize_module(model, tp_mesh["tp"], tp_plan)
        print(f"[Rank {rank}] ✅ TP parallelization successful")
    except Exception as e:
        print(f"[Rank {rank}] ❌ TP parallelization failed: {e}")
        print(f"[Rank {rank}] This may be a PyTorch version or ROCm compatibility issue.")
        print(f"[Rank {rank}] Try: pip install --upgrade torch --break-system-packages")
        cleanup_tp()
        sys.exit(1)

    # ---- Print post-TP info ----
    # DTensor.numel() returns global size; use ._local_tensor for actual per-GPU size
    def _local_numel(p):
        if hasattr(p, '_local_tensor'):
            return p._local_tensor.numel()
        return p.numel()

    n_params_local = sum(_local_numel(p) for p in model.parameters())
    if rank == 0:
        print(f"\n{'='*60}")
        print(f"[TP] Post-TP parameters per GPU: {n_params_local:,}")
        print(f"[TP] Reduction ratio: {n_params_local / n_params:.2%}")
        print(f"[TP] Estimated per-GPU memory (weights + optimizer):")
        weight_gb = n_params_local * (2 if load_dtype == torch.float16 else 4) / 1e9
        # FP16 model → no FP32 master weights, optimizer states are FP16 too
        # AdamW with FP16 params: m (FP16) + v (FP16) = 2 × 2 bytes
        optim_gb = n_params_local * 2 * 2 / 1e9
        grad_gb = n_params_local * 2 / 1e9  # FP16 gradients
        print(f"[TP]   Weights:   {weight_gb:.2f} GB")
        print(f"[TP]   Optimizer: {optim_gb:.2f} GB (m + v, FP16)")
        print(f"[TP]   Gradients: {grad_gb:.2f} GB")
        print(f"[TP]   Total:     {weight_gb + optim_gb + grad_gb:.2f} GB (+ activations)")
        print(f"{'='*60}\n")

    # ===== Monkey Patch: Tag dist.all_reduce with bytes =====
    # DTensor 內部呼叫 all_reduce 時不帶 bytes 資訊，
    # 導致 trace 裡只有 nccl:all_reduce 而沒有通訊量。
    # 這裡動態替換 all_reduce，加上 |bytes=...|pg=tp0 標記，
    # 讓 converter 可以解析出正確的 comm_size。
    # 原理與 DDP 腳本的 make_tagging_allreduce_hook 相同。
    #
    # 需要 patch 兩條路徑：
    #   1. dist.all_reduce — 標準 API
    #   2. torch.distributed._functional_collectives.all_reduce — DTensor 常用路徑
    _patched_count = 0

    # Path 1: dist.all_reduce
    _orig_all_reduce = dist.all_reduce

    def _tagged_all_reduce(tensor, op=dist.ReduceOp.SUM, group=None, async_op=False):
        bytes_ = tensor.numel() * tensor.element_size()
        with record_function("record_param_comms"):
            with record_function(f"nccl:all_reduce|bytes={bytes_}|pg=tp0"):
                return _orig_all_reduce(tensor, op=op, group=group, async_op=async_op)

    dist.all_reduce = _tagged_all_reduce
    _patched_count += 1

    # Path 2: functional collectives (DTensor may use this instead)
    try:
        import torch.distributed._functional_collectives as funcol
        if hasattr(funcol, 'all_reduce'):
            _orig_funcol_all_reduce = funcol.all_reduce

            def _tagged_funcol_all_reduce(tensor, reduceOp, group, *args, **kwargs):
                bytes_ = tensor.numel() * tensor.element_size()
                with record_function("record_param_comms"):
                    with record_function(f"nccl:all_reduce|bytes={bytes_}|pg=tp0"):
                        return _orig_funcol_all_reduce(tensor, reduceOp, group, *args, **kwargs)

            funcol.all_reduce = _tagged_funcol_all_reduce
            _patched_count += 1
    except ImportError:
        pass

    if rank == 0:
        print(f"[Patch] all_reduce monkey-patched ({_patched_count} paths) with bytes tagging")

    # ===== Optimizer =====
    # Note: NO DDP wrapping — this is pure TP
    # Note: NO GradScaler — model is FP16, gradients are FP16.
    #       GradScaler expects FP32 gradients and will error on FP16.
    #       For trace collection, loss scaling is not needed.
    optimiz = optim.AdamW(model.parameters(), lr=1e-4, foreach=False)

    # ===== Dataset =====
    # TP: both GPUs see SAME data → no DistributedSampler
    vocab_size = model.config.vocab_size
    trainset = DummyLLMDataset(
        vocab_size=vocab_size, seq_len=args.seq_len, num_samples=2000
    )
    loader = DataLoader(
        trainset, batch_size=args.batch_size, shuffle=True,
        num_workers=0, pin_memory=True, drop_last=True,
    )
    if rank == 0:
        print(f"[Data] DummyLLMDataset: vocab={vocab_size}, "
              f"seq_len={args.seq_len}, samples=2000")
        print(f"[Data] No DistributedSampler (TP: both GPUs see same data)")

    # ---- GPU Monitor ----
    gpu_mon = None
    if not args.disable_gpu_monitoring:
        gpu_mon = GPUMonitor(
            local_rank, args.gpu_sample_interval,
            output_dir=metrics_dir, rank=rank, model_name=MODEL_TAG,
        )

    # ---- Trace Paths ----
    host_path = traces_dir / f"host_{rank}_{MODEL_TAG}.json"
    device_path = traces_dir / f"device_{rank}_{MODEL_TAG}.json"

    # ---- Cleanup ----
    if rank == 0 and not args.no_cleanup:
        rm = 0
        for p in traces_dir.glob(f"host_*_{MODEL_TAG}.json"):
            p.unlink(missing_ok=True); rm += 1
        for p in traces_dir.glob(f"device_*_{MODEL_TAG}.json"):
            p.unlink(missing_ok=True); rm += 1
        for p in traces_dir.glob("*.tmp"):
            p.unlink(missing_ok=True); rm += 1
        print(f"[Cleanup] Removed {rm} traces for '{MODEL_TAG}'")

    # ---- Profiler Settings ----
    target_epoch = max(1, int(args.profile_epoch))
    wait_steps = max(0, int(args.trace_wait))
    active_steps = max(1, int(args.trace_steps))
    exp_cfg = _ExperimentalConfig(enable_cuda_sync_events=True)

    # ===== Warmup: verify forward+backward works before profiling =====
    if rank == 0:
        print(f"\n[Warmup] Running 1 test step to verify TP forward/backward...")
    try:
        test_x = torch.randint(0, vocab_size, (args.batch_size, args.seq_len)).to(device)
        test_y = test_x.clone()
        optimiz.zero_grad(set_to_none=True)
        with torch.autocast(device_type="cuda", dtype=torch.float16):
            out = model(input_ids=test_x, labels=test_y)
            loss = out.loss
        loss.backward()
        optimiz.step()
        if rank == 0:
            print(f"[Warmup] ✅ Test step passed! Loss = {loss.item():.4f}")
            # Report memory usage
            mem_allocated = torch.cuda.max_memory_allocated(device) / 1e9
            mem_reserved = torch.cuda.max_memory_reserved(device) / 1e9
            print(f"[Memory] Peak allocated: {mem_allocated:.2f} GB, "
                  f"Reserved: {mem_reserved:.2f} GB / 16.00 GB")
            if mem_allocated > 14.0:
                print(f"[Memory] ⚠️ High memory usage. Consider reducing --batch-size or --seq-len")
    except torch.cuda.OutOfMemoryError as e:
        print(f"[FATAL] OOM during warmup: {e}")
        print(f"[FATAL] Try: --batch-size 1 --seq-len 128 --model-dtype float16")
        cleanup_tp()
        sys.exit(1)
    except Exception as e:
        print(f"[FATAL] Warmup failed: {e}")
        import traceback; traceback.print_exc()
        cleanup_tp()
        sys.exit(1)

    # Reset peak memory stats after warmup
    torch.cuda.reset_peak_memory_stats(device)

    # ===================== Training Loop =====================
    for epoch in range(1, args.epochs + 1):
        if args.debug_epoch_print and rank == 0:
            print(f"[Debug] Begin Epoch {epoch} (profile={epoch == target_epoch})")

        if epoch != target_epoch:
            # Non-profile epoch: normal training
            model.train()
            for x, y in loader:
                train_step(model, x, y, optimiz, device)
            continue

        # ============ Profile Epoch ============
        it = iter(loader)

        # Skip warmup steps
        skipped = 0
        while skipped < wait_steps:
            try:
                x, y = next(it)
            except StopIteration:
                it = iter(loader)
                x, y = next(it)
            train_step(model, x, y, optimiz, device)
            skipped += 1

        if rank == 0:
            print(f"[Profile] Skipped {skipped} warmup steps, now profiling {active_steps} steps")

        # ===== Profiled Window: ETO + Kineto =====
        et = ExecutionTraceObserver()
        host_tmp = Path(str(host_path) + ".tmp")
        host_tmp.unlink(missing_ok=True)
        et.register_callback(str(host_tmp))
        et.start()

        if gpu_mon:
            gpu_mon.start()

        with profile(
            activities=supported_activities(),
            schedule=schedule(wait=0, warmup=0, active=active_steps, repeat=1),
            record_shapes=bool(args.trace_shapes),
            profile_memory=bool(args.trace_mem),
            with_stack=bool(args.trace_stack),
            experimental_config=exp_cfg,
        ) as prof:
            model.train()
            aux_stream = (
                torch.cuda.Stream(priority=0)
                if (args.inject_sync_hack and torch.cuda.is_available())
                else None
            )

            for step_idx in range(1, active_steps + 1):
                try:
                    x, y = next(it)
                except StopIteration:
                    it = iter(loader)
                    x, y = next(it)

                with record_function(f"ProfilerStep#{step_idx}"):
                    loss_val = train_step(model, x, y, optimiz, device)

                if rank == 0:
                    print(f"[Profile] Step {step_idx}/{active_steps}, loss={loss_val:.4f}")

                # Inject sync events for trace linker stability
                if args.inject_sync_hack and torch.cuda.is_available():
                    with record_function("manual_sync_event_hack"):
                        s0 = torch.cuda.current_stream()
                        evt = torch.cuda.Event(enable_timing=False)
                        with torch.cuda.stream(s0):
                            evt.record()
                        with torch.cuda.stream(aux_stream):
                            aux_stream.wait_event(evt)
                            _ = torch.empty(1, device='cuda').add_(1)

                if step_idx == active_steps:
                    with record_function("manual_blocking_sync"):
                        torch.cuda.synchronize()

                prof.step()

        # ---- Report memory & stats ----
        if rank == 0:
            mem_peak = torch.cuda.max_memory_allocated(device) / 1e9
            print(f"\n{'='*60}")
            print(f"[TP Stats] Peak GPU memory during profiling: {mem_peak:.2f} GB")
            print(f"[TP Stats] Model: {args.model_name}")
            print(f"[TP Stats] TP=2, Layers={n_layers}, Hidden={hidden_size}")
            print(f"[TP Stats] Expected COMM per step: ~{n_layers * 4} "
                  f"(fwd {n_layers*2} + bwd {n_layers*2})")
            print(f"{'='*60}\n")

        # ---- Export Device Trace ----
        try:
            tmp_dev = Path(str(device_path) + ".tmp")
            prof.export_chrome_trace(str(tmp_dev))
            os.replace(tmp_dev, device_path)
            _check_device_sync_and_steps(device_path)
        except Exception as e:
            print(f"[Rank {rank}] export device trace failed: {e}")
            if 'tmp_dev' in locals() and tmp_dev.exists():
                tmp_dev.unlink(missing_ok=True)

        # ---- Export Host Trace ----
        try:
            et.stop()
            try:
                et.unregister_callback()
            except TypeError:
                et.unregister_callback(None)
            _wait_file_stable(host_tmp, tries=20, sleep_s=0.05)
            try:
                json.load(open(host_tmp, "r", encoding="utf-8"))
            except Exception:
                _repair_host_json(host_tmp)
            os.replace(host_tmp, host_path)
            print(f"[Rank {rank}] host trace saved: {host_path}")
        except Exception as e:
            print(f"[Rank {rank}] export host trace failed: {e}")
            host_tmp.unlink(missing_ok=True)

        if gpu_mon:
            gpu_mon.stop()
        break  # Only profile one epoch

    print(f"[Log] Full log saved to: {log_file}")
    cleanup_tp()
    print(f"[Rank {rank}] Done. Next: conver_to_chakra_et.py --model-tag {MODEL_TAG}")


if __name__ == "__main__":
    main()
