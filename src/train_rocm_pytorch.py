#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
CIFAR-10 Training Framework for ROCm (AMD GPU) with Distributed Profiling
(Chakra-compatible; aligned with standard practice)

Usage:
  # 建議：把視窗移到第 2 個 epoch 的中段
  torchrun --standalone --nproc_per_node=2 ./src/train_rocm_pytorch.py \
    --epochs 3 --batch-size 128 \
    --profile-epoch 2 --trace-wait 32 --trace-steps 128

  # 若仍缺 EventRecord，可暫時加上工程保險（不影響數值結果）：
  #   --inject-sync-hack
"""

import os
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
import torchvision
import torchvision.transforms as transforms

from torch.utils.data import DataLoader
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.profiler import profile, ProfilerActivity, record_function, schedule, supported_activities
from torch.profiler import ExecutionTraceObserver
from torch._C._profiler import _ExperimentalConfig

from rocm_compat import ROCmCompat, GPUFrequencyInfo

# ---------- 檔案穩定 / host JSON 拼接修復 ----------
def _wait_file_stable(p: Path, tries: int = 10, sleep_s: float = 0.05) -> None:
    last = -1
    for _ in range(tries):
        sz = p.stat().st_size if p.exists() else -1
        if sz == last and sz > 0:
            return
        last = sz
        time.sleep(sleep_s)

def _repair_host_json(host_path: Path) -> bool:
    """
    若檔內有多個 JSON 物件拼接（常見於 ETO callback），
    解析所有片段，選擇「nodes 數量最多，若相同則 start_ts 最大」的片段覆寫。
    """
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
        print(f"[repair] host trace fixed -> {host_path} (picked max-nodes segment)")
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
        print(f"[check] {device_json.name}: sync_events=({'OK' if (has_rec and has_wait) else 'MISS'}) "
              f"(rec={'Y' if has_rec else 'N'}, wait={'Y' if has_wait else 'N'}), "
              f"prof_steps={'OK' if has_step else 'MISS'}")
    except Exception as e:
        print(f"[check] skip device check: {e}")

# ---------- GPU 監測（可關閉） ----------
class GPUMonitor:
    def __init__(self, device_id=0, sample_interval=0.1):
        self.device_id = device_id
        self.sample_interval = sample_interval
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
                if fi and fi.sclk_mhz is not None:
                    self.samples.append({"timestamp": fi.timestamp, "sclk_mhz": fi.sclk_mhz})
            except Exception as e:
                print(f"[GPUMonitor] Query failed: {e}")
            time.sleep(self.sample_interval)

    def start(self):
        if self.monitoring: return
        self.monitoring = True
        self.samples.clear()
        self._th = threading.Thread(target=self._poll, daemon=True)
        self._th.start()
        print(f"[GPUMonitor] Started on GPU {self.device_id}")

    def stop(self):
        if not self.monitoring: return
        self.monitoring = False
        if self._th: self._th.join(timeout=2.0)
        print(f"[GPUMonitor] Stopped on GPU {self.device_id} (samples={len(self.samples)})")

# ---------- 模型 ----------
class CIFAR10_CNN(nn.Module):
    def __init__(self, num_classes=10):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 64, 3, padding=1), nn.ReLU(True),
            nn.Conv2d(64, 128, 3, padding=1), nn.ReLU(True),
            nn.MaxPool2d(2),
            nn.Conv2d(128, 256, 3, padding=1), nn.ReLU(True),
            nn.MaxPool2d(2),
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(256*8*8, 1024), nn.ReLU(True),
            nn.Linear(1024, num_classes),
        )
    def forward(self, x):
        return self.classifier(self.features(x))

# ---------- DDP ----------
def setup_ddp():
    world = int(os.environ.get("WORLD_SIZE", "1"))
    if world > 1:
        local_rank = int(os.environ["LOCAL_RANK"])
        torch.cuda.set_device(local_rank)
        dev = torch.device(f"cuda:{local_rank}")
        try:
            dist.init_process_group(backend="nccl", init_method="env://", device_id=dev)
        except TypeError:
            dist.init_process_group(backend="nccl", init_method="env://")
        return int(os.environ["RANK"]), world, True
    return 0, 1, False

def cleanup_ddp(enabled):
    if enabled and dist.is_initialized():
        try:
            dist.barrier()
        finally:
            dist.destroy_process_group()

def is_main(rank: int) -> bool:
    return rank == 0

def make_tagging_allreduce_hook():
    from torch.distributed.algorithms.ddp_comm_hooks import default_hooks as dh
    def hook(state, bucket):
        with record_function("nccl_all_reduce"):
            return dh.allreduce_hook(state, bucket)
    return hook

# ---------- 主程式 ----------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--epochs", type=int, default=3)
    ap.add_argument("--batch-size", type=int, default=128)
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--omp-threads", type=int, default=2)
    ap.add_argument("--debug-epoch-print", action="store_true")

    # 視窗定位與大小
    ap.add_argument("--profile-epoch", type=int, default=2, help="要擷取的 epoch（1-based）")
    ap.add_argument("--trace-wait", type=int, default=32, help="先略過前 N 步再開始 profile（避開 warm-up）")
    ap.add_argument("--trace-steps", type=int, default=128, help="視窗內擷取的步數")
    ap.add_argument("--trace-shapes", action="store_true")
    ap.add_argument("--trace-stack", action="store_true")
    ap.add_argument("--trace-mem", action="store_true")

    # 監測與清理
    ap.add_argument("--disable-gpu-monitoring", action="store_true")
    ap.add_argument("--gpu-sample-interval", type=float, default=0.1)
    ap.add_argument("--no-cleanup", action="store_true")

    # 工程保險：人工注入 Record/Wait
    ap.add_argument("--inject-sync-hack", action="store_true",
                    help="在每步尾端注入一對 hipEventRecord/hipStreamWaitEvent（輔助 stream）以幫助 HTA 連結；升版穩定後可關閉。")

    args = ap.parse_args()
    os.environ["OMP_NUM_THREADS"] = str(args.omp_threads)

    # 路徑
    script_dir = os.path.dirname(os.path.abspath(__file__))
    base_dir   = os.path.abspath(os.path.join(script_dir, "../data/chakra"))
    traces_dir = Path(base_dir) / "pytorch_traces"
    metrics_dir= Path(base_dir) / "gpu_metrics"
    data_dir   = os.path.abspath(os.path.join(script_dir, "../data/cifar10"))
    traces_dir.mkdir(parents=True, exist_ok=True)
    metrics_dir.mkdir(parents=True, exist_ok=True)
    Path(data_dir).mkdir(parents=True, exist_ok=True)

    # DDP
    rank, world, ddp = setup_ddp()
    device = torch.device(f"cuda:{int(os.environ.get('LOCAL_RANK',0))}" if ddp else "cuda")

    # 清舊檔
    if is_main(rank) and not args.no_cleanup:
        rm = 0
        for p in traces_dir.glob("host_*.json"): p.unlink(missing_ok=True); rm += 1
        for p in traces_dir.glob("device_*.json"): p.unlink(missing_ok=True); rm += 1
        for p in traces_dir.glob("*.tmp"): p.unlink(missing_ok=True); rm += 1
        print(f"[Cleanup] removed {rm} traces under {traces_dir}")

    if ddp:
        dist.barrier()

    print(f"[Rank {rank}] World={world} | Device={device}")
    print(f"[Rank {rank}] Traces : {traces_dir}")
    print(f"[Rank {rank}] Dataset: {data_dir}")

    # 資料集
    tfm = transforms.Compose([transforms.ToTensor(),
                              transforms.Normalize((0.5,0.5,0.5),(0.5,0.5,0.5))])
    trainset = torchvision.datasets.CIFAR10(root=data_dir, train=True, download=True, transform=tfm)
    sampler = torch.utils.data.distributed.DistributedSampler(trainset, num_replicas=world, rank=rank) if ddp else None
    loader  = DataLoader(trainset, batch_size=args.batch_size, shuffle=(sampler is None),
                         sampler=sampler, num_workers=args.workers, pin_memory=True, drop_last=True)

    # 模型/優化器
    model = CIFAR10_CNN().to(device)
    if ddp:
        model = DDP(model, device_ids=[device.index], output_device=device.index, broadcast_buffers=False)
        model.register_comm_hook(None, make_tagging_allreduce_hook())
    optimiz = optim.AdamW(model.parameters(), lr=1e-3)
    scaler  = torch.amp.GradScaler('cuda')
    loss_fn = nn.CrossEntropyLoss()

    # 監測
    gpu_mon = None
    if not args.disable_gpu_monitoring:
        dev_id = int(os.environ.get('LOCAL_RANK', 0)) if ddp else 0
        gpu_mon = GPUMonitor(dev_id, args.gpu_sample_interval)

    # 名稱
    host_path   = traces_dir / f"host_{rank}.json"
    device_path = traces_dir / f"device_{rank}.json"

    # 視窗與同步事件
    target_epoch = max(1, int(args.profile_epoch))
    wait_steps   = max(0, int(args.trace_wait))
    active_steps = max(1, int(args.trace_steps))
    exp_cfg = _ExperimentalConfig(enable_cuda_sync_events=True)

    # ===== 訓練迴圈 =====
    for epoch in range(1, args.epochs + 1):
        if ddp and sampler is not None:
            sampler.set_epoch(epoch)

        if args.debug_epoch_print and is_main(rank):
            print(f"[Debug] Begin Epoch {epoch} (profile={epoch==target_epoch})")

        if epoch != target_epoch:
            # 不記錄 trace：正常訓練
            model.train()
            for x, y in loader:
                x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)
                optimiz.zero_grad(set_to_none=True)
                with torch.autocast(device_type="cuda", dtype=torch.float16):
                    logits = model(x); loss = loss_fn(logits, y)
                scaler.scale(loss).backward(); scaler.step(optimiz); scaler.update()
            continue

        # ---- 單一 iterator：先略過 wait_steps，再用同一 iterator 進入 profiler 視窗 ----
        it = iter(loader)
        skipped = 0
        while skipped < wait_steps:
            x, y = next(it)
            x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)
            optimiz.zero_grad(set_to_none=True)
            with torch.autocast(device_type="cuda", dtype=torch.float16):
                logits = model(x); loss = loss_fn(logits, y)
            scaler.scale(loss).backward(); scaler.step(optimiz); scaler.update()
            skipped += 1

        # ===== 目標視窗：ETO start/stop + profiler schedule（用同一 iterator） =====
        et = ExecutionTraceObserver()
        host_tmp = Path(str(host_path) + ".tmp"); host_tmp.unlink(missing_ok=True)
        et.register_callback(str(host_tmp))
        et.start()

        if gpu_mon: gpu_mon.start()

        with profile(
            activities=supported_activities(),
            schedule=schedule(wait=0, warmup=0, active=active_steps, repeat=1),
            record_shapes=bool(args.trace_shapes),
            profile_memory=bool(args.trace_mem),
            with_stack=bool(args.trace_stack),
            experimental_config=exp_cfg
        ) as prof:
            model.train()
            aux_stream = torch.cuda.Stream(priority=0) if (args.inject_sync_hack and torch.cuda.is_available()) else None

            for step_idx in range(1, active_steps + 1):
                t0 = time.time()
                x, y = next(it)
                with record_function(f"ProfilerStep#{step_idx}"):
                    x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)
                    optimiz.zero_grad(set_to_none=True)
                    with torch.autocast(device_type="cuda", dtype=torch.float16):
                        logits = model(x); loss = loss_fn(logits, y)
                    scaler.scale(loss).backward(); scaler.step(optimiz); scaler.update()

                # 工程保險：注入一對 Record/Wait (修改版：使用單一 stream)
                if args.inject_sync_hack and torch.cuda.is_available():
                    with record_function("manual_sync_event_hack"):
                        s0 = torch.cuda.current_stream()
                        s1 = torch.cuda.Stream(priority=0)  # 輔助 stream，模擬通訊同步
                        evt = torch.cuda.Event(enable_timing=False)  # HIP 等效事件
                        with torch.cuda.stream(s0):
                            evt.record()  # hipEventRecord
                        with torch.cuda.stream(s1):
                            s1.wait_event(evt)  # hipStreamWaitEvent
                            _ = torch.empty(1, device='cuda').add_(1)  # 小 kernel 確保記錄

                # with record_function("manual_blocking_sync"):
                #     torch.cuda.synchronize()

                prof.step()

        # device：原子寫入 + 自檢
        try:
            tmp_dev = Path(str(device_path) + ".tmp")
            prof.export_chrome_trace(str(tmp_dev))
            os.replace(tmp_dev, device_path)
            _check_device_sync_and_steps(device_path)
        except Exception as e:
            print(f"[Rank {rank}] export device trace failed: {e}")
            if 'tmp_dev' in locals() and tmp_dev.exists(): tmp_dev.unlink(missing_ok=True)

        # host：stop / unregister / 等待 flush / 修復拼接 / 原子改名
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
            print(f"[Rank {rank}] host trace saved:   {host_path}")
        except Exception as e:
            print(f"[Rank {rank}] export host trace failed: {e}")
            host_tmp.unlink(missing_ok=True)

        if gpu_mon: gpu_mon.stop()
        break  # 只擷取一個 epoch

    cleanup_ddp(ddp)

if __name__ == "__main__":
    main()
