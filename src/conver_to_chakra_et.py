#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Link PyTorch host/device traces -> HDT (ET+), then convert to Chakra ET (.et)
PyTorch Trace 轉 Chakra ET 轉換工具 (AMD ROCm 增強版)

主要功能：
1. **Trace Link & Convert**: 將 PyTorch Profiler (Kineto) 產生的 JSON 轉換為 ASTRA-sim 可用的 ET 格式。
2. **AMD ROCm 兼容修補**: 自動識別並修復 AMD NCCL Kernel (`ncclDevKernel_Generic`) 命名問題。
3. **系統感知校準 (System-Aware Calibration)**: 針對小模型 (Tiny Workload) 提供強制校準功能，將 System Overhead 攤提至計算節點。
4. **多模型支援**: 支援透過 `--model-tag` 處理不同模型的 Trace (如 cifar10, resnet50)。

=== AMD GPU 兼容性修補說明 ===
問題：Chakra 原本為 NVIDIA GPU 設計，無法識別 AMD GPU 的 NCCL kernel 命名格式。
- NVIDIA: "ncclKernel_AllReduce_..." (明確指出操作類型)
- AMD:    "ncclDevKernel_Generic_4(...)" (使用 Generic，不明確指出操作類型)
解決方案：
- 動態修補 (Monkey Patching) `PyTorchConverter`，將 "Generic" Kernel 映射為 `ALL_REDUCE` 通訊節點。

=== 系統感知校準 (System-Aware Calibration) ===
問題：在小任務 (如 CIFAR-10) 中，System Overhead (Launch Latency, Memcpy) 佔據 99% 時間，但模擬器會忽略這些間隙，導致模擬時間嚴重低估。
解決方案：
- 使用 `--force-avg-kernel-ns` 強制指定平均 Kernel 時間。
- 將真實世界的總 GPU 時間 (包含 Overhead) 平均攤提回計算節點，使模擬時間對齊真實時間。

資料夾結構（預設）：
  ./data/chakra/
    ├─ pytorch_traces/   # 輸入: host_{rank}_{tag}.json, device_{rank}_{tag}.json
    └─ workload_et/      # 輸出: workload.{tag}.{rank}.et
    └─ gpu_metrics/      # 輸入: gpu_metrics_{rank}_{tag}.json (由 train_rocm_pytorch.py 產生)

用法範例：

  1. 【標準模式】轉換 Compute-Bound 模型 (如 ResNet-50)
     直接使用 Trace 中的數據與真實頻率進行轉換。
     $ python src/conver_to_chakra_et.py --model-tag resnet50 --default-gpu-freq 2935

  2. 【校準模式】轉換 System-Bound 模型 (如 CIFAR-10)
     強制將所有 Kernel 設為 609us (609000ns) 以補償 System Overhead。
     $ python src/conver_to_chakra_et.py --model-tag cifar10 --force-avg-kernel-ns 609000 --default-gpu-freq 2935

  3. 【一般模式】不指定 Tag (相容舊檔)
     處理 host_{rank}.json
     $ python src/conver_to_chakra_et.py

  4. 【壓力測試模式】生成 All-to-All 合成負載
     將原本的 AllReduce 替換為 All-to-All，用於測試 Twisted Torus 拓撲。
     $ python src/conver_to_chakra_et.py --model-tag resnet50 --replace-comm all_to_all

參數說明：
  --model-tag STR          : 指定模型標籤 (用於檔名過濾與輸出命名)
  --force-avg-kernel-ns INT: [校準用] 強制指定平均 Kernel 時間 (奈秒)，若設定則忽略 Trace 真實平均值
  --default-gpu-freq FLOAT : 預設 GPU 頻率 (MHz)，當找不到 metrics 檔案時使用
  --replace-comm STR       : [壓力測試] 強制替換通訊模式 (目前支援: 'all_to_all')
  --no-clean               : 保留舊的輸出檔案
"""
from __future__ import annotations

import sys
import time
import argparse
import json
import re
import shutil
import subprocess
from pathlib import Path
from typing import List, Dict, Tuple, Optional

# ------------------------------------------------------------------
# 轉檔後：DAG 修正所需（Chakra protobuf）
# ------------------------------------------------------------------
from chakra.src.third_party.utils.protolib import decodeMessage as _decode_msg
from chakra.src.third_party.utils.protolib import encodeMessage as _encode_msg
from chakra.schema.protobuf.et_def_pb2 import GlobalMetadata, Node, AttributeProto, COMP_NODE, COMM_COLL_NODE, COMM_SEND_NODE, COMM_RECV_NODE, ALL_REDUCE, ALL_GATHER, REDUCE_SCATTER, ALL_TO_ALL  # type: ignore
try:
    from chakra.schema.protobuf.et_def_pb2 import BROADCAST  # type: ignore
except Exception:
    BROADCAST = None

class Logger(object):
    """
    將輸出同時導向終端機與檔案的輔助類別 (Tee)
    """
    def __init__(self, filename):
        self.terminal = sys.stdout
        self.log = open(filename, "w", encoding="utf-8") # 使用寫入模式

    def write(self, message):
        self.terminal.write(message)
        self.log.write(message)
        self.log.flush() # 確保即時寫入，避免程式崩潰時 log 是空的

    def flush(self):
        self.terminal.flush()
        self.log.flush()

# --------------------------- AMD GPU 修補 ---------------------------

def apply_amd_gpu_patch():
    """
    修補 PyTorchConverter 類以支援 AMD GPU 的 NCCL kernel 識別

    === 問題背景 ===
    AMD GPU 使用不同的 NCCL kernel 命名格式，原始 Chakra 只支援 NVIDIA 格式。

    原始支援的格式：
    - "ncclKernel_*" (NVIDIA 標準格式)
    - "ncclDevKernel_*" (NVIDIA 開發格式)

    AMD GPU 的實際格式：
    - "ncclDevKernel_Generic_4(ncclDevKernelArgsStorage<4096ul>)" (AMD 專用格式)

    === 修補方法 ===
    修補 get_protobuf_node_type_from_json_node 方法，擴展 NCCL kernel 識別邏輯

    該方法負責決定節點的類型：
    - COMP_NODE (4): 計算節點
    - COMM_COLL_NODE (7): 集體通訊節點
    - COMM_SEND_NODE (5): 點對點發送
    - COMM_RECV_NODE (6): 點對點接收
    """
    try:
        from chakra.src.converter.pytorch_converter import PyTorchConverter
        from chakra.schema.protobuf.et_def_pb2 import COMM_COLL_NODE

        # 保存原始方法的引用
        original_method = PyTorchConverter.get_protobuf_node_type_from_json_node

        def patched_get_protobuf_node_type_from_json_node(self, json_node_map, json_node):
            # 記錄 GPU 操作用於除錯
            if json_node.is_gpu_op():

                # 檢查 AMD GPU NCCL Generic kernel (底層 kernel 名稱)
                if "ncclDevKernel_Generic" in json_node.name:
                    print(f"[patch] 偵測到 AMD GPU NCCL Generic kernel -> COMM_COLL_NODE")
                    return COMM_COLL_NODE

            # 檢查所有 NCCL 相關操作，包括 PyTorch profiler 生成的名稱
            if ("nccl:all_reduce" in json_node.name or
                "nccl:all_gather" in json_node.name or
                "nccl:reduce_scatter" in json_node.name or
                "nccl:broadcast" in json_node.name or
                "nccl:all_to_all" in json_node.name or
                "ncclDevKernel_Generic" in json_node.name):
                # 確保 NCCL 操作被標記為 GPU 操作，以便 is_cpu_op 被正確設置為 False
                json_node.cat = "kernel"  # 設置為 GPU kernel 類別
                return COMM_COLL_NODE

            # 對於所有其他情況，使用原始方法處理
            return original_method(self, json_node_map, json_node)

        # 動態替換類方法
        PyTorchConverter.get_protobuf_node_type_from_json_node = patched_get_protobuf_node_type_from_json_node
        print("[patch] 已成功修補 get_protobuf_node_type_from_json_node 方法，支援 AMD GPU NCCL kernels")
        return True

    except Exception as e:
        print(f"[patch] 修補失敗: {e}")
        print("[patch] 這可能是因為 Chakra 版本不兼容或導入錯誤")
        return False


def patch_collective_comm_type_for_amd():
    """
    動態修補 Chakra 的 get_collective_comm_type 方法，支援 AMD GPU 的 NCCL kernel 名稱。

    === 問題背景 ===
    Chakra 原本為 NVIDIA GPU (CUDA) 設計，其 get_collective_comm_type 方法只能識別標準的
    NCCL collective 命名格式如 "allreduce", "allgather" 等。但是 AMD GPU (ROCm/HIP)
    在執行 NCCL 操作時會產生特殊的 kernel 名稱格式：
    "ncclDevKernel_Generic_4(ncclDevKernelArgsStorage<4096ul>)"

    === 根本原因 ===
    AMD GPU 的 HIP runtime 和 NVIDIA 的 CUDA runtime 在 profiling 時產生的 kernel 名稱
    格式不同。AMD GPU 使用 "Generic" 作為通用 NCCL kernel 的標識符，而不是明確指出
    操作類型（如 allreduce）。這導致 Chakra 無法從名稱中推斷出 collective 操作類型。

    === 修補策略 ===
    1. 非入侵性修補：不修改 Chakra 主程式碼，而是在運行時動態替換方法
    2. 向前兼容：保留原始方法的所有功能，只新增 AMD GPU 支援
    3. 合理推測：將 "ncclDevKernel_Generic" 映射到 ALL_REDUCE（深度學習中最常見的操作）

    === 技術實現 ===
    使用 Python 的動態特性（monkey patching）在運行時替換類方法。
    這樣可以在不修改原始程式碼的情況下擴展功能。
    """
    try:
        from chakra.src.converter.pytorch_converter import PyTorchConverter
        from chakra.schema.protobuf.et_def_pb2 import ALL_REDUCE

        # 保存原始方法的引用，以便在修補版本中調用
        original_method = PyTorchConverter.get_collective_comm_type

        def patched_get_collective_comm_type(self, name: str) -> int:
            ln = name.lower()
            # 處理標準 NCCL 操作（精確分類）
            if any(pattern in ln for pattern in ["nccl_all_reduce", "nccl:all_reduce", "c10d::allreduce", "allreduce"]):
                print(f"[patch] 偵測到 NCCL AllReduce 操作: {name} -> ALL_REDUCE")
                return ALL_REDUCE
            if any(pattern in ln for pattern in ["nccl_all_gather", "nccl:all_gather", "c10d::allgather", "allgather"]):
                print(f"[patch] 偵測到 NCCL AllGather 操作: {name} -> ALL_GATHER")
                return ALL_GATHER
            if BROADCAST is not None and any(pattern in ln for pattern in ["nccl_broadcast", "nccl:broadcast", "c10d::broadcast", "broadcast"]):
                print(f"[patch] 偵測到 NCCL Broadcast 操作: {name} -> BROADCAST")
                return BROADCAST

            # 處理 AMD GPU 的 ncclDevKernel_Generic_X 格式（需要推測），預設為 ALL_REDUCE
            if "nccldevkernel_generic" in ln or "nccldevkernel" in ln:
                print(f"[patch] 警告: AMD GPU Generic kernel 無法確定具體通信類型: {name}; 預設 -> ALL_REDUCE")
                return ALL_REDUCE

            # 其餘情況回退到原始實作
            return original_method(self, name)

        # 動態替換類方法（monkey patching）
        PyTorchConverter.get_collective_comm_type = patched_get_collective_comm_type
        print("[patch] 已成功修補 get_collective_comm_type 方法，支援 AMD GPU NCCL kernels")
        return True

    except Exception as e:
        print(f"[patch] 修補失敗: {e}")
        print("[patch] 這可能是因為 Chakra 版本不兼容或導入錯誤")
        return False

# --------------------------- 共用工具 ---------------------------

def run_one(cmd: List[str]) -> bool:
    print("  $", " ".join(map(str, cmd)), flush=True)
    out = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
    print(out.stdout.rstrip())
    return out.returncode == 0


def detect_ranks(pt_dir: Path, ranks_arg: Optional[List[str]], tag: str = None) -> List[int]:
    if ranks_arg:
        return sorted(set(int(r) for r in ranks_arg))

    # 根據 tag 決定搜尋模式
    # 檔名格式: host_{rank}.json 或 host_{rank}_{tag}.json
    pattern_host = f"host_*_{tag}.json" if tag else "host_*.json"
    pattern_device = f"device_*_{tag}.json" if tag else "device_*.json"

    hosts = set()
    for p in pt_dir.glob(pattern_host):
        try:
            # 假設格式固定為 host_{rank}_...
            parts = p.stem.split('_')
            if len(parts) >= 2 and parts[1].isdigit():
                hosts.add(int(parts[1]))
        except: pass

    devs = set()
    for p in pt_dir.glob(pattern_device):
        try:
            parts = p.stem.split('_')
            if len(parts) >= 2 and parts[1].isdigit():
                devs.add(int(parts[1]))
        except: pass

    common = sorted(hosts & devs)
    if not common:
        msg = f"匹配 tag='{tag}' 的" if tag else ""
        print(f"[warn] 在 {pt_dir} 找不到{msg}成對 trace (host_X/device_X)")
        if tag:
            print(f"       請確認 train_rocm_pytorch.py 是否有加上 --model {tag} 並成功執行")
        return []

    return common


def clean_outputs(pt_dir: Path, et_dir: Path) -> None:
    removed = 0
    for p in pt_dir.glob("hdt_*.json"):
        try:
            p.unlink(); removed += 1
        except Exception as e:
            print(f"[warn] 無法刪除 {p.name}: {e}")
    print(f"[clean] 移除 {removed} 個舊的 HDT 檔（pytorch_traces/hdt_*.json）")
    et_dir.mkdir(parents=True, exist_ok=True)
    removed = 0
    for p in et_dir.glob("*.et"):
        try:
            p.unlink(); removed += 1
        except Exception as e:
            print(f"[warn] 無法刪除 {p.name}: {e}")
    print(f"[clean] 移除 {removed} 個舊的 ET 檔（workload_et/*.et）\n")


def infer_prefix_from_device(device_json: Path) -> str:
    txt = device_json.read_text(encoding="utf-8", errors="ignore").lower()
    kinds = {
        "allreduce":      [r"nccl:all_reduce", r"nccl_all_reduce", r"c10d::allreduce_", r"\ballreduce\b"],
        "allgather":      [r"nccl:all_gather", r"\ballgather\b", r"c10d::allgather_"],
        "reducescatter":  [r"nccl:reduce_scatter", r"\breducescatter\b", r"c10d::reducescatter_"],
        "broadcast":      [r"nccl:broadcast", r"\bbroadcast\b", r"c10d::broadcast_"],
    }
    cnt: Dict[str, int] = {}
    for k, pats in kinds.items():
        c = 0
        for p in pats:
            c += len(re.findall(p, txt))
        cnt[k] = c
    if any(cnt.values()):
        return max(cnt, key=cnt.get)
    return "workload"


# --------------------------- 連結 & 轉換 ---------------------------

def link_host_device(rank: int, host: Path, device: Path, out_hdt: Path) -> None:
    print(f"[link] rank={rank}  host={host.name}  device={device.name}  -> {out_hdt.name}")
    out_hdt.parent.mkdir(parents=True, exist_ok=True)
    cmd = [
        "chakra_trace_link",
        f"--rank={rank}",
        "--chakra-host-trace", str(host),
        "--chakra-device-trace", str(device),
        "--output-file", str(out_hdt),
        "--log-level", "WARNING"  # 降低 log 噪音，聚焦錯誤
    ]
    if not run_one(cmd):
        raise RuntimeError(f"chakra_trace_link 執行失敗，請檢查 trace 檔案或更新 Chakra/HTA 版本。")
    print("[ok] chakra_trace_link 完成\n")


# [修改] 增加 force_avg_kernel_ns 與 tag 參數，並支援 Metrics 檔名後綴
def add_compute_cycles_to_compute_nodes(et_file: Path, hdt_file: Path, device_file: Path = None, pt_dir: Path = None, rank: int = 0, default_freq_mhz: float = 2400.0, force_avg_kernel_ns: float = None, tag: str = None) -> int:
    """
    為 ET 文件中的 COMPUTE 節點添加 compute_cycles 屬性

    優先順序:
    1. 強制校準 (--force-avg-kernel-ns) [新增]
    2. 讀取 train_rocm_pytorch 產生的 gpu_metrics_{rank}_{tag}.json (真實訓練頻率)
    3. 使用參數傳入的預設頻率 (default_freq_mhz)

    參數:
        et_file: ET 文件路徑
        hdt_file: HDT 文件路徑
        device_file: 設備 trace 文件路徑
        pt_dir: PyTorch trace 目錄
        rank: 當前處理的 rank
        default_freq_mhz: 保底頻率 (MHz)
        force_avg_kernel_ns: 強制指定的平均 Kernel 時間 (ns) [新增]
        tag: 模型標籤，用於讀取對應 metrics [新增]

    返回:
        修復的節點數量
    """
    print(f"[fix-compute-cycles] 修復 {et_file.name} 中的 compute_cycles")

    actual_freq_ghz = None

    # [1] 嘗試讀取訓練期間記錄的 metrics
    if pt_dir:
        # [修改] 根據 tag 決定 metrics 檔名
        metrics_filename = f"gpu_metrics_{rank}_{tag}.json" if tag else f"gpu_metrics_{rank}.json"
        metrics_file = pt_dir.parent / "gpu_metrics" / metrics_filename

        if metrics_file.exists():
            try:
                with metrics_file.open('r') as f:
                    mdata = json.load(f)

                    median_f = mdata.get("sclk_median_mhz", 0)
                    avg_f = mdata.get("sclk_avg_mhz", 0)
                    max_f = mdata.get("sclk_max_mhz", 0)

                    # 智慧判斷邏輯
                    if median_f > 500:
                        sclk = median_f
                        print(f"[fix-compute-cycles] 使用中位數頻率: {sclk} MHz")
                    elif max_f > 500:
                        sclk = max_f
                        print(f"[fix-compute-cycles] ⚠️ 中位數過低 ({median_f} MHz)，改用最大頻率: {sclk} MHz")
                    else:
                        sclk = default_freq_mhz
                        print(f"[fix-compute-cycles] ⚠️ 偵測到的頻率皆過低 (Max={max_f})，改用預設值")

                    if sclk:
                        actual_freq_ghz = sclk / 1000.0
                        print(f"[fix-compute-cycles] ✅ 載入頻率記錄 ({metrics_filename}): {sclk:.1f} MHz")

            except Exception as e:
                print(f"[warn] 讀取 metrics 失敗: {e}")
        else:
            # 若無檔案，不報錯，繼續往下走 (可能沒開監控)
            pass

    # [2] 若無 Metrics，使用命令列傳入的保底值
    if actual_freq_ghz is None:
        actual_freq_ghz = default_freq_mhz / 1000.0
        print(f"[fix-compute-cycles] ⚠️ 無法讀取 metrics，使用設定頻率: {default_freq_mhz:.1f} MHz ({actual_freq_ghz:.3f} GHz)")

    avg_compute_cycles = 0

    # [新增] 策略 B：強制校準邏輯 (最高優先順序)
    if force_avg_kernel_ns is not None:
        print(f"========================================================")
        print(f"[fix-compute-cycles] ⚠️⚠️⚠️ 啟用系統感知強制校準 (System-Aware Calibration) ⚠️⚠️⚠️")
        print(f"[fix-compute-cycles] 目標: 將 System Overhead 攤提至計算節點")
        print(f"[fix-compute-cycles] 強制設定平均 Kernel 時間: {force_avg_kernel_ns} ns")

        estimated_cycles = int(force_avg_kernel_ns * actual_freq_ghz)
        avg_compute_cycles = estimated_cycles

        print(f"[fix-compute-cycles] 計算結果: {force_avg_kernel_ns:.0f} ns @ {actual_freq_ghz:.3f} GHz -> {estimated_cycles} cycles")
        print(f"========================================================")

        # 即使強制校準，我們直接跳到寫入階段，略過 Trace 分析
        # 但為了代碼結構，我們將 gpu_kernel_durations 設為空，讓後面邏輯不執行
        gpu_kernel_durations = []

    else:
        # [原有邏輯] 策略 A：基於 Trace 真實計算
        # 方法1：從 device trace 提取實際 GPU kernel timing
        gpu_kernel_durations = []
        if device_file and device_file.exists():
            try:
                with device_file.open('r') as f:
                    device_data = json.load(f)

                trace_events = device_data.get('traceEvents', [])
                for event in trace_events:
                    cat = event.get('cat', '')
                    name = event.get('name', '')
                    is_gpu_kernel = (
                        cat == 'cuda_runtime' and 'hipLaunchKernel' in name
                    ) or (
                        event.get('args', {}).get('kernel') is not None
                    )

                    if is_gpu_kernel:
                        raw_dur = event.get('dur')
                        # [單位修正] Trace 裡的單位是 us, 需轉為 ns
                        if raw_dur is not None and raw_dur > 0:
                            dur_ns = raw_dur * 1000.0
                            gpu_kernel_durations.append(dur_ns)

                if gpu_kernel_durations:
                    # 過濾掉小於 2000 ns (2 us) 的微小核心
                    valid_durations = [d for d in gpu_kernel_durations if d >= 2000]
                    if not valid_durations:
                        valid_durations = gpu_kernel_durations

                    # [統計] 使用平均值 (Mean)
                    avg_kernel_ns_val = sum(valid_durations) / len(valid_durations)

                    # 為了對比，印出中位數
                    valid_durations.sort()
                    median_ns = valid_durations[len(valid_durations) // 2]

                    print(f"[fix-compute-cycles] 掃描 {len(gpu_kernel_durations)} 個事件")
                    print(f"[fix-compute-cycles] 有效運算 Kernel: {len(valid_durations)} 個")
                    print(f"[fix-compute-cycles] 比較: 中位數={median_ns:.0f} ns vs 平均值={avg_kernel_ns_val:.0f} ns")
                    print(f"[fix-compute-cycles] Trace 分析結果 (Mean): {avg_kernel_ns_val:.0f} ns")

                    estimated_cycles = int(avg_kernel_ns_val * actual_freq_ghz)
                    avg_compute_cycles = estimated_cycles
                    print(f"[fix-compute-cycles] -> 估計 cycles: {estimated_cycles}")
                else:
                    print("[fix-compute-cycles] ⚠️ 未找到任何 Kernel，使用保底值 50us")
                    avg_compute_cycles = int(50000 * actual_freq_ghz)

            except Exception as e:
                print(f"[warn] 無法從 device trace 提取 GPU kernel timing: {e}")
                avg_compute_cycles = int(50000 * actual_freq_ghz)

        # 方法2：從 HDT 提取 ProfilerStep 時長（後備方案）
        # 只有在方法 1 (Device Trace) 失敗且沒有強制校準時才執行
        if not gpu_kernel_durations and avg_compute_cycles == 0:
            step_durations = {}
            try:
                with hdt_file.open('r') as f:
                    hdt_data = json.load(f)

                for node in hdt_data.get('nodes', []):
                    name = node.get('name', '')
                    if name.startswith('ProfilerStep#'):
                        step_num = int(name.replace('ProfilerStep#', ''))
                        dur = node.get('dur')
                        if dur is not None:
                            step_durations[step_num] = dur
            except Exception as e:
                print(f"[warn] 無法從 HDT 提取時長信息: {e}")

            if step_durations:
                avg_step_ms = sum(step_durations.values()) / len(step_durations)
                # 1ms = 1,000,000 ns
                avg_compute_cycles = int(avg_step_ms * 1000000 * actual_freq_ghz)
                print(f"[fix-compute-cycles] 從 HDT ProfilerStep 計算平均 compute_cycles: {avg_compute_cycles}")
            else:
                if avg_compute_cycles == 0: # 確保有值
                    avg_compute_cycles = int(1000000 * actual_freq_ghz)
                    print(f"[fix-compute-cycles] 使用預設 compute_cycles (1ms): {avg_compute_cycles}")

    # 添加屬性到 COMPUTE 節點
    meta, nodes = _decode_et(et_file)
    fixed_count = 0

    for node in nodes:
        if node.type == COMP_NODE:
            # 檢查是否已有 compute_cycles
            has_compute_cycles = any(a.name in ["compute_cycles", "exec_cycles", "cycles"] for a in node.attr)

            if not has_compute_cycles:
                # 添加 compute_cycles 屬性
                attr = node.attr.add()
                attr.name = "compute_cycles"
                attr.int64_val = int(avg_compute_cycles)
                fixed_count += 1

    # 保存修復後的 ET 文件
    _encode_et(et_file, meta, nodes)
    print(f"[fix-compute-cycles] 已為 {fixed_count} 個 COMPUTE 節點添加 compute_cycles")
    return fixed_count


# [MODIFIED] 傳入 default_freq_mhz
def convert_hdt_to_et(hdt: Path, out_et: Path, device_file: Path = None, pt_dir: Path = None, rank: int = 0, default_freq_mhz: float = 2400.0, force_avg_kernel_ns: float = None, tag: str = None) -> None:
    """
    將 HDT (Chakra 中間格式) 轉換為 ET (執行追蹤) 格式

    === 修補策略說明 ===
    由於 AMD GPU 的特殊 NCCL kernel 命名格式問題，我們需要在轉換前先修補
    Chakra 的 collective communication 識別邏輯。

    === 為什麼要直接調用而不是使用命令行 ===
    1. 命令行工具 (chakra_converter) 啟動新的 Python 進程
    2. 新進程不會繼承我們的動態修補 (monkey patch)
    3. 直接調用 Python API 可以確保修補在同一進程中生效
    4. 如果直接調用失敗，提供命令行工具作為備選方案

    === AMD GPU 修補增強 ===
    在轉換後額外確保 COMPUTE 節點具有 compute_cycles 屬性
    """
    print(f"[convert] {hdt.name}  ->  {out_et.name}")

    # === 第一步：應用 AMD GPU 修補 ===
    # 在轉換前先修補 AMD GPU 支援，這必須在同一 Python 進程中完成
    node_type_patch_success = apply_amd_gpu_patch()
    comm_type_patch_success = patch_collective_comm_type_for_amd()
    if not (node_type_patch_success and comm_type_patch_success):
        print("[warn] AMD GPU 修補失敗，可能在遇到 ncclDevKernel_Generic 時出錯")
        print("[warn] 如果您使用的是 AMD GPU，轉換可能會失敗")

    out_et.parent.mkdir(parents=True, exist_ok=True)

    # === 第二步：直接調用 Chakra 轉換器 API ===
    # 這是關鍵：直接調用而不是使用命令行工具，確保修補生效
    try:
        from chakra.src.converter.pytorch_converter import PyTorchConverter
        converter = PyTorchConverter()
        # simulate=False: 不進行模擬，只做轉換
        converter.convert(str(hdt), str(out_et), False)
        print("[ok] chakra_converter (直接調用) 完成\n")
    except Exception as e:
        print(f"[error] chakra_converter 直接調用失敗: {e}")
        print("[fallback] 嘗試使用命令行工具作為備選方案...")
        print("[warn] 注意：命令行工具不會受到我們的 AMD GPU 修補影響")

        # === 備選方案：命令行工具 ===
        cmd = ["chakra_converter", "PyTorch", "--input", str(hdt), "--output", str(out_et)]
        if not run_one(cmd):
            raise RuntimeError(f"chakra_converter 執行失敗，請檢查 HDT 檔案。")

    # === 第三步：AMD GPU 修補增強 ===
    # 確保 COMPUTE 節點具有 compute_cycles 屬性
    try:
        # [MODIFIED] 傳遞參數
        fixed_count = add_compute_cycles_to_compute_nodes(
            out_et,
            hdt,
            device_file,
            pt_dir=pt_dir,
            rank=rank,
            default_freq_mhz=default_freq_mhz,
            force_avg_kernel_ns=force_avg_kernel_ns, # 傳遞校準值
            tag=tag
        )
        if fixed_count > 0:
            print(f"[fix] 添加了 {fixed_count} 個 compute_cycles 屬性到 COMPUTE 節點")
    except Exception as e:
        print(f"[warn] compute_cycles 修復失敗: {e}")
        print("[info] 模擬可能仍會卡住，請檢查 COMPUTE 節點是否有 compute_cycles 屬性")


# --------------------------- 轉檔後：DAG 修正 ---------------------------

def _decode_et(et_path: Path) -> tuple[GlobalMetadata, list[Node]]:
    """安全解碼整個 .et（metadata + nodes）"""
    size = et_path.stat().st_size
    with et_path.open("rb") as f:
        meta = GlobalMetadata()
        _decode_msg(f, meta)
        nodes: list[Node] = []
        while f.tell() < size:
            n = Node()
            try:
                _decode_msg(f, n)
            except Exception:
                break
            nodes.append(n)
    return meta, nodes


def _encode_et(out_path: Path, meta: GlobalMetadata, nodes: list[Node]) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("wb") as f:
        _encode_msg(f, meta)
        for n in nodes:
            _encode_msg(f, n)


def _get_ctrl(n: Node) -> list[int]:
    if hasattr(n, "ctrl_deps"):
        return list(n.ctrl_deps)
    if hasattr(n, "parent"):
        return list(n.parent)
    return []


def _set_ctrl(n: Node, deps: list[int]) -> None:
    if hasattr(n, "ctrl_deps"):
        del n.ctrl_deps[:]; n.ctrl_deps.extend(deps)
    elif hasattr(n, "parent"):
        del n.parent[:]; n.parent.extend(deps)


def _get_data(n: Node) -> list[int]:
    """獲取節點的 data_deps"""
    if hasattr(n, "data_deps"):
        return list(n.data_deps)
    return []


def _set_data(n: Node, deps: list[int]) -> None:
    """設置節點的 data_deps"""
    if hasattr(n, "data_deps"):
        del n.data_deps[:]; n.data_deps.extend(deps)


def _clean_all_deps(n: Node, unsupported_ids: set[int]) -> tuple[int, int]:
    """
    清理節點的所有依賴類型，移除對不支援節點的引用

    Returns:
        (ctrl_deps_cleaned_count, data_deps_cleaned_count)
    """
    ctrl_cleaned = 0
    data_cleaned = 0

    # 清理 ctrl_deps
    ctrl_deps = _get_ctrl(n)
    if ctrl_deps:
        new_ctrl = [d for d in ctrl_deps if d not in unsupported_ids]
        if len(new_ctrl) != len(ctrl_deps):
            _set_ctrl(n, new_ctrl)
            ctrl_cleaned = len(ctrl_deps) - len(new_ctrl)

    # 清理 data_deps
    data_deps = _get_data(n)
    if data_deps:
        new_data = [d for d in data_deps if d not in unsupported_ids]
        if len(new_data) != len(data_deps):
            _set_data(n, new_data)
            data_cleaned = len(data_deps) - len(new_data)

    return ctrl_cleaned, data_cleaned


def fix_et_dag_inplace(et_file: Path, break_cycles: bool = True, astra_sim_compat: bool = True) -> dict[str, int]:
    """
    就地修正：移除自依賴與不存在的依賴；必要時斷開循環（back-edges）；
    並可選擇性地移除不支援的節點類型以兼容 ASTRA-sim。

    Args:
        et_file: ET 檔案路徑
        break_cycles: 是否斷開循環依賴
        astra_sim_compat: 是否進行 ASTRA-sim 兼容性過濾

    回傳：修正統計 {self_deps_removed, missing_deps_removed, cycles_removed, unsupported_nodes_removed}
    """
    meta, nodes = _decode_et(et_file)
    id_set = {n.id for n in nodes}
    stats = {"self_deps_removed": 0, "missing_deps_removed": 0, "cycles_removed": 0, "unsupported_nodes_removed": 0}

    # === ASTRA-sim 兼容性檢查 ===
    if astra_sim_compat:
        # 經過源碼與實務測試，ASTRA-sim 的 overlap 抽取僅對 GPU/COMM 類事件計算重疊。
        # 因此：將 metadata / process_group 等「不應進排程」的節點移除，並清理對它們的依賴。
        # 另外：對 COMM 類事件補齊必要欄位（group_id/comm_size），避免 size=0 或缺欄。
        unsupported_ids: set[int] = set()
        cleaned_nodes: list[Node] = []

        # 1) 判斷「不該進排程」的節點（PG/metadata）
        def _is_pg_or_metadata(nn: Node) -> bool:
            nm = (getattr(nn, "name", "") or "").lower()
            if "process_group" in nm or nm.startswith("## process_group"):
                return True
            # 以 Attribute 名稱判斷：含 pg_name/pg_desc/backend_config/group_size/ranks 等
            for a in nn.attr:
                if a.name in ("pg_name", "pg_desc", "backend_config", "group_size", "ranks"):
                    return True
            return False

        for n in nodes:
            if _is_pg_or_metadata(n):
                unsupported_ids.add(n.id)

        # 2) 先清理所有節點對「不支援節點」的依賴
        if unsupported_ids:
            for nn in nodes:
                _clean_all_deps(nn, unsupported_ids)

        # 3) 過濾：移除不支援節點
        for n in nodes:
            if n.id in unsupported_ids:
                stats["unsupported_nodes_removed"] += 1
                continue
            cleaned_nodes.append(n)
        nodes = cleaned_nodes

        # 4) 過濾：移除所有 CPU-side 的 COMP 節點（is_cpu_op == True）
        # ASTRA-sim 的 overlap 計算只接受 GPU 與 COMM，因此要避免留下 CPU COMP
        cpu_comp_ids: set[int] = set()
        remaining: list[Node] = []
        for n in nodes:
            if n.type == COMP_NODE:
                # 檢查是否有 is_cpu_op 屬性且為 True
                cpu_attr = next((a for a in n.attr if a.name == "is_cpu_op"), None)
                if cpu_attr is not None and getattr(cpu_attr, 'bool_val', False):
                    cpu_comp_ids.add(n.id)
                    stats["unsupported_nodes_removed"] += 1
                    continue
            remaining.append(n)
        if cpu_comp_ids:
            # 清理其他節點對這些 CPU COMP 節點的依賴
            for nn in remaining:
                _clean_all_deps(nn, cpu_comp_ids)
        nodes = remaining

        # 重新計算 id_set（後面的清理/斷循環依賴需要最新的 id 集合）
        id_set = {n.id for n in nodes}

        # 4) 對 COMM 類事件補欄位：group_id / comm_size（若缺值或 <=0）
        for n in nodes:
            # 偵測 COMM 的方式：優先看 comm_type 屬性，或檢查節點類型為 COMM_COLL_NODE
            has_comm_attr = any(a.name == "comm_type" for a in n.attr)
            is_comm_node = n.type in (COMM_COLL_NODE, COMM_SEND_NODE, COMM_RECV_NODE)
            if not (has_comm_attr or is_comm_node):
                # 不是通信節點，跳過補欄位
                continue

            # group_id 補 0（若缺）
            if not any(a.name in ("group_id", "comm_group_id", "pg_id") for a in n.attr):
                b = n.attr.add()
                b.name = "group_id"
                b.int64_val = 0

            # comm_type：為 COMM_COLL_NODE 添加預設的 ALL_REDUCE
            if n.type == COMM_COLL_NODE and not any(a.name == "comm_type" for a in n.attr):
                comm_type_attr = n.attr.add()
                comm_type_attr.name = "comm_type"
                comm_type_attr.int64_val = ALL_REDUCE  # 從匯入中獲得

            # comm_size：若缺或 <=0，嘗試從其它欄位回推；最後保底 1KB
            size_attr = next((a for a in n.attr if a.name == "comm_size"), None)
            size_val = size_attr.int64_val if size_attr is not None else None
            if size_val is None or size_val <= 0:
                alt = next(
                    (a.int64_val for a in n.attr
                     if a.name in ("size_bytes", "tensor_bytes", "message_size", "nccl_size")
                     and hasattr(a, "int64_val") and a.int64_val > 0),
                    None
                )
                if alt is None:
                    numel = next((a.int64_val for a in n.attr if a.name == "numel"), None)
                    el_sz = next((a.int64_val for a in n.attr if a.name in ("dtype_size", "element_size")), None)
                    alt = (numel * el_sz) if (numel and el_sz) else None
                if alt is None:
                    alt = 1024  # 最小保底，避免 0 破壞 ASTRA 的網路時間估測
                if size_attr is None:
                    size_attr = n.attr.add(); size_attr.name = "comm_size"
                size_attr.int64_val = int(alt)

            # is_cpu_op：通信節點必須在 GPU 上運行
            # 注意：由於我們已經在 apply_amd_gpu_patch 中確保 NCCL 操作被標記為 GPU 操作，
            # Chakra 的轉換器應該已經正確設置了 is_cpu_op=False。
            # 這裡我們檢查現有的 is_cpu_op 值，如果不正確則報告錯誤。
            if n.type in (COMM_COLL_NODE, COMM_SEND_NODE, COMM_RECV_NODE):
                cpu_attr = next((a for a in n.attr if a.name == "is_cpu_op"), None)
                if cpu_attr is not None:
                    if cpu_attr.bool_val == True:
                        print(f"[warn] 通信節點 {n.id} 的 is_cpu_op 為 True，這可能導致 ASTRA-sim 錯誤")
                else:
                    print(f"[warn] 通信節點 {n.id} 缺少 is_cpu_op 屬性")
    # 去自依賴 & 移除不存在依賴（針對 ctrl_deps）
    for n in nodes:
        deps = _get_ctrl(n)
        nd = []
        for d in deps:
            if d == n.id:
                stats["self_deps_removed"] += 1
                continue
            if d not in id_set:
                stats["missing_deps_removed"] += 1
                continue
            nd.append(d)
        if len(nd) != len(deps):
            _set_ctrl(n, nd)

    # 同樣處理 data_deps
    for n in nodes:
        deps = _get_data(n)
        if deps:
            nd = []
            for d in deps:
                if d == n.id:
                    stats["self_deps_removed"] += 1
                    continue
                if d not in id_set:
                    stats["missing_deps_removed"] += 1
                    continue
                nd.append(d)
            if len(nd) != len(deps):
                _set_data(n, nd)

    if break_cycles:
        # dep -> node 邊（以 ctrl 依賴構圖）
        id_to_node = {n.id: n for n in nodes}
        out_edges: dict[int, list[int]] = {nid: [] for nid in id_set}
        for nn in nodes:
            for d in _get_ctrl(nn):
                out_edges.setdefault(d, []).append(nn.id)

        WHITE, GRAY, BLACK = 0, 1, 2
        color: dict[int, int] = {nid: WHITE for nid in id_set}

        def dfs(u: int):
            color[u] = GRAY
            for v in list(out_edges.get(u, [])):
                if color[v] == WHITE:
                    dfs(v)
                elif color[v] == GRAY:
                    # back-edge u->v：移除 v 的 ctrl_deps 中的 u
                    cn = id_to_node[v]
                    deps = _get_ctrl(cn)
                    if u in deps:
                        _set_ctrl(cn, [x for x in deps if x != u])
                        stats["cycles_removed"] += 1
            color[u] = BLACK

        for nid in list(id_set):
            if color[nid] == WHITE:
                dfs(nid)

    # 覆寫檔案
    _encode_et(et_file, meta, nodes)
    return stats


# --------------------------- ASTRA-sim 兼容版本生成 ---------------------------

def extract_comm_nodes_from_hdt(hdt_file: Path) -> List[Dict]:
    """從 HDT JSON 檔案中提取通訊相關節點"""
    with hdt_file.open('r') as f:
        data = json.load(f)

    nodes = data.get('nodes', [])
    comm_nodes = []

    for node in nodes:
        name = node.get('name', '').lower()
        # 識別通訊相關節點
        if any(keyword in name for keyword in [
            'nccl', 'allreduce', 'all_reduce', 'comm',
            'c10d::allreduce', 'nccldevkernel'
        ]):
            comm_nodes.append(node)

    print(f"[comm-extract] 從 {hdt_file.name} 找到 {len(comm_nodes)} 個通訊節點")
    return comm_nodes


def extract_comm_size_from_node(node: Dict) -> int:
    """從節點中提取通訊大小（以字節為單位）"""
    # 檢查 inputs 中的張量大小
    inputs = node.get('inputs', {})
    values = inputs.get('values', [])
    total_size = 0

    for value in values:
        if isinstance(value, list) and len(value) >= 4:
            try:
                # 計算張量大小 (假設 Float32 = 4 bytes)
                shape_elements = 1
                for dim in value[:-2]:  # 除了最後兩個元素（dtype, device）
                    if isinstance(dim, int):
                        shape_elements *= dim
                tensor_size = shape_elements * 4
                total_size += tensor_size
            except (ValueError, TypeError):
                total_size += 1024 * 1024  # 1MB 預設

    # 檢查屬性中是否有明確的大小信息
    attrs = node.get('attrs', [])
    for attr in attrs:
        if attr.get('name') == 'comm_size':
            return int(attr.get('value', total_size))

    return max(total_size, 1024 * 1024)  # 至少 1MB


def create_astra_sim_et(comm_nodes: List[Dict], output_file: Path, rank: int) -> None:
    """創建 ASTRA-sim 兼容的簡化 ET 檔案"""
    print(f"[astra-et] 創建簡化 ET 檔案: {output_file.name}")

    output_file.parent.mkdir(parents=True, exist_ok=True)

    with output_file.open("wb") as et:
        # 1. 寫入 metadata
        metadata = GlobalMetadata(version="0.0.4")
        _encode_msg(et, metadata)

        # 2. 為每個通訊操作創建簡化節點
        for i, comm_node in enumerate(comm_nodes):
            node = Node()
            node.id = i
            node.name = f"AMD_GPU_COMM_{rank}_{i}"
            node.type = COMM_COLL_NODE

            # 添加必要屬性
            node.attr.append(AttributeProto(name="is_cpu_op", bool_val=False))
            node.attr.append(AttributeProto(name="comm_type", int64_val=ALL_REDUCE))

            # 提取並設定通訊大小
            comm_size = extract_comm_size_from_node(comm_node)
            node.attr.append(AttributeProto(name="comm_size", int64_val=comm_size))

            # 設定必要的依賴關係（簡化版本：每個節點依賴前一個）
            if i > 0:
                node.data_deps.append(i - 1)

            _encode_msg(et, node)
            print(f"  [node] {i}: {comm_node.get('name', 'unknown')} -> 大小 {comm_size} bytes")

    print(f"[astra-et] ✅ 簡化 ET 檔案完成: {output_file}")


def map_comm_sizes_from_hdt_to_et(hdt_file: Path, et_file: Path) -> dict:
    """
    將 HDT 中抽出的 comm node sizes 寫回到已存在的 .et 檔案中的 COMM 節點。

    返回統計：{'updated': n_updated, 'before_total': bytes_before, 'after_total': bytes_after}
    此函式嘗試按順序對齊 HDT 的 comm_nodes 與 ET 中的 COMM_* 節點；若數量不同，會退回到名稱匹配的 best-effort。
    """
    if not hdt_file.exists() or not et_file.exists():
        raise FileNotFoundError(f"HDT or ET not found: {hdt_file}, {et_file}")

    comm_nodes = extract_comm_nodes_from_hdt(hdt_file)
    meta, nodes = _decode_et(et_file)

    # 找到 ET 中的通訊節點
    et_comm_indices = [i for i, n in enumerate(nodes) if n.type in (COMM_COLL_NODE, COMM_SEND_NODE, COMM_RECV_NODE)]
    before_total = 0
    for idx in et_comm_indices:
        n = nodes[idx]
        sa = next((a for a in n.attr if a.name == 'comm_size'), None)
        if sa is not None and hasattr(sa, 'int64_val'):
            before_total += int(sa.int64_val)

    updated = 0

    # Build helper data structures
    # HDT comm entries: derive bytes and optional time interval
    hdt_entries = []
    for hn in comm_nodes:
        entry = {}
        entry['name'] = (hn.get('name') or '').lower()
        entry['bytes'] = extract_comm_size_from_node(hn)
        # optional time bounds (if available in HDT)
        entry['start'] = hn.get('ts') or hn.get('start') or None
        entry['end'] = hn.get('end') or hn.get('duration') and ((hn.get('ts') or 0) + hn.get('duration')) or None
        entry['raw'] = hn
        entry['assigned'] = 0
        hdt_entries.append(entry)

    et_entries = []
    for idx in et_comm_indices:
        n = nodes[idx]
        ename = (getattr(n, 'name', '') or '').lower()
        esize = next((a.int64_val for a in n.attr if a.name == 'comm_size' and hasattr(a, 'int64_val')), 0)
        # attempt to read timestamps if present (start/end)
        estart = None
        eend = None
        # some schemas might store ts/duration on attributes
        ts_attr = next((a for a in n.attr if a.name in ('ts', 'time', 'start_ts')), None)
        dur_attr = next((a for a in n.attr if a.name in ('duration', 'dur', 'elapsed')), None)
        if ts_attr is not None and hasattr(ts_attr, 'int64_val'):
            estart = int(ts_attr.int64_val)
        if dur_attr is not None and hasattr(dur_attr, 'int64_val') and estart is not None:
            eend = estart + int(dur_attr.int64_val)
        et_entries.append({'idx': idx, 'name': ename, 'size': int(esize or 0), 'start': estart, 'end': eend, 'node': n, 'assigned': 0})

    # Strategy:
    # 1) Exact / substring name matches (high confidence)
    # 2) Time overlap matches (if timestamps available)
    # 3) Greedy bytes assignment: allocate remaining HDT bytes to unassigned ET nodes by best name score

    provenance = []

    # 1) Name exact/substring matching
    for he in hdt_entries:
        if he['bytes'] <= 0:
            continue
        for ee in et_entries:
            if ee['assigned']:
                continue
            if not he['name'] or not ee['name']:
                continue
            if he['name'] == ee['name'] or he['name'] in ee['name'] or ee['name'] in he['name']:
                # assign full bytes
                assign = int(he['bytes'])
                ee['assigned'] = assign
                he['assigned'] = assign
                # set attr
                a = next((a for a in ee['node'].attr if a.name == 'comm_size'), None)
                if a is None:
                    a = ee['node'].attr.add(); a.name = 'comm_size'
                a.int64_val = assign
                updated += 1
                provenance.append({'method': 'name', 'hdt_name': he['name'], 'et_name': ee['name'], 'bytes': assign, 'et_idx': ee['idx']})
                break

    # 2) Time overlap matching
    for he in hdt_entries:
        if he['assigned']:
            continue
        if he.get('start') is None or he.get('end') is None:
            continue
        for ee in et_entries:
            if ee['assigned']:
                continue
            if ee.get('start') is None or ee.get('end') is None:
                continue
            # check overlap
            if not (he['end'] < ee['start'] or he['start'] > ee['end']):
                assign = int(he['bytes'])
                ee['assigned'] = assign
                he['assigned'] = assign
                a = next((a for a in ee['node'].attr if a.name == 'comm_size'), None)
                if a is None:
                    a = ee['node'].attr.add(); a.name = 'comm_size'
                a.int64_val = assign
                updated += 1
                provenance.append({'method': 'time', 'hdt_name': he['name'], 'et_name': ee['name'], 'bytes': assign, 'et_idx': ee['idx']})
                break

    # 3) Greedy bytes assignment for remaining HDT entries -> distribute to remaining ET nodes by name-similarity score
    # prepare list of remaining ETs and remaining HDTs
    remaining_ets = [ee for ee in et_entries if not ee['assigned']]
    remaining_hdts = [he for he in hdt_entries if not he['assigned']]

    def name_score(a: str, b: str) -> int:
        # simple heuristic: common substring length
        if not a or not b:
            return 0
        # use longest common substring (O(n^2) but strings are short)
        best = 0
        for i in range(len(a)):
            for j in range(i+1, len(a)+1):
                sub = a[i:j]
                if sub and sub in b:
                    best = max(best, len(sub))
        return best

    for he in remaining_hdts:
        if he['bytes'] <= 0:
            continue
        # score remaining ET nodes
        scores = [(name_score(he['name'], ee['name']), ee) for ee in remaining_ets]
        scores.sort(key=lambda x: x[0], reverse=True)
        if not scores:
            continue
        best_score, best_ee = scores[0]
        # if best_score == 0, fall back to proportional distribution
        if best_score > 0:
            assign = int(he['bytes'])
            best_ee['assigned'] = assign
            he['assigned'] = assign
            a = next((a for a in best_ee['node'].attr if a.name == 'comm_size'), None)
            if a is None:
                a = best_ee['node'].attr.add(); a.name = 'comm_size'
            a.int64_val = assign
            updated += 1
            provenance.append({'method': 'greedy_name', 'hdt_name': he['name'], 'et_name': best_ee['name'], 'bytes': assign, 'et_idx': best_ee['idx'], 'score': best_score})
        else:
            # proportional distribution: split he['bytes'] among remaining_ets evenly
            if not remaining_ets:
                continue
            per = int(he['bytes'] // len(remaining_ets))
            rem = int(he['bytes']) - per * len(remaining_ets)
            for i, ee in enumerate(remaining_ets):
                assign = per + (1 if i == 0 else 0) if rem else per
                ee['assigned'] = assign
                # update node
                a = next((a for a in ee['node'].attr if a.name == 'comm_size'), None)
                if a is None:
                    a = ee['node'].attr.add(); a.name = 'comm_size'
                a.int64_val = assign
                updated += 1
                provenance.append({'method': 'proportional', 'hdt_name': he['name'], 'et_name': ee['name'], 'bytes': assign, 'et_idx': ee['idx']})

    # compute after_total
    after_total = 0
    for idx in et_comm_indices:
        n = nodes[idx]
        sa = next((a for a in n.attr if a.name == 'comm_size'), None)
        if sa is not None and hasattr(sa, 'int64_val'):
            after_total += int(sa.int64_val)

    # 覆寫 ET
    _encode_et(et_file, meta, nodes)

    # write provenance sidecar into runs/mapping/ for centralized storage (non-destructive)
    try:
        runs_mapping_dir = Path('runs') / 'mapping'
        runs_mapping_dir.mkdir(parents=True, exist_ok=True)
        # name provenance by et filename + rank if possible
        et_basename = et_file.stem
        prov_path = runs_mapping_dir / f"{et_basename}.mapping.json"
        with prov_path.open('w') as pf:
            json.dump({'provenance': provenance, 'hdt_file': str(hdt_file), 'et_file': str(et_file)}, pf, indent=2)
    except Exception:
        # fallback: write next to et_file
        try:
            prov_path = et_file.with_suffix(et_file.suffix + '.mapping.json')
            with prov_path.open('w') as pf:
                json.dump({'provenance': provenance, 'hdt_file': str(hdt_file), 'et_file': str(et_file)}, pf, indent=2)
        except Exception:
            pass

    print(f"[map-comm] {et_file.name}: updated={updated} before={before_total} bytes after={after_total} bytes")
    return {'updated': updated, 'before_total': before_total, 'after_total': after_total}


# --------------------------- [NEW] 通訊模式替換功能 ---------------------------

def override_comm_type_inplace(et_file: Path, target_pattern: str) -> int:
    """
    [Synthetic Workload]
    強制將 ET 中的集合通訊操作替換為指定的模式 (例如 All-to-All)。
    用於生成網路壓力測試的合成負載。

    Args:
        et_file: ET 檔案路徑
        target_pattern: 目標模式 ('all_to_all')

    Returns:
        替換的節點數量
    """
    print(f"[override] 正在將 {et_file.name} 的通訊模式替換為: {target_pattern.upper()}")

    target_enum = None
    if target_pattern == "all_to_all":
        target_enum = ALL_TO_ALL
    else:
        print(f"[override] 未知的模式 {target_pattern}，跳過替換")
        return 0

    meta, nodes = _decode_et(et_file)
    replaced_count = 0

    for n in nodes:
        # 檢查是否為集合通訊節點
        if n.type == COMM_COLL_NODE:
            # 尋找 comm_type 屬性
            comm_attr = None
            for attr in n.attr:
                if attr.name == "comm_type":
                    comm_attr = attr
                    break

            # 如果屬性存在且原本是 ALL_REDUCE，則替換
            # (我們只替換 AllReduce，保留 Broadcast/AllGather 等其他操作，除非這就是目的)
            if comm_attr:
                current_val = comm_attr.int64_val
                if current_val == ALL_REDUCE:
                    comm_attr.int64_val = target_enum
                    replaced_count += 1
            else:
                # 如果沒有 comm_type，強制添加並設為目標
                new_attr = n.attr.add()
                new_attr.name = "comm_type"
                new_attr.int64_val = target_enum
                replaced_count += 1

    if replaced_count > 0:
        _encode_et(et_file, meta, nodes)
        print(f"[override] ✅ 成功將 {replaced_count} 個節點轉換為 {target_pattern.upper()}")
    else:
        print(f"[override] ⚠️ 沒有發現可替換的 AllReduce 節點")

    return replaced_count


# --------------------------- 主流程 ---------------------------

def main():
    ap = argparse.ArgumentParser(description="Link PyTorch host/device -> HDT, then convert to Chakra ET (.et)")
    ap.add_argument("--base-dir", default="./data/chakra", help="根目錄（預設：./data/chakra）")
    ap.add_argument("--pt-dir",   default="pytorch_traces", help="host_*.json / device_*.json 與 hdt_*.json 目錄（相對 base-dir）")
    ap.add_argument("--et-dir",   default="workload_et",    help="輸出 .et 目錄（相對 base-dir）")
    ap.add_argument("--et-prefix",default="et",             help="輸出 .et 檔名前綴（可用 'auto' 自動偵測）")
    ap.add_argument("--ranks", nargs="*", help="只處理指定 ranks（預設自動偵測）")
    ap.add_argument("--no-clean", action="store_true", help="不要清空舊的 hdt_*.json 與 *.et")
    ap.add_argument("--no-force", action="store_true", help="不要覆寫已存在的 .et")
    ap.add_argument("--simple-astra", action="store_true", help="生成 ASTRA-sim 兼容的簡化版本（僅通訊節點）")
    # [MODIFIED] 新增參數，預設 2400 MHz (2.4 GHz)
    ap.add_argument("--default-gpu-freq", type=float, default=2400.0, help="當 metrics 不存在時的預設 GPU 頻率 (MHz)")
    # [新增] 支援強制校準與模型標籤
    ap.add_argument("--force-avg-kernel-ns", type=float, default=None, help="System-Aware Calibration: 強制指定平均 Kernel 時間 (ns)")
    ap.add_argument("--model-tag", type=str, default=None, help="指定模型標籤 (e.g., cifar10, resnet50) 用於過濾檔案與命名輸出")
    # [新增] 壓力測試參數
    ap.add_argument("--replace-comm", type=str, default=None, help="[壓力測試] 強制替換通訊模式 (例如: all_to_all)")

    args = ap.parse_args()

    base   = Path(args.base_dir).resolve()
    pt_dir = (base / args.pt_dir).resolve()
    et_dir = (base / args.et_dir).resolve()

    # --- [新增] Log 設定 ---
    log_dir = base / "log"
    log_dir.mkdir(parents=True, exist_ok=True)

    # 使用時間戳記建立 log 檔名，例如: convert_20251125_143000.log
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    log_file = log_dir / f"convert_{timestamp}.log"

    # 重導向 stdout 和 stderr
    # 這樣所有的 print() 和錯誤訊息都會同時出現在螢幕和檔案中
    sys.stdout = Logger(log_file)
    sys.stderr = sys.stdout

    print(f"========================================================")
    print(f"[Log] Log 記錄已啟動: {log_file}")
    if args.replace_comm:
        print(f"[Stress Test] ⚠️ 啟用通訊模式替換: ALL_REDUCE -> {args.replace_comm.upper()}")
    print(f"========================================================\n")
    # -----------------------
    if args.model_tag:
        print(f"[Config] Processing Model Tag: {args.model_tag}")
    print(f"[Config] Force Calibration: {args.force_avg_kernel_ns} ns" if args.force_avg_kernel_ns else "[Config] Trace-based Cycle Calculation")
    print(f"[paths]\n  base={base}\n  pt_dir={pt_dir}\n  et_dir={et_dir}\n")

    # 檢查工具是否存在
    if shutil.which("chakra_trace_link") is None:
        print("[warn] 找不到 chakra_trace_link（請確認已安裝且在 PATH）")
    if shutil.which("chakra_converter") is None:
        print("[warn] 找不到 chakra_converter（請確認已安裝且在 PATH）")

    # [修改] 若指定 tag 則不全域清理，避免誤刪其他實驗數據
    if not args.no_clean and not args.model_tag:
        clean_outputs(pt_dir, et_dir)
    elif args.model_tag:
        print(f"[info] 指定了 --model-tag {args.model_tag}，跳過全域清理")

    # 決定要處理哪些 rank
    ranks = detect_ranks(pt_dir, args.ranks, tag=args.model_tag)
    print(f"[ranks] 將處理：{ranks}\n")

    force = not args.no_force

    for r in ranks:
        # [修改] 組合檔名 (支援有 tag 與無 tag)
        suffix = f"_{args.model_tag}" if args.model_tag else ""

        # 優先找帶有 tag 的檔案 (例如 host_0_resnet50.json)
        host = pt_dir / f"host_{r}{suffix}.json"
        dev  = pt_dir / f"device_{r}{suffix}.json"

        # Fallback: 若找不到帶後綴的，找無後綴的 (相容舊檔 host_0.json)
        if not host.exists() and not suffix: host = pt_dir / f"host_{r}.json"
        if not dev.exists() and not suffix: dev = pt_dir / f"device_{r}.json"

        if not host.exists() or not dev.exists():
            print(f"[skip] rank {r} 檔案不完整 (找不到 host/device trace)")
            continue

        # HDT 放在 pytorch_traces 內
        hdt = pt_dir / f"hdt_{r}{suffix}.json"

        # 決定輸出的 et 檔名
        # [修改] 輸出檔名也加上 tag (例如 workload.resnet50.0.et)
        prefix = infer_prefix_from_device(dev) if args.et_prefix == "auto" else args.et_prefix
        final_prefix = f"{prefix}.{args.model_tag}" if args.model_tag else prefix

        # [新增] 若啟用替換模式，檔名加上後綴
        if args.replace_comm == "all_to_all":
            final_prefix += "_all2all"

        et  = et_dir / f"{final_prefix}.{r}.et"

        if et.exists() and not force:
            print(f"[skip] {et} 已存在。用 --no-force 可維持、或移除該檔再重跑。")
            continue

        # link → convert
        link_host_device(r, host, dev, hdt)

        # 根據模式選擇處理方式
        if args.simple_astra:
            # ASTRA-sim 兼容模式：直接從 HDT 生成簡化 ET
            print(f"[simple-astra] 生成 ASTRA-sim 兼容版本 for rank {r}")
            comm_nodes = extract_comm_nodes_from_hdt(hdt)
            if comm_nodes: create_astra_sim_et(comm_nodes, et, r)
        else:
            # 標準模式：完整轉換 + DAG 修正
            convert_hdt_to_et(hdt, et, dev, pt_dir=pt_dir, rank=r, default_freq_mhz=args.default_gpu_freq,
                              force_avg_kernel_ns=args.force_avg_kernel_ns,
                              tag=args.model_tag)

            # 轉檔後：就地修 DAG（清理不一致依賴/循環，並過濾 ASTRA-sim 不支援的節點）
            try:
                stats = fix_et_dag_inplace(et, break_cycles=True, astra_sim_compat=True)
                print(f"[post-fix] {et.name}: self={stats['self_deps_removed']}, "
                      f"missing={stats['missing_deps_removed']}, cycles={stats['cycles_removed']}, "
                      f"unsupported={stats['unsupported_nodes_removed']}")
            except Exception as e:
                print(f"[post-fix] DAG 修正失敗（{et.name}）：{e}")

            # [新增] 執行通訊模式替換
            if args.replace_comm:
                try:
                    override_comm_type_inplace(et, args.replace_comm)
                except Exception as e:
                    print(f"[override] 通訊模式替換失敗: {e}")

    print("\n[done] 轉換完成。請到以下資料夾查看輸出：", et_dir)
    print(f"[Log] 完整記錄已儲存至: {log_file}")


if __name__ == "__main__":
    main()
