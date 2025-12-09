# ROCm ASTRA-sim：分散式深度學習網路模擬框架

本專案提供一個完整的工作流程，用於在 AMD ROCm GPU 環境下，從 PyTorch 分散式訓練中擷取性能追蹤，並轉換為 ASTRA-sim ns-3 網路模擬器可用的格式，以進行大規模網路拓撲的效能分析與研究。

## 核心功能

*   **端到端工作流程**: 從 PyTorch/ROCm 訓練到 ASTRA-sim/ns-3 網路模擬的完整解決方案。
*   **模型支援**: 內建支援 **CIFAR-10 (System-Bound)** 與 **ResNet-50 (Compute-Bound)** 兩種典型工作負載。
*   **AMD ROCm 優化**:
    *   透過 `rocm_compat.py` 監控真實 GPU 頻率。
    *   在 `conver_to_chakra_et.py` 中動態修補 Chakra 以兼容 AMD GPU 的 trace 格式。
*   **系統感知校準 (System-Aware Calibration)**:
    *   專為 System-Bound 工作負載設計，透過 `--force-avg-kernel-ns` 將系統開銷（如 Kernel Launch Latency）攤提到計算節點，使模擬時間更貼近真實世界。
*   **自動化模擬與校準**:
    *   `run_ns3.py` 自動化拓撲生成、參數配置、虛擬擴展與模擬執行。
    *   自動比對真實 trace 時間與模擬 cycles，計算校準因子 `alpha_us`，並彙總至 `runs/calibration_all.csv`。

## 專案架構

```
.
├── rocm/
│   └── dockerfile          # Docker 環境定義 (ROCm + PyTorch + ASTRA-sim)
├── src/
│   ├── train_rocm_pytorch.py  # [階段 1] PyTorch 分散式訓練 + Kineto 追蹤生成
│   ├── conver_to_chakra_et.py # [階段 2] Trace 轉換：JSON -> HDT -> Chakra ET
│   └── rocm_compat.py         # ROCm 監控與兼容性工具
├── scripts/
│   └── run_ns3.py          # [階段 3] ASTRA-sim ns-3 模擬執行與校準
├── configs/                # ASTRA-sim 基準設定檔
├── data/
│   ├── chakra/
│   │   ├── pytorch_traces/ # (輸入) PyTorch Kineto 原始追蹤 (*.json)
│   │   ├── gpu_metrics/    # (輸入) 訓練期間的 GPU 頻率紀錄
│   │   └── workload_et/   # (輸出) Chakra ET 檔案 (*.et)
│   └── cifar10/           # 訓練資料集
├── runs/                   # 模擬結果與校準數據庫
└── tutorials/              # 教學範例
```

## 三階段工作流程

### 階段 1：生成訓練追蹤 (`train_rocm_pytorch.py`)

此腳本在 ROCm 環境下執行 PyTorch 分散式訓練，並使用 `torch.profiler` 生成 Kineto 格式的 host/device trace。

**常用指令**:

```bash
# 生成 CIFAR-10 (System-Bound) 的 Trace
# --model-tag 用於標記輸出檔案，方便後續階段識別
torchrun --standalone --nproc_per_node=2 ./src/train_rocm_pytorch.py \
  --model cifar10 --workers 0 \
  --trace-wait 32 --trace-steps 4 \
  --model-tag cifar10

# 生成 ResNet-50 (Compute-Bound) 的 Trace
torchrun --standalone --nproc_per_node=2 ./src/train_rocm_pytorch.py \
  --model resnet50 --workers 4 \
  --trace-wait 32 --trace-steps 2 \
  --model-tag resnet50
```

**輸出**:
*   `data/chakra/pytorch_traces/host_0_cifar10.json`, `device_0_cifar10.json`, ...
*   `data/chakra/gpu_metrics/gpu_metrics_0_cifar10.json`, ...

**提示**:
*   `--inject-sync-hack`: 建議在 ROCm 環境下開啟此選項，它透過注入額外同步事件來解決 `chakra_trace_link` 可能發生的 CPU/GPU 時間軸對不齊問題，提升 trace 連結成功率。

### 階段 2：轉換 Trace 為 Chakra ET (`conver_to_chakra_et.py`)

此腳本將 Kineto JSON trace 轉換為 ASTRA-sim 使用的 Chakra ET (`.et`) 格式，並包含 AMD GPU 兼容性修補。

**常用指令**:

```bash
# 轉換 CIFAR-10 (啟用系統感知校準)
# --force-avg-kernel-ns 將開銷攤提到計算節點，使模擬更真實
python ./src/conver_to_chakra_et.py \
  --model-tag cifar10 \
  --force-avg-kernel-ns 609000

# 轉換 ResNet-50 (標準模式)
# 不需攤提，直接依賴 trace 中的 kernel 時間
python ./src/conver_to_chakra_et.py --model-tag resnet50
```

**輸出**:
*   `data/chakra/workload_et/workload.cifar10.0.et`, ...
*   `data/chakra/workload_et/workload.resnet50.0.et`, ...

### 階段 3：執行網路模擬與校準 (`run_ns3.py`)

此腳本是模擬流程的總指揮，負責配置、執行與分析。

**常用指令**:

```bash
# [校準] 執行 2-GPU CIFAR-10 模擬，並自動校準 alpha_us
# 腳本會自動尋找 pytorch_traces/*_cifar10.json 以獲取真實時間
python ./scripts/run_ns3.py \
  --workload data/chakra/workload_et --model-tag cifar10 \
  --topo auto:1d \
  --phys-topo configs/astra-sim/topos/2_nodes_1_switch_topology.txt

# [模擬] 將 2-GPU ResNet-50 虛擬擴展到 128-GPU，並在 3D Mesh 拓撲上模擬
# 大規模模擬通常不進行校準 (--no-autocalib)
python ./scripts/run_ns3.py \
  --workload data/chakra/workload_et --model-tag resnet50 \
  --virtual-world 128 \
  --topo auto:3d \
  --phys-topo configs/astra-sim/topos/128_nodes_32_switch_topology.txt \
  --no-autocalib
```

**輸出**:
*   `runs/<timestamp>_*_ns3_run/`: 包含 `stdout.log` 和 `out/metrics.csv` 的詳細執行結果。
*   `runs/calibration_all.csv`: 所有校準運行的歷史紀錄資料庫，包含 `alpha_us` 等關鍵指標。

---

## 🛠️ 工具深度解析

### `train_rocm_pytorch.py`
此工具的核心是在 ROCm 環境下進行 PyTorch 分散式訓練，並精確地擷取性能追蹤。

*   **關鍵參數**:
    *   `--model`: 選擇 `cifar10` (System-Bound) 或 `resnet50` (Compute-Bound)。
    *   `--workers`: DataLoader 的工作執行緒數。設為 `0` 可放大 CPU System Overhead，用於研究 System-Aware Calibration。
    *   `--trace-steps`: 指定要追蹤的訓練步數。建議設為 `1-4` 以避免 trace 檔案過大。
    *   `--model-tag`: 為輸出檔案（trace, gpu_metrics）加上標籤，便於管理不同模型的實驗。
    *   `--inject-sync-hack`: 透過注入額外同步事件，提高 ROCm 上 trace 連結的穩定性。

### `conver_to_chakra_et.py`
此工具負責將 PyTorch Kineto trace 轉換為 ASTRA-sim 相容的 Chakra ET 格式。

*   **關鍵參數**:
    *   `--model-tag`: 讀取對應 tag 的 trace 檔案進行轉換。
    *   `--force-avg-kernel-ns`: **系統感知校準**的關鍵。此參數會強制設定一個平均的 Kernel 執行時間（奈秒），將真實世界的系統開銷攤提到計算節點上。
    *   `--default-gpu-freq`: 當 `gpu_metrics` 檔案不存在時，使用的預設 GPU 頻率。

### `scripts/run_ns3.py`
此腳本是整個模擬流程的啟動器與控制器。

*   **關鍵參數**:
    *   `--workload` & `--model-tag`: 指定要模擬的 `.et` 工作負載。
    *   `--virtual-world`: 將少量 GPU 的 trace（例如 2-GPU）虛擬擴展成大規模叢集（例如 128-GPU），自動調整通訊量。
    *   `--topo` & `--phys-topo`: 分別定義 ASTRA-sim 的邏輯拓撲與 ns-3 的物理拓撲。
    *   `--no-autocalib`: 禁用自動校準。在大規模虛擬擴展模擬時建議開啟。
    *   `--calib-db`: 指定儲存所有校準結果的 CSV 檔案路徑。

## 🔧 進階配置

### Docker 環境
若要使用特定版本的 ROCm 或 PyTorch，可以在啟動 `docker-compose` 時傳入環境變數：
```bash
# 範例：使用 ROCm 6.1 和 PyTorch 2.3.0
VERSION=rocm6.1_ubuntu22.04_py3.10_pytorch_2.3.0 docker-compose up
```

### ASTRA-sim 配置
所有 ASTRA-sim 的基準設定檔都位於 `configs/astra-sim/` 目錄下。`run_ns3.py` 在執行時會讀取這些檔案，並根據命令列參數（如 `--coll-opt`, `--buffer` 等）在 `runs/<timestamp>/tmp/` 目錄下生成一個 patch 過的版本，而不會修改原始設定檔。

*   `system/system.json`: 定義系統層行為，如 collective optimization 策略。
*   `ns3/config.txt`: 定義 ns-3 網路層參數，如 PFC、QCN、緩衝區大小等。
*   `topos/*.txt`: ns-3 使用的物理拓撲檔案。

## 🐛 常見問題 (FAQ)

**Q: 訓練追蹤檔案 (`.json`) 為空或不完整？**
A: 這通常是因為 `torch.profiler` 沒有足夠的時間來 warm-up 或擷取事件。請確保 `--trace-wait` 和 `--trace-steps` 的值足夠大。對於快速的迭代，`--trace-wait 32 --trace-steps 4` 是一個好的起點。

**Q: 在 ROCm 上執行 `conver_to_chakra_et.py` 時，`chakra_trace_link` 失敗？**
A: 這很可能是因為 CPU 和 GPU 的時間戳無法對齊。請在執行 `train_rocm_pytorch.py` 時加上 `--inject-sync-hack` 參數，這有助於提高連結成功率。

**Q: `run_ns3.py` 執行時出現 "Node ... not found in index" 錯誤？**
A: 這個錯誤通常表示 `.et` 檔案的格式與 ASTRA-sim feeder 的版本不相容。請確認您使用的 Chakra 版本與 ASTRA-sim 的版本是匹配的。可以嘗試使用 `src/tests/validate_et.py` 進行基本格式檢查。

## 🧪 測試與驗證

專案包含一系列測試腳本，以確保環境配置正確且各個工具鏈階段功能正常。

```bash
# 檢查 Python 環境與 Chakra/HTA 版本
python ./src/tests/check_version.py

# 檢查生成的 PyTorch trace 是否包含必要事件，適合轉換
python ./src/tests/check_trace_ready.py

# 驗證轉換後的 .et 檔案格式是否基本正確
python ./src/tests/validate_et.py
```

## 📚 教學範例

`tutorials/` 資料夾中包含了多個基於學術會議的完整教學範例，提供更深入的應用場景和練習。

探索 `tutorials/` 目錄中的完整範例：
- **hoti2024/**: HOT Interconnects 2024 示範
- **micro2024/**: MICRO 2024 研討會材料
- **asplos2023/**: ASPLOS 2023 練習

建議使用者在熟悉三階段工作流程後，進一步探索這些教學內容。
