# ROCm ASTRA-sim 深度學習網路模擬環境

一個完整的分散式深度學習網路模擬環境，支援從 PyTorch 訓練追蹤到 ASTRA-sim 網路模擬的完整工作流程。

## 📁 專案架構

```
.
├── rocm/                    # Docker 環境定義
│   └── dockerfile          # ROCm + PyTorch + Horovod + ASTRA-sim + HTA
├── src/                     # 核心工具
│   ├── train_rocm_pytorch.py      # PyTorch 分散式訓練 + 性能追蹤
│   ├── conver_to_chakra_et.py     # 追蹤轉換：JSON → HDT → ET
│   ├── rocm_compat.py             # ROCm 兼容性工具
│   └── tests/                     # 驗證測試
├── scripts/
│   └── run_ns3.py          # ASTRA-sim NS-3 網路模擬執行器
├── configs/                # ASTRA-sim 配置檔案
├── data/
│   ├── chakra/             # 追蹤數據
│   │   ├── pytorch_traces/ # PyTorch 原始追蹤
│   │   ├── log/           # 訓練日誌
│   │   └── workload_et/   # Chakra ET 檔案
│   └── cifar10/           # 訓練資料集
├── runs/                   # 模擬結果與校準數據
└── tutorials/              # 教學範例
```

## 🚀 快速開始

### 1. 環境設定

```bash
# 啟動 Docker 環境
docker-compose up -d

# 進入容器
docker exec -it rocm-horovod bash
```

### 2. 訓練並生成追蹤數據

```bash
# 執行分散式 CIFAR-10 訓練，並在第 2 個 epoch 生成性能追蹤
torchrun --standalone --nproc_per_node=2 ./src/train_rocm_pytorch.py \
  --epochs 3 --batch-size 128 \
  --profile-epoch 2 --trace-wait 32 --trace-steps 128
```

**輸出：**
- `data/chakra/pytorch_traces/host_*.json` - CPU 追蹤
- `data/chakra/pytorch_traces/device_*.json` - GPU 追蹤
- `data/chakra/log/training_*.json` - 訓練指標

### 3. 轉換追蹤格式

```bash
# 將 PyTorch 追蹤轉換為 Chakra ET 格式
python ./src/conver_to_chakra_et.py --et-prefix auto
```

**處理流程：**
1. **連結追蹤**：`chakra_trace_link` 將 host/device 追蹤合併為 HDT
2. **格式轉換**：`chakra_converter` 將 HDT 轉換為 ET 檔案
3. **自動命名**：根據主要通訊模式自動命名工作負載

**輸出：**
- `data/chakra/pytorch_traces/hdt_*.json` - 連結後的追蹤
- `data/chakra/workload_et/*.et` - Chakra 執行追蹤

### 4. 網路模擬

```bash
# 執行 ASTRA-sim NS-3 網路模擬
python ./scripts/run_ns3.py \
  --workload data/chakra/workload_et \
  --topology 2gpu_auto1d_2 \
  --bandwidth 25 \
  --latency 1 \
  --output-dir runs/
```

**功能特色：**
- 🔧 **自動拓撲生成**：根據 ET 檔案自動配置網路拓撲
- 📊 **性能校準**：從真實訓練指標校準模擬參數
- 🎯 **虛擬擴展**：將小規模追蹤擴展到大規模模擬
- 📈 **結果分析**：自動解析並導出性能指標

## 🛠️ 核心工具詳解

### train_rocm_pytorch.py
分散式深度學習訓練框架，專為 ROCm 環境最佳化：

```bash
# 基本訓練
torchrun --standalone --nproc_per_node=2 ./src/train_rocm_pytorch.py

# 進階選項
python ./src/train_rocm_pytorch.py \
  --dataset cifar10 \
  --model resnet18 \
  --epochs 5 \
  --batch-size 256 \
  --profile-epoch 3 \
  --trace-wait 50 \
  --trace-steps 200 \
  --inject-sync-hack  # ROCm 兼容性選項
```

### conver_to_chakra_et.py
智能追蹤轉換工具：

```bash
# 完全自動化
python ./src/conver_to_chakra_et.py

# 精細控制
python ./src/conver_to_chakra_et.py \
  --ranks 0 1 2 3 \
  --et-prefix allreduce \
  --no-clean \
  --no-autopatch
```

### run_ns3.py
進階網路模擬執行器：

```bash
# 標準模擬
python ./scripts/run_ns3.py \
  --workload data/chakra/workload_et \
  --topology 8gpu_ring2d_4 \
  --bandwidth 100 \
  --latency 0.5

# 虛擬擴展到 128 節點
python ./scripts/run_ns3.py \
  --workload data/chakra/workload_et \
  --topology 128gpu_torus3d_16 \
  --virtual-scale 16 \
  --auto-calibrate
```

## 📊 輸出與分析

### 性能指標文件
- **`runs/metrics.csv`**: 單次執行詳細指標
- **`runs/calibration_all.csv`**: 歷史校準數據庫
- **`runs/*/stdout.log`**: ASTRA-sim 原始輸出

### 關鍵指標
- **real_t_step_ms**: 真實訓練步驟時間
- **real_t_comm_ms**: 真實通訊時間
- **sim_wall_cycles**: 模擬時鐘週期
- **sim_comm_cycles**: 模擬通訊週期
- **alpha_us**: 校準參數 (微秒/週期)

## 🔧 進階配置

### Docker 環境自定義
```bash
# 使用特定 ROCm 版本
VERSION=rocm6.1_ubuntu22.04_py3.10_pytorch_2.3.0 docker-compose up
```

### ASTRA-sim 配置
配置檔案位於 `configs/astra-sim/`：
- `system/*.json` - 系統配置（記憶體、處理器）
- `topos/*.json` - 網路拓撲定義
- `ns3/*.txt` - NS-3 網路參數

## 🧪 測試與驗證

```bash
# 環境驗證
python ./src/tests/check_version.py

# 追蹤驗證
python ./src/tests/check_trace_ready.py

# ET 格式驗證
python ./src/tests/validate_et.py

# Horovod 通訊測試
python ./src/tests/horovod_allreduce_test.py
```

## 📚 教學範例

探索 `tutorials/` 目錄中的完整範例：
- **hoti2024/**: HOT Interconnects 2024 示範
- **micro2024/**: MICRO 2024 研討會材料
- **asplos2023/**: ASPLOS 2023 練習

## 🐛 常見問題

### Q: 訓練追蹤為空或不完整？
```bash
# 確保足夠的追蹤步驟和等待時間
--trace-wait 32 --trace-steps 128
```

### Q: ROCm 兼容性問題？
```bash
# 啟用同步修補
--inject-sync-hack
```

### Q: ET 轉換失敗？
```bash
# 檢查 HTA 和 Chakra 版本
python ./src/tests/check_version.py
```
