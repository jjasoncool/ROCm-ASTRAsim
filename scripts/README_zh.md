# scripts/ — ASTRA-sim ns-3 執行與校準工具

> **繁體中文** | [English](README.md)

本資料夾包含 ASTRA-sim × ns-3 網路模擬的執行與校準腳本，核心為 `run_ns3.py`。

---

## 目錄

1. [網路參數校準方法論](#1-網路參數校準方法論)
2. [完整工作流程](#2-完整工作流程)
3. [快速開始](#3-快速開始)
4. [`run_ns3.py` 參數詳解](#4-run_ns3py-參數詳解)
5. [校準原理](#5-校準原理)
6. [進階：大規模擴展分析](#6-進階大規模擴展分析)

---

## 1. 網路參數校準方法論

在執行模擬之前，必須建立正確的參數校準流程。本節說明如何將真實硬體的測量數據，轉換為模擬器 `topology.txt` 中的精確參數。

### 步驟 1：測量物理基準

使用 `rccl-tests` 等微基準測試工具測量「有效性能」，而非硬體規格書的理論峰值：

| 測量項目 | 工具 | 說明 |
|---|---|---|
| **有效頻寬** | `rccl-tests`（大封包）| 作為 `topology.txt` 頻寬設定的依據 |
| **端對端延遲** $T_{RCCL}$ | `rccl-tests`（小封包，如 4 bytes）| 校準的核心數據 |
| **本地記憶體頻寬** | `rocm-bandwidth-test` | 決定 `--lmbw` 參數 |

### 步驟 2：分析拓撲跳數

ASTRA-sim 與 ns-3 的延遲參數定義在**單條鏈路**上，而非端對端。需根據訊號路徑決定跳數 $N_{hops}$：

| 拓撲類型 | 路徑 | $N_{hops}$ |
|---|---|---|
| 直連 (P2P) | GPU → GPU | 1 |
| 單層 Switch | GPU → Switch → GPU | 2 |
| 多層 Switch（Fat-Tree）| GPU → Leaf → Spine → Leaf → GPU | 4 |

### 步驟 3：計算單鏈路延遲

$$T_{link} = \frac{T_{RCCL} - T_{overhead}}{N_{hops}}$$

- **$T_{link}$**：填入 `topology.txt` 的目標值
- **$T_{RCCL}$**：實測端對端延遲
- **$T_{overhead}$**：採「有效延遲」策略時設為 0，將軟體開銷平均攤提至物理鏈路

### 範例：雙節點單 Switch 架構

**測量**（`rccl-tests` 小封包）：$T_{RCCL} \approx 25\ \mu s$

**分析**：路徑為 GPU → Switch → GPU，$N_{hops} = 2$

**初始計算**：

$$T_{link,init} = \frac{25\ \mu s}{2} = 12.5\ \mu s$$

> 若直接填入 25 µs，模擬器會計算 $25 \times 2 = 50\ \mu s$，導致結果嚴重偏差。

**選擇性微調**：

計算出的 12.5 µs 是合理的起始值，可直接使用。若想進一步縮小誤差，可將模擬輸出與真實 PyTorch 執行 Log 比對，調整每條鏈路延遲直到誤差最小。在本設定環境下，**14 µs** 達到最佳吻合，因此本 Repository 的所有模擬設定均採用此數值。

**本地記憶體頻寬測量**（`rocm-bandwidth-test`）：

```text
          RocmBandwidthTest Version: 2.6.0
          Device: 1,  AMD Radeon RX 9070 XT
          Device: 2,  AMD Radeon RX 9070 XT

          Unidirectional copy peak bandwidth GB/s
          D/D       1           2
          1         540.849     14.045
          2         14.046      540.064
```

本地記憶體頻寬約為 **540 GB/s**，後續模擬建議加入 `--lmbw 540`（預設值為 1600）。

---

## 2. 完整工作流程

完整模擬流程分為三個獨立階段：

| 階段 | 腳本 | 功能 |
|---|---|---|
| **1. Trace 生成** | `src/train_rocm_pytorch.py` | ROCm 環境下以 PyTorch DDP 訓練，產生 Kineto JSON trace |
| **2. Trace 轉換** | `src/conver_to_chakra_et.py` | 將 Kineto trace 轉為 Chakra ET (`.et`) 格式 |
| **3. 網路模擬** | `scripts/run_ns3.py` | 以 `.et` workload 執行 ASTRA-sim ns-3 模擬，自動校準 |

關鍵特性：
- **AMD GPU 兼容性修補**（第 2 階段）：自動修復 AMD RCCL kernel 命名問題
- **系統感知校準**（第 2 階段）：對 System-Bound 模型可使用 `--force-avg-kernel-ns` 攤提系統開銷
- **虛擬擴展**（第 3 階段）：將小規模（2-GPU）trace 擴展至大規模（如 128-GPU）模擬
- **自動校準**（第 3 階段）：自動計算 `alpha_us` 因子並儲存至 `runs/calibration_all.csv`

---

## 3. 快速開始

### 情境 A：System-Bound 模型（CIFAR-10）

適用於計算量小、系統開銷佔主導的場景，需啟用**系統感知校準**。

**步驟 1：生成 Trace**

```bash
torchrun --standalone --nproc_per_node=2 ./src/train_rocm_pytorch.py \
  --model cifar10 --workers 0 \
  --trace-wait 32 --trace-steps 4 \
  --model-tag cifar10
```

> 使用 `--workers 0` 放大系統開銷以進行壓力測試。

**步驟 2：轉換 Trace（啟用系統感知校準）**

```bash
python src/conver_to_chakra_et.py \
  --model-tag cifar10 \
  --force-avg-kernel-ns 609000
```

> `--force-avg-kernel-ns 609000`（609 µs）將真實 GPU 時間攤提回計算節點。

**步驟 3：執行模擬與自動校準**

```bash
python scripts/run_ns3.py \
  --workload data/chakra/workload_et --model-tag cifar10 \
  --topo auto:1d \
  --phys-topo configs/astra-sim/topos/2_nodes_1_switch_topology.txt \
  --coll-opt localBWAware \
  --lmbw 540
```

---

### 情境 B：Compute-Bound 模型（ResNet-50）

適用於計算密集型場景，可直接使用 trace 數據並虛擬擴展至大規模拓撲。

**步驟 1：生成 Trace**

```bash
torchrun --standalone --nproc_per_node=2 ./src/train_rocm_pytorch.py \
  --model resnet50 --workers 4 \
  --trace-wait 32 --trace-steps 2 \
  --model-tag resnet50
```

**步驟 2：轉換 Trace（標準模式）**

```bash
python src/conver_to_chakra_et.py --model-tag resnet50
```

> 不使用 `--force-avg-kernel-ns`，讓轉換器根據 trace 中的真實 kernel 時間計算 cycles。

**步驟 3：基準校準（建議）**

先在 2-GPU 環境下驗證準確度，再進行大規模擴展：

```bash
python scripts/run_ns3.py \
  --workload data/chakra/workload_et --model-tag resnet50 \
  --topo auto:1d \
  --phys-topo configs/astra-sim/topos/2_nodes_1_switch_topology.txt \
  --coll-opt localBWAware \
  --lmbw 540
```

> 檢查 `runs/calibration_all.csv`，ResNet-50 誤差通常 < 5%。

---

## 4. `run_ns3.py` 參數詳解

### 核心參數

| 參數 | 描述 | 預設值 | 範例 |
|---|---|---|---|
| `--workload` | `.et` 工作負載資料夾 | 必要 | `data/chakra/workload_et` |
| `--model-tag` | 模型標籤，過濾 workload 與 trace 檔案 | 選用 | `cifar10`, `resnet50` |
| `--virtual-world N` | 虛擬擴展至 N 個節點 | 選用 | `128` |
| `--topo` | 邏輯拓撲（ASTRA-sim） | `auto:1d` | `auto:2d`, `dims:4x4`, `file:topo.json` |
| `--phys-topo` | 物理拓撲（ns-3） | 依 world size 推測 | `configs/astra-sim/topos/128_nodes_*.txt` |
| `--ns3-bin` | ns-3 執行檔路徑 | 環境變數 `ASTRA_NS3_BIN` | |
| `--system`, `--network`, `--remote` | baseline 設定檔路徑 | 預設值 | |

### 系統層覆蓋（影響 ASTRA-sim 內部排程）

| 參數 | 描述 | 範例 |
|---|---|---|
| `--coll-opt` | 集體操作優化策略 | `localBWAware` |
| `--lmbw` | 本地記憶體頻寬（GB/s） | `540` |

### 網路層覆蓋（影響 ns-3 封包行為）

| 參數 | 描述 | 範例 |
|---|---|---|
| `--qcn` | 啟用/禁用 QCN（量化擁塞通知） | `0` 或 `1` |
| `--pfc-dyn` | 啟用/禁用動態 PFC 門檻 | `0` 或 `1` |
| `--buffer` | 交換器緩衝區大小（封包數） | `64` |
| `--payload` | 封包 payload 大小（bytes） | `1500` |

### 校準與輸出參數

| 參數 | 描述 | 預設值 |
|---|---|---|
| `--no-autocalib` | 禁用自動校準 `alpha_us` | — |
| `--trace-dir` | Kineto trace 來源目錄 | `data/chakra/pytorch_traces` |
| `--calib-db` | 校準結果 CSV 路徑 | `runs/calibration_all.csv` |
| `--log-dir` | 模擬輸出根目錄 | `runs` |
| `--dry-run` | 僅產生設定檔與命令，不執行模擬 | — |

---

## 5. 校準原理

當 `run_ns3.py` 在 `world=2` 下執行時（未加 `--no-autocalib`），流程如下：

1. **解析模擬結果**：從 `stdout.log` 提取 `sim_cycles_step`（模擬的總 wall cycles）
2. **查找真實 Trace**：根據 `--model-tag` 在 `--trace-dir` 中找到對應的 Kineto trace，提取 `real_t_step_ms`
3. **計算 Alpha**：

   $$\alpha_{us} = \frac{real\_t\_step\_ms \times 1000}{sim\_cycles\_step}$$

   $\alpha_{us}$ 代表每個模擬 cycle 對應多少真實世界的微秒（µs）

4. **儲存結果**：寫入 `out/metrics.csv`，並追加至 `runs/calibration_all.csv`

---

## 6. 進階：大規模擴展分析

基於已校準的 2-GPU 模型，進行 128-GPU 虛擬擴展模擬。

### 步驟 1：確認校準基線

- 檢查 `runs/calibration_all.csv`
- ResNet-50：`rel_err_comm` < 2% 為優良基線，可進行擴展
- CIFAR-10：誤差過大（> 50%）代表 System Overhead 模型失準，不建議進行大規模擴展預測

### 步驟 2：執行虛擬擴展

```bash
python scripts/run_ns3.py \
  --workload data/chakra/workload_et --model-tag resnet50 \
  --virtual-world 128 \
  --topo file:configs/astra-sim/topos/logical_128nodes_FatTree_L16_S8.json \
  --phys-topo configs/astra-sim/topos/128nodes_FatTree_L16_S8.txt \
  --system configs/astra-sim/system/system_128nodes_FatTree_L16_S8.json \
  --no-autocalib \
  --lmbw 540
```

### 步驟 3：分析加速比

從 `out/metrics.csv` 讀取 `sim_t_step_ms`，計算通訊效率：

$$\text{Efficiency} = \frac{T_{ideal}}{T_{sim}} \times 100\%$$

其中 $T_{ideal} = T_{compute\_2gpu} \div (128/2)$（假設完美線性加速）。
