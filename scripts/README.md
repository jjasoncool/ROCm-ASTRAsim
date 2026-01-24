# `scripts/` 使用說明（ASTRA-sim ns-3 執行與校準工具）

本資料夾包含 ASTRA-sim × ns-3 網路模擬的執行與校準腳本，核心為 `run_ns3.py`。

---

# ASTRA-sim 網路參數校準方法論 (Network Parameter Calibration Methodology)

**在執行模擬之前，必須先建立正確的參數校準流程。**

本節詳述如何將真實硬體環境的測量數據，轉換為模擬器 (`topology.txt`) 中的精確參數。此過程包含三個關鍵步驟：**測量 (Measurement)**、**分析 (Analysis)** 與 **計算 (Calculation)**。

## 1. 測量階段：獲取物理基準 (Physical Baselines)
我們不應使用硬體規格書上的理論峰值，而應使用如 `rccl-tests` 等微基準測試工具測量「有效性能」。

*   **有效頻寬 (Effective Bandwidth)**: 使用大封包測量 Bus Bandwidth，作為 `topology.txt` 頻寬設定的依據。
*   **端對端延遲 (End-to-End Latency, $T_{RCCL}$)**: 使用小封包 (e.g., 4 bytes) 測量 Round-Trip Time 或 One-Way Latency。這是校準的核心數據。

## 2. 分析階段：拓撲結構拆解
ASTRA-sim 與 NS-3 的延遲參數是定義在「單條鏈路 (Per-Link)」上的，而非端對端。因此，直接填入測量到的 $T_{RCCL}$ 是錯誤的。我們必需分析訊號經過的路徑：

*   **定義跳數 ($N_{hops}$)**: 訊號從來源 GPU 到達目的 GPU 所經過的實體線路數量。
    *   **直連 (P2P)**: $N_{hops} = 1$ (GPU $\rightarrow$ GPU)
    *   **單層 Switch**: $N_{hops} = 2$ (GPU $\rightarrow$ Switch $\rightarrow$ GPU)
    *   **多層 Switch (e.g., Fat-Tree)**: $N_{hops} = 4$ (GPU $\rightarrow$ Leaf $\rightarrow$ Spine $\rightarrow$ Leaf $\rightarrow$ GPU)

## 3. 計算階段：延遲分配公式
根據物理疊加原理，總延遲約等於各段鏈路延遲之和（忽略 Switch 內部排隊與處理時間的簡化模型）。

我們使用以下公式推導單條鏈路的設定值 $T_{link}$：

$$T_{link} = \frac{T_{RCCL} - T_{overhead}}{N_{hops}}$$

*   **$T_{link}$**: 填入 `topology.txt` 的目標參數。
*   **$T_{RCCL}$**: 實測的端對端延遲。
*   **$T_{overhead}$**: 軟體與驅動開銷。若採取「有效延遲 (Effective Latency)」策略，可將此項設為 0，將軟體開銷平均攤提至物理鏈路中，以簡化模擬模型。

---

## 範例應用：雙節點單 Switch 架構校準

以下展示如何應用上述方法論於實際案例。

### 步驟 1: 測量
在 AMD 雙 GPU 環境下，使用 `rccl-tests` (小封包) 測得端對端延遲 $T_{RCCL} \approx 25 \mu s$。

此外，使用 `rocm-bandwidth-test` 測量記憶體頻寬，以決定 `lmbw` 參數。

**測量結果 (rocm-bandwidth-test):**
```text
          RocmBandwidthTest Version: 2.6.0
          Device: 1,  AMD Radeon RX 9070 XT
          Device: 2,  AMD Radeon RX 9070 XT

          Unidirectional copy peak bandwidth GB/s
          D/D       1           2
          1         540.849     14.045
          2         14.046      540.064
```
由此可知，本地記憶體頻寬 (Local Memory Bandwidth) 約為 **540 GB/s**。
因此，後續模擬指令中建議加入 `--lmbw 540` (預設為 1600)。

### 步驟 2: 分析
拓撲為 `2_nodes_1_switch`。路徑為 `GPU 0` $\rightarrow$ `Switch` $\rightarrow$ `GPU 1`。
路徑經過 2 條線路，因此 $N_{hops} = 2$。

### 步驟 3: 計算
代入公式 (假設軟體開銷已攤提)：

$$T_{link} = \frac{25 \mu s}{2} = 12.5 \mu s$$

### 結論
在 `topology.txt` 中，鏈路延遲應設定為 **12.5 $\mu$s**。
*   若直接填入 25 $\mu$s，模擬器會計算 $25 \times 2 = 50 \mu s$，導致模擬結果嚴重偏差。
*   此 12.5 $\mu$s 為物理常數，當未來擴展至 128 節點 (Fat-Tree, 4 hops) 時，模擬器將自動計算 $12.5 \times 4 = 50 \mu s$，正確反映大規模網路的物理特性。

---

## 完整工作流程

完整的模擬流程分為三的獨立階段，對應 `src` 和 `scripts` 中的腳本：

1.  **Trace 生成 (`src/train_rocm_pytorch.py`)**:
    *   在 ROCm 環境下，使用 PyTorch DDP 進行模型訓練（支援 CIFAR-10 CNN 和 ResNet-50）。
    *   透過 `torch.profiler` 產生詳細的 host 和 device Kineto JSON trace 檔案。
    *   使用 `--model-tag` 區分不同模型的輸出。

2.  **Trace 轉換 (`src/conver_to_chakra_et.py`)**:
    *   將 Kineto JSON trace 轉換為 ASTRA-sim 可用的 Chakra ET (`.et`) 格式。
    *   **關鍵功能**:
        *   **AMD GPU 兼容性修補**: 自動修復 AMD NCCL kernel 命名問題。
        *   **系統感知校準**: 對於 System-Bound 的模型（如 CIFAR-10），可使用 `--force-avg-kernel-ns` 將系統開銷攤提到計算節點，使模擬更接近真實時間。

3.  **網路模擬與校準 (`scripts/run_ns3.py`)**:
    *   使用轉換後的 `.et` 工作負載執行 ASTRA-sim ns-3 模擬。
    *   **關鍵功能**:
        *   **自動化配置**: 自動生成邏輯拓撲、修補系統和網路配置文件。
        *   **虛擬擴展**: 將小規模（如 2-GPU）的 trace 虛擬擴展到大規模（如 128-GPU）的模擬。
        *   **自動校準**: 透過比對原始 trace 的真實執行時間與模擬輸出的 cycles，自動計算 `alpha_us` 校準因子，並將結果存入 `runs/calibration_all.csv`。

---

## `run_ns3.py` 快速開始

### 情境 A：System-Bound 模型校準 (CIFAR-10)

此情境模擬一個計算量小、系統開銷佔主導的場景，需要啟用**系統感知校準**來獲得有意義的模擬結果。

1.  **生成 Trace (CIFAR-10)**
    *   使用 `workers=0` 放大系統開銷，`--model-tag cifar10` 標記輸出。
    ```bash
    torchrun --standalone --nproc_per_node=2 ./src/train_rocm_pytorch.py \
      --model cifar10 --workers 0 \
      --trace-wait 32 --trace-steps 4 \
      --model-tag cifar10
    ```

2.  **轉換 Trace (啟用系統感知校準)**
    *   使用 `--force-avg-kernel-ns` 將真實世界的 GPU 時間（包含開銷）攤提回計算節點。
    *   `609000` ns (609 µs) 是一個基於 CIFAR-10 在 MI250x 上的經驗值。
    ```bash
    python src/conver_to_chakra_et.py \
      --model-tag cifar10 \
      --force-avg-kernel-ns 609000
    ```

3.  **執行模擬與自動校準**
    *   `run_ns3.py` 會讀取 `pytorch_traces` 中的 `*_cifar10.json` 來獲取真實執行時間，並與模擬的 cycles 進行校準。
    ```bash
    python scripts/run_ns3.py \
      --workload data/chakra/workload_et --model-tag cifar10 \
      --topo auto:1d \
      --phys-topo configs/astra-sim/topos/2_nodes_1_switch_topology.txt
      --coll-opt localBWAware \
      --lmbw 540
    ```
    *   校準結果會自動追加到 `runs/calibration_all.csv`。

### 情境 B：Compute-Bound 模型模擬 (ResNet-50)

此情境模擬一個計算密集型場景，可直接使用 trace 中的數據進行模擬，並虛擬擴展到大規模拓撲。

1.  **生成 Trace (ResNet-50)**
    ```bash
    torchrun --standalone --nproc_per_node=2 ./src/train_rocm_pytorch.py \
      --model resnet50 --workers 4 \
      --trace-wait 32 --trace-steps 2 \
      --model-tag resnet50
    ```

2.  **轉換 Trace (標準模式)**
    *   不使用 `--force-avg-kernel-ns`，讓轉換器根據 trace 中的真實 kernel 時間計算 cycles。
    ```bash
    python src/conver_to_chakra_et.py --model-tag resnet50
    ```

3.  **基準校準 (Baseline Calibration - 建議)**
    *   在擴展前，先在小規模 (2-GPU) 環境下驗證模擬準確度。
    ```bash
    python scripts/run_ns3.py \
      --workload data/chakra/workload_et --model-tag resnet50 \
      --topo auto:1d \
      --phys-topo configs/astra-sim/topos/2_nodes_1_switch_topology.txt
      --coll-opt localBWAware \
      --lmbw 540
    ```
    *   檢查 `runs/calibration_all.csv` 確認誤差 (ResNet50 通常 < 5%)。

---

## `run_ns3.py` 參數詳解

### 核心參數

| 參數 | 描述 | 類型/預設值 | 範例 |
|---|---|---|---|
| `--workload` | 工作負載資料夾 (`.et` 檔案)。 | 字串 (必要) | `data/chakra/workload_et` |
| `--model-tag` | 模型標籤，用於過濾 workload 和 trace 檔案。 | 字串 (選用) | `cifar10`, `resnet50` |
| `--virtual-world N` | 將 workload 虛擬擴展到 N 個節點。 | 整數 (選用) | `128` |
| `--topo` | 邏輯拓撲 (ASTRA-sim)。 | 字串 ("auto:1d") | `auto:2d`, `dims:4x4`, `file:topo.json` |
| `--phys-topo` | 物理拓撲 (ns-3)，未指定時依 world size 自動推測。 | 字串 (選用) | `configs/astra-sim/topos/128_nodes_*.txt` |
| `--ns3-bin` | ns-3 執行檔路徑。 | 字串 (環境變數 `ASTRA_NS3_BIN`) | |
| `--system`, `--network`, `--remote` | baseline 設定檔路徑。 | 字串 (預設) | |

### 系統層覆蓋 (影響 ASTRA-sim 內部排程)

| 參數 | 描述 | 類型/預設值 | 範例 |
|---|---|---|---|
| `--coll-opt` | 集體操作優化策略。 | 字串 (選用) | `localBWAware` |
| `--lmbw` | 本地記憶體頻寬 (GB/s)。 | 整數 (選用) | `1600` |

### 網路層覆蓋 (影響 ns-3 封包行為)

| 參數 | 描述 | 類型/預設值 | 範例 |
|---|---|---|---|
| `--qcn` | 啟用/禁用 QCN (量化擁塞通知)。 | 0 或 1 (選用) | `1` |
| `--pfc-dyn` | 啟用/禁用動態 PFC 門檻。 | 0 或 1 (選用) | `1` |
| `--buffer` | 交換器緩衝區大小 (封包數)。 | 整數 (選用) | `64` |
| `--payload` | 封包 payload 大小 (bytes)。 | 整數 (選用) | `1500` |

### 校準與輸出參數

| 參數 | 描述 | 類型/預設值 | 範例 |
|---|---|---|---|
| `--no-autocalib` | 禁用自動校準 `alpha_us`。 | 布林 (選用) | |
| `--trace-dir` | 指定 PyTorch Kineto trace 的來源目錄以進行校準。 | 字串 (預設 `data/chakra/pytorch_traces`) | |
| `--calib-db` | 指定校準結果的 CSV 資料庫路徑。 | 字串 (預設 `runs/calibration_all.csv`) | |
| `--log-dir` | 模擬輸出的根目錄。 | 字串 ("runs") | `my_runs` |
| `--dry-run` | 僅產生設定檔和命令，不執行模擬。 | 布林 (選用) | |

---

## 校準原理 (`--no-autocalib` 未啟用時)

當 `run_ns3.py` 執行 `world=2` 的模擬時，它會：

1.  **解析模擬結果**: 從 `stdout.log` 提取 `sim_cycles_step` (模擬的總 wall cycles)。
2.  **查找真實 Trace**:
    *   根據 `--model-tag` 在 `--trace-dir` (預設 `data/chakra/pytorch_traces`) 中尋找對應的 Kineto trace (如 `device_0_cifar10.json`)。
    *   從 trace 中提取 `real_t_step_ms` (真實世界的每步執行時間)。
3.  **計算 Alpha**:
    *   `alpha_us = (real_t_step_ms * 1000) / sim_cycles_step`
    *   `alpha_us` 代表**每個模擬 cycle 對應多少真實世界的微秒 (µs)**。
4.  **儲存結果**:
    *   將本次模擬的完整參數、`real_t_*`、`sim_cycles_*` 和計算出的 `alpha_us` 存入 `out/metrics.csv`。
    *   將這筆紀錄追加到 `--calib-db` (預設 `runs/calibration_all.csv`) 中，方便後續分析比較。

---

## 4. 進階實驗：ResNet50 大規模擴展分析 (Scalability Analysis)

本節說明如何基於已校準的 2-GPU ResNet50 模型，進行 128-GPU 的虛擬擴展模擬，並探討其學術價值。

### 學術價值與實驗目的
1.  **低成本架構探索 (Cost-effective Exploration)**:
    *   建置 128 個高階 GPU 的實體叢集成本極高。透過 ASTRA-sim，我們僅需在 2 個 GPU 上採樣 (Profiling)，即可準確預測模型在數百個節點上的行為。
2.  **瓶頸識別 (Bottleneck Identification)**:
    *   ResNet50 雖為 Compute-Bound，但在大規模分佈式訓練下，All-Reduce 通訊開銷可能隨著節點數增加而成為瓶頸。此實驗可驗證目前的網路拓撲與頻寬是否足以支撐 128 節點的線性加速。
3.  **協同設計 (HW/SW Co-design)**:
    *   透過模擬結果，研究人員可以回答：「如果網路頻寬翻倍，訓練速度能提升多少？」或「更換 Switch 拓撲是否有助於效能？」，從而指導未來的硬體採購與資料中心設計。

### 操作步驟

#### 步驟 1: 確認校準基線 (Baseline Validation)
在擴張前，需確認小規模 (2-GPU) 的模擬誤差在可接受範圍內。
*   檢查 `runs/calibration_all.csv`。
*   **ResNet50 範例**: 相對誤差 `rel_err_comm` < 2% 為優良基線，可進行擴張。
*   **CIFAR-10 注意**: 若誤差過大 (e.g., > 50%)，代表 System Overhead 模型失準，不建議進行大規模擴張預測。

#### 步驟 2: 執行虛擬擴張 (Virtual Expansion)
使用 2-GPU 的 trace 驅動 128-GPU 的模擬環境。

```bash
# ResNet50: 2 GPUs -> 128 GPUs Expansion
python scripts/run_ns3.py \
  --workload data/chakra/workload_et --model-tag resnet50 \
  --virtual-world 128 \
  --topo auto:3d \
  --phys-topo configs/astra-sim/topos/128_nodes_32_switch_topology.txt \
  --no-autocalib \
  --lmbw 540
```

#### 步驟 3: 分析加速比 (Speedup Analysis)
模擬完成後，分析 `out/metrics.csv` 中的 `sim_t_step_ms` (每步模擬時間)。

*   **理想時間 (Ideal Time)**: $T_{ideal} = T_{compute\_2gpu} / (128/2)$ (假設完美線性加速)
*   **模擬時間 (Simulated Time)**: $T_{sim}$ (包含網路擁塞與通訊延遲)
*   **通訊效率 (Efficiency)**: $\frac{T_{ideal}}{T_{sim}} \times 100\%$
