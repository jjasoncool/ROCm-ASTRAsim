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
7. [支援量測工具](#7-支援量測工具)

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

**測量**（`rccl-tests` 小封包）： $T_{\mathrm{RCCL}} \approx 25\,\mu\mathrm{s}$

作為具體參考，本平台上的小訊息 `all_reduce_perf` 測試（8–1024 B、FP16、2 GPUs）量測到的端對端延遲約落在 **~25.8–33.8 µs**，對應到 2-hop 校準路徑上的 **12.9–16.9 µs per-link**。文件採用 **14 µs** 作為此量測範圍中央附近的代表性有效延遲。

**分析**：路徑為 GPU → Switch → GPU， $N_{hops} = 2$

**初始計算**：

$$T_{link,init} = \frac{25\ \mu s}{2} = 12.5\ \mu s$$

> 若直接填入 25 µs，模擬器會計算 $25 \times 2 = 50\ \mu s$，導致結果嚴重偏差。

**解讀方式**：

用 25 µs ÷ 2 = 12.5 µs 的做法，只能視為初始近似。此處採用的 **14 µs**，應理解為用於**相對拓撲比較**的有效延遲參數，而不是精確的物理 per-hop delay。

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
| **1. Trace 生成（DDP）** | `src/train_rocm_pytorch.py` | ROCm 環境下以 PyTorch DDP 訓練，產生 Kineto JSON trace |
| **1. Trace 生成（TP）**  | `src/train_rocm_tensor.py` | ROCm 環境下以 PyTorch TP=2 訓練，供 Qwen 1.5B TP+DDP 實驗使用 |
| **2. Trace 轉換** | `src/conver_to_chakra_et.py` | 將 Kineto trace 轉為 Chakra ET (`.et`) 格式；TP+DDP 使用 `--add-ddp` |
| **3. 網路模擬** | `scripts/run_ns3.py` | 以 `.et` workload 執行 ASTRA-sim ns-3 模擬，自動校準 |

關鍵特性：
- **AMD GPU 兼容性修補**（第 2 階段）：自動修復 AMD RCCL kernel 命名問題
- **系統感知校準**（第 2 階段）：對 System-Bound 模型可使用 `--force-avg-kernel-ns` 攤提系統開銷
- **TP+DDP 組合**（第 2 階段）：`--add-ddp --target-tp 8` 會把 DDP AllReduce 節點接到 TP trace 之後，並縮放計算時間
- **虛擬擴展**（第 3 階段）：將小規模（2-GPU）trace 擴展至大規模（如 128-GPU）模擬
- **自動校準**（第 3 階段）：對齊實測與模擬的通訊視窗，計算 `alpha_us`，並追加至 `runs/calibration_aligned.csv`（`calibration_all.csv` 是舊 log，見 §5）

**論文涵蓋的工作負載。** 此 Pipeline 在四種通訊強度下被驗證：

| 實驗 | 工作負載 | Trace 腳本 | 常用 tag | 備註 |
|---|---|---|---|---|
| 1. 計算主導 AllReduce | ResNet-50 DDP（~89.7 MiB） | `train_rocm_pytorch.py --model resnet50` | `resnet50` | 主要校準基準 |
| 2. 通訊密集 AllReduce | Qwen 0.5B DDP（~1.84 GiB） | `train_rocm_pytorch.py --model qwen05b` | `qwen05b` | Twisted Torus 上必須 `active-chunks=4` |
| 3. 階層式 TP+DDP | Qwen 1.5B，TP=8 × DDP=16 | `train_rocm_tensor.py` + `conver_to_chakra_et.py --add-ddp --target-tp 8` | `qwen15b_tp8ddp` | |
| 4. All-to-All 頻寬飽和 | 合成 1 GB All-to-All | 對 `resnet50_all2all` 跑 `scale_et_comm_workload.py --bytes 1G` | `resnet50_all2all_1GB` | ns-3 端建議 `--payload 12000` |

Qwen 0.5B 的 2-GPU 校準結果亦驗證 trace 形式正確：每 step 含 10,916 個 COMP 節點與 37 個 AllReduce COMM 節點，合計 1,884.6 MiB——正好是 494,032,768 個參數的 FP32 梯度量。

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
python src/conver_to_chakra_et.py --model-tag cifar10
```

> 若進行 latency-dominated 診斷，可再搭配 `--force-avg-kernel-ns` 將 wall-clock 時間攤提回計算節點；但目前已將 CIFAR-10 排除於大規模拓撲評估之外。

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
  --lmbw 540 --et-iters 1
```

> ResNet-50 的 device 視窗涵蓋一次完整訓練迭代——它的五個通訊 kernel 恰好對應 DDP 的五個梯度 bucket——因此即使 trace 是以 `--trace-steps 2` 收集，這裡仍是 `--et-iters 1`。檢查 `runs/calibration_aligned.csv`。ResNet-50 是主要校準基準；$\alpha_{step}$ 是主要 wall-clock 轉換係數，$\alpha_{comm}$ 僅作診斷用途。

---

## 4. `run_ns3.py` 參數詳解

### 核心參數

| 參數 | 描述 | 預設值 | 範例 |
|---|---|---|---|
| `--workload` | `.et` 工作負載資料夾 | 必要 | `data/chakra/workload_et` |
| `--model-tag` | 模型標籤，過濾 workload 與 trace 檔案 | 選用 | `cifar10`, `resnet50`, `qwen05b` |
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
| `--payload` | 封包 payload 大小（bytes）；All-to-All 1 GB 壓力測試建議 `12000` | `1500` |

### 工作負載擴展與長時間執行穩定性

| 參數 | 描述 | 範例 |
|---|---|---|
| `--virtual-world N` | 將每 rank trace 複製擴展到 `N` 節點模擬 | `128` |
| `--comm-scale F` | 將每個 `comm_size` 乘以 `F`。這是**工作負載的通訊工作點設定**，不是對集合演算法的修正——ASTRA-sim 會依設定的參與節點數自行拆解每個集合操作。論文值為 `127/64` = **`1.984375`**；Qwen 0.5B 必須用精確分數，縮放後的尺寸才能被 `preferred-dataset-splits=4` 整除，TP+DDP 則可接受四捨五入的 `1.984`。同一實驗內對所有拓撲一致套用 | `1.984375` |
| `--comm-group FILE` | 直接傳給 `--comm-group-configuration`；未指定則完全不帶該參數 | — |
| `--no-qlen` | 將 `qlen.txt` 導向 `/dev/null`，避免 128 節點時產生數百 GB 除錯輸出 | — |
| `--deadlock-timeout S` | `fct.txt` 連續 `S` 秒未更新即自動 kill（預設 `43200` = 12 h，`0` 表示停用） | `43200` |

### 校準與輸出參數

| 參數 | 描述 | 預設值 |
|---|---|---|
| `--no-autocalib` | 禁用自動校準 `alpha_us` | — |
| `--et-iters N` | 這個 ET 涵蓋幾次訓練迭代（= 產生它時的 `--trace-steps`）。`alpha_us` 需要「每步」的分母，故為必要；未給則留空，不以推測值填補 | — |
| `--trace-dir` | Kineto trace 來源目錄 | `data/chakra/pytorch_traces` |
| `--calib-db` | 校準結果 CSV 路徑 | `runs/calibration_aligned.csv` |
| `--log-dir` | 模擬輸出根目錄 | `runs` |
| `--dry-run` | 僅產生設定檔與命令，不執行模擬 | — |

### Twisted Torus + ring AllReduce 的排程死鎖

預設的 `active-chunks-per-dimension=1` 在 Twisted Torus 上跑高通訊量多維 ring AllReduce（Qwen 0.5B）時，會觸發確定性的 ASTRA-sim 排程死鎖。Twisted Torus 的 X 軸非對稱繞回鏈路造成各節點階段進度不同步，在 ASTRA-sim chunk queue 中產生跨維度循環等待——`fct.txt` 通常會在 ~5,337 個 flow（預期 ~985,088 個）後停止更新。Twisted Torus AllReduce 實驗請改用 `*_4chunks*.json` 系列設定（`active-chunks-per-dimension: 4`）。`*_4chunks_hd.json` 變體在 chunks=4 之上，把 X/Y 維的 ring 換成 halvingDoubling，作為 2×2 因子分析的第二臂；注意 HD 雖能移除 deadlock 並大幅降低 PFC，卻**無法**移除 twist 的 step-time 懲罰（Twisted Torus + HD 仍比 Torus + ring 慢 +74.7%——路徑不對稱仍在）。`active-chunks=4` 同樣只化解排程層級的 deadlock，並非根本的路徑不對稱。DDP 部署請用標準 Torus + ring（最快且不會 deadlock）。已回報為 [ASTRA-sim Issue #370](https://github.com/astra-sim/astra-sim/issues/370)。

---

## 5. 校準原理

當 `run_ns3.py` 在 `world=2` 下執行時（未加 `--no-autocalib`），流程如下：

1. **解析模擬結果**：從 `stdout.log` 提取 `sim_cycles_step` 與 `sim_cycles_comm`。
2. **查找真實 Trace**：根據 `--model-tag` 在 `--trace-dir` 中找到對應的 Kineto trace，提取 `real_t_step_ms`、RCCL kernel 總和（`real_t_net_comm_ms`）與非 RCCL 計算 kernel 總和（`real_t_kernel_ms`）。只計入 GPU 側的 `cat=kernel` 事件，CPU 側的 `user_annotation` 一律排除，避免同一操作被算兩次。
3. **對齊兩側視窗** —— 讓後面所有數字有意義的關鍵步驟。實測側是單一 rank 的 RCCL kernel，模擬側是 ET 實際重播的內容，兩者唯有對齊才涵蓋相同的工作量。因此腳本會：
   - 在程式內合併巢狀的 `ProfilerStep` 區間、保留未被包含者，藉此還原 trace 的迭代數（profiler 自己的 `steps_n` 與兩側都不相符，一律不使用）；
   - 檢查每個通訊 kernel 都落在某個迭代區間內——若有 kernel 落在區間之外，迭代邊界即不可信，執行**直接停止**；
   - 以**迭代數**而非 collective 計數推導 `window_ratio`。計數比會把「每迭代顆粒度差異」（TP trace 上很常見，ET 每迭代重播的 collective 比 trace 紀錄的多）誤讀成「涵蓋的迭代數不同」，結果拿 1 個模擬迭代去比 2 個實測迭代；
   - 當 `--et-iters` 小於還原出的 trace 迭代數、或兩者不成整數比時**直接 raise**。一個看起來合理的比值比停下來更糟；
   - 顆粒度不符時明確標記 `per_iter_granularity_mismatch`，而不是把差異折進除數。
4. **計算 Alpha**：

   $$
   \alpha_{\mathrm{us}} = \frac{\mathrm{real\_t\_step\_ms} \times 1000}{\mathrm{sim\_cycles\_step} / \mathrm{et\_iters}}
   $$

   分子是「每步」，分母也必須是——這正是 `--et-iters` 的用途。未提供時腳本會標記 `alpha_skipped_no_et_iters` 並把 `alpha_us` 留空。$\alpha_{\mathrm{us}}$（即 $\alpha_{step}$）代表每個模擬 cycle 對應多少真實世界的微秒（µs），為論文中所有 128 節點拓撲比較的**主要校準係數**。

   $\alpha_{comm}$ 由已對齊視窗的總和計算，僅作診斷、**不用於校準**：它作用在 ns-3 的奈秒 tick 上，而 $\alpha_{step}$ 作用在由 trace 導出的計算 cycle 上，兩者屬於不同的 cycle 域，本來就不該相等。

5. **儲存結果**：寫入 `out/metrics.csv`，並追加至 `runs/calibration_aligned.csv`。

> **`calibration_all.csv` 是舊的 append log。** 它早於視窗對齊機制，其中的誤差值比較的是涵蓋工作量不同的兩個視窗。論文中每一個校準數值都出自 `calibration_aligned.csv`；請勿引用舊檔。

若想從既有的 run 目錄重算這些指標而不重跑模擬——`calibrate_from_runs.py` 直接複用 `run_ns3.align_and_compare`，兩者不致分歧：

```bash
python3 scripts/calibrate_from_runs.py --out runs/calibration_aligned.csv \
  resnet50=1:<run_dir> cifar10=4:<run_dir> qwen05b=2:<run_dir> qwen15b_tp=2:<run_dir>
```

每個參數的格式是 `tag=et_iters:run_dir`。

**參考實測數據（2-GPU，已對齊視窗，每個訓練步驟）：**

| 指標 | ResNet-50 | CIFAR-10 |
|---|---|---|
| `real_t_step_ms` | 662.92 ms | 224.32 ms |
| `real_t_net_comm_ms`（硬體實測） | **15.87 ms** | **76.69 ms** |
| `real_t_kernel_ms` | 284.10 ms | 50.15 ms |
| comm / step | 2.4% | 34.2% |
| `ns3_comm_ms`（ns-3） | **15.06 ms** | **10.27 ms** |
| ns-3 vs real | **−5.1%** | **−86.6%** |
| $\alpha_{step}$ | **0.002411** | 0.004550 |
| $\alpha_{comm}$ | 0.001054 | 0.007469 |
| `--et-iters` | 1 | 4 |
| 校準狀態 | 主基準 | 排除（scope boundary） |

LLM 工作負載，相同對齊方式：

| Tag | `--et-iters` | `ns3_comm_ms` | `real_t_net_comm_ms` | ns-3 vs real | 旗標 |
|---|---|---|---|---|---|
| `qwen05b` | 2 | 573.08 ms | 3,393.27 ms | **−83.1%** | — |
| `qwen15b_tp` | 2 | 392.25 ms | 5,918.81 ms | **−93.4%** | `per_iter_granularity_mismatch` |

> 對封包 payload（1,000–8,000 B）、逐鏈路延遲（12.5–14 µs）與 QCN 開關的系統性掃描顯示：ResNet-50 的 ns-3 通訊時間在所有設定下都落在 14.0–15.1 ms，也就是每一組都比實測值低 5–12%。殘差對所有可調參數都不敏感。排程受限型的工作負載（CIFAR-10、兩個 Qwen trace）低估幅度顯著更大，因為 ns-3 有建模資料傳輸，卻沒有建模集合排程與 backward 計算之間的同步等待。該成分屬於共用的 trace 與排程，三種拓撲完全相同，因此不會進入相對比較。

---

## 6. 進階：大規模擴展分析

基於已校準的 2-GPU 模型，進行 128-GPU 虛擬擴展模擬。

### 步驟 1：確認校準基線

- 檢查 `runs/calibration_aligned.csv`
- ResNet-50：使用 $\alpha_{step}$ 作為 wall-clock 轉換係數，通訊時間則主要拿來做**相對拓撲比較**
- CIFAR-10：因未建模軟體堆疊開銷主導 step time，排除於大規模拓撲評估之外

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

### 步驟 3：檢查輸出結果

從 `out/metrics.csv`、`stdout.log` 與輸出目錄中的其他統計檔案檢查模擬結果，例如：

- `sim_t_step_ms`
- communication / wall time
- 各 rank 統計資訊

建議重點觀察下列參數與其對應關係：

- **`sim_t_step_ms`**：整體步驟時間，用來比較不同拓撲下的最終執行時間差異。
- **communication / wall time ratio**：觀察通訊時間在總時間中的占比；比例越高，代表 workload 越偏向 communication-bound。
- **per-rank statistics**：檢查是否有特定 rank 明顯較慢，協助辨識負載不均或局部壅塞。
- **`fct.txt` / 其他輸出統計檔**：可用來確認模擬仍持續進行，並觀察流量完成情況。

在拓撲比較上，可將這些指標對照來看：

- 若不同拓撲的 **`sim_t_step_ms` 幾乎相同**，通常表示通訊仍被計算遮蔽，拓撲差異尚未顯現。
- 若 **communication / wall time ratio 上升** 且 `sim_t_step_ms` 開始分化，通常表示已進入拓撲敏感區間。
- 若某一拓撲在 **相近 communication ratio 下仍有較低的 `sim_t_step_ms`**，可解讀為該拓撲在此工作負載下具有較佳的通訊效率或負載平衡效果。

---

## 7. 支援量測工具

以下腳本拆解 AllReduce 成本的*實測*側，使 §5 的 ns-3 落差可被歸因，而非僅能推測。輸出都落在 `runs/calibration/`。

| 腳本 | 量測內容 | 輸出 |
|---|---|---|
| `bucket_micro_allreduce.py` | 以 ET 解出的 DDP bucket 尺寸執行 `torch.distributed.all_reduce`，GPU 上沒有其他工作——無競爭下限。刻意走 PyTorch 路徑（而非 rccl-tests），因為那才是 trace 紀錄的路徑；每次 `all_reduce` 各自用一對 CUDA event 包起來，對應 Kineto 回報單一 kernel 的方式 | `q2_micro.csv` |
| `q4_overlap_off.py` | 同樣的 bucket，放在真實訓練迴圈內，但等 backward 完全結束後才發出——有框架成本、無競爭。第一階段以 DDP comm hook 記錄並印出每步的 bucket 尺寸，而非臆測 `bucket_cap_mb`。要確認兩邊量的是同一批 collective，請將該輸出與 ET 中記錄的尺寸清單比對——用 `python src/tests/validate_et.py --prefix <tag>` 取得（需在容器內執行，它會 import `chakra`） | `q4_overlap_off.csv` |
| `fit_envelope.py` | 對 rccl-tests 掃描與 ns-3 逐 collective 的 COMM interval 各自擬合 `T(M) = α + M/B`，再逐 bucket 相減。模型對參數為線性，普通最小平方即為精確解，不需要 scipy | `step1_rccl_sweep.csv`、`ns3_collective_times.csv`、`envelope_fit_table.md` |
| `gen_envelope_figures.py` | 畫出實測 RCCL 路徑與 ns-3 的 `T(M)` 與 `BW(M)` | `fig_envelope_T.png`、`fig_envelope_BW.png` |
| `gen_figures_science.py` | 論文圖表（IEEE 樣式） | `thesis_figures/` |

三個欄位合起來是：無競爭（`q2_micro`）、有框架成本但無競爭（`q4_overlap_off`）、以及與 backward 競爭的實地執行（Kineto trace）。相減即可分離出實測 RCCL kernel 時間中有多少是 ns-3 有建模的資料傳輸，有多少是它沒建模的同步等待。

`fit_envelope.py` 的 ns-3 側輸入來自 `rocm/patches/statistics_comm_intervals.py` 加進 ASTRA-sim statistics pass 的 `COMM interval` 行，因此容器必須以 `ASTRA_PATCHES=all`（預設值）建置，這些輸入才會存在。

```bash
# 在 rocm-horovod 容器內執行；/workspace/runs 為 bind mount
torchrun --nproc_per_node=2 scripts/bucket_micro_allreduce.py \
    --sizes-file runs/calibration/bucket_sizes.json \
    --out runs/calibration/q2_micro.csv

torchrun --standalone --nproc_per_node=2 scripts/q4_overlap_off.py \
    --out runs/calibration/q4_overlap_off.csv

# 在 repo 根目錄執行
python scripts/fit_envelope.py \
    --rccl-log runs/calibration/step1_rccl_sweep_raw.log \
    --sizes runs/calibration/bucket_sizes.json \
    --out-csv runs/calibration/step1_rccl_sweep.csv
python scripts/gen_envelope_figures.py
```
