# ROCm-ASTRAsim：AMD GPU AI 叢集追蹤驅動模擬框架

> **繁體中文** | [English](README.md)

> **論文：** *《Cost-Effective Twisted Torus for AI Training: An ASTRA-sim Evaluation Using Traces from Consumer-Grade AMD GPUs with ROCm》*
> 國立成功大學資訊工程研究所，2026

本 Repository 實作一套三階段的追蹤驅動 (trace-driven) 模擬 Pipeline，以 **AMD ROCm/RCCL 實體硬體**收集訓練追蹤資料並送入 **ASTRA-sim** 進行叢集規模網路模擬。其研究定位對應的是：既有公開 ASTRA-sim 研究多半假設 **NVIDIA CUDA/NCCL** 平台，而本工作補足 AMD ROCm/RCCL 的端到端流程文件化。

---

## 目錄結構

```
.
├── src/
│   ├── train_rocm_pytorch.py      # 階段 1 — DDP 訓練 + Kineto 追蹤生成
│   ├── conver_to_chakra_et.py     # 階段 2 — Kineto JSON → Chakra ET（含 AMD 修補）
│   ├── scale_et_comm_workload.py  # 工作負載擴增（All-to-All 壓力測試）
│   ├── topology_generator.py      # Torus / Twisted Torus 拓撲檔案產生器
│   └── rocm_compat.py             # ROCm GPU 頻率監控工具
├── scripts/
│   ├── run_ns3.py                 # 階段 3 — ASTRA-sim ns-3 執行與校準
│   ├── README.md                  # 校準方法論與 run_ns3.py 參數說明
│   └── commands.md                # 所有實驗的完整指令參考
├── configs/astra-sim/
│   ├── system/                    # 各拓撲的 ASTRA-sim 系統設定
│   ├── topos/                     # ns-3 實體拓撲檔案
│   └── ns3/                       # ns-3 網路層參數設定
├── data/chakra/
│   ├── pytorch_traces/            # （輸入）階段 1 產生的 Kineto JSON 追蹤
│   ├── gpu_metrics/               # （輸入）GPU 頻率紀錄
│   └── workload_et/               # （輸出）Chakra ET 檔案（.et）
├── docs/
│   ├── astra-sim/                 # ASTRA-sim 設定文件
│   └── archive/                   # 開發過程的除錯報告（歷史紀錄）
├── runs/                          # 模擬結果 + calibration_all.csv
├── tutorials/                     # 學術教學範例（MICRO'24、ASPLOS'23）
├── viz/                           # 互動式 3D 拓撲視覺化（Twisted Torus）
├── rocm/dockerfile                # Docker 環境（ROCm + PyTorch + ASTRA-sim）
└── docker-compose.yaml
```

---

## 硬體平台

所有實體測量在以下環境進行：

| 元件 | 規格 |
|---|---|
| CPU | AMD Ryzen 7 5800X |
| GPU | 2× AMD Radeon RX 9070 XT（Navi 48，16 GB GDDR6） |
| GPU 互連 | PCIe Gen4 x8（透過主機 PCIe Root Complex） |
| 作業系統 | Ubuntu 24.04 |
| 容器映像 | `rocm/pytorch:rocm6.4.4_ubuntu24.04_py3.12_pytorch_release_2.7.1` |

**實測物理層參數：**

| 參數 | 數值 | 量測工具 |
|---|---|---|
| 節點間有效頻寬 | 65 Gbps | `rccl-tests` 512 MB AllReduce |
| 每條鏈路有效延遲 | 14 µs | `rccl-tests` 4 B + 實驗校準 |
| 本地 GPU 記憶體頻寬 | 540 GB/s | `rocm-bandwidth-test` |

> **消費級 GPU 限制說明：** AMD Radeon（RDNA 架構）GPU 不支援 GPUDirect RDMA。所有節點間傳輸均須經由主機 CPU 的系統記憶體中轉（bounce-buffer）。校準後的 14 µs 有效延遲吸收了這部分軟體堆疊開銷，而非單純反映實體傳播延遲。

---

## 三階段 Pipeline

### 階段 1 — 追蹤收集（`src/train_rocm_pytorch.py`）

在 ROCm 環境下執行 PyTorch DDP 訓練，透過 Kineto Profiler 產生每個 rank 的 `host_*.json` 和 `device_*.json` 追蹤檔案。

```bash
# ResNet-50（頻寬限制型，主要校準工作負載）
torchrun --standalone --nproc_per_node=2 ./src/train_rocm_pytorch.py \
  --model resnet50 --workers 4 \
  --trace-wait 32 --trace-steps 2 \
  --model-tag resnet50

# Simple CNN / CIFAR-10（延遲限制型，適用範圍邊界診斷）
torchrun --standalone --nproc_per_node=2 ./src/train_rocm_pytorch.py \
  --model cifar10 --workers 0 \
  --trace-wait 32 --trace-steps 4 \
  --model-tag cifar10
```

**輸出：** `data/chakra/pytorch_traces/host_0_resnet50.json`、`device_0_resnet50.json`……

**重要參數：**
- `--inject-sync-hack` — 注入額外同步事件，穩定 ROCm 上的 `chakra_trace_link`（建議開啟）
- `--trace-steps 1–4` — 控制追蹤規模；過大的追蹤檔可能顯著拉長 ns-3 執行時間或造成不穩定

### 階段 2 — 追蹤轉換（`src/conver_to_chakra_et.py`）

將 Kineto JSON 轉換為 Chakra ET（`.et`）格式。在 Chakra upstream commit `df5204c` 加入 HIP kernel 辨識支援的基礎上，額外套用**兩項 AMD 專用修補**：

| 修補 | 問題 | 解法 |
|---|---|---|
| **修補 1 — RCCL 節點分類** | `ncclDevKernel_Generic` 被誤判為 `COMP_NODE` | 攔截 `get_protobuf_node_type_from_json_node`，將所有 `ncclDevKernel_Generic*` 強制歸類為 `COMM_COLL_NODE` |
| **修補 2 — RCCL 集合通訊類型** | Generic kernel 名稱不含集合操作類型資訊 | 將所有 `ncclDevKernel_Generic*` 對映至 `ALL_REDUCE`（DDP 工作負載適用） |
| **DAG 修復 Pass** | 自依賴、循環依賴、懸空參照會導致 ASTRA-sim ETFeeder 崩潰 | DFS 循環偵測 + 自依賴移除 + 懸空參照清除 |

```bash
# ResNet-50 — 標準模式（使用追蹤中的真實 kernel 執行時間）
python ./src/conver_to_chakra_et.py --model-tag resnet50

# CIFAR-10 — 系統感知校準模式
# 若進行 latency-dominated 診斷，可視需要再搭配 system-aware forcing
python ./src/conver_to_chakra_et.py --model-tag cifar10
```

**輸出：** `data/chakra/workload_et/et.resnet50.0.et`、`et.resnet50.1.et`……

### 階段 3 — 模擬執行（`scripts/run_ns3.py`）

協調 ASTRA-sim + ns-3，執行設定檔生成、虛擬節點擴展、模擬執行與自動校準。

```bash
# 2-GPU 校準執行（ResNet-50，約 39 分鐘）
python ./scripts/run_ns3.py \
  --workload data/chakra/workload_et --model-tag resnet50 \
  --topo auto:1d \
  --phys-topo configs/astra-sim/topos/2_nodes_1_switch_topology.txt \
  --coll-opt localBWAware --lmbw 540

# 128-GPU Torus 模擬
python scripts/run_ns3.py \
  --workload data/chakra/workload_et --model-tag resnet50 \
  --topo file:configs/astra-sim/topos/logical_128nodes_Torus_4x4x8.json \
  --phys-topo configs/astra-sim/topos/128nodes_Torus_4x4x8.txt \
  --system configs/astra-sim/system/system_128nodes_Torus_4x4x8.json \
  --virtual-world 128 --lmbw 540 --no-autocalib

# 128-GPU Twisted Torus 模擬
python scripts/run_ns3.py \
  --workload data/chakra/workload_et --model-tag resnet50 \
  --topo file:configs/astra-sim/topos/logical_128nodes_TwistedTorus_4x4x8.json \
  --phys-topo configs/astra-sim/topos/128nodes_TwistedTorus_4x4x8.txt \
  --system configs/astra-sim/system/system_128nodes_TwistedTorus_4x4x8.json \
  --virtual-world 128 --lmbw 540 --no-autocalib

# 128-GPU Fat-Tree 模擬
python scripts/run_ns3.py \
  --workload data/chakra/workload_et --model-tag resnet50 \
  --topo file:configs/astra-sim/topos/logical_128nodes_FatTree_L16_S8.json \
  --phys-topo configs/astra-sim/topos/128nodes_FatTree_L16_S8.txt \
  --system configs/astra-sim/system/system_128nodes_FatTree_L16_S8.json \
  --virtual-world 128 --lmbw 540 --no-autocalib
```

完整指令參考（含 All-to-All 壓力測試）請見 [scripts/commands.md](scripts/commands.md)。

---

## 校準

校準在 2-GPU 規模下執行，產生校準因子 α（µs/cycle），用於將模擬 cycle 數換算為真實時間。校準的主要目的在於透過系統性的參數敏感度分析，確認模擬適合用於**相對拓撲比較**，而非宣稱通訊時間的絕對精準預測。

- **ResNet-50（頻寬限制型）：** 主要校準基準。wall-clock 校準得到 **α_step = 0.002411 µs/cycle**。ns-3 的通訊時間相對實測 RCCL GPU kernel duration 約有 **+100% 高估**，最合理的解釋是 ns-3 的 Ethernet RDMA 傳輸模型與實體平台的 PCIe DMA 路徑不一致。
- **CIFAR-10（延遲限制型）：** 排除於大規模評估之外。此 workload 的 step time 主要由未建模的軟體堆疊開銷主導，因此 ASTRA-sim 不適合做絕對時間預測。

**實測校準結果（2-GPU，每個訓練步驟）：**

| 指標 | ResNet-50 | CIFAR-10 |
|---|---|---|
| `ns3_comm_ms`（ns-3 模擬值） | **15.06 ms** | **41.07 ms** |
| `real_t_comm_ms`（硬體實測值） | **7.54 ms** | **61.98 ms** |
| ns-3 vs real | **+100%**（高估） | **−34%**（低估） |
| 校準狀態 | 主基準 | 排除（scope boundary） |

> 針對 ResNet-50，論文進一步掃描頻寬、延遲、封包 payload 與擁塞控制設定，發現 ns-3 通訊時間穩定落在 14.0–15.1 ms，支持此差異主要是結構性偏差而非參數敏感度問題。對拓撲比較而言，這種系統性偏差預期會同樣作用於所有拓撲，因此不影響相對排序。

α 值與每次執行的校準結果記錄於 `runs/calibration_all.csv`。完整校準方法論請見 [scripts/README.md](scripts/README.md)。

### 額外工作負載驗證：Qwen2.5-0.5B

此 Pipeline 也已在相同 2-GPU AMD 平台上，以 **Qwen2.5-0.5B** 的 DDP trace 進行驗證。這個 LLM workload 具有 **494M 參數**，每個 training step 產生 **37 個 AllReduce 通訊節點**，總通訊量約 **1.84 GiB**。這說明 trace 蒐集與轉換流程並不限於 ResNet-50；至於 Qwen workload 的 128 節點拓撲評估，則保留為後續工作。

---

## 拓撲設定

已針對 128 節點評估預先設定三種拓撲：

此外，若需要自訂或重新產生 topology 與對應設定檔，也可以直接使用 `src/topology_generator.py` 透過指令自動產生。此工具可一鍵輸出：

- 物理拓撲檔（`configs/astra-sim/topos/*.txt`）
- 邏輯拓撲檔（`configs/astra-sim/topos/logical_*.json`）
- 系統設定檔（`configs/astra-sim/system/system_*.json`）

支援 **Torus**、**Twisted Torus** 與 **Fat-Tree**。例如：

```bash
# 產生 128 節點 4×4×8 Torus
python3 src/topology_generator.py \
  --type torus \
  --nodes 128 \
  --dims 4 4 8 \
  --bw-intra 65Gbps --lat-intra 0.014ms \
  --bw-inter 25Gbps --lat-inter 0.005ms

# 產生 128 節點 4×4×8 Twisted Torus
python3 src/topology_generator.py \
  --type twisted_torus \
  --nodes 128 \
  --dims 4 4 8 \
  --bw-intra 65Gbps --lat-intra 0.014ms \
  --bw-inter 25Gbps --lat-inter 0.005ms

# 產生 128 節點 Fat-Tree
python3 src/topology_generator.py \
  --type fattree \
  --nodes 128 \
  --bw-intra 65Gbps --lat-intra 0.014ms \
  --bw-inter 65Gbps --lat-inter 0.005ms
```

若只想快速重現本研究所使用的 topology，直接使用 `configs/astra-sim/topos/` 與 `configs/astra-sim/system/` 內既有檔案即可；若要修改維度、頻寬、延遲或拓撲型別，則可使用這個產生器重新輸出對應檔案。

| 參數 | Fat-Tree（L16_S8） | Torus（4×4×8） | Twisted Torus（4×4×8） |
|---|---|---|---|
| 實體交換器數量 | **24** | **0** | **0** |
| 節點間頻寬（Z 軸 / 節點內） | 65 Gbps | 65 Gbps | 65 Gbps |
| 節點間頻寬（X/Y 軸 / 節點間） | 65 Gbps | **25 Gbps** | **25 Gbps** |
| 每條鏈路延遲 | GPU→Leaf: 14 µs；Leaf→Spine: 5 µs | Z: 14 µs；X,Y: 5 µs | Z: 14 µs；X,Y: 5 µs |
| 集合通訊演算法 | halvingDoubling | ring × 3 | ring × 3 |
| Twist（X 軸繞回） | — | 無 | Y 偏移 +1 |

**Twisted Torus 繞線定義**（X 軸 wrap-around）：
```
(x=3, y, z) → (x=0, (y+1) mod 4, z)
```

---

## All-to-All 壓力測試（工作負載擴增）

使用修正後的原始 ResNet-50 追蹤執行 AllReduce 時，通訊通常被 GPU 計算完全掩蓋，拓撲間的差異無從觀察。每個 step 約有 **89.7 MiB** 的總 AllReduce 通訊量（分散於 5 個 DDP gradient buckets）。若要對網路施加壓力並使拓撲差異可見，可使用 `src/scale_et_comm_workload.py` 放大通訊量。

### 功能說明

`scale_et_comm_workload.py` 讀取指定前綴的 ET 檔案集，對每個 `COMM_COLL_NODE` 進行就地修補：

1. **`comm_type`** → 強制設定為 `ALL_TO_ALL`（從原本的 `ALL_REDUCE`）
2. **`comm_size`** → 設定為指定的位元組數（例如 1 GB = 1,073,741,824 bytes）

原始計算節點與 DAG 結構保持不變，模擬仍能保留真實的計算與通訊交錯模式，僅改變通訊語意與資料量。

### 檔案命名規則

```
輸入：et.<prefix>.<rank>.et         （例如 et.resnet50_all2all.0.et）
輸出：et.<prefix><suffix>.<rank>.et  （例如 et.resnet50_all2all_1GB.0.et）
```

若未指定 `--suffix`，後綴依 `--bytes` 自動產生：

| `--bytes` | 自動後綴 |
|---|---|
| `1G` / `1073741824` | `_1GB` |
| `128MB` / `128M` | `_128MB` |
| `512K` | `_512KB` |

### 使用方式

```bash
# 步驟 1 — 轉換原始 ResNet-50 追蹤（若尚未執行）
python src/conver_to_chakra_et.py --model-tag resnet50_all2all

# 步驟 2 — 放大為 1 GB All-to-All（產生 et.resnet50_all2all_1GB.*.et）
python src/scale_et_comm_workload.py \
  --workload-dir data/chakra/workload_et \
  --prefix resnet50_all2all \
  --bytes 1G

# 步驟 3 — 使用放大後的工作負載執行模擬（--payload 12000 控制 ns-3 事件數量）
python scripts/run_ns3.py \
  --workload data/chakra/workload_et \
  --model-tag resnet50_all2all_1GB \
  --topo file:configs/astra-sim/topos/logical_128nodes_TwistedTorus_4x4x8.json \
  --phys-topo configs/astra-sim/topos/128nodes_TwistedTorus_4x4x8.txt \
  --system configs/astra-sim/system/system_128nodes_TwistedTorus_4x4x8_direct.json \
  --virtual-world 128 --payload 12000 --lmbw 540 --no-autocalib
```

> **注意：** 可觀察性具有三個階段：原始 ~89.7 MiB trace 完全被遮蔽、100 MB 開始只讓標準 Torus 先暴露、512 MB–1 GB 則讓三種拓撲完全分化。因此 1 GB All-to-All 應視為 simulation-only 的上界壓力測試，而非實際生產工作負載。

---

## 環境設定

### Docker（建議）

```bash
docker-compose up
# 指定特定 ROCm / PyTorch 版本：
VERSION=rocm6.4.4_ubuntu24.04_py3.12_pytorch_release_2.7.1 docker-compose up
```

### 環境驗證

```bash
# 1. 硬體層
rocm-bandwidth-test

# 2. 通訊層
rccl-tests/build/all_reduce_perf -b 512M -e 512M -f 2 -g 2

# 3. 框架層
torchrun --standalone --nproc_per_node=2 src/train_rocm_pytorch.py --model resnet50 --epochs 1

# 4. 追蹤格式確認
python src/tests/check_trace_ready.py
python src/tests/validate_et.py
```

---

## 拓撲視覺化

互動式 3D Twisted Torus 拓撲視覺化工具：

```
viz/twisted_torus_3d.html
```

以任意瀏覽器開啟，可瀏覽 4×4×8 Twisted Torus 的完整繞線圖。

---

## 常見問題

**Q：ns-3 模擬看起來很久沒有結束，是不是卡死了？**
A：不一定。對真實拓樸與較大的 ET 檔案而言，模擬時間可能非常長；以本研究的 128-node 實驗為例，**單一實驗平均約需 5 天** 才會產出完整結果。因此不能用「跑了 10 分鐘還沒結束」來判斷卡死。

建議先檢查輸出目錄中的 `fct.txt` 是否仍持續產生內容，例如：

```text
runs/20260324-013221+0800_ns3_128gpu_qwen05b_file_logical_128nodes_FatTree_L16_S8/out/fct.txt
```

如果 `fct.txt` 持續有新數值寫入，通常代表模擬仍在正常進行，而不是卡死。

真正可能導致異常的情況仍包括 ET 檔案過大；生成追蹤時建議將 `--trace-steps` 控制在 1–4，因為非常大的 ET 檔可能耗盡 ASTRA-sim ETFeeder 的資源，或顯著拖慢模擬速度。另外，由於每個實驗可能執行數天，你也可以使用**多個 shell 同時跑多個實驗**，提高整體實驗效率。

**Q：ASTRA-sim 出現 `"Node X in ctrl_dep graph, but not found in index"` 錯誤？**
A：ET 檔案的 DAG 完整性異常（自依賴或循環依賴）。重新執行 `conver_to_chakra_et.py`，內建的 DAG 修復 Pass（`fix_et_dag_inplace`）應可自動解決。

**Q：ROCm 上 `chakra_trace_link` 因時間戳對不齊而失敗？**
A：在 `train_rocm_pytorch.py` 加上 `--inject-sync-hack`。此選項會注入同步事件，對齊 CPU（毫秒）與 GPU（微秒）的時間軸。

**Q：為什麼 ns-3 會把 ResNet-50 的通訊時間高估約 2 倍？**
A：這個差異對所有測試參數都不敏感，最合理的解釋是傳輸路徑不匹配：ns-3 使用 Ethernet RDMA 模型，而實體 2-GPU 平台實際走的是 PCIe DMA 路徑。由於同一個 ns-3 傳輸模型套用在所有拓撲上，這個偏差預期會在相對拓撲比較中相互抵消。

**Q：為什麼 CIFAR-10 被排除在大規模評估之外？**
A：因為它超過一半的 step time 都落在 ASTRA-sim 未建模的軟體堆疊開銷，導致 wall-clock 與 communication calibration factor 發散，因此不適合用來做大規模絕對時間預測。詳見論文第 4.3 節。

---

## 歷史開發報告

Pipeline 開發過程中遭遇並解決的問題，已記錄於 [`docs/archive/`](docs/archive/)。這些報告與正常使用無關，但有助於理解 AMD 相容性開發過程。

| 檔案 | 內容 |
|---|---|
| [ASTRA-sim_Analysis_Report.md](docs/archive/ASTRA-sim_Analysis_Report.md) | Alpha 校準分析、計算 cycle 解析問題、ns-3 卡死調查 |
| [AMD_GPU_ASTRA_SIM_Integration_Complete_Report.md](docs/archive/AMD_GPU_ASTRA_SIM_Integration_Complete_Report.md) | HIP Runtime 不相容、RCCL Kernel 命名差異、DAG 修復——初期技術突破報告 |

---

## 引用

若使用本 Pipeline 或相關模擬結果，請引用：

```bibtex
@mastersthesis{chen2026twisted,
  author  = {jjasoncool},
  title   = {Cost-Effective Twisted Torus for AI Training: An ASTRA-sim Evaluation
             Using Traces from Consumer-Grade AMD GPUs with ROCm},
  school  = {National Cheng Kung University},
  year    = {2026},
  note    = {Code available at \url{https://github.com/jjasoncool/ROCm-ASTRAsim}}
}
```

---

## 相關資源

- [ASTRA-sim](https://github.com/astra-sim/astra-sim) — 分散式 ML 訓練模擬器
- [Chakra](https://github.com/mlcommons/chakra) — Meta 開發的執行追蹤標準格式
- [ns-3](https://www.nsnam.org/) — 封包層級網路模擬器
- [RCCL](https://github.com/ROCm/rccl) — ROCm 集合通訊函式庫
- [rccl-tests](https://github.com/ROCm/rccl-tests) — RCCL 微基準測試
- [TPU v4 論文](https://dl.acm.org/doi/10.1145/3579371.3589350) — Google Twisted Torus 參考文獻（ISCA'23）
