# ROCm-ASTRAsim：AMD GPU AI 叢集追蹤驅動模擬框架

> **繁體中文** | [English](README.md)

> **論文：** *《基於AMD ROCm及追蹤驅動模擬器ASTRA-sim之具成本效益人工智慧訓練在環面拓撲上之效能評估》*
> *(Cost-Effective AI Training Performance Evaluation for Torus Topology based on AMD ROCm and the Trace-Driven simulator ASTRA-sim)*
> 國立成功大學 電機資訊學院 資訊工程學系,2026

一套三階段的 trace-driven 模擬 pipeline,從 AMD ROCm/RCCL 實體硬體收集訓練追蹤,再送進 ASTRA-sim 做叢集規模的網路模擬。多數已發表的 ASTRA-sim 研究假設的是 NVIDIA CUDA/NCCL,這裡補上 AMD ROCm/RCCL 這條路徑。

論文在 128 節點規模下評估 Fat-Tree、標準 3D Torus、Twisted Torus 三種拓撲,涵蓋**四種通訊強度**:

1. **計算主導 AllReduce** — ResNet-50 DDP(每 step ~89.7 MiB)
2. **通訊密集 AllReduce** — Qwen2.5-0.5B DDP(每 step ~1.84 GiB)
3. **階層式 TP+DDP** — Qwen2.5-1.5B(TP=8 × DDP=16)
4. **All-to-All 頻寬飽和** — 合成壓力測試(每次集合 1 GB)

論文的結論是:twist 的價值取決於 workload——它對頻寬受限的 All-to-All 有幫助(+18%),但對通訊密集的 DDP AllReduce 造成結構性效能損失(ring +77.9%、halving-doubling +74.7%),而且本研究評估的兩種集合演算法都消不掉。建議的預設部署是**標準 Torus**:在通訊密集 DDP 下與 Fat-Tree 差距不超過 0.6%,叢集成本約為其 58%,同時避開 twist 的損失。由於標準 Torus 與 Twisted Torus 硬體完全相同(只差線纜佈線),日後若量測顯示工作負載確實以 All-to-All 為主,可零硬體成本改佈線。

這裡的設定檔可以讓你自己重現這個比較,完整分析在論文第 5 章,參考數值整理在[128 節點結果](#128-節點結果)。README 裡引用的數字都來自單次模擬,只能當參考,不是保證;要採用前請先在自己的環境重跑。

---

## 目錄結構

```
.
├── src/
│   ├── train_rocm_pytorch.py      # 階段 1 — DDP 訓練 + Kineto 追蹤(CIFAR-10 / ResNet-50 / Qwen 0.5B)
│   ├── train_rocm_tensor.py       # 階段 1 — TP=2 訓練 + Kineto 追蹤(Qwen 1.5B,用於 TP+DDP)
│   ├── conver_to_chakra_et.py     # 階段 2 — Kineto JSON → Chakra ET(含 AMD 修補,可選 --add-ddp)
│   ├── add_ddp_to_et.py           # 階段 2 輔助 — 將 DDP AllReduce 節點附加到 TP ET(用於 TP+DDP)
│   ├── scale_et_comm_workload.py  # 工作負載擴增(All-to-All 壓力測試)
│   ├── topology_generator.py      # Torus / Twisted Torus / Fat-Tree 拓撲檔產生器
│   └── rocm_compat.py             # ROCm GPU 頻率監控工具
├── scripts/
│   ├── run_ns3.py                 # 階段 3 — ASTRA-sim ns-3 執行與校準
│   ├── calibrate_from_runs.py     # 由既有 run 目錄重算校準,不需重跑模擬
│   ├── bucket_micro_allreduce.py  # 以 ET 內的 DDP bucket 尺寸量測無競爭 AllReduce
│   ├── q4_overlap_off.py          # 同一批 bucket,取自真實訓練步,逐一單獨計時,無重疊
│   ├── fit_envelope.py            # 對 RCCL 路徑與 ns-3 各自擬合 T(M) = α + M/B 後逐 bucket 比對
│   ├── gen_envelope_figures.py    # T(M) / BW(M) envelope 圖
│   ├── gen_figures_science.py     # 論文圖表(IEEE 樣式)
│   ├── README.md                  # 校準方法論與 run_ns3.py 參數說明
│   └── commands.md                # 四個實驗的完整指令參考
├── configs/astra-sim/
│   ├── system/                    # 各拓撲的 ASTRA-sim 系統設定(含 chunk / 演算法變體)
│   ├── topos/                     # ns-3 物理拓撲 + ASTRA-sim 邏輯拓撲檔
│   └── ns3/                       # ns-3 網路層參數設定
├── data/chakra/
│   ├── pytorch_traces/            # (輸入)階段 1 產生的 Kineto JSON 追蹤
│   ├── gpu_metrics/               # (輸入)GPU 頻率紀錄
│   ├── models/                    # (輸入)HuggingFace 模型快取(供 TP+DDP 自動偵測參數量)
│   └── workload_et/               # (輸出)Chakra ET 檔案 (.et)
├── deadlock-reproduction/         # ASTRA-sim Issue #370 的最小重現包(設定檔 + 證據)
├── docs/                          # ASTRA-sim 設定文件與歷史報告
├── runs/                          # 模擬結果 + calibration_aligned.csv + calibration/
├── tutorials/                     # 學術教學範例(MICRO'24、ASPLOS'23)
├── viz/                           # 互動式 3D Twisted Torus 拓撲視覺化
├── rocm/
│   ├── dockerfile                 # Docker 環境(ROCm + PyTorch + ASTRA-sim + Chakra)
│   └── patches/                   # 版本化的 ASTRA-sim 原始碼修補(見 ASTRA_PATCHES)
└── docker-compose.yaml
```

---

## 硬體平台

所有實體測量在以下環境進行:

| 元件 | 規格 |
|---|---|
| CPU | AMD Ryzen 7 5700X |
| GPU | 2× AMD Radeon RX 9070 XT(Navi 48,16 GB GDDR6) |
| GPU 互連 | PCIe Gen4 x8(透過主機 PCIe Root Complex) |
| 作業系統 | Ubuntu 24.04 |
| 容器映像 | `rocm/pytorch:rocm6.4.4_ubuntu24.04_py3.12_pytorch_release_2.7.1` |

**實測物理層參數:**

| 參數 | 數值 | 量測工具 |
|---|---|---|
| 節點間有效頻寬 | 65 Gbps | `rccl-tests` 512 MB AllReduce |
| 每條鏈路有效延遲 | 14 µs | `rccl-tests` 4 B + 實驗校準 |
| 本地 GPU 記憶體頻寬 | 540 GB/s | `rocm-bandwidth-test` |

> **消費級 GPU 限制:** AMD Radeon(RDNA)GPU 不支援 GPUDirect RDMA,所有節點間傳輸都得經主機 CPU 的系統記憶體中轉(bounce-buffer)。校準後的 14 µs 有效延遲把這部分軟體堆疊開銷也算了進去,不是單純的實體傳播延遲。

---

## 三階段 Pipeline

### 階段 1 — 追蹤收集

四個論文實驗對應兩支 trace 收集腳本:

| 腳本 | 工作負載 | 用途 |
|---|---|---|
| `src/train_rocm_pytorch.py` | `cifar10`、`resnet50`、`qwen05b`、`llama1b` | DDP 訓練(實驗 1、2 與診斷) |
| `src/train_rocm_tensor.py`  | Qwen2.5-1.5B 搭配 `parallelize_module`(TP=2) | TP trace,用於實驗 3(TP+DDP) |

兩支腳本都使用 PyTorch Kineto Profiler,產生每個 rank 的 `host_*.json` / `device_*.json` 追蹤檔。

```bash
# 實驗 1 — ResNet-50 DDP(計算主導,主要校準工作負載)
torchrun --standalone --nproc_per_node=2 ./src/train_rocm_pytorch.py \
  --model resnet50 --workers 4 \
  --trace-wait 32 --trace-steps 2 \
  --inject-sync-hack

# 實驗 2 — Qwen2.5-0.5B DDP(通訊密集,每 step 約 1.84 GiB)
torchrun --standalone --nproc_per_node=2 ./src/train_rocm_pytorch.py \
  --model qwen05b --batch-size 4 --workers 0 \
  --seq-len 256 \
  --trace-wait 10 --trace-steps 2 \
  --inject-sync-hack

# 實驗 3 — Qwen2.5-1.5B,TP=2(在 2 GPU 上收集;後續複製/縮放至 TP=8 × DDP=16)
torchrun --standalone --nproc_per_node=2 ./src/train_rocm_tensor.py \
  --epochs 3 --batch-size 1 --workers 0 \
  --seq-len 256 \
  --trace-wait 10 --trace-steps 2 \
  --inject-sync-hack

# 診斷用 — Simple CNN / CIFAR-10(latency-bound,排除於 128 節點評估)
torchrun --standalone --nproc_per_node=2 ./src/train_rocm_pytorch.py \
  --model cifar10 --workers 0 \
  --trace-wait 32 --trace-steps 4 \
  --inject-sync-hack
```

**輸出(每個實驗一組):**
`data/chakra/pytorch_traces/host_<rank>_<model>.json`、`device_<rank>_<model>.json`、
`data/chakra/gpu_metrics/gpu_metrics_<rank>_<model>.json`。

**重要參數:**
- `--inject-sync-hack` — 注入額外同步事件,穩定 ROCm 上的 `chakra_trace_link`(建議開啟)
- `--trace-steps 1–4` — 控制追蹤規模;過大的追蹤檔會顯著拉長 ns-3 執行時間或耗盡 ASTRA-sim ETFeeder 資源
- `--seq-len` — LLM 序列長度(僅 `qwen05b` / `llama1b` / Qwen 1.5B TP 使用)。256 可兼顧檔案大小與真實通訊量

### 階段 2 — 追蹤轉換(`src/conver_to_chakra_et.py`)

將 Kineto JSON 轉換為 Chakra ET(`.et`)格式。在 Chakra upstream commit `df5204c` 加入 HIP kernel 辨識支援的基礎上,額外套用**兩項 AMD 專用修補**:

| 修補 | 問題 | 解法 |
|---|---|---|
| **修補 1 — RCCL 節點分類** | `ncclDevKernel_Generic` 被誤判為 `COMP_NODE` | 攔截 `get_protobuf_node_type_from_json_node`,將所有 `ncclDevKernel_Generic*` 強制歸類為 `COMM_COLL_NODE` |
| **修補 2 — RCCL 集合通訊類型** | Generic kernel 名稱不含集合操作類型資訊 | 將所有 `ncclDevKernel_Generic*` 對映至 `ALL_REDUCE`(DDP 工作負載適用) |
| **DAG 修復 Pass** | 自依賴、循環依賴、懸空參照會導致 ASTRA-sim ETFeeder 崩潰 | DFS 循環偵測 + 自依賴移除 + 懸空參照清除 |

只做分類還不夠:`ncclDevKernel_Generic` 同樣不帶**傳輸量**資訊。這個缺口是在更前一階段補上的——在 profiler 而非 converter。訓練腳本註冊一個帶標記的 DDP comm hook(`train_rocm_pytorch.py` 裡的 `make_tagging_allreduce_hook`),把每次 RCCL 集合操作包進 `record_function` 註記 `nccl:all_reduce|bytes=<N>|pg=dp0`,converter 再從中解析出 `comm_type` 與 `comm_size`。`train_rocm_tensor.py` 以同樣機制標記張量平行的集合操作(`dist.all_reduce` 與 `torch.distributed._functional_collectives.all_reduce`,即 DTensor `parallelize_module` 實際走的路徑),標籤為 `pg=tp0`。計算 kernel 不需要任何修補——時間直接讀自 Kineto device trace。

```bash
# ResNet-50 DDP — 標準模式(使用追蹤中的真實 kernel 時間)
python ./src/conver_to_chakra_et.py --model-tag resnet50

# Qwen 0.5B DDP — 同樣為標準模式
python ./src/conver_to_chakra_et.py --model-tag qwen05b

# Qwen 1.5B TP — 加入 DDP AllReduce + 將 TP=2 trace 縮放為 TP=8 模擬目標
# (從 data/models/ HuggingFace 快取自動偵測模型參數量)
python ./src/conver_to_chakra_et.py \
  --model-tag qwen15b_tp \
  --add-ddp --target-tp 8

# CIFAR-10 — 僅供診斷;若需 latency-bound 研究可搭配 --force-avg-kernel-ns
python ./src/conver_to_chakra_et.py --model-tag cifar10

# All-to-All 前置 — 將 resnet50 trace 以另一個 tag 複製出來
python ./src/conver_to_chakra_et.py --model-tag resnet50_all2all
```

**輸出:** `data/chakra/workload_et/et.<model_tag>.<rank>.et`。

對 TP+DDP 而言,`--add-ddp` 會將 DDP AllReduce 節點附加到 ET,`comm_size` 由自動偵測的模型參數量推算;同時將紀錄到的 TP=2 計算時間縮放至 TP=8 模擬目標。對應的獨立輔助腳本見 [src/add_ddp_to_et.py](src/add_ddp_to_et.py)。

### 階段 3 — 模擬執行(`scripts/run_ns3.py`)

協調 ASTRA-sim + ns-3,執行設定檔生成、虛擬節點擴展、模擬執行與自動校準。

```bash
# 2-GPU 校準執行(任一 workload 均可)。--et-iters 是「這個 ET 涵蓋幾次訓練迭代」
# (產生 ET 時使用的 --trace-steps);未給則 alpha_us 留空,不以推測值填補。
# 結果寫入 runs/calibration_aligned.csv。
python ./scripts/run_ns3.py \
  --workload data/chakra/workload_et --model-tag resnet50 \
  --topo auto:1d \
  --phys-topo configs/astra-sim/topos/2_nodes_1_switch_topology.txt \
  --coll-opt localBWAware --lmbw 540 --et-iters 1
```

若只想從既有的 run 目錄重算校準數值,不必重跑模擬:

```bash
python3 scripts/calibrate_from_runs.py --out runs/calibration_aligned.csv \
  resnet50=1:<run_dir> cifar10=4:<run_dir> qwen05b=2:<run_dir>
```

完整 128 節點實驗指令請見 [scripts/commands.md](scripts/commands.md),以下列出代表性範例:

```bash
# 實驗 1 — ResNet-50 DDP @ 128 節點(Torus / Twisted Torus / Fat-Tree)
python scripts/run_ns3.py \
  --workload data/chakra/workload_et --model-tag resnet50 \
  --topo file:configs/astra-sim/topos/logical_128nodes_Torus_4x4x8.json \
  --phys-topo configs/astra-sim/topos/128nodes_Torus_4x4x8.txt \
  --system configs/astra-sim/system/system_128nodes_Torus_4x4x8.json \
  --virtual-world 128 --lmbw 540 --no-autocalib

# 實驗 2 — Qwen 0.5B DDP,*必須*使用 active-chunks=4(避免 deadlock,詳見下節)
# 注意:--comm-scale 1.984375 = 127/64,是 128 節點模擬的「通訊工作點設定」,
# 不是對集合演算法的修正;同一實驗內對所有拓撲一致套用,不影響相對比較。
python scripts/run_ns3.py \
  --workload data/chakra/workload_et --model-tag qwen05b \
  --topo file:configs/astra-sim/topos/logical_128nodes_TwistedTorus_4x4x8.json \
  --phys-topo configs/astra-sim/topos/128nodes_TwistedTorus_4x4x8.txt \
  --system configs/astra-sim/system/system_128nodes_TwistedTorus_4x4x8_4chunks.json \
  --virtual-world 128 --lmbw 540 --comm-scale 1.984375 --no-autocalib --no-qlen

# 實驗 2(2×2 因子分析的 HD 那一臂)— Twisted Torus + X/Y 維度使用 Halving-Doubling
# 2×2 因子分析的 HD 那一臂,用來區分壅塞與拓撲路徑結構(HD 同時也可避開 deadlock)
python scripts/run_ns3.py \
  --workload data/chakra/workload_et --model-tag qwen05b \
  --topo file:configs/astra-sim/topos/logical_128nodes_TwistedTorus_4x4x8.json \
  --phys-topo configs/astra-sim/topos/128nodes_TwistedTorus_4x4x8.txt \
  --system configs/astra-sim/system/system_128nodes_TwistedTorus_4x4x8_4chunks_hd.json \
  --virtual-world 128 --lmbw 540 --comm-scale 1.984375 --no-autocalib --no-qlen

# 實驗 3 — Qwen 1.5B TP+DDP(TP=8 × DDP=16),Torus
python scripts/run_ns3.py \
  --workload data/chakra/workload_et --model-tag qwen15b_tp8ddp \
  --topo file:configs/astra-sim/topos/logical_128nodes_TP8_DDP16.json \
  --phys-topo configs/astra-sim/topos/128nodes_Torus_4x4x8.txt \
  --system configs/astra-sim/system/system_128nodes_Torus_TP8DDP.json \
  --virtual-world 128 --lmbw 540 --comm-scale 1.984 --no-autocalib --no-qlen

# 實驗 4 — All-to-All 1 GB 壓力測試(需先跑 scale_et_comm_workload.py)
python scripts/run_ns3.py \
  --workload data/chakra/workload_et --model-tag resnet50_all2all_1GB \
  --topo file:configs/astra-sim/topos/logical_128nodes_TwistedTorus_4x4x8.json \
  --phys-topo configs/astra-sim/topos/128nodes_TwistedTorus_4x4x8.txt \
  --system configs/astra-sim/system/system_128nodes_TwistedTorus_4x4x8.json \
  --virtual-world 128 --payload 12000 --lmbw 540 --no-autocalib
```

**論文實驗常用的 `run_ns3.py` 參數:**

| 參數 | 用途 |
|---|---|
| `--virtual-world N` | 將每個 rank 的 trace 複製擴展到 `N` 節點模擬(round-robin:偶數虛擬 rank 拿 rank-0 的 ET,奇數拿 rank-1;對映關係寫入 `expansion_map.json`) |
| `--comm-scale F`    | 將每個 COMM 節點的 `comm_size` 乘以 `F`。這是**工作負載的通訊工作點設定**,不是對集合演算法的修正——ASTRA-sim 會依設定的參與節點數自行拆解每個集合操作。論文用 `127/64 = 1.984375`;Qwen 0.5B 必須用精確分數,縮放後的 `comm_size` 才能被 `preferred-dataset-splits=4` 整除,TP+DDP 則可接受四捨五入的 `1.984`。同一實驗內對所有拓撲一致套用,不影響相對比較 |
| `--et-iters N`      | 這個 ET 涵蓋幾次訓練迭代(= 產生它時的 `--trace-steps`)。`alpha_us` 需要「每步」的分母,所以必須提供;未給則 `alpha_us` 留空,不以推測值填補 |
| `--comm-group FILE` | 直接傳給 `--comm-group-configuration`;未指定則完全不帶該參數 |
| `--calib-db PATH`   | 校準結果要追加的 CSV(預設 `runs/calibration_aligned.csv`) |
| `--no-qlen`         | 將 `qlen.txt` 導向 `/dev/null`,避免 128 節點時產生數百 GB 除錯輸出 |
| `--payload`         | 覆寫 ns-3 封包 payload(All-to-All 1 GB 壓力測試使用 `12000` 控制事件量) |
| `--no-autocalib`    | 停用自動 α 計算(僅在 2-GPU 校準時可用;128 節點必加此參數) |
| `--deadlock-timeout S` | 若 `fct.txt` 連續 `S` 秒未更新則自動 kill(預設 12 小時;對 ring 死鎖情境特別有用) |

---

## 校準

校準在 2-GPU 規模下執行,產生校準因子 α_step(µs/cycle),把模擬 cycle 數換算成真實時間。論文用它來支撐拓撲之間的相對比較,而不是預測絕對通訊時間。

**先對齊視窗,其他數字才有意義。** 實測側(Kineto trace 裡的 RCCL kernel)與模擬側(ET 上的 ns-3 通訊 cycle)除非對齊,否則涵蓋的工作量根本不同。`run_ns3.py` 會從 trace 的 `ProfilerStep` 區間還原迭代數,再與 `--et-iters` 對照;兩者無法以整數比對齊時直接停下,而不是輸出一個看起來合理的誤差,並且拒絕把「每迭代顆粒度差異」折進除數(這種情況改為標記旗標)。**在這套對齊機制存在之前產出的校準數字,與下表不可相互比較。**

**實測校準結果(2-GPU,已對齊視窗,每個訓練步驟):**

| 指標 | ResNet-50 | CIFAR-10 |
|---|---|---|
| `real_t_step_ms` | 662.92 ms | 224.32 ms |
| `real_t_net_comm_ms`(實測 RCCL kernel) | **15.87 ms** | **76.69 ms** |
| `real_t_kernel_ms`(GPU 計算) | 284.10 ms | 50.15 ms |
| comm / step | 2.4% | 34.2% |
| ns-3 通訊時間 | **15.06 ms** | **10.27 ms** |
| ns-3 vs 實測 | **−5.1%** | **−86.6%** |
| α_step(µs/cycle) | **0.002411** | 0.004550 |
| α_comm(µs/cycle) | 0.001054 | 0.007469 |
| ET 迭代數(`--et-iters`) | 1 | 4 |
| 校準狀態 | 主基準 | 排除(scope boundary) |

- **ResNet-50(頻寬主導型):** 主要基準。通訊只占 step time 的 2.4%,ns-3 的總和與實測 RCCL 總和差 −5.1%。這個聚合值內含互相抵銷的逐 collective 偏差——乾淨的傳輸被中度高估,而某個吸收了 backward 同步等待的 collective 被低估——所以它是*聚合*一致,不是逐 collective 準確。
- **CIFAR-10(延遲限制型):** 不納入大規模評估。它 43.5% 的 step time 是未建模殘差(kernel launch、RCCL handshake、CPU 排程、框架開銷),兩個 α 因子相差 1.64 倍,這個區間 ASTRA-sim 無法預測絕對時間。

> 掃過封包 payload(1,000–8,000 B)、逐鏈路延遲(12.5–14 µs)與 QCN 開關後,ResNet-50 的 ns-3 通訊時間都落在 14.0–15.1 ms,每一組設定都比實測值*低* 5–12%。殘差對所有可調參數都不敏感;而同一套 ns-3 傳輸模型套用在每條鏈路、每種拓撲上,因此與拓撲無關的乘性偏差會在比值中互相抵銷。

α 值與每次執行的校準結果寫入 `runs/calibration_aligned.csv`。`runs/calibration_all.csv` 是**視窗對齊機制之前的舊 append log**,其誤差值比較的是涵蓋工作量不同的兩個視窗,不要從中引用數字。完整校準方法論請見 [scripts/README.md](scripts/README.md)。

### 額外工作負載驗證:Qwen 0.5B 與 Qwen 1.5B TP

| 工作負載 | `--et-iters` | ns-3 通訊 | 實測 RCCL | ns-3 vs 實測 | 旗標 |
|---|---|---|---|---|---|
| `qwen05b` | 2 | 573.08 ms | 3,393.27 ms | **−83.1%** | — |
| `qwen15b_tp` | 2 | 392.25 ms | 5,918.81 ms | **−93.4%** | `per_iter_granularity_mismatch` |

兩個 LLM trace 都是排程受限型:ns-3 有建模資料傳輸,但沒有建模 DDP bucket 排程與 backward 計算之間的同步等待,而實測 RCCL kernel 時間大半就落在那裡。Qwen 0.5B 的 ET 每個 step 含 10,916 個 COMP 節點與 37 個 AllReduce COMM 節點,合計 1,884.6 MiB——正好是 494,032,768 個參數的 FP32 梯度量。`qwen15b_tp` 那一列被標記,是因為 ET 每迭代重播的 collective 數多於 trace 紀錄的數量,其誤差值需先人工確認語意才可引用。因此這兩個工作負載一律只用於**相對**拓撲比較(論文 5.2 / 5.3 節),以原始模擬 cycle 數呈現。

### 支援量測工具

以下三支腳本拆解 AllReduce 成本的*實測*側,使 ns-3 的落差可被歸因,而非僅能推測。輸出都在 `runs/calibration/`。

| 腳本 | 量測內容 |
|---|---|
| `scripts/bucket_micro_allreduce.py` | 以 ET 解出的 DDP bucket 尺寸執行 `torch.distributed.all_reduce`,GPU 上沒有其他工作——無競爭下限 |
| `scripts/q4_overlap_off.py` | 同一批 bucket,取自真實訓練步,每個集合操作都從 DDP 的 comm hook 內單獨發到閒置 GPU 上——不與 backward 重疊。只計時 `all_reduce` 本身,不含 DDP 自己的分桶與複製 |
| `scripts/fit_envelope.py` | 對 rccl-tests 掃描與 ns-3 逐 collective 的 COMM interval 各自擬合 `T(M) = α + M/B`,再逐 bucket 相減 |

第三個欄位由 Kineto trace 提供:同一批 collective 與 backward 競爭時的時間。`scripts/gen_envelope_figures.py` 會畫出兩側的 `T(M)` 與 `BW(M)`。

---

## 128 節點結果

以下為論文第 5 章的參考數值。每一筆都是單次模擬,採用前請自行重跑。

**實驗 1 — ResNet-50 DDP(每 step ~89.7 MiB),通訊被計算隱藏。** 三種拓撲都是同樣的 274,982,000 cycles(以 α_step 換算為 662.9 ms),暴露通訊為**零**。原始通訊 cycle 確實有差(Fat-Tree ~18.9M、Torus 27.5M、Twisted Torus 27.6M),但完全落在計算視窗內。此區間拓撲不可見。

**實驗 2 — Qwen 0.5B DDP(每 step 1.84 GiB),通訊密集。** {Torus, Twisted Torus} × {ring, halving-doubling} 的完整 2×2,全部使用 `active-chunks=4` 與 `--comm-scale 1.984375`,兩次迭代視窗:

| 拓撲 | 演算法 | Wall(M cycles) | vs Torus+ring | PFC 事件 | 完成 flow |
|---|---|---|---|---|---|
| **標準 Torus** | **ring × 3** | **5,057** | **基準** | 0 | 985,088 |
| Fat-Tree | halvingDoubling | 5,086 | +0.6% | 203,756 | 530,432 |
| 標準 Torus | X/Y 用 HD | 7,789 | +54.0% | 2,850 | 833,536 |
| Twisted Torus | X/Y 用 HD | 8,835 | +74.7% | 1,542 | 833,536 |
| Twisted Torus | ring × 3 | 8,998 | +77.9% | 24,348 | 985,088 |

這張表有兩種讀法,兩種都重要。對照部署基準(Torus + ring),**兩種** Twisted Torus 組態都慢約 75%。改為固定*演算法*來看,單獨衡量的扭轉損失在 ring 下是 +77.9%、在 HD 下是 +13.4%——HD 減輕了一部分,卻沒讓 Twisted Torus 具備競爭力,因為 HD 本身就不適合對稱的標準 Torus(+54.0%)。HD 讓 Twisted Torus 的 PFC 事件下降 94%,wall time 卻只買回 1.8%,這正是論文把剩餘損失歸因於扭轉造成的路由層不對稱、而非 PFC 可見壅塞的理由。另外注意 PFC 計數是壅塞指標而非 wall time 的預測值:Fat-Tree 的 pause 最多,卻是第二快的組態。

**實驗 3 — Qwen 1.5B,TP=8 × DDP=16。**

| 比較框架 | 拓撲 | 伺服器間頻寬 | 演算法 | Wall(M cycles) | vs Torus |
|---|---|---|---|---|---|
| 成本對齊 | **Torus** | 25 Gbps | ring × 3 | **7,613** | 基準 |
| 成本對齊 | Fat-Tree | 65 Gbps | HD, HD, ring | 7,883 | +3.5% |
| 成本對齊 | Twisted Torus | 25 Gbps | ring × 3 | 11,129 | +46.2% |
| 成本對齊 | Twisted Torus | 25 Gbps | HD, HD, ring | 11,446 | +50.3% |
| 頻寬對齊 | **Torus** | 65 Gbps | ring × 3 | **6,046** | 基準 |
| 頻寬對齊 | Fat-Tree | 65 Gbps | HD, HD, ring | 7,883 | +30.4% |
| 頻寬對齊 | Twisted Torus | 65 Gbps | ring × 3 | 8,598 | +42.2% |

針對 Twisted Torus 側的補充因子實驗顯示:X/Y 維度的 AllGather/ReduceScatter 設定在兩種 chunk 並行度下都**毫無差別**——TP 集合操作只走 Z 軸,唯一跨越 X/Y 的是 DDP AllReduce,而它由 `all-reduce-implementation` 控制。把 `active-chunks-per-dimension` 由 1 提高到 4 可讓 TT+HD 改善 22.7%(11,446M → 8,845M),但即使如此,TT+HD 仍比 chunks=1 的 Torus+ring 基準慢 16.2%。

**實驗 4 — All-to-All 頻寬飽和。** 隨每次集合操作 payload 增加的 wall time(cycles):

| Payload | Fat-Tree | Torus | Twisted Torus | 區間 |
|---|---|---|---|---|
| ~89.7 MiB(原始 AllReduce) | 274,982,000 | 274,982,000 | 274,982,000 | 全部隱藏 |
| 100 MB | 274,982,000 | 294,248,982 | 274,982,000 | Torus 最先暴露(+7%) |
| 512 MB | 879,553,198 | 1,523,300,236 | 1,288,871,156 | 全部暴露 |
| 1 GB | 1,886,210,767 | 3,051,005,526 | 2,576,544,108 | 全部暴露 |

1 GB 經 α 換算後:Fat-Tree 4,548 ms、Twisted Torus 6,212 ms、Torus 7,356 ms——扭轉比標準 Torus **快 1.18 倍**,Fat-Tree 比標準 Torus 快 1.62 倍。1.18 倍這個比值從 512 MB 到 1 GB 都很穩定。Google TPU v4 在相同 4×4×8 扭轉下報告 1.63 倍:方向一致,幅度縮小,因為扭轉增加的路徑多樣性正好落在本平台的低頻寬 X/Y 維度上(25 vs. 65 Gbps)。

### 成本

| 項目 | Fat-Tree | Torus / Twisted Torus |
|---|---|---|
| 128 × RX 9070 XT | NT$2,944,000 | NT$2,944,000 |
| 16 × EPYC 伺服器平台 | NT$3,600,000 | NT$3,600,000 |
| 48 × ConnectX-6 雙埠 100 GbE 網卡 | NT$1,296,000 | NT$1,296,000 |
| 24 × 100 GbE 管理型交換器(SN2700) | **NT$5,756,400** | **NT$0** |
| **總計** | **NT$13,596,400** | **NT$7,840,000** |
| vs. 新臺幣 1,000 萬預算 | 超出 36% | 低於預算 22% |

2025 年 11 月市價,US$1 = NT$30。兩種拓撲家族使用完全相同的網卡,因此全部差額就是管理型交換器——占 Fat-Tree 總成本的 42%,約為其 GPU 支出的兩倍。對以 DDP 與 TP+DDP 為主的工作負載,標準 Torus 以約 58% 的成本達到與 Fat-Tree 相當的效能(差距 0.6% 以內),這就是論文建議的預設方案。扭轉不增加任何硬體成本(只差佈線),因此可以日後再視工作負載是否轉為 All-to-All 主導而採用。

---

## 拓撲設定

128 節點評估提供三種預先設定的拓撲,並支援**成本對齊**(Torus/TT 25 Gbps inter-server vs. Fat-Tree 65 Gbps)與**頻寬對齊**(全部 65 Gbps)兩種比較框架:

| 參數 | Fat-Tree(L16_S8) | Torus(4×4×8) | Twisted Torus(4×4×8) |
|---|---|---|---|
| 實體交換器數量 | **24** | **0** | **0** |
| 節點間頻寬(Z 軸 / 節點內) | 65 Gbps | 65 Gbps | 65 Gbps |
| 節點間頻寬(X/Y 軸 / 節點間,成本對齊) | 65 Gbps | **25 Gbps** | **25 Gbps** |
| 節點間頻寬(X/Y 軸,頻寬對齊變體) | — | 65 Gbps | 65 Gbps |
| 每條鏈路延遲 | GPU→Leaf: 14 µs;Leaf→Spine: 5 µs | Z: 14 µs;X,Y: 5 µs | Z: 14 µs;X,Y: 5 µs |
| 預設集合通訊演算法 | halvingDoubling | ring × 3 | ring × 3 |
| Twist(X 軸繞回) | — | 無 | Y 偏移 +1 |

**Twisted Torus 繞線定義**(X 軸 wrap-around):
```
(x=3, y, z) → (x=0, (y+1) mod 4, z)
```

頻寬對齊的物理拓撲檔位於 `configs/astra-sim/topos/`,檔名分別為
`128nodes_Torus_4x4x8_65G.txt` 與 `128nodes_TwistedTorus_4x4x8_65G.txt`。

### 系統設定組合

`configs/astra-sim/system/` 下列出論文評估過的演算法 × chunk concurrency 組合:

| 檔名 | active-chunks | All-Reduce 演算法 | 對應實驗 |
|---|---|---|---|
| `system_128nodes_Torus_4x4x8.json` | 1 | ring × 3 | 實驗 1(ResNet-50) |
| `system_128nodes_Torus_4x4x8_4chunks.json` | 4 | ring × 3 | 實驗 2(Qwen 0.5B,Torus 基準) |
| `system_128nodes_TwistedTorus_4x4x8.json` | 1 | ring × 3 | 實驗 1 |
| `system_128nodes_TwistedTorus_4x4x8_4chunks.json` | 4 | ring × 3 | 實驗 2(TT + ring 那一臂) |
| `system_128nodes_TwistedTorus_4x4x8_4chunks_hd.json` | 4 | halvingDoubling、halvingDoubling、ring | 實驗 2(TT + HD 那一臂) |
| `system_128nodes_FatTree_L16_S8.json` | 1 | halvingDoubling | 實驗 1 |
| `system_128nodes_FatTree_L16_S8_4chunks.json` | 4 | halvingDoubling | 實驗 2 |
| `system_128nodes_*_TP8DDP*.json` | 1 / 4 | ring × 3 / X/Y 改用 HD | 實驗 3(TP+DDP) |

### 自行產生拓撲與設定檔

若需要自訂或重新產生 topology 與對應設定檔,可使用 `src/topology_generator.py`,可一鍵輸出 `*.txt`、`logical_*.json`、`system_*.json` 三種檔案:

```bash
# 128 節點 4×4×8 Twisted Torus(成本對齊:節點間 25 Gbps)
python3 src/topology_generator.py \
  --type twisted_torus \
  --nodes 128 --dims 4 4 8 \
  --bw-intra 65Gbps --lat-intra 0.014ms \
  --bw-inter 25Gbps --lat-inter 0.005ms

# 128 節點 Fat-Tree
python3 src/topology_generator.py \
  --type fattree \
  --nodes 128 \
  --bw-intra 65Gbps --lat-intra 0.014ms \
  --bw-inter 65Gbps --lat-inter 0.005ms
```

若只想快速重現論文使用的拓撲,直接使用 `configs/astra-sim/topos/` 與 `configs/astra-sim/system/` 內既有檔案即可。

---

## 實驗 2:Twisted Torus AllReduce(Qwen 0.5B)

實驗 2 在通訊密集 AllReduce 下,用 {Torus, Twisted Torus} × {Ring, Halving-Doubling} 的 2×2 組合,來分開「網路壅塞」和「拓撲路徑結構」兩個因素(四組都用 `active-chunks=4`、`--comm-scale 1.984375`)。參考結果見[128 節點結果](#128-節點結果)。四組設定如下:

| 拓撲(實體) | system 設定 | 演算法 |
|---|---|---|
| `128nodes_Torus_4x4x8.txt` | `system_128nodes_Torus_4x4x8_4chunks.json` | ring × 3 |
| `128nodes_Torus_4x4x8.txt` | `system_128nodes_TwistedTorus_4x4x8_4chunks_hd.json` | X/Y 用 HD、Z 用 ring |
| `128nodes_TwistedTorus_4x4x8.txt` | `system_128nodes_TwistedTorus_4x4x8_4chunks.json` | ring × 3 |
| `128nodes_TwistedTorus_4x4x8.txt` | `system_128nodes_TwistedTorus_4x4x8_4chunks_hd.json` | X/Y 用 HD、Z 用 ring |

此實驗中 Fat-Tree 那一臂使用它自然的扁平對映——跨全部 128 個端點的單階段 halvingDoubling(邏輯維度 `[128]`)——而 Torus 系列則在原生的 `[4, 4, 8]` 結構上逐維度做 ring,亦即每種拓撲都跑各自結構所導出的標準集合對映。

四組都跑完(指令見 [scripts/commands.md](scripts/commands.md)),從各自的 `out/metrics.csv` 比 `Wall time` 和 `PFC 事件`。要看的是這四個點彼此的關係,不是絕對數字;絕對值會隨你的校準和機器變動。兩種比較要分開讀:*對照 Torus+ring 部署基準*(兩種扭轉組態都慢約 75%)與*固定演算法*(扭轉損失在 ring 下 +77.9%、HD 下 +13.4%)。

---

## 多維度 Ring 排程死鎖(active-chunks-per-dimension)

> 已回報為 [ASTRA-sim Issue #370](https://github.com/astra-sim/astra-sim/issues/370)。

在 128 節點 Twisted Torus 跑高通訊量 AllReduce(Qwen 0.5B)時,預設的 `active-chunks-per-dimension=1` 在 `localBWAware` 優化下會觸發確定性的排程死鎖。Twisted Torus 的 X 軸非對稱繞回鏈路使各節點階段進度不同步,在 ASTRA-sim chunk queue 中產生跨維度的循環等待。徵狀:ns-3 在 ~5,337 個 flow 處停止發出新 flow(預期約 985,088)。

**解法:** 將 `active-chunks-per-dimension: 4` 設為與 `preferred-dataset-splits: 4` 相同。論文 Qwen 0.5B 實驗都使用 `*_4chunks*.json` 系列。`*_4chunks_hd.json` 是 2×2 因子分析的第二臂,把 X/Y 維度的 ring 換成 halvingDoubling,同樣可避開 deadlock。`active-chunks=4` 化解的是排程層級的 deadlock,並不改變拓撲的路徑結構(其含義見論文)。

標準 3D Torus 與 Fat-Tree 的對稱路徑可保證各節點階段同步,因此不會觸發此死鎖。其他論文實驗(ResNet-50 AllReduce、TP+DDP、All-to-All)亦不會觸發,皆使用預設 `active-chunks=1` 的設定檔。

**機制。** 在 `localBWAware` 下,3D AllReduce 被拆成五個階段(X 的 RS、Y 的 RS、Z 的 AllReduce、Y 的 AG、X 的 AG),而**同一維度**的 RS 與 AG 共用一個 queue——階段 0 與 4 共用 queue 0,階段 1 與 3 共用 queue 1。當 `active-chunks-per-dimension=1` 時一個 queue 同時只容得下一個 chunk,因此只要扭轉使各節點階段進度不同步,快節點的下一個 bucket 的 Reduce-Scatter 就會與慢節點當前 bucket 的 All-Gather 爭用同一個 queue 而死鎖。死鎖發生在哪一層是靠 instrumentation 確認的:在 ns-3 前端 `entry.h` 插入 `send_flow` 與 `qp_finish` 計數器後,兩者都停在 5,337,證明 ns-3 完成了所有收到的 flow,是 ASTRA-sim 停止發出新 flow。這個風險在 ASTRA-sim 自己的 `Sys.cc`(第 837–852 行)中已被承認,Issue #137 亦有部分討論,但預設組態仍未防範。該 instrumentation 以 `rocm/patches/entry_flow_diagnostics.py` 形式收錄。

**重現方式。** [`deadlock-reproduction/`](deadlock-reproduction/) 是一個自足的重現包——Twisted Torus 拓撲檔、會死鎖與已修正的 system 設定、Qwen 0.5B 的 ET,以及 `chunks=1`(FIFO 與 LIFO)和 `chunks=2` 的 stdout 證據。

---

## All-to-All 壓力測試(實驗 4)

用 ResNet-50 原始 trace(每 step 約 89.7 MiB)跑 AllReduce 時,通訊被 GPU 計算蓋掉,三種拓撲看起來一樣。要把流量壓到網路上,可以用 `src/scale_et_comm_workload.py` 就地改寫每個 `COMM_COLL_NODE`:

1. **`comm_type`** → 強制設為 `ALL_TO_ALL`(原為 `ALL_REDUCE`)
2. **`comm_size`** → 設為指定的位元組數(例如 1 GB = 1,073,741,824 bytes)

原始計算節點與 DAG 結構保持不變,模擬仍能保留真實的計算與通訊交錯模式。

### 檔案命名規則

```
輸入:et.<prefix>.<rank>.et         (例如 et.resnet50_all2all.0.et)
輸出:et.<prefix><suffix>.<rank>.et  (例如 et.resnet50_all2all_1GB.0.et)
```

若未指定 `--suffix`,後綴依 `--bytes` 自動產生:

| `--bytes` | 自動後綴 |
|---|---|
| `1G` / `1073741824` | `_1GB` |
| `512MB` / `512M`    | `_512MB` |
| `100MB` / `100M`    | `_100MB` |

### 使用方式

```bash
# 步驟 1 — 將原始 ResNet-50 trace 以另一個 tag 複製出來(若尚未執行)
python src/conver_to_chakra_et.py --model-tag resnet50_all2all

# 步驟 2 — 放大為 1 GB All-to-All(產生 et.resnet50_all2all_1GB.*.et)
python src/scale_et_comm_workload.py \
  --workload-dir data/chakra/workload_et \
  --prefix resnet50_all2all \
  --bytes 1G

# 步驟 3 — 使用放大後的工作負載執行模擬(--payload 12000 控制 ns-3 事件數量)
python scripts/run_ns3.py \
  --workload data/chakra/workload_et \
  --model-tag resnet50_all2all_1GB \
  --topo file:configs/astra-sim/topos/logical_128nodes_TwistedTorus_4x4x8.json \
  --phys-topo configs/astra-sim/topos/128nodes_TwistedTorus_4x4x8.txt \
  --system configs/astra-sim/system/system_128nodes_TwistedTorus_4x4x8.json \
  --virtual-world 128 --payload 12000 --lmbw 540 --no-autocalib
```

> **可觀察性三階段(論文 5.4.1):** 原始 ~89.7 MiB AllReduce 完全被計算遮蔽;小的 All-to-All payload 會讓拓撲差異逐步暴露,到 512 MB–1 GB 時三種拓撲完全分化。建議自己掃 `--bytes`(100 MB → 1 GB),看在你的環境下三種拓撲從哪裡開始分離;論文提供其參考值與比例。1 GB All-to-All 是 simulation-only 的上界壓力測試,而非實際生產工作負載——1 GB / collective 的通訊量在 128 節點規模下也已超過 16 GB VRAM,在實體硬體上無法直接執行。

---

## 環境設定

### Docker(建議)

```bash
docker-compose up
# 指定特定 ROCm / PyTorch 版本:
VERSION=rocm6.4.4_ubuntu24.04_py3.12_pytorch_release_2.7.1 docker-compose up
# 或跳過除錯用 instrumentation(只套必要的 build fix):
ASTRA_PATCHES=none docker-compose up --build
```

兩個旋鈕都放在 `.env`。映像檔把 Chakra 釘在 mlcommons 上游 `ec41090`(而非 astra-sim 的 fork,該 fork 沒有任何獨有 commit),並強制升級 `protobuf`——base image 內建 3.20.2,而上游已移除 `protobuf==5.*` 的版本上限;build 階段的 import 檢查會攔下由此產生的 gencode/runtime 不相容,而不是等它在執行中途才浮現。

`ASTRA_PATCHES` 決定套用 `rocm/patches/` 底下哪些 ASTRA-sim 原始碼修補:

| 修補 | 套用時機 | 用途 |
|---|---|---|
| `spdlog_fmt_compat.py` | 一律套用 | 補上 `spdlog_setup` 所需的 `fmt` include,ns-3 前端才能編譯 |
| `statistics_comm_intervals.py` | `all` | 每個 COMM interval 印一行;ASTRA-sim 原本只回報合併後的 COMM 總量,逐 collective 成本無法從一次執行中還原 |
| `entry_flow_diagnostics.py` | `all` | ns-3 前端的 `send_flow` / `qp_finish` 計數器——定位 Issue #370 死鎖所用的 instrumentation |

兩個 `all` 專屬的修補都是唯讀 instrumentation:只記錄本來就已在計算的量,不改動 `type_time` 或任何模擬結果。

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

互動式 3D Twisted Torus 拓撲視覺化工具:

```
viz/twisted_torus_3d.html
```

以任意瀏覽器開啟,可瀏覽 4×4×8 Twisted Torus 的完整繞線圖。

---

## 常見問題

**Q:ns-3 模擬看起來很久沒有結束,是不是卡死了?**
A:不一定。對真實拓樸與較大的 ET 檔案而言,模擬時間可能非常長;以本研究的 128-node 實驗為例,**單一實驗平均約需 4–5 天** 才會產出完整結果。

建議先檢查輸出目錄中的 `fct.txt` 是否仍持續產生內容,例如:

```text
runs/20260324-013221+0800_ns3_128gpu_qwen05b_file_logical_128nodes_FatTree_L16_S8/out/fct.txt
```

若 `fct.txt` 持續有新數值寫入,通常代表模擬仍在正常進行。可使用 `--deadlock-timeout`(預設 12 小時)自動 kill 真正卡死的實驗。生成追蹤時建議將 `--trace-steps` 控制在 1–4——非常大的 ET 檔可能耗盡 ASTRA-sim ETFeeder 的資源。多個實驗也可以使用不同 shell 視窗平行執行。

**Q:Twisted Torus 跑 Qwen 0.5B 時,fct.txt 在約 5,337 個 flow 後停下不動。**
A:這是上述的「多維度 Ring 排程死鎖」(亦即論文 6.2.7 節)。請改用 `*_4chunks*.json` 系列設定,使 `active-chunks-per-dimension=4`。`*_4chunks_hd.json`(X/Y 改用 halvingDoubling)是 2×2 因子分析的第二臂,同樣可避開 deadlock。`active-chunks=4` 化解的是排程層級的 deadlock,並不改變拓撲的路徑結構(其含義見論文第 5 章 / 6.2.7 節)。

**Q:ASTRA-sim 出現 `"Node X in ctrl_dep graph, but not found in index"` 錯誤?**
A:ET 檔案的 DAG 完整性異常(自依賴或循環依賴)。重新執行 `conver_to_chakra_et.py`,內建的 DAG 修復 Pass(`fix_et_dag_inplace`)應可自動解決。

**Q:ROCm 上 `chakra_trace_link` 因時間戳對不齊而失敗?**
A:在 trace 收集腳本加上 `--inject-sync-hack`。此選項會注入同步事件,對齊 CPU(毫秒)與 GPU(微秒)的時間軸。

**Q:ns-3 的通訊時間對照實體硬體有多準?**
A:以 ResNet-50 而言,把實測與模擬的視窗對齊之後,ns-3 的總和比實測 RCCL kernel 總和**低 5.1%**——這是聚合一致,內含互相抵銷的逐 collective 偏差。任何可調參數均無法改變此結果(payload、延遲、QCN 都讓它落在 14.0–15.1 ms)。排程受限型的工作負載低估幅度顯著更大(CIFAR-10 −86.6%、Qwen 0.5B −83.1%),因為模擬器有建模資料傳輸,卻沒有建模集合排程與 backward 計算之間的同步等待。那個未建模成分屬於共用的 trace 與排程,三種拓撲完全相同,因此不會進入相對比較。

> 本 README 的舊版本曾寫「ns-3 把 ResNet-50 *高估*約 2 倍」。該數字來自比較涵蓋工作量不同的兩個視窗,已作廢——見前面的「先對齊視窗,其他數字才有意義」。

**Q:為什麼 CIFAR-10 被排除在大規模評估之外?**
A:它 43.5% 的 step time 落在 ASTRA-sim 未建模的殘差(kernel launch、RCCL handshake、CPU scheduling、框架開銷),wall-clock 與 communication calibration factor 發散 1.64 倍,因此不適合用來做這種延遲主導區間的絕對時間預測。詳見論文第 4.3 節。

**Q:`--comm-scale 1.984375` 到底在做什麼?**
A:它設定的是複製後 128 節點執行的**通訊工作點**,而不是對集合演算法的修正——後者 ASTRA-sim 會依設定的參與節點數自行推導。這個值是精確分數 `127/64`。Qwen 0.5B 必須用精確分數,縮放後的 `comm_size` 才能被 `preferred-dataset-splits=4` 整除;TP+DDP 則可接受四捨五入的 `1.984`。同一實驗內同一倍率、同一 trace 套用到三種拓撲,因此每次比較都在相同的 offered load 下進行。詳見論文 4.2.6 / 4.6.2 節。

**Q:我的校準結果 `alpha_us` 是空的。**
A:你沒有給 `--et-iters`。`alpha_us` 需要「每步」的分母,腳本留空而不推測迭代數——猜出來的 α 會產生一個看起來合理、實際上默默算錯的換算係數。請填入產生該 ET 時使用的 `--trace-steps`。

**Q:我的校準列帶了 `per_iter_granularity_mismatch` 旗標。**
A:ET 每迭代重播的 collective 數與 trace 紀錄的不同,兩側加總的不是同一批工作。執行不會被中止,但誤差值需先人工確認語意才可引用。`qwen15b_tp` 那一列就處於這個狀態。

---

## 歷史開發報告

Pipeline 開發過程中遭遇並解決的問題,已記錄於 [`docs/archive/`](docs/archive/)。這些報告與正常使用無關,但有助於理解 AMD 相容性開發過程。

| 檔案 | 內容 |
|---|---|
| [ASTRA-sim_Analysis_Report.md](docs/archive/ASTRA-sim_Analysis_Report.md) | Alpha 校準分析、計算 cycle 解析問題、ns-3 卡死調查 |
| [AMD_GPU_ASTRA_SIM_Integration_Complete_Report.md](docs/archive/AMD_GPU_ASTRA_SIM_Integration_Complete_Report.md) | HIP Runtime 不相容、RCCL Kernel 命名差異、DAG 修復——初期技術突破報告 |

---

## 引用

若使用本 Pipeline 或相關模擬結果,請引用:

```bibtex
@mastersthesis{chen2026torus,
  author  = {jjasoncool},
  title   = {Cost-Effective AI Training Performance Evaluation for Torus Topology
             based on AMD ROCm and the Trace-Driven simulator ASTRA-sim},
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
- [TPU v4 論文](https://dl.acm.org/doi/10.1145/3579371.3589350) — Google Twisted Torus 參考文獻(ISCA'23)
