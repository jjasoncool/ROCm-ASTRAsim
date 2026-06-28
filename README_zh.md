# ROCm-ASTRAsim：AMD GPU AI 叢集追蹤驅動模擬框架

> **繁體中文** | [English](README.md)

> **論文：** *《Cost-Effective Twisted Torus for AI Training: An ASTRA-sim Evaluation Using Traces from Consumer-Grade AMD GPUs with ROCm》*
> 國立成功大學資訊工程研究所,2026

一套三階段的 trace-driven 模擬 pipeline,從 AMD ROCm/RCCL 實體硬體收集訓練追蹤,再送進 ASTRA-sim 做叢集規模的網路模擬。多數已發表的 ASTRA-sim 研究假設的是 NVIDIA CUDA/NCCL,這裡補上 AMD ROCm/RCCL 這條路徑。

論文在 128 節點規模下評估 Fat-Tree、標準 3D Torus、Twisted Torus 三種拓撲,涵蓋**四種通訊強度**:

1. **計算主導 AllReduce** — ResNet-50 DDP(每 step ~89.7 MiB)
2. **通訊密集 AllReduce** — Qwen2.5-0.5B DDP(每 step ~1.84 GiB)
3. **階層式 TP+DDP** — Qwen2.5-1.5B(TP=8 × DDP=16)
4. **All-to-All 頻寬飽和** — 合成壓力測試(每次集合 1 GB)

論文的結論是:twist 的價值取決於 workload——它對頻寬受限的 All-to-All 有幫助,對通訊密集的 DDP AllReduce 則是拖累。這裡的設定檔可以讓你自己重現這個比較,完整結果與分析在論文第 5 章。README 裡引用的數字都來自單次模擬,只能當參考,不是保證;要採用前請先在自己的環境重跑。

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
├── docs/                          # ASTRA-sim 設定文件與歷史報告
├── runs/                          # 模擬結果 + calibration_all.csv
├── tutorials/                     # 學術教學範例(MICRO'24、ASPLOS'23)
├── viz/                           # 互動式 3D Twisted Torus 拓撲視覺化
├── rocm/dockerfile                # Docker 環境(ROCm + PyTorch + ASTRA-sim)
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
# 2-GPU 校準執行(任一 workload 均可)。執行後可在 runs/calibration_all.csv 看到對應的 alpha_us。
python ./scripts/run_ns3.py \
  --workload data/chakra/workload_et --model-tag resnet50 \
  --topo auto:1d \
  --phys-topo configs/astra-sim/topos/2_nodes_1_switch_topology.txt \
  --coll-opt localBWAware --lmbw 540
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
# 注意:Qwen 0.5B 須用精確分數 127/64 = 1.984375(非四捨五入的 1.984),
# 以確保縮放後的 comm_size 能被 preferred-dataset-splits=4 整除(論文 §4.2.6 / §5.2.1)。
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
| `--virtual-world N` | 將每個 rank 的 trace 複製擴展到 `N` 節點模擬 |
| `--comm-scale F`    | 將 `comm_size` 乘以 `F`,對應 M=2 → N=128 修正係數。Qwen 0.5B(實驗 2)須用精確分數 `1.984375`(127/64)以確保 split 整除;TP+DDP(實驗 3)及其他實驗用四捨五入的 `1.984` |
| `--no-qlen`         | 將 `qlen.txt` 導向 `/dev/null`,避免 128 節點時產生數百 GB 除錯輸出 |
| `--payload`         | 覆寫 ns-3 封包 payload(All-to-All 1 GB 壓力測試使用 `12000` 控制事件量) |
| `--no-autocalib`    | 停用自動 α 計算(僅在 2-GPU 校準時可用;128 節點必加此參數) |
| `--deadlock-timeout S` | 若 `fct.txt` 連續 `S` 秒未更新則自動 kill(預設 12 小時;對 ring 死鎖情境特別有用) |

---

## 校準

校準在 2-GPU 規模下執行,產生校準因子 α(µs/cycle),把模擬 cycle 數換算成真實時間。論文用它來支撐拓撲之間的相對比較,而不是預測絕對通訊時間。

- **ResNet-50(頻寬限制型):** 主要基準。wall-clock 校準得到 α_step = 0.002411 µs/cycle。ns-3 的通訊時間穩定,但大約是實測 RCCL GPU kernel duration 的 2 倍。原因應該是傳輸路徑不一致:ns-3 模擬的是 Ethernet RDMA,而 2-GPU 平台走的是 PCIe DMA。
- **CIFAR-10(延遲限制型):** 不納入大規模評估。它的 step time 大半花在 ASTRA-sim 沒建模的軟體堆疊開銷上,所以這裡無法預測絕對時間。

**實測校準結果(2-GPU,每個訓練步驟):**

| 指標 | ResNet-50 | CIFAR-10 |
|---|---|---|
| `ns3_comm_ms`(ns-3 模擬值) | **15.06 ms** | **41.07 ms** |
| `real_t_comm_ms`(硬體實測值) | **7.54 ms** | **61.98 ms** |
| ns-3 vs real | **+100%**(高估) | **−34%**(低估) |
| 校準狀態 | 主基準 | 排除(scope boundary) |

> 針對 ResNet-50,掃過頻寬、延遲、封包 payload 與擁塞控制後,ns-3 通訊時間都穩定落在 14.0–15.1 ms。調參數動不了它,代表這是結構性偏差,不是參數沒調好。這個偏差對每種拓撲一視同仁,所以相對排序仍然成立。

α 值與每次執行的校準結果記錄於 `runs/calibration_all.csv`。完整校準方法論請見 [scripts/README.md](scripts/README.md)。

### 額外工作負載驗證:Qwen2.5-0.5B

這套 pipeline 也跑過相同 2-GPU AMD 平台上的 Qwen2.5-0.5B DDP trace。這個 494M 參數的模型每個 step 產生 37 個 AllReduce 節點、約 1.84 GiB 通訊量。2-GPU 校準結果看起來合理(comm/step ≈ 57.7%;這裡 ns-3 低估實測通訊 52%,跟「37 個 AllReduce 累積的啟動開銷 ns-3 沒建模」一致)。Qwen 0.5B 就是實驗 2 的工作負載。

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

實驗 2 在通訊密集 AllReduce 下,用 {Torus, Twisted Torus} × {Ring, Halving-Doubling} 的 2×2 組合,來分開「網路壅塞」和「拓撲路徑結構」兩個因素(四組都用 `active-chunks=4`、`comm-scale=1.984375`)。四組設定如下:

| 拓撲(實體) | system 設定 | 演算法 |
|---|---|---|
| `128nodes_Torus_4x4x8.txt` | `system_128nodes_Torus_4x4x8_4chunks.json` | ring × 3 |
| `128nodes_Torus_4x4x8.txt` | `system_128nodes_TwistedTorus_4x4x8_4chunks_hd.json` | X/Y 用 HD、Z 用 ring |
| `128nodes_TwistedTorus_4x4x8.txt` | `system_128nodes_TwistedTorus_4x4x8_4chunks.json` | ring × 3 |
| `128nodes_TwistedTorus_4x4x8.txt` | `system_128nodes_TwistedTorus_4x4x8_4chunks_hd.json` | X/Y 用 HD、Z 用 ring |

四組都跑完(指令見 [scripts/commands.md](scripts/commands.md)),從各自的 `out/metrics.csv` 比 `Wall time` 和 `PFC 事件`。要看的是這四個點彼此的關係,不是絕對數字;絕對值會隨你的校準和機器變動。參考值和分析在論文第 5 章。

---

## 多維度 Ring 排程死鎖(active-chunks-per-dimension)

> 已回報為 [ASTRA-sim Issue #370](https://github.com/astra-sim/astra-sim/issues/370)。

在 128 節點 Twisted Torus 跑高通訊量 AllReduce(Qwen 0.5B)時,預設的 `active-chunks-per-dimension=1` 在 `localBWAware` 優化下會觸發確定性的排程死鎖。Twisted Torus 的 X 軸非對稱繞回鏈路使各節點階段進度不同步,在 ASTRA-sim chunk queue 中產生跨維度的循環等待。徵狀:ns-3 在 ~5,337 個 flow 處停止發出新 flow(預期約 985,088)。

**解法:** 將 `active-chunks-per-dimension: 4` 設為與 `preferred-dataset-splits: 4` 相同。論文 Qwen 0.5B 實驗都使用 `*_4chunks*.json` 系列。`*_4chunks_hd.json` 是 2×2 因子分析的第二臂,把 X/Y 維度的 ring 換成 halvingDoubling,同樣可避開 deadlock。`active-chunks=4` 化解的是排程層級的 deadlock,並不改變拓撲的路徑結構(其含義見論文)。

標準 3D Torus 與 Fat-Tree 的對稱路徑可保證各節點階段同步,因此不會觸發此死鎖。其他論文實驗(ResNet-50 AllReduce、TP+DDP、All-to-All)亦不會觸發,皆使用預設 `active-chunks=1` 的設定檔。

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

**Q:為什麼 ns-3 會把 ResNet-50 的通訊時間高估約 2 倍?**
A:調任何參數都動不了它,所以不是校準調錯。原因應該是傳輸路徑不一致:ns-3 模擬 Ethernet RDMA,而 2-GPU 平台走 PCIe DMA。同一個模型套在每種拓撲上,所以拓撲互相比較時這個偏差會抵消掉。

**Q:為什麼 CIFAR-10 被排除在大規模評估之外?**
A:它超過一半的 step time 都落在 ASTRA-sim 未建模的軟體堆疊開銷(kernel launch、RCCL handshake、CPU scheduling),導致 wall-clock 與 communication calibration factor 發散 1.3 倍,因此不適合用來做大規模絕對時間預測。詳見論文第 4.3 節。

**Q:為什麼 Qwen 實驗用 `--comm-scale ≈ 1.984`,而 Qwen 0.5B 要用 `1.984375`?**
A:這是為了把 M=2 的來源 trace 複製成 N=128 ranks 後,修正每個集合的通訊量。具體公式為 `M(N-1) / (N(M-1)) = 2 × 127 / (128 × 1) = 127/64 = 1.984375`,使擴展後的 trace 與校準時的 2-GPU 基準對齊。**Qwen 0.5B DDP**(實驗 2)須用精確分數 `1.984375`,因為縮放後的 `comm_size` 必須能被 `preferred-dataset-splits=4` 整除;四捨五入成 `1.984` 會破壞整除性,正是先前污染某組舊結果的兩個 bug 之一。TP+DDP(實驗 3)與 ResNet-50 實驗無此整除需求,故用四捨五入的 `1.984`。在同一實驗內對所有拓撲一致套用,此倍率不影響相對比較。詳見論文 4.2.6 / 4.6.2 節。

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
- [TPU v4 論文](https://dl.acm.org/doi/10.1145/3579371.3589350) — Google Twisted Torus 參考文獻(ISCA'23)
