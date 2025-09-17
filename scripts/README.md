# `scripts/` 使用說明（ASTRA-sim ns-3 執行工具）

本資料夾保存與 **ASTRA-sim × ns-3** 模擬執行流程相關的腳本（核心：`run_ns3.py`）。\
目標：以同一批 **Chakra 工作負載（.et 或 JSON）**，快速在不同 **邏輯拓樸** × **物理拓樸** × **系統層策略** × **網路層參數** 下重現並比較結果，且可用 `--virtual-world` 在執行時臨時擴張節點數。

> 名詞對齊
> • **邏輯拓樸 (logical)**：ASTRA-sim 的 `logical-dims`（K 維座標），決定 **collective 如何分相/繞行**。
> • **物理拓樸 (physical)**：ns-3 的 `TOPOLOGY_FILE`（交換器/連結拓樸），決定 **封包如何排隊/擁塞**。
> • **world size**：工作負載 rank/節點數（由 `.et` 檔數或 `et_rank_*.json` 檔數決定）。

---

## 目錄與基準檔案

- **工作負載（輸入）**
  - `data/chakra/workload_et/`：官方 **protobuf .et**（`*.et` 索引需連續 0..N-1）。
  - `data/chakra/workload_trace/`：Chakra **JSON**（`manifest.json + et_rank_*.json`）注意：`--virtual-world` 目前僅支援 `.et`。。

- **ASTRA-sim 基線設定（輸入；執行時會產生 patched 副本）**
  - `configs/astra-sim/system/system.json`
  - `configs/astra-sim/ns3/config.txt`
  - `configs/astra-sim/topos/`（`*.json`＝邏輯拓樸；`*.txt`＝物理拓樸）
  - `configs/astra-sim/remote_memory.json`

- **本次執行輸出（自動建立）**
  - `runs/<timestamp>_ns3_run/`
    - `stdout.log`：ns-3 執行輸出
    - `command.txt`：本次完整命令列（可重現）
    - `tmp/`：`system.patched.json`、`config.patched.txt`、`logical_topology.json`、（若使用 `--virtual-world`）`workload_<N>/*.et`
    - `out/`：`flow.txt`、`trace.txt`、`mix.tr`、`fct.txt`、`pfc.txt`、`qlen.txt`

> **不會修改 baseline**；所有覆寫都寫在 `runs/<timestamp>_ns3_run/tmp/`。

---

## 快速開始

**情境**：雙卡（world=2），1D 邏輯拓樸，指定 2 節點物理拓樸；網路層沿用 baseline。
以兩份實測 .et 臨時擴張到 8 節點，2D 邏輯（2×4），並指定 8-node 物理拓樸：

    python scripts/run_ns3.py \
      --workload data/chakra/workload_et \
      --virtual-world 8 \
      --topo auto:2d \
      --phys-topo configs/astra-sim/topos/8_nodes_1_switch_topology.txt \
      --coll-opt localBWAware

完成後，結果位於：

    runs/<timestamp>_ns3_run/
      ├─ stdout.log
      ├─ command.txt
      ├─ tmp/{system.patched.json,config.patched.txt,logical_topology.json}
      └─ out/{fct.txt,qlen.txt,pfc.txt,trace.txt,flow.txt,mix.tr}

（若只想做通路驗證而不擴張，可先用 world=2 與 2-node 物理拓樸跑 smoke test，但研究建議用上面 8 節點起跳。）
---

## 參數一覽（`run_ns3.py`）

- **必要/常用**
  - `--workload`：工作負載資料夾（`.et` 或 `manifest.json + et_rank_*.json`）。**決定 world size**。
  - `--virtual-world N`：以你的實測 `.et` 為模板，在執行前臨時擴張到 `N` 份 `.et`（只調整通訊 `comm_size`；compute/時間形狀保留）。僅支援`.et`。
  - `--topo`（邏輯拓樸；ASTRA-sim）：
    - `auto:1d` / `auto:2d` / `auto:3d`：自動產生 `logical-dims`（乘積＝world）。
    - `dims:2x4`：直接指定維度（會檢查乘積＝world）。
    - `file:.../*.json`：使用現成 logical JSON（會檢查乘積＝world）。
  - `--phys-topo`（物理拓樸；ns-3）：`configs/astra-sim/topos/*.txt`。未指定時，會嘗試在該資料夾依 world 搜尋 `"<world>_nodes_*_topology.txt"`；若找不到，沿用 baseline 並給警告。
  - `--system / --network / --remote`：baseline 設定檔（執行時會產生 **patched 副本**使用）。
  - `--ns3-bin`：ns-3 後端執行檔（可用 `ASTRA_NS3_BIN` 環境變數）。

- **系統層覆蓋（collective/排程；影響演算法流程）**
  - `--coll-opt`：覆蓋 `collective-optimization`（例：`localBWAware`、`none`）。
  - `--lmbw`：覆蓋 `local-mem-bw`（校準 compute/comm 暴露比例）。

- **網路層覆蓋（ns-3 佇列/流控；影響排隊/長尾）**
  - `--qcn`：覆蓋 `ENABLE_QCN`（`0`/`1`）。
  - `--pfc-dyn`：覆蓋 `USE_DYNAMIC_PFC_THRESHOLD`（`0`/`1`）。
  - `--buffer`：覆蓋 `BUFFER_SIZE`（例：`16`/`32`/`64`）。
  - `--payload`：覆蓋 `PACKET_PAYLOAD_SIZE`（例：`512`/`1000`/`1500`）。

- **其他**
  - `--comm-group`：傳給 ASTRA-sim 的 `--comm-group-configuration`（常用 `empty`）。
  - `--log-dir`：本次輸出根目錄（預設 `runs/`）。
  - `--dry-run`：只產生 patched 檔與指令，不執行。

---

## 路徑補丁原理：`patch_network_cfg`

baseline 的 `configs/astra-sim/ns3/config.txt` 常含**相對路徑**（搬家後易失效），例如：

    TOPOLOGY_FILE ../../scratch/topology/8_nodes_1_switch_topology.txt
    FCT_OUTPUT_FILE ../../scratch/output/fct.txt

執行時腳本會在 `runs/<timestamp>_ns3_run/tmp/` 產生 `config.patched.txt` 並**改為絕對路徑**：

- `TOPOLOGY_FILE` → 你指定的 `--phys-topo`（或依 world 推測到的檔案）的 **絕對路徑**。
- `FLOW_FILE / TRACE_FILE / TRACE_OUTPUT_FILE / FCT_OUTPUT_FILE / PFC_OUTPUT_FILE / QLEN_MON_FILE` → 一律改為 `runs/<timestamp>_ns3_run/out/` 下的 **絕對路徑**（檔名分別為 `flow.txt, trace.txt, mix.tr, fct.txt, pfc.txt, qlen.txt`）。
- 若提供 `--qcn/--pfc-dyn/--buffer/--payload`，同步覆蓋對應鍵值。

> 目的：**不依賴 CWD、不污染 baseline**，所有輸出固定集中於 `runs/<timestamp>/out/`。

---

## 常見情境範例

1. Baseline：2D 邏輯（2×4）、localBWAware、網路層沿用 baseline（以兩份實測 .et 擴張到 8 節點）
```bash
python scripts/run_ns3.py \
  --workload data/chakra/workload_et \
  --virtual-world 8 \
  --topo auto:2d \
  --phys-topo configs/astra-sim/topos/8_nodes_1_switch_topology.txt \
  --coll-opt localBWAware
```
2. 校準對照：關閉 localBWAware，調整 `local-mem-bw`
```bash
python scripts/run_ns3.py \
  --workload data/chakra/workload_et \
  --virtual-world 8 \
  --topo auto:2d \
  --phys-topo configs/astra-sim/topos/8_nodes_1_switch_topology.txt \
  --coll-opt none --lmbw 2000
```
3. 流控掃描：QCN+動態 PFC、`BUFFER_SIZE=32`、`PACKET_PAYLOAD_SIZE=1000`
```bash
python scripts/run_ns3.py \
  --workload data/chakra/workload_et \
  --virtual-world 8 \
  --topo auto:2d \
  --phys-topo configs/astra-sim/topos/8_nodes_1_switch_topology.txt \
  --qcn 1 --pfc-dyn 1 --buffer 32 --payload 1000
```
4. 擴充到更多節點（未來）：自動 3D（接近立方），指定 128-node 物理拓樸
```bash
python scripts/run_ns3.py \
  --workload data/chakra/workload_et \
  --virtual-world 128 \
  --topo auto:3d \
  --phys-topo configs/astra-sim/topos/128_nodes_32_switch_topology.txt \
  --coll-opt localBWAware --buffer 32 --qcn 1
```
---

## 結果檔案與圖表建議

- `out/fct.txt`：Flow Completion Time（可做 CDF、p95/p99）。
- `out/qlen.txt`：佇列長度時序（可做 CDF、峰值、平均）。
- `out/pfc.txt`：PFC 暫停事件（次數、總時長）。
- `stdout.log`：ASTRA-sim 執行摘要（也可萃取迭代時間）。

建議圖表：**迭代時間 vs K 維**、**FCT CDF**、**Queue 95p**、**PFC 熱圖**。

---

## 疑難排解（FAQ）

- **world=2 卻看到 8 節點？**
  邏輯或物理拓樸指到 8-node 檔。程式會檢查邏輯乘積＝world；物理請改對應 `*_nodes_*.txt`，或使用 `--virtual-world` 與一致的物理拓樸。

- **baseline 相對路徑失效**
  檢查 `tmp/config.patched.txt` 是否已改為絕對路徑；輸出一律在 `out/`。

- **只想檢查設定**
  加 `--dry-run`，查看 `tmp/*.patched.*` 與 `command.txt`。

---

### 版本記錄（Changelog）
- 2025-09-16：初版加入參數表、路徑補丁原理、情境範例與擴充佔位。
- 2025-09-16：整併與修正；加入 --virtual-world、2D/3D 一致範例、路徑補丁說明。
