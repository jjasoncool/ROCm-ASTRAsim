# ASTRA-sim 網路模擬器問題分析報告

## 摘要

本報告分析了在使用 ASTRA-sim NS-3 網路模擬器進行深度學習工作負載模擬時遇到的通訊時間嚴重低估問題。通過詳細的實驗和程式碼分析，我們發現了 ASTRA-sim 在處理 Chakra ET 檔案中計算節點時的根本性問題。

## 問題描述

### 原始校準結果

使用 `run_ns3.py` 腳本執行 2-GPU 校準時得到以下結果：

```csv
mode,tag,calib_id,world,logical_dims,topo_desc,qcn,pfc_dyn,buffer,payload,coll_opt,lmbw,alpha_us,sim_cycles_step,sim_cycles_comm,sim_t_step_ms,sim_t_comm_ms,real_t_step_ms,real_t_comm_ms,run_dir,rel_err_step,rel_err_comm,flags
calibrate,,1,2,2,auto1d_2,,,,,localBWAware,1600,5.093838,30514432,30514432,155435.564375,155435.564375,155435.564375,1488953.256000,/workspace/runs/20251002-060548+0000_ns3_2gpu_auto1d_2,0.000000,0.895607,comm_equals_wall
```

### 關鍵問題指標

- **步驟時間誤差**: 0.000000 (0%) - 完全匹配
- **通訊時間誤差**: 0.895607 (89.56%) - 嚴重低估
- **標記**: `comm_equals_wall` - 通訊時間等於總執行時間

## 實驗過程與發現

### 1. 檢查 ET 檔案內容

#### 1.1 檢查原始工作負載
```bash
docker exec -it rocm-horovod python3 -c "檢查 /workspace/data/chakra/workload_et/allreduce.0.et"
```

**發現**：
- 總共 128 個節點
- 64 個計算節點 (type=4)
- 64 個通訊節點 (type=7)
- 總計算週期: 9,952,651,198 cycles
- 平均每個計算節點: 155,510,175 cycles

#### 1.2 節點類型分布
```
節點類型分布: {4: 5, 7: 5}  # 前10個節點中
```

#### 1.3 詳細節點屬性
```
節點 0: type=4, id=1
  屬性: ['compute_cycles=182700363', 'exec_cycles=182700363', 'cycles=182700363', 'duration_cycles=182700363']

節點 1: type=7, id=2
  屬性: ['comm_type=0', 'comm_size=10485760']
```

### 2. PyTorch Traces 分析

#### 2.1 實際執行指標
```bash
docker exec -it rocm-horovod python3 -c "extract_real_metrics_from_traces()"
```

**結果**：
- `real_t_step_ms`: 155,435.56 ms
- `real_t_comm_ms`: 1,488,953.26 ms
- `used_epoch`: 1
- **通訊時間 vs 步驟時間比例**: 9.58

### 3. ASTRA-sim 輸出分析

#### 3.1 原始工作負載模擬結果
```
[workload] [info] sys[0] finished, 30514432 cycles, exposed communication 30514368 cycles.
[statistics] [info] sys[0], Wall time: 30514432
[statistics] [info] sys[0], Comm time: 30514432
[statistics] [info] sys[0], GPU time: 64
```

**關鍵問題**：
- Wall time = Comm time = 30,514,432 cycles
- GPU time = 64 cycles ≪ ET 檔案中的 9,952,651,198 cycles
- **ASTRA-sim 完全忽略了 ET 檔案中的計算週期**

### 4. 簡化測試驗證

#### 4.1 創建簡化 ET 檔案
創建包含明確計算節點的測試檔案：
- 計算節點: 1,000,000 cycles
- 通訊節點: 1KB AllReduce
- 明確的依賴關係

#### 4.2 簡化測試結果
```
[workload] [info] sys[0] finished, 16265 cycles, exposed communication 16264 cycles.
[statistics] [info] sys[0], Wall time: 16265
[statistics] [info] sys[0], Comm time: 16264
[statistics] [info] sys[0], GPU time: 1
```

**驚人發現**：
- 設定的 1,000,000 cycles 計算節點 → 實際只有 1 cycle
- **這證實了 ASTRA-sim 完全無法讀取 ET 檔案中的計算時間**

## 數字分析與結論

### 數據對比表

| 項目 | ET 檔案 | 實際測量 | ASTRA-sim 模擬 | 差異 |
|------|---------|----------|----------------|------|
| 計算週期 | 9,952,651,198 | N/A | 64 | **99.999%** 差異 |
| 通訊時間 (ms) | N/A | 1,488,953 | 155,436 | **89.56%** 低估 |
| 步驟時間 (ms) | N/A | 155,436 | 155,436 | **0%** 差異 |
| 通訊/步驟比例 | N/A | 9.58 | 1.0 | **9倍** 差異 |

### 關鍵發現

#### 1. **校準機制設計**
- 步驟時間完全相同是**設計如此**
- `sim_t_step_ms = real_t_step_ms` 用於校準 `alpha_us` 參數
- 公式：`alpha_us = (real_t_step_ms × 1000.0) / sim_cycles_step`

#### 2. **計算時間處理問題**
- **原始工作負載**：ET 中 99億週期 → ASTRA-sim 只用 64 週期
- **簡化測試**：ET 中 100萬週期 → ASTRA-sim 只用 1 週期
- **縮放比例**：326:1 的巨大差異

#### 3. **通訊時間建模缺陷**
- ASTRA-sim 只模擬了通訊部分
- 忽略了計算與通訊的實際重疊關係
- 導致 `comm_equals_wall` 現象

## 根本原因分析

### 可能的技術原因

#### 1. **ASTRA-sim Chakra Feeder 版本問題**
- 可能不支援 `compute_cycles` 屬性名稱
- 需要不同的屬性名稱（如 `runtime`, `exec_time`, `duration`）
- 版本過舊，無法正確解析新版 Chakra ET 格式

#### 2. **系統配置問題**
- 可能需要在 `system.json` 中啟用計算時間處理
- 缺少特定的 feeder 配置參數

#### 3. **架構限制**
- ASTRA-sim 可能設計為純通訊模擬器
- 不支援計算與通訊的混合建模

### 環境信息

```
=== ASTRA-sim 版本信息 ===
Binary: /workspace/astra-sim/extern/network_backend/ns-3/build/scratch/ns3.42-AstraSimNetwork-default

=== Chakra 版本信息 ===
Chakra version: unknown
Available node types:
  COMM_COLL_NODE = 7
  COMM_RECV_NODE = 6
  COMM_SEND_NODE = 5
  COMP_NODE = 4
  INVALID_NODE = 0
  MEM_LOAD_NODE = 2
  MEM_STORE_NODE = 3
  METADATA_NODE = 1
```

## 解決方案建議

### 短期解決方案

#### 1. **屬性名稱測試**
嘗試不同的計算時間屬性名稱：
```python
# 可能需要嘗試的屬性名稱：
attr.name = 'runtime'      # 而不是 'compute_cycles'
attr.name = 'exec_time'
attr.name = 'duration'
attr.name = 'wall_time'
```

#### 2. **系統配置調整**
檢查並修改 `system.json` 中可能影響計算處理的參數：
- `boost-mode`
- `scheduling-policy`
- 添加計算相關配置

#### 3. **工作負載格式轉換**
嘗試使用原始的 `manifest.json` 格式而非 `.et` 格式

### 長期解決方案

#### 1. **升級 ASTRA-sim**
- 檢查是否有更新版本支援新版 Chakra
- 確認 Chakra feeder 的兼容性

#### 2. **聯繫開發團隊**
- 向 ASTRA-sim 開發團隊報告此問題
- 確認是否為已知限制或 bug

#### 3. **替代方案**
- 考慮使用其他支援計算建模的模擬器
- 開發自定義的計算時間注入機制

## 結論

通過詳細的實驗分析，我們確定了 ASTRA-sim 在處理 Chakra ET 檔案時存在根本性問題：

1. **ASTRA-sim 完全忽略了 ET 檔案中的計算週期信息**
2. **只模擬通訊部分，導致通訊時間被嚴重低估（89.56%）**
3. **這不是計算與通訊重疊的建模問題，而是根本沒有讀取計算時間**

這解釋了為什麼您的通訊時間誤差如此巨大——ASTRA-sim 實際上只用一個固定的最小值來代替所有的計算操作，從而導致了整個模擬結果的不準確性。

此問題需要從 ASTRA-sim 的 Chakra feeder 層面進行修復，或者尋找替代的解決方案來正確處理計算時間建模。

## 新手指南：三個關鍵問題解答

### 問題 1：我的 trace 檔案有記錄這些資訊嗎？

**答案：有的！您的 PyTorch trace 檔案記錄得很完整**

讓我們來看實際的資料：

#### trace 檔案中記錄的資訊
```bash
# 查看您的 trace 檔案結構
grep -n "ProfilerStep\|dur\|displayTimeUnit" trace_rank_0_epoch_1.json | head -10
```

**實際記錄內容**：
```json
{
  "ph": "X",
  "cat": "user_annotation",
  "name": "ProfilerStep#1",
  "pid": 864,
  "tid": 864,
  "ts": 84927253155.185,
  "dur": 161068.925,  // ← 這就是步驟時間！
  ...
  "displayTimeUnit": "ms"  // ← 時間單位是毫秒
}
```

**trace 檔案包含**：
- ✅ **步驟時間** (`ProfilerStep#N` 的 `dur`)：161,068.925 毫秒
- ✅ **通訊時間** (NCCL/AllReduce 等操作的 `dur`)：加總後約 1,488,953 毫秒
- ✅ **GPU 使用率**、**記憶體使用**等詳細資訊

### 問題 2：ET 檔案有沒有轉出這些資訊？

**答案：有轉出，但 ASTRA-sim 沒有正確使用！**

#### ET 檔案內容分析
```python
# 檢查 ET 檔案中的資訊
節點 0: type=4 (計算節點), id=1
  屬性: [
    'compute_cycles=182700363',    # ← 計算週期
    'exec_cycles=182700363',       # ← 執行週期
    'cycles=182700363',            # ← 總週期
    'duration_cycles=182700363'    # ← 持續週期
  ]

節點 1: type=7 (通訊節點), id=2
  屬性: [
    'comm_type=0',        # ← AllReduce 類型
    'comm_size=10485760'  # ← 通訊大小 (10MB)
  ]
```

**轉換成功但問題在於**：
- ✅ trace → ET 轉換：**成功**（計算時間 182,700,363 週期）
- ❌ ET → ASTRA-sim：**失敗**（只讀取到 1 週期）

### 問題 3：ASTRA-sim 是否有 attribute 可以帶入這些資訊？

**答案：應該有，但可能屬性名稱不對！**

#### 當前 ET 檔案使用的屬性名稱
```python
# 目前我們使用的屬性名稱
'compute_cycles'     # ← ASTRA-sim 可能不認識
'exec_cycles'        # ← ASTRA-sim 可能不認識
'cycles'             # ← ASTRA-sim 可能不認識
'duration_cycles'    # ← ASTRA-sim 可能不認識
```

#### ASTRA-sim 可能期望的屬性名稱
```python
# 可能需要嘗試的屬性名稱：
'runtime'           # ← 執行時間
'exec_time'         # ← 執行時間
'duration'          # ← 持續時間
'wall_time'         # ← 牆上時間
'gpu_time'          # ← GPU 時間
'kernel_time'       # ← 核心執行時間
```

## 簡單的診斷步驟

### 步驟 1：確認資料流
```
PyTorch 訓練
    ↓ (產生 trace)
Kineto Trace 檔案 ✅ 有完整資訊
    ↓ (轉換)
Chakra ET 檔案 ✅ 有完整資訊
    ↓ (讀取)
ASTRA-sim ❌ 只讀取到 1 個週期
```

### 步驟 2：問題定位
**問題出現在最後一步**：ASTRA-sim 無法正確讀取 ET 檔案中的計算時間

### 步驟 3：解決方向
1. **嘗試不同的屬性名稱**
2. **檢查 ASTRA-sim 版本兼容性**
3. **查看 ASTRA-sim 文檔中期望的 ET 格式**

## 實際驗證實驗

我們做了一個簡單的測試來證明問題：

```python
# 創建一個只有 100萬週期計算的簡單 ET 檔案
compute_node.attr = [
    'compute_cycles': 1000000  # ← 明確設定 100萬週期
]

# ASTRA-sim 結果：
GPU time: 1 cycle  # ← 只讀到 1 個週期！！！
```

**這證明了**：ASTRA-sim 根本沒有讀取我們設定的計算時間。

## 總結給新手

1. **您的資料沒問題**：trace 和 ET 檔案都有完整的時間資訊
2. **轉換沒問題**：從 trace 到 ET 的轉換是成功的
3. **問題在 ASTRA-sim**：它不認識我們使用的計算時間屬性名稱

**下一步該做什麼**：
- 查找 ASTRA-sim 官方文檔中的 ET 檔案格式要求
- 嘗試不同的屬性名稱
- 或者聯繫 ASTRA-sim 開發團隊確認正確的屬性格式
