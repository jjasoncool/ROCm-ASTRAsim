#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
ASTRA-sim NS-3 網路模擬器執行腳本

功能概述：
本腳本用於執行 ASTRA-sim ns-3 網路模擬，支援 Chakra 工作負載格式，
並提供自動拓撲生成、配置文件修補、虛擬工作負載擴展等功能。

核心功能：
- 自動生成和驗證邏輯拓撲配置
- 修補系統和網路配置文件
- 虛擬擴展工作負載到更大的世界規模
- 自動校準模擬參數 (alpha_us)
- 從 PyTorch Kineto trace 提取實際運行指標
- 解析 ASTRA-sim 輸出並生成校準報告

支援的工作負載格式：
- .et 文件格式 (支援虛擬擴展)
- manifest.json + et_rank_*.json 格式

主要特性：
- PyTorch Kineto trace 分析，提取 real_t_step_ms 和 real_t_comm_ms
- ASTRA-sim stdout.log 週期數解析 (Wall/Comm cycles)
- 自動校準 alpha_us (微秒/週期)，導出 metrics.csv 和 calibration_all.csv
- A-保護機制：檢測並標記不可信的通訊時間測量

輸出文件：
- out/metrics.csv: 單次執行的詳細指標
- runs/calibration_all.csv: 所有校準結果的彙總資料庫
- stdout.log: 完整的模擬執行日誌
- command.txt: 實際執行的命令記錄

架構說明：
程式採用模組化設計，主要包含以下功能模組：
1. 拓撲管理：邏輯和物理拓撲的生成與驗證
2. 工作負載處理：ET文件操作和虛擬擴展
3. 配置管理：系統和網路配置的修補
4. 指標分析：trace分析和模擬結果解析
5. 校準管理：自動校準和資料庫維護
"""

import argparse, json, math, os, re, subprocess, sys, time, shutil, csv, hashlib
from pathlib import Path
from statistics import median

# -------------------- Chakra protobuf I/O --------------------
from chakra.src.third_party.utils.protolib import encodeMessage as encode_message, decodeMessage as decode_message
from chakra.schema.protobuf.et_def_pb2 import (
    GlobalMetadata, Node, AttributeProto,
    ALL_REDUCE, ALL_GATHER, REDUCE_SCATTER, ALL_TO_ALL
)
# 嘗試支援不同版本的 Compute 節點 enum 名稱（不存在時為 None）
try:
    from chakra.schema.protobuf.et_def_pb2 import COMP_NODE as COMPUTE_ENUM
except Exception:
    try:
        from chakra.schema.protobuf.et_def_pb2 import COMPUTE_NODE as COMPUTE_ENUM
    except Exception:
        COMPUTE_ENUM = None

# ---------- 基本正則與常數 ----------
ET_PAT = re.compile(r"^(?P<prefix>.*)\.(?P<rank>\d+)\.et$")
KINETO_STEP_PAT = re.compile(r"^ProfilerStep#\d+$")  # Pytorch Profiler 的 step 名稱
# 嘗試辨識 NCCL/collective 名稱（不同版本/後端字串會不同，留多一點關鍵字）
KINETO_COMM_HINTS = ("nccl", "rccl", "all_reduce", "allreduce", "allgather", "all_gather", "reduce_scatter", "alltoall", "broadcast", "bytes")

# Calibration CSV 欄位定義
CALIB_FIELDS = [
    'mode', 'tag', 'calib_id', 'world', 'logical_dims', 'topo_desc',
    'qcn', 'pfc_dyn', 'buffer', 'payload', 'coll_opt', 'lmbw',
    'alpha_us', 'alpha_comm_us', 'alpha_gpu_us', 'sim_cycles_step', 'sim_cycles_comm', 'sim_cycles_gpu', 'sim_t_step_ms', 'sim_t_comm_ms',
    'real_t_step_ms', 'real_t_comm_ms', 'real_t_net_comm_ms', 'real_t_kernel_ms', 'run_dir', 'rel_err_step', 'rel_err_comm', 'sim_t_comm_ms_comm', 'rel_err_comm_comm',
    'flags'  # 執行狀態標記（例如 comm_equals_wall_no_compute）
]

# -------------------- World size / logical-dims --------------------

def list_et_rank_files(workload_dir: Path, tag: str = None) -> dict[int, Path]:
    files = {}
    for p in workload_dir.glob("*.et"):
        m = ET_PAT.match(p.name)
        if m:
            # 新增過濾邏輯：若指定 tag，檔名必須包含該字串
            if tag and tag not in m.group("prefix"):
                continue
            files[int(m.group("rank"))] = p
    return dict(sorted(files.items()))

def count_world_size(workload_dir: Path, tag: str = None) -> int:
    et_map = list_et_rank_files(workload_dir, tag)
    if et_map:
        return len(et_map)
    jfiles = list(workload_dir.glob("et_rank_*.json"))
    if workload_dir.joinpath("manifest.json").exists() and jfiles:
        return len(jfiles)
    msg = f" (tag={tag})" if tag else ""
    raise SystemExit(f"[ERR] {workload_dir} 內找不到符合條件{msg}的 .et 或 et_rank_*.json")

def squareish_2d(n: int) -> tuple[int, int]:
    if n < 1:
        raise ValueError("n must be >= 1")
    a = int(math.sqrt(n))
    while a > 1 and n % a != 0:
        a -= 1
    return (a, n // a) if a > 1 else (1, n)

def cubeish_3d(n: int) -> tuple[int, int, int]:
    if n < 1:
        raise ValueError("n must be >= 1")
    a = max(1, int(round(n ** (1/3))))
    best, bestgap = (1, 1, n), n
    for x in range(1, a + 2):
        if n % x: continue
        y, z = squareish_2d(n // x)
        dims = (x, y, z)
        gap = max(dims) - min(dims)
        if gap < bestgap:
            best, bestgap = dims, gap
    return best

def gen_topology_file(out_path: Path, dims: list[int]) -> Path:
    out_path.parent.mkdir(parents=True, exist_ok=True, mode=0o777)
    out_path.parent.chmod(0o777)
    out_path.write_text(json.dumps({"logical-dims": [str(d) for d in dims]}, indent=2), encoding="utf-8")
    return out_path

def load_logical_dims(json_path: Path) -> list[int]:
    obj = json.loads(json_path.read_text(encoding="utf-8"))
    dims = [int(x) for x in obj.get("logical-dims", [])]
    if not dims:
        raise ValueError(f"{json_path} 缺少 'logical-dims'")
    return dims

def assert_logical_dims_match_world(json_path: Path, world: int) -> None:
    dims = load_logical_dims(json_path)
    prod = math.prod(dims)
    if prod != world:
        raise SystemExit(f"[ERR] 邏輯拓樸 {json_path} 維度乘積={prod} ≠ world={world}")

# -------------------- Physical topo / baseline --------------------

def infer_nodes_from_topo_filename(path: Path) -> int | None:
    m = re.search(r"(\d+)_nodes?_", path.name)
    return int(m.group(1)) if m else None

def guess_phys_topo(topos_dir: Path, world: int) -> Path | None:
    cands = list(topos_dir.glob(f"{world}_nodes_*_topology.txt"))
    return cands[0].resolve() if cands else None

# -------------------- Patch system.json / ns-3 config.txt --------------------
def patch_system_json(src: Path, out: Path, coll_opt: str | None, lmbw: int | None) -> Path:
    obj = json.loads(src.read_text(encoding="utf-8"))
    if coll_opt is not None:
        obj["collective-optimization"] = coll_opt
    if lmbw is not None:
        obj["local-mem-bw"] = lmbw
    out.write_text(json.dumps(obj, indent=2), encoding="utf-8")
    return out

def patch_network_cfg(src: Path, out: Path, *,
                      topo_file: Path | None,
                      out_dir: Path,
                      qcn: int | None = None,
                      pfc_dyn: int | None = None,
                      buffer_size: int | None = None,
                      payload: int | None = None) -> Path:
    out_dir.mkdir(parents=True, exist_ok=True, mode=0o777)
    out_dir.chmod(0o777)
    lines = src.read_text(encoding="utf-8").splitlines()

    flow_placeholder = (out_dir / "flow.txt").resolve()
    flow_placeholder.touch(exist_ok=True)

    def replace_key(lines_list, key, value):
        pat = re.compile(rf'^\s*{re.escape(key)}(\s+|=).*$', re.IGNORECASE)
        replaced = False
        new_lines = []
        def _format_val(v):
            # For numeric or simple string values (like BUFFER_SIZE), return str(v).
            # For file paths, attempt to resolve to absolute posix path.
            if v is None:
                return None
            # if it's an int/float, just stringify
            if isinstance(v, (int, float)):
                return str(v)
            try:
                # try to treat as a path; if it fails, fallback to str(v)
                pv = Path(v)
                return pv.resolve().as_posix()
            except Exception:
                return str(v)

        val_str = _format_val(value)
        for ln in lines_list:
            if pat.match(ln):
                if value is not None and not replaced:
                    new_lines.append(f"{key} {val_str}")
                    replaced = True
            else:
                new_lines.append(ln)
        if (value is not None) and (not replaced):
            new_lines.append(f"{key} {val_str}")
        return new_lines

    lines_wo_flow = []
    flow_pat = re.compile(r'^\s*FLOW_FILE(\s+|=)', re.IGNORECASE)
    for ln in lines:
        if not flow_pat.match(ln):
            lines_wo_flow.append(ln)

    def set_optional_key(lines_list, key, value):
        return replace_key(lines_list, key, value) if value is not None else lines_list

    lines2 = lines_wo_flow
    lines2 = set_optional_key(lines2, "ENABLE_QCN", qcn)
    lines2 = set_optional_key(lines2, "USE_DYNAMIC_PFC_THRESHOLD", pfc_dyn)
    lines2 = set_optional_key(lines2, "BUFFER_SIZE", buffer_size)
    lines2 = set_optional_key(lines2, "PACKET_PAYLOAD_SIZE", payload)

    if topo_file is not None:
        lines2 = replace_key(lines2, "TOPOLOGY_FILE", topo_file)

    out_targets = {
        "TRACE_FILE":        out_dir / "trace.txt",
        "TRACE_OUTPUT_FILE": out_dir / "mix.tr",
        "FCT_OUTPUT_FILE":   out_dir / "fct.txt",
        "PFC_OUTPUT_FILE":   out_dir / "pfc.txt",
        "QLEN_MON_FILE":     out_dir / "qlen.txt",
    }
    for k, v in out_targets.items():
        v.parent.mkdir(parents=True, exist_ok=True, mode=0o777)
        v.parent.chmod(0o777)
        if not v.exists():
            v.touch()
        lines2 = replace_key(lines2, k, v)

    new_lines = [f"FLOW_FILE {flow_placeholder.as_posix()}"]
    new_lines.extend(lines2)

    out.write_text("\n".join(new_lines) + "\n", encoding="utf-8")
    return out

# -------------------- Virtual expand ET --------------------

COMM_TYPES = {ALL_REDUCE, ALL_GATHER, REDUCE_SCATTER, ALL_TO_ALL}

def _read_all_nodes(et_path: Path) -> tuple[GlobalMetadata, list[Node]]:
    nodes = []
    with et_path.open("rb") as f:
        meta = GlobalMetadata()
        decode_message(f, meta)
        size = et_path.stat().st_size
        while f.tell() < size:
            pos0 = f.tell()
            node = Node()
            try:
                decode_message(f, node)
            except Exception:
                break
            if f.tell() <= pos0:
                break
            nodes.append(node)
    return meta, nodes

def _scale_comm_size_inplace(node: Node, scale: float) -> None:
    comm_type = None
    size_idx = None
    for i, a in enumerate(node.attr):
        if a.name == "comm_type":
            comm_type = a.int64_val
        elif a.name == "comm_size":
            size_idx = i
    if comm_type in COMM_TYPES and size_idx is not None:
        old = node.attr[size_idx].int64_val
        new_val = max(1, int(round(old * scale)))
        node.attr[size_idx].int64_val = new_val

def _write_et(meta: GlobalMetadata, nodes: list[Node], out_path: Path) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True, mode=0o777)
    out_path.parent.chmod(0o777)
    with out_path.open("wb") as f:
        encode_message(f, meta)
        for n in nodes:
            encode_message(f, n)

def expand_workload_virtual_if_needed(workload_src: Path, tmp_dir: Path, virtual_world: int | None, tag: str = None) -> tuple[Path, int]:
    # 傳入 tag 以確保讀取正確的來源檔案
    et_map = list_et_rank_files(workload_src, tag)
    M = len(et_map)

    if virtual_world is None:
        return workload_src, M

    if virtual_world < 2:
        raise SystemExit("[ERR] --virtual-world 至少需要 2。")
    if not et_map:
        raise SystemExit("[ERR] --virtual-world 目前僅支援 .et 工作負載（manifest+JSON 請先轉 .et）。")

    N = int(virtual_world)
    if N == M:
        print(f"[INFO] virtual-world={N} 與來源相同；不做擴張。")
        return workload_src, M

    any_path = next(iter(et_map.values()))
    m = ET_PAT.match(any_path.name)
    prefix = m.group("prefix") if m else "et"

    scale = (M * (N - 1)) / (N * (M - 1))
    print(f"[INFO] 以來源 M={M} ({prefix}) 擴張到 N={N}；通訊 bytes 縮放係數 scale={scale:.6f}")

    out_dir = tmp_dir / f"workload_{N}"
    for r in range(N):
        src_rank = r % M
        src_path = et_map[src_rank]
        meta, nodes = _read_all_nodes(src_path)
        for node in nodes:
            _scale_comm_size_inplace(node, scale)

        # 保持與來源相同的 prefix (包含 tag)，僅變更 rank
        dst_path = out_dir / f"{prefix}.{r}.et"
        _write_et(meta, nodes, dst_path)

    print(f"[INFO] 產生虛擬工作負載：{out_dir} （共 {N} 份 .et）")
    return out_dir, N

# -------------------- Topo helpers --------------------

def extract_topo_description(topo_arg: str, logical_dims: list[int] = None) -> str:
    if topo_arg.startswith("auto:"):
        mode = topo_arg.split(":", 1)[1]
        if logical_dims:
            dims_str = "x".join(map(str, logical_dims))
            return f"auto{mode}_{dims_str}"
        else:
            return f"auto{mode}"
    elif topo_arg.startswith("dims:"):
        dims_str = topo_arg.replace("dims:", "").lower()
        return f"dims_{dims_str}"
    elif topo_arg.startswith("file:"):
        file_path = Path(topo_arg.replace("file:", ""))
        return f"file_{file_path.stem}"
    else:
        file_path = Path(topo_arg)
        return f"file_{file_path.stem}"

def build_workload_arg(workload_dir: Path, tag: str = None) -> tuple[str, str]:
    et_map = list_et_rank_files(workload_dir, tag)
    if et_map:
        any_path = next(iter(et_map.values()))
        m = ET_PAT.match(any_path.name)
        prefix = m.group("prefix") if m else "et"
        prefix_path = (workload_dir / prefix).as_posix()
        return prefix_path, f"prefix={prefix_path}"
    else:
        return workload_dir.as_posix(), f"dir={workload_dir.as_posix()}"

# ---------- 工作負載檔案驗證 ----------
def _validate_workload_integrity(workload_dir: Path, tag: str = None) -> None:
    """
    驗證工作負載檔案的基本完整性
    針對常見的 "Node X in ctrl_dep graph, but not found in index" 錯誤提供預警
    """
    et_map = list_et_rank_files(workload_dir, tag)
    if not et_map:
        return  # 非 .et 格式，跳過驗證

    # 取樣檢查第一個 .et 檔案
    first_et = next(iter(et_map.values()))
    try:
        _, nodes = _read_all_nodes(first_et)
        if not nodes:
            raise ValueError(f"{first_et.name} 包含 0 個節點")

        # 檢查是否有基本的節點屬性
        sample_node = nodes[0]
        if not hasattr(sample_node, 'id') or not hasattr(sample_node, 'type'):
            raise ValueError(f"{first_et.name} 節點結構不完整")

        # 檢查節點類型分布，識別潛在的格式問題
        node_types = {}
        for node in nodes[:min(100, len(nodes))]:  # 只檢查前100個節點避免性能問題
            node_types[node.type] = node_types.get(node.type, 0) + 1

        print(f"[INFO] 工作負載驗證通過: {len(et_map)} 個 .et 檔案，首檔包含 {len(nodes)} 個節點")
        print(f"[INFO] 節點類型分布: {dict(list(node_types.items())[:5])}")

        # 警告：如果檔案能被 Python 讀取但 ASTRA-sim 無法處理，可能是版本兼容性問題
        if len(nodes) > 1000:  # 大型檔案更容易出現兼容性問題
            print(f"[WARN] 檔案較大({len(nodes)}節點)，如果執行失敗可能是 Chakra ET 格式版本兼容性問題")

    except Exception as e:
        raise ValueError(f"工作負載檔案可能損壞: {e}")

# ---------- 解析 PyTorch Kineto traces（real_*） ----------
def _load_json(p: Path) -> dict | None:
    try:
        return json.loads(p.read_text(encoding="utf-8"))
    except Exception:
        return None

def _trace_dir_default(script_path: Path) -> Path:
    # scripts 與 data 同層：<root>/scripts/run_ns3.py 與 <root>/data/...
    root = script_path.parent.parent
    return root / "data" / "chakra" / "pytorch_traces"

def list_trace_groups(trace_dir: Path, tag: str = None) -> list[tuple[int, dict[int, Path], str]]:
    """
    列出 trace_dir 中可辨識的 trace 群組，回傳 list of (epoch_id, {rank->Path}, source_type)
    source_type in {'trace_rank', 'device', 'host', 'hdt'}

    - 如果存在 trace_rank_<r>_epoch_<e>.json，依 epoch 分群並以 rank-id 做鍵
    - 否則，如果存在 device_*.json / host_*.json / hdt_*.json，則把同類型檔案視為單一 epoch=0 的群組，rank 從檔名抽出數字
    - 支援依據 tag 過濾 (e.g. host_0_cifar10.json)
    """
    groups = []
    # 1) per-epoch trace_rank_* pattern
    epoch_map: dict[int, dict[int, Path]] = {}
    for p in trace_dir.glob('trace_rank_*_epoch_*.json'):
        if tag and tag not in p.name: continue
        m = re.search(r'trace_rank_(\d+)_epoch_(\d+)\.json$', p.name)
        if not m:
            continue
        r = int(m.group(1)); e = int(m.group(2))
        epoch_map.setdefault(e, {})[r] = p
    if epoch_map:
        for e in sorted(epoch_map.keys()):
            groups.append((e, epoch_map[e], 'trace_rank'))
        return groups

    # 2) device_*.json
    pattern = f"device_*_{tag}.json" if tag else "device_*.json"
    devs = sorted(trace_dir.glob(pattern))
    if devs:
        ranks = {}
        for p in devs:
            m = re.match(r"device_(\d+)(?:_.*)?\.json$", p.name)
            if not m:
                continue
            r = int(m.group(1)); ranks[r] = p
        if ranks:
            groups.append((0, ranks, 'device'))
            return groups

    # 3) host_*.json
    pattern = f"host_*_{tag}.json" if tag else "host_*.json"
    hosts = sorted(trace_dir.glob(pattern))
    if hosts:
        ranks = {}
        for p in hosts:
            m = re.match(r"host_(\d+)(?:_.*)?\.json$", p.name)
            if not m:
                continue
            r = int(m.group(1)); ranks[r] = p
        if ranks:
            groups.append((0, ranks, 'host'))
            return groups

    # 4) hdt_*.json (only accept if looks like containing timing info)
    hdt_pat = f"hdt_*_{tag}.json" if tag else "hdt_*.json"
    hdts = sorted(trace_dir.glob(hdt_pat))
    if hdts:
        ranks = {}
        def _hdt_may_have_timing(p: Path) -> bool:
            try:
                txt = p.open('r', encoding='utf-8', errors='ignore').read(200000)
            except Exception:
                return False
            if '"traceEvents"' in txt or 'ProfilerStep' in txt:
                return True
            if '"start_ts"' in txt or '"finish_ts"' in txt:
                return True
            for k in ('"duration"', '"dur"', '"duration_ms"', '"exec"', '"cycles"', '"timestamp"'):
                if k in txt:
                    return True
            return False
        for p in hdts:
            m = re.search(r'hdt_(\d+)(?:_.*)?\.json$', p.name)
            if not m:
                continue
            r = int(m.group(1))
            if _hdt_may_have_timing(p):
                ranks[r] = p
        if ranks:
            groups.append((0, ranks, 'hdt'))
            return groups

    return []

def _dur_units_to_ms(ev_obj: dict, dur_val: float) -> float:
    """
    將 Trace 中的持續時間數值 (dur) 統一轉換為毫秒 (ms)。

    [修訂歷史 - Critical Fix]
    原本邏輯會讀取 ev_obj["displayTimeUnit"]。
    但實測發現：AMD/PyTorch Profiler 雖然標註 "ms"，但內部數值 (如 87) 其實是微秒 (us)。
    - 若信賴 "ms" 標籤 -> 87ms -> 總時間膨脹 1000 倍 -> ResNet Alpha 暴增至 2.25。
    - 若視為 "us" (物理事實) -> 87us -> 總時間正常 -> ResNet Alpha 回歸 0.002。

    因此，這裡採取「強制轉型」策略，無視 JSON 標頭的單位宣告。
    """

    # [物理定義]
    # Google Chrome Trace Event Format 規範定義 dur 單位為 微秒 (microseconds)。
    # 我們需要回傳 毫秒 (milliseconds) 給 ASTRA-sim 進行後續計算。
    # 公式： ms = us / 1000.0
    name = str(ev_obj.get("name", ""))
    val = float(dur_val)

    # 1. 絕對特徵判定：如果是 ProfilerStep，必定是 CPU 端時間 (ns)
    if "ProfilerStep" in name:
        # 如果視為 ns (除以 100萬) 結果 < 1.0 ms，這不合理，代表它其實是 us
        if (val / 1_000_000.0) < 1.0:
            return val / 1_000.0     # 視為 us
        return val / 1_000_000.0     # 視為 ns

    # 2. 絕對特徵判定：如果是通訊 (nccl) 或計算 (Kernel)，通常是 us
    # 加強防呆：若沒有名字特徵，才退回到數值門檻法

    # 數值門檻法 (作為最後一道防線)
    # 邏輯：Kernel 不太可能跑超過 10秒 (10^7 us)，Step 不太可能短於 0.01ms (10^4 ns)
    # 但為了包含極短 Step，我們依賴上面的 Name Check。
    # 這裡保留門檻是為了處理那些沒有名字特徵的雜項事件。
    if val > 10_000_000:
        return val / 1_000_000.0 # 視為 ns
    else:
        return val / 1_000.0     # 視為 us

def _extract_step_and_comm_ms(trace_path: Path) -> tuple[list[float], float | None]:
    """
    從單支 PyTorch Kineto trace（Chrome trace JSON）抓出：
      1) 每個 ProfilerStep 的持續時間（ms）
      2) 每步平均的通訊時間（ms/step；若抓不到通訊事件則回傳 None）

    注意：
    - Kineto 事件時間單位可能是 us 或 ms；用 _dur_units_to_ms() 做一致化（轉成毫秒）。
    - 通訊事件靠 name/cat 中是否含 NCCL/RCCL/collective 關鍵字來辨識（KINETO_COMM_HINTS）。
    - 這裡「每步平均通訊」是把檔內通訊事件總時長除以步數，與 ASTRA‑sim 的 per‑step exposed comm 對齊。
    """
    obj = _load_json(trace_path)
    if not obj:
        return [], None

    steps_ms: list[float] = []     # 每一步（ProfilerStep#N）的時長（ms）
    comm_sum_ms: float = 0.0       # 檔內所有可辨識的通訊事件時長總和（ms）
    any_comm: bool = False         # 是否有抓到任何通訊事件

    for ev in obj.get("traceEvents", []):
        # 只看具有持續時間的 complete events（"ph" == "X"）
        if ev.get("ph") != "X":
            continue

        name = str(ev.get("name", "")).lower()
        cat  = str(ev.get("cat", "")).lower()
        # 將 dur 統一換算成毫秒（Kineto 可能為 us；若 displayTimeUnit=="ms" 則維持 ms）
        d_ms = _dur_units_to_ms(ev, ev.get("dur", 0.0))

        # 判定一步的界線：PyTorch Profiler 會發出 "ProfilerStep#<n>"
        if KINETO_STEP_PAT.match(str(ev.get("name", ""))):
            steps_ms.append(d_ms)

        # 條件 A: 名稱或分類包含通訊關鍵字 (nccl, allreduce, bytes...)
        is_comm_candidate = any(h in name for h in KINETO_COMM_HINTS) or any(h in cat for h in KINETO_COMM_HINTS)

        # 條件 B: 必須「不是」Kernel 實作 (避免重複計算 GPU 執行時間)
        # ROCm Trace 常有 "ncclKernel" 或 "RingKernel"，這些屬於 Compute/Kernel 類別，不應計入 Network Comm
        is_not_kernel = ('kernel' not in name) and ('kernel' not in cat)

        if is_comm_candidate and is_not_kernel:
            comm_sum_ms += d_ms
            any_comm = True

    # ★關鍵修正：回傳「每步通訊」（ms/step）而不是「整個檔案總和」
    # 這樣 extract_real_metrics_from_traces() 取得的 real_t_comm_ms 就是 per-step 的量，
    # 才能與 ASTRA‑sim 的 sim_t_comm_ms（per-step 的 exposed communication）一致比較。
    comm_ms_per_step = (comm_sum_ms / max(1, len(steps_ms))) if (any_comm and steps_ms) else None
    return steps_ms, comm_ms_per_step

def extract_real_metrics_from_traces(trace_dir: Path, tag: str = None) -> tuple[float | None, float | None, float | None, int | None]:
    """
    掃描 trace 群組，回傳多項 real metrics，用於研究比較（Strategy C 默認行為）：
      • real_t_step_ms: per-step wall time（median of per-rank medians）
      • real_t_net_comm_ms: per-step network-only communication time（只算 name 含 '|bytes=' 且非 kernel 的事件）
      • real_t_kernel_ms: per-step GPU NCCL kernel time（kernel 類事件或 name 含 nccldevkernel）
      • used_epoch: 選用的 epoch id

    回傳: (real_t_step_ms, real_t_net_comm_ms, real_t_kernel_ms, used_epoch)
    為向後相容，呼叫端若只解包三個值則會拿到 (step, net_comm, used_epoch)
    """
    groups = list_trace_groups(trace_dir, tag)
    if not groups:
        print(f"[WARN] {trace_dir} 找不到任何可用的 trace 群組（device_/host_/trace_rank_*/hdt_*）。")
        return None, None, None, None

    used_epoch, ranks, src = groups[0]
    per_rank_step_medians = []
    per_rank_net_comm = []
    per_rank_kernel = []

    print(f"[INFO] 正在分析 Trace (Source: {src})...")

    for r, p in sorted(ranks.items()):
        obj = _load_json(p)
        if not obj:
            continue

        # steps list and count
        steps = [ev for ev in obj.get('traceEvents', []) if KINETO_STEP_PAT.match(str(ev.get('name', '')))]
        steps_n = len(steps)

        # compute total network-only and kernel-only durations in ms
        net_ms_total = 0.0
        kernel_ms_total = 0.0
        step_durations = []

        for ev in obj.get('traceEvents', []):
            if ev.get('ph') != 'X':
                continue
            name = str(ev.get('name', ''))
            cat = str(ev.get('cat', ''))
            d_ms = _dur_units_to_ms(ev, ev.get('dur', 0.0))
            lname = name.lower()
            lcat = cat.lower()

            # step durations collected for step median
            if KINETO_STEP_PAT.match(name):
                step_durations.append(d_ms)

            # 邏輯：(名稱或分類包含通訊關鍵字) AND (不是 Kernel)
            # KINETO_COMM_HINTS 包含了 'nccl', 'rccl', 'bytes', 'allreduce' 等
            is_comm_candidate = any(h in lname for h in KINETO_COMM_HINTS) or any(h in lcat for h in KINETO_COMM_HINTS)
            is_not_kernel = ('kernel' not in lname) and ('kernel' not in lcat)

            if is_comm_candidate and is_not_kernel:
                net_ms_total += d_ms

            # 3. 抓取 Kernel 時間 (計算)
            # 包含 kernel category 或名字裡有 kenerl/nccldevkernel
            if 'kernel' in lcat or 'nccldevkernel' in lname or 'devkernel' in lname:
                kernel_ms_total += d_ms

        if step_durations:
            per_rank_step_medians.append(median(step_durations))
        # per-step metrics: divide totals by number of steps to get ms/step
        if steps_n > 0:
            per_rank_net_comm.append(net_ms_total / steps_n)
            per_rank_kernel.append(kernel_ms_total / steps_n)

    real_t_step_ms = median(per_rank_step_medians) if per_rank_step_medians else None
    real_t_net_comm_ms = median(per_rank_net_comm) if per_rank_net_comm else None
    real_t_kernel_ms = median(per_rank_kernel) if per_rank_kernel else None

    print(f"[INFO] 使用群組 source={src} epoch={used_epoch} 的 {len(ranks)} 個 rank 做為 real metrics (net_comm/kernel split)")
    return real_t_step_ms, real_t_net_comm_ms, real_t_kernel_ms, used_epoch

# ---------- 解析 ASTRA‑sim stdout.log（cycles） ----------
def parse_astra_stdout_cycles(stdout_path: Path) -> tuple[int | None, int | None, int | None]:
    """
    回傳 (sim_cycles_step, sim_cycles_comm, sim_cycles_gpu)：
      • 先找 [statistics] sys[X], Wall time: N / Comm time: M / GPU time: G
      • 否則回退抓 [workload] sys[X] finished, N cycles, exposed communication M cycles
      • 多卡時取各欄位的「最大值」代表整步完成時間。
      • 若只抓到 wall，comm 回傳 None（避免誤把 wall 當 comm）。
    """
    if not stdout_path.exists():
        return None, None

    s = stdout_path.read_text(encoding="utf-8", errors="ignore")

    wall: list[int] = []
    comm: list[int] = []
    gpu: list[int] = []

    # 1) 統計段，對大小寫／同義詞更寬鬆
    pat_stat_wall = re.compile(r"\[statistics\].*?sys\[\d+\].*?Wall\s*time:\s*(\d+)", re.IGNORECASE)
    pat_stat_comm = re.compile(r"\[statistics\].*?sys\[\d+\].*?(?:Comm|Communication)\s*time:\s*(\d+)", re.IGNORECASE)
    pat_stat_gpu = re.compile(r"\[statistics\].*?sys\[\d+\].*?GPU\s*time:\s*(\d+)", re.IGNORECASE)
    wall += [int(x) for x in pat_stat_wall.findall(s)]
    comm += [int(x) for x in pat_stat_comm.findall(s)]
    gpu += [int(x) for x in pat_stat_gpu.findall(s)]

    # 2) finished 行（多版本相容）
    finish_patterns: list[tuple[re.Pattern, str]] = [
        # 正常順序： total cycles, exposed communication cycles
        (re.compile(r"workload.*?sys\[\d+\].*?finished,\s*(\d+)\s*cycles,\s*exposed\s*communication\s*(\d+)\s*cycles", re.IGNORECASE), "wall_comm"),
        # 變體關鍵字（exposed comm / 冒號）
        (re.compile(r"workload.*?sys\[\d+\].*?finished,\s*(\d+)\s*cycles,\s*exposed\s*comm[:=]?\s*(\d+)\s*cycles", re.IGNORECASE), "wall_comm"),
        # 反序（先出 exposed comm，再 total）
        (re.compile(r"exposed\s*comm(?:unication)?[:=]?\s*(\d+)\s*cycles.*?sys\[\d+\].*?finished.*?(\d+)\s*cycles", re.IGNORECASE), "comm_wall"),
    ]
    for pat, order in finish_patterns:
        for m in pat.finditer(s):
            a = int(m.group(1)); b = int(m.group(2))
            if order == "wall_comm":
                wall.append(a); comm.append(b)
            else:
                comm.append(a); wall.append(b)

    sim_cycles_step = max(wall) if wall else None
    sim_cycles_comm = max(comm) if comm else None
    sim_cycles_gpu = max(gpu) if gpu else None

    return sim_cycles_step, sim_cycles_comm, sim_cycles_gpu

# ---------- 檢查 .et 是否含 Compute 節點（A-保護用） ----------
def _node_has_compute_cycles(n: Node) -> bool:
    """判斷節點是否具有可被 feeder 辨識的 compute 週期屬性（盡量相容多版本鍵名）。"""
    keys = {"compute_cycles", "exec_cycles", "cycles", "duration_cycles"}
    for a in n.attr:
        if a.name in keys and isinstance(getattr(a, "int64_val", None), int):
            return True
    return False

def detect_compute_nodes_in_et(workload_dir: Path, tag: str = None, sample_limit: int = 2) -> bool:
    """
    掃描工作負載目錄下的 .et，判定是否存在「Compute 節點」：
      • Node.type == COMP_NODE/COMPUTE_NODE（若 enum 有提供）
      • 或節點 attr 含 compute_cycles/exec_cycles/cycles/duration_cycles 任一鍵
    為降低成本，僅抽樣前 sample_limit 份 .et。
    """
    et_map = list_et_rank_files(workload_dir, tag)
    if not et_map:
        return False
    checked = 0
    for _, p in et_map.items():
        try:
            _, nodes = _read_all_nodes(p)
            for n in nodes:
                if COMPUTE_ENUM is not None and n.type == COMPUTE_ENUM:
                    return True
                if _node_has_compute_cycles(n):
                    return True
        except Exception:
            pass
        checked += 1
        if checked >= sample_limit:
            break
    return False

# ---------- metrics / calibration 輸出 ----------
def _row_key_for_dedup(row: dict) -> str:
    """對 calibration_all.csv 做去重用的 key。"""
    cols = ["world","logical_dims","topo_desc","qcn","pfc_dyn","buffer","payload","coll_opt","lmbw"]
    base = "|".join(str(row.get(c,"")) for c in cols)
    return hashlib.md5(base.encode("utf-8")).hexdigest()

def _append_calibration_db(db_path: Path, row: dict) -> None:
    db_path.parent.mkdir(parents=True, exist_ok=True)
    rows = []
    header = []
    if db_path.exists():
        with db_path.open("r", newline="", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            header = reader.fieldnames or []
            rows = list(reader)

    # 1) 若舊檔 header 與 CALIB_FIELDS 不一致，先用 CALIB_FIELDS 重寫一次歷史資料（能對齊就對齊，否則留空）
    if header and header != CALIB_FIELDS:
        print(f"[INFO] 重整 {db_path} 欄位順序 → CALIB_FIELDS")
        fixed_rows = []
        for r in rows:
            fixed = {k: r.get(k, "") for k in CALIB_FIELDS}
            fixed_rows.append(fixed)
        with db_path.open("w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=CALIB_FIELDS)
            writer.writeheader()
            writer.writerows(fixed_rows)
        rows = fixed_rows

    # 2) 去重：若 key 相同且 real/sim/alpha 完全一致就不再追加
    new_key = _row_key_for_dedup(row)
    def _almost_equal(a, b, rel_tol=1e-4, abs_tol=1e-6):
        try:
            fa = float(a)
            fb = float(b)
        except Exception:
            return False
        # handle zeros
        if abs(fa - fb) <= abs_tol:
            return True
        return abs(fa - fb) <= rel_tol * max(abs(fa), abs(fb), 1.0)

    for old in rows:
        if _row_key_for_dedup(old) == new_key:
            # Compare core calibration values using numeric tolerance when possible.
            compare_keys = [
                "alpha_us", "sim_t_step_ms", "sim_t_comm_ms",
                "real_t_step_ms", "real_t_comm_ms",
                "real_t_net_comm_ms", "real_t_kernel_ms"
            ]
            same = True
            for k in compare_keys:
                oldv = old.get(k, "")
                newv = row.get(k, "")
                if oldv == "" and newv == "":
                    continue
                # try numeric compare first
                if _almost_equal(oldv, newv):
                    continue
                # fallback to string equality
                if str(oldv) == str(newv):
                    continue
                same = False
                break
            if same:
                print(f"[INFO] calibration_all.csv 已有相同條目，略過追加。")
                return

    # 3) 追加一筆（確保用 CALIB_FIELDS 順序）
    with db_path.open("a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=CALIB_FIELDS)
        if f.tell() == 0:
            writer.writeheader()
        row_full = {k: row.get(k, "") for k in CALIB_FIELDS}
        writer.writerow(row_full)
        print(f"[INFO] 已追加校準紀錄 → {db_path}")

def export_metrics(out_dir: Path, row: dict) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    csv_path = out_dir / "metrics.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(row.keys()))
        writer.writeheader()
        writer.writerow(row)
    print(f"[INFO] 指標輸出 → {csv_path}")

# =============================================================================
# [優化版] 模擬執行核心函式 (記憶體緩衝模式 - High Performance)
# =============================================================================
def run_simulation(cmd: list, stdout_path: Path, logroot: Path) -> int:
    """
    執行 NS3 模擬，採用全記憶體緩衝以最大化效能。

    優化重點：
    1. 移除逐行寫檔 I/O，改用記憶體 list 暫存。
    2. 僅在最後一次性寫入 stdout.log。
    3. 保留所有即時監控、靜默偵測與錯誤診斷邏輯。

    Args:
        cmd: 執行指令列表
        stdout_path: stdout.log 的完整路徑
        logroot: 若執行失敗需要清理的根目錄

    Returns:
        int: 程序回傳碼 (0=成功, 非0=失敗)
    """

    # 準備記憶體緩衝區 (完全不開檔，減少 I/O)
    full_log_buffer = []

    start_time = time.time()
    last_output_time = start_time
    qp_enabled_seen = False
    simulation_started = False

    try:
        # 使用 Popen 啟動子程序
        # bufsize=1 確保我們可以即時讀到 stdout，不會被 OS 卡住
        process = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT, # 將 stderr 合併到 stdout
            universal_newlines=True,
            bufsize=1
        )

        # 逐行讀取輸出 (即時監控迴圈)
        for line in process.stdout:
            current_time = time.time()
            elapsed = current_time - start_time

            line_strip = line.rstrip()
            formatted_line = f"[{elapsed:6.1f}s] {line_strip}\n"
            # 1. 存入記憶體
            full_log_buffer.append(formatted_line)

            # 2. 螢幕顯示
            print(formatted_line.rstrip())

            # 3. 靜默/卡死偵測邏輯 (完全保留)
            line_lower = line.lower()

            # 狀態更新
            if "qp is enabled" in line_lower:
                qp_enabled_seen = True
                print(f"[INFO] NS3 網路模擬開始執行...")
            elif qp_enabled_seen and ("maxrtt" in line_lower or "finished" in line_lower):
                simulation_started = True

            # 計算靜默時間
            silence_duration = current_time - last_output_time

            # 根據階段決定警告門檻
            if qp_enabled_seen and not simulation_started:
                # NS3 計算密集階段：容忍 120 秒
                warning_threshold = 120
                if silence_duration > warning_threshold:
                    print(f"[WARN] NS3 模擬已靜默 {silence_duration:.1f} 秒（正常現象，網路模擬進行中）")
                    print(f"[INFO] 如確實卡死可按 Ctrl+C 中斷")
                    last_output_time = current_time # 重置以免洗版
            else:
                # 一般階段：容忍 30 秒
                warning_threshold = 30
                if silence_duration > warning_threshold:
                    print(f"[WARN] 已 {silence_duration:.1f} 秒無新輸出，可能卡死...")
                    print(f"[INFO] 如需中斷請按 Ctrl+C")
                    last_output_time = current_time

            last_output_time = current_time

        # 等待程序完全結束
        process.wait()

        # 模擬結束後，一次性將記憶體內容寫入硬碟
        try:
            with stdout_path.open("w", encoding="utf-8") as lf:
                lf.writelines(full_log_buffer)
        except Exception as io_err:
            print(f"[WARN] 無法寫入 Log 檔案: {io_err}")

        # 檢查回傳碼
        if process.returncode != 0:
            raise subprocess.CalledProcessError(process.returncode, cmd)

    except subprocess.CalledProcessError as e:
        print("-" * 80)
        print(f"[ERR] NS3 執行失敗，退出碼: {e.returncode}")

        # 錯誤診斷邏輯 (直接分析記憶體中的資料，不用再讀檔，速度更快)
        # 將 list 轉為大字串以便搜尋
        all_log_text = "".join(full_log_buffer)

        if "Node 0 in ctrl_dep graph, but not found in index" in all_log_text:
            print(f"[DIAG] 檢測到工作負載檔案損壞/不兼容錯誤")
            print(f"[ISSUE] 這通常是 Chakra ET 格式版本與 ASTRA-sim 版本不匹配造成的")
            print(f"[FIX] 解決方案: 請重新生成工作負載，或使用 --workload 指定兼容範例")
        elif "file might be corrupted" in all_log_text:
            print(f"[DIAG] 檢測到檔案損壞錯誤")
        elif "Permission denied" in all_log_text:
            print(f"[DIAG] 檢測到權限錯誤")
            print(f"[FIX] 請嘗試執行: chmod -R 755 /workspace")
        else:
            print(f"[DIAG] 通用錯誤，請檢查 {stdout_path} 獲取詳細資訊")

        # 清理失敗的目錄：僅保留 stdout.log
        if logroot.exists():
            try:
                for item in logroot.iterdir():
                    if item.name == "stdout.log":
                        continue
                    if item.is_dir():
                        shutil.rmtree(item)
                    else:
                        item.unlink()
                print(f"[INFO] 執行失敗，已刪除其他輸出，保留 {stdout_path}")
            except Exception as cleanup_err:
                print(f"[WARN] 清理失敗輸出時發生錯誤: {cleanup_err}")

        return e.returncode # 回傳非 0

    except KeyboardInterrupt:
        print("\n[INFO] 用戶中斷執行")
        if 'process' in locals():
            process.terminate()
            time.sleep(1)
            if process.poll() is None:
                print("[INFO] 強制終止進程...")
                process.kill()

        # 即使中斷，也嘗試把目前跑到的 Log 寫入，方便查看卡在哪
        if full_log_buffer:
             try:
                with stdout_path.open("w", encoding="utf-8") as lf:
                    lf.writelines(full_log_buffer)
             except: pass

        return 130 # 標準的中斷退出碼

    return 0 # 成功

# ---------- 主流程 ----------
def main():
    ap = argparse.ArgumentParser(description="Run ASTRA-sim ns-3 with Chakra workload")
    # 必要輸入
    ap.add_argument("--workload", required=True, help="資料夾：.et 或 manifest.json+et_rank_*.json")
    ap.add_argument("--system",  default="configs/astra-sim/system/system.json", help="system.json (baseline)")
    ap.add_argument("--network", default="configs/astra-sim/ns3/config.txt",     help="ns-3 config.txt (baseline)")
    ap.add_argument("--remote",  default="configs/astra-sim/remote_memory.json", help="remote_memory.json")
    ap.add_argument("--ns3-bin", default=os.environ.get(
        "ASTRA_NS3_BIN",
        os.path.join(os.environ.get("ASTRA_SIM", "/workspace/astra-sim"),
                     "extern/network_backend/ns-3/build/scratch/ns3.42-AstraSimNetwork-default")
    ), help="ns-3 backend binary")
    # 拓樸與擴張
    ap.add_argument("--phys-topo", default=None,
                    help="ns-3 實體拓樸 .txt；未指定嘗試在 configs/astra-sim/topos/ 依 world 推測")
    ap.add_argument("--topo", default="auto:1d",
                    help="邏輯拓樸：auto:1d|auto:2d|auto:3d|dims:2x4|dims:2x2x2|file:/path/to.json")
    ap.add_argument("--virtual-world", type=int, default=None,
                    help="以『實測 .et』當模板，臨時擴張到 N（僅支援 .et）")
    # 系統層/網路層覆蓋
    ap.add_argument("--coll-opt", default=None, help='覆蓋 system.json "collective-optimization"（例 localBWAware / none）')
    ap.add_argument("--lmbw", type=int, default=None, help="覆蓋 system.json local-mem-bw（例 1600）")
    ap.add_argument("--qcn", type=int, default=None, help="ENABLE_QCN 覆蓋（0/1）")
    ap.add_argument("--pfc-dyn", type=int, default=None, help="USE_DYNAMIC_PFC_THRESHOLD 覆蓋（0/1）")
    ap.add_argument("--buffer", type=int, default=None, help="BUFFER_SIZE 覆蓋（例 16/32/64）")
    ap.add_argument("--payload", type=int, default=None, help="PACKET_PAYLOAD_SIZE 覆蓋（例 512/1000/1500）")
    # 其它
    ap.add_argument("--comm-group", default=None, help="傳給 --comm-group-configuration 的檔案；若未指定則不帶參數")
    ap.add_argument("--log-dir", default="runs", help="本次輸出根目錄（預設 runs）")
    ap.add_argument("--dry-run", action="store_true", help="只產生 patched 檔與指令，不執行")
    # 自動校準/輸出
    ap.add_argument("--calib-db", default="runs/calibration_all.csv", help="校準彙總 CSV（去重追加）")
    ap.add_argument("--no-autocalib", action="store_true", help="停用從 traces 自動校準 alpha_us")
    ap.add_argument("--trace-dir", default=None, help="覆蓋 traces 位置（預設為 ../data/chakra/pytorch_traces）")
    # [新增] 模型標籤
    ap.add_argument("--model-tag", type=str, default=None, help="模型標籤 (e.g., cifar10, resnet50)，用於選擇對應的 .et 與 trace")
    # [新增這行] 除錯開關
    ap.add_argument("--debug-ns3", action="store_true", help="啟用 NS-3 底層詳細 Log (用於偵測卡死)")

    args = ap.parse_args()

    script_path = Path(__file__).resolve()
    workload_dir = Path(args.workload).resolve()
    sys_json     = Path(args.system).resolve()
    net_cfg      = Path(args.network).resolve()
    remote_json  = Path(args.remote).resolve()
    ns3_bin      = Path(args.ns3_bin).resolve()
    for p in [workload_dir, sys_json, net_cfg, remote_json, ns3_bin]:
        if not p.exists():
            raise SystemExit(f"[ERR] 找不到：{p}")

    # 建立目錄
    stamp = time.strftime("%Y%m%d-%H%M%S%z")
    # world（若將擴張，命名用 N，但實際跑會在 expand 時回傳 actual_world）
    # [修改] 計算 world size 時傳入 tag
    world_for_name = args.virtual_world if args.virtual_world is not None else count_world_size(workload_dir, args.model_tag)

    # 解析邏輯維度（用於命名）
    topo_arg = args.topo
    if topo_arg.startswith("file:"):
        topo_json_path = Path(topo_arg.replace("file:", "")).resolve()
        logical_dims = load_logical_dims(topo_json_path)
    elif topo_arg.startswith("dims:"):
        logical_dims = [int(x) for x in topo_arg.replace("dims:", "").lower().split("x")]
    elif topo_arg.startswith("auto:"):
        mode = topo_arg.split(":", 1)[1]
        if   mode == "1d": logical_dims = [world_for_name]
        elif mode == "2d": logical_dims = list(squareish_2d(world_for_name))
        elif mode == "3d": logical_dims = list(cubeish_3d(world_for_name))
        else: raise SystemExit(f"[ERR] 不支援的 auto 模式：{mode}")
    else:
        topo_json_path = Path(topo_arg).resolve()
        logical_dims = load_logical_dims(topo_json_path)

    topo_desc = extract_topo_description(topo_arg, logical_dims)
    # [修改] 在 log 目錄名加入 tag
    tag_str = f"_{args.model_tag}" if args.model_tag else ""
    logroot = Path(args.log_dir).resolve() / f"{stamp}_ns3_{world_for_name}gpu{tag_str}_{topo_desc}"
    tmp_dir = logroot / "tmp"
    out_dir = logroot / "out"
    tmp_dir.mkdir(parents=True, exist_ok=True, mode=0o777)
    out_dir.mkdir(parents=True, exist_ok=True, mode=0o777)
    logroot.chmod(0o777)

    # 虛擬擴張（如有）
    # [修改] 傳入 tag
    workload_dir2, actual_world = expand_workload_virtual_if_needed(workload_dir, tmp_dir, args.virtual_world, args.model_tag)
    print(f"[INFO] workload={workload_dir2}  world_size={actual_world}  tag={args.model_tag}")

    # 邏輯拓樸檔
    if topo_arg.startswith("file:"):
        topo_json = Path(topo_arg.replace("file:", "")).resolve()
        assert_logical_dims_match_world(topo_json, actual_world)
    elif topo_arg.startswith("dims:") or topo_arg.startswith("auto:"):
        # 針對擴張後的 actual_world 重新檢查/產出
        if topo_arg.startswith("dims:") and math.prod(logical_dims) != actual_world:
            raise SystemExit(f"[ERR] dims 乘積 {math.prod(logical_dims)} != world {actual_world}")
        if topo_arg.startswith("auto:"):
            mode = topo_arg.split(":", 1)[1]
            if   mode == "1d": logical_dims = [actual_world]
            elif mode == "2d": logical_dims = list(squareish_2d(actual_world))
            elif mode == "3d": logical_dims = list(cubeish_3d(actual_world))
        topo_json = gen_topology_file(tmp_dir / "logical_topology.json", logical_dims)
    else:
        topo_json = Path(topo_arg).resolve()
        assert_logical_dims_match_world(topo_json, actual_world)

    # system / network 打補丁
    sys_patched = tmp_dir / "system.patched.json"
    patch_system_json(sys_json, sys_patched, args.coll_opt, args.lmbw)

    phys_topo = Path(args.phys_topo).resolve() if args.phys_topo else None
    if phys_topo is None:
        guess = guess_phys_topo(Path("configs/astra-sim/topos").resolve(), actual_world)
        if guess:
            phys_topo = guess
            print(f"[INFO] 推測實體拓樸：{phys_topo}")
        else:
            topo_line = next((ln for ln in net_cfg.read_text().splitlines()
                              if ln.strip().startswith("TOPOLOGY_FILE ")), None)
            if topo_line:
                base_topo_path = Path(topo_line.split(maxsplit=1)[1])
                nodes = infer_nodes_from_topo_filename(base_topo_path)
                if nodes and nodes != actual_world:
                    print(f"[WARN] baseline TOPOLOGY_FILE 似為 {nodes} 節點，但 workload world={actual_world}。建議用 --phys-topo 指定正確檔案。")

    net_patched = tmp_dir / "config.patched.txt"
    patch_network_cfg(net_cfg, net_patched,
                      topo_file=phys_topo, out_dir=out_dir,
                      qcn=args.qcn, pfc_dyn=args.pfc_dyn,
                      buffer_size=args.buffer, payload=args.payload)

    # Workload prefix 參數
    # [修改] 傳入 tag
    workload_arg, workload_desc = build_workload_arg(workload_dir2, args.model_tag)
    print(f"[INFO] workload-configuration 以 {workload_desc}")

    # ================= NS3 debug =================
    if args.debug_ns3:
        print("[DEBUG] 已啟用 NS-3 底層 Log (AstraSimNetwork=level_all)")
        # 讓 NS-3 吐出每一個執行的函式名稱，證明它活著
        os.environ["NS_LOG"] = "Node=level_info|prefix_func"
    # ======================================================

    # 組指令
    cmd = [
        str(ns3_bin),
        f"--workload-configuration={workload_arg}",
        f"--system-configuration={sys_patched.as_posix()}",
        f"--network-configuration={net_patched.as_posix()}",
        f"--remote-memory-configuration={remote_json.as_posix()}",
        f"--logical-topology-configuration={topo_json.as_posix()}",
    ]

    # ★ 若系統有 stdbuf，包一層讓 ns-3 改成 line-buffered，避免 PIPE 下完全不輸出
    stdbuf_path = shutil.which("stdbuf")
    if stdbuf_path is not None:
        cmd = [stdbuf_path, "-oL", "-eL", *cmd]
        print(f"[INFO] 檢測到 stdbuf，已使用 line-buffered 模式執行 ns-3：{' '.join(cmd)}")
    else:
        print("[WARN] 系統找不到 stdbuf，ns-3 在 PIPE 模式下可能不會即時輸出（這是 C stdout 緩衝機制的正常現象）")

    if args.comm_group and args.comm_group.lower() not in {"empty", "none"}:
        cg_path = Path(args.comm_group).expanduser().resolve()
        if not cg_path.exists():
            raise SystemExit(f"[ERR] --comm-group 指定檔案不存在：{cg_path}")
        cmd.append(f"--comm-group-configuration={cg_path.as_posix()}")

    (logroot / "command.txt").write_text(" ".join(cmd) + "\n", encoding="utf-8")
    print("[CMD]", " ".join(cmd))
    if args.dry_run:
        print(f"[DRY] patched files at {tmp_dir}")
        return

    # 執行前的工作負載驗證
    # [修改] 傳入 tag
    _validate_workload_integrity(workload_dir2, args.model_tag)

    stdout_path = (logroot / "stdout.log")
    print(f"[INFO] 開始執行 NS3 模擬... (輸出同步顯示)")
    print(f"[INFO] 日誌檔案: {stdout_path}")
    print(f"[INFO] 工作負載: {args.workload} (world_size={actual_world})")
    print(f"[INFO] 拓撲: {args.topo}")
    print(f"[INFO] 提示: 執行過程中可按 Ctrl+C 中斷")
    print("-" * 80)

    # 呼叫剛剛寫好的函式
    ret_code = run_simulation(cmd, stdout_path, logroot)

    # 檢查回傳碼，若失敗直接退出
    if ret_code != 0:
        raise SystemExit(f"[ERR] 模擬執行失敗 (Code {ret_code})")

    print(f"[INFO] 完成。stdout → {stdout_path}")
    print(f"[INFO] FCT/QLEN/PFC/TRACE 等輸出 → {out_dir}")

    # ----------------- 指標彙整：cycles × traces 校準 -----------------
    sim_cycles_step, sim_cycles_comm, sim_cycles_gpu = parse_astra_stdout_cycles(stdout_path)

    # 取 traces 位置（預設 ../data/chakra/pytorch_traces）
    trace_dir = Path(args.trace_dir).resolve() if args.trace_dir else _trace_dir_default(script_path)
    real_t_step_ms = None
    real_t_comm_ms = None
    real_t_net_comm_ms = None
    real_t_kernel_ms = None
    used_epoch = None

    # [修改] 只有在未禁用且 world=2 時才校準，並傳入 tag
    if not args.no_autocalib and count_world_size(workload_dir, args.model_tag) == 2:
        if trace_dir.exists():
            # [修改] 傳入 tag
            res = extract_real_metrics_from_traces(trace_dir, args.model_tag)
            if not res or res[0] is None:
                print(f"[WARN] 無法從 {trace_dir} 抓到 ProfilerStep，略過 auto calibration。")
            else:
                # Expecting either (step, net_comm, kernel, epoch) or older (step, comm, epoch)
                if len(res) == 4:
                    real_t_step_ms, real_t_net_comm_ms, real_t_kernel_ms, used_epoch = res
                    # maintain legacy name real_t_comm_ms as network-only for downstream code
                    real_t_comm_ms = real_t_net_comm_ms
                elif len(res) == 3:
                    # older callers returned (step, comm, epoch)
                    real_t_step_ms, real_t_comm_ms, used_epoch = res
                    real_t_net_comm_ms = real_t_comm_ms
                else:
                    try:
                        real_t_step_ms = res[0]
                        real_t_comm_ms = res[1] if len(res) > 1 else None
                        used_epoch = res[2] if len(res) > 2 else None
                        real_t_net_comm_ms = real_t_comm_ms
                    except Exception:
                        print(f"[WARN] extract_real_metrics_from_traces() 回傳非預期格式: {res}")
        else:
            print(f"[WARN] {trace_dir} 不存在，略過 auto calibration。")

    alpha_us = None
    sim_t_step_ms = None
    sim_t_comm_ms = None

    # 若有 cycles 與 real_t_step_ms，計算 alpha_us 與對應的 sim_t_step_ms
    if sim_cycles_step is not None:
        if real_t_step_ms is not None:
            alpha_us = (real_t_step_ms * 1000.0) / max(1, sim_cycles_step)
            sim_t_step_ms = sim_cycles_step * (alpha_us / 1000.0)

            # [新增] System-Aware Calibration 檢查
            if 0.95 <= alpha_us <= 1.05:
                print(f"[INFO] Alpha={alpha_us:.4f} 接近 1.0，表示 System-Aware Calibration (計算時間攤提) 有效。")
        else:
            # 無實測（例如 N>2 虛擬擴張），sim_t_step_ms 與 alpha 只能留空或沿用前次（此處留空，只記 cycles）
            pass

    # 額外計算 comm 尺度的 alpha（alpha_comm_us），以便診斷 sim_comm 與 real_comm 是否共享相同尺度
    alpha_comm_us = None
    if sim_cycles_comm is not None and real_t_comm_ms is not None:
        # real_t_comm_ms (ms/step) -> microsecond per sim cycle
        alpha_comm_us = (real_t_comm_ms * 1000.0) / max(1, sim_cycles_comm)

    # 額外計算 gpu 尺度的 alpha（alpha_gpu_us），以便診斷 sim_gpu 與 real_kernel 是否匹配
    alpha_gpu_us = None
    if sim_cycles_gpu is not None and real_t_kernel_ms is not None:
        alpha_gpu_us = (real_t_kernel_ms * 1000.0) / max(1, sim_cycles_gpu)

    # 使用通用 alpha 計算 sim_t_comm_ms
    sim_t_comm_ms = None
    if sim_cycles_comm is not None and alpha_us is not None:
        sim_t_comm_ms = sim_cycles_comm * (alpha_us / 1000.0)

    # 若存在 comm-specific alpha，計算使用 comm alpha 的 sim comm 時間與對應誤差（便於比較）
    sim_t_comm_ms_comm = None
    rel_err_comm_comm = None
    if sim_cycles_comm is not None and alpha_comm_us is not None:
        sim_t_comm_ms_comm = sim_cycles_comm * (alpha_comm_us / 1000.0)
        # [修改] 加入 > 0.01ms (10us) 的門檻
        # 意義：如果真實通訊時間小於 10us，我們視為雜訊/Overhead，不計算誤差，避免 CIFAR-10 出現巨大誤差數值
        if real_t_comm_ms is not None and sim_t_comm_ms_comm is not None and real_t_comm_ms > 0.01:
            rel_err_comm_comm = (abs(sim_t_comm_ms_comm - real_t_comm_ms) / real_t_comm_ms)

    # ---------- A-保護：Comm==Wall 與 ET 是否有 Compute ----------
    flags: list[str] = []
    # [修改] 傳入 tag
    has_compute_nodes = detect_compute_nodes_in_et(workload_dir2, args.model_tag)
    comm_equals_wall = (sim_cycles_step is not None and sim_cycles_comm is not None and sim_cycles_comm == sim_cycles_step)

    if comm_equals_wall:
        if not has_compute_nodes:
            # A-保護：ET 無 Compute 節點導致 exposed_communication_cycles == wall_cycles
            # 此時通訊曝光時間不可信，需要抑制
            print("[NOTE] sim_cycles_comm == sim_cycles_step 且 ET 無 Compute 節點 → 抑制 sim_t_comm_ms（A-保護）")
            sim_t_comm_ms = None  # 抑制不可信的通訊時間
            flags.append("comm_equals_wall_no_compute")
        else:
            # ET 有 Compute 但 comm==wall：可能 feeder 版本未處理 compute，或 compute 被完全遮蔽
            print("[WARN] sim_cycles_comm == sim_cycles_step，但 ET 有 Compute 節點 → 請檢查 feeder 版本/compute 欄位鍵名")
            # ★feeder 未消化 compute 的常見情況；同樣抑制不可信的通訊時間
            flags.append("comm_equals_wall")

    # 相對誤差（只有在 real_* 與對應 sim_* 存在時才有）
    rel_err_step = (abs(sim_t_step_ms - real_t_step_ms) / real_t_step_ms) if (sim_t_step_ms is not None and real_t_step_ms) else None
    rel_err_comm = (abs(sim_t_comm_ms - real_t_comm_ms) / real_t_comm_ms) if (sim_t_comm_ms is not None and real_t_comm_ms) else None

    # 匯出 metrics
    row = {
        # [修改] 若有 real_t_step_ms 且是 2-GPU 才視為 calibration
        "mode": "calibrate" if (count_world_size(workload_dir, args.model_tag) == 2 and real_t_step_ms is not None) else "simulate",
        # [新增] 紀錄 tag
        "tag": args.model_tag if args.model_tag else "",
        "calib_id": used_epoch if used_epoch is not None else "",
        "world": actual_world,
        "logical_dims": "x".join(map(str, logical_dims)),
        "topo_desc": extract_topo_description(topo_arg, logical_dims),
        "qcn": args.qcn if args.qcn is not None else "",
        "pfc_dyn": args.pfc_dyn if args.pfc_dyn is not None else "",
        "buffer": args.buffer if args.buffer is not None else "",
        "payload": args.payload if args.payload is not None else "",
        "coll_opt": args.coll_opt if args.coll_opt is not None else "",
        "lmbw": args.lmbw if args.lmbw is not None else "",
        "alpha_us": f"{alpha_us:.6f}" if alpha_us is not None else "",
        "alpha_comm_us": f"{alpha_comm_us:.6f}" if alpha_comm_us is not None else "",
        "alpha_gpu_us": f"{alpha_gpu_us:.6f}" if alpha_gpu_us is not None else "",
        "sim_cycles_step": sim_cycles_step if sim_cycles_step is not None else "",
        "sim_cycles_comm": sim_cycles_comm if sim_cycles_comm is not None else "",
        "sim_cycles_gpu": sim_cycles_gpu if sim_cycles_gpu is not None else "",
        "sim_t_step_ms": f"{sim_t_step_ms:.6f}" if sim_t_step_ms is not None else "",
        "sim_t_comm_ms": f"{sim_t_comm_ms:.6f}" if sim_t_comm_ms is not None else "",
        "sim_t_comm_ms_comm": f"{sim_t_comm_ms_comm:.6f}" if sim_t_comm_ms_comm is not None else "",
        "real_t_step_ms": f"{real_t_step_ms:.6f}" if real_t_step_ms is not None else "",
        "real_t_comm_ms": f"{real_t_comm_ms:.6f}" if real_t_comm_ms is not None else "",
        "real_t_net_comm_ms": f"{real_t_net_comm_ms:.6f}" if real_t_net_comm_ms is not None else "",
        "real_t_kernel_ms": f"{real_t_kernel_ms:.6f}" if real_t_kernel_ms is not None else "",
        "alpha_comm_us": f"{alpha_comm_us:.6f}" if alpha_comm_us is not None else "",
        "alpha_gpu_us": f"{alpha_gpu_us:.6f}" if alpha_gpu_us is not None else "",
        "run_dir": str(logroot),
        "rel_err_step": f"{rel_err_step:.6f}" if rel_err_step is not None else "",
        "rel_err_comm": f"{rel_err_comm:.6f}" if rel_err_comm is not None else "",
        "rel_err_comm_comm": f"{rel_err_comm_comm:.6f}" if rel_err_comm_comm is not None else "",
        "flags": "|".join(flags) if flags else "",
    }
    export_metrics(out_dir, row)

    # 若是 2-GPU 且有實測，就把校準資訊追加到共用的 runs/calibration_all.csv（去重）
    if row["mode"] == "calibrate" and alpha_us is not None:
        _append_calibration_db(Path(args.calib_db).resolve(), row)

if __name__ == "__main__":
    main()
