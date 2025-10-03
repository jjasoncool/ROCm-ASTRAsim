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
KINETO_COMM_HINTS = ("nccl", "rccl", "all_reduce", "allreduce", "all_gather", "reduce_scatter", "alltoall")

# Calibration CSV 欄位定義
CALIB_FIELDS = [
    'mode', 'tag', 'calib_id', 'world', 'logical_dims', 'topo_desc',
    'qcn', 'pfc_dyn', 'buffer', 'payload', 'coll_opt', 'lmbw',
    'alpha_us', 'sim_cycles_step', 'sim_cycles_comm', 'sim_cycles_gpu', 'sim_t_step_ms', 'sim_t_comm_ms',
    'real_t_step_ms', 'real_t_comm_ms', 'run_dir', 'rel_err_step', 'rel_err_comm',
    'flags'  # 執行狀態標記（例如 comm_equals_wall_no_compute）
]

# -------------------- World size / logical-dims --------------------

def list_et_rank_files(workload_dir: Path) -> dict[int, Path]:
    files = {}
    for p in workload_dir.glob("*.et"):
        m = ET_PAT.match(p.name)
        if m:
            files[int(m.group("rank"))] = p
    return dict(sorted(files.items()))

def count_world_size(workload_dir: Path) -> int:
    et_map = list_et_rank_files(workload_dir)
    if et_map:
        return len(et_map)
    jfiles = list(workload_dir.glob("et_rank_*.json"))
    if workload_dir.joinpath("manifest.json").exists() and jfiles:
        return len(jfiles)
    raise SystemExit(f"[ERR] {workload_dir} 內找不到 .et 或 et_rank_*.json")

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
        for ln in lines_list:
            if pat.match(ln):
                if value is not None and not replaced:
                    new_lines.append(f"{key} {Path(value).resolve().as_posix()}")
                    replaced = True
            else:
                new_lines.append(ln)
        if (value is not None) and (not replaced):
            new_lines.append(f"{key} {Path(value).resolve().as_posix()}")
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

def expand_workload_virtual_if_needed(workload_src: Path, tmp_dir: Path, virtual_world: int | None) -> tuple[Path, int]:
    if virtual_world is None:
        return workload_src, count_world_size(workload_src)
    if virtual_world < 2:
        raise SystemExit("[ERR] --virtual-world 至少需要 2。")
    et_map = list_et_rank_files(workload_src)
    if not et_map:
        raise SystemExit("[ERR] --virtual-world 目前僅支援 .et 工作負載（manifest+JSON 請先轉 .et）。")
    M = len(et_map)
    if M < 2:
        raise SystemExit("[ERR] 來源 .et 檔數需至少 2（M>=2）才能合理推導縮放。")
    N = int(virtual_world)
    if N == M:
        print(f"[INFO] virtual-world={N} 與來源相同；不做擴張。")
        return workload_src, M
    any_path = next(iter(et_map.values()))
    m = ET_PAT.match(any_path.name)
    prefix = m.group("prefix") if m else "et"
    scale = (M * (N - 1)) / (N * (M - 1))
    print(f"[INFO] 以來源 M={M} 擴張到 N={N}；通訊 bytes 縮放係數 scale={scale:.6f}")
    out_dir = tmp_dir / f"workload_{N}"
    for r in range(N):
        src_rank = r % M
        src_path = et_map[src_rank]
        meta, nodes = _read_all_nodes(src_path)
        for node in nodes:
            _scale_comm_size_inplace(node, scale)
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

def build_workload_arg(workload_dir: Path) -> tuple[str, str]:
    et_map = list_et_rank_files(workload_dir)
    if et_map:
        any_path = next(iter(et_map.values()))
        m = ET_PAT.match(any_path.name)
        prefix = m.group("prefix") if m else "et"
        prefix_path = (workload_dir / prefix).as_posix()
        return prefix_path, f"prefix={prefix_path}"
    else:
        return workload_dir.as_posix(), f"dir={workload_dir.as_posix()}"

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

def _list_epoch_pairs(trace_dir: Path) -> list[tuple[Path, Path, int]]:
    """回傳 [(rank0_json, rank1_json, epoch_id), ...]，只收兩邊都有的 epoch。"""
    r0 = {int(re.search(r"epoch_(\d+)\.json$", p.name).group(1)): p
          for p in trace_dir.glob("trace_rank_0_epoch_*.json")}
    r1 = {int(re.search(r"epoch_(\d+)\.json$", p.name).group(1)): p
          for p in trace_dir.glob("trace_rank_1_epoch_*.json")}
    common = sorted(set(r0.keys()) & set(r1.keys()))
    return [(r0[e], r1[e], e) for e in common]

def _dur_units_to_ms(trace_obj: dict, dur_val: float) -> float:
    # 若 trace 宣告 displayTimeUnit == "ms"，多半 dur 就是毫秒；否則多半是微秒
    unit = str(trace_obj.get("displayTimeUnit", "")).lower()
    if unit == "ms":
        return float(dur_val)
    # 預設視為微秒（Kineto 常見），轉成毫秒
    return float(dur_val) / 1000.0

from pathlib import Path

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
        d_ms = _dur_units_to_ms(obj, ev.get("dur", 0.0))

        # 判定一步的界線：PyTorch Profiler 會發出 "ProfilerStep#<n>"
        if KINETO_STEP_PAT.match(str(ev.get("name", ""))):
            steps_ms.append(d_ms)

        # 判定通訊事件：名稱或分類包含 nccl/rccl/all_reduce/all_gather 等關鍵字
        # （RCCL 是 ROCm 上的 NCCL 對應實作）
        if any(h in name for h in KINETO_COMM_HINTS) or any(h in cat for h in KINETO_COMM_HINTS):
            comm_sum_ms += d_ms
            any_comm = True

    # ★關鍵修正：回傳「每步通訊」（ms/step）而不是「整個檔案總和」
    # 這樣 extract_real_metrics_from_traces() 取得的 real_t_comm_ms 就是 per-step 的量，
    # 才能與 ASTRA‑sim 的 sim_t_comm_ms（per-step 的 exposed communication）一致比較。
    comm_ms_per_step = (comm_sum_ms / max(1, len(steps_ms))) if (any_comm and steps_ms) else None
    return steps_ms, comm_ms_per_step

def extract_real_metrics_from_traces(trace_dir: Path) -> tuple[float | None, float | None, int | None]:
    """
    掃描兩張卡的 trace，鎖定「同一 epoch」的 pair：
      • real_t_step_ms：取兩張卡對應 epoch 的 step durations 的「中位數再取中位數」
      • real_t_comm_ms：若抓得到通訊事件，兩卡加總後取一半（粗估每卡），再取中位數
    回傳: (real_t_step_ms, real_t_comm_ms, used_epoch)
    """
    pairs = _list_epoch_pairs(trace_dir)
    if not pairs:
        print(f"[WARN] {trace_dir} 找不到 rank0/1 成對的 trace。")
        return None, None, None

    step_medians = []
    comm_per_epoch = []
    used_epoch = None

    for p0, p1, e in pairs:
        s0, c0 = _extract_step_and_comm_ms(p0)
        s1, c1 = _extract_step_and_comm_ms(p1)
        if s0 and s1:
            step_medians.append(median([median(s0), median(s1)]))
            used_epoch = e if used_epoch is None else used_epoch
        # 通訊：兩卡的總和 / 2 當作每卡平均
        comm_vals = []
        if c0 is not None: comm_vals.append(c0)
        if c1 is not None: comm_vals.append(c1)
        if comm_vals:
            comm_per_epoch.append(sum(comm_vals) / max(1, len(comm_vals)))
    real_t_step_ms = median(step_medians) if step_medians else None
    real_t_comm_ms = median(comm_per_epoch) if comm_per_epoch else None
    return real_t_step_ms, real_t_comm_ms, used_epoch

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

def detect_compute_nodes_in_et(workload_dir: Path, sample_limit: int = 2) -> bool:
    """
    掃描工作負載目錄下的 .et，判定是否存在「Compute 節點」：
      • Node.type == COMP_NODE/COMPUTE_NODE（若 enum 有提供）
      • 或節點 attr 含 compute_cycles/exec_cycles/cycles/duration_cycles 任一鍵
    為降低成本，僅抽樣前 sample_limit 份 .et。
    """
    et_map = list_et_rank_files(workload_dir)
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
    for old in rows:
        if _row_key_for_dedup(old) == new_key:
            same = all(str(old.get(k, "")) == str(row.get(k, "")) for k in
                       ["alpha_us","sim_t_step_ms","sim_t_comm_ms","real_t_step_ms","real_t_comm_ms"])
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
    world_for_name = args.virtual_world if args.virtual_world is not None else count_world_size(workload_dir)

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
    logroot = Path(args.log_dir).resolve() / f"{stamp}_ns3_{world_for_name}gpu_{topo_desc}"
    tmp_dir = logroot / "tmp"
    out_dir = logroot / "out"
    tmp_dir.mkdir(parents=True, exist_ok=True, mode=0o777)
    out_dir.mkdir(parents=True, exist_ok=True, mode=0o777)
    logroot.chmod(0o777)

    # 虛擬擴張（如有）
    workload_dir2, actual_world = expand_workload_virtual_if_needed(workload_dir, tmp_dir, args.virtual_world)
    print(f"[INFO] workload={workload_dir2}  world_size={actual_world}")

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
    workload_arg, workload_desc = build_workload_arg(workload_dir2)
    print(f"[INFO] workload-configuration 以 {workload_desc}")

    # 組指令
    cmd = [
        str(ns3_bin),
        f"--workload-configuration={workload_arg}",
        f"--system-configuration={sys_patched.as_posix()}",
        f"--network-configuration={net_patched.as_posix()}",
        f"--remote-memory-configuration={remote_json.as_posix()}",
        f"--logical-topology-configuration={topo_json.as_posix()}",
    ]
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

    # 執行 (同時輸出到終端和檔案)
    stdout_path = (logroot / "stdout.log")
    print(f"[INFO] 開始執行 NS3 模擬... (輸出同步顯示)")
    print(f"[INFO] 日誌檔案: {stdout_path}")
    print(f"[INFO] 工作負載: {args.workload} (world_size={actual_world})")
    print(f"[INFO] 拓撲: {args.topo}")
    print(f"[INFO] 提示: 執行過程中可按 Ctrl+C 中斷")
    print("-" * 80)

    with stdout_path.open("w", encoding="utf-8") as lf:
        try:
            # 使用 Popen 以便即時顯示輸出
            process = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                                     universal_newlines=True, bufsize=1)

            start_time = time.time()
            last_output_time = start_time
            line_count = 0
            qp_enabled_seen = False
            simulation_started = False

            for line in process.stdout:
                # 同時寫入檔案和顯示在終端
                lf.write(line)
                lf.flush()

                current_time = time.time()
                line_count += 1

                # 顯示帶時間戳的輸出
                elapsed = current_time - start_time
                print(f"[{elapsed:6.1f}s] {line.rstrip()}")

                # 檢測 NS3 執行階段
                line_lower = line.lower()
                if "qp is enabled" in line_lower:
                    qp_enabled_seen = True
                    print(f"[INFO] NS3 網路模擬開始執行（可能進入靜默計算階段）...")
                elif qp_enabled_seen and ("maxrtt" in line_lower or "finished" in line_lower):
                    simulation_started = True

                # 智能無輸出警告 - 考慮 NS3 執行階段
                silence_duration = current_time - last_output_time
                if qp_enabled_seen and not simulation_started:
                    # NS3 模擬階段：延長警告時間到 120 秒
                    warning_threshold = 120
                    if silence_duration > warning_threshold:
                        print(f"[WARN] NS3 模擬已靜默 {silence_duration:.1f} 秒（正常現象，網路模擬進行中）")
                        print(f"[INFO] 如確實卡死可按 Ctrl+C 中斷")
                        last_output_time = current_time  # 重置以避免重複警告
                else:
                    # 初始化或結果階段：30 秒警告
                    warning_threshold = 30
                    if silence_duration > warning_threshold:
                        print(f"[WARN] 已 {silence_duration:.1f} 秒無新輸出，可能卡死...")
                        print(f"[INFO] 如需中斷請按 Ctrl+C")
                        last_output_time = current_time  # 重置以避免重複警告

                last_output_time = current_time

            # 等待進程完成
            process.wait()
            if process.returncode != 0:
                raise subprocess.CalledProcessError(process.returncode, cmd)

        except subprocess.CalledProcessError as e:
            print("-" * 80)
            print(f"[ERR] NS3 執行失敗，退出碼: {e.returncode}")
            shutil.rmtree(logroot)
            print(f"[INFO] 執行失敗，已刪除 {logroot}")
            raise SystemExit(f"[ERR] ns-3 後端退出碼 {e.returncode}")
        except KeyboardInterrupt:
            print("\n[INFO] 用戶中斷執行")
            if 'process' in locals():
                process.terminate()
                time.sleep(2)  # 給進程時間終止
                if process.poll() is None:  # 如果還沒終止
                    print("[INFO] 強制終止進程...")
                    process.kill()
            raise SystemExit("[INFO] 執行被中斷")

    print(f"[INFO] 完成。stdout → {stdout_path}")
    print(f"[INFO] FCT/QLEN/PFC/TRACE 等輸出 → {out_dir}")

    # ----------------- 指標彙整：cycles × traces 校準 -----------------
    sim_cycles_step, sim_cycles_comm, sim_cycles_gpu = parse_astra_stdout_cycles(stdout_path)

    # 取 traces 位置（預設 ../data/chakra/pytorch_traces）
    trace_dir = Path(args.trace_dir).resolve() if args.trace_dir else _trace_dir_default(script_path)
    real_t_step_ms = None
    real_t_comm_ms = None
    used_epoch = None

    if not args.no_autocalib and actual_world == 2:
        if trace_dir.exists():
            real_t_step_ms, real_t_comm_ms, used_epoch = extract_real_metrics_from_traces(trace_dir)
            if real_t_step_ms is None:
                print(f"[WARN] 無法從 {trace_dir} 抓到 ProfilerStep，略過 auto calibration。")
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
        else:
            # 無實測（例如 N>2 虛擬擴張），sim_t_step_ms 與 alpha 只能留空或沿用前次（此處留空，只記 cycles）
            pass
    if sim_cycles_comm is not None and alpha_us is not None:
        sim_t_comm_ms = sim_cycles_comm * (alpha_us / 1000.0)

    # ---------- A-保護：Comm==Wall 與 ET 是否有 Compute ----------
    flags: list[str] = []
    has_compute_nodes = detect_compute_nodes_in_et(workload_dir2)
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
        "mode": "calibrate" if (actual_world == 2 and real_t_step_ms is not None) else "simulate",
        "tag": "",
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
        "sim_cycles_step": sim_cycles_step if sim_cycles_step is not None else "",
        "sim_cycles_comm": sim_cycles_comm if sim_cycles_comm is not None else "",
        "sim_cycles_gpu": sim_cycles_gpu if sim_cycles_gpu is not None else "",
        "sim_t_step_ms": f"{sim_t_step_ms:.6f}" if sim_t_step_ms is not None else "",
        # A-保護：若上面抑制了 sim_t_comm_ms，這裡自然留空
        "sim_t_comm_ms": f"{sim_t_comm_ms:.6f}" if sim_t_comm_ms is not None else "",
        "real_t_step_ms": f"{real_t_step_ms:.6f}" if real_t_step_ms is not None else "",
        "real_t_comm_ms": f"{real_t_comm_ms:.6f}" if real_t_comm_ms is not None else "",
        "run_dir": str(logroot),
        "rel_err_step": f"{rel_err_step:.6f}" if rel_err_step is not None else "",
        "rel_err_comm": f"{rel_err_comm:.6f}" if rel_err_comm is not None else "",
        "flags": "|".join(flags) if flags else "",
    }
    export_metrics(out_dir, row)

    # 若是 2-GPU 且有實測，就把校準資訊追加到共用的 runs/calibration_all.csv（去重）
    if row["mode"] == "calibrate" and alpha_us is not None:
        _append_calibration_db(Path(args.calib_db).resolve(), row)

if __name__ == "__main__":
    main()
