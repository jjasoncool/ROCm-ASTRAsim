#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Run ASTRA-sim ns-3 with a Chakra workload (.et or manifest+JSON),
auto-generate/validate logical topology, patch system/network configs,
and (optionally) virtual-expand your measured ETs to larger world sizes
without you managing hundreds of files.

輸出：runs/<timestamp>_ns3_run/{stdout.log, command.txt, tmp/*, out/*}
"""

import argparse, json, math, os, re, subprocess, sys, time, shutil
from pathlib import Path

# Chakra protobuf I/O
from chakra.src.third_party.utils.protolib import encodeMessage as encode_message, decodeMessage as decode_message
from chakra.schema.protobuf.et_def_pb2 import (
    GlobalMetadata, Node, AttributeProto,
    ALL_REDUCE, ALL_GATHER, REDUCE_SCATTER, ALL_TO_ALL
)

# 接受 "xxx.0.et / xxx.1.et ..." 這種帶數字索引的檔名
ET_PAT = re.compile(r"^(?P<prefix>.*)\.(?P<rank>\d+)\.et$")

# ---------- World size 與 logical-dims 工具 ----------

def list_et_rank_files(workload_dir: Path) -> dict[int, Path]:
    """取得 {rank: path} 映射（只收 .et 檔且檔名符合 prefix.{rank}.et）。"""
    files = {}
    for p in workload_dir.glob("*.et"):
        m = ET_PAT.match(p.name)
        if m:
            files[int(m.group("rank"))] = p
    return dict(sorted(files.items()))

def count_world_size(workload_dir: Path) -> int:
    """
    推斷 world size：
      1) 優先數 .et 檔數（檔名需帶數字索引）
      2) 否則數 et_rank_*.json（需有 manifest.json）
    找不到即報錯。
    """
    et_map = list_et_rank_files(workload_dir)
    if et_map:
        return len(et_map)
    jfiles = list(workload_dir.glob("et_rank_*.json"))
    if workload_dir.joinpath("manifest.json").exists() and jfiles:
        return len(jfiles)
    raise SystemExit(f"[ERR] {workload_dir} 內找不到 .et 或 et_rank_*.json")

def squareish_2d(n: int) -> tuple[int, int]:
    """把 n 拆成兩因數，盡量接近正方（例：8 -> (2,4)；質數 -> (1,n)）。"""
    if n < 1:
        raise ValueError("n must be >= 1")
    a = int(math.sqrt(n))
    while a > 1 and n % a != 0:
        a -= 1
    return (a, n // a) if a > 1 else (1, n)

def cubeish_3d(n: int) -> tuple[int, int, int]:
    """把 n 拆成三因數，盡量接近立方（例：8 -> (2,2,2)）。"""
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
    """寫出 ASTRA-sim 的「邏輯拓樸」JSON：{"logical-dims":["2","4",...]}。"""
    out_path.parent.mkdir(parents=True, exist_ok=True, mode=0o777)
    out_path.parent.chmod(0o777)  # 設置正確權限
    out_path.write_text(json.dumps({"logical-dims": [str(d) for d in dims]}, indent=2),
                        encoding="utf-8")
    return out_path

def load_logical_dims(json_path: Path) -> list[int]:
    """讀 logical-topology JSON 並回傳維度列表（轉成 int）。"""
    obj = json.loads(json_path.read_text(encoding="utf-8"))
    dims = [int(x) for x in obj.get("logical-dims", [])]
    if not dims:
        raise ValueError(f"{json_path} 缺少 'logical-dims'")
    return dims

def assert_logical_dims_match_world(json_path: Path, world: int) -> None:
    """驗證 logical-dims 的乘積是否等於 world，不符直接終止。"""
    dims = load_logical_dims(json_path)
    prod = math.prod(dims)
    if prod != world:
        raise SystemExit(f"[ERR] 邏輯拓樸 {json_path} 維度乘積={prod} ≠ world={world}")

# ---------- 實體拓樸與 baseline 檢查 ----------

def infer_nodes_from_topo_filename(path: Path) -> int | None:
    """從 '8_nodes_1_switch_topology.txt' 這類檔名推斷節點數（找不到回 None）。"""
    m = re.search(r"(\d+)_nodes?_", path.name)
    return int(m.group(1)) if m else None

def guess_phys_topo(topos_dir: Path, world: int) -> Path | None:
    """僅在找到 '<world>_nodes_*_topology.txt' 時回檔案；否則回 None（不要亂 fallback）。"""
    cands = list(topos_dir.glob(f"{world}_nodes_*_topology.txt"))
    return cands[0].resolve() if cands else None

# ---------- system.json / ns-3 config.txt 打補丁 ----------

def patch_system_json(src: Path, out: Path, coll_opt: str | None, lmbw: int | None) -> Path:
    """
    以 baseline system.json (src) 為底，產生「本次執行要用」的 patched 檔 (out)。
      - coll_opt → 覆蓋 "collective-optimization"（例：localBWAware / none）
      - lmbw     → 覆蓋 "local-mem-bw"（校準 compute/comm 比例）
    不改原檔，只寫副本，便於重現。
    """
    obj = json.loads(src.read_text(encoding="utf-8"))
    if coll_opt is not None:
        obj["collective-optimization"] = coll_opt
    if lmbw is not None:
        obj["local-mem-bw"] = lmbw
    out.write_text(json.dumps(obj, indent=2), encoding="utf-8")
    return out

def patch_network_cfg(src: Path, out: Path, *,
                      topo_file: Path | None,     # ns-3 的 TOPOLOGY_FILE（實體拓樸）
                      out_dir: Path,              # 本次 run 的輸出目錄（fct/qlen/pfc/trace...）
                      qcn: int | None = None,     # 覆蓋 ENABLE_QCN（0/1）
                      pfc_dyn: int | None = None, # 覆蓋 USE_DYNAMIC_PFC_THRESHOLD（0/1）
                      buffer_size: int | None = None,   # 覆蓋 BUFFER_SIZE
                      payload: int | None = None) -> Path:  # 覆蓋 PACKET_PAYLOAD_SIZE
    """
    讀 baseline ns-3 config.txt，輸出 patched 副本（與工作目錄無關）：
      1) TOPOLOGY_FILE/輸出檔一律改為絕對路徑；
      2) FLOW_FILE：移除 baseline 中所有既有設定，檔案最前面強制插入
         'FLOW_FILE <out_dir>/flow.txt'，並預先 touch 空檔，避免開檔失敗；
      3) 覆蓋 QCN/PFC_DYN/BUFFER/PAYLOAD（若指定）。
    """
    out_dir.mkdir(parents=True, exist_ok=True, mode=0o777)
    out_dir.chmod(0o777)  # 確保權限正確設置
    lines = src.read_text(encoding="utf-8").splitlines()

    # 建立 placeholder flow.txt（輸入檔，但內容不使用；需能被打開）
    flow_placeholder = (out_dir / "flow.txt").resolve()
    flow_placeholder.touch(exist_ok=True)

    # 小工具：以 regex 方式 robust 覆蓋 KEY，支援 "KEY value" / "KEY=value" / 前導空白
    import re
    def replace_key(lines_list, key, value):
        pat = re.compile(rf'^\s*{re.escape(key)}(\s+|=).*$', re.IGNORECASE)
        replaced = False
        new_lines = []
        for ln in lines_list:
            if pat.match(ln):
                if value is not None and not replaced:
                    new_lines.append(f"{key} {Path(value).resolve().as_posix()}")
                    replaced = True
                # 丟棄舊行（已用新值取代）
            else:
                new_lines.append(ln)
        if (value is not None) and (not replaced):
            # 若原本沒有此 KEY，於檔尾補上一行
            new_lines.append(f"{key} {Path(value).resolve().as_posix()}")
        return new_lines

    # 先把所有 FLOW_FILE 行剔除，待會插入在檔案最前面
    lines_wo_flow = []
    flow_pat = re.compile(r'^\s*FLOW_FILE(\s+|=)', re.IGNORECASE)
    for ln in lines:
        if not flow_pat.match(ln):
            lines_wo_flow.append(ln)

    # 覆蓋網路層參數（若指定）
    def set_optional_key(lines_list, key, value):
        return replace_key(lines_list, key, value) if value is not None else lines_list

    lines2 = lines_wo_flow
    lines2 = set_optional_key(lines2, "ENABLE_QCN", qcn)
    lines2 = set_optional_key(lines2, "USE_DYNAMIC_PFC_THRESHOLD", pfc_dyn)
    lines2 = set_optional_key(lines2, "BUFFER_SIZE", buffer_size)
    lines2 = set_optional_key(lines2, "PACKET_PAYLOAD_SIZE", payload)

    # 覆蓋 TOPOLOGY_FILE（若提供）
    if topo_file is not None:
        lines2 = replace_key(lines2, "TOPOLOGY_FILE", topo_file)

    # 覆蓋所有「輸出」檔為絕對路徑（寫到本次 run 的 out/）
    out_targets = {
        "TRACE_FILE":        out_dir / "trace.txt",
        "TRACE_OUTPUT_FILE": out_dir / "mix.tr",
        "FCT_OUTPUT_FILE":   out_dir / "fct.txt",
        "PFC_OUTPUT_FILE":   out_dir / "pfc.txt",
        "QLEN_MON_FILE":     out_dir / "qlen.txt",
    }
    for k, v in out_targets.items():
        v.parent.mkdir(parents=True, exist_ok=True, mode=0o777)  # 確保 out_dir 存在
        v.parent.chmod(0o777)  # 設置正確權限
        if not v.exists():
            v.touch()  # 預先建立輸出檔，避免開啟失敗
        lines2 = replace_key(lines2, k, v)

    # ★ 關鍵：把 FLOW_FILE 放在檔案最前面（避免某些分支取第一個匹配值）
    new_lines = [f"FLOW_FILE {flow_placeholder.as_posix()}"]
    new_lines.extend(lines2)

    out.write_text("\n".join(new_lines) + "\n", encoding="utf-8")
    return out

# ---------- 以「實測 .et」為模板的虛擬擴張（trace-preserving） ----------

COMM_TYPES = {ALL_REDUCE, ALL_GATHER, REDUCE_SCATTER, ALL_TO_ALL}

def _read_all_nodes(et_path: Path) -> tuple[GlobalMetadata, list[Node]]:
    """穩健地把一個 .et 檔的 GlobalMetadata + 全部 Node 讀進來。"""
    nodes = []
    with et_path.open("rb") as f:
        meta = GlobalMetadata()
        decode_message(f, meta)
        # 以檔案大小作哨兵，避免在 EOF 附近卡死
        size = et_path.stat().st_size
        while f.tell() < size:
            pos0 = f.tell()
            node = Node()
            try:
                decode_message(f, node)
            except Exception:
                break
            if f.tell() <= pos0:  # offset 未前進 → 損毀或 EOF
                break
            nodes.append(node)
    return meta, nodes

def _scale_comm_size_inplace(node: Node, scale: float) -> None:
    """若 node 是通訊事件（看 comm_type），就把 comm_size 乘以 scale（四捨五入到整數）。"""
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
    out_path.parent.chmod(0o777)  # 設置正確權限
    with out_path.open("wb") as f:
        encode_message(f, meta)
        for n in nodes:
            encode_message(f, n)

def expand_workload_virtual_if_needed(workload_src: Path, tmp_dir: Path, virtual_world: int | None) -> tuple[Path, int]:
    """
    若指定 --virtual-world=N，則以 workload_src 中的 .et 作模板，在 tmp_dir 生成 N 份 .et：
      - 來源 world = M（必須 M>=2），目標 world = N（N>=2）
      - 對所有通訊事件（AR/RS/AG/All-to-All）依比例縮放 comm_size：
          scale = (M * (N - 1)) / (N * (M - 1))
        （推導：以 per-GPU bytes_M / bytes_N 的理論式相除，2 倍係數會抵銷）
      - 其餘事件（COMPUTE/MEM）原樣複製，保留你實測的時間形狀
      - 檔名輸出為 {prefix}.{rank}.et（延用來源 prefix）
    回傳：(新工作負載目錄, 實際 world)
    """
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

    # 來源 prefix（沿用第一個 .et 的 prefix）
    any_path = next(iter(et_map.values()))
    m = ET_PAT.match(any_path.name)
    prefix = m.group("prefix") if m else "et"

    # 縮放比例（適用於 AR / RS / AG / A2A）
    scale = (M * (N - 1)) / (N * (M - 1))
    print(f"[INFO] 以來源 M={M} 擴張到 N={N}；通訊 bytes 縮放係數 scale={scale:.6f}")

    # 生成暫存工作負載
    out_dir = tmp_dir / f"workload_{N}"
    for r in range(N):
        src_rank = r % M
        src_path = et_map[src_rank]
        meta, nodes = _read_all_nodes(src_path)
        # 只縮放通訊事件的 comm_size；compute 原樣保留（忠實你的實測）
        for node in nodes:
            _scale_comm_size_inplace(node, scale)
        dst_path = out_dir / f"{prefix}.{r}.et"
        _write_et(meta, nodes, dst_path)

    print(f"[INFO] 產生虛擬工作負載：{out_dir} （共 {N} 份 .et）")
    return out_dir, N

# ---------- Workload 參數組裝：關鍵修正 ----------
def extract_topo_description(topo_arg: str, logical_dims: list[int] = None) -> str:
    """
    從拓撲參數中提取簡潔的描述信息，用於目錄命名。
    """
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
        # 直接是檔案路徑
        file_path = Path(topo_arg)
        return f"file_{file_path.stem}"

def build_workload_arg(workload_dir: Path) -> tuple[str, str]:
    """
    回傳 (傳給 --workload-configuration 的路徑字串, 顯示用途的說明字串)。
    - 若為 .et：傳回 <dir>/<prefix>（檔名前綴，不含 .rank.et）
    - 若為 manifest+JSON：傳回目錄（維持相容）
    """
    et_map = list_et_rank_files(workload_dir)
    if et_map:
        any_path = next(iter(et_map.values()))
        m = ET_PAT.match(any_path.name)
        prefix = m.group("prefix") if m else "et"
        prefix_path = (workload_dir / prefix).as_posix()
        return prefix_path, f"prefix={prefix_path}"
    else:
        # manifest 模式：仍傳資料夾
        return workload_dir.as_posix(), f"dir={workload_dir.as_posix()}"

# ---------- 主流程 ----------

def main():
    ap = argparse.ArgumentParser(description="Run ASTRA-sim ns-3 with Chakra workload")
    ap.add_argument("--workload", required=True, help="資料夾：.et 或 manifest.json+et_rank_*.json")
    ap.add_argument("--virtual-world", type=int, default=None,
                    help="用『實測 .et』當模板，臨時擴張到指定 world size（N）；僅支援 .et 工作負載")
    ap.add_argument("--system",  default="configs/astra-sim/system/system.json", help="system.json (baseline)")
    ap.add_argument("--network", default="configs/astra-sim/ns3/config.txt",     help="ns-3 config.txt (baseline)")
    ap.add_argument("--remote",  default="configs/astra-sim/remote_memory.json", help="remote_memory.json")
    ap.add_argument("--phys-topo", default=None,
                    help="ns-3 實體拓樸檔（.txt）。未指定時嘗試依 world size 在 configs/astra-sim/topos/ 推測；失敗則沿用 baseline 並警告")
    ap.add_argument("--topo", default="auto:1d",
                    help="邏輯拓樸：auto:1d | auto:2d | auto:3d | dims:2x4 | dims:2x2x2 | file:/path/to.json")
    ap.add_argument("--ns3-bin", default=os.environ.get(
        "ASTRA_NS3_BIN",
        os.path.join(os.environ.get("ASTRA_SIM", "/workspace/astra-sim"),
                     "extern/network_backend/ns-3/build/scratch/ns3.42-AstraSimNetwork-default")
    ), help="ns-3 backend binary")
    # System-layer overrides
    ap.add_argument("--coll-opt", default=None, help='覆蓋 system.json 的 "collective-optimization"（例 localBWAware / none）')
    ap.add_argument("--lmbw", type=int, default=None, help="覆蓋 system.json 的 local-mem-bw（例 1600）")
    # Network-layer overrides
    ap.add_argument("--qcn", type=int, default=None, help="ENABLE_QCN 覆蓋（0/1）")
    ap.add_argument("--pfc-dyn", type=int, default=None, help="USE_DYNAMIC_PFC_THRESHOLD 覆蓋（0/1）")
    ap.add_argument("--buffer", type=int, default=None, help="BUFFER_SIZE 覆蓋（例 16/32/64）")
    ap.add_argument("--payload", type=int, default=None, help="PACKET_PAYLOAD_SIZE 覆蓋（例 512/1000/1500）")
    # 重要修正：預設 None，不再使用 'empty'
    ap.add_argument("--comm-group", default=None,
                    help="傳給 --comm-group-configuration 的檔案路徑；不指定則不帶此參數")
    ap.add_argument("--log-dir", default="runs", help="儲存 ns-3 輸出與暫存檔（預設 runs）")
    ap.add_argument("--dry-run", action="store_true", help="只產生 patched 檔與指令，不執行")
    args = ap.parse_args()

    workload_dir = Path(args.workload).resolve()
    sys_json     = Path(args.system).resolve()
    net_cfg      = Path(args.network).resolve()
    remote_json  = Path(args.remote).resolve()
    ns3_bin      = Path(args.ns3_bin).resolve()
    for p in [workload_dir, sys_json, net_cfg, remote_json, ns3_bin]:
        if not p.exists():
            raise SystemExit(f"[ERR] 找不到：{p}")

    # 先獲取基本信息來決定目錄名稱
    stamp = time.strftime("%Y%m%d-%H%M%S%z")  # 加入時區信息 (+0800, +0000 等)

    # 如果有虛擬擴張，先用原始 workload 計算 world size，然後使用虛擬的 world size
    if args.virtual_world is not None:
        original_world = count_world_size(workload_dir)
        world = args.virtual_world
    else:
        world = count_world_size(workload_dir)

    # 解析拓撲參數獲得邏輯維度
    topo_arg = args.topo
    if topo_arg.startswith("file:"):
        topo_json_path = Path(topo_arg.replace("file:", "")).resolve()
        logical_dims = load_logical_dims(topo_json_path)
    elif topo_arg.startswith("dims:"):
        logical_dims = [int(x) for x in topo_arg.replace("dims:", "").lower().split("x")]
        if math.prod(logical_dims) != world:
            raise SystemExit(f"[ERR] dims 乘積 {math.prod(logical_dims)} != world {world}")
    elif topo_arg.startswith("auto:"):
        mode = topo_arg.split(":", 1)[1]
        if   mode == "1d": logical_dims = [world]
        elif mode == "2d": logical_dims = list(squareish_2d(world))
        elif mode == "3d": logical_dims = list(cubeish_3d(world))
        else: raise SystemExit(f"[ERR] 不支援的 auto 模式：{mode}")
        print(f"[INFO] logical-dims (auto:{mode}) → {logical_dims}")
    else:
        topo_json_path = Path(topo_arg).resolve()
        logical_dims = load_logical_dims(topo_json_path)

    # 建立最終的目錄名稱
    topo_desc = extract_topo_description(topo_arg, logical_dims)
    logroot = Path(args.log_dir).resolve() / f"{stamp}_ns3_{world}gpu_{topo_desc}"
    tmp_dir = logroot / "tmp"
    out_dir = logroot / "out"
    tmp_dir.mkdir(parents=True, exist_ok=True, mode=0o777)
    out_dir.mkdir(parents=True, exist_ok=True, mode=0o777)

    # 確保父目錄 logroot 也有正確的權限（mkdir 的 mode 不會影響父目錄）
    logroot.chmod(0o777)

    # （可選）虛擬擴張：以你的實測 .et 變形/放大到 N 份（只改 comm_size）
    workload_dir, actual_world = expand_workload_virtual_if_needed(workload_dir, tmp_dir, args.virtual_world)
    print(f"[INFO] workload={workload_dir}  world_size={actual_world}")

    # ---------- 邏輯拓樸（ASTRA-sim） ----------
    # 根據拓撲參數創建或驗證拓撲文件
    if topo_arg.startswith("file:"):
        topo_json = Path(topo_arg.replace("file:", "")).resolve()
        assert_logical_dims_match_world(topo_json, actual_world)
    elif topo_arg.startswith("dims:") or topo_arg.startswith("auto:"):
        topo_json = gen_topology_file(tmp_dir / "logical_topology.json", logical_dims)
    else:
        topo_json = Path(topo_arg).resolve()
        assert_logical_dims_match_world(topo_json, actual_world)

    # ---------- system.json 打補丁 ----------
    sys_patched = tmp_dir / "system.patched.json"
    patch_system_json(sys_json, sys_patched, args.coll_opt, args.lmbw)

    # ---------- ns-3 config.txt 打補丁 ----------
    phys_topo = Path(args.phys_topo).resolve() if args.phys_topo else None
    if phys_topo is None:
        guess = guess_phys_topo(Path("configs/astra-sim/topos").resolve(), world)
        if guess:
            phys_topo = guess
            print(f"[INFO] 推測實體拓樸：{phys_topo}")
        else:
            topo_line = next((ln for ln in net_cfg.read_text().splitlines()
                              if ln.strip().startswith("TOPOLOGY_FILE ")), None)
            if topo_line:
                base_topo_path = Path(topo_line.split(maxsplit=1)[1])
                nodes = infer_nodes_from_topo_filename(base_topo_path)
                if nodes and nodes != world:
                    print(f"[WARN] baseline TOPOLOGY_FILE 似為 {nodes} 節點，但 workload world={world}。"
                          f" 建議用 --phys-topo 指定正確的實體拓樸檔。")

    net_patched = tmp_dir / "config.patched.txt"
    patch_network_cfg(net_cfg, net_patched,
                      topo_file=phys_topo, out_dir=out_dir,
                      qcn=args.qcn, pfc_dyn=args.pfc_dyn,
                      buffer_size=args.buffer, payload=args.payload)

    # ---------- Workload prefix（關鍵修正）：把 .et 目錄轉成 prefix 路徑 ----------
    workload_arg, workload_desc = build_workload_arg(workload_dir)
    print(f"[INFO] workload-configuration 以 {workload_desc}")

    # ---------- 組指令並執行 ----------
    cmd = [
        str(ns3_bin),
        f"--workload-configuration={workload_arg}",
        f"--system-configuration={sys_patched.as_posix()}",
        f"--network-configuration={net_patched.as_posix()}",
        f"--remote-memory-configuration={remote_json.as_posix()}",
        f"--logical-topology-configuration={topo_json.as_posix()}",
    ]

    # 只有 args.comm_group 非空且不是 'empty'/'none' 時才附加，並檢查檔案存在
    if args.comm_group and args.comm_group.lower() not in {"empty", "none"}:
        cg_path = Path(args.comm_group).expanduser().resolve()
        if not cg_path.exists():
            raise SystemExit(f"[ERR] --comm-group 指定的檔案不存在：{cg_path}")
        cmd.append(f"--comm-group-configuration={cg_path.as_posix()}")

    (logroot / "command.txt").write_text(" ".join(cmd) + "\n", encoding="utf-8")
    print("[CMD]", " ".join(cmd))

    if args.dry_run:
        print(f"[DRY] patched files at {tmp_dir}")
        return

    # 執行並在錯誤時自動印出 stdout.log 尾端
    stdout_path = (logroot / "stdout.log")
    with stdout_path.open("w", encoding="utf-8") as lf:
        try:
            subprocess.run(cmd, check=True, stdout=lf, stderr=subprocess.STDOUT)
        except subprocess.CalledProcessError as e:
            try:
                lf.flush()
            except Exception:
                pass
            try:
                tail = stdout_path.read_text(encoding="utf-8").splitlines()[-60:]
                print("\n[NS3-STDOUT.TAIL]\n" + "\n".join(tail))
            except Exception:
                print("\n[NS3-STDOUT.TAIL] <無法讀取 stdout.log>")
            shutil.rmtree(logroot)
            print(f"[INFO] 執行失敗，已刪除 {logroot}")
            raise SystemExit(f"[ERR] ns-3 後端退出碼 {e.returncode}，上面已列出最後 60 行輸出。")

    print(f"[INFO] 完成。stdout → {stdout_path}")
    print(f"[INFO] FCT/QLEN/PFC/TRACE 等輸出 → {out_dir}")

if __name__ == "__main__":
    main()
