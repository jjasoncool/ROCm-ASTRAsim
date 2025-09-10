#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import argparse, json, math, os, re, shutil, subprocess, sys, time
from pathlib import Path

ET_PAT = re.compile(r".*\.(\d+)\.et$")

def count_world_size(workload_dir: Path) -> int:
    ets = sorted([p for p in workload_dir.glob("*.et") if ET_PAT.match(p.name)])
    if ets:
        return len(ets)
    # Chakra JSON 風格：manifest + et_rank_*.json
    jfiles = list(workload_dir.glob("et_rank_*.json"))
    if workload_dir.joinpath("manifest.json").exists() and jfiles:
        return len(jfiles)
    raise SystemExit(f"[ERR] {workload_dir} 內找不到 .et 或 et_rank_*.json")

def squareish_2d(n: int) -> tuple[int,int]:
    a = int(math.sqrt(n))
    while a>1 and n%a!=0: a -= 1
    return (a, n//a) if a>1 else (1, n)

def cubeish_3d(n: int) -> tuple[int,int,int]:
    a = int(round(n ** (1/3)))
    if a<1: a=1
    best = (1,1,n); bestgap=n
    for x in range(1, a+2):
        if n % x: continue
        y,z = squareish_2d(n//x)
        dims = (x,y,z)
        gap = max(dims) - min(dims)
        if gap < bestgap:
            best, bestgap = dims, gap
    return best

def gen_topology_file(out_path: Path, dims: list[int]):
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8") as f:
        json.dump({"logical-dims":[str(d) for d in dims]}, f, indent=2)
    return out_path

def patch_system_json(src: Path, out: Path, coll_opt: str|None, lmbw: int|None):
    obj = json.loads(src.read_text(encoding="utf-8"))
    if coll_opt is not None:
        obj["collective-optimization"] = coll_opt
    if lmbw is not None:
        obj["local-mem-bw"] = lmbw
    out.write_text(json.dumps(obj, indent=2), encoding="utf-8")
    return out

def main():
    ap = argparse.ArgumentParser(description="Run ASTRA-sim ns-3 with Chakra workload")
    ap.add_argument("--workload", required=True, help="資料夾：.et 或 manifest.json+et_rank_*.json")
    ap.add_argument("--system", default="configs/astra/system.json", help="system.json")
    ap.add_argument("--network", default="configs/astra/ns3/config.txt", help="ns-3 config.txt")
    ap.add_argument("--remote", default="configs/astra/remote_memory.json", help="remote_memory.json")
    ap.add_argument("--topo", default="auto:1d", help="拓樸：auto:1d | auto:2d | auto:3d | file:/path/to.json | dims:2x4 | dims:2x2x2")
    ap.add_argument("--ns3-bin", default=os.environ.get("ASTRA_NS3_BIN",
                        os.path.join(os.environ.get("ASTRA_SIM","/workspace/astra-sim"),
                                     "extern/network_backend/ns-3/build/scratch/ns3.42-AstraSimNetwork-default")) )
    ap.add_argument("--coll-opt", default=None, help='覆蓋 system.json 的 "collective-optimization"（例如 localBWAware 或 none）')
    ap.add_argument("--lmbw", type=int, default=None, help="覆蓋 system.json 的 local-mem-bw（例 1600）")
    ap.add_argument("--comm-group", default="empty", help='comm-group-configuration（預設 "empty"）')
    ap.add_argument("--log-dir", default="logs/ns3", help="儲存 ns-3 輸出與暫存檔")
    ap.add_argument("--dry-run", action="store_true", help="只印指令不執行")
    args = ap.parse_args()

    workload_dir = Path(args.workload).resolve()
    sys_json = Path(args.system).resolve()
    net_cfg = Path(args.network).resolve()
    remote_json = Path(args.remote).resolve()
    ns3_bin = Path(args.ns3-bin).resolve()

    for p in [workload_dir, sys_json, net_cfg, remote_json, ns3_bin]:
        if not p.exists():
            raise SystemExit(f"[ERR] 找不到：{p}")

    world = count_world_size(workload_dir)
    print(f"[INFO] workload={workload_dir}  world_size={world}")

    # 準備 log/暫存
    stamp = time.strftime("%Y%m%d-%H%M%S")
    log_root = Path(args.log_dir).resolve() / stamp
    tmp_dir = log_root / "tmp"
    out_dir = log_root / "out"
    tmp_dir.mkdir(parents=True, exist_ok=True)
    out_dir.mkdir(parents=True, exist_ok=True)

    # 1) 準拓樸檔
    topo_arg = args.topo
    if topo_arg.startswith("file:"):
        topo_json = Path(topo_arg.replace("file:","")).resolve()
    elif topo_arg.startswith("dims:"):
        dims = [int(x) for x in topo_arg.replace("dims:","").lower().split("x")]
        if math.prod(dims) != world:
            raise SystemExit(f"[ERR] dims 乘積 {math.prod(dims)} != world {world}")
        topo_json = gen_topology_file(tmp_dir/"logical_topology.json", dims)
    elif topo_arg.startswith("auto:"):
        mode = topo_arg.split(":")[1]
        if mode=="1d":
            dims=[world]
        elif mode=="2d":
            dims=list(squareish_2d(world))
        elif mode=="3d":
            dims=list(cubeish_3d(world))
        else:
            raise SystemExit(f"[ERR] 不支援的 auto 模式：{mode}")
        topo_json = gen_topology_file(tmp_dir/"logical_topology.json", dims)
        print(f"[INFO] auto {mode} → logical-dims={dims}")
    else:
        # 預設：用你現成檔案
        topo_json = Path(topo_arg).resolve()

    # 2) patch system.json（如有需要）
    sys_patched = tmp_dir/"system.patched.json"
    patch_system_json(sys_json, sys_patched, args.coll-opt, args.lmbw)

    # 3) 確保 ns-3 的輸出目錄存在（避免 config.txt 指到不存在的目錄）
    #    讀出 config.txt 裡的 *_OUTPUT_FILE 路徑，先建資料夾
    out_paths = []
    for key in ["FLOW_FILE","TRACE_FILE","TRACE_OUTPUT_FILE","FCT_OUTPUT_FILE","PFC_OUTPUT_FILE","QLEN_MON_FILE"]:
        pat = re.compile(rf"^{key}\s+(.+)$")
        for line in net_cfg.read_text().splitlines():
            m = pat.match(line.strip())
            if m:
                out_paths.append(Path(m.group(1)))
    for p in out_paths:
        p.parent.mkdir(parents=True, exist_ok=True)

    # 4) 組指令
    cmd = [
        str(ns3_bin),
        f"--workload-configuration={workload_dir}",
        f"--system-configuration={sys_patched}",
        f"--network-configuration={net_cfg}",
        f"--remote-memory-configuration={remote_json}",
        f"--logical-topology-configuration={topo_json}",
        f'--comm-group-configuration="{args.comm_group}"'
    ]
    print("[CMD]", " ".join(cmd))
    if args.dry_run:
        return

    # 5) 執行
    with (out_dir/"ns3.stdout.log").open("w") as lf:
        proc = subprocess.run(" ".join(cmd), shell=True, stdout=lf, stderr=subprocess.STDOUT)
    print(f"[INFO] 完成，stdout → {out_dir/'ns3.stdout.log'}")
    print(f"[HINT] 請查看 config.txt 指定的 FCT/QLEN/PFC 輸出檔（本次已建立目錄）")

if __name__=="__main__":
    main()
