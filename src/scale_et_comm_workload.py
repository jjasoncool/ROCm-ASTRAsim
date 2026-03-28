#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# 用法範例：
#   python src/scale_et_comm_workload.py \
#     --workload-dir data/chakra/workload_et \
#     --prefix resnet50_all2all \
#     --bytes 1G
#
# 也支援：
#   --bytes 128MB
#   --bytes 64M
#   --bytes 512K
#   --bytes 1048576
#
# 若不指定 --suffix，會依 --bytes 自動產生，例如：
#   1G / 1073741824 -> _1GB
#   128MB          -> _128MB
#   64M            -> _64MB
#   512K           -> _512KB
#
# 產生後可搭配 run_ns3.py 使用，例如：
#   python scripts/run_ns3.py \
#     --workload data/chakra/workload_et \
#     --model-tag resnet50_all2all_1GB \
#     --system configs/astra-sim/system/system_128nodes_Torus_4x4x8.json \
#     --topo file:configs/astra-sim/topos/logical_128nodes_Torus_4x4x8.json \
#     --phys-topo configs/astra-sim/topos/128nodes_Torus_4x4x8.txt \
#     --virtual-world 128 \
#     --payload 12000

"""
ET 檔案通訊量放大工具 (Communication Workload Scaler)
用途：將指定的 ET 檔案中的「集體通訊節點」強制設定為極大的資料量 (如 1GB)，
      用以對 NS-3 網路模擬器進行「頻寬飽和壓力測試」。
"""
from pathlib import Path
import argparse
import sys

from chakra.src.third_party.utils.protolib import decodeMessage as decode_message
from chakra.src.third_party.utils.protolib import encodeMessage as encode_message
from chakra.schema.protobuf.et_def_pb2 import GlobalMetadata, Node, COMM_COLL_NODE, ALL_TO_ALL


def parse_size_to_bytes(size_str: str) -> int:
    s = str(size_str).strip().upper()
    units = {
        "B": 1,
        "K": 1 << 10,
        "KB": 1 << 10,
        "M": 1 << 20,
        "MB": 1 << 20,
        "G": 1 << 30,
        "GB": 1 << 30,
    }
    for unit in ("GB", "MB", "KB", "G", "M", "K", "B"):
        if s.endswith(unit):
            num = s[:-len(unit)].strip()
            if not num:
                raise ValueError(f"invalid size: {size_str}")
            return int(float(num) * units[unit])
    return int(s, 0)


def auto_suffix_from_bytes(comm_bytes: int) -> str:
    units = [
        (1 << 30, "GB"),
        (1 << 20, "MB"),
        (1 << 10, "KB"),
    ]
    for base, unit in units:
        if comm_bytes >= base and comm_bytes % base == 0:
            return f"_{comm_bytes // base}{unit}"
    return f"_{comm_bytes}B"

def read_et(p: Path):
    nodes = []
    with p.open("rb") as f:
        meta = GlobalMetadata()
        decode_message(f, meta)
        size = p.stat().st_size
        while f.tell() < size:
            n = Node()
            pos0 = f.tell()
            try:
                decode_message(f, n)
            except Exception:
                break
            if f.tell() <= pos0:
                break
            nodes.append(n)
    return meta, nodes

def write_et(p: Path, meta, nodes):
    p.parent.mkdir(parents=True, exist_ok=True)
    with p.open("wb") as f:
        encode_message(f, meta)
        for n in nodes:
            encode_message(f, n)

def patch_one(in_path: Path, out_path: Path, comm_bytes: int):
    meta, nodes = read_et(in_path)
    changed = 0
    for n in nodes:
        if n.type != COMM_COLL_NODE:
            continue
        has_type = False
        has_size = False
        for a in n.attr:
            if a.name == "comm_type":
                a.int64_val = ALL_TO_ALL
                has_type = True
            if a.name == "comm_size":
                a.int64_val = comm_bytes
                has_size = True
        if not has_type:
            a = n.attr.add()
            a.name = "comm_type"
            a.int64_val = ALL_TO_ALL
        if not has_size:
            a = n.attr.add()
            a.name = "comm_size"
            a.int64_val = comm_bytes
        changed += 1

    write_et(out_path, meta, nodes)
    return changed

def main():
    ap = argparse.ArgumentParser(description="放大 ET 檔案的通訊量以進行壓力測試")
    ap.add_argument("--workload-dir", required=True, help="放 .et 的資料夾")
    ap.add_argument("--prefix", required=True, help="要讀取的檔案前綴 (例如: resnet50_all2all)")
    ap.add_argument("--suffix", default=None, help="新檔案要加上的後綴；未指定則依 --bytes 自動產生 (例如 _1GB)")
    ap.add_argument("--bytes", default="1G", help="每個 collective 的 comm_size，支援 1G/128MB/64K/1048576 (預設: 1G)")
    args = ap.parse_args()

    comm_bytes = parse_size_to_bytes(args.bytes)
    suffix = args.suffix if args.suffix is not None else auto_suffix_from_bytes(comm_bytes)
    d = Path(args.workload_dir)

    search_pattern = f"et.{args.prefix}.*.et"
    et_files = sorted(d.glob(search_pattern))

    if not et_files:
        print(f"[ERR] 在 {d} 找不到符合 {search_pattern} 的檔案！")
        sys.exit(1)

    total_changed = 0
    generated_files = []

    for in_path in et_files:
        parts = in_path.name.split('.')
        new_name = f"{parts[0]}.{parts[1]}{suffix}.{parts[2]}.{parts[3]}"
        out_path = in_path.parent / new_name

        changed = patch_one(in_path, out_path, comm_bytes)
        total_changed += changed
        generated_files.append(out_path.name)

    print(f"[OK] 成功讀取 {len(et_files)} 個檔案，共修改 {total_changed} 個通訊節點。")
    print(f"[OK] 新產生的檔案前綴為: {args.prefix}{suffix}")
    print(f"[DEBUG] 產生的檔案範例: {generated_files[0]}")
    print(f"[INFO] 建議 run_ns3.py 使用 --model-tag {args.prefix}{suffix}")

if __name__ == "__main__":
    main()
