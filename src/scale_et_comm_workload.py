#!/usr/bin/env python3
# -*- coding: utf-8 -*-
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
    ap.add_argument("--suffix", default="_1GB", help="新檔案要加上的後綴 (預設: _1GB)")
    ap.add_argument("--bytes", default=str(1<<30), help="每個 collective 的 comm_size (預設: 1GB)")
    args = ap.parse_args()

    comm_bytes = int(args.bytes, 0)
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
        new_name = f"{parts[0]}.{parts[1]}{args.suffix}.{parts[2]}.{parts[3]}"
        out_path = in_path.parent / new_name

        changed = patch_one(in_path, out_path, comm_bytes)
        total_changed += changed
        generated_files.append(out_path.name)

    print(f"[OK] 成功讀取 {len(et_files)} 個檔案，共修改 {total_changed} 個通訊節點。")
    print(f"[OK] 新產生的檔案前綴為: {args.prefix}{args.suffix}")
    print(f"[DEBUG] 產生的檔案範例: {generated_files[0]}")

if __name__ == "__main__":
    main()
