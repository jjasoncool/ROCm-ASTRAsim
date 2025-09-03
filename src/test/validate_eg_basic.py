#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
用途：
  逐檔驗證 Chakra protobuf .et（Execution Trace）是否可正確解碼，並摘要列出：
  - GlobalMetadata 版本
  - 節點（Node）數量
  - 各種 collective 類型的出現次數
  - comm_size 合計（bytes）

關鍵說明：
  1) .et 檔是由「多筆 protobuf 訊息串接」而成，每筆訊息前面有一個 varint32 的長度前綴。
     讀取流程是：讀長度 → 再讀該長度的訊息內容。
  2) 若 EOF/尾端殘缺（截斷）時直接呼叫 decode()，在嘗試讀取下一筆 varint 長度時可能「原地讀 1 byte」卡住。
     因此本程式以「檔案大小 + 檔案位移（offset）」當哨兵：
       - 只有在 f.tell() < file_size 時才嘗試讀下一筆；
       - 解完一筆後檢查 offset 是否前進，否則直接報錯，避免無限迴圈。

使用方式：
  直接執行（固定讀取相對於腳本目錄的 ../../data/chakra/workload_et/allreduce.{num}.et）：
    python ./src/et/validate_et_basic.py

  只抽樣每檔前 N 筆（加速）：
    python ./src/et/validate_et_basic.py --max-nodes 10
"""

import sys
import re
import argparse
from pathlib import Path

# Chakra 的 protolib：提供 encode/decodeMessage（長度前綴的 protobuf 串流）
from chakra.src.third_party.utils.protolib import decodeMessage as decode_message
# et_def_pb2：Chakra 的 protobuf schema（Node / GlobalMetadata / 常數）
from chakra.schema.protobuf.et_def_pb2 import (
    GlobalMetadata, Node, ALL_REDUCE, ALL_GATHER, REDUCE_SCATTER, ALL_TO_ALL
)

COMM_NAMES = {
    ALL_REDUCE: "ALL_REDUCE",
    ALL_GATHER: "ALL_GATHER",
    REDUCE_SCATTER: "REDUCE_SCATTER",
    ALL_TO_ALL: "ALL_TO_ALL",
}

# 僅匹配 allreduce.{num}.et 檔名（確保排序與 rank 一致）
PATTERN = re.compile(r"^allreduce\.(\d+)\.et$")


def read_one_et(p: Path, max_nodes: int | None = None):
    """
    解一個 .et 檔並回傳 (meta, nodes, total_bytes, kinds)。

    安全性要點：
      - 先記檔案總長度 size，僅在 f.tell() < size 時解下一筆，避免在 EOF 處 decode 而卡住。
      - 每筆解完後檢查位移是否前進（pos1 > pos0），若未前進視為壞資料直接丟錯。
    """
    size = p.stat().st_size
    with p.open("rb") as f:
        # 先讀 GlobalMetadata（開頭一定有）
        meta = GlobalMetadata()
        before = f.tell()
        decode_message(f, meta)         # 內部會先讀 varint32 長度，再讀對應 bytes
        after = f.tell()
        if after == before:
            raise RuntimeError(f"{p.name}: 無法讀取 GlobalMetadata（檔案格式不對或為空）")

        nodes = 0
        total_bytes = 0
        kinds: dict[int, int] = {}

        # 逐 Node 讀取：以「檔案大小」為上限，避免在 EOF 或壞尾巴處無限讀取
        while f.tell() < size:
            if max_nodes is not None and nodes >= max_nodes:
                break

            pos0 = f.tell()
            node = Node()
            try:
                decode_message(f, node)  # 一樣：先讀長度前綴，再讀內容
            except Exception:
                # 讀到壞節點 / 非預期 EOF：直接跳出（已讀到的資料仍可回報）
                break
            pos1 = f.tell()

            # 防呆：解完一筆後 offset 應該前進；否則表示檔案壞掉或 decode 出錯，避免卡死
            if pos1 <= pos0:
                raise RuntimeError(f"{p.name}: 解碼卡住（offset 未前進 at {pos0}），檔案可能毀損")

            nodes += 1

            # 抽取常用屬性（comm_type / comm_size）做統計摘要
            comm_type = None
            comm_size = None
            for a in node.attr:
                if a.name == "comm_type":
                    comm_type = a.int64_val
                elif a.name == "comm_size":
                    comm_size = a.int64_val

            if comm_type is not None:
                kinds[comm_type] = kinds.get(comm_type, 0) + 1
            if comm_size is not None:
                total_bytes += comm_size

        return meta, nodes, total_bytes, kinds


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--max-nodes", type=int, default=None,
        help="每檔最多解碼的節點數（抽樣驗證；預設全量）"
    )
    args = ap.parse_args()

    # 以「此程式所在目錄」為基準定位到 ../../data/chakra/workload_et
    script_dir = Path(__file__).resolve().parent
    trace_dir = (script_dir / "../../data/chakra/workload_et").resolve()

    # 只吃 allreduce.{num}.et，並依 {num} 數值排序（對應 rank）
    all_files = list(trace_dir.glob("allreduce.*.et"))
    files = [p for p in all_files if PATTERN.match(p.name)]
    files.sort(key=lambda p: int(PATTERN.match(p.name).group(1)))

    if not files:
        print(f"[ERR] 找不到符合樣式的檔案：{trace_dir}/allreduce.{{num}}.et")
        sys.exit(1)

    print(f"[INFO] 檢查目錄：{trace_dir}  檔數：{len(files)}")
    for p in files:
        meta, nodes, total_bytes, kinds = read_one_et(p, max_nodes=args.max_nodes)
        kind_str = ", ".join([f"{COMM_NAMES.get(k, k)}:{v}" for k, v in kinds.items()]) or "N/A"
        print(f"- {p.name:20s}  meta.version={meta.version or 'N/A'}  nodes={nodes:4d}  bytes_sum={total_bytes}  kinds=({kind_str})")

        # 版本提醒（僅提示，非致命）
        if meta.version and meta.version not in ["0.0.4", "0.0.3", "0.0.x"]:
            print(f"  [WARN] 未知的 metadata 版本：{meta.version}")

    # 連續命名檢查：rank 應為連續（0..N-1）
    idxs = [int(PATTERN.match(p.name).group(1)) for p in files]
    expect = list(range(min(idxs), min(idxs) + len(idxs)))
    if sorted(idxs) != expect:
        print(f"[WARN] 檔名 index 不是連續的：{sorted(idxs)}")


if __name__ == "__main__":
    main()
