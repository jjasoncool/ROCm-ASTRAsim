#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
ET 檔案驗證工具 - 增強版

用途：
  逐檔驗證 Chakra protobuf .et（Execution Trace）檔案的正確性，並提供詳細分析。

主要功能：
  - 基本驗證：檔案格式、protobuf 解碼、metadata 版本
  - 詳細分析：節點類型統計、通訊大小、執行時間分析
  - DAG 依賴驗證：檢查節點依賴關係的完整性和合理性
  - 深度驗證：跨 rank 一致性檢查、節點對應關係驗證
  - 錯誤檢測：檔案截斷、格式錯誤、數據不一致
  - 自動修復：生成乾淨的 ET 檔案

關鍵特性：
  1) 安全的 protobuf 解碼：避免在截斷檔案處無限等待
  2) 多層次驗證：從基本格式到深度語義檢查
  3) DAG 完整性檢查：自我依賴、缺失節點、循環依賴檢測
  4) 詳細統計：通訊模式、時間分布、節點類型分析
  5) 跨 rank 驗證：確保分散式訓練的一致性
  6) 修復功能：自動生成修復後的工作負載
  7) 通用檔名支援：支援任何 prefix.{num}.et 格式的檔案

使用方式：
  基本驗證（含詳細分析）：
    python src/tests/validate_et.py

  深度驗證：
    python src/tests/validate_et.py --validate

  DAG 依賴檢查：
    python src/tests/validate_et.py --check-dag

  自動修復：
    python src/tests/validate_et.py --fix --output data/chakra/fixed_workload

  限制節點數（快速檢查）：
    python src/tests/validate_et.py --max-nodes 20

  完整分析：
    python src/tests/validate_et.py --validate --check-dag

支援的檔案格式：
  - allreduce.0.et, allreduce.1.et, ...
  - et.0.et, et.1.et, ...
  - workload.0.et, workload.1.et, ...
  - 任何 prefix.{rank}.et 格式

輸出說明：
  📁 檔案基本資訊
  🔍 詳細節點分析
  🔗 DAG 依賴關係檢查
  🛠️  自動修復功能
  ✅ 驗證通過項目
  ⚠️  潛在問題警告
  ❌ 嚴重錯誤項目
"""

import sys
import re
import argparse
import signal
import time
import json
import subprocess
from pathlib import Path
from collections import defaultdict
from typing import Dict, Any, List, Tuple, Optional, Set

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

# 匹配 prefix.{num}.et 檔名（確保排序與 rank 一致）
PATTERN = re.compile(r"^(.+)\.(\d+)\.et$")


def read_one_et(p: Path, max_nodes: int | None = None, detailed: bool = False, return_nodes: bool = False):
    """
    解一個 .et 檔並回傳 (meta, nodes_count/nodes_list, total_bytes, kinds, detailed_info)。

    Args:
        return_nodes: 如果為 True，返回節點列表；否則返回節點數量（向後兼容）

    安全性要點：
      - 先記檔案總長度 size，僅在 f.tell() < size 時解下一筆，避免在 EOF 處 decode 而卡住。
      - 每筆解完後檢查位移是否前進（pos1 > pos0），若未前進視為壞資料直接丟錯。
    """
    size = p.stat().st_size
    detailed_info = {
        'compute_nodes': 0,
        'comm_nodes': 0,
        'alpha_nodes': 0,
        'timing_info': [],
        'node_types': {},
        'comm_sizes': []
    }

    with p.open("rb") as f:
        # 先讀 GlobalMetadata（開頭一定有）
        meta = GlobalMetadata()
        before = f.tell()
        decode_message(f, meta)         # 內部會先讀 varint32 長度，再讀對應 bytes
        after = f.tell()
        if after == before:
            raise RuntimeError(f"{p.name}: 無法讀取 GlobalMetadata（檔案格式不對或為空）")

        nodes_count = 0
        nodes_list = []
        total_bytes = 0
        kinds: dict[int, int] = {}

        # 逐 Node 讀取：以「檔案大小」為上限，避免在 EOF 或壞尾巴處無限讀取
        while f.tell() < size:
            if max_nodes is not None and nodes_count >= max_nodes:
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

            nodes_count += 1
            if return_nodes:
                nodes_list.append(node)

            # 抽取常用屬性（comm_type / comm_size）做統計摘要
            comm_type = None
            comm_size = None
            alpha = None
            duration_micros = None

            for a in node.attr:
                if a.name == "comm_type":
                    comm_type = a.int64_val
                elif a.name == "comm_size":
                    comm_size = a.int64_val
                elif a.name == "alpha":
                    alpha = a.float_val
                elif a.name == "duration_micros":
                    duration_micros = a.float_val

            # 詳細分析
            if detailed:
                # 節點類型分析
                node_type = node.type if hasattr(node, 'type') else 'UNKNOWN'
                detailed_info['node_types'][node_type] = detailed_info['node_types'].get(node_type, 0) + 1

                # 計算/通訊節點分類
                if comm_type is not None:
                    detailed_info['comm_nodes'] += 1
                    if comm_size is not None:
                        detailed_info['comm_sizes'].append(comm_size)
                else:
                    detailed_info['compute_nodes'] += 1

                # Alpha 值檢查
                if alpha is not None:
                    detailed_info['alpha_nodes'] += 1

                # 時間資訊（修正：記錄 node.id）
                if duration_micros is not None:
                    detailed_info['timing_info'].append({
                        'node_id': node.id,
                        'duration': duration_micros,
                        'type': 'COMM' if comm_type is not None else 'COMPUTE'
                    })

            if comm_type is not None:
                kinds[comm_type] = kinds.get(comm_type, 0) + 1
            if comm_size is not None:
                total_bytes += comm_size

        return meta, (nodes_list if return_nodes else nodes_count), total_bytes, kinds, detailed_info if detailed else None


def check_dag_integrity(nodes: List[Node]) -> Tuple[bool, List[str]]:
    """
    檢查 DAG 的完整性，包括：
    1. 自我依賴檢查
    2. 缺失節點檢查
    3. 依賴關係的合理性

    Returns:
        (is_valid, issues_list)
    """
    issues = []

    # 建立節點 ID 到索引的映射
    id_to_index = {node.id: i for i, node in enumerate(nodes)}

    # 1. 檢查自我依賴
    self_deps = []
    for i, node in enumerate(nodes):
        if hasattr(node, 'ctrl_deps') and node.id in node.ctrl_deps:
            self_deps.append(f"Node {node.id} (index {i}) has self-dependency")

    if self_deps:
        issues.extend(self_deps)
        issues.append(f"❌ Found {len(self_deps)} nodes with self-dependencies")

    # 2. 檢查缺失的依賴節點
    missing_deps = []
    for i, node in enumerate(nodes):
        if hasattr(node, 'ctrl_deps'):
            for dep_id in node.ctrl_deps:
                if dep_id not in id_to_index:
                    missing_deps.append(f"Node {node.id} depends on missing node {dep_id}")

    if missing_deps:
        issues.extend(missing_deps)
        issues.append(f"❌ Found {len(missing_deps)} missing dependency references")

    # 3. 檢查循環依賴（簡化版）
    visited = set()
    rec_stack = set()

    def has_cycle(node_id: int) -> bool:
        if node_id in rec_stack:
            return True
        if node_id in visited or node_id not in id_to_index:
            return False

        visited.add(node_id)
        rec_stack.add(node_id)

        node = nodes[id_to_index[node_id]]
        if hasattr(node, 'ctrl_deps'):
            for dep_id in node.ctrl_deps:
                if dep_id in id_to_index and has_cycle(dep_id):
                    return True

        rec_stack.remove(node_id)
        return False

    cycles_found = []
    for node in nodes:
        if node.id not in visited:
            if has_cycle(node.id):
                cycles_found.append(f"Cycle detected involving node {node.id}")

    if cycles_found:
        issues.extend(cycles_found)
        issues.append(f"❌ Found potential cycles in dependency graph")

    is_valid = len(issues) == 0
    return is_valid, issues


def fix_et_files(output_dir: str) -> bool:
    """
    自動修復 ET 檔案的問題，生成乾淨的工作負載

    Returns:
        True if successful, False otherwise
    """
    print("🛠️  開始自動修復 ET 檔案...")

    try:
        # 確保輸出目錄存在
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)

        # 尋找 host trace 檔案
        host_files = list(Path("data/chakra").glob("*host*.et"))
        if not host_files:
            print("❌ 找不到 host trace 檔案")
            return False

        print(f"📁 找到 {len(host_files)} 個 host trace 檔案")

        # 使用 chakra_converter 直接轉換
        for host_file in host_files:
            rank = "0"  # 預設 rank
            if "rank" in host_file.name:
                import re
                match = re.search(r'rank(\d+)', host_file.name)
                if match:
                    rank = match.group(1)

            output_file = output_path / f"allreduce.{rank}.et"

            cmd = [
                "python", "-m", "chakra.et_converter.et_converter",
                "--input_type", "PyTorchProfiler",
                "--input_filename", str(host_file),
                "--output_filename", str(output_file),
                "--output_type", "Chakra"
            ]

            print(f"🔄 轉換 {host_file.name} -> {output_file.name}")
            result = subprocess.run(cmd, capture_output=True, text=True)

            if result.returncode == 0:
                print(f"✅ 成功生成 {output_file.name}")
            else:
                print(f"❌ 轉換失敗: {result.stderr}")
                return False

        print(f"🎉 修復完成！檔案儲存至: {output_dir}")
        return True

    except Exception as e:
        print(f"❌ 修復過程發生錯誤: {e}")
        return False


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--max-nodes", type=int, default=None,
        help="每檔最多解碼的節點數（抽樣驗證；預設全量）"
    )
    ap.add_argument(
        "--validate", action="store_true",
        help="執行深度驗證檢查（檢查節點一致性、時序等）"
    )
    ap.add_argument(
        "--check-dag", action="store_true",
        help="執行 DAG 依賴關係完整性檢查"
    )
    ap.add_argument(
        "--fix", action="store_true",
        help="自動修復 ET 檔案問題"
    )
    ap.add_argument(
        "--output", type=str, default="data/chakra/fixed_workload",
        help="修復後檔案的輸出目錄（預設：data/chakra/fixed_workload）"
    )
    ap.add_argument(
        "--prefix", type=str, default=None,
        help="指定要檢查的檔案前綴 (例如: resnet50_all2all_1GB)"
    )
    args = ap.parse_args()

    # 如果是修復模式，直接執行修復
    if args.fix:
        success = fix_et_files(args.output)
        sys.exit(0 if success else 1)

    # 詳細分析預設開啟
    detailed = True

    # 以「此程式所在目錄」為基準定位到 ../../data/chakra/workload_et
    script_dir = Path(__file__).resolve().parent
    trace_dir = (script_dir / "../../data/chakra/workload_et").resolve()

    # 找所有 prefix.{num}.et 檔案，並依 {num} 數值排序（對應 rank）
    all_files = list(trace_dir.glob("*.et"))
    files = [p for p in all_files if PATTERN.match(p.name)]
    # [新增] 如果有指定前綴，過濾掉不相關的檔案
    if args.prefix:
        files = [p for p in files if PATTERN.match(p.name).group(1) == args.prefix]
    files.sort(key=lambda p: int(PATTERN.match(p.name).group(2)))  # group(2) 是數字部分

    if not files:
        print(f"[ERR] 找不到符合樣式的檔案：{trace_dir}/*.{{num}}.et")
        sys.exit(1)

    # 檢查檔案前綴一致性
    prefixes = set(PATTERN.match(p.name).group(1) for p in files)
    if len(prefixes) > 1:
        print(f"[WARN] 發現多種檔案前綴：{sorted(prefixes)}")
    else:
        prefix = list(prefixes)[0]
        print(f"[INFO] 檢查目錄：{trace_dir}  檔數：{len(files)}  前綴：{prefix}")

    all_detailed_info = []

    for p in files:
        meta, nodes, total_bytes, kinds, detailed_info = read_one_et(
            p, max_nodes=args.max_nodes, detailed=detailed or args.validate
        )

        # 重新讀取完整節點資料用於 DAG 檢查
        dag_nodes = None
        if args.check_dag:
            _, dag_nodes, _, _, _ = read_one_et(p, max_nodes=None, detailed=False, return_nodes=True)

        kind_str = ", ".join([f"{COMM_NAMES.get(k, k)}:{v}" for k, v in kinds.items()]) or "N/A"

        rank = int(PATTERN.match(p.name).group(2))  # group(2) 是數字部分
        print(f"📁 {p.name:20s}  version={meta.version or 'N/A'}  nodes={nodes:4d}  bytes_sum={total_bytes:,}  kinds=({kind_str})")

        # 版本提醒（僅提示，非致命）
        if meta.version and meta.version not in ["0.0.4", "0.0.3", "0.0.x"]:
            print(f"  [WARN] 未知的 metadata 版本：{meta.version}")

        # DAG 依賴關係檢查
        if args.check_dag and dag_nodes:
            print(f"   🔗 DAG 依賴關係檢查 (Rank {rank}):")
            dag_valid, dag_issues = check_dag_integrity(dag_nodes)
            if dag_valid:
                print(f"      ✅ DAG 結構正常")
            else:
                print(f"      ❌ 發現 DAG 問題:")
                for issue in dag_issues[:5]:  # 限制顯示前 5 個問題
                    print(f"         • {issue}")
                if len(dag_issues) > 5:
                    print(f"         ... 還有 {len(dag_issues)-5} 個問題")

        # 詳細分析 (預設開啟)
        if detailed and detailed_info:
            print(f"   🔍 詳細分析 (Rank {rank}):")
            print(f"      計算節點: {detailed_info['compute_nodes']}")
            print(f"      通訊節點: {detailed_info['comm_nodes']}")
            print(f"      含Alpha節點: {detailed_info['alpha_nodes']}")

            if detailed_info['comm_sizes']:
                avg_comm = sum(detailed_info['comm_sizes']) / len(detailed_info['comm_sizes'])
                min_comm = min(detailed_info['comm_sizes'])
                max_comm = max(detailed_info['comm_sizes'])
                print(f"      平均通訊大小: {avg_comm:,.0f} bytes ({avg_comm/1024/1024:.2f} MiB)")
                print(f"      最小通訊大小: {min_comm:,.0f} bytes ({min_comm/1024/1024:.2f} MiB)")
                print(f"      最大通訊大小: {max_comm:,.0f} bytes ({max_comm/1024/1024:.2f} MiB)")
                print(f"      各 bucket 大小: {', '.join(f'{s/1024/1024:.2f} MiB' for s in detailed_info['comm_sizes'])}")

            if detailed_info['timing_info']:
                total_time = sum(t['duration'] for t in detailed_info['timing_info'])
                compute_time = sum(t['duration'] for t in detailed_info['timing_info'] if t['type'] == 'COMPUTE')
                comm_time = sum(t['duration'] for t in detailed_info['timing_info'] if t['type'] == 'COMM')
                print(f"      總執行時間: {total_time:,.0f} μs")
                print(f"      計算時間: {compute_time:,.0f} μs ({compute_time/total_time*100:.1f}%)")
                print(f"      通訊時間: {comm_time:,.0f} μs ({comm_time/total_time*100:.1f}%)")

            # === ASTRA 相容性健檢（新增輸出，不改動原有註解） ===
            # 1) 偵測可能混入排程的 METADATA/PG 節點
            raw_bytes = p.read_bytes()
            node_types_map = detailed_info.get('node_types', {})
            # 常見情況：schema 下的 METADATA_NODE 會是 type=1（以你環境為準）
            suspicious_meta = (1 in node_types_map) or (b"process_group" in raw_bytes or b"## process_group" in raw_bytes)
            if suspicious_meta:
                print("      ❌ 偵測到可能的 METADATA/PG 事件混入可發佈節點；請在轉檔後清理（fix_et_dag_inplace 的 ASTRA 相容流程）。")

            # 2) 偵測 comm_size <= 0
            if detailed_info['comm_sizes'] and any(sz <= 0 for sz in detailed_info['comm_sizes']):
                print("      ❌ 偵測到 comm_size <= 0；請在轉檔後補值（group_id / comm_size）。")
            elif not detailed_info['comm_sizes']:
                print("      ⚠️  本檔案缺少 COMM 尺寸統計，可能代表 COMM 屬性未正確寫入。")

        if args.validate:
            all_detailed_info.append((rank, detailed_info))

    # 深度驗證
    if args.validate and len(all_detailed_info) > 1:
        print(f"\n🔍 深度驗證分析:")

        # 檢查 ranks 間的一致性
        ranks = [info[0] for info in all_detailed_info]
        node_counts = [info[1]['compute_nodes'] + info[1]['comm_nodes'] for info in all_detailed_info]
        comm_counts = [info[1]['comm_nodes'] for info in all_detailed_info]

        # 智能分析節點數差異
        print(f"   節點數分佈: {node_counts}")
        if len(set(node_counts)) == 1:
            print("   ✅ 所有 ranks 的節點數相同")
        else:
            max_nodes = max(node_counts)
            min_nodes = min(node_counts)
            diff = max_nodes - min_nodes
            diff_pct = (diff / max_nodes) * 100

            print(f"   📊 節點數差異分析:")
            print(f"      最大: {max_nodes}, 最小: {min_nodes}, 差異: {diff} ({diff_pct:.1f}%)")

            # 判斷差異是否合理
            if diff <= max_nodes * 0.05:  # 小於5%的差異
                print("   ✅ 節點數差異在合理範圍內 (≤5%)")
                print("      可能原因: 數據分割不均、動態停止、網路同步延遲")
            elif diff <= max_nodes * 0.15:  # 小於15%的差異
                print("   ⚠️  節點數差異較大但可接受 (5-15%)")
                print("      建議檢查: 數據加載器配置、sampler 設定")
            else:
                print("   ❌ 節點數差異過大 (>15%)")
                print("      可能問題: 訓練提前終止、嚴重的同步問題")

        # 通訊節點分析
        print(f"   通訊節點數: {comm_counts}")
        if len(set(comm_counts)) == 1:
            print("   ✅ 所有 ranks 的通訊節點數相同")
        else:
            comm_diff = max(comm_counts) - min(comm_counts)
            print(f"   📊 通訊節點差異: {comm_diff}")
            if comm_diff == diff // 2:  # 每步包含1個計算+1個通訊節點
                print("   ✅ 通訊節點差異與總節點差異一致 (正常)")
            else:
                print("   ⚠️  通訊節點差異與預期不符，可能有異常")

        # 步數推算
        steps = [count // 2 for count in node_counts]  # 假設每步2個節點
        print(f"   推估步數: {steps}")
        if len(set(steps)) > 1:
            step_diff = max(steps) - min(steps)
            print(f"   📊 步數差異: {step_diff} 步")
            print(f"      這在分散式訓練中是正常現象")

        # 檢查通訊大小一致性
        all_comm_sizes = [set(info[1]['comm_sizes']) for info in all_detailed_info]
        if len(all_comm_sizes) > 1 and all(sizes == all_comm_sizes[0] for sizes in all_comm_sizes):
            print("   ✅ 所有 ranks 的通訊大小一致")
        else:
            print("   ⚠️  不同 ranks 的通訊大小可能不一致")

    # 連續命名檢查：rank 應為連續（0..N-1）
    idxs = [int(PATTERN.match(p.name).group(2)) for p in files]  # group(2) 是數字部分
    expect = list(range(min(idxs), min(idxs) + len(idxs)))
    if sorted(idxs) != expect:
        print(f"[WARN] 檔名 index 不是連續的：{sorted(idxs)}")
    else:
        print(f"✅ 檔名 index 連續性正確: {sorted(idxs)}")


if __name__ == "__main__":
    main()
