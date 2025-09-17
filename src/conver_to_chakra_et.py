#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
將 PyTorch Chrome trace (Kineto) 轉成「官方 Chakra protobuf .et」：
- 輸入：../data/chakra/pytorch_traces/trace_rank_{rank}_epoch_{epoch}.json  # 支援多 epoch
- 輸出：../data/chakra/workload_et/allreduce.{rank}.et  （每個 rank 一個 .et，合併多 epoch steps）

做法：
1) 從 trace_rank_* 找 "ProfilerStep#k" 取得每個 step 的 [start_us, end_us]（跨 epoch）。
2) 從同一 trace 於該窗內累積 GPU compute 與 comm（用 rccl/nccl/allreduce 等關鍵字辨識）。
3) 估算每 step 的 All-Reduce bytes（預設：以參數數×dtype bytes，N=2 時每卡≈model_bytes）。
4) 以 protobuf 寫入：
   - GlobalMetadata(version="0.0.4")
   - 多個 ChakraNode：
       type = COMM_COLL_NODE
       attr:
         is_cpu_op      = False
         comm_type      = ALL_REDUCE
         comm_size      = <bytes_per_step>
         involved_dim   = BoolList([True])
       （可選校準欄位）
         ts_ns          = step 起點（ns）
         compute_ns     = 累計 GPU compute（ns）
         comm_ns        = 累計 GPU comm（ns）

需求：
- 已可 import `chakra.schema.protobuf.et_def_pb2` 與 `chakra.src.third_party.utils.protolib`.
"""
import re
import json
import argparse
from pathlib import Path
from typing import List, Dict, Tuple
# Chakra protobuf / encode
from chakra.src.third_party.utils.protolib import encodeMessage as encode_message
from chakra.schema.protobuf.et_def_pb2 import (
    Node as ChakraNode,
    BoolList,
    GlobalMetadata,
    AttributeProto as ChakraAttr,
    COMM_COLL_NODE,
    ALL_REDUCE,
)
# --------------------------------------------------------------------------
# 解析 Chrome trace
COMM_PAT = re.compile(
    r"(rccl|nccl|all[_\- ]?reduce|all[_\- ]?gather|reduce[_\- ]?scatter|broadcast)",
    re.IGNORECASE
)
def load_trace(path: Path) -> Dict:
    with path.open("r", encoding="utf-8", errors="ignore") as f:
        return json.load(f)
def extract_steps_from_trace(trace: Dict) -> List[Tuple[float, float]]:
    """回傳 [(start_us, end_us), ...] 依 ProfilerStep#k。"""
    steps = []
    for e in trace.get("traceEvents", []):
        if e.get("ph") == "X" and isinstance(e.get("name"), str) and e["name"].startswith("ProfilerStep#"):
            ts_us = float(e.get("ts", 0.0))
            dur_us = float(e.get("dur", 0.0))
            if dur_us > 0:
                steps.append((ts_us, ts_us + dur_us))
    steps.sort()
    return steps
def collect_gpu_events(trace: Dict, window: Tuple[float, float]) -> Tuple[float, float]:
    """在時間窗 [start_us, end_us] 內累計 (compute_us_sum, comm_us_sum)。"""
    start_us, end_us = window
    comp_us, comm_us = 0.0, 0.0
    for e in trace.get("traceEvents", []):
        if not isinstance(e, dict):  # 修正：跳過非 dict 元素
            continue
        if e.get("ph") != "X":
            continue
        ts_us = float(e.get("ts", 0.0))
        dur_us = float(e.get("dur", 0.0))
        if dur_us <= 0:
            continue
        s = ts_us
        t = ts_us + dur_us
        inter = max(0.0, min(end_us, t) - max(start_us, s))
        if inter <= 0:
            continue
        name = f"{e.get('name','')} {e.get('cat','')}"
        if COMM_PAT.search(name):
            comm_us += inter
        else:
            comp_us += inter
    return comp_us, comm_us
def get_world_size(trace: Dict) -> int:
    di = trace.get("distributedInfo", {}) or {}
    return int(di.get("world_size", 1))
def to_ns(us: float) -> int:
    return int(round(us * 1000.0))
def per_gpu_allreduce_bytes(world_size: int, model_param_bytes: int) -> int:
    """Ring All-Reduce：每 GPU 需傳 2*(N-1)/N * model_bytes。N=2 時≈model_bytes。"""
    return int(2 * (world_size - 1) / world_size * model_param_bytes)
# --------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--traces-dir", default=None,
                    help="trace 目錄（預設：腳本目錄/../data/chakra/pytorch_traces）")
    ap.add_argument("--out-dir", default=None,
                    help="輸出 .et 目錄（預設：腳本目錄/../data/chakra/workload_et）")
    ap.add_argument("--trace-pattern", default="trace_rank_*.json",
                    help="輸入 trace 檔名模式（預設 trace_rank_*.json，支持多 epoch）")
    ap.add_argument("--dtype", choices=["fp32", "fp16"], default="fp32",
                    help="梯度 dtype（影響 ALLREDUCE bytes 估算）")
    ap.add_argument("--model-param-count", type=int, default=17159306,
                    help="模型參數數量（預設為此專案 CIFAR10_CNN 約 17,159,306）")
    ap.add_argument("--override-bytes", type=int, default=10485760,
                    help="直接指定每 step 每 GPU 的 ALLREDUCE bytes（10MB/step）")
    ap.add_argument("--meta-version", default="0.0.4",
                    help="GlobalMetadata 版本（預設 0.0.4）")
    ap.add_argument("--file-prefix", default="allreduce",
                    help="輸出檔名前綴（預設 allreduce → allreduce.{rank}.et）")
    ap.add_argument("--debug", action="store_true", help="印出每個 step 的 comp_us / comm_us 進行 debug")
    args = ap.parse_args()
    script_dir = Path(__file__).resolve().parent
    traces_dir = Path(args.traces_dir).resolve() if args.traces_dir else (script_dir / "../data/chakra/pytorch_traces").resolve()
    out_dir    = Path(args.out_dir).resolve()    if args.out_dir    else (script_dir / "../data/chakra/workload_et").resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    print(f"[paths] traces={traces_dir}  out={out_dir}")
    trace_files = sorted(traces_dir.glob(args.trace_pattern))
    if not trace_files:
        raise FileNotFoundError(f"找不到 trace_rank_*.json 於 {traces_dir}")
    def rank_from_name(p: Path) -> int:
        m = re.search(r"rank_(\d+)_epoch_\d+\.json$", p.name)
        if not m:
            raise ValueError(f"檔名無法解析 rank：{p.name}")
        return int(m.group(1))
    traces_by_rank: Dict[int, List[Path]] = {}
    for p in trace_files:
        r = rank_from_name(p)
        traces_by_rank.setdefault(r, []).append(p)
    ranks = sorted(traces_by_rank.keys())
    if not ranks:
        raise RuntimeError("trace 檔名沒有 rank 交集，請確認。")
    # world_size：取 rank 最小的 trace 檔
    world_size = get_world_size(load_trace(traces_by_rank[ranks[0]][0]))
    dtype_bytes = 4 if args.dtype == "fp32" else 2
    model_param_bytes = args.model_param_count * dtype_bytes
    bytes_per_step = args.override_bytes if args.override_bytes is not None \
                     else per_gpu_allreduce_bytes(world_size, model_param_bytes)
    print(f"[info] world_size={world_size}  bytes_per_step={bytes_per_step}  dtype={args.dtype}")
    for r in ranks:
        all_steps = []
        for trace_path in sorted(traces_by_rank[r]):  # 跨多 epoch 收集 steps
            trace = load_trace(trace_path)
            steps = extract_steps_from_trace(trace)
            all_steps.extend(steps)  # 合併
        if not all_steps:
            raise RuntimeError(f"Trace 無 ProfilerStep 事件：{traces_by_rank[r]}")
        # 建立輸出 .et
        out_path = out_dir / f"{args.file_prefix}.{r}.et"
        with out_path.open("wb") as et:
            # 1) Global metadata
            encode_message(et, GlobalMetadata(version=args.meta_version))
            # 2) 逐 step 寫 COMM_COLL_NODE (ALL_REDUCE)
            node_id = 1
            for step_idx, (s_us, e_us) in enumerate(all_steps, 1):
                matching_trace = next((t for t in traces_by_rank[r] if any(s_us >= float(e.get("ts", 0.0)) for e in load_trace(t)["traceEvents"])), traces_by_rank[r][0])
                trace = load_trace(matching_trace)
                comp_us, comm_us = collect_gpu_events(trace, (s_us, e_us))
                if args.debug:
                    print(f"[Debug Rank {r} Step {step_idx}] comp_us={comp_us:.2f} comm_us={comm_us:.2f}")
                node = ChakraNode()
                node.id = node_id; node_id += 1
                node.name = f"All-Reduce Step {step_idx}"
                node.type = COMM_COLL_NODE
                # 必要屬性（官方教學風格）
                node.attr.append(ChakraAttr(name="is_cpu_op", bool_val=False))
                node.attr.append(ChakraAttr(name="comm_type", int64_val=ALL_REDUCE))
                node.attr.append(ChakraAttr(name="comm_size", int64_val=bytes_per_step))
                node.attr.append(ChakraAttr(name="involved_dim", bool_list=BoolList(values=[True])))
                # 可選：把實測時間寫成 attribute（供你校準/比對；feeder不識別會忽略）
                node.attr.append(ChakraAttr(name="ts_ns",      int64_val=to_ns(s_us)))
                node.attr.append(ChakraAttr(name="compute_ns", int64_val=to_ns(max(0.0, comp_us))))
                node.attr.append(ChakraAttr(name="comm_ns",    int64_val=to_ns(max(0.0, comm_us))))
                encode_message(et, node)
        print(f"[OK] wrote {out_path} with {len(all_steps)} steps")

if __name__ == "__main__":
    main()
