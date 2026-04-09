#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
add_ddp_to_et.py — 為 TP Execution Trace 加入 DDP AllReduce 通訊節點

=== 目的 ===
TP trace 只記錄 intra-server 的 TP AllReduce（每層做 partial sum 合併）。
但 128 GPU 的真實部署是 TP + DDP：
  - TP（intra-server）：每層 AllReduce，走 PCIe（Z-axis, 65G, Dim 2）
  - DDP（inter-server）：gradient sync AllReduce，走 Ethernet（X/Y-axes, 25G, Dim 0+1）

Logical topology: [4, 4, 8]
  - Dim 0 (size 4, offset 1):  X-axis, 25 Gbps → DDP
  - Dim 1 (size 4, offset 4):  Y-axis, 25 Gbps → DDP
  - Dim 2 (size 8, offset 16): Z-axis, 65 Gbps → TP

TP trace 裡沒有 DDP 通訊，因為只有 1 台 server。
這個腳本把 DDP AllReduce 節點加入 ET，讓 ASTRA-sim 能模擬完整的 TP+DDP。

=== 功能 ===
1. 在 ET 的 backward 末尾加入 DDP AllReduce COMM 節點
2. 根據 target_tp / trace_tp 縮放 COMP 時間（TP=2 trace → TP=8 模擬）
3. 輸出新的 ET 檔案（不覆蓋原始檔案）

=== 用法（獨立執行）===
  # 自動偵測 model params（推薦 — 讀取 model_info JSON）
  python src/add_ddp_to_et.py \
    --et-dir ./data/chakra/workload_et --model-tag qwen15b_tp \
    --target-tp 8

  # 手動指定 model params
  python src/add_ddp_to_et.py \
    --et-file ./data/chakra/workload_et/et.qwen15b_tp.0.et \
    --target-tp 8 --model-params 1543714304 --param-dtype fp16

=== 用法（被 conver_to_chakra_et.py import）===
  from add_ddp_to_et import add_ddp_to_tp_et
  add_ddp_to_tp_et(et_file, target_tp=8)  # auto-detects model params
"""

from __future__ import annotations

import argparse
import math
import sys
from pathlib import Path
from typing import Optional

# Chakra protobuf
from chakra.src.third_party.utils.protolib import (
    decodeMessage as _decode_msg,
    encodeMessage as _encode_msg,
)
from chakra.schema.protobuf.et_def_pb2 import (
    GlobalMetadata, Node, AttributeProto,
    COMP_NODE, COMM_COLL_NODE, ALL_REDUCE,
)


# =====================================================================
# ET I/O (same pattern as conver_to_chakra_et.py)
# =====================================================================

def _decode_et(et_path: Path):
    """Decode .et file → (GlobalMetadata, list[Node])"""
    size = et_path.stat().st_size
    with et_path.open("rb") as f:
        meta = GlobalMetadata()
        _decode_msg(f, meta)
        nodes = []
        while f.tell() < size:
            n = Node()
            try:
                _decode_msg(f, n)
            except Exception:
                break
            nodes.append(n)
    return meta, nodes


def _encode_et(et_path: Path, meta: GlobalMetadata, nodes: list):
    """Encode (GlobalMetadata, list[Node]) → .et file"""
    with et_path.open("wb") as f:
        _encode_msg(f, meta)
        for n in nodes:
            _encode_msg(f, n)


# =====================================================================
# Helper: get/set node attributes
# =====================================================================

def _get_attr(node: Node, name: str, default=None):
    """Get attribute value from node by name."""
    for a in node.attr:
        if a.name == name:
            if a.HasField('int64_val'):
                return a.int64_val
            if a.HasField('uint64_val'):
                return a.uint64_val
            if a.HasField('double_val'):
                return a.double_val
            if a.HasField('bool_val'):
                return a.bool_val
            if a.HasField('string_val'):
                return a.string_val
    return default


def _set_attr_int(node: Node, name: str, value: int):
    """Set or add int64 attribute."""
    for a in node.attr:
        if a.name == name:
            a.int64_val = int(value)
            return
    attr = node.attr.add()
    attr.name = name
    attr.int64_val = int(value)


# =====================================================================
# Auto-detect model info
# =====================================================================

def _detect_model_info(et_file: Path, model_tag: Optional[str] = None,
                       base_dir: Optional[Path] = None,
                       model_name: Optional[str] = None,
                       models_dir: Optional[Path] = None) -> Optional[dict]:
    """
    Auto-detect model parameters. Search order:
    1. data/models/ — read config from cached HuggingFace model (no download needed)
    2. --model-name — load config from HuggingFace Hub (download config only, ~1 KB)
    """
    import json

    # ---- Infer models_dir from et_file path ----
    # et_file is in data/chakra/workload_et/, models/ is at data/models/
    if models_dir is None:
        if base_dir:
            models_dir = Path(base_dir).parent / "models"
        else:
            models_dir = et_file.parent.parent.parent / "models"

    # ---- Infer model_name from model_tag ----
    # Common mappings: qwen15b_tp → Qwen/Qwen2.5-1.5B, qwen05b → Qwen/Qwen2.5-0.5B
    tag_to_name = {
        "qwen15b_tp": "Qwen/Qwen2.5-1.5B",
        "qwen15b": "Qwen/Qwen2.5-1.5B",
        "qwen05b": "Qwen/Qwen2.5-0.5B",
        "qwen3b": "Qwen/Qwen2.5-3B",
        "llama1b": "meta-llama/Llama-3.2-1B",
    }

    if model_tag is None:
        parts = et_file.stem.split('.')
        if len(parts) >= 3:
            model_tag = parts[1]

    if model_name is None and model_tag:
        # Strip _tp suffix for lookup
        base_tag = model_tag.replace("_tp", "")
        model_name = tag_to_name.get(model_tag) or tag_to_name.get(base_tag)

    if model_name is None:
        return None

    # ---- Method 1: Load model on meta device from cache (exact, no memory, no download) ----
    if models_dir and models_dir.exists():
        try:
            from transformers import AutoModelForCausalLM
            import torch
            model = AutoModelForCausalLM.from_pretrained(
                model_name, dtype=torch.float16,
                cache_dir=str(models_dir), device_map="meta",
            )
            n_params = sum(p.numel() for p in model.parameters())
            config = model.config
            print(f"[add-ddp] Read model from cache: {models_dir}")
            print(f"[add-ddp] Model: {model_name}, exact params: {n_params:,}")
            result = {
                "model_name": model_name,
                "n_params": n_params,
                "n_layers": getattr(config, 'num_hidden_layers', 0),
                "hidden_size": getattr(config, 'hidden_size', 0),
                "param_dtype_bytes": 2,
                "tp_degree": 2,
            }
            del model
            return result
        except Exception as e:
            print(f"[add-ddp] Cache read failed ({e}), trying hub...")

    # ---- Method 2: Load from HuggingFace Hub (no weight download if cached) ----
    try:
        from transformers import AutoModelForCausalLM
        import torch
        print(f"[add-ddp] Loading model structure: {model_name}...")
        model = AutoModelForCausalLM.from_pretrained(
            model_name, dtype=torch.float16, device_map="meta",
        )
        n_params = sum(p.numel() for p in model.parameters())
        config = model.config
        print(f"[add-ddp] Exact params: {n_params:,}")
        result = {
            "model_name": model_name,
            "n_params": n_params,
            "n_layers": getattr(config, 'num_hidden_layers', 0),
            "hidden_size": getattr(config, 'hidden_size', 0),
            "param_dtype_bytes": 2,
            "tp_degree": 2,
        }
        del model
        return result
    except ImportError:
        print(f"[add-ddp] ⚠️ transformers not installed")
    except Exception as e:
        print(f"[add-ddp] ⚠️ Hub load failed: {e}")

    return None


# =====================================================================
# Core: Add DDP AllReduce nodes
# =====================================================================

def _create_ddp_allreduce_node(node_id: int, bucket_idx: int,
                                comm_size: int, dep_id: int) -> Node:
    """Create a DDP AllReduce COMM node with involved_dim=[true,true,false] (Dim 0+1 only)."""
    n = Node()
    n.id = node_id
    n.name = f"DDP_AllReduce_bucket_{bucket_idx}"
    n.type = COMM_COLL_NODE

    # Required attributes for ASTRA-sim
    n.attr.append(AttributeProto(name="is_cpu_op", bool_val=False))
    n.attr.append(AttributeProto(name="comm_type", int64_val=ALL_REDUCE))
    n.attr.append(AttributeProto(name="comm_size", int64_val=comm_size))
    n.attr.append(AttributeProto(name="group_id", int64_val=1))  # group 1 = DDP

    # involved_dim: DDP runs on Dim 0 + Dim 1 (inter-server, X/Y axes, 25G)
    # Logical [4,4,8]: Dim 0 = X (DDP), Dim 1 = Y (DDP), Dim 2 = Z (TP, 65G)
    involved_attr = AttributeProto(name="involved_dim")
    involved_attr.bool_list.values.append(True)   # Dim 0 (X): run DDP
    involved_attr.bool_list.values.append(True)   # Dim 1 (Y): run DDP
    involved_attr.bool_list.values.append(False)  # Dim 2 (Z): skip (TP only)
    n.attr.append(involved_attr)

    # Dependency: chain buckets sequentially, first bucket depends on last COMP
    n.data_deps.append(dep_id)

    return n


def add_ddp_to_tp_et(
    et_file: Path,
    target_tp: int = 8,
    trace_tp: int = 2,
    model_params: Optional[int] = None,
    param_dtype_bytes: int = 2,
    ddp_gradient_bytes: Optional[int] = None,
    ddp_buckets: Optional[int] = None,
    bucket_size_bytes: int = 25_000_000,
    output_file: Optional[Path] = None,
    model_tag: Optional[str] = None,
    base_dir: Optional[Path] = None,
    model_name: Optional[str] = None,
    models_dir: Optional[Path] = None,
) -> dict:
    """
    Add DDP AllReduce nodes to a TP-only ET and scale COMP times.

    Args:
        et_file: Input .et file path
        target_tp: Target TP degree for simulation (e.g. 8)
        trace_tp: TP degree used when collecting the trace (auto-detected or default 2)
        model_params: Number of model parameters (auto-detected from model_info JSON if None)
        param_dtype_bytes: Bytes per parameter (auto-detected or default 2 for FP16)
        ddp_gradient_bytes: Total gradient bytes for DDP AllReduce (overrides auto-compute)
        ddp_buckets: Number of DDP buckets (overrides auto-compute)
        bucket_size_bytes: Target bucket size in bytes (default 25 MB, PyTorch default)
        output_file: Output path (default: input with _tp{N}ddp suffix)
        model_tag: Model tag for auto-detecting model_info JSON
        base_dir: Base directory for finding gpu_metrics/

    Returns:
        dict with stats: {comp_scaled, ddp_nodes_added, gradient_bytes, ...}
    """
    et_file = Path(et_file)

    # ---- Auto-detect model params if not provided ----
    if ddp_gradient_bytes is None and model_params is None:
        detected = _detect_model_info(et_file, model_tag, base_dir, model_name, models_dir)
        if detected:
            model_params = detected['n_params']
            param_dtype_bytes = detected['param_dtype_bytes']
            trace_tp = detected.get('tp_degree', trace_tp)
            print(f"[add-ddp] Auto-detected: {model_params:,} params, "
                  f"{param_dtype_bytes} bytes/param, trace_tp={trace_tp}")
        else:
            raise ValueError(
                "Cannot auto-detect model params. Provide one of:\n"
                "  --model-name Qwen/Qwen2.5-1.5B  (reads config from cache or hub)\n"
                "  --model-params N                  (manual)\n"
                "  --ddp-gradient-bytes N             (manual)\n"
                "Or ensure the model was downloaded to data/models/ by train_rocm_tensor.py"
            )

    # ---- Compute gradient bytes ----
    if ddp_gradient_bytes is not None:
        grad_bytes = ddp_gradient_bytes
    else:
        grad_bytes = model_params * param_dtype_bytes

    # ---- Compute bucket count ----
    if ddp_buckets is not None:
        n_buckets = ddp_buckets
    else:
        n_buckets = max(1, math.ceil(grad_bytes / bucket_size_bytes))

    bucket_bytes = grad_bytes // n_buckets
    last_bucket_bytes = grad_bytes - bucket_bytes * (n_buckets - 1)

    # ---- COMP scaling factor ----
    comp_scale = trace_tp / target_tp  # e.g. 2/8 = 0.25

    print(f"[add-ddp] Input: {et_file.name}")
    print(f"[add-ddp] TP scaling: trace_tp={trace_tp} → target_tp={target_tp} (COMP × {comp_scale})")
    print(f"[add-ddp] DDP gradient: {grad_bytes:,} bytes ({grad_bytes / 1024**3:.2f} GiB)")
    print(f"[add-ddp] DDP buckets: {n_buckets} × {bucket_bytes:,} bytes ({bucket_bytes / 1024**2:.1f} MiB)")

    # ---- Load ET ----
    meta, nodes = _decode_et(et_file)

    # ---- Stats before modification ----
    n_comp_before = sum(1 for n in nodes if n.type == COMP_NODE)
    n_comm_before = sum(1 for n in nodes if n.type == COMM_COLL_NODE)

    # ---- Scale COMP times ----
    comp_scaled = 0
    if comp_scale != 1.0:
        for n in nodes:
            if n.type == COMP_NODE:
                # Scale compute_cycles
                for a in n.attr:
                    if a.name in ("compute_cycles", "exec_cycles", "cycles"):
                        old_val = a.int64_val
                        a.int64_val = max(1, int(old_val * comp_scale))
                        comp_scaled += 1
                        break
    print(f"[add-ddp] COMP nodes scaled: {comp_scaled} (× {comp_scale})")

    # ---- Tag TP COMM nodes with involved_dim=[false,false,true] ----
    # Logical [4,4,8]: TP AllReduce only runs on Dim 2 (Z-axis, intra-server, 65G)
    tp_tagged = 0
    for n in nodes:
        if n.type == COMM_COLL_NODE:
            # Check if involved_dim already exists
            has_involved = any(a.name == "involved_dim" for a in n.attr)
            if not has_involved:
                involved_attr = AttributeProto(name="involved_dim")
                involved_attr.bool_list.values.append(False)  # Dim 0 (X): skip (DDP only)
                involved_attr.bool_list.values.append(False)  # Dim 1 (Y): skip (DDP only)
                involved_attr.bool_list.values.append(True)   # Dim 2 (Z): run TP AllReduce here
                n.attr.append(involved_attr)
                tp_tagged += 1
    print(f"[add-ddp] TP COMM nodes tagged with involved_dim=[false,false,true]: {tp_tagged}")

    # ---- Find insertion point: last node ID ----
    max_id = max(n.id for n in nodes)
    # Find last COMP node (end of backward pass)
    last_comp_id = None
    for n in reversed(nodes):
        if n.type == COMP_NODE:
            last_comp_id = n.id
            break

    if last_comp_id is None:
        print(f"[add-ddp] ⚠️ No COMP nodes found, appending DDP after last node")
        last_comp_id = max_id

    # ---- Add DDP AllReduce nodes ----
    new_nodes = []
    prev_dep_id = last_comp_id

    for i in range(n_buckets):
        node_id = max_id + 1 + i
        bkt_size = last_bucket_bytes if (i == n_buckets - 1) else bucket_bytes

        ddp_node = _create_ddp_allreduce_node(
            node_id=node_id,
            bucket_idx=i + 1,
            comm_size=bkt_size,
            dep_id=prev_dep_id,
        )
        new_nodes.append(ddp_node)
        prev_dep_id = node_id  # Chain: bucket 1 → bucket 2 → ...

    nodes.extend(new_nodes)

    # ---- Determine output path ----
    # Must keep et.{tag}.{rank}.et format for run_ns3.py compatibility
    # Input:  et.qwen15b_tp.0.et  (tag=qwen15b_tp, rank=0)
    # Output: et.qwen15b_tp8ddp.0.et  (tag=qwen15b_tp8ddp, rank=0)
    if output_file is None:
        import re
        m = re.match(r'^et\.(.+)\.(\d+)$', et_file.stem)
        if m:
            orig_tag, rank_str = m.group(1), m.group(2)
            # qwen15b_tp → qwen15b → qwen15b_tp8ddp
            base_tag = orig_tag.removesuffix('_tp') if orig_tag.endswith('_tp') else orig_tag
            new_tag = f"{base_tag}_tp{target_tp}ddp"
            output_file = et_file.parent / f"et.{new_tag}.{rank_str}.et"
        else:
            raise ValueError(
                f"ET filename '{et_file.name}' does not match expected pattern 'et.{{tag}}.{{rank}}.et'"
            )
    output_file = Path(output_file)

    if output_file.exists():
        print(f"[add-ddp] Overwriting existing: {output_file.name}")

    # ---- Save ----
    _encode_et(output_file, meta, nodes)

    # ---- Stats ----
    n_comp_after = sum(1 for n in nodes if n.type == COMP_NODE)
    n_comm_after = sum(1 for n in nodes if n.type == COMM_COLL_NODE)

    stats = {
        'input_file': str(et_file),
        'output_file': str(output_file),
        'target_tp': target_tp,
        'trace_tp': trace_tp,
        'comp_scale': comp_scale,
        'comp_scaled': comp_scaled,
        'gradient_bytes': grad_bytes,
        'n_buckets': n_buckets,
        'bucket_bytes': bucket_bytes,
        'ddp_nodes_added': len(new_nodes),
        'comp_before': n_comp_before,
        'comm_before': n_comm_before,
        'comp_after': n_comp_after,
        'comm_after': n_comm_after,
    }

    print(f"[add-ddp] ✅ Added {len(new_nodes)} DDP AllReduce nodes")
    print(f"[add-ddp]   Before: {n_comp_before} COMP + {n_comm_before} COMM")
    print(f"[add-ddp]   After:  {n_comp_after} COMP + {n_comm_after} COMM")
    print(f"[add-ddp]   Output: {output_file}")

    return stats


# =====================================================================
# CLI
# =====================================================================

def main():
    ap = argparse.ArgumentParser(
        description="Add DDP AllReduce to TP Execution Trace & scale COMP for target TP degree",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Auto-detect from model_info JSON (recommended)
  python src/add_ddp_to_et.py \\
    --et-dir ./data/chakra/workload_et --model-tag qwen15b_tp \\
    --target-tp 8

  # Auto-detect from HuggingFace model name (no weight download)
  python src/add_ddp_to_et.py \\
    --et-dir ./data/chakra/workload_et --model-tag qwen15b_tp \\
    --target-tp 8 --model-name Qwen/Qwen2.5-1.5B

  # Manual: specify model params
  python src/add_ddp_to_et.py \\
    --et-file ./data/chakra/workload_et/et.qwen15b_tp.0.et \\
    --target-tp 8 --model-params 1543714304 --param-dtype fp16
        """
    )

    # Input
    ap.add_argument("--et-file", type=str, default=None,
                    help="Single .et file to process")
    ap.add_argument("--et-dir", type=str, default=None,
                    help="Directory containing .et files (process all matching model-tag)")
    ap.add_argument("--model-tag", type=str, default="qwen15b_tp",
                    help="Model tag for filtering .et files and auto-detecting model_info")
    ap.add_argument("--base-dir", type=str, default="./data/chakra",
                    help="Base directory (for finding gpu_metrics/model_info_*.json)")
    ap.add_argument("--model-name", type=str, default=None,
                    help="HuggingFace model name for auto-detecting params "
                         "(e.g. Qwen/Qwen2.5-1.5B)")
    ap.add_argument("--models-dir", type=str, default="./data/models",
                    help="Directory where HuggingFace models are cached "
                         "(default: ./data/models/)")

    # TP scaling
    ap.add_argument("--target-tp", type=int, required=True,
                    help="Target TP degree for simulation (e.g. 2, 4, 8)")
    ap.add_argument("--trace-tp", type=int, default=2,
                    help="TP degree used when collecting the trace (default: 2)")

    # DDP params
    ap.add_argument("--model-params", type=int, default=None,
                    help="Number of model parameters (for auto-computing gradient bytes)")
    ap.add_argument("--param-dtype", type=str, default="fp16", choices=["fp16", "fp32"],
                    help="Parameter dtype for gradient size calculation")
    ap.add_argument("--ddp-gradient-bytes", type=int, default=None,
                    help="Override: total gradient bytes for DDP AllReduce")
    ap.add_argument("--ddp-buckets", type=int, default=None,
                    help="Override: number of DDP buckets")
    ap.add_argument("--bucket-size-mb", type=float, default=25.0,
                    help="Target bucket size in MB (default: 25, PyTorch default)")

    # Output
    ap.add_argument("--output-file", type=str, default=None,
                    help="Output .et file path (default: auto-generate)")

    args = ap.parse_args()

    param_dtype_bytes = 2 if args.param_dtype == "fp16" else 4
    bucket_size_bytes = int(args.bucket_size_mb * 1024 * 1024)

    # Determine files to process
    if args.et_file:
        files = [Path(args.et_file)]
    elif args.et_dir:
        et_dir = Path(args.et_dir)
        # Only match original ET files: et.{tag}.{rank}.et (rank = pure digit)
        # Exclude derived files like et.{tag}.0_tp8ddp.et
        import re
        rank_pat = re.compile(rf'^et\.{re.escape(args.model_tag)}\.(\d+)\.et$')
        files = sorted([f for f in et_dir.glob(f"et.{args.model_tag}.*.et")
                        if rank_pat.match(f.name)])
        if not files:
            print(f"[error] No original .et files matching 'et.{args.model_tag}.{{rank}}.et' in {et_dir}")
            print(f"[hint] Found files: {[f.name for f in et_dir.glob(f'et.{args.model_tag}.*.et')]}")
            sys.exit(1)
        print(f"[add-ddp] Found {len(files)} ET files: {[f.name for f in files]}")
    else:
        print("[error] Must provide either --et-file or --et-dir")
        sys.exit(1)

    # Process each file
    for et_file in files:
        print(f"\n{'='*60}")
        output = Path(args.output_file) if args.output_file and len(files) == 1 else None
        stats = add_ddp_to_tp_et(
            et_file=et_file,
            target_tp=args.target_tp,
            trace_tp=args.trace_tp,
            model_params=args.model_params,
            param_dtype_bytes=param_dtype_bytes,
            ddp_gradient_bytes=args.ddp_gradient_bytes,
            ddp_buckets=args.ddp_buckets,
            bucket_size_bytes=bucket_size_bytes,
            output_file=output,
            model_tag=args.model_tag,
            base_dir=Path(args.base_dir) if args.base_dir else None,
            model_name=args.model_name,
            models_dir=Path(args.models_dir) if args.models_dir else None,
        )
    print(f"\n{'='*60}")
    print(f"[done] All files processed.")


if __name__ == "__main__":
    main()
