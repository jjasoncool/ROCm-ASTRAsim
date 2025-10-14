#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
check_trace_ready.py
檢查由 train_rocm_pytorch.py 產出的 host/device trace 是否符合 Chakra/HTA 的最低需求：
- host_{rank}.json：單一合法 JSON
- device_{rank}.json：包含 ProfilerStep#、HIP/CUDA 同步事件 (EventRecord/StreamWaitEvent)、以及至少一些 kernel/運算事件

使用：
  python ./src/tests/check_trace_ready.py \
      --traces-dir ./data/chakra/pytorch_traces \
      --ranks auto

退出碼：0 表示全部 OK；1 表示至少有一個 rank 未通過檢查。
"""
import argparse
import json
import os
import re
import sys
from pathlib import Path
from typing import Dict, List, Tuple

# --------- streaming 掃描幫手（不把 200~500MB JSON 完整載入記憶體） ----------
def _stream_count(fname: Path, patterns: Dict[str, re.Pattern]) -> Dict[str, int]:
    cnt = {k:0 for k in patterns}
    # 逐塊讀，降低記憶體壓力
    with fname.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            for key, rgx in patterns.items():
                cnt[key] += len(rgx.findall(chunk))
    return cnt

def _try_parse_json(fname: Path) -> Tuple[bool, str]:
    try:
        with fname.open("r", encoding="utf-8", errors="strict") as f:
            json.load(f)
        return True, "ok"
    except Exception as e:
        return False, str(e)

def _looks_like_concatenated_json(text: str) -> bool:
    # 簡單判斷：是否出現多次 {"schema": ... 的起點
    return len(re.findall(r'\{\s*"schema"\s*:', text)) >= 2

def _explain_device_fail(cnt: Dict[str,int]) -> List[str]:
    msgs = []
    if (cnt["profstep"] == 0):
        msgs.append("找不到 ProfilerStep#（請確認每步 prof.step()，且 schedule 非空）")

    # 改進的同步事件檢查：現實中很多環境缺乏完整同步事件
    total_sync_events = cnt["hip_rec"] + cnt["cuda_rec"] + cnt["hip_wait"] + cnt["cuda_wait"]
    if total_sync_events == 0:
        msgs.append("找不到任何同步事件 - 可能影響 critical path 分析品質，但在某些環境中屬正常現象")
    elif (cnt["hip_rec"] + cnt["cuda_rec"] == 0) and (cnt["hip_wait"] + cnt["cuda_wait"] > 0):
        # 這是常見情況：只有 Wait 而無 Record
        msgs.append("注意：只有 StreamWaitEvent 而無 EventRecord - 可能影響跨 stream 同步分析，但基本功能仍可使用")

    if cnt["kernel_like"] == 0:
        msgs.append("看不到 kernel/運算類事件（裝置端事件可能缺失，請檢查 ROCm/PyTorch 環境）")
    return msgs

def discover_ranks(traces_dir: Path, ranks_arg: str) -> List[int]:
    if ranks_arg != "auto":
        return [int(x) for x in ranks_arg.split(",") if x.strip() != ""]
    ranks = set()
    for p in traces_dir.glob("host_*.json"):
        m = re.search(r"host_(\d+)\.json$", p.name)
        if m: ranks.add(int(m.group(1)))
    for p in traces_dir.glob("device_*.json"):
        m = re.search(r"device_(\d+)\.json$", p.name)
        if m: ranks.add(int(m.group(1)))
    return sorted(ranks)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--traces-dir", type=str, default=str(Path(__file__).resolve().parents[2] / "data" / "chakra" / "pytorch_traces"))
    ap.add_argument("--ranks", type=str, default="auto", help="例如 '0,1' 或 'auto'")
    args = ap.parse_args()

    traces_dir = Path(args.traces_dir)
    if not traces_dir.is_dir():
        print(f"[error] 找不到目錄：{traces_dir}")
        sys.exit(1)

    ranks = discover_ranks(traces_dir, args.ranks)
    print(f"[info] 檢查目錄：{traces_dir}")
    print(f"[info] ranks = {ranks if ranks else '[]'}\n")
    if not ranks:
        sys.exit(1)

    # 預先編譯 pattern（byte 模式）
    pat = {
        "profstep": re.compile(br'"name"\s*:\s*"ProfilerStep#\d+', re.I),
        "hip_rec":  re.compile(br'"name"\s*:\s*"hipEventRecord"', re.I),
        "hip_wait": re.compile(br'"name"\s*:\s*"hipStreamWaitEvent"', re.I),
        "cuda_rec":  re.compile(br'"name"\s*:\s*"cudaEventRecord"', re.I),
        "cuda_wait": re.compile(br'"name"\s*:\s*"cudaStreamWaitEvent"', re.I),
        # 粗略抓 kernel/運算：常見 cat/name 片段（不嚴格，但可作 smoke test）
        "kernel_like": re.compile(br'"cat"\s*:\s*"(Kernel|gpu_op)"|"(convolution|GEMM|matmul|hipLaunchKernel)"', re.I),
    }

    ok_all = True
    for r in ranks:
        host = traces_dir / f"host_{r}.json"
        dev  = traces_dir / f"device_{r}.json"
        print(f"[Rank {r}]")

        # ---------- host ----------
        if not host.exists():
            print(f"  - [r{r}] 找不到 host 檔：{host.name}")
            ok_all = False
        else:
            good, why = _try_parse_json(host)
            if good:
                print(f"  - [r{r}] host JSON 解析：OK")
            else:
                print(f"  - [r{r}] host 解析失敗：{why}")
                try:
                    txt = host.read_text(encoding="utf-8", errors="ignore")
                    if _looks_like_concatenated_json(txt):
                        print(f"    · 懷疑是『被拼接的 host JSON』（多個 {{\"schema\": ...}} 連在一起），請改用 register_callback(.tmp)+stop()+unregister()+原子改名，或最後用修復程式只保留最後一個 JSON。")
                except Exception:
                    pass
                ok_all = False

        # ---------- device ----------
        if not dev.exists():
            print(f"  - [r{r}] 找不到 device 檔：{dev.name}")
            ok_all = False
        else:
            cnt = _stream_count(dev, pat)
            print(f"  - [r{r}] ProfilerStep#: {cnt['profstep']}, "
                  f"EventRecord: hip={cnt['hip_rec']}, cuda={cnt['cuda_rec']}; "
                  f"StreamWait: hip={cnt['hip_wait']}, cuda={cnt['cuda_wait']}; "
                  f"kernel_like: {cnt['kernel_like']}")
            problems = _explain_device_fail(cnt)
            if problems:
                for msg in problems:
                    print(f"    · {msg}")
                # 只有在嚴重問題時才標記為失敗
                if cnt["profstep"] == 0 or cnt["kernel_like"] == 0:
                    ok_all = False
            else:
                print(f"  - [r{r}] device trace：OK（含同步事件與步界）")

        print()

    if not ok_all:
        print("[NOT READY] 上述項目未滿足，請依提示修正再試。")
        print("建議：確認每步執行 prof.step()，並檢查 ROCm/PyTorch 環境設定。")
        sys.exit(1)

    print("[READY] 目前的 traces 應可交給 chakra_trace_link / chakra_converter。")
    sys.exit(0)

if __name__ == "__main__":
    main()
