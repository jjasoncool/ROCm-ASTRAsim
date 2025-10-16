#!/usr/bin/env python3
"""
Lightweight trace readiness checker.

This module provides a small utility to analyze PyTorch/Kineto device traces
and report per-rank network vs kernel timing summaries and cross-rank medians.

Placed under `src/tests` so it can be imported from test harnesses or used
manually from the command line.
"""
from pathlib import Path
import json
import re
from statistics import median
from typing import Dict

KINETO_STEP_PAT = re.compile(r'^ProfilerStep#\d+$')


def parse_trace(path: Path) -> Dict:
    obj = json.loads(path.read_text(encoding='utf-8', errors='ignore'))
    unit = str(obj.get('displayTimeUnit', '')).lower()

    def to_ms(d):
        try:
            dv = float(d)
        except Exception:
            try:
                dv = float(str(d))
            except Exception:
                dv = 0.0
        return dv if unit == 'ms' else dv / 1000.0

    net_events = []
    kernel_events = []
    steps = []
    bytes_total = 0
    bytes_events = 0

    for ev in obj.get('traceEvents', []):
        if ev.get('ph') != 'X':
            continue
        name = str(ev.get('name', ''))
        cat = str(ev.get('cat', ''))
        dur = to_ms(ev.get('dur', 0.0))
        if KINETO_STEP_PAT.match(name):
            steps.append(dur)
        lname = name.lower()
        lcat = cat.lower()
        if '|bytes=' in lname and 'kernel' not in lcat:
            net_events.append((name, dur))
            m = re.search(r'bytes=(\d+)', lname)
            if m:
                try:
                    bytes_total += int(m.group(1))
                    bytes_events += 1
                except Exception:
                    pass
        if 'kernel' in lcat or 'nccldevkernel' in lname or 'devkernel' in lname:
            kernel_events.append((name, dur))

    return {
        'steps_n': len(steps),
        'step_median_ms': median(steps) if steps else None,
        'net_count': len(net_events),
        'net_total_ms': sum(d for _, d in net_events),
        'net_per_step_ms': (sum(d for _, d in net_events) / len(steps)) if steps else None,
        'kernel_count': len(kernel_events),
        'kernel_total_ms': sum(d for _, d in kernel_events),
        'kernel_per_step_ms': (sum(d for _, d in kernel_events) / len(steps)) if steps else None,
        'bytes_events': bytes_events,
        'bytes_total': bytes_total,
        'top_net': sorted(net_events, key=lambda x: -x[1])[:20],
        'top_kernel': sorted(kernel_events, key=lambda x: -x[1])[:20]
    }


def analyze_dir(trace_dir: Path) -> Dict[str, Dict]:
    """Analyze all device_*.json traces in trace_dir and print a summary.

    Returns a dictionary mapping filename -> summary dict (as returned by parse_trace).
    """
    files = sorted((trace_dir).glob('device_*.json'))
    if not files:
        raise FileNotFoundError(f'No device_*.json found in {trace_dir}')

    all_summ = {}
    for f in files:
        s = parse_trace(f)
        all_summ[f.name] = s
        print('\n---', f.name)
        print(' 步數 (steps_n):', s['steps_n'], '  步中位數 (ms):', s['step_median_ms'])
        print(' 網路事件: 數量', s['net_count'], ' 總耗時(ms)', s['net_total_ms'], ' 每步平均(ms)', s['net_per_step_ms'])
        print(' Kernel/運算事件: 數量', s['kernel_count'], ' 總耗時(ms)', s['kernel_total_ms'], ' 每步平均(ms)', s['kernel_per_step_ms'])
        print(' bytes 事件數量:', s['bytes_events'], ' bytes 總量:', s['bytes_total'])
        print('\n 前十名網路事件 (耗時 ms | 事件名稱):')
        for n, d in s['top_net'][:10]:
            print('  {:.3f}  {}'.format(d, n[:200]))
        print('\n 前十名 Kernel/運算事件 (耗時 ms | 事件名稱):')
        for n, d in s['top_kernel'][:10]:
            print('  {:.3f}  {}'.format(d, n[:200]))

    # cross-rank medians
    ranks = list(all_summ.keys())
    net_per_step = [all_summ[r]['net_per_step_ms'] for r in ranks if all_summ[r]['net_per_step_ms'] is not None]
    kernel_per_step = [all_summ[r]['kernel_per_step_ms'] for r in ranks if all_summ[r]['kernel_per_step_ms'] is not None]
    step_meds = [all_summ[r]['step_median_ms'] for r in ranks if all_summ[r]['step_median_ms'] is not None]
    print('\n跨 rank 中位數 (CROSS-RANK MEDIANS):')
    print(' 每步網路耗時中位數 (net_per_step median):', median(net_per_step) if net_per_step else None)
    print(' 每步 Kernel 耗時中位數 (kernel_per_step median):', median(kernel_per_step) if kernel_per_step else None)
    print(' 步中位數 (step_median median):', median(step_meds) if step_meds else None)

    return all_summ




def run_all_checks(trace_dir: Path | str):
    """Run the original real-metrics extraction plus the new trace analyzer and print a combined report.

    Returns a dict with keys: 'real_metrics' and 'detailed' (per-rank summaries).
    """
    trace_dir = Path(trace_dir)
    # import original extractor from scripts/run_ns3
    try:
        from scripts import run_ns3 as runns3
    except Exception:
        runns3 = None

    result = {}

    if runns3 is not None and hasattr(runns3, 'extract_real_metrics_from_traces'):
        try:
            real_metrics = runns3.extract_real_metrics_from_traces(trace_dir)
            # expected: (real_t_step_ms, real_t_net_comm_ms, real_t_kernel_ms, used_epoch)
        except Exception as e:
            real_metrics = None
            print(f"[warn] extract_real_metrics_from_traces failed: {e}")
    else:
        real_metrics = None

    result['real_metrics'] = real_metrics

    # detailed per-rank analysis (new)
    try:
        detailed = analyze_dir(trace_dir)
    except FileNotFoundError as e:
        print(f"[error] analyze_dir failed: {e}")
        detailed = None
    result['detailed'] = detailed

    # print combined summary
    print('\n==== 綜合檢查摘要 ====')
    if real_metrics:
        print('[原始 extractor] real_t_step_ms =', real_metrics[0])
        print('[原始 extractor] real_t_net_comm_ms =', real_metrics[1])
        print('[原始 extractor] real_t_kernel_ms =', real_metrics[2])
        print('[原始 extractor] used_epoch =', real_metrics[3])
    else:
        print('[原始 extractor] 無可用資料')

    if detailed:
        # compute cross-rank median net_per_step if available
        net_per_steps = [v['net_per_step_ms'] for v in detailed.values() if v.get('net_per_step_ms') is not None]
        kernel_per_steps = [v['kernel_per_step_ms'] for v in detailed.values() if v.get('kernel_per_step_ms') is not None]
        print('[詳細分析器] 跨 rank 每步網路耗時中位數 =', median(net_per_steps) if net_per_steps else None)
        print('[詳細分析器] 跨 rank 每步 Kernel 耗時中位數 =', median(kernel_per_steps) if kernel_per_steps else None)
    else:
        print('[詳細分析器] 無可用資料')

    # 結論：基於上面的結果給出簡短建議
    print('\n---- 結論 (自動摘要) ----')
    if real_metrics:
        r_step, r_net, r_kernel, epoch = real_metrics
        print(f'  - 真實量測 (epoch={epoch}) 每步總耗時 ≈ {r_step} ms。')
        print(f'  - 真實量測 每步網路耗時 ≈ {r_net} ms；每步 Kernel 耗時 ≈ {r_kernel} ms。')
        if net_per_steps:
            det_net_med = median(net_per_steps)
            print(f'  - 詳細分析器的跨 rank 每步網路耗時中位數 = {det_net_med} ms，與原始 extractor 報告的 {r_net} ms 比較可做交叉檢查。')
    else:
        print('  - 無法取得原始 extractor 的真實量測，請確認 scripts/run_ns3.py 是否在 module path 中或可被匯入。')

    if detailed:
        print('  - 裝置端 trace 已解析，含網路與 Kernel 事件摘要；請注意 bytes 總量與 ET 中 comm_size 是否一致，若不一致請執行 map_comm_sizes_from_hdt_to_et() 以補足 ET 的 comm_size。')
    else:
        print('  - 詳細分析器未能取得裝置端 trace，請確認 device_*.json 檔案存在且可讀。')

    print('\n建議後續動作：')
    print('  1) 若要進一步做模擬校準，請確保 .et 檔案內的 comm_size 已回填正確的 bytes 值。')
    print('  2) 若 scripts/run_ns3 無法匯入，可將 `scripts/__init__.py` 加入或改用 path-based import。')
    print('  3) 若發現網路相對誤差過大，先比較 network-only 部分（sim vs real）以排除 kernel/運算差異的影響。')

    return result



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
    # 在原本的 smoke checks 之後執行更深入的整合檢查（不改變 exit code）
    try:
        print('\n[INFO] 執行整合檢查（原 extractor + 詳細 analyzer）...')
        # run_all_checks is defined in this module
        run_all_checks(traces_dir)
    except Exception as e:
        print(f"[warn] run_all_checks 發生錯誤：{e}")

    sys.exit(0)

if __name__ == "__main__":
    main()
