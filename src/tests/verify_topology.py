"""
ASTRA-sim 拓撲驗證工具 (Topology Verifier) - 全自動與效能分析版

功能概述：
    1. 讀取 .txt 拓撲檔，支援 Switch ID List 格式。
    2. [自動] 判斷 Fat-Tree 或 Torus。
    3. [自動] 逆向推算 Torus 維度。
    4. [新增] 效能檢查 (--check-perf): 計算網路直徑與平均跳數，用於驗證 Twisted Torus 效果。

使用範例:
    1. 全自動驗證 (推薦):
       python3 src/tests/verify_topology.py ../configs/astra-sim/topos/128nodes_Torus_4x4x8.txt

    2. 驗證並計算效能 (比較 Standard vs Twisted 時使用):
       python3 src/tests/verify_topology.py ../configs/astra-sim/topos/128nodes_TwistedTorus_4x4x8.txt --check-perf
"""

import argparse
import sys
from collections import Counter
from pathlib import Path

# 嘗試匯入 networkx 用於效能分析，若無則跳過
try:
    import networkx as nx
    HAS_NETWORKX = True
except ImportError:
    HAS_NETWORKX = False

def parse_astra_topology(file_path):
    path = Path(file_path)
    if not path.exists():
        print(f"[Fatal] 找不到檔案: {path}")
        sys.exit(1)

    with open(path, 'r') as f:
        lines = [l.strip() for l in f if l.strip()]

    # 1. Header
    try:
        header = lines[0].split()
        total_nodes = int(header[0])
        num_switches = int(header[1])
        num_gpus = int(header[2])
    except:
        print("[Fatal] Header 格式錯誤")
        sys.exit(1)

    # 2. Switch IDs
    switch_ids = set()
    try:
        raw = lines[1].split()
        if len(raw) > 0:
            switch_ids = set([int(x) for x in raw])
    except: pass

    if not switch_ids and num_switches > 0:
        switch_ids = set(range(num_gpus, num_gpus + num_switches))

    # 3. Links
    links = []
    for line in lines[2:]:
        parts = line.split()
        if len(parts) >= 4:
            links.append({
                "u": int(parts[0]),
                "v": int(parts[1]),
                "bw": parts[2],
                "lat": parts[3]
            })

    return {
        "total_nodes": total_nodes,
        "gpus": num_gpus,
        "switches": num_switches,
        "switch_ids": switch_ids,
        "links": links
    }

def auto_detect_type(data):
    """ 自動判斷拓撲類型 (依據是否有 Spine Switch) """
    print("--- [Auto] 正在分析拓撲類型... ---")
    switches_connected_to_gpu = set()
    num_gpus = data['gpus']

    for link in data['links']:
        u, v = link['u'], link['v']
        if u < num_gpus and v in data['switch_ids']:
            switches_connected_to_gpu.add(v)
        if v < num_gpus and u in data['switch_ids']:
            switches_connected_to_gpu.add(u)

    spine_switches = data['switch_ids'] - switches_connected_to_gpu

    if len(spine_switches) > 0:
        print(f"-> 偵測到 {len(spine_switches)} 個 Spine Switches (不連 GPU)。")
        print("-> 判定類型: Fat-Tree")
        return 'fattree'
    else:
        print(f"-> 所有 Switch 皆直接連接 GPU。")
        print("-> 判定類型: Torus / Ring")
        return 'torus'

def auto_infer_dims(data):
    """ 自動推斷 Torus 維度 (修正版 v2) """
    print("--- [Auto] 正在分析 Torus 維度... ---")
    deltas = []
    switches = data['switch_ids']

    for link in data['links']:
        u, v = link['u'], link['v']
        if u in switches and v in switches:
            delta = abs(u - v)
            if delta > 0: deltas.append(delta)

    if not deltas: return None

    counts = Counter(deltas)

    # [修正] 將門檻提高到 0.7 (70%)
    # 這能過濾掉雙向 Torus 中大量的邊界回繞線 (Wrap-around links)，避免誤判為額外維度
    num_switches = len(switches)
    threshold = num_switches * 0.7

    major_strides = sorted([d for d, c in counts.items() if c > threshold])

    if not major_strides:
        print("[Warning] 無法識別主軸步長，嘗試寬鬆模式...")
        major_strides = sorted([d for d, c in counts.items()], key=lambda x: counts[x], reverse=True)[:3]
        major_strides.sort()

    dims = []
    for i in range(len(major_strides) - 1):
        curr_s = major_strides[i]
        next_s = major_strides[i+1]
        d = next_s // curr_s
        dims.append(d)

    if major_strides:
        last_s = major_strides[-1]
        last_d = num_switches // last_s
        dims.append(last_d)

    print(f"-> 推算維度: {dims} (特徵主步長: {major_strides})")
    return dims

def calculate_performance(data):
    """ 計算網路直徑與平均跳數 (需 networkx) """
    print("--- 效能分析 (Performance Check) ---")
    if not HAS_NETWORKX:
        print("[Warning] 未安裝 networkx，無法計算網路直徑。請 pip install networkx。")
        return

    G = nx.Graph()
    # 僅建立 Switch 之間的互連圖來分析拓撲特性
    switches = data['switch_ids']
    for link in data['links']:
        u, v = link['u'], link['v']
        if u in switches and v in switches:
            G.add_edge(u, v)

    if nx.is_connected(G):
        print("-> 計算中，請稍候...")
        try:
            # 計算平均最短路徑長度 (Average Shortest Path Length)
            avg_path = nx.average_shortest_path_length(G)
            diameter = nx.diameter(G)
            print(f"-> [結果] 網路直徑 (Diameter): {diameter}")
            print(f"-> [結果] 平均跳數 (Avg Hops): {avg_path:.4f}")
            print("   (數值越小代表通訊效率越好，Twisted 應該優於 Standard)")
        except:
             print("[Error] 計算過久或圖形過大。")
    else:
        print("[Error] Switch 網路未連通，無法計算。")

def verify_basic(data):
    connected = set()
    for l in data['links']: connected.add(l['u']); connected.add(l['v'])
    missing = [i for i in range(data['total_nodes']) if i not in connected]

    if missing: print(f"[Fail] 發現孤島節點: {missing}")
    else: print(f"[Pass] 所有節點皆有連線。")

def verify_torus(data):
    dims = auto_infer_dims(data)
    if not dims:
        print("[Warning] 無法推算維度，僅執行基礎檢查。")
        verify_basic(data)
        return

    # [修正邏輯] 支援單向環與雙向環
    # Unidirectional (ASTRA-sim Default): Degree = len(dims) + 1 (GPU)
    # Bidirectional: Degree = len(dims)*2 + 1 (GPU)
    expected_uni = len(dims) + 1
    expected_bi = (len(dims) * 2) + 1

    out_degree = Counter([l['u'] for l in data['links']])

    errors = 0
    valid_degrees = [expected_uni, expected_bi]

    for s in data['switch_ids']:
        deg = out_degree[s]
        if deg not in valid_degrees:
            errors += 1

    if errors == 0:
        print(f"[Pass] Torus 結構正確 (Switch Degree 符合預期 {valid_degrees})。")
    else:
        print(f"[Fail] {errors} 個 Switch 連線數異常 (預期 {valid_degrees})。")

def verify_fattree(data):
    verify_basic(data)
    out_degree = Counter([l['u'] for l in data['links']])

    errors = 0
    for i in range(data['gpus']):
        if out_degree[i] != 1: errors += 1

    if errors == 0: print(f"[Pass] Fat-Tree 葉節點正確 (所有 GPU 僅有 1 條上行)。")
    else: print(f"[Fail] {errors} 個 GPU 連線異常。")

def main():
    parser = argparse.ArgumentParser(formatter_class=argparse.RawTextHelpFormatter)
    parser.add_argument("file", help="拓撲檔案路徑")
    parser.add_argument("--type", choices=['auto', 'torus', 'fattree'], default='auto', help="驗證模式 (預設自動偵測)")
    parser.add_argument("--check-perf", action="store_true", help="啟用效能分析 (計算網路直徑與平均跳數)")

    args = parser.parse_args()

    print(f"讀取檔案: {args.file}")
    data = parse_astra_topology(args.file)
    print(f"基本資訊: {data['total_nodes']} Nodes, {len(data['links'])} Links")

    mode = args.type
    if mode == 'auto':
        mode = auto_detect_type(data)

    print("-" * 30)

    if mode == 'torus':
        verify_torus(data)
    elif mode == 'fattree':
        verify_fattree(data)
    else:
        verify_basic(data)

    if args.check_perf:
        calculate_performance(data)

if __name__ == "__main__":
    main()
