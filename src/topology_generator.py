"""
ASTRA-sim 拓撲與設定自動生成器 (Topology & Config Generator)

功能概述：
    此腳本為 ASTRA-sim 擴張實驗專用工具，能夠「一鍵生成」三種關鍵設定檔：
    1. 物理拓撲 (.txt): 定義 GPU 與 Switch 的連線、頻寬、延遲。
       - 支援 Standard 3D Torus (標準環面)
       - 支援 Twisted 3D Torus (扭曲環面 - Google TPU 風格)
       - 支援 Fat-Tree (胖樹 - NVIDIA SuperPOD 風格)
    2. 邏輯拓撲 (.json): 定義集體通訊 (Collective Communication) 的維度。
    3. 系統設定 (.json): 自動匹配對應的演算法 (Ring vs. HalvingDoubling) 與記憶體參數。

輸出路徑 (自動建立):
    - 拓撲檔: ../configs/astra-sim/topos/
    - 系統檔: ../configs/astra-sim/system/

使用範例 (在 src/ 目錄下執行):

    1. 生成 128 節點的高密度 3D Torus (主要實驗 - 8卡伺服器):
       # 模擬 16 台伺服器，每台 8 張 GPU (4x4x8 架構)
       # Z軸 (8): 代表單機 8 卡，走 65Gbps PCIe/NVLink
       # X/Y軸 (4x4): 代表機櫃間互連，走 25Gbps Ethernet
       python3 src/topology_generator.py \
           --type torus \
           --nodes 128 \
           --dims 4 4 8 \
           --bw-intra 65Gbps --lat-intra 0.014ms \
           --bw-inter 25Gbps --lat-inter 0.005ms

    2. 生成 128 節點的 Twisted Torus (Google TPU 風格 - 8卡伺服器):
       # 同樣是 4x4x8 配置，但加入 Twist 連接以減少網路直徑
       python3 src/topology_generator.py \
           --type twisted_torus \
           --nodes 128 \
           --dims 4 4 8 \
           --bw-intra 65Gbps --lat-intra 0.014ms \
           --bw-inter 25Gbps --lat-inter 0.005ms

    3. 生成 128 節點的高效能 Fat-Tree (全速對照組):
       python3 src/topology_generator.py \
           --type fattree \
           --nodes 128 \
           --bw-intra 65Gbps --lat-intra 0.014ms \
           --bw-inter 65Gbps --lat-inter 0.001ms
"""

import argparse
import json
import math
from pathlib import Path

# ==========================================
# 1. 基礎設定
# ==========================================

# 取得目前這個腳本 (topology_generator.py) 的絕對路徑 -> /workspace/src/topology_generator.py
SCRIPT_PATH = Path(__file__).resolve()
# 取得 src 目錄 -> /workspace/src
SRC_DIR = SCRIPT_PATH.parent
# 取得專案根目錄 (src 的上一層) -> /workspace
PROJECT_ROOT = SRC_DIR.parent

# 定義輸出路徑 (永遠相對於專案根目錄)
DEFAULT_TOPO_DIR = PROJECT_ROOT / "configs/astra-sim/topos"
DEFAULT_SYSTEM_DIR = PROJECT_ROOT / "configs/astra-sim/system"

def get_writer(path):
    # 這裡已經是基於 PROJECT_ROOT 的完整路徑，直接 resolve 確保格式正確
    path = path.resolve()
    path.mkdir(parents=True, exist_ok=True)
    return path

def write_file(filepath, content):
    with open(filepath, 'w') as f:
        f.write(content)
    print(f"[Created] {filepath}")

def write_json(filepath, data):
    with open(filepath, 'w') as f:
        json.dump(data, f, indent=4)
    print(f"[Created] {filepath}")

# ==========================================
# 2. 物理拓撲生成邏輯 (.txt)
# ==========================================

def format_topology_content(total_nodes, switches, gpus, links):
    """
    格式化拓撲內容 (關鍵修正)
    Line 1: Total_Nodes Switch_Count GPU_Count
    Line 2: Switch IDs (用空白分隔)
    Line 3+: Links
    """
    # Header
    content = f"{total_nodes} {len(switches)} {len(gpus)}\n"

    # Line 2: List of Switch IDs (關鍵修正！)
    if switches:
        content += " ".join(map(str, switches)) + "\n"
    else:
        content += "\n" # 若無 switch (如 ring), 留空行

    # Links
    for u, v, bw, lat in links:
        content += f"{u} {v} {bw} {lat} 0\n"

    return content

def gen_torus_phys(args, topo_dir, twisted=False):
    dims = args.dims
    if len(dims) != 3:
        raise ValueError("Torus 必須指定 3 個維度 (例如: 4 4 8)")

    dx, dy, dz = dims
    num_nodes = dx * dy * dz

    # ==========================================
    # 扭曲邏輯 (Twist Logic) 的物理意義
    # ==========================================
    # 舊版問題：Twist 加在 Z 軸 (Intra-node)。這物理上代表用 PCIe 線跨機櫃互連，
    #          這只有在 Google TPU Pod (專用光互連) 才做得到。
    # 新版修正：Twist 加在 X 軸 (Inter-node)。這物理上代表在插拔 Ethernet/InfiniBand
    #          網路線時，故意「錯位」連接不同機櫃，這是通用伺服器完全可以做到的。

    twist_step_y = 1 if twisted else 0

    # ID 分配
    gpus = list(range(num_nodes))
    switches = list(range(num_nodes, 2 * num_nodes))
    links = []

    for z in range(dz):
        for y in range(dy):
            for x in range(dx):
                curr_idx = z*(dx*dy) + y*dx + x
                gpu_id = curr_idx
                sw_id = num_nodes + curr_idx

                # 1. GPU <-> Switch (Node 內部連線)
                # 這是伺服器內部的 PCIe/NVLink 通道，使用 bw_intra (65G)
                links.append((gpu_id, sw_id, args.bw_intra, args.lat_intra))
                links.append((sw_id, gpu_id, args.bw_intra, args.lat_intra))

                # 2. Switch <-> Switch (互連)
                # 修改重點：加入反向連結 (v, u)，使網路變成雙向 (Bidirectional)

                # --- Z-Dimension (Intra-node / PCIe Ring) ---
                # Z 軸必須是「封閉」的
                # 模擬單機 8 卡形成的 PCIe Ring。無論是否 Twisted，這裡都不能錯位。
                # 每一台 Server 內部的 PCIe 都是獨立的閉環。
                next_z = (z + 1) % dz
                nz_idx = next_z*(dx*dy) + y*dx + x + num_nodes
                # 正向
                links.append((sw_id, nz_idx, args.bw_intra, args.lat_intra))
                # 反向 (新增)
                links.append((nz_idx, sw_id, args.bw_intra, args.lat_intra))

                # --- Y-Dimension (Inter-node / Ethernet) ---
                # 機櫃間的直連，使用 bw_inter (25G)
                next_y = (y + 1) % dy
                ny_idx = z*(dx*dy) + next_y*dx + x + num_nodes
                # 正向
                links.append((sw_id, ny_idx, args.bw_inter, args.lat_inter))
                # 反向 (新增)
                links.append((ny_idx, sw_id, args.bw_inter, args.lat_inter))

                # --- X-Dimension (Inter-node / Ethernet with Twist) ---
                # 扭曲發生在這裡 (Ethernet 層)
                # 當 X 軸走到邊界要繞回來時 (Wrap-around)，如果是 Twisted Torus，
                # 我們故意連到不同的 Y 座標 (錯位連接)。

                if x < dx - 1:
                    # 正常連接：連到右邊的機櫃
                    next_x = x + 1
                    target_y = y
                else:
                    # 邊界回繞 (Wrap-around)：從最右邊連回最左邊
                    next_x = 0
                    # 如果是 Twisted Mode，Y 座標會偏移，形成「螺旋」結構
                    target_y = (y + twist_step_y) % dy

                nx_idx = z*(dx*dy) + target_y*dx + next_x + num_nodes
                # 正向
                links.append((sw_id, nx_idx, args.bw_inter, args.lat_inter))
                # 反向 (新增)
                links.append((nx_idx, sw_id, args.bw_inter, args.lat_inter))

    tag = f"{num_nodes}nodes_{'TwistedTorus' if twisted else 'Torus'}_{dx}x{dy}x{dz}"

    # 寫入物理 topology 檔
    content = format_topology_content(num_nodes * 2, switches, gpus, links)
    write_file(topo_dir / f"{tag}.txt", content)

    return tag, dims

def gen_fattree_phys(args, topo_dir):
    """ 生成 Fat-Tree 物理拓撲 (2-Layer Leaf-Spine) """
    num_gpus = args.nodes
    gpus_per_leaf = args.gpus_per_leaf

    # 計算需要的 Switch 數量
    num_leafs = math.ceil(num_gpus / gpus_per_leaf)
    num_spines = max(1, int(num_leafs / 2))

    leaf_start_id = num_gpus
    spine_start_id = leaf_start_id + num_leafs

    gpus = list(range(num_gpus))
    leafs = list(range(leaf_start_id, leaf_start_id + num_leafs))
    spines = list(range(spine_start_id, spine_start_id + num_spines))
    switches = leafs + spines # 所有的 Switch ID

    links = []

    # 1. GPU <-> Leaf Switch (Access Layer - Intra)
    for i, leaf_id in enumerate(leafs):
        for j in range(gpus_per_leaf):
            gpu_id = i * gpus_per_leaf + j
            if gpu_id < num_gpus:
                links.append((gpu_id, leaf_id, args.bw_intra, args.lat_intra))
                links.append((leaf_id, gpu_id, args.bw_intra, args.lat_intra))

    # 2. Leaf <-> Spine Switch (Aggregation Layer - Inter)
    for leaf_id in leafs:
        for spine_id in spines:
            links.append((leaf_id, spine_id, args.bw_inter, args.lat_inter))
            links.append((spine_id, leaf_id, args.bw_inter, args.lat_inter))

    tag = f"{num_gpus}nodes_FatTree_L{len(leafs)}_S{len(spines)}"
    total_nodes = num_gpus + len(switches)

    # 使用修正後的格式化函式
    content = format_topology_content(total_nodes, switches, gpus, links)
    write_file(topo_dir / f"{tag}.txt", content)

    return tag, [num_gpus]

# ==========================================
# 3. 設定與主程式 (維持不變)
# ==========================================

def gen_configs(args, topo_dir, sys_dir, tag, logical_dims, topology_type):
    # 定義檔名
    logical_filename = f"logical_{tag}.json"
    system_filename = f"system_{tag}.json"
    phys_filename = f"{tag}.txt"

    # --- 1. 寫入 Logical Topology ---
    logical_data = {"logical-dims": [str(d) for d in logical_dims]}
    write_json(topo_dir / logical_filename, logical_data)

    # --- 2. 根據拓撲類型自動選擇最佳演算法 ---
    if topology_type == 'fattree':
        # Fat-Tree 適合 Tree-based 算法
        impl_list = ["halvingDoubling"]
    else:
        # Torus (含 Twisted) 適合 Ring-based 算法
        # 且演算法數量必須等於維度數量 (例如 3 維就要 3 個 Ring)
        # 對於 4x4x8，這裡會產生 ["ring", "ring", "ring"]，對應 X, Y, Z
        impl_list = ["ring"] * len(logical_dims)

    system_data = {
        "scheduling-policy": "LIFO",
        "endpoint-delay": 10,
        "active-chunks-per-dimension": 1,
        "preferred-dataset-splits": 4,
        "all-reduce-implementation": impl_list,
        "all-gather-implementation": impl_list,
        "reduce-scatter-implementation": impl_list,
        "all-to-all-implementation": impl_list,
        "collective-optimization": "localBWAware",
        "local-mem-bw": 540,
        "boost-mode": 0
    }

    # --- 3. 寫入 System Configuration ---
    write_json(sys_dir / system_filename, system_data)

    # --- 4. 顯示生成結果 ---
    print(f"--- 檔案組生成完畢 ---")
    print(f"1. 物理: {topo_dir}/{phys_filename}")
    print(f"2. 邏輯: {topo_dir}/{logical_filename}")
    print(f"3. 系統: {sys_dir}/{system_filename}")
    print("-" * 50)

def main():
    parser = argparse.ArgumentParser(
        description="ASTRA-sim Topo Generator",
        formatter_class=argparse.RawTextHelpFormatter # 為了讓說明文字換行生效
    )

    # 路徑設定 (預設為相對路徑)
    parser.add_argument("--topo-dir", type=str, default=str(DEFAULT_TOPO_DIR), help="拓撲檔輸出目錄")
    parser.add_argument("--sys-dir", type=str, default=str(DEFAULT_SYSTEM_DIR), help="系統檔輸出目錄")

    # 核心參數
    parser.add_argument("--type", type=str, required=True, choices=['torus', 'twisted_torus', 'fattree'], help="拓撲類型")
    parser.add_argument("--nodes", type=int, default=128, help="總 GPU 數量")

    # 頻寬參數 (支援混合頻寬)
    parser.add_argument("--bw-intra", type=str, default="65Gbps", help="節點內/下行頻寬 (GPU <-> Switch)")
    parser.add_argument("--lat-intra", type=str, default="0.014ms", help="節點內延遲")
    parser.add_argument("--bw-inter", type=str, default="25Gbps", help="節點間/上行頻寬 (Switch <-> Switch)")
    parser.add_argument("--lat-inter", type=str, default="0.005ms", help="節點間延遲")

    # 拓撲專用參數
    parser.add_argument("--dims", type=int, nargs='+', help="Torus 專用: 維度定義 (例如: 4 4 8)")
    parser.add_argument("--gpus-per-leaf", type=int, default=8, help="FatTree 專用: 每個 Leaf Switch 接幾顆 GPU")

    args = parser.parse_args()

    # 建立輸出目錄
    topo_dir = get_writer(Path(args.topo_dir))
    sys_dir = get_writer(Path(args.sys_dir))

    if args.type == 'torus':
        tag, log_dims = gen_torus_phys(args, topo_dir, twisted=False)
        gen_configs(args, topo_dir, sys_dir, tag, log_dims, 'torus')
    elif args.type == 'twisted_torus':
        tag, log_dims = gen_torus_phys(args, topo_dir, twisted=True)
        gen_configs(args, topo_dir, sys_dir, tag, log_dims, 'torus')
    elif args.type == 'fattree':
        tag, log_dims = gen_fattree_phys(args, topo_dir)
        gen_configs(args, topo_dir, sys_dir, tag, log_dims, 'fattree')

if __name__ == "__main__":
    main()
