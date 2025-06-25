#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
生成 ASTRA-sim 可用的 Chakra workload 配置檔案
使用 chakra_generator 命令行工具
"""

import os
import sys
import argparse
import subprocess
import shutil

def check_chakra_generator():
    """檢查 chakra_generator 是否可用"""
    return shutil.which("chakra_generator") is not None

def run_chakra_generator(num_npus=8, default_runtime=1000, default_tensor_size=1024,
                        default_comm_size=4096, output_dir=None):
    """運行 chakra_generator 命令"""

    if not check_chakra_generator():
        print("錯誤: 找不到 chakra_generator 命令")
        print("請確保已安裝 Chakra 並且 chakra_generator 在 PATH 中")
        return False

    # 設置輸出目錄
    if output_dir is None:
        output_dir = "/workspace/data/chakra"

    # 確保輸出目錄存在
    os.makedirs(output_dir, exist_ok=True)

    # 構建命令
    cmd = [
        "chakra_generator",
        "--num_npus", str(num_npus),
        "--default_runtime", str(default_runtime),
        "--default_tensor_size", str(default_tensor_size),
        "--default_comm_size", str(default_comm_size)
    ]

    print(f"執行命令: {' '.join(cmd)}")
    print(f"輸出目錄: {output_dir}")

    # 切換到輸出目錄執行命令
    original_cwd = os.getcwd()
    try:
        os.chdir(output_dir)
        result = subprocess.run(cmd, capture_output=True, text=True)

        if result.returncode == 0:
            print("✅ chakra_generator 執行成功!")
            print("輸出:")
            print(result.stdout)

            # 列出生成的檔案
            files = os.listdir('.')
            workload_files = [f for f in files if 'workload' in f or f.endswith('.et')]
            if workload_files:
                print(f"生成的 workload 檔案: {workload_files}")
                return workload_files[0]  # 返回第一個 workload 檔案
            else:
                print("警告: 未找到 workload 檔案")
                return True
        else:
            print("❌ chakra_generator 執行失敗!")
            print("錯誤輸出:")
            print(result.stderr)
            return False

    except Exception as e:
        print(f"執行 chakra_generator 時發生錯誤: {e}")
        return False
    finally:
        os.chdir(original_cwd)

def print_astra_sim_command(workload_file, num_npus=8):
    """打印運行 ASTRA-sim 的命令"""
    print("\n" + "="*60)
    print("您可以使用以下命令運行 ASTRA-sim:")
    print("="*60)

    workload_path = os.path.abspath(workload_file) if workload_file else "/workspace/data/chakra/workload_trace"

    print(f"""
${'{ASTRA_SIM}'}/extern/network_backend/ns-3/build/scratch/ns3.42-AstraSimNetwork-default \\
  --workload-configuration={workload_path} \\
  --system-configuration=/workspace/data/ASTRA-sim/system.json \\
  --network-configuration=/workspace/data/ASTRA-sim/scratch/config/config.txt \\
  --remote-memory-configuration=/workspace/data/ASTRA-sim/remote_memory.json \\
  --logical-topology-configuration=/workspace/data/ASTRA-sim/sample_{num_npus}nodes_1D.json \\
  --comm-group-configuration="empty"
""")

def main():
    parser = argparse.ArgumentParser(
        description='使用 chakra_generator 生成 ASTRA-sim workload 檔案',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
範例用法:
  python generate_chakra_workload.py
  python generate_chakra_workload.py --num_npus 16 --default_runtime 2000
  python generate_chakra_workload.py --output_dir ./my_workloads
        """
    )

    parser.add_argument('--num_npus', type=int, default=8,
                        help='NPU 數量 (預設: 8)')
    parser.add_argument('--default_runtime', type=int, default=1000,
                        help='預設運行時間 (微秒) (預設: 1000)')
    parser.add_argument('--default_tensor_size', type=int, default=1024,
                        help='預設張量大小 (預設: 1024)')
    parser.add_argument('--default_comm_size', type=int, default=4096,
                        help='預設通信大小 (預設: 4096)')
    parser.add_argument('--output_dir', type=str, default=None,
                        help='輸出目錄 (預設: /workspace/data/chakra)')
    parser.add_argument('--show_command', action='store_true',
                        help='顯示 ASTRA-sim 運行命令')

    args = parser.parse_args()

    print("=== Chakra Workload Generator ===")
    print(f"NPU 數量: {args.num_npus}")
    print(f"預設運行時間: {args.default_runtime} μs")
    print(f"預設張量大小: {args.default_tensor_size}")
    print(f"預設通信大小: {args.default_comm_size}")
    print()

    # 運行 chakra_generator
    result = run_chakra_generator(
        num_npus=args.num_npus,
        default_runtime=args.default_runtime,
        default_tensor_size=args.default_tensor_size,
        default_comm_size=args.default_comm_size,
        output_dir=args.output_dir
    )

    if result:
        print("✅ Workload 檔案生成成功!")

        if args.show_command or True:  # 總是顯示命令
            workload_file = result if isinstance(result, str) else None
            print_astra_sim_command(workload_file, args.num_npus)
    else:
        print("❌ Workload 檔案生成失敗!")
        print("\n備選方案:")
        print("您可以直接在 /workspace/data/chakra 目錄中運行:")
        print(f"chakra_generator --num_npus {args.num_npus} --default_runtime {args.default_runtime} --default_tensor_size {args.default_tensor_size} --default_comm_size {args.default_comm_size}")
        sys.exit(1)

if __name__ == '__main__':
    main()
