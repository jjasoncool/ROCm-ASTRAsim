#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
生成 ASTRA-sim 可用的 Chakra workload 配置檔案
"""

import os
import sys
import argparse

try:
    import chakra
except ImportError:
    print("錯誤: 找不到 chakra 模組。請確保已安裝 Chakra:")
    print("\npip install https://github.com/mlcommons/chakra/archive/refs/heads/main.zip\n")
    print("如果您在 Docker 中運行，請確保 Chakra 及其依賴已安裝:")
    print("1. PARAM: https://github.com/facebookresearch/param")
    print("2. Holistic Trace Analysis: https://github.com/facebookresearch/HolisticTraceAnalysis")
    sys.exit(1)

def get_available_models():
    """獲取 Chakra 支援的模型列表"""
    try:
        if hasattr(chakra, 'list_available_models'):
            return chakra.list_available_models()
        else:
            # 如果沒有 list_available_models 函數，返回常見的預設模型
            return ["mobilenet_v1", "mobilenet_v2", "resnet18", "resnet34", "resnet50", "resnet101",
                    "bert_base", "bert_large", "vgg16", "vgg19", "inception_v3"]
    except Exception as e:
        print(f"警告: 無法獲取可用模型列表: {e}")
        return []

def check_environment():
    """檢查環境變數和設置"""
    # 檢查 ASTRA_SIM 環境變數
    astra_sim = os.environ.get('ASTRA_SIM')
    if not astra_sim:
        print("警告: 未設置 ASTRA_SIM 環境變數")
        return False

    # 檢查 ASTRA_SIM_BIN 環境變數
    astra_sim_bin = os.environ.get('ASTRA_SIM_BIN')
    if not astra_sim_bin:
        print("警告: 未設置 ASTRA_SIM_BIN 環境變數")

    # 檢查 ASTRA-sim 目錄
    if astra_sim and not os.path.exists(astra_sim):
        print(f"警告: ASTRA_SIM 目錄不存在: {astra_sim}")
        return False

    return True

def build_model(model_name):
    """嘗試使用不同的 API 構建模型"""
    print(f"嘗試構建模型: {model_name}")

    # 檢查可用的 API
    if hasattr(chakra, 'build'):
        print("使用 chakra.build API")
        return chakra.build(model_name)

    # 嘗試替代 API: chakra.models
    elif hasattr(chakra, 'models') and hasattr(chakra.models, model_name):
        print(f"使用 chakra.models.{model_name}")
        model_class = getattr(chakra.models, model_name)
        return model_class()

    # 嘗試使用 chakra.ModelFactory (如果存在)
    elif hasattr(chakra, 'ModelFactory'):
        print("使用 chakra.ModelFactory API")
        factory = chakra.ModelFactory()
        return factory.create(model_name)

    # 嘗試直接匯入模型模組
    else:
        print(f"嘗試直接匯入模型: {model_name}")
        try:
            # 嘗試動態匯入
            module_name = f"chakra.models.{model_name}"
            model_module = __import__(module_name, fromlist=['*'])

            # 尋找模型類別 (通常與模型名稱相同或首字母大寫)
            model_class_name = model_name
            if not hasattr(model_module, model_class_name):
                model_class_name = model_name.capitalize()

            if hasattr(model_module, model_class_name):
                model_class = getattr(model_module, model_class_name)
                return model_class()
        except ImportError:
            pass

    # 所有嘗試都失敗
    raise ValueError(f"無法構建模型 {model_name}。Chakra API 可能已變更，請檢查文檔。")

def get_chakra_version():
    """嘗試獲取 Chakra 版本"""
    try:
        if hasattr(chakra, '__version__'):
            return chakra.__version__
        elif hasattr(chakra, 'version'):
            return chakra.version
        else:
            return "未知"
    except:
        return "無法獲取"

def main():
    # 檢查環境設置
    env_ok = check_environment()
    if not env_ok:
        print("注意: 環境設置不完整，但仍會嘗試生成 workload 檔案")

    # 解析命令行參數
    parser = argparse.ArgumentParser(description='生成 ASTRA-sim 的 Chakra workload 檔案')
    parser.add_argument('--model', type=str, default='mobilenet_v2',
                        help='模型名稱 (例如: mobilenet_v2, resnet50, bert)')
    parser.add_argument('--batch_size', type=int, default=32,
                        help='批次大小')
    parser.add_argument('--num_nodes', type=int, default=8,
                        help='節點數量 (需與 logical-topology-configuration 匹配)')
    parser.add_argument('--output', type=str, default=None,
                        help='輸出檔案路徑 (預設: <model>_workload.json)')
    parser.add_argument('--verbose', action='store_true',
                        help='顯示詳細信息')

    args = parser.parse_args()

    # 設定輸出檔案名稱
    if args.output is None:
        args.output = f"{args.model}_workload.json"

    # 列出可用的預設模型
    available_models = get_available_models()
    if available_models:
        print(f"Chakra 可用的預設模型: {', '.join(available_models)}")
    else:
        print("警告: 無法獲取可用模型列表，將繼續使用指定的模型")

    # 檢查選擇的模型是否在可用列表中
    if available_models and args.model not in available_models:
        print(f"警告: '{args.model}' 不在預設模型列表中，可能需要自定義模型定義")
        proceed = input("是否繼續? (y/n): ")
        if proceed.lower() != 'y':
            sys.exit(0)

    # 檢查環境變數
    if not check_environment():
        print("錯誤: 環境檢查未通過，請修正上述警告後重試")
        sys.exit(1)

    try:
        # 構建模型
        print(f"正在構建 {args.model} 模型...")
        model = build_model(args.model)

        # 生成 ASTRA-sim 的 workload 檔案
        print(f"正在生成 workload 檔案，批次大小={args.batch_size}，節點數={args.num_nodes}")

        # 檢查 to_astra 方法是否存在
        if not hasattr(chakra, 'to_astra'):
            print("錯誤: Chakra 版本不支持 to_astra 方法")
            print("提示: 您可能需要更新 Chakra 或檢查文檔以了解正確的 API")
            sys.exit(1)

        # 嘗試調用 to_astra 方法，增加錯誤處理
        try:
            chakra.to_astra(
                model,
                args.output,
                batch_size=args.batch_size,
                num_nodes=args.num_nodes
            )
        except TypeError as e:
            # 如果參數錯誤，嘗試不同的參數組合
            print(f"警告: API 參數錯誤: {e}")
            print("嘗試替代參數組合...")
            try:
                # 嘗試僅使用必要參數
                chakra.to_astra(model, args.output)
                print("成功使用替代參數組合")
            except Exception as e2:
                print(f"錯誤: 替代參數也失敗: {e2}")
                raise

        # 確認輸出檔案
        if os.path.exists(args.output):
            file_size = os.path.getsize(args.output)
            print(f"已成功生成 workload 檔案: {args.output} (大小: {file_size/1024:.2f} KB)")
            print(f"您可以使用以下命令運行 ASTRA-sim:")
            print(f"${{ASTRA_SIM}}/extern/network_backend/ns-3/build/scratch/ns3.42-AstraSimNetwork-default \\")
            print(f"  --workload-configuration={os.path.abspath(args.output)} \\")
            print(f"  --system-configuration=${{ASTRA_SIM}}/examples/ns3/system.json \\")
            print(f"  --network-configuration=${{ASTRA_SIM}}/extern/network_backend/ns-3/scratch/config/config.txt \\")
            print(f"  --remote-memory-configuration=${{ASTRA_SIM}}/examples/ns3/remote_memory.json \\")
            print(f"  --logical-topology-configuration=${{ASTRA_SIM}}/examples/ns3/sample_{args.num_nodes}nodes_1D.json \\")
            print(f"  --comm-group-configuration=\\\"empty\\\"")
        else:
            print(f"錯誤: 未能生成 workload 檔案 {args.output}")

    except Exception as e:
        print(f"生成 workload 檔案時發生錯誤: {e}")
        import traceback
        traceback.print_exc()

if __name__ == '__main__':
    main()
