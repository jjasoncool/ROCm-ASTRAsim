import torch, platform

print("python =", platform.python_version())

# 顯示 PyTorch 版本
print("PyTorch 版本:", torch.__version__)  # 預期顯示 2.3.0+rocm6.3 或類似版本

# 檢查是否支援 GPU
print("是否支援 GPU:", torch.cuda.is_available())  # 預期顯示 True

# 顯示可用 GPU 數量
print("可用的 GPU 數量:", torch.cuda.device_count())  # 預期顯示 2（你的兩張顯卡）

# 如果 GPU 可用，進一步檢查每個裝置的名稱並指定裝置
if torch.cuda.is_available():
    for i in range(torch.cuda.device_count()):
        print(f"GPU {i} 名稱:", torch.cuda.get_device_name(i))
    # 測試使用 hip:0 作為裝置名稱
    device = torch.device('hip:0')
    print("選定的裝置 (使用 hip):", device)
    # 測試使用 cuda:0 作為裝置名稱
    device_cuda = torch.device('cuda:0')
    print("選定的裝置 (使用 cuda):", device_cuda)
else:
    print("沒有可用的 GPU")

try:
    from torch._C import _kineto  # 有些 build 可見
    print("kineto available =", _kineto is not None)
except Exception as e:
    print("kineto import error:", e)
