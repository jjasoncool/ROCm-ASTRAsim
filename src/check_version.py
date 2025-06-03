import torch
print(torch.__version__)  # 預期顯示 2.3.0+rocm6.3 或類似版本
print(torch.cuda.is_available())  # 預期顯示 True
print(torch.cuda.device_count())  # 預期顯示 2（你的兩張顯卡）
