# horovodrun -np 2 python ./src/test/horovod_allreduce_test.py
# horovodrun -np 4 -H localhost:4 python ./src/test/horovod_allreduce_test.py

import torch
import horovod.torch as hvd
import os

hvd.init()
rank = hvd.rank()
local_rank = hvd.local_rank()
size = hvd.size()

num_gpus = torch.cuda.device_count()
torch.cuda.set_device(local_rank % num_gpus)
device = torch.device(f"cuda:{local_rank % num_gpus}")

print(f"Rank {rank}, Local Rank {local_rank}, Size {size}, Using GPU {local_rank % num_gpus}")
print(f"PyTorch version: {torch.__version__}")
print(f"CUDA available: {torch.cuda.is_available()}, ROCm: {torch.version.hip is not None}")
print(f"Current device: {torch.cuda.current_device()}, Device name: {torch.cuda.get_device_name(torch.cuda.current_device())}")

# Allreduce 測試
tensor = torch.ones(10, device=device) * rank
print(f"Rank {rank} initial tensor: {tensor}")

reduced_sum = hvd.allreduce(tensor, op=hvd.Sum)
print(f"Rank {rank} reduced tensor (Sum): {reduced_sum}")

reduced_avg = hvd.allreduce(tensor)
print(f"Rank {rank} reduced tensor (Average): {reduced_avg}")

# Broadcast 測試
broadcast_tensor = torch.ones(5, device=device) * (rank + 1)
broadcast_tensor = hvd.broadcast(broadcast_tensor, root_rank=0)
print(f"Rank {rank} after broadcast from root 0: {broadcast_tensor}")
print(f"Rank {rank} broadcast_tensor.device: {broadcast_tensor.device}, expected: {device}")

# Allgather 測試
gathered = hvd.allgather(torch.tensor([rank], device=device))
print(f"Rank {rank} allgather result: {gathered}")

# 測試 tensor 是否真的在正確的 GPU 上
assert str(broadcast_tensor.device) == str(device), f"Tensor not on expected device for rank {rank}: {broadcast_tensor.device} vs {device}"
