# test_tp.py
'''
使用以下指令測試

torchrun --nproc_per_node=2 ./src/tests/test_tensor_parallelism.py
'''


import os
import torch
import torch.nn as nn
import torch.distributed as dist
from torch.distributed.tensor.parallel import parallelize_module, ColwiseParallel, RowwiseParallel
from torch.distributed.device_mesh import init_device_mesh

def main():
    dist.init_process_group("nccl")
    rank = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(rank)

    # 建立 device mesh: 2 GPU 做 TP
    mesh = init_device_mesh("cuda", (2,), mesh_dim_names=("tp",))
    print(f"[Rank {rank}] Device mesh created: {mesh}")

    # 簡單的 2-layer model
    model = nn.Sequential(
        nn.Linear(1024, 1024),
        nn.ReLU(),
        nn.Linear(1024, 10),
    ).cuda()

    # TP 切割：第一層 column-wise，第二層 row-wise
    plan = {
        "0": ColwiseParallel(),
        "2": RowwiseParallel(),
    }
    model = parallelize_module(model, mesh["tp"], plan)
    print(f"[Rank {rank}] Model parallelized")

    # 測試 forward
    x = torch.randn(32, 1024, device=f"cuda:{rank}")
    out = model(x)
    print(f"[Rank {rank}] Forward OK, output shape: {out.shape}")

    # 測試 backward
    loss = out.sum()
    loss.backward()
    print(f"[Rank {rank}] Backward OK")

    dist.destroy_process_group()
    print(f"[Rank {rank}] ✅ TP test passed!")

if __name__ == "__main__":
    main()
