# 安裝說明流程
系統層相關套件
```bash
apt-get install rocm-dev
```

## 使用 Docker 執行已預載 PyTorch 的 ROCm 映像檔

1. 下載並執行官方 ROCm + PyTorch Docker 映像檔：
    ```bash
    docker run --rm -it --device=/dev/kfd --device=/dev/dri --group-add video \
      --ipc=host --shm-size 8G \
      rocm/pytorch:latest

    docker run -it --cap-add=SYS_PTRACE --security-opt seccomp=unconfined \
      --device=/dev/kfd --device=/dev/dri --group-add video \
      --ipc=host --shm-size 8G rocm/pytorch:latest
    ```

2. 進入容器後即可直接使用 PyTorch 進行開發與測試。

pip install ./src/horovod/

<!-- ## 1. 安裝 PyTorch 及相關套件
`conda create -p ./env python=3.10`

已安裝指令：
```bash
conda activate ./env
conda install -c conda-forge gcc=12.1.0
conda install -c conda-forge cmake=3.30
conda install mpi4py


# 安裝 rocm
pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/rocm6.3 --no-cache-dir

export HOROVOD_GPU=ROCM
export HOROVOD_ROCM_HOME=/opt/rocm
export HOROVOD_GPU_OPERATIONS=NCCL
export HOROVOD_WITHOUT_TENSORFLOW=1
export HOROVOD_WITH_PYTORCH=1
export HOROVOD_WITH_MPI=1
export HOROVOD_WITHOUT_MXNET=1

pip install horovod[pytorch] --no-cache-dir --use-pep517
``` -->
