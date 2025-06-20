# ROCm 與 ASTRA-sim 執行環境說明

## Docker 環境設定

### 基本執行模式
```bash
docker run --rm -it \
  --device=/dev/kfd \
  --device=/dev/dri \
  --group-add video \
  --ipc=host \
  --shm-size 8G \
  rocm/pytorch:latest
```

### 除錯執行模式
```bash
docker run -it \
  --cap-add=SYS_PTRACE \
  --security-opt seccomp=unconfined \
  --device=/dev/kfd \
  --device=/dev/dri \
  --group-add video \
  --ipc=host \
  --shm-size 8G \
  rocm/pytorch:latest
```

## ASTRA-sim 使用說明

### 網路拓樸模擬範例
- normal topology
```bash
$ASTRA_SIM_BIN \
  --workload-configuration=${ASTRA_SIM}/examples/network_analytical/workload/AllReduce_1MB \
  --system-configuration=${ASTRA_SIM}/examples/network_analytical/system.json \
  --network-configuration=${ASTRA_SIM}/examples/network_analytical/network.yml \
  --remote-memory-configuration=${ASTRA_SIM}/examples/network_analytical/remote_memory.json
```

- ns3 backend
```bash
${ASTRA_SIM}/extern/network_backend/ns-3/build/scratch/ns3.42-AstraSimNetwork-default \
  --workload-configuration=${WORKLOAD} \
  --system-configuration=${ASTRA_SIM}/examples/ns3/system.json \
  --network-configuration=${ASTRA_SIM}/extern/network_backend/ns-3/scratch/config/config.txt \
  --remote-memory-configuration=${ASTRA_SIM}/examples/ns3/remote_memory.json \
  --logical-topology-configuration=${ASTRA_SIM}/examples/ns3/sample_8nodes_1D.json \
  --comm-group-configuration=\"empty\"
```

### 參數說明
- `workload-configuration`: 工作負載設定檔
- `system-configuration`: 系統配置檔
- `network-configuration`: 網路拓樸設定檔
- `remote-memory-configuration`: 遠端記憶體配置檔
