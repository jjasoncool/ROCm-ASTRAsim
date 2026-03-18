# Commands

## 基本 ns-3 執行

```bash
python scripts/run_ns3.py \
 --workload data/chakra/workload_et
 --model-tag cifar10
 --topo auto:1d
 --phys-topo configs/astra-sim/topos/2_nodes_1_switch_topology.txt
 --coll-opt localBWAware
 --lmbw 540
```

## 產生 Torus topology

```bash
python3 src/topology_generator.py \
    --type torus \
    --nodes 8 \
    --dims 2 2 2 \
    --bw-intra 65Gbps --lat-intra 0.014ms \
    --bw-inter 25Gbps --lat-inter 0.005ms
```

## 8 nodes Torus

```bash
python scripts/run_ns3.py \
 --workload data/chakra/workload_et \
 --model-tag resnet50 \
 --topo file:configs/astra-sim/topos/logical_8nodes_Torus_2x2x2.json \
 --phys-topo configs/astra-sim/topos/8nodes_Torus_2x2x2.txt \
 --system configs/astra-sim/system/system_8nodes_Torus_2x2x2.json \
 --virtual-world 8 \
 --lmbw 540 \
 --no-autocalib
```

## 128 nodes Torus

```bash
python scripts/run_ns3.py \
 --workload data/chakra/workload_et \
 --model-tag resnet50 \
 --topo file:configs/astra-sim/topos/logical_128nodes_Torus_4x4x8.json \
 --phys-topo configs/astra-sim/topos/128nodes_Torus_4x4x8.txt \
 --system configs/astra-sim/system/system_128nodes_Torus_4x4x8.json \
 --virtual-world 128 \
 --lmbw 540 \
 --no-autocalib
```

## 128 nodes FatTree

```bash
python scripts/run_ns3.py \
 --workload data/chakra/workload_et \
 --model-tag resnet50 \
 --topo file:configs/astra-sim/topos/logical_128nodes_FatTree_L16_S8.json \
 --phys-topo configs/astra-sim/topos/128nodes_FatTree_L16_S8.txt \
 --system configs/astra-sim/system/system_128nodes_FatTree_L16_S8.json \
 --virtual-world 128 \
 --lmbw 540 \
 --no-autocalib
```

## 128 nodes Torus all-to-all

```bash
python scripts/run_ns3.py \
 --workload data/chakra/workload_et \
 --model-tag resnet50_all2all \
 --topo file:configs/astra-sim/topos/logical_128nodes_Torus_4x4x8.json \
 --phys-topo configs/astra-sim/topos/128nodes_Torus_4x4x8.txt \
 --system configs/astra-sim/system/system_128nodes_Torus_4x4x8.json \
 --virtual-world 128 \
 --lmbw 540 \
 --no-autocalib
```

## 128 nodes Twisted Torus all-to-all

```bash
python scripts/run_ns3.py \
 --workload data/chakra/workload_et \
 --model-tag resnet50_all2all \
 --topo file:configs/astra-sim/topos/logical_128nodes_TwistedTorus_4x4x8.json \
 --phys-topo configs/astra-sim/topos/128nodes_TwistedTorus_4x4x8.txt \
 --system configs/astra-sim/system/system_128nodes_TwistedTorus_4x4x8.json \
 --virtual-world 128 \
 --lmbw 540 \
 --no-autocalib
```

## 128 nodes FatTree all-to-all

```bash
python scripts/run_ns3.py \
 --workload data/chakra/workload_et \
 --model-tag resnet50_all2all \
 --topo file:configs/astra-sim/topos/logical_128nodes_FatTree_L16_S8.json \
 --phys-topo configs/astra-sim/topos/128nodes_FatTree_L16_S8.txt \
 --system configs/astra-sim/system/system_128nodes_FatTree_L16_S8.json \
 --virtual-world 128 \
 --lmbw 540 \
 --no-autocalib
```

## 128 nodes Torus all-to-all direct

```bash
python scripts/run_ns3.py \
 --workload data/chakra/workload_et \
 --model-tag resnet50_all2all \
 --topo file:configs/astra-sim/topos/logical_128nodes_Torus_4x4x8.json \
 --phys-topo configs/astra-sim/topos/128nodes_Torus_4x4x8.txt \
 --system configs/astra-sim/system/system_128nodes_Torus_4x4x8_direct.json \
 --virtual-world 128 \
 --lmbw 540 \
 --no-autocalib
```

## 128 nodes Twisted Torus all-to-all direct

```bash
python scripts/run_ns3.py \
 --workload data/chakra/workload_et \
 --model-tag resnet50_all2all \
 --topo file:configs/astra-sim/topos/logical_128nodes_TwistedTorus_4x4x8.json \
 --phys-topo configs/astra-sim/topos/128nodes_TwistedTorus_4x4x8.txt \
 --system configs/astra-sim/system/system_128nodes_TwistedTorus_4x4x8_direct.json \
 --virtual-world 128 \
 --lmbw 540 \
 --no-autocalib
```

## 128 nodes FatTree all-to-all direct

```bash
python scripts/run_ns3.py \
 --workload data/chakra/workload_et \
 --model-tag resnet50_all2all \
 --topo file:configs/astra-sim/topos/logical_128nodes_FatTree_L16_S8.json \
 --phys-topo configs/astra-sim/topos/128nodes_FatTree_L16_S8.txt \
 --system configs/astra-sim/system/system_128nodes_FatTree_L16_S8_direct.json \
 --virtual-world 128 \
 --lmbw 540 \
 --no-autocalib
```

## scale workload to 1G
```bash
python src/scale_et_comm_workload.py \
    --workload-dir data/chakra/workload_et \
    --prefix resnet50_all2all \
    --bytes 1G
```

## 128 nodes Torus all-to-all 1GB

```bash
python scripts/run_ns3.py \
  --workload data/chakra/workload_et \
  --model-tag resnet50_all2all_1GB \
  --topo file:configs/astra-sim/topos/logical_128nodes_Torus_4x4x8.json \
  --phys-topo configs/astra-sim/topos/128nodes_Torus_4x4x8.txt \
  --system configs/astra-sim/system/system_128nodes_Torus_4x4x8_direct.json \
  --virtual-world 128 \
  --payload 12000 \
  --lmbw 540 \
  --no-autocalib
```

## 128 nodes Twisted Torus all-to-all 1GB

```bash
python scripts/run_ns3.py \
  --workload data/chakra/workload_et \
  --model-tag resnet50_all2all_1GB \
  --topo file:configs/astra-sim/topos/logical_128nodes_TwistedTorus_4x4x8.json \
  --phys-topo configs/astra-sim/topos/128nodes_TwistedTorus_4x4x8.txt \
  --system configs/astra-sim/system/system_128nodes_TwistedTorus_4x4x8_direct.json \
  --virtual-world 128 \
  --payload 12000 \
  --lmbw 540 \
  --no-autocalib
```

## 128 nodes FatTree all-to-all 1GB

```bash
python scripts/run_ns3.py \
  --workload data/chakra/workload_et \
  --model-tag resnet50_all2all_1GB \
  --topo file:configs/astra-sim/topos/logical_128nodes_FatTree_L16_S8.json \
  --phys-topo configs/astra-sim/topos/128nodes_FatTree_L16_S8.txt \
  --system configs/astra-sim/system/system_128nodes_FatTree_L16_S8_direct.json \
  --virtual-world 128 \
  --payload 12000 \
  --lmbw 540 \
  --no-autocalib
```


## 128 nodes Torus all-to-all 100MB

```bash
python scripts/run_ns3.py \
  --workload data/chakra/workload_et \
  --model-tag resnet50_all2all_100MB \
  --topo file:configs/astra-sim/topos/logical_128nodes_Torus_4x4x8.json \
  --phys-topo configs/astra-sim/topos/128nodes_Torus_4x4x8.txt \
  --system configs/astra-sim/system/system_128nodes_Torus_4x4x8_direct.json \
  --virtual-world 128 \
  --payload 12000 \
  --lmbw 540 \
  --no-autocalib
```

## 128 nodes Twisted Torus all-to-all 100MB

```bash
python scripts/run_ns3.py \
  --workload data/chakra/workload_et \
  --model-tag resnet50_all2all_100MB \
  --topo file:configs/astra-sim/topos/logical_128nodes_TwistedTorus_4x4x8.json \
  --phys-topo configs/astra-sim/topos/128nodes_TwistedTorus_4x4x8.txt \
  --system configs/astra-sim/system/system_128nodes_TwistedTorus_4x4x8_direct.json \
  --virtual-world 128 \
  --payload 12000 \
  --lmbw 540 \
  --no-autocalib
```

## 128 nodes FatTree all-to-all 100MB

```bash
python scripts/run_ns3.py \
  --workload data/chakra/workload_et \
  --model-tag resnet50_all2all_100MB \
  --topo file:configs/astra-sim/topos/logical_128nodes_FatTree_L16_S8.json \
  --phys-topo configs/astra-sim/topos/128nodes_FatTree_L16_S8.txt \
  --system configs/astra-sim/system/system_128nodes_FatTree_L16_S8_direct.json \
  --virtual-world 128 \
  --payload 12000 \
  --lmbw 540 \
  --no-autocalib
```

## 128 nodes Torus all-to-all 512MB

```bash
python scripts/run_ns3.py \
  --workload data/chakra/workload_et \
  --model-tag resnet50_all2all_512MB \
  --topo file:configs/astra-sim/topos/logical_128nodes_Torus_4x4x8.json \
  --phys-topo configs/astra-sim/topos/128nodes_Torus_4x4x8.txt \
  --system configs/astra-sim/system/system_128nodes_Torus_4x4x8_direct.json \
  --virtual-world 128 \
  --payload 12000 \
  --lmbw 540 \
  --no-autocalib
```

## 128 nodes Twisted Torus all-to-all 512MB

```bash
python scripts/run_ns3.py \
  --workload data/chakra/workload_et \
  --model-tag resnet50_all2all_512MB \
  --topo file:configs/astra-sim/topos/logical_128nodes_TwistedTorus_4x4x8.json \
  --phys-topo configs/astra-sim/topos/128nodes_TwistedTorus_4x4x8.txt \
  --system configs/astra-sim/system/system_128nodes_TwistedTorus_4x4x8_direct.json \
  --virtual-world 128 \
  --payload 12000 \
  --lmbw 540 \
  --no-autocalib
```

## 128 nodes FatTree all-to-all 512MB

```bash
python scripts/run_ns3.py \
  --workload data/chakra/workload_et \
  --model-tag resnet50_all2all_512MB \
  --topo file:configs/astra-sim/topos/logical_128nodes_FatTree_L16_S8.json \
  --phys-topo configs/astra-sim/topos/128nodes_FatTree_L16_S8.txt \
  --system configs/astra-sim/system/system_128nodes_FatTree_L16_S8_direct.json \
  --virtual-world 128 \
  --payload 12000 \
  --lmbw 540 \
  --no-autocalib
```