# ROCm-ASTRAsim: Trace-Driven Simulation Pipeline for AMD GPU AI Clusters

> [繁體中文](README_zh.md) | **English**

> **Thesis:** *"Cost-Effective Twisted Torus for AI Training: An ASTRA-sim Evaluation Using Traces from Consumer-Grade AMD GPUs with ROCm"*
> National Cheng Kung University (NCKU), Graduate Institute of Computer Science and Information Engineering, 2026

This repository implements a three-stage trace-driven simulation pipeline for collecting real training traces from **AMD ROCm/RCCL hardware** and feeding them into **ASTRA-sim** for cluster-scale AI network simulation. It is positioned for the setting where prior published ASTRA-sim studies predominantly assume **NVIDIA CUDA/NCCL**.

The thesis evaluates Fat-Tree, standard 3D Torus, and Twisted Torus at 128-node scale across **four communication regimes**:

1. **Compute-dominated AllReduce** — ResNet-50 DDP (~89.7 MiB / step)
2. **Communication-intensive AllReduce** — Qwen2.5-0.5B DDP (~1.84 GiB / step)
3. **Hierarchical TP+DDP** — Qwen2.5-1.5B (TP=8 × DDP=16)
4. **All-to-All bandwidth saturation** — synthetic stress test (1 GB / collective)

The principal finding is that the Twisted Torus's behavior is *algorithm-dependent*: with Ring AllReduce it is 76% slower than the standard Torus on the Qwen 0.5B workload, but with Halving-Doubling on the twisted dimensions it becomes the fastest configuration (21% faster than standard Torus + ring) at zero additional hardware cost.

---

## Repository Structure

```
.
├── src/
│   ├── train_rocm_pytorch.py      # Stage 1 — DDP training + Kineto trace (CIFAR-10 / ResNet-50 / Qwen05B)
│   ├── train_rocm_tensor.py       # Stage 1 — TP=2 training + Kineto trace (Qwen 1.5B for TP+DDP)
│   ├── conver_to_chakra_et.py     # Stage 2 — Kineto JSON → Chakra ET (with AMD patches, optional --add-ddp)
│   ├── add_ddp_to_et.py           # Stage 2 helper — Append DDP AllReduce nodes to TP ET (TP+DDP)
│   ├── scale_et_comm_workload.py  # Workload augmentation (All-to-All stress test)
│   ├── topology_generator.py      # Torus / Twisted-Torus / Fat-Tree topology file generator
│   └── rocm_compat.py             # ROCm GPU frequency monitor
├── scripts/
│   ├── run_ns3.py                 # Stage 3 — ASTRA-sim ns-3 orchestration + calibration
│   ├── README.md                  # Calibration methodology and run_ns3.py reference
│   └── commands.md                # Complete command reference for all four experiments
├── configs/astra-sim/
│   ├── system/                    # ASTRA-sim system configs (per topology + chunk / algorithm variants)
│   ├── topos/                     # ns-3 physical topology + ASTRA-sim logical topology files
│   └── ns3/                       # ns-3 network parameter configs
├── data/chakra/
│   ├── pytorch_traces/            # (input) Kineto JSON traces from Stage 1
│   ├── gpu_metrics/               # (input) GPU frequency records
│   ├── models/                    # (input) HuggingFace cache for parameter-count detection (TP+DDP)
│   └── workload_et/               # (output) Chakra ET files (.et)
├── docs/                          # ASTRA-sim configuration docs and historical reports
├── runs/                          # Simulation results + calibration_all.csv
├── tutorials/                     # Academic tutorial materials (MICRO'24, ASPLOS'23)
├── viz/                           # Interactive 3D Twisted Torus topology visualizer
├── rocm/dockerfile                # Docker environment (ROCm + PyTorch + ASTRA-sim)
└── docker-compose.yaml
```

---

## Hardware Platform

All real-hardware measurements are performed on:

| Component | Specification |
|---|---|
| CPU | AMD Ryzen 7 5800X |
| GPUs | 2× AMD Radeon RX 9070 XT (Navi 48, 16 GB GDDR6) |
| GPU interconnect | PCIe Gen4 x8 (via host PCIe root complex) |
| OS | Ubuntu 24.04 |
| Container | `rocm/pytorch:rocm6.4.4_ubuntu24.04_py3.12_pytorch_release_2.7.1` |

**Measured physical parameters:**

| Parameter | Value | Method |
|---|---|---|
| Inter-node effective bandwidth | 65 Gbps | `rccl-tests` 512 MB AllReduce |
| Per-link effective latency | 14 µs | `rccl-tests` 4 B + empirical calibration |
| Local GPU memory bandwidth | 540 GB/s | `rocm-bandwidth-test` |

> **Consumer GPU limitation:** AMD Radeon (RDNA) GPUs lack GPUDirect RDMA. All inter-node transfers are CPU-mediated (bounce-buffer through host RAM). The calibrated 14 µs effective latency absorbs this software-stack overhead rather than reflecting physical propagation delay.

---

## Three-Stage Pipeline

### Stage 1 — Trace Collection

Two trace-collection scripts cover the four thesis experiments:

| Script | Workloads | Purpose |
|---|---|---|
| `src/train_rocm_pytorch.py` | `cifar10`, `resnet50`, `qwen05b`, `llama1b` | DDP training (Experiments 1 & 2 + diagnostic) |
| `src/train_rocm_tensor.py`  | Qwen2.5-1.5B with `parallelize_module` (TP=2) | TP trace for Experiment 3 (TP+DDP) |

Both scripts use the PyTorch Kineto profiler to produce per-rank `host_*.json` and `device_*.json` traces.

```bash
# Experiment 1 — ResNet-50 DDP (compute-dominated, primary calibration workload)
torchrun --standalone --nproc_per_node=2 ./src/train_rocm_pytorch.py \
  --model resnet50 --workers 4 \
  --trace-wait 32 --trace-steps 2 \
  --inject-sync-hack

# Experiment 2 — Qwen2.5-0.5B DDP (communication-intensive, ~1.84 GiB per step)
torchrun --standalone --nproc_per_node=2 ./src/train_rocm_pytorch.py \
  --model qwen05b --batch-size 4 --workers 0 \
  --seq-len 256 \
  --trace-wait 10 --trace-steps 2 \
  --inject-sync-hack

# Experiment 3 — Qwen2.5-1.5B with TP=2 (collected on 2 GPUs; later replicated/scaled to TP=8 × DDP=16)
torchrun --standalone --nproc_per_node=2 ./src/train_rocm_tensor.py \
  --epochs 3 --batch-size 1 --workers 0 \
  --seq-len 256 \
  --trace-wait 10 --trace-steps 2 \
  --inject-sync-hack

# Diagnostic — Simple CNN / CIFAR-10 (latency-bound; excluded from 128-node evaluation)
torchrun --standalone --nproc_per_node=2 ./src/train_rocm_pytorch.py \
  --model cifar10 --workers 0 \
  --trace-wait 32 --trace-steps 4 \
  --inject-sync-hack
```

**Output (one set per experiment):**
`data/chakra/pytorch_traces/host_<rank>_<model>.json`, `device_<rank>_<model>.json`,
`data/chakra/gpu_metrics/gpu_metrics_<rank>_<model>.json`.

**Key flags:**
- `--inject-sync-hack` — Inject extra sync events to stabilize `chakra_trace_link` on ROCm (recommended).
- `--trace-steps 1–4` — Keep traces small. Oversized traces significantly increase ns-3 runtime and may exhaust ASTRA-sim's ETFeeder.
- `--seq-len` — LLM sequence length (only `qwen05b` / `llama1b` / Qwen 1.5B TP). 256 keeps the trace size manageable while producing realistic communication volumes.

### Stage 2 — Trace Conversion (`src/conver_to_chakra_et.py`)

Converts Kineto JSON to Chakra ET (`.et`) format, applying **two AMD-specific patches** beyond the upstream HIP kernel recognition added in Chakra commit `df5204c`:

| Patch | Problem | Fix |
|---|---|---|
| **Patch 1 — RCCL node classification** | `ncclDevKernel_Generic` misidentified as `COMP_NODE` | Intercepts `get_protobuf_node_type_from_json_node` to classify all `ncclDevKernel_Generic*` variants as `COMM_COLL_NODE` |
| **Patch 2 — RCCL collective type** | Generic kernel names carry no collective type info | Maps all `ncclDevKernel_Generic*` to `ALL_REDUCE` (correct for DDP workloads) |
| **DAG repair pass** | Self-dependencies, cycles, dangling refs crash ASTRA-sim ETFeeder | DFS cycle detection + self-dep removal + dangling ref pruning |

```bash
# ResNet-50 DDP — standard mode (real kernel durations from trace)
python ./src/conver_to_chakra_et.py --model-tag resnet50

# Qwen 0.5B DDP — same standard mode
python ./src/conver_to_chakra_et.py --model-tag qwen05b

# Qwen 1.5B TP — append DDP AllReduce + scale TP=2 trace to TP=8 simulation
# (auto-detects parameter count from data/models/ HuggingFace cache)
python ./src/conver_to_chakra_et.py \
  --model-tag qwen15b_tp \
  --add-ddp --target-tp 8

# CIFAR-10 — diagnostic only; system-aware calibration may be applied via
# --force-avg-kernel-ns if needed for latency-bound studies
python ./src/conver_to_chakra_et.py --model-tag cifar10

# All-to-All preparation — duplicate the resnet50 trace under a separate tag
python ./src/conver_to_chakra_et.py --model-tag resnet50_all2all
```

**Output:** `data/chakra/workload_et/et.<model_tag>.<rank>.et`.

For TP+DDP, `--add-ddp` appends DDP AllReduce nodes whose `comm_size` is computed from the auto-detected model parameter count, and rescales compute durations from the recorded TP=2 trace to the TP=8 simulation target. See [src/add_ddp_to_et.py](src/add_ddp_to_et.py) for the standalone helper.

### Stage 3 — Simulation (`scripts/run_ns3.py`)

Orchestrates ASTRA-sim + ns-3, performing configuration generation, virtual scale-up, simulation execution, and automatic calibration.

```bash
# 2-GPU calibration (any workload). Verify alpha_us is recorded in runs/calibration_all.csv.
python ./scripts/run_ns3.py \
  --workload data/chakra/workload_et --model-tag resnet50 \
  --topo auto:1d \
  --phys-topo configs/astra-sim/topos/2_nodes_1_switch_topology.txt \
  --coll-opt localBWAware --lmbw 540
```

The full set of 128-node experiment commands is documented in [scripts/commands.md](scripts/commands.md). Selected examples:

```bash
# Experiment 1 — ResNet-50 DDP at 128 nodes (Torus / Twisted Torus / Fat-Tree)
python scripts/run_ns3.py \
  --workload data/chakra/workload_et --model-tag resnet50 \
  --topo file:configs/astra-sim/topos/logical_128nodes_Torus_4x4x8.json \
  --phys-topo configs/astra-sim/topos/128nodes_Torus_4x4x8.txt \
  --system configs/astra-sim/system/system_128nodes_Torus_4x4x8.json \
  --virtual-world 128 --lmbw 540 --no-autocalib

# Experiment 2 — Qwen 0.5B DDP, *requires* active-chunks=4 (deadlock workaround, see below)
python scripts/run_ns3.py \
  --workload data/chakra/workload_et --model-tag qwen05b \
  --topo file:configs/astra-sim/topos/logical_128nodes_TwistedTorus_4x4x8.json \
  --phys-topo configs/astra-sim/topos/128nodes_TwistedTorus_4x4x8.txt \
  --system configs/astra-sim/system/system_128nodes_TwistedTorus_4x4x8_4chunks.json \
  --virtual-world 128 --lmbw 540 --comm-scale 1.984 --no-autocalib --no-qlen

# Experiment 2 (best DDP configuration) — Twisted Torus + Halving-Doubling on X/Y dimensions
python scripts/run_ns3.py \
  --workload data/chakra/workload_et --model-tag qwen05b \
  --topo file:configs/astra-sim/topos/logical_128nodes_TwistedTorus_4x4x8.json \
  --phys-topo configs/astra-sim/topos/128nodes_TwistedTorus_4x4x8.txt \
  --system configs/astra-sim/system/system_128nodes_TwistedTorus_4x4x8_4chunks_hd.json \
  --virtual-world 128 --lmbw 540 --comm-scale 1.984 --no-autocalib --no-qlen

# Experiment 3 — Qwen 1.5B TP+DDP (TP=8 × DDP=16) on Torus
python scripts/run_ns3.py \
  --workload data/chakra/workload_et --model-tag qwen15b_tp8ddp \
  --topo file:configs/astra-sim/topos/logical_128nodes_TP8_DDP16.json \
  --phys-topo configs/astra-sim/topos/128nodes_Torus_4x4x8.txt \
  --system configs/astra-sim/system/system_128nodes_Torus_TP8DDP.json \
  --virtual-world 128 --lmbw 540 --comm-scale 1.984 --no-autocalib --no-qlen

# Experiment 4 — All-to-All 1 GB stress test (after running scale_et_comm_workload.py)
python scripts/run_ns3.py \
  --workload data/chakra/workload_et --model-tag resnet50_all2all_1GB \
  --topo file:configs/astra-sim/topos/logical_128nodes_TwistedTorus_4x4x8.json \
  --phys-topo configs/astra-sim/topos/128nodes_TwistedTorus_4x4x8.txt \
  --system configs/astra-sim/system/system_128nodes_TwistedTorus_4x4x8.json \
  --virtual-world 128 --payload 12000 --lmbw 540 --no-autocalib
```

**Important `run_ns3.py` flags used in the thesis experiments:**

| Flag | Purpose |
|---|---|
| `--virtual-world N` | Replicate the per-rank trace to a `N`-node simulation |
| `--comm-scale F`    | Scale `comm_size` by `F` (the thesis uses `1.984` for the M=2 → N=128 correction in Qwen experiments) |
| `--no-qlen`         | Redirect `qlen.txt` to `/dev/null` to avoid hundreds of GB of debug output at 128-node scale |
| `--payload`         | Override ns-3 packet payload (use `12000` for the All-to-All 1 GB stress test to keep event count manageable) |
| `--no-autocalib`    | Disable automatic α calculation (only use at 2-GPU calibration; required at 128 nodes) |
| `--deadlock-timeout S` | Auto-kill if `fct.txt` stops updating for `S` seconds (default 12 h; useful for the ring-deadlock case below) |

---

## Calibration

Calibration runs at 2-GPU scale and produces the conversion factor α (µs / cycle) that maps simulation cycles to wall-clock time. In the thesis, calibration is used primarily to validate the **relative-comparison regime** through systematic parameter sensitivity analysis, rather than to claim exact absolute communication-time prediction.

- **ResNet-50 (bandwidth-bound):** Primary calibration benchmark. Wall-clock calibration yields **α_step = 0.002411 µs/cycle**. ns-3 communication time is stable across parameter sweeps but shows an approximately **+100% overestimate** relative to measured RCCL GPU kernel duration, most plausibly due to the transport-path mismatch between ns-3's Ethernet RDMA model and the physical PCIe DMA path.
- **CIFAR-10 (latency-bound):** Excluded from large-scale evaluation. Software-stack overhead dominates step time, so ASTRA-sim is not suitable for absolute prediction in this regime.

**Measured calibration results (2-GPU, per training step):**

| Metric | ResNet-50 | CIFAR-10 |
|---|---|---|
| `ns3_comm_ms` (ns-3 simulation) | **15.06 ms** | **41.07 ms** |
| `real_t_comm_ms` (hardware measurement) | **7.54 ms** | **61.98 ms** |
| ns-3 vs real | **+100%** (overestimate) | **−34%** (underestimate) |
| Calibration status | primary baseline | excluded (scope boundary) |

> For ResNet-50, a parameter sweep across bandwidth, latency, packet payload, and congestion-control settings shows that ns-3 communication time remains in the 14.0–15.1 ms range. This supports the interpretation that the discrepancy is structural rather than parameter-sensitive. For topology studies, this systematic bias is expected to affect all topologies similarly, preserving relative comparisons.

The α value and per-run calibration results are logged to `runs/calibration_all.csv`. See [scripts/README.md](scripts/README.md) for the full calibration methodology.

### Additional workload validation: Qwen2.5-0.5B

The pipeline has also been validated on a **Qwen2.5-0.5B** DDP trace collected on the same 2-GPU AMD platform. This LLM workload contains **494M parameters**, producing **37 AllReduce communication nodes** and about **1.84 GiB** of total communication volume per training step. A 2-GPU calibration run confirms the trace produces reasonable metrics (comm/step ≈ 57.7%, ns-3 underestimates measured communication by 52%), consistent with accumulated per-collective startup overhead across 37 AllReduce operations that ns-3 does not model. Qwen 0.5B is the workload used for Experiment 2 in the thesis.

---

## Topology Configurations

Three pre-configured topologies are provided for 128-node evaluation under both **cost-matched** (Torus/TT at 25 Gbps inter-server vs. Fat-Tree at 65 Gbps) and **bandwidth-matched** (all 65 Gbps) framings:

| Parameter | Fat-Tree (L16_S8) | Torus (4×4×8) | Twisted Torus (4×4×8) |
|---|---|---|---|
| Physical switches | **24** | **0** | **0** |
| Inter-node BW (Z-axis / intra-server) | 65 Gbps | 65 Gbps | 65 Gbps |
| Inter-node BW (X/Y-axis / inter-server, cost-matched) | 65 Gbps | **25 Gbps** | **25 Gbps** |
| Inter-node BW (X/Y-axis, bandwidth-matched variant) | — | 65 Gbps | 65 Gbps |
| Per-link latency | GPU→Leaf: 14 µs; Leaf→Spine: 5 µs | Z: 14 µs; X,Y: 5 µs | Z: 14 µs; X,Y: 5 µs |
| Default collective algorithm | halvingDoubling | ring × 3 | ring × 3 |
| Twist (X wrap-around) | — | none | Y offset +1 |

**Twisted Torus wiring** (X-axis wrap-around):
```
(x=3, y, z) → (x=0, (y+1) mod 4, z)
```

The bandwidth-matched physical topology files for sensitivity studies are
`128nodes_Torus_4x4x8_65G.txt` and `128nodes_TwistedTorus_4x4x8_65G.txt` under
`configs/astra-sim/topos/`.

### System configuration matrix

The system configurations under `configs/astra-sim/system/` enumerate the algorithm × chunk-concurrency combinations evaluated in the thesis:

| File | active-chunks | All-Reduce algorithm | Used in |
|---|---|---|---|
| `system_128nodes_Torus_4x4x8.json` | 1 | ring × 3 | Experiment 1 (ResNet-50) |
| `system_128nodes_Torus_4x4x8_4chunks.json` | 4 | ring × 3 | Experiment 2 (Qwen 0.5B), Torus baseline |
| `system_128nodes_TwistedTorus_4x4x8.json` | 1 | ring × 3 | Experiment 1 |
| `system_128nodes_TwistedTorus_4x4x8_4chunks.json` | 4 | ring × 3 | Experiment 2 (TT + ring; demonstrates 76% slowdown) |
| `system_128nodes_TwistedTorus_4x4x8_4chunks_hd.json` | 4 | halvingDoubling, halvingDoubling, ring | Experiment 2 (TT + HD; **best DDP configuration**) |
| `system_128nodes_FatTree_L16_S8.json` | 1 | halvingDoubling | Experiment 1 |
| `system_128nodes_FatTree_L16_S8_4chunks.json` | 4 | halvingDoubling | Experiment 2 |
| `system_128nodes_*_TP8DDP*.json` | 1 / 4 | ring × 3 / HD on X/Y | Experiment 3 (TP+DDP) |

### Generating new topology / config files

If you need different dimensions, bandwidths, or topology types, `src/topology_generator.py` emits a matching set of files (`*.txt`, `logical_*.json`, `system_*.json`):

```bash
# 128-node 4×4×8 Twisted Torus (cost-matched: 25 Gbps inter-server)
python3 src/topology_generator.py \
  --type twisted_torus \
  --nodes 128 --dims 4 4 8 \
  --bw-intra 65Gbps --lat-intra 0.014ms \
  --bw-inter 25Gbps --lat-inter 0.005ms

# 128-node Fat-Tree
python3 src/topology_generator.py \
  --type fattree \
  --nodes 128 \
  --bw-intra 65Gbps --lat-intra 0.014ms \
  --bw-inter 65Gbps --lat-inter 0.005ms
```

If you only want to reproduce the thesis topologies, use the prebuilt files under `configs/astra-sim/topos/` and `configs/astra-sim/system/` directly.

---

## Topology-Aware Algorithm Selection (Experiment 2 highlight)

Under communication-intensive AllReduce (Qwen 0.5B, 1.84 GiB / step), the *combination* of topology and collective algorithm dominates the result:

| Configuration | Wall (M cycles) | vs. Torus + ring | Notes |
|---|---|---|---|
| 3D Torus + ring × 3 | 5,418 | baseline | symmetric paths, no straggler |
| Fat-Tree + halvingDoubling | 5,963 | +10.1% | 2.6× higher BW, but multi-hop switch traversal |
| **Twisted Torus + ring × 3** | **9,549** | **+76.3%** | path asymmetry compounds through ring's sequential chain |
| **Twisted Torus + HD on X/Y, ring on Z** | **4,302** | **−20.6%** | best observed configuration; PFC events drop 99.7% |

The twist is **neither inherently beneficial nor harmful**. Its effect depends on whether the collective algorithm exploits or conflicts with the modified path structure. Ring AllReduce's sequential chain magnifies the twist's path asymmetry into a system-wide straggler; Halving-Doubling's bidirectional pair exchanges are immune to it. Practical recommendation: for Twisted Torus deployments, use **Halving-Doubling** for AllReduce on the twisted dimensions (HD is natively supported by RCCL/NCCL).

---

## Multi-Dimensional Ring Scheduling Deadlock (active-chunks-per-dimension)

> Reported upstream as [ASTRA-sim Issue #370](https://github.com/astra-sim/astra-sim/issues/370).

When running 128-node Twisted Torus AllReduce experiments at high communication intensity (Qwen 0.5B), the default `active-chunks-per-dimension=1` triggers a deterministic scheduling deadlock under `localBWAware` optimization. The Twisted Torus's asymmetric X-axis wrap-around link causes phase desynchronization across nodes, producing a cross-dimensional circular wait in ASTRA-sim's chunk queues. Symptom: ns-3 stops issuing flows at ~5,337 of an expected ~985,088 flows.

**Workaround:** Set `active-chunks-per-dimension: 4` (matching `preferred-dataset-splits: 4`). The thesis Qwen 0.5B experiments use the `*_4chunks*.json` system configurations; the `*_4chunks_hd.json` variants additionally swap ring → halvingDoubling on the X/Y dimensions to address the underlying path-asymmetry root cause (which the chunks=4 setting only masks at the scheduler level).

The standard 3D Torus and Fat-Tree do not require this workaround because their symmetric paths keep node phase progress synchronized. Other thesis experiments (ResNet-50 AllReduce, TP+DDP, All-to-All) also do not trigger the deadlock and use the default `active-chunks=1` configurations.

---

## All-to-All Stress Test (Experiment 4)

When running AllReduce with the original ResNet-50 trace (~89.7 MiB total per step), communication is fully hidden by GPU computation and no topology difference is observable. To stress the network and expose topological differences, `src/scale_et_comm_workload.py` rewrites every `COMM_COLL_NODE` in-place:

1. **`comm_type`** → forced to `ALL_TO_ALL` (from the original `ALL_REDUCE`)
2. **`comm_size`** → set to the specified byte count (e.g. 1 GB = 1,073,741,824 bytes)

Original compute nodes and DAG structure are preserved unchanged, so the simulation still interleaves computation and communication realistically.

### File naming

```
Input:   et.<prefix>.<rank>.et         (e.g. et.resnet50_all2all.0.et)
Output:  et.<prefix><suffix>.<rank>.et (e.g. et.resnet50_all2all_1GB.0.et)
```

The suffix is auto-generated from `--bytes` if `--suffix` is not specified:

| `--bytes` | Auto-suffix |
|---|---|
| `1G` / `1073741824` | `_1GB` |
| `512MB` / `512M`    | `_512MB` |
| `100MB` / `100M`    | `_100MB` |

### Usage

```bash
# Step 1 — convert the original ResNet-50 trace under a separate tag (if not done yet)
python src/conver_to_chakra_et.py --model-tag resnet50_all2all

# Step 2 — scale to 1 GB All-to-All (produces et.resnet50_all2all_1GB.*.et)
python src/scale_et_comm_workload.py \
  --workload-dir data/chakra/workload_et \
  --prefix resnet50_all2all \
  --bytes 1G

# Step 3 — run simulations with the scaled workload (--payload 12000 to manage ns-3 event count)
python scripts/run_ns3.py \
  --workload data/chakra/workload_et \
  --model-tag resnet50_all2all_1GB \
  --topo file:configs/astra-sim/topos/logical_128nodes_TwistedTorus_4x4x8.json \
  --phys-topo configs/astra-sim/topos/128nodes_TwistedTorus_4x4x8.txt \
  --system configs/astra-sim/system/system_128nodes_TwistedTorus_4x4x8.json \
  --virtual-world 128 --payload 12000 --lmbw 540 --no-autocalib
```

> **Observability regimes (thesis Section 5.4.1):** the original ~89.7 MiB AllReduce trace is fully hidden; 100 MB All-to-All causes selective Torus exposure (Twisted Torus and Fat-Tree still hidden); 512 MB–1 GB produces full topology divergence with the Twisted Torus 18% faster than the standard Torus and Fat-Tree 1.62× faster than the standard Torus. The 1 GB All-to-All case is therefore a simulation-only upper-bound stress test, not a production trace. The 1 GB / collective volume also exceeds physical 16 GB VRAM at 128 nodes and is not directly executable on real hardware.

---

## Environment Setup

### Docker (recommended)

```bash
docker-compose up
# or specify a specific ROCm/PyTorch version:
VERSION=rocm6.4.4_ubuntu24.04_py3.12_pytorch_release_2.7.1 docker-compose up
```

### Environment Validation

```bash
# 1. Hardware layer
rocm-bandwidth-test

# 2. Communication layer
rccl-tests/build/all_reduce_perf -b 512M -e 512M -f 2 -g 2

# 3. Framework layer
torchrun --standalone --nproc_per_node=2 src/train_rocm_pytorch.py --model resnet50 --epochs 1

# 4. Trace format check
python src/tests/check_trace_ready.py
python src/tests/validate_et.py
```

---

## Topology Visualizer

An interactive 3D visualization of the Twisted Torus topology is available:

```
viz/twisted_torus_3d.html
```

Open in any browser to explore the 4×4×8 Twisted Torus wiring pattern.

---

## FAQ

**Q: The ns-3 simulation seems to run forever. Is it actually hung?**
A: Not necessarily. With real topologies and larger ET files, simulation can take a very long time. In the thesis 128-node experiments, **a single run typically took about 4–5 days** to finish.

A practical way to check progress is to inspect whether `fct.txt` is still being updated, e.g.:

```text
runs/20260324-013221+0800_ns3_128gpu_qwen05b_file_logical_128nodes_FatTree_L16_S8/out/fct.txt
```

If `fct.txt` continues to receive new values, the simulation is usually still progressing. Use `--deadlock-timeout` (default 12 h) to auto-kill genuinely stuck runs. Keep `--trace-steps` at 1–4 when generating traces — very large ET files can exhaust ASTRA-sim's ETFeeder. Multiple experiments can be run in parallel across separate shell sessions.

**Q: My Twisted Torus Qwen 0.5B run stops at ~5,337 completed flows.**
A: That is the multi-dimensional ring scheduling deadlock described above (also in thesis Section 6.2.7). Use the `*_4chunks*.json` system configuration so that `active-chunks-per-dimension=4`. For the best observed performance under DDP, use `system_128nodes_TwistedTorus_4x4x8_4chunks_hd.json`, which additionally switches ring → halvingDoubling on the X/Y dimensions.

**Q: `"Node X in ctrl_dep graph, but not found in index"` error from ASTRA-sim?**
A: The ET file has a DAG integrity issue (self-dependency or cycle). Run `conver_to_chakra_et.py` again — the built-in DAG repair pass (`fix_et_dag_inplace`) should resolve this automatically.

**Q: `chakra_trace_link` fails on ROCm with misaligned timestamps?**
A: Add `--inject-sync-hack` to the trace-collection script. This injects synchronization events to align CPU (ms) and GPU (µs) timestamps before trace linking.

**Q: Why does ns-3 overestimate ResNet-50 communication time by about 2×?**
A: The thesis finds this discrepancy to be structurally insensitive to all tested tuning parameters. The most plausible explanation is a transport-path mismatch: ns-3 models Ethernet RDMA transport, while the physical 2-GPU platform communicates over a PCIe DMA path. Because the same ns-3 transport model is applied to all evaluated topologies, this bias cancels in relative comparisons.

**Q: Why is CIFAR-10 excluded from large-scale evaluation?**
A: Its shallow architecture leaves more than half of the step time in unmodeled software overhead (kernel launch, RCCL handshake, CPU scheduling). This causes wall-clock and communication calibration factors to diverge by 1.3×, making ASTRA-sim unsuitable for absolute prediction in this latency-dominated regime. See thesis Section 4.3 for details.

**Q: Why is `--comm-scale 1.984` used for Qwen experiments?**
A: It corrects the per-collective communication size when the M=2 source trace is replicated to N=128 ranks. Specifically, `M(N-1) / (N(M-1)) = 2 × 127 / (128 × 1) ≈ 1.984` aligns the scaled trace's payload with the calibrated 2-GPU baseline. Applied uniformly across all topologies, it does not affect relative comparisons. See thesis Section 4.6.2 for the derivation.

---

## Historical Development Reports

Early-stage debugging and integration reports documenting issues encountered and resolved during pipeline development are preserved in [`docs/archive/`](docs/archive/). These are no longer relevant to normal usage but may be useful for understanding the AMD adaptation challenges.

| File | Contents |
|---|---|
| [ASTRA-sim_Analysis_Report.md](docs/archive/ASTRA-sim_Analysis_Report.md) | Alpha calibration analysis, compute-cycle parsing issues, ns-3 hang investigation |
| [AMD_GPU_ASTRA_SIM_Integration_Complete_Report.md](docs/archive/AMD_GPU_ASTRA_SIM_Integration_Complete_Report.md) | HIP runtime incompatibility, RCCL kernel naming, DAG repair — initial breakthrough report |

---

## Citation

If you use this pipeline or the simulation results, please cite:

```bibtex
@mastersthesis{chen2026twisted,
  author  = {jjasoncool},
  title   = {Cost-Effective Twisted Torus for AI Training: An ASTRA-sim Evaluation
             Using Traces from Consumer-Grade AMD GPUs with ROCm},
  school  = {National Cheng Kung University},
  year    = {2026},
  note    = {Code available at \url{https://github.com/jjasoncool/ROCm-ASTRAsim}}
}
```

---

## Related Resources

- [ASTRA-sim](https://github.com/astra-sim/astra-sim) — Distributed ML training simulator
- [Chakra](https://github.com/mlcommons/chakra) — Execution Trace format by Meta
- [ns-3](https://www.nsnam.org/) — Packet-level network simulator
- [RCCL](https://github.com/ROCm/rccl) — ROCm Collective Communication Library
- [rccl-tests](https://github.com/ROCm/rccl-tests) — RCCL micro-benchmarks
- [TPU v4 paper](https://dl.acm.org/doi/10.1145/3579371.3589350) — Google's Twisted Torus reference (ISCA'23)
