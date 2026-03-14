# ROCm-ASTRAsim: Trace-Driven Simulation Pipeline for AMD GPU AI Clusters

> [繁體中文](README_zh.md) | **English**

> **Thesis:** *"Cost-Effective Twisted Torus for AI Training: An ASTRA-sim Evaluation Using Traces from Consumer-Grade AMD GPUs with ROCm"*
> National Cheng Kung University (NCKU), Graduate Institute of Computer Science and Information Engineering, 2026

This repository implements a three-stage trace-driven simulation pipeline that — to the best of our knowledge — is one of the first publicly documented pipelines for collecting real training traces from **AMD ROCm/RCCL hardware** and feeding them into **ASTRA-sim** for cluster-scale AI network simulation.

---

## Repository Structure

```
.
├── src/
│   ├── train_rocm_pytorch.py      # Stage 1 — DDP training + Kineto trace generation
│   ├── conver_to_chakra_et.py     # Stage 2 — Kineto JSON → Chakra ET (with AMD patches)
│   ├── scale_et_comm_workload.py  # Workload augmentation (All-to-All stress test)
│   ├── topology_generator.py      # Torus/Twisted-Torus topology file generator
│   └── rocm_compat.py             # ROCm GPU frequency monitor
├── scripts/
│   ├── run_ns3.py                 # Stage 3 — ASTRA-sim ns-3 orchestration + calibration
│   ├── README.md                  # Calibration methodology and run_ns3.py reference
│   └── commands.md                # Complete command reference for all experiments
├── configs/astra-sim/
│   ├── system/                    # ASTRA-sim system configs per topology
│   ├── topos/                     # ns-3 physical topology files
│   └── ns3/                       # ns-3 network parameter configs
├── data/chakra/
│   ├── pytorch_traces/            # (input) Kineto JSON traces from Stage 1
│   ├── gpu_metrics/               # (input) GPU frequency records
│   └── workload_et/               # (output) Chakra ET files (.et)
├── docs/
│   ├── astra-sim/                 # ASTRA-sim configuration documentation
│   └── archive/                   # Development logs and debugging reports (historical)
├── runs/                          # Simulation results + calibration_all.csv
├── tutorials/                     # Academic tutorial materials (MICRO'24, ASPLOS'23)
├── viz/                           # Interactive topology visualizer (3D Twisted Torus)
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

> **Note on consumer GPU limitation:** AMD Radeon (RDNA) GPUs lack GPUDirect RDMA. All inter-node transfers are CPU-mediated (bounce-buffer through host RAM). The calibrated 14 µs effective latency absorbs this software-stack overhead rather than reflecting physical propagation delay.

---

## Three-Stage Pipeline

### Stage 1 — Trace Collection (`src/train_rocm_pytorch.py`)

Runs PyTorch DDP training with the Kineto profiler to generate per-rank `host_*.json` and `device_*.json` traces.

```bash
# ResNet-50 (bandwidth-bound, primary calibration workload)
torchrun --standalone --nproc_per_node=2 ./src/train_rocm_pytorch.py \
  --model resnet50 --workers 4 \
  --trace-wait 32 --trace-steps 2 \
  --model-tag resnet50

# Simple CNN / CIFAR-10 (latency-bound, scope-boundary diagnostic)
torchrun --standalone --nproc_per_node=2 ./src/train_rocm_pytorch.py \
  --model cifar10 --workers 0 \
  --trace-wait 32 --trace-steps 4 \
  --model-tag cifar10
```

**Output:** `data/chakra/pytorch_traces/host_0_resnet50.json`, `device_0_resnet50.json`, …

**Key flags:**
- `--inject-sync-hack` — Inject extra sync events to stabilize `chakra_trace_link` on ROCm (recommended)
- `--trace-steps 1–4` — Keep traces small; large traces (>5000 nodes) may cause ns-3 simulation hangs

### Stage 2 — Trace Conversion (`src/conver_to_chakra_et.py`)

Converts Kineto JSON to Chakra ET (`.et`) format, applying **two AMD-specific patches** beyond the upstream HIP kernel recognition added in Chakra commit `df5204c`:

| Patch | Problem | Fix |
|---|---|---|
| **Patch 1 — RCCL node classification** | `ncclDevKernel_Generic` misidentified as `COMP_NODE` | Intercepts `get_protobuf_node_type_from_json_node` to classify all `ncclDevKernel_Generic*` variants as `COMM_COLL_NODE` |
| **Patch 2 — RCCL collective type** | Generic kernel names carry no collective type info | Maps all `ncclDevKernel_Generic*` to `ALL_REDUCE` (correct for DDP workloads) |
| **DAG repair pass** | Self-dependencies, cycles, dangling refs crash ASTRA-sim ETFeeder | DFS cycle detection + self-dep removal + dangling ref pruning |

```bash
# ResNet-50 — standard mode (use real kernel durations from trace)
python ./src/conver_to_chakra_et.py --model-tag resnet50

# CIFAR-10 — system-aware calibration mode
# --force-avg-kernel-ns redistributes real step time back into compute nodes
python ./src/conver_to_chakra_et.py \
  --model-tag cifar10 \
  --force-avg-kernel-ns 609000
```

**Output:** `data/chakra/workload_et/et.resnet50.0.et`, `et.resnet50.1.et`, …

### Stage 3 — Simulation (`scripts/run_ns3.py`)

Orchestrates ASTRA-sim + ns-3, performing configuration generation, virtual scale-up, simulation execution, and automatic calibration.

```bash
# 2-GPU calibration run (ResNet-50, ~39 min)
python ./scripts/run_ns3.py \
  --workload data/chakra/workload_et --model-tag resnet50 \
  --topo auto:1d \
  --phys-topo configs/astra-sim/topos/2_nodes_1_switch_topology.txt \
  --coll-opt localBWAware --lmbw 540

# 128-GPU Torus simulation
python scripts/run_ns3.py \
  --workload data/chakra/workload_et --model-tag resnet50 \
  --topo file:configs/astra-sim/topos/logical_128nodes_Torus_4x4x8.json \
  --phys-topo configs/astra-sim/topos/128nodes_Torus_4x4x8.txt \
  --system configs/astra-sim/system/system_128nodes_Torus_4x4x8.json \
  --virtual-world 128 --lmbw 540 --no-autocalib

# 128-GPU Twisted Torus simulation
python scripts/run_ns3.py \
  --workload data/chakra/workload_et --model-tag resnet50 \
  --topo file:configs/astra-sim/topos/logical_128nodes_TwistedTorus_4x4x8.json \
  --phys-topo configs/astra-sim/topos/128nodes_TwistedTorus_4x4x8.txt \
  --system configs/astra-sim/system/system_128nodes_TwistedTorus_4x4x8.json \
  --virtual-world 128 --lmbw 540 --no-autocalib

# 128-GPU Fat-Tree simulation
python scripts/run_ns3.py \
  --workload data/chakra/workload_et --model-tag resnet50 \
  --topo file:configs/astra-sim/topos/logical_128nodes_FatTree_L16_S8.json \
  --phys-topo configs/astra-sim/topos/128nodes_FatTree_L16_S8.txt \
  --system configs/astra-sim/system/system_128nodes_FatTree_L16_S8.json \
  --virtual-world 128 --lmbw 540 --no-autocalib
```

See [scripts/commands.md](scripts/commands.md) for the complete command reference including All-to-All stress tests.

---

## Calibration

Calibration runs at 2-GPU scale and produces a factor α (µs/cycle) that maps simulation cycles to wall-clock time. Run it before any large-scale simulation to validate your specific hardware setup.

- **ResNet-50 (bandwidth-bound):** < 2% relative communication error — pipeline is valid for this regime.
- **CIFAR-10 (latency-bound):** ~91% error — software-stack overhead (RCCL handshake, kernel launch latency) dominates communication time and is not modeled by ASTRA-sim. **Do not use this pipeline for latency-dominated workloads.**

The α value and per-run calibration results are logged to `runs/calibration_all.csv`. See [scripts/README.md](scripts/README.md) for the full calibration methodology.

---

## Topology Configurations

Three pre-configured topologies are provided for 128-node evaluation:

| Parameter | Fat-Tree (L16_S8) | Torus (4×4×8) | Twisted Torus (4×4×8) |
|---|---|---|---|
| Physical switches | **24** | **0** | **0** |
| Inter-node BW (Z-axis / intra-node) | 65 Gbps | 65 Gbps | 65 Gbps |
| Inter-node BW (X/Y-axis / inter-node) | 65 Gbps | **25 Gbps** | **25 Gbps** |
| Per-link latency | GPU→Leaf: 14 µs; Leaf→Spine: 5 µs | Z: 14 µs; X,Y: 5 µs | Z: 14 µs; X,Y: 5 µs |
| Collective algorithm | halvingDoubling | ring × 3 | ring × 3 |
| Twist (X wrap-around) | — | none | Y offset +1 |

**Twisted Torus wiring** (X-axis wrap-around):
```
(x=3, y, z) → (x=0, (y+1) mod 4, z)
```

---

## All-to-All Stress Test (Workload Augmentation)

When running AllReduce with the original trace, communication is typically hidden by GPU computation and no topology difference is observable. To stress the network and expose topological differences, use `src/scale_et_comm_workload.py` to scale up communication volume.

### What it does

`scale_et_comm_workload.py` reads an existing set of ET files (by prefix) and produces a new set with every `COMM_COLL_NODE` patched in-place:

1. **`comm_type`** → forced to `ALL_TO_ALL` (from the original `ALL_REDUCE`)
2. **`comm_size`** → set to the specified byte count (e.g. 1 GB = 1,073,741,824 bytes)

Original compute nodes and DAG structure are preserved unchanged, so the simulation still interleaves computation and communication realistically. Only the communication semantics and volume change.

### File naming

```
Input:   et.<prefix>.<rank>.et        (e.g. et.resnet50_all2all.0.et)
Output:  et.<prefix><suffix>.<rank>.et (e.g. et.resnet50_all2all_1GB.0.et)
```

The suffix is auto-generated from `--bytes` if `--suffix` is not specified:

| `--bytes` | Auto-suffix |
|---|---|
| `1G` / `1073741824` | `_1GB` |
| `128MB` / `128M` | `_128MB` |
| `512K` | `_512KB` |

### Usage

```bash
# Step 1 — convert the original ResNet-50 trace (if not done yet)
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
  --system configs/astra-sim/system/system_128nodes_TwistedTorus_4x4x8_direct.json \
  --virtual-world 128 --payload 12000 --lmbw 540 --no-autocalib
```

> **Note:** The 1 GB All-to-All is a simulation-only stress test. It is not executable on physical hardware (a 128-node All-to-All at 1 GB per peer would require buffers far exceeding 16 GB VRAM). The tool is designed to saturate network links and expose topology differences in simulation.

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

**Q: ns-3 simulation hangs indefinitely without any output?**
A: This is caused by oversized ET files. Keep `--trace-steps` at 1–4 when generating traces. Files with >5000 nodes may exhaust ASTRA-sim's ETFeeder resources.

**Q: `"Node X in ctrl_dep graph, but not found in index"` error from ASTRA-sim?**
A: The ET file has a DAG integrity issue (self-dependency or cycle). Run `conver_to_chakra_et.py` again — the built-in DAG repair pass (`fix_et_dag_inplace`) should resolve this automatically.

**Q: `chakra_trace_link` fails on ROCm with misaligned timestamps?**
A: Add `--inject-sync-hack` to `train_rocm_pytorch.py`. This injects synchronization events to align CPU (ms) and GPU (µs) timestamps before trace linking.

**Q: Why is CIFAR-10 calibration error ~91% while ResNet-50 is 1.18%?**
A: CIFAR-10's shallow architecture completes computation so quickly that fixed software-stack overhead (kernel launch, RCCL handshake, ~25 µs) dominates communication time. ASTRA-sim models only network transfer time, not host-side OS overhead. ResNet-50's deep architecture produces enough compute time to render this overhead negligible. See thesis Section 4.3 for detailed analysis.

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
