# ROCm-ASTRAsim: Trace-Driven Simulation Pipeline for AMD GPU AI Clusters

> [繁體中文](README_zh.md) | **English**

> **Thesis:** *"Cost-Effective Twisted Torus for AI Training: An ASTRA-sim Evaluation Using Traces from Consumer-Grade AMD GPUs with ROCm"*
> National Cheng Kung University (NCKU), Graduate Institute of Computer Science and Information Engineering, 2026

This repository implements a three-stage trace-driven simulation pipeline for collecting real training traces from **AMD ROCm/RCCL hardware** and feeding them into **ASTRA-sim** for cluster-scale AI network simulation. It is positioned for the setting where prior published ASTRA-sim studies predominantly assume **NVIDIA CUDA/NCCL**.

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
- `--trace-steps 1–4` — Keep traces small; oversized traces may significantly increase ns-3 runtime or cause instability

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
# for latency-dominated diagnostics, optional system-aware forcing may still be used if needed
python ./src/conver_to_chakra_et.py --model-tag cifar10
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

Calibration runs at 2-GPU scale and produces a factor α (µs/cycle) that maps simulation cycles to wall-clock time. In the thesis, calibration is used primarily to validate the **relative-comparison regime** through systematic parameter sensitivity analysis, rather than to claim exact absolute communication-time prediction.

- **ResNet-50 (bandwidth-bound):** Primary calibration benchmark. Wall-clock calibration yields **α_step = 0.002411 µs/cycle**. ns-3 communication time is stable across parameter sweeps but shows an approximately **+100% overestimate** relative to measured RCCL GPU kernel duration, most plausibly due to the transport-path mismatch between ns-3's Ethernet RDMA model and the physical PCIe DMA path.
- **CIFAR-10 (latency-bound):** Excluded from large-scale evaluation. Software-stack overhead dominates step time, so ASTRA-sim is not suitable for absolute prediction in this regime.

**Measured calibration results (2-GPU, per training step):**

| Metric | ResNet-50 | CIFAR-10 |
|---|---|---|
| `ns3_comm_ms` (ns-3 simulation) | **15.06 ms** | **41.07 ms** |
| `real_t_comm_ms` (hardware measurement) | **7.54 ms** | **61.98 ms** |
| ns-3 vs real | **+100%** (overestimate) | **−34%** (underestimate) |
| Calibration status | primary baseline | excluded (scope boundary) |

> For ResNet-50, a parameter sweep across bandwidth, latency, packet payload, and congestion-control settings shows that ns-3 communication time remains in the 14.0–15.1 ms range, supporting the interpretation that the discrepancy is structural rather than parameter-sensitive. For topology studies, this systematic bias is expected to affect all topologies similarly, preserving relative comparisons.

The α value and per-run calibration results are logged to `runs/calibration_all.csv`. See [scripts/README.md](scripts/README.md) for the full calibration methodology.

### Additional workload validation: Qwen2.5-0.5B

The pipeline has also been validated on a **Qwen2.5-0.5B** DDP trace collected on the same 2-GPU AMD platform. This LLM workload contains **494M parameters**, producing **37 AllReduce communication nodes** and about **1.84 GiB** of total communication volume per training step. This confirms that the trace collection and conversion pipeline is not specific to ResNet-50. Full 128-node topology evaluation of the Qwen workload is left as future work.

---

## Topology Configurations

Three pre-configured topologies are provided for 128-node evaluation:

If you need to customize or regenerate topology/configuration files, you can also use `src/topology_generator.py` to produce them directly from the command line. This generator can automatically emit:

- physical topology files (`configs/astra-sim/topos/*.txt`)
- logical topology files (`configs/astra-sim/topos/logical_*.json`)
- system configuration files (`configs/astra-sim/system/system_*.json`)

It supports **Torus**, **Twisted Torus**, and **Fat-Tree** generation. For example:

```bash
# Generate a 128-node 4×4×8 Torus
python3 src/topology_generator.py \
  --type torus \
  --nodes 128 \
  --dims 4 4 8 \
  --bw-intra 65Gbps --lat-intra 0.014ms \
  --bw-inter 25Gbps --lat-inter 0.005ms

# Generate a 128-node 4×4×8 Twisted Torus
python3 src/topology_generator.py \
  --type twisted_torus \
  --nodes 128 \
  --dims 4 4 8 \
  --bw-intra 65Gbps --lat-intra 0.014ms \
  --bw-inter 25Gbps --lat-inter 0.005ms

# Generate a 128-node Fat-Tree
python3 src/topology_generator.py \
  --type fattree \
  --nodes 128 \
  --bw-intra 65Gbps --lat-intra 0.014ms \
  --bw-inter 65Gbps --lat-inter 0.005ms
```

If you simply want to reproduce the topologies used in this work, you can directly use the files already provided under `configs/astra-sim/topos/` and `configs/astra-sim/system/`. If you want to change dimensions, bandwidth, latency, or topology type, use the generator to emit a matching set of files.

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

When running AllReduce with the original corrected ResNet-50 trace, communication is typically hidden by GPU computation and no topology difference is observable. The thesis identifies roughly **89.7 MiB** of total AllReduce traffic per step across 5 DDP gradient buckets. To stress the network and expose topological differences, use `src/scale_et_comm_workload.py` to scale up communication volume.

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
  --system configs/astra-sim/system/system_128nodes_TwistedTorus_4x4x8.json \
  --virtual-world 128 --payload 12000 --lmbw 540 --no-autocalib
```

> **Note:** The thesis shows a three-phase observability pattern: the original ~89.7 MiB trace is fully hidden, 100 MB causes selective Torus exposure, and 512 MB–1 GB produces full topology divergence. The 1 GB All-to-All case is therefore a simulation-only upper-bound stress test, not a production trace.

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
A: Not necessarily. With real topologies and larger ET files, simulation can take a very long time. In our 128-node experiments, a **single run typically took about 5 days** to finish, so it should not be judged by a "10-minute expectation."

A practical way to check progress is to inspect whether `fct.txt` is still being updated, for example:

```text
runs/20260324-013221+0800_ns3_128gpu_qwen05b_file_logical_128nodes_FatTree_L16_S8/out/fct.txt
```

If `fct.txt` continues to receive new values, the simulation is usually still progressing normally rather than being stuck.

Oversized ET files can still cause real problems, so it is recommended to keep `--trace-steps` at 1–4 when generating traces. Very large ET files may exhaust ASTRA-sim's ETFeeder resources or substantially slow the simulation. Since individual experiments may take days, it is also practical to run multiple experiments in parallel across different shell sessions.

**Q: `"Node X in ctrl_dep graph, but not found in index"` error from ASTRA-sim?**
A: The ET file has a DAG integrity issue (self-dependency or cycle). Run `conver_to_chakra_et.py` again — the built-in DAG repair pass (`fix_et_dag_inplace`) should resolve this automatically.

**Q: `chakra_trace_link` fails on ROCm with misaligned timestamps?**
A: Add `--inject-sync-hack` to `train_rocm_pytorch.py`. This injects synchronization events to align CPU (ms) and GPU (µs) timestamps before trace linking.

**Q: Why does ns-3 overestimate ResNet-50 communication time by about 2×?**
A: The thesis finds this discrepancy to be structurally insensitive to all tested tuning parameters. The most plausible explanation is a transport-path mismatch: ns-3 models Ethernet RDMA transport, while the physical 2-GPU platform communicates over a PCIe DMA path. Because the same ns-3 transport model is applied to all evaluated topologies, this bias is expected to cancel in relative comparisons.

**Q: Why is CIFAR-10 excluded from large-scale evaluation?**
A: CIFAR-10's shallow architecture leaves more than half of the step time in unmodeled software overhead. This causes wall-clock and communication calibration factors to diverge, making ASTRA-sim unsuitable for absolute prediction in this latency-dominated regime. See thesis Section 4.3 for details.

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
