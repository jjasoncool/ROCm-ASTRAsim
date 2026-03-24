# scripts/ — ASTRA-sim ns-3 Runner & Calibration Tools

> [繁體中文](README_zh.md) | **English**

This folder contains scripts for running and calibrating ASTRA-sim × ns-3 network simulations. The core script is `run_ns3.py`.

---

## Table of Contents

1. [Network Parameter Calibration Methodology](#1-network-parameter-calibration-methodology)
2. [Full Workflow](#2-full-workflow)
3. [Quick Start](#3-quick-start)
4. [`run_ns3.py` Parameter Reference](#4-run_ns3py-parameter-reference)
5. [Calibration Internals](#5-calibration-internals)
6. [Advanced: Scalability Analysis](#6-advanced-scalability-analysis)

---

## 1. Network Parameter Calibration Methodology

Before running simulations, the network parameters must be calibrated from real hardware measurements. This section explains how to derive accurate values for `topology.txt`.

### Step 1: Measure Physical Baselines

Use microbenchmark tools such as `rccl-tests` to measure *effective* performance — not the theoretical peak from datasheets:

| Measurement | Tool | Purpose |
|---|---|---|
| **Effective bandwidth** | `rccl-tests` (large messages) | Sets the bandwidth in `topology.txt` |
| **End-to-end latency** $T_{RCCL}$ | `rccl-tests` (small messages, e.g., 4 bytes) | Core calibration data |
| **Local memory bandwidth** | `rocm-bandwidth-test` | Sets `--lmbw` |

### Step 2: Analyze Hop Count

ASTRA-sim and ns-3 define latency **per link**, not end-to-end. Determine $N_{hops}$ from the signal path:

| Topology | Path | $N_{hops}$ |
|---|---|---|
| Direct (P2P) | GPU → GPU | 1 |
| Single switch | GPU → Switch → GPU | 2 |
| Multi-level (Fat-Tree) | GPU → Leaf → Spine → Leaf → GPU | 4 |

### Step 3: Calculate Per-Link Latency

$$T_{link} = \frac{T_{RCCL} - T_{overhead}}{N_{hops}}$$

- **$T_{link}$**: value to write into `topology.txt`
- **$T_{RCCL}$**: measured end-to-end latency
- **$T_{overhead}$**: set to 0 when using the *effective latency* strategy (software overhead is spread evenly across physical links)

### Example: Two-Node Single-Switch Setup

**Measure** (`rccl-tests`, small message): $T_{RCCL} \approx 25\ \mu s$

As a concrete reference point, small-message `all_reduce_perf` runs on this platform (8–1024 B, FP16, 2 GPUs) show end-to-end latency in the **~25.8–33.8 µs** range, corresponding to **12.9–16.9 µs per link** on the 2-hop calibration path. We adopt **14 µs** as a representative effective per-link latency near the center of this measured range.

**Analyze**: path is GPU → Switch → GPU, so $N_{hops} = 2$

**Initial calculation**:

$$T_{link,init} = \frac{25\ \mu s}{2} = 12.5\ \mu s$$

> Writing 25 µs directly would cause the simulator to compute $25 \times 2 = 50\ \mu s$, doubling the latency.

**Interpretation**:

The simple 25 µs ÷ 2 = 12.5 µs calculation is only an initial approximation. In the thesis, the adopted **14 µs** is treated as an **effective latency parameter** for relative topology comparison, derived from the measured rccl-tests range rather than claimed as an exact physical per-hop delay.

**Local memory bandwidth** (`rocm-bandwidth-test`):

```text
          RocmBandwidthTest Version: 2.6.0
          Device: 1,  AMD Radeon RX 9070 XT
          Device: 2,  AMD Radeon RX 9070 XT

          Unidirectional copy peak bandwidth GB/s
          D/D       1           2
          1         540.849     14.045
          2         14.046      540.064
```

Local memory bandwidth ≈ **540 GB/s**. Add `--lmbw 540` to simulation commands (default is 1600).

---

## 2. Full Workflow

The simulation pipeline has three independent stages:

| Stage | Script | Function |
|---|---|---|
| **1. Trace collection** | `src/train_rocm_pytorch.py` | PyTorch DDP training on ROCm; produces Kineto JSON traces |
| **2. Trace conversion** | `src/conver_to_chakra_et.py` | Converts Kineto traces to Chakra ET (`.et`) format |
| **3. Network simulation** | `scripts/run_ns3.py` | Runs ASTRA-sim ns-3 with `.et` workloads; auto-calibrates |

Key features:
- **AMD GPU compatibility patch** (stage 2): automatically fixes AMD RCCL kernel naming issues
- **System-aware calibration** (stage 2): `--force-avg-kernel-ns` redistributes real GPU time into compute nodes for System-Bound workloads
- **Virtual scale-up** (stage 3): scales a small (2-GPU) trace to a large (e.g., 128-GPU) simulation
- **Auto-calibration** (stage 3): computes `alpha_us` and appends results to `runs/calibration_all.csv`

---

## 3. Quick Start

### Scenario A: System-Bound Workload (CIFAR-10)

For small compute, system-overhead-dominated workloads. Requires **system-aware calibration**.

**Step 1: Collect trace**

```bash
torchrun --standalone --nproc_per_node=2 ./src/train_rocm_pytorch.py \
  --model cifar10 --workers 0 \
  --trace-wait 32 --trace-steps 4 \
  --model-tag cifar10
```

> `--workers 0` amplifies system overhead for stress testing.

**Step 2: Convert trace (system-aware calibration)**

```bash
python src/conver_to_chakra_et.py --model-tag cifar10
```

> For latency-dominated diagnostics, `--force-avg-kernel-ns` may still be used to redistribute wall-clock time back into compute nodes, but CIFAR-10 is currently excluded from large-scale topology evaluation.

**Step 3: Run simulation and auto-calibrate**

```bash
python scripts/run_ns3.py \
  --workload data/chakra/workload_et --model-tag cifar10 \
  --topo auto:1d \
  --phys-topo configs/astra-sim/topos/2_nodes_1_switch_topology.txt \
  --coll-opt localBWAware \
  --lmbw 540
```

---

### Scenario B: Compute-Bound Workload (ResNet-50)

For compute-intensive workloads. Traces can be used directly and scaled to large topologies.

**Step 1: Collect trace**

```bash
torchrun --standalone --nproc_per_node=2 ./src/train_rocm_pytorch.py \
  --model resnet50 --workers 4 \
  --trace-wait 32 --trace-steps 2 \
  --model-tag resnet50
```

**Step 2: Convert trace (standard mode)**

```bash
python src/conver_to_chakra_et.py --model-tag resnet50
```

> Omit `--force-avg-kernel-ns`; the converter uses real kernel times from the trace.

**Step 3: Baseline calibration (recommended)**

Validate accuracy at 2-GPU scale before large-scale expansion:

```bash
python scripts/run_ns3.py \
  --workload data/chakra/workload_et --model-tag resnet50 \
  --topo auto:1d \
  --phys-topo configs/astra-sim/topos/2_nodes_1_switch_topology.txt \
  --coll-opt localBWAware \
  --lmbw 540
```

> Check `runs/calibration_all.csv`. In the thesis, ResNet-50 is the primary calibration benchmark; `α_step` is the main wall-clock conversion factor, while `α_comm` is diagnostic only.

---

## 4. `run_ns3.py` Parameter Reference

### Core Parameters

| Parameter | Description | Default | Example |
|---|---|---|---|
| `--workload` | Directory containing `.et` workload files | required | `data/chakra/workload_et` |
| `--model-tag` | Tag to filter workload and trace files | optional | `cifar10`, `resnet50` |
| `--virtual-world N` | Virtually scale workload to N nodes | optional | `128` |
| `--topo` | Logical topology (ASTRA-sim) | `auto:1d` | `auto:2d`, `dims:4x4`, `file:topo.json` |
| `--phys-topo` | Physical topology (ns-3) | inferred from world size | `configs/astra-sim/topos/128_nodes_*.txt` |
| `--ns3-bin` | Path to ns-3 binary | env `ASTRA_NS3_BIN` | |
| `--system`, `--network`, `--remote` | Baseline config file paths | defaults | |

### System-Level Overrides (affect ASTRA-sim scheduling)

| Parameter | Description | Example |
|---|---|---|
| `--coll-opt` | Collective operation optimization strategy | `localBWAware` |
| `--lmbw` | Local memory bandwidth (GB/s) | `540` |

### Network-Level Overrides (affect ns-3 packet behavior)

| Parameter | Description | Example |
|---|---|---|
| `--qcn` | Enable/disable QCN (Quantized Congestion Notification) | `0` or `1` |
| `--pfc-dyn` | Enable/disable dynamic PFC threshold | `0` or `1` |
| `--buffer` | Switch buffer size (packets) | `64` |
| `--payload` | Packet payload size (bytes) | `1500` |

### Calibration & Output Parameters

| Parameter | Description | Default |
|---|---|---|
| `--no-autocalib` | Disable automatic `alpha_us` calibration | — |
| `--trace-dir` | Source directory for Kineto traces | `data/chakra/pytorch_traces` |
| `--calib-db` | CSV database path for calibration results | `runs/calibration_all.csv` |
| `--log-dir` | Root directory for simulation output | `runs` |
| `--dry-run` | Generate configs and commands without running | — |

---

## 5. Calibration Internals

When `run_ns3.py` runs at `world=2` (without `--no-autocalib`):

1. **Parse simulation output**: extract `sim_cycles_step` from `stdout.log`
2. **Find real trace**: locate the Kineto trace matching `--model-tag` in `--trace-dir`; extract `real_t_step_ms`
3. **Compute alpha**:

   $$\alpha_{us} = \frac{real\_t\_step\_ms \times 1000}{sim\_cycles\_step}$$

   $\alpha_{us}$ (also written as `α_step`) represents how many real-world microseconds correspond to one simulation cycle. In the thesis, this is the **primary calibration factor** used for all 128-node topology comparisons.

   The script also computes `α_comm` as a diagnostic metric, but it is **not used for calibration** due to semantic differences between ASTRA-sim's Comm time (scheduling latency) and PyTorch Profiler's RCCL kernel duration (cumulative sum).

4. **Save results**: write to `out/metrics.csv` and append to `runs/calibration_all.csv`

**Reference measurements (2-GPU, ResNet-50 and CIFAR-10):**

| Metric | ResNet-50 | CIFAR-10 |
|---|---|---|
| `ns3_comm_ms` (ns-3 simulation) | **15.06 ms** | **41.07 ms** |
| `real_t_comm_ms` (hardware) | **7.54 ms** | **61.98 ms** |
| ns-3 vs real | **+100%** | **−34%** |
| Status | primary baseline | excluded (scope boundary) |

> A systematic parameter sensitivity analysis over bandwidth, latency, packet payload, and congestion-control settings shows that, for ResNet-50, ns-3 communication remains stable at about 14.0–15.1 ms across all tested configurations. This supports the interpretation that the discrepancy is structural (most plausibly Ethernet RDMA vs. PCIe DMA path mismatch) rather than parameter-sensitive.

---

## 6. Advanced: Scalability Analysis

Scale a calibrated 2-GPU model up to a 128-GPU virtual simulation.

### Step 1: Verify baseline accuracy

- Check `runs/calibration_all.csv`
- ResNet-50: use `α_step` for wall-clock conversion and treat communication time primarily as a **relative topology metric**
- CIFAR-10: excluded from large-scale topology evaluation because unmodeled software overhead dominates step time

### Step 2: Run virtual expansion

```bash
python scripts/run_ns3.py \
  --workload data/chakra/workload_et --model-tag resnet50 \
  --virtual-world 128 \
  --topo file:configs/astra-sim/topos/logical_128nodes_FatTree_L16_S8.json \
  --phys-topo configs/astra-sim/topos/128nodes_FatTree_L16_S8.txt \
  --system configs/astra-sim/system/system_128nodes_FatTree_L16_S8.json \
  --no-autocalib \
  --lmbw 540
```

### Step 3: Analyze speedup

Read `sim_t_step_ms` from `out/metrics.csv` and compute communication efficiency:

$$\text{Efficiency} = \frac{T_{ideal}}{T_{sim}} \times 100\%$$

where $T_{ideal} = T_{compute\_2gpu} \div (128/2)$ (assuming perfect linear scaling).
