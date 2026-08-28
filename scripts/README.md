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
7. [Supporting Measurement Tools](#7-supporting-measurement-tools)

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

**Measure** (`rccl-tests`, small message): $T_{\mathrm{RCCL}} \approx 25\,\mu\mathrm{s}$

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
| **1. Trace collection (DDP)** | `src/train_rocm_pytorch.py` | PyTorch DDP training on ROCm; produces Kineto JSON traces |
| **1. Trace collection (TP)**  | `src/train_rocm_tensor.py` | PyTorch TP=2 training on ROCm; used for the Qwen 1.5B TP+DDP experiment |
| **2. Trace conversion** | `src/conver_to_chakra_et.py` | Converts Kineto traces to Chakra ET (`.et`) format; supports `--add-ddp` for TP+DDP |
| **3. Network simulation** | `scripts/run_ns3.py` | Runs ASTRA-sim ns-3 with `.et` workloads; auto-calibrates |

Key features:
- **AMD GPU compatibility patch** (stage 2): automatically fixes AMD RCCL kernel naming issues
- **System-aware calibration** (stage 2): `--force-avg-kernel-ns` redistributes real GPU time into compute nodes for System-Bound workloads
- **TP+DDP composition** (stage 2): `--add-ddp --target-tp 8` appends DDP AllReduce nodes to a TP trace and rescales compute
- **Virtual scale-up** (stage 3): scales a small (2-GPU) trace to a large (e.g., 128-GPU) simulation
- **Auto-calibration** (stage 3): aligns the measured and simulated comm windows, computes `alpha_us`, and appends results to `runs/calibration_aligned.csv` (`calibration_all.csv` is a legacy log — see §5)

**Workloads covered by the thesis.** The pipeline is exercised across four communication regimes:

| Experiment | Workload | Trace script | Tag (typical) | Notes |
|---|---|---|---|---|
| 1. Compute-dominated AllReduce | ResNet-50 DDP (~89.7 MiB) | `train_rocm_pytorch.py --model resnet50` | `resnet50` | Primary calibration baseline |
| 2. Communication-intensive AllReduce | Qwen 0.5B DDP (~1.84 GiB) | `train_rocm_pytorch.py --model qwen05b` | `qwen05b` | Requires `active-chunks=4` on Twisted Torus |
| 3. Hierarchical TP+DDP | Qwen 1.5B TP=8 × DDP=16 | `train_rocm_tensor.py` + `conver_to_chakra_et.py --add-ddp --target-tp 8` | `qwen15b_tp8ddp` | |
| 4. All-to-All bandwidth saturation | Synthetic 1 GB All-to-All | `scale_et_comm_workload.py --bytes 1G` on `resnet50_all2all` | `resnet50_all2all_1GB` | Use `--payload 12000` on ns-3 |

The Qwen 0.5B 2-GPU calibration confirms the trace is well-formed: 10,916 COMP nodes and 37 AllReduce COMM nodes per step, totalling 1,884.6 MiB — exactly the FP32 gradient of 494,032,768 parameters.

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
  --lmbw 540 --et-iters 4
```

> `--et-iters` must match the `--trace-steps` used in Step 1 (4 here). Without it `alpha_us` is
> left blank rather than guessed.

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
  --lmbw 540 --et-iters 1
```

> The ResNet-50 device window spans one complete training iteration — its five communication
> kernels match DDP's five gradient buckets — so `--et-iters 1`, even though the trace was
> collected with `--trace-steps 2`. Check `runs/calibration_aligned.csv`. In the thesis,
> ResNet-50 is the primary calibration benchmark; $\alpha_{step}$ is the main wall-clock
> conversion factor, while $\alpha_{comm}$ is diagnostic only.

---

## 4. `run_ns3.py` Parameter Reference

### Core Parameters

| Parameter | Description | Default | Example |
|---|---|---|---|
| `--workload` | Directory containing `.et` workload files | required | `data/chakra/workload_et` |
| `--model-tag` | Tag to filter workload and trace files | optional | `cifar10`, `resnet50`, `qwen05b` |
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
| `--payload` | Packet payload size (bytes); use `12000` for All-to-All 1 GB stress test | `1500` |

### Workload Scaling and Long-Run Robustness

| Parameter | Description | Example |
|---|---|---|
| `--virtual-world N` | Replicate per-rank trace to `N`-node simulation | `128` |
| `--comm-scale F` | Multiply every `comm_size` by `F`. A **workload operating-point setting**, not a correction to the collective algorithm — ASTRA-sim decomposes each collective from the configured participant count independently. Thesis value: `127/64` = **`1.984375`**; Qwen 0.5B needs the exact fraction so the scaled size stays divisible by `preferred-dataset-splits=4`, TP+DDP tolerates the rounded `1.984`. Applied uniformly across topologies within an experiment | `1.984375` |
| `--comm-group FILE` | Passed through as `--comm-group-configuration`; omitted → the argument is not passed at all | — |
| `--no-qlen` | Redirect `qlen.txt` to `/dev/null` (avoid hundreds of GB of debug output at 128 nodes) | — |
| `--deadlock-timeout S` | Kill the run if `fct.txt` stops updating for `S` seconds (default `43200` = 12 h, set `0` to disable) | `43200` |

### Calibration & Output Parameters

| Parameter | Description | Default |
|---|---|---|
| `--no-autocalib` | Disable automatic `alpha_us` calibration | — |
| `--et-iters N` | How many training iterations the ET covers (= the `--trace-steps` used to generate it). Required for `alpha_us`, which needs a per-step denominator; omitted → left blank, never guessed | — |
| `--trace-dir` | Source directory for Kineto traces | `data/chakra/pytorch_traces` |
| `--calib-db` | CSV database path for calibration results | `runs/calibration_aligned.csv` |
| `--log-dir` | Root directory for simulation output | `runs` |
| `--dry-run` | Generate configs and commands without running | — |

### Scheduling deadlock on Twisted Torus + ring AllReduce

The default `active-chunks-per-dimension=1` triggers a deterministic ASTRA-sim scheduling deadlock when running multi-dimensional ring AllReduce on the Twisted Torus under heavy load (Qwen 0.5B). The Twisted Torus's asymmetric X-axis wrap-around link causes phase desynchronization that produces a cross-dimensional circular wait in ASTRA-sim's chunk queues — `fct.txt` typically stops updating at ~5,337 of an expected ~985,088 flows. Use the `*_4chunks*.json` system configurations (`active-chunks-per-dimension: 4`) for Twisted Torus AllReduce runs. The `*_4chunks_hd.json` variant additionally swaps ring → halvingDoubling on the X/Y dimensions as the second arm of the 2×2 factorial; note that HD removes the deadlock and collapses PFC but does **not** remove the twist's step-time penalty (Twisted Torus + HD is still +74.7% vs. the Torus + ring baseline — the route-level asymmetry persists). `active-chunks=4` likewise only resolves the scheduler-level deadlock, not the underlying path asymmetry. For DDP deployment, use the standard Torus with ring (the fastest configuration, which does not deadlock). Reported upstream as [ASTRA-sim Issue #370](https://github.com/astra-sim/astra-sim/issues/370).

---

## 5. Calibration Internals

When `run_ns3.py` runs at `world=2` (without `--no-autocalib`):

1. **Parse simulation output**: extract `sim_cycles_step` and `sim_cycles_comm` from `stdout.log`.
2. **Find real trace**: locate the Kineto trace matching `--model-tag` in `--trace-dir`; extract `real_t_step_ms`, the RCCL kernel total (`real_t_net_comm_ms`), and the non-RCCL compute kernel total (`real_t_kernel_ms`). Only GPU-side `cat=kernel` events count; CPU-side `user_annotation` entries are excluded so the same operation isn't counted twice.
3. **Align the two windows** — the step that makes the rest meaningful. The measured side holds one rank's RCCL kernels; the simulated side holds whatever the ET replayed. They cover the same amount of work only if aligned, so the script:
   - recovers the trace's iteration count in-code by merging nested `ProfilerStep` intervals and keeping the non-contained ones (the profiler's own `steps_n` matches neither side and is never used);
   - checks that every comm kernel falls inside some iteration — if any kernel lands outside, iteration boundaries are untrustworthy and the run **stops**;
   - takes `window_ratio` from the **iteration** counts, not the collective counts. A collective-count ratio misreads a per-iteration *granularity* difference (common on TP traces, where the ET replays more collectives per iteration than the trace records) as an iteration-count difference, and ends up comparing one simulated iteration against two measured ones;
   - **raises** rather than guessing when `--et-iters` < the recovered trace iterations, or when the two are not in integer ratio. A plausible-looking ratio is worse than a stopped run;
   - flags `per_iter_granularity_mismatch` loudly instead of folding the difference into the divisor.
4. **Compute alpha**:

   $$
   \alpha_{\mathrm{us}} = \frac{\mathrm{real\_t\_step\_ms} \times 1000}{\mathrm{sim\_cycles\_step} / \mathrm{et\_iters}}
   $$

   The numerator is per-step, so the denominator must be too — hence `--et-iters`. Without it the script sets `alpha_skipped_no_et_iters` and leaves `alpha_us` blank. $\alpha_{\mathrm{us}}$ (also written $\alpha_{step}$) is the **primary calibration factor** used for all 128-node topology comparisons in the thesis.

   $\alpha_{comm}$ is computed from window-consistent totals as a diagnostic, but is **not used for calibration**: it operates on ns-3 nanosecond ticks while $\alpha_{step}$ operates on trace-derived compute cycles, so the two live in different cycle domains and are not expected to match.

5. **Save results**: write to `out/metrics.csv` and append to `runs/calibration_aligned.csv`.

> **`calibration_all.csv` is a legacy append log.** It predates window alignment, and its error figures compare windows covering different amounts of work. Every calibration value in the thesis comes from `calibration_aligned.csv`; do not quote the old file.

To recompute these metrics from run directories you already have, without simulating again — `calibrate_from_runs.py` reuses `run_ns3.align_and_compare`, so the two cannot drift apart:

```bash
python3 scripts/calibrate_from_runs.py --out runs/calibration_aligned.csv \
  resnet50=1:<run_dir> cifar10=4:<run_dir> qwen05b=2:<run_dir> qwen15b_tp=2:<run_dir>
```

Each argument is `tag=et_iters:run_dir`.

**Reference measurements (2-GPU, window-aligned, per training step):**

| Metric | ResNet-50 | CIFAR-10 |
|---|---|---|
| `real_t_step_ms` | 662.92 ms | 224.32 ms |
| `real_t_net_comm_ms` (hardware) | **15.87 ms** | **76.69 ms** |
| `real_t_kernel_ms` | 284.10 ms | 50.15 ms |
| comm / step | 2.4% | 34.2% |
| `ns3_comm_ms` (ns-3) | **15.06 ms** | **10.27 ms** |
| ns-3 vs real | **−5.1%** | **−86.6%** |
| $\alpha_{step}$ | **0.002411** | 0.004550 |
| $\alpha_{comm}$ | 0.001054 | 0.007469 |
| `--et-iters` | 1 | 4 |
| Status | primary baseline | excluded (scope boundary) |

LLM workloads, same alignment:

| Tag | `--et-iters` | `ns3_comm_ms` | `real_t_net_comm_ms` | ns-3 vs real | Flags |
|---|---|---|---|---|---|
| `qwen05b` | 2 | 573.08 ms | 3,393.27 ms | **−83.1%** | — |
| `qwen15b_tp` | 2 | 392.25 ms | 5,918.81 ms | **−93.4%** | `per_iter_granularity_mismatch` |

> A systematic sweep over packet payload (1,000–8,000 B), per-link latency (12.5–14 µs), and QCN on/off keeps ResNet-50's ns-3 communication time inside 14.0–15.1 ms, i.e. 5–12% below the measured value in every configuration. The residual is insensitive to every knob tested. Schedule-bound workloads (CIFAR-10, both Qwen traces) are underestimated far more heavily because ns-3 models the transfer but not the synchronization wait between the collective schedule and backward computation. That component belongs to the shared trace and schedule, identical across all three topologies, so it does not enter the relative comparison.

---

## 6. Advanced: Scalability Analysis

Scale a calibrated 2-GPU model up to a 128-GPU virtual simulation.

### Step 1: Verify baseline accuracy

- Check `runs/calibration_aligned.csv`
- ResNet-50: use $\alpha_{step}$ for wall-clock conversion and treat communication time primarily as a **relative topology metric**
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

### Step 3: Inspect output metrics

Inspect `out/metrics.csv`, `stdout.log`, and related output files to review results such as:

- `sim_t_step_ms`
- communication / wall time
- per-rank statistics

The following metrics are especially useful and should be interpreted together:

- **`sim_t_step_ms`**: end-to-end step time, used to compare final execution time across topologies.
- **communication / wall time ratio**: indicates how communication-dominated the workload is; a higher ratio suggests a more communication-bound regime.
- **per-rank statistics**: useful for checking whether a subset of ranks is lagging behind, which may indicate imbalance or localized congestion.
- **`fct.txt` and related output statistics**: useful for confirming that the simulation is still making progress and for inspecting flow-completion behavior.

For topology comparison, these metrics can be read together as follows:

- If different topologies have nearly identical **`sim_t_step_ms`**, communication is likely still hidden by computation and topology effects are not yet exposed.
- If the **communication / wall time ratio increases** and `sim_t_step_ms` begins to diverge across topologies, the workload is entering a topology-sensitive regime.
- If one topology achieves a lower **`sim_t_step_ms`** under a similar communication ratio, it suggests better communication efficiency or load balancing for that workload.

---

## 7. Supporting Measurement Tools

These scripts decompose the *measured* side of the AllReduce cost, so the ns-3 gap in §5 can be attributed rather than guessed at. All outputs land under `runs/calibration/`.

| Script | What it measures | Output |
|---|---|---|
| `bucket_micro_allreduce.py` | `torch.distributed.all_reduce` at the exact DDP bucket sizes decoded from the ET, with nothing else on the GPU — the contention-free floor. Uses the PyTorch path (not rccl-tests) because that is the path the traces recorded, and brackets each individual `all_reduce` with its own CUDA event pair, mirroring how Kineto reports a kernel | `q2_micro.csv` |
| `q4_overlap_off.py` | The same buckets inside the real training loop, but issued only after backward has fully completed — framework cost present, contention absent. Phase 1 registers a DDP comm hook to record the real bucket sizes rather than guessing at `bucket_cap_mb`; the recorded multiset must match the sizes decoded from the ET | `q4_overlap_off.csv` |
| `fit_envelope.py` | Fits `T(M) = α + M/B` independently to the rccl-tests sweep and to ns-3's per-collective COMM intervals, then differences them per bucket. Ordinary least squares — the model is linear in its parameters, so no scipy | `step1_rccl_sweep.csv`, `ns3_collective_times.csv`, `envelope_fit_table.md` |
| `gen_envelope_figures.py` | Plots `T(M)` and `BW(M)` for the measured RCCL path against ns-3 | `fig_envelope_T.png`, `fig_envelope_BW.png` |
| `gen_figures_science.py` | Thesis figures (IEEE style) | `thesis_figures/` |

The three columns together are: contention-free (`q2_micro`), framework-present but contention-free (`q4_overlap_off`), and in-situ racing a backward pass (the Kineto trace). Differencing them isolates how much of the measured RCCL kernel time is transfer — which ns-3 models — and how much is synchronization wait, which it does not.

`fit_envelope.py` reads ns-3's per-collective timings from the `COMM interval` lines that `rocm/patches/statistics_comm_intervals.py` adds to ASTRA-sim's statistics pass, so the container must be built with `ASTRA_PATCHES=all` (the default) for those inputs to exist.

```bash
# inside the rocm-horovod container; /workspace/runs is bind-mounted
torchrun --nproc_per_node=2 scripts/bucket_micro_allreduce.py \
    --sizes-file runs/calibration/bucket_sizes.json \
    --out runs/calibration/q2_micro.csv

torchrun --standalone --nproc_per_node=2 scripts/q4_overlap_off.py \
    --out runs/calibration/q4_overlap_off.csv

# from the repo root
python scripts/fit_envelope.py \
    --rccl-log runs/calibration/step1_rccl_sweep_raw.log \
    --sizes runs/calibration/bucket_sizes.json \
    --out-csv runs/calibration/step1_rccl_sweep.csv
python scripts/gen_envelope_figures.py
```
