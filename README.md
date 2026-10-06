# ROCm-ASTRAsim: Trace-Driven Simulation Pipeline for AMD GPU AI Clusters

> [繁體中文](README_zh.md) | **English**

> **Thesis:** *"Cost-Effective AI Training Performance Evaluation for Torus Topology based on AMD ROCm and the Trace-Driven simulator ASTRA-sim"*
> National Cheng Kung University (NCKU), Department of Computer Science and Information Engineering, 2026

A three-stage pipeline that collects real training traces from AMD ROCm/RCCL hardware and feeds them into ASTRA-sim for cluster-scale network simulation. Most published ASTRA-sim work assumes NVIDIA CUDA/NCCL; this targets the AMD ROCm/RCCL path instead.

The thesis evaluates Fat-Tree, standard 3D Torus, and Twisted Torus at 128-node scale across **four communication regimes**:

1. **Compute-dominated AllReduce** — ResNet-50 DDP (~89.7 MiB / step)
2. **Communication-intensive AllReduce** — Qwen2.5-0.5B DDP (~1.84 GiB / step)
3. **Hierarchical TP+DDP** — Qwen2.5-1.5B (TP=8 × DDP=16)
4. **All-to-All bandwidth saturation** — synthetic stress test (1 GB / collective)

The thesis concludes that the twist's value depends on the workload: it helps bandwidth-bound All-to-All (+18%) but imposes a structural penalty on communication-intensive DDP AllReduce (+77.9% with ring, +74.7% with halving-doubling) that neither evaluated collective algorithm removes. The **standard** Torus is the recommended default — it matches Fat-Tree within 0.6% under communication-intensive DDP at roughly 58% of the cluster cost, and avoids the twist's penalty. Since standard and Twisted Torus are hardware-identical (only the cable routing differs), the twist can be adopted later at no hardware cost if a workload turns out to be All-to-All-dominated.

The configs here let you reproduce that comparison; the full analysis is in the thesis (Chapter 5), and the reference values are summarized under [Results at 128 Nodes](#results-at-128-nodes). Numbers quoted in this README come from single simulated runs and are meant as reference points, not guarantees, so re-run the configs on your own setup before relying on them.

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
│   ├── calibrate_from_runs.py     # Re-derive calibration from finished run dirs (no re-simulation)
│   ├── bucket_micro_allreduce.py  # Contention-free AllReduce timing at the ET's DDP bucket sizes
│   ├── q4_overlap_off.py          # Same buckets from a real step, each timed alone, no overlap
│   ├── fit_envelope.py            # Fit T(M) = α + M/B for the RCCL path and for ns-3, then compare
│   ├── gen_envelope_figures.py    # T(M) / BW(M) envelope figures
│   ├── gen_figures_science.py     # Thesis figures (IEEE style)
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
├── deadlock-reproduction/         # Minimal repro bundle for ASTRA-sim Issue #370 (configs + evidence)
├── docs/                          # ASTRA-sim configuration docs and historical reports
├── runs/                          # Simulation results + calibration_aligned.csv + calibration/
├── tutorials/                     # Academic tutorial materials (MICRO'24, ASPLOS'23)
├── viz/                           # Interactive 3D Twisted Torus topology visualizer
├── rocm/
│   ├── dockerfile                 # Docker environment (ROCm + PyTorch + ASTRA-sim + Chakra)
│   └── patches/                   # Versioned ASTRA-sim source patches (see ASTRA_PATCHES)
└── docker-compose.yaml
```

---

## Hardware Platform

All real-hardware measurements are performed on:

| Component | Specification |
|---|---|
| CPU | AMD Ryzen 7 5700X |
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

Classification alone is not enough: `ncclDevKernel_Generic` also carries no **payload size**. That gap is closed one stage earlier, at the profiler rather than the converter. The training scripts register a tagging DDP comm hook (`make_tagging_allreduce_hook` in `train_rocm_pytorch.py`) that wraps each RCCL collective in a `record_function` annotation — `nccl:all_reduce|bytes=<N>|pg=dp0` — which the converter parses back into `comm_type` and `comm_size`. `train_rocm_tensor.py` applies the same mechanism to the tensor-parallel collectives (`dist.all_reduce` and `torch.distributed._functional_collectives.all_reduce`, the paths DTensor's `parallelize_module` actually uses) with `pg=tp0`. Compute kernels need no patch — their durations are read straight from the Kineto device trace.

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
# 2-GPU calibration (any workload). --et-iters is how many training iterations the ET
# covers (the --trace-steps used to generate it); without it alpha_us is left blank
# rather than guessed. Results land in runs/calibration_aligned.csv.
python ./scripts/run_ns3.py \
  --workload data/chakra/workload_et --model-tag resnet50 \
  --topo auto:1d \
  --phys-topo configs/astra-sim/topos/2_nodes_1_switch_topology.txt \
  --coll-opt localBWAware --lmbw 540 --et-iters 1
```

To re-derive calibration numbers from run directories you already have, without simulating again:

```bash
python3 scripts/calibrate_from_runs.py --out runs/calibration_aligned.csv \
  resnet50=1:<run_dir> cifar10=4:<run_dir> qwen05b=2:<run_dir>
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

# Experiment 2 — Qwen 0.5B DDP, *requires* active-chunks=4 (deadlock workaround, see below).
# NOTE: --comm-scale ≈ 1.984 is the M=2 → N=128 ring-AllReduce correction 2·(N-1)/N;
# applied uniformly across topologies, so it does not affect the relative comparison.
python scripts/run_ns3.py \
  --workload data/chakra/workload_et --model-tag qwen05b \
  --topo file:configs/astra-sim/topos/logical_128nodes_TwistedTorus_4x4x8.json \
  --phys-topo configs/astra-sim/topos/128nodes_TwistedTorus_4x4x8.txt \
  --system configs/astra-sim/system/system_128nodes_TwistedTorus_4x4x8_4chunks.json \
  --virtual-world 128 --lmbw 540 --comm-scale 1.984375 --no-autocalib --no-qlen

# Experiment 2 (Twisted Torus + Halving-Doubling on X/Y dimensions) — the 2×2 factorial arm
# that isolates congestion from the topology's route structure (HD also removes the deadlock).
python scripts/run_ns3.py \
  --workload data/chakra/workload_et --model-tag qwen05b \
  --topo file:configs/astra-sim/topos/logical_128nodes_TwistedTorus_4x4x8.json \
  --phys-topo configs/astra-sim/topos/128nodes_TwistedTorus_4x4x8.txt \
  --system configs/astra-sim/system/system_128nodes_TwistedTorus_4x4x8_4chunks_hd.json \
  --virtual-world 128 --lmbw 540 --comm-scale 1.984375 --no-autocalib --no-qlen

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
| `--virtual-world N` | Replicate the per-rank trace to a `N`-node simulation (round-robin: even virtual ranks get the rank-0 ET, odd ranks the rank-1 ET; the mapping is written to `expansion_map.json`) |
| `--comm-scale F`    | Scale every COMM node's `comm_size` by `F`. This is a **workload operating-point setting**, not a correction to the collective algorithm — ASTRA-sim decomposes each collective from the configured participant count independently. The thesis uses `127/64 = 1.984375`; Qwen 0.5B needs the exact fraction so the scaled size stays divisible by `preferred-dataset-splits=4`, while TP+DDP tolerates the rounded `1.984`. Applied uniformly across all topologies within an experiment, so it does not affect relative comparisons |
| `--et-iters N`      | How many training iterations the ET covers (= the `--trace-steps` used to generate it). Needed to normalize `sim_cycles_step` to a per-step denominator for `alpha_us`; omitted → `alpha_us` is left blank rather than guessed |
| `--comm-group FILE` | File passed through as `--comm-group-configuration`; omitted → the argument is not passed at all |
| `--calib-db PATH`   | Calibration CSV to append to (default `runs/calibration_aligned.csv`) |
| `--no-qlen`         | Redirect `qlen.txt` to `/dev/null` to avoid hundreds of GB of debug output at 128-node scale |
| `--payload`         | Override ns-3 packet payload (use `12000` for the All-to-All 1 GB stress test to keep event count manageable) |
| `--no-autocalib`    | Disable automatic α calculation (only use at 2-GPU calibration; required at 128 nodes) |
| `--deadlock-timeout S` | Auto-kill if `fct.txt` stops updating for `S` seconds (default 12 h; useful for the ring-deadlock case below) |

---

## Calibration

Calibration runs at 2-GPU scale and produces the conversion factor α_step (µs / cycle) that maps simulation cycles to wall-clock time. The thesis uses it to support relative comparisons between topologies, not to predict absolute communication time.

**Window alignment comes first.** The measured side (RCCL kernels in the Kineto trace) and the simulated side (ns-3 comm cycles over the ET) do not cover the same amount of work unless the two windows are aligned. `run_ns3.py` recovers the trace's iteration count from its `ProfilerStep` intervals and checks it against `--et-iters`; if the two cannot be reconciled by an integer ratio it stops rather than emitting a plausible-looking error, and it refuses to fold a per-iteration *granularity* difference into the divisor (that case is flagged instead). **Calibration numbers produced before this alignment existed are not comparable to the ones below.**

**Measured calibration results (2-GPU, window-aligned, per training step):**

| Metric | ResNet-50 | CIFAR-10 |
|---|---|---|
| `real_t_step_ms` | 662.92 ms | 224.32 ms |
| `real_t_net_comm_ms` (measured RCCL kernels) | **15.87 ms** | **76.69 ms** |
| `real_t_kernel_ms` (GPU compute) | 284.10 ms | 50.15 ms |
| comm / step | 2.4% | 34.2% |
| ns-3 comm time | **15.06 ms** | **10.27 ms** |
| ns-3 vs measured | **−5.1%** | **−86.6%** |
| α_step (µs/cycle) | **0.002411** | 0.004550 |
| α_comm (µs/cycle) | 0.001054 | 0.007469 |
| ET iterations (`--et-iters`) | 1 | 4 |
| Status | primary baseline | excluded (scope boundary) |

- **ResNet-50 (bandwidth-dominated):** the primary benchmark. Communication is only 2.4% of step time, and ns-3's total lands within −5.1% of the measured RCCL total. That aggregate contains offsetting per-collective deviations — clean transfers are moderately overestimated, while a collective that absorbs synchronization wait from the backward pass is underestimated — so it is *aggregate* agreement, not per-collective accuracy.
- **CIFAR-10 (latency-bound):** excluded from large-scale runs. 43.5% of its step time is unmodeled residual (kernel launch, RCCL handshake, CPU scheduling, framework overhead), and the two α factors diverge by 1.64×. ASTRA-sim cannot predict absolute timing in this regime.

> Sweeping packet payload (1,000–8,000 B), per-link latency (12.5–14 µs), and QCN on/off keeps ResNet-50's ns-3 communication time inside 14.0–15.1 ms — every configuration lands 5–12% *below* the measured value. The residual is insensitive to every knob tested, and the same ns-3 transport model is applied to every link and every topology, so a topology-independent multiplicative bias cancels in the reported ratios.

α and per-run calibration results are written to `runs/calibration_aligned.csv`. `runs/calibration_all.csv` is a **legacy append log** produced by the pre-alignment pipeline; its error figures compare windows covering different amounts of work, so do not quote numbers from it. See [scripts/README.md](scripts/README.md) for the full methodology.

### Additional workload validation: Qwen 0.5B and Qwen 1.5B TP

| Workload | `--et-iters` | ns-3 comm | Measured RCCL | ns-3 vs measured | Flags |
|---|---|---|---|---|---|
| `qwen05b` | 2 | 573.08 ms | 3,393.27 ms | **−83.1%** | — |
| `qwen15b_tp` | 2 | 392.25 ms | 5,918.81 ms | **−93.4%** | `per_iter_granularity_mismatch` |

Both LLM traces are schedule-bound: ns-3 models the transfer but not the synchronization wait between the DDP bucket schedule and backward computation, which is where most of the measured RCCL kernel time sits. The Qwen 0.5B ET carries 10,916 COMP nodes and 37 AllReduce COMM nodes per step, totalling 1,884.6 MiB — exactly the FP32 gradient of 494,032,768 parameters. The `qwen15b_tp` row is flagged because the ET replays more collectives per iteration than the trace records, so its error figure needs manual interpretation before it can be quoted anywhere. Both workloads are therefore used strictly for **relative** topology comparison (thesis Sections 5.2 / 5.3), reported in raw simulation cycles.

### Supporting measurements

Three scripts decompose the measured side of the AllReduce cost, so the ns-3 gap can be attributed rather than guessed at. All write under `runs/calibration/`.

| Script | What it measures |
|---|---|
| `scripts/bucket_micro_allreduce.py` | `torch.distributed.all_reduce` at the exact DDP bucket sizes decoded from the ET, with nothing else on the GPU — the contention-free floor |
| `scripts/q4_overlap_off.py` | The same buckets from a real training step, each collective run alone on an idle GPU from inside DDP's comm hook — no overlap with backward. Only the `all_reduce` is timed; DDP's own bucketing and copies are not |
| `scripts/fit_envelope.py` | Fits `T(M) = α + M/B` independently to the rccl-tests sweep and to ns-3's per-collective COMM intervals, then differences them per bucket |

The Kineto trace supplies the third column: the same collectives racing a backward pass. `scripts/gen_envelope_figures.py` plots `T(M)` and `BW(M)` for both sides.

---

## Results at 128 Nodes

Reference values from the thesis (Chapter 5). Each is a single simulated run — re-run the configs before relying on them.

**Experiment 1 — ResNet-50 DDP (~89.7 MiB/step), compute-hidden.** All three topologies finish in the same 274,982,000 cycles (662.9 ms at α_step) with **zero** exposed communication. Raw communication cycles do differ (Fat-Tree ~18.9M, Torus 27.5M, Twisted Torus 27.6M) but sit entirely inside the compute window. Topology is invisible here.

**Experiment 2 — Qwen 0.5B DDP (1.84 GiB/step), communication-intensive.** Full 2×2 over {Torus, Twisted Torus} × {ring, halving-doubling}, all at `active-chunks=4` and `--comm-scale 1.984375`, two-iteration window:

| Topology | Algorithm | Wall (M cycles) | vs Torus+ring | PFC events | Completed flows |
|---|---|---|---|---|---|
| **Standard Torus** | **ring × 3** | **5,057** | **baseline** | 0 | 985,088 |
| Fat-Tree | halvingDoubling | 5,086 | +0.6% | 203,756 | 530,432 |
| Standard Torus | HD on X/Y | 7,789 | +54.0% | 2,850 | 833,536 |
| Twisted Torus | HD on X/Y | 8,835 | +74.7% | 1,542 | 833,536 |
| Twisted Torus | ring × 3 | 8,998 | +77.9% | 24,348 | 985,088 |

Two readings, and both matter. Against the deployment baseline (Torus + ring) **both** Twisted Torus arms are ~75% slower. Holding the *algorithm* fixed instead, the isolated twist penalty is +77.9% under ring and +13.4% under HD — HD mitigates part of it without making the Twisted Torus competitive, because HD is itself a poor fit for the symmetric standard Torus (+54.0%). HD cuts the Twisted Torus's PFC events by 94% yet buys back only 1.8% of wall time, which is why the thesis attributes the residual penalty to the twist's route-level asymmetry rather than to PFC-visible congestion. Note that PFC counts are a congestion indicator, not a wall-time predictor: Fat-Tree logs the most pauses and is the second-fastest configuration.

**Experiment 3 — Qwen 1.5B, TP=8 × DDP=16.**

| Framing | Topology | Inter-server BW | Algorithm | Wall (M cycles) | vs Torus |
|---|---|---|---|---|---|
| Cost-matched | **Torus** | 25 Gbps | ring × 3 | **7,613** | baseline |
| Cost-matched | Fat-Tree | 65 Gbps | HD, HD, ring | 7,883 | +3.5% |
| Cost-matched | Twisted Torus | 25 Gbps | ring × 3 | 11,129 | +46.2% |
| Cost-matched | Twisted Torus | 25 Gbps | HD, HD, ring | 11,446 | +50.3% |
| Bandwidth-matched | **Torus** | 65 Gbps | ring × 3 | **6,046** | baseline |
| Bandwidth-matched | Fat-Tree | 65 Gbps | HD, HD, ring | 7,883 | +30.4% |
| Bandwidth-matched | Twisted Torus | 65 Gbps | ring × 3 | 8,598 | +42.2% |

A supplementary factorial on the Twisted Torus side found the AllGather/ReduceScatter setting on X/Y makes **no** difference at either chunk level — TP collectives stay on Z, and the only collective crossing X/Y is DDP AllReduce, which `all-reduce-implementation` controls. Raising `active-chunks-per-dimension` from 1 to 4 improves TT+HD by 22.7% (11,446M → 8,845M), but even then TT+HD remains 16.2% slower than the chunks=1 Torus+ring baseline.

**Experiment 4 — All-to-All bandwidth saturation.** Wall time (cycles) as the per-collective payload grows:

| Payload | Fat-Tree | Torus | Twisted Torus | Regime |
|---|---|---|---|---|
| ~89.7 MiB (original AllReduce) | 274,982,000 | 274,982,000 | 274,982,000 | all hidden |
| 100 MB | 274,982,000 | 294,248,982 | 274,982,000 | Torus exposed first (+7%) |
| 512 MB | 879,553,198 | 1,523,300,236 | 1,288,871,156 | all exposed |
| 1 GB | 1,886,210,767 | 3,051,005,526 | 2,576,544,108 | all exposed |

At 1 GB, α-converted: Fat-Tree 4,548 ms, Twisted Torus 6,212 ms, Torus 7,356 ms — the twist is **1.18× faster** than the standard Torus, and Fat-Tree 1.62× faster than the Torus. The 1.18× ratio is stable from 512 MB to 1 GB. Google's TPU v4 reports 1.63× for the same 4×4×8 twist: direction matches, magnitude is compressed here because the twist adds path diversity precisely on this platform's low-bandwidth X/Y dimensions (25 vs. 65 Gbps).

### Cost

| Component | Fat-Tree | Torus / Twisted Torus |
|---|---|---|
| 128 × RX 9070 XT | NT$2,944,000 | NT$2,944,000 |
| 16 × EPYC server platforms | NT$3,600,000 | NT$3,600,000 |
| 48 × ConnectX-6 dual-port 100 GbE NICs | NT$1,296,000 | NT$1,296,000 |
| 24 × 100 GbE managed switches (SN2700) | **NT$5,756,400** | **NT$0** |
| **Total** | **NT$13,596,400** | **NT$7,840,000** |
| vs. NT$10M budget | 36% over | 22% under |

November 2025 market prices, US$1 = NT$30. Both families carry identical NICs, so the whole difference is the managed switch fabric — 42% of the Fat-Tree's total cost, roughly double its GPU spend. For DDP- and TP+DDP-dominated workloads the standard Torus matches Fat-Tree (within 0.6%) at ~58% of the cost; that is the thesis's recommended default. The twist adds no hardware cost (cable routing only), so it can be adopted later if a workload turns out to be All-to-All-dominated.

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
| `system_128nodes_TwistedTorus_4x4x8_4chunks.json` | 4 | ring × 3 | Experiment 2 (TT + ring arm) |
| `system_128nodes_TwistedTorus_4x4x8_4chunks_hd.json` | 4 | halvingDoubling, halvingDoubling, ring | Experiment 2 (TT + HD arm) |
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

## Experiment 2: Twisted Torus AllReduce (Qwen 0.5B)

Experiment 2 runs communication-intensive AllReduce as a 2×2 over
{Torus, Twisted Torus} × {Ring, Halving-Doubling} (all at `active-chunks=4`,
`--comm-scale 1.984375`), to tell apart two factors: network congestion and the
topology's route structure. Reference results are in
[Results at 128 Nodes](#results-at-128-nodes). The four configs to run and compare:

| Topology (physical) | System config | Algorithm |
|---|---|---|
| `128nodes_Torus_4x4x8.txt` | `system_128nodes_Torus_4x4x8_4chunks.json` | ring × 3 |
| `128nodes_Torus_4x4x8.txt` | `system_128nodes_TwistedTorus_4x4x8_4chunks_hd.json` | HD on X/Y, ring on Z |
| `128nodes_TwistedTorus_4x4x8.txt` | `system_128nodes_TwistedTorus_4x4x8_4chunks.json` | ring × 3 |
| `128nodes_TwistedTorus_4x4x8.txt` | `system_128nodes_TwistedTorus_4x4x8_4chunks_hd.json` | HD on X/Y, ring on Z |

In this experiment the Fat-Tree arm uses its natural flat mapping — a single-stage
halvingDoubling across all 128 endpoints (logical dimensions `[128]`) — while the Torus variants
use per-dimension ring over their native `[4, 4, 8]` structure, so each topology runs the
collective mapping its own structure induces.

Run all four (commands in [scripts/commands.md](scripts/commands.md)) and compare `Wall time` and
`PFC events` from each `out/metrics.csv`. What matters is how the four points relate to each
other, not the absolute numbers, which shift with your calibration and machine. Read the two
comparisons separately: *vs. the Torus+ring deployment baseline* (both twisted arms ~75% slower)
and *at fixed algorithm* (twist penalty +77.9% under ring, +13.4% under HD).

---

## Multi-Dimensional Ring Scheduling Deadlock (active-chunks-per-dimension)

> Reported upstream as [ASTRA-sim Issue #370](https://github.com/astra-sim/astra-sim/issues/370).

When running 128-node Twisted Torus AllReduce experiments at high communication intensity (Qwen 0.5B), the default `active-chunks-per-dimension=1` triggers a deterministic scheduling deadlock under `localBWAware` optimization. The Twisted Torus's asymmetric X-axis wrap-around link causes phase desynchronization across nodes, producing a cross-dimensional circular wait in ASTRA-sim's chunk queues. Symptom: ns-3 stops issuing flows at ~5,337 of an expected ~985,088 flows.

**Workaround:** Set `active-chunks-per-dimension: 4` (matching `preferred-dataset-splits: 4`). The thesis Qwen 0.5B experiments use the `*_4chunks*.json` system configurations. The `*_4chunks_hd.json` variant additionally swaps ring → halvingDoubling on the X/Y dimensions (the second arm of the 2×2 factorial), which also avoids the deadlock. `active-chunks=4` resolves the scheduler-level deadlock; it does not change the topology's route structure (see the thesis for what that implies).

The standard 3D Torus and Fat-Tree do not require this workaround because their symmetric paths keep node phase progress synchronized. Other thesis experiments (ResNet-50 AllReduce, TP+DDP, All-to-All) also do not trigger the deadlock and use the default `active-chunks=1` configurations.

**Mechanism.** Under `localBWAware`, a 3D AllReduce decomposes into five phases (RS on X, RS on Y, AllReduce on Z, AG on Y, AG on X), and the RS/AG phases of the *same* dimension share one queue — phases 0 and 4 share queue 0, phases 1 and 3 share queue 1. At `active-chunks-per-dimension=1` only one chunk may hold a queue, so once the twist desynchronizes phase progress across nodes, a fast node's next-bucket Reduce-Scatter and a slow node's current-bucket All-Gather contend for the same queue and deadlock. The layer was confirmed by instrumenting `send_flow` and `qp_finish` in the ns-3 frontend's `entry.h`: both counters stopped at exactly 5,337, proving ns-3 completed every flow it received and ASTRA-sim had stopped issuing new ones. The risk is acknowledged in ASTRA-sim's own `Sys.cc` (lines 837–852) and partly discussed in Issue #137, but the default config remains vulnerable. That instrumentation ships as `rocm/patches/entry_flow_diagnostics.py`.

**Reproducing it.** [`deadlock-reproduction/`](deadlock-reproduction/) is a self-contained bundle — the Twisted Torus topology, the deadlocking and fixed system configs, the Qwen 0.5B ET, and the captured `stdout` evidence for `chunks=1` (FIFO and LIFO) and `chunks=2`.

---

## All-to-All Stress Test (Experiment 4)

With the original ResNet-50 trace (~89.7 MiB per step), AllReduce hides behind GPU compute and the topologies look identical. To push traffic onto the network, `src/scale_et_comm_workload.py` rewrites every `COMM_COLL_NODE` in place:

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

> **Observability regimes (thesis Section 5.4.1):** the original ~89.7 MiB AllReduce trace is fully hidden by compute; small All-to-All payloads expose the topologies selectively, and at 512 MB–1 GB all three diverge. Sweep `--bytes` (100 MB → 1 GB) yourself to see where the topologies separate on your setup; the thesis reports its reference values and ratios. The 1 GB All-to-All case is a simulation-only upper-bound stress test, not a production trace — the 1 GB / collective volume also exceeds physical 16 GB VRAM at 128 nodes and is not directly executable on real hardware.

---

## Environment Setup

### Docker (recommended)

```bash
docker-compose up
# or pin a specific ROCm/PyTorch version:
VERSION=rocm6.4.4_ubuntu24.04_py3.12_pytorch_release_2.7.1 docker-compose up
# or skip the diagnostic instrumentation (build fixes only):
ASTRA_PATCHES=none docker-compose up --build
```

Both knobs live in `.env`. The image pins Chakra to mlcommons upstream `ec41090` (not the astra-sim
fork, which held no unique commits) and force-upgrades `protobuf`, because the base image ships
3.20.2 while upstream dropped the `protobuf==5.*` cap — a build-time import check catches the
resulting gencode/runtime mismatch instead of letting it surface mid-run.

`ASTRA_PATCHES` selects which ASTRA-sim source patches from `rocm/patches/` are applied:

| Patch | Applied at | Purpose |
|---|---|---|
| `spdlog_fmt_compat.py` | always | Adds the `fmt` include `spdlog_setup` needs for the ns-3 frontend to build |
| `statistics_comm_intervals.py` | `all` | Logs one line per COMM interval; ASTRA-sim otherwise reports only the merged total, so per-collective cost can't be recovered from a run |
| `entry_flow_diagnostics.py` | `all` | `send_flow` / `qp_finish` counters in the ns-3 frontend — the instrumentation that localized the Issue #370 deadlock |

The two `all`-only patches are read-only instrumentation: they log what is already being computed and do not change `type_time` or any simulation result.

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
A: That is the multi-dimensional ring scheduling deadlock described above (also in thesis Section 6.2.7). Use the `*_4chunks*.json` system configuration so that `active-chunks-per-dimension=4`. The `*_4chunks_hd.json` variant (ring → halvingDoubling on X/Y) is the second arm of the 2×2 factorial and also avoids the deadlock. Note that `active-chunks=4` resolves the scheduler-level deadlock but does not change the topology's route structure; see the thesis (Chapter 5 / Section 6.2.7) for what each arm implies.

**Q: `"Node X in ctrl_dep graph, but not found in index"` error from ASTRA-sim?**
A: The ET file has a DAG integrity issue (self-dependency or cycle). Run `conver_to_chakra_et.py` again — the built-in DAG repair pass (`fix_et_dag_inplace`) should resolve this automatically.

**Q: `chakra_trace_link` fails on ROCm with misaligned timestamps?**
A: Add `--inject-sync-hack` to the trace-collection script. This injects synchronization events to align CPU (ms) and GPU (µs) timestamps before trace linking.

**Q: How accurate is ns-3's communication time against the hardware?**
A: For ResNet-50, once the measured and simulated windows are aligned, ns-3's total is **5.1% below** the measured RCCL kernel total — aggregate agreement, containing offsetting per-collective deviations. No tunable parameter moves it (payload, latency, QCN all leave it in 14.0–15.1 ms). Schedule-bound workloads are underestimated much more heavily (CIFAR-10 −86.6%, Qwen 0.5B −83.1%), because the simulator models the transfer but not the synchronization wait between the collective schedule and backward computation. That unmodeled component belongs to the shared trace and schedule, identical across all three topologies, so it does not enter the relative comparison.

> An older revision of this README reported ns-3 *overestimating* ResNet-50 by ~2×. That figure came from comparing windows covering different amounts of work and is superseded — see **Window alignment comes first** above.

**Q: Why is CIFAR-10 excluded from large-scale evaluation?**
A: Its shallow architecture leaves 43.5% of step time in unmodeled residual (kernel launch, RCCL handshake, CPU scheduling, framework overhead). The wall-clock and communication calibration factors diverge by 1.64×, making ASTRA-sim unsuitable for absolute prediction in this latency-dominated regime. See thesis Section 4.3 for details.

**Q: What is `--comm-scale 1.984375` actually doing?**
A: It sets the **communication operating point** of the replicated 128-node run — it is not a correction to the collective algorithm, which ASTRA-sim derives independently from the configured participant count. The value is the exact fraction `127/64`. Qwen 0.5B needs the exact fraction so the scaled `comm_size` stays evenly divisible by `preferred-dataset-splits=4`; TP+DDP tolerates the rounded `1.984`. Within each experiment the same multiplier and the same trace go to all three topologies, so every comparison is made at the same offered load. See thesis Sections 4.2.6 / 4.6.2.

**Q: `alpha_us` came out blank in my calibration row.**
A: You didn't pass `--et-iters`. `alpha_us` needs a per-step denominator, and the script leaves it blank rather than guessing an iteration count — a guessed α produces a plausible-looking conversion factor that is silently wrong. Pass the `--trace-steps` value you used when generating the ET.

**Q: My calibration row carries a `per_iter_granularity_mismatch` flag.**
A: The ET replays a different number of collectives per iteration than the trace recorded, so the two sides are not summing the same batch of work. The run is not stopped, but the error value needs manual interpretation before it can be quoted. The `qwen15b_tp` row is in this state.

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
@mastersthesis{chen2026torus,
  author  = {jjasoncool},
  title   = {Cost-Effective AI Training Performance Evaluation for Torus Topology
             based on AMD ROCm and the Trace-Driven simulator ASTRA-sim},
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
