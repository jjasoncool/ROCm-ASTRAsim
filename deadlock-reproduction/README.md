# Deadlock Reproduction: `localBWAware` Ring AllReduce on Twisted Torus

This directory contains the minimal files needed to reproduce a deterministic deadlock
in ASTRA-sim's multi-dimensional ring AllReduce scheduler.

**Related Issue**: [astra-sim/astra-sim#137](https://github.com/astra-sim/astra-sim/issues/137)

## Quick Start

```bash
# Deadlock (hangs after ~5,337 flows):
./build/scratch/ns3.42-AstraSimNetwork-default \
  --workload-configuration=<path>/workload_128/et.qwen05b \
  --system-configuration=configs/system_deadlock.json \
  --network-configuration=<ns3-config-pointing-to-configs/128nodes_TwistedTorus_4x4x8.txt> \
  --remote-memory-configuration=<path>/remote_memory.json \
  --logical-topology-configuration=configs/logical_128nodes_TwistedTorus_4x4x8.json

# Fixed (completes all 37 buckets):
# Same command, but use configs/system_fixed.json instead
```

The only difference between the two configs is `active-chunks-per-dimension`: 1 (deadlock) vs 4 (success).

## File Listing

```
configs/
  128nodes_TwistedTorus_4x4x8.txt          Physical topology (128 nodes + 128 switches)
  logical_128nodes_TwistedTorus_4x4x8.json  Logical dims: [4, 4, 8]
  system_deadlock.json                       active-chunks=1 (triggers deadlock)
  system_fixed.json                          active-chunks=4 (workaround)

workload/
  et.qwen05b.0.et                           Chakra ET trace (even nodes)
  et.qwen05b.1.et                           Chakra ET trace (odd nodes)

evidence/
  stdout_chunks1_lifo.log     Trimmed log: chunks=1, LIFO (5,337 flows → hang)
  stdout_chunks1_fifo.log     Trimmed log: chunks=1, FIFO (5,337 flows → hang, identical)
  stdout_chunks2.log          Trimmed log: chunks=2 (7,384 flows → hang)
  per_bucket_analysis.md      Phase-by-phase breakdown of completed flows
```

## Trigger Conditions

The deadlock was observed when all three conditions are present:

1. **Asymmetric path latency**: Twisted Torus wrap-around creates unequal ring-step delays. *Verified*: Standard Torus with the same configuration does not deadlock.
2. **`active-chunks-per-dimension` < `preferred-dataset-splits`**: queue can only hold 1–3 streams, insufficient for RS+AG coexistence. *Verified*: chunks=1 and 2 deadlock; chunks=4 does not.
3. **Sufficient workload size**: the pipeline must be deep enough to create cross-bucket queue contention. *Observed*: Qwen 0.5B (37 buckets, 1,884 MiB) triggers it; ResNet-50 does not.

All experiments use `collective-optimization: localBWAware`, which triggers the 5-phase RS/AG decomposition where Phase 0/4 share Queue 0. See the Root Cause section in the linked Issue for details.

## Workload

The workload used is Qwen 0.5B AllReduce traces captured via PyTorch + ROCm and converted
to Chakra ET format. Two source traces are provided:

```
workload/
  et.qwen05b.0.et    (25 MB)
  et.qwen05b.1.et    (25 MB)
```

These 2 traces are round-robin assigned to 128 nodes: `node_r` uses `et.qwen05b.{r % 2}.et`.
The `run_ns3.py` script handles this expansion automatically via `--virtual-world 128`.

Any workload with ≥10 AllReduce buckets totaling ≥500 MiB should trigger the same deadlock
under the above conditions. ResNet-50 (small communication volume) does NOT trigger it.
