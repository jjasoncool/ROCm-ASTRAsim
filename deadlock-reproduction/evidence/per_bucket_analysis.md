# Per-Bucket Phase Completion Analysis

## Method

Each completed flow in `fct.txt` is categorized by:
- **Tag** (= bucket number): tag 10000 = bucket 0, tag 10001 = bucket 1, etc.
- **Direction**: determined by (src, dst) node coordinates in the 4×4×8 topology

Abbreviations:
- D0f = Dim 0 forward (ReduceScatter Phase 0)
- D0b = Dim 0 backward (AllGather Phase 4)
- D1f = Dim 1 forward (ReduceScatter Phase 1)
- D1b = Dim 1 backward (AllGather Phase 3)
- D2f = Dim 2 forward (AllReduce Phase 2)

128 = all nodes completed that phase for this bucket.

## active-chunks-per-dimension = 1 (deadlock at 5,337 flows)

| Bucket | D0f (RS) | D0b (AG) | D1f (RS) | D1b (AG) | D2f (AR) |
|--------|----------|----------|----------|----------|----------|
| B0     | 128      | 88       | 128      | 103      | 128      |
| B1     | 128      | 60       | 128      | 93       | 128      |
| B2     | 128      | 32       | 128      | 90       | 128      |
| B3     | 128      | **0**    | 119      | **0**    | 128      |
| B4     | 128      | 0        | 111      | 0        | 128      |
| B5     | 128      | 0        | 103      | 0        | 128      |
| B6     | 128      | 0        | 90       | 0        | 128      |
| B7–B11 | 128      | 0        | 88       | 0        | 128      |
| B12    | 0        | 0        | 0        | 0        | 128      |
| B13–B20| 0        | 0        | 0        | 0        | decreasing (128→8) |

## active-chunks-per-dimension = 2 (deadlock at 7,384 flows)

| Bucket | D0f (RS) | D0b (AG) | D1f (RS) | D1b (AG) | D2f (AR) |
|--------|----------|----------|----------|----------|----------|
| B0     | 128      | 60       | 128      | 99       | 128      |
| B1     | 128      | 60       | 128      | 94       | 128      |
| B2     | 128      | 24       | 128      | 76       | 128      |
| B3     | 128      | 24       | 128      | 69       | 128      |
| B4     | 128      | 12       | 128      | 62       | 128      |
| B5     | 128      | 12       | 128      | 62       | 128      |
| B6     | 128      | **0**    | 105      | **0**    | 128      |
| B7–B11 | 128      | 0        | 66–101   | 0        | 128      |
| B12–B27| 0        | 0        | 0        | 0        | 128      |
| B28–B36| 0        | 0        | 0        | 0        | decreasing (68→4) |

## Key Observations

1. **All forward (ReduceScatter) phases complete first**. Only backward (AllGather) phases stall.
2. **AG starvation is progressive**: D0b and D1b decrease over buckets, then drop to 0.
3. **AG death time scales with active-chunks**: dies at B3 for chunks=1, B6 for chunks=2 (2× linear).
4. **D2f (AllReduce, sole occupant of Queue 2)** survives longest — no queue contention.
5. **After AG dies, RS cascades**: nodes waiting for AG cannot start next RS → pipeline collapse.

## Control Experiment: Standard Torus + active-chunks=1

The identical system configuration on a Standard (non-twisted) Torus 4×4×8 completes
successfully with all 128 nodes finishing the full workload.

- Topology: `128nodes_Torus_4x4x8.txt` (symmetric path latency)
- System config: identical to `system_deadlock.json` (active-chunks=1)
- Result: all 128 nodes completed, wall time = 4,057,223,220 cycles

This confirms the deadlock is triggered specifically by asymmetric path latency,
not by the `active-chunks=1` setting alone.
