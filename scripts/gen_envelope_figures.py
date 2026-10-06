#!/usr/bin/env python3
"""Envelope figures: T(M) and BW(M) for the measured RCCL path vs ns-3.

Data sources (all produced 2026-08-20/21, all traceable to run products):
  runs/calibration/step1_rccl_sweep.csv - rccl-tests all_reduce_perf, FP32, 5 repeats
  runs/calibration/q2_micro.csv         - torch.distributed.all_reduce at DDP bucket sizes
  ns-3 points                           - per-collective COMM intervals from the three
                                          2-node runs under runs/20260820-055114*
"""
import csv
import statistics

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

try:
    import scienceplots  # noqa: F401
    plt.style.use(['science', 'ieee'])
except Exception:
    try:
        plt.style.use(['science', 'no-latex', 'ieee'])
    except Exception:
        plt.rcParams.update({'font.family': 'serif', 'font.size': 9,
                             'axes.grid': True, 'grid.alpha': 0.3})

plt.rcParams.update({'figure.dpi': 300, 'savefig.dpi': 300,
                     'savefig.bbox': 'tight', 'savefig.pad_inches': 0.05})

MIB = 1024.0 ** 2
C_REAL = '#0072B2'   # rccl-tests
C_MICRO = '#009E73'  # PyTorch path
C_SIM = '#D55E00'    # ns-3
C_GRAY = '#666666'

# Fitted envelopes (see scripts/fit_envelope.py output)
A_REAL, B_REAL = 0.04203, 7.7727   # ms, MiB/ms
A_SIM, B_SIM = 0.43927, 6.9725

SIM_POINTS = {
    1487360: 0.643752, 4292648: 1.026276, 9693440: 1.764972,
    17436160: 2.824024, 26255360: 4.030260, 26550272: 4.070604,
    27291648: 4.172044, 34865152: 5.207952, 42216960: 6.213520,
    67149864: 9.623732, 569323008: 78.309892,
}


def load_rccl(path='runs/calibration/step1_rccl_sweep.csv'):
    by = {}
    with open(path) as f:
        for r in csv.DictReader(f):
            if r['inplace'] == '1':
                by.setdefault(int(r['payload_bytes']), []).append(float(r['time_us']) / 1000.0)
    return {k: statistics.median(v) for k, v in sorted(by.items())}


def load_micro(path='runs/calibration/q2_micro.csv'):
    by = {}
    with open(path) as f:
        for r in csv.DictReader(f):
            by.setdefault(int(r['payload_bytes']), []).append(float(r['median_ms']))
    return {k: statistics.median(v) for k, v in sorted(by.items())}


def main():
    rccl = load_rccl()
    micro = load_micro()

    grid = np.logspace(np.log10(8 / MIB), np.log10(768), 400)

    # ---------- Figure A: T(M) ----------
    fig, ax = plt.subplots(figsize=(4.6, 3.3))
    ax.plot(grid, A_REAL + grid / B_REAL, color=C_REAL, lw=1.0, zorder=2,
            label=rf'fit: $\alpha$=42.0 $\mu$s, B=8.15 GB/s')
    ax.plot(grid, A_SIM + grid / B_SIM, color=C_SIM, lw=1.0, ls='--', zorder=2,
            label=rf'fit: $\alpha$=439.3 $\mu$s, B=7.31 GB/s')
    ax.scatter([k / MIB for k in rccl], list(rccl.values()), s=11, color=C_REAL,
               marker='o', zorder=4, label='rccl-tests (FP32, 5 runs)')
    ax.scatter([k / MIB for k in micro], list(micro.values()), s=16, color=C_MICRO,
               marker='^', zorder=4, label='PyTorch all\\_reduce (bucket sizes)')
    ax.scatter([k / MIB for k in SIM_POINTS], list(SIM_POINTS.values()), s=20,
               color=C_SIM, marker='s', zorder=4, label='ns-3 + ASTRA-sim')

    ax.set_xscale('log')
    ax.set_yscale('log')
    ax.set_xlabel('Message size (MiB)')
    ax.set_ylabel('AllReduce time (ms)')
    ax.legend(fontsize=5.8, loc='upper left', framealpha=0.9)
    plt.tight_layout()
    plt.savefig('runs/calibration/fig_envelope_T.png')
    plt.close()
    print('  wrote runs/calibration/fig_envelope_T.png')

    # ---------- Figure B: BW(M) + overestimate ratio ----------
    fig, ax1 = plt.subplots(figsize=(4.6, 3.3))
    bw = lambda m, t: m * MIB / (t / 1000.0) / 1e9
    ax1.plot(grid, [bw(m, A_REAL + m / B_REAL) for m in grid], color=C_REAL, lw=1.0)
    ax1.plot(grid, [bw(m, A_SIM + m / B_SIM) for m in grid], color=C_SIM, lw=1.0, ls='--')
    ax1.scatter([k / MIB for k in rccl], [bw(k / MIB, v) for k, v in rccl.items()],
                s=11, color=C_REAL, marker='o', zorder=4, label='measured (rccl-tests)')
    ax1.scatter([k / MIB for k in micro], [bw(k / MIB, v) for k, v in micro.items()],
                s=16, color=C_MICRO, marker='^', zorder=4, label='measured (PyTorch)')
    ax1.scatter([k / MIB for k in SIM_POINTS],
                [bw(k / MIB, v) for k, v in SIM_POINTS.items()],
                s=20, color=C_SIM, marker='s', zorder=4, label='ns-3 + ASTRA-sim')
    ax1.axhline(8.15, color=C_GRAY, ls=':', lw=0.8)
    ax1.text(0.011, 8.4, 'platform goodput 8.15 GB/s', fontsize=5.8, color=C_GRAY)
    ax1.set_xscale('log')
    ax1.set_xlabel('Message size (MiB)')
    ax1.set_ylabel('Achieved bandwidth (GB/s)')
    ax1.set_ylim(0, 10.5)
    ax1.legend(fontsize=5.8, loc='lower right', framealpha=0.9)

    ax2 = ax1.twinx()
    ratio_x = sorted(SIM_POINTS)
    ax2.plot([k / MIB for k in ratio_x],
             [SIM_POINTS[k] / (A_REAL + (k / MIB) / B_REAL) for k in ratio_x],
             color='#7B68EE', lw=0.9, marker='d', ms=3, zorder=3)
    ax2.set_ylabel(r'ns-3 / measured', color='#7B68EE', fontsize=8)
    ax2.tick_params(axis='y', labelcolor='#7B68EE')
    ax2.set_ylim(0, 3.4)
    ax2.axhline(1.0, color='#7B68EE', ls=':', lw=0.6)

    plt.tight_layout()
    plt.savefig('runs/calibration/fig_envelope_BW.png')
    plt.close()
    print('  wrote runs/calibration/fig_envelope_BW.png')


if __name__ == '__main__':
    main()
