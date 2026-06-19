#!/usr/bin/env python3
"""
Thesis Figure Generator — IEEE/ACM Publication Quality
Uses SciencePlots for professional academic styling.

=== CONDA 安裝指令 ===
conda create -p ./env python=3.12
conda activate ./env
conda install -c conda-forge matplotlib numpy
pip install SciencePlots

# SciencePlots 需要 LaTeX 引擎來渲染字體：
# Ubuntu/Debian:
sudo apt install texlive-latex-base texlive-fonts-recommended texlive-fonts-extra dvipng cm-super

========================

All data from confirmed experimental results.
α = 0.002411 μs/cycle for time conversion.
"""

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import scienceplots
import matplotlib.ticker as ticker
import numpy as np
import os

# ============================================================
# Style Configuration
# ============================================================
try:
    plt.style.use(['science', 'ieee'])
    print("[OK] Using SciencePlots with LaTeX (best quality)")
except Exception:
    try:
        plt.style.use(['science', 'no-latex', 'ieee'])
        print("[OK] Using SciencePlots without LaTeX (good quality)")
    except Exception:
        print("[WARN] SciencePlots not found, using fallback style")
        plt.rcParams.update({
            'font.family': 'serif',
            'font.serif': ['Times New Roman', 'DejaVu Serif'],
            'font.size': 9,
            'axes.labelsize': 10,
            'axes.titlesize': 11,
            'xtick.labelsize': 8,
            'ytick.labelsize': 8,
            'legend.fontsize': 8,
            'figure.dpi': 300,
            'savefig.dpi': 300,
            'savefig.bbox': 'tight',
            'savefig.pad_inches': 0.05,
            'axes.grid': True,
            'grid.alpha': 0.3,
            'axes.linewidth': 0.6,
            'lines.linewidth': 1.0,
        })

plt.rcParams.update({
    'figure.dpi': 300,
    'savefig.dpi': 300,
    'savefig.bbox': 'tight',
    'savefig.pad_inches': 0.05,
})

# Color palette — colorblind-safe, print-friendly
C_BLUE   = '#0072B2'  # Fat-Tree
C_ORANGE = '#E69F00'  # Standard Torus
C_GREEN  = '#009E73'  # Twisted Torus
C_RED    = '#D55E00'  # Error / alert
C_GRAY   = '#666666'  # Compute / neutral

TOPO_COLORS = [C_BLUE, C_ORANGE, C_GREEN]
TOPO_LABELS = ['Fat-Tree\n(L16\_S8)', 'Standard\nTorus (4×4×8)', 'Twisted\nTorus (4×4×8)']
TOPO_LABELS_SHORT = ['Fat-Tree', 'Std. Torus', 'Twisted Torus']

alpha_us = 0.002411  # μs/cycle (updated: bug B1/B2/B3 fix in run_ns3.py)

outdir = './thesis_figures'
os.makedirs(outdir, exist_ok=True)


# ============================================================
# Data — confirmed experimental results
# ============================================================

# 128-node All-to-All Comm time (cycles) — from stdout.log
# Used for Fig 4.3 calibration validation
COMM_CYCLES_128_A2A_100MB = {
    'Fat-Tree':       150_637_472,
    'Std. Torus':     294_248_982,   # note: comm_equals_wall flag
    'Twisted Torus':  252_528_351,
}

# 128-node All-to-All Wall time (cycles)
# AllReduce (original trace, ~89.7 MiB)
WALL_ALLREDUCE = {
    'Fat-Tree':       274_982_000,
    'Std. Torus':     274_982_000,
    'Twisted Torus':  274_982_000,
}

# All-to-All ~1KB (original comm_size, type changed to A2A)
WALL_A2A_1KB = {
    'Fat-Tree':       274_982_000,
    'Std. Torus':     274_982_000,
    'Twisted Torus':  274_982_000,
}

# All-to-All 100MB
WALL_A2A_100MB = {
    'Fat-Tree':       274_982_000,
    'Std. Torus':     294_248_982,
    'Twisted Torus':  274_982_000,
}

# All-to-All 512MB
WALL_A2A_512MB = {
    'Fat-Tree':       879_553_198,
    'Std. Torus':     1_523_300_236,
    'Twisted Torus':  1_288_871_156,
}

# All-to-All 1GB
WALL_A2A_1GB = {
    # 正式值 (metrics CSV, run_dirs 20260302-052720/053303/053448);
    # 舊值 1_886_164_000/3_049_068_000/2_575_180_000 為誤植,已棄用
    'Fat-Tree':       1_886_210_767,
    'Std. Torus':     3_051_005_526,
    'Twisted Torus':  2_576_544_108,
}

GPU_CYCLES = 274_982_000

# Qwen 0.5B DDP AllReduce (128 nodes, cost-matched)
# Clean canonical data (verified 2026-06-09 from machine output, double-checked).
# Supersedes the earlier contaminated values produced by two compounding bugs
# (comm-scale 1.0 instead of 127/64, and rounding 1.984 breaking split divisibility).
QWEN_DDP = {
    'Torus + ring':    5_056_827_752,
    'Fat-Tree + HD':   5_086_430_904,
    'TT + ring':       8_997_989_486,
    'TT + HD':         8_834_772_125,
}
QWEN_PFC = {
    'Torus + ring':    0,
    'Fat-Tree + HD':   203_756,
    'TT + ring':       24_348,
    'TT + HD':         1_542,
}


# ============================================================
# Fig 4.4: Scope Boundary — Step Time Composition
# Horizontal stacked bar: comm 1.6% vs 40% is the key contrast
# ============================================================
def fig_4_4():
    fig, ax = plt.subplots(figsize=(5.0, 2.2))

    # Percentages of step time (updated after B1/B2/B3 fix)
    # ResNet-50: total=662.92, gpu=141.70(21.4%), comm=7.54(1.1%), overhead=513.68(77.5%)
    # CIFAR-10:  total=224.32, gpu=39.96(17.8%), comm=61.98(27.6%), overhead=122.38(54.6%)
    workloads = ['CIFAR-10 CNN\n(Latency-bound)', 'ResNet-50\n(Bandwidth-bound)']
    gpu_pct   = [17.8, 21.4]
    comm_pct  = [27.6,  1.1]
    over_pct  = [54.6, 77.5]

    y = np.arange(2)
    h = 0.5

    ax.barh(y, gpu_pct, h, label='GPU compute',
            color=C_BLUE, edgecolor='black', linewidth=0.3, zorder=3)
    ax.barh(y, comm_pct, h, left=gpu_pct,
            label='Network comm (ns-3 models this)',
            color=C_GREEN, edgecolor='black', linewidth=0.3, zorder=3)
    ax.barh(y, over_pct, h, left=[g+c for g, c in zip(gpu_pct, comm_pct)],
            label='Framework + OS overhead (not modeled)',
            color='#CCCCCC', edgecolor='black', linewidth=0.3, zorder=3)

    # Labels inside bars
    for i in range(2):
        # GPU
        if gpu_pct[i] > 8:
            ax.text(gpu_pct[i]/2, y[i], f'{gpu_pct[i]:.0f}\\%',
                    ha='center', va='center', fontsize=7, fontweight='bold', color='white')
        # Comm
        cx = gpu_pct[i] + comm_pct[i]/2
        if comm_pct[i] > 5:
            ax.text(cx, y[i], f'{comm_pct[i]:.1f}\\%',
                    ha='center', va='center', fontsize=7, fontweight='bold', color='white')
        else:
            # Too narrow — separate text + diagonal arrow from bar to label
            mid_y = (y[0] + y[1]) / 2
            label_x = gpu_pct[i] + comm_pct[i] + 5
            ax.text(label_x, mid_y, f'comm = {comm_pct[i]:.1f}\\%',
                    fontsize=7, fontweight='bold', color=C_GREEN, va='center')
            # Arrow from bar area (upper-left) to lower-right
            ax.annotate('',
                        xytext=(gpu_pct[i] + comm_pct[i]/2, y[i] + 0.12),
                        xy=(label_x - 1, mid_y - 0.02),
                        arrowprops=dict(arrowstyle='->', color=C_GREEN, lw=0.8))
        # Overhead
        ox = gpu_pct[i] + comm_pct[i] + over_pct[i]/2
        ax.text(ox, y[i], f'{over_pct[i]:.0f}\\%',
                ha='center', va='center', fontsize=7, fontweight='bold', color='#555555')

    # Absolute times as annotation
    ax.text(101, y[1], '662.92 ms', va='center', fontsize=7, color=C_GRAY)
    ax.text(101, y[0], '224.32 ms', va='center', fontsize=7, color=C_GRAY)

    ax.set_xlim(0, 115)
    ax.set_yticks(y)
    ax.set_yticklabels(workloads, fontsize=9)
    ax.set_xlabel('Proportion of step time (\\%)', fontsize=9)
    ax.legend(loc='upper center', bbox_to_anchor=(0.5, -0.28), ncol=3, fontsize=6.5, framealpha=0.9)
    ax.set_title('Calibration Scope Boundary: Step Time Composition', fontsize=10)
    plt.tight_layout()
    plt.savefig(f'{outdir}/fig_4_4_scope_boundary.pdf')
    plt.savefig(f'{outdir}/fig_4_4_scope_boundary.png')
    plt.close()
    print("  Fig 4.4 done")



# (Fig 4.5 parameter sweep heatmap is produced from drawio, not Python)


# ============================================================
# Fig 5.1: AllReduce Wall Time (unchanged)
# ============================================================
def fig_5_1():
    fig, ax = plt.subplots(figsize=(3.5, 2.5))

    wall_cycles = [WALL_ALLREDUCE[k] for k in ['Fat-Tree', 'Std. Torus', 'Twisted Torus']]
    compute_ms = [c * alpha_us / 1000 for c in wall_cycles]

    x = np.arange(3)
    bars = ax.bar(x, compute_ms, 0.45,
                  color=TOPO_COLORS, edgecolor='black', linewidth=0.4, zorder=3)

    ax.set_ylabel('Wall Time (ms)')
    ax.set_xticks(x)
    ax.set_xticklabels(TOPO_LABELS_SHORT, fontsize=8)
    ax.set_ylim(0, 1000)

    for i, bar in enumerate(bars):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 15,
                f'{compute_ms[i]:.0f} ms', ha='center', va='bottom', fontsize=7)

    ax.text(0.5, 0.97, 'Exposed communication = 0\n(all topologies identical)',
            ha='center', va='top', fontsize=7, fontstyle='italic', color=C_GRAY,
            transform=ax.transAxes,
            bbox=dict(boxstyle='round,pad=0.3', facecolor='#f0f0f0',
                      edgecolor='#cccccc', alpha=0.8))

    ax.set_title('AllReduce Experiment (128 nodes, ResNet-50)')
    plt.tight_layout()
    plt.savefig(f'{outdir}/fig_5_1_allreduce.pdf')
    plt.savefig(f'{outdir}/fig_5_1_allreduce.png')
    plt.close()
    print("  Fig 5.1 done")


# ============================================================
# Fig 5.2: Qwen 0.5B DDP — Twist Imposes an Algorithm-Independent Penalty
# Grouped bar chart. Two short bars (Std. Torus, Fat-Tree) vs two tall bars
# (both Twisted Torus variants). Three messages encoded:
#   (1) grouping:   Std. Torus ~= Fat-Tree   << both TT variants
#   (2) reversal:   HD helps TT by only 1.8% (algorithm cannot rescue the twist)
#   (3) PFC labels: congestion does NOT predict step time (FT highest PFC yet fast;
#                   TT+HD lowest PFC yet slow) -> bottleneck is path length, not congestion
# Both TT bars share one colour to visually convey "algorithm-independent".
# ============================================================
def fig_5_2():
    fig, ax = plt.subplots(figsize=(4.8, 3.3))

    order = ['Torus + ring', 'Fat-Tree + HD', 'TT + ring', 'TT + HD']
    labels = ['3D Torus\n(ring)', 'Fat-Tree\n(HD)', 'Twisted Torus\n(ring)', 'Twisted Torus\n(HD)']
    values_M = [QWEN_DDP[k] / 1e6 for k in order]
    pfc      = [QWEN_PFC[k] for k in order]
    # Std. Torus = orange, Fat-Tree = blue, both Twisted Torus = red (same colour on purpose)
    colors = [C_ORANGE, C_BLUE, C_RED, C_RED]

    x = np.arange(4)
    bars = ax.bar(x, values_M, 0.6, color=colors, edgecolor='black', linewidth=0.4, zorder=3)

    ax.set_ylabel('Wall Time (M cycles)')
    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=7)

    # Wall-time value (bold) + PFC count (italic grey) stacked above each bar
    for i, bar in enumerate(bars):
        cx = bar.get_x() + bar.get_width() / 2
        ax.text(cx, bar.get_height() + 120, f'{values_M[i]:,.0f}',
                ha='center', va='bottom', fontsize=7.5, fontweight='bold')
        ax.text(cx, bar.get_height() + 120 + 580, f'PFC {pfc[i]:,}',
                ha='center', va='bottom', fontsize=5.8, color=C_GRAY, fontstyle='italic')

    ax.set_ylim(0, 13200)

    # (1) Grouping bracket over the two short bars
    gb = max(values_M[0], values_M[1]) + 1750
    ax.plot([0, 0], [gb - 220, gb], color=C_GRAY, lw=0.7)
    ax.plot([1, 1], [gb - 220, gb], color=C_GRAY, lw=0.7)
    ax.plot([0, 1], [gb, gb], color=C_GRAY, lw=0.7)
    ax.text(0.5, gb + 100, r'Std. Torus $\approx$ Fat-Tree (+0.6\%)',
            ha='center', va='bottom', fontsize=6.0, color=C_GRAY)

    # (2) Reversal bracket over the two tall (Twisted Torus) bars
    rb_top = max(values_M[2], values_M[3]) + 1750
    ax.plot([2, 2], [values_M[2] + 1300, rb_top], color=C_RED, lw=0.7)
    ax.plot([3, 3], [values_M[3] + 1300, rb_top], color=C_RED, lw=0.7)
    ax.plot([2, 3], [rb_top, rb_top], color=C_RED, lw=0.7)
    ax.text(2.5, rb_top + 100, r'HD only $-1.8\%$',
            ha='center', va='bottom', fontsize=6.8, color=C_RED, fontweight='bold')

    # (3) "+~75% structural penalty" arrow in the mid gap (does not overlap any bar)
    ax.annotate('', xy=(1.9, 9300), xytext=(1.05, 6500),
                arrowprops=dict(arrowstyle='->', color='black', lw=1.0))
    ax.text(1.4, 8700, 'twist: $+\\sim$75\\%\n(ring \\& HD)',
            ha='center', va='bottom', fontsize=6.5, fontweight='bold')

    ax.set_title('Qwen 0.5B DDP AllReduce (128 nodes)')
    ax.legend(
        [plt.Rectangle((0, 0), 1, 1, fc=c, ec='black', lw=0.4) for c in [C_ORANGE, C_BLUE, C_RED]],
        ['Std. Torus', 'Fat-Tree', 'Twisted Torus'],
        loc='upper left', framealpha=0.9, fontsize=6, bbox_to_anchor=(0.0, 0.78)
    )
    plt.tight_layout()
    plt.savefig(f'{outdir}/fig_5_2_qwen_ddp.pdf')
    plt.savefig(f'{outdir}/fig_5_2_qwen_ddp.png')
    plt.close()
    print("  Fig 5.2 done")


# ============================================================
# Fig 5.3: All-to-All 1GB Stacked Bar (was Fig 5.2)
# ============================================================
def fig_5_3():
    fig, ax = plt.subplots(figsize=(3.5, 3.0))

    wall_ms = [
        WALL_A2A_1GB['Fat-Tree'] * alpha_us / 1000,
        WALL_A2A_1GB['Std. Torus'] * alpha_us / 1000,
        WALL_A2A_1GB['Twisted Torus'] * alpha_us / 1000,
    ]
    gpu_ms = [GPU_CYCLES * alpha_us / 1000] * 3
    exposed = [w - g for w, g in zip(wall_ms, gpu_ms)]

    x = np.arange(3)
    width = 0.45

    ax.bar(x, gpu_ms, width, label='GPU Compute',
           color=C_GRAY, edgecolor='black', linewidth=0.4, zorder=3)
    ax.bar(x, exposed, width, bottom=gpu_ms, label='Exposed Comm.',
           color=TOPO_COLORS, edgecolor='black', linewidth=0.4, zorder=3)

    ax.set_ylabel('Wall Time (ms)')
    ax.set_xticks(x)
    ax.set_xticklabels(TOPO_LABELS_SHORT, fontsize=8)
    ax.set_ylim(0, 8800)
    ax.legend(loc='upper left', framealpha=0.9, fontsize=7)

    for i, w in enumerate(wall_ms):
        ax.text(x[i], w + 100, f'{w:,.0f} ms', ha='center', va='bottom',
                fontweight='bold', fontsize=8)

    rel_labels = ['1.00×', '1.62×', '1.37×\n(18\% faster\nthan Std.)']
    for i, label in enumerate(rel_labels):
        ax.annotate(label, xy=(x[i], wall_ms[i] + 700), fontsize=6.5,
                    ha='center', color=TOPO_COLORS[i], fontweight='bold')

    ax.set_title('All-to-All Stress Test (128 nodes, 1 GB)')
    plt.tight_layout()
    plt.savefig(f'{outdir}/fig_5_3_all2all.pdf')
    plt.savefig(f'{outdir}/fig_5_3_all2all.png')
    plt.close()
    print("  Fig 5.3 done")


# ============================================================
# Fig 5.4: Communication Volume Sweep (was Fig 5.3) (line chart)
# Shows wall time vs comm_size for all 3 topologies
# (was Fig 5.5; old Fig 5.3 3-panel bar chart deleted as redundant)
# ============================================================
def fig_5_4():
    fig, ax1 = plt.subplots(figsize=(4.5, 3.2))

    # X-axis: comm_size labels (not linear scale — use categorical)
    comm_sizes = ['AllReduce\n(~90 MiB)', '100 MB\nAll-to-All', '512 MB\nAll-to-All', '1 GB\nAll-to-All']
    x = np.arange(len(comm_sizes))

    topos = ['Fat-Tree', 'Std. Torus', 'Twisted Torus']
    data = {
        'Fat-Tree':       [WALL_ALLREDUCE['Fat-Tree'],
                           WALL_A2A_100MB['Fat-Tree'],
                           WALL_A2A_512MB['Fat-Tree'],
                           WALL_A2A_1GB['Fat-Tree']],
        'Std. Torus':     [WALL_ALLREDUCE['Std. Torus'],
                           WALL_A2A_100MB['Std. Torus'],
                           WALL_A2A_512MB['Std. Torus'],
                           WALL_A2A_1GB['Std. Torus']],
        'Twisted Torus':  [WALL_ALLREDUCE['Twisted Torus'],
                           WALL_A2A_100MB['Twisted Torus'],
                           WALL_A2A_512MB['Twisted Torus'],
                           WALL_A2A_1GB['Twisted Torus']],
    }

    # --- Wall time lines (left Y-axis) ---
    markers = ['s', '^', 'o']
    for i, topo in enumerate(topos):
        ms = [c * alpha_us / 1000 for c in data[topo]]
        ax1.plot(x, ms, color=TOPO_COLORS[i], marker=markers[i],
                label=TOPO_LABELS_SHORT[i], linewidth=1.2, markersize=5, zorder=3)

    # GPU compute baseline
    gpu_ms = GPU_CYCLES * alpha_us / 1000
    ax1.axhline(y=gpu_ms, color=C_GRAY, linestyle=':', linewidth=0.8, zorder=2)
    ax1.text(1.02, gpu_ms, f'GPU compute\n({gpu_ms:.0f} ms)',
            fontsize=6, color=C_GRAY, va='center',
            transform=ax1.get_yaxis_transform(), clip_on=False)

    # Tipping point annotation
    torus_100mb_ms = WALL_A2A_100MB['Std. Torus'] * alpha_us / 1000
    ax1.annotate('Torus exceeds\ncompute window',
                xy=(1, torus_100mb_ms), xytext=(-0.15, 1200),
                fontsize=6, color=C_RED,
                arrowprops=dict(arrowstyle='->', color=C_RED, lw=0.8))

    ax1.set_ylabel('Wall Time (ms)')
    ax1.set_ylim(0, 8000)
    ax1.set_xticks(x)
    ax1.set_xticklabels(comm_sizes, fontsize=7)
    ax1.set_xlim(-0.3, 3.5)
    ax1.set_title('Topology Divergence vs. Communication Volume (128 nodes)')

    # --- Improvement ratio (right Y-axis) ---
    ax2 = ax1.twinx()
    C_RATIO = '#7B68EE'  # Purple — distinct from topology colors

    ratio = []
    for j in range(len(comm_sizes)):
        to_val = data['Std. Torus'][j]
        tw_val = data['Twisted Torus'][j]
        if tw_val > 0 and to_val != tw_val:
            ratio.append(to_val / tw_val)
        else:
            ratio.append(1.0)

    ax2.plot(x, ratio, color=C_RATIO, marker='D', linewidth=1.0,
             markersize=4, linestyle='--', zorder=4, alpha=0.8,
             label='Torus/Twisted ratio')
    ax2.fill_between(x, 1.0, ratio, color=C_RATIO, alpha=0.12, zorder=1)
    ax2.axhline(y=1.0, color=C_GRAY, linestyle='--', linewidth=0.4, zorder=1, alpha=0.3)

    # Annotate non-1.0 ratio points (left of markers)
    offsets = {1: (-0.3, 0.015), 2: (-0.35, 0.015), 3: (0, 0.015)}
    for j in [1, 2, 3]:
        if ratio[j] > 1.001:
            dx, dy = offsets[j]
            ax2.annotate(f'{ratio[j]:.2f}\\texttimes' if plt.rcParams.get('text.usetex', False) else f'{ratio[j]:.2f}x',
                         xy=(x[j], ratio[j]),
                         xytext=(x[j] + dx, ratio[j] + dy),
                         fontsize=7, fontweight='bold', color=C_RATIO)

    ax2.set_ylabel('Std. Torus / Twisted Torus', fontsize=8, color=C_RATIO)
    ax2.set_ylim(1.0, 1.30)
    ax2.tick_params(axis='y', labelcolor=C_RATIO, labelsize=7)

    # Merge legends from both axes
    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, loc='upper left', fontsize=6.5, framealpha=0.9)

    plt.tight_layout()
    plt.savefig(f'{outdir}/fig_5_4_sweep.pdf')
    plt.savefig(f'{outdir}/fig_5_4_sweep.png')
    plt.close()
    print("  Fig 5.4 done")


# ============================================================
# Fig 5.5: Improvement Factor (was Fig 5.4) — This Study vs TPU v4
# ============================================================
def fig_5_5():
    fig, ax = plt.subplots(figsize=(3.5, 2.8))

    studies = ['This Study\n(Consumer HW,\nstatic, asym. BW)',
               'Google TPU v4\n(Custom ASIC,\nOCS, sym. BW)']
    factors = [1.18, 1.63]
    colors = [C_GREEN, C_BLUE]

    bars = ax.bar(studies, factors, color=colors, width=0.38,
                  edgecolor='black', linewidth=0.4, zorder=3)

    ax.set_ylabel('Twisted / Standard Torus\nThroughput Ratio')
    ax.set_ylim(0, 2.0)
    ax.axhline(y=1.0, color=C_GRAY, linestyle='--', linewidth=0.6, zorder=2)
    ax.text(0.5, 1.03, 'No improvement (1.0×)', color=C_GRAY, fontsize=6,
            ha='center', transform=ax.get_yaxis_transform())

    for bar, val in zip(bars, factors):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.03,
                f'{val:.2f}×', ha='center', va='bottom', fontweight='bold', fontsize=10)

    ax.annotate('', xy=(0.15, 1.55), xytext=(0.15, 1.25),
                arrowprops=dict(arrowstyle='->', color=C_GRAY, lw=1.2))
    ax.text(0.28, 1.38, 'Direction\nconsistent', fontsize=6, color=C_GRAY)

    ax.set_title('Twisted Torus Improvement Factor')
    plt.tight_layout()
    plt.savefig(f'{outdir}/fig_5_5_improvement_factor.pdf')
    plt.savefig(f'{outdir}/fig_5_5_improvement_factor.png')
    plt.close()
    print("  Fig 5.5 done")


# ============================================================
# Fig 6.1: Cost Breakdown (unchanged)
# ============================================================
# fig_6_1 — v79 NIC cost model (3×dual-port 100GbE NIC, both families identical)
# 變更內容：
#   1. 兩個拓樸都用 100GbE NIC（per-GPU-rank logical topology 需 physical NIC 承載 logical bandwidth）：
#        FT NIC    = 48 × ConnectX-6 dual-port 100GbE @ US$900 → NT$1.296M（每台3張=600G,匹配520G需求）
#        Torus NIC = 48 × ConnectX-6 dual-port 100GbE @ US$900 → NT$1.296M（每台3張=600G,同FT；NIC不進模擬,只需non-blocking）
#   2. Torus 總額 → 7.840M；FT 總額 → 13.596M；省下 → 5.76M；Torus ≈ 58% of FT；switch 佔 FT 42%
#   3. 省錢來源是 eliminating managed switch fabric；Torus NIC 反而比 FT 貴（誠實：不主打 cheap NIC）
# 若 Dx 鎖價不是 US$900，改 NIC_NTM_PER_CARD 一行；其餘需同步 Table 6.1 與正文。

def fig_6_1():
    fig, ax = plt.subplots(figsize=(4.5, 3.5))

    NIC_NTM_PER_CARD = 900 * 30 / 1e6      # 單張雙埠100GbE: US$900 × NT$30 / 1e6 = 0.027M
    FT_NIC_NTM    = 48 * NIC_NTM_PER_CARD  # 3×/server × 16 = 48 張 → 1.296M
    TORUS_NIC_NTM = 48 * NIC_NTM_PER_CARD  # 3×/server × 16 = 48 張 → 1.296M (同FT,non-blocking)

    categories = ['Fat-Tree\n(L16\\_S8)', 'Torus /\nTwisted Torus']
    gpu    = [2.944, 2.944]
    server = [3.600, 3.600]
    nic    = [FT_NIC_NTM, TORUS_NIC_NTM]
    switch = [5.756, 0.000]

    x = np.arange(len(categories))
    width = 0.45
    colors_cost = [C_ORANGE, C_BLUE, C_GREEN, C_RED]

    ax.bar(x, gpu, width, label='GPU (128×RX 9070 XT)', color=colors_cost[0],
           edgecolor='black', linewidth=0.4)
    ax.bar(x, server, width, bottom=gpu, label='Server platform (16×)',
           color=colors_cost[1], edgecolor='black', linewidth=0.4)
    ax.bar(x, nic, width, bottom=[g+s for g, s in zip(gpu, server)],
           label='NIC (3×dual-port 100GbE/server, both)', color=colors_cost[2],
           edgecolor='black', linewidth=0.4)
    ax.bar(x, switch, width, bottom=[g+s+n for g, s, n in zip(gpu, server, nic)],
           label='Switch (24×SN2700)', color=colors_cost[3],
           edgecolor='black', linewidth=0.4)

    totals = [sum(t) for t in zip(gpu, server, nic, switch)]
    for i, total in enumerate(totals):
        ax.text(x[i], total + 0.18, f'NT\\${total:.2f}M',
                ha='center', fontweight='bold', fontsize=9)

    ax.axhline(y=10.0, color=C_GRAY, linestyle='--', linewidth=0.8)
    ax.text(0.98, 10.25, 'NT\\$10M budget', ha='right', fontsize=7,
            color=C_GRAY, transform=ax.get_yaxis_transform())
    ax.text(0.30, 11.3, 'Switches = 42\\%\nof Fat-Tree cost',
            color=C_RED, fontweight='bold', fontsize=7, ha='left', va='center')

    ax.set_ylabel('Total Cost (NT\\$ millions)')
    ax.set_xticks(x)
    ax.set_xticklabels(categories, fontsize=9)
    ax.set_ylim(0, 16)
    ax.legend(loc='upper right', fontsize=5.4, framealpha=0.9)
    ax.set_title('Hardware Cost Breakdown (128 GPUs)')
    plt.tight_layout()
    plt.savefig(f'{outdir}/fig_6_1_cost.pdf')
    plt.savefig(f'{outdir}/fig_6_1_cost.png')
    plt.close()
    print(f"  Fig 6.1 done — FT={totals[0]:.3f}M Torus={totals[1]:.3f}M ratio={totals[1]/totals[0]*100:.1f}%")


if __name__ == '__main__':
    print(f"Output directory: {outdir}")
    print("Generating figures...")
    fig_4_4()
    fig_5_1()
    fig_5_2()   # NEW: Qwen 0.5B DDP bar chart
    fig_5_3()   # was fig_5_2: All-to-All 1GB stacked bar
    fig_5_4()   # was fig_5_3: Communication volume sweep
    fig_5_5()   # was fig_5_4: TPU v4 improvement factor
    fig_6_1()
    print(f"\n=== All 7 figures generated in {outdir}/ ===")
    print("Each figure has both .pdf (for LaTeX/Word) and .png (for preview)")
