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
α = 0.002254 μs/cycle for time conversion.
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

alpha_us = 0.002254  # μs/cycle

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

# All-to-All 1GB
WALL_A2A_1GB = {
    'Fat-Tree':       1_886_164_000,
    'Std. Torus':     3_049_068_000,
    'Twisted Torus':  2_575_180_000,
}

GPU_CYCLES = 274_982_000


# ============================================================
# Fig 4.4: Scope Boundary — Step Time Composition
# Horizontal stacked bar: comm 1.6% vs 40% is the key contrast
# ============================================================
def fig_4_4():
    fig, ax = plt.subplots(figsize=(5.0, 2.2))

    # Percentages of step time
    # ResNet-50: total=619.85, gpu=94.46(15.2%), comm=10.02(1.6%), overhead=515.37(83.2%)
    # CIFAR-10:  total=129.31, gpu=16.65(12.9%), comm=51.80(40.1%), overhead=60.86(47.0%)
    workloads = ['CIFAR-10 CNN\n(Latency-bound)', 'ResNet-50\n(Bandwidth-bound)']
    gpu_pct   = [12.9, 15.2]
    comm_pct  = [40.1,  1.6]
    over_pct  = [47.0, 83.2]

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
    ax.text(101, y[1], '619.85 ms', va='center', fontsize=7, color=C_GRAY)
    ax.text(101, y[0], '129.31 ms', va='center', fontsize=7, color=C_GRAY)

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
    ax.set_ylim(0, 800)

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
# Fig 5.2: All-to-All 1GB Stacked Bar (unchanged)
# ============================================================
def fig_5_2():
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
    plt.savefig(f'{outdir}/fig_5_2_all2all.pdf')
    plt.savefig(f'{outdir}/fig_5_2_all2all.png')
    plt.close()
    print("  Fig 5.2 done")


# ============================================================
# Fig 5.3: Communication Volume Sweep (line chart)
# Shows wall time vs comm_size for all 3 topologies
# (was Fig 5.5; old Fig 5.3 3-panel bar chart deleted as redundant)
# ============================================================
def fig_5_3():
    fig, ax = plt.subplots(figsize=(4.5, 3.0))

    # X-axis: comm_size labels (not linear scale — use categorical)
    comm_sizes = ['AllReduce\n(~90 MiB)', '100 MB\nAll-to-All', '1 GB\nAll-to-All']
    x = np.arange(len(comm_sizes))

    topos = ['Fat-Tree', 'Std. Torus', 'Twisted Torus']
    data = {
        'Fat-Tree':       [WALL_ALLREDUCE['Fat-Tree'],
                           WALL_A2A_100MB['Fat-Tree'],
                           WALL_A2A_1GB['Fat-Tree']],
        'Std. Torus':     [WALL_ALLREDUCE['Std. Torus'],
                           WALL_A2A_100MB['Std. Torus'],
                           WALL_A2A_1GB['Std. Torus']],
        'Twisted Torus':  [WALL_ALLREDUCE['Twisted Torus'],
                           WALL_A2A_100MB['Twisted Torus'],
                           WALL_A2A_1GB['Twisted Torus']],
    }

    markers = ['s', '^', 'o']
    for i, topo in enumerate(topos):
        ms = [c * alpha_us / 1000 for c in data[topo]]
        ax.plot(x, ms, color=TOPO_COLORS[i], marker=markers[i],
                label=TOPO_LABELS_SHORT[i], linewidth=1.2, markersize=5, zorder=3)

    # GPU compute baseline
    gpu_ms = GPU_CYCLES * alpha_us / 1000
    ax.axhline(y=gpu_ms, color=C_GRAY, linestyle=':', linewidth=0.8, zorder=2)
    ax.text(1.02, gpu_ms, f'GPU compute\n({gpu_ms:.0f} ms)',
            fontsize=6, color=C_GRAY, va='center',
            transform=ax.get_yaxis_transform(), clip_on=False)

    # Tipping point annotation
    torus_100mb_ms = WALL_A2A_100MB['Std. Torus'] * alpha_us / 1000
    ax.annotate('Torus exceeds\ncompute window',
                xy=(1, torus_100mb_ms), xytext=(0.3, 1800),
                fontsize=6, color=C_RED,
                arrowprops=dict(arrowstyle='->', color=C_RED, lw=0.8))

    ax.set_ylabel('Wall Time (ms)')
    ax.set_xticks(x)
    ax.set_xticklabels(comm_sizes, fontsize=7)
    ax.set_xlim(-0.3, 2.7)
    ax.set_ylim(0, 7500)
    ax.legend(loc='upper left', fontsize=7, framealpha=0.9)
    ax.set_title('Topology Divergence vs. Communication Volume\n(128 nodes)')
    plt.tight_layout()
    plt.savefig(f'{outdir}/fig_5_3_sweep.pdf')
    plt.savefig(f'{outdir}/fig_5_3_sweep.png')
    plt.close()
    print("  Fig 5.3 done")


# ============================================================
# Fig 5.4: Improvement Factor — This Study vs TPU v4
# ============================================================
def fig_5_4():
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
    plt.savefig(f'{outdir}/fig_5_4_improvement_factor.pdf')
    plt.savefig(f'{outdir}/fig_5_4_improvement_factor.png')
    plt.close()
    print("  Fig 5.4 done")


# ============================================================
# Fig 6.1: Cost Breakdown (unchanged)
# ============================================================
def fig_6_1():
    fig, ax = plt.subplots(figsize=(4.5, 3.5))

    categories = ['Fat-Tree\n(L16\_S8)', 'Torus /\nTwisted Torus']
    gpu    = [2.944, 2.944]
    server = [3.600, 3.600]
    nic    = [0.384, 0.384]
    switch = [5.756, 0.000]

    x = np.arange(len(categories))
    width = 0.45
    colors_cost = [C_ORANGE, C_BLUE, C_GREEN, C_RED]

    ax.bar(x, gpu, width, label='GPU (128×RX 9070 XT)', color=colors_cost[0],
           edgecolor='black', linewidth=0.4)
    ax.bar(x, server, width, bottom=gpu, label='Server platform (16×)',
           color=colors_cost[1], edgecolor='black', linewidth=0.4)
    ax.bar(x, nic, width, bottom=[g+s for g, s in zip(gpu, server)],
           label='NIC (32×ConnectX-6 Lx)', color=colors_cost[2],
           edgecolor='black', linewidth=0.4)
    ax.bar(x, switch, width, bottom=[g+s+n for g, s, n in zip(gpu, server, nic)],
           label='Switch (24×SN2700)', color=colors_cost[3],
           edgecolor='black', linewidth=0.4)

    totals = [sum(t) for t in zip(gpu, server, nic, switch)]
    for i, total in enumerate(totals):
        ax.text(x[i], total + 0.15, f'NT\${total:.1f}M',
                ha='center', fontweight='bold', fontsize=9)

    ax.axhline(y=10.0, color=C_GRAY, linestyle='--', linewidth=0.8)
    ax.text(0.98, 10.25, 'NT\$10M budget', ha='right', fontsize=7,
            color=C_GRAY, transform=ax.get_yaxis_transform())
    ax.annotate('Switches = 45\%\nof total cost', xy=(0.22, 10.5),
                xytext=(0.55, 12.5), fontsize=7, color=C_RED, fontweight='bold',
                arrowprops=dict(arrowstyle='->', color=C_RED, lw=1))

    ax.set_ylabel('Total Cost (NT\$ millions)')
    ax.set_xticks(x)
    ax.set_xticklabels(categories, fontsize=9)
    ax.set_ylim(0, 14)
    ax.legend(loc='upper right', fontsize=6, framealpha=0.9)
    ax.set_title('Hardware Cost Breakdown (128 GPUs)')
    plt.tight_layout()
    plt.savefig(f'{outdir}/fig_6_1_cost.pdf')
    plt.savefig(f'{outdir}/fig_6_1_cost.png')
    plt.close()
    print("  Fig 6.1 done")


# ============================================================
# Run all
# ============================================================
if __name__ == '__main__':
    print(f"Output directory: {outdir}")
    print("Generating figures...")
    fig_4_4()
    fig_5_1()
    fig_5_2()
    fig_5_3()
    fig_5_4()
    fig_6_1()
    print(f"\n=== All 6 figures generated in {outdir}/ ===")
    print("Each figure has both .pdf (for LaTeX/Word) and .png (for preview)")

    print("Each figure has both .pdf (for LaTeX/Word) and .png (for preview)")
