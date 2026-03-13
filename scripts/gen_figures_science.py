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
# 嘗試用完整 LaTeX 渲染；如果沒裝 LaTeX 就 fallback
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

# Override some defaults for our specific needs
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
# Fig 4.3: Calibration Results
# ============================================================
def fig_4_3():
    fig, ax = plt.subplots(figsize=(3.5, 2.5))

    models = ['ResNet-50\n(Bandwidth-bound)', 'CIFAR-10 CNN\n(Latency-bound)']
    errors = [1.18, 91.0]
    colors = [C_GREEN, C_RED]

    bars = ax.bar(models, errors, color=colors, width=0.45,
                  edgecolor='black', linewidth=0.4, zorder=3)

    ax.set_ylabel('Relative Communication Error (\%)')
    ax.set_ylim(0, 108)
    ax.axhline(y=5, color=C_GRAY, linestyle='--', linewidth=0.6, alpha=0.7, zorder=2)
    ax.text(0.5, 7, '5\% threshold', color=C_GRAY, fontsize=7,
            ha='center', transform=ax.get_yaxis_transform())

    for bar, val in zip(bars, errors):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 2,
                f'{val:.1f}\%', ha='center', va='bottom', fontweight='bold', fontsize=9)

    ax.set_title('Calibration Accuracy')
    plt.tight_layout()
    plt.savefig(f'{outdir}/fig_4_3_calibration.pdf')
    plt.savefig(f'{outdir}/fig_4_3_calibration.png')
    plt.close()
    print("  Fig 4.3 done")


# ============================================================
# Fig 5.1: AllReduce Wall Time Breakdown
# ============================================================
def fig_5_1():
    fig, ax = plt.subplots(figsize=(3.5, 2.5))

    wall_cycles = [274982000, 274982000, 274982000]
    compute_ms = [c * alpha_us / 1000 for c in wall_cycles]

    x = np.arange(3)
    width = 0.45

    bars = ax.bar(x, compute_ms, width,
                  color=TOPO_COLORS, edgecolor='black', linewidth=0.4, zorder=3)

    ax.set_ylabel('Wall Time (ms)')
    ax.set_xticks(x)
    ax.set_xticklabels(TOPO_LABELS_SHORT, fontsize=8)
    ax.set_ylim(0, 800)

    for i, bar in enumerate(bars):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 15,
                f'{compute_ms[i]:.0f} ms', ha='center', va='bottom', fontsize=7)

    # Annotation instead of legend
    ax.text(0.5, 0.97, 'Exposed communication = 0\n(all topologies identical)',
            ha='center', va='top', fontsize=7, fontstyle='italic', color=C_GRAY,
            transform=ax.transAxes,
            bbox=dict(boxstyle='round,pad=0.3', facecolor='#f0f0f0', edgecolor='#cccccc', alpha=0.8))

    ax.set_title('AllReduce Experiment (128 nodes, ResNet-50)')
    plt.tight_layout()
    plt.savefig(f'{outdir}/fig_5_1_allreduce.pdf')
    plt.savefig(f'{outdir}/fig_5_1_allreduce.png')
    plt.close()
    print("  Fig 5.1 done")


# ============================================================
# Fig 5.2: All-to-All Wall Time Breakdown (Stacked)
# ============================================================
def fig_5_2():
    fig, ax = plt.subplots(figsize=(3.5, 3.0))

    wall_ms   = [4252, 6877, 5808]
    gpu_ms    = [620, 620, 620]
    exposed   = [w - 620 for w in wall_ms]

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

    # Wall time labels
    for i, w in enumerate(wall_ms):
        ax.text(x[i], w + 100, f'{w:,} ms', ha='center', va='bottom',
                fontweight='bold', fontsize=8)

    # Relative labels — positioned above wall time labels
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
# Fig 5.3: Effect of Communication Scaling (side-by-side)
# ============================================================
def fig_5_3():
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(6.5, 2.8))

    # Left: Original ~1KB
    wall_1kb_ms = [274982000 * alpha_us / 1000] * 3  # all identical
    bars1 = ax1.bar(range(3), wall_1kb_ms, color=TOPO_COLORS, width=0.5,
                    edgecolor='black', linewidth=0.4, zorder=3)
    ax1.set_title('All-to-All (Original $\\sim$1 KB)', fontsize=9)
    ax1.set_ylabel('Wall Time (ms)')
    ax1.set_xticks(range(3))
    ax1.set_xticklabels(TOPO_LABELS_SHORT, fontsize=7)
    ax1.set_ylim(0, 800)

    for bar in bars1:
        ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 10,
                 f'{wall_1kb_ms[0]:.0f}', ha='center', fontsize=7, fontweight='bold')

    ax1.text(1, 720, 'No difference\n(comm hidden)',
             ha='center', fontsize=7, fontstyle='italic', color=C_GRAY,
             bbox=dict(boxstyle='round,pad=0.2', facecolor='#f0f0f0',
                       edgecolor='#cccccc', alpha=0.8))

    # Right: Scaled 1GB
    wall_1gb = [4252, 6877, 5808]
    bars2 = ax2.bar(range(3), wall_1gb, color=TOPO_COLORS, width=0.5,
                    edgecolor='black', linewidth=0.4, zorder=3)
    ax2.set_title('All-to-All (Scaled to 1 GB)', fontsize=9)
    ax2.set_ylabel('Wall Time (ms)')
    ax2.set_xticks(range(3))
    ax2.set_xticklabels(TOPO_LABELS_SHORT, fontsize=7)
    ax2.set_ylim(0, 8200)

    for i, bar in enumerate(bars2):
        ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 100,
                 f'{wall_1gb[i]:,}', ha='center', fontsize=7, fontweight='bold')

    ax2.text(1, 7400, 'Topology differences\nemerge under saturation',
             ha='center', fontsize=7, fontstyle='italic', color=C_GRAY,
             bbox=dict(boxstyle='round,pad=0.2', facecolor='#f0f0f0',
                       edgecolor='#cccccc', alpha=0.8))

    plt.tight_layout()
    plt.savefig(f'{outdir}/fig_5_3_scaling_effect.pdf')
    plt.savefig(f'{outdir}/fig_5_3_scaling_effect.png')
    plt.close()
    print("  Fig 5.3 done")


# ============================================================
# Fig 5.4: Improvement Factor Comparison
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

    # Direction arrow
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
# Fig 6.1: Cost Breakdown (Stacked Bar)
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

    ax.bar(x, gpu, width, label='GPU (128×RX 9070 XT)', color=colors_cost[0], edgecolor='black', linewidth=0.4)
    ax.bar(x, server, width, bottom=gpu, label='Server platform (16×)', color=colors_cost[1], edgecolor='black', linewidth=0.4)
    ax.bar(x, nic, width, bottom=[g+s for g, s in zip(gpu, server)], label='NIC (32×ConnectX-6 Lx)', color=colors_cost[2], edgecolor='black', linewidth=0.4)
    ax.bar(x, switch, width, bottom=[g+s+n for g, s, n in zip(gpu, server, nic)], label='Switch (24×SN2700)', color=colors_cost[3], edgecolor='black', linewidth=0.4)

    totals = [sum(t) for t in zip(gpu, server, nic, switch)]
    for i, total in enumerate(totals):
        ax.text(x[i], total + 0.15, f'NT\${total:.1f}M', ha='center', fontweight='bold', fontsize=9)

    ax.axhline(y=10.0, color=C_GRAY, linestyle='--', linewidth=0.8)
    ax.text(0.98, 10.25, 'NT\$10M budget', ha='right', fontsize=7, color=C_GRAY,
            transform=ax.get_yaxis_transform())
    ax.annotate('Switches = 45\%\nof total cost', xy=(0.22, 10.5), xytext=(0.55, 12.5),
                fontsize=7, color=C_RED, fontweight='bold',
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
    fig_4_3()
    fig_5_1()
    fig_5_2()
    fig_5_3()
    fig_5_4()
    fig_6_1()
    print(f"\n=== All 6 figures generated in {outdir}/ ===")
    print("Each figure has both .pdf (for LaTeX/Word) and .png (for preview)")
