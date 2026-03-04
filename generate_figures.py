#!/usr/bin/env python3
"""
Generate all 8 IEEE paper figures from existing CSV data.
No simulation code is executed — only CSV reading + matplotlib plotting.
"""

import csv
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


# ============================================================================
#  IEEE formatting (mirrors sim_common.ieee_setup / save_figure)
# ============================================================================

def ieee_setup():
    plt.rcParams.update({
        'font.family': 'serif',
        'font.serif': ['Times New Roman', 'Times', 'DejaVu Serif'],
        'mathtext.fontset': 'stix',
        'font.size': 8,
        'axes.labelsize': 9,
        'xtick.labelsize': 8,
        'ytick.labelsize': 8,
        'legend.fontsize': 7,
        'legend.framealpha': 0.9,
        'legend.edgecolor': '0.8',
        'figure.figsize': (3.5, 2.8),
        'figure.dpi': 150,
        'axes.grid': True,
        'grid.alpha': 0.3,
        'grid.linestyle': '--',
        'lines.linewidth': 1.0,
        'lines.markersize': 4,
        'savefig.dpi': 300,
        'savefig.bbox': 'tight',
        'savefig.pad_inches': 0.02,
    })


def save_figure(fig, basename):
    fig.savefig(f'{basename}.png', dpi=300, bbox_inches='tight', pad_inches=0.02)
    fig.savefig(f'{basename}.eps', format='eps', bbox_inches='tight', pad_inches=0.02)
    plt.close(fig)
    print(f"  Saved: {basename}.png, {basename}.eps")


def read_csv(filename):
    """Read CSV into dict of numpy arrays keyed by column header."""
    with open(filename, 'r') as f:
        reader = csv.DictReader(f)
        columns = {k: [] for k in reader.fieldnames}
        for row in reader:
            for k in reader.fieldnames:
                columns[k].append(float(row[k]))
    return {k: np.array(v) for k, v in columns.items()}


# ============================================================================
#  Figure 1: BLER vs Power
# ============================================================================

def fig_bler_vs_power():
    d = read_csv('bler_vs_power.csv')
    fig, ax = plt.subplots(figsize=(3.5, 2.8))
    p = d['P_total_dBW']

    ax.semilogy(p, np.maximum(d['PPV_no_diversity'], 1e-8),
                'h--', color='#9467bd', lw=0.8, ms=3,
                label='PPV bound (no diversity)')
    ax.semilogy(p, np.maximum(d['PPV_fixed_diversity'], 1e-8),
                'p--', color='#17becf', lw=0.8, ms=3,
                label='PPV bound (fixed diversity)')
    ax.semilogy(p, np.maximum(d['PPV_adaptive_diversity'], 1e-8),
                's--', color='#2ca02c', lw=0.8, ms=3,
                label='PPV bound (adaptive diversity)')
    ax.semilogy(p, np.maximum(d['no_interleaver'], 1e-8),
                'v-', color='#7f7f7f', lw=1.0, ms=4,
                label='Standard Polar (no interleaver)')
    ax.semilogy(p, np.maximum(d['interleaver'], 1e-8),
                '^-', color='#d62728', lw=1.0, ms=4,
                label='Standard Polar + interleaver')
    ax.semilogy(p, np.maximum(d['fixed'], 1e-8),
                'D-', color='#ff7f0e', lw=1.2, ms=4,
                label='Fixed diversity transform')
    ax.semilogy(p, np.maximum(d['adaptive'], 1e-8),
                'o-', color='#1f77b4', lw=1.4, ms=4,
                label='Adaptive diversity transform')

    ax.fill_between(p, np.maximum(d['adaptive'], 1e-8),
                    np.maximum(d['fixed'], 1e-8),
                    alpha=0.12, color='#1f77b4')

    ax.set_xlabel(r'Total transmit power $P_{\mathrm{total}}$ (dBW)')
    ax.set_ylabel('Block error rate (BLER)')
    ax.legend(fontsize=6, loc='lower left', framealpha=0.9)
    ax.set_ylim(5e-4, 2)
    ax.grid(True, which='both', ls='--', alpha=0.3)
    save_figure(fig, 'bler_comparison')


# ============================================================================
#  Figure 2: BLER vs Elevation
# ============================================================================

def fig_bler_vs_elevation():
    d = read_csv('bler_vs_elevation.csv')
    fig, ax = plt.subplots(figsize=(3.5, 2.8))
    elev = d['elevation_deg']

    ax.semilogy(elev, np.maximum(d['PPV_no_diversity'], 1e-7),
                'h--', color='#9467bd', lw=0.8, ms=3,
                label='PPV bound (no diversity)')
    ax.semilogy(elev, np.maximum(d['PPV_fixed_diversity'], 1e-7),
                'p--', color='#17becf', lw=0.8, ms=3,
                label='PPV bound (fixed diversity)')
    ax.semilogy(elev, np.maximum(d['PPV_adaptive_diversity'], 1e-7),
                's--', color='#2ca02c', lw=0.8, ms=3,
                label='PPV bound (adaptive diversity)')
    ax.semilogy(elev, np.maximum(d['no_interleaver'], 1e-7),
                'v-', color='#7f7f7f', lw=1.0, ms=4,
                label='Standard Polar (no interleaver)')
    ax.semilogy(elev, np.maximum(d['interleaver'], 1e-7),
                '^-', color='#d62728', lw=1.0, ms=4,
                label='Standard Polar + interleaver')
    ax.semilogy(elev, np.maximum(d['fixed'], 1e-7),
                'D-', color='#ff7f0e', lw=1.2, ms=4,
                label='Fixed diversity transform')
    ax.semilogy(elev, np.maximum(d['adaptive'], 1e-7),
                'o-', color='#1f77b4', lw=1.4, ms=4,
                label='Adaptive diversity transform')

    ax.fill_between(elev, np.maximum(d['adaptive'], 1e-7),
                    np.maximum(d['fixed'], 1e-7),
                    alpha=0.12, color='#1f77b4')

    ax.set_xlabel(r'Elevation angle (degrees)')
    ax.set_ylabel('Block error rate (BLER)')
    ax.legend(fontsize=6, loc='upper right', framealpha=0.9)
    ax.set_ylim(5e-4, 2)
    ax.grid(True, which='both', ls='--', alpha=0.3)
    save_figure(fig, 'bler_vs_elevation_comparison')


# ============================================================================
#  Figure 3: BLER vs Blocklength
# ============================================================================

def fig_bler_vs_blocklength():
    d = read_csv('bler_vs_blocklength.csv')
    fig, ax = plt.subplots(figsize=(3.5, 2.8))
    bl = d['blocklength']

    ax.semilogy(bl, np.maximum(d['ppv_no_diversity'], 1e-8),
                'h--', color='#9467bd', lw=0.8, ms=3,
                label='PPV bound (no diversity)')
    ax.semilogy(bl, np.maximum(d['ppv_fixed_diversity'], 1e-8),
                'p--', color='#17becf', lw=0.8, ms=3,
                label='PPV bound (fixed diversity)')
    ax.semilogy(bl, np.maximum(d['ppv_adaptive_diversity'], 1e-8),
                's--', color='#2ca02c', lw=0.8, ms=3,
                label='PPV bound (adaptive diversity)')
    ax.semilogy(bl, np.maximum(d['no_interleaver'], 1e-8),
                'v-', color='#7f7f7f', lw=1.0, ms=4,
                label='Standard Polar (no interleaver)')
    ax.semilogy(bl, np.maximum(d['interleaver'], 1e-8),
                '^-', color='#d62728', lw=1.0, ms=4,
                label='Standard Polar + interleaver')
    ax.semilogy(bl, np.maximum(d['fixed'], 1e-8),
                'D-', color='#ff7f0e', lw=1.2, ms=4,
                label='Fixed diversity transform')
    ax.semilogy(bl, np.maximum(d['adaptive'], 1e-8),
                'o-', color='#1f77b4', lw=1.4, ms=4,
                label='Adaptive diversity transform')

    ax.set_xlabel(r'Blocklength $n_c$')
    ax.set_ylabel('Block error rate (BLER)')
    ax.legend(fontsize=6, loc='upper right', framealpha=0.9)
    ax.grid(True, which='both', ls='--', alpha=0.3)
    ax.set_xscale('log', base=2)
    ax.set_xticks(bl)
    ax.set_xticklabels([str(int(x)) for x in bl])
    save_figure(fig, 'bler_vs_blocklength')


# ============================================================================
#  Figure 4: BLER vs Blockage (2-subplot)
# ============================================================================

def fig_bler_vs_blockage():
    d = read_csv('bler_vs_blockage.csv')
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(3.5, 4.0),
                                    height_ratios=[3, 1], sharex=True)
    p_blk = d['p_block']

    ax1.semilogy(p_blk, np.maximum(d['no_interleaver'], 1e-8),
                 'v-', color='#7f7f7f', lw=1.0, ms=4,
                 label='Standard Polar (no interleaver)')
    ax1.semilogy(p_blk, np.maximum(d['interleaver'], 1e-8),
                 '^-', color='#d62728', lw=1.0, ms=4,
                 label='Standard Polar + interleaver')
    ax1.semilogy(p_blk, np.maximum(d['fixed'], 1e-8),
                 'D-', color='#ff7f0e', lw=1.2, ms=4,
                 label='Fixed diversity transform')
    ax1.semilogy(p_blk, np.maximum(d['adaptive'], 1e-8),
                 'o-', color='#1f77b4', lw=1.4, ms=4,
                 label='Adaptive diversity transform')

    ax1.fill_between(p_blk, np.maximum(d['adaptive'], 1e-8),
                     np.maximum(d['fixed'], 1e-8),
                     alpha=0.12, color='#1f77b4')

    ax1.set_ylabel('Block error rate (BLER)')
    ax1.legend(fontsize=6, loc='lower right', framealpha=0.9)
    ax1.grid(True, which='both', ls='--', alpha=0.3)

    gain = np.clip(d['gain_ratio'], 0, 20)
    ax2.bar(p_blk, gain, width=(p_blk[1] - p_blk[0]) * 0.7,
            color='#1f77b4', alpha=0.7)
    ax2.axhline(y=1.0, color='gray', ls='--', lw=0.8)
    ax2.set_xlabel('Per-subchannel blockage probability')
    ax2.set_ylabel('Gain (Fixed/Adaptive)', fontsize=7)
    ax2.grid(True, ls='--', alpha=0.3)

    fig.tight_layout()
    save_figure(fig, 'bler_vs_blockage')


# ============================================================================
#  Figure 5: BLER vs Subchannels
# ============================================================================

def fig_bler_vs_n_subchannels():
    d = read_csv('bler_vs_n_subchannels.csv')
    fig, ax = plt.subplots(figsize=(3.5, 2.8))
    ns = d['n_subchannels']

    ax.semilogy(ns, np.maximum(d['ppv_fixed'], 1e-8),
                'p--', color='#17becf', lw=0.8, ms=4,
                label='PPV bound (fixed)')
    ax.semilogy(ns, np.maximum(d['ppv_adaptive'], 1e-8),
                's--', color='#2ca02c', lw=0.8, ms=4,
                label='PPV bound (adaptive)')
    ax.semilogy(ns, np.maximum(d['fixed'], 1e-8),
                'D-', color='#ff7f0e', lw=1.2, ms=5,
                label='Fixed diversity transform')
    ax.semilogy(ns, np.maximum(d['adaptive'], 1e-8),
                'o-', color='#1f77b4', lw=1.4, ms=5,
                label='Adaptive diversity transform')

    ax.set_xlabel(r'Number of subchannels $n_s$')
    ax.set_ylabel('Block error rate (BLER)')
    ax.legend(fontsize=7, loc='upper right', framealpha=0.9)
    ax.grid(True, which='both', ls='--', alpha=0.3)
    ax.set_xticks(ns)
    save_figure(fig, 'bler_vs_n_subchannels')


# ============================================================================
#  Figure 6: Throughput-Reliability
# ============================================================================

def fig_throughput_reliability():
    d = read_csv('throughput_reliability.csv')
    fig, ax = plt.subplots(figsize=(3.5, 2.8))
    p = d['P_total_dBW']

    ax.plot(p, d['capacity'], 'k--', lw=0.8, alpha=0.6,
            label=r'AWGN capacity $C(\bar{\gamma})$')
    ax.plot(p, d['ppv_adaptive_diversity'],
            's--', color='#2ca02c', lw=0.7, ms=3,
            label='PPV bound (adaptive)')
    ax.plot(p, d['ppv_fixed_diversity'],
            'p--', color='#17becf', lw=0.7, ms=3,
            label='PPV bound (fixed)')
    ax.plot(p, d['no_interleaver'],
            'v-', color='#7f7f7f', lw=1.0, ms=3,
            label='Standard Polar (no interleaver)')
    ax.plot(p, d['interleaver'],
            '^-', color='#d62728', lw=1.0, ms=3,
            label='Standard Polar + interleaver')
    ax.plot(p, d['fixed'],
            'D-', color='#ff7f0e', lw=1.2, ms=4,
            label='Fixed diversity transform')
    ax.plot(p, d['adaptive'],
            'o-', color='#1f77b4', lw=1.4, ms=4,
            label='Adaptive diversity transform')

    ax.set_xlabel(r'Total transmit power $P_{\mathrm{total}}$ (dBW)')
    ax.set_ylabel('Effective throughput (bits/channel use)')
    ax.legend(fontsize=6, loc='lower right', framealpha=0.9)
    ax.set_ylim(0, None)
    ax.grid(True, ls='--', alpha=0.3)
    save_figure(fig, 'throughput_reliability')


# ============================================================================
#  Figure 7: OTFS vs OFDM
# ============================================================================

def fig_otfs_vs_ofdm():
    d = read_csv('otfs_vs_ofdm.csv')
    fig, ax = plt.subplots(figsize=(3.5, 2.8))
    f_D = d['f_D_norm']

    ax.semilogy(f_D, np.maximum(d['otfs_fixed'], 1e-8),
                'D-', color='#ff7f0e', lw=1.2, ms=4,
                label='OTFS: Fixed diversity transform')
    ax.semilogy(f_D, np.maximum(d['otfs_adaptive'], 1e-8),
                'o-', color='#1f77b4', lw=1.4, ms=4,
                label='OTFS: Adaptive diversity transform')
    ax.semilogy(f_D, np.maximum(d['ofdm_fixed'], 1e-8),
                'D--', color='#ff7f0e', lw=1.2, ms=4, alpha=0.7,
                label='OFDM: Fixed diversity transform')
    ax.semilogy(f_D, np.maximum(d['ofdm_adaptive'], 1e-8),
                'o--', color='#1f77b4', lw=1.4, ms=4, alpha=0.7,
                label='OFDM: Adaptive diversity transform')

    # Doppler zone annotations
    ax.axvspan(0.01, 0.07, alpha=0.06, color='green')
    ax.axvspan(0.07, 0.20, alpha=0.06, color='orange')
    ax.axvspan(0.20, 0.35, alpha=0.06, color='red')
    ax.text(0.04, 1.3, 'Low', fontsize=5, ha='center', color='green')
    ax.text(0.13, 1.3, 'Medium', fontsize=5, ha='center', color='orange')
    ax.text(0.27, 1.3, 'High', fontsize=5, ha='center', color='red')

    ax.set_xlabel(r'Normalized Doppler $f_D / \Delta f$')
    ax.set_ylabel('Block error rate (BLER)')
    ax.legend(fontsize=5.5, loc='center left', framealpha=0.9, ncol=1,
              bbox_to_anchor=(0.0, 0.45))
    ax.set_ylim(5e-4, 2)
    ax.set_xlim(f_D[0], f_D[-1])
    ax.grid(True, which='both', ls='--', alpha=0.3)
    save_figure(fig, 'otfs_vs_ofdm')


# ============================================================================
#  Figure 8: Rank Recovery
# ============================================================================

def fig_rank_recovery():
    d = read_csv('rank_recovery.csv')
    fig, ax = plt.subplots(figsize=(3.5, 2.8))
    ne = d['n_erased']
    width = 0.35

    ax.bar(ne - width / 2, d['P_recovery_fixed'], width,
           color='#ff7f0e', alpha=0.85,
           label=r'$\mathbf{G}_{\mathrm{FIX}}$')
    ax.bar(ne + width / 2, d['P_recovery_adaptive'], width,
           color='#1f77b4', alpha=0.85,
           label=r'$\mathbf{G}_{\mathrm{DIV}} = \mathbf{T} \cdot \mathbf{G}_{\mathrm{FIX}}$')

    # d_min = n_s - k_c + 1 = 6 - 4 + 1 = 3
    d_min = 3
    ax.axvline(x=d_min - 1 + 0.5, color='red', ls='--', lw=1.0,
               alpha=0.7, label=f'$d_{{\\min}}-1 = {d_min - 1}$')

    ax.set_xlabel('Number of erased subchannels')
    ax.set_ylabel('Rank recovery probability')
    ax.legend(fontsize=7)
    ax.set_xticks(ne)
    ax.set_ylim(0, 1.15)
    ax.grid(True, axis='y', ls='--', alpha=0.3)
    save_figure(fig, 'rank_recovery')


# ============================================================================
#  Main
# ============================================================================

if __name__ == '__main__':
    ieee_setup()

    print("Generating IEEE paper figures from CSV data...")
    print()

    print("[1/8] BLER vs Power")
    fig_bler_vs_power()

    print("[2/8] BLER vs Elevation")
    fig_bler_vs_elevation()

    print("[3/8] BLER vs Blocklength")
    fig_bler_vs_blocklength()

    print("[4/8] BLER vs Blockage")
    fig_bler_vs_blockage()

    print("[5/8] BLER vs Subchannels")
    fig_bler_vs_n_subchannels()

    print("[6/8] Throughput-Reliability")
    fig_throughput_reliability()

    print("[7/8] OTFS vs OFDM")
    fig_otfs_vs_ofdm()

    print("[8/8] Rank Recovery")
    fig_rank_recovery()

    print()
    print("All 8 figures generated (8 PNG + 8 EPS).")
