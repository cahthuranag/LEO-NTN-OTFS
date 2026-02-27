"""
Throughput-Reliability Trade-off — effective throughput = R * (1 - BLER).
"""
import numpy as np
import matplotlib.pyplot as plt
import csv
from typing import Dict

from sim_common import (
    ieee_setup, save_figure, N_S, L_TAPS,
    slant_range_m, access_fspl_dB, atmospheric_loss_dB, power_to_link_snrs,
    SNRCalculator, capacity_awgn,
)
from sim_bler_vs_power import simulate as simulate_bler_vs_power


def simulate(results_power=None, N=256, K=128):
    if results_power is None:
        results_power = simulate_bler_vs_power(N=N, K=K, n_mc=3000)

    R = K / N
    p_arr = results_power['P_total_dBW']
    n_pts = len(p_arr)

    curve_keys = ['no_interleaver', 'interleaver', 'fixed', 'adaptive']
    ppv_keys = ['ppv_no_diversity', 'ppv_fixed_diversity', 'ppv_adaptive_diversity']

    throughput = {'P_total_dBW': p_arr}
    for key in curve_keys + ppv_keys:
        throughput[key] = R * (1.0 - np.clip(results_power[key], 0, 1))

    ref_slant = slant_range_m(50.0)
    ref_fspl = access_fspl_dB(ref_slant)
    ref_atm_loss = atmospheric_loss_dB(50.0)

    capacity_ref = np.zeros(n_pts)
    for i, p_dBW in enumerate(p_arr):
        gamma_gs, gamma_su = power_to_link_snrs(p_dBW, ref_fspl, ref_atm_loss)
        gamma_e2e_avg = SNRCalculator.cascaded_snr(gamma_gs, gamma_su)
        capacity_ref[i] = capacity_awgn(gamma_e2e_avg)
    throughput['capacity'] = capacity_ref

    print(f"Throughput-Reliability (R={R:.2f}):")
    for i in range(0, n_pts, max(1, n_pts // 6)):
        print(f"  P={p_arr[i]:5.1f}dBW: Ada={throughput['adaptive'][i]:.4f}, "
              f"Fix={throughput['fixed'][i]:.4f}, C={capacity_ref[i]:.4f}")

    return throughput


def plot(throughput):
    ieee_setup()
    fig, ax = plt.subplots(figsize=(3.5, 2.8))
    p_arr = throughput['P_total_dBW']

    ax.plot(p_arr, throughput['capacity'], 'k--', lw=0.8, alpha=0.6,
            label=r'AWGN capacity $C(\bar{\gamma})$')
    ax.plot(p_arr, throughput['ppv_adaptive_diversity'],
            's--', color='#2ca02c', lw=0.7, ms=3,
            label='PPV bound (adaptive)')
    ax.plot(p_arr, throughput['ppv_fixed_diversity'],
            'p--', color='#17becf', lw=0.7, ms=3,
            label='PPV bound (fixed)')
    ax.plot(p_arr, throughput['no_interleaver'],
            'v-', color='#7f7f7f', lw=1.0, ms=3,
            label='Standard Polar (no interleaver)')
    ax.plot(p_arr, throughput['interleaver'],
            '^-', color='#d62728', lw=1.0, ms=3,
            label='Standard Polar + interleaver')
    ax.plot(p_arr, throughput['fixed'],
            'D-', color='#ff7f0e', lw=1.2, ms=4,
            label='Fixed diversity transform')
    ax.plot(p_arr, throughput['adaptive'],
            'o-', color='#1f77b4', lw=1.4, ms=4,
            label='Adaptive diversity transform')

    ax.set_xlabel(r'Total transmit power $P_{\mathrm{total}}$ (dBW)')
    ax.set_ylabel('Effective throughput (bits/channel use)')
    ax.legend(fontsize=6, loc='lower right', framealpha=0.9)
    ax.set_ylim(0, None)
    ax.grid(True, ls='--', alpha=0.3)

    save_figure(fig, 'throughput_reliability')


def save_csv(throughput):
    keys = ['capacity', 'ppv_no_diversity', 'ppv_fixed_diversity',
            'ppv_adaptive_diversity', 'no_interleaver', 'interleaver',
            'fixed', 'adaptive']
    with open('throughput_reliability.csv', 'w', newline='') as f:
        w = csv.writer(f)
        w.writerow(['P_total_dBW'] + keys)
        for i in range(len(throughput['P_total_dBW'])):
            row = [f"{throughput['P_total_dBW'][i]:.2f}"]
            for k in keys:
                row.append(f"{throughput[k][i]:.6e}")
            w.writerow(row)
    print("Saved: throughput_reliability.csv")


if __name__ == "__main__":
    throughput = simulate()
    plot(throughput)
    save_csv(throughput)
