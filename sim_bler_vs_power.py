"""
BLER vs Total Transmit Power — 7-curve comparison.
"""
import numpy as np
import matplotlib.pyplot as plt
import csv
from typing import Dict, Tuple

from sim_common import (
    ieee_setup, save_figure, DT_CONFIG, N_S, L_TAPS, OTFS_PARAMS,
    slant_range_m, access_fspl_dB, atmospheric_loss_dB, power_to_link_snrs,
    create_ephemeris_predictor, mc_trial_one_snr,
    db_to_linear, linear_to_db, ALPHA_POWER,
    F_FEEDER_HZ, G_TX_GW_DBI, G_RX_SAT_DBI, D_FEEDER_M,
    F_ACCESS_HZ, G_TX_SAT_DBI, G_RX_UE_DBI,
)
from polar_code import PolarCode


def simulate(N=256, K=128, n_subchannels=N_S,
             max_erasures=DT_CONFIG.max_erasures,
             power_range_dBW=(-6, 15), ref_elevation_deg=50.0,
             n_points=16, n_mc=5000):
    rng = np.random.default_rng(2026)
    power_dBW_arr = np.linspace(power_range_dBW[0], power_range_dBW[1], n_points)

    ref_slant = slant_range_m(ref_elevation_deg)
    ref_fspl = access_fspl_dB(ref_slant)
    ref_atm_loss = atmospheric_loss_dB(ref_elevation_deg)

    K_rice = 2.0
    K_rice_dB = 10.0 * np.log10(K_rice)
    erasure_threshold = db_to_linear(-3.0)
    shadow_loss_dB = 20.0
    p_block_sub = 0.15

    polar_code = PolarCode(N, K, design_snr_dB=1.0, list_size=8)

    print(f"BLER vs P_total (7 curves, OTFS DD-domain MC)")
    print(f"  Polar({N},{K}), Rate={K/N:.2f}, MC={n_mc}")

    results = {key: np.zeros(n_points) for key in [
        'ppv_no_diversity', 'ppv_fixed_diversity', 'ppv_adaptive_diversity',
        'no_interleaver', 'interleaver', 'fixed', 'adaptive']}
    results['P_total_dBW'] = power_dBW_arr

    R = K / N
    for idx, p_dBW in enumerate(power_dBW_arr):
        gamma_gs, gamma_su = power_to_link_snrs(p_dBW, ref_fspl, ref_atm_loss)

        predictor = create_ephemeris_predictor(ref_elevation_deg, p_dBW, K_rice_dB)
        bler_eph = predictor.predict_subchannel_bler(ref_elevation_deg, R)
        eta_eph_scalar = max(1.0 - bler_eph, 0.0)
        eta_ephemeris = np.full(n_subchannels, eta_eph_scalar)

        ppv_no_acc, ppv_fix_acc, ppv_ada_acc = 0.0, 0.0, 0.0
        err_counts = [0, 0, 0, 0]

        for _ in range(n_mc):
            res = mc_trial_one_snr(
                gamma_su, gamma_gs, N, K, n_subchannels, max_erasures,
                K_rice, erasure_threshold, p_block_sub, shadow_loss_dB,
                polar_code, rng, use_otfs=True, eta_ephemeris=eta_ephemeris)
            ppv_no_acc += res['ppv_no_div']
            ppv_fix_acc += res['ppv_fix_div']
            ppv_ada_acc += res['ppv_ada_div']
            for c in range(4):
                err_counts[c] += int(res[f'err_{c+1}'])

        results['ppv_no_diversity'][idx] = ppv_no_acc / n_mc
        results['ppv_fixed_diversity'][idx] = ppv_fix_acc / n_mc
        results['ppv_adaptive_diversity'][idx] = ppv_ada_acc / n_mc
        results['no_interleaver'][idx] = err_counts[0] / n_mc
        results['interleaver'][idx] = err_counts[1] / n_mc
        results['fixed'][idx] = err_counts[2] / n_mc
        results['adaptive'][idx] = err_counts[3] / n_mc

        print(f"  P={p_dBW:5.1f}dBW: NoIntlv={results['no_interleaver'][idx]:.3e}, "
              f"Intlv={results['interleaver'][idx]:.3e}, "
              f"Fix={results['fixed'][idx]:.3e}, "
              f"Ada={results['adaptive'][idx]:.3e}")

    return results


def plot(results):
    ieee_setup()
    fig, ax = plt.subplots(figsize=(3.5, 2.8))
    p = results['P_total_dBW']

    ax.semilogy(p, np.maximum(results['ppv_no_diversity'], 1e-8),
                'h--', color='#9467bd', lw=0.8, ms=3,
                label='PPV bound (no diversity)')
    ax.semilogy(p, np.maximum(results['ppv_fixed_diversity'], 1e-8),
                'p--', color='#17becf', lw=0.8, ms=3,
                label='PPV bound (fixed diversity)')
    ax.semilogy(p, np.maximum(results['ppv_adaptive_diversity'], 1e-8),
                's--', color='#2ca02c', lw=0.8, ms=3,
                label='PPV bound (adaptive diversity)')
    ax.semilogy(p, np.maximum(results['no_interleaver'], 1e-8),
                'v-', color='#7f7f7f', lw=1.0, ms=4,
                label='Standard Polar (no interleaver)')
    ax.semilogy(p, np.maximum(results['interleaver'], 1e-8),
                '^-', color='#d62728', lw=1.0, ms=4,
                label='Standard Polar + interleaver')
    ax.semilogy(p, np.maximum(results['fixed'], 1e-8),
                'D-', color='#ff7f0e', lw=1.2, ms=4,
                label='Fixed diversity transform')
    ax.semilogy(p, np.maximum(results['adaptive'], 1e-8),
                'o-', color='#1f77b4', lw=1.4, ms=4,
                label='Adaptive diversity transform')

    fixed_clip = np.maximum(results['fixed'], 1e-8)
    adapt_clip = np.maximum(results['adaptive'], 1e-8)
    ax.fill_between(p, adapt_clip, fixed_clip, alpha=0.12, color='#1f77b4')

    ax.set_xlabel(r'Total transmit power $P_{\mathrm{total}}$ (dBW)')
    ax.set_ylabel('Block error rate (BLER)')
    ax.legend(fontsize=6, loc='lower left', framealpha=0.9)
    ax.set_ylim(5e-4, 2)
    ax.grid(True, which='both', ls='--', alpha=0.3)

    save_figure(fig, 'bler_comparison')


def save_csv(results):
    keys = ['ppv_no_diversity', 'ppv_fixed_diversity', 'ppv_adaptive_diversity',
            'no_interleaver', 'interleaver', 'fixed', 'adaptive']
    with open('bler_vs_power.csv', 'w', newline='') as f:
        w = csv.writer(f)
        w.writerow(['P_total_dBW'] + keys)
        for i in range(len(results['P_total_dBW'])):
            row = [f"{results['P_total_dBW'][i]:.2f}"]
            for k in keys:
                row.append(f"{results[k][i]:.6e}")
            w.writerow(row)
    print("Saved: bler_vs_power.csv")


if __name__ == "__main__":
    results = simulate(n_mc=3000)
    plot(results)
    save_csv(results)
