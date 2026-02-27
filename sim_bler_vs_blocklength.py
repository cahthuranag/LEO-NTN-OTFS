"""
BLER vs Blocklength — finite-blocklength performance.
"""
import numpy as np
import matplotlib.pyplot as plt
import csv
from typing import Dict

from sim_common import (
    ieee_setup, save_figure, DT_CONFIG, N_S, L_TAPS,
    slant_range_m, access_fspl_dB, atmospheric_loss_dB, power_to_link_snrs,
    create_ephemeris_predictor, mc_trial_one_snr,
    db_to_linear, linear_to_db,
)
from polar_code import PolarCode


def simulate(blocklengths=None, n_subchannels=N_S,
             max_erasures=DT_CONFIG.max_erasures,
             P_total_dBW=8.0, ref_elevation_deg=50.0, n_mc=2000):
    if blocklengths is None:
        blocklengths = [64, 128, 256, 512, 1024]
    rng = np.random.default_rng(2029)

    ref_slant = slant_range_m(ref_elevation_deg)
    ref_fspl = access_fspl_dB(ref_slant)
    ref_atm_loss = atmospheric_loss_dB(ref_elevation_deg)
    gamma_gs, gamma_su = power_to_link_snrs(P_total_dBW, ref_fspl, ref_atm_loss)

    K_rice = 2.0
    K_rice_dB = 10.0 * np.log10(K_rice)
    erasure_threshold = db_to_linear(-3.0)
    shadow_loss_dB = 20.0
    p_block_sub = 0.15

    n_pts = len(blocklengths)
    results = {key: np.zeros(n_pts) for key in [
        'ppv_no_diversity', 'ppv_fixed_diversity', 'ppv_adaptive_diversity',
        'no_interleaver', 'interleaver', 'fixed', 'adaptive']}
    results['blocklength'] = np.array(blocklengths)

    print(f"BLER vs Blocklength (P_total={P_total_dBW} dBW, MC={n_mc})")

    for idx, N in enumerate(blocklengths):
        K = N // 2
        R = K / N
        polar_code = PolarCode(N, K, design_snr_dB=1.0, list_size=8)

        predictor = create_ephemeris_predictor(ref_elevation_deg, P_total_dBW, K_rice_dB)
        bler_eph = predictor.predict_subchannel_bler(ref_elevation_deg, R)
        eta_eph_scalar = max(1.0 - bler_eph, 0.0)
        eta_ephemeris = np.full(n_subchannels, eta_eph_scalar)

        ppv_acc = [0.0, 0.0, 0.0]
        err_counts = [0, 0, 0, 0]

        for _ in range(n_mc):
            res = mc_trial_one_snr(
                gamma_su, gamma_gs, N, K, n_subchannels, max_erasures,
                K_rice, erasure_threshold, p_block_sub, shadow_loss_dB,
                polar_code, rng, use_otfs=True, eta_ephemeris=eta_ephemeris)
            ppv_acc[0] += res['ppv_no_div']
            ppv_acc[1] += res['ppv_fix_div']
            ppv_acc[2] += res['ppv_ada_div']
            for c in range(4):
                err_counts[c] += int(res[f'err_{c+1}'])

        results['ppv_no_diversity'][idx] = ppv_acc[0] / n_mc
        results['ppv_fixed_diversity'][idx] = ppv_acc[1] / n_mc
        results['ppv_adaptive_diversity'][idx] = ppv_acc[2] / n_mc
        results['no_interleaver'][idx] = err_counts[0] / n_mc
        results['interleaver'][idx] = err_counts[1] / n_mc
        results['fixed'][idx] = err_counts[2] / n_mc
        results['adaptive'][idx] = err_counts[3] / n_mc

        print(f"  N={N:5d}, K={K:4d}: Fix={results['fixed'][idx]:.3e}, "
              f"Ada={results['adaptive'][idx]:.3e}")

    return results


def plot(results):
    ieee_setup()
    fig, ax = plt.subplots(figsize=(3.5, 2.8))
    bl = results['blocklength']

    ax.semilogy(bl, np.maximum(results['ppv_no_diversity'], 1e-8),
                'h--', color='#9467bd', lw=0.8, ms=3,
                label='PPV bound (no diversity)')
    ax.semilogy(bl, np.maximum(results['ppv_fixed_diversity'], 1e-8),
                'p--', color='#17becf', lw=0.8, ms=3,
                label='PPV bound (fixed diversity)')
    ax.semilogy(bl, np.maximum(results['ppv_adaptive_diversity'], 1e-8),
                's--', color='#2ca02c', lw=0.8, ms=3,
                label='PPV bound (adaptive diversity)')
    ax.semilogy(bl, np.maximum(results['no_interleaver'], 1e-8),
                'v-', color='#7f7f7f', lw=1.0, ms=4,
                label='Standard Polar (no interleaver)')
    ax.semilogy(bl, np.maximum(results['interleaver'], 1e-8),
                '^-', color='#d62728', lw=1.0, ms=4,
                label='Standard Polar + interleaver')
    ax.semilogy(bl, np.maximum(results['fixed'], 1e-8),
                'D-', color='#ff7f0e', lw=1.2, ms=4,
                label='Fixed diversity transform')
    ax.semilogy(bl, np.maximum(results['adaptive'], 1e-8),
                'o-', color='#1f77b4', lw=1.4, ms=4,
                label='Adaptive diversity transform')

    ax.set_xlabel(r'Blocklength $n_c$')
    ax.set_ylabel('Block error rate (BLER)')
    ax.legend(fontsize=6, loc='upper right', framealpha=0.9)
    ax.grid(True, which='both', ls='--', alpha=0.3)
    ax.set_xscale('log', base=2)
    ax.set_xticks(results['blocklength'])
    ax.set_xticklabels([str(int(x)) for x in results['blocklength']])

    save_figure(fig, 'bler_vs_blocklength')


def save_csv(results):
    keys = ['ppv_no_diversity', 'ppv_fixed_diversity', 'ppv_adaptive_diversity',
            'no_interleaver', 'interleaver', 'fixed', 'adaptive']
    with open('bler_vs_blocklength.csv', 'w', newline='') as f:
        w = csv.writer(f)
        w.writerow(['blocklength'] + keys)
        for i in range(len(results['blocklength'])):
            row = [str(int(results['blocklength'][i]))]
            for k in keys:
                row.append(f"{results[k][i]:.6e}")
            w.writerow(row)
    print("Saved: bler_vs_blocklength.csv")


if __name__ == "__main__":
    results = simulate(n_mc=2000)
    plot(results)
    save_csv(results)
