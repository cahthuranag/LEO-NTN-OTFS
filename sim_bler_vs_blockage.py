"""
BLER vs Per-Subchannel Blockage Probability.
"""
import numpy as np
import matplotlib.pyplot as plt
import csv
from typing import Dict, Tuple

from sim_common import (
    ieee_setup, save_figure, DT_CONFIG, N_S, L_TAPS,
    slant_range_m, access_fspl_dB, atmospheric_loss_dB, power_to_link_snrs,
    create_ephemeris_predictor, mc_trial_one_snr,
    db_to_linear, linear_to_db,
)
from polar_code import PolarCode


def simulate(N=256, K=128, n_subchannels=N_S,
             max_erasures=DT_CONFIG.max_erasures,
             P_total_dBW=8.0, ref_elevation_deg=50.0,
             p_block_range=(0.0, 0.35), n_points=8, n_mc=2000):
    rng = np.random.default_rng(2028)
    p_block_arr = np.linspace(p_block_range[0], p_block_range[1], n_points)

    ref_slant = slant_range_m(ref_elevation_deg)
    ref_fspl = access_fspl_dB(ref_slant)
    ref_atm_loss = atmospheric_loss_dB(ref_elevation_deg)
    gamma_gs, gamma_su = power_to_link_snrs(P_total_dBW, ref_fspl, ref_atm_loss)

    K_rice = 2.0
    K_rice_dB = 10.0 * np.log10(K_rice)
    erasure_threshold = db_to_linear(-3.0)
    shadow_loss_dB = 20.0

    polar_code = PolarCode(N, K, design_snr_dB=1.0, list_size=8)
    R = K / N

    predictor = create_ephemeris_predictor(ref_elevation_deg, P_total_dBW, K_rice_dB)
    bler_eph = predictor.predict_subchannel_bler(ref_elevation_deg, R)
    eta_eph_scalar = max(1.0 - bler_eph, 0.0)
    eta_ephemeris = np.full(n_subchannels, eta_eph_scalar)

    print(f"BLER vs Blockage (P_total={P_total_dBW} dBW, MC={n_mc})")

    results = {key: np.zeros(n_points) for key in [
        'no_interleaver', 'interleaver', 'fixed', 'adaptive']}
    results['p_block'] = p_block_arr
    results['gain_ratio'] = np.zeros(n_points)

    for idx, p_blk in enumerate(p_block_arr):
        err_counts = [0, 0, 0, 0]
        for _ in range(n_mc):
            res = mc_trial_one_snr(
                gamma_su, gamma_gs, N, K, n_subchannels, max_erasures,
                K_rice, erasure_threshold, p_blk, shadow_loss_dB,
                polar_code, rng, use_otfs=True, eta_ephemeris=eta_ephemeris)
            for c in range(4):
                err_counts[c] += int(res[f'err_{c+1}'])

        results['no_interleaver'][idx] = err_counts[0] / n_mc
        results['interleaver'][idx] = err_counts[1] / n_mc
        results['fixed'][idx] = err_counts[2] / n_mc
        results['adaptive'][idx] = err_counts[3] / n_mc

        fix_val = results['fixed'][idx]
        ada_val = results['adaptive'][idx]
        if ada_val > 1e-10:
            results['gain_ratio'][idx] = fix_val / ada_val
        else:
            results['gain_ratio'][idx] = float('inf') if fix_val > 0 else 1.0

        print(f"  p_block={p_blk:.3f}: Fix={fix_val:.3e}, Ada={ada_val:.3e}, "
              f"Gain={results['gain_ratio'][idx]:.2f}x")

    return results


def plot(results):
    ieee_setup()
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(3.5, 4.0),
                                    height_ratios=[3, 1], sharex=True)
    p_blk = results['p_block']

    ax1.semilogy(p_blk, np.maximum(results['no_interleaver'], 1e-8),
                 'v-', color='#7f7f7f', lw=1.0, ms=4,
                 label='Standard Polar (no interleaver)')
    ax1.semilogy(p_blk, np.maximum(results['interleaver'], 1e-8),
                 '^-', color='#d62728', lw=1.0, ms=4,
                 label='Standard Polar + interleaver')
    ax1.semilogy(p_blk, np.maximum(results['fixed'], 1e-8),
                 'D-', color='#ff7f0e', lw=1.2, ms=4,
                 label='Fixed diversity transform')
    ax1.semilogy(p_blk, np.maximum(results['adaptive'], 1e-8),
                 'o-', color='#1f77b4', lw=1.4, ms=4,
                 label='Adaptive diversity transform')

    fixed_clip = np.maximum(results['fixed'], 1e-8)
    adapt_clip = np.maximum(results['adaptive'], 1e-8)
    ax1.fill_between(p_blk, adapt_clip, fixed_clip, alpha=0.12, color='#1f77b4')

    ax1.set_ylabel('Block error rate (BLER)')
    ax1.legend(fontsize=6, loc='lower right', framealpha=0.9)
    ax1.grid(True, which='both', ls='--', alpha=0.3)

    gain = np.clip(results['gain_ratio'], 0, 20)
    ax2.bar(p_blk, gain, width=(p_blk[1] - p_blk[0]) * 0.7,
            color='#1f77b4', alpha=0.7)
    ax2.axhline(y=1.0, color='gray', ls='--', lw=0.8)
    ax2.set_xlabel('Per-subchannel blockage probability')
    ax2.set_ylabel('Gain (Fixed/Adaptive)', fontsize=7)
    ax2.grid(True, ls='--', alpha=0.3)

    fig.tight_layout()
    save_figure(fig, 'bler_vs_blockage')


def save_csv(results):
    with open('bler_vs_blockage.csv', 'w', newline='') as f:
        w = csv.writer(f)
        w.writerow(['p_block', 'no_interleaver', 'interleaver',
                     'fixed', 'adaptive', 'gain_ratio'])
        for i in range(len(results['p_block'])):
            w.writerow([
                f"{results['p_block'][i]:.4f}",
                f"{results['no_interleaver'][i]:.6e}",
                f"{results['interleaver'][i]:.6e}",
                f"{results['fixed'][i]:.6e}",
                f"{results['adaptive'][i]:.6e}",
                f"{results['gain_ratio'][i]:.4f}",
            ])
    print("Saved: bler_vs_blockage.csv")


if __name__ == "__main__":
    results = simulate(n_mc=2000)
    plot(results)
    save_csv(results)
