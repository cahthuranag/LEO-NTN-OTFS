"""
BLER vs Elevation Angle — 7-curve comparison.
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
)
from polar_code import PolarCode


def simulate(N=256, K=128, n_subchannels=N_S,
             max_erasures=DT_CONFIG.max_erasures,
             P_total_dBW=8.0, elevation_range=(20, 90),
             n_points=8, n_mc=5000):
    rng = np.random.default_rng(2027)
    elevations = np.linspace(elevation_range[0], elevation_range[1], n_points)

    K_rice_min, K_rice_max = 1.0, 5.0
    erasure_threshold = db_to_linear(-3.0)
    shadow_loss_dB = 20.0
    p_block_sub_low, p_block_sub_high = 0.25, 0.08

    polar_code = PolarCode(N, K, design_snr_dB=1.0, list_size=8)

    print(f"BLER vs Elevation (7 curves, OTFS DD-domain MC)")
    print(f"  P_total={P_total_dBW:.1f} dBW, MC={n_mc}")

    results = {key: np.zeros(n_points) for key in [
        'ppv_no_diversity', 'ppv_fixed_diversity', 'ppv_adaptive_diversity',
        'no_interleaver', 'interleaver', 'fixed', 'adaptive']}
    results['elevation_deg'] = elevations
    results['P_total_dBW'] = P_total_dBW

    R = K / N
    for idx, eps_deg in enumerate(elevations):
        d_access = slant_range_m(eps_deg)
        fspl_access = access_fspl_dB(d_access)
        atm_loss = atmospheric_loss_dB(eps_deg)
        gamma_gs, gamma_su = power_to_link_snrs(P_total_dBW, fspl_access, atm_loss)

        elev_frac = (eps_deg - elevation_range[0]) / (elevation_range[1] - elevation_range[0])
        K_rice = K_rice_min + (K_rice_max - K_rice_min) * elev_frac
        K_rice_dB = 10.0 * np.log10(K_rice)
        p_block_sub = p_block_sub_low + (p_block_sub_high - p_block_sub_low) * elev_frac

        predictor = create_ephemeris_predictor(eps_deg, P_total_dBW, K_rice_dB)
        bler_eph = predictor.predict_subchannel_bler(eps_deg, R)
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

        print(f"  Elev={eps_deg:5.1f}deg: NoIntlv={results['no_interleaver'][idx]:.3e}, "
              f"Intlv={results['interleaver'][idx]:.3e}, "
              f"Fix={results['fixed'][idx]:.3e}, "
              f"Ada={results['adaptive'][idx]:.3e}")

    return results


def plot(results):
    ieee_setup()
    fig, ax = plt.subplots(figsize=(3.5, 2.8))
    elev = results['elevation_deg']

    ax.semilogy(elev, np.maximum(results['ppv_no_diversity'], 1e-7),
                'h--', color='#9467bd', lw=0.8, ms=3,
                label='PPV bound (no diversity)')
    ax.semilogy(elev, np.maximum(results['ppv_fixed_diversity'], 1e-7),
                'p--', color='#17becf', lw=0.8, ms=3,
                label='PPV bound (fixed diversity)')
    ax.semilogy(elev, np.maximum(results['ppv_adaptive_diversity'], 1e-7),
                's--', color='#2ca02c', lw=0.8, ms=3,
                label='PPV bound (adaptive diversity)')
    ax.semilogy(elev, np.maximum(results['no_interleaver'], 1e-7),
                'v-', color='#7f7f7f', lw=1.0, ms=4,
                label='Standard Polar (no interleaver)')
    ax.semilogy(elev, np.maximum(results['interleaver'], 1e-7),
                '^-', color='#d62728', lw=1.0, ms=4,
                label='Standard Polar + interleaver')
    ax.semilogy(elev, np.maximum(results['fixed'], 1e-7),
                'D-', color='#ff7f0e', lw=1.2, ms=4,
                label='Fixed diversity transform')
    ax.semilogy(elev, np.maximum(results['adaptive'], 1e-7),
                'o-', color='#1f77b4', lw=1.4, ms=4,
                label='Adaptive diversity transform')

    fixed_clip = np.maximum(results['fixed'], 1e-7)
    adapt_clip = np.maximum(results['adaptive'], 1e-7)
    ax.fill_between(elev, adapt_clip, fixed_clip, alpha=0.12, color='#1f77b4')

    ax.set_xlabel(r'Elevation angle (degrees)')
    ax.set_ylabel('Block error rate (BLER)')
    ax.legend(fontsize=6, loc='upper right', framealpha=0.9)
    ax.set_ylim(5e-4, 2)
    ax.grid(True, which='both', ls='--', alpha=0.3)

    save_figure(fig, 'bler_vs_elevation_comparison')


def save_csv(results):
    keys = ['ppv_no_diversity', 'ppv_fixed_diversity', 'ppv_adaptive_diversity',
            'no_interleaver', 'interleaver', 'fixed', 'adaptive']
    with open('bler_vs_elevation.csv', 'w', newline='') as f:
        w = csv.writer(f)
        w.writerow(['elevation_deg'] + keys)
        for i in range(len(results['elevation_deg'])):
            row = [f"{results['elevation_deg'][i]:.1f}"]
            for k in keys:
                row.append(f"{results[k][i]:.6e}")
            w.writerow(row)
    print("Saved: bler_vs_elevation.csv")


if __name__ == "__main__":
    results = simulate(n_mc=3000)
    plot(results)
    save_csv(results)
