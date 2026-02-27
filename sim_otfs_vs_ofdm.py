"""
OTFS vs OFDM — BLER comparison under high-Doppler LEO NTN channel.

OTFS resolves Doppler in the delay-Doppler domain via MMSE equalization,
maintaining performance across a wide range of Doppler spreads.
OFDM suffers inter-carrier interference (ICI) that scales with normalized
Doppler f_D / Delta_f, creating an SINR ceiling:
    sigma^2_ICI = (pi * f_D_norm)^2 / 3
    gamma_eff = gamma / (1 + gamma * sigma^2_ICI)

LEO context: orbital velocity ~7.56 km/s, S-band 2 GHz.
Raw max Doppler ~50 kHz; after pre-compensation, residual Doppler
ranges from ~100 Hz to ~3 kHz depending on accuracy.
"""
import numpy as np
import matplotlib.pyplot as plt
import csv
from typing import Dict, Tuple

from sim_common import (
    ieee_setup, save_figure, DT_CONFIG, N_S, L_TAPS, OTFS_PARAMS,
    slant_range_m, access_fspl_dB, atmospheric_loss_dB, power_to_link_snrs,
    create_ephemeris_predictor, mc_trial_one_snr,
    generate_multipath_per_subchannel, generate_dd_channel,
    diversity_transform_transmit_receive, diversity_transform_transmit_receive_otfs,
    standard_polar_otfs, mmse_equalize_dd,
    db_to_linear, linear_to_db, SNRCalculator,
    G_FIX, G_FIX_BLOCK_WEIGHTS, G_DIV_POOL, BLOCK_WEIGHTS_POOL, N_POOL,
    RHO, N_OUT, M_GF, N_CW, X_ALL, S_ALL_FIX, X_MASK_0, X_MASK_1,
    conditional_bler, bler_fixed_diversity,
    DDChannel, SUBCHANNEL_MAP, map_bpsk_to_dd, extract_from_dd,
    TAP_POWERS_LIN,
)
from polar_code import PolarCode


def _ofdm_ici_variance(f_D_norm):
    """ICI variance for OFDM with rectangular pulse shaping.
    sigma^2_ICI approx (pi * f_D_norm)^2 / 3
    """
    return (np.pi * f_D_norm) ** 2 / 3.0


def _mc_trial_otfs_ofdm(gamma_avg_SU, gamma_gs, N, K, n_subchannels,
                         max_erasures, K_rice, erasure_threshold,
                         p_block_sub, shadow_loss_dB,
                         polar_code, rng, f_D_norm,
                         eta_ephemeris=None):
    """
    MC trial comparing OTFS and OFDM at a given normalized Doppler.

    OTFS: full DD-domain pipeline (Doppler-resilient).
    OFDM: per-subchannel AWGN with ICI degradation.
    """
    # --- Shared channel realization ---
    # Generate per-subchannel fading + blockage (used by both)
    gamma_e2e_base, blocked_mask = generate_multipath_per_subchannel(
        gamma_avg_SU, gamma_gs, n_subchannels, K_rice,
        p_block_sub, shadow_loss_dB, rng)

    erased_mask = gamma_e2e_base < erasure_threshold
    n_erased = int(np.sum(erased_mask))
    k_c = DT_CONFIG.k_c

    R = K / N
    N_sub = N // n_subchannels

    # --- OFDM path: apply ICI degradation ---
    sigma2_ici = _ofdm_ici_variance(f_D_norm)
    gamma_e2e_ofdm = gamma_e2e_base / (1.0 + gamma_e2e_base * sigma2_ici)

    # Erasure mask for OFDM (ICI may push more subchannels below threshold)
    erased_mask_ofdm = gamma_e2e_ofdm < erasure_threshold
    n_erased_ofdm = int(np.sum(erased_mask_ofdm))

    # --- OTFS path: no ICI penalty (DD-domain resolves Doppler) ---
    gamma_e2e_otfs = gamma_e2e_base

    # Doppler bin for DD channel scales with f_D_norm
    N_D = OTFS_PARAMS.N_D
    max_doppler_bin = max(1, int(round(f_D_norm * N_D)))
    max_doppler_bin = min(max_doppler_bin, N_D // 2)
    doppler_bins = [0, max_doppler_bin]

    # Generate DD channel for OTFS
    h_dd, noise_var, gamma_e2e_otfs_dd, blocked_mask_otfs = generate_dd_channel(
        gamma_avg_SU, gamma_gs, n_subchannels, K_rice,
        p_block_sub, shadow_loss_dB, rng, doppler_bins=doppler_bins)
    erased_mask_otfs = gamma_e2e_otfs_dd < erasure_threshold
    n_erased_otfs = int(np.sum(erased_mask_otfs))

    # --- Instantaneous eta ---
    eta_inst_otfs = np.zeros(n_subchannels)
    eta_inst_ofdm = np.zeros(n_subchannels)
    for j in range(n_subchannels):
        if not erased_mask_otfs[j]:
            eta_inst_otfs[j] = 1.0 - conditional_bler(gamma_e2e_otfs_dd[j], N_sub, R)
        if not erased_mask_ofdm[j]:
            eta_inst_ofdm[j] = 1.0 - conditional_bler(gamma_e2e_ofdm[j], N_sub, R)

    eta_for_T = eta_ephemeris if eta_ephemeris is not None else eta_inst_otfs

    # --- Algorithm 2 T-selection ---
    best_Q = np.dot(G_FIX_BLOCK_WEIGHTS, eta_for_T)
    best_G_DIV = G_FIX
    for i in range(N_POOL):
        Q = np.dot(BLOCK_WEIGHTS_POOL[i], eta_for_T)
        if Q > best_Q:
            best_Q = Q
            best_G_DIV = G_DIV_POOL[i]

    # --- Shared info bits ---
    info_bits = rng.integers(0, 2, size=K).astype(np.int8)

    # ====================== OTFS CURVES ======================

    # OTFS: No interleaver
    llr_otfs_1 = standard_polar_otfs(
        info_bits, polar_code, h_dd, noise_var,
        gamma_e2e_otfs_dd, n_subchannels, perm=None, rng=rng)
    err_otfs_nointlv = polar_code.decode_check(info_bits, llr_otfs_1)

    # OTFS: Fixed diversity
    if n_erased_otfs > max_erasures:
        err_otfs_fix = True
    else:
        coded_bits = polar_code.encode(info_bits)
        llr_otfs_fix = diversity_transform_transmit_receive_otfs(
            coded_bits, G_FIX, h_dd, noise_var,
            gamma_e2e_otfs_dd, erased_mask_otfs, eta_inst_otfs, rng)
        if llr_otfs_fix is None:
            err_otfs_fix = True
        else:
            err_otfs_fix = not np.array_equal(polar_code.decode(llr_otfs_fix), info_bits)

    # OTFS: Adaptive diversity
    if n_erased_otfs > max_erasures:
        err_otfs_ada = True
    else:
        n_surviving = n_subchannels - n_erased_otfs
        power_alloc = np.ones(n_subchannels)
        if n_surviving > 0 and n_erased_otfs > 0:
            power_alloc[~erased_mask_otfs] = float(n_subchannels) / n_surviving
            power_alloc[erased_mask_otfs] = 0.0
        gamma_otfs_ada = gamma_e2e_otfs_dd * power_alloc

        coded_bits_ada = polar_code.encode(info_bits)
        llr_otfs_ada = diversity_transform_transmit_receive_otfs(
            coded_bits_ada, best_G_DIV, h_dd, noise_var,
            gamma_otfs_ada, erased_mask_otfs, eta_inst_otfs, rng)
        if llr_otfs_ada is None:
            err_otfs_ada = True
        else:
            err_otfs_ada = not np.array_equal(polar_code.decode(llr_otfs_ada), info_bits)

    # ====================== OFDM CURVES ======================

    # OFDM: No interleaver (per-subchannel AWGN with ICI-degraded SNR)
    # QPSK per-bit SNR: Eb/N0 = Es/(2*N0) = gamma_symbol / 2
    bits_per_sub = N // n_subchannels
    snr_per_bit = np.zeros(N)
    for s in range(n_subchannels):
        start = s * bits_per_sub
        end = min((s + 1) * bits_per_sub, N)
        snr_per_bit[start:end] = gamma_e2e_ofdm[s] / 2.0
    if bits_per_sub * n_subchannels < N:
        snr_per_bit[bits_per_sub * n_subchannels:] = gamma_e2e_ofdm[-1] / 2.0
    llr_ofdm_1 = polar_code.transmit_per_bit_snr(info_bits, snr_per_bit, rng)
    err_ofdm_nointlv = polar_code.decode_check(info_bits, llr_ofdm_1)

    # OFDM: Fixed diversity (legacy AWGN path with ICI-degraded SNR)
    if n_erased_ofdm > max_erasures:
        err_ofdm_fix = True
    else:
        coded_bits_ofdm = polar_code.encode(info_bits)
        llr_ofdm_fix = diversity_transform_transmit_receive(
            coded_bits_ofdm, G_FIX, gamma_e2e_ofdm, erased_mask_ofdm,
            eta_inst_ofdm, rng)
        if llr_ofdm_fix is None:
            err_ofdm_fix = True
        else:
            err_ofdm_fix = not np.array_equal(polar_code.decode(llr_ofdm_fix), info_bits)

    # OFDM: Adaptive diversity
    if n_erased_ofdm > max_erasures:
        err_ofdm_ada = True
    else:
        n_surviving_ofdm = n_subchannels - n_erased_ofdm
        power_alloc_ofdm = np.ones(n_subchannels)
        if n_surviving_ofdm > 0 and n_erased_ofdm > 0:
            power_alloc_ofdm[~erased_mask_ofdm] = float(n_subchannels) / n_surviving_ofdm
            power_alloc_ofdm[erased_mask_ofdm] = 0.0
        gamma_ofdm_ada = gamma_e2e_ofdm * power_alloc_ofdm

        coded_bits_ofdm_ada = polar_code.encode(info_bits)
        llr_ofdm_ada = diversity_transform_transmit_receive(
            coded_bits_ofdm_ada, best_G_DIV, gamma_ofdm_ada,
            erased_mask_ofdm, eta_inst_ofdm, rng)
        if llr_ofdm_ada is None:
            err_ofdm_ada = True
        else:
            err_ofdm_ada = not np.array_equal(polar_code.decode(llr_ofdm_ada), info_bits)

    return {
        'otfs_no_interleaver': err_otfs_nointlv,
        'otfs_fixed': err_otfs_fix,
        'otfs_adaptive': err_otfs_ada,
        'ofdm_no_interleaver': err_ofdm_nointlv,
        'ofdm_fixed': err_ofdm_fix,
        'ofdm_adaptive': err_ofdm_ada,
    }


def simulate(N=256, K=128, n_subchannels=N_S,
             max_erasures=DT_CONFIG.max_erasures,
             P_total_dBW=8.0, ref_elevation_deg=50.0,
             f_D_norm_range=(0.0, 0.35), n_points=10, n_mc=2000):
    """
    BLER vs normalized Doppler f_D * T_sym at fixed power.

    Shows OTFS is Doppler-invariant while OFDM degrades with ICI.
    """
    rng = np.random.default_rng(2030)
    f_D_arr = np.linspace(f_D_norm_range[0], f_D_norm_range[1], n_points)

    ref_slant = slant_range_m(ref_elevation_deg)
    ref_fspl = access_fspl_dB(ref_slant)
    ref_atm_loss = atmospheric_loss_dB(ref_elevation_deg)
    gamma_gs, gamma_su = power_to_link_snrs(P_total_dBW, ref_fspl, ref_atm_loss)

    K_rice = 2.0
    K_rice_dB = 10.0 * np.log10(K_rice)
    erasure_threshold = db_to_linear(-3.0)
    shadow_loss_dB = 20.0
    p_block_sub = 0.15
    R = K / N

    polar_code = PolarCode(N, K, design_snr_dB=1.0, list_size=8)

    predictor = create_ephemeris_predictor(ref_elevation_deg, P_total_dBW, K_rice_dB)
    bler_eph = predictor.predict_subchannel_bler(ref_elevation_deg, R)
    eta_eph_scalar = max(1.0 - bler_eph, 0.0)
    eta_ephemeris = np.full(n_subchannels, eta_eph_scalar)

    print(f"OTFS vs OFDM: BLER vs Normalized Doppler")
    print(f"  P_total={P_total_dBW} dBW, elev={ref_elevation_deg}deg, MC={n_mc}")

    curve_keys = ['otfs_no_interleaver', 'otfs_fixed', 'otfs_adaptive',
                  'ofdm_no_interleaver', 'ofdm_fixed', 'ofdm_adaptive']
    results = {'f_D_norm': f_D_arr}
    for key in curve_keys:
        results[key] = np.zeros(n_points)

    for idx, f_D_norm in enumerate(f_D_arr):
        err_counts = {k: 0 for k in curve_keys}

        for _ in range(n_mc):
            res = _mc_trial_otfs_ofdm(
                gamma_su, gamma_gs, N, K, n_subchannels, max_erasures,
                K_rice, erasure_threshold, p_block_sub, shadow_loss_dB,
                polar_code, rng, f_D_norm, eta_ephemeris=eta_ephemeris)
            for k in curve_keys:
                err_counts[k] += int(res[k])

        for k in curve_keys:
            results[k][idx] = err_counts[k] / n_mc

        # Also compute residual Doppler in Hz for reference
        delta_f = OTFS_PARAMS.delta_f_Hz
        f_D_Hz = f_D_norm * delta_f
        print(f"  f_D_norm={f_D_norm:.3f} ({f_D_Hz:.0f} Hz): "
              f"OTFS(Ada={results['otfs_adaptive'][idx]:.3e}) "
              f"OFDM(Ada={results['ofdm_adaptive'][idx]:.3e})")

    return results


def plot(results):
    ieee_setup()
    fig, ax = plt.subplots(figsize=(3.5, 2.8))
    f_D = results['f_D_norm']

    # OTFS curves (solid)
    ax.semilogy(f_D, np.maximum(results['otfs_no_interleaver'], 1e-8),
                'v-', color='#7f7f7f', lw=1.0, ms=4,
                label='OTFS: Standard Polar')
    ax.semilogy(f_D, np.maximum(results['otfs_fixed'], 1e-8),
                'D-', color='#ff7f0e', lw=1.2, ms=4,
                label='OTFS: Fixed diversity transform')
    ax.semilogy(f_D, np.maximum(results['otfs_adaptive'], 1e-8),
                'o-', color='#1f77b4', lw=1.4, ms=4,
                label='OTFS: Adaptive diversity transform')

    # OFDM curves (dashed)
    ax.semilogy(f_D, np.maximum(results['ofdm_no_interleaver'], 1e-8),
                'v--', color='#7f7f7f', lw=1.0, ms=4, alpha=0.7,
                label='OFDM: Standard Polar')
    ax.semilogy(f_D, np.maximum(results['ofdm_fixed'], 1e-8),
                'D--', color='#ff7f0e', lw=1.2, ms=4, alpha=0.7,
                label='OFDM: Fixed diversity transform')
    ax.semilogy(f_D, np.maximum(results['ofdm_adaptive'], 1e-8),
                'o--', color='#1f77b4', lw=1.4, ms=4, alpha=0.7,
                label='OFDM: Adaptive diversity transform')

    # Annotate typical LEO residual Doppler ranges
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


def save_csv(results):
    keys = ['otfs_no_interleaver', 'otfs_fixed', 'otfs_adaptive',
            'ofdm_no_interleaver', 'ofdm_fixed', 'ofdm_adaptive']
    with open('otfs_vs_ofdm.csv', 'w', newline='') as f:
        w = csv.writer(f)
        w.writerow(['f_D_norm'] + keys)
        for i in range(len(results['f_D_norm'])):
            row = [f"{results['f_D_norm'][i]:.4f}"]
            for k in keys:
                row.append(f"{results[k][i]:.6e}")
            w.writerow(row)
    print("Saved: otfs_vs_ofdm.csv")


if __name__ == "__main__":
    results = simulate(n_mc=2000)
    plot(results)
    save_csv(results)
