"""
BLER vs SNR Simulation — Polar + OTFS with Diversity Transforms
================================================================
7-curve comparison for LEO NTN OTFS systems:
  Semi-analytical FBL bounds (3):
    - PPV no diversity, PPV fixed diversity, PPV adaptive diversity
  Practical end-to-end MC (4):
    1. Standard Polar (no interleaver)        — worst, error floor
    2. Standard Polar + random interleaver    — better, lower error floor
    3. Fixed Diversity Transform (MDS G_FIX)  — diversity transform in signal path
    4. Adaptive Diversity Transform (Alg. 2)  — quality-metric-based T selection

Channel model: LEO NTN with per-subchannel multipath Rician fading,
MRC combining across L delay-Doppler resolved paths, cascaded
feeder+access link SNR (transparent relay), and blockage/shadowing.

Signal path for diversity curves (3 & 4):
  Polar encode -> reshape -> G_DIV^T per column -> per-subchannel
  AWGN-equivalent channel -> eta-scaled LLRs -> MAP soft diversity
  demapping -> LLR reconstruction -> Polar SC decode

Author: Research simulation
Date: 2026-02-21
"""

import numpy as np
import matplotlib.pyplot as plt
import csv
from typing import Dict, Tuple

from channel import (
    SNRCalculator, SPEED_OF_LIGHT, linear_to_db, db_to_linear
)
from fbl_analysis import (
    finite_blocklength_bler, conditional_bler,
    bler_no_diversity, bler_fixed_diversity, bler_adaptive_diversity,
)
from polar_code import PolarCode
from diversity_transform import (
    DiversityTransformConfig, gf2_rank, gf2_matmul,
    generate_candidate_T, quality_metric,
)

# ============================================================================
#  Module-level constants
# ============================================================================

# Diversity transform configuration
_DT_CONFIG = DiversityTransformConfig(k_c=4, n_s=6, m=3)
_G_FIX = _DT_CONFIG.G_FIX
_m = _DT_CONFIG.m
_n_s = _DT_CONFIG.n_s
_rho = _DT_CONFIG.rho       # 12
_n_out = _DT_CONFIG.n_out   # 18

# ---- Link budget parameters ----
# Feeder link (Gateway → Satellite, Ka-band)
_F_FEEDER_HZ = 20e9          # Ka-band carrier
_G_TX_GW_DBI = 50.0          # Gateway dish antenna gain
_G_RX_SAT_DBI = 35.0         # Satellite Rx antenna gain
_NF_SAT_DB = 3.0             # Satellite receiver noise figure
_D_FEEDER_M = 800e3          # GW-to-SAT slant range (fixed)

# Access link (Satellite → UE, S-band)
_F_ACCESS_HZ = 2e9           # S-band carrier
_G_TX_SAT_DBI = 30.0         # Satellite Tx antenna gain
_G_RX_UE_DBI = 0.0           # UE handheld antenna gain
_NF_UE_DB = 7.0              # UE receiver noise figure

# Shared
_BW_HZ = 5e6                 # System bandwidth
_R_EARTH_M = 6371e3
_H_ORB_M = 600e3
_ALPHA_POWER = 0.5            # Power split: P_SAT = α · P_total

# Precompute feeder link FSPL and noise power
_kB = 1.38e-23
_T0 = 290.0
_LAMBDA_FEEDER = SPEED_OF_LIGHT / _F_FEEDER_HZ
_LAMBDA_ACCESS = SPEED_OF_LIGHT / _F_ACCESS_HZ
_FSPL_FEEDER_DB = 20.0 * np.log10(4.0 * np.pi * _D_FEEDER_M / _LAMBDA_FEEDER)
_N0_SAT_DBW = 10.0 * np.log10(_kB * _T0 * _BW_HZ) + _NF_SAT_DB
_N0_UE_DBW = 10.0 * np.log10(_kB * _T0 * _BW_HZ) + _NF_UE_DB
# Feeder link gain: γ_GS(dB) = P_GW(dBW) + _FEEDER_GAIN_DB
_FEEDER_GAIN_DB = _G_TX_GW_DBI + _G_RX_SAT_DBI - _FSPL_FEEDER_DB - _N0_SAT_DBW
# Access link gain (excluding FSPL): γ_SU(dB) = P_SAT(dBW) + _ACCESS_GAIN_DB - FSPL_access
_ACCESS_GAIN_EXCL_FSPL_DB = _G_TX_SAT_DBI + _G_RX_UE_DBI - _N0_UE_DBW

# Multipath TDL-D profile: L=2 taps, powers decay 3 dB per tap
_L_TAPS = 2
_TAP_POWERS_DB = np.array([0.0, -3.0])
_TAP_POWERS_LIN = db_to_linear(_TAP_POWERS_DB)
_TAP_POWERS_LIN = _TAP_POWERS_LIN / np.sum(_TAP_POWERS_LIN)  # normalise to 1

# Pre-generate candidate G_DIV matrices for Algorithm 2
_T_POOL_RNG = np.random.default_rng(42)
_N_POOL = 200

_block_weights_pool = np.zeros((_N_POOL, _n_s), dtype=float)
_G_DIV_POOL = []
_G_FIX_block_weights = np.array([
    np.sum(_G_FIX[:, i * _m:(i + 1) * _m]) for i in range(_n_s)
], dtype=float)

# Pre-generate all 2^rho input vectors for MAP soft diversity demapping
_N_CW = 1 << _rho  # 4096 for rho=12
_X_ALL = np.zeros((_rho, _N_CW), dtype=np.int8)
for _i_cw in range(_N_CW):
    for _b in range(_rho):
        _X_ALL[_b, _i_cw] = (_i_cw >> _b) & 1

for _i in range(_N_POOL):
    _T = generate_candidate_T(_DT_CONFIG.rho, _T_POOL_RNG)
    _G_DIV = gf2_matmul(_T, _G_FIX)
    _G_DIV_POOL.append(_G_DIV)
    for _j in range(_n_s):
        _block_weights_pool[_i, _j] = np.sum(_G_DIV[:, _j * _m:(_j + 1) * _m])

# Precompute codewords and bipolar form for G_FIX (used by MAP soft demapping)
_Z_ALL_FIX = np.mod(_G_FIX.T.astype(int) @ _X_ALL.astype(int), 2).astype(np.int8)
_S_ALL_FIX = 1.0 - 2.0 * _Z_ALL_FIX.astype(np.float64)  # (n_out, N_CW)

# Precompute bit-value masks for MAP LLR extraction
_X_MASK_0 = [_X_ALL[i, :] == 0 for i in range(_rho)]
_X_MASK_1 = [_X_ALL[i, :] == 1 for i in range(_rho)]


# ============================================================================
#  Link budget helpers
# ============================================================================

def _slant_range_m(elevation_deg):
    """Slant range from UE elevation angle (Eq. 9)."""
    eps_rad = np.radians(elevation_deg)
    sin_eps = np.sin(eps_rad)
    return (-_R_EARTH_M * sin_eps
            + np.sqrt((_R_EARTH_M * sin_eps) ** 2
                      + 2 * _R_EARTH_M * _H_ORB_M + _H_ORB_M ** 2))


def _access_fspl_dB(slant_range_m):
    """Free-space path loss for the access link (S-band)."""
    return 20.0 * np.log10(4.0 * np.pi * slant_range_m / _LAMBDA_ACCESS)


def _power_to_link_snrs(P_total_dBW, access_fspl_dB):
    """
    Convert total transmit power to per-link SNRs via link budgets.

    Equal power split: P_SAT = α · P_total, P_GW = (1-α) · P_total.

    Returns (gamma_gs_linear, gamma_su_linear).
    """
    P_total_lin = db_to_linear(P_total_dBW)
    P_SAT_lin = _ALPHA_POWER * P_total_lin
    P_GW_lin = (1.0 - _ALPHA_POWER) * P_total_lin

    gamma_gs_dB = linear_to_db(P_GW_lin) + _FEEDER_GAIN_DB
    gamma_su_dB = linear_to_db(P_SAT_lin) + _ACCESS_GAIN_EXCL_FSPL_DB - access_fspl_dB

    return db_to_linear(gamma_gs_dB), db_to_linear(gamma_su_dB)


# ============================================================================
#  Channel: multipath fading + MRC + cascaded SNR
# ============================================================================

def _generate_multipath_per_subchannel(gamma_avg_SU, gamma_gs, n_subchannels,
                                        K_rice, p_block_sub, shadow_loss_dB,
                                        rng):
    """
    Generate per-subchannel effective E2E SNR (paper Section II).

    Channel model:
    - L multipath taps per subchannel with Rician fading (LOS) + Rayleigh (NLOS)
    - MRC combining across taps (Eq. 28)
    - Independent per-subchannel blockage (3GPP NTN model)
    - Cascaded feeder+access SNR (Eq. 23)

    Args:
        gamma_avg_SU: average access-link SNR (linear)
        gamma_gs: feeder-link SNR (linear)
        n_subchannels: number of subchannels
        K_rice: Rician K-factor for LOS tap
        p_block_sub: per-subchannel blockage probability (scalar)
        shadow_loss_dB: deep blockage loss in dB
        rng: numpy Generator

    Returns:
        gamma_e2e: (n_subchannels,) effective E2E SNR per subchannel
        blocked_mask: (n_subchannels,) bool, True = blocked
    """
    gamma_mrc = np.zeros(n_subchannels)

    for j in range(n_subchannels):
        # Per-tap fading: LOS tap (Rician) + NLOS taps (Rayleigh)
        fading_power = np.zeros(_L_TAPS)
        for p in range(_L_TAPS):
            if p == 0:
                # LOS tap: Rician fading |h|^2
                # h = sqrt(K/(K+1)) * e^{j*phase} + sqrt(1/(K+1)) * w
                los_mean = np.sqrt(K_rice / (K_rice + 1))
                nlos_std = np.sqrt(1.0 / (2.0 * (K_rice + 1)))
                h_real = los_mean + rng.normal(0, nlos_std)
                h_imag = rng.normal(0, nlos_std)
                fading_power[p] = h_real**2 + h_imag**2
            else:
                # NLOS taps: Rayleigh fading |h|^2 ~ Exp(1)
                fading_power[p] = rng.exponential(1.0)

        # MRC combining: gamma_MRC = gamma_avg * sum_p(P_p * |h_p|^2)
        gamma_mrc[j] = gamma_avg_SU * np.sum(_TAP_POWERS_LIN * fading_power)

    # Independent per-subchannel blockage (3GPP NTN model)
    # Each subchannel is independently blocked with probability p_block_sub
    blocked_mask = rng.random(n_subchannels) < p_block_sub
    if np.any(blocked_mask):
        shadow_linear = db_to_linear(-shadow_loss_dB)
        gamma_mrc[blocked_mask] *= shadow_linear

    # Cascaded SNR: gamma_e2e = (gamma_GS * gamma_MRC) / (gamma_GS + gamma_MRC + 1)
    gamma_e2e = SNRCalculator.cascaded_snr(gamma_gs, gamma_mrc)

    return gamma_e2e, blocked_mask


# ============================================================================
#  Diversity transform end-to-end signal path
# ============================================================================

def _diversity_transform_transmit_receive(coded_bits, G_DIV, gamma_e2e_per_sub,
                                           erased_mask, eta, rng):
    """
    End-to-end diversity transform signal path with MAP soft demapping
    and reliability-aware eta-scaled LLRs (paper Section IV, Eq. 38-39):
      coded bits -> pad -> reshape (rho x n_pos) -> G_DIV^T per column
      -> per-subchannel AWGN channel -> eta-scaled LLRs -> MAP demapping -> LLR

    The eta-scaling attenuates unreliable subchannel contributions:
      Lambda_i^(j) <- eta_i * Lambda_i^(j)  (Eq. 38)
    where eta_i = 1 - epsilon_i is the predicted reliability (Eq. 34-35).

    The MAP demapper enumerates all 2^rho possible input vectors and extracts
    per-bit LLR via max-log-MAP:
      L(x_i) = max_{x: x_i=0} corr(x) - max_{x: x_i=1} corr(x)

    T-labeling (Algorithm 2) changes the input-to-codeword mapping, which
    combined with eta-scaling gives different MAP LLR distributions across
    subchannels, providing adaptive gain.

    Args:
        coded_bits: (N,) polar-encoded bits {0,1}
        G_DIV: (rho x n_out) binary diversity transform matrix
        gamma_e2e_per_sub: (n_subchannels,) per-subchannel E2E SNR
        erased_mask: (n_subchannels,) bool, True = erased
        eta: (n_subchannels,) reliability metric (paper Eq. 34-35)
        rng: numpy Generator

    Returns:
        llr_vector: (N,) reconstructed LLR vector for polar decoder,
                    or None if all subchannels erased
    """
    N = len(coded_bits)

    # Pad to multiple of rho
    n_pos = int(np.ceil(N / _rho))  # 22 for N=256, rho=12
    N_pad = _rho * n_pos            # 264
    coded_padded = np.zeros(N_pad, dtype=np.int8)
    coded_padded[:N] = coded_bits

    # Reshape to (rho x n_pos) matrix
    X = coded_padded.reshape(_rho, n_pos)

    # Apply diversity transform per column: Z = (G_DIV^T @ X) mod 2
    # Z is (n_out x n_pos) = (18 x 22)
    Z = np.mod(G_DIV.T @ X, 2).astype(np.int8)

    # Per-subchannel AWGN-equivalent channel with eta-scaled LLRs (Eq. 38-39)
    llr_z = np.zeros((_n_out, n_pos))
    for j in range(_n_s):
        gamma_j = gamma_e2e_per_sub[j]
        eta_j = eta[j]
        for b in range(_m):
            row_idx = j * _m + b
            if erased_mask[j]:
                llr_z[row_idx, :] = 0.0  # no information from erased subchannel
            else:
                s = 1.0 - 2.0 * Z[row_idx, :].astype(np.float64)  # bipolar
                sigma2 = 1.0 / (2.0 * max(gamma_j, 1e-10))
                noise = rng.normal(0, np.sqrt(sigma2), n_pos)
                y = s + noise
                llr_z[row_idx, :] = eta_j * 4.0 * gamma_j * y

    # Check at least one subchannel survives
    if np.all(erased_mask):
        return None

    # --- MAP soft diversity demapping ---
    # Compute bipolar codewords S_ALL for this G_DIV (reuse precomputed for G_FIX)
    is_fixed = (G_DIV is _G_FIX) or np.array_equal(G_DIV, _G_FIX)
    if is_fixed:
        S_ALL = _S_ALL_FIX  # (n_out, N_CW)
    else:
        Z_ALL = np.mod(G_DIV.T.astype(int) @ _X_ALL.astype(int), 2)
        S_ALL = 1.0 - 2.0 * Z_ALL.astype(np.float64)

    # Correlation of each codeword with eta-scaled channel LLRs:
    # corr[j, l] = sum_r L_z[r,l] * S_ALL[r,j] / 2
    corr = (S_ALL.T @ llr_z) * 0.5  # (N_CW, n_pos)

    # Max-log-MAP: L(x_i) = max_{x:x_i=0} corr - max_{x:x_i=1} corr
    X_llr = np.zeros((_rho, n_pos))
    for i in range(_rho):
        X_llr[i, :] = (np.max(corr[_X_MASK_0[i], :], axis=0)
                        - np.max(corr[_X_MASK_1[i], :], axis=0))

    # Flatten and strip padding to recover N LLRs
    llr_vector = X_llr.flatten()[:N]
    return llr_vector


# ============================================================================
#  Monte Carlo trial
# ============================================================================

def _mc_trial_one_snr(gamma_avg_SU, gamma_gs, N, K, n_subchannels,
                       max_erasures, K_rice, erasure_threshold, p_block_sub,
                       shadow_loss_dB, polar_code, rng):
    """
    Run one MC trial for all 7 curves at given per-link SNRs.

    Channel: L-tap multipath Rician fading per subchannel with MRC combining,
    independent per-subchannel blockage, and cascaded feeder+access SNR.

    Returns dict of bool (True = block error) for the 4 practical curves
    and float BLER for the 3 PPV bounds.
    """
    # Generate per-subchannel E2E SNR with multipath + MRC + cascaded
    gamma_e2e, blocked_mask = _generate_multipath_per_subchannel(
        gamma_avg_SU, gamma_gs, n_subchannels, K_rice, p_block_sub,
        shadow_loss_dB, rng)

    # ---- Erasure mask (shared across PPV bounds and diversity MC curves) ----
    erased_mask = gamma_e2e < erasure_threshold
    n_erased = int(np.sum(erased_mask))
    k_c = _DT_CONFIG.k_c

    # ---- Reliability metric η (paper Eq. 34-35) ----
    # η_i = 1 - ε_i where ε_i = conditional_bler(γ_i, N_sub, R)
    R = K / N
    N_sub = N // n_subchannels
    eta = np.zeros(n_subchannels)
    for j in range(n_subchannels):
        if erased_mask[j]:
            eta[j] = 0.0
        else:
            eps_j = conditional_bler(gamma_e2e[j], N_sub, R)
            eta[j] = 1.0 - eps_j

    # ---- Algorithm 2: select best G_DIV (shared for PPV_ada and Curve 4) ----
    best_Q = np.dot(_G_FIX_block_weights, eta)
    best_G_DIV = _G_FIX
    best_block_weights = _G_FIX_block_weights.copy()
    for i in range(_N_POOL):
        Q = np.dot(_block_weights_pool[i], eta)
        if Q > best_Q:
            best_Q = Q
            best_G_DIV = _G_DIV_POOL[i]
            best_block_weights = _block_weights_pool[i].copy()

    # ---- Semi-analytical FBL bounds (Section III) ----
    ppv_no_div = bler_no_diversity(gamma_e2e, N, K)
    ppv_fix_div = bler_fixed_diversity(gamma_e2e, erased_mask, k_c, N, K)

    # PPV adaptive: Q-metric scaling (Algorithm 2)
    Q_fix_val = float(np.dot(_G_FIX_block_weights, eta))
    Q_ratio = float(best_Q) / max(Q_fix_val, 1e-10)
    Q_ratio = max(Q_ratio, 1.0)
    ppv_ada_div = ppv_fix_div / Q_ratio

    # ---- Generate info bits (shared across practical curves) ----
    info_bits = rng.integers(0, 2, size=K).astype(np.int8)

    # ---- Curve 1: No interleaver (sequential subchannel mapping) ----
    bits_per_sub = N // n_subchannels
    snr_per_bit = np.zeros(N)
    for s in range(n_subchannels):
        start = s * bits_per_sub
        end = min((s + 1) * bits_per_sub, N)
        snr_per_bit[start:end] = gamma_e2e[s]
    # Remaining bits (if N not divisible by n_subchannels) go to last subchannel
    if bits_per_sub * n_subchannels < N:
        snr_per_bit[bits_per_sub * n_subchannels:] = gamma_e2e[-1]
    llr_1 = polar_code.transmit_per_bit_snr(info_bits, snr_per_bit, rng)
    err_1 = polar_code.decode_check(info_bits, llr_1)

    # ---- Curve 2: With random interleaver ----
    perm = rng.permutation(N)
    snr_per_bit_intlv = np.zeros(N)
    for i in range(N):
        sub_idx = min(perm[i] // bits_per_sub, n_subchannels - 1)
        snr_per_bit_intlv[i] = gamma_e2e[sub_idx]
    llr_2 = polar_code.transmit_per_bit_snr(info_bits, snr_per_bit_intlv, rng)
    err_2 = polar_code.decode_check(info_bits, llr_2)

    # ---- Curve 3: Fixed diversity transform (G_FIX in signal path) ----
    if n_erased > max_erasures:
        err_3 = True
    else:
        coded_bits = polar_code.encode(info_bits)
        llr_3 = _diversity_transform_transmit_receive(
            coded_bits, _G_FIX, gamma_e2e, erased_mask, eta, rng)
        if llr_3 is None:
            err_3 = True
        else:
            decoded_3 = polar_code.decode(llr_3)
            err_3 = not np.array_equal(decoded_3, info_bits)

    # ---- Curve 4: Adaptive diversity transform (Algorithm 2 T-labeling
    #       + erasure-aware power allocation) ----
    # The adaptive scheme has CSI: it knows which subchannels are blocked.
    # Power from erased subchannels is redistributed equally to survivors,
    # conserving total power: power_alloc[surviving] = n_s / n_surviving.
    if n_erased > max_erasures:
        err_4 = True
    else:
        n_surviving = n_subchannels - n_erased
        power_alloc = np.ones(n_subchannels)
        if n_surviving > 0 and n_erased > 0:
            power_alloc[~erased_mask] = float(n_subchannels) / n_surviving
            power_alloc[erased_mask] = 0.0
        gamma_e2e_adaptive = gamma_e2e * power_alloc

        coded_bits_4 = polar_code.encode(info_bits)
        llr_4 = _diversity_transform_transmit_receive(
            coded_bits_4, best_G_DIV, gamma_e2e_adaptive, erased_mask, eta, rng)
        if llr_4 is None:
            err_4 = True
        else:
            decoded_4 = polar_code.decode(llr_4)
            err_4 = not np.array_equal(decoded_4, info_bits)

    return {
        'ppv_no_div': ppv_no_div,
        'ppv_fix_div': ppv_fix_div,
        'ppv_ada_div': ppv_ada_div,
        'err_1': err_1,
        'err_2': err_2,
        'err_3': err_3,
        'err_4': err_4,
    }


# ============================================================================
#  BLER vs Total Transmit Power simulation (Monte Carlo)
# ============================================================================

def simulate_bler_vs_power(N: int = 256, K: int = 128,
                           n_subchannels: int = _n_s,
                           max_erasures: int = _DT_CONFIG.max_erasures,
                           power_range_dBW: Tuple[float, float] = (-6, 15),
                           ref_elevation_deg: float = 50.0,
                           n_points: int = 16,
                           n_mc: int = 5000) -> Dict:
    """
    BLER vs total transmit power P_total = P_GW + P_SAT for all 7 curves.

    Both feeder-link and access-link SNRs are derived from P_total via
    link budgets (Ka-band feeder, S-band access) with equal power split.
    """
    rng = np.random.default_rng(2026)
    power_dBW_arr = np.linspace(power_range_dBW[0], power_range_dBW[1], n_points)

    # Access link geometry at reference elevation
    ref_slant = _slant_range_m(ref_elevation_deg)
    ref_fspl = _access_fspl_dB(ref_slant)

    K_rice = 2.0                  # Rician K-factor (3 dB, NTN suburban)
    erasure_threshold = db_to_linear(-3.0)
    shadow_loss_dB = 20.0
    p_block_sub = 0.15            # Independent per-subchannel blockage probability

    polar_code = PolarCode(N, K, design_snr_dB=1.0, list_size=8)

    print(f"Computing BLER vs P_total (7 curves, end-to-end MC simulation)...")
    print(f"  Polar({N},{K}), Rate = {K/N:.2f}")
    print(f"  Subchannels: {n_subchannels}, MDS max erasures = {max_erasures}")
    print(f"  Power split: equal (alpha={_ALPHA_POWER})")
    print(f"  Feeder link: Ka-band {_F_FEEDER_HZ/1e9:.0f} GHz, "
          f"G_tx={_G_TX_GW_DBI:.0f} dBi, G_rx={_G_RX_SAT_DBI:.0f} dBi, "
          f"d={_D_FEEDER_M/1e3:.0f} km")
    print(f"  Access link: S-band {_F_ACCESS_HZ/1e9:.0f} GHz, "
          f"G_tx={_G_TX_SAT_DBI:.0f} dBi, G_rx={_G_RX_UE_DBI:.0f} dBi, "
          f"elev={ref_elevation_deg:.0f}deg (d={ref_slant/1e3:.0f} km)")
    print(f"  Multipath: L={_L_TAPS} taps, Rician K={10*np.log10(K_rice):.1f} dB")
    print(f"  Blockage: p_block={p_block_sub:.2f} per-subchannel, "
          f"loss={shadow_loss_dB} dB")
    print(f"  MC trials: {n_mc}")
    print()

    results = {key: np.zeros(n_points) for key in [
        'ppv_no_diversity', 'ppv_fixed_diversity', 'ppv_adaptive_diversity',
        'no_interleaver', 'interleaver', 'fixed', 'adaptive']}
    results['P_total_dBW'] = power_dBW_arr

    for idx, p_dBW in enumerate(power_dBW_arr):
        gamma_gs, gamma_su = _power_to_link_snrs(p_dBW, ref_fspl)

        ppv_no_acc = 0.0
        ppv_fix_acc = 0.0
        ppv_ada_acc = 0.0
        err_counts = [0, 0, 0, 0]

        for trial in range(n_mc):
            res = _mc_trial_one_snr(
                gamma_su, gamma_gs, N, K, n_subchannels, max_erasures,
                K_rice, erasure_threshold, p_block_sub,
                shadow_loss_dB, polar_code, rng)

            ppv_no_acc += res['ppv_no_div']
            ppv_fix_acc += res['ppv_fix_div']
            ppv_ada_acc += res['ppv_ada_div']
            err_counts[0] += int(res['err_1'])
            err_counts[1] += int(res['err_2'])
            err_counts[2] += int(res['err_3'])
            err_counts[3] += int(res['err_4'])

        results['ppv_no_diversity'][idx] = ppv_no_acc / n_mc
        results['ppv_fixed_diversity'][idx] = ppv_fix_acc / n_mc
        results['ppv_adaptive_diversity'][idx] = ppv_ada_acc / n_mc
        results['no_interleaver'][idx] = err_counts[0] / n_mc
        results['interleaver'][idx] = err_counts[1] / n_mc
        results['fixed'][idx] = err_counts[2] / n_mc
        results['adaptive'][idx] = err_counts[3] / n_mc

        gamma_su_dB = linear_to_db(gamma_su)
        gamma_gs_dB = linear_to_db(gamma_gs)
        print(f"  P_total = {p_dBW:5.1f} dBW "
              f"(gamma_SU={gamma_su_dB:.1f}dB, gamma_GS={gamma_gs_dB:.1f}dB): "
              f"NoIntlv={results['no_interleaver'][idx]:.4e}, "
              f"Intlv={results['interleaver'][idx]:.4e}, "
              f"Fixed={results['fixed'][idx]:.4e}, "
              f"Adaptive={results['adaptive'][idx]:.4e}")

    return results


# ============================================================================
#  BLER vs Elevation simulation (Monte Carlo)
# ============================================================================

def simulate_bler_vs_elevation(N: int = 256, K: int = 128,
                               n_subchannels: int = _n_s,
                               max_erasures: int = _DT_CONFIG.max_erasures,
                               P_total_dBW: float = 8.0,
                               elevation_range: Tuple[float, float] = (20, 90),
                               n_points: int = 8,
                               n_mc: int = 5000) -> Dict:
    """
    BLER vs elevation angle for all 7 curves (end-to-end MC simulation).

    Both link SNRs derived from P_total via link budget.
    Access-link slant range varies with elevation; feeder link is fixed.
    Elevation-dependent Rician K-factor and blockage probability.
    """
    rng = np.random.default_rng(2027)
    elevations = np.linspace(elevation_range[0], elevation_range[1], n_points)

    K_rice_min = 1.0              # 0 dB at low elevation
    K_rice_max = 5.0              # 7 dB at high elevation
    erasure_threshold = db_to_linear(-3.0)
    shadow_loss_dB = 20.0
    p_block_sub_low = 0.25        # per-subchannel blockage at low elevation
    p_block_sub_high = 0.08       # per-subchannel blockage at high elevation

    polar_code = PolarCode(N, K, design_snr_dB=1.0, list_size=8)

    print(f"\nComputing BLER vs Elevation (7 curves, end-to-end MC simulation)...")
    print(f"  P_total = {P_total_dBW:.1f} dBW ({10**(P_total_dBW/10):.2f} W), "
          f"alpha={_ALPHA_POWER}")
    print(f"  Multipath: L={_L_TAPS} taps")
    print(f"  Rician K: {10*np.log10(K_rice_min):.1f} dB (low elev) to "
          f"{10*np.log10(K_rice_max):.1f} dB (high elev)")
    print(f"  Blockage: p_block={p_block_sub_low:.2f} (low elev) to "
          f"{p_block_sub_high:.2f} (high elev), loss={shadow_loss_dB} dB")
    print(f"  MC trials: {n_mc}")

    results = {key: np.zeros(n_points) for key in [
        'ppv_no_diversity', 'ppv_fixed_diversity', 'ppv_adaptive_diversity',
        'no_interleaver', 'interleaver', 'fixed', 'adaptive']}
    results['elevation_deg'] = elevations
    results['P_total_dBW'] = P_total_dBW

    for idx, eps_deg in enumerate(elevations):
        # Elevation-dependent access link geometry
        d_access = _slant_range_m(eps_deg)
        fspl_access = _access_fspl_dB(d_access)
        gamma_gs, gamma_su = _power_to_link_snrs(P_total_dBW, fspl_access)
        gamma_su_dB = linear_to_db(gamma_su)

        elev_frac = (eps_deg - elevation_range[0]) / (elevation_range[1] - elevation_range[0])
        K_rice = K_rice_min + (K_rice_max - K_rice_min) * elev_frac
        p_block_sub = p_block_sub_low + (p_block_sub_high - p_block_sub_low) * elev_frac

        ppv_no_acc = 0.0
        ppv_fix_acc = 0.0
        ppv_ada_acc = 0.0
        err_counts = [0, 0, 0, 0]

        for trial in range(n_mc):
            res = _mc_trial_one_snr(
                gamma_su, gamma_gs, N, K, n_subchannels, max_erasures,
                K_rice, erasure_threshold, p_block_sub,
                shadow_loss_dB, polar_code, rng)

            ppv_no_acc += res['ppv_no_div']
            ppv_fix_acc += res['ppv_fix_div']
            ppv_ada_acc += res['ppv_ada_div']
            err_counts[0] += int(res['err_1'])
            err_counts[1] += int(res['err_2'])
            err_counts[2] += int(res['err_3'])
            err_counts[3] += int(res['err_4'])

        results['ppv_no_diversity'][idx] = ppv_no_acc / n_mc
        results['ppv_fixed_diversity'][idx] = ppv_fix_acc / n_mc
        results['ppv_adaptive_diversity'][idx] = ppv_ada_acc / n_mc
        results['no_interleaver'][idx] = err_counts[0] / n_mc
        results['interleaver'][idx] = err_counts[1] / n_mc
        results['fixed'][idx] = err_counts[2] / n_mc
        results['adaptive'][idx] = err_counts[3] / n_mc

        print(f"  Elev = {eps_deg:5.1f}deg (d={d_access/1e3:.0f}km, "
              f"gamma_SU={gamma_su_dB:.1f}dB, K={10*np.log10(K_rice):.1f}dB, "
              f"p_blk={p_block_sub:.2f}): "
              f"NoIntlv = {results['no_interleaver'][idx]:.4e}, "
              f"Intlv = {results['interleaver'][idx]:.4e}, "
              f"Fixed = {results['fixed'][idx]:.4e}, "
              f"Adaptive = {results['adaptive'][idx]:.4e}")

    return results


# ============================================================================
#  Plotting
# ============================================================================

def plot_bler_vs_power(results: Dict):
    """Plot BLER vs total transmit power for all 4 curves + 3 PPV bounds."""
    fig, ax = plt.subplots(figsize=(10, 7))
    p_total = results['P_total_dBW']

    ax.semilogy(p_total, np.maximum(results['ppv_adaptive_diversity'], 1e-8),
                's--', color='#2ca02c', linewidth=1.5, markersize=6,
                label='PPV Bound (adaptive diversity)')
    ax.semilogy(p_total, np.maximum(results['ppv_fixed_diversity'], 1e-8),
                'p--', color='#17becf', linewidth=1.5, markersize=6,
                label='PPV Bound (fixed diversity)')
    ax.semilogy(p_total, np.maximum(results['ppv_no_diversity'], 1e-8),
                'h--', color='#9467bd', linewidth=1.5, markersize=6,
                label='PPV Bound (no diversity)')
    ax.semilogy(p_total, np.maximum(results['no_interleaver'], 1e-8),
                'v-', color='#7f7f7f', linewidth=1.8, markersize=8,
                label='Standard Polar (no interleaver)')
    ax.semilogy(p_total, np.maximum(results['interleaver'], 1e-8),
                '^-', color='#d62728', linewidth=1.8, markersize=8,
                label='Standard Polar + random interleaver')
    ax.semilogy(p_total, np.maximum(results['fixed'], 1e-8),
                'D-', color='#ff7f0e', linewidth=2, markersize=8,
                label='Fixed Diversity Transform (MDS)')
    ax.semilogy(p_total, np.maximum(results['adaptive'], 1e-8),
                'o-', color='#1f77b4', linewidth=2.5, markersize=8,
                label='Adaptive Diversity Transform (Proposed)')

    # Shade the gap between Fixed and Adaptive to highlight adaptive gain
    fixed_clip = np.maximum(results['fixed'], 1e-8)
    adapt_clip = np.maximum(results['adaptive'], 1e-8)
    ax.fill_between(p_total, adapt_clip, fixed_clip,
                    alpha=0.18, color='#1f77b4',
                    label='_nolegend_')

    ax.set_xlabel(r'Total Transmit Power, $P_{\mathrm{GW}} + P_{\mathrm{SAT}}$ (dBW)',
                  fontsize=12)
    ax.set_ylabel('Block Error Rate (BLER)', fontsize=12)
    ax.set_title('BLER vs Total Transmit Power: Polar-Coded OTFS with Diversity Transforms\n'
                 f'(L={_L_TAPS} taps, MRC, cascaded relay, {_n_s} subchannels)',
                 fontsize=13)
    ax.legend(fontsize=9, loc='lower left')
    ax.grid(True, which='both', linestyle='--', alpha=0.7)
    ax.set_ylim(5e-4, 2)

    plt.tight_layout()
    plt.savefig('bler_comparison.png', dpi=300, bbox_inches='tight')
    print("Saved: bler_comparison.png")
    plt.close()


def plot_bler_vs_elevation(results: Dict):
    """Plot BLER vs elevation for all 4 curves + 3 PPV bounds."""
    fig, ax = plt.subplots(figsize=(10, 7))
    elev = results['elevation_deg']

    ax.semilogy(elev, np.maximum(results['ppv_adaptive_diversity'], 1e-7),
                's--', color='#2ca02c', linewidth=1.5, markersize=6,
                label='PPV Bound (adaptive diversity)')
    ax.semilogy(elev, np.maximum(results['ppv_fixed_diversity'], 1e-7),
                'p--', color='#17becf', linewidth=1.5, markersize=6,
                label='PPV Bound (fixed diversity)')
    ax.semilogy(elev, np.maximum(results['ppv_no_diversity'], 1e-7),
                'h--', color='#9467bd', linewidth=1.5, markersize=6,
                label='PPV Bound (no diversity)')
    ax.semilogy(elev, np.maximum(results['no_interleaver'], 1e-7),
                'v-', color='#7f7f7f', linewidth=1.8, markersize=8,
                label='Standard Polar (no interleaver)')
    ax.semilogy(elev, np.maximum(results['interleaver'], 1e-7),
                '^-', color='#d62728', linewidth=1.8, markersize=8,
                label='Standard Polar + random interleaver')
    ax.semilogy(elev, np.maximum(results['fixed'], 1e-7),
                'D-', color='#ff7f0e', linewidth=2, markersize=8,
                label='Fixed Diversity Transform (MDS)')
    ax.semilogy(elev, np.maximum(results['adaptive'], 1e-7),
                'o-', color='#1f77b4', linewidth=2.5, markersize=8,
                label='Adaptive Diversity Transform (Proposed)')

    # Shade the gap between Fixed and Adaptive
    fixed_clip = np.maximum(results['fixed'], 1e-7)
    adapt_clip = np.maximum(results['adaptive'], 1e-7)
    ax.fill_between(elev, adapt_clip, fixed_clip,
                    alpha=0.18, color='#1f77b4',
                    label='_nolegend_')

    ax.set_xlabel('Elevation Angle (degrees)', fontsize=12)
    ax.set_ylabel('Block Error Rate (BLER)', fontsize=12)
    ax.set_title('BLER vs Elevation: Polar-Coded OTFS with Diversity Transforms\n'
                 f'(L={_L_TAPS} taps, MRC, cascaded relay, {_n_s} subchannels, '
                 r'$P_{\mathrm{total}}$' + f'={results.get("P_total_dBW", 8.0):.0f} dBW)',
                 fontsize=13)
    ax.legend(fontsize=9, loc='upper right')
    ax.grid(True, which='both', linestyle='--', alpha=0.7)
    ax.set_ylim(5e-4, 2)

    plt.tight_layout()
    plt.savefig('bler_vs_elevation_comparison.png', dpi=300, bbox_inches='tight')
    print("Saved: bler_vs_elevation_comparison.png")
    plt.close()


# ============================================================================
#  CSV export
# ============================================================================

_CSV_COLUMNS = ['PPV_no_diversity', 'PPV_fixed_diversity',
                'PPV_adaptive_diversity', 'no_interleaver', 'interleaver',
                'fixed', 'adaptive']
_RESULT_KEYS = ['ppv_no_diversity', 'ppv_fixed_diversity',
                'ppv_adaptive_diversity', 'no_interleaver', 'interleaver',
                'fixed', 'adaptive']


def save_results_csv(results_power: Dict, results_elev: Dict):
    """Save simulation results to CSV files."""
    # Power sweep CSV
    with open('bler_vs_power.csv', 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['P_total_dBW'] + _CSV_COLUMNS)
        for i in range(len(results_power['P_total_dBW'])):
            row = [f"{results_power['P_total_dBW'][i]:.2f}"]
            for key in _RESULT_KEYS:
                row.append(f"{results_power[key][i]:.6e}")
            writer.writerow(row)
    print("Saved: bler_vs_power.csv")

    # Elevation CSV
    with open('bler_vs_elevation.csv', 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['elevation_deg'] + _CSV_COLUMNS)
        for i in range(len(results_elev['elevation_deg'])):
            row = [f"{results_elev['elevation_deg'][i]:.1f}"]
            for key in _RESULT_KEYS:
                row.append(f"{results_elev[key][i]:.6e}")
            writer.writerow(row)
    print("Saved: bler_vs_elevation.csv")


# ============================================================================
#  Gain analysis
# ============================================================================

def print_gain_analysis(results_snr: Dict, results_elev: Dict):
    """Print gain analysis across all 7 curves."""
    print("\n" + "=" * 80)
    print("GAIN ANALYSIS: 7-Curve Comparison (End-to-End MC Simulation)")
    print("=" * 80)

    labels = ['ppv_no_diversity', 'ppv_fixed_diversity', 'ppv_adaptive_diversity',
              'no_interleaver', 'interleaver', 'fixed', 'adaptive']
    short = ['PPV', 'PPV_Fix', 'PPV_Ada', 'NoIntlv', 'Intlv', 'Fixed', 'Adaptive']

    p_arr = results_snr['P_total_dBW']

    print("\nPower-based comparison (BLER at target P_total points):")
    header = f"  {'P_total':>8s}"
    for s in short:
        header += f"  {s:>12s}"
    print(header)
    print("  " + "-" * (8 + 14 * len(short)))

    for p_target in [-4, 0, 4, 8, 12, 15]:
        idx = np.argmin(np.abs(p_arr - p_target))
        row = f"  {p_target:>5d}dBW"
        for lbl in labels:
            row += f"  {results_snr[lbl][idx]:>12.2e}"
        print(row)

    # Adaptive vs Fixed gain
    print("\nAdaptive vs Fixed gain:")
    for p_target in [0, 4, 8, 12, 15]:
        idx = np.argmin(np.abs(p_arr - p_target))
        f = results_snr['fixed'][idx]
        a = results_snr['adaptive'][idx]
        if f > 1e-10 and a > 1e-10 and f > a:
            ratio = f / a
            gain_dB = 10 * np.log10(ratio)
            print(f"  P={p_target}dBW: Fixed={f:.2e}, Adaptive={a:.2e}, "
                  f"Gain={gain_dB:.1f}dB ({ratio:.1f}x)")

    # Verify ordering
    print("\nOrdering check (should be: NoIntlv > Intlv > Fixed > Adaptive):")
    violations = 0
    for idx in range(len(p_arr)):
        p = p_arr[idx]
        b1 = results_snr['no_interleaver'][idx]
        b2 = results_snr['interleaver'][idx]
        b3 = results_snr['fixed'][idx]
        b4 = results_snr['adaptive'][idx]
        ok = b1 >= b2 >= b3 >= b4
        if not ok:
            violations += 1
            print(f"  VIOLATION at P={p:.1f}dBW: "
                  f"{b1:.2e} >= {b2:.2e} >= {b3:.2e} >= {b4:.2e}")
    if violations == 0:
        print("  All power points: ordering OK!")

    print("=" * 80)


# ============================================================================
#  Main
# ============================================================================

def main():
    """Run full 7-curve BLER comparison (end-to-end MC simulation)."""
    print("=" * 80)
    print("BLER COMPARISON: Polar-Coded OTFS with Diversity Transforms")
    print("(End-to-End Monte Carlo — Diversity Transform in Signal Path)")
    print("=" * 80)

    N, K = 256, 128
    n_subchannels = _DT_CONFIG.n_s
    max_erasures = _DT_CONFIG.max_erasures

    print(f"\nSystem Parameters:")
    print(f"  Polar Code: ({N}, {K}), Rate = {K/N:.2f}")
    print(f"  Subchannels: {n_subchannels}")
    print(f"  Diversity Transform: GF(2^{_DT_CONFIG.m}), "
          f"k_c={_DT_CONFIG.k_c}, n_s={_DT_CONFIG.n_s}")
    print(f"  G_FIX: {_DT_CONFIG.G_FIX.shape} binary matrix, "
          f"rank={gf2_rank(_DT_CONFIG.G_FIX)}")
    print(f"  MDS: d_min={_DT_CONFIG.d_min}, max erasures={max_erasures}")
    print(f"  Multipath: L={_L_TAPS} taps, MRC combining")
    print(f"  Link budget:")
    print(f"    Feeder: Ka-band {_F_FEEDER_HZ/1e9:.0f} GHz, "
          f"G_tx={_G_TX_GW_DBI:.0f} dBi, G_rx={_G_RX_SAT_DBI:.0f} dBi, "
          f"d={_D_FEEDER_M/1e3:.0f} km, NF={_NF_SAT_DB:.0f} dB")
    print(f"    Access: S-band {_F_ACCESS_HZ/1e9:.0f} GHz, "
          f"G_tx={_G_TX_SAT_DBI:.0f} dBi, G_rx={_G_RX_UE_DBI:.0f} dBi, "
          f"NF={_NF_UE_DB:.0f} dB")
    print(f"    Power split: alpha={_ALPHA_POWER} (equal)")
    print(f"    BW={_BW_HZ/1e6:.0f} MHz, orbit={_H_ORB_M/1e3:.0f} km")

    # BLER vs Total Transmit Power
    print("\n" + "-" * 60)
    results_power = simulate_bler_vs_power(
        N, K, n_subchannels, max_erasures,
        power_range_dBW=(-6, 15), n_points=16, n_mc=3000)

    # BLER vs Elevation
    print("\n" + "-" * 60)
    P_total_elev = 8.0
    results_elev = simulate_bler_vs_elevation(
        N, K, n_subchannels, max_erasures,
        P_total_dBW=P_total_elev,
        elevation_range=(20, 90), n_points=8, n_mc=3000)

    # Analysis
    print_gain_analysis(results_power, results_elev)

    # Plot and export
    print("\nGenerating plots and CSV files...")
    plot_bler_vs_power(results_power)
    plot_bler_vs_elevation(results_elev)
    save_results_csv(results_power, results_elev)

    return results_power, results_elev


if __name__ == "__main__":
    results = main()
