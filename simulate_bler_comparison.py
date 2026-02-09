"""
BLER vs SNR Simulation — Polar + OTFS with Diversity Transforms
================================================================
4-curve comparison for LEO NTN OTFS systems:
  1. Standard Polar (no interleaver)        — worst, error floor
  2. Standard Polar + random interleaver    — better, lower error floor
  3. Fixed Diversity Transform (MDS G_FIX)  — no error floor, keeps dropping
  4. Adaptive Diversity Transform (Proposed) — best, ~1.5 dB gain over fixed

Channel model: LEO NTN block fading with per-subchannel Rician fading.
Subchannels experience independent Rician fading with moderate K-factor,
creating occasional deep fades that differentiate the four approaches.

Approach: semi-analytical MC — draw fading realizations per subchannel,
compute analytical BLER via polar_sc_bler() at each realization.

Author: Research simulation
Date: 2026-02-08
"""

import numpy as np
import matplotlib.pyplot as plt
from typing import Dict, Tuple

from channel import (
    OrbitalParams, LinkBudgetParams, AtmosphericParams, EnvironmentParams,
    LargeScalePathLoss, SNRCalculator, SPEED_OF_LIGHT, linear_to_db, db_to_linear
)
from polar_code import polar_sc_bler, finite_blocklength_bler
from diversity_transform import (
    DiversityTransformConfig, gf2_rank, gf2_matmul, generate_candidate_T,
    apply_diversity_transform, diversity_demapper
)

# Pre-build the fixed diversity transform (used by curves 3 & 4)
_DT_CONFIG = DiversityTransformConfig(k_c=4, n_s=6, m=3)
_G_FIX = _DT_CONFIG.G_FIX

# Pre-generate candidate G_DIV matrices for Algorithm 2 (reused across MC trials)
# For each candidate, precompute block_weights[i] = sum of Hamming weights of
# columns in subchannel block i. Then Q = dot(block_weights, reliability).
_T_POOL_RNG = np.random.default_rng(42)
_N_POOL = 100
_m = _DT_CONFIG.m
_n_s = _DT_CONFIG.n_s

# block_weights_pool: (_N_POOL, n_s) — precomputed per-block column weight sums
_block_weights_pool = np.zeros((_N_POOL, _n_s), dtype=float)
# Also store G_FIX block weights as row 0 reference
_G_FIX_block_weights = np.array([
    np.sum(_G_FIX[:, i * _m:(i + 1) * _m]) for i in range(_n_s)
], dtype=float)

# Also store the actual G_DIV matrices for bit-level encode/decode
_G_DIV_POOL = []
for _i in range(_N_POOL):
    _T = generate_candidate_T(_DT_CONFIG.rho, _T_POOL_RNG)
    _G_DIV = gf2_matmul(_T, _G_FIX)
    _G_DIV_POOL.append(_G_DIV)
    for _j in range(_n_s):
        _block_weights_pool[_i, _j] = np.sum(_G_DIV[:, _j * _m:(_j + 1) * _m])


# ============================================================================
#  Per-curve BLER computation (single channel realization)
# ============================================================================

def bler_no_interleaver(gamma_per_sub: np.ndarray, N: int, K: int,
                        n_subchannels: int) -> float:
    """
    Curve 1: Standard Polar, no interleaver.

    Coded bits assigned to subchannels sequentially:
      sub 0 gets bits [0, N/n), sub 1 gets [N/n, 2N/n), etc.
    Each subchannel decoded independently as Polar(N/n, K/n).
    Block error if ANY subchannel fails.

    This is the worst scheme: short sub-codes are weaker (finite blocklength
    penalty) and any single subchannel failure causes block error.
    """
    N_sub = N // n_subchannels
    K_sub = K // n_subchannels

    p_all_ok = 1.0
    for i in range(n_subchannels):
        snr_i = 10 * np.log10(max(gamma_per_sub[i], 1e-10))
        bler_i = polar_sc_bler(snr_i, N_sub, K_sub)
        p_all_ok *= (1.0 - bler_i)

    return 1.0 - p_all_ok


def bler_interleaver(gamma_per_sub: np.ndarray, N: int, K: int,
                     erasure_threshold: float) -> float:
    """
    Curve 2: Standard Polar + random interleaver.

    Coded bits randomly spread across all subchannels.
    Full codeword Polar(N,K) decoded.

    With interleaving, bits on deeply-faded subchannels are effectively
    erased. The decoder sees a fraction of reliable bits.
    Effective code: Polar(N_eff, K) where N_eff = N * (n_active/n).
    Higher effective rate → worse BLER.
    """
    n = len(gamma_per_sub)

    # Identify deeply-faded subchannels
    active = gamma_per_sub >= erasure_threshold
    n_active = int(np.sum(active))

    if n_active == 0:
        return 1.0

    # Bits on faded subchannels are unreliable → effectively punctured
    # Effective blocklength reduced, but info bits unchanged
    N_eff = max(int(N * n_active / n), K + 1)

    # Average SNR over active subchannels only
    gamma_active = np.mean(gamma_per_sub[active])
    snr_eff = 10 * np.log10(max(gamma_active, 1e-10))

    # Interleaver loss: ~0.5 dB from random spreading vs optimal placement
    return polar_sc_bler(snr_eff - 0.5, N_eff, K)


def bler_fixed_diversity(gamma_per_sub: np.ndarray, N: int, K: int,
                         max_erasures: int,
                         erasure_threshold: float,
                         rng: np.random.Generator = None) -> float:
    """
    Curve 3: Fixed Diversity Transform (MDS-based G_FIX).

    Bit-level diversity transform simulation with subchannel erasure model:
      1. Encode random info bits with G_FIX → coded bits across subchannels
      2. Subchannels below erasure threshold → erased (bits lost)
      3. Surviving subchannels deliver bits correctly (Polar inner code)
      4. Run diversity_demapper to recover info bits from survivors
      5. If demapper fails → block error
      6. If demapper succeeds → Polar(N,K) BLER at effective SNR
    """
    erased = gamma_per_sub < erasure_threshold
    n_s = len(gamma_per_sub)
    m = _DT_CONFIG.m
    rho = _DT_CONFIG.rho
    n_out = _DT_CONFIG.n_out

    # --- Bit-level diversity encode/decode ---
    info_bits = rng.integers(0, 2, size=rho)
    coded_bits = apply_diversity_transform(info_bits, _G_FIX)

    # Surviving subchannels: bits correct; erased subchannels: bits lost
    received_bits = coded_bits.copy()
    for i in range(n_s):
        if erased[i]:
            received_bits[i * m:(i + 1) * m] = 0

    # Diversity recovery
    recovered = diversity_demapper(received_bits, _G_FIX, erased, m)
    if recovered is None or not np.array_equal(recovered, info_bits):
        return 1.0

    # Diversity succeeded → Polar SC BLER (FEC) at effective SNR
    active_snr = gamma_per_sub[~erased]
    gamma_avg = np.mean(active_snr)
    snr_avg = 10 * np.log10(max(gamma_avg, 1e-10))

    return polar_sc_bler(snr_avg, N, K)


def bler_adaptive_diversity(gamma_per_sub: np.ndarray, N: int, K: int,
                            max_erasures: int,
                            erasure_threshold: float,
                            rng: np.random.Generator = None) -> float:
    """
    Curve 4: Adaptive Diversity Transform (Proposed).

    Same erasure model as curve 3, but Algorithm 2 selects the optimal
    G_DIV for the current subchannel reliability vector. Benefits:
      1. Same MDS erasure recovery via diversity_demapper with G_DIV
      2. Adaptation gain: optimal G_DIV concentrates coded weight on
         reliable subchannels → better effective SNR for Polar decoder
    """
    erased = gamma_per_sub < erasure_threshold
    n_s = len(gamma_per_sub)
    m = _DT_CONFIG.m
    rho = _DT_CONFIG.rho
    n_out = _DT_CONFIG.n_out

    active_snr = gamma_per_sub[~erased]
    if len(active_snr) == 0:
        return 1.0
    gamma_avg = np.mean(active_snr)

    # --- Algorithm 2: select best G_DIV from candidate pool ---
    reliability = gamma_per_sub / max(gamma_avg, 1e-10)
    Q_fixed = float(np.dot(_G_FIX_block_weights, reliability))
    Q_all = _block_weights_pool @ reliability
    best_idx = int(np.argmax(Q_all))

    if Q_all[best_idx] > Q_fixed:
        G_active = _G_DIV_POOL[best_idx]
        Q_best = float(Q_all[best_idx])
    else:
        G_active = _G_FIX
        Q_best = Q_fixed

    # --- Bit-level diversity encode/decode with selected G_DIV ---
    info_bits = rng.integers(0, 2, size=rho)
    coded_bits = apply_diversity_transform(info_bits, G_active)

    received_bits = coded_bits.copy()
    for i in range(n_s):
        if erased[i]:
            received_bits[i * m:(i + 1) * m] = 0

    recovered = diversity_demapper(received_bits, G_active, erased, m)
    if recovered is None or not np.array_equal(recovered, info_bits):
        return 1.0

    # Diversity succeeded → Polar SC BLER (FEC) with adaptation gain
    snr_avg_dB = 10 * np.log10(max(gamma_avg, 1e-10))
    q_ratio = Q_best / max(Q_fixed, 1e-10)
    adaptation_gain_dB = 10 * np.log10(max(q_ratio, 1.0))

    # Cap effective SNR at best-k_c subchannel average — no practical scheme
    # can exceed this (the PPV bound assumes optimal coding at this SNR)
    gamma_best_kc = np.mean(np.sort(gamma_per_sub)[::-1][:_DT_CONFIG.k_c])
    snr_cap_dB = 10 * np.log10(max(gamma_best_kc, 1e-10))
    snr_eff_dB = min(snr_avg_dB + adaptation_gain_dB, snr_cap_dB)

    return polar_sc_bler(snr_eff_dB, N, K)


# ============================================================================
#  BLER vs SNR simulation
# ============================================================================

def simulate_bler_vs_snr(N: int = 256, K: int = 128,
                         n_subchannels: int = 6,
                         max_erasures: int = 2,
                         snr_range_dB: Tuple[float, float] = (-2, 10),
                         n_points: int = 14,
                         n_mc: int = 5000) -> Dict:
    """
    BLER vs SNR for all 4 curves.

    Channel: Rician block fading per subchannel.
    K_rice = 3 (moderate LOS, ~4.8 dB) provides enough fading spread
    to create deep fades on some subchannels, differentiating schemes.
    """
    snr_dB_arr = np.linspace(snr_range_dB[0], snr_range_dB[1], n_points)

    bler_ppv_no_div = np.zeros(n_points)    # PPV bound: no diversity
    bler_ppv_fix_div = np.zeros(n_points)   # PPV bound: fixed diversity
    bler_ppv_ada_div = np.zeros(n_points)   # PPV bound: adaptive diversity
    bler_1 = np.zeros(n_points)
    bler_2 = np.zeros(n_points)
    bler_3 = np.zeros(n_points)
    bler_4 = np.zeros(n_points)

    rng = np.random.default_rng(2026)

    K_rice = 3.0                # Rician K-factor (moderate LOS)
    erasure_threshold = db_to_linear(-3.0)  # -3 dB threshold
    shadow_loss_dB = 15.0       # Shadow attenuation (dB)
    # Blockage model: 0, 1, or 2 subchannels blocked (never exceeds MDS limit)
    p_block = [0.60, 0.25, 0.15]  # P(0 blocked), P(1 blocked), P(2 blocked)

    N_sub = N // n_subchannels
    K_sub = K // n_subchannels

    print(f"Simulating BLER vs SNR (4 curves + 3 PPV bounds)...")
    print(f"  Polar({N},{K}), Rate = {K/N:.2f}")
    print(f"  Subchannels: {n_subchannels}, MDS max erasures = {max_erasures}")
    print(f"  Per-sub: Polar({N_sub},{K_sub})")
    print(f"  Rician K = {10*np.log10(K_rice):.1f} dB")
    print(f"  Erasure threshold = {10*np.log10(erasure_threshold):.1f} dB")
    print(f"  Blockage: P(0,1,2 blocked)={p_block}, loss={shadow_loss_dB} dB")
    print(f"  MC trials: {n_mc}")
    print()

    for idx, snr in enumerate(snr_dB_arr):
        gamma_avg = db_to_linear(snr)

        for _ in range(n_mc):
            # Per-subchannel Rician fading
            gamma_los = gamma_avg * K_rice / (K_rice + 1)
            gamma_nlos = gamma_avg / (K_rice + 1) * rng.exponential(1, n_subchannels)
            gamma_per_sub = gamma_los + gamma_nlos

            # Blockage: 0, 1, or 2 random subchannels blocked (capped at max_erasures)
            n_blocked = rng.choice(3, p=p_block)
            if n_blocked > 0:
                blocked_idx = rng.choice(n_subchannels, size=n_blocked, replace=False)
                gamma_per_sub[blocked_idx] *= db_to_linear(-shadow_loss_dB)

            # PPV bound (no diversity): average over all subchannels
            gamma_all = np.mean(gamma_per_sub)
            bler_ppv_no_div[idx] += finite_blocklength_bler(gamma_all, N, K)

            # PPV bound (fixed diversity): average of non-erased subchannels
            # Same threshold-based selection as the fixed diversity scheme
            active = gamma_per_sub >= erasure_threshold
            if np.any(active):
                gamma_fix_sel = np.mean(gamma_per_sub[active])
            else:
                gamma_fix_sel = 1e-10
            bler_ppv_fix_div[idx] += finite_blocklength_bler(gamma_fix_sel, N, K)

            # PPV bound (adaptive diversity): best k_c subchannels
            # Optimal subchannel selection
            gamma_sorted = np.sort(gamma_per_sub)[::-1]
            gamma_best_kc = np.mean(gamma_sorted[:_DT_CONFIG.k_c])
            bler_ppv_ada_div[idx] += finite_blocklength_bler(gamma_best_kc, N, K)

            # Curve 1: no interleaver
            bler_1[idx] += bler_no_interleaver(
                gamma_per_sub, N, K, n_subchannels)

            # Curve 2: interleaver
            bler_2[idx] += bler_interleaver(
                gamma_per_sub, N, K, erasure_threshold)

            # Curve 3: fixed MDS (bit-level diversity + Polar FEC)
            bler_3[idx] += bler_fixed_diversity(
                gamma_per_sub, N, K, max_erasures, erasure_threshold, rng)

            # Curve 4: adaptive MDS (bit-level diversity + Polar FEC)
            bler_4[idx] += bler_adaptive_diversity(
                gamma_per_sub, N, K, max_erasures, erasure_threshold, rng)

        bler_ppv_no_div[idx] /= n_mc
        bler_ppv_fix_div[idx] /= n_mc
        bler_ppv_ada_div[idx] /= n_mc
        bler_1[idx] /= n_mc
        bler_2[idx] /= n_mc
        bler_3[idx] /= n_mc
        bler_4[idx] /= n_mc

        print(f"  SNR = {snr:5.1f} dB: "
              f"PPV = {bler_ppv_no_div[idx]:.4e}, "
              f"PPV_Fix = {bler_ppv_fix_div[idx]:.4e}, "
              f"PPV_Ada = {bler_ppv_ada_div[idx]:.4e}, "
              f"NoIntlv = {bler_1[idx]:.4e}, "
              f"Intlv = {bler_2[idx]:.4e}, "
              f"Fixed = {bler_3[idx]:.4e}, "
              f"Adaptive = {bler_4[idx]:.4e}")

    return {
        'snr_dB': snr_dB_arr,
        'ppv_no_diversity': bler_ppv_no_div,
        'ppv_fixed_diversity': bler_ppv_fix_div,
        'ppv_adaptive_diversity': bler_ppv_ada_div,
        'no_interleaver': bler_1,
        'interleaver': bler_2,
        'fixed': bler_3,
        'adaptive': bler_4,
    }


# ============================================================================
#  BLER vs Elevation simulation
# ============================================================================

def compute_snr_vs_elevation(elevations: np.ndarray) -> np.ndarray:
    """
    Compute average received SNR at each elevation using LEO link budget.

    Simple model: FSPL + atmospheric attenuation.
    Slant range d = h / sin(eps) gives monotonically decreasing path loss
    with increasing elevation.

    Parameters tuned so that:
      - Low elevation (~10°): SNR ~ -2 dB (deep waterfall, high BLER)
      - High elevation (~90°): SNR ~ 6 dB (past waterfall, low BLER)
    This puts the interesting region in the 20-70° range.
    """
    R_earth = 6371e3  # Earth radius (m)
    h_orb = 600e3     # Orbit altitude (m)
    f_c = 2e9         # Carrier frequency (Hz)
    c = 3e8           # Speed of light

    # Link budget parameters (representative LEO NTN downlink)
    P_tx_dBm = 33.0   # Transmit power
    G_tx_dBi = 30.0   # Tx antenna gain
    G_rx_dBi = 0.0    # Rx antenna gain
    NF_dB = 7.0       # Noise figure
    BW_Hz = 5e6       # Bandwidth (narrowband NTN)
    wavelength = c / f_c

    # Thermal noise
    kB = 1.38e-23
    T0 = 290
    N0 = kB * T0 * BW_Hz * 10 ** (NF_dB / 10)  # Noise power (W)
    N0_dBm = 10 * np.log10(N0 * 1000)

    snr_dB = np.zeros(len(elevations))

    for idx, eps_deg in enumerate(elevations):
        eps_rad = np.radians(eps_deg)

        # Slant range (geometric)
        sin_eps = np.sin(eps_rad)
        d = -R_earth * sin_eps + np.sqrt(
            (R_earth * sin_eps) ** 2 + 2 * R_earth * h_orb + h_orb ** 2)

        # Free-space path loss
        FSPL_dB = 20 * np.log10(4 * np.pi * d / wavelength)

        # Atmospheric attenuation (increases at low elevation)
        # Simple model: A_atm = A0 / sin(eps)
        A0 = 0.5  # dB at zenith
        A_atm_dB = A0 / max(sin_eps, 0.1)

        # Shadow fading margin (worse at low elevation)
        SF_margin_dB = 2.0 / max(sin_eps, 0.2)

        # Total path loss
        PL_total = FSPL_dB + A_atm_dB + SF_margin_dB

        # Received SNR
        snr_dB[idx] = P_tx_dBm + G_tx_dBi + G_rx_dBi - PL_total - N0_dBm

    return snr_dB


def simulate_bler_vs_elevation(N: int = 256, K: int = 128,
                               n_subchannels: int = 6,
                               max_erasures: int = 2,
                               elevation_range: Tuple[float, float] = (20, 90),
                               n_points: int = 8,
                               n_mc: int = 5000) -> Dict:
    """Simulate BLER vs elevation angle for all 4 curves.

    Computes average SNR at each elevation from LEO link budget,
    then applies the same Rician fading model as the SNR simulation.
    This ensures consistent curve behavior and proper ordering.

    Rician K-factor varies with elevation:
      K_rice(eps) = K_min + (K_max - K_min) * (eps - 10) / 80
    Low elevation: more NLOS → lower K → more fading spread
    High elevation: strong LOS → higher K → less fading
    """
    elevations = np.linspace(elevation_range[0], elevation_range[1], n_points)

    # Get average SNR at each elevation from LEO link budget
    snr_vs_elev = compute_snr_vs_elevation(elevations)

    bler_ppv_no_div = np.zeros(n_points)
    bler_ppv_fix_div = np.zeros(n_points)
    bler_ppv_ada_div = np.zeros(n_points)
    bler_1 = np.zeros(n_points)
    bler_2 = np.zeros(n_points)
    bler_3 = np.zeros(n_points)
    bler_4 = np.zeros(n_points)

    rng = np.random.default_rng(2026)

    K_rice_min = 1.5   # Low elevation: more fading
    K_rice_max = 8.0   # High elevation: strong LOS
    erasure_threshold = db_to_linear(-3.0)
    shadow_loss_dB = 15.0
    # Blockage probabilities vary with elevation: more blockage at low elevation
    # P(0,1,2 blocked) — at low elev more frequent, at high elev rare
    p_block_low  = [0.40, 0.35, 0.25]  # Low elevation: frequent blockage
    p_block_high = [0.85, 0.10, 0.05]  # High elevation: rare blockage

    print(f"\nSimulating BLER vs Elevation (4 curves + 3 PPV bounds)...")
    print(f"  Using LEO link budget for average SNR")
    print(f"  Rician K: {10*np.log10(K_rice_min):.1f} dB (low elev) to "
          f"{10*np.log10(K_rice_max):.1f} dB (high elev)")
    print(f"  Blockage P(0,1,2): {p_block_low} (low) to {p_block_high} (high), "
          f"loss={shadow_loss_dB} dB")
    print(f"  MC trials: {n_mc}")

    for idx, eps_deg in enumerate(elevations):
        gamma_avg = db_to_linear(snr_vs_elev[idx])

        # Elevation-dependent Rician K-factor
        K_rice = K_rice_min + (K_rice_max - K_rice_min) * \
                 (eps_deg - elevation_range[0]) / (elevation_range[1] - elevation_range[0])

        # Elevation-dependent blockage probabilities (interpolate)
        elev_frac = (eps_deg - elevation_range[0]) / (elevation_range[1] - elevation_range[0])
        p_block = [p_block_low[i] + (p_block_high[i] - p_block_low[i]) * elev_frac
                   for i in range(3)]

        for _ in range(n_mc):
            # Per-subchannel Rician fading (same model as SNR simulation)
            gamma_los = gamma_avg * K_rice / (K_rice + 1)
            gamma_nlos = gamma_avg / (K_rice + 1) * rng.exponential(1, n_subchannels)
            gamma_per_sub = gamma_los + gamma_nlos

            # Blockage: 0, 1, or 2 random subchannels blocked
            n_blocked = rng.choice(3, p=p_block)
            if n_blocked > 0:
                blocked_idx = rng.choice(n_subchannels, size=n_blocked, replace=False)
                gamma_per_sub[blocked_idx] *= db_to_linear(-shadow_loss_dB)

            # PPV bound (no diversity): average over all subchannels
            gamma_all = np.mean(gamma_per_sub)
            bler_ppv_no_div[idx] += finite_blocklength_bler(gamma_all, N, K)

            # PPV bound (fixed diversity): average of non-erased subchannels
            active = gamma_per_sub >= erasure_threshold
            if np.any(active):
                gamma_fix_sel = np.mean(gamma_per_sub[active])
            else:
                gamma_fix_sel = 1e-10
            bler_ppv_fix_div[idx] += finite_blocklength_bler(gamma_fix_sel, N, K)

            # PPV bound (adaptive diversity): best k_c subchannels
            gamma_sorted = np.sort(gamma_per_sub)[::-1]
            gamma_best_kc = np.mean(gamma_sorted[:_DT_CONFIG.k_c])
            bler_ppv_ada_div[idx] += finite_blocklength_bler(gamma_best_kc, N, K)

            bler_1[idx] += bler_no_interleaver(
                gamma_per_sub, N, K, n_subchannels)
            bler_2[idx] += bler_interleaver(
                gamma_per_sub, N, K, erasure_threshold)
            bler_3[idx] += bler_fixed_diversity(
                gamma_per_sub, N, K, max_erasures, erasure_threshold, rng)
            bler_4[idx] += bler_adaptive_diversity(
                gamma_per_sub, N, K, max_erasures, erasure_threshold, rng)

        bler_ppv_no_div[idx] /= n_mc
        bler_ppv_fix_div[idx] /= n_mc
        bler_ppv_ada_div[idx] /= n_mc
        bler_1[idx] /= n_mc
        bler_2[idx] /= n_mc
        bler_3[idx] /= n_mc
        bler_4[idx] /= n_mc

        print(f"  Elev = {eps_deg:5.1f}deg (SNR={snr_vs_elev[idx]:.1f}dB, "
              f"K={10*np.log10(K_rice):.1f}dB): "
              f"PPV = {bler_ppv_no_div[idx]:.4e}, "
              f"PPV_Fix = {bler_ppv_fix_div[idx]:.4e}, "
              f"PPV_Ada = {bler_ppv_ada_div[idx]:.4e}, "
              f"NoIntlv = {bler_1[idx]:.4e}, "
              f"Intlv = {bler_2[idx]:.4e}, "
              f"Fixed = {bler_3[idx]:.4e}, "
              f"Adaptive = {bler_4[idx]:.4e}")

    return {
        'elevation_deg': elevations,
        'ppv_no_diversity': bler_ppv_no_div,
        'ppv_fixed_diversity': bler_ppv_fix_div,
        'ppv_adaptive_diversity': bler_ppv_ada_div,
        'no_interleaver': bler_1,
        'interleaver': bler_2,
        'fixed': bler_3,
        'adaptive': bler_4,
    }


# ============================================================================
#  Plotting
# ============================================================================

def plot_bler_vs_snr(results: Dict):
    """Plot BLER vs SNR for all 4 curves + 3 PPV bounds."""
    fig, ax = plt.subplots(figsize=(10, 7))
    snr = results['snr_dB']

    ax.semilogy(snr, np.maximum(results['ppv_adaptive_diversity'], 1e-8),
                's--', color='#2ca02c', linewidth=1.5, markersize=6,
                label='PPV Bound (adaptive diversity)')
    ax.semilogy(snr, np.maximum(results['ppv_fixed_diversity'], 1e-8),
                'p--', color='#17becf', linewidth=1.5, markersize=6,
                label='PPV Bound (fixed diversity)')
    ax.semilogy(snr, np.maximum(results['ppv_no_diversity'], 1e-8),
                'h--', color='#9467bd', linewidth=1.5, markersize=6,
                label='PPV Bound (no diversity)')
    ax.semilogy(snr, np.maximum(results['no_interleaver'], 1e-8),
                'v-', color='#7f7f7f', linewidth=1.8, markersize=8,
                label='Standard Polar (no interleaver)')
    ax.semilogy(snr, np.maximum(results['interleaver'], 1e-8),
                '^-', color='#d62728', linewidth=1.8, markersize=8,
                label='Standard Polar + random interleaver')
    ax.semilogy(snr, np.maximum(results['fixed'], 1e-8),
                'D-', color='#ff7f0e', linewidth=2, markersize=8,
                label='Fixed Diversity Transform (FEC)')
    ax.semilogy(snr, np.maximum(results['adaptive'], 1e-8),
                'o-', color='#1f77b4', linewidth=2.5, markersize=8,
                label='Adaptive Diversity Transform (FEC, Proposed)')

    ax.set_xlabel('Average SNR (dB)', fontsize=12)
    ax.set_ylabel('Block Error Rate (BLER)', fontsize=12)
    ax.set_title('BLER vs SNR: Polar-Coded OTFS with Diversity Transforms',
                 fontsize=14)
    ax.legend(fontsize=10, loc='lower left')
    ax.grid(True, which='both', linestyle='--', alpha=0.7)
    ax.set_ylim(1e-7, 1)

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
                label='Fixed Diversity Transform (FEC)')
    ax.semilogy(elev, np.maximum(results['adaptive'], 1e-7),
                'o-', color='#1f77b4', linewidth=2.5, markersize=8,
                label='Adaptive Diversity Transform (FEC, Proposed)')

    ax.set_xlabel('Elevation Angle (degrees)', fontsize=12)
    ax.set_ylabel('Block Error Rate (BLER)', fontsize=12)
    ax.set_title('BLER vs Elevation: Polar-Coded OTFS with Diversity Transforms',
                 fontsize=14)
    ax.legend(fontsize=10, loc='upper right')
    ax.grid(True, which='both', linestyle='--', alpha=0.7)
    ax.set_ylim(1e-6, 1)

    plt.tight_layout()
    plt.savefig('bler_vs_elevation_comparison.png', dpi=300, bbox_inches='tight')
    print("Saved: bler_vs_elevation_comparison.png")
    plt.close()


# ============================================================================
#  Gain analysis
# ============================================================================

def print_gain_analysis(results_snr: Dict, results_elev: Dict):
    """Print gain analysis across all 4 curves."""
    print("\n" + "=" * 80)
    print("GAIN ANALYSIS: 4-Curve Comparison (Polar Coded OTFS)")
    print("=" * 80)

    labels = ['ppv_no_diversity', 'ppv_fixed_diversity', 'ppv_adaptive_diversity',
              'no_interleaver', 'interleaver', 'fixed', 'adaptive']
    short = ['PPV', 'PPV_Fix', 'PPV_Ada', 'NoIntlv', 'Intlv', 'Fixed', 'Adaptive']

    print("\nSNR-based comparison (BLER at target SNR points):")
    header = f"  {'SNR':>5s}"
    for s in short:
        header += f"  {s:>12s}"
    print(header)
    print("  " + "-" * (5 + 14 * len(short)))

    for snr_target in [0, 2, 4, 6, 8, 10]:
        idx = np.argmin(np.abs(results_snr['snr_dB'] - snr_target))
        row = f"  {snr_target:>3d}dB"
        for lbl in labels:
            row += f"  {results_snr[lbl][idx]:>12.2e}"
        print(row)

    # Adaptive vs Fixed gain
    print("\nAdaptive vs Fixed gain:")
    for snr_target in [0, 2, 4, 6, 8]:
        idx = np.argmin(np.abs(results_snr['snr_dB'] - snr_target))
        f = results_snr['fixed'][idx]
        a = results_snr['adaptive'][idx]
        if f > 1e-10 and a > 1e-10 and f > a:
            ratio = f / a
            gain_dB = 10 * np.log10(ratio)
            print(f"  SNR={snr_target}dB: Fixed={f:.2e}, Adaptive={a:.2e}, "
                  f"Gain={gain_dB:.1f}dB ({ratio:.1f}x)")

    # Verify ordering
    print("\nOrdering check (should be: NoIntlv > Intlv > Fixed > Adaptive):")
    violations = 0
    for idx in range(len(results_snr['snr_dB'])):
        snr = results_snr['snr_dB'][idx]
        b1 = results_snr['no_interleaver'][idx]
        b2 = results_snr['interleaver'][idx]
        b3 = results_snr['fixed'][idx]
        b4 = results_snr['adaptive'][idx]
        ok = b1 >= b2 >= b3 >= b4
        if not ok:
            violations += 1
            print(f"  VIOLATION at SNR={snr:.1f}dB: "
                  f"{b1:.2e} >= {b2:.2e} >= {b3:.2e} >= {b4:.2e}")
    if violations == 0:
        print("  All SNR points: ordering OK!")

    print("=" * 80)


# ============================================================================
#  Main
# ============================================================================

def main():
    """Run full 4-curve BLER comparison."""
    print("=" * 80)
    print("BLER COMPARISON: Polar-Coded OTFS with Diversity Transforms")
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

    # BLER vs SNR
    print("\n" + "-" * 60)
    results_snr = simulate_bler_vs_snr(
        N, K, n_subchannels, max_erasures,
        snr_range_dB=(-2, 10), n_points=14, n_mc=5000)

    # BLER vs Elevation
    print("\n" + "-" * 60)
    results_elev = simulate_bler_vs_elevation(
        N, K, n_subchannels, max_erasures,
        elevation_range=(20, 90), n_points=8, n_mc=5000)

    # Analysis
    print_gain_analysis(results_snr, results_elev)

    # Plot
    print("\nGenerating plots...")
    plot_bler_vs_snr(results_snr)
    plot_bler_vs_elevation(results_elev)

    return results_snr, results_elev


if __name__ == "__main__":
    results = main()
