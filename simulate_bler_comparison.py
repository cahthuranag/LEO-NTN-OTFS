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

Channel model: LEO NTN with OTFS DD-domain channel (DDChannel.apply_fast),
L=2 multipath Rician fading, cascaded feeder+access link SNR
(transparent relay), and per-subchannel blockage/shadowing.

Two-layer eta architecture:
  - Ephemeris eta (satellite side): elevation-based prediction via
    EphemerisPredictor -> used for Algorithm 2 T-selection
  - Instantaneous eta (receiver side): from per-subchannel SNR
    -> used for MAP LLR scaling

Signal path for diversity curves (3 & 4):
  Polar encode -> reshape -> G_DIV^T per column -> BPSK
  -> place on DD grid -> DDChannel.apply_fast() -> MMSE equalize
  -> extract per-subchannel -> eta-scaled LLRs (blockage via gamma)
  -> MAP soft diversity demapping -> LLR -> Polar SCL decode

Author: Research simulation
Date: 2026-02-21
"""

import numpy as np
import matplotlib.pyplot as plt
import csv
from typing import Dict, Tuple

from channel import (
    SNRCalculator, SPEED_OF_LIGHT, linear_to_db, db_to_linear,
    LargeScalePathLoss, AtmosphericParams, EnvironmentParams,
    OrbitalParams, LinkBudgetParams, OTFSParams, TDLProfile,
    DDChannel, EphemerisPredictor, SmallScaleFading,
)
from itertools import combinations
from fbl_analysis import (
    finite_blocklength_bler, conditional_bler, capacity_awgn,
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

# Atmospheric / environment models (ITU-R P.676, P.840, P.618; 3GPP NTN)
_ATM_PARAMS = AtmosphericParams(
    rho_wv=7.5,           # water-vapour density [g/m³]
    H_gas_m=6000.0,       # equivalent gaseous thickness [m]
    LWC=0.05,             # liquid-water content [g/m³]
    H_cf_m=1000.0,        # cloud/fog layer thickness [m]
    rain_rate_mmh=2.0,    # light rain [mm/h]
    H_rain_m=3000.0,      # rain height [m]
    kappa=0.0101,         # ITU rain atten. coeff (S-band)
    beta=1.276,           # ITU rain atten. exponent
)
_ENV_PARAMS = EnvironmentParams(env_type="suburban")
_PATH_LOSS_MODEL = LargeScalePathLoss(_ATM_PARAMS, _ENV_PARAMS)

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

# ---- OTFS DD-domain grid parameters ----
_OTFS_PARAMS = OTFSParams(N_D=16, M_D=64, delta_f_Hz=15e3)
_M_SUB = 10  # delay bins per subchannel (use first 60 of 64 delay bins)


def _build_subchannel_map(N_D=_OTFS_PARAMS.N_D, M_D=_OTFS_PARAMS.M_D,
                          n_s=_n_s, m_sub=_M_SUB):
    """
    Build mapping from subchannel index to DD grid positions.

    Subchannel j occupies delay bins [j*m_sub, (j+1)*m_sub) across all
    Doppler bins [0, N_D). Returns dict: j -> list of (doppler, delay) tuples.
    """
    sc_map = {}
    for j in range(n_s):
        positions = []
        for d in range(N_D):
            for l in range(j * m_sub, (j + 1) * m_sub):
                positions.append((d, l))
        sc_map[j] = positions
    return sc_map


def _map_bpsk_to_dd(bpsk_per_sub, subchannel_map, N_D, M_D):
    """
    Place BPSK symbols onto (N_D, M_D) complex DD grid.

    Args:
        bpsk_per_sub: dict j -> (n_symbols_j,) complex BPSK symbols
        subchannel_map: dict j -> list of (doppler, delay) index pairs
        N_D, M_D: DD grid dimensions

    Returns:
        x_dd: (N_D, M_D) complex DD grid
    """
    x_dd = np.zeros((N_D, M_D), dtype=complex)
    for j, symbols in bpsk_per_sub.items():
        positions = subchannel_map[j]
        n_sym = len(symbols)
        for idx in range(min(n_sym, len(positions))):
            d, l = positions[idx]
            x_dd[d, l] = symbols[idx]
    return x_dd


def _extract_from_dd(y_dd, subchannel_map, n_symbols_per_sub=None):
    """
    Extract per-subchannel received symbols from DD grid.

    Args:
        y_dd: (N_D, M_D) complex DD grid
        subchannel_map: dict j -> list of (doppler, delay) index pairs
        n_symbols_per_sub: dict j -> number of symbols to extract (None = all)

    Returns:
        dict j -> (n_symbols_j,) complex received symbols
    """
    result = {}
    for j, positions in subchannel_map.items():
        n = len(positions) if n_symbols_per_sub is None else n_symbols_per_sub.get(j, len(positions))
        syms = np.zeros(n, dtype=complex)
        for idx in range(n):
            d, l = positions[idx]
            syms[idx] = y_dd[d, l]
        result[j] = syms
    return result


def _qpsk_modulate(bits):
    """Gray-coded QPSK: pairs of bits -> unit-energy complex symbols."""
    bits = np.asarray(bits, dtype=np.int8).ravel()
    if len(bits) % 2 != 0:
        bits = np.concatenate([bits, np.zeros(1, dtype=np.int8)])
    I = 1.0 - 2.0 * bits[0::2].astype(np.float64)
    Q = 1.0 - 2.0 * bits[1::2].astype(np.float64)
    return (I + 1j * Q) / np.sqrt(2.0)


def _qpsk_soft_demod(y, snr):
    """QPSK soft demapper: received symbols + per-symbol SNR -> per-bit LLRs."""
    snr = np.asarray(snr, dtype=np.float64)
    scale = 2.0 * np.sqrt(2.0) * snr
    llr_I = scale * np.real(y)
    llr_Q = scale * np.imag(y)
    n_sym = len(y)
    llr = np.zeros(2 * n_sym)
    llr[0::2] = llr_I
    llr[1::2] = llr_Q
    return llr


def _extract_snr_from_dd(snr_post, subchannel_map, n_symbols_per_sub=None):
    """Extract per-position post-MMSE SNR for each subchannel."""
    result = {}
    for j, positions in subchannel_map.items():
        n = (len(positions) if n_symbols_per_sub is None
             else n_symbols_per_sub.get(j, len(positions)))
        snrs = np.zeros(n)
        for idx in range(n):
            d, l = positions[idx]
            snrs[idx] = np.real(snr_post[d, l])
        result[j] = snrs
    return result


# Pre-build the subchannel map (used throughout simulation)
_SUBCHANNEL_MAP = _build_subchannel_map()


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


def _atmospheric_loss_dB(elevation_deg):
    """
    Aggregate atmospheric attenuation for the access link (Eq. 7).

    Includes gaseous absorption (ITU-R P.676), cloud/fog (ITU-R P.840),
    and rain (ITU-R P.618) along the slant path at the given elevation.
    """
    epsilon_rad = np.radians(elevation_deg)
    return _PATH_LOSS_MODEL.atmospheric_loss_dB(_F_ACCESS_HZ, epsilon_rad)


def _power_to_link_snrs(P_total_dBW, access_fspl_dB, atm_loss_dB=0.0):
    """
    Convert total transmit power to per-link SNRs via link budgets.

    Equal power split: P_SAT = α · P_total, P_GW = (1-α) · P_total.
    Access link loss includes FSPL + atmospheric attenuation (Eq. 6).

    Returns (gamma_gs_linear, gamma_su_linear).
    """
    P_total_lin = db_to_linear(P_total_dBW)
    P_SAT_lin = _ALPHA_POWER * P_total_lin
    P_GW_lin = (1.0 - _ALPHA_POWER) * P_total_lin

    gamma_gs_dB = linear_to_db(P_GW_lin) + _FEEDER_GAIN_DB
    gamma_su_dB = (linear_to_db(P_SAT_lin) + _ACCESS_GAIN_EXCL_FSPL_DB
                   - access_fspl_dB - atm_loss_dB)

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
#  OTFS DD-domain channel generation and signal path
# ============================================================================

def _generate_dd_channel(gamma_avg_SU, gamma_gs, n_subchannels, K_rice,
                         p_block_sub, shadow_loss_dB, rng):
    """
    Generate DD-domain channel for OTFS simulation.

    Creates an L=2 tap channel kernel with LOS (Rician) + NLOS (Rayleigh) taps,
    applies per-subchannel blockage, and computes effective per-subchannel SNR.

    Returns:
        h_dd: (N_D, M_D) DD-domain channel kernel
        noise_var: scalar noise variance for DD-domain AWGN
        gamma_e2e_per_sub: (n_subchannels,) effective E2E SNR per subchannel
        blocked_mask: (n_subchannels,) bool, True = blocked
    """
    N_D = _OTFS_PARAMS.N_D
    M_D = _OTFS_PARAMS.M_D

    # Generate L=2 tap complex gains: LOS (Rician) + NLOS (Rayleigh)
    tap_gains = np.zeros(_L_TAPS, dtype=complex)
    for p in range(_L_TAPS):
        if p == 0:
            # LOS tap: Rician fading
            los_mean = np.sqrt(K_rice / (K_rice + 1))
            nlos_std = np.sqrt(1.0 / (2.0 * (K_rice + 1)))
            h_real = los_mean + rng.normal(0, nlos_std)
            h_imag = rng.normal(0, nlos_std)
            tap_gains[p] = (h_real + 1j * h_imag) * np.sqrt(_TAP_POWERS_LIN[p])
        else:
            # NLOS taps: Rayleigh fading
            h = (rng.normal(0, 1) + 1j * rng.normal(0, 1)) / np.sqrt(2)
            tap_gains[p] = h * np.sqrt(_TAP_POWERS_LIN[p])

    # Assign delay and Doppler bins: tap 0 at (0,0), tap 1 at (1,1)
    tap_delays_bins = np.array([0, 1])
    tap_dopplers_bins = np.array([0, 1])

    # Build DD kernel
    h_dd = DDChannel.build_dd_kernel(
        tap_gains, tap_delays_bins, tap_dopplers_bins, N_D, M_D)

    # Noise variance (complex): sigma^2_w for CN(0, noise_var) per DD bin
    noise_var = 1.0 / (2.0 * max(gamma_avg_SU, 1e-10))

    # DFT-domain channel for per-subchannel SNR computation
    H_freq = np.fft.fft2(h_dd)

    # Per-subchannel effective SNR: average |H|^2 over subchannel bins
    gamma_mrc = np.zeros(n_subchannels)
    for j in range(n_subchannels):
        positions = _SUBCHANNEL_MAP[j]
        h_vals = np.array([H_freq[d, l] for d, l in positions])
        gamma_mrc[j] = gamma_avg_SU * np.mean(np.abs(h_vals)**2)

    # Per-subchannel blockage (applied to gamma, not DD signal)
    blocked_mask = rng.random(n_subchannels) < p_block_sub
    if np.any(blocked_mask):
        shadow_linear = db_to_linear(-shadow_loss_dB)
        gamma_mrc[blocked_mask] *= shadow_linear

    # Cascaded SNR
    gamma_e2e = SNRCalculator.cascaded_snr(gamma_gs, gamma_mrc)

    return h_dd, noise_var, gamma_e2e, blocked_mask


def _mmse_equalize_dd(y_dd, h_dd, noise_var):
    """
    MMSE equalization in DFT domain for DD-domain OTFS signals.

    In the DFT domain, the DD circular convolution becomes point-wise
    multiplication: Y[n,m] = H[n,m] * X[n,m] + W[n,m].
    MMSE equalization: X_hat[n,m] = H*[n,m] * Y[n,m] / (|H[n,m]|^2 + noise_var)

    Returns:
        x_hat_dd: (N_D, M_D) equalized DD-domain signal
        snr_post: (N_D, M_D) post-MMSE SNR per DFT bin
    """
    H_freq = np.fft.fft2(h_dd)
    Y_freq = np.fft.fft2(y_dd)

    # MMSE filter
    H_sq = np.abs(H_freq)**2
    X_hat_freq = np.conj(H_freq) * Y_freq / (H_sq + noise_var)

    # Post-MMSE SNR per bin
    snr_post = H_sq / noise_var

    # Back to DD domain
    x_hat_dd = np.fft.ifft2(X_hat_freq)

    return x_hat_dd, snr_post


def _diversity_transform_transmit_receive_otfs(coded_bits, G_DIV,
                                                h_dd, noise_var,
                                                gamma_e2e_per_sub,
                                                erased_mask, eta, rng):
    """
    OTFS diversity transform with QPSK and post-MMSE SNR (paper-aligned).

    Signal path:
      coded bits -> diversity transform -> QPSK mod -> DD grid
      -> DDChannel (access-link noise only, Eq. 27) -> MMSE equalize
      -> QPSK soft demod (post-MMSE SNR) -> eta-scaled LLRs -> MAP demapper
    """
    N = len(coded_bits)
    N_D = _OTFS_PARAMS.N_D
    M_D = _OTFS_PARAMS.M_D

    n_pos = int(np.ceil(N / _rho))
    N_pad = _rho * n_pos
    coded_padded = np.zeros(N_pad, dtype=np.int8)
    coded_padded[:N] = coded_bits

    X = coded_padded.reshape(_rho, n_pos)
    Z = np.mod(G_DIV.T @ X, 2).astype(np.int8)

    # QPSK modulation per subchannel
    qam_per_sub = {}
    n_sym_per_sub = {}
    for j in range(_n_s):
        sub_bits = Z[j * _m:(j + 1) * _m, :].flatten()
        qpsk_syms = _qpsk_modulate(sub_bits)
        qam_per_sub[j] = qpsk_syms
        n_sym_per_sub[j] = len(qpsk_syms)

    x_dd = _map_bpsk_to_dd(qam_per_sub, _SUBCHANNEL_MAP, N_D, M_D)
    y_dd = DDChannel.apply_fast(x_dd, h_dd, noise_var, rng)
    x_hat_dd, snr_post = _mmse_equalize_dd(y_dd, h_dd, noise_var)
    rx_per_sub = _extract_from_dd(x_hat_dd, _SUBCHANNEL_MAP, n_sym_per_sub)
    snr_per_sub = _extract_snr_from_dd(snr_post, _SUBCHANNEL_MAP, n_sym_per_sub)

    if np.all(erased_mask):
        return None

    # QPSK soft demod + eta-scaling (no extra noise injection)
    llr_z = np.zeros((_n_out, n_pos))
    for j in range(_n_s):
        eta_j = eta[j]
        if erased_mask[j]:
            llr_z[j * _m:(j + 1) * _m, :] = 0.0
        else:
            llr_bits = _qpsk_soft_demod(rx_per_sub[j], snr_per_sub[j])
            llr_bits *= eta_j
            llr_z[j * _m:(j + 1) * _m, :] = llr_bits.reshape(_m, n_pos)

    # --- MAP soft diversity demapping ---
    is_fixed = (G_DIV is _G_FIX) or np.array_equal(G_DIV, _G_FIX)
    if is_fixed:
        S_ALL = _S_ALL_FIX
    else:
        Z_ALL = np.mod(G_DIV.T.astype(int) @ _X_ALL.astype(int), 2)
        S_ALL = 1.0 - 2.0 * Z_ALL.astype(np.float64)

    corr = (S_ALL.T @ llr_z) * 0.5
    X_llr = np.zeros((_rho, n_pos))
    for i in range(_rho):
        X_llr[i, :] = (np.max(corr[_X_MASK_0[i], :], axis=0)
                        - np.max(corr[_X_MASK_1[i], :], axis=0))

    llr_vector = X_llr.flatten()[:N]
    return llr_vector


def _standard_polar_otfs(info_bits, polar_code, h_dd, noise_var,
                         gamma_e2e_per_sub, n_subchannels, perm, rng):
    """Standard Polar over OTFS DD channel with QPSK and post-MMSE SNR."""
    N = polar_code.N
    N_D = _OTFS_PARAMS.N_D
    M_D = _OTFS_PARAMS.M_D

    coded = polar_code.encode(info_bits)

    # Bit-level permutation before modulation
    if perm is not None:
        coded_perm = coded[perm]
    else:
        coded_perm = coded

    # QPSK modulation: N bits -> N/2 complex symbols
    qpsk_syms = _qpsk_modulate(coded_perm)
    n_total_syms = len(qpsk_syms)

    # Distribute across subchannels
    syms_per_sub = n_total_syms // n_subchannels
    qam_per_sub = {}
    n_sym_per_sub = {}
    for s in range(n_subchannels):
        start_s = s * syms_per_sub
        end_s = min((s + 1) * syms_per_sub, n_total_syms)
        qam_per_sub[s] = qpsk_syms[start_s:end_s]
        n_sym_per_sub[s] = end_s - start_s
    if syms_per_sub * n_subchannels < n_total_syms:
        extra = qpsk_syms[syms_per_sub * n_subchannels:]
        qam_per_sub[n_subchannels - 1] = np.concatenate(
            [qam_per_sub[n_subchannels - 1], extra])
        n_sym_per_sub[n_subchannels - 1] += len(extra)

    # DD grid -> channel -> MMSE
    x_dd = _map_bpsk_to_dd(qam_per_sub, _SUBCHANNEL_MAP, N_D, M_D)
    y_dd = DDChannel.apply_fast(x_dd, h_dd, noise_var, rng)
    x_hat_dd, snr_post = _mmse_equalize_dd(y_dd, h_dd, noise_var)

    rx_per_sub = _extract_from_dd(x_hat_dd, _SUBCHANNEL_MAP, n_sym_per_sub)
    snr_per_sub = _extract_snr_from_dd(snr_post, _SUBCHANNEL_MAP, n_sym_per_sub)

    # QPSK soft demod using post-MMSE SNR (no double noise)
    llr_all = np.zeros(N)
    bit_offset = 0
    for s in range(n_subchannels):
        n_syms = n_sym_per_sub[s]
        llr_bits = _qpsk_soft_demod(rx_per_sub[s][:n_syms],
                                    snr_per_sub[s][:n_syms])
        n_bits = min(2 * n_syms, N - bit_offset)
        llr_all[bit_offset:bit_offset + n_bits] = llr_bits[:n_bits]
        bit_offset += n_bits

    # Inverse permutation
    if perm is not None:
        inv_perm = np.argsort(perm)
        llr_all = llr_all[inv_perm]

    return llr_all


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

    # Per-subchannel QPSK AWGN channel with eta-scaled LLRs (Eq. 38-39)
    llr_z = np.zeros((_n_out, n_pos))
    for j in range(_n_s):
        gamma_j = gamma_e2e_per_sub[j]
        eta_j = eta[j]
        if erased_mask[j]:
            llr_z[j * _m:(j + 1) * _m, :] = 0.0
        else:
            # QPSK: flatten bits, modulate, add complex AWGN, soft demod
            sub_bits = Z[j * _m:(j + 1) * _m, :].flatten()
            qpsk_syms = _qpsk_modulate(sub_bits)
            noise_std = np.sqrt(0.5 / max(gamma_j, 1e-10))
            noise = noise_std * (rng.normal(0, 1, len(qpsk_syms))
                                 + 1j * rng.normal(0, 1, len(qpsk_syms)))
            y = qpsk_syms + noise
            llr_bits = _qpsk_soft_demod(y, gamma_j)
            llr_bits *= eta_j
            llr_z[j * _m:(j + 1) * _m, :] = llr_bits.reshape(_m, n_pos)

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
#  EphemerisPredictor factory
# ============================================================================

def _create_ephemeris_predictor(elevation_deg, P_total_dBW, K_rice_dB=3.0):
    """
    Create an EphemerisPredictor from simulation scalar constants.

    Args:
        elevation_deg: reference elevation angle
        P_total_dBW: total transmit power
        K_rice_dB: Rician K-factor in dB

    Returns:
        EphemerisPredictor instance
    """
    orbital = OrbitalParams(h_orb=_H_ORB_M)

    feeder = LinkBudgetParams(
        P_tx_dBm=10.0 * np.log10(_ALPHA_POWER * db_to_linear(P_total_dBW)) + 30.0,
        G_tx_dBi=_G_TX_GW_DBI, G_rx_dBi=_G_RX_SAT_DBI,
        f_c_Hz=_F_FEEDER_HZ, noise_figure_dB=_NF_SAT_DB,
        bandwidth_Hz=_BW_HZ, T_sys_K=_T0,
    )
    access = LinkBudgetParams(
        P_tx_dBm=10.0 * np.log10((1.0 - _ALPHA_POWER) * db_to_linear(P_total_dBW)) + 30.0,
        G_tx_dBi=_G_TX_SAT_DBI, G_rx_dBi=_G_RX_UE_DBI,
        f_c_Hz=_F_ACCESS_HZ, noise_figure_dB=_NF_UE_DB,
        bandwidth_Hz=_BW_HZ, T_sys_K=_T0,
    )
    tdl = TDLProfile(
        num_taps=_L_TAPS,
        relative_powers_dB=_TAP_POWERS_DB.tolist(),
        delays_ns=[0, 100],
        K_factor_dB=K_rice_dB,
    )
    return EphemerisPredictor(
        orbital, feeder, access, _ATM_PARAMS, _ENV_PARAMS,
        _OTFS_PARAMS, tdl,
    )


# ============================================================================
#  Monte Carlo trial
# ============================================================================

def _mc_trial_one_snr(gamma_avg_SU, gamma_gs, N, K, n_subchannels,
                       max_erasures, K_rice, erasure_threshold, p_block_sub,
                       shadow_loss_dB, polar_code, rng,
                       use_otfs=False, eta_ephemeris=None):
    """
    Run one MC trial for all 7 curves at given per-link SNRs.

    Channel: L-tap multipath Rician fading per subchannel with MRC combining,
    independent per-subchannel blockage, and cascaded feeder+access SNR.
    When use_otfs=True, uses DD-domain channel via DDChannel.apply_fast().

    Two-layer eta architecture (Concern 3):
      - eta_ephemeris: elevation-based prediction for Algorithm 2 T-selection
        (satellite side, pre-transmission)
      - eta (instantaneous): from per-subchannel SNR for MAP LLR scaling
        (receiver side, post-reception)

    Returns dict of bool (True = block error) for the 4 practical curves
    and float BLER for the 3 PPV bounds.
    """
    if use_otfs:
        # Generate DD-domain channel
        h_dd, noise_var, gamma_e2e, blocked_mask = _generate_dd_channel(
            gamma_avg_SU, gamma_gs, n_subchannels, K_rice,
            p_block_sub, shadow_loss_dB, rng)
    else:
        # Legacy AWGN-equivalent channel
        gamma_e2e, blocked_mask = _generate_multipath_per_subchannel(
            gamma_avg_SU, gamma_gs, n_subchannels, K_rice, p_block_sub,
            shadow_loss_dB, rng)

    # ---- Erasure mask (shared across PPV bounds and diversity MC curves) ----
    erased_mask = gamma_e2e < erasure_threshold
    n_erased = int(np.sum(erased_mask))
    k_c = _DT_CONFIG.k_c

    # ---- Instantaneous reliability metric η (paper Eq. 34-35) ----
    # η_i = 1 - ε_i where ε_i = conditional_bler(γ_i, N_sub, R)
    # Used for MAP LLR scaling (receiver side)
    R = K / N
    N_sub = N // n_subchannels
    eta_inst = np.zeros(n_subchannels)
    for j in range(n_subchannels):
        if erased_mask[j]:
            eta_inst[j] = 0.0
        else:
            eps_j = conditional_bler(gamma_e2e[j], N_sub, R)
            eta_inst[j] = 1.0 - eps_j

    # ---- Algorithm 2 T-selection: use ephemeris eta if available ----
    # Ephemeris eta is computed pre-transmission at the satellite side
    # and is uniform across subchannels (same elevation for all).
    # Instantaneous eta is used for MAP scaling at the receiver.
    eta_for_T = eta_ephemeris if eta_ephemeris is not None else eta_inst

    best_Q = np.dot(_G_FIX_block_weights, eta_for_T)
    best_G_DIV = _G_FIX
    best_block_weights = _G_FIX_block_weights.copy()
    for i in range(_N_POOL):
        Q = np.dot(_block_weights_pool[i], eta_for_T)
        if Q > best_Q:
            best_Q = Q
            best_G_DIV = _G_DIV_POOL[i]
            best_block_weights = _block_weights_pool[i].copy()

    # ---- Semi-analytical FBL bounds (Section III) ----
    ppv_no_div = bler_no_diversity(gamma_e2e, erased_mask, k_c, N, K)
    ppv_fix_div = bler_fixed_diversity(gamma_e2e, erased_mask, k_c, N, K)

    # PPV adaptive: Q-metric scaling using ephemeris eta for Q_ratio
    Q_fix_val = float(np.dot(_G_FIX_block_weights, eta_for_T))
    Q_ratio = float(best_Q) / max(Q_fix_val, 1e-10)
    Q_ratio = max(Q_ratio, 1.0)
    ppv_ada_div = ppv_fix_div / Q_ratio

    # ---- Generate info bits (shared across practical curves) ----
    info_bits = rng.integers(0, 2, size=K).astype(np.int8)

    if use_otfs:
        # ---- OTFS signal path ----

        # Curve 1: No interleaver through OTFS DD channel
        llr_1 = _standard_polar_otfs(
            info_bits, polar_code, h_dd, noise_var,
            gamma_e2e, n_subchannels, perm=None, rng=rng)
        err_1 = polar_code.decode_check(info_bits, llr_1)

        # Curve 2: With random interleaver through OTFS DD channel
        perm = rng.permutation(N)
        llr_2 = _standard_polar_otfs(
            info_bits, polar_code, h_dd, noise_var,
            gamma_e2e, n_subchannels, perm=perm, rng=rng)
        err_2 = polar_code.decode_check(info_bits, llr_2)

        # Curve 3: Fixed diversity transform through OTFS
        if n_erased > max_erasures:
            err_3 = True
        else:
            coded_bits = polar_code.encode(info_bits)
            llr_3 = _diversity_transform_transmit_receive_otfs(
                coded_bits, _G_FIX, h_dd, noise_var,
                gamma_e2e, erased_mask, eta_inst, rng)
            if llr_3 is None:
                err_3 = True
            else:
                decoded_3 = polar_code.decode(llr_3)
                err_3 = not np.array_equal(decoded_3, info_bits)

        # Curve 4: Adaptive diversity transform through OTFS
        if n_erased > max_erasures:
            err_4 = True
        else:
            # Erasure-aware power allocation
            n_surviving = n_subchannels - n_erased
            power_alloc = np.ones(n_subchannels)
            if n_surviving > 0 and n_erased > 0:
                power_alloc[~erased_mask] = float(n_subchannels) / n_surviving
                power_alloc[erased_mask] = 0.0
            gamma_e2e_adaptive = gamma_e2e * power_alloc

            coded_bits_4 = polar_code.encode(info_bits)
            llr_4 = _diversity_transform_transmit_receive_otfs(
                coded_bits_4, best_G_DIV, h_dd, noise_var,
                gamma_e2e_adaptive, erased_mask, eta_inst, rng)
            if llr_4 is None:
                err_4 = True
            else:
                decoded_4 = polar_code.decode(llr_4)
                err_4 = not np.array_equal(decoded_4, info_bits)

    else:
        # ---- Legacy AWGN-equivalent signal path ----

        # Curve 1: No interleaver (sequential subchannel mapping)
        # QPSK: Eb/N0 = Es/(2*N0) = gamma_symbol / 2
        bits_per_sub = N // n_subchannels
        snr_per_bit = np.zeros(N)
        for s in range(n_subchannels):
            start = s * bits_per_sub
            end = min((s + 1) * bits_per_sub, N)
            snr_per_bit[start:end] = gamma_e2e[s] / 2.0
        if bits_per_sub * n_subchannels < N:
            snr_per_bit[bits_per_sub * n_subchannels:] = gamma_e2e[-1] / 2.0
        llr_1 = polar_code.transmit_per_bit_snr(info_bits, snr_per_bit, rng)
        err_1 = polar_code.decode_check(info_bits, llr_1)

        # Curve 2: With random interleaver
        perm = rng.permutation(N)
        snr_per_bit_intlv = np.zeros(N)
        for i in range(N):
            sub_idx = min(perm[i] // bits_per_sub, n_subchannels - 1)
            snr_per_bit_intlv[i] = gamma_e2e[sub_idx] / 2.0
        llr_2 = polar_code.transmit_per_bit_snr(info_bits, snr_per_bit_intlv, rng)
        err_2 = polar_code.decode_check(info_bits, llr_2)

        # Curve 3: Fixed diversity transform (G_FIX in signal path)
        if n_erased > max_erasures:
            err_3 = True
        else:
            coded_bits = polar_code.encode(info_bits)
            llr_3 = _diversity_transform_transmit_receive(
                coded_bits, _G_FIX, gamma_e2e, erased_mask, eta_inst, rng)
            if llr_3 is None:
                err_3 = True
            else:
                decoded_3 = polar_code.decode(llr_3)
                err_3 = not np.array_equal(decoded_3, info_bits)

        # Curve 4: Adaptive diversity transform
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
                coded_bits_4, best_G_DIV, gamma_e2e_adaptive, erased_mask,
                eta_inst, rng)
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

    Uses OTFS DD-domain channel (DDChannel.apply_fast) and two-layer eta:
    ephemeris eta for Algorithm 2 T-selection, instantaneous eta for MAP scaling.
    """
    rng = np.random.default_rng(2026)
    power_dBW_arr = np.linspace(power_range_dBW[0], power_range_dBW[1], n_points)

    # Access link geometry at reference elevation
    ref_slant = _slant_range_m(ref_elevation_deg)
    ref_fspl = _access_fspl_dB(ref_slant)
    ref_atm_loss = _atmospheric_loss_dB(ref_elevation_deg)

    K_rice = 2.0                  # Rician K-factor (3 dB, NTN suburban)
    K_rice_dB = 10.0 * np.log10(K_rice)
    erasure_threshold = db_to_linear(-3.0)
    shadow_loss_dB = 20.0
    p_block_sub = 0.15            # Independent per-subchannel blockage probability

    polar_code = PolarCode(N, K, design_snr_dB=1.0, list_size=8)

    print(f"Computing BLER vs P_total (7 curves, OTFS DD-domain MC simulation)...")
    print(f"  Polar({N},{K}), Rate = {K/N:.2f}")
    print(f"  OTFS grid: {_OTFS_PARAMS.N_D}x{_OTFS_PARAMS.M_D}, "
          f"DD channel with L={_L_TAPS} taps")
    print(f"  Subchannels: {n_subchannels}, MDS max erasures = {max_erasures}")
    print(f"  Power split: equal (alpha={_ALPHA_POWER})")
    print(f"  Feeder link: Ka-band {_F_FEEDER_HZ/1e9:.0f} GHz, "
          f"G_tx={_G_TX_GW_DBI:.0f} dBi, G_rx={_G_RX_SAT_DBI:.0f} dBi, "
          f"d={_D_FEEDER_M/1e3:.0f} km")
    print(f"  Access link: S-band {_F_ACCESS_HZ/1e9:.0f} GHz, "
          f"G_tx={_G_TX_SAT_DBI:.0f} dBi, G_rx={_G_RX_UE_DBI:.0f} dBi, "
          f"elev={ref_elevation_deg:.0f}deg (d={ref_slant/1e3:.0f} km)")
    print(f"  Atmospheric loss (Eq. 7): {ref_atm_loss:.2f} dB "
          f"(gas+cloud/fog+rain at {ref_elevation_deg:.0f}deg)")
    print(f"  Multipath: L={_L_TAPS} taps, Rician K={K_rice_dB:.1f} dB")
    print(f"  Blockage: p_block={p_block_sub:.2f} per-subchannel, "
          f"loss={shadow_loss_dB} dB")
    print(f"  Ephemeris eta: elevation-based T-selection (satellite side)")
    print(f"  MC trials: {n_mc}")
    print()

    results = {key: np.zeros(n_points) for key in [
        'ppv_no_diversity', 'ppv_fixed_diversity', 'ppv_adaptive_diversity',
        'no_interleaver', 'interleaver', 'fixed', 'adaptive']}
    results['P_total_dBW'] = power_dBW_arr

    R = K / N

    for idx, p_dBW in enumerate(power_dBW_arr):
        gamma_gs, gamma_su = _power_to_link_snrs(p_dBW, ref_fspl, ref_atm_loss)

        # Compute ephemeris eta (pre-transmission, satellite side)
        # Uniform across subchannels since all share same elevation
        predictor = _create_ephemeris_predictor(
            ref_elevation_deg, p_dBW, K_rice_dB)
        bler_eph = predictor.predict_subchannel_bler(
            ref_elevation_deg, R)
        eta_eph_scalar = max(1.0 - bler_eph, 0.0)
        eta_ephemeris = np.full(n_subchannels, eta_eph_scalar)

        ppv_no_acc = 0.0
        ppv_fix_acc = 0.0
        ppv_ada_acc = 0.0
        err_counts = [0, 0, 0, 0]

        for trial in range(n_mc):
            res = _mc_trial_one_snr(
                gamma_su, gamma_gs, N, K, n_subchannels, max_erasures,
                K_rice, erasure_threshold, p_block_sub,
                shadow_loss_dB, polar_code, rng,
                use_otfs=True, eta_ephemeris=eta_ephemeris)

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
              f"(gamma_SU={gamma_su_dB:.1f}dB, gamma_GS={gamma_gs_dB:.1f}dB, "
              f"eta_eph={eta_eph_scalar:.3f}): "
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
    BLER vs elevation angle for all 7 curves (OTFS DD-domain MC simulation).

    Uses OTFS DD-domain channel and two-layer eta architecture.
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

    print(f"\nComputing BLER vs Elevation (7 curves, OTFS DD-domain MC simulation)...")
    print(f"  P_total = {P_total_dBW:.1f} dBW ({10**(P_total_dBW/10):.2f} W), "
          f"alpha={_ALPHA_POWER}")
    print(f"  OTFS grid: {_OTFS_PARAMS.N_D}x{_OTFS_PARAMS.M_D}, "
          f"DD channel with L={_L_TAPS} taps")
    print(f"  Atmospheric loss (Eq. 7): gas + cloud/fog + rain, "
          f"elev-dependent slant path")
    print(f"  Rician K: {10*np.log10(K_rice_min):.1f} dB (low elev) to "
          f"{10*np.log10(K_rice_max):.1f} dB (high elev)")
    print(f"  Blockage: p_block={p_block_sub_low:.2f} (low elev) to "
          f"{p_block_sub_high:.2f} (high elev), loss={shadow_loss_dB} dB")
    print(f"  Ephemeris eta: elevation-based T-selection (satellite side)")
    print(f"  MC trials: {n_mc}")

    results = {key: np.zeros(n_points) for key in [
        'ppv_no_diversity', 'ppv_fixed_diversity', 'ppv_adaptive_diversity',
        'no_interleaver', 'interleaver', 'fixed', 'adaptive']}
    results['elevation_deg'] = elevations
    results['P_total_dBW'] = P_total_dBW

    R = K / N

    for idx, eps_deg in enumerate(elevations):
        # Elevation-dependent access link geometry + atmospheric loss (Eq. 6-7)
        d_access = _slant_range_m(eps_deg)
        fspl_access = _access_fspl_dB(d_access)
        atm_loss = _atmospheric_loss_dB(eps_deg)
        gamma_gs, gamma_su = _power_to_link_snrs(P_total_dBW, fspl_access,
                                                   atm_loss)
        gamma_su_dB = linear_to_db(gamma_su)

        elev_frac = (eps_deg - elevation_range[0]) / (elevation_range[1] - elevation_range[0])
        K_rice = K_rice_min + (K_rice_max - K_rice_min) * elev_frac
        K_rice_dB = 10.0 * np.log10(K_rice)
        p_block_sub = p_block_sub_low + (p_block_sub_high - p_block_sub_low) * elev_frac

        # Compute ephemeris eta for this elevation
        predictor = _create_ephemeris_predictor(
            eps_deg, P_total_dBW, K_rice_dB)
        bler_eph = predictor.predict_subchannel_bler(eps_deg, R)
        eta_eph_scalar = max(1.0 - bler_eph, 0.0)
        eta_ephemeris = np.full(n_subchannels, eta_eph_scalar)

        ppv_no_acc = 0.0
        ppv_fix_acc = 0.0
        ppv_ada_acc = 0.0
        err_counts = [0, 0, 0, 0]

        for trial in range(n_mc):
            res = _mc_trial_one_snr(
                gamma_su, gamma_gs, N, K, n_subchannels, max_erasures,
                K_rice, erasure_threshold, p_block_sub,
                shadow_loss_dB, polar_code, rng,
                use_otfs=True, eta_ephemeris=eta_ephemeris)

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
              f"atm={atm_loss:.1f}dB, "
              f"gamma_SU={gamma_su_dB:.1f}dB, K={K_rice_dB:.1f}dB, "
              f"p_blk={p_block_sub:.2f}, eta_eph={eta_eph_scalar:.3f}): "
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
#  Function 5: Throughput-Reliability Trade-off
# ============================================================================

def simulate_throughput_reliability(results_power: Dict, N: int = 256,
                                    K: int = 128) -> Dict:
    """
    Compute effective throughput = R * (1 - BLER) from pre-computed BLER vs power.

    Args:
        results_power: output from simulate_bler_vs_power()
        N, K: code parameters (for rate and capacity reference)

    Returns:
        dict with throughput arrays for each curve and capacity reference
    """
    R = K / N
    p_arr = results_power['P_total_dBW']
    n_pts = len(p_arr)

    curve_keys = ['no_interleaver', 'interleaver', 'fixed', 'adaptive']
    ppv_keys = ['ppv_no_diversity', 'ppv_fixed_diversity', 'ppv_adaptive_diversity']

    throughput = {'P_total_dBW': p_arr}
    for key in curve_keys + ppv_keys:
        throughput[key] = R * (1.0 - np.clip(results_power[key], 0, 1))

    # Capacity reference: C(gamma_e2e_avg) at each power point
    ref_slant = _slant_range_m(50.0)
    ref_fspl = _access_fspl_dB(ref_slant)
    ref_atm_loss = _atmospheric_loss_dB(50.0)

    capacity_ref = np.zeros(n_pts)
    for i, p_dBW in enumerate(p_arr):
        gamma_gs, gamma_su = _power_to_link_snrs(p_dBW, ref_fspl, ref_atm_loss)
        gamma_e2e_avg = SNRCalculator.cascaded_snr(gamma_gs, gamma_su)
        capacity_ref[i] = capacity_awgn(gamma_e2e_avg)
    throughput['capacity'] = capacity_ref

    print(f"\nThroughput-Reliability (R={R:.2f}):")
    for i in range(0, n_pts, max(1, n_pts // 6)):
        print(f"  P={p_arr[i]:5.1f}dBW: "
              f"Ada={throughput['adaptive'][i]:.4f}, "
              f"Fix={throughput['fixed'][i]:.4f}, "
              f"C={capacity_ref[i]:.4f}")

    return throughput


def plot_throughput_reliability(throughput: Dict):
    """Plot effective throughput vs total transmit power."""
    fig, ax = plt.subplots(figsize=(10, 7))
    p_arr = throughput['P_total_dBW']

    ax.plot(p_arr, throughput['capacity'], 'k--', linewidth=1.5, alpha=0.6,
            label=r'AWGN Capacity $C(\bar{\gamma})$')
    ax.plot(p_arr, throughput['ppv_adaptive_diversity'],
            's--', color='#2ca02c', linewidth=1.2, markersize=5,
            label='PPV Bound (adaptive)')
    ax.plot(p_arr, throughput['ppv_fixed_diversity'],
            'p--', color='#17becf', linewidth=1.2, markersize=5,
            label='PPV Bound (fixed)')
    ax.plot(p_arr, throughput['no_interleaver'],
            'v-', color='#7f7f7f', linewidth=1.8, markersize=7,
            label='Standard Polar (no interleaver)')
    ax.plot(p_arr, throughput['interleaver'],
            '^-', color='#d62728', linewidth=1.8, markersize=7,
            label='Standard Polar + interleaver')
    ax.plot(p_arr, throughput['fixed'],
            'D-', color='#ff7f0e', linewidth=2, markersize=7,
            label='Fixed Diversity Transform')
    ax.plot(p_arr, throughput['adaptive'],
            'o-', color='#1f77b4', linewidth=2.5, markersize=7,
            label='Adaptive Diversity Transform')

    ax.set_xlabel(r'Total Transmit Power, $P_{\mathrm{total}}$ (dBW)', fontsize=12)
    ax.set_ylabel('Effective Throughput (bits/channel use)', fontsize=12)
    ax.set_title('Throughput-Reliability Trade-off\n'
                 f'({_n_s} subchannels, L={_L_TAPS} taps, Polar SCL)',
                 fontsize=13)
    ax.legend(fontsize=9, loc='lower right')
    ax.grid(True, linestyle='--', alpha=0.7)
    ax.set_ylim(0, None)

    plt.tight_layout()
    plt.savefig('throughput_reliability.png', dpi=300, bbox_inches='tight')
    print("Saved: throughput_reliability.png")
    plt.close()


def save_throughput_csv(throughput: Dict):
    """Save throughput results to CSV."""
    keys = ['capacity', 'ppv_no_diversity', 'ppv_fixed_diversity',
            'ppv_adaptive_diversity', 'no_interleaver', 'interleaver',
            'fixed', 'adaptive']
    with open('throughput_reliability.csv', 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['P_total_dBW'] + keys)
        for i in range(len(throughput['P_total_dBW'])):
            row = [f"{throughput['P_total_dBW'][i]:.2f}"]
            for k in keys:
                row.append(f"{throughput[k][i]:.6e}")
            writer.writerow(row)
    print("Saved: throughput_reliability.csv")


# ============================================================================
#  Function 2: Rank Recovery Probability vs Number of Erased Subchannels
# ============================================================================

def simulate_rank_recovery(n_s: int = _n_s, k_c: int = _DT_CONFIG.k_c,
                           m: int = _m, n_T_samples: int = 200) -> Dict:
    """
    Compute rank recovery probability vs number of erased subchannels.

    Enumerates all C(n_s, n_erased) erasure patterns for each n_erased
    and checks if the surviving columns of G_FIX retain full rank (rho).
    Also tests with random T matrices (adaptive G_DIV = T @ G_FIX).

    Returns:
        dict with n_erased array, P_recovery for G_FIX and adaptive G_DIV
    """
    rho = k_c * m
    G_FIX = _DT_CONFIG.G_FIX
    rng = np.random.default_rng(42)

    n_erased_arr = np.arange(0, n_s + 1)
    p_fix = np.zeros(n_s + 1)
    p_ada = np.zeros(n_s + 1)

    print(f"\nRank Recovery Probability (n_s={n_s}, k_c={k_c}, m={m}):")
    print(f"  G_FIX: ({rho} x {n_s * m}), rho={rho}, d_min={n_s - k_c + 1}")
    print(f"  MDS guarantee: recovery for n_erased <= {n_s - k_c}")

    for ne in range(n_s + 1):
        if ne == 0:
            p_fix[ne] = 1.0
            p_ada[ne] = 1.0
            continue
        if ne == n_s:
            p_fix[ne] = 0.0
            p_ada[ne] = 0.0
            continue

        n_surviving = n_s - ne
        if n_surviving * m < rho:
            p_fix[ne] = 0.0
            p_ada[ne] = 0.0
            continue

        # Enumerate all erasure patterns
        n_pass_fix = 0
        n_pass_ada = 0
        patterns = list(combinations(range(n_s), ne))
        n_patterns = len(patterns)

        for erased_subs in patterns:
            erased_mask = np.zeros(n_s, dtype=bool)
            for s in erased_subs:
                erased_mask[s] = True

            # Check G_FIX
            surviving_cols = []
            for s in range(n_s):
                if not erased_mask[s]:
                    surviving_cols.extend(range(s * m, (s + 1) * m))
            G_sub = G_FIX[:, surviving_cols]
            if gf2_rank(G_sub) == rho:
                n_pass_fix += 1

            # Check adaptive: try n_T_samples random T matrices
            ada_ok = False
            if gf2_rank(G_sub) == rho:
                # If G_FIX works, any T @ G_FIX also works (rank preserved)
                ada_ok = True
            else:
                for _ in range(min(n_T_samples, 50)):
                    T = generate_candidate_T(rho, rng)
                    G_DIV = np.mod(T @ G_FIX, 2).astype(int)
                    G_sub_ada = G_DIV[:, surviving_cols]
                    if gf2_rank(G_sub_ada) == rho:
                        ada_ok = True
                        break
            if ada_ok:
                n_pass_ada += 1

        p_fix[ne] = n_pass_fix / n_patterns
        p_ada[ne] = n_pass_ada / n_patterns
        print(f"  n_erased={ne}: P_fix={p_fix[ne]:.4f} ({n_pass_fix}/{n_patterns}), "
              f"P_ada={p_ada[ne]:.4f} ({n_pass_ada}/{n_patterns})")

    return {
        'n_erased': n_erased_arr,
        'p_recovery_fixed': p_fix,
        'p_recovery_adaptive': p_ada,
    }


def plot_rank_recovery(results: Dict):
    """Plot rank recovery probability vs number of erased subchannels."""
    fig, ax = plt.subplots(figsize=(8, 6))
    ne = results['n_erased']
    width = 0.35

    bars1 = ax.bar(ne - width / 2, results['p_recovery_fixed'], width,
                   color='#ff7f0e', alpha=0.85, label=r'$G_{\mathrm{FIX}}$')
    bars2 = ax.bar(ne + width / 2, results['p_recovery_adaptive'], width,
                   color='#1f77b4', alpha=0.85, label=r'$G_{\mathrm{DIV}} = T \cdot G_{\mathrm{FIX}}$')

    # Annotate d_min line
    d_min = _DT_CONFIG.d_min
    ax.axvline(x=d_min - 1 + 0.5, color='red', linestyle='--', linewidth=1.5,
               alpha=0.7, label=f'$d_{{\\min}}-1 = {d_min - 1}$')

    ax.set_xlabel('Number of Erased Subchannels', fontsize=12)
    ax.set_ylabel('Rank Recovery Probability', fontsize=12)
    ax.set_title('MDS Rank Recovery vs Subchannel Erasures\n'
                 f'($n_s$={_n_s}, $k_c$={_DT_CONFIG.k_c}, $m$={_m}, '
                 f'$d_{{\\min}}$={d_min})',
                 fontsize=13)
    ax.legend(fontsize=10)
    ax.set_xticks(ne)
    ax.set_ylim(0, 1.15)
    ax.grid(True, axis='y', linestyle='--', alpha=0.5)

    plt.tight_layout()
    plt.savefig('rank_recovery.png', dpi=300, bbox_inches='tight')
    print("Saved: rank_recovery.png")
    plt.close()


def save_rank_recovery_csv(results: Dict):
    """Save rank recovery results to CSV."""
    with open('rank_recovery.csv', 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['n_erased', 'P_recovery_fixed', 'P_recovery_adaptive'])
        for i in range(len(results['n_erased'])):
            writer.writerow([results['n_erased'][i],
                             f"{results['p_recovery_fixed'][i]:.6f}",
                             f"{results['p_recovery_adaptive'][i]:.6f}"])
    print("Saved: rank_recovery.csv")


# ============================================================================
#  Function 3: Adaptive vs Fixed Gain vs Blockage Probability
# ============================================================================

def simulate_bler_vs_blockage(N: int = 256, K: int = 128,
                              n_subchannels: int = _n_s,
                              max_erasures: int = _DT_CONFIG.max_erasures,
                              P_total_dBW: float = 8.0,
                              ref_elevation_deg: float = 50.0,
                              p_block_range: Tuple[float, float] = (0.0, 0.35),
                              n_points: int = 8,
                              n_mc: int = 2000) -> Dict:
    """
    BLER vs per-subchannel blockage probability for all 4 MC curves.

    Shows how adaptive diversity gain varies with blockage severity.
    """
    rng = np.random.default_rng(2028)
    p_block_arr = np.linspace(p_block_range[0], p_block_range[1], n_points)

    ref_slant = _slant_range_m(ref_elevation_deg)
    ref_fspl = _access_fspl_dB(ref_slant)
    ref_atm_loss = _atmospheric_loss_dB(ref_elevation_deg)
    gamma_gs, gamma_su = _power_to_link_snrs(P_total_dBW, ref_fspl, ref_atm_loss)

    K_rice = 2.0
    K_rice_dB = 10.0 * np.log10(K_rice)
    erasure_threshold = db_to_linear(-3.0)
    shadow_loss_dB = 20.0

    polar_code = PolarCode(N, K, design_snr_dB=1.0, list_size=8)
    R = K / N

    # Ephemeris eta (fixed for all blockage points)
    predictor = _create_ephemeris_predictor(ref_elevation_deg, P_total_dBW, K_rice_dB)
    bler_eph = predictor.predict_subchannel_bler(ref_elevation_deg, R)
    eta_eph_scalar = max(1.0 - bler_eph, 0.0)
    eta_ephemeris = np.full(n_subchannels, eta_eph_scalar)

    print(f"\nComputing BLER vs Blockage Probability "
          f"(P_total={P_total_dBW} dBW, elev={ref_elevation_deg}deg)...")
    print(f"  MC trials: {n_mc}")

    results = {key: np.zeros(n_points) for key in [
        'no_interleaver', 'interleaver', 'fixed', 'adaptive']}
    results['p_block'] = p_block_arr
    results['gain_ratio'] = np.zeros(n_points)

    for idx, p_blk in enumerate(p_block_arr):
        err_counts = [0, 0, 0, 0]
        for _ in range(n_mc):
            res = _mc_trial_one_snr(
                gamma_su, gamma_gs, N, K, n_subchannels, max_erasures,
                K_rice, erasure_threshold, p_blk,
                shadow_loss_dB, polar_code, rng,
                use_otfs=True, eta_ephemeris=eta_ephemeris)
            err_counts[0] += int(res['err_1'])
            err_counts[1] += int(res['err_2'])
            err_counts[2] += int(res['err_3'])
            err_counts[3] += int(res['err_4'])

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

        print(f"  p_block={p_blk:.3f}: NoIntlv={err_counts[0]/n_mc:.4e}, "
              f"Intlv={err_counts[1]/n_mc:.4e}, "
              f"Fix={err_counts[2]/n_mc:.4e}, "
              f"Ada={err_counts[3]/n_mc:.4e}, "
              f"Gain={results['gain_ratio'][idx]:.2f}x")

    return results


def plot_bler_vs_blockage(results: Dict):
    """Plot BLER vs blockage probability with adaptive gain subplot."""
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 10), height_ratios=[3, 1],
                                    sharex=True)
    p_blk = results['p_block']

    # Top: BLER curves
    ax1.semilogy(p_blk, np.maximum(results['no_interleaver'], 1e-8),
                 'v-', color='#7f7f7f', linewidth=1.8, markersize=8,
                 label='Standard Polar (no interleaver)')
    ax1.semilogy(p_blk, np.maximum(results['interleaver'], 1e-8),
                 '^-', color='#d62728', linewidth=1.8, markersize=8,
                 label='Standard Polar + interleaver')
    ax1.semilogy(p_blk, np.maximum(results['fixed'], 1e-8),
                 'D-', color='#ff7f0e', linewidth=2, markersize=8,
                 label='Fixed Diversity Transform')
    ax1.semilogy(p_blk, np.maximum(results['adaptive'], 1e-8),
                 'o-', color='#1f77b4', linewidth=2.5, markersize=8,
                 label='Adaptive Diversity Transform')

    fixed_clip = np.maximum(results['fixed'], 1e-8)
    adapt_clip = np.maximum(results['adaptive'], 1e-8)
    ax1.fill_between(p_blk, adapt_clip, fixed_clip,
                     alpha=0.18, color='#1f77b4')

    ax1.set_ylabel('Block Error Rate (BLER)', fontsize=12)
    ax1.set_title('BLER vs Blockage Probability\n'
                  f'($P_{{\\mathrm{{total}}}}$=8 dBW, {_n_s} subchannels, '
                  f'L={_L_TAPS} taps)',
                  fontsize=13)
    ax1.legend(fontsize=9, loc='lower right')
    ax1.grid(True, which='both', linestyle='--', alpha=0.7)

    # Bottom: Adaptive gain ratio
    gain = np.clip(results['gain_ratio'], 0, 20)
    ax2.bar(p_blk, gain, width=(p_blk[1] - p_blk[0]) * 0.7,
            color='#1f77b4', alpha=0.7)
    ax2.axhline(y=1.0, color='gray', linestyle='--', linewidth=1)
    ax2.set_xlabel('Per-Subchannel Blockage Probability', fontsize=12)
    ax2.set_ylabel('Gain (Fixed/Adaptive)', fontsize=12)
    ax2.grid(True, linestyle='--', alpha=0.5)

    plt.tight_layout()
    plt.savefig('bler_vs_blockage.png', dpi=300, bbox_inches='tight')
    print("Saved: bler_vs_blockage.png")
    plt.close()


def save_bler_vs_blockage_csv(results: Dict):
    """Save BLER vs blockage results to CSV."""
    with open('bler_vs_blockage.csv', 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['p_block', 'no_interleaver', 'interleaver',
                         'fixed', 'adaptive', 'gain_ratio'])
        for i in range(len(results['p_block'])):
            writer.writerow([
                f"{results['p_block'][i]:.4f}",
                f"{results['no_interleaver'][i]:.6e}",
                f"{results['interleaver'][i]:.6e}",
                f"{results['fixed'][i]:.6e}",
                f"{results['adaptive'][i]:.6e}",
                f"{results['gain_ratio'][i]:.4f}",
            ])
    print("Saved: bler_vs_blockage.csv")


# ============================================================================
#  Function 1: BLER vs Blocklength
# ============================================================================

def simulate_bler_vs_blocklength(blocklengths=None,
                                 n_subchannels: int = _n_s,
                                 max_erasures: int = _DT_CONFIG.max_erasures,
                                 P_total_dBW: float = 8.0,
                                 ref_elevation_deg: float = 50.0,
                                 n_mc: int = 2000) -> Dict:
    """
    BLER vs blocklength n_c for all 7 curves at a fixed operating point.

    Sweeps N (blocklength) with K=N/2 (rate 1/2).
    """
    if blocklengths is None:
        blocklengths = [64, 128, 256, 512, 1024]
    rng = np.random.default_rng(2029)

    ref_slant = _slant_range_m(ref_elevation_deg)
    ref_fspl = _access_fspl_dB(ref_slant)
    ref_atm_loss = _atmospheric_loss_dB(ref_elevation_deg)
    gamma_gs, gamma_su = _power_to_link_snrs(P_total_dBW, ref_fspl, ref_atm_loss)

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

    print(f"\nComputing BLER vs Blocklength "
          f"(P_total={P_total_dBW} dBW, elev={ref_elevation_deg}deg)...")
    print(f"  MC trials: {n_mc}")

    for idx, N in enumerate(blocklengths):
        K = N // 2
        R = K / N
        polar_code = PolarCode(N, K, design_snr_dB=1.0, list_size=8)

        # Ephemeris eta
        predictor = _create_ephemeris_predictor(
            ref_elevation_deg, P_total_dBW, K_rice_dB)
        bler_eph = predictor.predict_subchannel_bler(ref_elevation_deg, R)
        eta_eph_scalar = max(1.0 - bler_eph, 0.0)
        eta_ephemeris = np.full(n_subchannels, eta_eph_scalar)

        ppv_acc = [0.0, 0.0, 0.0]
        err_counts = [0, 0, 0, 0]

        for _ in range(n_mc):
            res = _mc_trial_one_snr(
                gamma_su, gamma_gs, N, K, n_subchannels, max_erasures,
                K_rice, erasure_threshold, p_block_sub,
                shadow_loss_dB, polar_code, rng,
                use_otfs=True, eta_ephemeris=eta_ephemeris)
            ppv_acc[0] += res['ppv_no_div']
            ppv_acc[1] += res['ppv_fix_div']
            ppv_acc[2] += res['ppv_ada_div']
            err_counts[0] += int(res['err_1'])
            err_counts[1] += int(res['err_2'])
            err_counts[2] += int(res['err_3'])
            err_counts[3] += int(res['err_4'])

        results['ppv_no_diversity'][idx] = ppv_acc[0] / n_mc
        results['ppv_fixed_diversity'][idx] = ppv_acc[1] / n_mc
        results['ppv_adaptive_diversity'][idx] = ppv_acc[2] / n_mc
        results['no_interleaver'][idx] = err_counts[0] / n_mc
        results['interleaver'][idx] = err_counts[1] / n_mc
        results['fixed'][idx] = err_counts[2] / n_mc
        results['adaptive'][idx] = err_counts[3] / n_mc

        print(f"  N={N:5d}, K={K:4d}: "
              f"NoIntlv={results['no_interleaver'][idx]:.4e}, "
              f"Intlv={results['interleaver'][idx]:.4e}, "
              f"Fix={results['fixed'][idx]:.4e}, "
              f"Ada={results['adaptive'][idx]:.4e}")

    return results


def plot_bler_vs_blocklength(results: Dict):
    """Plot BLER vs blocklength for all 7 curves."""
    fig, ax = plt.subplots(figsize=(10, 7))
    bl = results['blocklength']

    ax.semilogy(bl, np.maximum(results['ppv_adaptive_diversity'], 1e-8),
                's--', color='#2ca02c', linewidth=1.5, markersize=6,
                label='PPV Bound (adaptive)')
    ax.semilogy(bl, np.maximum(results['ppv_fixed_diversity'], 1e-8),
                'p--', color='#17becf', linewidth=1.5, markersize=6,
                label='PPV Bound (fixed)')
    ax.semilogy(bl, np.maximum(results['ppv_no_diversity'], 1e-8),
                'h--', color='#9467bd', linewidth=1.5, markersize=6,
                label='PPV Bound (no diversity)')
    ax.semilogy(bl, np.maximum(results['no_interleaver'], 1e-8),
                'v-', color='#7f7f7f', linewidth=1.8, markersize=8,
                label='Standard Polar (no interleaver)')
    ax.semilogy(bl, np.maximum(results['interleaver'], 1e-8),
                '^-', color='#d62728', linewidth=1.8, markersize=8,
                label='Standard Polar + interleaver')
    ax.semilogy(bl, np.maximum(results['fixed'], 1e-8),
                'D-', color='#ff7f0e', linewidth=2, markersize=8,
                label='Fixed Diversity Transform')
    ax.semilogy(bl, np.maximum(results['adaptive'], 1e-8),
                'o-', color='#1f77b4', linewidth=2.5, markersize=8,
                label='Adaptive Diversity Transform')

    ax.set_xlabel('Blocklength $n_c$', fontsize=12)
    ax.set_ylabel('Block Error Rate (BLER)', fontsize=12)
    ax.set_title('BLER vs Blocklength: Finite-Blocklength Performance\n'
                 f'(Rate 1/2, $P_{{\\mathrm{{total}}}}$=8 dBW, {_n_s} subchannels)',
                 fontsize=13)
    ax.legend(fontsize=9, loc='upper right')
    ax.grid(True, which='both', linestyle='--', alpha=0.7)
    ax.set_xscale('log', base=2)
    ax.set_xticks(results['blocklength'])
    ax.set_xticklabels([str(int(x)) for x in results['blocklength']])

    plt.tight_layout()
    plt.savefig('bler_vs_blocklength.png', dpi=300, bbox_inches='tight')
    print("Saved: bler_vs_blocklength.png")
    plt.close()


def save_bler_vs_blocklength_csv(results: Dict):
    """Save BLER vs blocklength results to CSV."""
    keys = ['ppv_no_diversity', 'ppv_fixed_diversity', 'ppv_adaptive_diversity',
            'no_interleaver', 'interleaver', 'fixed', 'adaptive']
    with open('bler_vs_blocklength.csv', 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['blocklength'] + keys)
        for i in range(len(results['blocklength'])):
            row = [str(int(results['blocklength'][i]))]
            for k in keys:
                row.append(f"{results[k][i]:.6e}")
            writer.writerow(row)
    print("Saved: bler_vs_blocklength.csv")


# ============================================================================
#  Function 6: OTFS vs OFDM Comparison
# ============================================================================

def simulate_otfs_vs_ofdm(N: int = 256, K: int = 128,
                          n_subchannels: int = _n_s,
                          max_erasures: int = _DT_CONFIG.max_erasures,
                          power_range_dBW: Tuple[float, float] = (-6, 15),
                          ref_elevation_deg: float = 50.0,
                          n_points: int = 12,
                          n_mc: int = 2000) -> Dict:
    """
    Compare OTFS (DD-domain) vs OFDM (legacy per-subchannel AWGN) with diversity.

    Runs both OTFS and OFDM paths at each power point, collecting BLER
    for fixed and adaptive diversity curves.
    """
    rng = np.random.default_rng(2030)
    power_arr = np.linspace(power_range_dBW[0], power_range_dBW[1], n_points)

    ref_slant = _slant_range_m(ref_elevation_deg)
    ref_fspl = _access_fspl_dB(ref_slant)
    ref_atm_loss = _atmospheric_loss_dB(ref_elevation_deg)

    K_rice = 2.0
    K_rice_dB = 10.0 * np.log10(K_rice)
    erasure_threshold = db_to_linear(-3.0)
    shadow_loss_dB = 20.0
    p_block_sub = 0.15

    polar_code = PolarCode(N, K, design_snr_dB=1.0, list_size=8)
    R = K / N

    print(f"\nComputing OTFS vs OFDM comparison (elev={ref_elevation_deg}deg)...")
    print(f"  MC trials: {n_mc} per modulation per power point")

    # Result keys: otfs_* and ofdm_*
    curve_keys = ['no_interleaver', 'fixed', 'adaptive']
    results = {'P_total_dBW': power_arr}
    for prefix in ['otfs', 'ofdm']:
        for key in curve_keys:
            results[f'{prefix}_{key}'] = np.zeros(n_points)

    for idx, p_dBW in enumerate(power_arr):
        gamma_gs, gamma_su = _power_to_link_snrs(p_dBW, ref_fspl, ref_atm_loss)

        predictor = _create_ephemeris_predictor(
            ref_elevation_deg, p_dBW, K_rice_dB)
        bler_eph = predictor.predict_subchannel_bler(ref_elevation_deg, R)
        eta_eph_scalar = max(1.0 - bler_eph, 0.0)
        eta_ephemeris = np.full(n_subchannels, eta_eph_scalar)

        for use_otfs, prefix in [(True, 'otfs'), (False, 'ofdm')]:
            err_counts = {k: 0 for k in curve_keys}
            for _ in range(n_mc):
                res = _mc_trial_one_snr(
                    gamma_su, gamma_gs, N, K, n_subchannels, max_erasures,
                    K_rice, erasure_threshold, p_block_sub,
                    shadow_loss_dB, polar_code, rng,
                    use_otfs=use_otfs, eta_ephemeris=eta_ephemeris)
                err_counts['no_interleaver'] += int(res['err_1'])
                err_counts['fixed'] += int(res['err_3'])
                err_counts['adaptive'] += int(res['err_4'])

            for key in curve_keys:
                results[f'{prefix}_{key}'][idx] = err_counts[key] / n_mc

        print(f"  P={p_dBW:5.1f}dBW: "
              f"OTFS(Fix={results['otfs_fixed'][idx]:.3e}, "
              f"Ada={results['otfs_adaptive'][idx]:.3e}) | "
              f"OFDM(Fix={results['ofdm_fixed'][idx]:.3e}, "
              f"Ada={results['ofdm_adaptive'][idx]:.3e})")

    return results


def plot_otfs_vs_ofdm(results: Dict):
    """Plot OTFS vs OFDM comparison."""
    fig, ax = plt.subplots(figsize=(10, 7))
    p_arr = results['P_total_dBW']

    # OTFS curves (solid lines)
    ax.semilogy(p_arr, np.maximum(results['otfs_no_interleaver'], 1e-8),
                'v-', color='#7f7f7f', linewidth=1.8, markersize=7,
                label='OTFS — No interleaver')
    ax.semilogy(p_arr, np.maximum(results['otfs_fixed'], 1e-8),
                'D-', color='#ff7f0e', linewidth=2, markersize=8,
                label='OTFS — Fixed Diversity')
    ax.semilogy(p_arr, np.maximum(results['otfs_adaptive'], 1e-8),
                'o-', color='#1f77b4', linewidth=2.5, markersize=8,
                label='OTFS — Adaptive Diversity')

    # OFDM curves (dashed lines)
    ax.semilogy(p_arr, np.maximum(results['ofdm_no_interleaver'], 1e-8),
                'v--', color='#7f7f7f', linewidth=1.8, markersize=7,
                alpha=0.7, label='OFDM — No interleaver')
    ax.semilogy(p_arr, np.maximum(results['ofdm_fixed'], 1e-8),
                'D--', color='#ff7f0e', linewidth=2, markersize=8,
                alpha=0.7, label='OFDM — Fixed Diversity')
    ax.semilogy(p_arr, np.maximum(results['ofdm_adaptive'], 1e-8),
                'o--', color='#1f77b4', linewidth=2.5, markersize=8,
                alpha=0.7, label='OFDM — Adaptive Diversity')

    ax.set_xlabel(r'Total Transmit Power, $P_{\mathrm{total}}$ (dBW)', fontsize=12)
    ax.set_ylabel('Block Error Rate (BLER)', fontsize=12)
    ax.set_title('OTFS vs OFDM: Diversity Transform Performance\n'
                 f'(L={_L_TAPS} taps, {_n_s} subchannels, Polar SCL)',
                 fontsize=13)
    ax.legend(fontsize=9, loc='lower left', ncol=2)
    ax.grid(True, which='both', linestyle='--', alpha=0.7)
    ax.set_ylim(5e-4, 2)

    plt.tight_layout()
    plt.savefig('otfs_vs_ofdm.png', dpi=300, bbox_inches='tight')
    print("Saved: otfs_vs_ofdm.png")
    plt.close()


def save_otfs_vs_ofdm_csv(results: Dict):
    """Save OTFS vs OFDM results to CSV."""
    keys = ['otfs_no_interleaver', 'otfs_fixed', 'otfs_adaptive',
            'ofdm_no_interleaver', 'ofdm_fixed', 'ofdm_adaptive']
    with open('otfs_vs_ofdm.csv', 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['P_total_dBW'] + keys)
        for i in range(len(results['P_total_dBW'])):
            row = [f"{results['P_total_dBW'][i]:.2f}"]
            for k in keys:
                row.append(f"{results[k][i]:.6e}")
            writer.writerow(row)
    print("Saved: otfs_vs_ofdm.csv")


# ============================================================================
#  Function 4: BLER vs Number of Subchannels
# ============================================================================

def _mc_trial_variable_ns(gamma_avg_SU, gamma_gs, N, K, dt_config,
                          G_FIX_local, G_DIV_pool, block_weights_pool,
                          G_FIX_block_weights, X_ALL_local, S_ALL_FIX_local,
                          X_MASK_0_local, X_MASK_1_local,
                          subchannel_map, n_sym_per_sub,
                          K_rice, erasure_threshold, p_block_sub,
                          shadow_loss_dB, polar_code, rng,
                          eta_ephemeris=None):
    """
    MC trial for variable n_s configuration.

    Similar to _mc_trial_one_snr but uses locally-provided DT config
    instead of module-level globals. Only runs diversity curves (3 & 4)
    and PPV bounds.
    """
    n_s = dt_config.n_s
    m = dt_config.m
    k_c = dt_config.k_c
    rho = dt_config.rho
    n_out = dt_config.n_out
    max_erasures = dt_config.max_erasures

    # Generate per-subchannel channel (legacy AWGN path for simplicity)
    gamma_e2e, blocked_mask = _generate_multipath_per_subchannel(
        gamma_avg_SU, gamma_gs, n_s, K_rice, p_block_sub, shadow_loss_dB, rng)

    # Erasure mask
    erased_mask = gamma_e2e < erasure_threshold
    n_erased = int(np.sum(erased_mask))

    # Instantaneous eta
    R = K / N
    N_sub = N // n_s
    eta_inst = np.zeros(n_s)
    for j in range(n_s):
        if erased_mask[j]:
            eta_inst[j] = 0.0
        else:
            eps_j = conditional_bler(gamma_e2e[j], N_sub, R)
            eta_inst[j] = 1.0 - eps_j

    eta_for_T = eta_ephemeris if eta_ephemeris is not None else eta_inst

    # Algorithm 2 T-selection
    best_Q = np.dot(G_FIX_block_weights, eta_for_T)
    best_G_DIV = G_FIX_local
    best_block_weights = G_FIX_block_weights.copy()
    for i in range(len(G_DIV_pool)):
        Q = np.dot(block_weights_pool[i], eta_for_T)
        if Q > best_Q:
            best_Q = Q
            best_G_DIV = G_DIV_pool[i]
            best_block_weights = block_weights_pool[i].copy()

    # PPV bounds
    ppv_fix = bler_fixed_diversity(gamma_e2e, erased_mask, k_c, N, K)
    Q_fix_val = float(np.dot(G_FIX_block_weights, eta_for_T))
    Q_ratio = float(best_Q) / max(Q_fix_val, 1e-10)
    Q_ratio = max(Q_ratio, 1.0)
    ppv_ada = ppv_fix / Q_ratio

    # Info bits
    info_bits = rng.integers(0, 2, size=K).astype(np.int8)

    # Curve 3: Fixed diversity (legacy AWGN path)
    if n_erased > max_erasures:
        err_fix = True
    else:
        coded_bits = polar_code.encode(info_bits)
        llr_fix = _diversity_transform_transmit_receive_variable(
            coded_bits, G_FIX_local, gamma_e2e, erased_mask, eta_inst,
            rng, n_s, m, rho, n_out,
            X_ALL_local, S_ALL_FIX_local, X_MASK_0_local, X_MASK_1_local)
        if llr_fix is None:
            err_fix = True
        else:
            decoded = polar_code.decode(llr_fix)
            err_fix = not np.array_equal(decoded, info_bits)

    # Curve 4: Adaptive diversity
    if n_erased > max_erasures:
        err_ada = True
    else:
        n_surviving = n_s - n_erased
        power_alloc = np.ones(n_s)
        if n_surviving > 0 and n_erased > 0:
            power_alloc[~erased_mask] = float(n_s) / n_surviving
            power_alloc[erased_mask] = 0.0
        gamma_adaptive = gamma_e2e * power_alloc

        # Recompute S_ALL for best_G_DIV
        Z_ALL_ada = np.mod(best_G_DIV.T.astype(int) @ X_ALL_local.astype(int),
                           2).astype(np.int8)
        S_ALL_ada = 1.0 - 2.0 * Z_ALL_ada.astype(np.float64)

        coded_bits_4 = polar_code.encode(info_bits)
        llr_ada = _diversity_transform_transmit_receive_variable(
            coded_bits_4, best_G_DIV, gamma_adaptive, erased_mask, eta_inst,
            rng, n_s, m, rho, n_out,
            X_ALL_local, S_ALL_ada, X_MASK_0_local, X_MASK_1_local)
        if llr_ada is None:
            err_ada = True
        else:
            decoded_4 = polar_code.decode(llr_ada)
            err_ada = not np.array_equal(decoded_4, info_bits)

    return {
        'ppv_fix': ppv_fix,
        'ppv_ada': ppv_ada,
        'err_fix': err_fix,
        'err_ada': err_ada,
    }


def _diversity_transform_transmit_receive_variable(
        coded_bits, G_DIV, gamma_e2e_per_sub, erased_mask, eta,
        rng, n_s, m, rho, n_out,
        X_ALL, S_ALL, X_MASK_0, X_MASK_1):
    """
    Legacy AWGN-path diversity transform for variable n_s.

    Same logic as _diversity_transform_transmit_receive() but uses
    locally-provided dimensions and MAP lookup tables.
    """
    N = len(coded_bits)
    n_pos = int(np.ceil(N / rho))
    padded = np.zeros(rho * n_pos, dtype=np.int8)
    padded[:N] = coded_bits

    X = padded.reshape(rho, n_pos)
    Z = np.mod(G_DIV.T.astype(int) @ X.astype(int), 2).astype(np.int8)

    # Transmit through per-subchannel QPSK AWGN
    llr_z = np.zeros((n_out, n_pos))
    for j in range(n_s):
        eta_j = eta[j]
        gamma_j = gamma_e2e_per_sub[j]
        if erased_mask[j]:
            llr_z[j * m:(j + 1) * m, :] = 0.0
        else:
            sub_bits = Z[j * m:(j + 1) * m, :].flatten()
            qpsk_syms = _qpsk_modulate(sub_bits)
            noise_std = np.sqrt(0.5 / max(gamma_j, 1e-10))
            noise = noise_std * (rng.normal(0, 1, len(qpsk_syms))
                                 + 1j * rng.normal(0, 1, len(qpsk_syms)))
            y = qpsk_syms + noise
            llr_bits = _qpsk_soft_demod(y, gamma_j)
            llr_bits *= eta_j
            llr_z[j * m:(j + 1) * m, :] = llr_bits.reshape(m, n_pos)

    # Check at least one subchannel survives
    if np.all(erased_mask):
        return None

    # MAP soft demapping (vectorized, matching original function)
    corr = (S_ALL.T @ llr_z) * 0.5  # (N_CW, n_pos)

    X_llr = np.zeros((rho, n_pos))
    for i in range(rho):
        X_llr[i, :] = (np.max(corr[X_MASK_0[i], :], axis=0)
                        - np.max(corr[X_MASK_1[i], :], axis=0))

    # Flatten row-major and strip padding
    llr_coded = X_llr.flatten()[:N]
    return llr_coded


def simulate_bler_vs_n_subchannels(N: int = 256, K: int = 128,
                                    ns_values=None,
                                    P_total_dBW: float = 8.0,
                                    ref_elevation_deg: float = 50.0,
                                    n_mc: int = 1500) -> Dict:
    """
    BLER vs number of subchannels n_s.

    For each n_s, creates a DiversityTransformConfig with k_c = n_s - 2
    (maintaining max_erasures=2 when possible), builds local MAP tables,
    and runs MC trials.
    """
    if ns_values is None:
        ns_values = [2, 3, 4, 5, 6, 7]
    m = 3  # GF(2^3) extension degree

    rng = np.random.default_rng(2031)
    ref_slant = _slant_range_m(ref_elevation_deg)
    ref_fspl = _access_fspl_dB(ref_slant)
    ref_atm_loss = _atmospheric_loss_dB(ref_elevation_deg)
    gamma_gs, gamma_su = _power_to_link_snrs(P_total_dBW, ref_fspl, ref_atm_loss)

    K_rice = 2.0
    K_rice_dB = 10.0 * np.log10(K_rice)
    erasure_threshold = db_to_linear(-3.0)
    shadow_loss_dB = 20.0
    p_block_sub = 0.15
    R = K / N

    n_pts = len(ns_values)
    results = {key: np.zeros(n_pts) for key in [
        'ppv_fixed', 'ppv_adaptive', 'fixed', 'adaptive']}
    results['n_subchannels'] = np.array(ns_values)

    print(f"\nComputing BLER vs n_subchannels "
          f"(P_total={P_total_dBW} dBW, elev={ref_elevation_deg}deg)...")
    print(f"  MC trials: {n_mc}")

    for idx, n_s_val in enumerate(ns_values):
        # k_c = n_s - 2 ensures max_erasures=2, but k_c >= 1
        k_c_val = max(n_s_val - 2, 1)
        if k_c_val >= n_s_val:
            k_c_val = n_s_val - 1  # ensure n_out > rho

        dt_config = DiversityTransformConfig(k_c=k_c_val, n_s=n_s_val, m=m)
        rho = dt_config.rho
        n_out = dt_config.n_out
        G_FIX_local = dt_config.G_FIX

        # Build local MAP tables
        N_CW_local = 1 << rho
        X_ALL_local = np.zeros((rho, N_CW_local), dtype=np.int8)
        for i_cw in range(N_CW_local):
            for b in range(rho):
                X_ALL_local[b, i_cw] = (i_cw >> b) & 1

        Z_ALL_FIX_local = np.mod(
            G_FIX_local.T.astype(int) @ X_ALL_local.astype(int), 2
        ).astype(np.int8)
        S_ALL_FIX_local = 1.0 - 2.0 * Z_ALL_FIX_local.astype(np.float64)

        X_MASK_0_local = [X_ALL_local[i, :] == 0 for i in range(rho)]
        X_MASK_1_local = [X_ALL_local[i, :] == 1 for i in range(rho)]

        # Build G_DIV pool
        pool_rng = np.random.default_rng(42)
        n_pool = 200
        G_DIV_pool = []
        block_weights_pool = np.zeros((n_pool, n_s_val), dtype=float)
        G_FIX_bw = np.array([
            np.sum(G_FIX_local[:, i * m:(i + 1) * m]) for i in range(n_s_val)
        ], dtype=float)

        for i in range(n_pool):
            T = generate_candidate_T(rho, pool_rng)
            G_DIV = np.mod(T.astype(int) @ G_FIX_local.astype(int), 2).astype(int)
            G_DIV_pool.append(G_DIV)
            for j in range(n_s_val):
                block_weights_pool[i, j] = np.sum(G_DIV[:, j * m:(j + 1) * m])

        # Subchannel map
        m_sub = _OTFS_PARAMS.M_D // n_s_val
        sc_map = _build_subchannel_map(n_s=n_s_val, m_sub=m_sub)
        n_sym = {j: len(sc_map[j]) for j in range(n_s_val)}

        # Polar code
        polar_code = PolarCode(N, K, design_snr_dB=1.0, list_size=8)

        # Ephemeris eta
        predictor = _create_ephemeris_predictor(
            ref_elevation_deg, P_total_dBW, K_rice_dB)
        bler_eph = predictor.predict_subchannel_bler(ref_elevation_deg, R)
        eta_eph_scalar = max(1.0 - bler_eph, 0.0)
        eta_ephemeris = np.full(n_s_val, eta_eph_scalar)

        ppv_fix_acc = 0.0
        ppv_ada_acc = 0.0
        err_fix_count = 0
        err_ada_count = 0

        print(f"  n_s={n_s_val}, k_c={k_c_val}, rho={rho}, "
              f"max_erasures={dt_config.max_erasures}, "
              f"2^rho={N_CW_local}...")

        for _ in range(n_mc):
            res = _mc_trial_variable_ns(
                gamma_su, gamma_gs, N, K, dt_config,
                G_FIX_local, G_DIV_pool, block_weights_pool,
                G_FIX_bw, X_ALL_local, S_ALL_FIX_local,
                X_MASK_0_local, X_MASK_1_local,
                sc_map, n_sym,
                K_rice, erasure_threshold, p_block_sub,
                shadow_loss_dB, polar_code, rng,
                eta_ephemeris=eta_ephemeris)
            ppv_fix_acc += res['ppv_fix']
            ppv_ada_acc += res['ppv_ada']
            err_fix_count += int(res['err_fix'])
            err_ada_count += int(res['err_ada'])

        results['ppv_fixed'][idx] = ppv_fix_acc / n_mc
        results['ppv_adaptive'][idx] = ppv_ada_acc / n_mc
        results['fixed'][idx] = err_fix_count / n_mc
        results['adaptive'][idx] = err_ada_count / n_mc

        print(f"    Fix={results['fixed'][idx]:.4e}, "
              f"Ada={results['adaptive'][idx]:.4e}, "
              f"PPV_Fix={results['ppv_fixed'][idx]:.4e}, "
              f"PPV_Ada={results['ppv_adaptive'][idx]:.4e}")

    return results


def plot_bler_vs_n_subchannels(results: Dict):
    """Plot BLER vs number of subchannels."""
    fig, ax = plt.subplots(figsize=(9, 7))
    ns = results['n_subchannels']

    ax.semilogy(ns, np.maximum(results['ppv_fixed'], 1e-8),
                'p--', color='#17becf', linewidth=1.5, markersize=8,
                label='PPV Bound (fixed)')
    ax.semilogy(ns, np.maximum(results['ppv_adaptive'], 1e-8),
                's--', color='#2ca02c', linewidth=1.5, markersize=8,
                label='PPV Bound (adaptive)')
    ax.semilogy(ns, np.maximum(results['fixed'], 1e-8),
                'D-', color='#ff7f0e', linewidth=2, markersize=9,
                label='Fixed Diversity Transform')
    ax.semilogy(ns, np.maximum(results['adaptive'], 1e-8),
                'o-', color='#1f77b4', linewidth=2.5, markersize=9,
                label='Adaptive Diversity Transform')

    ax.set_xlabel('Number of Subchannels $n_s$', fontsize=12)
    ax.set_ylabel('Block Error Rate (BLER)', fontsize=12)
    ax.set_title('BLER vs Number of Subchannels\n'
                 f'($P_{{\\mathrm{{total}}}}$=8 dBW, GF($2^3$), '
                 f'$k_c = n_s - 2$)',
                 fontsize=13)
    ax.legend(fontsize=10, loc='upper right')
    ax.grid(True, which='both', linestyle='--', alpha=0.7)
    ax.set_xticks(ns)

    plt.tight_layout()
    plt.savefig('bler_vs_n_subchannels.png', dpi=300, bbox_inches='tight')
    print("Saved: bler_vs_n_subchannels.png")
    plt.close()


def save_bler_vs_n_subchannels_csv(results: Dict):
    """Save BLER vs n_subchannels results to CSV."""
    keys = ['ppv_fixed', 'ppv_adaptive', 'fixed', 'adaptive']
    with open('bler_vs_n_subchannels.csv', 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['n_subchannels'] + keys)
        for i in range(len(results['n_subchannels'])):
            row = [str(int(results['n_subchannels'][i]))]
            for k in keys:
                row.append(f"{results[k][i]:.6e}")
            writer.writerow(row)
    print("Saved: bler_vs_n_subchannels.csv")


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
    print(f"  Atmospheric model (Eq. 6-7):")
    print(f"    Gaseous: ITU-R P.676 (rho_wv={_ATM_PARAMS.rho_wv} g/m3)")
    print(f"    Cloud/fog: ITU-R P.840 (LWC={_ATM_PARAMS.LWC} g/m3)")
    print(f"    Rain: ITU-R P.618 (rate={_ATM_PARAMS.rain_rate_mmh} mm/h)")
    print(f"    Environment: {_ENV_PARAMS.env_type}")

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

    # ---- Additional simulations ----

    # Throughput-Reliability Trade-off (from existing power sweep results)
    print("\n" + "-" * 60)
    results_throughput = simulate_throughput_reliability(results_power, N, K)
    plot_throughput_reliability(results_throughput)
    save_throughput_csv(results_throughput)

    # Rank Recovery Probability (algebraic, no MC)
    print("\n" + "-" * 60)
    results_rank = simulate_rank_recovery()
    plot_rank_recovery(results_rank)
    save_rank_recovery_csv(results_rank)

    # BLER vs Blockage Probability
    print("\n" + "-" * 60)
    results_blockage = simulate_bler_vs_blockage(
        N, K, n_subchannels, max_erasures, n_mc=2000)
    plot_bler_vs_blockage(results_blockage)
    save_bler_vs_blockage_csv(results_blockage)

    # BLER vs Blocklength
    print("\n" + "-" * 60)
    results_blocklength = simulate_bler_vs_blocklength(n_mc=2000)
    plot_bler_vs_blocklength(results_blocklength)
    save_bler_vs_blocklength_csv(results_blocklength)

    # OTFS vs OFDM Comparison
    print("\n" + "-" * 60)
    results_otfs_ofdm = simulate_otfs_vs_ofdm(N, K, n_subchannels,
                                                max_erasures, n_mc=2000)
    plot_otfs_vs_ofdm(results_otfs_ofdm)
    save_otfs_vs_ofdm_csv(results_otfs_ofdm)

    # BLER vs Number of Subchannels
    print("\n" + "-" * 60)
    results_ns = simulate_bler_vs_n_subchannels(N, K, n_mc=1500)
    plot_bler_vs_n_subchannels(results_ns)
    save_bler_vs_n_subchannels_csv(results_ns)

    return results_power, results_elev


if __name__ == "__main__":
    results = main()
