"""
Finite-Blocklength BLER Analysis (Section III)
===============================================
Implements finite-blocklength error probability analysis for
LEO NTN OTFS systems with diversity transforms.

Provides:
  - PPV normal approximation (Polyanskiy-Poor-Verdu)
  - Conditional BLER epsilon(gamma) — Eq. (41)
  - Closed-form BLER with shifted gamma distribution — Proposition 1 / Eq. (47)
  - High-SNR approximation revealing diversity order — Proposition 2 / Eq. (49)
  - Semi-analytical BLER for no-diversity, fixed diversity, adaptive diversity
  - Vectorized PPV bounds for batch computation

Author: Research implementation
Date: 2026-02-21
"""

import numpy as np
import math
from typing import Optional

from scipy.special import erfc


# ============================================================================
#  Helper functions
# ============================================================================

def _gammainc_series(a, x):
    """Series for lower regularised incomplete gamma P(a, x)."""
    if x <= 0:
        return 0.0
    if a <= 0:
        return 1.0
    try:
        log_prefix = -x + a * math.log(x) - math.lgamma(a)
    except (ValueError, OverflowError):
        return 0.0
    if log_prefix < -700:
        return 0.0
    term = 1.0 / a
    total = term
    for k in range(1, 500):
        term *= x / (a + k)
        total += term
        if abs(term) < 1e-15 * abs(total):
            break
    result = total * math.exp(log_prefix)
    return max(0.0, min(1.0, result))


def regularised_lower_inc_gamma(a: float, x: float) -> float:
    """P(a, x) = gamma(a,x)/Gamma(a) — regularised lower incomplete gamma."""
    if x <= 0:
        return 0.0
    return _gammainc_series(a, x)


def q_function(x) -> np.ndarray:
    """Gaussian Q-function: Q(x) = 0.5 * erfc(x / sqrt(2))."""
    x = np.asarray(x, dtype=float)
    return 0.5 * erfc(x / np.sqrt(2.0))


def q_inv(p: float) -> float:
    """Inverse Q-function via bisection (scalar)."""
    if p <= 0:
        return np.inf
    if p >= 1:
        return -np.inf
    lo, hi = -10.0, 10.0
    for _ in range(100):
        mid = (lo + hi) / 2.0
        if float(q_function(mid)) > p:
            lo = mid
        else:
            hi = mid
    return (lo + hi) / 2.0


# ============================================================================
#  Core AWGN capacity and dispersion
# ============================================================================

def capacity_awgn(snr_linear: float) -> float:
    """AWGN channel capacity C(gamma) = log2(1 + gamma) [bits/channel use]."""
    return np.log2(1.0 + snr_linear)


def channel_dispersion(snr_linear: float) -> float:
    """Channel dispersion V(gamma) = (1 - 1/(1+gamma)^2)(log2 e)^2 — Eq. (42)."""
    if snr_linear <= 0:
        return 0.0
    return (1.0 - 1.0 / (1.0 + snr_linear) ** 2) * (np.log2(np.e)) ** 2


# ============================================================================
#  Conditional BLER — PPV normal approximation
# ============================================================================

def conditional_bler(gamma: float, n_c: int, R_c: float) -> float:
    """
    Conditional BLER epsilon(gamma) for given SNR — Eq. (41).

    Uses the Polyanskiy-Poor-Verdu normal approximation:
      epsilon approx Q( sqrt(n_c) * (C(gamma) - R_c) / sqrt(V(gamma)) )

    Args:
        gamma: instantaneous SNR (linear)
        n_c: blocklength
        R_c: coding rate [bits/channel use]

    Returns:
        Conditional block error probability.
    """
    if gamma <= 0:
        return 1.0
    C = capacity_awgn(gamma)
    V = channel_dispersion(gamma)
    if V <= 0 or n_c <= 0:
        return 0.0 if C >= R_c else 1.0
    arg = (np.sqrt(n_c) * (C - R_c)
           + 0.5 * np.log2(n_c) / np.sqrt(n_c)) / np.sqrt(V)
    return float(q_function(arg))


def finite_blocklength_bler(snr_linear: float, N: int, K: int) -> float:
    """
    Finite blocklength BLER via PPV normal approximation.

    Args:
        snr_linear: channel SNR (linear)
        N: block length
        K: information bits

    Returns:
        Block error probability.
    """
    if snr_linear <= 0:
        return 1.0
    R = K / N
    C = capacity_awgn(snr_linear)
    V = channel_dispersion(snr_linear)
    if V <= 0:
        return 1.0 if R > C else 0.0
    if C > R:
        arg = (C - R) * np.sqrt(N / V)
        bler = float(q_function(arg))
    else:
        bler = 1.0
    return np.clip(bler, 0, 1)


def finite_blocklength_bler_vec(snr_linear, N, K):
    """
    Vectorized finite blocklength BLER approximation.

    All arguments can be numpy arrays (broadcastable).
    """
    snr_linear = np.asarray(snr_linear, dtype=float)
    N = np.asarray(N, dtype=float)
    K = np.asarray(K, dtype=float)
    R = K / N
    C = 0.5 * np.log2(np.maximum(1 + snr_linear, 1e-30))
    V = 0.5 * (1 - 1 / (1 + snr_linear) ** 2) * (np.log2(np.e)) ** 2
    arg = np.where(
        (V > 0) & (C > R),
        (C - R) * np.sqrt(N / np.maximum(V, 1e-30)),
        0.0,
    )
    bler = np.where(
        (V > 0) & (C > R),
        0.5 * erfc(arg / np.sqrt(2)),
        np.where(snr_linear > 0, 1.0, 1.0),
    )
    return np.clip(bler, 0, 1)


def polar_sc_bler(snr_dB: float, N: int, K: int,
                  sc_loss_dB: float = 0.5) -> float:
    """Polar code BLER with SC decoding (0.5 dB loss model)."""
    effective_snr_dB = snr_dB - sc_loss_dB
    snr_linear = 10 ** (effective_snr_dB / 10)
    return finite_blocklength_bler(snr_linear, N, K)


def polar_sc_bler_vec(snr_dB, N, K, sc_loss_dB=0.5):
    """Vectorized Polar code BLER with SC decoding."""
    snr_dB = np.asarray(snr_dB, dtype=float)
    snr_linear = 10 ** ((snr_dB - sc_loss_dB) / 10)
    return finite_blocklength_bler_vec(snr_linear, N, K)


# ============================================================================
#  Closed-form and high-SNR BLER approximations (Propositions 1 & 2)
# ============================================================================

class FiniteBlocklengthBLER:
    """
    Closed-form and high-SNR BLER approximations.

    Implements Proposition 1 (closed-form with shifted gamma distribution)
    and Proposition 2 (high-SNR approximation revealing diversity order).
    """

    @staticmethod
    def capacity(gamma: float) -> float:
        """C(gamma) = log2(1+gamma)."""
        return capacity_awgn(gamma)

    @staticmethod
    def dispersion(gamma: float) -> float:
        """V(gamma) — Eq. (42)."""
        return channel_dispersion(gamma)

    @classmethod
    def conditional_bler(cls, gamma: float, n_c: int, R_c: float) -> float:
        """Conditional BLER epsilon(gamma) — Q-function approximation (Eq. 41)."""
        return conditional_bler(gamma, n_c, R_c)

    @classmethod
    def average_bler_mc(cls, gamma_bar_SU: float, P_LOS: float,
                        P_NLOS_taps: np.ndarray, K_LOS: float,
                        n_c: int, R_c: float,
                        gamma_bar_GS: float = np.inf,
                        n_samples: int = 10000,
                        rng: np.random.Generator = None) -> float:
        """
        Monte Carlo estimate of average BLER — Eq. (43).

        Averages the conditional BLER over the combined SNR distribution
        (shifted gamma from LOS + NLOS paths).
        """
        if rng is None:
            rng = np.random.default_rng(42)

        L_nlos = len(P_NLOS_taps)
        gamma_det = gamma_bar_SU * P_LOS

        bler_sum = 0.0
        for _ in range(n_samples):
            gamma_nlos = 0.0
            for p in range(L_nlos):
                fading = rng.exponential(1.0)
                gamma_nlos += gamma_bar_SU * P_NLOS_taps[p] * fading
            gamma_combined = gamma_det + gamma_nlos
            if np.isfinite(gamma_bar_GS):
                gamma_combined = (gamma_bar_GS * gamma_combined) / \
                                 (gamma_bar_GS + gamma_combined + 1.0)
            bler_sum += cls.conditional_bler(gamma_combined, n_c, R_c)

        return bler_sum / n_samples

    @staticmethod
    def bler_closed_form(gamma_th: float, gamma_det: float,
                         gamma_bar_NLOS: float, L_i: int) -> float:
        """
        Closed-form BLER approximation — Proposition 1 / Eq. (47).

        epsilon approx P(L-1, (gamma_th - gamma_det) / gamma_bar_NLOS)
        """
        if gamma_th <= gamma_det:
            return 0.0
        if gamma_bar_NLOS <= 1e-30:
            return 1.0
        x = (gamma_th - gamma_det) / gamma_bar_NLOS
        a = L_i - 1
        if a <= 0:
            return 1.0 - math.exp(-min(x, 700))
        return float(regularised_lower_inc_gamma(a, x))

    @staticmethod
    def bler_explicit(gamma_th: float, gamma_det: float,
                      gamma_bar_NLOS: float, L_i: int) -> float:
        """Explicit series BLER — Eq. (48)."""
        if gamma_th <= gamma_det:
            return 0.0
        if gamma_bar_NLOS <= 1e-30:
            return 1.0
        xi = (gamma_th - gamma_det) / gamma_bar_NLOS
        if xi > 700:
            return 1.0
        s = sum(xi ** m / math.factorial(m) for m in range(L_i - 1))
        return 1.0 - math.exp(-xi) * s

    @staticmethod
    def bler_high_snr(gamma_th: float, gamma_det: float,
                      gamma_bar_NLOS: float, L_i: int) -> float:
        """High-SNR BLER approximation — Proposition 2 / Eq. (49)."""
        if gamma_th <= gamma_det:
            return 0.0
        if gamma_bar_NLOS <= 1e-30:
            return 1.0
        xi = (gamma_th - gamma_det) / gamma_bar_NLOS
        a = L_i - 1
        if a <= 0:
            return 1.0 - math.exp(-min(xi, 700))
        if xi > 700:
            return 0.0
        return xi ** a * math.exp(-xi) / math.factorial(a)

    @staticmethod
    def bler_rayleigh(gamma_th: float, gamma_bar_br: float,
                      L_i: int) -> float:
        """Pure Rayleigh fading BLER (no LOS) — Eq. (51)."""
        if gamma_bar_br <= 0:
            return 1.0
        x = gamma_th / gamma_bar_br
        return regularised_lower_inc_gamma(L_i, x)

    @staticmethod
    def piecewise_conditional_bler(gamma: float, psi: float, beta: float,
                                   n_c: int) -> float:
        """Piecewise linear approximation of conditional BLER — Eq. (44)."""
        phi_val = psi - 1.0 / (2.0 * beta * np.sqrt(n_c))
        delta = psi + 1.0 / (2.0 * beta * np.sqrt(n_c))
        if gamma <= phi_val:
            return 1.0
        elif gamma >= delta:
            return 0.0
        else:
            return 0.5 - beta * np.sqrt(n_c) * (gamma - psi)


# ============================================================================
#  Semi-analytical BLER with diversity transforms (Section III)
# ============================================================================

def bler_no_diversity(gamma_per_sub: np.ndarray, N: int, K: int) -> float:
    """
    Semi-analytical BLER without diversity transform — Section III.

    Two complementary models capture different regimes:
    - Harmonic-mean model (N): the harmonic mean is dominated by the weakest
      subchannel, capturing the vulnerability of no-diversity schemes to deep
      fades and blockage. At low-to-moderate SNR, a single blocked or deeply
      faded subchannel drags the harmonic mean below the capacity threshold.
    - Per-subchannel average (N/n_s): at high SNR, each blocked subchannel
      contributes a fractional BLER proportional to its share of coded bits,
      providing a natural floor above PPV_fix.

    The maximum of the two models is used.

    Args:
        gamma_per_sub: (n_s,) per-subchannel SNR (linear)
        N: polar code block length
        K: information bits

    Returns:
        BLER at this channel realization.
    """
    n_s = len(gamma_per_sub)
    N_sub = N // n_s          # coded bits per subchannel (~43 for N=256, n_s=6)
    R = K / N                 # coding rate

    # Model A: harmonic mean SNR with full blocklength N
    # Harmonic mean = n_s / Σ(1/γ_j) — sensitive to weak subchannels.
    # Without diversity protection, the weakest subchannel limits performance.
    harm_inv = float(np.sum(1.0 / np.maximum(gamma_per_sub, 1e-10)))
    gamma_harm = n_s / harm_inv
    p_harm = conditional_bler(gamma_harm, N, R)

    # Model B: per-subchannel average with short blocklength N_sub
    # Captures high-SNR regime where blockage causes fractional BLER
    p_avg = 0.0
    for j in range(n_s):
        p_avg += conditional_bler(gamma_per_sub[j], N_sub, R)
    p_avg /= n_s

    return max(p_harm, p_avg)


def bler_fixed_diversity(gamma_per_sub: np.ndarray, erased_mask: np.ndarray,
                         k_c: int, N: int, K: int) -> float:
    """
    Semi-analytical BLER with fixed diversity transform (G_FIX) — Section III.

    The MDS diversity transform allows recovery from up to (n_s - k_c) erased
    subchannels. The effective SNR is the mean SNR of surviving (non-erased)
    subchannels, since the decoder operates on the combined information from
    surviving subchannels after MDS recovery.

    Returns 1.0 if fewer than k_c subchannels survive (unrecoverable).

    Args:
        gamma_per_sub: (n_s,) per-subchannel SNR (linear)
        erased_mask: (n_s,) bool, True = erased
        k_c: MDS code dimension (minimum surviving subchannels needed)
        N: polar code block length
        K: information bits

    Returns:
        BLER at this channel realization.
    """
    surviving = gamma_per_sub[~erased_mask]
    if len(surviving) < k_c:
        return 1.0
    gamma_eff = float(np.mean(surviving))
    R = K / N
    return conditional_bler(gamma_eff, N, R)


def bler_adaptive_diversity(gamma_per_sub: np.ndarray, erased_mask: np.ndarray,
                            block_weights: np.ndarray, k_c: int,
                            N: int, K: int) -> float:
    """
    Semi-analytical BLER with adaptive diversity transform (Algorithm 2) — Section III.

    Algorithm 2 selects a transform T that maximizes the quality metric
    Q = sum_j eta_j * w_j, placing more coded bits on stronger subchannels.
    The effective SNR is the quality-weighted mean of surviving subchannel SNRs,
    reflecting the optimized bit-to-subchannel mapping.

    Returns 1.0 if fewer than k_c subchannels survive.

    Args:
        gamma_per_sub: (n_s,) per-subchannel SNR (linear)
        erased_mask: (n_s,) bool, True = erased
        block_weights: (n_s,) block Hamming weights from Algorithm 2's G_DIV
        k_c: MDS code dimension
        N: polar code block length
        K: information bits

    Returns:
        BLER at this channel realization.
    """
    surviving_idx = np.where(~erased_mask)[0]
    if len(surviving_idx) < k_c:
        return 1.0
    w = block_weights[surviving_idx]
    g = gamma_per_sub[surviving_idx]
    w_sum = np.sum(w)
    if w_sum < 1e-10:
        return 1.0
    gamma_eff = float(np.sum(w * g) / w_sum)
    R = K / N
    return conditional_bler(gamma_eff, N, R)
