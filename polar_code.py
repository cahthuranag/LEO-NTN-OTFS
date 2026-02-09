"""
Polar Code — Construction, Encoding, SC Decoding & BLER Models
===============================================================
Provides:
  1. PPV finite-blocklength BLER approximation (analytical bounds)
  2. PolarCode class: Gaussian-Approximation frozen set design,
     recursive butterfly encoding, and SC (min-sum) decoding.

Author: Research implementation
Date: 2026-02-07
"""

import numpy as np
from typing import Tuple
from dataclasses import dataclass
from scipy.special import erfc
from scipy.stats import norm


@dataclass
class PolarCodeParams:
    N: int           # Block length
    K: int           # Info bits
    design_snr_dB: float = 1.0

    def __post_init__(self):
        self.R = self.K / self.N  # Code rate
        self.n = int(np.log2(self.N)) if self.N > 0 else 0


def Q_func(x: float) -> float:
    """Q-function: tail probability of standard normal."""
    return 0.5 * erfc(x / np.sqrt(2))


def capacity_awgn(snr_linear: float) -> float:
    """AWGN channel capacity in bits/symbol."""
    return 0.5 * np.log2(1 + snr_linear)


def channel_dispersion(snr_linear: float) -> float:
    """Channel dispersion for AWGN."""
    if snr_linear <= 0:
        return 0.0
    return 0.5 * (1 - 1/(1 + snr_linear)**2) * (np.log2(np.e))**2


def finite_blocklength_bler(snr_linear: float, N: int, K: int) -> float:
    """
    Finite blocklength BLER approximation.

    Based on Polyanskiy-Poor-Verdu normal approximation:
    ε ≈ Q((C - R)*sqrt(N/V))

    For Polar codes with SC decoding, there's additional loss
    compared to optimal decoding (~0.5-1 dB).
    """
    if snr_linear <= 0:
        return 1.0

    R = K / N
    C = capacity_awgn(snr_linear)
    V = channel_dispersion(snr_linear)

    if V <= 0:
        return 1.0 if R > C else 0.0

    # Normal approximation
    if C > R:
        arg = (C - R) * np.sqrt(N / V)
        bler = Q_func(arg)
    else:
        bler = 1.0

    return np.clip(bler, 0, 1)


def polar_sc_bler(snr_dB: float, N: int, K: int,
                   sc_loss_dB: float = 0.5) -> float:
    """
    Polar code BLER with SC decoding.

    SC decoder has ~0.5 dB loss compared to ML decoding.
    SCL decoder recovers most of this loss.

    Args:
        snr_dB: Channel SNR in dB
        N: Block length
        K: Information bits
        sc_loss_dB: SC decoder loss (default 0.5 dB)
    """
    # Apply SC decoder loss
    effective_snr_dB = snr_dB - sc_loss_dB
    snr_linear = 10 ** (effective_snr_dB / 10)

    return finite_blocklength_bler(snr_linear, N, K)


class PolarCodedSystem:
    """
    Polar coded system model.

    Uses theoretical BLER approximation for efficiency.
    """

    def __init__(self, N: int, K: int, design_snr_dB: float = 1.0):
        self.N = N
        self.K = K
        self.params = PolarCodeParams(N, K, design_snr_dB)
        self.R = K / N

    def bler_awgn(self, snr_dB: float, use_scl: bool = False) -> float:
        """
        Compute BLER for AWGN channel.

        Args:
            snr_dB: Channel SNR
            use_scl: If True, assume SCL decoder (less loss)
        """
        sc_loss = 0.2 if use_scl else 0.5
        return polar_sc_bler(snr_dB, self.N, self.K, sc_loss)

    def bler_fading(self, snr_dB: float, fading_type: str = "awgn") -> float:
        """
        Compute BLER considering fading.

        For Rayleigh fading, average BLER over SNR distribution.
        """
        if fading_type == "awgn":
            return self.bler_awgn(snr_dB)

        elif fading_type == "rayleigh":
            # Average over Rayleigh fading
            snr_lin = 10 ** (snr_dB / 10)
            # Use Gauss-Hermite quadrature for integration
            n_points = 20
            x, w = np.polynomial.hermite.hermgauss(n_points)

            bler_avg = 0.0
            for xi, wi in zip(x, w):
                # Map to exponential distribution
                gamma = snr_lin * np.exp(np.sqrt(2) * xi)
                snr_i = 10 * np.log10(max(gamma, 1e-10))
                bler_avg += wi * self.bler_awgn(snr_i)

            return np.clip(bler_avg / np.sqrt(np.pi), 0, 1)

        else:
            return self.bler_awgn(snr_dB)

    def simulate_mc(self, snr_dB: float, n_trials: int,
                    rng: np.random.Generator = None) -> float:
        """
        Monte Carlo simulation using BLER model.

        Generates random channel realizations and computes BLER.
        """
        if rng is None:
            rng = np.random.default_rng()

        snr_lin = 10 ** (snr_dB / 10)
        errors = 0

        for _ in range(n_trials):
            # Rayleigh fading gain
            h2 = rng.exponential(1.0)
            gamma = snr_lin * h2
            gamma_dB = 10 * np.log10(max(gamma, 1e-10))

            # BLER at this realization
            bler = self.bler_awgn(gamma_dB)

            # Random block error
            if rng.random() < bler:
                errors += 1

        return errors / n_trials


def finite_blocklength_bler_vec(snr_linear, N, K):
    """
    Vectorized finite blocklength BLER approximation.

    All arguments can be numpy arrays (broadcastable).
    Based on Polyanskiy-Poor-Verdu normal approximation.
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


def polar_sc_bler_vec(snr_dB, N, K, sc_loss_dB=0.5):
    """
    Vectorized Polar code BLER with SC decoding.

    snr_dB, N, K can be numpy arrays (broadcastable).
    """
    snr_dB = np.asarray(snr_dB, dtype=float)
    snr_linear = 10 ** ((snr_dB - sc_loss_dB) / 10)
    return finite_blocklength_bler_vec(snr_linear, N, K)


# ===================================================================
#  PolarCode — actual construction, encoding and SC decoding
# ===================================================================

def _phi_ga(x):
    """Gaussian-Approximation helper phi(x) ≈ exp(-0.4527*x^0.86 + 0.0218)."""
    x = np.asarray(x, dtype=float)
    out = np.where(x > 0, np.exp(-0.4527 * np.power(np.maximum(x, 1e-30), 0.86) + 0.0218), 1.0)
    return np.clip(out, 0, 1)


def _phi_inv_ga(y):
    """Inverse of phi: given y = phi(x), return x."""
    y = np.asarray(y, dtype=float)
    y = np.clip(y, 1e-12, 1.0 - 1e-12)
    return np.power((0.0218 - np.log(y)) / 0.4527, 1.0 / 0.86)


class PolarCode:
    """
    Polar code with Gaussian-Approximation frozen set design,
    recursive butterfly encoding, and successive-cancellation decoding.
    """

    def __init__(self, N, K, design_snr_dB=1.0):
        assert N & (N - 1) == 0 and N > 0, "N must be a power of 2"
        assert 0 < K <= N
        self.N = N
        self.K = K
        self.n = int(np.log2(N))
        self.design_snr_dB = design_snr_dB

        # --- Gaussian-Approximation channel reliability ---
        design_snr_lin = 10 ** (design_snr_dB / 10.0)
        mu = np.full(N, 4.0 * design_snr_lin)  # initial LLR variance

        for stage in range(self.n):
            half = 1 << stage
            mu_new = np.empty(N)
            for i in range(0, N, 2 * half):
                a = mu[i:i + half]
                b = mu[i + half:i + 2 * half]
                # W⁻ (check): phi_inv(1 - (1-phi(a))*(1-phi(b)))
                pa, pb = _phi_ga(a), _phi_ga(b)
                mu_new[i:i + half] = _phi_inv_ga(1.0 - (1.0 - pa) * (1.0 - pb))
                # W⁺ (variable): a + b
                mu_new[i + half:i + 2 * half] = a + b
            mu = mu_new

        # Select K best (highest reliability) as info positions
        sorted_idx = np.argsort(mu)[::-1]
        info_set = set(sorted_idx[:K].tolist())
        self.frozen_mask = np.array([i not in info_set for i in range(N)], dtype=bool)
        self.info_indices = np.sort(np.array([i for i in range(N) if not self.frozen_mask[i]]))

    # ---------- encoding ----------
    def encode(self, info_bits):
        """
        Polar encode: place info bits at info positions, 0 at frozen,
        then apply butterfly transform u → x = u F^⊗n (mod 2).

        Args:
            info_bits: (K,) array of {0,1}
        Returns:
            x: (N,) coded bits
        """
        u = np.zeros(self.N, dtype=np.int8)
        u[self.info_indices] = info_bits
        return self._butterfly(u)

    @staticmethod
    def _butterfly(u):
        """Arikan butterfly: x = u F^{⊗n} (mod 2). Works for any power-of-2 length."""
        n = len(u)
        x = u.copy()
        step = 1
        while step < n:
            for i in range(0, n, 2 * step):
                x[i:i + step] ^= x[i + step:i + 2 * step]
            step <<= 1
        return x

    # ---------- SC decoding ----------
    def decode(self, channel_llr):
        """
        Successive-cancellation decoder (min-sum), recursive.

        Uses in-place output array to avoid concatenation overhead.

        Args:
            channel_llr: (N,) LLR values  (positive = more likely 0)
        Returns:
            info_bits_hat: (K,) decoded info bits
        """
        u_hat = np.zeros(self.N, dtype=np.int8)
        self._sc_recurse(np.asarray(channel_llr, dtype=np.float64),
                         self.frozen_mask, u_hat, 0)
        return u_hat[self.info_indices]

    def _sc_recurse(self, llr, frozen, out, offset):
        """Recursive SC decode writing results into out[offset:offset+len(llr)]."""
        n = len(llr)
        if n == 1:
            if frozen[0]:
                out[offset] = 0
            else:
                out[offset] = 0 if llr[0] >= 0 else 1
            return

        half = n >> 1
        la = llr[:half]
        lb = llr[half:]

        # f-node (check): min-sum
        abs_a = np.abs(la)
        abs_b = np.abs(lb)
        llr_f = np.minimum(abs_a, abs_b)
        signs = np.signbit(la) ^ np.signbit(lb)
        llr_f[signs] = -llr_f[signs]

        self._sc_recurse(llr_f, frozen[:half], out, offset)

        # partial-sum encoding of left half (butterfly on decided bits)
        s = self._butterfly(out[offset:offset + half].copy())

        # g-node (variable)
        llr_g = la * (1 - 2 * s.astype(np.float64)) + lb

        self._sc_recurse(llr_g, frozen[half:], out, offset + half)

    # ---------- convenience methods ----------
    def encode_and_transmit(self, info_bits, snr_linear, rng):
        """
        Encode → BPSK modulate → AWGN channel → compute LLRs.

        Args:
            info_bits: (K,) info bits
            snr_linear: scalar Es/N0
            rng: numpy Generator
        Returns:
            channel_llr: (N,) LLR values
        """
        x = self.encode(info_bits)
        # BPSK: 0 → +1, 1 → −1
        bpsk = 1.0 - 2.0 * x.astype(np.float64)
        sigma2 = 1.0 / (2.0 * max(snr_linear, 1e-10))
        noise = rng.normal(0, np.sqrt(sigma2), self.N)
        y = bpsk + noise
        # LLR = 2*y / sigma^2 = 4*snr*y
        channel_llr = 4.0 * snr_linear * y
        return channel_llr

    def decode_check(self, info_bits, channel_llr):
        """
        SC decode and compare with original info bits.

        Returns:
            True if block error (decoded != original).
        """
        decoded = self.decode(channel_llr)
        return not np.array_equal(decoded, info_bits)

    def transmit_per_bit_snr(self, info_bits, snr_per_bit, rng):
        """
        Encode → BPSK → per-bit AWGN (each coded bit has its own SNR) → LLRs.

        Args:
            info_bits: (K,) info bits
            snr_per_bit: (N,) linear SNR per coded bit
            rng: numpy Generator
        Returns:
            channel_llr: (N,) LLR values
        """
        x = self.encode(info_bits)
        bpsk = 1.0 - 2.0 * x.astype(np.float64)
        sigma2 = 1.0 / (2.0 * np.maximum(snr_per_bit, 1e-10))
        noise = rng.normal(0, 1.0, self.N) * np.sqrt(sigma2)
        y = bpsk + noise
        channel_llr = 4.0 * snr_per_bit * y
        return channel_llr


if __name__ == "__main__":
    print("Polar Code — Construction, Encoding, SC Decoding")
    print("=" * 55)

    N, K = 256, 128
    pc = PolarCode(N, K, design_snr_dB=1.0)
    print(f"\nPolar({N},{K}), design SNR = 1.0 dB")
    print(f"  Frozen bits: {int(np.sum(pc.frozen_mask))}")
    print(f"  Info  bits:  {len(pc.info_indices)}")

    # Round-trip test (noiseless)
    rng = np.random.default_rng(42)
    info = rng.integers(0, 2, size=K).astype(np.int8)
    llr_noiseless = pc.encode_and_transmit(info, snr_linear=1e6, rng=rng)
    decoded = pc.decode(llr_noiseless)
    print(f"\n  Round-trip (noiseless): {'PASS' if np.array_equal(decoded, info) else 'FAIL'}")

    # BLER at a few SNR points via MC
    print(f"\n  MC BLER test (1000 trials):")
    for snr_dB in [0, 2, 4, 6]:
        snr_lin = 10 ** (snr_dB / 10)
        errors = 0
        n_trials = 1000
        for _ in range(n_trials):
            ib = rng.integers(0, 2, size=K).astype(np.int8)
            llr = pc.encode_and_transmit(ib, snr_lin, rng)
            if pc.decode_check(ib, llr):
                errors += 1
        print(f"    SNR = {snr_dB} dB: BLER = {errors/n_trials:.4f}")
