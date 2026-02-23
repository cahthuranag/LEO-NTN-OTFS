"""
Polar Code — Construction, Encoding & SC Decoding (Section V)
=============================================================
Provides:
  PolarCode class: Gaussian-Approximation frozen set design,
  recursive butterfly encoding, and SC (min-sum) decoding.

Author: Research implementation
Date: 2026-02-07
"""

import numpy as np
from dataclasses import dataclass


@dataclass
class PolarCodeParams:
    N: int           # Block length
    K: int           # Info bits
    design_snr_dB: float = 1.0

    def __post_init__(self):
        self.R = self.K / self.N  # Code rate
        self.n = int(np.log2(self.N)) if self.N > 0 else 0


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
        Encode → bipolar mapping → AWGN channel → compute LLRs.

        Args:
            info_bits: (K,) info bits
            snr_linear: scalar Es/N0
            rng: numpy Generator
        Returns:
            channel_llr: (N,) LLR values
        """
        x = self.encode(info_bits)
        s = 1.0 - 2.0 * x.astype(np.float64)  # bipolar: 0→+1, 1→-1
        sigma2 = 1.0 / (2.0 * max(snr_linear, 1e-10))
        noise = rng.normal(0, np.sqrt(sigma2), self.N)
        y = s + noise
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
        Encode → per-bit AWGN-equivalent channel → LLRs.

        Each coded bit is transmitted through an independent AWGN-equivalent
        channel (modelling the effective channel after OTFS demodulation)
        with its own SNR.

        Args:
            info_bits: (K,) info bits
            snr_per_bit: (N,) linear SNR per coded bit
            rng: numpy Generator
        Returns:
            channel_llr: (N,) LLR values
        """
        x = self.encode(info_bits)
        s = 1.0 - 2.0 * x.astype(np.float64)  # bipolar mapping
        sigma2 = 1.0 / (2.0 * np.maximum(snr_per_bit, 1e-10))
        noise = rng.normal(0, 1.0, self.N) * np.sqrt(sigma2)
        y = s + noise
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
