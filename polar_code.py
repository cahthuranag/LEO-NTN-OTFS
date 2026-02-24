"""
Polar Code — Construction, Encoding & SCL Decoding (Section V)
===============================================================
Provides:
  PolarCode class: Gaussian-Approximation frozen set design,
  recursive butterfly encoding, and SC / SCL decoding.

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

    def __init__(self, N, K, design_snr_dB=1.0, list_size=8):
        assert N & (N - 1) == 0 and N > 0, "N must be a power of 2"
        assert 0 < K <= N
        self.N = N
        self.K = K
        self.n = int(np.log2(N))
        self.design_snr_dB = design_snr_dB
        self.list_size = list_size

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

    # ---------- decoding ----------
    def decode(self, channel_llr):
        """
        Decode channel LLRs using SC (list_size=1) or SCL decoder.

        Args:
            channel_llr: (N,) LLR values  (positive = more likely 0)
        Returns:
            info_bits_hat: (K,) decoded info bits
        """
        if self.list_size > 1:
            return self._decode_scl(channel_llr)
        return self._decode_sc(channel_llr)

    # ---------- SC decoding ----------
    def _decode_sc(self, channel_llr):
        """Successive-cancellation decoder (min-sum), recursive."""
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

    # ---------- SCL decoding ----------
    def _decode_scl(self, channel_llr):
        """
        Successive Cancellation List (SCL) decoder with Tal-Vardy path metric.

        Maintains L candidate paths through the SC tree. At each info bit,
        forks all paths and prunes to the L best by accumulated path metric.

        Args:
            channel_llr: (N,) LLR values  (positive = more likely 0)
        Returns:
            info_bits_hat: (K,) decoded info bits from best path
        """
        N = self.N
        n = self.n
        L = self.list_size

        # Per-path data structures
        llr = np.zeros((L, n + 1, N))                # intermediate LLRs
        ps = np.zeros((L, n + 1, N), dtype=np.int8)  # partial sums
        u = np.zeros((L, N), dtype=np.int8)           # decided bits
        pm = np.full(L, np.inf)                       # path metrics
        pm[0] = 0.0
        n_active = 1

        # Channel LLRs at top stage for path 0
        llr[0, n, :] = np.asarray(channel_llr, dtype=np.float64)

        for i in range(N):
            # --- LLR propagation from top stage down to stage 0 ---
            top = n - 1 if i == 0 else (i & -i).bit_length() - 1

            for s in range(top, -1, -1):
                h = 1 << s
                pstart = (i >> (s + 1)) << (s + 1)

                if (i >> s) & 1 == 0:
                    # f-node: min-sum
                    a = llr[:n_active, s + 1, pstart:pstart + h]
                    b = llr[:n_active, s + 1, pstart + h:pstart + 2 * h]
                    abs_a, abs_b = np.abs(a), np.abs(b)
                    res = np.minimum(abs_a, abs_b)
                    signs = np.signbit(a) ^ np.signbit(b)
                    llr[:n_active, s, pstart:pstart + h] = np.where(
                        signs, -res, res)
                else:
                    # g-node: combine with partial sums
                    a = llr[:n_active, s + 1, pstart:pstart + h]
                    b = llr[:n_active, s + 1, pstart + h:pstart + 2 * h]
                    c = ps[:n_active, s, pstart:pstart + h].astype(np.float64)
                    llr[:n_active, s, pstart + h:pstart + 2 * h] = (
                        a * (1.0 - 2.0 * c) + b)

            # --- Decision ---
            llr_i = llr[:n_active, 0, i]

            if self.frozen_mask[i]:
                # Frozen bit: all paths set bit=0
                u[:n_active, i] = 0
                ps[:n_active, 0, i] = 0
                pm[:n_active] += np.maximum(0.0, -llr_i)
            else:
                # Info bit: fork each path into bit=0 and bit=1
                pm_0 = pm[:n_active] + np.maximum(0.0, -llr_i)
                pm_1 = pm[:n_active] + np.maximum(0.0, llr_i)
                cand_pm = np.concatenate([pm_0, pm_1])
                cand_parent = np.tile(np.arange(n_active), 2)
                cand_bit = np.concatenate([
                    np.zeros(n_active, dtype=np.int8),
                    np.ones(n_active, dtype=np.int8)])

                # Keep L best candidates
                n_keep = min(2 * n_active, L)
                order = np.argsort(cand_pm)[:n_keep]
                parents = cand_parent[order]
                bits = cand_bit[order]

                # Copy parent states (fancy indexing creates new arrays)
                new_llr = llr[parents].copy()
                new_ps = ps[parents].copy()
                new_u = u[parents].copy()
                new_pm = cand_pm[order]

                new_u[:, i] = bits
                new_ps[:, 0, i] = bits

                llr[:n_keep] = new_llr
                ps[:n_keep] = new_ps
                u[:n_keep] = new_u
                pm[:n_keep] = new_pm
                n_active = n_keep

            # --- Partial sum propagation upward ---
            ii, s = i, 0
            while ii & 1:
                h = 1 << s
                bstart = (ii >> 1) << (s + 1)
                left = ps[:n_active, s, bstart:bstart + h]
                right = ps[:n_active, s, bstart + h:bstart + 2 * h]
                ps[:n_active, s + 1, bstart:bstart + h] = left ^ right
                ps[:n_active, s + 1, bstart + h:bstart + 2 * h] = right
                ii >>= 1
                s += 1

        # Return info bits from path with lowest metric
        best = np.argmin(pm[:n_active])
        return u[best, self.info_indices]

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
        Decode and compare with original info bits.

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
    print("Polar Code — Construction, Encoding, SC/SCL Decoding")
    print("=" * 55)

    N, K = 256, 128

    # --- SCL(L=8) round-trip test ---
    pc_scl = PolarCode(N, K, design_snr_dB=1.0, list_size=8)
    print(f"\nPolar({N},{K}), design SNR = 1.0 dB, SCL L={pc_scl.list_size}")
    print(f"  Frozen bits: {int(np.sum(pc_scl.frozen_mask))}")
    print(f"  Info  bits:  {len(pc_scl.info_indices)}")

    rng = np.random.default_rng(42)
    info = rng.integers(0, 2, size=K).astype(np.int8)
    llr_noiseless = pc_scl.encode_and_transmit(info, snr_linear=1e6, rng=rng)
    decoded = pc_scl.decode(llr_noiseless)
    print(f"\n  Round-trip SCL (noiseless): {'PASS' if np.array_equal(decoded, info) else 'FAIL'}")

    # --- SC(L=1) round-trip test ---
    pc_sc = PolarCode(N, K, design_snr_dB=1.0, list_size=1)
    rng2 = np.random.default_rng(42)
    info2 = rng2.integers(0, 2, size=K).astype(np.int8)
    llr_noiseless2 = pc_sc.encode_and_transmit(info2, snr_linear=1e6, rng=rng2)
    decoded2 = pc_sc.decode(llr_noiseless2)
    print(f"  Round-trip SC  (noiseless): {'PASS' if np.array_equal(decoded2, info2) else 'FAIL'}")

    # --- BLER comparison: SC vs SCL ---
    print(f"\n  MC BLER comparison (1000 trials):")
    print(f"    {'SNR':>5s}  {'SC':>8s}  {'SCL-8':>8s}")
    for snr_dB in [0, 2, 4]:
        snr_lin = 10 ** (snr_dB / 10)
        errors_sc = 0
        errors_scl = 0
        n_trials = 1000
        for _ in range(n_trials):
            ib = rng.integers(0, 2, size=K).astype(np.int8)
            llr = pc_scl.encode_and_transmit(ib, snr_lin, rng)
            if pc_sc.decode_check(ib, llr):
                errors_sc += 1
            if pc_scl.decode_check(ib, llr):
                errors_scl += 1
        print(f"    {snr_dB:3d}dB  {errors_sc/n_trials:.4f}  {errors_scl/n_trials:.4f}")
