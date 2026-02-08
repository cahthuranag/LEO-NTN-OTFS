"""
Polar Code BLER Model
=====================
Uses theoretical approximation for Polar code BLER based on:
- Finite blocklength bounds
- SC decoder error probability model

For accurate simulations, a full SC implementation would be needed.
This model provides reasonable approximation for system-level comparison.

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


def demo_bler_curve():
    """Demo BLER vs SNR curve."""
    import matplotlib.pyplot as plt

    N, K = 256, 128
    system = PolarCodedSystem(N, K)

    snr_dB = np.linspace(-2, 8, 21)
    bler_awgn = [system.bler_awgn(s) for s in snr_dB]
    bler_scl = [system.bler_awgn(s, use_scl=True) for s in snr_dB]

    plt.figure(figsize=(10, 7))
    plt.semilogy(snr_dB, bler_awgn, 'o-', label='SC Decoder')
    plt.semilogy(snr_dB, bler_scl, 's--', label='SCL Decoder')
    plt.xlabel('SNR (dB)')
    plt.ylabel('BLER')
    plt.title(f'Polar({N},{K}) BLER vs SNR')
    plt.legend()
    plt.grid(True, which='both')
    plt.ylim(1e-5, 1)
    plt.savefig('polar_bler_curve.png', dpi=150)
    plt.show()


if __name__ == "__main__":
    print("Polar Code BLER Model")
    print("=" * 50)

    N, K = 256, 128
    system = PolarCodedSystem(N, K)

    print(f"\nPolar({N},{K}), Rate = {K/N:.2f}")
    print("\nBLER vs SNR (AWGN, SC decoder):")

    for snr in range(-2, 9):
        bler = system.bler_awgn(snr)
        print(f"  SNR = {snr:3d} dB: BLER = {bler:.4e}")

    print("\nRunning demo plot...")
    demo_bler_curve()
