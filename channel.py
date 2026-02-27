"""
LEO NTN OTFS Downlink Channel Model
====================================
Implementation of the channel model from:
"Diversity Transforms for Extreme-Reliability LEO NTN OTFS Downlinks"

This module implements:
- TDL-D doubly selective fading (Rician per-tap)
- Large-scale path loss: FSPL, gaseous absorption (ITU-R P.676),
  cloud/fog (ITU-R P.840), rain (ITU-R P.618), blockage/shadowing (3GPP NTN)
- Transparent (bent-pipe) satellite relay with cascaded SNR
- OTFS modulation / demodulation (ISFFT/SFFT + Heisenberg/Wigner)
- MRC diversity combining across delay-Doppler taps
- Finite-blocklength BLER (Polyanskiy normal approximation + closed-form)
- Ephemeris-driven outage prediction

Author : Auto-generated from LaTeX specification
Date   : 2026-02-06
"""

import numpy as np
import math
from dataclasses import dataclass, field
from typing import List, Tuple, Optional
import warnings

from fbl_analysis import (
    FiniteBlocklengthBLER, q_function, q_inv,
    regularised_lower_inc_gamma,
)


# ============================================================================
#  Constants
# ============================================================================
SPEED_OF_LIGHT = 3e8          # m/s
EARTH_RADIUS   = 6371e3       # m
BOLTZMANN      = 1.380649e-23 # J/K


# ============================================================================
#  Helper functions
# ============================================================================


def db_to_linear(x_db: float) -> float:
    return 10.0 ** (x_db / 10.0)


def linear_to_db(x_lin: float) -> float:
    return 10.0 * np.log10(np.maximum(x_lin, 1e-30))



# ============================================================================
#  Data classes for system configuration
# ============================================================================

@dataclass
class OrbitalParams:
    """LEO satellite orbital parameters."""
    h_orb: float = 600e3          # orbital altitude [m]
    inclination_deg: float = 53.0 # orbital inclination [deg]


@dataclass
class LinkBudgetParams:
    """Link budget parameters for a single link segment."""
    P_tx_dBm: float = 43.0       # transmit power [dBm]
    G_tx_dBi: float = 30.0       # transmit antenna gain [dBi]
    G_rx_dBi: float = 0.0        # receive antenna gain [dBi]
    f_c_Hz: float = 2e9          # carrier frequency [Hz]
    noise_figure_dB: float = 7.0 # receiver noise figure [dB]
    bandwidth_Hz: float = 20e6   # system bandwidth [Hz]
    T_sys_K: float = 290.0       # system temperature [K]


@dataclass
class AtmosphericParams:
    """Atmospheric / environmental propagation parameters."""
    P_atm_hPa: float = 1013.25   # atmospheric pressure [hPa]
    T_atm_K: float = 288.15      # temperature [K]
    rho_wv: float = 7.5          # water-vapour density [g/m³]
    H_gas_m: float = 6000.0      # equivalent gaseous thickness [m]
    LWC: float = 0.05            # liquid-water content [g/m³]
    H_cf_m: float = 1000.0       # cloud/fog layer thickness [m]
    rain_rate_mmh: float = 0.0   # rain rate [mm/h]
    H_rain_m: float = 3000.0     # rain height [m]
    h_UE_m: float = 0.0          # UE altitude [m]
    rain_reduction: float = 1.0  # path reduction factor r(t)
    kappa: float = 0.0101        # ITU rain attenuation coeff
    beta: float = 1.276          # ITU rain attenuation exponent


@dataclass
class EnvironmentParams:
    """3GPP NTN blockage / LOS probability parameters (Table in TR 38.811)."""
    env_type: str = "suburban"    # 'urban', 'suburban', 'rural'
    epsilon_min_deg: float = 10.0
    # Environment-dependent defaults set in __post_init__
    epsilon_max_deg: float = None
    theta_deg: float = None
    P_base: float = None
    mu_sh_dB: float = 3.0        # mean shadowing margin [dB]
    sigma_sh_dB: float = 4.0     # shadow fading std [dB]
    L_deep_dB: float = 30.0      # deep-blockage loss [dB]

    def __post_init__(self):
        _defaults = {
            "urban":    {"eps_max": 60.0, "theta": 15.0, "P_base": 0.2},
            "suburban": {"eps_max": 50.0, "theta": 20.0, "P_base": 0.5},
            "rural":    {"eps_max": 40.0, "theta": 25.0, "P_base": 0.8},
        }
        d = _defaults.get(self.env_type, _defaults["suburban"])
        if self.epsilon_max_deg is None:
            self.epsilon_max_deg = d["eps_max"]
        if self.theta_deg is None:
            self.theta_deg = d["theta"]
        if self.P_base is None:
            self.P_base = d["P_base"]


@dataclass
class TDLProfile:
    """Tapped-Delay-Line Doppler (TDL-D) channel profile for one link."""
    num_taps: int = 4                           # L
    relative_powers_dB: List[float] = None      # per-tap powers [dB]
    delays_ns: List[float] = None               # per-tap delays [ns]
    K_factor_dB: float = 10.0                   # Rician K-factor of LOS tap [dB]
    K_factors_per_tap_dB: List[float] = None    # per-tap K; None → LOS only

    def __post_init__(self):
        if self.relative_powers_dB is None:
            # Default: power decays 3 dB per tap
            self.relative_powers_dB = [-3.0 * i for i in range(self.num_taps)]
        if self.delays_ns is None:
            self.delays_ns = [100.0 * i for i in range(self.num_taps)]
        # Normalise relative powers to sum to 1 in linear scale
        lin = np.array([db_to_linear(p) for p in self.relative_powers_dB])
        self.relative_powers_linear = lin / lin.sum()  # P_{i,kl,p}
        # Per-tap K-factors
        if self.K_factors_per_tap_dB is None:
            # LOS tap gets the given K; NLOS taps are Rayleigh (K=0 → -inf dB)
            self.K_factors_per_tap_dB = [self.K_factor_dB] + \
                                        [-np.inf] * (self.num_taps - 1)
        self.K_factors_linear = np.array(
            [db_to_linear(k) if np.isfinite(k) else 0.0
             for k in self.K_factors_per_tap_dB]
        )


@dataclass
class OTFSParams:
    """OTFS modulation grid parameters."""
    N_D: int = 16       # number of Doppler bins
    M_D: int = 64       # number of delay bins
    delta_f_Hz: float = 15e3  # subcarrier spacing [Hz]

    @property
    def T_sym(self) -> float:
        """OTFS symbol duration satisfying Δf·T_sym = 1."""
        return 1.0 / self.delta_f_Hz

    @property
    def blocklength(self) -> int:
        """n_c = N_D × M_D."""
        return self.N_D * self.M_D


# ============================================================================
#  Large-Scale Path Loss Model  (Eqs. 6–18)
# ============================================================================

class LargeScalePathLoss:
    """Compute total large-scale path loss for a single link."""

    def __init__(self, atm: AtmosphericParams, env: EnvironmentParams):
        self.atm = atm
        self.env = env

    # --- Eq. (8): Free-space path loss [dB] ---
    @staticmethod
    def fspl_dB(d_m: float, wavelength_m: float) -> float:
        return 20.0 * np.log10(4.0 * np.pi * d_m / wavelength_m)

    # --- Eq. (9): Slant range from elevation angle [m] ---
    @staticmethod
    def slant_range(epsilon_rad: float, h_orb: float,
                    R_E: float = EARTH_RADIUS) -> float:
        ratio = 1.0 + h_orb / R_E
        return R_E * (np.sqrt(ratio**2 - np.cos(epsilon_rad)**2)
                      - np.sin(epsilon_rad))

    # --- Eqs. (10–11): Gaseous absorption [dB] ---
    def gaseous_loss_dB(self, f_c_Hz: float, epsilon_rad: float) -> float:
        """Simplified gaseous specific attenuation (ITU-R P.676 approx.)."""
        f_GHz = f_c_Hz / 1e9
        # Simplified O₂ + H₂O model (valid ~1–30 GHz)
        gamma_O2  = 7.2e-3 + 6.0e-3 / (1.0 + (f_GHz / 60.0)**2)
        gamma_H2O = 0.05 + 0.0021 * self.atm.rho_wv
        gamma_gas = gamma_O2 + gamma_H2O  # dB/km
        path_km = (self.atm.H_gas_m / 1e3) / np.sin(np.maximum(epsilon_rad, 0.01))
        return gamma_gas * path_km

    # --- Eqs. (12–13): Cloud and fog attenuation [dB] ---
    def cloud_fog_loss_dB(self, f_c_Hz: float, epsilon_rad: float) -> float:
        f_GHz = f_c_Hz / 1e9
        K_l = 0.819 * f_GHz / (1.0 + (f_GHz / 400.0)**2)  # approx K_l
        gamma_cf = K_l * self.atm.LWC  # dB/km
        path_km = (self.atm.H_cf_m / 1e3) / np.sin(np.maximum(epsilon_rad, 0.01))
        return gamma_cf * path_km

    # --- Eqs. (14–15): Rain attenuation [dB] ---
    def rain_loss_dB(self, epsilon_rad: float) -> float:
        if self.atm.rain_rate_mmh <= 0:
            return 0.0
        gamma_rain = self.atm.kappa * self.atm.rain_rate_mmh ** self.atm.beta
        d_rain_km = ((self.atm.H_rain_m - self.atm.h_UE_m)
                     / np.sin(np.maximum(epsilon_rad, 0.01))
                     * self.atm.rain_reduction) / 1e3
        return gamma_rain * d_rain_km

    # --- Eq. (7): Aggregate atmospheric loss ---
    def atmospheric_loss_dB(self, f_c_Hz: float, epsilon_rad: float) -> float:
        return (self.gaseous_loss_dB(f_c_Hz, epsilon_rad)
                + self.cloud_fog_loss_dB(f_c_Hz, epsilon_rad)
                + self.rain_loss_dB(epsilon_rad))

    # --- Eqs. (16–18): Blockage & shadowing [dB] ---
    def los_probability(self, epsilon_deg: float) -> float:
        """3GPP NTN LOS probability model — Eq. (16)."""
        e = self.env
        if epsilon_deg >= e.epsilon_max_deg:
            return 1.0
        elif epsilon_deg >= e.epsilon_min_deg:
            return np.exp(-(e.epsilon_max_deg - epsilon_deg) / e.theta_deg)
        else:
            return e.P_base

    def sample_blockage_loss_dB(self, epsilon_deg: float,
                                rng: np.random.Generator = None) -> float:
        """Sample blockage/shadowing loss — Eq. (18)."""
        if rng is None:
            rng = np.random.default_rng()
        P_los = self.los_probability(epsilon_deg)
        B = rng.binomial(1, P_los)  # 1 = LOS, 0 = blocked
        if B == 1:
            X_sh = rng.normal(0, self.env.sigma_sh_dB)
            return self.env.mu_sh_dB + X_sh
        else:
            return self.env.L_deep_dB

    # --- Eq. (6): Total path loss [dB] ---
    def total_path_loss_dB(self, d_m: float, wavelength_m: float,
                           f_c_Hz: float, epsilon_rad: float,
                           epsilon_deg: float,
                           rng: np.random.Generator = None) -> float:
        PL_fs   = self.fspl_dB(d_m, wavelength_m)
        L_atm   = self.atmospheric_loss_dB(f_c_Hz, epsilon_rad)
        L_blk   = self.sample_blockage_loss_dB(epsilon_deg, rng)
        return PL_fs + L_atm + L_blk

    # --- Eq. (19): Linear large-scale gain ---
    def large_scale_gain_linear(self, PL_tot_dB: float) -> float:
        return db_to_linear(-PL_tot_dB)


# ============================================================================
#  Small-Scale Fading (TDL-D Rician)  (Eqs. 1–5)
# ============================================================================

class SmallScaleFading:
    """Generate time-varying TDL-D Rician fading coefficients."""

    def __init__(self, profile: TDLProfile):
        self.profile = profile

    def generate_tap_gains(
        self,
        t_vec: np.ndarray,
        f_D_Hz: np.ndarray,
        rng: np.random.Generator = None,
    ) -> np.ndarray:
        """
        Generate complex tap gains  g_{p}(t)  for each tap p — Eq. (3).

        Parameters
        ----------
        t_vec   : (N_t,) array of time instants [s]
        f_D_Hz  : (L,) array of per-tap Doppler shifts [Hz]
        rng     : optional random generator

        Returns
        -------
        g : (L, N_t) complex array of small-scale fading coefficients
        """
        if rng is None:
            rng = np.random.default_rng()
        L = self.profile.num_taps
        N_t = len(t_vec)
        g = np.zeros((L, N_t), dtype=complex)

        for p in range(L):
            K = self.profile.K_factors_linear[p]
            # LOS component
            los = np.sqrt(K / (K + 1)) * np.exp(1j * 2 * np.pi * f_D_Hz[p] * t_vec)
            # Diffuse component  w ~ CN(0,1)
            w = (rng.standard_normal(N_t) + 1j * rng.standard_normal(N_t)) / np.sqrt(2)
            nlos = np.sqrt(1.0 / (K + 1)) * w
            g[p, :] = los + nlos
        return g

    def channel_impulse_response(
        self,
        t_vec: np.ndarray,
        f_D_Hz: np.ndarray,
        G_LS_linear: float,
        rng: np.random.Generator = None,
    ) -> np.ndarray:
        """
        Full impulse response  α_{p}(t) = sqrt(G_LS · P_p) · g_p(t)  — Eq. (2).

        Returns
        -------
        alpha : (L, N_t) complex array
        """
        g = self.generate_tap_gains(t_vec, f_D_Hz, rng)
        P = self.profile.relative_powers_linear
        alpha = np.sqrt(G_LS_linear * P[:, None]) * g
        return alpha


# ============================================================================
#  SNR Computation (Eqs. 20–32)
# ============================================================================

class SNRCalculator:
    """Compute instantaneous and average SNR for each link and end-to-end."""

    @staticmethod
    def noise_power(bandwidth_Hz: float, noise_figure_dB: float,
                    T_sys_K: float = 290.0) -> float:
        """σ²_n = k_B · T_sys · B · NF_linear."""
        NF_lin = db_to_linear(noise_figure_dB)
        return BOLTZMANN * T_sys_K * bandwidth_Hz * NF_lin

    @staticmethod
    def average_snr(P_tx_dBm: float, G_tx_dBi: float, G_rx_dBi: float,
                    G_LS_linear: float, sigma_n_sq: float) -> float:
        """Average SNR  γ̄ = P_tx · G_tx · G_LS · G_rx / σ²  — Eq. (22)."""
        P_tx_W = 1e-3 * db_to_linear(P_tx_dBm)
        G_tx   = db_to_linear(G_tx_dBi)
        G_rx   = db_to_linear(G_rx_dBi)
        return P_tx_W * G_tx * G_LS_linear * G_rx / sigma_n_sq

    @staticmethod
    def cascaded_snr(gamma_GS: float, gamma_SU: float) -> float:
        """End-to-end cascaded SNR for transparent relay — Eq. (23)."""
        return (gamma_GS * gamma_SU) / (gamma_GS + gamma_SU + 1.0)

    @staticmethod
    def mrc_combined_snr(gamma_bar_SU: float, P_taps: np.ndarray,
                         fading_power: np.ndarray) -> float:
        """MRC-combined access-link SNR — Eq. (28)."""
        return gamma_bar_SU * np.sum(P_taps * fading_power)

    @staticmethod
    def combined_snr_with_feeder(gamma_bar_GS: float,
                                  gamma_SU_MRC: float) -> float:
        """Combined SNR accounting for feeder link — Eq. (27)."""
        return (gamma_bar_GS * gamma_SU_MRC) / (gamma_bar_GS + gamma_SU_MRC + 1.0)

    @staticmethod
    def los_deterministic_snr(gamma_bar_SU: float, P_LOS: float) -> float:
        """Deterministic LOS SNR component — Eq. (30)."""
        return gamma_bar_SU * P_LOS

    @staticmethod
    def nlos_aggregate_snr(gamma_bar_SU: float, P_nlos_taps: np.ndarray,
                            fading_power_nlos: np.ndarray) -> float:
        """Aggregate NLOS SNR — Eq. (31)."""
        return gamma_bar_SU * np.sum(P_nlos_taps * fading_power_nlos)


# ============================================================================
#  OTFS Modulation / Demodulation (Eqs. 33–38)
# ============================================================================

class OTFSModulator:
    """OTFS modulator implementing ISFFT and Heisenberg transform."""

    def __init__(self, params: OTFSParams):
        self.p = params

    def isfft(self, x_dd: np.ndarray) -> np.ndarray:
        """
        Inverse Symplectic FFT: delay-Doppler → time-frequency — Eq. (33).

        Parameters
        ----------
        x_dd : (N_D, M_D) complex array of DD-domain symbols

        Returns
        -------
        X_tf : (N_D, M_D) complex array of TF-domain symbols
        """
        N, M = self.p.N_D, self.p.M_D
        assert x_dd.shape == (N, M)
        # ISFFT: X[p,q] = (1/√(NM)) Σ_u Σ_v x[u,v] exp(j2π(pu/N - qv/M))
        X_tf = np.zeros((N, M), dtype=complex)
        for p in range(N):
            for q in range(M):
                val = 0.0 + 0j
                for u in range(N):
                    for v in range(M):
                        phase = 2 * np.pi * (p * u / N - q * v / M)
                        val += x_dd[u, v] * np.exp(1j * phase)
                X_tf[p, q] = val / np.sqrt(N * M)
        return X_tf

    def isfft_fast(self, x_dd: np.ndarray) -> np.ndarray:
        """Fast ISFFT using FFT/IFFT along appropriate axes."""
        # ISFFT = IFFT along Doppler (axis=0), FFT along delay (axis=1),
        # with proper normalisation
        N, M = self.p.N_D, self.p.M_D
        # Along u (Doppler): exp(j2π pu/N) → IFFT * N
        # Along v (delay):   exp(-j2π qv/M) → FFT
        tmp = np.fft.ifft(x_dd, axis=0) * np.sqrt(N)
        X_tf = np.fft.fft(tmp, axis=1) / np.sqrt(M)
        return X_tf

    def sfft(self, Y_tf: np.ndarray) -> np.ndarray:
        """
        Symplectic FFT: time-frequency → delay-Doppler — Eq. (35).

        Parameters
        ----------
        Y_tf : (N_D, M_D) complex array of TF-domain received samples

        Returns
        -------
        y_dd : (N_D, M_D) complex array of DD-domain symbols
        """
        N, M = self.p.N_D, self.p.M_D
        # SFFT is the inverse of ISFFT
        tmp = np.fft.fft(Y_tf, axis=0) / np.sqrt(N)
        y_dd = np.fft.ifft(tmp, axis=1) * np.sqrt(M)
        return y_dd

    def heisenberg_transform(self, X_tf: np.ndarray,
                              g_tx: np.ndarray = None) -> np.ndarray:
        """
        Heisenberg transform: TF-domain → continuous-time samples — Eq. (34).

        Parameters
        ----------
        X_tf : (N_D, M_D) TF-domain symbols
        g_tx : optional pulse-shaping function samples

        Returns
        -------
        s    : (N_D * M_D,) time-domain samples (sampled at T_s = T_sym / M_D)
        """
        N, M = self.p.N_D, self.p.M_D
        n_samples = N * M

        if g_tx is None:
            # Rectangular pulse (ideal)
            g_tx = np.ones(M) / np.sqrt(M)

        s = np.zeros(n_samples, dtype=complex)
        for p in range(N):
            for q in range(M):
                for k in range(M):
                    idx = p * M + k
                    if 0 <= idx < n_samples:
                        s[idx] += X_tf[p, q] * g_tx[k] * np.exp(
                            1j * 2 * np.pi * q * k / M
                        )
        return s


# ============================================================================
#  Delay-Doppler Channel Application  (Eq. 36)
# ============================================================================

class DDChannel:
    """Apply discrete delay-Doppler channel to OTFS symbols."""

    @staticmethod
    def apply(x_dd: np.ndarray, h_dd: np.ndarray,
              noise_var: float,
              rng: np.random.Generator = None) -> np.ndarray:
        """
        DD-domain input-output relation — Eq. (36).

        y[u,v] = Σ_{u',v'} h[(u-u')_N, (v-v')_M] x[u',v'] + w[u,v]

        Parameters
        ----------
        x_dd     : (N_D, M_D) transmitted DD symbols
        h_dd     : (N_D, M_D) DD-domain channel kernel
        noise_var: effective noise variance σ²_w
        rng      : random generator

        Returns
        -------
        y_dd     : (N_D, M_D) received DD symbols
        """
        if rng is None:
            rng = np.random.default_rng()
        N, M = x_dd.shape
        y_dd = np.zeros((N, M), dtype=complex)
        for u in range(N):
            for v in range(M):
                val = 0.0 + 0j
                for u_p in range(N):
                    for v_p in range(M):
                        val += h_dd[(u - u_p) % N, (v - v_p) % M] * x_dd[u_p, v_p]
                y_dd[u, v] = val
        # Add noise
        w = np.sqrt(noise_var / 2) * (
            rng.standard_normal((N, M)) + 1j * rng.standard_normal((N, M))
        )
        return y_dd + w

    @staticmethod
    def build_dd_kernel(tap_gains: np.ndarray, tap_delays_bins: np.ndarray,
                        tap_dopplers_bins: np.ndarray,
                        N_D: int, M_D: int) -> np.ndarray:
        """
        Build DD-domain channel kernel h[u,v] from tap parameters.

        Parameters
        ----------
        tap_gains        : (L,) complex gains α_p
        tap_delays_bins  : (L,) integer delay indices
        tap_dopplers_bins: (L,) integer Doppler indices
        N_D, M_D         : grid dimensions

        Returns
        -------
        h_dd : (N_D, M_D) DD kernel
        """
        h_dd = np.zeros((N_D, M_D), dtype=complex)
        for p in range(len(tap_gains)):
            u_idx = int(tap_dopplers_bins[p]) % N_D
            v_idx = int(tap_delays_bins[p]) % M_D
            h_dd[u_idx, v_idx] += tap_gains[p]
        return h_dd

    @staticmethod
    def apply_fast(x_dd: np.ndarray, h_dd: np.ndarray,
                   noise_var: float,
                   rng: np.random.Generator = None) -> np.ndarray:
        """
        FFT-based DD-domain channel application (2D circular convolution).

        Equivalent to apply() but O(NM log NM) instead of O(N²M²).

        Y = IFFT2(FFT2(h_dd) * FFT2(x_dd)) + noise
        """
        if rng is None:
            rng = np.random.default_rng()
        N, M = x_dd.shape
        H = np.fft.fft2(h_dd)
        X = np.fft.fft2(x_dd)
        y_dd = np.fft.ifft2(H * X)
        # Add noise
        w = np.sqrt(noise_var / 2) * (
            rng.standard_normal((N, M)) + 1j * rng.standard_normal((N, M))
        )
        return y_dd + w


# ============================================================================
#  Ephemeris-Driven Outage Prediction (Eqs. 53–60)
# ============================================================================

class EphemerisPredictor:
    """
    Ephemeris-driven reliability prediction for subchannel outage
    classification.
    """

    def __init__(self, orbital: OrbitalParams,
                 feeder_budget: LinkBudgetParams,
                 access_budget: LinkBudgetParams,
                 atm: AtmosphericParams,
                 env: EnvironmentParams,
                 otfs: OTFSParams,
                 tdl_profile: TDLProfile):
        self.orbital = orbital
        self.feeder = feeder_budget
        self.access = access_budget
        self.path_loss = LargeScalePathLoss(atm, env)
        self.otfs = otfs
        self.tdl = tdl_profile
        self.atm = atm

    def predict_elevation_deg(self, sat_pos: np.ndarray,
                               ue_pos: np.ndarray) -> float:
        """Compute elevation angle [deg] from positions."""
        diff = sat_pos - ue_pos
        d = np.linalg.norm(diff)
        e_vert = np.array([0, 0, 1])  # local zenith
        sin_eps = np.dot(diff / d, e_vert)
        return np.degrees(np.arcsin(np.clip(sin_eps, -1, 1)))

    def predict_subchannel_bler(
        self,
        epsilon_deg: float,
        R_c: float,
        use_high_snr_approx: bool = True,
        rng: np.random.Generator = None,
    ) -> float:
        """
        Predict BLER for a subchannel given elevation angle — Eq. (56).

        Returns predicted ε_i^E2E.
        """
        epsilon_rad = np.radians(epsilon_deg)
        wavelength = SPEED_OF_LIGHT / self.access.f_c_Hz

        # Access-link slant range and path loss
        d_SU = self.path_loss.slant_range(epsilon_rad, self.orbital.h_orb)
        PL_SU = self.path_loss.total_path_loss_dB(
            d_SU, wavelength, self.access.f_c_Hz, epsilon_rad, epsilon_deg, rng
        )
        G_LS_SU = self.path_loss.large_scale_gain_linear(PL_SU)

        # Feeder-link (assume near-zenith, high gain, ~deterministic)
        d_GS = self.orbital.h_orb + 200e3  # approximate
        PL_GS = self.path_loss.fspl_dB(d_GS, SPEED_OF_LIGHT / self.feeder.f_c_Hz)
        G_LS_GS = self.path_loss.large_scale_gain_linear(PL_GS)

        # Noise powers
        sigma_n_GS = SNRCalculator.noise_power(
            self.feeder.bandwidth_Hz, self.feeder.noise_figure_dB
        )
        sigma_n_SU = SNRCalculator.noise_power(
            self.access.bandwidth_Hz, self.access.noise_figure_dB
        )

        # Average SNRs
        gamma_bar_GS = SNRCalculator.average_snr(
            self.feeder.P_tx_dBm, self.feeder.G_tx_dBi,
            self.feeder.G_rx_dBi, G_LS_GS, sigma_n_GS
        )
        gamma_bar_SU = SNRCalculator.average_snr(
            self.access.P_tx_dBm, self.access.G_tx_dBi,
            self.access.G_rx_dBi, G_LS_SU, sigma_n_SU
        )

        # Cascaded average E2E SNR — Eq. (55)
        gamma_bar_E2E = SNRCalculator.cascaded_snr(gamma_bar_GS, gamma_bar_SU)

        # Decompose into LOS + NLOS components
        L_i = self.tdl.num_taps
        P_LOS = self.tdl.relative_powers_linear[0]
        P_NLOS = 1.0 - P_LOS
        gamma_det = gamma_bar_SU * P_LOS
        gamma_bar_NLOS = gamma_bar_SU * P_NLOS / max(L_i - 1, 1)

        # SNR threshold
        gamma_th = 2.0 ** R_c - 1.0

        # BLER
        if use_high_snr_approx:
            bler = FiniteBlocklengthBLER.bler_high_snr(
                gamma_th, gamma_det, gamma_bar_NLOS, L_i
            )
        else:
            bler = FiniteBlocklengthBLER.bler_closed_form(
                gamma_th, gamma_det, gamma_bar_NLOS, L_i
            )
        return bler

    def classify_subchannels(
        self,
        elevation_angles_deg: np.ndarray,
        R_c: float,
        tau_safe: float = 0.9,
        rng: np.random.Generator = None,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Classify subchannels into safe/uncertain sets — Eqs. (58–59).

        Returns
        -------
        eta         : (n,) reliability scores
        S_safe      : indices of safe subchannels
        S_uncertain : indices of uncertain subchannels
        """
        n = len(elevation_angles_deg)
        eta = np.zeros(n)
        for i in range(n):
            bler = self.predict_subchannel_bler(
                elevation_angles_deg[i], R_c, rng=rng
            )
            eta[i] = 1.0 - bler  # Eq. (57)

        S_safe = np.where(eta > tau_safe)[0]
        S_uncertain = np.where(eta <= tau_safe)[0]
        return eta, S_safe, S_uncertain


# ============================================================================
#  Full Subchannel Simulation
# ============================================================================

class LEO_NTN_Subchannel:
    """
    End-to-end simulation of one LEO NTN subchannel including:
    - Large-scale path loss
    - Small-scale Rician TDL-D fading
    - OTFS modulation / DD-domain channel / demodulation
    - Cascaded SNR computation
    """

    def __init__(self,
                 orbital: OrbitalParams,
                 feeder_budget: LinkBudgetParams,
                 access_budget: LinkBudgetParams,
                 atm: AtmosphericParams,
                 env: EnvironmentParams,
                 otfs_params: OTFSParams,
                 access_tdl: TDLProfile):
        self.orbital = orbital
        self.feeder = feeder_budget
        self.access = access_budget
        self.atm = atm
        self.env = env
        self.otfs = otfs_params
        self.access_tdl = access_tdl
        self.path_loss_model = LargeScalePathLoss(atm, env)
        self.fading = SmallScaleFading(access_tdl)
        self.modulator = OTFSModulator(otfs_params)

    def simulate_frame(
        self,
        epsilon_deg: float,
        v_rel_ms: float = 7500.0,
        R_c: float = 1.0,
        rng: np.random.Generator = None,
    ) -> dict:
        """
        Simulate one OTFS frame through the full end-to-end channel.

        Parameters
        ----------
        epsilon_deg : elevation angle [degrees]
        v_rel_ms    : relative velocity [m/s]
        R_c         : coding rate [bits/channel use]
        rng         : random generator

        Returns
        -------
        results : dict with SNR values, BLER estimates, channel data
        """
        if rng is None:
            rng = np.random.default_rng()

        N_D, M_D = self.otfs.N_D, self.otfs.M_D
        n_c = self.otfs.blocklength
        epsilon_rad = np.radians(epsilon_deg)
        wavelength = SPEED_OF_LIGHT / self.access.f_c_Hz

        # --- Large-scale propagation ---
        d_SU = self.path_loss_model.slant_range(epsilon_rad, self.orbital.h_orb)
        PL_SU = self.path_loss_model.total_path_loss_dB(
            d_SU, wavelength, self.access.f_c_Hz,
            epsilon_rad, epsilon_deg, rng
        )
        G_LS_SU = self.path_loss_model.large_scale_gain_linear(PL_SU)

        # Feeder link (simplified: near-zenith, high gain)
        d_GS = self.orbital.h_orb
        PL_GS = self.path_loss_model.fspl_dB(
            d_GS, SPEED_OF_LIGHT / self.feeder.f_c_Hz
        )
        G_LS_GS = self.path_loss_model.large_scale_gain_linear(PL_GS)

        # --- Noise powers ---
        sigma_n_SU = SNRCalculator.noise_power(
            self.access.bandwidth_Hz, self.access.noise_figure_dB
        )
        sigma_n_GS = SNRCalculator.noise_power(
            self.feeder.bandwidth_Hz, self.feeder.noise_figure_dB
        )

        # --- Average SNRs ---
        gamma_bar_GS = SNRCalculator.average_snr(
            self.feeder.P_tx_dBm, self.feeder.G_tx_dBi,
            self.feeder.G_rx_dBi, G_LS_GS, sigma_n_GS
        )
        gamma_bar_SU = SNRCalculator.average_snr(
            self.access.P_tx_dBm, self.access.G_tx_dBi,
            self.access.G_rx_dBi, G_LS_SU, sigma_n_SU
        )

        # --- Small-scale fading ---
        L = self.access_tdl.num_taps
        f_D_max = v_rel_ms / wavelength
        # Per-tap Doppler shifts — Eq. (5)
        theta_p = rng.uniform(-np.pi, np.pi, L)
        theta_p[0] = 0.0  # LOS path: on-axis
        f_D_taps = f_D_max * np.cos(theta_p)

        T_frame = N_D * self.otfs.T_sym
        t_vec = np.array([0.0])  # single snapshot
        tap_gains = self.fading.channel_impulse_response(
            t_vec, f_D_taps, G_LS_SU, rng
        )[:, 0]  # (L,) complex

        # --- Per-tap and combined SNR ---
        P_taps = self.access_tdl.relative_powers_linear
        fading_power = np.abs(
            self.fading.generate_tap_gains(t_vec, f_D_taps, rng)[:, 0]
        )**2

        gamma_per_tap = gamma_bar_SU * P_taps * fading_power
        gamma_SU_MRC = np.sum(gamma_per_tap)
        gamma_E2E = SNRCalculator.combined_snr_with_feeder(
            gamma_bar_GS, gamma_SU_MRC
        )

        # --- BLER estimates ---
        gamma_th = 2.0**R_c - 1.0
        P_LOS = P_taps[0]
        gamma_det = gamma_bar_SU * P_LOS
        P_NLOS = 1.0 - P_LOS
        gamma_bar_NLOS = gamma_bar_SU * P_NLOS / max(L - 1, 1)

        bler_cf = FiniteBlocklengthBLER.bler_closed_form(
            gamma_th, gamma_det, gamma_bar_NLOS, L
        )
        bler_hs = FiniteBlocklengthBLER.bler_high_snr(
            gamma_th, gamma_det, gamma_bar_NLOS, L
        )
        bler_cond = FiniteBlocklengthBLER.conditional_bler(gamma_E2E, n_c, R_c)

        # --- OTFS DD-domain channel ---
        delays_s = np.array(self.access_tdl.delays_ns) * 1e-9
        delay_bins = np.round(delays_s * M_D * self.otfs.delta_f_Hz).astype(int)
        doppler_bins = np.round(f_D_taps * N_D * self.otfs.T_sym).astype(int)

        h_dd = DDChannel.build_dd_kernel(
            tap_gains, delay_bins, doppler_bins, N_D, M_D
        )

        return {
            "slant_range_km": d_SU / 1e3,
            "PL_access_dB": PL_SU,
            "PL_feeder_dB": PL_GS,
            "gamma_bar_GS_dB": linear_to_db(gamma_bar_GS),
            "gamma_bar_SU_dB": linear_to_db(gamma_bar_SU),
            "gamma_SU_MRC_dB": linear_to_db(gamma_SU_MRC),
            "gamma_E2E_dB": linear_to_db(gamma_E2E),
            "gamma_det_dB": linear_to_db(gamma_det),
            "bler_closed_form": bler_cf,
            "bler_high_snr": bler_hs,
            "bler_conditional": bler_cond,
            "h_dd": h_dd,
            "tap_gains": tap_gains,
            "diversity_order": L,
        }


# ============================================================================
#  Example / Demo
# ============================================================================

def main():
    """Demonstrate the channel model with a sweep over elevation angles."""
    print("=" * 72)
    print("LEO NTN OTFS Channel Model — Demonstration")
    print("=" * 72)

    # --- System parameters ---
    orbital = OrbitalParams(h_orb=600e3)

    feeder = LinkBudgetParams(
        P_tx_dBm=43.0, G_tx_dBi=45.0, G_rx_dBi=35.0,
        f_c_Hz=20e9, noise_figure_dB=3.0, bandwidth_Hz=250e6,
    )
    access = LinkBudgetParams(
        P_tx_dBm=33.0, G_tx_dBi=30.0, G_rx_dBi=0.0,
        f_c_Hz=2e9, noise_figure_dB=7.0, bandwidth_Hz=20e6,
    )
    atm = AtmosphericParams(rain_rate_mmh=5.0)
    env = EnvironmentParams(env_type="suburban")

    otfs = OTFSParams(N_D=16, M_D=64, delta_f_Hz=15e3)

    tdl = TDLProfile(
        num_taps=4,
        relative_powers_dB=[0, -3, -6, -9],
        delays_ns=[0, 100, 200, 400],
        K_factor_dB=10.0,
    )

    subchannel = LEO_NTN_Subchannel(
        orbital, feeder, access, atm, env, otfs, tdl
    )

    rng = np.random.default_rng(2026)
    R_c = 1.0  # bits/channel use

    print(f"\nOTFS grid: {otfs.N_D}×{otfs.M_D}  (blocklength n_c = {otfs.blocklength})")
    print(f"Access link: f_c = {access.f_c_Hz/1e9:.1f} GHz, BW = {access.bandwidth_Hz/1e6:.0f} MHz")
    print(f"Feeder link: f_c = {feeder.f_c_Hz/1e9:.1f} GHz, BW = {feeder.bandwidth_Hz/1e6:.0f} MHz")
    print(f"Orbital altitude: {orbital.h_orb/1e3:.0f} km")
    print(f"Coding rate R_c = {R_c} b/cu")
    print(f"Environment: {env.env_type}")
    print(f"Delay-Doppler taps: L = {tdl.num_taps}")
    print(f"Rician K-factor (LOS tap): {tdl.K_factor_dB} dB")
    print()

    # --- Elevation angle sweep ---
    elevations = np.arange(10, 91, 10)
    print(f"{'Elev [°]':>9} {'d [km]':>9} {'PL_acc[dB]':>11} "
          f"{'γ̄_GS[dB]':>10} {'γ̄_SU[dB]':>10} {'γ_E2E[dB]':>10} "
          f"{'BLER(CF)':>10} {'BLER(HS)':>10}")
    print("-" * 100)

    for eps in elevations:
        res = subchannel.simulate_frame(eps, v_rel_ms=7500.0, R_c=R_c, rng=rng)
        print(f"{eps:>9.0f} {res['slant_range_km']:>9.1f} "
              f"{res['PL_access_dB']:>11.1f} "
              f"{res['gamma_bar_GS_dB']:>10.1f} "
              f"{res['gamma_bar_SU_dB']:>10.1f} "
              f"{res['gamma_E2E_dB']:>10.1f} "
              f"{res['bler_closed_form']:>10.2e} "
              f"{res['bler_high_snr']:>10.2e}")

    # --- Ephemeris predictor demo ---
    print("\n" + "=" * 72)
    print("Ephemeris-Driven Subchannel Classification")
    print("=" * 72)

    predictor = EphemerisPredictor(
        orbital, feeder, access, atm, env, otfs, tdl
    )

    n_sats = 5
    sat_elevations = np.array([15.0, 35.0, 55.0, 25.0, 70.0])
    eta, S_safe, S_uncertain = predictor.classify_subchannels(
        sat_elevations, R_c=R_c, tau_safe=0.9, rng=rng
    )

    print(f"\nSubchannel reliabilities (τ_safe = 0.9):")
    for i in range(n_sats):
        label = "SAFE" if i in S_safe else "UNCERTAIN"
        print(f"  Subchannel {i+1}: ε = {sat_elevations[i]:5.1f}°, "
              f"η = {eta[i]:.6f}  [{label}]")
    print(f"\nSafe set:      {[int(x+1) for x in S_safe]}")
    print(f"Uncertain set: {[int(x+1) for x in S_uncertain]}")

    # --- OTFS modulation demo ---
    print("\n" + "=" * 72)
    print("OTFS Modulation Round-Trip (no channel)")
    print("=" * 72)

    N_D, M_D = 4, 8  # small grid for demo
    small_otfs = OTFSParams(N_D=N_D, M_D=M_D)
    mod = OTFSModulator(small_otfs)

    # Random QPSK symbols
    bits = rng.integers(0, 2, (N_D, M_D, 2))
    x_dd = (2 * bits[:, :, 0] - 1 + 1j * (2 * bits[:, :, 1] - 1)) / np.sqrt(2)

    X_tf = mod.isfft_fast(x_dd)
    y_dd = mod.sfft(X_tf)

    mse = np.mean(np.abs(x_dd - y_dd)**2)
    print(f"Grid: {N_D}×{M_D},  ISFFT→SFFT round-trip MSE: {mse:.2e}")

    print("\nDone.")


if __name__ == "__main__":
    main()