"""
BLER vs Number of Subchannels n_s.
"""
import numpy as np
import matplotlib.pyplot as plt
import csv
from typing import Dict

from sim_common import (
    ieee_setup, save_figure, DT_CONFIG, N_S, M_GF, OTFS_PARAMS,
    slant_range_m, access_fspl_dB, atmospheric_loss_dB, power_to_link_snrs,
    create_ephemeris_predictor, mc_trial_variable_ns,
    build_subchannel_map,
    db_to_linear, linear_to_db,
    gf2_rank, generate_candidate_T, gf2_matmul,
)
from polar_code import PolarCode
from diversity_transform import DiversityTransformConfig


def simulate(N=256, K=128, ns_values=None,
             P_total_dBW=8.0, ref_elevation_deg=50.0, n_mc=1500):
    if ns_values is None:
        ns_values = [2, 3, 4, 5, 6, 7]
    m = 3

    rng = np.random.default_rng(2031)
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

    n_pts = len(ns_values)
    results = {key: np.zeros(n_pts) for key in [
        'ppv_fixed', 'ppv_adaptive', 'fixed', 'adaptive']}
    results['n_subchannels'] = np.array(ns_values)

    print(f"BLER vs n_subchannels (P_total={P_total_dBW} dBW, MC={n_mc})")

    for idx, n_s_val in enumerate(ns_values):
        k_c_val = max(n_s_val - 2, 1)
        if k_c_val >= n_s_val:
            k_c_val = n_s_val - 1

        dt_config = DiversityTransformConfig(k_c=k_c_val, n_s=n_s_val, m=m)
        rho = dt_config.rho
        n_out = dt_config.n_out
        G_FIX_local = dt_config.G_FIX

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

        m_sub = OTFS_PARAMS.M_D // n_s_val
        sc_map = build_subchannel_map(n_s=n_s_val, m_sub=m_sub)
        n_sym = {j: len(sc_map[j]) for j in range(n_s_val)}

        polar_code = PolarCode(N, K, design_snr_dB=1.0, list_size=8)

        predictor = create_ephemeris_predictor(ref_elevation_deg, P_total_dBW, K_rice_dB)
        bler_eph = predictor.predict_subchannel_bler(ref_elevation_deg, R)
        eta_eph_scalar = max(1.0 - bler_eph, 0.0)
        eta_ephemeris = np.full(n_s_val, eta_eph_scalar)

        ppv_fix_acc, ppv_ada_acc = 0.0, 0.0
        err_fix_count, err_ada_count = 0, 0

        print(f"  n_s={n_s_val}, k_c={k_c_val}, rho={rho}, "
              f"max_erasures={dt_config.max_erasures}, 2^rho={N_CW_local}...")

        for _ in range(n_mc):
            res = mc_trial_variable_ns(
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
              f"Ada={results['adaptive'][idx]:.4e}")

    return results


def plot(results):
    ieee_setup()
    fig, ax = plt.subplots(figsize=(3.5, 2.8))
    ns = results['n_subchannels']

    ax.semilogy(ns, np.maximum(results['ppv_fixed'], 1e-8),
                'p--', color='#17becf', lw=0.8, ms=4,
                label='PPV bound (fixed)')
    ax.semilogy(ns, np.maximum(results['ppv_adaptive'], 1e-8),
                's--', color='#2ca02c', lw=0.8, ms=4,
                label='PPV bound (adaptive)')
    ax.semilogy(ns, np.maximum(results['fixed'], 1e-8),
                'D-', color='#ff7f0e', lw=1.2, ms=5,
                label='Fixed diversity transform')
    ax.semilogy(ns, np.maximum(results['adaptive'], 1e-8),
                'o-', color='#1f77b4', lw=1.4, ms=5,
                label='Adaptive diversity transform')

    ax.set_xlabel(r'Number of subchannels $n_s$')
    ax.set_ylabel('Block error rate (BLER)')
    ax.legend(fontsize=7, loc='upper right', framealpha=0.9)
    ax.grid(True, which='both', ls='--', alpha=0.3)
    ax.set_xticks(ns)

    save_figure(fig, 'bler_vs_n_subchannels')


def save_csv(results):
    keys = ['ppv_fixed', 'ppv_adaptive', 'fixed', 'adaptive']
    with open('bler_vs_n_subchannels.csv', 'w', newline='') as f:
        w = csv.writer(f)
        w.writerow(['n_subchannels'] + keys)
        for i in range(len(results['n_subchannels'])):
            row = [str(int(results['n_subchannels'][i]))]
            for k in keys:
                row.append(f"{results[k][i]:.6e}")
            w.writerow(row)
    print("Saved: bler_vs_n_subchannels.csv")


if __name__ == "__main__":
    results = simulate(n_mc=1500)
    plot(results)
    save_csv(results)
