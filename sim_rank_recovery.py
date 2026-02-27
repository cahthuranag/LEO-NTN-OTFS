"""
Rank Recovery Probability vs Number of Erased Subchannels.
"""
import numpy as np
import matplotlib.pyplot as plt
import csv
from itertools import combinations
from typing import Dict

from sim_common import (
    ieee_setup, save_figure, DT_CONFIG, N_S, M_GF,
    gf2_rank, generate_candidate_T,
)


def simulate(n_s=N_S, k_c=DT_CONFIG.k_c, m=M_GF, n_T_samples=200):
    rho = k_c * m
    G_FIX = DT_CONFIG.G_FIX
    rng = np.random.default_rng(42)

    n_erased_arr = np.arange(0, n_s + 1)
    p_fix = np.zeros(n_s + 1)
    p_ada = np.zeros(n_s + 1)

    print(f"Rank Recovery (n_s={n_s}, k_c={k_c}, m={m}):")
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

        n_pass_fix = 0
        n_pass_ada = 0
        patterns = list(combinations(range(n_s), ne))
        n_patterns = len(patterns)

        for erased_subs in patterns:
            erased_mask = np.zeros(n_s, dtype=bool)
            for s in erased_subs:
                erased_mask[s] = True

            surviving_cols = []
            for s in range(n_s):
                if not erased_mask[s]:
                    surviving_cols.extend(range(s * m, (s + 1) * m))
            G_sub = G_FIX[:, surviving_cols]
            if gf2_rank(G_sub) == rho:
                n_pass_fix += 1

            ada_ok = False
            if gf2_rank(G_sub) == rho:
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
        print(f"  n_erased={ne}: P_fix={p_fix[ne]:.4f}, P_ada={p_ada[ne]:.4f}")

    return {
        'n_erased': n_erased_arr,
        'p_recovery_fixed': p_fix,
        'p_recovery_adaptive': p_ada,
    }


def plot(results):
    ieee_setup()
    fig, ax = plt.subplots(figsize=(3.5, 2.8))
    ne = results['n_erased']
    width = 0.35

    ax.bar(ne - width / 2, results['p_recovery_fixed'], width,
           color='#ff7f0e', alpha=0.85, label=r'$\mathbf{G}_{\mathrm{FIX}}$')
    ax.bar(ne + width / 2, results['p_recovery_adaptive'], width,
           color='#1f77b4', alpha=0.85,
           label=r'$\mathbf{G}_{\mathrm{DIV}} = \mathbf{T} \cdot \mathbf{G}_{\mathrm{FIX}}$')

    d_min = DT_CONFIG.d_min
    ax.axvline(x=d_min - 1 + 0.5, color='red', ls='--', lw=1.0,
               alpha=0.7, label=f'$d_{{\\min}}-1 = {d_min - 1}$')

    ax.set_xlabel('Number of erased subchannels')
    ax.set_ylabel('Rank recovery probability')
    ax.legend(fontsize=7)
    ax.set_xticks(ne)
    ax.set_ylim(0, 1.15)
    ax.grid(True, axis='y', ls='--', alpha=0.3)

    save_figure(fig, 'rank_recovery')


def save_csv(results):
    with open('rank_recovery.csv', 'w', newline='') as f:
        w = csv.writer(f)
        w.writerow(['n_erased', 'P_recovery_fixed', 'P_recovery_adaptive'])
        for i in range(len(results['n_erased'])):
            w.writerow([results['n_erased'][i],
                         f"{results['p_recovery_fixed'][i]:.6f}",
                         f"{results['p_recovery_adaptive'][i]:.6f}"])
    print("Saved: rank_recovery.csv")


if __name__ == "__main__":
    results = simulate()
    plot(results)
    save_csv(results)
