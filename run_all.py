"""
Run all simulations sequentially.

Each simulation can also be run independently:
    python sim_bler_vs_power.py
    python sim_bler_vs_elevation.py
    python sim_throughput.py
    python sim_rank_recovery.py
    python sim_bler_vs_blockage.py
    python sim_bler_vs_blocklength.py
    python sim_otfs_vs_ofdm.py
    python sim_bler_vs_nsubchannels.py
"""

import sim_bler_vs_power
import sim_bler_vs_elevation
import sim_throughput
import sim_rank_recovery
import sim_bler_vs_blockage
import sim_bler_vs_blocklength
import sim_otfs_vs_ofdm
import sim_bler_vs_nsubchannels


def main():
    print("=" * 70)
    print("Running all simulations")
    print("=" * 70)

    # 1. BLER vs Power
    print("\n" + "-" * 60)
    results_power = sim_bler_vs_power.simulate(n_mc=3000)
    sim_bler_vs_power.plot(results_power)
    sim_bler_vs_power.save_csv(results_power)

    # 2. BLER vs Elevation
    print("\n" + "-" * 60)
    results_elev = sim_bler_vs_elevation.simulate(n_mc=3000)
    sim_bler_vs_elevation.plot(results_elev)
    sim_bler_vs_elevation.save_csv(results_elev)

    # 3. Throughput-Reliability (uses power sweep results)
    print("\n" + "-" * 60)
    results_throughput = sim_throughput.simulate(results_power)
    sim_throughput.plot(results_throughput)
    sim_throughput.save_csv(results_throughput)

    # 4. Rank Recovery (algebraic, no MC)
    print("\n" + "-" * 60)
    results_rank = sim_rank_recovery.simulate()
    sim_rank_recovery.plot(results_rank)
    sim_rank_recovery.save_csv(results_rank)

    # 5. BLER vs Blockage
    print("\n" + "-" * 60)
    results_blockage = sim_bler_vs_blockage.simulate(n_mc=2000)
    sim_bler_vs_blockage.plot(results_blockage)
    sim_bler_vs_blockage.save_csv(results_blockage)

    # 6. BLER vs Blocklength
    print("\n" + "-" * 60)
    results_bl = sim_bler_vs_blocklength.simulate(n_mc=2000)
    sim_bler_vs_blocklength.plot(results_bl)
    sim_bler_vs_blocklength.save_csv(results_bl)

    # 7. OTFS vs OFDM (high Doppler comparison)
    print("\n" + "-" * 60)
    results_otfs = sim_otfs_vs_ofdm.simulate(n_mc=2000)
    sim_otfs_vs_ofdm.plot(results_otfs)
    sim_otfs_vs_ofdm.save_csv(results_otfs)

    # 8. BLER vs Number of Subchannels
    print("\n" + "-" * 60)
    results_ns = sim_bler_vs_nsubchannels.simulate(n_mc=1500)
    sim_bler_vs_nsubchannels.plot(results_ns)
    sim_bler_vs_nsubchannels.save_csv(results_ns)

    print("\n" + "=" * 70)
    print("All simulations complete.")
    print("=" * 70)


if __name__ == "__main__":
    main()
