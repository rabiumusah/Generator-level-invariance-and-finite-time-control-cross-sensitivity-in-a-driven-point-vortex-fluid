"""
verify_theorem2.py -- Numerical sanity check of Theorem 2 (closure-level
circulation-fibre invariance in the conservative sector: kappa_gamma -> 0,
nu -> 0, kappa_pin finite implies every Gamma_i is frozen for all time,
while the positions r_i(t) remain free to evolve).

This is NOT the evidence for Theorem 2 -- the theorem is proved analytically
in the main text (Sec. Model / Theorem 2) directly from dGamma_i/dt =
-mu_local(r_i,t) Gamma_i, which vanishes identically once mu_local == 0.
This script is a numerical cross-check on the *actual code path*: it
instantiates pmvm_core.DeviceParams with kappa_gamma=0 and nu=0 while
kappa_pin is held at its ordinary finite value (kappa_pin=1.0, the same
number used throughout every reported run), integrates a real dynamically
seeded vortex ring with pmvm_hamiltonian.VortexSystem exactly as in every
other script in this repository (RK4, Biot-Savart + disk images, the
pinning guiding-center drift), and confirms two things simultaneously:

  (i)  every Gamma_i is bit-for-bit unchanged from its seeded value
       (dGamma_i/dt = -(mu_local + nu/sigma_c^2) Gamma_i is then exactly
       zero at every RK4 stage, since kappa_gamma=0 makes mu_local == 0
       identically and nu=0 kills the residual viscous-core channel), and

  (ii) the vortices are not sitting still: their positions visibly move
       under the conservative Biot-Savart + pinning-drift dynamics, and no
       merge/annihilation/nucleation event is recorded (nucleation_events
       stays empty; the vortex count is unchanged) -- i.e. the check is
       not vacuous.

Because kappa_gamma and kappa_pin are now genuinely independent fields on
DeviceParams (see pmvm_core.py), this state -- kappa_gamma=0 with kappa_pin
held finite -- is a literal, reachable configuration of the shipped code,
not a hypothetical one.

No rim injection (inject_ring) and no nucleation calls (maybe_nucleate) are
made during the run: the point of the check is the free relaxation of a
fixed multiset of circulations under the conservative dynamics alone, so
any source/sink term that could add or threshold-nucleate circulation is
deliberately excluded -- exactly the conservative-sector setting Theorem 2
is stated for.
"""
from __future__ import annotations
import json
import numpy as np

import pmvm_core as core
from pmvm_hamiltonian import VortexSystem


def run_check(seed: int, n_seed: int, r_seed: float, Gamma_total: float,
              well_amp: float, n_steps: int, dt: float = 2e-3):
    p = core.DeviceParams(kappa_gamma=0.0, nu=0.0, kappa_pin=1.0)
    well_signs = np.array([1.0, -1.0, 1.0, -1.0])
    sys_ = VortexSystem(p, well_signs, well_amp=well_amp, drag_coeff=0.4,
                         seed=seed, pin_mode="mag_only")
    sys_.seed_ring(Gamma_total=Gamma_total, n_seed=n_seed, r_seed=r_seed, jitter=1e-3)

    gamma0 = sys_.gamma.copy()
    xy0 = sys_.xy.copy()
    n0 = len(gamma0)

    for _ in range(n_steps):
        # No inject_ring call (no rim source) and no maybe_nucleate call
        # (no thresholded creation): purely the free conservative relaxation
        # of the seeded circulation multiset, I(t) == 0 throughout.
        sys_.step_rk4(dt, 0.0)

    gamma1 = sys_.gamma.copy()
    xy1 = sys_.xy.copy()
    n1 = len(gamma1)

    same_count = (n1 == n0)
    gamma_max_abs_diff = float(np.max(np.abs(gamma1 - gamma0))) if same_count else float("nan")
    gamma_frozen = same_count and np.array_equal(gamma1, gamma0)

    disp = np.hypot(xy1[:, 0] - xy0[:, 0], xy1[:, 1] - xy0[:, 1]) if same_count else np.array([])
    r_end = np.hypot(xy1[:, 0], xy1[:, 1]) if same_count else np.array([])

    return dict(
        seed=seed, n_seed=n_seed, r_seed=r_seed, Gamma_total=Gamma_total,
        well_amp=well_amp, n_steps=n_steps, dt=dt,
        n_vortices_start=int(n0), n_vortices_end=int(n1),
        vortex_count_unchanged=bool(same_count),
        nucleation_events=list(sys_.nucleation_events),
        n_nucleation_events=int(len(sys_.nucleation_events)),
        gamma_frozen_bitexact=bool(gamma_frozen),
        gamma_max_abs_diff=gamma_max_abs_diff,
        position_displacement_min=float(disp.min()) if disp.size else None,
        position_displacement_max=float(disp.max()) if disp.size else None,
        r_max_end=float(r_end.max()) if r_end.size else None,
        r_max_start=float(np.hypot(xy0[:, 0], xy0[:, 1]).max()),
        positions_moved=bool(disp.size and disp.max() > 1e-4),
    )


def main():
    # Three independent seeded configurations (different seed, ring radius,
    # vortex count, well amplitude) so the check is not an artifact of one
    # particular geometry.
    configs = [
        dict(seed=7, n_seed=8, r_seed=0.60, Gamma_total=0.30, well_amp=1.0, n_steps=2000),
        dict(seed=11, n_seed=8, r_seed=0.50, Gamma_total=0.20, well_amp=1.5, n_steps=1200),
        dict(seed=23, n_seed=6, r_seed=0.75, Gamma_total=0.15, well_amp=0.5, n_steps=2500),
    ]

    results = [run_check(**cfg) for cfg in configs]

    for r in results:
        if not r["vortex_count_unchanged"]:
            raise RuntimeError(
                f"seed={r['seed']}: vortex count changed ({r['n_vortices_start']} -> "
                f"{r['n_vortices_end']}) via a same-sign merge during the conservative-"
                f"sector run -- pick a less densely seeded configuration; a merge event "
                f"combines two circulations into one and is not itself a violation of "
                f"Theorem 2 (total circulation is still conserved), but it changes the "
                f"multiset being compared, so this script requires merge-free runs.")

    for r in results:
        print(f"seed={r['seed']:3d} n_v={r['n_vortices_start']:2d}->{r['n_vortices_end']:2d} | "
              f"gamma frozen bit-exact: {r['gamma_frozen_bitexact']} "
              f"(max|dGamma|={r['gamma_max_abs_diff']:.3e}) | "
              f"nucleation events: {r['n_nucleation_events']} | "
              f"position displacement max={r['position_displacement_max']:.3e} | "
              f"moved: {r['positions_moved']}")

    all_frozen = all(r["gamma_frozen_bitexact"] for r in results)
    all_no_events = all(r["n_nucleation_events"] == 0 for r in results)
    all_moved = all(r["positions_moved"] for r in results)
    all_nonvacuous = all_no_events and all_moved

    summary = dict(
        configs=configs,
        all_gamma_frozen_bitexact=bool(all_frozen),
        all_no_merge_or_nucleation_events=bool(all_no_events),
        all_positions_moved=bool(all_moved),
        check_nonvacuous=bool(all_nonvacuous),
        theorem2_conservative_sector_confirmed=bool(all_frozen and all_nonvacuous),
        results=results,
    )

    with open("../data/theorem2_verification.json", "w") as f:
        json.dump(summary, f, indent=2)

    print("\nWrote ../data/theorem2_verification.json")
    print(f"All circulations frozen bit-exact across {len(results)} independent "
          f"configurations: {all_frozen}")
    print(f"All checks non-vacuous (positions moved, no merge/nucleation events): "
          f"{all_nonvacuous}")
    print(f"Theorem 2 (conservative sector) confirmed on the actual code path: "
          f"{summary['theorem2_conservative_sector_confirmed']}")


if __name__ == "__main__":
    main()
