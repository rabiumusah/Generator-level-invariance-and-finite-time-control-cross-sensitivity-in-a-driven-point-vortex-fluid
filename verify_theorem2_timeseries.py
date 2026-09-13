"""
verify_theorem2_timeseries.py -- time-resolved companion to verify_theorem2.py.

verify_theorem2.py already confirms, at the two trajectory endpoints, that
kappa_gamma=0, nu=0 (kappa_pin held finite, no rim injection) leaves every
Gamma_i bit-exact while the vortex positions move substantially. This script
reruns the identical three seeded configurations, identical DeviceParams,
identical VortexSystem.step_rk4 integration, and identical integration
horizon -- the only difference is that circulations and positions are
recorded at regular checkpoints along the way, not just at t=0 and t=T, so
the "frozen circulation / freely evolving positions" claim can be shown as
a trajectory rather than asserted from two snapshots.

No new physics, no new parameters, no fabricated or hand-typed numbers: this
is the same conservative-sector check, sampled more finely. Its two
endpoints must reproduce verify_theorem2.py's summary numbers exactly
(checked below) as a consistency guard before anything is plotted.
"""
from __future__ import annotations
import json
import numpy as np

import pmvm_core as core
from pmvm_hamiltonian import VortexSystem

# Identical to verify_theorem2.py's three configurations.
CONFIGS = [
    dict(seed=7, n_seed=8, r_seed=0.60, Gamma_total=0.30, well_amp=1.0, n_steps=2000),
    dict(seed=11, n_seed=8, r_seed=0.50, Gamma_total=0.20, well_amp=1.5, n_steps=1200),
    dict(seed=23, n_seed=6, r_seed=0.75, Gamma_total=0.15, well_amp=0.5, n_steps=2500),
]
DT = 2e-3
N_CHECKPOINTS = 60  # sampled roughly evenly across each run, endpoints included


def run_timeseries(seed, n_seed, r_seed, Gamma_total, well_amp, n_steps, dt=DT):
    p = core.DeviceParams(kappa_gamma=0.0, nu=0.0, kappa_pin=1.0)
    well_signs = np.array([1.0, -1.0, 1.0, -1.0])
    sys_ = VortexSystem(p, well_signs, well_amp=well_amp, drag_coeff=0.4,
                         seed=seed, pin_mode="mag_only")
    sys_.seed_ring(Gamma_total=Gamma_total, n_seed=n_seed, r_seed=r_seed, jitter=1e-3)

    gamma0 = sys_.gamma.copy()
    xy0 = sys_.xy.copy()
    n0 = len(gamma0)

    checkpoint_steps = sorted(set(
        np.round(np.linspace(0, n_steps, N_CHECKPOINTS)).astype(int).tolist()
    ))

    t_list, max_abs_dgamma_list, max_disp_list, n_vortices_list = [], [], [], []
    xy_full_snapshots = {}  # step -> xy array, kept only at a few steps for the trajectory panel

    next_ci = 0
    for step in range(n_steps + 1):
        if next_ci < len(checkpoint_steps) and step == checkpoint_steps[next_ci]:
            n1 = len(sys_.gamma)
            same_count = (n1 == n0)
            max_abs_dgamma = float(np.max(np.abs(sys_.gamma - gamma0))) if same_count else float("nan")
            disp = np.hypot(sys_.xy[:, 0] - xy0[:, 0], sys_.xy[:, 1] - xy0[:, 1]) if same_count else np.array([np.nan])
            t_list.append(sys_.t)
            max_abs_dgamma_list.append(max_abs_dgamma)
            max_disp_list.append(float(np.max(disp)))
            n_vortices_list.append(int(n1))
            xy_full_snapshots[step] = sys_.xy.copy().tolist()
            next_ci += 1
        if step < n_steps:
            sys_.step_rk4(dt, 0.0)

    gamma_end = sys_.gamma.copy()
    xy_end = sys_.xy.copy()
    n_end = len(gamma_end)
    same_count_end = (n_end == n0)
    gamma_frozen_bitexact = bool(same_count_end and np.array_equal(gamma_end, gamma0))
    disp_end = np.hypot(xy_end[:, 0] - xy0[:, 0], xy_end[:, 1] - xy0[:, 1]) if same_count_end else np.array([])

    return dict(
        seed=seed, n_seed=n_seed, r_seed=r_seed, Gamma_total=Gamma_total,
        well_amp=well_amp, n_steps=n_steps, dt=dt,
        t=t_list,
        max_abs_dgamma=max_abs_dgamma_list,
        max_position_displacement=max_disp_list,
        n_vortices=n_vortices_list,
        xy_start=xy0.tolist(),
        xy_end=xy_end.tolist(),
        xy_snapshots_by_step=xy_full_snapshots,
        checkpoint_steps=checkpoint_steps,
        gamma_frozen_bitexact_end=gamma_frozen_bitexact,
        max_abs_dgamma_end=float(np.max(np.abs(gamma_end - gamma0))) if same_count_end else float("nan"),
        max_position_displacement_end=float(disp_end.max()) if disp_end.size else None,
        vortex_count_unchanged=bool(same_count_end),
    )


def main():
    results = [run_timeseries(**cfg) for cfg in CONFIGS]

    # Consistency guard: endpoint values must match verify_theorem2.py's own
    # reported table (data/theorem2_verification.json) to within exact
    # equality for the frozen-circulation flag and float equality for the
    # displacement, since both scripts run the identical integration.
    with open("../data/theorem2_verification.json") as f:
        ref = json.load(f)
    ref_by_seed = {r["seed"]: r for r in ref["results"]}
    for r in results:
        rr = ref_by_seed[r["seed"]]
        assert r["gamma_frozen_bitexact_end"] == rr["gamma_frozen_bitexact"], \
            f"seed={r['seed']}: frozen-flag mismatch vs verify_theorem2.py"
        assert abs(r["max_position_displacement_end"] - rr["position_displacement_max"]) < 1e-9, \
            f"seed={r['seed']}: displacement mismatch vs verify_theorem2.py " \
            f"({r['max_position_displacement_end']} vs {rr['position_displacement_max']})"
        print(f"seed={r['seed']:3d}: consistency check against verify_theorem2.py OK "
              f"(frozen={r['gamma_frozen_bitexact_end']}, "
              f"max disp={r['max_position_displacement_end']:.6f})")

    with open("../data/theorem2_timeseries.json", "w") as f:
        json.dump(dict(configs=CONFIGS, results=results), f)

    print("\nWrote ../data/theorem2_timeseries.json")


if __name__ == "__main__":
    main()
