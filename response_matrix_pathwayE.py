"""
response_matrix_pathwayE.py -- Pathway E (full-field pinning closure)
variant of the Route A empirical sensitivity-magnitude measurement.

This script is byte-for-byte identical to response_matrix.py in every
respect -- grid (I_vals, bw_vals), protocol (write/split/hold durations,
I_split_frac, nucleate_every), seed formula (base_seed + 100*i_idx + j_idx),
finite-difference definition, and eps_1/eps_2 formulas -- except for one
change: VortexSystem is constructed with pin_mode="full", so the position
equation of motion uses the full-field pinning closure

    U_pin^(full)(r,t) = kappa_pin/2 * (B_mag(r) + B_edge(t))^2

(main text Eq. eq:E) instead of the coded U_pin = kappa_pin/2 * B_mag(r)^2
used everywhere else in the manuscript. No coefficient is introduced beyond
p.kappa_pin, already used by pinning_potential in pmvm_core.py (and
structurally independent of the dissipative p.kappa_gamma used by
hartmann_rate/lorentz_source); no seed, grid point, or protocol timing differs from the
baseline sweep. The two runs are therefore directly comparable, point for
point, and the difference between their eps_1/eps_2 values is a genuine
measurement of the Pathway E contribution to the pilot cross-sensitivity
data -- not a separate or recalibrated experiment.
"""
from __future__ import annotations
import json, time, csv
import numpy as np

import pmvm_core as core
from pmvm_hamiltonian import VortexSystem

from response_matrix import finite_diff_matrix, _traj_diagnostics  # reuse verbatim


def run_point(I_write: float, well_amp: float, seed: int, P: int = 4,
              dt: float = 2e-3, t_write: float = 0.5, t_split: float = 1.0,
              t_hold: float = 1.5, I_split_frac: float = 0.5,
              nucleate_every: int = 20):
    """Identical to response_matrix.run_point, except VortexSystem is
    constructed with pin_mode="full" (Pathway E pinning closure)."""
    p = core.DeviceParams(P_wells=P)
    well_signs = np.array([1.0, -1.0, 1.0, -1.0])[:P]
    sys_ = VortexSystem(p, well_signs, well_amp, seed=seed, pin_mode="full")
    sys_.seed_ring(Gamma_total=np.sign(I_write) * 5e-3, n_seed=16, r_seed=0.93, jitter=2e-3)

    t_total = t_write + t_split + t_hold
    n_steps = int(t_total / dt)
    phase_bounds = dict(write=t_write, split=t_write + t_split, hold=t_total)

    I_hold_value = [0.0]
    hold_locked = [False]

    # See response_matrix.py's _traj_diagnostics/run_point for the rationale.
    gamma_min_full = np.inf
    gamma_min_postwrite = np.inf
    D_min_unclipped_traj = np.inf
    D_min_clipped_traj = np.inf
    gm0, du0, dc0 = _traj_diagnostics(sys_, p, I_write)
    gamma_min_full = min(gamma_min_full, gm0)
    D_min_unclipped_traj = min(D_min_unclipped_traj, du0)
    D_min_clipped_traj = min(D_min_clipped_traj, dc0)

    def I_profile(t):
        if t < t_write:
            return I_write
        elif t < phase_bounds["split"]:
            return np.sign(I_write) * I_split_frac * abs(I_write)
        else:
            return I_hold_value[0]

    held_state = {}
    for step in range(n_steps):
        t = sys_.t
        I_t = I_profile(t)

        if (not hold_locked[0]) and t >= phase_bounds["split"]:
            obs = sys_.observables()
            Gamma_target = max(obs.Gamma, 1e-6)
            sign_C = obs.C if obs.C != 0 else 1.0
            a_coef = p.kappa_gamma * p.B_edge_per_I ** 2 * Gamma_target
            b_coef = -p.alpha
            c_coef = (p.nu / p.sigma_c ** 2) * Gamma_target
            disc = b_coef ** 2 - 4 * a_coef * c_coef
            if disc >= 0 and a_coef > 1e-12:
                I_hold_value[0] = sign_C * (-b_coef - np.sqrt(disc)) / (2 * a_coef)
            else:
                I_hold_value[0] = sign_C * p.alpha / (p.kappa_gamma * p.B_edge_per_I ** 2 * Gamma_target)
            hold_locked[0] = True

        sys_.inject_ring(p.alpha, I_t, dt)
        if phase_bounds["write"] <= t < phase_bounds["split"] and step % nucleate_every == 0:
            sys_.maybe_nucleate(I_t)
        sys_.step_rk4(dt, I_t)

        gm, du, dc = _traj_diagnostics(sys_, p, I_t)
        gamma_min_full = min(gamma_min_full, gm)
        if sys_.t >= t_write:
            gamma_min_postwrite = min(gamma_min_postwrite, gm)
        D_min_unclipped_traj = min(D_min_unclipped_traj, du)
        D_min_clipped_traj = min(D_min_clipped_traj, dc)

        if (not held_state) and sys_.t >= phase_bounds["hold"] - 1e-9:
            obs_h = sys_.observables()
            spec_h = sys_.spectrum()
            held_state.update(dict(
                t=float(sys_.t),
                Gamma=float(obs_h.Gamma), N=int(obs_h.N), Omega=float(obs_h.Omega),
                E=float(obs_h.E), Q=float(obs_h.Q), C=float(obs_h.C),
                S=[float(s) for s in spec_h],
            ))

    if not held_state:
        obs_h = sys_.observables()
        spec_h = sys_.spectrum()
        held_state.update(dict(t=float(sys_.t), Gamma=float(obs_h.Gamma), N=int(obs_h.N),
                                Omega=float(obs_h.Omega), E=float(obs_h.E), Q=float(obs_h.Q),
                                C=float(obs_h.C), S=[float(s) for s in spec_h]))

    return dict(I_write=I_write, well_amp=well_amp, seed=seed, P=P, I_hold=I_hold_value[0],
                n_steps=n_steps,
                gamma_min_full=float(gamma_min_full),
                gamma_min_postwrite=float(gamma_min_postwrite),
                D_min_unclipped_traj=float(D_min_unclipped_traj),
                D_min_clipped_traj=float(D_min_clipped_traj),
                **held_state)


def run_grid(I_vals, bw_vals, P=4, base_seed=1000):
    grid = []
    t0 = time.time()
    for ii, I in enumerate(I_vals):
        for jj, bw in enumerate(bw_vals):
            seed = base_seed + 100 * ii + jj
            r = run_point(I, bw, seed, P=P)
            r["i_idx"], r["j_idx"] = ii, jj
            grid.append(r)
            print(f"[{len(grid):2d}/{len(I_vals)*len(bw_vals)}] I={I:.2f} b_w={bw:.2f} "
                  f"-> Gamma_held={r['Gamma']:.5f} N={r['N']} |S4|={r['S'][4]:.5f} "
                  f"({time.time()-t0:.1f}s elapsed)")
    return grid


def main():
    I_vals = [0.6, 0.9, 1.2, 1.5, 1.8]
    bw_vals = [0.5, 1.0, 1.5, 2.0, 2.5]
    P = 4

    grid = run_grid(I_vals, bw_vals, P=P)

    nI, nB = len(I_vals), len(bw_vals)
    Gamma_grid = np.full((nI, nB), np.nan)
    S4_grid = np.full((nI, nB), np.nan)
    N_grid = np.full((nI, nB), np.nan)
    for r in grid:
        Gamma_grid[r["i_idx"], r["j_idx"]] = r["Gamma"]
        S4_grid[r["i_idx"], r["j_idx"]] = r["S"][4]
        N_grid[r["i_idx"], r["j_idx"]] = r["N"]

    dGdI, dGdB, dGdI_mean, dGdB_mean = finite_diff_matrix(I_vals, bw_vals, Gamma_grid)
    dS4dI, dS4dB, dS4dI_mean, dS4dB_mean = finite_diff_matrix(I_vals, bw_vals, S4_grid)

    Delta_I = max(I_vals) - min(I_vals)
    Delta_bw = max(bw_vals) - min(bw_vals)

    eps_1 = (abs(dGdB_mean) * Delta_bw) / (abs(dGdI_mean) * Delta_I)
    eps_2 = (abs(dS4dI_mean) * Delta_I) / (abs(dS4dB_mean) * Delta_bw)

    Gamma_min_full_all = min(r["gamma_min_full"] for r in grid)
    Gamma_min_postwrite_all = min(r["gamma_min_postwrite"] for r in grid)
    D_min_unclipped_all = min(r["D_min_unclipped_traj"] for r in grid)
    D_min_clipped_all = min(r["D_min_clipped_traj"] for r in grid)

    summary = dict(
        pin_mode="full", I_vals=I_vals, bw_vals=bw_vals, P=P,
        Delta_I=Delta_I, Delta_bw=Delta_bw,
        dGamma_dI_mean=dGdI_mean, dGamma_dbw_mean=dGdB_mean,
        dS4_dI_mean=dS4dI_mean, dS4_dbw_mean=dS4dB_mean,
        eps_1_circulation_crosstalk=eps_1,
        eps_2_topology_crosstalk=eps_2,
        eps_1_definition="|dGamma/dbw|_I * Delta_bw / (|dGamma/dI|_bw * Delta_I), grid-mean central "
                          "differences over interior points",
        eps_2_definition="|dS4/dI|_bw * Delta_I / (|dS4/dbw|_I * Delta_bw), grid-mean central "
                          "differences over interior points, S4=|S_P| dominant well-count harmonic (P=4)",
        note="single seed per grid point (seed = 1000 + 100*i_idx + j_idx); not ensemble-averaged. "
             "Identical grid/protocol/seeds to response_matrix.py; only the pinning closure differs "
             "(pin_mode='full', Pathway E, Eq. eq:E) -- see module docstring.",
        Gamma_min_full_all=Gamma_min_full_all,
        Gamma_min_postwrite_all=Gamma_min_postwrite_all,
        D_min_unclipped_all=D_min_unclipped_all,
        D_min_clipped_all=D_min_clipped_all,
    )
    print("\n=== response matrix summary (Pathway E, full-field pinning closure) ===")
    for k, v in summary.items():
        print(f"  {k}: {v}")

    with open("../data/response_matrix_pathwayE.json", "w") as f:
        json.dump(dict(grid=grid, Gamma_grid=Gamma_grid.tolist(), S4_grid=S4_grid.tolist(),
                        N_grid=N_grid.tolist(), dGamma_dI=dGdI.tolist(), dGamma_dbw=dGdB.tolist(),
                        dS4_dI=dS4dI.tolist(), dS4_dbw=dS4dB.tolist(), summary=summary), f, indent=2)

    with open("../data/response_matrix_pathwayE_summary.json", "w") as f:
        json.dump(summary, f, indent=2)

    with open("../data/response_matrix_pathwayE.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["i_idx", "j_idx", "I_write", "well_amp", "seed", "I_hold", "Gamma_held",
                    "N_held", "Omega_held", "E_held", "Q_held", "C_held", "S0", "S1", "S2", "S3",
                    "S4", "S5", "S6", "S7", "gamma_min_full", "gamma_min_postwrite",
                    "D_min_unclipped_traj", "D_min_clipped_traj"])
        for r in grid:
            w.writerow([r["i_idx"], r["j_idx"], r["I_write"], r["well_amp"], r["seed"],
                        r["I_hold"], r["Gamma"], r["N"], r["Omega"], r["E"], r["Q"], r["C"],
                        *r["S"], r["gamma_min_full"], r["gamma_min_postwrite"],
                        r["D_min_unclipped_traj"], r["D_min_clipped_traj"]])

    print("\nWrote ../data/response_matrix_pathwayE.{json,csv} and response_matrix_pathwayE_summary.json")
    return summary


if __name__ == "__main__":
    main()
