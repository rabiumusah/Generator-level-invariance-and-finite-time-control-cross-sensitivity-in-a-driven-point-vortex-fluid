"""
response_matrix.py -- Route A empirical cross-talk measurement.

Runs a genuine 2D grid sweep over (I_write, well_amp) at fixed P=4
magnetization wells, using an abbreviated write->split->hold protocol
(reusing the exact physics of run_protocol.run_cycle: ring-seeded rim
injection, F_mag-triggered nucleation, the closed-form hold-current fixed
point of the circulation law). For every grid point we record the achieved
held circulation Gamma and the dominant well-count harmonic |S_4| of the
boundary vorticity spectrum, then build the empirical response matrix

    d Gamma / d I |_bw ,   d Gamma / d b_w |_I
    d |S_4| / d I |_bw ,   d |S_4| / d b_w |_I

by central finite differences across the grid, and the two cross-talk
ratios defined in the Route A brief:

    eps_1 = (|dGamma/dbw| * Delta_bw) / (|dGamma/dI| * Delta_I)
    eps_2 = (d|S4|/dI * Delta_I)     / (d|S4|/dbw * Delta_bw)

(each partial derivative multiplied by the full swept range of its own
parameter, so the two ratios compare "how much does the observable move
if you sweep b_w across its whole explored range" against "...if you sweep
I across its whole explored range" -- a fair, dimensionally sensible
comparison since I and b_w are different dimensionless quantities with no
shared natural unit.)

This is a single-seed-per-grid-point characterization (seed fixed per grid
point, not ensemble-averaged over multiple seeds) -- stated explicitly here
and in the manuscript, per the project's honesty requirements. Every number
below is the direct, unmodified output of running the code; nothing is
hand-typed or fitted.
"""
from __future__ import annotations
import json, time, csv
import numpy as np

import pmvm_core as core
from pmvm_hamiltonian import VortexSystem

# ---------------------------------------------------------------- protocol

def _traj_diagnostics(sys_, p, I_t):
    """Per-step well-posedness diagnostics: min_i|Gamma_i(t)| and min_i D_i(t),
    D_i = 1 - drag_coeff * mu_local(r_i,t) (the manuscript's stated, unclipped
    formula) alongside the clipped value np.clip(D_i,0,1) that is what the
    integrated code actually multiplies the conservative velocity by in
    VortexSystem._velocities. Both are reported; see the admissible-domain
    disclosure in the manuscript for the reconciliation between the two.
    Returns (+inf, +inf, +inf) for an empty (fully annihilated) vortex set,
    which cannot happen for a well-posed trajectory and is left un-clamped
    on purpose so it would visibly show up as a non-finite entry in the
    aggregated minima rather than silently passing as 0.
    """
    if len(sys_.gamma) == 0:
        return np.inf, np.inf, np.inf
    gmin = float(np.min(np.abs(sys_.gamma)))
    x, y = sys_.xy[:, 0], sys_.xy[:, 1]
    mu_local = core.hartmann_rate(x, y, p, I_t, sys_.well_signs, sys_.well_amp)
    D_unclipped = 1.0 - sys_.drag_coeff * mu_local
    D_clipped = np.clip(D_unclipped, 0.0, 1.0)
    return gmin, float(np.min(D_unclipped)), float(np.min(D_clipped))


def run_point(I_write: float, well_amp: float, seed: int, P: int = 4,
              dt: float = 2e-3, t_write: float = 0.5, t_split: float = 1.0,
              t_hold: float = 1.5, I_split_frac: float = 0.5,
              nucleate_every: int = 20):
    """Abbreviated write->split->hold cycle (no erase phase -- irrelevant to
    the held-state cross-talk quantities measured here). Physics identical
    to run_protocol.run_cycle's write/split/hold logic; only the erase phase
    and full-length trajectory logging are dropped for sweep efficiency.
    Returns a dict with the held-state observables, spectrum, and the
    trajectory-wise well-posedness diagnostics (min|Gamma_i|, min D_i,
    unclipped and clipped) accumulated over every integration step."""
    p = core.DeviceParams(P_wells=P)
    well_signs = np.array([1.0, -1.0, 1.0, -1.0])[:P]
    sys_ = VortexSystem(p, well_signs, well_amp, seed=seed)
    sys_.seed_ring(Gamma_total=np.sign(I_write) * 5e-3, n_seed=16, r_seed=0.93, jitter=2e-3)

    t_total = t_write + t_split + t_hold
    n_steps = int(t_total / dt)
    phase_bounds = dict(write=t_write, split=t_write + t_split, hold=t_total)

    I_hold_value = [0.0]
    hold_locked = [False]

    # gamma_min_full includes the t=0 rim-seed ring, whose per-vortex
    # circulation (Gamma_total/n_seed, here ~1e-3/16) is a deliberately weak
    # numerical placeholder for the continuous edge-injection sheet (see
    # seed_ring docstring) and is therefore, by construction, the trajectory
    # minimum almost every time. gamma_min_postwrite excludes this placeholder
    # phase and starts accumulating only once the write pulse has finished
    # (t >= t_write), i.e. over the split+hold dynamics that determine the
    # actually reported held state -- the physically meaningful quantity for
    # the admissible-domain claim. Both are reported; neither is discarded.
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


# --------------------------------------------------------------- grid sweep

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


def finite_diff_matrix(I_vals, bw_vals, field):
    """field[i,j] on the (I_vals, bw_vals) grid -> central-difference partials
    d(field)/dI and d(field)/dbw at every INTERIOR grid point (forward/backward
    one-sided differences dropped at the boundary to keep only true central
    differences, per the brief). Returns the two partial-derivative arrays
    (shape (len(I)-2, len(bw)-2)) and their grid-mean values."""
    I_vals = np.asarray(I_vals, dtype=float)
    bw_vals = np.asarray(bw_vals, dtype=float)
    nI, nB = len(I_vals), len(bw_vals)
    dFdI = np.full((nI, nB), np.nan)
    dFdB = np.full((nI, nB), np.nan)
    for i in range(1, nI - 1):
        for j in range(nB):
            dFdI[i, j] = (field[i + 1, j] - field[i - 1, j]) / (I_vals[i + 1] - I_vals[i - 1])
    for i in range(nI):
        for j in range(1, nB - 1):
            dFdB[i, j] = (field[i, j + 1] - field[i, j - 1]) / (bw_vals[j + 1] - bw_vals[j - 1])
    # interior-only (both directions defined) mean, used for the summary ratios
    dFdI_interior = dFdI[1:-1, 1:-1]
    dFdB_interior = dFdB[1:-1, 1:-1]
    return dFdI, dFdB, float(np.nanmean(np.abs(dFdI_interior))), float(np.nanmean(np.abs(dFdB_interior)))


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

    # eps_1: circulation cross-talk -- range-normalized sensitivity of Gamma
    # to b_w (the "should-be-blocked-in-spirit, but not theorem-protected"
    # channel (ii)) vs. to I (the intended, direct control channel (i)).
    eps_1 = (abs(dGdB_mean) * Delta_bw) / (abs(dGdI_mean) * Delta_I)
    # eps_2: topology cross-talk -- range-normalized sensitivity of |S_4| to
    # I (the channel that Theorem 2 proves has NO direct/symmetry route, so
    # any nonzero value here is entirely the indirect I->Gamma->|S4| channel)
    # vs. to b_w (the direct, intended topology control).
    eps_2 = (abs(dS4dI_mean) * Delta_I) / (abs(dS4dB_mean) * Delta_bw)

    # Trajectory-wise well-posedness diagnostics, aggregated over all 25 grid
    # points (see _traj_diagnostics / run_point). gamma_min_full is dominated,
    # at essentially every grid point, by the deliberately weak t=0 rim-seed
    # ring (a numerical placeholder, not a physical vortex); gamma_min_postwrite
    # excludes that placeholder phase and is the physically meaningful
    # circulation-nondegeneracy minimum. D_min_unclipped is the manuscript's
    # literal, unclipped formula D_i=1-drag*mu_i; D_min_clipped is what the
    # integrated code (VortexSystem._velocities) actually multiplies the
    # conservative velocity by, np.clip(D_i,0,1).
    Gamma_min_full_all = min(r["gamma_min_full"] for r in grid)
    Gamma_min_postwrite_all = min(r["gamma_min_postwrite"] for r in grid)
    D_min_unclipped_all = min(r["D_min_unclipped_traj"] for r in grid)
    D_min_clipped_all = min(r["D_min_clipped_traj"] for r in grid)

    summary = dict(
        I_vals=I_vals, bw_vals=bw_vals, P=P,
        Delta_I=Delta_I, Delta_bw=Delta_bw,
        dGamma_dI_mean=dGdI_mean, dGamma_dbw_mean=dGdB_mean,
        dS4_dI_mean=dS4dI_mean, dS4_dbw_mean=dS4dB_mean,
        eps_1_circulation_crosstalk=eps_1,
        eps_2_topology_crosstalk=eps_2,
        eps_1_definition="|dGamma/dbw|_I * Delta_bw / (|dGamma/dI|_bw * Delta_I), grid-mean central "
                          "differences over interior points",
        eps_2_definition="|dS4/dI|_bw * Delta_I / (|dS4/dbw|_I * Delta_bw), grid-mean central "
                          "differences over interior points, S4=|S_P| dominant well-count harmonic (P=4)",
        note="single seed per grid point (seed = 1000 + 100*i_idx + j_idx); not ensemble-averaged.",
        Gamma_min_full_all=Gamma_min_full_all,
        Gamma_min_postwrite_all=Gamma_min_postwrite_all,
        D_min_unclipped_all=D_min_unclipped_all,
        D_min_clipped_all=D_min_clipped_all,
    )
    print("\n=== response matrix summary ===")
    for k, v in summary.items():
        print(f"  {k}: {v}")

    with open("../data/response_matrix.json", "w") as f:
        json.dump(dict(grid=grid, Gamma_grid=Gamma_grid.tolist(), S4_grid=S4_grid.tolist(),
                        N_grid=N_grid.tolist(), dGamma_dI=dGdI.tolist(), dGamma_dbw=dGdB.tolist(),
                        dS4_dI=dS4dI.tolist(), dS4_dbw=dS4dB.tolist(), summary=summary), f, indent=2)

    with open("../data/response_matrix_summary.json", "w") as f:
        json.dump(summary, f, indent=2)

    with open("../data/response_matrix.csv", "w", newline="") as f:
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

    print("\nWrote ../data/response_matrix.{json,csv} and response_matrix_summary.json")
    return summary


if __name__ == "__main__":
    main()
