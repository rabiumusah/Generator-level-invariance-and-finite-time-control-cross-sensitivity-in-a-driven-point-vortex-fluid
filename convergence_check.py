"""
convergence_check.py -- dt and n_theta numerical-convergence check requested
by reviewer round 18 (Priority item 5). Scope, stated honestly: a
representative subset of the 5x5 (I_write, b_w) grid -- the two opposite
corners and the center -- rather than all 25 points, given the compute
budget available in this session. Both pin_mode settings are checked at
the center point; the corners are checked only for pin_mode="mag_only"
(the originally coded closure) to keep the sweep tractable.

Part 1 (time-step convergence): re-runs response_matrix.run_point at the
same seed and grid coordinates used in the reported pilot sweep, varying
only dt, and reports the held-state Gamma and |S4| at each resolution
relative to the finest resolution tested.

Part 2 (angular-sampling convergence): takes one converged held-state
vortex configuration and recomputes the boundary spectrum at a range of
n_theta, isolating the purely angular-discretization error identified in
the manuscript's Remark on the discrete FFT estimator, independent of any
time-integration error.

Every number below is the direct, unmodified output of running the code.
"""
from __future__ import annotations
import json, os, sys, time
import numpy as np

sys.path.insert(0, os.path.dirname(__file__))
import pmvm_core as core
from pmvm_hamiltonian import VortexSystem
from response_matrix import run_point
from response_matrix_pathwayE import run_point as run_point_pathwayE_orig

REPRESENTATIVE_POINTS = [
    # (label, I_write, well_amp, seed, pin_mode)
    # Round-21 self-assessment fix: the seeds for "center" and "corner_high"
    # previously did not match the formula seed=1000+100*i_idx+j_idx stated
    # in the trailing comment and in this module's own docstring ("same seed
    # ... used in the reported pilot sweep") -- 1212 and 1424 are not values
    # the grid's seed formula produces for any (i_idx,j_idx) in {0,...,4}^2.
    # Corrected to the actual grid-center/grid-corner seeds (1202, 1404) so
    # this check genuinely re-runs the same points reported in the pilot
    # sweep, as claimed.
    ("corner_low",  0.6, 0.5, 1000, "mag_only"),   # i=0,j=0 in the 5x5 grid, seed=1000+100*0+0
    ("center",      1.2, 1.5, 1202, "mag_only"),   # i=2,j=2, seed=1000+100*2+2
    ("corner_high", 1.8, 2.5, 1404, "mag_only"),   # i=4,j=4, seed=1000+100*4+4
    ("center_full", 1.2, 1.5, 1202, "full"),       # same center point, pathway-E closure
]

DT_VALUES = [4e-3, 2e-3, 1e-3, 5e-4]  # 2e-3 is the baseline used throughout the manuscript
NTHETA_VALUES = [64, 128, 256, 512, 1024, 2048]  # 256 is the baseline


def run_point_pathwayE(I_write, well_amp, seed, dt=2e-3):
    """Thin wrapper around response_matrix_pathwayE.run_point (imported
    directly, not reimplemented) so this script's call sites are uniform."""
    return run_point_pathwayE_orig(I_write, well_amp, seed, dt=dt)


def dt_convergence():
    results = {}
    for label, I_write, well_amp, seed, pin_mode in REPRESENTATIVE_POINTS:
        results[label] = []
        for dt in DT_VALUES:
            t0 = time.time()
            if pin_mode == "mag_only":
                r = run_point(I_write, well_amp, seed, dt=dt)
            else:
                r = run_point_pathwayE(I_write, well_amp, seed, dt=dt)
            elapsed = time.time() - t0
            row = dict(dt=dt, Gamma=r["Gamma"], S4=r["S"][4], N=r["N"], elapsed=elapsed)
            results[label].append(row)
            print(f"[dt-conv] {label} dt={dt:.0e} Gamma={r['Gamma']:.6f} "
                  f"S4={r['S'][4]:.6f} N={r['N']} ({elapsed:.1f}s)")
    # relative change vs finest dt tested, per point
    summary = {}
    for label, rows in results.items():
        finest = rows[-1]  # DT_VALUES sorted descending->smallest last
        rel_Gamma = [abs(r["Gamma"] - finest["Gamma"]) / max(abs(finest["Gamma"]), 1e-12) for r in rows]
        rel_S4 = [abs(r["S4"] - finest["S4"]) / max(abs(finest["S4"]), 1e-12) for r in rows]
        summary[label] = dict(dt_values=DT_VALUES, rel_Gamma_vs_finest=rel_Gamma, rel_S4_vs_finest=rel_S4,
                               N_values=[r["N"] for r in rows])
    return results, summary


def _build_held_state_system(I_write=1.2, well_amp=1.5, seed=1202, P=4, dt=2e-3,
                              t_write=0.5, t_split=1.0, t_hold=1.5, I_split_frac=0.5,
                              nucleate_every=20):
    """Reproduces response_matrix.run_point's exact integration loop (same
    API calls: inject_ring(alpha,I_t,dt), maybe_nucleate(I_t), step_rk4),
    but returns the live VortexSystem object instead of a summary dict, so
    the boundary spectrum can be recomputed at multiple n_theta afterward
    from the identical held-state configuration."""
    p = core.DeviceParams(P_wells=P)
    well_signs = np.array([1.0, -1.0, 1.0, -1.0])[:P]
    sys_ = VortexSystem(p, well_signs, well_amp, seed=seed)
    sys_.seed_ring(Gamma_total=np.sign(I_write) * 5e-3, n_seed=16, r_seed=0.93, jitter=2e-3)

    t_total = t_write + t_split + t_hold
    n_steps = int(t_total / dt)
    phase_bounds = dict(write=t_write, split=t_write + t_split, hold=t_total)
    I_hold_value = [0.0]
    hold_locked = [False]

    def I_profile(t):
        if t < t_write:
            return I_write
        elif t < phase_bounds["split"]:
            return np.sign(I_write) * I_split_frac * abs(I_write)
        else:
            return I_hold_value[0]

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
    return sys_


def ntheta_convergence():
    # build one representative held-state configuration (center point, baseline dt),
    # via the exact same integration loop as the reported response-matrix sweep
    sys_ = _build_held_state_system()

    rows = []
    for n_th in NTHETA_VALUES:
        S = sys_.spectrum(n_th=n_th)
        rows.append(dict(n_theta=n_th, S=S.tolist(), S4=float(S[4])))
        print(f"[ntheta-conv] n_theta={n_th} S4={S[4]:.8f}")

    finest = rows[-1]
    rel_S4 = [abs(r["S4"] - finest["S4"]) / max(abs(finest["S4"]), 1e-12) for r in rows]
    max_rel_all_m = []
    for r in rows:
        Sr = np.array(r["S"])
        Sf = np.array(finest["S"])
        max_rel_all_m.append(float(np.max(np.abs(Sr - Sf) / np.maximum(np.abs(Sf), 1e-12))))
    summary = dict(n_theta_values=NTHETA_VALUES, rel_S4_vs_finest=rel_S4, max_rel_any_m_vs_finest=max_rel_all_m)
    return rows, summary


def main():
    print("=== dt convergence ===")
    dt_results, dt_summary = dt_convergence()
    print("\n=== n_theta convergence ===")
    nth_results, nth_summary = ntheta_convergence()

    out = dict(dt_results=dt_results, dt_summary=dt_summary,
               ntheta_results=nth_results, ntheta_summary=nth_summary,
               representative_points=[dict(label=l, I_write=I, well_amp=w, seed=s, pin_mode=p)
                                       for l, I, w, s, p in REPRESENTATIVE_POINTS],
               note="Representative-subset convergence check (2 corners + center of the 5x5 grid, "
                    "both pinning closures at the center point), not a full 25-point sweep, per "
                    "session compute budget; see manuscript SM for scope disclosure.")
    out_path = os.path.join(os.path.dirname(__file__), "..", "data", "convergence_check.json")
    with open(out_path, "w") as f:
        json.dump(out, f, indent=2)
    print(f"\nWrote {out_path}")


if __name__ == "__main__":
    main()
