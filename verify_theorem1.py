"""
verify_theorem1.py -- Numerical sanity check of Theorem 1 (rigid-rotation
invariance of the topology observable set T = (N, Omega, E, Q, {|S_m|})).

This is NOT the evidence for Theorem 1 -- the theorem is proved analytically
(exact algebraic invariance of the disk Green's function and the boundary
Fourier transform under simultaneous z_i -> e^{i theta0} z_i) in the main
text/SI. This script is a numerical cross-check: take real, dynamically
generated vortex configurations from an actual run at several time points,
rigidly rotate them by a random angle, and confirm that N, Omega, E, Q and
|S_m| (m=0..7) agree with the un-rotated values to numerical (finite-
difference-free, closed-form) precision.
"""
from __future__ import annotations
import json
import numpy as np

import pmvm_core as core
from pmvm_hamiltonian import VortexSystem


def rotate_system(sys_, theta):
    """Return a fresh VortexSystem-like snapshot with all vortex positions
    rotated rigidly by theta about the origin; gamma unchanged."""
    c, s = np.cos(theta), np.sin(theta)
    Rmat = np.array([[c, -s], [s, c]])
    xy_rot = sys_.xy @ Rmat.T
    return xy_rot, sys_.gamma.copy()


def spectrum_from_state(xy, gamma, p, r_read=0.9, n_th=256):
    thetas = np.linspace(0, 2 * np.pi, n_th, endpoint=False)
    xr, yr = r_read * np.cos(thetas), r_read * np.sin(thetas)
    if len(gamma) == 0:
        return np.zeros(8)
    omega_r = np.zeros(n_th)
    for (xi, yi), Gi in zip(xy, gamma):
        d2 = (xr - xi) ** 2 + (yr - yi) ** 2
        omega_r += Gi / (np.pi * p.sigma_c ** 2) * np.exp(-d2 / p.sigma_c ** 2)
    Sm = np.fft.rfft(omega_r) / n_th
    return np.abs(Sm[:8])


def observables_from_state(xy, gamma, p):
    return core.compute_observables_pointvortex(xy, gamma, p, cluster_link=6.0 * p.sigma_c)


def main():
    rng = np.random.default_rng(42)
    p = core.DeviceParams(P_wells=4)
    well_signs = np.array([1.0, -1.0, 1.0, -1.0])
    sys_ = VortexSystem(p, well_signs, well_amp=1.5, seed=3)
    sys_.seed_ring(Gamma_total=5e-3, n_seed=16, r_seed=0.93, jitter=2e-3)

    dt = 2e-3
    t_write, t_split, t_hold = 0.5, 1.0, 1.5
    n_steps = int((t_write + t_split + t_hold) / dt)
    check_times = [0.6, 1.1, 1.6, 2.4, 3.0]
    results = []

    for step in range(n_steps):
        t = sys_.t
        I_t = 1.2 if t < t_write else (np.sign(1.2) * 0.6 if t < t_write + t_split else 0.3)
        sys_.inject_ring(p.alpha, I_t, dt)
        if t_write <= t < t_write + t_split and step % 20 == 0:
            sys_.maybe_nucleate(I_t)
        sys_.step_rk4(dt, I_t)

        if check_times and sys_.t >= check_times[0] - 1e-9:
            tc = check_times.pop(0)
            theta0 = rng.uniform(0, 2 * np.pi)
            xy_rot, g_rot = rotate_system(sys_, theta0)

            obs_raw = observables_from_state(sys_.xy, sys_.gamma, p)
            obs_rot = observables_from_state(xy_rot, g_rot, p)
            S_raw = spectrum_from_state(sys_.xy, sys_.gamma, p)
            S_rot = spectrum_from_state(xy_rot, g_rot, p)

            rec = dict(
                t=float(tc), n_vortices=int(len(sys_.gamma)), theta0=float(theta0),
                N_raw=int(obs_raw.N), N_rot=int(obs_rot.N),
                Omega_raw=float(obs_raw.Omega), Omega_rot=float(obs_rot.Omega),
                E_raw=float(obs_raw.E), E_rot=float(obs_rot.E),
                Q_raw=float(obs_raw.Q), Q_rot=float(obs_rot.Q),
                absS_raw=S_raw.tolist(), absS_rot=S_rot.tolist(),
                max_abs_S_diff=float(np.max(np.abs(S_raw - S_rot))),
                E_rel_diff=float(abs(obs_raw.E - obs_rot.E) / max(abs(obs_raw.E), 1e-14)),
                Omega_rel_diff=float(abs(obs_raw.Omega - obs_rot.Omega) / max(abs(obs_raw.Omega), 1e-14)),
                Q_abs_diff=float(abs(obs_raw.Q - obs_rot.Q)),
            )
            results.append(rec)
            print(f"t={tc:.2f} theta0={theta0:.3f} rad | N: {rec['N_raw']}=={rec['N_rot']} | "
                  f"Omega rel.diff={rec['Omega_rel_diff']:.2e} | E rel.diff={rec['E_rel_diff']:.2e} | "
                  f"Q abs.diff={rec['Q_abs_diff']:.2e} | max||S_m|_raw-|S_m|_rot|={rec['max_abs_S_diff']:.2e}")

    with open("../data/theorem1_verification.json", "w") as f:
        json.dump(results, f, indent=2)
    print("\nWrote ../data/theorem1_verification.json")
    print(f"Overall max |S_m| discrepancy across all checks: "
          f"{max(r['max_abs_S_diff'] for r in results):.3e}")
    print(f"Overall max Omega/E relative discrepancy: "
          f"{max(max(r['Omega_rel_diff'], r['E_rel_diff']) for r in results):.3e}")


if __name__ == "__main__":
    main()
