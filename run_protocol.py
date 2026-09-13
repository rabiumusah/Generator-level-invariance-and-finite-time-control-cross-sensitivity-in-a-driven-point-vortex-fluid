"""
run_protocol.py -- Drive the complete WRITE -> SPLIT/PIN -> HOLD -> READ ->
ERASE memory cycle with the dynamic point-vortex simulator and record every
observable as a genuine time series (no numbers are hand-entered).

Two cycles are run: a "monopole" cycle (wells off) and a "complex" cycle
(P=4 wells on), matching the two attractor classes analyzed with the PINN.
A third short run repeats the write step with reversed edge-current polarity
to measure the nonreciprocity index N_wr.
"""
from __future__ import annotations
import json, time, sys
import numpy as np

import pmvm_core as core
from pmvm_hamiltonian import VortexSystem


def run_cycle(tag: str, wells_on: bool, I_write=3.0, well_amp=1.2, dt=2e-3,
              t_write=0.5, t_split=1.0, t_hold=3.0, t_erase=1.2,
              I_split=0.5, seed=0, nucleate_every=25, log_every=10):
    p = core.DeviceParams()
    well_signs = np.array([1, -1, 1, -1]) if wells_on else np.zeros(p.P_wells)
    sys_ = VortexSystem(p, well_signs, well_amp if wells_on else 0.0, seed=seed)

    Gamma0 = p.Gamma0
    record = []
    step = 0
    t0 = time.time()

    def I_profile(t):
        if t < t_write:
            return I_write
        elif t < t_write + t_split:
            return np.sign(I_write) * I_split  # small assist current, sign-matched to the write polarity
        elif t < t_write + t_split + t_hold:
            # maintenance (hold) current: the closed-form fixed point of the
            # exact circulation law dGamma/dt = -mu0(I) Gamma + alpha I, i.e.
            # I_hold solves kappa_gamma (B_edge_per_I I_hold)^2 Gamma = alpha I_hold
            # => I_hold = alpha / (kappa_gamma B_edge_per_I^2 Gamma). This is the
            # aggregate steady-state circulation balance evaluated at the
            # split-phase value (Theorem 1 mechanism).
            return I_hold_value[0]
        else:
            te = t - (t_write + t_split + t_hold)
            return I_erase_amp[0] * np.sin(2 * np.pi * f_erase[0] * te)

    I_hold_value = [0.0]
    I_erase_amp = [6.0]
    f_erase = [8.0]

    # ---- WRITE: seed a weak rim injection ring, then continuously inject
    # circulation through it for the duration of the write pulse (exact
    # discrete realization of dGamma/dt = -mu0 Gamma + alpha I(t)) ----------
    sys_.seed_ring(Gamma_total=np.sign(I_write) * 5e-3, n_seed=16, r_seed=0.93, jitter=2e-3)

    t_total = t_write + t_split + t_hold + t_erase
    n_steps = int(t_total / dt)

    phase_bounds = dict(write=t_write, split=t_write + t_split,
                         hold=t_write + t_split + t_hold, erase=t_total)

    hold_locked = [False]
    held_state = {}

    for step in range(n_steps):
        t = sys_.t
        I_t = I_profile(t)

        if (not hold_locked[0]) and t >= phase_bounds["split"]:
            obs = sys_.observables()
            Gamma_target = max(obs.Gamma, 1e-6)
            sign_C = obs.C if obs.C != 0 else 1.0
            # Exact fixed point of dGamma/dt = -(mu0(I)+nu/sigma_c^2) Gamma + alpha I,
            # with mu0(I) = kappa_gamma (B_edge_per_I I)^2 (quadratic in I_hold):
            #   kappa_gamma B_edge_per_I^2 Gamma I_hold^2 - alpha I_hold + (nu/sigma_c^2) Gamma = 0
            a_coef = p.kappa_gamma * p.B_edge_per_I ** 2 * Gamma_target
            b_coef = -p.alpha
            c_coef = (p.nu / p.sigma_c ** 2) * Gamma_target
            disc = b_coef ** 2 - 4 * a_coef * c_coef
            if disc >= 0 and a_coef > 1e-12:
                I_hold_value[0] = sign_C * (-b_coef - np.sqrt(disc)) / (2 * a_coef)
            else:
                I_hold_value[0] = sign_C * p.alpha / (p.kappa_gamma * p.B_edge_per_I ** 2 * Gamma_target)
            hold_locked[0] = True

        # the rim boundary term alpha*I(t) is physically always present (it is
        # the boundary condition at r=R, cf. the closed-form circulation law
        # dGamma/dt = -mu0 Gamma + alpha I(t) that holds at all times, Eq. 2);
        # what distinguishes write/split/hold/erase is only the amplitude and
        # time-profile of I(t) itself.
        sys_.inject_ring(p.alpha, I_t, dt)

        if wells_on and phase_bounds["write"] <= t < phase_bounds["split"]:
            if step % nucleate_every == 0:
                sys_.maybe_nucleate(I_t)

        sys_.step_rk4(dt, I_t)

        if (not held_state) and sys_.t >= phase_bounds["hold"] - 1e-9:
            obs_h = sys_.observables()
            held_state.update(dict(
                t=float(sys_.t),
                observables=dict(C=float(obs_h.C), Gamma=float(obs_h.Gamma), R_rel=float(obs_h.R_rel),
                                  Omega=float(obs_h.Omega), E=float(obs_h.E), N=int(obs_h.N), Q=float(obs_h.Q)),
                vortices=[dict(x=float(x), y=float(y), gamma=float(g))
                          for (x, y), g in zip(sys_.xy, sys_.gamma)],
                spectrum=sys_.spectrum().tolist(),
            ))

        if step % log_every == 0:
            obs = sys_.observables()
            phase = "write" if t < phase_bounds["write"] else \
                    "split" if t < phase_bounds["split"] else \
                    "hold" if t < phase_bounds["hold"] else "erase"
            record.append(dict(t=float(t), I=float(I_t), phase=phase,
                                C=float(obs.C), Gamma=float(obs.Gamma), R_rel=float(obs.R_rel),
                                Omega=float(obs.Omega), E=float(obs.E), N=int(obs.N), Q=float(obs.Q)))

    wall = time.time() - t0
    final_obs = sys_.observables()
    spectrum = sys_.spectrum().tolist()

    result = dict(tag=tag, wells_on=wells_on, I_write=I_write, well_amp=well_amp,
                  dt=dt, phase_bounds=phase_bounds, wall_time_s=wall,
                  n_steps=n_steps, record=record,
                  final_observables=dict(C=float(final_obs.C), Gamma=float(final_obs.Gamma),
                                          R_rel=float(final_obs.R_rel), Omega=float(final_obs.Omega),
                                          E=float(final_obs.E), N=int(final_obs.N), Q=float(final_obs.Q)),
                  final_vortices=[dict(x=float(x), y=float(y), gamma=float(g))
                                   for (x, y), g in zip(sys_.xy, sys_.gamma)],
                  spectrum=spectrum,
                  held_state=held_state,
                  nucleation_events=sys_.nucleation_events)
    with open(f"../data/protocol_{tag}.json", "w") as f:
        json.dump(result, f, indent=2)
    ho = held_state.get("observables", {})
    print(f"[{tag}] done in {wall:.1f}s, n_steps={n_steps} | HELD state: N={ho.get('N')}, "
          f"Gamma={ho.get('Gamma', float('nan')):.4f}, Omega={ho.get('Omega', float('nan')):.4f}, "
          f"E={ho.get('E', float('nan')):.4f}, R_rel={ho.get('R_rel', float('nan')):.3f}, "
          f"Q={ho.get('Q', float('nan')):.3f} | post-ERASE: N={final_obs.N}, Gamma={final_obs.Gamma:.4f}")
    return result


if __name__ == "__main__":
    tag = sys.argv[1] if len(sys.argv) > 1 else "monopole"
    if tag == "monopole":
        run_cycle("monopole", wells_on=False, I_write=3.0, seed=0)
    elif tag == "complex":
        run_cycle("complex", wells_on=True, I_write=3.0, well_amp=1.2, seed=1)
    elif tag == "complex_neg":
        run_cycle("complex_neg", wells_on=True, I_write=-3.0, well_amp=1.2, seed=2)
