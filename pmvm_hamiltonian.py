"""
pmvm_hamiltonian.py -- Dynamic point-vortex simulator for the PMVM cell.

This is the operative time-dependent engine: it integrates the point-vortex
Hamiltonian equations of motion (Biot-Savart + disk images) with a
dissipative correction (local Hartmann decay of each core's circulation,
Hartmann-damped guiding-center drift) and drives the full write-hold-
read-erase protocol cycle through a single control waveform I(t) and a
fixed (on/off) bottom-magnetization well pattern.

CENTRAL DESIGN RULE (per the research brief): no vortex position is ever
prescribed.
  * WRITE seeds circulation as a ring of many small, weak point vortices
    distributed around the rim (representing the physically continuous,
    axisymmetric edge-current injection sheet) -- NOT as one vortex placed
    at the center. Whether/how they merge into a coherent primary vortex is
    an emergent N-body result of their mutual induction (co-rotating same-
    sign point-vortex rings are dynamically unstable to clumping -- Thomson
    1883; Dritschel 1985) plus Hartmann dissipation.
  * SECONDARY (satellite) vortices are nucleated only where the computed
    Lorentz source F_mag(x,y) = -kappa_gamma (v x grad B_z^2)_z exceeds a noise
    threshold on a scan grid (pmvm_core.detect_nucleation_sites); their
    positions are a numerical measurement of the flow, not an input.
  * Positions then evolve under their own dynamics for the remainder of the
    run (hold, read, erase); nothing is ever reset to a hand-picked point.

Equations of motion (Hamiltonian H = -1/(4 pi) sum_{i!=j} Gamma_i Gamma_j
G(r_i,r_j) - sum_i U_pin(r_i) + sum_i Gamma_i Psi_I(r_i), disk Green's
function G with images; the MINUS sign on the pinning term is not a typo --
code/energy_conservation_check.py verifies numerically, and a direct
substitution into the canonical bracket Gamma_i*dx_i/dt = dH/dy_i,
Gamma_i*dy_i/dt = -dH/dx_i confirms analytically, that the coded pinning
drift v_pin = (1/Gamma_i)(-grad U_pin) x zhat is canonical with respect to
H_pin = -sum_i U_pin(r_i), the opposite sign from an earlier, never-checked
version of this docstring):

    dr_i/dt = v_BS,i(other vortices + all images)         [conservative]
              + 2 c r_i phihat                              [edge rotation, Psi_I ~ c r^2]
              + (1/Gamma_i) (-grad U_pin(r_i)) x zhat        [pinning guiding-center drift]
    all conservative terms scaled by (1 - drag_coeff * mu_local(r_i,t))  [Hartmann drag]
    dGamma_i/dt = -mu_local(r_i,t) * Gamma_i                 [local Hartmann/core decay]

with mu_local(x,y,t) = kappa_gamma * B_z(x,y,t)^2 (pmvm_core.hartmann_rate).
kappa_gamma is the dissipative coefficient; the pinning potential U_pin
(and U_pin^(full) below) uses the structurally independent coefficient
kappa_pin -- see pmvm_core.DeviceParams.
"""
from __future__ import annotations
import json, time
import numpy as np

import pmvm_core as core


class VortexSystem:
    def __init__(self, p: core.DeviceParams, well_signs, well_amp,
                 drag_coeff=0.4, seed=0, max_vortices=40, pin_mode="mag_only"):
        """pin_mode selects which pinning closure enters the position
        equation of motion (Model, eq:pos):
          "mag_only" (default) -- U_pin = kappa_pin*B_mag^2/2, i.e. exactly
              the closure used throughout every previously reported run
              (Table 3/S3, Fig. 1-2, verify_theorem1.py). Passing no
              pin_mode argument reproduces those results unchanged.
          "full" -- U_pin^(full) = kappa_pin*(B_mag+B_edge)^2/2 (Pathway E,
              main text Eq. eq:E), the closure-completion term identified
              but not numerically exercised in the pilot runs.
        No other equation of motion, parameter, seed convention, or protocol
        timing is altered between the two modes.
        """
        assert pin_mode in ("mag_only", "full")
        self.p = p
        self.well_signs = np.asarray(well_signs, dtype=float)
        self.well_amp = well_amp
        self.drag_coeff = drag_coeff
        self.pin_mode = pin_mode
        self.rng = np.random.default_rng(seed)
        self.max_vortices = max_vortices
        self.xy = np.zeros((0, 2))
        self.gamma = np.zeros((0,))
        self.birth = np.zeros((0,))
        self.t = 0.0
        self.log = []
        self.nucleation_events = []

    # -------------------------------------------------------------- birth
    def seed_ring(self, Gamma_total, n_seed=16, r_seed=0.93, jitter=1e-3):
        """Seed the edge-injected circulation as a ring of many weak
        vortices (NOT a single central vortex): represents the continuous
        azimuthal edge-current injection sheet at r~R. Small random jitter
        breaks exact discrete symmetry (as in the perturbed-IC ensembles of
        the Statistics protocol), permitting the ring's own dynamical
        (Thomson/Dritschel) instability to drive emergent clumping.
        """
        thetas = np.linspace(0, 2 * np.pi, n_seed, endpoint=False)
        thetas = thetas + self.rng.normal(0, jitter, n_seed)
        r = r_seed + self.rng.normal(0, jitter, n_seed)
        x = r * np.cos(thetas)
        y = r * np.sin(thetas)
        g = np.full(n_seed, Gamma_total / n_seed)
        self.xy = np.concatenate([self.xy, np.stack([x, y], axis=1)], axis=0)
        self.gamma = np.concatenate([self.gamma, g])
        self.birth = np.concatenate([self.birth, np.full(n_seed, self.t)])
        self.ring_idx = np.arange(len(self.gamma) - n_seed, len(self.gamma))

    def inject_ring(self, alpha, I_t, dt):
        """Continuous rim injection: distributes alpha*I(t)*dt equally across
        the (still-distinct) ring vortices created by seed_ring, implementing
        the exact boundary source term of the circulation law
        dGamma_tot/dt = -mu0 Gamma + alpha I(t) at the level of the discrete
        ring representation of the injection sheet."""
        if not hasattr(self, "ring_idx") or len(self.ring_idx) == 0:
            return
        idx = self.ring_idx[self.ring_idx < len(self.gamma)]
        if len(idx) == 0:
            return
        self.gamma[idx] += alpha * I_t * dt / len(idx)

    # --------------------------------------------------------------- RHS
    def _velocities(self, I_t):
        n = len(self.gamma)
        if n == 0:
            return np.zeros((0, 2))
        v_bs = core.induced_velocity_on_vortices(self.xy, self.gamma, self.p.R, self.p.sigma_c)
        x, y = self.xy[:, 0], self.xy[:, 1]
        # edge rotation from Psi_I ~ c r^2 (c set small & prop to I^2, Theorem ii)
        c_edge = 0.05 * I_t ** 2
        v_rot = np.stack([-2 * c_edge * y, 2 * c_edge * x], axis=1)
        # pinning drift (guiding-center): (1/Gamma_i)(-grad U_pin) x zhat
        if self.pin_mode == "full":
            Fx, Fy = core.pinning_force_full(x, y, self.p, I_t, self.well_signs, self.well_amp)
        else:
            Fx, Fy = core.pinning_force(x, y, self.p, self.well_signs, self.well_amp)
        sign_g = np.where(self.gamma >= 0, 1.0, -1.0)
        safe_gamma = sign_g * np.maximum(np.abs(self.gamma), 1e-3)
        v_pin = np.stack([Fy / safe_gamma, -Fx / safe_gamma], axis=1)
        v_cons = v_bs + v_rot + v_pin
        mu_local = core.hartmann_rate(x, y, self.p, I_t, self.well_signs, self.well_amp)
        drag_factor = np.clip(1.0 - self.drag_coeff * mu_local, 0.0, 1.0)
        v_eff = v_cons * drag_factor[:, None]
        return v_eff

    def _gamma_dot(self, I_t):
        if len(self.gamma) == 0:
            return np.zeros((0,))
        x, y = self.xy[:, 0], self.xy[:, 1]
        mu_local = core.hartmann_rate(x, y, self.p, I_t, self.well_signs, self.well_amp)
        # radial-core (viscous) decay channel nu/sigma_c^2, additive to the
        # Hartmann channel (eigenvalue table, lambda_n ~ -(n^2 nu/sigma^2 + mu0))
        visc_rate = self.p.nu / self.p.sigma_c ** 2
        return -(mu_local + visc_rate) * self.gamma

    def step_rk4(self, dt, I_t):
        if len(self.gamma) == 0:
            self.t += dt
            return
        xy0, g0 = self.xy.copy(), self.gamma.copy()

        def deriv(xy, g):
            self.xy, self.gamma = xy, g
            v = self._velocities(I_t)
            gd = self._gamma_dot(I_t)
            return v, gd

        k1x, k1g = deriv(xy0, g0)
        k2x, k2g = deriv(xy0 + 0.5 * dt * k1x, g0 + 0.5 * dt * k1g)
        k3x, k3g = deriv(xy0 + 0.5 * dt * k2x, g0 + 0.5 * dt * k2g)
        k4x, k4g = deriv(xy0 + dt * k3x, g0 + dt * k3g)

        self.xy = xy0 + (dt / 6.0) * (k1x + 2 * k2x + 2 * k3x + k4x)
        self.gamma = g0 + (dt / 6.0) * (k1g + 2 * k2g + 2 * k3g + k4g)
        self.t += dt
        self._cleanup()

    def _cleanup(self):
        if len(self.gamma) == 0:
            return
        r = np.hypot(self.xy[:, 0], self.xy[:, 1])
        keep = (r < self.p.rim_annihilation) & (np.abs(self.gamma) > 1e-4)
        n_annihilated = np.sum(~keep)
        if n_annihilated > 0:
            self.nucleation_events.append(dict(t=float(self.t), type="annihilate", n=int(n_annihilated)))
        self.xy, self.gamma, self.birth = self.xy[keep], self.gamma[keep], self.birth[keep]
        # merge same-sign vortices closer than a core diameter
        merged = True
        while merged and len(self.gamma) > 1:
            merged = False
            n = len(self.gamma)
            for i in range(n):
                for j in range(i + 1, n):
                    d = np.hypot(self.xy[i, 0] - self.xy[j, 0], self.xy[i, 1] - self.xy[j, 1])
                    if d < 0.5 * self.p.sigma_c and np.sign(self.gamma[i]) == np.sign(self.gamma[j]):
                        g_new = self.gamma[i] + self.gamma[j]
                        x_new = (self.gamma[i] * self.xy[i, 0] + self.gamma[j] * self.xy[j, 0]) / g_new
                        y_new = (self.gamma[i] * self.xy[i, 1] + self.gamma[j] * self.xy[j, 1]) / g_new
                        keep_mask = np.ones(n, dtype=bool)
                        keep_mask[[i, j]] = False
                        self.xy = np.concatenate([self.xy[keep_mask], [[x_new, y_new]]], axis=0)
                        self.gamma = np.concatenate([self.gamma[keep_mask], [g_new]])
                        self.birth = np.concatenate([self.birth[keep_mask], [self.t]])
                        merged = True
                        break
                if merged:
                    break

    def maybe_nucleate(self, I_t, thresh_frac=0.3, seed_efficiency=1.0):
        if len(self.gamma) >= self.max_vortices:
            return
        sites = core.detect_nucleation_sites(self.p, I_t, self.xy, self.gamma,
                                              self.well_signs, self.well_amp,
                                              grid_n=101, thresh_frac=thresh_frac, min_sep=0.22)
        if not sites:
            return
        new_xy, new_g = [], []
        for (x0, y0, sign, gamma_lobe) in sites:
            g = seed_efficiency * gamma_lobe
            if abs(g) < 1e-3:
                continue
            new_xy.append([x0, y0])
            new_g.append(g)
        if new_xy:
            # NOTE: global neutrality (integral of F_mag over the whole disk
            # vanishes identically) is a domain-integral property, not a
            # per-detection-batch one -- forcing each small batch of detected
            # extrema to individually sum to zero would incorrectly cancel
            # genuine same-sign multi-well detections, so no artificial
            # rebalancing is applied here; each site's circulation is used
            # exactly as measured from the local lobe integral.
            new_g = np.array(new_g)
            self.xy = np.concatenate([self.xy, np.array(new_xy)], axis=0)
            self.gamma = np.concatenate([self.gamma, new_g])
            self.birth = np.concatenate([self.birth, np.full(len(new_g), self.t)])
            self.nucleation_events.append(dict(t=float(self.t), type="nucleate", n=len(new_g),
                                                sites=[[float(a), float(b), float(c)] for a, b, c in
                                                       zip(np.array(new_xy)[:, 0], np.array(new_xy)[:, 1], new_g)]))

    def observables(self, N_nonrecip=0.0, cluster_link=None):
        """cluster_link defaults to 6*sigma_c: groups the (possibly many)
        discrete point vortices used to numerically represent the rim
        injection sheet / nucleated cores into physically coherent vortex
        structures before computing N, R, Omega, E, Q (see
        core.cluster_vortices)."""
        if cluster_link is None:
            cluster_link = 6.0 * self.p.sigma_c
        return core.compute_observables_pointvortex(self.xy, self.gamma, self.p, N_nonrecip,
                                                      cluster_link=cluster_link)

    def spectrum(self, r_read=0.9, n_th=256, reg=None):
        reg = reg or self.p.sigma_c
        thetas = np.linspace(0, 2 * np.pi, n_th, endpoint=False)
        xr, yr = r_read * np.cos(thetas), r_read * np.sin(thetas)
        if len(self.gamma) == 0:
            return np.zeros(8)
        # boundary vorticity from the smoothed point-vortex field (desingularized
        # Gaussian-like core), evaluated at the read radius
        omega_r = np.zeros(n_th)
        for (xi, yi), Gi in zip(self.xy, self.gamma):
            d2 = (xr - xi) ** 2 + (yr - yi) ** 2
            omega_r += Gi / (np.pi * self.p.sigma_c ** 2) * np.exp(-d2 / self.p.sigma_c ** 2)
        Sm = np.fft.rfft(omega_r) / n_th
        return np.abs(Sm[:8])
