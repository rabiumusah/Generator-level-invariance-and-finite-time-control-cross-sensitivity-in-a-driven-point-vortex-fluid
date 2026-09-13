"""
pmvm_core.py -- Core physics module for the Programmable Magnetohydrodynamic
Vortex Memory (PMVM).

This module defines, from first principles and in a single consistent
dimensionless convention, every physical quantity used throughout the
manuscript and Supplementary Information:

    * device geometry and boundary fields (edge current + patterned bottom
      magnetization),
    * the vorticity-transport PDE terms (Hartmann damping, Lorentz vorticity
      source, edge injection) evaluated on arbitrary (x, y) arrays so they can
      be reused identically inside the PINN residual (pmvm_pinn.py) and the
      point-vortex simulator (pmvm_hamiltonian.py),
    * the disk (Dirichlet) Green's function with image vortices,
    * the eight macroscopic observables (C, Gamma, R, Omega, E, Sigma, N, Q)
      and the one-bit-per-observable 8-bit encoding,
    * dimensional scaling laws (density, write-speed, retention, fidelity)
      mapping the dimensionless simulation output onto graphene and
      liquid-metal realizations.

Units and convention
---------------------
All PDE/point-vortex computations are performed in DIMENSIONLESS units:
    length  -> R      (disk radius set to 1)
    velocity-> Gamma_0/(2 pi R)   (single-quantum monopole edge speed)
    time    -> R^2/Gamma_0 * 2*pi (viscous/advective unit)
    density, mu_0 (vacuum permeability) absorbed into O(1) coupling constants.
This mirrors standard practice in vortex-dynamics and MHD-cavity numerics
(e.g. Bozkaya & Tezer-Sezgin 2011) and is the same convention used in the
manuscript ("nondimensionalization uses (l, u0) = (R, Gamma_0/2*pi*R), giving
Re = 100, Re_m = 100 for the reported runs"). Physical (SI) numbers are
recovered only through the boxed scaling laws in Section VII, using the
material parameter table, never by direct dimensional simulation of a
nanometre-scale device (which is computationally intractable).

No vortex position is ever prescribed by hand anywhere in this module or in
the two solvers that import it: point-vortex creation events are decided by
thresholding the *computed* Lorentz source field F_mag on a grid (see
`detect_nucleation_sites`), and the resulting configuration is left free to
evolve under its own dynamics.
"""
from __future__ import annotations
import numpy as np
from dataclasses import dataclass, field
from typing import List, Tuple, Optional

# ----------------------------------------------------------------------------
# 0. Physical / dimensionless device parameters
# ----------------------------------------------------------------------------

@dataclass
class DeviceParams:
    R: float = 1.0                 # disk radius (=1 in dimensionless units)
    Gamma0: float = 1.0            # circulation quantum
    sigma_c: float = 0.07          # regularized vortex core radius (0.07 R)
    kappa_gamma: float = 1.0       # dissipative circulation-relaxation coefficient (kappa_Gamma
                                    # in the manuscript): sets mu_i in hartmann_rate/lorentz_source,
                                    # and hence eq:gam's decay rate and the drag factor D_i.
    kappa_pin: float = 1.0         # conservative (reactive) pinning coefficient (kappa_p in the
                                    # manuscript): sets U_pin/U_pin^(full) only. Structurally
                                    # independent of kappa_gamma -- both default to the same
                                    # numerical value (1.0) used throughout the pilot runs, but
                                    # nothing in the code ties them together; kappa_gamma=0 with
                                    # kappa_pin held finite is a literal, reachable state of this
                                    # dataclass (see verify_theorem2.py).
    alpha: float = 1.0             # edge injection coefficient (mu0*sigma_e/(rho h)), dimensionless
    nu: float = 1.0 / 100.0        # kinematic viscosity  (Re = 100)
    nu_H: float = 1.0e-2           # Hall viscosity (dimensionless)
    P_wells: int = 4               # number of magnetization wells
    r0_wells: float = 0.55         # well radius (fraction of R)
    w_well: float = 0.14           # Gaussian well width (fraction of R)
    B_edge_per_I: float = 1.0      # B_z^edge = B_edge_per_I * I(t)
    rim_annihilation: float = 0.985  # r/R beyond which vortices are annihilated


def well_centers(p: DeviceParams) -> np.ndarray:
    """Centers of the P magnetization wells (NOT vortex positions)."""
    phis = np.array([(2 * k - 1) * np.pi / p.P_wells for k in range(1, p.P_wells + 1)])
    xs = p.r0_wells * np.cos(phis)
    ys = p.r0_wells * np.sin(phis)
    return np.stack([xs, ys], axis=1), phis


# ----------------------------------------------------------------------------
# 1. Boundary fields: edge current + patterned bottom magnetization
# ----------------------------------------------------------------------------

def B_mag_field(x: np.ndarray, y: np.ndarray, p: DeviceParams,
                 well_signs: Optional[np.ndarray] = None,
                 well_amp: float = 1.0) -> np.ndarray:
    """Static patterned bottom-magnetization field B_z^mag(x,y).

    Sum of P Gaussian wells of amplitude `well_amp` and alternating (or
    user-specified) sign at radius r0_wells, azimuths (2p-1)pi/P. This is a
    *field*, not a set of vortex positions: it only biases where the Lorentz
    source F_mag is extremal; where vortices actually nucleate is decided
    later from the computed source field itself.
    """
    centers, _ = well_centers(p)
    if well_signs is None:
        well_signs = np.ones(p.P_wells)
    B = np.zeros_like(x, dtype=float)
    for (cx, cy), s in zip(centers, well_signs):
        r2 = (x - cx) ** 2 + (y - cy) ** 2
        B += s * well_amp * np.exp(-r2 / (2 * p.w_well ** 2))
    return B


def B_edge_field(p: DeviceParams, I: float) -> float:
    """Uniform axial field generated by the azimuthal rim current I(t)."""
    return p.B_edge_per_I * I


def B_total_field(x, y, p: DeviceParams, I: float, well_signs=None, well_amp=1.0):
    return B_mag_field(x, y, p, well_signs, well_amp) + B_edge_field(p, I)


def grad_B2(x, y, p: DeviceParams, I: float, well_signs=None, well_amp=1.0, h=1e-4):
    """Numerical gradient of B_z^2 (central differences); used by both the
    PINN residual (as a torch-differentiable analytic version, see
    pmvm_pinn.py) and the point-vortex nucleation detector."""
    Bp = B_total_field(x + h, y, p, I, well_signs, well_amp)
    Bm = B_total_field(x - h, y, p, I, well_signs, well_amp)
    dBdx = ((Bp ** 2 - Bm ** 2) / (2 * h))
    Bp = B_total_field(x, y + h, p, I, well_signs, well_amp)
    Bm = B_total_field(x, y - h, p, I, well_signs, well_amp)
    dBdy = ((Bp ** 2 - Bm ** 2) / (2 * h))
    return dBdx, dBdy


def hartmann_rate(x, y, p: DeviceParams, I: float, well_signs=None, well_amp=1.0):
    """mu(x,y;t) = kappa_gamma * B_z(x,y,t)^2 -- local Hartmann damping rate.
    Uses kappa_gamma (the dissipative coefficient), not kappa_pin."""
    B = B_total_field(x, y, p, I, well_signs, well_amp)
    return p.kappa_gamma * B ** 2


def lorentz_source(x, y, vx, vy, p: DeviceParams, I: float, well_signs=None, well_amp=1.0):
    """F_mag = -kappa_gamma (v x grad(B_z^2))_z = -kappa_gamma (vx dBy - vy dBx).
    Uses kappa_gamma, matching hartmann_rate's dissipative role: this is the
    continuum forcing term whose reduction gives eq:gam's decay rate."""
    dBdx, dBdy = grad_B2(x, y, p, I, well_signs, well_amp)
    return -p.kappa_gamma * (vx * dBdy - vy * dBdx)


def pinning_potential(x, y, p: DeviceParams, well_signs=None, well_amp=1.0):
    """U_pin(x,y) = kappa_pin * B_mag(x,y)^2 / 2 -- local magnetic pinning
    energy density felt by a vortex core (Sec. IV.2 of the theory note).
    Uses kappa_pin (the conservative/reactive coefficient), structurally
    independent of kappa_gamma above -- see DeviceParams."""
    Bm = B_mag_field(x, y, p, well_signs, well_amp)
    return 0.5 * p.kappa_pin * Bm ** 2


def pinning_force(x, y, p: DeviceParams, well_signs=None, well_amp=1.0, h=1e-4):
    """-grad(U_pin), used to build the E x B-like pinning drift."""
    Up = pinning_potential(x + h, y, p, well_signs, well_amp)
    Um = pinning_potential(x - h, y, p, well_signs, well_amp)
    Fx = -(Up - Um) / (2 * h)
    Up = pinning_potential(x, y + h, p, well_signs, well_amp)
    Um = pinning_potential(x, y - h, p, well_signs, well_amp)
    Fy = -(Up - Um) / (2 * h)
    return Fx, Fy


def pinning_potential_full(x, y, p: DeviceParams, I: float, well_signs=None, well_amp=1.0):
    """Full-field pinning closure, U_pin^(full) = kappa_pin*(B_mag+B_edge)^2/2.

    Route A of the manuscript's Pathway E: the continuum forcing term
    F_mag = -kappa_gamma (v x grad B_z^2)_z is built from the total field
    B_z = B_z^mag + B_z^edge (see lorentz_source/hartmann_rate above, both
    already called with B_total_field), but `pinning_potential` above uses
    B_z^mag alone. This function is the literal same-coefficient extension
    of `pinning_potential` to the total field, introducing no coefficient
    beyond the p.kappa_pin already used by pinning_potential; it isolates
    the closure-completion term identified analytically in the main text
    (Sec. Pathway E, Eq. eq:E),

        U_pin^(full)(r,t) = kappa_pin/2 * (B_mag(r) + B_edge(t))^2
                           = U_pin(r) + kappa_pin*B_mag(r)*B_edge(t) + kappa_pin*B_edge(t)^2/2,

    whose gradient adds kappa_pin*B_edge(t)*grad(B_mag(r)) (the uniform
    B_edge(t)^2/2 term has zero gradient) to the coded pinning force. As
    with `pinning_potential`, this closure uses kappa_pin (the conservative
    coefficient) and is structurally independent of kappa_gamma, which
    enters only hartmann_rate and lorentz_source above.
    """
    Bm = B_mag_field(x, y, p, well_signs, well_amp)
    Be = B_edge_field(p, I)
    return 0.5 * p.kappa_pin * (Bm + Be) ** 2


def pinning_force_full(x, y, p: DeviceParams, I: float, well_signs=None, well_amp=1.0, h=1e-4):
    """-grad(U_pin^(full)), the Pathway-E-augmented pinning drift force."""
    Up = pinning_potential_full(x + h, y, p, I, well_signs, well_amp)
    Um = pinning_potential_full(x - h, y, p, I, well_signs, well_amp)
    Fx = -(Up - Um) / (2 * h)
    Up = pinning_potential_full(x, y + h, p, I, well_signs, well_amp)
    Um = pinning_potential_full(x, y - h, p, I, well_signs, well_amp)
    Fy = -(Up - Um) / (2 * h)
    return Fx, Fy


# ----------------------------------------------------------------------------
# 2. Disk Dirichlet Green's function and point-vortex Biot--Savart law
# ----------------------------------------------------------------------------

def disk_image(z: complex, R: float) -> complex:
    """Image point of z=x+iy in a disk of radius R (Milne-Thomson circle
    theorem): z* = R^2 / conj(z)."""
    if abs(z) < 1e-12:
        return np.inf + 0j
    return R ** 2 / np.conj(z)


def biot_savart_velocity(x, y, vortices_xy: np.ndarray, vortices_gamma: np.ndarray,
                          R: float, reg: float) -> Tuple[np.ndarray, np.ndarray]:
    """Velocity induced at (x,y) [arrays] by N point vortices + their disk
    images, using the desingularized (Lamb-Oseen-like) kernel of core size
    `reg` to keep the dynamics regular at short range.

    v_x = -(1/2 pi) sum_i Gamma_i (y-y_i)/d_i^2 * (1-exp(-d_i^2/reg^2))
    v_y =  (1/2 pi) sum_i Gamma_i (x-x_i)/d_i^2 * (1-exp(-d_i^2/reg^2))
    plus identical contributions from the image vortices (opposite sign
    circulation), which enforce v_n=0 on r=R exactly (Milne-Thomson).
    """
    vx = np.zeros_like(x, dtype=float)
    vy = np.zeros_like(y, dtype=float)
    for (xi, yi), Gi in zip(vortices_xy, vortices_gamma):
        dx = x - xi
        dy = y - yi
        d2 = dx ** 2 + dy ** 2 + 1e-14
        kernel = (1.0 - np.exp(-d2 / reg ** 2)) / d2
        vx += -Gi / (2 * np.pi) * dy * kernel
        vy += Gi / (2 * np.pi) * dx * kernel
        # image vortex (opposite sign circulation) at R^2 r_i/|r_i|^2
        ri2 = xi ** 2 + yi ** 2
        if ri2 > 1e-10:
            xim = R ** 2 * xi / ri2
            yim = R ** 2 * yi / ri2
            dxm = x - xim
            dym = y - yim
            d2m = dxm ** 2 + dym ** 2 + 1e-14
            kernelm = (1.0 - np.exp(-d2m / reg ** 2)) / d2m
            vx += -(-Gi) / (2 * np.pi) * dym * kernelm
            vy += (-Gi) / (2 * np.pi) * dxm * kernelm
    return vx, vy


def induced_velocity_on_vortices(vortices_xy: np.ndarray, vortices_gamma: np.ndarray,
                                  R: float, reg: float) -> np.ndarray:
    """Velocity felt by each vortex due to all OTHER vortices + all images
    (self-image included: a single off-center vortex drifts under its own
    image, the classical 'vortex in a disk' rotation)."""
    n = len(vortices_gamma)
    out = np.zeros((n, 2))
    for i in range(n):
        xi, yi = vortices_xy[i]
        vx = vy = 0.0
        for j in range(n):
            if j == i:
                continue
            xj, yj = vortices_xy[j]
            Gj = vortices_gamma[j]
            dx, dy = xi - xj, yi - yj
            d2 = dx ** 2 + dy ** 2 + 1e-14
            kernel = (1.0 - np.exp(-d2 / reg ** 2)) / d2
            vx += -Gj / (2 * np.pi) * dy * kernel
            vy += Gj / (2 * np.pi) * dx * kernel
        # self-image contribution (every vortex, including i, has an image)
        for j in range(n):
            xj, yj = vortices_xy[j]
            Gj = vortices_gamma[j]
            rj2 = xj ** 2 + yj ** 2
            if rj2 > 1e-10:
                xim = R ** 2 * xj / rj2
                yim = R ** 2 * yj / rj2
                dx, dy = xi - xim, yi - yim
                d2 = dx ** 2 + dy ** 2 + 1e-14
                kernel = (1.0 - np.exp(-d2 / reg ** 2)) / d2
                vx += -(-Gj) / (2 * np.pi) * dy * kernel
                vy += (-Gj) / (2 * np.pi) * dx * kernel
        out[i] = [vx, vy]
    return out


# ----------------------------------------------------------------------------
# 3. Eight macroscopic observables and the 8-bit code
# ----------------------------------------------------------------------------

@dataclass
class Observables:
    C: float
    Gamma: float
    R_rel: float
    Omega: float
    E: float
    N: int
    Q: float
    N_nonrecip: float = 0.0


def cluster_vortices(vortices_xy: np.ndarray, vortices_gamma: np.ndarray, link_dist: float):
    """Coarse-grain a set of discrete point vortices into spatially coherent
    cores by single-linkage clustering at threshold `link_dist`. This
    separates the NUMERICAL discretization of the rim-injection sheet / core
    (many close point vortices approximating one smooth core) from the
    PHYSICAL vortex count used in the eight observables -- a standard
    coherent-structure identification step (cf. connected-component
    segmentation of the PINN vorticity field in pmvm_analyze_pinn.py), not a
    prescription of vortex positions: clustering only groups vortices that
    the dynamics has already brought close together."""
    n = len(vortices_gamma)
    if n == 0:
        return np.zeros((0, 2)), np.zeros((0,))
    parent = list(range(n))

    def find(a):
        while parent[a] != a:
            parent[a] = parent[parent[a]]
            a = parent[a]
        return a

    def union(a, b):
        ra, rb = find(a), find(b)
        if ra != rb:
            parent[ra] = rb

    for i in range(n):
        for j in range(i + 1, n):
            if np.sign(vortices_gamma[i]) == np.sign(vortices_gamma[j]):
                d = np.hypot(vortices_xy[i, 0] - vortices_xy[j, 0], vortices_xy[i, 1] - vortices_xy[j, 1])
                if d < link_dist:
                    union(i, j)
    groups = {}
    for i in range(n):
        r = find(i)
        groups.setdefault(r, []).append(i)
    xy_out, g_out = [], []
    for idxs in groups.values():
        g = np.sum(vortices_gamma[idxs])
        if abs(g) < 1e-12:
            continue
        x = np.sum(vortices_xy[idxs, 0] * vortices_gamma[idxs]) / g
        y = np.sum(vortices_xy[idxs, 1] * vortices_gamma[idxs]) / g
        xy_out.append([x, y])
        g_out.append(g)
    return np.array(xy_out), np.array(g_out)


def compute_observables_pointvortex(vortices_xy: np.ndarray, vortices_gamma: np.ndarray,
                                     p: DeviceParams, N_nonrecip: float = 0.0,
                                     cluster_link: Optional[float] = None) -> Observables:
    """Compute the eight macroscopic observables from a point-vortex
    configuration, using the equivalent-uniform-core convention (core radius
    p.sigma_c) for the enstrophy and energy of each core, exactly as defined
    analytically in Supplementary Note 3 & 5."""
    if len(vortices_gamma) == 0:
        return Observables(0.0, 0.0, 0.0, 0.0, 0.0, 0, 0.0, N_nonrecip)
    if cluster_link is not None:
        vortices_xy, vortices_gamma = cluster_vortices(np.asarray(vortices_xy), np.asarray(vortices_gamma), cluster_link)
        if len(vortices_gamma) == 0:
            return Observables(0.0, 0.0, 0.0, 0.0, 0.0, 0, 0.0, N_nonrecip)
    G = np.asarray(vortices_gamma, dtype=float)
    Gamma_tot = np.sum(G)
    C = np.sign(Gamma_tot) if abs(Gamma_tot) > 1e-9 else 0.0
    Gamma = abs(Gamma_tot)
    pos = np.sum(G[G > 0])
    neg = np.sum(np.abs(G[G < 0]))
    # Purity measure, symmetric under overall sign (chirality C already
    # records which sign dominates): 0 = single-sign (pure), 1 = perfectly
    # balanced dipole admixture. Using min/max rather than neg/pos avoids the
    # spurious R=1 that a naive neg/pos ratio would give for a purely
    # negative-circulation state (pos_sum=0), which is exactly as "pure" as
    # a purely positive one.
    R_rel = (min(pos, neg) / max(pos, neg)) if max(pos, neg) > 1e-12 else 0.0
    # Enstrophy: sum of per-core enstrophies (equivalent uniform-core disk of
    # radius sigma_c carries Gamma_i uniformly -> Omega_i = Gamma_i^2/(pi sigma_c^4) * pi sigma_c^2
    # (top-hat vorticity omega_i=Gamma_i/(pi sigma_c^2)) -> Omega_i = Gamma_i^2/(pi sigma_c^2).
    Omega = np.sum(G ** 2) / (np.pi * p.sigma_c ** 2)
    # Energy: pairwise point-vortex interaction energy (kinetic energy of the
    # induced flow) + self-energy of each regularized core, disk Green fn.
    n = len(G)
    E = 0.0
    for i in range(n):
        xi, yi = vortices_xy[i]
        # self energy (core finite-size regularization)
        E += (G[i] ** 2) / (4 * np.pi) * np.log(p.R / p.sigma_c)
        for j in range(i + 1, n):
            xj, yj = vortices_xy[j]
            dist = np.hypot(xi - xj, yi - yj) + 1e-12
            rjrjb = np.hypot(p.R ** 2 - (xi * xj + yi * yj), 0) / p.R  # |R^2 - r_i conj(r_j)|/R (real coords)
            zi = xi + 1j * yi
            zj = xj + 1j * yj
            denom = abs(p.R ** 2 - zi * np.conj(zj)) / p.R
            Gfun = np.log(dist / (denom + 1e-12))
            E += -(G[i] * G[j]) / (4 * np.pi) * Gfun * 2  # factor 2: i<j covers pair once, Hamiltonian sums i!=j
    N = n
    # Topological flow order parameter Q = (1/A) int sgn(w)|w| dA -> for point
    # vortices with uniform cores: Q = (1/A) sum sgn(Gamma_i)*|omega_i|*(pi sigma_c^2)
    #                                  = (1/A) sum Gamma_i   (since sgn*|w|*area = Gamma_i)
    A = np.pi * p.R ** 2
    Q = np.abs(np.sum(G)) / A * A / max(np.sum(np.abs(G)), 1e-12)  # normalized topological purity in [0,1]
    Q = abs(np.sum(G)) / max(np.sum(np.abs(G)), 1e-12)
    return Observables(C=C, Gamma=Gamma, R_rel=R_rel, Omega=Omega, E=E, N=N, Q=Q, N_nonrecip=N_nonrecip)


def bit_encode(obs: Observables, ref: Observables) -> Tuple[str, np.ndarray]:
    """One-bit-per-observable 8-bit code, Eq. (5) of the main text.

    ref supplies the reference scales (Gamma0, Omega0, E0) taken from the
    single-quantum monopole attractor.
    """
    b0 = 1 if obs.C > 0 else 0
    b1 = 1 if (obs.Gamma / max(ref.Gamma, 1e-9)) >= 2.0 else 0
    b2 = 1 if obs.R_rel >= (1.0 / 3.0) else 0
    b3 = 1 if (obs.Omega / max(ref.Omega, 1e-9)) >= 1.2 else 0
    b4 = 1 if (obs.E / max(ref.E, 1e-9)) >= 1.5 else 0
    b5 = 1 if obs.N >= 2 else 0
    b6 = 1 if abs(obs.N_nonrecip) >= 0.03 else 0
    b7 = 1 if obs.Q <= 0.6 else 0
    bits = np.array([b7, b6, b5, b4, b3, b2, b1, b0])
    word = "".join(str(b) for b in bits)
    return word, bits


def uncertainty_bound(Gamma: float, R: float, sigma_c: float, N: int) -> float:
    """Enstrophy floor Omega_min(Gamma,N) = Gamma^2/(2 pi R^2) + Gamma^2/(2 pi sigma_c^2 N)."""
    N = max(N, 1)
    return Gamma ** 2 / (2 * np.pi * R ** 2) + Gamma ** 2 / (2 * np.pi * sigma_c ** 2 * N)


# ----------------------------------------------------------------------------
# 4. Dynamic secondary-vortex nucleation from the computed source field
# ----------------------------------------------------------------------------

def detect_nucleation_sites(p: DeviceParams, I: float, vortices_xy: np.ndarray,
                             vortices_gamma: np.ndarray, well_signs=None, well_amp=1.0,
                             grid_n: int = 161, thresh_frac: float = 0.35,
                             min_sep: float = 0.10):
    """Evaluate the Lorentz source F_mag on a grid from the CURRENT velocity
    field (all existing vortices + images), locate its local extrema above a
    noise threshold, and return candidate nucleation sites (x, y, sign,
    integrated lobe circulation). No position is prescribed: the grid is
    scanned and extrema are found numerically from the actual computed
    field.
    """
    from scipy.ndimage import maximum_filter, minimum_filter
    lin = np.linspace(-p.R * 0.97, p.R * 0.97, grid_n)
    X, Y = np.meshgrid(lin, lin)
    mask = X ** 2 + Y ** 2 <= (p.R * 0.97) ** 2
    if len(vortices_gamma) > 0:
        VX, VY = biot_savart_velocity(X, Y, vortices_xy, vortices_gamma, p.R, p.sigma_c)
    else:
        VX, VY = np.zeros_like(X), np.zeros_like(Y)
    F = lorentz_source(X, Y, VX, VY, p, I, well_signs, well_amp)
    F = np.where(mask, F, 0.0)
    dA = (lin[1] - lin[0]) ** 2

    Fmax = np.max(np.abs(F)) if np.max(np.abs(F)) > 0 else 1.0
    thresh = thresh_frac * Fmax

    local_max = (maximum_filter(F, size=7) == F) & (F > thresh)
    local_min = (minimum_filter(F, size=7) == F) & (F < -thresh)

    sites = []
    for mask_arr, sign in [(local_max, +1), (local_min, -1)]:
        ys_idx, xs_idx = np.where(mask_arr)
        for yi, xi in zip(ys_idx, xs_idx):
            x0, y0 = X[yi, xi], Y[yi, xi]
            # avoid duplicate/too-close sites and sites too close to existing vortices
            too_close = False
            for (vx0, vy0) in vortices_xy:
                if np.hypot(x0 - vx0, y0 - vy0) < min_sep:
                    too_close = True
                    break
            for (sx, sy, _, _) in sites:
                if np.hypot(x0 - sx, y0 - sy) < min_sep:
                    too_close = True
                    break
            if too_close:
                continue
            # integrate F over a small lobe (disk of radius min_sep/2) around the peak
            lobe_mask = (X - x0) ** 2 + (Y - y0) ** 2 <= (min_sep / 2) ** 2
            gamma_lobe = np.sum(F[lobe_mask]) * dA
            sites.append((x0, y0, sign, gamma_lobe))
    return sites


# ----------------------------------------------------------------------------
# 5. Physical scaling laws (Sec. VII / Supplementary Note 7)
# ----------------------------------------------------------------------------

MATERIALS = {
    "graphene": dict(R=3e-6, h=0.3e-9, sigma_e=1e8, nu=1e-2, nu_H=1e-3,
                      sigma_core=3e-8, Gamma0=7.3e-4, T=4.0, B0=10.0, rho=1e3),
    "liquid_metal": dict(R=5e-5, h=5e-6, sigma_e=1e6, nu=1e-6, nu_H=0.0,
                           sigma_core=5e-6, Gamma0=1e-3, T=300.0, B0=0.05, rho=6.4e3),
}

PHYS = dict(mu0=4 * np.pi * 1e-7, kB=1.380649e-23, hbar=1.054571817e-34,
            h_planck=6.62607015e-34, m_e=9.1093837015e-31, e=1.602176634e-19,
            Phi0=2.067833848e-15)


def areal_density(material: str, N: int = 1) -> float:
    """rho_A = 8 bits / (N pi sigma_core^2), in bits/cm^2."""
    m = MATERIALS[material]
    rho_A_m2 = 8.0 / (N * np.pi * m["sigma_core"] ** 2)
    return rho_A_m2 * 1e-4  # bits/cm^2


def write_time_min(material: str) -> float:
    """SUPERSEDED estimate (Alfven-speed based), retained only for the
    explicit before/after comparison in Supplementary Note 7. This mixes a
    bulk Alfven speed v_A = B0/sqrt(mu0 rho) -- a wave speed of the *linearized
    field* equations -- with a vortex-core viscous time, neither of which is
    the actual physical process that sets the simulated write-saturation
    interval (Eq. circlaw reaching its steady state, Fig. 3a-b). It is NOT
    used for any number reported in the main text or tables; see
    `write_time_tau0` for the dimensionally consistent replacement actually
    used for the headline write-time figures."""
    m = MATERIALS[material]
    vA = m["B0"] / np.sqrt(PHYS["mu0"] * m["rho"])
    t_alfven = m["R"] / vA
    t_visc = m["sigma_core"] ** 2 / m["nu"]
    return max(t_alfven, t_visc)


def tau0(material: str) -> float:
    """Natural circulation (advective) time scale tau_0 = 4 pi^2 R^2 / Gamma_0.

    Derivation (Supplementary Note 7): the dimensionless simulation uses
    length unit R and circulation unit Gamma_0; a point vortex of circulation
    Gamma_0 induces an azimuthal speed Gamma_0/(2 pi r), so the characteristic
    advective time to traverse a distance ~2*pi*R at the edge speed
    Gamma_0/(2 pi R) is tau_0 = (2 pi R) / (Gamma_0/(2 pi R)) = 4 pi^2 R^2 /
    Gamma_0. This is the SAME combination (R^2/Gamma_0, up to the 4 pi^2
    geometric factor) that nondimensionalizes the vorticity-transport
    equation itself (Supplementary Note 1), so it is the unique dimensionally
    consistent bridge between simulation time and physical time -- unlike an
    Alfven-speed estimate, it requires no additional field (B0) or density
    (rho) input that does not already appear in Eq. (vorticity)."""
    m = MATERIALS[material]
    return 4 * np.pi ** 2 * m["R"] ** 2 / m["Gamma0"]


def E0(material: str) -> float:
    """Natural energy scale E_0 = rho * h * Gamma_0^2.

    Derivation (Supplementary Note 7): the point-vortex Hamiltonian energy
    (Sec. compute_observables_pointvortex, self+interaction terms) has the
    dimensionless form E ~ Gamma^2/(4 pi) * (log factor), which is an AREAL
    energy density (units of Gamma^2 = [length^2/time]^2, i.e. energy per
    unit mass per unit depth in the 2D reduction). Recovering a true energy
    requires (i) the mass density rho (areal mass = rho * h for a film of
    thickness h) and (ii) the film thickness h, giving E_0 = rho * h *
    Gamma_0^2. Omitting the explicit h factor -- as an earlier internal
    estimate did -- silently treats E as an energy per unit depth rather than
    a total energy, understating true energies by a factor of order 1/h
    (enormous for a sub-nanometre graphene film). This function is the
    corrected bridge used throughout Table 2/3 and Sec. 'Scaling laws'."""
    m = MATERIALS[material]
    return m["rho"] * m["h"] * m["Gamma0"] ** 2


def write_time_tau0(material: str, t_write_dimensionless=(0.15, 0.5)):
    """Physical write time = tau_0 * t_write, where t_write is the
    dimensionless simulation write-saturation interval read directly off the
    circulation traces of Fig. 3a-b (genuine simulation output, not a fitted
    or assumed number). This is the bridge actually used for the headline
    write-time figures quoted in the main text and Table 2/3; `write_time_min`
    above (Alfven-speed based) is the superseded estimate discussed for
    contrast in Supplementary Note 7 and is not used for any reported number.
    Returns (t_lo, t_hi) in seconds."""
    t0 = tau0(material)
    lo, hi = t_write_dimensionless
    return t0 * lo, t0 * hi


def hartmann_rate_hold(material: str, I_hold: float) -> float:
    m = MATERIALS[material]
    Bz = PHYS["mu0"] * I_hold / m["h"]
    return m["sigma_e"] * Bz ** 2 / m["rho"]


def energy_barrier_kT(material: str, d_min_over_sigma: float = 3.0) -> float:
    """Inter-attractor energy barrier, in units of k_B T.

    NOTE ON DIMENSIONAL CONSISTENCY: the point-vortex Hamiltonian energy
    density rho*Gamma^2/(4 pi) has units of N (force), not J -- it is a
    per-unit-DEPTH (2D) energy areal density. Converting to a true energy
    requires multiplying by the film thickness h (E = h * rho * Gamma^2/(4 pi)
    * ln(d_min/sigma)), which was omitted in an earlier internal estimate.
    We apply that correction here. Because Gamma0 = h_Planck/m_e is the
    single-electron circulation quantum, the resulting barrier for a bare
    single-quantum monopole is extremely small in absolute terms; physically
    relevant memory states carry n*Gamma0 with n up to 4 (Sec. II.2), and
    multi-quantum, multi-vortex complexes (as realized dynamically in this
    work, N=9) carry correspondingly larger barriers via the sum over cores
    and their mutual interaction energy (Methods; Supplementary Note 4/7).
    This function returns the single-quantum, two-core estimate for
    reference; the tabulated production values use the actual simulated
    multi-vortex interaction energy (see run_parametric_sweeps.py).
    """
    m = MATERIALS[material]
    Ebarrier = m["h"] * (m["rho"] * m["Gamma0"] ** 2 / (4 * np.pi)) * np.log(d_min_over_sigma)
    return Ebarrier / (PHYS["kB"] * m["T"])


def retention_time(material: str, mu0_rate: float, Ebarrier_over_kT: float) -> float:
    m = MATERIALS[material]
    prefac = 1.0 / (mu0_rate + m["nu"] / m["sigma_core"] ** 2)
    return prefac * np.exp(Ebarrier_over_kT)


def fidelity(material: str, t: float) -> float:
    m = MATERIALS[material]
    if m["nu_H"] <= 0:
        return np.nan
    return 1.0 - np.exp(-m["nu_H"] * t / m["R"] ** 2)
