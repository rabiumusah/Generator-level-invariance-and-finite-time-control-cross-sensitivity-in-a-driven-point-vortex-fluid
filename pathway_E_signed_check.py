"""
pathway_E_signed_check.py -- signed local Jacobian check at the interior
grid center (I0,bw0)=(1.2,1.5) for the pathway-E (full-field pinning)
closure, directly analogous to the "baseline" (no ablation) entry of
pathway_ablation.py but with VortexSystem constructed under
pin_mode="full" instead of "mag_only".

Reuses response_matrix_pathwayE.run_point verbatim (no ablation applied):
the same four grid points immediately adjacent to the interior center,
(I,bw) = (0.9,1.5), (1.5,1.5), (1.2,1.0), (1.2,2.0), with the manuscript's
own seed = 1000+100*i_idx+j_idx convention, and the identical central-
difference construction used throughout this Supplement:
    dGamma/dI|center  = (Gamma(1.5,1.5)-Gamma(0.9,1.5)) / 0.6
    dGamma/dbw|center = (Gamma(1.2,2.0)-Gamma(1.2,1.0)) / 1.0
(and the same for |S4|). This supplies the pathway-E analogue of the
signed check reported for pathways A--D via pathway_ablation.json's
"baseline" entry (SM Sec. S-jacobian).
"""
from __future__ import annotations
import json
from response_matrix_pathwayE import run_point

NEIGHBORS = dict(
    I_minus=dict(I=0.9, bw=1.5, i_idx=1, j_idx=2),
    I_plus=dict(I=1.5, bw=1.5, i_idx=3, j_idx=2),
    bw_minus=dict(I=1.2, bw=1.0, i_idx=2, j_idx=1),
    bw_plus=dict(I=1.2, bw=2.0, i_idx=2, j_idx=3),
)
D_I = 1.5 - 0.9   # = 0.6
D_BW = 2.0 - 1.0  # = 1.0
DELTA_I_GRID = 1.2   # full swept range, for range-scaled J_c^(u)
DELTA_BW_GRID = 2.0


def main():
    pts = {}
    for name, spec in NEIGHBORS.items():
        seed = 1000 + 100 * spec["i_idx"] + spec["j_idx"]
        r = run_point(spec["I"], spec["bw"], seed)
        pts[name] = dict(I=spec["I"], bw=spec["bw"], seed=seed,
                          Gamma=r["Gamma"], S4=r["S"][4], N=r["N"])

    dGamma_dI = (pts["I_plus"]["Gamma"] - pts["I_minus"]["Gamma"]) / D_I
    dGamma_dbw = (pts["bw_plus"]["Gamma"] - pts["bw_minus"]["Gamma"]) / D_BW
    dS4_dI = (pts["I_plus"]["S4"] - pts["I_minus"]["S4"]) / D_I
    dS4_dbw = (pts["bw_plus"]["S4"] - pts["bw_minus"]["S4"]) / D_BW

    out = dict(
        points=pts,
        dGamma_dI=dGamma_dI, dGamma_dbw=dGamma_dbw,
        dS4_dI=dS4_dI, dS4_dbw=dS4_dbw,
        J_c_scaled=dict(
            Gamma_I=dGamma_dI * DELTA_I_GRID, Gamma_bw=dGamma_dbw * DELTA_BW_GRID,
            S4_I=dS4_dI * DELTA_I_GRID, S4_bw=dS4_dbw * DELTA_BW_GRID,
        ),
        note=("Pathway-E (pin_mode='full') analogue of pathway_ablation.json's "
              "'baseline' entry: same four neighbor grid points, seeds, and "
              "central-difference construction as pathways A--D, only the "
              "pinning closure differs (full-field vs magnetization-only)."),
    )
    print(json.dumps(out, indent=2, default=float))
    with open("../data/pathway_E_signed_check.json", "w") as f:
        json.dump(out, f, indent=2, default=float)
    print("\nWrote ../data/pathway_E_signed_check.json")


if __name__ == "__main__":
    main()
