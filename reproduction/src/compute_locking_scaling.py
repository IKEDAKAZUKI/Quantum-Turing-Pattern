#!/usr/bin/env python3
"""Rebuild the phase-locking sweep on the commensurate stripe branch."""
from __future__ import annotations
import argparse
from pathlib import Path
import numpy as np
import pandas as pd
from fluctuation_analysis import branch_summary, continue_branches

DEFAULT_LAMBDAS = [0.4,0.35,0.3,0.25,0.2,0.175,0.15,0.125,0.1,0.075,0.05]


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--lambdas", type=float, nargs="+", default=DEFAULT_LAMBDAS)
    parser.add_argument("--out", type=Path, default=Path("generated"))
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    branches = continue_branches(args.lambdas)
    rows = []
    for lam in sorted(branches):
        s = branch_summary(branches[lam].profile, lam)
        rows.append({
            "lambda": lam,
            "state": "bond_centered_branch",
            "spectral_abscissa": s["spectral_abscissa"],
            "locking_rate": s["locking_rate"],
            "relative_lyapunov_residual": s["relative_lyapunov_residual"],
            "min_full_symplectic_eigenvalue": s["min_full_symplectic_eigenvalue"],
            "cp_margin": s["cp_margin"],
            "kstar_nu_physical": s["kstar_nu_physical"],
            "kstar_nu_pt": s["kstar_nu_pt"],
            "kstar_logarithmic_negativity": s["kstar_logarithmic_negativity"],
            "phase_tangent_overlap": s["phase_tangent_overlap"],
            "locking_over_lambda5": s["locking_rate"] / lam**5,
        })
    df = pd.DataFrame(rows)
    small = df[(df["lambda"] >= 0.05) & (df["lambda"] <= 0.2)]
    slope, intercept = np.polyfit(np.log(small["lambda"]), np.log(small["locking_rate"]), 1)
    df["small_window_compensated_fit"] = np.exp(intercept) * df["lambda"] ** (slope - 5.0)
    df.to_csv(args.out / "stripe_phase_locking_compensated.csv", index=False)
    print(f"small-window locking exponent={slope:.12g}")
    return 0

if __name__ == "__main__":
    raise SystemExit(main())
