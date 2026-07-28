#!/usr/bin/env python3
"""Rebuild homogeneous and branch diagnostics across the Turing threshold."""
from __future__ import annotations
import argparse
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.linalg import eigvals
from fluctuation_analysis import (
    DP_STRIPE, DQ_STRIPE, D_SCALAR, KSTAR, NU, branch_critical_amplitude,
    branch_diffusion, branch_jacobian, branch_summary, continue_branches,
    gaussian_cp_margin, lyapunov_residual, mode_pair_metrics,
    period_cell_npt_minimizer, reflection_fixed_spectral_abscissa,
    stationary_covariance, symplectic_eigenvalues,
)

NEGATIVE = [-0.4, -0.3, -0.2, -0.1, -0.05]
POSITIVE = [0.05, 0.075, 0.1, 0.125, 0.15, 0.175, 0.2, 0.25, 0.3, 0.35, 0.4]


def homogeneous_row(lam: float, Dq: float, Dp: float, state: str) -> dict[str, float | str]:
    profile = np.zeros((12, 2), dtype=float)
    A = branch_jacobian(profile, lam, Dq=Dq, Dp=Dp)
    D = branch_diffusion(profile, Dq=Dq, Dp=Dp)
    V = stationary_covariance(A, D)
    m = mode_pair_metrics(V, KSTAR)
    return {
        "lambda": lam,
        "state": state,
        "spectral_abscissa": float(np.max(np.real(eigvals(A)))),
        "locking_rate": np.nan,
        "relative_lyapunov_residual": lyapunov_residual(A, D, V),
        "min_full_symplectic_eigenvalue": float(np.min(symplectic_eigenvalues(V))),
        "cp_margin": gaussian_cp_margin(A, D),
        "kstar_nu_physical": m["nu_physical"],
        "kstar_nu_pt": m["nu_pt"],
        "kstar_logarithmic_negativity": m["logarithmic_negativity"],
        "phase_tangent_overlap": np.nan,
        "critical_amplitude": np.nan,
        "leading_critical_amplitude": np.nan,
        "critical_amplitude_relative_deviation": np.nan,
        "reflection_fixed_spectral_abscissa": np.nan,
        "leading_radial_eigenvalue": np.nan,
        "reflection_fixed_relative_deviation": np.nan,
        "period_cell_npt_min_mode": np.nan,
        "period_cell_npt_min_k_over_pi": np.nan,
        "period_cell_npt_min": np.nan,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", type=Path, default=Path("generated"))
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    rows = [homogeneous_row(lam, DQ_STRIPE, DP_STRIPE, "homogeneous") for lam in NEGATIVE]
    branches = continue_branches(POSITIVE)
    for lam in sorted(branches):
        profile = branches[lam].profile
        s = branch_summary(profile, lam)
        amp = branch_critical_amplitude(profile)
        amp_lead = float(np.sqrt(lam / (8.0 * np.sqrt(3.0) * NU)))
        alpha_ref = reflection_fixed_spectral_abscissa(profile, lam)
        alpha_lead = -0.5 * lam
        mode, k_over_pi, nu_min = period_cell_npt_minimizer(profile, lam)
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
            "critical_amplitude": amp,
            "leading_critical_amplitude": amp_lead,
            "critical_amplitude_relative_deviation": abs(amp / amp_lead - 1.0),
            "reflection_fixed_spectral_abscissa": alpha_ref,
            "leading_radial_eigenvalue": alpha_lead,
            "reflection_fixed_relative_deviation": abs(alpha_ref / alpha_lead - 1.0),
            "period_cell_npt_min_mode": mode,
            "period_cell_npt_min_k_over_pi": k_over_pi,
            "period_cell_npt_min": nu_min,
        })
    pd.DataFrame(rows).to_csv(args.out / "stripe_branch_linearized_across_threshold.csv", index=False)

    controls = [
        homogeneous_row(-0.2, D_SCALAR, D_SCALAR, "homogeneous_scalar_transport_same_lambda"),
        homogeneous_row(0.4, D_SCALAR, D_SCALAR, "homogeneous_scalar_transport_additional"),
    ]
    for row in controls:
        row["Dq"] = D_SCALAR
        row["Dp"] = D_SCALAR
    pd.DataFrame(controls).to_csv(args.out / "stripe_branch_linearized_controls.csv", index=False)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
