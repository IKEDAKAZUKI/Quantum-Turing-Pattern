#!/usr/bin/env python3
"""Rebuild the period-12 stripe and certified stationary Gaussian covariance."""
from __future__ import annotations
import argparse
from pathlib import Path
import pandas as pd
from fluctuation_analysis import (
    KSTAR, branch_summary, covariance_error_bound_fro, lyapunov_absolute_residual,
    lyapunov_operator_separation, pt_uncertainty_min_eigenvalue, uncertainty_min_eigenvalue,
    solve_bond_centered_branch, write_matrix,
)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--lambda", dest="lam", type=float, default=0.4)
    parser.add_argument("--out", type=Path, default=Path("generated"))
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)

    result = solve_bond_centered_branch(args.lam)
    summary = branch_summary(result.profile, args.lam)
    A, D, V = summary["A"], summary["D"], summary["V"]
    pair = summary["kstar_pair_covariance"]
    absolute_residual = lyapunov_absolute_residual(A, D, V)
    separation = lyapunov_operator_separation(A)
    error_bound = covariance_error_bound_fro(A, D, V)
    physical_min = uncertainty_min_eigenvalue(V)
    physical_certified_lower = physical_min - error_bound
    pt_min = pt_uncertainty_min_eigenvalue(pair)
    pt_certified_upper = pt_min + error_bound

    profile_df = pd.DataFrame({
        "site": range(result.profile.shape[0]),
        "q_star": result.profile[:, 0],
        "p_star": result.profile[:, 1],
    })
    profile_df.to_csv(args.out / "stripe_branch_profile_lambda0p4.csv", index=False)
    write_matrix(args.out / "stripe_branch_A_lambda0p4.csv", A)
    write_matrix(args.out / "stripe_branch_D_lambda0p4.csv", D)
    write_matrix(args.out / "stripe_branch_V_lambda0p4.csv", V)
    write_matrix(args.out / "stripe_branch_kstar_pair_covariance_lambda0p4.csv", pair)
    row = {
        "lambda": args.lam,
        "branch_residual_inf": result.residual_inf,
        "spectral_abscissa": summary["spectral_abscissa"],
        "locking_rate": summary["locking_rate"],
        "relative_lyapunov_residual": summary["relative_lyapunov_residual"],
        "absolute_lyapunov_residual_fro": absolute_residual,
        "lyapunov_operator_separation_2": separation,
        "covariance_error_bound_fro": error_bound,
        "min_full_symplectic_eigenvalue": summary["min_full_symplectic_eigenvalue"],
        "full_uncertainty_min_eigenvalue": physical_min,
        "full_uncertainty_certified_lower": physical_certified_lower,
        "cp_margin": summary["cp_margin"],
        "kstar": KSTAR,
        "kstar_nu_physical": summary["kstar_nu_physical"],
        "kstar_nu_pt": summary["kstar_nu_pt"],
        "kstar_logarithmic_negativity": summary["kstar_logarithmic_negativity"],
        "kstar_pt_uncertainty_min_eigenvalue": pt_min,
        "kstar_pt_uncertainty_certified_upper": pt_certified_upper,
        "phase_tangent_overlap": summary["phase_tangent_overlap"],
    }
    pd.DataFrame([row]).to_csv(args.out / "stripe_branch_covariance_summary.csv", index=False)
    print(pd.Series(row).to_string())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
