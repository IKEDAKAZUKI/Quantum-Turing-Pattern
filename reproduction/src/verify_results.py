#!/usr/bin/env python3
"""Scientific verifier for stripe and isotropic Gaussian fluctuation results."""
from __future__ import annotations

import argparse
from pathlib import Path

import mpmath as mp
import numpy as np
import pandas as pd

from fluctuation_analysis import (
    DP_STRIPE, DQ_STRIPE, D_SCALAR, KSTAR, branch_diffusion, branch_jacobian,
    covariance_error_bound_fro, fourier_symplectic, gaussian_cp_margin, homogeneous_pair_covariance,
    homogeneous_parameters, lyapunov_absolute_residual, lyapunov_operator_separation,
    lyapunov_residual, mode_pair_metrics, pt_uncertainty_min_eigenvalue, read_matrix,
    solve_bond_centered_branch, standing_to_traveling_symplectic, stationary_covariance,
    symplectic_eigenvalues, uncertainty_min_eigenvalue,
)


class Checks:
    def __init__(self) -> None:
        self.rows: list[tuple[str, bool, str]] = []

    def add(self, name: str, ok: bool, detail: str) -> None:
        self.rows.append((name, bool(ok), detail))
        print(f"{'PASS' if ok else 'FAIL'}  {name}: {detail}")

    def close(self) -> int:
        passed = sum(ok for _, ok, _ in self.rows)
        print(f"\n{passed}/{len(self.rows)} checks passed")
        return 0 if passed == len(self.rows) else 1


def maxdiff(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.max(np.abs(np.asarray(a) - np.asarray(b))))


def _sorted_table(path: Path, columns: list[str], sort: list[str]) -> pd.DataFrame:
    return pd.read_csv(path)[columns].sort_values(sort).reset_index(drop=True)


def nu_pt_invariant_high_precision(V: np.ndarray, dps: int = 80) -> float:
    """Evaluate the two-mode PPT invariant at high precision."""
    mp.mp.dps = dps
    arr = np.asarray(V, dtype=float)

    def mat(block: np.ndarray) -> mp.matrix:
        return mp.matrix([[mp.mpf(repr(float(value))) for value in row] for row in block])

    A = mat(arr[:2, :2])
    B = mat(arr[2:, 2:])
    C = mat(arr[:2, 2:])
    full = mat(arr)
    delta_tilde = mp.det(A) + mp.det(B) - 2 * mp.det(C)
    det_v = mp.det(full)
    radicand = delta_tilde * delta_tilde - 4 * det_v
    if radicand < 0 and abs(radicand) < mp.mpf("1e-60"):
        radicand = mp.mpf("0")
    root = mp.sqrt(radicand)
    # Rationalized form avoids subtractive cancellation.
    nu_sq = 2 * det_v / (delta_tilde + root)
    return float(mp.sqrt(nu_sq))


def eigenvalue_nu_pt(V: np.ndarray) -> float:
    PT = np.diag([1.0, 1.0, 1.0, -1.0])
    return float(np.min(symplectic_eigenvalues(PT @ np.asarray(V, float) @ PT)))


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--generated", type=Path, default=Path("generated"))
    parser.add_argument("--reference", type=Path, default=Path("data"))
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--smoke", action="store_true",
                      help="verify the branch, period-cell covariance, and critical mode pair only")
    mode.add_argument("--skip-isotropic", action="store_true",
                      help="run the stripe/homogeneous verification without Spot/Labyrinth")
    args = parser.parse_args()
    c = Checks()

    branch = solve_bond_centered_branch(0.4)
    ref_profile = pd.read_csv(args.reference / "stripe_branch_profile_lambda0p4.csv")[["q_star", "p_star"]].to_numpy()
    d_profile = maxdiff(branch.profile, ref_profile)
    c.add("commensurate stripe branch", d_profile < 2e-11 and branch.residual_inf < 1e-12,
          f"profile max diff={d_profile:.3e}, residual={branch.residual_inf:.3e}")

    A = branch_jacobian(branch.profile, 0.4)
    D = branch_diffusion(branch.profile)
    V = stationary_covariance(A, D)
    for name, value, ref_file, tol in (
        ("drift equals reference Jacobian", A, "stripe_branch_A_lambda0p4.csv", 2e-11),
        ("diffusion equals microscopic assembly", D, "stripe_branch_D_lambda0p4.csv", 2e-11),
    ):
        diff = maxdiff(value, read_matrix(args.reference / ref_file))
        c.add(name, diff < tol, f"max diff={diff:.3e}")

    # The soft phase mode makes the last digits of a Schur/Lyapunov solve
    # BLAS/LAPACK-backend dependent.  Compare two floating-point solutions
    # through the residual/separation budgets that certify their distance from
    # the same exact Lyapunov solution, rather than by a non-portable
    # elementwise threshold.
    V_ref = read_matrix(args.reference / "stripe_branch_V_lambda0p4.csv")
    residual = lyapunov_residual(A, D, V)
    c.add("Lyapunov equation", residual < 1e-9, f"relative residual={residual:.3e}")
    separation = lyapunov_operator_separation(A)
    c.add("Lyapunov operator separation", separation > 1e-7, f"sep_2={separation:.12g}")
    error_bound = covariance_error_bound_fro(A, D, V)
    reference_error_bound = covariance_error_bound_fro(A, D, V_ref)
    covariance_budget = error_bound + reference_error_bound
    covariance_diff_fro = float(np.linalg.norm(V - V_ref, ord="fro"))
    covariance_slack = 64.0 * np.finfo(float).eps * max(1.0, float(np.linalg.norm(V_ref, ord="fro")))
    c.add(
        "stationary covariance residual-certified agreement",
        covariance_diff_fro <= covariance_budget + covariance_slack,
        (f"Frobenius diff={covariance_diff_fro:.3e}, certified budget={covariance_budget:.3e} "
         f"(generated={error_bound:.3e}, reference={reference_error_bound:.3e})"),
    )
    min_phys = float(np.min(symplectic_eigenvalues(V)))
    c.add("full covariance physicality", min_phys > 0.5,
          f"minimum symplectic eigenvalue={min_phys:.12g}")
    physical_uncertainty_margin = uncertainty_min_eigenvalue(V)
    c.add(
        "direct uncertainty physicality test",
        physical_uncertainty_margin - error_bound > 0.0,
        (f"lambda_min(V+iJ/2)={physical_uncertainty_margin:.12g}, "
         f"certified lower={physical_uncertainty_margin-error_bound:.12g}"),
    )
    cp = gaussian_cp_margin(A, D)
    c.add("Gaussian complete positivity", cp > -2e-12, f"CP margin={cp:.3e}")

    pair = mode_pair_metrics(V, KSTAR)
    ref_pair = read_matrix(args.reference / "stripe_branch_kstar_pair_covariance_lambda0p4.csv")
    selector = fourier_symplectic(V.shape[0] // 2, KSTAR)
    selector_identity_error = float(np.max(np.abs(selector @ selector.T - np.eye(4))))
    selector_norm = float(np.linalg.norm(selector, ord=2))
    c.add(
        "Fourier selector contraction",
        selector_identity_error < 2e-13 and abs(selector_norm - 1.0) < 2e-13,
        f"max|S S^T-I|={selector_identity_error:.3e}, ||S||_2={selector_norm:.16g}",
    )
    pair_diff_fro = float(np.linalg.norm(pair["covariance"] - ref_pair, ord="fro"))
    pair_budget = selector_norm * selector_norm * covariance_budget
    pair_slack = 64.0 * np.finfo(float).eps * max(1.0, float(np.linalg.norm(ref_pair, ord="fro")))
    c.add(
        "Fourier pair covariance residual-certified agreement",
        pair_diff_fro <= pair_budget + pair_slack,
        f"Frobenius diff={pair_diff_fro:.3e}, contracted certified budget={pair_budget:.3e}",
    )
    c.add("Fourier transform symplecticity", pair["symplectic_transform_error"] < 2e-13,
          f"error={pair['symplectic_transform_error']:.3e}")
    c.add("critical-pair NPT", abs(pair["nu_pt"] - 0.2537366774) < 2e-8,
          f"nu_PT={pair['nu_pt']:.12g}, E_LN={pair['logarithmic_negativity']:.12g}")
    physicality_margin = min_phys - 0.5
    npt_margin = 0.5 - float(pair["nu_pt"])
    c.add("residual-certified covariance margins",
          error_bound < min(physicality_margin, npt_margin),
          f"error_bound={error_bound:.6g}, physicality_margin={physicality_margin:.6g}, NPT_margin={npt_margin:.6g}")
    pt_min = pt_uncertainty_min_eigenvalue(pair["covariance"])
    c.add("direct PT-uncertainty NPT test",
          pt_min + error_bound < 0.0,
          f"lambda_min(H_PT)={pt_min:.12g}, certified upper={pt_min+error_bound:.12g}")
    stripe_hp = nu_pt_invariant_high_precision(pair["covariance"])
    c.add("critical-pair high-precision PPT invariant",
          abs(stripe_hp - pair["nu_pt"]) < 2e-11,
          f"invariant nu_PT={stripe_hp:.12g}, method diff={abs(stripe_hp-pair['nu_pt']):.3e}")

    if args.smoke:
        return c.close()

    supercell = pd.read_csv(args.generated / "stripe_branch_linearized_supercell_mode_entanglement.csv")
    reference_supercell = pd.read_csv(args.reference / "stripe_branch_linearized_supercell_mode_entanglement.csv")
    common = ["L", "mode_index", "k", "nu_physical", "nu_pt", "logarithmic_negativity"]
    gen_sorted = supercell[common].sort_values(["L", "mode_index"]).reset_index(drop=True)
    ref_sorted = reference_supercell[common].sort_values(["L", "mode_index"]).reset_index(drop=True)
    table_diff = maxdiff(
        gen_sorted[["k", "nu_physical", "nu_pt", "logarithmic_negativity"]],
        ref_sorted[["k", "nu_physical", "nu_pt", "logarithmic_negativity"]],
    )
    c.add("regenerated supercell spectra", table_diff < 2e-8,
          f"maximum tabulated difference={table_diff:.3e}")
    for L in (48, 96):
        table = supercell[supercell.L == L]
        minimum = table.loc[table.nu_pt.idxmin()]
        c.add(f"L={L} longitudinal minimum", int(minimum.mode_index) == L // 12,
              f"mode={int(minimum.mode_index)}, k/pi={minimum.k_over_pi:.12g}")

    for Dq, Dp, label in ((DQ_STRIPE, DP_STRIPE, "differential"), (D_SCALAR, D_SCALAR, "scalar")):
        for mode in (1, 8, 16, 32, 47):
            k = 2 * np.pi * mode / 96
            pars = homogeneous_parameters(-0.2, k, Dq=Dq, Dp=Dp)
            A4, D4, V4 = homogeneous_pair_covariance(-0.2, k, Dq=Dq, Dp=Dp)
            direct_sw = stationary_covariance(A4, D4)
            T = standing_to_traveling_symplectic()
            direct = T @ direct_sw @ T.T
            md = maxdiff(V4, direct)
            nu_pt = eigenvalue_nu_pt(direct)
            c.add(f"homogeneous closed form ({label}, mode {mode})",
                  md < 5e-13 and abs(nu_pt - pars["nu_pt"]) < 5e-13,
                  f"cov diff={md:.3e}, nu_PT diff={abs(nu_pt-pars['nu_pt']):.3e}")

    generated_hom = pd.read_csv(args.generated / "stripe_homogeneous_same_lambda_controls.csv")
    diff_row = generated_hom[generated_hom.state == "differential"].iloc[0]
    scalar_row = generated_hom[generated_hom.state == "scalar"].iloc[0]
    c.add("same-lambda differential control",
          abs(diff_row.minimum_nu_pt - 0.2516526148) < 2e-9 and abs(diff_row.minimum_k_over_pi - 1/6) < 1e-12,
          f"min nu_PT={diff_row.minimum_nu_pt:.12g} at k/pi={diff_row.minimum_k_over_pi:.12g}")
    c.add("same-lambda scalar control",
          abs(scalar_row.minimum_nu_pt - 0.26002) < 2e-4 and abs(scalar_row.minimum_k_over_pi - 1/48) < 1e-12,
          f"min nu_PT={scalar_row.minimum_nu_pt:.12g}, k* nu_PT={scalar_row.kstar_nu_pt:.12g}")

    locking = pd.read_csv(args.generated / "stripe_phase_locking_compensated.csv")
    small = locking[(locking["lambda"] >= 0.05) & (locking["lambda"] <= 0.2)]
    exponent, _ = np.polyfit(np.log(small["lambda"]), np.log(small.locking_rate), 1)
    c.add("commensurate locking exponent", abs(exponent - 4.8866) < 0.03,
          f"exponent={exponent:.8f}")
    overlap = float(locking.loc[np.isclose(locking["lambda"], 0.4), "phase_tangent_overlap"].iloc[0])
    c.add("phase-tangent overlap", overlap > 0.997, f"overlap={overlap:.12g}")

    continuation = pd.read_csv(args.generated / "stripe_branch_linearized_across_threshold.csv")
    continuation = continuation[continuation.state == "bond_centered_branch"]
    amp_dev = float(continuation.critical_amplitude_relative_deviation.max())
    c.add("branch-amplitude continuation check", amp_dev < 0.01,
          f"maximum relative deviation from leading law={amp_dev:.7g}")
    stability_dev = float(continuation.reflection_fixed_relative_deviation.max())
    c.add("reflection-fixed branch stability continuation",
          bool((continuation.reflection_fixed_spectral_abscissa < 0.0).all()) and stability_dev < 0.06,
          f"maximum relative deviation from -lambda/2={stability_dev:.7g}")
    modes = sorted(set(int(x) for x in continuation.period_cell_npt_min_mode))
    ks = sorted(set(float(x) for x in continuation.period_cell_npt_min_k_over_pi))
    c.add("period-cell NPT minimizer along continued branch", modes == [1] and max(abs(x-1/6) for x in ks) < 1e-12,
          f"modes={modes}, k/pi={ks}")

    if not args.skip_isotropic:
        iso_cols = [
            "case", "mode_x", "mode_y", "dt_pde", "dt_cov", "k",
            "nu_physical_min", "nu_pt", "logarithmic_negativity",
            "covariance_min_eigenvalue", "selection_symplectic_error",
            "mode_class", "angle_degrees",
            "absolute_nu_pt_error_to_richardson",
            "absolute_logarithmic_negativity_error_to_richardson",
        ]
        iso_sort = ["case", "mode_x", "mode_y", "dt_cov"]
        iso_gen = _sorted_table(
            args.generated / "isotropic_finite_time_mode_covariance.csv", iso_cols, iso_sort
        )
        iso_ref = _sorted_table(
            args.reference / "isotropic_finite_time_mode_covariance.csv", iso_cols, iso_sort
        )
        iso_numeric = [column for column in iso_cols if column not in {"case", "mode_class"}]
        iso_diff = maxdiff(iso_gen[iso_numeric], iso_ref[iso_numeric])
        same_classes = iso_gen[["case", "mode_class"]].equals(iso_ref[["case", "mode_class"]])
        # Compact reference comparisons use portability tolerances that retain
        # at least six to seven significant digits.  The derived scientific
        # statements (physicality, NPT, shell/control gaps, Richardson ratios,
        # and coherent-step ordering) are verified independently below.
        c.add(
            "isotropic finite-time mode table",
            same_classes and iso_diff < 1e-7,
            (f"classes_equal={same_classes}, maximum tabulated difference={iso_diff:.3e}, "
             "portable tolerance=1e-7"),
        )

        zgen = np.load(args.generated / "isotropic_finite_time_covariances.npz", allow_pickle=False)
        zref = np.load(args.reference / "isotropic_finite_time_covariances.npz", allow_pickle=False)
        same_keys = set(zgen.files) == set(zref.files)
        cov_diff = max(maxdiff(zgen[key], zref[key]) for key in zgen.files) if same_keys else float("inf")
        c.add(
            "isotropic selected covariances",
            same_keys and cov_diff < 2e-7,
            (f"keys_equal={same_keys}, maximum covariance difference={cov_diff:.3e}, "
             "portable tolerance=2e-7"),
        )

        conv_cols = [
            "case", "mode_x", "mode_y", "mode_class", "dt_pde", "dt_cov",
            "nu_physical_min", "nu_pt", "logarithmic_negativity",
            "observed_refinement_ratio_nu_pt", "richardson_nu_pt",
            "absolute_nu_pt_error_to_richardson",
            "observed_refinement_ratio_logarithmic_negativity",
            "richardson_logarithmic_negativity",
            "absolute_logarithmic_negativity_error_to_richardson",
        ]
        conv_sort = ["case", "mode_x", "mode_y", "dt_cov"]
        conv_gen = _sorted_table(
            args.generated / "isotropic_finite_time_convergence.csv", conv_cols, conv_sort
        )
        conv_ref = _sorted_table(
            args.reference / "isotropic_finite_time_convergence.csv", conv_cols, conv_sort
        )
        conv_numeric = [column for column in conv_cols if column not in {"case", "mode_class"}]
        conv_diff = maxdiff(conv_gen[conv_numeric], conv_ref[conv_numeric])
        same_conv_classes = conv_gen[["case", "mode_class"]].equals(
            conv_ref[["case", "mode_class"]]
        )
        c.add(
            "isotropic covariance-time convergence table",
            same_conv_classes and conv_diff < 1e-6,
            (f"classes_equal={same_conv_classes}, maximum tabulated difference={conv_diff:.3e}, "
             "portable tolerance=1e-6 (includes derived refinement ratios)"),
        )

        pde_cols = [
            "case", "mode_x", "mode_y", "mode_class", "dt_pde", "dt_cov",
            "nu_physical_min", "nu_pt", "logarithmic_negativity",
            "selection_symplectic_error",
        ]
        pde_sort = ["case", "mode_x", "mode_y", "dt_pde"]
        pde_gen = _sorted_table(
            args.generated / "isotropic_pde_timestep_control.csv", pde_cols, pde_sort
        )
        pde_ref = _sorted_table(
            args.reference / "isotropic_pde_timestep_control.csv", pde_cols, pde_sort
        )
        pde_numeric = [column for column in pde_cols if column not in {"case", "mode_class"}]
        pde_diff = maxdiff(pde_gen[pde_numeric], pde_ref[pde_numeric])
        same_pde_classes = pde_gen[["case", "mode_class"]].equals(
            pde_ref[["case", "mode_class"]]
        )
        c.add(
            "isotropic coherent-trajectory time-step table",
            same_pde_classes and pde_diff < 1e-7,
            (f"classes_equal={same_pde_classes}, maximum tabulated difference={pde_diff:.3e}, "
             "portable tolerance=1e-7"),
        )

        iso_table = pd.read_csv(args.generated / "isotropic_finite_time_mode_covariance.csv")
        iso_conv = pd.read_csv(args.generated / "isotropic_finite_time_convergence.csv")
        iso_pde = pd.read_csv(args.generated / "isotropic_pde_timestep_control.csv")
        iso_summary = pd.read_csv(args.generated / "isotropic_finite_time_summary.csv")
        for case in ("spot", "labyrinth"):
            tab = iso_table[iso_table.case == case]
            shell = tab[tab.mode_class == "dominant_shell"]
            low = tab[tab.mode_class == "low_k_control"].iloc[0]
            high = tab[tab.mode_class == "high_k_control"].iloc[0]
            shell_min = float(shell.nu_pt.min())
            shell_max = float(shell.nu_pt.max())
            c.add(
                f"{case} selected-shell NPT ordering",
                shell_max < min(float(low.nu_pt), float(high.nu_pt)) and shell_min < 0.5,
                f"shell=[{shell_min:.9f},{shell_max:.9f}], controls=({float(low.nu_pt):.9f},{float(high.nu_pt):.9f})",
            )

            summary = iso_summary[iso_summary.case == case].iloc[0]
            shell_lower = float(summary.shell_logarithmic_negativity_error_aware_lower)
            low_upper = float(summary.low_k_control_error_aware_upper)
            high_upper = float(summary.high_k_control_error_aware_upper)
            c.add(
                f"{case} Richardson-estimated shell/control ordering",
                bool(summary.error_aware_shell_above_controls)
                and shell_lower > max(low_upper, high_upper),
                f"shell lower endpoint={shell_lower:.9f}, control upper endpoints=({low_upper:.9f},{high_upper:.9f})",
            )

            max_sel_err = float(tab.selection_symplectic_error.max())
            min_phys_case = float(tab.nu_physical_min.min())
            c.add(
                f"{case} physicality and canonical mode selection",
                min_phys_case > 0.5 and max_sel_err < 2e-13,
                f"min ordinary nu={min_phys_case:.12g}, selector error={max_sel_err:.3e}",
            )

            conv = iso_conv[iso_conv.case == case]
            ratios = conv.groupby(["mode_x", "mode_y"])[
                "observed_refinement_ratio_logarithmic_negativity"
            ].first()
            finite_ratios = ratios[np.isfinite(ratios)]
            min_ratio = float(finite_ratios.min())
            max_ratio = float(finite_ratios.max())
            c.add(
                f"{case} second-order logarithmic-negativity convergence for shell and controls",
                min_ratio > 3.4 and max_ratio < 4.6,
                f"refinement-ratio range=[{min_ratio:.6f},{max_ratio:.6f}] over {len(finite_ratios)} modes",
            )

            pde = iso_pde[iso_pde.case == case]
            ordering_by_dt: list[bool] = []
            gaps: list[float] = []
            for dt_pde, group in pde.groupby("dt_pde"):
                shell_values = group[group.mode_class == "dominant_shell"].logarithmic_negativity.to_numpy(float)
                control_values = group[group.mode_class != "dominant_shell"].logarithmic_negativity.to_numpy(float)
                gap = float(np.min(shell_values) - np.max(control_values))
                gaps.append(gap); ordering_by_dt.append(gap > 0.0)
            max_pde_change = 0.0
            for (mx, my), group in pde.groupby(["mode_x", "mode_y"]):
                values = group.sort_values("dt_pde")["logarithmic_negativity"].to_numpy(float)
                max_pde_change = max(max_pde_change, abs(float(values[-1] - values[0])))
            mode_count = int(pde[["mode_x", "mode_y"]].drop_duplicates().shape[0])
            expected_rows = 2 * mode_count
            c.add(
                f"{case} seven-mode coherent-trajectory time-step robustness",
                mode_count == 7 and len(pde) == expected_rows and all(ordering_by_dt) and max_pde_change < 0.02,
                f"modes={mode_count}/7, rows={len(pde)}/{expected_rows}, minimum shell/control gap={min(gaps):.6g}, max E_LN change={max_pde_change:.6g}",
            )

            endpoint = np.load(args.generated / f"{case}_endpoint.npz", allow_pickle=False)
            reference = np.load(args.reference / f"{case}_endpoint.npz", allow_pickle=False)
            end_diff = max(
                maxdiff(endpoint["q_final"], reference["q_final"]),
                maxdiff(endpoint["p_final"], reference["p_final"]),
            )
            c.add(
                f"{case} baseline trajectory endpoint",
                end_diff < 2e-8,
                f"maximum reference-endpoint difference={end_diff:.3e}",
            )

        hp_errors: list[float] = []
        conditions: list[float] = []
        for key in zgen.files:
            cov = np.asarray(zgen[key], float)
            eig_value = eigenvalue_nu_pt(cov)
            hp_value = nu_pt_invariant_high_precision(cov)
            hp_errors.append(abs(eig_value - hp_value))
            conditions.append(float(np.linalg.cond(cov)))
        c.add(
            "isotropic high-precision PPT invariant",
            max(hp_errors) < 2e-9,
            f"maximum method difference={max(hp_errors):.3e}, maximum cond(V)={max(conditions):.3e}",
        )

    return c.close()


if __name__ == "__main__":
    raise SystemExit(main())
