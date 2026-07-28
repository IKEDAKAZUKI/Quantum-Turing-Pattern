#!/usr/bin/env python3
"""Aggregate isotropic finite-time covariance and time-step controls."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from io_utils import save_npz

SURVEY = {
    "spot": [(12, 0), (11, 5), (8, 8), (5, 11), (0, 12), (4, 0), (20, 0)],
    "labyrinth": [(13, 0), (12, 5), (9, 9), (5, 12), (0, 13), (4, 0), (20, 0)],
}
SHELL = {
    "spot": {(12, 0), (11, 5), (8, 8), (5, 11), (0, 12)},
    "labyrinth": {(13, 0), (12, 5), (9, 9), (5, 12), (0, 13)},
}
SELECTED = {"spot": (12, 0), "labyrinth": (13, 0)}
LOW_CONTROL = (4, 0)
HIGH_CONTROL = (20, 0)
SURVEY_DT = 0.0125
CONVERGENCE_DTS = (0.1, 0.05, 0.025, 0.0125)
PDE_DTS = (0.05, 0.025)


def tag(value: float) -> str:
    return str(value).replace(".", "p")


def path_for(raw: Path, case: str, mode: tuple[int, int], dt_pde: float, dt_cov: float) -> Path:
    return raw / f"{case}_{mode[0]}_{mode[1]}_pde{tag(dt_pde)}_dtcov{tag(dt_cov)}.npz"


def load(path: Path) -> tuple[dict, np.ndarray, np.ndarray, np.ndarray, dict]:
    with np.load(path, allow_pickle=False) as data:
        return (
            json.loads(str(data["row_json"])),
            np.asarray(data["covariance"], float),
            np.asarray(data["q_final"], float),
            np.asarray(data["p_final"], float),
            json.loads(str(data["trajectory_params_json"])),
        )


def mode_class(case: str, mode: tuple[int, int]) -> str:
    if mode in SHELL[case]:
        return "dominant_shell"
    if mode == LOW_CONTROL:
        return "low_k_control"
    if mode == HIGH_CONTROL:
        return "high_k_control"
    raise ValueError((case, mode))


def richardson(rows: list[dict], field: str) -> tuple[float, float]:
    ordered = sorted(rows, key=lambda row: float(row["dt_cov"]), reverse=True)
    # Use the three finest steps for the asymptotic second-order estimate.
    fine_three = ordered[-3:]
    vals = np.asarray([float(row[field]) for row in fine_three], float)
    denominator = vals[1] - vals[2]
    ratio = float((vals[0] - vals[1]) / denominator) if abs(denominator) > 1e-16 else float("nan")
    extrapolated = float(vals[2] + (vals[2] - vals[1]) / 3.0)
    return ratio, extrapolated


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--raw", type=Path, default=Path("generated/raw_modes"))
    parser.add_argument("--out", type=Path, default=Path("generated"))
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)

    survey_rows: list[dict] = []
    convergence_rows_all: list[dict] = []
    pde_rows: list[dict] = []
    summaries: list[dict] = []
    covariances: dict[str, np.ndarray] = {}

    for case in ("spot", "labyrinth"):
        selected = SELECTED[case]
        kdom = 2.0 * np.pi * np.hypot(*selected) / 128.0
        qref = pref = params = None
        fine_by_mode: dict[tuple[int, int], dict] = {}
        conv_by_mode: dict[tuple[int, int], list[dict]] = {}

        for mode in SURVEY[case]:
            rows_for_mode: list[dict] = []
            for dt_cov in CONVERGENCE_DTS:
                row, cov, q, p, par = load(path_for(args.raw, case, mode, PDE_DTS[0], dt_cov))
                if qref is None:
                    qref, pref, params = q, p, par
                elif np.max(np.abs(q - qref)) > 1e-12 or np.max(np.abs(p - pref)) > 1e-12:
                    raise RuntimeError(f"{case} endpoint mismatch")
                row.update(
                    {
                        "mode_class": mode_class(case, mode),
                        "k_over_kdom": row["k"] / kdom,
                        "angle_degrees": float(np.degrees(np.arctan2(mode[1], mode[0]))),
                    }
                )
                rows_for_mode.append(row)
                covariances[
                    f"{case}_{mode[0]}_{mode[1]}_pde{tag(PDE_DTS[0])}_dtcov{tag(dt_cov)}"
                ] = cov

            ratio_nu, rich_nu = richardson(rows_for_mode, "nu_pt")
            ratio_en, rich_en = richardson(rows_for_mode, "logarithmic_negativity")
            for row in rows_for_mode:
                row.update(
                    {
                        "observed_refinement_ratio_nu_pt": ratio_nu,
                        "richardson_nu_pt": rich_nu,
                        "absolute_nu_pt_error_to_richardson": abs(float(row["nu_pt"]) - rich_nu),
                        "observed_refinement_ratio_logarithmic_negativity": ratio_en,
                        "richardson_logarithmic_negativity": rich_en,
                        "absolute_logarithmic_negativity_error_to_richardson": abs(
                            float(row["logarithmic_negativity"]) - rich_en
                        ),
                    }
                )
                convergence_rows_all.append(row.copy())
            fine = min(rows_for_mode, key=lambda row: float(row["dt_cov"]))
            fine_by_mode[mode] = fine.copy()
            conv_by_mode[mode] = rows_for_mode
            survey_rows.append(fine.copy())

        assert qref is not None and pref is not None and params is not None
        save_npz(
            args.out / f"{case}_endpoint.npz",
            q_final=qref,
            p_final=pref,
            params_json=json.dumps(params, sort_keys=True),
        )

        # PDE step control: baseline values are the dt_pde=0.05 fine-covariance
        # entries; refined values come from a separately regenerated trajectory.
        for dt_pde in PDE_DTS:
            for mode in SURVEY[case]:
                if dt_pde == PDE_DTS[0]:
                    row = fine_by_mode[mode].copy()
                else:
                    row, cov, _, _, _ = load(path_for(args.raw, case, mode, dt_pde, SURVEY_DT))
                    row.update(
                        {
                            "mode_class": mode_class(case, mode),
                            "k_over_kdom": row["k"] / kdom,
                            "angle_degrees": float(np.degrees(np.arctan2(mode[1], mode[0]))),
                        }
                    )
                    covariances[
                        f"{case}_{mode[0]}_{mode[1]}_pde{tag(dt_pde)}_dtcov{tag(SURVEY_DT)}"
                    ] = cov
                pde_rows.append(row)

        shell_fine = [fine_by_mode[mode] for mode in SHELL[case]]
        low = fine_by_mode[LOW_CONTROL]
        high = fine_by_mode[HIGH_CONTROL]

        def en_lower(row: dict) -> float:
            return float(row["logarithmic_negativity"]) - float(
                row["absolute_logarithmic_negativity_error_to_richardson"]
            )

        def en_upper(row: dict) -> float:
            return float(row["logarithmic_negativity"]) + float(
                row["absolute_logarithmic_negativity_error_to_richardson"]
            )

        shell_lower = min(en_lower(row) for row in shell_fine)
        low_upper = en_upper(low)
        high_upper = en_upper(high)

        pde_case = [row for row in pde_rows if row["case"] == case]
        pde_max_change = 0.0
        for mode in SURVEY[case]:
            values = sorted(
                [row for row in pde_case if int(row["mode_x"]) == mode[0] and int(row["mode_y"]) == mode[1]],
                key=lambda row: float(row["dt_pde"]),
            )
            pde_max_change = max(
                pde_max_change,
                abs(float(values[-1]["logarithmic_negativity"]) - float(values[0]["logarithmic_negativity"])),
            )
        pde_gaps: list[float] = []
        pde_ordering_flags: list[bool] = []
        for dt in PDE_DTS:
            group = [row for row in pde_case if np.isclose(float(row["dt_pde"]), dt)]
            shell_values = [float(row["logarithmic_negativity"]) for row in group if row["mode_class"] == "dominant_shell"]
            control_values = [float(row["logarithmic_negativity"]) for row in group if row["mode_class"] != "dominant_shell"]
            gap = min(shell_values) - max(control_values)
            pde_gaps.append(gap)
            pde_ordering_flags.append(gap > 0.0)
        pde_ordering = all(pde_ordering_flags)
        pde_min_gap = min(pde_gaps)

        selected_fine = fine_by_mode[selected]
        summaries.append(
            {
                "case": case,
                "L": 128,
                "T": 50.0 if case == "spot" else 80.0,
                "dominant_mode_x": selected[0],
                "dominant_mode_y": selected[1],
                "k_dom": kdom,
                "survey_dt_cov": SURVEY_DT,
                "baseline_dt_pde": PDE_DTS[0],
                "refined_dt_pde": PDE_DTS[1],
                "shell_nu_pt_min": min(float(row["nu_pt"]) for row in shell_fine),
                "shell_nu_pt_max": max(float(row["nu_pt"]) for row in shell_fine),
                "shell_logarithmic_negativity_min": min(
                    float(row["logarithmic_negativity"]) for row in shell_fine
                ),
                "shell_logarithmic_negativity_max": max(
                    float(row["logarithmic_negativity"]) for row in shell_fine
                ),
                "shell_logarithmic_negativity_error_aware_lower": shell_lower,
                "low_k_control_logarithmic_negativity": float(low["logarithmic_negativity"]),
                "high_k_control_logarithmic_negativity": float(high["logarithmic_negativity"]),
                "low_k_control_error_aware_upper": low_upper,
                "high_k_control_error_aware_upper": high_upper,
                "error_aware_shell_above_controls": bool(shell_lower > max(low_upper, high_upper)),
                "selected_finest_nu_pt": float(selected_fine["nu_pt"]),
                "selected_finest_logarithmic_negativity": float(
                    selected_fine["logarithmic_negativity"]
                ),
                "selected_richardson_nu_pt": float(selected_fine["richardson_nu_pt"]),
                "selected_observed_refinement_ratio": float(
                    selected_fine["observed_refinement_ratio_nu_pt"]
                ),
                "maximum_selection_symplectic_error": max(
                    float(row["selection_symplectic_error"]) for row in fine_by_mode.values()
                ),
                "maximum_pde_step_logarithmic_negativity_change": pde_max_change,
                "minimum_pde_shell_control_gap": pde_min_gap,
                "pde_step_ordering_preserved": bool(pde_ordering),
                "pde_step_sampled_mode_count": len(SURVEY[case]),
            }
        )

    survey_df = pd.DataFrame(survey_rows).sort_values(["case", "mode_class", "angle_degrees"])
    conv_df = pd.DataFrame(convergence_rows_all).sort_values(
        ["case", "mode_class", "angle_degrees", "dt_cov"],
        ascending=[True, True, True, False],
    )
    pde_df = pd.DataFrame(pde_rows).sort_values(["case", "mode_class", "dt_pde"], ascending=[True, True, False])
    summary_df = pd.DataFrame(summaries)

    survey_df.to_csv(args.out / "isotropic_finite_time_mode_covariance.csv", index=False)
    conv_df.to_csv(args.out / "isotropic_finite_time_convergence.csv", index=False)
    pde_df.to_csv(args.out / "isotropic_pde_timestep_control.csv", index=False)
    summary_df.to_csv(args.out / "isotropic_finite_time_summary.csv", index=False)
    save_npz(args.out / "isotropic_finite_time_covariances.npz", **covariances)
    print(summary_df.to_string(index=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
