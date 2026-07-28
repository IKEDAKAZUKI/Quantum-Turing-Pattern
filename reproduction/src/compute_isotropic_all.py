#!/usr/bin/env python3
"""Run the finite-time Spot/Labyrinth covariance workflow.

The default execution is restartable.  Each coherent trajectory is cached,
and each mode/covariance-step result is stored independently.  Existing files
are reused unless ``--force`` is supplied.
"""
from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor
import json
from pathlib import Path

import numpy as np

from io_utils import save_npz
from isotropic_fluctuations import (
    Trajectory,
    finite_time_selected_covariances,
    load_trajectory,
    mode_radius,
    pair_metrics,
    regenerate_trajectory,
    save_trajectory,
)

SURVEY = {
    "spot": [(12, 0), (11, 5), (8, 8), (5, 11), (0, 12), (4, 0), (20, 0)],
    "labyrinth": [(13, 0), (12, 5), (9, 9), (5, 12), (0, 13), (4, 0), (20, 0)],
}
SELECTED = {"spot": (12, 0), "labyrinth": (13, 0)}
LOW_CONTROL = (4, 0)
HIGH_CONTROL = (20, 0)
COVARIANCE_DTS = (0.0125, 0.025, 0.05, 0.1)
PDE_DTS = (0.05, 0.025)
PDE_CONTROL_MODES = {case: list(SURVEY[case]) for case in SURVEY}


def tag(value: float) -> str:
    return str(value).replace(".", "p")


def _mode_path(raw: Path, case: str, mode: tuple[int, int], *, dt_pde: float, dt_cov: float) -> Path:
    return raw / (
        f"{case}_{mode[0]}_{mode[1]}_pde{tag(dt_pde)}_dtcov{tag(dt_cov)}.npz"
    )


def _trajectory_path(raw: Path, case: str, dt_pde: float) -> Path:
    return raw / f"{case}_trajectory_pde{tag(dt_pde)}.npz"


def _save_mode(
    path: Path,
    *,
    case: str,
    mode: tuple[int, int],
    covariance: np.ndarray,
    meta: dict[str, float | int],
    q_final: np.ndarray,
    p_final: np.ndarray,
    params: dict[str, object],
) -> None:
    metrics = pair_metrics(covariance)
    row = {
        "case": case,
        "mode_x": mode[0],
        "mode_y": mode[1],
        "k": mode_radius(q_final.shape[0], mode),
        "dt_pde": float(params["dt"]),
        "dt_cov": float(meta["dt_cov"]),
        **metrics,
        **meta,
    }
    save_npz(
        path,
        covariance=np.asarray(covariance, np.float64),
        row_json=json.dumps(row, sort_keys=True),
        q_final=np.asarray(q_final, np.float64),
        p_final=np.asarray(p_final, np.float64),
        trajectory_params_json=json.dumps(params, sort_keys=True),
    )


def _get_trajectory(case: str, raw: Path, dt_pde: float, force: bool) -> Trajectory:
    cache = _trajectory_path(raw, case, dt_pde)
    if cache.exists() and not force:
        print(f"= reuse {cache.name}", flush=True)
        return load_trajectory(cache)
    print(f"+ regenerate {case} coherent trajectory at dt_pde={dt_pde:g}", flush=True)
    traj = regenerate_trajectory(case, dt=dt_pde)
    save_trajectory(cache, traj)
    return traj


def _compute_batch(
    *,
    case: str,
    raw: Path,
    traj: Trajectory,
    modes: list[tuple[int, int]],
    dt_cov: float,
    force: bool,
) -> None:
    dt_pde = float(traj.params["dt"])
    paths = [_mode_path(raw, case, mode, dt_pde=dt_pde, dt_cov=dt_cov) for mode in modes]
    missing = [idx for idx, path in enumerate(paths) if force or not path.exists()]
    if not missing:
        print(
            f"= reuse {case}: {len(modes)} modes at dt_pde={dt_pde:g}, dt_cov={dt_cov:g}",
            flush=True,
        )
        return

    # A single batched adjoint propagation is substantially cheaper than one
    # solve per mode.  Recompute the whole batch when any member is missing so
    # all entries share identical numerical inputs.
    print(
        f"+ {case}: {len(modes)} modes at dt_pde={dt_pde:g}, dt_cov={dt_cov:g}",
        flush=True,
    )
    covs, meta = finite_time_selected_covariances(traj, modes, dt_cov=dt_cov)
    q_final = np.asarray(traj.q[-1], np.float64)
    p_final = np.asarray(traj.p[-1], np.float64)
    for mode, cov, path in zip(modes, covs, paths, strict=True):
        _save_mode(
            path,
            case=case,
            mode=mode,
            covariance=cov,
            meta=meta,
            q_final=q_final,
            p_final=p_final,
            params=traj.params,
        )


def _compute_case(case: str, raw: Path, force: bool) -> None:
    # Baseline trajectory: all seven shell/control modes at four covariance
    # steps.  These data provide a Richardson estimate for every plotted point
    # and both radial controls.
    baseline = _get_trajectory(case, raw, PDE_DTS[0], force)
    for dt_cov in COVARIANCE_DTS:
        _compute_batch(
            case=case,
            raw=raw,
            traj=baseline,
            modes=SURVEY[case],
            dt_cov=dt_cov,
            force=force,
        )

    # Mean-field trajectory time-step control: repeat all five shell directions and both radial controls on the halved PDE step.
    refined = _get_trajectory(case, raw, PDE_DTS[1], force)
    _compute_batch(
        case=case,
        raw=raw,
        traj=refined,
        modes=PDE_CONTROL_MODES[case],
        dt_cov=COVARIANCE_DTS[0],
        force=force,
    )


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", type=Path, default=Path("generated"))
    parser.add_argument(
        "--jobs",
        type=int,
        default=1,
        help="number of cases to process concurrently; 1 is memory-conservative",
    )
    cache_group = parser.add_mutually_exclusive_group()
    cache_group.add_argument(
        "--resume",
        dest="force",
        action="store_false",
        help="reuse cached trajectories and completed mode files (the default)",
    )
    cache_group.add_argument(
        "--force",
        dest="force",
        action="store_true",
        help="discard cached trajectories/mode files and recompute everything",
    )
    parser.set_defaults(force=False)
    args = parser.parse_args()
    raw = args.out / "raw_modes"
    raw.mkdir(parents=True, exist_ok=True)

    cases = ("spot", "labyrinth")
    jobs = max(1, min(int(args.jobs), len(cases)))
    if jobs == 1:
        for case in cases:
            _compute_case(case, raw, args.force)
    else:
        with ThreadPoolExecutor(max_workers=jobs) as pool:
            list(pool.map(lambda case: _compute_case(case, raw, args.force), cases))

    import sys
    from aggregate_isotropic_modes import main as aggregate_main

    old_argv = sys.argv
    try:
        sys.argv = ["aggregate_isotropic_modes.py", "--raw", str(raw), "--out", str(args.out)]
        return aggregate_main()
    finally:
        sys.argv = old_argv


if __name__ == "__main__":
    raise SystemExit(main())
