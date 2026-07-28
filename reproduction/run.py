#!/usr/bin/env python3
"""Regenerate the smoke, quick, or full numerical workflow."""
from __future__ import annotations

import argparse
import os
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
SRC = ROOT / "src"
REFERENCE = ROOT / "data" / "reference"

CORE_STEPS: list[tuple[str, tuple[str, ...]]] = [
    ("build_branch_covariance.py", ("stripe_branch_covariance_summary.csv",)),
    ("compute_mode_entanglement.py", ("stripe_branch_linearized_mode_entanglement.csv",)),
    ("build_supercell_covariance.py", ("stripe_branch_linearized_supercell_mode_entanglement.csv",)),
    ("compute_homogeneous_spectrum.py", ("stripe_homogeneous_same_lambda_controls.csv",)),
    ("compute_across_threshold.py", ("stripe_branch_linearized_across_threshold.csv",)),
    ("compute_locking_scaling.py", ("stripe_phase_locking_compensated.csv",)),
]


def environment(threads: int) -> dict[str, str]:
    env = os.environ.copy()
    for key in (
        "OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS",
        "BLIS_NUM_THREADS", "VECLIB_MAXIMUM_THREADS", "NUMEXPR_NUM_THREADS",
    ):
        env[key] = str(max(1, threads))
    env["MPLBACKEND"] = "Agg"
    env["PYTHONDONTWRITEBYTECODE"] = "1"
    return env


def call(script: str, *args: str, env: dict[str, str]) -> None:
    command = [sys.executable, str(SRC / script), *args]
    print("+", " ".join(command), flush=True)
    subprocess.run(command, cwd=SRC, env=env, check=True)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=("smoke", "quick", "full"), default="smoke")
    parser.add_argument("--out", type=Path)
    parser.add_argument("--jobs", type=int, default=1)
    parser.add_argument("--blas-threads", type=int, default=1)
    cache = parser.add_mutually_exclusive_group()
    cache.add_argument("--force", action="store_true")
    cache.add_argument("--resume", action="store_true")
    args = parser.parse_args()

    out = (args.out.expanduser().resolve() if args.out else
           (ROOT.parent / f"quantum_turing_output_{args.mode}").resolve())
    if out == ROOT or ROOT in out.parents:
        raise SystemExit("Choose an output directory outside reproduction/.")
    out.mkdir(parents=True, exist_ok=True)
    env = environment(args.blas_threads)
    force = args.force or not args.resume

    steps = CORE_STEPS[:2] if args.mode == "smoke" else CORE_STEPS
    for script, outputs in steps:
        if not force and all((out / name).is_file() for name in outputs):
            print(f"= cached {script}", flush=True)
            continue
        flag = "--data" if script == "compute_mode_entanglement.py" else "--out"
        call(script, flag, str(out), env=env)

    if args.mode == "smoke":
        call("verify_results.py", "--generated", str(out), "--reference", str(REFERENCE), "--smoke", env=env)
        print(out)
        return 0

    figures = [
        ("make_morphology_figure.py", "fig1_quantum_turing_morphologies.pdf", True),
        ("make_turing_point_figure.py", "fig2_turing_point.pdf", False),
        ("make_momentum_entanglement_figure.py", "fig4_momentum_entanglement.pdf", True),
    ]
    for script, filename, uses_data in figures:
        target = out / filename
        if not force and target.is_file():
            print(f"= cached {script}", flush=True)
            continue
        arguments = ["--out", str(target)]
        if uses_data:
            arguments = ["--data", str(out), "--reference", str(REFERENCE), *arguments]
        call(script, *arguments, env=env)

    if args.mode == "quick":
        call("verify_results.py", "--generated", str(out), "--reference", str(REFERENCE), "--skip-isotropic", env=env)
        print(out)
        return 0

    isotropic = ["--out", str(out), "--jobs", str(max(1, args.jobs)), "--force" if force else "--resume"]
    call("compute_isotropic_all.py", *isotropic, env=env)
    call(
        "make_finite_time_spectra_figure.py",
        "--data", str(out), "--reference", str(REFERENCE),
        "--out", str(out / "fig5_finite_time_spectra.pdf"), env=env,
    )
    call("verify_results.py", "--generated", str(out), "--reference", str(REFERENCE), env=env)
    print(out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
