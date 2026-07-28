#!/usr/bin/env python3
"""Verify distributed reference data or regenerated numerical output."""
from __future__ import annotations

import argparse
import hashlib
import os
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
SRC = ROOT / "src"
REFERENCE = ROOT / "data" / "reference"


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


def verify_checksums() -> None:
    manifest = ROOT / "CHECKSUMS.sha256"
    for line in manifest.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        digest, relative = line.split("  ", 1)
        path = ROOT / relative
        if not path.is_file():
            raise RuntimeError(f"missing file: {relative}")
        if hashlib.sha256(path.read_bytes()).hexdigest() != digest:
            raise RuntimeError(f"checksum mismatch: {relative}")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=("reference", "smoke", "quick", "full"), default="reference")
    parser.add_argument("--generated", type=Path)
    parser.add_argument("--blas-threads", type=int, default=1)
    args = parser.parse_args()

    verify_checksums()
    if args.mode == "reference":
        data = REFERENCE
        flags: list[str] = []
    else:
        data = (args.generated.expanduser().resolve() if args.generated else
                (ROOT.parent / f"quantum_turing_output_{args.mode}").resolve())
        flags = ["--smoke"] if args.mode == "smoke" else (["--skip-isotropic"] if args.mode == "quick" else [])

    command = [
        sys.executable, str(SRC / "verify_results.py"),
        "--generated", str(data), "--reference", str(REFERENCE), *flags,
    ]
    print("+", " ".join(command), flush=True)
    subprocess.run(command, cwd=SRC, env=environment(args.blas_threads), check=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
