#!/usr/bin/env python3
"""Evaluate the exact homogeneous differential/scalar spectra."""
from __future__ import annotations
import argparse
from pathlib import Path
import pandas as pd
from fluctuation_analysis import (
    DP_STRIPE, DQ_STRIPE, D_SCALAR, homogeneous_spectrum_table
)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--lambda", dest="lam", type=float, default=-0.2)
    parser.add_argument("--L", type=int, default=96)
    parser.add_argument("--out", type=Path, default=Path("generated"))
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    differential = homogeneous_spectrum_table(
        args.lam, args.L, state="homogeneous_differential",
        Dq=DQ_STRIPE, Dp=DP_STRIPE,
    )
    scalar = homogeneous_spectrum_table(
        args.lam, args.L, state="homogeneous_scalar_transport_same_lambda",
        Dq=D_SCALAR, Dp=D_SCALAR,
    )
    table = pd.concat([differential, scalar], ignore_index=True)
    table.to_csv(args.out / "stripe_homogeneous_fluctuation_spectra.csv", index=False)
    controls = []
    for name, frame in (("differential", differential), ("scalar", scalar)):
        m = frame.loc[frame.nu_pt.idxmin()]
        kstar = frame.iloc[(frame.k_over_pi - 1/6).abs().argmin()]
        controls.append({
            "state": name,
            "lambda": args.lam,
            "minimum_mode_index": int(m.mode_index),
            "minimum_k_over_pi": m.k_over_pi,
            "minimum_nu_pt": m.nu_pt,
            "kstar_nu_pt": kstar.nu_pt,
            "kstar_logarithmic_negativity": kstar.logarithmic_negativity,
        })
    pd.DataFrame(controls).to_csv(args.out / "stripe_homogeneous_same_lambda_controls.csv", index=False)
    print(pd.DataFrame(controls).to_string(index=False))
    return 0

if __name__ == "__main__":
    raise SystemExit(main())
