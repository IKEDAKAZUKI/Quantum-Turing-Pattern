#!/usr/bin/env python3
"""Compute period-cell Fourier-pair and real-space two-site NPT metrics."""
from __future__ import annotations
import argparse
from pathlib import Path
import numpy as np
import pandas as pd
from fluctuation_analysis import (
    KSTAR, logarithmic_negativity, mode_pair_metrics, partial_transpose_covariance,
    read_matrix, symplectic_eigenvalues
)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=Path, default=Path("generated"))
    args = parser.parse_args()
    V = read_matrix(args.data / "stripe_branch_V_lambda0p4.csv")
    L = V.shape[0] // 2
    rows = []
    for mode in range(1, L // 2):
        k = 2 * np.pi * mode / L
        m = mode_pair_metrics(V, k)
        rows.append({
            "lambda": 0.4,
            "mode_index": mode,
            "k": k,
            "k_over_pi": k / np.pi,
            "nu_physical": m["nu_physical"],
            "nu_pt": m["nu_pt"],
            "logarithmic_negativity": m["logarithmic_negativity"],
            "symplectic_transform_error": m["symplectic_transform_error"],
        })
    pd.DataFrame(rows).to_csv(args.data / "stripe_branch_linearized_mode_entanglement.csv", index=False)

    nn_rows = []
    for x in range(L):
        y = (x + 1) % L
        idx = [2*x, 2*x+1, 2*y, 2*y+1]
        block = V[np.ix_(idx, idx)]
        nu_phys = float(np.min(symplectic_eigenvalues(block)))
        nu_pt = float(np.min(symplectic_eigenvalues(partial_transpose_covariance(block))))
        nn_rows.append({
            "bond_start": x, "bond_end": y, "nu_physical": nu_phys,
            "nu_pt": nu_pt, "logarithmic_negativity": logarithmic_negativity(nu_pt),
        })
    pd.DataFrame(nn_rows).to_csv(args.data / "stripe_branch_linearized_nearest_neighbor.csv", index=False)
    kstar = min(rows, key=lambda r: abs(r["k"] - KSTAR))
    print(f"k* nu_PT={kstar['nu_pt']:.12g}, E_LN={kstar['logarithmic_negativity']:.12g}")
    print(f"minimum nearest-neighbor nu_PT={min(r['nu_pt'] for r in nn_rows):.12g}")
    return 0

if __name__ == "__main__":
    raise SystemExit(main())
