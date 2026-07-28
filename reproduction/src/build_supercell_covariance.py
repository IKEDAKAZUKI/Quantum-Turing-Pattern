#!/usr/bin/env python3
"""Rebuild longitudinal mode-entanglement spectra on tiled supercells."""
from __future__ import annotations
import argparse
from pathlib import Path
import pandas as pd
from fluctuation_analysis import solve_bond_centered_branch, supercell_mode_table


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--lambda", dest="lam", type=float, default=0.4)
    parser.add_argument("--sizes", type=int, nargs="+", default=[12, 48, 96])
    parser.add_argument("--out", type=Path, default=Path("generated"))
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    branch = solve_bond_centered_branch(args.lam)
    tables = []
    for L in args.sizes:
        table, _, _, _ = supercell_mode_table(branch.profile, args.lam, L)
        tables.append(table)
        minimum = table.loc[table.nu_pt.idxmin()]
        print(f"L={L}: minimum at mode={int(minimum.mode_index)}, k/pi={minimum.k_over_pi:.12g}, nu_PT={minimum.nu_pt:.12g}")
    pd.concat(tables, ignore_index=True).to_csv(
        args.out / "stripe_branch_linearized_supercell_mode_entanglement.csv", index=False
    )
    return 0

if __name__ == "__main__":
    raise SystemExit(main())
