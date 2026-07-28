#!/usr/bin/env python3
"""Compute one selected finite-time covariance mode."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from io_utils import save_npz
from isotropic_fluctuations import (
    finite_time_selected_covariances,
    mode_radius,
    pair_metrics,
    regenerate_trajectory,
)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--case", choices=("spot", "labyrinth"), required=True)
    parser.add_argument("--mode-x", type=int, required=True)
    parser.add_argument("--mode-y", type=int, required=True)
    parser.add_argument("--dt", type=float, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    args.out.parent.mkdir(parents=True, exist_ok=True)

    trajectory = regenerate_trajectory(args.case)
    mode = (args.mode_x, args.mode_y)
    covariances, meta = finite_time_selected_covariances(
        trajectory, [mode], dt_cov=args.dt
    )
    covariance = covariances[0]
    row = {
        "case": args.case,
        "mode_x": args.mode_x,
        "mode_y": args.mode_y,
        "k": mode_radius(trajectory.q.shape[1], mode),
        "dt_cov": meta["dt_cov"],
        **pair_metrics(covariance),
        **meta,
    }
    save_npz(
        args.out,
        covariance=np.asarray(covariance, np.float64),
        row_json=json.dumps(row, sort_keys=True),
        q_final=np.asarray(trajectory.q[-1], np.float64),
        p_final=np.asarray(trajectory.p[-1], np.float64),
        trajectory_params_json=json.dumps(trajectory.params, sort_keys=True),
    )
    print(json.dumps(row, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
