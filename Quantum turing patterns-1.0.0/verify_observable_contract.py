#!/usr/bin/env python3
"""Self-test for the Fourier observables used by this package."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from qtp_observables import (
    OBSERVABLE_CONTRACT,
    normalized_spatial_extent,
    radial_shell_observables,
)


def _close(a: float, b: float, tol: float = 5e-13) -> bool:
    return abs(a - b) <= tol * max(1.0, abs(a), abs(b))


def main() -> int:
    L = 96
    x = np.arange(L)[:, None]
    y = np.arange(L)[None, :]
    m = 8
    field = np.cos(2 * np.pi * m * x / L) + 0 * y
    base = radial_shell_observables(field)
    shifted = radial_shell_observables(field + 17.25)
    expected = 2 * np.pi * m / L
    checks = {
        "contract_complete": set(OBSERVABLE_CONTRACT)
        == {
            "field_preprocessing",
            "dft_normalization",
            "wavevector_grid",
            "radial_bin_width",
            "radial_bin_rule",
            "radial_bin_statistic",
            "dominant_mode_rule",
            "shell_half_width",
            "coordinate_convention",
        },
        "plane_wave_kdom": _close(base.k_dom, expected),
        "mean_offset_invariant": _close(base.k_dom, shifted.k_dom)
        and _close(base.shell_concentration, shifted.shell_concentration),
        "plane_wave_shell": base.shell_concentration > 1 - 1e-12,
        "normalized_extent": normalized_spatial_extent() == (0.0, 1.0, 0.0, 1.0),
    }
    payload = {"checks": checks, "example": base.to_dict(), "contract": OBSERVABLE_CONTRACT}
    Path("verification_runtime").mkdir(exist_ok=True)
    Path("verification_runtime/observable_contract_report.json").write_text(
        json.dumps(payload, indent=2) + "\n", encoding="utf-8"
    )
    for name, ok in checks.items():
        print(f"{'PASS' if ok else 'FAIL'}  {name}")
    return 0 if all(checks.values()) else 1


if __name__ == "__main__":
    raise SystemExit(main())
