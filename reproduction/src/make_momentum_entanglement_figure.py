#!/usr/bin/env python3
"""Plot the momentum-resolved Gaussian entanglement results."""
from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib as mpl
mpl.use("Agg")
from plot_style import apply_figure_style, save_figure_pdf, style_line_axis
apply_figure_style()
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

BRANCH_COLOR = "#1f77b4"
DIFFERENTIAL_COLOR = "#ff7f0e"
SCALAR_COLOR = "#2ca02c"
THRESHOLD_COLOR = "#111111"
FIT_COLOR = "#6b6b6b"


def _load(data: Path, reference: Path, name: str) -> pd.DataFrame:
    path = data / name
    if not path.exists():
        path = reference / name
    return pd.read_csv(path)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=Path, default=Path("generated"))
    parser.add_argument("--reference", type=Path, default=Path("data"))
    parser.add_argument(
        "--out",
        type=Path,
        default=Path("generated/fig4_momentum_entanglement.pdf"),
    )
    args = parser.parse_args()

    branch = _load(args.data, args.reference, "stripe_branch_linearized_supercell_mode_entanglement.csv")
    hom = _load(args.data, args.reference, "stripe_homogeneous_fluctuation_spectra.csv")
    across = _load(args.data, args.reference, "stripe_branch_linearized_across_threshold.csv")
    locking = _load(args.data, args.reference, "stripe_phase_locking_compensated.csv")
    controls = _load(args.data, args.reference, "stripe_homogeneous_same_lambda_controls.csv")

    branch96 = branch[(branch.L == 96) & (branch.k_over_pi > 0) & (branch.k_over_pi < 1)].copy()
    diff = hom[hom.state == "homogeneous_differential"].copy()
    scalar = hom[hom.state.str.contains("scalar", case=False, na=False)].copy()

    # Native width and font scales match Figures 1--3. The detail panel and legends
    # occupy separate grid cells, so no overlay obscures the scientific data.
    fig = plt.figure(figsize=(9.0, 7.65), constrained_layout=False)
    gs = fig.add_gridspec(
        4,
        12,
        height_ratios=(3.75, 0.64, 3.12, 0.62),
        left=0.095,
        right=0.965,
        bottom=0.060,
        top=0.940,
        hspace=0.53,
        wspace=1.02,
    )
    axa = fig.add_subplot(gs[0, 0:8])
    axz = fig.add_subplot(gs[0, 9:12])
    lega = fig.add_subplot(gs[1, 0:12])
    axb = fig.add_subplot(gs[2, 0:5])
    axc = fig.add_subplot(gs[2, 7:12])
    legb = fig.add_subplot(gs[3, 0:5])
    legc = fig.add_subplot(gs[3, 7:12])
    for ax in (lega, legb, legc):
        ax.axis("off")

    kstar = 1.0 / 6.0

    line_branch, = axa.plot(
        branch96.k_over_pi,
        branch96.nu_pt,
        marker="o",
        color=BRANCH_COLOR,
        label=r"nonlinear stripe, $\lambda=0.4$",
    )
    line_diff, = axa.plot(
        diff.k_over_pi,
        diff.nu_pt,
        color=DIFFERENTIAL_COLOR,
        label=r"homogeneous differential, $\lambda=-0.2$",
    )
    line_scalar, = axa.plot(
        scalar.k_over_pi,
        scalar.nu_pt,
        color=SCALAR_COLOR,
        ls="-.",
        label=r"homogeneous scalar, $\lambda=-0.2$",
    )
    line_ppt = axa.axhline(
        0.5,
        ls="--",
        lw=1.4,
        color=THRESHOLD_COLOR,
        label="PPT threshold",
    )
    axa.axvline(kstar, ls=":", lw=1.45, color=THRESHOLD_COLOR)
    axa.text(
        kstar + 0.016,
        0.477,
        r"$k_*/\pi=1/6$",
        rotation=90,
        va="top",
        fontsize=11.4,
        bbox=dict(boxstyle="round,pad=0.10", fc="white", ec="none", alpha=0.84),
    )
    axa.set_xlim(0.0, 1.0)
    axa.set_ylim(0.245, 0.505)
    axa.set_xlabel(r"longitudinal wave number $k_1/\pi$")
    axa.set_ylabel(r"$\widetilde\nu_-(k,-k)$")
    axa.set_title("(a) opposite-momentum NPT spectrum", loc="left", pad=4)
    style_line_axis(axa)

    # The scalar curve lies outside the vertical range of this detail panel.
    axz.plot(branch96.k_over_pi, branch96.nu_pt, marker="o", color=BRANCH_COLOR)
    axz.plot(diff.k_over_pi, diff.nu_pt, color=DIFFERENTIAL_COLOR)
    axz.axvline(kstar, ls=":", lw=1.35, color=THRESHOLD_COLOR)
    axz.text(
        kstar + 0.0017,
        0.25825,
        r"$k_*$",
        rotation=90,
        ha="left",
        va="top",
        fontsize=11.2,
        bbox=dict(boxstyle="round,pad=0.08", fc="white", ec="none", alpha=0.84),
    )
    axz.set_xlim(0.135, 0.205)
    axz.set_ylim(0.249, 0.259)
    axz.set_xlabel(r"$k_1/\pi$")
    axz.set_ylabel(r"$\widetilde\nu_-$")
    axz.set_title(r"detail near $k_*$", loc="left", pad=3, fontsize=14.2)
    style_line_axis(axz)

    lega.legend(
        handles=(line_branch, line_diff, line_scalar, line_ppt),
        loc="center",
        ncol=2,
        frameon=False,
        handlelength=2.5,
        columnspacing=1.55,
    )

    homogeneous = across[across.state == "homogeneous"].sort_values("lambda")
    patterned = across[
        (across.state == "bond_centered_branch")
        & (across.relative_lyapunov_residual <= 1e-8)
    ].sort_values("lambda")
    hb, = axb.plot(
        homogeneous["lambda"],
        homogeneous.kstar_logarithmic_negativity,
        marker="o",
        color=DIFFERENTIAL_COLOR,
        label="stable homogeneous differential state",
    )
    pb, = axb.plot(
        patterned["lambda"],
        patterned.kstar_logarithmic_negativity,
        marker="s",
        color=BRANCH_COLOR,
        label="nonlinear stripe branch",
    )
    threshold, = axb.plot(
        [0.0],
        [1.0],
        marker="o",
        mfc="white",
        mec=THRESHOLD_COLOR,
        mew=1.2,
        ms=6.6,
        ls="none",
        color=THRESHOLD_COLOR,
        label="one-sided threshold limit",
    )
    scalar_row = controls[controls.state == "scalar"].iloc[0]
    sc, = axb.plot(
        [-0.2],
        [scalar_row.kstar_logarithmic_negativity],
        marker="D",
        ms=5.6,
        ls="none",
        color=SCALAR_COLOR,
        label=r"scalar control at $\lambda=-0.2$",
    )
    axb.axvline(0.0, lw=1.25, ls=":", color=THRESHOLD_COLOR)
    axb.set_xlim(-0.42, 0.42)
    axb.set_ylim(0.76, 1.015)
    axb.set_xlabel(r"bifurcation parameter $\lambda$")
    axb.set_ylabel(r"$E_{\rm LN}(k_*,-k_*)$")
    axb.set_title("(b) critical-pair entanglement", loc="left", pad=4)
    style_line_axis(axb)
    legb.legend(
        handles=(hb, pb, threshold, sc),
        loc="center",
        ncol=2,
        frameon=False,
        handlelength=2.3,
        columnspacing=1.25,
    )

    lc, = axc.plot(
        locking["lambda"],
        locking.locking_over_lambda5,
        marker="o",
        color=BRANCH_COLOR,
        label=r"$\Gamma_{\rm lock}/\lambda^5$",
    )
    fitc, = axc.plot(
        locking["lambda"],
        locking.small_window_compensated_fit,
        ls="--",
        lw=1.8,
        color=FIT_COLOR,
        label=r"compensated small-$\lambda$ fit",
    )
    small = locking[(locking["lambda"] >= 0.05) & (locking["lambda"] <= 0.2)]
    slope, _ = np.polyfit(np.log(small["lambda"]), np.log(small.locking_rate), 1)
    axc.text(
        0.96,
        0.90,
        rf"$\Gamma_{{\rm lock}}\propto\lambda^{{{slope:.2f}}}$",
        transform=axc.transAxes,
        ha="right",
        va="top",
        fontsize=11.5,
        bbox=dict(boxstyle="round,pad=0.16", fc="white", ec="none", alpha=0.88),
    )
    axc.set_xlim(0.04, 0.41)
    axc.set_xlabel(r"$\lambda$")
    axc.set_ylabel(r"$\Gamma_{\rm lock}/\lambda^5$")
    axc.set_title("(c) commensurate phase locking", loc="left", pad=4)
    style_line_axis(axc)
    legc.legend(
        handles=(lc, fitc),
        loc="center",
        ncol=2,
        frameon=False,
        handlelength=2.4,
        columnspacing=1.25,
    )

    save_figure_pdf(fig, args.out)
    plt.close(fig)
    print(args.out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
