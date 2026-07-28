#!/usr/bin/env python3
"""Plot the finite-time opposite-momentum spectra and convergence data."""
from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib as mpl
mpl.use("Agg")
from plot_style import apply_figure_style, save_figure_pdf, style_line_axis
apply_figure_style()
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.ticker import FixedFormatter, FixedLocator, NullFormatter
import numpy as np
import pandas as pd

SHELL_COLOR = "#1f77b4"
LOW_COLOR = "#ff7f0e"
HIGH_COLOR = "#2ca02c"
GUIDE_COLOR = "#555555"


def _scientific_tex(value: float, digits: int = 1) -> str:
    """Format a positive value in scientific notation for a TeX label."""
    if value <= 0:
        return "0"
    exponent = int(np.floor(np.log10(value)))
    mantissa = value / (10.0 ** exponent)
    return rf"{mantissa:.{digits}f}\times10^{{{exponent}}}"


def _plot_case(main_ax, detail_ax, tab: pd.DataFrame, *, case: str, title: str, panel: str, endpoint: int) -> float:
    shell = tab[tab.mode_class == "dominant_shell"].sort_values("angle_degrees")
    low = tab[tab.mode_class == "low_k_control"].iloc[0]
    high = tab[tab.mode_class == "high_k_control"].iloc[0]

    shell_x = shell.angle_degrees.to_numpy(float)
    shell_y = shell.logarithmic_negativity.to_numpy(float)
    shell_err = shell.absolute_logarithmic_negativity_error_to_richardson.to_numpy(float)

    main_ax.errorbar(
        shell_x,
        shell_y,
        yerr=shell_err,
        fmt="o",
        mfc="white",
        mec=SHELL_COLOR,
        mew=1.25,
        capsize=2.4,
        elinewidth=1.0,
        color=SHELL_COLOR,
        zorder=4,
    )
    for row, color, linestyle in ((low, LOW_COLOR, "--"), (high, HIGH_COLOR, "-.")):
        value = float(row.logarithmic_negativity)
        error = float(row.absolute_logarithmic_negativity_error_to_richardson)
        main_ax.axhspan(value - error, value + error, color=color, alpha=0.12, lw=0, zorder=1)
        main_ax.axhline(value, ls=linestyle, lw=1.9, color=color, zorder=2)

    main_ax.set_xlim(-4, 94)
    main_ax.set_xticks([0, 22.5, 45, 67.5, 90])
    main_ax.set_xticklabels(["0", "22.5", "45", "67.5", "90"])
    main_ax.set_xlabel("")
    main_ax.set_ylim(0.75, 0.99)
    if case == "spot":
        main_ax.set_ylabel(r"$E_{\rm LN}(k,-k;T)$")
    else:
        main_ax.tick_params(labelleft=False)
    main_ax.set_title(f"({panel}) {title}, $T={endpoint}$", loc="left", pad=4)
    style_line_axis(main_ax)

    # The detail axis occupies a separate full-width panel, preserving the
    # main spectrum while displaying the small shell/high-k separation.
    high_value = float(high.logarithmic_negativity)
    high_error = float(high.absolute_logarithmic_negativity_error_to_richardson)
    detail_ax.errorbar(
        shell_x,
        shell_y,
        yerr=shell_err,
        fmt="o",
        mfc="white",
        mec=SHELL_COLOR,
        mew=1.15,
        capsize=2.0,
        elinewidth=0.9,
        color=SHELL_COLOR,
        zorder=4,
    )
    detail_ax.axhspan(
        high_value - high_error,
        high_value + high_error,
        color=HIGH_COLOR,
        alpha=0.14,
        lw=0,
        zorder=1,
    )
    detail_ax.axhline(high_value, ls="-.", lw=1.8, color=HIGH_COLOR, zorder=2)
    shell_lower = float(np.min(shell_y - shell_err))
    high_upper = high_value + high_error
    gap = shell_lower - high_upper
    values = np.concatenate([shell_y - shell_err, shell_y + shell_err, [high_value - high_error, high_upper]])
    lo = float(np.min(values))
    hi = float(np.max(values))
    span = max(hi - lo, 0.004)
    detail_ax.set_ylim(lo - 0.18 * span, hi + 0.24 * span)
    detail_ax.set_xlim(-4, 94)
    detail_ax.set_xticks([0, 45, 90])
    detail_ax.set_xlabel("selected-shell angle (degrees)")
    detail_ax.set_ylabel(r"$E_{\rm LN}$")
    detail_ax.set_title(
        f"{title}: shell/high-$k$ detail",
        loc="left",
        pad=3,
        fontsize=14.2,
    )
    detail_ax.text(
        0.96,
        0.08,
        rf"$\Delta_{{\rm est}}={_scientific_tex(gap)}$",
        transform=detail_ax.transAxes,
        ha="right",
        va="bottom",
        fontsize=10.8,
        bbox=dict(boxstyle="round,pad=0.12", fc="white", ec="none", alpha=0.88),
    )
    style_line_axis(detail_ax)
    return gap


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=Path, default=Path("generated"))
    parser.add_argument("--reference", type=Path, default=Path("data"))
    parser.add_argument(
        "--out",
        type=Path,
        default=Path("generated/fig5_finite_time_spectra.pdf"),
    )
    args = parser.parse_args()

    def read(name: str) -> pd.DataFrame:
        path = args.data / name
        if not path.exists():
            path = args.reference / name
        return pd.read_csv(path)

    survey = read("isotropic_finite_time_mode_covariance.csv")
    convergence = read("isotropic_finite_time_convergence.csv")
    finest_dt = float(survey.dt_cov.min())

    # Main spectra, detail panels, and legends occupy separate grid rows.  This
    # keeps the small error-aware separation visible without covering either
    # endpoint spectrum.
    fig = plt.figure(figsize=(9.0, 9.30), constrained_layout=False)
    gs = fig.add_gridspec(
        5,
        12,
        height_ratios=(3.10, 3.10, 0.55, 2.62, 0.55),
        left=0.095,
        right=0.965,
        bottom=0.055,
        top=0.945,
        hspace=0.47,
        wspace=0.92,
    )
    ax_spot = fig.add_subplot(gs[0, 0:5])
    ax_lab = fig.add_subplot(gs[0, 7:12], sharey=ax_spot)
    ax_spot_detail = fig.add_subplot(gs[1, 0:5])
    ax_lab_detail = fig.add_subplot(gs[1, 7:12])
    leg_top = fig.add_subplot(gs[2, 1:11])
    ax_conv = fig.add_subplot(gs[3, 2:10])
    leg_conv = fig.add_subplot(gs[4, 2:10])
    leg_top.axis("off")
    leg_conv.axis("off")

    gaps: dict[str, float] = {}
    for main_ax, detail_ax, case, title, panel, endpoint in (
        (ax_spot, ax_spot_detail, "spot", "Spot", "a", 50),
        (ax_lab, ax_lab_detail, "labyrinth", "Labyrinth", "b", 80),
    ):
        tab = survey[(survey.case == case) & np.isclose(survey.dt_cov, finest_dt)].copy()
        gaps[case] = _plot_case(
            main_ax,
            detail_ax,
            tab,
            case=case,
            title=title,
            panel=panel,
            endpoint=endpoint,
        )
    top_handles = [
        Line2D(
            [0],
            [0],
            marker="o",
            markerfacecolor="white",
            markeredgecolor=SHELL_COLOR,
            markeredgewidth=1.2,
            color=SHELL_COLOR,
            linestyle="none",
            markersize=5.6,
            label="selected radial shell",
        ),
        Line2D([0], [0], color=LOW_COLOR, lw=1.9, ls="--", label="low-$k$ control"),
        Line2D([0], [0], color=HIGH_COLOR, lw=1.9, ls="-.", label="high-$k$ control"),
    ]
    leg_top.legend(
        handles=top_handles,
        loc="center",
        ncol=3,
        frameon=False,
        handlelength=2.5,
        columnspacing=1.45,
    )

    # Plot the worst-case error over the two morphologies.
    worst = (
        convergence.groupby("dt_cov", as_index=False)[
            "absolute_logarithmic_negativity_error_to_richardson"
        ]
        .max()
        .sort_values("dt_cov")
    )
    worst_line, = ax_conv.loglog(
        worst.dt_cov,
        worst.absolute_logarithmic_negativity_error_to_richardson,
        marker="o",
        markerfacecolor="white",
        markeredgecolor=SHELL_COLOR,
        markeredgewidth=1.2,
        color=SHELL_COLOR,
        label="worst case over both morphologies",
        zorder=4,
    )
    dts = np.array([0.0125, 0.1])
    ref = 0.72 * max(float(worst.absolute_logarithmic_negativity_error_to_richardson.max()), 1e-8)
    guide, = ax_conv.loglog(
        dts,
        ref * (dts / dts[-1]) ** 2,
        ls="--",
        lw=1.5,
        color=GUIDE_COLOR,
        label=r"$O(\Delta t_{\rm cov}^2)$",
        zorder=2,
    )
    ax_conv.set_xlabel(r"covariance step $\Delta t_{\rm cov}$")
    ax_conv.set_ylabel(r"worst-case $|E_{\rm LN}-E_{\rm LN}^{\rm Rich}|$")
    ax_conv.set_title("(c) worst-case covariance-time convergence", loc="left", pad=4)
    ax_conv.xaxis.set_major_locator(FixedLocator([0.0125, 0.025, 0.05, 0.1]))
    ax_conv.xaxis.set_major_formatter(FixedFormatter(["0.0125", "0.025", "0.05", "0.1"]))
    ax_conv.xaxis.set_minor_formatter(NullFormatter())
    ax_conv.set_xlim(0.011, 0.115)
    ax_conv.grid(True, alpha=0.18, linewidth=0.60, which="both")
    ax_conv.tick_params(axis="both", which="major", pad=3)
    leg_conv.legend(
        handles=(worst_line, guide),
        loc="center",
        ncol=2,
        frameon=False,
        handlelength=2.5,
        columnspacing=1.45,
    )

    save_figure_pdf(fig, args.out)
    plt.close(fig)
    print(f"{args.out} (estimated gaps: Spot={gaps['spot']:.6g}, Labyrinth={gaps['labyrinth']:.6g})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
