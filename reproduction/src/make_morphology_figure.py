#!/usr/bin/env python3
"""Plot representative morphologies, Fourier power, and selected spectra."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib as mpl
mpl.use("Agg")
from matplotlib.patches import Circle, Patch
from matplotlib.lines import Line2D
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from plot_style import apply_figure_style, save_figure_pdf, style_line_axis

apply_figure_style()


def pattern_covector() -> float:
    return float((np.sqrt(3.0) - 1.0) / np.sqrt(2.0 * np.sqrt(3.0)))


def load_endpoint(data: Path, reference: Path, case: str) -> tuple[np.ndarray, np.ndarray, dict[str, object]]:
    candidates = [
        data / f"{case}_endpoint.npz",
        reference / f"{case}_endpoint.npz",
    ]
    for path in candidates:
        if path.is_file():
            with np.load(path, allow_pickle=False) as archive:
                params = json.loads(str(archive["params_json"].item()))
                return np.asarray(archive["q_final"], float), np.asarray(archive["p_final"], float), params
    raise FileNotFoundError(f"endpoint data not found for {case}")


def load_stripe(data: Path, reference: Path, *, L: int = 192) -> tuple[np.ndarray, np.ndarray]:
    path = data / "stripe_branch_profile_lambda0p4.csv"
    if not path.is_file():
        path = reference / path.name
    table = pd.read_csv(path).sort_values("site")
    qcell = table["q_star"].to_numpy(float)
    pcell = table["p_star"].to_numpy(float)
    if L % qcell.size:
        raise ValueError("stripe lattice size must be a multiple of the period-cell length")
    qline = np.tile(qcell, L // qcell.size)
    pline = np.tile(pcell, L // pcell.size)
    return np.tile(qline, (L, 1)), np.tile(pline, (L, 1))


def field_and_power(q: np.ndarray, p: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    field = q - pattern_covector() * p
    centered = field - np.mean(field)
    transform = np.fft.fftshift(np.fft.fft2(centered))
    power = np.abs(transform) ** 2
    power /= max(float(np.max(power)), np.finfo(float).tiny)
    k = np.fft.fftshift(2.0 * np.pi * np.fft.fftfreq(field.shape[0]))
    return field, power, k


def radial_maximum(power: np.ndarray, k: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    kx, ky = np.meshgrid(k, k, indexing="xy")
    radius = np.sqrt(kx * kx + ky * ky)
    dr = 2.0 * np.pi / power.shape[0]
    index = np.floor(radius / dr + 0.5).astype(int)
    values = np.zeros(int(index.max()) + 1, dtype=float)
    for j in range(values.size):
        mask = index == j
        if np.any(mask):
            values[j] = float(np.max(power[mask]))
    radii = np.arange(values.size, dtype=float) * dr
    mask = radii <= 1.5
    return radii[mask], values[mask]


def stripe_spectrum(power: np.ndarray, k: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    center = power.shape[0] // 2
    nonnegative = k >= 0.0
    return k[nonnegative], power[center, nonnegative]


def growth_rate(k: np.ndarray, *, omega: float, dq: float, dp: float) -> np.ndarray:
    symbol = 2.0 * (1.0 - np.cos(k))
    a11 = 1.0 - dq * symbol
    a22 = -3.0 - dp * symbol
    trace = a11 + a22
    determinant = a11 * a22 + omega * omega
    discriminant = np.maximum(trace * trace - 4.0 * determinant, 0.0)
    return 0.5 * (trace + np.sqrt(discriminant))


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=Path, default=Path("generated"))
    parser.add_argument("--reference", type=Path, default=Path("data"))
    parser.add_argument("--out", type=Path, default=Path("generated/fig1_quantum_turing_morphologies.pdf"))
    args = parser.parse_args()

    stripe_q, stripe_p = load_stripe(args.data, args.reference)
    spot_q, spot_p, spot_params = load_endpoint(args.data, args.reference, "spot")
    lab_q, lab_p, lab_params = load_endpoint(args.data, args.reference, "labyrinth")

    cases = [
        ("Stripe", stripe_q, stripe_p, {"Omega": np.sqrt(2.0 * np.sqrt(3.0) - 0.4), "Dq": 1.0, "Dp": 3.0 + 2.0 * np.sqrt(3.0)}),
        ("Spot", spot_q, spot_p, spot_params),
        ("Labyrinth", lab_q, lab_p, lab_params),
    ]
    processed = [field_and_power(q, p) for _, q, p, _ in cases]
    vmax = max(float(np.max(np.abs(field))) for field, _, _ in processed)

    fig = plt.figure(figsize=(9.4, 9.25), constrained_layout=False)
    grid = fig.add_gridspec(
        3, 3, left=0.085, right=0.91, bottom=0.07, top=0.96,
        hspace=0.32, wspace=0.24, height_ratios=(1.0, 1.0, 0.72),
    )
    field_axes = [fig.add_subplot(grid[0, j]) for j in range(3)]
    power_axes = [fig.add_subplot(grid[1, j]) for j in range(3)]
    spectrum_axes = [fig.add_subplot(grid[2, j]) for j in range(3)]

    field_image = None
    power_image = None
    letters = "abcdefghi"
    for j, ((title, _q, _p, params), (field, power, k)) in enumerate(zip(cases, processed)):
        ax = field_axes[j]
        field_image = ax.imshow(
            field, origin="lower", extent=(0.0, 1.0, 0.0, 1.0),
            cmap="RdYlBu_r", vmin=-vmax, vmax=vmax, interpolation="bilinear",
        )
        ax.set_title(title, pad=5)
        ax.set_xlabel(r"$x/L$")
        if j == 0:
            ax.set_ylabel(r"$y/L$")
        else:
            ax.set_yticklabels([])
        ax.set_xticks([0.0, 0.5, 1.0])
        ax.set_yticks([0.0, 0.5, 1.0])
        ax.text(-0.12, 1.04, f"({letters[j]})", transform=ax.transAxes, fontsize=14.5)

        axp = power_axes[j]
        log_power = np.log10(np.maximum(power, 1e-6))
        power_image = axp.imshow(
            log_power, origin="lower", extent=(-np.pi, np.pi, -np.pi, np.pi),
            cmap="magma", vmin=-6.0, vmax=0.0, interpolation="nearest",
        )
        axp.set_xlabel(r"$k_x$")
        if j == 0:
            axp.set_ylabel(r"$k_y$")
        else:
            axp.set_yticklabels([])
        axp.set_xticks([-np.pi, 0.0, np.pi], [r"$-\pi$", "0", r"$\pi$"])
        axp.set_yticks([-np.pi, 0.0, np.pi], [r"$-\pi$", "0", r"$\pi$"])
        axp.text(-0.12, 1.04, f"({letters[3+j]})", transform=axp.transAxes, fontsize=14.5)

        if j == 0:
            x, y = stripe_spectrum(power, k)
            xlabel = r"$|k_x|$"
        else:
            x, y = radial_maximum(power, k)
            xlabel = r"$|k|$"
        valid = (x >= 0.0) & (x <= (1.25 if j == 0 else 1.5))
        x, y = x[valid], y[valid]
        dominant = float(x[int(np.argmax(y))])
        growth = growth_rate(x, omega=float(params["Omega"]), dq=float(params["Dq"]), dp=float(params["Dp"]))
        positive = np.maximum(growth, 0.0)
        if np.max(positive) > 0:
            positive /= np.max(positive)
        unstable = growth > 0.0

        axs = spectrum_axes[j]
        axs.plot(x, y, label=r"$P/P_{\max}$")
        axs.plot(x, positive, linestyle="--", label=r"$\alpha_+(k)$")
        if np.any(unstable):
            axs.fill_between(x, 0.0, 1.05, where=unstable, alpha=0.12, label="linear unstable band")
        axs.axvline(dominant, linestyle="--", linewidth=1.4)
        axs.text(0.96, 0.10, rf"$k_{{\rm dom}}={dominant:.2f}$", transform=axs.transAxes, ha="right")
        axs.set_xlim(0.0, 1.25 if j == 0 else 1.5)
        axs.set_ylim(0.0, 1.08)
        axs.set_xlabel(xlabel)
        if j == 0:
            axs.set_ylabel("normalized spectrum")
        else:
            axs.set_yticklabels([])
        axs.text(-0.12, 1.04, f"({letters[6+j]})", transform=axs.transAxes, fontsize=14.5)
        style_line_axis(axs)
        if j == 0:
            handles = [
                Patch(alpha=0.12, label="linear unstable band"),
                Line2D([0], [0], lw=2.2, label=r"$P/P_{\max}$"),
                Line2D([0], [0], lw=2.2, ls="--", label=r"$\alpha_+(k)$"),
            ]
            axs.legend(handles=handles, frameon=False, fontsize=8.8, loc="upper right")

    cax1 = fig.add_axes([0.93, 0.705, 0.014, 0.245])
    cb1 = fig.colorbar(field_image, cax=cax1)
    cb1.set_label(r"$R^{\rm pat}$")
    cax2 = fig.add_axes([0.93, 0.395, 0.014, 0.245])
    cb2 = fig.colorbar(power_image, cax=cax2, ticks=[-6, -3, 0])
    cb2.set_label(r"$\log_{10}(P/P_{\max})$")

    save_figure_pdf(fig, args.out)
    plt.close(fig)
    print(args.out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
