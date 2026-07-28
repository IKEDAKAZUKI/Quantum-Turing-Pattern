#!/usr/bin/env python3
"""Plot the finite-wave-number Turing point and its parameter crossing."""
from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib as mpl
mpl.use("Agg")
from plot_style import apply_figure_style, save_figure_pdf, style_line_axis
apply_figure_style()
import matplotlib.pyplot as plt
import numpy as np



def dispersion(k: np.ndarray) -> np.ndarray:
    return 2.0 * (1.0 - np.cos(k))


def drift(k: np.ndarray, lam: float) -> np.ndarray:
    a, b = 1.0, 3.0
    dq = 1.0
    dp = 3.0 + 2.0 * np.sqrt(3.0)
    omega = np.sqrt(2.0 * np.sqrt(3.0) - lam)
    w = dispersion(k)
    out = np.empty((k.size, 2, 2), dtype=float)
    out[:, 0, 0] = a - dq * w
    out[:, 0, 1] = omega
    out[:, 1, 0] = -omega
    out[:, 1, 1] = -b - dp * w
    return out


def growth(k: np.ndarray, lam: float) -> np.ndarray:
    return np.array([np.max(np.linalg.eigvals(a).real) for a in drift(k, lam)])


def determinant_at_threshold(k: np.ndarray) -> np.ndarray:
    return np.linalg.det(drift(k, 0.0))


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", type=Path, default=Path("generated/fig2_turing_point.pdf"))
    args = parser.parse_args()

    k = np.linspace(0.0, np.pi, 1001)
    kstar = np.pi / 6.0
    wstar = 2.0 - np.sqrt(3.0)

    fig = plt.figure(figsize=(8.5, 7.15), constrained_layout=False)
    gs = fig.add_gridspec(
        2,
        2,
        left=0.10,
        right=0.97,
        bottom=0.09,
        top=0.94,
        hspace=0.42,
        wspace=0.34,
    )
    axa = fig.add_subplot(gs[0, 0])
    axb = fig.add_subplot(gs[0, 1])
    axc = fig.add_subplot(gs[1, 0])
    axd = fig.add_subplot(gs[1, 1])

    for lam in (0.0, 0.2, 0.4):
        axa.plot(k, growth(k, lam), label=rf"$\lambda={lam:g}$")
    g04 = growth(k, 0.4)
    axa.fill_between(k, 0.0, g04, where=g04 > 0.0, alpha=0.16, label=r"unstable band, $\lambda=0.4$")
    axa.axhline(0.0, linewidth=1.0)
    axa.axvline(kstar, linestyle=":", linewidth=1.3)
    axa.text(kstar + 0.045, 0.94 * axa.get_ylim()[1], r"$k_* = \pi/6$", rotation=90, va="top")
    axa.set_xlim(0.0, np.pi)
    axa.set_xlabel(r"$k_1$")
    axa.set_ylabel(r"$s_\lambda(k_1,0)$")
    axa.set_title("(a) finite-wave-number crossing", loc="left")
    axa.legend(frameon=False, fontsize=9.0)
    style_line_axis(axa)

    det = determinant_at_threshold(k)
    det_factorized = (3.0 + 2.0 * np.sqrt(3.0)) * (dispersion(k) - wstar) ** 2
    axb.plot(k, det, label="computed")
    axb.plot(k, det_factorized, linestyle="--", label="factorized")
    axb.axvline(kstar, linestyle=":", linewidth=1.3)
    axb.set_xlim(0.0, np.pi)
    axb.set_xlabel(r"$k_1$")
    axb.set_ylabel(r"$\det \widehat A_0(k_1,0)$")
    axb.set_title("(b) determinant", loc="left")
    axb.legend(frameon=False)
    style_line_axis(axb)

    mask = (k >= 0.30) & (k <= 0.72)
    axc.plot(k[mask], det[mask], label="computed")
    axc.plot(k[mask], det_factorized[mask], linestyle="--", label="factorized")
    axc.axvline(kstar, linestyle=":", linewidth=1.3)
    axc.set_xlim(0.30, 0.72)
    axc.set_xlabel(r"$k_1$")
    axc.set_ylabel(r"$\det \widehat A_0(k_1,0)$")
    axc.set_title("(c) quadratic zero near $k_*$", loc="left")
    axc.legend(frameon=False)
    style_line_axis(axc)

    lams = np.linspace(-0.1, 0.4, 251)
    kstar_vec = np.full_like(lams, kstar)
    svals = np.array([growth(np.array([kstar]), float(lam))[0] for lam in lams])
    axd.plot(lams, svals, label=r"$s_\lambda(k_*)$")
    axd.plot(lams, lams / 4.0, linestyle="--", label=r"slope $1/4$")
    axd.axhline(0.0, linewidth=1.0)
    axd.axvline(0.0, linestyle=":", linewidth=1.3)
    axd.set_xlim(-0.1, 0.4)
    axd.set_xlabel(r"$\lambda$")
    axd.set_ylabel(r"$s_\lambda(k_*)$")
    axd.set_title("(d) crossing slope", loc="left")
    axd.legend(frameon=False)
    style_line_axis(axd)

    save_figure_pdf(fig, args.out)
    plt.close(fig)
    print(args.out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
