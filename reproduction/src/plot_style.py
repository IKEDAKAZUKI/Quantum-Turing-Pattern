#!/usr/bin/env python3
"""Plot settings used by the numerical figures."""
from __future__ import annotations

from pathlib import Path
from tempfile import NamedTemporaryFile

import matplotlib as mpl
from pypdf import PdfReader, PdfWriter

SERIF_STACK = [
    "Times New Roman",
    "Times",
    "TeX Gyre Termes",
    "Nimbus Roman",
    "Liberation Serif",
]


def apply_figure_style() -> None:
    mpl.rcParams.update(
        {
            "figure.dpi": 180,
            "savefig.dpi": 320,
            "font.family": "serif",
            "font.serif": SERIF_STACK,
            "mathtext.fontset": "stix",
            "axes.unicode_minus": False,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "font.size": 15.3,
            "axes.titlesize": 16.2,
            "axes.labelsize": 14.6,
            "xtick.labelsize": 11.9,
            "ytick.labelsize": 11.9,
            "legend.fontsize": 10.4,
            "lines.linewidth": 2.2,
            "lines.markersize": 5.2,
            "axes.grid": False,
            "grid.alpha": 0.20,
            "grid.linewidth": 0.45,
            "axes.linewidth": 0.8,
            "xtick.major.width": 0.8,
            "ytick.major.width": 0.8,
            "xtick.minor.width": 0.6,
            "ytick.minor.width": 0.6,
            "image.interpolation": "nearest",
        }
    )


def style_line_axis(ax) -> None:
    ax.grid(True, alpha=0.18, linewidth=0.60)
    ax.tick_params(axis="both", which="major", pad=3)


def _remove_pdf_metadata(source: Path, destination: Path) -> None:
    reader = PdfReader(source)
    writer = PdfWriter()
    for page in reader.pages:
        writer.add_page(page)
    writer.metadata = None
    with destination.open("wb") as stream:
        writer.write(stream)


def save_figure_pdf(fig, path: Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with NamedTemporaryFile(
        dir=path.parent,
        prefix=f".{path.name}.",
        suffix=".pdf",
        delete=False,
    ) as handle:
        temporary = Path(handle.name)
    try:
        fig.savefig(
            temporary,
            bbox_inches="tight",
            pad_inches=0.025,
        )
        _remove_pdf_metadata(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)
