"""Fourier and radial-shell observables used by the QTP packages."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any

import numpy as np

OBSERVABLE_CONTRACT: dict[str, str] = {
    "field_preprocessing": "subtract_spatial_mean",
    "dft_normalization": "unitary_1_over_sqrt_volume",
    "wavevector_grid": "k_j=2*pi*n_j/L",
    "radial_bin_width": "2*pi/L",
    "radial_bin_rule": "half_open_centered_annuli",
    "radial_bin_statistic": "arithmetic_mean_power",
    "dominant_mode_rule": "maximum_nonzero_radial_mean_tie_to_smaller_radius",
    "shell_half_width": "one_fourier_spacing=2*pi/L",
    "coordinate_convention": "real_space_axes_x_over_L_y_over_L",
}


@dataclass(frozen=True)
class SpectralObservables:
    k_dom: float
    shell_concentration: float
    radial_bin_width: float
    total_nonzero_power: float
    dominant_radial_mean_power: float
    dominant_bin_index: int

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def normalized_spatial_extent() -> tuple[float, float, float, float]:
    """Matplotlib extent for a lattice heat map displayed in x/L and y/L."""
    return (0.0, 1.0, 0.0, 1.0)


def _square_field(field: np.ndarray) -> np.ndarray:
    arr = np.asarray(field)
    if arr.ndim != 2 or arr.shape[0] != arr.shape[1]:
        raise ValueError(f"expected a square 2D field, received shape {arr.shape}")
    if arr.shape[0] < 2:
        raise ValueError("the lattice side length must be at least two")
    return arr


def unitary_power_spectrum(field: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return mean-subtracted unitary FFT power and angular wave-number grids."""
    arr = _square_field(field)
    centered = arr - np.mean(arr)
    fft = np.fft.fftn(centered, norm="ortho")
    power = np.abs(fft) ** 2
    L = arr.shape[0]
    k = 2.0 * np.pi * np.fft.fftfreq(L)
    kx, ky = np.meshgrid(k, k, indexing="ij")
    return power, kx, ky


def radial_power_spectrum(field: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return canonical radial-bin centers, mean power, and the 2D power array.

    Bins have width ``2*pi/L`` and are centered at integer multiples of that
    width.  The bin assignment ``floor(r/dk + 1/2)`` implements half-open
    centered annuli.  The radial statistic is the arithmetic mean of Fourier
    power in each annulus.
    """
    power, kx, ky = unitary_power_spectrum(field)
    L = power.shape[0]
    dk = 2.0 * np.pi / L
    radius = np.hypot(kx, ky)
    bin_index = np.floor(radius / dk + 0.5).astype(np.int64)
    flat_bin = bin_index.ravel()
    flat_power = power.ravel()
    counts = np.bincount(flat_bin)
    sums = np.bincount(flat_bin, weights=flat_power)
    means = np.divide(sums, counts, out=np.zeros_like(sums), where=counts > 0)
    centers = dk * np.arange(len(means), dtype=float)
    return centers, means, power


def radial_bin_width(field: np.ndarray) -> float:
    """Return the canonical radial Fourier-bin width ``2*pi/L``."""
    arr = _square_field(field)
    return float(2.0 * np.pi / arr.shape[0])


def dominant_radial_wavenumber(field: np.ndarray) -> float:
    """Return the strongest nonzero radial-bin center.

    ``numpy.argmax`` selects the first maximum, so exact ties are resolved in
    favor of the smaller radius.
    """
    centers, means, _ = radial_power_spectrum(field)
    if len(means) <= 1 or not np.any(np.isfinite(means[1:])):
        return 0.0
    dominant = int(np.nanargmax(means[1:]) + 1)
    return float(centers[dominant])


def shell_concentration(
    field: np.ndarray,
    k0: float,
    half_width: float | None = None,
) -> float:
    """Return nonzero Fourier power within a shell around ``k0``.

    The default radial half-width is one Fourier spacing, ``2*pi/L``.  This
    resolution-normalized convention is used for stripe, spot, and labyrinth.
    """
    power, kx, ky = unitary_power_spectrum(field)
    radius = np.hypot(kx, ky)
    width = radial_bin_width(field) if half_width is None else float(half_width)
    nonzero = radius > 0.0
    shell = nonzero & (np.abs(radius - float(k0)) <= width + 32.0 * np.finfo(float).eps)
    total = float(np.sum(power[nonzero]))
    return float(np.sum(power[shell]) / total) if total > 0.0 else 0.0


def radial_shell_observables(field: np.ndarray) -> SpectralObservables:
    """Compute the canonical dominant radial mode and shell concentration."""
    centers, means, power = radial_power_spectrum(field)
    if len(means) <= 1:
        raise ValueError("no nonzero radial Fourier bin is available")
    dominant = int(np.nanargmax(means[1:]) + 1)
    k_dom = float(centers[dominant])
    _, kx, ky = unitary_power_spectrum(field)
    radius = np.hypot(kx, ky)
    total = float(np.sum(power[radius > 0.0]))
    concentration = shell_concentration(field, k_dom)
    return SpectralObservables(
        k_dom=k_dom,
        shell_concentration=concentration,
        radial_bin_width=radial_bin_width(field),
        total_nonzero_power=total,
        dominant_radial_mean_power=float(means[dominant]),
        dominant_bin_index=dominant,
    )
