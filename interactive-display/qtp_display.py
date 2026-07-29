from __future__ import annotations

import base64
import csv
import io
import json
import sys
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict

import imageio
import imageio.v2 as imageio_v2
import matplotlib

matplotlib.use("Agg")

import numpy as np
from matplotlib import pyplot as plt
from PIL import Image

import qtp_kernels as q
import qtp_observables as observables
plt.rcParams.update(
    {
        "figure.dpi": 180,
        "savefig.dpi": 220,
        "font.family": "serif",
        "font.serif": [
            "Times New Roman",
            "Times",
            "TeX Gyre Termes",
            "Nimbus Roman",
            "Liberation Serif",
        ],
        "mathtext.fontset": "stix",
        "axes.unicode_minus": False,
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
        "font.size": 15.3,
        "axes.titlesize": 16.2,
        "axes.labelsize": 14.6,
        "xtick.labelsize": 11.9,
        "ytick.labelsize": 11.9,
        "legend.fontsize": 11.7,
        "lines.linewidth": 2.2,
        "axes.grid": False,
        "grid.alpha": 0.22,
        "grid.linewidth": 0.5,
        "image.interpolation": "bicubic",
    }
)

FIELD_CMAP = q.CMAP_FIELD
VMIN = q.VMIN
VMAX = q.VMAX

CERTIFIED_ISO_COEFFICIENTS = dict(Omega=1.8, Dq=0.6, Dp=4.5, nu=4.0, beta=0.40)
DEFAULT_PRESETS: Dict[str, Dict[str, Any]] = {
    "spot": dict(
        L=128,
        dt=0.05,
        T=50.0,
        frame_dt=0.5,
        seed=4,
        bump_amp=0.12,
        bump_width=2.4,
        n_bumps=32,
        p_to_q_ratio=-0.25,
        additive_noise=1e-3,
    ),
    "labyrinth": dict(
        L=128,
        dt=0.05,
        T=80.0,
        frame_dt=1.0,
        seed=0,
        noise_scale=1e-2,
    ),
    "stripe": dict(
        L=192,
        dt=0.05,
        T=160.0,
        frame_dt=2.0,
        rng_seed=7,
        random_perturbation_amplitude=1e-3,
        cosine_amplitude=1e-3,
    ),
}
CASE_DEFAULT_SEEDS = {"spot": 4, "labyrinth": 0}
SPECTRAL_THRESHOLDS = {"stripe": 0.95, "spot": 0.50, "labyrinth": 0.50}
LIVE_PREVIEW_DEBOUNCE_SECONDS = 0.45
SCHEMA_VERSION = "1.0"

SPOT_STRENGTH_OPTIONS = (
    ("Very gentle", 0.03),
    ("Gentle", 0.06),
    ("Moderate", 0.09),
    ("Reference", 0.12),
    ("Strong", 0.15),
    ("Very strong", 0.18),
    ("Maximum", 0.22),
)
SPOT_SIZE_OPTIONS = (
    ("Compact", 1.6),
    ("Small", 2.0),
    ("Reference", 2.4),
    ("Broad", 2.8),
    ("Very broad", 3.2),
)
LABYRINTH_ROUGHNESS_OPTIONS = (
    ("Very quiet", 0.003),
    ("Quiet", 0.005),
    ("Fine", 0.0075),
    ("Reference", 0.010),
    ("Rough", 0.015),
    ("Very rough", 0.022),
)


def _read_release_version() -> str:
    path = Path(__file__).resolve().with_name("VERSION")
    return path.read_text(encoding="utf-8").strip() if path.exists() else "unknown"


RELEASE_VERSION = _read_release_version()


def qualitative_options(case: str, control: str):
    table = {
        ("spot", "strength"): SPOT_STRENGTH_OPTIONS,
        ("spot", "size"): SPOT_SIZE_OPTIONS,
        ("labyrinth", "roughness"): LABYRINTH_ROUGHNESS_OPTIONS,
    }
    key = (str(case), str(control))
    try:
        return table[key]
    except KeyError as exc:
        raise ValueError(f"unknown qualitative control {key!r}") from exc


def make_live_preview_preset(
    case: str,
    *,
    seed=None,
    bump_amp=None,
    bump_width=None,
    noise_scale=None,
):
    """Return a compact preset for an in-memory preview."""
    case = str(case)
    preset = dict(DEFAULT_PRESETS[case])
    if case == "spot":
        amp = float(preset["bump_amp"] if bump_amp is None else bump_amp)
        width = float(preset["bump_width"] if bump_width is None else bump_width)
        horizon = 65.0 if amp <= 0.03 and width >= 2.4 else 50.0
        preset.update(L=64, T=horizon, frame_dt=0.5, bump_amp=amp, bump_width=width)
        if seed is not None:
            preset["seed"] = int(seed)
    elif case == "labyrinth":
        preset.update(L=64, T=80.0, frame_dt=1.0)
        if seed is not None:
            preset["seed"] = int(seed)
        if noise_scale is not None:
            preset["noise_scale"] = float(noise_scale)
    elif case == "stripe":
        preset.update(L=72, T=160.0, frame_dt=2.0)
    else:
        raise ValueError(f"unknown case={case!r}")
    return preset


@dataclass
class DisplaySummary:
    case: str
    regime: str
    mode: str
    run_scope: str
    claim_level: str
    theorem_level_claimed: bool
    reference_configuration_exact: bool
    k_dom: float
    shell_concentration: float
    shell_width: float
    no_nan: bool
    spectral_selection_passed: bool
    persistence_passed: bool
    pattern_checks_passed: bool
    finite_run_state: str
    relative_field_change: float
    final_frame_correlation: float
    k_dom_std_final_window: float
    shell_concentration_min_final_window: float
    stationary_residual: float


def _same_reference_value(value, reference) -> bool:
    return bool(np.isclose(float(value), float(reference), rtol=0.0, atol=1.0e-12))


def _is_certified_isotropic(Omega, Dq, Dp, nu, beta) -> bool:
    values = dict(Omega=Omega, Dq=Dq, Dp=Dp, nu=nu, beta=beta)
    return all(
        _same_reference_value(values[key], CERTIFIED_ISO_COEFFICIENTS[key])
        for key in CERTIFIED_ISO_COEFFICIENTS
    )


def _reference_parameters_exact(case: str, params: dict) -> bool:
    reference = DEFAULT_PRESETS[case]
    for key, value in reference.items():
        if key not in params:
            return False
        current = params[key]
        if isinstance(value, (float, int)) and isinstance(current, (float, int, np.number)):
            if not _same_reference_value(current, value):
                return False
        elif current != value:
            return False
    return True


def _simulate_custom_isotropic(
    init_kind="spot",
    L=192,
    dt=0.05,
    T=None,
    frame_dt=None,
    seed=0,
    bump_amp=0.12,
    bump_width=2.4,
    n_bumps=32,
    p_to_q_ratio=-0.25,
    additive_noise=1e-3,
    noise_scale=1e-2,
    Omega=1.8,
    Dq=0.6,
    Dp=4.5,
    nu=4.0,
    beta=0.40,
    progress_callback=None,
):
    if T is None:
        T = 50.0 if init_kind == "spot" else 80.0
    if frame_dt is None:
        frame_dt = 0.5 if init_kind == "spot" else 1.0
    L = int(L)
    dt = float(dt)
    T = float(T)
    frame_dt = float(frame_dt)

    if init_kind == "spot":
        q0, p0 = q.init_spot(
            L,
            amp=bump_amp,
            nbumps=n_bumps,
            width=bump_width,
            seed=seed,
            p_to_q_ratio=p_to_q_ratio,
            additive_noise=additive_noise,
        )
        title = "Spot"
        initial_condition = {
            "type": "localized_multi_bump",
            "rng": "numpy.PCG64",
            "seed": int(seed),
            "n_bumps": int(n_bumps),
            "bump_width": float(bump_width),
            "bump_amplitude": float(bump_amp),
            "additive_noise_amplitude": float(additive_noise),
            "p_to_q_ratio": float(p_to_q_ratio),
        }
    elif init_kind == "labyrinth":
        q0, p0 = q.init_noise(L, seed=seed, scale=noise_scale)
        title = "Labyrinth"
        initial_condition = {
            "type": "independent_gaussian_noise",
            "rng": "numpy.PCG64",
            "seed": int(seed),
            "noise_scale": float(noise_scale),
        }
    else:
        raise ValueError(f"unknown isotropic initial condition {init_kind!r}")

    qv = np.asarray(q0, dtype=float).copy()
    pv = np.asarray(p0, dtype=float).copy()
    k = 2 * np.pi * np.fft.fftfreq(L)
    kx, ky = np.meshgrid(k, k)
    lattice_symbol = q.ell(kx) + q.ell(ky)
    l11 = 1.0 - Dq * lattice_symbol
    l22 = -3.0 - Dp * lattice_symbol
    a11 = 1 - dt * l11
    a12 = -dt * Omega
    a21 = dt * Omega
    a22 = 1 - dt * l22
    determinant = a11 * a22 - a12 * a21

    nsteps = int(round(T / dt))
    sample = max(1, int(round(frame_dt / dt)))
    frames_q: list[np.ndarray] = []
    frames_p: list[np.ndarray] = []
    times: list[float] = []
    contrast: list[float] = []
    shell: list[float] = []
    progress_stride = max(1, nsteps // 100)

    for step in range(nsteps + 1):
        if progress_callback is not None and (
            step == 0 or step == nsteps or step % progress_stride == 0
        ):
            try:
                progress_callback(step, nsteps, step * dt)
            except Exception:
                pass
        pattern = qv - beta * pv
        if step % sample == 0:
            frames_q.append(qv.copy())
            frames_p.append(pv.copy())
            times.append(step * dt)
            radial_k, radial_power, _ = q.radial_spectrum(pattern)
            peak = int(np.argmax(radial_power[1:]) + 1)
            contrast.append(float(pattern.std()))
            shell.append(float(radial_power[peak]))
        if step == nsteps:
            break
        qv, pv = q.semiimplicit_step(
            qv,
            pv,
            dt,
            a11,
            a12,
            a21,
            a22,
            determinant,
            nu,
        )

    q_final = np.asarray(qv, dtype=np.float64).copy()
    p_final = np.asarray(pv, dtype=np.float64).copy()
    pattern_final = q_final - beta * p_final
    frames_q_array = np.asarray(frames_q, dtype=np.float32)
    frames_p_array = np.asarray(frames_p, dtype=np.float32)
    pattern_frames = frames_q_array - beta * frames_p_array
    kpeak = q.dominant_radial_wavenumber(pattern_final)
    certified = _is_certified_isotropic(Omega, Dq, Dp, nu, beta)

    params = dict(
        L=L,
        dt=dt,
        T=T,
        frame_dt=frame_dt,
        Omega=Omega,
        Dq=Dq,
        Dp=Dp,
        nu=nu,
        beta=beta,
        seed=int(seed),
        bump_amp=float(bump_amp),
        bump_width=float(bump_width),
        n_bumps=int(n_bumps),
        p_to_q_ratio=float(p_to_q_ratio),
        additive_noise=float(additive_noise),
        noise_scale=float(noise_scale),
        init_kind=init_kind,
        pattern_covector=[1.0, -float(beta)],
        fastest_linear_mode_k=float(q.ISOTROPIC_FASTEST_K),
        fastest_linear_right_eigenvector=[
            1.0,
            float(q.ISOTROPIC_FASTEST_RIGHT_P_OVER_Q),
        ],
    )
    return dict(
        kind=init_kind,
        title=title,
        regime="isotropic",
        times=np.asarray(times, dtype=float),
        q_frames=frames_q_array,
        p_frames=frames_p_array,
        R_frames=pattern_frames,
        q_final_float64=q_final,
        p_final_float64=p_final,
        R_final_float64=pattern_final,
        contrast=np.asarray(contrast, dtype=float),
        shell_metric=np.asarray(shell, dtype=float),
        kpeak=kpeak,
        dominant_k_measured=kpeak,
        spectral_concentration=q.shell_concentration(pattern_final, kpeak),
        stationary_residual=q.relative_stationary_residual(
            q_final,
            p_final,
            regime="isotropic",
            Omega=Omega,
            Dq=Dq,
            Dp=Dp,
            nu=nu,
        ),
        beta=beta,
        mode="certified_demo" if certified else "exploration",
        initial_condition=initial_condition,
        storage_precision={"time_series": "float32", "final_state": "float64"},
        params=params,
    )


def run_case(case="spot", progress_callback=None, **kwargs):
    """Run one pattern-formation case."""
    run_scope = kwargs.pop("run_scope", "reference")
    if run_scope not in {"reference", "preview"}:
        raise ValueError("run_scope must be 'reference' or 'preview'")

    if case == "stripe":
        allowed = {
            "L",
            "dt",
            "T",
            "frame_dt",
            "rng_seed",
            "random_perturbation_amplitude",
            "cosine_amplitude",
        }
        unknown = set(kwargs) - allowed
        if unknown:
            raise TypeError(f"Unsupported parameters for stripe: {sorted(unknown)}")
        effective = dict(DEFAULT_PRESETS["stripe"])
        effective.update(kwargs)
        if int(effective["L"]) % 12 != 0:
            raise ValueError("The stripe grid size must be divisible by 12.")
        result = q.simulate_stripe(progress_callback=progress_callback, **effective)
        result["mode"] = "certified_demo"
    elif case in {"spot", "labyrinth"}:
        allowed = {
            "L",
            "dt",
            "T",
            "frame_dt",
            "seed",
            "bump_amp",
            "bump_width",
            "n_bumps",
            "p_to_q_ratio",
            "additive_noise",
            "noise_scale",
            "Omega",
            "Dq",
            "Dp",
            "nu",
            "beta",
        }
        unknown = set(kwargs) - allowed
        if unknown:
            raise TypeError(f"Unsupported parameters for {case}: {sorted(unknown)}")
        effective = dict(DEFAULT_PRESETS[case])
        effective.update(kwargs)
        result = _simulate_custom_isotropic(
            init_kind=case,
            progress_callback=progress_callback,
            **effective,
        )
    else:
        raise ValueError(f"unknown case={case!r}")

    result["run_scope"] = run_scope
    result["reference_configuration_exact"] = bool(
        run_scope == "reference"
        and _reference_parameters_exact(case, result.get("params", {}))
        and (case == "stripe" or result.get("mode") == "certified_demo")
    )
    return result


def _final_field(result: dict) -> np.ndarray:
    return np.asarray(
        result.get("R_final_float64", result["R_frames"][-1]),
        dtype=np.float64,
    )


def persistence_checks(result: dict, window: int = 6) -> dict[str, Any]:
    """Measure whether the final pattern is stable over the saved time window."""
    frames = np.asarray(result["R_frames"], dtype=float)
    count = min(int(window), len(frames))
    tail = frames[-count:]
    eps = np.finfo(float).eps

    if len(tail) >= 2:
        previous, final = tail[-2], tail[-1]
        relative_change = float(
            np.linalg.norm(final - previous) / max(np.linalg.norm(final), eps)
        )
        a = previous.ravel() - float(previous.mean())
        b = final.ravel() - float(final.mean())
        correlation = float(np.dot(a, b) / max(np.linalg.norm(a) * np.linalg.norm(b), eps))
    else:
        relative_change = 0.0
        correlation = 1.0

    shell_width = q.radial_bin_width(tail[-1])
    k_values: list[float] = []
    shell_values: list[float] = []
    for frame in tail:
        radial_k, radial_power, _ = q.radial_spectrum(frame)
        peak = int(np.argmax(radial_power[1:]) + 1)
        dominant = float(radial_k[peak])
        k_values.append(dominant)
        shell_values.append(float(q.shell_concentration(frame, dominant, shell_width)))

    k_std = float(np.std(k_values))
    shell_min = float(np.min(shell_values))
    threshold = SPECTRAL_THRESHOLDS.get(result.get("kind"), 0.50)
    passed = bool(
        np.isfinite(relative_change)
        and relative_change <= 0.02
        and np.isfinite(correlation)
        and correlation >= 0.999
        and np.isfinite(k_std)
        and k_std <= 0.02
        and np.isfinite(shell_min)
        and shell_min >= threshold
    )
    return {
        "window_frames": count,
        "relative_field_change": relative_change,
        "final_frame_correlation": correlation,
        "k_dom_std_final_window": k_std,
        "shell_concentration_min_final_window": shell_min,
        "passed": passed,
        "thresholds": {
            "relative_field_change_max": 0.02,
            "final_frame_correlation_min": 0.999,
            "k_dom_std_max": 0.02,
            "shell_concentration_min": threshold,
        },
        "scope": "finite-run pattern stability check",
    }


def _finite_array_checks(result: dict, radial_k=None, radial_power=None) -> dict[str, bool]:
    arrays = {
        "q_frames": result.get("q_frames"),
        "p_frames": result.get("p_frames"),
        "R_frames": result.get("R_frames"),
        "times": result.get("times"),
        "R_final_float64": result.get("R_final_float64"),
        "radial_k": radial_k,
        "radial_power": radial_power,
    }
    return {
        name: bool(np.all(np.isfinite(np.asarray(value))))
        for name, value in arrays.items()
        if value is not None
    }


def summarize_result(result: dict) -> DisplaySummary:
    pattern = _final_field(result)
    radial_k, radial_power, _ = q.radial_spectrum(pattern)
    spectrum = q.canonical_spectral_observables(pattern)
    finite_checks = _finite_array_checks(result, radial_k, radial_power)
    result["finite_checks"] = finite_checks
    no_nan = bool(finite_checks and all(finite_checks.values()))

    k_dom = float(spectrum.k_dom)
    shell_width = float(spectrum.radial_bin_width)
    shell = float(spectrum.shell_concentration)
    threshold = SPECTRAL_THRESHOLDS.get(result["kind"], 0.50)
    spectral_passed = bool(np.isfinite(k_dom) and shell >= threshold)
    persistence = persistence_checks(result)
    result["persistence_checks"] = persistence
    pattern_passed = bool(no_nan and spectral_passed and persistence["passed"])

    run_scope = str(result.get("run_scope", "reference"))
    if pattern_passed:
        state = "PASS"
    elif run_scope == "preview" and no_nan and spectral_passed:
        state = "DEVELOPING"
    else:
        state = "REVIEW"
    result["finite_run_state"] = state

    reference_exact = bool(result.get("reference_configuration_exact", False))
    if reference_exact and result["kind"] == "stripe":
        claim_level = "theorem_level_reference"
    elif reference_exact and result["kind"] in {"spot", "labyrinth"}:
        claim_level = "numerical_reference_demonstration"
    else:
        claim_level = "exploratory_run"
    theorem_claimed = bool(pattern_passed and claim_level == "theorem_level_reference")

    stationary_residual = float(
        result.get(
            "stationary_residual",
            q.relative_stationary_residual(
                result["q_final_float64"],
                result["p_final_float64"],
                regime=result["regime"],
                **{
                    key: result.get("params", {}).get(key)
                    for key in ("Omega", "Dq", "Dp", "nu", "lam", "Dy")
                    if result.get("params", {}).get(key) is not None
                },
            ),
        )
    )
    result["stationary_residual"] = stationary_residual

    return DisplaySummary(
        case=str(result["kind"]),
        regime=str(result["regime"]),
        mode=str(result.get("mode", "certified_demo")),
        run_scope=run_scope,
        claim_level=claim_level,
        theorem_level_claimed=theorem_claimed,
        reference_configuration_exact=reference_exact,
        k_dom=k_dom,
        shell_concentration=shell,
        shell_width=shell_width,
        no_nan=no_nan,
        spectral_selection_passed=spectral_passed,
        persistence_passed=bool(persistence["passed"]),
        pattern_checks_passed=pattern_passed,
        finite_run_state=state,
        relative_field_change=float(persistence["relative_field_change"]),
        final_frame_correlation=float(persistence["final_frame_correlation"]),
        k_dom_std_final_window=float(persistence["k_dom_std_final_window"]),
        shell_concentration_min_final_window=float(
            persistence["shell_concentration_min_final_window"]
        ),
        stationary_residual=stationary_residual,
    )


def check_preview_scope(result: dict, summary: DisplaySummary | None = None):
    if summary is None:
        summary = summarize_result(result)
    checks = {
        "run_scope_preview": result.get("run_scope") == "preview",
        "claim_level_exploratory": summary.claim_level == "exploratory_run",
        "theorem_level_not_claimed": not summary.theorem_level_claimed,
    }
    return bool(all(checks.values())), checks


def claim_label(summary: DisplaySummary) -> str:
    return {
        "theorem_level_reference": "Theorem-level reference",
        "numerical_reference_demonstration": "Numerical reference example",
        "exploratory_run": "Exploratory run",
    }.get(summary.claim_level, summary.claim_level)


def status_cards_html(summary: DisplaySummary) -> str:
    threshold = SPECTRAL_THRESHOLDS.get(summary.case, 0.50)

    def card(title: str, state: str, details: str) -> str:
        color = "#0b6b3a" if state == "PASS" else ("#345995" if state == "DEVELOPING" else "#9b1c1c")
        return (
            '<div style="border:1px solid #ddd;border-radius:10px;padding:10px 12px;background:#fafafa;">'
            f'<div style="display:flex;justify-content:space-between;gap:12px;align-items:center;">'
            f"<b>{title}</b>"
            f'<span style="background:{color};color:white;border-radius:999px;padding:2px 9px;font-size:12px;">{state}</span>'
            "</div>"
            f'<div style="margin-top:5px;font-size:13px;color:#444;">{details}</div>'
            "</div>"
        )

    spectral_state = "PASS" if summary.spectral_selection_passed else "REVIEW"
    persistence_state = "PASS" if summary.persistence_passed else summary.finite_run_state
    return (
        '<div style="font-family:Times New Roman,Times,serif;max-width:980px;">'
        f'<div style="font-size:18px;font-weight:700;margin-bottom:8px;">{claim_label(summary)}</div>'
        '<div style="display:grid;grid-template-columns:repeat(auto-fit,minmax(250px,1fr));gap:8px;">'
        + card(
            "Turing spectral selection",
            spectral_state,
            f"k<sub>dom</sub> = {summary.k_dom:.3f}; shell concentration = {summary.shell_concentration:.3f}; threshold = {threshold:.2f}",
        )
        + card(
            "Temporal persistence",
            persistence_state,
            f"relative change = {summary.relative_field_change:.4g}; final-frame correlation = {summary.final_frame_correlation:.6f}",
        )
        + card(
            "Final-state residual",
            "PASS" if np.isfinite(summary.stationary_residual) else "REVIEW",
            f"normalized residual = {summary.stationary_residual:.4g}",
        )
        + "</div></div>"
    )


def _style_pattern_axis(axis, grid_size: int) -> None:
    ticks = [0, grid_size // 2, grid_size]
    axis.set_xticks(ticks, ["0.0", "0.5", "1.0"])
    axis.set_yticks(ticks, ["0.0", "0.5", "1.0"])
    axis.set_xlabel(r"$x/L$")
    axis.set_ylabel(r"$y/L$")
    axis.tick_params(width=0.8, length=4)
    for spine in axis.spines.values():
        spine.set_linewidth(0.8)


def _plot_radial_spectrum(axis, pattern, k_dom, shell_width, regime=None):
    radial_k, radial_power, _ = q.radial_spectrum(pattern)
    axis.plot(radial_k, radial_power, color="#222222", lw=1.8)
    axis.axvline(k_dom, color="#8b1a1a", ls="--", lw=1.2, label=rf"$k_{{\rm dom}}={k_dom:.3f}$")
    axis.axvspan(
        max(0.0, k_dom - shell_width),
        k_dom + shell_width,
        color="#999999",
        alpha=0.18,
        lw=0,
    )
    if regime:
        try:
            curve_k, growth, _ = q.linear_dispersion_curve(regime)
            growth = np.asarray(growth, dtype=float)
            if np.nanmax(growth) > np.nanmin(growth):
                scaled = (growth - np.nanmin(growth)) / (np.nanmax(growth) - np.nanmin(growth))
                scaled *= max(float(np.nanmax(radial_power)), 1e-12)
                axis.plot(curve_k, scaled, color="#4f6d7a", lw=1.0, alpha=0.7, label="scaled linear growth")
        except Exception:
            pass
    axis.set_xlim(left=0)
    axis.set_ylim(bottom=0)
    axis.set_xlabel(r"radial wave number $k$")
    axis.set_ylabel("radial mean power")
    axis.set_title("Radial spectrum")
    axis.legend(frameon=True, fontsize=8, loc="upper right")


def make_case_figure(result: dict, out_path, title: str | None = None) -> DisplaySummary:
    """Save a pattern-focused reference figure."""
    out_path = Path(out_path)
    summary = summarize_result(result)
    pattern = _final_field(result)
    grid_size = int(pattern.shape[-1])

    fig = plt.figure(figsize=(12.0, 7.4), constrained_layout=False)
    layout = fig.add_gridspec(
        2,
        2,
        width_ratios=[1.55, 1.0],
        height_ratios=[1.0, 0.75],
        left=0.065,
        right=0.97,
        bottom=0.10,
        top=0.90,
        wspace=0.28,
        hspace=0.35,
    )
    pattern_axis = fig.add_subplot(layout[:, 0])
    image = pattern_axis.imshow(
        pattern,
        origin="lower",
        extent=(0, grid_size, 0, grid_size),
        cmap=FIELD_CMAP,
        vmin=VMIN,
        vmax=VMAX,
        interpolation="nearest",
        aspect="equal",
    )
    _style_pattern_axis(pattern_axis, grid_size)
    pattern_axis.set_title("Final pattern field", fontsize=14)
    colorbar = fig.colorbar(image, ax=pattern_axis, fraction=0.045, pad=0.035)
    colorbar.set_ticks([VMIN, 0.0, VMAX])
    colorbar.set_label(r"$R^{\rm pat}$")

    spectrum_axis = fig.add_subplot(layout[0, 1])
    _plot_radial_spectrum(
        spectrum_axis,
        pattern,
        summary.k_dom,
        summary.shell_width,
        regime=result.get("regime"),
    )

    text_axis = fig.add_subplot(layout[1, 1])
    text_axis.axis("off")
    text_axis.text(
        0.0,
        1.0,
        claim_label(summary),
        ha="left",
        va="top",
        fontsize=13,
        fontweight="bold",
    )
    lines = [
        f"Pattern checks: {summary.finite_run_state}",
        f"Dominant wave number: {summary.k_dom:.6f}",
        f"Shell concentration: {summary.shell_concentration:.6f}",
        f"Relative final-frame change: {summary.relative_field_change:.6g}",
        f"Final-frame correlation: {summary.final_frame_correlation:.8f}",
        f"Final-state residual: {summary.stationary_residual:.6g}",
    ]
    text_axis.text(
        0.0,
        0.84,
        "\n".join(lines),
        ha="left",
        va="top",
        fontsize=11,
        linespacing=1.6,
        color="#333333",
    )
    text_axis.text(
        0.0,
        0.10,
        "File integrity and numerical comparison are reported by the verification tools.",
        ha="left",
        va="bottom",
        fontsize=9,
        color="#555555",
        wrap=True,
    )

    fig.suptitle(title or f"{result['title']} reference pattern", fontsize=16, y=0.965)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)
    return summary


def _open_mp4_writer(out_path, fps: float):
    out_path = Path(out_path)
    if out_path.suffix.lower() != ".mp4":
        out_path = out_path.with_suffix(".mp4")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    return imageio_v2.get_writer(
        str(out_path),
        format="FFMPEG",
        fps=fps,
        codec="libx264",
        quality=8,
        macro_block_size=16,
    )


def _selected_frame_indices(count: int, maximum: int = 61) -> np.ndarray:
    if count <= maximum:
        return np.arange(count, dtype=int)
    return np.unique(np.linspace(0, count - 1, maximum, dtype=int))


def movie_frame_times(result: dict, maximum: int = 61) -> list[float]:
    times = np.asarray(result["times"], dtype=float)
    return [float(times[index]) for index in _selected_frame_indices(len(times), maximum)]


def make_case_movie(
    result: dict,
    out_path,
    fps: float = 8,
    pattern_vmax: float = VMAX,
):
    """Write a pattern-only research movie."""
    import qtp_movie_renderer as renderer

    frames = np.asarray(result["R_frames"])
    times = np.asarray(result["times"], dtype=float)
    indices = _selected_frame_indices(len(frames))
    writer = _open_mp4_writer(out_path, fps)
    try:
        for index in indices:
            writer.append_data(
                renderer.movie_frame(
                    result,
                    frames[index],
                    float(times[index]),
                    pattern_color_vmax=pattern_vmax,
                )
            )
    finally:
        writer.close()
    return Path(out_path)


def _browser_frame(field: np.ndarray, *, pattern_vmax: float, width: int = 760) -> str:
    values = np.asarray(field, dtype=float)
    normalized = np.clip(
        (values + abs(pattern_vmax)) / max(2 * abs(pattern_vmax), np.finfo(float).eps),
        0.0,
        1.0,
    )
    rgba = plt.get_cmap(FIELD_CMAP)(normalized)
    image = Image.fromarray((rgba[..., :3] * 255).astype(np.uint8), mode="RGB")
    image = image.resize((width, width), Image.Resampling.BICUBIC)
    buffer = io.BytesIO()
    image.save(buffer, format="JPEG", quality=86, optimize=True)
    return "data:image/jpeg;base64," + base64.b64encode(buffer.getvalue()).decode("ascii")


def client_side_time_explorer_html(result: dict, initial: str = "final") -> str:
    """Build a browser-based player for the precomputed pattern frames."""
    frames = np.asarray(result["R_frames"])
    times = np.asarray(result["times"], dtype=float)
    indices = _selected_frame_indices(len(frames))
    selected_frames = [_browser_frame(frames[index], pattern_vmax=VMAX) for index in indices]
    selected_times = [float(times[index]) for index in indices]
    uid = "qtp_time_" + uuid.uuid4().hex
    start_index = 0 if initial == "zero" else len(selected_frames) - 1
    frames_json = json.dumps(selected_frames)
    times_json = json.dumps(selected_times)
    alt = f"{result.get('title', result.get('kind', 'Pattern'))} pattern through time"

    return f'''<div id="{uid}" class="qtp-time-explorer" data-qtp-time-explorer="true">
<style>
#{uid}.qtp-time-explorer {{
  width:min(96vw,980px); margin:0 auto; padding:8px; box-sizing:border-box;
  font-family:"Times New Roman",Times,"TeX Gyre Termes","Nimbus Roman",serif;
  color:#171717;
}}
#{uid} .qtp-player {{
  display:flex; flex-direction:column; align-items:center; gap:8px;
  border:1px solid #d7d7d7; border-radius:14px; background:#fff; padding:8px;
}}
#{uid} img {{
  display:block; width:min(100%,760px); max-height:72vh; object-fit:contain;
  image-rendering:auto; background:#fff;
}}
#{uid} .qtp-controls {{
  width:100%; display:grid; grid-template-columns:auto auto minmax(220px,1fr) auto auto;
  gap:8px; align-items:center;
}}
#{uid} button, #{uid} select {{
  min-height:40px; border:1px solid #777; border-radius:8px; background:#fff;
  padding:5px 10px; font:inherit; cursor:pointer;
}}
#{uid} input[type=range] {{ width:100%; }}
#{uid} .qtp-time {{ min-width:120px; text-align:right; font-variant-numeric:tabular-nums; }}
@media (max-width:720px) {{
  #{uid} .qtp-controls {{ grid-template-columns:repeat(2,1fr); }}
  #{uid} input[type=range] {{ grid-column:1/-1; }}
  #{uid} .qtp-time {{ grid-column:1/-1; text-align:center; }}
}}
</style>
<div class="qtp-player">
  <img data-role="frame" alt="{alt}">
  <div class="qtp-controls" role="group" aria-label="Pattern playback controls">
    <button type="button" data-action="replay">Replay formation</button>
    <button type="button" data-action="pause">Pause</button>
    <input data-role="slider" type="range" min="0" max="{max(len(selected_frames)-1,0)}" value="{start_index}" step="1" aria-label="Evolution frame">
    <select data-role="speed" aria-label="Playback speed">
      <option value="0.5">0.5×</option>
      <option value="1" selected>1×</option>
      <option value="2">2×</option>
    </select>
    <label><input data-role="loop" type="checkbox" checked> Loop</label>
    <span class="qtp-time" data-role="time"></span>
  </div>
</div>
<script>(function(){{
  const root=document.getElementById('{uid}');
  const frames={frames_json};
  const times={times_json};
  const image=root.querySelector('[data-role=frame]');
  const slider=root.querySelector('[data-role=slider]');
  const speed=root.querySelector('[data-role=speed]');
  const loop=root.querySelector('[data-role=loop]');
  const timeLabel=root.querySelector('[data-role=time]');
  let index={start_index}; let timer=null;
  function update(){{
    index=Math.min(frames.length-1,Math.max(0,Number(index)||0));
    image.src=frames[index]; slider.value=String(index);
    const finalTime=times.length?Number(times[times.length-1]):0;
    const current=times.length?Number(times[index]):0;
    timeLabel.textContent=`t = ${{current.toFixed(1)}} / ${{finalTime.toFixed(1)}}`;
  }}
  function pause(){{ if(timer!==null){{window.clearInterval(timer);timer=null;}} }}
  function play(fromStart=false){{
    pause(); if(fromStart) index=0; update();
    const interval=Math.max(40,125/Math.max(Number(speed.value)||1,0.1));
    timer=window.setInterval(()=>{{
      if(index>=frames.length-1){{ if(loop.checked) index=0; else {{pause();return;}} }} else index+=1;
      update();
    }},interval);
  }}
  root.querySelector('[data-action=replay]').addEventListener('click',()=>play(true));
  root.querySelector('[data-action=pause]').addEventListener('click',pause);
  slider.addEventListener('input',()=>{{pause();index=Number(slider.value);update();}});
  speed.addEventListener('change',()=>{{if(timer!==null)play(false);}});
  update();
  window['{uid}']={{root,play,pause,update,get index(){{return index;}}}};
}})();</script>
</div>'''


def effective_parameters(result: dict) -> dict[str, Any]:
    return dict(result.get("params", {}))


def summary_to_provenance(summary: DisplaySummary, result: dict | None = None) -> dict[str, Any]:
    params = effective_parameters(result or {})
    frame_times = movie_frame_times(result) if result is not None else []
    grid_size = int(np.asarray(result["R_frames"][0]).shape[-1]) if result is not None else None
    verification_state = {
        "status": "not_run",
        "checked_at": None,
        "report": None,
    }
    persistence = dict((result or {}).get("persistence_checks", {}))
    finite_checks = dict((result or {}).get("finite_checks", {}))

    return {
        "schema_version": SCHEMA_VERSION,
        "case": summary.case,
        "mode": summary.mode,
        "run_scope": summary.run_scope,
        "claim_level": summary.claim_level,
        "release_version": RELEASE_VERSION,
        "parameters": params,
        "initial_condition": dict((result or {}).get("initial_condition", {})),
        "notebook_execution": (result or {}).get("notebook_execution"),
        "storage_precision": dict((result or {}).get("storage_precision", {})),
        "grid": {
            "pattern_L": grid_size,
            "pattern_extent": f"0..{grid_size}" if grid_size is not None else None,
        },
        "movie_plan": {
            "frame_times": frame_times,
            "n_frames_expected": len(frame_times),
            "fps": 8,
            "expected_final_time": frame_times[-1] if frame_times else None,
            "field_vmin": float(VMIN),
            "field_vmax": float(VMAX),
            "layout": "single_pattern_panel",
        },
        "computed_quantities": {
            "k_dom": summary.k_dom,
            "k_dom_definition": observables.OBSERVABLE_CONTRACT["dominant_mode_rule"],
            "shell_concentration": summary.shell_concentration,
            "shell_width": summary.shell_width,
            "spectrum_normalization": observables.OBSERVABLE_CONTRACT["dft_normalization"],
            "stationary_residual": summary.stationary_residual,
            "stationary_residual_definition": "||F(q_final,p_final)||_2 / ||(q_final,p_final)||_2",
            "finite_run_state": summary.finite_run_state,
        },
        "numerical_observable_contract": observables.OBSERVABLE_CONTRACT,
        "persistence_checks": persistence,
        "finite_value_checks": finite_checks,
        "verification_tolerances": {
            "numerical_comparison_rtol": 1e-7,
            "numerical_comparison_atol": 1e-10,
            "movie_fps_atol": 0.2,
        },
        "certification_scope": {
            "mode": summary.mode,
            "run_scope": summary.run_scope,
            "claim_level": summary.claim_level,
            "reference_configuration_exact": summary.reference_configuration_exact,
            "pattern_checks_passed": summary.pattern_checks_passed,
            "finite_run_state": summary.finite_run_state,
            "theorem_level_claimed": summary.theorem_level_claimed,
            "current_output_verified": False,
            "current_output_verification": verification_state,
            "bundled_reference_verified": False,
            "bundled_reference_verification": verification_state.copy(),
            "note": (
                "The theorem-level designation applies only to the exact Stripe reference parameters. "
                "Spot and Labyrinth are numerical reference examples. Modified or preview runs are exploratory."
            ),
        },
        "verification": {
            "no_nan": summary.no_nan,
            "spectral_selection_passed": summary.spectral_selection_passed,
            "persistence_passed": summary.persistence_passed,
            "pattern_checks_passed": summary.pattern_checks_passed,
            "finite_run_state": summary.finite_run_state,
            "theorem_level_claimed": summary.theorem_level_claimed,
        },
        "software": {
            "python": sys.version.split()[0],
            "supported_environment": "qtp-display Python 3.13, see environment.yml",
            "numpy": np.__version__,
            "matplotlib": matplotlib.__version__,
            "imageio": getattr(imageio, "__version__", "unknown"),
        },
    }


def _sha256_file(path: Path) -> str:
    import hashlib

    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def _movie_metadata(path: Path) -> dict[str, Any]:
    reader = imageio_v2.get_reader(str(path), "ffmpeg")
    try:
        try:
            n_frames = int(reader.count_frames())
        except Exception:
            metadata = reader.get_meta_data()
            n_frames = int(
                round(
                    float(metadata.get("duration", 0.0))
                    * float(metadata.get("fps", 0.0))
                )
            )
        metadata = reader.get_meta_data()
        width, height = tuple(metadata.get("size", (None, None)))
        fps = float(metadata.get("fps", 0.0))
    finally:
        reader.close()
    return {
        "n_frames": n_frames,
        "fps": fps,
        "width": width,
        "height": height,
    }


def write_exhibit_manifest(
    results,
    display_dir,
    *,
    fps: float = 8,
    pattern_vmin: float = VMIN,
    pattern_vmax: float = VMAX,
    filename: str = "display_exhibit_manifest.csv",
):
    """Bind exhibit movies to their simulation and video metadata."""
    import qtp_movie_renderer as renderer

    display_dir = Path(display_dir)
    display_dir.mkdir(parents=True, exist_ok=True)
    rows = []
    for result in results:
        case = str(result["kind"])
        path = display_dir / f"display_{case}_pattern.mp4"
        if not path.exists() or path.stat().st_size <= 0:
            raise FileNotFoundError(f"missing exhibit movie: {path}")
        metadata = _movie_metadata(path)
        frame_times = movie_frame_times(result)
        rows.append(
            {
                "case": case,
                "filename": path.name,
                "sha256": _sha256_file(path),
                "size_bytes": path.stat().st_size,
                "n_frames": metadata["n_frames"],
                "fps": metadata["fps"],
                "width": metadata["width"],
                "height": metadata["height"],
                "physical_final_time": float(frame_times[-1]),
                "field_vmin": float(pattern_vmin),
                "field_vmax": float(pattern_vmax),
                "pattern_grid_L": int(np.asarray(result["R_frames"][0]).shape[-1]),
                "layout": "single_pattern_panel",
                "font_family": "Times",
                "axis_label_font_size": renderer.AXIS_LABEL_FONT_SIZE,
                "axis_label_scale_reference": renderer.AXIS_LABEL_SCALE_REFERENCE,
            }
        )

    out_path = display_dir / filename
    fieldnames = [
        "case",
        "filename",
        "sha256",
        "size_bytes",
        "n_frames",
        "fps",
        "width",
        "height",
        "physical_final_time",
        "field_vmin",
        "field_vmax",
        "pattern_grid_L",
        "layout",
        "font_family",
        "axis_label_font_size",
        "axis_label_scale_reference",
    ]
    with out_path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    return out_path


def attach_movie_metadata(summary_path, movie_path):
    summary_path = Path(summary_path)
    movie_path = Path(movie_path)
    if not summary_path.exists() or not movie_path.exists():
        return
    data = json.loads(summary_path.read_text(encoding="utf-8"))
    metadata = _movie_metadata(movie_path)
    data["movie_file"] = {
        "path": movie_path.name,
        "sha256": _sha256_file(movie_path),
        "size_bytes": movie_path.stat().st_size,
        **metadata,
    }
    summary_path.write_text(json.dumps(data, indent=2), encoding="utf-8")


def write_run_manifest(run_dir, files, manifest_name="run_manifest.csv"):
    run_dir = Path(run_dir)
    rows = []
    for file_path in files:
        path = Path(file_path)
        if not path.exists():
            continue
        try:
            relative_path = str(path.relative_to(run_dir))
        except ValueError:
            relative_path = str(path)
        rows.append(
            {
                "file": path.name,
                "relative_path": relative_path,
                "size_bytes": path.stat().st_size,
                "sha256": _sha256_file(path),
            }
        )
    if not rows:
        return None
    out_path = run_dir / manifest_name
    with out_path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(
            stream,
            fieldnames=["file", "relative_path", "size_bytes", "sha256"],
        )
        writer.writeheader()
        writer.writerows(rows)
    return out_path


def save_summary(summary: DisplaySummary, out_path, result=None):
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(
        json.dumps(summary_to_provenance(summary, result=result), indent=2),
        encoding="utf-8",
    )


