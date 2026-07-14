"""Simulation, diagnostics, and visualization for the display package."""

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

import matplotlib
import numpy as np

matplotlib.use("Agg")
import imageio
import imageio.v2 as imageio_v2
from matplotlib import font_manager
from matplotlib import pyplot as plt
from matplotlib.patches import FancyBboxPatch
from PIL import Image, ImageDraw, ImageFont

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
    }
)

FIELD_CMAP = q.CMAP_FIELD
DIAGNOSTIC_CMAP = q.CMAP_ENT
VMIN = q.VMIN
VMAX = q.VMAX

CERTIFIED_ISO_COEFFICIENTS = dict(Omega=1.8, Dq=0.6, Dp=4.5, nu=4.0, beta=0.40)
# Defaults match the bundled reference results in display/.
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
    "labyrinth": dict(L=128, dt=0.05, T=80.0, frame_dt=1.0, seed=0, noise_scale=1e-2),
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


def _read_release_version() -> str:
    path = Path(__file__).resolve().with_name("VERSION")
    return path.read_text(encoding="utf-8").strip() if path.exists() else "unknown"


RELEASE_VERSION = _read_release_version()


# Labels used by the qualitative controls. Exact values are stored with exported results.
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
LIVE_PREVIEW_DEBOUNCE_SECONDS = 0.45


def _times_pil_font(size=18, bold=False):
    """Return the first available Times-family font for PIL renderings."""
    try:
        prop = font_manager.FontProperties(
            family=[
                "Times New Roman",
                "Times",
                "TeX Gyre Termes",
                "Nimbus Roman",
                "Liberation Serif",
            ],
            weight="bold" if bold else "normal",
        )
        return ImageFont.truetype(font_manager.findfont(prop, fallback_to_default=True), size)
    except Exception:
        return ImageFont.load_default()


def qualitative_options(case: str, control: str):
    """Return labeled parameter choices for the research interface."""
    key = (str(case), str(control))
    table = {
        ("spot", "strength"): SPOT_STRENGTH_OPTIONS,
        ("spot", "size"): SPOT_SIZE_OPTIONS,
        ("labyrinth", "roughness"): LABYRINTH_ROUGHNESS_OPTIONS,
    }
    try:
        return table[key]
    except KeyError as exc:
        raise ValueError(f"unknown qualitative control {key!r}") from exc


def make_live_preview_preset(
    case: str, *, seed=None, bump_amp=None, bump_width=None, noise_scale=None
):
    """Return the settings used for a quick live preview.

    The reduced grid and time interval are chosen for responsive interaction.
    They are display settings, not theorem constants.
    """
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


def check_preview_scope(result, summary=None):
    """Check that a live preview is recorded as exploratory."""
    if summary is None:
        summary = summarize_result(result)
    checks = {
        "run_scope_preview": result.get("run_scope") == "preview",
        "claim_level_exploratory": summary.claim_level == "exploratory_run",
        "theorem_level_not_claimed": not summary.theorem_level_claimed,
        "certified_claim_not_asserted": not summary.certified_strong_qtp_claimed,
    }
    return bool(all(checks.values())), checks


SYMPLECTIC_THRESHOLD = 0.5
STABILITY_THRESHOLD = 0.5
EXCESS_TOL = 1.0e-6
SPECTRAL_THRESHOLDS = {"stripe": 0.95, "spot": 0.50, "labyrinth": 0.50}
SCHEMA_VERSION = "1.0"


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
    mean_diagnostic: float
    max_diagnostic: float
    min_nupt: float
    min_nu_phys: float
    strong_qtp: bool
    diagnostics_passed: bool
    certified_strong_qtp_claimed: bool
    max_geff_over_keff: float
    no_nan: bool
    physical_covariance: bool
    npt_diagnostic: bool
    stable_gaussian_reduction: bool
    pattern_excess_positive: bool
    spectral_selection_passed: bool
    persistence_passed: bool
    finite_run_state: str
    relative_field_change: float
    final_frame_correlation: float
    k_dom_std_final_window: float
    shell_concentration_min_final_window: float
    diagnostic_relative_drift: float
    stationary_residual: float
    physical_margin: float
    npt_margin: float
    stability_margin: float
    excess_margin: float


def _same_reference_value(value, reference) -> bool:
    return bool(np.isclose(float(value), float(reference), rtol=0.0, atol=1.0e-12))


def _is_certified_isotropic(Omega, Dq, Dp, nu, beta):
    vals = dict(Omega=Omega, Dq=Dq, Dp=Dp, nu=nu, beta=beta)
    return all(
        _same_reference_value(vals[k], CERTIFIED_ISO_COEFFICIENTS[k])
        for k in CERTIFIED_ISO_COEFFICIENTS
    )


def _reference_parameters_exact(case: str, params: dict) -> bool:
    reference = DEFAULT_PRESETS[case]
    for key, value in reference.items():
        if key not in params:
            return False
        if isinstance(value, (float, int)) and isinstance(params[key], (float, int, np.number)):
            if not _same_reference_value(params[key], value):
                return False
        elif params[key] != value:
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
        title = "Spot from bump initial data"
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
        title = "Labyrinth from noise initial data"
        initial_condition = {
            "type": "independent_gaussian_noise",
            "rng": "numpy.PCG64",
            "seed": int(seed),
            "noise_scale": float(noise_scale),
        }
    else:
        raise ValueError(f"unknown isotropic init_kind={init_kind}")
    qv = np.asarray(q0, float).copy()
    pv = np.asarray(p0, float).copy()
    k = 2 * np.pi * np.fft.fftfreq(L)
    KX, KY = np.meshgrid(k, k)
    eltot = q.ell(KX) + q.ell(KY)
    L11 = 1.0 - Dq * eltot
    L22 = -3.0 - Dp * eltot
    A11 = 1 - dt * L11
    A12 = -dt * Omega
    A21 = dt * Omega
    A22 = 1 - dt * L22
    det = A11 * A22 - A12 * A21
    nsteps = int(round(T / dt))
    sample = max(1, int(round(frame_dt / dt)))
    frames_q = []
    frames_p = []
    times = []
    contrast = []
    shell = []
    progress_stride = max(1, nsteps // 100)
    for n in range(nsteps + 1):
        if progress_callback is not None and (n == 0 or n == nsteps or n % progress_stride == 0):
            try:
                progress_callback(n, nsteps, n * dt)
            except Exception:
                pass
        R = qv - beta * pv
        if n % sample == 0:
            frames_q.append(qv.copy())
            frames_p.append(pv.copy())
            times.append(n * dt)
            kr, ps, _ = q.radial_spectrum(R)
            j = np.argmax(ps[1:]) + 1
            contrast.append(R.std())
            shell.append(ps[j])
        if n == nsteps:
            break
        qv, pv = q.semiimplicit_step(qv, pv, dt, A11, A12, A21, A22, det, nu)
    q_final_float64 = np.asarray(qv, dtype=np.float64).copy()
    p_final_float64 = np.asarray(pv, dtype=np.float64).copy()
    R_final_float64 = q_final_float64 - beta * p_final_float64
    frames_q = np.asarray(frames_q, dtype=np.float32)
    frames_p = np.asarray(frames_p, dtype=np.float32)
    R_frames = frames_q - beta * frames_p
    kpeak = q.dominant_radial_wavenumber(R_final_float64)
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
        times=np.array(times),
        q_frames=frames_q,
        p_frames=frames_p,
        R_frames=R_frames,
        q_final_float64=q_final_float64,
        p_final_float64=p_final_float64,
        R_final_float64=R_final_float64,
        contrast=np.array(contrast),
        shell_metric=np.array(shell),
        kpeak=kpeak,
        dominant_k_measured=kpeak,
        spectral_concentration=q.shell_concentration(R_final_float64, kpeak),
        stationary_residual=q.relative_stationary_residual(
            q_final_float64,
            p_final_float64,
            regime="isotropic",
            Omega=Omega,
            Dq=Dq,
            Dp=Dp,
            nu=nu,
        ),
        beta=beta,
        mode=("certified_demo" if certified else "exploration"),
        initial_condition=initial_condition,
        storage_precision={"time_series": "float32", "final_diagnostic": "float64"},
        params=params,
    )


def run_case(case="spot", progress_callback=None, **kwargs):
    """Run one display case and classify the resulting reference claim.

    Unknown parameters are rejected.  The theorem-level reference claim is
    available only for the exact period-compatible stripe preset.  Spot and
    labyrinth reference presets are numerical reference demonstrations.
    Parameter-modified or preview runs are exploratory even when their finite-
    run diagnostics pass.
    """
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
        stripe_L = int(effective["L"])
        if stripe_L % 12 != 0:
            raise ValueError(
                "Certified stripe claim unavailable: grid size L is not commensurate with the period-12 branch. Choose L divisible by 12, such as 96 or 192."
            )
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
            init_kind=case, progress_callback=progress_callback, **effective
        )
    else:
        raise ValueError(f"unknown case={case}")
    result["run_scope"] = run_scope
    result["reference_configuration_exact"] = bool(
        run_scope == "reference"
        and _reference_parameters_exact(case, result.get("params", {}))
        and (case == "stripe" or result.get("mode") == "certified_demo")
    )
    diagnostic_stride = 1 if run_scope == "reference" else 2
    E_frames, mean_E, min_nupt, min_nu_phys, E_times = q.fluctuation_entanglement_maps(
        result, stride=diagnostic_stride, max_frames=61
    )
    storage = dict(result.get("storage_precision", {}))
    storage.update(
        {
            "diagnostic_time_series": "float32",
            "diagnostic_spatial_stride": int(diagnostic_stride),
            "diagnostic_grid": "native"
            if diagnostic_stride == 1
            else f"stride-{diagnostic_stride}",
        }
    )
    result["storage_precision"] = storage
    result["diagnostic_spatial_stride"] = int(diagnostic_stride)
    E_final_hi, mean_final, max_final, min_nupt_hi, min_nu_phys_hi = (
        q.fluctuation_entanglement_final_map(result)
    )
    result.update(
        dict(
            E_frames=E_frames,
            mean_E=mean_E,
            min_nupt=min_nupt,
            min_nu_phys=min_nu_phys,
            E_times=E_times,
            E_final_hi=E_final_hi,
            mean_E_final=mean_final,
            max_E_final=max_final,
            min_nupt_final=min_nupt_hi,
            min_nu_phys_final=min_nu_phys_hi,
        )
    )
    return result


def _final_field(result):
    return np.asarray(result.get("R_final_float64", result["R_frames"][-1]), dtype=np.float64)


def persistence_diagnostics(result, window=6):
    """Finite-run stationarity/persistence checks (display diagnostics only)."""
    R_frames = np.asarray(result["R_frames"])
    n = min(int(window), len(R_frames))
    tail = R_frames[-n:].astype(np.float64, copy=False)
    eps = 1e-15
    if len(tail) >= 2:
        a, b = tail[-2], tail[-1]
        relative_change = float(np.linalg.norm(b - a) / (np.linalg.norm(b) + eps))
        aa = a.ravel() - float(a.mean())
        bb = b.ravel() - float(b.mean())
        correlation = float(np.dot(aa, bb) / (np.linalg.norm(aa) * np.linalg.norm(bb) + eps))
    else:
        relative_change, correlation = 0.0, 1.0
    shell_width = q.radial_bin_width(tail[-1])
    k_values, shell_values = [], []
    for frame in tail:
        kr, ps, _ = q.radial_spectrum(frame)
        j = int(np.argmax(ps[1:]) + 1)
        kval = float(kr[j])
        k_values.append(kval)
        shell_values.append(float(q.shell_concentration(frame, kval, shell_width)))
    k_std = float(np.std(k_values)) if k_values else float("nan")
    shell_min = float(np.min(shell_values)) if shell_values else float("nan")
    E_frames = np.asarray(result.get("E_frames", []))
    if len(E_frames) >= 2:
        Etail = E_frames[-min(n, len(E_frames)) :]
        maxima = np.nanmax(Etail, axis=tuple(range(1, Etail.ndim)))
        diagnostic_drift = float(abs(maxima[-1] - maxima[0]) / (abs(maxima[-1]) + eps))
    else:
        diagnostic_drift = 0.0
    passed = bool(
        np.isfinite(relative_change)
        and relative_change <= 0.02
        and np.isfinite(correlation)
        and correlation >= 0.999
        and np.isfinite(k_std)
        and k_std <= 0.02
        and np.isfinite(shell_min)
        and shell_min >= SPECTRAL_THRESHOLDS.get(result.get("kind"), 0.50)
        and np.isfinite(diagnostic_drift)
        and diagnostic_drift <= 0.05
    )
    return {
        "window_frames": n,
        "relative_field_change": relative_change,
        "final_frame_correlation": correlation,
        "k_dom_std_final_window": k_std,
        "shell_concentration_min_final_window": shell_min,
        "diagnostic_relative_drift": diagnostic_drift,
        "passed": passed,
        "thresholds": {
            "relative_field_change_max": 0.02,
            "final_frame_correlation_min": 0.999,
            "k_dom_std_max": 0.02,
            "shell_concentration_min": SPECTRAL_THRESHOLDS.get(result.get("kind"), 0.50),
            "diagnostic_relative_drift_max": 0.05,
        },
        "scope": "finite-run display diagnostic, not a theorem constant",
    }


def _finite_array_checks(result, kr=None, ps=None):
    arrays = {
        "q_frames": result.get("q_frames"),
        "p_frames": result.get("p_frames"),
        "R_frames": result.get("R_frames"),
        "E_frames": result.get("E_frames"),
        "E_final_hi": result.get("E_final_hi"),
        "times": result.get("times"),
        "E_times": result.get("E_times"),
        "mean_E": result.get("mean_E"),
        "min_nupt": result.get("min_nupt"),
        "min_nu_phys": result.get("min_nu_phys"),
        "R_final_float64": result.get("R_final_float64"),
        "radial_k": kr,
        "radial_power": ps,
    }
    checks = {}
    for name, value in arrays.items():
        if value is None:
            continue
        checks[name] = bool(np.all(np.isfinite(np.asarray(value))))
    return checks


def summarize_result(result):
    R = _final_field(result)
    kr, ps, _ = q.radial_spectrum(R)
    spectral_observables = q.canonical_spectral_observables(R)
    k_dom = float(spectral_observables.k_dom)
    shell_width = float(spectral_observables.radial_bin_width)
    shell = float(spectral_observables.shell_concentration)
    max_ratio = float(q.max_geff_kappa_ratio(result))
    finite_checks = _finite_array_checks(result, kr=kr, ps=ps)
    result["finite_checks"] = finite_checks
    no_nan = bool(finite_checks and all(finite_checks.values()))
    physical_margin = float(result["min_nu_phys_final"] - SYMPLECTIC_THRESHOLD)
    npt_margin = float(SYMPLECTIC_THRESHOLD - result["min_nupt_final"])
    stability_margin = float(STABILITY_THRESHOLD - max_ratio)
    excess_margin = float(result["max_E_final"] - EXCESS_TOL)
    physical = bool(physical_margin > 0)
    npt = bool(npt_margin > 0)
    stable = bool(stability_margin > 0)
    excess = bool(excess_margin > 0)
    spectral_threshold = SPECTRAL_THRESHOLDS.get(result["kind"], 0.50)
    spectral = bool(shell >= spectral_threshold and np.isfinite(k_dom))
    persistence = persistence_diagnostics(result)
    result["persistence_diagnostics"] = persistence
    pre_persistence_ok = bool(no_nan and physical and npt and stable and excess and spectral)
    strong = bool(pre_persistence_ok and persistence["passed"])
    run_scope = result.get("run_scope", "reference")
    if strong:
        finite_run_state = "PASS"
    elif run_scope == "preview" and pre_persistence_ok:
        finite_run_state = "DEVELOPING"
    else:
        finite_run_state = "REVIEW"
    result["finite_run_state"] = finite_run_state
    reference_exact = bool(result.get("reference_configuration_exact", False))
    if reference_exact and result["kind"] == "stripe":
        claim_level = "theorem_level_reference"
    elif reference_exact and result["kind"] in {"spot", "labyrinth"}:
        claim_level = "numerical_reference_demonstration"
    else:
        claim_level = "exploratory_run"
    theorem_claimed = bool(strong and claim_level == "theorem_level_reference")
    stationary_residual = float(
        result.get(
            "stationary_residual",
            q.relative_stationary_residual(
                result["q_final_float64"],
                result["p_final_float64"],
                regime=result["regime"],
                **{
                    k: result.get("params", {}).get(k)
                    for k in ("Omega", "Dq", "Dp", "nu", "lam", "Dy")
                    if result.get("params", {}).get(k) is not None
                },
            ),
        )
    )
    result["stationary_residual"] = stationary_residual
    return DisplaySummary(
        case=result["kind"],
        regime=result["regime"],
        mode=result.get("mode", "certified_demo"),
        run_scope=run_scope,
        claim_level=claim_level,
        theorem_level_claimed=theorem_claimed,
        reference_configuration_exact=reference_exact,
        k_dom=k_dom,
        shell_concentration=shell,
        shell_width=shell_width,
        mean_diagnostic=float(result["mean_E_final"]),
        max_diagnostic=float(result["max_E_final"]),
        min_nupt=float(result["min_nupt_final"]),
        min_nu_phys=float(result["min_nu_phys_final"]),
        strong_qtp=strong,
        diagnostics_passed=strong,
        certified_strong_qtp_claimed=theorem_claimed,
        max_geff_over_keff=max_ratio,
        no_nan=no_nan,
        physical_covariance=physical,
        npt_diagnostic=npt,
        stable_gaussian_reduction=stable,
        pattern_excess_positive=excess,
        spectral_selection_passed=spectral,
        persistence_passed=bool(persistence["passed"]),
        finite_run_state=finite_run_state,
        relative_field_change=float(persistence["relative_field_change"]),
        final_frame_correlation=float(persistence["final_frame_correlation"]),
        k_dom_std_final_window=float(persistence["k_dom_std_final_window"]),
        shell_concentration_min_final_window=float(
            persistence["shell_concentration_min_final_window"]
        ),
        diagnostic_relative_drift=float(persistence["diagnostic_relative_drift"]),
        stationary_residual=stationary_residual,
        physical_margin=physical_margin,
        npt_margin=npt_margin,
        stability_margin=stability_margin,
        excess_margin=excess_margin,
    )


def _pass(flag):
    return "PASS" if bool(flag) else "FAIL"


def claim_label(summary: DisplaySummary) -> str:
    return {
        "theorem_level_reference": "Theorem-level reference",
        "numerical_reference_demonstration": "Numerical reference example",
        "exploratory_run": "Exploratory run",
    }.get(summary.claim_level, summary.claim_level)


def status_cards_html(summary: DisplaySummary) -> str:
    """Return the status cards shown in the research notebook."""

    def card(title, flag, main, sub, margin=None, status_label=None, neutral=False):
        status = status_label or ("PASS" if flag else "FAIL")
        color = "#345995" if neutral else ("#0b6b3a" if flag else "#9b1c1c")
        margin_line = (
            ""
            if margin is None
            else f'<div style="font-size:12px;color:#444;">margin {margin:+.4f}</div>'
        )
        return f"""<div style="border:1px solid #ddd;border-radius:10px;padding:10px;margin:6px;background:#fafafa;min-width:205px;">
        <div style="font-weight:700;font-size:14px;">{title}</div>
        <div style="display:inline-block;background:{color};color:white;border-radius:12px;padding:2px 9px;margin:6px 0;font-size:12px;font-weight:700;">{status}</div>
        <div style="font-size:13px;">{main}</div>
        <div style="font-size:12px;color:#444;">{sub}</div>{margin_line}
        </div>"""

    claim_status = {
        "theorem_level_reference": "THEOREM REFERENCE",
        "numerical_reference_demonstration": "NUMERICAL REFERENCE",
        "exploratory_run": "EXPLORATORY",
    }.get(summary.claim_level, "RESULT SCOPE")
    developing = summary.finite_run_state == "DEVELOPING"
    cards = [
        card(
            "Scope of this result",
            True,
            claim_label(summary),
            "numerical checks and file integrity are reported separately",
            status_label=claim_status,
            neutral=True,
        ),
        card(
            "Pattern selection",
            summary.spectral_selection_passed,
            f"k_dom = {summary.k_dom:.3f}",
            f"shell = {summary.shell_concentration:.3f}, width = {summary.shell_width:g}",
            summary.shell_concentration - SPECTRAL_THRESHOLDS.get(summary.case, 0.50),
        ),
        card(
            "Physical covariance",
            summary.physical_covariance,
            f"min nu_phys = {summary.min_nu_phys:.3f}",
            "threshold > 0.5",
            summary.physical_margin,
        ),
        card(
            "NPT diagnostic",
            summary.npt_diagnostic,
            f"min nu_PT = {summary.min_nupt:.3f}",
            "threshold < 0.5",
            summary.npt_margin,
        ),
        card(
            "Gaussian stability",
            summary.stable_gaussian_reduction,
            f"max g_eff/kappa_eff = {summary.max_geff_over_keff:.3f}",
            "threshold < 0.5",
            summary.stability_margin,
        ),
        card(
            "Finite-run persistence",
            summary.finite_run_state == "PASS",
            f"relative change = {summary.relative_field_change:.4g}",
            f"correlation = {summary.final_frame_correlation:.6f}; preview may still be forming",
            status_label=summary.finite_run_state,
            neutral=developing,
        ),
        card(
            "File integrity",
            True,
            "not yet checked",
            "the read-only verifier reports this separately",
            status_label="NOT RUN",
            neutral=True,
        ),
    ]
    return '<div style="display:flex;flex-wrap:wrap;gap:4px;">' + "".join(cards) + "</div>"


def _style_im_ax(ax, L):
    ax.set_xticks([0, L // 2, L])
    ax.set_yticks([0, L // 2, L])
    ax.set_xlabel(r"$x$", labelpad=5)
    ax.set_ylabel(r"$y$", labelpad=6)
    for spine in ax.spines.values():
        spine.set_linewidth(0.7)
        spine.set_color("#666666")
    ax.tick_params(width=0.7, length=3.2, color="#666666", labelsize=10)


def _plot_radial_spectrum(ax, R, k_dom, shell_width, regime=None, kind=None):
    kr, ps, _ = q.radial_spectrum(R)
    if np.max(ps) > 0:
        ps = ps / np.max(ps)
    if regime is not None:
        curve_regime = "stripe" if kind == "stripe" or regime == "stripe" else "isotropic"
        try:
            kval, alpha, band = q.linear_dispersion_curve(curve_regime)
            if np.all(np.isfinite(band)):
                ax.axvspan(band[0], band[1], alpha=0.12, label="linear unstable band")
            apos = np.maximum(alpha, 0.0)
            if np.nanmax(apos) > 0:
                ax.plot(
                    kval, apos / np.nanmax(apos), "--", lw=1.2, alpha=0.78, label=r"$\alpha_+(k)$"
                )
        except Exception:
            pass
    ax.plot(kr, ps, lw=1.8, label=r"$P/P_{\max}$")
    ax.axvline(k_dom, color="k", ls="--", lw=1.0, alpha=0.75)
    ax.axvspan(k_dom - shell_width, k_dom + shell_width, alpha=0.16, label="selected shell")
    ax.set_xlim(0, min(np.pi, max(1.2 * k_dom, 1.0)))
    ax.set_ylim(-0.03, 1.05)
    ax.set_xlabel(r"$|k|$")
    ax.set_ylabel(r"$P/P_{\max}$")
    ax.set_title("spectrum, selected shell, and linear band", fontsize=11.5, loc="left")
    ax.legend(frameon=True, fontsize=7.9, loc="upper right")
    for spine in ax.spines.values():
        spine.set_linewidth(0.7)
        spine.set_color("#666666")
    ax.tick_params(width=0.7, length=3.2, color="#666666", labelsize=9)


def _draw_status_card(
    ax,
    y,
    title,
    ok,
    line1,
    line2="",
    line3="",
    margin=None,
    height=0.145,
    status_label=None,
    neutral=False,
):
    color = "#345995" if neutral else ("#0b6b3a" if bool(ok) else "#9b1c1c")
    patch = FancyBboxPatch(
        (0.0, y - height),
        0.96,
        height,
        boxstyle="round,pad=0.012,rounding_size=0.02",
        transform=ax.transAxes,
        facecolor="#fafafa",
        edgecolor="#dddddd",
        lw=0.85,
    )
    ax.add_patch(patch)
    ax.text(
        0.035,
        y - 0.025,
        title,
        transform=ax.transAxes,
        ha="left",
        va="top",
        fontsize=9.5,
        fontweight="bold",
    )
    label = status_label or ("PASS" if ok else "FAIL")
    ax.text(
        0.72,
        y - 0.027,
        label,
        transform=ax.transAxes,
        ha="left",
        va="top",
        fontsize=8.1,
        color="white",
        fontweight="bold",
        bbox=dict(boxstyle="round,pad=0.20", fc=color, ec="none"),
    )
    ax.text(0.035, y - 0.062, line1, transform=ax.transAxes, ha="left", va="top", fontsize=8.4)
    if line2:
        ax.text(
            0.035,
            y - 0.092,
            line2,
            transform=ax.transAxes,
            ha="left",
            va="top",
            fontsize=7.95,
            color="#444444",
        )
    if line3:
        ax.text(
            0.035,
            y - 0.119,
            line3,
            transform=ax.transAxes,
            ha="left",
            va="top",
            fontsize=7.75,
            color="#444444",
        )
    if margin is not None:
        ax.text(
            0.72,
            y - 0.116,
            f"margin {margin:+.4f}",
            transform=ax.transAxes,
            ha="left",
            va="top",
            fontsize=7.4,
            color="#444444",
        )


def _draw_dashboard_cards(ax, summary: DisplaySummary):
    ax.axis("off")
    ax.text(
        0.0,
        0.99,
        claim_label(summary),
        ha="left",
        va="top",
        fontsize=11.2,
        fontweight="bold",
    )
    theorem_text = (
        "The theorem-level reference statement applies to this case."
        if summary.theorem_level_claimed
        else "This case is presented as a numerical reference example."
    )
    ax.text(
        0.0,
        0.94,
        theorem_text + " File integrity is reported by the verification tools.",
        ha="left",
        va="top",
        fontsize=8.4,
        color="#444444",
    )
    _draw_status_card(
        ax,
        0.875,
        "Turing spectral selection",
        summary.spectral_selection_passed,
        rf"$k_{{\rm dom}}={summary.k_dom:.3f}$, shell={summary.shell_concentration:.3f}",
        rf"threshold={SPECTRAL_THRESHOLDS.get(summary.case, 0.50):.2f}, width={summary.shell_width:g}",
        margin=summary.shell_concentration - SPECTRAL_THRESHOLDS.get(summary.case, 0.50),
    )
    _draw_status_card(
        ax,
        0.705,
        "Physical covariance",
        summary.physical_covariance,
        rf"$\min\nu_{{\rm phys}}={summary.min_nu_phys:.3f}$",
        "threshold > 0.5",
        margin=summary.physical_margin,
    )
    _draw_status_card(
        ax,
        0.535,
        "NPT finite-region diagnostic",
        summary.npt_diagnostic,
        rf"$\min\widetilde{{\nu}}_-={summary.min_nupt:.3f}$",
        "threshold < 0.5",
        margin=summary.npt_margin,
    )
    state = summary.finite_run_state
    _draw_status_card(
        ax,
        0.365,
        "Finite-run persistence",
        summary.persistence_passed,
        rf"$\epsilon_R={summary.relative_field_change:.4g}$",
        rf"correlation={summary.final_frame_correlation:.6f}",
        rf"$\sigma(k_{{\rm dom}})={summary.k_dom_std_final_window:.3g}$; diagnostic drift={summary.diagnostic_relative_drift:.3g}",
        height=0.155,
        status_label=state,
        neutral=(state == "DEVELOPING"),
    )
    _draw_status_card(
        ax,
        0.185,
        "Local Gaussian excess",
        summary.pattern_excess_positive,
        rf"$\max 10^3\Delta E={1e3 * summary.max_diagnostic:.1f}$",
        "over homogeneous bond-channel baseline",
        margin=summary.excess_margin,
    )


def make_case_figure(result, out_path, title=None):
    out_path = Path(out_path)
    summary = summarize_result(result)
    R = _final_field(result)
    E = result["E_final_hi"]
    L = R.shape[1]
    vmaxE = max(0.012, float(np.nanmax(E)))
    fig = plt.figure(figsize=(11.2, 6.15), constrained_layout=False)
    gs = fig.add_gridspec(
        2,
        3,
        width_ratios=[1, 1, 1.08],
        height_ratios=[1, 0.42],
        left=0.06,
        right=0.985,
        bottom=0.12,
        top=0.88,
        wspace=0.30,
        hspace=0.42,
    )
    ax1 = fig.add_subplot(gs[0, 0])
    im1 = ax1.imshow(
        R,
        origin="lower",
        extent=(0, L, 0, L),
        cmap=FIELD_CMAP,
        vmin=VMIN,
        vmax=VMAX,
        interpolation="nearest",
        aspect="equal",
    )
    _style_im_ax(ax1, L)
    ax1.set_title(r"final $R^{\rm pat}$", fontsize=13)
    c1 = fig.colorbar(im1, ax=ax1, fraction=0.046, pad=0.03)
    c1.set_ticks([VMIN, 0, VMAX])
    c1.ax.tick_params(labelsize=9)

    ax2 = fig.add_subplot(gs[0, 1])
    im2 = ax2.imshow(
        1e3 * E,
        origin="lower",
        extent=(0, L, 0, L),
        cmap=DIAGNOSTIC_CMAP,
        vmin=0.0,
        vmax=1e3 * vmaxE,
        interpolation="nearest",
        aspect="equal",
    )
    _style_im_ax(ax2, L)
    ax2.set_title(r"$10^3\Delta E_{N,G}^{\rm loc}$", fontsize=13)
    c2 = fig.colorbar(im2, ax=ax2, fraction=0.046, pad=0.03)
    c2.ax.tick_params(labelsize=9)

    ax3 = fig.add_subplot(gs[:, 2])
    ax3.text(
        0.0,
        1.06,
        title or result["title"],
        ha="left",
        va="top",
        fontsize=13.5,
        fontweight="bold",
        transform=ax3.transAxes,
    )
    _draw_dashboard_cards(ax3, summary)
    ax4 = fig.add_subplot(gs[1, 0:2])
    _plot_radial_spectrum(
        ax4,
        R,
        summary.k_dom,
        summary.shell_width,
        regime=result.get("regime"),
        kind=result.get("kind"),
    )
    fig.suptitle(
        title or f"{result['title']}: pattern and finite-region NPT diagnostic",
        fontsize=14.5,
        y=0.975,
    )
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, bbox_inches="tight")
    if out_path.suffix.lower() != ".png":
        fig.savefig(out_path.with_suffix(".png"), bbox_inches="tight")
    plt.close(fig)
    return summary


def _frame_to_rgb_array(fig):
    fig.canvas.draw()
    return np.asarray(fig.canvas.buffer_rgba())[..., :3].copy()


def _open_mp4_writer(out_path, fps):
    out_path = Path(out_path)
    if out_path.suffix.lower() != ".mp4":
        out_path = out_path.with_suffix(".mp4")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    return imageio_v2.get_writer(
        str(out_path), format="FFMPEG", fps=fps, codec="libx264", quality=8, macro_block_size=16
    )


def _excess_stage(val, final_max):
    """Return only the stage label; callers add their own descriptive prefix."""
    if final_max <= 0 or val < 0.02 * final_max:
        return "negligible"
    if val < 0.70 * final_max:
        return "developing"
    return "high"


def _turing_stage(result, R):
    try:
        final_R = result["R_frames"][-1]
        kr, ps, _ = q.radial_spectrum(final_R)
        j = int(np.argmax(ps[1:]) + 1)
        k_dom = float(kr[j])
        shell_width = q.radial_bin_width(R)
        threshold = SPECTRAL_THRESHOLDS.get(result.get("kind"), 0.50)
        shell_t = float(q.shell_concentration(R, k_dom, shell_width))
        if shell_t < 0.25:
            stage = "not yet selected"
        elif shell_t < threshold:
            stage = "developing"
        else:
            stage = "selected"
        return stage, shell_t
    except Exception:
        return "not evaluated", float("nan")


def make_case_movie(
    result,
    out_path,
    fps=8,
    diagnostic_vmax=None,
    diagnostic_scale_scope="per_case_notebook_movie",
):
    """Render a fixed-size MP4 with independent color and temporal references."""
    out_path = Path(out_path)
    times = result["E_times"]
    E_frames = result["E_frames"]
    result_times = result["times"]
    idx = [int(np.argmin(np.abs(result_times - t))) for t in times]
    R_frames = result["R_frames"][idx]
    color_vmax = max(
        float(diagnostic_vmax) if diagnostic_vmax is not None else float(np.nanmax(E_frames)), 1e-12
    )
    case_final_max = max(float(np.nanmax(E_frames[-1])), 1e-12)
    result = dict(result)
    result["movie_diagnostic_scale"] = str(diagnostic_scale_scope)
    try:
        import qtp_movie_renderer as _movie_style

        frame_func = _movie_style.movie_frame
        use_paper_renderer = True
    except Exception:
        frame_func = None
        use_paper_renderer = False
    with _open_mp4_writer(out_path, fps) as writer:
        if use_paper_renderer:
            for R, E, t in zip(R_frames, E_frames, times):
                writer.append_data(frame_func(result, R, E, float(t), color_vmax, case_final_max))
            return
        L = R_frames[0].shape[1]
        vmaxE = max(0.012, color_vmax)
        for i, t in enumerate(times):
            fig = plt.figure(figsize=(12.8, 7.2), dpi=100, constrained_layout=False)
            gs = fig.add_gridspec(
                1,
                3,
                width_ratios=[1, 1, 1.08],
                left=0.06,
                right=0.985,
                bottom=0.15,
                top=0.88,
                wspace=0.30,
            )
            ax1 = fig.add_subplot(gs[0, 0])
            im1 = ax1.imshow(
                R_frames[i],
                origin="lower",
                extent=(0, L, 0, L),
                cmap=FIELD_CMAP,
                vmin=VMIN,
                vmax=VMAX,
                interpolation="nearest",
                aspect="equal",
            )
            _style_im_ax(ax1, L)
            ax1.set_title(r"pattern heatmap $R^{\rm pat}$", fontsize=12.6)
            cb1 = fig.colorbar(im1, ax=ax1, fraction=0.046, pad=0.03)
            cb1.set_ticks([VMIN, 0, VMAX])
            cb1.ax.tick_params(labelsize=8.5)
            ax2 = fig.add_subplot(gs[0, 1])
            im2 = ax2.imshow(
                1e3 * E_frames[i],
                origin="lower",
                extent=(0, L, 0, L),
                cmap=DIAGNOSTIC_CMAP,
                vmin=0.0,
                vmax=1e3 * vmaxE,
                interpolation="bicubic",
                aspect="equal",
            )
            _style_im_ax(ax2, L)
            ax2.set_title(r"diagnostic map $10^3\Delta E_{N,G}^{\rm loc}$", fontsize=12.6)
            cb2 = fig.colorbar(im2, ax=ax2, fraction=0.046, pad=0.03)
            cb2.ax.tick_params(labelsize=8.5)
            ax3 = fig.add_subplot(gs[0, 2])
            ax3.axis("off")
            phys_ok = result["min_nu_phys"][i] > 0.5
            npt_ok = result["min_nupt"][i] < 0.5
            stage = _excess_stage(float(np.nanmax(E_frames[i])), case_final_max)
            tstage, shell_t = _turing_stage(result, R_frames[i])
            ax3.text(
                0.0, 0.99, result["title"], ha="left", va="top", fontsize=14, fontweight="bold"
            )
            ax3.text(0.0, 0.91, rf"$t={t:.1f}$", ha="left", va="top", fontsize=13)
            lines = [
                "Gaussian bond channel",
                f"  physical covariance: {_pass(phys_ok)}  min nu_phys = {result['min_nu_phys'][i]:.3f}",
                f"  NPT condition: {_pass(npt_ok)}  min nu_PT = {result['min_nupt'][i]:.3f}",
                "",
                "Baseline-subtracted local Gaussian excess",
                f"  mean Delta E = {result['mean_E'][i]:.4f}",
                f"  max  Delta E = {float(np.nanmax(E_frames[i])):.4f}",
                f"  {stage}",
                f"  Turing spectral selection: {tstage} (shell={shell_t:.3f})",
            ]
            ax3.text(
                0.0,
                0.82,
                "\n".join(lines),
                ha="left",
                va="top",
                fontsize=10.6,
                bbox=dict(boxstyle="round,pad=0.45", facecolor="#f8f8f8", edgecolor="#cccccc"),
            )
            fig.suptitle(
                f"{result['title']}: synchronized pattern and finite-region diagnostic",
                fontsize=14.2,
                y=0.975,
            )
            writer.append_data(_frame_to_rgb_array(fig))
            plt.close(fig)


def make_exhibit_movie(result, out_path, fps=8, pattern_vmax=VMAX, diagnostic_vmax=None):
    """Render a clean precomputed museum movie without research metadata."""
    out_path = Path(out_path)
    times = np.asarray(result["E_times"], dtype=float)
    result_times = np.asarray(result["times"], dtype=float)
    idx = [int(np.argmin(np.abs(result_times - t))) for t in times]
    R_frames = np.asarray(result["R_frames"])[idx]
    E_frames = np.asarray(result["E_frames"])
    raw_shape = tuple(np.asarray(R_frames[0]).shape[-2:])
    diagnostic_shape = tuple(np.asarray(E_frames[0]).shape[-2:])
    if diagnostic_shape != raw_shape:
        raise ValueError(
            "Exhibit movies require native-grid entanglement diagnostics: "
            f"pattern grid {raw_shape}, diagnostic grid {diagnostic_shape}. "
            "Regenerate the reference data with diagnostic stride=1."
        )
    diagnostic_vmax = max(
        float(diagnostic_vmax) if diagnostic_vmax is not None else float(np.nanmax(E_frames)),
        1e-12,
    )
    import qtp_movie_renderer as _movie_style

    with _open_mp4_writer(out_path, fps) as writer:
        for R, E, t in zip(R_frames, E_frames, times):
            writer.append_data(
                _movie_style.exhibit_frame(
                    result,
                    R,
                    E,
                    float(t),
                    pattern_color_vmax=float(pattern_vmax),
                    diagnostic_color_vmax=diagnostic_vmax,
                )
            )


def reference_diagnostic_vmax(root=None, fallback=0.025):
    """Return the diagnostic scale used by the bundled reference results."""
    root = Path(root) if root is not None else Path(__file__).resolve().parent
    values = []
    for path in (root / "display").glob("*_strong_qtp_summary.json"):
        try:
            plan = json.loads(path.read_text()).get("movie_plan", {})
            value = plan.get("diagnostic_vmax_raw")
            if value is not None and np.isfinite(float(value)) and float(value) > 0:
                values.append(float(value))
        except Exception:
            continue
    return max(values) if values else float(fallback)


def _array_rgb(array, cmap_name, vmin, vmax):
    cmap = plt.get_cmap(cmap_name)
    scale = max(float(vmax) - float(vmin), 1e-15)
    normalized = np.clip((np.asarray(array, dtype=float) - float(vmin)) / scale, 0.0, 1.0)
    return (255 * cmap(normalized)[..., :3]).astype(np.uint8)


def client_side_time_explorer_html(result, diagnostic_vmax=None, initial="final"):
    """Build the browser-based time explorer from precomputed frames.

    Playback, pause, seeking, speed selection, and looping remain responsive
    throughout an exhibit session.
    """
    times = np.asarray(result["E_times"], dtype=float)
    result_times = np.asarray(result["times"], dtype=float)
    idx = [int(np.argmin(np.abs(result_times - t))) for t in times]
    R_frames = np.asarray(result["R_frames"])[idx]
    E_frames = np.asarray(result["E_frames"])
    color_vmax = float(
        diagnostic_vmax if diagnostic_vmax is not None else reference_diagnostic_vmax()
    )
    color_vmax = max(color_vmax, 1e-12)
    case_final_max = max(float(np.nanmax(E_frames[-1])), 1e-12)
    encoded = []
    stages = []
    for R, E, t in zip(R_frames, E_frames, times):
        left = Image.fromarray(_array_rgb(R, FIELD_CMAP, VMIN, VMAX)).resize(
            (360, 360), Image.Resampling.BICUBIC
        )
        right = Image.fromarray(_array_rgb(E, DIAGNOSTIC_CMAP, 0.0, color_vmax)).resize(
            (360, 360), Image.Resampling.BICUBIC
        )
        canvas = Image.new("RGB", (752, 400), "white")
        canvas.paste(left, (8, 34))
        canvas.paste(right, (384, 34))
        draw = ImageDraw.Draw(canvas)
        draw.text((8, 8), "Pattern field", fill="black", font=_times_pil_font(18))
        draw.text((384, 8), "Entanglement diagnostic", fill="black", font=_times_pil_font(18))
        buffer = io.BytesIO()
        canvas.save(buffer, format="JPEG", quality=84, optimize=True)
        encoded.append(
            "data:image/jpeg;base64," + base64.b64encode(buffer.getvalue()).decode("ascii")
        )
        excess = _excess_stage(float(np.nanmax(E)), case_final_max)
        turing, shell = _turing_stage(result, R)
        stages.append({"t": float(t), "excess": excess, "turing": turing, "shell": float(shell)})
    uid = "qtp_" + uuid.uuid4().hex
    start = len(encoded) - 1 if initial == "final" else 0
    frames_json = json.dumps(encoded)
    stages_json = json.dumps(stages)
    alt = f"{result.get('title', result.get('kind', 'QTP'))}: synchronized pattern field and finite-region diagnostic"
    html_text = f'''<div id="{uid}" style="max-width:780px;border:1px solid #ddd;border-radius:12px;padding:10px;background:#fff;">
      <div style="display:flex;gap:8px;align-items:center;flex-wrap:wrap;margin-bottom:8px;">
        <button data-action="replay">Replay formation</button>
        <button data-action="pause">Pause</button>
        <label>Evolution <input data-role="slider" type="range" min="0" max="{len(encoded) - 1}" value="{start}" style="width:330px;"></label>
        <label>Speed <select data-role="speed"><option value="350">0.5x</option><option value="180" selected>1x</option><option value="90">2x</option></select></label>
        <label><input data-role="loop" type="checkbox" checked> loop</label>
      </div>
      <img data-role="image" src="{encoded[start]}" alt="{alt}" style="width:100%;height:auto;border-radius:8px;display:block;">
      <div data-role="status" aria-live="polite" style="margin-top:7px;font:14px/1.35 Times New Roman,Times,serif;"></div>
      <div style="font:12px/1.35 Times New Roman,Times,serif;color:#555;margin-top:4px;">Fixed diagnostic color scale: {color_vmax:.5g}. Playback is browser-side; changing time does not rerun the PDE.</div>
    </div>
    <script>(function(){{
      const root=document.getElementById('{uid}'); const frames={frames_json}; const status={stages_json};
      const img=root.querySelector('[data-role=image]'), slider=root.querySelector('[data-role=slider]');
      const label=root.querySelector('[data-role=status]'), speed=root.querySelector('[data-role=speed]'), loop=root.querySelector('[data-role=loop]');
      let timer=null;
      function show(i){{ i=Math.max(0,Math.min(frames.length-1,Number(i))); slider.value=i; img.src=frames[i];
        const s=status[i]; label.textContent=`t = ${{s.t.toFixed(1)}} | local Gaussian excess: ${{s.excess}} | Turing spectral selection: ${{s.turing}} (shell=${{s.shell.toFixed(3)}})`; }}
      function pause(){{ if(timer!==null){{clearInterval(timer); timer=null;}} }}
      function play(){{ pause(); timer=setInterval(()=>{{ let i=Number(slider.value)+1; if(i>=frames.length){{ if(loop.checked)i=0; else{{pause();return;}} }} show(i); }},Number(speed.value)); }}
      root.querySelector('[data-action=replay]').onclick=()=>{{show(0);play();}};
      root.querySelector('[data-action=pause]').onclick=pause;
      slider.oninput=()=>{{pause();show(slider.value);}};
      speed.onchange=()=>{{if(timer!==null)play();}};
      show({start});
    }})();</script>'''
    return html_text


def effective_parameters(result):
    """Return the parameters that affect the displayed run.

    Controls that are not used by the selected initial condition are omitted
    from the saved run details.
    """
    params = dict(result.get("params", {}))
    kind = result.get("kind")
    common = [
        "L",
        "dt",
        "T",
        "frame_dt",
        "Omega",
        "Dq",
        "Dp",
        "nu",
        "pattern_covector",
        "fastest_linear_mode_k",
        "fastest_linear_right_eigenvector",
    ]
    if kind == "spot":
        keys = common + [
            "beta",
            "seed",
            "bump_amp",
            "bump_width",
            "n_bumps",
            "p_to_q_ratio",
            "additive_noise",
        ]
    elif kind == "labyrinth":
        keys = common + ["beta", "seed", "noise_scale"]
    elif kind == "stripe":
        keys = [
            "L",
            "dt",
            "T",
            "frame_dt",
            "Omega",
            "Dq",
            "Dp",
            "Dy",
            "nu",
            "lam",
            "rng_seed",
            "random_perturbation_amplitude",
            "cosine_amplitude",
        ]
    else:
        keys = list(params.keys())
    out = {}
    for k in keys:
        if k in params:
            v = params[k]
            try:
                if isinstance(v, (np.floating, float)):
                    v = float(v)
                elif isinstance(v, (np.integer, int)):
                    v = int(v)
            except Exception:
                pass
            out[k] = v
    return out


def summary_to_provenance(summary: DisplaySummary, result=None):
    params = effective_parameters(result) if result is not None else {}
    raw_L = int(result["R_frames"][-1].shape[1]) if result is not None else None
    diagnostic_final_L = int(result["E_final_hi"].shape[1]) if result is not None else None
    diagnostic_movie_L = int(result["E_frames"][0].shape[1]) if result is not None else None
    movie_stride = (raw_L // diagnostic_movie_L) if raw_L and diagnostic_movie_L else None
    frame_times = [float(x) for x in result.get("E_times", [])] if result is not None else []
    try:
        base_EN, base_nupt, base_phys = q.local_physical_gaussian_covariance(
            0, 0, 0, 0, result["regime"]
        )[1:]
    except Exception:
        base_EN = base_nupt = base_phys = None
    persistence = dict(result.get("persistence_diagnostics", {})) if result is not None else {}
    finite_checks = dict(result.get("finite_checks", {})) if result is not None else {}
    initial_condition = dict(result.get("initial_condition", {})) if result is not None else {}
    storage_precision = (
        dict(
            result.get(
                "storage_precision", {"time_series": "float32", "final_diagnostic": "float64"}
            )
        )
        if result is not None
        else {}
    )
    notebook_execution = dict(result.get("notebook_execution", {})) if result is not None else {}
    verification_state = {
        "status": "not_run",
        "checked_at": None,
        "report": None,
        "file_integrity": None,
        "scientific_diagnostics": summary.diagnostics_passed,
    }
    return {
        "schema_version": SCHEMA_VERSION,
        "case": summary.case,
        "mode": summary.mode,
        "run_scope": summary.run_scope,
        "claim_level": summary.claim_level,
        "release_version": RELEASE_VERSION,
        "parameters": params,
        "initial_condition": initial_condition,
        "notebook_execution": notebook_execution,
        "storage_precision": storage_precision,
        "grid": {
            "raw_L": raw_L,
            "final_diagnostic_L": diagnostic_final_L,
            "movie_diagnostic_L": diagnostic_movie_L,
            "movie_diagnostic_extent": f"0..{raw_L}" if raw_L is not None else None,
            "final_diagnostic_stride": 1,
            "movie_diagnostic_stride": movie_stride,
            "movie_diagnostic_display_resampling": f"bicubic display-only; diagnostics use raw stride-{movie_stride} map"
            if movie_stride
            else "bicubic display-only; diagnostics use raw arrays",
        },
        "movie_plan": {
            "fps": 8,
            "n_frames_expected": len(frame_times),
            "expected_final_time": frame_times[-1] if frame_times else None,
            "frame_times": frame_times,
            "diagnostic_scale_scope": "not_attached_until_movie_rendered",
            "diagnostic_vmax_raw": None,
            "case_final_diagnostic_max_raw": float(np.nanmax(result["E_frames"][-1]))
            if result is not None
            else None,
            "diagnostic_units": "Delta E_N,G^loc",
            "display_units": "10^3 Delta E_N,G^loc",
        },
        "computed_quantities": {
            "k_dom": summary.k_dom,
            "k_dom_definition": "maximum nonzero canonical radial-mean-power bin; exact ties use the smaller radius",
            "shell_concentration": summary.shell_concentration,
            "shell_width": summary.shell_width,
            "spectrum_normalization": "unitary mean-subtracted FFT; P_shell / total_nonzero_power",
            "mean_delta_E": summary.mean_diagnostic,
            "max_delta_E": summary.max_diagnostic,
            "min_nu_pt": summary.min_nupt,
            "min_nu_phys": summary.min_nu_phys,
            "max_g_eff_over_kappa_eff": summary.max_geff_over_keff,
            "stationary_residual": summary.stationary_residual,
            "stationary_residual_definition": "||F(q_final,p_final)||_2 / ||(q_final,p_final)||_2",
            "finite_run_state": summary.finite_run_state,
        },
        "numerical_observable_contract": dict(observables.OBSERVABLE_CONTRACT),
        "persistence_diagnostics": persistence,
        "finite_value_checks": finite_checks,
        "baseline": {
            "E_N_homogeneous": base_EN,
            "nu_pt_homogeneous": base_nupt,
            "nu_phys_homogeneous": base_phys,
        },
        "verification_tolerances": {
            "excess_tol": EXCESS_TOL,
            "symplectic_threshold": SYMPLECTIC_THRESHOLD,
            "stability_threshold": STABILITY_THRESHOLD,
            "spectral_concentration_threshold": SPECTRAL_THRESHOLDS.get(summary.case, 0.50),
            "spectral_threshold_type": "display diagnostic, not theorem constant",
            "reference_parameter_comparison": {"rtol": 0.0, "atol": 1e-12},
        },
        "verification_margins": {
            "physical_margin": summary.physical_margin,
            "npt_margin": summary.npt_margin,
            "stability_margin_to_0p5": summary.stability_margin,
            "excess_margin": summary.excess_margin,
            "spectral_concentration_margin": summary.shell_concentration
            - SPECTRAL_THRESHOLDS.get(summary.case, 0.50),
        },
        "certification_scope": {
            "mode": summary.mode,
            "run_scope": summary.run_scope,
            "claim_level": summary.claim_level,
            "reference_configuration_exact": summary.reference_configuration_exact,
            "diagnostics_passed": summary.diagnostics_passed,
            "finite_run_state": summary.finite_run_state,
            "theorem_level_claimed": summary.theorem_level_claimed,
            "current_output_verified": False,
            "current_output_verification": verification_state,
            "bundled_reference_verified": False,
            "bundled_reference_verification": {
                "status": "not_run",
                "checked_at": None,
                "report": None,
            },
            "certified_strong_qtp_claimed": summary.certified_strong_qtp_claimed,
            "note": "The theorem-level designation applies only to the exact Stripe reference parameters. Spot and Labyrinth are numerical reference examples. Preview or modified runs are exploratory; a developing preview has not yet reached the persistence threshold.",
        },
        "verification": {
            "no_nan": summary.no_nan,
            "physical_covariance": summary.physical_covariance,
            "npt_diagnostic": summary.npt_diagnostic,
            "stable_gaussian_reduction": summary.stable_gaussian_reduction,
            "pattern_excess_positive": summary.pattern_excess_positive,
            "spectral_selection_passed": summary.spectral_selection_passed,
            "persistence_passed": summary.persistence_passed,
            "finite_run_state": summary.finite_run_state,
            "strong_qtp": summary.strong_qtp,
            "diagnostics_passed": summary.diagnostics_passed,
            "theorem_level_claimed": summary.theorem_level_claimed,
            "certified_strong_qtp_claimed": summary.certified_strong_qtp_claimed,
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

    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _movie_metadata(path: Path) -> dict[str, Any]:
    """Read MP4 metadata used in the integrity manifests."""
    path = Path(path)
    reader = imageio_v2.get_reader(str(path), "ffmpeg")
    try:
        try:
            n_frames = int(reader.count_frames())
        except Exception:
            meta0 = reader.get_meta_data()
            n_frames = int(round(float(meta0.get("duration", 0.0)) * float(meta0.get("fps", 0.0))))
        meta = reader.get_meta_data()
        width, height = tuple(meta.get("size", (None, None)))
        fps = float(meta.get("fps", 0.0))
    finally:
        reader.close()
    return {"n_frames": n_frames, "fps": fps, "width": width, "height": height}


def write_exhibit_manifest(
    results,
    display_dir,
    *,
    fps=8,
    pattern_vmin=VMIN,
    pattern_vmax=VMAX,
    diagnostic_vmin=0.0,
    diagnostic_vmax=None,
    filename="display_exhibit_manifest.csv",
):
    """Write the scientific and video metadata for the exhibit movies."""
    import qtp_movie_renderer as movie_style

    display_dir = Path(display_dir)
    display_dir.mkdir(parents=True, exist_ok=True)
    if diagnostic_vmax is None:
        diagnostic_vmax = max(
            float(np.nanmax(np.asarray(result["E_frames"]))) for result in results
        )
    rows = []
    for result in results:
        case = str(result["kind"])
        path = display_dir / f"display_{case}_exhibit.mp4"
        if not path.exists() or path.stat().st_size <= 0:
            raise FileNotFoundError(f"missing exhibit movie: {path}")
        meta = _movie_metadata(path)
        times = np.asarray(result["E_times"], dtype=float)
        rows.append(
            {
                "case": case,
                "filename": path.name,
                "sha256": _sha256_file(path),
                "size_bytes": path.stat().st_size,
                "n_frames": meta["n_frames"],
                "fps": meta["fps"],
                "width": meta["width"],
                "height": meta["height"],
                "physical_final_time": float(times[-1]),
                "field_vmin": float(pattern_vmin),
                "field_vmax": float(pattern_vmax),
                "diagnostic_vmin": float(diagnostic_vmin),
                "diagnostic_vmax": float(diagnostic_vmax),
                "pattern_grid_L": int(np.asarray(result["R_frames"][0]).shape[-1]),
                "diagnostic_grid_L": int(np.asarray(result["E_frames"][0]).shape[-1]),
                "diagnostic_spatial_stride": int(
                    np.asarray(result["R_frames"][0]).shape[-1]
                    // np.asarray(result["E_frames"][0]).shape[-1]
                ),
                "inter_panel_extra_gap_px": 40,
                "y_label_x_offset_px": 76,
                "visitor_right_panel_title": "Entanglement diagnostic",
                "font_family": "Times",
                "axis_label_font_size": movie_style.EXHIBIT_AXIS_LABEL_FONT_SIZE,
                "axis_label_scale_reference": movie_style.EXHIBIT_AXIS_LABEL_SCALE_REFERENCE,
            }
        )
    out = display_dir / filename
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
        "diagnostic_vmin",
        "diagnostic_vmax",
        "pattern_grid_L",
        "diagnostic_grid_L",
        "diagnostic_spatial_stride",
        "inter_panel_extra_gap_px",
        "y_label_x_offset_px",
        "visitor_right_panel_title",
        "font_family",
        "axis_label_font_size",
        "axis_label_scale_reference",
    ]
    with out.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    return out


def attach_movie_metadata(
    summary_path,
    movie_path,
    diagnostic_vmax=None,
    diagnostic_scale_scope=None,
    diagnostic_units="Delta E_N,G^loc",
    display_units="10^3 Delta E_N,G^loc",
):
    summary_path = Path(summary_path)
    movie_path = Path(movie_path)
    if not summary_path.exists() or not movie_path.exists():
        return
    data = json.loads(summary_path.read_text())
    data["movie_file"] = {
        "path": movie_path.name,
        "sha256": _sha256_file(movie_path),
        "size_bytes": movie_path.stat().st_size,
    }
    movie_plan = data.setdefault("movie_plan", {})
    scope = diagnostic_scale_scope or (
        "bundled_reference_movies" if diagnostic_vmax is not None else "per_case_notebook_movie"
    )
    vmax_value = float(diagnostic_vmax) if diagnostic_vmax is not None else None
    movie_plan["diagnostic_vmax_raw"] = vmax_value
    movie_plan["diagnostic_scale_scope"] = scope
    movie_plan["diagnostic_units"] = diagnostic_units
    movie_plan["display_units"] = display_units
    summary_path.write_text(json.dumps(data, indent=2))


def write_run_manifest(run_dir, files, manifest_name="run_manifest.csv"):
    """Write size and SHA-256 records for an exported run."""
    run_dir = Path(run_dir)
    rows = []
    for file_path in files:
        p = Path(file_path)
        if not p.exists():
            continue
        rows.append(
            {
                "file": p.name,
                "relative_path": str(p.relative_to(run_dir))
                if p.is_relative_to(run_dir)
                else str(p),
                "size_bytes": p.stat().st_size,
                "sha256": _sha256_file(p),
            }
        )
    if not rows:
        return None
    out = run_dir / manifest_name
    with out.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["file", "relative_path", "size_bytes", "sha256"])
        writer.writeheader()
        writer.writerows(rows)
    return out


def save_summary(summary: DisplaySummary, out_path, result=None):
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(summary_to_provenance(summary, result=result), indent=2))


