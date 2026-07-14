"""Numerical kernels for the three display reference cases."""

from __future__ import annotations

import matplotlib
import numpy as np

matplotlib.use("Agg")
from matplotlib import pyplot as plt

import qtp_observables as observables

# -----------------------------------------------------------------------------
# Style and constants
# -----------------------------------------------------------------------------
CMAP_FIELD = "RdYlBu_r"
CMAP_ENT = "viridis"
VMIN, VMAX = -0.24, 0.24
ISOTROPIC_PATTERN_BETA = 0.40
ISOTROPIC_FASTEST_K = 0.6491851977897728
ISOTROPIC_FASTEST_RIGHT_P_OVER_Q = -0.3651541709271028
plt.rcParams.update(
    {
        "figure.dpi": 180,
        "savefig.dpi": 300,
        "font.family": "serif",
        "font.serif": [
            "TeX Gyre Termes",
            "Nimbus Roman",
            "Times New Roman",
            "Times",
            "STIXGeneral",
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


# -----------------------------------------------------------------------------
# PDE utilities
# -----------------------------------------------------------------------------
def ell(k):
    return 2.0 * (1.0 - np.cos(k))


def semiimplicit_step(q, p, dt, A11, A12, A21, A22, det, nu):
    r2 = q * q + p * p
    bq = np.fft.fft2(q + dt * (-nu * r2 * q))
    bp = np.fft.fft2(p + dt * (-nu * r2 * p))
    qh = (A22 * bq - A12 * bp) / det
    ph = (-A21 * bq + A11 * bp) / det
    return np.fft.ifft2(qh).real, np.fft.ifft2(ph).real


def radial_spectrum(R):
    """Canonical radial spectrum with bin width 2*pi/L."""
    return observables.radial_power_spectrum(np.asarray(R, dtype=float))


def radial_bin_width(R):
    """Canonical radial-bin width 2*pi/L."""
    return observables.radial_bin_width(np.asarray(R, dtype=float))


def shell_concentration(R, k0, width=None):
    """Power concentration in a shell of one-Fourier-spacing half-width by default."""
    return observables.shell_concentration(np.asarray(R, dtype=float), k0, width)


def dominant_radial_wavenumber(R):
    """Dominant nonzero radial Fourier-bin center, with ties to smaller radius."""
    return observables.dominant_radial_wavenumber(np.asarray(R, dtype=float))


def canonical_spectral_observables(R):
    """Return the Fourier-observable definitions for a final field."""
    return observables.radial_shell_observables(np.asarray(R, dtype=float))


def _laplacian_axis(field, axis):
    field = np.asarray(field, dtype=float)
    return np.roll(field, -1, axis=axis) + np.roll(field, 1, axis=axis) - 2.0 * field


def stripe_vector_field(q, p, *, lam=0.4, nu=4.0, Dy=0.2):
    """Deterministic stripe reaction--transport vector field."""
    q = np.asarray(q, dtype=float)
    p = np.asarray(p, dtype=float)
    sqrt3 = np.sqrt(3.0)
    Dq, Dp = 1.0, 3.0 + 2.0 * sqrt3
    Omega = np.sqrt(2.0 * sqrt3 - float(lam))
    radius2 = q * q + p * p
    fq = (
        q
        + Omega * p
        - float(nu) * radius2 * q
        + Dq * _laplacian_axis(q, 1)
        + float(Dy) * _laplacian_axis(q, 0)
    )
    fp = (
        -Omega * q
        - 3.0 * p
        - float(nu) * radius2 * p
        + Dp * _laplacian_axis(p, 1)
        + float(Dy) * _laplacian_axis(p, 0)
    )
    return fq, fp


def isotropic_vector_field(q, p, *, Omega=1.8, Dq=0.6, Dp=4.5, nu=4.0):
    """Deterministic isotropic reaction--transport vector field."""
    q = np.asarray(q, dtype=float)
    p = np.asarray(p, dtype=float)
    lap_q = _laplacian_axis(q, 0) + _laplacian_axis(q, 1)
    lap_p = _laplacian_axis(p, 0) + _laplacian_axis(p, 1)
    radius2 = q * q + p * p
    fq = q + float(Omega) * p - float(nu) * radius2 * q + float(Dq) * lap_q
    fp = -float(Omega) * q - 3.0 * p - float(nu) * radius2 * p + float(Dp) * lap_p
    return fq, fp


def relative_stationary_residual(q, p, *, regime="stripe", **params):
    """Return ||F(q,p)||_2 / max(||(q,p)||_2, eps) for the final field."""
    if regime == "stripe":
        fq, fp = stripe_vector_field(
            q, p, **{k: params[k] for k in ("lam", "nu", "Dy") if k in params}
        )
    elif regime == "isotropic":
        fq, fp = isotropic_vector_field(
            q, p, **{k: params[k] for k in ("Omega", "Dq", "Dp", "nu") if k in params}
        )
    else:
        raise ValueError(f"unknown regime={regime!r}")
    numerator = np.sqrt(np.sum(fq * fq) + np.sum(fp * fp))
    denominator = np.sqrt(np.sum(np.asarray(q, float) ** 2) + np.sum(np.asarray(p, float) ** 2))
    return float(numerator / max(float(denominator), np.finfo(float).eps))


# -----------------------------------------------------------------------------
# Simulations
# -----------------------------------------------------------------------------
def simulate_stripe(
    L=192,
    dt=0.05,
    T=160.0,
    frame_dt=2.0,
    rng_seed=7,
    random_perturbation_amplitude=1e-3,
    cosine_amplitude=1e-3,
    progress_callback=None,
):
    """Simulate the commensurate stripe reference branch.

    ``cosine_amplitude`` controls the weak analytic-branch seed.  Setting it
    to zero gives the broadband-noise control used to test wavelength selection
    of the commensurate critical wave number.  The time-series frames are
    stored as float32; the final field is retained separately in float64 for
    diagnostics and convergence controls.
    """
    L = int(L)
    dt = float(dt)
    T = float(T)
    frame_dt = float(frame_dt)
    if L % 12 != 0:
        raise ValueError(
            "The analytic stripe reference requires a period-compatible grid with L divisible by 12."
        )
    a, d = 1.0, -3.0
    sqrt3 = np.sqrt(3.0)
    Dq, Dp, Dy = 1.0, 3.0 + 2 * sqrt3, 0.2
    nu, lam = 4.0, 0.4
    Omega0 = np.sqrt(2 * sqrt3)
    Omega = np.sqrt(2 * sqrt3 - lam)
    beta = (sqrt3 - 1.0) / Omega0
    kstar = np.pi / 6
    rng_seed = int(rng_seed)
    random_perturbation_amplitude = float(random_perturbation_amplitude)
    cosine_amplitude = float(cosine_amplitude)
    phase_convention = (
        "cos(k_* x) analytic-branch seed plus broadband noise"
        if cosine_amplitude != 0.0
        else "broadband noise only; no injected k_* Fourier component"
    )
    rng = np.random.default_rng(rng_seed)
    x = np.arange(L)
    X = np.tile(x, (L, 1))
    qv = random_perturbation_amplitude * rng.standard_normal((L, L)) + cosine_amplitude * np.cos(
        kstar * X
    )
    pv = random_perturbation_amplitude * rng.standard_normal(
        (L, L)
    ) - beta * cosine_amplitude * np.cos(kstar * X)
    k = 2 * np.pi * np.fft.fftfreq(L)
    KX, KY = np.meshgrid(k, k)
    L11 = a - Dq * ell(KX) - Dy * ell(KY)
    L22 = d - Dp * ell(KX) - Dy * ell(KY)
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
    amp = []
    contrast = []
    phase = np.exp(-1j * kstar * x)[None, :]
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
            amp.append(abs(np.sum(R * phase)) / R.size)
            contrast.append(R.std())
        if n == nsteps:
            break
        qv, pv = semiimplicit_step(qv, pv, dt, A11, A12, A21, A22, det, nu)
    q_final_float64 = np.asarray(qv, dtype=np.float64).copy()
    p_final_float64 = np.asarray(pv, dtype=np.float64).copy()
    R_final_float64 = q_final_float64 - beta * p_final_float64
    frames_q = np.asarray(frames_q, dtype=np.float32)
    frames_p = np.asarray(frames_p, dtype=np.float32)
    R_frames = frames_q - beta * frames_p
    initial_condition = {
        "type": "commensurate_cosine_seed_plus_noise"
        if cosine_amplitude != 0.0
        else "broadband_noise_only",
        "rng": "numpy.PCG64",
        "seed": rng_seed,
        "random_perturbation_amplitude": random_perturbation_amplitude,
        "cosine_amplitude": cosine_amplitude,
        "k_star": float(kstar),
        "beta": float(beta),
        "phase_convention": phase_convention,
    }
    storage_precision = {"time_series": "float32", "final_diagnostic": "float64"}
    return dict(
        kind="stripe",
        title="Stripe",
        regime="stripe",
        times=np.array(times),
        q_frames=frames_q,
        p_frames=frames_p,
        R_frames=R_frames,
        q_final_float64=q_final_float64,
        p_final_float64=p_final_float64,
        R_final_float64=R_final_float64,
        amp=np.array(amp),
        contrast=np.array(contrast),
        kstar=kstar,
        kpeak=kstar,
        dominant_k_measured=dominant_radial_wavenumber(R_final_float64),
        spectral_concentration=shell_concentration(R_final_float64, kstar),
        beta=beta,
        stationary_residual=relative_stationary_residual(
            q_final_float64, p_final_float64, regime="stripe", lam=lam, nu=nu, Dy=Dy
        ),
        initial_condition=initial_condition,
        storage_precision=storage_precision,
        params=dict(
            L=L,
            dt=dt,
            T=T,
            frame_dt=frame_dt,
            Omega=Omega,
            Dq=Dq,
            Dp=Dp,
            Dy=Dy,
            nu=nu,
            lam=lam,
            rng_seed=rng_seed,
            random_perturbation_amplitude=random_perturbation_amplitude,
            cosine_amplitude=cosine_amplitude,
        ),
    )


def init_spot(L, amp=0.12, nbumps=32, width=2.4, seed=4, p_to_q_ratio=-0.25, additive_noise=1e-3):
    """Localized multi-bump initial data with fully specified controls."""
    rng = np.random.default_rng(int(seed))
    Y, X = np.mgrid[0:L, 0:L]
    q = np.zeros((L, L))
    p = np.zeros((L, L))
    for _ in range(int(nbumps)):
        cy = rng.integers(0, L)
        cx = rng.integers(0, L)
        rr = ((X - cx + L / 2) % L - L / 2) ** 2 + ((Y - cy + L / 2) % L - L / 2) ** 2
        bump = np.exp(-rr / (2 * float(width) ** 2))
        q += float(amp) * bump
        p += float(p_to_q_ratio) * float(amp) * bump
    q += float(additive_noise) * rng.standard_normal((L, L))
    p += float(additive_noise) * rng.standard_normal((L, L))
    return q, p


def init_noise(L, seed=0, scale=1e-2):
    rng = np.random.default_rng(seed)
    return scale * rng.standard_normal((L, L)), scale * rng.standard_normal((L, L))


# -----------------------------------------------------------------------------
# Turing certificate helpers
# -----------------------------------------------------------------------------
def eigen_max(lam, k1, k2=0.0, Dy=0.2):
    sqrt3 = np.sqrt(3.0)
    Dp = 3 + 2 * sqrt3
    Dq = 1.0
    Om = np.sqrt(2 * sqrt3 - lam)
    l1 = ell(k1)
    l2 = ell(k2)
    M = np.array([[1 - Dq * l1 - Dy * l2, Om], [-Om, -3 - Dp * l1 - Dy * l2]])
    return np.max(np.linalg.eigvals(M).real), np.linalg.det(M)


def linear_dispersion_curve(regime="stripe", lam=0.4, n=600):
    """One-dimensional linear dispersion used as a plotting certificate.

    For the anisotropic stripe construction this is the longitudinal k_y=0 cut.
    For the isotropic spot/labyrinth examples it is the axial cut through the
    isotropic shell; the 2D simulation spectra are radialized separately.
    """
    kval = np.linspace(0.0, np.pi, n)
    if regime == "stripe":
        alpha = np.array([eigen_max(lam, k, 0.0)[0] for k in kval], dtype=float)
    else:
        Omega = 1.8
        Dq = 0.6
        Dp = 4.5
        alpha = []
        for k in kval:
            ell_value = ell(k)
            M = np.array(
                [
                    [1.0 - Dq * ell_value, Omega],
                    [-Omega, -3.0 - Dp * ell_value],
                ],
                dtype=float,
            )
            alpha.append(np.max(np.linalg.eigvals(M).real))
        alpha = np.array(alpha, dtype=float)
    unstable = kval[alpha > 0]
    band = (float(unstable.min()), float(unstable.max())) if unstable.size else (np.nan, np.nan)
    return kval, alpha, band


# -----------------------------------------------------------------------------
# Physical Gaussian fluctuation covariance from a local linear Lindblad reduction
# -----------------------------------------------------------------------------
def closed_form_bond_covariance(k_eff, g_eff, Omega):
    """Closed-form covariance for the auxiliary two-mode Gaussian bond model.

    Ordering is (q1,p1,q2,p2).  The formula is valid in the Hurwitz range
    k_eff^2 + 4 Omega^2 - 4 g_eff^2 > 0, satisfied by the reported branches.
    """
    k_eff = float(k_eff)
    g_eff = float(g_eff)
    Omega = float(Omega)
    S = k_eff * k_eff + 4.0 * Omega * Omega
    Delta = S - 4.0 * g_eff * g_eff
    if Delta <= 0:
        raise ValueError(
            f"Gaussian bond model is outside the stable closed-form range: Delta={Delta}"
        )
    a = S / (2.0 * Delta)
    c = g_eff * k_eff / Delta
    d = -2.0 * Omega * g_eff / Delta
    V = np.array([[a, 0.0, c, d], [0.0, a, d, -c], [c, d, a, 0.0], [d, -c, 0.0, a]], dtype=float)
    nu_phys = 0.5 * np.sqrt(S / Delta)
    nu_pt = np.sqrt(S) / (2.0 * (np.sqrt(S) + 2.0 * g_eff))
    EN = max(0.0, -np.log2(2.0 * nu_pt))
    return V, EN, float(nu_pt), float(nu_phys)


def local_physical_gaussian_covariance(q1, p1, q2, p2, regime="stripe"):
    """Two-site covariance from the displayed auxiliary Gaussian bond model.

    The branch changes the effective damping and squeezing through the local
    two-photon-loss linearization.  The covariance is evaluated by the analytic
    Lyapunov solution used for the reported reference cases; no fitted activation or
    numerical Lyapunov solve is needed for this two-mode reduction.
    """
    sqrt3 = np.sqrt(3.0)
    if regime == "stripe":
        Omega = np.sqrt(2 * sqrt3 - 0.4)
        kappa = 2.0
        gamma = 8.0
        Dq, Dp, Dy = 1.0, 3 + 2 * sqrt3, 0.2
        K = (Dq + Dp) / 2.0
        Kp = (Dq - Dp) / 2.0
        # one retained e1 bond plus omitted e1/e2 dissipative neighbours
        k_env = 2.0 * K + 2.0 * Dy
    else:
        Omega = 1.8
        kappa = 2.0
        gamma = 8.0
        Dq, Dp = 0.6, 4.5
        K = (Dq + Dp) / 2.0
        Kp = (Dq - Dp) / 2.0
        # isotropic local reduction: one retained bond, three omitted neighbours
        k_env = 2.0 * K
    amp1 = q1 * q1 + p1 * p1
    amp2 = q2 * q2 + p2 * p2
    bond_intensity = 0.5 * (amp1 + amp2)
    k_eff = kappa + k_env + gamma * bond_intensity
    g_eff = abs(Kp) + 0.5 * gamma * bond_intensity
    return closed_form_bond_covariance(k_eff, g_eff, Omega)


def gaussian_bond_arrays_from_amp(Aamp, regime="stripe"):
    """Vectorized closed-form finite-region Gaussian NPT diagnostic.

    Aamp is the retained-bond intensity
        0.5*(|alpha_x|^2 + |alpha_y|^2)
    on each retained horizontal bond.  The computation uses the same
    closed-form covariance/PPT formulas as Proposition 4.1; no rounded cache
    or interpolation is used.  The returned dE is the logarithmic negativity
    above the homogeneous bond-channel baseline for the same regime.
    """
    Aamp = np.asarray(Aamp, dtype=float)
    sqrt3 = np.sqrt(3.0)
    if regime == "stripe":
        Omega = np.sqrt(2.0 * sqrt3 - 0.4)
        kappa = 2.0
        gamma = 8.0
        Dq, Dp, Dy = 1.0, 3.0 + 2.0 * sqrt3, 0.2
        K = 0.5 * (Dq + Dp)
        Kp = 0.5 * (Dq - Dp)
        k_env = 2.0 * K + 2.0 * Dy
    else:
        Omega = 1.8
        kappa = 2.0
        gamma = 8.0
        Dq, Dp = 0.6, 4.5
        K = 0.5 * (Dq + Dp)
        Kp = 0.5 * (Dq - Dp)
        k_env = 2.0 * K

    k_eff = kappa + k_env + gamma * Aamp
    g_eff = abs(Kp) + 0.5 * gamma * Aamp
    S = k_eff * k_eff + 4.0 * Omega * Omega
    Delta = S - 4.0 * g_eff * g_eff
    if np.any(Delta <= 0):
        raise ValueError("Gaussian bond model outside stable range: Delta <= 0")
    nu_phys = 0.5 * np.sqrt(S / Delta)
    sqrtS = np.sqrt(S)
    nu_pt = sqrtS / (2.0 * (sqrtS + 2.0 * g_eff))
    EN = np.maximum(0.0, -np.log2(2.0 * nu_pt))

    k0 = kappa + k_env
    g0 = abs(Kp)
    S0 = k0 * k0 + 4.0 * Omega * Omega
    Delta0 = S0 - 4.0 * g0 * g0
    if Delta0 <= 0:
        raise ValueError("Homogeneous Gaussian bond model outside stable range")
    nu_pt0 = np.sqrt(S0) / (2.0 * (np.sqrt(S0) + 2.0 * g0))
    EN0 = max(0.0, -np.log2(2.0 * nu_pt0))
    dE = np.maximum(0.0, EN - EN0)
    return dE, nu_pt, nu_phys


def _bond_diagnostic_arrays(amp_array, regime):
    """Evaluate the Gaussian bond diagnostic from the closed-form formulas."""
    return gaussian_bond_arrays_from_amp(amp_array, regime)


def fluctuation_entanglement_maps(result, stride=4, max_frames=61):
    """Time-resolved excess Gaussian local logarithmic-negativity maps.

    Computed from the intensity-dependent Lindblad-linearized bond-environment
    covariance.  No rounded field cache is used.
    """
    qf = result["q_frames"]
    pf = result["p_frames"]
    if len(qf) > max_frames:
        idx = np.unique(np.r_[np.linspace(0, len(qf) - 1, max_frames, dtype=int), len(qf) - 1])
    else:
        idx = np.arange(len(qf))
    stride = int(stride)
    if stride < 1:
        raise ValueError("stride must be a positive integer")
    # Every sampled frame is evaluated directly with the vectorized closed-form
    # Gaussian bond-channel formulas. No lookup table or preliminary amplitude
    # scan is used.
    maps = []
    minnu = []
    minphys = []
    meanE = []
    out_times = []
    for ii in idx:
        qq = qf[ii][::stride, ::stride]
        pp = pf[ii][::stride, ::stride]
        intensity = qq * qq + pp * pp
        Aamp = 0.5 * (intensity + np.roll(intensity, -1, axis=1))
        ENbond, NuBond, NphysBond = _bond_diagnostic_arrays(Aamp, result["regime"])
        E = 0.5 * (ENbond + np.roll(ENbond, 1, axis=1))
        Nu = np.minimum(NuBond, np.roll(NuBond, 1, axis=1))
        Nphys = np.minimum(NphysBond, np.roll(NphysBond, 1, axis=1))
        maps.append(E.astype(np.float32))
        minnu.append(float(np.nanmin(Nu)))
        minphys.append(float(np.nanmin(Nphys)))
        meanE.append(float(E.mean()))
        out_times.append(result["times"][ii])
    return (
        np.asarray(maps, dtype=np.float32),
        np.array(meanE),
        np.array(minnu),
        np.array(minphys),
        np.array(out_times),
    )


def fluctuation_entanglement_final_map(result, stride=1):
    """Final-time excess Gaussian local logarithmic-negativity map.

    The reported value is Delta E_N,G^{loc}: the local bond-environment
    Gaussian logarithmic negativity minus the homogeneous-branch value for the
    same regime.  The final static map uses stride=1 by default and uses no
    rounded cache; the local covariance depends on the local fields through the
    branch intensity and is evaluated pointwise by the vectorized closed-form Gaussian bond-channel formulas.
    """
    qf = result.get("q_final_float64", result["q_frames"][-1])
    pf = result.get("p_final_float64", result["p_frames"][-1])
    qq = qf[::stride, ::stride]
    pp = pf[::stride, ::stride]
    intensity = qq * qq + pp * pp
    Aamp = 0.5 * (intensity + np.roll(intensity, -1, axis=1))
    ENbond, NuBond, NphysBond = _bond_diagnostic_arrays(Aamp, result["regime"])
    E = 0.5 * (ENbond + np.roll(ENbond, 1, axis=1))
    Nu = np.minimum(NuBond, np.roll(NuBond, 1, axis=1))
    Nphys = np.minimum(NphysBond, np.roll(NphysBond, 1, axis=1))
    return (
        E.astype(np.float32),
        float(E.mean()),
        float(E.max()),
        float(np.nanmin(Nu)),
        float(np.nanmin(Nphys)),
    )


def max_geff_kappa_ratio(result):
    """Maximum g_eff/kappa_eff over final horizontal bonds.

    This is the stability-margin diagnostic used in the finite-region Gaussian
    reduction.  It is computed from the raw final branch data, not from the
    display-resampled heatmaps.
    """
    qf = result.get("q_final_float64", result["q_frames"][-1])
    pf = result.get("p_final_float64", result["p_frames"][-1])
    intensity = qf * qf + pf * pf
    bond_intensity = 0.5 * (intensity + np.roll(intensity, -1, axis=1))
    sqrt3 = np.sqrt(3.0)
    if result["regime"] == "stripe":
        kappa = 2.0
        gamma = 8.0
        Dq, Dp, Dy = 1.0, 3 + 2 * sqrt3, 0.2
        K = (Dq + Dp) / 2.0
        Kp = (Dq - Dp) / 2.0
        k_env = 2.0 * K + 2.0 * Dy
    else:
        kappa = 2.0
        gamma = 8.0
        Dq, Dp = 0.6, 4.5
        K = (Dq + Dp) / 2.0
        Kp = (Dq - Dp) / 2.0
        k_env = 2.0 * K
    k_eff = kappa + k_env + gamma * bond_intensity
    g_eff = abs(Kp) + 0.5 * gamma * bond_intensity
    return float(np.max(g_eff / k_eff))


