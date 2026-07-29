from __future__ import annotations

import numpy as np

import qtp_observables as observables

CMAP_FIELD = "RdYlBu_r"
VMIN, VMAX = -0.24, 0.24
ISOTROPIC_PATTERN_BETA = 0.40
ISOTROPIC_FASTEST_K = 0.6491851977897728
ISOTROPIC_FASTEST_RIGHT_P_OVER_Q = -0.3651541709271028


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
    pattern checks and convergence controls.
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
    storage_precision = {"time_series": "float32", "final_state": "float64"}
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
# Linear-dispersion diagnostics
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
    """Return the one-dimensional linear dispersion used in the figures.

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


