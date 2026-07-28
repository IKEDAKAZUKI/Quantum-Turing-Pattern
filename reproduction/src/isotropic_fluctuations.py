"""Finite-time Gaussian fluctuation analysis for the isotropic examples.

The Spot and Labyrinth reference fields are numerical trajectories rather
than rigorously selected stationary branches. This module therefore evolves
the time-dependent linearized covariance along each first-moment
trajectory and extracts selected opposite-momentum mode pairs by a low-rank
adjoint calculation. It never applies a stationary Lyapunov solve to the
nonstationary reference fields.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Sequence
import json

import numpy as np

from io_utils import save_npz
from scipy.linalg import eigvals

A_REACTION = 1.0
B_REACTION = 3.0
OMEGA = 1.8
DQ = 0.6
DP = 4.5
NU = 4.0
GAMMA = 2.0 * NU
KAPPA = B_REACTION - A_REACTION
K_BOND = 0.5 * (DQ + DP)
L_DEFAULT = 128
PDE_DT = 0.05


@dataclass(frozen=True)
class Trajectory:
    case: str
    times: np.ndarray
    q: np.ndarray
    p: np.ndarray
    params: dict[str, object]


def ell(k: np.ndarray) -> np.ndarray:
    return 2.0 * (1.0 - np.cos(k))


def _init_spot(L: int) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(4)
    yy, xx = np.meshgrid(np.arange(L), np.arange(L), indexing="ij")
    q = np.zeros((L, L), dtype=float)
    p = np.zeros_like(q)
    for _ in range(32):
        cy = int(rng.integers(0, L))
        cx = int(rng.integers(0, L))
        rr = ((xx - cx + L / 2) % L - L / 2) ** 2
        rr += ((yy - cy + L / 2) % L - L / 2) ** 2
        bump = np.exp(-rr / (2.0 * 2.4**2))
        q += 0.12 * bump
        p += -0.25 * 0.12 * bump
    q += 1e-3 * rng.standard_normal((L, L))
    p += 1e-3 * rng.standard_normal((L, L))
    return q, p


def _init_labyrinth(L: int) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(0)
    return 1e-2 * rng.standard_normal((L, L)), 1e-2 * rng.standard_normal((L, L))


def _pde_cache(L: int, dt: float) -> tuple[np.ndarray, ...]:
    k = 2.0 * np.pi * np.fft.fftfreq(L)
    kx, ky = np.meshgrid(k, k)
    esum = ell(kx) + ell(ky)
    L11 = A_REACTION - DQ * esum
    L22 = -B_REACTION - DP * esum
    A11 = 1.0 - dt * L11
    A12 = -dt * OMEGA
    A21 = dt * OMEGA
    A22 = 1.0 - dt * L22
    det = A11 * A22 - A12 * A21
    return A11, A12, A21, A22, det


def _pde_step(q: np.ndarray, p: np.ndarray, dt: float, cache: tuple[np.ndarray, ...]) -> tuple[np.ndarray, np.ndarray]:
    A11, A12, A21, A22, det = cache
    r2 = q * q + p * p
    bq = np.fft.fft2(q - dt * NU * r2 * q)
    bp = np.fft.fft2(p - dt * NU * r2 * p)
    qh = (A22 * bq - A12 * bp) / det
    ph = (-A21 * bq + A11 * bp) / det
    return np.fft.ifft2(qh).real, np.fft.ifft2(ph).real


def regenerate_trajectory(case: str, *, L: int = L_DEFAULT, dt: float = PDE_DT) -> Trajectory:
    case = case.lower()
    if case == "spot":
        q, p = _init_spot(L)
        T = 50.0
        initial = {
            "type": "localized_multi_bump", "rng": "numpy.PCG64", "seed": 4,
            "n_bumps": 32, "bump_amplitude": 0.12, "bump_width": 2.4,
            "p_to_q_ratio": -0.25, "additive_noise": 1e-3,
        }
    elif case == "labyrinth":
        q, p = _init_labyrinth(L)
        T = 80.0
        initial = {
            "type": "independent_gaussian_noise", "rng": "numpy.PCG64",
            "seed": 0, "noise_scale": 1e-2,
        }
    else:
        raise ValueError(f"unknown isotropic case {case!r}")
    nsteps = int(round(T / dt))
    T = nsteps * dt
    times = np.linspace(0.0, T, nsteps + 1)
    qhist = np.empty((nsteps + 1, L, L), dtype=np.float64)
    phist = np.empty_like(qhist)
    qhist[0], phist[0] = q, p
    cache = _pde_cache(L, dt)
    for n in range(nsteps):
        q, p = _pde_step(q, p, dt, cache)
        qhist[n + 1], phist[n + 1] = q, p
    params: dict[str, object] = {
        "case": case, "L": L, "dt": dt, "T": T, "Omega": OMEGA,
        "Dq": DQ, "Dp": DP, "nu": NU, "gamma": GAMMA,
        "kappa": KAPPA, "K": K_BOND, "initial_condition": initial,
    }
    return Trajectory(case, times, qhist, phist, params)


def _state_at(traj: Trajectory, t: float) -> tuple[np.ndarray, np.ndarray]:
    times = traj.times
    if t <= 0.0:
        return np.asarray(traj.q[0], float), np.asarray(traj.p[0], float)
    if t >= times[-1]:
        return np.asarray(traj.q[-1], float), np.asarray(traj.p[-1], float)
    u = t / (times[1] - times[0])
    j = min(int(np.floor(u)), len(times) - 2)
    a = float(u - j)
    return ((1.0 - a) * np.asarray(traj.q[j], float) + a * np.asarray(traj.q[j + 1], float),
            (1.0 - a) * np.asarray(traj.p[j], float) + a * np.asarray(traj.p[j + 1], float))


def _laplacian(u: np.ndarray) -> np.ndarray:
    return (np.roll(u, 1, 0) + np.roll(u, -1, 0)
            + np.roll(u, 1, 1) + np.roll(u, -1, 1) - 4.0 * u)


def _exp2_arrays(a: np.ndarray, b: np.ndarray, c: np.ndarray, d: np.ndarray, h: float) -> tuple[np.ndarray, ...]:
    tr = 0.5 * (a + d)
    x = 0.5 * (a - d)
    delta = np.sqrt((x * x + b * c).astype(complex))
    z = h * delta
    etr = np.exp(h * tr)
    ch = np.cosh(z)
    sh_over = np.empty_like(delta, dtype=complex)
    mask = np.abs(delta) > 1e-14
    sh_over[mask] = np.sinh(z[mask]) / delta[mask]
    sh_over[~mask] = h
    return (etr * (ch + sh_over * x), etr * sh_over * b,
            etr * sh_over * c, etr * (ch - sh_over * x))


def _constant_adjoint_half_step(L: int, h: float) -> tuple[np.ndarray, ...]:
    k = 2.0 * np.pi * np.fft.fftfreq(L)
    kx, ky = np.meshgrid(k, k)
    esum = ell(kx) + ell(ky)
    a = A_REACTION - DQ * esum
    d = -B_REACTION - DP * esum
    b = -OMEGA * np.ones_like(a)
    c = OMEGA * np.ones_like(a)
    return _exp2_arrays(a, b, c, d, h)


def _constant_step(Zq: np.ndarray, Zp: np.ndarray, E: tuple[np.ndarray, ...]) -> tuple[np.ndarray, np.ndarray]:
    E11, E12, E21, E22 = E
    qh = np.fft.fft2(Zq, axes=(0, 1))
    ph = np.fft.fft2(Zp, axes=(0, 1))
    qn = np.fft.ifft2(E11[..., None] * qh + E12[..., None] * ph, axes=(0, 1)).real
    pn = np.fft.ifft2(E21[..., None] * qh + E22[..., None] * ph, axes=(0, 1)).real
    return qn, pn


def _local_step(Zq: np.ndarray, Zp: np.ndarray, q: np.ndarray, p: np.ndarray, h: float) -> tuple[np.ndarray, np.ndarray]:
    # N(t)^T=-nu(r^2 I+2uu^T), so its exponential is available pointwise.
    r2 = q * q + p * p
    base = np.exp(-NU * r2 * h)
    extra = np.exp(-2.0 * NU * r2 * h) - 1.0
    dot = q[..., None] * Zq + p[..., None] * Zp
    coeff = np.zeros_like(dot)
    mask = r2 > 1e-30
    coeff[mask] = extra[mask, None] * dot[mask] / r2[mask, None]
    return (base[..., None] * (Zq + q[..., None] * coeff),
            base[..., None] * (Zp + p[..., None] * coeff))


def _strang_step(Zq: np.ndarray, Zp: np.ndarray, qmid: np.ndarray, pmid: np.ndarray,
                 Ehalf: tuple[np.ndarray, ...], dt: float) -> tuple[np.ndarray, np.ndarray]:
    Zq, Zp = _constant_step(Zq, Zp, Ehalf)
    Zq, Zp = _local_step(Zq, Zp, qmid, pmid, dt)
    return _constant_step(Zq, Zp, Ehalf)


def _diffusion_quadratic_blocks(q: np.ndarray, p: np.ndarray, Zq: np.ndarray, Zp: np.ndarray) -> np.ndarray:
    """Return only the independent 4x4 selected-mode blocks of Z^T D Z.

    Different selected pairs are never mixed in the reported reductions, so
    forming their cross-covariances would add quadratic work without changing
    any output.
    """
    n_modes = Zq.shape[-1] // 4
    local = (KAPPA / 2.0 + GAMMA * (q * q + p * p))[..., None]
    DqZ = local * Zq - K_BOND * _laplacian(Zq)
    DpZ = local * Zp - K_BOND * _laplacian(Zp)
    zq = Zq.reshape(Zq.shape[0], Zq.shape[1], n_modes, 4)
    zp = Zp.reshape(Zp.shape[0], Zp.shape[1], n_modes, 4)
    dq = DqZ.reshape(DqZ.shape[0], DqZ.shape[1], n_modes, 4)
    dp = DpZ.reshape(DpZ.shape[0], DpZ.shape[1], n_modes, 4)
    return (np.einsum("xyma,xymb->mab", zq, dq, optimize=True)
            + np.einsum("xyma,xymb->mab", zp, dp, optimize=True))


def canonical_mode_columns(L: int, modes: Sequence[tuple[int, int]]) -> tuple[np.ndarray, np.ndarray]:
    yy, xx = np.meshgrid(np.arange(L), np.arange(L), indexing="ij")
    qcols: list[np.ndarray] = []
    pcols: list[np.ndarray] = []
    norm = np.sqrt(float(L * L))
    for mx, my in modes:
        phase = 2.0 * np.pi * (mx * xx + my * yy) / L
        co = np.cos(phase) / norm
        si = np.sin(phase) / norm
        qcols.extend([co, -si, co, si])
        pcols.extend([si, co, -si, co])
    return np.stack(qcols, axis=-1), np.stack(pcols, axis=-1)


def symplectic_form(n_modes: int) -> np.ndarray:
    J = np.zeros((2 * n_modes, 2 * n_modes), dtype=float)
    for j in range(n_modes):
        J[2*j, 2*j+1] = 1.0
        J[2*j+1, 2*j] = -1.0
    return J


def selection_symplectic_error(L: int, modes: Sequence[tuple[int, int]]) -> float:
    Zq, Zp = canonical_mode_columns(L, modes)
    SJST = np.einsum("xyi,xyj->ij", Zq, Zp) - np.einsum("xyi,xyj->ij", Zp, Zq)
    return float(np.max(np.abs(SJST - symplectic_form(2 * len(modes)))))


def symplectic_eigenvalues(V: np.ndarray) -> np.ndarray:
    V = np.asarray(V, float)
    vals = np.sort(np.abs(eigvals(1j * symplectic_form(V.shape[0] // 2) @ V)))
    return np.real(vals[::2])


def pair_metrics(V: np.ndarray) -> dict[str, float]:
    V = 0.5 * (np.asarray(V, float) + np.asarray(V, float).T)
    PT = np.diag([1.0, 1.0, 1.0, -1.0])
    ordinary = symplectic_eigenvalues(V)
    pt = symplectic_eigenvalues(PT @ V @ PT)
    nu_pt = float(np.min(pt))
    return {
        "nu_physical_min": float(np.min(ordinary)),
        "nu_pt": nu_pt,
        "logarithmic_negativity": float(max(0.0, -np.log2(2.0 * nu_pt))),
        "covariance_min_eigenvalue": float(np.min(np.linalg.eigvalsh(V))),
    }


def finite_time_selected_covariances(traj: Trajectory, modes: Sequence[tuple[int, int]], *,
                                      dt_cov: float = 0.05) -> tuple[list[np.ndarray], dict[str, float | int]]:
    """Return selected final covariances without constructing the full covariance.

    If Z(tau)=Phi(T,T-tau)^T S^T, then
      S V(T) S^T = 1/2 Z(T)^T Z(T) + integral Z^T D Z d tau.
    The adjoint equation is integrated by a second-order Strang splitting;
    the noise integral uses the trapezoidal rule.
    """
    L = traj.q.shape[1]
    T = float(traj.times[-1])
    nsteps = int(round(T / float(dt_cov)))
    dt = T / nsteps
    Ehalf = _constant_adjoint_half_step(L, dt / 2.0)
    Zq, Zp = canonical_mode_columns(L, modes)
    n_modes = len(modes)
    P = np.zeros((n_modes, 4, 4), dtype=float)
    for n in range(nsteps):
        s = T - n * dt
        q, p = _state_at(traj, s)
        G0 = _diffusion_quadratic_blocks(q, p, Zq, Zp)
        qmid, pmid = _state_at(traj, s - dt / 2.0)
        Zqn, Zpn = _strang_step(Zq, Zp, qmid, pmid, Ehalf, dt)
        qn, pn = _state_at(traj, s - dt)
        G1 = _diffusion_quadratic_blocks(qn, pn, Zqn, Zpn)
        P += 0.5 * dt * (G0 + G1)
        Zq, Zp = Zqn, Zpn
    zq = Zq.reshape(L, L, n_modes, 4)
    zp = Zp.reshape(L, L, n_modes, 4)
    P += 0.5 * (np.einsum("xyma,xymb->mab", zq, zq, optimize=True)
                + np.einsum("xyma,xymb->mab", zp, zp, optimize=True))
    P = 0.5 * (P + np.swapaxes(P, -1, -2))
    covs = [P[j].copy() for j in range(n_modes)]
    meta: dict[str, float | int] = {
        "dt_cov": dt, "nsteps": nsteps,
        "selection_symplectic_error": selection_symplectic_error(L, modes),
        "final_adjoint_frobenius_norm": float(np.sqrt(np.sum(Zq * Zq) + np.sum(Zp * Zp))),
    }
    return covs, meta


def mode_radius(L: int, mode: tuple[int, int]) -> float:
    return float(2.0 * np.pi * np.hypot(mode[0], mode[1]) / L)


def representative_modes(case: str) -> tuple[list[tuple[int, int]], set[tuple[int, int]]]:
    case = case.lower()
    if case == "spot":
        shell = [(12, 0), (11, 5), (8, 8), (5, 11), (0, 12)]
    elif case == "labyrinth":
        shell = [(13, 0), (12, 5), (9, 9), (5, 12), (0, 13)]
    else:
        raise ValueError(case)
    modes = shell + [(4, 0), (20, 0)]
    return modes, set(shell)


def save_trajectory(path: Path, traj: Trajectory) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    save_npz(
        path,
        case=np.asarray(traj.case),
        times=np.asarray(traj.times, np.float64),
        q=np.asarray(traj.q, np.float64),
        p=np.asarray(traj.p, np.float64),
        params_json=json.dumps(traj.params, sort_keys=True),
    )


def load_trajectory(path: Path) -> Trajectory:
    with np.load(path, allow_pickle=False) as data:
        return Trajectory(
            case=str(data["case"]),
            times=np.asarray(data["times"], np.float64),
            q=np.asarray(data["q"], np.float64),
            p=np.asarray(data["p"], np.float64),
            params=json.loads(str(data["params_json"])),
        )
