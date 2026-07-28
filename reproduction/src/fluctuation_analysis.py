"""Gaussian fluctuation analysis for the Lindblad lattice model.

The module reproduces the commensurate stripe branch (period 12), its branch-linearized
Gaussian covariance, homogeneous opposite-momentum spectra, longitudinal
supercell spectra, and commensurate phase-locking diagnostics.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd
from scipy.linalg import eigvals, solve_continuous_lyapunov, svdvals
from scipy.optimize import root

A_REACTION = 1.0
B_REACTION = 3.0
NU = 4.0
GAMMA = 2.0 * NU
KAPPA = B_REACTION - A_REACTION
EPSILON = 0.5 * (A_REACTION + B_REACTION)
DQ_STRIPE = 1.0
DP_STRIPE = 3.0 + 2.0 * np.sqrt(3.0)
D_SCALAR = 0.5 * (DQ_STRIPE + DP_STRIPE)
KSTAR = np.pi / 6.0
PERIOD = 12
BETA = (np.sqrt(3.0) - 1.0) / np.sqrt(2.0 * np.sqrt(3.0))
PHASE = np.pi / 12.0


@dataclass(frozen=True)
class BranchResult:
    lam: float
    profile: np.ndarray  # shape (L, 2), columns q,p
    residual_inf: float
    solver_success: bool
    solver_message: str


def omega(lam: float) -> float:
    value = 2.0 * np.sqrt(3.0) - float(lam)
    if value <= 0.0:
        raise ValueError(f"Omega^2 must be positive, got {value} for lambda={lam}")
    return float(np.sqrt(value))


def transport_to_bonds(Dq: float, Dp: float) -> tuple[float, float]:
    return 0.5 * (Dq + Dp), 0.5 * (Dq - Dp)


def _full_from_reflection_reduced(v: np.ndarray) -> np.ndarray:
    """Expand six bond-reflection sites to the commensurate period-12 profile."""
    v = np.asarray(v, dtype=float)
    if v.shape != (12,):
        raise ValueError(f"Expected 12 reduced variables, got {v.shape}")
    sites = v.reshape(6, 2)
    return np.vstack([sites, sites[::-1]])


def _reduced_from_full(profile: np.ndarray) -> np.ndarray:
    profile = np.asarray(profile, dtype=float)
    if profile.shape != (PERIOD, 2):
        raise ValueError(f"Expected {(PERIOD, 2)}, got {profile.shape}")
    return profile[:6].reshape(-1)


def stationary_residual(
    profile: np.ndarray,
    lam: float,
    *,
    Dq: float = DQ_STRIPE,
    Dp: float = DP_STRIPE,
    nu: float = NU,
) -> np.ndarray:
    profile = np.asarray(profile, dtype=float)
    L = profile.shape[0]
    if profile.shape != (L, 2):
        raise ValueError("profile must have shape (L,2)")
    q, p = profile[:, 0], profile[:, 1]
    r2 = q * q + p * p
    lap_q = np.roll(q, -1) + np.roll(q, 1) - 2.0 * q
    lap_p = np.roll(p, -1) + np.roll(p, 1) - 2.0 * p
    Fq = A_REACTION * q + omega(lam) * p - nu * r2 * q + Dq * lap_q
    Fp = -omega(lam) * q - B_REACTION * p - nu * r2 * p + Dp * lap_p
    return np.column_stack([Fq, Fp])


def branch_jacobian(
    profile: np.ndarray,
    lam: float,
    *,
    Dq: float = DQ_STRIPE,
    Dp: float = DP_STRIPE,
    nu: float = NU,
) -> np.ndarray:
    profile = np.asarray(profile, dtype=float)
    L = profile.shape[0]
    q, p = profile[:, 0], profile[:, 1]
    J = np.zeros((2 * L, 2 * L), dtype=float)
    om = omega(lam)
    for x in range(L):
        iq, ip = 2 * x, 2 * x + 1
        J[iq, iq] = A_REACTION - nu * (3.0 * q[x] ** 2 + p[x] ** 2) - 2.0 * Dq
        J[iq, ip] = om - 2.0 * nu * q[x] * p[x]
        J[ip, iq] = -om - 2.0 * nu * q[x] * p[x]
        J[ip, ip] = -B_REACTION - nu * (q[x] ** 2 + 3.0 * p[x] ** 2) - 2.0 * Dp
        for y in ((x - 1) % L, (x + 1) % L):
            J[iq, 2 * y] += Dq
            J[ip, 2 * y + 1] += Dp
    return J


_REFLECTION_EXPAND = np.zeros((2 * PERIOD, 12), dtype=float)
for _x in range(PERIOD):
    _y = _x if _x < 6 else 11 - _x
    _REFLECTION_EXPAND[2 * _x, 2 * _y] = 1.0
    _REFLECTION_EXPAND[2 * _x + 1, 2 * _y + 1] = 1.0
_REFLECTION_RESTRICT = np.zeros((12, 2 * PERIOD), dtype=float)
_REFLECTION_RESTRICT[:, :12] = np.eye(12)


def leading_branch_seed(lam: float) -> np.ndarray:
    if lam <= 0.0:
        raise ValueError("The nonlinear stripe seed requires lambda>0")
    B = np.sqrt(lam / (8.0 * np.sqrt(3.0) * NU))
    x = np.arange(PERIOD, dtype=float)
    q = 2.0 * B * np.cos(KSTAR * x + PHASE)
    p = -BETA * q
    return _reduced_from_full(np.column_stack([q, p]))


def solve_bond_centered_branch(
    lam: float,
    *,
    seed: np.ndarray | None = None,
    xtol: float = 1e-12,
) -> BranchResult:
    if lam <= 0.0:
        raise ValueError("The patterned branch requires lambda>0")
    v0 = leading_branch_seed(lam) if seed is None else np.asarray(seed, dtype=float)

    def residual_reduced(v: np.ndarray) -> np.ndarray:
        profile = _full_from_reflection_reduced(v)
        return _REFLECTION_RESTRICT @ stationary_residual(profile, lam).reshape(-1)

    def jacobian_reduced(v: np.ndarray) -> np.ndarray:
        profile = _full_from_reflection_reduced(v)
        return _REFLECTION_RESTRICT @ branch_jacobian(profile, lam) @ _REFLECTION_EXPAND

    sol = root(
        residual_reduced,
        v0,
        jac=jacobian_reduced,
        method="hybr",
        options={"xtol": xtol, "maxfev": 8000},
    )
    profile = _full_from_reflection_reduced(sol.x)
    residual_inf = float(np.max(np.abs(stationary_residual(profile, lam))))
    if not sol.success or residual_inf > 5e-11:
        raise RuntimeError(
            f"Branch solve failed at lambda={lam}: success={sol.success}, "
            f"residual={residual_inf:.3e}, message={sol.message}"
        )
    return BranchResult(float(lam), profile, residual_inf, bool(sol.success), str(sol.message))


def continue_branches(lambdas: Iterable[float]) -> dict[float, BranchResult]:
    values = sorted({float(x) for x in lambdas}, reverse=True)
    if not values or values[-1] <= 0.0:
        raise ValueError("All continuation values must be positive")
    results: dict[float, BranchResult] = {}
    seed: np.ndarray | None = None
    for lam in values:
        result = solve_bond_centered_branch(lam, seed=seed)
        results[lam] = result
        seed = _reduced_from_full(result.profile)
    return results


def graph_laplacian(L: int) -> np.ndarray:
    out = 2.0 * np.eye(L)
    for x in range(L):
        out[x, (x - 1) % L] = -1.0
        out[x, (x + 1) % L] = -1.0
    return out


def branch_diffusion(
    profile: np.ndarray,
    *,
    Dq: float = DQ_STRIPE,
    Dp: float = DP_STRIPE,
) -> np.ndarray:
    profile = np.asarray(profile, dtype=float)
    L = profile.shape[0]
    K, _ = transport_to_bonds(Dq, Dp)
    local = KAPPA / 2.0 + GAMMA * np.sum(profile * profile, axis=1)
    D = np.kron(np.diag(local), np.eye(2)) + K * np.kron(graph_laplacian(L), np.eye(2))
    return D


def stationary_covariance(A: np.ndarray, D: np.ndarray) -> np.ndarray:
    V = solve_continuous_lyapunov(np.asarray(A, float), -np.asarray(D, float))
    return 0.5 * (V + V.T)


def symplectic_form(n_modes: int) -> np.ndarray:
    J = np.zeros((2 * n_modes, 2 * n_modes), dtype=float)
    for m in range(n_modes):
        J[2 * m, 2 * m + 1] = 1.0
        J[2 * m + 1, 2 * m] = -1.0
    return J


def symplectic_eigenvalues(V: np.ndarray) -> np.ndarray:
    V = np.asarray(V, dtype=float)
    n_modes = V.shape[0] // 2
    values = np.sort(np.abs(eigvals(1j * symplectic_form(n_modes) @ V)))
    # Every symplectic eigenvalue occurs twice in the ordinary spectrum.
    return np.real(values[::2])


def partial_transpose_covariance(V: np.ndarray, mode: int = 1) -> np.ndarray:
    V = np.asarray(V, dtype=float)
    n_modes = V.shape[0] // 2
    if not (0 <= mode < n_modes):
        raise ValueError("mode out of range")
    P = np.eye(2 * n_modes)
    P[2 * mode + 1, 2 * mode + 1] = -1.0
    return P @ V @ P


def logarithmic_negativity(nu_pt: float) -> float:
    return float(max(0.0, -np.log2(2.0 * float(nu_pt))))


def fourier_symplectic(L: int, k: float) -> np.ndarray:
    x = np.arange(L, dtype=float)
    c = np.cos(k * x) / np.sqrt(L)
    s = np.sin(k * x) / np.sqrt(L)
    S = np.zeros((4, 2 * L), dtype=float)
    for j in range(L):
        S[:, 2 * j : 2 * j + 2] = np.array(
            [[c[j], s[j]], [-s[j], c[j]], [c[j], -s[j]], [s[j], c[j]]]
        )
    return S


def mode_pair_metrics(V: np.ndarray, k: float) -> dict[str, float | np.ndarray]:
    L = V.shape[0] // 2
    S = fourier_symplectic(L, k)
    pair = S @ V @ S.T
    ordinary = symplectic_eigenvalues(pair)
    pt = symplectic_eigenvalues(partial_transpose_covariance(pair, 1))
    J_err = float(np.max(np.abs(S @ symplectic_form(L) @ S.T - symplectic_form(2))))
    return {
        "covariance": pair,
        "nu_physical": float(np.min(ordinary)),
        "nu_pt": float(np.min(pt)),
        "logarithmic_negativity": logarithmic_negativity(float(np.min(pt))),
        "symplectic_transform_error": J_err,
    }


def gaussian_cp_margin(A: np.ndarray, D: np.ndarray) -> float:
    J = symplectic_form(A.shape[0] // 2)
    M = D - 0.5j * (A @ J + J @ A.T)
    M = 0.5 * (M + M.conj().T)
    return float(np.min(np.linalg.eigvalsh(M)))


def lyapunov_residual(A: np.ndarray, D: np.ndarray, V: np.ndarray) -> float:
    R = A @ V + V @ A.T + D
    return float(np.linalg.norm(R, ord="fro") / np.linalg.norm(D, ord="fro"))



def lyapunov_absolute_residual(A: np.ndarray, D: np.ndarray, V: np.ndarray) -> float:
    """Frobenius norm of the algebraic Lyapunov residual."""
    return float(np.linalg.norm(A @ V + V @ A.T + D, ord="fro"))


def lyapunov_operator_separation(A: np.ndarray) -> float:
    """Smallest singular value of X -> A X + X A^T in Frobenius geometry."""
    A = np.asarray(A, float)
    n = A.shape[0]
    operator = np.kron(np.eye(n), A) + np.kron(A, np.eye(n))
    return float(np.min(svdvals(operator)))


def covariance_error_bound_fro(A: np.ndarray, D: np.ndarray, V: np.ndarray) -> float:
    """Residual/separation a-posteriori bound for the stationary covariance."""
    sep = lyapunov_operator_separation(A)
    if sep <= 0.0:
        return float("inf")
    return lyapunov_absolute_residual(A, D, V) / sep


def uncertainty_min_eigenvalue(V: np.ndarray) -> float:
    """Smallest eigenvalue of V+iJ/2 for a Gaussian covariance."""
    V = np.asarray(V, float)
    J = symplectic_form(V.shape[0] // 2)
    H = 0.5 * (V + V.T) + 0.5j * J
    H = 0.5 * (H + H.conj().T)
    return float(np.min(np.linalg.eigvalsh(H)))


def pt_uncertainty_min_eigenvalue(V: np.ndarray, mode: int = 1) -> float:
    """Smallest eigenvalue of V^PT+iJ/2 for a Gaussian two-mode covariance."""
    return uncertainty_min_eigenvalue(partial_transpose_covariance(np.asarray(V, float), mode))


def branch_critical_amplitude(profile: np.ndarray) -> float:
    """Magnitude B of the +k* Fourier coefficient projected onto r."""
    profile = np.asarray(profile, float)
    x = np.arange(profile.shape[0], dtype=float)
    coeff = np.sum(np.exp(-1j * KSTAR * x)[:, None] * profile, axis=0) / profile.shape[0]
    r = np.array([1.0, -BETA])
    return float(abs(np.vdot(r, coeff) / np.dot(r, r)))


def reflection_fixed_spectral_abscissa(profile: np.ndarray, lam: float) -> float:
    """Spectral abscissa on the bond-reflection fixed period-cell space."""
    A = branch_jacobian(profile, lam)
    reduced = _REFLECTION_RESTRICT @ A @ _REFLECTION_EXPAND
    return float(np.max(np.real(eigvals(reduced))))


def period_cell_npt_minimizer(profile: np.ndarray, lam: float) -> tuple[int, float, float]:
    """Return (mode index, k/pi, minimum nu_PT) on the period-12 cell."""
    V = stationary_covariance(branch_jacobian(profile, lam), branch_diffusion(profile))
    rows: list[tuple[int, float]] = []
    for mode in range(1, profile.shape[0] // 2):
        k = 2.0 * np.pi * mode / profile.shape[0]
        rows.append((mode, float(mode_pair_metrics(V, k)["nu_pt"])))
    mode, nu_pt = min(rows, key=lambda item: item[1])
    return int(mode), float(2.0 * mode / profile.shape[0]), float(nu_pt)


def phase_tangent(L: int = PERIOD) -> np.ndarray:
    x = np.arange(L, dtype=float)
    q = -np.sin(KSTAR * x + PHASE)
    p = -BETA * q
    vec = np.column_stack([q, p]).reshape(-1)
    return vec / np.linalg.norm(vec)


def spectral_abscissa_and_overlap(A: np.ndarray) -> tuple[float, float]:
    values, vectors = np.linalg.eig(A)
    idx = int(np.argmax(np.real(values)))
    abscissa = float(np.real(values[idx]))
    v = np.real_if_close(vectors[:, idx]).astype(complex)
    if np.linalg.norm(np.imag(v)) < 1e-9:
        v = np.real(v)
    else:
        # Choose the real or imaginary component with the larger norm.
        v = np.real(v) if np.linalg.norm(np.real(v)) >= np.linalg.norm(np.imag(v)) else np.imag(v)
    v = np.asarray(v, dtype=float)
    v /= np.linalg.norm(v)
    tangent = phase_tangent(A.shape[0] // 2)
    overlap = float(abs(np.dot(v, tangent)))
    return abscissa, overlap


def homogeneous_parameters(
    lam: float,
    k: float,
    *,
    Dq: float = DQ_STRIPE,
    Dp: float = DP_STRIPE,
) -> dict[str, float]:
    ell = 2.0 - 2.0 * np.cos(float(k))
    K, Kprime = transport_to_bonds(Dq, Dp)
    kappa_k = KAPPA + 2.0 * K * ell
    g_k = EPSILON - Kprime * ell
    R_k = float(np.sqrt(kappa_k * kappa_k + 4.0 * omega(lam) ** 2))
    eta_k = 2.0 * abs(g_k) / R_k
    if eta_k >= 1.0:
        raise ValueError(f"Unstable homogeneous mode: eta={eta_k} at lambda={lam}, k={k}")
    nu_physical = 1.0 / (2.0 * np.sqrt(1.0 - eta_k * eta_k))
    nu_pt = 1.0 / (2.0 * (1.0 + eta_k))
    return {
        "ell": float(ell),
        "kappa_k": float(kappa_k),
        "g_k": float(g_k),
        "R_k": float(R_k),
        "eta_k": float(eta_k),
        "nu_physical": float(nu_physical),
        "nu_pt": float(nu_pt),
        "logarithmic_negativity": logarithmic_negativity(nu_pt),
    }



def standing_to_traveling_symplectic() -> np.ndarray:
    """Passive 50:50 recombination of the two real standing-wave modes."""
    return (1.0 / np.sqrt(2.0)) * np.array(
        [
            [1.0, 0.0, 1.0, 0.0],
            [0.0, 1.0, 0.0, 1.0],
            [1.0, 0.0, -1.0, 0.0],
            [0.0, 1.0, 0.0, -1.0],
        ],
        dtype=float,
    )

def homogeneous_pair_covariance(
    lam: float,
    k: float,
    *,
    Dq: float = DQ_STRIPE,
    Dp: float = DP_STRIPE,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    pars = homogeneous_parameters(lam, k, Dq=Dq, Dp=Dp)
    kappa_k, g_k, R_k = pars["kappa_k"], pars["g_k"], pars["R_k"]
    Delta = R_k * R_k - 4.0 * g_k * g_k
    s = R_k * R_k / (2.0 * Delta)
    C = g_k / Delta * np.array([[kappa_k, -2.0 * omega(lam)], [-2.0 * omega(lam), -kappa_k]])
    V = np.block([[s * np.eye(2), C], [C, s * np.eye(2)]])
    Aplus = np.array(
        [[-kappa_k / 2.0 + g_k, omega(lam)], [-omega(lam), -kappa_k / 2.0 - g_k]]
    )
    Aminus = np.array(
        [[-kappa_k / 2.0 - g_k, omega(lam)], [-omega(lam), -kappa_k / 2.0 + g_k]]
    )
    A = np.block([[Aplus, np.zeros((2, 2))], [np.zeros((2, 2)), Aminus]])
    D = kappa_k / 2.0 * np.eye(4)
    return A, D, V


def homogeneous_spectrum_table(
    lam: float,
    L: int,
    *,
    state: str,
    Dq: float,
    Dp: float,
    max_k_over_pi: float = 1.0,
) -> pd.DataFrame:
    rows: list[dict[str, float | int | str]] = []
    max_mode = min(int(np.floor(max_k_over_pi * L / 2.0 + 1e-12)), L // 2 - 1)
    for mode in range(1, max_mode + 1):
        k = 2.0 * np.pi * mode / L
        pars = homogeneous_parameters(lam, k, Dq=Dq, Dp=Dp)
        rows.append(
            {
                "state": state,
                "lambda": lam,
                "L": L,
                "mode_index": mode,
                "k": k,
                "k_over_pi": k / np.pi,
                **pars,
            }
        )
    return pd.DataFrame(rows)


def tile_profile(profile: np.ndarray, L: int) -> np.ndarray:
    profile = np.asarray(profile, dtype=float)
    period = profile.shape[0]
    if L % period:
        raise ValueError(f"L={L} is not a multiple of period={period}")
    return np.tile(profile, (L // period, 1))


def supercell_mode_table(profile: np.ndarray, lam: float, L: int) -> tuple[pd.DataFrame, np.ndarray, np.ndarray, np.ndarray]:
    tiled = tile_profile(profile, L)
    A = branch_jacobian(tiled, lam)
    D = branch_diffusion(tiled)
    V = stationary_covariance(A, D)
    abscissa = float(np.max(np.real(eigvals(A))))
    residual = lyapunov_residual(A, D, V)
    min_symp = float(np.min(symplectic_eigenvalues(V)))
    cp = gaussian_cp_margin(A, D)
    rows = []
    for mode in range(1, L // 2):
        k = 2.0 * np.pi * mode / L
        metrics = mode_pair_metrics(V, k)
        rows.append(
            {
                "L": L,
                "mode_index": mode,
                "k": k,
                "k_over_pi": k / np.pi,
                "nu_physical": metrics["nu_physical"],
                "nu_pt": metrics["nu_pt"],
                "logarithmic_negativity": metrics["logarithmic_negativity"],
                "symplectic_transform_error": metrics["symplectic_transform_error"],
                "spectral_abscissa": abscissa,
                "relative_lyapunov_residual": residual,
                "min_full_symplectic_eigenvalue": min_symp,
                "cp_margin": cp,
            }
        )
    return pd.DataFrame(rows), A, D, V


def branch_summary(profile: np.ndarray, lam: float) -> dict[str, float | np.ndarray]:
    A = branch_jacobian(profile, lam)
    D = branch_diffusion(profile)
    V = stationary_covariance(A, D)
    abscissa, overlap = spectral_abscissa_and_overlap(A)
    metrics = mode_pair_metrics(V, KSTAR)
    return {
        "A": A,
        "D": D,
        "V": V,
        "spectral_abscissa": abscissa,
        "locking_rate": -abscissa,
        "relative_lyapunov_residual": lyapunov_residual(A, D, V),
        "min_full_symplectic_eigenvalue": float(np.min(symplectic_eigenvalues(V))),
        "cp_margin": gaussian_cp_margin(A, D),
        "kstar_pair_covariance": metrics["covariance"],
        "kstar_nu_physical": metrics["nu_physical"],
        "kstar_nu_pt": metrics["nu_pt"],
        "kstar_logarithmic_negativity": metrics["logarithmic_negativity"],
        "phase_tangent_overlap": overlap,
    }


def write_matrix(path: Path, matrix: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savetxt(path, np.asarray(matrix, dtype=float), delimiter=",", fmt="%.17g")


def read_matrix(path: Path) -> np.ndarray:
    return np.loadtxt(path, delimiter=",")
