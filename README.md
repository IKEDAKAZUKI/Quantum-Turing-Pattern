# Quantum Turing Patterns

<p align="center">
  <img
    src="display/assets/labyrinth-loop.gif"
    alt="Formation of a labyrinth quantum Turing pattern"
    width="640"
  >
</p>

<p align="center">
  <em>Labyrinth pattern formation — animated GIF preview.</em>
</p>

<p align="center">
  <a href="display/"><strong>Interactive Pattern Explorer</strong></a>
  &nbsp;·&nbsp;
  <a href="reproduction/"><strong>Numerical reproduction</strong></a>
  &nbsp;·&nbsp;
  <a href="lean4%20formalization/"><strong>Lean 4 formalization</strong></a>
</p>

This repository accompanies **Quantum Turing Patterns**. It separates the reader-facing display from the numerical reproduction workflow and the Lean 4 formalization of the algebraic core.

## Highlights

- constructive Turing pattern formation in Lindblad lattice dynamics;
- finite-wave-number selection and commensurate stripe branches;
- two-dimensional stripe, spot, and labyrinth morphologies;
- microscopic Bragg order, Gaussian covariance dynamics, and opposite-momentum entanglement;
- Lean 4 verification of the finite-dimensional identities used in the construction.

## Repository guide

| Path | Contents |
|---|---|
| [`display/`](display/) | Interactive entry point, runnable Jupyter notebooks, and direct links to the stripe, spot, and labyrinth movies. |
| [`reproduction/`](reproduction/) | Numerical code, compact reference data, figure generation, and scientific verification. |
| [`lean4 formalization/`](lean4%20formalization/) | Lean 4 project for the spectral design, parameter map, stripe coefficients, and Gaussian formulas. |

The display and reproduction workflows are intentionally separate: `display/` is optimized for direct exploration and visualization, while `reproduction/` contains the paper-scale calculations and verification tools.

## Interactive Pattern Explorer

The display package includes three Jupyter entry points:

- **[Portable Explorer](Quantum%20turing%20patterns-1.0.0/qtp_explorer_portable.ipynb)** — recommended for most readers; it uses a standard Python kernel and falls back to a regular notebook preview when widget controls are unavailable.
- **[Research Explorer](Quantum%20turing%20patterns-1.0.0/qtp_explorer.ipynb)** — the full interactive interface for varying initial conditions, following the time evolution, and saving selected results.
- **[Exhibit Notebook](Quantum%20turing%20patterns-1.0.0/qtp_exhibit.ipynb)** — a streamlined presentation of the precomputed stripe, spot, and labyrinth movies.

To launch the full Research Explorer:

```bash
cd "Quantum turing patterns-1.0.0"
conda env create -f environment.yml
conda activate qtp-display
python launch_qtp_explorer.py
```

For a standard Jupyter environment, open the portable notebook directly:

```bash
cd "Quantum turing patterns-1.0.0"
jupyter lab qtp_explorer_portable.ipynb
```

See [`display/README.md`](display/README.md) for the notebook roles and direct movie links.

## Numerical reproduction

```bash
cd reproduction
conda env create -f environment.yml
conda activate quantum-turing-patterns
python verify.py
```

A short fresh calculation is available through:

```bash
python run.py --mode smoke --force
python verify.py --mode smoke
```

The `quick` workflow regenerates the compact stripe and Gaussian results. The `full` workflow also generates the complete spot and labyrinth trajectories. See [`reproduction/README.md`](reproduction/README.md) for details.

## Lean 4 formalization

```bash
cd "lean4 formalization"
lake build
```

The Lean toolchain and mathlib revision are pinned. The formalization verifies the algebraic identities used in the spectral design, Lindblad parameter map, explicit stripe coefficients, and homogeneous Gaussian analysis. See [`lean4 formalization/README.md`](lean4%20formalization/README.md) for the precise scope.

## Author

Kazuki Ikeda
