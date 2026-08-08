# Quantum Turing Patterns

<p align="center">
  <img
    src="display/assets/labyrinth-animation_v2.gif"
    alt="Formation of a labyrinth quantum Turing pattern"
    width="1040"
  >
</p>

<p align="center">
  <em>Labyrinth pattern formation.</em>
</p>

<p align="center">
  <a href="display/"><strong>Public display</strong></a>
  &nbsp;·&nbsp;
  <a href="interactive-display/"><strong>Interactive explorer</strong></a>
  &nbsp;·&nbsp;
  <a href="reproduction/"><strong>Numerical reproduction</strong></a>
  &nbsp;·&nbsp;
  <a href="lean4%20formalization/"><strong>Lean 4 formalization</strong></a>
</p>

This repository accompanies **Quantum Turing Patterns**. The visualization, interactive explorer, numerical reproduction workflow, and Lean 4 formalization are kept in separate directories.

## Highlights

- constructive Turing pattern formation in Lindblad lattice dynamics;
- finite-wave-number selection and commensurate stripe branches;
- two-dimensional stripe, spot, and labyrinth morphologies;
- microscopic Bragg order, Gaussian covariance dynamics, and opposite-momentum entanglement;
- Lean 4 verification of the finite-dimensional identities used in the construction.

[`A popular article for the general public`](https://www.linkedin.com/pulse/quantum-turing-patterns-kazuki-ikeda-373ic/)  

## Repository guide

| Path | Contents |
|---|---|
| [`display/`](display/) | Lightweight GitHub display and direct links to the stripe, spot, and labyrinth movies. |
| [`interactive-display/`](interactive-display/) | Jupyter explorers, presentation viewer, and display-generation tools. |
| [`reproduction/`](reproduction/) | Numerical code, compact reference data, figure generation, and scientific verification. |
| [`lean4 formalization/`](lean4%20formalization/) | Lean 4 project for the spectral design, parameter map, stripe coefficients, and Gaussian formulas. |

The interactive display and reproduction workflows are intentionally separate: `interactive-display/` is designed for direct exploration and visualization, while `reproduction/` contains the paper-scale calculations and verification tools.

## Interactive Pattern Explorer

The interactive display package includes three Jupyter entry points:

- **[Portable Explorer](interactive-display/qtp_explorer_portable.ipynb)** — recommended for most readers; it uses a standard Python kernel and falls back to a regular notebook preview when widget controls are unavailable.
- **[Research Explorer](interactive-display/qtp_explorer.ipynb)** — the full interactive interface for varying initial conditions, following the time evolution, and saving selected results.
- **[Exhibit Notebook](interactive-display/qtp_exhibit.ipynb)** — a streamlined presentation of the precomputed stripe, spot, and labyrinth movies.

To launch the full Research Explorer:

```bash
cd interactive-display
conda env create -f environment.yml
conda activate qtp-display
python launch_qtp_explorer.py
```

For a standard Jupyter environment, open the portable notebook directly:

```bash
cd interactive-display
jupyter lab qtp_explorer_portable.ipynb
```

See [`interactive-display/README.md`](interactive-display/README.md) for the notebook roles and [`display/README.md`](display/README.md) for the movie gallery.

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

## Reference

K. Ikeda,  
**“Quantum Turing Patterns,” [`arXiv:2607.26331 [math-ph]`](https://arxiv.org/abs/2607.26331)**  
[`Quantum_Turing_Patterns__Kazuki_Ikeda(2026).pdf`](Quantum_Turing_Patterns__Kazuki_Ikeda(2026).pdf)
