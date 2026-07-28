# Quantum Turing Patterns

<p align="center">
  <a href="Quantum%20turing%20patterns-1.0.0/display/display_labyrinth_pattern.mp4">
    <img src="display/assets/labyrinth-loop.gif" alt="Formation of a labyrinth quantum Turing pattern" width="720">
  </a>
</p>

<p align="center">
  <em>Formation of the Labyrinth pattern. Click the animation to open the full-resolution movie.</em>
</p>

<p align="center">
  <a href="display/"><strong>Public display</strong></a>
  &nbsp;·&nbsp;
  <a href="reproduction/"><strong>Numerical reproduction</strong></a>
  &nbsp;·&nbsp;
  <a href="lean4%20formalization/"><strong>Lean 4 formalization</strong></a>
</p>

This repository accompanies **Quantum Turing Patterns**. It separates reader-facing visualizations from the numerical reproduction workflow and the Lean 4 formalization of the algebraic core.

## Highlights

- constructive Turing pattern formation in Lindblad lattice dynamics;
- finite-wave-number selection and commensurate stripe branches;
- two-dimensional stripe, spot, and labyrinth morphologies;
- microscopic Bragg order, Gaussian covariance dynamics, and opposite-momentum entanglement;
- Lean 4 verification of the finite-dimensional identities used in the construction.

## Repository guide

| Path | Contents |
|---|---|
| [`display/`](display/) | Lightweight public gallery and direct links to the stripe, spot, and labyrinth movies. |
| [`reproduction/`](reproduction/) | Numerical code, compact reference data, figure generation, and scientific verification. |
| [`lean4 formalization/`](lean4%20formalization/) | Lean 4 project for the spectral design, parameter map, stripe coefficients, and Gaussian formulas. |

The display and reproduction workflows are intentionally separate: `display/` is optimized for readers, while `reproduction/` contains the scientific calculations and verification tools.

## Public display

Open [`display/`](display/) to view the reference movies. The animation above is a lightweight loop prepared for the repository landing page; clicking it opens the full-resolution Labyrinth MP4.

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
