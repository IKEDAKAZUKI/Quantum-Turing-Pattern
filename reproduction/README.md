# Numerical reproduction

This directory contains the numerical workflows and compact reference data used in **Quantum Turing Patterns**. It is independent of the static display in `../display/`.

## Environment

```bash
conda env create -f environment.yml
conda activate quantum-turing-patterns
```

A pip installation may use `requirements.txt`.

## Verify the distributed results

```bash
python verify.py
```

The reference verifier checks the stripe branch, Lyapunov residual and error bound, physicality, opposite-momentum entanglement, transport controls, continuation data, and the Spot and Labyrinth finite-time calculations.

## Regenerate results

Short smoke test:

```bash
python run.py --mode smoke --force
python verify.py --mode smoke
```

Paper figures and the compact stripe/Gaussian workflow:

```bash
python run.py --mode quick --force
python verify.py --mode quick
```

Complete Spot and Labyrinth trajectories:

```bash
python run.py --mode full --jobs 2 --force
python verify.py --mode full
```

Generated output is written next to the repository by default. Use `--out PATH` to select another directory outside `reproduction/`.

## Contents

- `src/` — numerical model, continuation, covariance, figure, and verification code.
- `data/reference/` — compact reference matrices, tables, and endpoint fields.
- `run.py` — smoke, quick, and full workflows.
- `verify.py` — checksum and scientific verification.

The static videos in `../display/` are reader-facing illustrations and are not part of the numerical verification workflow.
