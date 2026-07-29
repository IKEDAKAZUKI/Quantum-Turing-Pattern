# Lean 4 formalization

This project verifies the finite-dimensional algebraic identities used in **Quantum Turing Patterns**.

## Build

```bash
lake build
```

The Lean toolchain and mathlib dependency are pinned by `lean-toolchain` and `lake-manifest.json`. `QuantumTuringPatterns.lean` imports all modules.

## Verified identities

- lattice symbol, selected wave number, and primitive period;
- determinant factorization and critical vectors;
- cubic coefficient, leading stripe amplitude, and reduced radial derivative;
- Lindblad-to-reaction parameter map;
- local diffusion and logistic inequalities;
- homogeneous Lyapunov identities and stability ratio;
- partial-transpose formulas, thermal NPT threshold, and transport selection;
- the rational inequality used in the strict NPT sign bound.

The analytic cutoff estimates, semigroup bounds, Lyapunov-Schmidt construction, Bragg theorem, covariance convergence, and volume limit are proved in the paper rather than formalized here.

The source contains no `sorry`, `admit`, custom `axiom`, or `unsafe` declarations.
