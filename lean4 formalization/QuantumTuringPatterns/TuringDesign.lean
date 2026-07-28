import QuantumTuringPatterns.LatticeSymbol

set_option autoImplicit false

/-!
# Algebra for the Turing design theorem

This module formalizes the finite-dimensional part of the paper's
Turing-design theorem.  It proves that the determinant of the Fourier
symbol is a positive quadratic with a quadratic zero at the prescribed
nonzero lattice symbol.  It also records why scalar transport cannot
produce the same finite-wave-number destabilization.
-/

namespace QuantumTuringPatterns

noncomputable section

/-- Determinant of `J - ω diag(Dq,Dp)` for
`J = [[a,Ω],[-Ω,-b]]`. -/
def designDet (a b Dq Dp Ω ω : ℝ) : ℝ :=
  Dq * Dp * ω ^ 2 + (b * Dq - a * Dp) * ω + (Ω ^ 2 - a * b)

/-- Vertex of the determinant polynomial. -/
def criticalSymbol (a b Dq Dp : ℝ) : ℝ :=
  (a * Dp - b * Dq) / (2 * Dq * Dp)

/-- Frequency-square selected by the double-contact design. -/
def designedOmegaSq (a b Dq Dp : ℝ) : ℝ :=
  a * b + (a * Dp - b * Dq) ^ 2 / (4 * Dq * Dp)

/-- Exact completed-square identity behind Theorem `the corresponding article result`. -/
theorem designDet_factorization
    {a b Dq Dp Ω ω : ℝ}
    (hDq : Dq ≠ 0) (hDp : Dp ≠ 0)
    (hΩ : Ω ^ 2 = designedOmegaSq a b Dq Dp) :
    designDet a b Dq Dp Ω ω =
      Dq * Dp * (ω - criticalSymbol a b Dq Dp) ^ 2 := by
  rw [hΩ]
  unfold designDet designedOmegaSq criticalSymbol
  field_simp [hDq, hDp]
  ring

/-- The designed determinant vanishes at the selected symbol. -/
theorem designDet_at_critical
    {a b Dq Dp Ω : ℝ}
    (hDq : Dq ≠ 0) (hDp : Dp ≠ 0)
    (hΩ : Ω ^ 2 = designedOmegaSq a b Dq Dp) :
    designDet a b Dq Dp Ω (criticalSymbol a b Dq Dp) = 0 := by
  rw [designDet_factorization hDq hDp hΩ]
  ring

/-- With positive transport, the determinant is nonnegative for every symbol. -/
theorem designDet_nonneg
    {a b Dq Dp Ω ω : ℝ}
    (hDq : 0 < Dq) (hDp : 0 < Dp)
    (hΩ : Ω ^ 2 = designedOmegaSq a b Dq Dp) :
    0 ≤ designDet a b Dq Dp Ω ω := by
  rw [designDet_factorization (ne_of_gt hDq) (ne_of_gt hDp) hΩ]
  positivity

/-- The quadratic contact is unique when both transport coefficients are positive. -/
theorem designDet_eq_zero_iff
    {a b Dq Dp Ω ω : ℝ}
    (hDq : 0 < Dq) (hDp : 0 < Dp)
    (hΩ : Ω ^ 2 = designedOmegaSq a b Dq Dp) :
    designDet a b Dq Dp Ω ω = 0 ↔
      ω = criticalSymbol a b Dq Dp := by
  rw [designDet_factorization (ne_of_gt hDq) (ne_of_gt hDp) hΩ]
  constructor
  · intro h
    have hprod : Dq * Dp ≠ 0 := mul_ne_zero (ne_of_gt hDq) (ne_of_gt hDp)
    have hsquare : (ω - criticalSymbol a b Dq Dp) ^ 2 = 0 :=
      (mul_eq_zero.mp h).resolve_left hprod
    nlinarith
  · intro h
    rw [h]
    ring

/-- The homogeneous determinant is strictly positive when the designed
contact lies at a nonzero symbol. -/
theorem designDet_homogeneous_pos
    {a b Dq Dp Ω : ℝ}
    (hDq : 0 < Dq) (hDp : 0 < Dp)
    (hΩ : Ω ^ 2 = designedOmegaSq a b Dq Dp)
    (hcrit : 0 < criticalSymbol a b Dq Dp) :
    0 < designDet a b Dq Dp Ω 0 := by
  rw [designDet_factorization (ne_of_gt hDq) (ne_of_gt hDp) hΩ]
  have hcritne : criticalSymbol a b Dq Dp ≠ 0 := ne_of_gt hcrit
  have hsquare : 0 < (0 - criticalSymbol a b Dq Dp) ^ 2 := by
    positivity
  positivity

/-- Trace of the Fourier symbol. -/
def designTrace (a b Dq Dp ω : ℝ) : ℝ :=
  a - b - (Dq + Dp) * ω

/-- At a nonnegative symbol, positive transport makes an already negative
reaction trace remain negative. -/
theorem designTrace_neg
    {a b Dq Dp ω : ℝ}
    (hba : a < b) (hDq : 0 ≤ Dq) (hDp : 0 ≤ Dp) (hω : 0 ≤ ω) :
    designTrace a b Dq Dp ω < 0 := by
  unfold designTrace
  have htransport : 0 ≤ (Dq + Dp) * ω := by positivity
  linarith

/-- Determinant of a stable `2×2` reaction block after a scalar shift `s I`. -/
def scalarShiftDet (trace det s : ℝ) : ℝ := det - s * trace + s ^ 2

/-- A nonnegative scalar transport shift preserves determinant positivity
when the unshifted trace is negative and determinant is positive. -/
theorem scalarShiftDet_pos
    {trace det s : ℝ}
    (htrace : trace < 0) (hdet : 0 < det) (hs : 0 ≤ s) :
    0 < scalarShiftDet trace det s := by
  unfold scalarShiftDet
  have hst : 0 ≤ -(s * trace) := by positivity
  have hsquare : 0 ≤ s ^ 2 := sq_nonneg s
  linarith

/-- A scalar shift makes the trace strictly more negative. -/
theorem scalarShiftTrace_neg
    {trace s : ℝ} (htrace : trace < 0) (hs : 0 ≤ s) :
    trace - 2 * s < 0 := by
  linarith

/-- Algebraic form of “equal diffusion shifts every eigenvalue to the left”:
for a stable `2×2` block, both Hurwitz inequalities persist. -/
theorem scalar_transport_preserves_hurwitz_inequalities
    {trace det D ω : ℝ}
    (htrace : trace < 0) (hdet : 0 < det)
    (hD : 0 ≤ D) (hω : 0 ≤ ω) :
    trace - 2 * (D * ω) < 0 ∧
      0 < scalarShiftDet trace det (D * ω) := by
  have hs : 0 ≤ D * ω := mul_nonneg hD hω
  exact ⟨scalarShiftTrace_neg htrace hs, scalarShiftDet_pos htrace hdet hs⟩

end

end QuantumTuringPatterns
