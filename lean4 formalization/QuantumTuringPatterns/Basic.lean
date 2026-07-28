import Mathlib

set_option autoImplicit false

/-!
# Basic constants for the explicit Quantum Turing pattern model

The paper's explicit construction repeatedly uses `√3` and
`√(2√3)`.  This file isolates the elementary algebraic facts used by
all later modules.
-/

namespace QuantumTuringPatterns

noncomputable section

/-- The exact constant `√3`. -/
def sqrt3 : ℝ := Real.sqrt 3

lemma sqrt3_nonneg : 0 ≤ sqrt3 := by
  exact Real.sqrt_nonneg 3

lemma sqrt3_pos : 0 < sqrt3 := by
  unfold sqrt3
  positivity

lemma sqrt3_ne_zero : sqrt3 ≠ 0 := ne_of_gt sqrt3_pos

lemma sqrt3_sq : sqrt3 ^ 2 = 3 := by
  simpa [sqrt3] using Real.sq_sqrt (show (0 : ℝ) ≤ 3 by norm_num)

lemma one_lt_sqrt3 : 1 < sqrt3 := by
  nlinarith [sqrt3_sq, sqrt3_nonneg]

lemma three_halves_lt_sqrt3 : (3 : ℝ) / 2 < sqrt3 := by
  nlinarith [sqrt3_sq, sqrt3_nonneg]

lemma sqrt3_lt_two : sqrt3 < 2 := by
  nlinarith [sqrt3_sq, sqrt3_nonneg]

lemma two_minus_sqrt3_pos : 0 < 2 - sqrt3 := by
  linarith [sqrt3_lt_two]

lemma two_minus_sqrt3_lt_four : 2 - sqrt3 < 4 := by
  linarith [sqrt3_pos]

/-- The positive off-diagonal frequency at the explicit Turing point. -/
def explicitOmega0 : ℝ := Real.sqrt (2 * sqrt3)

lemma explicitOmega0_nonneg : 0 ≤ explicitOmega0 := by
  exact Real.sqrt_nonneg _

lemma explicitOmega0_pos : 0 < explicitOmega0 := by
  unfold explicitOmega0
  positivity

lemma explicitOmega0_ne_zero : explicitOmega0 ≠ 0 := ne_of_gt explicitOmega0_pos

lemma explicitOmega0_sq : explicitOmega0 ^ 2 = 2 * sqrt3 := by
  simpa [explicitOmega0] using
    Real.sq_sqrt (show (0 : ℝ) ≤ 2 * sqrt3 by positivity)

lemma two_div_sqrt3 : (2 : ℝ) / sqrt3 = 2 * sqrt3 / 3 := by
  apply (div_eq_iff sqrt3_ne_zero).2
  nlinarith [sqrt3_sq]

end

end QuantumTuringPatterns
