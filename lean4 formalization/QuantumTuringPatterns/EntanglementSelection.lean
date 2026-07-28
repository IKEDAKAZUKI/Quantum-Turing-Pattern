import QuantumTuringPatterns.Gaussian

set_option autoImplicit false

/-!
# Finite-wave-number selection of homogeneous NPT correlations

For the explicit differential-transport stripe family, differentiating
`η(ω)²` produces a positive prefactor times `selectionNumerator λ ω`.
This module verifies the exact affine factorization of that numerator
and hence the unique stationary symbol stated in the paper.
-/

namespace QuantumTuringPatterns

noncomputable section

/-- Explicit longitudinal damping in the homogeneous Gaussian block. -/
def stripeKappa (ω : ℝ) : ℝ :=
  2 + 2 * (2 + sqrt3) * ω

/-- Explicit longitudinal squeezing coefficient. -/
def stripeG (ω : ℝ) : ℝ :=
  2 + (1 + sqrt3) * ω

/-- `Ω_λ²` in the explicit family. -/
def stripeOmegaSq (λ : ℝ) : ℝ := 2 * sqrt3 - λ

/-- The bracket occurring in the derivative of `η(ω)²`. -/
def selectionNumerator (λ ω : ℝ) : ℝ :=
  (1 + sqrt3) * (stripeKappa ω ^ 2 + 4 * stripeOmegaSq λ) -
    stripeG ω * stripeKappa ω * (2 * (2 + sqrt3))

/-- Selected homogeneous entanglement symbol on the stable side. -/
def omegaEnt (λ : ℝ) : ℝ :=
  2 - sqrt3 - (2 / sqrt3 - 1) * λ

lemma omegaEnt_at_zero : omegaEnt 0 = explicitCriticalSymbol := by
  simp [omegaEnt, explicitCriticalSymbol]

lemma omegaEnt_slope_pos : 0 < 2 / sqrt3 - 1 := by
  apply sub_pos.mpr
  apply (lt_div_iff₀ sqrt3_pos).2
  simpa using sqrt3_lt_two

/-- Exact factorization of the derivative numerator. -/
theorem selectionNumerator_factorization (λ ω : ℝ) :
    selectionNumerator λ ω =
      4 * (5 * sqrt3 + 9) * (omegaEnt λ - ω) := by
  unfold selectionNumerator stripeKappa stripeG stripeOmegaSq omegaEnt
  rw [two_div_sqrt3]
  linear_combination
    (-(4 : ℝ) / 3 * (-10 * λ + 3 * ω - 21)) * sqrt3_sq

lemma selection_prefactor_pos : 0 < 4 * (5 * sqrt3 + 9) := by
  positivity

/-- The derivative numerator has exactly one zero. -/
theorem selectionNumerator_eq_zero_iff (λ ω : ℝ) :
    selectionNumerator λ ω = 0 ↔ ω = omegaEnt λ := by
  rw [selectionNumerator_factorization]
  have hcoef : 4 * (5 * sqrt3 + 9) ≠ 0 := ne_of_gt selection_prefactor_pos
  constructor
  · intro h
    have hdiff : omegaEnt λ - ω = 0 :=
      (mul_eq_zero.mp h).resolve_left hcoef
    linarith
  · intro h
    rw [h]
    ring

lemma selectionNumerator_pos_iff (λ ω : ℝ) :
    0 < selectionNumerator λ ω ↔ ω < omegaEnt λ := by
  rw [selectionNumerator_factorization]
  constructor
  · intro h
    by_contra hnot
    have hle : omegaEnt λ ≤ ω := le_of_not_gt hnot
    have hdiff : omegaEnt λ - ω ≤ 0 := sub_nonpos.mpr hle
    have hnonpos :
        4 * (5 * sqrt3 + 9) * (omegaEnt λ - ω) ≤ 0 :=
      mul_nonpos_of_nonneg_of_nonpos (le_of_lt selection_prefactor_pos) hdiff
    linarith
  · intro h
    exact mul_pos selection_prefactor_pos (sub_pos.mpr h)

lemma selectionNumerator_neg_iff (λ ω : ℝ) :
    selectionNumerator λ ω < 0 ↔ omegaEnt λ < ω := by
  rw [selectionNumerator_factorization]
  have hcoef : 0 < 4 * (5 * sqrt3 + 9) := selection_prefactor_pos
  constructor
  · intro h
    by_contra hnot
    have hle : ω ≤ omegaEnt λ := le_of_not_gt hnot
    have hdiff : 0 ≤ omegaEnt λ - ω := sub_nonneg.mpr hle
    have hnonneg :
        0 ≤ 4 * (5 * sqrt3 + 9) * (omegaEnt λ - ω) :=
      mul_nonneg (le_of_lt hcoef) hdiff
    linarith
  · intro h
    exact mul_neg_of_pos_of_neg hcoef (sub_neg.mpr h)

/-- For negative `λ`, the maximizing symbol is displaced to the right of
its Turing-threshold value. -/
lemma omegaEnt_gt_critical_of_lambda_neg {λ : ℝ} (hλ : λ < 0) :
    explicitCriticalSymbol < omegaEnt λ := by
  unfold omegaEnt explicitCriticalSymbol
  have hslope := omegaEnt_slope_pos
  have hprod : (2 / sqrt3 - 1) * λ < 0 :=
    mul_neg_of_pos_of_neg hslope hλ
  linarith

end

end QuantumTuringPatterns
