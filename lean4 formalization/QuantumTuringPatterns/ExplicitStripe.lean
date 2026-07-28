import QuantumTuringPatterns.TuringDesign

set_option autoImplicit false

/-!
# Algebra for the explicit commensurate stripe

This module verifies the algebra used in the explicit family:

* `Dq = 1`, `Dp = 3+2√3`;
* critical symbol `ω⋆ = 2-√3`, corresponding to `k⋆ = π/6`;
* determinant factorization and critical trace;
* the stated right/left critical vectors;
* the cubic coefficient, leading branch amplitude, and radial eigenvalue.

The analytic Lyapunov--Schmidt existence theorem itself is outside this
finite-dimensional algebraic calculation; see `SCOPE.md`.
-/

namespace QuantumTuringPatterns

noncomputable section

/-- Explicit longitudinal `p`-diffusion. -/
def explicitDp : ℝ := 3 + 2 * sqrt3

/-- Critical scalar lattice symbol. -/
def explicitCriticalSymbol : ℝ := 2 - sqrt3

/-- The paper uses `Ω_λ² = 2√3-λ`. -/
def explicitOmegaSq (λ : ℝ) : ℝ := 2 * sqrt3 - λ

/-- Determinant of the explicit longitudinal Fourier block. -/
def explicitDet (λ ω : ℝ) : ℝ :=
  (1 - ω) * (-3 - explicitDp * ω) + explicitOmegaSq λ

/-- Trace of the explicit longitudinal Fourier block. -/
def explicitTrace (ω : ℝ) : ℝ :=
  (1 - ω) + (-3 - explicitDp * ω)

lemma explicitDp_pos : 0 < explicitDp := by
  unfold explicitDp
  positivity

lemma explicitCriticalSymbol_eq_latticeSymbol :
    explicitCriticalSymbol = latticeSymbol (Real.pi / 6) := by
  symm
  exact latticeSymbol_pi_div_six

lemma explicitCriticalSymbol_in_band :
    explicitCriticalSymbol ∈ Set.Ioo (0 : ℝ) 4 := by
  simpa [explicitCriticalSymbol] using two_minus_sqrt3_mem_Ioo

/-- Exact determinant factorization used in the proof of `the corresponding article result`. -/
theorem explicitDet_factorization (λ ω : ℝ) :
    explicitDet λ ω =
      explicitDp * (ω - explicitCriticalSymbol) ^ 2 - λ := by
  unfold explicitDet explicitDp explicitOmegaSq explicitCriticalSymbol
  linear_combination -(2 * sqrt3 + 4 * ω - 5) * sqrt3_sq

lemma explicitDet_at_critical (λ : ℝ) :
    explicitDet λ explicitCriticalSymbol = -λ := by
  rw [explicitDet_factorization]
  ring

lemma explicitDet_at_threshold_nonneg (ω : ℝ) :
    0 ≤ explicitDet 0 ω := by
  rw [explicitDet_factorization]
  simp only [sub_zero]
  positivity

lemma explicitDet_threshold_eq_zero_iff (ω : ℝ) :
    explicitDet 0 ω = 0 ↔ ω = explicitCriticalSymbol := by
  rw [explicitDet_factorization]
  simp only [sub_zero]
  constructor
  · intro h
    have hDp : explicitDp ≠ 0 := ne_of_gt explicitDp_pos
    have hsquare : (ω - explicitCriticalSymbol) ^ 2 = 0 :=
      (mul_eq_zero.mp h).resolve_left hDp
    nlinarith
  · intro h
    rw [h]
    ring

lemma explicitTrace_at_critical :
    explicitTrace explicitCriticalSymbol = -4 := by
  unfold explicitTrace explicitCriticalSymbol explicitDp
  nlinarith [sqrt3_sq]

lemma explicitHomogeneousTrace : explicitTrace 0 = -2 := by
  norm_num [explicitTrace]

lemma explicitHomogeneousDet_at_threshold :
    explicitDet 0 0 = 2 * sqrt3 - 3 := by
  unfold explicitDet explicitOmegaSq
  ring

lemma explicitHomogeneousDet_pos : 0 < explicitDet 0 0 := by
  rw [explicitHomogeneousDet_at_threshold]
  nlinarith [three_halves_lt_sqrt3]

/-- The positive number appearing in the critical eigenvectors. -/
def beta : ℝ := (sqrt3 - 1) / explicitOmega0

lemma beta_pos : 0 < beta := by
  unfold beta
  exact div_pos (sub_pos.mpr one_lt_sqrt3) explicitOmega0_pos

lemma beta_mul_explicitOmega0 : beta * explicitOmega0 = sqrt3 - 1 := by
  simp [beta, explicitOmega0_ne_zero]

lemma sqrt3_sub_one_sq : (sqrt3 - 1) ^ 2 = 2 * (2 - sqrt3) := by
  nlinarith [sqrt3_sq]

lemma critical_second_balance :
    (3 + sqrt3) * beta = explicitOmega0 := by
  unfold beta
  field_simp [explicitOmega0_ne_zero]
  nlinarith [sqrt3_sq, explicitOmega0_sq]

/-- First component of `A⋆ r = 0` for `r=(1,-β)`. -/
lemma rightCritical_first :
    (1 - explicitCriticalSymbol) - explicitOmega0 * beta = 0 := by
  rw [mul_comm explicitOmega0 beta, beta_mul_explicitOmega0]
  unfold explicitCriticalSymbol
  ring

/-- Second component of `A⋆ r = 0` for `r=(1,-β)`. -/
lemma rightCritical_second :
    -explicitOmega0 + (-3 - explicitDp * explicitCriticalSymbol) * (-beta) = 0 := by
  have hentry : -3 - explicitDp * explicitCriticalSymbol = -(3 + sqrt3) := by
    unfold explicitDp explicitCriticalSymbol
    nlinarith [sqrt3_sq]
  rw [hentry]
  rw [show (-(3 + sqrt3)) * (-beta) = (3 + sqrt3) * beta by ring]
  rw [critical_second_balance]
  ring

/-- First component of `ℓᵀ A⋆ = 0` for `ℓ=(1,β)`. -/
lemma leftCritical_first :
    (1 - explicitCriticalSymbol) + beta * (-explicitOmega0) = 0 := by
  rw [show beta * (-explicitOmega0) = -(beta * explicitOmega0) by ring]
  rw [beta_mul_explicitOmega0]
  unfold explicitCriticalSymbol
  ring

/-- Second component of `ℓᵀ A⋆ = 0` for `ℓ=(1,β)`. -/
lemma leftCritical_second :
    explicitOmega0 + beta * (-3 - explicitDp * explicitCriticalSymbol) = 0 := by
  have hentry : -3 - explicitDp * explicitCriticalSymbol = -(3 + sqrt3) := by
    unfold explicitDp explicitCriticalSymbol
    nlinarith [sqrt3_sq]
  rw [hentry]
  rw [show beta * (-(3 + sqrt3)) = -((3 + sqrt3) * beta) by ring]
  rw [critical_second_balance]
  ring

/-- Squared Euclidean norm of the right critical vector. -/
lemma one_add_beta_sq : 1 + beta ^ 2 = 2 / sqrt3 := by
  unfold beta
  rw [div_pow, explicitOmega0_sq, sqrt3_sub_one_sq]
  field_simp [sqrt3_ne_zero]
  ring

lemma beta_sq_lt_one : beta ^ 2 < 1 := by
  rw [show beta ^ 2 = 2 / sqrt3 - 1 by nlinarith [one_add_beta_sq]]
  have h : (2 : ℝ) / sqrt3 < 2 := by
    apply (div_lt_iff₀ sqrt3_pos).2
    nlinarith [one_lt_sqrt3]
  linarith

lemma left_right_pairing_pos : 0 < 1 - beta ^ 2 := by
  linarith [beta_sq_lt_one]

/-- Resonant cubic coefficient before the crossing-slope normalization. -/
def resonantCubicCoefficient (ν : ℝ) : ℝ :=
  3 * ν * (1 + beta ^ 2)

lemma resonantCubicCoefficient_eq (ν : ℝ) :
    resonantCubicCoefficient ν = 2 * sqrt3 * ν := by
  unfold resonantCubicCoefficient
  rw [one_add_beta_sq, two_div_sqrt3]
  ring

/-- Leading squared branch amplitude in the complex Fourier convention. -/
def leadingAmplitudeSq (λ ν : ℝ) : ℝ :=
  λ / (8 * sqrt3 * ν)

lemma leadingAmplitudeSq_pos
    {λ ν : ℝ} (hλ : 0 < λ) (hν : 0 < ν) :
    0 < leadingAmplitudeSq λ ν := by
  unfold leadingAmplitudeSq
  positivity

/-- Cubic part of the reflection-fixed reduced stationary equation. -/
def reducedStationaryFactor (λ ν B : ℝ) : ℝ :=
  λ / 4 - 2 * sqrt3 * ν * B ^ 2

/-- The leading amplitude law solves the leading reduced stationary equation. -/
theorem leadingAmplitude_solves_reduced_equation
    {λ ν B : ℝ} (hν : ν ≠ 0)
    (hB : B ^ 2 = leadingAmplitudeSq λ ν) :
    reducedStationaryFactor λ ν B = 0 := by
  rw [hB]
  unfold reducedStationaryFactor leadingAmplitudeSq
  field_simp [sqrt3_ne_zero, hν]
  ring

/-- Derivative of the leading reflection-fixed amplitude vector field. -/
def reducedRadialDerivative (λ ν B : ℝ) : ℝ :=
  λ / 4 - 6 * sqrt3 * ν * B ^ 2

/-- Substituting the branch amplitude gives the paper's leading radial law `-λ/2`. -/
theorem reducedRadialDerivative_at_branch
    {λ ν B : ℝ} (hν : ν ≠ 0)
    (hB : B ^ 2 = leadingAmplitudeSq λ ν) :
    reducedRadialDerivative λ ν B = -λ / 2 := by
  rw [hB]
  unfold reducedRadialDerivative leadingAmplitudeSq
  field_simp [sqrt3_ne_zero, hν]
  ring

end

end QuantumTuringPatterns
