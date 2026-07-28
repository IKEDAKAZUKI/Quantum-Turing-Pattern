import QuantumTuringPatterns.Basic

set_option autoImplicit false

/-!
# Nearest-neighbour lattice symbol

For a one-dimensional nearest-neighbour Laplacian, the scalar Fourier
symbol is `ω(k) = 2(1-cos k)`.  The identities here connect the exact
commensurate wave number `π/6` to `2-√3`.
-/

namespace QuantumTuringPatterns

noncomputable section

/-- Scalar symbol of the one-dimensional nearest-neighbour lattice Laplacian. -/
def latticeSymbol (k : ℝ) : ℝ := 2 * (1 - Real.cos k)

lemma latticeSymbol_nonneg (k : ℝ) : 0 ≤ latticeSymbol k := by
  unfold latticeSymbol
  nlinarith [Real.cos_le_one k]

lemma latticeSymbol_le_four (k : ℝ) : latticeSymbol k ≤ 4 := by
  unfold latticeSymbol
  nlinarith [Real.neg_one_le_cos k]

lemma latticeSymbol_mem_Icc (k : ℝ) : latticeSymbol k ∈ Set.Icc (0 : ℝ) 4 := by
  exact ⟨latticeSymbol_nonneg k, latticeSymbol_le_four k⟩

/-- Exact Fourier-symbol action on a cosine mode. -/
lemma cosine_laplacian_symbol (x k : ℝ) :
    Real.cos (x + k) + Real.cos (x - k) - 2 * Real.cos x =
      -latticeSymbol k * Real.cos x := by
  unfold latticeSymbol
  rw [Real.cos_add, Real.cos_sub]
  ring

/-- The paper's commensurate wave number has symbol `2-√3`. -/
lemma latticeSymbol_pi_div_six :
    latticeSymbol (Real.pi / 6) = 2 - sqrt3 := by
  unfold latticeSymbol sqrt3
  rw [Real.cos_pi_div_six]
  ring

lemma two_minus_sqrt3_mem_Ioo :
    (2 - sqrt3 : ℝ) ∈ Set.Ioo 0 4 := by
  exact ⟨two_minus_sqrt3_pos, two_minus_sqrt3_lt_four⟩

/-- The primitive phase period is twelve lattice sites. -/
lemma twelve_mul_pi_div_six : (12 : ℝ) * (Real.pi / 6) = 2 * Real.pi := by
  ring

end

end QuantumTuringPatterns
