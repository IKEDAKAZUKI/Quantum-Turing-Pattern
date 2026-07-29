import QuantumTuringPatterns.LindbladParameters

set_option autoImplicit false

/-!
# Local algebra behind the classical a-priori bound

The global-existence argument evaluates the radial derivative at a lattice
site where `q²+p²` is maximal. The ODE/Dini-derivative step is proved in the
paper; this module verifies the sitewise inequalities used in the logistic
estimate.
-/

namespace QuantumTuringPatterns

noncomputable section

/-- At a site maximizing `q²+p²`, the `q`-Laplacian contribution is
bounded by `2 p²` for a pair of nearest neighbours. -/
theorem q_laplacian_at_pairwise_max
    {q p qPlus pPlus qMinus pMinus : ℝ}
    (hPlus : qPlus ^ 2 + pPlus ^ 2 ≤ q ^ 2 + p ^ 2)
    (hMinus : qMinus ^ 2 + pMinus ^ 2 ≤ q ^ 2 + p ^ 2) :
    2 * q * (qPlus + qMinus - 2 * q) ≤ 2 * p ^ 2 := by
  nlinarith [sq_nonneg (q - qPlus), sq_nonneg (q - qMinus),
    sq_nonneg pPlus, sq_nonneg pMinus]

/-- The analogous bound for the `p`-Laplacian contribution. -/
theorem p_laplacian_at_pairwise_max
    {q p qPlus pPlus qMinus pMinus : ℝ}
    (hPlus : qPlus ^ 2 + pPlus ^ 2 ≤ q ^ 2 + p ^ 2)
    (hMinus : qMinus ^ 2 + pMinus ^ 2 ≤ q ^ 2 + p ^ 2) :
    2 * p * (pPlus + pMinus - 2 * p) ≤ 2 * q ^ 2 := by
  nlinarith [sq_nonneg (p - pPlus), sq_nonneg (p - pMinus),
    sq_nonneg qPlus, sq_nonneg qMinus]

/-- Directional diffusion estimate used in the global a-priori bound. -/
theorem weighted_diffusion_at_pairwise_max
    {q p qPlus pPlus qMinus pMinus Dq Dp : ℝ}
    (hPlus : qPlus ^ 2 + pPlus ^ 2 ≤ q ^ 2 + p ^ 2)
    (hMinus : qMinus ^ 2 + pMinus ^ 2 ≤ q ^ 2 + p ^ 2)
    (hDq : 0 ≤ Dq) (hDp : 0 ≤ Dp) :
    2 * Dq * q * (qPlus + qMinus - 2 * q) +
        2 * Dp * p * (pPlus + pMinus - 2 * p) ≤
      2 * max Dq Dp * (q ^ 2 + p ^ 2) := by
  have hq := q_laplacian_at_pairwise_max hPlus hMinus
  have hp := p_laplacian_at_pairwise_max hPlus hMinus
  have hqD :
      Dq * (2 * q * (qPlus + qMinus - 2 * q)) ≤ Dq * (2 * p ^ 2) :=
    mul_le_mul_of_nonneg_left hq hDq
  have hpD :
      Dp * (2 * p * (pPlus + pMinus - 2 * p)) ≤ Dp * (2 * q ^ 2) :=
    mul_le_mul_of_nonneg_left hp hDp
  have hqMax : Dq * (2 * p ^ 2) ≤ max Dq Dp * (2 * p ^ 2) :=
    mul_le_mul_of_nonneg_right (le_max_left Dq Dp) (by positivity)
  have hpMax : Dp * (2 * q ^ 2) ≤ max Dq Dp * (2 * q ^ 2) :=
    mul_le_mul_of_nonneg_right (le_max_right Dq Dp) (by positivity)
  calc
    2 * Dq * q * (qPlus + qMinus - 2 * q) +
        2 * Dp * p * (pPlus + pMinus - 2 * p) =
        Dq * (2 * q * (qPlus + qMinus - 2 * q)) +
          Dp * (2 * p * (pPlus + pMinus - 2 * p)) := by ring
    _ ≤ Dq * (2 * p ^ 2) + Dp * (2 * q ^ 2) := add_le_add hqD hpD
    _ ≤ max Dq Dp * (2 * p ^ 2) + max Dq Dp * (2 * q ^ 2) :=
      add_le_add hqMax hpMax
    _ = 2 * max Dq Dp * (q ^ 2 + p ^ 2) := by ring

/-- The Hamiltonian rotation cancels exactly from the radial derivative. -/
theorem radial_reaction_identity
    (a b Ω ν q p : ℝ) :
    2 * q * (a * q + Ω * p - ν * (q ^ 2 + p ^ 2) * q) +
        2 * p * (-Ω * q - b * p - ν * (q ^ 2 + p ^ 2) * p) =
      2 * a * q ^ 2 - 2 * b * p ^ 2 -
        2 * ν * (q ^ 2 + p ^ 2) ^ 2 := by
  ring

/-- Dropping the favorable `-2 b p²` term and replacing `q²` by
`q²+p²` gives the reaction part of the logistic estimate. -/
theorem radial_reaction_upper_bound
    {a b Ω ν q p : ℝ} (ha : 0 ≤ a) (hb : 0 ≤ b) :
    2 * q * (a * q + Ω * p - ν * (q ^ 2 + p ^ 2) * q) +
        2 * p * (-Ω * q - b * p - ν * (q ^ 2 + p ^ 2) * p) ≤
      2 * a * (q ^ 2 + p ^ 2) -
        2 * ν * (q ^ 2 + p ^ 2) ^ 2 := by
  rw [radial_reaction_identity]
  have hap : 0 ≤ 2 * a * p ^ 2 := by positivity
  have hbp : 0 ≤ 2 * b * p ^ 2 := by positivity
  linarith

/-- Scalar logistic right-hand side. -/
def logisticRhs (C ν M : ℝ) : ℝ := C * M - 2 * ν * M ^ 2

/-- Above the barrier `C/(2ν)`, the logistic right-hand side is nonpositive. -/
theorem logisticRhs_nonpos_above_barrier
    {C ν M : ℝ} (hC : 0 ≤ C) (hν : 0 < ν) (hM : 0 ≤ M)
    (hbarrier : C / (2 * ν) ≤ M) :
    logisticRhs C ν M ≤ 0 := by
  have hden : 0 < 2 * ν := by positivity
  have hCM := (div_le_iff₀ hden).mp hbarrier
  unfold logisticRhs
  nlinarith

/-- Combined one-direction estimate: reaction plus diffusion has the
paper's logistic form with linear coefficient `2a+2 max(Dq,Dp)`. -/
theorem one_direction_logistic_bound
    {a b Ω ν q p qPlus pPlus qMinus pMinus Dq Dp : ℝ}
    (hPlus : qPlus ^ 2 + pPlus ^ 2 ≤ q ^ 2 + p ^ 2)
    (hMinus : qMinus ^ 2 + pMinus ^ 2 ≤ q ^ 2 + p ^ 2)
    (ha : 0 ≤ a) (hb : 0 ≤ b) (hDq : 0 ≤ Dq) (hDp : 0 ≤ Dp) :
    2 * q * (a * q + Ω * p - ν * (q ^ 2 + p ^ 2) * q) +
        2 * p * (-Ω * q - b * p - ν * (q ^ 2 + p ^ 2) * p) +
        2 * Dq * q * (qPlus + qMinus - 2 * q) +
        2 * Dp * p * (pPlus + pMinus - 2 * p) ≤
      (2 * a + 2 * max Dq Dp) * (q ^ 2 + p ^ 2) -
        2 * ν * (q ^ 2 + p ^ 2) ^ 2 := by
  have hr := radial_reaction_upper_bound (Ω := Ω) (ν := ν) ha hb
  have hd := weighted_diffusion_at_pairwise_max hPlus hMinus hDq hDp
  nlinarith

end

end QuantumTuringPatterns
