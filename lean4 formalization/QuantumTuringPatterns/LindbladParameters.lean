import QuantumTuringPatterns.ExplicitStripe

set_option autoImplicit false

/-!
# Lindblad-to-reaction parameter map

The paper chooses local squeezing/damping and bond channels so that the
first-moment drift has prescribed reaction coefficients `a,b`, cubic
coefficient `ν`, and directional diffusion coefficients `Dq,Dp`.
This file proves the inverse linear identities exactly.
-/

namespace QuantumTuringPatterns

noncomputable section

/-- Local squeezing parameter. -/
def epsilonOfReaction (a b : ℝ) : ℝ := (a + b) / 2

/-- Local one-photon damping parameter. -/
def kappaOfReaction (a b : ℝ) : ℝ := b - a

/-- Symmetric bond-channel coefficient. -/
def KOfDiffusion (Dq Dp : ℝ) : ℝ := (Dq + Dp) / 2

/-- Directional squeezing-bond coefficient. -/
def KPrimeOfDiffusion (Dq Dp : ℝ) : ℝ := (Dq - Dp) / 2

/-- Two-photon damping coefficient corresponding to the radial cubic drift. -/
def gammaOfCubic (ν : ℝ) : ℝ := 2 * ν

lemma reaction_q_gain_recovered (a b : ℝ) :
    epsilonOfReaction a b - kappaOfReaction a b / 2 = a := by
  unfold epsilonOfReaction kappaOfReaction
  ring

lemma reaction_p_damping_recovered (a b : ℝ) :
    epsilonOfReaction a b + kappaOfReaction a b / 2 = b := by
  unfold epsilonOfReaction kappaOfReaction
  ring

lemma q_diffusion_recovered (Dq Dp : ℝ) :
    KOfDiffusion Dq Dp + KPrimeOfDiffusion Dq Dp = Dq := by
  unfold KOfDiffusion KPrimeOfDiffusion
  ring

lemma p_diffusion_recovered (Dq Dp : ℝ) :
    KOfDiffusion Dq Dp - KPrimeOfDiffusion Dq Dp = Dp := by
  unfold KOfDiffusion KPrimeOfDiffusion
  ring

lemma cubic_coefficient_recovered (ν : ℝ) : gammaOfCubic ν / 2 = ν := by
  unfold gammaOfCubic
  ring

lemma kappa_pos_of_reaction_order {a b : ℝ} (h : a < b) :
    0 < kappaOfReaction a b := by
  unfold kappaOfReaction
  linarith

lemma K_pos_of_diffusions {Dq Dp : ℝ} (hq : 0 < Dq) (hp : 0 < Dp) :
    0 < KOfDiffusion Dq Dp := by
  unfold KOfDiffusion
  positivity

/-- Explicit reaction parameters give `ε=2`, `κ=2`. -/
lemma explicit_reaction_map :
    epsilonOfReaction 1 3 = 2 ∧ kappaOfReaction 1 3 = 2 := by
  constructor <;> norm_num [epsilonOfReaction, kappaOfReaction]

/-- Explicit longitudinal diffusion parameters give
`K=2+√3`, `K' = -(1+√3)`. -/
lemma explicit_diffusion_map :
    KOfDiffusion 1 explicitDp = 2 + sqrt3 ∧
      KPrimeOfDiffusion 1 explicitDp = -(1 + sqrt3) := by
  constructor <;> unfold KOfDiffusion KPrimeOfDiffusion explicitDp <;> ring

end

end QuantumTuringPatterns
