import QuantumTuringPatterns.ClassicalBound

set_option autoImplicit false

/-!
# Homogeneous Gaussian sector

This file formalizes the scalar and componentwise algebra in the paper's
homogeneous Gaussian calculation.  It verifies

* the common stability parameter `η = 2|g|/R`;
* the `2×2` drift trace/determinant criterion;
* the stated solutions of the two Lyapunov equations;
* recombination into the traveling-mode covariance blocks;
* the ordinary/PT scalar spectra and the NPT criterion;
* the common-temperature threshold `η > 2 n̄`.

The results are expressed through exact scalar identities and the three
independent component equations of each symmetric `2×2` Lyapunov
problem, avoiding any reliance on floating-point matrix computation.
-/

namespace QuantumTuringPatterns

noncomputable section

/-- `R² = κ²+4Ω²`. -/
def gaussianR2 (κ Ω : ℝ) : ℝ := κ ^ 2 + 4 * Ω ^ 2

/-- Positive square root `R`. -/
def gaussianR (κ Ω : ℝ) : ℝ := Real.sqrt (gaussianR2 κ Ω)

/-- Stability denominator `Δ = R²-4g²`. -/
def gaussianDelta (κ Ω g : ℝ) : ℝ := gaussianR2 κ Ω - 4 * g ^ 2

/-- Dimensionless squeezing/stability ratio. -/
def gaussianEta (κ Ω g : ℝ) : ℝ := 2 * |g| / gaussianR κ Ω

lemma gaussianR2_nonneg (κ Ω : ℝ) : 0 ≤ gaussianR2 κ Ω := by
  unfold gaussianR2
  positivity

lemma gaussianR_nonneg (κ Ω : ℝ) : 0 ≤ gaussianR κ Ω := by
  exact Real.sqrt_nonneg _

lemma gaussianR_sq (κ Ω : ℝ) : gaussianR κ Ω ^ 2 = gaussianR2 κ Ω := by
  simpa [gaussianR] using Real.sq_sqrt (gaussianR2_nonneg κ Ω)

lemma gaussianR_pos {κ Ω : ℝ} (hκ : 0 < κ) : 0 < gaussianR κ Ω := by
  unfold gaussianR gaussianR2
  positivity

lemma gaussianR_ne_zero {κ Ω : ℝ} (hκ : 0 < κ) : gaussianR κ Ω ≠ 0 :=
  ne_of_gt (gaussianR_pos hκ)

lemma gaussianEta_nonneg (κ Ω g : ℝ) : 0 ≤ gaussianEta κ Ω g := by
  unfold gaussianEta
  positivity

lemma gaussianEta_pos_iff {κ Ω g : ℝ} (hκ : 0 < κ) :
    0 < gaussianEta κ Ω g ↔ g ≠ 0 := by
  constructor
  · intro hη hg
    subst g
    simp [gaussianEta] at hη
  · intro hg
    unfold gaussianEta
    exact div_pos (mul_pos (by norm_num) (abs_pos.mpr hg)) (gaussianR_pos hκ)

/-- Factorization of `Δ` into the two stability factors. -/
lemma gaussianDelta_factor (κ Ω g : ℝ) :
    gaussianDelta κ Ω g =
      (gaussianR κ Ω - 2 * |g|) * (gaussianR κ Ω + 2 * |g|) := by
  unfold gaussianDelta
  nlinarith [gaussianR_sq κ Ω, sq_abs g]

/-- Positive `Δ` is equivalent to `R>2|g|` when `κ>0`. -/
theorem gaussianDelta_pos_iff_R_gt {κ Ω g : ℝ} (hκ : 0 < κ) :
    0 < gaussianDelta κ Ω g ↔ 2 * |g| < gaussianR κ Ω := by
  rw [gaussianDelta_factor]
  constructor
  · intro hprod
    have hsum : 0 < gaussianR κ Ω + 2 * |g| := by
      have hR := gaussianR_pos hκ
      positivity
    by_contra hnot
    have hle : gaussianR κ Ω ≤ 2 * |g| := le_of_not_gt hnot
    have hleft : gaussianR κ Ω - 2 * |g| ≤ 0 := sub_nonpos.mpr hle
    have hright : 0 ≤ gaussianR κ Ω + 2 * |g| := le_of_lt hsum
    have hnonpos :
        (gaussianR κ Ω - 2 * |g|) * (gaussianR κ Ω + 2 * |g|) ≤ 0 :=
      mul_nonpos_of_nonpos_of_nonneg hleft hright
    linarith
  · intro hlt
    have hsum : 0 < gaussianR κ Ω + 2 * |g| := by
      exact add_pos_of_pos_of_nonneg (gaussianR_pos hκ) (by positivity)
    exact mul_pos (sub_pos.mpr hlt) hsum

/-- Main stability equivalence from the homogeneous Gaussian theorem. -/
theorem gaussianEta_lt_one_iff_delta_pos {κ Ω g : ℝ} (hκ : 0 < κ) :
    gaussianEta κ Ω g < 1 ↔ 0 < gaussianDelta κ Ω g := by
  rw [gaussianDelta_pos_iff_R_gt hκ]
  have hR : 0 < gaussianR κ Ω := gaussianR_pos hκ
  unfold gaussianEta
  constructor
  · intro h
    have hmul := (div_lt_iff₀ hR).mp h
    nlinarith
  · intro h
    apply (div_lt_iff₀ hR).2
    nlinarith

/-- Trace of either standing-wave drift block. -/
def standingDriftTrace (κ : ℝ) : ℝ :=
  (-κ / 2) + (-κ / 2)

/-- Determinant of the `+g` standing-wave drift block. -/
def standingDriftDet (κ Ω g : ℝ) : ℝ :=
  (-κ / 2 + g) * (-κ / 2 - g) + Ω ^ 2

lemma standingDriftTrace_eq (κ : ℝ) : standingDriftTrace κ = -κ := by
  unfold standingDriftTrace
  ring

lemma standingDriftDet_eq_delta (κ Ω g : ℝ) :
    standingDriftDet κ Ω g = gaussianDelta κ Ω g / 4 := by
  unfold standingDriftDet gaussianDelta gaussianR2
  ring

/-- The two scalar Hurwitz inequalities for the standing-wave block are
exactly `κ>0` and `η<1`. -/
theorem standingDrift_hurwitz_inequalities_iff
    {κ Ω g : ℝ} (hκ : 0 < κ) :
    (standingDriftTrace κ < 0 ∧ 0 < standingDriftDet κ Ω g) ↔
      gaussianEta κ Ω g < 1 := by
  constructor
  · rintro ⟨_, hdet⟩
    apply (gaussianEta_lt_one_iff_delta_pos hκ).2
    rw [standingDriftDet_eq_delta] at hdet
    nlinarith
  · intro hη
    have hΔ : 0 < gaussianDelta κ Ω g :=
      (gaussianEta_lt_one_iff_delta_pos hκ).1 hη
    constructor
    · rw [standingDriftTrace_eq]
      linarith
    · rw [standingDriftDet_eq_delta]
      linarith

/-! ## Componentwise Lyapunov equations -/

/-- `(1,1)` entry of the `+g` standing-wave covariance. -/
def vPlus11 (κ Ω g : ℝ) : ℝ :=
  (gaussianR2 κ Ω / 2 + κ * g) / gaussianDelta κ Ω g

/-- Symmetric off-diagonal entry of the `+g` covariance. -/
def vPlus12 (κ Ω g : ℝ) : ℝ :=
  (-2 * Ω * g) / gaussianDelta κ Ω g

/-- `(2,2)` entry of the `+g` covariance. -/
def vPlus22 (κ Ω g : ℝ) : ℝ :=
  (gaussianR2 κ Ω / 2 - κ * g) / gaussianDelta κ Ω g

/-- The `-g` block is obtained by reversing the sign of the correlations. -/
def vMinus11 (κ Ω g : ℝ) : ℝ :=
  (gaussianR2 κ Ω / 2 - κ * g) / gaussianDelta κ Ω g

def vMinus12 (κ Ω g : ℝ) : ℝ :=
  (2 * Ω * g) / gaussianDelta κ Ω g

def vMinus22 (κ Ω g : ℝ) : ℝ :=
  (gaussianR2 κ Ω / 2 + κ * g) / gaussianDelta κ Ω g

/-- `(1,1)` component of `A₊V₊+V₊A₊ᵀ+(κ/2)I=0`. -/
theorem lyapunov_plus_11
    {κ Ω g : ℝ} (hΔ : gaussianDelta κ Ω g ≠ 0) :
    2 * (-κ / 2 + g) * vPlus11 κ Ω g +
        2 * Ω * vPlus12 κ Ω g + κ / 2 = 0 := by
  unfold vPlus11 vPlus12
  field_simp [hΔ]
  simp only [gaussianR2, gaussianDelta]
  ring

/-- Off-diagonal component of the `+g` Lyapunov equation. -/
theorem lyapunov_plus_12
    {κ Ω g : ℝ} (hΔ : gaussianDelta κ Ω g ≠ 0) :
    -κ * vPlus12 κ Ω g + Ω * vPlus22 κ Ω g -
        Ω * vPlus11 κ Ω g = 0 := by
  unfold vPlus11 vPlus12 vPlus22
  field_simp [hΔ]
  simp only [gaussianR2, gaussianDelta]
  ring

/-- `(2,2)` component of the `+g` Lyapunov equation. -/
theorem lyapunov_plus_22
    {κ Ω g : ℝ} (hΔ : gaussianDelta κ Ω g ≠ 0) :
    -2 * Ω * vPlus12 κ Ω g +
        2 * (-κ / 2 - g) * vPlus22 κ Ω g + κ / 2 = 0 := by
  unfold vPlus12 vPlus22
  field_simp [hΔ]
  simp only [gaussianR2, gaussianDelta]
  ring

/-- `(1,1)` component of the `-g` Lyapunov equation. -/
theorem lyapunov_minus_11
    {κ Ω g : ℝ} (hΔ : gaussianDelta κ Ω g ≠ 0) :
    2 * (-κ / 2 - g) * vMinus11 κ Ω g +
        2 * Ω * vMinus12 κ Ω g + κ / 2 = 0 := by
  unfold vMinus11 vMinus12
  field_simp [hΔ]
  simp only [gaussianR2, gaussianDelta]
  ring

/-- Off-diagonal component of the `-g` Lyapunov equation. -/
theorem lyapunov_minus_12
    {κ Ω g : ℝ} (hΔ : gaussianDelta κ Ω g ≠ 0) :
    -κ * vMinus12 κ Ω g + Ω * vMinus22 κ Ω g -
        Ω * vMinus11 κ Ω g = 0 := by
  unfold vMinus11 vMinus12 vMinus22
  field_simp [hΔ]
  simp only [gaussianR2, gaussianDelta]
  ring

/-- `(2,2)` component of the `-g` Lyapunov equation. -/
theorem lyapunov_minus_22
    {κ Ω g : ℝ} (hΔ : gaussianDelta κ Ω g ≠ 0) :
    -2 * Ω * vMinus12 κ Ω g +
        2 * (-κ / 2 + g) * vMinus22 κ Ω g + κ / 2 = 0 := by
  unfold vMinus12 vMinus22
  field_simp [hΔ]
  simp only [gaussianR2, gaussianDelta]
  ring

/-! ## Traveling-mode recombination -/

/-- Diagonal traveling-mode covariance entry. -/
def travelingS (κ Ω g : ℝ) : ℝ :=
  gaussianR2 κ Ω / (2 * gaussianDelta κ Ω g)

/-- Entries of the traveling-mode correlation block. -/
def travelingC11 (κ Ω g : ℝ) : ℝ := κ * g / gaussianDelta κ Ω g
def travelingC12 (κ Ω g : ℝ) : ℝ := -2 * Ω * g / gaussianDelta κ Ω g
def travelingC22 (κ Ω g : ℝ) : ℝ := -κ * g / gaussianDelta κ Ω g

lemma standing_average_11
    {κ Ω g : ℝ} (hΔ : gaussianDelta κ Ω g ≠ 0) :
    (vPlus11 κ Ω g + vMinus11 κ Ω g) / 2 = travelingS κ Ω g := by
  unfold vPlus11 vMinus11 travelingS
  field_simp [hΔ]
  ring

lemma standing_average_22
    {κ Ω g : ℝ} (hΔ : gaussianDelta κ Ω g ≠ 0) :
    (vPlus22 κ Ω g + vMinus22 κ Ω g) / 2 = travelingS κ Ω g := by
  unfold vPlus22 vMinus22 travelingS
  field_simp [hΔ]
  ring

lemma standing_average_12
    {κ Ω g : ℝ} (hΔ : gaussianDelta κ Ω g ≠ 0) :
    (vPlus12 κ Ω g + vMinus12 κ Ω g) / 2 = 0 := by
  unfold vPlus12 vMinus12
  field_simp [hΔ]
  ring

lemma standing_difference_11
    {κ Ω g : ℝ} (hΔ : gaussianDelta κ Ω g ≠ 0) :
    (vPlus11 κ Ω g - vMinus11 κ Ω g) / 2 = travelingC11 κ Ω g := by
  unfold vPlus11 vMinus11 travelingC11
  field_simp [hΔ]
  ring

lemma standing_difference_12
    {κ Ω g : ℝ} (hΔ : gaussianDelta κ Ω g ≠ 0) :
    (vPlus12 κ Ω g - vMinus12 κ Ω g) / 2 = travelingC12 κ Ω g := by
  unfold vPlus12 vMinus12 travelingC12
  field_simp [hΔ]
  ring

lemma standing_difference_22
    {κ Ω g : ℝ} (hΔ : gaussianDelta κ Ω g ≠ 0) :
    (vPlus22 κ Ω g - vMinus22 κ Ω g) / 2 = travelingC22 κ Ω g := by
  unfold vPlus22 vMinus22 travelingC22
  field_simp [hΔ]
  ring

/-! ## Scalar spectrum and partial transpose -/

/-- Euclidean radius of the `2×2` correlation block. -/
def correlationRadius (κ Ω g : ℝ) : ℝ :=
  |g| * gaussianR κ Ω / gaussianDelta κ Ω g

/-- Algebraic invariant under the square root in the ordinary symplectic spectrum. -/
theorem traveling_invariant
    {κ Ω g : ℝ} (hΔ : gaussianDelta κ Ω g ≠ 0) :
    travelingS κ Ω g ^ 2 - correlationRadius κ Ω g ^ 2 =
      gaussianR κ Ω ^ 2 / (4 * gaussianDelta κ Ω g) := by
  unfold travelingS correlationRadius
  field_simp [hΔ]
  rw [gaussianR_sq, sq_abs]
  unfold gaussianDelta
  ring

/-- Ordinary physical symplectic eigenvalue from the paper. -/
def physicalNu (κ Ω g : ℝ) : ℝ :=
  gaussianR κ Ω / (2 * Real.sqrt (gaussianDelta κ Ω g))

lemma physicalNu_nonneg (κ Ω g : ℝ) : 0 ≤ physicalNu κ Ω g := by
  unfold physicalNu
  positivity

lemma physicalNu_sq
    {κ Ω g : ℝ} (hΔ : 0 < gaussianDelta κ Ω g) :
    physicalNu κ Ω g ^ 2 =
      gaussianR κ Ω ^ 2 / (4 * gaussianDelta κ Ω g) := by
  unfold physicalNu
  rw [div_pow, mul_pow, Real.sq_sqrt (le_of_lt hΔ)]
  norm_num

/-- Identity `1-η² = Δ/R²`. -/
theorem one_sub_eta_sq
    {κ Ω g : ℝ} (hκ : 0 < κ) :
    1 - gaussianEta κ Ω g ^ 2 =
      gaussianDelta κ Ω g / gaussianR κ Ω ^ 2 := by
  have hR : gaussianR κ Ω ≠ 0 := gaussianR_ne_zero hκ
  unfold gaussianEta
  field_simp [hR]
  rw [gaussianR_sq, sq_abs]
  unfold gaussianDelta
  ring

/-- Smaller partially-transposed symplectic eigenvalue. -/
def ptNuMinus (κ Ω g : ℝ) : ℝ :=
  gaussianR κ Ω / (2 * (gaussianR κ Ω + 2 * |g|))

/-- Larger partially-transposed symplectic eigenvalue. -/
def ptNuPlus (κ Ω g : ℝ) : ℝ :=
  gaussianR κ Ω / (2 * (gaussianR κ Ω - 2 * |g|))

/-- Exact reduction of the smaller PT eigenvalue to `1/[2(1+η)]`. -/
theorem ptNuMinus_eq_eta
    {κ Ω g : ℝ} (hκ : 0 < κ) :
    ptNuMinus κ Ω g = 1 / (2 * (1 + gaussianEta κ Ω g)) := by
  have hRpos : 0 < gaussianR κ Ω := gaussianR_pos hκ
  have hR : gaussianR κ Ω ≠ 0 := ne_of_gt hRpos
  have hsum : gaussianR κ Ω + 2 * |g| ≠ 0 := by positivity
  have hη : 0 ≤ gaussianEta κ Ω g := gaussianEta_nonneg κ Ω g
  have hηden : 1 + gaussianEta κ Ω g ≠ 0 := by positivity
  unfold ptNuMinus gaussianEta
  field_simp [hR, hsum, hηden]
  ring

/-- Exact reduction of the larger PT eigenvalue in the stable regime. -/
theorem ptNuPlus_eq_eta
    {κ Ω g : ℝ} (hκ : 0 < κ) (hΔ : 0 < gaussianDelta κ Ω g) :
    ptNuPlus κ Ω g = 1 / (2 * (1 - gaussianEta κ Ω g)) := by
  have hRpos : 0 < gaussianR κ Ω := gaussianR_pos hκ
  have hR : gaussianR κ Ω ≠ 0 := ne_of_gt hRpos
  have hgap : 0 < gaussianR κ Ω - 2 * |g| := by
    have hlt := (gaussianDelta_pos_iff_R_gt hκ).1 hΔ
    linarith
  have hgapne : gaussianR κ Ω - 2 * |g| ≠ 0 := ne_of_gt hgap
  have hηlt : gaussianEta κ Ω g < 1 :=
    (gaussianEta_lt_one_iff_delta_pos hκ).2 hΔ
  have hηden : 1 - gaussianEta κ Ω g ≠ 0 := by linarith
  unfold ptNuPlus gaussianEta
  field_simp [hR, hgapne, hηden]
  ring

/-- Every stable pair with nonzero squeezing is NPT; at `g=0` the value
is exactly the PPT threshold `1/2`.  Stability is not needed for the
smaller scalar formula, but is included in the paper's physical regime. -/
theorem ptNuMinus_lt_half_iff
    {κ Ω g : ℝ} (hκ : 0 < κ) :
    ptNuMinus κ Ω g < 1 / 2 ↔ g ≠ 0 := by
  rw [ptNuMinus_eq_eta hκ]
  constructor
  · intro h hg
    subst g
    norm_num [gaussianEta] at h
  · intro hg
    have hη : 0 < gaussianEta κ Ω g :=
      (gaussianEta_pos_iff hκ).2 hg
    have hden : 0 < 2 * (1 + gaussianEta κ Ω g) := by positivity
    apply (div_lt_iff₀ hden).2
    nlinarith

/-- The determinant identity linking classical stability and entanglement. -/
theorem standingDriftDet_eq_eta
    {κ Ω g : ℝ} (hκ : 0 < κ) :
    standingDriftDet κ Ω g =
      gaussianR κ Ω ^ 2 / 4 * (1 - gaussianEta κ Ω g ^ 2) := by
  rw [standingDriftDet_eq_delta, one_sub_eta_sq hκ]
  have hR : gaussianR κ Ω ≠ 0 := gaussianR_ne_zero hκ
  field_simp [hR]
  ring

/-! ## Common thermal occupation -/

/-- Thermal scaling of the smaller PT eigenvalue. -/
def thermalPtNuMinus (nbar κ Ω g : ℝ) : ℝ :=
  (2 * nbar + 1) * ptNuMinus κ Ω g

lemma thermalPtNuMinus_eq_eta
    {nbar κ Ω g : ℝ} (hκ : 0 < κ) :
    thermalPtNuMinus nbar κ Ω g =
      (2 * nbar + 1) / (2 * (1 + gaussianEta κ Ω g)) := by
  unfold thermalPtNuMinus
  rw [ptNuMinus_eq_eta hκ]
  ring

/-- Exact common-temperature NPT threshold. -/
theorem thermal_npt_iff
    {nbar κ Ω g : ℝ} (hκ : 0 < κ) :
    thermalPtNuMinus nbar κ Ω g < 1 / 2 ↔
      2 * nbar < gaussianEta κ Ω g := by
  rw [thermalPtNuMinus_eq_eta hκ]
  have hη : 0 ≤ gaussianEta κ Ω g := gaussianEta_nonneg κ Ω g
  have hden : 0 < 2 * (1 + gaussianEta κ Ω g) := by positivity
  constructor
  · intro h
    have hmul := (div_lt_iff₀ hden).mp h
    nlinarith
  · intro h
    apply (div_lt_iff₀ hden).2
    nlinarith

end

end QuantumTuringPatterns
