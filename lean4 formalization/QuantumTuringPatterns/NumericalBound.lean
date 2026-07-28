import QuantumTuringPatterns.EntanglementSelection

set_option autoImplicit false

/-!
# Exact rational inequality for the NPT sign

The numerical analysis reports

`λ_min(Ĥ_PT) = -0.2462`, `‖H_exact-Ĥ‖ ≤ 3.66×10⁻⁴`,

and uses Weyl's inequality to conclude strict negativity.  The matrix
norm/Weyl step is represented by an explicit upper-bound hypothesis;
all decimal arithmetic and the strict sign margin are checked exactly
as rational arithmetic in Lean.
-/

namespace QuantumTuringPatterns

noncomputable section

/-- Rounded numerical minimum eigenvalue. -/
def numericalPtMinimum : ℝ := -(2462 : ℝ) / 10000

/-- A-posteriori covariance/eigenvalue perturbation bound. -/
def numericalErrorBound : ℝ := (366 : ℝ) / 1000000

/-- Convenient displayed strict margin from the numerical analysis. -/
def reportedNegativeMargin : ℝ := -(2458 : ℝ) / 10000

lemma reported_sum_exact :
    numericalPtMinimum + numericalErrorBound =
      -(245834 : ℝ) / 1000000 := by
  norm_num [numericalPtMinimum, numericalErrorBound]

lemma reported_sum_lt_margin :
    numericalPtMinimum + numericalErrorBound < reportedNegativeMargin := by
  norm_num [numericalPtMinimum, numericalErrorBound, reportedNegativeMargin]

lemma reportedNegativeMargin_neg : reportedNegativeMargin < 0 := by
  norm_num [reportedNegativeMargin]

lemma reported_sum_neg : numericalPtMinimum + numericalErrorBound < 0 := by
  exact lt_trans reported_sum_lt_margin reportedNegativeMargin_neg

/-- Generic one-sided Weyl-error transfer. -/
theorem strict_negative_from_upper_error
    {exactValue approximateValue error : ℝ}
    (hupper : exactValue ≤ approximateValue + error)
    (hnegative : approximateValue + error < 0) :
    exactValue < 0 :=
  lt_of_le_of_lt hupper hnegative

/-- Instantiation of the strict-sign bound.  The hypothesis
`hWeyl` is precisely the matrix-analytic conclusion of Weyl's inequality
and the contractive Fourier/PT selection. -/
theorem npt_sign_bound
    {exactPtMinimum : ℝ}
    (hWeyl : exactPtMinimum ≤ numericalPtMinimum + numericalErrorBound) :
    exactPtMinimum < 0 := by
  exact strict_negative_from_upper_error hWeyl reported_sum_neg

/-- The displayed physicality margin `0.168` also survives the same
`3.66×10⁻⁴` perturbation. -/
theorem reported_physicality_margin :
    (168 : ℝ) / 1000 - numericalErrorBound > 0 := by
  norm_num [numericalErrorBound]

end

end QuantumTuringPatterns
