# D12 Pre-Registration: Public Diff v1.1 → v2.0

**Document type**: OSF supplementary attachment for v2.0 registration, providing public diff against v1.1
**Created**: 2026-04-30
**Branch**: `claude/hexaco-harassment-simulation-69jZp`
**Author**: Eisuke Tokiwa (sole-authored; ORCID: 0009-0009-7124-6669; SUNBLAZE Co., Ltd.)
**v1.1 OSF DOI** (LOCKED, historical): 10.17605/OSF.IO/45QP9 — https://osf.io/45qp9
**v2.0 OSF DOI** (pending): TBD

---

## 0. Purpose of this document

This document is the **public diff** required by Section 6.5 of the preregistration (Level 3 deviation policy) when transitioning from v1.1 (locked on OSF) to v2.0. It is uploaded as supplementary to the v2.0 OSF project alongside the v2.0 master document.

Per Section 6.5 Level 3:

> Level 3 (analysis-plan revision): e.g., introducing 14-cell pairwise inference or other substantive change → register a v2 of this preregistration on OSF; publish a diff against v1.

This document fulfills the "publish a diff against v1" requirement.

---

## 1. Why v2.0 was needed

Following OSF registration of v1.1 (DOI 10.17605/OSF.IO/45QP9, registered 2026-04-30), the author commissioned an independent methodologist review per Section 8.1 (mode B: anonymous external methodologist with mathematical biology background).

The reviewer raised **two substantive pre-review concerns**:

### Concern 1: MoM stability decision rule was ad-hoc

Section 5.2 (v1.1) noted that "MoM may diverge → switch to Marginal MLE / Stan as triangulation," but the **decision rule** for when to switch was not pre-specified. This created a Section 6.5 Level 2 (data-driven adjustment) opening that the reviewer judged inelegant for a method choice known in advance to be at risk.

### Concern 2 (★ critical): Bootstrap MAPE CI propagation not addressed

The **most important concern**: D13 power analysis showed cell-level binary-rate 95% CI half-widths up to ±30 percentage points. When propagated through population aggregation, the resulting bootstrap CI on aggregate MAPE could be wide enough that:

- Point MAPE could fall in SUCCESS region (≤30%)
- 95% CI could simultaneously overlap PARTIAL (30-60%) and FAILURE (>60%) regions

A SUCCESS classification under v1.1's single-tier point-estimate criterion would be technically met but inferentially indefensible. The reviewer noted that resolving this in the Discussion (rather than the decision rule) would weaken Nosek 2018 PNAS Challenge 9 (narrative inference selectivity) protection — the core function of preregistration.

### Reviewer recommendation

Path C (Level 3 deviation = v2 OSF registration) with refinements: 4-tier hierarchy + bootstrap MAPE CI procedure pre-specification + MoM/BCa Level 1 clarifications bundled into v2.

Author accepted the recommendation. v2.0 was prepared on 2026-04-30 (same-day turnaround) and registered as OSF v2 with this diff document attached.

---

## 2. Summary of changes

| Change # | Section | Type | Affects SUCCESS / FAILURE judgment? |
|---|---|---|---|
| 1 | 5.1 (Stage 0 cell-level) | Added | No (numerical robustness only) |
| 2 | 5.2 (28-cell EB) | Added | No (estimator selection rule for sensitivity tier) |
| 3 | 5.4 (Validation triangulation) | **Substantive replacement** | **YES** — replaces single-tier point-MAPE with 4-tier hierarchy + bootstrap MAPE CI |
| 4 | 6.1 (Inference criteria summary) | Modified to match Section 5.4 | YES (mirrors Section 5.4) |
| 5 | 14.1 (Version log) | New v2.0 entry added | No (administrative) |

**Unchanged in v2.0** (preserved verbatim from v1.1):

- All hypotheses (H1, H2, H2.industry, H3, H4–H7) — wording unchanged; only H1 decision rule expanded
- Sample size, power analysis (D13), all sensitivity sweep ranges
- Phase 2 counterfactual specifications (δ_A, δ_B, effect_C, transportability_factor) — values and ranges identical
- Pasona 2022 / MHLW R5 / MHLW R4 citations and roles (introduced in v1.1)
- Sections 7 (negative-result publication commitment), 8 (reproducibility), 9 (ethics), 10 (limitations), 11 (reflexivity), 12 (other), 13 (citations) — entirely unchanged
- Random seed (20260429) — unchanged
- Section 6.5 deviation policy — unchanged

---

## 3. Detailed section-by-section diff

### 3.1 Section 5.1 — Stage 0 cell-level propensity estimation

**v1.1**: BCa correction described, but no fallback specified for numerical instability.

**v2.0 ADDED**:

> **BCa numerical stability fallback chain** (★ added v2):
> - BCa is the primary CI method.
> - If acceleration parameter |a| > 10 (jackknife instability) or BCa computation raises numerical error (e.g., division by zero at edge rates 0/N or N/N), automatically fallback to **Bias-Corrected (BC) bootstrap**.
> - If BC bootstrap also fails (rare; sample distribution too degenerate), fallback to **percentile bootstrap**.
> - All cells record which CI method was actually used; the supplementary table reports per-cell method (BCa / BC / percentile).
> - Aggregate-level (Stage 1, Stage 2) MAPE bootstrap CI inherits the cell-level method choice automatically.

**Justification**: At cell N=10 with binary outcomes, edge rates (0/10 or 10/10) cause jackknife acceleration parameter degeneracy. BCa as theoretically 2nd-order accurate (DiCiccio & Efron 1996) cannot maintain that property under numerical breakdown; explicit fallback chain restores robustness without ad-hoc analyst judgment.

### 3.2 Section 5.2 — 28-cell EB shrinkage

**v1.1**: Method-of-moments hyperprior described. Triangulation note ("Marginal MLE and Stan / brms hierarchical Bayes posteriors are run as auxiliary checks") existed but no formal switch criterion.

**v2.0 ADDED**:

> **MoM rejection decision rule** (★ added v2, pre-specified):
> - IF σ̂² / [μ̂(1−μ̂)] < 0.05 (variance too small relative to maximum-possible binomial variance, indicating MoM will produce extreme α̂, β̂) → **REJECT MoM, switch to Stan hierarchical Bayes (HMC, brms) as primary 28-cell estimator**.
> - IF max(α̂, β̂) > 100 (overshrink risk: pseudo-count exceeds 7× the median cell N=14) → **REJECT MoM, switch to Stan**.
> - IF both conditions clear → **ACCEPT MoM as primary 28-cell estimator**.
> - IN ALL CASES, both MoM and Stan posteriors (and Marginal MLE for triangulation) are computed and reported in supplementary; the "primary" choice only determines which is used in the main text.
> - This rule is fixed by this v2 preregistration; no post-hoc threshold adjustment.
>
> **Note on threshold sensitivity** (Matsuda methodologist consultation, anonymous, math-bio): the reviewer noted that σ̂² is itself a sample estimate from N=14 cells with sampling variability. To address this, the supplementary additionally reports the MoM result alongside Stan even when MoM is "accepted", enabling readers to verify that the threshold-based switch does not produce qualitatively different conclusions. An alternative diagnostic (α̂ + β̂ as a fraction of median cell N) is also reported as cross-check.

**Justification**: With only 14 cells driving the hyperprior, σ̂² has nontrivial sampling variability, and MoM α̂, β̂ can become extreme when σ̂² is small relative to μ̂(1−μ̂). Pre-specifying numerical thresholds removes analyst discretion while reporting both MoM and Stan estimates ensures the reader can assess sensitivity to the switch threshold itself.

### 3.3 Section 5.4 — Validation triangulation (★ central change)

**v1.1**: single-tier point-estimate criterion:

> Primary metric (★ pre-registered):
> - MAPE(P̂_FY2016, MHLW H28 FY2016 past-3-year power harassment 32.5%) ≤ 30% → SUCCESS
> - 30% < MAPE ≤ 60% → PARTIAL SUCCESS
> - MAPE > 60% → FAILURE (publish anyway, see Section 7.3)

**v2.0 REPLACEMENT** (4-tier judgment hierarchy + bootstrap MAPE CI procedure):

#### 3.3a Bootstrap MAPE CI procedure

> For each bootstrap iteration b ∈ {1..2,000}:
>
> 1. Resample N=354 with replacement, **stratified by 14 cell** (preserving cell membership marginals).
> 2. Re-classify each resampled individual to the nearest of 7 centroids (centroids fixed from N=13,668; not bootstrapped).
> 3. Re-compute each cell's binary harassment outcome propensity p̂_c^(b) on the resampled data, applying the BCa numerical stability fallback chain (Section 5.1) at the cell level.
> 4. Apply population weights W_c (from MHLW Labor Force Survey, FIXED across iterations — not bootstrapped) to compute the resampled national latent prevalence:
>    P̂_t^(b) = Σ_c (p̂_c^(b) × W_c) / Σ_c W_c
> 5. Compute the resampled MAPE:
>    MAPE^(b) = |P̂_t^(b) − MHLW_observed_t| / MHLW_observed_t × 100
>    where MHLW_observed_t is the FIXED point estimate from MHLW survey for period t (no bootstrap of MHLW data).
> 6. Record MAPE^(b).
>
> After all B = 2,000 iterations:
> - Point estimate MAPE is computed on the original (non-bootstrapped) data, NOT the mean of MAPE^(b).
> - 95% BCa CI for MAPE is constructed from the empirical distribution of {MAPE^(b)}.

**Why the procedure is pre-specified explicitly**: without this specification, the analyst could choose between (i) cell-level resample only, (ii) cell-level + population-weight resample, (iii) cell-level + MHLW-observed resample, or other combinations. Each choice produces different CIs. Pre-specifying the procedure removes Section 6.5 Level 2 (data-driven) opening identified by the reviewer.

#### 3.3b 4-tier judgment hierarchy

| Tier | Condition | Interpretation |
|---|---|---|
| **Strict SUCCESS** | Point MAPE ≤ 30% **AND** 95% BCa CI upper bound ≤ 30% | Strong confirmatory; CI rules out PARTIAL or FAILURE regions |
| **Standard SUCCESS** | Point MAPE ≤ 30% (CI permitted to overlap) | Weak confirmatory; "Pre-registered ambiguity Tier" qualifier required in Discussion |
| **PARTIAL SUCCESS** | 30% < point MAPE ≤ 60% | Mixed evidence |
| **FAILURE** | Point MAPE > 60% | Publish per Section 7.3 |

**Tier-specific reporting requirements**:
- Strict SUCCESS: "Strict SUCCESS achieved" allowed in headline / abstract without qualification.
- Standard SUCCESS: must be qualified with explicit CI ambiguity statement (e.g., "Standard SUCCESS achieved (point MAPE = X%, 95% CI = [Y%, Z%]). The CI overlapped the PARTIAL region; the result falls in the Pre-registered ambiguity Tier, and confirmatory claims are correspondingly weakened.").
- PARTIAL / FAILURE: standard framing per Section 7.3.

**Justification**: The 4-tier hierarchy resolves the "point estimate SUCCESS but CI ambiguous" scenario by **encoding the ambiguity in the decision rule itself**, rather than relegating it to post-hoc Discussion treatment. This satisfies Nosek 2018 Challenge 9 (preventing narrative inference selectivity) at the inferential layer, not just the reporting layer. All tiers continue to satisfy Section 7 negative-result publication commitment.

### 3.4 Section 6.1 — Inference criteria summary table

**v1.1** (relevant rows):

| H# | Criterion | Threshold | Decision |
|---|---|---|---|
| **H1** | MAPE | ≤ 30% | SUCCESS |
| H1 | MAPE | 30 < x ≤ 60% | PARTIAL SUCCESS |
| H1 | MAPE | > 60% | FAILURE |

**v2.0** (replaces 3 rows with 4 rows):

| H# | Criterion | Threshold | Decision |
|---|---|---|---|
| **H1 (Strict SUCCESS)** ★ v2 | Point MAPE AND 95% BCa CI upper bound | both ≤ 30% | Strong confirmatory |
| **H1 (Standard SUCCESS)** ★ v2 | Point MAPE | ≤ 30% (CI may overlap) | Weak confirmatory; CI ambiguity Tier note required |
| H1 (Partial) | Point MAPE | 30 < x ≤ 60% | Mixed evidence |
| H1 (Failure) | Point MAPE | > 60% | Publish per Section 7.3 |

This is purely a reflection of the substantive change in Section 5.4.

### 3.5 Section 14.1 — Version log

**v2.0 ADDS**:

| Version | Date | Changes |
|---|---|---|
| v2.0 draft | 2026-04-30 | Path C upgrade per anonymous methodologist review (Matsuda mode B, mathematical biology). Four substantive changes (1-4 above). |
| v2.0 LOCKED | TBD | 🔒 To be registered on OSF after author submits v2 with public diff against v1.1. v2 OSF DOI: TBD. v1.1 (DOI 10.17605/OSF.IO/45QP9) remains valid as historical record but is superseded by v2 for primary inference. |

---

## 4. Operational impact on Stage 0 implementation

The four v2.0 changes affect Stage 0 code implementation as follows:

| Change | Implementation file (Section 8.2 repo structure) | Implementation cost |
|---|---|---|
| 1. BCa fallback chain | `code/stage0_cell_propensity.py` + `code/utils_bootstrap.py` | ~2 hours (try/except + diagnostic logging) |
| 2. MoM rejection rule | `code/stage0_eb_shrinkage.py` + `code/utils_diagnostics.py` | ~3 hours (threshold check + Stan brms triangulation) |
| 3. Bootstrap MAPE CI procedure | `code/stage1_population_aggregation.py` + `code/stage2_validation.py` | ~4 hours (procedure encoded; CI computation + tier classification) |
| 4. 4-tier hierarchy reporting | `code/stage2_validation.py` + reporting templates | ~2 hours (tier classification logic + reporting Markdown templates) |

**Total additional implementation cost vs v1.1**: ~11 hours; offset by reduced post-hoc justification effort.

Random seed (20260429), all hypotheses, sample sizes, and Phase 2 specifications are unchanged → no change to other Stage 0 / Stage 1 / Stage 2 / Stage 6-8 code modules.

---

## 5. Backwards compatibility with v1.1

- v1.1 OSF registration (DOI 10.17605/OSF.IO/45QP9) **remains valid** as historical record. It is not withdrawn.
- v2.0 supersedes v1.1 for **primary inference** only. All sections of v1.1 outside the four changed sections are inherited verbatim by v2.0.
- Should v2.0 require subsequent revision (Level 3 deviation), a v3 OSF registration with a public diff against v2.0 will be created. v1.1 → v3 cumulative diff will not be necessary because each v_i diff is referenced from v_(i+1).
- The Tokiwa harassment preprint and clustering paper (IEEE-published) remain the data sources of record; no change to data collection, IRB, or licensing.

---

## 6. Acknowledgments

The author thanks an anonymous external methodologist with a mathematical biology background for the substantive pre-review concerns that motivated this v2.0 upgrade. The methodologist's identity is held privately by the author; their disciplinary background is disclosed as required by Munafò et al. 2017 Box 1 lightweight variant of independent statistical oversight.

The reviewer's specific contributions to v2.0:
- Identification of MoM stability decision rule absence (Concern 1)
- Identification of bootstrap MAPE CI propagation issue (Concern 2)
- Recommendation of Path C (Level 3 deviation = v2 OSF registration)
- Refinement: 4-tier judgment hierarchy in lieu of "point estimate + secondary CI criterion" parallelism
- Refinement: bootstrap MAPE CI procedure pre-specification (cell-stratified resample, fixed weights, fixed MHLW values)
- Refinement: MoM threshold rule with sampling-variability cross-check (α̂ + β̂ / median cell N)

The full review memo (forthcoming, anticipated 2026-05-21) will be cited in the eventual paper Acknowledgments.

---

## 7. References

- v1.1 OSF DOI: 10.17605/OSF.IO/45QP9 — https://osf.io/45qp9
- v2.0 master document: `simulation/docs/pre_registration/D12_pre_registration_OSF.md` (JP, 1,108 lines, 25 pages PDF)
- v2.0 English version: `simulation/docs/pre_registration/D12_pre_registration_OSF.en.md` (1,138 lines, 25 pages PDF)
- D13 power analysis report: `simulation/docs/power_analysis/D13_power_analysis.md`
- Methodologist review memo (anticipated 2026-05-21): not yet available

Section anchors mentioned above all refer to v2.0 master document (`D12_pre_registration_OSF.md`), unless otherwise stated.

---

**End of v1.1 → v2.0 Public Diff document.**
