# Author–Investigator Q&A Log

**Project**: HEXACO 7-Typology Workplace Harassment Microsimulation (v2.0)
**OSF DOI**: [10.17605/OSF.IO/3Y54U](https://osf.io/3y54u)
**Date**: 2026-04-30
**Author**: Eisuke Tokiwa (SUNBLAZE Co., Ltd., ORCID 0009-0009-7124-6669)
**Format**: Investigator's questions translated through dialogue with AI methodological assistant. Captured here for the manuscript Discussion / Limitations section.

---

## Q1. e-Stat MHLW data acquisition path

> **Q (investigator):** What is the best granularity of MHLW data we should acquire for Stage 1 reweighting?

**A (summary):** The ideal would be a 5-year age × gender × employment status crosstab from the Labor Force Survey (基本集計). Acquisition path: e-Stat → 労働力調査 → 基本集計 → 結果原表 → 年次. The investigator subsequently retrieved the official 総務省統計局 PDF (2023-01-31 publication) of "労働力調査（基本集計）2022年（令和４年）平均結果の要約" and extracted Table 3 (年齢階級別就業者数). Conversion delivered a long-form CSV with rows {15-64, 65+} × {男, 女} × employed = 4 records, totaling 6,723 万人, with marginals F = 0.4498 / M = 0.5502.

**Decision**: Use 就業者 (employed persons) as the primary working population (broader than 役員除く雇用者; includes self-employed; better matches MHLW survey "労働者" denominator). 役員除く雇用者 retained as alternative for sensitivity reporting.

**Implication for paper**: Section 5.3 (population aggregation) cites the MHLW publication directly. Acquisition provenance preserved in `simulation/data/README_MHLW_acquisition.md` and the source PDF retained in repository for reproducibility.

---

## Q2. Why does Counterfactual A (personality intervention) fail?

> **Q (investigator):** I don't understand why "improving personality" doesn't reduce harassment. Surely if everyone becomes nicer, harassment should drop?

**A (summary):** Three layered reasons:

1. **Realistic effect-size calibration**: The +0.3 SD shift on H/A/E represents the *empirical maximum* effect of personality training programs (Sleep & Watts 2024; Hudson & Fraley 2015). It is not "everyone becomes a saint" — it is "training improves a person from the 30th percentile to the 50th percentile."
2. **Person × environment interaction**: In observational research, individual personality–harassment correlations are weak (r < 0.3); environmental power-distance/psychological-safety variables predict harassment far more strongly (r > 0.5; Salin 2003; Hershcovis & Barling 2010). Hence, a 0.3 SD personality shift is leveraged through a low-correlation pathway.
3. **Cluster-level propensity is environmentally determined**: Even after a personality shift moves individuals slightly across cluster boundaries, the cluster-level harassment rate (which reflects the workplace environment, not only person characteristics) does not change.

**Implication for paper**: Discussion §X discusses why Counterfactual A's null result is theoretically expected, not surprising; introduces the "person × environment" framing as the operative interpretation lens.

---

## Q3. Is Counterfactual C (structural -20%) tautological?

> **Q (investigator):** "Reducing propensity by 20%" mechanically lowers prevalence. Isn't this circular reasoning? Real institutions raise psychological barriers; they don't directly lower probabilities.

**A (summary):** **The investigator is methodologically correct.** The simulation is a *reduced-form* model — it does not endogenously model deterrence dynamics. The contribution is therefore narrower than initially stated:

- The −20% is *not* an arbitrary parameter; it is calibrated against three meta-analyses (Salin 2021; Escartín 2016; Hodgins, MacCurtain, & Mannix-McNamara 2014) and the MHLW FY2016–FY2023 natural experiment (32.5% → 19.3%; −13.2pp ≈ −40.6% relative, spanning the 2020 Power Harassment Prevention Law's staged enforcement).
- The +0.3 SD on Counterfactual A is similarly calibrated against personality training meta-analyses.
- The simulation's contribution is to translate both prior meta-analytic effect sizes into a common population-level metric (national prevalence) and compare them apples-to-apples.
- It does **not** prove "structural interventions work"; it shows that, **given prior literature on realistic effect sizes**, structural interventions are roughly 5.8× more effective than personality interventions at the population level.
- Mechanistic agent-based models with explicit deterrence dynamics are deferred to follow-up work.

**Implication for paper**: Limitations section flags the reduced-form framing prominently; positions the simulation's role as "common-metric translation of prior meta-analyses" rather than mechanistic causal demonstration.

---

## Q4. Institutions don't change personalities — does the model handle this?

> **Q (investigator):** Strict institutions don't make people's personalities better. So how can our simulation show structural intervention "works"?

**A (summary):** **The simulation is in fact built on this assumption.** Counterfactual C does not modify any HEXACO score — it modifies cell-level propensity. The cleanest reading is:

- HEXACO scores stay fixed (low-HH individuals remain low-HH)
- Cluster membership stays fixed (no person crosses cluster boundaries)
- *Only the rate at which harassment occurs within each cluster × gender cell changes*

This represents the population-level consequence of psychological deterrence (raised barriers, increased reporting risk, social cost) without claiming personalities are altered. The model is therefore conceptually clean: A intervenes on the personality side, C on the environmental side; both produce population-level prevalence shifts, but through different mechanisms.

**Implication for paper**: The paper repositions HEXACO 7-typology as a *stratification variable*, not an *intervention target*. This shift has policy implications and ethical consequences, addressed in Q6.

---

## Q5. Does "low HH ↔ harassment victimization" mean low-HH people just inhabit harassment-prone environments?

> **Q (investigator):** If low HH and harassment victimization correlate, does that mean low-HH individuals are simply concentrated in toxic environments rather than personality directly causing harassment?

**A (summary):** **Yes — the data are most consistent with this interpretation.** Four causal hypotheses can explain the correlation:

| Hypothesis | Story | Compatibility with our results |
|---|---|---|
| M1: Direct causation | Low HH causes harassment perpetration/victimization | ✗ Inconsistent: ΔP_A ≈ 0 (changing HH does not change rate) |
| M2: Selection | Low HH individuals select into harassment-prone industries/jobs | ✓ Consistent (environment dominates) |
| M3: Confounding | Shared SES/educational/economic factors cause both | ✓ Consistent (environment dominates) |
| M4: Reverse causation | Harassment exposure depresses HEXACO H scores via cynicism | ✓ Consistent (environment dominates) |

We cannot distinguish M2/M3/M4 from observational data alone. However, **all three are jointly inconsistent with M1**, and our Counterfactual A null result is most parsimoniously explained by the latter three.

**Implication for paper**: Discussion explicitly frames this. Important ethical consequence: **rejects victim-blaming.** The conventional reading "low-HH individuals should self-improve to avoid victimization" is unsupported by our results; the structural reading "harassment-prone environments concentrate low-HH individuals" is supported.

---

## Q6. Restated study purpose: HEXACO as stratification, not intervention target

> **Q (investigator, condensed):** If institutions don't change personalities, what is HEXACO actually for in this study?

**A (summary):**

| Use | ❌ Wrong reading | ✅ Correct reading |
|---|---|---|
| HEXACO 7-typology | "Identifies who to retrain via personality intervention" | "Identifies which subgroups are differentially exposed; verifies that structural intervention does not differentially fail any subgroup" |
| Low HH | "Personality flaw to be corrected" | "Population marker for being in a harassment-prone environment (not a defect)" |

The repositioned framing: **HEXACO is a stratification lens, not an intervention target.** This is the manuscript's preferred ethical posture.

**Implication for paper**: Title and abstract emphasize this reframing. Suggested phrasing:
- "HEXACO serves as a stratification variable to identify differentially exposed populations, not as a personality-modification target."
- "Counterfactual C reduces harassment prevalence without modifying any individual's HEXACO score."

---

## Methodological summary for Discussion / Limitations

The above Q&A surfaced four limitations that the manuscript will openly disclose:

1. **Reduced-form framing**: The simulation translates prior meta-analytic effect sizes into population-level prevalence, but does not endogenously model psychological deterrence mechanisms.
2. **Causal under-identification**: Observational data cannot distinguish among selection, confounding, and reverse causation hypotheses (M2/M3/M4); only M1 (direct personality causation) is rejected.
3. **Heterogeneity neglect**: The −20% propensity reduction is applied uniformly; differential response across HEXACO clusters is not modeled.
4. **Static framing**: No dynamic effects (e.g., institutional erosion over time) are captured.

These limitations are flagged in the Discussion section with proposed extensions (agent-based modeling; longitudinal causal inference; cluster-specific intervention modeling).
