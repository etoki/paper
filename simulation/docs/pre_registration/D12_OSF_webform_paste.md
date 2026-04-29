# OSF Standard Pre-Registration Web Form — Paste Sheet

**Source**: `D12_pre_registration_OSF.en.md` (English version)
**Created**: 2026-04-29
**Branch**: `claude/hexaco-harassment-simulation-69jZp`
**Author**: Eisuke Tokiwa (sole-authored, ORCID: 0009-0009-7124-6669)

---

## How to use this sheet

1. Log in to OSF, create a new project (e.g., "HEXACO 7-Typology Workplace Harassment Microsimulation").
2. **Upload supplementary files first** to the project:
   - `D12_pre_registration_OSF.pdf` (Japanese master)
   - `D12_pre_registration_OSF.en.pdf` (English version)
   - `D12_pre_registration_OSF.md` (Japanese markdown source)
   - `D12_pre_registration_OSF.en.md` (English markdown source)
3. Click **"New Registration"** → select **"OSF Preregistration"** template (also called "OSF Standard Pre-Registration").
4. For each field below, copy the labeled content into the corresponding OSF web form field.
5. After registering, the OSF DOI is issued. Record the DOI in the Header of both the JP and EN markdown documents (Section 14.3 lock procedure).

> **Tip**: Most OSF fields accept Markdown formatting (tables, headers, bold). Paste blocks below are formatted accordingly.

---

## OSF Field 1 — Title

```
HEXACO 7-Typology Workplace Harassment Microsimulation: Latent Prevalence Prediction and Target Trial Emulation of Personality-Based and Structural Counterfactuals in Japan
```

(Source: EN Section 1.1)

---

## OSF Field 2 — Description / Abstract

```
Workplace harassment is a multi-causal phenomenon. Organizational stressors (Bowling & Beehr 2006, ρ = .30–.53), subjective social status (Tsuno et al. 2015, OR = 4.21), industry composition, and legal/normative climate exert effects that are larger in magnitude than personality factors. The present study acknowledges these and instead isolates the personality contribution, asking how well a HEXACO 7-typology probabilistic model predicts Japanese workplace harassment prevalence.

Phase 1 (descriptive simulation): Using existing N = 354 (harassment behavior) and N = 13,668 (HEXACO clustering) data, we estimate 14-cell (7 types × 2 genders) conditional harassment propensities, scale these via bootstrap to the Japanese workforce (~68 million), and triangulate the resulting latent prevalence against the Ministry of Health, Labour and Welfare (MHLW) national surveys' expressed prevalence. Primary success criterion: MAPE ≤ 30% against MHLW 2016 (32.5%).

Phase 2 (intervention counterfactuals): Within the target trial emulation framework (Hernán & Robins 2020), we simulate three interventions: (A) universal HH (Honesty–Humility) intervention anchored in Kruse et al. (2014); (B) targeted high-risk-type intervention anchored in Hudson (2023), the primary intervention of interest; and (C) structural-only intervention anchored in Pruckner & Sausgruber (2013). For each, we estimate population-level harassment reduction.

No large language models are used. All mechanisms are transparent probability tables. The full preregistration document (English and Japanese versions) is attached as supplementary; the present field summarizes Sections 1.1–1.3 of that document.

Negative-result publication is committed in advance (Section 7): the author commits to journal submission regardless of whether MAPE is within or beyond the 30% / 60% thresholds. Target submission venue: Royal Society Open Science (Registered Report track), with seven fallback venues specified (Section 7.2).
```

(Source: EN Section 1.3 + brief synthesis from Sections 7 and 14)

---

## OSF Field 3 — Hypotheses

```
This preregistration commits to seven hypotheses (H1–H7), summarized below. Full operationalization, decision rules, and inference criteria are in the attached preregistration document (Section 1.4 + Section 6).

H1 (Phase 1 main hypothesis):
The aggregate national prediction obtained by scaling 14-cell (7 type × 2 gender) conditional harassment propensities to the Japanese workforce reproduces the MHLW 2016 (pre-Power Harassment Prevention Law) past-3-year harassment victimization rate of 32.5% within MAPE ≤ 30%.
- Primary validation target: MHLW 2016 R2 (32.5%)
- Secondary validation targets: MHLW 2020 R2 (31.4%), MHLW 2024 R5 (19.3%)

H2 (Phase 1 baseline hierarchy):
The mean absolute percentage error (MAPE) is monotonically non-increasing across the baseline hierarchy:
B0 (random) ≥ B1 (gender only) ≥ B2 (HEXACO 6-domain linear) ≥ B3 (7 typology) ≥ B4 (B3 + age + industry estimate + employment type).

H3 (Phase 1 latent vs expressed gap):
The gap between MHLW 2016 (pre-law, 32.5%) and our latent prediction is smaller than the gap between MHLW 2024 (post-law, 19.3%) and our latent prediction.

H4 (Phase 2 Counterfactual A: Universal HH intervention):
Under a +0.3 SD population-wide HH shift (a conservative discount of Kruse 2014's d = 0.71), the predicted national prevalence decreases by ΔP_A relative to baseline.
- Direction: ΔP_A < 0 (reduction); sensitivity range δ ∈ [0.1, 0.5] SD

H5 (Phase 2 Counterfactual B: Targeted intervention — primary intervention of interest):
Targeting high-risk types (primary: Cluster 0 = Self-Oriented Independent profile; secondary: Clusters 4 and 6) with a +0.4 SD HH shift yields a reduction ΔP_B such that the cost-effectiveness ratio (ΔP_B / number treated) exceeds that of Counterfactual A.
- Direction: ΔP_B < 0 AND ΔP_B / N_treated > ΔP_A / N_total; sensitivity range δ ∈ [0.2, 0.6] SD

H6 (Phase 2 Counterfactual C: Structural-only intervention):
Reducing all cell-conditional probabilities by 20% while leaving individual personality unchanged yields a reduction ΔP_C, but ΔP_C is smaller in magnitude than ΔP_B from Counterfactual B.
- Direction: ΔP_C < 0; sensitivity range effect_C ∈ [0.10, 0.30]

H7 (Phase 2 main contrast — primary predictive commitment):
ΔP_B > ΔP_A AND ΔP_B > ΔP_C (targeted intervention exceeds both universal and structural-only interventions in population-level reduction).

The author commits in advance (Section 7) to publish the study even if H1 fails (MAPE > 60%), if the H2 monotonicity is reversed, or if H7 is reversed. Inference criteria, multiple-comparison correction (Bonferroni–Holm), and deviation policy are specified in Sections 6.1–6.5 of the attached document.
```

(Source: EN Sections 1.4 + 6.1)

---

