# D12 OSF Pre-Registration — HEXACO 7-Typology Workplace Harassment Microsimulation (Phase 1 + Phase 2)

**Document type**: OSF Standard Pre-Registration draft (English version for OSF submission)
**Drafted**: 2026-04-29
**Branch**: `claude/hexaco-harassment-simulation-69jZp`
**Author (corresponding)**: Eisuke Tokiwa
**ORCID**: 0009-0009-7124-6669
**Affiliation**: SUNBLAZE Co., Ltd.
**Email**: eisuke.tokiwa@sunblaze.jp
**Status**: ⏳ DRAFT — to be finalized and registered on OSF **prior to Stage 0 code execution**
**Anchor template**: OSF Standard Pre-Registration (Bowman et al. 2020, https://osf.io/rh8jc) + Nosek et al. 2018 PNAS "preregistration revolution" 9-Challenge framework
**Companion document (Japanese master)**: `simulation/docs/pre_registration/D12_pre_registration_OSF.md`

---

## 0. Document Status (meta)

### 0.1 Purpose

This document is the English version of the master Pre-Registration draft for the HEXACO 7-typology workplace harassment microsimulation study. It is intended to be transcribed into the OSF Standard Pre-Registration web form and accompanied by the Japanese master as a supplementary PDF.

The Japanese master document (`D12_pre_registration_OSF.md`) is authoritative; this English version is a faithful translation prepared for OSF submission and international peer-review.

OSF submission workflow:
1. Internal review of this English version
2. Transcribe Sections 1–6 into the OSF Standard Pre-Registration web form
3. Attach the Japanese master as supplementary PDF
4. Obtain OSF DOI; record DOI in the Header of both documents
5. Lock both versions; subsequent edits follow Section 6.5 deviation policy

### 0.2 Compliance with Nosek et al. 2018 PNAS "preregistration revolution" 9 Challenges

| Challenge | Study situation | Response |
|---|---|---|
| **C3: Data Are Preexisting** | N=354 (harassment) and N=13,668 (clustering) are **prior IRB-approved data**. However, (a) the 7-type × gender 14-cell harassment cross-tabulation, (b) national-level aggregate predictions, and (c) Counterfactual A/B/C simulation outputs are **all unobserved**. | **Pure preregistration is achievable** for these unobserved analyses. Section 3.1 specifies who has observed what. |
| C1: Procedure changes during data collection | Simulation only; no new data collection | N/A |
| C2: Discovery of assumption violations during analysis | 14-cell main analysis uses frequentist bootstrap with BCa (light assumptions). 28-cell EB uses Beta-Binomial conjugate (method of moments + sensitivity sweep) for robustness. | Section 6.5 specifies deviation policy |
| **C6: Program-level null result reporting** | This study commits to publication even if MAPE > 60% (D-NEW8) | Section 7.3 codifies the commitment |
| C9: Selectivity in narrative inference | All Stage 3 sensitivity sweeps (V, f1, f2, EB strength, threshold, K) are pre-registered | Section 6.4 fixes the sweeps |

### 0.3 Relationship between this preregistration and the research plan

- **Research plan v6/v7** (`research_plan_harassment_typology_simulation.md`): Internal document containing motivation, theory, and literature foundation (L1+L2+L3+L4 layers, where L3/L4 are normative claims about social responsibility)
- **This preregistration**: Restricted to **L1 descriptive/predictive commitments** only
- Normative claims (L3/L4) are deliberately excluded from this preregistration following the four-layer separation principle (research plan Part 0.4)

---
