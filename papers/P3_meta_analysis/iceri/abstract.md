# ICERI 2026 — Abstract

**Title**:
*Where the Big Five evidence really lives: A cross-tabulated
meta-analytic interaction of education level and academic discipline in
online learning.*

**Author**: Eisuke Tokiwa (SUNBLAZE Co., Ltd.; ORCID 0009-0009-7124-6669)
**Word target**: ~300 words (IATED ICERI standard).

> Numbers in `{{...}}` are filled from
> `papers/P3_meta_analysis/iceri/results/*.csv` after the cross-tab
> analysis is run.

---

**Background.** The parent meta-analysis (Research Square preprint, DOI
10.21203/rs.3.rs-9513298/v1) pools k = 9–10 primary studies per trait
across education levels and disciplines. Whether the pooled estimates
mask interaction patterns between education level and discipline remains
an open empirical question.

**Aim.** Cross-tabulate the primary-pool studies by 3 education levels
(K-12 / Undergraduate / Graduate) and 3 disciplines (STEM / Humanities /
Mixed), pool within each cell where k >= 2, and fit a long-format
interaction model on Fisher's z-transformed correlations.

**Methods.** Random-effects pooling (REML + HKSJ). Long-format mixed-
effects meta-regression with study random intercept. Heterogeneity
attribution via tau-squared decomposition. Empty cells reported transparently
as evidence about the field's coverage.

**Results.** Cross-tab cell counts: {{cell_counts_summary}}. Cells with
k >= 2: {{cells_pooled_count}} of 9. Largest cell: {{largest_cell_label}}
(k = {{largest_cell_k}}, r_C = {{largest_cell_rC}}). Interaction Wald
chi-squared({{wald_df}}) = {{wald_chi2}}, p = {{wald_p}}. Heterogeneity
absorbed by the cross-tab structure: delta tau-squared = {{delta_tau2}}
({{pct_tau2_absorbed}}% of overall tau-squared).

**Conclusion.** The Big Five — achievement correlation literature in
online learning is geographically and methodologically diverse but
*structurally narrow*: most evidence sits in undergraduate / mixed-
discipline cells. The interaction model {{detects / fails to detect}}
significant non-additive structure, but its inferential weight is limited
by the cell-level k. The clear field-level recommendation is to incentivise
research in K-12 STEM-online and Graduate-Humanities-online cells, where
the Big Five evidence is currently almost absent.

**Keywords**: meta-analysis, Big Five, online learning, education level,
discipline, interaction model, ICERI.

---

## Submission notes

- Disclose preprint per `templates/preprint_disclosure_template.md` (ICERI block).
- Virtual presentation possible — reduces travel commitment.
