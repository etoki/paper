# ICERI 2026 — Abstract (numbers filled from results CSVs)

**Title**:
*Where the Big Five evidence really lives: A cross-tabulated
meta-analytic interaction of education level and academic discipline in
online learning.*

**Author**: Eisuke Tokiwa (SUNBLAZE Co., Ltd.; ORCID 0009-0009-7124-6669)
**Word target**: ~300 words.

**Numerical sources**:
- `results/cross_tab_pools.csv`
- `results/interaction_terms.csv`
- `results/cross_tab_summary.md`

---

## Draft (numbers filled 2026-05-09 after run)

**Background.** The parent meta-analysis (Research Square preprint, DOI
10.21203/rs.3.rs-9513298/v1) pools k = 9 — 10 primary studies per Big
Five trait across heterogeneous education levels (K-12 / Undergraduate /
Graduate / Mixed) and disciplines (STEM / Humanities / Psychology /
Mixed). Education-level was pre-registered as a moderator but was not
quantitatively executed (k constraint); discipline was not pre-registered
at all. Whether the pooled estimates mask interaction patterns between
level and discipline remains untested.

**Aim.** Cross-tabulate the primary-pool studies by 3 education levels
collapsed (K-12 / UG / Graduate / Mixed_UG_Grad) and 4 disciplines (STEM
/ Humanities / Psychology / Mixed), pool within each cell where k >= 2,
and fit a long-format weighted-OLS interaction model on Fisher's z
correlations.

**Methods.** Random-effects pooling (REML + HKSJ). Long-format model
with study random-style weights = 1 / (v + tau-squared). tau-squared
estimated as median of per-trait REML estimates. Joint Wald test on
level x discipline interaction terms.

**Results.** Cross-tab cell-counts confirm an undergraduate-dominated
literature: of 12 (4 levels x 3 disciplines + Mixed buckets) candidate
cells, only **2 cells reach k >= 2**: UG x Mixed (k = 3) and UG x
Psychology (k = 3 for C / N, k = 2 for O / E / A). Within these, UG x
Psychology shows the strongest Conscientiousness correlation
(r = 0.292 [0.123, 0.444]) and a positive UG x Psychology Agreeableness
effect (r = 0.152 [0.098, 0.205]). Joint Wald test on all six level x
discipline interactions: **chi-squared(6) = 1.64, p = .95** — the
interaction is null with the present evidence base.

**Implications.** The empirical message is asymmetric: where evidence is
dense (UG x Psychology / UG x Mixed), Conscientiousness is the strongest
positive predictor; everywhere else, k <= 1 prohibits inference. The
structural message is sharper: K-12 STEM-online and Graduate-Humanities-
online are evidence deserts. The pooled Big Five — achievement
correlations from any current meta-analysis should not be applied
uniformly across teaching contexts; the field needs targeted recruitment
of K-12 STEM and Graduate-Humanities online samples.

**Keywords**: meta-analysis, Big Five, online learning, education level,
discipline, interaction model, evidence map, ICERI.
