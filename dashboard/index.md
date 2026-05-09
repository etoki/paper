# Conference Submission Dashboard

**Owner**: Eisuke Tokiwa (single author)
**Generated**: 2026-05-09
**Goal**: 2 acceptances within 2026 to satisfy Keio SFC PhD point requirement.
**Strategy**: P3 (Big Five x online-learning meta-analysis) preprint stays on
Research Square; each conference paper adds a *novel analysis* not in the
preprint, and explicitly discloses the preprint to satisfy each venue's prior-publication policy.

> **Hierarchy**: future Journal version  >  Research Square preprint (current)  >  per-conference sub-papers

---

## Critical-path calendar (next 12 days)

| Date (2026) | Action |
|------------|--------|
| 05-09 (Sat) | Repo bootstrap + reconstructed strategy artefacts (this commit). ECEL extension email already sent (awaiting reply). |
| 05-10 (Sun) | Verify `papers/P3_meta_analysis/inputs/studies.csv` against the preprint Table 1 + `data_extraction_populated.csv`. Fix any discrepancies. |
| 05-11 (Mon) | Run `papers/P3_meta_analysis/ecel/scripts/run_modality_meta.py`. Inspect modality-stratified pooled effects + Q_between. |
| 05-12 (Tue) | Re-run with sensitivity layer (exclude beta-converted; exclude RoB<5; exclude COI study A-25). Lock numbers. |
| 05-13 (Wed) | Plug locked numbers into ECEL `abstract.md`. Internal author check (do **not** auto-send). |
| **05-14 (Thu)** | **ECEL Abstract submission deadline (Excellence Awards window).** |
| 05-15 (Fri) → 05-20 (Wed) | Draft ECEL Full Paper (10 pages, ACI Word template). Iterate. |
| **05-21 (Thu)** | **ECEL Full Paper submission deadline.** |

---

## Venue table

| ID | Venue | Location | Abstract | Full paper | Subset axis | Novel analysis |
|----|-------|----------|----------|------------|-------------|----------------|
| ECEL 2026 | European Conference on e-Learning | Lund, SE | **2026-05-14** | **2026-05-21** | Modality (sync/async/blended) | Modality-stratified meta-regression + interaction |
| IEEE TALE 2026 | IEEE Conf. on Teaching, Assessment, Learning for Engineering | Pattaya, TH | — | **2026-06-30** | Engineering / STEM students | LR / RF / XGBoost + SHAP + fairlearn |
| ICEEL 2026 | Int'l Conf. on Education and E-Learning | Tokyo, JP | — | **2026-06-30** | East Asia + Japan | Hofstede cultural-dimensions moderator |
| ICERI 2026 | Int'l Conf. of Education, Research, Innovation | Seville, ES | **2026-07-09** | TBD | Education-level x discipline | 3x3 cross-tab + interaction |

(Full schemas: `conferences/<id>.json`. Update those JSON files first, then regenerate this dashboard.)

---

## Status flags

- [x] ECEL extension email sent 2026-05-09 (info@academic-conferences.org). Awaiting reply.
- [ ] Studies.csv derived from preprint extraction.
- [ ] ECEL modality meta-analysis run.
- [ ] ECEL abstract finalised + author-checked.
- [ ] ECEL Abstract submitted.
- [ ] ECEL Full Paper submitted.
- [ ] IEEE TALE ML pipeline run.
- [ ] IEEE TALE paper drafted.
- [ ] ICEEL Hofstede analysis run.
- [ ] ICEEL paper drafted.
- [ ] ICERI cross-tab analysis run.
- [ ] ICERI abstract drafted.

---

## Hard rules (do not break)

1. **Preprint stays up.** Every conference submission must disclose `doi:10.21203/rs.3.rs-9513298/v1` in the cover letter / acknowledgment, using `templates/preprint_disclosure_template.md` as the base.
2. **Each conference paper must contain at least one analysis or synthesis that is NOT in the preprint** (see `novel_analysis` field of each JSON). This is the self-plagiarism firewall.
3. **No email is sent automatically.** All outgoing correspondence is drafted in `templates/` and reviewed by the author before sending. Sent items are mirrored back into `templates/` with a `_sent_<date>.md` suffix for the record.
4. **Single author only.** Do not add co-authors without explicit instruction.
5. **Numbers in any abstract or paper must trace to a CSV in `papers/P3_meta_analysis/<venue>/results/`** — never hand-typed.

---

## How to update

1. Edit `conferences/<id>.json` when an official CFP changes.
2. Edit the date table above to reflect the new deadlines.
3. Mark dashboard checkboxes as work progresses.
4. Re-run the venue's analysis script if input data changes; the script regenerates result CSVs in place.
