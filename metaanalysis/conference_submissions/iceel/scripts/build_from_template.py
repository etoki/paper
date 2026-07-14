"""
Build ICEEL EL2036 full paper from the ICEEL template.

Approach: load the vendor template, keep its styles / margins / sectPr,
strip all placeholder body content, then rebuild the paper body using the
template's own paragraph styles (Title_document / Authors / Affiliation /
Abstract / KeyWords / Head1 / Head2 / Head3 / Para / PostHeadPara /
TableCaption / FigureCaption / AckHead / AckPara / ReferenceHead /
Bib_entry).

The paper structure is hard-coded here rather than parsed from markdown so
that table cells, style choices, and inline emphasis are exactly right.
Numerical values are traced to the CSVs under
metaanalysis/conference_submissions/iceel/results/ .
"""
from __future__ import annotations

from copy import deepcopy
from pathlib import Path

from docx import Document
from docx.oxml.ns import qn
from docx.shared import Cm, Pt

ROOT = Path(__file__).resolve().parents[4]
ICEEL = ROOT / "metaanalysis" / "conference_submissions" / "iceel"
TEMPLATE = ICEEL / "ICEEL-Full paper Template.docx"
OUTPUT = ICEEL / "full_paper.docx"
PRISMA_FIG = ROOT / "metaanalysis" / "conference_submissions" / "figures" / "prisma_flow_iceel.png"

TITLE = (
    "Cultural dimensions in online learning: A Hofstede-moderated "
    "within-Asia synthesis of the Big Five and academic achievement, "
    "with a focused look at the Japanese context"
)

AUTHOR = "Eisuke Tokiwa"
AFFILIATION = (
    "SUNBLAZE Co., Ltd., Tokyo, Japan; eisuke.tokiwa@sunblaze.jp; "
    "ORCID 0009-0009-7124-6669"
)

ABSTRACT_BLOCKS = [
    ("Background. ",
     "The parent meta-analysis (Research Square preprint, "
     "DOI 10.21203/rs.3.rs-9513298/v1) reports a highly significant "
     "Extraversion × Region moderator in Big Five – "
     "academic-achievement research on online learning "
     "(Q_between = 46.43, p < .001; Asian pooled r = −0.131 vs "
     "non-Asian r = +0.050), but does not decompose the Asia bin into its "
     "country-level constituents."),
    ("Objective. ",
     "To (i) attach Hofstede 6-D country scores to the Asian primary-pool "
     "studies, and (ii) test whether observed Big Five – outcome "
     "directions in Japan-based studies match Hofstede & McCrae (2004) "
     "trait-dimension predictions."),
    ("Methods. ",
     "The Asian primary pool comprises k = 2 studies with extractable "
     "Pearson r (Yu 2021, China; Rivers 2021, Japan); a third Japan-based "
     "study (Tokiwa 2025, K-12) contributes descriptive Spearman "
     "ρ. Random-effects pools (REML + HKSJ) replicate the parent "
     "preprint. Because residual df = 0, single-dimension weighted-OLS "
     "Hofstede meta-regressions yield descriptive slopes only. Direction "
     "predictions for Japan, derived from Hofstede & McCrae (2004), are "
     "cross-referenced against observed r in both Japan studies."),
    ("Results. ",
     "The direction-match analysis yields 3 of 5 trait matches "
     "(Conscientiousness+, Extraversion−, Neuroticism+; "
     "Agreeableness and Openness mismatch). The diagnostic finding is "
     "Neuroticism+: observed in both Japan studies, doubly predicted by "
     "Japan's very high Uncertainty Avoidance (UAI = 92) and Masculinity "
     "(MAS = 95), and opposite to the consistent N− pattern reported "
     "in Western Big Five meta-analyses (Poropat 2009; Mammadov 2022). "
     "The Agreeableness mismatch (predicted −, observed +) suggests "
     "that online-learning behaviour-level engagement may decouple from "
     "population trait-level A in collectivist contexts."),
    ("Conclusion. ",
     "Country-level Hofstede context materially reshapes the Big Five "
     "– achievement pattern in ways the binary Asia/non-Asia "
     "contrast obscures. With k = 2, the contribution is methodological "
     "rather than confirmatory: it flags within-Asia heterogeneity and "
     "instrument confounds (60-item BFI-2-J vs 10-item TIPI-J) as "
     "prerequisites for future Japan-specific syntheses."),
]

KEYWORDS = (
    "Additional Keywords and Phrases: meta-analysis; Big Five personality "
    "traits; online learning; Hofstede cultural dimensions; Japan; "
    "cross-cultural psychology"
)


def clear_body(doc):
    """Strip all body-level children EXCEPT the trailing sectPr."""
    body = doc.element.body
    sect_pr = body.find(qn("w:sectPr"))
    for ch in list(body):
        if ch is sect_pr:
            continue
        body.remove(ch)
    return body, sect_pr


def add_para(doc, style, text=""):
    """Append a paragraph with the given style name."""
    p = doc.add_paragraph(style=style)
    if text:
        p.add_run(text)
    return p


def add_para_with_lead(doc, style, lead, tail):
    """Append a paragraph with a bold lead-in and normal tail."""
    p = doc.add_paragraph(style=style)
    lead_run = p.add_run(lead)
    lead_run.bold = True
    if tail:
        p.add_run(tail)
    return p


def add_table(doc, header, rows, caption=None):
    """Add a simple table with header row and body rows."""
    if caption:
        add_para(doc, "TableCaption", caption)
    n_cols = len(header)
    t = doc.add_table(rows=1 + len(rows), cols=n_cols)
    t.style = "Table Grid"
    hdr_cells = t.rows[0].cells
    for j, h in enumerate(header):
        hdr_cells[j].text = ""
        p = hdr_cells[j].paragraphs[0]
        run = p.add_run(h)
        run.bold = True
        run.font.size = Pt(8)
    for i, row in enumerate(rows, start=1):
        cells = t.rows[i].cells
        for j, v in enumerate(row):
            cells[j].text = ""
            p = cells[j].paragraphs[0]
            run = p.add_run(str(v))
            run.font.size = Pt(8)
    return t


def add_figure(doc, path, caption, width_cm=12.0):
    if path.exists():
        doc.add_picture(str(path), width=Cm(width_cm))
    add_para(doc, "FigureCaption", caption)


# ---------------------------------------------------------------------------
# Body content
# ---------------------------------------------------------------------------
BODY = [
    ("Head1", "1 INTRODUCTION"),
    ("PostHeadPara",
     "The Big Five – academic-achievement link is robust at the "
     "trait level (Poropat 2009; Mammadov 2022; Chen, Cheung, & Zeng "
     "2025) but is increasingly recognised as culturally conditional, "
     "with Mammadov flagging Asian samples as showing amplified "
     "Conscientiousness and sometimes inverted Extraversion. The parent "
     "preprint (Tokiwa, 2026, DOI 10.21203/rs.3.rs-9513298/v1) confirms "
     "a binary Asia / non-Asia contrast with Extraversion × Region "
     "the headline (Q_between = 46.43, p < .001; Asian r = −0.131 "
     "vs non-Asian r = +0.050) but does not decompose the Asian bin "
     "into its country-level constituents — even though Japan, "
     "China, Korea, and Taiwan differ sharply on Hofstede Power "
     "Distance, UAI, MAS, and IND scores (Hofstede Insights 2024)."),
    ("Para",
     "This paper asks: once you decompose the Asia bin, do Hofstede "
     "cultural dimensions predict the direction of Big Five – "
     "achievement correlations within Asia? With only k = 2 Asian "
     "primary-pool studies contributing extractable r (A-28 Yu, China; "
     "A-31 Rivers, Japan), no inference is possible from the "
     "meta-regression. We therefore reframe the analysis as a Hofstede "
     "direction-prediction match against the two Japan-based studies "
     "(Tokiwa 2025 K-12, Rivers 2021 UG; Hofstede & McCrae 2004 "
     "trait-dimension correlations as the prediction source). The "
     "headline finding is a 3-of-5 trait direction-match, with the "
     "Neuroticism positive direction in both Japan studies opposite to "
     "the Western pattern and doubly predicted by Japan's very high UAI "
     "and MAS."),

    ("Head1", "2 RELATED WORK"),
    ("Head2", "2.1 Hofstede's framework, Big Five linkages, and educational implications"),
    ("PostHeadPara",
     "Hofstede's (2001) 2nd-edition model operationalises national "
     "culture in 6 dimensions — Power Distance (PDI), Individualism "
     "(IDV), Masculinity (MAS), Uncertainty Avoidance (UAI), Long-Term "
     "Orientation (LTO), Indulgence (IND); latest scores from Hofstede "
     "Insights (2024). Critiques (Minkov & Hofstede, 2014; McSweeney, "
     "2002) for ecological-fallacy risk and for the 1970s IBM-sample "
     "provenance are acknowledged: we use the dimensions as "
     "country-level proxies for shared cultural orientation, not as "
     "individual-level claims."),
    ("Para",
     "Big Five × Hofstede linkages. Hofstede & McCrae (2004) "
     "report country-level correlations between aggregate NEO-PI-R Big "
     "Five trait means and Hofstede dimensions. Correlations relevant "
     "to the Asian subset (HM04 Table 4): UAI × N r = +.31; "
     "UAI × A r = −.45; PDI × C r = +.43; "
     "PDI × E r = −.31; IDV × E r = +.39; "
     "MAS × N r = +.30; MAS × O r = +.37. Hofstede (1986), "
     "applying the 4-D model directly to education, characterises "
     "strong-UAI societies as ones in which 'students prefer "
     "structured learning situations … rewarded for accuracy in "
     "problem solving', and where students 'are allowed to behave "
     "emotionally' — sketching a plausible mechanism by which "
     "trait Neuroticism (anxiety, sensitivity) may promote academic "
     "engagement in high-UAI contexts. Schmitt et al. (2007) place "
     "Japan's trait N mean at 57.87, near the top of 56 nations; Allik "
     "& McCrae (2004) further note that Japan does not cluster cleanly "
     "with other East Asian samples in their 36-culture geography. "
     "Together these undermine the 'Asia' moderator level as a "
     "substitute for country-specific analysis."),
    ("Para",
     "Japan's profile (PDI = 54, IDV = 46, MAS = 95, UAI = 92) yields "
     "HM04-derived direction predictions tabulated in Section 4.4 "
     "(Table 4). The N+ prediction is particularly diagnostic because "
     "it is opposite to the consistent N− pattern in Western Big "
     "Five – achievement meta-analyses (Poropat, 2009; Mammadov, "
     "2022)."),

    ("Head2", "2.2 East Asian samples in personality – academic-achievement research"),
    ("PostHeadPara",
     "Reviews specific to East Asian samples (Chen et al. 2025; "
     "Mammadov 2022) report two recurring patterns: (i) Conscientiousness "
     "is at least as strong a predictor of achievement as in Western "
     "samples, often slightly stronger in collectivist sub-populations; "
     "and (ii) Extraversion's role is attenuated or reversed in "
     "collectivist contexts where individual social initiative is "
     "valued less highly. The preprint's Asian subset findings "
     "(E r = −0.131; C r = +0.111) are consistent with this pattern."),

    ("Head2", "2.3 Japanese online learning context"),
    ("PostHeadPara",
     "Japan-specific online-learning research is dominated by two "
     "patterns. First, the use of the StudySapuri commercial LMS by "
     "K-12 cohorts (asynchronous, self-paced video lectures plus "
     "practice problems) — A-25 Tokiwa (2025) is one such cohort. "
     "Second, the use of Moodle in undergraduate language education "
     "(asynchronous LMS-paced reading and writing assignments) — "
     "A-31 Rivers (2021) is the example here. Both studies use "
     "asynchronous formats, but their personality instruments (60-item "
     "BFI-2-J vs 10-item TIPI-J) differ by an order of magnitude in "
     "measurement granularity, which is a major between-study confound "
     "that the present synthesis surfaces."),

    ("Head1", "3 METHOD"),
    ("Head2", "3.1 Data"),
    ("PostHeadPara",
     "The corpus is inherited from the parent preprint via the derived "
     "studies dataset. PRISMA 2020 standards (Page et al., 2021) were "
     "followed; Figure 1 reproduces the flow diagram with the ICEEL "
     "terminal box marking the Asian-subset extraction (k = 2 with "
     "extractable r per trait; A-25 Tokiwa Japan retained for "
     "qualitative synthesis only). The Asian subset is extracted by "
     "region == 'Asia'. After requiring extractable Pearson r per "
     "trait, k = 2 Asian primary-pool studies remain: A-28 Yu (China; "
     "β-converted r) and A-31 Rivers (Japan; direct r). Two "
     "further Asian primary-pool studies (A-25 Tokiwa, A-26 Wang) are "
     "present in the qualitative synthesis but do not contribute "
     "extractable r values."),
    ("_FIGURE", None),  # placeholder — insert PRISMA figure

    ("Head2", "3.2 Hofstede cultural-dimensions table"),
    ("PostHeadPara",
     "Country-level Hofstede 6-D scores are encoded inline in the "
     "analysis script using the canonical Hofstede Insights values. For "
     "the Asian primary-pool countries:"),
    ("Para",
     "Japan: PDI = 54, IDV = 46, MAS = 95, UAI = 92, LTO = 88, IND = 42. "
     "China: PDI = 80, IDV = 20, MAS = 66, UAI = 30, LTO = 87, IND = 24."),
    ("Para",
     "These two countries differ most sharply on Individualism (Japan "
     "+26), Power Distance (China +26), Masculinity (Japan +29), "
     "Uncertainty Avoidance (Japan +62), and Indulgence (Japan +18), "
     "with similar Long-Term Orientation."),

    ("Head2", "3.3 Statistical model"),
    ("PostHeadPara",
     "Per trait, REML + HKSJ pooled r is computed on the k = 2 Asian "
     "primary-pool studies (back-transformed 95 % CI reported). For "
     "each (trait, dimension) pair, a 2-parameter weighted-OLS Hofstede "
     "meta-regression is fit (weights = 1 / (v + τ²), "
     "τ² from REML). Because k = 2, residual df = 0; slopes "
     "are computable but no SE / t / p estimable, so slopes are "
     "reported as descriptive only."),
    ("Head3", "3.3.1 Japan synthesis."),
    ("PostHeadPara",
     "The two Japan primary-pool studies are tabulated side-by-side on "
     "N, modality, education level, instrument, and per-trait r (where "
     "available). The narrative discussion is the analytic vehicle; no "
     "formal pooling is attempted at k = 2."),

    ("Head2", "3.4 Reproducibility"),
    ("PostHeadPara",
     "All numerical results are produced by "
     "run_hofstede_meta.py under scripts/. Pooling primitives (Fisher "
     "z, var z, REML τ², HKSJ-adjusted CI) are imported from "
     "analysis/pool.py in the parent preprint repository."),

    ("Head1", "4 RESULTS"),
    ("Head2", "4.1 Asian-subset pooled correlations"),
    ("PostHeadPara",
     "Table 1 reports the per-trait pooled correlations on the Asian "
     "primary-pool subset."),
    ("_TABLE1", None),
    ("Para",
     "C, E, and N pools have I² = 0 (the two studies agree closely "
     "in z-space); O and A pools have I² = 96 % driven by sharply "
     "different point estimates. These numbers replicate the preprint "
     "Region: Asia row exactly."),

    ("Head2", "4.2 Hofstede single-dimension meta-regression"),
    ("PostHeadPara",
     "Table 2 reports the per-trait per-dimension slope estimates. "
     "With df_resid = 0 the inferential columns (SE, t, p) are not "
     "estimable; only the slope is reported."),
    ("_TABLE2", None),
    ("Para",
     "(Selected rows; full table in "
     "results/hofstede_meta_regression.csv. PDI vs IDV slopes are "
     "mirror images because Japan and China differ on these in "
     "opposite directions; LTO is exception-near-zero because the two "
     "countries' LTO scores happen to be very close.)"),
    ("Para",
     "The pattern that does emerge is consistent with the directional "
     "theoretical expectation: higher Individualism (lower Collectivism) "
     "is associated with more positive Extraversion – achievement "
     "correlations, since Extraversion-as-asset is tied to individualistic "
     "affordances. But with df_resid = 0 these slopes carry no "
     "inferential weight; they are visualisations of the two-country "
     "contrast, not statistical tests."),

    ("Head2", "4.3 Japan synthesis"),
    ("PostHeadPara",
     "Table 3 compares the two Japan-based primary-pool studies "
     "side-by-side."),
    ("_TABLE3", None),
    ("Para",
     "A-25 Tokiwa's correlations are reported as Spearman ρ "
     "against StudySapuri usage outcomes (Tokiwa 2025, Tables 3 + body "
     "text); they did not survive the meta-analysis extraction protocol "
     "(which required Pearson r against achievement, not Spearman "
     "against engagement) but are included here as the "
     "original-publication descriptive comparator. A-31 Rivers's r "
     "values are extractable Pearson correlations against course grade "
     "and contribute to the meta-analysis pool."),
    ("Para",
     "The instrument-heterogeneity contrast is striking: A-25 uses the "
     "60-item BFI-2-J (~12 items per trait, alphas .80 – .96); "
     "A-31 uses the 10-item TIPI-J (2 items per trait, alphas "
     "constrained by definition to .50 – .60 for most traits). "
     "The per-trait reliability difference alone could account for "
     "substantial measurement-noise heterogeneity even before any "
     "cultural-context interaction is considered."),

    ("Head2", "4.4 Hofstede direction-prediction match in Japan studies"),
    ("PostHeadPara",
     "Table 4 cross-references HM04-derived direction predictions for "
     "Japan with observed Big Five – outcome direction in the two "
     "Japan studies. Direction is coded + if observed magnitude "
     "≥ +0.05, − if ≤ −0.05, flat otherwise; A-25 "
     "Tokiwa's direction is the dominant-pattern Spearman ρ "
     "against StudySapuri outcomes."),
    ("_TABLE4", None),
    ("Para",
     "3 of 5 traits (C, E, N) match. The Neuroticism+ match is the "
     "headline: opposite to the Western N− pattern (Poropat 2009; "
     "Mammadov 2022), supported in both Japan studies, and doubly "
     "predicted via UAI and MAS. The A mismatch (predicted −, "
     "observed +) is informative rather than refuting: it suggests "
     "online-learning behaviour-level A engagement may decouple from "
     "population trait-level A in collectivist contexts — a "
     "hypothesis future cross-cultural syntheses can test directly."),

    ("Head1", "5 DISCUSSION"),
    ("Head2", "5.1 The within-Asia evidence problem"),
    ("PostHeadPara",
     "The headline finding of this paper is structural: the binary "
     "Asia/non-Asia contrast in the parent preprint, while statistically "
     "significant for Extraversion, is supported by a very thin Asian "
     "primary pool (k = 2 with extractable r). This is not a critique "
     "of the preprint — its k constraints are entirely transparent "
     "— but a reminder that 'Asia' as a moderator level is doing "
     "more work than k = 2 can sustain. The interpretation "
     "'Extraversion is more negative in Asia' is best read as "
     "'Extraversion is more negative in our two Asian sample-points "
     "than in our seven non-Asian sample-points', and the fact that "
     "the two Asian points happen to be from different countries "
     "(China, Japan) is a confound that no current corpus can resolve."),

    ("Head2", "5.2 Hofstede slopes as direction-finders"),
    ("PostHeadPara",
     "Even at k = 2, the slope signs in Table 2 are interpretable as "
     "directional indicators rather than tests. The Extraversion slope "
     "on Individualism is small but consistent with theoretical "
     "prediction (+ slope = more positive Extraversion in more "
     "individualistic contexts; the two Asian countries differ by "
     "IDV = 26 points). The Conscientiousness slope on Long-Term "
     "Orientation is small and positive (+0.0376), which is consistent "
     "with cultural amplification of self-regulation in "
     "long-term-oriented societies — but at k = 2 this could "
     "reflect any number of alternative explanations (instrument "
     "differences, education-level differences, era differences)."),

    ("Head2", "5.3 The Japan instrument problem"),
    ("PostHeadPara",
     "The Japan synthesis's strongest contribution is the surfacing of "
     "instrument heterogeneity as a major confound. Two Japan-based "
     "primary-pool studies, both asynchronous, both with ~100 – "
     "150 students, but with personality measures differing by a "
     "factor of 6 in number of items. Future syntheses that aim to "
     "make Japan-specific claims should require either (a) "
     "standardised instrument families across studies or (b) explicit "
     "instrument-level moderator analyses. The current corpus does "
     "neither."),

    ("Head2", "5.4 The Hofstede direction-match in Japan"),
    ("PostHeadPara",
     "The 3-of-5 match in Table 4 — particularly the "
     "doubly-supported Neuroticism positive direction — is the "
     "strongest cultural-context finding in this paper. The N+ "
     "direction is opposite to the Western N null/negative pattern "
     "(Poropat 2009; Mammadov 2022) and is consistent with Hofstede's "
     "(1986) characterisation of strong-UAI societies as ones in which "
     "students are rewarded for structured, accuracy-focused study "
     "— behaviours plausibly fuelled by trait-level anxiety. The "
     "Agreeableness mismatch (predicted − by HM04 UAI .45; "
     "observed + in both Japan studies) is informative rather than "
     "refuting: population trait scores need not predict "
     "individual-level personality–outcome correlations within a "
     "particular activity domain, and online-learning may afford "
     "agreeable students engagement channels that decouple from "
     "population A."),

    ("Head2", "5.5 Self-plagiarism firewall"),
    ("PostHeadPara",
     "The within-Asia decomposition and the Hofstede direction-match "
     "analysis (Table 4) are both absent from the parent preprint, "
     "which only reports the binary Asia / non-Asia contrast."),

    ("Head1", "6 LIMITATIONS"),
    ("PostHeadPara",
     "The dominant limitation is k. With k = 2 Asian primary-pool "
     "studies, the Hofstede meta-regression has zero residual degrees "
     "of freedom; slopes and direction-matches are descriptive only. "
     "The Hofstede framework itself is critiqued (Minkov & Hofstede, "
     "2014; McSweeney, 2002) for ecological-fallacy risk and IBM-sample "
     "provenance; Minkov-revised dimensions might yield different "
     "direction predictions. Western-derived Big Five may not capture "
     "culture-specific personality structure in Japan (Cheung et al., "
     "2003). Single-author, single-coder; country-to-Hofstede-dimension "
     "assignment uses the canonical Hofstede Insights (2024) table "
     "rather than hand-coding."),

    ("Head1", "7 CONCLUSION"),
    ("PostHeadPara",
     "The contribution of this paper is threefold: (a) the binary "
     "Asia / non-Asia contrast in the parent preprint conceals a "
     "within-Asia structure that the current corpus cannot statistically "
     "resolve (k = 2); (b) a Hofstede direction-prediction analysis "
     "(Hofstede & McCrae 2004 trait-dimension correlations) yields 3 of "
     "5 trait direction-matches in two Japan-based studies, with the "
     "Neuroticism positive direction in both being doubly predicted by "
     "Japan's very high UAI and MAS and being opposite to the Western "
     "pattern; and (c) the Agreeableness mismatch (predicted −, "
     "observed +) opens an empirical question about whether "
     "collectivist online-learning behaviour decouples from population "
     "trait-level A."),
    ("Para",
     "Practically, Japanese ed-tech researchers should treat Big "
     "Five-based personalisation claims with caution: only "
     "Conscientiousness shows a consistent direction; instrument "
     "standardisation (60-item BFI-2-J vs 10-item TIPI-J) and "
     "consistent trait-by-achievement reporting are prerequisites for "
     "any future Hofstede moderator analysis with df > 0. Future work: "
     "extend the Asian primary-pool corpus by targeted recruitment of "
     "Korean and Taiwanese samples and test Minkov-revised dimensions "
     "as a robustness check on the HM04 mapping."),
]

TABLE1_HEADER = ["Trait", "k", "N (pooled)", "r [95 % CI]", "I²"]
TABLE1_ROWS = [
    ["O", "2", "1301", "0.164 [−0.989, 0.994]", "96.0 %"],
    ["C", "2", "1301", "0.111 [−0.039, 0.257]", "0.0 %"],
    ["E", "2", "1301", "−0.131 [−0.314, 0.061]", "0.0 %"],
    ["A", "2", "1301", "0.330 [−0.981, 0.995]", "95.6 %"],
    ["N", "2", "1301", "0.089 [0.008, 0.169]", "0.0 %"],
]

TABLE2_HEADER = ["Trait", "Dimension", "slope (Fisher z per unit)", "note"]
TABLE2_ROWS = [
    ["O", "PDI", "+0.0168", "descriptive"],
    ["O", "IDV", "−0.0168", "descriptive"],
    ["O", "LTO", "−0.4372", "descriptive"],
    ["C", "PDI", "−0.0014", "descriptive"],
    ["C", "IDV", "+0.0014", "descriptive"],
    ["C", "LTO", "+0.0376", "descriptive"],
    ["E", "PDI", "+0.0018", "descriptive"],
    ["E", "IDV", "−0.0018", "descriptive"],
    ["E", "LTO", "−0.0481", "descriptive"],
    ["A", "PDI", "+0.0162", "descriptive"],
    ["A", "LTO", "−0.4201", "descriptive"],
    ["N", "PDI", "−0.0008", "descriptive"],
    ["N", "IDV", "+0.0008", "descriptive"],
    ["N", "LTO", "+0.0202", "descriptive"],
]

TABLE3_HEADER = ["Field", "A-25 Tokiwa (2025)", "A-31 Rivers (2021)"]
TABLE3_ROWS = [
    ["N (analytic)", "103", "149"],
    ["Education level", "K-12 (Year 3 high school)", "Undergraduate"],
    ["Modality", "Asynchronous (StudySapuri LMS)", "Asynchronous (Moodle)"],
    ["Instrument", "BFI-2-J (60 items)", "TIPI-J (10 items)"],
    ["Outcome", "Test completion + mastery (Spearman ρ)", "Course grade (Pearson r)"],
    ["Era", "post-COVID", "COVID"],
    ["r_O", "not in extractable column", "−0.066"],
    ["r_C", "ρ = +0.30 to +0.35", "+0.144"],
    ["r_E", "mixed (Assertiveness +0.26; Sociability −)", "−0.173"],
    ["r_A", "ρ = +0.27 to +0.29 (Lectures Watched)", "+0.118"],
    ["r_N", "ρ = +0.24 to +0.29 (Anxiety subscale)", "+0.107"],
]

TABLE4_HEADER = ["Trait", "HM04 prediction (Japan)", "Driver", "Tokiwa", "Rivers", "Match?"]
TABLE4_ROWS = [
    ["C", "+", "PDI .43", "+ (.30–.35)", "+ (.14)", "✓"],
    ["E", "−", "IDV .39", "mixed", "− (−.17)", "✓"],
    ["N", "+", "UAI .31 + MAS .30 (double)", "+ (.24–.29)", "+ (.11)", "✓"],
    ["A", "−", "UAI −.45", "+ (.27–.29)", "+ (.12)", "✗"],
    ["O", "+ (weak)", "MAS .37", "weak", "− (−.07)", "✗"],
]

TABLE_MAP = {
    "_TABLE1": (TABLE1_HEADER, TABLE1_ROWS, "Table 1: Asian-subset random-effects pooled correlations (REML + HKSJ; k = 2 per trait; A-28 Yu China; A-31 Rivers Japan)."),
    "_TABLE2": (TABLE2_HEADER, TABLE2_ROWS, "Table 2: Hofstede single-dimension meta-regression slopes, descriptive only (k = 2; df_resid = 0)."),
    "_TABLE3": (TABLE3_HEADER, TABLE3_ROWS, "Table 3: Japan-based primary-pool studies, narrative comparison."),
    "_TABLE4": (TABLE4_HEADER, TABLE4_ROWS, "Table 4: Hofstede-predicted vs observed direction in Japan-based studies."),
}


ACKNOWLEDGMENTS_TEXT = (
    "This paper was prepared by the sole author. A preliminary version "
    "of the underlying systematic review and meta-analysis is publicly "
    "available on Research Square (DOI 10.21203/rs.3.rs-9513298/v1, "
    "posted 27 April 2026). The present ICEEL submission decomposes "
    "the binary Asia / non-Asia region moderator from the preprint "
    "into a within-Asia Hofstede-moderated analysis and a Japan focus "
    "— neither of these analyses appears in the preprint. "
    "ORCID: 0009-0009-7124-6669. The author used a large language "
    "model (Anthropic's Claude) as a research assistant for drafting, "
    "code generation, and formatting; all statistical results were "
    "re-executed by the author and all bibliographic entries verified "
    "against primary-source PDFs."
)

REFERENCES = [
    # Primary studies
    "Rivers, D. J. 2021. The role of personality traits and online "
    "academic self-efficacy in acceptance, actual use and achievement "
    "in Moodle. Education and Information Technologies 26, 4 (2021), "
    "4353–4378. DOI: 10.1007/s10639-021-10478-3.",
    "Tokiwa, E. 2025. Who excels in online learning in Japan? "
    "Frontiers in Psychology 16, Article 1420996 (2025). "
    "DOI: 10.3389/fpsyg.2025.1420996.",
    "Wang, P., Wang, F., and Li, Z. 2023. Exploring the ecosystem of "
    "K-12 online learning: An empirical study of impact mechanisms in "
    "the post-pandemic era. Frontiers in Psychology 14, 1241477 "
    "(2023). DOI: 10.3389/fpsyg.2023.1241477.",
    "Yu, Z. 2021. The effects of gender, educational level, and "
    "personality on online learning outcomes during the COVID-19 "
    "pandemic. International Journal of Educational Technology in "
    "Higher Education 18, 1, Article 14 (2021). "
    "DOI: 10.1186/s41239-021-00252-3.",
    # Benchmark meta-analyses
    "Chen, S., Cheung, A. C. K., and Zeng, Z. 2025. Big Five "
    "personality traits and university students' academic performance: "
    "A meta-analysis. Personality and Individual Differences 240, "
    "113163 (2025). DOI: 10.1016/j.paid.2025.113163.",
    "Mammadov, S. 2022. Big Five personality traits and academic "
    "performance: A meta-analysis. Journal of Personality 90, 2 "
    "(2022), 222–255. DOI: 10.1111/jopy.12663.",
    "Poropat, A. E. 2009. A meta-analysis of the five-factor model of "
    "personality and academic performance. Psychological Bulletin 135, "
    "2 (2009), 322–338. DOI: 10.1037/a0014996.",
    # Hofstede framework
    "Hofstede, G. 1986. Cultural differences in teaching and learning. "
    "International Journal of Intercultural Relations 10, 3 (1986), "
    "301–320. DOI: 10.1016/0147-1767(86)90015-5.",
    "Hofstede, G. 2001. Culture's Consequences: Comparing Values, "
    "Behaviors, Institutions and Organizations across Nations "
    "(2nd. ed.). Sage Publications, Thousand Oaks, CA.",
    "Hofstede, G. and McCrae, R. R. 2004. Personality and culture "
    "revisited: Linking traits and dimensions of culture. "
    "Cross-Cultural Research 38, 1 (2004), 52–88. "
    "DOI: 10.1177/1069397103259443.",
    "Hofstede Insights. 2024. Country comparison tool. Retrieved 2024 "
    "from https://www.hofstede-insights.com/.",
    "Allik, J. and McCrae, R. R. 2004. Toward a geography of "
    "personality traits: Patterns of profiles across 36 cultures. "
    "Journal of Cross-Cultural Psychology 35, 1 (2004), 13–28. "
    "DOI: 10.1177/0022022103260382.",
    "McCrae, R. R. and Terracciano, A. 2005. Universal features of "
    "personality traits from the observer's perspective: Data from 50 "
    "cultures. Journal of Personality and Social Psychology 88, 3 "
    "(2005), 547–561. DOI: 10.1037/0022-3514.88.3.547.",
    "Schmitt, D. P., Allik, J., McCrae, R. R., and Benet-Martínez, V. "
    "2007. The geographic distribution of Big Five personality traits: "
    "Patterns and profiles of human self-description across 56 nations. "
    "Journal of Cross-Cultural Psychology 38, 2 (2007), 173–212. "
    "DOI: 10.1177/0022022106297299.",
    "Migliore, L. A. 2011. Relation between big five personality "
    "traits and Hofstede's cultural dimensions: Samples from the USA "
    "and India. Cross Cultural Management: An International Journal "
    "18, 1 (2011), 38–54. DOI: 10.1108/13527601111104287.",
    "Cheung, F. M., Cheung, S. F., Wada, S., and Zhang, J. 2003. "
    "Indigenous measures of personality assessment in Asian countries: "
    "A review. Psychological Assessment 15, 3 (2003), 280–289. "
    "DOI: 10.1037/1040-3590.15.3.280.",
    "Lynn, R. and Martin, T. 1995. National differences for "
    "thirty-seven nations in extraversion, neuroticism, psychoticism "
    "and economic, demographic and other correlates. Personality and "
    "Individual Differences 19, 3 (1995), 403–406. "
    "DOI: 10.1016/0191-8869(95)00054-A.",
    "McSweeney, B. 2002. Hofstede's model of national cultural "
    "differences and their consequences: A triumph of faith — a "
    "failure of analysis. Human Relations 55, 1 (2002), 89–118. "
    "DOI: 10.1177/0018726702551004.",
    "Minkov, M. and Hofstede, G. 2014. A replication of Hofstede's "
    "uncertainty avoidance dimension across nationally representative "
    "samples from Europe. International Journal of Cross-Cultural "
    "Management 14, 1 (2014), 7–22. DOI: 10.1177/1470595814521600.",
    # Methodological
    "Page, M. J., McKenzie, J. E., Bossuyt, P. M., Boutron, I., "
    "Hoffmann, T. C., Mulrow, C. D., Shamseer, L., Tetzlaff, J. M., "
    "Akl, E. A., Brennan, S. E., et al. 2021. The PRISMA 2020 "
    "statement: An updated guideline for reporting systematic reviews. "
    "BMJ 372, n71 (2021). DOI: 10.1136/bmj.n71.",
    # Author's own preprint
    "Tokiwa, E. 2026. Big Five personality traits and academic "
    "achievement in online learning environments: A systematic review "
    "and meta-analysis. Research Square preprint (2026). "
    "DOI: 10.21203/rs.3.rs-9513298/v1.",
]


def build():
    doc = Document(str(TEMPLATE))
    body, sect_pr = clear_body(doc)

    # Title & authors
    add_para(doc, "Title_document", TITLE)
    add_para(doc, "Authors", AUTHOR)
    add_para(doc, "Affiliation", AFFILIATION)

    # Abstract (multi-part) - concatenate with bold lead-ins inside one paragraph? ACM style uses one paragraph.
    abs_p = doc.add_paragraph(style="Abstract")
    for i, (lead, tail) in enumerate(ABSTRACT_BLOCKS):
        lead_run = abs_p.add_run(lead)
        lead_run.bold = True
        abs_p.add_run(tail + " ")

    # Keywords
    add_para(doc, "KeyWords", KEYWORDS)

    # Body
    for entry in BODY:
        style, text = entry
        if style == "_FIGURE":
            add_figure(
                doc,
                PRISMA_FIG,
                "Figure 1: PRISMA 2020 flow diagram (ICEEL 2026 submission). "
                "Identification → Screening → Eligibility → Included "
                "counts. The ICEEL terminal box marks the Asian primary-pool "
                "subset (k = 2 with extractable r per trait: A-28 Yu, China; "
                "A-31 Rivers, Japan; plus A-25 Tokiwa Japan in narrative "
                "synthesis). Adapted from Page et al. (2021), BMJ 372, n71.",
                width_cm=12.0,
            )
        elif style in TABLE_MAP:
            hdr, rows, caption = TABLE_MAP[style]
            add_table(doc, hdr, rows, caption=caption)
        else:
            add_para(doc, style, text)

    # Acknowledgments
    add_para(doc, "AckHead", "ACKNOWLEDGMENTS")
    add_para(doc, "AckPara", ACKNOWLEDGMENTS_TEXT)

    # References
    add_para(doc, "ReferenceHead", "REFERENCES")
    for ref in REFERENCES:
        add_para(doc, "Bib_entry", ref)

    doc.save(str(OUTPUT))
    print(f"Wrote {OUTPUT}")


if __name__ == "__main__":
    build()
