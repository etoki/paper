# ICEEL 2026 — Paper EL2036: list of changes in the corrected version

Paper: "Cultural dimensions in online learning: A Hofstede-moderated within-Asia synthesis of the Big Five and academic achievement, with a focused look at the Japanese context" (Eisuke Tokiwa)

The main findings, structure, figures and references are unchanged, and the corrected paper is 9 pages long. All numbers can be reproduced with the script iceel/scripts/run_hofstede_meta.py in the project repository (https://github.com/etoki/paper, folder metaanalysis/conference_submissions).

| # | Location | Before | After | Reason |
|---|---|---|---|---|
| 1 | Abstract (Background); Section 1, first paragraph | "a highly significant Extraversion × Region moderator … (Q_between = 46.43, p < .001; …)" | "an Extraversion × Region moderator … (Asian pooled r = −0.131, k = 2, vs non-Asian r = +0.050, k = 7; Knapp–Hartung meta-regression F(1, 7) = 29.52, p < .001)". Section 1 adds that the preprint's Q_between applied a large-sample χ² test to small-sample HKSJ standard errors and overstates the evidence, and that the contrast rests on two Asian studies. | The cited Q_between value was computed with an inappropriate procedure. The conclusion (p < .001) is unchanged. |
| 2 | Abstract (Methods) | "Random-effects pools (REML + HKSJ) replicate the parent preprint." | "Random-effects pools use REML + HKSJ." | See 3. |
| 3 | Table 1; Section 4.1 | O: 0.164 [−0.989, 0.994]; A: 0.330 [−0.981, 0.995]; "These numbers replicate the preprint Region: Asia row exactly." | O: 0.158 [−0.989, 0.994]; A: 0.324 [−0.981, 0.995]; C, E and N unchanged; the text now states that O and A differ from the preprint by 0.006. | τ² is now estimated by exact REML; the previous routine under-estimated τ². |
| 4 | Section 3.3 | — | Adds that τ² is estimated by exact REML. | See 3. |
| 5 | Section 4.2 (note under Table 2) | "LTO is exception-near-zero because the two countries' LTO scores happen to be very close." | LTO slopes are the largest in magnitude only because the two LTO scores differ by one point (88 vs 87), and they carry no information. | The previous sentence contradicted Table 2. |
| 6 | Section 4.2 (paragraph after Table 2) | "higher Individualism … is associated with more positive Extraversion – achievement correlations" | The Extraversion slope on Individualism is negative (−0.0018): Japan, the more individualistic country, shows the more negative correlation (−0.173 vs −0.126), which runs against the expectation. | The previous sentence contradicted Table 2. |
| 7 | Section 5.1 | — | Adds: both Asian studies are asynchronous and use objective outcomes, so a region effect cannot be separated from a modality or outcome-type effect. | Additional confound. |
| 8 | Section 5.2 | Heading "Hofstede slopes as direction-finders"; text read the IDV and LTO slopes as supporting theory. | Heading "What the Hofstede slopes can and cannot show"; text explains that the slopes re-express one confounded Japan – China contrast, the IDV slope runs against expectation, and the LTO slopes cannot be interpreted. | Consistency with Table 2. |
| 9 | Section 5.4 | "predicted − by HM04 UAI .45" | "predicted − by HM04 UAI −.45" | Sign typo (Table 4 shows −.45). |
| 10 | Section 6 | — | Adds: Hofstede & McCrae (2004) correlations describe country-level trait means, so using them to predict within-country trait – achievement correlations is an assumption. | Additional limitation. |
| 11 | Abstract (Conclusion) | "materially reshapes" | "may reshape" | Wording matched to the descriptive (k = 2) evidence. |
| 12 | All section headings | Numbers appeared twice (e.g., "1 1 INTRODUCTION", "4.2 4.2 …") | Numbered once by the template's heading styles | Formatting fix. |
