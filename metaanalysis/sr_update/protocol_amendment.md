# OSF 事前登録の修正（Amendment 1）草案

## 日本語の要約（登録するのは下の英語部分だけ）

- **何を変えるか**: 検索のやり方だけ。対象研究の条件、効果量、分析方法は変えない。
- **なぜ**: 2026年4月の検索は、作業環境から登録したデータベースに接続できず、ウェブ検索8回（約80件）で代用した。題名・要旨段階の件数も記録が残っていない。このため v4 は「網羅的でない検索のスコーピングレビュー」として書いている。
- **新しくやること**:
  1. PubMed、ERIC、OpenAlex を API で検索する。OpenAlex は、アクセスできない Scopus と Web of Science の代わり。
  2. 組み入れた25本と、先行メタ分析6本の引用を前後にたどる（スノーボーリング）。
  3. 登録した検索語はそのまま使い、つづりの違いや複数形などを足す。足した語の分の件数も別に報告する。
- **選別**:
  - 登録どおり、常盤が1人で選別し、10%を7日以上あけて選別し直して κ を出す。
  - 追加で、Claude を2人目の独立した選別者として使う。最終判断は常盤が行う。
- **A-37 の扱いを一般ルールにする**: 対面の期間とオンラインの期間を混ぜたデータで、オンラインだけの統計がない研究は、地図には入れるが統合には入れない。
- **正直に書くこと**: この修正は、最初の分析結果を見た後に登録する。
- **結果を見て枠組みを決めない**: 「システマティックレビュー」と名乗るかどうかは、検索がどこまでできたかだけで決める。

**登録前に確認してほしいこと**:
- [ ] A3 の「Claude を2人目の選別者にする」を入れるか。入れない場合は、その段落を消せば登録どおりの1人選別になる。
- [ ] A1 の Google Scholar を手作業でやるか。やらない場合は「実施しない」に書き換える。
- [ ] 実際に OSF に登録されている §10 の検索語が、`osf_registration_draft.md` と同じか。Concept 1 が "openness" か "openness to experience" か、だけ見ればよい。

---

## Amendment 1: Replication search in the registered information sources

**Registration amended:** Big Five personality traits and academic achievement in online learning environments (OSF Registries, https://osf.io/e5w47/; DOI 10.17605/OSF.IO/E5W47; registered 23 April 2026)

**Date of amendment:** [date of registration]

### Why this amendment is needed

The search run in April 2026 did not follow §9–§10 of the protocol. The computing environment could not reach the registered databases (PubMed, PsycINFO, ERIC, Web of Science, Scopus, ProQuest). The search was therefore replaced by:
- eight web search engine queries (about 80 records);
- previously known primary studies;
- the reference lists of benchmark meta-analyses.

The number of records screened and excluded before the full-text stage was not logged. The report based on that search (manuscript v4) is framed as a scoping review whose search is not comprehensive.

This amendment specifies a replication search in the registered sources, run before any updated synthesis.

### Timing and what was already known

This amendment is registered after the first synthesis was completed and its results were known. Known results include the pooled correlations for the nine studies in the quantitative pool, for example Conscientiousness r = .185, k = 9.

The amendment does not change the eligibility criteria (§8), the outcomes, the effect-size metric, or the analysis model. The analysis model is the one reported in manuscript v4, including its listed deviations from §14–§16. The amendment only specifies how the search and selection are run and how new studies are handled.

### A1. Information sources

The following sources will be searched through their public APIs, using a script that saves every response unmodified. All sources are searched on the same date, and that date is recorded.

| Source | Status relative to §9 |
|---|---|
| PubMed/MEDLINE (NCBI E-utilities) | registered; to be run |
| ERIC (IES ERIC API) | registered; to be run |
| OpenAlex | added, as an open substitute for Web of Science and Scopus, to which the author has no access |
| PsycINFO, Web of Science, Scopus, ProQuest Dissertations & Theses | registered; not run (no institutional access); reported as not searched |
| Google Scholar | registered; run by hand on the first 200 records of one simplified query, because Google Scholar limits query length and has no API |
| Citation searching | registered (§9); run through OpenAlex (see below) |

Citation searching works as follows:
- **Backward:** the full reference list of each seed is retrieved.
- **Forward:** every work citing each seed is retrieved.
- **Seeds:**
  - the 25 studies included in manuscript v4;
  - the four benchmark meta-analyses named in §9 (Poropat, 2009; Vedel, 2014; Stajkovic et al., 2018; Mammadov, 2022);
  - two later reviews (Meyer et al., 2023; Hunter et al., 2025).
- Seeds without a DOI are searched by hand.
- Forward citations of the six benchmark reviews are restricted to works whose title or abstract matches the online-learning block. Those reviews are cited thousands of times, mostly by studies of face-to-face instruction.

### A2. Search terms

The three registered concept blocks (§10) are used unchanged. Terms are added to each block to catch spelling variants, plural forms and close synonyms; no registered term is removed.

- **Personality:** "five factor", "extroversion", "personality".
- **Online learning:** "elearning", "online education", "distance education", "online instruction", "MOOCs", "massive open online courses", "online courses", "emergency remote teaching", "emergency remote learning", "online class", "online classes".
- **Outcome:** "test scores", "course grades", "learning outcomes", "grades", "exam performance", "examination performance", "final grade", "course performance", "course completion", "academic outcomes".

For each source, the number of hits is reported twice: for the registered terms alone and for the amended terms.

Limits:
- English language;
- no limit on publication year.

The exact strings run in each source will be archived with the search results.

### A3. Selection process

Selection follows §11b:
- **Reviewer:** ET screens every record.
- **Title/abstract stage:** a 10% random subsample is re-screened after at least seven days, and Cohen's κ is reported.
- **Full-text stage:** the same is done for 20% of the full texts.

In addition, a large language model (Claude, Anthropic) screens every title and abstract independently, as a second screener. It is given the eligibility criteria of §8 and does not see ET's decisions:
- Every disagreement is resolved by ET.
- The model never makes a final decision.
- Agreement between ET and the model is reported.
- The prompts and the model's decisions are archived.

The 37 reports assessed at full text for manuscript v4 keep their recorded decisions. Records new to this search go through the full process.

### A4. Data management (deviation from §11a)

§11a named Zotero and Rayyan. They will not be used. Instead:
- Records are de-duplicated by script: first by DOI, then by normalised title plus publication year.
- Records are screened in a spreadsheet that holds one row per unique record, with the decision and reason at each stage.
- All files and scripts are published in the project's GitHub repository (https://github.com/etoki/paper, directory `metaanalysis/sr_update/`).

### A5. Studies with mixed delivery

Some studies pool students taught face to face with students taught online, for example cohorts before, during and after the COVID-19 pandemic, without reporting statistics for the online students alone. Such studies are included in the evidence map but left out of the quantitative pool.

This rule was applied post hoc to one study in manuscript v4 (Zheng & Zheng, 2023). It is applied prospectively to every study found by this search.

### A6. Framing of the report

The report is framed according to search completeness, not according to the results:
- If the searches in A1 are run and selection follows A3, the report will be a systematic review with meta-analysis (PRISMA 2020).
- Otherwise, it stays a scoping review (PRISMA-ScR).

Either way, the sources not searched and the deviations listed here will be reported.

### A7. Timing

The search will be run within one month of this amendment. As §9 requires, it will be re-run before the final analysis if more than six months pass.
