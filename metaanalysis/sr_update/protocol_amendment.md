# OSF 事前登録の修正（Amendment 1）

## 日本語の要約（登録するのは下の英語部分だけ）

- **何を変えるか**: 検索のやり方と、足りない統計値の扱いだけ。対象研究の条件、効果量、分析方法は変えない。
- **なぜ**: 2026年4月の検索は、作業環境から登録したデータベースに接続できず、ウェブ検索8回（約80件）で代用した。題名・要旨段階の件数も記録が残っていない。このため v4 は「網羅的でない検索のスコーピングレビュー」として書いている。
- **新しくやること**:
  1. PubMed、ERIC、OpenAlex を API で検索する。OpenAlex は、アクセスできない Scopus と Web of Science の代わり。
  2. Google Scholar は Publish or Perish で、簡略化した検索式1本の上位200件を取る（2026-10-27 決定）。
  3. 組み入れた25本と、先行メタ分析6本の引用を前後にたどる（スノーボーリング）。新しく組み入れた研究からも、登録どおり「2回続けて新しい研究が見つからなくなるまで」たどる。
  4. 登録した検索語はそのまま使い、つづりの違いや複数形などを足す。足した語の分の件数も別に報告する。
- **選別**: 登録どおり、常盤が1人で選別し、10%を7日以上あけて選別し直して κ を出す（Claude を2人目の選別者にはしない。2026-10-27 決定）。
- **著者への問い合わせはしない**（2026-10-27 決定）。v4 の37本でも問い合わせておらず、すべての研究を同じ手順で扱うため。足りない統計値は「換算 → 除外」の順で扱う。
- **A-37 の扱いを一般ルールにする**: 対面の期間とオンラインの期間を混ぜたデータで、オンラインだけの統計がない研究は、地図には入れるが統合には入れない。
- **正直に書くこと**: この修正は、最初の分析結果を見た後に登録する。
- **結果を見て枠組みを決めない**: 「システマティックレビュー」と名乗るかどうかは、検索がどこまでできたかだけで決める。

**登録前の確認（済み）**:
- [x] Google Scholar は Publish or Perish で行う（2026-10-27）。
- [x] OSF に登録されている検索語は `search_strategy.py` の `PREREGISTERED` と同じ。Concept 1 は "openness"（2026-10-27 に OSF の API で照合）。

**OSF での入れ方**: 登録ページの Update から新しい版を作り、次の3つの欄の先頭に、それぞれ `osf_amendment1_fields.md` の文面を足す。元の文面は消さない。修正理由（justification）の欄には、同じファイルの justification の文面を入れる。

| OSF の欄 | 足す内容 |
|---|---|
| Sampling → Data collection procedures | A1〜A6（情報源、検索語、選別、全文の入手、データ管理、混合データの研究、報告の枠組み） |
| Sampling → Starting and stopping rules | 検索の開始と終了の条件 |
| Analysis Plan → Missing data | 著者に問い合わせないこと |

---

## Amendment 1: Replication search in the registered information sources

**Registration amended:** Big Five personality traits and academic achievement in online learning environments (OSF Registries, https://osf.io/e5w47/; DOI 10.17605/OSF.IO/E5W47; registered 23 April 2026)

**Date of amendment:** 27 October 2026

### Why this amendment is needed

The search run in April 2026 did not follow the "Data collection procedures" section of the registration. The computing environment could not reach the registered databases (PubMed, PsycINFO, ERIC, Web of Science, Scopus, ProQuest). The search was therefore replaced by:
- eight web search engine queries (about 80 records);
- previously known primary studies;
- the reference lists of benchmark meta-analyses.

The number of records screened and excluded before the full-text stage was not logged. The report based on that search (manuscript v4) is framed as a scoping review whose search is not comprehensive.

This amendment specifies a replication search in the registered sources, run before any updated synthesis.

### Timing and what was already known

This amendment is registered after the first synthesis was completed and its results were known. Known results include the pooled correlations for the nine studies in the quantitative pool, for example Conscientiousness r = .185, k = 9.

The amendment does not change the eligibility criteria, the outcomes, the effect-size metric, or the analysis model. The analysis model is the one reported in manuscript v4, including its listed deviations from the "Statistical models", "Inference criteria" and "Other planned analysis" sections. The amendment only specifies how the search and selection are run, how missing statistics are handled, and how new studies are handled.

### A1. Information sources

The following sources will be searched through their public APIs, using a script that saves every response unmodified. All sources are searched on the same date, and that date is recorded.

| Source | Status |
|---|---|
| PubMed/MEDLINE (NCBI E-utilities) | registered; to be run |
| ERIC (IES ERIC API) | registered; to be run |
| OpenAlex | added, as an open substitute for Web of Science and Scopus, to which the author has no access |
| PsycINFO, Web of Science, Scopus, ProQuest Dissertations & Theses | registered; not run (no institutional access); reported as not searched |
| Google Scholar | registered; run once by the author with Publish or Perish (Harzing), on the first 200 records of one simplified query, because Google Scholar has no API and limits query length; the exported records are merged and de-duplicated with the other sources |
| Citation searching | registered; run through OpenAlex (see below) |

Citation searching works as follows:
- **Backward:** the full reference list of each seed is retrieved.
- **Forward:** every work citing each seed is retrieved.
- **Seeds:**
  - the 25 studies included in manuscript v4;
  - the four benchmark meta-analyses named in the registration (Poropat, 2009; Vedel, 2014; Stajkovic et al., 2018; Mammadov, 2022);
  - two later reviews (Meyer et al., 2023; Hunter et al., 2025).
- Seeds without a DOI are searched by hand.
- Forward citations of the six benchmark reviews are restricted to works whose title or abstract matches the online-learning block. Those reviews are cited thousands of times, mostly by studies of face-to-face instruction.
- As registered, citation searching is repeated from studies newly included by this search, until two consecutive rounds yield no new eligible study.

### A2. Search terms

The three registered concept blocks are used unchanged. Terms are added to each block to catch spelling variants, plural forms and close synonyms; no registered term is removed.

- **Personality:** "five factor", "extroversion", "personality".
- **Online learning:** "elearning", "online education", "distance education", "online instruction", "MOOCs", "massive open online courses", "online courses", "emergency remote teaching", "emergency remote learning", "online class", "online classes".
- **Outcome:** "test scores", "course grades", "learning outcomes", "grades", "exam performance", "examination performance", "final grade", "course performance", "course completion", "academic outcomes".

For each source, the number of hits is reported twice: for the registered terms alone and for the amended terms.

Limits:
- English language;
- no limit on publication year.

The exact strings run in each source will be archived with the search results.

### A3. Selection process

Selection follows the registered screening workflow:
- **Reviewer:** ET screens every record.
- **Title/abstract stage:** a 10% random subsample is re-screened after at least seven days, and Cohen's κ is reported.
- **Full-text stage:** the same is done for 20% of the full texts.

The 37 reports assessed at full text for manuscript v4 keep their recorded decisions. Records new to this search go through the full process.

**Full-text retrieval:** open-access copies are retrieved by script (OpenAlex open-access locations, PubMed Central, ERIC, OSTI). The author obtains the other reports through the author's own access. Reports that cannot be obtained are reported as not retrieved. Authors are not contacted (see A8).

### A4. Data management (deviation from the registered screening workflow)

The registration named Zotero and Rayyan. They will not be used. Instead:
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

The search will be run within one month of this amendment. As registered, it will be re-run before the final analysis if more than six months pass.

### A8. Missing statistics (deviation from the "Missing data" section)

Authors are not contacted for missing statistics, for studies found by this search or any other. When a statistic is missing, it is converted from a reported alternative under the registered conversion rules; if that is not possible, the study is left out of that trait's pool and the reason is logged.

Reason: none of the 37 reports assessed for manuscript v4 involved author contact. Using the same procedure for every study keeps their treatment consistent. As a consequence, the registered counts of studies excluded after failed author contact and the sensitivity analysis by author-response status do not apply.
