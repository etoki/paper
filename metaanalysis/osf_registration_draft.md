# OSF Registries Pre-Registration Draft

本ドキュメントは OSF Registries (https://osf.io/registries) への systematic review / meta-analysis 事前登録用の下書き。PRISMA-P 2015 (Moher et al., 2015) の 17 項目構造に準拠。

**背景**: PROSPERO は health-related outcome を必須とし、教育心理領域の systematic review は原則として scope 外のため、OSF Registries を pre-registration venue として採用する（2026-04 時点の PROSPERO scope policy に基づく判断）。

---

## 登録前チェックリスト

- [ ] OSF アカウント作成（https://osf.io/）
- [ ] OSF Project 作成（名称: `Big Five Online Learning Meta-Analysis`）
- [ ] 既存 OSF Registries で同テーマの先行登録を検索
  - 検索語例: `"Big Five" "online learning" meta-analysis`
  - 検索語例: `personality "e-learning" systematic review`
- [ ] 既存 PROSPERO 登録も併せて確認（重複回避）
- [ ] テンプレート選択: **OSF Preregistration** (汎用) で PRISMA-P 準拠本文を登録
  - 代替案: **Open-Ended Registration** で本 markdown と付属ファイルを PDF 化してアップロード
- [ ] Supplementary files を OSF Project に事前アップロード
  - `literature_review.md` / `meta_analysis_plan.md` / `data_extraction.csv` / `data_extraction_README.md`

---

## OSF Registries テンプレート選定メモ

| テンプレート | 適合性 | 備考 |
|---|---|---|
| **OSF Preregistration** | ◎ 本命 | 汎用フォーム。PRISMA-P 17 項目を自由記述欄にマッピング可。Time-stamped, immutable |
| **Open-Ended Registration** | ○ 代替 | 完全自由記述。本ドキュメントを PDF にして添付するのみで登録可 |
| **Registered Report Protocol** | × | 雑誌連動 Registered Report 専用。今回は該当しない |
| **PRISMA for Abstracts** | × | 完成論文のアブストラクト用。プロトコル用ではない |

**推奨**: OSF Preregistration で本文を登録し、追加資料は OSF Project 本体にアップロード。

---

# PRISMA-P 2015 準拠 本文

## 1a. Title — Identification

```
Big Five personality traits and academic achievement in online learning
environments: A systematic review and meta-analysis
```

## 1b. Title — Update

該当なし（初回登録）。

## 2. Registration

```
Registration platform: OSF Registries (https://osf.io/registries)
Registration type: OSF Preregistration (PRISMA-P compliant)
Registration date: （登録時に自動付与）
Registration DOI: （登録後に更新）

Note: PROSPERO was initially considered but excluded because the review
does not address a health-related outcome (scope restriction of PROSPERO).
```

## 3. Authors

### 3a. Contact

```
Name: Eisuke Tokiwa
Affiliation: SUNBLAZE Inc., Tokyo, Japan
Email: eisuke.tokiwa@sunblaze.jp
ORCID: （登録時に記入）
```

### 3b. Contributions

```
Eisuke Tokiwa (ET):
- Conceptualization, protocol development
- Search strategy design and execution
- Title/abstract and full-text screening
- Data extraction and risk of bias assessment
- Statistical analysis and interpretation
- Drafting and revising the manuscript

（共同研究者が加わる場合は CRediT taxonomy に従って記載）
```

## 4. Amendments

```
該当なし（初回登録時点）。

登録後に方針変更が発生した場合は、OSF 上で新たな version を登録し、
変更点・変更理由・変更日を明記する。論文本体の Methods 章にも逸脱を
透明に記載する。
```

## 5. Support

### 5a. Sources

```
No external funding. Self-funded research.
```

### 5b. Sponsor

```
None.
```

### 5c. Role of sponsor or funder

```
Not applicable.
```

## 6. Introduction — Rationale

```
Existing meta-analyses of Big Five personality and academic performance
(Poropat, 2009; Vedel, 2014; Stajkovic et al., 2018; Mammadov, 2022)
consistently identify Conscientiousness as the strongest predictor
(ρ ≈ .19–.22 overall; up to .35 in Asian samples). However, all prior
syntheses aggregate face-to-face and online learning contexts, treating
learning modality as noise rather than as a substantive moderator.

Online learning environments differ substantively from face-to-face
instruction in (a) self-regulation demands, (b) social presence,
(c) temporal flexibility, and (d) technology mediation. These differences
plausibly alter which personality traits matter most. For example,
Openness (curiosity, self-directed learning) may become more salient
when learners must navigate asynchronous materials autonomously, while
Extraversion's effect may reverse if in-class participation rewards no
longer apply.

The COVID-19 pandemic (2020–2023) triggered a sharp increase in primary
studies on personality and online learning outcomes, creating — for the
first time — a corpus sufficient for quantitative synthesis (estimated
k ≈ 25–30 studies for achievement outcomes).

No prior quantitative meta-analysis has focused specifically on online
learning contexts. This review fills that gap.
```

## 7. Objectives

### Review questions

```
RQ1: In online learning environments, what is the pooled magnitude and
     direction of the association between each Big Five trait and
     academic achievement?

RQ2: How do these associations compare with existing meta-analyses of
     general (mixed-modality) academic performance?

RQ3: How do the associations vary by (a) learning modality
     (fully online / blended / MOOC / synchronous / asynchronous),
     (b) education level (K-12 / undergraduate / graduate / adult),
     (c) region (Asia / Europe / North America / Other), and
     (d) era (pre-COVID ≤ 2019 / COVID 2020–2022 / post-COVID 2023–)?
```

### Hypotheses

```
H1: Conscientiousness shows the strongest positive pooled effect
    (ρ = .20–.35), consistent with prior meta-analyses.
H2: Openness shows the second-strongest positive effect — potentially
    stronger than in face-to-face contexts, reflecting self-directed
    learning demands.
H3: Agreeableness shows a small positive effect, weaker than in
    face-to-face contexts.
H4: Neuroticism shows a negative effect, more pronounced in fully
    online than blended modalities.
H5: Extraversion shows a null or weak positive effect, with possible
    facet-level cancellation (Sociability − / Assertiveness +).

H2, H4, and H5 constitute the novel contribution of this review, as
they predict modality-specific divergence from established patterns.
A null result for H2/H4/H5 would itself be informative (i.e., online
and face-to-face personality–achievement patterns are equivalent).
```

---

## 8. Eligibility criteria

### 8a. Participants / population

```
Included: Students of any educational level (K-12, undergraduate, graduate,
adult learners) enrolled in online learning environments. No geographic,
linguistic, or demographic restriction on participants.

Excluded:
- Fully face-to-face samples with no online learning component
- Non-human learners (e.g., AI agents, simulated students)
- Samples with N < 10 (insufficient statistical information)
```

### 8b. Intervention / exposure

```
Big Five personality traits (Conscientiousness, Openness to Experience,
Extraversion, Agreeableness, Neuroticism) or HEXACO personality traits
(adding Honesty-Humility; H-Emotionality mapped to Big Five Neuroticism
per Ashton & Lee, 2007) measured by validated inventories:

- BFI / BFI-2 / BFI-44
- NEO-PI-R / NEO-FFI / NEO-FFI-3
- IPIP (any Big Five-scored variant)
- HEXACO-PI-R / HEXACO-60
- Other peer-reviewed Big Five-aligned scales with published reliability

Excluded:
- MBTI (non-Big Five typology)
- Ad-hoc / unvalidated scales without published psychometric properties
- Single-trait measures (e.g., Grit-only) not mappable to Big Five
```

### 8c. Comparator

```
Not applicable. This is a meta-analysis of observational correlational
studies. No explicit comparator group is required. Moderator analyses
(§15) effectively compare effect sizes across modalities, eras, etc.
```

### 8d. Outcomes

```
Primary: Academic achievement — GPA, course grade, exam/test score,
         learning performance, or equivalent quantitative academic outcome.

Secondary (if sufficient studies): academic satisfaction, academic
         engagement, learning-related behaviors (LMS activity, completion
         rates, dropout).
```

### 8e. Study design

```
Included:
- Cross-sectional correlational studies
- Longitudinal / prospective studies reporting cross-sectional statistics
- Dissertations with peer-review-equivalent quality

Excluded:
- Qualitative-only studies
- Commentary, editorial, narrative review (but reference lists will be
  hand-searched)
- Single case studies (N < 10)
- Studies without extractable effect size statistics (after author
  contact)
```

### 8f. Report characteristics

```
Language: English only (due to single-reviewer resource constraints;
          limitation acknowledged in Discussion).
Date: No restriction. Year of publication included as moderator.
Publication status: Peer-reviewed journal articles, conference papers,
          and dissertations included. Unpublished manuscripts and
          preprints excluded to maintain quality threshold.
```

## 9. Information sources

```
Electronic databases to be searched:
- PubMed / MEDLINE
- PsycINFO (APA PsycNET)
- ERIC (Education Resources Information Center)
- Web of Science (Core Collection)
- Scopus
- ProQuest Dissertations & Theses Global

Supplementary sources:
- Google Scholar (first 200 hits per query, as recommended by Haddaway
  et al., 2015)
- Forward and backward reference snowballing from all included studies
  and from prior meta-analyses (Poropat 2009, Vedel 2014, Stajkovic 2018,
  Mammadov 2022)
- Author contact for unpublished statistics or clarifications

Grey literature:
- ProQuest Dissertations (covered above)
- No additional grey literature search (limitation acknowledged)

Search date: Initial search planned 2026-MM-DD. Search will be updated
             prior to final analysis if > 6 months elapse.
```

## 10. Search strategy

```
Three concept blocks combined with AND. Within each block, terms
combined with OR. Draft strategy (for PubMed; adapted per database):

Concept 1 — Personality (Big Five / HEXACO):
  "Big Five"[tiab] OR "Five-Factor Model"[tiab] OR "FFM"[tiab]
  OR "HEXACO"[tiab] OR "BFI"[tiab] OR "NEO-PI-R"[tiab] OR "NEO-FFI"[tiab]
  OR "IPIP"[tiab] OR "conscientiousness"[tiab] OR "openness"[tiab]
  OR "extraversion"[tiab] OR "agreeableness"[tiab] OR "neuroticism"[tiab]
  OR "emotional stability"[tiab] OR "personality traits"[tiab]

AND

Concept 2 — Online learning:
  "online learning"[tiab] OR "e-learning"[tiab] OR "distance learning"[tiab]
  OR "remote learning"[tiab] OR "virtual learning"[tiab]
  OR "blended learning"[tiab] OR "hybrid learning"[tiab] OR "MOOC"[tiab]
  OR "massive open online course"[tiab] OR "web-based learning"[tiab]
  OR "computer-mediated learning"[tiab] OR "learning management system"[tiab]
  OR "LMS"[tiab] OR "online course"[tiab] OR "synchronous online"[tiab]
  OR "asynchronous online"[tiab]

AND

Concept 3 — Academic outcomes:
  "academic performance"[tiab] OR "academic achievement"[tiab] OR "GPA"[tiab]
  OR "grade point average"[tiab] OR "test score"[tiab] OR "course grade"[tiab]
  OR "learning outcome"[tiab] OR "learning performance"[tiab]
  OR "academic success"[tiab]

Limits: English language; publication date unrestricted.

The full executed strategy (database-specific syntax, filters, hit counts,
dates) will be deposited on OSF as a supplementary file upon completion.
```

## 11. Study records

### 11a. Data management

```
- Search results exported as RIS/BibTeX from each database
- Imported into Zotero (master library) for de-duplication
- Screening conducted in Rayyan (https://rayyan.ai/) for blinded
  title/abstract screening workflow
- Included full-text PDFs stored under metaanalysis/prior_research/
  (already collected 27/27 primary studies as of protocol registration)
- Data extraction in structured CSV (metaanalysis/data_extraction.csv)
- All files version-controlled on Git + archived on OSF
```

### 11b. Selection process

```
Single-reviewer workflow (resource constraint; mitigated by procedures
below):

Stage 1 — Title/abstract screening:
  One reviewer (ET) screens all records against the eligibility criteria
  (§8). A 10% random subsample is re-screened after a ≥ 7-day interval
  to compute intra-rater reliability (Cohen's κ); target κ ≥ 0.80.
  If κ < 0.80, criteria are refined and full set is re-screened.

Stage 2 — Full-text assessment:
  Retrieved full texts are assessed against detailed eligibility criteria.
  Exclusion reasons are recorded per PRISMA 2020.
  A 20% random subsample is re-assessed after a ≥ 7-day interval
  (κ target ≥ 0.80).

Disagreements (intra-rater): resolved by re-reading the full text and
  documenting the final decision with rationale.

Limitation: absence of a second independent reviewer is an acknowledged
  limitation; sensitivity analyses will be reported.
```

### 11c. Data collection process

```
One reviewer (ET) extracts data using the pre-specified extraction form
(metaanalysis/data_extraction.csv, see data_extraction_README.md).
A 10% random subsample is re-extracted after a ≥ 7-day interval to
compute extraction reliability; target κ ≥ 0.80 for categorical fields,
ICC(2,1) ≥ 0.90 for continuous fields.

Authors will be contacted by email (up to two attempts, ≥ 2 weeks apart)
for missing statistics, unclear sample characteristics, or data on
online-only subsamples. Non-response after two attempts will be recorded
as "author unreachable."
```

## 12. Data items

```
Extracted fields (full specification in data_extraction_README.md):

Study identification:
  - Author(s), year, DOI, source (journal/proceedings/dissertation)
  - Country of first author, country of data collection

Sample characteristics:
  - Total N, analytic N
  - Age mean/SD/range
  - Gender composition (% female)
  - Education level (K-12 / undergraduate / graduate / adult / mixed)
  - Discipline / subject domain
  - Recruitment method

Learning context:
  - Modality (fully online / blended / MOOC / synchronous /
    asynchronous / mixed)
  - Platform (LMS name, if reported)
  - Course duration (weeks)
  - Era (pre-COVID ≤ 2019 / COVID 2020–2022 / post-COVID 2023–)

Personality measure:
  - Instrument name (BFI, NEO-FFI, IPIP, HEXACO, etc.)
  - Item count
  - Reliability (α or ω) per trait
  - HEXACO → Big Five mapping flag

Outcome measure:
  - Instrument (GPA, course grade, exam score, LMS behavior, etc.)
  - Timing (concurrent / prospective)
  - Reliability (if applicable)

Effect sizes:
  - Pearson r per trait × outcome (primary)
  - β, partial r, d, η² (secondary — convertible to r)
  - 95% CI or SE
  - p-value
  - Sample size used for each effect size

Risk of bias (§14):
  - 8-domain JBI ratings
  - Aggregate score (0–8)

Notes:
  - Any deviation from protocol, author contact outcomes, coding
    ambiguities
```

---

（以降 §13 以降は次コミットで追記）
