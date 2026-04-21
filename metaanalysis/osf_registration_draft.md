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

（以降 §8 以降は次コミットで追記）
