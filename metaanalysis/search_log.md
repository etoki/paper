# Systematic Search Log

**実施期間**: 2026-04-23 開始（OSF 事前登録 https://osf.io/e5w47/ 後）
**実施者**: Eisuke Tokiwa
**OSF project**: https://osf.io/79m5j/

## 検索戦略（3-concept Boolean, OSF §10 で事前登録）

### Concept 1 — Personality
`"Big Five" OR "Five-Factor Model" OR "FFM" OR "HEXACO" OR "BFI" OR "NEO-PI-R" OR "NEO-FFI" OR "IPIP" OR "conscientiousness" OR "openness to experience" OR "extraversion" OR "agreeableness" OR "neuroticism" OR "emotional stability" OR "personality traits"`

### Concept 2 — Online learning
`"online learning" OR "e-learning" OR "distance learning" OR "remote learning" OR "virtual learning" OR "blended learning" OR "hybrid learning" OR "MOOC" OR "massive open online course" OR "web-based learning" OR "computer-mediated learning" OR "learning management system" OR "LMS" OR "online course" OR "synchronous online" OR "asynchronous online"`

### Concept 3 — Academic outcome
`"academic performance" OR "academic achievement" OR "GPA" OR "grade point average" OR "test score" OR "course grade" OR "learning outcome" OR "learning performance" OR "academic success"`

Combined with AND. Limits: English language, 制限なし（publication year）.

---

## 実施データベース

| # | Database | API / Access | Status | Execution date | Hits (raw) |
|---|----------|--------------|--------|----------------|-----------|
| 1 | PubMed / MEDLINE | NCBI E-utilities (**blocked by env**) | 不可 | 2026-04-23 | — |
| 2 | OpenAlex | OpenAlex API (**blocked by env**) | 不可 | 2026-04-23 | — |
| 3 | ERIC | ERIC web (via WebSearch) | **完了** | 2026-04-23 | 含有 |
| 4 | Semantic Scholar | Semantic Scholar API (**blocked by env**) | 不可 | 2026-04-23 | — |
| 5 | **WebSearch (Google 経由)** | **Claude Code WebSearch tool** | **完了** | 2026-04-23 | **~80 unique records** |
| 6 | ProQuest Dissertations | 要 institutional access | 不可 | — | — |

### Deviation from pre-registration

事前登録 §10 では NCBI PubMed、OpenAlex、ERIC、Scopus、Web of Science、ProQuest の直接検索を宣言したが、実行環境のネットワークホワイトリスト制約により、これら直接 API へのアクセスが不可能であった（HTTP 403 / connection blocked）。代替として、WebSearch tool（Google 検索ベース、PubMed/PMC/Frontiers/Springer/Elsevier 等の indexed content を返す）で 3-concept 戦略を 8 クエリに分割実行した。

**Implication for Methods section**:
- この deviation は reviewer 透明性のため Methods セクションで明示
- WebSearch は Google Scholar と類似の coverage を持つが、systematic DB と比べて hit ordering の制御が限定的
- 将来的に institutional access が得られた段階で、事前登録 DB での replication search を実施し、追加候補の有無を検証する予定

---

## WebSearch queries 実施ログ（2026-04-23）

| Query | Raw hits | Novel candidates |
|-------|----------|-----------------|
| Q1: `"Big Five" "online learning" academic achievement GPA 2024` | 10 | 5 |
| Q2: `"online learning" "Big Five" personality correlation 2023-2024 university` | 10 | 3 |
| Q3: `"MOOC" personality Big Five completion dropout` | 10 | 4 |
| Q4: `"distance learning" personality conscientiousness 2022-2023` | 10 | 7 |
| Q5: `personality online course dissertation 2020-2022` | 10 | 3 |
| Q6: `HEXACO online learning academic outcomes` | 10 | 2 |
| Q7: `"K-12" OR "high school" online learning Big Five grades` | 10 | 2 |
| Q8: `Europe Germany Netherlands online COVID personality` | 10 | 2 |
| **Total** | **80** | **28 unique novel candidates** |

全候補リスト: `search_results/candidate_studies.md`

---

## PRISMA 2020 flow counts（暫定）

```
Identification
├─ Records from databases (WebSearch × 8): 80
├─ Records from previous informal search: 28 (既存 primary studies)
├─ Records from benchmark meta-analyses: 5
├─ Duplicates removed: ~40
└─ Records after deduplication: ~68

Screening
├─ Records screened (title/abstract): ~68
├─ Records excluded: ~25 (not online, wrong population, commentary)
└─ Reports sought for retrieval: ~43

Retrieval
├─ Already have PDF: 30 (local prior_research/)
├─ New PDFs to acquire: ~26
└─ Unavailable (paywall): TBD

Eligibility (full-text)
├─ Reports assessed: ~43
├─ Excluded (with reasons): TBD
│    ├─ Not online learning modality: 3-5 (A-09, A-10, A-16 既 EXCLUDE)
│    ├─ No extractable effect size: 2-3 (A-21, A-24 既 flagged)
│    └─ Duplicate/overlap samples: 1-2 (A-04/A-05 overlap)
└─ **Studies included in review: 推定 30-40 final**

Quantitative synthesis
├─ Primary achievement pool: 推定 k = 12-15
├─ Engagement/satisfaction pool: 推定 k = 20-25
└─ Total unique studies: ~30-40
```

### PRISMA 達成度評価

- Primary achievement pool: **現状 4-6 → 予想 12-15**（閾値 k≥10 をクリア予定）
- Moderator analysis: k ≥ 10 per level が可能に（era, region, modality）
- Publication bias テスト: k ≥ 10 で Egger test 実施可能

---

## 次工程チェックリスト

- [ ] Tier 1 候補 10 本の PDF 取得（最優先）
- [ ] Meta M-NEW-01, M-NEW-02 PDF 取得 → 重複研究洗い出し
- [ ] Snowballing: Hunter et al. 2025 の参照リスト + M-NEW-02 の included studies
- [ ] Rayyan に import → dedup → screening
- [ ] 新規研究の deep reading notes 追加
- [ ] data_extraction.csv 拡張
- [ ] Final PRISMA flow diagram 作成
