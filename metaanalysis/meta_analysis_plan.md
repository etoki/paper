# Meta-Analysis Plan: Big Five / HEXACO × Online Learning × Academic Achievement

本ドキュメントは meta-analysis 論文の実施計画（執筆前段階）。PRISMA 2020 と Cochrane Handbook 原則に準拠。

---

## 1. 研究背景と目的

### 1.1 背景

既存の Big Five × 学業成績メタ分析は全て **一般的学業**を対象（Poropat, 2009; Vedel, 2014; Mammadov, 2022; Stajkovic et al., 2018）。これらは Conscientiousness が最強（ρ ≈ .19–.22、アジア層では .35）と一貫して報告する。一方、**オンライン学習環境に特化した定量的 meta-analysis は存在しない**。

narrative systematic review は少数存在する（e.g., Gray & DiLoreto 推定, 2024）が、**pooled effect size や moderator 分析は未実施**である。

COVID-19 期（2020–2023）に primary study が急増し、現在は **初の quantitative synthesis を実施するのに十分な研究数**が蓄積されている（推定 k ≈ 25–30 for achievement outcome）。

### 1.2 研究目的

本メタ分析は以下を定量化する:

1. **RQ1**: オンライン学習環境において、Big Five 各特性と学業成績の関連はどの程度強いか？（各特性の pooled ρ を推定）
2. **RQ2**: この関連は一般的学業（既存メタ分析）と比較してどう異なるか？（差分検証）
3. **RQ3**: モダリティ（完全オンライン/blended/MOOC）、教育段階（K-12/大学/大学院）、文化圏（アジア/欧米/その他）、COVID 期 vs pre-COVID により効果はどう変動するか？（moderator 分析）

### 1.3 仮説

既存文献と著者の先行研究（Tokiwa, 2025）に基づく事前仮説:

- **H1**: Conscientiousness が Big Five の中で最強の正の効果を示す（pooled ρ = .20–.35）
- **H2**: Openness が二番目に強い正の効果を示す（一般的学業メタ分析での順位と異なる可能性 — オンライン環境での自己主導性と適合）
- **H3**: Agreeableness は正だが中程度の効果（一般的学業より弱い可能性）
- **H4**: Neuroticism は負の効果だが、モダリティで変動（fully online で強め、blended で弱め）
- **H5**: Extraversion は中立または弱正（facet level では Sociability 負 / Assertiveness 正の相殺）

### 1.4 仮説のオンライン学習特化性

- **H2 と H5 が本メタ分析の novel contribution**（既存の一般的学業メタ分析と異なるパターンを予測）
- Null 結果でも意味あり（「オンラインと対面で Big Five 効果が同じ」という知見）

---

## 2. プレ・レジストレーション

### 2.1 PROSPERO 登録

**実施予定**: 本計画を PROSPERO (https://www.crd.york.ac.uk/prospero/) に登録する。登録内容:

- Title, Review Question, Searches, Types of study to be included
- Participants/population, Intervention/exposure, Comparators, Outcomes
- Analysis plan (including moderators)
- Timing
- Funding, Conflicts of interest

### 2.2 事前チェック

登録前に以下を確認:
- [ ] 同一テーマの先行登録がないか PROSPERO 検索
- [ ] 関連の新規 systematic review が Open Science Framework (OSF) 等に登録されていないか

---

## 3. PRISMA 2020 フレームワーク

本メタ分析は [PRISMA 2020 statement](http://www.prisma-statement.org/) に完全準拠する。

### 3.1 主要フェーズ

```
Phase 1: Identification
  ├─ Database search (English-language only)
  └─ Other sources (reference lists, Google Scholar)
         ↓
Phase 2: Screening
  ├─ Title/abstract screening (2 reviewers ideal)
  └─ Duplicate removal
         ↓
Phase 3: Eligibility
  ├─ Full-text retrieval
  └─ Eligibility assessment against inclusion/exclusion criteria
         ↓
Phase 4: Included
  ├─ Data extraction
  ├─ Risk of bias assessment
  └─ Quantitative synthesis
```

### 3.2 PRISMA フロー図作成

最終的に PRISMA 2020 Flow Diagram を作成し、論文の Methods 章で提示する。記録する数値:

- n identified (database)
- n identified (other)
- n after duplicates removed
- n screened
- n excluded (title/abstract)
- n full-text assessed
- n excluded (reason-by-reason)
- n included in qualitative synthesis
- n included in quantitative synthesis (meta-analysis)

---

## 4. PICOS（包含基準）

| 要素 | 内容 |
|------|------|
| **P (Population)** | 学生全般（K-12, 高校, 大学, 大学院, 成人学習者）。言語・地域制限なし（ただし英語論文のみ） |
| **I (Intervention/Exposure)** | Big Five 性格特性（C, O, E, A, N）または HEXACO（6 因子、5 共通）を validated inventory で測定 |
| **C (Comparator)** | 明示的 comparator なし（相関研究の pooled effect を対象）。必要に応じてモダリティ別サブグループ分析 |
| **O (Outcome)** | **Primary**: 学業成績（GPA、テストスコア、コース成績、ランクなど定量化可能な achievement）。**Secondary**: 満足度、エンゲージメント |
| **S (Study design)** | 観察研究（cross-sectional、longitudinal）、相関研究。実験研究は除外（personality は treatment ではない） |
- **Learning context**: オンライン学習環境（完全オンライン、blended, MOOC, LMS ベースなど）

---

## 5. 包含／除外基準（詳細）

### 5.1 Inclusion criteria

1. **性格測定**: Big Five または HEXACO を validated inventory（BFI, BFI-2, NEO-PI-R, NEO-FFI, IPIP, HEXACO-PI-R など）で測定
2. **学習環境**: Online / blended / MOOC / LMS-mediated のいずれかを含む
3. **アウトカム**: 学業成績（primary）または満足度・エンゲージメント（secondary）を量的に報告
4. **効果量抽出可能**: Pearson r, Spearman ρ, β, d, 変換可能な統計量（F, t, OR 等）
5. **言語**: 英語
6. **発表形式**: Peer-reviewed journal, conference proceedings, 修士・博士学位論文（gray literature 含む）

### 5.2 Exclusion criteria

1. MBTI, Big Six, 3-factor/PEN model 等、Big Five/HEXACO 以外のフレームワーク
2. 完全対面学習のみ（オンライン要素ゼロ）
3. 質的研究のみ（効果量抽出不可）
4. 非英語論文
5. 同一サンプルを別論文で重複報告（最も包括的な 1 本を採用）
6. 学習者が人間でない（AI training data 等）
7. レビュー論文（ただし snowballing の情報源として活用）

### 5.3 グレーゾーン判定

- **Blended learning**: 50% 以上オンラインなら include。判定困難な場合 authors に問合せ
- **フルオンライン前の準備期間のみ含むデータ**: 文脈をよく読み、実質的に online exposure があれば include
- **複数アウトカム報告**: 全て extract、primary 分析では achievement を優先

---

## 6. 研究 ID 割当

各 included study に一意な ID を付与（例: `A-01`, `A-02`, ...）。同一著者グループの複数論文は `A-01a`, `A-01b` で区別。`literature_review.md` の ID 体系と統一。

---

## 7. 検索戦略

### 7.1 対象データベース

| 優先度 | データベース | 対象分野 |
|--------|-------------|---------|
| 必須 | PubMed / MEDLINE | 心理学・教育・医学 |
| 必須 | PsycINFO (APA) | 心理学専門 |
| 必須 | ERIC | 教育学専門 |
| 必須 | Web of Science (Core Collection) | 多分野 SSCI |
| 必須 | Scopus | 多分野、引用網羅 |
| 推奨 | Google Scholar | gray literature, snowballing |
| 推奨 | ProQuest Dissertations & Theses | 修士・博士論文 |
| 補助 | IEEE Xplore | 工学系 online learning |

### 7.2 検索文字列（英語）

**Concept 1: Personality framework**
```
("Big Five" OR "Five-Factor Model" OR "FFM" OR "HEXACO" OR
 "BFI" OR "NEO-PI-R" OR "NEO-FFI" OR "IPIP" OR
 "conscientiousness" OR "openness to experience" OR "extraversion" OR
 "agreeableness" OR "neuroticism" OR "emotional stability")
```

**Concept 2: Online learning**
```
("online learning" OR "e-learning" OR "distance learning" OR
 "remote learning" OR "virtual learning" OR "blended learning" OR
 "hybrid learning" OR "MOOC" OR "massive open online course" OR
 "web-based learning" OR "computer-mediated learning" OR
 "learning management system" OR "LMS" OR "online course" OR
 "synchronous online" OR "asynchronous online")
```

**Concept 3: Outcome**
```
("academic performance" OR "academic achievement" OR "GPA" OR
 "grade point average" OR "test score" OR "course grade" OR
 "learning outcome" OR "learning performance" OR
 "satisfaction" OR "engagement" OR "completion")
```

**最終検索式**: `Concept 1 AND Concept 2 AND Concept 3`

### 7.3 検索フィルタ

- 言語: English
- Document type: Journal article, conference paper, thesis/dissertation
- 期間: 制限なし（実質 2003 以降のみヒット見込み）
- 除外: review article（メタ分析対象からは除外、ただし reference list は参考）

### 7.4 Snowballing

- Primary studies の reference list から遡及（backward）
- Google Scholar で引用している論文をチェック（forward）

### 7.5 検索ログ

各 DB で実施した検索式・日付・ヒット数を記録する検索ログを `metaanalysis/search_log.md` に保存。

---

## 8. スクリーニング

### 8.1 推奨: 2 名レビュアー体制

独立した 2 名レビュアーで以下を実施:

1. **Title/abstract screening**: 明らかに対象外を除外
2. **Full-text eligibility**: 包含基準を厳密適用
3. **Disagreement resolution**: 第 3 名仲裁 or 議論

### 8.2 1 名実施時の対応

単独実施の場合、ランダムに抽出した 10% サブセットを 2 回目レビュー（時間差）で実施し **intra-rater reliability** を Cohen's κ で報告。κ ≥ 0.80 推奨。

### 8.3 スクリーニングツール

推奨ツール（ライセンス問題なく無料）:
- **Rayyan** (https://www.rayyan.ai/) — 体系的レビュー専用
- **Covidence** (https://www.covidence.org/) — Cochrane 推奨
- **Zotero** — 書誌管理＋タグ付けで代用可能

---

## 9. データ抽出テーブル

### 9.1 抽出フィールド

各 included study から以下を抽出。Excel/CSV で管理。

| カテゴリ | フィールド |
|---------|----------|
| **Study identification** | Study ID, 1st author, year, country, DOI |
| **Publication** | Journal/conference, volume, issue, pages |
| **Sample** | N (total), N(analyzed), age (mean/SD/range), gender (% female), education level, sampling method |
| **Design** | Cross-sectional / longitudinal / experimental, measurement waves |
| **Learning context** | Modality (fully online / blended / MOOC / LMS), synchronous/asynchronous, duration, subject domain, platform name |
| **Personality measurement** | Instrument (BFI/BFI-2/NEO-FFI/IPIP/HEXACO-PI-R 等), item count, Cronbach α (per trait), facet-level reported (Y/N) |
| **Outcome measurement** | Type (GPA/exam/LMS activity 等), instrument, reliability, range, unit |
| **Effect sizes** | For each Big Five × outcome pair: r (Pearson), ρ (Spearman), β (from regression), N for that pair, p-value, 95% CI |
| **Moderator variables** | Country, year collected, COVID era (pre/during/post), education level, platform type |
| **Risk of bias** | (Section 10 参照) |

### 9.2 効果量の変換ルール

報告形式が異なる場合、以下の変換を適用:

| 報告形式 | 変換先 (Pearson r) |
|---------|-------------------|
| Spearman ρ | 近似: r ≈ ρ（軽微な調整: r = 2·sin(π·ρ/6)） |
| Cohen's d | r = d / √(d² + 4) |
| t-statistic | r = √(t² / (t² + df)) |
| F(1, df) | r = √(F / (F + df)) |
| OR | r = log(OR) × √3 / π（近似） |
| β (bivariate) | ≈ r（近似、共変量なしの場合） |
| β (multivariate) | 非推奨 — bivariate が報告されていれば優先 |

### 9.3 Fisher's z 変換

pooling 時は r を Fisher's z に変換:
- z = 0.5 × ln((1+r)/(1-r))
- 逆変換: r = (e^(2z) − 1) / (e^(2z) + 1)

### 9.4 不足情報の対応

- 相関行列が報告されていない → 著者にメールで請求（6 週間待機）
- 返信なし → 当該論文を分析から除外（PRISMA フローで記録）
- 部分的報告（例: 3 traits のみ）→ 該当 traits のみ pool

---

## 10. Risk of Bias 評価

### 10.1 評価ツール

**推奨: ROBINS-E (for observational exposure studies)** または **Joanna Briggs Institute (JBI) checklist for analytical cross-sectional studies**。

### 10.2 評価ドメイン（JBI ベース、8 項目）

1. サンプリングフレームの適切性
2. 参加者・セッティングの記述の詳細
3. 性格測定の validity/reliability
4. サンプルサイズの正当性
5. アウトカム測定の客観性
6. 統計解析の適切性
7. 交絡因子の同定と調整
8. 倫理承認の報告

各項目 Yes / No / Unclear で評価、総合スコア算出。

### 10.3 スコア使用方針

- Risk of bias を sensitivity analysis の moderator として使用
- 低リスク研究のみの subset 分析を実施
- 極端に低品質の研究は予備的に除外を検討

---

## 11. 研究間のサンプル重複対策

同一著者グループが複数論文で同一データを使用することがある（特に Baruth & Cohen シリーズ、Audet et al. シリーズ）:

1. 著者グループ内で複数論文を特定
2. 各論文のサンプル特性（N, 収集期間, 国, etc.）を比較
3. **重複と判断した場合: 最も包括的な 1 本を採用**（原則）
4. 独立サブサンプルの場合: 別 ID として併存

---