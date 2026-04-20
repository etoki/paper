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

## 12. 統計解析計画

### 12.1 Pooling モデル

**Random-effects model を採用**（true effect が study 間で変動する前提）。
- Restricted maximum likelihood (REML) estimator を使用
- Hartung-Knapp-Sidik-Jonkman (HKSJ) 調整で信頼区間を補正（k が少なめの本メタ分析で推奨）

### 12.2 効果量

- 主要効果量: **Pearson r**（Fisher's z 変換後に pooling、最後に r に逆変換）
- 報告: pooled r, 95% CI, k, N (total), p-value

### 12.3 ペアワイズ統合

Big Five の各特性ごとに **5 つの独立メタ分析**を実施:
1. Conscientiousness × achievement
2. Openness × achievement
3. Extraversion × achievement
4. Agreeableness × achievement
5. Neuroticism × achievement

副次分析で:
6. Each trait × satisfaction
7. Each trait × engagement

### 12.4 HEXACO マッピング

HEXACO 研究（特に A-19 MacLean 2022）からは以下を抽出:
- H-Honesty-Humility: 補助分析のみ
- H-Emotionality → Big Five Neuroticism (軸は近いが一致ではない)
- H-eXtraversion → Big Five Extraversion
- H-Agreeableness → Big Five Agreeableness (定義に差あり)
- H-Conscientiousness → Big Five Conscientiousness
- H-Openness → Big Five Openness

マッピング詳細は Ashton & Lee (2007) に依拠し、感度分析で HEXACO-only pool と Big Five-only pool の差分を検証。

---

## 13. 異質性評価（Heterogeneity）

### 13.1 異質性統計量

- **Q-statistic** (Cochran's Q, chi-square test)
- **I²**: 異質性の割合（%）
  - < 25%: 低
  - 25–50%: 中
  - 50–75%: 高
  - > 75%: 極めて高
- **τ²**: Between-study variance
- **95% prediction interval**: 新研究の予測区間

### 13.2 異質性が高い場合の対応

I² > 50% の場合:
- Moderator 分析で説明力を評価
- Sub-group 分析で層化
- Outlier 診断（Studentized residuals）

---

## 14. Moderator / Meta-regression 分析

### 14.1 事前登録するモデレーター

1. **Modality**: Fully online / Blended / MOOC / Synchronous / Asynchronous
2. **Education level**: K-12 / Undergraduate / Graduate / Adult
3. **Region**: Asia / Europe / North America / Other
4. **Era**: Pre-COVID (≤ 2019) / COVID (2020–2022) / Post-COVID (2023–)
5. **Outcome type**: GPA / Exam / LMS behavior / Composite
6. **Personality instrument**: BFI / BFI-2 / NEO-FFI / NEO-PI-R / IPIP / HEXACO
7. **Publication year** (continuous)
8. **Sample size** (continuous, log-transformed)
9. **Risk of bias score** (continuous)

### 14.2 Meta-regression

連続モデレーターには mixed-effects meta-regression を使用。カテゴリカルモデレーターにはサブグループ分析。

### 14.3 多重比較の調整

複数のモデレーター × 複数 trait で多重検定となるため、**Holm-Bonferroni 法**で調整。

---

## 15. 出版バイアス評価

### 15.1 Funnel plot

各 trait ごとに funnel plot を作成し、非対称性を視覚的に検出。

### 15.2 統計的検定

- **Egger's regression test**: 非対称性の検出
- **Begg & Mazumdar's rank correlation test**: 補助的
- **Trim-and-fill method** (Duval & Tweedie): 欠損研究を推定して調整後効果量を報告

### 15.3 p-curve analysis

Simonsohn, Nelson, & Simmons (2014) の p-curve を使用:
- 真の効果の有無を p 値分布で判定
- Publication bias に対する追加検証

---

## 16. 感度分析

以下を実施し結果の頑健性を確認:

1. **Risk of bias 低リスクのみ**: 高品質研究のみで再解析
2. **Peer-reviewed のみ**: gray literature を除いた解析
3. **Leave-one-out**: 各研究を逐次除外した再解析
4. **Outlier 除外**: Studentized residuals > |3| を除外
5. **Influential case 除外**: Cook's distance で判定
6. **Facet-level vs domain-level**: facet 報告研究と domain 報告研究の差分

---

## 17. 使用ソフトウェア

### 17.1 推奨ソフトウェア

| 用途 | ツール |
|------|--------|
| Meta-analysis 計算 | R: `metafor` package（Viechtbauer 2010）推奨 |
| 代替 | R: `meta` package、Stata `metan`、Comprehensive Meta-Analysis (CMA) |
| 検索管理 | Rayyan / Covidence |
| 書誌管理 | Zotero / EndNote |
| Risk of bias | ROBIS / JBI checklist (manual) |
| フォレストプロット | R `metafor::forest()` |
| Funnel plot | R `metafor::funnel()` |

### 17.2 再現性

- 解析スクリプトを `metaanalysis/analysis/` 配下に保存
- データ抽出 CSV も同 dir に保存
- すべて Git 管理

---

## 18. タイムライン（推定）

| フェーズ | 期間 | 内容 |
|---------|------|------|
| **Phase 1**: プロトコル確定 | 1 週間 | 本計画の最終化、PROSPERO 登録 |
| **Phase 2**: データベース検索 | 1 週間 | 全 DB での検索実施、検索ログ記録 |
| **Phase 3**: スクリーニング | 2–3 週間 | Title/abstract → Full-text eligibility |
| **Phase 4**: Full-text 入手 | 1–2 週間 | PDF 取得、著者問合せ |
| **Phase 5**: データ抽出 | 2–3 週間 | 全 included study の抽出 |
| **Phase 6**: Risk of bias 評価 | 1 週間 | JBI/ROBINS-E 適用 |
| **Phase 7**: 統計解析 | 1–2 週間 | Pooling, heterogeneity, moderators |
| **Phase 8**: 執筆 | 3–4 週間 | Methods, Results, Discussion |
| **Phase 9**: 内部レビュー・改訂 | 1–2 週間 | |
| **Phase 10**: 投稿・査読対応 | 変動 | |

**合計**: 約 12–20 週間（3–5 ヶ月）

---

## 19. 報告・成果物

### 19.1 論文本体に含める図表

- **Table 1**: Study characteristics of included studies
- **Figure 1**: PRISMA 2020 flow diagram
- **Figure 2**: Forest plot (per trait)
- **Figure 3**: Funnel plot (per trait)
- **Table 2**: Heterogeneity statistics and meta-regression results
- **Table 3**: Sensitivity analysis summary
- **Figure 4** (optional): Sub-group forest plots by modality/region

### 19.2 Supplementary materials

- 検索ログ（全 DB の検索式・日付・ヒット数）
- データ抽出 CSV
- Risk of bias 評価表
- 除外研究リスト（除外理由付き）
- 分析スクリプト

### 19.3 投稿先候補

| 優先度 | ジャーナル | IF 目安 |
|--------|-----------|---------|
| 高 | *Educational Psychology Review* | 10+ |
| 高 | *Computers & Education* | 12+ |
| 中 | *Review of Educational Research* | 13+ |
| 中 | *Internet and Higher Education* | 8+ |
| 中 | *Journal of Educational Psychology* | 6+ |
| 標準 | *Education and Information Technologies* | 5+ |
| 標準 | *Frontiers in Psychology* (Educational Psychology) | 2.5+ |

### 19.4 PROSPERO 報告

完成後、PROSPERO 登録内容に対する逸脱があれば透明に報告。

---

## 20. 倫理・利益相反

### 20.1 倫理

- 二次データ（既発表論文）のみ扱うため、参加者個別の倫理承認は不要
- 著者への問合せでは、データ使用範囲・引用方針を明示

### 20.2 利益相反

- 筆者の先行研究（Tokiwa, 2025）を含むため、本メタ分析に利益相反の可能性あり
- 論文内で明示的に開示
- 可能なら先行研究を **除外した sensitivity analysis** も報告

---

## 21. 当面の実施ステップ（Next Actions）

本計画に基づき、**次のチャットまたはセッションで最初に実施するタスク**:

### Step 1: PROSPERO 事前確認（1 日）
- PROSPERO で "Big Five online learning" 等で検索
- 同じテーマの登録がないか確認
- あれば scope 再検討、なければ自分の登録準備

### Step 2: 検索ログ作成（2–3 日）
- 各 DB で Section 7.2 の検索式を実行
- ヒット数・実施日を記録
- 全 DB のヒットを Rayyan/Zotero にインポート

### Step 3: 重複除去・初回スクリーニング（1 週間）
- Rayyan で重複除去
- Title/abstract で第一次スクリーニング
- この時点で既知の 28 論文（`literature_review.md` A-01 〜 A-28）がヒットすることを確認

### Step 4: Full-text 取得（並行、1–2 週間）
- `literature_review.md` の「PDF 取得優先度」に従って PDF 入手
- 不明な書誌情報（MacLean 2022, Nishino 2018 等）を PDF で確認

### Step 5: データ抽出開始
- Section 9.1 のフィールドに基づく抽出 CSV 作成
- まず優先度高 10 本から開始

---

## 22. 既知のリスクと対策

| リスク | 対策 |
|--------|------|
| オンライン学習論文が少なすぎる（k < 10 per trait） | satisfaction 等の secondary outcome を追加、narrative synthesis に変更 |
| 同じテーマの先行登録がある | scope を教育段階や特定地域に狭めて差別化 |
| 効果量抽出不能論文が多い | 著者問合せ、部分的報告論文も含める、限界として明示 |
| Publication bias が深刻 | Trim-and-fill と p-curve で調整、限界として明示 |
| LLM 由来の著者誤認が残存 | 全 PDF で目視確認、共著者と相互チェック |
| 単独執筆での reviewer 不足 | intra-rater reliability で代替、限界として明示 |

---

## 23. 参考資料

### 23.1 PRISMA 関連
- Page, M. J., et al. (2021). The PRISMA 2020 statement: An updated guideline for reporting systematic reviews. *BMJ, 372*, n71.
- [PRISMA 2020 Checklist](http://www.prisma-statement.org/PRISMAStatement/Checklist)

### 23.2 メタ分析手法
- Borenstein, M., Hedges, L. V., Higgins, J. P. T., & Rothstein, H. R. (2021). *Introduction to Meta-Analysis* (2nd ed.). Wiley.
- Viechtbauer, W. (2010). Conducting meta-analyses in R with the metafor package. *Journal of Statistical Software, 36*(3), 1–48.
- Harrer, M., et al. (2021). *Doing Meta-Analysis with R: A Hands-On Guide*. CRC Press. [Free online book](https://bookdown.org/MathiasHarrer/Doing_Meta_Analysis_in_R/)

### 23.3 Cochrane
- [Cochrane Handbook for Systematic Reviews of Interventions](https://training.cochrane.org/handbook)

### 23.4 Risk of bias
- [Joanna Briggs Institute Critical Appraisal Tools](https://jbi.global/critical-appraisal-tools)

---

**最終更新**: 本計画は執筆前段階の方針書。実際の実施時には、検索結果や PROSPERO 下調べの結果を踏まえ適宜改訂する。