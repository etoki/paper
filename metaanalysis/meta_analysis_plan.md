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