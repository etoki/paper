# Deep Reading Notes: Big Five × Online Learning × Academic Achievement

本ドキュメントは本メタ分析対象論文の精読ノート。Introduction 執筆と Methods の Risk-of-Bias 評価に必要な情報を、PDF 直接参照で構造化記録する。

`literature_review.md` が narrative 要約であるのに対し、本ファイルは **効果量・方法論・質評価の抽出** を目的とする。

---

## 凡例

- **r**, **β**, **d**, **η²**, **F**, **t**: 効果量・検定統計量
- **N**: サンプルサイズ（analytic N）
- **α**: Cronbach's alpha（信頼性）
- **ρ**: 母相関（メタ分析の pooled 値）
- **CI95**: 95% 信頼区間
- **RoB**: Risk of Bias 8 項目（JBI）スコア（0–8, 高い方が低リスク）
- ⚠ **要追加確認**: 著者名・書誌情報で LLM 由来の不確実性あり
- 🔴 **特記**: 本メタ分析の pooled 結果と比較すべき重要知見

---

# Part A: Benchmark Meta-Analyses（一般学業、オンライン未分化）

本節の目的: 既存メタ分析の pooled ρ 値を**ベンチマーク**として確立し、本メタ分析の「オンライン特化 pooled ρ」との差分検証 (RQ2) に使用する。

## C-01. Poropat (2009)

**Citation**: Poropat, A. E. (2009). A meta-analysis of the five-factor model of personality and academic performance. *Psychological Bulletin, 135*(2), 322–338. https://doi.org/10.1037/a0014996

**Institution**: Griffith University, Australia

### Study characteristics

| 項目 | 値 |
|------|------|
| k (samples) | 109–138 (trait による) |
| Total N | 58,522–70,926 |
| Coverage | ~2008 まで |
| Analysis model | Random-effects with Hunter-Schmidt correction |
| Effect size | ρ (disattenuated correlation) |
| Moderators tested | Academic level (primary/secondary/tertiary), age |

### Overall pooled effect sizes（Table 1）

| Trait | k | N | r | ρ | d | 95% Cred. Int. | ρg (intelligence 控除) |
|-------|---|---|---|---|---|---------------|----------------------|
| **Conscientiousness** | 138 | 70,926 | 0.19 | **0.22** | 0.46 | [-0.09, 0.54] | 0.24 ⭐ 増加 |
| Openness | 113 | 60,442 | 0.10 | 0.12 | 0.24 | [0.09, 0.17] | 0.09 |
| Agreeableness | 109 | 58,522 | 0.07 | 0.07 | 0.14 | [-0.16, 0.30] | 0.07 |
| Emotional Stability | 114 | 59,554 | 0.01 | 0.02 | 0.03 | [-0.29, 0.32] | 0.01 |
| Extraversion | 113 | 59,986 | -0.01 | -0.01 | -0.02 | [-0.32, 0.30] | -0.01 |
| Intelligence | 47 | 31,955 | 0.23 | 0.25 | 0.52 | [-0.18, 0.68] | — |

- **Q, I²**: すべて p < .001 で heterogeneous（I² = 88–97%）
- Conscientiousness は intelligence 控除後も ρg = 0.24（むしろ増加）— C は intelligence と負相関 (-.03) のため
- 全相関は p < .001（大サンプルゆえ practical significance でない差も統計有意）

### Moderation by academic level（Table 2）🔴 重要

| Level | k | N | ρ(A) | ρ(C) | ρ(ES) | ρ(E) | ρ(O) |
|-------|---|---|------|------|-------|------|------|
| Primary | 8 | 3,196 | 0.30 | **0.28** | 0.20 | 0.18 | 0.24 |
| Secondary | 24–35 | 25,488–31,980 | 0.05 | **0.21** | 0.01 | -0.03 | 0.12 |
| Tertiary | 75–92 | 27,944–32,887 | 0.06 | **0.23** | -0.01 | -0.01 | 0.07 |

**重要な発見**:
- **Conscientiousness のみ**教育段階を通じて安定（0.21–0.28）
- 他の全特性は primary から secondary/tertiary で急激に減衰
- Openness は secondary まで持続 (0.12) → tertiary で減衰 (0.07)
- Intelligence も同様に減衰（primary 0.58 → tertiary 0.23）= range restriction

### Incremental validity over secondary GPA（著者の独自分析）

| Predictor | rpartial（tertiary GPA への追加予測力） |
|-----------|-------------------------------------|
| **Conscientiousness** | **0.17** ⭐（intelligence より大） |
| Intelligence | 0.14 |
| Agreeableness | 0.05 |
| Openness | 0.03 |
| Emotional Stability | -0.01 |
| Extraversion | 0.00 |

- 過去の成績 (secondary GPA, ρ=.35) 控除後も C は tertiary GPA を独立に予測
- これは C が過去成績に captured されない「新しい情報」を持つことを示す

### 仕事パフォーマンスとの比較（Barrick et al. 2001）

Poropat は FFM × 仕事パフォーマンス meta-analysis と本結果を Westen & Rosenthal 比較:
- Primary education: 仕事との対応小 (ralerting = .37)
- Secondary: 中程度 (.60)
- Tertiary: 高 (.15 alerting / .79 contrast) ⭐

**含意**: 高等教育は職場行動により近く、C が支配的になる。

### 年齢の moderation（Table 3）

- Primary: 年齢↑ → C, ES, E ↑ / O ↓
- Secondary: 年齢↑ → A, ES, O ↓
- Tertiary: 年齢の影響 n.s.

### Limitations（Poropat 自身の記述）

1. k < 10 for primary education → 推定不安定
2. Agreeableness の credibility interval が広い [-0.16, 0.30]
3. 公開バイアス未検証（本メタ分析の課題）
4. Cross-cultural 分析なし（Mammadov 2022 が補填）

### 🔴 本メタ分析（Tokiwa）への含意

1. **ベンチマーク ρ(C) = 0.22**: 本メタ分析（オンライン学習特化）の pooled ρ との第一次比較対象
2. **Tertiary level の ρ は primary/secondary より低い**: 本研究サンプル（主に大学）での効果量期待値の目安
3. **Intelligence 控除後も C は ρg = 0.24**: オンライン学習でも C の独自寄与が期待される
4. **Poropat vs Mammadov の差**: 
   - Poropat 2009: C ρ = 0.22 (N=70,926)
   - Mammadov 2022: C ρ = 0.27 (N=413,074)
   - 13 年の研究蓄積で effect size が上方修正 → 本研究も類似の上昇があり得る
5. **Intro 骨子への直接引用**: 「Conscientiousness is the most robust non-cognitive predictor of academic performance (Poropat, 2009; Mammadov, 2022)」が確立された定式

### RoB（JBI 8 項目推定）

| 項目 | 評価 | 根拠 |
|------|------|------|
| 1. 包含基準明示 | Yes | Method 詳述 |
| 2. サンプル記述 | Yes | Table 1, 2 |
| 3. Exposure 妥当性 | Yes | validated FFM measures |
| 4. Outcome 客観性 | Partial | GPA/grade 混在 |
| 5. Confounder 特定 | Yes | Intelligence 含む |
| 6. Confounder 対処 | Yes | Partial correlations |
| 7. Outcome 信頼性 | Partial | 測定法 moderator 未実施 |
| 8. 統計適切性 | Yes | WLS regression |
| **Aggregate** | **7/8** | **低リスク**（Mammadov 2022 より若干劣る：publication bias 検定なし） |

## C-02. Vedel (2014)

**Citation**: Vedel, A. (2014). The Big Five and tertiary academic performance: A systematic review and meta-analysis. *Personality and Individual Differences, 71*, 66–76. https://doi.org/10.1016/j.paid.2014.07.011

**Institution**: Department of Psychology and Behavioural Sciences, Aarhus University, Denmark

### Study characteristics
| 項目 | 値 |
|------|------|
| k (samples) | 21 independent samples (from 20 studies) |
| Total N | 17,717 |
| Coverage | 1996–2013 |
| Analysis model | Random-effects (CMA software) |
| Effect size metric | r+ (weighted, uncorrected); trim-and-fill adjusted |

### Overall pooled effect sizes (Table 2)
| Trait | r+ | 95% CI | I² | Q | Fail-safe N |
|-------|-----|--------|-----|---|-------------|
| **Conscientiousness** | **.26*** | [.23, .30] | 72.10% | 71.67*** | 3,625 |
| Openness | .07*** | [.03, .11] | 71.18% | 69.39*** | 222 |
| Agreeableness | .08*** | [.05, .11] | 54.50% | 43.96*** | 280 |
| Neuroticism | −.01 ns | [−.05, .03] | 73.64% | 75.86*** | 0 |
| Extraversion | −.00 ns | [−.03, .02] | 29.08% | 28.20 | 0 |

Trim-and-fill 調整後の Conscientiousness r+ = .24***（微減、堅牢）

### Moderator findings（academic major: psychology vs. other）
- **Conscientiousness**: Psychology r+ = .31 [.26, .37] vs. Other r+ = .22 [.17, .27], Q = 5.05*, **R² = .68** 🔴
- Openness: no moderation (psy .09 vs. other .06, ns)
- Agreeableness: no moderation (psy .07 vs. other .09, ns)
- **Validity threat flagged**: 心理学専攻への過剰依存が結果を押し上げている可能性

### Key findings
- C が tertiary GPA の最強予測因子 (r+ = .26), trim-and-fill および one-study-removed でも堅牢
- A と O は統計有意だが weak (r+ = .07–.08)
- N と E は GPA と実質的に無関係
- 限定的な 5 validated Big Five measures のみ使用しても、Poropat (2009) より大きな効果は得られず
- Major が C-GPA 関係を moderate（68% の heterogeneity を説明）

### Limitations（著者の記述）
- Psychology 専攻依存、subgroup は 2 群比較のみ
- 線形性仮定（曲線関係未検証）
- Range restriction 未補正（GPA は上方選択集団）
- Achievement Motivation mediation 未検証

### 🔴 含意（本メタ分析への）
- **Benchmark value**: C-GPA の FtF tertiary ceiling = ρ ≈ .24–.26
- **Hypothesis test (H2/H4)**: オンラインで C-GPA ρ ≥ .26 なら self-regulation premium 仮説を支持
- **Gap filled**: Vedel は delivery mode (online vs. FtF) を全く検討せず → 本研究が online-specific benchmark を提供
- **Moderator implication**: 専攻 (major/discipline) を本研究でも moderator に入れるべき

### RoB（JBI 8-item）

| 項目 | 評価 |
|------|------|
| Review Q 明示 | Yes |
| 包含基準適切 | Yes (5 validated measures) |
| 検索戦略適切 | Yes (3 DBs + manual) |
| Source 網羅 | Yes |
| Critical appraisal tool | Unclear（formal なし） |
| 複数 reviewer | **No（単著）** |
| Error 最小化 | Yes (sensitivity + trim-fill) |
| Pooling 適切 | Yes |
| **Aggregate** | **6/8** |

## C-03. Mammadov (2022) 🔴

**Citation**: Mammadov, S. (2022). Big Five personality traits and academic performance: A meta-analysis. *Journal of Personality, 90*(2), 222–255. https://doi.org/10.1111/jopy.12663

**Received**: 14 Feb 2021 | **Revised**: 9 Jul 2021 | **Accepted**: 12 Jul 2021

### Study characteristics

| 項目 | 値 |
|------|------|
| k (independent samples) | **267** |
| k (unique studies) | **228** |
| Total N | **413,074** |
| Range of publication years | 〜2020 まで |
| Analysis model | Random-effects, Hunter-Schmidt corrected for measurement error |
| Effect size metric | ρ (disattenuated correlation) |
| Data availability | OSF https://osf.io/tkwdu/ |

### Overall pooled effect sizes（corrected ρ）

| Trait | ρ | 95% CI | d (Cohen) | 統計的有意 |
|-------|---|--------|-----------|-----------|
| **Conscientiousness** | **0.27** | — | 0.56 | p < .001 ⭐ 最強 |
| Openness | 0.16 | — | 0.33 | p < .001 |
| Agreeableness | 0.09 | — | 0.19 | p < .001 |
| Extraversion | 0.01 | — | 0.03 | **n.s.** (CI includes 0) |
| Neuroticism | -0.02 | — | -0.04 | **n.s.** (CI includes 0) |

- Uncorrected overall mean: r = 0.08
- Heterogeneity: **I² ≈ 97.63%** (extreme) → moderator 必須
- Conscientiousness は cognitive ability 統制後も **28% の説明分散**を保持

### Moderation by education level（Table 3）

| Level | k | ρ(O) | ρ(C) | ρ(E) | ρ(A) | ρ(N) |
|-------|---|------|------|------|------|------|
| Elementary/Middle (ref) | 24–31 | **0.40** 🔴 | 0.31 | 0.15 | 0.18 | -0.01 |
| Secondary | 48–55 | 0.22 | 0.27 | 0.01 | 0.08 | -0.01 |
| Postsecondary | 152–175 | 0.10 | **0.26** | -0.01 | 0.08 | -0.02 |

**含意**:
- Openness は**年齢とともに減衰**（小学生最強、大学生最弱）
- Conscientiousness は**教育段階を通じて安定**（0.26–0.31）
- Extraversion, Agreeableness も elementary で最強 → secondary で減衰
- Neuroticism は全段階で n.s.

### Moderation by region（Table 5）🔴 本メタ分析 H1 の根拠

| Region | k | ρ(O) | ρ(C) | ρ(E) | ρ(A) | ρ(N) |
|--------|---|------|------|------|------|------|
| **Asia** | 16 | 0.29 | **0.35** 🔴 | 0.16 | 0.23 | **-0.19** |
| Australia | 7–9 | 0.17 | 0.24 | 0.06 | 0.12 | -0.02 |
| E. Europe/Russia | 24–26 | 0.18 | 0.30 | 0.00 | 0.07 | 0.03 |
| Middle East | 10–11 | 0.15 | **0.35** | 0.07 | 0.17 | -0.02 |
| **N. America** | 74–96 | 0.14 | 0.23 | -0.01 | 0.09 | 0.00 |
| W. Europe | 93–104 | 0.17 | 0.28 | -0.01 | 0.07 | -0.01 |

**Mammadov 本人の caveat**: Asian samples は全体の **1.5%** しかなく、結果の解釈には注意が必要。WEIRD 偏重の既存文献に対する限界として報告。

### Moderation by personality measure（Table 4）

| Measure | ρ(C) | ρ(O) | ρ(N) |
|---------|------|------|------|
| BFI | 0.26 | 0.10 | 0.00 |
| BFQ | 0.30 | 0.45 (outlier?) | -0.01 |
| IPIP | 0.31 | 0.10 | -0.02 |
| Markers (Mini) | 0.18 (最低) | 0.06 | 0.01 |
| NEO-FFI | 0.28 | 0.13 | **-0.07** ⭐ |
| NEO-PI-R | 0.28 | 0.13 | 0.01 |

**含意**:
- Conscientiousness の範囲 0.18–0.31 — Markers が最弱（performance-striving facet を欠くため）
- BFI は C のサンプル内異質性大
- Neuroticism は NEO-FFI でのみ有意負（他は n.s.）

### Comparison with prior meta-analyses（自己批判）

| Study | ρ(C) | N |
|-------|------|---|
| Poropat (2009) | 0.22 | 70,926 |
| McAbee & Oswald (2013) | 0.26 | 26,382 |
| Vedel (2014) | 0.26 | 17,717 |
| **Mammadov (2022)** | **0.27** | **413,074** |

- Mammadov 2022 は C の estimation を **最も堅牢**に確立（N が 5.8 倍）
- Openness は過去の meta-analyses より大きく推定（Poropat 0.12 → Mammadov 0.16）

### Incremental validity（Table 8）

| Predictor | B | RW | RW% |
|-----------|---|-----|-----|
| Cognitive ability | 0.42 | 0.177 | **63.59%** |
| Conscientiousness | 0.35 | 0.078 | **27.93%** |
| Openness | 0.03 | 0.011 | 3.94% |
| Neuroticism | 0.13 | 0.005 | 1.88% |
| Agreeableness | -0.02 | 0.005 | 1.84% |
| Extraversion | -0.05 | 0.002 | 0.81% |
| Total R² | — | — | **27.8%** |

- Conscientiousness は cognitive ability 控除後も**相対重要度 27.93%** を保持
- Openness は 3.94%（誤差レベル）
- 他の Big Five は negligible

### Meta-regression（gender, age）

- Openness: 女性比率↑ → ρ↓ (B=-0.003***), 年齢↑ → ρ↓ (B=-0.014***)
- Extraversion: 女性比率↑ → ρ↓ (B=-0.001*), 年齢↑ → ρ↓ (B=-0.004*)
- C, A, N: gender・age の影響小

### Publication bias

- Egger's regression: **C 以外すべて有意**（β0 = -0.59〜0.83）
- C は CI = [-2.62, 1.44] で bias なし
- Trim-and-fill 実施済み
- 出版バイアスと extreme heterogeneity の区別は困難と注記

### Limitations（本人記述）

1. 出版バイアスの証拠（Egger test）→ 効果量過大推定の可能性
2. Academic performance 測定の異質性（GPA, exam, test score が混在）
3. Asian 16 samples 少なく、region moderation の解釈に留保
4. Facet-level 分析なし（長尺版 inventory の研究限定のため）

### 🔴 本メタ分析（Tokiwa）への含意

1. **H1 (C 最強) の根拠**: 本 meta-analysis で C ρ = 0.27 は教育段階・地域・測定法を問わず堅牢 → オンライン学習でも同様が予測される
2. **H1 の Asian 特化**: Asian samples で C ρ = 0.35 → 日本サンプル（Nakayama, Tokiwa）でも高めの推定期待
3. **ベンチマーク比較 (RQ2)**: 本メタ分析の online-specific pooled ρ と Mammadov の ρ を差分検定すべき
4. **Openness の age decline**: オンライン学習は主に大学生 → 本研究では O の効果は小さい可能性（H2 と整合するかは k 次第）
5. **Extraversion の elementary 効果**: K-12 sub-analysis で検証可能か → A-26 Wang (2023) K-12 中国が重要
6. **測定法モデレーター**: NEO-FFI 採用研究では N が有意負 → 本研究でも measure を moderator に入れる必要

### RoB（Risk of Bias, JBI 8 項目、推定）

| 項目 | 評価 | 根拠 |
|------|------|------|
| 1. サンプル包含基準明示 | Yes | Method 2.1 |
| 2. サンプル設定記述 | Yes | Table 1 詳細 |
| 3. Exposure 妥当性 | Yes | 6 validated measures |
| 4. Outcome 客観性 | Partial | GPA/exam/test mixed |
| 5. Confounder 特定 | Yes | Cognitive ability 含む |
| 6. Confounder 対処 | Yes | Incremental validity 検定 |
| 7. Outcome 信頼性 | Yes | 測定法 moderator |
| 8. 統計適切性 | Yes | Hunter-Schmidt + subgroups |
| **Aggregate** | **8/8** | **低リスク**（C-01〜C-05 中最高品質） |

## C-04. Stajkovic, Bandura, Locke, Lee, & Sergent (2018)

**Citation**: Stajkovic, A. D., Bandura, A., Locke, E. A., Lee, D., & Sergent, K. (2018). Test of three conceptual models of influence of the big five personality traits and self-efficacy on academic performance: A meta-analytic path-analysis. *Personality and Individual Differences, 120*, 238–245. https://doi.org/10.1016/j.paid.2017.08.014

**Institution**: University of Wisconsin-Madison

### Study characteristics
| 項目 | 値 |
|------|------|
| k (samples) | 5 independent primary samples（従来的文献 meta-analysis でなく、著者収集データ） |
| Total N | 875 (GMA subset N = 744) |
| Coverage | Cross-sectional, 3 universities（Midwestern US + 韓国私立） |
| Analysis model | Meta-analytic path-analysis (LISREL 8), Hunter-Schmidt 方式 |
| Effect size metric | r → G(r) → Z+ → weighted-average ρ; β for path coefficients |

### Overall pooled effect sizes（Table 3: meta-analytic ρ with performance）

| Predictor → Performance | G(r+) | 95% CI | Qt |
|------------------------|-------|--------|-----|
| **Conscientiousness** | **0.21** | [.14, .28] | 4.03 (homog.) |
| Agreeableness | 0.05 | [-.02, .12] | 3.03 |
| Extraversion | -0.02 | [-.09, .05] | 10.22† (heterog.) |
| Openness | -0.01 | [-.08, .06] | 0.07 |
| Emotional stability | 0.00 | [-.07, .07] | 3.03 |
| Self-efficacy | 0.33 | [.26, .40] | 2.34 |
| GPA (prior) | 0.48 | [.41, .55] | 5.21 |
| GMA | 0.26 | [.19, .33] | 4.88 |

### Path-analytic findings（Table 4, harmonic mean N = 842）

- **C → SE** β = .16**; **C → performance** β = .11** (direct)
- **ES → SE** β = .18**; **ES → performance** β = -.08* (direct)
- A, E, O: "fleeting" — 有意性が散発的
- **SE → performance** β = .24** (一貫して強い mediator)
- R² performance: Trait model = .30, Independent = .26, Intrapersonal = .29
- **Intrapersonal (fully-mediated) model が最 parsimonious** (χ² = 18.06, CFI = .99, RMSEA = .056)

### Key findings
- C と ES（逆コード N）のみが SE を経由して academic performance に一貫寄与
- A, E, O は 6 モデル中で有意性散発
- Self-efficacy が最も predictive な proximal variable — Big Five を部分〜完全 mediate
- Full mediation via SE の model が trait model にほぼ劣らず parsimony で勝る

### 🔴 含意（本メタ分析への）
- **Benchmark value**: C → Performance ρ ≈ .21; SE が強い mediator (β = .24–.33)
- **Gap**: オンライン環境で SE mediation が強まる/弱まるかは未検証 → 本研究で SE / self-regulation を mediator として含めれば online-specific path が描ける
- **仮説**: Online mode では **SE mediation が強化** される（physical oversight なしで学習行動を自律的に調整する必要）→ C の direct path は弱く、indirect path (via SE) が強い可能性

### RoB（JBI 8-item）
**Aggregate: 3–4/8** — 伝統的メタ分析ではない点に注意（独自データ収集、5 sample のみ）

---

## C-05. McAbee & Oswald (2013)

**Citation**: McAbee, S. T., & Oswald, F. L. (2013). The criterion-related validity of personality measures for predicting GPA: A meta-analytic validity competition. *Psychological Assessment, 25*(2), 532–544. https://doi.org/10.1037/a0031748

**Institution**: Rice University

### Study characteristics
| 項目 | 値 |
|------|------|
| k (samples) | 57 (C); 49–51 (他特性) |
| Total N | 26,382 (C) |
| Coverage | 1992–2012 literature |
| Analysis model | Hunter-Schmidt psychometric meta-analysis (random-effects) |
| Effect size metric | r, r+ (operational validity), ρ (true-score) |

### Overall pooled effect sizes（Tables 2–6）

| Trait | N | k | r | r+ (operational) | 95% CI | ρ |
|-------|---|---|---|------------------|--------|---|
| **Conscientiousness** | 26,382 | 57 | .22 | **.23** | [.20, .24] | **.26** |
| Openness | 24,996 | 51 | .07 | .07 | [.05, .09] | .08 |
| Agreeableness | 24,615 | 49 | .06 | .06 | [.04, .07] | .07 |
| Neuroticism | 24,968 | 51 | -.00 | -.00 | [-.02, .02] | -.00 |
| Extraversion | 24,740 | 50 | -.02 | -.02 | [-.04, -.00] | -.03 |

### Measure-specific operational validities for Conscientiousness

- **NEO-PI-R** r+ = **.26**
- NEO-FFI r+ = .24
- BFI r+ = .24
- IPIP r+ = .21
- **Markers r+ = .15** (significantly lower; diff from NEO by .09–.10, CI excluding 0)

### Moderator findings
- 測定尺度は C-GPA 以外で有意な moderator（O の NEO-FFI r+ = .12 > BFI r+ = .02; A の NEO-FFI r+ = .11 highest）
- Outlier 除外後 (Noftle & Robins 2007, McKenzie & Gow 2004) も結果は堅牢

### Key findings
- C が GPA の最強 validity (r+ = .23; ρ = .26) — 5 measures で堅牢
- Markers は C 予測で弱め (r+ = .15) — performance-striving facet 少なめのため
- E と N は全 measure で r+ < .05（実質ゼロ）
- O と A の validity は measure 依存 — NEO-FFI で最高

### 🔴 含意（本メタ分析への）
- **Benchmark value**: C-GPA の FtF ceiling (ρ = .26, r+ = .23) = 1992–2012 pooled estimate
- **Gap**: McAbee & Oswald は instrument comparison が焦点で、delivery mode (online vs. FtF) は未検討 → 本研究が online-specific benchmark を提供
- **Moderator implication**: 本研究でも **measure** を moderator として扱うべき（BFI/NEO/IPIP で online studies の効果が異なる可能性）

### RoB（JBI 8-item）: **6/8**

---

# Part B: Existing Systematic Reviews（online-specific）

## D-01. Hunter et al. (2025) ⚠ 著者名訂正

**重要訂正**: lit review では "Gray & DiLoreto (2024)" とされていたが、実際は **Hunter et al. (2025)** (Journal of Occupational Therapy Education)。

**Citation**: Hunter, E. G., Niblock, J., Barefoot, S., Miller, J., Hughes, J., Kite, L., & Scarletto, E. (2025). The Influence of Student Personality Traits on Satisfaction and Success in Online Education in Higher Education: A Systematic Review. *Journal of Occupational Therapy Education, 9*(2), Article 9. https://doi.org/10.26681/jote.2025.090209

**Institution**: Bowling Green State University, USA

### Study characteristics
| 項目 | 値 |
|------|------|
| k (included) | 23 (from 848 de-dup, 99 full-text) |
| Coverage | Jan 2000 – June 2024 |
| Analysis model | **Qualitative thematic synthesis**（narrative/vote-counting） |
| Effect size metric | **None pooled** — narrative only |
| 全研究 Level | **IV (low strength)** — RCT なし |
| Critical appraisal | MMAT used |

### 3 themes identified

1. **Interaction between personality & online education**: GPA / satisfaction / assignment / engagement
2. **Comparing personality on multiple class formats**: satisfaction / grades
3. **Personality × student choice of class format**

### Key narrative findings（効果量の pooling なし）

- **GPA (fully online, 5 studies, N=1,542)**: C, O, A, ES が 3/5 studies で正相関; N が 3/5 で負相関; **E は全ての study で GPA と無関連**
- **Satisfaction (4 studies, N=862)**: C は consistent に higher satisfaction を予測; A と O も正; E は無関連
- **Engagement (Wu & Yu 2023, N=1,004)**: E, A, O, C 正 / N 負; adaptability が媒介
- **Class format × personality**: 外向性は FtF を好み、内向性は online を好む; 神経症傾向が高い学生は online で苦戦
- **Choice of format**: High C, A, O → online を選択; high N → online 回避

### 🔴 含意（本メタ分析への）
- **Benchmark value**: Hunter et al. は **qualitative direction** のみで pooled effect sizes なし → 本メタ分析が **初の quantitative online-specific pooling** となる
- **Gap filled by our study**: (1) 量的 pooled effect sizes, (2) moderator analyses, (3) RoB-weighted synthesis — 全て Hunter et al. が提供していない
- **Specific hypothesis support**:
  - H1 (C → online GPA positive): 支持（3/5 studies）
  - H2 (N → online GPA negative): 概ね支持、contradicting evidence あり
  - H3 (E is null online): 強く支持（"None of the 5 studies found extraversion influenced GPA"）
  - H4 (satisfaction vs GPA は異なる pattern): Myers-Briggs が satisfaction-only effects を示したことで支持

### RoB（JBI 8-item）: **7/8**（最高品質 review、但し量的 synthesis 欠落）

---

# Part C: Primary Studies（本メタ分析対象）

## A-01. Abe (2020)

**Citation**: Abe, J. A. A. (2020). Big Five, linguistic styles, and successful online learning. *The Internet and Higher Education, 45*, 100724. https://doi.org/10.1016/j.iheduc.2019.100724
**Modality**: Fully online / asynchronous | **Country**: USA (Southern Connecticut State Univ.) | **Era**: pre-COVID

### Sample
- N=92 (1 outlier excluded), undergraduate Psychology, response 77%
- Personality course context

### Measures
- BFI-44; α range .74–.84
- Outcome 1: Quiz avg (8 quizzes); Outcome 2: Final paper grade (13-pt scale)

### Effect sizes（Table 2 correlations）🔴 **本研究の primary achievement 数値**

| Trait | r × Quiz | r × Paper |
|-------|---------|-----------|
| **C** | **.48** (p<.01) | **.37** (p<.01) |
| O | .13 (ns) | **.35** (p<.01) |
| E | -.07 (ns) | .03 (ns) |
| A | .13 (ns) | .16 (trend) |
| N | -.10 (ns) | -.02 (ns) |

Regression: Quiz avg — C β=.46 (R²=.28); Paper — C β=.35, O β=.26 (R²=.31)

### Key findings
- C は quiz/paper 両方で最強予測 → cognitive task と essay 両方で safety net
- O は essay grade のみ予測（深い処理を要する課題で発現）
- E/A/N は achievement と直交
- Word count（LIWC）が最 robust な linguistic predictor

### Status: **Include** ✅（primary outcome × full Big Five 揃う、稀少）
### RoB: **5/8**（demographics 報告なし）

---

## A-02. Alkış & Taşkaya Temizel (2018)

**Citation**: Alkış, N., & Taşkaya Temizel, T. (2018). The impact of motivation and personality on academic performance in online and blended learning environments. *Educational Technology & Society, 21*(3), 35–47.
**Modality**: Mixed (online N=189 + blended N=127, 別計算) | **Country**: Turkey (METU) | **Era**: pre-COVID

### Sample
- N=316 total, undergraduate, mixed disciplines
- Online: 58% female, M age 22.27; Blended: 60% female, M age 22.03
- Response 59% (381/658)

### Measures
- BFI-44 Turkish; α: E=.84, A=.60, C=.76, N=.81, O=.81
- Outcome: Course grade (0–100, midterm + final + assignments) + LMS access count

### Effect sizes（Tables 3 & 4）🔴 **online setting で primary achievement**

| Trait | r × Grade (Online N=189) | r × Grade (Blended N=127) |
|-------|------------------------|---------------------------|
| **C** | **.205** (p<.01) | **.244** (p<.01) |
| O | -.092 (ns) | .082 (ns) |
| E | .051 (ns) | .004 (ns) |
| A | .094 (ns) | .024 (ns) |
| N | .03 (ns) | -.005 (ns) |

BSEM: Online — Grade ← C (b=6.19, 95%CI [0.54, 11.94]); Blended — Grade ← C (b=8.63, 95%CI [2.43, 15.06])

### Key findings
- C のみ両モダリティで grade 予測（online .205, blended .244）
- 他 4 特性は orthogonal
- LMS use mediates C → grade in online (not blended)
- Self-efficacy predicts grade in online only

### Status: **Include** ✅（online subset N=189 を primary, blended は moderator として別扱い推奨）
### RoB: **6/8**

---

## A-03. Ashouri et al. (2025)

**Citation**: Ashouri, A., Taheri, M., Rasouli, M., & Rouhalamin, S. (2025). Personality traits and e-learning course satisfaction: A study of health science students. *Cureus, 17*(7), e89131.
**Modality**: Fully online (Navid LMS) | **Country**: Iran | **Era**: COVID 2022 collection

### Sample
- N=183, undergraduate health science, 75% female, M age ~23
- BFI-44 Persian (Nosratabadi)

### Effect sizes（hierarchical regression β only — no zero-order r）

| Trait | β × Satisfaction subscales |
|-------|---------------------------|
| N | -.17 〜 -.23 (有意, 全 subscale で負) |
| A | +.19 (Tech subscale only) |
| C/E/O | ns |

### Status: **Exclude** for primary (achievement なし、satisfaction のみ); 副次プールに include 可
### RoB: **5/8**

---

## A-04. Audet, Levine, Metin, Koestner & Barcan (2021)

**Citation**: Audet, É. C., et al. (2021). Zooming their way through university: Which Big 5 traits facilitated students' adjustment to online courses during the COVID-19 pandemic. *Personality and Individual Differences, 180*, 110969.
**Modality**: Fully online (synchronous Zoom) | **Country**: Canada (McGill) | **Era**: COVID Fall 2020

### Sample
- N=350 T1 → N=167 T2 (48% retention), undergraduate
- 87.8% female, M age 19.75

### Measures
- BFI-44 (α > .80 全特性)
- Outcomes: self-efficacy, motivation, engagement, SWB（**No GPA — explicitly noted limitation**）

### Effect sizes（regression b, Table 2）

| Trait | b × Self-efficacy | b × Engagement (T2) |
|-------|------------------|--------------------|
| C | .22 (p<.01) | ns |
| O | .14 (p<.01) | **.27** (p<.001) ⭐ |
| N | -.13 (p<.05) | ns |
| E/A | ns | ns |

### Key findings
- O が longitudinal engagement の単独予測（COVID 期 isolated 環境）
- C と O が self-efficacy で優位
- N → controlled motivation 上昇

### Status: **Exclude** for achievement（GPA 測定なし）; engagement secondary プール候補
### ⚠ **Sample overlap warning**: A-05 と Fall 2020 cohort 共有
### RoB: **6/8**（longitudinal だが retention 48%）

---

## A-05. Audet et al. (2023) ⚠ A-04 と sample overlap

**Citation**: Audet, É., Levine, S., Dubois, P., Koestner, S., & Koestner, R. (2023). The unanticipated virtual year: How the Big 5 personality traits of Openness to Experience and Conscientiousness impacted engagement in online classes during the COVID-19 crisis. *Journal of College Reading and Learning, 53*(4), 298–315.
**Modality**: Fully online | **Country**: Canada (McGill) | **Era**: COVID Fall 2020 + Winter 2021

### Sample
- Fall 350 + Winter 323（Fall は A-04 と同 cohort）
- BFI-44

### Effect sizes（Table 2 correlations combined, engagement のみ）

| | C | O | SE | IM |
|---|---|---|----|-----|
| Engagement | .14* | .15** | .33** | .19** |

Subgroup regression:
- O × engagement Fall b=.29 (p<.001), Winter b=.03 (ns)
- C × engagement Fall b=-.05 (ns), Winter b=.28 (p<.001) ← **temporal interaction**

### Key findings
- O drove Fall engagement（"training period"）
- C drove Winter engagement（"work period"）
- Mediation: O→IM→engagement (partial); C→SE→engagement (full)

### Status: **Exclude** for achievement; **重大な double-counting risk**（Fall=A-04 cohort）→ A-04 か A-05 のどちらか一方のみ採用
### RoB: **6/8**

---

## A-06. Sahinidis & Tsaknis (2021) 🔴 **lit review 著者誤認 — 訂正必要**

**重要**: lit review では "Baruth & Cohen 2021" とされていたが、実 PDF は **Sahinidis & Tsaknis (2021, Greece)** — LLM 由来の著者誤認。

**Citation**: Sahinidis, A. G., & Tsaknis, P. A. (2021). Exploring the relationship of the Big Five personality traits with student satisfaction with synchronous online academic learning: The case of Covid-19-induced changes. In *Strategic Innovative Marketing and Tourism* (Springer Proceedings).
**Modality**: Synchronous online (lockdown) | **Country**: Greece (Univ. of West Attica) | **Era**: COVID March-April 2020

### Sample
- N=555, undergraduate, 59% female
- 30-item Big Five (著者独自 scale, NOT BFI); α: O=.79, C=.75, E=.75, N=.70, A=.57 (低)

### Effect sizes（standardized β only — no zero-order r）

| Trait | β × Satisfaction (synchronous online) |
|-------|---------------------------------------|
| O | **.469** (p<.001) ⭐ 最強 |
| C | **.338** (p<.001) |
| N | -.082 (p=.009) |
| E | .018 (ns) |
| A | -.024 (ns) |

R²=.507

### Status: **Exclude** for primary（satisfaction only, non-standard scale, A α<.70）
### ⚠ **Authorship correction needed in literature_review.md**
### RoB: **4/8**

---

## A-07. Cohen & Baruth (2017) 🔴 **lit review 著者・年次誤認 — 訂正必要**

**重要**: lit review では "Baruth & Cohen 2023" だが、実 PDF は **Cohen & Baruth (2017)** — author 順 + 年次が異なる。

**Citation**: Cohen, A., & Baruth, O. (2017). Personality, learning, and satisfaction in fully online academic courses. *Computers in Human Behavior, 72*, 1–12. https://doi.org/10.1016/j.chb.2017.02.030
**Modality**: Fully online (mostly async) | **Country**: Israel (Tel Aviv U.) | **Era**: pre-COVID

### Sample
- N=72, post-BA teacher education, 63% female, M age ~30
- BFI-44 Hebrew; α: E=.81, N=.89, A=.84, C=.78, O=.73

### Effect sizes（Tables 3 & 5）

| Trait | r (Spearman) × Satisfaction | β |
|-------|----------------------------|----|
| O | **.376** (p<.01) | .416 (p<.001) |
| C | **.390** (p<.01) | .451 (p<.001) |
| A | .099 (ns) | -.067 (ns) |
| E | .025 (ns) | -.212 (p=.066) |
| N | .041 (ns) | .074 (ns) |

R²=.30

### Key findings
- O と C のみ satisfaction を予測
- E は trend で負（外向者は online で満足度低い）
- 3 cluster 解析で synchronous preference 差異

### Status: **Exclude** for primary（satisfaction only, N=72 small, bespoke scale）
### ⚠ **Authorship correction needed in literature_review.md** (this is 2017, not 2023)
### RoB: **5/8**

---

## A-08. Keller & Karau (2013) 🔴 **lit review 著者誤認 — 訂正必要**

**重要**: lit review では "Bhagat et al. 2019" だが、実 PDF は **Keller & Karau (2013)** — 完全に異なる論文（Bhagat 2019 は実は A-18 の PDF）。

**Citation**: Keller, H., & Karau, S. J. (2013). The importance of personality in students' perceptions of the online learning experience. *Computers in Human Behavior, 29*(6), 2494–2500. https://doi.org/10.1016/j.chb.2013.06.007
**Modality**: Fully online (Blackboard, async) | **Country**: USA Southeast | **Era**: pre-COVID

### Sample
- N=250, mixed (59% UG + 41% Grad), M age 35 (range 19–57), 73% female
- Discipline: business 36%, nursing 26%, integrated 13%, education 10%, etc.
- IPIP 50-item; α: E=.87, A=.83, C=.81, ES=.86, O=.77

### Effect sizes (Table 2: r with OCI subscales)

| Trait | r × Engagement | r × Career value | r × Overall eval | r × Anxiety | r × Online pref |
|-------|---------------|------------------|------------------|-------------|----------------|
| **C** | **.39** | .26 | **.32** | -.24 | .17 |
| O | .18 | .21 | — | — | — |
| A | — | .25 | — | — | — |
| E | .02–.12 (ns) | — | — | — | — |
| ES | — | .18 | — | — | — |

(全て p<.01)

### Key findings
- C が 5 OCI subscales 全てで最 stable な予測因子
- O と A は career value のみ
- E は全 OCI と無関連

### Status: **Ambiguous** — 副次（OCI=perceptions/satisfaction, no GPA）プールに include 可
### ⚠ **Authorship correction needed in literature_review.md**（A-08 = Keller & Karau, NOT Bhagat）
### RoB: **5/8**

---

## A-09. Rani Bhattacharjee & Ramkumar (2025) ❌ **EXCLUDE**

**Citation**: Rani Bhattacharjee, R., & Ramkumar, A. (2025). Effect of big five personality dimensions on the academic performance of college students. *Frontiers in Psychology, 16*, 1490427.
**Modality**: **Face-to-face** (Tamil Nadu engineering college; not online) | **Country**: India | **Era**: post-COVID

### Sample
- N=384 (purposive from 600), 1st-year engineering, 43% female
- BFI 44; Goldberg 1993; α .76–.84

### Effect sizes
- 個別 r 報告なし; group means (GPA>7 vs <7) のみ
- C のみ有意差（M=36.26 vs 29.75, t<.001, d≈1.35）
- N が逆方向で高 GPA 群高い（反直感）

### Status: ❌ **EXCLUDE** — オンライン要素なし（face-to-face engineering）
### RoB: **4/8**

---

## A-10. Boonyapison, Sittironnarit & Rattanaumpawan (2025) ❌ **EXCLUDE**

**Citation**: Boonyapison, K., Sittironnarit, G., & Rattanaumpawan, P. (2025). Association between the big five personalities and academic performance among grade 12 students at international high school in Thailand. *Scientific Reports, 15*, 16484.
**Modality**: **Face-to-face** (international high school physical classes) | **Country**: Thailand | **Era**: post-COVID

### Sample
- N=203 (response 81%), Grade 12, M age 17.19, 66% female
- BFI 44 English; α 未報告

### Effect sizes (group means; multivariable OR Table 3)
- C のみ独立予測（aOR=1.89 [1.12–3.20]）, AA M=3.31 vs AN M=3.12 (p=.02)
- 他 4 特性 ns
- 関連: Female gender (aOR=2.66), high BMI 負, physical activity 負

### Status: ❌ **EXCLUDE** — オンライン要素なし（K-12 face-to-face）
### RoB: **6/8**

---

## A-11. Cheng, Chang, Quilantan-Garza & Gutierrez (2023)

**Citation**: Cheng, S.-L., Chang, J.-C., Quilantan-Garza, K., & Gutierrez, M. L. (2023). Conscientiousness, prior experience, achievement emotions and academic procrastination in online learning environments. *British Journal of Educational Technology, 54*(4), 898–923.
**Modality**: **Fully online** (COVID lockdown Taiwan) | **Country**: Taiwan | **Era**: COVID 2021

### Sample
- N=746 (response 65%), mixed (68% secondary + 32% postsecondary)
- 53% female, M age 18.02
- BFI 9 C items only（C only — proactive ω=.92, inhibitive ω=.86）
- Procrastination: Yockey 2016 5 items, ω=.96

### Effect sizes（latent SEM, Table 2）

| Trait | r × Procrastination | r × Enjoyment | r × Anxiety | r × Hopelessness | r × Boredom |
|-------|---------------------|---------------|-------------|------------------|-------------|
| C-proactive | -.24 | +.24 | — | — | — |
| C-inhibitive | **-.39** | — | -.27 | -.26 | -.30 |

O/E/A/N: **未測定**

### Key findings
- C-inhibitive facet が procrastination と最強関連
- 負の感情が procrastination の主要近接因
- Prior online experience は indirect via emotions

### Status: **Include with caveats** — C のみ、procrastination 副次プール
### RoB: **6/8**

---

## A-12. Cohen & Baruth (2017)
**TBD**: Phase 2 で精読

## A-13. Dang et al. (2024) ⚠
**TBD**: Phase 2 で精読

## A-14. Eilam et al. (2009) ❌（対面、除外予定）
**TBD**: Phase 2 で精読（除外根拠の確認）

## A-15. Elvers et al. (2003)
**TBD**: Phase 2 で精読

## A-16. Garzón-Umerenkova et al. (2024) ⚠
**TBD**: Phase 2 で精読

## A-17. Kara et al. (2024)
**TBD**: Phase 2 で精読

## A-18. Keller & Karau (2013)
**TBD**: Phase 2 で精読

## A-19. MacLean (2022) ⚠
**TBD**: Phase 2 で精読

## A-20. Mustafa et al. (2022)
**TBD**: Phase 2 で精読

## A-21. Nakayama et al. (2014)
**TBD**: Phase 2 で精読（Japanese context）

## A-22. Quigley et al. (2022)
**TBD**: Phase 2 で精読

## A-23. Rodrigues et al. (2024)
**TBD**: Phase 2 で精読

## A-24. Tlili et al. (2023)
**TBD**: Phase 2 で精読

## A-25. Tokiwa (2025)
**TBD**: Phase 2 で精読（COI 研究）

## A-26. Wang et al. (2023)
**TBD**: Phase 2 で精読

## A-27. Wu & Yu (2024)
**TBD**: Phase 2 で精読

## A-28. Yu (2021)
**TBD**: Phase 2 で精読

---

# Part D: Cross-cutting Synthesis（Introduction 執筆用）

## D1. 性格 × 学業のベンチマーク効果量（一般的学業）
**TBD**: Phase 1 後に Part A から抽出して構築

## D2. オンライン学習環境の特殊性（理論的根拠）
**TBD**: Phase 2 後に Part C から抽出

## D3. 本メタ分析の novel contribution
**TBD**: 全精読完了後に構築

## D4. 仮説 H1–H5 の文献的根拠
**TBD**: 全精読完了後に構築

## D5. Methodological heterogeneity（抽出時の注意点）
**TBD**: 全精読完了後に構築

---

**最終更新**: Phase 1 開始時点（Introduction 執筆前）
