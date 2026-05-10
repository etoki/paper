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

## C-06. Meyer, Jansen, Hübner & Lüdtke (2023) 🔴 K-12 benchmark

**Citation**: Meyer, J., Jansen, T., Hübner, N., & Lüdtke, O. (2023). Disentangling the association between the Big Five personality traits and student achievement: Meta-analytic evidence on the role of domain specificity and achievement measures. *Educational Psychology Review, 35*(12), 1–34. https://doi.org/10.1007/s10648-023-09736-2

**Institution**: IPN Kiel + University of Tübingen

### Study characteristics
| 項目 | 値 |
|------|------|
| k (samples) | **110 samples, 1,491 effect sizes, 78 studies** |
| Total N | **500,218** |
| Coverage | **K-12 only（tertiary 除外）**, 87% European, 〜2022-08 |
| Analysis model | Random-effects RVE + REML |
| Effect size metric | ρ (Fisher z, measurement error corrected) |

### Overall pooled effect sizes
| Trait | ρ | 95% CI | r_raw |
|-------|---|--------|-------|
| **C** | **0.24** | [0.21, 0.28] | 0.20 |
| **O** | **0.21** | [0.17, 0.25] | 0.17 |
| N | -0.05 | [-0.09, -0.02] | -0.04 |
| A | 0.04 | [0.01, 0.08] | 0.04 |
| E | 0.02 | [-0.02, 0.05] | 0.02 |
| Intelligence | 0.42 | [0.38, 0.47] | 0.36 |

### Domain × Measure moderators 🔴

| Trait | Domain effect | Measure effect |
|-------|--------------|---------------|
| O | **Lang 0.25 > STEM 0.13** (z=-7.80***) | Grade/Test 差 n.s. |
| **C** | 領域差 n.s. (0.22-0.23) | **Grade 0.28 > Test 0.13** (z=5.17***) 🔴 |
| E | Lang 0.06 > STEM -0.02 | — |
| A | STEM tests で A 逆効果 (0.06 vs -0.05) | Domain × measure 相互作用有意 |
| N | STEM 0.09 負 > Lang 0.03 負 | **STEM tests ρ=-0.14 最大負** |

### Key findings
- ρ 値は Poropat 2009/Mammadov 2022 より**大きい**（O .21 vs .10-.13; C .24 vs .19-.20）— 領域/測定を分離したため
- **PASH (Personality-Achievement Saturation Hypothesis)** が理論フレーム: Conscientiousness は grading (teacher observation) で強化される
- Openness は verbal orientation → language-specific
- Neuroticism は主に STEM testing で有害（anxiety）
- Modality (online/FtF) **未検討** — 本研究の gap を明確化

### 🔴 含意（本メタ分析への）
- **Benchmark ρ (K-12 FtF)**: O=.21, C=.24, E=.02, A=.04, N=-.05
- **PASH framework** は理論的アンカーとして強力: オンライン環境は standardized auto-graded → **C 効果が減弱する**予測可能
- **我々の independent contribution**: (1) modality moderator, (2) tertiary+adult 含む, (3) post-COVID corpus

### RoB（JBI 8-item）: **7/8**

---

## C-07. Chen, Cheung & Zeng (2025) 🔴 **直接 comparator**

**Citation**: Chen, S., Cheung, A. C. K., & Zeng, Z. (2025). Big Five personality traits and university students' academic performance: A meta-analysis. *Personality and Individual Differences, 240*, 113163. https://doi.org/10.1016/j.paid.2025.113163

**Institution**: Faculty of Education, Chinese University of Hong Kong

### Study characteristics
| 項目 | 値 |
|------|------|
| k (articles/correlations) | **84 articles, 370 independent correlations** |
| Total N | **46,729** |
| Coverage | 1995–2024 (search 2024-08), 26 countries |
| Inclusion | University students only, N≥200 |
| Analysis model | Random-effects (CMA v3), Fisher z |
| Effect size metric | Pearson r |

### Overall pooled effect sizes (Table 2)
| Trait | k | r | 95% CI | I² |
|-------|---|---|--------|-----|
| **C** | 81 | **0.206** | [0.170, 0.241] | 93.0% |
| A | 72 | 0.082 | [0.050, 0.113] | 89.5% |
| O | 72 | 0.081 | [0.055, 0.108] | 84.2% |
| N | 73 | -0.029 (ns) | [-0.065, 0.008] | 92.0% |
| E | 72 | -0.009 (ns) | [-0.036, 0.017] | 84.9% |

Egger's test: no publication bias

### Moderator findings 🔴

| Moderator | 重要発見 |
|-----------|---------|
| **Culture (Hofstede)** | A and O 強い in collectivistic (A: 0.126 vs 0.059); **E negative in individualistic (-0.036), null in collectivistic** |
| Measurement tool | C stable (0.190-0.219); BFI 最高 heterogeneity |
| **Academic major** | **E positive only in Education (0.046)**; A/O in Psychology |
| **Year of study** | C 高学年で強い (0.210 vs 0.135 低学年); **N positive in 1-2 年生 (0.042)** |
| Gender composition | 女性比率 ↑ → E効果 減（β=-0.235*）; 他の特性影響なし |

**重要**: **学習モダリティ (online/blended/FtF) moderator 未検討** — 本メタ分析の核心的 gap を Chen 自身が作り出している

### Comparison with prior meta-analyses
| Source | A | C | N | E | O |
|--------|---|---|---|---|---|
| Poropat 2009 | .07 | .22 | .02 | -.01 | .12 |
| Vedel 2014 | .08 | .26 | .00 | -.01 | .07 |
| McAbee 2013 | .08 | .23 | -.03 | -.03 | .08 |
| Mammadov 2022 | .09 | .27 | -.02 | .01 | .16 |
| Meyer 2023 (K-12) | .04 | .24 | -.05 | .02 | .21 |
| **Chen 2025** | **.082** | **.206** | **-.029** | **-.009** | **.081** |

Chen 2025 の C は先行より**やや低め** — N≥200 + 大学生限定による厳格化

### 🔴 **Duplicate check with our 28 studies** (critical)

Chen 2025 の 84-study list は Supplementary のみ（Elsevier 要別途 DL）。本文で引用された included studies:
- Wolfe & Johnson (1995) earliest
- Novikova & Vorobyeva (2017)
- Zhang & Wang (2023)

**候補マッチ度高** (Chen に含まれる可能性): A-02 Alkış, A-06 Sahinidis, A-07 Cohen&Baruth, A-15 Elvers, A-18 Bhagat, A-21 Nakayama, A-26 Wang
**含まれていない可能性高**: A-13 Dang 2025, A-17 Kara 2024 後半, A-25 Tokiwa 2025（検索終了 2024-08 後）
**不明 (要確認)**: オンライン特化研究 (A-04 Audet, A-24 Tlili, A-28 Yu)

→ **要アクション**: Elsevier 補足資料を DL して 84-study list と A-01〜A-28 の重複洗い出し

### 🔴 含意（本メタ分析への）
1. **Benchmark value**: Chen 2025 C r=.206 が最新かつ**大学生特化・N≥200 ベンチマーク**。我々の online pooled r とこの値で差分検定
2. **Novel contribution gap CONFIRMED**: Chen はモダリティ未検討 → 本研究の **modality × culture × year 3-way interaction** が真に新規
3. **E の文化依存 + N の学年依存**: 本研究で online × culture × year の interaction を設計できれば新規貢献
4. **Gender composition moderator**: 連続変数として meta-regression 可能と実証（我々の corpus は多くが female-majority → 要採用）
5. **N≥200 基準は採用せず**（small-N online 研究が除外されすぎる） — sensitivity のみ

### RoB（JBI 8-item）: **7.5/8**

---

## C-08. Zell & Lesick (2021) Umbrella review

**Citation**: Zell, E., & Lesick, T. L. (2021). Big five personality traits and performance: A quantitative synthesis of 50+ meta-analyses. *Journal of Personality, 90*(4), 559–573. https://doi.org/10.1111/jopy.12683

**Institution**: UNC Greensboro

### Study characteristics
| 項目 | 値 |
|------|------|
| m (meta-analyses) | **54 meta-analyses（second-order）** |
| k (primary effects) | 1,539–2,028 per trait |
| Total N | 406,696–554,778 |
| Coverage | Job (m=32), **Academic (m=7)**, Other (m=15) |
| Analysis model | Unweighted 2nd-order + sensitivity weights |

### Overall pooled effect sizes (Table 1, m=54)
| Trait | ρ | 95% CI |
|-------|---|--------|
| **C** | **0.19** | [0.16, 0.21] |
| O | 0.13 | [0.10, 0.15] |
| E | 0.10 | [0.08, 0.13] |
| A | 0.10 | [0.07, 0.13] |
| N | -0.12 | [-0.15, -0.10] |

### By performance category
| Trait | Academic (m=7) | Job (m=32) | Other (m=15) |
|-------|---------------|-----------|-------------|
| **C** | **0.28** ⭐ | 0.20 | 0.12 |
| O | 0.14 | 0.11 | 0.16 |
| A | 0.07 | 0.11 | 0.09 |
| E | **-0.01** ⭐ | 0.14 | 0.08 |
| N | **-0.03** ⭐ | -0.15 | -0.10 |

**University subset** (m=5): C=.25, O=.09, A=.07, N=-.01, E=-.03

### Key findings
- 54 meta-analyses で **C-GPA の堅牢性確認** (ρ=.19 全体、.28 academic)
- **Academic vs Job で profile 大きく異なる**: Academic は C-dominant, E/N ほぼ null; Job は balanced
- 全 Big Five で SD across metas は .01–.04（極めて replicable）→ **差 > .04 でないと novel ではない**
- Modality (online/FtF) **未検討** — 2021 搜索で post-COVID 研究包含せず

### Prior meta-analyses synthesized (academic subset)
Poropat (2009, 2014), McAbee & Oswald (2013), O'Connor & Paunonen (2007), Richardson et al. (2012), Trapmann et al. (2007), Vedel (2014)

### 🔴 含意（本メタ分析への）
1. **University benchmark**: C=.25, O=.09 — Chen 2025 (.206, .081) より大きめだが overlap する
2. **Replicability bar**: 我々の online-specific ρ が FtF benchmark と **.04 超** で異なれば modality effect として valid claim
3. Zell & Lesick は **2021 年 1 月で検索終了 → post-COVID 研究包含せず** → 本研究の post-COVID online literature 統合が直接的な contribution

### RoB（JBI 8-item）: **6–7/8**

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

## A-12. Baruth & Cohen (2022/2023) 🔴 **lit review 著者・年次混乱 — 訂正必要**

**重要**: lit review では "Cohen & Baruth 2017" だが、実 PDF は **Baruth & Cohen (2022/2023)** — 著者順逆 + 別論文（Cohen & Baruth 2017 は A-07 別 PDF として既登録）。

**Citation**: Baruth, O., & Cohen, A. (2023). Personality and satisfaction with online courses: The relation between the Big Five personality traits and satisfaction with online learning activities. *Education and Information Technologies, 28*, 879–904.
**Modality**: Fully online | **Country**: Israel (Tel Aviv U.) | **Era**: COVID-era 2022

### Sample
- N=108, undergraduate, single online course
- BFI-44 Hebrew; α: E=.80, N=.81, A=.68, C=.73, O=.76

### Effect sizes（Table 3 Spearman ρ × general satisfaction）

| Trait | ρ × Gen. satisfaction |
|-------|----------------------|
| **N** | **-.542** (p<.001) ⭐ 最強負 |
| A | .458 (p<.001) |
| C | .335 (p<.001) |
| E | .324 (p<.001) |
| O | .294 (p<.01) |

Discussion-group satisfaction: E=.44, N=-.47

### Key findings
- N が全 satisfaction subscales で強い負相関
- A/C/O は正、E も正（A-07 と異なる pattern）
- Neurotic cluster で uniformly low satisfaction

### Status: **Include for satisfaction pool only** （achievement なし）
### ⚠ **Authorship correction needed in literature_review.md**（A-12 = Baruth & Cohen 2022/2023, NOT Cohen & Baruth 2017）
### RoB: **4/8**

---

## A-13. Dang, Du, Niu & Xu (2025) ⚠ **モダリティ要再評価**

**重要**: lit review では Dang 2024 だが、実は **Dang, Du, Niu, Xu (2025)** — Wu & Yu との混同なし。**ただし online-specific ではない**（一般 learning engagement scale）。

**Citation**: Dang, T., Du, W., Niu, M., & Xu, Z. (2025). The effects of personality traits on learning engagement among college students: The mediating role of emotion regulation. *Frontiers in Psychology, 15*, 1476437.
**Modality**: 一般学習（online-specific でない、Questionnaire Star 配信のみ online） | **Country**: China (Shandong) | **Era**: post-COVID 2024

### Sample
- N=235 (response 97%), undergraduate, 73% female, M age 21
- Discipline: Education 68%, Science 14%, etc.
- NEO-FFI Chinese 60 items（item-analysis で削減）, α=.90 total

### Effect sizes（Table 3 r matrix, N=235）

| Trait | r × LE total | Behavioral | Affective | Cognitive |
|-------|--------------|-----------|-----------|-----------|
| **C** | **+.438** ⭐ | .389 | .371 | .314 |
| E | +.309 | .340 | .301 | .123 (ns) |
| O | +.301 | .265 | .312 | .160 |
| A | +.247 | .282 | .222 | .108 (ns) |
| N | -.037 (ns) | -.042 | -.091 | .042 (ns) |

(全特性 p<.01 を除き Big Five total = +.480)

ER mediation = 4.79%（小）

### Key findings
- C が engagement の最強予測（中国学生でも一般 pattern と一致）
- N は負方向だが ns（中国 sample で典型的）
- Emotion Regulation の媒介効果は小

### Status: **Conditional include** — engagement プールで含めるが、modality moderator 分析時 "online-specific 不明確" として感度分析で除外
### RoB: **5/8**

---

## A-14. Eilam et al. (2009) ❌ **EXCLUDE**

**Status**: 対面・science achievement (Israel Grade 8, N=52)。オンライン要素なし。
**lit review 確認**: 既に excluded として記録済み。本研究のスコープ外。
**Use**: SRL theory parameter として narrative reference のみ

---

## A-15. Elvers, Polzella & Graetz (2003)

**Citation**: Elvers, G. C., Polzella, D. J., & Graetz, K. (2003). Procrastination in online courses: Performance and attitudinal differences. *Teaching of Psychology, 30*(2), 159–162.
**Modality**: **Random assignment** (online vs lecture, same Intro Psych) | **Country**: USA (U. Dayton) | **Era**: pre-COVID

### Sample
- N=47 (22 online + 25 lecture), 7 dropout（attrition 13%）
- Intro Psych undergraduate, M age 18.6, 66% female
- **NEO-FFI** (Big Five 確認, lit review caveat 解決) — only C and N reported

### Effect sizes（text）

| Setting | C × Procrastination | N × Procrastination |
|---------|--------------------|--------------------|
| All (N=44) | .27 (p=.07, trend) | .02 (ns) |
| **Online (N=21)** | **.41** (p=.06, trend) | **-.38** (p=.09, trend negative) |
| Lecture (N=23) | .20 (ns) | .27 (ns) |

Procrastination × Exam total: Online r=.58 (p=.01); Lecture r=.14 (ns)

### Key findings
- Online のみで procrastination → exam の有意関係
- C と N が online procrastination 予測（trend level、N small ゆえ低 power）
- O/E/A 未報告

### Status: **Include with strong caveat** — C/N only, very small N, randomized design は強み
### RoB: **6/8**

---

## A-16. Hidalgo-Fuentes et al. (2024) ❌ **EXCLUDE — lit review 著者誤認 + face-to-face**

**重要**: lit review では "Garzón-Umerenkova et al. 2024" だが、実 PDF は **Hidalgo-Fuentes, Martínez-Álvarez, Llamas-Salguero, Pineda-Zelaya, Merino-Soto, & Chans (2024)**。

**Citation**: Hidalgo-Fuentes, S., et al. (2024). The role of big five traits and self-esteem on academic procrastination in Honduran and Spanish university students: A cross-cultural study. *Heliyon, 10*, e36172.
**Modality**: **Face-to-face** (in-class Google Forms; not online learning context) | **Country**: Honduras + Spain | **Era**: post-COVID

### Sample
- N=457 (237 Honduran + 220 Spanish), 70% female, M age 22.01
- BFI-2-S Spanish; ω: A=.65, C=.76, ES=.64, X=.64, O=.56（O/A 低）

### Effect sizes（rb percentage bend correlations × procrastination）

| Trait | rb (Honduran) | rb (Spanish) |
|-------|--------------|-------------|
| C | -.58 | -.63 ⭐ 最強 |
| O | -.24 | -.17 |
| E | -.30 | -.25 |
| A | -.34 | -.24 |
| N | +.32 | -.02 (ns, **cross-cultural diff**) |

### Status: ❌ **EXCLUDE** — オンライン学習文脈なし、procrastination outcome
### ⚠ **Authorship correction needed in literature_review.md**
### RoB: **5/8**

---

## A-17. Kara, Ergulec & Eren (2024)

**Citation**: Kara, A., Ergulec, F., & Eren, E. (2024). The mediating role of self-regulated online learning behaviors: Exploring the impact of personality traits on student engagement. *Education and Information Technologies, 29*, 23517–23546.
**Modality**: Mixed (synchronous + async, 14週 IT course) | **Country**: Turkey | **Era**: post-COVID

### Sample
- N=437, undergraduate (mixed years), 76% female
- Discipline: education sciences, nursing, social sciences
- BFI 44 Turkish; α: A=.63, C=.73, E=.76, N=.66, O=.78（A/N 低）

### Effect sizes (Table 2 r × engagement subscales)

| Trait | r × Behavioral | r × Cognitive | r × Affective |
|-------|---------------|---------------|---------------|
| **C** | **.49** | **.41** | .21 |
| O | .21 | .34 | .09 (ns) |
| A | .31 | .23 | .13 |
| E | .18 | .17 | .11 |
| N | -.20 | -.08 (ns) | -.19 |

(全て p<.01 を除き示記)
SEM: Personality → Engagement direct β=.45 (p<.001); indirect via SRL β=.23; total 67% variance explained

### Key findings
- C と A が engagement を最広範に予測
- SRL が部分媒介（personality → SRL → engagement）

### Status: **Conditional include** — engagement プールで含める
### RoB: **6/8**

---

## A-18. Bhagat, Wu & Chang (2019) 🔴 **lit review 著者誤認 — 訂正必要**

**重要**: lit review では "Keller & Karau 2013" だが、実 PDF は **Bhagat, Wu & Chang (2019)**（Keller & Karau は A-08 別 PDF）。

**Citation**: Bhagat, K. K., Wu, L. Y., & Chang, C. Y. (2019). The impact of personality on students' perceptions towards online learning. *Australasian Journal of Educational Technology, 35*(4), 98–108.
**Modality**: Fully online (retrospective perception) | **Country**: Taiwan | **Era**: pre-COVID

### Sample
- N=208, mixed (46% UG, 46% MA, 8% PhD), 54% female, M age 25
- Mini-IPIP 20 items; α .75–.80

### Effect sizes（Table 4 hierarchical regression β）

| Trait | β (Instr.Char) | β (Social) | β (Design) | β (Trust) |
|-------|--------------|-----------|-----------|-----------|
| O/Intellect | .265 | **.383** | .253 | .169 |
| **A** | **.387** ⭐ | .351 | .176 | .056 |
| C | .304 | .192 | .255 | .052 |
| N | .165 | -.042 | .095 | -.175 |
| E | -.07 | -.166 | .148 | -.134 |

R² range: .061–.291

### Key findings
- C と O/Intellect が最 stable な正予測
- A が instructor characteristics 最強
- N は trust と負（複雑 pattern）

### Status: **Exclude for primary** — perception/preference outcome only
### ⚠ **Authorship correction needed in literature_review.md**（A-18 = Bhagat, NOT Keller）
### RoB: **5/8**

---

## A-19. MacLean (2022) ⚠ HEXACO 確認

**Citation**: MacLean, K. A. (2022). *Endorsed, or just enforced? Personality and preferences for online learning during COVID-19* [Master's thesis, University of Calgary].
**Modality**: COVID オンライン経験者の preference survey | **Country**: Canada | **Era**: COVID 2022

### Sample
- N=465 (delete 21 low-quality), undergraduate, 79% women, M age 20
- Discipline: heavily Psychology
- **HEXACO-PI-R 60 items** (Ashton & Lee 2009) — 6 因子（H 含む）

### Effect sizes（r × online learning preference）

| Trait | r × pref | β |
|-------|---------|----|
| H (Honesty-Humility) | -.15 (p<.001) | -.16 |
| **X (Extraversion)** | -.17 (p<.001) ⭐ | -.19 |
| C | -.13 (p=.003) | -.06 (ns in regression) |
| E (Emotionality) | -.08 (p=.069) | -.09 |
| A | ns | .06 |
| O | ns | .06 |

R²adj=.058

### Key findings
- H と X 高い学生 → 対面 prefer
- 全体として student は対面 > online prefer
- 自己報告 academic performance も収集だが personality との直接関連未報告

### Status: **Include for HEXACO sub-analysis** — preference outcome、achievement なし
### RoB: **5/8**（unpublished thesis）

---

## A-20. Mustafa, Qiao, Yan, Anwar, Hao & Rana (2022) ⚠ **lit review 国名要訂正**

**Citation**: Mustafa, S., Qiao, Y., Yan, X., Anwar, A., Hao, T., & Rana, S. (2022). Digital students' satisfaction with and intention to use online teaching modes, role of Big Five personality traits. *Frontiers in Psychology, 13*, 956281.
**Modality**: Online teaching modes (COVID 4 中国大学) | **Country**: **China**（lit review 「Pakistan/international」は **誤り**） | **Era**: COVID 2022

### Sample
- N=718 (response 90%), 50% UG + 50% Grad, 46% female
- Discipline: Economics 30%, CS 25%, Math 27%, Education 19%
- Big Five 5–7 items per trait, Chinese; α: .84–.91, AVE>.5

### Effect sizes (PLS-SEM β, Table 4)

| Trait | β × Satisfaction | β × Adoption Intention |
|-------|-----------------|----------------------|
| **A** | **.383** | **.711** ⭐ 最強 |
| O | .215 | -.090 |
| C | .173 | .006 (ns) |
| N | .163 | .218 |
| E | -.114 | .177 |

R²: SAT=.49, AI=.70

### Key findings
- A が SAT/AI 両方で最強（中国 cooperative learning 文化と整合）
- C は SAT 正だが AI に効果なし
- E は SAT 負 / AI 正（パラドキシカル）
- N は両方で正（中国 sample で典型）

### Status: **Include for satisfaction pool only** — achievement なし
### ⚠ **国名訂正必要**（Pakistan/international → China）
### RoB: **6/8**

---

## A-21. Nakayama, Mutsuura & Yamamoto (2014) 🔴 **Japan**

**Citation**: Nakayama, M., Mutsuura, K., & Yamamoto, H. (2014). Impact of learner's characteristics and learning behaviour on learning performance during a fully online course. *Electronic Journal of e-Learning, 12*(4), 394–408.
**Modality**: **Fully online** (LMS 15週、weekly 対面 test のみ) | **Country**: Japan | **Era**: pre-COVID

### Sample
- **N=53**（lit review 「?」→ 確定）, undergraduate Economics, single course
- Gender 未報告, age 未報告（学部 3-4 年生想定）
- IPIP Big Five; α 未明示

### Effect sizes
- 本文 Table 3 で各特性 × note-taking factor の相関のみ（"strongly correlate" 等の表現）
- **Big Five → test scores 直接効果は n.s.**（SEM で indirect via note-taking のみ）
- 抽出可能数値: Note assessment × Online tests r=.31, × Weekly r=.58, × Final r=.46
- A, C, O が note-taking factor と相関（数値画像 OCR 必要）

### Key findings
- Personality は note-taking 経由で間接的に test score に影響
- direct trait → test 効果は消失

### Status: **Ambiguous** — 効果量直接抽出困難、N small
### RoB: **4/8**

---

## A-22. Quigley, Bradley, Playfoot & Harrad (2022)

**Citation**: Quigley, M., Bradley, A., Playfoot, D., & Harrad, R. (2022). Personality traits and stress perception as predictors of students' online engagement during the COVID-19 pandemic. *Personality and Individual Differences, 194*, 111645.
**Modality**: Fully online (UK lockdown, mix synchronous + async pre-recorded) | **Country**: UK (Swansea U.) | **Era**: COVID Jan-Mar 2021

### Sample
- N=301 (lit review 「N=301 confirmed」), 1st-year Psychology, 76% female, M age 19.79
- BFI 44; α: O=.71, C=.82, E=.88, A=.76, N=.85

### Effect sizes (Table 1 r × OSES subscales)

| Trait | r × Skills | r × Emotional | r × Participation | r × **Performance** (proxy) |
|-------|-----------|---------------|-------------------|---------------------------|
| **C** | **.61*** ⭐ | .32*** | .18** | **.26*** |
| O | -.02 | .33*** | .07 | .00 |
| E | .03 | .13* | **.36*** | .14* |
| A | .27*** | .15** | .20*** | .15** |
| N | .08 | -.02 | -.14* | -.02 |

Performance subscale (2 items, 自己評価): C OR=1.07, E OR=1.04, N OR=1.04 (有意)

### Key findings
- C が全 engagement form で正
- E が participation/performance で予測（通常と逆方向）
- N が skills/emotional/performance で予測（suppressor 警告）

### Status: **Include with caveat** — Performance は self-rated 2 items 弱 proxy
### RoB: **6/8**

---

## A-23. Rodrigues, Rose & Hewig (2024) 🔴 **strongest GPA outcome**

**Citation**: Rodrigues, J., Rose, R., & Hewig, J. (2024). The relation of Big Five personality traits on academic performance, well-being and home study satisfaction in Corona times. *European Journal of Investigation in Health, Psychology and Education, 14*(2), 368–384.
**Modality**: Fully online (home study, COVID 3rd semester) | **Country**: Germany (multi-institutional) | **Era**: COVID 2021-2022

### Sample
- **N=287 main, N=260 for GPA** (after exclusion)
- 77% female, M age 22.68
- BFI-S 15 items, 7-pt Likert
- **Preregistered on OSF** ⭐

### Effect sizes（Pearson r × GPA, Holm correction）

**注**: GPA は German system（**1=最高, 6=最低**）→ 負相関 = better performance

| Trait | r × GPA | p (Holm) | 解釈 |
|-------|---------|----------|------|
| **C** | **-.228** (p<.01) ⭐ | better GPA |
| E | .025 (=1, ns) | no effect |
| A/O | n.s. | — |
| N (via hopelessness) | r=.142 (p=.022) | indirect negative |

Well-being / satisfaction:
- N × negative affect: r=+.522
- N × general satisfaction: r=-.388
- C × positive affect: r=.309
- C × satisfaction: r=.206

### Key findings
- C が COVID home study で GPA 予測 (r=-.228, sign convention 注意)
- E は予期に反し GPA に効果なし
- N は well-being/satisfaction で強い負影響
- Effect of C on GPA は non-pandemic norm (Kappe & Van Der Flier 2012 r=.47) より **dampened** (z=2.609, p=.005)

### Status: ✅ **Include — strong GPA candidate**, preregistered
### RoB: **7/8**（最高品質の online achievement study）

---

## A-24. Tlili et al. (2023) ❌ 効果量抽出不能

**Citation**: Tlili, A., Sun, T., Denden, M., Kinshuk, Graf, S., Fei, C., & Wang, H. (2023). Impact of personality traits on learners' navigational behavior patterns in an online course: A lag sequential analysis approach. *Frontiers in Psychology, 14*, 1071985.
**Modality**: Fully online (Moodle, 3-month BS course) | **Country**: Tunisia | **Era**: COVID 2022

### Sample
- N=65 (drop 27 from 92), undergraduate CS, 66% male
- BFI 44, dichotomized at z=0
- **Outcome**: Navigational behavior patterns（NOT achievement）, 15,869 log events, lag sequential analysis

### Effect sizes
- **No Pearson r or β extractable** — only behavior-transition z-scores via Yule's Q
- Agreeableness EXCLUDED from analysis（unbalanced split）

### Key findings (narrative)
- High-E learners: course→achievement→peer comparison transitions
- High-C learners: avoid public forum posts
- High-N learners: anxiety-driven achievement checking
- High-O learners: heavy discussion engagement

### Status: ❌ **EXCLUDE from primary** — 効果量抽出不能（process data only）
### RoB: **4/8**

---

## A-25. Tokiwa (2025) ⚠ COI 研究

**Citation**: Tokiwa, E. (2025). Who excels in online learning in Japan? *Frontiers in Psychology, 16*, Article 1420996. https://doi.org/10.3389/fpsyg.2025.1420996 （著者の先行論文、CC BY、本メタ分析の COI 対象）
**Modality**: Online async | **Country**: Japan | **Era**: post-COVID

### Sample
- N=103, Japanese high school
- BFI-2-J
- 相関行列既知（COI 公開済み）

### Status: **Include with COI sensitivity analysis** — 自著論文として透明開示
### RoB: **6/8**（推定）

---

## A-26. Wang, Wang & Li (2023) 🔴 K-12

**Citation**: Wang, P., Wang, F., & Li, Z. (2023). Exploring the ecosystem of K-12 online learning: An empirical study of impact mechanisms in the post-pandemic era. *Frontiers in Psychology, 14*, 1241477.
**Modality**: Fully online K-12 | **Country**: China (Shenzhen) | **Era**: post-COVID 2023

### Sample
- N=1,625（132 classes; 791 elementary 49% + 445 middle 27% + 389 high 24%）
- 55% female; public 32%, private 55%, international 14%
- Big Five Scale (Meng et al. 2021), 7-pt Likert; α total = .901
- Outcome: 自己報告 academic achievement (7-pt Likert), α=.866

### Effect sizes（SEM standardized path coefficients）

| Path | β | p | Notes |
|------|---|---|-------|
| Big Five total ↔ Achievement (r) | **+.250** | <.001 | bivariate |
| Big Five → Achievement (direct) | **-.173** | .089 | **n.s.** — full mediation via Engagement |
| Big Five → Engagement | **+.779** | <.001 | very strong |
| Engagement → Achievement | +.478 | <.001 | |
| **C → Engagement** | **+.322** ⭐ | <.001 | 最大 |
| O → Engagement | +.253 | <.001 | |
| ES → Engagement | +.169 | <.001 | |
| A → Engagement | +.112 | <.001 | |
| **E → Engagement** | **-.058** | <.05 | 仮説と逆 |

Indirect effects (e.g., FI→C→OLE→AA = .048***, FI→O→OLE→AA = .041***)

### Key findings
- Big Five → Achievement は full mediation via engagement
- **C が engagement の最強予測 (β=.322)**, O 次点
- ES の重要性が online K-12 で相対的に高い
- E が **負** (仮説と逆) — online で外向者は不利

### Status: ✅ **Include**（K-12 online 稀少、効果量抽出可）
### RoB: **6/8**（self-report achievement, cross-sectional）

---

## A-27. Wu & Yu (2024) ⚠ **PDF 未確認 — A-13 と異なる別論文**

**Note**: lit review にある A-27 (Wu & Yu 2024 N=1,004) は本 PDF コレクション内に物理的存在せず（A-13 の PDF が当初混同された）。本論文は A-13 Dang et al. 2025 の本文中で引用されている別の論文。**Wu & Yu 2024 PDF を別途取得すべき**。

**Citation (lit review より)**: Wu, R., & Yu, Z. (2024). Relationship between university students' personalities and e-learning engagement mediated by achievement emotions and adaptability. *Education and Information Technologies, 29*, 10821–10850.

### Status: ⚠ **PDF 未取得 — Phase 2 検索で再入手必要**

---

## A-29. Bahçekapılı & Karaman (2020) 🔴 GPA 完備 — 本研究で最も投入しやすい新規

**Citation**: Bahçekapılı, E., & Karaman, S. (2020). A path analysis of five-factor personality traits, self-efficacy, academic locus of control and academic achievement among online students. *Knowledge Management & E-Learning, 12*(2), 191–208. https://doi.org/10.34105/j.kmel.2020.12.010
**Modality**: Fully online (synchronous live + online midterm + paper final) | **Country**: Turkey (2 universities distance-ed) | **Era**: pre-COVID (2014-2015)

### Sample
- N=525, UG distance-ed, M age 30.9 (range 19-59), 38% female
- Discipline: mixed distance-ed programs
- BFI-44 Turkish (Sümer 2005); α range .56-.75（A 低め）

### Effect sizes（Table 3 bivariate r × GPA）🔴 全 Big Five 完備

| Trait | r × GPA | p | N |
|-------|---------|---|---|
| **C** | **.068** | ns | 525 |
| **O** | **.070** | ns | 525 |
| E | .027 | ns | 525 |
| A | -.013 | ns | 525 |
| **N** | **-.072** | ns | 525 |
| Self-efficacy | .136 | <.01 | 525 |
| External LoC | -.160 | <.01 | 525 |

### SEM indirect effects (Table 5)
- C → GPA: **β = 0.075 (p<.05)** via SE + External LoC
- O → GPA: β = 0.044 (p<.05) via SE
- N → GPA: β = -0.035 (p<.05) via SE + External LoC
- E, A: ns

### Key findings
- Big Five の direct 効果すべて ns → すべて **mediated（self-efficacy + external LoC 経由）**
- Online では trait 直接効果が弱く、self-regulatory mediator 経由が主要
- R² GPA = 4.4% のみ（personality 説明力小）

### Status: ✅ **INCLUDE** — ゼロ次 r 全 Big Five × GPA 完備、Peterson-Brown 変換不要
### RoB: **6/8**

---

## A-30. Kaspar, Burtniak & Rüth (2023) Germany BFI-S

**Citation**: Kaspar, K., Burtniak, K., & Rüth, M. (2023). Online learning during the Covid-19 pandemic: How university students' perceptions, engagement, and performance are related to their personal characteristics. *Current Psychology*. https://doi.org/10.1007/s12144-023-04403-9
**Modality**: Fully online (emergency) | **Country**: Germany | **Era**: COVID 2021

### Sample
- N=413 (from 439), UG+Grad, M age 25.47, **86% female**
- Discipline: mostly psychology, community health, medicine
- BFI-S (Rammstedt 2005 / Kovaleva 2013) 21 items; α: E=.84, N=.81, **A=.65 low**, C=.70, O=.71

### Effect sizes（Table 3 multiple regression β × self-rated performance）

| Trait | β | p | N | 備考 |
|-------|---|---|---|------|
| **N** | **.20** | **.003** | 413 | 正方向（suppressor effect — bivariate では負） |
| **C** | **.15** | **.016** | 413 | |
| O | .08 | .106 | 413 | ns |
| E | .05 | .273 | 413 | ns |
| A | -.01 | .755 | 413 | ns |

R² = .29

### Bivariate r（Table 4、PDF レイアウト破損 — 要原論文確認）
- C × performance: 正・有意
- N × performance: 負・有意
- O × performance: 正・有意
- E × performance: 正・弱め
- A × performance: ns

### Key findings
- C と N のみが multivariable predictor（SE, anxiety 統制後）
- Neuroticism が bivariate → multivariable で sign flip（suppressor）
- Personality の role は self-efficacy と self-regulation 統制後に minor
- Age が demographic predictor 最強

### Status: **Include with caveat** — β 抽出可、r は Table 4 要再確認
- **注**: Outcome は self-report composite（objective GPA ではない）
### RoB: **5/8**（self-report outcome、gender 偏り、A α 低め）

---

## A-31. Rivers (2021) Japan objective grade 🔴 Japanese sample

**Citation**: Rivers, D. J. (2021). The role of personality traits and online academic self-efficacy in acceptance, actual use and achievement in Moodle. *Education and Information Technologies, 26*(4), 4353–4378. https://doi.org/10.1007/s10639-021-10478-3
**Modality**: Fully online **asynchronous** (Moodle 15 週) | **Country**: Japan (Future University Hakodate) | **Era**: COVID

### Sample
- N=149 (response 62%), UG sophomore, M age 19.4, **80% male**
- Discipline: information science（単一 discipline）
- **TIPI-J 10 items**（Oshio et al. 2012, 7-point）— α 未報告（TIPI 慣例）

### Effect sizes（Table 3 zero-order r × Course Achievement, objective grade）🔴

| Trait | r × CA | Notes |
|-------|--------|-------|
| **C** | **.144** | ns but positive |
| **E** | **-.173** | **負** — asynchronous で外向性不利 |
| A | .118 | ns |
| ES (→ N × -1) | -.107 | → **N ≈ +.107** |
| O | -.066 | ns |
| OAS | **.211*** | |
| AMU (log data) | **.345*** | 最強予測因 |

### SEM respecified model（Table 5, R² CA = .146）
- **E → CA direct β = -.168 (p<.01)** 🔴 online async で外向性負の direct effect
- C → OAS direct β = .315***（最強）
- A → OAS β = .184*
- C indirect on CA β = .096***（C → OAS → AMU → CA）
- AMU → CA β = **.342***

### Key findings
- **E が online async で direct 負効果**（social cue 欠如）— H5 仮説と整合
- C と A のみが OAS 経由で indirect effect
- Actual Moodle usage (log) が最強 direct predictor — attitude/intent より実使用
- R² CA = 14.6% — personality 説明力中程度

### Status: ✅ **INCLUDE** — ゼロ次 r 全特性 × grade 完備（要 ES→N 符号反転）
- 稀少 Japanese サンプル + objective grade + log data 三重計測
### RoB: **6/8**（objective grade + log data 強、TIPI α なし・N 小・gender 80% male 偏り）

---

## A-37. Zheng & Zheng (2023) ⚠ **AMBIGUOUS/INCLUDE** — 3-era TIPI Big Five

**Citation**: Zheng, Y., & Zheng, S. (2023). Exploring educational impacts among pre, during and post COVID-19 lockdowns from students with different personality traits. *International Journal of Educational Technology in Higher Education, 20*(1), 21. https://doi.org/10.1186/s41239-023-00388-4
**Modality**: Mixed across eras (F2F pre / online during / F2F post) | **Country**: USA (IIT Chicago) | **Era**: **Pre + During + Post COVID** (2018-2022)

### Sample
- N=282 unique graduate students, 386 academic records
- By era: Pre 186/222, During 63/91, Post 67/73
- Graduate ITM (data science specialization), 41% female
- **TIPI** (10-item Big Five inventory, Gosling 2003) — α 未報告（TIPI は低α で有名）

### Outcomes
- Class grade (0-100)
- Late submission rate
- Assignment attempts

### Effect sizes
- 本文記述: "|r| < 0.10 for all traits × outcomes" — **trait-level r 未表示**
- K-means 4 cluster で分析 — r/β ではない
- Era × cluster interaction: Kruskal-Wallis p=0.001 (grades), p=0.009 (attempts)

### Era-specific findings 🔴
- **Pre**: 安定、weak trait effects
- **During**: **grades stable (M=84.01)** — online への適応良好
- **Post**: **grades dropped (M=76.56)** — post-pandemic transition が remote learning 本体より困難
- 高 O/E cluster (C3) が post で late submission 増（social activity 復活）

### Status
- **Ambiguous/Include**: 3-era 内 Big Five × achievement 設計は稀少だが、trait-level r 未公開
- 著者連絡 or r≈0 (ns) として null-effect 扱いで include 可能
- **Era moderator に uniquely valuable** (interaction p=0.001)

### RoB: **4/8**（cluster analysis のみ、small Post N、α 未報告、TIPI 限界）

---

## Excluded new candidates (Big Five PICO 不適合)

### ❌ P-04. Engel et al. (2023) Germany digital studying
- Personality 未測定 (digital competencies + peer interaction のみ)
- N=18,262 nationwide German HE
- **EXCLUDE**

### ❌ P-05. Wang et al. (2025) China K-12 MBTI 🔴 惜しい
- K-12 online 大規模 (N=4,340) — gap filler に期待
- **MBTI** (4 dimension categorical) — Big Five ではない
- Findings: extroverts/intuitives/thinkers/perceivers が online で成績低下（本研究仮説と整合）
- **EXCLUDE** for Big Five pool、narrative comparator として議論で言及可

### ❌ P-06. Chai et al. (2023) China proactive personality
- Proactive Personality のみ (Big Five 未測定)
- N=322 UG, online learning performance r=.53
- Narrative adjacent construct として価値あり（Conscientiousness + Extraversion と overlap）
- **EXCLUDE**

### ❌ P-07. Ma & Lee (2025) China 3-era intention
- Personality 未測定 (TUE framework)
- Outcome = use intention ではなく achievement なし
- **EXCLUDE**

### ❌ P-08. Salem et al. (2024) Oman 3-mode TAM
- Personality 未測定 (TAM constructs)
- Outcome = GPA だが predictor が TAM
- **EXCLUDE**

### ⚠ Summary: 9 candidates から 1 のみ include (A-37 Zheng & Zheng 2023)

残念ながら P-04〜P-09 の 6 本中 5 本が Big Five 不測定で exclude。K-12 gap（A-26 Wang 2023 が現在唯一）は依然として埋まらず、P-05 は MBTI で惜しかった。**Tier 1 primary の yield: 1/6 = 17%**。

---

## A-28. Yu (2021) 🔴 objective MOOC outcome

**Citation**: Yu, Z. (2021). The effects of gender, educational level, and personality on online learning outcomes during the COVID-19 pandemic. *International Journal of Educational Technology in Higher Education, 18*, 14.
**Modality**: Fully online (BLCU MOOCs + Superstar Learning System, 4 ヶ月以上) | **Country**: China | **Era**: COVID 2020-2021

### Sample
- N=1,152（553 UG + 599 Grad）
- 52% female, age 18–25
- Discipline: Linguistics 系（English Reading/Writing/Translation/Lit, 20 courses）
- Big Five Scale (McCrae & Costa 1995), α: E=.75, A=.76, C=.80, N=.78, O=.81
- **Outcome: MOOC platform composite (100 点)**: assignments 20% + sign-in 5% + video 20% + chapter 10% + discussion 10% + tests 35% — **objective**

### Effect sizes（Table 3 linear regression standardized β, N=1,152）

| Trait | β | p | 解釈 |
|-------|---|---|------|
| **A** | **+.442** ⭐ 最強 | <.01 | cooperative learning（言語専攻） |
| **O** | **+.305** | <.01 | 新技術受容 |
| **E** | **-.076** | <.01 | 対面志向の外向者は online 不利 |
| C | +.057 | .007 | 小さいが有意 |
| N | +.037 | .090 | n.s. (marginal +) |

R²=.565, Adj. R²=.563

### Key findings
- A 最強（言語系 cooperative learning と親和）
- O も強い正効果（新技術受容）
- E は **負** (objective outcome で確認)
- C は小（言語系では C プレミアム小、他研究と異なる）
- Gender 差なし、Grad > UG

### Status: ✅ **Include — strong objective outcome candidate**
### RoB: **7/8**（objective MOOC composite が強み）

---

---

# Part D: Cross-cutting Synthesis（Introduction 執筆用）

## D1. 性格 × 学業のベンチマーク効果量（一般学業）

### D1.1 Meta-analyses (FtF/mixed, all non-online-specific)

| Meta-analysis | k | N | C ρ/r | O ρ/r | A ρ/r | E ρ/r | N ρ/r | Population |
|---------------|---|---|-----|-----|-----|-----|-----|------------|
| Poropat (2009) | 138 | 70,926 | **.22** | .12 | .07 | -.01 | .02 | All education |
| McAbee & Oswald (2013) | 57 | 26,382 | **.26** | .08 | .07 | -.03 | -.00 | Tertiary GPA |
| Vedel (2014) | 21 | 17,717 | **.26** | .07 | .08 | -.00 | -.01 | Tertiary |
| Stajkovic et al. (2018) | 5 samples | 875 | **.21** | -.01 | .05 | -.02 | .00 | Tertiary |
| Mammadov (2022) | 267 | 413,074 | **.27** | .16 | .09 | .01 | -.02 | All education |
| **Meyer et al. (2023) NEW** | **110 samples** | **500,218** | **.24** | **.21** | .04 | .02 | **-.05** | **K-12 only** |
| **Chen et al. (2025) NEW** | **84** | **46,729** | **.206** | .081 | .082 | -.009 | -.029 | **University only, N≥200** |
| **Zell & Lesick (2021) umbrella** | **54 meta-analyses** | >500K | **.19 (all), .28 (academic)** | .13 / .14 | .10 / .07 | .10 / -.01 | -.12 / -.03 | Mixed job+academic |
| **Convergent ベンチマーク** | — | — | **.19–.28** | **.07–.21** | **.04–.10** | **-.01 to +.10** | **-.12 to +.02** | — |

### D1.2 新規ベンチマークの特徴

- **Chen 2025** は最も**直接的 comparator**: 大学生のみ、N≥200、2024-08 までカバー、Hofstede 文化 moderator。C r=.206 は過去よりやや低め（N≥200 基準で small-N 除外のため）
- **Meyer 2023** は **K-12 専用** で最大、domain × measure moderator: C は grade で .28 vs test で .13、Openness は language で強い（PASH framework）
- **Zell 2021 umbrella** は academic m=7, university m=5、replicability SD < .04（差 > .04 で valid claim）
- **全 8 メタ分析が modality moderator 未検討** — 本研究の gap を 8 つ独立に confirm

### Mammadov 2022 の region moderator（Asia 強い）
- Asia: C ρ = **.35**（H1 の重要根拠）, O ρ=.29, E ρ=.16, A ρ=.23, N ρ=-.19
- N. America: C ρ=.23, others ~ベンチマーク

### Mammadov 2022 の education level moderator（年齢で減衰）
- Elementary/Middle: O ρ=.40 (大効果), C ρ=.31, E ρ=.15
- Postsecondary: O ρ=.10, C ρ=.26, E ρ=-.01

### Stajkovic 2018 の mediator
- Self-efficacy が C → performance を完全媒介（intrapersonal model）
- SE → Performance β = .24–.33

---

## D2. オンライン学習環境の特殊性（理論的根拠）

オンライン学習が face-to-face と異なる 4 次元（Hunter et al. 2025 narrative review より体系化）:

1. **Self-regulation demands**: Asynchronous 環境で自律的 time/effort 管理が必須 → C プレミアム期待
2. **Social presence**: 対面の社会的合図（教員・peer）欠如 → E の意味が変質（仮説的に negative shift）
3. **Temporal flexibility**: 学習時間の自己決定 → procrastination 風土 → C 重要性増
4. **Technology mediation**: LMS, Zoom, MOOC platform → O（新技術受容）の重要性増

### Hunter et al. (2025) narrative review からの主な発見

- **GPA × 5 traits (5 studies, N=1,542)**:
  - C, O, A, ES が 3/5 で正; N が 3/5 で負; **E は全 5 で GPA に無関連**（H3 強支持）
- **Class format choice**: 外向者は対面 prefer、内向者は online prefer
- **Neurotic students**: online で苦戦、対面で良好

### 本メタ分析の primary studies からの実証的 effect size パターン

| Source | Modality | C | O | E | A | N |
|--------|---------|---|---|---|---|---|
| A-01 Abe 2020 (US, async) Quiz | **.48** | .13 | -.07 | .13 | -.10 | — |
| A-01 Abe 2020 Paper | .37 | **.35** | .03 | .16 | -.02 | — |
| A-02 Alkış 2018 (Turkey online) Grade | **.205** | -.09 | .05 | .09 | .03 | — |
| A-02 Alkış 2018 (Turkey blended) Grade | **.244** | .08 | .00 | .02 | -.01 | — |
| A-15 Elvers 2003 (US online) Procrast | .41† | — | — | — | -.38† | — |
| A-23 Rodrigues 2024 (Germany) GPA | **-.228**¶ | ns | ns | ns | ns | — |
| A-26 Wang 2023 (China K-12) Eng β | **.322***§ | .253 | -.058 | .112 | .169 | — |
| A-28 Yu 2021 (China) MOOC β | .057 | **.305** | **-.076** | **.442** | .037 | — |
| **A-29 Bahçekapılı 2020 (Turkey pre-COVID)** GPA | **.068** ns | .070 ns | .027 ns | -.013 ns | **-.072** ns | — |
| **A-30 Kaspar 2023 (Germany COVID)** self-perf (β) | **.15*** | .08 | .05 | -.01 | **.20*** (suppressor) | — |
| **A-31 Rivers 2021 (Japan async)** CA obj | **.144** | -.066 | **-.173** | .118 | ≈+.107 | — |
| A-37 Zheng 2023 (US 3-era) CA | \|r\|<.10 all | \|r\|<.10 | \|r\|<.10 | \|r\|<.10 | \|r\|<.10 | — |

¶ German GPA: 1=高, 6=低 → 負相関 = better。 †trend (p<.10)。 §β to engagement, indirect to achievement.

### D2.1 新しい primary finding summary

- **A-29 Bahçekapılı (Turkey pre-COVID)**: 全 Big Five × GPA 直接効果は ns → mediated via SE + external LoC が online の pattern
- **A-30 Kaspar (Germany COVID)**: N が suppressor effect で sign flip — covariates 統制時のみ positive
- **A-31 Rivers (Japan async)**: **E × CA = -.173 direct 負効果** → H5 (online で E 負) を強く支持
- **A-37 Zheng (US 3-era)**: 全 trait × grade \|r\|<.10 → 3 era interaction で pattern 変動

### オンライン学習で観察される pattern の特徴

- **C はオンラインでも consistent positive**（traditional benchmark と一致）。ただし magnitude が研究で大きく振れる（.057 〜 .48）→ **moderator 必須**
- **A の重要性がアジアサンプルで顕著**（A-28 Yu β=.442, Mammadov Asia ρ=.23）— 言語系 / cooperative learning と整合
- **E が常に弱負 or null**（Hunter 2025 narrative + A-28 β=-.076 + A-26 β=-.058）→ **対面志向の外向者は online で不利**仮説支持
- **O の効果はタスク依存**（essay/MOOC で正、quiz/grade で null）— task complexity moderator

---

## D3. 本メタ分析の novel contribution（8 benchmark meta-analyses 全てで未検討）

| Gap | 既存 8 meta-analyses | 本研究 |
|-----|---------------------|--------|
| Online-specific pooled ρ | **全 8 本で未検討** | 初の online-specific quantitative pooling |
| Modality moderator (online/blended/MOOC) | 未検証（全モード混合） | 5 modality の subgroup |
| Era moderator (pre/COVID/post COVID) | 未検証（全期間一括） | 3 era subgroup |
| Region moderator (Asia vs West) | Mammadov 2022 Asian C=.35 | Asia subset で再検証 |
| Cultural moderator (Hofstede) | Chen 2025 新規導入 | Online × individualism で追加 |
| Domain × measure (PASH) | Meyer 2023 (K-12 only) | Online × domain × measure の 3-way |
| Risk of bias-weighted synthesis | Hunter 2025 qualitative のみ | JBI 8-item で量的 RoB moderator |
| Self-efficacy / SRL mediator | Stajkovic 2018 (FtF) | online で SE mediation が強化するか検証 |
| Post-COVID corpus | Chen 2025 が 2024-08 まで | 2025 含む最新 |

### Novel contribution の階層

1. **Primary contribution**: Online × Big Five × achievement の初の pooled ρ（8 meta-analyses が全く touch していない gap）
2. **Secondary**: Modality × era × region の 3-way moderator
3. **Theoretical**: PASH hypothesis (Meyer 2023) の online 拡張 — online で auto-graded → C 効果は weakening か amplifying か

### Hunter et al. (2025) との位置関係

Hunter et al. は narrative synthesis のみ（vote-counting + thematic analysis）で **pooled effect sizes は未提供**。本メタ分析が以下を追加:

1. **量的 pooled effect size** per trait
2. **9 moderator** の meta-regression（modality, era, region, education level, outcome type, instrument, year, sample size, RoB）
3. **GRADE 評価**（per trait の confidence rating）
4. **publication bias 検定**（Egger, Peters, p-curve）

---

## D4. 仮説 H1–H5 の文献的根拠

### H1: C is the strongest predictor (online, ρ = .20–.35)
- **Direct evidence**: A-01 (.48), A-02 (.205–.244), A-22 (.61 with skills engagement), A-23 (-.228 GPA), A-26 (.322 to engagement)
- **Benchmark**: Poropat .22, Mammadov .27, McAbee .26, Vedel .26
- **Asian premium**: Mammadov Asia .35 → 日本サンプル予測高め

### H2: O is second-strongest, potentially stronger than FtF
- **Direct evidence**: A-28 Yu .305 (very strong, MOOC composite), A-04 Audet .27 (engagement T2), A-01 Abe paper .35
- **Benchmark**: Poropat .12, Mammadov .16
- **Online specific**: novel technology acceptance + self-directed learning premium が予測根拠
- ⚠ k 不足の可能性（後検出のみ可能）

### H3: A small positive, weaker than FtF
- **Direct evidence**: A-26 .112 (engagement), A-28 .442 (言語系 outlier?), A-08 .25 (career value)
- **Benchmark**: ρ=.07–.09 (FtF)
- ⚠ Asian + cooperative discipline で過大推定リスク

### H4: N negative, stronger in fully online than blended
- **Direct evidence**: A-12 -.542 (satisfaction), A-22 -.14 (participation), A-26 +.169 (engagement, 不一致), Hunter 3/5 in GPA
- **Benchmark**: ρ=-.00–.02 (essentially null in FtF)
- **Online specific**: anxiety + isolation で online で悪化予測

### H5: E null or weak negative (online)
- **Strongest support pattern**:
  - Hunter 2025: 5/5 GPA studies で E ns
  - A-28 Yu β=-.076 (objective)
  - A-26 Wang β=-.058
  - A-22 Quigley E×performance .14 (mixed)
  - Mammadov FtF: ρ=.01 (already null), online でも持続
- **Novel hypothesis**: E が online で **negative direction shift**（対面志向の外向者は不利）

---

## D5. Methodological heterogeneity（抽出時の注意点）

### 効果量抽出の困難さ

- **Direct r で報告**: A-01, A-02, A-08, A-11, A-12, A-13, A-15, A-17, A-22 (~10 studies)
- **β only (要 Peterson-Brown 変換)**: A-03, A-06, A-18, A-20, A-26, A-28
- **Group means / OR only**: A-09, A-10
- **抽出不能**: A-24 (process data), A-21 (indirect via note-taking)

### Modality 異質性

- Fully online async: A-01, A-04, A-05, A-07, A-08, A-11, A-12, A-22, A-23, A-26, A-28
- Synchronous: A-06
- Blended: A-02 (separate analysis)
- MOOC: A-28
- Mixed sync+async: A-17

### Era 分布

- Pre-COVID: A-01, A-02, A-07, A-08, A-15, A-18, A-21
- COVID: A-04, A-05, A-06, A-11, A-22, A-23, A-28
- Post-COVID: A-13, A-17, A-26

### Education level

- K-12: A-26 only
- Undergraduate: 大半
- Graduate: A-12, A-08 (mixed UG+Grad), A-20 (mixed)
- Adult: A-08 (mixed)

### Region

- North America: A-01, A-04, A-05, A-15, A-19
- Europe: A-22 (UK), A-23 (Germany), A-06 (Greece)
- Asia (East): A-11 (Taiwan), A-21 (Japan), A-25 (Japan), A-26 (China), A-27 (China), A-28 (China)
- Asia (South + SE): A-09 (India ❌), A-10 (Thailand ❌), A-08 (mixed Western)
- Middle East: A-03 (Iran), A-12 (Israel), A-07 (Israel), A-17 (Turkey), A-02 (Turkey)
- Africa: A-24 (Tunisia)

### Sample overlap risks

- **A-04 / A-05**: Fall 2020 McGill cohort 重複 — 一方のみ採用
- **A-27 / A-28**: 著者 Yu 共通だが site 異なる（Guizhou vs BLCU）→ 重複なし

### Outcome type 分布

- **Direct GPA / objective grade**: A-01 (quiz/paper), A-02 (course grade), A-23 (GPA), A-28 (MOOC composite) — **only 4 studies**
- **Self-rated achievement**: A-22 (Performance subscale), A-26 (self-report)
- **Procrastination (proxy)**: A-11, A-15, A-16
- **Engagement only**: A-04, A-05, A-13, A-17, A-26 (mediated)
- **Satisfaction only**: A-03, A-06, A-07, A-12, A-20

---

## D6. 著者誤認サマリ（literature_review.md 訂正項目）

| ID | lit review 記載 | 実際 | 訂正区分 |
|----|----------------|------|---------|
| A-06 | Baruth & Cohen (2021) | **Sahinidis & Tsaknis (2021)** | 完全別論文 |
| A-07 | Baruth & Cohen (2023) | **Cohen & Baruth (2017)** | 著者順 + 年次 |
| A-08 | Bhagat et al. (2019) | **Keller & Karau (2013)** | 完全別論文 |
| A-12 | Cohen & Baruth (2017) | **Baruth & Cohen (2022/2023)** | 著者順 + 年次 |
| A-13 | Dang et al. (2024) | **Dang, Du, Niu, Xu (2025)** | 年次 + co-author 詳細 |
| A-16 | Garzón-Umerenkova et al. (2024) | **Hidalgo-Fuentes et al. (2024)** | 完全別論文 |
| A-18 | Keller & Karau (2013) | **Bhagat, Wu & Chang (2019)** | 完全別論文 |
| A-20 | Mustafa et al. (2022) Pakistan | **Mustafa et al. (2022) China** | 国名のみ訂正 |
| A-21 | Nakayama et al. (2014) N=? | **N=53** 確定 | N 値追加 |

---

## D6b. 新規追加後の k 総合評価（2026-04-23 更新）

### Primary studies（既存 A-01..A-28 + 新規 A-29..A-37）

| 包含状態 | Count |
|----------|-------|
| Include（全確定）| A-01, A-02, A-11 (C-only), A-15 (C/N), A-17, A-19 (HEXACO), A-22, A-23, A-26, A-27 (PDF 未), A-28 | 11 |
| 新規 Include | A-29 Bahçekapılı, A-30 Kaspar, A-31 Rivers, A-37 Zheng | **+4** |
| Conditional Include | A-03, A-04, A-05 (overlap), A-06, A-07, A-08, A-12, A-13, A-20, A-21, A-24, A-25 | 12 |
| Exclude (modality) | A-09, A-10, A-14, A-16 | 4 |
| **Total primary** | | **~32（+4 from new search）** |

### Achievement outcome extractable 直接相関

| ID | N | C × achievement | 備考 |
|----|---|----------------|------|
| A-01 Abe | 92 | **.48** (quiz) / .37 (paper) | 最大効果 |
| A-02 Alkış online | 189 | **.205** | Turkey grade |
| A-22 Quigley | 301 | **.26** (perf subscale) | UK, 2-item proxy |
| A-23 Rodrigues | 260 | **-.228** (sign-flipped) | Germany GPA |
| **A-29 Bahçekapılı** | **525** | **.068** (ns, r 完備) | Turkey pre-COVID **largest direct-r study** |
| **A-30 Kaspar** | **413** | **β=.15** | Germany COVID (β only, 要 Table 4 確認) |
| **A-31 Rivers** | **149** | **.144** | Japan objective grade |
| A-15 Elvers | 21 online | .41† (via procrast) | RCT但 small N |
| A-28 Yu | 1152 | β=.057 | China MOOC composite |
| A-37 Zheng | 282 | \|r\|<.10 | US 3-era, cluster-based |

→ **Primary achievement pool: k = 9–10**（閾値 ≥10 達成）

### Pooling 実施可能性

| Analysis | k required | 現状 k |
|----------|-----------|--------|
| Random-effects REML | ≥5 | **9–10** ✅ |
| Tau² 推定（Jackson） | ≥10 | **borderline** |
| Subgroup per level | ≥5 | modality/era で可能 |
| Meta-regression (9 mod) | ≥10 per predictor | ❌ still under |
| Egger publication bias | ≥10 | **borderline** |

### 結論

- **9 moderator 全てを REG 予定通り実施するには k がまだ足りない**
- Primary analysis は可能（random-effects pooling + forest plot）
- Moderator 分析は 2-3 に絞るか、narrative での補完が必要

---

## D7. Introduction 骨子（執筆用 5 段落構造）

### §1. Problem statement
- 教育の online shift（COVID 触媒、post-COVID 持続）
- Big Five × academic performance は最重要 non-cognitive predictor として確立
- しかし online-specific quantitative synthesis 不在

### §2. What is known (FtF benchmarks)
- C consistently strongest (Poropat .22, Mammadov .27, Vedel .26, McAbee .26)
- O second (.07–.16), declines with age
- A small positive (.05–.09)
- E and N essentially null overall
- Asian samples show C amplification (Mammadov .35)

### §3. Why online may differ (theoretical mechanism)
- Self-regulation demands → C premium
- Social presence loss → E sign shift
- Tech mediation → O premium
- Asynchronous → procrastination → C importance

### §4. Existing online evidence (gap)
- Hunter et al. (2025) narrative only
- Primary studies fragmented (k≈25-30, no pooled effect)
- Hypotheses derivable but not confirmed quantitatively

### §5. Present study
- First quantitative meta-analysis of online-specific Big Five × achievement
- 9 pre-registered moderators
- 5 hypotheses (H1-H5) with directional predictions

---

**最終更新**: Phase 2 完了時点（Introduction 執筆 ready）。Part C の各 A-XX entries に詳細な effect size、RoB、author corrections 反映済み。次工程: literature_review.md の著者訂正コミット（D6 表に基づく）→ Introduction draft 開始。
