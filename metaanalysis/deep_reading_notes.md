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

**TBD**: Phase 2 で精読

---

# Part B: Existing Systematic Reviews（online-specific）

## D-01. Gray & DiLoreto (2024 推定) ⚠

**TBD**: Phase 2 で精読

---

# Part C: Primary Studies（本メタ分析対象）

## A-01. Abe (2020)
**TBD**: Phase 2 で精読

## A-02. Alkış & Temizel (2018)
**TBD**: Phase 2 で精読

## A-03. Ashouri et al. (2025)
**TBD**: Phase 2 で精読

## A-04. Audet et al. (2021)
**TBD**: Phase 2 で精読

## A-05. Audet et al. (2023)
**TBD**: Phase 2 で精読

## A-06. Baruth & Cohen (2021)
**TBD**: Phase 2 で精読

## A-07. Baruth & Cohen (2023)
**TBD**: Phase 2 で精読

## A-08. Bhagat et al. (2019)
**TBD**: Phase 2 で精読

## A-09. Bhattacharjee & Ramkumar (2025) ⚠
**TBD**: Phase 2 で精読

## A-10. Boonyapison et al. (2025) ⚠
**TBD**: Phase 2 で精読

## A-11. Cheng et al. (2023)
**TBD**: Phase 2 で精読

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
