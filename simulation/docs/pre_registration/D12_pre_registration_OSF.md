# D12 OSF Pre-Registration — HEXACO 7-Typology Workplace Harassment Microsimulation (Phase 1 + Phase 2)

**Document type**: OSF Standard Pre-Registration draft (internal master version)
**Drafted**: 2026-04-29
**Branch**: `claude/hexaco-harassment-simulation-69jZp`
**Author (corresponding)**: Tokiwa et al.
**Status**: ⏳ DRAFT — to be reviewed, finalized, then registered on OSF **prior to Stage 0 code execution**
**Anchor template**: OSF Standard Pre-Registration (Bowman et al. 2020, https://osf.io/rh8jc) + Nosek et al. 2018 PNAS "preregistration revolution" 9-Challenge framework
**Referenced research plan**: `simulation/docs/notes/research_plan_harassment_typology_simulation.md` (v6/v7, 1,458 行)
**Referenced power analysis**: `simulation/docs/power_analysis/D13_power_analysis.md`
**Referenced literature audit**: `simulation/docs/literature_audit/deep_reading_notes.md` (40 papers deep reading)

---

## 0. 本ドキュメントの位置づけ（meta）

### 0.1 目的

本ドキュメントは、研究計画 v6/v7（Part 8.3 D12）で確定した「Stage 0 コード実装に着手する**前**に OSF 登録する」方針に従い、研究計画から **pre-registration 該当事項のみ抽出して固定する** master draft である。

OSF 提出時には：
1. 本ドキュメントを英訳した英語版を作成
2. OSF Standard Pre-Registration の Web フォームに転記
3. 本ドキュメント（日本語版）を OSF supplementary として PDF で添付
4. 登録日時の DOI を取得し、本ドキュメント Header に記録

### 0.2 Nosek et al. 2018 PNAS 9 Challenges への適合性確認

| Challenge | 本研究の状況 | 対応 |
|---|---|---|
| **C3: Data Are Preexisting** | N=354（harassment）/ N=13,668（clustering）は **既存 IRB 通過済データ**。ただし (a) 7 類型 × gender 14-cell harassment cross-tabulation、(b) 国レベル aggregate prediction、(c) Counterfactual A/B/C simulation outputs は **すべて未観測** | **"Pure" pre-registration 達成可能**。Section 3.1 で「誰が何を observed したか」を明示 |
| C1: 手順変更 | Simulation のため data collection はなし | N/A |
| C2: 分析時の仮定違反発見 | 14-cell main は frequentist bootstrap + BCa（仮定 light）。28-cell EB は Beta-Binomial conjugate（method of moments + sensitivity sweep）で robust 化 | Section 6.5 で deviation policy 明記 |
| **C6: Program-level null result reporting** | 本研究は MAPE > 60% でも publication する commitment（D-NEW8） | Section 7.3 で commit |
| C9: Narrative inference selectivity | Stage 3 sensitivity sweep（V, f1, f2, EB strength, threshold, K）すべてを pre-reg で固定、selective reporting 防止 | Section 6.4 で固定 |

### 0.3 本 pre-reg と研究計画 v6/v7 の関係

- **研究計画 v6/v7**: 動機・思想・文献根拠を含む **internal document**（L1+L2+L3+L4 すべて）
- **本 pre-reg**: 実装と検証の **predictive commitment** に純化（**L1 描述・予測 only**）
- L3/L4 規範主張（自己責任批判等）は本 pre-reg には含めない（4 層分離原則、研究計画 Part 0.4）

---

## 1. Study Information

### 1.1 Title

**HEXACO 7-Typology Workplace Harassment Microsimulation: Latent Prevalence Prediction and Target Trial Emulation of Personality-Based and Structural Counterfactuals in Japan**

短縮形: "HEXACO Harassment Microsim (Phase 1 + Phase 2)"

### 1.2 Authors

- **Sole author**: Eisuke Tokiwa
- **ORCID**: 0009-0009-7124-6669
- **Affiliation**: SUNBLAZE Co., Ltd.
- **Corresponding email**: eisuke.tokiwa@sunblaze.jp
- Co-authors: なし（sole-authored study）
- **Independent methodologist consultation**: pre-reg lock 後・Stage 1 validation 着手前に外部方法論者（**mode B: anonymous external methodologist**、専門分野は数理生物学 / mathematical biology）に Section 5 (Analysis Plan) のみ review 依頼予定。本人の意向により本 pre-reg および将来の論文 Acknowledgments では **anonymous** として記載（"We thank an anonymous external methodologist with mathematical biology background for review of the analysis plan"）。氏名は author の private record として保持。Munafò et al. 2017 Box 1 CHDI Foundation 例の軽量化適用。

### 1.3 Description（abstract レベル）

職場ハラスメントは多要因現象であり、組織 stressor（Bowling & Beehr 2006、ρ=.30–.53）、subjective social status（Tsuno et al. 2015、OR=4.21）、業種、法律・社会規範等の効果は personality より大きい。本研究はこれらを認めた上で、**personality contribution を isolate** し、HEXACO 7 類型ベースの確率モデルが日本の職場ハラスメント prevalence をどの程度予測できるかを検証する。

**Phase 1（実証 simulation）**: N=354 (harassment behavior) と N=13,668 (HEXACO clustering) の既存データから 14-cell（7 類型 × 2 gender）conditional harassment propensity を推定し、bootstrap で日本労働人口（約 6,800 万）にスケールアップした **latent prevalence** を、MHLW 全国統計（**expressed prevalence**）と triangulate する。

**Phase 2（介入 counterfactual）**: Hernán & Robins (2020) target trial emulation framework 下で、(A) universal HH intervention（Kruse 2014 anchor）、(B) targeted high-risk type intervention（Hudson 2023 anchor、**主軸**）、(C) structural-only intervention（Pruckner 2013 anchor）の 3 種介入を simulate し、population-level harassment 削減量を予測。

LLM は使用しない。すべての mechanism は確率テーブルベースで透明。

### 1.4 Hypotheses

#### H1（Phase 1 main）

**14-cell（7 type × 2 gender）conditional harassment propensity を population aggregate に scale up した予測値は、MHLW 2016（pre-law、過去 3 年間ハラスメント被害経験 32.5%）に対して MAPE ≤ 30% の精度で再現される。**

- 主 validation target: MHLW 2016 R2「職場のハラスメントに関する実態調査」（過去 3 年間 prevalence 32.5%）
- 副 validation target: MHLW 2020 R2 (31.4%、移行期)、MHLW 2024 R5 (19.3%、post-law)
- 国際 baseline: ILO (2022) Asia-Pacific lifetime 19.2%
- 周辺分布検証: Tsuno et al. (2015) N=1,546 random sample, 30-day prevalence 6.1%

#### H2（Phase 1 baseline hierarchy）

**B0 (random) < B1 (gender only) < B2 (HEXACO 6 領域線形) ≤ B3 (7 typology) ≤ B4 (B3 + age + 業種推定 + 雇用形態)** の単調増加が観測される。

- 関心評価: B3 vs B2 の MAPE 差（typology の incrementality）、B4 vs B3 の MAPE 差（personality slice の incrementality）

#### H3（Phase 1 latent vs expressed gap）

**MHLW 2016 (pre-law、32.5%) と本 simulation の latent prediction の gap は、MHLW 2024 (post-law、19.3%) との gap より小さい。** すなわち、pre-law condition は latent-proximal、post-law は environmental gating 強。

#### H4（Phase 2 Counterfactual A: Universal HH intervention）

**全人口の HH を +0.3 SD shift（Kruse 2014 d=0.71 の保守的 discount）した条件下で、predicted national prevalence が baseline から ΔP_A だけ削減される。**

- δ main = +0.3 SD; sensitivity range [0.1, 0.5] SD
- ΔP_A の方向性予測: 削減（負の方向）

#### H5（Phase 2 Counterfactual B: Targeted intervention、★ 本研究の主軸）

**高リスク類型（Cluster 0 = Self-Oriented Independent profile、副候補 Cluster 4, 6）に対する HH +0.4 SD shift（Hudson 2023 self-selected effect の保守的 discount）は、ΔP_B 削減を予測し、ΔP_B / 介入対象人数 比 (cost-effectiveness) は Counterfactual A のそれを上回る。**

- δ main = +0.4 SD; sensitivity range [0.2, 0.6] SD
- 主 target: Cluster 0; 副 target: Cluster 4, 6（実装時に IEEE Cluster 番号と CSV 番号を centroid 数値で照合確定）

#### H6（Phase 2 Counterfactual C: Structural-only intervention）

**個人 personality を変えず確率テーブルを 20% 下方修正した条件下で、predicted national prevalence は baseline から削減されるが、その削減量 ΔP_C は Counterfactual B の ΔP_B より小さい。**

- effect_C main = 20%; sensitivity range [10%, 30%]（30% は上限、Bezrukova 2016 / Roehling 2018 / Dobbin & Kalev 2018 / Pruckner 2013 の 4 系統 triangulation）
- domain transfer caveat（Pruckner = newspaper honor、本研究 = workplace harassment）を Discussion で明記

#### H7（Phase 2 main contrast、★ 主たる predictive commitment）

**ΔP_B > ΔP_A and ΔP_B > ΔP_C**（targeted > universal AND targeted > structural-only at population level）。

- 失敗時（ΔP_B ≤ ΔP_A or ΔP_B ≤ ΔP_C）も publish（Section 7.3 negative result commitment）

#### Hypothesis 階層整理

| H# | Phase | 種別 | 検証指標 |
|---|---|---|---|
| H1 | 1 | Confirmatory predictive | MAPE vs MHLW 2016 |
| H2 | 1 | Confirmatory ordinal | B0–B4 MAPE 単調 |
| H3 | 1 | Exploratory descriptive | MHLW 2016 gap < MHLW 2024 gap |
| H4 | 2 | Conditional projection | ΔP_A direction & magnitude |
| H5 | 2 | Conditional projection | ΔP_B direction & cost-eff |
| H6 | 2 | Conditional projection | ΔP_C direction & magnitude |
| H7 | 2 | Confirmatory ordinal | ΔP_B > ΔP_A and ΔP_B > ΔP_C |

### 1.5 本研究が **検証しない** こと（scope の honest acknowledgment）

- 個人レベル予測（cell N=10–30、pairwise MDE d ≥ 0.92 では individual prediction 不可、D13 power analysis）
- Personality と SSS の独立寄与（N=354 に SSS 直接測定なし、研究計画 Part 1.5）
- 因果方向（cross-sectional design、reverse causation を完全否定不能、Roberts & DelVecchio 2000 の plateau r=.74 が mitigating evidence）
- 介入の長期効果（Roberts 2017 anchor は 24 週時点のみ）
- 西欧 anchor の日本 transportability（Sapouna 2010 cultural moderator 例、Section 4.4 で sensitivity sweep）

---

## 2. Design Plan

### 2.1 Study Type

**Secondary analysis of preexisting data + microsimulation + counterfactual projection**

- 新規データ収集なし
- LLM / generative agent 不使用
- Mechanism: 確率テーブル + Monte Carlo bootstrap + Empirical Bayes shrinkage（Beta-Binomial conjugate、method of moments）
- Causal framing: target trial emulation (Hernán & Robins 2020) + structural causal model (Pearl 2009 do-operator)

### 2.2 Blinding

OSF Standard Pre-Registration の "Blinding" 概念を本研究に適用すると：

| 項目 | 状態 |
|---|---|
| 個人の HEXACO score | **既に observed**（N=354 / N=13,668、既存 IRB 通過済） |
| 個人の harassment self-report (N=354) | **既に observed**（Tokiwa harassment preprint で集計済） |
| 7 類型 × gender 14-cell harassment cross-tabulation | **未観測**（本 pre-reg で固定する分析仕様で初めて生成） |
| Stage 1 国レベル aggregate prediction | **未観測** |
| Counterfactual A/B/C simulation outputs | **未観測** |
| MHLW survey との MAPE | **未観測** |

→ Nosek 2018 PNAS Challenge 3 の strict adherence 下で **"pure" pre-registration** が成立する範囲を明確化。

**追加の blinding-equivalent commitment**:
- Stage 0 コード実装着手以前に本 pre-reg を OSF 登録
- 14-cell cross-tabulation 集計を行う前に inference criteria（Section 6.6）と sensitivity sweep（Section 6.4）を固定
- MHLW survey との比較を行う前に MAPE success/failure threshold（30% / 60%）を固定

### 2.3 Study Design

#### 2.3.1 Phase 1 Pipeline

```
Stage 0: Type assignment & probability table construction
  ├─ Input: harassment/raw.csv (N=354), clustering/csv/clstr_kmeans_7c.csv (centroids)
  ├─ Step 1: 各 N=354 個人を 7 centroid に Euclidean 最近傍分類 → 7 類型 membership
  ├─ Step 2: 14-cell (7 type × 2 gender) crosstab で harassment binary outcome 集計
  │            (binarization: mean + 0.5 SD per outcome、main; sensitivity: +0.25 / +1.0 SD)
  ├─ Step 3: Bootstrap B=2,000 iter / cell、BCa CI（Efron 1987、DiCiccio & Efron 1996）
  └─ Output: 14-cell propensity table with 95% CI

Stage 1: Population aggregation
  ├─ Input: Stage 0 output + N=13,668 type distribution + MHLW labor force composition
  ├─ Step 1: 13,668 から 7 類型 population proportion を取得
  ├─ Step 2: 厚労省労働力調査の age × gender 分布で reweighting
  ├─ Step 3: Cell-conditional probability × population weights → expected national latent prevalence
  └─ Output: National latent prevalence with bootstrap CI

Stage 2: Validation triangulation
  ├─ Compare national latent prediction to MHLW 2016 (32.5%, primary) / 2020 (31.4%) / 2024 (19.3%)
  ├─ Metrics: MAPE (primary), Pearson r, Spearman ρ, KS distance, Wasserstein distance, calibration plot
  └─ Output: Validation report + cell-level prediction error map

Stage 3: Sensitivity sweeps
  ├─ V (victim multiplier) ∈ {2, 3, 4, 5}
  ├─ f1 (turnover rate) ∈ {0.05, 0.10, 0.15, 0.20}
  ├─ f2 (mental disorder rate) ∈ {0.10, 0.20, 0.30}
  ├─ EB shrinkage strength ∈ {0.5×, 1.0×, 2.0×}
  ├─ Binarization threshold ∈ {mean + 0.25 SD, +0.5 SD (main), +1.0 SD}
  ├─ Cluster K ∈ {4, 5, 6, 7 (main, IEEE 掲載済), 8}
  ├─ Role estimation models: (a) personality 線形, (b) tree-based, (c) 文献ベース
  └─ Output: Robustness diagnostic table

Stage 4: Baseline hierarchy comparison (B0–B4)
  ├─ B0: Uniform random
  ├─ B1: Gender only
  ├─ B2: HEXACO 6 領域 linear regression
  ├─ B3: 7 typology + gender (★ proposed method)
  ├─ B4: B3 + age + 業種推定 + 雇用形態
  └─ Output: MAPE for each baseline, monotonicity check

Stage 5: CMV diagnostic
  ├─ Harman's single-factor test on N=13,668 personality data (target < 50% variance)
  ├─ Marker variable correction (Lindell & Whitney 2001) using HEXACO Openness as theoretical-marker (low expected correlation with harassment)
  └─ Output: CMV diagnostic supplementary
```

#### 2.3.2 Phase 2 Pipeline

```
Stage 6: Target trial emulation specification
  ├─ Target trial protocol (PICO + duration) for each counterfactual
  ├─ Eligibility criteria: 日本労働者 20–64 歳
  ├─ Treatment strategies: A (universal HH +δ SD), B (targeted HH +δ SD on Cluster 0/4/6), C (structural -effect_C × cell prob)
  ├─ Assignment: simulated random
  ├─ Outcome: population-level harassment prevalence
  ├─ Follow-up: 24 weeks (Roberts 2017 anchor)
  └─ Output: 4 identifying assumptions explicit (exchangeability, positivity, consistency, transportability)

Stage 7: Counterfactual simulation
  ├─ A (Kruse 2014 anchor):  shift全 N の HH by +δ SD, re-classify cluster, re-estimate propensity, aggregate to national
  ├─ B (Hudson 2023 anchor): shift Cluster 0/4/6 individuals' HH by +δ SD only
  ├─ C (Pruckner 2013 + Bezrukova 2016 + Dobbin & Kalev 2018 + Roehling 2018 triangulation): multiply cell-conditional probability by (1 − effect_C)
  └─ Output: ΔP_A, ΔP_B, ΔP_C with bootstrap CI

Stage 8: Counterfactual sensitivity
  ├─ A: δ ∈ [0.1, 0.5] SD sweep
  ├─ B: δ ∈ [0.2, 0.6] SD sweep
  ├─ C: effect_C ∈ [0.10, 0.30] sweep
  ├─ Transportability: 西欧 anchor effect × {0.3, 0.5, 0.7, 1.0} (Sapouna 2010 / Nielsen 2017 cultural moderator)
  └─ Output: Robustness table
```

### 2.4 Randomization

物理的 randomization なし（observational + simulation）。Bootstrap resample および Monte Carlo は **固定 seed**（NumPy `default_rng(seed=20260429)`）で deterministic に再現。Seed value は本 pre-reg で固定。

---

## 3. Sampling Plan

### 3.1 Existing Data Statement (Nosek 2018 PNAS Challenge 3 strict adherence)

#### 3.1.1 既に収集済かつ part of authors により observed

- **N=354 harassment data** (`harassment/raw.csv`):
  - Tokiwa harassment preprint で HEXACO + Dark Triad → harassment の HC3-robust regression 集計済
  - **Observed by authors at individual level**: HEXACO 6 領域 score, Dark Triad 3 score, power harassment scale, gender harassment scale, age, gender, area
- **N=13,668 clustering data**:
  - Tokiwa clustering paper（IEEE 掲載済）で 7 類型 centroid と分布が確定
  - **Observed by authors at aggregate level**: 7 centroid (HEXACO 6 領域)、cluster proportion、aggregate descriptives

#### 3.1.2 未観測（本 pre-reg で初めて固定する分析）

- N=354 を 7 centroid に最近傍分類した結果の 7 類型 membership distribution
- 7 type × 2 gender 14-cell crosstab の harassment binary outcome
- Cell-conditional propensity の bootstrap BCa CI
- 28-cell EB-shrunken estimate
- 国レベル aggregate latent prevalence
- MHLW 全国統計との MAPE
- Counterfactual A/B/C の ΔP estimate

#### 3.1.3 Pre-registration "purity" の honest acknowledgment

著者は N=354 の HEXACO ↔ harassment associations を harassment preprint の HC3 regression で部分観測している。本 pre-reg で固定する **type-conditional cell-level propensity table** および **国レベル aggregate prediction** は当該 regression からは直接導出不可だが、**完全 blind ではない**。

→ Nosek 2018 Challenge 3 の "pure pre-registration" は理論的最善であるが、本研究は **partial blinding 状態** であることを明示し、findings の interpretation で「7 類型 cross-tab は preregistered だが、HEXACO 領域レベルの associations は exploratory replication of prior work」と区別する。

### 3.2 Sample Size

#### 3.2.1 Cell-level (Phase 1 main analysis)

D13 power analysis (`simulation/docs/power_analysis/D13_power_analysis.md`) より：

| 項目 | 値 |
|---|---|
| Cell 数（main） | 14 (7 type × 2 gender) |
| Cell N 最小 / 最大 / 中央値 | 10 / 70 / 18 |
| Cell N < 10 | 0 cells (0%) |
| Cell N < 20 | 7 cells (50%) |
| One-sample MDE (Cohen's d) 中央値 / 最大 | 0.67 / 0.89 |
| Pairwise MDE (Cohen's d) 中央値 / 最大 | 0.92 / 1.25 |
| Binary rate CI ±half-width 中央値 | ±13 percentage points (power harassment) |
| Bootstrap iterations | 2,000 / cell |
| Bootstrap CI method | BCa (Efron 1987) |

#### 3.2.2 Cell-level (Phase 1 sensitivity)

| 項目 | 値 |
|---|---|
| Cell 数（sensitivity） | 28 (7 type × 2 gender × 2 role) |
| Cell N < 10 | 16 cells (57%) |
| Cell N = 0 | 4 cells |
| Cell N ≤ 3 | 9 cells |
| 必須対応 | Empirical Bayes shrinkage (Beta-Binomial conjugate, Casella 1985 + Clayton & Kaldor 1987 + Efron 2014 + Greenland 2000) |

#### 3.2.3 National-level (Phase 1 + Phase 2)

母集団 6,800 万（厚労省労働力調査 base）。Cell-conditional probability に population weights を適用、bootstrap で aggregate CI を生成。

### 3.3 Sample Size Rationale

- D13 で **N ≥ 250 (Funder & Ozer 2019 stable r estimation 推奨)** を aggregate analysis で満たす（N=354 ≥ 250）
- 14-cell main で **N ≥ 10 / cell 全充足** → shrinkage なしで bootstrap 推定可能
- Pairwise MDE d ≥ 0.92 は "very large effect" (Cohen 1988) / "rarely found in replication" (Funder & Ozer 2019) → **cell-level pairwise inference は avoid、aggregate-level に focus** という pre-registered limitation

### 3.4 Stopping Rule

新規データ収集なしのため stopping rule は適用外。**Sensitivity sweep の追加・拡張は本 pre-reg で固定した範囲（Section 6.4）に限定**。Pre-reg 後の追加 sensitivity は exploratory として明示し、confirmatory inference には用いない。

---

## 4. Variables

### 4.1 Manipulated Variables (counterfactual operators)

| 変数 | 定義 | 適用範囲 | Main 値 | Sensitivity range | 出典 anchor |
|---|---|---|---|---|---|
| **δ_A**（universal HH shift） | 全 N の HH score を δ × SD(HH) shift | All N=354 → re-classify cluster | **+0.3 SD** | [0.1, 0.5] SD | Kruse 2014 d=0.71 を保守的 discount |
| **δ_B**（targeted HH shift） | Cluster 0/4/6 の N のみ HH score を δ × SD(HH) shift | Cluster 0 主、Cluster 4, 6 副 | **+0.4 SD** | [0.2, 0.6] SD | Hudson 2023 self-selected effect の保守的 discount |
| **effect_C**（structural reduction） | Cell-conditional probability に (1 − effect_C) を乗じる | All 14 cells | **0.20** | [0.10, 0.30] | Pruckner 2013 + Bezrukova 2016 + Roehling 2018 + Dobbin & Kalev 2018 triangulation |
| **transportability_factor** | Phase 2 anchor effect × factor で日本適用 | All counterfactuals | 1.0 (main) | [0.3, 0.5, 0.7, 1.0] | Sapouna 2010 / Nielsen 2017 |

### 4.2 Measured Variables (existing observed data)

#### 4.2.1 個人 level (N=354 harassment data)

| 変数 | 種別 | 操作的定義 | 出典 |
|---|---|---|---|
| **HEXACO 6 領域** | Continuous (Likert 1–5 mean) | Wakabayashi 2014 Japanese HEXACO-60 | `harassment/raw.csv` |
| **Dark Triad 3** | Continuous | Shimotsukasa & Oshio 2017 SD3-J | `harassment/raw.csv` |
| **Power harassment** | Continuous (item mean) → binarized at mean+0.5 SD | Tou et al. 2017 Workplace Power Harassment Scale | `harassment/raw.csv` |
| **Gender harassment** | Continuous → binarized at mean+0.5 SD | Kobayashi & Tanaka 2010 | `harassment/raw.csv` |
| **Age** | Continuous (years) | Self-report | `harassment/raw.csv` |
| **Gender** | Binary (0/1, n=133/220) | Self-report | `harassment/raw.csv` |
| **Area** | Categorical | Self-report | `harassment/raw.csv` |

#### 4.2.2 個人 level (N=13,668 clustering data)

| 変数 | 種別 | 用途 |
|---|---|---|
| **HEXACO 6 領域** | Continuous | Centroid 抽出（既に集計済、`clustering/csv/clstr_kmeans_7c.csv`） |
| **Cluster proportion** | Categorical 7-type | Population scaling weight |

#### 4.2.3 集団 level (MHLW、external validation target)

| 変数 | データソース | 役割 |
|---|---|---|
| **過去 3 年間ハラスメント被害経験率** | MHLW 2016 R2 (32.5%、★ primary), 2020 R2 (31.4%), 2024 R5 (19.3%) | National validation target |
| **業種別 prevalence** | MHLW 2020 R2 図表 | Subgroup validation |
| **30 日 prevalence** | Tsuno et al. 2015 N=1,546 (6.1%) | 周辺分布参照 |
| **国際 baseline** | ILO 2022 Asia-Pacific lifetime 19.2% | 比較参照 |
| **離職理由「人間関係」** | 厚労省雇用動向調査 | f1 anchor |
| **メンタル疾患発症率** | 厚労省労働安全衛生調査 + Tsuno & Tabuchi 2022 PR=3.20 | f2 anchor |

### 4.3 Indices (derived)

| Index | 定義 | 算出 |
|---|---|---|
| **7-type membership** | Each N=354 individual の最近傍 centroid (Euclidean, HEXACO 6 領域) | Stage 0 で生成 |
| **Cell ID (14-cell)** | type ∈ {0..6} × gender ∈ {0,1} | Stage 0 で生成 |
| **Cell ID (28-cell)** | type × gender × role ∈ {0,1} | Sensitivity 用 |
| **Role probability** | Continuous, predicted from C + 0.5·X composite (top 15% → manager) | Stage 0; D1 sensitivity で 3 モデル比較 |
| **National latent prevalence** | Σ_cell (cell_propensity × cell_population_weight) | Stage 1 |
| **MAPE** | mean(|predicted − observed| / observed × 100) | Stage 2, primary metric |
| **ΔP_x** (counterfactual reduction) | predicted_baseline − predicted_counterfactual_x | Stage 7 |
| **Cost-effectiveness ratio** | ΔP_x / N_treated_x | Stage 7, Phase 2 |

---

## 5. Analysis Plan (statistical models)

### 5.1 Phase 1 Stage 0: Cell-level propensity estimation

For each cell c ∈ {1..14} (7 type × 2 gender):
- X_c = number of "harassment perpetrator" (binarized at mean+0.5 SD per outcome) in cell c
- N_c = total N in cell c
- Observed propensity p̂_c = X_c / N_c
- Bootstrap distribution: B = 2,000 BCa resamples / cell (Efron 1987 J Am Stat Assoc; DiCiccio & Efron 1996 Stat Sci)
- BCa correction: bias z₀ + acceleration a from jackknife

**Output**: 14-cell table with point estimate p̂_c and 95% BCa CI [p̂_c^lo, p̂_c^hi]

### 5.2 Phase 1 Stage 0 (sensitivity): 28-cell EB shrinkage

Beta-Binomial conjugate (Casella 1985 + Clayton & Kaldor 1987 + Efron 2014 Stat Sci + Greenland 2000 IJE):

1. **Method of moments で hyper-prior 推定** (from 14-cell main):
   - μ̂ = mean(p̂_j), σ̂² = var(p̂_j) for j=1..14
   - α̂ = μ̂ × [μ̂(1−μ̂)/σ̂² − 1]
   - β̂ = (1−μ̂) × [μ̂(1−μ̂)/σ̂² − 1]
2. **28-cell posterior** (each cell k):
   - E[p_k | X_k, N_k] = (α̂ + X_k) / (α̂ + β̂ + N_k)
   - 95% credible interval from Beta(α̂ + X_k, β̂ + N_k − X_k) quantiles
3. **Strength sensitivity sweep** (★ pre-registered):
   - scale ∈ {0.5×, 1.0× (main), 2.0×}: hyper-parameters (α̂, β̂) を scale で乗じて weak / medium / strong shrinkage を比較

**MoM 不安定性 diagnostic** (research plan Part 11.9 caveat):
- N=14 cell で σ̂² 小 → α̂, β̂ が極端に大 (強 prior) になる risk
- **Triangulation**: Marginal MLE および Stan / brms hierarchical Bayes posterior を補助実行
- Diagnostic plot: μ̂ ± SE plot、shrunken vs raw scatter で過剰縮約 visual check

### 5.3 Phase 1 Stage 1: Population aggregation

For target validation period t ∈ {2016, 2020, 2024}:
- W_c (cell weight) = MHLW labor force population × cluster proportion (from N=13,668) × gender proportion × age weight
- National latent prevalence: P̂_t = Σ_c (p̂_c × W_c) / Σ_c W_c
- Bootstrap CI for P̂_t: 2,000 iter, BCa

**Demographic reweighting source**: 厚労省労働力調査（age × gender × 雇用形態）

### 5.4 Phase 1 Stage 2: Validation triangulation

**Primary metric** (★ pre-registered):
- **MAPE**(P̂_2016, MHLW 2016 32.5%) ≤ 30% → SUCCESS
- 30% < MAPE ≤ 60% → PARTIAL SUCCESS
- MAPE > 60% → FAILURE (publish anyway, see Section 7.3)

**Secondary metrics** (descriptive):
- Pearson r between cell-level p̂_c and MHLW subgroup rates
- Spearman ρ (rank correlation)
- KS distance (distribution shape)
- Wasserstein distance (earth mover)
- Calibration plot (cell predicted vs observed)

**Subgroup MAPE** (★ pre-registered):
- Gender × age band → subgroup MAPE で failure mode の可視化

### 5.5 Phase 1 Stage 4: Baseline hierarchy

For each baseline B ∈ {B0, B1, B2, B3 (proposed), B4}:
- Train: 同一 N=354
- Predict national prevalence
- Compute MAPE vs MHLW 2016
- Pre-registered ordinal hypothesis (H2): MAPE_B0 ≥ MAPE_B1 ≥ MAPE_B2 ≥ MAPE_B3 ≥ MAPE_B4

**Models**:
- **B0**: Uniform p_random = MHLW 2016 grand mean
- **B1**: gender-only logistic (P̂(harassment | gender))
- **B2**: HEXACO 6 領域 linear (logistic regression, all 6 domains)
- **B3** (proposed): 7 type × gender cell-conditional (本研究 main)
- **B4**: 7 type + age + 業種推定 + 雇用形態 cell-conditional

**Decision rule**:
- B3 > B2 → "typology が線形を上回る情報追加"
- B3 ≈ B2 → "typology は線形より優位ではない"（finding として report）
- B3 < B2 → "typology が overfit"（critical finding として report）
- B4 ≫ B3 → "personality slice 単独不十分、周辺変数必要"
- B4 ≈ B3 → "personality typology が周辺変数を informationally subsume"

### 5.6 Phase 1 Stage 5: CMV diagnostic

**Harman's single-factor test** (Podsakoff et al. 2003 J Appl Psychol):
- N=13,668 personality data 全 HEXACO items を 1 factor unrotated EFA
- 第 1 因子 variance explained < 50% → CMV concern 限定的（pre-registered threshold）

**Marker variable correction** (Lindell & Whitney 2001 J Appl Psychol):
- HEXACO Openness を theoretical-marker として selection
- Marker と target correlations の partial-out で adjusted associations を report

### 5.7 Phase 2 Stage 6–8: Counterfactual estimation

#### 5.7.1 Target trial emulation specification

各 counterfactual に対し PICO + duration を pre-register:

| Element | Counterfactual A | Counterfactual B (★ main) | Counterfactual C |
|---|---|---|---|
| **P** (population) | 日本労働者 20–64 歳 | 日本労働者 20–64 歳 | 日本労働者 20–64 歳 |
| **I** (intervention) | Universal HH +δ_A SD | Targeted (Cluster 0/4/6) HH +δ_B SD | Structural × (1 − effect_C) on cell prob |
| **C** (control) | Pre-intervention baseline | Pre-intervention baseline | Pre-intervention baseline |
| **O** (outcome) | National harassment prevalence | National harassment prevalence + cost-effectiveness | National harassment prevalence |
| **Duration** | 24 weeks (Roberts 2017) | 24 weeks | 24 weeks |
| **Anchor** | Kruse 2014 d=0.71 | Hudson 2023 b=.03/week | Pruckner 2013 + Bezrukova 2016 + Roehling 2018 + Dobbin & Kalev 2018 |

#### 5.7.2 Pearl 2009 do-operator notation

- Counterfactual A: do(HH := HH + δ_A × SD(HH)) for all individuals
- Counterfactual B: do(HH := HH + δ_B × SD(HH)) for individuals in Cluster ∈ {0, 4, 6}
- Counterfactual C: do(p_c := p_c × (1 − effect_C)) for all cells c

#### 5.7.3 Estimation

For each counterfactual x ∈ {A, B, C}:
- Apply do-operator to N=354 / cell-prob table per spec
- Re-run Stage 0 → Stage 1 (Stage 2 validation 不要、prediction のみ)
- ΔP_x = P̂_baseline − P̂_x
- Bootstrap CI for ΔP_x (2,000 iter, propagate cell-level uncertainty)
- Cost-effectiveness for B: ΔP_B / |Cluster 0 ∪ 4 ∪ 6 in population|

#### 5.7.4 Identifying assumptions (Hernán & Robins 2020)

Discussion で **明示的に honest 評価**:

1. **Exchangeability**: Y^a ⊥⊥ A | L
   - Violation risk: 文化・組織風土の unmeasured confounding
   - Mitigation: B4 baseline で周辺変数調整、sensitivity sweep
2. **Positivity**: P(A=a | L=l) > 0 for all l
   - Violation risk: Cluster 6 dominant 32% で intervention coverage 不均一
   - Mitigation: Cluster 6 を意図的に counterfactual analysis から除外しない
3. **Consistency**: observed Y when A=a equals Y^a (no interference between agents)
   - Violation risk: 職場での peer effect、displacement
   - Mitigation: Counterfactual C の displacement を Discussion で明示
4. **Transportability**: anchor study population effect が target Japanese workforce に transport
   - Violation risk: Kruse / Hudson / Pruckner はすべて西欧 / 米国 sample
   - Mitigation: Section 5.8 transportability sensitivity sweep + Sapouna 2010 / Nielsen 2017 cultural moderator 引用

### 5.8 Phase 2 Stage 8: Transportability sensitivity

| Factor | Range | 解釈 |
|---|---|---|
| 0.3× | Strong cultural attenuation (Sapouna 2010 UK→Germany null worst case) | 保守的 lower bound |
| 0.5× | Moderate attenuation (Nielsen 2017 Asia/Oceania Neuroticism r=.16 vs Europe .33 比) | "expected" attenuation |
| 0.7× | Mild attenuation | Optimistic |
| 1.0× (main) | No attenuation (anchor effect = Japan effect) | Reference |

→ Conclusions の robustness を transportability_factor に対して report

---

## 6. Inference Criteria & Sensitivity Master Table

### 6.1 Pre-registered inference criteria summary

| H# | Criterion | Threshold | Decision |
|---|---|---|---|
| **H1** | MAPE(P̂_2016, MHLW 2016 32.5%) | ≤ 30% | SUCCESS |
| H1 | MAPE | 30 < x ≤ 60% | PARTIAL SUCCESS |
| H1 | MAPE | > 60% | FAILURE (publish) |
| **H2** | MAPE_B0 ≥ MAPE_B1 ≥ MAPE_B2 ≥ MAPE_B3 ≥ MAPE_B4 | Strict monotonicity | Pre-registered direction confirmed if ≥ 3 pairwise inequalities hold |
| H3 | gap(2016) < gap(2024) | Direction | Confirmed if MAPE_2016 < MAPE_2024 |
| H4 | sign(ΔP_A) | < 0 (削減) | Confirmed if 95% CI excludes 0 in negative direction |
| H5 | sign(ΔP_B) and ΔP_B / N_treated > ΔP_A / N_total | Cost-eff | Confirmed if both conditions hold |
| H6 | sign(ΔP_C) | < 0 | Confirmed if 95% CI excludes 0 |
| **H7** | ΔP_B > ΔP_A AND ΔP_B > ΔP_C | Magnitude ranking | Confirmed if both inequalities at point estimate; uncertain if 95% CI overlap |

### 6.2 Failure mode commitment

- **H1 failure** (MAPE > 60%): Publish as failure mode discovery (Doc 1 戦略 2 + Nosek 2018 Challenge 6)
- **H2 reversal** (B3 < B2): Publish as critical finding (typology overfitting)
- **H7 reversal** (ΔP_B ≤ ΔP_A or ΔP_B ≤ ΔP_C): Publish; main thesis claim 修正

### 6.3 Multiple comparison correction

- Phase 1 H1: Single primary test, no correction needed
- Phase 1 H2: 4 ordinal pairwise comparisons (B0-B1, B1-B2, B2-B3, B3-B4) → Bonferroni-Holm at α = .05 family-wise
- Phase 2 H4–H7: 3 counterfactuals × 1 main test each = 3 tests → Bonferroni-Holm at α = .05
- H7 ranking: Already a single composite test (no further correction)

### 6.4 Pre-registered sensitivity sweep master table (★ Stage 3 / Stage 8)

**Phase 1 sensitivity** (all combinations report MAPE table):

| Parameter | Main | Sweep range | Stage |
|---|---|---|---|
| **V** (victim multiplier) | 3 | {2, 3, 4, 5} | 3 |
| **f1** (turnover rate) | 0.10 | {0.05, 0.10, 0.15, 0.20} | 3 |
| **f2** (mental disorder rate) | 0.20 | {0.10, 0.20, 0.30} | 3 |
| **EB shrinkage scale** | 1.0× | {0.5×, 1.0×, 2.0×} | 0 (28-cell) |
| **Binarization threshold** | mean+0.5 SD | {mean+0.25 SD, +0.5 SD, +1.0 SD} | 0 |
| **Cluster K** | 7 (IEEE 掲載済 main) | {4, 5, 6, 7, 8} | 0 |
| **Role estimation** | C+0.5X composite top 15% | {(a) linear, (b) tree-based, (c) literature} | 0 (28-cell) |
| **Bootstrap iterations** | 2,000 | (fixed) | All |

**Phase 2 sensitivity**:

| Parameter | Main | Sweep range | Stage |
|---|---|---|---|
| **δ_A** (universal HH shift) | +0.3 SD | [0.1, 0.5] SD step 0.1 | 8 |
| **δ_B** (targeted HH shift) | +0.4 SD | [0.2, 0.6] SD step 0.1 | 8 |
| **effect_C** (structural reduction) | 0.20 | [0.10, 0.30] step 0.05 | 8 |
| **transportability_factor** | 1.0× | {0.3×, 0.5×, 0.7×, 1.0×} | 8 |
| **Target type set** (B) | {Cluster 0, 4, 6} | {Cluster 0 only}, {0+4}, {0+4+6} (main), {0+4+6+others} | 8 |

**Reporting**: 全 sensitivity 結果は supplementary table として open data に publish (Section 8 reproducibility)。Main text は main 値のみを quantitative claim で使用、sensitivity は robustness 確認の qualitative statement に限定。

### 6.5 Deviation policy (Nosek 2018 Challenge 1, 2 対応)

Pre-reg からの逸脱は以下に分類して report:

1. **Level 0 (no deviation)**: 仕様通り実装
2. **Level 1 (minor specification clarification)**: 例 — bootstrap seed の lookup table での明示化、CI 計算 library version の固定化等。Methods で 1 行記録
3. **Level 2 (data-driven adjustment)**: 例 — Method of Moments で α̂, β̂ が発散 → MLE / hierarchical Bayes に切替。Discussion 1 段落で justification
4. **Level 3 (analysis plan revision)**: 例 — 14-cell pairwise inference を行う等の重大変更 → **本 pre-reg を update し、版 v2 を OSF に新規登録**。Original v1 と diff を public 化

→ Level 2 以上の逸脱を Discussion の専用 subsection "Deviations from Pre-Registration" で comprehensive 報告。

### 6.6 Inference data lock-in commitments

- **MHLW survey との比較** は本 pre-reg を OSF 登録した後に初めて実施
- **MAPE 計算** は本 pre-reg で固定された binarization threshold (mean+0.5 SD) と population weights で先行実施
- **MAPE threshold (30% / 60%)** の post-hoc 変更は禁止

---

## 7. Negative Result & Failure Mode Publication Commitment (D-NEW8)

Anchor: **Nosek et al. 2018 PNAS Challenge 6** (program-level null result reporting) + Doc 1 戦略 2 (失敗の主題化)

### 7.1 公表 commitment 内容

以下のすべてのケースで投稿論文として publish する:

1. **H1 SUCCESS** (MAPE ≤ 30%): 通常論文として投稿
2. **H1 PARTIAL SUCCESS** (30 < MAPE ≤ 60%): "partial validation" 論文として投稿、failure mode を主題化
3. **H1 FAILURE** (MAPE > 60%): "negative result" 論文として投稿、failure mode を発見化
4. **H7 reversal** (ΔP_B ≤ ΔP_A or ΔP_C): "thesis revision" 論文として投稿、original framing を修正
5. **B3 < B2** (typology overfitting): "critical finding" 論文として投稿、本研究の typology approach の限界を明示

### 7.2 Target journal の幅

Negative / null / partial 結果も受け入れる candidacy を確保:

| Journal | 種別 | Negative result 対応 |
|---|---|---|
| **JBE** (Journal of Business Ethics) | 第一候補 | Section editor 判断、ethics-relevant null は受付歴あり |
| **PAID** (Personality and Individual Differences) | 第二 | 通常論文 |
| **PLOS ONE** | バックアップ | "Negative results welcome" 公式 policy |
| **RIO Journal** | バックアップ | Registered Report 受付 |
| **Cortex** | Registered Report 受付 | Negative result publication track 確立 |
| **J Comp Soc Sci** | Computational バックアップ | 通常論文 |

### 7.3 Publication 拒否時の対応

- Pre-print (OSF / SocArXiv / PsyArXiv) で公開
- Failure mode の technical report を OSF supplementary として permanent 公開
- 本 pre-reg を最低 10 年間 OSF で active 維持

---

## 8. Reproducibility Infrastructure (D-NEW9)

Anchor: **Munafò et al. 2017 Nature Human Behaviour** "A manifesto for reproducible science" 5 themes (Methods / Reporting / Reproducibility / Evaluation / Incentives) + TOP guidelines.

### 8.1 Pre-registered infrastructure commitments

| Item | Implementation |
|---|---|
| **Random seed** | NumPy `default_rng(seed=20260429)`, Python `random.seed(20260429)`, Stan `seed=20260429`. Seed value 固定、本 pre-reg に明記。Bootstrap resample state は HDF5 で永続保存。 |
| **環境 pinning** | (a) `uv` lock file (Python 3.11+) **OR** (b) Dockerfile (Python:3.11-slim base) のいずれかで完全 pinning |
| **Make reproduce** | `make reproduce` で全 figure / table / supplementary を 30 分以内に再生成（D-NEW9 README 必須項目） |
| **Open data** | Aggregated cell-level statistics (14-cell, 28-cell EB) は OSF / GitHub で完全公開。Cell-level raw data は restricted access (Section 9.5 ethics) |
| **Open code** | GitHub + OSF mirror。MIT license。 |
| **Reporting checklist** | OSF Pre-Registration template + STROBE (observational study) + STARD (where applicable) を Methods に統合 |
| **Independent statistical oversight** | Co-author / external methodologist による review を **Stage 1 完了後・Stage 2 着手前** に実施（Munafò 2017 Box 1 CHDI Foundation 例） |

### 8.2 Pre-registered repository structure

```
simulation/
├── docs/
│   ├── pre_registration/
│   │   ├── D12_pre_registration_OSF.md (本ドキュメント、内部 master)
│   │   ├── D12_pre_registration_OSF.en.md (英訳版、OSF 提出用)
│   │   └── osf_registration_metadata.json (登録日時、DOI、version log)
│   ├── notes/
│   ├── literature_audit/
│   └── power_analysis/
├── code/
│   ├── stage0_type_assignment.py
│   ├── stage0_cell_propensity.py
│   ├── stage1_population_aggregation.py
│   ├── stage2_validation.py
│   ├── stage3_sensitivity.py
│   ├── stage4_baselines.py
│   ├── stage5_cmv_diagnostic.py
│   ├── stage6_target_trial.py
│   ├── stage7_counterfactual.py
│   └── stage8_transportability.py
├── tests/
│   └── test_*.py (each stage)
├── output/
│   ├── tables/
│   ├── figures/
│   └── supplementary/
├── Dockerfile
├── pyproject.toml + uv.lock
├── Makefile (with `reproduce` target)
└── README.md (with "How to reproduce in 30 minutes" section)
```

### 8.3 Pre-registered reporting items

論文 Methods + supplementary に必須記載:

- [ ] Random seed 値 (20260429)
- [ ] Software version: Python, NumPy, SciPy, scikit-learn, statsmodels, Stan/brms (適用時), PyMC (適用時)
- [ ] Hardware: CPU model, RAM, OS (再現性に影響する範囲)
- [ ] Bootstrap iterations: 2,000 (per cell)
- [ ] BCa CI parameters: jackknife-based bias z₀ + acceleration a の数値
- [ ] EB shrinkage hyper-parameters: α̂, β̂ の point estimate と sensitivity sweep の各 scale
- [ ] MoM diagnostic: σ̂² が μ̂(1−μ̂) に対してどの fraction かを report
- [ ] CMV diagnostic: Harman's first factor variance %, marker variable correction adjustments

---

## 9. Ethical Commitments (D-NEW10、研究計画 v6 Part 7 全項目)

### 9.1 Pre-registered triple-locking 構造

Anchor: 研究計画 v6 Part 7（Sensitive topic としての harassment 研究の特性に対する 8 セクション対応）

| Lock 位置 | Content |
|---|---|
| **(a) Methods 内** | "Individual application disavowal" statement (Section 9.2 全文) を Methods 末尾に明記 |
| **(b) Discussion 内** | "How this should NOT be used" 専用段落、Cluster X 個人を加害者と決めつけない言語選択遵守 |
| **(c) Pre-registration 内 (本 Section 9)** | Part 7 全項目を ethical commitment として明示、本 pre-reg を OSF で公開 |

### 9.2 Individual application disavowal (Anti-screening Statement)

論文 Methods / Discussion / 著作の各所に以下相当の disclaimer を配置:

> **Statement on individual application**: This study estimates **population-level statistical patterns**, not individual risk profiles. The cell-conditional probabilities reported here have **no validity for individual prediction**, screening, hiring, promotion, or any other personnel decision. Application of these findings to individual personnel decisions would constitute (a) statistical misuse, given the pairwise MDE of d ≥ 0.92 and absent individual-level predictive validation, and (b) ethically unacceptable discrimination based on personality profiles. The authors explicitly disavow such application.

加えて以下 4 点を併記:

- Cluster 0 でも加害確率は < 100%
- Cluster 1–6 でも加害は発生する
- Personality は変化可能（Roberts 2017）→ Cluster は固定属性ではない
- 採用・昇進判断への利用は研究倫理違反

### 9.3 Authorial framing guidelines (pre-registered)

| 避ける表現 | 推奨表現 |
|---|---|
| "perpetrator personality" | "personality profile associated with elevated risk in our model" |
| "high-risk individuals" | "cell with elevated baseline rate (population-level pattern)" |
| "Cluster X should be screened" | "Cluster X represents a target group for **opt-in voluntary intervention**" |
| "predict perpetrators" | "predict population-level prevalence" |
| "effective intervention reduces harassment" | "anchor literature suggests intervention can reduce harassment under specified transportability assumptions" |

### 9.4 Voluntary, opt-in 政策提言原則 (Counterfactual B 主軸の必須条件)

- **Voluntary participation**: 強制的研修・カウンセリングではなく opt-in
- **No employment consequence**: 参加の有無が雇用・評価に影響しない
- **Anonymity / Confidentiality**: 参加者の personality profile が組織内同定不能
- **Resource provision, not coercion**: 社会の役割は支援提供、強制ではない

### 9.5 Data availability ethics (D17)

| Data tier | Public availability |
|---|---|
| **Aggregated statistics** (14-cell propensity, 28-cell EB, national prevalence, ΔP_x) | 完全公開 (OSF + GitHub) |
| **Cell-level raw data** (cell ごとの個人 records) | Restricted access (要研究目的審査、IRB 確認、再同定 risk 評価) — 28-cell の N<10 cells で再同定 risk あり |
| **N=354 / N=13,668 個人データ** | 既存論文の公開方針に準拠（個人特定情報削除済 anonymized data のみ要望ベース提供） |

### 9.6 Long-term ethical monitoring (10-year commitment)

- **誤用 case の monitoring**: discrimination / screening 転用報告に対し correction / commentary 発表
- **二次解析 collaboration の選別**: ethical principle 適合審査
- **Author contact 維持**: corresponding author 連絡先 10 年以上維持
- **OSF maintenance**: 本 pre-reg と supplementary を 10 年以上 active 維持

---

## 10. Limitations Pre-Acknowledged (研究計画 Part 4.2 11 limitations)

本 pre-reg で予め以下の limitation を acknowledge し、Discussion での post-hoc rationalization を回避（Nosek 2018 Challenge 9）:

| # | Limitation | 対応文献 |
|---|---|---|
| L1 | Self-report → Self-report 循環性 | Berry et al. 2012, Anderson & Bushman 2002, Vazire 2010 SOKA (direction = conservative) |
| L2 | N=354 代表性 (cloudsourcing self-selection) | Tsuno et al. 2015 N=1,546 random sample triangulation |
| L3 | 役職推定の誤差 | D13 で continuous covariate 化、D1 で 3 モデル比較 |
| L4 | Cell-level 統計検出力 (pairwise MDE d ≥ 0.92) | D13 power analysis、aggregate inference 重視 |
| L5 | 被害者倍率 V / 離職率 f / メンタル疾患率 f2 不確実性 | Stage 3 sensitivity sweep |
| L6 | Cross-sectional 因果不可能 + reverse causation | Roberts & DelVecchio 2000 plateau r=.74、Specht 2011 mid-life stability |
| L7 | GAM situational 欠落 | "Personality slice" framing、Bowling & Beehr 2006 acknowledged |
| L8 | Common Method Bias | Podsakoff 2003 standard、Stage 5 diagnostic |
| L9 | Phase 2 transportability (西欧 anchor → 日本) | Section 5.8 sensitivity sweep、Sapouna 2010 / Nielsen 2017 |
| L10 | Latent vs expressed prevalence 分離不可 | Section 1.4 framing で「境界連続的」明示 |
| **L11** | **Construct validity** (harassment scale が "latent propensity" を測っているか未検証) | Vazire 2010 SOKA、convergent / discriminant validity check (supplementary) |

---

## 11. Researcher Reflexivity (Positionality)

### 11.1 Pre-registered positionality statement

論文 Methods / Acknowledgments に以下を明記:

> "The first author has previously argued in non-peer-reviewed work for a systemic-causation framing of workplace harassment, emphasizing social systems' responsibility over individual self-responsibility (Tokiwa, *forthcoming*). This normative stance may influence which limitations are emphasized and how findings are framed in the Discussion. To mitigate, (a) the empirical analysis section is restricted to L1 descriptive/predictive claims, (b) all causal language is constrained by the target trial emulation framework (Hernán & Robins, 2020) with explicit identifying assumptions, (c) anti-screening and anti-discrimination statements are included regardless of findings, (d) limitations are presented in a structured 11-item framework derived from a 60+ paper literature audit (not selectively), (e) negative result publication is committed in advance via Section 7 of this preregistration (D-NEW8), and (f) independent methodologist review is sought prior to Stage 2 validation (Section 8.1 D-NEW9)."

### 11.2 Bias mitigation procedures (pre-registered)

- Pre-registered analysis plan (本ドキュメント) で post-hoc 修正可能性を制限 (Nosek 2018)
- Co-author / external review for Discussion interpretation (Munafò 2017 Box 1)
- Failure mode の事前 commitment (Section 7)
- Self-report direction-of-bias awareness: Vazire 2010 SOKA 上 conservative (lower bound) として framing

---

## 12. Other (Conflicts, Funding, Data Sources)

### 12.1 Conflicts of Interest

- 著者は本研究結果から商業的利益を受けない
- 著者所属（SUNBLAZE Co., Ltd.）は本研究結果に対する商業的利害関係を有しない
- 本研究は既存 IRB 通過済データの secondary analysis であり、新規データ収集なし
- Pre-registration の OSF 登録は無料、author cost なし

### 12.2 Funding

- **Simulation phase（本研究）**: No external funding. 著者所属（SUNBLAZE Co., Ltd.）の通常業務時間内で実施。
- **Original data collection**:
  - N=354 (harassment data): Tokiwa harassment preprint で記載済の元 IRB / funding を継承
  - N=13,668 (clustering data): Tokiwa clustering paper (IEEE 掲載済) で記載済の元 IRB / funding を継承
- **新規 IRB**: 不要（本研究は simulation のみで人を対象とする新規研究行為を含まない、既存 anonymized data の secondary analysis）

### 12.3 Data Sources Summary

| Source | Type | Access |
|---|---|---|
| `harassment/raw.csv` (N=354) | Primary | Authors' prior IRB-approved collection (Tokiwa harassment preprint) |
| `clustering/csv/clstr_kmeans_7c.csv` | Derived | Tokiwa clustering paper IEEE 掲載済 (centroid table) |
| MHLW 2016 R2 実態調査 | Public | https://www.mhlw.go.jp/ (要 specific URL on registration) |
| MHLW 2020 R2 実態調査 | Public | https://www.mhlw.go.jp/ |
| MHLW 2024 R5 実態調査 | Public | https://www.mhlw.go.jp/ |
| MHLW 雇用動向調査 | Public | https://www.mhlw.go.jp/ |
| MHLW 労働安全衛生調査 | Public | https://www.mhlw.go.jp/ |
| MHLW 労働力調査 | Public | https://www.stat.go.jp/data/roudou/ |
| ILO 2022 Global survey | Public | https://www.ilo.org/ |
| Tsuno et al. 2015 N=1,546 | Published | Tsuno et al. 2015 PLOS ONE |
| Tsuno & Tabuchi 2022 | Published | Tsuno & Tabuchi 2022 |

---

## 13. Citation Anchors (Tier 1+2+3+4 文献基盤)

本 pre-reg の各 commitment は以下文献に依拠。完全 list は `simulation/docs/literature_audit/deep_reading_notes.md` (40 paper deep reading) を参照。

### 13.1 Tier 4 metascience anchors (★ 本 pre-reg の核)

- **Nosek, Ebersole, DeHaven & Mellor (2018)**. The preregistration revolution. *PNAS, 115*(11), 2600–2606. https://doi.org/10.1073/pnas.1708274114 [→ Section 0.2, 3.1, 7]
- **Munafò et al. (2017)**. A manifesto for reproducible science. *Nature Human Behaviour, 1*, 0021. https://doi.org/10.1038/s41562-016-0021 [→ Section 8]
- **Vazire (2010)**. Who knows what about a person? The Self–Other Knowledge Asymmetry (SOKA) Model. *JPSP, 98*, 281–300. [→ Section 10 L1, L11]
- **Funder & Ozer (2019)**. Evaluating effect size in psychological research: Sense and nonsense. *AMPPS, 2*, 156–168. [→ Section 3.3]

### 13.2 Causal inference framework

- **Hernán & Robins (2020)**. Causal Inference: What If. Chapman & Hall/CRC. [→ Section 5.7.4]
- **Pearl (2009)**. Causality: Models, Reasoning, and Inference (2nd ed.). Cambridge. [→ Section 5.7.2]

### 13.3 Statistical methods anchors

- **Efron (1987)**. Better bootstrap confidence intervals. *J Am Stat Assoc*. [→ BCa]
- **DiCiccio & Efron (1996)**. Bootstrap confidence intervals. *Stat Sci*. [→ BCa]
- **Casella (1985)**. An introduction to empirical Bayes data analysis. *Am Stat*. [→ EB]
- **Clayton & Kaldor (1987)**. Empirical Bayes estimates of age-standardized relative risks. *Biometrics*. [→ EB epidemiology]
- **Efron (2014)**. Two modeling strategies for empirical Bayes estimation. *Stat Sci*. [→ EB modern]
- **Greenland (2000)**. Principles of multilevel modelling. *Int J Epidemiol*. [→ EB / multilevel]
- **Podsakoff, MacKenzie, Lee & Podsakoff (2003)**. Common method biases in behavioral research. *J Appl Psychol*. [→ CMV]
- **Lindell & Whitney (2001)**. Accounting for common method variance. *J Appl Psychol*. [→ marker variable]
- **Schofield et al. (2018)**. Health-related microsimulation. *Eur J Health Econ*. [→ MAPE microsim]

### 13.4 Phase 2 intervention anchors

- **Kruse, Chancellor, Ruberton & Lyubomirsky (2014)**. An upward spiral between gratitude and humility. *Soc Psychol Personal Sci, 5*(7), 805–814. [→ Counterfactual A]
- **Hudson (2023)**. Lighten the darkness: Personality interventions targeting agreeableness. *J Personality, 91*(4). [→ Counterfactual B 主軸]
- **Pruckner & Sausgruber (2013)**. Honesty on the streets. *J Eur Econ Assoc, 11*(3), 661–679. [→ Counterfactual C]
- **Bezrukova et al. (2016)**. Diversity training meta-analysis. [→ Counterfactual C triangulation]
- **Roehling & Huang (2018)**. Sexual harassment training meta-analysis. [→ Counterfactual C triangulation]
- **Dobbin & Kalev (2018)**. 985 studies meta. [→ Counterfactual C triangulation]
- **Roberts et al. (2017)**. Personality trait change through intervention systematic review. *Psychol Bull, 143*(2), 117–141. [→ 24-week duration anchor]

### 13.5 Personality + harassment anchors

- **Pletzer et al. (2019)**. HEXACO meta. [→ HH × CWB ρ ≈ -.20 to -.35]
- **Nielsen, Glasø & Einarsen (2017)**. FFM × harassment meta. [→ cultural moderator]
- **Bowling & Beehr (2006)**. Harassment from victim's perspective meta. [→ environmental ρ]
- **Roberts & DelVecchio (2000)**. Rank-order consistency meta. [→ stability r=.74]
- **Specht et al. (2011)**. SOEP N=14,718 stability. [→ mid-life stability]
- **Ashton & Lee (2007)**. HEXACO model. [→ trait taxonomy]
- **Wakabayashi (2014)**. Japanese HEXACO-60. [→ Japanese measurement]
- **Tou et al. (2017)**. Workplace Power Harassment Scale. [→ measurement]
- **Kobayashi & Tanaka (2010)**. Gender Harassment Scale. [→ measurement]
- **Shimotsukasa & Oshio (2017)**. SD3-J. [→ Dark Triad measurement]
- **Lee & Ashton (2005)**. HEXACO ↔ Dark Triad correlations. [→ trait inter-correlation]
- **Berry, Carpenter & Barratt (2012)**. CWB self-other meta. [→ self-report defense]
- **Anderson & Bushman (2002)**. GAM. [→ self-report → real behavior]

### 13.6 Personality upstream of SES

- **Heckman, Stixrud & Urzua (2006)**. Noncognitive skills predict outcomes. [→ Section 1.5]
- **Grijalva et al. (2015)**. Narcissism → leadership emergence meta. [→ Section 1.5]
- **Roberts et al. (2007)**. The power of personality. [→ Section 1.5]

### 13.7 Japanese context

- **MHLW (2021)** R2 職場のハラスメント実態調査 [→ primary validation]
- **MHLW (2024)** R5 実態調査 [→ post-law validation]
- **Tsuno et al. (2010)** Japanese NAQ-R. [→ measurement]
- **Tsuno et al. (2015)** Socioeconomic determinants Japan national rep. [→ Section 1.5, 4.2]
- **Tsuno & Tabuchi (2022)**. Bullying → SPD PR=3.20. [→ f2 anchor]

### 13.8 Self-citation hub

- **Tokiwa et al.** Clustering paper (IEEE 掲載済). [→ N=13,668 7-type centroid]
- **Tokiwa et al.** Harassment preprint. [→ N=354 HEXACO + Dark Triad regression]

---

## 14. Version Log & Implementation Checklist

### 14.1 Version log

| Version | Date | Changes |
|---|---|---|
| **v1.0 draft** | 2026-04-29 | Initial draft based on research plan v6/v7 (1,458 行) + D13 power analysis (209 行) + 40 paper deep reading. Pending OSF registration. |

### 14.2 Pre-registration submission checklist

- [N/A] **Internal review**: Co-author confirmation — **sole-authored study につき不要**
- [ ] **Self-review against Nosek 2018 9 challenges**: Section 0.2 全埋
- [ ] **Self-review against Munafò 2017 5 themes**: Section 8 全埋
- [ ] **English translation**: `D12_pre_registration_OSF.en.md` 作成
- [x] **OSF account**: 既存 (metaanalysis/ で使用) → **新規 project 作成のみ pending**
- [ ] **OSF Standard Pre-Registration template**: Section 1–6 を OSF Web フォームに転記
- [ ] **PDF supplementary**: 本ドキュメント (日本語版) を OSF に attach
- [ ] **DOI 取得**: 登録後 DOI を本ドキュメント Header に記録
- [ ] **GitHub mirror**: `simulation/docs/pre_registration/` を public commit
- [x] **Funding & affiliation**: Section 12.2 fill-in 完了（SUNBLAZE Co., Ltd. / no external funding for simulation phase）
- [ ] **Anti-screening triple-lock**: Section 9.1 三箇所すべて準備済
- [ ] **Independent methodologist contact**: Section 8.1 review 依頼確定

### 14.3 Stage 0 着手前の最終確認

以下がすべて完了した時点で本 pre-reg を **lock**、Stage 0 コード実装を開始:

- [ ] OSF DOI 取得済
- [ ] 本ドキュメントの Header に DOI 追記済
- [N/A] Co-author sign-off — sole-authored につき不要
- [ ] Independent methodologist による Section 5 (Analysis Plan) review 完了
- [ ] Repository structure (Section 8.2) 初期化
- [ ] `make reproduce` skeleton 作成済
- [ ] Random seed (20260429) を全 stage で hard-code 確認

---

## 15. End-of-Document Statement

本ドキュメントは Pre-registration の **internal master draft** である。OSF への正式登録 (DOI 取得) を以て本 pre-reg は **lock** され、Stage 0 コード実装が解禁される。本ドキュメントの修正は Section 6.5 の Level 3 (analysis plan revision) 手続に従う。

**Lock 状態**: ⏳ DRAFT (pending internal review and OSF submission)
**Next step**: Section 14.2 checklist の internal review item から順次実施

---

**End of D12 OSF Pre-Registration v1.0 draft.**



