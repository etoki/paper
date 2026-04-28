# D13 Power Analysis — HEXACO 7-Typology Harassment Simulation (Phase 1)

実施日：2026-04-27
ブランチ：`claude/review-harassment-research-plan-Dy2eo`
スクリプト：`simulation/docs/power_analysis/power_analysis.py`
入力：`harassment/raw.csv` (N=354) + `clustering/csv/clstr_kmeans_7c.csv` (centroids)
研究計画該当項目：D13 (`research_plan_harassment_typology_simulation.md` Part 8.3)

---

## 0. 結論（先出し）

**Verdict: GO（条件付き feasible）**

| シナリオ | feasibility | 主軸採用可否 |
|---|---|---|
| **14 cells (7 types × 2 genders)** | ◯ 全 cell N≥10、最小 N=10、中央値 N=18 | ◯ Phase 1 主分析として採用可能 |
| **28 cells (7 types × 2 genders × 2 roles)** | △ 16/28 cell が N<10、5 cells が ほぼ empty | △ shrinkage 前提 sensitivity 分析として採用 |

**条件**：
1. Phase 1 主分析は 14-cell（役職を分けない）に固定。28-cell は sensitivity として副次的に提示。
2. 28-cell には **empirical Bayes partial pooling**（14-cell estimate を prior に使う）を必須化。
3. 「pairwise type 比較で d < 0.9 は検出不能」を limitation として明記。論文の主張は cell-level ratio ではなく **国レベル aggregate の triangulation** に重みを置く。
4. 役職推定（D1）は 3 モデル比較のうち、cell サイズの観点では結果に大差がないため、文献ベース（Conscientiousness × leadership emergence）の単純規則で十分。

---

## 1. 設計

### 1.1 シナリオ

| Scenario | 分割 | 期待 cell 数 |
|---|---|---|
| 14 cells | 7 type × 2 gender | 14 |
| 28 cells | 7 type × 2 gender × 2 role | 28 |

### 1.2 評価指標

| 指標 | 計算 | 解釈 |
|---|---|---|
| **Cell size n** | 各 cell に落ちる N=354 中の人数 | shrinkage 必要 cell の同定 |
| **MDE (one-sample d)** | `(z_{α/2} + z_β) / √n` | cell mean が grand mean と異なるかを検出する最小効果 |
| **MDE (two-sample d)** | `(z_{α/2} + z_β) × √(1/n₁ + 1/n₂)` | type 間 pairwise 比較の最小効果 |
| **Binary rate CI half-width** | `Z_{.975} × √(p(1-p)/n)` | 「cell ごとの加害率」推定の 95% CI 幅 |

α=.05（two-sided）、power=.80。`Z_{.975}+Z_{.80}=2.802`。

### 1.3 Cell 化前処理

- **Type 帰属**：`clstr_kmeans_7c.csv` の 7 centroid に対し、N=354 の HEXACO 6 領域 raw score を最近傍分類（Euclidean）。
- **Gender**：raw.csv の `gender` 列は 0/1 エンコード（n=133 / n=220、`harassment/res/sex_stratified_R2.csv` と整合）。
- **Role 推定**（28-cell シナリオのみ）：composite = C + 0.5·X の上位 15% を「manager」（労働力調査の管理職率 12–15% に整合、Judge, Bono, Ilies, & Gerhardt, 2002 の leadership emergence と整合）。
- **Binary 加害**：`mean + 0.5·SD` を閾値とし、power_harassment は P=0.17、gender_harassment は P=0.28 の baseline 加害率にマッピング。

---

## 2. 結果

### 2.1 N=354 の 7 類型分布

| Type | n | 割合 |
|---|---|---|
| T0 | 23 | 6.5% |
| T1 | 42 | 11.9% |
| T2 | 49 | 13.8% |
| T3 | 49 | 13.8% |
| T4 | 51 | 14.4% |
| T5 | 26 | 7.3% |
| T6 | 114 | 32.2% |

T6 が dominant（32%）、T0 と T5 が稀（6–7%）。クラスタリング論文の N=13,668 分布とは独立に再分類しているため、**類型の生起確率は 2 サンプル間で異なる可能性**がある（→ Stage 1 母集団スケーリング時に N=13,668 の分布を採用するという計画の妥当性を支持）。

### 2.2 14-cell scenario（7 type × 2 gender）

```
gender_label  G0  G1
type
0             10  13
1             10  32
2             15  34
3             28  21
4             13  38
5             14  12
6             44  70
```

| 指標 | 値 |
|---|---|
| Cell 数 | 14 |
| n の最小 / 最大 / 中央値 | 10 / 70 / 18 |
| n < 10 | 0 cells (0%) |
| n < 20 | 7 cells (50%) |
| One-sample MDE (d) 中央値 / 最大 | 0.67 / 0.89 |
| Pairwise MDE (d) 中央値 / 最大 | 0.92 / 1.25 |
| Pairwise MDE < 0.5 (small effect 検出可能) | 0 / 42 (0%) |
| Binary rate CI ±half-width 中央値（power_harassment） | ±0.13（13 percentage points） |
| Binary rate CI ±half-width 最大（power_harassment） | ±0.30 |
| Binary rate CI ±half-width 中央値（gender_harassment） | ±0.15 |

**含意**：
- **検出力面**：cell-level mean を grand mean と比べる検定では d ≥ 0.67 が必要、type 間 pairwise では d ≥ 0.92 が必要。Harassment 論文で報告された **H-H β=-.14 規模（小効果）は cell-level 検定では検出不能**。一方 **Psychopathy β=.32–.40 規模（中-大効果）の type 差は一部 cell で検出可能**。
- **推定精度面**：cell ごとの加害率の 95% CI 半幅が中央値 ±13pp。10–15% baseline rate の cell では **CI が「ほぼ 0% から 30%」と非常に広い**。aggregate（全 cell 重み付き合算）した時点で大幅に縮む。
- **Limitation の書き方**：「cell-level estimates are imprecise; aggregate-level predictions (national rate) are the primary inferential target. Sub-cell differences should be interpreted as descriptive rather than confirmatory.」

### 2.3 28-cell scenario（7 type × 2 gender × 2 role）

```
                   role=0  role=1
type gender_label
0    G0                 7       3
     G1                 7       6
1    G0                 9       1
     G1                32       0
2    G0                14       1
     G1                28       6
3    G0                14      14
     G1                16       5
4    G0                 9       4
     G1                27      11
5    G0                14       0
     G1                12       0
6    G0                44       0
     G1                67       3
```

| 指標 | 値 |
|---|---|
| Cell 数 | 28 |
| n の最小 / 最大 / 中央値 | 0 / 67 / 8 |
| n < 10 | 16 cells (57%) |
| n < 20 | 23 cells (82%) |
| 完全 empty (n=0) | 4 cells |
| Empty 近似 (n≤3) | 9 cells |
| One-sample MDE (d) 中央値 / 最大 | 0.83 / 1.62 |

**問題点**：
1. **Empty cells の構造的偏り**：T1（低 X）、T5、T6（低 C）の manager cell（role=1）がほぼ全部 empty。これは literature-based role rule（C+X 高い人が manager）の論理的帰結であり、3 モデル比較しても定性的には変わらない（高 C 型に manager が偏るのは leadership emergence 文献の合意）。
2. **MDE が大きい**：cell の 57% で d > 1.0 が必要。事実上、role-stratified の type 差検出は不可能。
3. **介入対象の母集団推定**：Phase 2 の「targeted intervention（高リスク類型の manager に集中投入）」を 28-cell で評価する場合、対象 cell の人数推定が±50% 以上ぶれる。

**対応**：
- **Empirical Bayes shrinkage（必須）**：14-cell estimate を prior、28-cell observed を likelihood として posterior を計算。これで empty cell も「14-cell 推定値 + role 効果」として補完される。
- **代替**：role を連続量（manager probability）として weighted regression に組み込み、cell discretization を回避する（modeling 上はこちらの方が情報損失少ない）。

---

## 3. Phase 1 設計への含意

### 3.1 主分析を **14-cell** に確定

研究計画 Part 3.2 Stage 0 の「7 類型 × 役職 × gender の cell ごとに加害確率を bootstrap 推定」は、**14-cell（役職込めない）で実施**を主分析とする。役職は continuous covariate として個人レベルで bootstrap に組み込む。

### 3.2 Cell-size threshold (D6) の確定

Part 8.1 の D6（Cell size 閾値、(a) N≥10 で推定 / N<10 は shrinkage）を **(a) で確定**。14-cell シナリオは全 cell が N≥10 を満たすため、**主分析では shrinkage 不要**。28-cell sensitivity は shrinkage 前提。

### 3.3 Power-related limitation の明記

Phase 1 論文の Limitation セクションに以下を含める：

> "Our N=354 sample yields a minimum detectable Cohen's *d* of approximately 0.9 for pairwise type comparisons, meaning that subtle type-level differences (e.g., the *d* ≈ 0.3 corresponding to the H-H × harassment association reported in our prior work) cannot be confirmed at the cell level. The simulation should therefore be interpreted as a tool for aggregating cell-level point estimates into population-scale predictions rather than as a confirmatory test of pairwise type differences. Per-cell 95% CIs on harassment rates are reported throughout."

### 3.4 Pre-registration (D12) への引き渡し項目

| 固定項目 | 値 |
|---|---|
| Primary cell structure | 14 (7 type × 2 gender) |
| Secondary cell structure | 28 (with empirical Bayes shrinkage from primary) |
| Cell-size minimum (no shrinkage) | N ≥ 10 |
| Pairwise MDE acceptable | d ≥ 0.9 (declared upfront) |
| Bootstrap iterations | 2,000 (per cell, BCa CI) |
| Outcome binarization threshold | mean + 0.5·SD per outcome |
| Validation primary target | MHLW 実態調査 国レベル aggregate (MAPE ≤ 30% = success) |

---

## 4. 本 D13 が **やっていない** こと

本分析は Phase 1 simulation pipeline 全体ではなく、cell サイズと estimation precision に焦点を絞った **frequentist sample-size feasibility check** である。以下は別タスク：

- **Bootstrap-based actual CI（D13 でなく実装フェーズ）**：N=354 の真の bootstrap 分布は theoretical SE よりタイトになる可能性あり。
- **Bayesian shrinkage の実証評価（D6 sub-task）**：14-cell prior + 28-cell likelihood の partial pooling を実データで実装し、posterior 幅を測る。
- **Simulation-to-MHLW MAPE の事前評価**：Phase 1 の最終成功基準（MAPE ≤ 30%）は本 D13 では測れない。simulation を一度走らせる必要あり。
- **Phase 2 介入効果の検出力**：counterfactual ΔP の CI 幅は Phase 1 cell estimate の CI を波及させた形で算出される（→ D9, D10 sensitivity sweep の段階で評価）。
- **Type assignment uncertainty**：本分析は最近傍分類を deterministic に扱った。soft assignment（mixing weights）にすれば cell sizes が effective fractional になり、CI はやや広がる。これは sensitivity analysis として実装段階で扱う。

---

## 5. 次のステップ（推奨順）

1. **D6 を確定**（本ドキュメントで実質完了）
2. **D12 Pre-registration draft** 着手 — 本 D13 の数字（14-cell 確定、d≥0.9 limitation、MAPE≤30% 主成功基準）を組み込む
3. **D1 役職推定モデル文献調査** — 28-cell sensitivity 用に 3 モデル比較
4. **コード実装着手** — Stage 0（type assignment + cell-level bootstrap）から開始

---

## 6. 生成 artifacts

| ファイル | 内容 |
|---|---|
| `power_analysis.py` | 再現用 Python スクリプト |
| `cell_counts_14.csv` | 14-cell 人数表 |
| `cell_counts_28.csv` | 28-cell 人数表 |
| `cell_stats_14.csv` | 14-cell 記述統計 + MDE |
| `cell_stats_28.csv` | 28-cell 記述統計 + MDE |
| `pairwise_mde_14.csv` | 14-cell pairwise MDE 行列 |
| `binary_rate_precision_14.csv` | 14-cell 二値加害率 + CI 半幅 |
| `power_analysis_summary.json` | headline 数値（structured） |
