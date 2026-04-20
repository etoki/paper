# Data Extraction Template — 使用方法

`data_extraction.csv` は primary studies からのデータ抽出テンプレート。

## ファイル構成

- **data_extraction.csv**: 各 study を 1 行とし、`literature_review.md` の ID 体系と対応
- **data_extraction_README.md**: 本ファイル（記入手順）

## 記入済みの初期フィールド

以下は既知情報として記入済み:

- `study_id`, `first_author`, `year`, `country`, `journal`, `volume`, `issue`, `pages`, `doi`
- `n_total` / `n_analyzed`（既知のみ）
- `education_level`
- `modality`（判明分）
- `personality_instrument`（判明分）

## 記入すべきフィールド

PDF を開いて以下を埋める:

### Sample 情報
- `n_total`: 募集サンプル
- `n_analyzed`: 分析に使用したサンプル
- `age_mean`, `age_sd`, `age_range`: 年齢情報
- `pct_female`: 女性割合（%）
- `education_level`: K-12 / Undergraduate / Graduate / Adult 等
- `sampling_method`: convenience / random / stratified

### Design
- `design`: cross-sectional / longitudinal / experimental
- `modality`: fully_online / blended / MOOC / LMS / TBC
- `modality_subtype`: synchronous / asynchronous
- `duration`: 学習期間（週数/ヶ月等）
- `subject_domain`: 全科目 / 特定科目
- `platform_name`: StudySapuri / Zoom / Moodle / Canvas 等

### Personality measurement
- `personality_item_count`: BFI-2 なら 60 等
- `alpha_O`, `alpha_C`, `alpha_E`, `alpha_A`, `alpha_N`: 各特性の Cronbach α
- `facet_level_reported`: Y/N（下位ファセット報告ありか）

### Outcome measurement
- `outcome_type`: GPA / exam_score / course_grade / LMS_behavior / satisfaction / engagement
- `outcome_instrument`: 測定機器（e.g., 大学の公式 GPA, 研究者作成スコア）
- `outcome_reliability`: reliability が報告されていれば

### Effect sizes（最重要）
- `r_O_outcome`, `r_C_outcome`, `r_E_outcome`, `r_A_outcome`, `r_N_outcome`: Pearson r
- `n_for_correlations`: 相関計算に使われた N（全体 N と同じ場合が多いが異なることあり）
- `p_value_O` ... `p_value_N`: 有意水準
- `beta_O` ... `beta_N`: 標準化回帰係数（もし r 報告なし時）
- `effect_size_type`: r / rho / beta / d

### Context
- `era`: pre-COVID (2003-2019) / COVID (2020-2022) / post-COVID (2023-)
- `risk_of_bias_score`: 後工程（JBI チェックリスト後に記入）

### Notes
- `notes`: 抽出時の気付き、限界、不明点

## 記入時のルール

### 効果量報告パターン別の対応

1. **Pearson r が相関行列で報告**（最頻出）→ `r_*_outcome` にそのまま記入
2. **Spearman ρ**（Tokiwa 2025 等）→ r 欄に記入し、notes に「Spearman」明記
3. **β のみ報告（相関なし）**→ `beta_*` 欄に記入、`r_*_outcome` は空欄
4. **SEM path coefficients のみ**→ path を `beta_*` に記入、notes に「SEM latent」明記
5. **t-test / F / d 報告**→ `notes` に数値を書き、変換は後工程

### 欠損値の扱い

- 不明 / 未報告 → 空欄（NA と書かない）
- 報告されているが 0 → `0` と明記
- 該当なし（e.g., facet なしの研究の facet 欄）→ 空欄

### 複数アウトカムがある研究

複数のアウトカム（例: 成績 + 満足度）がある場合、追加行として複製:
- `A-04_achievement`, `A-04_satisfaction` のように suffix で ID を派生
- または別シートに分けて管理

## 推奨ワークフロー

1. PDF を PDF viewer で開く
2. Abstract, Methods, Results を速読
3. Table 1（descriptive stats）から Sample 情報
4. Table 2 または 3（相関行列）から効果量
5. Reliability（Cronbach α）は Method セクション
6. CSV に記入
7. 不明点は `notes` に記載して次に進む（完璧主義で止まらない）

## 優先順位

先に抽出する順序（N が大きい / 効果量豊富）:
1. A-28 Yu (2021) — N=1,152
2. A-26 Wang (2023) — N=1,625
3. A-27 Wu & Yu (2024) — N=1,004（PDF 到着後）
4. A-20 Mustafa (2022) — N=718
5. A-19 MacLean (2022) — N=465（HEXACO マッピング必要）
6. A-17 Kara (2024) — N=437
7. A-22 Quigley (2022) — N=301, 全 Big Five 報告
8. A-23 Rodrigues (2024) — N=287
9. A-02 Alkis & Temizel (2018) — N=316
10. A-25 Tokiwa (2025) — N=103（自己論文）

## 記入完了後のチェック

- [ ] 全 primary study で `r_*` 欄のいずれか 1 つ以上が埋まっているか
- [ ] `n_for_correlations` が全て記入されているか（pooling 時に重み付けで使用）
- [ ] `effect_size_type` が全て明記されているか
- [ ] `modality` TBC が確定したか（A-09, A-10, A-13, A-16 要確認）
- [ ] HEXACO 研究（A-19）の Big Five マッピングが完了したか
- [ ] サンプル重複がないか確認（Baruth & Cohen シリーズ、Audet シリーズ）

## 次のフェーズ

CSV が完成したら:
1. R `metafor` パッケージで読み込み
2. `escalc()` で効果量標準化
3. `rma()` で random-effects meta-analysis
4. Forest plot / Funnel plot 作成
5. Moderator analysis / meta-regression
