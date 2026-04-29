# 研究計画：HEXACO 類型ベースの職場ハラスメント simulation（Phase 1 + Phase 2）

作成日：2026-04-27
**最終更新：2026-04-29**（Tier 1/2/3 文献基盤確立、致命的弱点解消、framing 更新後）
ブランチ：`main`

関連ドキュメント：
- `simulation/docs/notes/simulation_paper_evaluation_integrated.md`（Doc 1：方法論評価）
- `simulation/docs/notes/research_vision_integrated.md`（Doc 2：研究ビジョン）
- `simulation/docs/literature_audit/literature_audit.md`（8 pillar 文献監査）
- `simulation/docs/literature_audit/tier1_search_results.md`（Tier 1 取得文献 26 件）
- `simulation/docs/literature_audit/tier2_search_results.md`（Tier 2 取得文献 12 件）
- `simulation/docs/literature_audit/deep_reading_notes.md`（**40 件 deep reading 統合、3,950 行**）
- `simulation/docs/power_analysis/D13_power_analysis.md`（D13 power analysis、14-cell 設計確定）

---

## Part 0：本ドキュメントの位置づけ

### 0.1 目的

これまでの議論を統合し、**実装可能な研究計画**として固定する。

これまで作成したドキュメントとの関係：
- **Doc 1（評価フレーム）**：simulation 論文の質を測る基準
- **Doc 2（研究ビジョン）**：「自己責任ではなく社会システムの問題」の主張
- **本ドキュメント（研究計画）**：上記 2 つを統合した**具体的研究プロジェクト**

### 0.2 本研究の特徴

- **新規データ収集なし**（既存 N=354 + N=13,668 を再利用）
- **LLM 不使用**（古典的確率モデル + Monte Carlo + Empirical Bayes shrinkage）
- **倫理審査軽負荷**（既存 IRB 通過データのみ）
- **2 段階構成**：Phase 1（実証 simulation）→ Phase 2（介入 counterfactual）
- **既存自己引用 hub**：
  - Tokiwa et al. **Clustering 論文（IEEE 掲載済）**：N=13,668、HEXACO 7 類型
  - Tokiwa et al. **Harassment 論文（preprint）**：N=354、HEXACO + Dark Triad × 加害
- **Tier 1+2+3 文献基盤**：**40 件 deep reading + 約 20 件 既存ライブラリ補完**（合計 60 paper foundation）

### 0.3 Doc 1 のフレームでの評価サマリー（再掲）

| 観点 | 本研究計画 |
|---|---|
| L1 再現性 | ◎ 厚労省統計と直接照合可能 |
| L4 機構透明性 | ◎ 確率テーブル（black box でない） |
| 戦略 1（集団予測） | ◎ 国レベル aggregate |
| 戦略 3（triangulation） | ◎ 厚労省 4 種データソース |
| LLM 系問題（A-D） | ✗ 該当しない（LLM 不使用） |
| Cost | ◎ ほぼゼロ |

---

### 0.4 重要な区別：研究計画 vs 論文 vs 著作のメッセージ階層

本ドキュメントは **研究計画（internal document）** であり、研究の動機・思想（capability approach、社会システムの責任、自己責任批判等の規範的主張）を明文化している。

しかし、**実際の論文（peer-reviewed publication）には L3/L4 の規範主張を含めない**。これは Doc 2 Part 2 の **4 層分離原則**に基づく方法論的判断である。

#### 各 format で許容されるメッセージ層

| Format | 許容される層 | 例 |
|---|---|---|
| **本研究計画（internal）** | L1 + L2 + L3 + L4 すべて | 動機・思想含む包括的記述 |
| **論文（Phase 1, 2）** | **L1（記述・予測）に純化** | 「類型 X の加害率は Y%」「介入 B で Z% 削減と予測」 |
| **著作 / エッセイ** | L1 + L3 + L4 の統合 | 「自己責任 framing の限界と社会の責任」 |

#### 論文で言えること / 言えないこと

##### ✓ 論文で言えること（L1 empirical claim）
- 「Type X の加害確率は他より Y% 高い」
- 「Counterfactual B（targeted intervention）は加害率を Z% 削減と予測」
- 「Structural-only intervention（Pruckner 2013 系）は extensive margin に効果が乏しい」
- Discussion で「These findings may inform future research on intervention design」（modest implication）

##### ✗ 論文で言えないこと（L3/L4 normative claim）
- 「自己責任 framing は誤り」
- 「社会には変化機会を提供する道徳的義務がある」
- 「workplace harassment は本質的に加害者の責任ではない」
- 「社会システムが責任を負うべき」

#### この分離が必要な理由

1. **査読耐性**：empirical claim と normative claim が混在すると "ideologically motivated" "outside scope" として却下されやすい
2. **論理的厳密性**：シミュレーションは L1（記述・予測）まで支持。L3/L4 への飛躍は empirical 根拠を超える
3. **長期的信頼性**：L3/L4 主張を論文に込めると、後の論文も "あの著者は論文を主張の道具にする" と見られる
4. **影響力の最大化**：L1 を堅実に published → 著作で L3/L4 を展開、の方が**読者層が広がる**

#### Discussion section での慎重な書き方（論文向け）

論文の Discussion で触れていい limit：

- ✓ 「Our findings are consistent with capability-based perspectives (Sen, 1999; Nussbaum, 2011)」 — 既存文献への positioning
- ✓ 「These results may have implications for personality-based intervention design」 — modest
- ✓ 「Future research could examine X」 — future direction
- ✗ 「This proves that...」 — 強い断定はしない
- ✗ 「Society should...」 — 規範を直接言わない

→ **「示唆する」「整合的である」「将来研究を促す」レベルに留める**

---

## Part 1：背景と動機

### 1.1 研究ビジョンからの逆算（Doc 2 より）

> **「個人 outcome は遺伝と環境の関数として確率的に予測可能であり、性格特性はその合成シグナルとして介入の手がかりになる。資源（収入・教育）が部分的に zero-sum である現実において、非ポジショナルな性格特性への集団介入こそが社会全体の幸福を底上げする最もスケーラブルな戦略である。」**

### 1.2 主張の中核

> **「特定の性格タイプ集団が特定の社会問題を起こしやすい・起こされやすい場合、それは個人の自己責任ではなく社会システムの設計の問題である。」**

### 1.3 既存研究との関係

#### Clustering 論文（**IEEE 掲載済**、N=13,668）
- 日本人の HEXACO 7 類型を多手法クラスタリング（Ward / k-means / spectral）で同定、cross-method validation 済
- Reserved / Emotionally Sensitive / Exploratory Extravert / Conscientious Introvert / Self-Oriented Independent / Emotionally Volatile Extravert / Reliable Introvert
- **本研究の Stage 0 入力**として使用

#### Harassment 論文（Preprint 改訂版、N=354）
- HEXACO + Dark Triad → 加害傾向の HC3-robust 階層回帰
- Power harassment：Psychopathy β=.32–.40、H–H β=−.14
- Gender harassment：H–H β=−.23、Openness β=−.24
- **本研究の Stage 1 確率テーブル推定**に使用

### 1.4 ★ 中核的 framing：Latent vs Expressed Prevalence

**重要**：本研究の central conceptual structure：

| 測定 | 性質 | 意味 |
|---|---|---|
| **N=354 personality** | Stable trait（Roberts & DelVecchio 2000、Specht et al. 2011 で r=.74 plateau）| 法律変化の影響を受けない |
| **N=354 harassment scale** | Behavioral disposition / propensity（Tou 2017 + Kobayashi 2010 の "trait-style" items）| 環境条件下で expressed される **latent propensity** |
| **MHLW survey** | 過去 3 年間の **expressed incident** | Environment（法律、組織文化、社会規範）による gating 後 |

#### 因果構造

```
Personality（HEXACO type、stable）
        ↓
Latent harassment propensity（N=354 で測定）
        ↓
Environmental gating（法律・組織風土・社会規範）
        ↓
Expressed harassment incidence（MHLW で観測）
```

#### 含意（論文 framing の core）

1. **Phase 1 simulation は latent prevalence を予測** → MHLW expressed prevalence と triangulate → **gap = environmental moderation**
2. **MHLW 2016（32.5%、pre-law）→ MHLW 2024（19.3%、post-law）の −12.1pp は environmental gating の自然実験**（Counterfactual C 系統の natural experiment）
3. **Phase 2 Counterfactual A/B（personality-based）は latent propensity 自体を変化** ＝ upstream intervention
4. **Phase 2 Counterfactual C（structural）は expression rate を抑制** ＝ downstream gating

→ 本研究の novelty：**latent propensity を personality typology から projection できる初の研究**

### 1.5 Personality は SSS / 職位 / キャリアの上流共通原因（Tier 2 で確立）

伝統的 framing：「個人 personality vs 環境（SSS）」を **competing predictors** として比較

本研究 framing：**personality は SSS の上流**

```
[遺伝・初期環境]
        ↓
[Personality]   ← Tier 2 D で確立
   （HH、Narcissism、Conscientiousness 等）
   ↙        ↓        ↘
[キャリア   [対人行動  [生活習慣
 選択]      パターン]   選択]
   ↓                      ↓
[職位・SSS]           [健康・婚姻]
   ↘        ↓        ↙
       [Harassment]
```

**支持文献**：
- Heckman, Stixrud, & Urzua (2006) — noncognitive skills が schooling/wages/occupation を強く予測
- Roberts et al. (2007) "The Power of Personality"
- Grijalva et al. (2015) — narcissism → leadership emergence ρ=+.16
- Lee & Ashton (2005) — HEXACO HH ↔ Dark Triad r=−.53 to −.72
- Tsuno et al. (2015) — SSS effect OR=4.21（personality 統制せず）→ personality が部分 mediate

**含意**：本研究の cell-conditional probability は **direct + indirect (SSS-mediated) effects の合計**を capture。Phase 2 personality intervention は **upstream leverage point**。

### 1.6 なぜこの設計か

#### サンプル収集なしで simulation 論文として成立する条件
1. ✅ **既存データの再利用**（N=354 + N=13,668）
2. ✅ **公開統計との照合可能性**（厚労省 4 種データソース）
3. ✅ **方法論的新規性**（type-conditional simulation の harassment への初応用）
4. ✅ **論理透明性**（LLM black box でない確率モデル）
5. ✅ **Causal framing**（Hernán & Robins 2020 target trial emulation、Pearl 2009 SCM）
6. ✅ **EB shrinkage by established methodology**（Casella 1985、Clayton & Kaldor 1987、Efron 2014、Greenland 2000）

#### 候補 A/B/C との比較（Doc 1 評価）
- 候補 A（Harassment LLM）：safety risk 致命的
- 候補 B（Clustering LLM）：tautology risk
- 候補 C（A+B）：両方の問題を継承
- **本案**：これらの問題群を**構造的に回避**

---

## Part 2：リサーチクエスチョン

### 2.1 Phase 1（実証 simulation）

#### Main RQ（latent/expressed framing 反映、更新版）
**「N=354 と N=13,668 から構築した HEXACO 7 類型ベースの確率モデルが予測する latent harassment prevalence は、日本の MHLW 全国ハラスメント統計（expressed prevalence）をどの程度再現できるか？ Latent と expressed の gap は environmental gating の magnitude をどう示唆するか？」**

#### Sub-RQs（更新版）
- **RQ1.1**：7 類型 × gender の **14 cell**（D13 で確定、役職は continuous covariate に格上げ）ごとの加害 propensity はどう異なるか？
- **RQ1.2**：階層 baseline（B0 random / B1 gender / B2 HEXACO 線形 / B3 7 類型 / **B4 = B3 + age + 業種推定 + 雇用形態** = personality + 周辺変数）と比較して、7 類型モデル（B3）と拡張 model（B4）の informativeness gain は？
- **RQ1.3**：simulation 出力（latent prevalence）は **MHLW 2016（pre-law、32.5%、主 target）/ 2020 R2（移行期、31.4%、副）/ 2024 R5（post-law、19.3%、副）** のどれと最も合致するか？ Gap は environmental gating の証拠か？
- **RQ1.4**：simulation の予測誤差は、どの cell（類型・gender）で最大か？（failure mode の特定）

### 2.2 Phase 2（介入 counterfactual、causal framing 強化版）

#### Main RQ
**「既存の HH 改善介入研究の効果量を anchor として、target trial emulation framework（Hernán & Robins, 2020）下で、population-wide / targeted / structural の 3 種介入は、それぞれ harassment expressed prevalence をどの程度削減すると予測されるか（明示的 transportability 仮定下で）？」**

#### Sub-RQs
- **RQ2.1**：Universal HH 介入（全人口を対象とする教育・訓練）の population-level 効果は？（Counterfactual A、Kruse 2014 anchor）
- **RQ2.2**：Targeted 介入（高リスク類型のみに集中投入）の効果と cost-effectiveness は？（Counterfactual B、**主軸**、Hudson 2023 anchor）
- **RQ2.3**：Structural 介入（個人 personality 不変、組織 / 制度のみ変更）の効果は？（Counterfactual C、Pruckner 2013 + Bezrukova 2016 + Dobbin & Kalev 2018 で **30% 削減を保守的上限**）
- **RQ2.4**：MHLW 2016 → 2024 の 12.1pp 自然減は **Counterfactual C 系統の natural experiment** として、本研究 Counterfactual C 予測値とどの程度整合するか？
- **RQ2.5**：3 種介入のうち、どれが最も plausible に harassment 削減に寄与すると予測されるか？

### 2.3 共通 RQ（両 Phase に渡る）

- **RQ-X**：本シミュレーションの限界（self-report 依存、N=354 代表性、cross-sectional 因果、CMV、Phase 2 transportability 等）が結果にどう影響するか sensitivity analysis で評価

---

## Part 3：Phase 1 設計（実証 simulation）

### 3.1 データ層

#### 既存データ（再利用）
| データ | N | 含まれる変数 | 役割 |
|---|---|---|---|
| **Clustering 論文データ** | 13,668 | HEXACO 6 領域、簡易デモグラ | 7 類型 centroid と分布の推定 |
| **Harassment 論文データ** | 354 | HEXACO 6 領域、Dark Triad 3、power harassment、gender harassment、age、gender、area | type-conditional 加害確率の推定 |

#### 生成変数
| 変数 | 生成方法 | 根拠 |
|---|---|---|
| **7 類型 membership**（N=354 各個人） | 13,668 から得た centroid に N=354 を最近傍分類 | クラスタリング論文の centroid を transfer |
| **役職（管理職 / 一般）** | personality（特に Conscientiousness, Extraversion）から確率的予測 | 文献的に C・E が leadership emergence と関連 |

#### 必要な外部データ（公開統計）
- 厚労省「職場のハラスメントに関する実態調査」（2020 年度版）
- 厚労省「個別労働紛争解決制度の施行状況」
- 厚労省「雇用動向調査」（離職理由別）
- 厚労省「労働安全衛生調査」（メンタル疾患関連）

### 3.2 シミュレーション層

#### Stage 0：類型分類と確率テーブル構築（**D13 確定 + EB shrinkage 仕様**）
1. N=13,668 の HEXACO data から 7 類型 centroid を取得（クラスタリング論文済み、IEEE 掲載済）
2. N=354 の各個人を最近傍 centroid に分類 → 7 類型 membership 付与
3. **主分析：7 類型 × gender = 14 cell**（D13 で確定、最小 N=10、全 cell N≥10）
4. **副分析：7 類型 × gender × role = 28 cell**（cell N<10 が 16 cells で、empirical Bayes shrinkage 必須）
5. 役職は **continuous covariate**（Conscientiousness × Extraversion 文献ベース linear、D1 で 3 モデル比較は副次）
6. **Bootstrap 2,000 iter / cell** で BCa CI（Casella 1985 base、Clayton & Kaldor 1987 analog）
7. **EB shrinkage 仕様（Beta-Binomial conjugate prior、Casella 1985 + Clayton & Kaldor 1987 + Efron 2014 + Greenland 2000）**：
   - 14-cell 分布から method of moments で α, β を estimate
   - 28-cell posterior：$E[p_i | X_i, N_i] = (\alpha + X_i) / (\alpha + \beta + N_i)$
   - **Sensitivity strength sweep**：scale ∈ {0.5, 1.0, 2.0}（3 値、weak / medium / strong）

#### Stage 1：母集団スケーリング
1. 日本労働人口（約 6,800 万）を仮想母集団とする
2. 13,668 の類型分布を母集団に適用（必要に応じてデモグラ重み付け）
3. 役職分布を厚労省統計に合わせる（労働力調査の管理職率約 12–15%）
4. gender 分布を厚労省労働力調査に合わせる
5. 各 cell に対して bootstrap で**期待 latent 加害者数**を計算（CI 付き）
6. → 出力は **latent prevalence**（environmental gating 前）

#### Stage 2：加害行動の連鎖
1. **加害者数**：Stage 1 で算出（latent）
2. **被害者数**：加害者 1 人あたり平均 V 人の被害者（V は厚労省実態調査から推定）
3. **離職**：被害者の f1 割が離職（f1 は雇用動向調査から推定）
4. **メンタル疾患**：被害者の f2 割が adjustment disorder / depression（f2 は労働安全衛生調査から推定。Tsuno & Tabuchi 2022 の bullying → SPD PR=3.20 を anchor）

#### Stage 3：sensitivity analysis
- **V を [2, 3, 4, 5] でスイープ**
- **f1 を [0.05, 0.10, 0.15, 0.20] でスイープ**
- **f2 を [0.10, 0.20, 0.30] でスイープ**
- **役職予測モデルを 3 種類で比較**（D1：(a) personality 線形、(b) tree-based、(c) 文献ベース）
- **EB shrinkage strength を 3 値スイープ**（0.5, 1.0, 2.0）
- **Binarization threshold を 3 値スイープ**（mean+0.25SD, mean+0.5SD, mean+1.0SD）— Notelaers 2006 continuum framework に対応
- **Cluster K を比較**（K=4, 5, 6, 7, 8）— typology robustness check
- **CMV diagnostic**（Harman's single-factor test、marker variable correction、Podsakoff et al. 2003）
- → robust な結果と fragile な結果を分離

### 3.3 検証層（Triangulation、latent/expressed framing 反映）

| 検証 target | データソース | 比較指標 | 期待される latent vs expressed gap |
|---|---|---|---|
| **加害者率 / 被害者率（主）** | **厚労省実態調査 2016 (32.5%、pre-law)** | 国全体・業種別 | **最 small gap**（環境 gating 最弱） |
| 加害者率（副 1） | 厚労省実態調査 2020 R2 (31.4%) | 国全体・業種別 | 移行期、small gap |
| 加害者率（副 2） | 厚労省実態調査 2024 R5 (19.3%) | 国全体・業種別 | **大 gap**（post-law 環境 gating 強）|
| 国際 baseline | ILO 2022 Asia-Pacific 19.2% lifetime | 国レベル | 比較参照 |
| 被害体験（peer-reviewed） | Tsuno et al. 2015 N=1,546 (6.1%, 30 days) | 周辺分布 | reference period 補正後 |
| 相談件数の時系列 | 個別労働紛争 | 過去 10 年トレンド | environment moderation の動的 evidence |
| ハラスメント由来離職 | 雇用動向調査 | 年次推定値 | （Stage 2 連鎖検証）|
| ハラスメント由来メンタル疾患 | 労働安全衛生調査 + Tsuno & Tabuchi 2022 PR=3.20 | 業種別発症率 | （Stage 2 連鎖検証）|

#### 検証指標
- **Pearson r / Spearman ρ**：simulation 出力と実測の相関
- **KS 距離 / Wasserstein 距離**：分布形状の一致度
- **Mean Absolute Percentage Error (MAPE)**：絶対値の精度
- **Calibration plot**：cell ごとの予測 vs 実測

#### 成功基準（Pre-registration で固定）
- **★ 主 validation target**：**MHLW 2016（32.5%、pre-law）**
- **★ 主成功基準**：MAPE ≤ 30% で「成功」
- **★ 失敗基準**：MAPE ≥ 60% で「失敗」（だが**failure mode を発見化**）
- **△ 中間**：MAPE 30–60% を「partial success」

> 注：MAPE ≤ 30% 閾値は **pre-registered strict criterion**。微妙小ulation field に MAPE convention は確立していないため、本研究で固定値として宣言（Pre-registration document で明示）。

### 3.4 出力

1. 7 類型 × gender の加害 propensity テーブル（**14 cell**、bootstrap CI 付き）
2. 28-cell EB-shrunken 推定（**副分析、sensitivity**）
3. 国レベル predicted vs 実測の比較表（**MHLW 2016 主、2020 + 2024 副**）
4. cell ごとの prediction error map（どこで合うか合わないか）
5. sensitivity analysis 結果（V, f1, f2, EB strength, threshold, K）
6. 階層 baseline との比較（B0–B4、**B4 = personality + age + 業種推定 + 雇用形態**）
7. **CMV diagnostic results**（Harman's single-factor test 等、Podsakoff et al. 2003 準拠）

---

## Part 4：Phase 1 階層 baseline と限界

### 4.1 階層 baseline 設計（**B4 拡張済**、Doc 1 戦略 4 に準拠）

| Baseline | 入力 | 仮説 |
|---|---|---|
| **B0：完全ランダム** | uniform 加害確率 | informativeness ゼロ |
| **B1：gender のみ** | gender × 加害確率 | demographic baseline |
| **B2：HEXACO 6 領域（線形）** | 6 領域 → 線形回帰で加害確率 | trait-level prediction |
| **B3：7 類型モデル（提案手法）** | 7 類型 + gender | type-conditional prediction |
| **B4：拡張モデル**（personality + 周辺変数）| 7 類型 + gender + age + 業種推定 + 雇用形態 | personality + environmental sorting |

#### 評価
- **B3 > B2 > B1 > B0** の単調増加 → 「7 類型モデルが情報を加えている」
- **B3 ≈ B2** の場合 → 「typology は線形より優位ではない」（発見として報告）
- **B3 < B2** の場合 → 「typology が overfitting」（より深刻な発見）
- **B4 ≫ B3** の場合 → 「personality 単独では不十分、周辺変数を含む拡張が必要」（honest reporting）
- **B4 ≈ B3** の場合 → 「personality typology が周辺変数 informationally subsume」（type-conditional probability の latent + indirect 効果包含を支持）

### 4.2 主要な限界（**Tier 1+2+3 文献 evidence 統合版**）

#### ⚠ 限界 1：Self-report → Self-report の循環性
- **問題**：N=354 の加害は self-report、厚労省実態調査も部分的に self-report
- **対応**：
  - 厚労省実態調査の **被害者報告（より客観的）** を主 validation target にする
  - **Berry, Carpenter, & Barratt 2012 CWB meta**：CWB-I（harassment 含）self-other ρ=.51、self が MORE 報告 → social desirability suppression は支持されない
  - **Anderson & Bushman 2002 GAM**：self-reported trait aggressiveness は real-world behavior と correlate → external validity 確立
  - limitation セクションで明記

#### ⚠ 限界 2：N=354 の代表性
- **問題**：クラウドソーシング自己選択 sample。日本労働者を代表しない
- **対応**：
  - デモグラ（年齢、性別）で重み付け補正
  - 13,668 の類型分布を主、N=354 を type-conditional 確率推定にだけ使う
  - sensitivity analysis で代表性仮定の影響を評価
  - **Tsuno et al. 2015 N=1,546 random sample** との triangulation で外部 validity 推定

#### ⚠ 限界 3：役職推定の誤差
- **問題**：N=354 に役職データなく、personality から推定
- **対応**：
  - 推定モデルを 3 種類（線形、tree-based、文献ベース）で比較
  - **D13 結論**：役職を **continuous covariate に格上げ**、cell 分割軸から外す（28-cell より 14-cell が確実）
  - **Tsuno & Tabuchi 2022 COVID 逆転**：manager > non-manager の risk 逆転報告 → 役職は加害／被害両側 risk factor。本研究の Discussion で両側性明示

#### ⚠ 限界 4：Cell-level 統計検出力（power）
- **問題**：N=354 を 7 類型 × gender × 役職に分割すると一部 cell が小（< 10）
- **対応**：
  - ✅ **D13 power analysis 完了**：14-cell 主分析で全 cell N≥10 確保
  - 28-cell 副分析は **empirical Bayes shrinkage 必須**（Beta-Binomial conjugate prior、Casella 1985 + Clayton & Kaldor 1987 + Efron 2014 + Greenland 2000）
  - 推定の不確実性を CI で明示
  - Pairwise MDE d≥0.92 → cell-level inference を avoid、aggregate-level に focus

#### ⚠ 限界 5：被害者倍率 V と離職率 f の不確実性
- **問題**：V, f1, f2 の点推定は文献から、確実性はない
- **対応**：sensitivity analysis（Stage 3）で複数値スイープ
- **f2（メンタル疾患率）の anchor**：Tsuno & Tabuchi 2022 の bullying → SPD PR=3.20 → f2 ≈ 0.30 妥当範囲

#### ⚠ 限界 6：因果主張の不可能性 + reverse causation 懸念
- **問題**：cross-sectional design、reverse causation（harassment 経験 → personality 自己認知変化）が原理的に否定不可
- **対応**：
  - **Roberts & DelVecchio 2000 meta**：personality rank-order consistency r=.74 plateau（age 50–70）→ **personality は数十年単位で stable**、reverse causation の magnitude は small
  - **Specht et al. 2011 SOEP N=14,718**：mid-life（40–60）で stability 最高 → 本研究 N=354（労働者）は最 stable 帯
  - 「再現性 validation 研究」と明記、causal claim を避ける
  - Phase 2（counterfactual）でも **target trial emulation framework**（Hernán & Robins 2020）下で identifying assumptions（exchangeability、positivity、consistency、transportability）を明示

#### ⚠ 限界 7：simulation 内の simplification（GAM situational variables 欠落）
- **問題**：matching、組織風土、リーダーシップ等を捨象。Anderson & Bushman 2002 GAM が要求する person × situation interaction を modeling せず
- **対応**：
  - **「Personality slice」framing**：「本研究は personality contribution を isolate する。Bowling & Beehr 2006 が示す環境系効果（work constraints ρ=.53、role conflict ρ=.44）は scope 外」
  - **Personality は SSS の上流共通原因**（Heckman 2006、Grijalva 2015、Lee & Ashton 2005、Roberts 2007、Roberts & DelVecchio 2000）→ cell-conditional probability は personality の direct + SSS-mediated indirect effects を total に包含
  - 「first-order approximation」と明示、future work で organizational variables の組込み

#### ⚠ 限界 8：Common Method Bias（CMV）
- **問題**：personality と harassment が同一回答者の self-report → CMV 懸念
- **対応**：
  - **Podsakoff et al. 2003 standard reference**：CMV diagnostic を Methods で実施
  - **Harman's single-factor test、marker variable correction（Lindell-Whitney 2001）**を Stage 3 sensitivity に組込
  - Limitation で acknowledge
  - **Cote & Buckley 1987 magnitude reference**：CMV-present 35% vs without 11% → 本研究 effect size を **保守的に解釈**

#### ⚠ 限界 9：Phase 2 transportability（西欧 anchor → 日本）
- **問題**：介入 anchor papers すべて西欧 / 米国 sample（Hudson 2023 N=467 US、Kruse 2014 US、Pruckner 2013 オーストリア等）
- **対応**：
  - **Hernán & Robins 2020 framework** で transportability assumption を **explicit に状態**
  - **Sapouna et al. 2010** の UK vs Germany null finding（cultural moderator 実例）を引用、「日本での効果 attenuation 可能性」明示
  - Sensitivity sweep を **西欧 anchor の 0.3–1.0 倍** で wide range
  - **Nielsen et al. 2017 FFM meta**：Asia/Oceania での personality effect 弱（Neuroticism r=.16 vs Europe .33）→ 本研究の cell estimates も保守的解釈

#### ⚠ 限界 10：Latent vs expressed prevalence の分離不可
- **問題**：本研究は latent prevalence を予測するが、N=354 propensity items も完全に environmental gating-free ではない
- **対応**：
  - 「**latent と expressed の境界は連続的**」と明示
  - MHLW 2016（pre-law、最 latent-proximal）vs MHLW 2024（post-law、expressed）の triangulation で gap を visualize
  - 「**Latent prevalence prediction is the simulation's primary inferential target; Expressed prevalence requires environmental moderators not captured here**」と framing

### 4.3 Doc 1「良い論文がやらないこと」との照合

| やらないこと | 本研究での対応 |
|---|---|
| 単一 LLM・単一プロンプトで結論 | LLM 不使用 |
| Self-report のみで validate | 厚労省被害者報告との対比 + Berry 2012 + Anderson & Bushman 2002 で defend |
| 「85% accurate」だけで終わる | Baseline 比較 B0–B4 実施 |
| Counterfactual を因果と書く | Hernán & Robins 2020 framework で identifying assumptions 明示 |
| WEIRD 偏向に触れない | 日本データ + Nielsen 2017 cultural moderator 言及 |
| Refusal/null を隠す | LLM 不使用なので該当なし |
| CMV concern を ignore | Podsakoff 2003 diagnostic 実施、Methods で報告 |

---

## Part 5：Phase 2 設計（介入 counterfactual）

### 5.1 Phase 2 の核心主張

Phase 1 が描写的（"類型 X はハラスメントを起こしやすい"）に留まると、**自己責任解釈と区別困難**。

Phase 2 で counterfactual を加えることで：
- "**もし HH 介入が実装されたら、ハラスメントは X% 減る**" を示せる
- → "**介入可能 = 社会の責任**" を実証できる
- → あなたの主張「自己責任ではなく社会システムの問題」を**論理的に完成**

### 5.1.5 ★ Causal framing（Tier 2 で確立、必読）

Phase 2 全 counterfactual は **target trial emulation framework**（Hernán & Robins 2020、Pearl 2009）で formal に positioning：

| Causal element | Target trial emulation での記述 | 本研究での具体化 |
|---|---|---|
| Eligibility criteria | "Who would be in the trial?" | 日本労働者 20–64 歳 |
| Treatment strategies | "What interventions are compared?" | A 普遍 / B 集中 / C 構造 |
| Assignment procedures | "How is treatment assigned?" | Simulated random（hypothetical RCT）|
| Outcome | "What's measured?" | Population-level harassment prevalence |
| Follow-up | "When?" | 24 週時点（Roberts 2017 anchor）|
| Causal contrast | "What's the effect estimand?" | E[Y^{a=intervention}] − E[Y^{a=control}] |

#### 4 identifying assumptions（Hernán & Robins 2020）

1. **Exchangeability**：Y^a ⊥⊥ A | L（介入と outcome の confounding なし、L 統制下で）
2. **Positivity**：すべての type が intervention を受けられる（Type 6 dominant 32% でも possible）
3. **Consistency**：observed Y when A=a equals Y^a（no interference between agents）
4. **Transportability**：anchor study population effect が target Japanese workforce に transport

→ Discussion で **4 仮定すべてを明示**、各 violation 可能性を honest に述べる

#### Pearl 2009 ladder of causation での positioning

- 本研究は **Rung 2（intervention prediction）**
- Rung 3（individual counterfactual）への extension は cross-sectional data では不可能 → future work

### 5.2 3 種の Counterfactual 設計

#### Counterfactual A：Universal HH intervention（population-wide）

**仮定**：教育・職場研修等で日本人口全体の HH が +X SD シフト

**操作**：
```
shifted_HH = current_HH + δ × SD(HH)
ここで δ = 0.1, 0.2, 0.3, 0.4 SD
```

**実装**：
1. 全 N の HH 値を δ × SD だけシフト
2. シフト後の値で類型 membership を再計算（一部の個人が type 5 → type 1 に "移動"）
3. 集団 type 分布が変化
4. 新分布に対して Phase 1 の確率テーブルを適用
5. 新しい加害者数・被害者数・離職・メンタル疾患を計算

**政策含意**：「moral education / character education / digital coaching の population scale 効果」

---

#### Counterfactual B：Targeted intervention（高リスク類型のみ）

**仮定**：Self-Oriented Independent（低 HH 中心）など高加害確率類型のみに集中介入

**操作**：
```
target_types = [Type 5: Self-Oriented Independent, Type 6: Emotionally Volatile Extravert]
for individual in target_types:
    individual.HH += δ × SD(HH)  # δ = 0.5 を想定（self-selected motivated 層、効果大）
```

**実装**：
1. 該当類型の個人のみ HH シフト
2. 他類型は不変
3. 介入対象人数 × 介入コストで cost-effectiveness を計算
4. Counterfactual A と比較：「全員介入 vs ターゲット介入の効率」

**政策含意**：「リスク類型への counseling 集中投入」

---

#### Counterfactual C：Structural intervention（個人 personality 不変）

**仮定**：個人を変えず、**組織 / 制度の設計**でハラスメント発生確率を抑制

**操作**：
```
for cell in (type, gender):
    P(harassment | cell, structural) = P(harassment | cell, baseline) × (1 - effect_C)
ここで effect_C ≈ 0.10–0.30（30% 削減を保守的上限、Tier 2 4 系統 triangulation で確定）
```

**実装**：
1. 個人 personality 不変
2. 確率テーブルそのものを下方修正（介入された cell では加害確率減）
3. 加害者数・連鎖を再計算

**Tier 2 文献による effect_C 上限の triangulation**：
- **Bezrukova et al. 2016**（260 samples meta）：behavioral g=.48、attitudinal g=.30 で time decay
- **Roehling & Huang 2018**：Kirkpatrick Level 3/4（behavior, results）で弱
- **Dobbin & Kalev 2018**：985 studies meta 引用、"least effective diversity program"
- **Pruckner & Sausgruber 2013**：moral reminder、intensive margin のみ 2.4–2.5x、extensive null
- → **30% 削減上限は保守的、効果は intensive margin に限定可能性**

**政策含意**：**「個人を変えなくても制度設計で harassment は減らせる、ただし extensive margin（治らない人）には効きにくい」**
→ Counterfactual C の **限界の visualization** が、Counterfactual B（root cause）の優位性 argument を支援

---

### 5.3 介入 anchor の効果量推定

Roberts et al. (2017) の personality intervention systematic review より、平均 d ≈ 0.37（24 週時点）を **upper bound** として使用。

加えて、HH-specific 介入の anchor として下記を選定（詳細は Part 6.2）：
- **Kruse et al. (2014)** "Gratitude and Humility" — d = 0.71（Counterfactual A 主 anchor）
- **Hudson (2023)** "Lighten the Darkness" — Agreeableness 介入が Dark Triad を副次的に減少（Counterfactual B 主 anchor）
- **Pruckner & Sausgruber (2013)** "Honesty on the Streets" — moral reminder の field experiment、intensive margin で 2.4–2.5 倍効果（Counterfactual C 主 anchor、★ **本研究の主軸**）

### 5.4 Phase 2 出力

1. 3 種 counterfactual のそれぞれで予測される harassment 削減率（CI 付き）
2. 削減率を介入コスト軸で比較（cost-effectiveness ranking）
3. sensitivity analysis：δ や effect_C を変動させた場合の robustness
4. 政策含意：個人介入 vs 構造介入の論理的位置づけ
5. **「自己責任 → 社会責任」frame shift の実証的根拠**

### 5.5 主軸の決定：Counterfactual B を主軸とする

3 つの counterfactual のうち、**Counterfactual B（targeted individual intervention）を本研究の主軸**とする。

#### 「社会システム」の定義の明確化

本研究で「社会システム」とは：
- **狭義**：組織ルール、罰則、moral reminder（symptom 抑制）
- **広義**（本研究の立場）：**教育・研修・カウンセリング等の、社会が個人の変化を支援する仕組み**

Counterfactual B はこの広義の社会システムを表す。**個人を変える、しかし変化の装置は社会が用意する**という構造。

#### B が主軸である根拠

1. **根本治療**：加害傾向の根を変える（symptom 抑制ではない）
2. **Displacement の防止**：職場外への問題転嫁が起きない（C の限界）
3. **Capability approach との整合**：社会が高リスク者に enhanced 支援を提供する規範
4. **「自己責任ではない」との整合**：
   - 高リスク類型であることは遺伝 × 環境の結果（個人責任ではない）
   - しかし変化の機会は社会が公平に提供する
5. **強い anchor**：Hudson (2023) が Agreeableness 介入で Dark Triad 低下を実証（N=467、causal evidence）

#### A と C の位置づけ

- **A（universal）**：cost が高いが stigma なし。比較として併記。
- **C（structural）**：implementation 容易だが、Pruckner 2013 が示したように **extensive margin（治らない人）には効かない**。displacement リスクあり。比較として併記。

#### Abstract 核心メッセージ案

> "Our simulation shows that targeted intervention (e.g., enhanced humility/agreeableness training for high-risk personality types) reduces predicted workplace harassment by approximately X%, exceeding both universal personality interventions and structural-only interventions (e.g., moral reminders) which suffer from displacement effects. Importantly, providing such targeted intervention is the responsibility of social systems—including education, training, and counseling institutions—not the individual. This supports the conclusion that workplace harassment is a system-level problem, where the social system's role is to provide change opportunities rather than to demand individual self-responsibility."

---

## Part 6：Phase 2 限界と介入 anchor 選定

### 6.1 Phase 2 主要な限界

#### ⚠ 限界 P2-1：HH-specific 介入エビデンスの相対的弱さ
- **問題**：Neuroticism や Conscientiousness と比べ、HH を直接 target にした実験的介入研究が少ない
- **対応**：
  - 「最も近い介入」（humility workbook、moral reminder 等）の効果量を anchor に使う
  - 限界として明記：「HH 直接介入の effect size は不確実性が大きい」
  - sensitivity analysis で δ を 0.1–0.6 SD でスイープ

#### ⚠ 限界 P2-2：Counterfactual の構造保存的性質
- **問題**：「介入が effective である」と仮定して「effective である」と予測する潜在的循環性
- **対応**：
  - 介入の効果サイズを **外部 empirical anchor**（Roberts 2017 等）から取得
  - 「conditional projection given established intervention efficacy」と明記
  - causal language を避け、"if-then projection" の枠組みで議論

#### ⚠ 限界 P2-3：副作用の不考慮
- **問題**：HH 介入が他形質（C, A 等）に与える二次効果は modeling されない
- **対応**：limitation 明記、future work とする

#### ⚠ 限界 P2-4：介入の持続性
- **問題**：Roberts 2017 の d ≈ 0.37 は 24 週時点。長期効果は不明
- **対応**：「short-to-medium term projection」と限定

#### ⚠ 限界 P2-5：介入コストの現実的考慮なし
- **問題**：simulation は efficacy のみ、cost-effectiveness は別問題
- **対応**：limitation 明記、cost 情報があれば cost-effectiveness ranking を補助的に提示

#### ⚠ 限界 P2-6：個人 vs 集団スクリーニング誤読リスク
- **問題**：「Type 5 への介入が効く」が個人プロファイリングに転用される懸念
- **対応**：
  - Discussion で「集団政策の根拠であって個人スクリーニングではない」と繰り返し明示
  - 倫理的注意点を Doc 2 Part 5 から引き継ぐ

### 6.2 介入 anchor の確定（PDF 精読により determined）

8 件の介入研究 PDF 精読により、各 Counterfactual の主 anchor を確定：

#### ★ Counterfactual A（Universal individual intervention）の主 anchor
**Kruse, E., Chancellor, J., Ruberton, P. M., & Lyubomirsky, S. (2014). An Upward Spiral Between Gratitude and Humility. *Social Psychological and Personality Science, 5*(7), 805–814.** https://doi.org/10.1177/1948550614534700

- 設計：3 研究（うち 14 日間日記研究 1 つ）
- 介入：感謝の手紙 / 感謝 prompts
- **効果量：Cohen's d = 0.71（Study 1）、d = 0.77（Study 2）**
- δ assumption：**+0.5 SD**（Kruse の効果を population scale に保守的に外挿）
- Sensitivity range：0.2–0.7 SD

**補助 anchor**：
- Hudson et al. (2019) "You Have to Follow Through" — N=377、follow-through が trait change の鍵
- Naini et al. (2021) Online Group Counseling — humility 上昇

#### ★★★ Counterfactual B（Targeted individual intervention）の主 anchor — **研究の主軸**
**Hudson, N. W. (2023). Lighten the darkness: Personality interventions targeting agreeableness also reduce participants' levels of the dark triad. *Journal of Personality, 91*(4).** https://doi.org/10.1111/jopy.12714

- 設計：N=467、volitional personality change intervention
- 発見：**Agreeableness 介入が Dark Triad 3 つすべてを副次的に減少**
- 効果量：Agreeableness 介入 adherence b = .03 [CI .02–.04]（週単位、累積で d ≈ 0.4–0.6 想定）、Dark Triad average b = -.01
- Target：Self-Oriented Independent（低 HH・低 A 中心）など高リスク類型に集中介入
- δ assumption：**+0.7 SD**（self-selected motivated 層、effect 大）
- Sensitivity range：0.3–1.0 SD

**補助 anchor**：
- Barghamadi (2014) PROVE workbook（N=72、d ≈ 0.53、ただし undergraduate poster）
- Hudson et al. (2019) follow-through の重要性（completion-conditional effect）

**主軸根拠**：
- 個人を変える（root cause を治療）
- しかし変化の装置（教育・研修・カウンセリング）は社会が提供 → capability approach 整合
- Displacement 問題を回避（C のように職場外に転嫁しない）

#### ★ Counterfactual C（Structural-only intervention）の主 anchor — **比較対象**
**Pruckner, G. J., & Sausgruber, R. (2013). Honesty on the Streets: A Field Study on Newspaper Purchasing. *Journal of the European Economic Association, 11*(3), 661–679.** https://doi.org/10.1111/jeea.12016

- 設計：field experiment、127 newspaper sale locations、honor system payments
- 介入：moral reminder（"Sei ehrlich" を貼り出す）
- **効果量**：
  - CONTROL（中立）：平均支払 €0.16
  - LEGAL（法的告知）：€0.15
  - **MORAL（道徳リマインダー）：€0.38（≈ 2.4–2.5 倍）**
  - p = 0.038（vs CONTROL）、p = 0.008（vs LEGAL、Wilcoxon）
  - **Intensive margin のみに効果、extensive margin に効果なし**
- effect_C assumption：**加害確率を 30% 削減**（保守的な外挿）
- Sensitivity range：10–50% 削減

**補助 anchor**：
- Casali, Metselaar, & Thielmann (2025) Personality Feedback for Moral Traits（N=17、qualitative + quantitative）

**比較対象としての位置づけ**：
- **Pruckner の発見そのものが C の限界を示す**：「治らない人」（extensive margin）には効かない
- 職場で moral reminder を貼っても、加害者は職場外（after-work、SNS）に displace する可能性
- → **本研究は C の限界を可視化し、B（root cause を治す介入）の優位性を示す**ロジック

### 6.3 anchor 一覧表

| Counterfactual | 主 anchor | 文献 | 効果量 | δ / effect_C | 役割 |
|---|---|---|---|---|---|
| A: Universal individual | Kruse 2014 | SPPS 5(7) | d = 0.71 | +0.5 SD（範囲 0.2–0.7） | 比較対象 |
| **B: Targeted individual** ★★★ | **Hudson 2023** | **J Personality 91(4)** | **b = .03 / week** | **+0.7 SD（範囲 0.3–1.0）** | **主軸** |
| C: Structural-only | Pruckner 2013 | JEEA 11(3) | 2.4–2.5x（intensive） | 30% 削減（範囲 10–50%） | 比較対象（限界の可視化） |

### 6.4 補助参照文献（11 件中の残り）

- Kruse et al. (2014) → A の主 anchor として使用
- Barghamadi (2014) → B の補助
- Pruckner & Sausgruber (2013) → C の主 anchor として使用
- Hudson (2023) → B の主 anchor として使用
- Naini et al. (2021) → A の補助（Indonesian Online Group Counseling）
- Casali et al. (2025) → C の補助（personality feedback）
- Understanding Machiavellianism (review) → 理論的参考（effect anchor 不向き）
- Hudson et al. (2019) → A の補助（follow-through 重要性）

---

## Part 8：意思決定事項（**最終更新版**）

研究を実装段階に進める前に**確定すべき項目**。

### 8.1 Phase 1 関連

| # | 項目 | 推奨 | 状態 |
|---|---|---|---|
| D1 | 役職推定モデル | **continuous covariate に格上げ**（D13 で 14-cell 主分析確定）、3 モデル比較は副次 sensitivity | ✓ **確定**（D13 経由） |
| D2 | 主 validation target | **MHLW 2016 (32.5%, pre-law) 主、2020 R2 + 2024 R5 副**（latent vs expressed framing 反映） | ✓ **確定** |
| D3 | 被害者倍率 V の anchor | 厚労省実態調査 + 範囲 sweep [2, 3, 4, 5] | ✓ **確定**（Stage 3 sensitivity）|
| D4 | 離職率 f1 の anchor | ハラスメント specific 推定 + 雇用動向補。範囲 sweep [0.05, 0.10, 0.15, 0.20] | ✓ **確定** |
| D5 | メンタル疾患 f2 | 労働安全衛生調査 + Tsuno & Tabuchi 2022 PR=3.20 anchor。範囲 sweep [0.10, 0.20, 0.30] | ✓ **確定** |
| D6 | Cell size 閾値 | **N≥10 で推定**（D13 で確認、14-cell 全充足）、28-cell は EB shrinkage | ✓ **確定** |
| D7 | デモグラ重み付け | 厚労省労働力調査に合わせる | ✓ **確定** |
| **D-NEW1** | **EB shrinkage prior strength** | **Beta-Binomial conjugate、scale ∈ {0.5, 1.0, 2.0} sweep**（Casella 1985 + Clayton & Kaldor 1987 + Efron 2014 + Greenland 2000）| ✓ **確定** |
| **D-NEW2** | **Binarization threshold** | **mean+0.5SD 主、mean+0.25SD / mean+1.0SD で sweep**（Notelaers continuum framework）| ✓ **確定** |
| **D-NEW3** | **Cluster K robustness** | **K=4–8 sensitivity**（Tokiwa clustering paper 7 type は IEEE 掲載済の主分析） | ✓ **確定** |
| **D-NEW4** | **CMV diagnostic** | **Harman's single-factor + marker variable**（Podsakoff et al. 2003 準拠） | ✓ **確定** |
| **D-NEW5** | **Baseline B4 拡張** | personality + age + 業種推定 + 雇用形態（**B3 vs B4 比較**で personality slice の incrementality 評価） | ✓ **確定** |

### 8.2 Phase 2 関連

| # | 項目 | 推奨 | 状態 |
|---|---|---|---|
| D8 | 介入 anchor | A: Kruse 2014 / B: Hudson 2023（**主軸**）/ C: Pruckner 2013 + Bezrukova 2016 + Dobbin & Kalev 2018 + Roehling 2018 で triangulation | ✓ **確定** |
| D9 | δ レンジ（Counterfactual A） | 0.1–0.6 SD sweep | ✓ **確定** |
| D10 | 構造介入の effect_C | **0.10–0.30 sweep（30% 削減保守的上限）**、Tier 2 4 系統 triangulation | ✓ **確定** |
| D11 | Counterfactual 主軸 | **B を主軸、A/C を比較対象** | ✓ **確定** |
| **D-NEW6** | **Causal framing** | **Hernán & Robins 2020 target trial emulation**、4 identifying assumptions 明示 | ✓ **確定** |
| **D-NEW7** | **Phase 2 transportability sensitivity** | 西欧 anchor の **0.3–1.0 倍** wide range sweep | ✓ **確定** |

### 8.3 共通

| # | 項目 | 推奨 | 状態 |
|---|---|---|---|
| D12 | Pre-registration | OSF（執筆完了後に予定、user 指示） | ⏳ **執筆完了後** |
| D13 | Power analysis | **D13 完了**：14-cell 主分析確定、28-cell は EB shrinkage | ✓ **完了** |
| D14 | Target journal（Phase 1） | JBE 第一候補、PAID / J Comp Soc Sci / PLOS ONE 等候補 | 未決定 |
| D15 | Target journal（Phase 2） | Phase 1 の状況次第、Public Health 系も候補 | 未決定 |
| D16 | コード公開先 | GitHub + OSF 両方 | ✓ **確定** |
| D17 | データ公開ポリシー | 既存論文の公開方針に準拠 | 未決定 |
| D18 | Phase 1 と Phase 2 を別論文か単一論文か | **別 2 本**（impact 分散、scope 適切） | ✓ **確定** |

### 8.4 直近の実装着手前に必要な確認

- [x] **D13（Power analysis）の実施** — ✅ 完了（`simulation/docs/power_analysis/D13_power_analysis.md`）
- [x] **D8（介入 anchor）の確定** — ✅ 完了（Tier 2 で更に triangulation 強化）
- [x] **文献基盤確立** — ✅ 完了（Tier 1+2+3 で 40 paper deep reading）
- [ ] **論文 Introduction draft 着手**（5 段落構成、Tier 1+2+3 完全引用）
- [ ] **Stage 0 コード実装**（D13 14-cell + EB shrinkage 仕様、4 paper methodological foundation）
- [ ] **D12（Pre-registration）の作成** — 執筆完了後（user 指示）

---

## Part 9：Doc 1 / Doc 2 / 既存資産との Link

### 9.1 ドキュメント間の関係

```
research_vision_integrated.md (Doc 2)
        ↓
        【「自己責任ではなく社会システム」の主張】
        ↓
simulation_paper_evaluation_integrated.md (Doc 1)
        ↓
        【方法論評価フレーム：5 レベル / 6 戦略 / 12 ポイント】
        ↓
research_plan_harassment_typology_simulation.md（本ドキュメント）
        ↓
        【具体的研究プロジェクト：Phase 1 + Phase 2】
        ↓
        実装フェーズ → 論文化
```

### 9.2 Doc 1 との具体的対応

| Doc 1 のフレーム | 本研究計画での対応 |
|---|---|
| L1 再現性 | Phase 1 Stage 3 で MAPE / KS / Wasserstein で評価 |
| L2 予測 | Phase 1 で hold-out validation |
| L3 反実仮想 | Phase 2 で counterfactual A/B/C |
| L4 機構 | 確率テーブル設計 + sensitivity analysis |
| L5 仮説生成 | failure mode の発見化 |
| 戦略 1 集団予測 | 国レベル aggregate に純化 |
| 戦略 2 失敗の主題化 | Pre-registration で成功 / 失敗閾値固定、failure を発見化 |
| 戦略 3 Triangulation | 厚労省 4 種データソース |
| 戦略 4 階層 baseline | B0–B3 設計 |
| 戦略 5 Pre-registration | OSF Registration（D12） |
| 戦略 6 ツール positioning | "policy planning tool" 明示 |
| 評価ポイント A-D（LLM） | 該当なし（LLM 不使用） |
| 評価ポイント E-L | E, F, G, I, J, K, L すべて満たす予定。H（power analysis）が要対応 |

### 9.3 Doc 2 との具体的対応

| Doc 2 の主張 | 本研究計画での実装 |
|---|---|
| 個人予測の限界（L1 / L4） | Phase 1 は集団予測に純化 |
| 性格は遺伝×環境の合成シグナル | 7 類型を介入ターゲットとして使用 |
| 物質資源の zero-sum / 性格の positive-sum | Phase 2 で「個人介入 vs 構造介入」比較 |
| Capability approach（区別 vs 差別） | 結果解釈で「集団政策、個人スクリーニングではない」明示 |
| Compatibilism への配慮 | 「介入可能 = 自己責任ではない」の論理を慎重に展開 |
| 4 層構造（L1/L2/L3/L4） | 本研究は L1（経験的問い）に純化、L3/L4 は著作で展開 |

### 9.4 既存自己引用 hub

```
        Tokiwa (Clustering paper, IEEE)
                 ↓ 提供：N=13,668 + 7 類型 centroid
                 ↓
本研究計画 ← Tokiwa (Harassment paper, Preprint)
                 ↑ 提供：N=354 + 加害確率推定
                 
                 既存：Tokiwa (2025) Frontiers Psych
                       Tokiwa (2026 preprint) OSF E5W47
                       既存 simulation 論文 (HANDOFF)
```

### 9.5 関連既存資産（このリポジトリ内）

- `clustering/` — N=13,668 データと clustering スクリプト
- `clustering/paper_IEEE/Manuscript_IEEE_rivision.docx` — Clustering 論文 IEEE 投稿原稿
- `harassment/raw.csv` — N=354 の完全データ
- `harassment/analysis.py` — 既存の解析コード
- `harassment/paper/Manuscript_only.docx` — Harassment 論文原稿
- `simulation/agent/` — 既存 LLM agent pipeline（本研究では未使用）
- `simulation/HANDOFF.md` — 既存 Big Five simulation 論文の状態
- `simulation/prior_research/_text/` — 主要参考文献の PDF + テキスト

### 9.6 主要参考文献（本研究計画で参照する核心文献）

#### 性格類型・HEXACO
- Ashton, M. C., & Lee, K. (2007). Empirical, theoretical, and practical advantages of the HEXACO model of personality structure. *Personality and Social Psychology Review, 11*(2), 150–166.
- Wakabayashi, A. (2014). Japanese HEXACO-60. *Japanese Psychological Research, 56*, 211–223.
- Daljeet et al. (2017), Espinoza et al. (2020), Gerlach et al. (2018), Kerber et al. (2021) — クラスタリング先行研究

#### Harassment 測定
- Tou, S. et al. (2017). Workplace Power Harassment Scale.
- Kobayashi, A., & Tanaka, K. (2010). Gender Harassment Scale.
- Shimotsukasa, T., & Oshio, A. (2017). SD3-J.

#### 性格 × 介入
- Roberts, B. W. et al. (2017). A systematic review of personality trait change through intervention. *Psychological Bulletin, 143*(2), 117–141. https://doi.org/10.1037/bul0000088
- Stieger, M. et al. (2021). Changing personality traits with the help of a digital personality change intervention. *PNAS, 118*(8). [PEACH 研究]
- Hudson, N. W., & Fraley, R. C. (2015). Volitional personality trait change.
- Lavelock et al. (2014). Workbook on humility. *Journal of Positive Psychology*.
- Mazar, N., Amir, O., & Ariely, D. (2008). The dishonesty of honest people.

#### Simulation 方法論
- Park, J. S. et al. (2024). Generative agent simulations of 1,000 people. arXiv:2411.10109.
- Hewitt, L. et al. (2024). Predicting results of social science experiments using large language models.
- Argyle, L. P. et al. (2023). Out of one, many. *Political Analysis, 31*(3).
- Lundberg, I., Brand, J. E., & Jeon, N. (2024). The origins of unpredictability in life outcome prediction tasks. *PNAS, 121*(24).
- Salganik, M. J. et al. (2020). Measuring the predictability of life outcomes. *PNAS, 117*(15).

詳細な参考文献は Doc 2 Part 8 を参照。

---

## Part 10：文献基盤の最終構成（Tier 1/2/3 deep reading 後の総括）

本研究は **40 件 deep reading + 約 20 件 既存ライブラリ補完 = 約 60 paper foundation** で全主張を防御。

### 10.1 Tier 別概要

| Tier | 件数 | 重点 pillar | 状態 |
|---|---|---|---|
| **Tier 1** | 24 | Pillar 1 (epidemiology) + Pillar 3 (microsim 系譜 + LCA) | ✅ 全 deep reading 完了 |
| **Tier 2** | 9 | Pillar D (personality upstream) + B (intervention reviews) + C (causal) + A (self-report validity) | ✅ 全 deep reading 完了 |
| **Tier 3** | 7 | Personality stability + CMV + Bayesian shrinkage methodology | ✅ 全 deep reading 完了 |
| 自己引用 | 2 | Tokiwa Clustering (IEEE 掲載済) + Harassment preprint | ✅ 確定 |
| 既存ライブラリ補完 | ~20 | Pillar 2 (HEXACO/DT meta) + 性格介入 anchor 等 | ✅ 既所有 |
| **合計** | **約 60 papers** | 8 pillars 全 cover | |

### 10.2 致命的弱点 → 文献装備 mapping

| 弱点 | 装備（文献） |
|---|---|
| 1. Cross-sectional 因果（reverse causation） | Roberts & DelVecchio 2000 + Specht 2011（personality stability r=.74 plateau）|
| 2. ~~Self-citation hub fragility~~ | Clustering paper IEEE 掲載済で **解消** |
| 3. GAM situational 欠落 | "Personality slice" framing（既存文献で対応） |
| 4. Phase 2 transportability | Hernán & Robins 2020 + Pearl 2009 + sensitivity wide range |
| 5. Reverse-direction self-report bias | Berry 2012 + Anderson & Bushman 2002 + Roberts & DelVecchio 2000 |
| 6. Common Method Bias | Podsakoff 2003 standard reference + Harman / marker variable diagnostic |
| 28-cell shrinkage 実装 | Casella 1985 + Clayton & Kaldor 1987 + Efron 2014 + Greenland 2000 |
| Personality vs SSS independence | Heckman 2006 + Grijalva 2015 + Lee & Ashton 2005 + Roberts 2007 + Roberts & DelVecchio 2000 |

### 10.3 Introduction 5 段落構成（Tier 1+2+3 完全引用）

| 段落 | 内容 | 主要引用 |
|---|---|---|
| 1. Global concern | 23% global prevalence | ILO 2022、Nielsen et al. 2010 meta、Bowling & Beehr 2006 |
| 2. Japan context | 31.4% パワハラ、6.1% national rep | MHLW 2021/2024、Tsuno et al. 2010/2015/2022 |
| 3. **Predictor lineage with personality upstream** | HEXACO HH、Dark Triad、**personality is upstream of SSS** | Nielsen et al. 2017、Pletzer et al. 2019、Tokiwa preprint、Heckman et al. 2006、Grijalva et al. 2015、Lee & Ashton 2005、Roberts et al. 2007、**Roberts & DelVecchio 2000** |
| 4. **Methodological gap** | Microsim lineage vs LLM | Orcutt 1957、Spielauer 2011、Schofield et al. 2018、Bruch & Atwell 2015、Park et al. 2024、**Greenland 2000** |
| 5. **Existing precursors and study aim** | Ho et al. 2025 等の差異、本研究 3 軸 novelty | Ho et al. 2025、Notelaers et al. 2006/2011、Lanza & Rhoades 2013、**Hernán & Robins 2020**、**Pearl 2009** |

### 10.4 Discussion limitation framework（10 limitation × 文献 × 対応）

Part 4.2 で詳述。Tier 1+2+3 evidence で全 limitation に **2–4 系統 defense** 装備。

### 10.5 ファイル参照

- `simulation/docs/literature_audit/literature_audit.md` — 8 pillar audit
- `simulation/docs/literature_audit/tier1_search_results.md` — Tier 1 26 件
- `simulation/docs/literature_audit/tier2_search_results.md` — Tier 2 12 件
- `simulation/docs/literature_audit/deep_reading_notes.md` — **40 件 deep reading 統合、3,950 行**
- `simulation/docs/power_analysis/D13_power_analysis.md` — D13 power analysis report

---

**本ドキュメントは「研究計画」です。**  
**評価フレームは** `simulation_paper_evaluation_integrated.md`、  
**研究ビジョン全体像は** `research_vision_integrated.md`、  
**文献基盤の精読詳細は** `simulation/docs/literature_audit/deep_reading_notes.md` **を参照してください。**

