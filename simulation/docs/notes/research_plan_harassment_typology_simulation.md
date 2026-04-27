# 研究計画：HEXACO 類型ベースの職場ハラスメント simulation（Phase 1 + Phase 2）

作成日：2026-04-27
ブランチ：`main`
関連ドキュメント：
- `simulation/docs/notes/simulation_paper_evaluation_integrated.md`（Doc 1：方法論評価）
- `simulation/docs/notes/research_vision_integrated.md`（Doc 2：研究ビジョン）
- `simulation/HANDOFF.md`（既存 Big Five simulation 論文）

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
- **LLM 不使用**（古典的確率モデル + Monte Carlo）
- **倫理審査軽負荷**（既存 IRB 通過データのみ）
- **2 段階構成**：Phase 1（実証 simulation）→ Phase 2（介入 counterfactual）
- **既存自己引用 hub**：Clustering 論文 + Harassment 論文 の自然な拡張

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

## Part 1：背景と動機

### 1.1 研究ビジョンからの逆算（Doc 2 より）

> **「個人 outcome は遺伝と環境の関数として確率的に予測可能であり、性格特性はその合成シグナルとして介入の手がかりになる。資源（収入・教育）が部分的に zero-sum である現実において、非ポジショナルな性格特性への集団介入こそが社会全体の幸福を底上げする最もスケーラブルな戦略である。」**

### 1.2 主張の中核

> **「特定の性格タイプ集団が特定の社会問題を起こしやすい・起こされやすい場合、それは個人の自己責任ではなく社会システムの設計の問題である。」**

### 1.3 既存研究との関係

#### Clustering 論文（IEEE 投稿中、N=13,668）
- 日本人の HEXACO 7 類型を多手法クラスタリングで同定
- Reserved / Emotionally Sensitive / Exploratory Extravert / Conscientious Introvert / Self-Oriented Independent / Emotionally Volatile Extravert / Reliable Introvert
- **本研究の Stage 0 入力**として使用

#### Harassment 論文（Preprint 改訂版、N=354）
- HEXACO + Dark Triad → 加害傾向の HC3-robust 階層回帰
- Power harassment：Psychopathy β=.32–.40、H–H β=−.14
- Gender harassment：H–H β=−.23、Openness β=−.24
- **本研究の Stage 1 確率テーブル推定**に使用

#### Simulation 既存論文（HANDOFF、N=103）
- Big Five + LLM agent + 大学入試結果予測
- **本研究は方法論的に独立**（LLM 不使用）

### 1.4 なぜこの設計か

#### サンプル収集なしで simulation 論文として成立する条件
1. ✅ **既存データの再利用**（N=354 + N=13,668）
2. ✅ **公開統計との照合可能性**（厚労省 4 種データソース）
3. ✅ **方法論的新規性**（type-conditional simulation の harassment への初応用）
4. ✅ **論理透明性**（LLM black box でない確率モデル）

#### 候補 A/B/C との比較（Doc 1 評価）
- 候補 A（Harassment LLM）：safety risk 致命的
- 候補 B（Clustering LLM）：tautology risk
- 候補 C（A+B）：両方の問題を継承
- **本案**：これらの問題群を**構造的に回避**

---

## Part 2：リサーチクエスチョン

### 2.1 Phase 1（実証 simulation）

#### Main RQ
**「N=354 と N=13,668 から構築した HEXACO 7 類型ベースの確率モデルは、日本の全国ハラスメント統計をどの程度再現できるか？」**

#### Sub-RQs
- **RQ1.1**：7 類型 × gender × 推定役職 の cell ごとの加害確率はどう異なるか？
- **RQ1.2**：階層 baseline（B0 random / B1 gender / B2 HEXACO 線形 / B3 7 類型）と比較して、7 類型モデル（B3）はどれだけ informativeness gain があるか？
- **RQ1.3**：simulation 出力は厚労省実態調査・雇用動向・労働安全衛生のうちどれと最も合致し、どれと乖離するか？
- **RQ1.4**：simulation の予測誤差は、どの cell（類型・役職・gender）で最大か？（failure mode の特定）

### 2.2 Phase 2（介入 counterfactual）

#### Main RQ
**「既存の HH 改善介入研究の効果量を anchor として、population-wide / targeted / structural の 3 種介入は、それぞれハラスメント発生率をどの程度削減すると予測されるか？」**

#### Sub-RQs
- **RQ2.1**：Universal HH 介入（全人口を対象とする教育・訓練）の population-level 効果は？
- **RQ2.2**：Targeted 介入（高リスク類型のみに集中投入）の効果と cost-effectiveness は？
- **RQ2.3**：Structural 介入（個人 personality 不変、組織 / 制度のみ変更）の効果は？
- **RQ2.4**：3 種介入のうち、どれがあなたの主張「自己責任ではなく社会システムの問題」を最も強く支持するか？

### 2.3 共通 RQ（両 Phase に渡る）

- **RQ-X**：本シミュレーションの限界（self-report 依存、N=354 代表性、役職推定誤差等）が結果にどう影響するか sensitivity analysis で評価

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

#### Stage 0：類型分類と確率テーブル構築
1. N=13,668 の HEXACO data から 7 類型 centroid を取得（クラスタリング論文済み）
2. N=354 の各個人を最近傍 centroid に分類 → 7 類型 membership 付与
3. 役職を personality から確率的に予測（文献値で重み付け線形モデル）
4. 7 類型 × 役職 × gender の cell ごとに加害確率を bootstrap 推定（Bayesian CI 付き）

#### Stage 1：母集団スケーリング
1. 日本労働人口（約 6,800 万）を仮想母集団とする
2. 13,668 の類型分布を母集団に適用（必要に応じてデモグラ重み付け）
3. 役職分布を厚労省統計に合わせる
4. gender 分布を厚労省労働力調査に合わせる
5. 各 cell に対して bootstrap で**期待加害者数**を計算（CI 付き）

#### Stage 2：加害行動の連鎖
1. **加害者数**：Stage 1 で算出
2. **被害者数**：加害者 1 人あたり平均 V 人の被害者（V は厚労省実態調査から推定）
3. **離職**：被害者の f1 割が離職（f1 は雇用動向調査から推定）
4. **メンタル疾患**：被害者の f2 割が adjustment disorder / depression（f2 は労働安全衛生調査から推定）

#### Stage 3：sensitivity analysis
- V を [2, 3, 4, 5] でスイープ
- f1 を [0.05, 0.10, 0.15, 0.20] でスイープ
- f2 を [0.10, 0.20, 0.30] でスイープ
- 役職予測モデルを 3 種類で比較
- → robust な結果と fragile な結果を分離

### 3.3 検証層（Triangulation）

| 検証 target | データソース | 比較指標 |
|---|---|---|
| 加害者率 / 被害者率 | 厚労省実態調査 2020 | 国全体・業種別 |
| 相談件数の時系列 | 個別労働紛争 | 過去 10 年トレンド |
| ハラスメント由来離職 | 雇用動向調査 | 年次推定値 |
| ハラスメント由来メンタル疾患 | 労働安全衛生調査 | 業種別発症率 |

#### 検証指標
- **Pearson r / Spearman ρ**：simulation 出力と実測の相関
- **KS 距離 / Wasserstein 距離**：分布形状の一致度
- **Mean Absolute Percentage Error (MAPE)**：絶対値の精度
- **Calibration plot**：cell ごとの予測 vs 実測

#### 成功基準（Pre-registration で固定）
- **★ 主成功基準**：MAPE ≤ 30% で「成功」
- **★ 失敗基準**：MAPE ≥ 60% で「失敗」（だが**failure mode を発見化**）
- **△ 中間**：MAPE 30–60% を「partial success」

### 3.4 出力

1. 7 類型 × 役職 × gender の加害確率テーブル（CI 付き）
2. 国レベル predicted vs 実測の比較表
3. cell ごとの prediction error map（どこで合うか合わないか）
4. sensitivity analysis 結果
5. 階層 baseline との比較（B0–B3）

---

## Part 4：Phase 1 階層 baseline と限界

### 4.1 階層 baseline 設計（Doc 1 戦略 4 に準拠）

| Baseline | 入力 | 仮説 |
|---|---|---|
| **B0：完全ランダム** | uniform 加害確率 | informativeness ゼロ |
| **B1：gender のみ** | gender × 加害確率 | demographic baseline |
| **B2：HEXACO 6 領域（線形）** | 6 領域 → 線形回帰で加害確率 | trait-level prediction |
| **B3：7 類型モデル（提案手法）** | 7 類型 + gender + 推定役職 | type-conditional prediction |

#### 評価
- **B3 > B2 > B1 > B0** の単調増加 → 「7 類型モデルが情報を加えている」
- **B3 ≈ B2** の場合 → 「typology は線形より優位ではない」（発見として報告）
- **B3 < B2** の場合 → 「typology が overfitting」（より深刻な発見）

### 4.2 主要な限界（Doc 1 「やらないこと」と照合済み）

#### ⚠ 限界 1：Self-report → Self-report の循環性
- **問題**：N=354 の加害は self-report、厚労省実態調査も部分的に self-report
- **対応**：
  - 厚労省実態調査の **被害者報告（より客観的）** を主 validation target にする
  - 加害者自己報告は「比較参照」として使う
  - limitation セクションで明記

#### ⚠ 限界 2：N=354 の代表性
- **問題**：クラウドソーシング自己選択 sample。日本労働者を代表しない
- **対応**：
  - デモグラ（年齢、性別）で重み付け補正
  - 13,668 の類型分布を主、N=354 を type-conditional 確率推定にだけ使う
  - sensitivity analysis で代表性仮定の影響を評価

#### ⚠ 限界 3：役職推定の誤差
- **問題**：N=354 に役職データなく、personality から推定
- **対応**：
  - 推定モデルを 3 種類（線形、tree-based、文献ベース）で比較
  - sensitivity analysis：役職分布を fix した場合 vs 推定の場合
  - limitation で「役職は personality から推定された approximation」と明示

#### ⚠ 限界 4：Cell-level 統計検出力（power）
- **問題**：N=354 を 7 類型 × 役職 × gender に分割すると一部 cell が小（< 10）
- **対応**：
  - **事前 power analysis 必須**（Doc 1 評価ポイント H）
  - cell size < 10 の cell は Bayesian shrinkage で隣接 cell から借入
  - 推定の不確実性を CI で明示

#### ⚠ 限界 5：被害者倍率 V と離職率 f の不確実性
- **問題**：V, f1, f2 の点推定は文献から、確実性はない
- **対応**：sensitivity analysis（Stage 3）で複数値スイープ

#### ⚠ 限界 6：因果主張の不可能性
- **問題**：シミュレーションは描写的、因果ではない
- **対応**：
  - 「再現性 validation 研究」と明記、causal claim を避ける
  - Phase 2（counterfactual）でも「conditional projection given established intervention efficacy」と限定

#### ⚠ 限界 7：simulation 内の simplification
- **問題**：matching、組織風土、リーダーシップ等を捨象
- **対応**：「first-order approximation」と明示、future work で言及

### 4.3 Doc 1「良い論文がやらないこと」との照合

| やらないこと | 本研究での対応 |
|---|---|
| 単一 LLM・単一プロンプトで結論 | LLM 不使用 |
| Self-report のみで validate | 厚労省被害者報告との対比で部分的に対処 |
| 「85% accurate」だけで終わる | Baseline 比較を実施 |
| Counterfactual を因果と書く | 控えめに positioning（Phase 2 でも） |
| WEIRD 偏向に触れない | 日本データなので強み（non-WEIRD 貢献） |
| Refusal/null を隠す | LLM 不使用なので該当なし |

---

## Part 5：Phase 2 設計（介入 counterfactual）

### 5.1 Phase 2 の核心主張

Phase 1 が描写的（"類型 X はハラスメントを起こしやすい"）に留まると、**自己責任解釈と区別困難**。

Phase 2 で counterfactual を加えることで：
- "**もし HH 介入が実装されたら、ハラスメントは X% 減る**" を示せる
- → "**介入可能 = 社会の責任**" を実証できる
- → あなたの主張「自己責任ではなく社会システムの問題」を**論理的に完成**

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
for cell in (type, position, gender):
    P(harassment | cell, structural) = P(harassment | cell, baseline) × (1 - effect_C)
ここで effect_C ≈ 0.20–0.40（moral reminder、organizational ethics 介入の典型効果）
```

**実装**：
1. 個人 personality 不変
2. 確率テーブルそのものを下方修正（介入された cell では加害確率減）
3. 加害者数・連鎖を再計算

**政策含意**：**「個人を変えなくても制度設計で harassment は減らせる」**
→ あなたの「自己責任ではなく組織責任」主張に最も直結

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

### 5.5 主軸の決定：Counterfactual C を主軸とする

3 つの counterfactual のうち、**Counterfactual C（structural intervention）を本研究の主軸**とする。

**根拠**：
1. **個人 personality 不変、組織・制度のみ変更** → 「個人の自己責任ではなく社会の責任」を最も直接的に示す
2. **Pruckner 2013 の field experiment が clean な structural anchor**（causal evidence あり）
3. **Scalable & low-cost** な政策実装が可能 → policy implication が strong
4. **Counterfactual A/B/C 比較**で「個人介入 vs 構造介入の効率」も同時に示せる

**Abstract 核心メッセージ案**：
> "Our simulation shows that structural intervention (e.g., moral reminders) reduces predicted workplace harassment by approximately 30%, comparable to or exceeding population-wide personality interventions, supporting the conclusion that workplace harassment is a system-level problem amenable to institutional design rather than solely a matter of individual personality."

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

#### ★ Counterfactual A（Universal HH intervention）の主 anchor
**Kruse, E., Chancellor, J., Ruberton, P. M., & Lyubomirsky, S. (2014). An Upward Spiral Between Gratitude and Humility. *Social Psychological and Personality Science, 5*(7), 805–814.** https://doi.org/10.1177/1948550614534700

- 設計：3 研究（うち 14 日間日記研究 1 つ）
- 介入：感謝の手紙 / 感謝 prompts
- **効果量：Cohen's d = 0.71（Study 1）、d = 0.77（Study 2）**
- δ assumption：**+0.5 SD**（Kruse の効果を population scale に保守的に外挿）
- Sensitivity range：0.2–0.7 SD

**補助 anchor**：
- Hudson et al. (2019) "You Have to Follow Through" — N=377、follow-through が trait change の鍵
- Naini et al. (2021) Online Group Counseling — humility 上昇

#### ★ Counterfactual B（Targeted intervention）の主 anchor
**Hudson, N. W. (2023). Lighten the darkness: Personality interventions targeting agreeableness also reduce participants' levels of the dark triad. *Journal of Personality, 91*(4).** https://doi.org/10.1111/jopy.12714

- 設計：N=467、volitional personality change intervention
- 発見：**Agreeableness 介入が Dark Triad 3 つすべてを副次的に減少**
- 効果量：Agreeableness 介入 adherence b = .03 [CI .02–.04]（週単位）、Dark Triad average b = -.01
- Target：Self-Oriented Independent（低 HH・低 A 中心）類型に集中介入
- δ assumption：**+0.7 SD**（self-selected motivated 層、effect 大）
- Sensitivity range：0.3–1.0 SD

**補助 anchor**：
- Barghamadi (2014) PROVE workbook（N=72、d ≈ 0.53、ただし undergraduate poster）

#### ★★★ Counterfactual C（Structural intervention）の主 anchor — **研究の主軸**
**Pruckner, G. J., & Sausgruber, R. (2013). Honesty on the Streets: A Field Study on Newspaper Purchasing. *Journal of the European Economic Association, 11*(3), 661–679.** https://doi.org/10.1111/jeea.12016

- 設計：field experiment、127 newspaper sale locations、honor system payments
- 介入：moral reminder（"Sei ehrlich" を貼り出す）
- **効果量**：
  - CONTROL（中立）：平均支払 €0.16
  - LEGAL（法的告知）：€0.15
  - **MORAL（道徳リマインダー）：€0.38（≈ 2.4–2.5 倍）**
  - p = 0.038（vs CONTROL）、p = 0.008（vs LEGAL、Wilcoxon）
  - Intensive margin（払う人がより多く払う）に効果、extensive margin に効果なし
- effect_C assumption：**加害確率を 30% 削減**（保守的な外挿）
- Sensitivity range：10–50% 削減

**補助 anchor**：
- Casali, Metselaar, & Thielmann (2025) Personality Feedback for Moral Traits（N=17、qualitative + quantitative）

### 6.3 anchor 一覧表

| Counterfactual | 主 anchor | 文献 | 効果量 | δ / effect_C |
|---|---|---|---|---|
| A: Universal | Kruse 2014 | SPPS 5(7) | d = 0.71 | +0.5 SD（範囲 0.2–0.7） |
| B: Targeted | Hudson 2023 | J Personality 91(4) | b = .03 / week | +0.7 SD（範囲 0.3–1.0） |
| **C: Structural** ★ | **Pruckner 2013** | **JEEA 11(3)** | **2.4–2.5x（intensive）** | **30% 削減（範囲 10–50%）** |

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

## Part 8：繰り越しの意思決定事項

研究を実装段階に進める前に**確定すべき項目**：

### 8.1 Phase 1 関連

| # | 項目 | 選択肢 | 推奨 | 状態 |
|---|---|---|---|---|
| D1 | 役職推定モデル | (a) personality 線形、(b) tree-based、(c) 文献ベース、(d) 全て比較 | (d) | 未決定 |
| D2 | 主 validation target | (a) 厚労省実態調査、(b) 雇用動向、(c) 労働安全衛生 | (a) を主、(b)(c) を補 | 仮決定 |
| D3 | 被害者倍率 V の anchor | (a) 厚労省実態調査の被害者率、(b) 国際比較値、(c) 範囲 sweep | (a)+(c) | 未決定 |
| D4 | 離職率 f1 の anchor | (a) 雇用動向「人間関係」カテゴリ全体、(b) ハラスメント specific 推定 | (b) を試み、(a) で補 | 未決定 |
| D5 | Memo: メンタル疾患 f2 | (a) 労働安全衛生調査、(b) 業種別自殺統計併用 | (a) を主、(b) を補 | 未決定 |
| D6 | Cell size 閾値 | (a) N≥10 で推定、N<10 は shrinkage、(b) 別の閾値 | (a) | 仮決定 |
| D7 | デモグラ重み付け | (a) 厚労省労働力調査に合わせる、(b) しない | (a) | 未決定 |

### 8.2 Phase 2 関連

| # | 項目 | 選択肢 | 推奨 | 状態 |
|---|---|---|---|---|
| D8 | 介入 anchor の確定 | A: Kruse 2014 / B: Hudson 2023 / C: Pruckner 2013 を主 anchor | C を主軸として確定 | ✓ **確定** |
| D9 | δ レンジ（Counterfactual A） | 0.1–0.6 SD でスイープ vs 単一値 | スイープ | 仮決定 |
| D10 | 構造介入の effect_C | 0.20、0.30、0.40 でスイープ | スイープ | 仮決定 |
| D11 | Counterfactual の主軸 | A、B、C のどれを主結果にするか | C を主、A/B を比較 | 仮決定 |

### 8.3 共通

| # | 項目 | 選択肢 | 推奨 | 状態 |
|---|---|---|---|---|
| D12 | Pre-registration | OSF Registries / AsPredicted / なし | OSF（メタ分析論文 E5W47 と同手順） | 未決定 |
| D13 | Power analysis | 事前に実施 / 事後 / なし | **事前必須**（Doc 1 評価ポイント H） | 未決定 |
| D14 | Target journal（Phase 1） | JBE / J Comp Soc Sci / PAID / Behav Res Methods / PLOS ONE | JBE 第一候補 | 未決定 |
| D15 | Target journal（Phase 2） | 同上 + Public Health 系 | Phase 1 の状況次第 | 未決定 |
| D16 | コード公開先 | GitHub / OSF | 両方 | 未決定 |
| D17 | データ公開ポリシー | 既存 raw を公開 / 集計のみ | 既存論文の公開方針に準拠 | 未決定 |
| D18 | Phase 1 と Phase 2 を別論文か単一論文か | 別 2 本 / 結合 1 本 | **別 2 本**（impact 分散、scope 適切） | 仮決定 |

### 8.4 直近の実装着手前に必要な確認

- [ ] **D1（役職推定モデル）の文献調査**：Conscientiousness × leadership emergence の効果量
- [ ] **D8（介入 anchor）の確定**：あなたの PDF 受領 + 実 effect size の整理
- [ ] **D12（Pre-registration）の作成**：OSF テンプレートでの draft
- [ ] **D13（Power analysis）の実施**：N=354 を 7 × 2 × 2 cell に分割した時の cell size と power

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

**本ドキュメントは「研究計画」です。**  
**評価フレームは** `simulation_paper_evaluation_integrated.md`、  
**研究ビジョン全体像は** `research_vision_integrated.md` **を参照してください。**

