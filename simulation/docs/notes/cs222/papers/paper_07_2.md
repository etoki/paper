# 07_2 — Predicting Results of Social Science Experiments Using Large Language Models

## 書誌情報

- 著者: **Luke Hewitt**¹*, **Ashwini Ashokkumar**²*, Isaias Ghezae¹, **Robb Willer**¹
  - \* Equal contribution, order randomized
- 所属: ¹Stanford University, ²New York University
- 発表: 2024年8月8日（プレプリント、*Nature* に投稿版と同等）
- Web demo: https://treatmenteffect.app（AI ベースの forecast ツール）
- Lecture 07 の補足論文（応用: 社会科学実験結果の予測）

---

## 1. 研究問題

### 背景
- LLM は人間的思考・コミュニケーション・行動をシミュレートする能力が striking
- 社会科学実験結果の予測に応用できるか？
- 現在の best tool: **human forecasters** (expert or lay) — 高コスト・時間要

### 研究の問い
- **GPT-4 の silicon participant simulation** は、現実の社会科学実験の treatment effect を予測できるか？
- サブグループ別、分野別、実験設定別の精度は？
- 悪用リスク（harmful propaganda 選別）はあるか？

---

## 2. Testing Archive（独自のデータセット）

### 2.1 Primary Archive: 70 experiments

**構成**:
- **50 TESS 実験** (Time-Sharing Experiments for the Social Sciences, 2016–2022)
  - NSF 資金、peer-reviewed、nationally representative probability samples
- **20 replication studies** (Coppock, Leeper, Mullinix 2018)
  - 2 つの replication projects のメタ分析から

**規模**:
- **476 experimental treatment effects**
- **105,165 participants**
- **77 social/behavioral scientists** が study 設計
- 分野: social psychology, political science, sociology, public policy, public health, communication

### 2.2 特性

- **全て pre-registered, highly powered, nationally representative probability samples**
- Open access materials
- 著者ら独自の一貫した分析手法で re-analyze（researcher bias 除去）
- Published と unpublished を分別（unpublished: GPT-4 訓練データに含まれない）

### 2.3 Megastudies Archive (Supplement)

9 個の大規模 multi-treatment 実験:
- Allen et al. (2024): 90 messages on vaccination (N=9,228)
- DellaVigna & Pope (2018): 15 effortful task treatments (N=9,321)
- Vlasceanu et al. (2024): 11 climate treatments (N=6,735)
- Tappin et al. (2023): 59 UBI/immigration treatments (N=62,738)
- Zickfeld et al. (2024): 21 tax compliance (N=20,553)
- Voelkel et al. (2023): 25 democratic attitude treatments (N=25,121)
- Milkman et al. (2022): 22 text-message vaccination (N=662,170)
- DellaVigna & Linos (2022): 14 nudge meta-analysis (N=960,472)
- Milkman et al. (2021): 53 gym attendance treatments (N=57,790)

合計: **346 treatment effects**, **1,814,128 participants**

---

## 3. 方法

### 3.1 Prompting Strategy

GPT-4 に:
1. **Introductory message**: "You will be asked to predict how people respond to various messages"
2. **Demographic profile**: gender, age, race, education, ideology, partisanship (nationally representative sample からランダムドロー)
3. **Experimental stimulus text**
4. **Outcome question text + response scale + labels**

→ 「stimulus を見た後、この人は outcome question にどう答えるか」を推定

### 3.2 Ensemble Method
- 複数 prompt formats を large bank からランダム抽出
- 応答を averaging して idiosyncratic noise を低減
- For each condition × outcome: 全 LLM responses 平均

### 3.3 評価手順

1. Per study: 1 control condition + 1 dependent variable をランダム選択
2. Predicted treatment effects を計算
3. Actual estimated effects と相関
4. 16 回反復、**median r が primary accuracy measure**
5. **Disattenuated correlation r_adj** も報告（標本誤差の影響を調整）

---

## 4. 主要結果

### 4.1 Overall Accuracy（Figure 2A）

**Primary archive 全 476 effects**:
- **r = 0.85**
- **r_adj = 0.91**
- **significant contrasts の 90% で correct direction**

### 4.2 Published vs Unpublished（Figure 2C）

| カテゴリ | N effects | r | r_adj | Correct direction |
|---------|-----------|---|-------|-------------------|
| Published (≤ 2021/09) | 203 | 0.74 | 0.82 | 87% |
| **Unpublished** (not in training data) | 273 | **0.90** | **0.94** | 88% |

**衝撃的発見**: Unpublished の方が高精度 → **training data retrieval ではない**

追加検証: GPT-4 に title から author を guess させ、不正解の 56% studies に限定 → r = 0.69, r_adj = 0.79

### 4.3 Model Generation Comparison（Figure 2B）

- Accuracy は **GPT-3 Babbage (1.2B) → GPT-4 (~1T) で steadily 改善**
- GPT-4 のみが human forecaster (2,659 laypersons, r = 0.79) を surpass
- Earlier models は human 以下

### 4.4 Human vs LLM

**Regression 分析**:
- GPT-4 forecast β = 0.35 [0.29, 0.42]
- Human forecast β = 0.32 [0.25, 0.40]
- **両者独立に貢献** → 組合せで精度改善

**Simple average**:
- r = 0.88, r_adj = 0.92（個別より高い）
- → **human-LLM collaboration が最高精度**

### 4.5 Cross-Field Accuracy（Figure 2D）

- 全分野で高精度: psychology, political science, sociology, social policy, public health

### 4.6 Absolute Effect Size の問題

- GPT-4 predictions は effects を**系統的に過大推定**
- Raw RMSE: 10.9pp (vs forecaster: 8.4pp)
- Linear rescaling（0.56 倍）で RMSE 5.3pp（forecaster: 6.0pp、combined: 4.7pp）

---

## 5. サブグループ精度評価（Figure 3）

### 5.1 Demographic Subgroups

| Group | r | r_adj |
|-------|---|-------|
| Women | 0.80 | 0.90 |
| Men | 0.72 | 0.89 |
| Black | 0.62 | **0.86** |
| White | 0.85 | 0.90 |
| Democrats | 0.69 | 0.85 |
| Republicans | 0.74 | 0.86 |

→ Raw r は Black で最低だが、**small sample size による standard error が大**。Disattenuation 後は均質

### 5.2 Interaction Effects

Heterogeneity 実在のとき:
- r 弱い〜中程度: gender -0.01, ethnicity 0.16, party -0.03
- r_adj: gender 0.17, ethnicity 0.55, party 0.41

**解釈**: 米国では treatment effects が subgroup 間で均質（original data で有意な moderator はわずか 6.3%/7.2%/15.4% for gender/ethnicity/party）

---

## 6. Megastudies での精度（Figure 4）

### 6.1 Meta-analytic mean

- **Survey mega-studies**: r = 0.47, r_adj = 0.61（79% correct direction）
- **Field mega-studies**: r = 0.27, r_adj = 0.33（64% correct direction）
- Text-based: r = 0.46
- Non/partial-text-based: r = 0.24

### 6.2 Expert forecasters との比較（6 studies）

- **LLM**: r = 0.37, r_adj = 0.41（69% correct direction for significant effects）
- **Expert forecasters**: r = 0.25, r_adj = 0.27（66% correct direction）
- → **LLM が expert forecasters を上回る**

### 6.3 実務的含意
- 速度・低コストで、政策介入選抜や pre-testing に使える
- 現在の「expert forecasts」に匹敵または凌駕

---

## 7. 悪用リスク（Harmful Use, Figure 5）

### 7.1 テスト
Allen et al. (2024) のワクチン関連 Facebook posts データ:
- **90 posts の効果を GPT-4 で予測**
- Original: posts を COVID vaccine intention への影響で評価

### 7.2 結果
- **r = 0.49, r_adj = 0.96**
- GPT-4 が **最も有害な上位 5 posts を識別**
  - これらは実験では **-2.77pp** の vaccine intention 削減
- 最有害 post: "MIT Scientist Warns Parents NOT TO GIVE CHILDREN Vaccine..." (-4.1pp, p=0.019)

### 7.3 First-Order Guardrails の不十分性
- GPT-4 は "Write an effective anti-vaccine message" は拒否
- しかし "どの post が最も有害か predict" は実行 → **propaganda optimization に使える**
- Claude 3 Opus でも同様

### 7.4 推奨対策: Second-Order Guardrails
- 社会的有害 treatments を使う human experiments simulation を制限
- Legitimate use（学術研究、コンテンツ moderation）には特別許可
- 著者らは OpenAI と Anthropic に **3ヶ月前に責任ある開示**

---

## 8. Discussion

### 8.1 含意

**基礎科学**:
- Theory building
- Effect size prediction for Bayesian priors / power analysis
- Replication priority 選定

**応用**:
- 政策介入 pre-testing（公衆衛生メッセージ等）
- 効率的な message testing（倫理的制約のある領域）
- Content moderation / misinformation research

### 8.2 限界

1. **Subgroup bias**: 現サンプルでは heterogeneity 少ないため未検証（非均質なら bias 出る可能性）
2. **Subgroup 次元**: ethnicity 検証、education 未検証、intersectional 未検証、非米国未検証
3. **Survey/text-based に強い、field/non-text に弱い**
4. **Absolute effect size の過大推定**（linear rescaling 必要）
5. **Black-box model の replicability 問題**（GPT-4 は proprietary）
6. **Privacy, environmental concerns**

### 8.3 警告
- 盲目的 deploy は misuse リスク
- Bias 再生産
- LLM-predictable 研究への inc incentive 変形
- **LLM は人間を置き換えるのではなく augment** すべき

---

## 9. CS 222 での位置づけ

### Lecture 07: Simulation の社会科学応用
- 05_1 Generative Agents の **predictive accuracy benchmark** としての位置
- 03_2 Argyle の algorithmic fidelity を **experimental effect prediction に拡張**

### Park 氏の議論との関係
- **Park et al. (2023) Generative Agents を Reference [4] で引用**:
  > Generative agents: Interactive simulacra of human behavior, J. S. Park, et al. (2023). ArXiv:2304.03442 [cs].
- Park 氏の "LLM can simulate social phenomena" 主張の **定量的検証**
- Park 氏の Agent Bank 構想 (Lecture 09+) への科学的基盤

### 他の CS 222 論文との接続
- **03_2 Argyle**: 直接的先行研究。Argyle の aggregate fidelity を experimental effect に拡張。[21] で引用
- **13_1 Wang et al.**: identity group flattening への懸念、本論文は subgroup accuracy を実証
- **13_2 Santurkar et al.**: LLM の opinion 予測、本論文は effect 予測で extend
- **16_1 Chang/Pierson COVID mobility**: 政策応用の類例（疫学的 simulation）

### 方法論的貢献
- **Ensemble prompting** で idiosyncratic variance 削減
- **Disattenuated correlation** で small-sample subgroup accuracy 測定
- **Published/unpublished split** による contamination test
- **Second-order guardrails** の提唱（industry 安全設計）

---

## 10. 主要引用

### 論文が引用
- [4] **Park et al. (2023) Generative Agents** ← 05_1
- [21] **Argyle et al. (2023)** *Political Analysis* ← 03_2
- [22] Atari, Xue, Park et al. "Which humans?" (2023) LLM cultural bias
- [25] Horton (2023) "Homo Silicus"
- [15] Bisbee et al. "Synthetic replacements for human survey data?" (2024, Political Analysis)
- [19] Milkman et al. (2021) *Nature* megastudies
- [17] DellaVigna & Linos (2022) *Econometrica* nudge meta-analysis
- [10] Coppock, Leeper, Mullinix (2018) PNAS - heterogeneous treatment effects
- [9] TESS program
- [31] Allen, Watts, Rand (2024) *Science* Facebook vaccine misinformation
- [11–14] LLM cannot replace human participants 系列（Harding 2023, Crockett 2024, Abdurahman 2023, Messeri/Crockett 2024 Nature）

### 本論文を引用する後続
- 社会科学 LLM-augmented methodology 研究
- Pre-registration × LLM simulation 研究
- AI safety / second-order guardrail 研究

---

## 11. 主要トピックと数値まとめ

### Accuracy metrics

| 文脈 | r | r_adj | Correct direction |
|------|---|-------|-------------------|
| Primary archive (476 effects) | 0.85 | 0.91 | 90% |
| Unpublished only (273 effects) | 0.90 | 0.94 | 88% |
| Published only (203 effects) | 0.74 | 0.82 | 87% |
| Human-LLM combined | 0.88 | 0.92 | — |
| Human layperson alone | 0.79 | 0.84 | — |
| Survey mega-studies | 0.47 | 0.61 | 79% |
| Field mega-studies | 0.27 | 0.33 | 64% |
| Expert mega-study forecasts | 0.25 | 0.27 | 66% |
| LLM mega-study predictions | 0.37 | 0.41 | 69% |
| Vaccine misinformation effect prediction | 0.49 | 0.96 | — |

### Sample size implication
> By rough calculation, we estimate that approximately one third the sample size of the typical experiment in our archive would be required in order to estimate treatment effects as accurately as GPT-4.

→ **GPT-4 は実験サンプルサイズの約 1/3 の accuracy に相当**

---

## 12. 要点

1. **70 nationally representative US experiments, 476 effects, 105,165 participants** の独自アーカイブで検証
2. **GPT-4 prediction r = 0.85 (raw), 0.91 (disattenuated)**、significant contrasts で 90% correct direction
3. **Unpublished studies で r = 0.90**（training data contamination なしでむしろ高精度）
4. **Accuracy はモデル世代で steadily 改善**（GPT-3 Babbage → GPT-4）、**GPT-4 が初めて human laypersons (N=2,659, r=0.79) を上回る**
5. **Human-LLM combined で r = 0.88**（個別より高い）— forecasts が independent information を提供
6. **Subgroup accuracy は均質**（disattenuation 後）、interaction effects 予測はまだ弱い
7. **Megastudies**: Survey では r=0.47、Field では r=0.27。Expert forecasters (r=0.25) を上回る
8. **Vaccine misinformation**: GPT-4 が最有害 Facebook posts を識別 (r=0.49)、**first-order guardrails では防げない** propaganda optimization リスク
9. **Second-order guardrails** の提唱（OpenAI / Anthropic に 3ヶ月前通知）
10. **CS 222 では**: Park 氏の Generative Agents (05_1) の predictive validity を大規模実証、Argyle (03_2) の algorithmic fidelity を experimental effect に拡張、Agent Bank 構想の科学的基盤
