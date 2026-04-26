# Deep Reading Notes — Simulation Paper Prior Research

**目的**: `prior_research/` に保存した 15 PDF を直接精読し、本研究で引用すべき key findings、数値、quotable elements を一元管理する。すべて原文 PDF（`prior_research/`）から確認した内容。

**最終更新**: 2026-04-26

**検証フォーマット**:
- 原文ノート: 原典を直接読んで取得した内容のみ
- 数値: 原文に明示された値のみ転記（推測・補間禁止）
- 著者名・タイトル・venue は `reference_index.md` ✅ と一致

---

# Tier 1: 中核論文（手法・主張の直接根拠）

## 1. Park et al. (2024) v2 — LLM Agents Grounded in Self-Reports

**Citation**: Park, J. S., Zou, C. Q., Kamphorst, J., Egan, N., Shaw, A., Hill, B. M., Cai, C., Morris, M. R., Liang, P., Willer, R., & Bernstein, M. S. (2024). *LLM agents grounded in self-reports enable general-purpose simulation of individuals* [Preprint]. arXiv:2411.10109. (v1: "Generative agent simulations of 1,000 people")

### Research Question
「Big Five を含む豊富な self-report data に基づいた LLM agent は、被験者個人の attitudes・behaviors を、新規ドメインに対しても予測できるか？」

### Method
- **Sample**: N = 1,052 米国成人（年齢・性別・人種・地域・教育・党派層別化）
- **Data collection**:
  - 2 時間の voice-to-voice AI interview（American Voices Project プロトコル、平均 6,491 words/participant）
  - GSS（General Social Survey）core モジュール
  - **BFI-44**（44 項目 Big Five）
  - 5 種の経済ゲーム（Dictator, Trust×2, Public Goods, Prisoner's Dilemma; 実報酬あり）
  - 5 種の social science 実験（Ames & Fiske 2015, Cooney et al. 2016 等）
  - **2 週間後に再測定 → test-retest consistency** が分母
- **Agent variants**:
  1. Interview agent（インタビュー transcript のみ）
  2. Survey agent（GSS + BFI のみ）
  3. Survey+Interview agent（両方）
  4. Demographic baseline（age, gender, race, ideology のみ）
  5. Persona baseline（自由記述プロフィール 1 段落）
- **Metric**: Normalized accuracy = agent accuracy / participant 2-week consistency。1.0 で「2 週間後の自分自身と同等の精度で予測できる」。

### Key Findings（exact numbers from Table & text）

| Task | Interview | Survey | Survey+Int | Demographic | Persona |
|---|---|---|---|---|---|
| **GSS** normalized accuracy | **0.83** (raw 65.67%/79.53%) | 0.82 | **0.86** | 0.74 | 0.71 |
| **Big Five (BFI-44)** normalized r | **0.80** (raw r=0.78/0.95) | 0.65 | 0.77 | 0.61 | 0.75 |
| Economic games normalized r | 0.66 | 0.38 | 0.49 | (n/s) | (n/s) |
| Replication 5 studies effect size r vs human | r=0.98 | r=0.91 | r=0.99 | r=0.93 | r=0.94 |

**Ablation**: インタビュー transcript の 80% を削除（96 分削減）→ GSS=0.79, BigFive=0.73 を維持。

**Bias reduction (DPD)**: 政治イデオロギー DPD は demographic 13.75% → interview 8.60% → survey 6.22%（survey が最低）

### Relevance to Our Paper
🟢 **直接の architecture 比較対象**。
- 彼らは Survey-only agent（BFI-44 含む）で **Big Five normalized r = 0.65**。生 r では 0.65 × 0.95 ≈ **0.62**。
- 我々は Big Five 5 dimensions のみを入力 → 偏差値予測。彼らの input は BFI-44 の 44 項目 + GSS。我々のほうが情報量少。
- 彼らの **Big Five → Big Five 予測**は「同じ概念の自己一致」。我々の **Big Five → 大学入試結果**は「異なる概念への外挿」。タスク性質が異なる。

### Quotable Elements
> "Survey agents performed comparatively worse than for the GSS (normalized correlation = 0.65), and survey-plus-interview agents were similar to interview only."

> "Even after randomly removing 80% of the interview transcript (equivalent to removing 96 minutes of the 120-minute interview)... interview-based agents still achieved an average normalized accuracy of 0.79 (std = 0.11) on the GSS and normalized correlation of 0.73 (std = 0.74) on Big Five."

---

## 2. Hewitt, Ashokkumar, Ghezae, & Willer (2024) — Predicting Social Science Experiments

**Citation**: Hewitt, L., Ashokkumar, A., Ghezae, I., & Willer, R. (2024). *Predicting results of social science experiments using large language models* [Preprint, August 8, 2024]. Stanford University.

注: Hewitt と Ashokkumar は equal contribution、order randomized。表紙で Hewitt 先頭。

### Research Question
LLM で社会科学実験の処理効果を事前予測できるか？

### Method
- **Test archive**: 70 pre-registered survey experiments（全米代表サンプル）
  - 50 from TESS (NSF-funded Time-Sharing Experiments for Social Sciences, 2016–2022)
  - 20 from a recent replication archive
- **N = 105,165** total participants across studies
- **476 experimental treatment effects** estimated
- **GPT-4** simulates demographically diverse American samples
- Ensemble prompting (multiple prompt formats averaged)

### Key Findings（exact numbers）

| Comparison | r | r_adj | Direction agreement |
|---|---|---|---|
| **Primary archive (476 effects)** | **0.85** | **0.91** | 90% |
| Unpublished only (273 effects, no training leak) | **0.90** | 0.94 | 88% |
| Published only (203 effects) | 0.74 | 0.82 | 87% |
| When GPT-4 fails to guess authors (56% of studies) | 0.69 | 0.79 | — |
| Layperson forecasters (N=2,659) | 0.79 | 0.84 | — |
| **GPT-4 + human ensemble** | **0.88** | 0.92 | — |

**Effect size bias**: GPT-4 systematically over-estimates effects. RMSE = 10.9pp raw → 5.3pp after linear rescale (scaling factor 0.56).

**Forecaster RMSE**: 8.4pp; **Combined RMSE**: 4.7pp (best).

### Relevance to Our Paper
🟢 **Group-level prediction の benchmark**。
- 我々は **個人レベル予測**（Park 2024 が言う "未解決")。彼らは集団レベル treatment effect。
- 比較すべき: 我々の synthetic outcome ground truth との Pearson r が彼らの 0.85 にどこまで迫れるか。
- ensemble strategy（複数 prompt 平均）は我々の N=30 sampling と概念上同等。

### Quotable Elements
> "We find that GPT4-derived predictions were strongly correlated with original effect sizes (r = 0.85; r_adj = 0.91)."

> "Predictive accuracy was slightly higher for the unpublished studies (r = 0.90; r_adj = 0.94)... than the published studies (r = 0.74; r_adj = 0.82)."

> "Predictions derived using GPT-4 surpassed this human accuracy benchmark, where earlier generation models did not."

---

## 3. Salganik et al. (2020) — Fragile Families Challenge

**Citation**: Salganik, M. J., Lundberg, I., Kindel, A. T., Ahearn, C. E., Al-Ghoneim, K., Almaatouq, A., Altschul, D. M., Brand, J. E., Carnegie, N. B., Compton, R. J., Datta, D., Davidson, T., Filippova, A., Gilroy, C., Goode, B. J., Jahani, E., Kashyap, R., Kirchner, A., McKay, S., ... McLanahan, S. (2020). Measuring the predictability of life outcomes with a scientific mass collaboration. *PNAS, 117*(15), 8398–8403.

100+ 著者（mass collaboration）。APA 7 では先頭 19 人 + ... + last author。

### Research Question
個人の人生結果（GPA、転居、雇用喪失等）はどこまで予測可能か？

### Method
- **Common task method**: 標準予測タスクを公開、各チームが独自手法で予測、holdout で評価
- **Data**: Fragile Families and Child Wellbeing Study, N=4,242 families × 12,942 predictor variables（waves 1–5, birth to age 9）
- **6 outcomes** (wave 6, age 15): child GPA, child grit, household eviction, household material hardship, primary caregiver layoff, primary caregiver job training participation
- **160 teams** with valid submissions; 457 applications
- **Metric**: $R^2_{holdout} = 1 - \frac{\sum(y_i-\hat y_i)^2}{\sum(y_i-\bar y_{train})^2}$（学習平均からの改善）

### Key Findings（exact numbers）
- **Best $R^2_{holdout}$ (best submission per outcome)**:
  - Material hardship: **~0.20**
  - Child GPA: **~0.20**
  - Child grit: **~0.05**
  - Eviction: **~0.05**
  - Job training: **~0.05**
  - Layoff: **~0.05**
- **Key conclusion**: Best ML submissions only "somewhat better than" 4-variable benchmark (race/ethnicity, marital status, education level, prior outcome).

### Relevance to Our Paper
🟢 **個人予測困難の canonical evidence**。
- 12,942 predictors を使ってさえ R² ≈ 0.05–0.20。
- 我々は Big Five 5 次元のみで偏差値予測。期待値の上限が低いことを正当化。
- 我々の合成 ground truth では C → outcome r ≈ 0.165（メタ分析整合）。これは R² ≈ 0.027 程度。Salganik の 0.05–0.20 範囲下限近く。

### Quotable Elements
> "Despite using a rich dataset and applying machine-learning methods optimized for prediction, the best predictions were not very accurate and were only slightly better than those from a simple benchmark model."

> "The best submissions, which often used complex machine-learning methods and had access to thousands of predictor variables, were only somewhat better than the results from a simple benchmark model that used linear regression... with four predictor variables selected by a domain expert."

---

## 4. Serapio-García et al. (2025) — A Psychometric Framework

**Citation**: Serapio-García, G., Safdari, M., Crepy, C., Sun, L., Fitz, S., Romero, P., Abdulhai, M., Faust, A., & Matarić, M. (2025). A psychometric framework for evaluating and shaping personality traits in large language models. *Nature Machine Intelligence, 7*(12), 1954–1968.

### Research Question
LLM の synthetic personality は信頼性・妥当性のある測定として確立できるか？意図的に shape できるか？

### Method
- **18 LLMs** 評価
- **Tools**: IPIP-NEO (300 item Big Five), BFI（標準 Big Five Inventory）
- **Resampling**: 1,250 paired responses per LLM
- **5-step validity framework**:
  1. Reliability (Cronbach α, Guttman λ_6, McDonald ω ≥ 0.70)
  2. Convergent validity（IPIP-NEO ↔ BFI 同一 trait, r ≥ 0.80）
  3. Discriminant validity（異 trait の r 差 > 0.40）
  4. Criterion validity（外部 criteria 例: Positive Affect, Aggression）
  5. Construct validity 統合判断

### Key Findings
- **大型 instruction-tuned LLMs** で reliability + validity 確立
- 小型/non-instruct LLM は信頼性低
- **Personality shaping**: prompt 操作で意図的 trait shape 可能
- Shaped personality は **downstream tasks**（social media post generation 等）に measurable な影響

### Relevance to Our Paper
🟢 **Big Five LLM 条件付けの psychometric foundation**。
- 我々の Opus 4.7 + thinking + Big Five 入力が「信頼できる Big Five 表現」を引き出す根拠。
- 引用箇所: Methods § agent design

### Quotable Elements
> "Personality measurements in the outputs of some LLMs under specific prompting configurations are reliable and valid; evidence of reliability and validity of synthetic LLM personality is stronger for larger and instruction-fine-tuned models."

---

# Tier 2: 強く推奨（Methods/Discussion 補強）

## 5. Aher, Arriaga, & Kalai (2023) — Turing Experiments

**Citation**: Aher, G. V., Arriaga, R. I., & Kalai, A. T. (2023). Using large language models to simulate multiple humans and replicate human subject studies. *PMLR, 202* (ICML 2023).

### Method
**Turing Experiment (TE)** = LLM を多数の人間として模擬し、古典実験を replicate。
- Ultimatum Game (Charness & Rabin 2002)
- Garden Path Sentences (linguistic experiment)
- Milgram Shock Experiment
- Wisdom of Crowds

### Key Findings
- **3/4 実験で defining qualitative results を recent LLM が replicate**
- Wisdom of Crowds で **"hyper-accuracy distortion"**: LLM はクラウドの多様性を simulate できず、正答に過度に集中

### Relevance
- **Zero-shot な TE 方法論**の原典
- 限界: hyper-accuracy distortion は教育・芸術応用で問題

### Quotable Elements
> "In the first three TEs, the existing findings were replicated using recent models, while the last TE reveals a 'hyper-accuracy distortion' present in some language models (including ChatGPT and GPT-4)."

---

## 6. Manning, Zhu, & Horton (2024) — Automated Social Science

**Citation**: Manning, B. S., Zhu, K., & Horton, J. J. (2024). *Automated social science: Language models as scientist and subjects* (NBER WP 32381).

### Method
**Structural causal models (SCM)** を使って LLM を scientist + subject として運用。Negotiation, bail hearing, job interview, auction の 4 シナリオ。

### Key Findings
- LLM は **effect sign** 予測は得意、**magnitude** は信頼できない
- **Auction**: 結果は auction theory と整合、しかし直接 LLM に「価格はいくら？」と聞くと不正確
- **Conditioning on fitted SCM** が予測を劇的改善

### Relevance
🟢 **Counterfactual / causal grounding**: 我々の Part 2（C +1 SD 介入）の方法論的支柱。

### Quotable Elements
> "When given its proposed structural causal model for each scenario, the LLM is good at predicting the signs of estimated effects, but it cannot reliably predict the magnitudes of those estimates."

> "The LLM knows more than it can (immediately) tell."

---

## 7. Jiang, Zhang, Cao, Breazeal, Roy, & Kabbara (2024) — PersonaLLM

**Citation**: Jiang, H., Zhang, X., Cao, X., Breazeal, C., Roy, D., & Kabbara, J. (2024). PersonaLLM: Investigating the ability of large language models to express personality traits. In *Findings of NAACL 2024*.

### Method
- GPT-3.5 と GPT-4 を Big Five プロファイルで persona 化
- BFI-44 受検 + 物語生成タスク
- LIWC（Linguistic Inquiry and Word Count）+ human 評価

### Key Findings
- **LLM persona の自己 BFI スコア**は割当 trait と一致（large effect sizes、5 形質すべて）
- 物語に **trait に対応する linguistic patterns** が出現
- Human 評価者は性格特性を **最大 80% 精度**で物語から推測
- ただし「これは AI 著」と知らされると精度が大幅低下

### Relevance
🟢 **Big Five プロンプト → LLM 行動再現の最近接先行研究**。
- 我々は GPT/Claude を Big Five で条件付け → 大学入試結果予測。彼らは BFI 受検 + 物語生成。
- 物語 quality が trait と一貫することは我々の thinking trace の信頼性も支える。

### Quotable Elements
> "LLM personas' self-reported BFI scores are consistent with their designated personality types, with large effect sizes observed across five traits."

> "Human evaluation shows that humans can perceive some personality traits with an accuracy of up to 80%. Interestingly, the accuracy drops significantly when the annotators were informed of AI authorship."

---

## 8. Salecha et al. (2024) — Social Desirability Bias

**Citation**: Salecha, A., Ireland, M. E., Subrahmanya, S., Sedoc, J., Ungar, L. H., & Eichstaedt, J. C. (2024). Large language models display human-like social desirability biases in Big Five personality surveys. *PNAS Nexus, 3*(12), pgae533.

### Method
- 100 項目 Big Five 質問紙
- バッチサイズ Q_n = 1, 5, 10, 20 と変動させる
- 各 batch で fresh context window
- GPT-4, GPT-3.5, Claude 3, Llama 3, PaLM-2 をテスト

### Key Findings
- **すべての LLM が socially desirable 方向にスコアを skew**:
  - ↑ Extraversion, Conscientiousness, Openness, Agreeableness
  - ↓ Neuroticism
- **GPT-4**: Q_n が 1→20 で score が **約 0.75 ポイント (1.22 human SD)** 上昇
- **Llama 3**: 0.98 SD
- **新しい・大型モデルほど bias が強い**
- LLM は **5 問だけで「これは性格テスト」と 90%+ で推論**できる
- Order randomization・paraphrasing で軽減せず
- Reverse coding で軽減はするが消えない（acquiescence bias ではない）

### Relevance
🔴 **CRITICAL LIMITATION**。我々の Methods で必ず議論：
- 我々は **LLM に BFI 受検させない**（Big Five は input、outcome は別構成概念）
- それでも LLM が「これは Big Five 文脈」と察知してデフォルト bias 発動する可能性
- 我々の prompt は外的 outcome（偏差値）を求めるので bias 影響は限定的、と論じられる

### Quotable Elements
> "When personality evaluation is inferred, LLMs skew their scores towards the desirable ends of trait dimensions (i.e. increased extraversion, decreased neuroticism, etc.)."

> "GPT-4's survey responses changing by 1.20 (human) SD and Llama 3's by 0.98 SD, which are very large effects."

---

## 9. Roberts, Luo, Briley, Chow, Su, & Hill (2017) — Personality Trait Change

**Citation**: Roberts, B. W., Luo, J., Briley, D. A., Chow, P. I., Su, R., & Hill, P. L. (2017). A systematic review of personality trait change through intervention. *Psychological Bulletin, 143*(2), 117–141.

### Method
- 207 studies のメタ分析
- 介入による Big Five 変化を集計
- True experiments + prepost change designs
- 平均介入期間: 24 weeks

### Key Findings
- **平均効果量 d = 0.37** (24 週)
- 形質別:
  - **Emotional stability** (= Neuroticism 反転) が最大変化
  - **Extraversion** 2 番目
  - Conscientiousness は中程度
- Longitudinal follow-up でも persist
- Anxiety patients が最大変化、substance use 治療が最小

### Relevance
🟢 **Part 2 反実仮想の ecological validity**。
- 「Conscientiousness を +1 SD 上げたら結果は？」を simulate する妥当性は、現実に介入で変化する事実が支える
- 24 週で d=0.37 = 約 0.4 SD → 我々の +1 SD は理論的最大効果に近い

### Quotable Elements
> "Interventions were associated with marked changes in personality trait measures over an average time of 24 weeks (e.g., d = .37)."

> "Emotional stability was the primary trait domain showing changes as a result of therapy, followed by extraversion."

---

# Tier 3: 関連研究支援

## 10. Lundberg, Brown-Weinstock, Clampet-Lundquist, Pachman, Nelson, Yang, Edin, & Salganik (2024) — Origins of Unpredictability

**Citation**: Lundberg, I., Brown-Weinstock, R., Clampet-Lundquist, S., Pachman, S., Nelson, T. J., Yang, V., Edin, K., & Salganik, M. J. (2024). The origins of unpredictability in life outcome prediction tasks. *PNAS, 121*(24), e2322973121.

### Method
- N=40 families の qualitative 深層インタビュー（FFC データから sampling）
- 数理 prediction error decomposition

### Key Findings
- **Two-source framework**:
  1. **Irreducible error**: タスク定義による不可避不確実性
  2. **Learning error**: 学習手順の限界
- 質的事例研究:「Bella」のケース — 安定家庭で平穏な幼少期 → 15 歳で脱落
- **Conclusion**: Unpredictability は many life outcome prediction tasks で予期されるべき

### Relevance
🟢 Salganik 2020 の **質的フォローアップ**。我々の paper の Discussion で Lundberg + Salganik をペア引用し、「個人予測困難は data 量の問題ではなく構造的」と主張。

---

## 11. Zimmerman (2008) — Self-Regulated Learning

**Citation**: Zimmerman, B. J. (2008). Investigating self-regulation and motivation: Historical background, methodological developments, and future prospects. *American Educational Research Journal, 45*(1), 166–183.

### Method
Review article。SRL 測定の歴史的展開（質問紙 → online 測定 → process tracing）

### Key Findings
- Initial SRL questionnaires が **学業 outcomes を significantly predict**
- "Second wave": online measures（computer traces, think-aloud, diaries, microanalyses）
- **SRL → academic achievement** は確立された経路

### Relevance
🟢 **Big Five (特に C) → SRL → academic outcome の媒介経路**の理論的根拠。Tokiwa (2025) の StudySapuri analysis を補強。

---

## 12. Tjuatja, Chen, Wu, Talwalkar, & Neubig (2024) — LLM Response Biases

**Citation**: Tjuatja, L., Chen, V., Wu, T., Talwalkar, A., & Neubig, G. (2024). Do LLMs exhibit human-like response biases? A case study in survey design. *TACL, 12*, 1011–1026.

### Method
- 9 LLMs 評価
- 古典 survey biases（acquiescence, primacy, response order 等）を test
- BiasMonkey データセット

### Key Findings
- 主要 LLM は **人間 response biases を確実に再現しない**
- **RLHF を経たモデルほど不一致**
- 同方向に変化しても、人間が変化しない perturbation に sensitive

### Relevance
- LLM ≠ human respondent の限界
- 我々の Discussion で「LLM の sensitivity vs human の sensitivity は系統的に異なる」と引用

### Quotable Elements
> "Popular open and commercial LLMs generally fail to reflect human-like behavior, particularly in models that have undergone RLHF."

---

## 13. Bisbee, Clinton, Dorff, Kenkel, & Larson (2024) — Synthetic Replacements

**Citation**: Bisbee, J., Clinton, J. D., Dorff, C., Kenkel, B., & Larson, J. M. (2024). Synthetic replacements for human survey data? The perils of large language models. *Political Analysis, 32*(4), 401–416.

### Method
- ChatGPT に persona 付与 → 11 sociopolitical groups の feeling thermometer
- ANES 2016–2020 を baseline
- Prompt wording 操作・3 ヶ月の time stability test

### Key Findings
- **平均**は ANES と整合
- **Variance が低すぎる**（実調査より分散小）
- **Regression coefficients が異なる** — 統計推論には不適
- Prompt wording の minor change で結果変化
- 3 ヶ月で同じ prompt が異なる結果

### Relevance
🔴 **Limitation citation**: 我々が ensemble (n=30) で variance を取っているのは、まさにこの問題への mitigation。

### Quotable Elements
> "Sampling by ChatGPT is not reliable for statistical inference: there is less variation in responses than in the real surveys, and regression coefficients often differ significantly from equivalent estimates obtained using ANES data."

> "We document how the distribution of synthetic responses varies with minor changes in prompt wording, and we show how the same prompt yields significantly different results over a 3-month period."

---

## 14. Dillion, Tandon, Gu, & Gray (2023) — Replace Participants?

**Citation**: Dillion, D., Tandon, N., Gu, Y., & Gray, K. (2023). Can AI language models replace human participants? *Trends in Cognitive Sciences, 27*(7), 597–600.

### Method
Science & Society review article + 464 moral scenarios の GPT-3.5 vs human comparison

### Key Findings
- **Moral judgments**: GPT-3.5 ↔ human r = **0.95** (464 scenarios)
- LLM は構造的 features（intentionality, harm, victim vulnerability 等）を捕捉
- **Divergence**: humans condemn coaches rooting for opposing team; GPT-3.5 doesn't（subtle moral conflict struggle）

### Framework: 「LLM が良い participant になりうる時」
- **Specific topics**: explicit features が判断を駆動する場合
- **Specific tasks**: 長くて退屈なタスクで LLM 有利
- **Specific research stages**: pilot, hypothesis generation
- **Specific samples**: well-represented populations

### Relevance
🟢 **Cautious optimism framework**。我々の paper は "specific topic + specific sample (Japanese high schoolers) + specific task (university admission)" として position。

### Quotable Elements
> "The moral judgments of GPT-3.5 were extremely well aligned with human moral judgments in our analysis (r = 0.95)."

> "LLMs may be most useful as participants when studying specific topics, when using specific tasks, at specific research stages, and when simulating specific samples."

---

## 15. Horton, Filippas, & Manning (2023) — Homo Silicus

**Citation**: Horton, J. J., Filippas, A., & Manning, B. S. (2023). *Large language models as simulated economic agents: What can we learn from Homo Silicus?* (NBER WP 31122). arXiv:2301.07543.

### Method
LLM を **Homo silicus**（人間の implicit 計算モデル）として扱う。古典経済実験を replicate:
- Charness & Rabin (2002) — fairness
- Kahneman, Knetsch, & Thaler (1986) — fairness in pricing
- Samuelson & Zeckhauser (1988) — status quo bias
- Oprea (2024b)
- Horton (2025)

### Key Findings
- LLM の振る舞いは **qualitatively similar** to original results
- 異なる場合は "**generative for future research**"
- LLM 入力（endowments, preferences）を変えて scenario を simulate

### Relevance
🟢 **LLM-as-economic-agent の foundational paper**。我々は LLM-as-student-agent への外挿。
- 共通の枠組み: 自然言語で endowments（Big Five）を与え、scenario（受験）の振る舞いを simulate

### Quotable Elements
> "LLMs—because of how they are trained and designed—can be thought of as implicit computational models of humans—a Homo silicus."

> "Such flexibility means that agent fidelity to human responses is not fixed: the underlying instructions or 'prompts,' which control their behavior can always be adjusted."

---

# Synthesis: 本研究への含意マップ

## 主張別 citation lineup

### 主張 1: 個人レベル予測は構造的に困難
- **Salganik et al. (2020)** PNAS: 12,942 predictors で R² ≤ 0.20
- **Lundberg et al. (2024)** PNAS: irreducible vs learning error 二分

### 主張 2: 集団レベル LLM 予測は確立済
- **Hewitt et al. (2024)**: 70 experiments, r = 0.85
- **Park et al. (2024) v2**: 1,052 individuals, normalized accuracy 0.83
- **Argyle et al. (2023)**: Silicon sampling, r = .90+ for vote prediction
- **Aher et al. (2023)**: 3/4 classic experiments replicated

### 主張 3: LLM は Big Five を再現可能
- **Serapio-García et al. (2025)** Nature MI: 信頼性・妥当性確立
- **Jiang et al. (2024)** PersonaLLM: human 80% accuracy で trait 検出可

### 主張 4: ただし重要な limitation がある
- **Salecha et al. (2024)**: Social desirability bias, 1.20 SD shift
- **Wang et al. (2024)** [既存]: identity flattening
- **Bisbee et al. (2024)**: 低分散・prompt sensitivity・time instability
- **Tjuatja et al. (2024)**: response bias 不一致
- **Aher et al. (2023)**: hyper-accuracy distortion

### 主張 5: Counterfactual/intervention は妥当
- **Manning, Zhu, Horton (2024)**: SCM-based causal simulation, sign 予測可
- **Park et al. (2023)** [CS222]: Smallville interventions
- **Roberts et al. (2017)**: 性格は介入で d=0.37 変化（ecological validity）

### 主張 6: 媒介経路（Big Five → 学習 → 成果）
- **Zimmerman (2008)** AERJ: SRL → academic outcomes
- **Tokiwa (2025)** [既存]: StudySapuri analysis
- **Tokiwa (2026)** [既存]: Meta-analysis pooled r = 0.167

### 主張 7: Cautious optimism framework
- **Dillion et al. (2023)** TICS: Specific topic + task + sample で適用可

---

# 重要な未解決事項

1. **Park et al. (2024) v2 の published version**: 現状 arXiv preprint。Nature 等への査読版が出れば差替。
2. **Hewitt et al. (2024) の published version**: 同様、preprint のまま。
3. **Roberts et al. (2017) の Big Five 別 d 値**: 現時点 emotional stability 最大としか確認、各 trait の正確な d は再読要。
4. **Salganik et al. (2020) の各 outcome 別 best R²**: GPA/material hardship が ~0.20、他 4 つが ~0.05 と確認。Figure 3 の正確な値は再読要。

---

# 引用時の注意

- **すべての ✅ citation は `reference_index.md` で APA 7 形式が確定済**。本ファイルは内容理解用。
- 数値を引用するときは原文 PDF（`prior_research/<paper>.pdf`）を再確認するのが推奨。
- **AI に citation を生成させない**（ユーザーの絶対ルール）。
