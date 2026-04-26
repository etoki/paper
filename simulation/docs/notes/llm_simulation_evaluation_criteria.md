# LLMシミュレーション論文の評価基準と良い論文の方法論

作成日：2026-04-26
ブランチ：`claude/upload-research-papers-OhYKD`
関連：`simulation_paper_candidates_analysis.md`

---

## 背景

候補B（Clusteringの7類型を使ったtype-conditional simulation）を含む、LLM-based simulation 論文の根本問題と、良い論文がそれをどう乗り越えているかを整理する。

---

## 1. シミュレーション論文に期待される成果（5レベル）

「実際と乖離がないか」は **必要条件であって十分条件ではない**。シミュレーション研究の成果は通常 5 レベルに分かれる。

### レベル1：再現性（Calibration / Replication）
- 集団レベルの分布が実データと一致するか（KS検定、Wasserstein距離）
- 個人レベルの予測精度（Pearson r、RMSE）
- Hewitt et al. (2024) の **r ≈ 0.85** が social science experiment の参照値
- これは前提。これだけでは "OK so?" になる

### レベル2：予測（Prediction）
- 実験していない新条件での結果を予測できるか
- 新しい人口・新しい介入を扱える generalizability

### レベル3：反実仮想（Counterfactual）
- 実際には測れない / 倫理的にできない介入を仮想的に試す
- 既存 Simulation 論文の Part 2 がこれ
- **因果証拠ではなく、構造保存的な思考実験**

### レベル4：機構の発見（Mechanism）
- 観察データでは見えなかった因果経路を可視化
- 心理学では稀。経済学・疫学のagent-based modelでは中心的目標

### レベル5：仮説生成（Hypothesis Generation）
- シミュレーションで見つかった現象を後続の人間実験で検証
- "シミュレーション → 実証" の研究 pipeline を作る

### LLMシミュレーションの現実的な落としどころ

| レベル | LLMで狙える？ | 何を示せばOK？ |
|---|---|---|
| 1. 再現 | ◎ | 集団分布の一致、 r ≥ 0.7 |
| 2. 予測 | ○ | hold-out sample での精度 |
| 3. 反実仮想 | △ | 構造的整合性（因果ではない） |
| 4. 機構 | × | LLMはblack box |
| 5. 仮説生成 | ○ | "次に実証で確かめるべき仮説" を提示 |

---

## 2. 全LLMシミュレーション論文に共通する根本問題

LLMシミュレーションには「**結局LLMはtraining dataの相関を再生しているだけ**」という構造的な問題が常につきまとう。

具体的には：

- **Construct validity gap**：「アンケートに答える」と「実際にその人物である」は別
- **Stochastic parrot 批判**：表面統計の再生にすぎない可能性
- **Flattening / stereotyping**（Wang et al. 2024）：demographic group のステレオタイプ化
- **WEIRD bias**：training data の英語圏偏向
- **No causal access**：相関は学習できても因果は学習できない
- **Self-report と behavior の解離**（Personality Illusion 2024）
- **Analytic flexibility**：prompt/model 選択の自由度が信頼性を破壊（threat of analytic flexibility 2024）

良い論文はこれを「回避」するのではなく、**正面から認めた上で問いを立て直す**ことで成立している。

---

## 3. 良い論文の6つの戦略

### 戦略1：個人ではなく集団・効果量を予測する（最強の回避策）

**思想**：「個人の心を再現する」と主張せず、「集団分布」や「効果サイズ」だけを予測対象にする

代表例：
- **Argyle et al. (2023)** "Out of one, many"
  - "algorithmic fidelity" という用語を発明
  - 「個人ではなく分布が一致すればOK」と明示的に bar を下げた
- **Hewitt et al. (2024)**
  - 70個の実験の効果量を予測 → r ≈ 0.85
  - 「LLMが個人の応答を当てられる」ではなく「実験全体の方向性を予測できる」

**学べること**：何を予測するかを賢く選ぶ。個人レベル予測は地雷原。

---

### 戦略2：失敗そのものを論文の主題にする

**思想**：「LLMは人間と乖離する」を**発見**として報告する

代表例：
- **Wang et al. (2024)** "Funhouse Mirrors"：5種類のsystematic distortion を分類
- **Personality Illusion (2024)**：LLM の self-report と behavior の解離を示す
- **The threat of analytic flexibility (2024)**：解析の柔軟性が信頼性を破壊する

**学べること**：null result / negative result を前向きに論文化する。「再現できなかった」も貢献。

---

### 戦略3：複数の検証基準で triangulate する

**思想**：単一指標で「合った」と言わない。複数の角度から見る

代表例：
- **Park et al. (2024)** 
  - GSS replication + Big Five + 行動ゲーム + interview hold-out
  - 4種類の validation を重ねる

**学べること**：1つの良い結果を複数の方法で確かめる。

---

### 戦略4：意味のあるbaselineと比較する

**思想**：「LLM > random」では不十分。**LLMが情報を加えているか**を示す

階層的な baseline 設計が標準：
- B0：ランダム応答
- B1：demographic 平均
- B2：単純回帰
- B3：naive LLM（persona なし）
- B4：提案手法（persona あり）

**B4 > B3 > B2 > B1 > B0** という単調増加を示せれば「persona seeding が情報を加えている」と言える

**学べること**：「人間と一致した」ではなく「ベースラインを超えた」を示す。

---

### 戦略5：Pre-registration で柔軟性を縛る

**思想**：分析を走らせる前に手順を固定し、cherry-pickingを防ぐ

- OSF Registries / AsPredicted で事前登録
- Prompt、model、validation 指標、success 基準すべて固定
- "analytic flexibility" 批判への直接的対応

**学べること**：自分の自由度が論文の弱点になることを認識する。

---

### 戦略6：「ツール」として位置づけ、「真実」と主張しない

**思想**：何のためのシミュレーションかを限定する

良い論文の控えめな claim：
- ✅ 「実証研究の事前pilotとして使える」
- ✅ 「power analysis の補助になる」
- ✅ 「仮説生成のツール」
- ✅ 「実験不可能な反実仮想を試せる」
- ❌ 「人間被験者の代替になる」
- ❌ 「個人の心を再現できる」
- ❌ 「因果関係を明らかにする」

---

## 4. 良い論文が**やらない**こと

| やらないこと | 理由 |
|---|---|
| 単一LLM・単一プロンプトで結論 | training data idiosyncrasy のリスク |
| Self-report のみで validate | LLM-LLM の循環参照 |
| 「85%accurate」だけで終わる | baseline 比較なしでは無意味 |
| Counterfactualを因果と書く | 構造保存的思考実験にすぎない |
| WEIRD偏向に触れない | 必ず突かれる |
| Refusal/null responseを隠す | 隠したことが致命傷になる |

---

## 5. 候補Bへの適用

これら6戦略を候補B（Clustering 7類型 type-conditional simulation）に適用すると、自然に強くなる：

| 戦略 | 候補Bへの適用 |
|---|---|
| 1. 集団・効果量予測 | **type-level** で予測（個人予測を回避） |
| 2. 失敗を主題化 | 類型collapseを「LLMのWEIRDバイアス定量化」として発見にする |
| 3. Triangulation | 複数LLM（Opus / Sonnet / GPT / Gemini）で実施 |
| 4. 階層baseline | B0–B3 で informativeness gain を示す |
| 5. Pre-registration | OSF Registries 登録（メタ分析論文 E5W47 と同手順） |
| 6. ツール positioning | "Type-conditional generative agents as a research tool" |

候補Bは**構造的にこの6戦略と相性が良い**。

候補Aは戦略1（個人予測を避ける）から離れにくい点で不利、というのが先の評価でB寄りになった理由でもある。

---

## 6. 主要参考文献（既に `simulation/prior_research/_text/` に格納済み）

**戦略1の系譜**
- Argyle, L. P. et al. (2023). Out of one, many. *Political Analysis, 31*(3), 337–351.
- Hewitt, L. et al. (2024). Predicting results of social science experiments using large language models.

**戦略2の系譜**
- Wang, A. et al. (2024). LLMs that replace human participants can harmfully misportray and flatten identity groups.
- Personality Illusion (2024). Revealing Dissociation Between Self-Reports & Behavior in LLMs.
- The threat of analytic flexibility in using large language models to simulate human data.

**戦略3の系譜**
- Park, J. S. et al. (2024). Generative agent simulations of 1,000 people. arXiv:2411.10109.
- Park, J. S. et al. (2023). Generative agents: Interactive simulacra of human behavior. UIST '23.

**情報的上限**
- Lundberg, I. et al. (2024). The origins of unpredictability in life outcome prediction tasks. PNAS.

---

## 7. 次のステップ候補

候補Bを進めるとして、6戦略の具体的実装を決める必要がある：

1. **戦略1の選択**：何をtype-levelで予測するか
   - HEXACO covariance 構造のみ
   - + harassment outcome（候補C寄り）
   - + online learning outcome
2. **戦略2の準備**：失敗基準を事前にどう定義するか
   - ARI ≥ 0.6 / 0.3–0.6 / < 0.3 の3段階
3. **戦略3の範囲**：何モデルで triangulate するか
4. **戦略4の baseline 数**：B0–B3 すべて実装するか、簡略化するか
5. **戦略5のレジストリ**：OSF Registries / AsPredicted どちらか
6. **戦略6の target journal**：methodology-friendly な journal を選ぶ
