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

## 5. 追加の評価ポイント（補助的だが重要）

戦略 1–6 は論文構造の柱。以下は実装・記述レベルで論文の質を左右する補助的だが重要な評価軸。

### A. Prompt sensitivity 分析
- 結果が prompt の wording / order / format に依存しないか
- 複数 prompt variant で robustness を示す
- "analytic flexibility" 批判への直接的対応

### B. Refusal / null response の透明な報告
- LLM が拒否・non-answer を出した割合を必ず明示
- 隠さず exclusion criteria に含める
- 候補 A で特に重要（safety alignment による refusal が観測される領域）

### C. Model version / 日付の固定
- "Opus 4.7 as of 2026-04-26" レベルで明記
- model deprecation で再現できなくなる問題への明示的対応
- API のスナップショット ID を記録

### D. Inter-LLM reliability の定量化
- 戦略 3（triangulation）を ICC で数値化
- 「Claude と GPT の応答が類似していること」を客観的に示す
- 単に複数モデルで走らせるのではなく、agreement を測る

### E. Convergent / Discriminant validity
- LLM 出力が related trait と相関し、unrelated trait と相関しないことを確認
- 心理測定学の標準を LLM simulation にも適用
- "LLM 応答が signal であって noise ではない" 証拠

### F. Effect size の報告
- p 値だけでなく Cohen's d / R² / partial η² / 95%CI を必ず併記
- 「有意」より「どの程度」が重要

### G. Open material（再現性インフラ）
- Prompts, code, data, agent traces を OSF / GitHub で公開
- 読者が同条件で実行できるか
- LLM simulation はコストが高いので、partial reproduction 用の小規模 subset 公開も評価される

### H. Sample size の事前 power analysis
- 「予算で決めた N」ではなく「効果検出に必要な N」と justify
- 既存 simulation 論文も post hoc しか書いていないので、新論文ではここが差別化要因になる

### I. External objective criterion
- self-report 同士の照合に閉じない
- 行動ゲーム、実世界記録、生理指標などの外的基準と接続
- Park (2024) の経済ゲーム validation はこの典型

### J. Limitations の質（深さと固有性）
- ボイラープレート的な限界記述ではなく、研究固有の限界を深く議論
- LLM-specific な限界（training data cutoff, refusal pattern, prompt sensitivity 等）を必ず含める
- 自分で先回りして批判を書くことで reviewer の口を封じる

### K. Cost / compute transparency
- API コスト、calls 数、トータル token を報告
- 再現可能性と環境影響の両方に関わる
- 既存 simulation 論文の HANDOFF.md レベルで本番 paper でも書く

### L. Theoretical grounding
- 「なぜこの construct」「なぜこの measure」「なぜこの population」
- "We ran LLM on X because we could" を避ける
- 候補 B なら「なぜ HEXACO」「なぜ 7 類型」「なぜ日本」を冒頭で正当化
