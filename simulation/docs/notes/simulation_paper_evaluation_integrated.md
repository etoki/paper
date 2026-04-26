# シミュレーション論文の評価：HEXACO / Dark Triad simulation の設計と妥当性

作成日：2026-04-26
ブランチ：`main`
関連ドキュメント：`research_vision_integrated.md`（後続）

---

## Part 0：本ドキュメントの位置づけ

### 0.1 目的

既存 Simulation 論文（Big Five + generative agent + 大学入試結果予測、Park et al. 2024 系のアーキテクチャを日本教育ドメインに外挿）に続く、**Harassment 論文・Clustering 論文を起点とした新たな simulation 論文**の可能性を、保守的に評価する。

本ドキュメントは「**何が書けるか / 何が書けないか / どう書くべきか**」の方法論的評価に閉じる。

> **研究目的そのもの（なぜ simulation したいのか、究極的に何を達成したいのか）は別ドキュメント** `research_vision_integrated.md` **に分離して記述する**。

### 0.2 検討した3候補（概略）

| 候補 | 概要 | 一言 |
|---|---|---|
| **A** | Harassment 加害傾向の agent simulation | ハイリスク・ハイリターン |
| **B** | Clustering 7類型の type-conditional simulation | ローリスク・ミドルリターン |
| **C** | A + B の統合版 | 中間 |

詳細は Part 2 で展開。

### 0.3 評価フレームワーク

LLM-based simulation 論文の質を判定するため、以下の3層フレームを使用する：

1. **5レベル成果**（Calibration / Prediction / Counterfactual / Mechanism / Hypothesis Generation）— 何を達成すれば論文として成立するか
2. **良い論文の6戦略**（集団予測、失敗の主題化、triangulation、階層 baseline、pre-registration、ツール positioning）— 根本問題（construct validity gap）への対処法
3. **12 の追加評価ポイント**（A–L）— 実装・記述レベルの質

詳細は Part 3 で展開。

### 0.4 結論の早出し

- **候補 A は単独での実施を推奨しない**：safety alignment による variance flattening、construct validity 疑問、ethical/flattening 批判リスクが構造的に大きい
- **候補 B が最有力**：方法論的に堅実、failure mode も論文化可能、コスト低
- **候補 C（B の Stage 2 として A 的要素を内包する設計）が impact 最大**：ただし B の Stage 1 が成功した場合に限る

最も保守的な推奨設計：

> **「Partial-information type inference + multi-LLM triangulation + functional validation via outcome scales」**

---

## Part 1：起点となる2論文の概要

### 1a. Harassment 論文（Preprint 改訂版）

#### サンプルと方法
- **N = 354** 日本人有職成人（男性 134、女性 220）
- 年齢分布：18–19 歳 32 名、20 代 100 名、30 代 124 名、40 代 70 名、50 歳以上 28 名
- クラウドソーシング募集、Google Forms による横断調査
- 雇用状態を冒頭で確認、無職は除外
- 不注意・重複応答を除外

#### 測定尺度
- **HEXACO-60**（Wakabayashi 2014）：6 因子各 10 項目、5 件法
  - α：H–H .671 / E .830 / X .621 / A .783 / C .815 / O .804
- **SD3-J**（Shimotsukasa & Oshio 2017）：Dark Triad 27 項目、5 件法
  - α：M .767 / N .778 / P .708 / 合成 .842
- **パワハラ尺度**（Tou et al. 2017）：18 項目 3 因子（行為 12 / 雰囲気 4 / 態度 2）、3 件法
  - α：行為 .862 / 雰囲気 .730 / 態度 .760 / 合成 .880
- **ジェンダーハラ尺度**（Kobayashi & Tanaka 2010）：13 項目 2 因子（commission / omission）、5 件法
  - α：commission .876 / omission .812 / 合成 .901

#### 分析戦略
- **HC3-robust standard errors** による階層的重回帰
- 全連続変数を z 標準化、性別・地域はダミー
- 残差正規性（Shapiro–Wilk）、自己相関（Durbin–Watson）、不均一分散（Breusch–Pagan）診断
- 影響観察値（Cook's distance > 4/n）の感度分析
- 多重共線性（VIF 5–10）チェック

#### モデル定義
- **Model A**：統制変数（年齢・性別・地域）+ Dark Triad 3 因子
- **Model B**：Model A + HEXACO 6 因子
- **Model C**：Model B + H–H × Dark Triad 交互作用

#### 主結果
- **Power harassment**
  - Model A：Psychopathy のみ有意（β = .396, p < .001）
  - Model B：Psychopathy（β = .317, p < .001）、H–H（β = −.143, p = .049）
  - ΔR² = .032（A → B）、F change = 2.28, p = .036
- **Gender harassment**
  - Model A：M（β = −.138）、N（β = .180）、P（β = .165）すべて有意
  - Model B：H–H（β = −.230, p < .001）、Openness（β = −.236, p < .001）負、DT 効果は残存
  - ΔR² = .096（A → B）、F change = 6.91, p < .001
  - Gender 効果（β = −.312, p = .006、男性が高い）
  - **Machiavellianism は多変量で負係数**（抑制パターン）
- **Model C（H–H × DT 交互作用）**
  - R² with interactions：.218（power）/ .219（gender）
  - Model B からの改善は marginal にとどまる

#### Preprint からの主な改善点
- HC3 頑健推定の採用
- 増分妥当性検定（ΔR², F change）の明示
- Cook 距離による感度分析
- 性別層別分析
- Common Method Bias への明示的言及
- 結論トーンの慎重化（"perpetration" → "perpetration tendencies"）

### 1b. Clustering 論文（IEEE 投稿中・revision）

#### サンプルと方法
- **N = 13,668** 日本人成人
- SNS 経由募集、Google Forms、2024 年 7 月 1 日–9 月 30 日
- 18 歳以上、日本居住、日本語可
- Google アカウント認証で重複防止
- 不注意・極端時間応答を除外

#### 測定尺度
- **HEXACO-60** 5 件法
- α：全 6 因子で .78–.84

#### 分析戦略
- Python 3.12.3
- 多手法クラスタリング併用：
  - Ward + k-means
  - Latent Profile Analysis（LPA, Gaussian mixture, diagonal covariance）
  - Spectral clustering
- **内部妥当性指標**：
  - Silhouette、Generalized Dunn、S_Dbw、C-index、Baker–Hubert Gamma、G⁺
  - Elbow、AIC、BIC
- **外部妥当性指標**：
  - SVM cross-sample classification accuracy
  - Cohen's κ、Adjusted Rand Index（Hubert–Arabie）

#### 主結果：7 類型
| # | 名称 | 特徴 |
|---|---|---|
| 1 | **Reserved** | 高 H–H・高 A、moderate E・C |
| 2 | **Emotionally Sensitive** | 高 E、低 X、moderate A・H–H |
| 3 | **Exploratory Extravert** | 高 X・O、moderate A・C |
| 4 | **Conscientious Introvert** | 高 C、moderate O |
| 5 | **Self-Oriented Independent** | 低 A・H–H、moderate X・O |
| 6 | **Emotionally Volatile Extravert** | 高 E、低 A |
| 7 | **Reliable Introvert** | 高 A、moderate-high C・E |

#### 理論的貢献
- 欧米先行研究の 4–5 類型より**分化した 7 類型構造**を発見
- **Honesty–Humility が中心軸として機能**：協力的・信頼可能なプロファイル（Reserved, Reliable Introvert）と自己志向プロファイル（Self-Oriented Independent）を分離する重要次元
- これは謙遜・誠実を重視する日本文化の影響を反映している可能性
- 多手法による頑健性確認
- Daljeet et al. (2017)、Espinoza et al. (2020)、Gerlach et al. (2018)、Kerber et al. (2021) などの先行研究を非 WEIRD 文脈に拡張

---

## Part 2：3つの simulation 候補の詳細分析

### 2a. 3候補（A / B / C）の overview

| 候補 | 概要 | 一言評価 |
|---|---|---|
| **A：Harassment 加害傾向の agent simulation** | HEXACO + Dark Triad プロファイルを seed → harassment 尺度に回答 → N=354 実証データと照合 + counterfactual | ハイリスク・ハイリターン |
| **B：Clustering 7類型の type-conditional simulation** | 7プロトタイプを persona seed → cluster 構造の再現 + 行動予測 | ローリスク・ミドルリターン |
| **C：A + B の統合版** | 7類型で agent 生成 → harassment 尺度に回答 → 類型ごとのリスクプロファイル作成 | 中間（B 成功時の自然な拡張） |

候補Bは候補Cの Stage 1 として位置づけられるため、独立評価の観点では実質「A vs B」の対比が中心となる。

### 2b. 候補 A の詳細分析

#### 良い点

1. **実装コストが低い**
   - `simulation/agent/` の既存 pipeline（agent.py, run_pilot.py, baselines.py, reliability.py）がほぼ流用可能
   - prompts.py と tool schema の差し替えだけで動作
2. **実証論文の検出力不足を補完**
   - 実証は ΔR² = .032（power）と小さく、Model C の交互作用も marginal にとどまった
   - シミュレーションなら任意の H–H × Dark Triad 組み合わせを大量生成し、交互作用の輪郭を描ける
3. **データ整備が完了している**
   - HEXACO-60、SD3-J、harassment 尺度の α、descriptives、回帰係数が揃っている
   - Validation 指標（個人レベル r / RMSE、集団分布 KS / Wasserstein、ΔR² の再現可能性）が即定義可能
4. **自己引用構造が綺麗**
   - Harassment 実証 ← Simulation → Clustering の hub になる

#### 弱い点

1. **Construct validity の根本疑問（最大の懸念）**
   - LLM は実体験ではなく、相関知識から harassment スコアを逆算しているだけの可能性
   - Lundberg et al. (2024) の "individual prediction problem" 限界に直面
2. **Safety alignment による variance flattening（致命傷リスク）**
   - Claude は harassment を 1 人称で語ることに強い拒否傾向
   - "I would never do this" 一辺倒で variance が消える危険
   - Pilot 失敗時に論文成立不可
3. **Wang et al. (2024) flattening バイアス**
   - LLM は demographic group をステレオタイプ化
   - 「日本人男性高 Psychopathy → harassment」という結果は flattening bias の典型例として critique されやすい
4. **因果主張の限界**
   - "H–H + 1 SD で harassment 減少" は LLM の encode した相関の証拠でしかない
   - Counterfactual を causal evidence と誤読されるリスク
5. **文化表現の問題**
   - LLM training data は英語＝WEIRD 偏向
   - 日本固有の構成概念（Tou et al. の supervisor-centered coercive dynamics 等）を再現できる保証がない
   - 実証論文が non-WEIRD 売りで書かれているため整合性問題
6. **比較で「勝てる」点が限定的**
   - 実証と同じ結果なら so what?
   - 異なる結果なら LLM 欠陥なのか実証検出力不足なのか判定不能
7. **コスト**
   - N=354 × 30 calls = 10,620 calls、約 $1,000–1,500
   - Counterfactual 条件追加で倍々
8. **新規性評価が微妙**
   - Park (2023, 2024)、Argyle (2023)、Hewitt (2024) 系列で先行
   - Harassment 領域でも Salecha et al. (2025) 等 toxicity 検出系が登場

#### 総合評価
- 書ける確率：**50–60%**
- 論文として成立する確率：**30–40%**
- **ハイリスク・ハイリターン**
- Pilot 段階での safety refusal 率と response variance が決定要因

### 2c. 候補 B の詳細分析と各弱点への解決策

#### 良い点

1. **Safety alignment と衝突しない（最大の利点）**
   - HEXACO-60 は中性的な性格項目
   - Claude は正常応答、variance flattening リスクが大幅に低い
2. **安く試せる**
   - Pilot $100–300 で回せる、失敗ダメージが小さい
3. **元データが強い**
   - N = 13,668 の多手法妥当性で 7 類型が確立
4. **Park (2024) への新規 reframing**
   - "individual prediction problem" → "type-level prediction" という新しい問い
   - Lundberg (2024) の predictability ceiling 議論と接続
5. **異文化表現テストとしての価値**
   - LLM の WEIRD bias を定量化できる
6. **既存 pipeline 流用可能**

#### 弱点と解決策

##### 弱点1：tautology risk（自明・最重要）
**問題**：Type X を seed → agent が Type X を出力 は当たり前で論文にならない

**解決策：Partial-information inference に再設計**
- 60 項目のうち 30 項目だけ提示 → 残り 30 項目を推論
- もしくは 3 因子分のみ提示 → 残り 3 因子を推論
- もしくは verbal persona description のみ → numeric HEXACO 応答が centroid に収束するかテスト
- **「LLM が HEXACO covariance structure を内部表現しているか」** という substantive question になる

##### 弱点2：descriptive only（外的基準の欠如）
**問題**：構造再現だけでは "OK so?"

**解決策：2 段階 validation 設計**
- Stage 1（structural）：cluster structure recovery → ARI / Cohen's κ で定量化
- Stage 2（functional）：type-conditioned 行動予測 → harassment 尺度などで type 別予測妥当性
- Stage 2 で harassment 尺度を使えば自然に候補 A/C と統合

##### 弱点3：LLM が日本 7 類型を欧米 4–5 類型に潰す可能性
**問題**：失敗が "LLM failure" なのか "original cluster instability" なのか判別不能

**解決策：Failure mode を研究主題にする**
- 事前登録：ARI ≥ 0.6 = 成功、0.3–0.6 = partial、< 0.3 = failure
- 複数 LLM で triangulation（Opus, Sonnet, GPT, Gemini）
  - 全モデルが類似 collapse → "systemic WEIRD bias of LLMs" という強い知見
  - モデル間で差異 → architecture / training data 依存性の証拠
- Cultural priming あり / なしで recovery が変わるかを定量化
- **Null result も "LLM bias quantification" として価値を持つ**

##### 弱点4：persona seeding の方法論が定まらない
**問題**：numeric scores vs. verbal description で結果が変わる

**解決策：Seeding strategy を比較条件にする**
- (a) numeric only、(b) verbal description only、(c) hybrid
- どの seeding が最も centroid に忠実な応答を生むかを比較
- これ自体が generative agent 方法論への貢献になる

##### 弱点5：比較 baseline が不明確
**解決策：階層 baseline 設計**（候補 B 固有版）
- B0：完全ランダム（uniform Gaussian）
- B1：HEXACO covariance を保つランダム生成（cluster 構造なし）
- B2：単一 trait agent（type seeding なし）
- B3：提案手法（type seeding あり）

※ Part 3c 戦略4 の一般論版（B0–B4）とは粒度が異なる。Stage 2 の outcome 予測まで含めるなら戦略4 の 5 階層版を採用

##### 弱点6：sample size の設計
**解決策：多段階生成 + bootstrap**
- 7 types × 100–200 agents × 1 call = 700–1,400 agents（コスト ~$70–140）
- bootstrap で cluster solution stability を評価
- 元データの bootstrap stability と直接比較

##### 弱点7：theoretical impact が直接的でない
**解決策：Infrastructure paper として位置づけ**
- "Type-conditional generative agents as a research tool for cross-cultural personality science"
- 後続研究の foundation として positioning
- IEEE Access のような methodology-friendly journal が target

#### 総合評価
- 書ける確率：**80–85%**
- 論文として成立する確率：**60–70%**
- **ローリスク・ミドルリターン**
- どちらに転んでも論文化可能なのが最大の強み

### 2d. A vs B 比較表と推奨設計

| 観点 | A（Harassment） | B（Clustering） |
|---|---|---|
| 実装可能性 | △（safety risk） | ○（中性的タスク） |
| Pilot 失敗リスク | 高 | 低 |
| 新規性 | 高（が flattening 批判を受けやすい） | 中（reframing で補強可） |
| 因果主張 | できない（誤読リスク） | そもそも狙わない |
| コスト | $1,000–1,500 | $100–300 |
| 結果が null でも論文化可能か | 困難 | 可能（LLM bias 定量化として） |
| Reviewer の厳しさ | 高（ethics + flattening） | 中（methodology のみ） |
| 自己引用 hub 性 | 高 | 中（Stage 2 で harassment 統合可） |
| 倫理懸念 | 中–高 | 低 |

#### 推奨設計

最も保守的かつ論文化確率が高いのは：

> **「Partial-information type inference + multi-LLM triangulation + functional validation via outcome scales」**

候補 B と C の中間で、以下の構造：

1. **Stage 1**：Partial HEXACO 入力 → Full HEXACO + cluster 推論（structural）
2. **Stage 2**：推論結果を seed として外的尺度に回答（functional）
3. **Triangulation**：Claude / GPT / Gemini で再現性確認

Stage 1 だけでも論文化可能、Stage 2 を加えると impact factor の高い journal を狙える、という拡張性。失敗時の damage control も設計に組み込まれている。

---

## Part 3：LLM シミュレーション論文の評価フレームワーク

### 3a. シミュレーション論文に期待される 5 レベル成果

「実際と乖離がないか」は **必要条件であって十分条件ではない**。シミュレーション研究の成果は通常 5 レベルに分かれる。

#### レベル1：再現性（Calibration / Replication）
- 集団レベルの分布が実データと一致するか（KS 検定、Wasserstein 距離）
- 個人レベルの予測精度（Pearson r、RMSE）
- Hewitt et al. (2024) では **70 件の social science experiment の効果量（effect size）を LLM が r ≈ 0.85 で予測** した。これは「個人応答」ではなく「集計効果量」レベルでの再現性ベンチマーク
- これは前提。これだけでは "OK so?" になる

#### レベル2：予測（Prediction）
- 実験していない新条件での結果を予測できるか
- 新しい人口・新しい介入を扱える generalizability

#### レベル3：反実仮想（Counterfactual）
- 実際には測れない / 倫理的にできない介入を仮想的に試す
- 既存 Simulation 論文の Part 2 がこれ
- **因果証拠ではなく、構造保存的な思考実験**

#### レベル4：機構の発見（Mechanism）
- 観察データでは見えなかった因果経路を可視化
- 心理学では稀。経済学・疫学の agent-based model では中心目標

#### レベル5：仮説生成（Hypothesis Generation）
- シミュレーションで見つかった現象を後続の人間実験で検証
- "シミュレーション → 実証" の研究 pipeline を作る

#### LLM シミュレーションの現実的な落としどころ

| レベル | LLM で狙える？ | 何を示せばOK？ |
|---|---|---|
| 1. 再現 | ◎ | 集団分布の一致、 r ≥ 0.7 |
| 2. 予測 | ○ | hold-out sample での精度 |
| 3. 反実仮想 | △ | 構造的整合性（因果ではない） |
| 4. 機構 | × | LLM は black box |
| 5. 仮説生成 | ○ | "次に実証で確かめるべき仮説" を提示 |

### 3b. 全 LLM シミュレーション論文に共通する根本問題

LLM シミュレーションには「**結局 LLM は training data の相関を再生しているだけ**」という構造的な問題が常につきまとう。

具体的には：

- **Construct validity gap**：「アンケートに答える」と「実際にその人物である」は別
- **Stochastic parrot 批判**：表面統計の再生にすぎない可能性
- **Flattening / stereotyping**（Wang et al. 2024）：demographic group のステレオタイプ化
- **WEIRD bias**：training data の英語圏偏向
- **No causal access**：相関は学習できても因果は学習できない
- **Self-report と behavior の解離**（Personality Illusion 2024）
- **Analytic flexibility**：prompt/model 選択の自由度が信頼性を破壊（threat of analytic flexibility 2024）

良い論文はこれを「回避」するのではなく、**正面から認めた上で問いを立て直す**ことで成立している。

### 3c. 良い論文の 6 戦略

#### 戦略1：個人ではなく集団・効果量を予測する（最強の回避策）

**思想**：「個人の心を再現する」と主張せず、「集団分布」や「効果サイズ」だけを予測対象にする

代表例：
- **Argyle et al. (2023)** "Out of one, many"
  - "algorithmic fidelity" という用語を発明
  - 「個人ではなく分布が一致すれば OK」と明示的に bar を下げた
- **Hewitt et al. (2024)**
  - 70 個の実験の効果量を予測 → r ≈ 0.85
  - 「LLM が個人の応答を当てられる」ではなく「実験全体の方向性を予測できる」

**学べること**：何を予測するかを賢く選ぶ。個人レベル予測は地雷原。

#### 戦略2：失敗そのものを論文の主題にする

**思想**：「LLM は人間と乖離する」を**発見**として報告する

代表例：
- **Wang et al. (2024)** "Funhouse Mirrors"：5 種類の systematic distortion を分類
- **Personality Illusion (2024)**：LLM の self-report と behavior の解離を示す
- **The threat of analytic flexibility (2024)**：解析の柔軟性が信頼性を破壊する

**学べること**：null result / negative result を前向きに論文化する。

#### 戦略3：複数の検証基準で triangulate する

**思想**：単一指標で「合った」と言わない。複数の角度から見る

代表例：
- **Park et al. (2024)** 
  - GSS replication + Big Five + 行動ゲーム + interview hold-out
  - 4 種類の validation を重ねる

**学べること**：1 つの良い結果を複数の方法で確かめる。

#### 戦略4：意味のある baseline と比較する

**思想**：「LLM > random」では不十分。**LLM が情報を加えているか**を示す

階層的な baseline 設計が標準：
- B0：ランダム応答
- B1：demographic 平均
- B2：単純回帰
- B3：naive LLM（persona なし）
- B4：提案手法（persona あり）

**B4 > B3 > B2 > B1 > B0** という単調増加を示せれば「persona seeding が情報を加えている」と言える

**学べること**：「人間と一致した」ではなく「ベースラインを超えた」を示す。

#### 戦略5：Pre-registration で柔軟性を縛る

**思想**：分析を走らせる前に手順を固定し、cherry-picking を防ぐ

- OSF Registries / AsPredicted で事前登録
- Prompt、model、validation 指標、success 基準すべて固定
- "analytic flexibility" 批判への直接的対応

**学べること**：自分の自由度が論文の弱点になることを認識する。

#### 戦略6：「ツール」として位置づけ、「真実」と主張しない

**思想**：何のためのシミュレーションかを限定する

良い論文の控えめな claim：
- ✅ 「実証研究の事前 pilot として使える」
- ✅ 「power analysis の補助になる」
- ✅ 「仮説生成のツール」
- ✅ 「実験不可能な反実仮想を試せる」
- ❌ 「人間被験者の代替になる」
- ❌ 「個人の心を再現できる」
- ❌ 「因果関係を明らかにする」

### 3d. 追加の 12 評価ポイント（A–L）

戦略 1–6 は論文構造の柱。以下は実装・記述レベルで論文の質を左右する補助的だが重要な評価軸。

#### A. Prompt sensitivity 分析
- 結果が prompt の wording / order / format に依存しないか
- 複数 prompt variant で robustness を示す
- "analytic flexibility" 批判への直接的対応

#### B. Refusal / null response の透明な報告
- LLM が拒否・non-answer を出した割合を必ず明示
- 隠さず exclusion criteria に含める
- 候補 A で特に重要（safety alignment による refusal が観測される領域）

#### C. Model version / 日付の固定
- "Opus 4.7 as of 2026-04-26" レベルで明記
- model deprecation で再現できなくなる問題への明示的対応
- API のスナップショット ID を記録

#### D. Inter-LLM reliability の定量化
- 戦略 3（triangulation）を ICC で数値化
- 「Claude と GPT の応答が類似していること」を客観的に示す
- 単に複数モデルで走らせるのではなく、agreement を測る

#### E. Convergent / Discriminant validity
- LLM 出力が related trait と相関し、unrelated trait と相関しないことを確認
- 心理測定学の標準を LLM simulation にも適用
- "LLM 応答が signal であって noise ではない" 証拠

#### F. Effect size の報告
- p 値だけでなく Cohen's d / R² / partial η² / 95%CI を必ず併記
- 「有意」より「どの程度」が重要

#### G. Open material（再現性インフラ）
- Prompts, code, data, agent traces を OSF / GitHub で公開
- 読者が同条件で実行できるか
- LLM simulation はコストが高いので、partial reproduction 用の小規模 subset 公開も評価される

#### H. Sample size の事前 power analysis
- 「予算で決めた N」ではなく「効果検出に必要な N」と justify
- 既存 simulation 論文も post hoc しか書いていないので、新論文ではここが差別化要因になる

#### I. External objective criterion
- self-report 同士の照合に閉じない
- 行動ゲーム、実世界記録、生理指標などの外的基準と接続
- Park (2024) の経済ゲーム validation はこの典型

#### J. Limitations の質（深さと固有性）
- ボイラープレート的な限界記述ではなく、研究固有の限界を深く議論
- LLM-specific な限界（training data cutoff, refusal pattern, prompt sensitivity 等）を必ず含める
- 自分で先回りして批判を書くことで reviewer の口を封じる

#### K. Cost / compute transparency
- API コスト、calls 数、トータル token を報告
- 再現可能性と環境影響の両方に関わる
- 既存 simulation 論文の HANDOFF.md レベルで本番 paper でも書く

#### L. Theoretical grounding
- 「なぜこの construct」「なぜこの measure」「なぜこの population」
- "We ran LLM on X because we could" を避ける
- 候補 B なら「なぜ HEXACO」「なぜ 7 類型」「なぜ日本」を冒頭で正当化

### 3e. 良い論文が**やらない**こと

| やらないこと | 理由 |
|---|---|
| 単一 LLM・単一プロンプトで結論 | training data idiosyncrasy のリスク |
| Self-report のみで validate | LLM-LLM の循環参照 |
| 「85% accurate」だけで終わる | baseline 比較なしでは無意味 |
| Counterfactual を因果と書く | 構造保存的思考実験にすぎない |
| WEIRD 偏向に触れない | 必ず突かれる |
| Refusal / null response を隠す | 隠したことが致命傷になる |

---

## Part 4：参考文献

### 4.1 LLM シミュレーション方法論（戦略 1：集団・効果量予測の系譜）
- Argyle, L. P., Busby, E. C., Fulda, N., Gubler, J. R., Rytting, C., & Wingate, D. (2023). Out of one, many: Using language models to simulate human samples. *Political Analysis, 31*(3), 337–351.
- Hewitt, L., Ashokkumar, A., Ghezae, I., & Willer, R. (2024). *Predicting results of social science experiments using large language models*.
- Park, J. S., O'Brien, J. C., Cai, C. J., Morris, M. R., Liang, P., & Bernstein, M. S. (2023). Generative agents: Interactive simulacra of human behavior. *UIST '23*.
- Park, J. S., Zou, C. Q., Shaw, A., Hill, B. M., Cai, C., Morris, M. R., Willer, R., Liang, P., & Bernstein, M. S. (2024). *Generative agent simulations of 1,000 people*. arXiv:2411.10109.
- Park, J. S. (2024). *CS 222: AI agents and simulations* [Lecture notes]. Stanford University.

### 4.2 LLM シミュレーション批判（戦略 2：失敗の主題化の系譜）
- Wang, A., Morgenstern, J., & Dickerson, J. P. (2024). *LLMs that replace human participants can harmfully misportray and flatten identity groups*. arXiv:2402.01908.
- *The Personality Illusion: Revealing dissociation between self-reports and behavior in LLMs* (2024). [in `simulation/prior_research/_text/`]
- *The threat of analytic flexibility in using large language models to simulate human data* (2024). [in `simulation/prior_research/_text/`]
- *Digital Twins are Funhouse Mirrors: Five systematic distortions* (2024). [in `simulation/prior_research/_text/`]

### 4.3 個人予測の情報的上限
- Lundberg, I., Brand, J. E., & Jeon, N. (2024). The origins of unpredictability in life outcome prediction tasks. *PNAS, 121*(24), e2322973121.
- Salganik, M. J. et al. (2020). Measuring the predictability of life outcomes with a scientific mass collaboration. *PNAS, 117*(15), 8398–8403.

### 4.4 関連 LLM-agent 研究（既収集）
- *A foundation model to predict and capture human cognition* (2024). [in `simulation/prior_research/_text/`]
- *LLM Agent-Based Simulation of Student Activities and Mental Health Using Smartphone Sensing Data* (2024). [in `simulation/prior_research/_text/`]
- *Personality-Driven Student Agent-Based Modeling in Mathematics Education: How well do student agents align with human learners?* (2024). [in `simulation/prior_research/_text/`]
- *Predicting personality from patterns of behavior collected with smartphones* (2024). [in `simulation/prior_research/_text/`]

### 4.5 HEXACO 測定・モデル
- Ashton, M. C., & Lee, K. (2007). Empirical, theoretical, and practical advantages of the HEXACO model of personality structure. *Personality and Social Psychology Review, 11*(2), 150–166.
- Wakabayashi, A. (2014). A sixth personality domain that is independent of the Big Five domains: The psychometric properties of the HEXACO Personality Inventory in a Japanese sample. *Japanese Psychological Research, 56*, 211–223.

### 4.6 Harassment / Dark Triad 測定（候補 A 関連）
※ 以下は Harassment 論文（`harassment/paper/Manuscript_only.docx`）から引用。著者・年・タイトルは確定しているが、掲載誌の正確な書誌情報は原稿を参照
- Shimotsukasa, T., & Oshio, A. (2017). Japanese version of the Short Dark Triad (SD3-J).
- Tou, S. et al. (2017). Workplace Power Harassment Scale（パワハラ尺度、3 因子 18 項目）.
- Kobayashi, A., & Tanaka, K. (2010). Gender Harassment Scale（ジェンダーハラ尺度、commission/omission 2 因子）.

### 4.7 Clustering 方法論先行研究（候補 B 関連）
- Daljeet, K. N., Bremner, N. L., Giammarco, E. A., Meyer, J. P., & Paunonen, S. V. (2017). Taking a person-centered approach to personality: A latent-profile analysis of the HEXACO model of personality. *Journal of Research in Personality, 70*, 241–251.
- Espinoza, J. A., Daljeet, K. N., & Meyer, J. P. (2020). Establishing the structure and replicability of personality profiles using the HEXACO-PI-R. *Nature Human Behaviour, 4*(7), 713–724.
- Gerlach, M., Farb, B., Revelle, W., & Amaral, L. A. N. (2018). A robust data-driven approach identifies four personality types across four large data sets. *Nature Human Behaviour, 2*(10), 735–742.
- Kerber, A., Roth, M., & Herzberg, P. Y. (2021). Personality types revisited—A literature-informed and data-driven approach to an integration of prototypical and dimensional constructs of personality description. *PLOS ONE, 16*(1), e0244849.
- She, M. H. C., Ronay, R., & den Hartog, D. N. (2025). The Sociable and the Deviant: A Latent Profile Analysis of HEXACO and the Dark Triad. *Journal of Business Ethics, 199*, 529–547.

### 4.8 Clustering validity 指標
- Hubert, L., & Arabie, P. (1985). Comparing partitions. *Journal of Classification, 2*(1), 193–218.
- Halkidi, M., & Vazirgiannis, M. (2001). Clustering validity assessment: Finding the optimal partitioning of a data set. *Proc. IEEE ICDM*, 187–194.
- Rousseeuw, P. J. (1987). Silhouettes: A graphical aid to the interpretation and validation of cluster analysis. *Journal of Computational and Applied Mathematics, 20*, 53–65.

### 4.9 著者の関連自己引用
- Tokiwa, E. (2025). Who excels in online learning in Japan? *Frontiers in Psychology, 16*, 1420996.
- Tokiwa, E. (2026). *Big Five personality traits and academic achievement in online learning environments: A systematic review and meta-analysis* [Preprint]. OSF. https://doi.org/10.17605/OSF.IO/E5W47

---

## 関連既存資産（このリポジトリ内）

- `simulation/agent/` — Opus 4.7 + Extended Thinking + Tool Use pipeline
- `simulation/HANDOFF.md` — 既存 simulation 論文の状態
- `simulation/prior_research/_text/` — 上記参考文献の PDF + テキスト抽出
- `clustering/` — N=13,668 データと clustering スクリプト
- `harassment/` — N=354 データと analysis.py
- `clustering/paper_IEEE/Manuscript_IEEE_rivision.docx` — Clustering 論文 IEEE 投稿原稿
- `harassment/paper/Manuscript_only.docx` — Harassment 論文原稿

---

**本ドキュメントは「方法論的評価」に閉じます。**  
**研究目的そのもの（なぜ simulation するか）は** `research_vision_integrated.md` **を参照してください。**

