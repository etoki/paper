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

- **候補 A は推奨しない**：safety alignment による variance flattening、construct validity 疑問、ethical/flattening 批判リスクが構造的
- **候補 B が最有力**：方法論的に堅実、failure mode も論文化可能、コスト低
- **候補 C（B の Stage 2 として A を内包する設計）が impact 最大**：ただし B の Stage 1 が成功した場合に限る

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
- **Honesty–Humility が文化的中心軸**として機能（協力的・信頼可能 vs 自己志向・情動易変的）
- 多手法による頑健性確認
- Daljeet et al. (2017)、Espinoza et al. (2020)、Gerlach et al. (2018)、Kerber et al. (2021) などの先行研究を非 WEIRD 文脈に拡張

---

（Part 2 以降、続く）
