# Simulation論文候補の評価分析

作成日：2026-04-26
ブランチ：`claude/upload-research-papers-OhYKD`

---

## 背景

既存の Simulation 論文（Big Five + generative agent + 大学入試結果予測、Park et al. 2024系のアーキテクチャを日本教育ドメインに外挿）に続く、Harassment 論文・Clustering 論文を起点とした新たな simulation 論文の可能性を検討した。

---

## 起点となる2論文の概要

### Harassment 論文（Preprint改訂版）
- **N = 354** 日本人有職成人
- HEXACO + Dark Triad → power / gender harassment 加害傾向
- HC3-robust hierarchical regression
  - Model A：controls + Dark Triad
  - Model B：+ HEXACO 6 因子
  - Model C：+ H–H × Dark Triad 交互作用
- 主結果
  - Power harassment：Psychopathy β = .32–.40、H–H β = −.14
  - Gender harassment：H–H β = −.23、Openness β = −.24
  - HEXACO 増分：ΔR² = .032 (power) / .096 (gender)
  - Model C は marginal improvement にとどまる

### Clustering 論文（IEEE 投稿中）
- **N = 13,668** 日本人
- HEXACO 7 類型を多手法クラスタリングで同定
  - Ward + k-means、LPA、spectral clustering で確認
  - 内部指標：silhouette / Dunn / S_Dbw / C-index / Baker–Hubert Gamma / G⁺
  - 外部指標：SVM cross-classification、Cohen's κ、Adjusted Rand Index
- 7 類型
  1. Reserved
  2. Emotionally Sensitive
  3. Exploratory Extravert
  4. Conscientious Introvert
  5. Self-Oriented Independent
  6. Emotionally Volatile Extravert
  7. Reliable Introvert
- Honesty–Humility が文化的中心軸

---

## 検討した3候補

| 候補 | 概要 |
|---|---|
| **A** | Harassment 加害傾向の agent simulation。HEXACO + DT を seed → 尺度回答 → 実証 N=354 と照合 + counterfactual |
| **B** | Clustering 7 類型を persona とした type-conditional simulation。cluster 構造の再現 + 行動予測 |
| **C** | A+B 統合。7 類型で agent 生成 → harassment 尺度回答 → 類型ごとのリスクプロファイル |

---

## 候補 A 詳細分析

### 良い点
1. **実装コスト低**：既存 `simulation/agent/` pipeline をほぼ流用可能
2. **実証論文の検出力不足を補完**：ΔR² が小さく Model C が marginal だった点を、counterfactual 大量生成で補える
3. **データ整備済み**：α、descriptives、回帰係数が揃っている
4. **自己引用構造が綺麗**：Harassment ← Simulation → Clustering の hub になる

### 弱い点
1. **Construct validity の根本疑問**：LLM は実体験ではなく相関知識から harassment スコアを逆算しているだけの可能性
2. **Safety alignment による variance flattening（致命傷リスク）**：Claude が harassment を 1 人称で語ることを拒否し、応答 variance が消える危険
3. **Wang et al. (2024) flattening バイアス**：demographic group のステレオタイプ強化リスク
4. **因果主張の限界**：counterfactual を causal evidence と誤読されるリスク
5. **文化表現の問題**：Japan 特有の構成（Tou et al. の supervisor-centered coercive dynamics 等）を WEIRD 偏向の LLM が再現できる保証がない
6. **比較で「勝てる」点が限定的**：実証と同じ結果なら so what?、異なる結果なら判定不能
7. **コスト**：N=354 × 30 calls = 10,620 calls、約 $1,000–1,500
8. **新規性評価が微妙**：先行 generative agent × personality 研究が既に複数

### 総合評価
- 書ける確率：**50–60%**
- 論文として成立する確率：**30–40%**
- **ハイリスク・ハイリターン**
- 最大の決定要因は pilot 段階での safety refusal 率と response variance

---

## 候補 B 詳細分析

### 良い点
1. **Safety alignment と衝突しない**：HEXACO-60 は中性項目、AI が拒否しない
2. **安く試せる**：pilot $100–300 で回せる
3. **元データが強い**：N=13,668 の多手法妥当性で 7 類型が確立
4. **Park (2024) への新規 reframing**："individual prediction problem" → "type-level prediction"
5. **異文化表現テストとしての価値**：LLM の WEIRD bias を定量化できる
6. **既存 pipeline 流用可能**

### 弱い点 と 解決策

#### 弱点 1：tautology risk（自明・最重要）
**問題**：「Type X を seed → agent が Type X を出力」は当たり前で論文にならない

**解決策：Partial-information inference に再設計**
- 60 項目のうち 30 項目だけ提示 → 残り 30 項目を推論
- もしくは 3 因子分のみ提示 → 残り 3 因子を推論
- もしくは verbal persona description のみ → numeric HEXACO 応答が centroid に収束するかテスト
- **「LLM が HEXACO covariance structure を内部表現しているか」**という substantive question になる

#### 弱点 2：descriptive only（外的基準の欠如）
**問題**：構造再現だけでは "OK so?"

**解決策：2 段階 validation 設計**
- Stage 1（structural）：cluster structure recovery → ARI / Cohen's κ で定量化
- Stage 2（functional）：type-conditioned 行動予測 → harassment 尺度などで type 別予測妥当性
- Stage 2 で harassment 尺度を使えば自然に候補 A/C と統合できる

#### 弱点 3：LLM が日本 7 類型を欧米 4–5 類型に潰す可能性
**問題**：失敗が "LLM failure" なのか "original cluster instability" なのか判別不能

**解決策：Failure mode を研究主題にする**
- 事前登録：ARI ≥ 0.6 = 成功、0.3–0.6 = partial、< 0.3 = failure
- 複数 LLM で triangulation（Opus, Sonnet, GPT, Gemini）
  - 全モデルが類似の collapse → "systemic WEIRD bias of LLMs" という強い知見
  - モデル間で差異 → architecture / training data 依存性の証拠
- Cultural priming あり / なしで recovery が変わるかを定量化
- **Null result も "LLM bias quantification" として価値を持つ**

#### 弱点 4：persona seeding の方法論が定まらない
**問題**：numeric scores vs. verbal description で結果が変わる

**解決策：Seeding strategy を比較条件にする**
- (a) numeric only、(b) verbal description only、(c) hybrid
- どの seeding 方法が最も centroid に忠実な応答を生むかを比較
- それ自体が generative agent 方法論への貢献

#### 弱点 5：比較 baseline が不明確
**解決策：階層 baseline 設計**
- B0：完全ランダム（uniform Gaussian）
- B1：HEXACO covariance を保つランダム生成（cluster 構造なし）
- B2：単一 trait agent（type seeding なし）
- B3：提案手法（type seeding あり）

#### 弱点 6：sample size の設計
**解決策：多段階生成 + bootstrap**
- 7 types × 100–200 agents × 1 call = 700–1,400 agents（コスト ~$70–140）
- bootstrap で cluster solution stability を評価
- 元データの bootstrap stability と直接比較

#### 弱点 7：theoretical impact が直接的でない
**解決策：Infrastructure paper として位置づけ**
- "Type-conditional generative agents as a research tool for cross-cultural personality science"
- 後続研究の foundation として positioning
- IEEE Access のような methodology-friendly journal が target

### 総合評価
- 書ける確率：**80–85%**
- 論文として成立する確率：**60–70%**
- **ローリスク・ミドルリターン**
- どちらに転んでも論文化可能なのが最大の強み

---

## A vs B 比較表

| 観点 | A (Harassment) | B (Clustering) |
|---|---|---|
| 実装可能性 | △（safety risk） | ○（中性的タスク） |
| Pilot 失敗リスク | 高 | 低 |
| 新規性 | 高（が flattening 批判を受けやすい） | 中（reframing で補強可） |
| 因果主張 | できない（誤読リスク） | そもそも狙わない |
| コスト | $1,000–1,500 | $100–300 |
| 結果が null でも論文化可能か | 困難 | 可能 |
| Reviewer の厳しさ | 高（ethics + flattening） | 中（methodology のみ） |
| 自己引用 hub 性 | 高 | 中（Stage 2 で harassment 統合可） |

---

## 推奨設計

最も保守的かつ論文化確率が高いのは：

> **「Partial-information type inference + multi-LLM triangulation + functional validation via harassment outcomes」**

候補 B と C の中間で、以下の構造：

1. **Stage 1**：Partial HEXACO 入力 → Full HEXACO + cluster 推論（structural）
2. **Stage 2**：推論結果を seed として harassment 尺度に回答（functional）
3. **Triangulation**：Claude / GPT / Gemini で再現性確認

Stage 1 だけでも論文化可能、Stage 2 を加えると impact factor の高い journal を狙える、という拡張性。失敗時の damage control も設計に組み込まれている。

---

## 次の意思決定ポイント

候補 B を進める場合に決めるべきこと：

1. **partial-information の粒度**
   - 3 因子 vs 30 項目 vs verbal description
2. **target journal**
   - IEEE Access / Behavior Research Methods / Computers in Human Behavior 等
3. **pre-registration の有無**
   - OSF Registries 推奨（メタ分析論文 E5W47 と同手順）
4. **複数 LLM の選定**
   - Opus 4.7 + Sonnet 4.6 + GPT-5 + Gemini 3 など
5. **Stage 2 を含めるか**
   - 含めると C 寄り、含めないと B 純粋版

---

## 関連既存資産

- `simulation/agent/`：Opus 4.7 + Extended Thinking + Tool Use pipeline
- `simulation/HANDOFF.md`：既存 simulation 論文の状態
- `clustering/`：N=13,668 データと clustering スクリプト
- `harassment/`：N=354 データと analysis.py
- `metaanalysis/simulation_paper_introduction_draft.md`：既存 simulation 論文の introduction draft
