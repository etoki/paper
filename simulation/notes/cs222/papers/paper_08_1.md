# 08_1 — The Strength of Issues: Using Multiple Measures to Gauge Preference Stability, Ideological Constraint, and Issue Voting

## 書誌情報

- 著者: **Stephen Ansolabehere**（MIT, Department of Political Science）／ **Jonathan Rodden**（Stanford University, Department of Political Science）／ **James M. Snyder, Jr.**（MIT, Departments of Political Science and Economics）
- 掲載誌: *American Political Science Review* Vol. 102, No. 2 (May 2008), pp. 215–232
- DOI: 10.1017/S0003055408080210
- Lecture 08 の補足論文。個人のイデオロギー/態度の安定性と「issue voting」の可能性についての実証的再評価

---

## 1. 研究問題

アメリカ選挙研究の「古典的通説」は、Converse (1964) "The Nature of Belief Systems in Mass Publics" に代表される:

- 有権者の政策態度は不整合（no constraint）かつ不安定（no stability）
- よって政策位置は投票決定にほとんど効かない（issues matter little）
- 代わりに party identification（政党帰属）や候補者印象が投票を説明する

本論文の問い:

> Are voter preferences really incoherent and unstable, or are these findings a manifestation of **measurement error** associated with individual survey items?

つまり **Converse 的診断は、単一項目の調査設問に由来する測定誤差のアーティファクトではないか?**

---

## 2. 中心主張（アブストラクト）

> A venerable supposition of American survey research is that the vast majority of voters have incoherent and unstable preferences about political issues, which in turn have little impact on vote choice. We demonstrate that these findings are manifestations of measurement error associated with individual survey items.

二本柱の発見:

1. **同一の広い争点領域**（経済への政府関与、道徳問題、人種問題、等）に属する複数の調査項目を**平均化**すると、測定誤差が劇的に減少し、安定で構造化された**真の態度**が現れる。項目数を増やすほど安定性は単調に上昇し、党派帰属（party ID）の安定性に接近する。
2. 測定誤差を補正すると、**争点選好は大統領投票選択において大きな説明力**を持ち、これも党派帰属に匹敵する効果量となる。

---

## 3. 測定誤差モデル（理論）

### 標準モデル

各観測項目 Wi は真値 Xi とランダム誤差 ei の和:

- Wi = Xi + ei
- E[ei] = 0, Var(ei) = σ²_ei
- 誤差 ei は Xi, Xj, ej (i≠j) と無相関

2つの項目間の観測相関は真相関よりゼロ側にバイアスされる:

- ρ²_W1,W2 = ρ²_X1,X2 × [σ²_X1 σ²_X2 / ((σ²_X1 + σ²_e1)(σ²_X2 + σ²_e2))] < ρ²_X1,X2

バイアスは signal-to-noise 比 σ²_X/σ²_e に依存。

### K 項目平均によるシグナル抽出

K 項目の平均を W̄i = (1/K) Σ Wik とすると:

- σ²_W̄i = σ²_Xi + σ̄²_ei / K
- ρ²_W̄1,W̄2 → ρ²_X1,X2 (K → ∞)

**誤差分散が 1/K の速度で縮む**（classical measurement theory, Lord & Novick 1968, Kuder & Richardson 1937 の系譜）。

項目を追加する判断基準（大きな K における rule of thumb）:

> Add an item if and only if the variance of the measurement error in that item is less than **twice** the measurement error of the existing items.

### パラメータ推定

単純な式で信号ノイズ比 σ²_X/σ²_e と真の相関 ρ_X1,X2 が計算可能:

- σ²_X/σ²_e = [1/ρ_W1k,W2k − 1/ρ_W̄1,W̄2 / K] / [1/ρ_W̄1,W̄2 − 1/ρ_W1k,W2k]
- ρ_X1,X2 = (K − 1) / [K/ρ_W̄1,W̄2 − 1/ρ_W1k,W2k]

overidentified ゆえにモデルのテストも可能。

---

## 4. データ

American National Election Study (ANES) パネル:
- 1956–1960 パネル
- 1972–1976 パネル
- 1990–1992 パネル
- 1992–1996 パネル

各パネルの最初と最後の年に同一の争点項目が反復されている。争点領域別に項目を整理（経済、道徳、人種、女性、法秩序など）。スケール化は主成分分析（principal factors factor analysis）。

> In all cases we find a single dominant dimension.

各争点領域で単一の主次元が存在。factor scores と単純平均の相関は .97 以上で、実質的にどちらを使っても同じ。

---

## 5. 結果1: 時系列安定性（Table 1 のハイライト）

**個別項目 vs スケールの時系列相関**:

| 争点領域 | 項目数 | スケールの相関 | 個別項目の平均相関 |
|---------|-------|---------------|------------------|
| Economic Issues 1992-1996 | 25 | **.76** | .42 |
| Moral Issues 1992-1996 | 12 | **.83** | .52 |
| Economic Issues 1990-1992 | 23 | **.76** | .41 |
| Racial Issues 1990-1992 | 11 | **.77** | .51 |
| Party ID (single item) 1992-1996 | 1 | .79 | — |
| Party ID (multi-item) 1992-1996 | 3 | .80 | .67 |

**5つの争点領域（1990-92, 1992-96）で平均 .77** — 個別項目平均 .46 を大きく上回る。**Converse が観察した .2〜.45 の不安定性は消える**。

> Perhaps most striking, the issue scales exhibit a degree of intertemporal stability **on par with party identification**.

### Figure 1: 項目数と安定性（Monte Carlo）

経済争点 (1990–92) で項目数 K を 1〜23 に変化させ、ランダムに K 項目選んでスケール構築→時系列相関を計算。

- K=1 単一項目: 相関 .41
- K=23 全項目: 相関 .76（党派 ID ≈ .79 の理論上限に接近）
- 曲線は滑らかに上昇し凹に収束

### 信号ノイズ比

1990 & 1992 経済争点のデータから算出: 平均 signal-to-noise ratio ≈ **.95**。
→ 個別項目の分散の **半分が真の選好、半分が測定誤差**。Achen (1975) の 1958–1960 パネル分析と一致。

---

## 6. 結果2: Within-Survey 安定性（"Constraint"）

同一サーベイ内で M 項目を半分に分割 → 各半分からスケール構築 → スケール同士の相関。Monte Carlo で平均化。

**Table 2 のハイライト** (14項目以上の領域):

| 争点領域 | 項目数 | スケール相関 | 個別ペア平均 |
|---------|-------|-------------|-------------|
| Economic Issues 1996 | 34 | .84 | .24 |
| Moral Issues 1996 | 14 | .65 | .25 |
| Racial Issues 1972 | 22 | .80 | .27 |
| Economic Issues 1990 | 27 | .74 | .19 |

項目を増やすと within-survey 相関も単調に上昇し 1.0 に接近 → **constraint 不足の観察もほぼ全て測定誤差**。

---

## 7. 結果3: Issue Voting（Table 5）

大統領投票（Republican=1, Democrat=0）の probit 回帰。

### 個別項目を使うと（1992 ANES, 1996 ANES）

| 指標 | 1992 | 1996 |
|------|------|------|
| Party ID 係数 | 1.18 (.12) | .99 (.08) |
| Ideology Scale 係数 | .58 (.12) | .54 (.10) |
| 経済争点 平均 |coef| | .09 | .07 |
| 経済争点 fraction significant (.05) | .06 | .05 |
| 道徳争点 平均 |coef| | .14 | .12 |
| 道徳争点 fraction significant | .08 | .06 |

個別項目では statistical significance がほぼ出ない → 「issue voting は周辺的」という通説。

### スケール化すると

| 指標 | 1992 | 1996 |
|------|------|------|
| Party ID | .99 (.09) | .86 (.08) |
| Ideology Scale | .40 (.10) | .53 (.09) |
| **Economic Issues Scale** | **.33 (.08)** | **.52 (.09)** |
| **Moral Issues Scale** | **.43 (.09)** | **.29 (.07)** |

> The combined effect of changing both the Economic Issues Scale and the Moral Issues Scale by 1 SD is **nearly as large, or even larger than, a change of 1 SD in party identification**.

Pseudo-R² も大幅に改善（individual items が 1–2% しか加えないのに対し、スケールは 7–9% 加える）。

---

## 8. Black-White モデル（異質性）の検討

Converse の Black-White モデル: 一部の人は正確に回答、他は純ランダム → 低洗練層では issue voting が消える。

**本論文の反論**:
- 政治情報量別に分析（very high/high/medium/low/very low）
- 各群で項目数が増えるほど相関が上昇
- very low 群でも 23 項目スケールで分散 .18（純ランダム なら 1/23 = .043 のはず）
- 教育や情報と issue scale の **交互作用項はどれも非有意**

> Survey respondents with high education and high information may simply be **better test takers, not better citizens**.

= 高洗練 ≠ issue voter。全ての有権者が自らの選好から投票。

---

## 9. CONCLUSIONS のコア命題

1. **Voters have stable policy preferences** — Converse の nonattitudes 命題は誤り。項目平均相関は .8 に接近。
2. **Issues matter to the electorate** — 測定誤差補正で党派 ID に匹敵する効果。
3. **Heterogeneity は投票には影響しない** — 洗練度と issue voting の交互作用は null。

実務的提言:
- Multiple items を使う（ask more questions）
- 項目バッテリー（battery design）を重視する
- ideology 測定も議会 roll-call 風のスケール化が望ましい

---

## 10. CS 222 での位置づけ

### Lecture 08: 個人モデル（Individual Models）

本論文は Lecture 08 "個人の内的状態をどう計測するか" の系譜に位置する。Park氏の問いは:

- **Generative agent**（Park et al. 2023）のパーソナリティを設定するとき、どの粒度で属性を入れるべきか?
- 単一質問の応答は「ノイズだらけの真値」である—個人シミュレーションの**条件付け情報**も同様にノイズを含む
- 複数測定の平均化が示す「1/K 誤差縮約」は、**LLM のペルソナ条件付けで "多数の small observations" を与えると挙動が安定化する**直観の情報理論的基礎

### Park氏の議論との接続

- シミュレーション研究では「1人の人の単一の行動」から「その人の真の属性」を推論する場面が多い
- 本論文は **シングルポイント観察では不十分、複数観察を集約すべき** という原則を示す
- Lecture 08 における Agent Bank / ペルソナ構築の方針（できるだけ多様な観察情報を取り込む）を正当化

### 関連する CS 222 議論
- 13_1 "LLMs misportray and flatten identity groups": single-item 質問でアイデンティティを扱う危険
- 13_2 "Whose Opinions Do Language Models Reflect?": 単一項目ポーリングでの LLM 評価の限界
- 06_2 Roleplay-doh: ロールプレイ条件付けにも「項目の多重性」の効果

---

## 11. 主要引用

### 本論文が挑戦する文献
- **Converse, P. E. (1964)** "The Nature of Belief Systems in Mass Publics" — 本論文の主要対照
- **Campbell, Miller, Converse, Stokes (1960)** *The American Voter*
- **Kinder, D. R. (1998)** "Opinion and Action in the Realm of Politics"
- **Polsby & Wildavsky (2000)** — テキストブック的通説

### 本論文が依拠する文献
- **Achen, C. H. (1975)** "Mass Political Attitudes and the Survey Response" *APSR* 69 — 測定誤差モデルの原点
- **Erikson, R. S. (1978, 1979)** — 3-wave panel の分析
- **Feldman, S. (1989)** "Measuring Issue Preferences" — 5-wave panel
- **Kuder & Richardson (1937)** — 心理計量の古典
- **Lord & Novick (1968)** *Statistical Theories of Mental Test Scores*

### 異論・補足
- Green, Palmquist, Schickler (2002) *Partisan Hearts and Minds* — Party ID の安定性
- Zaller (1992) *The Nature and Origins of Mass Opinion*

---

## 12. 限界と留保

- ANES と GSS を主に扱う（他国のサーベイでの検証は今後）
- non-random measurement error（Green & Citrin 1994）は別論文扱い
- autocorrelation in errors の可能性（Footnote 3 で言及）— ANES でわずかに存在
- 因果関係（Party ID ⟷ Issues）は不明。本論文は両方を右辺に置いて比較するが、全効果は測定していない
- 単一次元モデルを仮定（実際は各争点領域内で単一因子が強く支配）

---

## 要点

1. アメリカ選挙研究の通説（Converse 1964）の「nonattitudes, no issue voting」は**単一項目サーベイの測定誤差のアーティファクト**
2. 複数項目を平均化すれば測定誤差が 1/K で縮む — **信号ノイズ比は真の情報を浮かび上がらせる**
3. ANES データで争点スケールの時系列相関は **.77 平均**（個別項目 .46）で、**党派 ID の .79 に並ぶ**
4. within-survey constraint も項目数を増やすと .84 まで上昇（半分割スケール相関）
5. **Issue voting は党派 ID に匹敵する効果を持つ**（Economic + Moral 1 SD の投票への影響は Party ID 1 SD と同等以上）
6. 政治情報量・教育と issue voting の交互作用は null — 高洗練層だけの現象ではない
7. 方法論的教訓: 調査設計で **batteries of questions** を重視せよ
8. CS 222 への含意: 個人エージェント構築でも**複数の弱い観察を集約する**アプローチが真値抽出の鍵
