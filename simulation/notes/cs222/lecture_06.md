# Lecture 06: Interactive Worlds

- 講師: Joon Sung Park
- ソース: `docs/stanford univ AI Agents and Simulations/06 Interactive Worlds.pdf`
- 位置づけ: エージェントの次に「環境」をどう設計するか。シミュレーションの精度は**エージェントと同等に環境に依存する**という主張。

---

## 1. 環境とは何か

> An **environment** is a description of the settings that agents perceive in order to take actions.

環境の種類（同じフレームで扱える）:
- **Chat**（会話）
- **World**（仮想世界）
- **Survey**（アンケート）

シミュレーションの形式:
```
W(t) = (S_E(t), S_A1(t), …, S_AN(t))
```
エージェントと環境の相互作用。

> Today: How do we effectively describe the environments in which agents operate?

---

## 2. なぜ環境が重要か — 2つのケーススタディ

### Case study 1: Salganikの音楽ラボ実験

**Salganik, Dodds, Watts (2006) *Science* 311, 854-856** "Experimental Study of Inequality and Unpredictability in an Artificial Cultural Market"

主要発見（引用）:
> Increasing the strength of social influence **increased both inequality and unpredictability of success**. Success was also only partly determined by quality:
> - The best songs rarely did poorly
> - The worst rarely did well
> - But any other result was possible.

含意: **同じ楽曲でも、環境（社会的影響の強度）によって市場の結果が劇的に変わる**。

### LLMでの失敗例: Generative agents are overly eager to make purchases

2つの詩を提示（`Echoes of Tomorrow`: 希望 / `Alone in the Abyss`: 抑うつ的）。
ChatGPTは両方に対して「Sold!」と返す。
→ 環境設計を誤ると、エージェントの判断が不自然に一様化する。

### Case study 2: SNSのLike挙動

Social Simulacra (Park et al. UIST 2022) で観察:
> Generative agents are **overly eager to like content**.

理由: 環境がLike以外の「選択肢のコスト」を表現していない。

### Q: あなたは1日にいくつのSNS投稿に反応するか？なぜ？

講義内での問いかけ。大半の人は「意外に少ない」と答える。その背景が**メンタル・アカウンティング**。

---

## 3. Mental Accounting（Thaler）

**Richard Thaler (1985) "Mental Accounting and Consumer Choice" *Marketing Science* 4, 199-214**（2017年ノーベル経済学賞）

3つの主要概念:

1. **Categorization of money**: 人はお金を複数の「心のアカウント」（家賃・娯楽・貯蓄など）に分類する。本来は fungible（交換可能）なのに
2. **Framing effects**: 金融判断はフレーミングで変わる。利得と損失を非対称に扱う（loss aversion）
3. **Behavioral budgeting**: 非公式の予算を心で作り、アカウントごとに支出傾向が異なる（ボーナスと通常給与を別扱い）

Park氏の中心主張:

> A good simulation environment **presents the right set of choices** to the agents.
>
> The "accuracy" of a simulation is as much a function of **the agents** as it is of **the environment**.

→ エージェントの意思決定は、環境が**どのような選択肢次元**を提示するかに依存する。

### 選択肢の次元（opportunity costs）

- **Social capital**（社会関係資本）
- **Budget**（予算）
- **Emotional/mental energy**（感情的・精神的エネルギー）
- ... その他多数

環境は agents に「これらのコスト次元」を提示しなければならない。

---

## 4. 生成AI以前の環境設計

### Schelling の分離モデル
- 赤と青の点のグリッド世界
- エージェントは近隣のセルを "perceive"
- 環境 = 単純な2次元グリッド

### ゲーム理論
- 抽象シナリオ（例: 囚人のジレンマ — 自白するか？）
- エージェントは「自白を求める文」を知覚するのみ
- von Neumann & Morgenstern (1944)

### 生成エージェントでは機能するか？

> Traditional agents simplify human contingencies.
> Generative agents aim to **embody the full complexity** of human behavior.
> An abstract, stylized environment may not allow us to leverage generative agents effectively.

抽象環境では生成エージェントの表現力が無駄になる。

---

## 5. 生成エージェント時代の環境の例

| 環境タイプ | 代表研究 |
|-----------|----------|
| **Survey**（アンケート） | Argyle et al. (2023) *Political Analysis* 31, 337-355 "Out of One, Many" |
| **Experiments**（実験） | Ashokkumar et al. (2024); Horton (2023) "Homo silicus" |
| **Conversational**（会話） | Louie et al. (2024) "Roleplay-doh"（補足 06_2）; Park et al. Social Simulacra (UIST 2022); **Shaikh et al. (CHI 2024) "Rehearsal: Simulating Conflict to Teach Conflict Resolution"** |
| **World**（仮想世界） | Park et al. (UIST 2023) Generative Agents; ChatDev (ACL 2024); DiscoveryWorld; Agent Hospital |

---

## 6. Smallville の環境実装

### Scene Graph としての表現

内部表現は**シンプルな scene graph**（シーングラフ）。

引用: Rosinol et al. (2021) *Int'l J. Robotics Research* 40, 1510-1546 "Kimera: from SLAM to Spatial Perception with 3D Dynamic Scene Graphs"

### 位置決定 = 再帰的分類タスク

エージェントが「どこへ行くか」を決めるのはLLMへの再帰的プロンプトで実装:

```
!<INPUT 0>! is in {!<INPUT 1>!} in !<INPUT 2>!.
!<INPUT 3>! is going to !<INPUT 4>! that has ONLY the
following areas: {!<INPUT 5>!}
Stay in the current area if the activity can be done there. Never
go into other people's rooms unless necessary.
!<INPUT 6>! is !<INPUT 7>!. For !<INPUT 8>!, !<INPUT 9>! should
go to the following area in !<INPUT 10>!: {
```

World → Building → Room → Sub-area と再帰的に絞り込んで位置を決定。

---

## 7. 既存環境の限界

> Our virtual environments are still **stylized and simplified** compared to the real world.

- Smallvilleに店・トイレ・学校などが無かったら？
- 車も道路もない世界でエージェントは移動できるか？
- 環境設計は**リソース集約的**（膨大な設計コスト）
- SNS投稿を1つずつ提示するのは、社会資本・個人関係などの文脈を失う

---

## 8. 今後の方向性

### 環境設計の Desiderata（望ましい性質）

1. **Rich and accurate**: 世界の複雑性を符号化
2. **Scalable**: 80億人規模にスケール可能

### 方向1: Networks を環境とする

> Can networks be the environment for simulations?

- ノード+リンク（+重み）で構成
- 社会ネットワークでは個人がノード、関係強度がリンク

関連引用:
- **Granovetter (1973)** "The Strength of Weak Ties" *Am. J. Sociol.* 78, 1360-1380 — 弱い紐帯の強さ
- **Barabási (2016)** *Network Science*
- **Barabási & Albert (1999)** "Emergence of Scaling in Random Networks" *Science* 286, 509-512 — **優先的選択（preferential attachment）**

### 生成的な社会ネットワーク

**Chang, Chaszczewicz, Wang, Josifovska, Pierson, Leskovec (2024)** "LLMs generate structurally realistic social networks but overestimate political homophily"（補足 `06_1`）

- LLMは構造的に現実的な社会ネットワークを生成できる
- しかし**政治的同質性（homophily）を過大評価**する
- "realistic" は「創発的現象の類似」を指す

### 方向2: 世界も生成する（Generative Environments）

**Bruce et al. (2024) "Genie: Generative Interactive Environments"**

エージェント行動を生成するのと同じように、世界そのものを生成してしまう。

---

## 主要引用文献（Lecture 06）

### 環境が結果を決める証拠
- **Salganik, Dodds, Watts (2006)** *Science* 311, 854-856
- Park et al. (UIST 2022) Social Simulacra
- **Thaler (1985)** "Mental Accounting and Consumer Choice" *Marketing Science* 4, 199-214

### 伝統的環境
- Schelling (1971)
- von Neumann & Morgenstern (1944)

### 生成エージェント時代の環境
- Argyle et al. (2023); Horton (2023); Ashokkumar et al. (2024)
- Park et al. (UIST 2023) Generative Agents
- **Louie et al. (2024)** "Roleplay-doh"（補足 06_2）
- **Shaikh et al. (CHI 2024)** "Rehearsal: Simulating Conflict"
- Qian et al. (ACL 2024) ChatDev; Jansen et al. (2024) DiscoveryWorld; Li et al. (2024) Agent Hospital

### 環境設計
- Rosinol et al. (2021) *IJRR* Kimera / Scene Graphs

### ネットワーク環境
- **Granovetter (1973)** "Strength of Weak Ties" *Am. J. Sociol.* 78
- Barabási (2016) *Network Science*
- **Barabási & Albert (1999)** "Scaling in Random Networks" *Science* 286
- **Chang et al. (2024)** (補足 06_1)

### 生成的環境
- Bruce et al. (2024) "Genie: Generative Interactive Environments"

---

## 要点

1. **環境 = エージェントが知覚して行動するための設定記述**。Chat / World / Survey のいずれも同じフレームで扱える
2. **精度はエージェントと同等に環境に依存する**（Salganik実験: 同じ楽曲でも環境で結果が劇的に変わる）
3. Thalerの**mental accounting**が示すように、人は social capital / budget / emotional energy など**複数の選択肢次元**で判断する。環境はこれらを提示しなければならない
4. 伝統的抽象環境（Schelling、ゲーム理論）は生成エージェントの力を引き出せない
5. 現状の仮想世界（Smallville含む）は **stylized、resource-intensive**
6. 将来: **Networks**（弱い紐帯、優先的選択）や **Generative Environments**（Genie）で、スケーラブルかつ豊かな環境設計を目指す
7. 警告: Chang et al. 2024 — LLMは構造的に現実的な社会ネットを作るが **political homophily を過大評価**
