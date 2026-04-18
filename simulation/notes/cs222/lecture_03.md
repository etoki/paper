# Lecture 03: Individuals, Groups, and Populations

- 講師: Joon Sung Park
- ソース: `docs/stanford univ AI Agents and Simulations/03 Individuals, Groups, and Populations.pdf`
- 位置づけ: シミュレーションの「量子単位」である個人エージェントの定義と、分析粒度（個人/集団/人口）が提供する違いを明確化。

---

## 0. 位置づけ（ロードマップ上）

前回までの到達点:
- Lecture 1: シミュレーションには豊かな歴史があり、今エキサイティングな機会がある
- Lecture 2: シミュレーションは wicked problems に取り組むべき

本講義からはシミュレーションの**構成要素**を設計する段階へ:

> - What are the building blocks of simulations?
> - How do we create individual agents?
> - How do we create the environment?
> - How do we establish interactions between agents?
> - How do we evaluate the agents?
> - How might we envision the language and schema for building simulations?

対応する課題:
- **Assignment 1**: 個別エージェントを作る
- **Assignment 2**: エージェント間の相互作用を作る
- **AgentBank-CS222**（授業内活動）
- **最終プロジェクト**

---

## 1. シミュレーションの"量子単位" = 個人

> Individuals are the **quantum unit** of simulations.

シミュレーションの形式定義を再掲:
```
W(t+1) = f(S_E(t), S_A1(t), …, S_AN(t))
```

**Q: 我々は "個人" エージェントをどう定義するか？**

- 生成エージェントの例（Park et al. UIST 2023）は個人の定義を明示
- セルオートマトン（von Neumann, Wolfram）は個人の定義を持たない

→ シミュレーションの成否は「個人をどう定義するか」に強く依存する。

---

## 2. 分析の3粒度

| レベル | 定義 | 答えられる問い |
|--------|------|----------------|
| **Individuals** | 固有の性格・信念・外見・行動を持つ単一の人物 | この個人は特定の検索結果を好むか？ 実験処置にどう反応するか？ |
| **Groups** | 相互作用（regular contact）と相互依存（influence each other）で区別される集合 | 2人の対立をどう解消するか？ クラウドワーカーの集団は協力できるか？ |
| **Populations** | 共通属性（地理・種・人口統計）で結ばれた集団 | 民主党員は共和党員よりこの政策を支持するか？ 高齢者は若者より読書時間が長いか？ |

粒度の選択基準:
> The granularity of simulations is often chosen based on **practicality and the specific goal at hand**.
> （Card, Moran, Newell 1983 の引用）

各粒度の焦点:
- **Groups**: 構成個体間の相互作用の効果を理解
- **Populations**: 集計レベルでの介入の処置効果（treatment effect）を理解

→ **「どの粒度をシミュレートしたいか」を理解することが、求める答えを得る鍵**。

---

## 3. 現状分析: 生成AIシミュレーションは人口レベルに偏っている

> Recent works that leverage generative AI to simulate human behaviors predominantly take the approach of **modeling populations**.

代表例:
- Argyle et al. (2023) "Out of One, Many" *Political Analysis* 31, 337-355
- Horton (2023) "Homo silicus"
- Ashokkumar et al. (2024) "Predicting Social Science Experiments"

### なぜ人口レベルに偏るのか？

**Reason 1: 評価がしやすい**
既存の人口集団研究（世論調査、社会科学実験）を再現することで、シミュレーションを評価できる。

**Reason 2: 個人モデル構築の方法が不明**
個人データの希少性、個人行動の非一貫性などにより、「個人のモデル」を作る道筋が確立していない。

### 人口レベルの落とし穴

> But population-level simulations ought to reckon with bias and stereotyping.

- **Wang, Morgenstern, Dickerson (2024)** "LLMs cannot replace human participants because they cannot portray identity groups" — 現行LLMはアイデンティティ集団を**誤描写（misportray）**かつ**平坦化（flatten）**する
- **Cheng, Piccardi, Yang (2023)** *EMNLP 2023* — ステレオタイプが埋め込まれていることを実証

---

## 4. 各粒度モデルの作り方（Park氏の方針）

### 個人モデル
> We build models of individuals by providing a **detailed description of a persona** representing a person.

（Social Simulacra, Park et al. UIST 2022）

### 集団モデル
> We build models of groups by providing **memories** to the models of individuals so that they can interact.

（Generative Agents, Park et al. UIST 2023）

個人にペルソナ、集団に記憶（相互作用の履歴）を与えることで段階的に複雑性を増やす。

### 評価方法: 収束と発散（convergence and divergence）

> Idea: convergence and divergence.

同じ初期条件で何度走らせても似た結果に収束するか（再現性・信頼性）／ わずかな条件変更で結果が分岐するか（感度・創発）を測る。

参考: **Chang, Chaszczewicz, Wang, Josifovska, Pierson, Leskovec (2024)** "LLMs generate structurally realistic social networks but overestimate political homophily" (arXiv:2408.16629) ← 補足論文 `06_1`

---

## 5. 関連する個人モデリングの系譜

- **GroupLens** (Resnick, Iacovou, Suchak, Bergstrom, Riedl, CSCW 1994) — 協調フィルタリング（個人の選好モデル）
- **Jury Learning** (Gordon, Lam, Park, Patel, Hancock, Hashimoto, Bernstein, CHI 2022) — 異議（dissenting voices）をMLモデルに統合

これらは「個人の選好・判断をどうモデルに反映するか」というPark氏の関心の系譜。

---

## 6. Bets（今日我々が賭けていること）

スライドタイトル: **"Bets that we place today"**

講義を通じてPark氏が立てる3つの仮説的な賭け:

1. **個人モデル化の賭け**: 個人にペルソナ記述を与えるアプローチは十分スケールするか
2. **集団モデル化の賭け**: 記憶共有による相互作用で創発的な集団行動が再現できるか
3. **評価の賭け**: 収束/発散の指標で信頼性を語れるか

---

## 7. まとめ

> The quantum unit of simulations — individual agents — is an important determinant of a simulation's success.
> Today, many generative AI-based simulations focus on populations.

3粒度のトレードオフ（授業スライドそのまま）:
- **Population**: **not granular enough**（粒度不足）
- **Individuals**: **too noisy**（ノイズ過多）
- **Groups**: **might never be predictable**（決して予測可能にならないかもしれない）

これらのトレードオフが、後続講義（05: アーキテクチャ、07: 信憑性 vs 正確性、08: 個人モデル、13: 倫理）で繰り返し俎上に上がる。

---

## 主要引用文献（Lecture 03）

### 個人モデルの系譜
- Resnick et al. (1994) GroupLens *CSCW '94*
- Gordon et al. (2022) Jury Learning *CHI '22*

### 集団・環境モデル
- Park et al. (UIST 2022) Social Simulacra
- Park et al. (UIST 2023) Generative Agents
- Schelling (1971) *J. Math. Sociol.* 1, 143-186

### 人口レベル生成AIシミュレーション
- **Argyle et al. (2023)** "Out of One, Many" *Political Analysis* 31, 337-355 ← 補足 `03_2`
- **Horton (2023)** "Homo silicus"
- **Ashokkumar et al. (2024)** "Predicting Social Science Experiments"

### 人口レベルの限界
- **Wang, Morgenstern, Dickerson (2024)** "LLMs cannot replace human participants" ← 補足 `13_1`
- **Cheng, Piccardi, Yang (2023)** *EMNLP 2023*

### 評価手法
- Chang, Chaszczewicz, Wang, Josifovska, Pierson, Leskovec (2024) arXiv:2408.16629 ← 補足 `06_1`

### その他
- von Neumann (1966); Wolfram (2002) — セルオートマトン
- Card, Moran, Newell (1983) *The Psychology of Human-Computer Interaction*

---

## 要点

1. **個人エージェントはシミュレーションの量子単位**。どう定義するかが成否を決める
2. 分析は **個人／集団／人口** の3粒度で、答えられる問いが異なる
3. 現状の生成AIシミュレーションは**人口レベルに偏在** — 評価容易性と個人モデル構築困難のため
4. 人口レベルは bias・stereotyping の罠があり（Wang 2024、Cheng 2023）、misportray/flattening を警戒すべき
5. Park氏の方針: **個人→ペルソナ記述、集団→記憶付与による相互作用、評価→収束/発散**
6. 3粒度のトレードオフ: 人口=粒度不足 / 個人=ノイズ過多 / 集団=予測不能かもしれない
