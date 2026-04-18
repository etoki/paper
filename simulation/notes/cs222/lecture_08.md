# Lecture 08: Models of Individuals

- 講師: Joon Sung Park
- ソース: `docs/stanford univ AI Agents and Simulations/08 Models of Individuals.pdf`
- 位置づけ: 現状の生成AIシミュレーションは人口レベル中心だが、wicked problemsの多くは**個人レベル**のモデルを要する。どう作るか、なぜ難しいか、過去の系譜から何を学べるか。

---

## 0. 前回からの接続

- Believable vs accurate
- Wicked problems の多くは正確なシミュレーションを要する
- 現状は人口レベルの構築・評価が中心

本日の問い: **How might we build models of individuals, and why?**

---

## 1. 「モデル」とは何か

スライドで例示される広い意味での "model":
- ロボティクス / 自動運転
- コンテンツ・モデレーション
- スパムフィルタ
- 医療診断
- そして Ashokkumar, Horton, Argyle などの**人口レベル人間行動モデル**

観察:
> Today's models of human behavior are often created at the **population level**.

---

## 2. 個人モデルの定義と重要性

### 定義

> While models of a **population** predict the **average behavior** of a population, models of **individuals** predict the behavior of **a particular person**.

- 人口モデル: 集団の平均挙動を予測
- 個人モデル: 特定の人の挙動を予測

### なぜ重要か

> This opens up **genuinely new opportunities**.

個人モデルが実現すると、従来の集計統計では答えられない問いに答えられる（パーソナライズド予測、rehearsal、personal assistant 等）。

---

## 3. 2つの主要な難しさ

### Challenge 1: 個人の訓練データが希少（sparse）

> - Creating an effective model requires a large amount of data.
> - Today, we gather this data from the web (at the population level).
> - However, data on individuals, by definition, are much more scarce.

関連: Deng et al. (2009) ImageNet ← 大規模・人口レベルのデータセットの代表例。個人ではこのスケールが望めない。

### Challenge 2: 個人は一貫していない（not consistent）

> - Individual-level behavior measurements/observations can be riddled with inconsistencies due to the inherent instability of individuals and measurement errors.
> - **"Regression toward the mean" does not apply to models of individuals.**

個人の測定は不安定性＋測定誤差で揺れる。集団では「平均への回帰」が働くが、**個人ではこの緩和が効かない**。

引用:
- **Ansolabehere, Rodden, Snyder (2008)** "The Strength of Issues: Using Multiple Measures to Gauge Preference Stability, Ideological Constraint, and Issue Voting" *American Political Science Review* 102, 215-232 ← 補足 `08_1`
- **Lundberg et al. (2024)** "The origins of unpredictability in life outcome prediction tasks" *PNAS* 121, e2322973121 — 人生結果予測の不可予測性の起源

---

## 4. 過去の個人モデリング: 一般的スキーム

### General Scheme

> 1. **Create a central model that represents a population.**
> 2. **Quickly tune the components of that central model to describe individuals.**

= 人口を代表する中心モデルを作り、それを個人の情報で素早く調整する。

### 例1: 協調フィルタリング/推薦システム

**Resnick, Iacovou, Suchak, Bergstrom, Riedl (1994) "GroupLens: an open architecture for collaborative filtering of netnews" *CSCW '94*** ← 補足 `08_2`

- **仮定**: 2人のユーザーが1つの問題で合意すれば、他でも合意する可能性が高い
- **方法**: 似たユーザーを探して推薦（User A と User B の嗜好が似ていれば、B が好きで A が未接触の項目を A に推薦）
- **類似度計算**: Pearson相関、コサイン類似度、Jaccard類似度など
- **推薦**: ターゲットユーザーに類似ユーザーを特定し、彼らが好きな項目のうち未評価のものを提示

### 例2: Jury Learning (CHI 2022)

**Gordon, Lam, Park, Patel, Hancock, Hashimoto, Bernstein (2022) "Jury Learning: Integrating Dissenting Voices into Machine Learning Models" *CHI '22***

- データセット中の**各ラベラーを個別にモデル化**
- N回の試行で juror をサンプル → 指定された jury composition を形成
- 各 juror の判断を予測 → **median-of-means** で jury 出力を算出

意義: 「群衆の多数決」でなく、**異議を取り込んだモデル**。個人差を反映しつつ集約する。

### 例3: 人間心理モデル

**Big Five 性格検査**:
- John & Srivastava (1999) "The Big Five trait taxonomy" in *Handbook of Personality* (Guilford, 2nd ed.)
- 観察に基づいて性格を5次元で記述
- 「中心モデル＋個人差分」の原始的形態

---

## 5. 前進の道: LLM を中心モデルに

### Park氏の提案

> **Idea: Use an LLM as the central model. The LLM then roleplays as a specific person based on given information about that individual.**

- LLM = 人口を代表する中心モデル
- 個人情報を与えて**ロールプレイ**させる
- 協調フィルタリング的「中心モデル＋チューニング」を LLM 時代に再定式化

---

## 6. 核心的な問い

> **Q. What information would most effectively describe a person holistically?**

何の情報が、一人の人間を「全体的に（holistically）」記述するのに最も有効か？

### Class Activity

2グループに分かれて10分議論:

**Group 1**: 「初めて会った人に、30分で知るために何を聞くか？」

**Group 2**: 「自分を一人の人間として最も意味深く記述する事実は何か？（n=25）」

→ Park氏は授業でこれらの回答を実データとして集め、後続の AgentBank（Lecture 09）に繋げる。

---

## 主要引用文献（Lecture 08）

### 人口レベルの基準
- Ashokkumar et al. (2024); Horton (2023); Argyle et al. (2023)
- Deng et al. (2009) ImageNet *CVPR*

### 個人の非一貫性
- **Ansolabehere, Rodden, Snyder (2008)** *Am. Political Sci. Rev.* 102, 215-232（補足 08_1）
- **Lundberg et al. (2024)** *PNAS* 121, e2322973121

### 個人モデルの系譜
- **Resnick et al. (1994)** GroupLens *CSCW '94*（補足 08_2）
- **Gordon et al. (2022)** Jury Learning *CHI '22*
- **John & Srivastava (1999)** Big Five taxonomy

---

## 要点

1. 現状の生成AIモデルは**人口レベル中心**。個人モデルは wicked problems の多くで必須だが未開拓
2. 個人モデルの2大課題: (1) **訓練データの希少性** (2) **個人の非一貫性** — 平均への回帰が効かない
3. 過去の個人モデリングの系譜: **中心モデル＋個人チューニング** の枠組み
   - **協調フィルタリング** (GroupLens 1994)
   - **Jury Learning** (CHI 2022) — 異議を統合
   - **Big Five 性格理論**
4. **Park氏の提案**: LLMを中心モデルに据え、個人情報でロールプレイさせる
5. 核心的問い: **「一人の人間を全体的に記述するのに最も有効な情報は何か？」**
6. Class activity で学生から集めた回答が AgentBank-CS222 (Lecture 09) の土台になる
