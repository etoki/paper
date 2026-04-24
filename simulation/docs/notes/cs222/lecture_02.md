# Lecture 02: Wicked Problems

- 講師: Joon Sung Park
- ソース: `docs/stanford univ AI Agents and Simulations/02 Wicked Problems.pdf`
- 位置づけ: 「シミュレーションは何の役に立つか」の答えとして、シミュレーションが特に価値を発揮する問題クラス=wicked problemsを同定する。

---

## 1. 導入: 2つのヴィネット（想像のための物語）

講義は「我々が既に知っている失敗」を思い出させる2つのケーススタディから始まる。これらは後で"wicked problem"の7特性に当てはめられる。

### ヴィネット1: SNS

3段階の問題ドミノ:

| # | 問題 | 解決策 | 意図せざる結果 |
|---|------|--------|----------------|
| 1 | **Winner takes all**：社会ネットは勝者総取りで、急速な規模拡大が必要 | できるだけ早く成長する → 2006年 Facebook News Feed とアルゴリズムで注意喚起 | 反社会的コンテンツが拡散 |
| 2 | **Anti-social content spreads**：センセーショナル/感情的/誤解を招く投稿がアルゴリズム上優遇 | コンテンツ・モデレーション（ファクトチェック、Politifact等） | 「何が真実か」の対立が深刻化 |
| 3 | **Claims of bias or censorship**：モデレーション自体が偏向・検閲だと批判 | — | — |

引用:
- Lazer et al. (2018) "The science of fake news" *Science* 359(6380)
- Bail et al. (2018) "Exposure to opposing views on social media can increase political polarization" *PNAS* 115(37)
- Bergengruen & Perrigo (2021) "Facebook acted too late to tackle misinformation on 2020 election"

### ヴィネット2: パンデミック対応

| # | 問題 | 解決策 | 意図せざる結果 |
|---|------|--------|----------------|
| 1 | **Spread of illness**：感染拡大 | ロックダウン | 経済への統計的に有意な負の影響、感染抑制効果は限定的 |
| 2 | **Negative economic impact** | ロックダウン解除＋ソーシャル・ディスタンシング | 政策の揺れと経済悪化で公的信頼が低下 |
| 3 | **Drop in public trust** | — | 初期のワクチン接種率に悪影響 |

引用:
- Yamaka, Lomwanawong, Magel, Maneejuk (2022) *PLoS One* 17, e0268184
- Privor-Dumm et al. (2023) *BMJ Global Health* 8, e011881
- Ruiz et al. (2023) *JMIR Human Factors* 10, e39697

共通構造: **解決策自体が新しい問題を生む再帰的ドミノ**。

他の例: work-life balance / 移民政策 / サステナビリティ / 麻薬戦争 / 国民皆保険。

---

## 2. 中心概念: Wicked Problems

> Wicked problems are complex, ill-defined social or policy challenges that defy straightforward solutions.

出典: **Horst W. J. Rittel & Melvin M. Webber (1973) "Dilemmas in a general theory of planning" *Policy Sciences* 4, 155-169**

- Rittel: UC Berkeley デザイン理論家
- Webber: UC Berkeley 都市デザイナー・理論家

「計画理論のジレンマ」論文が本講義のアンカー論文（`02_1` として配布）。

### Wicked Problemsの7特性（Rittel & Webber 1973）

1. **確定的定式化が存在しない**（問題の定義自体が解法と同じくらい重要）
2. **停止規則がない**（どこで終わるかの基準が無い）
3. **真偽ではなく、より良い／より悪い**で評価される
4. **即時・最終テストが存在しない**
5. **すべての解法は "one-shot operation"**（試行錯誤の機会がなく、毎回の試みが重大）
6. **列挙可能な解の集合も、許容される操作の集合も記述できない**
7. **すべての wicked problem は本質的に唯一無二**

講義ではこの7特性を SNS とパンデミック対応の各ヴィネットに当てはめる演習を行う。

### 追加の洞察: **Wicked problem は再帰的（recursive）**

> 8. Every wicked problem can be considered to be a symptom of another problem.

問題を "より高いレベル" で定式化すると、一般性は増すが対処の難易度も上がる。しかし低レベル（症状対処）で叩くと事態は悪化しうる（incrementalismの危険）。

→ 本講義の核心的問い:

> **Can simulations help us identify the most general versions of our problems?**
> （シミュレーションは、我々の問題の最も一般化された版を特定する助けになるか？）

---

## 3. Science と Design の区別

Rittel & Webberの中心的主張:

> The kinds of problems that planners deal with—societal problems—are inherently different from the problems that scientists and perhaps some classes of engineers deal with.

- **Tame/Benign problems** (科学・工学): 定式化可能、分離可能、解が見つかる（例: 方程式、有機化合物の構造解析、5手詰めチェス）
- **Wicked problems** (設計・都市計画・政策): 曖昧、政治的判断に依存、「解かれる」のでなく「再解され続ける」

> Social problems are never solved. At best they are only re-solved—over and over again.

---

## 4. 系譜: Wicked Problems → Design → AI / HCI

Rittel & WebberのデザインサイエンスはAI/HCIの開祖に直結:

- **Herbert A. Simon** — *The Science of Design: Creating the Artificial* (MIT Press, 1996)
- **Allen Newell** — *Unified Theories of Cognition* (Harvard University Press, 1990)
- **Card, Moran, Newell** (1983) *The Psychology of Human-Computer Interaction* ← HCIの基礎
- **Jonathan Grudin** (1994) "Groupware and social dynamics: eight challenges for developers" *Communications of the ACM* 37, 92-105 ← 設計と工学は "soft-edged problems" を扱う

この講義の位置づけ: Simon/Newellの問題意識が、生成AIエージェント時代のシミュレーションに再来している。

---

## 5. 理想化された計画システム（Rittel & Webber 1973）

Rittel & Webberの理想像（引用）:

> Many now have an image of how an idealized planning system would function. It is being seen as an on-going, cybernetic process of governance, incorporating systematic procedures for continuously searching out goals, identifying problems, forecasting uncontrollable contextual changes, inventing alternative strategies, tactics, and time-sequenced actions, stimulating alternative and plausible action sets and their consequences, evaluating alternatively forecasted outcomes, statistically monitoring those conditions of the publics and of systems that are judged to be germane, **feeding back information to the simulation and decision channels so that errors can be corrected**—all in a simultaneously functioning governing process.

しかし Rittel自身が「そんな計画システムは到達不能（unattainable）」と認める。

**Park氏の論点**: 生成AIシミュレーションは、この"到達不能な"サイバネティック計画システムに再挑戦する技術基盤になり得る。

---

## 6. Schelling のモデル（ブリッジ）

講義の最後で Thomas Schelling を紹介し、Lecture 03 以降へ橋渡し。

- **Thomas Schelling** — University of Maryland 経済学者
- **Micromotives and Macrobehavior** (W.W. Norton, 1978) ← 補足論文 `02_2`
- **Dynamic models of segregation** (*J. Math. Sociol.* 1971)

主要発見:
- 個人の微小な選好（「少なくとも同じ種類の人が近所にいてほしい」）から、マクロな分離（segregation）が創発する
- **Tipping point**: 15%閾値・30%閾値・75%閾値で、結果のランドスケープが劇的に変わる

Schellingモデルの意義: 単純なルールから予測不能な集合現象が生まれる=シミュレーションが wicked problems を攻略できる可能性の原型。

---

## 7. 講義の締め: 世代的課題

> What would it take for the field of simulation to earn a Nobel Prize?

Schellingは2005年にノーベル経済学賞を受賞。次世代のシミュレーション研究が同等の社会的インパクトを残すには何が必要か、という挑発的問いで締める。

---

## 主要引用文献（Lecture 02）

- **Rittel & Webber (1973)** "Dilemmas in a general theory of planning" *Policy Sciences* 4, 155-169
- Simon (1996) *The Science of Design: Creating the Artificial*
- Newell (1990) *Unified Theories of Cognition*
- Card, Moran, Newell (1983) *The Psychology of Human-Computer Interaction*
- Grudin (1994) "Groupware and social dynamics: eight challenges for developers" *CACM* 37
- **Schelling (1971)** "Dynamic models of segregation" *J. Math. Sociol.* 1, 143-186
- Schelling (1978) *Micromotives and Macrobehavior*
- Lazer et al. (2018) *Science* 359(6380)
- Bail et al. (2018) *PNAS* 115(37)
- Yamaka et al. (2022) *PLoS One*
- Privor-Dumm et al. (2023) *BMJ Global Health*
- Ruiz et al. (2023) *JMIR Human Factors*

---

## 要点

1. **Wicked problems** = 定式化不能・停止規則なし・真偽でなく善悪・一発勝負・唯一無二・再帰的な社会問題
2. 科学が扱う tame problems とは根本的に異なる → **設計**の問題
3. 解決策自体が次の問題を生むドミノ構造を、SNS とパンデミック対応で実演
4. **Rittel の理想像**（サイバネティック計画システム）は到達不能とされたが、生成AIシミュレーションで再挑戦できるか
5. Schellingの分離モデルは、単純ルールから創発する現象の原型であり、シミュレーションの可能性を示す先駆例
