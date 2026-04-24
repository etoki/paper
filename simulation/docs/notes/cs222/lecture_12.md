# Lecture 12: Language and Schema of Simulations

- 講師: Joon Sung Park
- ソース: `docs/stanford univ AI Agents and Simulations/12 Language and Schema of Simulations.pdf`
- 位置づけ: シミュレーションを構築するためのツール・言語・スキーマを何に設計すべきか。表象（representation）が解法を規定するという Simon/Norman 系譜から、Assignment 1 repo の設計哲学へ。

---

## 0. 連絡事項

- 最終プロジェクト**提案プレゼン**: 6分発表+2分Q&A（テンプレート公開予定）
- カバー項目: **動機、手法、仮説**

### 本日の問い

> How can we create tools to build simulations?

---

## 1. 表象（Representation）の重要性

H/T: Michael S. Bernstein, CS347 (Visualization) の講義から引用。

### Cognitive Amplification（認知増幅）

> Visualization can help, but ultimately this power comes from **better representation**.
> By better understanding human cognition, we can design technology that makes us smarter.

### Simon/Norman の引用

> The powers of cognition come from **abstraction and representation**: the ability to represent perceptions, experiences, and thoughts in some medium other than that in which they have occurred, abstracted away from irrelevant details.

出典:
- **Simon, H. A. (1981)** *The Sciences of the Artificial* 2nd ed., MIT Press, p. 153
- **Norman, D. A. (1994)** *Things That Make Us Smart: Defending Human Attributes in the Age of the Machine*, Perseus Books, p. 47

---

## 2. 表象が問題を易しくする例

### Number Scrabble（Simon 1988）

ルール:
- 1-9の数字を順に取り合う（重複なし）
- 自分の手のうち**3つの和が15**になったら勝ち
- 余分な数字があってもOK、3つ組で15を作ればよい

出典: **Simon, H. A. (1988) "The Science of Design: Creating the Artificial" *Design Issues* 4(1/2), 67-82**

### ゲームの実演

> A takes 4, B takes 9, A takes 2, B takes 8, A takes 5. What should B do?

直感的にすぐ答えが分からない。

### 再符号化（魔方陣でTic-Tac-Toe化）

```
A4  B9  A2
3   A5  7
B8  1   6
```

3x3の魔方陣（各行・列・対角線の和=15）に数字を配置 → Number Scrabble は**Tic-Tac-Toeと同型**。

> Representation: Changing representation to spatial tic-tac-toe board **facilitates choice**.

**本質的洞察**: 同じ問題でも、**表象を変えると解きやすさが劇的に変わる**。

---

## 3. 歴史的な視覚表象の革命

### Playfair の貿易グラフ（1786-1801）

**Playfair, W. (1786)** *The Commercial and Political Atlas*（統計グラフの草創）

- ユーザーのタスク: 英国と北米の貿易収支の歴史的推移を理解
- 表象: **輸出/輸入の折線を重ね、差の領域を陰影で塗る** → 貿易不均衡が一目瞭然

### Data Types（表象の基礎語彙）

| 種別 | 例 | 比較操作 |
|------|----|---------| 
| N (Nominal) | 果物: りんご、オレンジ | = |
| O (Ordered) | 卵のグレード: AA, A, B | =, <, > |
| Q (Interval) | 日付、座標（緯度経度）| =, <, >, - |
| Q (Ratio) | 長さ、質量、カウント | =, <, >, -, ÷ |

---

## 4. 言語とスキーマは「視点」を提供する

### 言語の種類

- **Programming languages**: 出力を生成する命令の集合
- **Domain-Specific Languages (DSLs)**: 特定ドメインに特化した構文・関数

### スキーマ

> Schemas define a structured layout or format for data, describing relationships between data types and fields.

### 核心的主張

> Language and schema offer a perspective on the **future we are heading toward**.
>
> The clearer and more prescient this perspective is, the more powerful the language and schema become.

良い言語・スキーマは、向かう先の未来を明確に示す。

---

## 5. 既存言語・スキーマが示す「視点」の例

### SQL

**Chamberlin & Boyce (1974) "SEQUEL: A Structured English Query Language"**

SQLが体現する視点: 「非専門家が動的にデータを操作できるべき」
- 宣言型言語（何を取得するかだけ書く、どう取得するかは処理系任せ）
- **CRUD操作**（Create, Read, Update, Delete）で柔軟に管理
- データライフサイクル全体を単一言語で

### HTML5

**Hickson et al. (2014)** W3C Recommendation

HTML5が体現する視点: 「マルチメディアと様々な画面サイズを支持すべき」
- ネイティブ `<audio>`/`<video>` 要素（Flashプラグイン不要）
- レスポンシブデザインに対応

### 強化学習（RL）の図式

**Sutton & Barto (1998)** *Reinforcement Learning: An Introduction*

RLが体現する視点: 「エージェントは環境と行動の相互作用として記述すべき」
- agent vs environment の分離を明示する図
- 矢印で因果・フィードバックを示す

---

## 6. Discussion — 講読論文の視点

### D3: Data-Driven Documents

**Bostock, Ogievetsky, Heer (2011) "D3: Data-Driven Documents" *IEEE TVCG* 17(12), 2301-2309** ← 補足 `12_1`

データとDOM要素を直接バインドする視点。

### Generative Agents（再訪）

Park et al. UIST 2023 の視点: **agent, population, environment** の3軸で記述。

---

## 7. Assignment 1 Repo の視点

https://github.com/joonspk-research/gabm-stanford-main

### 問い

> How can we provide representations for creating **robust and replicable** simulations?

### 設計哲学

1. **エージェントの記憶とアーキテクチャを「凍結」する**
   - 再現性のため、エージェント状態をスナップショット可能に
2. **環境の実装を標準化する**
   - 誰が書いても同じ結果が得られるよう

### 構造

- **Agent Bank → Generative Agent**（個人エージェント群）
- Generative Agent → Responses: **Categorical / Numerical / Freeform**（多様な出力形式）
- Environment: **Survey / Interview / Network**（環境タイプの分類）

---

## 主要引用文献（Lecture 12）

### 表象と認知の系譜
- **Simon (1981)** *The Sciences of the Artificial* 2nd ed.
- **Norman (1994)** *Things That Make Us Smart*
- **Simon (1988)** "Science of Design" *Design Issues* 4(1/2)
- Playfair (1786) *Commercial and Political Atlas*

### 言語・スキーマの例
- Chamberlin & Boyce (1974) SEQUEL/SQL
- Hickson et al. (2014) HTML5 W3C
- Sutton & Barto (1998) RL
- **Bostock, Ogievetsky, Heer (2011)** D3 *IEEE TVCG* 17(12)（補足 12_1）

### シミュレーション
- Park et al. (UIST 2023) Generative Agents

---

## 要点

1. **表象（representation）は問題の解きやすさを決める**（Simon/Norman）
2. **Number Scrabble**: 3数和=15 の問題を Tic-Tac-Toe魔方陣に再符号化すると瞬時に解ける
3. **言語とスキーマは、向かう未来への視点**を提供する
4. SQL（非専門家の動的操作）、HTML5（マルチメディア+レスポンシブ）、RL図式（agent-環境分離）など、各々が明確な視点を体現
5. Assignment 1 repo の視点: **記憶・アーキテクチャを凍結**、**環境実装を標準化**で、再現可能で堅牢なシミュレーション基盤を作る
6. 補足 `12_1` D3 はデータと表象を直接バインドする強力な視点の例
