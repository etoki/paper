# Lecture 11: Equilibria and Butterflies

- 講師: Joon Sung Park
- ソース: `docs/stanford univ AI Agents and Simulations/11 Equilibria and Butterflies.pdf`
- 位置づけ: GABM をどう評価・活用するか、Park氏の**推論（conjectures）**を語る回。複雑系・カオス・均衡の観点から。

---

## 0. 連絡事項

- 最終プロジェクトのチーム名と概要を提出
- **AgentBank-CS222 を公開**（Canvas Announcements）

## 1. クォーターの振り返りと本講義の位置

これまでのカバー:
- 生成エージェントのアーキテクチャと実装
- 個人・人口レベルでの評価・活用
- **GABM の基礎** — 生成エージェント同士の相互作用

次の問い:
> But **how do you evaluate and leverage GABM**?

Park氏の宣言:
> I have taught you everything I (and the field) know. Today (and Wednesday), I will share my **conjectures** on where I think we are headed.

＝ここから先は Park氏の予想・推論。

---

## 2. 複雑系とバタフライ効果

### 複雑系の定義

> A system composed of many interconnected components that interact in dynamic and often nonlinear ways, producing **collective behaviors that are difficult to predict** from the behavior of individual parts.

### 身の回りの複雑系

**自然界**:
- 惑星軌道
- 鳥の群れ
- 心臓リズム
- 雲の形成
- 海洋波

**社会生活**:
- ファントム渋滞
- 市場暴落
- 消費者行動
- 社会運動
- 都市の成長
- バイラルコンテンツ

### ABM/GABM は複雑系

- Schellingの分離モデル、Smallville も複雑系の例

### Chaos（カオス）

> Chaos is prevalent in complex systems: **tiny variations in the initial conditions of a system can lead to vastly different outcomes**, to the point where the outcome seems random.

（Lorenzの*The Essence of Chaos*、補足 `11_1` と対応）

### Class Q
> Imagine Sam won the election in our simulation. **What might cause the outcome to differ in real life?**

GABMの予測力と限界を直感させる問い。

### 根本問い

> **Is our world inherently unpredictable? And if so, what can we learn from simulations?**

---

## 3. 均衡（Equilibria）

### 定義

> Equilibria are states in which a system remains balanced, with no net change in the absence of external disturbances, as opposing forces or influences are in a stable relationship.

### Nash 均衡

**Nash (1951) "Non-cooperative games" *Ann. Math.* 54, 286-295**
**Holt & Roth (2004) "The Nash equilibrium: A perspective" *PNAS*** ← 補足 `11_2`

> Each player in a game has chosen a strategy, and no one can benefit by changing their strategy while the others keep theirs unchanged.

### 具体例

**囚人のジレンマ** (Tucker 1950)
- 両者黙秘=最小刑、一方だけ自白=自白者釈放+他者重刑、両者自白=中刑
- Nash均衡: **両者自白**（個別最適だが協調最適ではない）

**ジャンケン**
- Nash均衡: 1/3ずつの**混合戦略**（予測可能なパターンは exploit される）

**Public Goods Game** (Marwell & Ames 1981)
- 各人が公共財への拠出を選ぶ
- Nash均衡: **誰も拠出しない**（free-rider 問題）

---

## 4. 大規模社会システムにおける均衡と創発

3ケーススタディ:

### Case 1: Social Norms

**Hawkins, Goodman, Goldstone (2019) "The emergence of social norms and conventions" *Trends Cogn. Sci.* 23, 158-169**

- 並ぶ、挨拶の仕方など
- 受け入れられたルール周辺での均衡

### Case 2: Political Polarization

**Fiorina & Abrams (2008) "Political polarization in the American public" *Annu. Rev. Polit. Sci.* 11, 563-588**

- 2つの主要な見解が支配する分極均衡
- 個別イシューでの揺れはあれど、全体構造は安定
- 対立する見解が互いを牽制

### Case 3: Scale-Free Networks

**Barabási & Albert (1999) "Emergence of scaling in random networks" *Science* 286, 509-512**

- 少数の「ハブ」が多数のリンクを持つ冪則分布
- ネットワークの自然な創発構造

---

## 5. 均衡の不安定性

> **In complex systems, there is no such thing as "happily ever after."**
>
> Nations rise, fall, and rise again.

### Economic Bubbles

**Kindleberger (1978) *Manias, Panics, and Crashes***
**Shiller (2015) *Irrational Exuberance*** (3rd ed.)

- 投機で価格が内在価値から乖離（不安定均衡）
- 小さな引き金（失望決算、金利変動）で崩壊
- 市場急落と経済危機

---

## 6. GABM から洞察を引き出す

### 3つのモード

**A. Anecdotal insights（逸話的洞察）**
- Social Simulacra (Park et al. UIST 2022) のようにスケール前にエッジケースを発見

**B. Equilibria and emergence（均衡と創発）**
- ネットワークシミュレーション、分離モデルなどで創発現象を予測

**C. Interventions（介入）**
> Can simulations provide a **step-by-step guideline** for interventions to shape the future?
- シミュレーションで介入の段階的な指針を得られるか

---

## 7. 結論: GPT-X は wicked problems を解くか？

### 中心仮説（Park氏の大きな予想）

> **Hypothesis: While LLMs will serve as cognitive CPUs, simulations will function as cognitive GPUs in the era of generative AI.**

- **LLM = CPU**（中央の複雑な推論器）
- **シミュレーション = GPU**（比較的単純な認知単位の並列相互作用）

### 根拠

> The grand challenges of our generation do not require **a complex central reasoning unit**.
> Rather, they require **relatively simple cognitive units that come together to form complex phenomena**.

我々の世代の大課題（wicked problems）は、中央に巨大推論器を置くのでなく、**多数の単純な認知単位の相互作用**で解く必要がある。

---

## 主要引用文献（Lecture 11）

### 複雑系・カオス
- （Lorenz *Essence of Chaos* は補足 11_1）
- Schelling (1971, 1978)
- Park et al. (UIST 2023)

### 均衡
- **Nash (1951)** *Ann. Math.* 54, 286-295
- **Holt & Roth (2004)** "Nash equilibrium: A perspective" *PNAS*（補足 11_2）
- Tucker (1950) 囚人のジレンマ
- Marwell & Ames (1981) Public Goods

### 創発と均衡
- **Hawkins, Goodman, Goldstone (2019)** *Trends Cogn. Sci.* 23
- **Fiorina & Abrams (2008)** *Annu. Rev. Polit. Sci.* 11
- **Barabási & Albert (1999)** *Science* 286

### 不安定均衡
- Kindleberger (1978) *Manias, Panics, and Crashes*
- Shiller (2015) *Irrational Exuberance*

### GABM 応用
- Park et al. (UIST 2022) Social Simulacra

---

## 要点

1. 本講義から先は Park氏の**推論（conjectures）**
2. 世界は**複雑系でカオス的**（バタフライ効果）だが、同時に**均衡や創発**を示す
3. Nash均衡の古典例（囚人のジレンマ、ジャンケン、公共財ゲーム）と社会的均衡の例（規範、分極、スケールフリー網）
4. 均衡は**しばしば不安定**（バブル崩壊のように小さな引き金で崩れる）
5. GABMからの洞察抽出モード: **逸話的洞察／均衡・創発／介入の段階的指針**
6. **大予想**: **LLM = 認知CPU、シミュレーション = 認知GPU**
7. Wicked problems は中央巨大推論器でなく、**単純な認知単位の相互作用**で解く
