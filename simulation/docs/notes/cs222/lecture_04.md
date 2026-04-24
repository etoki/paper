# Lecture 04: Cognitive Architectures

- 講師: Joon Sung Park
- ソース: `docs/stanford univ AI Agents and Simulations/04 Cognitive Architectures.pdf`
- 位置づけ: 認知アーキテクチャの歴史と概念を整理し、Lecture 05 で扱う Generative Agents の設計的起源を示す。

---

## 1. 「アーキテクチャ」とは何か

例示:
- **OS/PC ハードウェア**: Mathew (2023)
- **AI**: 強化学習（Sutton & Barto 2018）、Transformer（Vaswani et al. 2017 *NeurIPS*）
- **認知アーキテクチャ**: SOAR（Lehman et al. 2006）、GOMS（Card, Moran, Newell 1983）

### 講義の定義

> Architectures are both a **description of a functional system** and a **theory**.
> They are not a step-by-step recipe; rather, they offer a perspective on how a system should work.

アーキテクチャ = 機能記述 + 理論。レシピではなく「システムがどう動くべきか」の視点。

本日のテーマ: 認知アーキテクチャと生成エージェントのアーキテクチャ。

---

## 2. 認知科学の略史（Newell を軸に）

### Allen Newell の経歴
- Stanford で物理学学士 (1949)
- Princeton で数学院生 (1949-50)、当時「未知分野」のゲーム理論に触れる
- Carnegie Mellon Tepper School で Herbert Simon の下で PhD
- 参考講演: Newell "Desires and Diversions" (CMU, 1991)

### 歴史的マイルストーン

| 年 | 成果 | 注記 |
|---|------|------|
| **1955** | **Logic Theorist**（Newell, Simon, Shaw） | 史上初のAIプログラムとされる。Whitehead & Russell *Principia Mathematica* 第2章の52定理中38個を証明、いくつかはより短い証明を発見。Dartmouth Workshop で発表 |
| **1956** | **General Problem Solver (GPS)** | 整形式論理式/Horn節で表現でき、源点と終点（所望の結論）を持つ有向グラフとして書ける問題を原理的に解く |
| **1990** | **Unified Theories of Cognition**（Newell 単著） | 認知のすべての側面を説明する「統一認知理論」＝**cognitive architecture** の必要性を主張 |

引用:
- Newell, Simon, Shaw (1956) "The Logic Theory Machine" *IRE Trans. Inf. Theory* 2, 61-79
- Newell, Shaw, Simon (1959) "Report on a general problem-solving program" *Proc. Int'l Conf. on Information Processing*
- Newell (1990) *Unified Theories of Cognition*

### 認知心理学の初期観察

> Scholars in cognitive psychology began to propose that computers process information **similarly to the human mind**.

2つの問い:
1. **人の心の仕組みを、認知アーキテクチャで構築して理解できるか？**
2. **人間のタスクを解く汎用的な計算エージェントを作れるか？**

---

## 3. 歴史は繰り返す — この問いと生成AIの平行

Park氏の中心的論点（スライドで2度強調される）:

| **過去**：古典認知アーキテクチャ | **現在**：生成エージェント |
|--------------------------------|----------------------------|
| 心理学・AI学者が「コンピュータは人間の心と類似の情報処理をする」と提案 | HCI・AI学者が「生成AIは人間的な行動を符号化・生成する」と提案 |
| Q1: 認知アーキテクチャで人の心を理解できるか？ | Q1': 生成エージェントで**創発的集団行動**を理解できるか？ |
| Q2: 汎用エージェントは作れるか？ | Q2': 汎用エージェントは作れるか？（ほぼ同じ） |

> **History often repeats itself** and provides us with a useful guide as we navigate and build a new field.

過去の野心的目標は、ときに時期尚早でも将来の指針となる。

---

## 4. 認知アーキテクチャの定義（引用）

> A cognitive architecture as a theory of the **fixed mechanisms and structures** that underlie human cognition. Factoring out what is common across cognitive behaviors, across the phenomena explained by microtheories, seems to us to be a significant step toward producing a unified theory of cognition...

= 人間の認知の下にある固定された機構と構造の理論。

---

## 5. SOAR（1983-）

**起源**: John Laird の博士論文（Allen Newell, Paul Rosenbloom と共同）

### Problem Space Hypothesis（問題空間仮説）

> All goal-oriented behavior can be cast as **search through a space of possible states (a problem space)** while attempting to achieve a goal. At each step, a single operator is selected, and then applied to the agent's current state, which can lead to internal changes, such as retrieval of knowledge from long-term memory or modifications or external actions in the world.

- すべての目的志向的行動は「問題空間の探索」として書ける
- 各ステップで単一オペレータを選択・適用
- 内部変化（長期記憶からの知識検索）か外部行動かに帰着

**アーキテクチャの構造**: SOARの構造はコンピュータアーキテクチャに類似（スライド図）。

**静的ビュー vs 動的ビュー**:
- Joe の人生の静的ビュー: 特定状況で起こり得る全ての行動
- 動的ビュー: Joeの行動が実際に辿る経路

引用: Laird, Newell, Rosenbloom (1987) "SOAR: An architecture for general intelligence" *Artificial Intelligence* 33, 1-64

### 認知アーキテクチャの性格

> In a way, cognitive architectures are a **stylized caricature** of human cognition.
> "Soar is one theory of what is common to the wide array of behaviors we think of as intelligent."

Soar だけが唯一ではなく、Anderson (1993 ACT-R), Kieras-Wood-Meyer (1997), Langley-Laird (2002) など複数理論が並存。

### 他の古典アーキテクチャ
- **ACT-R**: Anderson & Lebiere (1998) *The Atomic Components of Thought*
- **GOMS**: Card, Moran, Newell (1983) *The Psychology of Human-Computer Interaction*

---

## 6. ゲームと NPC: 認知アーキテクチャのテストベッド

> Games and game NPCs have often served as a testbed for cognitive architectures.

この文脈で、Lecture 05 で扱う Generative Agents (Smallville) が登場。

---

## 7. Generative Agents のアーキテクチャ（概要）

本講義では Lecture 05 への橋渡しとして要点のみ紹介。

**舞台**: Smallville — 25体の生成エージェントが住むゲーム世界
**ペルソナ例**:
> "Isabella Rodriguez is the owner of Hobbs Cafe who loves to make people feel welcome; [...] Isabella Rodriguez is planning on having a Valentine's Day party at Hobbs Cafe at 5pm."

**観測される挙動**:
- 日次行動の計画と実行（起床 6:05am → 開店 8:00am → ランチ提供 12:00pm → 仕入れ 6:25pm）
- 行動は環境に影響（コーヒー作成 → カップ洗浄、マシンON、椅子占有）
- 相互作用を記憶（Sam と Latoya の出会い → 翌日の再開時に前日の話題を持ち出す）
- ユーザーは Smallville と対話可能

### アーキテクチャ本体

基盤: LLM（プロンプト "[name] is a [description]" で人間行動を生成、Social Simulacra UIST 2022 から）

**Perception → Action のループ**:
> We remember and make sense of our experiences. Prompt-based agents alone cannot.

プロンプトだけのエージェントには記憶と意味生成がない → これを補うのが **Memory Stream** アーキテクチャ:

1. **Perceive**: 環境からの観察（例: Maria が Klaus と話している / 椅子が空いている / Giorgio がピアノを弾いている）
2. **Memory Stream** に自然言語の記録として蓄積（タイムスタンプ付き）
3. **Retrieve**: 問い（"What are you excited about, Isabella?"）に応じて関連記憶を取得
4. 出力として **[Plan] [Action] [Reflection]** の3種類を生成
   - [Plan] Let's decorate the cafe later this afternoon
   - [Action] Heading to the local grocery store to buy supplies
   - [Reflection] I enjoy organizing events and making people feel welcome

> To those in the cognitive architecture communities, these new architectures are **immediately recognized**.

= SOAR/ACT-R の系譜で見ると、Generative Agents はその自然な延長として映る。

---

## 8. Simulation agents vs Tool-based agents

講義の終盤で現代 AI エージェントを2分類:

| 種別 | 例 |
|------|----|
| **Simulation agents**（人間行動の再現） | Generative Agents |
| **Tool-based agents**（タスク実行） | AutoGPT, Devin, Rabbit |

両者で異なる「アーキテクチャの反復」が起きている。Park氏の問題意識は主に前者（人間行動のシミュレーション）だが、同じ認知アーキテクチャの概念枠組みが両者に再登場している。

---

## 主要引用文献（Lecture 04）

### 歴史的系譜
- **Newell, Simon, Shaw (1956)** "Logic Theory Machine" *IRE Trans. Inf. Theory* 2, 61-79
- **Newell, Shaw, Simon (1959)** "General Problem Solver" *Proc. Int'l Conf. on Information Processing*
- **Newell (1990)** *Unified Theories of Cognition* (Harvard University Press)
- Newell (1991) "Desires and Diversions" (CMU 講演)

### SOAR と古典認知アーキテクチャ
- **Laird, Newell, Rosenbloom (1987)** "SOAR: An architecture for general intelligence" *Artificial Intelligence* 33, 1-64
- **Lehman et al. (2006)** "A Gentle Introduction to Soar: 2006 Update" ← 補足 `04_2`
- **Anderson & Lebiere (1998)** *The Atomic Components of Thought* (ACT-R)
- **Card, Moran, Newell (1983)** *The Psychology of Human-Computer Interaction* (GOMS)

### アーキテクチャの他領域例
- Sutton & Barto (2018) *Reinforcement Learning* 2nd ed.
- Vaswani et al. (2017) "Attention is all you need" *NeurIPS*
- Mathew (2023) "Understanding OS Architecture"

### Generative Agents への接続
- **Park et al. (UIST 2023)** "Generative Agents" ← 補足 `05_1`
- **Park et al. (UIST 2022)** "Social Simulacra"

### 補足論文
- `04_1` Newell "Précis of Unified Theories of Cognition" — *BBS* ダイジェスト
- `04_2` Lehman et al. "A Gentle Introduction to Soar, 2006 Update"

---

## 要点

1. **アーキテクチャ = 機能記述 + 理論**。レシピではなく視点を提供
2. Newell, Simon, Shaw の **Logic Theorist (1955)** → **GPS (1956)** → **Unified Theories of Cognition (1990)** が認知アーキテクチャの系譜
3. 古典認知アーキテクチャ（SOAR, ACT-R, GOMS）は **"stylized caricature"** として人間認知を理論化
4. **SOAR の Problem Space Hypothesis**: すべての目的志向行動は問題空間の探索に帰着
5. **歴史の反復**: 認知心理学×AI = 認知アーキテクチャ / HCI×生成AI = 生成エージェント。同じ2問（理解できるか？ 汎用エージェントを作れるか？）が再来
6. **Generative Agents は認知アーキテクチャコミュニティに"即座に認識される"**（Perception-Memory-Retrieve-Plan/Action/Reflection の骨格）
7. 現代 AI エージェントは **Simulation agents** と **Tool-based agents** に二分され、両者で認知アーキテクチャの反復が起きている
