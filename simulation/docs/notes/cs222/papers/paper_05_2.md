# 05_2 — Cognitive Architectures for Language Agents (CoALA)

## 書誌情報

- 著者: **Theodore R. Sumers***, **Shunyu Yao***, Karthik Narasimhan, **Thomas L. Griffiths**
  - \* Equal contribution, order decided by coin flip
- 所属: Princeton University
- 掲載: **Transactions on Machine Learning Research (TMLR)**, February 2024
- OpenReview: https://openreview.net/forum?id=1i6ZCvflQJ
- arXiv: 2309.02427v3 (2024-03-15)
- GitHub: https://github.com/ysymyth/awesome-language-agents
- Lecture 05 の補足論文（05_1 Generative Agents と対）

---

## 1. 研究問題

> We lack a framework to organize existing agents and plan future developments.

### 背景

- LLM が急速に **language agents**（外部環境と対話する agent）の構築に使われ始めた
  - SayCan (Ahn 2022), ReAct (Yao 2022), Voyager (Wang 2023), Generative Agents (Park 2023), Tree of Thoughts (Yao 2023)
- 各研究が独自用語（'tool use', 'grounding', 'actions'）を使い、**比較・体系化が困難**
- 一方、AI と認知科学には **production systems** と **cognitive architectures** の長い歴史がある
- → 歴史的枠組みを現代の language agents に適用

### 提案: CoALA

> **Cognitive Architectures for Language Agents** organize agents along three key dimensions:
> 1. information storage (working + long-term memory)
> 2. action space (internal + external)
> 3. decision-making procedure (interactive loop with planning + execution)

---

## 2. 歴史的アナロジー

### 2.1 Production Systems（Post 1943 から Newell まで）

- **Post (1943)**: 論理系を string rewriting として表現 (XYZ → XWZ)
- **Markov algorithms**: 優先順位付き productions で任意の計算（Turing 完全）
- **Newell & Simon (1972)**: production system を agent の problem solving に拡張
  - 例: thermostat
    ```
    (temp > 70) ∧ (temp < 72) → stop
    (temp < 32) → call for repairs; turn on heater
    (temp < 70) ∧ (furnace off) → turn on furnace
    (temp > 72) ∧ (furnace on) → turn off furnace
    ```
- **Cognitive architectures** = production systems + 知覚+記憶+計画+意思決定
  - SOAR (Laird), ACT-R (Anderson), CAPS, EPIC など

### 2.2 SOAR の構造（CoALA の直接の祖先）

**Memory**:
- Working memory: 現在状況（perceptual input, goals, intermediate results）
- Long-term memory:
  - Procedural: production system（ルール）
  - Semantic: 世界についての事実
  - Episodic: 過去経験の系列

**Grounding**:
- 物理ロボット（perception → WM, motor commands out）
- 対話学習（Mohan, Kirk, Laird）

**Decision making**: propose → evaluate → select → execute (+ impasse → subgoal hierarchy)

**Learning**: chunking, RL, semantic/episodic writes, new productions

### 2.3 伝統的 cognitive architectures の衰退理由
- 論理述語で記述可能なドメインに限定
- 多数の pre-specified rules が必要

### 2.4 LLM がこれらの限界を解決
- 任意 text で動作（flexible）
- pre-training で productions の事前分布を学習（手動記述不要）

---

## 3. Language Models as Production Systems

### 3.1 アナロジー
- Production: X → XY または X → XYi（複数継続）
- LLM: X の prompt で P(Yi|X) を定義
- → **LLMs are probabilistic production systems**

### 3.2 Prompt Engineering as Control Flow

Prompting 手法を production 列として定式化 (Table 1):

| Method | Production Sequence |
|--------|---------------------|
| Zero-shot | Q → LLM → Q ∥ A |
| Few-shot | Q → Q1 ∥ A1 ∥ Q2 ∥ A2 ∥ Q → LLM → ... |
| RAG | Q → Wiki → Q ∥ O → LLM → Q ∥ O ∥ A |
| Socratic Models | Q → VLM → Q ∥ O → LLM → Q ∥ O ∥ A |
| Self-Critique | Q → LLM → Q ∥ A → LLM → Q ∥ A ∥ C → LLM → ... |

### 3.3 Cognitive Language Agents

- Early agents: LLM で directly actions を選択（SayCan）
- Mid-stage: reasoning を挟む（ReAct）
- Recent: episodic memory、program code、reflections（Generative Agents, Voyager）

→ **これら全てを CoALA で体系化**

---

## 4. CoALA 詳細（Section 4）

### 4.1 Memory

#### Working Memory
- 現在 decision cycle の active symbolic variables
- perceptual input, active knowledge, goals
- **LLM の context window より一般的**（persist across LLM calls）
- 各 LLM call で WM から subset が prompt 化、output が parse されて WM へ

#### Episodic Memory
- 過去 decision cycles の経験
- 例: training input-output pairs (Rubin 2021), event flows (Weston 2014, Park 2023), game trajectories (Yao 2020, Tuyls 2022)

#### Semantic Memory
- 世界と自分についての知識
- 従来: Wikipedia のような unstructured text を読み取る retrieval-augmented methods (Lewis 2020)
- 新機能: **LLM の推論で自己生成した知識も書き込める**（Reflexion の "there is no dishwasher in kitchen"）

#### Procedural Memory
- 2 種:
  1. **Implicit**: LLM weights（暗黙的手続き知識）
  2. **Explicit**: Agent の code（reasoning, retrieval, grounding, learning, decision-making procedures）
- 更新はリスク大（bug 導入、alignment 破壊）
- Designer が初期化必須

### 4.2 Action Space（Figure 5）

| 種類 | Read/Write | Target |
|------|-----------|--------|
| **Grounding** | External | 外部環境 |
| **Retrieval** | Read | Long-term memory → WM |
| **Reasoning** | R/W | WM ↔ WM（LLM 経由） |
| **Learning** | Write | LTM |

### 4.3 Grounding

3 種の外部環境:
1. **Physical**: ロボット（SayCan, RT-1, PaLM-E）
2. **Human/agent dialogue**: 指示受け取り、質問、複数エージェント社会シミュレーション（Park et al. 2023）
3. **Digital**: ゲーム、API、Web（ReAct, WebGPT, Voyager Minecraft）

### 4.4 Retrieval

- rule-based, sparse, dense retrieval
- 例:
  - Voyager: skill library の dense retrieval
  - Generative Agents: **recency (rule) + importance (reasoning) + relevance (embedding)** のハイブリッド
  - DocPrompting: documentation からの semantic retrieval

### 4.5 Reasoning

- WM の内容を処理して new information を生成
- Retrieval と違い WM の read/write
- 例: observation 要約 (ReAct)、trajectory 反省 (Reflexion)、retrieved memories への推論 (Park 2023)

### 4.6 Learning（5 種）

1. **Episodic への書き込み**: RL trajectories, Park 2023 の event flow
2. **Semantic への書き込み**: Reflexion の self-generated 知識、VLM で semantic map 構築
3. **LLM parameter 更新 (procedural)**: fine-tuning（supervised/RL/RLHF）
4. **Agent code 更新 (procedural)**:
   - Reasoning の更新（APE: prompt 指示を input-output から推論）
   - Grounding の更新（Voyager: new code skills をライブラリに追加）
   - Retrieval 更新（未開拓）
   - Learning/decision-making 更新（未開拓、リスク大）
5. Modification/deletion（unlearning）は未開拓

### 4.7 Decision Making

```
Decision Cycle:
  Planning stage:
    Proposal (reasoning + optional retrieval)
    Evaluation (heuristic / LLM values / learned values / LLM reasoning)
    Selection (argmax / softmax / majority vote)
  Execution:
    External action (grounding) or Internal action (learning/reasoning)
  Observe → loop
```

---

## 5. Case Studies（Table 2）

各エージェントを CoALA に cast:

### 5.1 SayCan (Ahn 2022)
- **LTM**: procedural only（LLM + learned value function）
- **Grounding**: physical（551 robotic skills）
- **Internal actions**: なし
- **Decision making**: evaluate（LLM value × affordance）
- → single-step planner

### 5.2 ReAct (Yao 2022b)
- **LTM**: なし
- **Grounding**: digital（Wikipedia API, games, websites）
- **Internal actions**: reasoning
- **Decision**: propose（reasoning → action）
- → 最小の internal+external action agent

### 5.3 Voyager (Wang 2023)
- **LTM**: procedural（hierarchical code skill library）
- **Grounding**: digital（Minecraft）
- **Internal actions**: reason, retrieve, learn
- **Decision**: propose（task + skill generation）
- 特徴: code-based skill library を RL で拡張、無 reward で curriculum 学習

### 5.4 Generative Agents (Park 2023) ← 05_1
- **LTM**: episodic + semantic（reflections）
- **Grounding**: digital + multi-agent dialogue
- **Internal actions**: reason, retrieve, learn
- **Decision**: propose（plan → execute → replan）
- 特徴: reflection tree, hierarchical plan

### 5.5 Tree of Thoughts (Yao 2023)
- **LTM**: なし
- **Grounding**: digital（最終解のみ提出）
- **Internal actions**: reason のみ
- **Decision**: **propose + evaluate + select**（BFS/DFS で global exploration）
- 特徴: deliberate decision-making

---

## 6. Actionable Insights（Section 6）

### 6.1 Modular Agents（単一 monolith からの脱却）

- 学術: 標準用語で論文比較可能に、OpenAI Gym 的な empirical 抽象化（Memory, Action, Agent class）
- 産業: 企業全体の "language agent library" で technical debt 削減、顧客体験の統一
- **Code vs LLM**: code は解釈可能・脆い、LLM は flexible・opaque → 相補的に使う（例: tree search で myopia を補う）

### 6.2 Agent Design: Beyond Simple Reasoning

設計ステップ:
1. どの memory modules が必要か（例: retail assistant → semantic (inventory) + episodic (user history) + procedural (queries) + working (dialogue state)）
2. Internal action space の定義（read/write 権限）
3. Decision-making procedure の定義（performance vs generalization トレードオフ）

### 6.3 Structured Reasoning: Beyond Prompt Engineering

- LangChain, LlamaIndex のような framework の利用
- Guidance, OpenAI function calling で output 構造化
- Training 時からも **agent 向け reasoning 形式**（CoT, ReAct, Reflexion）を増やすべき

### 6.4 Long-term Memory: Beyond Retrieval Augmentation

- 人間 knowledge（manuals, textbooks）+ self-generated experience/skills の組合せ
- Retrieval と reasoning の統合（Zhou 2023 human memory recall）

### 6.5 Learning: Beyond In-context/Fine-tuning

- Meta-learning（agent code 修正で学習方法自体を学習）
- Selective forgetting / unlearning
- Mixture of different learning types を agent 自身が選択

### 6.6 Decision Making: Beyond Action Selection

- より deliberate な decision（MCTS, Tree of Thoughts）
- 現実での risk assessment

---

## 7. Open Questions（Section 7）

1. **Alignment / Safety**: LLM base model の alignment が agent に継承。agent 特有の alignment 課題（code 自己改変）
2. **Calibration / Confidence**: agent が自身の不確実性を認識し、必要なら助けを求める
3. **Explainability**: 理由の合理化ではなく、真の decision 過程を示す
4. **Cognitive science connections**: agent を認知 model の仮説検証器として使える可能性
5. **AGI への道筋**: CoALA は一つの有望な道

---

## 8. CS 222 での位置づけ

### Lecture 05: 05_1 Generative Agents との対
- 05_1 は個別の case study、05_2 は全体の taxonomy
- **05_2 は 05_1 を最もよく使う case**（Section 5.4 で具体分析）
- Generative Agents は "all four action types" を持つ唯一の例として挙げられる

### Lecture 04 からの継承
- 04_1 Newell UTCs、04_2 SOAR の直接的後継
- SOAR の memory modules（working / procedural / semantic / episodic）をそのまま language agents に移植
- Newell の production system のアナロジーで LLM を再解釈

### Park 氏の議論との関係
- Generative Agents のアーキテクチャ分類は CoALA の枠組みで明確化される
- Park 氏自身が SOAR 的 cognitive architecture からの系譜を意識
- Lecture 09 以降（Agent Bank）でも CoALA 的 modular 設計が議論

### 他の CS 222 論文への接続
- 03_1 Social Simulacra → memory なし、single-shot LLM → CoALA では最小構成
- 03_2 Argyle → LLM を stateless simulator として使用
- 06_1 Chang (社会ネットワーク) → Generative Agents を CoALA 用語で議論
- 06_2 Roleplay-doh → principle-adherence pipeline は CoALA の reasoning/learning action の具体化

---

## 9. 主要引用

### 論文が引用
- Newell & Simon (1972) *Human Problem Solving*
- Laird, Newell, Rosenbloom (1987) SOAR
- Laird (2012, 2019, 2022) SOAR の最新版
- Kotseruba & Tsotsos (2020) cognitive architecture survey
- Brown et al. (2020) GPT-3
- Vaswani et al. (2017) Transformer
- Wei et al. (2022b) Chain-of-Thought
- **Park et al. (2023) Generative Agents** ← 05_1
- Ahn et al. (2022) SayCan
- Yao et al. (2022b) ReAct
- Yao et al. (2023) Tree of Thoughts
- Shinn et al. (2023) Reflexion
- Wang et al. (2023a) Voyager
- Anderson & Lebiere (2003) ACT-R

### 本論文を引用する後続
- CS 222 のアーキテクチャ議論
- Language agent 研究の標準参照

---

## 10. 要点

1. **CoALA = Cognitive Architectures for Language Agents**: 30年の認知アーキテクチャ研究を LLM agents に適用する統一枠組み
2. **核心的アナロジー**: Production systems (Post 1943, Newell 1972) と LLM は string manipulation system として同型
3. **3次元分類**: Memory (WM + episodic/semantic/procedural LTM) × Action (grounding/retrieval/reasoning/learning) × Decision-making (plan→propose→evaluate→select→execute)
4. **Memory 4分類**: Working (LLM context 超越), Episodic (experiences), Semantic (facts, self-generated too), Procedural (implicit LLM weights + explicit agent code)
5. **Action space**: Internal (retrieval/reasoning/learning) + External (grounding) が対称的
6. **Learning の spectrum**: episodic write → semantic write → LLM fine-tune → agent code modification (reasoning/grounding/retrieval/learning procedures)
7. **Case studies**: SayCan (grounding only) → ReAct (+ reasoning) → Voyager (+ procedural learning) → Generative Agents (all four) → ToT (deliberate decision)
8. **実用指針**: modular design, LLM と code を相補的に使う, structured reasoning (LangChain 等), long-term memory の self-generation
9. **SOAR の直接的後継**: 04_1/04_2 の三層 memory を言語ドメインに継承、productions を LLM で確率化
10. **CS 222 では**: 05_1 Generative Agents の体系的位置づけを提供する taxonomy、Park 氏のアーキテクチャ設計に CoALA の枠組みを与える
