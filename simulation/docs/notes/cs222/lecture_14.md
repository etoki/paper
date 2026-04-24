# Lecture 14: Simulating Ourselves and Our Societies With AI

- 講師: Joon Sung Park
- ソース: `docs/stanford univ AI Agents and Simulations/14 Simulating Ourselves and Our Societies With AI.pdf`
- 位置づけ: **クォーター全体のまとめ** + 1年〜32年の未来予想。Park氏の「pre-registration」。

---

## 0. 次週のゲスト講師

- **Monday: Meredith Ringel Morris**（Google DeepMind, Director and Principal Scientist, Human-AI Interaction）
- **Wednesday: Serina Chang**（UC Berkeley EECS, Assistant Professor）

### Class activity（10分）
**Agent voting** — これまでの学びを集大成するデモ投票。

---

## 1. クォーターのまとめ（9点）

### 1. シミュレーションの定義

> Simulations are programs that **define an environment and the behaviors of individuals**, then output the resulting world.

- 例: The Sims（ゲーム）、The Matrix（映画）、ABM（Schelling 1971）

### 2. 生成AIが開く新しい機会

> Generative AI presents a new opportunity to create more **open-ended simulations** of human behaviors.

Park et al. (UIST 2023) Generative Agents。

### 3. Wicked problems への約束

> The promise of human behavioral simulation is to enable us to address **wicked problems**.

Rittel & Webber (1973): 複雑で定義困難な社会・政策課題。

### 4. 分析レベルの選定

> To build simulations, you start by **understanding the level of analysis** you want to conduct.

Individuals / Groups / Populations の3粒度（Lecture 03）。

### 5. エージェントのアーキテクチャ構築

Lecture 04-05 の Generative Agents: Memory / Retrieval / Reflection / Plan & Action。

### 6. 環境の設計

- Park et al. (UIST 2023) Smallville
- Qian et al. (ACL 2024) ChatDev
- Jansen et al. (2024) DiscoveryWorld
- Li et al. (2024) Agent Hospital

### 7. 現状の評価軸

> So far, we have evaluated the success of simulations by testing their **believability** and their **ability to predict known phenomena**.

- Bates (1994) Believable Agents
- Ashokkumar et al. (2024) 社会科学実験予測

### 8. 未見世界のシミュレーションを信頼するための科学的基盤

> Going forward, we ought to establish a **scientific foundation** for simulations that will allow us to **trust simulations of unseen worlds**. Agent banks might serve this purpose.

**Agent Bank** が未知世界シミュレーション信頼の基盤になる可能性。

### 9. 新しい社会科学的問い

ファントム渋滞、市場暴落、消費者行動、社会運動、都市成長、バイラルコンテンツ…
これら**今日では対処困難な問い**に取り組めるようになる。

---

## 2. 未来予想（1年 〜 32年）

Park氏の **pre-registration**（スライドはGitHubに残す）:

> So... where is the field headed? Figuring that out is a **wicked problem in itself**, but let me speculate.

### Year 1: Scientific Foundation and Models of Individuals

- 現在の焦点: シミュレーションの「科学的基盤」確立
- 問い:
  - シミュレーションの正しい building blocks は？
  - 堅牢なシミュレーションをどう作るか？ 欠陥をどう判定するか？
- 現在は何が正しい building blocks かを巡って**異なる賭け（bets）**が進行中

### Year 2: Models of Interactions

- 次の2年で、エージェント間**相互作用**の構築・評価に本格進出
- 複数エージェントの GABM を作る前提となる building blocks

### Year 4: Merging of Tool-Based Agents and Simulation Agents

現在の「エージェント」コミュニティ内の分化:
- **Tool-based agents**: タスク自動化（AutoGPT, Devin, Rabbit）
- **Simulation agents**: 相互作用のシミュレート・予測

Park氏の予想:
> The core ingredient for advancing **tool-based agents** (and realizing Mark Weiser's vision) is **simulations**.

- 4年後に両アプローチが成熟し、**本格的収束**が起こる
- これが中期に新しい応用の波を開く

### Year 8: Societal Simulations（社会規模シミュレーション）

> The field of simulation is making a significant promise: creating large, multi-agent simulations of **societies** to address wicked problems.

- 現在は手が届かない（モデルは弱く、世界を表象する包括モデルがない）
- 8年目までに **準大規模（100万規模）**の社会シミュレーションが可能に

重要な予想:
> If this field were to win a Nobel Prize, the prize-winning (or catalyzing) work, **akin to Schelling's, would likely emerge around this time**.

（Schellingは2005年ノーベル経済学賞。）

### Year 16: Simulation as a New Computing Platform

- 16年後、AIもシミュレーションも「新規」ではなく**生活の事実**に
- 基盤技術も成熟: 少数の**超大規模中央モデル** + 多数の**小さく高性能なモデル**

核心予想（Lecture 11 の再掲）:

> Where a large central model will function like a **CPU**, **simulations will play the role of a GPU**.

- 単一大モデルに依存するより、多数の小モデルを使う**マルチエージェントシミュレーション**が独自に強力
- 多様な視点を要する科学的問題や wicked problems に特に有効

### Year 32: Multiverse

> I hope that simulation will be viewed as the **killer application of AI**.

さらに先:
> What is initially a 'killer application' of a platform often becomes **a platform itself**.

- シミュレーション上に構築されたアプリケーションが、**無数の多元宇宙（multiverses）** を作り
- 我々の未来をナビゲートする手助けをする

---

## 3. 講義の締め

### Q: 将来の本コースで扱ってほしいトピックは？

pollev.com/helenav330 で学生から収集。

### スライドの位置づけ

> Let this serve as my **pre-registration** — the slides are posted to Github :)

Park氏自身の予想を pre-registered hypothesis として将来検証できるよう公開。

---

## 主要引用文献（Lecture 14）

- **Schelling (1971)** *J. Math. Sociol.* 1, 143-186
- **Park et al. (UIST 2023)** Generative Agents
- **Rittel & Webber (1973)** *Policy Sciences* 4, 155-169
- Qian et al. (ACL 2024) ChatDev
- Jansen et al. (2024) DiscoveryWorld
- Li et al. (2024) Agent Hospital
- **Bates (1994)** *CACM* 37, 122-125
- **Ashokkumar et al. (2024)** "Predicting Social Science Experiments"

---

## 要点

1. **クォーターの9メッセージ**: 環境+個体挙動の出力装置 → 生成AIが open-ended 化 → wicked problems 対処 → 粒度選定 → アーキテクチャ → 環境 → 信憑性+既知予測で評価 → **Agent Bank**で未見世界への信頼基盤を作る → 新しい社会科学的問い
2. **32年ロードマップ（Park氏 pre-registration）**:
   - **Year 1**: 科学的基盤と個人モデル
   - **Year 2**: 相互作用モデル
   - **Year 4**: Tool-based agents と Simulation agents の収束
   - **Year 8**: 100万規模社会シミュレーション、**Schelling級ノーベル賞級の成果がここで**
   - **Year 16**: **LLM=CPU / シミュレーション=GPU** が計算プラットフォームとして確立
   - **Year 32**: シミュレーションが **AIのキラーアプリ**、**Multiverse** がプラットフォーム化
3. 大目標: シミュレーションで無数の多元宇宙を作り、未来の航行を助ける道具にする
