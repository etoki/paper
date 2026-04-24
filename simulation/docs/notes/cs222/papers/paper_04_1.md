# 04_1 — Précis of Unified Theories of Cognition

## 書誌情報

- 著者: **Allen Newell**（1927–1992）
- 所属: School of Computer Science, Carnegie Mellon University
- 掲載誌: **Behavioral and Brain Sciences** 15(3): 425–492 (1992)
- 元書: *Unified Theories of Cognition*, Harvard University Press (1990)
  - ハーバード大学での **William James Lectures**（1987年春）の書籍化
- Lecture 04 のアンカー論文（認知アーキテクチャ）

---

## 1. 中心主張（アブストラクト要約）

> Cognitive science should turn its attention to developing theories of human cognition that cover the full range of human perceptual, cognitive, and action phenomena.

- 認知科学は細分化したマイクロ理論の時代を超え、**Unified Theories of Cognition (UTCs)** に向かうべき
- 単一論ではなく複数候補が競合する時代を想定
- **SOAR** を exemplar（模範）として提示
  - **唯一の候補ではなく、実現可能性の存在証明**
- 19年前の "You can't play 20 questions with nature and win" (Newell 1973) の延長線上

### 主要論点

> A single system (the mind) produces behavior.

単一システム（心）が行動を生み出すなら、理論も統一されるべき。

---

## 2. 本書の構造（Précis の章立て）

| 章 | 内容 |
|----|------|
| 1. Introduction | UTCの必要性と戦略 |
| 2. Foundations of Cognitive Science | 知識システム、表現、計算、シンボル、アーキテクチャ、知能、問題空間 |
| 3. Human Cognitive Architecture | 時間スケール、実時間制約、認知帯域、合理的帯域 |
| 4. Symbolic Processing for Intelligence | SOAR アーキテクチャ導入 |
| 5. Immediate Behavior | 即時応答（chronometric 実験） |
| 6. Memory, Learning, Skill | 熟達、エピソード記憶 |
| 7. Intendedly Rational Behavior | 問題解決（cryptarithmetic, syllogism） |
| 8. Along the Symbolic Road | 発達、言語 |
| 9. Toward UTC | 次のステップ |

---

## 3. 重要概念

### 3.1 Knowledge Systems

- 人間を**知識システム**として記述: body of knowledge + goals → 行動
- 知識は信念（computer science 的意味）— 正誤は問わない
- 行動は **principle of rationality** で決まる: "knowledge is used in the service of the agent's goals"
- Fermat の最小時間原理のような目的論的原理

### 3.2 Representation Law

- X が T で Y に変換される外界の事象に対応して、内部での x → y 変換が存在
- 多様な外界に対応するには **composable transformations**（計算可能性）が必要
- これが symbol systems の必要性の論拠

### 3.3 Symbol Systems（象徴系）

**4要素**:
1. Memory（独立に変更可能な構造を保持）
2. Symbols（パターンで distal access を提供）
3. Operations（symbol structures を入出力）
4. Interpretation processes（structures を operations として実行）

> 人間の心は symbol system である。

根拠: 人間の応答関数の多様性（読む、書く、料理、踊り、ラップ…）は symbol system でなければ実現不能。

### 3.4 Architecture（アーキテクチャ）

- **structure と content を分離する固定構造**
- knowledge-level system を実現するメカニズム
- computer hierarchy では register-transfer level、生物では neural structure level

---

## 4. Real-Time Constraint（実時間制約）

本書の最も重要な理論的貢献の一つ。

### 時間スケール階層

| 帯域 | レベル | 時間 | 内容 |
|------|--------|------|------|
| **Biological** | Organelle | 100μs | — |
| | Neuron | 1ms | — |
| | Neural circuit | 10ms | — |
| **Cognitive** | Deliberate act | 100ms | fastest conscious action |
| | Cognitive operation | 1s | simple operation |
| | Unit task | 10s | composed operation |
| **Rational** | Task | min–hour | goal-directed |
| **Social** | Society | days+ | distributed activities |

### 制約

- ニューロン回路は 10ms、即時認知は 1s 以内
- → **わずか約 100 operation times しか使えない**
- **認知は 2 levels だけ**で神経から構築されねばならない
- この制約は AI の "too-slow" 算法（視覚・言語）への批判と、**認知アーキテクチャの形状への強い制約**

---

## 5. SOAR アーキテクチャ（Chapter 4以降の核）

### 5.1 Performance

- **問題空間** (problem space) 全タスクを表現
- **Long-term memory**: production system（条件→アクションのルール）
- **Working memory**: 現在の状況
- **Decision cycle**: 2段階
  1. **Elaboration**: 適合するすべての production が発火、preferences を生成
  2. **Decision**: preferences を解釈して operator 選択

### 5.2 Impasse（行き詰まり）

- 知識が operator を一意に決定できないとき impasse 発生
- アーキテクチャが自動的に **subgoal を生成**
- サブゴール内で選択/適用を再帰

### 5.3 Chunking（唯一の学習機構）

- impasse が解決されたとき、**automatic に新 production（chunk）を生成**
- Pre-impasse 環境で使われた要素を "if"、導出結果を "then"
- 再帰的 impasse → chunk hierarchy

> Chunking applies to all impasses, so learning can be of any kind whatever.

**Law of Practice**（べき法則）の説明機構として機能。

### 5.4 Perception/Motor

- Working memory を共通バス
- Perceptual modules: 感覚 → working memory
- Encoding/decoding productions は他知識と相互作用
- 感覚・運動モジュール自体は cognitively impenetrable

---

## 6. SOAR による認知現象のカバレッジ（Chapter 5–9）

Chapter ごとに SOAR を異なる現象に適用:

| 章 | 現象 | 代表的 SOAR モデル |
|----|------|--------------------|
| 5 | Immediate behavior（SR compatibility, Sternberg） | Kornblum の重複仮説 |
| 5 | Transcription typing | 運動スキル |
| 6 | Episodic memory, practice | 熟練獲得 |
| 7 | Cryptarithmetic, syllogistic reasoning | Polk & Newell (1988) |
| 8 | Sentence verification, Instruction-taking | NL-SOAR (Lehman, Lewis, Newell 1991) |
| 9 | Balance beam task | 発達（Piagetian transitions） |

→ **3000 以上の経験的 regularities** が存在すると推定

---

## 7. 主要な方法論的立場

### Lakatosian vs Popperian

> A science has investments in its theories and it is better to correct one than to discard it.

- SOAR は反証可能な単一理論ではなく **research programme**
- 予測が失敗しても即捨てるのではなく、理論を改訂
- この立場が後に議論を招く（Popperian から批判される）

### Exemplar という戦略

- SOAR が唯一の UTC ではない
- ACT*（Anderson）、Model Human Processor（Card et al.）、CAPS（Just & Carpenter）なども候補
- **具体例を示すことで UTC の実現可能性を論証**

---

## 8. 限界と批判への応答

- SOAR は underspecified（アーキテクチャは進化中、SOAR 4 → 5）
- 誤りがあると分かっていても、Lakatos 的に改訂
- "Don't bite my finger, look where I'm pointing" (McCulloch)
- "You can't play 20 questions with nature and win" — 単一実験では勝てない

---

## 9. CS 222 での位置づけ

- **Lecture 04**: 認知アーキテクチャ概説のアンカー論文
  - 04_2 "A Gentle Introduction to SOAR" (Lehman, Laird, Rosenbloom 2006) と対をなす
- **Lecture 05**: Generative Agents (Park et al. 2023) の理論的祖先
  - Park 論文 §2.2 でも Newell 1990 を引用
  - 05_2 CoALA (Sumers et al. 2024) は CoALA を "cognitive architecture for language agents" として SOAR を参照
- **Park氏の議論との関係**:
  - Newell の "exemplar" 戦略は Park が Generative Agents で採った戦略（唯一解ではなく可能性の存在証明）
  - Real-time constraint は Park の即時応答シミュレーションの概念的背景
  - Memory/retrieval/planning の三層構造は SOAR の WM/LTM/decision cycle に対応

---

## 10. 主要引用

### Newell が引用
- Card, Moran, Newell (1983) *The Psychology of Human-Computer Interaction*（Model Human Processor）
- Anderson (1983, 1991) ACT*
- Laird, Newell, Rosenbloom (1987) "SOAR: An architecture for general intelligence"
- Fodor & Ballard (1982) 100-step constraint
- Salthouse (1986) transcription typing

### 本論文/書を引用する後続
- Anderson (1993) *Rules of the Mind* (ACT-R)
- Laird, Lehman, Rosenbloom (2006) "A Gentle Introduction to SOAR" (04_2)
- Sumers et al. (2024) CoALA (05_2)
- Park et al. (2023) Generative Agents (05_1)
- すべての cognitive architecture 系文献

---

## 要点

1. **Unified Theories of Cognition**: マイクロ理論の時代は終わり、認知全体を覆う単一機構理論を構築すべき
2. **Exemplar 戦略**: SOAR を唯一解ではなく可能性の存在証明として提示
3. **Real-time constraint**: ニューロン 10ms から認知 1s まで、わずか 2 system levels
4. **Symbol systems**: 人間の応答関数多様性から必然的。4要素 (memory, symbols, operations, interpretation)
5. **SOAR の 3 機構**: Problem space + Decision cycle + Chunking (唯一の学習機構)
6. **Impasse-driven learning**: 知識不足を自動検出、subgoal 生成、結果を chunk 化
7. **Lakatosian 立場**: 誤りを捨てず理論を改訂するほうが cumulative に正しい
8. **CS 222 では**: 認知アーキテクチャの歴史的出発点として、Generative Agents と CoALA への系譜の起点
