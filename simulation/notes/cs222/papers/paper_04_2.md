# 04_2 — A Gentle Introduction to SOAR, 2006 Update

## 書誌情報

- 著者: **Jill Fain Lehman**, **John E. Laird**, **Paul S. Rosenbloom**
- 原著: Sternberg & Scarborough (1996) *Invitation to Cognitive Science, Volume 4* に収録
- 2006 年更新版: John Laird が 2006年1月に更新（SOAR 9 を記述）
  - 原著は SOAR 6、更新版は SOAR 9
  - 主要変更: architectural support for goal/problem space の削除、substate maintenance の変更 (Wray & Laird 2003)、new long-term memories と learning mechanisms 追加
- 資金: NSF Grant No. 0413013
- Lecture 04 の補足論文（04_1 Newell と対）

---

## 1. 研究問題

認知科学（psychology, linguistics, anthropology, AI）は **microtheories** を生み出してきたが、**各理論はバラバラ**で統合されていない:
- 心理言語学の **garden path** 現象: "(a) Without her contributions we failed." は易しく、"(b) Without her contributions failed to come in." は難しい
- Fitts' Law（運動）、Sternberg（短期記憶）、Tulving（serial position curve）…
- 心は単一システムなのに、理論群は単一パズルを組み立てていない

**Newell の主張**: Unified Theories of Cognition (UTCs) を目指すべき。

本論文は **SOAR を UTC 候補として教育的に紹介**する。

---

## 2. 核となる等式

> **BEHAVIOR = ARCHITECTURE + CONTENT**

- **Architecture**: 固定されたメカニズムと構造
- **Content**: それが処理する知識・情報
- どちらか片方では行動は生まれない

### Architecture は何の理論か

> An architecture is a theory of what is common among much of the behavior at the level above it.

認知アーキテクチャ = **人間認知の基底機構の理論**

---

## 3. Cognitive Behaviors の 6 共通特性（Newell 1990 を踏襲）

Soar の設計を動機づける:
1. **Goal-oriented**（目標志向）
2. **Rich, complex, detailed environment で作動**
3. **Requires large knowledge**
4. **Symbols & abstractions** を使う
5. **Flexible, environment-responsive**
6. **Learns from environment and experience**

---

## 4. 具体例: 野球投手 Joe Rookie

全章を貫く教材的シナリオ:
> Joe Rookie は Pirates のルーキー投手。Sam Pro に curve ball を投げる。Sam が打つ。Joe が 1バウンドで捕球し、一塁送球で Sam をアウト。

必要知識 (Table 1):
- K1: ゲーム内オブジェクト知識（baseball, inning, out...）
- K2: 抽象事象・特定エピソード
- K3: ゲームのルール
- K4: 目的（get batter out）
- K5: 行動/方法（curve ball, throw to first...）
- K6: 行動選択の条件
- K7: 物理的アクション（how to throw）

---

## 5. Behavior as Movement Through Problem Spaces

### 問題空間 (Problem Space) の構成要素

| 要素 | 記号 | 意味 |
|------|------|------|
| 初期状態 | S0 | 入力から作られる出発点 |
| 目標状態 | G (shaded) | 目的達成を示す features を持つ |
| 状態 | Si | features & values の組（内部+外部） |
| Operator | 矢印 | 状態を変換 |

### Principle of Rationality

> If an agent has knowledge that an operator application will lead to one of its goals then the agent will select that operator. (Newell 1982)

---

## 6. Memory の 3 種（SOAR 9 の革新）

| 種類 | 内容 | 例 |
|------|------|-----|
| **Procedural LTM** | ルール（if-then）。行動制御の中心 | "If batter left-handed and 2 tied pitches, prefer curve" |
| **Semantic LTM** | 世界についての事実 | "Sam Pro の nickname は Crash"、"9 innings 3 outs" |
| **Episodic LTM** | 特定経験の記憶 | 過去の対戦時の特定状況 |

**Working Memory (WM)**: 現在状況。LTM からの retrieval で構築。State 階層を保持。

キー性質:
- LTM は impenetrable: **直接 examine できず**、WM への retrieval でのみアクセス
- Procedural: automatic retrieval (decision cycle で)
- Semantic / Episodic: deliberate retrieval (cue を WM に作成)

---

## 7. Decision Cycle（5 フェーズ）

1. **Input**: perception → WM
2. **Elaboration**: 並列に rule fire、preferences 生成
   - 並列 wave で進行、新 WM 要素が更なる発火を誘発
   - quiescence（新規 fire なし）で終了
3. **Decision**: preferences を解釈、operator 選択
   - 不可能なら **impasse** 発生
4. **Application**: 選択 operator を実行、state 変換
5. **Output**: motor command → 環境

### 制約

- **単一 operator のみ選択可能**（cognitive bottleneck）
- 並列動作は「パッケージ化された operator」としてのみ

---

## 8. Impasse と Substate

### 4 種の Impasse

| 種類 | 条件 |
|------|------|
| **Operator-tie** | 複数 operator が提案されて優先順位が不明 |
| **Operator no-change** | operator 選択されたが application 知識不足 |
| **State no-change** | operator application で state が変わらない |
| **Conflict** | 矛盾する preferences |

Impasse 発生 → **自動的に substate を生成**。下位ゴールで operator を提案・適用。

### 例: Joe の選択困難

左打者 Sam に curve/fast の両方が提案。r5（優先ルール）が欠けていると操作-tie 発生。

Substate で:
- **Recall operator**: episodic memory から過去試合を cue で検索
- **Evaluate operator**: その episode から現在の選択肢を評価
- 「風が強い日に fast ball でホームラン打たれた」という事例 → curve を preferred

---

## 9. Learning の 4 機構（SOAR 9 の大改訂）

### 9.1 Chunking（SOAR 最古）

- Impasse が解決されたとき、**自動的に新ルール（chunk）生成**
- Pre-impasse environment の "used" 要素を "if"、得られた preference を "then"
- 例: c1 = "If curve/fast tied AND batter=Sam Pro AND left-handed AND weather=windy THEN prefer curve"
- **Deductive, compositional learning**
- 一度学んだ chunk は類似状況で自動発火 → 同 impasse に再遭遇しない

### 9.2 Reinforcement Learning（Nason & Laird 2004）

- Preferences の value を報酬に応じて調整
- TD 的更新: 現在予測 vs 次 decision の最大予測 + reward
- **Principle of rationality を直接支持**

### 9.3 Episodic Memory（Nuxoll & Laird 2004）

- Automatic recording: WM の近傍サブセットを時系列で保存
- Cue based retrieval: best match を WM に再構成
- **No generalization**（パッシブ学習）
- 過去質問応答、行動シミュレーション、長期目標追跡に使用

### 9.4 Semantic Memory

- Working memory 構造の共起から静的な宣言的知識を抽出
- 場所・時間から切り離された "知っている" 知識

→ **BEHAVIOR = ARCHITECTURE + CONTENT** の content を4方向で更新可能

---

## 10. 全体モデル: Joe Rookie の拡張

- Top state: perception/motor、comprehend/get-batter-out/walk-batter
- Throw-to-base, Recall, Pitch（subspaces）
- Comprehend → Language → Gesture（catcher 信号理解）

各 impasse で chunking、全体の knowledge を integration。

---

## 11. Soar as UTC 候補

### 実装された content theories (examples)

- **NL-Soar**: 人間自然言語理解・生成 (Lehman, Lewis, Newell 1991; Lewis 1993)
- **NTD-Soar**: NASA Test Director モデル (Nelson, Lehman, John 1994) — NL-Soar + NOVA を統合
- **Instructo-Soar**: 対話型指示学習 (Huffman 1993)
- **IMPROV**: 知識修正 (Pearson & Laird 1995)
- **TacAir-Soar** (Jones et al. 1999): 米空軍 tactical air mission シミュレータ、>8000 rules
- **RWA-Soar** (Tambe et al. 1995): 回転翼機
- **Soar MOUTBOT** (Wray et al. 2004): 市街戦訓練 adversary

→ **構成要素の互換性**（NL-Soar を使うと NTD-Soar は "language の制約を全面的に受ける"）

### 現状の謙虚な立場

> As these examples show, we are still a long way from creating a full unified theory of cognition.

Emotion の統合も進行中 (Marinier & Laird 2004)。

---

## 12. CS 222 での位置づけ

- **Lecture 04**: SOAR の現代的教育的紹介として 04_1 Newell と対で読まれる
- **Lecture 05**:
  - Generative Agents (Park et al. 2023) の Related Work で **Quakebot-SOAR** (Laird 2001) や **TacAir-SOAR** を "cognitive architecture が NPC を生成した祖先事例" として引用
  - ただし Park 氏は SOAR の限界（manual procedural knowledge の制約、open world 未対応）を指摘
- **Lecture 05 (CoALA, 05_2)**:
  - Sumers et al. (2024) が **SOAR を CoALA の直接的祖先**として扱う
  - CoALA は SOAR の memory modules（procedural/semantic/episodic）をそのまま言語エージェントに移植
  - "just as productions indicate possible ways to modify strings, LLMs define a distribution over changes or additions to text"
- **Park氏の議論との関係**:
  - SOAR の rigid な symbol-based 制約 vs LLM の flexible 自然言語 representation
  - SOAR の manual-rule vs LLM の pretrained prior
  - しかし SOAR の三層メモリ（procedural/semantic/episodic）は CoALA・Generative Agents で形を変えて継承

---

## 13. 主要引用

### 論文が引用
- Newell (1982) "The knowledge level"
- Newell (1990) *Unified Theories of Cognition*
- Newell & Simon (1972) *Human Problem Solving*
- Laird, Newell, Rosenbloom (1987) "SOAR: An architecture for general intelligence" *Artificial Intelligence* 33
- Tulving (1983, 2002) Episodic memory
- Fitts (1954), Sternberg (1975)

### 本論文を引用する後続
- Sumers et al. (2024) CoALA
- Laird (2012, 2019, 2022) SOAR の最新更新
- Kotseruba & Tsotsos (2020) cognitive architecture survey

---

## 要点

1. **BEHAVIOR = ARCHITECTURE + CONTENT**: アーキテクチャと知識内容の分離が理論的核
2. **Problem space hypothesis**: 全認知行動を goals, states, operators の movement として表現
3. **6 cognitive properties**: goal-oriented, complex env, much knowledge, symbols, flexibility, learning
4. **3 LTM types**: Procedural (rules), Semantic (facts), Episodic (experiences) ← SOAR 9 の拡張
5. **Decision cycle**: input → elaboration → decision → application → output、単一 operator が selection される bottleneck
6. **Impasse-driven learning**: 知識不足を automatic に検出し subgoal 生成、chunking で結果を ルール化
7. **4 learning mechanisms**: Chunking（演繹的合成）、RL（報酬）、Episodic（自動記録）、Semantic（共起抽出）
8. **UTC 実装事例**: NL-Soar, TacAir-Soar (8000+ rules), Soar MOUTBOT など大規模シミュレータで実用
9. **CoALA (05_2) の直接的祖先**: memory modules の三層構造を LLM 時代に継承
10. **Generative Agents (05_1) の知的祖先**: memory stream, reflection, planning の概念的源流
