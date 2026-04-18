# 07_1 — The Role of Emotion in Believable Agents

## 書誌情報

- 著者: **Joseph Bates**
- 所属: Carnegie Mellon University, School of Computer Science, College of Fine Arts
- 掲載: **Communications of the ACM** 37(7): 122–125 (July 1994)
- DOI: https://doi.org/10.1145/176789.176803
- プロジェクト: The Oz Project (CMU, 1990s 初期の believable agent 研究)
- 資金: Fujitsu Laboratories, Mitsubishi Electric Research Laboratories
- Lecture 07 の補足論文（believable agents の哲学的・芸術的基盤）

---

## 1. 中心問い

> How do we build believable agents?

「Believable agents」とは Disney 映画の登場人物のような **生命の幻想 (illusion of life)** を与える創造物。「誠実」「信頼できる」ではなく「観客の不信停止 (suspension of disbelief) を可能にする」意味。

**本論文の主張**: 従来 AI が推論・問題解決に注力してきたが、**believability の本質には emotion が必要**。Arts（特に Disney animators）からの洞察が決定的。

---

## 2. 伝統的 AI vs 芸術的キャラクター構築

### 2.1 AI の系譜
- 推論 (reasoning)、問題解決、学習、概念形成 → 「知能」として追求
- 理想化された科学者の特性 → 研究者コミュニティの価値観を反映
- Bledsoe の 1985 年 AAAI 会長演説: コンピュータ友達の夢、「歩く」「ping-pong する」エージェント

### 2.2 芸術（特にキャラクターアニメーション）の系譜
- **Disney animators (1930s)**: Thomas & Johnston の *The Illusion of Life* (1981)
- 制約: 手描きの flat-shaded line drawings、何十万枚もの frame 作成
- 簡素で非現実的な imagery でありながら「生命」を伝える → **必然的に本質 (essence) を抽出**

### 2.3 Bates の主張
> It can be argued that while scientists may have more effectively recreated scientists, it is the artists who have come closest to understanding and perhaps capturing the essence of humanity that Bledsoe, and other AI researchers, ultimately seek.

芸術家の洞察、特に *The Illusion of Life* は **computational believable interactive characters** 構築の鍵。著者はこれを "**believable agents**" と呼び、**Oz project** の研究目標に据える。

---

## 3. Emotion in Believable Characters

### 3.1 Disney animators の判断
> The apparent desires of a character, and the way the character feels about what happens in the world with respect to those desires, are what make us care about that character.

- キャラクターの**欲望 (desires)** と、世界で起きる出来事への**感情的反応**が、観客の関心を生む
- **感情のないキャラクターは生命なし** (lifeless; a machine)

### 3.2 Thomas & Johnston の 3 つの Maxim（感情表現の要件）

1. **Clearly defined emotional state**
   - 作家 (animator) は各瞬間のキャラクターの感情状態を知っていなければならない
   - 観客が感情を確実に帰属できるように

2. **The thought process reveals the feeling**
   - Disney の技法: 行動が思考過程を示し、思考が感情で色付けられる
   - 観客は **キャラクターがどう考えるか** から感情を読み取る

3. **Accentuate the emotion**
   - Time を wisely 使って emotion を確立・伝達・味わわせる
   - **Foreshadowing**, **exaggeration**, **toning down other simultaneous action**

---

## 4. Oz Project の Woggles（実装例）

### 4.1 システム概要
- 1992 年の art work "**Edge of Intention**"
  - AAAI-92 AI-based Arts Exhibition で初公開
  - SIGGRAPH-93、Ars Electronica '93、Boston Computer Museum で展示
- **Woggles**: 3 自動化生物
  - Shrimp（小）
  - Bear（中、Shrimp と対等）
  - Wolf（大、敵対的）

### 4.2 アーキテクチャ
- **Goal-directed, behavior-based architecture** (Brooks, Maes 系)
  - 計画なし、ほとんど世界モデルなし
  - Minimalist conception of goals → 動的な behavior 操作
- **Emotion module** (OCC theory 基盤: Ortony, Clore, Collins 1988)
  - Goals + event appraisals → emotional state
  - Emotions は **explicit に表現**、behaviors を compose
- 設計狙い: robot 的 robustness + キャラクター的 personality

### 4.3 感情 → 行動マッピング

OCC theory のアナログ例:
- **Anger**: 重要 goal が別 Woggle に阻害されたとき

**複数感情の同時存在**:
- 異なる強度の多感情を統合し、1–2 個の primary emotion を**明確に表現**
- **Maxim 1 に対応**（clearly defined emotional state）
- 著者たちは「この解決策が十分か不確か」と述べる

### 4.4 Personality-specific Feature Mapping

Maxim 2（thought process reveals feeling）に対応:
- 各 emotion → **personality-specific な behavior feature** にマップ
- "fear" → Shrimp: "alarmed" feature / Wolf: "aggressive" feature
- Features は action-generating components に systemic 影響
- あらゆる組合せで emotion を伝える action を生成

**Behavioral features**: モジュール化抽象
- 将来の再利用を意図
- しかし実装中、emotion → action マッピングが **ad hoc に depend on abstraction barrier の break** になることが多い
- Oz グループ内で討論中（抽象化は可能か？）

### 4.5 Emotion Quirk と Abstraction Breaking

**例 (Shrimp の programming error)**:
- ある action が nervous tick として地面を頭で叩き続ける
- 観察者は Shrimp の mental state への incorrect theory を発展させる
- → Chuck Jones の insight: "**quirks give life to characters**"
- Bugs Bunny の 8歳の Warner Brothers animator の逸話: 食べた grapefruit rind を頭に wear
- → architectures must **support quirks**, regularities を barrier で拘束しすぎない

### 4.6 Exaggeration & Anticipation（Maxim 3）

**Woggles は不十分**:
- Moping behavior: muscle tone 下げ、jumps 短く、actions 遅く
- 創造者の感想: "started with reasonable looking behavior but it wasn't clear enough, so I tripled and then doubled again, and still only got a mope that people recognize"

**Anticipation**:
- Action 発生前のブロードキャスト（例: rapid forward motion 前の rear back）
- 厳密に unrealistic だが、人が motion の meaning を把握するのに必須
- 未経験 sport 観戦が分かりにくい理由と同じ

**Staging**:
- 複数キャラクター間で action のタイミング・空間調整
- Edge of Intention で複数 Woggles が同時 jump/gesture し、observer が主要 event を追えない
- Warner Brothers 式の staging が未実装

---

## 5. Conclusion（Bates の主張）

### 5.1 Believability は emotion が一次要素

> Emotion is one of the primary means to achieve this believability, this illusion of life, because it helps us know that characters really care about what happens in the world, that they truly have desires.

### 5.2 現代 video game の反例

> In contrast, the fighting characters of current video games show almost no reaction to the tremendous violence that engulfs them. Such showing of reaction is a key to making characters believable and engaging, which is perhaps why these characters engender no concern—they are merely abstract symbols of action.

### 5.3 "Illusion of Life" と "genuine life" の区別

- Turing's test, Dennett の "intentional stance" が関連
- 哲学的論争は continue するが、**実用的には構築を進めるべき**
- "Sufficient success in this regard may alter or eliminate the philosophical debate"

### 5.4 Reality vs Realism

> Artists use reality in the service of realism, for example by carefully studying nature, but they do not elevate it above their fundamental goal. Mimicking reality is a method, to be used when but only when appropriate, to convey a strong subjective sense of realism. It is art which must lead the charge here.

### 5.5 Alternative AI との関係

- Brooks の reactive robotics は一つの believability 要件（reactivity）を過大評価
- 他の要件（**emotion, personality**）を過小評価
- 研究者は **emotion/personality** など芸術家が重要と言う要素で同様の方法論的シフトをすべき

---

## 6. Oz Project の他の概念

### Goals
- Minimalist goal system: 計画を持たない、short-term behavior を変調
- Brooks (1986)、Maes (1989) の反応型 architecture と親和

### Appraisals
- Goals に対する event の evaluations
- OCC のようなルール体系で emotion を derive

### Reactivity
- 世界変化への即時反応
- しかし Oz project は reactivity 以外も強調

### Situated social competence
- Social 文脈への適切な振る舞い
- Goals, emotions, reactivity と統合

---

## 7. CS 222 での位置づけ

### Lecture 07: Emotion in Agents
- **アンカー論文として引用**
- 07_2 (Hewitt/Ashokkumar) とは異なる視点: 生成 emotion vs LLM simulation の predictive accuracy
- 認知と感情の統合議論で再登場

### Park 氏の Generative Agents (05_1) との関係
- **Park 論文 Reference [10] で明示引用**: "Joseph Bates. 1994. The Role of Emotion in Believable Agents. Commun. ACM 37, 7 (1994), 122–125"
- "believable agents" 概念の**起源**。Park 氏は同用語を継承
- Park 論文 §2.2: "Believable agents are designed to provide an illusion of life and present a facade of realism..."（Bates の文言をほぼ直接継承）
- Disney animators からの啓発を共有（Thomas & Johnston も Park が引用）
- Oz project は Park 氏の Smallville の精神的前身

### 他の CS 222 論文との接続
- **03_1 Social Simulacra** (Park 2022): "believable" の用語系譜
- **04_1 Newell / 04_2 SOAR**: emotion を加える試み (Marinier & Laird 2004) で接続
- **15_1 Generative Ghosts** (Morris 2024): emotion と persistence の倫理議論
- **15_2 The Code That Binds Us**: human-AI emotional relationships
- **11_1 Lorenz The Essence of Chaos**: quirks が characters の「生命」を生む洞察と類比

### 論点
- LLM-based agents (Park 2023) が**明示的 emotion module なし**に believable になるのは、LLM が training data で人間の感情反応を学習しているから
- しかし Park の future work に "emotion integration" が挙げられる（Marinier & Laird 2004 の延長）
- **Oz の behavior-based + OCC** vs **Generative Agents の LLM + memory**: 両者のアーキテクチャ比較
- Bates の "quirks" 概念は Park 氏の **embellishments/hallucinations** の positive reframing としても読める

---

## 8. 主要引用

### 論文が引用
- [12] **Thomas, F. and Johnston, O. (1981)** *Disney Animation: The Illusion of Life* — 芸術家の側の礎
- [3] **Bledsoe, W. (1986)** "I had a dream: AAAI presidential address" — AI 側の願望
- [10] **Ortony, A., Clore, G., Collins, A. (1988)** *The Cognitive Structure of Emotions* — 感情計算モデル
- [4] Brooks, R. (1986) 反応型ロボット制御
- [8] Loyall, A.B. and Bates, J. (1993) "Real-time control of animated broad agents"
- [9] Maes, P. (1989) "How to do the right thing"
- [11] Reilly, W.S. and Bates, J. (1992) "Building emotional agents" (CMU-CS-92-143)
- [6] Jones, C. (1989) *Chuck Amuck*
- [7] Lasseter, J. (1987) SIGGRAPH "Principles of traditional animation applied to 3D computer animation"
- [2] Bates, Loyall, Reilly (1992) European Workshop on Autonomous Agents
- [5] Dennett, D. (1987) *The Intentional Stance*

### 本論文を引用する後続研究
- **Park et al. (2023) Generative Agents** [10] ← 05_1
- Believable agents 研究全般
- Social robotics
- Game AI (NPCs)
- Affective computing（Picard ら）

---

## 9. 要点

1. **Believability ≠ honesty**: 観客の「不信停止」を可能にする **illusion of life**
2. **AI と Arts の対比**: AI は reasoning/problem-solving を追求、artists は **essence of humanity** を抽象化
3. **Disney animators の洞察**: Thomas & Johnston *Illusion of Life* の 3 maxim:
   - (1) Clearly defined emotional state
   - (2) Thought process reveals feeling
   - (3) Accentuate emotion (foreshadowing, exaggeration)
4. **Oz project の Woggles**: Behavior-based architecture + OCC emotion theory による実装
   - 3 生物（Shrimp, Bear, Wolf）、"Edge of Intention" 芸術作品として公開
5. **Personality-specific feature mapping**: fear → Shrimp "alarmed" / Wolf "aggressive"。行動モジュール化は抽象化困難
6. **Quirks の重要性** (Chuck Jones): 一貫性のない奇癖が character に生命を与える
7. **Emotionless characters は lifeless**: 現代ビデオゲームの暴力への無反応が非信憑の原因
8. **Believable agents という用語の起源**: Park 氏 Generative Agents (05_1) が直接継承
9. **Art leads science**: Bates は芸術家に学ぶ姿勢を強調、"Mimicking reality is a method, to be used only when appropriate"
10. **CS 222 では**: 40 年前の論文だが、**Park 氏の "believable proxies of human behavior" の用語・思想的源流**。LLM 時代の generative agent の哲学的基盤
