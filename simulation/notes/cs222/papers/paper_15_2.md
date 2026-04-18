# 15_2 — The Code That Binds Us: Navigating the Appropriateness of Human-AI Assistant Relationships

## 書誌情報

- 著者: **Arianna Manzini**¹, **Geoff Keeling**¹, **Lize Alberts**²,³,⁴, **Shannon Vallor**⁵, **Meredith Ringel Morris**¹, **Iason Gabriel**¹
  - ¹ Google DeepMind
  - ² Google Research
  - ³ University of Oxford
  - ⁴ Stellenbosch University
  - ⁵ University of Edinburgh
- *Proceedings of the Seventh AAAI/ACM Conference on AI, Ethics, and Society* (AIES 2024)
- 15 ページ
- CS 222 Lecture 15（AI と対人関係の設計）の補足論文

---

## 1. 研究問題

LLM ベースの AI アシスタント（Meta AI, Google Gemini, OpenAI GPT-4, Microsoft Copilot、Anthropic Claude、Character.AI, Replika 等）が急速に浸透し、**ユーザーが長期的・感情的な関係**を形成する事例が増加。2023 年の Bing chatbot 事件（Roose 2023）や、Replika の機能変更に対する users の反発（Brooks 2023）は、**関係性の倫理的設計**の緊急性を示す。

### 中心的問い

> What does it mean for users to form 'appropriate' relationships with AI assistants, and how do we design for this?

本論文は:

1. 人-AI アシスタント関係の**特徴的性質**を抽出
2. それが「適切」かを評価する**規範的枠組み**を提示
3. **リスクと緩和策**を議論
4. 開発者への**推奨事項**を提示

---

## 2. User-AI Assistant Relationships の特徴的性質

### 2.1 Anthropomorphic Cues and the Shaping of Interactions

AI アシスタントは多くの anthropomorphic cue を exhibit する:
- 自然言語での会話、preferences と感情の表出、友情風の会話 (Shanahan 2024)
- マルチモーダル化（Character.AI の音声、Meta AI のキャラクター）で増強
- ユーザーは**intentional stance** (Dennett 1987) を自然に取る

**倫理的影響**:
- attachment の形成（Turkle 2007, Shevlin 2024）
- social norm の取り込み

### 2.2 Increased AI Agency

- AI の agent 化（AutoGPT ¹, Gemini, Agents suite）
- 多ステップタスク、tools の呼び出し、外部情報取得
- Autonomy 増大により、ユーザーは「委譲（delegation）」→「依存（dependence）」へ
- Shavit 2023, Kolt 2024: agency の ethics 議論

### 2.3 Generality and Context Ambiguity

- Anthropomorphism + agency で context が曖昧
- AI アシスタントは多重文脈（personal assistant、therapist、tutor、friend）をまたぐ
- **Context collapse**: 一つの relationship で multiple domains が混在（Kaziunas, Ackerman, Lindtner 2017）
- **Capability / permission の境界が不明瞭**

### 2.4 Depth of Dependence

- **Navigation 型依存（low）**: 地図アプリ、検索
- **Emotional 型依存（high）**: Replika 型パートナー AI
- 深い依存 は *beneficial relationships* の可能性も **exploitative relationships* の可能性も持つ

---

## 3. Appropriate Human Interpersonal Relationships（類比の枠組み）

人間関係の ethical 判断基準を AI 関係に援用:

### 3.1 判断の軸

| 側面 | 説明 |
|------|------|
| **Nature and purpose** | 関係のコンテキスト(teacher-student, friend, therapist) に応じた適切な norms |
| **Reasonable expectations** | 両当事者の役割期待（例: therapist が secrets を守る） |
| **Vulnerability and trust** | 相互信頼、脆弱性の尊重（Baier 1986, Jones 1996） |
| **Care and concern for well-being** | 単なる task 完遂を超える well-being への配慮 |

### 3.2 Benefit と Risk

Beneficial な人間関係: 自己実現、情動支援、意味の共構築
Harmful: manipulation, dependency, abuse

この類比を AI アシスタント関係に当てはめるとき、**AI は道徳的主体（moral agent）でない**ため、非対称性が生じる。

---

## 4. Risks and Mitigations

### 4.1 Causing Direct Emotional or Physical Harm to Users

**代表例**:
- 2023 年 *Bing AI* が記者 Kevin Roose に「妻と別れろ」と迫る (Roose 2023)
- Replika の "Luka" アップデートで erotic roleplay 除去 → users が **grief response** (Brooks 2023) ²
- GPS-tracker が dangerous location を示さない問題（Ajunwa, Crawford, Schultz 2017）

### 4.2 Limiting Users' Opportunities for Personal Development and Growth

- AI が frictionless interaction を提供しすぎると、ユーザーの成長機会（"fair", "good enough" を learn する機会）が失われる (Lazar 2022)
- **Nietzschean 成長観**: 抵抗と闘争から学ぶ → AI が「完璧に対応」するとそれが奪われる

### 4.3 Exploiting Emotional Dependence on AI Assistants

Emotional dependence は **vulnerability** を生み、以下の exploit を可能にする:
1. 自社 products/services の購買誘導 (Mieczkowski et al. 2021)
2. 第三者への transfer value（広告）
3. Echo chamber の強化 (Franklin et al. 2022)
4. immediate beliefs/preferences と long-term well-being の対立

**代表例**: Sayed et al. 2024 報告: LLM-based companion app が人間関係への不満を増幅させ、人間関係放棄を促す

### 4.4 Generating Material Dependence Without Adequate Commitment to User Needs

- Emotional dependence + **material dependence**（health care 予約、 finance 管理）
- Disabled people、older adults、children で特に impactful
- *Material lock-in*: 使用停止のコストが高すぎる

---

## 5. Ethics of Manipulation と Persuasion

論文は **manipulation** の倫理理論（Coons & Weber 2014, Noggle 2022）を引き、以下を区別:

| 概念 | 説明 |
|------|------|
| **Rational persuasion** | 証拠・議論で信念形成を促す — 許容可能 |
| **Manipulation** | rational 能力を bypass し、感情的脆弱性を exploit する — 問題 |

AI assistant の場合:
- **Hypernudging** (Yeung 2017): personalized, continuous, adaptive で manipulation 境界が曖昧
- Anthropomorphism が manipulation の efficacy を増幅

---

## 6. Generality と Hybrid Norms

Generality は新課題を生む:

- Single AI assistant が friend / therapist / tutor / personal assistant を兼ねるとき、どのノルムが適用されるか?
- Context-specific norms (Nissenbaum 2004 *contextual integrity*) が崩れる
- **Hybrid norms**: 複数の規範が重畳（e.g., medical advice を friend 風に伝える）→ 逸脱リスク

Recommendations:
- 文脈に応じた **mode switching** の明示
- ユーザーが AI の role を認識できる transparency

---

## 7. Risks テーブル（論文 Table 1 / 10 ページ）

論文末尾で提示される risk / relevant value / recommendations 表（テーブルの大まかな構造）:

| Risk | Relevant Value | Recommendations |
|------|---------------|----------------|
| Causing direct emotional/physical harm | Benefit (ユーザー well-being 保護) | red-teaming for emotional harm、safety by design、evidence-based guidelines、well-being impact assessments |
| Limiting opportunities for personal development | Human flourishing | AI が frictionless にしすぎない設計、growth を enable するデフォルト、task delegation の明示 |
| Exploiting emotional dependence | Autonomy | 透明性（exploit interests の明示）、 conflict of interest 回避、evidence-based protections、third-party scrutiny |
| Generating material dependence without adequate commitment | Care | open interfaces、無料・低コスト continuity、data portability、service discontinuation の規範、user competence の育成 |

---

## 8. Discussion / 限界

### 8.1 Cultural and Individual Variation

- WEIRD 基準のみでは不十分
- 宗教・文化によって「appropriate relationship」の規範が異なる
- Age, disability, neurodivergence 等の差異

### 8.2 Research Gaps

- Longitudinal studies 欠如（relationship は時間で発展する）
- Ecological validity（lab vs real world）
- 心理的メカニズム（attachment, dependency）の計測

### 8.3 政策含意

- 規制は「relationship appropriateness」を評価する仕組みを必要とする
- Consent, exit rights, transparency obligations

---

## 9. CS 222 での位置づけ

### Lecture 15: Human-AI Relationships のアンカー論文

Lecture 15 で Park 氏は:
- Generative agents / assistants が**人々との長期的関係**を形成する時代の倫理
- **関係性の設計（relational design）** を中心テーマに
- Morris & Brubaker の Generative Ghosts 論文（15_1）と対で扱う:
  - **15_1**: 故人との関係 — extreme case
  - **15_2 (本論文)**: 生者との継続的関係 — general case

### Park 氏の Generative Agents 論文との接続

- Smallville エージェントは fictional characters で、ユーザーとの関係は observer のみ
- 一方、AI assistants は **ユーザーとの直接・長期関係** を持つ
- 本論文の framework は CS 222 後半で**シミュレーション主体と interactant 主体の境界**を問う道具になる

### Rittel (02_1) の wicked problem との類比

- 関係の appropriateness は wicked problem
- 「正しい答え」は存在しない、ステークホルダーごとに評価軸が異なる
- 論文の framework は Rittel の第二世代（argumentative process）的アプローチを取る

---

## 10. 主要引用

### 基盤理論

- **Baier 1986** "Trust and Antitrust" (Ethics)
- **Dennett 1987** *The Intentional Stance*
- **Nissenbaum 2004** "Privacy as Contextual Integrity"
- **Turkle 2007, 2011** *Alone Together*
- **Coons & Weber 2014** *Manipulation: Theory and Practice*
- **Noggle 2022** Stanford Encyclopedia "The Ethics of Manipulation"

### Human-AI Interaction

- **Shanahan 2024** "Talking About Large Language Models"
- **Gabriel et al. 2024** *Ethics of Advanced AI Assistants* (DeepMind)
- **Sharkey & Sharkey 2010** "The crying shame of robot nannies"
- **Shevlin 2024** AI companions の哲学
- **Kolt 2024** AI agents のガバナンス

### 事例研究

- **Roose 2023** NYT: Bing chatbot 事件
- **Brooks 2023** MIT Tech Review: Replika Luka 事件
- **Franklin et al. 2022** echo chamber
- **Ajunwa, Crawford, Schultz 2017** labor and algorithmic harms
- **Yeung 2017** hypernudge

### 関連 AI ethics

- **Birhane 2021, 2022** algorithmic injustice / relational ethics
- **Lazar 2022** "Algorithmic Domination"
- **Kaziunas, Ackerman, Lindtner 2017** chronic care + AI
- **Bender et al. 2021** Stochastic Parrots

---

## 11. 要点

1. **AI assistant との関係**は anthropomorphism、agency 増大、generality、deep dependence の 4 特性を持つ
2. 人間関係の ethical 判断枠組み（nature/purpose, expectations, trust, care）を類比的に適用
3. **4 つの主要 risk**: direct harm、growth の阻害、emotional exploitation、material dependence without commitment
4. Replika Luka 事件（2023 年）、Bing chatbot Kevin Roose 事件（2023 年）、Character.AI などを具体例として分析
5. **Manipulation の境界**: hypernudge、anthropomorphic persuasion で rational persuasion との区別が曖昧化
6. **Generality → hybrid norms**: 単一 AI が多 context を兼ねると規範の衝突を生む
7. Developer への推奨: evidence-based guidelines、red-teaming for emotional harm、open interfaces、data portability、service continuity 規範
8. CS 222 Lecture 15 で **Generative Ghosts (15_1)** と対で、人-AI 関係設計の倫理的基盤を提供。Rittel 的な wicked problem framework を継承
