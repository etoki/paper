# 05_1 — Generative Agents: Interactive Simulacra of Human Behavior

## 書誌情報

- 著者: **Joon Sung Park**, Joseph C. O'Brien, Carrie J. Cai, Meredith Ringel Morris, Percy Liang, **Michael S. Bernstein**
- 所属: Stanford University / Google Research / Google DeepMind
- 掲載: **UIST 2023**（36th Annual ACM Symposium on User Interface Software and Technology）
  - 2023年10月29日–11月1日、San Francisco
- DOI: https://doi.org/10.1145/3586183.3606763
- arXiv: 2304.03442v2
- **22 pages** の長編論文（UIST では例外的）
- デモ: https://reverie.herokuapp.com/UIST_Demo/
- コード: https://github.com/joonspk-research/generative_agents
- **CS 222 の中心論文**。Park 氏の代表作、Social Simulacra (UIST 2022, 03_1) の直接的発展形

---

## 1. 研究問題

> How might we craft an interactive artificial society that reflects believable human behavior?

### 背景

- **believable agent** という概念は 40 年以上の歴史（The Sims, Disney animators, Laird & van Lent 2001）
- 従来アプローチの限界:
  - **Rule-based (FSM, behavior trees)**: open world で網羅できず、新手続きも定義不能
  - **Reinforcement learning (AlphaStar, OpenAI Five)**: 報酬が明確な敵対的ゲームに限定
  - **Cognitive architectures (SOAR, ICARUS)**: manually crafted procedural knowledge に制約
  - **LLMs alone**: single-point prediction（Argyle 2023, Park 2022 Social Simulacra）は可能だが、**long-term coherence**を持つ general agent は不可
- 必要なもの: **長期記憶 + 反省 + 計画** を持つ包括的アーキテクチャ

### 研究の問い

- LLM を適切なアーキテクチャで augment すれば、個人・集団の believable behavior を長期的に生成できるか？
- どの構成要素（memory, reflection, planning）が critical か？

---

## 2. 貢献（4つ）

1. **Generative agents**: 動的に変化する経験と環境に応じて条件付けられる、信頼できる人間行動の simulacra
2. **新アーキテクチャ**: 記憶・検索・反省・計画が相互作用、LLM の powerful prompting を補完
3. **2つの評価**: controlled evaluation（interviews with agents）+ end-to-end evaluation（2日間の自律シミュレーション）
4. **倫理的・社会的リスクの議論**: parasocial relationship, deepfakes, displacement of stakeholders

---

## 3. Smallville: Sandbox 環境

### 3.1 設定

- **25 unique agents** が住む The Sims 風サンドボックス
- 各エージェントは 1 段落の自然言語 description（例: John Lin の description が 10行以上）
- Phaser ゲームエンジン、sprite-based avatar

### 3.2 Environment as Tree

- 環境 = tree data structure（root: 世界、中間: 地域、leaf: オブジェクト）
- "stove" が "kitchen" の子 → "there is a stove in the kitchen" として LLM に渡す
- Agents は subgraph を保持（知覚した範囲のみ）。非 omniscient

### 3.3 Inter-Agent Communication

- 各 time step で natural language statement を出力（例: "Isabella Rodriguez is writing in her journal"）
- 絵文字表示 + 自然言語アクセス
- Agents 間は full natural language で会話（例: 選挙について Isabella と Tom が会話）

### 3.4 User Controls

- User が agent の persona を指定して会話（"you are a reporter"）
- User が agent の "inner voice" になって命令（例: "You are going to run against Sam in the upcoming election" → John が立候補を決定）
- Environment の object status を自然言語で書き換え（"stove is burning" → Isabella が火を消す）

---

## 4. Emergent Social Behaviors（観察例）

### 4.1 Information Diffusion
- Sam の市長選への立候補情報が、Sam→Tom→John と拡散
- エージェントが会話の際に自発的に情報共有

### 4.2 Relationship Memory
- Sam が Latoya と Johnson Park で出会い、後日 "Hi, Latoya. How is your project going?" と尋ねる

### 4.3 Coordination（有名な Valentine's Day Party 例）
- **単一の seed**: Isabella に "Valentine's Day パーティーを開きたい" と設定
- **発生したこと**:
  - Isabella がカフェで友人を招待
  - Maria が Klaus（片思いの相手）を誘う
  - 2月14日に5人が Hobbs Cafe に集合
- すべて自発的で pre-programmed ではない

---

## 5. Architecture（本論文の核）

### 5.1 Memory Stream

> A comprehensive record of the agent's experience.

各 memory object は3要素:
- 自然言語 description
- creation timestamp
- most recent access timestamp

**Observation** が最小単位。例:
1. Isabella Rodriguez is setting out the pastries
2. Maria Lopez is studying for a Chemistry test while drinking coffee
3. Isabella and Maria are conversing about planning a Valentine's day party
4. The refrigerator is empty

### 5.2 Retrieval の 3 コンポーネント

| 要素 | 計算 |
|------|------|
| **Recency** | exponential decay（decay factor = 0.995 per sandbox game hour） |
| **Importance** | LLM に 1–10 で聞く（"brushing teeth"=2, "asking crush out"=8） |
| **Relevance** | query memory embedding との cosine similarity |

**Retrieval score** = α_recency × Recency + α_importance × Importance + α_relevance × Relevance
- 実装では α = 1 すべて
- min-max scaling で [0,1] に正規化
- Top-ranked memories が LLM の context window に入る

### 5.3 Reflection

**課題**: 生 observation だけでは generalization 不可
- 例: Klaus に「1時間過ごすなら誰と？」→ 最頻出対話相手の Wolfgang を選ぶ（誤った選択、Wolfgang とは浅い関係）
- **正解の根拠**: Klaus は研究に情熱的、Maria も別分野の研究に情熱的 → 共通点を反映した選択

**Approach**: Reflection は memory stream に書き込まれる higher-level thoughts
- 最近 observations の importance 合計が **閾値 150** を超えたら生成（1日 2–3 回）
- 手順:
  1. 最近 100 records を LLM に渡し、"What are 3 most salient high-level questions?"
  2. 各質問を query として retrieval
  3. retrieved memories を "What 5 high-level insights can you infer?" に与える
  4. 引用元メモリ番号 (because of 1, 5, 3) と共に insight を保存
- **再帰的**: reflection は他の reflection も参照可能 → **reflection tree**
  - Leaf: base observations
  - Non-leaf: 抽象度高い thoughts

例: "Klaus Mueller is dedicated to his research on gentrification (because of 1, 2, 8, 15)"

### 5.4 Planning

**課題**: LLM 単体だと 12pm lunch の直後 12:30pm, 1pm にもまた lunch を食べてしまう（moment-wise plausible だが long-term incoherent）

**Approach**: 階層的な plan
1. **Top-down**: 5–8 chunks の日程大枠
   - Input: name, traits, agent's summary description, previous day summary
   - Output: "1) wake up 8:00am, 2) go to Oak Hill College 10:00am, 5) work on composition 1–5pm, 6) dinner 5:30pm..."
2. **Hour-long chunks**: "1:00pm: brainstorm ideas..., 4:00pm: take break and recharge..."
3. **5–15 min chunks**: "4:00pm: grab a light snack..., 4:05pm: take a short walk..."

Plans も memory stream に保存され、retrieval で活用される。

### 5.5 Reacting and Updating Plans

- 各 time step で perception → observation を memory stream に追加
- LLM に "Should the agent react?" を問う
- 例: John が Eddy の散歩を目撃 → memory から context summary 生成 → "John could ask about music composition"
- Reaction があれば plan を該当時刻から再生成

### 5.6 Dialogue

- Agent の summary + relevant memory + intended reaction から発話生成
- 例: "Hey Eddy, how's the music composition project for your class coming along?"
- Eddy 側も同様に retrieve & respond
- どちらかが終了を決めるまで続く

### 5.7 Environment Grounding

- Agent の environment tree を flatten して LLM に渡す
- "Which area should Eddy Lin go to?" → "The Lin family's house: garden" など leaf まで再帰的に決定
- Object の状態変化も LLM に問う（"espresso making" → 'coffee machine: off → brewing'）

---

## 6. Implementation

- **GPT-3.5-turbo** (ChatGPT) を使用
- GPT-4 は invitation-only だったため未使用
- 25 agents × 2 日間シミュレーション: **thousands of dollars in token credits**, **multiple days** to complete
- Sandbox server が JSON でエージェント状態を保持

---

## 7. Controlled Evaluation

### 7.1 設定

- 5カテゴリの質問で agent を "interview":
  1. **Self-knowledge**: "Give an introduction of yourself"
  2. **Memory**: "Who is [name]?"
  3. **Plans**: "What will you be doing at 10am tomorrow?"
  4. **Reactions**: "Your breakfast is burning! What would you do?"
  5. **Reflections**: "If you were to spend time with one person you met recently, who would it be?"
- 各カテゴリ 5 質問 × 25 agents
- **2日間のシミュレーション後**に interview

### 7.2 比較条件（4 ablations + 1 human）

| 条件 | Memory | Reflection | Planning |
|------|--------|-----------|----------|
| Full architecture | ✓ | ✓ | ✓ |
| No reflection | ✓ | ✗ | ✓ |
| No reflection, no planning | ✓ | ✗ | ✗ |
| No memory (observation/plan/reflection) | ✗ | ✗ | ✗ |
| Human crowdworker | — | — | — |

- No-memory 条件 = **prior LLM agent の state of art**（Aher 2023, Argyle 2023, Park 2022 social simulacra）
- Crowdworker baseline: 同じ agent の replay を見せて回答作成

### 7.3 評価者

- **100 participants from Prolific**、30分、$15/hour
- Within-subjects design、各質問への 5 条件の回答をランキング
- TrueSkill rating で interval scale に変換

### 7.4 主要結果（Figure 8）

| 条件 | TrueSkill μ | σ |
|------|------------|---|
| **Full architecture** | **29.89** | 0.72 |
| No reflection | 26.88 | 0.69 |
| No reflection, no planning | 25.64 | 0.68 |
| Crowdworker | 22.95 | 0.69 |
| No memory (prior SOTA) | 21.21 | 0.70 |

**Cohen's d**（Full vs Prior SOTA）: **d = 8.16**（**8 標準偏差**の効果）

**統計検定**:
- Kruskal-Wallis: H(4) = 150.29, **p < 0.001**
- Dunn post-hoc + Holm-Bonferroni: Crowdworker と No-memory ペア以外、すべての pairwise 差が **p < 0.001**

### 7.5 定性的発見

**Generative agents remember, with embellishments**:
- Abigail が自己紹介: "Hi, I'm Abigail. I'm 25 years old and passionate about creative projects..."
- Maria が Rajiv を「動画・アートプロジェクトに熱心」と正しく回想
- **失敗**: Rajiv が選挙について "I haven't been following" と答えるが、実は Sam の立候補情報を受取済み
- **Hallucination 例**: Isabella が Sam の立候補について「明日発表する」と embellish（実際には話し合いなし）
- Yuriko が Adam Smith（架空の近所）を「Wealth of Nations の著者」と世界知識で embellish

**Reflection is required for synthesis**:
- Maria が Wolfgang の誕生日プレゼント: no-reflection → "don't know what he likes"
- with reflection → "Since he's interested in mathematical music composition, I could get him something related"

---

## 8. End-to-End Evaluation（2日間の自律シミュレーション）

### 8.1 測定 3 指標

**Information diffusion**:
- Sam の市長選立候補: 4% → 32%（1→8人が知る）
- Isabella の Valentine's Day Party: 4% → 52%（1→13人が知る）
- Hallucination 率: 1.3%（6/453 responses）

**Relationship formation**:
- Network density: 0.167 → **0.740**
- 453 agent responses のうち 1.3% が hallucinated relationships

**Coordination (Valentine's Day Party)**:
- Isabella が 12 人を招待（前日に装飾、Maria の協力も得る）
- **5/12 人が 2月14日 5pm に Hobbs Cafe に現れた**
- 欠席 7名の内訳:
  - 3 名: 競合する予定（Rajiv: "I'm focusing on my upcoming show"）
  - 4 名: 関心はあるが当日 plan にならず

### 8.2 Boundaries and Errors（3つ）

1. **Memory retrieval と location 判断の困難**:
   - Bar が lunch 場所として過度に選ばれる（本来は evening の集まり場所）
   - メモリが増えると retrieval 精度が下がる

2. **Physical norm の誤解**:
   - Dorm bathroom が「1人用」なのに複数人同時入室
   - Stores が 5pm 閉店なのに入店試行
   - 解決案: location status に "one-person bathroom" のような記述を追加

3. **Instruction tuning の影響**:
   - Agents が overly polite / cooperative
   - Mei が John に "It was good talking to you as always" と formal な挨拶
   - Isabella が他 agents の提案を断れず、特性と異なる interests（Shakespearean reading 会、networking イベント）を引き受ける

---

## 9. Applications

### 9.1 Social Simulacra 系
- 03_1 の発展: stateless persona → **persistent agents with memory**
- オンラインフォーラム、VR metaverse、社会ロボットへの応用可能性

### 9.2 Human-centered Design
- Weiser の ubiquitous computing vignette "Sal" の生成型ユーザーモデル
- GOMS / KLM 的な認知モデルの現代版

### 9.3 Future Work
- Retrieval module の fine-tuning
- Cost 削減（並列化、specialized models）
- 長期シミュレーション evaluation
- Robustness（prompt hacking, memory hacking, hallucination）

---

## 10. Ethics and Societal Impact（4つのリスク）

### 10.1 Parasocial Relationships
- Users が agent を擬人化、emotional attachment
- **原則1**: agent は computational entity であることを明示
- **原則2**: value-aligned（例: 愛の告白に対する相互返答を避ける）

### 10.2 Error Propagation
- ubiquitous computing 応用で誤推論が user goal に反映
- HCI best practices (Amershi 2019 Guidelines for Human-AI Interaction)

### 10.3 Deepfakes / Misinformation / Persuasion
- **Audit log の保持**（入力+生成物）
- 認可された設計者のみのアクセス
- 自前でアーキテクチャを構築すると約1年かかる → 参入障壁

### 10.4 Over-reliance
- Human stakeholders を置き換えてはならない
- Early-stage prototype のみに使用すべき

---

## 11. 先行研究との関係（Related Work 詳細）

### 11.1 Believable Agents の伝統
- Bates (1994) CACM "The Role of Emotion in Believable Agents" ← CS 222 07_1
- Thomas & Johnston *The Illusion of Life* (Disney animators)
- Laird & van Lent (2001) game worlds as testbeds

### 11.2 Cognitive Architectures
- Newell (1990) UTCs ← 04_1
- SOAR (Laird et al. 1987) ← 04_2
- ICARUS (Langley), ACT-R (Anderson)
- Quakebot-SOAR, TacAir-SOAR
- **批判**: manually crafted procedural knowledge に制約

### 11.3 LLM Human Behavior Simulation
- Social Simulacra (Park et al. 2022) ← 03_1
- Argyle et al. (2023) silicon sampling ← 03_2
- Aher, Arriaga, Kalai (2023) - 既存実験の複製
- Horton (2023) Homo Silicus - 経済実験
- **批判**: first-order templates（few-shot, CoT）では **long-term coherence** が不可

### 11.4 本論文の超越点
> This paper extends these ideas to craft an agent architecture that handles retrieval where past experience is dynamically updated at each time step and mixed with agents' current context and plans.

---

## 12. CS 222 での位置づけ

### Lecture 05: アンカー論文
- Park 氏自身の講義での中心論文。Module 全体の構成要素を示す
- 05_2 CoALA との対照で読む
  - CoALA は Generative Agents の architectural taxonomy を提供
  - 本論文は CoALA の "case study" として参照

### Lecture 03-04 からの発展系譜
- **03_1 Social Simulacra**: 集団レベル、stateless persona
- **03_2 Argyle**: 集合の silicon sample、aggregate only
- **04_1/04_2 SOAR**: symbolic cognitive architecture
- **→ 05_1 Generative Agents**: 上記を融合した persistent individual agents

### Lecture 06 以降への接続
- **06_1 Chang et al.**: 社会ネットワーク生成で Park 2023 を baseline 引用
- **06_2 Roleplay-doh**: generative agents の domain-specialization（患者 persona）
- **07_1 Bates**: emotion を加える future work の源流
- **07_2 Hewitt/Ashokkumar**: 実験結果予測で Park 2023 を引用

### Park 氏の研究系譜
- **Social Simulacra (UIST 2022)** → **Generative Agents (UIST 2023)** → **Agent Bank / 1000 Generative Agents (2024)** → **CS 222 (2024–)**

---

## 13. 主要引用

### 論文が引用
- [10] Bates (1994) CACM "The Role of Emotion in Believable Agents" ← 07_1
- [80] Park et al. (2022) Social Simulacra (UIST) ← 03_1
- [92] Argyle et al. (2023) *Political Analysis* ← 03_2
- [76] Newell (1990) UTCs ← 04_1
- [25, 64] Langley ICARUS, [60, 81] Quakebot/TacAir-SOAR ← 04_2 系
- [59] Laird & van Lent (2001) "Human-level AI's Killer Application: Interactive Computer Games"
- [77] ChatGPT / OpenAI
- [12] Binz & Schulz "Using cognitive psychology to understand GPT-3"
- [39] Aher, Arriaga, Kalai (2023)
- [46] Horton (2023) Homo Silicus
- [100] Wei et al. chain-of-thought
- [42] TrueSkill (Herbrich)
- [29] Elo chess rating
- [56] Kruskal-Wallis, [98] Dunn test, [45] Holm-Bonferroni
- [96] Thomas & Johnston *Illusion of Life*

### 本論文を引用する後続
- Sumers et al. (2024) CoALA ← 05_2（本論文を case study に使用）
- Chang et al. (2024) "LLMs generate social networks" ← 06_1
- Louie et al. (2024) Roleplay-doh ← 06_2
- Hewitt et al. (2024) "Predicting Social Science Experiments" ← 07_2
- CS 222 全体

---

## 14. 批判・未解決問題

### 技術的
- 高コスト（thousands of dollars / 2日間）
- スケーリング問題（現在 25 agents）
- Hallucination, embellishment の制御
- Instruction tuning バイアス（cooperative/polite）

### 方法論的
- Crowdworker baseline の妥当性（maximal human performance ではない）
- 2日間は短い
- Robustness testing（prompt/memory hacking）未実施

### 理論的
- Bias の継承（underlying LLM のバイアスを直接反映）
- Marginalized population の believability への懸念
- Value alignment の未解決課題

---

## 15. 要点

1. **Generative Agent = LLM + 3 architectural components**: Memory stream, Reflection, Planning
2. **Memory retrieval**: Recency (decay 0.995) + Importance (LLM 1-10) + Relevance (cosine sim) で weighted combination
3. **Reflection**: 重要度合計 > 150 で trigger、3高次質問→関連記憶→5 insights を抽出し再帰 tree 構築
4. **Planning**: Top-down recursive decomposition（日→時間→5-15分 chunk）
5. **25 agents × 2 days** の Smallville シミュレーションで、Valentine's Day Party の自発的組織化など **emergent social behaviors** を観察
6. **Controlled evaluation (N=100)**: Full > no-reflection > no-reflection/plan > crowdworker > no-memory、**Cohen's d = 8.16**
7. **End-to-end**: 情報拡散 4%→52%、ネットワーク密度 0.167→0.740、5/12 人が party に参加
8. **失敗モード**: memory retrieval の location 判断低下、physical norm 誤解、instruction-tuning の formality 過多
9. **Ethical guardrails**: parasocial 関係防止、audit log、over-reliance 警告
10. **Park 氏の研究系譜の中心**: Social Simulacra (03_1) の発展形、CoALA (05_2) の case study、CS 222 全体のアンカー
