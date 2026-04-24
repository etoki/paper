# Lecture 10: Generative Agent-Based Models (GABM)

- 講師: Joon Sung Park
- ソース: `docs/stanford univ AI Agents and Simulations/10 Generative Agent-Based Models.pdf`
- 位置づけ: 個人モデル同士を相互作用させる枠組み=**GABM**を提示。後半はエージェントバンクの倫理論点。

---

## 0. 連絡事項

- **AgentBank-CS222** を翌週早々にクラスに公開（redactしたい学生は申告）
- **ゲスト講師**（次回以降）:
  - **Meredith Ringel Morris**（Google DeepMind, Director for Human-AI Interaction Research）
  - **Serina Chang**（UC Berkeley EECS, Assistant Professor）

### 本日の2パート
1. **Generative Agent-Based Models (GABM)**
2. **Ethics and considerations for agent banks**

---

## 1. Pt. 1 — Generative Agent-Based Models

### 前回からの接続

> Models of individuals predict the behavior of a particular person. This opens up genuinely new opportunities.

個人モデルができれば、彼らを**相互作用**させられる。それが GABM。

### ABM の起源（再掲）

> Agent-based models (ABM) studies the **interaction between individual agents**.

- Schelling (1971) "Dynamic models of segregation" *J. Math. Sociol.* 1, 143-186
- Schelling (1978) *Micromotives and Macrobehavior*

### Generative Agents は個人モデル同士の相互作用

Park et al. (UIST 2023) の Smallville も、個人エージェント間の**相互作用**を扱う点で ABM の系譜。

### Q: ここで見える相互作用のタイプは？

授業の問いかけ。協力、対立、情報伝播、規範の形成などを想起させる。

---

## 2. Class Activity: Simulated Voting (Demo)

デモ形式:
1. 学生が自分の**ペルソナ**を記述 → それでエージェント生成
2. Park氏が学生エージェントを同じ「部屋」に配置
3. エージェントたちが**誰をリーダーに選ぶか**を議論・決定

→ GABMの最小例として「投票/選抜」シナリオを実演。

---

## 3. 核心仮説: GABM は安定均衡を超えるか

### ABM の典型的結果

> ABM often resulted in **stable equilibria**. What about GABM?

Schelling分離モデルは tipping point 後に**安定した分離パターン**に落ち着く。他の古典ABMも概ね安定均衡を生む。

### GABM への期待

LLM駆動のエージェントは、固定ルールでなく**柔軟かつ文脈依存な行動**を取る。ゆえに:
- 動的・非定型の集合挙動
- 従来ABMでは見られなかった現象（例: 集団がイベントを協働企画、噂が伝播・変形しながら広がる）
- ただし安定均衡の欠如は予測困難さを意味するかもしれない

（この仮説は Lecture 11「Equilibria and Butterflies」で展開される。）

---

## 4. Pt. 2 — エージェントバンクの倫理

### Human Genome Project の類比

**International Human Genome Sequencing Consortium (2001) "Initial sequencing and analysis of the human genome" *Nature* 409(6822), 860-921** ← 補足 `10_1`

問い: **Genome Project の貢献は何だったか？**

- 個人ゲノムを集約することで「ヒトという種の基盤データ」を作った
- 医療・遺伝学研究の飛躍的進展
- 同時に所有権・倫理問題を惹起

### Genome Bank と Agent Bank の類比

- 両者とも「個人データの集積」
- 科学的基盤となる可能性
- 類似の倫理問題: 誰が所有するか、同意、二次利用、偽造・悪用

### Q: Who owns your agents?

- 自分のエージェントの所有権は誰に？
- ハリウッド脚本家ストライキ（2023-2024）を例に:
  - Kinder, M. (2024) "Hollywood writers went on strike to protect their livelihoods from generative AI. Their remarkable victory matters for all workers." Brookings, April 12
  - 「自分のアウトプットをAIに学習させる/模倣させる」権利を誰が持つか

### Generative Ghosts

**Morris & Brubaker (2024) "Generative Ghosts: Anticipating Benefits and Risks of AI Afterlives"** ← 補足 `15_1`

- 故人を模したAIエージェント（AI afterlives）
- 遺族支援・文化継承などの便益
- アイデンティティ・同意・搾取などのリスク
- エージェントバンクの所有権論議の尖鋭例

---

## 主要引用文献（Lecture 10）

### ABM の系譜
- **Schelling (1971)** "Dynamic models of segregation" *J. Math. Sociol.* 1, 143-186
- **Schelling (1978)** *Micromotives and Macrobehavior*
- **Park et al. (UIST 2023)** Generative Agents

### エージェントバンク倫理
- **International Human Genome Sequencing Consortium (2001)** *Nature* 409(6822), 860-921（補足 10_1）
- Kinder (2024) Brookings（ハリウッドストライキ）
- **Morris & Brubaker (2024)** "Generative Ghosts"（補足 15_1）

---

## 要点

1. **GABM** = Generative Agent-Based Models = 個人モデル（LLM駆動）同士の相互作用をモデリング
2. 古典ABM（Schelling分離モデル）は**安定均衡**を生むのに対し、GABM は動的・非定型の集合挙動を生み得る仮説
3. Class demo: 学生ペルソナを投入した**模擬投票/リーダー選抜**で GABM の最小例を実演
4. **Agent Bank** は **Human Genome Project** の類比で構想される — 個人データの科学的基盤
5. 倫理問題: **所有権**（ハリウッド脚本家ストライキ）、**生成的ゴースト**（故人AI, Morris & Brubaker 2024）
6. Lecture 11「Equilibria and Butterflies」で GABM と均衡・カオスの議論に接続
