# 15_1 — Generative Ghosts: Anticipating Benefits and Risks of AI Afterlives

## 書誌情報

- 著者: **Meredith Ringel Morris**（Google DeepMind, USA）、**Jed R. Brubaker**（University of Colorado Boulder, USA）
- arXiv: 2402.01662v4 [cs.CY], 2024年12月12日改訂
- ACM manuscript submission（CHI/CSCW 系列を想定）
- CS 222 Lecture 10（Cognitive Architectures and Death）および Lecture 14（AI-Afterlives の社会的含意）で引用
- Keywords: AI, AI agents, Generative AI, HCI, digital afterlife, digital legacy, post-mortem AI, post-mortem data management, end-of-life planning, death, griefbots

---

## 1. 研究問題

生成 AI 能力（GPT-4, PaLM 2, Llama 2）の急速な進化で、**特定の人物をモデルにした AI エージェント**の生成が可能になりつつある。特に死者を表現する AI（startups は既に東アジアで急増）が出現。

研究問題:

> How might advances in AI change personal and cultural practices around death and dying?

本論文が貢献するもの:
1. **Generative ghosts** という新現象を identify・characterize
2. **Design space**（7 dimensions）を提示
3. Benefits/risks の analytic framework
4. 研究アジェンダ

---

## 2. Generative Ghost の定義

> AI agents that represent a deceased person, capable of **generating novel content** rather than merely parroting content produced by their creator while living.

### Griefbot との違い

- Griefbot: 故人の素材（メール、日記）を学習し「会話」するチャットボット（Fredbot、Roman bot）。多くは regurgitation のみ
- Generative ghost: **novel content 生成**、**evolution**（時間進化）、**agentic**（tool use、経済参加）の能力を含む
- Generative clone（生前のAI複製）が死後に generative ghost に移行する可能性も

### 既存の例

- **Fredbot** (Ray Kurzweil): 亡父の手紙を学習、親族にのみ公開
- **Roman bot** (Eugenia Kuyda): 事故死した親友のテキストを学習、後に Replika へ発展
- The Beatles "Now and Then" (2023): AI で故 John Lennon の歌声を処理
- **Laurie Anderson x Lou Reed bot** (Adelaide University, 2024): 故恋人の chatbot
- **韓国 Jang Ji-Sun**: 亡き娘 Na-yeon の VR 再現 (1900万再生)
- **中国**: 数千人顧客規模の digital immortality 産業 (NPR 2024)
- **Character.AI, Re;memory, HereAfter, Project December, MIT Augmented Eternity**

---

## 3. Design Space（7 dimensions）

### 3.1 Provenance（出自）

| タイプ | 説明 |
|--------|------|
| **First-party** | 本人が生前作成（遺言・legacy planning） |
| **Third-party (authorized)** | 遺族が本人の consent 下で作成 |
| **Third-party (unauthorized)** | 歴史人物・公人を第三者が勝手に作成（Character.AI の Shakespeare など） |

### 3.2 Deployment Timeline（デプロイ時期）

- Pre-mortem: 生前から generative clone として運用、死後 ghost へ遷移
- Post-mortem: 死後に作成

### 3.3 Anthropomorphism Paradigm（擬人化パラダイム）

- **Reincarnation**（転生）: 「私は [故人] です」と語る — 一人称、現在形、本人の名前
- **Representation**（代理）: 「私は [故人] の AI 表象です」と語る — 三人称的、"Fredbot" のような別名

設計選択:
- first-person vs third-person pronouns
- present vs past tense
- 故人の名前 vs 別称
- 「魂がある」「生きている」を許すか

### 3.4 Multiplicity（複数性）

- Single ghost
- Multiple ghosts: 異なる audience 向けに context collapse を回避（友人用、職場用、子供用）
- 意図しない重複: 複数の第三者が独立に作成

### 3.5 Cutoff Date（更新期限）

- Static: 死亡時点の性格・知識で固定
- Evolving: 新情報（世界ニュース、対話相手の結婚）を反映。仮想的な「加齢」も
- LivesOn (2013) が rudimentary な先駆

### 3.6 Embodiment（具現化）

- Virtual only（chatbot）
- Physical: robotics
- Mixed reality avatar

### 3.7 Representee（被表現者）

- Human
- Non-human（ペット、サービス動物）

---

## 4. Benefits（潜在的便益）

### 4.1 Representee への便益

- **Digital afterlife** への安心感
- 人生物語・価値観を次世代に伝える（Jamison-Powell 2016, Gulotta 2017）
- 子孫の重要イベント（結婚、出産）にコメント
- **経済参加**: 著書を生成し続ける作家など、生命保険の代替・補完

### 4.2 遺族への便益

- **Continuing bonds** (Klass, Silverman, Nickman 1996)
- 死別の "accommodation" (Neimeyer 等) 支援
- 一方通行の「墓前での会話」が bidirectional になる
- 実用情報（パスワード、レシピ、家のメンテナンス）の継承

### 4.3 社会への便益

- **Cultural preservation**: 消滅言語、宗教伝統、ホロコースト生存者の証言
- 歴史研究・アンスロポロジー: Talmudic scholar と対話、Pompeii 市民のamalgamation
- **Khan Academy** 等の教育応用（歴史人物と対話）

---

## 5. Risks（潜在的リスク）— 4 カテゴリ

### 5.1 Mental Health Risks

- **Delayed accommodation**: 喪失の統合が遅れる
- **Complicated grief / Prolonged Grief Disorder**: 過度依存。Replika 型の addictive parasocial
- **Information overload / Choice paralysis**: ghost の助言に意思決定を委ね過ぎ
- **Anthropomorphism**: ghost を故人本人と誤認
- **Deification**: 超自然的信仰の発生、cult 化
- **Second deaths / Second losses**: サービス終了・hack・規制で ghost が失われる二次喪失

> The body of a hanged man is in equilibrium when it finally stops swinging, but nobody is going to insist that the man is all right. — Schelling paraphrase（均衡≠望ましい、の類比）

**文化依存**: Chinese 祖先供養、Māori の kaitiaki (guardian) 観などでは Western の「complicated grief」観念と異なる判断になる。

### 5.2 Reputational Risks

- **Privacy risks**: 故人が秘匿したかった情報が漏洩（隠していた性的指向、不倫、犯罪）
- **Context-dependent**: 配偶者にはよいが職場友人には不適な情報
- **Hallucination risks**: 誤情報生成で故人の名誉を傷つける
- **Fidelity risks**: 正確情報でも persistence が問題（"rose-colored glasses" の喪失）。digital media の forgetting 不全（Mayer-Schönberger 2009）

### 5.3 Security Risks

- **Post-mortem identity theft**: ghost から passwords, financial info を抽出（jailbreak prompts）
- **Hijacking**: ransomware-style attack、prompt injection、puppetry（ghost になりすまし）
- **Malicious ghost**: 虐待配偶者が死後も家族を emotional abuse する設計など

### 5.4 Socio-cultural Risks

- **Labor market 影響**: 死者が経済参加することで賃金・雇用に影響
- **Cultural stagnation**: 新しい創造的発想が出ない（ghosts は過去の value に anchor）
- **Interpersonal relationships の変容**: smartphone/social media 時代に匹敵する disruption
- **Religion**: 既存宗教への挑戦、新宗教の発生、major religions の公式対応（fatwa 等）

---

## 6. Discussion / 政策提言

### 6.1 Interfaces to Mitigate Risk

- Reincarnation vs representation の選択は重要（reincarnation は unsettling になりうる）
- Dark patterns の回避（addictive push notifications）
- Ghost が **usage patterns を monitor** して mental health referral を提案する meta-design
- Transparency（故人の AI 表象であることを watermark/fingerprint で示す）
  - 2024 Polish Radio が故 Wisława Szymborska の AI 生成 interview を未告知で放送 (NYT 2024)
  - 中国 Super Brain: 高齢親族に死を告げずに故人からの「ビデオ通話」を提供

### 6.2 Policies

- **Third-party ghosts の consent 問題**: 2023 SAG-AFTRA strike で言及
- George Carlin の未認可 AI 特集 "I'm Glad I'm Dead" (2024年1月) — 娘が distress
- Policy 提案:
  - First-party creation を優先的に許可、third-party は条件付き
  - 公人と私人で別の規範
  - Representee または遺族による termination 権
  - **Emergency override** の技術的実装義務
  - Data governance（discontinued products 時の扱い）

### 6.3 Societal Impacts

- 不可予測性の承認（religious movements 誕生等）
- Digital divide: 経済力による ghost 差
- Compute cost / 環境コスト at scale

### 6.4 Future Work

- Prototype studies with stakeholders（representee, bereaved, clergy, legal experts）
- **WEIRD 以外**のサンプルの重要性（中国、韓国、日本、先住民文化）
- 参加型デザイン

---

## 7. CS 222 での位置づけ

### Lecture 10 での引用

Park 氏は Lecture 10（Cognitive Architectures and the Soul? / Death）で、generative agent が「death」を意味的に扱えるかを問う。Morris & Brubaker の論文は、生成エージェント技術の**最も親密で高リスクな応用**としての「故人の複製」を提示。

### Lecture 14 での引用

Lecture 14 では**社会シミュレーションと AI の societal implications**を扱う。Generative ghosts は:

- Individual scale から societal scale へ（Lecture 14 の中心概念）
- 経済・宗教・対人関係の構造変化
- **シミュレーションの倫理的限界**（誰をシミュレートしてよいか？）

### Design space アプローチの原型

CS 222 全体を通して、**Design space 分析**は Park 氏の好むメソッド（Social Simulacra、Generative Agents でも採用）。Morris & Brubaker の 7 dimensions フレームワークはその典型。

### Park 氏の自著との関係

- Smallville のエージェント（Park 2023）は fictional character
- Generative ghosts は **実在の個人**の agent
- 両者の design choices の違い（identity preservation、consent、evolution）が Lecture 14 で議論される

---

## 8. 主要引用

### 歴史的・理論的基盤

- **Brubaker 2015** *Death, Identity, and the Social Network* (PhD 論文)
- **DeGroot 2018** "Transcorporeal Communication" (grief と会話)
- **Stroebe & Schut 1999** "Dual process model of coping with bereavement"
- **Klass, Silverman, Nickman 1996** "Continuing bonds"
- **Massimi & Charise 2009** "Dying, Death, and Mortality" — thanatosensitive design

### Generative AI / HCI

- **GPT-4, PaLM 2, Llama 2, Gemini** の技術文献
- **Park et al. 2023** *Generative Agents* (UIST)
- **Gabriel et al. 2024** *The Ethics of Advanced AI Assistants* (DeepMind)
- **Abercrombie et al. 2023** "Mirages: On Anthropomorphism in Dialogue Systems"
- **Constitutional AI** (Bai et al. 2022)

### Griefbot / AI Afterlives 先行研究

- **Hollanek & Nowaczyk-Basińska 2024** "Griefbots, Deadbots, Postmortem Avatars" (Philosophy & Technology): data donors / data recipients / service interactants の三者モデル
- **Fagone 2021** "The Jessica Simulation" (San Francisco Chronicle)
- **Feng 2024** NPR の中国 digital immortality 報道
- **Kurzweil 2023** *Artificial: A Love Story* (graphic novel)

---

## 9. 要点

1. **Generative ghost** = 故人を表現し、novel content を生成でき、agentic 能力を持つ AI agent。2020 年代半ばに急速に産業化
2. **7 dimensions の design space**: provenance、deployment timeline、anthropomorphism paradigm、multiplicity、cutoff date、embodiment、representee type
3. **Benefits**: 遺族の grief 支援、文化保存、歴史教育、経済的継承
4. **Risks 4 category**: mental health（addiction, complicated grief, second death）、reputational（privacy, hallucination）、security（identity theft, hijacking, malicious ghosts）、socio-cultural（labor, religion, relationship patterns）
5. First-party vs third-party、reincarnation vs representation などの設計選択が benefit/risk tradeoff を規定
6. 東アジア（中国・韓国）での採用が先行、Western thanatology フレームでは捉えきれない文化依存性
7. **Design space analytical framework** は CS 222 Park 氏の方法論と強く共鳴
8. CS 222 Lecture 10 と Lecture 14 で生成エージェントの**最も親密で倫理的に高リスクな応用**として引用される
