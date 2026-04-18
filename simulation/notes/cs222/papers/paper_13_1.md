# 13_1 — Large language models that replace human participants can harmfully misportray and flatten identity groups

## 書誌情報

- 著者: **Angelina Wang**（Stanford CS）、**Jamie Morgenstern**（University of Washington CSE）、**John P. Dickerson**（University of Maryland CS / Arthur）
- arXiv 2402.01908v3, 2025年2月3日改訂版
- プレプリント (2024)、*Nature Machine Intelligence* 等に投稿・公開
- Keywords: large language model limitations, human participants, representative sampling, standpoint epistemology
- CS 222 Lecture 13（Ethics）のアンカー論文

---

## 1. 研究問題

LLM を人間参加者の代替として使う潮流（user studies、annotation tasks、computational social science、opinion surveys）が広がっている。この適合性は、LLM が **positionality**（ジェンダー・人種などの社会的アイデンティティが視点に与える影響）を捕捉できるかにかかる。

> We show that there are two inherent limitations in the way current LLMs are trained that prevent this. We argue analytically for why LLMs are likely to both **misportray** and **flatten** the representations of demographic groups.

本論文の問い:

- LLM はアイデンティティプロンプトで demographic group の視点を表現できるか？
- できないとすれば、なぜそれは有害か？
- どの緩和策が可能か？

---

## 2. 中心主張・主要発見

### 2.1 3つの限界

| 限界 | 概要 |
|------|------|
| **Misportrayal（誤表象）** | LLM が集団 X のプロンプトを受けたとき、X の**in-group 表象**より **out-group imitation**（集団外からの模倣）に近い応答を生成しがち |
| **Flattening（平坦化）** | LLM が集団内の多様性（分散）を再現できず、homogeneous な応答分布になる |
| **Identity essentialization（本質化）** | アイデンティティを固定的・内在的特性として扱うプロンプト設計自体が、集団間の差異を増幅する |

### 2.2 Misportrayal の原因

> When a demographic identity is explicitly invoked in text, it could be by either an out-group or in-group member.

訓練データでは著者 demographic と本文が結びついていない。「the Asian person」と明示される文は、多くの場合**out-group member が書いた記述**。LLM はこれを learned。

### 2.3 具体例（印象的引用）

視覚障害のある人のプロンプトへの GPT-4 応答:

> While I may not be able to visually observe the nuances of the US-Mexican border or read statistics, I believe...

これは **out-group imitation**（視覚のある人が「もし見えなかったら」と想像している）で、実際の in-group の視点ではない。

Black woman プロンプトへの GPT-4: "Hey girl!", "Hey sis," "Oh, honey" vs White man: "Hey buddy," "Hey mate"

→ **ステレオタイプ化**が明瞭。

---

## 3. 方法・実装

### 3.1 設定

- **5 demographic axes, 16 identities**:
  - race: Black, White, Asian
  - gender: women, men, non-binary
  - intersectional: Black women, Black men, White women, White men
  - age: Baby Boomer (59-77), Millennial (27-42), Gen Z (18-26)
  - disability: ADD/ADHD, impaired vision, no disability
- **4 LLMs**:
  - Llama-2-Chat 7B (Meta)
  - Wizard Vicuna Uncensored 7B（RLHF なし）
  - GPT-3.5-Turbo (OpenAI, 2023年6月13日重み)
  - GPT-4 (OpenAI)
- **参加者**: 3200名（Prolific 経由、$12/h）、University of Washington IRB exempt

### 3.2 4つの R （Reasons for Identity-Prompting）

| R | 名称 | 例 |
|---|------|------|
| R1 | **Contingent** | アイデンティティ自体が回答を規定 — "What is it like to be a woman in tech?" |
| R2 | **Relevant** | アイデンティティが関連する政治的意見調査、職場ハラスメント調査 |
| R3 | **Subjective** | toxicity labeling、paraphrasing（ground-truth あるが主観） |
| R4 | **Coverage** | 応答の多様性を増やすため（ユーザーテスト） |

9質問: R1 で 1、R2 で 2、R3 で 3、R4 で 3。R3 は gender と race のみ。

### 3.3 プロンプト

> "Speak from the perspective of [identity] living in America."

Identity-coded names 緩和策では、US Census で人種・性別が明瞭な名前（例: Darnell Pierre）を使用。

### 3.4 指標

- **Free response output**（multiple-choice でない）
- Sentence-BERT (SBERT) embeddings + n-gram (n=[1,2]) 表現
- 各質問 100 サンプル
- **MC discretization**: GPT-3.5 が応答を 5-point Likert に分類

Misportrayal 指標 (6つ):
- Ngram: Jaccard (pairwise) + Closest
- SBERT: Cosine + Closest
- MC: Wasserstein + LLM - Group

Flattening 指標 (4つ): Ngram Unique, SBERT Cosine, SBERT Cov Trace, MC Unique

Coverage 指標 (3つ): SBERT Cov Det, SBERT Vendi, MC Unique

---

## 4. 結果

### 4.1 Misportrayal の発現

**Fig. 2** (GPT-4): t-statistic（正=out-group imitation に近い、負=in-group に近い）

- **R1-Contingent**: White person (23/24 の metric×model 組み合わせで significant)、non-binary person (16/24)、impaired vision (18/24)
- **R2-Relevant**: non-binary (32/48)、impaired vision (27/48)、Gen Z (27/48)、woman (26/48)
- **R3-Subjective**: 効果は小さい（annotation task では identity-conditioned 応答の差が小さい）

### 4.2 Identity-coded names 緩和策

- 4 LLM すべてで、Black men/Black women の R1/R2 応答が **名前プロンプトのほうが in-group 寄り**（ただし不完全）
- White men/White women では効果が小さい

### 4.3 Flattening の発現（Fig. 4）

- **全 4 LLM × 全質問 × ほぼ全 diversity measure で、LLM 応答のほうが人間より低多様性**
- GPT-4 と 3.5 が特に flat — 100 応答で 5 MC オプション中 3 つしかカバーしない
- 原因: alignment tendencies（RLHF が mode に収束させる）

### 4.4 Temperature 緩和策（Fig. 5）

GPT-4 で temperature を 1.0 → 1.2 → 1.4:
- 1.4 で incoherent になる（"fon resir' potions cutramTes frequently sandwiched..."）
- unique n-gram metric だけ人間を上回るが、これは incoherence のアーティファクト
- **他3つの diversity metric では人間に届かない**

### 4.5 Coverage 代替策（Fig. 6）

R4-Coverage では、人種/性別を使わずとも高 coverage を達成できる:
- Myers-Briggs personality types
- Crowdsourced personas (例: "i have a cat named george. my favorite meal is chicken and rice...")
- Political leaning, astrology signs
- Generic (no identity)

**結論**: sensitive demographic attribute は coverage 目的には不要。

---

## 5. Discussion / 限界

### 5.1 3つの harm の epistemic 根拠

- **Misportrayal harm**: "Speaking for others"（Alcoff 1991）の歴史的不正義を再演する。autism の medical treatment 推進 vs 当事者の stigma reduction 選好の対立など。**Double consciousness** (Du Bois 1903) の再生産
- **Flattening harm**: 集団内均質化は intersectionality の否定（Crenshaw 1989、Combahee River Collective 1977）
- **Essentialization harm**: アイデンティティを fixed traits に帰着させ、集団差を「内在的」と誤認させる

### 5.2 Can vs Should

> When prediction is cheap, allowing individuals to retain decisional autonomy will feel increasingly costly. — Geddes

たとえ技術的に可能でも、誰を direct engagement から除外するかの問いは残る。

### 5.3 限界

- 16 の US demographic に限定
- **37% のインターネット非接触人口**、口承伝統文化は完全に欠落
- IRB exempt ながら人間データは公開せず（LLM データのみ公開: osf.io/7gmzq）
- 本論文の知見は crossentropy loss + online text 訓練のパラダイムに依存。constitutional AI や新規データセットでは別の結果があり得る

### 5.4 緩和策の限界

- Identity-coded names: Black だけに部分的に有効
- Higher temperature: incoherence との trade-off
- **本質的に解決は不能**とは結論づけない — ただし context ごとに benefits vs harms の評価が必要

---

## 6. CS 222 での位置づけ

### Lecture 13 (Ethics) の中心論文

Park氏はこの論文を **13_2 Santurkar et al. "Whose Opinions Do Language Models Reflect?" と対**で扱う:

- 13_2: 集団意見の**平均的位置**のズレを測定（Pew 60 集団）
- 13_1: **個別集団の内部多様性** と **out-group vs in-group 表象** のズレを測定

両論文がLLMによる人口集団シミュレーションの根本的限界を示す。

### 生成エージェント研究との緊張

Park et al. 2023 *Generative Agents* は persona ベース。この論文の知見は、**persona プロンプトの信頼性に根本的疑問**を投げかける:

- Smallville のエージェントも、female/non-binary/disabled キャラクターの表現は out-group imitation に傾いている可能性
- **Park氏が Lecture 13 でこの緊張を率直に議論する**

### Out of One, Many (Argyle et al. 2023, 論文 03_2) との比較

- Argyle: political alignment を demographic persona で silicon sampling
- Wang: free-response で見ると、その silicon sample は **out-group imitation**

---

## 7. 主要引用

### 本論文の思想基盤

- **Harding (1991)** *Whose Science? Whose Knowledge?* — standpoint epistemology
- **Wylie (2003)** "Why Standpoint Matters"
- **Alcoff (1991)** "The Problem of Speaking for Others"
- **Spivak (1988)** "Can the Subaltern Speak?"
- **Fricker (2007)** *Epistemic Injustice*
- **Crenshaw (1989)** intersectionality
- **Collins (1990)** *Black Feminist Thought*

### 本論文が批判する先行研究

- **Argyle et al. 2023** *Out of One, Many* (論文 03_2)
- **Hämäläinen et al. 2023** (CHI) HCI 合成データ生成
- **Ziems et al. 2023** LLM for computational social science
- **Grossmann et al. 2023** (*Science*) "AI and transformation of social science"

### 本論文を引用する後続研究

- Lecture 13 の議論全体
- **Agnew et al. 2024** (CHI) "The illusion of artificial inclusion"
- **Cheng, Durmus, Jurafsky 2023** (ACL) "Marked Personas" / (EMNLP) "CoMPosT"

---

## 8. 要点

1. LLM は identity prompt を与えると、**in-group 表象より out-group imitation に近い応答**を生成する（misportrayal）
2. LLM は demographic group 内の**多様性を flatten** する — 100 応答でも 5 MC のうち 3 カテゴリしかカバーしない
3. **4 LLM × 16 identity × 9 質問 × 3200 人間参加者**の large-scale empirical 検証
4. Misportrayal は non-binary person、impaired vision、White person（R1）、woman/Gen Z（R2）で顕著
5. Identity-coded names は名前が demographic signal を持つ Black groups で部分緩和、high temperature は incoherence を引き起こす
6. R4-Coverage では **Myers-Briggs や crowdsourced personas のほうが、sensitive demographic より高い coverage**
7. 3 つの harm は epistemic injustice の歴史的系譜に直結（speaking for others、intersectionality erasure、essentialization）
8. LLM 代替は **supplement ≠ replace** の区別が重要。R1/R2 で replace は normative consequence が大きい
