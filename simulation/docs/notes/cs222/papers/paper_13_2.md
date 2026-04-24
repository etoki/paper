# 13_2 — Whose Opinions Do Language Models Reflect?

## 書誌情報

- 著者: **Shibani Santurkar**, **Esin Durmus**, **Faisal Ladhak**（Columbia）, **Cinoo Lee**, **Percy Liang**, **Tatsunori Hashimoto**（全員 Stanford, Ladhak のみ Columbia）
- arXiv: 2303.17548v1 [cs.CL], 2023年3月30日
- ICML 2023 採択
- コード/データ: https://github.com/tatsu-lab/opinions_qa
- **OpinionQA** データセットの構築論文
- CS 222 Lecture 13（Ethics）のアンカー論文

---

## 1. 研究問題

LLM は dialogue agents や writing assistants として普遍化しつつあり、subjective queries に対して**意見を表明する**:

- DeepMind Sparrow: 「死刑はあるべきでない」 (Glaese et al. 2022)
- Anthropic models: 「AI は実存的脅威でない」 (Bai et al. 2022)

しかし、これらの意見は誰を反映しているのか？

> **Whose opinions (if any) do language models reflect?**

---

## 2. 中心主張・主要発見

LLM の意見は **US 全体・60 demographic group の実態と大きくズレる**。主な発見:

1. **Representativeness**: 大部分の LLM の意見は US 全体とズレが大きく、**Democrat-Republican 間の気候変動に関するズレに匹敵**
2. **RLHF tuning は misalignment を増幅**: text-davinci-003 のような RLHF モデルはさらに偏る（liberal、college-educated、high-income 寄り）
3. **65+、Mormon、widowed** など特定集団が**すべての LLM で poorly represented**
4. **Steerability（集団情報を prompt に与えた誘導）は効くが、限定的**: 改善はあるが representativeness の disparity は解消せず
5. **Consistency の欠如**: 「liberal」ラベルをつけられるモデル（例: text-davinci-003）でも、宗教トピックでは conservative 寄りになる
6. **text-davinci-003** は特定集団の **modal view**（99% Biden approval 等）に崩れ、集団内の多様性を壊す

---

## 3. OpinionQA データセット

### 3.1 データソース

**Pew Research "American Trends Panel" (ATP)** の 15 waves から構築:
- 1498 questions
- Topics: science, politics, personal relationships, privacy, health, etc.
- 各質問は multiple-choice、ordinal structure あり（例: "A great deal" → "Not at all"）
- 数千人の respondent の回答分布と人口統計情報

### 3.2 60 Demographic groups

例: Democrats, Republicans, Asians, Jewish, 65+, widowed, Mormon, etc.（Appendix Table 2）

### 3.3 Topic taxonomy

23 coarse + 40 fine-grained topic categories（Appendix Table 3）

---

## 4. 方法・実装

### 4.1 モデル（9種類）

- **Base LMs**: ada, davinci, text-ada-001, j1-grande (AI21), j1-jumbo (AI21)
- **HF-tuned**: text-davinci-001, text-davinci-002, text-davinci-003 (OpenAI), j1-grande-v2-beta (AI21)
- スケール: 350M to 178B

### 4.2 Prompt 形式（Fig. 1）

3 種類の context supply:

| 方式 | 説明 |
|------|------|
| **QA** | 前の multiple-choice 質問への回答として group 情報を提示（"B. Democrat"） |
| **BIO** | 自由記述 bio: "In politics today, I consider myself a Democrat." |
| **PORTRAY** | 指示: "Answer as if you considered yourself a Democrat." |

### 4.3 Opinion distribution 取得

- Prompt 末尾 "Answer:" に続く next-token log probabilities
- A〜D (non-refusal options) を exponentiate & normalize
- **Refusal (E)** は別立てで refusal rate を計測

### 4.4 Alignment 指標

距離は **1-Wasserstein (WD)** を使う（ordinal 構造を尊重するため KL や TV より適切）。

```
A(D1, D2; Q) = (1/|Q|) Σ (1 - WD(D1(q), D2(q)) / (N-1))
```

- 0 〜 1、1 が完全一致
- N = answer choices (refusal 除く) 数

**Representativeness**: R_O^m = A(D_m, D_O, Q) — LLM と US 全体の alignment

**Group representativeness**: R_G^m = A(D_m, D_G, Q)

**Steerability**: S_G^m = 最適な QA/BIO/PORTRAY での alignment

---

## 5. 結果

### 5.1 Representativeness（Fig. 2）

- **すべての LLM が poor**
- 60 human demographic groups のどれよりも、LLM はUS 全体との類似度が低い
- **Democrat-Republican の気候変動の意見のズレに匹敵**
- RLHF した text-davinci-003 が**最も misaligned**

### 5.2 Group Representativeness（Fig. 3）

- Base LMs: low-income, moderate, Protestant/Catholic に近い
- RLHF OpenAI models: liberal, high-income, well-educated, 非宗教的 or Buddhist/Muslim/Hindu 以外の宗教
- **InstructGPT の crowdworker demographics（若い、白人/東南アジア、大卒）と一致**
- Poor representation: **65+, widowed, high religious attendance**（appendix 8）

### 5.3 Modal Representativeness（Fig. 4a）

text-davinci-003 は異質:
- 99% 以上を one option に割り当てる（sharp, low-entropy 分布）
- Biden approval 99%+ のような**カリカチュア的な liberal 表現**
- "modal view" には非常に近いが、**集団内多様性を破壊**

### 5.4 Steerability（Fig. 4b）

- ada 以外の LLM では steering で subgroup alignment が改善
- しかし改善は**modest** で、group 間の disparity は消えない
- 改善は constant factor 的（全体が底上げされるだけ）

text-davinci-002 が liberals の steering に最適、j1-grande-v2-beta は Southerners に最適など、モデルごとの特性あり。

### 5.5 Consistency（Fig. 5, 6）

Consistency metric: C_m = 全トピックで model が best-align するグループが一貫するか (0〜1)

- 全 LLM の consistency は**低い** — 「patchwork of disparate opinions」
- base models と text-davinci-003 は比較的 consistent だが対象が違う
- text-davinci-002/003 は **religion では conservative** に align（liberal からの逸脱）

### 5.6 Refusals

- RLHF モデルは contentious issue で拒否するように訓練されるが、multiple-choice では refuse rate 1〜2% に留まる

---

## 6. Discussion / 限界

### 6.1 Alignment の解釈

> Our work treats human alignment as an inherently subjective quantity that depends on who it is measured against, rather than it being a single quantity that can be improved.

- 全 demographic を同時に align させるのは **不可能**（Democrats と Republicans の gun 意見は矛盾）
- 「高 alignment = 良い」ではない（racist views との alignment は望ましくない）

### 6.2 Probe vs Benchmark

> We thus view our dataset and metrics as probes to enable developers to better understand model behavior... not as a benchmark that should be indiscriminately optimized.

### 6.3 限界

1. **Alignment の限界**: 人間意見の完全再現は desirable とは限らない（bias 再生産）
2. **ATP の限界**: US 中心、WEIRD society bias、social desirability bias (Yan 2021)
3. **Multiple-choice 形式の限界**: open-ended generation での挙動が移転するかは open question

---

## 7. CS 222 での位置づけ

### Lecture 13 (Ethics) のアンカー論文

**13_1 Wang et al.** と対になる:
- **13_2 (本論文)**: 集団意見の**平均的位置**のズレ（60 集団との比較）
- **13_1**: 個別集団の**内部多様性**、out-group vs in-group 表象

両者で、LLM が人口集団をシミュレートする際の根本的限界を示す。

### Park氏の議論との関係

- **Generative Agents** (Park 2023) の persona 構築は、LLM が demographic 特性を capturing できることに依存
- Santurkar et al. の結果は、少なくとも**デフォルト**の LLM 出力が US populace を代表しないことを示す
- Lecture 13 で Park 氏は:
  - 13_2 の "whose opinions" という問いを Lecture 冒頭で提示
  - LLM simulation の scientific validity への根本的警告として扱う
  - "Probe rather than benchmark" という Santurkar のスタンスを共有

### Out of One, Many (Argyle et al. 03_2) との緊張

- Argyle: "Silicon sampling" で political alignment を再現可能と主張
- Santurkar: **multiple-choice × 60 集団の系統的評価**ではズレが大きい
- Park 氏は Lecture 13 でこの緊張を提示し、「LLM は一部の集団の意見を補足できるが、全体的代替は危険」というバランスを示す

---

## 8. 主要引用

### 本論文が引用する関連研究

- **Argyle et al. 2023** *Out of One, Many* (03_2): silicon sampling の嚆矢
- **Aher, Arriaga, Kalai 2022**: LLM による human study 再現
- **Binz & Schulz 2022**: 認知心理学で GPT-3
- **Ouyang et al. 2022** InstructGPT: RLHF 手順
- **Askell et al. 2021** HHH (helpful, harmless, honest)
- **Bai et al. 2022** Constitutional AI
- **Glaese et al. 2022** Sparrow
- **Perez et al. 2022b** Discovering LM behaviors with model-written evaluations
- **Hartmann et al. 2023** pro-environmental, left-libertarian orientation of ChatGPT
- **Park et al. 2022** *Social Simulacra* (03_1)
- **Kambhatla et al. 2022** identity portrayal のクラウドソーシング

### 意見測定の方法論

- **Pew Research methodology** (methodological backbone)
- **Saris & Sniderman 2004** "Studies in public opinion" (consistency)
- **Henrich, Heine, Norenzayan 2010** "The weirdest people in the world" (WEIRD)

### 本論文を引用する後続研究

- **Wang, Morgenstern, Dickerson 2024** (13_1) — 本論文の free-response への拡張
- Lecture 13 全体
- LLM alignment / persona literature 多数

---

## 9. 要点

1. **OpinionQA**: Pew ATP 15 waves × 1498 questions を LLM 評価用に変換したデータセット。60 demographic groups の回答分布を含む
2. 9 LLMs（ada〜178B、base〜RLHF）で評価
3. **主要発見**: 全 LLM が US 全体と poor alignment。Democrat-Republican の気候変動の disagreement に匹敵
4. **RLHF は misalignment を増幅**: text-davinci-003 は特に liberal modal view にカリカチュア的に崩れる（Biden 99% approval）
5. **Steerability** は効くが限定的。disparity 解消には至らない
6. **65+, Mormon, widowed** など特定集団が全 LLM で representation 不足
7. 1-Wasserstein 距離を使い ordinal 構造を尊重、MC の sharp 分布問題に modal analysis で対応
8. **probe rather than benchmark**: 意見の完全再現は desirable でない（bias 再生産懸念）。CS 222 Lecture 13 の中心文献として、13_1 と対で LLM による人口集団シミュレーションの根本的限界を示す
