# Lecture 07: Believability vs. Accuracy

- 講師: Joon Sung Park
- ソース: `docs/stanford univ AI Agents and Simulations/07 Believability vs Accuracy.pdf`
- 位置づけ: シミュレーションの評価軸を**信憑性（belivability）**と**正確性（accuracy）**に二分し、それぞれの応用・測定法・限界を整理。

---

## 0. 週4までのまとめ

- シミュレーション = エージェントと環境の相互作用
- シミュレーションは wicked problems に取り組むべき
- 生成エージェントの一般的アーキテクチャ
- 環境をどう設計してエージェントを grounding するか

本講義から評価論に入る。

### Assignment 1 Q&A
- 会話中のタイムステップ: 全てのmemory nodeで0にするか、1ずつ増やしてよい
- Importance score: 0〜100の範囲

### 最終プロジェクト提案書
- 3〜4人グループ
- 締切: 2024/11/4（プレゼン当日）

---

## 1. Believable Agents（信憑性のあるエージェント）

### 起源: Disney の「生命の幻想」

**Thomas & Johnston (1981) *Disney Animation: The Illusion of Life* (Abbeville Press)**

> Disney animation makes audiences really believe in ... characters, whose adventures and misfortunes make people laugh—and even cry. There is a special ingredient in our type of animation that produces drawings that appear to **think and make decisions and act of their own volition**; it is what creates the **illusion of life**.

### AIへの移植: Bates (1994)

**Bates, J. (1994) "The Role of Emotion in Believable Agents" *Commun. ACM* 37, 122-125** ← 補足 `07_1`

> ... the idea of **believable agents**, by which we mean an interactive analog of the believable characters discussed in the Arts... We have argued that these artists hold some of the same central goals as AI researchers... may serve as a component of new user interacts for the broad human population.
>
> Believability. That is what we were striving for... **belief in the life of the characters**.

Bates の主要な主張:
> These include the appearance of reactivity, goals emotions, and situated social competence, among others. The emphasis in "alternative AI" on reactivity could be seen as choosing one of the believability requirements and **elevating it to a position of importance, while downgrading other qualities, such as those related to our idealized view of intelligence**.

→ 信憑性は「我々の理想化された知能観」とは**別方向**の目標。反応性・目標・感情・社会的能力などを立てる。

---

## 2. 信憑性の測定

### Turing Test
Turing (1950) "Computing machinery and intelligence" *Mind* 59(236), 433-460

### ABM は信憑性で評価されたか？
Schelling (1971) のモデルも、結局「現実の分離現象に似ているか」という信憑性基準で評価されたと解釈できる。

### Generative Agents の評価

Park et al. (UIST 2023):
> Generative agents were evaluated based on (essentially) a **behavioral Turing test**.

人間審査員にエージェント行動と人間行動を比較してもらう**行動版チューリングテスト**を採用。

---

## 3. 信憑性エージェントの応用領域

| 領域 | 例 |
|------|----|
| **ゲーム・ストーリーテリング** | The Sims, Minecraft |
| **AIコンパニオン** | character.ai, Replika |
| **リハーサル空間（人間向け）** | Shaikh et al. "Rehearsal: Simulating Conflict to Teach Conflict Resolution" (CHI 2024) |
| **リハーサル空間（エージェント向け）** | Louie et al. "Roleplay-doh" (2024) 補足 `06_2` |
| **医療訓練** | Li et al. "Agent Hospital: A Simulacrum of Hospital with Evolvable Medical Agents" (2024) |

### 結論

> Believable agents offer an **illusion of life**. (But still only a plausible simulacra.)

「生命の幻想」を提供するが、あくまで「もっともらしいシミュラクラ」に留まる。

---

## 4. Accuracy（正確性）

### 定義
> Accurate simulations are **predictions of the future**.

正確性 = 未来を予測すること。

### 挑戦: 構築か評価か？

> Is the challenge in "building" accurate agents or in understanding when they are accurate through "evaluation"?

Park氏の立場: **評価こそが主要な課題**。

### 一般的な評価スキーム

> Gather ground-truth data and see if the simulation replicates it.

Ground-truthデータを集めて、シミュレーションが再現するかを見る。

---

## 5. 粒度別の評価課題

| 粒度 | 課題 |
|------|------|
| **Individual** | Open-endedでどの軸で評価？ 一貫性の欠如（Inconsistency） |
| **Group** | 複雑なダイナミクス（不可能だと主張する人もいる） |
| **Population** | Ground truth が欠けることがある。既存研究を再現する場合、**モデルが既に研究を記憶（memorize）している可能性** |

---

## 6. 人口レベル評価の最新成果

### Ashokkumar et al. (2024) 「言語モデルは未公刊の実験も予測できるか？」

**Ashokkumar, Hewitt, Ghezae, Willer (2024) "Predicting Results of Social Science Experiments Using Large Language Models"** ← 補足 `07_2`

- LMの訓練データに**含まれていない**新しい研究を予測できるか？
- 答え: **Yes**

意義: 既存研究の再現だけなら memorization の可能性があるが、未公刊の新実験も予測できれば、真の予測力と言える。

### 現状の評価可能性

| 粒度 | 可否 |
|------|------|
| **Population-level** | **Yes**（できる） |
| **Individual-level** | **Verdict is still out**（判定保留） |
| **Group** | **The real question — not sure**（本質的問いだが不明） |

---

## 7. 正確性エージェントの応用

### Wicked problems の多くは正確なシミュレーションを要する

- **Social sciences**（社会科学）
- **Market research**（市場調査）
- **Urban studies**（都市研究）

### パーソナルアシスタント

> Can we build personal assistants that **simulate their users** to create a model of their needs?

ユーザーを模擬するシミュレーションで、ニーズのモデルを構築するパーソナルアシスタント。

---

## 8. 信憑性 vs 正確性の関係

Park氏の重要な観察:

> As simulations become more accurate, it does not necessarily mean they become more believable.
> For example, does an accurate simulation need to provide the illusion of life through emotions? **Maybe, but maybe not.**

- 正確でも信憑性がないことは有り得る（例: 統計モデルは予測力があるが「生命の幻想」はない）
- 信憑性があっても不正確なことは有り得る（Smallvilleの魅力的な挙動は、必ずしも未来予測ではない）

→ **信憑性と正確性は独立軸**。目的に応じてどちらを優先するか決める。

---

## 主要引用文献（Lecture 07）

### 信憑性の系譜
- **Thomas & Johnston (1981)** *Disney Animation: The Illusion of Life*
- **Bates (1994)** "The Role of Emotion in Believable Agents" *CACM* 37, 122-125（補足 07_1）
- Turing (1950) *Mind* 59(236)
- Schelling (1971) *J. Math. Sociol.* 1

### 信憑性エージェントの応用
- Park et al. (UIST 2023) Generative Agents
- **Louie et al. (2024)** Roleplay-doh（補足 06_2）
- **Shaikh et al. (CHI 2024)** "Rehearsal"
- Li et al. (2024) Agent Hospital

### 正確性の評価
- **Ashokkumar, Hewitt, Ghezae, Willer (2024)** "Predicting Social Science Experiments"（補足 07_2）

---

## 要点

1. **信憑性** = Disney の「生命の幻想」、Bates (1994) の believable agents に起源。**行動版チューリングテスト**で測れる。**理想化された知能観とは別方向**
2. 信憑性の応用: ゲーム（Sims, Minecraft）、AIコンパニオン（character.ai, Replika）、リハーサル空間（Rehearsal, Roleplay-doh）、医療訓練（Agent Hospital）
3. **正確性** = 未来予測。課題は**構築**より**評価**
4. 評価は粒度ごとに困難: individual=評価軸不定＋非一貫 / group=複雑動態 / population=ground truth不足＋学習データ漏洩リスク
5. **Ashokkumar et al. (2024)**: LMは**未公刊の社会科学実験**も予測でき、人口レベルでは予測力がある
6. 現状の可否: Population=Yes / Individual=未判定 / Group=不明
7. **正確性と信憑性は独立軸** — 正確だが信憑性がない／信憑性はあるが不正確、のどちらも有り得る
