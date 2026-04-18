# 03_1 — Social Simulacra: Creating Populated Prototypes for Social Computing Systems

## 書誌情報

- 著者: **Joon Sung Park**, Lindsay Popowski, Carrie J. Cai, Meredith Ringel Morris, Percy Liang, **Michael S. Bernstein**
- 所属: Stanford University / Google Research
- 発表: **UIST 2022**（35th Annual ACM Symposium on User Interface Software and Technology）
- Lecture 03（および Lecture 06）の補足論文。CS 222 講義全体の着想源の一つ

---

## 1. 研究問題

> How do we anticipate the interactions that will arise when a social computing system is populated?

ソーシャル・コンピューティング（SNS、コミュニティサイト）の設計では、少数のテストユーザーでは見えない**大規模運用時の挙動**（アンチソーシャル行為、誤情報、規範の崩壊）が最大のリスク。しかし従来のプロトタイピング手法は:
- 少人数リクルートが前提（experience prototype）
- cold start 時と critical mass 時の挙動差が大きい（Bernstein et al. 2011）
- 実ユーザーに未検証設計を晒すのは倫理的問題（Facebook emotional contagion、Kramer et al. 2014）

---

## 2. 提案: Social Simulacra（社会的シミュラクラ）

> Social simulacra take as input the designer's description of a community's design — **goal, rules, and member personas** — and produce as output an instance of that design with simulated behavior, including posts, replies, and anti-social behaviors.

- 入力: コミュニティの目標・ルール・メンバーペルソナ
- 出力: 投稿、返信、反社会行為を含むシミュレートされた社会相互作用
- LLM（GPT-3 davinci）を使う
- 実装: **SimReddit**（Reddit コミュニティ用のプロトタイプツール）

### キー洞察

> Large language models' training data already includes **a wide variety of positive and negative behavior on social media platforms**.

LLM は既に SNS 上の肯定的/否定的行動を学習している。**トロール挙動を生成できることは"バグではなく機能"** — 設計者がその挙動に備えられる。

---

## 3. SimReddit の3つの主要機能

### 3.1 Generate（生成）

| 入力 | 説明 |
|------|------|
| Target population | ユーザーペルソナ（名前+簡潔な説明、例: "Yuna Kim: a tennis fan rooting for Roger Federer"） |
| Community goal | コミュニティ目的（例: "modern art aficionados discussing their art interest"） |
| Rules | ルール（例: "be kind", "no advertisements"）— ナッジとして機能 |

デザイナーは**10個程度のシード・ペルソナ**を提供、SimRedditが**1,000個のペルソナ**に拡張。

### プロンプト手法（Prompt Chaining）

**Step 1: Persona expansion** — 10人のseedから1000人を few-shot prompting で生成

**Step 2: Top-level post generation** — HTMLタグ（`<span class="headline_reddit">`）を使って境界を明示し、コンテキスト窓制限を回避。ペルソナ記述＋ルール記述を含めたプロンプトで投稿を生成

**Step 3: Replies generation** — 確率 p(.65, 1.0) で反復、上限8返信。50%の確率で新ペルソナ、そうでなければ既参加者から選ぶ

### 3.2 WhatIf

会話中の発話を選択し、「別のペルソナが介入したら？」を探索。例:
- トロールが返信したら → 生成してみる
- モデレーターが介入したら → どう応答するか？
- モデレーション文言を変えたら → どう変わるか？

### 3.3 Multiverse

> The aim of social simulacra is not to make a single point prediction... but rather to provide **inductive insights on what could happen**.

- 同じ設計から**複数の代替宇宙**を生成
- Community-wide: コミュニティ全体を別の組み合わせで再生成
- Utterance-specific: 特定発話から複数の分岐シナリオを生成
- GPT-3 の temperature を 0.7→0.8 に上げて多様性を増す

---

## 4. 技術的評価（Technical Evaluation）

### 設計

- **GPT-3 のリリース後（2020年6月11日以降）に作られた50 subreddits**をサンプル（記憶化を避ける）
- 実会話と生成会話のペアを200人の参加者に見せ、どちらが本物かを判別させる
- 4条件（ablation）:
  - Crowdworker baseline（Prolific ワーカーが生成）
  - SimReddit w/o community description
  - SimReddit w/o personas
  - Full SimReddit

### 主要結果（Table 1）

| 条件 | 誤認率（高いほど区別困難） | SD |
|------|--------------------------|---|
| **SimReddit（完全）** | **41%** | 10% |
| Crowdworker | 32% | 13% |
| SimReddit w/o description | 21% | 15% |
| SimReddit w/o personas | 34% | 10% |

**One-way ANOVA**: F(3, 196) = 22.49, p < 0.001

**Tukey's HSD**: SimReddit は他の3条件すべてに対して有意に優れる（p < 0.01）

### 含意
- 参加者はランダムより少しだけ優れるレベル（50%が完全区別不能） → SimReddit の生成は高度にもっともらしい
- **クラウドワーカーより SimReddit の方がよりリアル**（32% vs 41%）
- ペルソナと community description の両方が重要

### 興味深い挙動例

**未知ドメインへの汎化**: COVID-19 を知らない davinci モデル（訓練データ打切り前）でも、"COVID vaccination" という community goal から反ワクチン sentiment を生成:
> Each year, the new flu vaccine is not an exact match... Why put a shot in your arm when you really don't need one?

**失敗例**: "covidlonghaulers" のような訓練時に存在しない専門分野では生成が意味をなさない場合もある。

---

## 5. 設計者評価（Designer Evaluation, N=16）

### プロトコル

90-120分のインタビュー:
1. 事前デザインタスク: 被験者が作りたい subreddit を draft
2. SimReddit による生成を見せ、think-aloud
3. 設計を revise
4. WhatIf でトロール・モデレーション介入を試す
5. Multiverse で不確実性を探索
6. 再生成を確認

### 主要発見

#### 6.4.1 Cold start 設計の難しさ
- 13/16 が「圧倒的（daunting）」と証言
- P1: 「playful か dry か分からない」
- P11: 「全てが頭の中、具体的に見えないと不安」
- 9/16: 未検証設計を実ユーザーに出すのは倫理問題
- 3/16: 過去経験から「全ルールは大事故後に作られる」（P8: "after fragmenting people and killing our community for a while"）

#### 6.4.2 具体的な設計洞察
- **ポジティブな驚き**: P1のピッツバーグ・イベント共有コミュニティで、予想外に**友達探し行動**（"Pittsburgh, I need a friend to see the sights with"）が生成され、学生向けの価値を発見
- **ネガティブな警告**: P5の国際問題コミュニティで、**ロシアのトロール農場**的な誤情報投稿が生成され、モデレーションの必要性を認識
- **曖昧ケース**: P13は "Opinion on the living room?" のような漠然とした投稿を許すか議論

#### 6.4.3 反復による改善（15/16 が設計を revise）
- 失敗ケース予防: "no business-promotional" ルール追加
- 文化・規範の形成: "happy" な場を目指す、"creepy でない" を明示
- 10/16 が結果の改善に満足

#### 6.4.4 WhatIf のモデレーション計画支援
P27 の Counter Strike コミュニティでトロール返信の3例を見た後:
> I should definitely have a rule for not calling other players noobs or washed up. Maybe even ban that word... Also, swearing.

P11 はトロール応答の3パターン（謝罪 / 開き直り / 主張）から、それぞれ異なる措置（残留 / 永久BAN / 一時BAN）を決める段階的判断を組み立てた。

#### 6.4.5 役割評価
- 14/16 が「リアル」と評価（P26: "I'm assuming that someone actually wrote these, right?"）
- 15/16 が「非現実的な部分」にも気づく（P12: "people wouldn't actually use a long paragraph"）
- 全員が**設計プロセスに付加価値を与える**と報告

#### 6.4.6 マージナライズド・グループの設計
- 女性有色人種5名、宗教/民族マイノリティ3名を含む
- マイノリティ対象の嫌がらせ・ヘイト行動を simulacra が surface
- P9: 非英語話者への白人至上主義書籍の共有という攻撃パターンを発見
- P25: 男性被験者が、危険地域探訪に関する「ミソジニー的嘲笑」を発見
- → **マージナライズド・グループ保護ルール**を明示的に追加するきっかけに

---

## 6. Discussion

### 7.1 Social Simulacra の役割

> Social simulacra fulfill the role that many of the early prototyping techniques fulfill... they **push the designer to question their assumptions**... Like any prototyping tool, social simulacra must be coupled with designer expertise.

- **Cue recall**: チェックリストより強く記憶喚起する
- **Proactive design**: 事後対応ではなく事前対応を可能にする

### 7.1.1 False Negatives
- Multiverse でも全ての可能性は尽くせない
- **Implied truth effect**（Pennycook et al. 2020）: ツールが一部の問題を surface すると、他の未表示問題がないと誤解する危険
- ただし、少人数テストよりは格段に広い範囲をカバー

### 7.2 限界と今後

- **予測装置ではない**: 社会動態は不予測（Salganik et al. 2006 を引用）
- **Reddit 以外への一般化**: Facebook グループ、Instagram 型空間（DALL-E 併用）
- **GPT-3 の技術制約**: 8,000文字プロンプト制限、英語のみ

### 7.3 倫理・社会的影響

- 生成コンテンツにバイアス・有害性のリスク
- 悪用リスク（astroturfing、大規模嫌がらせ、プロパガンダ）
- **推奨原則**:
  1. 認可された設計者のみが利用
  2. 生成コンテンツを中央でログ・監査
  3. 生成物を定期サンプリング・検索でスクレイピング悪用をフラグ
- バイアス再生産リスク（女性・マイノリティの SNS 上の沈黙を学習してしまう）
- **参加型デザインの代替ではなく補完**

---

## 7. 論文の締め: ユーモラスな自己参照

著者たちは「UIST 査読委員会 subreddit」のsimulacraを作り、自論文の査読を自動生成させる:

- **R2 (cynical)**: 「アイデアは興味深いが、文章は粗く、技法の詳細が不足」
- **R1 (thrilled)**: 「優れた論文。ソーシャル・コンピューティング設計に大きく貢献」
- **AC**: 「刺激的なアイデア。ただ利点と限界のより深い議論が望まれる」

---

## 8. CS 222 での位置づけ

- **Lecture 03**: 「集団モデル」の例としての代表研究
- **Lecture 06**: 環境設計の例（目標・ルール・ペルソナが選択肢次元を定義）
- **Lecture 09-10**: Agent Bank への構想の起源（ペルソナから多数エージェントを生成）
- **Lecture 13**: 倫理議論で再登場

Park氏の個人的研究系譜上で:
- **Social Simulacra (UIST 2022, 本論文)**: 集団生成の最初の重要論文
- → **Generative Agents (UIST 2023)**: 長期記憶・リフレクション・計画を持つ個別エージェントへ発展

---

## 9. 主要引用

### 論文が引用する関連研究
- **Rittel & Webber (1973)**: Wicked problems（設計の認識論的基盤）
- **Salganik, Dodds, Watts (2006)** *Science*: 社会動態の不予測性
- **Grudin (1994)** *CACM* 37: Groupware の8課題、cold start 問題
- **Bernstein et al. (2011)**: ソーシャル・コンピューティング研究の困難
- Brown et al. (2020) GPT-3、Gehman et al. (2020) RealToxicityPrompts
- Cheng et al. (2017) トロール行動、Kiene et al. (2016) Eternal September
- Fiesler et al. (2018) Reddit rules 特徴付け

### 本論文を引用する後続研究
- Park et al. (UIST 2023) Generative Agents — 本論文の直接の発展形
- CS 222 全体

---

## 要点

1. **Social Simulacra** = コミュニティ設計（目的・ルール・ペルソナ）を入力に、LLM で populated prototype を生成する手法
2. **LLM が SNS の肯定/否定行動を既に学習している**ことを活用。トロール生成は機能
3. **3つの機能**: Generate（1000人規模）、WhatIf（シナリオ探索）、Multiverse（多元宇宙）
4. **プロンプト手法**: few-shot persona expansion + HTML tag で境界明示 + 再帰的返信生成
5. **技術評価**: 200人の判別実験で **SimReddit 誤認率 41%、クラウドワーカー 32%**。LLM 生成が人間生成より「リアル」
6. **設計者評価** (N=16): 全員が設計プロセスへの付加価値を確認、15/16 が設計を revise。特にマージナライズド・グループ保護で価値
7. **予測装置ではなく**仮説生成/再帰的設計反復の道具。Rittel の "wicked problems" のコース原点に接続
8. **Park氏の研究系譜**で、Social Simulacra (2022) → Generative Agents (2023) → CS 222 (2024) と発展
