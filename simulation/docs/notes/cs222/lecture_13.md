# Lecture 13: Ethics and Limitations

- 講師: **Helena Vasconcelos** と **Carolyn Zou**（本コースCA、ゲスト講義）
- ソース: `docs/stanford univ AI Agents and Simulations/13 Ethics and Limitations.pdf`
- 位置づけ: LLM ベースのシミュレーションを**訓練 → 訓練データ → 推論 → 検証 → 依存**の5段階で分析し、各段階の倫理・限界を整理。

---

## 0. Lecture Roadmap（5段階）

| 段階 | 問い |
|------|------|
| **LLM Training** | RLHF 等の訓練法はエージェント挙動にどう影響するか？ |
| **Training Data** | モデルは何をどこから学んだか？ それが精度をどう制約するか？ |
| **Running Inference** | 確率性と記憶化はどう精度に影響するか？ アーキテクチャは？ |
| **Validation** | シミュレーション出力をどう検証するか？ 何を信頼するか？ |
| **Reliance** | どの程度信頼すべきか？ 過信するとどうなるか？ |

---

## 1. LLM Training（訓練段階）

### 重要な主張

> These models are **not optimized to act like people**.

- Next token prediction は人間行動予測と直接対応しない
- GPT-3 → ChatGPT の飛躍は **instruction tuning**（命令従順化）
- RLHF でさらに「知識豊富・親切・有益」に整形

結果として、LLM は **"humanlike, but always follows instructions, always knows the answer, is friendly"** — 本当の人間とは乖離した"助手"に偏向している。

### 含意
一部研究者は挙動予測を直接訓練するアプローチを探るが、まだ初期段階。

---

## 2. Training Data（訓練データ段階）

### 訓練データの出所
- ウェブからスクレイプ（Wikipedia, Reddit）
- Q&A、情報系データ
- RLHF データ（ペア比較ランキング）

### データの制約

- **シミュレーションに適したタスク**: オンラインダイナミクス、知識ベースのタスク（訓練データに近い）
- **シミュレーションに不向きなタスク**: 物理ダイナミクスを要するもの（LLM パラダイムでは捉えられない）

---

## 3. Running Inference（推論段階）

### シミュレーションの堅牢性への脅威

人間行動のメタファーで意味付けしたくなるが、LLM 特有の3挙動がこれを危険にする:

1. **Prompt sensitivity**（プロンプト感度）
2. **Stochasticity**（確率性）
3. **Memorization**（記憶化）

### 推奨アプローチ: Perturb and Iterate

**Perturb（摂動）** — 以下の次元で probe:

| Dimension | 方法 |
|-----------|------|
| Protocol | 実験条件を自明な範囲で拡張 |
| Language | セマンティクスを保ちつつプロンプトを書き直し |
| Settings | ハイパーパラメータ・モデルバージョンを反復 |
| Format | 入出力フォーマット、桁数、改行を変更 |
| Strategy | CoT 有無、前置き要素の切替 |

**Iterate（反復）**:
- プロンプトは（隠れた）母集団からのドロー
- 多数ドローで（模擬）サンプル、多数サンプルで標本分布

### Stochasticity の具体的問題

- 予測トークン分布が実際の人間テキスト頻度とずれる可能性
- **分布的ミスアライメント**が累積エラーを生む
- 脅威:
  - 統計的にありえない結果が成功として報告される
  - 再現不能になる

引用: Zhang et al. (2024) "Forcing Diffuse Distributions out of Language Models"
Dillion, Tandon, Gu, Gray (2023) "Can AI language models replace human participants?" *Trends in Cognitive Sciences* 27(7)

### Memorization の具体的問題

**Wason Selection Task**（4枚カード課題）の例:
- "A, K, 4, 7" — "母音の裏は偶数" を検証するにはどのカードを裏返すか？
- 正解: A (modus ponens) + 7 (modus tollens)

**発見**（Binz & Schulz 2023 *PNAS* 120(6)）:
- 正典版 "vowel and even number": **75%** 正答
- 新バージョン "consonant and odd number": **9%** 正答
- → モデルは正典テキストを**記憶**している可能性
- 再現研究で正典の instrument を使うと**交絡**になる

### アーキテクチャの影響

Assignment 1 で retrieval と memory を実装したのを想起せよ。実装しなければエージェントは正しく答えられない。Generative Agents のアーキテクチャは強力だが、人々は今後も実験を続ける。

---

## 4. Validation（検証段階）

### 核心問題

> Believability ≠ Accuracy（Lecture 7の再掲）

**新規な結果**をどう検証するか？

### 動機付けの例: Feed Algorithm 研究

2つのフィードアルゴリズム:
- Engagement-based: 関心ベース並び
- Reverse chronological: 時系列並び

### 研究が期待に反したパターン

**Engagement-based vs Chronological で political polarization に予測通りの差は出なかった**。論文の限界記述を引用:

> It is possible that such downstream effects require a more sustained intervention period...
> ...if this study were not run during a polarized election campaign...
> ...with fewer institutionalized protections (for example, a less-independent media or a weaker regulatory environment)...
> These factors may in turn have affected each other... so that in aggregate we did not observe discernible changes.

→ **現実実験でもしばしば検証は極めて困難**。シミュレーションで「なぜ効かなかったか」を解明したいが、結果を信頼できる必要がある。

### 認識論的分類

- **Real**: シミュレーションが反映したい結果
- **Realistic**: シミュレーションが実結果と整合
- **Believable**: LLM出力の多くが信憑的 — リアリズムを除外できない

### Vasconcelos & Zou et al. (2024) の中心主張

> Since there is **no validation without ground truth**, generative agent-based modeling has **threats to epistemic validity**.
> However, simulations can be useful! So, what can we do?

### 2つの問い

- **Q1**: 新規結果を持つシミュレーションに方法論的にどう信頼を得るか？
- **Q2**: これら結果にどれだけの認識論的確信を持つべきか？

**"Trust in a simulation"の定義**:
> A belief in the simulation's correctness along the axes of human behavior that are **known and relevant**.

### 伝統的ABMとの比較

| 特徴 | Traditional ABM | Generative ABM |
|------|-----------------|----------------|
| データ | 少 | 多 |
| 予測可能・解釈可能 | 中 | 中 |
| 潜在因子を捉える | 可 | 可 |

---

## 5. Local Inspection — 新しい検証法

### 基本アイデア

> Because we can't confirm or deny novel outcomes, we can only **reject individual simulations on the basis of inconsistency with some standard**.

新規結果の真偽は確定できない → 既知の基準と矛盾するシミュレーションを**棄却**することしかできない。

### Local Inspection とは

> Inspired by agent-based modeling, we present a class of methods to establish trust in novel outcomes simulated with LLM agents by **validating at the level of agents, rather than outcomes**.

アウトプットの妥当性ではなく**エージェントレベル**での妥当性を検証する。

### 具体的な Reject 基準例

- **性別アイデンティティだけで主要結果が決まり、かつ強い説明理論がない場合** → 棄却
- **破壊的エージェントを投入しても他エージェントの挙動が変化しない場合** → 棄却
- **多様な思考を示さず、不自然に反復的な挙動** → 棄却

### 含めるべき他の検証

- 特定の認知バイアスが再現されるか
- 煽動的言辞を増やすと即座に分極が増すか
- 社会的伝染が見られるか

### 限界

> But it's nearly impossible to check for all behaviors!

すべての挙動を検査することは不可能。研究者の裁量が必要。

### Global Audit / AgentBank の活用

フィールドはまだ「LLMシミュレーションの科学」を模索中。グローバル監査や AgentBank の活用も既に提案されている。

---

## 6. Reliance（依存段階）

### 目標: Human-AI Complementarity

人とAIの**相補性**が目標。しかし実現されていない。

### Overreliance（過度依存）

> **When people agree with an AI, even when the AI is wrong.**

Human Decision-Maker × AI Agent の2×2:

|  | Reject AI | Accept AI |
|--|-----------|-----------|
| AI Correct | Under-reliance | Appropriate |
| AI Incorrect | Appropriate | **Overreliance** |

実証研究:
- Bansal, Wu et al. (2021)
- Buçinca et al. (2021)
- Lai & Tan (2019)
- Panigutti, Beretta et al. (2022)

### 対処

> Making it easy to verify the AI (or alternatively, find errors in the model), through explanations or other means, will reduce overreliance.

Vasconcelos et al. (2023): **検証しやすさ**が過度依存を減らす鍵。

### シミュレーションへの含意

検証方法があっても、**エラーの検証が容易**でなければ過度依存は減らない → HCI システム設計が必要（open area）。

---

## 7. どれくらいの認識論的確信を置くべきか

確信のスペクトラム:
- **Less confidence** ←→ **More confidence**
- 既知挙動への確信 / 未知挙動への確信 / 全挙動への確信 / 検証法を通した確信

### 適切なアプリケーション選定マトリックス

| | 低確信で十分 | 高確信が必要 |
|--|--------------|--------------|
| **代替手段なし** | フィードアルゴリズム変更の仮説生成 | コミュニティの毒性耐性測定 |
| **代替手段が高コスト** | ユーザーインタビュー前の探索 | 選挙予測 |
| **代替手段あり** | コンテンツ・モデレーション変更のテスト | 参加型デザイン |

理想: 低確信で十分かつ代替手段が無いケースで使う。

---

## 8. 依存段階の残るリスク

> These simulations are purposefully used by people in ways that justify **unethical ends**... [or] researchers, policy makers, industry professionals get the wrong insights from simulations.

- **意図的な悪用**: 非倫理的目的の正当化材料
- **善意の誤用**: 誤った洞察を引き出してしまう

### 解釈エラー（interpretive errors）への推奨

1. **適切な抽象度を選ぶ**（Pick the right level of abstraction）
2. **設計判断を摂動させて因果を理解する**（Perturb design decisions to understand causality）
3. **人間認知のメタファーは目的を持って使う**（Use human cognition metaphors with purpose）
4. **データの出自を追跡する**（Track data provenance）

---

## 主要引用文献（Lecture 13）

### Stochasticity / Memorization
- Zhang et al. (2024) "Forcing Diffuse Distributions out of LMs"
- Dillion, Tandon, Gu, Gray (2023) *Trends in Cognitive Sciences* 27(7)
- Binz & Schulz (2023) *PNAS* 120(6)

### Overreliance
- Bansal, Wu et al. (2021)
- Buçinca et al. (2021)
- Lai & Tan (2019)
- Panigutti, Beretta et al. (2022)
- Vasconcelos et al. (2023)

### 中心論文
- **Vasconcelos & Zou et al. (2024)** — 本講義の核

---

## 要点

1. LLM は人間ではなく**"常に答えを返す助手"**に最適化されている（RLHF/instruction tuning の副作用）
2. 推論時の脅威: **prompt sensitivity / stochasticity / memorization**。対処は **"Perturb and Iterate"**
3. Memorization の実証: Wason 課題の正典版=75% / 非正典版=9%（Binz & Schulz 2023）
4. **Ground truth なしでは検証不能** → **Local Inspection**: エージェントレベルで既知挙動との矛盾を検査し、矛盾するシミュレーションを棄却
5. 棄却基準例: 性別だけで結果決定／破壊的エージェントが影響しない／多様性欠如
6. **Overreliance**: AIが誤っていても人が同意してしまう現象。検証しやすさ が鍵（Vasconcelos et al. 2023）
7. アプリ選定は「**認識論的確信の必要度 × 代替手段の有無**」の2軸で判断
8. 解釈エラー抑止の4原則: 適切な抽象度／因果摂動／目的ある比喩／データ出自追跡
