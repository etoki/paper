# 06_2 — Roleplay-doh: Enabling Domain-Experts to Create LLM-simulated Patients via Eliciting and Adhering to Principles

## 書誌情報

- 著者: **Ryan Louie**, Ananjan Nandi, William Fang, Cheng Chang, **Emma Brunskill**, **Diyi Yang**
- 所属: Stanford University（Computer Science、Diyi Yang research group 中心）
- 掲載: arXiv:2407.00870v2 (2024-07-14)、EMNLP 2024 投稿版
- プロジェクトサイト: https://roleplay-doh.github.io/
- Contact: rylouie@stanford.edu, diyiy@stanford.edu
- Lecture 06 の補足論文（応用: simulation での domain expert との協働）

---

## 1. 研究問題

### 背景
- LLM シミュレーションは sensitive domain（メンタルヘルス）でも activity as practice partner として有望
- しかし課題:
  - プライバシーで real patient data が取れない → finetuning 不可
  - Naive prompting では「feelings を直接言う」等、実患者らしくない
  - ドメインエキスパートは **how to prompt を知らない**（Zamfirescu-Pereira et al. 2023）
  - 研究者が think-aloud から prompt design に訳すのは非効率

### 具体的応用
- 新米カウンセラーの training partner としての **AI patient** 作成
- 既存の digital patient は **tailored dialogue setup** で練習コンテキストが限定

### 研究の問い
- Domain-experts が **直接 LLM 出力を制御**できる tool を設計し、その効果を検証

---

## 2. 手法: Roleplay-doh の 2 フェーズ

### 2.1 Principle Elicitation

Petridis et al. (2024) のユーザー駆動 chatbot デザイン枠組を継承。

**流れ**:
1. Counselor が AI patient と対話
2. 各応答に対してフィードバック:
   - "Kudos": 強化したい振る舞い
   - "Critique": 望ましくない振る舞い
   - "Rewrite": より望ましい response を示す
3. LLM (GPT-3.5) がフィードバックを principle に変換
   - GPT-3.5: kudos/critique の翻訳
   - GPT-4: rewrite と original の差分から principle 推論
4. Principles（例: "Respond to encouraging words with hesitation, doubting their significance"）が AI patient の constitution として保存

**Testing principles**:
- 最新 response を rewind して再生成
- 複数 response ではなく **単一 response** を生成（テストを manageable に）

**AI Patient prompt**:
- システムプロンプトで "roleplay as patient" とせず、"simulate a patient's next response in a dialogue"
- → role consistency 問題（LLM が assistant 的に振る舞う）を緩和

---

## 3. Pilot Testing で発覚した 2 課題

N=6 のカウンセラーで初版テスト。共著者4名が各4 AI patient を評価。

### 3.1 O1: "realistic" の定義が曖昧
- Counselor が imagined scenario を作ると、どの振る舞いが「現実的」か判断難
- 解決: **"過去の実ケースの再現"** と task を再定義

### 3.2 O2: 20% の応答が expert principles / dialogue conventions を満たさない

GPT-4 prompted simulation を評価: 276 件中 55 件（20%）が moderately/slightly/not-at-all satisfying

**3 種のエラー源**:
1. **複数 principles 同時満たせない**: 多数 / 複雑構成 principles で部分満たし
2. **Dialogue context に awkward**: 例: 会話途中で "Hi, A. Yes that's exactly what I mean..." と挨拶を挿入
3. **状況的 principle の誤適用**: 例: "encouraging words への hesitation" を encouraging 文脈でないのに適用

---

## 4. Principle-Adherence Pipeline（本論文の技術的貢献）

O2 を解決する 3 モジュール構成（Figure 2）:

### 4.1 Stage 1: Questions 生成

#### Principle-as-Questions Rewriter
- 各 expert principle を **複数の yes/no 質問** に分解
- 例: "Respond in short sentences and avoid using terms like 'anxious'"
  → Q1: "Does the patient's response employ short sentences?"
  → Q2: "Is the patient's language devoid of terms like 'anxious'?"

#### Automatic Principle Generator
- Dialogue convention に関する追加 question を生成
  - Coherence, consistency
- LLM に「patient/therapist の personality を仮定するな」と指示

### 4.2 Stage 2: Assessment and Refinement

#### Applicability and Adherence Evaluator
- 各 question に対して:
  - N/A: 状況に適用不能
  - Yes: 応答が principle 遵守
  - No: 違反
- 例: "Show willingness to engage in a suggested activity by affirming the proposal" は、therapist が activity 提案していないときに N/A

#### Refinement
- どれか "No" なら、evaluation 結果に基づき応答を rewrite
- 理想的には全 questions に Yes になるまで

---

## 5. User Study（Section 5）

### 5.1 設定

**Within-subjects design**、**25 counseling experts**:
- Counseling/Clinical psychology 大学院生 (with practicum)
- 7 Cups で 30+ clients にオンラインカウンセリング経験者
- Peer counselor

**2 条件**:
1. **Scenario-only**: counselor が patient scenario description を書くのみ
2. **Scenario + Expert-principles**: Roleplay-doh で principles も定義

**評価軸**（Himmelbauer 2018 Standardized Patients から）:
- Authenticity
- Stayed in role
- Resembled past case / Resembled typical case
- Mirrored challenging aspects / Challenged the counselor
- Ready as training partner
- Recommend to novices

### 5.2 主要結果（Table 1）

| Measure | Scenario Only | +Principles Δ | 有意性 |
|---------|---------------|---------------|--------|
| Authenticity | 5.24 | **+0.80** | ** (p<0.01) |
| Stayed in Role | 6.32 | +0.08 | ns |
| Resembled Past Case | 4.80 | **+0.76** | * |
| Mirrored Challenging Aspects | 4.52 | **+1.00** | * |
| Ready as Training Partner | 5.16 | **+0.64** | * |
| Recommend to Novices | 5.76 | **+0.52** | * |

→ Stayed-in-role 以外の **全 measure で有意改善**

**Scenario-Only の問題**:
- 感情の深みなし: "patients don't state a feeling such as 'I feel hopeless'. They display their current emotional state in their manner of speech."
- 真実を共有しすぎ: "as challenging as pulling teeth" べきだが too cooperative
- Behavioral trait（"not talkative"）を書いても反映されず

### 5.3 Principle の質的分析（123 total principles、median 5/patient）

**Conversation stage 別テーマ** (Liu et al. 2021 emotional support stages):

| Stage | # Patients | テーマ | 例 |
|-------|-----------|--------|-----|
| Stage-agnostic | 14 | Keep responses concise | — |
| Stage-agnostic | 14 | Use colloquial language | — |
| **Exploration** | 19 | Show mistrust/hesitation with seeking help | — |
| **Comforting** | 9 | Show emotions in detail \* | — |
| **Comforting** | 3 | Be less self-aware, disorganized articulation \* | — |
| **Action** | 12 | Do not seek solutions, share feelings \* | — |
| **Action** | 3 | Proactively seek solutions \* | — |

\* = **prior work で未定義の新 principle**

**矛盾する principles** も観察:
- Concise/direct (14) vs disorganized/conflicted (9)
- Proactively ask advice (12) vs share thoughts only (3)
- → **単一 principle set で AI patient を定義する限界**

### 5.4 Tool User Experience

Likert 7 点:
- Effective for guiding AI patient: **μ = 6.04, σ = 1.06**
- Easy to convert thoughts to principles: **μ = 6.12, σ = 1.13**
- Efficient writing: μ = 6.30, σ = 1.29
- Low mental demand: μ = 3.20, σ = 1.70（低いほど良い）
- **11.4% の principles が manual edit 必要**

---

## 6. Third-Party Evaluation（bias 除去）

### 6.1 設定
- Creator study には bias（創作者が自作を好む）の懸念
- **5 third-party counselors**（全員 creator study 参加者）が判定
- Randomized order で transcripts を提示
- Power analysis: 5 judges で 80% statistical power
- Linear mixed-effect model: `Rating ~ Treatment + CreatorID + (1|AnnotatorID)`

### 6.2 結果（Table 1 右）

**+Principles が有意に高評価**:
- Authenticity: +0.31 *
- Resembled Typical Case: +0.49 **
- Ready as Training Partner: +0.39 **
- Recommend to Novices: +0.38 *

**Creator study より小さい効果**。理由:
- Third-party が注目する principles と creator の principles が異なる
- 特定 creator が加えた principles への agreement が低い

---

## 7. Principle-Adherence Pipeline の評価（Section 6）

### 7.1 比較対象
40 error test cases を user study logs から選択（§3.2 のエラー範疇）:

1. **Full**: 完全 pipeline
2. **No Critique**: GPT-4 直接生成（baseline）
3. **No Principle Rewrites**: Rewriter なし
4. **No Autogenerated Criteria**: Auto Principle Generator なし
5. **Naive**: 全モジュールなしの pipeline（refinement のみ）

### 7.2 評価
カウンセラーが 5 モデルの応答を 1 (best) – 5 (worst) でランク:
- M1: Consistency with context
- M2: Awkward style (Yes/No)
- M3: Principle adherence
- Overall

### 7.3 主要結果（Figure 3）

**[Full] vs [No Critique] の prefer rate**:
- Consistency with Context: **35% win, 10% loss**
- Principle Adherence: **35% win, 5% loss**
- Overall: 類似

**Awkwardness (M2)**:
- Full: **2.5%** awkward
- No Critique: 15%
- Naive: 7.5%
- No Principle Rewrites: 7.5%
- No Autogenerated Criteria: 15%

**含意**:
- Principle-as-Questions Rewriter と Auto Principle Generator が**必須**
- Naive pipeline（module なし）は error cases でも改善なし
- Auto Principle Generator が Awkwardness を 12.5% 減

---

## 8. Discussion

### 8.1 貢献
- Domain experts が直接 LLM simulation を customize できる tool
- Principle 抽出+prompt engineering の two-level collaboration
- 技術的新規性: multi-faceted + contextual principles の constrained text generation pipeline

### 8.2 含意
- **Data-scarce domain** での LLM simulation が expert feedback で可能
- メンタルヘルス以外（教育、コーチング、対人訓練）にも汎化可能

### 8.3 限界
- 25 experts、US-based
- AI patient の realism と training efficacy の因果関係未検証
- Principle の矛盾で「単一 AI patient」の定義に限界

---

## 9. CS 222 での位置づけ

### Lecture 06: Simulation の応用と expert collaboration
- 05_1 Generative Agents の**応用拡張**（集団から個人シミュレーションへ）
- Park 氏の "believable proxies" が実臨床ドメインでどう機能するかの事例
- 06_1 Chang et al.（ネットワーク次元）と対照: 06_2 は対話・persona 次元

### 関連する CS 222 論文
- **03_1 Social Simulacra**: persona 生成の先行研究、AI patient は persistent版
- **05_1 Generative Agents**: Park 2023 を引用（believable proxies of human behavior）
- **05_2 CoALA**: principle-adherence pipeline は reasoning + learning action の具体化
- **13_1 Wang et al.**: identity group flatten の問題、本論文の "expert principles" がその mitigation
- **15_2 (人間-AI 関係)**: AI patient と therapist の関係で generative ghosts 的倫理議論

### Park 氏の議論との関係
- Generative Agents の "persona + memory" を、**domain expert の principle injection** で制御
- Social Simulacra の "rules as nudge" (Park 2022) の正統的発展
- **Expert-in-the-loop** vs Park 氏の **end-user-in-the-loop** の対比
- Lecture 06 (Agent Bank 構想) への応用接続

### 方法論的貢献
- **Principle-as-questions decomposition**: multi-facet constraints の generic な解決法
- **Applicability evaluator**: contextual principle の false-positive 防止
- Self-refine (Madaan 2024) を principle-driven に拡張

---

## 10. 主要引用

### 論文が引用
- **Park et al. (2023) Generative Agents** ← 05_1
- **Park et al. (2022) Social Simulacra** ← 03_1
- Petridis et al. (2024) user-driven chatbot design（基礎フレームワーク）
- Bai et al. (2022) Constitutional AI
- Madaan et al. (2024) Self-Refine
- Cheng et al. (2023) caricature in LLM simulations
- Zamfirescu-Pereira et al. (2023) "Why Johnny can't prompt"
- Chen et al. (2023) AI patients with colloquial/resistant language
- Stapleton et al. (2023) AI patients
- Yang et al. (2024) social skill training
- Shaikh et al. (2024) conflict resolution
- Markel et al. (2023) K12 teaching
- Liu et al. (2021) emotional support conversation stages
- Himmelbauer et al. (2018) Standardized Patient evaluation
- Braun & Clarke (2006) thematic analysis

### 本論文を引用する後続
- LLM expert collaboration 研究
- Mental health AI training tools
- Domain-expert prompting 研究

---

## 11. 要点

1. **Roleplay-doh = Expert-in-the-loop LLM simulation tool**: カウンセラーが自然言語で AI patient を customize
2. **Principle elicitation**: Kudos/Critique/Rewrite feedback を LLM で principle（自然言語 rule）に変換
3. **O2 の発見**: 20% の GPT-4 応答が expert principles を満たさない（3 エラー源: multi-part principle, awkward style, misapplied situational principle）
4. **Principle-Adherence Pipeline**:
   - Stage 1: **Questions 化 Rewriter** + **Auto Principle Generator**（dialogue convention）
   - Stage 2: **Applicability Evaluator** (N/A/Yes/No) + Refinement
5. **User study (N=25)**: Scenario+Principles が Authenticity (+0.80), Resembled Case (+0.76), Mirror Challenge (+1.00), Training Ready (+0.64) など有意に改善
6. **Third-party (N=5)**: Linear mixed-effect model で creator bias 除去後も有意効果
7. **Principle Adherence pipeline**: **35% win rate** over No-Critique baseline、**Awkwardness を 15%→2.5%** に削減
8. **Principle の多様性**: 123 principles を 7 テーマに分類、**4 つは prior work で未発見**（detailed emotion, disorganized articulation, share-only, proactive solutions）
9. **矛盾する principles の存在**: 単一 set で "realistic patient" は定義不能 → diverse AI patients の必要性
10. **CS 222 では**: Park 氏の believable agents の **臨床応用と expert collaboration**、05_1 を domain-specific に refine する方法論として位置づけ
