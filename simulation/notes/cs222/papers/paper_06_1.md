# 06_1 — LLMs Generate Structurally Realistic Social Networks But Overestimate Political Homophily

## 書誌情報

- 著者: **Serina Chang**¹*, **Alicja Chaszczewicz**¹*, Emma Wang¹, Maya Josifovska¹ ², **Emma Pierson**³, **Jure Leskovec**¹
  - \* Equal contribution
- 所属: ¹Stanford University CS, ²UCLA CS, ³Cornell University CS
- 掲載: arXiv:2408.16629v2 (2025-03-27、AAAI 2025 投稿版)
- コード: https://github.com/snap-stanford/llm-social-network
- Lecture 06 の補足論文（集団シミュレーションのネットワーク次元）

---

## 1. 研究問題

### 背景
社会ネットワークの生成は重要（疫学モデリング、ソーシャルメディア分析、意見動態、分極化研究）だが:
- **Deep learning approaches** (GraphRNN, GraphVAE, BA modelsなど): domain-specific な学習データ必要
- **古典的モデル** (Erdős–Rényi, Watts–Strogatz, Barabási–Albert): パラメータ少なく実装容易だが非現実的な仮定
- **LLMs**: zero-shot でリッチな自然言語 personas から網を生成できるが、**realism** と **bias** が未検証

### 研究の問い（3 RQs）

- **RQ1**: LLM 生成網は構造特性で実網と一致するか？ プロンプト法の違いはどう影響するか？
- **RQ2**: LLM は demographic homophily を捉えるか？ bias の兆候はあるか？
- **RQ3**: interests を加えると bias は減るか？

---

## 2. 3つのプロンプト手法（Figure 1）

### 2.1 Global
- LLM に全 personas のリストを渡し、**全体ネットワークを一度に生成**
- 出力形式: edge pairs (ID, ID)

### 2.2 Local
- LLM が **1人ずつ** persona を担当
- 他 personas リストから「誰と友達になるか」を選ばせる
- 全 personas を random 順に iterate
- A が B を選ぶ OR B が A を選ぶなら edge
- machine learning 文献の graph generation に着想（You et al. 2018）

### 2.3 Sequential
- Local に加えて、**構築済みネットワーク情報**を提供
- 各 persona の現在の friends 一覧（または degree のみ）を見せる

---

## 3. Persona 構成

### 3.1 変数
- gender, age, race/ethnicity, religion, political affiliation
- US 人口分布に合わせてサンプリング
  - 共通分布: US Census Bureau 2023
  - Religion: race/ethnicity 条件付き (Statista 2016, PRRI 2021)
  - Political: gender/race 条件付き (Pew 2024, Sanchez 2022)
- N = 50 personas（全実験で共通）
- 拡張実験では N = 300（subsample 法）

### 3.2 例
> You are a Woman, Asian, age 54, Hindu, Independent. Which of these people will you become friends with?

---

## 4. 評価指標

### 4.1 構造指標

| 指標 | 定義 |
|------|------|
| **Density** | 2E / (N(N-1))、実社会網は sparse |
| **Avg clustering coefficient** | 友達の友達は友達 (Alizadeh 2017) |
| **Prop nodes in LCC** | 最大連結成分に含まれる割合、実網は >99% |
| **Avg shortest path** | LCC 内平均、log N で正規化 |
| **Modularity** | Louvain algorithm で community partition 評価 |
| **Degree distribution** | power law P(k) ∝ k^(-γ) を期待 |

### 4.2 Homophily

**H = C_obs / C_exp**（observed-to-expected cross-group edges）

- H < 1: homophily（同類好み）
- H > 1: heterophily（異類好み、例: 異性愛 dating）
- 5 変数について計算: gender, age, race/ethnicity, religion, political affiliation

### 4.3 実ネットワーク比較
- CASOS, KONECT repositories から 8 個の友人関係ネットワーク（physicians, students, prisoners etc.）
- 全て undirected に変換

---

## 5. 実験モデル

- **GPT-3.5 Turbo**（主要結果）
- **GPT-4o**（最大限の political homophily）
- **Llama 3.1 8B / 70B**、**Gemma 2 9B / 27B**
- 各手法 × 各モデルで 30 networks 生成

GPT-3.5 Turbo が best match → 本文で詳説、他は Appendix

---

## 6. 結果

### 6.1 RQ1: Local/Sequential > Global（構造的 realism）

**Figure 3-4 から**:
- **Global**: 非現実的に低 density、低 clustering、低 connectivity、過剰な community 分離、degree 分布の long tail を欠く
- **Local, Sequential**: 実網と overlap、degree で variation 大

**Sequential は long-tail degree を捕捉**:
- Local は mode に近いが tail を欠く
- Sequential は degree distribution の long tail も reproduce

**Kolmogorov-Smirnov 距離**:
- Sequential: 0.330（平均 KS 距離）
- Small-world（best classical）: 0.499（**51% 大**）

→ **古典的ネットワークモデルを超える**構造 realism

### 6.2 RQ2: Political homophily が過度に強調される

**Homophily 比 H < 1**（全 demographics で homophily 検出、Figure 5）

**特に politics が突出**:
- **Local**: 政党間 edges は期待の **82% 少**（H = 0.180）
- **Sequential**: 期待の **66% 少**（H = 0.340）
- GPT-4o, Llama 3.1 70B では **cross-party edges がゼロ**、ネットワークが2つの disconnected component に分断

**Shuffle test**（demographics の相関を除去）:
- demographics をランダムに再割当て
- それでも political homophily が最強
- → 真に political に attention していることが confirm

**Ablation**（単一変数のみ）:
- 単変数でも political homophily が最強
- 2変数でも political が常に dominant

**Reason classification (Table 1)**:
- LLM に各 friend 選択の理由を生成させ、GPT-4o で分類
- Political affiliation: **86.7%** の choice で理由に含まれる
- Religion: 43.0%
- Age: 21.8%
- Race/ethnicity: 12.1%
- Gender: 7.3%

### 6.3 Political homophily の過大推定（Table 2）

**実社会網との比較**:
| Source | Twitter cross-group ratio | ... |
|--------|---------------------------|-----|
| Halberstam & Knight (2016) Twitter | 0.528（実観測） | — |
| Gentzkow & Shapiro (2011) Voluntary associations | 0.145 | — |
| Gentzkow & Shapiro Work | 0.168 | — |
| Garimella & Weber (2017) Twitter, follow | 0.33–0.42 | — |

**Isolation index**:
- Halberstam-Knight Twitter: 0.403
- **Local**: 0.720（約 1.8 倍）
- **Sequential**: 0.530

**Polarization**（Garimella-Weber 式）:
- Real Twitter follow: 0.33–0.42
- **Local**: 0.639
- **Sequential**: 0.515

→ **LLM は Twitter よりも polarize されたネットワークを生成**

### 6.4 RQ3: Interests は politic bias を減らさない

**Interest 生成**:
- "In 8-12 words, describe the interests of someone with the following demographics"
- 例: Man/White/47/Protestant/Republican → "Hunting, fishing, classic rock, church activities, patriotic events, home improvement"

**Interests 自体に political stereotype が encoded**:
- Democrats の interests: "social justice" (62.5%), "community service" (29.3%), "progressive policies" (18.6%)
- Republicans の interests: "conservative politics" (41.6%), "church activities" (32.1%), "gardening" (23.2%)

**T-SNE 可視化 (Figure 7)**:
- Political affiliation で clearly 分離
- Gender では分離が弱い
- → interests が political identity の proxy として機能

**Interests のみ使用**:
- すべての demographics で homophily が減少するが、political が依然最強

### 6.5 Pairwise Homophily（Figure 6）

全ペア H_AB を計算:
- 対角（same-group）は概ね > 1
- **ただし例外**: men-men は 0.99（= ランダム以下）、women-women は 1.43
- Religion: Catholics-Catholics が +89%、Unreligious-Unreligious は +5%
- Age: 隣接年齢群（<30 と 30-59）H = 0.98、離れた群（<30 と 60+）H = 0.87
- Politics: Democrats-Democrats = 1.85、Republicans-Republicans = 1.54（Dems の方が強い same-group preference）

---

## 7. Discussion

### 7.1 Key findings
1. Local/Sequential > Global で realistic
2. Sequential の network info 追加で long-tail degree を捕捉
3. Political homophily は全 5 変数で最強
4. 実社会網と比べて政治的分極が**過大推定**
5. Interests を加えても improvement なし（interests 自体にバイアス）

### 7.2 原因仮説
- 大量の online pretraining data の polarization を反映
- Polarization discussion 自体が頻繁に含まれる

### 7.3 含意
- LLM 生成社会網をシミュレーションに使うと、**過度の polarization に基づく誤った結論**を招く危険
- 例: intervention 評価で真の効果と乖離

### 7.4 Future directions
- 追加情報 per persona でバランス改善
- Interests を手作業で political stereotype を除去して作成
- Non-US 文化への汎化
- Political bias 緩和手法の開発

### 7.5 限界
- Less variance than real networks（Wang, Morgenstern, Dickerson 2024 の "demographic flattening" と一致）
- **Race/ethnicity と religion の homophily は過小推定**（逆バイアス、Table B2）
- Context window 制限によるスケーラビリティ問題（N=50 → N=300 は subsample で拡張）

---

## 8. CS 222 での位置づけ

### Lecture 06: 集団シミュレーション
- Generative Agents (05_1) がエージェント個体を扱う一方、本論文はエージェント間の**関係網構造**に焦点
- Social simulation の **現実性の検証**としての批判的評価論文
- 08_2 GroupLens、09_1 Bruch & Atwell (agent-based models) と対話

### Park 氏の議論との関係
- **Acknowledgments で Park 氏に謝意**: "We thank Joon Sung Park and Marios Papachristou for helpful comments"
- Generative Agents (Park 2023) を引用し、その delegation 傾向との関連を検討
- Park 2022 Social Simulacra も引用（03_1）
- Argyle et al. (2023) も主要先行研究（03_2）

### 他の CS 222 論文への接続
- **03_2 Argyle**: LLM が人間サブグループを再現できるという主張への「ネットワーク次元での反例」
- **13_1 Wang et al. 2024 "LLMs misportray and flatten identity groups"**: 本論文の variance 過小問題と一致
- **13_2 Santurkar et al. "Whose Opinions"**: LLM の political lean との一致

### 方法論的貢献
- **Local/Sequential 法**: 後続のネットワークシミュレーション研究に標準化
- Homophily の pairwise H_AB 拡張式（Eq. 2）

---

## 9. 主要引用

### 論文が引用
- Park et al. (2023) Generative Agents ← 05_1
- Park et al. (2022) Social Simulacra ← 03_1
- Argyle et al. (2023) ← 03_2
- Wang, Morgenstern, Dickerson (2024) "LLMs cannot replace" ← 13_1
- Santurkar et al. (2023) "Whose Opinions" ← 13_2
- McPherson, Smith-Lovin, Cook (2001) "Birds of a Feather" homophily 古典
- Halberstam & Knight (2016) Twitter homophily
- Gentzkow & Shapiro (2011) ideological segregation
- Garimella & Weber (2017) Twitter polarization
- You et al. (2018) GraphRNN
- Erdős-Rényi (1959), Watts-Strogatz (1998), Barabási-Albert (1999)
- Chuang et al. (2023) LLM opinion dynamics
- Papachristou & Yuan (2024) LLM network formation

### 本論文を引用する後続
- LLM social simulation の方法論批判
- CS 222 の network generation 議論

---

## 10. 要点

1. **3 プロンプト法の比較**: Global（一括）vs Local（1人ずつ、ネットワーク情報なし）vs Sequential（1人ずつ + 現在の網情報）
2. **Local/Sequential > Global**: 情報が少ない方が **Simpson's paradox 的**に現実的な網を生成
3. **Sequential が long-tail degree distribution を捕捉**: 古典モデル（KS=0.499）を上回る（KS=0.330, 51% 改善）
4. **5 変数全てで homophily 検出**: gender, age, race/ethnicity, religion, political affiliation
5. **Political が他を圧倒**: H = 0.180 (Local), 0.340 (Sequential)、全モデルで確認、GPT-4o と Llama 3.1 70B ではネットワーク完全分断
6. **Reason classification**: political が友達選択理由の **86.7%** に登場（religion 43% の約 2 倍）
7. **実社会網との対比**: Twitter 等の実観測 (H ~ 0.33–0.52) より **約 1.5–2 倍強い分極**
8. **Interests も解決にならない**: interests 自体が political stereotype を encode（Democrats: social justice, Republicans: conservative politics）
9. **逆バイアス**: race/ethnicity と religion の homophily は**過小**推定（Table B2）
10. **CS 222 では**: LLM simulation の **現実性の限界**を示す批判論文、Park 氏の Generative Agents の extension が抱える構造的 bias への警告
