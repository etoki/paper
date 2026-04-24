# 09_1 — Agent-Based Models in Empirical Social Research

## 書誌情報

- 著者: **Elizabeth Bruch**（University of Michigan, Department of Sociology and Complex Systems）／ **Jon Atwell**（同, Complex Systems）
- 掲載誌: *Sociological Methods & Research* 2015, Vol. 44(2), pp. 186–221
- DOI: 10.1177/0049124113506405（受理: 2013年）
- Lecture 09 の補足論文。ABM の実証的応用のメタ方法論レビュー

---

## 1. 研究問題

エージェントベース・モデリング (Agent-Based Models, ABM) は近年人気を博しているが、実証的研究プログラムでの使い方に**成文化された推奨や実践**がまだない。社会学、生物学、計算機科学、疫学、統計学を横断して:

- ABM をいつ・どう使うべきか
- 個人行動と集団データをどうモデルに取り込むか
- モデルの validation と sensitivity test はどう行うか

を整理し、ABM を「実証的研究プログラムの道具」として位置づけることが目的。

---

## 2. ABM の定義と核心特性

> Agent-based models are computer programs in which artificial agents interact based on a set of rules and within an environment specified by the researcher (Miller and Page 2007).

### 核心: Micro-Macro Link

- ミクロ単位（agent）は予測可能なルールで振る舞う
- 彼らの相互作用と環境の累積が**予想外の集合パターン**（emergent）を生む
- 社会学の古典問題（Coleman 1994, Granovetter 1978, Hedström & Bearman 2009）

### Interdependent Behavior（相互依存行動）

ほぼ全ての人間行動は、他者の過去・現在・予測行動に**条件付けられている**（contingent）:
- 近隣選好は自分と他者の特性に依存
- 集団への加入/離脱自体が集団構成を変える
- 短期: 個人が環境に応答。長期: 個人の累積が環境を変える

この **feedback** ゆえに、独立サンプル・単方向因果を前提とする標準統計モデルは不適切。**非線形**関係を表現するには ABM のような手法が必要。

### Mechanism-based Explanations（機構論的説明）

ABM は Hedström & Swedberg (1998)、Elster (2007)、Hedström & Ylikoski (2010) の枠組みに合致:
- Tipping, contagion, diffusion, self-fulfilling prophecy, tragedy of the commons
- Selection (Hedström & Bearman 2009), offsetting (Bruch 2013), vacancy chains (White 1970), network externalities (DiMaggio & Garip 2011)

### ABM の5つの利点

1. **micro → macro のメカニズムを明示化**
2. 代替仮定下の挙動を比較探索できる
3. 研究者の "implicit models"（暗黙の仮定）を surface させる
4. **統計モデル特定と data 収集を誘導**できる（データが多すぎるときも、データがないときも）
5. 豊富・多層のデータを取り込める（独立観察や単方向因果を超えて）

---

## 3. Feedback 効果と Public Policy

従来手法で設計した政策が逆効果になる例:
- **低脂肪食品**（La Berge 2008）: 食べる量が増えて肥満助長
- **Section 8 housing voucher**（Galster 2005, Rosin 2008）: 中程度貧困地域に貧困家庭を集中させ、貧困率の非線形上昇で暴力犯罪が増加

ABM は self-reinforcing feedback loops を同定し、より良い政策設計を助ける。

### 既存の成功例
- **MIDAS (Models of Infectious Disease Agent Study)**: H5N1、H1N1 疫病応答計画で影響力。学校閉鎖・ワクチン配分の効果を検討（Epstein 2009）
- **UrbanSim**（Waddell 2002）: 都市交通・ライトレール・高速道路延長・土地利用規制への政策判断を誘導

---

## 4. High- vs Low-Dimensional Realism

ABM の抽象度は連続的:

| レベル | 特徴 | 典型例 |
|--------|------|--------|
| **抽象的** | agent は単一属性、決定的ルール、grid/torus 環境 | Schelling tipping model (1978)、Axelrod's PD tournament |
| **低次元現実性** | 1-2次元で実データに接地、他は stylized | Epstein et al. (2008) 疫病+行動適応、Todd & Billari (2003) 配偶者探索 |
| **高次元現実性** | 多属性、動的環境、多様な行動 | MIDAS、UrbanSim、Artificial Anasazi (Dean et al. 2000) |

### 選択基準

> A model's success is determined not by how realistic it is but by **how useful it is for helping understand the problem at hand**.

シミュレーションが有益な3つの場面:
1. ミクロ行動が既知 → 抽象/低次元で十分
2. 集合パターンが観察されており代替機構を探索 → 低次元が illuminating
3. 政策/予測のために系の挙動を探索 → 高次元が必要

---

## 5. Empirically Grounded ABM の要素

ABM に実データを接地できる側面:
- age-specific mortality, fertility, disease risk
- population size, demographic composition
- geographic boundaries, spatial relationships
- decision payoffs
- temporal granularity
- preferences, behavior, memory, environmental perception
- labor/marriage/housing market organization
- social network structure

### Incorporating Population Characteristics

- 個人属性: Census、IPUMS (5% サンプル, USA)、U.K. Sample of Anonymized Records
- 細かい地理識別子は不足（IPUMS PUMA ≈ 100,000人単位）
- 合成母集団生成 (Beckman, Baggerly, McKay 1996): 限界分布を満たす個人群を作る

### Incorporating Individual Behavior

- **離散選択モデル**（Discrete Choice Analysis, McFadden 1973）: ランダム効用の多項ロジット
- 行動パラメータを選好・機会から推定
- Bruch & Mare (2006) の居住移動モデル、Todd & Billari (2003) の結婚探索

---

## 6. Validation と Sensitivity Analysis

### モデル検証の2側面
- **Calibration**: モデルがデータを再現するようパラメータ推定
- **Validation**: モデル出力を独立データと比較

### Input Uncertainty: Monte Carlo sampling
- 入力パラメータの分布からサンプリング → 出力分布を評価
- Bruch (2014) の residential segregation での例

### Sensitivity Analysis
- 重要パラメータを 1 つずつ変化 → outcome の感度を測定
- Bayarri et al. (2007a, 2007b) の formal framework

### Validation Criteria（粒度別）

| 粒度 | 例 | 難しさ |
|------|---|-------|
| 高集約統計 | 全体のセグリゲーション index | 容易だが多モデルが合致しうる |
| 時系列集約統計 | Index の時間推移 | 中程度 |
| ミクロ統計 | 個人レベル軌跡 | 困難、データ要求が高い |

Berk (2008) の枠組みが **state of the art**。低粒度でのみ合致しても、メカニズムが正しい保証はない。

### Bruch (2013) の例
- ABM で between-race と within-race 所得不平等が residential segregation に及ぼす影響を探索
- **Offsetting effects**: within-race 不平等が高いと、between-race 不平等の変化が高所得端と低所得端で相殺
- 30年の Census データで固定効果モデルによる実証的裏付け

### Speed-Dating 実験（Todd et al.）
- ABM で indirect competition が mate choice を早めると予測
- 速度結婚実験でこの予測を確認

---

## 7. Future Directions

1. **行動経済学・認知科学の知見の取り込み**: 限定認知資源、認知バイアス、情報収集戦略、学習メカニズム
2. **動的な社会ネットワーク**: network を内生的 outcome として扱う。Padgett & Powell (2012) が先駆
3. **地理情報の統合**: 高速道路、鉄道、混雑交差点などの物理的障壁。携帯センサーデータ
4. **ソフトウェア・ドキュメンテーション**: Repast, NetLogo, MASON, Swarm, R, Python/Matlab。**UML (Unified Model Language)** の標準化推奨。OpenABM リポジトリでの共有。IPython notebook で再現性確保

---

## 8. ABM の典型的応用（本論文が引用）

| 応用 | 文献 |
|------|------|
| Residential segregation | Schelling 1971, 1978; Bruch 2013 |
| Cooperation evolution | Axelrod 1997, Axelrod & Hamilton 1981 |
| Disease spread | Epstein 2009 (MIDAS) |
| Urban planning | Waddell 2002 (UrbanSim) |
| Anasazi civilization | Dean et al. 2000 |
| Spanish flu | Epstein et al. 2008 |
| Marriage market | Todd & Billari 2003, Todd & Miller 1999 |
| Leaving unemployment | Hedström & Åberg 2005 |
| Game theory | Centola & Macy 2007, Willer et al. 2009 |

---

## 9. CS 222 での位置づけ

### Lecture 09: 行動モデルとエージェント設計の基盤

本論文は Lecture 09 "従来のエージェント・ベース・モデリングと LLM による発展" の系譜を繋ぐ:

- Schelling tipping (1971) → Axelrod PD → MIDAS → UrbanSim → **Generative Agents** (Park et al. 2023) → CS 222 の今

### Park氏の議論との接続

1. **Empirically grounded agent への要求**: Generative Agents は LLM でこの問題に新規解を提供 — ペルソナ・記憶・計画で agent に"現実性"を付与
2. **Low- vs High-dimensional realism** の選択: Generative Agents は高次元、Social Simulacra は中次元、古典 Schelling は低次元
3. **Mechanism-based explanation**: LLM エージェントでも "mechanism を surface させる" のが目的であるべき
4. **Validation の難しさ**: LLM agent の validation はさらに困難 — Bruch & Atwell の Berk (2008) 枠組みが参考

### 方法論的継承
- **Population initialization**: Social Simulacra の 1000 ペルソナ拡張（Census 風の合成母集団）
- **Sensitivity analysis**: Multiverse（同じ設計から複数実行）
- **Validation**: 人間との判別実験（SimReddit の 41% 誤認率）

### 本論文が提起するが LLM 時代にも未解決の問題
- Interdependent behavior の feedback を本当に捉えているか
- High-dimensional reality への overfit vs abstract insight のトレードオフ
- Documentation/reproducibility — LLM ランダム性の追跡可能性

---

## 10. 主要引用

### 古典的 ABM
- **Schelling, T. C. (1971)** "Dynamic Models of Segregation" *J. Math. Sociol.*
- **Schelling, T. C. (1978)** *Micromotives and Macrobehavior*
- **Axelrod, R. (1997)** *The Complexity of Cooperation*
- **Axelrod, R. & Hamilton, W. (1981)** *Science* 211 "The Evolution of Cooperation"
- **Miller, J. H. & Page, S. E. (2007)** *Complex Adaptive Systems*

### 社会学理論
- **Coleman, J. S. (1994)** — micro-macro problem
- **Granovetter, M. (1978)** — threshold models
- **Hedström, P. & Bearman, P. (2009)** *Oxford Handbook of Analytic Sociology*
- **Hedström, P. & Swedberg, R. (1998)** — mechanism-based explanation
- **Elster, J. (2007)** *Explaining Social Behavior*

### 実証 ABM 応用
- Epstein (2009) MIDAS
- Waddell (2002) UrbanSim
- Dean et al. (2000) Artificial Anasazi
- Todd & Billari (2003) mate search
- Hedström & Åberg (2005) empirically calibrated ABM
- Bruch & Mare (2006) residential mobility

### 方法論
- **Berk, R. (2008)** "How Can You Tell if the Simulations in Computational Criminology Are Any Good?" — validation framework
- **Grimm et al. (2006)** ODD (Overview, Design concepts, Details) documentation
- **Railsback & Grimm (2011)** *Agent-Based and Individual-Based Modeling*

---

## 11. 限界

- メタ・レビューなので個別モデルの詳細は浅い
- 2013年時点の執筆で、深層学習・LLM 時代以前
- 実証的 ABM の「成功例」は epidemiology と urban planning に偏重、他領域での普及は限定的
- validation criteria の粒度トレードオフは最終的に解けない（低粒度は説得力弱く、高粒度はデータ要求が高すぎる）

---

## 要点

1. **ABM の核心**: 個人のルールベース行動と環境の相互作用から、集合的な emergent パターンを生成 — micro-macro の明示的連結
2. **Interdependent behavior + feedback** を扱える点が標準統計モデルへの優位性。非線形関係と過程依存性を捉える
3. Realism のスペクトラム: 抽象（Schelling）→ 低次元（Todd & Billari）→ 高次元（MIDAS、UrbanSim）。**目的に応じて選ぶべき**
4. 実データ接地の戦略: Census/IPUMS による populations、discrete choice analysis による behaviors
5. Validation は粒度ごと（aggregate → trajectory → micro）。Berk (2008) 枠組み
6. Sensitivity analysis（Monte Carlo, parameter sweep）が必須。特に複雑モデルで
7. 成功例: MIDAS（疫病対応）、UrbanSim（都市計画）、Artificial Anasazi（考古学）
8. 未来方向: 行動経済学・認知科学の取り込み、動的ネットワーク、地理情報、ソフトウェア標準化。LLM 時代の Generative Agents はこの系譜の発展形
