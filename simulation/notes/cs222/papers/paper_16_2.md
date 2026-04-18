# 16_2 — Learning Production Functions for Supply Chains with Graph Neural Networks

## 書誌情報

- 著者: **Serina Chang**¹, **Zhiyin Lin**¹, **Benjamin Yan**¹, **Swapnil Bembde**², **Qi Xiu**², **Chi Heem Wong**¹ᐟ², **Yu Qin**²ᐟ³, **Frank Kloster**², **Xi Luo**², **Raj Palleti**¹ᐟ², **Jure Leskovec**¹
  - ¹ Stanford University, Department of Computer Science
  - ² Hitachi America, Ltd.
  - ³ Tulane University
- arXiv: 2407.18772v3, 2025年2月24日改訂
- 採択: **AAAI 2025**
- コード・simulator: https://github.com/snap-stanford/supply-chains/tree/tgb
- CS 222 Lecture 16（シミュレーションの産業応用・大規模ネットワーク動態）の補足論文

---

## 1. 研究問題

グローバル経済はサプライチェーンネットワーク（nodes=firms、edges=transactions）を通じた商品の流れに依存。disruption は数兆ドルの損失（Baumgartner, Malik, Padhi 2020）と国家安全保障リスク（White House 2021）を引き起こす。

既存手法の限界:

- Mechanistic model: aggregate counts（country-level production）ですら適合困難（Inoue & Todo 2019）
- Graph Neural Networks (GNNs): supply chain 応用は静的のみ（Aziz 2021, Kosasih & Brintrup 2021）
- Temporal GNNs (Huang 2023): 一般的だが **production functions** の学習に未対応

研究問題:

> How can we develop temporal GNNs for supply chains that can (1) **infer hidden production functions** (inputs → outputs) and (2) **forecast future transactions** accurately, even under shocks?

---

## 2. 中心主張・主要発見

### 2.1 新しいグラフ ML 問題: Temporal Production Graphs (TPGs)

- 時変エッジを持つ有向グラフ
- 各ノードの in-edges は **隠れた production function** で out-edges に変換される
- サプライチェーンが典型だが、代謝経路（酵素 → 代謝物）や組織内チーム間相互作用にも適用可

### 2.2 提案モデル

**Inventory module** = 新規の発明:
- 各 firm の inventory を明示的に表現
- 買収 (buy) と消費 (consume) で更新
- **Attention weights** α_{p,p'} で「product p を作るのに p' の何単位が必要か」を学習

これを既存の temporal GNN に attach:
- **SC-TGN**（Temporal Graph Network 拡張, Rossi 2020 ベース）
- **SC-GraphMixer**（GraphMixer 拡張, Cong 2023 ベース）

### 2.3 結果

- **Production function learning**: 強baseline より **6%〜50% MAP 改善**
- **Future edge prediction**: 11%〜62% MRR 改善
- SS-shocks（供給ショック時）、SS-missing（20% firm データ欠損時）でも頑健

---

## 3. 方法・実装

### 3.1 TPG の定式化

- Heterogeneous temporal graph G_txns = {N, E}
- ノード: n firm nodes + m product nodes
- ハイパーエッジ: e(s, b, p, t) = supplier s → buyer b が product p を時刻 t に取引
- 未観測の production graph G_prod: p_1 が p_2 を作るのに必要なら p_1 → p_2

目的:
1. **G_prod の推定**
2. **G_txns の将来エッジ予測**

### 3.2 Inventory Module

Firm i の時刻 t の inventory x_i^(t) ∈ R_+^m を保持。

**Buy**（観測可能）:
buy(i, p, t) = Σ_{e(s,i,p,t) ∈ E} amt(s, i, p, t)

**Consume**（推論）:
cons(i, p, t) = Σ_{e(i,b,p_s,t) ∈ E} α_{p_s p} · amt(i, b, p_s, t)

ここで α_{p_s p} は **learnable attention weight**: product p_s を作るのに p が必要な量。

**Inventory 更新**:
x_i^(t+1) = max(0, x_i^(t) + b_i^(t) - c_i^(t))

### 3.3 Inventory Loss（鍵）

ℓ_inv(i, t) = λ_debt · Σ max(0, cons - x_i^(t)) - λ_cons · Σ cons

- **Debt penalty**: 在庫不足で消費すると penalty（firms can't consume what they never received）
- **Consumption reward**: 自明解 (all α=0) を防ぐ
- 実験的に λ_debt ≈ 1.25 × λ_cons が最良

### 3.4 Attention Weights

**Direct**: α_{p_1 p_2} を直接学習

**Embedding-based**:
α_{p_1 p_2} = ReLU(z_{p_1}^T W_att z_{p_2} + β_{p_1 p_2})

GNN の product embedding を base rate に使い、β は L2 regularized adjustment。**スパースな実データ**で product pair 情報共有に有効。

### 3.5 SC-TGN (Temporal Graph Network 拡張)

オリジナル TGN (Rossi 2020) から:

1. **Hyperedges**: 3 nodes (supplier/buyer/product) に message 送信
2. **Edge weight 予測**: existence と weight の2段 decoder
3. **Update penalty**: memory update の regularization
4. **Learnable initial memory**: 新ノードの表現力向上
5. **Historical negatives で training**: テスト一致（+10 MRR）

### 3.6 SC-GraphMixer (GraphMixer 拡張)

GraphMixer (Cong 2023) は MLP-only simple architecture。拡張:

1. **Hyperedges**: 3 nodes の embedding を concat して decoder へ
2. **Learnable node features**

### 3.7 Decoder

Edge existence と weight で別々の MLP:

y^(s, b, p, t) = MLP([z_s^(t) | z_b^(t) | z_p^(t)])

- Existence: softmax + cross-entropy (positive vs 18 hard negatives)
- Weight: RMSE on log-scaled transaction amounts

Inventory module が attach されると、y^ に:
- **Penalty** (edge existence): supplier s が必要 parts を持たない (s, b, p, t) を減点
  pen(s, b, p, t) = -Σ max(0, α_{pp'} - x_s^(t)[p'])
- **Cap** (edge weight): s が生産可能な最大量で capping

---

## 4. データ

### 4.1 Real-world データ（TradeSparq）

Hitachi 協業で提供:

1. **Tesla dataset**: EV + EV parts を扱う Tesla の makers、直接 suppliers/buyers、そして suppliers の suppliers、2019/1/1〜2022/12/31 の全取引
2. **Industrial Equipment Dataset (IED)**: microscopes + 特殊 analytical/inspection equipment、maker と直接取引先、2023 年全取引
   - 微鏡の ground-truth parts (HS codes で表現) も取得

Harmonized System (HS) codes で product 分類、国際的標準。

### 4.2 Synthetic データ（SupplySim）

新規オープンソース simulator（この論文の貢献の一つ）:

- **3 settings**: SS-std（高供給）、SS-shocks（供給ショック）、SS-missing（20% firm データ欠損）
- Power law degree distribution、community structure、low clustering を real に match（Fig. 2）
- ARIO model (Hallegatte 2008) に基づく transaction 生成 — 経済学で広く使用される agent-based model
- Tier 構造: 原材料 → 完成品

**構築手順**:
1. Products を tiers に分割、2D position を Uniform(0, 1) からサンプル
2. 各 tier の product に前 tier の parts を距離の逆数比例で割当
3. Firms も 2D position を持つ、2 連続 tiers に制約
4. Firm-product 関係を preferential attachment で構築 → power law
5. ARIO でタイムステップごと transaction 生成

---

## 5. 結果

### 5.1 Production Learning (Table 1, MAP)

| Method | SS-std | SS-shocks | SS-missing | IED |
|--------|--------|-----------|------------|-----|
| Random | 0.124 | 0.124 | 0.124 | — |
| Temporal correlations | 0.745 | 0.653 | 0.706 | 0.060 |
| PMI | 0.602 | 0.602 | 0.606 | 0.128 |
| node2vec | 0.280 | 0.280 | 0.287 | 0.175 |
| **Inventory (direct)** | **0.771** | 0.770 | 0.744 | 0.127 |
| **Inventory (emb)** | **0.790** | **0.778** | **0.755** | **0.262** |

- Embedding-based inventory が consistent に best
- **Shocks に robust**: SS-std 0.790 → SS-shocks 0.778（わずか -1.5%）。一方 temporal correlations は 0.745 → 0.653（-12.3%）
- **Missing に robust**: -3.5 MAP points (-4.4%)
- 実データ IED で **4× 改善** (0.060 → 0.262)

### 5.2 Future Edge Prediction (Table 2, MRR)

| Method | SS-std | SS-shocks | SS-missing | Tesla | IED |
|--------|--------|-----------|------------|-------|-----|
| Edgebank (binary) | 0.174 | 0.173 | 0.175 | — | — |
| Edgebank (count) | 0.441 | 0.415 | 0.445 | 0.131 | 0.164 |
| Static | 0.439 | 0.392 | 0.442 | 0.189 | 0.335 |
| Graph transformer | 0.431 | 0.396 | 0.428 | 0.321 | 0.358 |
| SC-TGN | 0.522 | 0.449 | 0.494 | 0.507 | **0.613** |
| **SC-TGN+inv** | **0.540** | **0.461** | 0.494 | **0.820** | **0.842** |
| SC-GraphMixer | 0.453 | 0.426 | 0.446 | 0.818 | 0.841 |
| SC-GraphMixer+inv | 0.497 | 0.448 | 0.446 | 0.690 | 0.791 |

- **SC-TGN+inv** が一般に最強。実データで特に劇的（Tesla 0.507 → 0.820、IED 0.613 → 0.842）
- Shock 時が最難
- Inventory を付けると **edge prediction も向上**（時にほぼ変化なし or 小改善、劣化なし）

### 5.3 Case Study: Shock 伝播の生成 (Table 3)

SS-shocks で最後 (t=138) まで訓練、t=139〜145 の 7 timesteps を predict-and-update で生成:

| t | AUROC (all negs) | AUROC (train negs) |
|---|------|------|
| 139 | 0.996 | 0.824 |
| 140 | 0.986 | 0.733 |
| 141 | 0.966 | 0.721 |
| 142 | 0.938 | 0.637 |
| 145 | 0.936 | 0.634 |

- 7 timesteps 先でも AUROC > 0.93（all negatives）
- Disruption 中の timestep でも予測可能

---

## 6. Discussion / 限界

### 6.1 産業影響

Hitachi America 協業から生まれた:
- **Supply chain visibility 向上**: 自社の直接 suppliers だけでなく、全生産プロセスを理解
- **Bottleneck 特定** (Aigner & Chu 1968, Coelli 2005)
- **Demand forecasting** (Seyedan & Mafakheri 2020)
- **Early risk detection** (Sheffi 2015)
- **Inventory optimization** (Vandeput 2020)

### 6.2 オープンソース貢献

- **SupplySim** を公開
- 独自データは共有不可だが、synthetic data で research community の ML on supply chains 研究を enable

### 6.3 限界

- 各 product に **1つ**の production function のみ仮定（substitute products、firm-specific graphs への拡張は future work）
- Inventory estimation は complete transaction data を前提（systematically missing では underestimate）
- Theoretical: identifiability 条件、causal inference との接続は未解決

### 6.4 Future Work

- 他の TPG ドメイン（代謝経路、組織チーム間）
- Identifiability の理論的結果
- Causal discovery との接続

---

## 7. CS 222 での位置づけ

### Lecture 16: シミュレーションの産業応用

Lecture 16 で Park 氏は**シミュレーションの産業・政策応用**を扱う。本論文は:

1. **産業協業**（Hitachi）の成功例
2. **オープンソース simulator 公開**（SupplySim）で community 貢献
3. **Mechanism + ML の融合**: ARIO (mechanistic) と GNN (ML) の補完
4. **Domain-specific inductive bias**: inventory module は「firms can't consume what they never received」という**物理的制約**をモデルに組み込む
5. **Temporal Production Graph** という**新しい問題クラス**を提起

### 16_1 との連続性

- 筆頭著者 **Serina Chang** は 16_1 (COVID mobility) の筆頭著者の一人でもある
- 方法論的連続性:
  - **IPFP** を両者で使用（16_1 では mobility matrix 推定、16_2 では inventory/transaction 推定）
  - **Mechanism + data-driven** アプローチ
  - **Counterfactual / shock analysis**
  - **Large-scale、multi-scale** なネットワーク

### Park 氏の Stanford lineage

- Jure Leskovec 研（両論文の senior author）は CS 222 全体の intellectual backbone の一部
- Generative agents (Park) と network simulations (Chang/Leskovec) は**異なるスケール**で社会をモデル化するアプローチ

### Domain-specific Inductive Bias の教訓

本論文の Inventory module は、**「ドメイン知識を ML モデルに埋め込む」**典型例:
- Physics simulation (Wang, Walters, Yu 2021)
- Epidemiological forecasting (Liu et al. 2024)
- Supply chains (本論文)

CS 222 でこのアプローチは Park 氏が強調する — **"generative AI + domain mechanism"** の融合が次世代シミュレーションの鍵。

### Schelling (02_2) との対応

- Schelling: individual → aggregate の非自明な関係
- Chang et al.: individual transaction → aggregate production network dynamics
- 両者とも**micro-level 制約（inventory, individual motive）からの emergent aggregate behavior**

---

## 8. 主要引用

### Temporal GNNs

- **Rossi et al. 2020** (TGN): Temporal Graph Networks
- **Cong et al. 2023** (GraphMixer, ICLR): MLP-only simple temporal model
- **Huang et al. 2023** (Temporal Graph Benchmark, NeurIPS)
- **Kazemi et al. 2020**: representation learning for dynamic graphs survey

### Supply Chain Networks

- **Fujiwara & Aoyama 2010** (*European Physical Journal B*): nationwide production network の大規模構造
- **Acemoglu et al. 2012** (*Econometrica*): "Network Origins of Aggregate Fluctuations"
- **Carvalho & Tahbaz-Salehi 2019** (*Annual Review of Economics*): production networks primer
- **Carvalho et al. 2021** (*QJE*): supply chain disruptions from Great East Japan Earthquake
- **Hallegatte 2008** (ARIO model): Hurricane Katrina economic cost

### Supply Chain ML

- **Aziz et al. 2021, Kosasih & Brintrup 2021, Wasi et al. 2024**: 静的 GNN for supply chains
- **Baryannis, Dani, Antoniou 2019**: ML for supply chain risks

### COVID-19 supply chain

- **Guan et al. 2020** (*Nature Human Behaviour*): global supply-chain effects of COVID-19
- **Inoue & Todo 2020** (*PLoS ONE*): megacity lockdown propagation
- **Li et al. 2021**: ripple effect of disruptions

### Data sources

- **TradeSparq**: 60+ 国の customs declarations, bills of lading
- **Harmonized System (HS) codes**

### 関連 Stanford 論文

- **Chang et al. 2021** (*Nature*, 16_1): COVID mobility networks
- **Chang et al. 2023** (AAAI): geographic spillover effects
- **Chang et al. 2024** (ICML): inferring dynamic networks from marginals — IPFP extension

---

## 9. 要点

1. **Temporal Production Graphs (TPGs)** という新しい graph ML 問題クラスを提起。nodes の in-edges が未観測の production function で out-edges に変換される
2. **Inventory module** = この論文の発明。firm の inventory を明示表現し、attention weights で production function を学習、特殊 loss で debt penalty + consumption reward
3. SC-TGN（TGN 拡張）と SC-GraphMixer（GraphMixer 拡張）に inventory module を attach
4. **Production learning**: ベースラインより **6-50% MAP 改善**。embedding-based attention が consistent に最良
5. **Future edge prediction**: **11-62% MRR 改善**。SC-TGN+inv が一般に最強、Tesla 0.507→0.820、IED 0.613→0.842
6. **Shock・missing data に robust**: ground-truth inventory logic に基づくため
7. **SupplySim** オープンソース simulator 公開: power law、community structure、low clustering を real データに match、ARIO ベース
8. Hitachi America 産業協業の成果。CS 222 Lecture 16 で**domain-specific inductive bias を ML に埋め込む成功例**として、また Stanford Leskovec 研系譜（16_1 と連続）として引用される
