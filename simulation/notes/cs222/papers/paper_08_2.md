# 08_2 — GroupLens: An Open Architecture for Collaborative Filtering of Netnews

## 書誌情報

- 著者: **Paul Resnick**（MIT Center for Coordination Science）／ **Neophytos Iacovou**（University of Minnesota, CS）／ **Mitesh Suchak**（MIT）／ **Peter Bergstrom**（University of Minnesota, CS）／ **John Riedl**（University of Minnesota, CS）
- 掲載: *CSCW '94: Proceedings of the 1994 ACM Conference on Computer Supported Cooperative Work*, October 22–26, 1994, Chapel Hill, NC
- DOI: 10.1145/192844.192905, ISBN 0-89791-689-1
- Lecture 08 の補足論文。**Collaborative Filtering（協調フィルタリング）** の原典

---

## 1. 研究問題

1994年当時、Usenet netnews は 8,000 超のニュースグループ、日量 100 MB 超、14万人以上が2週間で投稿という巨大な bulletin board system。

- 誰もが "signal-to-noise 比が低い" と嘆く
- 各ユーザーに興味のある記事を見つけるのは極めて困難
- Kill-file や moderated newsgroup では不十分

> How can we help a reader find the few valuable articles among the huge stream of available articles?

**中核アイデア**:

> People who **agreed in the past** are likely to **agree again in the future** on articles in the same newsgroup.

---

## 2. 中心主張（Abstract より）

> GroupLens is a system for collaborative filtering of netnews, to help people find articles they will like in the huge stream of available articles.

技術的構成:
- **News reader clients** が予測スコアを表示、ユーザーが記事を読んだ後評価を付与
- **Rating servers（"Better Bit Bureaus", BBBs）** が評価を集約・配布し、スコアを予測
- スコア予測は "過去に一致した人は将来も一致する" ヒューリスティック
- **pseudonym** でプライバシー保護、予測精度は落ちない
- アーキテクチャ全体がオープン — クライアントや BBB は独立に開発可能

---

## 3. 関連研究の分類（Malone et al. 1987 の枠組み）

### フィルタリング技法3分類

| 分類 | 根拠 | 例 |
|------|------|-----|
| **Cognitive / Content-based** | 記事テキストの内容 | Kill files（テキスト文字列）、Boolean queries、重みベクトル、Bayesian probability、genetic algorithms、relevance feedback（Salton & Buckley 1988, 1990） |
| **Social** | 人と人の関係、主観評価 | Tapestry（Goldberg et al. 1992）、moderated newsgroup（単一の moderator による評価） |
| **Economic** | 生産コスト・便益 | Stodolsky (1990) の招待付きジャーナル、cross-post への低優先度 |

### Collaborative Filtering の位置

**Social filtering の中でも最も有望**:
- 人間評価者は synonymy/polysemy/context の問題を持たない
- 品質・権威性・尊重さなどで判断可能

### 先行システムとの差分
- **Tapestry**（Goldberg et al. 1992, Xerox PARC）: 内部単一サイトでの評価共有、aggregate query 無し
- **Maltz (1994)**: 記事ごとに単一スコア集約。スケール良好だが個人カスタマイズ無し
- **GoodNews**（Suchak 1994）: 以前の Macintosh 版。ポジティブ endorsements のみ、実名

---

## 4. 設計目標（5 goals）

| 目標 | 内容 |
|------|------|
| **Openness** | 多様なニュースクライアント・BBB が相互運用可能 |
| **Ease of Use** | 評価の入力と予測の解釈が軽い |
| **Compatibility** | 既存 netnews 機構との互換 |
| **Scalability** | ユーザー増加で品質が上がり、速度が落ちない |
| **Privacy** | pseudonym でも評価が有効 |

---

## 5. アーキテクチャ

### 既存 Usenet

- 各サイトが news server を持ち、クライアントが接続
- 記事は globally unique ID を持ち、一度 post されたら不変
- サイト間でホップ伝播

### GroupLens の追加要素: Better Bit Bureau (BBB)

- **BBB** は評価を収集、他 BBB と netnews 経由で交換、予測スコアをクライアントに送信
- 評価は **専用の ratings newsgroup** に post される（Usenet の伝播機構を再利用）
- クライアントはローカル BBB にもリモート BBB にも接続可能 → 段階的展開が可能

---

## 6. 評価の形式

- 1〜5 の数値 + 任意で秒単位の閲覧時間
- ユーザーは pseudonym で提出（投稿時の本名と分離可能）
- 修正した3つのニュースクライアント:
  - **Macintosh: NewsWatcher**（Figure 3, 数値ボタンをクリック or 1-5 キーで入力）
  - **UNIX: Emacs Gnus**（数字直接入力）
  - **UNIX: NN**（'v' で rating mode → 1-5 or a-e 文字入力）

### 評価記事のフォーマット（Figure 4）

- 複数の rating を1本の記事にバッチ
- ヘッダ: From, Subject, Message-ID, Groups_Rated, Raters
- 本文1行が1評価: article_id, rater_pseudonym, rating, seconds, newsgroup

---

## 7. 予測アルゴリズム

### ピアソン相関係数ベース

ユーザー Ken, Lee, Meg, Nan の既読評価行列から、Ken に対する未読記事 6 のスコアを予測:

1. Ken と他者のピアソン相関 r_KL, r_KM, r_KN を計算
2. 記事6への他者の評価 r_L, r_M, r_N を相関で加重平均

例示:
- r_KL = −0.8（不一致）, r_KM = +1（完全一致）, r_KN = 0
- Ken の予測スコア: x̄_K + [Σ r_Kj (r_j − x̄_j)] / Σ |r_Kj|
- 記事6: Meg 評価5、Lee 評価1 → Ken 予測 = 3 + (1×2 + (−0.8)×(−2))/(1 + 0.8) = 3 + 2.0 = **約 4.56**

### 複数の手法を実装比較
- Reinforcement learning（[12] Maes & Kozierok）
- Multivariate regression
- Pairwise correlation coefficients（linear or squared error minimizing）

### ロバスト性
- 2ユーザーが perfectly correlated でも片方が 3–5 のみ, 他方が 1–3 のみで可 → 5 → 3 を予測
- 逆方向誤解（1 を「良い」と思う人）→ 負相関で -1 扱い
- → ユーザー教示: "Rate such that you wish GroupLens had predicted this score for you."

### Newsgroup 分割
- 個人間相関は領域特異: 技術記事で一致してもジョークで一致しないかも
- BBB は **各 newsgroup 内で別の相関行列**を保持

---

## 8. クライアント UI の設計

### Figure 6: NN client
- 3列目にライン数、4列目に A-E の letter grade 予測（5=A, 4=B...）
- 評価がない記事はスコア表示なし

### Figure 7: NewsWatcher
- 予測スコアを **棒グラフ**として表示（高い=長い）
- スレッドごとにクラスタリング、トライアングルで展開

### Modified Gnus
- スレッドをスレッド内最大予測スコアで**ソート**
- スレッド内は時系列順

各クライアントの UI は既存パラダイムを尊重。

---

## 9. 技術評価

### 初期パイロット
1. **Schlumberger 研究所**（7人、以前の endorsement 版）: 7人では collaborative filtering の benefit が不十分
2. **University of Minnesota**（4人、GroupLens 初期版）:
   - 起動時に start-up interval（予測が不安定な期間）が必要
   - 全員が予測スコアが最終的に好みにマッチしたと報告
   - rec.arts.movies のような高流量ニュースグループでは 4人だとカバー不能

### スケール議論

**計算・ネットワーク的負荷**:

| cluster size | 日量 rating traffic |
|-------------|---------------------|
| 100 users | 1 MB |
| 50,000 users | 100 MB |
| 1,000,000 users | 10 GB |

(比較: 当時の netnews 総流量は約 100 MB/day)

スケール対策:
- BBB を BBB 間でクラスタリング（地理・関心別）
- 記事ごとの rater 数上限
- バッチ化（session 単位）

### 次の pilot（MIT & Minnesota, 夏 1994 予定）
- 参加者に共通 training set を rating させ、bias なしにアルゴリズム比較可能
- 目的: ① unexpected scaling 問題の発見、② 代替アルゴリズム比較用のデータセット生成

---

## 10. 社会的含意（著者の予測）

### Moderated Newsgroup を代替
- 個人ごとに異なる "moderator" 相当が選べる
- moderator + GroupLens の二重フィルタも可

### Newsgroup Splits の減少
- 現在は rec.sport.football が .bills, .college などに分割される
- GroupLens では暗黙の peer group 形成が splits を不要に
- **Cross-pollination**: Bills ファンの良記事が Cowboys ファンにも届く可能性

### Kill Files の代替
- 手動 kill-file より、peer group の低評価が自動的に kill と同等に機能

### Incentives 問題
- 評価はコストゼロではない (altruism/guilt が前提)
- 社会的最適より少ない評価が生成される可能性
- **非対称性**: 高評価されすぎた記事は後続に読まれて下方修正されるが、低評価された記事は誰も読まないため埋もれたまま
- → 外部インセンティブ（金銭、名声、BBB access）の必要性

### Global Village の分断
- 共有経験としての newsgroup が失われる懸念
- "fracture into tribes" vs "permeable subgroups" — short-lived interest groups なら後者か

---

## 11. Conclusion

> Shared evaluations are useful in all sorts of activities. We ask friends, colleagues, and professional reviewers for their opinions about books, movies, journal articles, cars, schools, and neighborhoods.

本論文のコアは:
1. **単一の数値評価**でまずは始める
2. クライアントは簡単に数値評価できる UI を持つ
3. ピアソン相関で重み付け → 未観測セルの予測
4. pseudonym で信頼性は維持される
5. **開放アーキテクチャ**で新クライアント・BBB は自由に参加

> Right now, people read news articles and react to them, but those reactions are wasted. **GroupLens is a first step toward mining this hidden resource.**

---

## 12. CS 222 での位置づけ

### Lecture 08: 個人モデルの系譜（協調フィルタリング）

Park氏は Lecture 08 で **"個人を推論する / 個人の嗜好を予測する"** 系譜を扱う。本論文はその原点の一つ:

- **個人モデルの構築**: Ken の嗜好を直接 polling せずに、**他人との類似性パターン**から推定
- **Agent Bank / Generative Agents との類比**:
  - GroupLens: 過去評価 → 相関 → 未来評価予測
  - Generative Agents (Park 2023): 過去観察 → memory → 計画・行動予測
  - 両者とも「観察された行動系列から未観察状況での行動を予測する」枠組み

### キーコンセプトの CS 222 への接続

1. **"People who agreed in the past will agree again"** 仮説 → LLM エージェント設計における consistency 仮定
2. **Pseudonym で情報有効性が保たれる** → プライバシー保護エージェント
3. **ピアソン相関ベース類似性** → LLM の埋め込みベース類似性（より高次元）
4. **"Eternal September" 問題（new user の cold start）** → Social Simulacra, Generative Agents の初期化問題

### 歴史的重要性
- Recommender systems 分野の誕生点の一つ
- Netflix Prize (2006-2009), Amazon 推薦, YouTube などの直接的な知的祖先
- 被引用 3,475+ 回（2026年3月時点）

### Park氏の議論との関係
- 集団の総体から個人を浮かび上がらせる手法（collective → individual predictions）
- Social Simulacra は **集団をシミュレート**、GroupLens は **既存集団データから個人を予測**
- Generative Agents は両者を統合：集団的相互作用の中で個人を生成

---

## 13. 主要引用

### 本論文が引用
- **Goldberg, Nichols, Oki, Terry (1992)** *CACM* 35 "Using Collaborative Filtering to Weave an Information Tapestry" — Tapestry
- **Malone, Grant, Turbak, Brobst, Cohen (1987)** *CACM* 30 "Intelligent Information Sharing Systems"
- Salton & Buckley (1988, 1990) — 情報検索の古典
- Maes & Kozierok (1993) — learning interface agents
- Kumon (1992) — CSCW '92 keynote（reputation への着想源）

### 本論文を引用した後続研究（選抜）
- Sarwar et al. (2001) item-based CF
- Breese, Heckerman, Kadie (1998) empirical analysis of CF algorithms
- Herlocker, Konstan, Borchers, Riedl (1999) explanation-aware CF
- Amazon, Netflix などの商用レコメンダー

---

## 要点

1. **Collaborative filtering の原典** — 集団の評価パターンから個人へのスコアを予測する枠組み
2. **"People who agreed in the past will agree again"** という単純なヒューリスティックをピアソン相関で実装
3. **Open architecture**: netnews 互換、ニュースクライアント・BBB は独立開発可能
4. **Pseudonym でもスコア予測は有効** — プライバシーと有用性の両立
5. 5つの設計目標: Openness, Ease of Use, Compatibility, Scalability, Privacy
6. 3つの Modified clients（NewsWatcher, Gnus, NN）で UI パラダイムを尊重した統合
7. 社会的含意の予測: moderated newsgroup 代替、kill-file 代替、newsgroup split の減少、global village の分断リスク
8. CS 222 の Lecture 08 では「集団的観察から個人を推論する」系譜の源泉として位置づけられ、Generative Agents など LLM エージェント設計の知的祖先
