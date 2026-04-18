# 11_2 — The Nash Equilibrium: A Perspective

## 書誌情報

- 著者: **Charles A. Holt**（University of Virginia, Department of Economics, Charlottesville, VA 22904-4182）／ **Alvin E. Roth**（Harvard University, Department of Economics and Harvard Business School, Cambridge, MA 02138）
- 掲載: *Proceedings of the National Academy of Sciences (PNAS)* Vol. 101, No. 12 (March 23, 2004), pp. 3999–4002
- 編集: Vernon L. Smith（George Mason University）、2004年1月28日受理
- カテゴリ: Perspective — PNAS の "landmark papers" シリーズ
- 資金: NSF Infrastructure Grant SES 0094800
- Lecture 11 の補足論文。**Nash 均衡概念の原典**（Nash 1950）の半世紀後の回顧

---

## 1. 研究問題

1950 年、John Forbes Nash, Jr.（プリンストン大学院生、数学者）が PNAS に **一頁の論文**を発表。n 人ゲームの均衡概念を定義・特徴づけた。この概念は後に "Nash equilibrium" と呼ばれ、経済学ほか行動科学全般に革命をもたらした。

> Indeed, **game theory, with the Nash equilibrium as its centerpiece, is becoming the most prominent unifying theory of social science.**

本 Perspective は、Nash の貢献の**歴史的文脈**、その**定義の核心**、**実験経済学・市場設計・進化動力学などへの波及**、そして **quantal response equilibrium など後続拡張**を概観。

---

## 2. Nash 以前の状況

- **von Neumann, J.（1928）**: Math. Annal. の最小最大定理（two-person zero-sum のみ）
- **von Neumann & Morgenstern (1944)** *Theory of Games and Economic Behavior*: ゲーム理論の基盤。均衡概念は 2 人ゼロ和に限定
- **Cooperative games**: 契約が外部強制力で enforce 可能
- **Noncooperative games**: 外部強制力なし、均衡合意のみ sustainable

von Neumann の反応: Nash の研究に対し "polite but not enthusiastic"（Nash 自身が後に著者に伝えた回想として、von Neumann は "European gentleman" だったが熱心な支持者ではなかった）。

---

## 3. Nash 均衡の定義（1950年論文 p. 48-49 からの引用）

### ゲームの枠組み

> One may define a concept of an **n-person game** in which each player has a finite set of **pure strategies** and in which a definite set of payments to the n players corresponds to each n-tuple of pure strategies, one strategy being taken for each player.

### 均衡点の定義

> Any n-tuple of strategies, one for each player, may be regarded as a point in the product space obtained by multiplying the n strategy spaces of the players. One such n-tuple counters another if the strategy of each player in the countering n-tuple yields the highest obtainable expectation for its player against the n−1 strategies of the other players in the countered n-tuple. **A self-countering n-tuple is called an equilibrium point.**

すなわち、**各プレイヤーの戦略が他のプレイヤーの戦略に対する最適応答（best response）**である点。

### 混合戦略

- **Mixed strategies**: 決定の確率分布（例: 無作為監査する検査官、ブラフする poker player）
- 代替解釈: 各役割に無作為マッチした**集団**の中での割合分布

### 核心的特徴

> A Nash equilibrium is a set of strategies ... that has the property that each player's choice is his best response to the choices of the n−1 other players. **It would survive an announcement test**: if all players announced their strategies simultaneously, nobody would want to reconsider.

= **announcement test**（公表テスト）。

---

## 4. Nash 均衡の3つの解釈

### (1) Advice（助言としての解釈）

助言がすべてのプレイヤーに与えられるとき、**均衡でない助言**は、少なくとも1人のプレイヤーが逸脱したほうがよい状態を生む。均衡助言のみが「全員が受諾するに足る」。

### (2) Prediction（動的調整の安定点）

完全合理性を仮定せずに、プレイヤーが他者の戦略に適応して戦略を調整する**動的調整過程の安定点**として解釈。

**Biology での応用**:
- 混合戦略を**集団内の戦略分布**と解釈
- payoff を inclusive fitness の変化と解釈
- 動力学を**集団動力学**（population dynamics）と解釈
- → **Evolutionary game theory**（Maynard Smith 1974, Hofbauer & Sigmund 1988, Weibull 1995）

合理性仮定なし、単純な self-interested dynamics のみ。

### (3) Self-enforcing Agreement（自己強制的合意）

外部強制力なしの合意が維持される条件 = **自己利益に従っても遵守する**こと。合意が Nash 均衡なら、他者が守る限り自分も守る方が得。

### 協力 vs 非協力ゲームの区別の希薄化

**Nash program**: 強制力機構をゲームモデル内に含めれば、全ゲームを non-cooperative にモデル化可能。

---

## 5. 存在証明と数学

- Nash は **Kakutani (1941)** の fixed point theorem で存在を証明
- Kakutani 自身は von Neumann の 1930年代の経済学的議論に触発された（Nash の私信）
- この証明技法はのちに経済学で標準化（競争均衡を予想価格ベクトルの fixed point として扱う）

---

## 6. Nash の受賞とその拡張

### 1994 Nobel Economics Prize
- **John Nash**: Nash equilibrium
- **John Harsanyi**: games of incomplete information への拡張（players が他者の preferences/choices を知らない）
- **Reinhard Selten**: 均衡精緻化（refinement）— 完全合理的プレイヤーへの助言として Nash 均衡は必要条件、十分ではない。superfluous equilibria を除去

### その他の拡張
- **Correlated equilibrium**（Aumann 1974）: 共同無作為戦略、グループ間調整を許容

---

## 7. Nash 均衡と社会的ジレンマ

### Prisoner's Dilemma の起源（1950年1月）

**Melvin Dresher & Merrill Flood** が RAND Corporation で実験として考案。目的は **"Nash equilibrium は必ずしも良い行動予測ではない"** を示すこと。

- 各プレイヤーは "cooperate" or "defect" を選択
- Payoff 構造: 相手の選択に対する best counter は常に "defect"
- しかし (cooperate, cooperate) の方が (defect, defect) より両者 better off
- → Nash 均衡 = (defect, defect) = 集団的最適でない

### Tucker の物語化
Nash の論文指導教員 Albert Tucker がスタンフォード心理学科向け講演準備中に RAND の blackboard で payoff を見て、**検察官と2囚人の有名な story** を考案（1950）。

### 実験データ
実際の実験（Flood 1958、Rapoport & Chammah 1965、Axelrod 1984）では、プレイヤーは **ある程度 cooperation に成功**し、均衡から逸脱する。Raiffa も 1950 年に独自実験したが未発表。

### 社会的ジレンマの広範性
- 生態系破壊（Hardin 1968 "The Tragedy of the Commons"）
- 軍拡競争
- Traveler's Dilemma（Goeree & Holt 1999）

ゲームが Nash 均衡にない **"unstable"** 領域にあるとき、cooperation の維持は困難。

---

## 8. 市場・社会制度の設計

### Elinor Ostrom (1998)
APSA 大統領講演で「社会制度は PD を cooperation が均衡となるゲームに変える」ことを論じた。

### 労働市場の Unraveling
federal appellate court clerks 市場で、裁判官が早期オファーを出し合い、**法学生が 1 年生成績のみで 2 年前にオファーを受ける**状況まで悪化。2003年には1年間のモラトリアム導入。

### Clearinghouse 設計
- 医学生のマッチング（US 1940年代、UK 1960年代）
- **National Resident Matching Program (NRMP)**（Roth & Peranson 1999）: 応募者が真の選好を提出するのが Nash 均衡になるよう設計
- Roth (1984, 1990): matching markets の安定性分析

### Auction 設計
- Vickrey (1961) で 1996 Nobel
- Milgrom (2004) *Putting Auction Theory to Work*
- Wilson (2002) — オークション設計の実用化
- 「auction rules の違いによる均衡の違い」を Nash 均衡分析で明らかにする

---

## 9. 実験経済学との接続

### Nash の関与
Nash 自身が初期の経済実験に参加（Kalisch, Milnor, Nash, Nering 1954）。

### Vernon Smith (1962, 2002 Nobel)
- 小人数・不完全情報でも competitive outcomes が実現可能であることを示す実験
- Experimental economics 誕生

### Kahneman と Smith の 2002 年 Nobel
経済学と心理学の相互作用を認識した象徴的受賞。

### Kagel & Roth (1995) *Handbook of Experimental Economics*
実験ゲーム理論の集大成。

### 教室実験
- Internet の普及で大規模なゲーム実験が容易に
- veconlab.econ.virginia.edu — 30種類のゲーム、オークション、市場を教室で実行可能

---

## 10. Learning Models と Stochastic Equilibrium

### Nash 均衡が予測として失敗する場面
- ゲームの**初回**: 学習前は非均衡行動が一般
- 繰り返しで均衡に収束することもあればしないことも

### Learning Models
- **Fictitious play**（Fudenberg & Kreps 1993）: 相手の過去行動の頻度から最適応答
- **Reinforcement learning**（Erev & Roth 1998）: 過去 payoff に応じた確率調整
- **Fudenberg & Levine (1998)** *Learning in Games*

### One-shot games と Introspection
軍事・法律・政治の多くの戦略的場面は一回限り。**introspection**（他者が何を考えるかを思考）による learning。**Noisy introspection models**（Goeree & Holt 1999, 2003）で非 Nash 行動を説明。

### Quantal Response Equilibrium (QRE)
**McKelvey & Palfrey (1995)**:
- プレイヤーの応答は expected payoff 差が大きいときは sharp、小さいときは random
- Nash 均衡は noise → 0 の極限
- 場合によっては Nash 予測と逆側に振れることも

### Stochastic Learning
Capra, Goeree, Gomez, Holt (2002), Goeree & Holt (2002) など。

### Fairness と非利己的選好
小人数交渉実験（bargaining）で **fairness への関心**が重要。Bolton & Ockenfels (2000), Fehr & Gächter (2002) *Nature*, Nowak & Sigmund (1998) *Nature* — 進化的説明と合流。

---

## 11. Nash's Contributions in Perspective（総括）

> In the last 20 years, the notion of a Nash equilibrium has become a required part of the tool kit for economists and other social and behavioral scientists, so well known that it does not need explicit citation, any more than one needs to cite Adam Smith when discussing competitive equilibrium.

- **Nash 均衡 = Adam Smith の competitive equilibrium に並ぶ基本概念**
- 学生は Nash の名前を他のどの経済学者よりも多く耳にする
- 小人数・小グループの相互作用分析では **"the place to begin and sometimes end"**

### 今後50年の課題（著者の展望）

1. より多様で現実的な**個人行動モデル**の取り込み
2. 解析・実験・計算手法の併用による複雑戦略環境への対応

---

## 12. CS 222 での位置づけ

### Lecture 11: 均衡、予測、社会的ジレンマ

Park氏は Lecture 11 で **均衡概念の認識論**を論じる。Schelling (*Micromotives and Macrobehavior*, 02_2) と Nash の併読で:

- **Schelling の均衡定義** (p. 26): "An equilibrium is simply a result. It is what is there after something has settled down, if something ever does settle down."
- **Nash の均衡定義**: "best response to best response" の fixed point
- **両者ともに 均衡 ≠ 望ましい** — Schelling の "hanged man's body" と Nash の Prisoner's Dilemma は同じ警告

### Park氏の議論との接続

1. **Social Simulacra のシミュレーション結果は "均衡" か?**
   - 生成エージェントは学習・適応する → 動的な "approximate equilibrium" に達する可能性
   - LLM エージェントの均衡は Nash 的か QRE 的か — noise 込みの Quantal response に近い可能性
2. **PD と social dilemmas の生成**:
   - SimReddit（Park 2022）で **トロール挙動**が生成される = 社会的ジレンマの emergent な現れ
   - モデレーション rules 追加 = Ostrom 的 "institutional design" の類比
3. **Market design との parallel**:
   - NRMP のような matching と、Social Simulacra の参加者ペルソナの matching
   - Mechanism design の視点で生成エージェント環境を設計

### シミュレーション・生成エージェントへの含意

- **Nash 均衡 = "announcement test" に耐える点**: 生成エージェントも "show me your plan" に耐えうる戦略を出すか?
- **Multiverse (Park 2022) と equilibrium selection**: 複数の Nash 均衡がある場合、どれが実現するか — シミュレーションは**選択問題**の探索道具
- **Learning dynamics と generative agents**: agents の reflection + planning は fictitious play + reinforcement learning の LLM 版

### 関連する CS 222 の他論文
- **02_2 Schelling**: 均衡 ≠ 望ましい、Harvard ダイニングホール例は PD 型均衡
- **03_1 Social Simulacra**: 社会的ジレンマの生成装置
- **09_1 Bruch & Atwell**: ABM での mechanism 分析
- **16_1 Mobility networks**: COVID モデルで均衡的拡散

---

## 13. 主要引用

### Nash 原典
- **Nash, J. F. (1950)** "Equilibrium points in n-person games" *Proc. Natl. Acad. Sci. USA* 36, 48–49
- Nash, J. F. (1951) "Non-cooperative games" *Annals of Mathematics* 54, 286–295
- Nash, J. F. (1950) "The bargaining problem" *Econometrica* 18, 155–162
- Nash, J. F. (1953) "Two-person cooperative games" *Econometrica* 21, 128–140

### 基盤
- **von Neumann, J. & Morgenstern, O. (1944)** *Theory of Games and Economic Behavior*
- **Kakutani, S. (1941)** Duke Math. J. 8, 457–459
- **von Neumann, J. (1928)** Math. Annal. 100, 295–320

### Nobel 共同受賞
- **Harsanyi, J. (1967–68)** "Games with Incomplete Information" *Management Science* 14
- **Selten, R. (1965, 1975)** — equilibrium refinements

### Social Dilemmas
- **Flood, M. M. (1958)** — original PD experiment
- **Axelrod, R. (1984)** *The Evolution of Cooperation*
- **Hardin, G. (1968)** *Science* 162 "Tragedy of the Commons"
- **Ostrom, E. (1998)** APSA presidential address

### Market Design
- **Roth, A. E. & Peranson, E. (1999)** *Amer. Econ. Rev.* 89 — NRMP
- **Vickrey, W. (1961)** *J. Finance* 16
- **Milgrom, P. R. (2004)** *Putting Auction Theory to Work*

### Experimental / Learning
- **Smith, V. L. (1962)** *J. Polit. Econ.* 70
- **McKelvey, R. M. & Palfrey, T. R. (1995)** — QRE
- **Erev, I. & Roth, A. E. (1998)** *Amer. Econ. Rev.* 88
- **Fehr, E. & Gächter, S. (2002)** *Nature* 415

### Evolution
- **Maynard Smith, J. (1974)** *J. Theor. Biol.* 47
- **Hofbauer, J. & Sigmund, K. (1988)** *The Theory of Evolution and Dynamical Systems*
- **Nowak, M. A. & Sigmund, K. (1998)** *Nature* 393

---

## 14. 著者について

- **Charles A. Holt**: UVa 経済学教授、experimental economics の大家。McKelvey-Palfrey QRE の発展に貢献、Goeree との共同研究で noisy introspection model を開発
- **Alvin E. Roth**: Harvard 教授、後に 2012 年 Nobel 経済学賞（Shapley と共同受賞）を market design・matching theory で受賞。NRMP、kidney exchange などの応用

両者とも Nash 均衡を実験・応用面で継承発展させた当事者であり、この Perspective は単なる歴史記述ではなく**生きた証言**でもある。

---

## 要点

1. Nash の 1950 年 PNAS 一頁論文が、n 人ゲームの**自己対応 n-tuple = 均衡点**を定義。社会科学の統一理論の中心概念に
2. **Announcement test**: 全員が戦略を同時公表しても誰も変えたくない点 = Nash 均衡
3. **3つの解釈**: 助言 (prescription)、予測 (dynamic stable point)、自己強制合意 (self-enforcing)
4. **生物学への拡張**: 混合戦略=集団分布、payoff=fitness、動力学=population dynamics → evolutionary game theory
5. **Prisoner's Dilemma** は Dresher & Flood (1950) が Nash 均衡と実行為の乖離を示すために設計した実験。Nash 均衡 ≠ 望ましい
6. **Market design への応用**: NRMP、auction theory、labor market unraveling の解決 — Nash 均衡を制度設計の道具に
7. **Learning models, QRE** で非 Nash 行動を説明。one-shot ゲームでは noisy introspection
8. CS 222 Lecture 11 での位置: Schelling と並び**均衡概念の認識論**の原典。生成エージェント社会における「均衡は何か」「どう選ばれるか」「望ましさと均衡の乖離」を考える基盤
