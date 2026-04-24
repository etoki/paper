# 11_1 — The Essence of Chaos (Lorenz 1993)

## 書誌情報

- 著者: **Edward N. Lorenz**（MIT, Department of Meteorology, 気象学者）
- 書籍: *The Essence of Chaos*, University of Washington Press, 1993
- シリーズ: Jessie and John Danz Lecture Series（1990年春の講義3本を基に増補）
- ISBN: 1-85728-454-2 (PB); Taylor & Francis eBook 2005
- 配布範囲: Preface, Chapter 1 "Glimpses of Chaos", Appendix 1 "The Butterfly Effect"
- Lecture 11 の補足論文。**バタフライ効果・初期値敏感性の原典**

---

## 1. Preface の位置づけ

30 年前（1960年代）、気象予測の大規模実験中に Lorenz が発見したのが「のちに chaos と呼ばれる現象」:

> Seemingly random and unpredictable behavior that nevertheless proceeds according to precise and often easily expressed rules.

他の研究者も散発的に遭遇していたが、方程式が解けないとか計算が進まないという「障害」としてしか認識していなかった。Lorenz 自身は、自分の実験が **chaotic な解を持つ方程式系を構築しないと成功しない**という逆説的な必要性に気づき、chaos を独立した研究対象として受容した。

> Chaos suddenly became something to be **welcomed**, at least under some conditions.

本書は Danz Lectures (1990) の再構成:
- Chapter 1, 2, 5: 第1講義（chaos の定義と特性、nonlinearity, complexity, fractality）
- Chapter 3: 第2講義（気象という chaotic system）
- Chapter 4: 第3講義（chaos の歴史、strange attractor、Lorenz 自身の関与）
- Appendix 1: 1972 年 AAAS 発表論文 "Does the Flap of a Butterfly's Wings in Brazil Set Off a Tornado in Texas?"
- Appendix 2: 数学的 excursion

---

## 2. Chapter 1 "Glimpses of Chaos" の核心

### 2.1 "It Only Looks Random"

**Chaos の意味変化**:
- 古典ギリシャ: 形と秩序の完全欠如
- 現代日常: "あるべき秩序の欠如"
- 科学的用法の多様化: Prigogine & Stengers (*Order Out of Chaos*), Norbert Wiener (複数の chaoses: ガス分子、水滴)

**Lorenz の選ぶ定義**:
> Processes that appear to proceed according to chance even though their behavior is in fact determined by precise laws.

すなわち **deterministic だが random に見える**。

### 2.2 Dynamical Systems の2類型

- **Flow**（連続的変化）: 振り子、転がる岩、砕ける波 — 微分方程式
- **Mapping**（離散ステップ変化）: ピンボールの連続衝突 — 差分方程式

いずれも状態 (state) は少数の数値変数で指定される。

### 2.3 Sensitive Dependence on Initial Conditions の定義

**Chaos の作業定義**:
> Systems in which [two almost identical states are followed by two states bearing no more resemblance than two states chosen at random from a long sequence] are said to be **sensitively dependent on initial conditions**.

重要な区別:
- 単なる差の時間増大 ≠ chaos
- **初期差が任意に小さくても、有限時間後に同じ有限サイズの差に膨れる** = chaos
- 例: 差 1 → 100 は chaos かもしれない; 差 0.01 → 100 も chaos（100倍のゲインがどの差でも起こる）
- 一方、差 1 → 100 だが 差 0.01 → 1 のシステムは chaos ではない（ゲインは差に比例）

### 2.4 "Initial conditions" の柔軟性

> "Initial conditions" need not be the ones that existed when a system was created. Often they are the conditions at the beginning of an experiment or a computation, but they may also be the ones at the beginning of any stretch of time that interests an investigator, so that **one person's initial conditions may be another's midstream or final conditions**.

研究者の関心時点が initial でよい。

---

## 3. Pinballs and Butterflies（ピンボールとバタフライ）

### Pinball Machine: Chaos の Clean Example

- コインの投げ・カードの切りは人間の介入で randomness を注入
- ピンボールは打ち出し後は純粋に物理法則 → chaos の clean な例

**Lorenz's Dartmouth experience (1930年代)**: 学生時代、ドラッグストアのピンボール機。町の当局は gambling law 違反と主張したが、"skill of contests" として合法化された。

**だが学生は上達しない** — なぜか? **Chaos のため**。

ピンから離れた2球の経路:
- 次のピンに到達時、同一球半径内に入るかどうかすら不確実
- 一方が直撃、他方が斜め衝突 → 進行方向が 90 度違う
- **1ピンで角度差が約10倍に拡大**
- プレイヤーが次のピンまでのコースを1つ延ばすには、**制御精度を10倍**上げる必要

**pinball の陥穽**: 有限のピン数で chaos が停止。真の chaos には「無限に長い」pinball 機が必要（実際には compactness の議論で代替可能）。

### 状態空間マッピングとしての pinball

300 球の初期位置を座標平面に plot → 次ピン衝突時の位置を plot → 連続的な "mapping"。これは flow から時間サンプリングで導ける。

### バタフライのシンボル

- James Gleick の書籍 *Chaos: Making a New Science*（1987）の最初の章が "The Butterfly Effect"
- 由来は Lorenz の 1972 年 AAAS 発表 "Does the Flap of a Butterfly's Wings in Brazil Set Off a Tornado in Texas?"
- それ以前は **sea gull** を symbol にしていた
- 発表時のプログラムタイトル変更をしたのは座長 Philip Merilees
- **Lorenz attractor** の形状が蝶に似ていたためとの説もあるが不確実
- **Ray Bradbury "A Sound of Thunder"**（prehistoric butterfly が未来の選挙結果を変える短編）も独立的に知られていた
- George R. Stewart *Storm* では "中国で sneeze する人が New York で雪かきを起こす" という気象学者の発言あり
- → **小さいものが大きな結果を生む** 象徴としての蝶は自然選択

---

## 4. 含意: 予測不可能性

**sensitive dependence の即時的帰結**:

> An immediate consequence of sensitive dependence in any system is the impossibility of making perfect predictions, or even mediocre predictions sufficiently far into the future.

- 測定に必ず不確実性がある（完全測定不能）
- ピンボールなら 1/10 度の測定では 3-4 ピン先が限界
- 1/1000 度でも 2-3 ピン延長のみ
- **weather forecast の根本的限界**の源泉

---

## 5. Compact Systems とリズム

pinball machine は非 compact（摩擦で状態空間を狭まる）。しかし「1ブロック長の傾斜歩道に配置された同一パターンのピン列」を想像すれば、近似反復が保証される compact system に。

Compact な dynamical system:
- 状態が有限変数・有界 → golf green の例
- 十分長い観察で「ほぼ同じ状態」が再帰的に現れる
- 周期的挙動 vs chaos の区別はここで議論可能

---

## 6. Appendix 1: "The Butterfly Effect" 1972 年原典

Lorenz が 1972年12月29日、American Association for the Advancement of Science (AAAS) 第139回大会の Global Atmospheric Research Program セッションで発表した press release 用原稿。

### タイトル
**"Predictability: Does the Flap of a Butterfly's Wings in Brazil Set Off a Tornado in Texas?"**

### 2つの命題（緩衝として）

1. 「単発の蝶の羽ばたきが tornado を起こせるなら、**その前後のすべての羽ばたき、無数の他の蝶の羽ばたき、さらには他種（人類含む）のあらゆる活動も同じく起こせる**」
2. 「単発の羽ばたきが tornado を起こすなら、同じ羽ばたきが **tornado を未然に防ぐこと**も同じくできる」

要するに、微小擾乱は **tornado 発生頻度を増減させない**。せいぜい「どの tornado がどの順序で起きるか」の sequence を modify するだけ。

### 本質的な問い
> Is the behavior of the atmosphere **unstable** with respect to perturbations of small amplitude?

答えは "一匹の蝶が違いを生むか" ではなく、**大気が小擾乱に対して不安定か**。

### 方法論
- 大気を実験できないので **コンピュータシミュレーション**で調べる
- 2つの数値解を比較: 「実際の天気」と「僅かに異なる初期条件の天気」 — 前者は perfect technique + perfect observations、後者は perfect technique + imperfect observations に相当
- 差が forecast error をシミュレートする

### 主要結果（1972年時点）

1. **粗構造の誤差は約 3 日で2倍**に成長。成長率は誤差が大きくなると低下。→ 観察誤差を半分にすれば予測可能範囲が 3 日延長。数週間先の予測も希望あり
2. **細構造の誤差（個々の雲の位置など）は数時間以内に2倍**に。そのままでは長期予測に響かない（通常細構造は forecast しない）
3. **細構造誤差は appreciable になると粗構造に波及**。1日程度で粗構造に誤差が現れ、そこから通常通り成長。細構造を半減しても粗構造予測は数時間延長のみ。**2週間以上の予測希望は大きく減退**
4. 週平均気温や週降水量などの特殊量は全パターンが predictable でない範囲でも predictable かも

### 蝶の影響に関する追加論点

- **微小・局所**: 蝶の影響は fine detail かつ localized。拡散過程のモデル化は難しい
- **Brazil vs Texas は別半球**: 熱帯と温帯の大気はほぼ別の流体。誤差が赤道を越えられない可能性
- → 原問いへの答えは **"unanswered for a few more years"** と保留しつつ、大気の不安定性への信頼は維持

### 実務的結論

> Today's errors in weather forecasting cannot be blamed entirely nor even primarily upon the finer structure of weather patterns. They arise mainly from our failure to observe even the coarser structure with near completeness, our somewhat incomplete knowledge of the governing physical principles, and the inevitable approximations which must be introduced in formulating these principles as procedures which the human brain or the computer can carry out.

GARP (Global Atmospheric Research Program) の目的は **"making not exact forecasts but the best forecasts which the atmosphere is willing to have us make"**.

---

## 7. Appendix 2 から: Lorenz の "Butterfly" 方程式

Lorenz 1963 "Deterministic Nonperiodic Flow" で導入された 3 変数連立微分方程式:

```
dx/dt = σ(y − x)
dy/dt = rx − y − xz
dz/dt = xy − bz
```

- σ = 10, r = 28, b = 8/3 で chaotic
- (x, z) 平面に投影すると **蝶の形の strange attractor**
- Lorenz の典型的エピソード: "t = 0.002 のつもりが小数点を間違えた" → 新種の butterfly を発見

Rössler 方程式（nonlinear 項 1 つ）もシンプルな chaotic 例。

---

## 8. CS 222 での位置づけ

### Lecture 11: 均衡、予測、不可知性

Park氏は Lecture 11 で「シミュレーションは未来予測の道具か、あるいは仮説生成の道具か」を論じる。Lorenz の本書は **"予測可能性の根本的限界"** の原点として引用される:

1. **"Single point prediction" の拒否**: Social Simulacra (Park 2022) の Multiverse は、ひとつの予測ではなく**複数宇宙**を示す — これは Lorenz の "small disturbances modify sequence, not statistics" 命題の派生形
2. **"Initial conditions matter"**: Generative Agent の prompt 条件付けの微小変化が行動を変える — 同じ命題の LLM 版
3. **"Sensitive dependence ≠ random"**: LLM の temperature は意図的に randomness を導入するが、low-temperature でも prompt 差で大きく分岐しうる → 構造的な chaos の現れ

### Park氏の論点との接続

- **"Simulation as predictor" への慎重**: 社会シミュレーションは Lorenz の weather forecast と同様、数日 / 数ステップまでは妥当だが、長期的 point prediction は不可能
- **"The atmosphere is willing to have us make"**: 社会シミュレーションも「社会が許す最良の予測」を目指すべき — 過度な主張は科学倫理に反する
- **Validation の困難性**: Lorenz は「大気は制御実験できない」と明言 — 生成エージェント社会も同じ
- **Chaos と emergence**: ABM (09_1) が扱う emergent patterns は、本質的に chaotic な過程から生まれる

### 引用される派生的議論

- **Butterfly effect の誤用**: 因果的な「1つの小原因が1つの大結果を招く」ではなく、統計的な「系が不安定」という意味
- **Epistemology of simulation**: 完全予測を諦めつつ、**構造的な洞察**を抽出する姿勢
- **"Welcome chaos"**: Lorenz は chaos を邪魔者ではなく研究対象として歓迎した — CS 222 の研究姿勢にも通じる

---

## 9. 主要引用

### 本書が参照する文献
- **Prigogine & Stengers** *Order Out of Chaos*
- **Norbert Wiener** — "chaoses" の複数化
- **James Gleick (1987)** *Chaos: Making a New Science* — "Butterfly Effect" 章
- **Ray Bradbury** "A Sound of Thunder" — 蝶が未来の選挙を変える短編
- **George R. Stewart** *Storm* — "中国で sneezing、New York で雪かき"
- **B. B. Mandelbrot** *The Fractal Geometry of Nature*
- **Colin Sparrow** — Lorenz 方程式についての書籍
- **Henri Poincaré** — chaos の歴史的起点

### 本書が生み出した後続
- Strogatz *Nonlinear Dynamics and Chaos* (1994)
- Peitgen, Jürgens, Saupe *Chaos and Fractals* (1992)
- Ott *Chaos in Dynamical Systems* (1993)

### Lorenz 自身の関連論文
- **Lorenz, E. N. (1963)** "Deterministic Nonperiodic Flow" *J. Atmos. Sci.* 20 — 原典論文
- **Lorenz, E. N. (1972)** "Predictability: Does the Flap of a Butterfly's Wings in Brazil Set Off a Tornado in Texas?" — AAAS 発表（本書 Appendix 1）

---

## 10. 書誌的補足

- 書籍タイトル（1993）= 1990年 Danz Lectures のタイトル "The Essence of Chaos"
- 書籍版は図の多くをコンピュータグラフィックスで再制作
- 1995年に UK (UCL Press) からペーパーバック、2005年に Taylor & Francis eBook
- 数学的正確性を保ちつつ一般読者向け

---

## 要点

1. **Chaos = deterministic だが random に見える挙動**。定義的核心は **sensitive dependence on initial conditions**
2. 単なる誤差成長 ≠ chaos。**初期差が任意に小さくても、時間経過で有限差に到達する** のが chaos
3. **Pinball と Butterfly** が chaos の 2 大 symbol。Lorenz は元々 sea gull を使っていたが、1972 年 AAAS 発表で butterfly に改称（座長 Merilees の判断）
4. **"Does the Flap of a Butterfly's Wings in Brazil Set Off a Tornado in Texas?"** の本質は **因果命題ではなく大気の不安定性（instability）** の問い
5. 微小擾乱は tornado の頻度を増減しない。**sequence を modify するだけ**。しかしそれゆえに **point prediction は不可能**
6. 1972 年時点の Lorenz の診断: 粗構造誤差は 3 日で倍増、細構造は数時間で倍増、2 週間以上の予測は希望薄
7. Lorenz 方程式 dx/dt=σ(y-x), dy/dt=rx-y-xz, dz/dt=xy-bz は (σ=10, r=28, b=8/3) で chaotic — 蝶形の strange attractor を生む
8. CS 222 の Lecture 11 では **予測可能性の根本的限界** を論じる原点として引用。Generative Agent の Multiverse 思想（Park 2022）の理論的基盤、"simulation as hypothesis generator" の認識論
