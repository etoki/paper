# 12_1 — D³: Data-Driven Documents

## 書誌情報

- 著者: **Michael Bostock**, **Vadim Ogievetsky**, **Jeffrey Heer**（Stanford University Department of Computer Science）
- 掲載誌: *IEEE Transactions on Visualization and Computer Graphics* (TVCG), Vol. 17, No. 12, pp. 2301-2309, December 2011
- InfoVis 2011 Best Paper Award 受賞
- CS 222 Lecture 12（Data and Representation）の補足論文

---

## 1. 研究問題

インタラクティブなデータビジュアライゼーションを Web ブラウザ上で構築する際、既存のツールキット（Flash、Processing、Protovis 等）は**独自の中間表現**（scene graph、カスタムオブジェクトモデル）を採用していた。そのため:

- ブラウザネイティブな CSS/SVG/HTML のデバッガが使えない
- DOM インスペクタで要素の状態を見られない
- アクセシビリティツールが機能しない
- カスタムレンダラとブラウザ標準の二重維持コスト

研究問題:

> How can we design a visualization toolkit that embraces the browser's native representations (HTML, SVG, CSS) as **first-class citizens**, rather than abstracting them away?

---

## 2. 中心主張

**D³ (Data-Driven Documents)** は、データに基づいて Document Object Model (DOM) を**直接操作・変換**する JavaScript ライブラリである。

主要アプローチ:

1. **Representational transparency**: 独自のレイヤーを挟まず、SVG/HTML/CSS の DOM ノードを直接操作する
2. **Data binding**: データ配列と DOM 要素を join し、enter/update/exit の3パターンに分解
3. **Declarative transformations**: method chaining と selection を通じて宣言的に視覚属性を指定
4. **Transitions**: CSS/SVG 属性の値を時間関数として補間

### Bostock の前身 Protovis との比較

- Protovis (2009): 専用 scene graph を持つ宣言型ツールキット
- D³ (2011): scene graph を捨て、**DOM 自体**を scene graph として使う

> With direct access to the document, designers can leverage the full capabilities of CSS3, HTML5, and SVG, such as external stylesheets, complex selectors, media queries, animations, and hardware acceleration.

---

## 3. 主要な設計要素

### 3.1 Selection

jQuery ライクな selector:

```javascript
d3.selectAll("p").style("color", "white");
```

しかし jQuery と違い、selection は**データを伴う**。

### 3.2 Data join（enter / update / exit パターン）

```javascript
var circle = svg.selectAll("circle").data(numbers);
circle.enter().append("circle");       // データ > 既存要素（追加）
circle.attr("r", d => Math.sqrt(d));   // データ = 既存要素（更新）
circle.exit().remove();                // データ < 既存要素（削除）
```

この**enter/update/exit 三分割**が D³ の核心的イノベーションであり、Lecture 12 で引用される「データと表象の bind」。

### 3.3 Transitions

```javascript
circle.transition().duration(750)
      .attr("r", d => Math.sqrt(d * 10));
```

- `tween()` で補間関数をカスタマイズ可能
- CSS transitions より柔軟（path 属性、polygon 頂点なども補間可）

### 3.4 Scales と Layouts

- `d3.scale.linear()`, `d3.scale.log()`, `d3.scale.ordinal()` — 定義域から値域へのマッピング
- レイアウト: treemap, force-directed graph, chord diagram, bundle, stack, pie, histogram 等
- layout は**データ構造を変換**するだけで、描画自体は設計者に委ねる → 分離の原則

---

## 4. 方法と実装

### 4.1 Document Transformation

D³ は "data → DOM" の変換を**ネイティブブラウザ機構**で実現:

- 要素の作成: `append()`, `insert()`
- 属性の設定: `attr()`, `style()`
- 値の動的計算: `attr("cx", function(d, i) { return x(d); })`
  - `d`: データ要素、`i`: インデックス

この関数ベースの属性設定は、**データへの依存を明示的に**し、dependency tracking なしに再計算を可能にする。

### 4.2 Immediate Evaluation

Protovis は遅延評価（lazy）だった。D³ は**即時評価**:

| 利点 | 説明 |
|------|------|
| デバッグ容易 | 各ステップで DOM を調べられる |
| 予測可能性 | side effect の順序が明示的 |
| 性能 | アニメーションのフレーム単位更新がスムーズ |

トレードオフ: オペレータの宣言的・モジュール的組み合わせは難しくなる。

### 4.3 Performance Benchmarks

論文 Section 5 で Protovis、Flare、Processing.js、Raphaël との比較:

- **D3 の DOM 更新は既存の JS-SVG ツールキットと同等以上**
- 1万要素規模まで滑らかに動作
- Canvas ベース（Processing.js 等）より遅いケースもあるが、SVG 構造（CSS 適用、イベント処理）の便益で補える

---

## 5. 主要な結果・実例

### D³ で作られた著名なビジュアライゼーション

- **NYT "512 Paths to the White House"**: 2012年米大統領選の経路木
- **NYT "How Different Groups Spend Their Day"**: 時系列面積図
- **OECD Better Life Index**: 多変量円グラフ
- **Mike Bostock's bl.ocks.org ギャラリー**: 数千の再現可能な例

### インパクト

- npm ダウンロード: 数億規模
- GitHub スター: 100k+（2024時点で D3 と後継の Observable）
- ジャーナリズム（FT、NYT、Guardian）とアカデミアの事実上の標準

---

## 6. Discussion / 限界

### 6.1 学習曲線

> D³ is not a charting library; it's a framework for constructing your own visualizations from primitives.

初学者には難しい。Chart.js、Highcharts と違い「棒グラフ一行」はできない。

### 6.2 SVG 依存の限界

- 数万〜数十万要素で DOM が重くなる → Canvas/WebGL へのフォールバック（D³ v4 以降で強化）
- モバイル: 古い iOS/Android での SVG パフォーマンス

### 6.3 アクセシビリティ

SVG ベースなので ARIA 属性を後付けできるが、**D³ 自体は a11y を強制しない** → 多くの D³ ビジュアライゼーションはスクリーンリーダー非対応。

### 6.4 Reactive Paradigm の欠如

D³ は命令型（imperative）。後続の Vega / Vega-Lite（Heer 研究室）、Observable Plot は**宣言型（grammar of graphics）** へシフト。

---

## 7. CS 222 での位置づけ

### Lecture 12: "Data and Representation" のアンカー例

Park氏は Lecture 12 で **「シミュレーションから得たデータを、人間が理解できる表象にどう bind するか」** を問う。D³ は:

- データ配列と DOM 要素の **1対1 対応を明示的に管理**
- **enter/update/exit** は「データ分布の変化に応じて表象を動的に再配置」する正準的パターン
- 生成エージェントシミュレーションの出力を Web で可視化する際の事実上のインフラ

### Generative Agents との関係

Park et al. (2023) *Generative Agents* の Smallville 可視化（オープンタウンマップ上にエージェントを配置）は、D³ 類似の DOM 操作パターンで実装されている。CS 222 のプロジェクト演習で学生が D³ を使う可能性が高い。

### Rittel の wicked problem との関係

D³ の「primitive を提供し、設計者が組み立てる」哲学は、**決まった解のない可視化問題**に対する Rittel 的アプローチ:

- 「正しいグラフ」は存在しない → 組み合わせの自由度で応じる
- Stopping rule なし → 反復的リファインメント

---

## 8. 主要引用

### D³ が引用する関連研究

- **Protovis** (Bostock & Heer 2009, InfoVis): D³ の直接の前身
- **Polaris/Tableau** (Stolte, Tang, Hanrahan 2002): grammar of graphics の実装
- **Prefuse** (Heer, Card, Landay 2005): scene-graph ベースの Java ツールキット
- **Flare** (ActionScript 版 Prefuse): Flash 時代のビジュアライゼーション
- **Processing.js** (John Resig): Canvas ベース、Ben Fry & Casey Reas
- **Wilkinson (2005)** *The Grammar of Graphics*: 理論的基盤

### D³ を引用する後続研究

- **Vega** (Satyanarayan et al. 2014): D³ を内部で使う宣言型ビジュアライゼーション文法
- **Vega-Lite** (Satyanarayan et al. 2017): Vega の簡易版、altair の基盤
- **Observable** (Bostock 2018-): ノートブック形式の D³ 後継
- 数千の HCI / VIS 論文

---

## 要点

1. **D³ = Data-Driven Documents**: データ配列を Web ネイティブな DOM（SVG/HTML/CSS）に直接 bind する JS ライブラリ
2. **enter / update / exit** パターンが中心概念。データと DOM 要素の対応を明示的に管理
3. **Representational transparency**: 独自 scene graph を捨て、ブラウザのネイティブツール（デバッガ、インスペクタ、CSS）をフル活用
4. **Selection + method chaining** で宣言的に視覚属性を指定。`attr()`, `style()`, `transition()`
5. パフォーマンス: Protovis/Flare/Raphaël 同等以上、SVG 規模で1万要素オーダー
6. **Stanford Heer 研**の系譜（Protovis → D³ → Vega → Vega-Lite → Observable）を代表する論文
7. CS 222 では「シミュレーションから得たデータを人間の理解可能な表象へ bind する」ための正準ツールとして引用
8. 学習曲線・a11y・Canvas との比較などの限界あり、D³ v4 以降で一部解消
