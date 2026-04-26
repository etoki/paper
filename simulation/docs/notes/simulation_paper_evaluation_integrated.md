# シミュレーション論文の評価：HEXACO / Dark Triad simulation の設計と妥当性

作成日：2026-04-26
ブランチ：`main`
関連ドキュメント：`research_vision_integrated.md`（後続）

---

## Part 0：本ドキュメントの位置づけ

### 0.1 目的

既存 Simulation 論文（Big Five + generative agent + 大学入試結果予測、Park et al. 2024 系のアーキテクチャを日本教育ドメインに外挿）に続く、**Harassment 論文・Clustering 論文を起点とした新たな simulation 論文**の可能性を、保守的に評価する。

本ドキュメントは「**何が書けるか / 何が書けないか / どう書くべきか**」の方法論的評価に閉じる。

> **研究目的そのもの（なぜ simulation したいのか、究極的に何を達成したいのか）は別ドキュメント** `research_vision_integrated.md` **に分離して記述する**。

### 0.2 検討した3候補（概略）

| 候補 | 概要 | 一言 |
|---|---|---|
| **A** | Harassment 加害傾向の agent simulation | ハイリスク・ハイリターン |
| **B** | Clustering 7類型の type-conditional simulation | ローリスク・ミドルリターン |
| **C** | A + B の統合版 | 中間 |

詳細は Part 2 で展開。

### 0.3 評価フレームワーク

LLM-based simulation 論文の質を判定するため、以下の3層フレームを使用する：

1. **5レベル成果**（Calibration / Prediction / Counterfactual / Mechanism / Hypothesis Generation）— 何を達成すれば論文として成立するか
2. **良い論文の6戦略**（集団予測、失敗の主題化、triangulation、階層 baseline、pre-registration、ツール positioning）— 根本問題（construct validity gap）への対処法
3. **12 の追加評価ポイント**（A–L）— 実装・記述レベルの質

詳細は Part 3 で展開。

### 0.4 結論の早出し

- **候補 A は推奨しない**：safety alignment による variance flattening、construct validity 疑問、ethical/flattening 批判リスクが構造的
- **候補 B が最有力**：方法論的に堅実、failure mode も論文化可能、コスト低
- **候補 C（B の Stage 2 として A を内包する設計）が impact 最大**：ただし B の Stage 1 が成功した場合に限る

最も保守的な推奨設計：

> **「Partial-information type inference + multi-LLM triangulation + functional validation via outcome scales」**

---

（Part 1 以降、続く）
