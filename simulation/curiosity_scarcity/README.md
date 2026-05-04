# Curiosity: Scarcity × Self-Control × Institution

**Status**: 完全な好奇心ベースの探索。本論文（v2.0 master / OSF DOI 10.17605/OSF.IO/3Y54U）の confirmatory 部分には反映しない。

## 問い

> 性格が悪くても制度が良ければ悪い行為は起こらず、良い性格でも制度が悪ければ悪い行為が促進される — これが本シミュレーションの結論。
> ただし厳罰化（=制度介入 C）しても犯罪が消えないことがある。資源不足・技術不足・生産性低下が悪い行為を後押しする可能性。
> ハラスメント文脈では具体的にどう現れ、どこまでシミュレーションに落とし込めるか？

## マッピング

| 概念 | 操作可能な変数 | 文献 anchor |
|---|---|---|
| 資源不足 (人手不足→過労→ストレス) | `s ∈ [0, 1]` (組織 stressor 強度) | Bowling & Beehr 2006 ρ=.30–.53 |
| 技術不足 (自制心の低さ) | `e ∈ [0, 1]` (self-control deficit) | HEXACO C low / Hudson 2023 trait shift |
| 生産性低下 | `s × e` 相互作用 | Hobfoll COR / Spector & Jex 1998 |
| 制度 | `effect_C ∈ [0, 0.30]` (cell propensity reduction) | v2.0 master Counterfactual C |

## 介入式

baseline cell propensity を `p_c` (14 cells = 7 cluster × 2 gender) としたとき:

```
P_C(s, e, effect_C) = p_c × (1 − effect_C) × (1 + γ_D · s) × (1 + γ_E · e)
```

- γ_D = 0.5 (Bowling & Beehr ρ=.40 中央値の保守的多重化変換)
- γ_E = 0.4 (Hudson 2023 d=0.71 を logit→risk 変換した一次近似)
- 集計: cell 重み (N=354 の経験分布) で加重平均

## 結果の読み方

3 つの視点で報告する：

1. **境界面**: 制度介入 C=20% reduction が (s, e) 平面上のどこまで baseline を下回らせるか（"institution efficacy frontier"）
2. **寄与分解**: scarcity と self-control deficit の周辺寄与
3. **HEXACO-C shift 版 (E')**: 自制心を直接 cell propensity で modulate するのではなく、HEXACO Conscientiousness を `−δ_E SD` シフトして再クラスタリング → 性格通路と直接通路の比較

## 走らせ方

```bash
cd simulation/
uv run python -m curiosity_scarcity.scarcity_simulation
```

出力: `output/figures/*.png`, `output/tables/*.csv`, `output/REPORT.md`

## 制約 (honest)

- N=354 には workload / 自制心の直接測定なし → γ_D, γ_E は外部 meta calibration
- 個人レベル予測ではなく population-level の "what-if"
- v2.0 master pre-registration 範囲外（exploratory only）
- 生産性は `s × e` 合成代理。実際の TFP は別軸
