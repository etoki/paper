# v3: 3 follow-up analyses

> 完全な好奇心ベース。本論文には反映しない。v2 校正の上に 3 つの追加診断を載せた。

---

## Analysis 1: N=354 cluster × harassment cross-tab

Bowling & Eschleman 2010 (N=726) は「Conscientiousness が低い人ほど stressor →
CWB の relation が強い」を 6/6 interaction 全部で示した。本データの cluster
レベルでも同じ pattern が見えるかを直接検証。

```
 cluster  centroid_C   n  harassment_rate     se  ci_lo  ci_hi
       0      2.8900  23           0.3913 0.1018 0.1918 0.5908
       1      3.1300  42           0.0952 0.0453 0.0065 0.1840
       2      2.8500  49           0.1633 0.0528 0.0598 0.2668
       3      3.9400  49           0.0816 0.0391 0.0050 0.1583
       4      3.6800  51           0.2353 0.0594 0.1189 0.3517
       5      2.2700  26           0.0385 0.0377 0.0000 0.1124
       6      2.7100 114           0.2018 0.0376 0.1281 0.2754
```

**Pearson r (cluster centroid C, cluster harassment rate) = +0.034, p = 0.942**

解釈:
- B&E 2010 予測: r < 0（低-C クラスタほど高 harassment 率）
- 本データ: r = +0.034 → 予測不支持

注：本データの r は cluster-level (k=7) なので power が低い。CI は wide。

---

## Analysis 2: Tornado plot

Reference 点: s = e = 0.5、effect_C = 0.20、identity link、empirical γ。
P_base = 0.1813

各仮定を CI / range の両端に振ったときの prevalence 変化:

```
            variation      P   delta  abs_delta
      link=log_linear 0.1851  0.0038     0.0038
    CMV discount 0.85 0.1773 -0.0039     0.0039
       γ_E high (.27) 0.1858  0.0046     0.0046
 γ_c uniform (no amp) 0.1760 -0.0053     0.0053
        γ_E low (.15) 0.1749 -0.0064     0.0064
           link=logit 0.1743 -0.0070     0.0070
γ_c spread (0.5, 2.0) 0.1887  0.0075     0.0075
    CMV discount 0.70 0.1734 -0.0079     0.0079
        γ_D low (.19) 0.1716 -0.0096     0.0096
       γ_D high (.43) 0.1927  0.0114     0.0114
        effect_C 0.30 0.1586 -0.0227     0.0227
        effect_C 0.10 0.2039  0.0227     0.0227
```

**結論ドライバ ranking** (abs_delta 大きい順):
  effect_C 0.10: Δ = +0.0227
  effect_C 0.30: Δ = -0.0227
  γ_D high (.43): Δ = +0.0114
  γ_D low (.19): Δ = -0.0096
  CMV discount 0.70: Δ = -0.0079

解釈:
- **最も結論を動かす仮定**: 上位 3 つを見る → どれかが大きく effect 持っているなら
  そこに reference data を投資すべき
- **小さい仮定**: ほぼ無視できる、modeling choice として transparent に処理可

---

## Analysis 3: Time-axis sensitivity

γ_D を時間 regime で scaling した場合の institution-defeat threshold:

```
                      regime  scaling_factor  gamma_D_effective  frontier_s  frontier_e  frontier_sum  P_ref(s=e=0.5)  P_extreme(s=e=1.0)  p_baseline
         acute (1-day spike)          0.5000             0.1500      0.0250      1.0000        1.0250          0.1681              0.2014      0.1723
cross-sectional (Hershcovis)          1.0000             0.3000      0.6500      0.1000        0.7500          0.1813              0.2307      0.1723
     prospective (De Ridder)          0.6087             0.1826      0.0250      1.0000        1.0250          0.1710              0.2077      0.1723
          Hobfoll COR spiral          1.4000             0.4200      0.5000      0.0500        0.5500          0.1918              0.2541      0.1723
```

**4 regime の経験的 / 理論的根拠**:
1. **acute (× 0.5)**: 単発 stressor — heuristic（直接 anchor 無し）
2. **cross-sectional (× 1.0)**: Hershcovis 2007 / De Ridder 2012 等の主流 design
3. **prospective (× 0.61)**: De Ridder 2012 直接 anchor
   (cross-sect r=-.23 vs prospective r=-.14, 比率 0.61)
4. **Hobfoll COR spiral (× 1.4)**: 慢性 stressor の累積 loss spiral
   理論予測（empirical anchor 無し）

**重要 finding（contradiction）**:
- 経験的に時間が経つと effect は **減衰** する (De Ridder prospective)
- 理論的には Hobfoll COR は **増幅** を予測する
- これら 2 つは反対方向 → どっちが正しいか文献では決着ついていない

frontier_sum で見ると:
- prospective regime: institution が **広い範囲で hold** (frontier_sum =
  1.02)
- COR spiral regime: institution が **早く defeat** される (frontier_sum =
  0.55)

---

## まとめ

| Analysis | 主な finding |
|---|---|
| 1. Cluster cross-tab | r = +0.034 で B&E 予測 不支持。本データの cluster 構造は personality × stressor interaction の cell-level mapping を 正当化しない |
| 2. Tornado | 最も結論を動かす仮定: effect_C 0.10 (Δ = +0.0227) |
| 3. Time axis | 経験 (De Ridder) と理論 (Hobfoll) で逆方向の予測 → 時間軸の校正には longitudinal stressor-aggression データが必要 |

## 残る重要 caveats

- **本データの cluster 数 = 7** で cluster-level Pearson r の power が低い
- **時間軸の anchor は De Ridder の self-control prospective のみ** で、
  stressor の longitudinal anchor は本 5-PDF set に含まれない
- **COR spiral factor 1.4 は parametric assumption**、empirical anchor 無し

## さらなる diagnostic（PDF 不要）

- N=354 cluster × stressor (本データに workload measure が無いので不可)
- 7-cluster 内での Cluster 0 と Cluster 5 の direct comparison（Tier 1 finding：
  Cluster 0 が high-X × low-HH なら harassment 高位、Cluster 5 が低 C なのに
  低 harassment ならば B&E 予測 contradict）
