# Curiosity simulation report

> 完全な好奇心ベース。本論文には反映しない。pre-registration v2.0 の confirmatory 範囲外。

## Setup

- N = 354 (harassment + HEXACO domains)
- 7 clusters × 2 genders = 14 cells
- Binarization: power_harassment ≥ mean + 0.5·SD
- γ_D = 0.5 (Bowling & Beehr 2006 ρ メタ解析の保守換算)
- γ_E = 0.4 (Hudson 2023 d=0.71 の risk-multiplier 一次近似)
- effect_C (institution) = 0.2
- Bootstrap B = 2000, seed = 20260429

## Baseline prevalence

P_baseline = **0.1723** (binary harassment proportion in N=354, cell-weighted)

## 観察 1: Institution efficacy frontier

institution C=0.20 が baseline を保てない最小の (s, e) ≈ (0.50, 0.00)

→ 制度介入 C(20% 削減) は scarcity と self-control deficit の合成圧力に対して
**finite な防壁**であって、frontier を超える領域では「制度を維持しても baseline
を超えるハラスメントが起こる」ことが示される。

## 観察 2: Marginal effects (point estimates)

軸を 0 → 1 に動かしたときの prevalence (各 scenario は別軸):

```
         scenario      s      e  effect_C      P
           D_only 0.0000 0.0000    0.0000 0.1723
           D_only 0.1000 0.0000    0.0000 0.1809
           D_only 0.2000 0.0000    0.0000 0.1895
           D_only 0.3000 0.0000    0.0000 0.1982
           D_only 0.4000 0.0000    0.0000 0.2068
           D_only 0.5000 0.0000    0.0000 0.2154
           D_only 0.6000 0.0000    0.0000 0.2240
           D_only 0.7000 0.0000    0.0000 0.2326
           D_only 0.8000 0.0000    0.0000 0.2412
           D_only 0.9000 0.0000    0.0000 0.2499
           D_only 1.0000 0.0000    0.0000 0.2585
           E_only 0.0000 0.0000    0.0000 0.1723
           E_only 0.0000 0.1000    0.0000 0.1792
           E_only 0.0000 0.2000    0.0000 0.1861
           E_only 0.0000 0.3000    0.0000 0.1930
           E_only 0.0000 0.4000    0.0000 0.1999
           E_only 0.0000 0.5000    0.0000 0.2068
           E_only 0.0000 0.6000    0.0000 0.2137
           E_only 0.0000 0.7000    0.0000 0.2206
           E_only 0.0000 0.8000    0.0000 0.2275
           E_only 0.0000 0.9000    0.0000 0.2344
           E_only 0.0000 1.0000    0.0000 0.2412
           C_only 0.0000 0.0000    0.0000 0.1723
           C_only 0.0000 0.0000    0.0500 0.1637
           C_only 0.0000 0.0000    0.1000 0.1551
           C_only 0.0000 0.0000    0.1500 0.1465
           C_only 0.0000 0.0000    0.2000 0.1379
           C_only 0.0000 0.0000    0.2500 0.1292
           C_only 0.0000 0.0000    0.3000 0.1206
    D+E (no inst) 0.0000 0.0000    0.0000 0.1723
D+E + Institution 0.0000 0.0000    0.2000 0.1379
    D+E (no inst) 0.1000 0.1000    0.0000 0.1882
D+E + Institution 0.1000 0.1000    0.2000 0.1505
    D+E (no inst) 0.2000 0.2000    0.0000 0.2047
D+E + Institution 0.2000 0.2000    0.2000 0.1638
    D+E (no inst) 0.3000 0.3000    0.0000 0.2219
D+E + Institution 0.3000 0.3000    0.2000 0.1776
    D+E (no inst) 0.4000 0.4000    0.0000 0.2399
D+E + Institution 0.4000 0.4000    0.2000 0.1919
    D+E (no inst) 0.5000 0.5000    0.0000 0.2585
D+E + Institution 0.5000 0.5000    0.2000 0.2068
    D+E (no inst) 0.6000 0.6000    0.0000 0.2778
D+E + Institution 0.6000 0.6000    0.2000 0.2222
    D+E (no inst) 0.7000 0.7000    0.0000 0.2978
D+E + Institution 0.7000 0.7000    0.2000 0.2382
    D+E (no inst) 0.8000 0.8000    0.0000 0.3184
D+E + Institution 0.8000 0.8000    0.2000 0.2548
    D+E (no inst) 0.9000 0.9000    0.0000 0.3398
D+E + Institution 0.9000 0.9000    0.2000 0.2718
    D+E (no inst) 1.0000 1.0000    0.0000 0.3619
D+E + Institution 1.0000 1.0000    0.2000 0.2895
```

## 観察 3: Bootstrap CI on headline scenarios

| scenario               |   P_point |   P_lo95 |   P_hi95 |    B |
|:-----------------------|----------:|---------:|---------:|-----:|
| baseline               |    0.1723 |   0.1328 |   0.2147 | 2000 |
| C(0.20)                |    0.1379 |   0.1062 |   0.1695 | 2000 |
| D(0.5)                 |    0.2154 |   0.1695 |   0.2684 | 2000 |
| E(0.5)                 |    0.2068 |   0.1593 |   0.2576 | 2000 |
| C+D(0.5)               |    0.1723 |   0.1328 |   0.2119 | 2000 |
| C+E(0.5)               |    0.1654 |   0.1302 |   0.2061 | 2000 |
| C+D+E (all 0.5)        |    0.2068 |   0.1593 |   0.2542 | 2000 |
| D+E (no inst, all 0.5) |    0.2585 |   0.2034 |   0.3178 | 2000 |
| D+E (no inst, all 1.0) |    0.3619 |   0.2788 |   0.4390 | 2000 |
| C+D+E (all 1.0)        |    0.2895 |   0.2231 |   0.3585 | 2000 |

## 観察 4: Self-control deficit — trait-mediated vs direct

Mediated 経路（HEXACO-C を SD だけ下方シフト → 再クラスタリング）と
Direct 経路（cell propensity を multiplier で増幅）は **異なる感度** を示す:

```
 delta_E_sd  P_mediated  n_reassigned
     0.0000      0.1723             0
     0.1000      0.1742             9
     0.2000      0.1760            18
     0.3000      0.1761            25
     0.4000      0.1759            32
     0.5000      0.1756            43
     0.6000      0.1749            46
     0.7000      0.1711            59
     0.8000      0.1697            66
     0.9000      0.1687            76
     1.0000      0.1681            80
     1.1000      0.1686            83
     1.2000      0.1669            89
     1.3000      0.1665            94
     1.4000      0.1646           102
     1.5000      0.1635           111
     1.6000      0.1641           116
     1.7000      0.1625           124
     1.8000      0.1613           133
     1.9000      0.1606           136
     2.0000      0.1596           139
     2.1000      0.1568           147
     2.2000      0.1556           148
     2.3000      0.1548           151
     2.4000      0.1538           155
     2.5000      0.1510           160
     2.6000      0.1489           162
     2.7000      0.1465           167
     2.8000      0.1450           168
     2.9000      0.1423           173
     3.0000      0.1383           178
```

クラスタ membership は 6 次元の Euclidean 最近傍で決まるので、HEXACO-C を
1 軸だけ下方シフトしてもクラスタ越境は段階的にしか起こらない。`n_reassigned`
列は実際に new cluster に切り替わった個体数。Direct 経路は連続的に上昇する
のに対し、Mediated 経路は **離散的なジャンプ** を示すのが特徴。

## 解釈

1. **資源不足 (s)**: cell propensity を一様に拡大するため、制度の reduction を
   即座に飽和させる。代数的には s* ≈ effect_C / (γ_D · (1 − effect_C)) =
   0.20 / (0.5 · 0.80) = 0.50 で institution C を完全相殺。観察値 s ≈ 0.55
   と近い。
2. **自制心不足 (e)** の direct 経路: 同じ機序で e* ≈ effect_C / (γ_E · (1 − effect_C)) =
   0.20 / (0.4 · 0.80) = 0.625 で institution C を相殺。
3. **D × E 相互作用** は乗法的なので、s = e = 0.4 付近で既に baseline (C 適用後)
   を超える: (1.20)(1.16)/(1 − 0.20) = 1.74 > 1.0。これが「制度を強化しても
   消えない残差」を最も簡潔に示す代数。
4. **HEXACO-C 経由の mediated 経路は本データでは non-monotonic**: δ_E ∈ [0, 0.3]
   ではわずかに上昇するが、δ_E > 0.5 で **逆に減少** する。これは Cluster 5
   (HEXACO-C 最低 = 2.27) が偶然 baseline propensity の低いプロファイル
   (p ≈ 0.07/0) を持つため。本データでは "low C → high harassment" の単純な
   伝達は成立せず、harassment 高位プロファイルは Cluster 0 (low HH, high X)
   と Cluster 4 (low A, high O) に分布。**自制心 = HEXACO-C** という単純化
   は本データでは失敗するという empirical finding。
5. **生産性低下** を s × e として読むと、資源と自制心の両方を同時に欠く環境
   (低 TFP の代理) では、厳罰化が単独で機能する余地が急速に縮小する。
6. **政策含意としての示唆**: 制度介入 C は資源・技術の十分性を前提とした効果と
   して読むべきで、stressor 環境の改善（労働時間規制、人員配置、教育）が
   並走しないと law-on-the-books と law-in-action が乖離する。

## Caveats

- N=354 に workload も self-control も直接測定がない → γ_D, γ_E は外部 anchor
- `s × e` を生産性合成にしたのは粗い heuristic。realistic な TFP は別モデル
- Multiplier 形は cap(1.0) を入れているが、低 baseline cells で linear extrapolation
- v2.0 master の counterfactual A/B/C との直接比較は意図しない (異なる介入空間)
- Mediated path の非単調性は cluster identity の高次元構造に依存し、別データセット
  では逆向きの結果が出る可能性。本知見は datasets-dependent。

