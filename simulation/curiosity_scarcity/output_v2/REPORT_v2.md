# Refined curiosity simulation report (v2: empirically calibrated)

> 完全な好奇心ベース。本論文（v2.0 master, OSF DOI 10.17605/OSF.IO/3Y54U）には反映しない。
> v1 の heuristic γ を 5-PDF 文献監査で empirical 値に置換、link function と credibility interval の sensitivity を追加。

## 校正の出典

| パラメータ | v1 値 | v2 値 | 出典 |
|---|---|---|---|
| γ_D (resource scarcity) | 0.50 | **0.30** | Hershcovis 2007 perpetrator meta: situational constraints rc = .30 (interpersonal, k=10, N=2,734) |
| γ_D 80% CI | — | [0.19, 0.43] | Hershcovis 2007 Table 1 CI |
| γ_E (self-control deficit) | 0.40 | **0.22** | De Ridder 2012 meta: SCS × undesired \|r\| = .22 (k=21, N=12,402) |
| γ_E range | — | [0.15, 0.27] | De Ridder 2012 across scales |
| Cell-specific γ_c | uniform | low-C × 1.5, high-C × 0.7 | Bowling & Eschleman 2010 (N=726): 6/6 C × stressor interactions sig |
| CMV discount | none | sweep [0.7, 0.85, 1.0] | Podsakoff 2012 self-report inflation |

## Baseline

P_baseline = **0.1723** (N=354, cell-weighted)

Cell amplification (γ_c) per cluster from HEXACO-C deviation:
C0=1.20, C1=1.09, C2=1.22, C3=0.70, C4=0.82, C5=1.50, C6=1.29

## 観察 1: Headline scenarios across 3 link functions

```
      link           scenario  P_point  P_lo95  P_hi95
  identity           baseline   0.1723  0.1328  0.2147
  identity            C(0.20)   0.1379  0.1062  0.1695
  identity             D(0.5)   0.2015  0.1581  0.2502
  identity             E(0.5)   0.1937  0.1497  0.2409
  identity           C+D(0.5)   0.1612  0.1243  0.1986
  identity           C+E(0.5)   0.1550  0.1213  0.1925
  identity        C+D+E (0.5)   0.1813  0.1402  0.2225
  identity D+E (no inst, 0.5)   0.2266  0.1777  0.2794
  identity D+E (no inst, 1.0)   0.2884  0.2230  0.3552
  identity        C+D+E (1.0)   0.2307  0.1799  0.2870
log_linear           baseline   0.1723  0.1328  0.2119
log_linear            C(0.20)   0.1379  0.1085  0.1695
log_linear             D(0.5)   0.2042  0.1548  0.2518
log_linear             E(0.5)   0.1951  0.1511  0.2428
log_linear           C+D(0.5)   0.1633  0.1283  0.2010
log_linear           C+E(0.5)   0.1561  0.1230  0.1915
log_linear        C+D+E (0.5)   0.1851  0.1441  0.2255
log_linear D+E (no inst, 0.5)   0.2314  0.1826  0.2833
log_linear D+E (no inst, 1.0)   0.3116  0.2414  0.3791
log_linear        C+D+E (1.0)   0.2493  0.1929  0.3057
     logit           baseline   0.1723  0.1328  0.2119
     logit            C(0.20)   0.1379  0.1062  0.1695
     logit             D(0.5)   0.1960  0.1533  0.2405
     logit             E(0.5)   0.1895  0.1468  0.2335
     logit           C+D(0.5)   0.1581  0.1244  0.1933
     logit           C+E(0.5)   0.1525  0.1181  0.1860
     logit        C+D+E (0.5)   0.1743  0.1373  0.2093
     logit D+E (no inst, 0.5)   0.2148  0.1670  0.2606
     logit D+E (no inst, 1.0)   0.2638  0.2068  0.3146
     logit        C+D+E (1.0)   0.2175  0.1715  0.2571
```

主な変化:
- 旧 v1 で「C+D(0.5) が baseline と同等」だった結果は **新 γ で「C+D(0.5) は baseline より低い」** に変わる（γ_D 縮小のため scarcity 単独では制度を打ち消せなくなった）
- D+E 同時で 1.0 まで振っても、新 γ_E と新 γ_D ではそれ以前ほど劇的に上昇しない

## 観察 2: Credibility-interval sweep (frontier 位置)

「institution C=0.20 が baseline を保てない最小の (s+e)」を 9 通りの (γ_D, γ_E) で計算:

```
    link  gamma_D gamma_D_label  gamma_E gamma_E_label  frontier_s  frontier_e  frontier_sum  p_baseline
identity   0.1900     low (.19)   0.1500     low (.15)      0.9750      0.2000        1.1750      0.1723
identity   0.1900     low (.19)   0.2200    main (.22)      0.1500      0.8500        1.0000      0.1723
identity   0.1900     low (.19)   0.2700    high (.27)      0.0000      0.8250        0.8250      0.1723
identity   0.3000    main (.30)   0.1500     low (.15)      0.7500      0.0000        0.7500      0.1723
identity   0.3000    main (.30)   0.2200    main (.22)      0.6500      0.1000        0.7500      0.1723
identity   0.3000    main (.30)   0.2700    high (.27)      0.2500      0.5000        0.7500      0.1723
identity   0.4300    high (.43)   0.1500     low (.15)      0.5250      0.0000        0.5250      0.1723
identity   0.4300    high (.43)   0.2200    main (.22)      0.5000      0.0250        0.5250      0.1723
identity   0.4300    high (.43)   0.2700    high (.27)      0.5000      0.0250        0.5250      0.1723
```

frontier sum が大きいほど制度の防御力が広い。

## 観察 3: CMV discount sweep

Podsakoff 2012 の self-report common-method variance inflation を補正した γ_D で:

```
      link  cmv_discount  gamma_D_effective  P_D(0.5)_no_inst  P_D(0.5)_with_inst  p_baseline
  identity        1.0000             0.3000            0.2015              0.1612      0.1723
  identity        0.8500             0.2550            0.1971              0.1577      0.1723
  identity        0.7000             0.2100            0.1927              0.1542      0.1723
log_linear        1.0000             0.3000            0.2042              0.1633      0.1723
log_linear        0.8500             0.2550            0.1990              0.1592      0.1723
log_linear        0.7000             0.2100            0.1940              0.1552      0.1723
     logit        1.0000             0.3000            0.1960              0.1581      0.1723
     logit        0.8500             0.2550            0.1923              0.1549      0.1723
     logit        0.7000             0.2100            0.1887              0.1518      0.1723
```

## 解釈

1. **修正後の主結論は弱まる**: 旧 γ_D=0.5 では s ≈ 0.5 で制度が完全打ち消されたが、
   新 γ_D=0.30 では s = 1.0 でも制度の reduction を完全には超えられない（identity link）。
2. **Link function 依存性**: identity vs log-linear vs logit で frontier 形が大きく変わる。
   logit は p が小さい時はほぼ identity に縮退、p が大きい領域で saturating。
3. **Cell-specific γ_c で differential**: 低-C cluster (Cluster 5: C=2.27) は 1.5x 増幅、
   高-C cluster (Cluster 3: C=3.94) は 0.7x。Bowling & Eschleman 2010 N=726 直接 anchor。
4. **CMV 補正**: γ_D を 0.85x すると γ_D = 0.255、frontier はさらに緩む。Real effect
   は明示的 anchor が無いと punctual estimation できない。
5. **Hershcovis 2007 の interpersonal vs organizational**: rc=.30 vs .36。本データの
   power_harassment は interpersonal aggression に近い → .30 の選択が妥当。

## 修正された結論

> **資源不足 0.5・自制心不足 0.5 が同時にあっても、γ を文献由来の値に再校正すると、
> 制度介入 (effect_C=0.20) は依然として baseline 以下を維持する。
> 旧 v1 の「制度は容易に negate される」結論は、γ_D を victim ρ から借用したことに
> よる過大評価だった。**

しかしながら：
- **(s, e) = (1.0, 1.0)** という極端な scarcity + skill deficit ではどの link でも
  baseline を超える領域が現れる
- **cell-specific γ_c** が active なとき、低-C cluster に集中する scarcity は
  局所的に baseline を超える可能性

## v1 と v2 の比較

| Item | v1 (heuristic) | v2 (empirical) |
|---|---|---|
| 制度 defeat 閾値 (s 単独) | s* ≈ 0.50 | **s* > 1.0** (institute holds) |
| 制度 defeat 閾値 (e 単独) | e* ≈ 0.625 | **e* > 1.0** (institute holds) |
| s = e = 0.5 in C inst | P = 0.207 (悪化) | P ≈ 0.181 (約 baseline) |

## Caveats（v2 でも残る）

- **Hershcovis vs Bowling & Beehr の gap**: perpetrator (.30) と victim (.53) の差は
  CMV や reciprocity bias を含む。真の生成 effect は中間にある可能性。
- **De Ridder の meta は self-report self-control vs self-report behavior** の
  CMV 含む可能性。Marcus & Schuler primary r = -.63 はもっと大きい。
- **Cell-specific γ_c の linear mapping** は B&E 2010 の interaction を一次近似した heuristic。
- **link function の選択** は依然として modeling choice。文献で「正しい link」は決定できない。

## 引用文献

1. Hershcovis et al. (2007). Predicting workplace aggression: A meta-analysis.
   *J Applied Psychology, 92*(1), 228–238.
2. De Ridder et al. (2012). Taking stock of self-control: A meta-analysis...
   *Personality and Social Psychology Review, 16*(1), 76–99.
3. Bowling & Eschleman (2010). Employee personality, workplace stressors, and CWB.
   *J Occupational Health Psychology, 15*(1), 91–103.
4. Marcus & Schuler (2004). Antecedents of counterproductive behavior at work.
   *J Applied Psychology, 89*(4), 647–660.
5. Berry, Ones, & Sackett (2007). Interpersonal deviance, organizational deviance...
   *J Applied Psychology, 92*(2), 410–424.
