# Conscientiousness × Hensachi 診断（N=103）

## 1. 分布 / 範囲制限
- **Conscientiousness**: mean=2.77, sd=0.69, min=1.00, max=4.50, IQR=[2.33, 3.25]
- **Hensachi**: mean=52.18, sd=8.43, min=35.00, max=72.60, IQR=[47.50, 58.20]

母集団基準 SD=10 に対して観測 SD=8.43 → 範囲制限の兆候あり

## 2. 相関と 95% CI
- Pearson  r = +0.148 [95% CI -0.047, +0.332], p=0.136
- Spearman ρ = +0.128, p=0.196

- 5% Winsorized Pearson r = +0.157, p=0.113

Cook's D 上位 5 サンプル（影響の強い観測）
| ID | C | Hensachi | resid | leverage | Cook's D |
|---|---:|---:|---:|---:|---:|
| TKYHS072 | 1.00 | 37.20 | -11.80 | 0.074 | 0.085 |
| TKYHS037 | 1.83 | 70.50 | +20.01 | 0.028 | 0.084 |
| TKYHS062 | 4.50 | 45.00 | -10.28 | 0.070 | 0.061 |
| TKYHS061 | 2.17 | 70.30 | +19.20 | 0.017 | 0.046 |
| TKYHS006 | 3.42 | 35.00 | -18.34 | 0.018 | 0.045 |

上位 5 観測除外後: Pearson r = +0.227, p=0.024, n=98

## 3. Conscientiousness ファセット（BFI-2 3 ファセット）
| ファセット | r | p |
|---|---:|---:|
| Organization | +0.101 | 0.309 |
| Productiveness | +0.084 | 0.401 |
| Responsibility | +0.195 | 0.049 |

### 参考: 全 15 ファセット
| ファセット | r | p |
|---|---:|---:|
| IntellectualCuriosity | -0.101 | 0.308 |
| AestheticSensitivity | +0.077 | 0.442 |
| CreativeImagination | -0.033 | 0.738 |
| Organization | +0.101 | 0.309 |
| Productiveness | +0.084 | 0.401 |
| Responsibility | +0.195 * | 0.049 |
| Sociability | -0.067 | 0.501 |
| Assertiveness | +0.134 | 0.177 |
| EnergyLevel | -0.023 | 0.817 |
| Compassion | +0.078 | 0.436 |
| Respectfulness | -0.037 | 0.711 |
| Trust | +0.019 | 0.847 |
| Anxiety | +0.056 | 0.572 |
| Depression | +0.041 | 0.684 |
| EmotionalVolatility | +0.034 | 0.736 |

## 4. 検定力 / 必要 N
- 観測 r=0.148 を α=.05 両側で有意検出する必要 N: **357**（現 N=103）
- 先行メタ分析 r=0.167 (Tokiwa 2026) を検出する必要 N: **280**
- 古典 Poropat (2009) r=0.19 を検出する必要 N: **216**
- 現 N=103, r=0.148 の post-hoc power: **0.32**
- 現 N=103 で r=0.20 を検出する power: **0.53**

## 5. 非線形性チェック
- 線形 R² = 0.0219
- 二次項追加 R² = 0.0485（F=2.80, p=0.097）
