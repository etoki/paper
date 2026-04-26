# 相関 / 予測精度分析 結果（N = 103）

Outcome: AcceptedUniversityHensachi（合格大学の偏差値）

## 1. 単変量相関（Pearson r / Spearman ρ, vs Hensachi）

| 変数 | Pearson r | p | Spearman ρ | p |
|---|---:|---:|---:|---:|
| Openness | -0.015 | 0.883 | -0.046 | 0.644 |
| Conscientiousness | +0.148 | 0.136 | +0.128 | 0.196 |
| Extraversion | +0.015 | 0.882 | +0.003 | 0.973 |
| Agreeableness | +0.022 | 0.826 | +0.021 | 0.836 |
| Neuroticism | +0.049 | 0.622 | +0.060 | 0.548 |
| NumberOfLecturesWatched | +0.204 | 0.038 | +0.103 | 0.300 |
| ViewingTime | +0.096 | 0.336 | +0.052 | 0.602 |
| NumberOfConfirmationTestsCompleted | +0.479 | <.001 | +0.423 | <.001 |
| NumberOfConfirmationTestsMastered | +0.478 | <.001 | +0.443 | <.001 |
| AverageFirstAttemptCorrectAnswerRate | +0.120 | 0.227 | +0.096 | 0.335 |

## 2. ブロック予測精度比較（10-fold CV と LOO-CV）

| モデル | In-sample R² | Adj R² | CV10 r | CV10 RMSE | LOO r | LOO RMSE |
|---|---:|---:|---:|---:|---:|---:|
| BigFive_full | 0.029 | -0.021 | -0.110 | 8.794 | -0.150 | 8.756 |
| BigFive_C_only | 0.022 | 0.012 | +0.059 | 8.422 | +0.020 | 8.457 |
| Behavior_full | 0.247 | 0.209 | +0.419 | 7.670 | +0.418 | 7.650 |
| Behavior_TestsMastered_only | 0.228 | 0.221 | +0.440 | 7.569 | +0.447 | 7.519 |
| Combined_BigFive_plus_Behavior | 0.257 | 0.176 | +0.319 | 8.204 | +0.336 | 8.037 |

## 3. Conscientiousness × オンライン学習行動 相関

| 行動指標 | Pearson r vs C | p |
|---|---:|---:|
| NumberOfLecturesWatched | +0.076 | 0.446 |
| ViewingTime | +0.065 | 0.516 |
| NumberOfConfirmationTestsCompleted | +0.196 | 0.047 |
| NumberOfConfirmationTestsMastered | +0.205 | 0.038 |
| AverageFirstAttemptCorrectAnswerRate | -0.007 | 0.948 |

## 4. 各モデル係数（in-sample OLS）

### BigFive_full
- intercept: +46.055
- Openness: -0.552
- Conscientiousness: +2.047
- Extraversion: +0.078
- Agreeableness: -0.067
- Neuroticism: +0.764

### BigFive_C_only
- intercept: +47.200
- Conscientiousness: +1.797

### Behavior_full
- intercept: +44.459
- NumberOfLecturesWatched: +0.056
- ViewingTime: -0.000
- NumberOfConfirmationTestsCompleted: +0.038
- NumberOfConfirmationTestsMastered: +0.003
- AverageFirstAttemptCorrectAnswerRate: +6.130

### Behavior_TestsMastered_only
- intercept: +48.636
- NumberOfConfirmationTestsMastered: +0.044

### Combined_BigFive_plus_Behavior
- intercept: +43.307
- Openness: -0.730
- Conscientiousness: +0.947
- Extraversion: +0.281
- Agreeableness: -0.428
- Neuroticism: +0.527
- NumberOfLecturesWatched: +0.065
- ViewingTime: -0.000
- NumberOfConfirmationTestsCompleted: +0.022
- NumberOfConfirmationTestsMastered: +0.017
- AverageFirstAttemptCorrectAnswerRate: +6.430
