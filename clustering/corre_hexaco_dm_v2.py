import pandas as pd
import numpy as np
from scipy.stats import spearmanr
import numpy as np

# データ読み込み
df = pd.read_csv("csv/hexaco_domain.csv")

cols = df.columns
n = len(cols)

# 結果保存用
corr_matrix = pd.DataFrame(np.zeros((n, n)), columns=cols, index=cols)
p_matrix = pd.DataFrame(np.zeros((n, n)), columns=cols, index=cols)
ci_lower = pd.DataFrame(np.zeros((n, n)), columns=cols, index=cols)
ci_upper = pd.DataFrame(np.zeros((n, n)), columns=cols, index=cols)

# Spearman + CI の関数
def spearman_ci_pair(x, y, n_boot=2000, alpha=0.05, seed=0):
    data = np.column_stack([x, y])
    n = len(data)
    rng = np.random.default_rng(seed)

    boot_stats = np.empty(n_boot)
    for i in range(n_boot):
        idx = rng.integers(0, n, n)
        boot_stats[i], _ = spearmanr(data[idx, 0], data[idx, 1])

    lower = np.percentile(boot_stats, 100 * alpha / 2)
    upper = np.percentile(boot_stats, 100 * (1 - alpha / 2))
    r, p = spearmanr(x, y)
    return r, p, lower, upper

# 各セルに計算
for i in range(n):
    for j in range(n):
        if i == j:
            corr_matrix.iloc[i, j] = 1.0
            p_matrix.iloc[i, j] = 0.0
            ci_lower.iloc[i, j] = np.nan
            ci_upper.iloc[i, j] = np.nan
        else:
            r, p, low, high = spearman_ci_pair(df[cols[i]], df[cols[j]])
            corr_matrix.iloc[i, j] = r
            p_matrix.iloc[i, j] = p
            ci_lower.iloc[i, j] = low
            ci_upper.iloc[i, j] = high

# CSVへ出力
corr_matrix.to_csv("csv/spearman_corr_matrix.csv")
p_matrix.to_csv("csv/spearman_pvalues.csv")
ci_lower.to_csv("csv/spearman_ci_lower.csv")
ci_upper.to_csv("csv/spearman_ci_upper.csv")

print("Done.")
