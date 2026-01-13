import pandas as pd
from scipy.stats import shapiro

# HEXACO ドメインデータの読み込み
df = pd.read_csv("csv/hexaco_domain.csv")

results = []

for col in df.columns:
    stat, p = shapiro(df[col])
    results.append([col, stat, p])

normality_df = pd.DataFrame(results, columns=["Trait", "Shapiro_W", "p_value"])

# 結果を保存
normality_df.to_csv("csv/hexaco_normality_results.csv", index=False)

print(normality_df)
