import pandas as pd
import pingouin as pg
import numpy as np

# ===== データ読み込み =====
df = pd.read_csv("raw_sd3j.csv")

# ===== 逆転項目がある場合ここで処理 =====
# 例：5件法なら reverse = 6 - 元の値
# df["MC3"] = 6 - df["MC3"]
# （逆転項目がある場合のみ使う）

# ===== Machiavellianism =====
mc_cols = [f"MC{i}" for i in range(1, 10)]
alpha_mc = pg.cronbach_alpha(data=df[mc_cols])

# ===== Narcissism =====
nr_cols = [f"NR{i}" for i in range(1, 10)]
alpha_nr = pg.cronbach_alpha(data=df[nr_cols])

# ===== Psychopathy =====
ps_cols = [f"PS{i}" for i in range(1, 10)]
alpha_ps = pg.cronbach_alpha(data=df[ps_cols])

# ===== SD3全体（27項目） =====
all_cols = mc_cols + nr_cols + ps_cols
alpha_total = pg.cronbach_alpha(data=df[all_cols])

# ===== 結果表示 =====
print("Machiavellianism α =", round(alpha_mc[0], 3))
print("Narcissism α =", round(alpha_nr[0], 3))
print("Psychopathy α =", round(alpha_ps[0], 3))
print("SD3 Total α =", round(alpha_total[0], 3))
