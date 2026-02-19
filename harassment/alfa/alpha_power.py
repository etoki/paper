import pandas as pd
import pingouin as pg

# ===== データ読み込み =====
df = pd.read_csv("raw_power-harassment.csv")

# ===== 列名確認（最初に一度実行）=====
print(df.columns)

# ===== 行動（behavior）=====
phb_cols = [col for col in df.columns if col.startswith("PHB")]

# ===== 状態／風土（state/climate）=====
phs_cols = [col for col in df.columns if col.startswith("PHS")]

# ===== 権威主義的態度（attitude）=====
pha_cols = [col for col in df.columns if col.startswith("AA")]

# ===== 欠損除去（必要なら）=====
df = df.dropna()

# ===== α計算 =====
alpha_behavior = pg.cronbach_alpha(data=df[phb_cols])
alpha_state = pg.cronbach_alpha(data=df[phs_cols])
alpha_attitude = pg.cronbach_alpha(data=df[pha_cols])

# ===== 全体（18項目）=====
alpha_total = pg.cronbach_alpha(data=df[phb_cols + phs_cols + pha_cols])

# ===== 結果表示 =====
print("Behavior α =", round(alpha_behavior[0], 3))
print("State/Climate α =", round(alpha_state[0], 3))
print("Authoritarian Attitude α =", round(alpha_attitude[0], 3))
print("Total Power Harassment α =", round(alpha_total[0], 3))
