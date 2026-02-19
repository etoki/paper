import pandas as pd
import pingouin as pg

# ===== データ読み込み =====
df = pd.read_csv("raw_gender-harassment.csv")

# ===== 列名確認（最初に一度実行）=====
print(df.columns)

# ===== Commission（GHC）列抽出 =====
ghc_cols = [col for col in df.columns if col.startswith("GHC")]

# ===== Omission（GHO）列抽出 =====
gho_cols = [col for col in df.columns if col.startswith("GHO")]

# ===== 欠損除去（必要なら）=====
df = df.dropna()

# ===== α計算 =====
alpha_commission = pg.cronbach_alpha(data=df[ghc_cols])
alpha_omission = pg.cronbach_alpha(data=df[gho_cols])

# ===== 全体（13項目）=====
alpha_total = pg.cronbach_alpha(data=df[ghc_cols + gho_cols])

# ===== 結果表示 =====
print("Commission (GHC) α =", round(alpha_commission[0], 3))
print("Omission (GHO) α =", round(alpha_omission[0], 3))
print("Total Gender Harassment α =", round(alpha_total[0], 3))
