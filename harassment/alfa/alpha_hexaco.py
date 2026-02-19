import pandas as pd
import pingouin as pg
import re

# ===== 入力ファイル（同じフォルダに置く）=====
FILES = [
    "raw_hexaco_app_hexaco-jp.csv",
    "raw_hexaco_w60.csv",
    "raw_hexaco_app_hexaco.csv",
]

# HEXACOドメイン（列名にこの文字列が入っている）
DOMAINS = [
    "Honesty-Humility",
    "Emotionality",
    "Extraversion",
    "Agreeableness",
    "Conscientiousness",
    "Openness",
]

def domain_cols(columns, domain_name: str):
    """
    例:
      Openness, Openness.1, Openness.2 ... を全部拾う
    """
    pattern = re.compile(rf"^{re.escape(domain_name)}(\.\d+)?$")
    return [c for c in columns if pattern.match(c)]

# ===== 1) 読み込み =====
dfs = []
for f in FILES:
    df = pd.read_csv(f)
    dfs.append(df)

# ===== 2) 列の整合性チェック（列順が違ってもOK。列名が一致しているか確認）=====
base_cols = set(dfs[0].columns)
for i, df in enumerate(dfs[1:], start=2):
    if set(df.columns) != base_cols:
        missing = base_cols - set(df.columns)
        extra = set(df.columns) - base_cols
        raise ValueError(
            f"[ERROR] Columns differ in file #{i}.\n"
            f"Missing: {sorted(list(missing))[:20]}\n"
            f"Extra: {sorted(list(extra))[:20]}"
        )

# ===== 3) 結合（列名で自動整列されるので列順差は問題なし）=====
df_all = pd.concat(dfs, ignore_index=True)

# ===== 4) 数値化（email以外を数値へ）=====
non_email_cols = [c for c in df_all.columns if c != "email"]
df_all[non_email_cols] = df_all[non_email_cols].apply(pd.to_numeric, errors="coerce")

print(f"Total rows (combined) = {len(df_all)}")
print(f"Columns = {len(df_all.columns)} (including email)")

# ===== 5) ドメイン別 Cronbach's alpha =====
print("\n=== HEXACO Domain Cronbach's α (combined 3 files) ===")

results = []
for d in DOMAINS:
    cols = domain_cols(df_all.columns, d)
    if len(cols) < 2:
        print(f"{d}: ERROR (found {len(cols)} items)")
        continue

    # ドメイン計算に必要な列だけで欠損を落とす（ドメインごと）
    tmp = df_all[cols].dropna()

    alpha_val, ci = pg.cronbach_alpha(data=tmp)

    print(f"{d:18s} α = {alpha_val:.3f} | k={len(cols)} | n={len(tmp)}")

    results.append({
        "domain": d,
        "alpha": round(alpha_val, 3),
        "k_items": len(cols),
        "n_used": len(tmp)
    })

# ===== 6) 結果をCSV保存（任意）=====
out = pd.DataFrame(results)
out.to_csv("hexaco_alpha_results.csv", index=False)
print("\nSaved: hexaco_alpha_results.csv")
