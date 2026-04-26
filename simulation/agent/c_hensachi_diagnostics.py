"""Diagnose why Conscientiousness × Hensachi is not significant.

Checks:
  1. Distributions / range restriction
  2. Outliers (Cook's D, leverage, robust correlations)
  3. Conscientiousness facets (Organization / Productiveness / Responsibility)
  4. Power analysis (post hoc) and confidence intervals
  5. Non-linearity (rank-based, quadratic)
"""

from pathlib import Path
import json
import numpy as np
import pandas as pd
from scipy import stats

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "data" / "raw.csv"
DS = ROOT / "data" / "raw_dataset.csv"
OUT = ROOT / "data" / "c_hensachi_diagnostics.md"

raw = pd.read_csv(RAW)
raw_all = raw[raw["category"] == "all"].copy()
ds = pd.read_csv(DS)[["ID", "AcceptedUniversityHensachi"]]
df = raw_all.merge(ds, on="ID").dropna(subset=["AcceptedUniversityHensachi"])
N = len(df)
y = df["AcceptedUniversityHensachi"].to_numpy()
c = df["Conscientiousness"].to_numpy()

lines = [f"# Conscientiousness × Hensachi 診断（N={N}）", ""]


# 1. Distributions
def desc(x, name):
    return (f"- **{name}**: mean={np.mean(x):.2f}, sd={np.std(x, ddof=1):.2f}, "
            f"min={np.min(x):.2f}, max={np.max(x):.2f}, "
            f"IQR=[{np.percentile(x,25):.2f}, {np.percentile(x,75):.2f}]")


lines += ["## 1. 分布 / 範囲制限",
          desc(c, "Conscientiousness"),
          desc(y, "Hensachi"),
          ""]

# Hensachi の母集団基準は mean=50, sd=10。観測 sd と比べる。
y_sd = np.std(y, ddof=1)
lines.append(f"母集団基準 SD=10 に対して観測 SD={y_sd:.2f} "
             f"→ {'範囲制限の兆候あり' if y_sd < 9 else '顕著な範囲制限はない'}")
lines.append("")


# 2. Robust / outlier check
def fisher_ci(r, n, alpha=0.05):
    z = np.arctanh(r)
    se = 1 / np.sqrt(n - 3)
    zcrit = stats.norm.ppf(1 - alpha / 2)
    lo, hi = np.tanh(z - zcrit * se), np.tanh(z + zcrit * se)
    return lo, hi


r_p, p_p = stats.pearsonr(c, y)
rho_s, p_s = stats.spearmanr(c, y)
lo, hi = fisher_ci(r_p, N)
lines += ["## 2. 相関と 95% CI",
          f"- Pearson  r = {r_p:+.3f} [95% CI {lo:+.3f}, {hi:+.3f}], p={p_p:.3f}",
          f"- Spearman ρ = {rho_s:+.3f}, p={p_s:.3f}",
          ""]

# Winsorize 5% の Pearson（外れ値耐性）
from scipy.stats.mstats import winsorize
cw = winsorize(c, limits=[0.05, 0.05]).data
yw = winsorize(y, limits=[0.05, 0.05]).data
r_w, p_w = stats.pearsonr(cw, yw)
lines.append(f"- 5% Winsorized Pearson r = {r_w:+.3f}, p={p_w:.3f}")

# Cook's distance / leverage
import numpy.linalg as la
X = np.column_stack([np.ones(N), c])
beta = la.lstsq(X, y, rcond=None)[0]
yhat = X @ beta
resid = y - yhat
mse = float(np.sum(resid ** 2) / (N - 2))
H = X @ la.inv(X.T @ X) @ X.T
h = np.diag(H)
cook = (resid ** 2 / (2 * mse)) * (h / (1 - h) ** 2)
top = np.argsort(cook)[-5:][::-1]
lines.append("")
lines.append("Cook's D 上位 5 サンプル（影響の強い観測）")
lines.append("| ID | C | Hensachi | resid | leverage | Cook's D |")
lines.append("|---|---:|---:|---:|---:|---:|")
for i in top:
    lines.append(f"| {df['ID'].iloc[i]} | {c[i]:.2f} | {y[i]:.2f} "
                 f"| {resid[i]:+.2f} | {h[i]:.3f} | {cook[i]:.3f} |")
lines.append("")

# 上位 5 個を除外したときの r
keep = np.ones(N, dtype=bool)
keep[top] = False
r_drop, p_drop = stats.pearsonr(c[keep], y[keep])
lines.append(f"上位 5 観測除外後: Pearson r = {r_drop:+.3f}, p={p_drop:.3f}, n={int(keep.sum())}")
lines.append("")


# 3. Facets
lines += ["## 3. Conscientiousness ファセット（BFI-2 3 ファセット）",
          "| ファセット | r | p |", "|---|---:|---:|"]
for facet in ["Organization", "Productiveness", "Responsibility"]:
    rr, pp = stats.pearsonr(df[facet].to_numpy(), y)
    lines.append(f"| {facet} | {rr:+.3f} | {pp:.3f} |")
lines.append("")

# 全 BFI-2 ファセット 15 個
all_facets = ["IntellectualCuriosity", "AestheticSensitivity", "CreativeImagination",
              "Organization", "Productiveness", "Responsibility",
              "Sociability", "Assertiveness", "EnergyLevel",
              "Compassion", "Respectfulness", "Trust",
              "Anxiety", "Depression", "EmotionalVolatility"]
lines.append("### 参考: 全 15 ファセット")
lines.append("| ファセット | r | p |")
lines.append("|---|---:|---:|")
for f in all_facets:
    rr, pp = stats.pearsonr(df[f].to_numpy(), y)
    star = " *" if pp < 0.05 else ""
    lines.append(f"| {f} | {rr:+.3f}{star} | {pp:.3f} |")
lines.append("")


# 4. Power / required N
def required_n_for_r(r, alpha=0.05, power=0.80):
    zcrit = stats.norm.ppf(1 - alpha / 2)
    zpow = stats.norm.ppf(power)
    z = np.arctanh(r)
    return int(np.ceil(((zcrit + zpow) / z) ** 2 + 3))


# Post-hoc power for observed r
def power_pearson(r, n, alpha=0.05):
    zcrit = stats.norm.ppf(1 - alpha / 2)
    z = np.arctanh(r) * np.sqrt(n - 3)
    return float(stats.norm.cdf(z - zcrit) + stats.norm.cdf(-z - zcrit))


lines += ["## 4. 検定力 / 必要 N",
          f"- 観測 r={r_p:.3f} を α=.05 両側で有意検出する必要 N: "
          f"**{required_n_for_r(r_p)}**（現 N={N}）",
          f"- 先行メタ分析 r=0.167 (Tokiwa 2026) を検出する必要 N: "
          f"**{required_n_for_r(0.167)}**",
          f"- 古典 Poropat (2009) r=0.19 を検出する必要 N: "
          f"**{required_n_for_r(0.19)}**",
          f"- 現 N={N}, r={r_p:.3f} の post-hoc power: "
          f"**{power_pearson(r_p, N):.2f}**",
          f"- 現 N={N} で r=0.20 を検出する power: "
          f"**{power_pearson(0.20, N):.2f}**",
          ""]


# 5. Non-linearity
# Quadratic
c_centered = c - c.mean()
Xq = np.column_stack([np.ones(N), c_centered, c_centered ** 2])
beta_q, *_ = la.lstsq(Xq, y, rcond=None)
yhat_q = Xq @ beta_q
ss_res_q = float(np.sum((y - yhat_q) ** 2))
ss_tot = float(np.sum((y - y.mean()) ** 2))
r2_q = 1 - ss_res_q / ss_tot
# F-test for the quadratic term
ss_res_l = float(np.sum((y - yhat) ** 2))
F = ((ss_res_l - ss_res_q) / 1) / (ss_res_q / (N - 3))
p_F = 1 - stats.f.cdf(F, 1, N - 3)
lines += ["## 5. 非線形性チェック",
          f"- 線形 R² = {1 - ss_res_l / ss_tot:.4f}",
          f"- 二次項追加 R² = {r2_q:.4f}（F={F:.2f}, p={p_F:.3f}）",
          ""]

OUT.write_text("\n".join(lines))
print(OUT.read_text())
