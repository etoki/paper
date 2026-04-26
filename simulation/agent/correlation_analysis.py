"""Correlation and predictive comparison: Online learning behavior vs Big Five vs exam outcome.

Outcome: AcceptedUniversityHensachi (admission outcome on hensachi scale).
Predictor blocks:
  - Big Five domains (Openness, Conscientiousness, Extraversion, Agreeableness, Neuroticism)
  - Online learning behavior (NumberOfLecturesWatched, ViewingTime,
    NumberOfConfirmationTestsCompleted, NumberOfConfirmationTestsMastered,
    AverageFirstAttemptCorrectAnswerRate)
"""

from pathlib import Path
import json
import numpy as np
import pandas as pd
from scipy import stats
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import KFold, LeaveOneOut

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "data" / "raw.csv"
DS = ROOT / "data" / "raw_dataset.csv"
OUT_JSON = ROOT / "data" / "correlation_results.json"
OUT_MD = ROOT / "data" / "correlation_results.md"

BIGFIVE = ["Openness", "Conscientiousness", "Extraversion",
           "Agreeableness", "Neuroticism"]
BEHAVIOR = ["NumberOfLecturesWatched", "ViewingTime",
            "NumberOfConfirmationTestsCompleted",
            "NumberOfConfirmationTestsMastered",
            "AverageFirstAttemptCorrectAnswerRate"]

raw = pd.read_csv(RAW)
raw_all = raw[raw["category"] == "all"].copy()
ds = pd.read_csv(DS)[["ID", "AcceptedUniversityHensachi"]]
df = raw_all.merge(ds, on="ID", how="inner").dropna(subset=["AcceptedUniversityHensachi"])
N = len(df)

y = df["AcceptedUniversityHensachi"].to_numpy()


def corr(x):
    r, p = stats.pearsonr(x, y)
    rho, p_s = stats.spearmanr(x, y)
    return {"pearson_r": float(r), "pearson_p": float(p),
            "spearman_rho": float(rho), "spearman_p": float(p_s)}


univariate = {}
for col in BIGFIVE + BEHAVIOR:
    univariate[col] = corr(df[col].to_numpy())


def cv_metrics(X, y, splitter):
    preds = np.zeros_like(y, dtype=float)
    for tr, te in splitter.split(X):
        m = LinearRegression().fit(X[tr], y[tr])
        preds[te] = m.predict(X[te])
    r, _ = stats.pearsonr(preds, y)
    rmse = float(np.sqrt(np.mean((preds - y) ** 2)))
    mae = float(np.mean(np.abs(preds - y)))
    ss_res = float(np.sum((y - preds) ** 2))
    ss_tot = float(np.sum((y - y.mean()) ** 2))
    r2_oof = 1 - ss_res / ss_tot
    return {"pearson_r": float(r), "rmse": rmse, "mae": mae, "r2_oof": r2_oof}


def insample_r2(X, y):
    m = LinearRegression().fit(X, y)
    yhat = m.predict(X)
    ss_res = float(np.sum((y - yhat) ** 2))
    ss_tot = float(np.sum((y - y.mean()) ** 2))
    n, p = X.shape
    r2 = 1 - ss_res / ss_tot
    adj = 1 - (1 - r2) * (n - 1) / (n - p - 1) if (n - p - 1) > 0 else float("nan")
    coefs = {name: float(c) for name, c in zip(getattr(X, "columns", range(p)), m.coef_)}
    return {"r2": r2, "r2_adj": adj, "intercept": float(m.intercept_),
            "coefs": coefs}


blocks = {
    "BigFive_full": BIGFIVE,
    "BigFive_C_only": ["Conscientiousness"],
    "Behavior_full": BEHAVIOR,
    "Behavior_TestsMastered_only": ["NumberOfConfirmationTestsMastered"],
    "Combined_BigFive_plus_Behavior": BIGFIVE + BEHAVIOR,
}

kf = KFold(n_splits=10, shuffle=True, random_state=20260426)
loo = LeaveOneOut()

results = {"N": int(N), "univariate": univariate, "models": {}}
for name, cols in blocks.items():
    Xdf = df[cols]
    X = Xdf.to_numpy()
    ins = insample_r2(Xdf, y)
    cv10 = cv_metrics(X, y, kf)
    cvloo = cv_metrics(X, y, loo)
    results["models"][name] = {
        "predictors": cols,
        "in_sample": ins,
        "cv10": cv10,
        "loo": cvloo,
    }

OUT_JSON.write_text(json.dumps(results, indent=2, ensure_ascii=False))


def fmt_p(p):
    if p < 0.001:
        return "<.001"
    return f"{p:.3f}"


lines = []
lines.append(f"# 相関 / 予測精度分析 結果（N = {N}）\n")
lines.append("Outcome: AcceptedUniversityHensachi（合格大学の偏差値）\n")

lines.append("## 1. 単変量相関（Pearson r / Spearman ρ, vs Hensachi）\n")
lines.append("| 変数 | Pearson r | p | Spearman ρ | p |")
lines.append("|---|---:|---:|---:|---:|")
for col in BIGFIVE + BEHAVIOR:
    u = univariate[col]
    lines.append(f"| {col} | {u['pearson_r']:+.3f} | {fmt_p(u['pearson_p'])} "
                 f"| {u['spearman_rho']:+.3f} | {fmt_p(u['spearman_p'])} |")
lines.append("")

lines.append("## 2. ブロック予測精度比較（10-fold CV と LOO-CV）\n")
lines.append("| モデル | In-sample R² | Adj R² | CV10 r | CV10 RMSE | LOO r | LOO RMSE |")
lines.append("|---|---:|---:|---:|---:|---:|---:|")
for name, m in results["models"].items():
    ins = m["in_sample"]
    c = m["cv10"]
    l = m["loo"]
    lines.append(f"| {name} | {ins['r2']:.3f} | {ins['r2_adj']:.3f} "
                 f"| {c['pearson_r']:+.3f} | {c['rmse']:.3f} "
                 f"| {l['pearson_r']:+.3f} | {l['rmse']:.3f} |")
lines.append("")

lines.append("## 3. Conscientiousness × オンライン学習行動 相関\n")
lines.append("| 行動指標 | Pearson r vs C | p |")
lines.append("|---|---:|---:|")
c = df["Conscientiousness"].to_numpy()
for col in BEHAVIOR:
    r, p = stats.pearsonr(df[col].to_numpy(), c)
    lines.append(f"| {col} | {r:+.3f} | {fmt_p(p)} |")
lines.append("")

lines.append("## 4. 各モデル係数（in-sample OLS）\n")
for name, m in results["models"].items():
    lines.append(f"### {name}")
    lines.append(f"- intercept: {m['in_sample']['intercept']:+.3f}")
    for k, v in m["in_sample"]["coefs"].items():
        lines.append(f"- {k}: {v:+.3f}")
    lines.append("")

OUT_MD.write_text("\n".join(lines))
print(OUT_MD.read_text())
