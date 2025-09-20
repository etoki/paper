import os
import sys
import platform
import json
import numpy as np
import pandas as pd

from scipy.stats import spearmanr
import statsmodels.api as sm
from statsmodels.stats.multitest import multipletests
from statsmodels.stats.outliers_influence import variance_inflation_factor
from sklearn.preprocessing import StandardScaler
from datetime import datetime

# ---------------------------------------------------------------------
# 0) Setup
# ---------------------------------------------------------------------
os.makedirs("res", exist_ok=True)
np.set_printoptions(suppress=True)
pd.set_option("display.width", 140)
pd.set_option("display.max_columns", 200)

# User-editable column groups
traits_cols = ["hexaco_HH","hexaco_E","hexaco_X","hexaco_A","hexaco_C","hexaco_O"]
dark_cols   = ["Machiavellianism","Narcissism","Psychopathy"]
harass_cols = ["power_harassment","gender_harassment"]

# Other variables expected in raw.csv
# gender: nominal; age: numeric; area: nominal
numeric_like = traits_cols + dark_cols + harass_cols + ["age"]  # gender & area handled as dummies

# ---------------------------------------------------------------------
# 1) Load & cast
# ---------------------------------------------------------------------
df = pd.read_csv("raw.csv", na_values=["NA","N/A","na","NaN","-", ""])
# Try to coerce known numeric-like columns
for c in numeric_like:
    if c in df.columns:
        df[c] = pd.to_numeric(df[c], errors="coerce")

# Dummy-encode gender and area (drop first category as baseline)
# Only if columns exist and are not already dummies
cat_cols = []
if "gender" in df.columns:
    cat_cols.append("gender")
if "area" in df.columns:
    cat_cols.append("area")

if len(cat_cols) > 0:
    df = pd.get_dummies(df, columns=cat_cols, drop_first=True, dtype=float)

# Collect control dummies created
control_dummy_cols = [c for c in df.columns if c.startswith("gender_") or c.startswith("area_")]
control_vars = (["age"] if "age" in df.columns else []) + control_dummy_cols

# ---------------------------------------------------------------------
# 2) Descriptive statistics
# ---------------------------------------------------------------------
numeric_cols = df.select_dtypes(include=[np.number]).columns
desc_stats = df[numeric_cols].describe().T
desc_stats["median"] = df[numeric_cols].median()
desc_stats.to_csv("res/descriptive_statistics.csv")

# ---------------------------------------------------------------------
# 3) Spearman correlations (+p, BH-FDR on upper triangle), pretty table
# ---------------------------------------------------------------------
all_cols = [c for c in (traits_cols + dark_cols + harass_cols) if c in df.columns]
corr_matrix = pd.DataFrame(np.nan, index=all_cols, columns=all_cols, dtype=float)
pval_matrix = pd.DataFrame(np.nan, index=all_cols, columns=all_cols, dtype=float)

for r in all_cols:
    for c in all_cols:
        rho, p = spearmanr(df[r], df[c], nan_policy="omit")
        corr_matrix.loc[r, c] = rho
        pval_matrix.loc[r, c] = p

# FDR (Benjamini–Hochberg) for upper triangle p-values
pairs = []
for i, r in enumerate(all_cols):
    for j, c in enumerate(all_cols):
        if j > i:
            pairs.append((r, c, pval_matrix.loc[r, c]))

pvals = [p for (_,_,p) in pairs]
if len(pvals) > 0:
    rej, qvals, _, _ = multipletests(pvals, method="fdr_bh")
    # Map back to matrix
    qmap = {}
    for (rc, qc) in zip(pairs, qvals):
        r, c, _ = rc
        qmap[(r,c)] = qc
        qmap[(c,r)] = qc  # mirror for convenience
else:
    qmap = {}

# Create star table (upper triangle), showing rho with stars based on raw p-values
def star(p):
    return "***" if p < 0.001 else "**" if p < 0.01 else "*" if p < 0.05 else ""

corr_with_stars = pd.DataFrame("", index=all_cols, columns=all_cols, dtype=object)
for i, r in enumerate(all_cols):
    for j, c in enumerate(all_cols):
        if r == c:
            corr_with_stars.loc[r, c] = "1.00"
        elif j > i:
            rho = corr_matrix.loc[r, c]
            p = pval_matrix.loc[r, c]
            corr_with_stars.loc[r, c] = "" if pd.isna(rho) else f"{rho:.2f}{star(p)}"
        else:
            corr_with_stars.loc[r, c] = ""

# Save correlation artifacts
corr_matrix.to_csv("res/spearman_corr_rho.csv")
pval_matrix.to_csv("res/spearman_corr_p.csv")
# Save q-values as long table (upper triangle)
corr_pq_long = []
for i, r in enumerate(all_cols):
    for j, c in enumerate(all_cols):
        if j > i:
            corr_pq_long.append({
                "row": r, "col": c,
                "rho": corr_matrix.loc[r, c],
                "p": pval_matrix.loc[r, c],
                "q_bh": qmap.get((r,c), np.nan)
            })
pd.DataFrame(corr_pq_long).to_csv("res/spearman_corr_p_q_upper.csv", index=False)
corr_with_stars.to_csv("res/spearman_corr_table.csv")

# ---------------------------------------------------------------------
# 4) Helpers for regression
# ---------------------------------------------------------------------
def standardize_xy(X: pd.DataFrame, y: pd.Series):
    """Return standardized X and y (z-scores) as DataFrame/Series with original names."""
    Xz = pd.DataFrame(StandardScaler().fit_transform(X), columns=X.columns, index=X.index)
    yz = pd.Series(StandardScaler().fit_transform(y.to_numpy().reshape(-1,1)).ravel(),
                   name=y.name, index=y.index)
    return Xz, yz

def compute_vif(X: pd.DataFrame):
    """Compute VIF (excludes constant)."""
    Xc = sm.add_constant(X, has_constant="add")
    vif = pd.DataFrame({
        "variable": Xc.columns,
        "VIF": [variance_inflation_factor(Xc.values, i) for i in range(Xc.shape[1])]
    })
    vif = vif[vif["variable"] != "const"]
    return vif

def clean_and_align_for_model(df: pd.DataFrame, dv: str, ivs: list):
    """Select dv + ivs, coerce numeric, drop rows with NA."""
    use = pd.concat([df[[dv]], df[ivs]], axis=1)
    use = use.apply(pd.to_numeric, errors="coerce").dropna()
    y = use[dv].astype(float)
    X = use.drop(columns=[dv]).astype(float)
    return X, y, len(use)

# ---------------------------------------------------------------------
# 5) Multiple regression (standardized β, HC3 robust SE) + VIF + BH-FDR per DV
# ---------------------------------------------------------------------
# Base IVs = HEXACO + controls
iv_base = [c for c in traits_cols if c in df.columns] + control_vars
iv_base = [c for c in iv_base if c in df.columns]  # keep existing

targets = [c for c in (dark_cols + harass_cols) if c in df.columns]

reg_rows = []
vif_rows = []

for dv in targets:
    # Align data
    X, y, n_used = clean_and_align_for_model(df, dv, iv_base)
    if X.empty or y.empty or n_used < 10:
        continue

    # Standardize (to get betas)
    Xz, yz = standardize_xy(X, y)
    Xz = sm.add_constant(Xz, has_constant="add")
    # Fit with HC3 robust SE
    model = sm.OLS(yz, Xz).fit(cov_type="HC3")

    # Collect coefficients
    coefs = pd.DataFrame({
        "Dependent_Var": dv,
        "Variable": model.params.index,
        "Beta": model.params.values,
        "Std_Error(HC3)": model.bse.values,
        "t_value": model.tvalues.values,
        "p_value": model.pvalues.values,
        "N_used": n_used,
        "R2": model.rsquared,
        "Adj_R2": model.rsquared_adj
    })
    coefs = coefs[coefs["Variable"] != "const"]
    reg_rows.append(coefs)

    # VIF diagnostics (on unstandardized X is fine)
    vif = compute_vif(X)
    vif["Dependent_Var"] = dv
    vif_rows.append(vif)

# Combine and FDR-adjust p-values within each DV (recommended)
if len(reg_rows) > 0:
    reg_all = pd.concat(reg_rows, axis=0).reset_index(drop=True)
    reg_all["q_value_bh_withinDV"] = np.nan
    for dv in reg_all["Dependent_Var"].unique():
        idx = reg_all["Dependent_Var"] == dv
        rej, q, _, _ = multipletests(reg_all.loc[idx, "p_value"], method="fdr_bh")
        reg_all.loc[idx, "q_value_bh_withinDV"] = q
    reg_all["Signif(q<.05)"] = reg_all["q_value_bh_withinDV"] < 0.05
    reg_all.sort_values(["Dependent_Var","p_value"], inplace=True)
    reg_all.to_csv("res/regression_beta_hc3_fdr.csv", index=False)
else:
    reg_all = pd.DataFrame(columns=[
        "Dependent_Var","Variable","Beta","Std_Error(HC3)","t_value","p_value",
        "N_used","R2","Adj_R2","q_value_bh_withinDV","Signif(q<.05)"
    ]).to_csv("res/regression_beta_hc3_fdr.csv", index=False)

if len(vif_rows) > 0:
    pd.concat(vif_rows, axis=0).to_csv("res/vif.csv", index=False)
else:
    pd.DataFrame(columns=["variable","VIF","Dependent_Var"]).to_csv("res/vif.csv", index=False)

# ---------------------------------------------------------------------
# 6) Hierarchical regression (incremental validity: add Dark Triad after HEXACO+controls)
#     Run only for harassment DVs (if available)
# ---------------------------------------------------------------------
def hierarchical_r2_change(y, X_step1, X_step2):
    """Return dict with R², ΔR², approx ΔF (using standard formula)."""
    X1 = sm.add_constant(X_step1, has_constant="add")
    X2 = sm.add_constant(pd.concat([X_step1, X_step2], axis=1), has_constant="add")
    m1 = sm.OLS(y, X1).fit(cov_type="HC3")
    m2 = sm.OLS(y, X2).fit(cov_type="HC3")

    r2_change = m2.rsquared - m1.rsquared
    df1 = X_step2.shape[1]
    df2 = m2.df_resid
    # Approximate ΔF (same as nested models ANOVA under homoskedasticity; robust p is not trivial)
    try:
        F = ((m2.rsquared - m1.rsquared) / df1) / ((1 - m2.rsquared) / df2)
    except ZeroDivisionError:
        F = np.nan

    return {
        "R2_step1": m1.rsquared,
        "R2_step2": m2.rsquared,
        "Delta_R2": r2_change,
        "Delta_F_approx": F
    }

hier_rows = []

hex_controls = [c for c in traits_cols if c in df.columns] + control_vars
hex_controls = [c for c in hex_controls if c in df.columns]
dt_cols = [c for c in dark_cols if c in df.columns]

for dv in [c for c in harass_cols if c in df.columns]:
    # Prepare aligned data for dv with HEXACO+controls+DT
    need_cols = [dv] + hex_controls + dt_cols
    use = df[need_cols].apply(pd.to_numeric, errors="coerce").dropna()
    if use.shape[0] < 10:
        continue

    y = use[dv].astype(float)
    X_hex = use[hex_controls].astype(float)
    X_dt  = use[dt_cols].astype(float)

    # Standardize y and each block (to report comparable betas if needed)
    yz = StandardScaler().fit_transform(y.to_numpy().reshape(-1,1)).ravel()
    X_hex_z = pd.DataFrame(StandardScaler().fit_transform(X_hex), columns=X_hex.columns, index=use.index)
    X_dt_z  = pd.DataFrame(StandardScaler().fit_transform(X_dt),  columns=X_dt.columns,  index=use.index)

    out = hierarchical_r2_change(yz, X_hex_z, X_dt_z)
    out["Dependent_Var"] = dv
    out["N_used"] = use.shape[0]
    hier_rows.append(out)

if len(hier_rows) > 0:
    pd.DataFrame(hier_rows).sort_values("Dependent_Var").to_csv("res/hierarchical_incremental_hexaco_then_dt.csv", index=False)
else:
    pd.DataFrame(columns=["R2_step1","R2_step2","Delta_R2","Delta_F_approx","Dependent_Var","N_used"])\
      .to_csv("res/hierarchical_incremental_hexaco_then_dt.csv", index=False)

# ---------------------------------------------------------------------
# 7) Session info for reproducibility
# ---------------------------------------------------------------------
session = {
    "datetime": datetime.now().isoformat(),
    "python": sys.version,
    "platform": platform.platform(),
    "packages": {
        "numpy": np.__version__,
        "pandas": pd.__version__,
        "statsmodels": sm.__version__
    },
    "columns_present": list(df.columns),
    "traits_cols_used": [c for c in traits_cols if c in df.columns],
    "dark_cols_used": [c for c in dark_cols if c in df.columns],
    "harass_cols_used": [c for c in harass_cols if c in df.columns],
    "control_vars_used": control_vars
}
with open("res/_sessioninfo.txt", "w", encoding="utf-8") as f:
    f.write(json.dumps(session, ensure_ascii=False, indent=2))

print("Analysis finished. Outputs written to ./res")