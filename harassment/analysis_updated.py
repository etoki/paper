import os
import pandas as pd
import numpy as np
from scipy.stats import spearmanr, shapiro
import statsmodels.api as sm
from statsmodels.stats.outliers_influence import variance_inflation_factor
from statsmodels.stats.diagnostic import het_breuschpagan
from statsmodels.stats.stattools import durbin_watson

# -----------------------------
# Setup
# -----------------------------
os.makedirs("res", exist_ok=True)

# -----------------------------
# Load dataset & type conversion
# -----------------------------
df = pd.read_csv("raw.csv", na_values=["NA", "N/A", "na", "NaN", "-", ""])

traits_cols = ["hexaco_HH","hexaco_E","hexaco_X","hexaco_A","hexaco_C","hexaco_O"]
dark_cols   = ["Machiavellianism","Narcissism","Psychopathy"]
harass_cols = ["power_harassment","gender_harassment"]

# Convert main columns to numeric
cols_to_numeric = traits_cols + dark_cols + harass_cols + ["age","gender"]
for c in cols_to_numeric:
    if c in df.columns:
        df[c] = pd.to_numeric(df[c], errors="coerce")

# -----------------------------
# 0) Missingness report
# -----------------------------
miss = df[[c for c in cols_to_numeric if c in df.columns]].isna().mean().rename("missing_rate")
miss.to_csv("res/missingness_report.csv")

# -----------------------------
# 1) Descriptive statistics
# -----------------------------
numeric_cols = df.select_dtypes(include=[np.number]).columns
desc_stats = df[numeric_cols].describe().T
desc_stats["median"] = df[numeric_cols].median()
desc_stats["skew"] = df[numeric_cols].skew()
desc_stats["kurtosis"] = df[numeric_cols].kurt()
desc_stats.to_csv("res/descriptive_statistics.csv")

# -----------------------------
# 2) Spearman correlations (with stars, upper triangle)
# -----------------------------
all_cols = [c for c in (traits_cols + dark_cols + harass_cols) if c in df.columns]

corr_matrix = pd.DataFrame(index=all_cols, columns=all_cols, dtype=float)
pval_matrix = pd.DataFrame(index=all_cols, columns=all_cols, dtype=float)
for c1 in all_cols:
    for c2 in all_cols:
        rho, p = spearmanr(df[c1], df[c2], nan_policy="omit")
        corr_matrix.loc[c1, c2] = rho
        pval_matrix.loc[c1, c2] = p

def _star(p):
    return "***" if p < 0.001 else "**" if p < 0.01 else "*" if p < 0.05 else ""

corr_with_stars = pd.DataFrame(index=all_cols, columns=all_cols, dtype=object)
for r in all_cols:
    for c in all_cols:
        if r == c:
            corr_with_stars.loc[r, c] = "1.00"
        else:
            rho = corr_matrix.loc[r, c]
            p = pval_matrix.loc[r, c]
            corr_with_stars.loc[r, c] = "" if pd.isna(rho) else f"{rho:.2f}{_star(p)}"

corr_upper = corr_with_stars.copy()
for i, r in enumerate(all_cols):
    for j, c in enumerate(all_cols):
        if j < i:
            corr_upper.loc[r, c] = ""

corr_upper.to_csv("res/spearman_corr_table.csv")

# -----------------------------
# 3) Multiple regression with controls
#    - Standardized betas
#    - Robust SE (HC3)
#    - VIF per model
#    - Model fit: R2, adj R2, F, AIC/BIC, n
#    - Diagnostics: DW, Shapiro (residual normality), BP (heteroskedasticity)
#    - Influence: Cook's distance flags (> 4/n)
# -----------------------------

# Create area dummies if present
if "area" in df.columns:
    df_dummies = pd.get_dummies(df, columns=["area"], drop_first=True, dtype=float)
else:
    df_dummies = df.copy()

control_vars = [v for v in ["age","gender"] if v in df_dummies.columns] + \
               [c for c in df_dummies.columns if c.startswith("area_")]

# Base predictors = HEXACO traits + controls
X_base_cols = [c for c in traits_cols if c in df_dummies.columns] + control_vars

targets = [c for c in (dark_cols + harass_cols) if c in df_dummies.columns]

def standardize(series: pd.Series) -> pd.Series:
    s = series.astype(float)
    return (s - s.mean()) / s.std(ddof=0)

reg_rows = []
coef_rows = []
vif_all = []

for target in targets:
    use_cols = [target] + X_base_cols
    use = df_dummies[use_cols].apply(pd.to_numeric, errors="coerce").dropna()
    n = len(use)
    if n < 10:
        continue

    # Standardize to get beta coefficients
    y = standardize(use[target])
    X = use[X_base_cols].copy().apply(standardize)
    X = sm.add_constant(X, has_constant="add")

    # OLS with robust (HC3) SE
    model = sm.OLS(y, X).fit(cov_type="HC3")

    # Model-level stats
    dw = durbin_watson(model.resid)
    shapiro_p = np.nan
    try:
        if n <= 5000:
            shapiro_p = shapiro(model.resid)[1]
    except Exception:
        pass

    # Breusch-Pagan (heteroskedasticity)
    try:
        bp_lm, bp_lm_p, bp_f, bp_f_p = het_breuschpagan(model.resid, model.model.exog)
    except Exception:
        bp_lm, bp_lm_p, bp_f, bp_f_p = [np.nan]*4

    # Influence / Cook's distance
    infl = model.get_influence()
    cooks = infl.cooks_distance[0]
    cooks_flag = (cooks > (4/n)).sum()

    reg_rows.append({
        "Dependent_Var": target,
        "n": n,
        "R2": model.rsquared,
        "Adj_R2": model.rsquared_adj,
        "F_stat": model.fvalue,
        "F_p": model.f_pvalue,
        "AIC": model.aic,
        "BIC": model.bic,
        "Durbin_Watson": dw,
        "Shapiro_p": shapiro_p,
        "BP_LM_p": bp_lm_p,
        "BP_F_p": bp_f_p,
        "Max_CooksD": np.nanmax(cooks) if len(cooks)>0 else np.nan,
        "Num_CooksD_gt_4n": int(cooks_flag)
    })

    # Coefficients table (standardized betas with robust SE)
    ctab = pd.DataFrame({
        "Dependent_Var": target,
        "Variable": model.params.index,
        "Std_Beta": model.params.values,
        "Std_Error_Robust": model.bse.values,
        "t_value_Robust": model.tvalues.values,
        "p_value_Robust": model.pvalues.values
    })
    ctab = ctab[ctab["Variable"] != "const"]
    ctab["Significance"] = ctab["p_value_Robust"].apply(lambda p: "***" if p<0.001 else "**" if p<0.01 else "*" if p<0.05 else "")
    coef_rows.append(ctab)

    # VIF (on unstandardized X without constant)
    X_vif = sm.add_constant(use[X_base_cols].astype(float), has_constant="add")
    vif_tbl = pd.DataFrame({
        "Dependent_Var": target,
        "Variable": ["const"] + X_base_cols,
        "VIF": [np.nan] + [variance_inflation_factor(X_vif.values, i) for i in range(1, X_vif.shape[1])]
    })
    vif_all.append(vif_tbl)

# Save model-level summary
reg_df = pd.DataFrame(reg_rows)
reg_df = reg_df.sort_values(["Dependent_Var"]).reset_index(drop=True)
reg_df.to_csv("res/regression_model_fit.csv", index=False)

# Save coefficients (significant and full)
if len(coef_rows):
    coef_df = pd.concat(coef_rows, axis=0).reset_index(drop=True)
    coef_df.to_csv("res/regression_coefficients_full.csv", index=False)
    sig_df = coef_df[coef_df["p_value_Robust"] < 0.05].copy()
    sig_df.to_csv("res/regression_coefficients_sig.csv", index=False)
else:
    pd.DataFrame(columns=["Dependent_Var","Variable","Std_Beta","Std_Error_Robust","t_value_Robust","p_value_Robust","Significance"]).to_csv("res/regression_coefficients_full.csv", index=False)
    pd.DataFrame(columns=["Dependent_Var","Variable","Std_Beta","Std_Error_Robust","t_value_Robust","p_value_Robust","Significance"]).to_csv("res/regression_coefficients_sig.csv", index=False)

# Save VIFs
if len(vif_all):
    vif_df = pd.concat(vif_all, axis=0).reset_index(drop=True)
    vif_df.to_csv("res/vif_by_model.csv", index=False)
else:
    pd.DataFrame(columns=["Dependent_Var","Variable","VIF"]).to_csv("res/vif_by_model.csv", index=False)

print("Analysis complete. Files saved under ./res")
