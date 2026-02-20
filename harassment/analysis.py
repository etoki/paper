
import os
import pandas as pd
import numpy as np

from scipy.stats import spearmanr, shapiro
import scipy.stats as st
import statsmodels.api as sm
from statsmodels.stats.outliers_influence import variance_inflation_factor
from statsmodels.stats.diagnostic import het_breuschpagan
from statsmodels.stats.stattools import durbin_watson
from statsmodels.stats.anova import anova_lm

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

# Convert main columns to numeric when present
cols_to_numeric = traits_cols + dark_cols + harass_cols + ["age","gender"]
for c in cols_to_numeric:
    if c in df.columns:
        df[c] = pd.to_numeric(df[c], errors="coerce")

# -----------------------------
# 0) Missingness report
# -----------------------------
miss_cols = [c for c in cols_to_numeric if c in df.columns]
if len(miss_cols):
    miss = df[miss_cols].isna().mean().rename("missing_rate")
    miss.to_csv("res/missingness_report.csv")

# -----------------------------
# 1) Descriptive statistics
# -----------------------------
numeric_cols = df.select_dtypes(include=[np.number]).columns
if len(numeric_cols):
    desc_stats = df[numeric_cols].describe().T
    desc_stats["median"] = df[numeric_cols].median()
    desc_stats["sk"] = df[numeric_cols].skew()
    desc_stats["ku"] = df[numeric_cols].kurt()
    desc_stats.to_csv("res/descriptive_statistics.csv")

# -----------------------------
# 2) Spearman correlations (no plots)
# -----------------------------
all_cols = [c for c in (traits_cols + dark_cols + harass_cols) if c in df.columns]

if len(all_cols) >= 2:
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

    # Upper triangle only
    corr_upper = corr_with_stars.copy()
    for i, r in enumerate(all_cols):
        for j, c in enumerate(all_cols):
            if j < i:
                corr_upper.loc[r, c] = ""
    corr_upper.to_csv("res/spearman_corr_table.csv")
    corr_matrix.to_csv("res/spearman_rho.csv")
    pval_matrix.to_csv("res/spearman_p.csv")

# -----------------------------
# Helpers
# -----------------------------
def add_constant(X):
    return sm.add_constant(X, has_constant="add")

def prepare_area_dummies(df_src):
    if "area" not in df_src.columns:
        return df_src.copy(), []
    df_tmp = df_src.copy()
    counts = df_tmp["area"].astype(str).value_counts(dropna=True)
    keep = counts.index.tolist()[:2]
    df_tmp["area_simplified"] = df_tmp["area"].astype(str).apply(lambda x: x if x in keep else "Other")
    df_dum = pd.get_dummies(df_tmp, columns=["area_simplified"], drop_first=True, dtype=float)
    area_dummy_cols = [c for c in df_dum.columns if c.startswith("area_simplified_")]
    return df_dum, area_dummy_cols

def run_ols(y, X, robust=True):
    return sm.OLS(y, X).fit(cov_type="HC3" if robust else "nonrobust")

def cooks_distance_flags(model, n):
    infl = model.get_influence()
    cooks = infl.cooks_distance[0]
    threshold = 4 / n
    flags = cooks > threshold
    return cooks, flags, threshold

def vif_table(X, dep_name):
    X_v = add_constant(X.astype(float))
    return pd.DataFrame({
        "Dependent_Var": dep_name,
        "Variable": ["const"] + list(X.columns),
        "VIF": [np.nan] + [variance_inflation_factor(X_v.values, i) for i in range(1, X_v.shape[1])]
    })

# -----------------------------
# 3) Regression models
# -----------------------------
traits_cols_present = [c for c in traits_cols if c in df.columns]
dark_cols_present   = [c for c in dark_cols if c in df.columns]

def analyze_harassment_models(df0, target):
    df1, area_cols = prepare_area_dummies(df0)
    control_vars = [v for v in ["age","gender"] if v in df1.columns] + area_cols

    X_DT = [c for c in dark_cols_present if c in df1.columns]
    X_HX = [c for c in traits_cols_present if c in df1.columns]
    if (target not in df1.columns) or (len(X_DT)==0) or (len(X_HX)==0):
        return None

    use_cols = [target] + control_vars + X_DT + X_HX
    use = df1[use_cols].apply(pd.to_numeric, errors="coerce").dropna().copy()
    if len(use) < 20:
        return None

    n = len(use)

    Z = use.copy()
    for c in [target] + X_DT + X_HX + [v for v in ["age"] if v in Z.columns]:
        Z[c] = (Z[c] - Z[c].mean()) / Z[c].std(ddof=0)

    XA_cols = control_vars + X_DT
    XB_cols = control_vars + X_DT + X_HX

    y = Z[target].astype(float)
    XA = add_constant(Z[XA_cols].astype(float))
    XB = add_constant(Z[XB_cols].astype(float))

    mA = run_ols(y, XA, robust=True)
    mB = run_ols(y, XB, robust=True)

    delta_R2 = mB.rsquared - mA.rsquared

    # --- Power for incremental R^2 (HEXACO added beyond controls+DT) ---
    u = len(X_HX)  # HEXACOの追加本数（通常6）
    k_full = len(XB_cols)  # フルモデルの予測子数（定数除く）
    v = n - k_full - 1

    # Cohen's f^2 for incremental set:
    # f2 = ΔR^2 / (1 - R^2_full)
    f2_inc = delta_R2 / (1 - mB.rsquared)

    # noncentrality parameter
    lam = f2_inc * (u + v + 1)

    # critical F
    fcrit = st.f.isf(0.05, u, v)

    # achieved power
    power_inc = st.ncf.sf(fcrit, u, v, lam)

    # Nested F (non-robust reference)
    mA_nr = sm.OLS(y, XA).fit()
    mB_nr = sm.OLS(y, XB).fit()
    try:
        an = anova_lm(mA_nr, mB_nr)
        f_change = float(an["F"][1])
        p_change = float(an["Pr(>F)"][1])
    except Exception:
        f_change, p_change = np.nan, np.nan

    # Diagnostics for model B
    dw = durbin_watson(mB.resid)
    try:
        shapiro_p = shapiro(mB.resid)[1] if n <= 5000 else np.nan
    except Exception:
        shapiro_p = np.nan
    try:
        bp_lm, bp_lm_p, bp_f, bp_f_p = het_breuschpagan(mB.resid, mB.model.exog)
    except Exception:
        bp_lm, bp_lm_p, bp_f, bp_f_p = [np.nan]*4

    cooks, flags, th = cooks_distance_flags(mB, n)

    def coef_table(model, dep):
        dfc = pd.DataFrame({
            "Dependent_Var": dep,
            "Variable": model.params.index,
            "Std_Beta": model.params.values,
            "Std_Error_Robust": model.bse.values,
            "t_value_Robust": model.tvalues.values,
            "p_value_Robust": model.pvalues.values
        })
        dfc = dfc[dfc["Variable"]!="const"].copy()
        dfc["Significance"] = dfc["p_value_Robust"].apply(lambda p: "***" if p<0.001 else "**" if p<0.01 else "*" if p<0.05 else "")
        return dfc

    coefA = coef_table(mA, target); coefA["Model"] = "A_controls+DT"
    coefB = coef_table(mB, target); coefB["Model"] = "B_controls+DT+HEXACO"
    coef = pd.concat([coefA, coefB], axis=0).reset_index(drop=True)

    vif_B = vif_table(use[XB_cols], target)

    # Sensitivity: drop high Cook's
    use_sens = use.loc[~flags, :].copy()
    if len(use_sens) >= 20:
        Z2 = use_sens.copy()
        for c in [target] + X_DT + X_HX + [v for v in ["age"] if v in Z2.columns]:
            Z2[c] = (Z2[c] - Z2[c].mean()) / Z2[c].std(ddof=0)
        y2 = Z2[target].astype(float)
        XB2 = add_constant(Z2[XB_cols].astype(float))
        mB_sens = run_ols(y2, XB2, robust=True)
        sens_R2 = mB_sens.rsquared
        coef_sens = coef_table(mB_sens, target); coef_sens["Model"] = "B_sensitivity_(drop_high_Cooks)"
    else:
        sens_R2 = np.nan
        coef_sens = pd.DataFrame(columns=coef.columns)

    # Interactions: DT x H-H
    if "hexaco_HH" in Z.columns:
        inter_cols = []
        for dt in X_DT:
            Z[f"{dt}_x_HH"] = Z[dt] * Z["hexaco_HH"]
            inter_cols.append(f"{dt}_x_HH")
        Xint_cols = XB_cols + inter_cols
        Xint = add_constant(Z[Xint_cols].astype(float))
        mInt = run_ols(y, Xint, robust=True)
        coef_int = coef_table(mInt, target); coef_int["Model"] = "C_controls+DT+HEXACO+DTxHH"
        R2_int = mInt.rsquared
    else:
        coef_int = pd.DataFrame(columns=coef.columns)
        R2_int = np.nan

    # Sex-stratified Model B
    sex_tables, R2_sex = [], []
    if "gender" in use.columns:
        sexes = sorted(use["gender"].dropna().unique().tolist())
        for g in sexes:
            sub = use[use["gender"]==g].copy()
            if len(sub) >= 30:
                Zg = sub.copy()
                for c in [target] + X_DT + X_HX + [v for v in ["age"] if v in Zg.columns]:
                    Zg[c] = (Zg[c] - Zg[c].mean()) / Zg[c].std(ddof=0)
                yg = Zg[target].astype(float)
                XBg = add_constant(Zg[XB_cols].astype(float))
                mg = run_ols(yg, XBg, robust=True)
                cg = coef_table(mg, f"{target}_gender={g}"); cg["Model"] = "B_by_gender"
                sex_tables.append(cg)
                R2_sex.append({"Dependent_Var": target, "gender": g, "n": len(sub), "R2": mg.rsquared, "Adj_R2": mg.rsquared_adj})
    sex_coef = pd.concat(sex_tables, axis=0).reset_index(drop=True) if len(sex_tables) else pd.DataFrame(columns=coef.columns)
    sex_r2 = pd.DataFrame(R2_sex) if len(R2_sex) else pd.DataFrame(columns=["Dependent_Var","gender","n","R2","Adj_R2"])

    model_fit = pd.DataFrame([{
        "Dependent_Var": target,
        "n": n,
        "R2_A": mA.rsquared, "Adj_R2_A": mA.rsquared_adj,
        "R2_B": mB.rsquared, "Adj_R2_B": mB.rsquared_adj,
        "Delta_R2_B_minus_A": delta_R2,
        "Fchange_nonrobust": f_change, "pchange_nonrobust": p_change,
        "Durbin_Watson_B": dw,
        "Shapiro_p_B": shapiro_p,
        "BP_LM_p_B": bp_lm_p,
        "BP_F_p_B": bp_f_p,
        "Max_CooksD_B": np.nanmax(cooks) if len(cooks)>0 else np.nan,
        "Num_CooksD_gt_4n_B": int(flags.sum()),
        "R2_B_sensitivity": sens_R2,
        "R2_with_interactions": R2_int,
        "Power_inc_HEXACO": power_inc,
        "f2_inc_HEXACO": f2_inc
    }])

    return {
        "model_fit": model_fit,
        "coef": pd.concat([coef, coef_sens, coef_int, sex_coef], axis=0, ignore_index=True),
        "vif_B": vif_B,
        "sex_r2": sex_r2
    }

all_model_fit, all_coefs, all_vif, all_sex_r2 = [], [], [], []
for target in harass_cols:
    out = analyze_harassment_models(df, target)
    if out is not None:
        all_model_fit.append(out["model_fit"])
        all_coefs.append(out["coef"])
        all_vif.append(out["vif_B"])
        all_sex_r2.append(out["sex_r2"])

if len(all_model_fit):
    pd.concat(all_model_fit, axis=0).reset_index(drop=True).to_csv("res/model_fit_incremental.csv", index=False)
if len(all_coefs):
    pd.concat(all_coefs, axis=0).reset_index(drop=True).to_csv("res/regression_coefficients_extended.csv", index=False)
if len(all_vif):
    pd.concat(all_vif, axis=0).reset_index(drop=True).to_csv("res/vif_modelB.csv", index=False)
if len(all_sex_r2):
    pd.concat(all_sex_r2, axis=0).reset_index(drop=True).to_csv("res/sex_stratified_R2.csv", index=False)

print("Core analysis complete. Files saved under ./res")
