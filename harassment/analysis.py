import os
import pandas as pd
import numpy as np
from scipy.stats import spearmanr
import statsmodels.api as sm

# Create output folder
os.makedirs("res", exist_ok=True)

# --------------------------------------------
# Load dataset & type conversion
# --------------------------------------------
df = pd.read_csv("raw.csv", na_values=["NA", "N/A", "na", "NaN", "-", ""])
traits_cols = [
    "hexaco_HH","hexaco_E","hexaco_X","hexaco_A","hexaco_C","hexaco_O"  # HEXACO traits only
]
dark_cols = ["Machiavellianism","Narcissism","Psychopathy"]
harass_cols = ["power_harassment","gender_harassment"]

# Convert main columns to numeric
cols_to_numeric = traits_cols + dark_cols + harass_cols + ["age","gender"]
df[cols_to_numeric] = df[cols_to_numeric].apply(pd.to_numeric, errors="coerce")

# --------------------------------------------
# 1) Descriptive statistics (save to CSV)
# --------------------------------------------
numeric_cols = df.select_dtypes(include=[np.number]).columns
desc_stats = df[numeric_cols].describe().T
desc_stats["median"] = df[numeric_cols].median()
desc_stats.to_csv("res/descriptive_statistics.csv")

# --------------------------------------------
# 2) Spearman correlation: star-marked correlation table
# --------------------------------------------
all_cols = traits_cols + dark_cols + harass_cols

# Compute correlation coefficients and p-values
corr_matrix = pd.DataFrame(index=all_cols, columns=all_cols, dtype=float)
pval_matrix = pd.DataFrame(index=all_cols, columns=all_cols, dtype=float)
for c1 in all_cols:
    for c2 in all_cols:
        rho, p = spearmanr(df[c1], df[c2], nan_policy="omit")
        corr_matrix.loc[c1, c2] = rho
        pval_matrix.loc[c1, c2] = p

# Function for significance stars
def _star(p):
    return "***" if p < 0.001 else "**" if p < 0.01 else "*" if p < 0.05 else ""

# Create correlation matrix with stars
corr_with_stars = pd.DataFrame(index=all_cols, columns=all_cols, dtype=object)
for r in all_cols:
    for c in all_cols:
        if r == c:
            corr_with_stars.loc[r, c] = "1.00"
        else:
            rho = corr_matrix.loc[r, c]
            p = pval_matrix.loc[r, c]
            corr_with_stars.loc[r, c] = "" if pd.isna(rho) else f"{rho:.2f}{_star(p)}"

# Blank out lower triangle
corr_upper = corr_with_stars.copy()
for i, r in enumerate(all_cols):
    for j, c in enumerate(all_cols):
        if j < i:
            corr_upper.loc[r, c] = ""

# Save correlation table
corr_upper.to_csv("res/spearman_corr_table.csv")

# --------------------------------------------
# 3) Multiple regression (with control variables) + save significant results
# --------------------------------------------
# Create dummy variables for "area" (drop first as baseline)
df_dummies = pd.get_dummies(df, columns=["area"], drop_first=True, dtype=float)

# Control variables
control_vars = ["age", "gender"] + [c for c in df_dummies.columns if c.startswith("area_")]

# Independent variables = personality traits + controls
X_base = df_dummies[traits_cols + control_vars].astype(float)

targets = dark_cols + harass_cols  # Dependent variables (Dark Triad + harassment)

results_list = []
for target in targets:
    use = pd.concat([df_dummies[[target]], X_base], axis=1)
    use = use.apply(pd.to_numeric, errors="coerce").dropna()

    y = use[target].astype(float)
    X = use.drop(columns=[target]).astype(float)
    X = sm.add_constant(X, has_constant="add")

    model = sm.OLS(y, X).fit()

    out = pd.DataFrame({
        "Variable": model.params.index,
        "Coefficient": model.params.values,
        "Std_Error": model.bse.values,
        "t_value": model.tvalues.values,
        "p_value": model.pvalues.values
    })
    out["Significance"] = out["p_value"].apply(_star)
    out["Dependent_Var"] = target

    # Exclude const & keep only significant results (p < .05)
    out = out[(out["Variable"] != "const") & (out["p_value"] < 0.05)]

    results_list.append(out)

# Combine all regression results (handle empty case)
if len(results_list) > 0 and not all(df.empty for df in results_list):
    sig_results = pd.concat(results_list, axis=0).sort_values(["Dependent_Var", "p_value"])
else:
    sig_results = pd.DataFrame(columns=["Variable","Coefficient","Std_Error","t_value","p_value","Significance","Dependent_Var"])

sig_results.to_csv("res/regression.csv", index=False)
