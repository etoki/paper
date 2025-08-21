# -*- coding: utf-8 -*-
import os
import numpy as np
import pandas as pd
from scipy import stats
import pingouin as pg
from statsmodels.stats.multicomp import pairwise_tukeyhsd

# ========= Settings =========
INPUT_PATH = "raw.csv"
OUT_DIR = "res"
os.makedirs(OUT_DIR, exist_ok=True)

traits = [
    "hexaco_HH","hexaco_E","hexaco_X","hexaco_A","hexaco_C","hexaco_O",
    "Machiavellianism","Narcissism","Psychopathy",
    "power_harassment","gender_harassment"
]

def sigmark(p):
    """Return significance marker according to p-value"""
    if p < 0.001: return "***"
    elif p < 0.01: return "**"
    elif p < 0.05: return "*"
    else: return ""

# ========= Load Data =========
df = pd.read_csv(INPUT_PATH)

# ========= 1) Gender difference (t-test, only significant results) =========
t_rows = []
for var in traits:
    female = df.loc[df["gender"]==1, var].dropna()  # 1 = female
    male   = df.loc[df["gender"]==0, var].dropna()  # 0 = male
    if len(female) >= 2 and len(male) >= 2:
        t_stat, p_val = stats.ttest_ind(female, male, equal_var=False)
        # Cohen's d (female − male)
        nx, ny = len(female), len(male)
        var_x, var_y = female.var(ddof=1), male.var(ddof=1)
        pooled_sd = np.sqrt(((nx-1)*var_x + (ny-1)*var_y) / (nx+ny-2)) if (nx+ny-2)>0 else np.nan
        cohend = (female.mean() - male.mean())/pooled_sd if pooled_sd not in [0, np.nan] else np.nan
        t_rows.append([var, female.mean(), male.mean(), t_stat, p_val, cohend, sigmark(p_val)])

t_df = pd.DataFrame(t_rows, columns=["Variable","Female_mean","Male_mean","t","p","Cohen_d","sig"])
t_sig = t_df.loc[t_df["p"] < 0.05].sort_values("p")
t_sig.to_csv(os.path.join(OUT_DIR, "t_test_sig.csv"), index=False)

# ========= 2) ANOVA by age group (6 groups, only significant results) =========
# Define age bins: 10s, 20s, 30s, 40s, 50s, 60+
df["age_group"] = pd.cut(
    df["age"], bins=[0,19,29,39,49,59,200],
    labels=["10s","20s","30s","40s","50s","60s"]
)

a_rows = []
for var in traits:
    sub = df[["age_group", var]].dropna()
    if sub["age_group"].nunique() < 2: 
        continue
    aov = pg.anova(dv=var, between="age_group", data=sub, detailed=True)
    f_val  = aov.loc[aov.index[0], "F"]
    p_val  = aov.loc[aov.index[0], "p-unc"]
    eta_sq = aov.loc[aov.index[0], "np2"]  # partial eta squared
    a_rows.append([var, f_val, p_val, eta_sq, sigmark(p_val)])

a_df  = pd.DataFrame(a_rows, columns=["Variable","F","p","eta_sq","sig"])
a_sig = a_df.loc[a_df["p"] < 0.05].sort_values("p")
a_sig.to_csv(os.path.join(OUT_DIR, "anova_sig.csv"), index=False)

# ========= 3) Post-hoc Tukey HSD (only for variables significant in ANOVA, only significant pairs) =========
posthoc_list = []
sig_vars = a_sig["Variable"].tolist()

for var in sig_vars:
    sub = df[["age_group", var]].dropna()
    if sub["age_group"].nunique() < 2:
        continue
    tukey = pairwise_tukeyhsd(endog=sub[var], groups=sub["age_group"], alpha=0.05)
    res_df = pd.DataFrame(tukey.summary().data[1:], columns=tukey.summary().data[0])
    res_df["Variable"] = var
    res_df["p-adj"] = res_df["p-adj"].astype(float)
    res_df["sig"]   = res_df["p-adj"].apply(sigmark)
    res_sig = res_df.loc[res_df["reject"]==True].copy()
    posthoc_list.append(res_sig)

if len(posthoc_list) > 0:
    posthoc_sig = pd.concat(posthoc_list, ignore_index=True)
else:
    posthoc_sig = pd.DataFrame(columns=["group1","group2","meandiff","p-adj","lower","upper","reject","Variable","sig"])

posthoc_sig.to_csv(os.path.join(OUT_DIR, "posthoc_sig.csv"), index=False)

# ========= 4) Spearman correlation between age and traits (only significant results) =========
corr_rows = []
for var in traits:
    r, p = stats.spearmanr(df["age"], df[var], nan_policy="omit")
    corr_rows.append([var, r, p, sigmark(p)])

corr_df  = pd.DataFrame(corr_rows, columns=["Variable","Spearman_r","p","sig"])
corr_sig = corr_df.loc[corr_df["p"] < 0.05].sort_values("p")
corr_sig.to_csv(os.path.join(OUT_DIR, "corr_with_age_sig.csv"), index=False)

# ========= Console check (optional) =========
print("\n[ Significant t-test results ]\n", t_sig)
print("\n[ Significant ANOVA results ]\n", a_sig)
print("\n[ Significant Tukey HSD pairs ]\n", posthoc_sig.head())
print("\n[ Significant correlations with age ]\n", corr_sig)
