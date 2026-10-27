"""
Random-effects meta-analysis for Big Five × online learning × achievement.

Implementation follows the pre-registered protocol:
- Fisher's z transformation of Pearson r
- REML estimator for tau² (iterative)
- Hartung-Knapp-Sidik-Jonkman (HKSJ) adjustment for CI
- Heterogeneity: Q, I², tau², tau
- 95% prediction interval

Input: data_extraction_populated.csv
Output: pooling_results.csv + pooling_summary.md

Notes:
- Pure NumPy/SciPy implementation (no R/metafor available in this environment)
- tau² is estimated by reml_tau2_exact(), which matches PyMARE's REML
  estimator. reml_tau2() is the earlier fixed-point update, which
  under-estimates tau² (found 2026-10); it is kept only for scripts that
  reproduce already-submitted numbers (pass exact_reml=False).
- Moderator tests use a Knapp-Hartung mixed-effects meta-regression
  (metareg_knapp_hartung / metareg_knapp_hartung_groups). Until 2026-10,
  pool_by_subgroup() fed HKSJ subgroup SEs into a chi-square test, which
  overstated the evidence (e.g. E x Region Q = 46.43).
- Peterson-Brown β→r conversion applied to studies with only β reported
"""
import csv
from math import log, exp, sqrt, atanh, tanh
from pathlib import Path

import numpy as np
from scipy import optimize, stats

INPUT_CSV = Path("/home/user/paper/metaanalysis/analysis/data_extraction_populated.csv")
RESULTS_CSV = Path("/home/user/paper/metaanalysis/analysis/pooling_results.csv")
SUMMARY_MD = Path("/home/user/paper/metaanalysis/analysis/pooling_summary.md")
MODERATOR_CSV = Path("/home/user/paper/metaanalysis/analysis/moderator_results.csv")

TRAITS = ["O", "C", "E", "A", "N"]


# -----------------------------------------------------------------------------
# Helpers
# -----------------------------------------------------------------------------
def fisher_z(r):
    return 0.5 * log((1 + r) / (1 - r))


def back_transform_z(z):
    return (exp(2 * z) - 1) / (exp(2 * z) + 1)


def var_z(n):
    return 1.0 / (n - 3)


def beta_to_r(beta):
    """Peterson & Brown (2005) approximation (applies only when source model
    has ≤ 2 predictors; caller must verify)."""
    if beta >= 0:
        return beta + 0.05 * 1
    else:
        return beta - 0.05 * 0  # λ=0 if β<0 per formula; (sign absorbed)


# -----------------------------------------------------------------------------
# REML estimator for tau²
# -----------------------------------------------------------------------------
def reml_tau2(y, v, max_iter=200, tol=1e-8):
    """REML iterative estimation of between-study variance.

    NOTE: this fixed-point update under-estimates the REML tau² (by about
    15-20 % on the parent pool). Kept unchanged to reproduce published
    numbers; use reml_tau2_exact() for new analyses.

    y: effect sizes (Fisher z)
    v: sampling variances (1/(N-3))
    Returns: tau² estimate
    """
    y = np.asarray(y, dtype=float)
    v = np.asarray(v, dtype=float)
    k = len(y)
    if k < 2:
        return 0.0

    tau2 = max(0.0, np.var(y, ddof=1) - np.mean(v))  # Starting value

    for _ in range(max_iter):
        w = 1.0 / (v + tau2)
        W = np.sum(w)
        y_bar = np.sum(w * y) / W
        num = np.sum(w**2 * ((y - y_bar) ** 2 - v)) + np.sum(w) / W * np.sum(w * v) / W  # REML expectation
        # Standard REML update (Viechtbauer 2005 eq. 10):
        # tau2_new = sum(w^2 * ((y-ybar)^2 - v)) / (sum(w^2) - sum(w^2)/sum(w))
        # Using the simplified form from Raudenbush (2009):
        denom = np.sum(w**2) - (np.sum(w**2) / W)
        if denom <= 0:
            break
        residual = np.sum(w**2 * ((y - y_bar) ** 2 - v))
        tau2_new = residual / denom
        if tau2_new < 0:
            tau2_new = 0.0
        if abs(tau2_new - tau2) < tol:
            tau2 = tau2_new
            break
        tau2 = tau2_new
    return tau2


def reml_tau2_exact(y, v, X=None):
    """REML tau² by direct maximisation of the restricted log-likelihood.

    X is the moderator design matrix (intercept-only when None). Matches
    PyMARE's REML estimator.
    """
    y = np.asarray(y, dtype=float)
    v = np.asarray(v, dtype=float)
    X = np.ones((len(y), 1)) if X is None else np.asarray(X, dtype=float)
    if len(y) <= X.shape[1]:
        return 0.0

    def neg_restricted_loglik(tau2):
        w = 1.0 / (v + tau2)
        XtWX = (X.T * w) @ X
        beta = np.linalg.solve(XtWX, (X.T * w) @ y)
        resid = y - X @ beta
        return 0.5 * (np.sum(np.log(v + tau2))
                      + np.linalg.slogdet(XtWX)[1]
                      + np.sum(w * resid**2))

    upper = max(10.0 * float(np.var(y)), 1.0)
    res = optimize.minimize_scalar(neg_restricted_loglik, bounds=(0.0, upper),
                                   method="bounded", options={"xatol": 1e-12})
    if neg_restricted_loglik(0.0) <= res.fun:
        return 0.0
    return float(res.x)


def metareg_knapp_hartung(y, v, g):
    """Mixed-effects meta-regression y ~ 1 + g (g = 0/1 dummy) with REML
    residual tau² and the Knapp-Hartung adjustment (untruncated, as in
    metafor's test = "knha"). Use this for two-level moderator tests.
    """
    y = np.asarray(y, dtype=float)
    v = np.asarray(v, dtype=float)
    X = np.column_stack([np.ones_like(y), np.asarray(g, dtype=float)])
    k, p = X.shape
    tau2 = reml_tau2_exact(y, v, X)
    w = 1.0 / (v + tau2)
    XtWX_inv = np.linalg.inv((X.T * w) @ X)
    beta = XtWX_inv @ ((X.T * w) @ y)
    s2 = float(np.sum(w * (y - X @ beta) ** 2) / (k - p))
    se = float(np.sqrt(s2 * XtWX_inv[1, 1]))
    t_val = float(beta[1] / se)
    df = k - p
    t_crit = stats.t.ppf(0.975, df=df)
    return {"k": k, "tau2_resid": tau2, "diff_z": float(beta[1]), "se": se,
            "ci_lo": float(beta[1] - t_crit * se),
            "ci_hi": float(beta[1] + t_crit * se),
            "F": t_val**2, "df1": 1, "df2": df,
            "p": float(2.0 * (1.0 - stats.t.cdf(abs(t_val), df=df)))}


def metareg_knapp_hartung_groups(y, v, groups):
    """Omnibus moderator test for a categorical moderator with any number of
    levels: mixed-effects meta-regression on level dummies (first level in
    order of appearance = reference), REML residual tau², Knapp-Hartung F
    with df = (levels - 1, k - levels). Equals metareg_knapp_hartung() for
    two levels.
    """
    y = np.asarray(y, dtype=float)
    v = np.asarray(v, dtype=float)
    levels = list(dict.fromkeys(groups))
    X = np.column_stack([np.ones_like(y)] +
                        [np.array([1.0 if g == lv else 0.0 for g in groups])
                         for lv in levels[1:]])
    k, p = X.shape
    m = p - 1
    tau2 = reml_tau2_exact(y, v, X)
    w = 1.0 / (v + tau2)
    XtWX_inv = np.linalg.inv((X.T * w) @ X)
    beta = XtWX_inv @ ((X.T * w) @ y)
    s2 = float(np.sum(w * (y - X @ beta) ** 2) / (k - p))
    b = beta[1:]
    cov_b = s2 * XtWX_inv[1:, 1:]
    F = float(b @ np.linalg.solve(cov_b, b) / m)
    return {"k": k, "levels": levels, "tau2_resid": tau2, "F": F,
            "df1": m, "df2": k - p,
            "p": float(1.0 - stats.f.cdf(F, m, k - p))}


# -----------------------------------------------------------------------------
# Pooling
# -----------------------------------------------------------------------------
def dl_tau2(y, v):
    """DerSimonian-Laird moment estimator of tau² (sensitivity analysis)."""
    y = np.asarray(y, dtype=float)
    v = np.asarray(v, dtype=float)
    w = 1.0 / v
    y_bar = np.sum(w * y) / np.sum(w)
    q = np.sum(w * (y - y_bar) ** 2)
    c = np.sum(w) - np.sum(w**2) / np.sum(w)
    return max(0.0, float((q - (len(y) - 1)) / c))


def holm_adjust(pvals):
    """Holm (1979) step-down adjustment; returns adjusted p in input order."""
    p = np.asarray(pvals, dtype=float)
    m = len(p)
    order = np.argsort(p)
    adj = np.empty(m)
    running = 0.0
    for rank, idx in enumerate(order):
        running = max(running, min(1.0, (m - rank) * p[idx]))
        adj[idx] = running
    return adj


def pool_random_effects(y, v, labels=None, exact_reml=True, estimator="REML"):
    """Random-effects meta-analysis with REML + HKSJ CI.

    exact_reml=False selects the earlier, biased tau² estimator and is only
    for scripts that reproduce already-submitted numbers. estimator="DL"
    swaps in the DerSimonian-Laird tau² for the sensitivity analysis.

    Returns dict with pooled estimate on z scale, back-transformed r, CI, PI,
    heterogeneity stats.
    """
    y = np.asarray(y, dtype=float)
    v = np.asarray(v, dtype=float)
    k = len(y)
    if k == 0:
        return None

    if estimator == "DL":
        tau2 = dl_tau2(y, v)
    else:
        tau2 = reml_tau2_exact(y, v) if exact_reml else reml_tau2(y, v)
    w = 1.0 / (v + tau2)
    W = np.sum(w)
    y_bar = np.sum(w * y) / W  # pooled z

    # HKSJ variance (Hartung-Knapp-Sidik-Jonkman)
    # SE_HKSJ = sqrt( q / (k*(k-1)) * sum(w*(y-y_bar)^2) / (sum(w)) )  with q factor
    # Simpler form: variance = (1/(k-1)) * sum(w_i (y_i - y_bar)^2) / sum(w_i)
    if k >= 2:
        se_hksj = sqrt((1.0 / (k - 1)) * np.sum(w * (y - y_bar) ** 2) / W)
    else:
        se_hksj = sqrt(1.0 / W)

    # t-distribution critical value for HKSJ (df = k-1)
    if k >= 2:
        t_crit = stats.t.ppf(0.975, df=k - 1)
    else:
        t_crit = 1.96
    ci_lo = y_bar - t_crit * se_hksj
    ci_hi = y_bar + t_crit * se_hksj

    # Cochran's Q, I², tau
    # Q uses fixed-effect weights (1/v), not random-effect weights
    w_fe = 1.0 / v
    y_bar_fe = np.sum(w_fe * y) / np.sum(w_fe)
    Q = np.sum(w_fe * (y - y_bar_fe) ** 2)
    df = k - 1
    I2 = max(0.0, (Q - df) / Q * 100) if Q > 0 and df > 0 else 0.0
    tau = sqrt(tau2)
    p_Q = 1 - stats.chi2.cdf(Q, df) if df > 0 else 1.0

    # 95% Prediction interval (Higgins et al. 2009)
    # PI = y_bar ± t_{k-2} * sqrt(se_pooled² + tau²)
    if k >= 3:
        t_crit_pi = stats.t.ppf(0.975, df=k - 2)
        var_pooled = 1.0 / W
        pi_se = sqrt(var_pooled + tau2)
        pi_lo = y_bar - t_crit_pi * pi_se
        pi_hi = y_bar + t_crit_pi * pi_se
    else:
        pi_lo = pi_hi = float("nan")

    # Back-transform to r
    r_pooled = back_transform_z(y_bar)
    r_ci_lo = back_transform_z(ci_lo)
    r_ci_hi = back_transform_z(ci_hi)
    r_pi_lo = back_transform_z(pi_lo) if not np.isnan(pi_lo) else float("nan")
    r_pi_hi = back_transform_z(pi_hi) if not np.isnan(pi_hi) else float("nan")

    return {
        "k": k,
        "N_total": None,  # filled by caller
        "z_pooled": y_bar,
        "se_hksj": se_hksj,
        "r_pooled": r_pooled,
        "r_ci_lo": r_ci_lo, "r_ci_hi": r_ci_hi,
        "r_pi_lo": r_pi_lo, "r_pi_hi": r_pi_hi,
        "tau2": tau2, "tau": tau,
        "Q": Q, "df": df, "p_Q": p_Q, "I2": I2,
        "labels": labels,
        "y": y.tolist(), "v": v.tolist(),
    }


# -----------------------------------------------------------------------------
# Data loading + pooling
# -----------------------------------------------------------------------------
def load_extractions():
    """Read populated CSV; return list of dict rows."""
    with INPUT_CSV.open("r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        return list(reader)


def peterson_brown_r(beta, legacy=False):
    """Peterson & Brown (2005) imputation: r = .98β + .05λ, λ = 1 if β ≥ 0
    else 0.

    legacy=True reproduces the earlier, incorrect rule r = β ± .05, which
    pushed negative β a further .05 away from zero (found 2026-10); only for
    scripts that reproduce already-submitted numbers.
    """
    if legacy:
        r = beta + 0.05 if beta >= 0 else beta - 0.05
    else:
        r = 0.98 * beta + (0.05 if beta >= 0 else 0.0)
    return max(-0.99, min(0.99, r))


# Studies kept in the scoping map but left out of the quantitative pool
# (decision of 2026-10-27). Scripts that reproduce numbers submitted before
# that decision pass exclude=() to keep the original pool.
POOL_EXCLUSIONS = {
    "A-37": ("pooled graduate cohorts taught face to face before and after "
             "the pandemic and online during it; correlations reported only "
             "as |r| < .10"),
}


def extract_effect_for_trait(row, trait, legacy_conversion=False,
                             exclude=POOL_EXCLUSIONS):
    """For a given row and trait (O/C/E/A/N), return (r, N, source) or None.

    Applies Peterson-Brown conversion to β if r not available.
    Only returns values for studies flagged as primary_achievement=yes
    or include_with_caveat, and not listed in `exclude`.
    """
    sid = row.get("study_id", "")
    if sid in exclude:
        return None
    inc = row.get("inclusion_status", "")
    prim = row.get("primary_achievement", "")

    # Only studies contributing to primary achievement pool
    eligible = inc in (
        "include", "include_COI", "include_with_caveat",
    ) and prim in ("yes", "partial")
    if not eligible:
        return None

    # Try r first
    r_str = row.get(f"r_{trait}_outcome", "").strip()
    n_str = row.get("n_for_correlations", "").strip() or row.get("n_analyzed", "").strip()
    if r_str and n_str:
        try:
            r = float(r_str)
            n = int(float(n_str))
            if abs(r) <= 0.99 and n > 3:
                return (r, n, "r")
        except ValueError:
            pass

    # Fall back to β with Peterson-Brown
    b_str = row.get(f"beta_{trait}", "").strip()
    if b_str and n_str:
        try:
            b = float(b_str)
            n = int(float(n_str))
            if abs(b) <= 0.99 and n > 3:
                return (peterson_brown_r(b, legacy=legacy_conversion), n,
                        "beta_converted")
        except ValueError:
            pass

    return None


# -----------------------------------------------------------------------------
# Publication bias tests
# -----------------------------------------------------------------------------
def egger_test(y, v):
    """Egger's regression test for funnel-plot asymmetry.

    Regress standardized effect size (y/sqrt(v)) on precision (1/sqrt(v)).
    The intercept tests for small-study effects (publication bias).

    Returns: (intercept, intercept_se, t, p, k)
    """
    y = np.asarray(y, dtype=float)
    v = np.asarray(v, dtype=float)
    k = len(y)
    if k < 3:
        return None

    se = np.sqrt(v)
    precision = 1.0 / se
    std_effect = y / se  # standardized normal deviate

    # Fit: std_effect = intercept + slope * precision + e
    # Weighted by 1/se² is equivalent to unweighted regression of std_effect on precision
    X = np.column_stack([np.ones(k), precision])
    beta, *_ = np.linalg.lstsq(X, std_effect, rcond=None)
    intercept, slope = beta

    # Residuals and SE of intercept
    y_hat = X @ beta
    resid = std_effect - y_hat
    df = k - 2
    if df <= 0:
        return None
    sigma2 = np.sum(resid**2) / df
    XtX_inv = np.linalg.inv(X.T @ X)
    se_intercept = np.sqrt(sigma2 * XtX_inv[0, 0])
    t = intercept / se_intercept
    p = 2 * (1 - stats.t.cdf(abs(t), df=df))

    return {
        "intercept": intercept, "se_intercept": se_intercept,
        "t": t, "p": p, "df": df, "k": k,
    }


def trim_and_fill(y, v, max_iter=100):
    """Duval & Tweedie (2000) trim-and-fill with the L0 estimator, following
    metafor::trimfill: the side of the missing studies comes from the sign
    of the slope of a REML meta-regression of y on sqrt(v); the centre is
    re-estimated on the trimmed set each iteration, while the signed ranks
    are always taken over all k studies.

    (Until 2026-10 this function used argsort positions as ranks and
    re-ranked only the already-trimmed studies, which changed k0.)

    Returns dict with side, k_imputed, r_orig, r_adj and its CI.
    """
    y = np.asarray(y, dtype=float)
    v = np.asarray(v, dtype=float)
    k = len(y)
    if k < 3:
        return None

    X = np.column_stack([np.ones(k), np.sqrt(v)])
    w = 1.0 / (v + reml_tau2_exact(y, v, X))
    slope = np.linalg.solve((X.T * w) @ X, (X.T * w) @ y)[1]
    side = "right" if slope < 0 else "left"

    # Work with the missing studies on the left: flip, then sort ascending.
    yy = -y if side == "right" else y.copy()
    order = np.argsort(yy, kind="stable")
    yy, vv = yy[order], v[order]

    def centre(ys, vs):
        wt = 1.0 / (vs + reml_tau2_exact(ys, vs))
        return float(np.sum(wt * ys) / np.sum(wt))

    k0 = 0
    for _ in range(max_iter):
        k0_prev = k0
        mu = centre(yy[:k - k0], vv[:k - k0])
        dev = yy - mu
        ranks = stats.rankdata(np.abs(dev), method="ordinal")
        s_r = np.sum(ranks[dev > 0])
        k0 = max(0, int(round((4 * s_r - k * (k + 1)) / (2 * k - 1))))
        if k0 == k0_prev:
            break

    if k0 > 0:
        fill = 2 * mu - yy[k - k0:]
        if side == "right":
            fill = -fill
        y_adj = np.concatenate([y, fill])
        v_adj = np.concatenate([v, vv[k - k0:]])
    else:
        y_adj, v_adj = y, v

    orig = pool_random_effects(y, v)
    adj = pool_random_effects(y_adj, v_adj)
    return {
        "k_imputed": k0,
        "side": side,
        "r_orig": orig["r_pooled"],
        "r_adj": adj["r_pooled"],
        "r_adj_ci_lo": adj["r_ci_lo"],
        "r_adj_ci_hi": adj["r_ci_hi"],
        "k_total_adj": len(y_adj),
    }


def run_publication_bias():
    """Run Egger's test + trim-and-fill per trait."""
    rows = load_extractions()
    results = {}
    for trait in TRAITS:
        ys, vs = [], []
        for row in rows:
            ext = extract_effect_for_trait(row, trait)
            if ext is None:
                continue
            r, n, source = ext
            try:
                z = fisher_z(r)
            except (ValueError, ZeroDivisionError):
                continue
            ys.append(z)
            vs.append(var_z(n))
        if len(ys) >= 3:
            results[trait] = {
                "egger": egger_test(ys, vs),
                "trim_fill": trim_and_fill(ys, vs),
            }
        else:
            results[trait] = None
    return results


# -----------------------------------------------------------------------------
# Subgroup (moderator) analysis
# -----------------------------------------------------------------------------
def classify_region(raw):
    """Collapse region to 'Asia' vs 'non-Asia' (Europe/NA/Other)."""
    if not raw:
        return None
    r = raw.strip()
    if r == "Asia":
        return "Asia"
    if r in ("Europe", "North_America", "Other"):
        return "non-Asia"
    return None


def classify_era(raw):
    """Collapse era to 3 levels."""
    if not raw:
        return None
    r = raw.strip()
    if r.startswith("pre"):
        return "pre-COVID"
    if r == "COVID":
        return "COVID"
    if r.startswith("post"):
        return "post-COVID"
    if r == "Mixed_3era":
        return "mixed"  # not used in primary subgroup; reported separately
    return None


def classify_outcome_type(raw):
    """Classify achievement outcomes into objective vs self-report.

    Objective: official grade records, MOOC platform composites, test scores.
    Self-report: self-rated performance, achievement self-report.
    (Engagement-like outcomes are excluded from the primary pool altogether.)
    """
    if not raw:
        return None
    r = raw.strip().lower()
    if any(k in r for k in ("gpa", "course_grade", "test_score",
                              "mooc_composite", "procrastination_exam",
                              "test_completion")):
        return "objective"
    if any(k in r for k in ("self_rated", "self_report", "engagement_performance")):
        return "self-report"
    return None


def pool_by_subgroup(rows, trait, classifier, moderator_name):
    """Run random-effects meta-analysis within each subgroup level, and test
    the moderator with a Knapp-Hartung mixed-effects meta-regression over the
    levels that have k >= 2.

    Returns dict: {level: pool_result, ..., "_between": {"F", "df1", "df2",
    "p", "k_subgroups", "tau2_resid"}}
    """
    by_level = {}
    for row in rows:
        ext = extract_effect_for_trait(row, trait)
        if ext is None:
            continue
        r, n, source = ext
        level = classifier(row.get("region", "") if moderator_name == "region"
                           else row.get("era", "") if moderator_name == "era"
                           else row.get("outcome_type", ""))
        if level is None:
            continue
        try:
            z = fisher_z(r)
        except (ValueError, ZeroDivisionError):
            continue
        by_level.setdefault(level, {"y": [], "v": [], "labels": [], "N": []})
        by_level[level]["y"].append(z)
        by_level[level]["v"].append(var_z(n))
        by_level[level]["labels"].append(
            f"{row['study_id']} {row.get('first_author','')} {row.get('year','')}"
        )
        by_level[level]["N"].append(n)

    results = {}
    tested = []  # levels with k >= 2 enter the moderator test
    for level, data in by_level.items():
        if len(data["y"]) < 2:
            results[level] = {"k": len(data["y"]), "note": "k<2"}
            continue
        res = pool_random_effects(data["y"], data["v"], data["labels"])
        res["N_total"] = sum(data["N"])
        res["level"] = level
        results[level] = res
        tested.append(level)

    if len(tested) >= 2:
        y = [z for lv in tested for z in by_level[lv]["y"]]
        v = [s for lv in tested for s in by_level[lv]["v"]]
        groups = [lv for lv in tested for _ in by_level[lv]["y"]]
        test = metareg_knapp_hartung_groups(y, v, groups)
        results["_between"] = {"F": test["F"], "df1": test["df1"],
                               "df2": test["df2"], "p": test["p"],
                               "tau2_resid": test["tau2_resid"],
                               "k_subgroups": len(tested)}
    else:
        results["_between"] = {"F": None, "df1": 0, "df2": 0, "p": None,
                               "k_subgroups": len(tested)}

    return results


def run_moderator_analyses():
    """Run Region/Era/Outcome-type subgroup analyses for each trait."""
    rows = load_extractions()
    moderators = {
        "region": classify_region,
        "era": classify_era,
        "outcome_type": classify_outcome_type,
    }
    all_results = {}
    for mod_name, classifier in moderators.items():
        all_results[mod_name] = {}
        for trait in TRAITS:
            all_results[mod_name][trait] = pool_by_subgroup(
                rows, trait, classifier, mod_name
            )
    # Holm correction within each trait across the moderators tested
    # (the pre-registered family).
    for trait in TRAITS:
        tested = [m for m in moderators
                  if all_results[m][trait]["_between"]["p"] is not None]
        adj = holm_adjust([all_results[m][trait]["_between"]["p"] for m in tested])
        for m, p_holm in zip(tested, adj):
            all_results[m][trait]["_between"]["p_holm"] = float(p_holm)
    return all_results


def write_moderator_csv(mod_results):
    with MODERATOR_CSV.open("w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([
            "moderator", "trait", "level", "k", "N_total",
            "r_pooled", "r_ci_lo", "r_ci_hi",
            "I2", "tau2",
            "F_between", "df1_between", "df2_between", "p_between", "p_holm",
        ])
        for mod_name, by_trait in mod_results.items():
            for trait in TRAITS:
                res = by_trait.get(trait, {})
                between = res.get("_between", {})
                tested = between.get("F") is not None
                for level, sub in res.items():
                    if level == "_between":
                        continue
                    if "r_pooled" in sub:
                        writer.writerow([
                            mod_name, trait, level, sub["k"], sub["N_total"],
                            f"{sub['r_pooled']:.4f}",
                            f"{sub['r_ci_lo']:.4f}", f"{sub['r_ci_hi']:.4f}",
                            f"{sub['I2']:.1f}", f"{sub['tau2']:.4f}",
                            f"{between['F']:.4f}" if tested else "NA",
                            between["df1"] if tested else "NA",
                            between["df2"] if tested else "NA",
                            f"{between['p']:.6f}" if tested else "NA",
                            f"{between['p_holm']:.6f}" if tested else "NA",
                        ])
                    else:
                        writer.writerow([
                            mod_name, trait, level,
                            sub.get("k", 0), "", "", "", "", "", "",
                            "", "", "", "", "",
                        ])


def run_pooling():
    rows = load_extractions()
    results = {}

    for trait in TRAITS:
        ys, vs, labels, Ns = [], [], [], []
        for row in rows:
            ext = extract_effect_for_trait(row, trait)
            if ext is None:
                continue
            r, n, source = ext
            try:
                z = fisher_z(r)
            except (ValueError, ZeroDivisionError):
                continue
            ys.append(z)
            vs.append(var_z(n))
            labels.append(f"{row['study_id']} {row.get('first_author','')} {row.get('year','')} ({source})")
            Ns.append(n)
        if len(ys) >= 2:
            res = pool_random_effects(ys, vs, labels)
            res["N_total"] = sum(Ns)
            res["trait"] = trait
            results[trait] = res
        else:
            results[trait] = {"trait": trait, "k": len(ys), "note": "insufficient k"}

    return results


# -----------------------------------------------------------------------------
# Output writers
# -----------------------------------------------------------------------------
def write_results_csv(results):
    with RESULTS_CSV.open("w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([
            "trait", "k", "N_total",
            "r_pooled", "r_ci_lo", "r_ci_hi",
            "r_pi_lo", "r_pi_hi",
            "tau2", "tau", "I2", "Q", "df", "p_Q",
            "z_pooled", "se_hksj",
        ])
        for t in TRAITS:
            r = results.get(t, {})
            if "r_pooled" in r:
                writer.writerow([
                    t, r["k"], r["N_total"],
                    f"{r['r_pooled']:.4f}", f"{r['r_ci_lo']:.4f}", f"{r['r_ci_hi']:.4f}",
                    f"{r['r_pi_lo']:.4f}" if not np.isnan(r['r_pi_lo']) else "",
                    f"{r['r_pi_hi']:.4f}" if not np.isnan(r['r_pi_hi']) else "",
                    f"{r['tau2']:.6f}", f"{r['tau']:.4f}", f"{r['I2']:.1f}",
                    f"{r['Q']:.3f}", r['df'], f"{r['p_Q']:.4f}",
                    f"{r['z_pooled']:.4f}", f"{r['se_hksj']:.4f}",
                ])
            else:
                writer.writerow([t, r.get("k", 0), "", "", "", "", "", "", "", "", "", "", "", "", "", ""])


def write_summary_md_with_moderators(results, mod_results):
    """Extended summary with moderator findings."""
    lines = [
        "# Meta-Analysis Pooling Results",
        "",
        "**Pre-registered protocol**: Random-effects meta-analysis, REML estimator, HKSJ confidence intervals, Fisher's z transformation.",
        "",
        "**Input**: `data_extraction_populated.csv`",
        "**Output**: `pooling_results.csv`, `moderator_results.csv`",
        "",
        "## Per-trait pooled effects (primary analysis)",
        "",
        "| Trait | k | N_total | r (95% CI) | 95% PI | I² | τ² | Q(df), p |",
        "|-------|---|---------|-----------|--------|-----|-----|----------|",
    ]
    for t in TRAITS:
        r = results.get(t, {})
        if "r_pooled" in r:
            ci = f"{r['r_pooled']:.3f} [{r['r_ci_lo']:.3f}, {r['r_ci_hi']:.3f}]"
            if not np.isnan(r['r_pi_lo']):
                pi = f"[{r['r_pi_lo']:.3f}, {r['r_pi_hi']:.3f}]"
            else:
                pi = "(k<3)"
            lines.append(
                f"| **{t}** | {r['k']} | {r['N_total']} | {ci} | {pi} | "
                f"{r['I2']:.1f}% | {r['tau2']:.4f} | "
                f"{r['Q']:.2f}({r['df']}), p={r['p_Q']:.3f} |"
            )

    lines.append("")
    lines.append("## Moderator analyses (exploratory; k < 10 per level caveats apply)")
    lines.append("")
    lines.append("Three pre-registered moderators are reported; the remaining six (instrument, publication year, sample size, RoB score, modality, education level) are reported narratively in the manuscript due to insufficient k per level.")
    lines.append("")
    lines.append("Moderator test: Knapp–Hartung mixed-effects meta-regression on the levels with k ≥ 2 (REML residual τ²), F(df1, df2). p_Holm: Holm correction within each trait across the three moderators.")
    lines.append("")

    for mod_name, by_trait in mod_results.items():
        lines.append(f"### Moderator: {mod_name}")
        lines.append("")
        lines.append("| Trait | Level | k | N | r [95% CI] | I² | F (df1, df2), p, p_Holm |")
        lines.append("|-------|-------|---|---|-----------|-----|-------------------------|")
        for t in TRAITS:
            res = by_trait.get(t, {})
            between = res.get("_between", {})
            f_val = between.get("F")
            p = between.get("p")
            q_str = (f"{f_val:.2f} ({between['df1']}, {between['df2']}), p={p:.3f}, "
                     f"p_Holm={between['p_holm']:.3f}"
                     if f_val is not None else "NA")
            first = True
            for level, sub in res.items():
                if level == "_between":
                    continue
                if "r_pooled" in sub:
                    ci = f"{sub['r_pooled']:.3f} [{sub['r_ci_lo']:.3f}, {sub['r_ci_hi']:.3f}]"
                    lines.append(
                        f"| {t if first else ''} | {level} | {sub['k']} | "
                        f"{sub['N_total']} | {ci} | {sub['I2']:.1f}% | "
                        f"{q_str if first else ''} |"
                    )
                else:
                    lines.append(
                        f"| {t if first else ''} | {level} | "
                        f"{sub.get('k', 0)} | — | (k<2) | — | "
                        f"{q_str if first else ''} |"
                    )
                first = False
        lines.append("")

    lines.append("## Moderator tests with p < .10")
    lines.append("")
    flagged = []
    for mod_name, by_trait in mod_results.items():
        for t in TRAITS:
            res = by_trait.get(t, {})
            between = res.get("_between", {})
            if between.get("p") is None or between["p"] >= 0.10:
                continue
            levels = "; ".join(
                f"{lv} (k={sub['k']}): r = {sub['r_pooled']:.3f}"
                for lv, sub in res.items()
                if lv != "_between" and "r_pooled" in sub)
            flagged.append(
                f"- **{t} × {mod_name}**: F({between['df1']}, {between['df2']}) = "
                f"{between['F']:.2f}, p = {between['p']:.3f} (Holm {between['p_holm']:.3f}) — {levels}")
    lines.extend(flagged or ["- None."])
    lines.append("")
    lines.append("Every moderator contrast rests on k ≤ 7 per level (k = 2 for Asia and for self-report outcomes); treat all of them as exploratory.")
    lines.append("")
    lines.append("## Contributing studies per trait")
    for t in TRAITS:
        r = results.get(t, {})
        if "labels" in r and r["labels"]:
            lines.append(f"\n### Trait {t} (k = {r['k']})")
            for label in r["labels"]:
                lines.append(f"- {label}")

    lines.append("")
    lines.append("---")
    lines.append("")
    lines.append("## Notes")
    lines.append("- **Power caveat**: With fewer than 10 studies per trait pool and the Asian subgroup limited to k = 2, moderator findings are **underpowered** and should be interpreted as exploratory.")
    lines.append("- **β-to-r conversion**: Peterson & Brown (2005) applied where only β was reported. The exclude-converted sensitivity analysis is in `sensitivity_results.md`.")
    lines.append("- **Sign conventions**: A-23 Rodrigues GPA sign-flipped so positive r = better performance; A-31 Rivers Emotional Stability sign-reversed to Neuroticism.")
    lines.append("- **Remaining 6 pre-registered moderators** (instrument, publication year, sample size, RoB score, modality, education level) not quantitatively analyzed due to insufficient k per level; reported narratively in Methods Deviations subsection.")

    SUMMARY_MD.write_text("\n".join(lines), encoding="utf-8")


def append_publication_bias_to_summary(bias_results):
    """Append publication bias section (Egger + trim-and-fill) to summary."""
    lines = ["", "## Publication Bias Assessment", ""]
    lines.append("### Egger's Regression Test")
    lines.append("")
    lines.append("| Trait | k | Intercept | SE | t | p |")
    lines.append("|-------|---|-----------|-----|---|---|")
    for t in TRAITS:
        r = bias_results.get(t)
        if r is None:
            lines.append(f"| {t} | — | insufficient k | — | — | — |")
        else:
            e = r["egger"]
            sig = " 🔴" if e["p"] < .05 else ""
            lines.append(
                f"| {t} | {e['k']} | {e['intercept']:.3f} | "
                f"{e['se_intercept']:.3f} | {e['t']:.2f} | {e['p']:.3f}{sig} |"
            )
    lines.append("")
    lines.append("### Duval & Tweedie (2000) Trim-and-Fill")
    lines.append("")
    lines.append("| Trait | k_orig | k_imputed | Side | r_original | r_adjusted [95% CI] |")
    lines.append("|-------|--------|-----------|------|-----------|---------------------|")
    for t in TRAITS:
        r = bias_results.get(t)
        if r is None or r["trim_fill"] is None:
            lines.append(f"| {t} | — | — | — | — | — |")
        else:
            tf = r["trim_fill"]
            k_orig = tf["k_total_adj"] - tf["k_imputed"]
            if tf["r_adj"] is not None and tf["r_adj_ci_lo"] is not None:
                ci = f"{tf['r_adj']:.3f} [{tf['r_adj_ci_lo']:.3f}, {tf['r_adj_ci_hi']:.3f}]"
            else:
                ci = "—"
            lines.append(
                f"| {t} | {k_orig} | {tf['k_imputed']} | {tf['side']} | "
                f"{tf['r_orig']:.3f} | {ci} |"
            )
    lines.append("")
    lines.append("**Interpretation**:")
    lines.append("- Egger's test: non-zero intercept suggests small-study effects or publication bias. With fewer than 10 studies per trait, the test has low power; p ≥ .05 does not rule out bias.")
    lines.append("- Trim-and-fill (L0 estimator, side from the REML regression of y on its standard error, as in metafor::trimfill): imputes hypothetical missing studies on the under-represented side of the funnel. Reported as a *sensitivity check* only; adjusted estimates should NOT replace primary pooled estimates.")
    for t in TRAITS:
        r = bias_results.get(t)
        if r is not None and r["egger"]["p"] < .05:
            lines.append(f"- {t}: Egger p = {r['egger']['p']:.3f}; trim-and-fill imputes "
                         f"{r['trim_fill']['k_imputed']} on the {r['trim_fill']['side']}.")
    lines.append("")
    with SUMMARY_MD.open("a", encoding="utf-8") as f:
        f.write("\n".join(lines))


def write_summary_md(results):
    lines = [
        "# Meta-Analysis Pooling Results",
        "",
        "**Pre-registered protocol**: Random-effects meta-analysis, REML estimator, HKSJ confidence intervals, Fisher's z transformation.",
        "",
        "**Input**: `data_extraction_populated.csv`",
        "**Output**: `pooling_results.csv`",
        "",
        "## Per-trait pooled effects",
        "",
        "| Trait | k | N_total | r (95% CI) | 95% PI | I² | τ² | Q(df), p |",
        "|-------|---|---------|-----------|--------|-----|-----|----------|",
    ]
    for t in TRAITS:
        r = results.get(t, {})
        if "r_pooled" in r:
            ci = f"{r['r_pooled']:.3f} [{r['r_ci_lo']:.3f}, {r['r_ci_hi']:.3f}]"
            if not np.isnan(r['r_pi_lo']):
                pi = f"[{r['r_pi_lo']:.3f}, {r['r_pi_hi']:.3f}]"
            else:
                pi = "(k<3)"
            lines.append(
                f"| **{t}** | {r['k']} | {r['N_total']} | {ci} | {pi} | "
                f"{r['I2']:.1f}% | {r['tau2']:.4f} | "
                f"{r['Q']:.2f}({r['df']}), p={r['p_Q']:.3f} |"
            )
        else:
            lines.append(f"| {t} | {r.get('k', 0)} | — | insufficient k | — | — | — | — |")

    lines.append("")
    lines.append("## Contributing studies per trait")
    for t in TRAITS:
        r = results.get(t, {})
        if "labels" in r and r["labels"]:
            lines.append(f"\n### Trait {t} (k = {r['k']})")
            for label in r["labels"]:
                lines.append(f"- {label}")

    lines.append("")
    lines.append("---")
    lines.append("")
    lines.append("**Interpretation notes**:")
    lines.append("- r_pooled is the random-effects pooled Pearson correlation (back-transformed from Fisher's z).")
    lines.append("- 95% CI uses HKSJ adjustment (t-distribution, df = k-1).")
    lines.append("- 95% PI is the range within which a future study's true effect is expected to fall (df = k-2).")
    lines.append("- I² quantifies the proportion of variance due to heterogeneity (vs. sampling error).")
    lines.append("- τ² is the REML-estimated between-study variance.")
    lines.append("- β-only studies converted via Peterson & Brown (2005): r ≈ β + 0.05 if β ≥ 0 else β − 0.05.")

    SUMMARY_MD.write_text("\n".join(lines), encoding="utf-8")


def main():
    results = run_pooling()
    write_results_csv(results)
    write_summary_md(results)

    mod_results = run_moderator_analyses()
    write_moderator_csv(mod_results)

    bias_results = run_publication_bias()

    # Overwrite summary with extended version that includes moderators + bias
    write_summary_md_with_moderators(results, mod_results)
    # Append bias section
    append_publication_bias_to_summary(bias_results)

    print(f"Wrote {RESULTS_CSV}")
    print(f"Wrote {SUMMARY_MD}")
    print(f"Wrote {MODERATOR_CSV}")
    print()
    print("=== Pooled results ===")
    for t in TRAITS:
        r = results.get(t, {})
        if "r_pooled" in r:
            ci_low = r['r_ci_lo']
            ci_hi = r['r_ci_hi']
            print(f"{t}: k={r['k']}, N={r['N_total']}, "
                  f"r={r['r_pooled']:.3f} [{ci_low:.3f}, {ci_hi:.3f}], "
                  f"I²={r['I2']:.1f}%, τ²={r['tau2']:.4f}")
        else:
            print(f"{t}: k={r.get('k', 0)} — insufficient data")

    print()
    print("=== Moderator analyses ===")
    for mod_name, by_trait in mod_results.items():
        print(f"\n[{mod_name}]")
        for t in TRAITS:
            res = by_trait.get(t, {})
            between = res.get("_between", {})
            if between.get("F") is None:
                continue
            parts = []
            for level, sub in res.items():
                if level == "_between" or "r_pooled" not in sub:
                    continue
                parts.append(
                    f"{level}(k={sub['k']}): r={sub['r_pooled']:.3f}"
                )
            print(f"  {t}: " + "; ".join(parts) +
                  f"  | F({between['df1']},{between['df2']})={between['F']:.2f}"
                  f" p={between['p']:.3f}")


if __name__ == "__main__":
    main()
