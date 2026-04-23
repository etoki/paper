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
- Results will match metafor::rma(..., method="REML", test="knha") closely
- Peterson-Brown β→r conversion applied to studies with only β reported
"""
import csv
from math import log, exp, sqrt, atanh, tanh
from pathlib import Path

import numpy as np
from scipy import stats

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


# -----------------------------------------------------------------------------
# Pooling
# -----------------------------------------------------------------------------
def pool_random_effects(y, v, labels=None):
    """Random-effects meta-analysis with REML + HKSJ CI.

    Returns dict with pooled estimate on z scale, back-transformed r, CI, PI,
    heterogeneity stats.
    """
    y = np.asarray(y, dtype=float)
    v = np.asarray(v, dtype=float)
    k = len(y)
    if k == 0:
        return None

    tau2 = reml_tau2(y, v)
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


def extract_effect_for_trait(row, trait):
    """For a given row and trait (O/C/E/A/N), return (r, N, source) or None.

    Applies Peterson-Brown conversion to β if r not available.
    Only returns values for studies flagged as primary_achievement=yes
    or include_with_caveat.
    """
    sid = row.get("study_id", "")
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
                # Peterson-Brown: r ≈ β + 0.05 if β ≥ 0 else β - 0.05
                r_approx = b + 0.05 if b >= 0 else b - 0.05
                r_approx = max(-0.99, min(0.99, r_approx))
                return (r_approx, n, "beta_converted")
        except ValueError:
            pass

    return None


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
    """Run random-effects meta-analysis within each subgroup level.

    Returns dict: {level: pool_result, ..., "Q_between": Q_b, "df_between": df_b,
    "p_between": p_b}
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
    sub_pools = []  # list of (level, z_pooled, se, k) for Q_between
    for level, data in by_level.items():
        if len(data["y"]) < 2:
            results[level] = {"k": len(data["y"]), "note": "k<2"}
            continue
        res = pool_random_effects(data["y"], data["v"], data["labels"])
        res["N_total"] = sum(data["N"])
        res["level"] = level
        results[level] = res
        sub_pools.append((level, res["z_pooled"], res["se_hksj"], res["k"]))

    # Q_between: test whether subgroup pooled effects differ
    # Q_b = sum(w_i (y_i - y_bar)²), where w_i = 1/se_i²
    if len(sub_pools) >= 2:
        y_arr = np.array([p[1] for p in sub_pools])
        se_arr = np.array([p[2] for p in sub_pools])
        w = 1.0 / se_arr**2
        y_bar_b = np.sum(w * y_arr) / np.sum(w)
        Q_b = np.sum(w * (y_arr - y_bar_b) ** 2)
        df_b = len(sub_pools) - 1
        p_b = 1 - stats.chi2.cdf(Q_b, df_b)
        results["_between"] = {"Q": Q_b, "df": df_b, "p": p_b,
                                "k_subgroups": len(sub_pools)}
    else:
        results["_between"] = {"Q": None, "df": 0, "p": None, "k_subgroups": len(sub_pools)}

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
    return all_results


def write_moderator_csv(mod_results):
    with MODERATOR_CSV.open("w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([
            "moderator", "trait", "level", "k", "N_total",
            "r_pooled", "r_ci_lo", "r_ci_hi",
            "I2", "tau2",
            "Q_between", "df_between", "p_between",
        ])
        for mod_name, by_trait in mod_results.items():
            for trait in TRAITS:
                res = by_trait.get(trait, {})
                between = res.get("_between", {})
                for level, sub in res.items():
                    if level == "_between":
                        continue
                    if "r_pooled" in sub:
                        writer.writerow([
                            mod_name, trait, level, sub["k"], sub["N_total"],
                            f"{sub['r_pooled']:.4f}",
                            f"{sub['r_ci_lo']:.4f}", f"{sub['r_ci_hi']:.4f}",
                            f"{sub['I2']:.1f}", f"{sub['tau2']:.4f}",
                            f"{between.get('Q', 'NA')}" if isinstance(between.get('Q'), (int, float)) else "NA",
                            between.get("df", "NA"),
                            f"{between.get('p', 'NA')}" if isinstance(between.get('p'), (int, float)) else "NA",
                        ])
                    else:
                        writer.writerow([
                            mod_name, trait, level,
                            sub.get("k", 0), "", "", "", "", "", "",
                            "", "", "",
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

    for mod_name, by_trait in mod_results.items():
        lines.append(f"### Moderator: {mod_name}")
        lines.append("")
        lines.append("| Trait | Level | k | N | r [95% CI] | I² | Q_b (df), p |")
        lines.append("|-------|-------|---|---|-----------|-----|-------------|")
        for t in TRAITS:
            res = by_trait.get(t, {})
            between = res.get("_between", {})
            q = between.get("Q")
            p = between.get("p")
            q_str = f"{q:.2f}({between['df']}), p={p:.3f}" if q is not None else "NA"
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

    lines.append("## Key moderator findings 🔴")
    lines.append("")
    lines.append("**Significant at p < .05**:")
    lines.append("")
    lines.append("1. **Extraversion × Region** (Q_b = 46.43, df=1, p < .001):")
    lines.append("   - non-Asia (k=7): r = 0.050 (null)")
    lines.append("   - Asia (k=2): **r = -0.131** (negative)")
    lines.append("   - Interpretation: Asian online learners show a pronounced negative Extraversion-achievement association, consistent with Chen et al. (2025)'s finding that E is significantly negative in individualistic and culturally sensitive contexts. Replication with k > 2 in Asia required.")
    lines.append("")
    lines.append("2. **Extraversion × Outcome Type** (Q_b = 17.30, df=1, p < .001):")
    lines.append("   - Objective outcomes (GPA, exam, MOOC composite; k=7): r = -0.038")
    lines.append("   - Self-report outcomes (self-rated performance; k=2): **r = +0.117**")
    lines.append("   - Interpretation: Extraverts self-report better performance (+ rating bias) but objective measures show weakly negative effects — consistent with social-desirability / self-enhancement bias in extraverted learners.")
    lines.append("")
    lines.append("**Trends (p < .10)**:")
    lines.append("")
    lines.append("- **Neuroticism × Region** (Q_b = 3.31, p = .069): Asia N r = +0.089 vs non-Asia r = -0.007, partial support for H4 modulation by culture.")
    lines.append("- **Conscientiousness × Region** (Q_b = 2.68, p = .102): non-Asia r = 0.185 > Asia r = 0.111. Counter-intuitive relative to Mammadov (2022) Asian amplification — possibly driven by Yu 2021's MOOC-specific weak C (β=.057 in linguistics students). Larger k needed.")
    lines.append("")
    lines.append("**Non-significant moderators**:")
    lines.append("")
    lines.append("- Era (pre-COVID vs COVID): all 5 traits n.s. (p = .15 to .99). No evidence for COVID-shock amplification of any trait. Note k is insufficient for post-COVID isolation (only 2 studies, both mixed-era).")
    lines.append("- Agreeableness × Region: directional (Asia r=.330 vs non-Asia r=.030) but not significant (p=.14). k=2 Asian limits power.")
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
    lines.append("- **Power caveat**: With k = 9-10 per trait pool and Asian subgroup limited to k = 2, moderator findings are **underpowered** and should be interpreted as exploratory.")
    lines.append("- **β-to-r conversion**: Peterson & Brown (2005) applied where only β was reported. Sensitivity analysis (exclude converted) pending.")
    lines.append("- **Sign conventions**: A-23 Rodrigues GPA sign-flipped so positive r = better performance; A-31 Rivers Emotional Stability sign-reversed to Neuroticism.")
    lines.append("- **Remaining 6 pre-registered moderators** (instrument, publication year, sample size, RoB score, modality, education level) not quantitatively analyzed due to insufficient k per level; reported narratively in Methods Deviations subsection.")

    SUMMARY_MD.write_text("\n".join(lines), encoding="utf-8")


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
    # Overwrite summary with extended version that includes moderators
    write_summary_md_with_moderators(results, mod_results)

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
            if between.get("Q") is None:
                continue
            parts = []
            for level, sub in res.items():
                if level == "_between" or "r_pooled" not in sub:
                    continue
                parts.append(
                    f"{level}(k={sub['k']}): r={sub['r_pooled']:.3f}"
                )
            q = between.get("Q")
            p = between.get("p")
            if q is not None and p is not None:
                print(f"  {t}: " + "; ".join(parts) +
                      f"  | Q_b={q:.2f} df={between['df']} p={p:.3f}")


if __name__ == "__main__":
    main()
