"""
Sensitivity analyses. Pre-registered (OSF Registration §15e) and run here:
(a) Exclude low-quality studies (RoB < 5)
(b) Exclude the author's own prior primary study (Tokiwa A-25)
(c) Exclude studies whose effects were converted from β (Peterson-Brown)
(d) Exclude studies with analytic N < 50
(f) Leave-one-out
(g) DerSimonian-Laird instead of REML for tau²
Added after the pre-registration:
(h) Include A-37 Zheng, which the primary analysis leaves out of the pool
    (pool.POOL_EXCLUSIONS); its correlations, reported only as |r| < .10,
    are entered as r = 0
Not run: (e) HEXACO crosswalk variants — no HEXACO study is in the pool.

Reads data_extraction_populated.csv, writes sensitivity_results.md.
"""
import csv
import sys
from pathlib import Path
from math import log, exp, sqrt

import numpy as np
from scipy import stats

sys.path.insert(0, str(Path(__file__).parent))
from pool import (  # noqa: E402
    TRAITS, fisher_z, back_transform_z, var_z, reml_tau2,
    pool_random_effects, load_extractions, extract_effect_for_trait,
    POOL_EXCLUSIONS,
)

OUT = Path(__file__).resolve().parent / "sensitivity_results.md"


def collect_effects(rows, trait, filter_fn=None, exclude=POOL_EXCLUSIONS):
    """Collect (y, v, labels) for a trait, optionally filtering rows.

    filter_fn(row) -> bool. If returns False, study is excluded.
    """
    ys, vs, labels = [], [], []
    for row in rows:
        if filter_fn is not None and not filter_fn(row):
            continue
        ext = extract_effect_for_trait(row, trait, exclude=exclude)
        if ext is None:
            continue
        r, n, source = ext
        try:
            z = fisher_z(r)
        except (ValueError, ZeroDivisionError):
            continue
        ys.append(z)
        vs.append(var_z(n))
        labels.append(f"{row['study_id']} {row.get('first_author','')}({source})")
    return ys, vs, labels


def run_sensitivity(rows):
    results = {}

    # Primary for reference
    results["primary"] = {}
    for t in TRAITS:
        y, v, _ = collect_effects(rows, t)
        if len(y) >= 2:
            results["primary"][t] = pool_random_effects(y, v)
        else:
            results["primary"][t] = None

    # (b) Exclude Tokiwa A-25
    results["exclude_tokiwa"] = {}
    fn_not_tokiwa = lambda r: r.get("study_id") != "A-25"
    for t in TRAITS:
        y, v, _ = collect_effects(rows, t, filter_fn=fn_not_tokiwa)
        results["exclude_tokiwa"][t] = (
            pool_random_effects(y, v) if len(y) >= 2 else None
        )

    # (c) Exclude Peterson-Brown converted. Filter on the extraction source,
    # not effect_size_type: that field also marks A-37 as "r_bounded", which
    # wrongly dropped it from this scenario until 2026-10.
    results["exclude_beta_converted"] = {}
    for t in TRAITS:
        fn_r_only = lambda r, t=t: (extract_effect_for_trait(r, t) or (0, 0, ""))[2] != "beta_converted"
        y, v, _ = collect_effects(rows, t, filter_fn=fn_r_only)
        results["exclude_beta_converted"][t] = (
            pool_random_effects(y, v) if len(y) >= 2 else None
        )

    # (a) Exclude low-quality (RoB < 5)
    def fn_high_quality(row):
        try:
            return int(row.get("risk_of_bias_score", "0") or "0") >= 5
        except ValueError:
            return False
    results["exclude_low_quality"] = {}
    for t in TRAITS:
        y, v, _ = collect_effects(rows, t, filter_fn=fn_high_quality)
        results["exclude_low_quality"][t] = (
            pool_random_effects(y, v) if len(y) >= 2 else None
        )

    # (d) Exclude analytic N < 50
    def fn_n50(row):
        n = (row.get("n_for_correlations") or row.get("n_analyzed") or "").strip()
        try:
            return int(float(n)) >= 50
        except ValueError:
            return True
    results["exclude_small_n"] = {}
    for t in TRAITS:
        y, v, _ = collect_effects(rows, t, filter_fn=fn_n50)
        results["exclude_small_n"][t] = (
            pool_random_effects(y, v) if len(y) >= 2 else None
        )

    # (g) DerSimonian-Laird tau² instead of REML
    results["dl_estimator"] = {}
    for t in TRAITS:
        y, v, _ = collect_effects(rows, t)
        results["dl_estimator"][t] = (
            pool_random_effects(y, v, estimator="DL") if len(y) >= 2 else None
        )

    # (h) Include A-37 Zheng (|r| < .10 entered as r = 0)
    results["include_a37"] = {}
    keep = {k: v for k, v in POOL_EXCLUSIONS.items() if k != "A-37"}
    for t in TRAITS:
        y, v, _ = collect_effects(rows, t, exclude=keep)
        results["include_a37"][t] = (
            pool_random_effects(y, v) if len(y) >= 2 else None
        )

    # (f) Leave-one-out
    results["leave_one_out"] = {}
    for t in TRAITS:
        y_all, v_all, labels_all = collect_effects(rows, t)
        k = len(y_all)
        if k < 3:
            results["leave_one_out"][t] = []
            continue
        loo = []
        for i in range(k):
            y_sub = y_all[:i] + y_all[i+1:]
            v_sub = v_all[:i] + v_all[i+1:]
            res = pool_random_effects(y_sub, v_sub)
            loo.append({
                "dropped": labels_all[i],
                "r_pooled": res["r_pooled"],
                "r_ci_lo": res["r_ci_lo"], "r_ci_hi": res["r_ci_hi"],
                "I2": res["I2"], "tau2": res["tau2"],
            })
        results["leave_one_out"][t] = loo

    return results


def write_md(results):
    lines = [
        "# Sensitivity Analyses",
        "",
        "Pre-registered sensitivity analyses per OSF Registration §15e.",
        "",
        "## Primary pooled estimates (reference)",
        "",
        "| Trait | k | r [95% CI] | I² |",
        "|-------|---|-----------|-----|",
    ]
    for t in TRAITS:
        r = results["primary"].get(t)
        if r:
            ci = f"{r['r_pooled']:.3f} [{r['r_ci_lo']:.3f}, {r['r_ci_hi']:.3f}]"
            lines.append(f"| {t} | {r['k']} | {ci} | {r['I2']:.1f}% |")

    sections = [
        ("exclude_tokiwa", "Exclude Author's Own Study (A-25 Tokiwa 2025, COI)",
         "The author's own prior primary study was excluded to address the pre-declared conflict of interest."),
        ("exclude_beta_converted", "Exclude Peterson-Brown β-converted Studies",
         "Studies contributing only β (not zero-order r), requiring Peterson & Brown (2005) conversion, were excluded. Remaining pool uses direct r or Spearman ρ only."),
        ("exclude_low_quality", "Exclude Low-Quality Studies (RoB < 5)",
         "Studies with JBI aggregate risk-of-bias score below 5 were excluded. Higher-quality subset."),
        ("exclude_small_n", "Exclude Studies with Analytic N < 50",
         "Studies with fewer than 50 participants in the analysed sample were excluded."),
        ("dl_estimator", "DerSimonian-Laird Estimator",
         "All studies; tau² estimated with the DerSimonian-Laird moment estimator instead of REML."),
        ("include_a37", "Include A-37 Zheng (|r| < .10 entered as r = 0)",
         "A-37 pooled face-to-face and online cohorts and reported only that all trait-grade correlations were below .10 in absolute value; the primary analysis leaves it out of the pool. Here it is added with r = 0. Not pre-registered."),
    ]

    for key, title, desc in sections:
        lines.append("")
        lines.append(f"## {title}")
        lines.append("")
        lines.append(desc)
        lines.append("")
        lines.append("| Trait | k | r [95% CI] | Δr vs primary |")
        lines.append("|-------|---|-----------|---------------|")
        for t in TRAITS:
            prim = results["primary"].get(t)
            sens = results[key].get(t)
            if sens and prim:
                ci = f"{sens['r_pooled']:.3f} [{sens['r_ci_lo']:.3f}, {sens['r_ci_hi']:.3f}]"
                delta = sens["r_pooled"] - prim["r_pooled"]
                flag = " 🔴" if abs(delta) > 0.05 else ""
                lines.append(f"| {t} | {sens['k']} | {ci} | {delta:+.3f}{flag} |")
            else:
                lines.append(f"| {t} | — | insufficient k | — |")

    # Leave-one-out
    lines.append("")
    lines.append("## Leave-One-Out Analysis")
    lines.append("")
    lines.append("Each study is removed in turn to assess influence on the pooled estimate. Studies where removal causes |Δr| > 0.05 are flagged.")
    lines.append("")
    for t in TRAITS:
        loo = results["leave_one_out"].get(t, [])
        if not loo:
            continue
        prim = results["primary"][t]["r_pooled"]
        lines.append(f"### Trait {t}  (primary r = {prim:.3f})")
        lines.append("")
        lines.append("| Dropped | r_pooled | Δr | Flag |")
        lines.append("|---------|----------|-----|------|")
        for item in loo:
            delta = item["r_pooled"] - prim
            flag = "⚠" if abs(delta) > 0.05 else ""
            lines.append(
                f"| {item['dropped']} | {item['r_pooled']:.3f} | {delta:+.3f} | {flag} |"
            )
        lines.append("")

    lines.append("---")
    lines.append("")
    lines.append("## Interpretation notes")
    lines.append("- **|Δr| > 0.05** is flagged as a potentially influential change per Cochrane Handbook guidance for correlation meta-analyses.")
    lines.append("- COI sensitivity (exclude Tokiwa A-25) is particularly relevant given the author's prior study is included in the primary pool.")
    lines.append("- β-conversion sensitivity tests robustness of the Peterson-Brown approximation for studies reporting only standardized regression coefficients.")
    lines.append("- Low-quality exclusion tests whether findings are driven by studies below the pre-specified RoB threshold of 5.")
    lines.append("- Not run: the pre-registered HEXACO crosswalk variants, because no HEXACO study is in the quantitative pool.")

    OUT.write_text("\n".join(lines), encoding="utf-8")
    print(f"Wrote {OUT}")


def main():
    rows = load_extractions()
    results = run_sensitivity(rows)
    write_md(results)
    print("\n=== Sensitivity summary ===")
    for key in ("exclude_tokiwa", "exclude_beta_converted", "exclude_low_quality",
                "exclude_small_n", "dl_estimator", "include_a37"):
        print(f"\n[{key}]")
        for t in TRAITS:
            prim = results["primary"].get(t)
            sens = results[key].get(t)
            if sens and prim:
                delta = sens["r_pooled"] - prim["r_pooled"]
                print(f"  {t}: primary={prim['r_pooled']:.3f} sens={sens['r_pooled']:.3f} (Δ={delta:+.3f}) k={sens['k']}")


if __name__ == "__main__":
    main()
