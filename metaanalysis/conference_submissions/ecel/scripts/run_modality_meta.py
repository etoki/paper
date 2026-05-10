"""ECEL 2026 — Modality-stratified meta-regression and modality x trait
interaction model.

Imports the proven pooling primitives from
`metaanalysis/analysis/pool.py` (REML + HKSJ + Fisher z) so the per-trait
pooled estimates reproduce the parent preprint exactly.

Usage:
    python metaanalysis/conference_submissions/ecel/scripts/run_modality_meta.py

Outputs (to metaanalysis/conference_submissions/ecel/results/):
    modality_pools.csv         per (modality, trait) pooled r, k, N, CI, I^2
    modality_qbetween.csv      per-trait Q_between across modality levels
    interaction_terms.csv      weighted OLS interaction coefficients
    sensitivity.csv            modality pools under sensitivity scenarios
    summary.md                 human-readable summary of all of the above
"""
from __future__ import annotations

import csv
import sys
from math import sqrt
from pathlib import Path

import numpy as np
from scipy import stats

REPO_ROOT = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(REPO_ROOT / "metaanalysis" / "analysis"))

# Re-use the canonical pooling primitives from the parent preprint
import pool as parent  # noqa: E402

INPUT_CSV = REPO_ROOT / "metaanalysis" / "conference_submissions" / "inputs" / "studies.csv"
RESULTS_DIR = REPO_ROOT / "metaanalysis" / "conference_submissions" / "ecel" / "results"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

TRAITS = ["O", "C", "E", "A", "N"]
MODALITY_LEVELS = ["A", "M", "S", "U"]  # B (blended) currently empty in primary pool


# ---------------------------------------------------------------------------
# Data loader: reads the derived studies.csv (not the master extraction)
# ---------------------------------------------------------------------------
def load_studies() -> list[dict]:
    with INPUT_CSV.open("r", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def primary_pool(rows: list[dict]) -> list[dict]:
    keep = ("include", "include_with_caveat", "include_COI")
    return [
        r for r in rows
        if r.get("inclusion_status") in keep
        and r.get("primary_achievement") in ("yes", "partial")
    ]


def trait_observations(rows: list[dict], trait: str) -> list[dict]:
    """Return rows that contribute a finite r_<trait> to the pool, with N
    parsed and Fisher z computed.
    """
    out = []
    for r in rows:
        rstr = (r.get(f"r_{trait}") or "").strip()
        nstr = (r.get("N") or "").strip()
        if not rstr or not nstr:
            continue
        try:
            rho = float(rstr)
            n = int(float(nstr))
        except ValueError:
            continue
        if abs(rho) >= 0.999 or n <= 3:
            continue
        try:
            z = parent.fisher_z(rho)
        except (ValueError, ZeroDivisionError):
            continue
        out.append({
            **r,
            "_trait": trait,
            "_r": rho,
            "_n": n,
            "_z": z,
            "_v": parent.var_z(n),
        })
    return out


# ---------------------------------------------------------------------------
# Per-modality x per-trait pools
# ---------------------------------------------------------------------------
def pool_modality_for_trait(obs: list[dict]) -> dict:
    by_mod = {m: [] for m in MODALITY_LEVELS}
    for o in obs:
        m = o.get("modality", "U") or "U"
        if m not in by_mod:
            by_mod[m] = []
        by_mod[m].append(o)

    out = {}
    sub_pools_for_qb = []  # (level, z_pooled, se_hksj, k)
    for level, items in by_mod.items():
        k = len(items)
        if k == 0:
            out[level] = {"k": 0, "note": "k=0"}
            continue
        if k == 1:
            o = items[0]
            out[level] = {
                "k": 1, "N_total": o["_n"], "z_pooled": o["_z"],
                "r_pooled": o["_r"],
                "r_ci_lo": float("nan"), "r_ci_hi": float("nan"),
                "I2": float("nan"), "tau2": float("nan"),
                "Q": float("nan"), "df": 0, "p_Q": float("nan"),
                "note": "k=1 (narrative only)",
                "labels": [f"{o.get('study_id','')} {o.get('author_year','')}"],
            }
            continue
        ys = [o["_z"] for o in items]
        vs = [o["_v"] for o in items]
        labels = [f"{o.get('study_id','')} {o.get('author_year','')}" for o in items]
        res = parent.pool_random_effects(ys, vs, labels)
        res["N_total"] = sum(o["_n"] for o in items)
        res["level"] = level
        out[level] = res
        sub_pools_for_qb.append((level, res["z_pooled"], res["se_hksj"], res["k"]))

    # Q_between across modality levels with k >= 2
    if len(sub_pools_for_qb) >= 2:
        y_arr = np.array([p[1] for p in sub_pools_for_qb])
        se_arr = np.array([p[2] for p in sub_pools_for_qb])
        w = 1.0 / se_arr**2
        y_bar = float(np.sum(w * y_arr) / np.sum(w))
        Q_b = float(np.sum(w * (y_arr - y_bar) ** 2))
        df_b = len(sub_pools_for_qb) - 1
        p_b = float(1 - stats.chi2.cdf(Q_b, df_b))
        out["_between"] = {"Q": Q_b, "df": df_b, "p": p_b,
                            "k_subgroups": len(sub_pools_for_qb)}
    else:
        out["_between"] = {"Q": None, "df": 0, "p": None,
                            "k_subgroups": len(sub_pools_for_qb)}
    return out


# ---------------------------------------------------------------------------
# Long-format weighted-OLS interaction model
# ---------------------------------------------------------------------------
def interaction_model(
    primary_rows: list[dict],
    drop_modality_levels: tuple[str, ...] = ("S",),
) -> dict:
    """Build long-format design matrix with modality and trait fixed effects
    plus their interaction; fit weighted OLS with weights = 1 / (v + tau2).

    tau2 is estimated as the median of per-trait tau2 from the random-effects
    per-trait pools — a simple approximation in lieu of a full random-intercept
    mixed model. Treated as exploratory; reported with caveats.
    """
    long_obs = []
    for trait in TRAITS:
        for o in trait_observations(primary_rows, trait):
            mod = o.get("modality", "U") or "U"
            if mod in drop_modality_levels:
                continue
            long_obs.append({
                "study_id": o.get("study_id", ""),
                "trait": trait,
                "modality": mod,
                "z": o["_z"],
                "v": o["_v"],
                "n": o["_n"],
            })

    if len(long_obs) < 8:
        return {"note": "too few observations for interaction model",
                "k_obs": len(long_obs)}

    used_modalities = sorted({o["modality"] for o in long_obs})
    used_traits = TRAITS

    # Estimate tau^2 using Fisher z residuals and per-trait pools
    tau2_per_trait = []
    for trait in used_traits:
        ys = [o["z"] for o in long_obs if o["trait"] == trait]
        vs = [o["v"] for o in long_obs if o["trait"] == trait]
        if len(ys) >= 2:
            tau2_per_trait.append(parent.reml_tau2(np.array(ys), np.array(vs)))
    tau2 = float(np.median(tau2_per_trait)) if tau2_per_trait else 0.0

    # Build design matrix:
    # intercept + (T-1) trait dummies + (M-1) modality dummies +
    # (T-1)(M-1) interaction dummies
    ref_trait = used_traits[0]      # O
    ref_mod = used_modalities[0]    # alphabetical first; whichever it is

    def design_row(trait, mod):
        cols = [1.0]
        for t in used_traits[1:]:
            cols.append(1.0 if trait == t else 0.0)
        for m in used_modalities[1:]:
            cols.append(1.0 if mod == m else 0.0)
        for t in used_traits[1:]:
            for m in used_modalities[1:]:
                cols.append(1.0 if (trait == t and mod == m) else 0.0)
        return cols

    X = np.array([design_row(o["trait"], o["modality"]) for o in long_obs])
    y = np.array([o["z"] for o in long_obs])
    w = 1.0 / (np.array([o["v"] for o in long_obs]) + tau2)

    # Weighted least squares
    Wsqrt = np.sqrt(w)
    Xw = X * Wsqrt[:, None]
    yw = y * Wsqrt
    beta, *_ = np.linalg.lstsq(Xw, yw, rcond=None)

    # Covariance matrix and SE
    resid = yw - Xw @ beta
    df_resid = max(1, X.shape[0] - X.shape[1])
    sigma2 = float(np.sum(resid**2) / df_resid)
    XtWX = Xw.T @ Xw
    try:
        cov = sigma2 * np.linalg.inv(XtWX)
    except np.linalg.LinAlgError:
        cov = sigma2 * np.linalg.pinv(XtWX)
    se = np.sqrt(np.diag(cov))

    # Coefficient names
    names = ["(Intercept)"]
    names += [f"trait[{t}]" for t in used_traits[1:]]
    names += [f"modality[{m}]" for m in used_modalities[1:]]
    names += [f"trait[{t}]:modality[{m}]"
              for t in used_traits[1:] for m in used_modalities[1:]]

    coef_table = []
    for name, b, s in zip(names, beta, se):
        if s > 0:
            t_val = b / s
            p_val = 2 * (1 - stats.t.cdf(abs(t_val), df=df_resid))
        else:
            t_val = float("nan")
            p_val = float("nan")
        coef_table.append({
            "term": name, "estimate": float(b),
            "se": float(s), "t": float(t_val), "p": float(p_val),
        })

    # Joint Wald test on interaction terms only
    n_main = 1 + (len(used_traits) - 1) + (len(used_modalities) - 1)
    n_inter = X.shape[1] - n_main
    if n_inter > 0:
        L = np.zeros((n_inter, X.shape[1]))
        for i in range(n_inter):
            L[i, n_main + i] = 1.0
        Lb = L @ beta
        LcovL = L @ cov @ L.T
        try:
            wald = float(Lb @ np.linalg.inv(LcovL) @ Lb)
        except np.linalg.LinAlgError:
            wald = float(Lb @ np.linalg.pinv(LcovL) @ Lb)
        wald_p = float(1 - stats.chi2.cdf(wald, n_inter))
    else:
        wald = float("nan")
        wald_p = float("nan")

    return {
        "k_obs": int(X.shape[0]),
        "tau2_used": tau2,
        "ref_trait": ref_trait,
        "ref_modality": ref_mod,
        "coefficients": coef_table,
        "wald_chi2_interaction": wald,
        "wald_df_interaction": int(n_inter),
        "wald_p_interaction": wald_p,
        "dropped_modalities": list(drop_modality_levels),
    }


# ---------------------------------------------------------------------------
# Sensitivity scenarios
# ---------------------------------------------------------------------------
def filter_rows(rows: list[dict], scenario: str) -> list[dict]:
    if scenario == "primary":
        return primary_pool(rows)
    if scenario == "drop_beta_converted":
        return [r for r in primary_pool(rows)
                if r.get("effect_source", "") != "beta_converted"]
    if scenario == "drop_coi":
        return [r for r in primary_pool(rows)
                if r.get("inclusion_status", "") != "include_COI"]
    if scenario == "drop_unspecified_modality":
        return [r for r in primary_pool(rows) if (r.get("modality") or "U") != "U"]
    raise ValueError(f"unknown scenario: {scenario}")


# ---------------------------------------------------------------------------
# Output writers
# ---------------------------------------------------------------------------
def write_modality_pools(per_trait_results: dict[str, dict]) -> None:
    path = RESULTS_DIR / "modality_pools.csv"
    with path.open("w", encoding="utf-8", newline="") as f:
        w = csv.writer(f)
        w.writerow([
            "trait", "modality", "k", "N_total",
            "r_pooled", "r_ci_lo", "r_ci_hi",
            "I2", "tau2", "note",
        ])
        for trait in TRAITS:
            res = per_trait_results.get(trait, {})
            for level in MODALITY_LEVELS:
                sub = res.get(level)
                if sub is None:
                    w.writerow([trait, level, 0, "", "", "", "", "", "", "k=0"])
                    continue
                if "r_pooled" in sub and not isinstance(sub["r_pooled"], str):
                    w.writerow([
                        trait, level, sub["k"], sub.get("N_total", ""),
                        f"{sub['r_pooled']:.4f}",
                        ("" if (isinstance(sub.get("r_ci_lo"), float)
                                and np.isnan(sub.get("r_ci_lo")))
                         else f"{sub['r_ci_lo']:.4f}"),
                        ("" if (isinstance(sub.get("r_ci_hi"), float)
                                and np.isnan(sub.get("r_ci_hi")))
                         else f"{sub['r_ci_hi']:.4f}"),
                        ("" if (isinstance(sub.get("I2"), float)
                                and np.isnan(sub.get("I2")))
                         else f"{sub['I2']:.1f}"),
                        ("" if (isinstance(sub.get("tau2"), float)
                                and np.isnan(sub.get("tau2")))
                         else f"{sub['tau2']:.4f}"),
                        sub.get("note", ""),
                    ])
                else:
                    w.writerow([trait, level, sub.get("k", 0), "", "", "",
                                "", "", "", sub.get("note", "")])


def write_qbetween(per_trait_results: dict[str, dict]) -> None:
    path = RESULTS_DIR / "modality_qbetween.csv"
    with path.open("w", encoding="utf-8", newline="") as f:
        w = csv.writer(f)
        w.writerow(["trait", "Q_between", "df_between", "p_between", "k_subgroups"])
        for trait in TRAITS:
            res = per_trait_results.get(trait, {}).get("_between", {})
            q = res.get("Q")
            p = res.get("p")
            w.writerow([
                trait,
                "" if q is None else f"{q:.4f}",
                res.get("df", ""),
                "" if p is None else f"{p:.4f}",
                res.get("k_subgroups", 0),
            ])


def write_interaction(model: dict) -> None:
    path = RESULTS_DIR / "interaction_terms.csv"
    with path.open("w", encoding="utf-8", newline="") as f:
        w = csv.writer(f)
        w.writerow(["term", "estimate", "se", "t", "p"])
        for c in model.get("coefficients", []):
            w.writerow([
                c["term"],
                f"{c['estimate']:.4f}",
                f"{c['se']:.4f}",
                "" if (isinstance(c["t"], float) and np.isnan(c["t"])) else f"{c['t']:.3f}",
                "" if (isinstance(c["p"], float) and np.isnan(c["p"])) else f"{c['p']:.4f}",
            ])
        # Append summary lines as free-form rows
        w.writerow([])
        w.writerow(["#", "k_obs", model.get("k_obs", ""), "", ""])
        w.writerow(["#", "tau2_used", f"{model.get('tau2_used', 0):.4f}", "", ""])
        w.writerow(["#", "ref_trait", model.get("ref_trait", ""), "", ""])
        w.writerow(["#", "ref_modality", model.get("ref_modality", ""), "", ""])
        wald = model.get("wald_chi2_interaction")
        wp = model.get("wald_p_interaction")
        w.writerow(["#", "wald_chi2_interaction",
                     "" if wald is None or (isinstance(wald, float) and np.isnan(wald))
                     else f"{wald:.4f}",
                     model.get("wald_df_interaction", ""),
                     "" if wp is None or (isinstance(wp, float) and np.isnan(wp))
                     else f"{wp:.4f}"])


def write_sensitivity(rows: list[dict]) -> None:
    """Re-run modality pools under each sensitivity scenario for trait C and N.

    C and N are the two traits with the largest k in the primary pool, so
    sensitivity comparisons are most stable on these. Other traits are run
    too but reported with caveats.
    """
    path = RESULTS_DIR / "sensitivity.csv"
    scenarios = [
        "primary", "drop_beta_converted", "drop_coi", "drop_unspecified_modality",
    ]
    with path.open("w", encoding="utf-8", newline="") as f:
        w = csv.writer(f)
        w.writerow(["scenario", "trait", "modality", "k", "r_pooled", "r_ci_lo", "r_ci_hi", "tau2"])
        for sc in scenarios:
            sub_rows = filter_rows(rows, sc)
            for trait in TRAITS:
                obs = trait_observations(sub_rows, trait)
                pools = pool_modality_for_trait(obs)
                for level in MODALITY_LEVELS:
                    sub = pools.get(level, {})
                    if "r_pooled" not in sub:
                        w.writerow([sc, trait, level, sub.get("k", 0), "", "", "", ""])
                        continue
                    rci_lo = sub.get("r_ci_lo")
                    rci_hi = sub.get("r_ci_hi")
                    rci_lo_s = ("" if isinstance(rci_lo, float) and np.isnan(rci_lo)
                                else f"{rci_lo:.4f}")
                    rci_hi_s = ("" if isinstance(rci_hi, float) and np.isnan(rci_hi)
                                else f"{rci_hi:.4f}")
                    tau2 = sub.get("tau2")
                    tau2_s = ("" if isinstance(tau2, float) and np.isnan(tau2)
                              else f"{tau2:.4f}")
                    w.writerow([sc, trait, level, sub["k"],
                                f"{sub['r_pooled']:.4f}", rci_lo_s, rci_hi_s, tau2_s])


def write_summary_md(per_trait_results: dict[str, dict],
                      interaction: dict,
                      n_primary_rows: int) -> None:
    path = RESULTS_DIR / "summary.md"
    lines = [
        "# ECEL 2026 — Modality-Stratified Meta-Analysis Results",
        "",
        f"**Generated by**: `metaanalysis/conference_submissions/ecel/scripts/run_modality_meta.py`",
        f"**Input**: `metaanalysis/conference_submissions/inputs/studies.csv`",
        f"**Primary-pool rows considered**: {n_primary_rows}",
        "",
        "## Per-modality x per-trait pooled correlations",
        "",
        "| Trait | Modality | k | N | r [95% CI] | I^2 | tau^2 | Note |",
        "|-------|----------|---|---|-----------|------|-------|------|",
    ]
    for trait in TRAITS:
        res = per_trait_results.get(trait, {})
        for level in MODALITY_LEVELS:
            sub = res.get(level, {})
            k = sub.get("k", 0)
            if "r_pooled" in sub and not (isinstance(sub.get("r_pooled"), float) and np.isnan(sub.get("r_pooled"))):
                rci_lo = sub.get("r_ci_lo")
                rci_hi = sub.get("r_ci_hi")
                if isinstance(rci_lo, float) and np.isnan(rci_lo):
                    ci = f"{sub['r_pooled']:.3f} (no CI; k={k})"
                else:
                    ci = f"{sub['r_pooled']:.3f} [{rci_lo:.3f}, {rci_hi:.3f}]"
                i2 = sub.get("I2")
                t2 = sub.get("tau2")
                i2s = "—" if (isinstance(i2, float) and np.isnan(i2)) else f"{i2:.1f}%"
                t2s = "—" if (isinstance(t2, float) and np.isnan(t2)) else f"{t2:.4f}"
                lines.append(f"| {trait} | {level} | {k} | {sub.get('N_total','')} | {ci} | {i2s} | {t2s} | {sub.get('note','')} |")
            else:
                lines.append(f"| {trait} | {level} | {k} | — | — | — | — | {sub.get('note','')} |")
    lines.append("")

    lines.append("## Q_between across modality levels (per trait)")
    lines.append("")
    lines.append("| Trait | Q_between | df | p_between | k_subgroups |")
    lines.append("|-------|-----------|----|----------|------------|")
    for trait in TRAITS:
        res = per_trait_results.get(trait, {}).get("_between", {})
        q = res.get("Q")
        p = res.get("p")
        lines.append(
            f"| {trait} | "
            f"{'—' if q is None else f'{q:.3f}'} | "
            f"{res.get('df','—')} | "
            f"{'—' if p is None else f'{p:.4f}'} | "
            f"{res.get('k_subgroups', 0)} |"
        )
    lines.append("")

    lines.append("## Modality x trait interaction (long-format weighted OLS)")
    lines.append("")
    if "coefficients" in interaction:
        lines.append(
            f"- k observations: {interaction['k_obs']}  ")
        lines.append(
            f"- tau^2 used in weights: {interaction['tau2_used']:.4f}  ")
        lines.append(
            f"- reference cell: trait = {interaction['ref_trait']}, modality = {interaction['ref_modality']}  ")
        lines.append(
            f"- dropped modality levels (k<2): {interaction['dropped_modalities']}  ")
        lines.append("")
        lines.append("| Term | Estimate (Fisher z) | SE | t | p |")
        lines.append("|------|--------------------:|---:|---:|---:|")
        for c in interaction["coefficients"]:
            t_str = ("" if (isinstance(c["t"], float) and np.isnan(c["t"]))
                     else f"{c['t']:.2f}")
            p_str = ("" if (isinstance(c["p"], float) and np.isnan(c["p"]))
                     else f"{c['p']:.3f}")
            lines.append(f"| {c['term']} | {c['estimate']:.4f} | {c['se']:.4f} | {t_str} | {p_str} |")
        wald = interaction.get("wald_chi2_interaction")
        wp = interaction.get("wald_p_interaction")
        df_inter = interaction.get("wald_df_interaction")
        lines.append("")
        lines.append(
            f"**Joint Wald test on interaction terms**: "
            f"chi^2({df_inter}) = "
            f"{'—' if wald is None or (isinstance(wald, float) and np.isnan(wald)) else f'{wald:.3f}'}, "
            f"p = "
            f"{'—' if wp is None or (isinstance(wp, float) and np.isnan(wp)) else f'{wp:.4f}'}."
        )
    else:
        lines.append(f"- {interaction.get('note', 'unavailable')}")
    lines.append("")

    lines.append("## Caveats")
    lines.append("")
    lines.append("- Per-modality cell k is small (k = 1–4 per cell). The synchronous bucket has k = 1 and is reported narratively only.")
    lines.append("- The interaction model uses a tau^2 estimate equal to the median of per-trait REML tau^2 estimates; this is an approximation in lieu of a full random-intercept mixed model and is reported as exploratory.")
    lines.append("- The modality classification is derived from `data_extraction_populated.csv` via `inputs/derive_studies_csv.py::classify_modality`; ambiguous cases are recorded as 'U' (unspecified).")
    lines.append("- Sensitivity scenarios are reported in `sensitivity.csv`.")
    path.write_text("\n".join(lines), encoding="utf-8")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main() -> None:
    if not INPUT_CSV.exists():
        raise SystemExit(
            f"Missing studies.csv: run `python {INPUT_CSV.parent / 'derive_studies_csv.py'}` first."
        )

    rows = load_studies()
    primary = primary_pool(rows)

    print(f"[ECEL] {len(rows)} total rows; {len(primary)} eligible primary-pool studies.")

    # Per-modality x per-trait pools
    per_trait_results: dict[str, dict] = {}
    for trait in TRAITS:
        obs = trait_observations(primary, trait)
        per_trait_results[trait] = pool_modality_for_trait(obs)
        print(f"[ECEL] trait={trait}: {len(obs)} observations across modalities")

    # Long-format interaction model (drops S because k=1 in current corpus)
    interaction = interaction_model(primary, drop_modality_levels=("S",))
    if "wald_chi2_interaction" in interaction:
        print(
            f"[ECEL] interaction Wald: chi2({interaction['wald_df_interaction']})="
            f"{interaction['wald_chi2_interaction']:.3f}, p={interaction['wald_p_interaction']:.4f}"
        )

    # Outputs
    write_modality_pools(per_trait_results)
    write_qbetween(per_trait_results)
    write_interaction(interaction)
    write_sensitivity(rows)
    write_summary_md(per_trait_results, interaction, len(primary))

    print(f"[ECEL] wrote {RESULTS_DIR}/modality_pools.csv")
    print(f"[ECEL] wrote {RESULTS_DIR}/modality_qbetween.csv")
    print(f"[ECEL] wrote {RESULTS_DIR}/interaction_terms.csv")
    print(f"[ECEL] wrote {RESULTS_DIR}/sensitivity.csv")
    print(f"[ECEL] wrote {RESULTS_DIR}/summary.md")


if __name__ == "__main__":
    main()
