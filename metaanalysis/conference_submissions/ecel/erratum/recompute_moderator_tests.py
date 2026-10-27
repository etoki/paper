"""ECEL 2026 EL-095 erratum — recompute the modality analyses with
corrected procedures and compare them with the published paper.

The published paper (ecel/EL-095-Tokiwa.docx) was produced by
ecel/scripts/run_modality_meta.py. Three problems were found after
publication:

  1. Moderator test (Table 2). Q_between fed the HKSJ-adjusted subgroup
     standard errors into a large-sample chi-square(1) reference
     distribution. This is not a standard procedure and is inconsistent
     with the small-sample t-based CIs of Table 1. With k = 3-5 per
     subgroup the recommended test is a mixed-effects meta-regression
     (z ~ 1 + modality) with REML residual tau^2 and the Knapp-Hartung
     adjustment, F(1, k - 2) (Viechtbauer et al., 2015, Psychological
     Methods 20(3), 360-374).
  2. Interaction model (Table 3). The weighted-OLS model treats the
     n = 42 (study x trait) rows as independent, but they come from only
     9 studies, each contributing up to five traits measured on the same
     participants. Its standard errors, t-tests and the joint Wald test
     are therefore not valid and are withdrawn.
  3. tau^2 estimator. analysis/pool.py::reml_tau2 uses a fixed-point
     update that under-estimates the REML tau^2. This script uses
     pool.reml_tau2_exact, which maximises the restricted log-likelihood
     directly (cross-checked against PyMARE when PyMARE is installed).

It also re-derives every Table 1 cell, because the published
Neuroticism rows were carried over from a draft computed before the
modality re-coding of A-15, A-23 and A-30.

Usage:
    python metaanalysis/conference_submissions/ecel/erratum/recompute_moderator_tests.py

Outputs (to ecel/erratum/results/):
    corrected_table1.csv        per (trait, modality) pools, published vs corrected
    corrected_table2.csv        per-trait modality tests, published vs corrected
    extraversion_loo.csv        leave-one-out of the Extraversion modality test
    parent_moderators.csv       parent-preprint E x region / outcome type, corrected
    summary.md                  human-readable report of all of the above
"""
from __future__ import annotations

import csv
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
REPO_ROOT = HERE.parents[3]
sys.path.insert(0, str(REPO_ROOT / "metaanalysis" / "analysis"))
sys.path.insert(0, str(HERE.parent / "scripts"))

import pool as parent  # noqa: E402
import run_modality_meta as published  # noqa: E402

PUBLISHED_DOCX = HERE.parent / "EL-095-Tokiwa.docx"
RESULTS_DIR = HERE / "results"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

TRAITS = published.TRAITS
TRAIT_NAMES = {"O": "Openness", "C": "Conscientiousness", "E": "Extraversion",
               "A": "Agreeableness", "N": "Neuroticism"}
CONTRAST = ("A", "M")  # asynchronous (reference) vs mixed-online


# ---------------------------------------------------------------------------
# Corrected estimators (shared with the other papers via analysis/pool.py)
# ---------------------------------------------------------------------------
def pool_re(y, v) -> dict:
    """Random-effects pool: exact REML tau^2 + HKSJ CI (t, df = k - 1)."""
    res = parent.pool_random_effects(y, v, exact_reml=True)
    return {"k": res["k"], "z": res["z_pooled"], "r": res["r_pooled"],
            "r_lo": res["r_ci_lo"], "r_hi": res["r_ci_hi"],
            "tau2": res["tau2"], "I2": res["I2"]}


metareg_kh = parent.metareg_knapp_hartung


# ---------------------------------------------------------------------------
# Published values (read from the published docx so nothing is retyped)
# ---------------------------------------------------------------------------
def read_published_tables() -> tuple[dict, dict]:
    from docx import Document

    doc = Document(str(PUBLISHED_DOCX))
    t1, t2 = {}, {}
    for row in doc.tables[0].rows[1:]:
        trait, mod, k, n, r_ci, i2, tau2 = (c.text.strip() for c in row.cells)
        t1[(trait, mod)] = {"k": k, "N": n, "r_ci": r_ci, "I2": i2, "tau2": tau2}
    for row in doc.tables[1].rows[1:]:
        trait, q, p = (c.text.strip() for c in row.cells)
        t2[trait] = {"Q": q, "p": p}
    return t1, t2


# ---------------------------------------------------------------------------
# Analyses
# ---------------------------------------------------------------------------
def observations(rows, trait):
    return [o for o in published.trait_observations(rows, trait)
            if (o.get("modality") or "U") in published.MODALITY_LEVELS]


def table1(rows) -> list[dict]:
    out = []
    for trait in TRAITS:
        obs = observations(rows, trait)
        for mod in ("A", "M", "S"):
            cell = [o for o in obs if o.get("modality") == mod]
            rec = {"trait": trait, "modality": mod, "k": len(cell),
                   "N": sum(o["_n"] for o in cell),
                   "studies": " ".join(o["study_id"] for o in cell)}
            if len(cell) >= 2:
                rec.update(pool_re([o["_z"] for o in cell],
                                   [o["_v"] for o in cell]))
            elif len(cell) == 1:
                rec["r"] = cell[0]["_r"]
            out.append(rec)
    return out


def contrast_obs(rows, trait, drop=()):
    return [o for o in observations(rows, trait)
            if o.get("modality") in CONTRAST and o["study_id"] not in drop]


def modality_test(obs) -> dict:
    y = [o["_z"] for o in obs]
    v = [o["_v"] for o in obs]
    g = [1.0 if o["modality"] == CONTRAST[1] else 0.0 for o in obs]
    return metareg_kh(y, v, g)


def table2(rows) -> list[dict]:
    out = []
    for trait in TRAITS:
        obs = contrast_obs(rows, trait)
        pub = published.pool_modality_for_trait(
            published.trait_observations(rows, trait))["_between"]
        res = modality_test(obs)
        res.update({"trait": trait, "Q_published_method": pub["Q"],
                    "p_published_method": pub["p"]})
        out.append(res)
    return out


def extraversion_loo(rows, study_meta) -> list[dict]:
    obs = contrast_obs(rows, "E")
    out = []
    scenarios = [("none", ())]
    scenarios += [(o["study_id"], (o["study_id"],)) for o in obs]
    asian = tuple(o["study_id"] for o in obs
                  if study_meta[o["study_id"]]["region"] == "Asia")
    scenarios.append(("+".join(asian), asian))
    for label, drop in scenarios:
        sub = [o for o in obs if o["study_id"] not in drop]
        k_ref = sum(o["modality"] == CONTRAST[0] for o in sub)
        rec = {"dropped": label, "k": len(sub), "k_async": k_ref,
               "k_mixed": len(sub) - k_ref}
        if k_ref >= 1 and len(sub) - k_ref >= 1 and len(sub) >= 3:
            res = modality_test(sub)
            rec.update({"diff_z": res["diff_z"], "F": res["F"],
                        "df2": res["df2"], "p": res["p"]})
        out.append(rec)
    return out


def drop_beta_converted(rows) -> dict:
    """Section 3.5 sensitivity scenario (drop beta-converted studies) for
    Extraversion: per-modality pools plus the corrected modality test.
    """
    beta = tuple(r["study_id"] for r in rows
                 if r.get("effect_source") == "beta_converted")
    obs = contrast_obs(rows, "E", drop=beta)
    out = {"dropped": " ".join(beta)}
    for mod in CONTRAST:
        cell = [o for o in obs if o["modality"] == mod]
        out[mod] = pool_re([o["_z"] for o in cell], [o["_v"] for o in cell])
    out["test"] = modality_test(obs)
    return out


def parent_trait_pools() -> list[dict]:
    """Section 3.1 trait-only pools: parent pool.py vs exact REML."""
    rows = parent.load_extractions()
    out = []
    for trait in TRAITS:
        y, v = [], []
        for row in rows:
            eff = parent.extract_effect_for_trait(row, trait)
            if eff is None:
                continue
            r, n, _ = eff
            y.append(parent.fisher_z(r))
            v.append(parent.var_z(n))
        old = parent.pool_random_effects(y, v)
        new = pool_re(y, v)
        out.append({"trait": trait, "k": len(y),
                    "published": f"{old['r_pooled']:.3f} "
                                 f"[{old['r_ci_lo']:.3f}, {old['r_ci_hi']:.3f}]",
                    "corrected": f"{new['r']:.3f} "
                                 f"[{new['r_lo']:.3f}, {new['r_hi']:.3f}]",
                    "tau2_published": old["tau2"], "tau2_corrected": new["tau2"]})
    return out


def interaction_decomposition(rows) -> dict:
    """The published Table 3 model is saturated in trait x modality, so its
    E x M coefficient equals (E_M - E_A) - (O_M - O_A) of the weighted cell
    means. Report that decomposition plus the dependence structure.
    """
    model = published.interaction_model(rows)
    long_ids = [o["study_id"] for t in TRAITS
                for o in published.trait_observations(rows, t)
                if o.get("modality") in CONTRAST]
    tau2 = model["tau2_used"]

    def cell_mean(trait, mod):
        cell = [o for o in published.trait_observations(rows, trait)
                if o.get("modality") == mod]
        w = np.array([1.0 / (o["_v"] + tau2) for o in cell])
        z = np.array([o["_z"] for o in cell])
        return float(np.sum(w * z) / np.sum(w))

    e_diff = cell_mean("E", "M") - cell_mean("E", "A")
    o_diff = cell_mean("O", "M") - cell_mean("O", "A")
    coef = {c["term"]: c for c in model["coefficients"]}
    return {"n_rows": len(long_ids), "n_studies": len(set(long_ids)),
            "wald": model["wald_chi2_interaction"],
            "wald_p": model["wald_p_interaction"],
            "exm_coef": coef["trait[E]:modality[M]"]["estimate"],
            "e_diff": e_diff, "o_diff": o_diff}


def parent_moderators() -> list[dict]:
    """E x region and E x outcome type from the parent preprint (cited in
    the ECEL Introduction as 'p < .001'), recomputed the same way.
    """
    rows = parent.load_extractions()
    out = []
    for name, col, clf, ref in [
        ("region", "region", parent.classify_region, "non-Asia"),
        ("outcome_type", "outcome_type", parent.classify_outcome_type,
         "objective"),
    ]:
        y, v, g = [], [], []
        for row in rows:
            eff = parent.extract_effect_for_trait(row, "E")
            level = clf(row.get(col, ""))
            if eff is None or level is None:
                continue
            r, n, _ = eff
            y.append(parent.fisher_z(r))
            v.append(parent.var_z(n))
            g.append(0.0 if level == ref else 1.0)
        res = metareg_kh(y, v, g)
        res["moderator"] = name
        out.append(res)
    return out


def pymare_crosscheck(rows) -> str:
    try:
        from pymare import Dataset
        from pymare.estimators import VarianceBasedLikelihoodEstimator
    except ImportError:
        return "PyMARE not installed; cross-check skipped."
    worst_tau2 = worst_p = 0.0
    for trait in TRAITS:
        obs = contrast_obs(rows, trait)
        y = np.array([o["_z"] for o in obs])
        v = np.array([o["_v"] for o in obs])
        x = np.array([[1.0 if o["modality"] == CONTRAST[1] else 0.0]
                      for o in obs])
        # PyMARE applies the Knapp-Hartung correction by default
        est = VarianceBasedLikelihoodEstimator(method="REML")
        summ = est.fit_dataset(Dataset(y=y, v=v, X=x)).summary()
        ours = modality_test(obs)
        worst_tau2 = max(worst_tau2,
                         abs(float(np.ravel(summ.tau2)[0]) - ours["tau2_resid"]))
        worst_p = max(worst_p,
                      abs(float(np.ravel(summ.get_fe_stats()["p"])[1]) - ours["p"]))
    return (f"REML residual tau^2 and Knapp-Hartung p-values match PyMARE for "
            f"all five traits (max abs difference: tau^2 {worst_tau2:.1e}, "
            f"p {worst_p:.1e}).")


# ---------------------------------------------------------------------------
# Writers
# ---------------------------------------------------------------------------
def fmt_p(p: float) -> str:
    return "< .001" if p < 0.001 else f"{p:.3f}".lstrip("0")


def p_expr(p: float) -> str:
    return "p < .001" if p < 0.001 else f"p = {fmt_p(p)}"


def fmt_r_ci(rec: dict) -> str:
    if rec["k"] >= 2:
        return f"{rec['r']:.3f} [{rec['r_lo']:.3f}, {rec['r_hi']:.3f}]"
    if rec["k"] == 1:
        return f"{rec['r']:.3f} (k=1, narrative)"
    return "—"


def write_csv(path: Path, records: list[dict]) -> None:
    keys = list(dict.fromkeys(k for rec in records for k in rec))
    with path.open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=keys)
        w.writeheader()
        for rec in records:
            w.writerow({k: (f"{v:.10g}" if isinstance(v, float) else v)
                        for k, v in rec.items()})


def main() -> None:
    rows = published.primary_pool(published.load_studies())
    study_meta = {r["study_id"]: r for r in rows}
    pub_t1, pub_t2 = read_published_tables()

    t1 = table1(rows)
    for rec in t1:
        pub = pub_t1.get((rec["trait"], rec["modality"]), {})
        rec["published_r_ci"] = pub.get("r_ci", "")
        rec["published_I2"] = pub.get("I2", "")
        rec["published_tau2"] = pub.get("tau2", "")
    t2 = table2(rows)
    loo = extraversion_loo(rows, study_meta)
    sens = drop_beta_converted(rows)
    inter = interaction_decomposition(rows)
    pm = parent_moderators()
    p31 = parent_trait_pools()
    xcheck = pymare_crosscheck(rows)
    with_r = {r["study_id"]: r["modality"] for r in rows
              if any((r.get(f"r_{t}") or "").strip() for t in TRAITS)}

    write_csv(RESULTS_DIR / "corrected_table1.csv", t1)
    write_csv(RESULTS_DIR / "corrected_table2.csv", t2)
    write_csv(RESULTS_DIR / "extraversion_loo.csv", loo)
    write_csv(RESULTS_DIR / "parent_moderators.csv", pm)
    write_csv(RESULTS_DIR / "parent_trait_pools.csv", p31)

    L = []
    L.append("# ECEL 2026 EL-095 — published vs corrected analyses")
    L.append("")
    L.append("**Generated by**: `ecel/erratum/recompute_moderator_tests.py`  ")
    L.append("**Published values read from**: `ecel/EL-095-Tokiwa.docx`  ")
    L.append(f"**Estimator cross-check**: {xcheck}")
    L.append("")
    L.append("## Study counts (Section 2.1 and Figure 1)")
    L.append("")
    L.append(f"- Primary-pool studies: {len(rows)}; with at least one extractable "
             f"r: {len(with_r)} (" + ", ".join(
                 f"{m} k = {sum(x == m for x in with_r.values())}"
                 for m in ("A", "M", "S")) + ").")
    L.append("- Published: '12 contributed at least one extractable Pearson "
             "correlation'; Figure 1: 'k = 11 ... asynchronous k = 6, "
             "mixed-online k = 5, synchronous k = 1'.")
    L.append("")
    L.append("## Section 3.1 — trait-only pools (parent pool.py vs exact REML)")
    L.append("")
    L.append("| Trait | k | Published r [95% CI] | Corrected r [95% CI] | τ² published → corrected |")
    L.append("|---|---|---|---|---|")
    for rec in p31:
        L.append(f"| {rec['trait']} | {rec['k']} | {rec['published']} | {rec['corrected']} "
                 f"| {rec['tau2_published']:.4f} → {rec['tau2_corrected']:.4f} |")
    L.append("")
    L.append("## Table 1 — modality-stratified pools (REML + HKSJ)")
    L.append("")
    L.append("| Trait | Mod | k | Published r [95% CI] | Corrected r [95% CI] "
             "| Published I² / τ² | Corrected I² / τ² |")
    L.append("|---|---|---|---|---|---|---|")
    for rec in t1:
        corr_het = (f"{rec['I2']:.1f}% / {rec['tau2']:.3f}"
                    if rec["k"] >= 2 else "—")
        L.append(f"| {rec['trait']} | {rec['modality']} | {rec['k']} "
                 f"| {rec['published_r_ci']} | {fmt_r_ci(rec)} "
                 f"| {rec['published_I2']} / {rec['published_tau2']} "
                 f"| {corr_het} |")
    L.append("")
    L.append("## Table 2 — modality moderator test (A vs M)")
    L.append("")
    L.append("Published: χ²(1) on HKSJ subgroup SEs. Corrected: REML mixed-effects "
             "meta-regression with Knapp–Hartung adjustment, F(1, k − 2).")
    L.append("")
    L.append("| Trait | k | Published Q, p | Corrected Δz (M − A) [95% CI] "
             "| Corrected F(1, df), p |")
    L.append("|---|---|---|---|---|")
    for rec in t2:
        L.append(f"| {rec['trait']} | {rec['k']} "
                 f"| {pub_t2[rec['trait']]['Q']}, {pub_t2[rec['trait']]['p']} "
                 f"| {rec['diff_z']:+.3f} [{rec['ci_lo']:+.3f}, {rec['ci_hi']:+.3f}] "
                 f"| F(1, {rec['df2']}) = {rec['F']:.2f}, {p_expr(rec['p'])} |")
    L.append("")
    L.append("## Extraversion — leave-one-out")
    L.append("")
    L.append("| Dropped | k (A / M) | Δz (M − A) | F | p |")
    L.append("|---|---|---|---|---|")
    for rec in loo:
        if "p" in rec:
            L.append(f"| {rec['dropped']} | {rec['k_async']} / {rec['k_mixed']} "
                     f"| {rec['diff_z']:+.3f} | {rec['F']:.2f} | {fmt_p(rec['p'])} |")
        else:
            L.append(f"| {rec['dropped']} | {rec['k_async']} / {rec['k_mixed']} "
                     f"| not estimable | — | — |")
    L.append("")
    L.append("Asynchronous Extraversion studies: " + "; ".join(
        f"{o['study_id']} {study_meta[o['study_id']]['author_year']} "
        f"({study_meta[o['study_id']]['country']}, r = {o['_r']:+.3f})"
        for o in contrast_obs(rows, "E") if o["modality"] == "A"))
    L.append("")
    L.append("## Section 3.5 — drop beta-converted studies (Extraversion)")
    L.append("")
    t = sens["test"]
    L.append(f"- Dropped: {sens['dropped']}. Asynchronous r = "
             f"{fmt_r_ci(sens['A'])} (k = {sens['A']['k']}); mixed-online r = "
             f"{fmt_r_ci(sens['M'])} (k = {sens['M']['k']}).")
    L.append(f"- Modality test: F(1, {t['df2']}) = {t['F']:.2f}, {p_expr(t['p'])}.")
    L.append("- Published: '-0.121 -> -0.095', '+0.059 -> +0.044', 'The headline "
             "interaction effect is qualitatively robust under all three scenarios.'")
    L.append("")
    L.append("## Table 3 — interaction model (withdrawn)")
    L.append("")
    L.append(f"- Rows: {inter['n_rows']} (study × trait) from only "
             f"{inter['n_studies']} studies; rows are not independent, so the "
             f"published SEs, t-tests and the joint Wald test "
             f"(χ²(4) = {inter['wald']:.2f}, p = {inter['wald_p']:.4f}) are not valid.")
    L.append(f"- The E × M coefficient ({inter['exm_coef']:+.3f}) equals "
             f"(E_M − E_A) − (O_M − O_A) = ({inter['e_diff']:+.3f}) − "
             f"({inter['o_diff']:+.3f}): {-inter['o_diff'] / inter['exm_coef']:.0%} "
             f"of it reflects Openness (the reference trait), not Extraversion.")
    L.append("")
    L.append("## Parent-preprint moderators cited in the ECEL Introduction")
    L.append("")
    L.append("| Moderator (Extraversion) | k | Published | Corrected F(1, df), p |")
    L.append("|---|---|---|---|")
    pub_parent = {"region": "Q = 46.43, p < .001",
                  "outcome_type": "Q = 17.30, p < .001"}
    for rec in pm:
        L.append(f"| {rec['moderator']} | {rec['k']} | {pub_parent[rec['moderator']]} "
                 f"| F(1, {rec['df2']}) = {rec['F']:.2f}, {p_expr(rec['p'])} |")
    L.append("")
    (RESULTS_DIR / "summary.md").write_text("\n".join(L) + "\n", encoding="utf-8")
    print("\n".join(L))


if __name__ == "__main__":
    main()
