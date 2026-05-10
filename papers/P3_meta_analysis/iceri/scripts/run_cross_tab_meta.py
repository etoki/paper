"""ICERI 2026 — Education-level x discipline cross-tabulated meta-analytic
interaction.

Re-uses the parent pooling primitives from
`metaanalysis/analysis/pool.py`.

Usage:
    python papers/P3_meta_analysis/iceri/scripts/run_cross_tab_meta.py

Outputs (to papers/P3_meta_analysis/iceri/results/):
    cross_tab_pools.csv      per (level, discipline, trait) pooled r when k>=2
    interaction_terms.csv    long-format interaction model coefficients
    cross_tab_summary.md     human-readable cross-tab table + narrative
"""
from __future__ import annotations

import csv
import sys
from pathlib import Path

import numpy as np
from scipy import stats

REPO_ROOT = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(REPO_ROOT / "metaanalysis" / "analysis"))
import pool as parent  # noqa: E402

INPUT_CSV = REPO_ROOT / "papers" / "P3_meta_analysis" / "inputs" / "studies.csv"
RESULTS_DIR = REPO_ROOT / "papers" / "P3_meta_analysis" / "iceri" / "results"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

TRAITS = ("O", "C", "E", "A", "N")

LEVEL_MAP = {
    "K-12": "K-12",
    "HS_Year3": "K-12",
    "HS_Grade12": "K-12",
    "Undergraduate": "UG",
    "Mixed_secondary_postsecondary": "UG",
    "Mixed_UG_Grad": "Mixed_UG_Grad",
    "Graduate": "Graduate",
}

DISCIPLINES = ("STEM", "Humanities", "Psychology", "Mixed")
LEVELS = ("K-12", "UG", "Graduate", "Mixed_UG_Grad")


def load_studies():
    with INPUT_CSV.open("r", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def primary_pool(rows):
    keep = ("include", "include_with_caveat", "include_COI")
    return [
        r for r in rows
        if r.get("inclusion_status") in keep
        and r.get("primary_achievement") in ("yes", "partial")
    ]


def collapse_level(raw):
    return LEVEL_MAP.get(raw or "", "")


def collapse_discipline(raw):
    return raw if raw in DISCIPLINES else ""


def trait_observations(rows, trait):
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
            **r, "_trait": trait, "_r": rho, "_n": n, "_z": z, "_v": parent.var_z(n),
            "_level": collapse_level(r.get("education_level", "")),
            "_disc": collapse_discipline(r.get("discipline", "")),
        })
    return out


def cross_tab_pools(primary):
    """Per (level, discipline, trait) pooled r when k>=2."""
    out = {}
    for trait in TRAITS:
        obs = trait_observations(primary, trait)
        cells = {}
        for o in obs:
            key = (o["_level"], o["_disc"])
            cells.setdefault(key, []).append(o)
        for (lvl, disc), items in cells.items():
            k = len(items)
            cell = {"trait": trait, "level": lvl, "discipline": disc, "k": k}
            if k >= 2:
                ys = [it["_z"] for it in items]
                vs = [it["_v"] for it in items]
                res = parent.pool_random_effects(ys, vs)
                cell.update({
                    "N_total": sum(it["_n"] for it in items),
                    "r_pooled": res["r_pooled"],
                    "r_ci_lo": res["r_ci_lo"],
                    "r_ci_hi": res["r_ci_hi"],
                    "I2": res["I2"],
                    "tau2": res["tau2"],
                })
            elif k == 1:
                it = items[0]
                cell.update({
                    "N_total": it["_n"], "r_pooled": it["_r"],
                    "r_ci_lo": "", "r_ci_hi": "",
                    "I2": "", "tau2": "", "note": "k=1 single study",
                })
            out[(trait, lvl, disc)] = cell
    return out


def interaction_model(primary):
    """Long-format weighted OLS with level + discipline + interaction terms.

    Levels and disciplines with k<2 across all observations are dropped to
    avoid singular design matrices.
    """
    long_obs = []
    for trait in TRAITS:
        for o in trait_observations(primary, trait):
            if not o["_level"] or not o["_disc"]:
                continue
            long_obs.append({
                "study_id": o.get("study_id", ""),
                "trait": trait, "level": o["_level"], "discipline": o["_disc"],
                "z": o["_z"], "v": o["_v"], "n": o["_n"],
            })

    if len(long_obs) < 6:
        return {"note": "too few observations", "k_obs": len(long_obs)}

    used_levels = sorted({o["level"] for o in long_obs})
    used_disc = sorted({o["discipline"] for o in long_obs})
    used_traits = TRAITS

    tau2_list = []
    for trait in used_traits:
        ys = [o["z"] for o in long_obs if o["trait"] == trait]
        vs = [o["v"] for o in long_obs if o["trait"] == trait]
        if len(ys) >= 2:
            tau2_list.append(parent.reml_tau2(np.array(ys), np.array(vs)))
    tau2 = float(np.median(tau2_list)) if tau2_list else 0.0

    ref_lvl = used_levels[0]
    ref_disc = used_disc[0]
    ref_trait = used_traits[0]

    def design_row(trait, lvl, disc):
        cols = [1.0]
        for t in used_traits[1:]:
            cols.append(1.0 if trait == t else 0.0)
        for l in used_levels[1:]:
            cols.append(1.0 if lvl == l else 0.0)
        for d in used_disc[1:]:
            cols.append(1.0 if disc == d else 0.0)
        # interaction terms
        for l in used_levels[1:]:
            for d in used_disc[1:]:
                cols.append(1.0 if (lvl == l and disc == d) else 0.0)
        return cols

    X = np.array([design_row(o["trait"], o["level"], o["discipline"]) for o in long_obs])
    y = np.array([o["z"] for o in long_obs])
    w = 1.0 / (np.array([o["v"] for o in long_obs]) + tau2)

    Wsqrt = np.sqrt(w)
    Xw = X * Wsqrt[:, None]
    yw = y * Wsqrt
    beta, *_ = np.linalg.lstsq(Xw, yw, rcond=None)
    resid = yw - Xw @ beta
    df_resid = max(1, X.shape[0] - X.shape[1])
    sigma2 = float(np.sum(resid ** 2) / df_resid)
    try:
        cov = sigma2 * np.linalg.inv(Xw.T @ Xw)
    except np.linalg.LinAlgError:
        cov = sigma2 * np.linalg.pinv(Xw.T @ Xw)
    se = np.sqrt(np.diag(cov))

    names = ["(Intercept)"]
    names += [f"trait[{t}]" for t in used_traits[1:]]
    names += [f"level[{l}]" for l in used_levels[1:]]
    names += [f"discipline[{d}]" for d in used_disc[1:]]
    names += [f"level[{l}]:discipline[{d}]"
              for l in used_levels[1:] for d in used_disc[1:]]

    coefs = []
    for nm, b, s in zip(names, beta, se):
        if s > 0:
            t_val = b / s
            p_val = 2 * (1 - stats.t.cdf(abs(t_val), df=df_resid))
        else:
            t_val = float("nan")
            p_val = float("nan")
        coefs.append({"term": nm, "estimate": float(b), "se": float(s),
                      "t": float(t_val), "p": float(p_val)})

    # Joint Wald on interaction terms
    n_main = 1 + (len(used_traits) - 1) + (len(used_levels) - 1) + (len(used_disc) - 1)
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
        "ref_trait": ref_trait, "ref_level": ref_lvl, "ref_disc": ref_disc,
        "coefficients": coefs,
        "wald_chi2_interaction": wald,
        "wald_df_interaction": int(n_inter),
        "wald_p_interaction": wald_p,
        "used_levels": used_levels, "used_disciplines": used_disc,
    }


def write_cross_tab_csv(cells):
    path = RESULTS_DIR / "cross_tab_pools.csv"
    with path.open("w", encoding="utf-8", newline="") as f:
        w = csv.writer(f)
        w.writerow(["trait", "level", "discipline", "k", "N_total",
                     "r_pooled", "r_ci_lo", "r_ci_hi", "I2", "tau2", "note"])
        for key in sorted(cells.keys()):
            c = cells[key]
            r = c.get("r_pooled", "")
            row = [c["trait"], c["level"], c["discipline"], c["k"],
                    c.get("N_total", "")]
            if isinstance(r, float):
                row += [f"{r:.4f}",
                        f"{c['r_ci_lo']:.4f}" if isinstance(c["r_ci_lo"], float) else c["r_ci_lo"],
                        f"{c['r_ci_hi']:.4f}" if isinstance(c["r_ci_hi"], float) else c["r_ci_hi"],
                        f"{c['I2']:.1f}" if isinstance(c["I2"], float) else c["I2"],
                        f"{c['tau2']:.4f}" if isinstance(c["tau2"], float) else c["tau2"],
                        c.get("note", "")]
            else:
                row += ["", "", "", "", "", c.get("note", "")]
            w.writerow(row)


def write_interaction_csv(model):
    path = RESULTS_DIR / "interaction_terms.csv"
    with path.open("w", encoding="utf-8", newline="") as f:
        w = csv.writer(f)
        w.writerow(["term", "estimate", "se", "t", "p"])
        for c in model.get("coefficients", []):
            w.writerow([c["term"], f"{c['estimate']:.4f}", f"{c['se']:.4f}",
                        f"{c['t']:.3f}" if not np.isnan(c["t"]) else "",
                        f"{c['p']:.4f}" if not np.isnan(c["p"]) else ""])
        w.writerow([])
        w.writerow(["#", "k_obs", model.get("k_obs", ""), "", ""])
        w.writerow(["#", "tau2_used", f"{model.get('tau2_used', 0):.4f}", "", ""])
        wald = model.get("wald_chi2_interaction")
        wp = model.get("wald_p_interaction")
        w.writerow(["#", "wald_chi2_interaction",
                    f"{wald:.4f}" if wald is not None and not (isinstance(wald, float) and np.isnan(wald)) else "",
                    model.get("wald_df_interaction", ""),
                    f"{wp:.4f}" if wp is not None and not (isinstance(wp, float) and np.isnan(wp)) else ""])


def write_summary(cells, model):
    lines = [
        "# ICERI 2026 — Education-level x Discipline Cross-Tab",
        "",
        "## Cell-level k map (across all 5 traits, max k per cell)",
        "",
    ]
    counts = {}
    for (trait, lvl, disc), c in cells.items():
        counts[(lvl, disc)] = max(counts.get((lvl, disc), 0), c["k"])
    cell_disc = sorted({d for (_, d) in counts.keys()})
    cell_lvl = sorted({l for (l, _) in counts.keys()})
    header = "| level \\ discipline | " + " | ".join(cell_disc) + " |"
    lines.append(header)
    lines.append("|" + "---|" * (len(cell_disc) + 1))
    for lvl in cell_lvl:
        row = [f"**{lvl}**"]
        for d in cell_disc:
            row.append(str(counts.get((lvl, d), 0)))
        lines.append("| " + " | ".join(row) + " |")
    lines.append("")
    lines.append("## Pooled effects per cell (k>=2 only)")
    lines.append("")
    lines.append("| Trait | Level | Discipline | k | r [95% CI] |")
    lines.append("|-------|-------|------------|---|------------|")
    for key in sorted(cells.keys()):
        c = cells[key]
        if c["k"] >= 2 and isinstance(c.get("r_pooled"), float):
            lines.append(
                f"| {c['trait']} | {c['level']} | {c['discipline']} | {c['k']} | "
                f"{c['r_pooled']:.3f} [{c['r_ci_lo']:.3f}, {c['r_ci_hi']:.3f}] |"
            )
    lines.append("")
    lines.append("## Interaction model")
    lines.append("")
    if "coefficients" in model:
        lines.append(f"- k observations: {model['k_obs']}")
        lines.append(f"- tau^2 used in weights: {model['tau2_used']:.4f}")
        lines.append(f"- reference cell: trait={model['ref_trait']}, level={model['ref_level']}, discipline={model['ref_disc']}")
        wald = model.get("wald_chi2_interaction")
        wp = model.get("wald_p_interaction")
        df_int = model.get("wald_df_interaction")
        lines.append("")
        if wald is not None and not (isinstance(wald, float) and np.isnan(wald)):
            lines.append(
                f"**Joint Wald test on level x discipline interactions**: "
                f"chi^2({df_int}) = {wald:.3f}, p = {wp:.4f}"
            )
    else:
        lines.append(f"- {model.get('note', 'unavailable')}")
    lines.append("")
    lines.append("## Caveats")
    lines.append("- Most cells have k <= 1; the cross-tab is mostly a *coverage map* showing where evidence is dense vs sparse.")
    lines.append("- Education-level was pre-registered as a moderator in the parent preprint but not quantitatively executed (k constraint). This paper completes that registered analysis and adds the discipline crossing.")
    lines.append("- Discipline classification is the heuristic in `inputs/derive_studies_csv.py::classify_discipline`; single coder.")
    (RESULTS_DIR / "cross_tab_summary.md").write_text("\n".join(lines), encoding="utf-8")


def main():
    rows = load_studies()
    primary = primary_pool(rows)
    print(f"[ICERI] {len(primary)} primary-pool studies")

    cells = cross_tab_pools(primary)
    write_cross_tab_csv(cells)
    print(f"[ICERI] wrote cross_tab_pools.csv ({len(cells)} cells)")

    model = interaction_model(primary)
    write_interaction_csv(model)
    if "wald_chi2_interaction" in model and not (isinstance(model.get("wald_chi2_interaction"), float) and np.isnan(model.get("wald_chi2_interaction"))):
        print(
            f"[ICERI] interaction Wald: chi2({model['wald_df_interaction']})="
            f"{model['wald_chi2_interaction']:.3f}, p={model['wald_p_interaction']:.4f}"
        )

    write_summary(cells, model)
    print(f"[ICERI] wrote cross_tab_summary.md")


if __name__ == "__main__":
    main()
