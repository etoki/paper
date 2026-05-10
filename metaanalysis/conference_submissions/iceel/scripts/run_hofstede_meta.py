"""ICEEL 2026 — Hofstede cultural-dimensions moderator on the East-Asian
subset, plus a focused Japan synthesis.

Imports the proven pooling primitives from
`metaanalysis/analysis/pool.py` so the per-trait pooled estimates remain
identical to the parent preprint.

Usage:
    python metaanalysis/conference_submissions/iceel/scripts/run_hofstede_meta.py

Outputs (to metaanalysis/conference_submissions/iceel/results/):
    asia_subset_pools.csv          per-trait pooled r within the Asian subset
    hofstede_meta_regression.csv   coefficient per (trait, dimension) pair
    japan_synthesis.md             narrative table of the two Japan studies
    summary.md                     human-readable summary

Hofstede scores are encoded inline (Hofstede Insights canonical 6-D model
as of 2024). When a country lacks a published score the regression skips
the dimension for that observation.
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

INPUT_CSV = REPO_ROOT / "metaanalysis" / "conference_submissions" / "inputs" / "studies.csv"
RESULTS_DIR = REPO_ROOT / "metaanalysis" / "conference_submissions" / "iceel" / "results"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

TRAITS = ("O", "C", "E", "A", "N")

# Hofstede 6-D scores per country (Hofstede Insights, accessed 2024)
# PDI = Power Distance Index, IDV = Individualism, MAS = Masculinity,
# UAI = Uncertainty Avoidance, LTO = Long-Term Orientation, IND = Indulgence
HOFSTEDE = {
    "Japan":  {"PDI": 54, "IDV": 46, "MAS": 95, "UAI": 92, "LTO": 88, "IND": 42},
    "China":  {"PDI": 80, "IDV": 20, "MAS": 66, "UAI": 30, "LTO": 87, "IND": 24},
    "Taiwan": {"PDI": 58, "IDV": 17, "MAS": 45, "UAI": 69, "LTO": 93, "IND": 49},
    "Korea":  {"PDI": 60, "IDV": 18, "MAS": 39, "UAI": 85, "LTO": 100, "IND": 29},
    # Non-Asian countries included for context-aware narrative; not used in regression
    "US":     {"PDI": 40, "IDV": 91, "MAS": 62, "UAI": 46, "LTO": 26, "IND": 68},
    "UK":     {"PDI": 35, "IDV": 89, "MAS": 66, "UAI": 35, "LTO": 51, "IND": 69},
    "Germany":{"PDI": 35, "IDV": 67, "MAS": 66, "UAI": 65, "LTO": 83, "IND": 40},
    "Turkey": {"PDI": 66, "IDV": 37, "MAS": 45, "UAI": 85, "LTO": 46, "IND": 49},
    "Canada": {"PDI": 39, "IDV": 80, "MAS": 52, "UAI": 48, "LTO": 36, "IND": 68},
    "Israel": {"PDI": 13, "IDV": 54, "MAS": 47, "UAI": 81, "LTO": 38, "IND": None},
    "Iran":   {"PDI": 58, "IDV": 41, "MAS": 43, "UAI": 59, "LTO": 14, "IND": 40},
    "Tunisia":{"PDI": 70, "IDV": 40, "MAS": 40, "UAI": 75, "LTO": None, "IND": None},
    "Greece": {"PDI": 60, "IDV": 35, "MAS": 57, "UAI": 100,"LTO": 45, "IND": 50},
    "India":  {"PDI": 77, "IDV": 48, "MAS": 56, "UAI": 40, "LTO": 51, "IND": 26},
    "Pakistan":{"PDI": 55,"IDV": 14, "MAS": 50, "UAI": 70, "LTO": 50, "IND": 0},
    "Thailand":{"PDI": 64,"IDV": 20, "MAS": 34, "UAI": 64, "LTO": 32, "IND": 45},
}

DIMENSIONS = ("PDI", "IDV", "MAS", "UAI", "LTO", "IND")


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


def trait_observations(rows, trait, region_filter=None):
    out = []
    for r in rows:
        if region_filter and r.get("region") != region_filter:
            continue
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
        country = (r.get("country") or "").strip()
        # Normalise country aliases
        if country.startswith("Honduras"):
            country = "Spain"  # multi-country sample; use first
        out.append({
            **r, "_trait": trait, "_r": rho, "_n": n, "_z": z,
            "_v": parent.var_z(n), "_country": country,
        })
    return out


def asia_pools(primary):
    """Per-trait pooled r within the Asian subset."""
    out = {}
    for trait in TRAITS:
        obs = trait_observations(primary, trait, region_filter="Asia")
        if len(obs) >= 2:
            ys = [o["_z"] for o in obs]
            vs = [o["_v"] for o in obs]
            labels = [f"{o.get('study_id','')} {o.get('author_year','')}" for o in obs]
            res = parent.pool_random_effects(ys, vs, labels)
            res["N_total"] = sum(o["_n"] for o in obs)
            res["k"] = len(obs)
            out[trait] = res
        else:
            out[trait] = {"k": len(obs), "note": "k<2"}
    return out


def hofstede_meta_regression(primary):
    """Per (trait, Hofstede dimension): weighted-OLS on Fisher z scale.

    Each row is one Asian study x one trait. The dimension is centred at
    the corpus (Asian-subset) mean before regression.
    """
    out_rows = []
    for trait in TRAITS:
        obs = trait_observations(primary, trait, region_filter="Asia")
        if len(obs) < 2:
            continue

        # Estimate tau^2 for the trait pool to use as weight
        ys = np.array([o["_z"] for o in obs])
        vs = np.array([o["_v"] for o in obs])
        tau2 = parent.reml_tau2(ys, vs) if len(ys) >= 2 else 0.0

        for dim in DIMENSIONS:
            # Build x, y for this dimension
            xs, ys2, ws = [], [], []
            for o in obs:
                country = o["_country"]
                hd = HOFSTEDE.get(country, {})
                v = hd.get(dim)
                if v is None:
                    continue
                xs.append(float(v))
                ys2.append(o["_z"])
                ws.append(1.0 / (o["_v"] + tau2))
            if len(xs) < 2:
                out_rows.append({
                    "trait": trait, "dimension": dim, "k": len(xs),
                    "intercept": "", "slope": "", "se_slope": "",
                    "t_slope": "", "p_slope": "",
                    "note": "k<2",
                })
                continue
            x = np.array(xs)
            y = np.array(ys2)
            w = np.array(ws)
            # Centre x at mean
            x_c = x - x.mean()
            X = np.column_stack([np.ones(len(x_c)), x_c])
            Wsqrt = np.sqrt(w)
            Xw = X * Wsqrt[:, None]
            yw = y * Wsqrt
            beta, *_ = np.linalg.lstsq(Xw, yw, rcond=None)
            df_resid = len(x_c) - 2
            # k < 4 -> df_resid <= 0 in this 2-parameter model.  Report the
            # slope as descriptive only and skip inference.
            if df_resid < 1 or x.std(ddof=1) == 0:
                out_rows.append({
                    "trait": trait, "dimension": dim, "k": len(xs),
                    "intercept": float(beta[0]),
                    "slope": float(beta[1]),
                    "se_slope": "",
                    "t_slope": "", "p_slope": "",
                    "note": "descriptive only (df_resid<1)",
                })
                continue
            resid = yw - Xw @ beta
            sigma2 = float(np.sum(resid ** 2) / df_resid)
            try:
                cov = sigma2 * np.linalg.inv(Xw.T @ Xw)
            except np.linalg.LinAlgError:
                cov = sigma2 * np.linalg.pinv(Xw.T @ Xw)
            se = np.sqrt(np.diag(cov))
            t_slope = beta[1] / se[1] if se[1] > 0 else float("nan")
            p_slope = (2 * (1 - stats.t.cdf(abs(t_slope), df=df_resid))
                        if not np.isnan(t_slope) else float("nan"))
            out_rows.append({
                "trait": trait, "dimension": dim, "k": len(xs),
                "intercept": float(beta[0]),
                "slope": float(beta[1]),
                "se_slope": float(se[1]),
                "t_slope": float(t_slope),
                "p_slope": float(p_slope),
                "note": "exploratory; k tiny",
            })
    return out_rows


def japan_synthesis(primary):
    """Narrative table of Japan-based primary-pool studies."""
    rows = [r for r in primary if (r.get("country") or "").strip() == "Japan"]
    return rows


def write_asia_pools(pools):
    path = RESULTS_DIR / "asia_subset_pools.csv"
    with path.open("w", encoding="utf-8", newline="") as f:
        w = csv.writer(f)
        w.writerow(["trait", "k", "N_total", "r_pooled", "r_ci_lo", "r_ci_hi", "I2", "tau2", "note"])
        for t in TRAITS:
            r = pools.get(t, {})
            if "r_pooled" in r:
                w.writerow([t, r["k"], r.get("N_total", ""),
                            f"{r['r_pooled']:.4f}",
                            f"{r['r_ci_lo']:.4f}", f"{r['r_ci_hi']:.4f}",
                            f"{r['I2']:.1f}", f"{r['tau2']:.4f}", ""])
            else:
                w.writerow([t, r.get("k", 0), "", "", "", "", "", "", r.get("note", "")])


def _fmt(v, fmt):
    return format(v, fmt) if isinstance(v, float) and not np.isnan(v) else (v if isinstance(v, str) else "")


def write_hofstede_csv(rows):
    path = RESULTS_DIR / "hofstede_meta_regression.csv"
    with path.open("w", encoding="utf-8", newline="") as f:
        w = csv.writer(f)
        w.writerow(["trait", "dimension", "k", "intercept", "slope", "se_slope", "t_slope", "p_slope", "note"])
        for r in rows:
            w.writerow([
                r["trait"], r["dimension"], r["k"],
                _fmt(r["intercept"], ".4f"),
                _fmt(r["slope"], ".4f"),
                _fmt(r["se_slope"], ".4f"),
                _fmt(r["t_slope"], ".3f"),
                _fmt(r["p_slope"], ".4f"),
                r["note"],
            ])


def write_japan_synthesis_md(rows):
    path = RESULTS_DIR / "japan_synthesis.md"
    lines = [
        "# Japan studies — narrative comparison",
        "",
        "Two Japan-based primary-pool studies appear in the corpus. They sample different education levels and instruments.",
        "",
        "| study_id | author_year | N | education | modality | instrument | r_O | r_C | r_E | r_A | r_N |",
        "|---|---|---|---|---|---|---|---|---|---|---|",
    ]
    for r in rows:
        if (r.get("country") or "") != "Japan":
            continue
        instr = "BFI-2-J" if r.get("study_id") == "A-25" else "TIPI-J" if r.get("study_id") == "A-31" else "?"
        lines.append("| " + " | ".join([
            r.get("study_id", ""),
            r.get("author_year", ""),
            r.get("N", ""),
            r.get("education_level", ""),
            r.get("modality", ""),
            instr,
            r.get("r_O", "") or "—",
            r.get("r_C", "") or "—",
            r.get("r_E", "") or "—",
            r.get("r_A", "") or "—",
            r.get("r_N", "") or "—",
        ]) + " |")
    lines.append("")
    lines.append("Discussion notes:")
    lines.append("- Both studies are asynchronous (StudySapuri / Moodle).")
    lines.append("- Instrument heterogeneity is severe: BFI-2-J has 60 items; TIPI-J has 10. This single-item-per-trait design is a known threat to inter-study comparability.")
    lines.append("- A-25 Tokiwa is K-12 (highschool year 3); A-31 Rivers is undergraduate. Pooling within Japan is not justifiable on k = 2 alone.")
    path.write_text("\n".join(lines), encoding="utf-8")


def write_summary(asia, hof_rows, japan_rows):
    lines = [
        "# ICEEL 2026 — Hofstede x East Asia synthesis",
        "",
        "## Asian-subset pooled correlations",
        "",
        "| Trait | k | N | r [95% CI] | I^2 | tau^2 |",
        "|-------|---|---|-----------|------|-------|",
    ]
    for t in TRAITS:
        r = asia.get(t, {})
        if "r_pooled" in r:
            lines.append(
                f"| {t} | {r['k']} | {r.get('N_total','')} | "
                f"{r['r_pooled']:.3f} [{r['r_ci_lo']:.3f}, {r['r_ci_hi']:.3f}] | "
                f"{r['I2']:.1f}% | {r['tau2']:.4f} |"
            )
        else:
            lines.append(f"| {t} | {r.get('k',0)} | — | — | — | — |")
    lines.append("")
    lines.append("## Hofstede single-dimension meta-regressions (per trait)")
    lines.append("")
    lines.append("Each cell is a weighted-OLS on Fisher z, slope coefficient + p-value.")
    lines.append("All numbers are exploratory; k <= 4 within Asia means the regression is severely underpowered.")
    lines.append("")
    lines.append("| Trait | Dimension | k | slope | SE | t | p | note |")
    lines.append("|-------|-----------|---|------:|---:|---:|---:|------|")
    for r in hof_rows:
        slope = r["slope"]
        if isinstance(slope, float):
            se_s = f"{r['se_slope']:.4f}" if isinstance(r['se_slope'], float) else "—"
            t_s = f"{r['t_slope']:+.2f}" if isinstance(r['t_slope'], float) and not np.isnan(r['t_slope']) else "—"
            p_s = f"{r['p_slope']:.3f}" if isinstance(r['p_slope'], float) and not np.isnan(r['p_slope']) else "—"
            lines.append(
                f"| {r['trait']} | {r['dimension']} | {r['k']} | "
                f"{slope:+.4f} | {se_s} | {t_s} | {p_s} | {r['note']} |"
            )
        else:
            lines.append(f"| {r['trait']} | {r['dimension']} | {r['k']} | — | — | — | — | {r['note']} |")
    lines.append("")
    lines.append("## Japan focus (k = 2)")
    lines.append(f"")
    for r in japan_rows:
        if (r.get("country") or "") != "Japan":
            continue
        lines.append(f"- **{r.get('study_id')} {r.get('author_year')}**: N={r.get('N')}, modality={r.get('modality')}, education={r.get('education_level')}")
    lines.append("")
    lines.append("See `japan_synthesis.md` for the side-by-side comparison.")
    lines.append("")
    lines.append("## Caveats")
    lines.append("- Asian primary-pool k is **2 per trait** (A-28 Yu China; A-31 Rivers Japan). Two further Asian studies in the qualitative synthesis (A-25 Tokiwa, A-26 Wang) do not contribute extractable Pearson r values.")
    lines.append("- A 2-parameter regression with k=2 has df_resid = 0; slopes are reported as **descriptive only** (no SE / t / p).")
    lines.append("- Country-level Hofstede scores are an *ecological* proxy; no individual-level cultural data are available in the corpus.")
    lines.append("- The Hofstede meta-regression in this paper is therefore a *coverage demonstration* — it shows what the within-Asia structure looks like with the present evidence base, not a confirmatory test of cultural-dimension effects.")
    (RESULTS_DIR / "summary.md").write_text("\n".join(lines), encoding="utf-8")


def main():
    rows = load_studies()
    primary = primary_pool(rows)
    print(f"[ICEEL] {len(primary)} primary-pool studies")
    asia = asia_pools(primary)
    write_asia_pools(asia)
    print(f"[ICEEL] wrote asia_subset_pools.csv")

    hof_rows = hofstede_meta_regression(primary)
    write_hofstede_csv(hof_rows)
    print(f"[ICEEL] wrote hofstede_meta_regression.csv ({len(hof_rows)} rows)")

    japan = japan_synthesis(primary)
    write_japan_synthesis_md(japan)
    print(f"[ICEEL] wrote japan_synthesis.md ({len(japan)} Japan studies)")

    write_summary(asia, hof_rows, japan)
    print(f"[ICEEL] wrote summary.md")


if __name__ == "__main__":
    main()
