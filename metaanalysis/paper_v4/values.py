"""Numbers for the v4 manuscript, computed from the analysis pipeline at
build time so that the text cannot drift from the results.

Usage in build_docx_v4.py:
    from values import V, TRAITS, TRAIT_NAMES
    f"r = {V['C.r']}, 95% CI {V['C.ci']}"

All statistics are APA-formatted strings (no leading zero for values
bounded by 1, U+2212 minus sign).
"""
from __future__ import annotations

import csv
import statistics
import sys
from collections import Counter
from pathlib import Path

ANALYSIS = Path(__file__).resolve().parents[1] / "analysis"
sys.path.insert(0, str(ANALYSIS))

import pool  # noqa: E402
import sensitivity  # noqa: E402

MINUS = "−"
TRAITS = ["C", "O", "E", "A", "N"]  # manuscript order
TRAIT_NAMES = {"O": "Openness", "C": "Conscientiousness", "E": "Extraversion",
               "A": "Agreeableness", "N": "Neuroticism"}
SENS_KEYS = ["exclude_tokiwa", "exclude_beta_converted", "exclude_low_quality",
             "exclude_small_n", "dl_estimator", "include_a37"]


def apa(x: float, d: int = 3) -> str:
    """APA number: no leading zero when |x| < 1, true minus sign."""
    s = f"{x:.{d}f}"
    if abs(x) < 1:
        s = s.replace("0.", ".", 1)
    return s.replace("-", MINUS)


def num(x: float, d: int = 2) -> str:
    return f"{x:.{d}f}".replace("-", MINUS)


def ci(lo: float, hi: float) -> str:
    return f"[{apa(lo)}, {apa(hi)}]"


def p_txt(p: float) -> str:
    """'p < .001' or 'p = .009'."""
    return "p < .001" if p < 0.001 else f"p = {apa(p)}"


def p_cell(p: float) -> str:
    return "< .001" if p < 0.001 else apa(p)


def signed(x: float) -> str:
    return ("+" if x >= 0 else "") + apa(x)


def _rows():
    with (ANALYSIS / "data_extraction_populated.csv").open(encoding="utf-8") as f:
        return list(csv.DictReader(f))


def build() -> dict:
    V: dict[str, object] = {}
    rows = _rows()
    extr = pool.load_extractions()

    # ---- pooled estimates --------------------------------------------------
    P = pool.run_pooling()
    for t in TRAITS:
        r = P[t]
        V[f"{t}.k"] = str(r["k"])
        V[f"{t}.N"] = f"{r['N_total']:,}"
        V[f"{t}.r"] = apa(r["r_pooled"])
        V[f"{t}.r_raw"] = r["r_pooled"]
        V[f"{t}.ci"] = ci(r["r_ci_lo"], r["r_ci_hi"])
        V[f"{t}.pi"] = ci(r["r_pi_lo"], r["r_pi_hi"])
        V[f"{t}.pi_lo_raw"] = r["r_pi_lo"]
        V[f"{t}.pi_hi_raw"] = r["r_pi_hi"]
        V[f"{t}.I2"] = f"{r['I2']:.1f}%"
        V[f"{t}.tau2"] = apa(r["tau2"])
        V[f"{t}.tau"] = apa(r["tau"])
        V[f"{t}.Q"] = f"Q({r['df']}) = {r['Q']:.2f}, {p_txt(r['p_Q'])}"
        V[f"{t}.Qcell"] = f"{r['Q']:.2f}({r['df']}), {p_cell(r['p_Q'])}"

    # ---- moderators ----------------------------------------------------------
    M = pool.run_moderator_analyses()
    for mod, by_trait in M.items():
        for t in TRAITS:
            res = by_trait[t]
            b = res["_between"]
            key = f"{mod}.{t}"
            if b["F"] is not None:
                V[f"{key}.F"] = f"F({b['df1']}, {b['df2']}) = {b['F']:.2f}"
                V[f"{key}.p"] = p_txt(b["p"])
                V[f"{key}.p_raw"] = b["p"]
                V[f"{key}.pholm"] = p_txt(b["p_holm"]).replace("p ", "Holm-adjusted p ")
                V[f"{key}.pholm_raw"] = b["p_holm"]
                V[f"{key}.cell"] = (f"{b['F']:.2f}({b['df1']}, {b['df2']}), "
                                    f"{p_cell(b['p'])}; {p_cell(b['p_holm'])}")
            for lv, sub in res.items():
                if lv == "_between":
                    continue
                lk = f"{key}.{lv}"
                V[f"{lk}.k"] = str(sub["k"])
                if "r_pooled" in sub:
                    V[f"{lk}.r"] = apa(sub["r_pooled"])
                    V[f"{lk}.ci"] = ci(sub["r_ci_lo"], sub["r_ci_hi"])
    V["_moderators"] = M

    # ---- sensitivity -----------------------------------------------------------
    S = sensitivity.run_sensitivity(extr)
    for key in SENS_KEYS:
        for t in TRAITS:
            s = S[key].get(t)
            prim = S["primary"][t]
            V[f"sens.{key}.{t}.k"] = str(s["k"])
            V[f"sens.{key}.{t}.r"] = apa(s["r_pooled"])
            V[f"sens.{key}.{t}.ci"] = ci(s["r_ci_lo"], s["r_ci_hi"])
            V[f"sens.{key}.{t}.d"] = signed(s["r_pooled"] - prim["r_pooled"])
            V[f"sens.{key}.{t}.d_raw"] = s["r_pooled"] - prim["r_pooled"]
    for t in TRAITS:
        loo = S["leave_one_out"][t]
        prim = S["primary"][t]["r_pooled"]
        top = max(loo, key=lambda it: abs(it["r_pooled"] - prim))
        study = top["dropped"].split("(")[0].strip()
        V[f"loo.{t}.study"] = study
        V[f"loo.{t}.r"] = apa(top["r_pooled"])
        V[f"loo.{t}.d"] = signed(top["r_pooled"] - prim)
        V[f"loo.{t}.d_raw"] = top["r_pooled"] - prim
        V[f"loo.{t}.range"] = (f"{apa(min(it['r_pooled'] for it in loo))} to "
                               f"{apa(max(it['r_pooled'] for it in loo))}")

    # ---- publication bias ------------------------------------------------------
    B = pool.run_publication_bias()
    for t in TRAITS:
        e = B[t]["egger"]
        tf = B[t]["trim_fill"]
        V[f"egger.{t}"] = (f"intercept = {num(e['intercept'])}, SE = {num(e['se_intercept'])}, "
                           f"t({e['df']}) = {num(e['t'])}, {p_txt(e['p'])}")
        V[f"egger.{t}.short"] = f"intercept = {num(e['intercept'])}, {p_txt(e['p'])}"
        V[f"egger.{t}.p_raw"] = e["p"]
        V[f"tf.{t}.k0"] = tf["k_imputed"]
        V[f"tf.{t}.side"] = tf["side"]
        V[f"tf.{t}.r"] = apa(tf["r_adj"])
        V[f"tf.{t}.ci"] = ci(tf["r_adj_ci_lo"], tf["r_adj_ci_hi"])
        V[f"tf.{t}.d"] = signed(tf["r_adj"] - tf["r_orig"])

    # ---- study-level descriptives --------------------------------------------
    retained_status = ("include", "include_with_caveat", "include_COI",
                       "include_secondary", "exclude_from_primary")
    ret = [r for r in rows if r["inclusion_status"] in retained_status]
    excl = [r for r in rows if r["inclusion_status"] not in retained_status]
    pool_ids = sorted({r["study_id"] for r in extr for t in TRAITS
                       if pool.extract_effect_for_trait(r, t)})
    beta_ids = sorted({r["study_id"] for r in extr for t in TRAITS
                       if (pool.extract_effect_for_trait(r, t) or (0, 0, ""))[2] == "beta_converted"})
    V["n_catalogued"] = len(rows)
    V["n_retained"] = len(ret)
    V["n_excluded_catalogued"] = len(excl)
    V["excluded_catalogued"] = excl
    V["n_pool"] = len(pool_ids)
    V["pool_ids"] = pool_ids
    V["n_beta"] = len(beta_ids)
    V["beta_ids"] = beta_ids
    V["n_direct"] = len(pool_ids) - len(beta_ids)

    def n_of(r):
        return int(float(r.get("n_analyzed") or r.get("n_total")))

    Ns = [n_of(r) for r in ret]
    V["ret.N_min"] = f"{min(Ns):,}"
    V["ret.N_max"] = f"{max(Ns):,}"
    V["ret.N_median"] = f"{statistics.median(Ns):,.0f}"
    V["ret.N_min_study"] = [r for r in ret if n_of(r) == min(Ns)][0]
    V["ret.N_max_study"] = [r for r in ret if n_of(r) == max(Ns)][0]
    yrs = [int(r["year"]) for r in ret]
    V["ret.year_min"], V["ret.year_max"] = min(yrs), max(yrs)
    for field in ("education_level", "region", "era", "modality"):
        V[f"ret.{field}"] = Counter(r[field].strip() for r in ret)
    rob = [int(r["risk_of_bias_score"]) for r in ret]
    V["ret.rob_mean"] = f"{statistics.mean(rob):.2f}"
    V["ret.rob_sd"] = f"{statistics.stdev(rob):.2f}"
    V["ret.rob_min"], V["ret.rob_max"] = min(rob), max(rob)
    V["ret.rob_ge5"] = sum(x >= 5 for x in rob)
    V["ret.rob_lt5"] = sum(x < 5 for x in rob)
    V["ret.rob_lt5_ids"] = [r["study_id"] for r in ret if int(r["risk_of_bias_score"]) < 5]
    pool_rob = [int(r["risk_of_bias_score"]) for r in rows if r["study_id"] in pool_ids]
    V["pool.rob_mean"] = f"{statistics.mean(pool_rob):.2f}"
    for t in TRAITS:
        ids = {r["study_id"] for r in extr if pool.extract_effect_for_trait(r, t)}
        V[f"{t}.rob_mean"] = f"{statistics.mean(int(r['risk_of_bias_score']) for r in rows if r['study_id'] in ids):.2f}"
    return V


V = build()

if __name__ == "__main__":
    for k, v in V.items():
        if isinstance(v, (str, int, float)):
            print(f"{k}: {v}")
