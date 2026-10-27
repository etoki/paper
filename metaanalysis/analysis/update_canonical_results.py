"""Regenerate sections 6-11 of CANONICAL_RESULTS.md from the
pipeline (pool.py, sensitivity.py) so the reference snapshot cannot drift
from the code. Sections 1-5 and 12 are left as they are.

Usage:
    python metaanalysis/analysis/update_canonical_results.py
"""
import re
import sys
from datetime import date
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

import pool  # noqa: E402

NAMES = {"O": "Openness", "C": "Conscientiousness", "E": "Extraversion",
         "A": "Agreeableness", "N": "Neuroticism"}


def ci(r):
    return f"[{r['r_ci_lo']:.3f}, {r['r_ci_hi']:.3f}]"


def section6():
    P = pool.run_pooling()
    out = ["## 6. Primary pooled random-effects meta-analysis (REML + HKSJ)", "",
           "τ² is the exact REML estimate (matches PyMARE); β is converted with "
           "Peterson & Brown's r = .98β + .05λ; studies in pool.POOL_EXCLUSIONS "
           f"({', '.join(pool.POOL_EXCLUSIONS)}) are left out of the pool.", "",
           "| Trait | k | N_total | r_pooled | 95% CI | 95% PI | I² | τ² | Q(df), p |",
           "|-------|---|---------|----------|--------|--------|----|----|---------|"]
    for t in pool.TRAITS:
        r = P[t]
        out.append(f"| {t} | {r['k']} | {r['N_total']} | {r['r_pooled']:.3f} | {ci(r)} | "
                   f"[{r['r_pi_lo']:.3f}, {r['r_pi_hi']:.3f}] | {r['I2']:.1f}% | "
                   f"{r['tau2']:.4f} | {r['Q']:.2f}({r['df']}), p={r['p_Q']:.4f} |")
    return "\n".join(out) + "\n\n"


def section7():
    P = pool.run_pooling()
    out = ["## 7. Contributing studies per trait pool", ""]
    for t in pool.TRAITS:
        out.append(f"### Trait {t} (k = {P[t]['k']})")
        out += [f"- {label}" for label in P[t]["labels"]]
        out.append("")
    return "\n".join(out) + "\n"


def section8_9():
    M = pool.run_moderator_analyses()
    out = ["## 8. Moderator analyses", "",
           "Test: Knapp–Hartung mixed-effects meta-regression over the levels with "
           "k ≥ 2 (REML residual τ²), F(df1, df2); p_Holm is Holm-corrected within "
           "each trait across the three moderators.", ""]
    sig = []
    for mod in M:
        out += [f"### Moderator: {mod}", "",
                "| Trait | Level | k | N | r [95% CI] | I² | F(df1, df2), p, p_Holm |",
                "|-------|-------|---|---|------------|----|------------------------|"]
        for t in pool.TRAITS:
            res = M[mod][t]
            b = res["_between"]
            fs = (f"{b['F']:.2f}({b['df1']}, {b['df2']}), p={b['p']:.4f}, "
                  f"p_Holm={b['p_holm']:.4f}" if b["F"] is not None else "—")
            for lv, sub in res.items():
                if lv == "_between":
                    continue
                if "r_pooled" in sub:
                    out.append(f"| {t} | {lv} | {sub['k']} | {sub['N_total']} | "
                               f"{sub['r_pooled']:.3f} {ci(sub)} | {sub['I2']:.1f}% | {fs} |")
                else:
                    out.append(f"| {t} | {lv} | {sub['k']} | — | (k<2) | — | — |")
            if b["p"] is not None and b["p"] < .05:
                sig.append((mod, t, b, res))
        out.append("")
    out += ["## 9. Significant moderator effects (p < .05, uncorrected)", ""]
    for i, (mod, t, b, res) in enumerate(sig, 1):
        out.append(f"{i}. **{NAMES[t]} × {mod}** — F({b['df1']}, {b['df2']}) = "
                   f"{b['F']:.2f}, p = {b['p']:.3f}, Holm p = {b['p_holm']:.3f}")
        for lv, sub in res.items():
            if lv != "_between" and "r_pooled" in sub:
                out.append(f"   - {lv} (k={sub['k']}): r = {sub['r_pooled']:.3f} {ci(sub)}")
        out.append("")
    out += ["Every contrast has k = 2 in its smaller subgroup; treat them as exploratory.", "", ""]
    return "\n".join(out)


def table_block(text, title):
    m = re.search(r"## " + re.escape(title) + r".*?\n\n(?:.*?\n\n)?(\|.*?)\n\n", text, re.S)
    return m.group(1)


def section10():
    sens = (HERE / "sensitivity_results.md").read_text(encoding="utf-8")
    out = "## 10. Sensitivity analyses\n\n"
    for title, label in [
        ("Exclude Author's Own Study", "Exclude Author's own study (A-25 Tokiwa, COI)"),
        ("Exclude Peterson-Brown", "Exclude β-converted (A-28 Yu, A-30 Kaspar)"),
        ("Exclude Low-Quality", "Exclude RoB < 5"),
        ("Exclude Studies with Analytic N", "Exclude analytic N < 50 (A-15 Elvers)"),
        ("DerSimonian-Laird Estimator", "DerSimonian–Laird τ² instead of REML"),
        ("Include A-37 Zheng", "Include A-37 Zheng (|r| < .10 entered as r = 0)"),
    ]:
        out += f"### {label}\n\n{table_block(sens, title)}\n\n"
    return out


def section11():
    summ = (HERE / "pooling_summary.md").read_text(encoding="utf-8")
    eg = re.search(r"### Egger's Regression Test\n\n(\|.*?)\n\n", summ, re.S).group(1)
    tf = re.search(r"### Duval & Tweedie \(2000\) Trim-and-Fill\n\n(\|.*?)\n\n", summ, re.S).group(1)
    return ("## 11. Publication bias\n\n### Egger's regression test\n\n" + eg +
            "\n\n### Duval & Tweedie trim-and-fill (L0; metafor::trimfill procedure)\n\n"
            + tf + "\n\n")


def main():
    path = HERE / "CANONICAL_RESULTS.md"
    c = path.read_text(encoding="utf-8")
    c = re.sub(r"## 6\. .*?(?=## 7\. )", lambda m: section6(), c, flags=re.S)
    c = re.sub(r"## 7\. .*?(?=## 8\. )", lambda m: section7(), c, flags=re.S)
    c = re.sub(r"## 8\. .*?(?=## 10\. )", lambda m: section8_9(), c, flags=re.S)
    c = re.sub(r"## 10\. .*?(?=## 11\. )", lambda m: section10(), c, flags=re.S)
    c = re.sub(r"## 11\. .*?(?=## 12\. )", lambda m: section11(), c, flags=re.S)
    c = re.sub(r"\*\*Generated\*\*:.*\n",
               f"**Generated**: 2026-05-01; sections 6–11 regenerated "
               f"{date.today().isoformat()} by `update_canonical_results.py`\n", c, count=1)
    path.write_text(c, encoding="utf-8")
    print(f"Wrote {path}")


if __name__ == "__main__":
    main()
