"""
Hallucination check for harassment/ paper.

Modeled after metaanalysis/analysis/check_hallucinations.py.

Tasks:
  T1  Table 1 (Descriptives): means / SDs / N for 13 variables
  T2  Table 2 (Spearman): selected key correlations
  T3  Table 3-4 (Regression β / R² / ΔR² / F change): primary numbers
  T4  Table 5 (sex-stratified R²)
  T5  Table 6 (diagnostics) and Table 7 (VIF)
  T6  Cronbach's α: manuscript text vs alfa script outputs
  T7  Sample / counts (N, missing rate, age bins, gender)
  T8  Reference audit: in-text citations ⇔ reference list

Usage:
    python3 harassment/check_hallucinations.py [--task tN] [--verbose]
"""
from __future__ import annotations
import argparse
import re
import sys
import zipfile
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent
RES = ROOT / "res"
PAPER = ROOT / "paper"
ALFA = ROOT / "alfa"


# --------------------------------------------------------------------- #
# Manuscript text loader
# --------------------------------------------------------------------- #
def docx_text(path: Path) -> str:
    """Extract paragraph text from a .docx file via zip+regex."""
    with zipfile.ZipFile(path) as z:
        xml = z.read("word/document.xml").decode("utf-8")
    paragraphs = []
    for pmatch in re.finditer(r"<w:p[ >].*?</w:p>", xml, re.DOTALL):
        ptext = pmatch.group(0)
        texts = re.findall(r"<w:t[^>]*>([^<]*)</w:t>", ptext)
        line = "".join(texts).strip()
        if line:
            paragraphs.append(line)
    return "\n".join(paragraphs)


def normalize(s: str) -> str:
    """Normalise text for substring matching."""
    s = s.replace("­", "").replace("‐", "-").replace("‑", "-")
    s = s.replace("‒", "-").replace("–", "-").replace("—", "-")
    s = s.replace("−", "-")  # minus sign
    s = s.replace("&#8722;", "-").replace("&minus;", "-")
    s = s.replace("&lt;", "<").replace("&gt;", ">").replace("&amp;", "&")
    s = s.replace("\xa0", " ")
    return s


# --------------------------------------------------------------------- #
# Findings tracker
# --------------------------------------------------------------------- #
class Findings:
    def __init__(self) -> None:
        self.passes: list[tuple[str, str]] = []
        self.fails: list[tuple[str, str, str]] = []  # (task, claim, why)

    def passed(self, task: str, msg: str) -> None:
        self.passes.append((task, msg))

    def failed(self, task: str, claim: str, why: str) -> None:
        self.fails.append((task, claim, why))

    def report(self, verbose: bool = False) -> int:
        if verbose:
            for task, msg in self.passes:
                print(f"  [PASS] {task}: {msg}")
        for task, claim, why in self.fails:
            print(f"  [FAIL] {task}: {claim}")
            print(f"         ↳ {why}")
        print()
        print(f"Summary: {len(self.passes)} pass, {len(self.fails)} fail")
        return 0 if not self.fails else 1


# --------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------- #
def expect_substring(
    f: Findings, task: str, text: str, needle: str, *, hint: str = ""
) -> None:
    """Pass if `needle` appears in `text` (after normalisation)."""
    n_text = normalize(text)
    n_needle = normalize(needle)
    if n_needle in n_text:
        f.passed(task, f"found: {needle!r}")
    else:
        # Allow some flexibility for tags like <em>
        n_text_strip = re.sub(r"\s+", " ", n_text)
        n_needle_strip = re.sub(r"\s+", " ", n_needle)
        if n_needle_strip in n_text_strip:
            f.passed(task, f"found (whitespace-flex): {needle!r}")
            return
        f.failed(task, f"missing string: {needle!r}", hint or "not present in text")


def expect_one_of(
    f: Findings, task: str, text: str, alternatives: list[str], *, hint: str = ""
) -> None:
    n_text = normalize(text)
    n_text = re.sub(r"\s+", " ", n_text)
    for alt in alternatives:
        if re.sub(r"\s+", " ", normalize(alt)) in n_text:
            f.passed(task, f"found one of {alternatives!r} → {alt!r}")
            return
    f.failed(task, f"none of {alternatives!r} present", hint or "no match")


# --------------------------------------------------------------------- #
# Tasks
# --------------------------------------------------------------------- #
def task_t1_descriptives(text: str, f: Findings) -> None:
    """Means and SDs in the Results / Table 1."""
    desc = pd.read_csv(RES / "descriptive_statistics.csv", index_col=0)

    cases = [
        # (label, manuscript_value_format)
        ("hexaco_HH", "3.25", "0.59"),
        ("hexaco_E", "3.30", "0.78"),
        ("hexaco_X", "2.70", "0.84"),
        ("hexaco_A", "2.88", "0.73"),
        ("hexaco_C", "3.21", "0.68"),
        ("hexaco_O", "3.46", "0.80"),
        ("Machiavellianism", "3.58", "0.66"),
        ("Narcissism", "2.40", "0.72"),
        ("Psychopathy", "2.50", "0.66"),
        ("power_harassment", "1.26", "0.32"),
        ("gender_harassment", "1.70", "0.70"),
    ]
    # Verify canonical (CSV) matches reported rounding
    for var, m, sd in cases:
        m_calc = round(float(desc.loc[var, "mean"]), 2)
        sd_calc = round(float(desc.loc[var, "std"]), 2)
        m_rep = float(m)
        sd_rep = float(sd)
        if m_calc != m_rep or sd_calc != sd_rep:
            f.failed(
                "T1",
                f"{var}: M={m_rep} SD={sd_rep}",
                f"computed M={m_calc} SD={sd_calc}",
            )
        else:
            f.passed("T1", f"{var}: M={m_rep} SD={sd_rep}")

    # Verify the explicit Results paragraph references each pair
    for _var, m, sd in cases:
        # text may render as "3.25 (0.59)" or "3.25(0.59)"
        if not (re.search(rf"{m}\s*\(\s*{sd}\s*\)", text)):
            # Some entries (e.g. "1.26 (0.32)") definitely should appear in Results
            pass

    # Sanity: power & gender harassment summary sentence
    expect_substring(f, "T1", text, "1.26 (0.32)", hint="power harassment M(SD)")
    expect_substring(f, "T1", text, "1.70 (0.70)", hint="gender harassment M(SD)")


def task_t2_correlations(text: str, f: Findings) -> None:
    """Selected Spearman correlations in Results / Table 2."""
    rho = pd.read_csv(RES / "spearman_rho.csv", index_col=0)

    pairs = [
        ("hexaco_HH", "Machiavellianism", -0.32, "***"),
        ("hexaco_HH", "Narcissism", -0.36, "***"),
        ("hexaco_HH", "Psychopathy", -0.38, "***"),
        ("hexaco_HH", "power_harassment", -0.23, "***"),
        ("hexaco_HH", "gender_harassment", -0.18, "***"),
        ("hexaco_A", "Machiavellianism", -0.27, "***"),
        ("hexaco_A", "Psychopathy", -0.36, "***"),
        ("hexaco_A", "power_harassment", -0.26, "***"),
        ("hexaco_X", "Narcissism", 0.56, "***"),
        ("hexaco_X", "Psychopathy", 0.16, "**"),
        ("hexaco_X", "gender_harassment", 0.11, "*"),
        ("power_harassment", "gender_harassment", 0.35, "***"),
    ]

    for v1, v2, expected, _ in pairs:
        actual = round(float(rho.loc[v1, v2]), 2)
        if actual != expected:
            f.failed(
                "T2",
                f"{v1} × {v2}: ρ={expected}",
                f"canonical ρ={actual}",
            )
        else:
            f.passed("T2", f"{v1} × {v2}: ρ={actual}")

    # In-text appearance for select claims
    in_text_claims = [
        "ρ = −.32",  # H-H × M
        "ρ = −.36",  # H-H × N (and A × Psy)
        "ρ = −.38",  # H-H × P
        "ρ = −.23",  # H-H × power
        "ρ = −.18",  # H-H × gender
        "ρ = .56",   # X × N
    ]
    for s in in_text_claims:
        expect_substring(f, "T2", text, s, hint="key correlation in Results")


def task_t3_regression(text: str, f: Findings) -> None:
    """Model A / Model B coefficients & ΔR² / F change."""
    fit = pd.read_csv(RES / "model_fit_incremental.csv")
    coefs = pd.read_csv(RES / "regression_coefficients_extended.csv")

    # ----- Model A (DT only) sanity ----- #
    A_power = coefs[
        (coefs["Dependent_Var"] == "power_harassment")
        & (coefs["Model"] == "A_controls+DT")
    ]
    A_gender = coefs[
        (coefs["Dependent_Var"] == "gender_harassment")
        & (coefs["Model"] == "A_controls+DT")
    ]

    def beta(table: pd.DataFrame, var: str) -> float:
        return round(float(table.loc[table["Variable"] == var, "Std_Beta"].iloc[0]), 3)

    # Manuscript: Power Model A — Psy β=.396, Mach=−.062 p=.342, Narc=−.005 p=.933
    cases_A = [
        ("power", A_power, "Psychopathy", 0.396),
        ("power", A_power, "Machiavellianism", -0.062),
        ("power", A_power, "Narcissism", -0.005),
        # Gender Model A — Mach −.138 p=.006, Narc .180 p<.001, Psy .165 p=.004
        ("gender", A_gender, "Machiavellianism", -0.138),
        ("gender", A_gender, "Narcissism", 0.180),
        ("gender", A_gender, "Psychopathy", 0.165),
    ]
    for outcome, tab, var, exp in cases_A:
        got = beta(tab, var)
        if got != exp:
            f.failed("T3", f"Model A {outcome} {var} β={exp}", f"canonical {got}")
        else:
            f.passed("T3", f"Model A {outcome} {var} β={got}")

    # ----- Model B coefficients ----- #
    B_power = coefs[
        (coefs["Dependent_Var"] == "power_harassment")
        & (coefs["Model"] == "B_controls+DT+HEXACO")
    ]
    B_gender = coefs[
        (coefs["Dependent_Var"] == "gender_harassment")
        & (coefs["Model"] == "B_controls+DT+HEXACO")
    ]

    cases_B = [
        # Power Model B
        ("power", B_power, "Psychopathy", 0.317),
        ("power", B_power, "hexaco_HH", -0.143),
        ("power", B_power, "hexaco_A", -0.108),
        # Gender Model B
        ("gender", B_gender, "Machiavellianism", -0.164),
        ("gender", B_gender, "Narcissism", 0.190),
        ("gender", B_gender, "Psychopathy", 0.154),
        ("gender", B_gender, "hexaco_HH", -0.230),
        ("gender", B_gender, "hexaco_O", -0.236),
        ("gender", B_gender, "gender", -0.312),
    ]
    for outcome, tab, var, exp in cases_B:
        got = beta(tab, var)
        if got != exp:
            f.failed("T3", f"Model B {outcome} {var} β={exp}", f"canonical {got}")
        else:
            f.passed("T3", f"Model B {outcome} {var} β={got}")

    # ----- Model fit ----- #
    p_row = fit[fit["Dependent_Var"] == "power_harassment"].iloc[0]
    g_row = fit[fit["Dependent_Var"] == "gender_harassment"].iloc[0]

    fit_cases = [
        ("power R²(A)", round(float(p_row["R2_A"]), 3), 0.166),
        ("power R²(B)", round(float(p_row["R2_B"]), 3), 0.198),
        ("power ΔR²", round(float(p_row["Delta_R2_B_minus_A"]), 3), 0.032),
        ("power F-change", round(float(p_row["Fchange_nonrobust"]), 2), 2.28),
        ("power p-change", round(float(p_row["pchange_nonrobust"]), 3), 0.036),
        ("power R²(B sens)", round(float(p_row["R2_B_sensitivity"]), 3), 0.221),
        ("power R²(C)", round(float(p_row["R2_with_interactions"]), 3), 0.218),
        ("power Power(inc)", round(float(p_row["Power_inc_HEXACO"]), 3), 0.803),
        ("power f²(inc)", round(float(p_row["f2_inc_HEXACO"]), 3), 0.040),
        ("gender R²(A)", round(float(g_row["R2_A"]), 3), 0.117),
        ("gender R²(B)", round(float(g_row["R2_B"]), 3), 0.213),
        ("gender ΔR²", round(float(g_row["Delta_R2_B_minus_A"]), 3), 0.096),
        ("gender F-change", round(float(g_row["Fchange_nonrobust"]), 2), 6.91),
        ("gender R²(B sens)", round(float(g_row["R2_B_sensitivity"]), 3), 0.203),
        ("gender R²(C)", round(float(g_row["R2_with_interactions"]), 3), 0.219),
        ("gender Power(inc)", round(float(g_row["Power_inc_HEXACO"]), 4), 0.9997),
        ("gender f²(inc)", round(float(g_row["f2_inc_HEXACO"]), 3), 0.122),
    ]
    for label, got, exp in fit_cases:
        if got != exp:
            f.failed("T3", f"{label} = {exp}", f"canonical {got}")
        else:
            f.passed("T3", f"{label} = {got}")

    # ----- Manuscript text claims ----- #
    text_claims = [
        # Power harassment paragraph in Results
        "(β = .396, p < .001)",
        ".166 to .198",
        "ΔR² = .032",
        "F change = 2.28",
        "p = .036",
        "(β = .317, p < .001)",
        "(β = −.143, p = .049)",
        "(β = −.108, p = .063)",
        # Gender harassment paragraph
        "(β = −.138, p = .006)",
        "(β = .180, p < .001)",
        "(β = .165, p = .004)",
        ".117 to .213",
        "ΔR² = .096",
        "F change = 6.91",
        "p < .001",
        "(β = −.230, p < .001)",
        "(β = −.236, p < .001)",
        "(β = −.312, p = .006",
        # Sensitivity
        "Cook’s distance > 4/n yielded R² B sensitivity of .221",
        "(power harassment) and .203 (gender harassment)",
        "max Cook’s D = 0.058/0.043",
        "exceedances = 23/15",
        # Interactions
        "R² with interactions of .218 (power) and .219 (gender)",
        # Sample size sensitivity (post hoc)
        "Cohen’s f² = 0.040 (ΔR² = 0.032) for power harassment",
        "achieved power = 0.803",
        "f² = 0.122 (ΔR² = 0.096)",
        "achieved power = 0.9997",
    ]
    for s in text_claims:
        expect_substring(f, "T3", text, s)


def task_t4_sex_stratified(text: str, f: Findings) -> None:
    """Sex-stratified R² (Table 5)."""
    sx = pd.read_csv(RES / "sex_stratified_R2.csv")
    cases = [
        ("power_harassment", 0, 133, 0.287, 0.215),
        ("power_harassment", 1, 220, 0.203, 0.157),
        ("gender_harassment", 0, 133, 0.283, 0.211),
        ("gender_harassment", 1, 220, 0.166, 0.118),
    ]
    for dv, gnd, n, r2, ar2 in cases:
        row = sx[(sx["Dependent_Var"] == dv) & (sx["gender"] == gnd)].iloc[0]
        if int(row["n"]) != n:
            f.failed("T4", f"{dv} g={gnd} n={n}", f"canonical {row['n']}")
        else:
            f.passed("T4", f"{dv} g={gnd} n={n}")
        gotr2 = round(float(row["R2"]), 3)
        if gotr2 != r2:
            f.failed("T4", f"{dv} g={gnd} R²={r2}", f"canonical {gotr2}")
        else:
            f.passed("T4", f"{dv} g={gnd} R²={gotr2}")

    # Manuscript paragraph: ".287 for participants coded as 0 (n = 133)"
    text_claims = [
        ".287 for participants coded as 0 (n = 133)",
        ".203 for participants coded as 1 (n = 220)",
        ".283 for participants coded as 0 (n = 133)",
        ".166 for participants coded as 1 (n = 220)",
    ]
    for s in text_claims:
        expect_substring(f, "T4", text, s)


def task_t5_diag_vif(text: str, f: Findings) -> None:
    """Table 6 (diagnostics) and Table 7 (VIF)."""
    fit = pd.read_csv(RES / "model_fit_incremental.csv")
    vif = pd.read_csv(RES / "vif_modelB.csv")

    p_row = fit[fit["Dependent_Var"] == "power_harassment"].iloc[0]
    g_row = fit[fit["Dependent_Var"] == "gender_harassment"].iloc[0]

    # DW
    cases = [
        ("DW power", round(float(p_row["Durbin_Watson_B"]), 2), 1.95),
        ("DW gender", round(float(g_row["Durbin_Watson_B"]), 2), 1.84),
        ("Max Cook D power", round(float(p_row["Max_CooksD_B"]), 2), 0.06),
        ("Max Cook D gender", round(float(g_row["Max_CooksD_B"]), 2), 0.04),
        ("n>4/n power", int(p_row["Num_CooksD_gt_4n_B"]), 23),
        ("n>4/n gender", int(g_row["Num_CooksD_gt_4n_B"]), 15),
    ]
    for label, got, exp in cases:
        if got != exp:
            f.failed("T5", f"{label} = {exp}", f"canonical {got}")
        else:
            f.passed("T5", f"{label} = {got}")

    # VIF: max should be 2.00 (Narcissism), top values N=2.00, X=1.70, H-H=1.57
    vif_p = vif[vif["Dependent_Var"] == "power_harassment"].set_index("Variable")
    cases_vif = [
        ("Narcissism", 2.00),
        ("hexaco_X", 1.70),
        ("hexaco_HH", 1.57),
        ("Psychopathy", 1.56),
        ("hexaco_A", 1.53),
        ("hexaco_E", 1.33),
        ("Machiavellianism", 1.31),
        ("hexaco_O", 1.31),
        ("age", 1.24),
        ("hexaco_C", 1.16),
    ]
    for var, exp in cases_vif:
        got = round(float(vif_p.loc[var, "VIF"]), 2)
        if got != exp:
            f.failed("T5", f"VIF {var} = {exp}", f"canonical {got}")
        else:
            f.passed("T5", f"VIF {var} = {got}")

    # Manuscript text
    text_claims = [
        "Durbin–Watson statistics were 1.95 (power) and 1.84 (gender)",
        "VIFs were modest (max 2.00 for N; mean 1.40)",
        "N = 2.00, X = 1.70, H–H = 1.57",
        "each Dark Triad variable 0.28%",
        "n = 353",
    ]
    for s in text_claims:
        expect_substring(f, "T5", text, s)


def task_t6_alphas(text: str, f: Findings) -> None:
    """Cronbach's α — manuscript text vs alfa script outputs."""
    alfa_csv = ALFA / "hexaco_alpha_results.csv"
    if not alfa_csv.exists():
        f.failed("T6", "alfa CSV missing", f"file not found: {alfa_csv}")
        return
    alpha_df = pd.read_csv(alfa_csv).set_index("domain")

    # Manuscript-reported α (post-fix: H-H corrected from .671 to .571)
    reported = {
        "Honesty-Humility": 0.571,
        "Emotionality": 0.830,
        "Extraversion": 0.621,
        "Agreeableness": 0.783,
        "Conscientiousness": 0.815,
        "Openness": 0.804,
    }
    for k, rep in reported.items():
        canon = round(float(alpha_df.loc[k, "alpha"]), 3)
        if canon != rep:
            f.failed(
                "T6",
                f"α({k}) reported = {rep:.3f}",
                f"alfa script canonical = {canon:.3f} (Δ = {rep - canon:+.3f})",
            )
        else:
            f.passed("T6", f"α({k}) = {rep:.3f}")

    # Manuscript wording check
    text_claims = [
        "α = .571 for Honesty",
        "α = .830 for Emotionality",
        "α = .621 for Extraversion",
        "α = .783 for Agreeableness",
        "α = .815 for Conscientiousness",
        "α = .804 for Openness",
        # SD3-J alphas (cannot verify without item-level computation)
        "α = .767 for Machiavellianism",
        "α = .778 for Narcissism",
        "α = .708 for Psychopathy",
        "α = .842 for the SD3-J total",
        # Power harassment subscale alphas
        "α = .862 for behaviors",
        "α = .730 for climate",
        "α = .760 for attitudes",
        "α = .880 for the total index",
        # Gender harassment
        "α = .876 for commission behaviors",
        "α = .812 for omission behaviors",
        "α = .901 for the total gender harassment index",
    ]
    for s in text_claims:
        expect_substring(f, "T6", text, s)


def task_t7_sample_counts(text: str, f: Findings) -> None:
    """Sample N, missing rates, demographics."""
    df = pd.read_csv(ROOT / "raw.csv")
    n_total = len(df)
    n_male = int((df["gender"] == 0).sum())
    n_female = int((df["gender"] == 1).sum())

    age_bins = {
        10: int((df["age"] == 10).sum()),
        20: int((df["age"] == 20).sum()),
        30: int((df["age"] == 30).sum()),
        40: int((df["age"] == 40).sum()),
        50: int((df["age"] == 50).sum()),
        60: int((df["age"] == 60).sum()),
    }
    n_50plus = age_bins[50] + age_bins[60]

    expectations = [
        ("final N", n_total, 354),
        ("males", n_male, 134),
        ("females", n_female, 220),
        ("teens (10s)", age_bins[10], 32),
        ("20s", age_bins[20], 100),
        ("30s", age_bins[30], 124),
        ("40s", age_bins[40], 70),
        ("≥ 50", n_50plus, 28),
    ]
    for label, got, exp in expectations:
        if got != exp:
            f.failed("T7", f"{label}: reported {exp}", f"canonical {got}")
        else:
            f.passed("T7", f"{label}: {got}")

    # Missing rate per DT trait
    miss_pct = round(float(df["Machiavellianism"].isna().mean()) * 100, 2)
    if miss_pct != 0.28:
        f.failed("T7", "DT missing 0.28%", f"canonical {miss_pct}%")
    else:
        f.passed("T7", "DT missing 0.28%")

    # Manuscript wording
    text_claims = [
        "380 individuals responded",
        "354 currently employed participants",
        "134 were men and 220 were women",
        "32 participants aged 18–19",
        "100 in their 20s",
        "124 in their 30s",
        "70 in their 40s",
        "28 aged 50 or older",
        "less than 0.3%",
    ]
    for s in text_claims:
        expect_substring(f, "T7", text, s)


def task_t8_references(text: str, f: Findings) -> None:
    """Audit: every (Author, YYYY) in body should appear in reference list.

    Heuristic only - we look for in-text citations (Author(s), YYYY)
    and verify each surname appears somewhere in the reference list section.
    """
    # Split into body and references
    if "References" not in text:
        f.failed("T8", "References section header missing", "no 'References' line")
        return
    head, _, refs = text.partition("\nReferences\n")
    body = head

    citations: set[tuple[str, str]] = set()

    # ---- (a) Parenthetical citations "(Author, YYYY)" ---- #
    cite_re = re.compile(r"\(([^()]*?\b\d{4}[a-z]?\b[^()]*?)\)")
    for m in cite_re.finditer(body):
        chunk = m.group(1)
        if not re.search(r"\b(19|20)\d{2}[a-z]?\b", chunk):
            continue
        for sub in re.split(r";", chunk):
            # Possibly "Fukui et al., 2017, 2018" — pick up *all* years.
            head_text = sub
            year_match = re.search(r"\b((?:19|20)\d{2})[a-z]?\b", head_text)
            if not year_match:
                continue
            head_only = head_text[: year_match.start()].strip().rstrip(",").strip()
            head_only = re.sub(r"^(e\.g\.,|see|cf\.|also)\s*", "", head_only, flags=re.I)
            if not head_only:
                continue
            sname = re.split(r",| &| and | et al", head_only)[0].strip()
            if not sname or not re.match(r"^[A-Za-zÀ-ÿ]", sname):
                continue
            for y_m in re.finditer(r"\b((?:19|20)\d{2})[a-z]?\b", head_text):
                citations.add((sname, y_m.group(1)))

    # ---- (b) Narrative citations "Author (YYYY)" / "Author et al. (YYYY)" ---- #
    narr_re = re.compile(
        r"([A-Z][a-zA-ZÀ-ÿ’\-]+(?:\s+(?:and|&)\s+[A-Z][a-zA-ZÀ-ÿ’\-]+)?(?:\s+et\s+al\.?)?)\s*\(((?:19|20)\d{2})[a-z]?\)"
    )
    for m in narr_re.finditer(body):
        sname = m.group(1).strip()
        # Skip if the leading word looks like a section header (e.g., "Cohen")
        sname = re.split(r",| and | & | et al", sname)[0].strip()
        if not sname:
            continue
        citations.add((sname, m.group(2)))

    # Extract reference-list surnames + years (rough)
    # Heuristic: publication year appears early in line, either as "(YYYY)"
    # or, occasionally, "YYYY." after a citation index (e.g. PubMed style).
    # We take ONLY the first such year to avoid grabbing DOI/URL year tokens.
    ref_idx = []
    for line in refs.splitlines():
        line = line.strip()
        if not line:
            continue
        # Try parenthesised year first
        m = re.search(r"\(((?:19|20)\d{2})[a-z]?\)", line)
        if not m:
            # Fallback: first 19xx/20xx in line (e.g. "Henrich J... 2010 Jun;")
            m = re.search(r"\b((?:19|20)\d{2})\b", line)
        if not m:
            continue
        year = m.group(1)
        surname_match = re.match(r"^([^,&\(]+)", line)
        if not surname_match:
            continue
        sname = surname_match.group(1).strip().rstrip(".")
        ref_idx.append((sname, year, line))

    # For each in-text citation, check at least one ref entry matches
    missing = []
    for sname, year in sorted(citations):
        # Allow surname to be short prefix of ref surname (e.g. "Lee" matches "Lee, K.")
        # And accept reasonable variants (Wakabayashi 2014 → Wakabayashi 2014)
        hit = False
        for r_sname, r_year, _line in ref_idx:
            # surname compare: case-insensitive prefix match (first word)
            r_first = r_sname.split()[0].rstrip(",.").lower()
            c_first = sname.split()[0].rstrip(",.").lower()
            # Strip diacritics for comparison
            import unicodedata

            def fold(s: str) -> str:
                return "".join(
                    c for c in unicodedata.normalize("NFD", s) if not unicodedata.combining(c)
                ).lower()

            if fold(r_first) == fold(c_first) and r_year == year:
                hit = True
                break
        if not hit:
            missing.append(f"{sname} ({year})")

    if missing:
        # de-duplicate
        unique = sorted(set(missing))
        f.failed(
            "T8",
            "in-text citations missing from reference list",
            f"{len(unique)} citation(s): " + ", ".join(unique[:30]) + ("…" if len(unique) > 30 else ""),
        )
    else:
        f.passed("T8", f"all {len(citations)} in-text citations resolve to a reference")

    # Manuscript reports presence of references — sanity
    if not refs.strip():
        f.failed("T8", "References section empty", "no entries after 'References'")

    # Spelling-mismatch sweep: surnames cited in body but spelled differently
    # in the reference list. We detect this by looking for body surnames that
    # do NOT match any ref surname exactly, but DO match if a single character
    # is changed (Levenshtein distance == 1).
    def lev1(a: str, b: str) -> bool:
        if abs(len(a) - len(b)) > 1:
            return False
        # substitution
        if len(a) == len(b):
            return sum(x != y for x, y in zip(a, b)) == 1
        # insertion / deletion
        if len(a) > len(b):
            a, b = b, a  # ensure a shorter
        i = j = diffs = 0
        while i < len(a) and j < len(b):
            if a[i] != b[j]:
                diffs += 1
                if diffs > 1:
                    return False
                j += 1
            else:
                i += 1
                j += 1
        return True

    import unicodedata as _u
    def fold(s: str) -> str:
        return "".join(
            c for c in _u.normalize("NFD", s) if not _u.combining(c)
        ).lower()

    cited_surnames = {fold(sn.split()[0]) for sn, _ in citations}
    ref_surnames = {fold(rs.split()[0].rstrip(",.")) for rs, _, _ in ref_idx}
    spelling_warnings: list[str] = []
    for csn in cited_surnames:
        if csn in ref_surnames or len(csn) < 4:
            continue
        for rsn in ref_surnames:
            if csn != rsn and lev1(csn, rsn):
                spelling_warnings.append(f"{csn!r} (body) ↔ {rsn!r} (ref list)")
                break
    if spelling_warnings:
        f.failed(
            "T8",
            "possible surname spelling mismatch",
            "; ".join(sorted(set(spelling_warnings))),
        )

    # Unused references: every ref-list entry should be cited at least once
    cited_pairs = {(fold(sn.split()[0]), yr) for sn, yr in citations}
    unused = []
    for rs, ry, line in ref_idx:
        first = fold(rs.split()[0].rstrip(",."))
        if (first, ry) not in cited_pairs:
            # Tolerate a, b suffixes (Pletzer 2019a/b) — count any year match
            if not any(c == first and y == ry for c, y in cited_pairs):
                unused.append(f"{rs.split()[0]} ({ry})")
    if unused:
        unique_unused = sorted(set(unused))
        f.failed(
            "T8",
            "reference-list entries with no in-text citation",
            f"{len(unique_unused)}: " + ", ".join(unique_unused[:20]),
        )


# --------------------------------------------------------------------- #
# Main
# --------------------------------------------------------------------- #
def task_t9_doi_audit(text: str, f: Findings) -> None:
    """Dry-run DOI audit on Manuscript_all.docx via the shared auditor."""
    import sys as _sys
    shared = ROOT.parent / "metaanalysis" / "analysis"
    if str(shared) not in _sys.path:
        _sys.path.insert(0, str(shared))
    import check_doi as _cd  # type: ignore
    refs_path = PAPER / "Manuscript_all.docx"
    records = _cd.load_references(refs_path)
    passed, warn, failed = _cd.dry_run(records)
    if failed:
        f.failed("T9", f"DOI dry-run: {failed} hard failure(s)",
                 "see stdout above")
    else:
        f.passed("T9",
                 f"DOI dry-run: {passed} passed / {warn} warn / "
                 f"{failed} failed (use harassment/check_doi.py --mode online "
                 f"for Crossref verification)")


ALL_TASKS = {
    "t1": ("Descriptives (Table 1)", task_t1_descriptives),
    "t2": ("Spearman correlations (Table 2)", task_t2_correlations),
    "t3": ("Regression (Tables 3 & 4)", task_t3_regression),
    "t4": ("Sex-stratified R² (Table 5)", task_t4_sex_stratified),
    "t5": ("Diagnostics & VIF (Tables 6 & 7)", task_t5_diag_vif),
    "t6": ("Cronbach's α (Methods)", task_t6_alphas),
    "t7": ("Sample / counts", task_t7_sample_counts),
    "t8": ("Reference list audit", task_t8_references),
    "t9": ("DOI audit (Crossref-backed; dry-run here)", task_t9_doi_audit),
}


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--task", default="all", help="run a single task (t1, …, t8)")
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args()

    body = docx_text(PAPER / "Manuscript_only.docx")
    body += "\n" + docx_text(PAPER / "Table.docx")
    # Title page also contains some declarations
    body += "\n" + docx_text(PAPER / "Title page with Declarations.docx")
    text = normalize(body)

    f = Findings()
    if args.task == "all":
        chosen = list(ALL_TASKS.keys())
    else:
        chosen = [t.strip().lower() for t in args.task.split(",")]

    for tid in chosen:
        if tid not in ALL_TASKS:
            print(f"unknown task: {tid}", file=sys.stderr)
            return 2
        title, fn = ALL_TASKS[tid]
        print(f"=== {tid.upper()} {title} ===")
        fn(text, f)
        print()

    return f.report(verbose=args.verbose)


if __name__ == "__main__":
    sys.exit(main())
