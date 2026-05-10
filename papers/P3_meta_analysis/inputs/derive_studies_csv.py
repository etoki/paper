"""Derive papers/P3_meta_analysis/inputs/studies.csv from the master
extraction file metaanalysis/analysis/data_extraction_populated.csv.

This is the single canonical pivot point: every conference sub-paper in
this portfolio reads `studies.csv`. Re-run this script whenever the master
extraction changes.

Output columns
--------------
- study_id        e.g. "A-01"
- author_year     e.g. "Abe 2020"
- N               sample N analysed for the correlations
- modality        single letter S/A/B/M/U (see classify_modality below)
- country         country of the sample (raw)
- region          Asia / Europe / North_America / Other
- education_level Undergraduate / Graduate / K-12 / Mixed_UG_Grad
- discipline      STEM / Humanities / Psychology / Mixed (heuristic, see classify_discipline)
- era             pre-COVID / COVID / post-COVID / mixed
- inclusion_status include / include_with_caveat / include_COI / include_secondary / exclude*
- primary_achievement yes / partial / no / unknown
- r_O, r_C, r_E, r_A, r_N   Pearson correlations with achievement; blank if not extractable
- effect_source   r / beta_converted (Peterson-Brown) / mixed
"""
from __future__ import annotations

import csv
from pathlib import Path

SRC = Path(__file__).resolve().parents[3] / "metaanalysis" / "analysis" / "data_extraction_populated.csv"
DST = Path(__file__).resolve().parent / "studies.csv"


def classify_modality(row: dict) -> str:
    """Map the master extraction's free-text modality to one letter.

    S = synchronous (live videoconferencing-driven course)
    A = asynchronous (LMS-paced, MOOC-paced)
    B = blended (mix of online + face-to-face)
    M = mixed-online (within an online setting, both sync and async used)
    U = unspecified (the source paper does not say)

    Manual overrides are applied for studies whose `modality_subtype` is
    blank in the master extraction but where 2026-05-09 PDF re-reading
    pinned down the format (see preprint_audit.md section 4).
    """
    sid = (row.get("study_id") or "").strip()
    override = MODALITY_OVERRIDES.get(sid)
    if override is not None:
        return override

    sub = (row.get("modality_subtype") or "").strip().lower()
    mod = (row.get("modality") or "").strip().lower()
    platform = (row.get("platform_name") or "").strip().lower()

    if sub == "synchronous":
        return "S"
    if sub == "asynchronous":
        return "A"
    if sub == "mixed":
        return "M"
    if mod == "online_and_blended":
        return "B"
    # Heuristics for unspecified subtype
    if "mooc" in platform or "moodle" in platform or "studysapuri" in platform:
        return "A"
    return "U"


# Manual modality overrides — see papers/P3_meta_analysis/preprint_audit.md
# section 4 for evidence and PDF page citations.
MODALITY_OVERRIDES: dict[str, str] = {
    # A-15 Elvers 2003 — Web-based class with logged self-paced LMS access; no live online sessions.
    "A-15": "A",
    # A-23 Rodrigues 2024 — German university COVID home study; "partly asynchronous" + sync Zoom lectures.
    "A-23": "M",
    # A-26 Wang 2023 — China K-12 post-COVID; "teacher teaching" + platform usage suggest sync + async.
    "A-26": "M",
    # A-30 Kaspar 2023 — German university COVID-2021; cites synchronous Zoom + async materials.
    "A-30": "M",
}


def classify_region(row: dict) -> str:
    raw = (row.get("region") or "").strip()
    return raw if raw in ("Asia", "Europe", "North_America", "Other") else ""


def classify_discipline(row: dict) -> str:
    """Coarse STEM / Humanities / Psychology / Mixed split.

    Heuristics:
    - subject_domain matches Psychology -> Psychology
    - subject_domain matches IT, Engineering, CS, Health -> STEM
    - subject_domain matches Linguistics, Humanities, Language -> Humanities
    - All_5_subjects, blank, or other -> Mixed
    Notes column is consulted for clues if subject_domain is blank.
    """
    sd = (row.get("subject_domain") or "").strip().lower()
    notes = (row.get("notes") or "").strip().lower()
    blob = sd + " " + notes

    if "psychology" in sd:
        return "Psychology"
    if any(k in sd for k in ("it", "engineering", "computer", "health", "stem", "math")):
        return "STEM"
    if any(k in blob for k in ("linguistics", "language", "humanities", "history")):
        return "Humanities"
    return "Mixed"


def classify_era(row: dict) -> str:
    raw = (row.get("era") or "").strip()
    if raw.startswith("pre"):
        return "pre-COVID"
    if raw == "COVID":
        return "COVID"
    if raw.startswith("post"):
        return "post-COVID"
    if raw == "Mixed_3era":
        return "mixed"
    return ""


def get_r(row: dict, trait: str) -> tuple[str, str]:
    """Return (r_value, source) for a trait. Source is 'r' / 'beta_converted' / ''.

    Mirrors metaanalysis/analysis/pool.py extract_effect_for_trait().
    """
    r_str = (row.get(f"r_{trait}_outcome") or "").strip()
    if r_str:
        try:
            r = float(r_str)
            if abs(r) <= 0.99:
                return (f"{r:.4f}", "r")
        except ValueError:
            pass
    b_str = (row.get(f"beta_{trait}") or "").strip()
    if b_str:
        try:
            b = float(b_str)
            if abs(b) <= 0.99:
                r_approx = b + 0.05 if b >= 0 else b - 0.05
                r_approx = max(-0.99, min(0.99, r_approx))
                return (f"{r_approx:.4f}", "beta_converted")
        except ValueError:
            pass
    return ("", "")


def derive_row(row: dict) -> dict:
    fa = (row.get("first_author") or "").strip()
    yr = (row.get("year") or "").strip()
    n = (row.get("n_for_correlations") or "").strip() or (row.get("n_analyzed") or "").strip()

    out = {
        "study_id":          row.get("study_id", "").strip(),
        "author_year":       f"{fa} {yr}".strip(),
        "N":                 n,
        "modality":          classify_modality(row),
        "country":           (row.get("country") or "").strip(),
        "region":            classify_region(row),
        "education_level":   (row.get("education_level") or "").strip(),
        "discipline":        classify_discipline(row),
        "era":               classify_era(row),
        "inclusion_status":  (row.get("inclusion_status") or "").strip(),
        "primary_achievement": (row.get("primary_achievement") or "").strip(),
    }
    sources = []
    for trait in ("O", "C", "E", "A", "N"):
        r_str, src = get_r(row, trait)
        out[f"r_{trait}"] = r_str
        if src:
            sources.append(src)
    if not sources:
        out["effect_source"] = ""
    elif all(s == "r" for s in sources):
        out["effect_source"] = "r"
    elif all(s == "beta_converted" for s in sources):
        out["effect_source"] = "beta_converted"
    else:
        out["effect_source"] = "mixed"
    return out


def main() -> None:
    if not SRC.exists():
        raise SystemExit(f"Source CSV missing: {SRC}")

    with SRC.open("r", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))

    derived = [derive_row(r) for r in rows]

    fieldnames = [
        "study_id", "author_year", "N", "modality",
        "country", "region", "education_level", "discipline", "era",
        "inclusion_status", "primary_achievement",
        "r_O", "r_C", "r_E", "r_A", "r_N", "effect_source",
    ]
    with DST.open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        for d in derived:
            w.writerow(d)

    n_primary = sum(
        1 for d in derived
        if d["inclusion_status"] in ("include", "include_with_caveat", "include_COI")
        and d["primary_achievement"] in ("yes", "partial")
    )
    print(f"Wrote {DST}: {len(derived)} rows; {n_primary} primary-pool studies.")


if __name__ == "__main__":
    main()
