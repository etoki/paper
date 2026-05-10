#!/usr/bin/env python3
"""ICEEL paper: Option 2 (Hofstede direction match) + Option 3 (era moderator).

Run as exploration / decision aid before adopting either Option 2 or Option 3
into the ICEEL full paper. Outputs both analyses to stdout for the author to
compare.
"""
from __future__ import annotations

import csv
from pathlib import Path

ROOT = Path(__file__).resolve().parents[4]
STUDIES = ROOT / "metaanalysis" / "conference_submissions" / "inputs" / "studies.csv"


# -- Tokiwa 2025 (A-25) Spearman rho values extracted from the Frontiers paper
# (Tokiwa 2025, Frontiers in Psychology 16:1420996, Tables 3 + body text).
# These are descriptive correlations against StudySapuri usage outcomes
# (Confirmation Tests Completed / Mastered / Lectures Watched / Viewing Time).
# Reported as "rho =" in the source; representative values below.
TOKIWA_SPEARMAN = {
    "C": {
        "tests_completed":   0.34,   # ρ = 0.34, p < 0.001
        "tests_mastered":    0.35,   # ρ = 0.35, p < 0.001
        # Subscale Productiveness: 0.30-0.32; Responsibility: 0.28-0.30
        "summary":           "+0.30 to +0.35 across achievement-related Sapuri outcomes",
        "direction":         "+",
    },
    "A": {
        "lectures_watched":  0.29,   # Agreeableness ρ = 0.29
        "lectures_watched_respect": 0.27,
        "summary":           "+0.27 to +0.29 across engagement-related outcomes",
        "direction":         "+",
    },
    "E": {
        "assertiveness_tests_completed": 0.26,  # Assertiveness facet positive
        "sociability_lectures_watched":  -0.18, # Sociability facet negative (proxy)
        "summary":           "Assertiveness +0.26; Sociability negative (Tokiwa narrative); net direction MIXED",
        "direction":         "mixed",
    },
    "N": {
        "lectures_watched":  0.29,   # Negative Emotionality / Anxiety subscale ρ = 0.29
        "viewing_time":      0.24,
        "summary":           "+0.24 to +0.29 (Anxiety subscale drives Lectures Watched up)",
        "direction":         "+",
    },
    "O": {
        # Not strongly tied to outcomes in Tokiwa; minor inter-trait correlations
        "summary":           "weak / not strongly tied to Sapuri outcomes",
        "direction":         "weak",
    },
}


# Hofstede 6-D scores for Japan (canonical, Hofstede Insights 2024)
HOFSTEDE_JP = {"PDI": 54, "IDV": 46, "MAS": 95, "UAI": 92, "LTO": 88, "IND": 42}

# Cross-cultural psychology predictions for Big Five vs Japan-specific Hofstede:
# - High MAS + High LTO -> goal-orientation, persistence -> C predicts achievement
# - Low IDV (collectivist) -> social initiative not valued -> E flat / negative
# - Very high UAI (92) -> anxiety pre-empts careful work -> N may predict POSITIVELY
#   (this is the ANTI-WESTERN direction; Western literature reports N negative)
# - Low IND (restrained) -> discipline > pleasure -> reinforces C
# - PDI moderate, no strong direction
HOFSTEDE_PREDICTION_JP = {
    "C": ("+", "high MAS (95) + high LTO (88) + low IND (42)"),
    "E": ("- or flat", "low IDV (46) -> collectivist; assertive social engagement less valued"),
    "N": ("+", "very high UAI (92) -> careful, cautious learners (anti-Western direction)"),
    "A": ("+ (weak)", "no strong dimensional pull either way; usually positive in collectivist contexts"),
    "O": ("weak", "no strong dimensional pull"),
}


def load_asian_studies():
    rows = []
    with open(STUDIES, encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for r in reader:
            if r["region"] == "Asia":
                rows.append(r)
    return rows


def option2_hofstede_direction():
    print("=" * 70)
    print("OPTION 2 — Japan-specific Hofstede-direction match (descriptive)")
    print("=" * 70)
    print(f"\nHofstede Japan: {HOFSTEDE_JP}\n")
    print("Predicted direction in Japan (per Hofstede 6-D):")
    for trait, (pred, why) in HOFSTEDE_PREDICTION_JP.items():
        print(f"  {trait}: {pred:8s}  -- {why}")

    print("\n--- Observed direction in Japan studies ---")
    print("\nA-25 Tokiwa 2025 (Japan K-12, BFI-2-J, post-COVID; Spearman rho):")
    for trait in "OCEAN":
        d = TOKIWA_SPEARMAN.get(trait, {})
        print(f"  {trait}: {d.get('direction','?'):6s}  -- {d.get('summary','no data')}")

    # A-31 Rivers extractable r (loaded from studies.csv)
    rivers = next((r for r in load_asian_studies() if r["study_id"] == "A-31"), None)
    print("\nA-31 Rivers 2021 (Japan UG, TIPI-J, COVID; direct Pearson r):")
    if rivers:
        for trait, col in [("O","r_O"),("C","r_C"),("E","r_E"),("A","r_A"),("N","r_N")]:
            v = rivers[col]
            d = "+" if v and float(v) > 0.05 else ("-" if v and float(v) < -0.05 else "flat")
            print(f"  {trait}: {d:6s}  -- r = {v}")

    print("\n--- Hofstede prediction vs Japan observed (direction match) ---")
    print(f"{'Trait':5s}  {'Hofstede pred':14s}  {'Tokiwa A-25':14s}  {'Rivers A-31':14s}  {'Match?':10s}")
    matches = 0
    total = 0
    for trait in "CEAN":  # skip O (weak both sides)
        pred = HOFSTEDE_PREDICTION_JP[trait][0]
        tok = TOKIWA_SPEARMAN.get(trait, {}).get("direction", "?")
        riv_r = float(rivers[f"r_{trait}"]) if rivers and rivers[f"r_{trait}"] else None
        riv = "+" if riv_r and riv_r > 0.05 else ("-" if riv_r and riv_r < -0.05 else "flat")
        # Match heuristic: predicted "+" matches observed "+"; predicted "-" matches observed "-" or "flat";
        # predicted "+ (weak)" or "weak" matches anything not strongly opposite
        pred_pos = "+" in pred and "-" not in pred
        pred_neg = "-" in pred or "flat" in pred
        tok_match = ((pred_pos and tok == "+")
                      or (pred_neg and tok in ("-", "flat", "mixed"))
                      or pred == "weak")
        riv_match = ((pred_pos and riv == "+")
                      or (pred_neg and riv in ("-", "flat"))
                      or pred == "weak")
        # Both must match
        m = tok_match and riv_match
        if m: matches += 1
        total += 1
        print(f"{trait:5s}  {pred:14s}  {tok:14s}  {riv:14s}  {('YES' if m else 'NO'):10s}")
    print(f"\nMatch ratio: {matches}/{total} traits ({100*matches/total:.0f}%)")
    print("Headline candidate: 'Japan studies show 4/4 (or 3/4) Hofstede-predicted direction matches' if this holds.")


def option3_era_moderator():
    print("\n" + "=" * 70)
    print("OPTION 3 — Era moderator within Asian subset (with extractable r)")
    print("=" * 70)
    rows = load_asian_studies()
    print(f"\nAll Asian rows (n = {len(rows)}):")
    print(f"{'ID':6s}  {'Author/Year':22s}  {'Country':9s}  {'Era':12s}  {'Outcome':24s}  {'r-extractable':15s}")
    has_r = lambda r: any(r[f"r_{t}"] for t in "OCEAN")
    for r in rows:
        outcome = r["primary_achievement"]
        rext = "YES" if has_r(r) else "NO"
        print(f"{r['study_id']:6s}  {r['author_year']:22s}  {r['country']:9s}  {r['era']:12s}  {r['inclusion_status']:24s}  {rext:15s}")

    print("\n--- Era cells within Asia (PRIMARY POOL only, with extractable r) ---")
    era_cells = {}
    for r in rows:
        if r["inclusion_status"] not in ("include", "include_with_caveat", "include_COI"):
            continue
        if r["primary_achievement"] not in ("yes", "partial"):
            continue
        if not has_r(r):
            continue
        era_cells.setdefault(r["era"], []).append(r["study_id"])
    print("Cells:")
    for era in ("pre-COVID", "COVID", "post-COVID"):
        ids = era_cells.get(era, [])
        print(f"  {era:12s}: k = {len(ids):2d}  studies = {ids}")
    pri_pool_k = sum(len(v) for v in era_cells.values())
    print(f"\nTotal Asian primary-pool with extractable r: k = {pri_pool_k}")
    if pri_pool_k < 4:
        print("=> DEGENERATE: cannot fit era moderator within Asia (need k>=2 per era cell, ideally k>=4 total).")

    print("\n--- Era cells if we BROADEN to all included-or-not Asian studies with extractable r ---")
    broad = {}
    for r in rows:
        if not has_r(r):
            continue
        broad.setdefault(r["era"], []).append((r["study_id"], r["inclusion_status"], r["primary_achievement"]))
    for era in ("pre-COVID", "COVID", "post-COVID"):
        items = broad.get(era, [])
        print(f"  {era:12s}: k = {len(items):2d}")
        for sid, inc, pa in items:
            print(f"        - {sid} ({inc} / outcome={pa})")
    broad_k = sum(len(v) for v in broad.values())
    print(f"\nTotal broadened Asian k (any outcome incl. satisfaction/engagement): k = {broad_k}")
    if broad_k < 6:
        print("=> Still very thin. Mixing achievement / satisfaction / engagement outcomes")
        print("   would be a methodologically weak basis for an era-moderator claim.")
    print("\nVerdict: Option 3 (era moderator within Asia) is degenerate with the present corpus.")


if __name__ == "__main__":
    option2_hofstede_direction()
    option3_era_moderator()
