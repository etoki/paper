"""
Populate data_extraction.csv with effect sizes extracted from deep reading notes.

Produces `data_extraction_populated.csv` with:
- Author/year corrections from deep reading
- Effect sizes (r or beta) per trait
- Modality / era / region classifications
- Inclusion/exclusion flags per study

All values come from `deep_reading_notes.md` Part C (A-XX entries).
Built incrementally — currently contains A-01 to A-10.
"""
import csv
from pathlib import Path

INPUT_CSV = Path("/home/user/paper/metaanalysis/data_extraction.csv")
OUTPUT_CSV = Path("/home/user/paper/metaanalysis/analysis/data_extraction_populated.csv")
OUTPUT_CSV.parent.mkdir(parents=True, exist_ok=True)


# -----------------------------------------------------------------------------
# Data per study
# -----------------------------------------------------------------------------
# Keys match the CSV column names where possible.
# Extra fields added:
#   inclusion_status: include / exclude / ambiguous
#   primary_achievement: yes / no (whether effect sizes are on achievement outcome)
#   region: Asia / Europe / North_America / Other
#   author_correction: if original CSV had wrong author

STUDIES = {
    "A-01": dict(
        first_author="Abe", year=2020, country="US",
        journal="The Internet and Higher Education", volume="45",
        doi="10.1016/j.iheduc.2019.100724",
        n_total=92, n_analyzed=92,
        pct_female=None, education_level="Undergraduate",
        sampling_method="convenience", design="cross-sectional",
        modality="fully_online", modality_subtype="asynchronous",
        duration="semester", subject_domain="Psychology",
        platform_name="LMS", personality_instrument="BFI-44",
        personality_item_count=44,
        alpha_O=None, alpha_C=None, alpha_E=None, alpha_A=None, alpha_N=None,
        outcome_type="course_grade",
        outcome_instrument="quiz_avg + paper grade",
        # Use paper grade as primary achievement outcome for pooling (larger N validity)
        # Abe 2020 provides both quiz avg and paper grade; we use paper grade (essay)
        # for the primary analysis since it's more standardized as a course product.
        # Quiz-avg r values preserved in notes.
        r_O_outcome=0.35, r_C_outcome=0.37,
        r_E_outcome=0.03, r_A_outcome=0.16, r_N_outcome=-0.02,
        n_for_correlations=92,
        p_value_O="<.01", p_value_C="<.01",
        p_value_E="ns", p_value_A="<.10", p_value_N="ns",
        effect_size_type="r",
        era="pre-COVID", region="North_America",
        risk_of_bias_score=5,
        inclusion_status="include", primary_achievement="yes",
        notes="Used paper grade as primary; quiz r (C=.48, O=.13) in sensitivity",
    ),
    "A-02": dict(
        first_author="Alkis", year=2018, country="Turkey",
        journal="Educational Technology & Society", volume="21", issue="3",
        pages="35-47",
        n_total=316, n_analyzed=189,  # online subset for primary pool
        pct_female=58, education_level="Undergraduate",
        sampling_method="convenience", design="cross-sectional",
        modality="fully_online", modality_subtype="mixed",
        personality_instrument="BFI-44",
        personality_item_count=44,
        alpha_O=0.81, alpha_C=0.76, alpha_E=0.84, alpha_A=0.60, alpha_N=0.81,
        outcome_type="course_grade",
        outcome_instrument="course_grade_0-100",
        # Online subset values (N=189)
        r_O_outcome=-0.092, r_C_outcome=0.205,
        r_E_outcome=0.051, r_A_outcome=0.094, r_N_outcome=0.03,
        n_for_correlations=189,
        p_value_O="ns", p_value_C="<.01", p_value_E="ns", p_value_A="ns", p_value_N="ns",
        effect_size_type="r",
        era="pre-COVID", region="Europe",  # Turkey typically classified Europe/Middle East
        risk_of_bias_score=6,
        inclusion_status="include", primary_achievement="yes",
        notes="Online N=189 used for primary; blended N=127 available as moderator sensitivity",
    ),
    "A-03": dict(
        first_author="Ashouri", year=2025, country="Iran",
        journal="Cureus", volume="17", issue="7",
        n_total=183, n_analyzed=183,
        pct_female=75, education_level="Undergraduate",
        sampling_method="convenience", design="cross-sectional",
        modality="fully_online", modality_subtype="e-learning_LMS",
        personality_instrument="BFI-44", personality_item_count=44,
        outcome_type="satisfaction",
        # Beta-only reported; satisfaction outcome, EXCLUDE from primary achievement pool
        effect_size_type="beta",
        era="COVID", region="Asia",
        risk_of_bias_score=5,
        inclusion_status="exclude_from_primary", primary_achievement="no",
        notes="Satisfaction only; N beta range -.17 to -.23; no achievement r reported",
    ),
    "A-04": dict(
        first_author="Audet", year=2021, country="Canada",
        journal="Personality and Individual Differences", volume="180",
        pages="110969", doi="10.1016/j.paid.2021.110969",
        n_total=350, n_analyzed=350,
        pct_female=87.8, education_level="Undergraduate",
        sampling_method="convenience", design="longitudinal_2wave",
        modality="fully_online", modality_subtype="synchronous",
        personality_instrument="BFI-44", personality_item_count=44,
        alpha_O=0.80, alpha_C=0.80, alpha_E=0.80, alpha_A=0.80, alpha_N=0.80,
        outcome_type="engagement",
        # No GPA; engagement only. EXCLUDE from primary achievement pool.
        effect_size_type="beta",
        era="COVID", region="North_America",
        risk_of_bias_score=6,
        inclusion_status="exclude_from_primary", primary_achievement="no",
        notes="Engagement/SE only; no achievement. Fall 2020 McGill cohort OVERLAPS with A-05",
    ),
    "A-05": dict(
        first_author="Audet", year=2023, country="Canada",
        journal="Journal of College Reading and Learning", volume="53", issue="4",
        pages="298-315",
        n_total=678, n_analyzed=678,
        pct_female=87.8,
        education_level="Undergraduate",
        sampling_method="convenience", design="longitudinal_2wave",
        modality="fully_online", modality_subtype="mixed",
        personality_instrument="BFI-44", personality_item_count=44,
        outcome_type="engagement",
        effect_size_type="beta",
        era="COVID", region="North_America",
        risk_of_bias_score=6,
        inclusion_status="exclude_overlap",
        primary_achievement="no",
        notes="Fall 2020 McGill cohort OVERLAPS A-04; keep A-04 and drop A-05 per protocol",
    ),
    "A-06": dict(
        first_author="Sahinidis", year=2021, country="Greece",
        journal="Springer Proceedings (Strategic Innovative Marketing)",
        n_total=555, n_analyzed=555,
        pct_female=59,
        education_level="Undergraduate",
        sampling_method="convenience",
        design="cross-sectional",
        modality="fully_online", modality_subtype="synchronous",
        personality_instrument="Custom 30-item Big Five",
        personality_item_count=30,
        alpha_O=0.791, alpha_C=0.753, alpha_E=0.749, alpha_A=0.573, alpha_N=0.695,
        outcome_type="satisfaction",
        effect_size_type="beta",
        era="COVID", region="Europe",
        risk_of_bias_score=4,
        inclusion_status="exclude_from_primary",
        primary_achievement="no",
        notes="Satisfaction only; non-BFI scale; AUTHOR CORRECTION from Baruth&Cohen to Sahinidis&Tsaknis",
    ),
    "A-07": dict(
        first_author="Cohen", year=2017, country="Israel",
        journal="Computers in Human Behavior", volume="72", pages="1-12",
        doi="10.1016/j.chb.2017.02.030",
        n_total=72, n_analyzed=72,
        pct_female=63, education_level="Graduate",  # post-BA teacher education
        sampling_method="convenience", design="cross-sectional",
        modality="fully_online", modality_subtype="asynchronous",
        personality_instrument="BFI-44 Hebrew",
        personality_item_count=44,
        alpha_O=0.73, alpha_C=0.78, alpha_E=0.81, alpha_A=0.84, alpha_N=0.89,
        outcome_type="satisfaction",
        r_O_outcome=0.376, r_C_outcome=0.390,
        r_E_outcome=0.025, r_A_outcome=0.099, r_N_outcome=0.041,
        n_for_correlations=72,
        effect_size_type="r",
        era="pre-COVID", region="Asia",  # Middle East but Israel sometimes treated as Asia
        risk_of_bias_score=5,
        inclusion_status="exclude_from_primary", primary_achievement="no",
        notes="Satisfaction only; AUTHOR CORRECTION from Baruth2023 to Cohen&Baruth2017",
    ),
    "A-08": dict(
        first_author="Keller", year=2013, country="US",
        journal="Computers in Human Behavior", volume="29", issue="6",
        pages="2494-2500", doi="10.1016/j.chb.2013.06.007",
        n_total=250, n_analyzed=250,
        pct_female=72.8,
        education_level="Mixed_UG_Grad",
        sampling_method="convenience", design="cross-sectional",
        modality="fully_online", modality_subtype="asynchronous",
        platform_name="Blackboard",
        personality_instrument="IPIP-50", personality_item_count=50,
        alpha_O=0.77, alpha_C=0.81, alpha_E=0.87, alpha_A=0.83, alpha_N=None,
        outcome_type="perception",
        # Perception/OCI only - not achievement
        effect_size_type="r",
        era="pre-COVID", region="North_America",
        risk_of_bias_score=5,
        inclusion_status="exclude_from_primary", primary_achievement="no",
        notes="OCI perceptions only; AUTHOR CORRECTION from Bhagat to Keller&Karau",
    ),
    "A-09": dict(
        first_author="Bhattacharjee", year=2025, country="India",
        journal="Frontiers in Psychology", volume="16",
        doi="10.3389/fpsyg.2025.1490427",
        n_total=600, n_analyzed=384,
        pct_female=42.7,
        education_level="Undergraduate",
        sampling_method="purposive", design="cross-sectional",
        modality="face-to-face",
        personality_instrument="BFI-44",
        outcome_type="GPA",
        effect_size_type="group_means",
        era="post-COVID", region="Asia",
        risk_of_bias_score=4,
        inclusion_status="exclude", primary_achievement="no",
        notes="FACE-TO-FACE engineering college; modality ineligible",
    ),
    "A-10": dict(
        first_author="Boonyapison", year=2025, country="Thailand",
        journal="Scientific Reports", volume="15",
        doi="10.1038/s41598-025-01038-7",
        n_total=250, n_analyzed=203,
        age_mean=17.19, age_sd=0.53,
        pct_female=66, education_level="K-12",  # Grade 12
        sampling_method="convenience", design="cross-sectional",
        modality="face-to-face",
        personality_instrument="BFI-44", personality_item_count=44,
        outcome_type="GPA",
        effect_size_type="group_means",
        era="post-COVID", region="Asia",
        risk_of_bias_score=6,
        inclusion_status="exclude", primary_achievement="no",
        notes="FACE-TO-FACE international high school; modality ineligible",
    ),
}


def render_row(study_id, data, header):
    """Produce CSV row matching header column order."""
    row = {h: "" for h in header}
    row["study_id"] = study_id
    for k, v in data.items():
        if k in row:
            if v is None:
                row[k] = ""
            else:
                row[k] = str(v)
    # Append extra fields as notes if not in header
    extras = {k: v for k, v in data.items() if k not in row and v is not None}
    if extras:
        extra_str = "; ".join(f"{k}={v}" for k, v in extras.items())
        existing_notes = row.get("notes", "")
        row["notes"] = (existing_notes + " | " + extra_str).strip(" |")
    return row


def main():
    with INPUT_CSV.open("r", encoding="utf-8") as f:
        reader = csv.reader(f)
        header = next(reader)
        original_rows = {r[0]: r for r in reader if r and r[0]}

    # Extend header with new fields
    new_fields = ["inclusion_status", "primary_achievement", "region", "author_correction"]
    for nf in new_fields:
        if nf not in header:
            header.append(nf)

    out_rows = []
    for sid, row_list in original_rows.items():
        # Start from original row
        base = dict(zip(header[:len(row_list)], row_list + [""] * (len(header) - len(row_list))))
        if sid in STUDIES:
            updated = render_row(sid, STUDIES[sid], header)
            # Keep original values for fields not overridden
            for k in header:
                if updated[k]:
                    base[k] = updated[k]
            # Mark author correction flag
            new_author = STUDIES[sid].get("first_author", "")
            orig_author = row_list[1] if len(row_list) > 1 else ""
            if new_author and orig_author and new_author.lower() != orig_author.lower():
                base["author_correction"] = f"corrected from {orig_author} to {new_author}"
        out_rows.append(base)

    with OUTPUT_CSV.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=header)
        writer.writeheader()
        writer.writerows(out_rows)

    print(f"Wrote {OUTPUT_CSV} with {len(out_rows)} rows and {len(header)} columns")
    print(f"Populated studies: {len(STUDIES)}")


if __name__ == "__main__":
    main()
