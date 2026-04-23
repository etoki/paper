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
    "A-11": dict(
        first_author="Cheng", year=2023, country="Taiwan",
        journal="British Journal of Educational Technology",
        volume="54", issue="4", pages="898-923",
        doi="10.1111/bjet.13302",
        n_total=1150, n_analyzed=746,
        age_mean=18.02, age_sd=2.74,
        pct_female=53.22,
        education_level="Mixed_secondary_postsecondary",
        sampling_method="convenience", design="cross-sectional",
        modality="fully_online", modality_subtype="asynchronous",
        personality_instrument="BFI_C_only",  # only Conscientiousness
        outcome_type="procrastination",
        # Only C measured; outcome = procrastination (not achievement)
        r_C_outcome=-0.39,  # inhibitive facet, strongest; also proactive -.24
        effect_size_type="r",
        era="COVID", region="Asia",
        risk_of_bias_score=6,
        inclusion_status="exclude_from_primary", primary_achievement="no",
        notes="C only (no O/E/A/N); procrastination outcome, not achievement",
    ),
    "A-12": dict(
        first_author="Baruth", year=2023, country="Israel",
        journal="Education and Information Technologies",
        volume="28", pages="879-904",
        n_total=108, n_analyzed=108,
        education_level="Undergraduate",
        sampling_method="convenience", design="cross-sectional",
        modality="fully_online",
        personality_instrument="BFI-44 Hebrew", personality_item_count=44,
        alpha_O=0.76, alpha_C=0.73, alpha_E=0.80, alpha_A=0.68, alpha_N=0.81,
        outcome_type="satisfaction",
        r_O_outcome=0.294, r_C_outcome=0.335,
        r_E_outcome=0.324, r_A_outcome=0.458, r_N_outcome=-0.542,
        n_for_correlations=108,
        p_value_O="<.01", p_value_C="<.001", p_value_E="<.001",
        p_value_A="<.001", p_value_N="<.001",
        effect_size_type="rho",  # Spearman
        era="COVID", region="Asia",  # Israel
        risk_of_bias_score=4,
        inclusion_status="exclude_from_primary", primary_achievement="no",
        notes="Satisfaction only; Spearman rho; AUTHOR CORRECTION from Cohen&Baruth2017 to Baruth&Cohen2023",
    ),
    "A-13": dict(
        first_author="Dang", year=2025, country="China",
        journal="Frontiers in Psychology", volume="15",
        doi="10.3389/fpsyg.2024.1476437",
        n_total=243, n_analyzed=235,
        pct_female=73.2, education_level="Undergraduate",
        sampling_method="convenience", design="cross-sectional",
        modality="not_online_specific",  # general learning engagement scale
        personality_instrument="NEO-FFI_Chinese", personality_item_count=60,
        outcome_type="engagement",
        r_O_outcome=0.301, r_C_outcome=0.438,
        r_E_outcome=0.309, r_A_outcome=0.247, r_N_outcome=-0.037,
        n_for_correlations=235,
        p_value_C="<.01",
        effect_size_type="r",
        era="post-COVID", region="Asia",
        risk_of_bias_score=5,
        inclusion_status="exclude_from_primary", primary_achievement="no",
        notes="Engagement not online-specific; general learning scale; YEAR CORRECTION to 2025",
    ),
    # A-14 Eilam was excluded at the earlier stage (face-to-face Grade 8 Israel)
    "A-15": dict(
        first_author="Elvers", year=2003, country="US",
        journal="Teaching of Psychology", volume="30", issue="2",
        pages="159-162",
        n_total=54, n_analyzed=47,  # 22 online + 25 lecture; use online subset
        age_mean=18.6, pct_female=66,
        education_level="Undergraduate",
        sampling_method="random_assignment", design="experimental",
        modality="fully_online",
        personality_instrument="NEO-FFI", personality_item_count=60,
        outcome_type="procrastination_exam",
        r_C_outcome=0.41,  # online subset, C × procrastination; exam trend
        r_N_outcome=-0.38,
        n_for_correlations=21,
        p_value_C="=.06", p_value_N="=.09",
        effect_size_type="r",
        era="pre-COVID", region="North_America",
        risk_of_bias_score=6,
        inclusion_status="include_with_caveat", primary_achievement="partial",
        notes="RCT online vs lecture; only C and N reported; small N (21 online subset)",
    ),
    "A-16": dict(
        first_author="Hidalgo-Fuentes", year=2024, country="Honduras+Spain",
        journal="Heliyon", volume="10",
        doi="10.1016/j.heliyon.2024.e36172",
        n_total=457, n_analyzed=457,
        age_mean=22.01, pct_female=70,
        education_level="Undergraduate",
        sampling_method="convenience", design="cross-sectional",
        modality="face-to-face",  # in-class Google Forms
        personality_instrument="BFI-2-S",
        personality_item_count=30,
        outcome_type="procrastination",
        effect_size_type="r",
        era="post-COVID", region="Europe",
        risk_of_bias_score=5,
        inclusion_status="exclude", primary_achievement="no",
        notes="FACE-TO-FACE; procrastination outcome; AUTHOR CORRECTION from Garzon-Umerenkova to Hidalgo-Fuentes",
    ),
    "A-17": dict(
        first_author="Kara", year=2024, country="Turkey",
        journal="Education and Information Technologies", volume="29",
        pages="23517-23546",
        n_total=525, n_analyzed=437,
        pct_female=76.2, education_level="Undergraduate",
        sampling_method="convenience", design="cross-sectional",
        modality="fully_online", modality_subtype="mixed",
        personality_instrument="BFI-44", personality_item_count=44,
        alpha_O=0.78, alpha_C=0.73, alpha_E=0.76, alpha_A=0.63, alpha_N=0.66,
        outcome_type="engagement",
        # Use behavioral engagement as primary (largest C effect)
        r_O_outcome=0.21, r_C_outcome=0.49,
        r_E_outcome=0.18, r_A_outcome=0.31, r_N_outcome=-0.20,
        n_for_correlations=437,
        p_value_O="<.01", p_value_C="<.01", p_value_E="<.01",
        p_value_A="<.01", p_value_N="<.01",
        effect_size_type="r",
        era="post-COVID", region="Europe",  # Turkey
        risk_of_bias_score=6,
        inclusion_status="include_secondary", primary_achievement="no",
        notes="Engagement only; behavioral subscale used",
    ),
    "A-18": dict(
        first_author="Bhagat", year=2019, country="Taiwan",
        journal="Australasian Journal of Educational Technology",
        volume="35", issue="4", pages="98-108",
        doi="10.14742/ajet.4162",
        n_total=208, n_analyzed=208,
        age_mean=25.45, pct_female=53.8,
        education_level="Mixed_UG_Grad",
        sampling_method="convenience", design="cross-sectional",
        modality="fully_online",
        personality_instrument="Mini-IPIP", personality_item_count=20,
        outcome_type="perception",
        effect_size_type="beta",
        era="pre-COVID", region="Asia",
        risk_of_bias_score=5,
        inclusion_status="exclude_from_primary", primary_achievement="no",
        notes="Perceptions only; AUTHOR CORRECTION from Keller&Karau 2013 to Bhagat et al. 2019",
    ),
    "A-19": dict(
        first_author="MacLean", year=2022, country="Canada",
        journal="Master's thesis, University of Calgary",
        n_total=486, n_analyzed=465,
        age_mean=20, pct_female=79.4,
        education_level="Undergraduate",
        sampling_method="convenience", design="cross-sectional",
        modality="fully_online",
        personality_instrument="HEXACO-PI-R", personality_item_count=60,
        outcome_type="preference",
        # HEXACO × preference; no achievement r with Big Five mapping
        effect_size_type="r",
        era="COVID", region="North_America",
        risk_of_bias_score=5,
        inclusion_status="exclude_from_primary", primary_achievement="no",
        notes="HEXACO 6-factor; preference outcome not achievement; grey literature (MSc thesis)",
    ),
    "A-20": dict(
        first_author="Mustafa", year=2022, country="China",
        journal="Frontiers in Psychology", volume="13",
        pages="956281",
        n_total=800, n_analyzed=718,
        pct_female=46, education_level="Mixed_UG_Grad",
        sampling_method="convenience", design="cross-sectional",
        modality="fully_online",
        personality_instrument="Chinese Big Five", personality_item_count=30,
        alpha_O=0.860, alpha_C=0.889, alpha_E=0.848, alpha_A=0.843, alpha_N=0.910,
        outcome_type="satisfaction",
        effect_size_type="beta",  # PLS-SEM β
        era="COVID", region="Asia",
        risk_of_bias_score=6,
        inclusion_status="exclude_from_primary", primary_achievement="no",
        notes="Satisfaction + intention only; PLS-SEM β; COUNTRY CORRECTION from Pakistan to China",
    ),
    "A-21": dict(
        first_author="Nakayama", year=2014, country="Japan",
        journal="Electronic Journal of e-Learning",
        volume="12", issue="4", pages="394-408",
        n_total=53, n_analyzed=53,
        education_level="Undergraduate",
        sampling_method="single_course", design="cross-sectional",
        modality="fully_online",
        personality_instrument="IPIP",
        outcome_type="test_score",
        effect_size_type="path_indirect",
        era="pre-COVID", region="Asia",
        risk_of_bias_score=4,
        inclusion_status="exclude_from_primary", primary_achievement="no",
        notes="N=53 very small; Big Five→test direct effect n.s.; indirect via note-taking only; r values not extractable from OCR",
    ),
    "A-22": dict(
        first_author="Quigley", year=2022, country="UK",
        journal="Personality and Individual Differences",
        volume="194", pages="111645",
        n_total=301, n_analyzed=301,
        age_mean=19.79, age_sd=3.21,
        pct_female=76.08,
        education_level="Undergraduate",
        sampling_method="convenience", design="cross-sectional",
        modality="fully_online", modality_subtype="mixed",
        personality_instrument="BFI-44", personality_item_count=44,
        alpha_O=0.71, alpha_C=0.82, alpha_E=0.88, alpha_A=0.76, alpha_N=0.85,
        outcome_type="engagement_performance",
        # Performance subscale is weak proxy (2 items self-report)
        r_O_outcome=0.00, r_C_outcome=0.26,
        r_E_outcome=0.14, r_A_outcome=0.15, r_N_outcome=-0.02,
        n_for_correlations=301,
        p_value_C="<.001", p_value_E="<.05", p_value_A="<.01",
        effect_size_type="r",
        era="COVID", region="Europe",
        risk_of_bias_score=6,
        inclusion_status="include_with_caveat", primary_achievement="partial",
        notes="Performance subscale of OSES (2 items self-report, weak proxy); primary is engagement",
    ),
    "A-23": dict(
        first_author="Rodrigues", year=2024, country="Germany",
        journal="European Journal of Investigation in Health, Psychology and Education",
        volume="14", issue="2", pages="368-384",
        n_total=287, n_analyzed=260,  # N=260 for GPA after exclusions
        age_mean=22.68, age_sd=3.47,
        pct_female=76.7,
        education_level="Undergraduate",
        sampling_method="convenience", design="cross-sectional",
        modality="fully_online",
        personality_instrument="BFI-S", personality_item_count=15,
        outcome_type="GPA",
        # German GPA: lower = better; reverse for pooling so that positive r = better performance
        # Rodrigues reports C×GPA = -.228 (better); convert to +.228 for pooling consistency
        r_O_outcome=0.00, r_C_outcome=0.228,  # sign-flipped
        r_E_outcome=-0.025, r_A_outcome=0.00, r_N_outcome=-0.142,
        n_for_correlations=260,
        p_value_C="<.01", p_value_E="ns", p_value_N="<.05",
        effect_size_type="r",
        era="COVID", region="Europe",
        risk_of_bias_score=7,
        inclusion_status="include", primary_achievement="yes",
        notes="German GPA sign-reversed so that positive r = better; pre-registered on OSF; best quality COVID GPA study",
    ),
    "A-24": dict(
        first_author="Tlili", year=2023, country="Tunisia",
        journal="Frontiers in Psychology", volume="14",
        pages="1071985",
        n_total=92, n_analyzed=65,
        pct_female=34, education_level="Undergraduate",
        sampling_method="single_course", design="cross-sectional",
        modality="fully_online", platform_name="Moodle",
        personality_instrument="BFI-44", personality_item_count=44,
        outcome_type="navigational_behavior",
        effect_size_type="lag_sequential",
        era="COVID", region="Other",  # North Africa
        risk_of_bias_score=4,
        inclusion_status="exclude", primary_achievement="no",
        notes="LSA process data only; no r or β extractable; N=65; Agreeableness dropped",
    ),
    "A-25": dict(
        first_author="Tokiwa", year=2025, country="Japan",
        journal="In preparation", n_total=103, n_analyzed=103,
        education_level="K-12", sampling_method="convenience",
        design="cross-sectional", modality="fully_online",
        modality_subtype="asynchronous",
        personality_instrument="BFI-2-J",
        outcome_type="test_completion",
        effect_size_type="r",
        era="post-COVID", region="Asia",
        risk_of_bias_score=6,
        inclusion_status="include_COI", primary_achievement="yes",
        notes="COI: author's own prior study; sensitivity analysis will exclude",
    ),
    "A-26": dict(
        first_author="Wang", year=2023, country="China",
        journal="Frontiers in Psychology", volume="14",
        pages="1241477",
        n_total=1625, n_analyzed=1625,
        pct_female=54.8, education_level="K-12",
        sampling_method="stratified", design="cross-sectional",
        modality="fully_online",
        personality_instrument="Chinese Big Five", personality_item_count=None,
        outcome_type="achievement_self_report",
        # Direct Big Five × achievement β = -.173 (n.s.) — full mediation via engagement
        # Bivariate r(Big_Five_total × achievement) = .250
        # Use β to engagement as moderator-relevant:
        # C β_eng=.322, O β_eng=.253, ES β_eng=.169, A β_eng=.112, E β_eng=-.058
        # For primary achievement pool, use bivariate total=0.25 as placeholder;
        # no per-trait direct r with achievement
        effect_size_type="path_mediated",
        era="post-COVID", region="Asia",
        risk_of_bias_score=6,
        inclusion_status="include_with_caveat", primary_achievement="partial",
        notes="K-12 online; per-trait direct r with achievement not reported (only β to engagement); achievement self-report",
    ),
    "A-27": dict(
        first_author="Wu_Yu", year=2024,
        inclusion_status="PDF_unavailable", primary_achievement="unknown",
        notes="PDF not in corpus; cited in A-13 Dang 2025; expected N=1004 engagement",
    ),
    "A-28": dict(
        first_author="Yu", year=2021, country="China",
        journal="International Journal of Educational Technology in Higher Education",
        volume="18", pages="14",
        n_total=1152, n_analyzed=1152,
        pct_female=51.6, education_level="Mixed_UG_Grad",
        sampling_method="random", design="cross-sectional",
        modality="fully_online", platform_name="BLCU_MOOC",
        personality_instrument="Big Five Scale (McCrae)", personality_item_count=40,
        alpha_O=0.81, alpha_C=0.80, alpha_E=0.75, alpha_A=0.76, alpha_N=0.78,
        outcome_type="MOOC_composite",
        # β standardized; total R²=.565
        # Text says zero-order r not reported, only β
        beta_O=0.305, beta_C=0.057, beta_E=-0.076, beta_A=0.442, beta_N=0.037,
        n_for_correlations=1152,
        p_value_O="<.01", p_value_C="=.007", p_value_E="<.01",
        p_value_A="<.01", p_value_N="=.09",
        effect_size_type="beta",
        era="COVID", region="Asia",
        risk_of_bias_score=7,
        inclusion_status="include", primary_achievement="yes",
        notes="MOOC composite score objective; β only (Peterson-Brown conversion needed); Linguistics majors",
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
