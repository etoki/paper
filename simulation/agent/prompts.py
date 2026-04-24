"""Prompt templates for the Big Five -> university entrance exam outcome agent.

Design principles
-----------------
1. English prompts for publication in an English-language journal and direct
   reproduction in the Methods section.
2. No effect-size hints. The agent draws on its pre-trained knowledge of
   personality psychology, not on hints about which Big Five dimension
   correlates with which outcome. This preserves the causal pathway inference
   as the primary object of evaluation.
3. Scale anchors (hensachi reference points, StudySapuri usage ranges) are
   included because they are units of measurement, not effect sizes.
4. The three-stage reasoning (backstory -> learning behavior -> outcome) is
   mandated in the user prompt but executed inside extended thinking.
5. Final output is produced via a forced tool call, guaranteeing a valid
   JSON payload conforming to a pre-declared schema.
"""

from typing import Mapping


SYSTEM_PROMPT = """You are a research assistant simulating Japanese high school seniors preparing for university entrance examinations.

Your task is to embody a specific student based on their Big Five personality profile (BFI-2, 1-5 scale) and predict:
(a) their StudySapuri online-learning behavior over nine months (April-December 2023), and
(b) the selectivity (hensachi) of the university to which they will ultimately be admitted.

COHORT CONTEXT
- Japanese private senior high school in the Kanto region (first graduating cohort, class of 2026).
- Students apply to a broad portfolio of institutions: Japanese national and private universities as well as overseas universities (UK Russell Group, Australian Group of Eight, US colleges, Asian universities in QS rankings).
- All students have access to StudySapuri, a major Japanese online-learning platform providing video lectures and post-lecture confirmation tests.

HENSACHI SCALE (Japanese standardized university-selectivity score; overseas universities converted via QS ranking percentile)
- >=70 : University of Tokyo, Kyoto University, top medical schools, QS Top 50 universities (e.g., Sydney, Manchester, ETH).
- 65-69 : Hitotsubashi, Tokyo Tech, Keio, Waseda top faculties, QS 50-100 universities (e.g., Bristol, Auckland, Birmingham).
- 60-64 : Sophia, upper MARCH / Kankan-Doritsu, QS 100-300 universities (e.g., Cardiff, Queen Mary London, Deakin).
- 55-59 : Mid-tier national universities, Nittokomasen upper tier, QS 300-500 universities.
- 50-54 : Mid-tier private universities, QS 500-700 universities.
- 45-49 : Lower mid-tier private universities, QS 700-1000 universities.
- <=44 : Lower-tier private universities.

STUDYSAPURI USAGE RANGES (9-month window)
- Number of confirmation tests completed : 0-600 typical; 0 = minimal usage, ~60 = median, >200 = heavy usage.
- Number of confirmation tests mastered  : always less than or equal to tests completed.

OUTPUT FORMAT
You must call the submit_prediction tool exactly once. All reasoning (backstory -> learning behavior -> examination outcome) must be completed in your internal reasoning before the tool call; the tool-call arguments are the final answer only.
"""


USER_PROMPT_TEMPLATE = """Simulate the following Japanese high school senior and submit a prediction.

BIG FIVE SCORES (BFI-2, 1-5 scale; 3.0 is the theoretical midpoint)
- Openness            : {O:.2f}
- Conscientiousness   : {C:.2f}
- Extraversion        : {E:.2f}
- Agreeableness       : {A:.2f}
- Neuroticism         : {N:.2f}

REASONING STAGES (to be completed in your internal thinking before calling the tool)
Stage 1 (backstory). Describe this student's typical study habits, social behavior in class and online, relationship to academic pressure, and ways of coping with stress. Draw on what personality psychology tells you about how each Big Five dimension shapes daily behavior in a Japanese high school context.

Stage 2 (learning-behavior simulation). Given the backstory, predict how intensively this student will use StudySapuri during April-December 2023 -- number of confirmation tests completed, number mastered, qualitative pattern of usage (steady vs. bursty, selective vs. exhaustive).

Stage 3 (examination outcome). Given the backstory and learning behavior, predict the hensachi of the university to which this student will be admitted. Consider their application strategy (safe, balanced, or ambitious), their performance under real examination stress, and how their personality mediates both. Name one concrete university and faculty as an example.

After reasoning through all three stages, call submit_prediction with your final answer.
"""


def build_user_prompt(big_five: Mapping[str, float]) -> str:
    """Format the per-participant user prompt from a Big Five dict."""
    return USER_PROMPT_TEMPLATE.format(
        O=big_five["Openness"],
        C=big_five["Conscientiousness"],
        E=big_five["Extraversion"],
        A=big_five["Agreeableness"],
        N=big_five["Neuroticism"],
    )


PREDICTION_TOOL = {
    "name": "submit_prediction",
    "description": (
        "Submit the final predicted StudySapuri learning behavior and university "
        "admission outcome for this student. Call this exactly once after "
        "completing all three reasoning stages."
    ),
    "input_schema": {
        "type": "object",
        "properties": {
            "backstory": {
                "type": "string",
                "description": (
                    "2-3 sentence narrative describing the student's study habits, "
                    "social behavior, and stress-coping style, grounded in their Big Five profile."
                ),
            },
            "learning_behavior": {
                "type": "string",
                "description": (
                    "1-2 sentence qualitative description of their StudySapuri usage pattern "
                    "(e.g., steady daily use with high completion rate, bursty pre-exam use, "
                    "selective by subject, exhaustive mastery, etc.)."
                ),
            },
            "predicted_tests_completed": {
                "type": "integer",
                "minimum": 0,
                "maximum": 800,
                "description": "Predicted number of StudySapuri confirmation tests completed during Apr-Dec 2023.",
            },
            "predicted_tests_mastered": {
                "type": "integer",
                "minimum": 0,
                "maximum": 800,
                "description": "Predicted number of tests mastered (must be less than or equal to predicted_tests_completed).",
            },
            "predicted_hensachi": {
                "type": "number",
                "minimum": 30.0,
                "maximum": 80.0,
                "description": "Predicted hensachi (Japanese selectivity score) of the admitted university.",
            },
            "predicted_university_example": {
                "type": "string",
                "description": "One concrete example of an admitted university and faculty (Japanese or overseas).",
            },
            "application_strategy": {
                "type": "string",
                "enum": ["ambitious", "balanced", "safe"],
                "description": "Overall application portfolio strategy inferred from the student's personality.",
            },
            "confidence": {
                "type": "string",
                "enum": ["low", "medium", "high"],
                "description": "Self-reported confidence in the prediction given only Big Five input.",
            },
            "reasoning": {
                "type": "string",
                "description": (
                    "1-2 sentence explanation of how each Big Five dimension contributed to the predicted hensachi. "
                    "Cite specific trait values and the mechanism."
                ),
            },
        },
        "required": [
            "backstory",
            "learning_behavior",
            "predicted_tests_completed",
            "predicted_tests_mastered",
            "predicted_hensachi",
            "predicted_university_example",
            "application_strategy",
            "confidence",
            "reasoning",
        ],
    },
}
