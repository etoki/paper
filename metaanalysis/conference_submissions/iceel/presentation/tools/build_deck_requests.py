"""Build Google Slides batchUpdate requests for the ICEEL 2026 (EL2036) deck.

Slide plan: ../20261128_slides.md. Colours: Japan = red {r:}, China = blue {b:},
same direction = green {g:}. Output: b0.json, b1.json ... (one batchUpdate each).
"""
import json
from slides_requests import slide

S = []
S += slide("iceel_s02", 1, "About Me",
"**Eisuke Tokiwa** — Founder, SUNBLAZE Co., Ltd.\n\n"
"Since 2024: research in **personality psychology**\n"
"Main topic: **online learning of Japanese students**\n\n"
"Today's talk comes from my **review** of online-learning studies.\n"
"The question: **culture.**", size=22)

S += slide("iceel_s03", 2, "Key Words: The Big Five", None,
 table={"box": (25, 78, 670, 190), "size": 16, "colw": [250, 420], "rows": [
  ["Trait", "In simple words"],
  ["Openness", "Interest in new things"],
  ["**Conscientiousness**", "Careful and well-planned"],
  ["**Extraversion**", "Outgoing"],
  ["Agreeableness", "Kind and cooperative"],
  ["**Neuroticism**", "Easily anxious or worried"]]},
 extra_text=[("Face-to-face classes: **Conscientiousness** has the strongest link with grades.", (25, 290, 670, 40), 18)],
 source="Sources: Poropat (2009), Psychological Bulletin; Mammadov (2022), Journal of Personality.")

S += slide("iceel_s04", 3, "Extraversion and Grades: Asia vs. Outside Asia", None,
 table={"box": (120, 82, 480, 120), "size": 22, "colw": [300, 180], "rows": [
  ["", "Correlation (r)"],
  ["Outside Asia (7 studies)", "**+0.050**"],
  ["{r:Asia (2 studies)}", "{r:−0.095}"]]},
 extra_text=[("Difference: **p = .009**", (25, 222, 670, 32), 22),
             ("In Asia, more extraverted students had **slightly lower grades.**", (25, 265, 670, 40), 20)],
 source="Knapp–Hartung meta-regression, F(1, 7) = 12.94. Parent review: Tokiwa (2026), Research Square, DOI 10.21203/rs.3.rs-9513298/v1.")

S += slide("iceel_s05", 4, "But \"Asia\" Is Only Two Studies", None,
 table={"box": (110, 82, 500, 130), "size": 20, "colw": [250, 250], "rows": [
  ["{b:China}", "{r:Japan}"],
  ["**Yu (2021)**", "**Rivers (2021)**"],
  ["N = 1,152", "N = 149"],
  ["University (MOOC)", "University (Moodle)"]]},
 extra_text=[("China and Japan have **quite different cultures.**\n**Can we call them one \"Asia\"?**", (25, 245, 670, 70), 22)],
 source="Sources: Yu (2021), Int. J. Educ. Technol. High. Educ.; Rivers (2021), Educ. Inf. Technol.")

S += slide("iceel_s06", 5, "Hofstede's Cultural Dimensions (3 of 6)", None,
 table={"box": (25, 80, 670, 160), "size": 17, "colw": [210, 260, 100, 100], "rows": [
  ["Dimension", "Meaning", "{r:Japan}", "{b:China}"],
  ["Individualism", "Individual or group?", "62", "43"],
  ["**Uncertainty Avoidance**", "Avoid unclear situations?", "{r:92}", "30"],
  ["**Masculinity**", "Value competition and success?", "{r:95}", "66"]]},
 extra_text=[("Japan − China on Uncertainty Avoidance: **+62 points**", (25, 270, 670, 32), 22)],
 source="Sources: The Culture Factor Group (2026), country comparison tool (formerly Hofstede Insights), retrieved 27 Oct 2026; Hofstede, Hofstede & Minkov (2010).")

S += slide("iceel_s07", 6, "Two Questions",
"**1.** When we split \"Asia\" by country, can **cultural differences** explain the results?\n\n"
"**2.** In Japanese studies, do personality–learning links go in the **direction that Japanese culture suggests**?", size=24)

S += slide("iceel_s08", 7, "Method",
"**Step 1 — China vs. Japan**\n"
"> Culture scores + personality–grade correlations, side by side\n"
"> ⚠ Only 2 studies → **no statistical test** (descriptive only)\n"
"**Step 2 — Two studies from Japan**\n"
"> Rivers (2021): university / Tokiwa (2025): high school\n"
"**Reference: Hofstede & McCrae (2004)**\n"
"> 33 countries: culture scores × average personality", size=20,
 source="Sources: Hofstede & McCrae (2004), Cross-Cultural Research; Tokiwa (2025), Frontiers in Psychology.")

S += slide("iceel_s09", 8, "Result 1: Extraversion Goes the Opposite Way", None,
 table={"box": (100, 80, 520, 110), "size": 20, "colw": [280, 120, 120], "rows": [
  ["", "{b:China}", "{r:Japan}"],
  ["Individualism", "43", "{r:62}"],
  ["Extraversion × grades (r)", "−0.074", "{r:−0.173}"]]},
 extra_text=[("Expected: Extraversion helps more in individualistic cultures.\nFound: **the opposite.**", (25, 210, 670, 62), 20),
             ("⚠ {i:Only a two-country comparison}", (25, 292, 670, 28), 17)],
 source="Sources: Yu (2021); Rivers (2021). China r converted from β (Peterson & Brown, 2005).")

S += slide("iceel_s10", 9, "Result 2: Two Japanese Studies Are Very Different", None,
 table={"box": (25, 78, 670, 200), "size": 15, "colw": [150, 260, 260], "rows": [
  ["", "Rivers (2021)", "Tokiwa (2025)"],
  ["Students", "University (N = 149)", "High school (N = 103)"],
  ["Format", "Asynchronous (Moodle)", "Asynchronous (StudySapuri)"],
  ["**Outcome**", "**Grades**", "**Learning behavior** (videos, tests)"],
  ["**Personality test**", "**10 items** (TIPI-J)", "**60 items** (BFI-2-J)"]]},
 extra_text=[("**6×** difference in test length → more measurement error", (25, 300, 670, 32), 20)],
 source="Sources: Rivers (2021), Educ. Inf. Technol.; Tokiwa (2025), Frontiers in Psychology.")

S += slide("iceel_s11", 10, "Result 3: Three of Five Traits Match", None,
 table={"box": (25, 76, 670, 215), "size": 15, "colw": [180, 130, 140, 130, 90], "highlight": [3], "rows": [
  ["Trait", "Culture suggests", "Tokiwa (behavior)", "Rivers (grades)", "Same?"],
  ["Conscientiousness", "+", "+", "+", "{g:✓}"],
  ["Extraversion", "−", "mixed", "−", "{g:✓}"],
  ["**Neuroticism**", "**+**", "{r:+ 0.17}", "{r:+ 0.107}", "{g:✓}"],
  ["Agreeableness", "−", "+", "+", "✗"],
  ["Openness", "+", "weak", "−", "✗"]]},
 extra_text=[("Earlier reviews: Neuroticism × grades is **zero or negative** (Asian data: **−0.19**)", (25, 312, 670, 30), 17)],
 source="Sources: Hofstede & McCrae (2004); Mammadov (2022), Journal of Personality. Tokiwa (2025): Spearman ρ; Rivers (2021): Pearson r.")

S += slide("iceel_s12", 11, "Why Is Neuroticism Positive? (A Hypothesis)",
"**Japan: very high Uncertainty Avoidance (92)**\n"
"Hofstede (1986): in such cultures, students\n"
"> like **structured learning**\n"
"> are **rewarded for accuracy**\n"
"→ Students who worry a little may **study more carefully**\n"
"**⚠ Only a hypothesis**\n"
"> Both values are small and **not significant**\n"
"> Culture scores = **countries**; correlations = **individual students**", size=19,
 source="Source: Hofstede (1986), International Journal of Intercultural Relations.")

S += slide("iceel_s13", 12, "Three Messages",
"**1. \"Asia\" is not one thing.**\n"
"> But by country, there are **only two studies.**\n"
"**2. In Japan, Neuroticism was positive.**\n"
"> Maybe related to culture — **still a hypothesis.**\n"
"**3. Future studies need:**\n"
"> same personality test / same grade measure / **each student's cultural values**", size=22)

S += slide("iceel_s14", 13, "Thank You",
"Questions are welcome — also by email:\n"
"**eisuke.tokiwa@sunblaze.jp**\n\n"
"Paper ID: EL2036\n"
"Eisuke Tokiwa / SUNBLAZE Co., Ltd.\n"
"ORCID: 0009-0009-7124-6669", size=24)

json.dump(S, open("all.json", "w"), ensure_ascii=False)
batches = []; cur = []
for r in S:
    if "createSlide" in r and len(cur) > 120:
        batches.append(cur); cur = []
    cur.append(r)
batches.append(cur)
for i, b in enumerate(batches):
    json.dump(b, open(f"b{i}.json", "w"), ensure_ascii=False, separators=(",", ":"))
print(len(S), [len(b) for b in batches], [len(open(f"b{i}.json").read()) for i in range(len(batches))])
