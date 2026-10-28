# Title/abstract screening: criteria and instructions

This file is given, unchanged, to every Claude run that proposes a
title/abstract decision. It restates the eligibility criteria registered
on OSF (https://osf.io/e5w47/, "Data collection procedures", PICOS) and
adds the decision rules for this stage. Claude only **proposes** a
decision; the reviewer (ET) makes the final decision for each record.

Version: 2 (2026-10-27). Any change gets a new version number and date.
- v1 to v2: rule 4 changed after the pilot. v1 excluded records whose
  abstract did not mention an online setting; v2 excludes (E2) only when
  the abstract states a face-to-face-only setting or the study is not
  about learning, and proposes `maybe` when the setting is not stated.

## Eligibility criteria (as registered)

- **Population:** students of any educational level (K-12,
  undergraduate, graduate, adult learners) enrolled in online learning
  environments (fully online, blended, MOOC, synchronous or
  asynchronous). No geographic or demographic restriction. Samples with
  fewer than 10 learners are excluded.
- **Exposure:** Big Five or HEXACO personality traits (Conscientiousness,
  Openness, Extraversion, Agreeableness, Neuroticism or Emotional
  Stability; HEXACO Honesty-Humility and Emotionality) measured with a
  validated inventory (BFI, BFI-2, NEO-PI-R, NEO-FFI, IPIP, TIPI,
  HEXACO-PI-R, HEXACO-60, or another Big-Five-aligned scale with
  published psychometric properties). MBTI, ad-hoc scales, and
  single-trait measures (for example grit only, proactive personality
  only) are excluded.
- **Outcome:** primary: academic achievement (GPA, course grade, exam or
  test score, learning performance, including self-rated performance).
  Secondary: academic satisfaction, academic engagement, LMS behaviour,
  dropout or persistence.
- **Study design:** quantitative primary studies (cross-sectional
  correlational, longitudinal, prospective) that could report an
  association between a trait and an outcome. Qualitative-only studies,
  commentaries, editorials, narrative reviews, and case studies are
  excluded. Systematic reviews and meta-analyses are excluded as studies
  (their reference lists were already searched).
- **Report:** English only; peer-reviewed journal articles, conference
  papers, or dissertations and theses. Preprints and unpublished
  manuscripts are excluded.

## Decisions at this stage

Propose exactly one of:

- `include`: the title or abstract indicates that all of population,
  exposure and outcome are likely met in a quantitative primary study.
- `maybe`: one or more criteria cannot be judged from the title and
  abstract, but nothing shows that a criterion is failed. Use `maybe`
  whenever in doubt.
- `exclude`: the title or abstract shows that at least one criterion is
  clearly not met. Give the first failed criterion as the reason.

Rules:

1. **Err towards `maybe`.** A wrong exclusion loses a study for good; a
   wrong `maybe` only costs one full-text check.
2. **Personality not mentioned is not enough to exclude** if the study
   is about individual differences of online learners and the outcome
   fits ("learner characteristics", "individual differences",
   "psychological factors"): use `maybe`.
3. **No abstract:** decide from the title. Exclude only if the title
   alone clearly fails a criterion (for example a clinical study, a
   study of employees, a study of a non-personality construct with no
   learning outcome). Otherwise `maybe`.
4. **Online setting:** studies during COVID-19 emergency remote teaching
   count as online. Studies that compare online and face-to-face
   students are `include` or `maybe`. Exclude with E2 only when the
   title or abstract **states** that teaching was face to face only, or
   the study is not about learning at all. If a study of students'
   personality and an eligible outcome does not say how teaching was
   delivered, propose `maybe`: many abstracts omit the setting.
5. **Non-English:** exclude only when the title and abstract are not in
   English and nothing indicates an English full text.
6. Do not use knowledge of the study beyond the title, abstract, year
   and journal shown.

## Exclusion reason codes

| Code | Meaning |
|---|---|
| E1 | No Big Five or HEXACO trait measure (other construct, MBTI, single trait only, or personality not studied) |
| E2 | Not an online learning setting: stated face to face only, or not about learning |
| E3 | Population is not students (for example employees, patients, teachers) |
| E4 | No relevant outcome (neither achievement nor satisfaction, engagement, LMS behaviour, dropout or persistence) |
| E5 | Not a quantitative primary study (review, meta-analysis, qualitative only, commentary, editorial, case study, protocol) |
| E6 | Not in English |
| E7 | Not an eligible report type (preprint, unpublished manuscript, book review, erratum, abstract only) |

## Output format

One JSON object per record, one per line (JSON Lines), in the same order
as the input, with exactly these keys:

```
{"record_id": "R00001", "suggestion": "include|maybe|exclude", "reason": "E1..E7 or empty", "note": "one short sentence in English"}
```

- `reason` is empty for `include` and `maybe`.
- `note` says in a few words what decided it, for example "MOOC
  students; BFI; course grade" or "employees' job performance".
