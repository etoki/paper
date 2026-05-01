#!/usr/bin/env python3
"""
Automated hallucination checker for metaanalysis/paper_v2.

Runs T1, T2, T4, T6 (fully automatic tasks) and reports any mismatch
against canonical analysis results.

Half-automatic tasks (T3, T5, T7) print machine-detectable issues but
require Claude/human follow-up for context judgment.

Usage:
    python3 check_hallucinations.py                # All auto tasks
    python3 check_hallucinations.py --task t1      # Just T1
    python3 check_hallucinations.py --task t1,t2   # Multiple
    python3 check_hallucinations.py --all          # Include semi-auto

See ../HALLUCINATION_CHECK_PROTOCOL.md for task definitions.
"""

import argparse
import csv
import re
import sys
from pathlib import Path

ROOT = Path(__file__).parent.parent
ANALYSIS = ROOT / "analysis"
PAPER = ROOT / "paper_v2"
BUILD_DOCX = PAPER / "build_docx.py"
REFS_PY = PAPER / "references_data.py"
DEEP_NOTES = ROOT / "deep_reading_notes.md"

# ----------------------------------------------------------------------
# Helpers
# ----------------------------------------------------------------------
def fmt(x):
    """Format float as in pool.py output: -0.123 → −.123, 0.123 → .123"""
    s = f"{float(x):.3f}"
    if s.startswith("0."):
        return "." + s[2:]
    if s.startswith("-0."):
        return "−." + s[3:]
    return s


def fmt_str(s):
    """Format already-formatted string like '0.225' → '.225' or '-0.123' → '−.123'."""
    s = s.strip()
    if s.startswith("0."):
        return "." + s[2:]
    if s.startswith("-0."):
        return "−." + s[3:]
    return s


def parse_summary_md_pooled():
    """Parse pooling_summary.md for the canonical formatted r/CI/PI/I²/Q strings."""
    md = (ANALYSIS / "pooling_summary.md").read_text(encoding="utf-8")
    section = re.search(r"## Per-trait pooled effects.*?(?=##)", md, re.DOTALL).group(0)
    out = {}
    for line in section.split("\n"):
        m = re.match(r"\| \*\*(\w)\*\* \| (\d+) \| (\d+) \| (-?\d+\.\d+) \[(-?\d+\.\d+), (-?\d+\.\d+)\] \| \[(-?\d+\.\d+), (-?\d+\.\d+)\] \| (\d+\.\d+)% \| (\d+\.\d+) \| (\d+\.\d+)\((\d+)\), p=(\d+\.\d+) \|", line)
        if m:
            t, k, N, r, ci_lo, ci_hi, pi_lo, pi_hi, I2, tau2, Q, df, p = m.groups()
            out[t] = {
                "k": k, "N": int(N),
                "r": fmt_str(r), "ci_lo": fmt_str(ci_lo), "ci_hi": fmt_str(ci_hi),
                "pi_lo": fmt_str(pi_lo), "pi_hi": fmt_str(pi_hi),
                "I2": I2, "tau2": tau2, "Q": Q, "df": df, "p": p,
            }
    return out


def parse_summary_md_moderators():
    """Parse pooling_summary.md for moderator strings."""
    md = (ANALYSIS / "pooling_summary.md").read_text(encoding="utf-8")
    out = []
    # Match each moderator section
    for mod_match in re.finditer(r"### Moderator: (\w+)\s*\n.*?(?=###|##|\Z)", md, re.DOTALL):
        mod_name = mod_match.group(1)
        section = mod_match.group(0)
        current_trait = None
        for line in section.split("\n"):
            m = re.match(r"\| ([OCEAN]) \| (\S+(?:\s*\S+)*?) \| (\d+) \| (\d+|—) \| (-?\d+\.\d+) \[(-?\d+\.\d+), (-?\d+\.\d+)\] \| (\d+\.\d+)% \| (\d+\.\d+)\(\d+\), p=(\d+\.\d+) \|", line)
            if m:
                t, level, k, N, r, ci_lo, ci_hi, I2, Q, p = m.groups()
                current_trait = t
                out.append({
                    "mod": mod_name, "trait": t, "level": level.strip(), "k": k,
                    "r_str": f"{fmt_str(r)} [{fmt_str(ci_lo)}, {fmt_str(ci_hi)}]",
                    "Q": Q, "p": p,
                })
            else:
                # continuation row (no trait, no Q_b)
                m2 = re.match(r"\|\s+\| (\S+(?:\s*\S+)*?) \| (\d+) \| (\d+|—) \| (-?\d+\.\d+) \[(-?\d+\.\d+), (-?\d+\.\d+)\] \| (\d+\.\d+)% \|", line)
                if m2 and current_trait:
                    level, k, N, r, ci_lo, ci_hi, I2 = m2.groups()
                    out.append({
                        "mod": mod_name, "trait": current_trait, "level": level.strip(), "k": k,
                        "r_str": f"{fmt_str(r)} [{fmt_str(ci_lo)}, {fmt_str(ci_hi)}]",
                        "Q": None, "p": None,
                    })
    return out


def parse_sensitivity_md():
    """Parse sensitivity_results.md for canonical sensitivity values."""
    md = (ANALYSIS / "sensitivity_results.md").read_text(encoding="utf-8")
    sections = {}
    for sec_match in re.finditer(r"## (Exclude [^\n]+)\n(.*?)(?=##|\Z)", md, re.DOTALL):
        name = sec_match.group(1).strip()
        body = sec_match.group(2)
        rows = []
        for line in body.split("\n"):
            m = re.match(r"\| ([OCEAN]) \| (\d+) \| (-?\d+\.\d+) \[(-?\d+\.\d+), (-?\d+\.\d+)\] \| (.+?) \|", line)
            if m:
                t, k, r, ci_lo, ci_hi, dr = m.groups()
                rows.append({
                    "trait": t, "k": k,
                    "r_str": f"{fmt_str(r)} [{fmt_str(ci_lo)}, {fmt_str(ci_hi)}]",
                })
        sections[name] = rows
    return sections


def load_v2():
    return BUILD_DOCX.read_text(encoding="utf-8")


def load_v2_flat():
    """Flatten string concatenation in build_docx.py to recover the rendered
    manuscript text. Uses AST to reliably extract every string literal regardless
    of cross-line concatenation, docstrings, or quote style."""
    import ast
    src = BUILD_DOCX.read_text(encoding="utf-8")
    tree = ast.parse(src)
    strings = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Constant) and isinstance(node.value, str):
            strings.append(node.value)
    return " ".join(strings)


def load_pooling():
    with (ANALYSIS / "pooling_results.csv").open() as f:
        return {r["trait"]: r for r in csv.DictReader(f)}


def load_moderators():
    with (ANALYSIS / "moderator_results.csv").open() as f:
        return list(csv.DictReader(f))


def load_extractions():
    with (ANALYSIS / "data_extraction_populated.csv").open(encoding="utf-8") as f:
        return list(csv.DictReader(f))


def report(passed, failed):
    print()
    print("=" * 70)
    print(f"  PASSED: {passed}  |  FAILED: {failed}")
    print("=" * 70)
    return 1 if failed > 0 else 0


# ----------------------------------------------------------------------
# T1: Tables 2-5 numerical check
# ----------------------------------------------------------------------
def task_t1():
    print("\n" + "=" * 70)
    print("T1: Tables 2-5 numerical check")
    print("=" * 70)
    v2 = load_v2()
    passed = failed = 0

    # Table 2 — pooled effects (from pooling_summary.md)
    print("\n[Table 2 — Pooled effects]")
    pool = parse_summary_md_pooled()
    table2_section = re.search(r"def build_table2_pooled.*?def build_table3", v2, re.DOTALL)
    table2 = table2_section.group(0) if table2_section else ""
    for trait, r in pool.items():
        ci = f"[{r['ci_lo']}, {r['ci_hi']}]"
        pi = f"[{r['pi_lo']}, {r['pi_hi']}]"
        for label, val in [
            (f"{trait} k", r["k"]),
            (f"{trait} N", f"{r['N']:,}"),
            (f"{trait} r", r["r"]),
            (f"{trait} CI", ci),
            (f"{trait} PI", pi),
            (f"{trait} I²", f"{r['I2']}%"),
        ]:
            if val in table2:
                passed += 1
            else:
                print(f"  ❌ MISSING in Table 2: {label} = {val!r}")
                failed += 1

    # Table 3 — moderators (from pooling_summary.md)
    print("\n[Table 3 — Moderator subgroups]")
    mods = parse_summary_md_moderators()
    table3_section = re.search(r"def build_table3_moderators.*?def build_table4", v2, re.DOTALL)
    table3 = table3_section.group(0) if table3_section else ""
    for m in mods:
        if m["r_str"] in table3:
            passed += 1
        else:
            print(f"  ❌ MISSING in Table 3: {m['mod']} {m['trait']} {m['level']} = {m['r_str']!r}")
            failed += 1

    # Table 4 — sensitivity (from sensitivity_results.md)
    print("\n[Table 4 — Sensitivity analyses]")
    sens = parse_sensitivity_md()
    table4_section = re.search(r"def build_table4_sensitivity.*?def build_table5", v2, re.DOTALL)
    table4 = table4_section.group(0) if table4_section else ""
    for section_name, rows in sens.items():
        for r in rows:
            if r["r_str"] in table4:
                passed += 1
            else:
                print(f"  ❌ MISSING in Table 4: {section_name} {r['trait']} k={r['k']} = {r['r_str']!r}")
                failed += 1

    return report(passed, failed)


# ----------------------------------------------------------------------
# T2: Body text pooled effects check
# ----------------------------------------------------------------------
def task_t2():
    print("\n" + "=" * 70)
    print("T2: Body text pooled effects check")
    print("=" * 70)
    v2 = load_v2()
    passed = failed = 0
    pool = parse_summary_md_pooled()

    for trait_full, trait in [
        ("Conscientiousness", "C"),
        ("Openness", "O"),
        ("Extraversion", "E"),
        ("Agreeableness", "A"),
        ("Neuroticism", "N"),
    ]:
        r = pool[trait]
        r_str = r["r"]
        ci_str = f"[{r['ci_lo']}, {r['ci_hi']}]"
        # Each pooled value should appear at least once (Abstract + Results + Discussion)
        for label, val, min_count in [
            (f"{trait_full} r={r_str}", f"r = {r_str}", 1),
            (f"{trait_full} CI={ci_str}", ci_str, 1),
        ]:
            count = v2.count(val)
            if count >= min_count:
                passed += 1
            else:
                print(f"  ❌ Missing or insufficient: {label} (found {count} times)")
                failed += 1

    return report(passed, failed)


# ----------------------------------------------------------------------
# T3: Contributor claims (semi-auto)
# ----------------------------------------------------------------------
def task_t3():
    print("\n" + "=" * 70)
    print("T3: Contributor claims (machine portion)")
    print("=" * 70)
    print("MANUAL FOLLOWUP REQUIRED for context-dependent claims")
    v2 = load_v2()
    passed = failed = warn = 0

    # The 10 primary pool studies
    primary_pool = ["A-01", "A-02", "A-15", "A-22", "A-23", "A-28", "A-29", "A-30", "A-31", "A-37"]
    not_in_primary = ["A-03", "A-04", "A-05", "A-06", "A-07", "A-08", "A-09", "A-10", "A-11",
                      "A-12", "A-13", "A-16", "A-17", "A-18", "A-19", "A-20", "A-21", "A-24",
                      "A-25", "A-26", "A-27"]
    # β-converted contributors per pooling_summary
    beta_converted = ["Yu", "Kaspar"]
    not_beta_converted_in_primary = ["Mustafa", "Wang", "Audet"]  # have β but not in primary pool

    print("\n[Check 1: β-converted list mentions Mustafa/Wang etc. as primary pool contributor]")
    issues = []
    for word in not_beta_converted_in_primary:
        # Find sentences containing both "β" and the author name in the same paragraph
        for match in re.finditer(rf"\(.{{0,400}}{word}.{{0,400}}β-converted|β-converted.{{0,400}}{word}.{{0,400}}\)", v2):
            issues.append(f"  ⚠ Possible: '{word}' near 'β-converted' — verify context\n     {match.group(0)[:200]}")
    if issues:
        for i in issues:
            print(i)
            warn += 1
    else:
        print("  ✓ No suspicious co-occurrences")
        passed += 1

    print("\n[Check 2: 'driven by' / 'principally' mentions of non-primary studies as primary pool driver]")
    # Look for "driven by X" where X is one of not_in_primary studies
    for kw in ["driven principally by", "driven largely by", "drove the pooled", "principally"]:
        for match in re.finditer(rf"{kw}.{{0,300}}", v2, re.DOTALL):
            ctx = match.group(0)
            for s in ["Wang", "Mustafa", "Baruth", "Cohen", "Tokiwa"]:
                if s in ctx[:300]:
                    print(f"  ⚠ '{kw}' near '{s}' — verify primary-pool claim:")
                    print(f"     {ctx[:200]!r}")
                    warn += 1

    print("\n[Check 3: Leave-one-out shifts match canonical sensitivity_results.md]")
    sens_md = (ANALYSIS / "sensitivity_results.md").read_text(encoding="utf-8")
    # Canonical max influence: Yu in A (.112 → .038), Yu in E (.002 → .031)
    # If body text mentions 'shifted r from .112 to', the second number must be .038
    for trait, primary, expected_after in [("A", ".112", ".038"), ("E", ".002", ".031")]:
        m = re.search(rf"shifted r from \\{re.escape(primary)} to \\.(\\d+)", v2)
        if m:
            actual = "." + m.group(1)
            if actual == expected_after:
                print(f"  ✓ {trait}: shifted from {primary} to {actual}")
                passed += 1
            else:
                print(f"  ❌ {trait}: shifted from {primary} to {actual} but canonical = {expected_after}")
                failed += 1

    print(f"\n  passed={passed}, warn={warn}, failed={failed}")
    return 1 if failed > 0 else 0


# ----------------------------------------------------------------------
# T4: Counts / breakdowns check
# ----------------------------------------------------------------------
def task_t4():
    print("\n" + "=" * 70)
    print("T4: Counts and breakdowns check")
    print("=" * 70)
    v2 = load_v2()
    rows = load_extractions()
    passed = failed = 0

    retained = [r for r in rows if r["inclusion_status"].startswith("include") or
                r["inclusion_status"] == "exclude_from_primary"]
    not_in_synthesis = [r for r in rows if r["inclusion_status"] in
                        ("exclude", "exclude_overlap", "PDF_unavailable")]

    expected = {
        "31 catalogued": "31 catalogued",
        "25 retained": "25 retained",
        "K-12 (2 studies": "K-12 (2 studies",
        "undergraduate (15": "undergraduate (15",
        "mixed undergraduate/graduate (5": "mixed undergraduate/graduate (5",
        "graduate only (2": "graduate only (2",
        "Asia (12 studies": "Asia (12 studies",
        "Europe (7": "Europe (7",
        "North America (6": "North America (6",
        "8 pre-COVID, 12 COVID": "8 pre-COVID, 12 COVID",
        "5.44": "5.44",
        "SD = 0.87": "SD = 0.87",
        "Twenty-one studies": "Twenty-one studies",
    }
    for label, pat in expected.items():
        if pat in v2:
            passed += 1
        else:
            print(f"  ❌ Missing: '{label}'")
            failed += 1

    forbidden = {
        "31 included primary studies": "old v1 phrasing",
        "K-12 (3 studies: A-10": "old v1 K-12 list",
        "undergraduate (22 studies)": "old v1 UG count",
        "Asia (13 studies": "old v1 Asia",
        "North America (5 studies": "old v1 NA",
        "Twenty-four studies\nscored": "old v1 RoB count",
        "k = 2 studies explicitly post-COVID": "old k=2 hallucination",
    }
    for label, pat in forbidden.items():
        if pat in v2:
            print(f"  ❌ FORBIDDEN found: '{label}'")
            failed += 1
        else:
            passed += 1

    return report(passed, failed)


# ----------------------------------------------------------------------
# T5: References audit (semi-auto)
# ----------------------------------------------------------------------
def task_t5():
    print("\n" + "=" * 70)
    print("T5: References audit")
    print("=" * 70)
    v2 = load_v2_flat()  # use flattened text to handle cross-line citations
    passed = failed = warn = 0

    # Import the actual REFERENCES list (AST-equivalent) so multi-line
    # concatenated strings become single entries
    import importlib.util
    spec = importlib.util.spec_from_file_location("references_data", REFS_PY)
    rd = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(rd)

    # Build set of (first_author_surname, year) tuples from REFERENCES
    ref_first_authors = set()
    for entry in rd.REFERENCES:
        if not entry or not entry[0].isupper():
            continue
        m = re.match(r"^([\wÀ-ɏĀ-ſçñı\-]+)", entry)
        my = re.search(r"\((\d{4})\)", entry)
        if m and my:
            surname = m.group(1)
            surname_norm = (surname.replace("ı", "i").replace("ş", "s")
                            .replace("ç", "c").replace("ğ", "g")
                            .replace("ü", "u").replace("ö", "o").replace("ä", "a"))
            ref_first_authors.add((surname, my.group(1)))
            ref_first_authors.add((surname_norm, my.group(1)))

    # Find citations of form "FirstAuthor (YYYY)" or "FirstAuthor et al. (YYYY)"
    # or "FirstAuthor and SecondAuthor (YYYY)" — only FirstAuthor is the citation key.
    # Use negative lookbehind to skip second-authors preceded by "and"/"&"/", N.,"
    citations = set()
    citation_re = re.compile(
        r"(?<!and\s)(?<!\&\s)(?<!,\s)"  # not preceded by "and ", "& ", or ", "
        r"\b([A-ZÀ-Ɏ][a-zçñışığéüöäÀ-ɏĀ-ſ\-]{2,})"  # Author1
        r"(?:\s+et\s+al\.,?|"                          # " et al."
        r"\s+and\s+(?:[A-Z][a-zçñışığéüöäÀ-ɏĀ-ſ\-]+\s*)+|"  # " and Author2"
        r"\s*&\s*[A-Z][a-zçñışığéüöäÀ-ɏĀ-ſ\-]+|"           # " & Author2"
        r"(?:,\s+[A-Z]\.\s*)+(?:&|and)\s+[A-Z][a-zçñışığéüöäÀ-ɏĀ-ſ\-]+)?"  # ", X. & Y" or ", X. and Y"
        r",?\s*\((\d{4})\)"
    )
    skip = {
        "Conscientiousness", "Openness", "Extraversion", "Agreeableness", "Neuroticism",
        "Big", "Five", "Asian", "Europe", "Pearson", "Hartung", "Bayes",
        "European", "Note", "January", "Knapp", "Sidik", "Jonkman", "PRISMA",
        "Standards", "Records", "Of", "Eleven", "Author", "Acknowledgments",
        "Pre", "Preprint", "Conflict", "Funding", "Ethics", "Informed",
        "Availability", "Authors", "Hypothesis", "Hypotheses", "Supplementary",
        "Online", "MOOC", "Methods", "Results", "Discussion", "Introduction",
        "Limitations", "Strengths", "Future", "Practical", "Theoretical",
        "First", "Second", "Third", "Fourth", "Fifth", "Sixth", "Seventh",
        "Across", "Within", "Among", "Examination", "Several", "When",
        "While", "Although", "Notably", "Counterintuitively", "Critically",
        "Conversely", "Importantly", "Cohen", "Spearman"
    }
    for m in citation_re.finditer(v2):
        author, year = m.group(1), m.group(2)
        if author in skip:
            continue
        if not (1990 <= int(year) <= 2030):
            continue
        citations.add((author, year))

    # Check each citation has corresponding entry
    for author, year in sorted(citations):
        author_norm = (author.replace("ı", "i").replace("ş", "s")
                       .replace("ç", "c").replace("ğ", "g")
                       .replace("ü", "u").replace("ö", "o").replace("ä", "a"))
        if (author, year) in ref_first_authors or (author_norm, year) in ref_first_authors:
            passed += 1
        else:
            print(f"  ❌ Cited but not in references_data.py: {author} ({year})")
            failed += 1

    # Check every reference is cited at least once in body (flat).
    # Use one canonical surname per year (deduplicate diacritic variants)
    seen_years = {}
    for surname, year in ref_first_authors:
        # Prefer the original (with diacritics) over normalized
        if year not in seen_years or any(c not in "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ-" for c in surname):
            seen_years[year] = surname
    for year, surname in sorted(seen_years.items(), key=lambda x: x[1]):
        author_pat = re.escape(surname)
        author_norm = (surname.replace("ı", "i").replace("ş", "s")
                       .replace("ç", "c").replace("ğ", "g")
                       .replace("ü", "u").replace("ö", "o").replace("ä", "a"))
        if (re.search(rf"{author_pat}.{{0,200}}{year}", v2) or
            re.search(rf"{re.escape(author_norm)}.{{0,200}}{year}", v2)):
            passed += 1
        else:
            print(f"  ⚠ Reference present but not cited in body: {surname} ({year})")
            warn += 1

    print(f"\n  passed={passed}, warn={warn}, failed={failed}")
    return 1 if failed > 0 else 0


# ----------------------------------------------------------------------
# T6: PRISMA flow arithmetic
# ----------------------------------------------------------------------
def task_t6():
    print("\n" + "=" * 70)
    print("T6: PRISMA flow arithmetic")
    print("=" * 70)
    v2 = load_v2()
    rows = load_extractions()
    passed = failed = 0

    # Find PRISMA caption + flatten cross-line string concatenation
    # (Python source has manuscript text split across "..." "..." lines)
    m = re.search(r"def build_figure1_prisma.*?def _add_figure_image", v2, re.DOTALL)
    prisma_src = m.group(0) if m else v2
    # Extract all string literals and concatenate to recover the rendered text
    prisma = " ".join(re.findall(r'"([^"]*)"', prisma_src))

    # Check arithmetic claims
    checks = [
        ("38 full-text assessed", r"38 (?:were assessed|reports were assessed|assessed for eligibility)"),
        ("Eleven excluded (or 11)", r"Eleven\s+reports\s+were\s+excluded|11\s+reports\s+were\s+excluded"),
        ("non-Big-Five n=5", r"non-Big-Five\s+framework\s+\(n = 5"),
        ("face-to-face n=4", r"face-to-face\s+modality\s+\(n = 4"),
        ("overlap n=1", r"sample\s+overlap.{0,80}\(n = 1"),
        ("not extractable n=1", r"not\s+extractable\s+\(n = 1"),
        ("31 catalogued in review", r"31\s+primary\s+studies\s+catalogued"),
        ("25 retained for qualitative", r"25\s+were\s+retained\s+for\s+qualitative"),
        ("4 newly added", r"four\s+newly\s+identified"),
    ]
    for label, pat in checks:
        if re.search(pat, prisma, re.DOTALL):
            passed += 1
        else:
            print(f"  ❌ Missing in PRISMA caption: {label}")
            failed += 1

    # Verify CSV consistency
    n_excluded = sum(1 for r in rows if r["inclusion_status"] in
                     ("exclude", "exclude_overlap", "PDF_unavailable"))
    n_retained = len(rows) - n_excluded
    if n_retained == 25:
        print(f"  ✓ CSV check: retained=25 (CSV total {len(rows)} − {n_excluded} excluded)")
        passed += 1
    else:
        print(f"  ❌ CSV inconsistent: retained={n_retained} (should be 25)")
        failed += 1

    return report(passed, failed)


# ----------------------------------------------------------------------
# T7: Prior meta-analyses benchmark check
# ----------------------------------------------------------------------
def task_t7():
    print("\n" + "=" * 70)
    print("T7: Prior meta-analyses benchmark values")
    print("=" * 70)
    v2 = load_v2_flat()  # use flattened text to handle cross-line phrases
    passed = failed = 0

    # All values verified against deep_reading_notes.md
    benchmarks = [
        ("Poropat", r"138\s+samples", "k=138"),
        ("Poropat N", r"70,000", "N>70K"),
        ("McAbee k", r"k\s*=\s*57", "k=57"),
        ("McAbee N", r"N\s*=\s*26,382", "N=26,382"),
        ("Vedel k", r"k\s*=\s*21", "k=21"),
        ("Vedel N", r"N\s*=\s*17,717", "N=17,717"),
        ("Stajkovic β", r"β\s*=\s*\.24\s+to\s+\.33", "β SE→Perf .24-.33"),
        ("Mammadov samples", r"267\s+independent\s+samples", "k=267"),
        ("Mammadov N", r"413,074", "N=413,074"),
        ("Mammadov C ρ", r"ρ\s*=\s*\.27\s+for\s+Conscientiousness", "C ρ=.27"),
        ("Mammadov Asia C", r"C\s+ρ\s*=\s*\.35", "Asian C=.35"),
        ("Meyer samples", r"110\s+samples", "k=110"),
        ("Meyer N", r"500,218", "N=500,218"),
        ("Meyer K-12", r"K-12\s+samples", "K-12 only"),
        ("Zell k", r"54\s+meta-analyses", "k=54"),
        ("Zell C", r"C\s+ρ\s*=\s*\.28", "academic-specific C=.28"),
        ("Chen articles", r"84\s+articles", "k=84"),
        ("Chen correlations", r"370\s+independent\s+correlations", "k=370"),
        ("Chen C r", r"C\s+r\s*=\s*\.206", "C r=.206"),
        ("Hunter records", r"848\s+records", "Hunter 848"),
        ("Hunter studies", r"23\s+primary\s+studies", "k=23"),
    ]
    for label, pat, desc in benchmarks:
        cnt = len(re.findall(pat, v2))
        if cnt >= 1:
            passed += 1
        else:
            print(f"  ❌ Missing: {label} (expected: {desc})")
            failed += 1

    return report(passed, failed)


# ----------------------------------------------------------------------
# Main dispatcher
# ----------------------------------------------------------------------
TASKS = {
    "t1": task_t1,
    "t2": task_t2,
    "t3": task_t3,
    "t4": task_t4,
    "t5": task_t5,
    "t6": task_t6,
    "t7": task_t7,
}
AUTO = ["t1", "t2", "t4", "t6"]
SEMI_AUTO = ["t3", "t5", "t7"]


def main():
    parser = argparse.ArgumentParser(description="Hallucination checker for paper_v2")
    parser.add_argument("--task", default="auto",
                        help="Task IDs comma-separated (t1,t2,...) or 'auto' / 'all'")
    parser.add_argument("--all", action="store_true", help="Run all tasks (auto + semi-auto)")
    args = parser.parse_args()

    if args.all:
        tasks = AUTO + SEMI_AUTO
    elif args.task == "auto":
        tasks = AUTO
    elif args.task == "all":
        tasks = AUTO + SEMI_AUTO
    else:
        tasks = [t.strip().lower() for t in args.task.split(",")]

    print(f"\nRunning tasks: {tasks}")
    total_failed = 0
    for t in tasks:
        if t not in TASKS:
            print(f"\n⚠ Unknown task: {t}")
            continue
        rc = TASKS[t]()
        if rc != 0:
            total_failed += 1

    print("\n" + "=" * 70)
    if total_failed == 0:
        print("  🟢 ALL CHECKED TASKS PASSED")
    else:
        print(f"  🔴 {total_failed} TASK(S) FAILED — investigate above")
    print("=" * 70)
    sys.exit(total_failed)


if __name__ == "__main__":
    main()
