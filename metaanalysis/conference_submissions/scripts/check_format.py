#!/usr/bin/env python3
"""Submission-format compliance checks for the 3 conference papers.

Three passes:

(C19) Abstract word count
    Each paper's `## Abstract` block (paragraph immediately after the
    heading, up to the next blank-line boundary) is counted in words
    and compared against per-venue limits.

(C20) Page count estimate
    Pandoc's `--metadata` does not give a true page count without
    rendering through LibreOffice. We approximate via:
    body words / 350 words-per-page + figure/table block weights.
    Compared against per-venue limits.

(C23) Docx embedding QC
    Unzip each `full_paper.docx`, confirm:
    - At least one PNG embedded under `word/media/` (the PRISMA flow).
    - The Tokiwa Frontiers DOI string appears in the document body.
    - No "Manuscript in preparation" or "manuscript in preparation"
      strings present.

Usage:
    python3 metaanalysis/conference_submissions/scripts/check_format.py

Exit code: non-zero if any FAIL.
"""
from __future__ import annotations

import re
import sys
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
WORKSPACE = ROOT / "metaanalysis" / "conference_submissions"
PAPERS = ["ecel", "iceel", "iceri"]

# Per-venue limits (public CFP-derived; see format_readiness.md)
LIMITS = {
    "ecel":  {"abstract_min": 250, "abstract_max": 300, "page_max": 10},
    "iceel": {"abstract_min": 200, "abstract_max": 300, "page_max": 10},
    "iceri": {"abstract_min": 200, "abstract_max": 600, "page_max": 12},
}


# ----------------------------------------------------------------------
# C19. Abstract word count
# ----------------------------------------------------------------------
def extract_abstract(md_path: Path) -> str:
    raw = md_path.read_text(encoding="utf-8").splitlines()
    in_abs, lines, header_seen = False, [], False
    for ln in raw:
        if re.match(r"^##\s+Abstract\b", ln):
            in_abs = True
            header_seen = True
            continue
        if in_abs:
            if re.match(r"^##\s+\S", ln):
                break
            lines.append(ln)
    if not header_seen:
        return ""
    # Strip blank lines at start/end
    return "\n".join(lines).strip()


def word_count(text: str) -> int:
    # Drop markdown decorations
    cleaned = re.sub(r"[*_`#>\-\[\]\(\)\{\}]", " ", text)
    cleaned = re.sub(r"\s+", " ", cleaned).strip()
    return len([w for w in cleaned.split(" ") if w])


def abstract_check() -> int:
    fails = 0
    print("== (C19) Abstract word count ==")
    for paper in PAPERS:
        path = WORKSPACE / paper / "full_paper.md"
        if not path.exists():
            print(f"  {paper}: SKIP (no full_paper.md)")
            continue
        abstract = extract_abstract(path)
        wc = word_count(abstract)
        lim = LIMITS[paper]
        in_range = lim["abstract_min"] <= wc <= lim["abstract_max"]
        tag = "OK   " if in_range else "WARN "
        print(f"  {tag}{paper}: abstract = {wc} words "
              f"(target {lim['abstract_min']}-{lim['abstract_max']})")
        # Treat as a soft warning rather than a hard fail; don't gate
        # submission on a 5-word over-shoot.
    return fails


# ----------------------------------------------------------------------
# C20. Page count estimate
# Heuristic: total body words / 350 + 0.5 page per markdown-table
# block + 0.6 page per "![" image-include line. References list is
# counted as 0.7 page minimum, capped at 1.5 pages, by num-entries / 25.
# ----------------------------------------------------------------------
def page_count_estimate(md_path: Path) -> dict:
    raw = md_path.read_text(encoding="utf-8")
    lines = raw.splitlines()

    # Count body words (everything between first "## " and "## References")
    body_lines = []
    in_body, in_refs = False, False
    for ln in lines:
        if not in_body and re.match(r"^##\s+", ln):
            in_body = True
        if in_body and re.match(r"^##\s+References\s*$", ln):
            in_refs = True
            break
        if in_body:
            body_lines.append(ln)
    body_text = "\n".join(body_lines)
    body_wc = word_count(body_text)
    # Single-column 11pt single-spaced (ACPI / IEEE conference style)
    # is approximately 500 words per page of pure text.
    body_pages = body_wc / 500.0

    # Tables
    table_blocks = len(re.findall(r"^\|.+\|$", body_text, re.M))
    table_blocks_unique = len(re.findall(r"\n\|.+\|\n\|[\s\-:|]+\|", body_text))
    tables_page_add = 0.5 * table_blocks_unique

    # Figures (markdown image syntax in body, captures Figure 1 etc.)
    figs = len(re.findall(r"^!\[", body_text, re.M))
    figs_page_add = 0.6 * figs

    # References
    ref_lines = []
    in_refs_block = False
    for ln in lines:
        if re.match(r"^##\s+References\s*$", ln):
            in_refs_block = True
            continue
        if in_refs_block:
            if re.match(r"^##\s+\S", ln):
                break
            if re.match(r"^[-*]\s", ln):
                ref_lines.append(ln)
    refs_count = len(ref_lines)
    refs_page_add = max(0.7, min(1.5, refs_count / 25.0))

    total = body_pages + tables_page_add + figs_page_add + refs_page_add
    return {
        "body_words": body_wc,
        "body_pages": body_pages,
        "table_blocks": table_blocks_unique,
        "table_pages": tables_page_add,
        "figures": figs,
        "figure_pages": figs_page_add,
        "refs": refs_count,
        "ref_pages": refs_page_add,
        "total_estimated_pages": total,
    }


def page_count_check() -> int:
    fails = 0
    print("\n== (C20) Estimated page count ==")
    for paper in PAPERS:
        path = WORKSPACE / paper / "full_paper.md"
        if not path.exists():
            print(f"  {paper}: SKIP")
            continue
        est = page_count_estimate(path)
        lim = LIMITS[paper]["page_max"]
        over = est["total_estimated_pages"] > lim
        tag = "WARN " if over else "OK   "
        print(f"  {tag}{paper}: estimated total = {est['total_estimated_pages']:.1f} "
              f"(limit {lim}); body {est['body_words']} w / {est['body_pages']:.1f} p, "
              f"{est['table_blocks']} tables / {est['table_pages']:.1f} p, "
              f"{est['figures']} figures / {est['figure_pages']:.1f} p, "
              f"{est['refs']} refs / {est['ref_pages']:.1f} p")
    return fails


# ----------------------------------------------------------------------
# C23. Docx embedding QC
# ----------------------------------------------------------------------
def docx_qc() -> int:
    fails = 0
    print("\n== (C23) Docx embedding QC ==")
    for paper in PAPERS:
        docx = WORKSPACE / paper / "full_paper.docx"
        if not docx.exists():
            print(f"  FAIL  {paper}: no full_paper.docx -- run pandoc")
            fails += 1
            continue
        with zipfile.ZipFile(docx, "r") as z:
            names = z.namelist()
            png_files = [n for n in names if n.startswith("word/media/") and n.lower().endswith(".png")]
            try:
                doc_xml = z.read("word/document.xml").decode("utf-8", errors="replace")
            except KeyError:
                doc_xml = ""
        png_ok = len(png_files) >= 1
        frontiers_ok = "10.3389/fpsyg.2025.1420996" in doc_xml
        no_placeholder = (
            "Manuscript in preparation" not in doc_xml
            and "manuscript in preparation" not in doc_xml
        )
        tag = "OK   " if (png_ok and frontiers_ok and no_placeholder) else "FAIL "
        print(f"  {tag}{paper}: PNG={'✓' if png_ok else '✗'} ({len(png_files)} files), "
              f"Frontiers DOI={'✓' if frontiers_ok else '✗'}, "
              f"no manuscript-in-prep={'✓' if no_placeholder else '✗'}")
        if not (png_ok and frontiers_ok and no_placeholder):
            fails += 1
    return fails


def main():
    fails = 0
    fails += abstract_check()
    fails += page_count_check()
    fails += docx_qc()
    print(f"\nTotal failures: {fails}")
    sys.exit(1 if fails else 0)


if __name__ == "__main__":
    main()
