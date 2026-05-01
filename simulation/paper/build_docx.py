"""Build APA-formatted docx variants for the HEXACO Workplace Harassment Microsim paper.

Generates four output artifacts:
  - manuscript_preprint.docx        : single-file APA preprint (Times New Roman 12pt, double-spaced)
  - manuscript_journal.docx         : RSOS Stage 2 RR journal format (slightly different metadata block)
  - split/01_title_declarations.docx
  - split/02_body.docx               (Abstract + Introduction + Methods + Results + Discussion)
  - split/03_tables.docx             (Tables 1-6)
  - split/04_figures.docx            (Figure captions; figure files referenced separately)

Usage:
    python build_docx.py

Reads:
  - 01_intro.md, 02_methods.md, 03_results.md, 04_discussion.md, 05_refs.md, 06_tables_figures.md

All content sourced from markdown files; no inline duplication.
"""
from __future__ import annotations

import re
from pathlib import Path

from docx import Document
from docx.enum.style import WD_STYLE_TYPE
from docx.enum.text import WD_ALIGN_PARAGRAPH
from docx.oxml import OxmlElement
from docx.oxml.ns import qn
from docx.shared import Inches, Pt

# ============================================================================
# Paths and constants
# ============================================================================

PAPER_DIR = Path(__file__).resolve().parent
SPLIT_DIR = PAPER_DIR / "split"
SPLIT_DIR.mkdir(exist_ok=True)

OUT_PREPRINT = PAPER_DIR / "manuscript_preprint.docx"
OUT_JOURNAL = PAPER_DIR / "manuscript_journal.docx"
OUT_SPLIT_01 = SPLIT_DIR / "01_title_declarations.docx"
OUT_SPLIT_02 = SPLIT_DIR / "02_body.docx"
OUT_SPLIT_03 = SPLIT_DIR / "03_tables.docx"
OUT_SPLIT_04 = SPLIT_DIR / "04_figures.docx"

MD_INTRO = PAPER_DIR / "01_intro.md"
MD_METHODS = PAPER_DIR / "02_methods.md"
MD_RESULTS = PAPER_DIR / "03_results.md"
MD_DISCUSSION = PAPER_DIR / "04_discussion.md"
MD_REFS = PAPER_DIR / "05_refs.md"
MD_TABLES_FIGURES = PAPER_DIR / "06_tables_figures.md"

TITLE = (
    "Person-Level versus System-Level Anti-Harassment Interventions: "
    "HEXACO 7-Typology Evidence That Structure Dominates Personality "
    "in Japanese Workplaces"
)
AUTHOR = "Eisuke Tokiwa"
ORCID = "0009-0009-7124-6669"
AFFILIATION = "SUNBLAZE Co., Ltd., Tokyo, Japan"
EMAIL = "eisuke.tokiwa@sunblaze.jp"
OSF_DOI = "10.17605/OSF.IO/3Y54U"
OSF_URL = "https://osf.io/3y54u"
GITHUB_URL = "https://github.com/etoki/paper"


# ============================================================================
# Style helpers (APA 7th edition; Times New Roman 12pt, double-spaced, 1" margins)
# ============================================================================


def set_font(run, name: str = "Times New Roman", size: int = 12, bold: bool | None = None,
             italic: bool | None = None):
    run.font.name = name
    run.font.size = Pt(size)
    if bold is not None:
        run.bold = bold
    if italic is not None:
        run.italic = italic
    rPr = run._element.get_or_add_rPr()
    rFonts = rPr.find(qn("w:rFonts"))
    if rFonts is None:
        rFonts = OxmlElement("w:rFonts")
        rPr.append(rFonts)
    rFonts.set(qn("w:eastAsia"), name)


def set_double_space(p):
    pf = p.paragraph_format
    pf.line_spacing = 2.0
    pf.space_before = Pt(0)
    pf.space_after = Pt(0)


def add_para(doc: Document, text: str, *, bold: bool = False, italic: bool = False,
             align: int | None = None, indent_first: bool = False, size: int = 12,
             style: str | None = None) -> "Paragraph":
    p = doc.add_paragraph(style=style) if style else doc.add_paragraph()
    if align is not None:
        p.alignment = align
    if indent_first:
        p.paragraph_format.first_line_indent = Inches(0.5)
    set_double_space(p)
    if text:
        run = p.add_run(text)
        set_font(run, size=size, bold=bold or None, italic=italic or None)
    return p


def add_heading(doc: Document, text: str, level: int = 1):
    """APA-style heading (Level 1 = bold centered; Level 2 = bold left; Level 3 = bold italic indent)."""
    p = doc.add_paragraph()
    set_double_space(p)
    if level == 1:
        p.alignment = WD_ALIGN_PARAGRAPH.CENTER
        run = p.add_run(text)
        set_font(run, size=12, bold=True)
    elif level == 2:
        p.alignment = WD_ALIGN_PARAGRAPH.LEFT
        run = p.add_run(text)
        set_font(run, size=12, bold=True)
    else:  # level >= 3
        p.alignment = WD_ALIGN_PARAGRAPH.LEFT
        p.paragraph_format.first_line_indent = Inches(0.5)
        run = p.add_run(text + ".")
        set_font(run, size=12, bold=True, italic=True)
    return p


def configure_styles(doc: Document):
    """Set default Normal style to APA 7th: Times New Roman 12pt, double-spaced."""
    normal = doc.styles["Normal"]
    normal.font.name = "Times New Roman"
    normal.font.size = Pt(12)
    rPr = normal.element.get_or_add_rPr()
    rFonts = rPr.find(qn("w:rFonts"))
    if rFonts is None:
        rFonts = OxmlElement("w:rFonts")
        rPr.append(rFonts)
    rFonts.set(qn("w:eastAsia"), "Times New Roman")
    sections = doc.sections
    for section in sections:
        section.top_margin = Inches(1.0)
        section.bottom_margin = Inches(1.0)
        section.left_margin = Inches(1.0)
        section.right_margin = Inches(1.0)


# ============================================================================
# Markdown → docx conversion (lightweight, paper-specific)
# ============================================================================

# Lines we want to skip (markdown front-matter, our internal notes)
SKIP_PATTERNS = [
    re.compile(r"^\s*---\s*$"),  # horizontal rules
]


def parse_md_blocks(md_path: Path) -> list[tuple[str, str]]:
    """Yield (block_type, text) tuples from a markdown file.

    Block types: h1, h2, h3, h4, para, table, list_item, blockquote.
    Tables are returned as a single multi-line string for downstream parsing.
    """
    text = md_path.read_text(encoding="utf-8")
    lines = text.split("\n")
    blocks: list[tuple[str, str]] = []
    i = 0
    while i < len(lines):
        line = lines[i]
        if any(p.match(line) for p in SKIP_PATTERNS):
            i += 1
            continue
        if line.startswith("#### "):
            blocks.append(("h4", line[5:].strip()))
            i += 1
        elif line.startswith("### "):
            blocks.append(("h3", line[4:].strip()))
            i += 1
        elif line.startswith("## "):
            blocks.append(("h2", line[3:].strip()))
            i += 1
        elif line.startswith("# "):
            blocks.append(("h1", line[2:].strip()))
            i += 1
        elif line.startswith("|") and "|" in line[1:]:
            # Markdown table: collect contiguous lines starting with |
            tbl_lines = [line]
            j = i + 1
            while j < len(lines) and lines[j].startswith("|"):
                tbl_lines.append(lines[j])
                j += 1
            blocks.append(("table", "\n".join(tbl_lines)))
            i = j
        elif line.startswith("> "):
            blocks.append(("blockquote", line[2:].strip()))
            i += 1
        elif line.startswith("- ") or line.startswith("* "):
            # List item; group consecutive
            list_lines = [line[2:]]
            j = i + 1
            while j < len(lines) and (lines[j].startswith("- ") or lines[j].startswith("* ")):
                list_lines.append(lines[j][2:])
                j += 1
            for li in list_lines:
                blocks.append(("list_item", li.strip()))
            i = j
        elif line.strip() == "":
            i += 1
        else:
            # Plain paragraph; collect until blank line / structural marker
            para_lines = [line]
            j = i + 1
            while j < len(lines):
                nxt = lines[j]
                if nxt.strip() == "" or nxt.startswith("#") or nxt.startswith("|") or nxt.startswith("- ") or nxt.startswith("> "):
                    break
                para_lines.append(nxt)
                j += 1
            blocks.append(("para", " ".join(line.strip() for line in para_lines)))
            i = j
    return blocks


def render_inline(run, text: str):
    """Apply minimal inline formatting (**bold**, *italic*) to a single docx run.

    Note: For multi-style runs we'd need separate runs; here we choose simplest
    matching since most paragraphs are plain text in our content.
    """
    # Strip simple markdown inline markers — APA gets bold/italic via font props
    # If a paragraph is wholly bold (entirely wrapped in ** **), apply bold
    text = text.strip()
    if text.startswith("**") and text.endswith("**") and text.count("**") == 2:
        run.text = text[2:-2]
        run.bold = True
    elif text.startswith("*") and text.endswith("*") and text.count("*") == 2:
        run.text = text[1:-1]
        run.italic = True
    else:
        # Strip any remaining markers but preserve content
        text = re.sub(r"\*\*([^\*]+)\*\*", r"\1", text)
        text = re.sub(r"\*([^\*]+)\*", r"\1", text)
        text = re.sub(r"`([^`]+)`", r"\1", text)
        run.text = text


def render_table(doc: Document, table_md: str):
    """Render a markdown pipe table as a native docx table with APA borders."""
    rows = [r for r in table_md.split("\n") if r.strip()]
    # Strip outer pipes and split each row
    parsed_rows: list[list[str]] = []
    for row in rows:
        cells = [c.strip() for c in row.strip().strip("|").split("|")]
        parsed_rows.append(cells)
    # Detect separator row (---|---|---) and skip it
    parsed_rows = [r for r in parsed_rows if not all(re.match(r"^[-:\s]+$", c) for c in r)]
    if not parsed_rows:
        return
    n_cols = len(parsed_rows[0])
    table = doc.add_table(rows=len(parsed_rows), cols=n_cols)
    for i, cells in enumerate(parsed_rows):
        for j, cell in enumerate(cells[:n_cols]):
            tcell = table.cell(i, j)
            # Replace existing paragraph text
            p = tcell.paragraphs[0]
            for run in list(p.runs):
                run._element.getparent().remove(run._element)
            run = p.add_run()
            render_inline(run, cell)
            set_font(run, size=11, bold=(i == 0))
    add_para(doc, "")  # spacer after table


def render_blocks(doc: Document, blocks: list[tuple[str, str]], *, title_seen: bool = True):
    for kind, text in blocks:
        if kind == "h1":
            if not title_seen:
                title_seen = True
                add_heading(doc, text, level=1)
            else:
                add_heading(doc, text, level=2)
        elif kind == "h2":
            add_heading(doc, text, level=2)
        elif kind == "h3":
            add_heading(doc, text, level=3)
        elif kind == "h4":
            add_heading(doc, text, level=3)
        elif kind == "para":
            add_para(doc, text, indent_first=True)
        elif kind == "list_item":
            p = doc.add_paragraph(style="List Bullet")
            run = p.add_run()
            render_inline(run, text)
            set_font(run, size=12)
            set_double_space(p)
        elif kind == "blockquote":
            p = doc.add_paragraph()
            p.paragraph_format.left_indent = Inches(0.5)
            run = p.add_run()
            render_inline(run, text)
            set_font(run, size=12, italic=True)
            set_double_space(p)
        elif kind == "table":
            render_table(doc, text)


# ============================================================================
# Title page builder (APA + journal variants)
# ============================================================================


def build_title_page(doc: Document, *, journal_variant: bool = False):
    # Title (centered, bold, ~ middle of page)
    for _ in range(2):
        add_para(doc, "")  # spacer
    p = add_para(doc, TITLE, bold=True, align=WD_ALIGN_PARAGRAPH.CENTER)
    add_para(doc, "")
    # Author block
    add_para(doc, AUTHOR, align=WD_ALIGN_PARAGRAPH.CENTER)
    add_para(doc, AFFILIATION, align=WD_ALIGN_PARAGRAPH.CENTER, italic=True)
    add_para(doc, f"ORCID: {ORCID}", align=WD_ALIGN_PARAGRAPH.CENTER)
    add_para(doc, "")
    add_para(doc, "Author note", bold=True, align=WD_ALIGN_PARAGRAPH.CENTER)
    add_para(doc,
             f"Correspondence concerning this article should be addressed to {AUTHOR}, "
             f"{AFFILIATION}. Email: {EMAIL}", indent_first=True)
    add_para(doc,
             f"Pre-registration: OSF DOI {OSF_DOI} ({OSF_URL}). "
             f"Code, data, and intermediate artifacts are publicly archived "
             f"under MIT license at {GITHUB_URL}.",
             indent_first=True)
    if journal_variant:
        add_para(doc, "")
        add_para(doc,
                 "Submitted to: Royal Society Open Science (Stage 2 Registered Report). "
                 "This manuscript implements the analysis pre-registered as v2.0 at OSF "
                 f"({OSF_URL}). Conditional acceptance based on Stage 1 review of the "
                 f"v2.0 protocol document is acknowledged in the cover letter.",
                 indent_first=True)
    add_para(doc, "")
    # Page break
    doc.add_page_break()


def build_declarations_page(doc: Document):
    add_heading(doc, "Declarations", level=1)
    add_para(doc, "Funding", bold=True)
    add_para(doc, "This research was self-funded by SUNBLAZE Co., Ltd. No external funding was received.",
             indent_first=True)
    add_para(doc, "Conflicts of interest", bold=True)
    add_para(doc, "The author declares no competing interests.", indent_first=True)
    add_para(doc, "Author contributions", bold=True)
    add_para(doc,
             "The author (sole-authored) was responsible for conceptualization, methodology, software, "
             "formal analysis, investigation, data curation, writing — original draft, and writing — "
             "review & editing.", indent_first=True)
    add_para(doc, "Data and code availability", bold=True)
    add_para(doc,
             f"All code, data, and intermediate artifacts are publicly archived under MIT license at "
             f"{GITHUB_URL} (directory: simulation/) and at OSF DOI {OSF_DOI} ({OSF_URL}). "
             "Reproducibility is verified via SHA-256 hash comparison against the v2.0 OSF registration.",
             indent_first=True)
    add_para(doc, "Ethics statement", bold=True)
    add_para(doc,
             "The N=354 individual-level harassment data analyzed in this study were collected under "
             "an IRB-approved protocol (Tokiwa, 2025, harassment preprint). Analysis of de-identified "
             "data does not require additional ethics review under SUNBLAZE Co., Ltd. policy.",
             indent_first=True)
    doc.add_page_break()


# ============================================================================
# Document builders
# ============================================================================


def build_preprint(out_path: Path, *, journal_variant: bool = False):
    """Single-file APA preprint (or journal variant)."""
    doc = Document()
    configure_styles(doc)
    build_title_page(doc, journal_variant=journal_variant)
    build_declarations_page(doc)

    # Body: Abstract + Introduction + Methods + Results + Discussion
    for md_path in [MD_INTRO, MD_METHODS, MD_RESULTS, MD_DISCUSSION]:
        blocks = parse_md_blocks(md_path)
        # Skip top-level "01. ..." heading and our metadata; start from Abstract / first H2
        filtered = []
        skip_metadata = True
        for kind, text in blocks:
            if skip_metadata and kind == "para" and (text.startswith("**") or "OSF DOI" in text or "Working title" in text):
                continue
            if skip_metadata and kind == "h1":
                # Drop the file's "01. Abstract + Introduction" wrapper
                continue
            skip_metadata = False
            filtered.append((kind, text))
        render_blocks(doc, filtered)
        doc.add_page_break()

    # References
    add_heading(doc, "References", level=1)
    refs_blocks = parse_md_blocks(MD_REFS)
    # Skip the file's own "# 05. References ..." header
    refs_filtered = [(k, t) for k, t in refs_blocks if not (k == "h1" and "References" in t)]
    for kind, text in refs_filtered:
        if kind == "para":
            # Hanging indent for references
            p = doc.add_paragraph()
            p.paragraph_format.left_indent = Inches(0.5)
            p.paragraph_format.first_line_indent = Inches(-0.5)
            set_double_space(p)
            run = p.add_run()
            render_inline(run, text)
            set_font(run, size=12)
        elif kind == "h2" or kind == "h3":
            add_heading(doc, text, level=2 if kind == "h2" else 3)
        elif kind == "list_item":
            # Treat as reference item with hanging indent
            p = doc.add_paragraph()
            p.paragraph_format.left_indent = Inches(0.5)
            p.paragraph_format.first_line_indent = Inches(-0.5)
            set_double_space(p)
            run = p.add_run()
            render_inline(run, text)
            set_font(run, size=12)
    doc.add_page_break()

    # Tables
    add_heading(doc, "Tables", level=1)
    tf_blocks = parse_md_blocks(MD_TABLES_FIGURES)
    in_tables = False
    in_figures = False
    for kind, text in tf_blocks:
        if kind == "h2" and text.strip().lower().startswith("tables"):
            in_tables = True
            in_figures = False
            continue
        if kind == "h2" and text.strip().lower().startswith("figures"):
            in_tables = False
            in_figures = True
            add_heading(doc, "Figures", level=1)
            continue
        if kind == "h2" and "generation notes" in text.lower():
            in_tables = False
            in_figures = False
            continue
        if in_tables or in_figures:
            if kind == "h3":
                add_heading(doc, text, level=2)
            elif kind == "table":
                render_table(doc, text)
            elif kind == "para":
                add_para(doc, text, italic=text.lower().startswith("note") or text.lower().startswith("caption"))
            elif kind == "list_item":
                p = doc.add_paragraph(style="List Bullet")
                run = p.add_run()
                render_inline(run, text)
                set_font(run, size=12)
                set_double_space(p)

    doc.save(str(out_path))
    print(f"  Wrote {out_path}")


def build_split_01(out_path: Path):
    doc = Document()
    configure_styles(doc)
    build_title_page(doc)
    build_declarations_page(doc)
    doc.save(str(out_path))
    print(f"  Wrote {out_path}")


def build_split_02(out_path: Path):
    doc = Document()
    configure_styles(doc)
    for md_path in [MD_INTRO, MD_METHODS, MD_RESULTS, MD_DISCUSSION]:
        blocks = parse_md_blocks(md_path)
        filtered = []
        skip_metadata = True
        for kind, text in blocks:
            if skip_metadata and (kind == "h1" or (kind == "para" and ("OSF DOI" in text or "Working title" in text or text.startswith("**Author**") or text.startswith("**Pre-registration**") or text.startswith("**Reporting standard**")))):
                continue
            skip_metadata = False
            filtered.append((kind, text))
        render_blocks(doc, filtered)
        doc.add_page_break()
    # References live in the body for split-02
    add_heading(doc, "References", level=1)
    refs_blocks = parse_md_blocks(MD_REFS)
    refs_filtered = [(k, t) for k, t in refs_blocks if not (k == "h1" and "References" in t)]
    for kind, text in refs_filtered:
        if kind == "para":
            p = doc.add_paragraph()
            p.paragraph_format.left_indent = Inches(0.5)
            p.paragraph_format.first_line_indent = Inches(-0.5)
            set_double_space(p)
            run = p.add_run()
            render_inline(run, text)
            set_font(run, size=12)
        elif kind == "h2" or kind == "h3":
            add_heading(doc, text, level=2 if kind == "h2" else 3)
    doc.save(str(out_path))
    print(f"  Wrote {out_path}")


def build_split_03(out_path: Path):
    """Tables only."""
    doc = Document()
    configure_styles(doc)
    add_para(doc, TITLE, bold=True, align=WD_ALIGN_PARAGRAPH.CENTER)
    add_para(doc, "Tables", bold=True, align=WD_ALIGN_PARAGRAPH.CENTER)
    add_para(doc, "")
    tf_blocks = parse_md_blocks(MD_TABLES_FIGURES)
    in_tables = False
    for kind, text in tf_blocks:
        if kind == "h2" and text.strip().lower().startswith("tables"):
            in_tables = True
            continue
        if kind == "h2" and (text.strip().lower().startswith("figures") or "generation" in text.lower()):
            in_tables = False
            continue
        if in_tables:
            if kind == "h3":
                add_heading(doc, text, level=2)
            elif kind == "table":
                render_table(doc, text)
            elif kind == "para":
                add_para(doc, text, italic=text.lower().startswith("note"))
    doc.save(str(out_path))
    print(f"  Wrote {out_path}")


def build_split_04(out_path: Path):
    """Figure captions only (figure files referenced separately)."""
    doc = Document()
    configure_styles(doc)
    add_para(doc, TITLE, bold=True, align=WD_ALIGN_PARAGRAPH.CENTER)
    add_para(doc, "Figures (captions)", bold=True, align=WD_ALIGN_PARAGRAPH.CENTER)
    add_para(doc, "")
    tf_blocks = parse_md_blocks(MD_TABLES_FIGURES)
    in_figures = False
    for kind, text in tf_blocks:
        if kind == "h2" and text.strip().lower().startswith("figures"):
            in_figures = True
            continue
        if kind == "h2" and "generation" in text.lower():
            in_figures = False
            continue
        if in_figures:
            if kind == "h3":
                add_heading(doc, text, level=2)
            elif kind == "para":
                add_para(doc, text, italic=text.lower().startswith("caption"))
    doc.save(str(out_path))
    print(f"  Wrote {out_path}")


def main():
    print("[build_docx] HEXACO Workplace Harassment Microsim manuscript builder")
    print(f"  Source markdown: {PAPER_DIR}/0[1-6]_*.md")
    print(f"  Output preprint: {OUT_PREPRINT}")
    print(f"  Output journal: {OUT_JOURNAL}")
    print(f"  Split outputs: {SPLIT_DIR}/")
    print()
    build_preprint(OUT_PREPRINT, journal_variant=False)
    build_preprint(OUT_JOURNAL, journal_variant=True)
    build_split_01(OUT_SPLIT_01)
    build_split_02(OUT_SPLIT_02)
    build_split_03(OUT_SPLIT_03)
    build_split_04(OUT_SPLIT_04)
    print()
    print("[build_docx] Done.")


if __name__ == "__main__":
    main()
