"""
Apply hallucination-fix patches to harassment/ manuscript .docx files.

Fixes the 4 findings from HALLUCINATION_REPORT.md:

1. Honesty–Humility α typo: .671 → .571 (Methods → Measurement Instruments)
2. Surname spelling typo: Saltuküğlu → Saltukoğlu (body)
3. Add 13 missing reference entries to the reference list, in alphabetical
   order. For "Breevaart & de Vries, 2019" — which has no matching paper —
   we instead change the body citation year to 2017 (which IS in the ref
   list and matches the Honesty-Humility / abusive supervision theme).
4. Remove the unused Jones & Paulhus (2011) entry from the reference list.

Plus T9 (live Crossref) follow-up:

5. Vize et al. (2016) → (2018): per APA 7 §10.1 example 47 the print year
   takes precedence over the advance-online date for periodicals. The
   article appeared online in 2016 but was assigned to *Personality
   Disorders* vol 9 issue 2 in 2018. Updates both the body citation and
   the reference-list entry.

Run:
    python3 harassment/apply_hallucination_fixes.py
"""
from __future__ import annotations
import io
import re
import shutil
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent
PAPER = ROOT / "paper"

# ---------------------------------------------------------------------- #
# 13 missing references, formatted as docx <w:p> XML.                   #
# Each entry is "<sort_key>" → ("<reference text>",).                   #
# Alphabetical sort_key: surname plus year for stable ordering.         #
# ---------------------------------------------------------------------- #
NEW_REFS: dict[str, str] = {
    "Babiak 2006": (
        "Babiak, P., & Hare, R. D. (2006). Snakes in suits: When psychopaths "
        "go to work. Regan Books / HarperCollins."
    ),
    "Bass 1990": (
        "Bass, B. M. (1990). From transactional to transformational "
        "leadership: Learning to share the vision. Organizational Dynamics, "
        "18(3), 19–31. https://doi.org/10.1016/0090-2616(90)90061-S"
    ),
    "Cherniss 2001": (
        "Cherniss, C., & Goleman, D. (Eds.). (2001). The emotionally "
        "intelligent workplace: How to select for, measure, and improve "
        "emotional intelligence in individuals, groups, and organizations. "
        "Jossey-Bass."
    ),
    "Christie 1970": (
        "Christie, R., & Geis, F. L. (1970). Studies in Machiavellianism. "
        "Academic Press."
    ),
    "Cleckley 1941": (
        "Cleckley, H. (1941). The mask of sanity: An attempt to clarify "
        "some issues about the so-called psychopathic personality. C. V. "
        "Mosby."
    ),
    "Fehr 1992": (
        "Fehr, B., Samsom, D., & Paulhus, D. L. (1992). The construct of "
        "Machiavellianism: Twenty years later. In C. D. Spielberger & "
        "J. N. Butcher (Eds.), Advances in personality assessment (Vol. 9, "
        "pp. 77–116). Lawrence Erlbaum."
    ),
    "Gandolfi 2017": (
        "Gandolfi, F., Stone, S., & Deno, F. (2017). Servant leadership: "
        "An ancient style with 21st century relevance. Review of "
        "International Comparative Management, 18(4), 350–361."
    ),
    "Hare 2003": (
        "Hare, R. D. (2003). Manual for the Hare Psychopathy Checklist–"
        "Revised (2nd ed.). Multi-Health Systems."
    ),
    "Jones 2009": (
        "Jones, D. N., & Paulhus, D. L. (2009). Machiavellianism. In M. R. "
        "Leary & R. H. Hoyle (Eds.), Handbook of individual differences in "
        "social behavior (pp. 93–108). Guilford Press."
    ),
    "Jones 2010": (
        "Jones, D. N., & Paulhus, D. L. (2010). Different provocations "
        "trigger aggression in narcissists and psychopaths. Social "
        "Psychological and Personality Science, 1(1), 12–18. "
        "https://doi.org/10.1177/1948550609347591"
    ),
    "Lynam 2006": (
        "Lynam, D. R., & Derefinko, K. J. (2006). Psychopathy and "
        "personality. In C. J. Patrick (Ed.), Handbook of psychopathy "
        "(pp. 133–155). Guilford Press."
    ),
    "Paulhus 2001": (
        "Paulhus, D. L. (2001). Normal narcissism: Two minimalist accounts. "
        "Psychological Inquiry, 12(4), 228–230."
    ),
    "Zuckerman 1994": (
        "Zuckerman, M. (1994). Behavioral expressions and biosocial bases "
        "of sensation seeking. Cambridge University Press."
    ),
}


def _xml_escape(text: str) -> str:
    return (
        text.replace("&", "&amp;")
        .replace("<", "&lt;")
        .replace(">", "&gt;")
    )


def _make_ref_paragraph(text: str) -> str:
    """Build a docx paragraph for a reference entry."""
    return (
        '<w:p><w:pPr><w:ind w:left="720" w:hanging="720"/></w:pPr>'
        '<w:r><w:t xml:space="preserve">'
        + _xml_escape(text)
        + "</w:t></w:r></w:p>"
    )


# ---------------------------------------------------------------------- #
# XML transforms                                                         #
# ---------------------------------------------------------------------- #
def fix_alpha(xml: str) -> tuple[str, bool]:
    """Change H-H α from .671 to .571.

    The "6" in ".671" is in its own <w:r> run because of an earlier
    track-change/edit. We replace exactly that run's "<w:t>6</w:t>"
    with "<w:t>5</w:t>" — but only when followed by the matching
    "71 for Honesty–Humility" run.
    """
    pattern = re.compile(
        r'(coefficients were α = \.</w:t></w:r>'
        r'<w:r[^>]*?w:rsidR="006A2B90"[^>]*?>'
        r'<w:rPr>[^<]*?(?:<[^>]+>)*?</w:rPr>'
        r'<w:t>)6(</w:t></w:r>'
        r'<w:r[^>]*?>'
        r'<w:rPr>[^<]*?(?:<[^>]+>)*?</w:rPr>'
        r'<w:t>71 for Honesty)',
        re.DOTALL,
    )
    new_xml, n = pattern.subn(r'\g<1>5\g<2>', xml)
    if n == 0:
        # Fallback: looser pattern - just locate the specific "<w:t>6</w:t>"
        # near the alpha context.
        anchor = "<w:t>71 for Honesty"
        idx = new_xml.find(anchor)
        if idx < 0:
            return new_xml, False
        # Look backwards for "<w:t>6</w:t>"
        win = new_xml[max(0, idx - 400):idx]
        if "<w:t>6</w:t>" in win:
            # Replace last occurrence only
            j = win.rfind("<w:t>6</w:t>")
            abs_pos = max(0, idx - 400) + j
            new_xml = new_xml[:abs_pos] + "<w:t>5</w:t>" + new_xml[abs_pos + len("<w:t>6</w:t>"):]
            return new_xml, True
        return new_xml, False
    return new_xml, True


def fix_saltukoglu(xml: str) -> tuple[str, bool]:
    """Body misspells Saltuküğlu — change to Saltukoğlu (matches reference)."""
    if "Saltuküğlu" in xml:
        return xml.replace("Saltuküğlu", "Saltukoğlu"), True
    return xml, False


def fix_breevaart_year(xml: str) -> tuple[str, int]:
    """Replace body citations Breevaart & de Vries, 2019 → 2017.

    The reference list has 2017 (abusive supervision) and 2021 (followers'
    leadership preferences). No 2019 paper exists; the body intent fits
    2017 best.

    Body citations span runs in the XML, so we anchor on the literal
    "&amp; de Vries, 2019" string. Reference-list lines use the form
    ", R. E. (2019)" with parentheses around the year, so they would
    not match this anchor (in fact the ref list contains no 2019 entry
    for this author pair).
    """
    target = "&amp; de Vries, 2019"
    repl = "&amp; de Vries, 2017"
    n = xml.count(target)
    return xml.replace(target, repl), n


def fix_vize_year(xml: str) -> tuple[str, int]:
    """Replace Vize et al., 2016 → 2018 in both body and reference list.

    The DOI 10.1037/per0000222 resolved on Crossref to print year 2018
    (Personality Disorders vol 9 issue 2); the local "(2016)" reflected
    the advance-online date. APA 7 §10.1 prefers the final/print year
    once it is available.

    Two literal anchors are sufficient:
      - body in-text:    "Vize et al., 2016"
      - reference list:  "Miller, J. D. (2016)" (the unique tail of the
                         Vize et al. author block, which guarantees we
                         only touch the Vize entry's year and not any
                         other Miller-2016 occurrence).
    Idempotent: after the first run neither anchor exists.
    """
    body_target = "Vize et al., 2016"
    body_repl = "Vize et al., 2018"
    refs_target = "Miller, J. D. (2016). Differences among Dark Triad"
    refs_repl = "Miller, J. D. (2018). Differences among Dark Triad"
    n_body = xml.count(body_target)
    n_refs = xml.count(refs_target)
    new_xml = xml.replace(body_target, body_repl).replace(
        refs_target, refs_repl
    )
    return new_xml, n_body + n_refs


def remove_jones_2011(xml: str) -> tuple[str, bool]:
    """Remove the Jones, D. N., & Paulhus, D. L. (2011)... reference paragraph."""
    anchor = "Jones, D. N., &amp; Paulhus, D. L. (2011)"
    idx = xml.find(anchor)
    if idx < 0:
        return xml, False
    # Find the enclosing <w:p ...> ... </w:p>
    p_start = xml.rfind("<w:p ", 0, idx)
    if p_start < 0:
        p_start = xml.rfind("<w:p>", 0, idx)
    p_end_tag = xml.find("</w:p>", idx)
    if p_start < 0 or p_end_tag < 0:
        return xml, False
    p_end = p_end_tag + len("</w:p>")
    return xml[:p_start] + xml[p_end:], True


def insert_references(xml: str) -> tuple[str, list[str]]:
    """Insert the 13 missing reference paragraphs in alphabetical order.

    Strategy: because the reference list is already alphabetical, we find
    the first existing paragraph whose surname (lower-cased, ASCII-folded)
    is greater than the new entry's surname, and insert before it.
    """
    # Build list of (paragraph_start, surname_key) for every reference paragraph
    refs_header_idx = xml.find("<w:t>References</w:t>")
    if refs_header_idx < 0:
        return xml, []
    # Body-of-references starts after the header paragraph closes
    refs_start = xml.find("</w:p>", refs_header_idx) + len("</w:p>")
    refs_end = len(xml)  # We'll append at end if needed
    # Slice out the references region and find paragraphs
    region = xml[refs_start:refs_end]
    p_re = re.compile(r"<w:p[ >].*?</w:p>", re.DOTALL)
    paragraphs: list[tuple[int, int, str]] = []  # (start, end, surname_key)
    text_re = re.compile(r"<w:t[^>]*>([^<]*)</w:t>")
    for m in p_re.finditer(region):
        para_xml = m.group(0)
        first_text = "".join(text_re.findall(para_xml))[:80].strip()
        # Surname = first token before "," or first whitespace
        surname = re.split(r"[,\s]", first_text, 1)[0]
        paragraphs.append((refs_start + m.start(), refs_start + m.end(), surname))

    def fold(s: str) -> str:
        import unicodedata as u
        return "".join(
            c for c in u.normalize("NFD", s) if not u.combining(c)
        ).lower()

    # Idempotency: build a set of (surname, year) pairs that already exist
    # in the reference list so that re-running this script does not insert
    # duplicate paragraphs.
    existing_pairs: set[tuple[str, str]] = set()
    for _start, _end, surname in paragraphs:
        # Re-extract year from the same paragraph
        para_text = "".join(text_re.findall(xml[_start:_end]))
        ym = re.search(r"\((\d{4})", para_text)
        if ym:
            existing_pairs.add((fold(surname), ym.group(1)))

    inserted: list[str] = []
    new_xml = xml
    # Process in reverse alphabetical order so earlier insertions don't shift
    # later positions.
    sorted_keys = sorted(NEW_REFS.keys(), key=lambda k: fold(k), reverse=True)
    for key in sorted_keys:
        new_text = NEW_REFS[key]
        new_para = _make_ref_paragraph(new_text)
        # Compare key surname (e.g., "Babiak 2006") only on surname portion
        new_surname = fold(key.split()[0])
        new_year = key.split()[-1]
        if (new_surname, new_year) in existing_pairs:
            continue  # already present — keep idempotent
        # Find insertion point: first paragraph where (fold(surname), year) > (new)
        insert_at = None
        # Re-scan paragraphs from current new_xml — they may have shifted by
        # earlier insertions. Recompute on each iteration to stay correct.
        region_idx = new_xml.find("<w:t>References</w:t>")
        region_start = new_xml.find("</w:p>", region_idx) + len("</w:p>")
        region_end = len(new_xml)
        cur_paragraphs: list[tuple[int, str, str]] = []
        for m in p_re.finditer(new_xml[region_start:region_end]):
            para_xml = m.group(0)
            first_text = "".join(text_re.findall(para_xml))[:80].strip()
            surname = re.split(r"[,\s]", first_text, 1)[0]
            year_m = re.search(r"\((\d{4})", first_text)
            yr = year_m.group(1) if year_m else "0"
            cur_paragraphs.append((region_start + m.start(), surname, yr))
        for p_start_pos, p_surname, p_year in cur_paragraphs:
            cmp_a = (fold(p_surname), p_year)
            cmp_b = (new_surname, new_year)
            if cmp_a > cmp_b:
                insert_at = p_start_pos
                break
        if insert_at is None:
            # Insert at very end of references region (before the next non-
            # reference content). We'll just put before the closing of the
            # body — append at the end of last reference paragraph.
            if cur_paragraphs:
                last_start = cur_paragraphs[-1][0]
                last_end = new_xml.find("</w:p>", last_start) + len("</w:p>")
                insert_at = last_end
            else:
                insert_at = region_end
        new_xml = new_xml[:insert_at] + new_para + new_xml[insert_at:]
        inserted.append(key)
    return new_xml, list(reversed(inserted))


# ---------------------------------------------------------------------- #
# docx I/O                                                               #
# ---------------------------------------------------------------------- #
def patch_docx(path: Path) -> dict:
    """Apply all four fixes to a docx file in place."""
    src = path.read_bytes()
    in_zip = zipfile.ZipFile(io.BytesIO(src))

    files: dict[str, bytes] = {}
    for name in in_zip.namelist():
        files[name] = in_zip.read(name)
    in_zip.close()

    if "word/document.xml" not in files:
        raise RuntimeError(f"{path}: missing word/document.xml")

    xml = files["word/document.xml"].decode("utf-8")

    report = {"file": str(path)}

    new_xml, ok_alpha = fix_alpha(xml)
    report["alpha_fixed"] = ok_alpha

    new_xml, ok_saltuk = fix_saltukoglu(new_xml)
    report["saltukoglu_fixed"] = ok_saltuk

    new_xml, n_breevaart = fix_breevaart_year(new_xml)
    report["breevaart_2019_to_2017"] = n_breevaart

    new_xml, n_vize = fix_vize_year(new_xml)
    report["vize_2016_to_2018"] = n_vize

    new_xml, ok_jones = remove_jones_2011(new_xml)
    report["jones_2011_removed"] = ok_jones

    new_xml, inserted = insert_references(new_xml)
    report["refs_inserted"] = inserted

    files["word/document.xml"] = new_xml.encode("utf-8")

    # Write to a backup first, then overwrite
    backup = path.with_suffix(path.suffix + ".bak")
    if not backup.exists():
        shutil.copy2(path, backup)

    with zipfile.ZipFile(path, "w", zipfile.ZIP_DEFLATED) as out_zip:
        for name, data in files.items():
            out_zip.writestr(name, data)

    return report


def main() -> int:
    targets = [
        PAPER / "Manuscript_only.docx",
        PAPER / "Manuscript_all.docx",
    ]
    for path in targets:
        if not path.exists():
            print(f"WARN: {path} not found, skipping")
            continue
        report = patch_docx(path)
        print(f"\nPatched: {report['file']}")
        for k, v in report.items():
            if k == "file":
                continue
            print(f"  {k}: {v}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
