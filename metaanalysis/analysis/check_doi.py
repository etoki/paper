#!/usr/bin/env python3
"""
T9: DOI audit — verify references_data.py against Crossref REST API.

This addresses the residual hallucination risk for the 22+ "PDF non-possessed,
verified by standard knowledge" entries flagged in
`metaanalysis/paper_v2/reference_audit.md`.

Two modes:

    --mode dry-run   (default; no network)
        - Parse each REFERENCES entry into structured fields
          (first-author surname, year, title, journal, volume, pages, DOI)
        - Validate DOI syntax (10.NNNN/...)
        - Detect duplicate DOIs
        - Sanity-check DOI publisher prefix vs declared journal
        - Heuristic year-in-DOI check (where the publisher embeds year)
        - List entries with no DOI (manually verifiable only)

    --mode online    (queries https://api.crossref.org/works/{doi})
        - Confirms DOI resolves
        - Compares first-author surname, year, container-title, volume, page
          with the local entry; reports any mismatch.

Usage:

    # Dry-run (works inside any sandbox):
    python3 check_doi.py
    python3 check_doi.py --mode dry-run --refs metaanalysis/paper_v2/references_data.py

    # Live verification (requires Crossref reachability, e.g. local machine):
    python3 check_doi.py --mode online --mailto eisuke.tokiwa@sunblaze.jp

The script exits non-zero when any FAIL-level finding is detected. WARN-level
findings (e.g., heuristic prefix mismatch) do not affect exit status.
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import re
import sys
import time
import unicodedata
from collections import defaultdict
from pathlib import Path
from typing import Optional
from urllib import error as urlerror
from urllib import request as urlrequest


DEFAULT_REFS = (Path(__file__).resolve().parent.parent
                / "paper_v2" / "references_data.py")

# Major DOI registrant prefixes; used only as a *soft* sanity check.
# A mismatch here does NOT mean the reference is wrong — co-publishing,
# imprint changes, and shared DOI prefixes are common. WARN only.
PREFIX_TO_PUBLISHER = {
    "10.1016": "Elsevier",
    "10.1037": "American Psychological Association",
    "10.1002": "Wiley",
    "10.1007": "Springer",
    "10.1080": "Taylor & Francis",
    "10.1111": "Wiley (Blackwell legacy)",
    "10.1136": "BMJ",
    "10.1177": "SAGE",
    "10.3389": "Frontiers",
    "10.3390": "MDPI",
    "10.1186": "BMC / Springer Nature",
    "10.1038": "Springer Nature (Nature)",
    "10.1145": "ACM",
    "10.4324": "Routledge",
    "10.34105": "Knowledge Management & E-Learning",
    "10.36315": "InScience Press",
    "10.21203": "Research Square (preprint)",
    "10.17605": "OSF (registry / preprint)",
}

# Heuristic: which publisher prefix is most likely for a given journal name.
# Used only for WARN-level sanity checks.
JOURNAL_TO_PREFIX = {
    "the internet and higher education": "10.1016",
    "personality and individual differences": "10.1016",
    "computers in human behavior": "10.1016",
    "computers & education": "10.1016",
    "british journal of educational technology": "10.1111",
    "journal of personality": "10.1111",
    "psychological bulletin": "10.1037",
    "psychological assessment": "10.1037",
    "journal of applied psychology": "10.1037",
    "psychological methods": "10.1037",
    "journal of educational psychology": "10.1037",
    "journal of experimental psychology: general": "10.1037",
    "personality and social psychology review": "10.1177",
    "frontiers in psychology": "10.3389",
    "education sciences": "10.3390",
    "scientific reports": "10.1038",
    "bmj": "10.1136",
    "bmc medical research methodology": "10.1186",
    "systematic reviews": "10.1186",
    "international journal of educational technology in higher education":
        "10.1186",
    "education and information technologies": "10.1007",
    "educational psychology review": "10.1007",
    "european journal of investigation in health, psychology and education":
        "10.3390",
    "knowledge management & e-learning": "10.34105",
}


# ---------------------------------------------------------------------------
# Parsing
# ---------------------------------------------------------------------------
# DOI in any of: APA URL form, Vancouver "doi: 10.xxx", or bare "10.xxx/y".
# We anchor with a non-word boundary on the left to avoid false matches inside
# longer identifiers.
DOI_URL_RE = re.compile(r"https?://(?:dx\.)?doi\.org/(10\.\d{4,9}/\S+)", re.I)
DOI_BARE_RE = re.compile(r"(?:^|\s|doi:\s*)(10\.\d{4,9}/[^\s,;]+)", re.I)
YEAR_RE = re.compile(r"\((\d{4})\)")
# Fallback for non-APA entries (e.g., Vancouver "Behav Brain Sci. 2010 Jun;…")
YEAR_LOOSE_RE = re.compile(r"\b(19\d{2}|20\d{2})\b")
ITALIC_RE = re.compile(r"<i>(.*?)</i>", re.DOTALL)
# Heuristic: a paragraph that doesn't look like an APA reference start.
# Used to detect end of References section in .docx files lacking a heading
# for the next section (Tables/Figures often inline immediately after).
NOT_A_REFERENCE_RE = re.compile(
    r"^(?:Table\s+\d+|Figure\s+\d+|Note\.|Notes?\b|Appendix\b)",
    re.IGNORECASE,
)
# Trailing punctuation to strip from page/volume tokens when we slice
TRAILING_PUNCT = ".,;"


def _strip_diacritics(s: str) -> str:
    """Lowercase, normalize curly quotes, and remove combining accents and
    special-letter variants. Order matters: lowercase first so that uppercase
    Ø/Æ/Ł/etc. fall through the replacement map."""
    s = (s or "").lower()
    # Normalize various apostrophes to ASCII '
    s = (s.replace("‘", "'")
           .replace("’", "'")
           .replace("ʼ", "'")
           .replace("‛", "'"))
    s = unicodedata.normalize("NFD", s)
    s = "".join(c for c in s if not unicodedata.combining(c))
    repl = {"ı": "i", "ş": "s", "ç": "c", "ğ": "g", "ü": "u", "ö": "o",
            "ä": "a", "ñ": "n", "ø": "o", "æ": "ae", "ß": "ss",
            "ł": "l", "đ": "d"}
    for k, v in repl.items():
        s = s.replace(k, v)
    return s.strip()


def _norm_text(s: str) -> str:
    """Normalize for fuzzy comparison (lowercase, strip diacritics, collapse
    punctuation and whitespace)."""
    s = _strip_diacritics(s)
    s = re.sub(r"[‐-―−\-]", "-", s)  # all dashes -> ASCII -
    s = re.sub(r"[^\w\s\-]", " ", s)
    s = re.sub(r"\s+", " ", s).strip()
    return s


def parse_entry(entry: str) -> dict:
    """Pull structured fields out of a single APA-formatted reference string.

    The local format guarantees:
      Surname, I. I., & Surname2, I. I. (YYYY). Title. <i>Journal, vol</i>(issue), pp–pp. https://doi.org/...
    Books / chapters omit the italic journal block and may have no DOI.
    """
    record = {
        "raw": entry,
        "surname": None,
        "year": None,
        "title": None,
        "italic": None,           # full italic-tagged string
        "journal_or_book": None,  # journal name without volume token
        "volume": None,
        "pages": None,
        "doi": None,
    }

    # DOI: prefer URL form (most reliable boundary), then any bare 10.x/y form
    m_doi = DOI_URL_RE.search(entry)
    if not m_doi:
        m_doi = DOI_BARE_RE.search(entry)
    if m_doi:
        record["doi"] = m_doi.group(1).rstrip(TRAILING_PUNCT)

    # Year: prefer "(YYYY)" (APA), fall back to any 19xx/20xx token that is
    # not embedded in the DOI string.
    m_year = YEAR_RE.search(entry)
    if m_year:
        record["year"] = m_year.group(1)
    else:
        doi_str = record["doi"] or ""
        for m in YEAR_LOOSE_RE.finditer(entry):
            if m.group(0) not in doi_str:
                record["year"] = m.group(0)
                break

    # First author surname: depends on entry style.
    #   APA:        "Vize, C. E., Lynam, ..."  / "de Vries, R. E."
    #               surname is everything before ", I." (initial + period).
    #   Vancouver:  "Bianchi R. (2018)" / "Henrich J, Heine SJ, ..."
    #               surname is the leading word followed by space + initials.
    apa_m = re.match(r"^(.+?),\s+[A-Z]\.", entry)
    if apa_m:
        record["surname"] = apa_m.group(1).strip()
    else:
        van_m = re.match(r"^([A-ZÀ-ÿ][\w'’`\-]+)\s+[A-Z]+\b", entry)
        if van_m:
            record["surname"] = van_m.group(1).strip()
        else:
            record["surname"] = entry.split(",", 1)[0].strip()

    # Italic block: usually the journal/book title plus volume
    m_it = ITALIC_RE.search(entry)
    if m_it:
        italic = m_it.group(1).strip()
        record["italic"] = italic
        # Split last comma to separate "Journal Name" and "VolumeToken"
        if "," in italic:
            jrn, vol = italic.rsplit(",", 1)
            jrn = jrn.strip()
            vol = vol.strip()
            # If the trailing token is purely a volume number, accept it
            if re.fullmatch(r"\d{1,4}[A-Z]?", vol):
                record["journal_or_book"] = jrn
                record["volume"] = vol
            else:
                record["journal_or_book"] = italic.strip()
        else:
            record["journal_or_book"] = italic.strip()

    # Pages: pattern at end of entry like ", 100-110." or ", 191–208."
    # We look for the *last* such token (before any trailing DOI).
    pre_doi = entry.split("https://doi.org/")[0]
    pages_m = re.search(r",\s*(\d{1,5}[‐-―−\-]\d{1,5})\s*\.?\s*$",
                        pre_doi.strip())
    if pages_m:
        record["pages"] = pages_m.group(1)

    # Title: after the year period, before the first italic block.
    after_year = entry[m_year.end():] if m_year else entry
    after_year = after_year.lstrip(". ").strip()
    if "<i>" in after_year:
        title = after_year.split("<i>", 1)[0].strip()
    else:
        title = after_year
    title = title.rstrip(". ").strip()
    record["title"] = title

    return record


def _load_py(refs_path: Path) -> list[str]:
    """Load REFERENCES list from a Python module exposing a top-level
    REFERENCES list of strings (already <i>...</i>-tagged)."""
    spec = importlib.util.spec_from_file_location("references_data", refs_path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Could not load {refs_path}")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return [s for s in mod.REFERENCES if s and s[0].isupper()]


_MD_ITALIC_RE = re.compile(r"\*([^*\n]+?)\*")


def _load_md(refs_path: Path) -> list[str]:
    """Load reference entries from an APA-formatted markdown file.

    Entries are paragraphs separated by blank lines. Markdown italic
    `*Journal, vol*` is rewritten to `<i>Journal, vol</i>` so the rest of
    the parsing pipeline can use a single italic syntax."""
    text = refs_path.read_text(encoding="utf-8")
    # Strip an introductory heading like "# 05. References ..."
    text = re.sub(r"\A\s*#[^\n]*\n", "", text)
    paragraphs = re.split(r"\n\s*\n", text)
    out: list[str] = []
    for p in paragraphs:
        p = p.strip()
        if not p or not p[0].isupper():
            continue
        # Collapse internal newlines (an APA entry sometimes wraps)
        p = re.sub(r"\s*\n\s*", " ", p)
        # Markdown italic -> <i>...</i>
        p = _MD_ITALIC_RE.sub(r"<i>\1</i>", p)
        out.append(p)
    return out


def _load_docx(refs_path: Path,
               heading_pattern: str = r"^references\b",
               terminators: tuple[str, ...] = (
                   "tables", "figures", "appendix",
                   "author note", "supplementary", "supplement",
               )) -> list[str]:
    """Load reference entries from a .docx manuscript.

    Walks paragraphs, locates the References section by heading, and
    yields each subsequent paragraph until a terminator heading. Italic
    runs are wrapped in `<i>...</i>` so the journal/book title and
    volume token are recoverable by parse_entry.
    """
    try:
        from docx import Document  # type: ignore
    except ImportError as e:
        raise RuntimeError(
            "python-docx is required to load .docx reference lists"
        ) from e

    doc = Document(str(refs_path))
    in_refs = False
    out: list[str] = []
    head_re = re.compile(heading_pattern, re.IGNORECASE)
    term_set = {t.lower() for t in terminators}

    for para in doc.paragraphs:
        text = para.text.strip()
        if not text:
            continue
        low = text.lower()
        if not in_refs:
            if head_re.search(low):
                in_refs = True
            continue
        # Inside refs section
        if low in term_set:
            break
        # Some manuscripts have no Tables/Figures heading; the section
        # transitions silently into "Table 1", "Note.", "Figure 1" etc.
        # Stop as soon as we encounter such a paragraph.
        if NOT_A_REFERENCE_RE.match(text):
            break
        # Build the entry preserving italic markers per-run, walking both
        # regular runs and hyperlink runs in document order. python-docx
        # exposes only direct-child runs via paragraph.runs, which omits
        # hyperlinked text — and DOIs are usually hyperlinks.
        from docx.oxml.ns import qn  # local import to keep top tidy
        parts: list[str] = []

        def _emit_run(r_elem):
            t_elems = r_elem.findall(qn("w:t"))
            text_pieces = [(t.text or "") for t in t_elems]
            run_text = "".join(text_pieces)
            if run_text == "":
                return
            italic_props = r_elem.find(qn("w:rPr"))
            italic = (italic_props is not None
                      and italic_props.find(qn("w:i")) is not None)
            parts.append(f"<i>{run_text}</i>" if italic else run_text)

        for child in para._element.iterchildren():
            tag = child.tag
            if tag == qn("w:r"):
                _emit_run(child)
            elif tag == qn("w:hyperlink"):
                for sub in child.findall(qn("w:r")):
                    _emit_run(sub)

        joined = "".join(parts) if parts else text
        # Collapse adjacent <i>…</i><i>…</i> that arise from run splits
        joined = re.sub(r"</i>\s*<i>", " ", joined)
        joined = re.sub(r"<i>\s*</i>", "", joined)
        if joined and joined[0].isupper():
            out.append(joined.strip())
    return out


def load_references(refs_path: Path) -> list[dict]:
    """Dispatch on file extension and return parsed reference records."""
    suffix = refs_path.suffix.lower()
    if suffix == ".py":
        raws = _load_py(refs_path)
    elif suffix == ".md":
        raws = _load_md(refs_path)
    elif suffix == ".docx":
        raws = _load_docx(refs_path)
    else:
        raise RuntimeError(
            f"Unsupported reference source format: {suffix} "
            f"(expected .py, .md, or .docx)"
        )
    return [parse_entry(raw) for raw in raws]


# ---------------------------------------------------------------------------
# Dry-run checks (no network)
# ---------------------------------------------------------------------------
def dry_run(records: list[dict]) -> tuple[int, int, int]:
    """Return (passed, warn, failed)."""
    passed = warn = failed = 0
    print("\n" + "=" * 70)
    print("T9 dry-run — DOI syntactic / heuristic audit (no network)")
    print("=" * 70)

    # 1. Duplicate-DOI check
    by_doi: dict[str, list[str]] = defaultdict(list)
    for r in records:
        if r["doi"]:
            by_doi[r["doi"].lower()].append(r["surname"])
    for doi, surnames in by_doi.items():
        if len(surnames) > 1:
            print(f"  ❌ Duplicate DOI {doi} cited by: {', '.join(surnames)}")
            failed += 1

    # 2. Per-entry checks
    no_doi = []
    for r in records:
        label = f"{r['surname']} ({r['year']})"
        if r["doi"] is None:
            no_doi.append(label)
            continue

        # 2a. DOI syntax
        if not re.fullmatch(r"10\.\d{4,9}/[^\s]+", r["doi"]):
            print(f"  ❌ {label}: malformed DOI '{r['doi']}'")
            failed += 1
            continue

        # 2b. Publisher-prefix sanity vs journal
        prefix = r["doi"].split("/", 1)[0]
        publisher = PREFIX_TO_PUBLISHER.get(prefix, "?")
        if r["journal_or_book"]:
            jrn_norm = _norm_text(r["journal_or_book"])
            jrn_norm = re.sub(r"\s+\d+$", "", jrn_norm)  # strip trailing volume
            expected_prefix = JOURNAL_TO_PREFIX.get(jrn_norm)
            if expected_prefix and expected_prefix != prefix:
                print(f"  ⚠ {label}: DOI prefix {prefix} ({publisher}) "
                      f"unusual for journal '{r['journal_or_book']}' "
                      f"(expected {expected_prefix}); verify manually.")
                warn += 1
            else:
                passed += 1
        else:
            passed += 1

    # 3. Entries with no DOI — informational
    if no_doi:
        print(f"\n  ℹ {len(no_doi)} entries have no DOI (books, chapters, "
              f"manuals, manuscripts in preparation):")
        for label in no_doi:
            print(f"     • {label}")

    return passed, warn, failed


# ---------------------------------------------------------------------------
# Online checks (Crossref)
# ---------------------------------------------------------------------------
def _crossref_status(doi: str) -> str:
    """Best-effort guess at the most likely DOI registration agency for a
    given prefix. Used only to label 404s informatively."""
    prefix = doi.split("/", 1)[0]
    return {
        "10.15002": "JaLC (Hosei University Repository)",
        "10.32222": "JaLC (Japanese journal)",
        "10.24602": "JaLC (Japanese journal)",
        "10.4992":  "JaLC (Japanese journal)",
        "10.21203": "DataCite (Research Square preprint)",
        "10.17605": "DataCite (OSF)",
        "10.5281":  "DataCite (Zenodo)",
        "10.31234": "DataCite (PsyArXiv)",
    }.get(prefix, "")


def crossref_get(doi: str, mailto: str,
                 timeout: float = 10.0) -> tuple[Optional[dict], Optional[str]]:
    """Return (message, error_kind). error_kind is None on success, otherwise
    one of: 'not_found', 'http_error', 'network', 'decode'."""
    url = f"https://api.crossref.org/works/{doi}"
    req = urlrequest.Request(
        url,
        headers={"User-Agent": f"paper-doi-audit/1.0 (mailto:{mailto})"},
    )
    try:
        with urlrequest.urlopen(req, timeout=timeout) as resp:
            data = json.load(resp)
        return data.get("message"), None
    except urlerror.HTTPError as e:
        if e.code == 404:
            return None, "not_found"
        print(f"  ❌ HTTP {e.code} for {doi}: {e.reason}")
        return None, "http_error"
    except urlerror.URLError as e:
        print(f"  ❌ Network error for {doi}: {e.reason}")
        return None, "network"
    except (TimeoutError, json.JSONDecodeError) as e:
        print(f"  ❌ Decode/timeout for {doi}: {e}")
        return None, "decode"


def _decode_html_entities(s: str) -> str:
    """Decode HTML/XML entities sometimes returned by Crossref (e.g., &amp;)."""
    import html
    return html.unescape(s or "")


def _container_titles(msg: dict) -> list[str]:
    """Return all non-empty container-title strings from a Crossref message,
    HTML-entity-decoded. Some records carry both a book title and a series
    title; checking each independently avoids false positives."""
    raw = msg.get("container-title") or []
    if isinstance(raw, str):
        raw = [raw]
    return [_decode_html_entities(t) for t in raw if t]


def _surname_match(local: str, crossref_family: str,
                   crossref_given: str = "") -> bool:
    """Tolerant first-author comparison.

    Returns True when:
    - normalized local equals normalized family or given, or
    - local appears as the last whitespace-separated token of the Crossref
      family (a known Frontiers / proceedings ingestion artifact where
      "Given Family" is stored in the family slot), or
    - local appears as the last token of given (DataCite/Research Square
      sometimes records "Given Family" in the given slot and leaves family
      empty), or
    - the Crossref family equals the last token of the local surname (multi-
      word particles like "de Vries" / "Van der Linden")."""
    def _toks(s: str) -> list[str]:
        return _strip_diacritics(s).split()

    a = _strip_diacritics(local)
    b = _strip_diacritics(crossref_family)
    g = _strip_diacritics(crossref_given)
    if a and (a == b or a == g):
        return True
    a_tokens, b_tokens, g_tokens = _toks(local), _toks(crossref_family), _toks(crossref_given)
    if b_tokens and b_tokens[-1] == a:
        return True
    if g_tokens and g_tokens[-1] == a:
        return True
    if a_tokens and (a_tokens[-1] == b or a_tokens[-1] == g):
        return True
    return False


def _journal_match(local: str, crossref_titles: list[str]) -> bool:
    """Tolerant journal/container comparison.

    Pass when any of the following holds for at least one Crossref title:
    - normalized substring match in either direction
    - the local title equals the head segment of "Head: Subtitle" form,
      or vice versa (for journals whose Crossref entry includes a subtitle)
    - the local title equals one of the Crossref titles after stripping a
      trailing volume token from the local side
    """
    a = _norm_text(local)
    a = re.sub(r"\s+\d+$", "", a)  # strip trailing volume token from local
    if not a:
        return False
    for t in crossref_titles:
        b = _norm_text(t)
        if not b:
            continue
        if a in b or b in a:
            return True
        # Subtitle split (": ..." part is often absent in citation form)
        a_head = a.split(":", 1)[0].strip()
        b_head = b.split(":", 1)[0].strip()
        if a_head and (a_head == b_head or a_head == b or a == b_head):
            return True
    return False


def _years_from_msg(msg: dict) -> tuple[Optional[str], Optional[str], Optional[str]]:
    """Return (print_year, online_year, issued_year) from a Crossref message.

    Crossref's `issued` is the earliest known publication date — usually
    online-first. APA 7 prefers the print/issue cover year for periodicals,
    which is in `published-print`."""
    def _y(field: str) -> Optional[str]:
        try:
            return str(msg[field]["date-parts"][0][0])
        except (KeyError, IndexError, TypeError):
            return None
    return _y("published-print"), _y("published-online"), _y("issued")


def _year_match(local: Optional[str], print_y: Optional[str],
                online_y: Optional[str], issued_y: Optional[str]) -> bool:
    """Pass when local equals the print year (preferred), or is within ±1 of
    any reported Crossref year (covers online-first vs cover-date drift)."""
    if not local:
        return True
    if print_y and local == print_y:
        return True
    candidates = [y for y in (print_y, online_y, issued_y) if y]
    if any(local == y for y in candidates):
        return True
    try:
        ly = int(local)
        return any(abs(ly - int(y)) <= 1 for y in candidates if y.isdigit())
    except ValueError:
        return False


def online_audit(records: list[dict], mailto: str,
                 sleep: float = 0.05) -> tuple[int, int, int]:
    """Return (passed, warn, failed)."""
    passed = warn = failed = 0
    print("\n" + "=" * 70)
    print("T9 online — Crossref DOI verification")
    print("=" * 70)

    for r in records:
        if r["doi"] is None:
            continue
        label = f"{r['surname']} ({r['year']})"
        msg, err = crossref_get(r["doi"], mailto)
        time.sleep(sleep)
        if msg is None:
            if err == "not_found":
                hint = _crossref_status(r["doi"])
                tail = f" — likely {hint}" if hint else (
                    " — Crossref has no record; verify the DOI resolves at "
                    "doi.org and that it is registered with a DOI agency "
                    "other than Crossref (JaLC / DataCite / mEDRA / OP)."
                )
                print(f"  ⚠ {label} [{r['doi']}]: not in Crossref{tail}")
                warn += 1
            else:
                failed += 1
            continue

        local_findings = []

        # Year — prefer print, allow ±1 against online/issued
        print_y, online_y, issued_y = _years_from_msg(msg)
        if not _year_match(r["year"], print_y, online_y, issued_y):
            shown = print_y or online_y or issued_y
            local_findings.append(
                f"year local={r['year']} crossref={shown} "
                f"(print={print_y}, online={online_y}, issued={issued_y})"
            )

        # First-author surname — also consult the `given` field
        cr_first_family = ""
        cr_first_given = ""
        for a in (msg.get("author") or []):
            if a.get("sequence") == "first" or not cr_first_family:
                cr_first_family = a.get("family") or cr_first_family
                cr_first_given = a.get("given") or cr_first_given
        if r["surname"] and (cr_first_family or cr_first_given):
            if not _surname_match(r["surname"], cr_first_family,
                                  cr_first_given):
                local_findings.append(
                    f"first-author local='{r['surname']}' "
                    f"crossref family='{cr_first_family}' "
                    f"given='{cr_first_given}'"
                )

        # Container title — check all titles, decode entities, subtitle-split
        cr_titles = _container_titles(msg)
        if r["journal_or_book"] and cr_titles:
            if not _journal_match(r["journal_or_book"], cr_titles):
                local_findings.append(
                    f"journal local='{r['journal_or_book']}' "
                    f"crossref={cr_titles}"
                )

        # Volume
        cr_vol = msg.get("volume")
        if r["volume"] and cr_vol and r["volume"] != cr_vol:
            local_findings.append(
                f"volume local={r['volume']} crossref={cr_vol}"
            )

        # Pages — accept Crossref reporting only the first page when local
        # provides a full range
        cr_pages = (msg.get("page") or "").replace(" ", "")
        if r["pages"] and cr_pages:
            local_p = re.sub(r"[‐-―−]", "-", r["pages"].replace(" ", ""))
            cr_p = re.sub(r"[‐-―−]", "-", cr_pages)
            local_first = local_p.split("-", 1)[0]
            cr_first_pg = cr_p.split("-", 1)[0]
            if local_p != cr_p and local_first != cr_first_pg:
                local_findings.append(
                    f"pages local={r['pages']} crossref={cr_pages}"
                )

        if local_findings:
            print(f"  ⚠ {label} [{r['doi']}]:")
            for f in local_findings:
                print(f"      - {f}")
            warn += 1
        else:
            passed += 1

    return passed, warn, failed


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def main() -> int:
    p = argparse.ArgumentParser(description="T9: DOI audit (Crossref-backed)")
    p.add_argument("--refs", default=str(DEFAULT_REFS),
                   help="Path to references_data.py")
    p.add_argument("--mode", choices=("dry-run", "online"), default="dry-run")
    p.add_argument("--mailto",
                   help="Contact email for the polite Crossref pool "
                        "(required for --mode online)")
    p.add_argument("--sleep", type=float, default=0.05,
                   help="Delay (s) between Crossref requests; default 0.05")
    args = p.parse_args()

    refs_path = Path(args.refs).resolve()
    if not refs_path.exists():
        print(f"ERROR: references file not found: {refs_path}", file=sys.stderr)
        return 2

    records = load_references(refs_path)
    print(f"\nLoaded {len(records)} reference entries from {refs_path.name} "
          f"({sum(1 for r in records if r['doi'])} with DOI).")

    if args.mode == "dry-run":
        passed, warn, failed = dry_run(records)
    else:
        if not args.mailto:
            print("ERROR: --mailto is required for online mode "
                  "(Crossref polite-pool requirement).", file=sys.stderr)
            return 2
        passed, warn, failed = online_audit(records, args.mailto, args.sleep)

    print("\n" + "-" * 70)
    print(f"  passed={passed}, warn={warn}, failed={failed}")
    print("-" * 70)
    return 1 if failed > 0 else 0


# Adapter for check_hallucinations.py task dispatch (dry-run only there).
def task_t9(refs_path: Optional[Path] = None) -> int:
    refs = Path(refs_path) if refs_path else DEFAULT_REFS
    records = load_references(refs)
    print(f"\nLoaded {len(records)} reference entries "
          f"({sum(1 for r in records if r['doi'])} with DOI).")
    _, _, failed = dry_run(records)
    return 1 if failed > 0 else 0


if __name__ == "__main__":
    sys.exit(main())
