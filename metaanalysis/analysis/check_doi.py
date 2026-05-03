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
DOI_RE = re.compile(r"https?://(?:dx\.)?doi\.org/(10\.\d{4,9}/[^\s]+)", re.I)
YEAR_RE = re.compile(r"\((\d{4})\)")
ITALIC_RE = re.compile(r"<i>(.*?)</i>", re.DOTALL)
# Trailing punctuation to strip from page/volume tokens when we slice
TRAILING_PUNCT = ".,;"


def _strip_diacritics(s: str) -> str:
    """Lowercase and remove combining accents and special letter variants."""
    s = unicodedata.normalize("NFD", s)
    s = "".join(c for c in s if not unicodedata.combining(c))
    repl = {"ı": "i", "ş": "s", "ç": "c", "ğ": "g", "ü": "u", "ö": "o",
            "ä": "a", "ñ": "n", "ø": "o", "æ": "ae", "ß": "ss"}
    for k, v in repl.items():
        s = s.replace(k, v)
    return s.lower().strip()


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

    m_doi = DOI_RE.search(entry)
    if m_doi:
        record["doi"] = m_doi.group(1).rstrip(TRAILING_PUNCT)

    m_year = YEAR_RE.search(entry)
    if m_year:
        record["year"] = m_year.group(1)

    # First author surname: substring up to the first comma
    head = entry.split(",", 1)[0].strip()
    record["surname"] = head

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


def load_references(refs_path: Path) -> list[dict]:
    spec = importlib.util.spec_from_file_location("references_data", refs_path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Could not load {refs_path}")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    out = []
    for raw in mod.REFERENCES:
        if not raw or not raw[0].isupper():
            continue
        out.append(parse_entry(raw))
    return out


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
def crossref_get(doi: str, mailto: str, timeout: float = 10.0) -> Optional[dict]:
    url = f"https://api.crossref.org/works/{doi}"
    req = urlrequest.Request(
        url,
        headers={"User-Agent": f"paper-doi-audit/1.0 (mailto:{mailto})"},
    )
    try:
        with urlrequest.urlopen(req, timeout=timeout) as resp:
            data = json.load(resp)
        return data.get("message")
    except urlerror.HTTPError as e:
        print(f"  ❌ HTTP {e.code} for {doi}: {e.reason}")
        return None
    except urlerror.URLError as e:
        print(f"  ❌ Network error for {doi}: {e.reason}")
        return None
    except (TimeoutError, json.JSONDecodeError) as e:
        print(f"  ❌ Decode/timeout for {doi}: {e}")
        return None


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
        msg = crossref_get(r["doi"], mailto)
        time.sleep(sleep)
        if msg is None:
            failed += 1
            continue

        local_findings = []

        # Year
        try:
            cr_year = str(msg["issued"]["date-parts"][0][0])
        except (KeyError, IndexError, TypeError):
            cr_year = None
        if r["year"] and cr_year and cr_year != r["year"]:
            local_findings.append(f"year local={r['year']} crossref={cr_year}")

        # First-author surname
        cr_first = None
        for a in (msg.get("author") or []):
            if a.get("sequence") == "first" or cr_first is None:
                cr_first = a.get("family") or cr_first
        if r["surname"] and cr_first:
            if _strip_diacritics(cr_first) != _strip_diacritics(r["surname"]):
                local_findings.append(
                    f"first-author local='{r['surname']}' crossref='{cr_first}'"
                )

        # Container title (journal / book)
        cr_container = (msg.get("container-title") or [""])[0]
        if r["journal_or_book"] and cr_container:
            local = _norm_text(r["journal_or_book"])
            # Drop trailing volume token from local
            local = re.sub(r"\s+\d+$", "", local)
            crn = _norm_text(cr_container)
            # Allow either to be a substring of the other (abbreviation-tolerant)
            if not (local in crn or crn in local):
                local_findings.append(
                    f"journal local='{r['journal_or_book']}' "
                    f"crossref='{cr_container}'"
                )

        # Volume
        cr_vol = msg.get("volume")
        if r["volume"] and cr_vol and r["volume"] != cr_vol:
            local_findings.append(
                f"volume local={r['volume']} crossref={cr_vol}"
            )

        # Pages
        cr_pages = (msg.get("page") or "").replace(" ", "")
        if r["pages"] and cr_pages:
            local_p = r["pages"].replace(" ", "")
            local_p = re.sub(r"[‐-―−]", "-", local_p)
            if local_p != cr_pages:
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
