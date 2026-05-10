#!/usr/bin/env python3
"""DOI resolution checker for the 3 conference paper References sections.

Extracts DOIs from each `full_paper.md` References block, sends a HEAD
request to https://doi.org/{doi}, and reports resolution status.

Pattern adapted from metaanalysis/analysis/check_doi.py (which targets
the parent-preprint references_data.py module). This script targets the
markdown bullet lists in the conference papers.

Usage:
    python3 metaanalysis/conference_submissions/scripts/check_dois.py
    python3 metaanalysis/conference_submissions/scripts/check_dois.py --offline
        # syntax-only, no network

Note on environment:
    The Claude Code sandbox blocks doi.org and api.crossref.org
    (`host_not_allowed`); --online mode therefore returns 403 for every
    DOI from inside the sandbox. Run on the author's local machine or
    in a CI environment with outbound HTTPS to verify resolution.

Exit code: non-zero if any DOI fails to resolve in --online mode, or any
syntactic issue in --offline mode.
"""
from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

ROOT = Path(__file__).resolve().parents[3]
WORKSPACE = ROOT / "metaanalysis" / "conference_submissions"
PAPERS = ["ecel", "iceel", "iceri"]

DOI_RE = re.compile(
    r"https?://(?:dx\.)?doi\.org/(10\.\d{4,9}/[^\s\)\]]+)|"
    r"\bDOI[: ]*(10\.\d{4,9}/[^\s\)\]]+)|"
    r"\b(10\.\d{4,9}/[^\s\)\]]+)",
    re.IGNORECASE,
)
SYNTAX_RE = re.compile(r"^10\.\d{4,9}/[^\s]+$")


def extract_dois(md_path: Path) -> list[tuple[int, str, str]]:
    """Yield (line_no, raw_line, doi) for every DOI seen in the References
    section of `md_path`. Stops at next `## ` heading after References.
    """
    in_refs = False
    found: list[tuple[int, str, str]] = []
    for n, line in enumerate(md_path.read_text(encoding="utf-8").splitlines(), 1):
        if re.match(r"^##\s+References\s*$", line):
            in_refs = True
            continue
        if in_refs and re.match(r"^##\s+(?!References)\S", line):
            break
        if not in_refs:
            continue
        for m in DOI_RE.finditer(line):
            doi = next((g for g in m.groups() if g), None)
            if doi:
                doi = doi.rstrip(".,;)")
                found.append((n, line.strip(), doi))
    return found


def resolve(doi: str, mailto: str = "eisuke.tokiwa@sunblaze.jp", timeout: float = 10.0) -> tuple[bool, int | str]:
    url = f"https://doi.org/{doi}"
    headers = {
        "User-Agent": f"P3-conference-doi-check/1.0 (mailto:{mailto})",
        "Accept": "*/*",
    }
    try:
        req = Request(url, method="HEAD", headers=headers)
        with urlopen(req, timeout=timeout) as resp:
            return (200 <= resp.status < 400), resp.status
    except HTTPError as e:
        return False, e.code
    except (URLError, TimeoutError) as e:
        return False, f"network: {e}"
    except Exception as e:
        return False, f"error: {e}"


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--offline", action="store_true", help="Skip network resolution; check DOI syntax only.")
    ap.add_argument("--mailto", default="eisuke.tokiwa@sunblaze.jp")
    args = ap.parse_args()

    fails = 0
    for paper in PAPERS:
        path = WORKSPACE / paper / "full_paper.md"
        if not path.exists():
            print(f"[{paper}] SKIP (no full_paper.md)")
            continue
        entries = extract_dois(path)
        # de-dup keeping first occurrence
        seen, unique = set(), []
        for ln, line, doi in entries:
            if doi in seen:
                continue
            seen.add(doi)
            unique.append((ln, line, doi))

        print(f"\n=== {paper}/full_paper.md ({len(unique)} unique DOIs) ===")
        for ln, line, doi in unique:
            if not SYNTAX_RE.match(doi):
                print(f"  [{paper}:{ln}] SYNTAX-FAIL  {doi}")
                fails += 1
                continue
            if args.offline:
                print(f"  [{paper}:{ln}] OK-syntax    {doi}")
                continue
            ok, status = resolve(doi, mailto=args.mailto)
            tag = "RESOLVE-OK  " if ok else "RESOLVE-FAIL"
            print(f"  [{paper}:{ln}] {tag} {doi:60s} ({status})")
            if not ok:
                fails += 1

    print(f"\n{'OFFLINE' if args.offline else 'ONLINE'} run: {fails} failure(s).")
    sys.exit(1 if fails else 0)


if __name__ == "__main__":
    main()
