#!/usr/bin/env python3
"""T9 wrapper for simulation/.

Runs the shared DOI auditor (metaanalysis/analysis/check_doi.py) against
the simulation paper's reference list at paper/05_refs.md.

Usage (from any directory):

    # Dry-run (no network)
    python3 simulation/check_doi.py

    # Live verification against Crossref (requires reachability)
    python3 simulation/check_doi.py --mode online --mailto YOUR_EMAIL

All other flags (--sleep, etc.) are forwarded to the shared auditor.
"""
from __future__ import annotations

import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
SHARED = HERE.parent / "metaanalysis" / "analysis"
DEFAULT_REFS = HERE / "paper" / "05_refs.md"

if str(SHARED) not in sys.path:
    sys.path.insert(0, str(SHARED))

import check_doi  # noqa: E402

if __name__ == "__main__":
    if not any(a.startswith("--refs") for a in sys.argv[1:]):
        sys.argv = [sys.argv[0], "--refs", str(DEFAULT_REFS)] + sys.argv[1:]
    sys.exit(check_doi.main())
