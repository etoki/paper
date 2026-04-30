"""Verify reproduction by hash-comparing local outputs vs reference.

Specification:
- v2.0 master Section 8 (Reproducibility, D-NEW9): byte-identical
  reproduction is the verification target on the canonical platform
  (pinned via Dockerfile + uv.lock).

Reference hashes are stored in ``output/reference_hashes.json`` after
the canonical reproduction run; this script re-computes hashes on the
local outputs and compares.

Usage (from Makefile):

    make verify

Or:

    python -m code.verify_reproduction \\
        --output-dir output \\
        --reference-hashes output/reference_hashes.json \\
        --strict

Exit code 0 = all hashes match; non-zero = mismatch (with details
printed to stderr).
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path


def file_sha256(path: Path) -> str:
    """Compute SHA-256 hex digest of a file."""
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(65536), b""):
            h.update(chunk)
    return h.hexdigest()


def collect_local_hashes(output_dir: Path) -> dict[str, str]:
    """Compute SHA-256 hashes of all output files (relative paths)."""
    hashes: dict[str, str] = {}
    for path in sorted(output_dir.rglob("*")):
        if path.is_file() and not path.name.startswith("reference_hashes"):
            rel = path.relative_to(output_dir).as_posix()
            hashes[rel] = file_sha256(path)
    return hashes


def compare_hashes(
    local: dict[str, str], reference: dict[str, str], strict: bool = False
) -> tuple[list[str], list[str], list[str]]:
    """Compare two hash dictionaries.

    Returns ``(matching, mismatching, missing)`` as relative-path lists.
    ``missing`` are files in reference but not in local.
    Files present locally but not in reference are flagged in
    ``mismatching`` if strict=True.
    """
    matching: list[str] = []
    mismatching: list[str] = []
    missing: list[str] = []

    ref_keys = set(reference)
    local_keys = set(local)

    for key in sorted(ref_keys & local_keys):
        if local[key] == reference[key]:
            matching.append(key)
        else:
            mismatching.append(key)

    for key in sorted(ref_keys - local_keys):
        missing.append(key)

    if strict:
        for key in sorted(local_keys - ref_keys):
            mismatching.append(f"{key} (local-only; not in reference)")

    return matching, mismatching, missing


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("output"),
        help="Directory containing local outputs",
    )
    parser.add_argument(
        "--reference-hashes",
        type=Path,
        default=Path("output/reference_hashes.json"),
        help="JSON file mapping relative paths to canonical SHA-256 hashes",
    )
    parser.add_argument(
        "--strict",
        action="store_true",
        help="Fail if local has files not present in reference",
    )
    args = parser.parse_args()

    if not args.reference_hashes.is_file():
        print(
            f"ERROR: reference hash file not found: {args.reference_hashes}\n"
            "Either run `make reproduce` to generate one, or download from\n"
            "the OSF v2.0 project supplementary if available.",
            file=sys.stderr,
        )
        sys.exit(2)

    with args.reference_hashes.open("r", encoding="utf-8") as f:
        reference = json.load(f)

    local = collect_local_hashes(args.output_dir)
    matching, mismatching, missing = compare_hashes(local, reference, strict=args.strict)

    print(f"Reproduction verification (output dir: {args.output_dir})")
    print(f"  Reference hash file: {args.reference_hashes}")
    print(f"  Matching files:    {len(matching)}")
    print(f"  Mismatching files: {len(mismatching)}")
    print(f"  Missing files:     {len(missing)}")

    if mismatching:
        print("\nMISMATCHED files (local hash differs from reference):", file=sys.stderr)
        for m in mismatching:
            print(f"  - {m}", file=sys.stderr)

    if missing:
        print("\nMISSING files (in reference but not in local output):", file=sys.stderr)
        for m in missing:
            print(f"  - {m}", file=sys.stderr)

    if mismatching or missing:
        print(
            "\nReproduction FAILED. Check that you used the locked seed (20260429)\n"
            "and the canonical Docker image (per simulation/Dockerfile).",
            file=sys.stderr,
        )
        sys.exit(1)

    print(
        "\nReproduction VERIFIED: all output files match reference hashes.\n"
        f"v2.0 OSF DOI: 10.17605/OSF.IO/3Y54U\n"
        f"Methods Clarifications Log: v1.0 (locked 2026-05-21)"
    )
    sys.exit(0)


if __name__ == "__main__":
    main()
