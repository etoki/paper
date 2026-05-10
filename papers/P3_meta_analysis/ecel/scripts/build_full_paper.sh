#!/usr/bin/env bash
# Build the ECEL Full Paper docx from the canonical Markdown source.
#
# Usage:  bash papers/P3_meta_analysis/ecel/scripts/build_full_paper.sh
# Output: papers/P3_meta_analysis/ecel/full_paper.docx
#
# Notes:
# - Uses pandoc default styling. Final ACI Word template formatting is
#   applied manually by pasting the rendered content into the official
#   ACI template before submission.
# - All numbers in full_paper.md are traceable to results/*.csv; do not
#   hand-edit numbers in the docx — re-run run_modality_meta.py and
#   re-build instead.
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/../../../.." && pwd)"
SRC="${ROOT}/papers/P3_meta_analysis/ecel/full_paper.md"
OUT="${ROOT}/papers/P3_meta_analysis/ecel/full_paper.docx"
pandoc "${SRC}" -o "${OUT}"
echo "Wrote ${OUT}"
