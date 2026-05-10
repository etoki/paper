#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/../../../.." && pwd)"
SRC="${ROOT}/papers/P3_meta_analysis/ieee_tale/full_paper.md"
OUT="${ROOT}/papers/P3_meta_analysis/ieee_tale/full_paper.docx"
pandoc "${SRC}" -o "${OUT}"
echo "Wrote ${OUT}"
