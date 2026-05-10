#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/../../../.." && pwd)"
SRC="${ROOT}/metaanalysis/conference_submissions/iceri/full_paper.md"
OUT="${ROOT}/metaanalysis/conference_submissions/iceri/full_paper.docx"
pandoc "${SRC}" -o "${OUT}"
echo "Wrote ${OUT}"
