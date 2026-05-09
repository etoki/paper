# Conferences — Fact JSON Registry

This directory holds one JSON file per target venue. Each file captures the
**verifiable facts** needed to plan, draft, and submit a paper.

## Schema

```jsonc
{
  "id": "<short id used in paths>",
  "name": "<full conference name>",
  "year": 2026,
  "host": "<organising body>",
  "location": "<city, country>",
  "dates": "<event dates>",
  "url": "<official site>",                 // omit if unverified
  "deadlines": {
    "abstract":    "YYYY-MM-DD",
    "full_paper":  "YYYY-MM-DD",
    "camera_ready":"YYYY-MM-DD",
    "registration":"YYYY-MM-DD"
  },
  "submission": {
    "system":  "<EasyChair / OpenReview / EDAS / portal>",
    "format":  "<word/latex template>",
    "max_length_pages": <int>,
    "anonymous": false,                      // or true
    "review":   "<single-blind / double-blind / open>"
  },
  "indexing":     ["<Scopus>", "<IEEE Xplore>", "<DBLP>"],
  "fees":         { "currency": "EUR", "early": null, "late": null },
  "travel_required": true,
  "subset_axis":      "<which subset of preprint we target here>",
  "novel_analysis":   "<analysis NOT in preprint, included to avoid self-plagiarism>",
  "preprint_disclosure_required": true,
  "notes":            "<free-form notes from official CFP>",
  "verified_on":      "YYYY-MM-DD",          // when these facts were last cross-checked against the official site
  "source":           "<URL or 'user-provided'>"
}
```

## Files

| File | Status |
|---|---|
| `ecel_2026.json`     | Tier 1 — submitting (deadline 5/14, 5/21) |
| `ieee_tale_2026.json`| Tier 1 — submitting (deadline 6/30) |
| `iceel_2026.json`    | Tier 1 — submitting (deadline 6/30) |
| `iceri_2026.json`    | Tier 1 — submitting (deadline 7/9) |
| `_fallback_template.json` | Template for additional venues (verify facts before using) |

## Update rule

Treat these JSON files as the single source of truth for deadlines and
formatting. Whenever a CFP page changes, edit the JSON, bump `verified_on`,
and re-render `dashboard/index.md`.
