"""Put hand-downloaded PDFs under their record_id.

    python metaanalysis/sr_update/place_pdfs.py --dry-run
    python metaanalysis/sr_update/place_pdfs.py

Publishers name a download whatever they like: `theobald2021.pdf`,
`EJ1196465.pdf`, `10.4018@ijhcitp.2016070103.pdf`. The full-text stage
needs `fulltext/pdf/<record_id>.pdf`. This script reads each PDF dropped
into `fulltext/`, works out which record it is, and moves it into
`fulltext/pdf/` under the right name.

How a file is identified, in order:
  1. a DOI in the first three pages that matches a record's DOI. The text
     layer often loses the punctuation, so "10 1007 s10639 023 12222 5"
     is read as 10.1007/s10639-023-12222-5.
  2. the record's title appearing in the first two pages.
  3. nothing: the file stays where it is and is listed as unidentified.

Titles are compared after unescaping HTML entities, because ERIC stores
"Students&apos;" where other sources store "Students'".

Duplicate records: the search kept some papers twice, as an ERIC record
with no DOI and a record with a DOI, under different record_ids and
sometimes different years (print vs online). When a placed record has
such a twin that is also still in the screening, the same PDF is copied
there too, so the reviewer is not sent to download it a second time.

Output: `fulltext/placed_by_hand.csv` records, for each file, the
record_id it became, how it was identified, and any twin it was also
copied to. The PDFs themselves are not committed.

Run `fetch_fulltext.py --rebuild-lists` afterwards to take the placed
records off the to-download list.
"""
from __future__ import annotations

import argparse
import collections
import csv
import html
import re
import shutil
import sys
import unicodedata
from difflib import SequenceMatcher
from pathlib import Path

try:
    import pypdf
except ImportError:
    sys.exit("pypdf is needed. On this machine it is installed for "
             "/usr/local/bin/python3, so run this script with that one.")

HERE = Path(__file__).resolve().parent
csv.field_size_limit(10 ** 7)
TITLE_MIN = 20       # shorter titles match too much
TITLE_SCORE = 0.90   # accept a title match only above this
PAGES = 3

# Files whose text layer is unusable (scanned, or glyph-encoded so that
# "An Exploration" comes out as "g36 g81 g3 g40 g91..."). Identified by
# hand from the filename and the record's title, and kept here so the
# run stays reproducible.
BY_HAND = {
    "The_Association_of_Formative_Feedback_with_Engagement_and_Student_Performance.pdf": "R03949",
    "downs2019.pdf": "R06021",
}


def norm(t):
    t = html.unescape(html.unescape(t or ""))
    t = unicodedata.normalize("NFKD", t).encode("ascii", "ignore").decode().lower()
    return re.sub(r"[^a-z0-9]+", " ", t).strip()


def year(r):
    try:
        return int(r["year"])
    except (ValueError, KeyError, TypeError):
        return 0


def dois_in(text):
    """DOIs in a PDF's text, repairing punctuation the text layer dropped."""
    out = []
    for m in re.finditer(r"10[\s.]?\d{4,5}[\s/]+[^\s]{3,60}", (text or "").lower()):
        s = re.sub(r"^10[\s.]?(\d{4,5})[\s/]+", r"10.\1/", m.group(0))
        out.append(s.strip().rstrip(".,;)"))
    return out


def twins(rows):
    """record_ids that hold the same paper: same title, year within one."""
    groups = collections.defaultdict(list)
    for r in rows:
        n = norm(r["title"])
        if len(n) >= 25:
            groups[n].append(r)
    out = collections.defaultdict(list)
    for v in groups.values():
        if len(v) > 1 and max(year(x) for x in v) - min(year(x) for x in v) <= 1:
            for a in v:
                out[a["record_id"]] = [b["record_id"] for b in v
                                       if b["record_id"] != a["record_id"]]
    return out


def kept_ids(path):
    if not path.exists():
        return set()
    out = set()
    with path.open(encoding="utf-8-sig") as f:
        for r in csv.DictReader(f):
            d = ((r.get("et_decision") or "").strip().lower()
                 or (r.get("claude_suggestion") or "").strip().lower())
            if d in ("include", "maybe"):
                out.add(r["record_id"])
    return out


def text_of(path):
    try:
        rd = pypdf.PdfReader(str(path))
        return " ".join((rd.pages[i].extract_text() or "")
                        for i in range(min(PAGES, len(rd.pages))))
    except Exception:
        return ""


def identify(path, by_doi, cands):
    if path.name in BY_HAND:
        return BY_HAND[path.name], "title (by hand)"
    raw = text_of(path)
    for c in dois_in(raw):
        hit = next((r for d, r in by_doi.items() if c.startswith(d)), None)
        if hit:
            return hit["record_id"], "doi"
    head = norm(raw)[:1200]
    best, score = None, 0.0
    for rid, nt in cands:
        if len(nt) < TITLE_MIN:
            continue
        s = 1.0 if nt in head else SequenceMatcher(None, nt, head[:len(nt) + 60]).ratio()
        if s > score:
            best, score = rid, s
    return (best, "title") if score >= TITLE_SCORE else ("", "")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--records", type=Path, default=HERE / "records_dedup.csv")
    ap.add_argument("--out", type=Path, default=HERE / "fulltext")
    ap.add_argument("--screening", type=Path,
                    default=HERE / "screening" / "screening_ta.csv")
    ap.add_argument("--dry-run", action="store_true",
                    help="say what would be placed and move nothing")
    args = ap.parse_args()

    with args.records.open(encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    by_doi = {(r["doi"] or "").strip().lower(): r for r in rows if (r["doi"] or "").strip()}
    cands = [(r["record_id"], norm(r["title"])) for r in rows if norm(r["title"])]
    twin = twins(rows)
    keep = kept_ids(args.screening)
    pdf_dir = args.out / "pdf"
    pdf_dir.mkdir(parents=True, exist_ok=True)

    files = sorted(p for p in args.out.glob("*.pdf"))
    if not files:
        sys.exit(f"No PDFs directly in {args.out}. Nothing to place.")

    log = []
    for p in files:
        rid, how = identify(p, by_doi, cands)
        if not rid:
            log.append({"file": p.name, "record_id": "", "matched_by": "",
                        "also_placed_as": "", "note": "not identified"})
            print(f"  ?  {p.name}")
            continue
        dest = pdf_dir / f"{rid}.pdf"
        if dest.exists():
            log.append({"file": p.name, "record_id": rid, "matched_by": how,
                        "also_placed_as": "", "note": "already present; left in place"})
            print(f"  =  {p.name} -> {rid} (already there)")
            continue
        also = [q for q in twin.get(rid, [])
                if q in keep and not (pdf_dir / f"{q}.pdf").exists()]
        if not args.dry_run:
            shutil.move(str(p), str(dest))
            for q in also:
                shutil.copy2(str(dest), str(pdf_dir / f"{q}.pdf"))
        log.append({"file": p.name, "record_id": rid, "matched_by": how,
                    "also_placed_as": ";".join(also),
                    "note": "same paper under another record_id" if also else ""})
        print(f"  ok {p.name} -> {rid} [{how}]"
              + (f" + {','.join(also)}" if also else ""))

    if not args.dry_run:
        with (args.out / "placed_by_hand.csv").open("a" if (args.out / "placed_by_hand.csv").exists()
                                                    else "w", encoding="utf-8", newline="") as f:
            w = csv.DictWriter(f, fieldnames=["file", "record_id", "matched_by",
                                              "also_placed_as", "note"])
            if f.tell() == 0:
                w.writeheader()
            w.writerows(log)

    placed = [l for l in log if l["record_id"] and "not identified" not in l["note"]]
    print(f"\n{len(placed)} of {len(log)} files placed"
          + (" (dry run, nothing moved)" if args.dry_run else ""))
    print("identified by:", dict(collections.Counter(l["matched_by"] for l in placed)))
    missing = [l["file"] for l in log if not l["record_id"]]
    if missing:
        print(f"{len(missing)} not identified, left in {args.out}: {missing}")
    print(f"{len(list(pdf_dir.glob('*.pdf')))} PDFs now in {pdf_dir}")


if __name__ == "__main__":
    main()
