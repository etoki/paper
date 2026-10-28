"""Download the full texts needed for the full-text screening stage.

    python metaanalysis/sr_update/fetch_fulltext.py              # records kept at title/abstract
    python metaanalysis/sr_update/fetch_fulltext.py --ids R0012 R0040
    python metaanalysis/sr_update/fetch_fulltext.py --records other.csv --out /tmp/ft

Which records: those kept at the title/abstract stage. By default these
come from screening/screening_ta.csv, using the reviewer's et_decision
where it is filled in and Claude's proposal otherwise, so retrieval can
start before every row is confirmed. Without that sheet, rows of
records_dedup.csv whose ta_decision is "include" or "maybe" are used.
--ids overrides both. Records that already have a PDF are skipped,
so the script can be re-run after downloading some files by hand.

Where files go (all under sr_update/fulltext/ unless --out is given):
  pdf/<record_id>.pdf      every PDF, named by record_id. Not committed
                           (copyrighted); kept on this machine only.
  fetch_log.csv            one row per record: status, route, URL used
  to_download_by_hand.csv  records that could not be fetched, grouped by
                           publisher (largest group first) and within a
                           group by decision (include first, then maybe),
                           with the title,
                           DOI and DOI link, for the author to download.
                           Save each file as pdf/<record_id>.pdf.

Routes, in order. Only open-access copies are fetched, with an honest
User-Agent; a publisher that refuses the request is not retried, and the
record goes to the by-hand list instead.
  1. OpenAlex: every open-access location with a PDF URL. OpenAlex holds
     the same open-access data as Unpaywall, so Unpaywall is not called
     and no further e-mail address is sent anywhere.
  2. PubMed Central: Europe PMC maps the DOI to a PMCID, and the PDF is
     taken from PMC's official open-data bucket (PMC Article Datasets).
  3. ERIC: the ERIC full-text PDF for ERIC records (EJ/ED numbers).
  4. OSTI: the full-text link of the US Department of Energy record.

Every download is checked: a file that does not start with "%PDF" (for
example a landing page served as HTML) is discarded and the next route is
tried.
"""
from __future__ import annotations

import argparse
import csv
import json
import re
import sys
import time
import urllib.parse
import urllib.request
from collections import Counter
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from run_search import MAILTO, OPENALEX_KEY  # noqa: E402  same contact and key as the search

KEEP = {"include", "maybe"}
UA = f"sr-update-fulltext (mailto:{MAILTO})"
MIN_BYTES = 10_000
PAUSE = 1.0


def _open(url, timeout=60):
    headers = {"User-Agent": UA,
               "Accept": "application/pdf,application/json;q=0.9,*/*;q=0.8"}
    if OPENALEX_KEY and url.startswith("https://api.openalex.org/"):
        headers["Authorization"] = f"Bearer {OPENALEX_KEY}"
    req = urllib.request.Request(url, headers=headers)
    return urllib.request.urlopen(req, timeout=timeout)


def get_json(url):
    try:
        with _open(url) as r:
            return json.load(r)
    except Exception:
        return None


def download_pdf(url, dest: Path) -> str:
    """Save url to dest if it is a PDF. Returns "" on success, else a reason."""
    try:
        with _open(url) as r:
            body = r.read()
    except Exception as e:  # HTTP errors (403 etc.), timeouts, TLS problems
        return f"{type(e).__name__}: {e}"[:120]
    if body[:5] != b"%PDF-":
        return "not a PDF (landing page)"
    if len(body) < MIN_BYTES:
        return f"PDF too small ({len(body)} bytes)"
    dest.write_bytes(body)
    return ""


# --------------------------------------------------------------------------
# Routes. Each returns a list of (route name, PDF URL) candidates.
# --------------------------------------------------------------------------

def route_openalex(doi):
    if not doi:
        return []
    w = get_json("https://api.openalex.org/works/doi:" + urllib.parse.quote(doi)
                 + "?mailto=" + urllib.parse.quote(MAILTO))
    if not w:
        return []
    out, seen = [], set()
    for loc in [w.get("best_oa_location") or {}, *(w.get("locations") or [])]:
        u = loc.get("pdf_url")
        if loc.get("is_oa") and u and u not in seen:
            seen.add(u)
            out.append(("openalex", u))
    return out


PMC_BUCKET = "https://pmc-oa-opendata.s3.amazonaws.com/"


def route_pmc(doi):
    """PubMed Central copy: Europe PMC maps the DOI to a PMCID, and the PDF is
    taken from PMC's official open-data bucket (PMC Article Datasets)."""
    if not doi:
        return []
    q = urllib.parse.quote(f'DOI:"{doi}"')
    d = get_json("https://www.ebi.ac.uk/europepmc/webservices/rest/search?query="
                 f"{q}&format=json&resultType=lite")
    hits = ((d or {}).get("resultList") or {}).get("result") or []
    out = []
    for pmcid in dict.fromkeys(h["pmcid"] for h in hits if h.get("pmcid")):
        try:
            with _open(f"{PMC_BUCKET}?list-type=2&prefix={pmcid}.") as r:
                listing = r.read().decode("utf-8", "replace")
        except Exception:
            continue
        keys = re.findall(r"<Key>([^<]+\.pdf)</Key>", listing)
        own = [k for k in keys if k.split("/")[-1].startswith(pmcid)]
        out += [("pmc", PMC_BUCKET + k) for k in sorted(own, reverse=True)[:1]]
    return out


def route_eric(row):
    ids = re.findall(r"\b(E[DJ]\d{6})\b", " ".join(
        [row.get("source_id", ""), row.get("sources", "")]))
    return [("eric", f"https://files.eric.ed.gov/fulltext/{i}.pdf")
            for i in dict.fromkeys(ids)]


def route_osti(doi):
    if not doi:
        return []
    d = get_json("https://www.osti.gov/api/v1/records?doi=" + urllib.parse.quote(doi))
    return [("osti", link["href"]) for rec in (d or [])
            for link in rec.get("links", [])
            if link.get("rel") == "fulltext" and link.get("href")]


def candidates(row):
    doi = (row.get("doi") or "").strip().lower()
    yield from route_openalex(doi)
    yield from route_pmc(doi)
    yield from route_eric(row)
    yield from route_osti(doi)


# --------------------------------------------------------------------------

# DOI prefixes of the publishers that hold most of the by-hand records.
# Grouping by publisher lets the author work through one platform at a
# time instead of jumping between sites for every record.
PUBLISHERS = {
    "10.1016": "Elsevier (ScienceDirect)",
    "10.1007": "Springer",
    "10.1080": "Taylor and Francis",
    "10.1177": "SAGE",
    "10.1037": "APA",
    "10.1111": "Wiley",
    "10.1002": "Wiley",
    "10.1109": "IEEE",
    "10.1108": "Emerald",
    "10.1145": "ACM",
    "10.3390": "MDPI",
    "10.4018": "IGI Global",
    "10.3389": "Frontiers",
    "10.1348": "Wiley (BPS)",
    "10.1207": "Taylor and Francis",
}
SCHOLAR = "https://scholar.google.com/scholar?q="


def publisher(doi):
    return PUBLISHERS.get((doi or "").split("/")[0], "その他" if doi else "DOI なし")


def search_link(row):
    """Where to go to get this PDF: the DOI, or a title search."""
    doi = (row.get("doi") or "").strip()
    if doi:
        return f"https://doi.org/{doi}"
    return SCHOLAR + urllib.parse.quote('"' + (row.get("title") or "")[:200] + '"')


def kept_in_screening(path):
    """record_ids kept at the title/abstract stage, from the screening sheet.

    Uses the reviewer's et_decision when it is filled in, and Claude's
    proposal otherwise, so retrieval can start before every row is
    confirmed. The reviewer's decision always wins where it exists.
    """
    want = {}
    with path.open(encoding="utf-8-sig") as f:
        for row in csv.DictReader(f):
            d = (row.get("et_decision") or "").strip().lower()
            if not d:
                d = (row.get("claude_suggestion") or "").strip().lower()
            if d in KEEP:
                want[row["record_id"]] = d
    return want


def select(rows, ids):
    if ids:
        want = set(ids)
        return [r for r in rows if r["record_id"] in want]
    return [r for r in rows if (r.get("ta_decision") or "").strip().lower() in KEEP]


def write_csv(path, data, fields):
    with path.open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerows(data)


def twin_notes(rows, kept):
    """record_id -> the other record_id holding the same paper.

    The search kept some papers twice: an ERIC record with no DOI, and a
    record with a DOI, under different ids and sometimes different years
    (print vs online). Both can sit in the to-download list, so the note
    says which pairs are one paper. place_pdfs.py copies a placed PDF to
    the twin, so only one download is needed.
    """
    import html as _html
    groups = {}
    for r in rows:
        n = _norm_title(r["title"])
        if len(n) >= 25:
            groups.setdefault(n, []).append(r)
    out = {}
    for v in groups.values():
        if len(v) < 2:
            continue
        ys = [int(x["year"]) if (x["year"] or "").isdigit() else 0 for x in v]
        if max(ys) - min(ys) > 1:
            continue
        ids = [x["record_id"] for x in v if x["record_id"] in kept]
        if len(ids) > 1:
            for a in ids:
                out[a] = [b for b in ids if b != a]
    return out


def _norm_title(t):
    import html as _html
    import unicodedata as _ud
    t = _html.unescape(_html.unescape(t or ""))
    t = _ud.normalize("NFKD", t).encode("ascii", "ignore").decode().lower()
    return re.sub(r"[^a-z0-9]+", " ", t).strip()


def rebuild_lists(args, rows):
    """Rewrite the by-hand list from the existing log, with no network use.

    Used when the screening decisions changed: records dropped at the
    title/abstract stage leave the list, and the publisher grouping and
    the decision column are rebuilt from the current screening sheet.
    """
    rec = {r["record_id"]: r for r in rows}
    kept = (kept_in_screening(args.screening) if args.screening.exists() else {})
    log_path = args.out / "fetch_log.csv"
    if not log_path.exists():
        sys.exit(f"{log_path} not found. Run the fetch first.")
    with log_path.open(encoding="utf-8") as f:
        log = list(csv.DictReader(f))
    pdf_dir = args.out / "pdf"
    twin = twin_notes(rows, set(kept))
    by_hand, dropped, now_have = [], 0, 0
    for e in log:
        rid = e["record_id"]
        if e["status"] != "by_hand":
            continue
        if kept and rid not in kept:
            dropped += 1
            continue
        dest = pdf_dir / f"{rid}.pdf"
        if dest.exists() and dest.read_bytes()[:5] == b"%PDF-":
            now_have += 1
            continue
        row = rec.get(rid, {})
        doi = (row.get("doi") or "").strip()
        by_hand.append({
            "record_id": rid, "decision": kept.get(rid, ""),
            "publisher": publisher(doi), "title": row.get("title", ""),
            "year": row.get("year", ""), "authors": row.get("authors", ""),
            "journal": row.get("journal", ""), "doi": doi,
            "link": search_link(row), "save_as": f"pdf/{rid}.pdf",
            "note": " ".join(filter(None, [
                "" if doi else "no DOI; the link is a title search",
                ("same paper as " + ", ".join(twin[rid]) + "; download once")
                if rid in twin else ""]))})
    write_by_hand(args.out, by_hand)
    print(f"{len(by_hand)} to download by hand: "
          f"{args.out / 'to_download_by_hand.csv'}")
    if now_have:
        print(f"{now_have} records left the list: the PDF is now in {pdf_dir.name}/")
    if dropped:
        print(f"{dropped} records left the list: no longer kept at title/abstract")


def write_by_hand(out, by_hand):
    rank = {"include": 0, "maybe": 1}
    size = Counter(r["publisher"] for r in by_hand)
    by_hand.sort(key=lambda r: (-size[r["publisher"]], r["publisher"],
                                rank.get(r["decision"], 2), r["record_id"]))
    write_csv(out / "to_download_by_hand.csv", by_hand,
              ["record_id", "decision", "publisher", "title", "year", "authors",
               "journal", "doi", "link", "save_as", "note"])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--records", type=Path, default=HERE / "records_dedup.csv")
    ap.add_argument("--out", type=Path, default=HERE / "fulltext")
    ap.add_argument("--ids", nargs="*")
    ap.add_argument("--rebuild-lists", action="store_true",
                    help="rewrite to_download_by_hand.csv from fetch_log.csv "
                         "without fetching anything, after the screening "
                         "decisions changed")
    ap.add_argument("--screening", type=Path,
                    default=HERE / "screening" / "screening_ta.csv",
                    help="screening sheet to take the kept record_ids from; "
                         "ignored when --ids is given or the file is missing")
    args = ap.parse_args()

    if not args.records.exists():
        sys.exit(f"{args.records} not found. Run the search and the "
                 "title/abstract screening first.")
    with args.records.open(encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    ids, kept = args.ids, {}
    if args.rebuild_lists:
        rebuild_lists(args, rows)
        return
    if not ids and args.screening.exists():
        kept = kept_in_screening(args.screening)
        ids = list(kept)
        print(f"{len(ids)} records kept at title/abstract in {args.screening.name}")
    todo = select(rows, ids)
    if not todo:
        sys.exit("No records selected. Fill et_decision in the screening "
                 "sheet, mark rows 'include' or 'maybe' in ta_decision, or "
                 "pass --ids.")

    pdf_dir = args.out / "pdf"
    pdf_dir.mkdir(parents=True, exist_ok=True)
    log, by_hand = [], []
    for n, row in enumerate(todo, 1):
        rid = row["record_id"]
        dest = pdf_dir / f"{rid}.pdf"
        entry = {"record_id": rid, "doi": row.get("doi", ""),
                 "title": row.get("title", ""), "status": "", "route": "",
                 "url": "", "note": ""}
        if dest.exists() and dest.read_bytes()[:5] == b"%PDF-":
            entry["status"] = "already_present"
        else:
            notes = []
            for route, url in candidates(row):
                reason = download_pdf(url, dest)
                time.sleep(PAUSE)
                if not reason:
                    entry.update(status="downloaded", route=route, url=url)
                    break
                notes.append(f"{route}: {reason}")
            else:
                entry.update(status="by_hand",
                             note=" | ".join(notes) or "no open-access copy found")
                doi = row.get("doi", "")
                by_hand.append({
                    "record_id": rid, "decision": kept.get(rid, ""),
                    "publisher": publisher(doi),
                    "title": row.get("title", ""),
                    "year": row.get("year", ""), "authors": row.get("authors", ""),
                    "journal": row.get("journal", ""),
                    "doi": doi, "link": search_link(row),
                    "save_as": f"pdf/{rid}.pdf",
                    "note": "" if doi else "no DOI; the link is a title search"})
        log.append(entry)
        print(f"[{n}/{len(todo)}] {rid}: {entry['status']} {entry['route']}")

    # Merge into the existing log instead of replacing it, so a run over
    # a few --ids does not erase the record of the full run.
    log_path = args.out / "fetch_log.csv"
    merged = {}
    if log_path.exists():
        with log_path.open(encoding="utf-8") as f:
            for e in csv.DictReader(f):
                merged[e["record_id"]] = e
    for e in log:
        merged[e["record_id"]] = e
    write_csv(log_path, sorted(merged.values(), key=lambda e: e["record_id"]),
              ["record_id", "doi", "title", "status", "route", "url", "note"])
    write_by_hand(args.out, by_hand)
    got = sum(e["status"] in ("downloaded", "already_present") for e in log)
    print(f"\n{got} of {len(log)} PDFs in {pdf_dir}")
    print(f"{len(by_hand)} to download by hand: {args.out / 'to_download_by_hand.csv'}")


if __name__ == "__main__":
    main()
