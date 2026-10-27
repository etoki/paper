"""Run the systematic-review update search and log every record.

    python metaanalysis/sr_update/run_search.py --check       # connectivity only
    python metaanalysis/sr_update/run_search.py               # databases + snowballing
    python metaanalysis/sr_update/run_search.py --only pubmed
    python metaanalysis/sr_update/run_search.py --merge       # add Google Scholar, re-dedup

Outputs (all under sr_update/):
  raw/<source>/          every API response, unmodified
  records_<source>.csv   normalised records per source
  records_dedup.csv      de-duplicated union, with empty screening columns
  search_log.json        date, exact queries, hit counts (registered-only
                         and amended), and the PRISMA identification numbers

Sources: PubMed (E-utilities), ERIC (IES API), OpenAlex, and OpenAlex
backward/forward citation searching from seeds.csv. Google Scholar is run
by hand with Publish or Perish (search_strategy.GOOGLE_SCHOLAR_QUERY) and its
CSV export saved as scholar/pop_export.csv; --merge then converts it to
records_scholar.csv and rebuilds records_dedup.csv from every records_*.csv.
Run --merge before screening starts, because it renumbers record_ids. Set OPENALEX_API_KEY
if OpenAlex asks for a key.

Written before the computing environment could reach these APIs; run
--check first and look at the first page of each source before trusting
the counts.
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import re
import sys
import time
import unicodedata
import urllib.error
import urllib.parse
import urllib.request
import xml.etree.ElementTree as ET
from datetime import date
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from search_strategy import (DATABASES, GOOGLE_SCHOLAR_MAX,  # noqa: E402
                             GOOGLE_SCHOLAR_QUERY, online_block)

MAILTO = "eisuke.tokiwa@sunblaze.jp"  # NCBI / OpenAlex contact address
KEY_FILE = HERE / ".openalex_key"  # one line, the key only; git-ignored
OPENALEX_KEY = (os.environ.get("OPENALEX_API_KEY", "")
                or (KEY_FILE.read_text(encoding="utf-8").strip()
                    if KEY_FILE.exists() else ""))
EUTILS = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/"
ERIC = "https://api.ies.ed.gov/eric/"
OPENALEX = "https://api.openalex.org/works"
HOSTS = {
    "pubmed": EUTILS + "einfo.fcgi?retmode=json",
    "eric": ERIC + "?search=personality&rows=1&format=json",
    "openalex": OPENALEX + "?per-page=1",
}
FIELDS = ["source", "source_id", "doi", "title", "abstract", "year", "authors",
          "journal"]
SCREEN = ["ta_decision", "ta_reason", "ft_decision", "ft_reason"]


def get(url, data=None, retries=3):
    for attempt in range(retries):
        try:
            headers = {"User-Agent": f"sr-update (mailto:{MAILTO})"}
            if OPENALEX_KEY and url.startswith(OPENALEX):
                headers["Authorization"] = f"Bearer {OPENALEX_KEY}"
            req = urllib.request.Request(
                url, data=urllib.parse.urlencode(data).encode() if data else None,
                headers=headers)
            with urllib.request.urlopen(req, timeout=60) as resp:
                return resp.read()
        except Exception:
            if attempt == retries - 1:
                raise
            time.sleep(2 ** attempt)


def check():
    ok = True
    for db, url in HOSTS.items():
        try:
            get(url, retries=1)
            print(f"  OK    {db}: {url.split('/')[2]}")
        except Exception as exc:  # noqa: BLE001
            ok = False
            print(f"  FAIL  {db}: {url.split('/')[2]} ({exc})")
    print(f"  OpenAlex key: {'set' if OPENALEX_KEY else 'not set (shared free budget)'}; "
          f"remaining today: {openalex_budget()}")
    return ok


def openalex_budget():
    """Remaining OpenAlex budget in USD, from the rate-limit headers."""
    headers = {"User-Agent": f"sr-update (mailto:{MAILTO})"}
    if OPENALEX_KEY:
        headers["Authorization"] = f"Bearer {OPENALEX_KEY}"
    req = urllib.request.Request(HOSTS["openalex"], headers=headers)
    try:
        with urllib.request.urlopen(req, timeout=30) as resp:
            h = resp.headers
    except urllib.error.HTTPError as e:
        h = e.headers
    except Exception as exc:  # noqa: BLE001
        return f"unknown ({exc})"
    return (f"${h.get('X-RateLimit-Remaining-USD', '?')} of "
            f"${h.get('X-RateLimit-Limit-USD', '?')}")


def save_raw(source, name, payload):
    d = HERE / "raw" / source
    d.mkdir(parents=True, exist_ok=True)
    (d / name).write_bytes(payload)


def clean_doi(doi):
    doi = (doi or "").strip().lower()
    return re.sub(r"^(https?://(dx\.)?doi\.org/|doi:)", "", doi)


# --------------------------------------------------------------------------
# PubMed: esearch (POST, history server) + efetch XML in batches of 200
# --------------------------------------------------------------------------
def pubmed_count(query):
    meta = json.loads(get(EUTILS + "esearch.fcgi", {
        "db": "pubmed", "term": query, "retmode": "json", "retmax": 0,
        "email": MAILTO}))["esearchresult"]
    return int(meta["count"])


def search_pubmed(query):
    meta = json.loads(get(EUTILS + "esearch.fcgi", {
        "db": "pubmed", "term": query, "usehistory": "y", "retmode": "json",
        "retmax": 0, "email": MAILTO}))["esearchresult"]
    count = int(meta["count"])
    records = []
    for start in range(0, count, 200):
        xml = get(EUTILS + "efetch.fcgi", {
            "db": "pubmed", "query_key": meta["querykey"], "WebEnv": meta["webenv"],
            "retstart": start, "retmax": 200, "retmode": "xml", "email": MAILTO})
        save_raw("pubmed", f"efetch_{start:05d}.xml", xml)
        for art in ET.fromstring(xml).iter("PubmedArticle"):
            title = art.find(".//ArticleTitle")
            year = (art.findtext(".//PubDate/Year")
                    or (art.findtext(".//PubDate/MedlineDate") or "")[:4])
            records.append({
                "source": "pubmed",
                "source_id": art.findtext(".//PMID", ""),
                "doi": clean_doi(next((i.text for i in art.iter("ArticleId")
                                       if i.get("IdType") == "doi"), "")),
                "title": "".join(title.itertext()) if title is not None else "",
                "abstract": " ".join("".join(a.itertext())
                                     for a in art.iter("AbstractText")),
                "year": year,
                "authors": "; ".join(
                    f"{a.findtext('LastName', '')} {a.findtext('Initials', '')}".strip()
                    for a in art.iter("Author")),
                "journal": art.findtext(".//Journal/Title", ""),
            })
        time.sleep(0.4)  # NCBI: at most 3 requests per second without a key
    return count, records


# --------------------------------------------------------------------------
# ERIC: paged JSON (English only via the language filter)
# --------------------------------------------------------------------------
ERIC_FIELDS = "id,title,author,description,publicationdateyear,source,language,url"


def eric_page(query, start, rows):
    return get(ERIC + "?" + urllib.parse.urlencode({
        "search": query, "format": "json", "rows": rows, "start": start,
        "fields": ERIC_FIELDS}))


def eric_count(query):
    return int(json.loads(eric_page(query, 0, 0))["response"]["numFound"])


def search_eric(query):
    records, start, count = [], 0, None
    while count is None or start < count:
        raw = eric_page(query, start, 200)
        save_raw("eric", f"page_{start:05d}.json", raw)
        resp = json.loads(raw)["response"]
        count = int(resp["numFound"])
        for doc in resp["docs"]:
            lang = doc.get("language") or []
            if lang and "English" not in lang:
                continue
            records.append({
                "source": "eric",
                "source_id": doc.get("id", ""),
                "doi": "",
                "title": doc.get("title", ""),
                "abstract": doc.get("description", ""),
                "year": str(doc.get("publicationdateyear", "")),
                "authors": "; ".join(doc.get("author") or []),
                "journal": doc.get("source", ""),
            })
        if not resp["docs"]:
            break
        start += 200
    return count, records


# --------------------------------------------------------------------------
# OpenAlex: title_and_abstract.search with cursor paging; citation searching
# --------------------------------------------------------------------------
def _abstract(inv):
    if not inv:
        return ""
    return " ".join(w for _, w in sorted((i, w) for w, idx in inv.items() for i in idx))


def _openalex_record(w, source):
    return {
        "source": source,
        "source_id": w.get("id", ""),
        "doi": clean_doi(w.get("doi")),
        "title": w.get("title") or "",
        "abstract": _abstract(w.get("abstract_inverted_index")),
        "year": str(w.get("publication_year") or ""),
        "authors": "; ".join(a["author"]["display_name"]
                             for a in w.get("authorships", [])),
        "journal": ((w.get("primary_location") or {}).get("source")
                    or {}).get("display_name", ""),
    }


def openalex_works(filt, source, tag):
    """All works matching an OpenAlex filter string, via cursor paging."""
    records, cursor, count, page = [], "*", 0, 0
    while cursor:
        params = {"filter": filt, "per-page": 200, "cursor": cursor, "mailto": MAILTO}
        raw = get(OPENALEX + "?" + urllib.parse.urlencode(params))
        save_raw(source, f"{tag}_{page:04d}.json", raw)
        data = json.loads(raw)
        count = data["meta"]["count"]
        records += [_openalex_record(w, source) for w in data["results"]]
        cursor = data["meta"].get("next_cursor") if data["results"] else None
        page += 1
    return count, records


def openalex_filter(query):
    return f"title_and_abstract.search:{query},language:en"


def openalex_count(query):
    params = {"filter": openalex_filter(query), "per-page": 1, "mailto": MAILTO}
    return json.loads(get(OPENALEX + "?" + urllib.parse.urlencode(params)))["meta"]["count"]


def search_openalex(query):
    return openalex_works(openalex_filter(query), "openalex", "page")


def openalex_by_doi(doi):
    params = {"mailto": MAILTO}
    url = f"{OPENALEX}/https://doi.org/{doi}?" + urllib.parse.urlencode(params)
    return json.loads(get(url))


def search_citations():
    """Backward (reference lists) and forward (citing works) searching from
    every seed in seeds.csv that has a DOI or an OpenAlex id (several ids,
    separated by "|", when OpenAlex holds duplicate records of one work).
    Returns per-seed counts and the records; seeds with neither are reported
    for manual searching."""
    seeds = list(csv.DictReader((HERE / "seeds.csv").open(encoding="utf-8")))
    per_seed, records, missing = [], [], []
    for s in seeds:
        ids = [i for i in (s.get("openalex_id") or "").split("|") if i]
        if not s["doi"] and not ids:
            missing.append(s["seed_id"])
            continue
        try:
            works = ([openalex_by_doi(s["doi"])] if s["doi"]
                     else [json.loads(get(f"{OPENALEX}/{i}?mailto={MAILTO}")) for i in ids])
        except Exception as exc:  # noqa: BLE001
            missing.append(f"{s['seed_id']} (not found in OpenAlex: {exc})")
            continue
        wid = "|".join(w["id"].rsplit("/", 1)[-1] for w in works)
        refs = list(dict.fromkeys(r.rsplit("/", 1)[-1] for w in works
                                  for r in w.get("referenced_works", [])))
        back = []
        for i in range(0, len(refs), 50):  # OR-filter takes up to 50 ids
            _, recs = openalex_works("ids.openalex:" + "|".join(refs[i:i + 50]),
                                     "citations", f"{s['seed_id']}_refs{i // 50}")
            back += recs
        # Benchmark meta-analyses are cited thousands of times, mostly by
        # face-to-face studies, so their citing works must also match the
        # online-learning block; included studies' citing works are not filtered.
        filt = f"cites:{wid}"
        if s["role"] == "benchmark":
            filt += f",title_and_abstract.search:{online_block()}"
        _, fwd = openalex_works(filt, "citations", f"{s['seed_id']}_citedby")
        for r in back + fwd:
            r["source"] = "citations"
        per_seed.append({"seed_id": s["seed_id"], "openalex_id": wid,
                         "backward": len(back), "forward": len(fwd)})
        records += back + fwd
    return per_seed, records, missing


SEARCHERS = {"pubmed": search_pubmed, "eric": search_eric, "openalex": search_openalex}
COUNTERS = {"pubmed": pubmed_count, "eric": eric_count, "openalex": openalex_count}


# --------------------------------------------------------------------------
# De-duplication: DOI first, then normalised title + year
# --------------------------------------------------------------------------
def norm_title(t):
    t = unicodedata.normalize("NFKD", t).encode("ascii", "ignore").decode().lower()
    return re.sub(r"[^a-z0-9]+", " ", t).strip()


def dedup(records):
    seen, out = {}, []
    for r in records:
        keys = [("title", norm_title(r["title"]), r["year"])]
        if r["doi"]:
            keys.insert(0, ("doi", r["doi"]))
        hit = next((seen[k] for k in keys if k in seen), None)
        if hit is not None:
            hit["sources"] = ";".join(sorted(set(hit["sources"].split(";")) | {r["source"]}))
            for f in ("doi", "abstract"):
                if not hit[f] and r[f]:
                    hit[f] = r[f]
                    if f == "doi":
                        seen[("doi", r["doi"])] = hit
            continue
        rec = {**r, "sources": r["source"], "record_id": f"R{len(out) + 1:05d}",
               **{c: "" for c in SCREEN}}
        for k in keys:
            seen[k] = rec
        out.append(rec)
    return out


def write_csv(path, rows, fields):
    with path.open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
        w.writeheader()
        w.writerows(rows)


SOURCES_ORDER = ["pubmed", "eric", "openalex", "citations", "scholar"]
POP_CSV = HERE / "scholar" / "pop_export.csv"


def read_csv(path):
    with path.open(encoding="utf-8-sig", newline="") as f:
        return list(csv.DictReader(f))


def convert_pop(path):
    """Publish or Perish CSV export -> normalised records (source 'scholar')."""
    recs = []
    for row in read_csv(path)[:GOOGLE_SCHOLAR_MAX]:
        recs.append({
            "source": "scholar",
            "source_id": row.get("ArticleURL", "") or row.get("GSRank", ""),
            "doi": clean_doi(row.get("DOI")),
            "title": row.get("Title", ""),
            "abstract": row.get("Abstract", ""),
            "year": (row.get("Year") or "").strip(),
            "authors": row.get("Authors", ""),
            "journal": row.get("Source", ""),
        })
    return recs


def finalize(all_records, log):
    unique = dedup(all_records)
    write_csv(HERE / "records_dedup.csv", unique,
              ["record_id", "sources", *FIELDS[1:], *SCREEN])
    log["prisma"] = {"records_identified": len(all_records),
                     "duplicates_removed": len(all_records) - len(unique),
                     "records_screened": len(unique)}
    (HERE / "search_log.json").write_text(
        json.dumps(log, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"  total {len(all_records)}, after de-duplication {len(unique)}")


def merge():
    """Add the Publish or Perish export and rebuild records_dedup.csv."""
    log_path = HERE / "search_log.json"
    if not log_path.exists():
        sys.exit("search_log.json not found: run the database search first.")
    log = json.loads(log_path.read_text(encoding="utf-8"))
    if not POP_CSV.exists():
        sys.exit(f"{POP_CSV} not found: save the Publish or Perish CSV there.")
    recs = convert_pop(POP_CSV)
    write_csv(HERE / "records_scholar.csv", recs, FIELDS)
    log["sources"]["scholar"] = {
        "name": "Google Scholar (Publish or Perish, run by the author)",
        "query": GOOGLE_SCHOLAR_QUERY, "records_retrieved": len(recs),
        "max_records": GOOGLE_SCHOLAR_MAX,
        "export_file": str(POP_CSV.relative_to(HERE)),
        "export_modified": date.fromtimestamp(POP_CSV.stat().st_mtime).isoformat()}
    print(f"  scholar: {len(recs)} records")
    all_records = []
    for src in SOURCES_ORDER:
        path = HERE / f"records_{src}.csv"
        if path.exists():
            all_records += read_csv(path)
    finalize(all_records, log)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--check", action="store_true")
    ap.add_argument("--only", choices=[*SEARCHERS, "citations"])
    ap.add_argument("--merge", action="store_true")
    args = ap.parse_args()
    if args.merge:
        merge()
        return

    print("Connectivity:")
    if not check() and not args.check:
        sys.exit("Some databases are unreachable; allow the hosts first.")
    if args.check:
        return

    log = {"date": date.today().isoformat(), "sources": {}}
    all_records = []
    for db in ([args.only] if args.only in SEARCHERS else [] if args.only else SEARCHERS):
        query = DATABASES[db]["query"]()
        count, recs = SEARCHERS[db](query)
        write_csv(HERE / f"records_{db}.csv", recs, FIELDS)
        log["sources"][db] = {
            "name": DATABASES[db]["name"], "query": query,
            "query_preregistered": DATABASES[db]["query"](amended=False),
            "hits": count, "records_retrieved": len(recs),
            "hits_preregistered_terms_only": COUNTERS[db](DATABASES[db]["query"](amended=False)),
        }
        print(f"  {db}: {count} hits, {len(recs)} retrieved")
        all_records += recs

    if args.only in (None, "citations"):
        per_seed, recs, missing = search_citations()
        write_csv(HERE / "records_citations.csv", recs, FIELDS)
        log["sources"]["citations"] = {
            "name": "OpenAlex backward and forward citation searching",
            "per_seed": per_seed, "records_retrieved": len(recs),
            "seeds_to_search_by_hand": missing}
        print(f"  citations: {len(recs)} records from {len(per_seed)} seeds; "
              f"by hand: {', '.join(missing) or 'none'}")
        all_records += recs

    finalize(all_records, log)


if __name__ == "__main__":
    main()
