"""Fill in the abstracts that the search did not return.

    python metaanalysis/sr_update/fetch_abstracts.py --check
    python metaanalysis/sr_update/fetch_abstracts.py              # every record with no abstract
    python metaanalysis/sr_update/fetch_abstracts.py --ids R00034 R00040
    python metaanalysis/sr_update/fetch_abstracts.py --kept-only  # only the full-text candidates

Why: 1,230 of the 8,670 records reached the title/abstract stage with no
abstract, almost all of them from citation searching and OpenAlex. They
had to be judged from the title alone, which pushes them to `maybe` and
so into the full-text pile. Recovering the abstract lets them be judged
properly at the title/abstract stage, which both shrinks that pile and
reduces the chance of excluding an eligible study from its title.

This adds no records and changes no eligibility criterion. It only
completes the metadata of records the registered search already found.

Routes, in order, all free and all with an honest User-Agent:
  1. Crossref        api.crossref.org/works/<doi>        (abstract, JATS)
  2. Europe PMC      ebi.ac.uk/europepmc/.../search      (abstractText)
  3. PubMed          eutils efetch by DOI                (AbstractText)
Records without a DOI are searched by title and accepted only when the
normalised title matches exactly. A DOI lookup is accepted only when the
record that comes back carries that same DOI: PubMed answers an unknown
DOI by searching its parts and returning a different paper, which in the
first run of 2026-10-28 attached one paramedic abstract to seven Springer
book chapters.

Output:
  abstracts_added.csv   record_id, route, chars, title  (committed; no abstracts)
  records_dedup.csv     the abstract column is filled in place, and the
                        route is recorded in a new abstract_source column.
                        A copy of the previous file is kept next to it as
                        records_dedup.csv.bak before anything is written.
"""
from __future__ import annotations

import argparse
import csv
import html
import json
import re
import sys
import time
import unicodedata
import urllib.parse
import urllib.request
import xml.etree.ElementTree as ET
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

MAILTO = "eisuke.tokiwa@sunblaze.jp"
CROSSREF = "https://api.crossref.org/works/"
EUROPEPMC = "https://www.ebi.ac.uk/europepmc/webservices/rest/search"
EUTILS = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/"
PAUSE = 0.15
MIN_CHARS = 120  # shorter than this is a label or a stub, not an abstract

csv.field_size_limit(10 ** 7)


def get(url, retries=3):
    for attempt in range(retries):
        try:
            req = urllib.request.Request(
                url, headers={"User-Agent": f"sr-update (mailto:{MAILTO})"})
            with urllib.request.urlopen(req, timeout=60) as resp:
                return resp.read()
        except Exception:
            if attempt == retries - 1:
                raise
            time.sleep(2 ** attempt)


def norm(t):
    t = unicodedata.normalize("NFKD", t or "").encode("ascii", "ignore").decode().lower()
    return re.sub(r"[^a-z0-9]+", " ", t).strip()


def clean(text):
    """Plain text from JATS or HTML, with the leading 'Abstract' label gone."""
    if not text:
        return ""
    text = re.sub(r"<[^>]+>", " ", text)
    text = html.unescape(text)
    text = re.sub(r"\s+", " ", text).strip()
    text = re.sub(r"^(abstract|summary)[:.\s-]+", "", text, flags=re.I)
    return text.strip()


# --------------------------------------------------------------------------
# routes
# --------------------------------------------------------------------------

def from_crossref(row):
    doi = (row.get("doi") or "").strip()
    if not doi:
        return ""
    url = CROSSREF + urllib.parse.quote(doi, safe="") + "?mailto=" + MAILTO
    msg = json.loads(get(url)).get("message", {})
    return clean(msg.get("abstract", ""))


def _europepmc(query):
    url = (EUROPEPMC + "?query=" + urllib.parse.quote(query)
           + "&resultType=core&pageSize=5&format=json")
    return json.loads(get(url)).get("resultList", {}).get("result", [])


def from_europepmc(row):
    doi = (row.get("doi") or "").strip()
    hits = _europepmc(f'DOI:"{doi}"') if doi else _europepmc(f'TITLE:"{row["title"]}"')
    want, want_doi = norm(row["title"]), doi.lower()
    for h in hits:
        # Accept a hit only when it is provably the same work: the same DOI,
        # or the same normalised title. A search that matches neither has
        # found a different paper.
        if want_doi and (h.get("doi") or "").strip().lower() == want_doi:
            return clean(h.get("abstractText", ""))
        if norm(h.get("title", "")) == want:
            return clean(h.get("abstractText", ""))
    return ""


def from_pubmed(row):
    doi = (row.get("doi") or "").strip()
    if not doi:
        return ""
    url = (EUTILS + "esearch.fcgi?db=pubmed&retmode=json&email=" + MAILTO
           + "&term=" + urllib.parse.quote(f"{doi}[DOI]"))
    ids = json.loads(get(url)).get("esearchresult", {}).get("idlist", [])
    if not ids:
        return ""
    url = (EUTILS + "efetch.fcgi?db=pubmed&retmode=xml&email=" + MAILTO
           + "&id=" + ids[0])
    root = ET.fromstring(get(url))
    # PubMed does not fail on an unknown DOI: it splits the string into
    # terms and returns whatever matches, which is a different paper. Only
    # accept the record when it carries the DOI we asked for.
    got = [(e.text or "").strip().lower() for e in root.iter("ArticleId")
           if e.get("IdType") == "doi"]
    if doi.lower() not in got:
        return ""
    parts = [(e.text or "") for e in root.iter("AbstractText")]
    return clean(" ".join(parts))


def from_crossref_title(row):
    """Crossref by title, accepted only on an exact normalised title match."""
    if (row.get("doi") or "").strip():
        return ""
    url = (CROSSREF.rstrip("/") + "?rows=5&mailto=" + MAILTO
           + "&query.bibliographic=" + urllib.parse.quote(row["title"][:300]))
    items = json.loads(get(url)).get("message", {}).get("items", [])
    want = norm(row["title"])
    for it in items:
        titles = it.get("title") or []
        if titles and norm(titles[0]) == want:
            return clean(it.get("abstract", ""))
    return ""


ROUTES = [("crossref", from_crossref), ("europepmc", from_europepmc),
          ("pubmed", from_pubmed), ("crossref_title", from_crossref_title)]


# --------------------------------------------------------------------------

def check():
    ok = True
    for name, url in [("crossref", CROSSREF + "10.1038/nature12373?mailto=" + MAILTO),
                      ("europepmc", EUROPEPMC + "?query=DOI:%2210.1038/nature12373%22&format=json"),
                      ("pubmed", EUTILS + "einfo.fcgi?retmode=json&email=" + MAILTO)]:
        try:
            get(url)
            print(f"{name:16} reachable")
        except Exception as e:
            ok = False
            print(f"{name:16} FAILED: {e}")
    return ok


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--records", type=Path, default=HERE / "records_dedup.csv")
    ap.add_argument("--screening", type=Path,
                    default=HERE / "screening" / "screening_ta.csv")
    ap.add_argument("--ids", nargs="*")
    ap.add_argument("--kept-only", action="store_true",
                    help="only records kept at the title/abstract stage")
    ap.add_argument("--check", action="store_true")
    args = ap.parse_args()

    if args.check:
        sys.exit(0 if check() else 1)

    with args.records.open(encoding="utf-8") as f:
        reader = csv.DictReader(f)
        fields = list(reader.fieldnames)
        rows = list(reader)
    if "abstract_source" not in fields:
        fields.append("abstract_source")

    todo = [r for r in rows if not (r.get("abstract") or "").strip()]
    if args.ids:
        want = set(args.ids)
        todo = [r for r in todo if r["record_id"] in want]
    elif args.kept_only and args.screening.exists():
        keep = set()
        with args.screening.open(encoding="utf-8-sig") as f:
            for r in csv.DictReader(f):
                d = ((r.get("et_decision") or "").strip().lower()
                     or (r.get("claude_suggestion") or "").strip().lower())
                if d in ("include", "maybe"):
                    keep.add(r["record_id"])
        todo = [r for r in todo if r["record_id"] in keep]
    if not todo:
        sys.exit("No records without an abstract to fill.")

    print(f"{len(todo)} records with no abstract")
    added = []
    for n, row in enumerate(todo, 1):
        got, route = "", ""
        for name, fn in ROUTES:
            try:
                got = fn(row)
            except Exception as e:
                got = ""
                print(f"  {row['record_id']} {name}: {type(e).__name__}: {e}")
            time.sleep(PAUSE)
            if len(got) >= MIN_CHARS:
                route = name
                break
            got = ""
        if route:
            row["abstract"] = got
            row["abstract_source"] = route
            added.append({"record_id": row["record_id"], "route": route,
                          "chars": len(got), "title": row.get("title", "")})
        print(f"[{n}/{len(todo)}] {row['record_id']}: {route or 'not found'}")

    bak = args.records.with_suffix(args.records.suffix + ".bak")
    bak.write_bytes(args.records.read_bytes())
    with args.records.open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
        w.writeheader()
        w.writerows(rows)
    with (HERE / "abstracts_added.csv").open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["record_id", "route", "chars", "title"])
        w.writeheader()
        w.writerows(added)

    import collections
    print(f"\n{len(added)} of {len(todo)} abstracts recovered")
    print("routes:", dict(collections.Counter(a["route"] for a in added)))
    print(f"previous file kept as {bak.name}")


if __name__ == "__main__":
    main()
