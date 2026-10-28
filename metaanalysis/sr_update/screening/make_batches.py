"""Split the records to screen into batches for the title/abstract stage.

    python metaanalysis/sr_update/screening/make_batches.py            # all records
    python metaanalysis/sr_update/screening/make_batches.py --pilot    # pilot set only

Input: ../records_dedup.csv (local only; it holds abstracts).
Output: batches/batch_NNN.jsonl, one record per line with record_id,
title, abstract, year and journal. Batches hold abstracts, so they are
git-ignored.

Order: records that mention both a personality term and an online-learning
term come first, then records that mention either, then the rest; within a
group, by record_id. The order only decides which records are proposed and
checked first; every record is screened.

--rescreen writes batches/rescreen_NNN.jsonl: the records whose abstract
fetch_abstracts.py recovered after the first pass, so they can be judged
from the abstract instead of the title alone.

--pilot writes batches/pilot.jsonl: the records of the 28 studies from
manuscript v4 that the search found, plus a fixed random sample of other
records (seed 20261027), shuffled together so the known studies are not
recognisable by position.
"""
from __future__ import annotations

import argparse
import csv
import json
import random
import re
import unicodedata
from pathlib import Path

HERE = Path(__file__).resolve().parent
SR = HERE.parent
ROOT = SR.parent
OUT = HERE / "batches"
BATCH_SIZE = 200
SEED = 20261027
PILOT_RANDOM = 72

PERSONALITY = ["big five", "big-five", "five factor", "five-factor", "personality",
               "conscientious", "neurotic", "emotional stability", "extravers",
               "extrovers", "agreeable", "openness", "hexaco"]
ONLINE = ["online", "e-learning", "elearning", "distance", "mooc", "virtual",
          "blended", "remote", "web-based", "learning management", " lms"]

V4_TITLES = {
    "A-02": "The impact of motivation and personality on academic performance in online and blended learning environments",
    "A-03": "Personality traits and e-learning course satisfaction: A study of health science students",
    "A-19": "Endorsed, or just enforced? Personality and preferences for online learning during COVID-19",
    "A-21": "Impact of learner's characteristics and learning behaviour on learning performance during a fully online course",
}


def norm(t):
    t = unicodedata.normalize("NFKD", t or "").encode("ascii", "ignore").decode().lower()
    return re.sub(r"[^a-z0-9]+", " ", t).strip()


def tier(r):
    text = f" {r['title']} {r['abstract']} ".lower()
    p = any(w in text for w in PERSONALITY)
    o = any(w in text for w in ONLINE)
    return 0 if (p and o) else 1 if (p or o) else 2


def slim(r):
    return {"record_id": r["record_id"], "title": r["title"],
            "abstract": r["abstract"], "year": r["year"], "journal": r["journal"]}


def known_v4_records(rows):
    """record_ids of the v4 studies (by DOI, or title for those without)."""
    by_doi = {r["doi"].lower(): r["record_id"] for r in rows if r["doi"]}
    by_title = {norm(r["title"]): r["record_id"] for r in rows if norm(r["title"])}
    found = {}
    with (ROOT / "analysis" / "data_extraction_populated.csv").open(encoding="utf-8") as f:
        for e in csv.DictReader(f):
            rid = by_doi.get(e["doi"].lower()) if e["doi"] else None
            if not rid and e["study_id"] in V4_TITLES:
                rid = by_title.get(norm(V4_TITLES[e["study_id"]]))
            if rid:
                found[e["study_id"]] = rid
    return found


def write_jsonl(path, recs):
    with path.open("w", encoding="utf-8") as f:
        for r in recs:
            f.write(json.dumps(slim(r), ensure_ascii=False) + "\n")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pilot", action="store_true")
    ap.add_argument("--rescreen", action="store_true",
                    help="only the records whose abstract was recovered later "
                         "by fetch_abstracts.py, for a second pass")
    args = ap.parse_args()
    with (SR / "records_dedup.csv").open(encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    OUT.mkdir(exist_ok=True)

    if args.pilot:
        known = known_v4_records(rows)
        ids = set(known.values())
        rng = random.Random(SEED)
        others = rng.sample([r for r in rows if r["record_id"] not in ids], PILOT_RANDOM)
        pilot = [r for r in rows if r["record_id"] in ids] + others
        rng.shuffle(pilot)
        write_jsonl(OUT / "pilot.jsonl", pilot)
        (OUT / "pilot_known.json").write_text(
            json.dumps(known, indent=1, ensure_ascii=False), encoding="utf-8")
        print(f"pilot: {len(pilot)} records ({len(known)} known v4 studies)")
        return

    if args.rescreen:
        path = SR / "abstracts_added.csv"
        if not path.exists():
            raise SystemExit(f"{path} not found. Run fetch_abstracts.py first.")
        with path.open(encoding="utf-8") as f:
            want = {r["record_id"] for r in csv.DictReader(f)}
        recs = [r for r in rows if r["record_id"] in want]
        for i in range(0, len(recs), BATCH_SIZE):
            write_jsonl(OUT / f"rescreen_{i // BATCH_SIZE + 1:03d}.jsonl",
                        recs[i:i + BATCH_SIZE])
        print(f"rescreen: {len(recs)} records with a recovered abstract")
        return

    ordered = sorted(rows, key=lambda r: (tier(r), r["record_id"]))
    n = 0
    for i in range(0, len(ordered), BATCH_SIZE):
        write_jsonl(OUT / f"batch_{i // BATCH_SIZE + 1:03d}.jsonl", ordered[i:i + BATCH_SIZE])
        n += 1
    counts = [sum(tier(r) == t for r in rows) for t in range(3)]
    print(f"{len(rows)} records in {n} batches of up to {BATCH_SIZE}; "
          f"tiers (both/either/neither): {counts}")


if __name__ == "__main__":
    main()
