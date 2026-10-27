"""Check Claude's proposed decisions and build the reviewer's sheet.

    python metaanalysis/sr_update/screening/merge_suggestions.py --pilot
    python metaanalysis/sr_update/screening/merge_suggestions.py

Reads suggestions/<name>.jsonl (written by the Claude runs from the
matching batches/<name>.jsonl) and checks that every input record has
exactly one proposal with an allowed value and reason code.

--pilot also reports how the v4 studies in the pilot set were classed.

The full run writes screening_ta.csv (local only: it holds abstracts) for
the reviewer, with Claude's proposal next to empty columns for ET's
decision and reason. Records proposed as include or maybe come first.
suggestions/*.jsonl hold no abstracts and are committed.
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
from collections import Counter
from pathlib import Path

HERE = Path(__file__).resolve().parent
SR = HERE.parent
ALLOWED = {"include", "maybe", "exclude"}
CODES = {f"E{i}" for i in range(1, 8)}


def read_jsonl(path):
    with path.open(encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def check(batch_name):
    inp = read_jsonl(HERE / "batches" / f"{batch_name}.jsonl")
    out_path = HERE / "suggestions" / f"{batch_name}.jsonl"
    if not out_path.exists():
        return inp, {}, [f"{batch_name}: no suggestions file"]
    out = read_jsonl(out_path)
    problems = []
    by_id = {}
    for o in out:
        rid = o.get("record_id")
        if rid in by_id:
            problems.append(f"{batch_name}: {rid} proposed twice")
        if o.get("suggestion") not in ALLOWED:
            problems.append(f"{batch_name}: {rid} bad suggestion {o.get('suggestion')!r}")
        if o.get("suggestion") == "exclude" and o.get("reason") not in CODES:
            problems.append(f"{batch_name}: {rid} exclude without a valid reason")
        if o.get("suggestion") != "exclude" and o.get("reason"):
            problems.append(f"{batch_name}: {rid} reason given for {o.get('suggestion')}")
        by_id[rid] = o
    want = [r["record_id"] for r in inp]
    missing = [r for r in want if r not in by_id]
    extra = [r for r in by_id if r not in set(want)]
    if missing:
        problems.append(f"{batch_name}: {len(missing)} records without a proposal: {missing[:5]}")
    if extra:
        problems.append(f"{batch_name}: {len(extra)} proposals for unknown records: {extra[:5]}")
    return inp, by_id, problems


def pilot():
    inp, by_id, problems = check("pilot")
    for p in problems:
        print("PROBLEM", p)
    known = json.loads((HERE / "batches" / "pilot_known.json").read_text(encoding="utf-8"))
    print("Proposals:", dict(Counter(o["suggestion"] for o in by_id.values())))
    print("\nv4 studies (all reached the full-text stage in v4):")
    missed = 0
    for sid, rid in sorted(known.items()):
        o = by_id.get(rid, {})
        flag = "" if o.get("suggestion") in ("include", "maybe") else "  <-- would be excluded"
        missed += bool(flag)
        print(f"  {sid} {rid} {o.get('suggestion', '-'):8} {o.get('reason', ''):3} {o.get('note', '')[:70]}{flag}")
    print(f"\nknown studies kept (include or maybe): {len(known) - missed} of {len(known)}")


def full():
    batches = sorted(p.stem for p in (HERE / "batches").glob("batch_*.jsonl"))
    rows, problems = [], []
    for b in batches:
        inp, by_id, pr = check(b)
        problems += pr
        for r in inp:
            o = by_id.get(r["record_id"], {})
            rows.append({**r, "claude_suggestion": o.get("suggestion", ""),
                         "claude_reason": o.get("reason", ""),
                         "claude_note": o.get("note", ""), "batch": b,
                         "et_decision": "", "et_reason": "", "et_note": ""})
    for p in problems:
        print("PROBLEM", p)
    if problems:
        sys.exit("Fix the problems above before building the sheet.")
    order = {"include": 0, "maybe": 1, "exclude": 2}
    rows.sort(key=lambda r: (order[r["claude_suggestion"]], r["record_id"]))
    fields = ["record_id", "claude_suggestion", "claude_reason", "claude_note",
              "et_decision", "et_reason", "et_note", "title", "year", "journal",
              "abstract", "batch"]
    with (HERE / "screening_ta.csv").open("w", encoding="utf-8-sig", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
        w.writeheader()
        w.writerows(rows)
    print(f"{len(rows)} records; proposals: {dict(Counter(r['claude_suggestion'] for r in rows))}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--pilot", action="store_true")
    a = ap.parse_args()
    pilot() if a.pilot else full()
