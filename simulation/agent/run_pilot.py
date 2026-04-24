"""Pilot run: first 10 participants x N=30 samples each (300 API calls).

Reads Big Five scores from ../data/raw.csv (category == 'all'), runs the
agent.run_one pipeline with Opus 4.7 + extended thinking + submit_prediction
tool, and saves results to ../data/agent_pilot_results.csv.

Claude Opus 4.7 pricing (2026-04):
  input:    $15 / MTok
  output:   $75 / MTok
  cache write:  $18.75 / MTok
  cache read:   $1.50  / MTok

Estimated pilot cost with prompt caching active: ~$30-40.
"""

from __future__ import annotations

import os
import sys
import time
from pathlib import Path

import pandas as pd
from anthropic import Anthropic

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

sys.path.insert(0, str(Path(__file__).resolve().parent))
from load_env import load_env  # noqa: E402
from agent import run_one  # noqa: E402

load_env()


N_PARTICIPANTS = 10
N_SAMPLES = 30

PRICING = {
    "input":       15.0 / 1_000_000,
    "output":      75.0 / 1_000_000,
    "cache_read":   1.5 / 1_000_000,
    "cache_write": 18.75 / 1_000_000,
}


def main() -> None:
    if not os.environ.get("ANTHROPIC_API_KEY"):
        raise SystemExit("ANTHROPIC_API_KEY is not set. Set it before running.")

    here = Path(__file__).resolve().parent
    data_dir = here.parent / "data"
    raw = pd.read_csv(data_dir / "raw.csv")
    raw = raw[raw["category"] == "all"].reset_index(drop=True).head(N_PARTICIPANTS)

    client = Anthropic()

    rows = []
    t_start = time.time()
    total_in = total_out = total_cache_r = total_cache_w = 0
    failures = 0

    for i, row in raw.iterrows():
        big_five = {
            "Openness":          float(row["Openness"]),
            "Conscientiousness": float(row["Conscientiousness"]),
            "Extraversion":      float(row["Extraversion"]),
            "Agreeableness":     float(row["Agreeableness"]),
            "Neuroticism":       float(row["Neuroticism"]),
        }
        for s in range(N_SAMPLES):
            try:
                out = run_one(client, big_five)
            except Exception as e:
                failures += 1
                print(f"  !! ID={row['ID']} sample={s}: {e}")
                continue
            p = out.parsed
            rows.append({
                "ID": row["ID"],
                "sample": s,
                **big_five,
                "predicted_tests_completed":  p.get("predicted_tests_completed"),
                "predicted_tests_mastered":   p.get("predicted_tests_mastered"),
                "predicted_hensachi":         p.get("predicted_hensachi"),
                "predicted_university_example": p.get("predicted_university_example"),
                "application_strategy":       p.get("application_strategy"),
                "confidence":                 p.get("confidence"),
                "backstory":                  p.get("backstory"),
                "learning_behavior":          p.get("learning_behavior"),
                "reasoning":                  p.get("reasoning"),
                "input_tokens":               out.input_tokens,
                "output_tokens":              out.output_tokens,
                "cache_read_tokens":          out.cache_read_tokens,
                "cache_write_tokens":         out.cache_write_tokens,
                "stop_reason":                out.stop_reason,
            })
            total_in += out.input_tokens
            total_out += out.output_tokens
            total_cache_r += out.cache_read_tokens
            total_cache_w += out.cache_write_tokens
            h = p.get("predicted_hensachi")
            univ = (p.get("predicted_university_example") or "")[:40]
            print(f"  [{i+1}/{N_PARTICIPANTS}] ID={row['ID']} s={s:02d}: "
                  f"hensachi={h} univ={univ}")

        # Periodic save so we don't lose progress mid-run.
        df_partial = pd.DataFrame(rows)
        df_partial.to_csv(data_dir / "agent_pilot_results.csv", index=False)

    elapsed = time.time() - t_start
    cost = (
        total_in       * PRICING["input"] +
        total_out      * PRICING["output"] +
        total_cache_r  * PRICING["cache_read"] +
        total_cache_w  * PRICING["cache_write"]
    )

    print()
    print("=" * 70)
    print(f"Rows saved: {len(rows)} (expected: {N_PARTICIPANTS * N_SAMPLES})")
    print(f"Failures:   {failures}")
    print(f"Elapsed:    {elapsed:.1f}s  ({elapsed/max(len(rows),1):.2f}s/call)")
    print(f"Tokens:     in={total_in}, out={total_out}, "
          f"cache_r={total_cache_r}, cache_w={total_cache_w}")
    print(f"Est. cost:  ${cost:.2f}")

    df = pd.DataFrame(rows)
    if not df.empty:
        print()
        print("Per-ID prediction summary:")
        g = df.groupby("ID")["predicted_hensachi"].agg(["mean", "std", "min", "max"]).round(2)
        print(g.to_string())


if __name__ == "__main__":
    main()
