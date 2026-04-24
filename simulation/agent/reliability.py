"""Reliability analysis for the generative agent's per-participant predictions.

Given N samples per participant, computes:

1. ICC(1,1) and ICC(2,1) via one-way and two-way ANOVA decompositions.
   ICC values near 1 indicate high between-participant variance relative to
   within-participant (sampling) variance, i.e. the LLM produces stable
   person-level predictions.

2. ICC(1,k) / ICC(2,k): reliability of the ensemble mean over k samples.

3. Per-participant SD summary: distribution of within-participant
   prediction spread across the cohort.

Reference: Shrout & Fleiss (1979). Intraclass correlations: Uses in assessing
rater reliability. Psychological Bulletin, 86, 420-428.

Inputs : ../data/agent_pilot_results.csv (or agent_full_results.csv for 103 x N)
Outputs: ../data/reliability_results.json
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")


def icc_from_df(
    df: pd.DataFrame, *, subject_col: str = "ID", score_col: str = "predicted_hensachi"
) -> dict[str, float]:
    """One-way random-effects ICC(1,1) and ICC(1,k) via MS decomposition.

    For each subject (participant), we have k ratings (samples). Because
    "rater" identity (sample index) is not a fixed factor of interest --
    samples are interchangeable draws from the agent -- ICC(1) is the
    appropriate formulation (Shrout & Fleiss, 1979).

    Formulas (one-way random-effects):
      MSB = n * Var(subject_means)                 # between-subject MS
      MSW = mean( Var(within each subject) )       # within-subject MS
      ICC(1,1) = (MSB - MSW) / (MSB + (k-1) * MSW)
      ICC(1,k) = (MSB - MSW) / MSB                 # Spearman-Brown corrected
    where k is the number of samples per subject.
    """
    g = df.groupby(subject_col)[score_col]
    k_per_subject = g.count().values
    if not np.all(k_per_subject == k_per_subject[0]):
        raise ValueError(f"Unequal samples per subject: {np.unique(k_per_subject)}")
    k = int(k_per_subject[0])

    subject_means = g.mean().values
    subject_vars  = g.var(ddof=1).values
    n = len(subject_means)

    grand_mean = df[score_col].mean()

    # MSB: between-subject mean square. Var of subject means times k.
    msb = k * np.sum((subject_means - grand_mean) ** 2) / (n - 1)
    # MSW: within-subject mean square. Average of subject variances.
    msw = np.mean(subject_vars)

    icc11 = (msb - msw) / (msb + (k - 1) * msw) if (msb + (k - 1) * msw) > 0 else 0.0
    icc1k = (msb - msw) / msb if msb > 0 else 0.0

    return {
        "n_subjects":  int(n),
        "k_samples":   int(k),
        "grand_mean":  round(float(grand_mean), 4),
        "msb":         round(float(msb), 4),
        "msw":         round(float(msw), 4),
        "icc_1_1":     round(float(icc11), 4),
        "icc_1_k":     round(float(icc1k), 4),
    }


def per_subject_spread(
    df: pd.DataFrame, *, subject_col: str = "ID", score_col: str = "predicted_hensachi"
) -> dict[str, object]:
    g = df.groupby(subject_col)[score_col]
    sds = g.std(ddof=1)
    ranges = g.max() - g.min()
    return {
        "sd_mean":    round(float(sds.mean()), 3),
        "sd_median":  round(float(sds.median()), 3),
        "sd_max":     round(float(sds.max()), 3),
        "range_mean": round(float(ranges.mean()), 3),
        "range_max":  round(float(ranges.max()), 3),
    }


def main() -> None:
    here = Path(__file__).resolve().parent
    data_dir = here.parent / "data"
    candidates = [
        data_dir / "agent_full_results.csv",
        data_dir / "agent_pilot_results.csv",
    ]
    src = next((p for p in candidates if p.exists()), None)
    if src is None:
        raise SystemExit(
            "No agent results found. Run run_pilot.py (or the full run) first."
        )
    df = pd.read_csv(src)
    print(f"Loaded {src.name}: {len(df)} rows, {df['ID'].nunique()} unique IDs")

    icc = icc_from_df(df)
    spread = per_subject_spread(df)

    results = {"source": src.name, "icc": icc, "spread": spread}
    out_path = data_dir / "reliability_results.json"
    out_path.write_text(json.dumps(results, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"Wrote {out_path}")
    print()
    print("One-way random-effects ICC on predicted_hensachi:")
    for k, v in icc.items():
        print(f"  {k:<14s} {v}")
    print()
    print("Within-participant prediction spread (hensachi units):")
    for k, v in spread.items():
        print(f"  {k:<14s} {v}")


if __name__ == "__main__":
    main()
