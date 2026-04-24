"""Baseline predictors for comparison against the generative agent.

Three baselines, each predicting AcceptedUniversityHensachi from Big Five:

1. Random-from-empirical: ignores Big Five, draws hensachi from the
   empirical marginal distribution of raw_dataset.csv. Establishes the
   chance-level floor (expected Pearson r ~= 0).

2. Linear regression (full Big Five): OLS on all five traits. Represents
   the upper bound of what a simple supervised baseline can extract when
   trained on the same synthetic ground-truth data that the LLM is
   evaluated against. Reported as in-sample R^2.

3. Conscientiousness-only linear regression: single-predictor OLS on C.
   Tests whether the LLM's advantage over regression (if any) comes from
   multi-dimensional integration rather than picking up C alone.

Ground truth: simulation/data/raw_dataset.csv, joined with raw.csv to
recover full Big Five.

Outputs per baseline:
- Pearson r, RMSE, MAE against ground-truth hensachi
- Mean and SD of predicted distribution (for distribution-level comparison)

These baselines are computed deterministically from the ground-truth data
and do not require the agent to have been run.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

SEED = 42
BIG_FIVE = ["Openness", "Conscientiousness", "Extraversion", "Agreeableness", "Neuroticism"]


def load_dataset() -> pd.DataFrame:
    here = Path(__file__).resolve().parent
    data_dir = here.parent / "data"
    raw = pd.read_csv(data_dir / "raw.csv")
    raw = raw[raw["category"] == "all"].reset_index(drop=True)
    synth = pd.read_csv(data_dir / "raw_dataset.csv")
    merged = synth.merge(
        raw[["ID"] + [t for t in BIG_FIVE if t not in synth.columns]],
        on="ID",
        how="left",
        validate="1:1",
    )
    assert len(merged) == 103
    for t in BIG_FIVE:
        assert t in merged.columns, f"missing Big Five column: {t}"
    return merged


def metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
    r = float(np.corrcoef(y_true, y_pred)[0, 1]) if np.std(y_pred) > 0 else 0.0
    rmse = float(np.sqrt(np.mean((y_true - y_pred) ** 2)))
    mae = float(np.mean(np.abs(y_true - y_pred)))
    return {
        "pearson_r":  round(r, 4),
        "rmse":       round(rmse, 3),
        "mae":        round(mae, 3),
        "pred_mean":  round(float(np.mean(y_pred)), 3),
        "pred_sd":    round(float(np.std(y_pred, ddof=1)), 3),
    }


def baseline_random(df: pd.DataFrame) -> dict[str, object]:
    rng = np.random.default_rng(SEED)
    y_true = df["AcceptedUniversityHensachi"].values
    y_pred = rng.permutation(y_true)  # preserves marginal exactly
    return {"name": "random_from_empirical", **metrics(y_true, y_pred)}


def _ols_fit_predict(X: np.ndarray, y: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Closed-form OLS with intercept. Returns (coefficients, y_hat)."""
    X1 = np.column_stack([np.ones(len(X)), X])
    coef, *_ = np.linalg.lstsq(X1, y, rcond=None)
    y_hat = X1 @ coef
    return coef, y_hat


def baseline_bigfive_ols(df: pd.DataFrame) -> dict[str, object]:
    X = df[BIG_FIVE].values
    y = df["AcceptedUniversityHensachi"].values
    coef, y_hat = _ols_fit_predict(X, y)
    m = metrics(y, y_hat)
    ss_res = np.sum((y - y_hat) ** 2)
    ss_tot = np.sum((y - np.mean(y)) ** 2)
    r2 = 1 - ss_res / ss_tot
    return {
        "name": "ols_bigfive_full",
        **m,
        "r2_insample": round(float(r2), 4),
        "coefficients": {
            name: round(float(b), 4)
            for name, b in zip(["intercept"] + BIG_FIVE, coef)
        },
    }


def baseline_conscientiousness_only(df: pd.DataFrame) -> dict[str, object]:
    X = df[["Conscientiousness"]].values
    y = df["AcceptedUniversityHensachi"].values
    coef, y_hat = _ols_fit_predict(X, y)
    m = metrics(y, y_hat)
    ss_res = np.sum((y - y_hat) ** 2)
    ss_tot = np.sum((y - np.mean(y)) ** 2)
    r2 = 1 - ss_res / ss_tot
    return {
        "name": "ols_conscientiousness_only",
        **m,
        "r2_insample": round(float(r2), 4),
        "coefficients": {
            "intercept":        round(float(coef[0]), 4),
            "Conscientiousness": round(float(coef[1]), 4),
        },
    }


def main() -> None:
    df = load_dataset()
    results = [
        baseline_random(df),
        baseline_conscientiousness_only(df),
        baseline_bigfive_ols(df),
    ]
    out_path = Path(__file__).resolve().parent.parent / "data" / "baseline_results.json"
    out_path.write_text(json.dumps(results, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"Wrote {out_path}")
    print()
    print("Baseline performance (predicting AcceptedUniversityHensachi):")
    print(f"{'baseline':<32s} {'r':>8s} {'RMSE':>7s} {'MAE':>7s} {'pred_mean':>10s} {'pred_sd':>8s}")
    for r in results:
        print(f"{r['name']:<32s} {r['pearson_r']:>+8.4f} {r['rmse']:>7.3f} "
              f"{r['mae']:>7.3f} {r['pred_mean']:>10.3f} {r['pred_sd']:>8.3f}")
    print()
    print("OLS (Big Five) coefficients:")
    for k, v in results[-1]["coefficients"].items():
        print(f"  {k:<20s} {v:>+8.4f}")


if __name__ == "__main__":
    main()
