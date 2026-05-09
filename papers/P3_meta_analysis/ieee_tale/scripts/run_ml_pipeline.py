"""IEEE TALE 2026 — Study-level ML predictive layer over the meta-analytic
corpus, with SHAP-based interpretation and a fairlearn-based fairness audit.

This is the *skeleton* implementation. With study-level N around 12 in
the primary pool, it is reported in the paper as an interpretability
case study, not a deployable model.

The script is designed to degrade gracefully:
- if `xgboost`, `shap`, or `fairlearn` are not installed, the corresponding
  section is skipped with a printed warning;
- if the input studies.csv has too few rows for LOSO-CV, the script writes
  a `results/insufficient_data.md` and exits 0.

Usage:
    python papers/P3_meta_analysis/ieee_tale/scripts/run_ml_pipeline.py

Outputs (to papers/P3_meta_analysis/ieee_tale/results/):
    feature_matrix.csv       the engineered X / y / sensitive matrix
    stem_subset_pools.csv    per-trait pooled r within the STEM substratum
    ml_loso_metrics.csv      LOSO-CV metrics per model
    shap_ranking.csv         mean(|SHAP|) per feature
    fairness_metrics.csv     demographic-parity / equalised-odds gaps
    summary.md               human-readable summary
"""
from __future__ import annotations

import csv
import math
import sys
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(REPO_ROOT / "metaanalysis" / "analysis"))
import pool as parent  # noqa: E402

INPUT_CSV = REPO_ROOT / "papers" / "P3_meta_analysis" / "inputs" / "studies.csv"
RESULTS_DIR = REPO_ROOT / "papers" / "P3_meta_analysis" / "ieee_tale" / "results"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

TRAITS = ("O", "C", "E", "A", "N")
EFFECT_THRESHOLD = 0.10  # Cohen-style small-effect threshold for the binary label
SEED = 20260509


# ---------------------------------------------------------------------------
# Optional dependencies (degrade gracefully)
# ---------------------------------------------------------------------------
def _maybe_import():
    mods = {}
    try:
        from sklearn.linear_model import LogisticRegression  # noqa
        from sklearn.ensemble import RandomForestClassifier  # noqa
        from sklearn.metrics import (
            roc_auc_score, balanced_accuracy_score, f1_score, brier_score_loss,
        )  # noqa
        from sklearn.preprocessing import StandardScaler  # noqa
        from sklearn.model_selection import LeaveOneOut  # noqa
        mods["sklearn"] = True
    except ImportError:
        mods["sklearn"] = False
    try:
        import xgboost  # noqa
        mods["xgboost"] = True
    except ImportError:
        mods["xgboost"] = False
    try:
        import shap  # noqa
        mods["shap"] = True
    except ImportError:
        mods["shap"] = False
    try:
        import fairlearn  # noqa
        from fairlearn.metrics import (
            demographic_parity_difference, equalized_odds_difference,
        )  # noqa
        mods["fairlearn"] = True
    except ImportError:
        mods["fairlearn"] = False
    return mods


# ---------------------------------------------------------------------------
# Data loading + feature engineering
# ---------------------------------------------------------------------------
def load_studies():
    with INPUT_CSV.open("r", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def primary_pool(rows):
    keep = ("include", "include_with_caveat", "include_COI")
    return [
        r for r in rows
        if r.get("inclusion_status") in keep
        and r.get("primary_achievement") in ("yes", "partial")
    ]


def build_features(rows):
    """Build the (X, y, sensitive, labels) matrices for the ML stage."""
    X_rows = []
    y = []
    region_sens = []
    era_sens = []
    labels = []
    feat_names = None

    modality_levels = ("A", "M", "S", "U")
    era_levels = ("pre-COVID", "COVID", "post-COVID", "mixed")
    region_levels = ("Asia", "Europe", "North_America", "Other")

    for r in rows:
        rs = []
        for trait in TRAITS:
            rs_str = (r.get(f"r_{trait}") or "").strip()
            if rs_str:
                try:
                    rs.append(abs(float(rs_str)))
                except ValueError:
                    pass
        if not rs:
            continue
        max_abs_r = max(rs)
        label = 1 if max_abs_r >= EFFECT_THRESHOLD else 0

        modality = (r.get("modality") or "U") or "U"
        era = (r.get("era") or "").strip()
        region = (r.get("region") or "").strip()

        try:
            log_n = math.log(max(int(float((r.get("N") or "1"))), 1))
        except ValueError:
            log_n = 0.0

        x_dict = {}
        for m in modality_levels:
            x_dict[f"mod_{m}"] = 1.0 if modality == m else 0.0
        for e in era_levels:
            x_dict[f"era_{e}"] = 1.0 if era == e else 0.0
        for rg in region_levels:
            x_dict[f"region_{rg}"] = 1.0 if region == rg else 0.0
        x_dict["log_N"] = log_n

        if feat_names is None:
            feat_names = list(x_dict.keys())
        X_rows.append([x_dict[k] for k in feat_names])
        y.append(label)
        region_sens.append(0 if region == "Asia" else 1)  # 0=Asia, 1=non-Asia
        era_sens.append(1 if era in ("COVID", "post-COVID") else 0)  # 1=COVID-and-after
        labels.append(r.get("study_id", ""))

    return (np.array(X_rows), np.array(y),
            np.array(region_sens), np.array(era_sens),
            labels, feat_names or [])


def write_feature_matrix(X, y, region_sens, era_sens, labels, feat_names):
    path = RESULTS_DIR / "feature_matrix.csv"
    with path.open("w", encoding="utf-8", newline="") as f:
        w = csv.writer(f)
        w.writerow(["study_id"] + feat_names + ["y_label", "sens_region_nonAsia", "sens_era_COVID_or_after"])
        for i, sid in enumerate(labels):
            w.writerow([sid] + list(X[i]) + [int(y[i]), int(region_sens[i]), int(era_sens[i])])


# ---------------------------------------------------------------------------
# STEM subset replication (uses the parent pooling primitives)
# ---------------------------------------------------------------------------
def stem_subset_pools(rows):
    """Per-trait pooled r restricted to discipline=STEM."""
    out = {}
    for trait in TRAITS:
        ys, vs, Ns = [], [], []
        for r in rows:
            if (r.get("discipline") or "") != "STEM":
                continue
            rs_str = (r.get(f"r_{trait}") or "").strip()
            n_str = (r.get("N") or "").strip()
            if not rs_str or not n_str:
                continue
            try:
                rho = float(rs_str)
                n = int(float(n_str))
            except ValueError:
                continue
            try:
                ys.append(parent.fisher_z(rho))
                vs.append(parent.var_z(n))
                Ns.append(n)
            except (ValueError, ZeroDivisionError):
                continue
        if len(ys) >= 2:
            res = parent.pool_random_effects(ys, vs)
            res["N_total"] = sum(Ns)
            out[trait] = res
        else:
            out[trait] = {"k": len(ys), "note": "k<2"}
    return out


def write_stem_pools(pools):
    path = RESULTS_DIR / "stem_subset_pools.csv"
    with path.open("w", encoding="utf-8", newline="") as f:
        w = csv.writer(f)
        w.writerow(["trait", "k", "N_total", "r_pooled", "r_ci_lo", "r_ci_hi", "I2", "tau2", "note"])
        for t in TRAITS:
            r = pools.get(t, {})
            if "r_pooled" in r:
                w.writerow([t, r["k"], r.get("N_total", ""),
                            f"{r['r_pooled']:.4f}",
                            f"{r['r_ci_lo']:.4f}", f"{r['r_ci_hi']:.4f}",
                            f"{r['I2']:.1f}", f"{r['tau2']:.4f}", ""])
            else:
                w.writerow([t, r.get("k", 0), "", "", "", "", "", "", r.get("note", "")])


# ---------------------------------------------------------------------------
# ML pipeline (sklearn + optional xgboost)
# ---------------------------------------------------------------------------
def run_ml(X, y, feat_names, mods):
    if not mods["sklearn"]:
        return None
    from sklearn.linear_model import LogisticRegression
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.preprocessing import StandardScaler
    from sklearn.model_selection import LeaveOneOut
    from sklearn.metrics import (
        roc_auc_score, balanced_accuracy_score, f1_score, brier_score_loss,
    )
    from sklearn.exceptions import NotFittedError  # noqa

    n = len(y)
    if n < 6:
        return {"note": f"only {n} usable studies; LOSO-CV not informative"}

    # Standardise log_N feature for LR; tree models do not need it.
    log_n_idx = feat_names.index("log_N") if "log_N" in feat_names else None

    def fit_predict(model_factory, X_train, y_train, X_test, scale=False):
        if scale and log_n_idx is not None:
            sc = StandardScaler()
            X_train = X_train.copy()
            X_test = X_test.copy()
            X_train[:, log_n_idx:log_n_idx + 1] = sc.fit_transform(X_train[:, log_n_idx:log_n_idx + 1])
            X_test[:, log_n_idx:log_n_idx + 1] = sc.transform(X_test[:, log_n_idx:log_n_idx + 1])
        m = model_factory()
        m.fit(X_train, y_train)
        if hasattr(m, "predict_proba"):
            proba = m.predict_proba(X_test)[:, 1]
        else:
            proba = m.decision_function(X_test)
        pred = (proba >= 0.5).astype(int)
        return proba, pred, m

    factories = {
        "lr": (lambda: LogisticRegression(C=1.0, max_iter=1000,
                                           class_weight="balanced",
                                           random_state=SEED), True),
        "rf": (lambda: RandomForestClassifier(n_estimators=200,
                                                class_weight="balanced",
                                                random_state=SEED), False),
    }
    if mods["xgboost"]:
        import xgboost as xgb
        factories["xgb"] = (
            lambda: xgb.XGBClassifier(
                max_depth=3, eta=0.1, n_estimators=100,
                use_label_encoder=False, eval_metric="logloss",
                random_state=SEED, scale_pos_weight=max(
                    1.0,
                    float((y == 0).sum()) / max(1.0, float((y == 1).sum())),
                ),
            ),
            False,
        )

    metrics = {name: {"y_true": [], "y_proba": [], "y_pred": []}
               for name in factories}
    loo = LeaveOneOut()
    last_models = {}
    for train_idx, test_idx in loo.split(X):
        X_train, X_test = X[train_idx], X[test_idx]
        y_train, y_test = y[train_idx], y[test_idx]
        if y_train.sum() == 0 or y_train.sum() == len(y_train):
            # degenerate fold; skip cleanly
            continue
        for name, (factory, scale) in factories.items():
            proba, pred, model = fit_predict(factory, X_train, y_train, X_test, scale=scale)
            metrics[name]["y_true"].extend(y_test.tolist())
            metrics[name]["y_proba"].extend(proba.tolist())
            metrics[name]["y_pred"].extend(pred.tolist())
            last_models[name] = model

    out = {}
    for name, d in metrics.items():
        if not d["y_true"]:
            out[name] = {"note": "no usable folds"}
            continue
        yt = np.array(d["y_true"])
        yp = np.array(d["y_proba"])
        yh = np.array(d["y_pred"])
        try:
            auroc = roc_auc_score(yt, yp) if len(set(yt)) > 1 else float("nan")
        except ValueError:
            auroc = float("nan")
        bal = balanced_accuracy_score(yt, yh) if len(set(yt)) > 1 else float("nan")
        f1 = f1_score(yt, yh, zero_division=0) if len(set(yt)) > 1 else float("nan")
        brier = brier_score_loss(yt, yp) if len(set(yt)) > 1 else float("nan")
        out[name] = {"auroc": auroc, "balanced_acc": bal, "f1": f1, "brier": brier}

    return {"metrics": out, "last_models": last_models, "X": X, "y": y, "feat_names": feat_names}


def write_ml_metrics(ml_res):
    path = RESULTS_DIR / "ml_loso_metrics.csv"
    with path.open("w", encoding="utf-8", newline="") as f:
        w = csv.writer(f)
        w.writerow(["model", "auroc", "balanced_acc", "f1", "brier", "note"])
        if ml_res is None:
            w.writerow(["", "", "", "", "", "sklearn unavailable"])
            return
        if "note" in ml_res:
            w.writerow(["", "", "", "", "", ml_res["note"]])
            return
        for name, m in ml_res["metrics"].items():
            if "auroc" in m:
                w.writerow([name,
                            f"{m['auroc']:.3f}" if not np.isnan(m['auroc']) else "",
                            f"{m['balanced_acc']:.3f}" if not np.isnan(m['balanced_acc']) else "",
                            f"{m['f1']:.3f}" if not np.isnan(m['f1']) else "",
                            f"{m['brier']:.3f}" if not np.isnan(m['brier']) else "",
                            ""])
            else:
                w.writerow([name, "", "", "", "", m.get("note", "")])


# ---------------------------------------------------------------------------
# SHAP interpretation
# ---------------------------------------------------------------------------
def run_shap(ml_res, mods):
    if ml_res is None or "last_models" not in ml_res:
        return None
    if not mods["shap"]:
        return None
    import shap
    X = ml_res["X"]
    feat_names = ml_res["feat_names"]
    rankings = {}
    for name, model in ml_res["last_models"].items():
        try:
            explainer = shap.Explainer(model.predict, X) if name == "lr" \
                else shap.TreeExplainer(model)
            sv = explainer(X) if name == "lr" else explainer.shap_values(X)
            arr = sv.values if hasattr(sv, "values") else np.array(sv)
            mean_abs = np.mean(np.abs(arr), axis=0)
            if mean_abs.ndim > 1:  # multi-class output
                mean_abs = mean_abs.mean(axis=-1)
            ranking = sorted(zip(feat_names, mean_abs.tolist()),
                              key=lambda kv: kv[1], reverse=True)
            rankings[name] = ranking
        except Exception as e:
            rankings[name] = [("ERROR", str(e))]
    return rankings


def write_shap_ranking(rankings):
    path = RESULTS_DIR / "shap_ranking.csv"
    with path.open("w", encoding="utf-8", newline="") as f:
        w = csv.writer(f)
        w.writerow(["model", "feature", "mean_abs_shap"])
        if not rankings:
            return
        for model_name, feats in rankings.items():
            for fname, val in feats:
                if isinstance(val, str):
                    w.writerow([model_name, fname, val])
                else:
                    w.writerow([model_name, fname, f"{val:.4f}"])


# ---------------------------------------------------------------------------
# Fairness audit (fairlearn)
# ---------------------------------------------------------------------------
def run_fairness(ml_res, region_sens, era_sens, mods):
    if ml_res is None or "metrics" not in ml_res:
        return None
    if not mods["fairlearn"]:
        return None
    from fairlearn.metrics import (
        demographic_parity_difference, equalized_odds_difference,
    )

    out = {}
    last_models = ml_res["last_models"]
    X = ml_res["X"]
    y = ml_res["y"]
    for name, model in last_models.items():
        try:
            yhat = model.predict(X)
        except Exception:
            continue
        out[name] = {}
        for sens_name, sens in (("region_nonAsia", region_sens),
                                  ("era_COVID_or_after", era_sens)):
            try:
                dp = demographic_parity_difference(y, yhat, sensitive_features=sens)
                eo = equalized_odds_difference(y, yhat, sensitive_features=sens)
            except Exception as e:
                dp = float("nan")
                eo = float("nan")
            out[name][sens_name] = {"dp_diff": float(dp), "eo_diff": float(eo)}
    return out


def write_fairness(fair_res):
    path = RESULTS_DIR / "fairness_metrics.csv"
    with path.open("w", encoding="utf-8", newline="") as f:
        w = csv.writer(f)
        w.writerow(["model", "sensitive_attr", "demographic_parity_diff", "equalized_odds_diff"])
        if not fair_res:
            return
        for model_name, by_sens in fair_res.items():
            for sens_name, d in by_sens.items():
                w.writerow([model_name, sens_name,
                            "" if math.isnan(d["dp_diff"]) else f"{d['dp_diff']:.3f}",
                            "" if math.isnan(d["eo_diff"]) else f"{d['eo_diff']:.3f}"])


# ---------------------------------------------------------------------------
# Summary
# ---------------------------------------------------------------------------
def write_summary(stem_pools, ml_res, shap_rankings, fair_res, mods, n_studies):
    lines = [
        "# IEEE TALE 2026 — ML Pipeline Summary",
        "",
        f"**Generated by**: `papers/P3_meta_analysis/ieee_tale/scripts/run_ml_pipeline.py`",
        f"**Studies in the ML stage**: {n_studies}",
        "",
        "## Optional dependencies",
        "",
        "| Package | Available |",
        "|---------|----------|",
        f"| sklearn   | {mods['sklearn']} |",
        f"| xgboost   | {mods['xgboost']} |",
        f"| shap      | {mods['shap']} |",
        f"| fairlearn | {mods['fairlearn']} |",
        "",
        "## STEM-subset pooled correlations",
        "",
        "| Trait | k | r [95% CI] |",
        "|-------|---|-----------|",
    ]
    for t in TRAITS:
        r = stem_pools.get(t, {})
        if "r_pooled" in r:
            lines.append(f"| {t} | {r['k']} | {r['r_pooled']:.3f} [{r['r_ci_lo']:.3f}, {r['r_ci_hi']:.3f}] |")
        else:
            lines.append(f"| {t} | {r.get('k', 0)} | — (k<2) |")
    lines.append("")

    lines.append("## ML LOSO-CV metrics")
    lines.append("")
    if ml_res is None:
        lines.append("- sklearn not available; ML stage skipped.")
    elif "note" in ml_res:
        lines.append(f"- {ml_res['note']}")
    else:
        lines.append("| Model | AUROC | Balanced Acc | F1 | Brier |")
        lines.append("|-------|------:|-------------:|---:|------:|")
        for name, m in ml_res["metrics"].items():
            if "auroc" in m:
                lines.append(
                    f"| {name} | "
                    f"{m['auroc']:.3f} | {m['balanced_acc']:.3f} | "
                    f"{m['f1']:.3f} | {m['brier']:.3f} |"
                )
            else:
                lines.append(f"| {name} | — | — | — | — |")
    lines.append("")

    lines.append("## SHAP (top features per model)")
    lines.append("")
    if not shap_rankings:
        lines.append("- shap not available or no models fit; section skipped.")
    else:
        for name, ranks in shap_rankings.items():
            lines.append(f"### {name}")
            for f, v in ranks[:5]:
                lines.append(f"- {f}: {v if isinstance(v, str) else f'{v:.4f}'}")
            lines.append("")

    lines.append("## Fairness audit")
    lines.append("")
    if not fair_res:
        lines.append("- fairlearn not available or no models fit; section skipped.")
    else:
        lines.append("| Model | Attribute | DP diff | EO diff |")
        lines.append("|-------|-----------|--------:|--------:|")
        for model_name, by_sens in fair_res.items():
            for sens_name, d in by_sens.items():
                dp = d["dp_diff"]
                eo = d["eo_diff"]
                dp_str = "—" if math.isnan(dp) else f"{dp:.3f}"
                eo_str = "—" if math.isnan(eo) else f"{eo:.3f}"
                lines.append(
                    f"| {model_name} | {sens_name} | {dp_str} | {eo_str} |"
                )
    lines.append("")

    lines.append("## Caveats")
    lines.append("")
    lines.append("- Study-level N is small (about 12 in the primary pool). All performance numbers are reported as proof-of-concept.")
    lines.append("- LOSO-CV is the most defensible split given the data structure but has high variance.")
    lines.append("- Fairness disparities computed on N about 6 per stratum are unstable; treat as descriptive.")
    lines.append("- The primary scientific value of this stage is *interpretability* (SHAP rankings), not prediction.")

    (RESULTS_DIR / "summary.md").write_text("\n".join(lines), encoding="utf-8")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    rows = load_studies()
    primary = primary_pool(rows)
    print(f"[TALE] {len(primary)} primary-pool studies")

    X, y, region_sens, era_sens, labels, feat_names = build_features(primary)
    print(f"[TALE] feature matrix: X.shape={X.shape}, y mean={y.mean() if len(y) else float('nan'):.2f}")

    write_feature_matrix(X, y, region_sens, era_sens, labels, feat_names)

    stem_pools = stem_subset_pools(primary)
    write_stem_pools(stem_pools)

    mods = _maybe_import()
    print(f"[TALE] optional packages: {mods}")

    ml_res = run_ml(X, y, feat_names, mods) if X.shape[0] >= 6 else \
        {"note": f"only {X.shape[0]} usable rows"}
    write_ml_metrics(ml_res)

    shap_rankings = run_shap(ml_res, mods) if ml_res and "last_models" in ml_res else None
    if shap_rankings:
        write_shap_ranking(shap_rankings)

    fair_res = run_fairness(ml_res, region_sens, era_sens, mods) \
        if ml_res and "last_models" in ml_res else None
    if fair_res:
        write_fairness(fair_res)

    write_summary(stem_pools, ml_res, shap_rankings, fair_res, mods, len(labels))
    print(f"[TALE] wrote {RESULTS_DIR}/summary.md")


if __name__ == "__main__":
    main()
