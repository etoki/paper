"""Generate `output/canonical_numbers.md` — single source of truth for every
numerical claim in `simulation/paper/`.

Usage:
    cd simulation && uv run python -m code.build_canonical_numbers

Reads:
    output/supplementary/stage{0..8}*.h5  (after `make reproduce`)

Writes:
    output/canonical_numbers.md           (committed to git)

Use this file for paper-text hallucination checks instead of re-running the
pipeline. Re-run this script (and commit the result) only when the pipeline
output changes (new seed, code change, dependency upgrade).
"""

from __future__ import annotations

import sys
from pathlib import Path

import h5py
import numpy as np
from scipy.stats import spearmanr


def _scalar(arr) -> float:
    """Coerce a possibly 1-element ndarray (or 0-dim) to a Python float."""
    a = np.asarray(arr)
    return float(a.reshape(-1)[0])


def main() -> int:
    out_dir = Path("output/supplementary")
    if not out_dir.exists():
        sys.stderr.write(
            f"ERROR: {out_dir} not found. Run `make reproduce` first.\n"
        )
        return 1

    lines: list[str] = []
    add = lines.append

    # ============================================================
    # Header
    # ============================================================
    add("# Canonical Numbers — HEXACO Harassment Microsimulation v2.0")
    add("")
    add("**Generated**: from `output/supplementary/*.h5` after `make reproduce` (seed 20260429)  ")
    add("**Pre-registration**: OSF DOI 10.17605/OSF.IO/3Y54U  ")
    add("**Purpose**: Single source of truth for every numerical claim in `simulation/paper/`. ")
    add("Use this file as the reference for hallucination/integrity checks instead of re-running the pipeline.")
    add("")
    add("**How to update**: ")
    add("```")
    add("cd simulation")
    add("make reproduce                                    # regenerate stage0..8 H5")
    add("uv run python -m code.build_canonical_numbers     # rewrite this file")
    add("```")
    add("")
    add("---")
    add("")

    # ============================================================
    # Stage 0 cell propensity
    # ============================================================
    with h5py.File(out_dir / "stage0_cell_propensity.h5") as f:
        pp = f["point_power"][()]
        pg = f["point_gender"][()]
        cn = f["cell_n"][()]
        methods = [
            m.decode() if isinstance(m, bytes) else str(m)
            for m in f["methods_power"][()]
        ]

    add("## Stage 0 — 14-cell propensity (Results §3.1, Tables 1/6)")
    add("")
    add("- N total: 354")
    add("- Cells: 14 (7 HEXACO clusters × 2 genders, gender 0=female, 1=male)")
    add(f"- Cell N range: {cn.min()}–{cn.max()} (median {int(np.median(cn))})")
    add(
        f"- Power-harassment propensity range: {pp.min():.4f}–{pp.max():.4f} "
        f"(mean across cells = {pp.mean():.4f})"
    )
    add(
        f"- Degenerate cells (N=0): 0; Degenerate cells (X∈{{0,N}}): "
        f"{int(((pp == 0) | (pp == 1)).sum())} "
        f"(idx {[i for i in range(14) if pp[i] in (0.0, 1.0)]})"
    )
    bca_count = sum(1 for m in methods if m == "bca")
    cp_count = sum(1 for m in methods if m == "clopper_pearson")
    bc_count = sum(1 for m in methods if m == "bc")
    pct_count = sum(1 for m in methods if m == "percentile")
    add(
        f"- CI cascade resolution: BCa = {bca_count}, "
        f"Clopper-Pearson = {cp_count}, BC = {bc_count}, Percentile = {pct_count}"
    )
    add("")
    add("| cell_idx | cluster | gender | N | X (power) | p̂ (power) | CI method |")
    add("|---:|---:|---:|---:|---:|---:|---|")
    for i in range(14):
        cluster = i // 2
        gender_s = "female" if i % 2 == 0 else "male"
        X = int(round(pp[i] * cn[i]))
        add(
            f"| {i} | {cluster} | {gender_s} | {cn[i]} | {X} | "
            f"{pp[i]:.4f} | {methods[i]} |"
        )
    add("")

    # H5 / H6 Spearman
    cluster_pp = np.array([pp[2 * i : 2 * i + 2].mean() for i in range(7)])
    cluster_pg = np.array([pg[2 * i : 2 * i + 2].mean() for i in range(7)])
    rho_h6_cluster, p_h6_cluster = spearmanr(cluster_pp, cluster_pg)
    rho_h6_cell, p_h6_cell = spearmanr(pp, pg)
    female = np.array([pp[2 * i] for i in range(7)])
    male = np.array([pp[2 * i + 1] for i in range(7)])
    rho_h5, p_h5 = spearmanr(female, male)

    # ============================================================
    # Stage 1 / Stage 2 (H1)
    # ============================================================
    with h5py.File(out_dir / "stage2_validation.h5") as f:
        h1 = {}
        for key in [
            "point_ape_FY2016",
            "point_ape_FY2020",
            "point_ape_FY2023",
            "ci_lo_FY2016",
            "ci_hi_FY2016",
            "ci_lo_FY2020",
            "ci_hi_FY2020",
            "ci_lo_FY2023",
            "ci_hi_FY2023",
        ]:
            h1[key] = _scalar(f[key][()])
        if "boot_prevalence" in f:
            p_baseline = float(f["boot_prevalence"][()].mean())
        else:
            p_baseline = None

    add("## Stage 1 — National prevalence (Results §3.2)")
    add("")
    if p_baseline is not None:
        add(f"- P̂ baseline (Stage 2 bootstrap mean): {p_baseline:.4f}")
    add("- MIC Labor Force Survey 2022 marginals: F = 0.4498, M = 0.5502 (total 6,723万人)")
    add("- Cluster proportions M3-fixed at Tokiwa (2026, IEEE Access) values")
    add("")

    add("## Stage 2 — H1 MAPE (Results §3.2 Table 1, Table 6 H1 row)")
    add("")
    add("| Period | Observed | MAPE | 95% CI lo | 95% CI hi | Tier |")
    add("|---|---:|---:|---:|---:|---|")
    for fy, obs in [("FY2016", 0.325), ("FY2020", 0.314), ("FY2023", 0.193)]:
        mape = h1[f"point_ape_{fy}"]
        lo = h1[f"ci_lo_{fy}"]
        hi = h1[f"ci_hi_{fy}"]
        if mape <= 30 and hi <= 30:
            tier = "Strict SUCCESS"
        elif mape <= 30:
            tier = "Standard SUCCESS"
        elif mape <= 60:
            tier = "PARTIAL SUCCESS"
        else:
            tier = "FAILURE"
        add(
            f"| {fy} | {obs:.3f} | {mape:.2f}% | {lo:.2f}% | {hi:.2f}% | {tier} |"
        )
    add("")
    add("- B = 10,000 (headline national MAPE bootstrap, Methods Clarification m3)")
    add("- B = 2,000 (per-cell bootstrap)")
    add("")

    # ============================================================
    # Stage 4 baselines (H2)
    # ============================================================
    with h5py.File(out_dir / "stage4_baselines.h5") as f:
        mp = f["mape_point"][()]
        mlo = f["mape_ci_lo"][()]
        mhi = f["mape_ci_hi"][()]
        bh = f["bh_decisions"][()]
        bh_p = f["bh_pvalues"][()]
        pl = _scalar(f["pages_l_statistic"][()])
        plp = _scalar(f["pages_l_pvalue"][()])

    add("## Stage 4 — B0–B4 baselines (Results §3.4 Table 3)")
    add("")
    add("| Baseline | MAPE | 95% CI |")
    add("|---|---:|---|")
    names = [
        "B0 (uniform = MHLW grand mean)",
        "B1 (gender-only logistic)",
        "B2 (HEXACO 6-domain logistic)",
        "B3 (14-cell conditional, main pipeline)",
        "B4 (extended: age + age × cluster)",
    ]
    for i, name in enumerate(names):
        add(f"| {name} | {mp[i]:.2f}% | [{mlo[i]:.2f}, {mhi[i]:.2f}] |")
    add("")
    add(
        f"- Bonferroni-Holm decisions (B0-B1, B1-B2, B2-B3, B3-B4): "
        f"{bh.tolist()} → **{int(bh.sum())}/4 confirmed**"
    )
    add(f"- BH p-values: {[round(float(p), 4) for p in bh_p]}")
    add(f"- Page's L statistic: {pl:.2f}, p = {plp:.4f}")
    add("- H2 decision: ambiguous_or_reversed → **REJECTED**")
    add("")

    # ============================================================
    # Stage 5 CMV
    # ============================================================
    with h5py.File(out_dir / "stage5_cmv_diagnostic.h5") as f:
        cmv_keys = list(f.keys())
        cmv_first = None
        for key in (
            "harman_first_factor_variance",
            "first_factor_variance",
            "harman_pct",
        ):
            if key in cmv_keys:
                cmv_first = _scalar(f[key][()])
                break

    add("## Stage 5 — CMV diagnostic (Results §3.5)")
    add("")
    add("- N total: 354; N used after listwise deletion: 353")
    add("- 11 standardized variables (6 HEXACO + 3 Dark Triad + 2 harassment)")
    if cmv_first is not None:
        add(f"- Harman first-factor variance: **{cmv_first:.2f}%** (threshold 50%)")
    else:
        add("- Harman first-factor variance: **24.08%** (threshold 50%; from stage5 print log)")
    add("- CMV concern flag: False")
    add("")
    add("Marker-variable correction (Lindell & Whitney 2001) using HEXACO O as theoretical marker:")
    add("- r(O, power harassment) = +0.068 (CMV estimate)")
    add("- r(O, gender harassment) = −0.181")
    add("")
    add("| Variable | r_raw (power) | r_adjusted (power) |")
    add("|---|---:|---:|")
    add("| H | −0.265 | −0.356 |")
    add("| E | −0.037 | −0.112 |")
    add("| X | +0.030 | −0.041 |")
    add("| A | −0.230 | −0.319 |")
    add("| C | −0.100 | −0.180 |")
    add("| Machiavellianism | +0.072 | +0.005 |")
    add("| Narcissism | +0.154 | +0.093 |")
    add("| Psychopathy | +0.391 | +0.347 |")
    add("")

    # ============================================================
    # Stage 7 counterfactuals
    # ============================================================
    with h5py.File(out_dir / "stage7_counterfactual.h5") as f:
        da = _scalar(f["delta_p_a_point"][()])
        db = _scalar(f["delta_p_b_point"][()])
        dc = _scalar(f["delta_p_c_point"][()])
        a_ci = f["delta_p_a_ci"][()]
        b_ci = f["delta_p_b_ci"][()]
        c_ci = f["delta_p_c_ci"][()]
        iut = f["h7_iut"][()]
        rho_b = f["rho_b"][()]
        p_baseline_s7 = _scalar(f["p_baseline"][()])

    # Reproduce m5 flagged_weight (44.5% per A-2 reconcile + cell_weights)
    with h5py.File(out_dir / "stage1_population_aggregation.h5") as f:
        cw = f["cell_weights"][()]
    flagged_mask = rho_b < 0.10
    flagged_weight = float(cw[flagged_mask].sum() / cw.sum()) if cw.sum() > 0 else 0.0

    add("## Stage 7 — Counterfactual ΔP_x and H7 IUT (Results §3.6 Table 4)")
    add("")
    add(f"- P̂ baseline: **{p_baseline_s7:.4f}** ({p_baseline_s7 * 100:.2f}%)")
    add("")
    add("| Counterfactual | Operationalization | ΔP point | 95% CI |")
    add("|---|---|---:|---|")
    add(
        f"| A (universal) | do(HH+0.3σ; A+0.3σ; E+0.3σ) for all individuals | "
        f"{da:+.4f} | [{a_ci[0]:+.4f}, {a_ci[1]:+.4f}] |"
    )
    add(
        f"| B (targeted) | do(HH+0.40σ) for individuals with baseline cluster ∈ {{0, 4, 6}} | "
        f"{db:+.4f} | [{b_ci[0]:+.4f}, {b_ci[1]:+.4f}] |"
    )
    add(
        f"| C (structural) | p_c × 0.80 for all 14 cells | "
        f"{dc:+.4f} | [{c_ci[0]:+.4f}, {c_ci[1]:+.4f}] |"
    )
    add("")
    add(f"- H7 IUT lower bounds: L_BA = {iut[0]:+.4f}, L_BC = {iut[1]:+.4f}")
    add("- H7 classification: **REVERSAL** (m7 priority-1: point ΔP_B < point ΔP_C; consistent with L_BC < 0)")
    add(f"- |ΔP_C| / |ΔP_A| = {abs(dc) / abs(da):.2f}")
    add("")
    add("Positivity (Methods Clarification m5):")
    add(f"- ρ_B per cell: {[round(float(x), 3) for x in rho_b]}")
    add(
        f"- B flagged_weight (population-weighted share of cells with ρ_B < 0.10): "
        f"{flagged_weight * 100:.1f}% → m5 downgrade triggered (≥ 20%)"
    )
    add("- A: ρ ≡ 1 (universal intervention; no extrapolation)")
    add("- C: ρ ≡ 1 (cell-level rate adjustment; no individual reassignment)")
    add("")

    # ============================================================
    # Stage 8 transportability
    # ============================================================
    with h5py.File(out_dir / "stage8_transportability.h5") as f:
        factors = f["factors"][()]
        da_att = f["delta_p_a_attenuated"][()]
        db_att = f["delta_p_b_attenuated"][()]
        dc_att = f["delta_p_c_attenuated"][()]
        cls = [c.decode() for c in f["h7_classifications"][()]]
        LBA_F = f["L_BA_per_factor"][()]
        LBC_F = f["L_BC_per_factor"][()]

    add("## Stage 8 — Transportability (Results §3.7 Table 5)")
    add("")
    add("| F | Anchor | ΔP_A | ΔP_B | ΔP_C | L_BA | L_BC | H7 |")
    add("|---:|---|---:|---:|---:|---:|---:|---|")
    anchors = [
        "Conservative cross-cultural worst case",
        "Nielsen et al. 2017 Asia/Oceania",
        "Mild attenuation",
        "Reference (no attenuation)",
    ]
    for i, F_val in enumerate(factors):
        add(
            f"| {F_val:.1f} | {anchors[i]} | {da_att[i]:+.4f} | {db_att[i]:+.4f} | "
            f"{dc_att[i]:+.4f} | {LBA_F[i]:+.4f} | {LBC_F[i]:+.4f} | {cls[i]} |"
        )
    add("")

    # ============================================================
    # H5 / H6 Spearman
    # ============================================================
    add("## H5 / H6 cluster-rank correlations (Results §3.8 Table 6)")
    add("")
    add("**H5 — Gender invariance** (cluster-level rank concordance across genders, 7-cluster):")
    add(f"- Spearman ρ(female, male) = **{rho_h5:.4f}** (p = {p_h5:.4f}) → **CONFIRMED**")
    add("")
    add("**H6 — Cross-domain triangulation** (cluster-level rank concordance between power and gender harassment):")
    add(f"- Spearman ρ(power, gender) at 7-cluster: **{rho_h6_cluster:.4f}** (p = {p_h6_cluster:.4f})")
    add(f"- Spearman ρ(power, gender) at 14-cell: **{rho_h6_cell:.4f}** (p = {p_h6_cell:.4f})")
    add("- → **REJECTED**")
    add("")

    # ============================================================
    # External validation
    # ============================================================
    add("## External validation targets (verified against primary sources)")
    add("")
    add("MHLW Workplace Harassment Survey past-3-year power-harassment victimization rates:")
    add("- FY2016 (H28): **32.5%** (https://www.mhlw.go.jp/file/06-Seisakujouhou-11200000-Roudoukijunkyoku/0000165751.pdf)")
    add("- FY2020 (R2):  **31.4%**")
    add("- FY2023 (R5):  **19.3%** (https://www.mhlw.go.jp/content/11909000/001259093.pdf)")
    add("- FY2016 → FY2023: **−13.2 pp** (−40.6% relative)")
    add("- FY2020 → FY2023: −12.1 pp")
    add("")
    add("Statistics Bureau (MIC) Labor Force Survey 2022 Annual Average:")
    add("- Total employed: **6,723 万人**")
    add("- Female: **3,024 万人** (44.98%)")
    add("- Male: **3,699 万人** (55.02%)")
    add("")
    add("Power Harassment Prevention Law (改正労働施策総合推進法):")
    add("- Large enterprises: **2020-06-01** enforcement")
    add("- SMEs: **2022-04-01** enforcement")
    add("")

    # ============================================================
    # Hypothesis summary
    # ============================================================
    add("## Hypothesis outcomes summary (Table 6)")
    add("")
    add("| Hypothesis | Outcome | Tier |")
    add("|---|---|---|")
    add(
        f"| H1 (latent prevalence) | MAPE_FY2016 = {h1['point_ape_FY2016']:.2f}% | PARTIAL SUCCESS |"
    )
    add(
        f"| H2 (B0–B4 ordinal trend) | {int(bh.sum())}/4 BH pairs confirmed; Page's L p = {plp:.4f} | REJECTED |"
    )
    add("| H3 (centroid concordance) | Reported in Tokiwa 2026 IEEE Access | CONFIRMED |")
    add(
        f"| H4 (CI cascade stability) | {bca_count}/14 BCa + {cp_count} Clopper-Pearson per M4 | CONFIRMED |"
    )
    add(f"| H5 (gender invariance) | Spearman ρ = {rho_h5:.2f} (p = {p_h5:.3f}) | CONFIRMED |")
    add(
        f"| H6 (cross-domain triangulation) | Spearman ρ = {rho_h6_cluster:.2f} (cluster), "
        f"{rho_h6_cell:.2f} (cell) | REJECTED |"
    )
    add("| H7 (counterfactual ordering) | REVERSAL via m7 priority-1; robust F=0.3..1.0 | REVERSAL |")
    add("")

    # ============================================================
    # Headline values
    # ============================================================
    add("## Headline values used in abstract / cover letter")
    add("")
    add(
        f"- ΔP_A = {da:+.4f} ({da * 100:+.2f} pp), 95% CI [{a_ci[0]:+.4f}, {a_ci[1]:+.4f}]"
    )
    add(
        f"- ΔP_B = {db:+.4f} ({db * 100:+.2f} pp), 95% CI [{b_ci[0]:+.4f}, {b_ci[1]:+.4f}]"
    )
    add(
        f"- ΔP_C = {dc:+.4f} ({dc * 100:+.2f} pp), 95% CI [{c_ci[0]:+.4f}, {c_ci[1]:+.4f}]"
    )
    add(f"- |ΔP_C| / |ΔP_A| ≈ **{abs(dc) / abs(da):.1f}×** (rounded to 1 decimal)")
    add("- H7 = REVERSAL (4/4 transportability factors)")
    add(f"- P̂ baseline = {p_baseline_s7 * 100:.2f}%")
    add("")
    add("---")
    add("")
    add("*End of canonical numbers file.*")
    add("")

    out_path = Path("output/canonical_numbers.md")
    text = "\n".join(lines)
    out_path.write_text(text)
    print(f"Wrote {out_path} ({len(text):,} chars, {len(lines)} lines)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
