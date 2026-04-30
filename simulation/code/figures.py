"""Figure 1-6 generation for the HEXACO Workplace Harassment Microsim manuscript.

Produces the six manuscript figures specified in `simulation/paper/06_tables_figures.md`:

  Figure 1: Pipeline schematic (box-and-arrow diagram, 9 stages)
  Figure 2: 14-cell propensity heatmap (cell × propensity, with cell sizes)
  Figure 3: National prevalence vs MHLW validation targets (FY2016/2020/2023)
  Figure 4: OAT sensitivity tornado plot (Stage 3 sweep)
  Figure 5: Counterfactual ΔP_x forest plot (Stage 7)
  Figure 6: Transportability robustness (Stage 8 cultural attenuation factors)

Each figure is saved at 300 DPI PNG + PDF + SVG to:
  output/figures/figure{N}_*.{png,pdf,svg}

Reads from:
  output/supplementary/stage0_cell_propensity.h5
  output/supplementary/stage1_population_aggregation.h5
  output/supplementary/stage2_validation.h5
  output/supplementary/stage3_sensitivity.h5
  output/supplementary/stage7_counterfactual.h5
  output/supplementary/stage8_transportability.h5

Random seed: 20260429.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np

from .utils_io import MHLW_VALIDATION_TARGETS, load_artifacts

# ============================================================================
# Configuration
# ============================================================================

DPI = 300
FIG_FORMATS = ("png", "pdf", "svg")
FONT_FAMILY = "DejaVu Serif"  # Times-like; widely available across platforms

# Default output directory (relative to working directory)
DEFAULT_FIGURES_DIR = Path("output/figures")


def _save_all_formats(fig, output_dir: Path, basename: str) -> list[Path]:
    """Save fig in all configured formats; return list of written paths."""
    output_dir.mkdir(parents=True, exist_ok=True)
    written = []
    for fmt in FIG_FORMATS:
        out = output_dir / f"{basename}.{fmt}"
        fig.savefig(out, format=fmt, dpi=DPI, bbox_inches="tight")
        written.append(out)
    plt.close(fig)
    return written


def _set_font(fig):
    for ax in fig.axes:
        for label in (ax.get_xticklabels() + ax.get_yticklabels()):
            label.set_fontfamily(FONT_FAMILY)
        ax.title.set_fontfamily(FONT_FAMILY)
        ax.xaxis.label.set_fontfamily(FONT_FAMILY)
        ax.yaxis.label.set_fontfamily(FONT_FAMILY)


# ============================================================================
# Figure 1: Pipeline schematic
# ============================================================================


def figure1_pipeline_schematic(output_dir: Path) -> list[Path]:
    """9-stage pipeline as box-and-arrow diagram."""
    fig, ax = plt.subplots(figsize=(11, 6.5), dpi=DPI)

    stages = [
        ("Stage 0\nType assignment\n+ Cell propensity\n+ EB shrinkage", 0, 2),
        ("Stage 1\nPopulation\naggregation\n(MHLW reweight)", 1, 2),
        ("Stage 2\nH1 4-tier\nclassification", 2, 2),
        ("Stage 3\nOAT sensitivity\nsweep", 3, 3),
        ("Stage 4\nB0–B4 baselines\n+ H2 trend", 3, 1),
        ("Stage 5\nCMV diagnostic\n(Harman + marker)", 4, 3),
        ("Stage 6\nTarget trial\nPICO doc", 4, 1),
        ("Stage 7\nCounterfactuals\nA, B, C\n+ H7 IUT", 5, 2),
        ("Stage 8\nTransportability\nfactor sweep", 6, 2),
    ]

    box_w, box_h = 0.85, 0.85
    for (label, x, y) in stages:
        rect = mpatches.FancyBboxPatch(
            (x - box_w / 2, y - box_h / 2),
            box_w, box_h,
            boxstyle="round,pad=0.05",
            edgecolor="#222222", facecolor="#e8eef8", linewidth=1.0,
        )
        ax.add_patch(rect)
        ax.text(x, y, label, ha="center", va="center",
                fontsize=8, family=FONT_FAMILY)

    # Arrows (A->B sequential)
    arrows = [
        (0, 2, 1, 2),  # 0 -> 1
        (1, 2, 2, 2),  # 1 -> 2
        (2, 2, 3, 3),  # 2 -> 3 (sensitivity)
        (2, 2, 3, 1),  # 2 -> 4 (baselines)
        (2, 2, 4, 3),  # 2 -> 5 (CMV)
        (2, 2, 4, 1),  # 2 -> 6 (target trial)
        (2, 2, 5, 2),  # 2 -> 7 (counterfactuals)
        (5, 2, 6, 2),  # 7 -> 8 (transportability)
    ]
    for (x1, y1, x2, y2) in arrows:
        ax.annotate(
            "", xy=(x2 - 0.43, y2), xytext=(x1 + 0.43, y1),
            arrowprops=dict(arrowstyle="->", color="#444444", lw=1.0,
                            connectionstyle="arc3,rad=0.05"),
        )

    # Inputs at left
    ax.text(-1.0, 2.7, "Inputs", ha="center", va="center",
            fontsize=10, weight="bold", family=FONT_FAMILY)
    ax.text(-1.0, 2.3, "N=354 individual\nharassment data",
            ha="center", va="center", fontsize=8, family=FONT_FAMILY)
    ax.text(-1.0, 1.7, "IEEE 7-cluster\ncentroids (M3-fixed)",
            ha="center", va="center", fontsize=8, family=FONT_FAMILY)
    ax.text(-1.0, 1.1, "MHLW Labor Force\n2022 marginals",
            ha="center", va="center", fontsize=8, family=FONT_FAMILY)
    # Arrows from inputs to Stage 0/1
    ax.annotate("", xy=(-0.42, 2.0), xytext=(-0.55, 2.3),
                arrowprops=dict(arrowstyle="->", color="#444444", lw=0.8))
    ax.annotate("", xy=(-0.42, 2.0), xytext=(-0.55, 1.7),
                arrowprops=dict(arrowstyle="->", color="#444444", lw=0.8))
    ax.annotate("", xy=(0.58, 2.0), xytext=(-0.55, 1.1),
                arrowprops=dict(arrowstyle="->", color="#444444", lw=0.8))

    ax.set_xlim(-1.6, 6.7)
    ax.set_ylim(0.2, 3.7)
    ax.set_aspect("equal")
    ax.set_axis_off()
    ax.set_title(
        "Figure 1. HEXACO 7-Typology Workplace Harassment Microsimulation Pipeline",
        fontsize=11, weight="bold", family=FONT_FAMILY, pad=12,
    )

    return _save_all_formats(fig, output_dir, "figure1_pipeline")


# ============================================================================
# Figure 2: 14-cell propensity heatmap
# ============================================================================


def figure2_cell_propensity_heatmap(
    cell_propensity_path: str | Path, output_dir: Path
) -> list[Path]:
    arrays, _ = load_artifacts(cell_propensity_path)
    p = arrays["point_power"]   # (14,)
    n = arrays["cell_n"]        # (14,)

    # Reshape to 7 clusters × 2 genders (cell = type * 2 + gender)
    p_grid = np.zeros((7, 2), dtype=float)
    n_grid = np.zeros((7, 2), dtype=int)
    for c in range(14):
        type_idx = c // 2
        gender_idx = c % 2
        p_grid[type_idx, gender_idx] = p[c]
        n_grid[type_idx, gender_idx] = int(n[c])

    fig, ax = plt.subplots(figsize=(5.5, 7.5), dpi=DPI)
    im = ax.imshow(p_grid, aspect="auto", cmap="RdYlBu_r", vmin=0, vmax=0.45)
    cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label("P̂_c (power harassment victimization)", fontsize=10,
                   family=FONT_FAMILY)

    # Annotations: P̂ + N
    for i in range(7):
        for j in range(2):
            txt = f"{p_grid[i, j]:.3f}\n(N={n_grid[i, j]})"
            color = "white" if p_grid[i, j] > 0.25 else "black"
            ax.text(j, i, txt, ha="center", va="center",
                    fontsize=8, family=FONT_FAMILY, color=color)
            # Dashed border for small cells (N < 20)
            if n_grid[i, j] < 20:
                rect = mpatches.Rectangle(
                    (j - 0.45, i - 0.45), 0.9, 0.9,
                    linewidth=1.5, edgecolor="black",
                    facecolor="none", linestyle="--",
                )
                ax.add_patch(rect)

    ax.set_xticks([0, 1])
    ax.set_xticklabels(["Female (0)", "Male (1)"])
    ax.set_yticks(range(7))
    ax.set_yticklabels([f"Cluster {k}" for k in range(7)])
    ax.set_xlabel("Gender", fontsize=10, family=FONT_FAMILY)
    ax.set_ylabel("HEXACO 7-typology cluster", fontsize=10, family=FONT_FAMILY)
    ax.set_title(
        "Figure 2. 14-Cell Power-Harassment Propensity Heatmap",
        fontsize=11, weight="bold", family=FONT_FAMILY, pad=10,
    )

    _set_font(fig)
    return _save_all_formats(fig, output_dir, "figure2_cell_propensity")


# ============================================================================
# Figure 3: National prevalence vs MHLW targets
# ============================================================================


def figure3_prevalence_vs_mhlw(
    aggregation_path: str | Path, output_dir: Path
) -> list[Path]:
    agg, _ = load_artifacts(aggregation_path)
    p_pred = float(agg["national_prevalence_point"][0])

    fig, ax = plt.subplots(figsize=(8, 5), dpi=DPI)
    target_keys = ["FY2016", "FY2020", "FY2023"]
    colors_target = ["#d62728", "#ff7f0e", "#2ca02c"]

    # MHLW targets as horizontal lines
    for k, color in zip(target_keys, colors_target):
        v = MHLW_VALIDATION_TARGETS[k]["value"]
        ax.axhline(v, linestyle="--", color=color, lw=1.5,
                   label=f"MHLW {k}: {v*100:.1f}%")

    # Power Harassment Prevention Law shaded period (2020-06 onwards)
    ax.axvspan(2020.4, 2026.0, alpha=0.13, color="#888888",
               label="2020 Power Harassment Prevention Law")

    # Predicted prevalence as bar at "model" position
    ax.bar([2024.0], [p_pred], width=0.7, color="#1f77b4", edgecolor="black",
           linewidth=1.0, label=f"Model predicted: {p_pred*100:.2f}%")

    # MHLW point markers along the year axis
    ax.scatter([2017.0, 2021.0, 2024.0], [0.325, 0.314, 0.193],
               s=80, c=colors_target, marker="D",
               edgecolors="black", linewidths=1.0, zorder=5,
               label="MHLW survey value (timeline)")

    ax.set_xlim(2015.5, 2025.5)
    ax.set_ylim(0, 0.45)
    ax.set_xlabel("Year of MHLW survey", fontsize=10, family=FONT_FAMILY)
    ax.set_ylabel("Past-3-year power harassment victimization rate",
                  fontsize=10, family=FONT_FAMILY)
    ax.set_title(
        "Figure 3. Model-Predicted National Prevalence vs MHLW Validation Targets",
        fontsize=11, weight="bold", family=FONT_FAMILY, pad=10,
    )
    ax.legend(loc="upper right", fontsize=8, framealpha=0.9)
    ax.grid(True, alpha=0.3)

    _set_font(fig)
    return _save_all_formats(fig, output_dir, "figure3_prevalence_vs_mhlw")


# ============================================================================
# Figure 4: OAT sensitivity tornado plot
# ============================================================================


def figure4_sensitivity_tornado(
    sensitivity_path: str | Path, output_dir: Path
) -> list[Path]:
    arrays, _ = load_artifacts(sensitivity_path)
    params = arrays["sweep_parameter"]
    values = arrays["sweep_value"]
    mape = arrays["point_mape_FY2016"]
    ci_lo = arrays["ci_lo_FY2016"]
    ci_hi = arrays["ci_hi_FY2016"]

    # Sort rows by parameter family then by sensitivity range (ascending)
    family_min = {}
    family_max = {}
    for i in range(len(params)):
        fam = params[i].decode()
        if fam == "baseline":
            continue
        family_min[fam] = min(family_min.get(fam, 1e9), float(mape[i]))
        family_max[fam] = max(family_max.get(fam, -1e9), float(mape[i]))
    families_by_sens = sorted(
        family_min.keys(),
        key=lambda f: family_max[f] - family_min[f],
        reverse=True,
    )

    fig, ax = plt.subplots(figsize=(9, 6), dpi=DPI)
    yticklabels = []
    yticks = []
    y_pos = 0
    baseline_mape = 45.51  # main config
    palette = {"binarization_threshold": "#d62728",
               "cluster_weight_perturbation": "#ff7f0e",
               "soft_tau_factor": "#1f77b4",
               "eb_scale": "#2ca02c"}

    for fam in families_by_sens:
        for i in range(len(params)):
            if params[i].decode() != fam:
                continue
            label_value = values[i].decode()
            mp = float(mape[i])
            lo = float(ci_lo[i])
            hi = float(ci_hi[i])
            color = palette.get(fam, "#888888")
            ax.barh(y_pos, mp - baseline_mape, left=baseline_mape,
                    color=color, edgecolor="black", linewidth=0.5, alpha=0.85)
            ax.errorbar(mp, y_pos, xerr=[[mp - lo], [hi - mp]],
                        fmt="none", ecolor="black", capsize=3, lw=0.8)
            yticklabels.append(f"{fam}  ({label_value})")
            yticks.append(y_pos)
            y_pos += 1
        y_pos += 0.3

    ax.axvline(baseline_mape, linestyle="--", color="#222222",
               linewidth=1.5, label=f"Baseline (main): {baseline_mape:.2f}%")
    ax.axvline(30.0, linestyle=":", color="green",
               linewidth=1.0, alpha=0.7, label="30% (Standard SUCCESS)")
    ax.axvline(60.0, linestyle=":", color="red",
               linewidth=1.0, alpha=0.7, label="60% (FAILURE)")

    ax.set_yticks(yticks)
    ax.set_yticklabels(yticklabels, fontsize=8, family=FONT_FAMILY)
    ax.invert_yaxis()
    ax.set_xlabel("FY2016 MAPE (%)", fontsize=10, family=FONT_FAMILY)
    ax.set_title(
        "Figure 4. OAT Sensitivity Tornado Plot Around Locked v2.0 Configuration",
        fontsize=11, weight="bold", family=FONT_FAMILY, pad=10,
    )
    ax.legend(loc="lower right", fontsize=8)
    ax.grid(True, axis="x", alpha=0.3)

    _set_font(fig)
    return _save_all_formats(fig, output_dir, "figure4_sensitivity_tornado")


# ============================================================================
# Figure 5: Counterfactual ΔP_x forest plot
# ============================================================================


def figure5_counterfactual_forest(
    counterfactual_path: str | Path, output_dir: Path
) -> list[Path]:
    arrays, meta = load_artifacts(counterfactual_path)
    pa = float(arrays["delta_p_a_point"][0])
    pb = float(arrays["delta_p_b_point"][0])
    pc = float(arrays["delta_p_c_point"][0])
    ci_a = arrays["delta_p_a_ci"]
    ci_b = arrays["delta_p_b_ci"]
    ci_c = arrays["delta_p_c_ci"]

    fig, ax = plt.subplots(figsize=(8, 4.5), dpi=DPI)
    labels = [
        "C: Structural (-20% propensity)",
        "B: Cluster reassign {0,4,6}",
        "A: Personality (+0.3 SD H/A/E)",
    ]
    points = [pc, pb, pa]
    cis_lo = [float(ci_c[0]), float(ci_b[0]), float(ci_a[0])]
    cis_hi = [float(ci_c[1]), float(ci_b[1]), float(ci_a[1])]
    colors = ["#2ca02c", "#888888", "#888888"]

    y_pos = np.arange(len(labels))
    for y, p, lo, hi, color in zip(y_pos, points, cis_lo, cis_hi, colors):
        # CI whisker
        ax.plot([lo, hi], [y, y], color=color, lw=2.0, solid_capstyle="round")
        # Point estimate
        ax.scatter([p], [y], s=140, c=color, edgecolors="black",
                   linewidths=1.0, zorder=5)
        # Annotation
        ax.text(hi + 0.005, y, f"  {p:+.4f} [{lo:+.4f}, {hi:+.4f}]",
                va="center", ha="left", fontsize=9, family=FONT_FAMILY)

    ax.axvline(0, linestyle="--", color="#555555", lw=1.0)
    ax.set_yticks(y_pos)
    ax.set_yticklabels(labels, fontsize=10, family=FONT_FAMILY)
    ax.set_xlim(-0.04, 0.09)
    ax.set_xlabel("ΔP (positive = reduction in prevalence)",
                  fontsize=10, family=FONT_FAMILY)
    h7 = meta.get("h7_classification", "REVERSAL")
    if isinstance(h7, bytes):
        h7 = h7.decode()
    ax.set_title(
        f"Figure 5. Counterfactual ΔP_x Forest Plot (H7 IUT: {h7})",
        fontsize=11, weight="bold", family=FONT_FAMILY, pad=10,
    )
    ax.grid(True, axis="x", alpha=0.3)

    _set_font(fig)
    return _save_all_formats(fig, output_dir, "figure5_counterfactual_forest")


# ============================================================================
# Figure 6: Transportability robustness
# ============================================================================


def figure6_transportability(
    transportability_path: str | Path, output_dir: Path
) -> list[Path]:
    arrays, _ = load_artifacts(transportability_path)
    factors = arrays["factors"]
    pa = arrays["delta_p_a_attenuated"]
    pb = arrays["delta_p_b_attenuated"]
    pc = arrays["delta_p_c_attenuated"]
    ci_a = arrays["ci_a"]
    ci_b = arrays["ci_b"]
    ci_c = arrays["ci_c"]
    classifications = arrays["h7_classifications"]

    fig, ax = plt.subplots(figsize=(8.5, 5.5), dpi=DPI)
    palette = {"A": "#1f77b4", "B": "#888888", "C": "#2ca02c"}

    # ΔP_C with shaded CI
    ax.fill_between(factors, ci_c[:, 0], ci_c[:, 1], alpha=0.18,
                    color=palette["C"], label="ΔP_C 95% CI")
    ax.plot(factors, pc, "o-", color=palette["C"], lw=2.0, markersize=8,
            label="C: Structural (-20%)")

    # ΔP_A with shaded CI
    ax.fill_between(factors, ci_a[:, 0], ci_a[:, 1], alpha=0.18,
                    color=palette["A"], label="ΔP_A 95% CI")
    ax.plot(factors, pa, "s-", color=palette["A"], lw=1.5, markersize=7,
            label="A: Personality (+0.3 SD)")

    # ΔP_B with shaded CI
    ax.fill_between(factors, ci_b[:, 0], ci_b[:, 1], alpha=0.18,
                    color=palette["B"], label="ΔP_B 95% CI")
    ax.plot(factors, pb, "^-", color=palette["B"], lw=1.5, markersize=7,
            label="B: Cluster reassign")

    ax.axhline(0, linestyle="--", color="#555555", lw=0.8)
    # Annotate H7 classification per factor
    for i, f in enumerate(factors):
        c = classifications[i].decode()
        ax.text(float(f), float(pc[i]) + 0.005,
                f"H7={c}", fontsize=7, ha="center",
                family=FONT_FAMILY, color="#2ca02c")

    ax.set_xlabel("Cultural attenuation factor F",
                  fontsize=10, family=FONT_FAMILY)
    ax.set_ylabel("ΔP_x (after attenuation)",
                  fontsize=10, family=FONT_FAMILY)
    ax.set_title(
        "Figure 6. H7 IUT Robustness Across Cultural Attenuation Factors",
        fontsize=11, weight="bold", family=FONT_FAMILY, pad=10,
    )
    ax.legend(loc="upper left", fontsize=8, framealpha=0.9)
    ax.grid(True, alpha=0.3)

    # Annotate factor anchor labels
    anchor_labels = {
        0.3: "Sapouna 2010\n(worst case)",
        0.5: "Nielsen 2017\n(expected)",
        0.7: "Mild",
        1.0: "Reference",
    }
    for f in factors:
        v = float(f)
        ax.text(v, ax.get_ylim()[0] + 0.005, anchor_labels.get(v, ""),
                ha="center", va="bottom", fontsize=7, alpha=0.7,
                family=FONT_FAMILY)

    _set_font(fig)
    return _save_all_formats(fig, output_dir, "figure6_transportability")


# ============================================================================
# Pipeline driver
# ============================================================================


def run_all(
    cell_propensity_path: str | Path,
    aggregation_path: str | Path,
    sensitivity_path: str | Path,
    counterfactual_path: str | Path,
    transportability_path: str | Path,
    output_dir: str | Path = DEFAULT_FIGURES_DIR,
) -> None:
    """Generate all 6 figures."""
    output_dir = Path(output_dir)

    print("[figures] Generating Figure 1 (pipeline schematic) ...")
    f1 = figure1_pipeline_schematic(output_dir)
    for f in f1:
        print(f"  Wrote {f}")

    print("[figures] Generating Figure 2 (14-cell propensity heatmap) ...")
    f2 = figure2_cell_propensity_heatmap(cell_propensity_path, output_dir)
    for f in f2:
        print(f"  Wrote {f}")

    print("[figures] Generating Figure 3 (prevalence vs MHLW) ...")
    f3 = figure3_prevalence_vs_mhlw(aggregation_path, output_dir)
    for f in f3:
        print(f"  Wrote {f}")

    print("[figures] Generating Figure 4 (sensitivity tornado) ...")
    f4 = figure4_sensitivity_tornado(sensitivity_path, output_dir)
    for f in f4:
        print(f"  Wrote {f}")

    print("[figures] Generating Figure 5 (counterfactual forest) ...")
    f5 = figure5_counterfactual_forest(counterfactual_path, output_dir)
    for f in f5:
        print(f"  Wrote {f}")

    print("[figures] Generating Figure 6 (transportability) ...")
    f6 = figure6_transportability(transportability_path, output_dir)
    for f in f6:
        print(f"  Wrote {f}")

    print("[figures] Done. All 6 figures generated in 3 formats each (PNG/PDF/SVG).")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--cell-propensity",
        type=Path,
        default=Path("output/supplementary/stage0_cell_propensity.h5"),
    )
    parser.add_argument(
        "--aggregation",
        type=Path,
        default=Path("output/supplementary/stage1_population_aggregation.h5"),
    )
    parser.add_argument(
        "--sensitivity",
        type=Path,
        default=Path("output/supplementary/stage3_sensitivity.h5"),
    )
    parser.add_argument(
        "--counterfactual",
        type=Path,
        default=Path("output/supplementary/stage7_counterfactual.h5"),
    )
    parser.add_argument(
        "--transportability",
        type=Path,
        default=Path("output/supplementary/stage8_transportability.h5"),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_FIGURES_DIR,
    )
    args = parser.parse_args()
    run_all(
        cell_propensity_path=args.cell_propensity,
        aggregation_path=args.aggregation,
        sensitivity_path=args.sensitivity,
        counterfactual_path=args.counterfactual,
        transportability_path=args.transportability,
        output_dir=args.output_dir,
    )


if __name__ == "__main__":
    main()
