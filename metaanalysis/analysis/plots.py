"""
Generate forest and funnel plots for each Big Five trait.

Outputs to metaanalysis/analysis/figures/.
"""
import sys
from pathlib import Path
from math import log, exp, sqrt

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
from pool import (  # noqa: E402
    TRAITS, fisher_z, back_transform_z, var_z,
    pool_random_effects, load_extractions, extract_effect_for_trait,
)

FIG_DIR = Path("/home/user/paper/metaanalysis/analysis/figures")
FIG_DIR.mkdir(exist_ok=True)

TRAIT_NAMES = {
    "O": "Openness", "C": "Conscientiousness",
    "E": "Extraversion", "A": "Agreeableness",
    "N": "Neuroticism",
}


def collect_trait_data(rows, trait):
    data = []
    for row in rows:
        ext = extract_effect_for_trait(row, trait)
        if ext is None:
            continue
        r, n, source = ext
        try:
            z = fisher_z(r)
        except (ValueError, ZeroDivisionError):
            continue
        label = f"{row['study_id']} {row.get('first_author','')} ({row.get('year','')})"
        ci_z_lo = z - 1.96 / sqrt(n - 3)
        ci_z_hi = z + 1.96 / sqrt(n - 3)
        data.append({
            "label": label, "r": r, "n": n,
            "r_ci_lo": back_transform_z(ci_z_lo),
            "r_ci_hi": back_transform_z(ci_z_hi),
            "z": z, "v": var_z(n), "source": source,
        })
    return data


def forest_plot(trait, data, output):
    if len(data) < 2:
        return

    # Sort by effect size (descending)
    data = sorted(data, key=lambda d: d["r"], reverse=True)

    # Pool
    ys = [d["z"] for d in data]
    vs = [d["v"] for d in data]
    pooled = pool_random_effects(ys, vs)
    pooled_r = pooled["r_pooled"]
    pooled_lo = pooled["r_ci_lo"]
    pooled_hi = pooled["r_ci_hi"]

    k = len(data)
    fig_h = max(3.5, k * 0.35 + 2.5)
    fig, ax = plt.subplots(figsize=(9, fig_h))

    # Y positions: studies top-down, pooled at bottom
    y_positions = list(range(k, 0, -1))

    for i, d in enumerate(data):
        y = y_positions[i]
        # Square marker sized by weight (1/SE²)
        w = 1.0 / d["v"]
        marker_size = 50 + 300 * (w / max(1.0 / dd["v"] for dd in data))
        ax.errorbar(
            d["r"], y,
            xerr=[[d["r"] - d["r_ci_lo"]], [d["r_ci_hi"] - d["r"]]],
            fmt="s", color="steelblue", markersize=sqrt(marker_size) / 2,
            capsize=3, elinewidth=1,
        )

    # Pooled diamond
    y_pooled = 0
    diamond_x = [pooled_lo, pooled_r, pooled_hi, pooled_r]
    diamond_y = [y_pooled, y_pooled + 0.3, y_pooled, y_pooled - 0.3]
    ax.fill(diamond_x, diamond_y, color="darkred", alpha=0.8)

    # Zero line
    ax.axvline(0, color="gray", linestyle="--", linewidth=0.8)
    # Pooled line
    ax.axvline(pooled_r, color="darkred", linestyle=":", linewidth=0.6, alpha=0.5)

    # Labels
    ax.set_yticks([0] + y_positions)
    ax.set_yticklabels(
        [f"Random-effects pooled (k={k})"] + [d["label"] for d in data],
        fontsize=9,
    )
    ax.set_xlabel("Correlation r (95% CI)", fontsize=11)
    ax.set_title(
        f"Forest plot: {TRAIT_NAMES[trait]} × Academic Achievement "
        f"in Online Learning Environments\n"
        f"Random-effects REML + HKSJ: r = {pooled_r:.3f} "
        f"[{pooled_lo:.3f}, {pooled_hi:.3f}], I² = {pooled['I2']:.1f}%",
        fontsize=11,
    )
    ax.set_xlim(-0.6, 0.7)
    ax.set_ylim(-0.8, k + 0.5)
    ax.grid(axis="x", alpha=0.3)

    # Add r values text on right side
    for i, d in enumerate(data):
        y = y_positions[i]
        ax.text(0.65, y,
                f"{d['r']:+.3f} [{d['r_ci_lo']:+.3f}, {d['r_ci_hi']:+.3f}]  N={d['n']}",
                va="center", fontsize=8, family="monospace")
    ax.text(0.65, 0,
            f"{pooled_r:+.3f} [{pooled_lo:+.3f}, {pooled_hi:+.3f}]",
            va="center", fontsize=9, weight="bold", family="monospace")

    plt.tight_layout()
    fig.savefig(output, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved {output}")


def funnel_plot(trait, data, output):
    if len(data) < 3:
        return

    ys = np.array([d["z"] for d in data])
    vs = np.array([d["v"] for d in data])
    ses = np.sqrt(vs)

    # Pooled
    pooled = pool_random_effects(ys.tolist(), vs.tolist())
    z_pooled = pooled["z_pooled"]

    fig, ax = plt.subplots(figsize=(7, 6))

    # Plot each study
    rs = [back_transform_z(y) for y in ys]
    ax.scatter(rs, ses, s=60, color="steelblue", edgecolor="black", zorder=3)

    # Pseudo-95% CI triangle (funnel)
    se_range = np.linspace(0, max(ses) * 1.2, 100)
    r_center = back_transform_z(z_pooled)
    # Approximate CI in r space (after back-transform)
    lo_line = np.array([back_transform_z(z_pooled - 1.96 * s) for s in se_range])
    hi_line = np.array([back_transform_z(z_pooled + 1.96 * s) for s in se_range])

    ax.plot(lo_line, se_range, color="gray", linestyle="--", linewidth=1)
    ax.plot(hi_line, se_range, color="gray", linestyle="--", linewidth=1)
    ax.axvline(r_center, color="darkred", linestyle=":", linewidth=1,
               label=f"Pooled r = {r_center:.3f}")
    ax.axvline(0, color="gray", linestyle="-", linewidth=0.5, alpha=0.5)

    ax.invert_yaxis()
    ax.set_xlabel("Correlation r", fontsize=11)
    ax.set_ylabel("Standard Error (Fisher z)", fontsize=11)
    ax.set_title(
        f"Funnel plot: {TRAIT_NAMES[trait]} × Academic Achievement\n"
        f"Pseudo-95% CI bounds",
        fontsize=11,
    )
    ax.legend(fontsize=9)
    ax.grid(alpha=0.3)

    plt.tight_layout()
    fig.savefig(output, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved {output}")


def main():
    rows = load_extractions()
    for trait in TRAITS:
        data = collect_trait_data(rows, trait)
        if not data:
            print(f"{trait}: no data")
            continue
        print(f"{trait}: {len(data)} studies")
        forest_plot(trait, data, FIG_DIR / f"forest_{trait}.png")
        funnel_plot(trait, data, FIG_DIR / f"funnel_{trait}.png")


if __name__ == "__main__":
    main()
