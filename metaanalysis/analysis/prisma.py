"""
Generate PRISMA 2020 flow diagram for the Big Five × online learning meta-analysis.

Output: metaanalysis/analysis/figures/prisma_flow.png

Follows PRISMA 2020 template (Page et al., 2021) with four stages:
Identification → Screening → Eligibility → Included.
"""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
from pathlib import Path

OUT = Path("/home/user/paper/metaanalysis/analysis/figures/prisma_flow.png")


def draw_box(ax, x, y, w, h, text, fontsize=9, fill="#f0f0f0", edge="black"):
    box = FancyBboxPatch(
        (x, y), w, h,
        boxstyle="round,pad=0.02,rounding_size=0.03",
        edgecolor=edge, facecolor=fill, linewidth=1.2,
    )
    ax.add_patch(box)
    ax.text(x + w / 2, y + h / 2, text,
            ha="center", va="center", fontsize=fontsize,
            wrap=True, family="serif")


def draw_arrow(ax, x1, y1, x2, y2):
    arrow = FancyArrowPatch(
        (x1, y1), (x2, y2),
        arrowstyle="-|>", mutation_scale=15,
        linewidth=1.2, color="black",
    )
    ax.add_patch(arrow)


def main():
    """PRISMA-ScR flow. Counts before the full-text stage were not logged
    and are shown as approximate; full-text counts come from Table 1
    (data_extraction_populated.csv) and the full-text exclusions recorded
    in deep_reading_notes.md (A-14 and five reports without a Big Five
    measure)."""
    fig, ax = plt.subplots(figsize=(10, 12))
    ax.set_xlim(-0.2, 10.2)
    ax.set_ylim(0, 12)
    ax.axis("off")

    stage_x = 0.2
    for y, label in [(10.6, "Identification"), (8.5, "Screening"),
                     (5.6, "Eligibility"), (2.4, "Included")]:
        ax.text(stage_x, y, label, fontsize=12, weight="bold",
                rotation=90, va="center", ha="center", family="serif")

    main_x, main_w = 1.0, 4.9
    excl_x, excl_w = main_x + main_w + 0.3, 3.7
    mid = main_x + main_w / 2

    draw_box(ax, main_x, 10.0, main_w, 1.3,
             "Records identified\n"
             "  Web search engine, 8 queries (n \u2248 80)\n"
             "  Previously known primary studies (n = 28)\n"
             "  Reference lists of benchmark meta-analyses (n = 5)",
             fontsize=9, fill="#e8f1f8")
    draw_arrow(ax, mid, 10.0, mid, 9.3)

    draw_box(ax, main_x, 8.1, main_w, 1.2,
             "Records screened (title/abstract)\n"
             "Duplicates removed and records excluded at this stage:\n"
             "numbers not logged",
             fontsize=9, fill="#fff4d8")
    draw_arrow(ax, mid, 8.1, mid, 7.4)

    draw_box(ax, main_x, 6.6, main_w, 0.8,
             "Reports sought for retrieval (n = 37)",
             fontsize=9, fill="#fff4d8")
    draw_box(ax, excl_x, 6.6, excl_w, 0.8,
             "Reports not retrieved (n = 1)\n  Full text unavailable: A-27",
             fontsize=8.5, fill="#fde9e9")
    draw_arrow(ax, main_x + main_w, 7.0, excl_x, 7.0)
    draw_arrow(ax, mid, 6.6, mid, 5.9)

    draw_box(ax, main_x, 5.1, main_w, 0.8,
             "Reports assessed for eligibility (n = 36)",
             fontsize=9, fill="#fef3cd")
    draw_box(ax, excl_x, 4.2, excl_w, 2.1,
             "Reports excluded (n = 11)\n"
             "  No Big Five measure (n = 5)\n"
             "  Face-to-face only (n = 4):\n"
             "    A-09, A-10, A-14, A-16\n"
             "  Sample overlap (n = 1): A-05\n"
             "  No extractable effect size (n = 1): A-24",
             fontsize=8.5, fill="#fde9e9")
    draw_arrow(ax, main_x + main_w, 5.5, excl_x, 5.3)
    draw_arrow(ax, mid, 5.1, mid, 4.0)

    draw_box(ax, main_x, 1.6, main_w, 2.4,
             "Studies included in the scoping review (n = 25)\n"
             "\n"
             "  Exploratory quantitative pool (n = 10)\n"
             "    Direct correlations (n = 8; one entered as r = 0)\n"
             "    Converted from \u03b2 (n = 2): A-28, A-30\n"
             "  Narrative map only (n = 15)\n"
             "    Secondary outcomes only (n = 13)\n"
             "    No trait\u2013achievement r (n = 2): A-25, A-26",
             fontsize=8.5, fill="#d4edda")

    ax.text(5, 0.9, "PRISMA-ScR flow diagram (Tricco et al., 2018; layout adapted from Page et al., 2021).",
            ha="center", fontsize=9, style="italic", family="serif")
    ax.text(5, 0.45, "Counts before the full-text stage were not logged; 31 of the 37 reports are catalogued in Table 1.",
            ha="center", fontsize=7.5, family="serif")

    plt.tight_layout()
    fig.savefig(OUT, dpi=150, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"Wrote {OUT}")


if __name__ == "__main__":
    main()
