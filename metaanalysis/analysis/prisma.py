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
    fig, ax = plt.subplots(figsize=(10, 13))
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 13)
    ax.axis("off")

    # Stage labels (left side)
    stage_x = 0.15
    stages = [
        (11.5, "Identification"),
        (9.0, "Screening"),
        (5.5, "Eligibility"),
        (2.5, "Included"),
    ]
    for y, label in stages:
        ax.text(stage_x, y, label, fontsize=12, weight="bold",
                rotation=90, va="center", ha="center", family="serif")

    # Main flow boxes
    main_x = 1.5
    main_w = 5.5

    # 1. Identification
    draw_box(ax, main_x, 11.5, main_w, 1.0,
             "Records identified from databases (n = 80)*\n"
             "Records identified from other sources:\n"
             "  Preliminary informal search (n = 28)\n"
             "  Citation snowballing + open-access DL (n = 12)\n"
             "Total records identified (n = 120)",
             fontsize=9, fill="#e8f1f8")

    # Arrow down
    draw_arrow(ax, main_x + main_w / 2, 11.5, main_x + main_w / 2, 10.9)

    # Dedup → right branch
    draw_box(ax, main_x, 10.0, main_w, 0.9,
             "Records after duplicates removed (n ≈ 80)\n"
             "Duplicates removed (n ≈ 40)",
             fontsize=9, fill="#e8f1f8")

    # 2. Screening
    draw_arrow(ax, main_x + main_w / 2, 10.0, main_x + main_w / 2, 9.4)
    draw_box(ax, main_x, 8.5, main_w, 0.9,
             "Records screened (title/abstract) (n = 80)",
             fontsize=9, fill="#fff4d8")

    # Right side: exclusions at t/a
    excl_x = main_x + main_w + 0.5
    excl_w = 3.2
    draw_box(ax, excl_x, 8.3, excl_w, 1.1,
             "Records excluded at title/abstract (n = 25)\n"
             "  Wrong population / non-students (n = 10)\n"
             "  Clearly non-Big-Five framework (n = 8)\n"
             "  No online learning component (n = 7)",
             fontsize=8, fill="#fde9e9")
    draw_arrow(ax, main_x + main_w, 8.9, excl_x, 8.9)

    # 3. Eligibility / full-text
    draw_arrow(ax, main_x + main_w / 2, 8.5, main_x + main_w / 2, 7.9)
    draw_box(ax, main_x, 7.0, main_w, 0.9,
             "Full-text reports sought for retrieval (n = 43)",
             fontsize=9, fill="#fff4d8")

    draw_arrow(ax, main_x + main_w / 2, 7.0, main_x + main_w / 2, 6.4)
    draw_box(ax, main_x, 5.5, main_w, 0.9,
             "Reports not retrieved / paywall (n = 5)\n"
             "Reports assessed for eligibility (n = 38)",
             fontsize=9, fill="#fef3cd")

    # Right side: exclusions at full-text
    draw_box(ax, excl_x, 4.6, excl_w, 1.9,
             "Reports excluded at full-text (n = 7):\n"
             "  Non-Big-Five framework (n = 5)\n"
             "    MBTI (n = 1), Proactive (n = 1)\n"
             "    TAM (n = 1), TUE (n = 1), other (n = 1)\n"
             "  Face-to-face modality (n = 4)\n"
             "    A-09, A-10, A-14, A-16\n"
             "  Sample overlap (n = 1): A-05 / A-04\n"
             "  Effect size not extractable (n = 1): A-24",
             fontsize=8, fill="#fde9e9")
    draw_arrow(ax, main_x + main_w, 5.9, excl_x, 5.5)

    # 4. Included
    draw_arrow(ax, main_x + main_w / 2, 5.5, main_x + main_w / 2, 4.4)
    draw_box(ax, main_x, 3.0, main_w, 1.3,
             "Studies included in systematic review (n = 31)\n"
             "  Primary achievement pool (direct r): n = 10\n"
             "    [A-01, A-02, A-15, A-17, A-22, A-23, A-29,\n"
             "     A-30*, A-31, A-37]\n"
             "  β-converted contributors (P-Brown): n = 4\n"
             "  Secondary/narrative only: n = 17",
             fontsize=8.5, fill="#d4edda")

    # Bottom note / caption
    ax.text(5, 0.8,
            "PRISMA 2020 flow diagram (adapted from Page et al., 2021).",
            ha="center", fontsize=9, style="italic", family="serif")
    ax.text(5, 0.3,
            "*Records identified via WebSearch-based formal search "
            "(deviation from pre-registered direct database access; see Methods).",
            ha="center", fontsize=7.5, family="serif")

    plt.tight_layout()
    fig.savefig(OUT, dpi=150, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"Wrote {OUT}")


if __name__ == "__main__":
    main()
