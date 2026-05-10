#!/usr/bin/env python3
"""Build PRISMA 2020 flow diagrams for the 3 conference papers.

The parent meta-analysis corpus is shared, so the Identification /
Screening / Eligibility / Included counts are identical across the
three papers. The "Included in synthesis" terminal box differs per
venue: each paper restricts the parent primary pool to its own subset
(modality-stratified for ECEL, Asian-only for ICEEL, education x
discipline cross-tab for ICERI).

Outputs (one PNG per venue, plus a shared parent-flow PNG):
    figures/prisma_flow_parent.png    -- Identification -> Included (k=12)
    figures/prisma_flow_ecel.png      -- + ECEL subset (modality cells, k=11 after U->A/M)
    figures/prisma_flow_iceel.png     -- + Asian subset (k=2)
    figures/prisma_flow_iceri.png     -- + cross-tab cells (12 candidate cells)

Counts traced from:
    metaanalysis/search_log.md (Identification / Dedup / Screened)
    metaanalysis/conference_submissions/ecel/full_paper.md
        ("Of the 31 primary studies cataloged at full-text assessment,
         25 were retained for qualitative synthesis and 12 contributed
         at least one extractable Pearson correlation")
    metaanalysis/conference_submissions/inputs/studies.csv
        (31 rows; 12 primary-pool studies)
"""
from __future__ import annotations

from pathlib import Path

import matplotlib.patches as mpatches
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[3]
WORKSPACE = ROOT / "metaanalysis" / "conference_submissions"
FIG_DIR = WORKSPACE / "figures"
FIG_DIR.mkdir(exist_ok=True)


# ----------------------------------------------------------------------
# Reusable PRISMA box drawer
# ----------------------------------------------------------------------
def box(ax, xy, w, h, text, *, fc="#FFFFFF", ec="#222222", fontsize=9):
    rect = mpatches.FancyBboxPatch(
        xy, w, h, boxstyle="round,pad=0.02,rounding_size=0.04",
        linewidth=1.0, edgecolor=ec, facecolor=fc,
    )
    ax.add_patch(rect)
    ax.text(
        xy[0] + w / 2, xy[1] + h / 2, text,
        ha="center", va="center", fontsize=fontsize, wrap=True,
    )


def arrow(ax, x1, y1, x2, y2):
    ax.annotate(
        "", xy=(x2, y2), xytext=(x1, y1),
        arrowprops=dict(arrowstyle="-|>", color="#222222", linewidth=1.0),
    )


# ----------------------------------------------------------------------
# Build a single PRISMA figure with optional venue-specific terminal box
# ----------------------------------------------------------------------
def build_flow(venue_terminal: tuple[str, str] | None, out_path: Path, title: str):
    """venue_terminal = (terminal-box-text, caption-line) or None."""
    fig, ax = plt.subplots(figsize=(8.0, 10.0))
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 12 if venue_terminal else 11)
    ax.set_aspect("equal")
    ax.axis("off")
    ax.set_title(title, fontsize=11, weight="bold", pad=12)

    # Section labels (left margin)
    ax.text(0.3, 10.5, "Identification", rotation=90, ha="center", va="center",
            fontsize=10, weight="bold", color="#444")
    ax.text(0.3, 8.0,  "Screening",      rotation=90, ha="center", va="center",
            fontsize=10, weight="bold", color="#444")
    ax.text(0.3, 5.6,  "Eligibility",    rotation=90, ha="center", va="center",
            fontsize=10, weight="bold", color="#444")
    ax.text(0.3, 3.0,  "Included",       rotation=90, ha="center", va="center",
            fontsize=10, weight="bold", color="#444")

    # ---- Identification ----
    box(ax, (1.0, 10.3), 4.5, 0.8,
        "Records identified from databases\n(WebSearch x 8 queries; n = 80)",
        fc="#E8F1FB")
    box(ax, (5.7, 10.3), 3.6, 0.8,
        "Records identified from\nother sources\n(prior informal n = 28; benchmark metas n = 5)",
        fc="#E8F1FB", fontsize=8)

    # ---- Dedup ----
    box(ax, (3.4, 9.0), 3.2, 0.6,
        "Duplicates removed (n ~ 45)", fc="#FFF7E0")
    arrow(ax, 3.25, 10.3, 4.5, 9.6)
    arrow(ax, 7.5, 10.3, 5.6, 9.6)

    # ---- Screening ----
    box(ax, (1.0, 7.6), 4.5, 0.8,
        "Records screened\n(title + abstract; n ~ 68)",
        fc="#E8F1FB")
    box(ax, (5.8, 7.6), 3.5, 0.8,
        "Records excluded at T/A\n(n ~ 25: not online,\nwrong population, commentary)",
        fc="#FBEAEA", fontsize=8)
    arrow(ax, 5.0, 9.0, 3.25, 8.4)
    arrow(ax, 5.5, 8.0, 5.8, 8.0)

    # ---- Retrieval ----
    box(ax, (1.0, 6.3), 4.5, 0.8,
        "Reports sought for retrieval\n(n ~ 43)",
        fc="#E8F1FB")
    arrow(ax, 3.25, 7.6, 3.25, 7.1)

    # ---- Eligibility ----
    box(ax, (1.0, 5.0), 4.5, 0.8,
        "Reports assessed for eligibility\n(n = 31, full-text, catalogued)",
        fc="#E8F1FB")
    box(ax, (5.8, 5.0), 3.5, 0.8,
        "Reports excluded at full-text\n(n ~ 12: face-to-face,\nno extractable r, sample overlap)",
        fc="#FBEAEA", fontsize=8)
    arrow(ax, 3.25, 6.3, 3.25, 5.8)
    arrow(ax, 5.5, 5.4, 5.8, 5.4)

    # ---- Included: qualitative synthesis ----
    box(ax, (1.0, 3.5), 4.5, 0.8,
        "Studies in qualitative synthesis\n(k = 25)",
        fc="#E5F4E8")
    arrow(ax, 3.25, 5.0, 3.25, 4.3)

    # ---- Included: quantitative pool ----
    box(ax, (1.0, 2.0), 4.5, 0.8,
        "Studies in primary quantitative pool\n(k = 12; per-trait k = 9-10)",
        fc="#E5F4E8")
    arrow(ax, 3.25, 3.5, 3.25, 2.8)

    # ---- Optional venue terminal box ----
    if venue_terminal:
        terminal, caption = venue_terminal
        box(ax, (1.0, 0.4), 4.5, 1.0,
            terminal, fc="#D4ECDA", fontsize=9)
        arrow(ax, 3.25, 2.0, 3.25, 1.4)
        ax.text(5.0, 0.1, caption, ha="center", va="center",
                fontsize=7, style="italic", color="#555")

    fig.text(0.5, 0.02,
             "Adapted from Page et al. (2021), PRISMA 2020 statement (BMJ 372:n71). "
             "Identification + Screening counts are approximate (~) where the deduplication "
             "and screening logs in metaanalysis/search_log.md report bracketed estimates.",
             ha="center", fontsize=7, color="#555", wrap=True)

    fig.tight_layout()
    fig.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"Wrote {out_path}")


def main():
    build_flow(
        None,
        FIG_DIR / "prisma_flow_parent.png",
        "PRISMA 2020 flow diagram: parent meta-analysis corpus",
    )
    build_flow(
        (
            "ECEL subset: modality-stratified pool\n"
            "k = 11 contribute extractable r\n"
            "(asynchronous k = 6, mixed-online k = 5;\n"
            "synchronous k = 1 reported narratively)",
            "ECEL 2026 — modality-stratified random-effects meta-regression."
        ),
        FIG_DIR / "prisma_flow_ecel.png",
        "PRISMA 2020 flow diagram (ECEL 2026 submission)",
    )
    build_flow(
        (
            "ICEEL subset: Asian primary pool\n"
            "k = 2 with extractable r per trait\n"
            "(A-28 Yu, China; A-31 Rivers, Japan)\n"
            "+ A-25 Tokiwa Japan in narrative synthesis",
            "ICEEL 2026 — Hofstede-moderated within-Asia synthesis."
        ),
        FIG_DIR / "prisma_flow_iceel.png",
        "PRISMA 2020 flow diagram (ICEEL 2026 submission)",
    )
    build_flow(
        (
            "ICERI subset: 4 (level) x 4 (discipline) cross-tab\n"
            "12 candidate cells; 2 cells reach k >= 2\n"
            "(UG x Mixed k = 3; UG x Psychology k = 2-3)\n"
            "K-12 row + Graduate row are evidence deserts"
        ,
            "ICERI 2026 — education x discipline cross-tabulated interaction."
        ),
        FIG_DIR / "prisma_flow_iceri.png",
        "PRISMA 2020 flow diagram (ICERI 2026 submission)",
    )


if __name__ == "__main__":
    main()
