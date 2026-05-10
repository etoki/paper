# ICERI 2026 — Education-level x Discipline Cross-Tab

## Cell-level k map (across all 5 traits, max k per cell)

| level \ discipline | Humanities | Mixed | Psychology | STEM |
|---|---|---|---|---|
| **Graduate** | 0 | 1 | 0 | 0 |
| **Mixed_UG_Grad** | 1 | 1 | 0 | 0 |
| **UG** | 0 | 3 | 3 | 1 |

## Pooled effects per cell (k>=2 only)

| Trait | Level | Discipline | k | r [95% CI] |
|-------|-------|------------|---|------------|
| A | UG | Mixed | 3 | 0.011 [-0.130, 0.152] |
| A | UG | Psychology | 2 | 0.152 [0.098, 0.205] |
| C | UG | Mixed | 3 | 0.136 [-0.081, 0.340] |
| C | UG | Psychology | 3 | 0.292 [0.123, 0.444] |
| E | UG | Mixed | 3 | -0.028 [-0.252, 0.199] |
| E | UG | Psychology | 2 | 0.115 [-0.444, 0.610] |
| N | UG | Mixed | 3 | -0.055 [-0.324, 0.223] |
| N | UG | Psychology | 3 | -0.037 [-0.268, 0.199] |
| O | UG | Mixed | 3 | 0.029 [-0.126, 0.183] |
| O | UG | Psychology | 2 | 0.160 [-0.973, 0.986] |

## Interaction model

- k observations: 47
- tau^2 used in weights: 0.0103
- reference cell: trait=O, level=Graduate, discipline=Humanities

**Joint Wald test on level x discipline interactions**: chi^2(6) = 1.638, p = 0.9498

## Caveats
- Most cells have k <= 1; the cross-tab is mostly a *coverage map* showing where evidence is dense vs sparse.
- Education-level was pre-registered as a moderator in the parent preprint but not quantitatively executed (k constraint). This paper completes that registered analysis and adds the discipline crossing.
- Discipline classification is the heuristic in `inputs/derive_studies_csv.py::classify_discipline`; single coder.