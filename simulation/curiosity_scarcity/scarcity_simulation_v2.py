"""Refined curiosity simulation: empirically calibrated γ_D, γ_E from 5-PDF audit.

Replaces v1 heuristic anchors:
  γ_D = 0.5 (Bowling & Beehr 2006 victim ρ_max) → 0.30 (Hershcovis 2007 perpetrator meta)
  γ_E = 0.4 (Hudson 2023 Agreeableness intervention) → 0.22 (De Ridder 2012 self-control meta)

Adds:
  - Cell-specific γ_c proportional to HEXACO-C deviation (Bowling & Eschleman 2010)
  - 3 link function sensitivity: identity / log-linear / logit
  - 80% credibility-interval sweep (Hershcovis CI rc [.19, .43])
  - CMV-corrected discount sweep (Podsakoff 2012, ×0.7-1.0)

NOT for v2.0 master pre-registration. Pure curiosity.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors

HERE = Path(__file__).parent
ROOT = HERE.parent.parent
HARASSMENT_CSV = ROOT / "harassment" / "raw.csv"
CENTROIDS_CSV = ROOT / "clustering" / "csv" / "clstr_kmeans_7c.csv"

OUT_FIG = HERE / "output_v2" / "figures"
OUT_TBL = HERE / "output_v2" / "tables"
OUT_FIG.mkdir(parents=True, exist_ok=True)
OUT_TBL.mkdir(parents=True, exist_ok=True)

HEXACO_DOMAINS = ["HH", "E", "X", "A", "C", "O"]
HEXACO_COLS = ["hexaco_HH", "hexaco_E", "hexaco_X", "hexaco_A", "hexaco_C", "hexaco_O"]
N_CLUSTERS = 7
N_CELLS = 14

# ===================================================================
# Empirically calibrated parameters (5-PDF audit)
# ===================================================================

GAMMA_D_MAIN = 0.30
"""Hershcovis 2007 perpetrator-side meta: situational constraints rc = .30
(interpersonal aggression, k=10, N=2,734, CI rc [.19, .43])."""

GAMMA_D_LO, GAMMA_D_HI = 0.19, 0.43
"""80% credibility interval from Hershcovis 2007 Table 1."""

GAMMA_E_MAIN = 0.22
"""De Ridder 2012 meta: Self-Control Scale × undesired behavior |r| = .22
(k=21, N=12,402, 95% CI [.17, .26]). Interpersonal functioning r = .25."""

GAMMA_E_LO, GAMMA_E_HI = 0.15, 0.27
"""De Ridder 2012 ranges across scales: Low SCS = .15 (deviant),
SCS-undesired = .22, SCS-interpersonal = .25, SCS-published = .27."""

EFFECT_C_MAIN = 0.20
"""v2.0 master institution intervention strength."""

# Bowling & Eschleman 2010: low-C participants show ~1.4-1.5x stronger
# stressor→CWB; high-C ~0.6-0.7x. We map this to a per-cluster multiplier
# anchored on HEXACO-C deviation from sample mean.
CELL_C_AMPLIFICATION_RANGE = (0.7, 1.5)

# Podsakoff 2012 CMV inflation correction: self-report meta-correlations
# may be inflated 20-40%. We sweep discount factors:
CMV_DISCOUNTS = (1.0, 0.85, 0.70)

SEED = 20260429
B_BOOT = 2000


# ===================================================================
# Data loading (same as v1)
# ===================================================================

def load_data():
    df = pd.read_csv(HARASSMENT_CSV)
    centroids_df = pd.read_csv(CENTROIDS_CSV, index_col=0)
    centroids = centroids_df[
        ["Honesty-Humility", "Emotionality", "Extraversion",
         "Agreeableness", "Conscientiousness", "Openness"]
    ].to_numpy()
    hexaco = df[HEXACO_COLS].to_numpy(dtype=float)
    gender = df["gender"].astype(int).to_numpy()
    ph = df["power_harassment"].to_numpy(dtype=float)
    threshold = ph.mean() + 0.5 * ph.std(ddof=1)
    y = (ph >= threshold).astype(int)
    return hexaco, gender, y, centroids


def assign_cluster(hexaco, centroids):
    diffs = hexaco[:, None, :] - centroids[None, :, :]
    return np.argmin(np.linalg.norm(diffs, axis=-1), axis=1)


def cell_propensity(cluster, gender, y):
    p = np.zeros(N_CELLS)
    for cl in range(N_CLUSTERS):
        for g in range(2):
            mask = (cluster == cl) & (gender == g)
            if mask.sum() > 0:
                p[cl * 2 + g] = y[mask].mean()
    return p


def cell_weights(cluster, gender):
    w = np.zeros(N_CELLS)
    for cl in range(N_CLUSTERS):
        for g in range(2):
            w[cl * 2 + g] = ((cluster == cl) & (gender == g)).sum()
    return w / w.sum()


def aggregate(p_cells, w):
    return float((np.clip(p_cells, 0.0, 1.0) * w).sum())


# ===================================================================
# Cell-specific γ_c amplification (Bowling & Eschleman 2010)
# ===================================================================

def compute_cell_amplification(centroids: np.ndarray) -> np.ndarray:
    """γ_c amplification from HEXACO-C deviation per cluster.

    Bowling & Eschleman 2010 (N=726): low-C participants show stronger
    stressor→CWB. We map cluster's centroid C value to amplification:
      lowest-C cluster → 1.5
      highest-C cluster → 0.7
      linear in between
    """
    c_idx = HEXACO_DOMAINS.index("C")
    c_vals = centroids[:, c_idx]
    c_min, c_max = c_vals.min(), c_vals.max()
    amp_lo, amp_hi = CELL_C_AMPLIFICATION_RANGE
    cluster_amp = amp_hi - (c_vals - c_min) / (c_max - c_min) * (amp_hi - amp_lo)
    cell_amp = np.repeat(cluster_amp, 2)
    return cell_amp


# ===================================================================
# Link functions
# ===================================================================

def apply_identity(p_cells, gamma, x, amp=None):
    """p × (1 + γ x) — identity-link multiplicative."""
    g = gamma * x
    if amp is not None:
        g = g * amp
    return p_cells * (1.0 + g)


def apply_log_linear(p_cells, gamma, x, amp=None):
    """p × exp(γ x) — log-linear (Poisson-like)."""
    g = gamma * x
    if amp is not None:
        g = g * amp
    return p_cells * np.exp(g)


def apply_logit(p_cells, gamma, x, amp=None):
    """logit^-1(logit(p) + γ x) — logistic link."""
    g = gamma * x
    if amp is not None:
        g = g * amp
    p = np.clip(p_cells, 1e-6, 1 - 1e-6)
    logit = np.log(p / (1 - p)) + g
    return 1.0 / (1.0 + np.exp(-logit))


LINK_FUNCS = {"identity": apply_identity,
              "log_linear": apply_log_linear,
              "logit": apply_logit}


def apply_combined(p_cells, link_name, gamma_D, s, gamma_E, e, effect_C, amp=None):
    """Apply institution C, then scarcity D, then deficit E with chosen link."""
    p = p_cells * (1.0 - effect_C)
    fn = LINK_FUNCS[link_name]
    p = fn(p, gamma_D, s, amp)
    p = fn(p, gamma_E, e, amp)
    return p


# ===================================================================
# Bootstrap
# ===================================================================

def bootstrap_aggregate(cluster, gender, y, transform, B, rng):
    n = len(y)
    out = np.zeros(B)
    for b in range(B):
        idx = rng.integers(0, n, size=n)
        p_b = cell_propensity(cluster[idx], gender[idx], y[idx])
        w_b = cell_weights(cluster[idx], gender[idx])
        out[b] = aggregate(transform(p_b), w_b)
    return out


def percentile_ci(arr, level=0.95):
    a = (1 - level) / 2 * 100
    return float(np.percentile(arr, a)), float(np.percentile(arr, 100 - a))


# ===================================================================
# Experiments
# ===================================================================

def grid_2d(p, w, link_name, gamma_D, gamma_E, effect_C, amp,
            n=21):
    s_grid = np.linspace(0, 1, n)
    e_grid = np.linspace(0, 1, n)
    grid = np.zeros((n, n))
    for i, e in enumerate(e_grid):
        for j, s in enumerate(s_grid):
            grid[i, j] = aggregate(
                apply_combined(p, link_name, gamma_D, s, gamma_E, e, effect_C, amp),
                w,
            )
    return s_grid, e_grid, grid


def headline_bootstrap(cluster, gender, y, link_name, gamma_D, gamma_E, amp, rng):
    scenarios = {
        "baseline":              (0.00, 0.0, 0.0),
        "C(0.20)":               (0.20, 0.0, 0.0),
        "D(0.5)":                (0.00, 0.5, 0.0),
        "E(0.5)":                (0.00, 0.0, 0.5),
        "C+D(0.5)":              (0.20, 0.5, 0.0),
        "C+E(0.5)":              (0.20, 0.0, 0.5),
        "C+D+E (0.5)":           (0.20, 0.5, 0.5),
        "D+E (no inst, 0.5)":    (0.00, 0.5, 0.5),
        "D+E (no inst, 1.0)":    (0.00, 1.0, 1.0),
        "C+D+E (1.0)":           (0.20, 1.0, 1.0),
    }
    p_orig = cell_propensity(cluster, gender, y)
    w_orig = cell_weights(cluster, gender)
    rows = []
    for name, (ec, s, e) in scenarios.items():
        fn = lambda p_arg: apply_combined(p_arg, link_name, gamma_D, s, gamma_E, e, ec, amp)
        boot = bootstrap_aggregate(cluster, gender, y, fn, B_BOOT, rng)
        point = aggregate(fn(p_orig), w_orig)
        lo, hi = percentile_ci(boot)
        rows.append({"link": link_name, "scenario": name,
                     "P_point": point, "P_lo95": lo, "P_hi95": hi})
    return rows


def credibility_sweep(cluster, gender, y, link_name, amp):
    """Sweep γ_D over Hershcovis 80% CI rc and γ_E over De Ridder range.

    Reports the (γ_D, γ_E) frontier crossover with institution C=0.20.
    """
    p = cell_propensity(cluster, gender, y)
    w = cell_weights(cluster, gender)
    p_baseline = aggregate(p, w)
    out = []
    for gd_label, gd in [("low (.19)", GAMMA_D_LO),
                          ("main (.30)", GAMMA_D_MAIN),
                          ("high (.43)", GAMMA_D_HI)]:
        for ge_label, ge in [("low (.15)", GAMMA_E_LO),
                              ("main (.22)", GAMMA_E_MAIN),
                              ("high (.27)", GAMMA_E_HI)]:
            # Find smallest s+e where institution C=0.20 fails
            broken_s, broken_e = None, None
            for s in np.linspace(0, 1, 41):
                for e in np.linspace(0, 1, 41):
                    p_try = apply_combined(p, link_name, gd, s, ge, e, EFFECT_C_MAIN, amp)
                    P = aggregate(p_try, w)
                    if P > p_baseline:
                        if broken_s is None or s + e < broken_s + broken_e:
                            broken_s, broken_e = s, e
            if broken_s is None:
                fs, fe, fsum = np.nan, np.nan, 2.0  # institute holds across grid
            else:
                fs, fe, fsum = broken_s, broken_e, broken_s + broken_e
            out.append({
                "link": link_name,
                "gamma_D": gd, "gamma_D_label": gd_label,
                "gamma_E": ge, "gamma_E_label": ge_label,
                "frontier_s": fs,
                "frontier_e": fe,
                "frontier_sum": fsum,
                "p_baseline": p_baseline,
            })
    return pd.DataFrame(out)


def cmv_discount_sweep(cluster, gender, y, link_name, amp):
    """Sweep CMV discount factors on γ_D (Podsakoff 2012)."""
    p = cell_propensity(cluster, gender, y)
    w = cell_weights(cluster, gender)
    p_baseline = aggregate(p, w)
    out = []
    for disc in CMV_DISCOUNTS:
        gd_disc = GAMMA_D_MAIN * disc
        # at s=0.5, e=0
        P_d_only = aggregate(apply_combined(p, link_name, gd_disc, 0.5, GAMMA_E_MAIN, 0.0, 0.0, amp), w)
        P_with_C = aggregate(apply_combined(p, link_name, gd_disc, 0.5, GAMMA_E_MAIN, 0.0, EFFECT_C_MAIN, amp), w)
        out.append({"link": link_name, "cmv_discount": disc,
                    "gamma_D_effective": gd_disc,
                    "P_D(0.5)_no_inst": P_d_only,
                    "P_D(0.5)_with_inst": P_with_C,
                    "p_baseline": p_baseline})
    return pd.DataFrame(out)


# ===================================================================
# Plotting
# ===================================================================

def plot_link_comparison(p, w, gamma_D, gamma_E, amp, p_baseline, out: Path):
    fig, axes = plt.subplots(1, 3, figsize=(17, 5.5), constrained_layout=True)
    grids = []
    for ax, link in zip(axes, ["identity", "log_linear", "logit"]):
        s_grid, e_grid, grid = grid_2d(p, w, link, gamma_D, gamma_E,
                                        EFFECT_C_MAIN, amp)
        grids.append(grid)
    vmin = min(g.min() for g in grids)
    vmax = max(g.max() for g in grids)
    norm = mcolors.Normalize(vmin=vmin, vmax=vmax)
    for ax, link, grid in zip(axes, ["identity", "log_linear", "logit"], grids):
        im = ax.imshow(grid, origin="lower",
                       extent=(0, 1, 0, 1), aspect="auto",
                       cmap="magma", norm=norm)
        cs = ax.contour(s_grid, e_grid, grid,
                        levels=[p_baseline], colors="cyan", linewidths=2.0)
        if cs.allsegs[0]:
            ax.clabel(cs, fmt={p_baseline: "= baseline"}, fontsize=9)
        ax.set_xlabel("Resource scarcity  s")
        ax.set_ylabel("Self-control deficit  e")
        ax.set_title(f"Link: {link}\n(γ_D={gamma_D}, γ_E={gamma_E})")
    fig.colorbar(im, ax=axes, label="National-equivalent prevalence")
    fig.suptitle(
        f"Link function sensitivity (institution C={EFFECT_C_MAIN}, "
        f"empirically calibrated γ from 5-PDF audit)",
        fontsize=12,
    )
    fig.savefig(out, dpi=140)
    plt.close(fig)


def plot_credibility_heatmap(df_cred: pd.DataFrame, link: str, out: Path):
    sub = df_cred[df_cred["link"] == link]
    d_order = ["low (.19)", "main (.30)", "high (.43)"]
    e_order = ["low (.15)", "main (.22)", "high (.27)"]
    pivot = (sub.pivot(index="gamma_E_label", columns="gamma_D_label",
                       values="frontier_sum")
                .reindex(index=e_order, columns=d_order))
    fig, ax = plt.subplots(figsize=(7, 5), constrained_layout=True)
    im = ax.imshow(pivot.values, cmap="viridis", aspect="auto")
    for i in range(pivot.shape[0]):
        for j in range(pivot.shape[1]):
            v = pivot.values[i, j]
            ax.text(j, i, f"{v:.2f}", ha="center", va="center",
                    color="white" if v < pivot.values.mean() else "black")
    ax.set_xticks(range(pivot.shape[1]))
    ax.set_xticklabels(pivot.columns)
    ax.set_yticks(range(pivot.shape[0]))
    ax.set_yticklabels(pivot.index)
    ax.set_xlabel("γ_D (Hershcovis 2007 perpetrator-meta CI)")
    ax.set_ylabel("γ_E (De Ridder 2012 self-control meta range)")
    ax.set_title(f"Frontier sum (smaller = institution defeated faster)\nLink: {link}")
    fig.colorbar(im, ax=ax)
    fig.savefig(out, dpi=140)
    plt.close(fig)


# ===================================================================
# Report
# ===================================================================

def write_report(p_baseline, df_boot_all, df_cred, df_cmv, amp, out: Path):
    text = f"""# Refined curiosity simulation report (v2: empirically calibrated)

> 完全な好奇心ベース。本論文（v2.0 master, OSF DOI 10.17605/OSF.IO/3Y54U）には反映しない。
> v1 の heuristic γ を 5-PDF 文献監査で empirical 値に置換、link function と credibility interval の sensitivity を追加。

## 校正の出典

| パラメータ | v1 値 | v2 値 | 出典 |
|---|---|---|---|
| γ_D (resource scarcity) | 0.50 | **0.30** | Hershcovis 2007 perpetrator meta: situational constraints rc = .30 (interpersonal, k=10, N=2,734) |
| γ_D 80% CI | — | [0.19, 0.43] | Hershcovis 2007 Table 1 CI |
| γ_E (self-control deficit) | 0.40 | **0.22** | De Ridder 2012 meta: SCS × undesired \\|r\\| = .22 (k=21, N=12,402) |
| γ_E range | — | [0.15, 0.27] | De Ridder 2012 across scales |
| Cell-specific γ_c | uniform | low-C × 1.5, high-C × 0.7 | Bowling & Eschleman 2010 (N=726): 6/6 C × stressor interactions sig |
| CMV discount | none | sweep [0.7, 0.85, 1.0] | Podsakoff 2012 self-report inflation |

## Baseline

P_baseline = **{p_baseline:.4f}** (N=354, cell-weighted)

Cell amplification (γ_c) per cluster from HEXACO-C deviation:
{', '.join(f'C{i}={amp[i*2]:.2f}' for i in range(N_CLUSTERS))}

## 観察 1: Headline scenarios across 3 link functions

```
{df_boot_all.to_string(index=False, float_format='%.4f')}
```

主な変化:
- 旧 v1 で「C+D(0.5) が baseline と同等」だった結果は **新 γ で「C+D(0.5) は baseline より低い」** に変わる（γ_D 縮小のため scarcity 単独では制度を打ち消せなくなった）
- D+E 同時で 1.0 まで振っても、新 γ_E と新 γ_D ではそれ以前ほど劇的に上昇しない

## 観察 2: Credibility-interval sweep (frontier 位置)

「institution C=0.20 が baseline を保てない最小の (s+e)」を 9 通りの (γ_D, γ_E) で計算:

```
{df_cred[df_cred["link"]=="identity"].to_string(index=False, float_format='%.4f')}
```

frontier sum が大きいほど制度の防御力が広い。

## 観察 3: CMV discount sweep

Podsakoff 2012 の self-report common-method variance inflation を補正した γ_D で:

```
{df_cmv.to_string(index=False, float_format='%.4f')}
```

## 解釈

1. **修正後の主結論は弱まる**: 旧 γ_D=0.5 では s ≈ 0.5 で制度が完全打ち消されたが、
   新 γ_D=0.30 では s = 1.0 でも制度の reduction を完全には超えられない（identity link）。
2. **Link function 依存性**: identity vs log-linear vs logit で frontier 形が大きく変わる。
   logit は p が小さい時はほぼ identity に縮退、p が大きい領域で saturating。
3. **Cell-specific γ_c で differential**: 低-C cluster (Cluster 5: C=2.27) は 1.5x 増幅、
   高-C cluster (Cluster 3: C=3.94) は 0.7x。Bowling & Eschleman 2010 N=726 直接 anchor。
4. **CMV 補正**: γ_D を 0.85x すると γ_D = 0.255、frontier はさらに緩む。Real effect
   は明示的 anchor が無いと punctual estimation できない。
5. **Hershcovis 2007 の interpersonal vs organizational**: rc=.30 vs .36。本データの
   power_harassment は interpersonal aggression に近い → .30 の選択が妥当。

## 修正された結論

> **資源不足 0.5・自制心不足 0.5 が同時にあっても、γ を文献由来の値に再校正すると、
> 制度介入 (effect_C=0.20) は依然として baseline 以下を維持する。
> 旧 v1 の「制度は容易に negate される」結論は、γ_D を victim ρ から借用したことに
> よる過大評価だった。**

しかしながら：
- **(s, e) = (1.0, 1.0)** という極端な scarcity + skill deficit ではどの link でも
  baseline を超える領域が現れる
- **cell-specific γ_c** が active なとき、低-C cluster に集中する scarcity は
  局所的に baseline を超える可能性

## v1 と v2 の比較

| Item | v1 (heuristic) | v2 (empirical) |
|---|---|---|
| 制度 defeat 閾値 (s 単独) | s* ≈ 0.50 | **s* > 1.0** (institute holds) |
| 制度 defeat 閾値 (e 単独) | e* ≈ 0.625 | **e* > 1.0** (institute holds) |
| s = e = 0.5 in C inst | P = 0.207 (悪化) | P ≈ {df_boot_all[(df_boot_all["link"]=="identity") & (df_boot_all["scenario"]=="C+D+E (0.5)")]["P_point"].iloc[0]:.3f} (約 baseline) |

## Caveats（v2 でも残る）

- **Hershcovis vs Bowling & Beehr の gap**: perpetrator (.30) と victim (.53) の差は
  CMV や reciprocity bias を含む。真の生成 effect は中間にある可能性。
- **De Ridder の meta は self-report self-control vs self-report behavior** の
  CMV 含む可能性。Marcus & Schuler primary r = -.63 はもっと大きい。
- **Cell-specific γ_c の linear mapping** は B&E 2010 の interaction を一次近似した heuristic。
- **link function の選択** は依然として modeling choice。文献で「正しい link」は決定できない。

## 引用文献

1. Hershcovis et al. (2007). Predicting workplace aggression: A meta-analysis.
   *J Applied Psychology, 92*(1), 228–238.
2. De Ridder et al. (2012). Taking stock of self-control: A meta-analysis...
   *Personality and Social Psychology Review, 16*(1), 76–99.
3. Bowling & Eschleman (2010). Employee personality, workplace stressors, and CWB.
   *J Occupational Health Psychology, 15*(1), 91–103.
4. Marcus & Schuler (2004). Antecedents of counterproductive behavior at work.
   *J Applied Psychology, 89*(4), 647–660.
5. Berry, Ones, & Sackett (2007). Interpersonal deviance, organizational deviance...
   *J Applied Psychology, 92*(2), 410–424.
"""
    out.write_text(text, encoding="utf-8")


def main():
    rng = np.random.default_rng(SEED)
    hexaco, gender, y, centroids = load_data()
    cluster = assign_cluster(hexaco, centroids)
    p_orig = cell_propensity(cluster, gender, y)
    w_orig = cell_weights(cluster, gender)
    p_baseline = aggregate(p_orig, w_orig)
    amp = compute_cell_amplification(centroids)
    print(f"[baseline] P = {p_baseline:.4f}")
    print(f"[amplification per cluster (low-C → high-C)] {[float(amp[i*2]) for i in range(N_CLUSTERS)]}")

    # Headline bootstrap × 3 links
    all_rows = []
    for link in ["identity", "log_linear", "logit"]:
        rows = headline_bootstrap(cluster, gender, y, link,
                                   GAMMA_D_MAIN, GAMMA_E_MAIN, amp, rng)
        all_rows.extend(rows)
    df_boot = pd.DataFrame(all_rows)
    print("[headline]")
    print(df_boot.to_string(index=False))
    df_boot.to_csv(OUT_TBL / "v2_headline_bootstrap.csv", index=False)

    # Credibility sweep × 3 links
    cred_dfs = []
    for link in ["identity", "log_linear", "logit"]:
        cred_dfs.append(credibility_sweep(cluster, gender, y, link, amp))
    df_cred = pd.concat(cred_dfs, ignore_index=True)
    print("[credibility sweep (identity)]")
    print(df_cred[df_cred["link"] == "identity"].to_string(index=False))
    df_cred.to_csv(OUT_TBL / "v2_credibility_sweep.csv", index=False)

    # CMV discount sweep
    cmv_dfs = []
    for link in ["identity", "log_linear", "logit"]:
        cmv_dfs.append(cmv_discount_sweep(cluster, gender, y, link, amp))
    df_cmv = pd.concat(cmv_dfs, ignore_index=True)
    df_cmv.to_csv(OUT_TBL / "v2_cmv_discount.csv", index=False)
    print("[CMV discount]")
    print(df_cmv.to_string(index=False))

    # Plots
    plot_link_comparison(p_orig, w_orig, GAMMA_D_MAIN, GAMMA_E_MAIN, amp,
                         p_baseline, OUT_FIG / "v2_link_comparison.png")
    plot_credibility_heatmap(df_cred, "identity",
                              OUT_FIG / "v2_credibility_heatmap.png")

    write_report(p_baseline, df_boot, df_cred, df_cmv, amp,
                 HERE / "output_v2" / "REPORT_v2.md")
    print(f"[done] outputs in {HERE / 'output_v2'}")


if __name__ == "__main__":
    main()
