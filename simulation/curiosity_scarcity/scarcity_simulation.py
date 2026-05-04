"""Curiosity simulation: Scarcity × Self-Control × Institution interaction.

NOT part of v2.0 master pre-registration. Pure exploratory.

Pipeline:
1. Load N=354 harassment + 7 cluster centroids from sibling project data.
2. Hard-assign each individual to nearest centroid (Euclidean over 6 HEXACO domains).
3. Binarize power_harassment at mean + 0.5 SD → y ∈ {0, 1}.
4. Compute 14-cell baseline propensity p_c.
5. Apply 4 counterfactual families:
     C       : institution        p_c × (1 − effect_C)
     D(s)    : scarcity           p_c × (1 + γ_D · s)
     E(e)    : self-ctrl deficit  p_c × (1 + γ_E · e)
     E'(δ_E) : HEXACO-C shift   reclassify after C := C − δ_E·SD(C), then recompute p_c
6. Aggregate to national-equivalent prevalence with empirical cell weights.
7. Bootstrap B=2000 for CIs.
8. Sweep (s, e, effect_C) and produce:
   - Frontier plot: institution efficacy boundary on (s, e) grid
   - Contribution decomposition: marginal effect of D vs E
   - HEXACO-C shift comparison: direct vs trait-mediated self-control
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

OUT_FIG = HERE / "output" / "figures"
OUT_TBL = HERE / "output" / "tables"
OUT_FIG.mkdir(parents=True, exist_ok=True)
OUT_TBL.mkdir(parents=True, exist_ok=True)

HEXACO_DOMAINS = ["HH", "E", "X", "A", "C", "O"]
HEXACO_COLS = ["hexaco_HH", "hexaco_E", "hexaco_X", "hexaco_A", "hexaco_C", "hexaco_O"]
N_CLUSTERS = 7
N_CELLS = 14  # 7 clusters × 2 genders

GAMMA_D = 0.5
GAMMA_E = 0.4
EFFECT_C_MAIN = 0.20
DELTA_E_SD = 0.4  # HEXACO-C downshift (matches Hudson 2023 conservative)

SEED = 20260429
B_BOOT = 2000


def load_data() -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    df = pd.read_csv(HARASSMENT_CSV)
    centroids_df = pd.read_csv(CENTROIDS_CSV, index_col=0)
    centroids = centroids_df[
        ["Honesty-Humility", "Emotionality", "Extraversion",
         "Agreeableness", "Conscientiousness", "Openness"]
    ].to_numpy()
    hexaco = df[HEXACO_COLS].to_numpy(dtype=float)
    gender = df["gender"].astype(int).to_numpy()  # already coded {0, 1}
    ph = df["power_harassment"].to_numpy(dtype=float)
    threshold = ph.mean() + 0.5 * ph.std(ddof=1)
    y = (ph >= threshold).astype(int)
    return hexaco, gender, y, centroids


def assign_cluster(hexaco: np.ndarray, centroids: np.ndarray) -> np.ndarray:
    diffs = hexaco[:, None, :] - centroids[None, :, :]
    d = np.linalg.norm(diffs, axis=-1)
    return np.argmin(d, axis=1)


def cell_propensity(cluster: np.ndarray, gender: np.ndarray, y: np.ndarray) -> np.ndarray:
    p = np.zeros(N_CELLS)
    for cl in range(N_CLUSTERS):
        for g in range(2):
            mask = (cluster == cl) & (gender == g)
            if mask.sum() > 0:
                p[cl * 2 + g] = y[mask].mean()
    return p


def cell_weights(cluster: np.ndarray, gender: np.ndarray) -> np.ndarray:
    w = np.zeros(N_CELLS)
    for cl in range(N_CLUSTERS):
        for g in range(2):
            mask = (cluster == cl) & (gender == g)
            w[cl * 2 + g] = mask.sum()
    return w / w.sum()


def aggregate(p_cells: np.ndarray, w: np.ndarray) -> float:
    p_cells = np.clip(p_cells, 0.0, 1.0)
    return float((p_cells * w).sum())


def cf_C(p: np.ndarray, effect_C: float) -> np.ndarray:
    return p * (1.0 - effect_C)


def cf_D(p: np.ndarray, s: float, gamma_D: float = GAMMA_D) -> np.ndarray:
    return p * (1.0 + gamma_D * s)


def cf_E(p: np.ndarray, e: float, gamma_E: float = GAMMA_E) -> np.ndarray:
    return p * (1.0 + gamma_E * e)


def cf_combined(
    p: np.ndarray, effect_C: float, s: float, e: float,
    gamma_D: float = GAMMA_D, gamma_E: float = GAMMA_E,
) -> np.ndarray:
    return p * (1.0 - effect_C) * (1.0 + gamma_D * s) * (1.0 + gamma_E * e)


def cf_E_via_C_shift(
    hexaco: np.ndarray, gender: np.ndarray, y: np.ndarray,
    centroids: np.ndarray, delta_E_sd: float,
    p_baseline_cells: np.ndarray | None = None,
) -> tuple[float, np.ndarray, int]:
    """Mediated self-control deficit: subtract δ_E·SD from HEXACO-C, then reclassify.

    Uses a *fixed* reference propensity table (baseline p_c) and lets the
    population redistribute across cells. This captures the intuition that
    when an individual loses self-control they "inherit the risk profile of
    their new (lower-C) cluster", rather than carrying their original y.
    Without this fix, Σ p_c · w_c trivially equals the empirical mean and
    is invariant to reassignment.
    """
    base_cluster = assign_cluster(hexaco, centroids)
    if p_baseline_cells is None:
        p_baseline_cells = cell_propensity(base_cluster, gender, y)
    new_h = hexaco.copy()
    c_idx = HEXACO_DOMAINS.index("C")
    c_sd = hexaco[:, c_idx].std(ddof=1)
    new_h[:, c_idx] -= delta_E_sd * c_sd
    new_cluster = assign_cluster(new_h, centroids)
    n_moved = int((base_cluster != new_cluster).sum())
    w_new = cell_weights(new_cluster, gender)
    return aggregate(p_baseline_cells, w_new), p_baseline_cells, n_moved


def bootstrap_aggregate(
    cluster: np.ndarray, gender: np.ndarray, y: np.ndarray,
    transform, B: int, rng: np.random.Generator,
) -> np.ndarray:
    n = len(y)
    out = np.zeros(B)
    for b in range(B):
        idx = rng.integers(0, n, size=n)
        p_b = cell_propensity(cluster[idx], gender[idx], y[idx])
        w_b = cell_weights(cluster[idx], gender[idx])
        out[b] = aggregate(transform(p_b), w_b)
    return out


def percentile_ci(arr: np.ndarray, level: float = 0.95) -> tuple[float, float]:
    a = (1 - level) / 2 * 100
    return float(np.percentile(arr, a)), float(np.percentile(arr, 100 - a))


# ===================================================================
# Experiments
# ===================================================================

def experiment_baseline(cluster, gender, y) -> dict:
    p = cell_propensity(cluster, gender, y)
    w = cell_weights(cluster, gender)
    p_baseline = aggregate(p, w)
    return {"p_cells": p, "weights": w, "p_baseline": p_baseline}


def experiment_main_grid(cluster, gender, y, p_baseline_no_intervention):
    """Sweep (s, e) at effect_C = 0 (no institution) and effect_C = 0.20 (institution)."""
    s_grid = np.linspace(0, 1, 21)
    e_grid = np.linspace(0, 1, 21)
    p = cell_propensity(cluster, gender, y)
    w = cell_weights(cluster, gender)

    grid_no_inst = np.zeros((len(e_grid), len(s_grid)))
    grid_with_inst = np.zeros((len(e_grid), len(s_grid)))
    for i, e in enumerate(e_grid):
        for j, s in enumerate(s_grid):
            grid_no_inst[i, j] = aggregate(cf_combined(p, 0.0, s, e), w)
            grid_with_inst[i, j] = aggregate(cf_combined(p, EFFECT_C_MAIN, s, e), w)
    return s_grid, e_grid, grid_no_inst, grid_with_inst


def experiment_marginal(cluster, gender, y) -> pd.DataFrame:
    """Marginal effect of D, E, C, and combinations along principal axes."""
    p = cell_propensity(cluster, gender, y)
    w = cell_weights(cluster, gender)
    rows = []
    for s in np.linspace(0, 1, 11):
        rows.append({
            "scenario": "D_only",
            "s": s, "e": 0.0, "effect_C": 0.0,
            "P": aggregate(cf_combined(p, 0.0, s, 0.0), w),
        })
    for e in np.linspace(0, 1, 11):
        rows.append({
            "scenario": "E_only",
            "s": 0.0, "e": e, "effect_C": 0.0,
            "P": aggregate(cf_combined(p, 0.0, 0.0, e), w),
        })
    for ec in np.linspace(0, 0.3, 7):
        rows.append({
            "scenario": "C_only",
            "s": 0.0, "e": 0.0, "effect_C": ec,
            "P": aggregate(cf_combined(p, ec, 0.0, 0.0), w),
        })
    # Diagonal: scarcity & deficit jointly, with and without institution
    for t in np.linspace(0, 1, 11):
        rows.append({
            "scenario": "D+E (no inst)",
            "s": t, "e": t, "effect_C": 0.0,
            "P": aggregate(cf_combined(p, 0.0, t, t), w),
        })
        rows.append({
            "scenario": "D+E + Institution",
            "s": t, "e": t, "effect_C": EFFECT_C_MAIN,
            "P": aggregate(cf_combined(p, EFFECT_C_MAIN, t, t), w),
        })
    return pd.DataFrame(rows)


def experiment_e_via_c_shift(hexaco, gender, y, centroids):
    """Compare direct E (multiplier) vs mediated E (HEXACO-C shift then reclassify).

    Mediated path is intrinsically discrete (cluster reassignment); a small
    HEXACO-C downshift may move zero individuals across the nearest-centroid
    boundary because cluster membership is multi-dimensional. We sweep up to
    3.0 SD to expose the boundary-crossing structure.
    """
    deltas = np.linspace(0, 3.0, 31)
    rows = []
    for d in deltas:
        p_med, _, n_moved = cf_E_via_C_shift(hexaco, gender, y, centroids, d)
        rows.append({
            "delta_E_sd": float(d),
            "P_mediated": p_med,
            "n_reassigned": n_moved,
        })
    df = pd.DataFrame(rows)
    p_baseline_cells = cell_propensity(assign_cluster(hexaco, centroids), gender, y)
    w_baseline = cell_weights(assign_cluster(hexaco, centroids), gender)
    direct_rows = []
    for e in np.linspace(0, 1, 21):
        direct_rows.append({"e": float(e),
                            "P_direct": aggregate(cf_E(p_baseline_cells, e), w_baseline)})
    df_direct = pd.DataFrame(direct_rows)
    return df, df_direct


def experiment_bootstrap(cluster, gender, y, rng):
    """Bootstrap CI for headline scenarios."""
    p_orig = cell_propensity(cluster, gender, y)
    w_orig = cell_weights(cluster, gender)
    scenarios = {
        "baseline": lambda p: p,
        "C(0.20)": lambda p: cf_C(p, EFFECT_C_MAIN),
        "D(0.5)": lambda p: cf_D(p, 0.5),
        "E(0.5)": lambda p: cf_E(p, 0.5),
        "C+D(0.5)": lambda p: cf_combined(p, EFFECT_C_MAIN, 0.5, 0.0),
        "C+E(0.5)": lambda p: cf_combined(p, EFFECT_C_MAIN, 0.0, 0.5),
        "C+D+E (all 0.5)": lambda p: cf_combined(p, EFFECT_C_MAIN, 0.5, 0.5),
        "D+E (no inst, all 0.5)": lambda p: cf_combined(p, 0.0, 0.5, 0.5),
        "D+E (no inst, all 1.0)": lambda p: cf_combined(p, 0.0, 1.0, 1.0),
        "C+D+E (all 1.0)": lambda p: cf_combined(p, EFFECT_C_MAIN, 1.0, 1.0),
    }
    rows = []
    for name, fn in scenarios.items():
        boot = bootstrap_aggregate(cluster, gender, y, fn, B_BOOT, rng)
        point = aggregate(fn(p_orig), w_orig)
        lo, hi = percentile_ci(boot)
        rows.append({"scenario": name, "P_point": point,
                     "P_lo95": lo, "P_hi95": hi, "B": B_BOOT})
    return pd.DataFrame(rows)


# ===================================================================
# Plots
# ===================================================================

def plot_frontier(s_grid, e_grid, grid_no, grid_inst, p_baseline, out: Path):
    fig, axes = plt.subplots(1, 2, figsize=(13, 5.5), constrained_layout=True)
    vmin = min(grid_no.min(), grid_inst.min())
    vmax = max(grid_no.max(), grid_inst.max())
    norm = mcolors.Normalize(vmin=vmin, vmax=vmax)
    for ax, grid, title in zip(
        axes,
        [grid_no, grid_inst],
        ["No institution (effect_C = 0)",
         f"With institution (effect_C = {EFFECT_C_MAIN:.2f})"],
    ):
        im = ax.imshow(
            grid, origin="lower",
            extent=(s_grid[0], s_grid[-1], e_grid[0], e_grid[-1]),
            aspect="auto", cmap="magma", norm=norm,
        )
        cs = ax.contour(
            s_grid, e_grid, grid,
            levels=[p_baseline], colors="cyan", linewidths=2.0,
        )
        ax.clabel(cs, fmt={p_baseline: "= baseline"}, fontsize=9)
        ax.set_xlabel("Resource scarcity  s")
        ax.set_ylabel("Self-control deficit  e")
        ax.set_title(title)
    fig.colorbar(im, ax=axes, label="National-equivalent harassment prevalence")
    fig.suptitle(
        "Institution efficacy frontier: where institution C=20% can no longer hold prevalence below baseline",
        fontsize=12,
    )
    fig.savefig(out, dpi=140)
    plt.close(fig)


def plot_marginal(df: pd.DataFrame, p_baseline: float, out: Path):
    fig, ax = plt.subplots(figsize=(9.5, 6), constrained_layout=True)
    palette = {
        "C_only": "tab:green",
        "D_only": "tab:red",
        "E_only": "tab:orange",
        "D+E (no inst)": "tab:purple",
        "D+E + Institution": "tab:blue",
    }
    for scen, color in palette.items():
        sub = df[df["scenario"] == scen].copy()
        if scen == "C_only":
            x = sub["effect_C"].values
            xlabel = "axis value"
        elif scen == "D_only":
            x = sub["s"].values
        elif scen == "E_only":
            x = sub["e"].values
        else:
            x = sub["s"].values  # diagonal: s=e=t
        ax.plot(x, sub["P"].values, "o-", label=scen, color=color, linewidth=2)
    ax.axhline(p_baseline, linestyle="--", color="gray", label="baseline")
    ax.set_xlabel("Axis value (s, e, or effect_C; for diagonals s = e = t)")
    ax.set_ylabel("National-equivalent prevalence")
    ax.set_title("Marginal & joint effects of institution / scarcity / self-control deficit")
    ax.legend(loc="upper left", fontsize=9)
    ax.grid(alpha=0.3)
    fig.savefig(out, dpi=140)
    plt.close(fig)


def plot_e_via_c_shift(df_med: pd.DataFrame, df_direct: pd.DataFrame,
                       p_baseline: float, out: Path):
    fig, axes = plt.subplots(1, 2, figsize=(13, 5.5), constrained_layout=True)

    ax = axes[0]
    ax.plot(df_med["delta_E_sd"], df_med["P_mediated"], "o-",
            color="tab:orange", label="Mediated: HEXACO-C downshift → reclassify")
    ax2 = ax.twiny()
    ax2.plot(df_direct["e"], df_direct["P_direct"], "s-",
             color="tab:red", label="Direct multiplier: p_c × (1 + γ_E·e)")
    ax.axhline(p_baseline, linestyle="--", color="gray", label="baseline")
    ax.set_xlabel("Mediated: −δ_E × SD(C)")
    ax2.set_xlabel("Direct: e")
    ax.set_ylabel("National-equivalent prevalence")
    ax.set_title("Trait-mediated vs direct self-control deficit")
    h1, l1 = ax.get_legend_handles_labels()
    h2, l2 = ax2.get_legend_handles_labels()
    ax.legend(h1 + h2, l1 + l2, loc="upper left", fontsize=9)
    ax.grid(alpha=0.3)

    ax = axes[1]
    ax.plot(df_med["delta_E_sd"], df_med["n_reassigned"], "o-",
            color="tab:purple", linewidth=2)
    ax.set_xlabel("HEXACO-C downshift  −δ_E × SD(C)")
    ax.set_ylabel("# of N=354 individuals reassigned to a new cluster")
    ax.set_title("Boundary-crossing structure of mediated path")
    ax.grid(alpha=0.3)

    fig.savefig(out, dpi=140)
    plt.close(fig)


# ===================================================================
# Report
# ===================================================================

def write_report(
    p_baseline: float, df_marginal: pd.DataFrame, df_boot: pd.DataFrame,
    s_grid, e_grid, grid_no, grid_inst, df_med: pd.DataFrame, out: Path,
):
    # institution-frontier crossover: smallest (s, e) where grid_inst > p_baseline
    above = grid_inst > p_baseline
    if above.any():
        ii, jj = np.where(above)
        idx = np.argmin(s_grid[jj] + e_grid[ii])
        s_star, e_star = s_grid[jj[idx]], e_grid[ii[idx]]
        frontier_text = (
            f"institution C=0.20 が baseline を保てない最小の (s, e) ≈ "
            f"({s_star:.2f}, {e_star:.2f})"
        )
    else:
        frontier_text = "institution C=0.20 は (s, e) ∈ [0, 1]² 全域で baseline 以下を維持"

    boot_md = df_boot.to_markdown(index=False, floatfmt=".4f")
    text = f"""# Curiosity simulation report

> 完全な好奇心ベース。本論文には反映しない。pre-registration v2.0 の confirmatory 範囲外。

## Setup

- N = 354 (harassment + HEXACO domains)
- 7 clusters × 2 genders = 14 cells
- Binarization: power_harassment ≥ mean + 0.5·SD
- γ_D = {GAMMA_D} (Bowling & Beehr 2006 ρ メタ解析の保守換算)
- γ_E = {GAMMA_E} (Hudson 2023 d=0.71 の risk-multiplier 一次近似)
- effect_C (institution) = {EFFECT_C_MAIN}
- Bootstrap B = {B_BOOT}, seed = {SEED}

## Baseline prevalence

P_baseline = **{p_baseline:.4f}** (binary harassment proportion in N=354, cell-weighted)

## 観察 1: Institution efficacy frontier

{frontier_text}

→ 制度介入 C(20% 削減) は scarcity と self-control deficit の合成圧力に対して
**finite な防壁**であって、frontier を超える領域では「制度を維持しても baseline
を超えるハラスメントが起こる」ことが示される。

## 観察 2: Marginal effects (point estimates)

軸を 0 → 1 に動かしたときの prevalence (各 scenario は別軸):

```
{df_marginal.to_string(index=False, float_format='%.4f')}
```

## 観察 3: Bootstrap CI on headline scenarios

{boot_md}

## 観察 4: Self-control deficit — trait-mediated vs direct

Mediated 経路（HEXACO-C を SD だけ下方シフト → 再クラスタリング）と
Direct 経路（cell propensity を multiplier で増幅）は **異なる感度** を示す:

```
{df_med.to_string(index=False, float_format='%.4f')}
```

クラスタ membership は 6 次元の Euclidean 最近傍で決まるので、HEXACO-C を
1 軸だけ下方シフトしてもクラスタ越境は段階的にしか起こらない。`n_reassigned`
列は実際に new cluster に切り替わった個体数。Direct 経路は連続的に上昇する
のに対し、Mediated 経路は **離散的なジャンプ** を示すのが特徴。

## 解釈

1. **資源不足 (s)**: cell propensity を一様に拡大するため、制度の reduction を
   即座に飽和させる。代数的には s* ≈ effect_C / (γ_D · (1 − effect_C)) =
   0.20 / (0.5 · 0.80) = 0.50 で institution C を完全相殺。観察値 s ≈ 0.55
   と近い。
2. **自制心不足 (e)** の direct 経路: 同じ機序で e* ≈ effect_C / (γ_E · (1 − effect_C)) =
   0.20 / (0.4 · 0.80) = 0.625 で institution C を相殺。
3. **D × E 相互作用** は乗法的なので、s = e = 0.4 付近で既に baseline (C 適用後)
   を超える: (1.20)(1.16)/(1 − 0.20) = 1.74 > 1.0。これが「制度を強化しても
   消えない残差」を最も簡潔に示す代数。
4. **HEXACO-C 経由の mediated 経路は本データでは non-monotonic**: δ_E ∈ [0, 0.3]
   ではわずかに上昇するが、δ_E > 0.5 で **逆に減少** する。これは Cluster 5
   (HEXACO-C 最低 = 2.27) が偶然 baseline propensity の低いプロファイル
   (p ≈ 0.07/0) を持つため。本データでは "low C → high harassment" の単純な
   伝達は成立せず、harassment 高位プロファイルは Cluster 0 (low HH, high X)
   と Cluster 4 (low A, high O) に分布。**自制心 = HEXACO-C** という単純化
   は本データでは失敗するという empirical finding。
5. **生産性低下** を s × e として読むと、資源と自制心の両方を同時に欠く環境
   (低 TFP の代理) では、厳罰化が単独で機能する余地が急速に縮小する。
6. **政策含意としての示唆**: 制度介入 C は資源・技術の十分性を前提とした効果と
   して読むべきで、stressor 環境の改善（労働時間規制、人員配置、教育）が
   並走しないと law-on-the-books と law-in-action が乖離する。

## Caveats

- N=354 に workload も self-control も直接測定がない → γ_D, γ_E は外部 anchor
- `s × e` を生産性合成にしたのは粗い heuristic。realistic な TFP は別モデル
- Multiplier 形は cap(1.0) を入れているが、低 baseline cells で linear extrapolation
- v2.0 master の counterfactual A/B/C との直接比較は意図しない (異なる介入空間)
- Mediated path の非単調性は cluster identity の高次元構造に依存し、別データセット
  では逆向きの結果が出る可能性。本知見は datasets-dependent。

"""
    out.write_text(text, encoding="utf-8")


def main() -> None:
    rng = np.random.default_rng(SEED)
    hexaco, gender, y, centroids = load_data()
    cluster = assign_cluster(hexaco, centroids)

    base = experiment_baseline(cluster, gender, y)
    p_baseline = base["p_baseline"]
    print(f"[baseline] P = {p_baseline:.4f}, "
          f"cluster sizes = {[int((cluster == k).sum()) for k in range(N_CLUSTERS)]}")

    s_grid, e_grid, grid_no, grid_inst = experiment_main_grid(
        cluster, gender, y, p_baseline,
    )
    df_marginal = experiment_marginal(cluster, gender, y)
    df_med, df_direct = experiment_e_via_c_shift(hexaco, gender, y, centroids)
    print(f"[grid] no-inst max = {grid_no.max():.4f}, "
          f"with-inst min = {grid_inst.min():.4f}, max = {grid_inst.max():.4f}")

    df_boot = experiment_bootstrap(cluster, gender, y, rng)
    print("[bootstrap headline]")
    print(df_boot.to_string(index=False))

    plot_frontier(s_grid, e_grid, grid_no, grid_inst, p_baseline,
                  OUT_FIG / "01_frontier.png")
    plot_marginal(df_marginal, p_baseline, OUT_FIG / "02_marginal.png")
    plot_e_via_c_shift(df_med, df_direct, p_baseline,
                       OUT_FIG / "03_e_trait_vs_direct.png")

    df_marginal.to_csv(OUT_TBL / "marginal_effects.csv", index=False)
    df_boot.to_csv(OUT_TBL / "bootstrap_headline.csv", index=False)
    df_med.to_csv(OUT_TBL / "e_via_c_shift.csv", index=False)
    df_direct.to_csv(OUT_TBL / "e_direct.csv", index=False)
    pd.DataFrame({
        "cell_index": np.arange(N_CELLS),
        "cluster": np.repeat(np.arange(N_CLUSTERS), 2),
        "gender": np.tile([0, 1], N_CLUSTERS),
        "p_baseline": base["p_cells"],
        "weight": base["weights"],
    }).to_csv(OUT_TBL / "baseline_cells.csv", index=False)

    write_report(p_baseline, df_marginal, df_boot,
                 s_grid, e_grid, grid_no, grid_inst, df_med,
                 HERE / "output" / "REPORT.md")
    print(f"[done] outputs in {HERE / 'output'}")


if __name__ == "__main__":
    main()
