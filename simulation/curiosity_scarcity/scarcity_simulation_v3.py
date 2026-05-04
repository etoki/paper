"""v3 curiosity simulation: 3 follow-up analyses on top of v2.

Analyses:
  1. N=354 cluster × harassment cross-tab + check Bowling & Eschleman 2010
     prediction: low-C clusters should have higher harassment propensity.
  2. Tornado plot: how much does each assumption (γ_D, γ_E, CMV, link,
     effect_C, cell amplification range, scenario point) shift the
     prevalence at the s=e=0.5 + institution C=0.20 reference scenario?
  3. Time-axis sensitivity: acute / cross-sectional / chronic-prospective
     / Hobfoll-COR-spiral parametric scaling on γ_D.

NOT for v2.0 master pre-registration. Curiosity-only.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Reuse v2 calibration
from curiosity_scarcity.scarcity_simulation_v2 import (
    HEXACO_DOMAINS, HEXACO_COLS, N_CLUSTERS, N_CELLS,
    GAMMA_D_MAIN, GAMMA_D_LO, GAMMA_D_HI,
    GAMMA_E_MAIN, GAMMA_E_LO, GAMMA_E_HI,
    EFFECT_C_MAIN,
    CELL_C_AMPLIFICATION_RANGE,
    LINK_FUNCS,
    SEED, B_BOOT,
    load_data, assign_cluster, cell_propensity, cell_weights, aggregate,
    apply_combined, compute_cell_amplification,
    bootstrap_aggregate, percentile_ci,
)

HERE = Path(__file__).parent
OUT_FIG = HERE / "output_v3" / "figures"
OUT_TBL = HERE / "output_v3" / "tables"
OUT_FIG.mkdir(parents=True, exist_ok=True)
OUT_TBL.mkdir(parents=True, exist_ok=True)


# ===================================================================
# Analysis 1: N=354 cluster × harassment cross-tab
# ===================================================================

def cluster_cross_tab(hexaco, gender, y, centroids, cluster):
    """Direct empirical: cluster's HEXACO-C centroid vs cluster's harassment rate.

    Tests Bowling & Eschleman 2010 prediction: low-C clusters should have
    HIGHER harassment propensity (negative correlation between cluster C
    centroid and cluster harassment rate).
    """
    rows = []
    c_idx = HEXACO_DOMAINS.index("C")
    for cl in range(N_CLUSTERS):
        mask = (cluster == cl)
        n = int(mask.sum())
        if n == 0:
            continue
        p_cluster = float(y[mask].mean())
        # Empirical SE for binomial proportion
        se = float(np.sqrt(p_cluster * (1 - p_cluster) / n))
        rows.append({
            "cluster": cl,
            "centroid_C": float(centroids[cl, c_idx]),
            "n": n,
            "harassment_rate": p_cluster,
            "se": se,
            "ci_lo": max(0.0, p_cluster - 1.96 * se),
            "ci_hi": min(1.0, p_cluster + 1.96 * se),
        })
    df = pd.DataFrame(rows)
    # Pearson correlation (cluster-level, weighted by n)
    from scipy import stats
    r, p_val = stats.pearsonr(df["centroid_C"], df["harassment_rate"])
    weighted = stats.pearsonr(
        df["centroid_C"].values,
        df["harassment_rate"].values,
    )
    return df, r, p_val


def plot_cluster_cross_tab(df: pd.DataFrame, r: float, p_val: float, out: Path):
    fig, ax = plt.subplots(figsize=(8.5, 5.5), constrained_layout=True)
    ax.errorbar(df["centroid_C"], df["harassment_rate"],
                yerr=[df["harassment_rate"] - df["ci_lo"],
                      df["ci_hi"] - df["harassment_rate"]],
                fmt="o", capsize=5, color="tab:blue", markersize=10)
    for _, row in df.iterrows():
        ax.annotate(f"  C{int(row['cluster'])} (n={int(row['n'])})",
                    (row["centroid_C"], row["harassment_rate"]),
                    fontsize=9)
    # Linear regression line
    x = df["centroid_C"].values
    y = df["harassment_rate"].values
    coef = np.polyfit(x, y, 1)
    xs = np.linspace(x.min() - 0.1, x.max() + 0.1, 100)
    ax.plot(xs, np.polyval(coef, xs), "--", color="gray", alpha=0.5,
            label=f"Linear fit: slope={coef[0]:.3f}")
    ax.set_xlabel("Cluster centroid HEXACO-Conscientiousness")
    ax.set_ylabel("Cluster harassment rate (binary)")
    direction = "supports" if r < 0 else "contradicts"
    ax.set_title(
        f"N=354: Cluster centroid C vs harassment rate\n"
        f"Pearson r = {r:+.3f}, p = {p_val:.3f} → {direction} B&E 2010 (predict r < 0)"
    )
    ax.legend()
    ax.grid(alpha=0.3)
    fig.savefig(out, dpi=140)
    plt.close(fig)


# ===================================================================
# Analysis 2: Tornado plot of assumption sensitivity
# ===================================================================

def reference_scenario(p_orig, w_orig, link, gamma_D, gamma_E, effect_C, amp,
                       s=0.5, e=0.5):
    """Compute prevalence at the reference test point (s=e=0.5, C=0.20)."""
    return aggregate(
        apply_combined(p_orig, link, gamma_D, s, gamma_E, e, effect_C, amp),
        w_orig,
    )


def tornado_analysis(p_orig, w_orig, centroids):
    """For each assumption, vary it from min to max and record P_ref change."""
    amp_main = compute_cell_amplification(centroids)
    base = reference_scenario(p_orig, w_orig, "identity",
                              GAMMA_D_MAIN, GAMMA_E_MAIN, EFFECT_C_MAIN,
                              amp_main)

    rows = []
    # γ_D credibility interval
    for label, gd in [("γ_D low (.19)", GAMMA_D_LO),
                      ("γ_D high (.43)", GAMMA_D_HI)]:
        P = reference_scenario(p_orig, w_orig, "identity",
                               gd, GAMMA_E_MAIN, EFFECT_C_MAIN, amp_main)
        rows.append({"variation": label, "P": P, "delta": P - base})
    # γ_E range
    for label, ge in [("γ_E low (.15)", GAMMA_E_LO),
                      ("γ_E high (.27)", GAMMA_E_HI)]:
        P = reference_scenario(p_orig, w_orig, "identity",
                               GAMMA_D_MAIN, ge, EFFECT_C_MAIN, amp_main)
        rows.append({"variation": label, "P": P, "delta": P - base})
    # CMV discount on γ_D
    for label, disc in [("CMV discount 0.85", 0.85),
                        ("CMV discount 0.70", 0.70)]:
        P = reference_scenario(p_orig, w_orig, "identity",
                               GAMMA_D_MAIN * disc, GAMMA_E_MAIN, EFFECT_C_MAIN,
                               amp_main)
        rows.append({"variation": label, "P": P, "delta": P - base})
    # Link function
    for link in ["log_linear", "logit"]:
        P = reference_scenario(p_orig, w_orig, link,
                               GAMMA_D_MAIN, GAMMA_E_MAIN, EFFECT_C_MAIN,
                               amp_main)
        rows.append({"variation": f"link={link}", "P": P, "delta": P - base})
    # effect_C range
    for label, ec in [("effect_C 0.10", 0.10),
                      ("effect_C 0.30", 0.30)]:
        P = reference_scenario(p_orig, w_orig, "identity",
                               GAMMA_D_MAIN, GAMMA_E_MAIN, ec, amp_main)
        rows.append({"variation": label, "P": P, "delta": P - base})
    # Cell amplification range
    amp_uniform = np.ones(N_CELLS)
    P = reference_scenario(p_orig, w_orig, "identity",
                           GAMMA_D_MAIN, GAMMA_E_MAIN, EFFECT_C_MAIN, amp_uniform)
    rows.append({"variation": "γ_c uniform (no amp)", "P": P, "delta": P - base})
    amp_strong = compute_cell_amplification(centroids)
    # stronger spread (0.5, 2.0)
    c_idx = HEXACO_DOMAINS.index("C")
    c_vals = centroids[:, c_idx]
    c_min, c_max = c_vals.min(), c_vals.max()
    cluster_amp_strong = 2.0 - (c_vals - c_min) / (c_max - c_min) * (2.0 - 0.5)
    amp_strong = np.repeat(cluster_amp_strong, 2)
    P = reference_scenario(p_orig, w_orig, "identity",
                           GAMMA_D_MAIN, GAMMA_E_MAIN, EFFECT_C_MAIN, amp_strong)
    rows.append({"variation": "γ_c spread (0.5, 2.0)", "P": P, "delta": P - base})

    df = pd.DataFrame(rows)
    df["abs_delta"] = df["delta"].abs()
    df = df.sort_values("abs_delta", ascending=True)
    return df, base


def plot_tornado(df: pd.DataFrame, base: float, out: Path):
    fig, ax = plt.subplots(figsize=(11, 6.5), constrained_layout=True)
    colors = ["tab:red" if d > 0 else "tab:blue" for d in df["delta"]]
    ax.barh(df["variation"], df["delta"], color=colors, alpha=0.75)
    ax.axvline(0, color="black", linewidth=0.5)
    for i, (_, row) in enumerate(df.iterrows()):
        ax.text(row["delta"] + (0.001 if row["delta"] >= 0 else -0.001),
                i,
                f"{row['P']:.3f}",
                va="center",
                ha="left" if row["delta"] >= 0 else "right",
                fontsize=9)
    ax.set_xlabel(f"Δ prevalence from base scenario (s=e=0.5, C=0.20, base P={base:.3f})")
    ax.set_title("Tornado plot: which assumption shifts the conclusion most?")
    ax.grid(alpha=0.3, axis="x")
    fig.savefig(out, dpi=140)
    plt.close(fig)


# ===================================================================
# Analysis 3: Time-axis sensitivity
# ===================================================================

# De Ridder 2012 explicitly reports cross-sectional vs prospective:
#   Cross-sectional: r = -.23 (k=32, dominant in literature)
#   Prospective:     r = -.14 (k=8, attenuated)
# Ratio: prospective / cross-sect = .14 / .23 ≈ 0.61
DE_RIDDER_PROSPECTIVE_RATIO = 0.14 / 0.23

# Hobfoll COR theory: chronic loss spirals predict ~1.4x amplification
# (parametric assumption; no direct meta-analytic anchor)
COR_SPIRAL_FACTOR = 1.4

TIME_REGIMES = {
    "acute (1-day spike)":           0.5,   # heuristic: single event << sustained
    "cross-sectional (Hershcovis)":  1.0,   # baseline
    "prospective (De Ridder)":       DE_RIDDER_PROSPECTIVE_RATIO,
    "Hobfoll COR spiral":            COR_SPIRAL_FACTOR,
}


def time_axis_sensitivity(p_orig, w_orig, centroids):
    """How does the institution-defeat frontier shift across time regimes?"""
    amp = compute_cell_amplification(centroids)
    rows = []
    for regime, factor in TIME_REGIMES.items():
        gd = GAMMA_D_MAIN * factor
        # Scan (s, e) for institution-defeat threshold
        broken_sum = None
        broken_s, broken_e = np.nan, np.nan
        for s in np.linspace(0, 1, 41):
            for e in np.linspace(0, 1, 41):
                P = aggregate(
                    apply_combined(p_orig, "identity", gd, s,
                                   GAMMA_E_MAIN, e, EFFECT_C_MAIN, amp),
                    w_orig,
                )
                p_baseline = aggregate(p_orig, w_orig)
                if P > p_baseline:
                    if broken_sum is None or s + e < broken_sum:
                        broken_sum = s + e
                        broken_s, broken_e = s, e
        # Reference scenario: P at s=e=0.5 with institution
        P_ref = aggregate(
            apply_combined(p_orig, "identity", gd, 0.5,
                           GAMMA_E_MAIN, 0.5, EFFECT_C_MAIN, amp),
            w_orig,
        )
        # Worst-case: (s=e=1.0)
        P_extreme = aggregate(
            apply_combined(p_orig, "identity", gd, 1.0,
                           GAMMA_E_MAIN, 1.0, EFFECT_C_MAIN, amp),
            w_orig,
        )
        rows.append({
            "regime": regime,
            "scaling_factor": factor,
            "gamma_D_effective": gd,
            "frontier_s": broken_s,
            "frontier_e": broken_e,
            "frontier_sum": broken_sum if broken_sum is not None else 2.0,
            "P_ref(s=e=0.5)": P_ref,
            "P_extreme(s=e=1.0)": P_extreme,
            "p_baseline": float(p_baseline),
        })
    return pd.DataFrame(rows)


def plot_time_axis(df: pd.DataFrame, p_baseline: float, out: Path):
    fig, axes = plt.subplots(1, 2, figsize=(13, 5.5), constrained_layout=True)

    ax = axes[0]
    regimes = df["regime"].tolist()
    factors = df["scaling_factor"].values
    p_ref = df["P_ref(s=e=0.5)"].values
    p_ext = df["P_extreme(s=e=1.0)"].values
    x = np.arange(len(regimes))
    w = 0.35
    ax.bar(x - w / 2, p_ref, w, label="s=e=0.5 (moderate)", color="tab:blue")
    ax.bar(x + w / 2, p_ext, w, label="s=e=1.0 (extreme)", color="tab:red")
    ax.axhline(p_baseline, linestyle="--", color="gray", label=f"baseline {p_baseline:.3f}")
    ax.set_xticks(x)
    ax.set_xticklabels(regimes, rotation=20, ha="right", fontsize=9)
    ax.set_ylabel("Prevalence (with institution C=0.20)")
    ax.set_title("Time-regime effect on prevalence (γ_D scaled by regime)")
    ax.legend()
    ax.grid(alpha=0.3, axis="y")

    ax = axes[1]
    fsum = df["frontier_sum"].values
    colors = ["tab:green" if f >= 1.5 else "tab:orange" if f >= 0.8 else "tab:red"
              for f in fsum]
    ax.barh(regimes, fsum, color=colors)
    for i, (val, factor) in enumerate(zip(fsum, factors)):
        ax.text(val + 0.02, i, f"{val:.2f} (γ_D × {factor:.2f})",
                va="center", fontsize=9)
    ax.set_xlabel("Frontier sum (small = institution defeated faster)")
    ax.set_title("Time regime → institution defeat threshold")
    ax.set_xlim(0, 2.2)
    ax.grid(alpha=0.3, axis="x")

    fig.savefig(out, dpi=140)
    plt.close(fig)


# ===================================================================
# Report
# ===================================================================

def write_report(df_cluster, r_cluster, p_cluster_val,
                 df_tornado, base_tornado,
                 df_time, p_baseline, out: Path):
    text = f"""# v3: 3 follow-up analyses

> 完全な好奇心ベース。本論文には反映しない。v2 校正の上に 3 つの追加診断を載せた。

---

## Analysis 1: N=354 cluster × harassment cross-tab

Bowling & Eschleman 2010 (N=726) は「Conscientiousness が低い人ほど stressor →
CWB の relation が強い」を 6/6 interaction 全部で示した。本データの cluster
レベルでも同じ pattern が見えるかを直接検証。

```
{df_cluster.to_string(index=False, float_format='%.4f')}
```

**Pearson r (cluster centroid C, cluster harassment rate) = {r_cluster:+.3f}, p = {p_cluster_val:.3f}**

解釈:
- B&E 2010 予測: r < 0（低-C クラスタほど高 harassment 率）
- 本データ: r = {r_cluster:+.3f} → {'予測支持' if r_cluster < 0 else '予測不支持'}

注：本データの r は cluster-level (k=7) なので power が低い。CI は wide。

---

## Analysis 2: Tornado plot

Reference 点: s = e = 0.5、effect_C = 0.20、identity link、empirical γ。
P_base = {base_tornado:.4f}

各仮定を CI / range の両端に振ったときの prevalence 変化:

```
{df_tornado.to_string(index=False, float_format='%.4f')}
```

**結論ドライバ ranking** (abs_delta 大きい順):
"""
    for _, row in df_tornado.sort_values("abs_delta", ascending=False).head(5).iterrows():
        text += f"  {row['variation']}: Δ = {row['delta']:+.4f}\n"

    text += f"""
解釈:
- **最も結論を動かす仮定**: 上位 3 つを見る → どれかが大きく effect 持っているなら
  そこに reference data を投資すべき
- **小さい仮定**: ほぼ無視できる、modeling choice として transparent に処理可

---

## Analysis 3: Time-axis sensitivity

γ_D を時間 regime で scaling した場合の institution-defeat threshold:

```
{df_time.to_string(index=False, float_format='%.4f')}
```

**4 regime の経験的 / 理論的根拠**:
1. **acute (× 0.5)**: 単発 stressor — heuristic（直接 anchor 無し）
2. **cross-sectional (× 1.0)**: Hershcovis 2007 / De Ridder 2012 等の主流 design
3. **prospective (× {DE_RIDDER_PROSPECTIVE_RATIO:.2f})**: De Ridder 2012 直接 anchor
   (cross-sect r=-.23 vs prospective r=-.14, 比率 0.61)
4. **Hobfoll COR spiral (× {COR_SPIRAL_FACTOR})**: 慢性 stressor の累積 loss spiral
   理論予測（empirical anchor 無し）

**重要 finding（contradiction）**:
- 経験的に時間が経つと effect は **減衰** する (De Ridder prospective)
- 理論的には Hobfoll COR は **増幅** を予測する
- これら 2 つは反対方向 → どっちが正しいか文献では決着ついていない

frontier_sum で見ると:
- prospective regime: institution が **広い範囲で hold** (frontier_sum =
  {df_time[df_time['regime']=='prospective (De Ridder)']['frontier_sum'].iloc[0]:.2f})
- COR spiral regime: institution が **早く defeat** される (frontier_sum =
  {df_time[df_time['regime']=='Hobfoll COR spiral']['frontier_sum'].iloc[0]:.2f})

---

## まとめ

| Analysis | 主な finding |
|---|---|
| 1. Cluster cross-tab | r = {r_cluster:+.3f} で B&E 予測 {'支持' if r_cluster < 0 else '不支持'}。本データの cluster 構造は personality × stressor interaction の cell-level mapping を {'部分的に正当化' if r_cluster < 0 else '正当化しない'} |
| 2. Tornado | 最も結論を動かす仮定: {df_tornado.sort_values("abs_delta", ascending=False).iloc[0]['variation']} (Δ = {df_tornado.sort_values("abs_delta", ascending=False).iloc[0]['delta']:+.4f}) |
| 3. Time axis | 経験 (De Ridder) と理論 (Hobfoll) で逆方向の予測 → 時間軸の校正には longitudinal stressor-aggression データが必要 |

## 残る重要 caveats

- **本データの cluster 数 = 7** で cluster-level Pearson r の power が低い
- **時間軸の anchor は De Ridder の self-control prospective のみ** で、
  stressor の longitudinal anchor は本 5-PDF set に含まれない
- **COR spiral factor 1.4 は parametric assumption**、empirical anchor 無し

## さらなる diagnostic（PDF 不要）

- N=354 cluster × stressor (本データに workload measure が無いので不可)
- 7-cluster 内での Cluster 0 と Cluster 5 の direct comparison（Tier 1 finding：
  Cluster 0 が high-X × low-HH なら harassment 高位、Cluster 5 が低 C なのに
  低 harassment ならば B&E 予測 contradict）
"""
    out.write_text(text, encoding="utf-8")


def main():
    rng = np.random.default_rng(SEED)
    hexaco, gender, y, centroids = load_data()
    cluster = assign_cluster(hexaco, centroids)
    p_orig = cell_propensity(cluster, gender, y)
    w_orig = cell_weights(cluster, gender)
    p_baseline = aggregate(p_orig, w_orig)

    print(f"[baseline] P = {p_baseline:.4f}")

    # Analysis 1
    df_cluster, r_cluster, p_val = cluster_cross_tab(
        hexaco, gender, y, centroids, cluster
    )
    print("[cluster cross-tab]")
    print(df_cluster.to_string(index=False))
    print(f"  Pearson r(C centroid, harassment rate) = {r_cluster:+.3f}, p = {p_val:.3f}")
    df_cluster.to_csv(OUT_TBL / "v3_cluster_cross_tab.csv", index=False)
    plot_cluster_cross_tab(df_cluster, r_cluster, p_val,
                           OUT_FIG / "v3_cluster_cross_tab.png")

    # Analysis 2
    df_tornado, base_t = tornado_analysis(p_orig, w_orig, centroids)
    print("[tornado]")
    print(df_tornado.sort_values("abs_delta", ascending=False).to_string(index=False))
    df_tornado.to_csv(OUT_TBL / "v3_tornado.csv", index=False)
    plot_tornado(df_tornado, base_t, OUT_FIG / "v3_tornado.png")

    # Analysis 3
    df_time = time_axis_sensitivity(p_orig, w_orig, centroids)
    print("[time-axis]")
    print(df_time.to_string(index=False))
    df_time.to_csv(OUT_TBL / "v3_time_axis.csv", index=False)
    plot_time_axis(df_time, p_baseline, OUT_FIG / "v3_time_axis.png")

    write_report(df_cluster, r_cluster, p_val,
                 df_tornado, base_t,
                 df_time, p_baseline,
                 HERE / "output_v3" / "REPORT_v3.md")
    print(f"[done] outputs in {HERE / 'output_v3'}")


if __name__ == "__main__":
    main()
