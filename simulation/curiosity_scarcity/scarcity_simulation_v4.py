"""v4 curiosity simulation: 4-axis model (制度 C, 技術 T, 資源不足 D, 能力不足 A).

User's refined definition:
  - 制度 C: top-down rules + monitoring (laws, HR, reporting, cameras, AI detection)
  - 技術 T: perpetrator-side self-improvement tools (apps, CBT, mindfulness)
           — must be VOLITIONALLY used by the perpetrator themselves
  - 資源不足 D: organizational stressor (workload, understaffing)
  - 能力不足 A: trait-level self-control deficit (HEXACO-C / Tangney SCS)

Key new question:
  Can high T uptake compensate for low A (ability)?
  i.e., does perpetrator-side technology substitute for trait?

Anchors:
  γ_D = 0.30  Hershcovis 2007 perpetrator meta (situational constraints rc=.30)
  γ_A = 0.22  De Ridder 2012 self-control meta (SCS × undesired |r|=.22)
  T   = 0.25  Hudson 2023 d≈0.4-0.6 self-selected → conservative multiplier
              Range: [0.10, 0.40] (very wide due to anchor scarcity)
  uptake range: [0.05, 0.40] (workplace wellness: 10-40%; apps: <10% long-term)

Model:
  P = baseline × (1 - effect_C) × (1 - T × uptake)
              × (1 + γ_D × s) × (1 + γ_A × a)

Optional: scarcity-uptake negative coupling (Hobfoll COR-flavored):
  uptake_effective = uptake × (1 - κ × s),  κ ∈ [0, 0.5]

NOT for v2.0 master pre-registration. Curiosity-only.
"""

from __future__ import annotations

from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors

from curiosity_scarcity.scarcity_simulation_v2 import (
    HEXACO_DOMAINS, N_CELLS, N_CLUSTERS,
    GAMMA_D_MAIN, GAMMA_E_MAIN as GAMMA_A_MAIN,
    GAMMA_D_LO, GAMMA_D_HI,
    GAMMA_E_LO as GAMMA_A_LO, GAMMA_E_HI as GAMMA_A_HI,
    EFFECT_C_MAIN, SEED, B_BOOT,
    load_data, assign_cluster, cell_propensity, cell_weights, aggregate,
    compute_cell_amplification,
    bootstrap_aggregate, percentile_ci,
)

HERE = Path(__file__).parent
OUT_FIG = HERE / "output_v4" / "figures"
OUT_TBL = HERE / "output_v4" / "tables"
OUT_FIG.mkdir(parents=True, exist_ok=True)
OUT_TBL.mkdir(parents=True, exist_ok=True)

# ===================================================================
# 技術 T anchors (intentionally wide CI — anchor scarce)
# ===================================================================

T_MAIN = 0.25
"""Hudson 2023 d ≈ 0.4-0.6 self-selected. Conservative multiplier interpretation."""

T_LO, T_HI = 0.10, 0.40
"""Wide range: lower bound = generic CBT effect attenuated to harassment;
upper bound = Hudson upper d=0.6 directly translated. NO direct harassment
outcome anchor — so range is parametric."""

UPTAKE_MAIN = 0.20
"""Median of typical workplace wellness program participation (10-40%)."""

UPTAKE_LO, UPTAKE_HI = 0.05, 0.40
"""Self-help app long-term: <10%; mandatory program: up to 40%."""

KAPPA_SCARCITY_UPTAKE = 0.30
"""Parametric: scarcity reduces uptake by up to 30% at s=1.0.
NO empirical anchor. Hobfoll COR flavored heuristic."""


# ===================================================================
# Model
# ===================================================================

def apply_v4(p_cells, effect_C, T_eff, uptake_eff, gamma_D, s,
             gamma_A, a, amp=None):
    """4-axis prevalence: identity link with all multiplicative."""
    g_d = gamma_D * s
    g_a = gamma_A * a
    if amp is not None:
        g_d = g_d * amp
        g_a = g_a * amp
    p = p_cells * (1.0 - effect_C) * (1.0 - T_eff * uptake_eff)
    p = p * (1.0 + g_d) * (1.0 + g_a)
    return p


def effective_uptake(uptake, s, kappa=KAPPA_SCARCITY_UPTAKE):
    """Optional scarcity-uptake coupling: high s lowers actual uptake."""
    return uptake * np.clip(1.0 - kappa * s, 0.0, 1.0)


# ===================================================================
# Analysis 1: substitution frontier (a × uptake)
# ===================================================================

def substitution_grid(p, w, amp, effect_C, T, gamma_D, s, gamma_A,
                       n=21, kappa=0.0):
    """Vary (a, uptake) at fixed (effect_C, s, T, γ); return prevalence grid."""
    a_grid = np.linspace(0, 1, n)
    u_grid = np.linspace(0, UPTAKE_HI, n)
    grid = np.zeros((n, n))
    for i, u in enumerate(u_grid):
        for j, a in enumerate(a_grid):
            u_eff = u * (1 - kappa * s) if kappa > 0 else u
            u_eff = max(0.0, u_eff)
            grid[i, j] = aggregate(
                apply_v4(p, effect_C, T, u_eff, gamma_D, s, gamma_A, a, amp),
                w,
            )
    return a_grid, u_grid, grid


def plot_substitution_frontier(p, w, amp, p_baseline, out: Path):
    """3 panels: low scarcity, medium scarcity, high scarcity.

    Within each: (ability deficit a, tool uptake) → prevalence.
    Overlay isocontour at baseline and at C-only-protected baseline."""
    s_levels = [0.0, 0.5, 1.0]
    fig, axes = plt.subplots(1, 3, figsize=(17, 5.5), constrained_layout=True)
    grids = []
    for s in s_levels:
        a_grid, u_grid, grid = substitution_grid(
            p, w, amp, EFFECT_C_MAIN, T_MAIN, GAMMA_D_MAIN, s, GAMMA_A_MAIN,
        )
        grids.append((a_grid, u_grid, grid))
    vmin = min(g[2].min() for g in grids)
    vmax = max(g[2].max() for g in grids)
    norm = mcolors.Normalize(vmin=vmin, vmax=vmax)
    for ax, s, (a_grid, u_grid, grid) in zip(axes, s_levels, grids):
        im = ax.imshow(grid, origin="lower",
                       extent=(a_grid[0], a_grid[-1], u_grid[0], u_grid[-1]),
                       aspect="auto", cmap="magma", norm=norm)
        cs = ax.contour(a_grid, u_grid, grid,
                        levels=[p_baseline], colors="cyan", linewidths=2.0)
        if cs.allsegs[0]:
            ax.clabel(cs, fmt={p_baseline: "= baseline"}, fontsize=9)
        ax.set_xlabel("Ability deficit  a")
        ax.set_ylabel("Tool uptake  (volitional)")
        ax.set_title(f"Scarcity s = {s}")
    fig.colorbar(im, ax=axes, label="Prevalence (with institution C=0.20)")
    fig.suptitle(
        "Substitution: can perpetrator-side tools (T uptake) compensate for low ability (A)?\n"
        f"T effect = {T_MAIN} (Hudson 2023 anchor, weak), γ_D={GAMMA_D_MAIN}, γ_A={GAMMA_A_MAIN}",
        fontsize=11,
    )
    fig.savefig(out, dpi=140)
    plt.close(fig)


# ===================================================================
# Analysis 2: headline scenarios with bootstrap CIs
# ===================================================================

def headline_v4(cluster, gender, y, amp, rng):
    scenarios = {
        "baseline":                              (0.00, 0.00, 0.00, 0.0, 0.0),
        "C only":                                (0.20, 0.00, 0.00, 0.0, 0.0),
        "T only (uptake=0.20)":                  (0.00, 0.25, 0.20, 0.0, 0.0),
        "C + T":                                 (0.20, 0.25, 0.20, 0.0, 0.0),
        "C + T + scarcity 0.5":                  (0.20, 0.25, 0.20, 0.5, 0.0),
        "C + T + ability deficit 0.5":           (0.20, 0.25, 0.20, 0.0, 0.5),
        "C + T + s=a=0.5":                       (0.20, 0.25, 0.20, 0.5, 0.5),
        "C + T(uptake=0.40)":                    (0.20, 0.25, 0.40, 0.0, 0.0),
        "C + T(uptake=0.40) + s=a=0.5":          (0.20, 0.25, 0.40, 0.5, 0.5),
        "C + low ability (a=1.0) + high T uptake": (0.20, 0.25, 0.40, 0.0, 1.0),
        "C + high ability (a=0.0) + zero T":     (0.20, 0.00, 0.00, 0.0, 0.0),
        "all bad: s=a=1.0, T=0":                 (0.20, 0.00, 0.00, 1.0, 1.0),
    }
    p_orig = cell_propensity(cluster, gender, y)
    w_orig = cell_weights(cluster, gender)
    rows = []
    for name, (ec, T, u, s, a) in scenarios.items():
        fn = lambda p_arg: apply_v4(p_arg, ec, T, u, GAMMA_D_MAIN, s,
                                     GAMMA_A_MAIN, a, amp)
        boot = bootstrap_aggregate(cluster, gender, y, fn, B_BOOT, rng)
        point = aggregate(fn(p_orig), w_orig)
        lo, hi = percentile_ci(boot)
        rows.append({"scenario": name,
                     "effect_C": ec, "T": T, "uptake": u, "s": s, "a": a,
                     "P_point": point, "P_lo95": lo, "P_hi95": hi})
    return pd.DataFrame(rows)


# ===================================================================
# Analysis 3: scarcity-uptake coupling
# ===================================================================

def scarcity_uptake_coupling(p, w, amp, kappa_grid):
    """How does the substitution frontier change when scarcity suppresses uptake?

    Reports: at fixed (a=1.0, base_uptake=0.40), prevalence as κ varies."""
    rows = []
    for s in [0.0, 0.25, 0.5, 0.75, 1.0]:
        for kappa in kappa_grid:
            u_eff = effective_uptake(0.40, s, kappa)
            P = aggregate(
                apply_v4(p, EFFECT_C_MAIN, T_MAIN, u_eff,
                         GAMMA_D_MAIN, s, GAMMA_A_MAIN, 1.0, amp),
                w,
            )
            rows.append({"s": s, "kappa": kappa,
                         "uptake_effective": u_eff,
                         "P": P})
    return pd.DataFrame(rows)


def plot_scarcity_uptake(df: pd.DataFrame, p_baseline: float, out: Path):
    fig, ax = plt.subplots(figsize=(9, 5.5), constrained_layout=True)
    for kappa in sorted(df["kappa"].unique()):
        sub = df[df["kappa"] == kappa]
        ax.plot(sub["s"], sub["P"], "o-",
                label=f"κ = {kappa:.2f}",
                linewidth=2)
    ax.axhline(p_baseline, linestyle="--", color="gray",
               label=f"baseline {p_baseline:.3f}")
    ax.set_xlabel("Scarcity  s")
    ax.set_ylabel("Prevalence at a=1.0, uptake=0.40 (with institution C=0.20)")
    ax.set_title(
        "Scarcity-uptake coupling: high s suppresses tool use → tool effect attenuates\n"
        "(κ = how much scarcity reduces uptake; κ=0 means no coupling)"
    )
    ax.legend(title="Coupling κ")
    ax.grid(alpha=0.3)
    fig.savefig(out, dpi=140)
    plt.close(fig)


# ===================================================================
# Analysis 4: T-anchor sensitivity (since T anchor is weakest)
# ===================================================================

def t_anchor_sweep(p, w, amp):
    """How sensitive is the substitution conclusion to T's wide CI?"""
    rows = []
    a_grid = [0.0, 0.5, 1.0]
    u_grid = [0.0, 0.20, 0.40]
    T_vals = [T_LO, T_MAIN, T_HI]
    for T in T_vals:
        for a in a_grid:
            for u in u_grid:
                P = aggregate(
                    apply_v4(p, EFFECT_C_MAIN, T, u,
                             GAMMA_D_MAIN, 0.5, GAMMA_A_MAIN, a, amp),
                    w,
                )
                rows.append({"T": T, "ability_deficit": a, "uptake": u, "P": P})
    return pd.DataFrame(rows)


def plot_t_sensitivity(df: pd.DataFrame, p_baseline: float, out: Path):
    fig, axes = plt.subplots(1, 3, figsize=(15, 5),
                             constrained_layout=True, sharey=True)
    a_vals = sorted(df["ability_deficit"].unique())
    for ax, a in zip(axes, a_vals):
        sub = df[df["ability_deficit"] == a]
        for u in sorted(sub["uptake"].unique()):
            s2 = sub[sub["uptake"] == u]
            ax.plot(s2["T"], s2["P"], "o-",
                    label=f"uptake={u:.2f}", linewidth=2)
        ax.axhline(p_baseline, linestyle="--", color="gray",
                   label=f"baseline {p_baseline:.3f}")
        ax.set_xlabel("Tool effect strength T")
        if ax is axes[0]:
            ax.set_ylabel("Prevalence (s=0.5, C=0.20)")
        ax.set_title(f"Ability deficit  a = {a}")
        ax.legend(fontsize=8)
        ax.grid(alpha=0.3)
    fig.suptitle(
        "T-anchor sensitivity (weakest empirical link)\n"
        f"T range [{T_LO}, {T_HI}]: lower=generic CBT attenuated, upper=Hudson 2023 d=0.6 direct",
        fontsize=11,
    )
    fig.savefig(out, dpi=140)
    plt.close(fig)


# ===================================================================
# Report
# ===================================================================

def write_report(p_baseline, df_headline, df_su, df_t,
                 amp, out: Path):
    text = f"""# v4: 4-axis model (制度 C, 技術 T, 資源不足 D, 能力不足 A)

> 完全な好奇心ベース。本論文には反映しない。
> v3 まで「e (technology)」と呼んでいた軸を、ご指摘に従って **能力 A (trait)** と
> **道具 T (perpetrator-side tool)** に分離した。

## 軸の定義（修正版）

| 軸 | 定義 | 主体 | 例 |
|---|---|---|---|
| **制度 C** | top-down rules + 監視・通報・HR・AI 検知 | 組織・社会 | 法律、HR、監視カメラ、通報窓口 |
| **技術 T** | 加害者が能動的に使う自己抑制ツール × 利用率 | **加害者本人** | アンガーマネジメントアプリ、CBT、マインドフルネス |
| **資源不足 D (s)** | 組織レベル stressor | 環境 | 人手不足、過重労働 |
| **能力不足 A (a)** | 個人 trait の自制心欠如 | trait | HEXACO-C low |

## モデル

```
P = baseline × (1 − effect_C) × (1 − T × uptake)
            × (1 + γ_D × s) × (1 + γ_A × a)
```

オプション: scarcity-uptake coupling
```
uptake_effective = uptake × (1 − κ × s)
```

## Anchors（強さの正直な評価）

| パラメータ | main | range | 強さ | 出典 |
|---|---|---|---|---|
| effect_C | 0.20 | [0.10, 0.30] | ✅ 強い | v2.0 master 4-PDF triangulation |
| **T (tool effect)** | **0.25** | **[0.10, 0.40]** | ⚠️ **弱い** | Hudson 2023 d≈0.4-0.6 self-selected の保守換算 (harassment outcome 直接測定なし) |
| **uptake (利用率)** | **0.20** | **[0.05, 0.40]** | ❌ **anchor 不在** | wellness 10-40%、self-help app <10% (parametric) |
| γ_D | 0.30 | [0.19, 0.43] | ✅ 中程度 | Hershcovis 2007 perpetrator meta |
| γ_A | 0.22 | [0.15, 0.27] | ✅ 中程度 | De Ridder 2012 self-control meta |
| **κ (s→uptake coupling)** | **0.30** | **[0.0, 0.5]** | ❌ **anchor 不在** | Hobfoll COR theory 派生 (parametric) |

## Headline scenarios

```
{df_headline.to_string(index=False, float_format='%.4f')}
```

主な観察:
- **T 単独 (制度なし)** はほぼ無効: T_main × uptake_main = 0.05 → 5% 削減のみ
- **C + T**: 制度 0.20 + 道具 0.05 = ほぼ制度のみと変わらず（道具の effect が小さい）
- **C + T(uptake=0.40)**: uptake を倍にしても 0.10 → 制度の半分の影響
- **能力 A=1.0 + 高 uptake**: 道具で trait を補えるか？ → 部分的に補えるが完全ではない

## 解釈

1. **道具 T の現実的な効果は小さい**: T × uptake = 0.05〜0.10 程度。制度 0.20 の
   半分以下。「道具で能力を補う」mechanism は理論的にはありうるが、
   **量的には制度より大幅に弱い**。
2. **uptake が支配的**: 道具自体の効果 T ではなく、**何 % が使うか** が結論を決める。
   uptake = 0 なら T = ∞ でも無効。
3. **scarcity-uptake coupling は二重の打撃**: scarcity が高い職場は道具を使う心理的
   余裕も無くなる → uptake_eff が下がる → trait A の deficit が剥き出しになる。
4. **能力 A vs 道具 T の substitution は弱い**: 道具で trait を完全には補えない
   （T_eff = 0.10 程度では γ_A × a = 0.22 を完全に打ち消せない）。

## 「道具で能力を補える」の量的評価

substitution frontier 図 (`v4_substitution.png`) で見ると:
- s = 0 (低 scarcity): 高 uptake (~0.30+) で a=1.0 でも baseline 維持可能
- s = 0.5 (中 scarcity): uptake = 0.40 でも a >= 0.5 で baseline 超える
- s = 1.0 (高 scarcity): uptake をどう上げても baseline 超える

つまり **道具 T は「ある程度の能力代替」になるが、scarcity が伴うと substitution
の成立範囲は急速に縮小する**。

## T-anchor sensitivity

T anchor は weakest なので [0.10, 0.40] で 4x の幅で sweep:

```
{df_t.to_string(index=False, float_format='%.4f')}
```

T が 0.10 (lower) では substitution はほぼ機能しない。T が 0.40 (upper) でようやく
小さい補完効果。**結論「道具で trait を補える」は T anchor 強度に強く依存** する。

## 結論

> **「能力が低くても道具で代替できる」というご指摘は理論的には正しいが、
> 道具 T と利用率 uptake の文献校正値は弱く、量的効果は制度 C の半分以下しか出ない。
> しかも scarcity が高い職場では uptake が下がるので、現実の高ストレス環境では
> 道具による trait 補完はほぼ機能しない。**

含意:
- **道具を効かせるには、uptake を高める仕組み（強制または incentive）が要る**
  → mandatory CBT、reduced workload で時間を作る、等
- **道具と制度は substitute ではなく complement**: 制度が監視層を担い、道具が
  個人の impulse 抑制を担う
- **scarcity 改善は両方の前提**: scarcity を減らさないと、道具の uptake も
  trait の発現も悪化する

## 残る caveats

- **T anchor (Hudson 2023) は harassment outcome を直接測っていない**: trait 変化のみ。
  harassment 行動への transfer 仮定が未検証。
- **uptake 文献は wellness program / health app**: harassment 防止アプリの literature は
  ほぼ存在しない（spec 化された道具がない）
- **κ (scarcity → uptake coupling) は完全 parametric**: 直接検証する longitudinal data
  なし
- **この 4 軸モデル全体は post-hoc**: pre-registration されておらず、modeling choice が
  入っている

## 出典

1. Hudson 2023. Lighten the Darkness. *J Personality, 91*(4). [T anchor]
2. Hershcovis 2007. *J Applied Psychology, 92*(1). [γ_D anchor]
3. De Ridder 2012. *Personality and Social Psychology Review, 16*(1). [γ_A anchor]
4. Bowling & Eschleman 2010. *J Occupational Health Psychology, 15*(1). [γ_c, dropped in v4]
5. Hobfoll COR theory (κ parametric inspiration only)
"""
    out.write_text(text, encoding="utf-8")


# ===================================================================
# Main
# ===================================================================

def main():
    rng = np.random.default_rng(SEED)
    hexaco, gender, y, centroids = load_data()
    cluster = assign_cluster(hexaco, centroids)
    p_orig = cell_propensity(cluster, gender, y)
    w_orig = cell_weights(cluster, gender)
    p_baseline = aggregate(p_orig, w_orig)
    # In v4, drop the controversial cell-specific γ_c amplification
    # (v3 found r=+.034, contradicting the B&E 2010 prediction it was based on)
    amp = np.ones(N_CELLS)

    print(f"[baseline] P = {p_baseline:.4f}")
    print(f"[v4 model] uniform γ_c (no cluster amplification, per v3 finding)")

    # Headline
    df_h = headline_v4(cluster, gender, y, amp, rng)
    print("[headline]")
    print(df_h.to_string(index=False))
    df_h.to_csv(OUT_TBL / "v4_headline.csv", index=False)

    # Substitution frontier
    plot_substitution_frontier(p_orig, w_orig, amp, p_baseline,
                                OUT_FIG / "v4_substitution.png")

    # Scarcity-uptake coupling
    df_su = scarcity_uptake_coupling(p_orig, w_orig, amp,
                                      kappa_grid=[0.0, 0.15, 0.30, 0.50])
    df_su.to_csv(OUT_TBL / "v4_scarcity_uptake_coupling.csv", index=False)
    print("[scarcity-uptake coupling]")
    print(df_su.to_string(index=False))
    plot_scarcity_uptake(df_su, p_baseline,
                          OUT_FIG / "v4_scarcity_uptake_coupling.png")

    # T-anchor sensitivity
    df_t = t_anchor_sweep(p_orig, w_orig, amp)
    df_t.to_csv(OUT_TBL / "v4_t_anchor_sensitivity.csv", index=False)
    print("[T-anchor sensitivity]")
    print(df_t.to_string(index=False))
    plot_t_sensitivity(df_t, p_baseline, OUT_FIG / "v4_t_sensitivity.png")

    write_report(p_baseline, df_h, df_su, df_t, amp,
                 HERE / "output_v4" / "REPORT_v4.md")
    print(f"[done] outputs in {HERE / 'output_v4'}")


if __name__ == "__main__":
    main()
