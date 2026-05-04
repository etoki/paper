# v4: 4-axis model (制度 C, 技術 T, 資源不足 D, 能力不足 A)

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
                               scenario  effect_C      T  uptake      s      a  P_point  P_lo95  P_hi95
                               baseline    0.0000 0.0000  0.0000 0.0000 0.0000   0.1723  0.1328  0.2147
                                 C only    0.2000 0.0000  0.0000 0.0000 0.0000   0.1379  0.1062  0.1695
                   T only (uptake=0.20)    0.0000 0.2500  0.2000 0.0000 0.0000   0.1637  0.1288  0.2040
                                  C + T    0.2000 0.2500  0.2000 0.0000 0.0000   0.1310  0.1009  0.1632
                   C + T + scarcity 0.5    0.2000 0.2500  0.2000 0.5000 0.0000   0.1506  0.1160  0.1852
            C + T + ability deficit 0.5    0.2000 0.2500  0.2000 0.0000 0.5000   0.1454  0.1144  0.1811
                        C + T + s=a=0.5    0.2000 0.2500  0.2000 0.5000 0.5000   0.1672  0.1288  0.2055
                     C + T(uptake=0.40)    0.2000 0.2500  0.4000 0.0000 0.0000   0.1241  0.0976  0.1525
           C + T(uptake=0.40) + s=a=0.5    0.2000 0.2500  0.4000 0.5000 0.5000   0.1584  0.1220  0.1947
C + low ability (a=1.0) + high T uptake    0.2000 0.2500  0.4000 0.0000 1.0000   0.1514  0.1166  0.1886
      C + high ability (a=0.0) + zero T    0.2000 0.0000  0.0000 0.0000 0.0000   0.1379  0.1062  0.1695
                  all bad: s=a=1.0, T=0    0.2000 0.0000  0.0000 1.0000 1.0000   0.2186  0.1720  0.2688
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
     T  ability_deficit  uptake      P
0.1000           0.0000  0.0000 0.1585
0.1000           0.0000  0.2000 0.1554
0.1000           0.0000  0.4000 0.1522
0.1000           0.5000  0.0000 0.1760
0.1000           0.5000  0.2000 0.1725
0.1000           0.5000  0.4000 0.1689
0.1000           1.0000  0.0000 0.1934
0.1000           1.0000  0.2000 0.1895
0.1000           1.0000  0.4000 0.1857
0.2500           0.0000  0.0000 0.1585
0.2500           0.0000  0.2000 0.1506
0.2500           0.0000  0.4000 0.1427
0.2500           0.5000  0.0000 0.1760
0.2500           0.5000  0.2000 0.1672
0.2500           0.5000  0.4000 0.1584
0.2500           1.0000  0.0000 0.1934
0.2500           1.0000  0.2000 0.1837
0.2500           1.0000  0.4000 0.1741
0.4000           0.0000  0.0000 0.1585
0.4000           0.0000  0.2000 0.1458
0.4000           0.0000  0.4000 0.1332
0.4000           0.5000  0.0000 0.1760
0.4000           0.5000  0.2000 0.1619
0.4000           0.5000  0.4000 0.1478
0.4000           1.0000  0.0000 0.1934
0.4000           1.0000  0.2000 0.1779
0.4000           1.0000  0.4000 0.1625
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
