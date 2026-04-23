# Deep Reading Notes: Big Five × Online Learning × Academic Achievement

本ドキュメントは本メタ分析対象論文の精読ノート。Introduction 執筆と Methods の Risk-of-Bias 評価に必要な情報を、PDF 直接参照で構造化記録する。

`literature_review.md` が narrative 要約であるのに対し、本ファイルは **効果量・方法論・質評価の抽出** を目的とする。

---

## 凡例

- **r**, **β**, **d**, **η²**, **F**, **t**: 効果量・検定統計量
- **N**: サンプルサイズ（analytic N）
- **α**: Cronbach's alpha（信頼性）
- **ρ**: 母相関（メタ分析の pooled 値）
- **CI95**: 95% 信頼区間
- **RoB**: Risk of Bias 8 項目（JBI）スコア（0–8, 高い方が低リスク）
- ⚠ **要追加確認**: 著者名・書誌情報で LLM 由来の不確実性あり
- 🔴 **特記**: 本メタ分析の pooled 結果と比較すべき重要知見

---

# Part A: Benchmark Meta-Analyses（一般学業、オンライン未分化）

本節の目的: 既存メタ分析の pooled ρ 値を**ベンチマーク**として確立し、本メタ分析の「オンライン特化 pooled ρ」との差分検証 (RQ2) に使用する。

## C-01. Poropat (2009)

**TBD**: Phase 2 で精読

## C-02. Vedel (2014)

**TBD**: Phase 2 で精読

## C-03. Mammadov (2022) 🔴

**Citation**: Mammadov, S. (2022). Big Five personality traits and academic performance: A meta-analysis. *Journal of Personality, 90*(2), 222–255. https://doi.org/10.1111/jopy.12663

**Received**: 14 Feb 2021 | **Revised**: 9 Jul 2021 | **Accepted**: 12 Jul 2021

### Study characteristics

| 項目 | 値 |
|------|------|
| k (independent samples) | **267** |
| k (unique studies) | **228** |
| Total N | **413,074** |
| Range of publication years | 〜2020 まで |
| Analysis model | Random-effects, Hunter-Schmidt corrected for measurement error |
| Effect size metric | ρ (disattenuated correlation) |
| Data availability | OSF https://osf.io/tkwdu/ |

### Overall pooled effect sizes（corrected ρ）

| Trait | ρ | 95% CI | d (Cohen) | 統計的有意 |
|-------|---|--------|-----------|-----------|
| **Conscientiousness** | **0.27** | — | 0.56 | p < .001 ⭐ 最強 |
| Openness | 0.16 | — | 0.33 | p < .001 |
| Agreeableness | 0.09 | — | 0.19 | p < .001 |
| Extraversion | 0.01 | — | 0.03 | **n.s.** (CI includes 0) |
| Neuroticism | -0.02 | — | -0.04 | **n.s.** (CI includes 0) |

- Uncorrected overall mean: r = 0.08
- Heterogeneity: **I² ≈ 97.63%** (extreme) → moderator 必須
- Conscientiousness は cognitive ability 統制後も **28% の説明分散**を保持

### Moderation by education level（Table 3）

| Level | k | ρ(O) | ρ(C) | ρ(E) | ρ(A) | ρ(N) |
|-------|---|------|------|------|------|------|
| Elementary/Middle (ref) | 24–31 | **0.40** 🔴 | 0.31 | 0.15 | 0.18 | -0.01 |
| Secondary | 48–55 | 0.22 | 0.27 | 0.01 | 0.08 | -0.01 |
| Postsecondary | 152–175 | 0.10 | **0.26** | -0.01 | 0.08 | -0.02 |

**含意**:
- Openness は**年齢とともに減衰**（小学生最強、大学生最弱）
- Conscientiousness は**教育段階を通じて安定**（0.26–0.31）
- Extraversion, Agreeableness も elementary で最強 → secondary で減衰
- Neuroticism は全段階で n.s.

### Moderation by region（Table 5）🔴 本メタ分析 H1 の根拠

| Region | k | ρ(O) | ρ(C) | ρ(E) | ρ(A) | ρ(N) |
|--------|---|------|------|------|------|------|
| **Asia** | 16 | 0.29 | **0.35** 🔴 | 0.16 | 0.23 | **-0.19** |
| Australia | 7–9 | 0.17 | 0.24 | 0.06 | 0.12 | -0.02 |
| E. Europe/Russia | 24–26 | 0.18 | 0.30 | 0.00 | 0.07 | 0.03 |
| Middle East | 10–11 | 0.15 | **0.35** | 0.07 | 0.17 | -0.02 |
| **N. America** | 74–96 | 0.14 | 0.23 | -0.01 | 0.09 | 0.00 |
| W. Europe | 93–104 | 0.17 | 0.28 | -0.01 | 0.07 | -0.01 |

**Mammadov 本人の caveat**: Asian samples は全体の **1.5%** しかなく、結果の解釈には注意が必要。WEIRD 偏重の既存文献に対する限界として報告。

### Moderation by personality measure（Table 4）

| Measure | ρ(C) | ρ(O) | ρ(N) |
|---------|------|------|------|
| BFI | 0.26 | 0.10 | 0.00 |
| BFQ | 0.30 | 0.45 (outlier?) | -0.01 |
| IPIP | 0.31 | 0.10 | -0.02 |
| Markers (Mini) | 0.18 (最低) | 0.06 | 0.01 |
| NEO-FFI | 0.28 | 0.13 | **-0.07** ⭐ |
| NEO-PI-R | 0.28 | 0.13 | 0.01 |

**含意**:
- Conscientiousness の範囲 0.18–0.31 — Markers が最弱（performance-striving facet を欠くため）
- BFI は C のサンプル内異質性大
- Neuroticism は NEO-FFI でのみ有意負（他は n.s.）

### Comparison with prior meta-analyses（自己批判）

| Study | ρ(C) | N |
|-------|------|---|
| Poropat (2009) | 0.22 | 70,926 |
| McAbee & Oswald (2013) | 0.26 | 26,382 |
| Vedel (2014) | 0.26 | 17,717 |
| **Mammadov (2022)** | **0.27** | **413,074** |

- Mammadov 2022 は C の estimation を **最も堅牢**に確立（N が 5.8 倍）
- Openness は過去の meta-analyses より大きく推定（Poropat 0.12 → Mammadov 0.16）

### Incremental validity（Table 8）

| Predictor | B | RW | RW% |
|-----------|---|-----|-----|
| Cognitive ability | 0.42 | 0.177 | **63.59%** |
| Conscientiousness | 0.35 | 0.078 | **27.93%** |
| Openness | 0.03 | 0.011 | 3.94% |
| Neuroticism | 0.13 | 0.005 | 1.88% |
| Agreeableness | -0.02 | 0.005 | 1.84% |
| Extraversion | -0.05 | 0.002 | 0.81% |
| Total R² | — | — | **27.8%** |

- Conscientiousness は cognitive ability 控除後も**相対重要度 27.93%** を保持
- Openness は 3.94%（誤差レベル）
- 他の Big Five は negligible

### Meta-regression（gender, age）

- Openness: 女性比率↑ → ρ↓ (B=-0.003***), 年齢↑ → ρ↓ (B=-0.014***)
- Extraversion: 女性比率↑ → ρ↓ (B=-0.001*), 年齢↑ → ρ↓ (B=-0.004*)
- C, A, N: gender・age の影響小

### Publication bias

- Egger's regression: **C 以外すべて有意**（β0 = -0.59〜0.83）
- C は CI = [-2.62, 1.44] で bias なし
- Trim-and-fill 実施済み
- 出版バイアスと extreme heterogeneity の区別は困難と注記

### Limitations（本人記述）

1. 出版バイアスの証拠（Egger test）→ 効果量過大推定の可能性
2. Academic performance 測定の異質性（GPA, exam, test score が混在）
3. Asian 16 samples 少なく、region moderation の解釈に留保
4. Facet-level 分析なし（長尺版 inventory の研究限定のため）

### 🔴 本メタ分析（Tokiwa）への含意

1. **H1 (C 最強) の根拠**: 本 meta-analysis で C ρ = 0.27 は教育段階・地域・測定法を問わず堅牢 → オンライン学習でも同様が予測される
2. **H1 の Asian 特化**: Asian samples で C ρ = 0.35 → 日本サンプル（Nakayama, Tokiwa）でも高めの推定期待
3. **ベンチマーク比較 (RQ2)**: 本メタ分析の online-specific pooled ρ と Mammadov の ρ を差分検定すべき
4. **Openness の age decline**: オンライン学習は主に大学生 → 本研究では O の効果は小さい可能性（H2 と整合するかは k 次第）
5. **Extraversion の elementary 効果**: K-12 sub-analysis で検証可能か → A-26 Wang (2023) K-12 中国が重要
6. **測定法モデレーター**: NEO-FFI 採用研究では N が有意負 → 本研究でも measure を moderator に入れる必要

### RoB（Risk of Bias, JBI 8 項目、推定）

| 項目 | 評価 | 根拠 |
|------|------|------|
| 1. サンプル包含基準明示 | Yes | Method 2.1 |
| 2. サンプル設定記述 | Yes | Table 1 詳細 |
| 3. Exposure 妥当性 | Yes | 6 validated measures |
| 4. Outcome 客観性 | Partial | GPA/exam/test mixed |
| 5. Confounder 特定 | Yes | Cognitive ability 含む |
| 6. Confounder 対処 | Yes | Incremental validity 検定 |
| 7. Outcome 信頼性 | Yes | 測定法 moderator |
| 8. 統計適切性 | Yes | Hunter-Schmidt + subgroups |
| **Aggregate** | **8/8** | **低リスク**（C-01〜C-05 中最高品質） |

## C-04. Stajkovic et al. (2018)

**TBD**: Phase 2 で精読

## C-05. McAbee & Oswald (2013)

**TBD**: Phase 2 で精読

---

# Part B: Existing Systematic Reviews（online-specific）

## D-01. Gray & DiLoreto (2024 推定) ⚠

**TBD**: Phase 2 で精読

---

# Part C: Primary Studies（本メタ分析対象）

## A-01. Abe (2020)
**TBD**: Phase 2 で精読

## A-02. Alkış & Temizel (2018)
**TBD**: Phase 2 で精読

## A-03. Ashouri et al. (2025)
**TBD**: Phase 2 で精読

## A-04. Audet et al. (2021)
**TBD**: Phase 2 で精読

## A-05. Audet et al. (2023)
**TBD**: Phase 2 で精読

## A-06. Baruth & Cohen (2021)
**TBD**: Phase 2 で精読

## A-07. Baruth & Cohen (2023)
**TBD**: Phase 2 で精読

## A-08. Bhagat et al. (2019)
**TBD**: Phase 2 で精読

## A-09. Bhattacharjee & Ramkumar (2025) ⚠
**TBD**: Phase 2 で精読

## A-10. Boonyapison et al. (2025) ⚠
**TBD**: Phase 2 で精読

## A-11. Cheng et al. (2023)
**TBD**: Phase 2 で精読

## A-12. Cohen & Baruth (2017)
**TBD**: Phase 2 で精読

## A-13. Dang et al. (2024) ⚠
**TBD**: Phase 2 で精読

## A-14. Eilam et al. (2009) ❌（対面、除外予定）
**TBD**: Phase 2 で精読（除外根拠の確認）

## A-15. Elvers et al. (2003)
**TBD**: Phase 2 で精読

## A-16. Garzón-Umerenkova et al. (2024) ⚠
**TBD**: Phase 2 で精読

## A-17. Kara et al. (2024)
**TBD**: Phase 2 で精読

## A-18. Keller & Karau (2013)
**TBD**: Phase 2 で精読

## A-19. MacLean (2022) ⚠
**TBD**: Phase 2 で精読

## A-20. Mustafa et al. (2022)
**TBD**: Phase 2 で精読

## A-21. Nakayama et al. (2014)
**TBD**: Phase 2 で精読（Japanese context）

## A-22. Quigley et al. (2022)
**TBD**: Phase 2 で精読

## A-23. Rodrigues et al. (2024)
**TBD**: Phase 2 で精読

## A-24. Tlili et al. (2023)
**TBD**: Phase 2 で精読

## A-25. Tokiwa (2025)
**TBD**: Phase 2 で精読（COI 研究）

## A-26. Wang et al. (2023)
**TBD**: Phase 2 で精読

## A-27. Wu & Yu (2024)
**TBD**: Phase 2 で精読

## A-28. Yu (2021)
**TBD**: Phase 2 で精読

---

# Part D: Cross-cutting Synthesis（Introduction 執筆用）

## D1. 性格 × 学業のベンチマーク効果量（一般的学業）
**TBD**: Phase 1 後に Part A から抽出して構築

## D2. オンライン学習環境の特殊性（理論的根拠）
**TBD**: Phase 2 後に Part C から抽出

## D3. 本メタ分析の novel contribution
**TBD**: 全精読完了後に構築

## D4. 仮説 H1–H5 の文献的根拠
**TBD**: 全精読完了後に構築

## D5. Methodological heterogeneity（抽出時の注意点）
**TBD**: 全精読完了後に構築

---

**最終更新**: Phase 1 開始時点（Introduction 執筆前）
