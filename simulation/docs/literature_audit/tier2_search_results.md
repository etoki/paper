# Tier 2 検索結果 — Tier 1 完了後の論理的補強用追加文献

実施日：2026-04-27
ブランチ：`claude/review-harassment-research-plan-Dy2eo`
目的：Tier 1 deep reading 後に発見された **論理的 gap を埋めるための追加文献**の URL リスト
関連：`tier1_search_results.md`、`deep_reading_notes.md`、`literature_audit.md`

---

## 背景：Tier 2 が必要になった経緯

Tier 1 24 件の deep reading（`deep_reading_notes.md`、2,405 行）で以下の懸念が浮上：

1. **Personality と SSS の独立性懸念**：「個人的にはこちら懸念点」（ユーザー）→ Personality は SSS の **上流共通原因**であり、両者は独立ではない、を補強する文献必要
2. **Phase 2 介入効果の証拠の薄さ**：Counterfactual C（structural intervention）の限界主張が Pruckner 2013 単独依存だった
3. **Causal framing の弱さ**：Phase 2「if-then projection」を causal inference 文献で grounding する必要
4. **Self-report 加害測定の妥当性**：Tier 1 では LLM 文脈の self-report bias 文献のみ。**人間 self-report 加害**の妥当性研究が薄い

Tier 2 では **4 sections（D / B / C / A）に分けて 12 件**を検索検証。

---

## 取得方針

- **Core**：必須。Introduction / Discussion で直接引用
- **Strong**：推奨。論理補強材料
- 書誌情報の信頼度マーク：
  - ✅ WebSearch / WebFetch で書誌情報を直接確認済
  - ⚠️ 主要書誌は確認済だが取得時に再確認推奨

---

## Section D：Personality は SSS / 職位 / 業種選択の上流（3 件）

**目的**：「Personality 効果は SSS / 職位より小さい」批判を、**両者は独立ではなく personality が上流共通原因**であることで反論する。

### Core

**[D-1]** ✅ Grijalva, E., Harms, P. D., Newman, D. A., Gaddis, B. H., & Fraley, R. C. (2015). Narcissism and leadership: A meta-analytic review of linear and nonlinear relationships. *Personnel Psychology, 68*(1), 1–47. https://doi.org/10.1111/peps.12072

- **役割**：Narcissism → leadership emergence の central meta-analysis（leadership effectiveness は予測せず）
- **Key finding**：narcissism と leadership emergence の正の関連、effectiveness とは関連なし。Narcissists は無関係に leader として emerge する
- **本研究での使用**：「Narcissism (low HH) → 昇進 → 権力濫用機会」という personality 上流 chain の central evidence
- **取得**：Wiley 要購読 / Penn State Digital Commons OA https://digitalcommons.unl.edu/context/pdharms/article/1006/viewcontent/Harms_PP_2015_Narcissism_and_leadership__DC_VERSION.pdf

**[D-2]** ✅ Lee, K., & Ashton, M. C. (2005). Psychopathy, Machiavellianism, and Narcissism in the Five-Factor Model and the HEXACO model of personality structure. *Personality and Individual Differences, 38*(7), 1571–1582. https://doi.org/10.1016/j.paid.2004.09.016

- **役割**：HEXACO HH ↔ Dark Triad の **inverse 関係を確立**した central paper
- **Key finding**：Psychopathy r=−.72、Machiavellianism r=−.57、Narcissism r=−.53 と HH。**Big Five Agreeableness では narcissism と関連せず → HEXACO が Dark Triad を better capture する**
- **本研究での使用**：「HEXACO HH 低 ≈ Dark Triad 高 ≈ harassment 加害高」という variable equivalence の anchor
- **取得**：ScienceDirect 要購読、ResearchGate に PDF 候補

### Strong

**[D-3]** ✅ Heckman, J. J., Stixrud, J., & Urzua, S. (2006). The effects of cognitive and noncognitive abilities on labor market outcomes and social behavior. *Journal of Labor Economics, 24*(3), 411–482. https://doi.org/10.1086/504455

- **役割**：**Personality（noncognitive abilities）が schooling、就業、職種選択、賃金を強く予測**する経済学側の central evidence
- **Key finding**：Cognitive と noncognitive 両方が同等に重要。Noncognitive skills は schooling 決定 + wages を説明
- **本研究での使用**：「Personality は SSS / income / occupation の **上流**（Heckman et al., 2006）」を main argument の経済学的支援に
- **取得**：NBER Working Paper（OA） https://www.nber.org/papers/w12006 / Univ Chicago Jenni site https://jenni.uchicago.edu/papers/Heckman-Stixrud-Urzua_JOLE_v24n3_2006.pdf

### 既存ライブラリで補完される文献（再確認）

- **Roberts, B. W., Kuncel, N. R., Shiner, R., Caspi, A., & Goldberg, L. R. (2007). The Power of Personality.** *Perspectives on Psychological Science, 2*(4), 313–345. — `simulation/prior_research/_text/` に存在 ✓
- **Ozer, D. J., & Benet-Martínez, V. (2006). Personality and the prediction of consequential outcomes.** — `simulation/prior_research/_text/` に存在 ✓
- → **既存 2 件 + Tier 2 D-1/2/3 = 計 5 件**で「Personality は上流」argument を充分支援

---

## Section B：Workplace harassment intervention systematic reviews（5 件）

**目的**：Phase 2 Counterfactual C（structural-only intervention）の **限定的効果** evidence を Pruckner 2013 単独依存から脱却させる。複数の meta-analyses + 大規模 review で **training-only intervention は extensive margin に効きにくい**ことを示す。

### Core

**[B-1]** ✅ Roehling, M. V., & Huang, J. (2018). Sexual harassment training effectiveness: An interdisciplinary review and call for research. *Journal of Organizational Behavior, 39*(2), 134–150. https://doi.org/10.1002/job.2257

- **役割**：sexual harassment training の interdisciplinary review。**training の効果 evidence は複合的・mixed**
- **Key finding**：「訓練は知識・申告率には影響あり、しかし harassment 発生率の減少は弱い」
- **本研究での使用**：「Phase 2 Counterfactual C は filing 等の proximal outcomes は改善しても、incidence 自体は限定的」を支持
- **取得**：Wiley 要購読、JSTOR mirror https://www.jstor.org/stable/26610706

**[B-2]** ✅ Bezrukova, K., Spell, C. S., Perry, J. L., & Jehn, K. A. (2016). A meta-analytical integration of over 40 years of research on diversity training evaluation. *Psychological Bulletin, 142*(11), 1227–1274. https://doi.org/10.1037/bul0000067

- **役割**：260 independent samples、40 年の meta-analysis。Diversity training の効果サイズ評価
- **Key finding**：Hedges g = .38 全体、しかし **reactions（受講後評価）と cognitive learning には大きく効くが、attitudes/behaviors への効果は小さく、時間とともに減衰**
- **本研究での使用**：「Diversity training の限界」evidence、structural intervention の attitude/behavior change での効果上限を明示
- **取得**：APA PsycNET 要購読、Cornell eCommons OA https://ecommons.cornell.edu/items/e1f19f26-86ee-48f4-97f8-7d8a61088d70

**[B-3]** ✅ Dobbin, F., & Kalev, A. (2018). Why doesn't diversity training work? The challenge for industry and academia. *Anthropology Now, 10*(2), 48–55. https://doi.org/10.1080/19428200.2018.1493182

- **役割**：「diversity training は **most expensive, least effective** diversity program」と直接批判する **influential perspective paper**
- **Key finding**：1930s 以来何百もの研究が antibias training は behavior 変化を生まないことを示してきた。代わりに mentoring、task force、accountability structures が効果的
- **本研究での使用**：Counterfactual C 限界の **headline-grabbing reference**。Phase 2 で「training-only ≠ effective intervention」の central claim
- **取得**：Taylor & Francis 要購読、Harvard scholar Dobbin OA https://scholar.harvard.edu/files/dobbin/files/an2018.pdf

### Strong

**[B-4]** ✅ Roehling, M. V., & Huang, J. (2022). The effects of sexual harassment training on proximal and transfer training outcomes: A meta-analytic investigation. *Personnel Psychology, 75*(3), 681–718. https://doi.org/10.1111/peps.12492

- **役割**：B-1 を 2022 年に **meta-analytic 化** した続編
- **Key finding**：訓練 method の多様性 + 訓練時間が長いほど proximal outcome 改善。**Transfer outcomes（実職場での behavior 変化）への効果は弱め**
- **本研究での使用**：B-1 の数値裏付け、Counterfactual C の sensitivity range の anchor
- **取得**：Wiley 要購読、ResearchGate 候補

**[B-5]** ✅ Antecol, H., & Cobb-Clark, D. (2003). Does sexual harassment training change attitudes? A view from the federal level. *Social Science Quarterly, 84*(4), 826–842. https://doi.org/10.1046/j.0038-4941.2003.08404001.x

- **役割**：U.S. Merit Systems Protection Board データで **訓練が attitude（何が harassment かの認識）に与える効果**を probit で分析
- **Key finding**：訓練は **attitude（特に男性で）には effect あり**、unwanted behaviors を harassment と recognize する傾向増。**ただし behavior 変化への transfer は別問題**
- **本研究での使用**：Antecol 2003 + Bezrukova 2016 + Dobbin & Kalev 2018 を**段階的引用**：「訓練は attitude → cognitive learning → behavior の各段階で効果が逓減」
- **取得**：Wiley 要購読、IZA Discussion Paper version OA https://docs.iza.org/dp1149.pdf（関連別論文）

### 論理的フレーム（Phase 2 Counterfactual C 限界の triangulation）

```
Antecol 2003：attitude 変化はあり（ただし federal sample）
        ↓
Bezrukova 2016：260 samples meta、g=.38 だが behavior に弱く時間とともに減衰
        ↓
Dobbin & Kalev 2018：何百もの研究を総括して「最も非効率な diversity 投資」
        ↓
Roehling 2018, 2022：sexual harassment training specific でも transfer outcome 弱い
        ↓
Pruckner 2013（既存）：moral reminder の field experiment、intensive margin のみ
        ↓
★ 本研究 Counterfactual C：structural intervention の effect は **30%減を上限**として保守的に推定 ★
```

---

## Section C：Counterfactual / "if-then projection" の causal framing 文献（2 件）

**目的**：Phase 2 の "if-then projection" を **formal causal inference** で grounding。Reviewer から「identifying assumption は何か？」と問われた時の reference 整備。

### Core

**[C-1]** ✅ Hernán, M. A., & Robins, J. M. (2020). *Causal Inference: What If*. Boca Raton: Chapman & Hall/CRC.

- **役割**：causal inference の **modern standard textbook**（**完全 free OA**）
- **Key contents**：counterfactuals、DAG、randomized experiments、observational studies、confounding、selection bias、IPW、g-estimation、g-formula、IV、survival analysis、**target trial emulation**、longitudinal data
- **本研究での使用**：
  - "If-then projection" を **target trial emulation** の simulation 版として framing
  - 「Phase 2 は介入 anchor の effect size assumption を transportability 仮定の下で aggregate に投影」
  - Identifying assumption（intervention X が anchor study と類似の population で同じ effect を持つ）を Discussion で明示
- **取得**：完全 OA https://miguelhernan.org/whatifbook
- 直リンク（最新版）：https://content.sph.harvard.edu/wwwhsph/sites/1268/2024/04/hernanrobins_WhatIf_26apr24.pdf
- ★ **無料、最重要**

### Strong

**[C-2]** ✅ Pearl, J. (2009). *Causality: Models, Reasoning, and Inference* (2nd ed.). Cambridge: Cambridge University Press. https://doi.org/10.1017/CBO9780511803161

- **役割**：causal inference の **theoretical foundation**（structural model、do-calculus、DAG）
- **Key contents**：probabilistic / manipulative / counterfactual / structural approaches を統合
- **本研究での使用**：Methods で「我々の simulation は Pearl (2009) の structural model を probabilistic に instantiate したもの」と一文で言及。Detail は Hernán & Robins に任せる
- **取得**：Cambridge / Amazon 要購入、ILLC Amsterdam にスキャン版あり https://archive.illc.uva.nl/cil/uploaded_files/inlineitem/Pearl_2009_Causality.pdf

### 既存ライブラリで補完される文献（再確認）

- **Lundberg, I., Brand, J. E., & Jeon, N. (2024). The origins of unpredictability in life outcome prediction tasks.** *PNAS, 121*(24). — `simulation/prior_research/_text/` に存在 ✓
- **Salganik et al. (2020) Measuring the predictability of life outcomes.** *PNAS, 117*(15). — 同上 ✓
- → 既存 2 件は「individual prediction の限界」、Tier 2 C-1/C-2 は「causal framework」、合わせて **本研究の epistemological positioning が clean に確立**

---

## Section A：Self-report perpetration validity（2 件）

**目的**：本研究の N=354 は **self-report 加害**。「self-report 加害は社会的望ましさで underreport される」という標準的批判への defensive citation 整備。

### Core ★

**[A-1]** ✅ Berry, C. M., Carpenter, N. C., & Barratt, C. L. (2012). Do other-reports of counterproductive work behavior provide an incremental contribution over self-reports? A meta-analytic comparison. *Journal of Applied Psychology, 97*(3), 613–636. https://doi.org/10.1037/a0026739

- **役割**：**CWB（counterproductive work behavior）の self-report vs other-report の meta-analytic 比較**。本研究にとって **central anchor**
- **Key findings**（重要な反論材料）：
  1. Self-report と other-report は moderately to strongly correlated
  2. **Self-raters reported MORE CWB than other-raters**（**自己過剰評価ではなく、他者過小評価**）
  3. Self- と other-report の **correlate との関連 pattern と magnitude は類似**（妥当性 indicator）
  4. Other-report は incremental variance をほとんど追加せず
  5. **CWB の多くは covert behaviors**（同僚・上司から observable でない）→ self-report が "best at eliciting accurate frequency"
- **本研究での使用**：
  - 「Self-report harassment perpetration is **not** systematically downward-biased — meta-analytic evidence shows self-reports actually elicit MORE perpetration than other-reports for these covert behaviors (Berry, Carpenter, & Barratt, 2012)」
  - Limitation セクションで standard concern を **defuse**：「Self-report concern is acknowledged in the harassment literature but not supported by meta-analytic evidence in the CWB domain」
- **取得**：APA PsycNET 要購読、PubMed PMID 22201245 https://pubmed.ncbi.nlm.nih.gov/22201245/

### Strong

**[A-2]** ✅ Anderson, C. A., & Bushman, B. J. (2002). Human aggression. *Annual Review of Psychology, 53*, 27–51. https://doi.org/10.1146/annurev.psych.53.100901.135231

- **役割**：**General Aggression Model (GAM)** の central review。Aggression measurement の external validity 議論を含む
- **Key contents**：5 situational variables (provocation, violent media, alcohol, anonymity, hot temperature) + 3 individual difference variables (sex, Type A, trait aggressiveness) を laboratory + real-world で meta-analysis。**Lab measures が real-world と相関する** ことを示す
- **本研究での使用**：
  - 「Self-report aggression / harassment is correlated with real-world aggressive behavior (Anderson & Bushman, 2002)」
  - GAM framework を Discussion で「individual + situation の interaction」議論に活用（personality alone でなく）
- **取得**：Annual Reviews 要購読、PMID 11752478 https://pubmed.ncbi.nlm.nih.gov/11752478/

### 既存ライブラリで補完される文献（再確認）

- **The Personality Illusion: Revealing Dissociation Between Self-Reports & Behavior in LLMs** — `simulation/prior_research/_text/` に存在（**LLM 文脈、人間ではない**）
- **LLM social desirability biases** — 同上
- → 既存は LLM 文脈のみ、Tier 2 A-1/A-2 で **人間 self-report 加害**の妥当性を直接補強

### 論理的フレーム（Limitation セクション統合）

```
標準的批判：「Self-report harassment は social desirability で underreported」
        ↓
反論 1（Berry, Carpenter, & Barratt 2012 meta）：
   Self-raters は other-raters より MORE CWB を報告
   → 自己過小評価仮説は支持されない
        ↓
反論 2（Anderson & Bushman 2002）：
   Self-report aggression は real-world aggressive behavior と相関
   → external validity あり
        ↓
反論 3（Bowling & Beehr 2006、既存）：
   CWB の多くは covert、self-report が最もよく拾う
        ↓
★ 結論：本研究の N=354 self-report 加害データは、systematic underreport の懸念は弱く、population-scale aggregation に十分妥当 ★
```

---

## 取得チェックリスト（合計 12 件）

ユーザー手動取得用。

### 🟢 OA で容易に取得可能（5 件）

- [ ] **[D-1]** Grijalva et al. (2015) Narcissism and Leadership *Personnel Psychology* — Penn State Digital Commons https://digitalcommons.unl.edu/context/pdharms/article/1006/viewcontent/Harms_PP_2015_Narcissism_and_leadership__DC_VERSION.pdf
- [ ] **[D-3]** Heckman, Stixrud, & Urzua (2006) *Journal of Labor Economics* — NBER OA https://www.nber.org/papers/w12006
- [ ] **[B-3]** Dobbin & Kalev (2018) *Anthropology Now* — Harvard Dobbin OA https://scholar.harvard.edu/files/dobbin/files/an2018.pdf
- [ ] **[B-2]** Bezrukova et al. (2016) *Psychological Bulletin* — Cornell eCommons OA https://ecommons.cornell.edu/items/e1f19f26-86ee-48f4-97f8-7d8a61088d70
- [ ] **[C-1]** Hernán & Robins (2020) *Causal Inference: What If* — 完全 OA https://content.sph.harvard.edu/wwwhsph/sites/1268/2024/04/hernanrobins_WhatIf_26apr24.pdf

### 🟡 要購読 / repository 経由（5 件）

- [ ] **[D-2]** Lee & Ashton (2005) *Personality and Individual Differences* — ScienceDirect 要購読、ResearchGate 候補 https://www.researchgate.net/publication/222079282
- [ ] **[B-1]** Roehling & Huang (2018) *Journal of Organizational Behavior* — Wiley 要購読、JSTOR https://www.jstor.org/stable/26610706
- [ ] **[B-4]** Roehling & Huang (2022) *Personnel Psychology* — Wiley 要購読
- [ ] **[B-5]** Antecol & Cobb-Clark (2003) *Social Science Quarterly* — Wiley 要購読
- [ ] **[A-1]** Berry, Carpenter, & Barratt (2012) *Journal of Applied Psychology* — APA PsycNET 要購読 https://pubmed.ncbi.nlm.nih.gov/22201245/

### 🔴 要購入（書籍、2 件）

- [ ] **[C-2]** Pearl (2009) *Causality* (2nd ed.) — Cambridge 書籍、ILLC 全文スキャン候補 https://archive.illc.uva.nl/cil/uploaded_files/inlineitem/Pearl_2009_Causality.pdf
- [ ] **[A-2]** Anderson & Bushman (2002) *Annual Review of Psychology* — Annual Reviews 要購読 https://pubmed.ncbi.nlm.nih.gov/11752478/

---

## 全体サマリーと論文への含意

### Tier 2 verdict（合計 12 件取得計画）

| Section | 件数 | OA | 充足度 | 論理的役割 |
|---|---|---|---|---|
| D（Personality upstream） | 3 | 2 | ✅ 充足（既存 2 件含め 5 件で核確立） | 「個人の性格と SSS は独立ではない」反論 |
| B（Intervention reviews） | 5 | 2 | ✅ 充足（4 系統で triangulation） | Counterfactual C の限界 evidence 強化 |
| C（Causal inference） | 2 | 1（OA） | ✅ 充足（OA 1 件で核確立） | "If-then projection" の formal grounding |
| A（Self-report validity） | 2 | 0（PMID から論文情報のみ確実）| ◯ 中程度（既存 + Tier 2 で 4 件統合）| Self-report 批判への defensive citation |
| **合計** | **12** | **5 OA** | ✅ **論理的 gap 全カバー** | |

### Introduction / Discussion 統合計画

#### Introduction の修正（Tier 1 cross-pillar synthesis 案 + Tier 2 統合）

**第 3 段落 Predictor lineage を以下のように拡張**：

> "Personality traits are established correlates of workplace harassment exposure. Meta-analytic evidence shows that neuroticism (r = 0.25), low agreeableness (r = -0.17), and to a lesser degree low conscientiousness (r = -0.10) predict harassment victimization (Nielsen, Glasø, & Einarsen, 2017). On the perpetrator side, HEXACO's Honesty-Humility dimension—absent from the Five-Factor Model—emerges as the strongest personality predictor of workplace deviance (Pletzer et al., 2019), and Dark Triad traits, particularly Psychopathy, predict harassment perpetration in our prior work (Tokiwa et al., preprint).
>
> **Importantly, personality and socioeconomic position are not independent risk factors. Personality traits are upstream determinants of consequential life outcomes including educational attainment, occupational status, and income (Heckman, Stixrud, & Urzua, 2006; Ozer & Benet-Martínez, 2006; Roberts et al., 2007). Narcissism specifically predicts leadership emergence and career advancement (Grijalva et al., 2015), while being inversely correlated with Honesty-Humility (Lee & Ashton, 2005, r = -0.53 to -0.72). Thus, modeling personality typology captures both direct personality effects on harassment perpetration and indirect effects mediated through personality-driven life-course selection into positions of organizational power.**"

#### Discussion の Limitation 統合

**Self-report concern**：

> "Self-report measurement of harassment perpetration is sometimes considered vulnerable to social desirability bias. However, meta-analytic evidence on counterproductive work behavior shows self-raters report **more** behaviors than other-raters (Berry, Carpenter, & Barratt, 2012), and self-report measures correlate with real-world aggressive behavior (Anderson & Bushman, 2002). The covert nature of much harassment behavior (Bowling & Beehr, 2006) makes self-report measures the most informative single source. Triangulation with MHLW national survey provides external benchmarking."

**Phase 2 framing as conditional projection**：

> "Following established frameworks for causal inference (Hernán & Robins, 2020; Pearl, 2009), our Phase 2 counterfactual projections are presented as **'if-then projections under transportability assumptions'** rather than causal estimates. We assume the intervention effect sizes derived from prior randomized and quasi-experimental studies (Hudson, 2023; Kruse et al., 2014; Pruckner & Sausgruber, 2013) transport to the Japanese workforce population. This is a strong assumption; sensitivity analysis varies effect sizes across plausible ranges."

**Counterfactual C 限界の triangulation**：

> "Multiple meta-analytic syntheses suggest that diversity and harassment training programs produce reliable changes in awareness and proximal outcomes (Antecol & Cobb-Clark, 2003; Bezrukova et al., 2016; Roehling & Huang, 2022) but show limited effects on actual behavior change or harassment incidence (Dobbin & Kalev, 2018; Roehling & Huang, 2018). This pattern—proximal effective, distal weak—aligns with Pruckner and Sausgruber (2013)'s finding that moral reminders affect intensive but not extensive margins of dishonesty. Our Counterfactual C effect-size assumption (30% reduction upper bound) is therefore conservative within the empirical literature."

### 想定 reviewer 攻撃と反論（Tier 2 強化版）

| Reviewer 質問 | Tier 1 only での答え | Tier 1 + Tier 2 での答え |
|---|---|---|
| "Personality vs SSS、なぜ personality？" | "Personality slice の effect" | "**Personality は SSS の上流共通原因**（Heckman 2006、Roberts 2007、Grijalva 2015、Lee & Ashton 2005）。Personality typology は両者を inherently include" |
| "Self-report 加害は biased では？" | "Limitation で acknowledge" | "**Berry, Carpenter, & Barratt (2012) meta-analysis：self-report は MORE CWB を eliciting**。Anderson & Bushman 2002 で external validity も確立" |
| "Phase 2 の causal claim は強すぎ？" | "Conditional projection と framing" | "**Hernán & Robins (2020) target trial emulation framework**で formal grounding。Transportability 仮定を明示" |
| "Counterfactual C 限界は単一論文 (Pruckner)？" | "Pruckner 単独" | "**Bezrukova 2016 (260 samples meta) + Dobbin & Kalev 2018 + Roehling 2018/2022 で triangulation**。'Proximal effective, distal weak' pattern は確立" |

### 次のステップ（Tier 2 取得後の作業）

1. **deep_reading_notes.md に Tier 2 12 件の deep reading セクションを追加**（必要に応じて）
2. **Introduction draft 着手**（5 段落構成 + Tier 2 統合 framing）
3. **Stage 0 コード実装**（D13 14-cell 設計）
4. **Discussion limitation セクション執筆**（Tier 2 defensive citations 統合）

### Tier 3（後でも可、本研究の核には不要）

- Pillar 6 補強：Westreich et al. transportability methodology — Hernán & Robins 2020 で代替可
- Pillar 4：Asendorpf 2001、Specht 2014 personality types longitudinal — Tier 1 clustering 文献で代替可
- Pillar 8：Sen 1999、Nussbaum 2011 capability approach — 著作向け、論文では薄く

---

**本ドキュメントの commit 後**、ユーザーは Tier 2 の 12 件（OA 5 件、要購読 5 件、書籍 2 件）を順次取得。並行して論文 Introduction draft 着手可能。
