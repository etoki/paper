# 先行研究 Audit — HEXACO 7-Typology Harassment Simulation

実施日：2026-04-27
ブランチ：`claude/review-harassment-research-plan-Dy2eo`
目的：研究の **value claim**（novelty + positioning）を支える文献基盤の現状把握と gap 抽出
関連：`research_plan_harassment_typology_simulation.md`、`D13_power_analysis.md`

---

## 0. 監査の枠組み

本研究の論文としての価値は、以下 3 つの主張で成立する：

1. **Substantive**：日本のハラスメント実態を性格類型ベースで再現できる
2. **Methodological**：LLM 不使用の確率論的 type-conditional simulation は LLM 系譜とは別の方法論的系譜を成す
3. **Applied**：3 種介入 counterfactual の比較から、社会システムレベルの政策含意が引き出せる

これらを支える文献領域を **8 pillar** に分解し、既存ライブラリ（`harassment/prior_research/`、`clustering/prior_research/`、`simulation/prior_research/_text/`、`simulation/reference_index.md` の noted refs）と照合する。

| Pillar | 内容 | 役割 |
|---|---|---|
| 1 | Workplace harassment epidemiology | Phase 1 validation target の正当化 |
| 2 | Personality → harassment perpetration | 「なぜ HEXACO で予測できるか」の根拠 |
| 3 | Non-LLM population microsimulation | 方法論的 novelty の系譜 |
| 4 | Type-conditional / cluster-based prediction | 7 類型を介入単位にする論理 |
| 5 | Harassment intervention reviews | Phase 2 anchor の補強 |
| 6 | Counterfactual / what-if modeling | Phase 2 の方法論的位置付け |
| 7 | Self-report perpetration validity | Limitation 1 への対応 |
| 8 | Capability approach in social policy | Discussion での "consistent with" 引用 |

---

## 1. 既存ライブラリ全体像

| ディレクトリ | PDF 件数 | 補助 .txt | 主領域 |
|---|---|---|---|
| `simulation/prior_research/_text/` | 約 50 | 約 19 | LLM simulation、性格介入、heritability、性格-成果予測 |
| `harassment/prior_research/` | 6 | 0 | HEXACO/DT × 職場逸脱、Japanese harassment 尺度 |
| `clustering/prior_research/` | 6 | 0 | 性格類型・clustering 方法論 |
| `simulation/reference_index.md` | n/a（記載のみ） | n/a | CS222 reading list、ABM 古典、Schelling 等 |

**特徴**：
- LLM 系 simulation 文献は Park 2024 を含めて充実（reference_index.md に約 25 件）
- HEXACO 関連の psychometric 基礎は Soto, Power of Personality 等で十分
- 性格介入の anchor（Roberts 2017、Hudson 2023、Kruse 2014、Pruckner 2013）は確定済
- **Harassment 領域の epidemiology / 介入レビュー / 国際比較は手薄**

---

## 2. Pillar 別監査

（以下、各 pillar につき：所有文献 / verdict / 不足のクリティカル度 / 想定検索クエリ）

---

### Pillar 1：Workplace harassment epidemiology

**役割**：Phase 1 の validation target（厚労省実態調査）に学術的脈絡を与える。「なぜ harassment 率を国レベルで再現することが学問的に意味があるか」を示す。

**所有文献**：

| ファイル | 著者 / 出典 | 性格 |
|---|---|---|
| `harassment/prior_research/24_15.pdf` | 小林敦子・田中堅一郎 (2010) 産業・組織心理学研究 24(1): 15–27 | Gender Harassment Scale 開発（**測定**、prevalence ではない） |
| `harassment/prior_research/81_3C-094.pdf` | 鄧科ほか (2017) 日心第81回大会 | Power Harassment Scale 新規開発（**測定**、prevalence ではない） |

**Verdict**：❌ **致命的に不足**

**何が不足か**：

| 必要だが無い文献領域 | 重要度 | 理由 |
|---|---|---|
| Einarsen 系 workplace bullying epidemiology（NAQ-R, prevalence rates） | ★★★ | 国際 prevalence の reference frame |
| Notelaers et al. のラテンクラス分析（exposure types） | ★★★ | typology-based harassment の先行例（**直接競合 candidate**） |
| Bowling & Beehr (2006) workplace harassment meta-analysis | ★★★ | 加害・被害の predictor 全体像 |
| Nielsen, Matthiesen, & Einarsen 2010 prevalence systematic review | ★★ | 国際比較ベース |
| 日本の workplace harassment prevalence 査読論文（厚労省以外） | ★★ | 国内文脈 |
| 厚労省実態調査（2020 年度）一次資料 | ★★★ | Phase 1 validation の primary target、**未取得** |

**想定検索クエリ**（Tier 1 用）：
- `"workplace bullying" prevalence meta-analysis`
- `"workplace harassment" Japan prevalence epidemiology`
- `Einarsen NAQ "negative acts questionnaire" prevalence`
- `Notelaers latent class workplace bullying`
- `Bowling Beehr workplace harassment`
- `厚生労働省 職場のハラスメントに関する実態調査 2020`

**論文への影響**：この pillar が薄いと、Phase 1 が「なぜ厚労省統計と照合するのか」を学術的に正当化できず、**single-country case report** に格下げされる。Tier 1 必須。

---

### Pillar 2：Personality → harassment perpetration（直接エビデンス）

**役割**：「HEXACO 7 類型から加害確率を推定する」ことの妥当性根拠。Phase 1 Stage 0 の確率テーブル設計を裏付ける。

**所有文献**：

| ファイル | 著者 / 出典 | 性格 |
|---|---|---|
| `harassment/prior_research/A meta-analysis ... Big Five versus HEXACO.pdf` | Pletzer, Bentvelzen, Oostrom, & De Vries (2019) JVB | **Gold standard**：HEXACO vs Big Five の workplace deviance 予測力比較 |
| `harassment/prior_research/Considering sadism in the shadow of the Dark Triad.pdf` | Bonfá-Araujo et al. (Dark Tetrad meta-analysis) | DT 各成分 × outcomes |
| `harassment/prior_research/DifferencesAmongDarkTriadComponentsAMetaAnalyticInvestigation.pdf` | Muris, Merckelbach, Otgaar, & Meijer (2017) PPS | DT 個別成分の差別化 |
| `harassment/prior_research/The Malevolent Side Of Human Nature.pdf` | Furnham, Richards, & Paulhus (2013) PPS | DT 古典 review |
| `simulation/prior_research/_text/Linking big personality traits to anxiety, depressive...meta-analysis.pdf` | Kotov et al. (2010) PB | personality × psychopathology（peripheral） |

**著者本人の Harassment 論文**（自己引用 hub）：
- Tokiwa et al. (preprint) — N=354、HEXACO + DT × power/gender harassment、HC3-robust 階層回帰

**Verdict**：◯ **MEDIUM-STRONG**

**何が強いか**：
- Pletzer 2019 メタ分析が HEXACO の HH を workplace deviance 最強 predictor として確立
- DT × workplace aggression は 3 件の meta で多角的に押さえている
- 自己研究 (Tokiwa) で日本サンプルでの再現を確認済

**何が弱いか**：

| 必要だが薄い領域 | 重要度 | 理由 |
|---|---|---|
| HEXACO × harassment **specific**（deviance ではなく harassment）の meta-analysis | ★★ | Pletzer 2019 は deviance 全般、harassment 直接 meta は別途存在するか要確認 |
| Workplace bullying perpetrator personality（Einarsen 系） | ★★ | 加害者プロファイル研究 |
| Honesty-Humility × counterproductive work behavior の review 系 | ★★ | HH を主軸に置く論文設計の補強 |
| Japan-specific personality × harassment 研究（自己引用以外） | ★ | 国内文脈の厚み |

**想定検索クエリ**（Tier 1）：
- `HEXACO "Honesty-Humility" workplace harassment perpetration`
- `personality "workplace bullying" perpetrator meta-analysis`
- `Big Five Dark Triad harassment perpetration meta-analysis`

**論文への影響**：基盤は十分。**追加 1–3 本**で「HH を中心に置く理論的選択」の正当化を補強できれば理想。Tier 2 で十分（Phase 1 の核心は simulation であり、predictor の妥当性は副次）。

---

### Pillar 3：Non-LLM population microsimulation lineage

**役割**：本研究の方法論的 novelty 主張の核。「**LLM ではない確率論的 microsimulation**」が独立した方法論的系譜であり、本研究はその系譜の harassment 領域への初応用、と positioning する。

**所有文献（PDF）**：

| ファイル | 著者 / 出典 | 性格 |
|---|---|---|
| `Personality-Driven Student Agent-Based Modeling...` | Xiao & Shen (2026) arXiv | LLM-based ABM、**非 LLM ではない** |
| `LLM Agent-Based Simulation of Student Activities...` | LLM-based、**非 LLM ではない** | |
| `Measuring the predictability of life outcomes...` | Salganik et al. (2020) PNAS | mass collaboration、ML 予測（simulation ではない） |
| `The origins of unpredictability in life outcome prediction tasks` | Lundberg, Brand, & Jeon (2024) PNAS | 個人予測の限界 review |

**所有文献（reference_index.md にあるが PDF 未取得 / 浅い）**：

| 著者 / 出典 | 関連性 |
|---|---|
| Schelling (1978) Micromotives and macrobehavior | ABM 古典、segregation model（**該当系譜の祖**） |
| Bruch & Atwell (2015) ABMs in Empirical Social Research | sociological ABM レビュー |
| Chang et al. (2021) Mobility network COVID Nature | Public health microsim 例 |

**Verdict**：❌ **致命的に不足**

**何が不足か**：

| 必要だが無い文献領域 | 重要度 | 理由 |
|---|---|---|
| Microsimulation in epidemiology / public health（古典 + 現代）| ★★★ | 「個人 → 集団」の確率モデル系譜の祖 |
| Static microsimulation in economics / tax-benefit policy | ★★ | 政策 simulation の方法論 |
| System dynamics / ABM for organizational behavior | ★★ | bullying / harassment を ABM で扱った先行例 |
| Probabilistic risk models in occupational health | ★★ | risk allocation simulation の系譜 |
| Latent class × prevalence projection（疫学） | ★★ | typology-conditional aggregate 予測の最近接系譜 |

**想定検索クエリ**（Tier 1）：
- `microsimulation epidemiology population health review`
- `agent-based model workplace bullying harassment`
- `Schelling segregation simulation social science methodology`
- `microsimulation tax benefit policy methodology`
- `latent class analysis population prevalence projection`
- `system dynamics organizational behavior bullying`

**論文への影響**：本 pillar が薄いと、論文が **「Park 2024 の劣化版（LLM を使わない）」と読まれかねない**。逆に充実させれば「Microsimulation という別系譜の harassment 応用」と positioning でき、**方法論的 novelty 主張が成立**する。**最重要 Tier 1**。

---

### Pillar 4：Type-conditional / cluster-based prediction の応用

**役割**：「7 類型を介入単位とする」論理の方法論的根拠。clustering 自体は確立しているが、**類型を予測モデルの入力として使う**先行例が必要。

**所有文献**：

| ファイル | 著者 / 出典 | 性格 |
|---|---|---|
| `clustering/prior_research/A robust data-driven approach...four personality types.pdf` | Gerlach et al. (2018) Nat Hum Behav | Big Five 4 type の確立（**clustering 方法論**） |
| `clustering/prior_research/Personality types revisited...Kerber.pdf` | Kerber, Roth, & Herzberg (2021) PLOS ONE | latent profile + clustering 統合 |
| `clustering/prior_research/Personality types based on the Big Five model A cluster analysis over the Romanian population.pdf` | Espinoza et al. | Romanian population types |
| `clustering/prior_research/Taking a person-centered approach...HEXACO latent-profile.pdf` | Daljeet et al. (2017) | HEXACO LPA |
| `clustering/prior_research/Establishing the structure and replicability of personality profiles using the HEXACO-PI-R.pdf` | HEXACO-PI-R 類型 replicability | |
| `clustering/prior_research/...Resilients, Overcontrollers, Undercontrollers...` | 日本語論文、Resilients/Over/Undercontrollers | 古典 3-type model |

**Verdict**：◯/△ **方法論基盤は十分、応用は薄い**

**何が強いか**：
- Clustering 論文（自己引用）の延長として、類型同定の方法論的妥当性は確保
- Gerlach 2018, Kerber 2021 で類型の robustness が押さえられている

**何が弱いか**：

| 必要だが薄い領域 | 重要度 | 理由 |
|---|---|---|
| 性格類型 → 行動 outcome 予測（health, employment, criminal） | ★★ | 「類型を介入単位として使う」根拠 |
| Person-centered vs variable-centered approach の使い分け論争 | ★★ | dimensional vs typological のトレードオフ |
| Latent class × outcome modeling の epidemiology 応用 | ★★ | Pillar 3 と橋渡し |

**想定検索クエリ**（Tier 2）：
- `personality types prediction health outcomes longitudinal`
- `person-centered approach personality outcome workplace`
- `latent profile analysis HEXACO outcome prediction`
- `personality clusters intervention targeting`

**論文への影響**：clustering 論文の延長で論理的に繋げれば充足可能。Tier 2。

---

### Pillar 5：Harassment intervention systematic reviews

**役割**：Phase 2 の 3 種 counterfactual（A 普遍 / B 集中 / C 構造）の効果量 anchor。研究計画 Part 6.2 で確定済み。

**所有文献（介入 anchor）**：

| ファイル | 著者 / 出典 | Counterfactual 役割 |
|---|---|---|
| `simulation/prior_research/_text/A Systematic Review of Personality Trait Change Through Intervention.pdf` | Roberts et al. (2017) Psych Bull | 全体 anchor（d=0.37 24 週） |
| `An upward spiral between gratitude and humility.pdf` | Kruse et al. (2014) SPPS | **Counterfactual A 主 anchor** (d=0.71) |
| `Lighten the darkness...agreeableness Dark Triad.pdf` | Hudson (2023) J Personality | **Counterfactual B 主 anchor** ★★★ |
| `Honesty on the Streets.pdf` | Pruckner & Sausgruber (2013) JEEA | **Counterfactual C 主 anchor**（intensive のみ効く） |
| `Beta-Testing of an Intervention Workbook to Promote Humility.pdf` | Lavelock et al. (2014) | A 補助 |
| `Online Group Counseling...Humility.pdf` | Naini et al. (2021) | A 補助 |
| `Personality Feedback as an Intervention...Moral Traits.pdf` | Casali, Metselaar, & Thielmann (2025) | C 補助 |
| `You have to follow through...volitional personality change.pdf` | Hudson et al. (2019) | follow-through 重要性 |

**Verdict**：◯ **anchor は確定、harassment-specific intervention レビューが薄い**

**何が強いか**：
- 個人レベル性格介入の anchor は完全に揃っている
- Counterfactual A/B/C それぞれの主 anchor が確定済（D8）

**何が弱いか**：

| 必要だが薄い領域 | 重要度 | 理由 |
|---|---|---|
| Sexual harassment training program meta-analysis（Roehling 系） | ★★ | C の補強。「研修だけでは効かない」エビデンス |
| Bystander intervention review（college / workplace） | ★★ | C の代替 anchor、displacement 問題の議論 |
| Workplace anti-harassment policy 効果研究 | ★★ | C 構造介入の効果量補強 |
| Power harassment specific intervention（日本） | ★ | 国内文脈 |

**想定検索クエリ**（Tier 2）：
- `sexual harassment training meta-analysis effectiveness`
- `bystander intervention workplace harassment review`
- `anti-harassment policy organizational effectiveness`
- `workplace bullying intervention systematic review Roehling`

**論文への影響**：Phase 2 の核 anchor は確定。「**C 構造介入の効果が限定的**」を補強する null-finding 文献を 2–3 本追加すると、B 主軸の論理が強化される。Tier 2。

---

### Pillar 6：Counterfactual / what-if modeling in policy psychology

**役割**：Phase 2 の方法論的 framing。「介入効果が anchor 値で実装されたら、aggregate output はこう変わる」という条件付き予測の正当化。

**所有文献**：

| ファイル | 著者 / 出典 | 性格 |
|---|---|---|
| なし — `simulation/reference_index.md` にも該当する直接文献は無し | | |

**間接的に関連**：
- Pruckner 2013（field experiment 自体が counterfactual 手法）
- Salganik 2020（予測の限界、counterfactual ではない）

**Verdict**：❌ **ほぼゼロ**

**何が不足か**：

| 必要だが無い文献領域 | 重要度 | 理由 |
|---|---|---|
| Causal inference under intervention assumption（Pearl, Hernán & Robins 系入門） | ★★ | Phase 2 の framing 正当化 |
| Decomposition / mediation analysis methodology | ★★ | type ごとの効果分解 |
| Conditional projection / what-if analysis in epidemiology | ★★ | Phase 2 の "if-then projection" framing |
| Personality-based policy projection（社会政策 simulation） | ★★ | 直接 precursor が無いことが novelty を支える可能性 |

**想定検索クエリ**（Tier 2 / 3）：
- `counterfactual prediction intervention causal inference`
- `what-if simulation public health policy`
- `decomposition analysis intervention effect`
- `Hernán Robins causal inference target trial`

**論文への影響**：methodology section で**簡潔に**触れれば足りる。"if-then projection" として framing するために 2–3 本の方法論引用があれば十分。Tier 3。

---

### Pillar 7：Self-report perpetration validity

**役割**：研究計画の限界 1（self-report → self-report の循環性、Part 4.2）への対応根拠。

**所有文献**：

| ファイル | 著者 / 出典 | 性格 |
|---|---|---|
| `simulation/prior_research/_text/Large language models display human-like social desirability biases in Big Five personality surveys.txt` | LLM の social desirability bias | 関連だが **LLM 文脈** |
| `The Personality Illusion Revealing Dissociation Between Self-Reports & Behavior in LLMs.pdf` | LLM の self-report vs behavior | 関連だが **LLM 文脈** |
| `Test-retest reliability of the HEXACO-100.pdf` | HEXACO の信頼性 | 測定一般 |

**Verdict**：❌ **ほぼ薄い**（LLM 文脈の self-report bias はあるが、**人間** self-report harassment の妥当性研究が無い）

**何が不足か**：

| 必要だが無い文献領域 | 重要度 | 理由 |
|---|---|---|
| Self-report harassment perpetration の妥当性研究 | ★★ | limitation 1 への対応 |
| Self-report aggression / antisocial behavior の social desirability 研究 | ★★ | 対応根拠 |
| Multi-method（self / peer / observer）harassment 比較研究 | ★ | triangulation 補強 |
| Forced-choice / IRT 系の harassment 測定 | ★ | 測定改善先行例 |

**想定検索クエリ**（Tier 2 / 3）：
- `self-report harassment perpetration validity social desirability`
- `multi-source harassment measurement workplace`
- `self-report aggression validity peer report comparison`

**論文への影響**：Limitation セクションで明示する以上、参考文献 2–3 本は欲しい。Tier 2。

---

### Pillar 8：Capability approach in social policy operationalization

**役割**：研究計画 Doc 2 の規範的核（「社会システムが変化機会を提供する責任」）の哲学的位置付け。論文（Discussion）では "consistent with" レベルの言及に留める。

**所有文献**：

| ファイル | 著者 / 出典 | 性格 |
|---|---|---|
| なし — Sen, Nussbaum 一次文献は collected しておらず、ノート上のみ参照 | | |

**Verdict**：❌ **論文用には足りる、著作用には全く足りない**

**何が不足か**：

| 必要だが無い文献領域 | 重要度 | 理由 |
|---|---|---|
| Sen (1999) Development as Freedom | ★ | 古典の core reference |
| Nussbaum (2011) Creating Capabilities | ★ | core reference |
| Capability approach の operationalization 文献（Robeyns 系） | ★ | 政策応用 |
| Personality × capability の架橋論文（もしあれば） | ★ | Discussion での positioning |

**想定検索クエリ**（Tier 3）：
- `Sen capability approach social policy operationalization`
- `Nussbaum capabilities health policy`
- `Robeyns capability approach review`

**論文への影響**：Discussion で "consistent with capability-based perspectives (Sen, 1999; Nussbaum, 2011)" と一言入れる程度。**論文には Tier 3、著作には Tier 1**。

---

## 3. 全体 gap サマリー

### 3.1 Pillar 別 verdict 一覧

| Pillar | 領域 | 既存 | Verdict | 論文への影響度 | 推奨 Tier |
|---|---|---|---|---|---|
| 1 | Harassment epidemiology | 2（測定のみ）| ❌ 致命的不足 | **致命的**：validation の正当化不可 | **Tier 1** |
| 2 | Personality → harassment | 4 件 + 自己引用 | ◯ MEDIUM-STRONG | 中：基盤十分 | Tier 2 |
| 3 | Non-LLM microsimulation | 1 関連弱、Schelling/Bruch は note のみ | ❌ 致命的不足 | **致命的**：novelty 主張不可 | **Tier 1** |
| 4 | Type-conditional prediction | 6 件（clustering） | △ 方法論 ◯、応用 ❌ | 中：clustering 論文の延長で可 | Tier 2 |
| 5 | Harassment intervention reviews | 8 件（anchor 確定） | ◯ 主 anchor は揃う | 中：null-finding 補強でより強く | Tier 2 |
| 6 | Counterfactual modeling | 0 | ❌ ほぼゼロ | 小：Methods で簡潔に触れる程度 | Tier 3 |
| 7 | Self-report validity | 3（うち LLM 系 2）| ❌ 人間文脈薄い | 小：limitation 用 2–3 本 | Tier 2 |
| 8 | Capability approach | 0 | ❌ 論文用は最小限 | 小（論文）/ 大（著作）| Tier 3（論文）/ Tier 1（著作） |

### 3.2 Tier 別検索計画

#### Tier 1（必須、論文の主張を成立させる）— **Pillar 1 + Pillar 3**

| # | 領域 | 検索クエリ候補 | 期待件数 | 主目的 |
|---|---|---|---|---|
| 1.1 | Harassment epidemiology 国際 | `"workplace bullying" prevalence meta-analysis`、`Einarsen NAQ prevalence`、`Notelaers latent class workplace bullying` | 5–10 件 | Phase 1 validation の reference frame |
| 1.2 | Harassment epidemiology 日本 | `"workplace harassment" Japan prevalence`、`厚生労働省 職場のハラスメント 実態調査 2020`（一次資料）| 3–5 件 | 国内文脈、validation の primary target |
| 3.1 | Microsimulation 古典 | `microsimulation epidemiology population health review`、`microsimulation tax benefit policy methodology` | 5–8 件 | 方法論的系譜 |
| 3.2 | ABM workplace / harassment | `agent-based model workplace bullying harassment`、`system dynamics organizational behavior` | 3–5 件 | 直接 precursor が無いことを confirm（novelty 強化）|
| 3.3 | Latent class × population projection | `latent class analysis population prevalence projection` | 3–5 件 | typology-conditional aggregate の系譜 |

**所要見込み**：1 セッション = 1 サブクエリ（5 サブクエリ → **約 5 セッション**、深く読み込むなら 8 セッション）

#### Tier 2（重要、補強）— **Pillar 2 / 4 / 5 / 7**

| # | 領域 | 想定 | 主目的 |
|---|---|---|---|
| 2.1 | HEXACO HH × harassment specific | 2–3 件 | predictor 妥当性補強 |
| 4.1 | 性格類型 → outcome 予測応用 | 3–5 件 | type を介入単位に使う論理 |
| 5.1 | Sexual / power harassment training meta | 2–3 件 | C 構造介入の限界補強 |
| 5.2 | Bystander intervention | 2–3 件 | C 補強 |
| 7.1 | Self-report harassment 妥当性 | 2–3 件 | Limitation 1 対応 |

**所要見込み**：約 3–5 セッション（Tier 1 完了後）

#### Tier 3（後でも可）— **Pillar 6 / 8**

| # | 領域 | 想定 | 主目的 |
|---|---|---|---|
| 6.1 | Counterfactual methodology | 2 件 | Methods framing |
| 8.1 | Capability approach | 1–2 件（Sen、Nussbaum 一次） | Discussion |

**所要見込み**：1 セッション

### 3.3 検索 → 取得 → 統合のワークフロー（提案）

各 Tier 1 サブクエリで以下を回す：

1. **Web 検索**：Google Scholar / 通常 Web で候補リストを作成（タイトル + 著者 + DOI/URL）
2. **Triage**：本研究との関連性を 4 段階（Core / Strong / Peripheral / Reject）で評価。Reject 理由も記録
3. **取得**：Core / Strong は PDF 取得を試みる（OA で読めれば PDF を `simulation/prior_research/` に追加）
4. **要約ノート**：1 PDF = 1 要約（research question / method / key finding / 本研究での citation 用途）
5. **reference_index.md 更新**：✅/⚠️ ステータスで管理

### 3.4 「やらないこと」の明示

- **網羅的 systematic review はやらない**：本研究は simulation 論文であり、literature review 論文ではない。各 Pillar は **claim を支える最小限の文献群**で足りる
- **古い文献の untargeted な収集はしない**：1990 年代以前は古典（Schelling 等）以外は除外
- **Tier 3 を Tier 1 より先に進めない**：論文の核（novelty + validation）から固める
- **Pillar 8（capability）を論文に深入りさせない**：Doc 2 の 4 層分離原則（research_plan Part 0.4）に従い、論文では "consistent with" 程度に留める

---

## 4. 次のステップ

このドキュメントを commit/push 後、ユーザーに以下を確認：

1. **Tier 1 のスコープと優先順位は適切か？**（特に Pillar 1 vs Pillar 3 の優先度）
2. **検索を進める前に、Pillar の追加 / 削除 / 統合があるか？**
3. **Tier 1 の最初のサブクエリ（推奨：1.1 harassment epidemiology 国際）から進めて良いか？**
4. **所要セッション数（Tier 1 で 5–8 セッション）は許容できるか？**
5. **PDF 取得方針**：OA で取れない有料文献はメタデータのみ記録 / 後日対応で良いか？







