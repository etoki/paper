# Tier 1 検索結果 — PDF 取得用論文リスト

実施日：2026-04-27
ブランチ：`claude/review-harassment-research-plan-Dy2eo`
目的：`literature_audit.md` で Tier 1 とした 5 サブクエリについて、ユーザーが手動で PDF 取得できるよう書誌情報を整理
関連：`literature_audit.md`、`research_plan_harassment_typology_simulation.md`

---

## 取得方針（ユーザー作業）

- **Core**：必須。OA で取れない場合も他経路で取得を試みる
- **Strong**：取得を試みる、取れなければメタデータのみで可
- **Peripheral**：参考。OA で簡単に取れる範囲で
- 取得した PDF は `simulation/prior_research/` に追加し、`reference_index.md` に追記

書誌情報の信頼度マーク：
- ✅ WebSearch / WebFetch で書誌情報を直接確認済
- ⚠️ 主要書誌情報は確認済だが DOI / 一部詳細は未検証
- 📖 標準知識ベース、取得時に再確認推奨

---

## サブクエリ 3.1：Microsimulation 古典（Pillar 3 — Non-LLM simulation lineage）

**目的**：「LLM ではない確率論的 microsimulation が社会科学・公衆衛生で確立した方法論である」ことを示す系譜の祖となる文献を確保。本研究を Park 2024 の劣化版ではなく、**microsimulation の harassment 領域への応用**として positioning する。

### 検索クエリ（実行済み）

- `Orcutt 1957 "new type of socio-economic system" microsimulation`
- `Spielauer 2011 "what is social science microsimulation"`
- `Rutter Zaslavsky Feuer 2011 dynamic microsimulation health outcomes`
- `Krijkamp Alarid-Escudero microsimulation R tutorial`
- `Bonabeau 2002 agent-based modeling PNAS`
- `Macal North 2010 tutorial agent-based modelling`
- `Bruch Atwell 2015 agent-based models empirical social research`
- `brief global history microsimulation models health`
- `Schelling 1971 dynamic models segregation`

### 取得対象論文（9 件）

#### Core（必須、4 件）

**[3.1-C1]** ✅ Orcutt, G. H. (1957). A new type of socio-economic system. *Review of Economics and Statistics, 39*(2), 116–123.
- **役割**：microsimulation の founding paper。「個人意思決定単位を集約して社会経済システムを記述する」という発想の祖
- **本研究での使用**：Introduction で「microsimulation は 1957 年 Orcutt 以来の系譜であり、本研究はその harassment 領域への適用である」
- **取得**：International Journal of Microsimulation で再掲（OA） https://microsimulation.pub/articles/00002

**[3.1-C2]** ✅ Spielauer, M. (2011). What is social science microsimulation? *Social Science Computer Review, 29*(1), 9–20. https://doi.org/10.1177/0894439310370085
- **役割**：microsimulation の現代的レビュー。social science での運用を定義
- **本研究での使用**：Methods 冒頭「microsimulation の定義」、Introduction で 3 driving forces（policy demand, individual-level paradigm, computational power）の引用
- **取得**：SAGE 査読版（要購読）。著者個人サイトに preprint がある可能性

**[3.1-C3]** ✅ Rutter, C. M., Zaslavsky, A. M., & Feuer, E. J. (2011). Dynamic microsimulation models for health outcomes: A review. *Medical Decision Making, 31*(1), 10–18. https://doi.org/10.1177/0272989X10369005
- **役割**：health 領域での microsimulation review。calibration / validation / sensitivity analysis のフレームを提供
- **本研究での使用**：Phase 1 の MAPE / triangulation 設計の方法論的根拠
- **取得**：PMC で OA 版あり https://pmc.ncbi.nlm.nih.gov/articles/PMC3404886/

**[3.1-C4]** ✅ Bruch, E., & Atwell, J. (2015). Agent-based models in empirical social research. *Sociological Methods & Research, 44*(2), 186–221. https://doi.org/10.1177/0049124113506405
- **役割**：sociological ABM のレビュー。empirical data との接続を主題化
- **本研究での使用**：「N=354 の empirical data を simulation の確率テーブルに使う」設計の方法論的正当化
- **取得**：PMC で OA 版あり https://pmc.ncbi.nlm.nih.gov/articles/PMC4430112/

#### Strong（推奨、3 件）

**[3.1-S1]** ✅ Bonabeau, E. (2002). Agent-based modeling: Methods and techniques for simulating human systems. *PNAS, 99*(suppl. 3), 7280–7287. https://doi.org/10.1073/pnas.082080899
- **役割**：ABM 古典 review。flow / organizational / market / diffusion の 4 領域整理
- **本研究での使用**：「ABM は organizational behavior に適用可能」の根拠
- **取得**：PNAS で OA https://www.pnas.org/doi/10.1073/pnas.082080899

**[3.1-S2]** ✅ Krijkamp, E. M., Alarid-Escudero, F., Enns, E. A., Jalal, H. J., Hunink, M. G. M., & Pechlivanoglou, P. (2018). Microsimulation modeling for health decision sciences using R: A tutorial. *Medical Decision Making, 38*(3), 400–422. https://doi.org/10.1177/0272989X18754513
- **役割**：R による microsimulation 実装の tutorial。コード公開（DARTH-git/Microsimulation-tutorial）
- **本研究での使用**：実装段階のリファレンス（コード設計の参考）+ 「reproducibility のための tutorial 系統」への positioning
- **取得**：SAGE（要購読）。GitHub にコードあり https://github.com/DARTH-git/Microsimulation-tutorial

**[3.1-S3]** ✅ Schofield, D. J., Zeppel, M. J. B., Tan, O., Lymer, S., Cunich, M. M., & Shrestha, R. N. (2018). A brief, global history of microsimulation models in health: Past applications, lessons learned and future directions. *International Journal of Microsimulation, 11*(1), 97–142.
- **役割**：health microsimulation の歴史 review
- **本研究での使用**：Introduction で「health 領域での microsimulation 普及」を簡潔にまとめる際の総合引用
- **取得**：OA https://microsimulation.pub/articles/00175

#### Peripheral（参考、2 件）

**[3.1-P1]** ✅ Macal, C. M., & North, M. J. (2010). Tutorial on agent-based modelling and simulation. *Journal of Simulation, 4*(3), 151–162. https://doi.org/10.1057/jos.2010.3
- **役割**：ABM の汎用 tutorial（Bonabeau 2002 の補完）
- **本研究での使用**：ABM 概念の入門引用（Bruch & Atwell 2015 で代替可）
- **取得**：Springer / Taylor & Francis 経由（要購読）。Iowa State の faculty サイトに著者 PDF あり

**[3.1-P2]** ✅ Schelling, T. C. (1971). Dynamic models of segregation. *Journal of Mathematical Sociology, 1*(2), 143–186. https://doi.org/10.1080/0022250X.1971.9989794
- **役割**：ABM 的思考の祖（micromotive → macrobehavior）。reference_index.md の Schelling 1978 書籍と同系統
- **本研究での使用**：「個人選好が集団 outcome を生成する」古典として簡潔に引用
- **取得**：UZH などの講義サイトに preprint あり

### 既存ライブラリで補完される文献

- Schelling (1978) Micromotives and Macrobehavior — `reference_index.md` に記載済（PDF 未取得）
- 本研究では Schelling 1971 article + 1978 book の片方を引用すれば足りる

### サブクエリ 3.1 の verdict

✅ **本サブクエリは充足**。Microsimulation 系譜の祖（Orcutt 1957）、現代 review（Spielauer 2011、Rutter 2011）、ABM 文献（Bruch & Atwell 2015、Bonabeau 2002）、health 領域 history（Schofield 2018）、実装 tutorial（Krijkamp 2018）、ABM 古典（Schelling 1971）の 7 系統が揃う。

論文での positioning は次の構造を想定：

> "Microsimulation, originating in Orcutt's (1957) framework for socio-economic systems and now widely used in health policy (Rutter et al., 2011; Schofield et al., 2018) and social science (Bruch & Atwell, 2015; Spielauer, 2011), provides a probabilistic, individual-level lineage distinct from recent LLM-based agent simulations (Park et al., 2024). The present study applies this lineage to workplace harassment for the first time."

---

## サブクエリ 3.2：ABM for workplace bullying / harassment（precursor 確認）

**目的**：「workplace bullying / harassment を ABM・microsimulation で扱った先行研究」を網羅的に把握。novelty 主張のため、**直接の precursor が無いこと**または**precursor とは異なる問題設定であること**を明示する。

### 検索クエリ（実行済み）

- `"agent-based model" workplace bullying simulation organizational behavior`
- `"system dynamics" workplace bullying mobbing model simulation`
- `Sapouna Wolke 2013 agent-based model bullying school simulation`
- `Merlone Argentero modelling dysfunctional behaviours workplace mobbing`
- `"workplace harassment" "agent-based" OR "microsimulation" population national prevalence model`
- `"personality" agent-based model workplace counterproductive aggression simulation`

### 取得対象論文（4 件）

#### Core（必須、2 件）

**[3.2-C1]** ✅ Ho, C.-H., Campenni, M., Manolchev, C., Lewis, D., & Mustafee, N. (2025). Exploring the coping strategies of bullying targets in organisations through abductive reasoning: An agent-based simulation approach. *Journal of Business Ethics, 199*(4). https://doi.org/10.1007/s10551-024-05861-2
- **役割**：**最も近い直接 precursor**。UK NHS のデータで bullying targets の coping strategy を ABSS でモデル化
- **本研究との差**：
  - 対象が **target side**（被害者の対処戦略）であり、**perpetrator side**（加害確率）ではない
  - 単一組織のミクロ動態であり、**国レベル aggregate prevalence** ではない
  - **personality を入力としない**（perceived organizational support と TU membership が主変数）
- **本研究での使用**：Introduction で「workplace bullying 領域での ABM 応用は coping strategy 分析（Ho et al., 2025）等に限られ、加害発生率の population-scale projection は未開拓」と明示
- **取得**：Springer（要購読）。Exeter 大学 repository に preprint 候補 https://ore.exeter.ac.uk/repository/handle/10871/137934

**[3.2-C2]** ✅ Merlone, U., & Argentero, P. (2018). Modelling dysfunctional behaviours in organizations: The case of workplace mobbing/bullying. In *Handbooks of Workplace Bullying, Emotional Abuse and Harassment, Vol. 1: Concepts, Approaches and Methods*. Springer. https://doi.org/10.1007/978-981-10-5334-4_15-1
- **役割**：system dynamics による mobbing/bullying モデル化の章。Leymann の seminal paper を 3 段階の系統で modeling
- **本研究との差**：
  - **causal loop / qualitative dynamics** の formalization であり、**probabilistic prevalence projection** ではない
  - **personality typology を input としない**
  - empirical data との照合が中心ではない
- **本研究での使用**：「workplace bullying の computational modeling は qualitative system dynamics（Merlone & Argentero, 2018）の系譜があるが、empirically calibrated probabilistic microsimulation は本研究が初」
- **取得**：Springer Handbook（要購読）

#### Strong（推奨、2 件）

**[3.2-S1]** ✅ Bowes, L., Maughan, B., Ball, H., Shakoor, S., Ouellet-Morin, I., Caspi, A., Moffitt, T. E., & Arseneault, L. (2013). Chronic bullying victimization across school transitions: The role of genetic and environmental influences. *Development and Psychopathology, 25*(2), 333–346. — 関連、確認要
- **代替候補**：Eslea & Smith (1998) school bullying / Sapouna et al. (2010) "Virtual learning intervention to reduce bullying victimization in primary school" *J Child Psychol Psychiatry* — school 領域の review として参照可能
- **役割**：bullying 領域での computational / experimental modeling の異なる aspect（school setting, longitudinal）。**workplace と異なる文脈**であることを明示
- **本研究での使用**：「bullying の computational modeling は school setting（Sapouna et al., 2010 の virtual learning intervention 等）が中心であり、workplace × population scale は未開拓」

**[3.2-S2]** ⚠️ Tucker, M. K., Jimmieson, N. L., & Bordia, P. (2013). Modeling workplace bullying using catastrophe theory. *Nonlinear Dynamics, Psychology, and Life Sciences* — タイトル正確性要確認
- **役割**：cusp catastrophe model で workplace bullying を nonlinear に modeling
- **本研究との差**：linear と nonlinear の比較で linear が勝ち、結果として nonlinear modeling の有用性が限定的であることを示す
- **本研究での使用**：「workplace bullying の statistical modeling は catastrophe theory 等が試みられたが、近接的にも linear が優位」と参照
- **取得**：PubMed (PMID: 24011118) https://pubmed.ncbi.nlm.nih.gov/24011118/

### サブクエリ 3.2 の verdict

✅ **novelty 主張は成立**。本サブクエリの主目的は「直接 precursor が無いことを confirm」だが、3 件の近接研究が見つかった。それぞれ：

| 既存研究 | 何を扱うか | 本研究との差異 |
|---|---|---|
| Ho et al. (2025) | Target coping strategy (NHS、ABSS) | 対象が target、scale が単一組織 |
| Merlone & Argentero (2018) | System dynamics（causal loops）| Qualitative、非 probabilistic、population scale でない |
| Tucker et al. (2013) | Cusp catastrophe model | Statistical model、simulation でない |

**結論**：「**HEXACO 7 類型の確率テーブル** × **国レベル aggregate prevalence** × **3 種介入 counterfactual**」の組合せを行った先行研究は無い。論文の novelty 主張は 3 つの異なる軸（typological input / population aggregate / counterfactual intervention design）で支えられる。

**論文での positioning（Introduction 後半案）**：

> "While agent-based and system-dynamics models have been applied to workplace bullying (Ho et al., 2025; Merlone & Argentero, 2018), they have focused on within-organization dynamics—target coping strategies, organizational feedback loops—rather than national-scale prevalence projection. Likewise, no prior work has used HEXACO-based personality typology as input to such simulations. The present study fills this gap by combining type-conditional probability tables with public statistics-based scaling, and by examining intervention counterfactuals at the population level."

---

## サブクエリ 3.3：Latent class × population prevalence projection

**目的**：「latent class / cluster で同定した subgroup ごとに outcome 率を推定し、population-level に集約する」方法論の最近接系譜を確保。本研究の **7 類型 × cell-conditional probability × aggregate scaling** の方法論的祖となる。

### 検索クエリ（実行済み）

- `Lanza Rhoades 2013 latent class analysis subgroup prevention treatment`
- `Notelaers Einarsen latent class cluster workplace bullying validity Work Stress`
- `Notelaers Vermunt Baillien Einarsen exploring risk groups workplace bullying Industrial Health`

### 取得対象論文（3 件）

#### Core（必須、3 件すべて）

**[3.3-C1]** ✅ Lanza, S. T., & Rhoades, B. L. (2013). Latent class analysis: An alternative perspective on subgroup analysis in prevention and treatment. *Prevention Science, 14*(2), 157–168. https://doi.org/10.1007/s11121-011-0201-1
- **役割**：LCA を介入科学（prevention / treatment）の subgroup 分析に応用する方法論的 review
- **本研究との対応**：「7 類型を介入の差別的効果分析の単位として使う」設計の方法論的祖
- **本研究での使用**：
  - Phase 1 の type-conditional 確率テーブル設計の正当化
  - Phase 2 の Counterfactual B（高リスク類型への targeted 介入）の論理的根拠
  - 「Type I error / power の課題は LCA で部分的に対応可能」と limitation 議論
- **取得**：PMC で OA 版あり https://pmc.ncbi.nlm.nih.gov/articles/PMC3173585/

**[3.3-C2]** ✅ Notelaers, G., Einarsen, S., De Witte, H., & Vermunt, J. K. (2006). Measuring exposure to bullying at work: The validity and advantages of the latent class cluster approach. *Work & Stress, 20*(4), 289–302. https://doi.org/10.1080/02678370601071594
- **役割**：N=6,175 ベルギーサンプルで NAQ を LCA → 6 つの被害 exposure groups を同定。**bullying 領域での typology-based prevalence の最近接系譜**
- **本研究との対応**：被害側の 6 群 typology に対応する形で、本研究は **加害側の 7 類型** を性格ベースで構築
- **本研究での使用**：「Notelaers et al. (2006) は被害体験の typology を NAQ から推定したが、加害確率の personality-based typology は未検討」と novelty 主張
- **取得**：Tilburg University repository に PDF（OA 候補）https://research.tilburguniversity.edu/en/publications/measuring-exposure-to-bullying-at-work-the-validity-and-advantage

**[3.3-C3]** ✅ Notelaers, G., Vermunt, J. K., Baillien, E., Einarsen, S., & De Witte, H. (2011). Exploring risk groups and risk factors for workplace bullying. *Industrial Health, 49*(1), 73–88. https://doi.org/10.2486/indhealth.MS1155
- **役割**：上記の続編。large, heterogeneous sample で **risk group prevalence の population estimate**（30.5% not bullied / 27.2% limited criticism / 20.8% limited negative / 8.3% occasionally bullied 等の 6 群）
- **本研究との対応**：本研究は personality-based 加害確率を使って同等の **加害側 prevalence breakdown** を国レベルで再構築
- **本研究での使用**：「victim-side prevalence は LCA で 6 群が同定済（Notelaers et al., 2011）。本研究は perpetrator-side で対応する 7 群を personality typology から構築」
- **取得**：J-STAGE で OA https://www.jstage.jst.go.jp/article/indhealth/49/1/49_MS1155/_article

### サブクエリ 3.3 の verdict

✅ **本サブクエリは十分に充足**（3 件で核は揃う）。

**論文での positioning（Methods 冒頭案）**：

> "Latent class methods have established the validity of identifying typological subgroups for both differential treatment effects (Lanza & Rhoades, 2013) and workplace bullying victim profiles (Notelaers et al., 2006, 2011). The present study extends this approach to the perpetrator side by deriving HEXACO-based personality types from a large reference population (N=13,668) and estimating type-conditional perpetration probabilities from an independent sample (N=354)."

---

## サブクエリ 1.1：Workplace harassment epidemiology 国際

**目的**：「workplace bullying / harassment は世界的に研究されており、prevalence の reference frame が国際的に確立している」ことを示す。Phase 1 で厚労省統計を validation target に使う妥当性を学術的に裏付ける。

### 検索クエリ（実行済み）

- `Nielsen Matthiesen Einarsen 2010 impact methodological moderators prevalence rates workplace bullying meta-analysis`
- `Bowling Beehr 2006 workplace harassment victim's perspective theoretical model meta-analysis Journal Applied Psychology`
- `Einarsen Hoel Notelaers 2009 Negative Acts Questionnaire-Revised validity factor structure psychometric Work Stress`
- `"exposure to workplace harassment" "Five Factor Model" personality meta-analysis 2017`
- `ILO 2022 experiences of violence and harassment at work global first survey prevalence`

### 取得対象論文（5 件）

#### Core（必須、4 件）

**[1.1-C1]** ✅ Nielsen, M. B., Matthiesen, S. B., & Einarsen, S. (2010). The impact of methodological moderators on prevalence rates of workplace bullying. A meta-analysis. *Journal of Occupational and Organizational Psychology, 83*(4), 955–979. https://doi.org/10.1348/096317909X481256
- **役割**：**workplace bullying prevalence の決定的 meta-analysis**。86 sample / N=130,973 を統合
- **Key numbers**：self-labeling (with definition) 11.3% / behavioral 14.8% / self-labeling (no definition) 18.1%
- **本研究での使用**：
  - Phase 1 introduction で「international prevalence baseline」として引用
  - 厚労省実態調査（self-labeling）の数値を国際 baseline と比較する根拠
  - 「測定方法によって prevalence が大きく変動する」という limitation 議論
- **取得**：Wiley（要購読）。ResearchGate に PDF 候補

**[1.1-C2]** ✅ Bowling, N. A., & Beehr, T. A. (2006). Workplace harassment from the victim's perspective: A theoretical model and meta-analysis. *Journal of Applied Psychology, 91*(5), 998–1012. https://doi.org/10.1037/0021-9010.91.5.998
- **役割**：harassment の **antecedents / consequences の決定的 meta-analysis**。attribution × reciprocity model 提示
- **本研究との対応**：本研究は **antecedent** 側（personality）を扱う。Bowling & Beehr の枠組みで本研究の input 側を位置付け可能
- **本研究での使用**：「harassment は environmental + individual difference factors の両方が antecedent」（Bowling & Beehr, 2006）の引用、本研究は personality individual difference の一形態
- **取得**：APA PsycNET（要購読）。PMID: 16953764

**[1.1-C3]** ✅ Einarsen, S., Hoel, H., & Notelaers, G. (2009). Measuring exposure to bullying and harassment at work: Validity, factor structure and psychometric properties of the Negative Acts Questionnaire-Revised. *Work & Stress, 23*(1), 24–44. https://doi.org/10.1080/02678370902815673
- **役割**：**NAQ-R の確立論文**。N=5,288 UK サンプルで personal / work-related / physically intimidating の 3 因子構造を確立
- **本研究での使用**：
  - 「Workplace harassment 測定の international gold standard」として引用
  - 本研究の日本語 power harassment scale（Tou et al., 2017）と gender harassment scale（Kobayashi & Tanaka, 2010）を NAQ-R 系統に位置付け
  - measurement validity の議論で参照
- **取得**：Taylor & Francis（要購読）

**[1.1-C4]** ✅ International Labour Organization (ILO). (2022). *Experiences of violence and harassment at work: A global first survey*. ILO/Lloyd's Register Foundation/Gallup. https://www.ilo.org/publications/major-publications/experiences-violence-and-harassment-work-global-first-survey
- **役割**：**初の global prevalence survey**。74,000 workers / 121 countries / 23% prevalence
- **Key numbers**：psychological 17.9% / physical 8.5% / sexual 6.3%（global lifetime）
- **本研究での使用**：
  - Introduction の opening で「workplace violence and harassment は global health concern」の根拠
  - 厚労省 prevalence 数値を ILO global baseline と比較
- **取得**：ILO 公式 PDF（OA） https://www.ilo.org/sites/default/files/wcmsp5/groups/public/@dgreports/@dcomm/documents/publication/wcms_863095.pdf

#### Strong（推奨、1 件 — Pillar 2 との架橋）

**[1.1-S1]** ✅ Nielsen, M. B., Glasø, L., & Einarsen, S. (2017). Exposure to workplace harassment and the Five Factor Model of personality: A meta-analysis. *Personality and Individual Differences, 104*, 195–206. https://doi.org/10.1016/j.paid.2016.08.015
- **役割**：FFM × harassment exposure の meta-analysis。N=13,896 / 36 samples
- **Key numbers**：Neuroticism r=.25**, Agreeableness r=−.17**, Extraversion r=−.10*, Conscientiousness r=−.10*, Openness r=.04 ns
- **本研究との対応**：**target side** の personality predictor。本研究は perpetrator side だが、HEXACO の Agreeableness × harassment 関係の根拠として引用可能。**Pillar 2 と併用**
- **本研究での使用**：「victim 側の Big Five 効果は確立しているが（Nielsen et al., 2017）、perpetrator 側の HEXACO typology による prevalence projection は未確立」と novelty 主張
- **取得**：BI Open archive で著者版 PDF が OA https://biopen.bi.no/bi-xmlui/bitstream/handle/11250/2400459/NielsenGlasoEinarsen_2017_PaID.pdf

### サブクエリ 1.1 の verdict

✅ **本サブクエリは充足**。International epidemiology の reference frame が完全に揃った：
- Prevalence meta（Nielsen 2010）
- Antecedent / consequence meta（Bowling & Beehr 2006）
- 測定 gold standard（Einarsen, Hoel, & Notelaers 2009 NAQ-R）
- Global survey（ILO 2022）
- Personality predictor meta（Nielsen, Glasø, & Einarsen 2017、Pillar 2 と橋渡し）

**論文での positioning（Introduction opening 案）**：

> "Workplace harassment affects approximately 23% of employed adults globally (ILO, 2022), making it a substantial public health and organizational concern. Meta-analytic evidence places the prevalence of workplace bullying at 11–18% depending on measurement methodology (Nielsen, Matthiesen, & Einarsen, 2010), with both individual difference and environmental factors contributing to its occurrence (Bowling & Beehr, 2006). Personality traits—particularly neuroticism and low agreeableness—are established predictors of harassment exposure (Nielsen, Glasø, & Einarsen, 2017). However, the question of whether population-scale harassment prevalence can be reproduced from individual-level personality typology remains untested..."

---

## サブクエリ 1.2：Japan workplace harassment prevalence（厚労省 + 査読論文）

**目的**：Phase 1 の **primary validation target（厚労省実態調査）の一次資料**確保と、**日本の workplace harassment 査読研究**の reference frame 構築。

### 検索クエリ（実行済み）

- `厚生労働省 職場のハラスメント 実態調査 令和2年度 2020 報告書`
- `Tsuno Kawakami Japanese version Negative Acts Questionnaire workplace bullying validity Journal Occupational Health`
- `Tsuno Kawakami Tsutsumi socioeconomic determinants of bullying in the workplace Japan national representative`
- `Tsuno Kawakami workplace bullying Japan COVID-19 prevalence 2021 nationwide internet survey`

### 取得対象論文（5 件）

#### Core（必須、3 件）

**[1.2-C1]** ✅ 厚生労働省 (2021). *令和2年度厚生労働省委託事業 職場のハラスメントに関する実態調査 報告書*. 東京海上日動リスクコンサルティング株式会社.
- **役割**：**Phase 1 の primary validation target**。日本全国規模の workplace harassment 実態調査（過去 3 年間）
- **Key numbers**：パワハラ経験 31.4%（2016 年 32.5% から微減）/ セクハラ 10.2% / 顧客等からの著しい迷惑行為 15.0%
- **本研究での使用**：Phase 1 simulation 出力との triangulation（MAPE ≤ 30% 主成功基準）
- **取得**：
  - 概要版 PDF（OA、NDL warp 経由）：https://warp.da.ndl.go.jp/info:ndljp/pid/14120359/www.mhlw.go.jp/content/11200000/000783140.pdf
  - 厚労省ハラスメント対策ページ：https://www.mhlw.go.jp/stf/seisakunitsuite/bunya/0000165756.html
  - 全文版（3.7MB）も同ページから取得可能

**[1.2-C2]** ✅ Tsuno, K., Kawakami, N., Inoue, A., & Abe, K. (2010). Measuring workplace bullying: Reliability and validity of the Japanese version of the Negative Acts Questionnaire. *Journal of Occupational Health, 52*(4), 216–226. https://doi.org/10.1539/joh.L10036
- **役割**：**日本語版 NAQ-R の妥当化論文**。N=830 男性 + 796 女性の civil servants で α=0.91–0.95 を確立
- **本研究での使用**：
  - 「日本における workplace bullying 測定の international compatibility」の根拠
  - 本研究の日本語 power harassment / gender harassment scales（Tou et al., 2017; Kobayashi & Tanaka, 2010）を NAQ-R 系統に位置付ける橋渡し
- **取得**：J-STAGE で OA https://www.jstage.jst.go.jp/article/joh/52/4/52_L10036/_article、Oxford Academic でも公開 https://academic.oup.com/joh/article/52/4/216/7270253

**[1.2-C3]** ✅ Tsuno, K., Kawakami, N., Tsutsumi, A., Shimazu, A., Inoue, A., Odagiri, Y., Yoshikawa, T., Haratani, T., Shimomitsu, T., & Kawakami, N. (2015). Socioeconomic determinants of bullying in the workplace: A national representative sample in Japan. *PLoS ONE, 10*(3), e0119435. https://doi.org/10.1371/journal.pone.0119435
- **役割**：**日本の national representative sample（N=5,000、20–60 歳）による prevalence + determinants 研究**
- **Key numbers**：被害 6%、目撃 15%（NAQ-R, past 6 months 基準）
- **本研究との対応**：本研究の N=354 を national representative の Tsuno et al. 2015 と prevalence 比較 → calibration / external validity 議論
- **本研究での使用**：
  - Phase 1 で 「self-report harassment N=354 → 国レベル aggregate」の compositional 妥当化
  - 「日本の workplace bullying 被害 prevalence は 6–15% range（Tsuno et al., 2015）」と引用
- **取得**：PLoS ONE OA https://journals.plos.org/plosone/article?id=10.1371/journal.pone.0119435、PMC でも OA https://pmc.ncbi.nlm.nih.gov/articles/PMC4353706/

#### Strong（推奨、2 件）

**[1.2-S1]** ⚠️ 厚生労働省 (2024). *令和5年度厚生労働省委託事業 職場のハラスメントに関する実態調査 結果概要*. （2024-05-17 公表）
- **役割**：**最新の同シリーズ調査**。2020 年度版との時系列比較が可能
- **本研究での使用**：sensitivity analysis で「2020 年度 vs 2023 年度の prevalence 推移」と本研究の予測値を併せて triangulate（厳密には別 cross-section だが trend 評価可能）
- **取得**：厚労省公式 https://www.mhlw.go.jp/content/11909000/001259093.pdf

**[1.2-S2]** ✅ Tsuno, K., Hori, A., Tabuchi, T., et al. (2022). Risk factors for workplace bullying, severe psychological distress, and suicidal ideation during the COVID-19 pandemic among the general working population in Japan: A large-scale cross-sectional study. *BMJ Open* (or related venue — exact venue 要確認). https://www.medrxiv.org/content/10.1101/2021.11.18.21266501v1.full
- **役割**：COVID-19 文脈での Japan workplace bullying prevalence + risk factor research（large-scale cross-sectional, JACSIS）
- **本研究での使用**：「日本の workplace bullying は近年データが充実（Tsuno et al., 2022）。本研究はこれを population-scale simulation 視点から補完」
- **取得**：medRxiv preprint OA。published version は要確認（PMC https://pmc.ncbi.nlm.nih.gov/articles/PMC9638740/ も参照）

### サブクエリ 1.2 の verdict

✅ **本サブクエリは充足**。日本語文脈で必要なものはすべて確保：
- **Validation target 一次資料**（厚労省 2021、最新版 2024）
- **測定 validity の international bridge**（Tsuno et al., 2010）
- **Japan national prevalence の peer-reviewed source**（Tsuno et al., 2015、PLoS ONE）
- **最近の context**（Tsuno et al., 2022 COVID 関連）

**論文での positioning（Introduction Japan-context paragraph 案）**：

> "In Japan, workplace harassment is a well-documented public concern. The Ministry of Health, Labour and Welfare's nationwide survey reported that 31.4% of workers experienced power harassment and 10.2% experienced sexual harassment in the past three years (MHLW, 2021). Independent peer-reviewed estimates using the Japanese Negative Acts Questionnaire (Tsuno et al., 2010) place the past-six-month bullying prevalence at 6% in a national representative sample (Tsuno et al., 2015), with elevated rates observed during the COVID-19 pandemic (Tsuno et al., 2022). The present study uses the MHLW national survey as a primary validation target for our type-conditional simulation."

---

## 取得チェックリスト（5 sub-queries 統合、合計 26 件）

ユーザー手動取得用。✓ を入れて進捗管理してください。

### 取得状況サマリー（2026-04-27 更新）

**24 / 26 件取得完了（main commit `314ceb9` + `a09dcc8`、保存先：`metaanalysis/prior_research/`）**

未取得：
- ✗ **[3.2-C2]** Merlone & Argentero (2018) — Springer Handbook 章、paywall 強い。**重要度：低〜中、抄録ベースの間接引用で代替可**
- ✗ **[3.2-S2]** Tucker et al. (2013) catastrophe theory — paywalled。**重要度：低、引用しなくても論文の主張に影響なし**

→ **Tier 1 文献収集は実質完了**。残り 2 件は論文の核には不要（peripheral）。

### 取得済み（24 件、`metaanalysis/prior_research/` に保存）

#### Pillar 3 — 非 LLM simulation 系譜（11 件）
- [x] **[3.1-C1]** Orcutt (1957) "A new type of socio-economic system" *Rev Econ Stat 39(2)*
- [x] **[3.1-C2]** Spielauer (2011) "What is Social Science Microsimulation?" *SSCR 29(1)*
- [x] **[3.1-C3]** Rutter, Zaslavsky, & Feuer (2011) "Dynamic microsimulation models for health outcomes: A review" *MDM 31(1)*
- [x] **[3.1-C4]** Bruch & Atwell (2015) "Agent-based models in empirical social research" *Sociol Methods Res 44(2)*
- [x] **[3.1-S1]** Bonabeau (2002) "Agent-based modeling: Methods and techniques" *PNAS 99(suppl 3)*
- [x] **[3.1-S2]** Krijkamp et al. (2018) "Microsimulation modeling for health decision sciences using R" *MDM 38(3)*
- [x] **[3.1-S3]** Schofield et al. (2018) "Brief, global history of microsimulation models in health" *IJM 11(1)*
- [x] **[3.1-P1]** Macal & North (2010) "Tutorial on agent-based modelling and simulation" *J Simulation 4(3)*
- [x] **[3.1-P2]** Schelling (1971) "Dynamic models of segregation" *J Math Sociol 1(2)*
- [x] **[3.2-C1]** Ho et al. (2025) "Exploring the coping strategies of bullying targets... ABSS" *JBE*
- [x] **[3.2-S1]** Sapouna et al. (2010) "Virtual learning intervention to reduce bullying victimization" *J Child Psychol Psychiatry*
- [x] **[3.3-C1]** Lanza & Rhoades (2013) "Latent class analysis: subgroup analysis in prevention" *Prev Sci 14(2)*
- [x] **[3.3-C2]** Notelaers et al. (2006) "Measuring exposure to bullying at work: latent class cluster" *Work & Stress 20(4)*
- [x] **[3.3-C3]** Notelaers et al. (2011) "Exploring risk groups workplace bullying" *Ind Health 49(1)*

#### Pillar 1 — Harassment epidemiology（10 件）
- [x] **[1.1-C1]** Nielsen, Matthiesen, & Einarsen (2010) "Methodological moderators on prevalence rates of workplace bullying" *JOOP 83(4)*
- [x] **[1.1-C2]** Bowling & Beehr (2006) "Workplace harassment from the victim's perspective" *JAP 91(5)*
- [x] **[1.1-C3]** Einarsen, Hoel, & Notelaers (2009) "Negative Acts Questionnaire-Revised" *Work & Stress 23(1)*
- [x] **[1.1-C4]** ILO (2022) "Experiences of violence and harassment at work: A global first survey"
- [x] **[1.1-S1]** Nielsen, Glasø, & Einarsen (2017) "Exposure to workplace harassment and FFM" *PAID 104*
- [x] **[1.2-C1]** 厚労省 (2021) 令和2年度 職場のハラスメントに関する実態調査 報告書（概要版）
- [x] **[1.2-C2]** Tsuno et al. (2010) "Japanese version of NAQ" *J Occup Health 52(4)*
- [x] **[1.2-C3]** Tsuno et al. (2015) "Socioeconomic determinants of bullying in Japan" *PLoS ONE 10(3)*
- [x] **[1.2-S1]** 厚労省 (2024) 令和5年度 職場のハラスメントに関する実態調査 結果概要
- [x] **[1.2-S2]** Tsuno et al. (2022) "Risk factors for workplace bullying during COVID-19 in Japan"

### 未取得（2 件、論文の核には不要）

- [ ] **[3.2-C2]** Merlone & Argentero (2018) "Modelling Dysfunctional Behaviours in Organizations: Mobbing/Bullying" Springer Handbook ch. — paywall 強い、抄録ベースの間接引用で代替可
- [ ] **[3.2-S2]** Tucker et al. (2013) "Modeling workplace bullying using catastrophe theory" — peripheral、引用不要

---

## 全体サマリーと next steps

### Tier 1 リテラチャー強化の verdict

| Pillar | Sub-query | 取得対象数 | OA 件数 | Verdict |
|---|---|---|---|---|
| 3 | 3.1 Microsimulation 古典 | 9 | 5 | ✅ 充足 |
| 3 | 3.2 ABM workplace harassment | 4 | 0–1 | ✅ novelty 主張成立 |
| 3 | 3.3 Latent class projection | 3 | 2 | ✅ 充足 |
| 1 | 1.1 国際 epidemiology | 5 | 2 | ✅ 充足 |
| 1 | 1.2 日本 prevalence | 5 | 4 | ✅ 充足 |
| **合計** | **5 sub-queries** | **26** | **13** | ✅ **Tier 1 充足** |

### 論文骨格への含意

これらが揃うと、Introduction の論理は以下の構造で書ける：

1. **Opening（global concern）**：ILO 2022、Nielsen et al. 2010 → workplace harassment は 23% global prevalence の health concern
2. **Japan context**：MHLW 2021、Tsuno et al. 2015、Tsuno et al. 2010 → 日本でもパワハラ 31.4% / 被害 6% national representative
3. **Predictor lineage**：Bowling & Beehr 2006 → Pletzer 2019（既存）→ Nielsen, Glasø, & Einarsen 2017 → personality は確立した predictor
4. **Methodological gap**：Park 2024（既存）vs Orcutt 1957 / Spielauer 2011 / Bruch & Atwell 2015 → microsimulation は別系譜、harassment 領域に未適用
5. **Existing precursors and what they don't do**：Ho et al. 2025、Merlone & Argentero 2018、Notelaers et al. 2006/2011 → coping / dynamics / victim typology は扱われたが、population-scale perpetration projection × HEXACO typology は未着手
6. **Present study**：HEXACO 7 typology × probability table × MHLW triangulation × 3 counterfactuals

### Tier 2 / Tier 3 への引き継ぎ

Tier 1 は十分。Tier 2（Pillar 2 補強、Pillar 5 null-finding 補強、Pillar 7 self-report validity）は **執筆中に必要に応じて随時追加**でも論文の核は保てる。

### このセッションの commit 内容

- `simulation/docs/literature_audit/tier1_search_results.md` — 26 件の論文リスト + 取得チェックリスト + 論文骨格への含意


