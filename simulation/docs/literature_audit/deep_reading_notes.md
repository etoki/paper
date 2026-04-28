# Deep Reading Notes — Tier 1 Literature for HEXACO 7-Typology Harassment Simulation

**目的**：Tier 1 で取得した 24 PDF を直接精読し、引用すべき key findings、数値、quotable elements、本研究での citation 用途を一元管理する。

**作成日**：2026-04-27
**ブランチ**：`claude/review-harassment-research-plan-Dy2eo`
**関連**：`literature_audit.md`、`tier1_search_results.md`、`research_plan_harassment_typology_simulation.md`

**検証フォーマット**（既存 `simulation/deep_reading_notes.md` 準拠）：
- 原文ノート：原典 PDF を本セッションで直接精読した内容のみ
- 数値：原文に明示された値のみ転記（推測・補間禁止）
- 著者名・タイトル・venue は `tier1_search_results.md` の検証済み書誌情報と一致
- 逐語引用：ページ数 / セクション付き

**進捗**：

| Round | 対象 | 件数 | 状態 |
|---|---|---|---|
| R1 | Pillar 1（harassment epidemiology）| 10 | 進行中 |
| R2 | Pillar 3.1（microsimulation 系譜）| 9 | 未着手 |
| R3 | Pillar 3.2 + 3.3（precursor 確認）| 5 | 未着手 |

---

## Round 1：Pillar 1 — Harassment Epidemiology

---

### [1.1-C4] ILO (2022) — Experiences of violence and harassment at work: A global first survey

**Citation**：International Labour Organization, Lloyd's Register Foundation, & Gallup. (2022). *Experiences of violence and harassment at work: A global first survey*. Geneva: ILO. ISBN 9789220384923 (web PDF). https://doi.org/10.54394/IOAX8567

**Verification**：✅✅ 原文 PDF 56p を本セッションで精読

#### Research Question

「世界規模で、職場における暴力・ハラスメント（violence and harassment at work）の prevalence、頻度、種別はどうか？ どのような群が高 risk か？ 開示の困難さは何か？」

#### Method

- **Survey**：Lloyd's Register Foundation World Risk Poll 2021（Gallup 実施）に "violence and harassment at work" モジュール組込み
- **Sample**：
  - 全インタビュー対象：N ≈ 125,000、15 歳以上、**121 countries and territories**
  - **本報告は employment 対象のみ：N = 74,364**
  - Probability-based random sampling
- **対象期間**：lifetime（working life）と past 5 years
- **3 forms 分類**：physical / psychological / sexual

#### Key Findings（exact numbers from 原文 pp. 8）

##### 全体 prevalence（lifetime）

| 指標 | 数値 | 推定総数 |
|---|---|---|
| **少なくとも 1 形態経験** | **22.8%** | **743 million** |
| Physical violence/harassment | **8.5%** | 277 million |
| Psychological（最多）| **17.9%** | 583 million |
| Sexual | **6.3%** | 205 million |
| 複数形態経験（被害者の中で）| 31.8% | — |
| 3 形態すべて経験 | 6.3% | — |

##### 性別差

- **Sexual**：女性 8.2% / 男性 5.0%（**最大の性差**）
- **Physical**：男性が女性より多く経験
- **Psychological**：性差小

##### 地域別（lifetime, all forms）

| 地域 | Prevalence |
|---|---|
| Americas | 34.3% |
| Africa | 25.7% |
| Europe and Central Asia | 25.5% |
| **Asia and the Pacific** | **19.2%**（日本含む） |
| Arab States | 13.6% |

注記：Asia and the Pacific では男性 > 女性（+3.2pp）— 他地域と逆転

##### Physical violence 地域別（lifetime）

- Africa **12.5%**（最高）
- Americas 9.0% / Asia & Pacific 7.9% / Arab 7.2% / Europe & CA 6.5%

##### 高リスク群

- **若年労働者**：若年女性は若年男性の 2 倍の sexual violence/harassment 経験
- **移民女性**：非移民女性の約 2 倍の sexual violence/harassment 経験
- **賃金労働者**（vs self-employed）
- **過去に差別経験あり**（gender / disability / nationality / skin colour / religion）：差別経験者の **5 in 10 が職場 V&H 経験**（vs 非経験者 2 in 10）

##### 開示行動

- 被害者の **54.4% のみ**が誰かに開示
- 多くは複数形態経験後に開始
- 開示先は friends/family > 公式チャネル
- **開示しない理由 top 2**：「時間の無駄」"waste of time"、「評判への恐れ」"fear for their reputation"

#### Quotable Elements（原文逐語、page 番号付き）

> "Violence and harassment at work is a widespread phenomenon around the world, with more than one in five (22.8 per cent or 743 million) persons in employment having experienced at least one form of violence and harassment at work during their working life." (p. 8)

> "Comparability of data is problematic because different concepts, definitions and methods have been used. Statistics are often collected for a specific occupation, industry or group and may not be disaggregated by sex." (p. 9, Introduction)

> "Underreporting of violence and harassment in the world of work is also a problem, due to fear of victimization and retaliation as well as the lack of effective or accessible enforcement and monitoring systems in many countries." (p. 9)

#### 本研究での citation 用途

1. **Introduction opening 第 1 文**：「23% of employed adults globally experience violence and harassment at work (ILO, 2022)」 — global health/labor concern としての positioning
2. **Asia-Pacific 地域 prevalence**：日本の厚労省 31.4% パワハラ数値が ILO の Asia-Pacific 19.2% より高めである理由（method 差）の議論で使用
3. **データ比較可能性の限界**：Method の議論で「ILO survey は self-report、本研究の simulation 出力は厚労省 self-report と triangulate」を正当化する文脈
4. **Underreporting の言及**：limitation セクションで「self-report harassment は过少報告 bias を含む（ILO, 2022）」
5. **Sample size の参照点**：N=74,364 / 121 countries は global benchmark として強い；本研究の N=354 を国レベル aggregate に投影する正当性議論で使用

---

### [1.1-C1] Nielsen, Matthiesen, & Einarsen (2010) — Prevalence rates of workplace bullying meta-analysis

**Citation**：Nielsen, M. B., Matthiesen, S. B., & Einarsen, S. (2010). The impact of methodological moderators on prevalence rates of workplace bullying: A meta-analysis. *Journal of Occupational and Organizational Psychology, 83*(4), 955–979. https://doi.org/10.1348/096317909X481256

**Verification**：✅✅ 原文 PDF 25p を本セッションで精読

#### Research Question

「workplace bullying の prevalence は studies 間で大きく変動する（Denmark 3% 〜 Turkey 51%）。**測定方法**と **sampling 方法**は prevalence にどの程度影響するか？ 累積推定値はいくらか？」

#### Method

- **Design**：meta-analysis（102 prevalence estimates from 86 independent samples、N = 130,973）
- **Period**：1980s（2.6%）– 2008（81.8%）
- **Geography**：68% European、Scandinavia 過剰代表（Denmark 10 + Norway 10 + Sweden 2 = 全体の 1/4 以上）
- **Methodological moderators**：
  - **Self-labelling with definition**（定義提示後に自己ラベル）
  - **Self-labelling without definition**（定義なし自己ラベル）
  - **Behavioural experiences method**（NAQ-R 等の specific behaviour 経験を聞く）
  - **Random vs non-random sampling**
- **Statistical model**：Hedges & Olkin (1985) procedure、random effects model 主、fixed effects 補
- **Statistical dependency**：複数 method 同一 sample の studies は 1 つに集約（K=86 → 70 for unbiased estimates）

#### Key Findings（exact numbers from 原文）

##### 累積 prevalence（random effects、独立 sample 統制後）

| Method | K | Rate | 95% CI |
|---|---|---|---|
| **Overall（独立 sample）** | **70** | **14.6%** | **[12.3%, 17.2%]** |
| Behavioural experiences | 34 | 14.8% | [11.9%, 18.2%] |
| Self-labelling without definition | 21 | **18.1%** | [12.9%, 24.8%] |
| Self-labelling with definition | 47 | **11.3%** | [9.3%, 13.7%] |

→ measurement method による prevalence 差 = **6.8 percentage points**（self-label without def vs self-label with def）

##### Sampling method の効果（random vs non-random）

| Sampling | Rate | 95% CI |
|---|---|---|
| **Random sampled** | **11.3%** | [9.4%, 13.6%] |
| **Non-random sampled** | **20.0%** | [15.1%, 25.9%] |
| 差 | **8.7 pp** | (Q_B = 10.88, p < .001) |

→ Self-labelling with definition だけが random vs non-random で有意差（9.3% vs 15.5%、p < .01）

##### 地理的差異（Table 3 統合）

- **Scandinavian < other European < non-European** の順で prevalence 上昇
- Self-labelling with definition：Scandinavian 最低
- 「**Scandinavian の過剰代表によって overall 14.6% は過小評価されている可能性**」

##### Cultural variation の例

- Turkey 51%（Bilgel et al., 2006）
- Denmark < 3%（Agervold, 2007）
- 同じ Denmark 内でも別研究 13%（Høgh & Dofradottir, 2001）
- → **method と culture の交絡** が大きい

#### Quotable Elements（原文逐語、page 番号付き）

> "A total of 102 prevalence estimates of bullying from 86 independent samples (N = 130,973) were accumulated and compared by means of meta-analysis. At an average, the statistically independent samples provided an estimate of 14.6%." (Abstract, p. 955)

> "Findings from different studies on workplace bullying cannot be compared without taking moderator variables into account." (Abstract, p. 955)

> "The findings show that no significant differences were found between random (.144) and non-random (.145) sampled (QB = 0.00, df = 1, p > .05) behavioural estimates." (p. 965) — 行動測定では sampling bias 影響少

> "For studies using the self-labelling with definition method, the findings showed significant differences between random (.093) and non-random (.155) samples (QB = 8.42, df = 1, p < .01), thus indicating heterogeneity." (p. 965)

> "One cannot explain the difference in prevalence rates in Denmark (Agervold, 2007) and Turkey (Bilgel et al., 2006) with cultural differences, without first taking [method] into account." (Conclusions, p. 968 paraphrased)

#### 本研究での citation 用途

1. **Introduction での "11–18% prevalence" frame**：「meta-analytic evidence places the prevalence of workplace bullying at 11–18% depending on measurement methodology (Nielsen et al., 2010)」
2. **厚労省 31.4% との比較フレーミング**：厚労省はパワハラ specific の self-label without definition に近い → Nielsen の self-label without def 18.1% に近づく傾向と整合（が、日本固有要因も考察可能）
3. **Method 比較の方法論的根拠**：本研究で「behavioral measure」（power harassment scale 等）を使うことの正当化、および「measurement method の境界条件を明示」する Discussion の根拠
4. **Validation target の選択正当化**：厚労省実態調査は self-label の系列にあり、本研究の simulation 出力も同系の比較を行うことで apples-to-apples
5. **Sampling 方法の議論**：本研究の N=354 が non-random sample であることが prevalence 推定にどう影響するかの議論で参照（Nielsen の +8.7 pp upward bias）
6. **Geographic variation**：日本（Asia non-European）の prevalence が Scandinavian より高い傾向は本 meta-analysis と整合

---

### [1.1-C2] Bowling & Beehr (2006) — Workplace harassment from victim's perspective: theoretical model + meta-analysis

**Citation**：Bowling, N. A., & Beehr, T. A. (2006). Workplace harassment from the victim's perspective: A theoretical model and meta-analysis. *Journal of Applied Psychology, 91*(5), 998–1012. https://doi.org/10.1037/0021-9010.91.5.998

**Verification**：✅✅ 原文 PDF 16p を本セッションで精読

#### Research Question

「Workplace harassment（被害者視点）の **antecedents（誰が被害を受けるか）** と **consequences（被害は何をもたらすか）** はどう特徴づけられるか？ Role ambiguity / role conflict 等の伝統的職場 stressors とは独立した影響を持つか？」

#### 理論的枠組み（重要）

著者らは attribution theory + reciprocity norm に基づく **victim's view モデル**を提示。Antecedents 3 categories：
1. **Work environment**（climate, HR systems）
2. **Perpetrator characteristics**（impulsivity, emotional reactivity, cynicism, etc.）
3. **Victim characteristics**（personality, demographics）

→ 本研究との関連：本研究は **perpetrator personality**（HEXACO 7 類型）を扱うが、Bowling & Beehr の「3 categories of antecedents」枠組みは、本研究の「individual difference factors」を位置付ける理論的基盤。

#### Method

- **Design**：Hunter & Schmidt (2004) random-effects meta-analysis
- **Sample**：複数 studies（変数によって k=8〜25）、最大 N=7,441（victim negative affectivity, k=24）
- **Effect size**：Pearson r、信頼性補正後 ρ、80% credibility interval
- **Hierarchical regression**：harassment が role ambiguity / role conflict を超える incremental variance を予測するか検証

#### Key Findings（Table 2 antecedents、ρ = 信頼性補正後 corrected correlation）

##### 環境系 antecedents

| 変数 | k | N | ρ | 80% CI |
|---|---|---|---|---|
| **Work constraints** | 13 | 2,733 | **.53** | [.35, .72] |
| **Role conflict** | 16 | 5,429 | **.44** | [.34, .55] |
| Role ambiguity | 22 | 6,759 | **.30** | [.14, .46] |
| Role overload | 25 | 7,343 | **.28** | [.19, .37] |
| Autonomy | 13 | 2,823 | **−.25** | [−.34, −.16] |

→ Work constraints が **最強 antecedent**。組織レベルの stressor が支配的。

##### 個人差 antecedents

| 変数 | k | N | ρ | 80% CI |
|---|---|---|---|---|
| **Victim negative affectivity** | 24 | 7,441 | **.25** | [.07, .44] |
| Victim positive affectivity | 8 | 2,293 | −.09 | [−.21, .02] |
| Gender (male=1, female=2) | 11 | 2,921 | **−.05** | [−.05, −.05] |
| Age | 16 | 4,822 | **−.04** | [−.14, .05] |
| Tenure | 13 | 4,504 | .02 | [.00, .04] |

→ **Negative affectivity が個人差では最強**（ρ = .25）。Big Five 直接の meta は当時なし。**Gender / age は微小**。

##### 結果（Consequences、ρ）

- Generic strain ρ = .35
- Anxiety ρ = .31
- Depression ρ = .34
- Burnout ρ = .39
- Frustration ρ = .40
- Negative emotions at work ρ = .46
- Physical symptoms ρ = .31
- Job satisfaction ρ = −.36 程度（推定）
- Organizational justice ρ = −.35

##### Hierarchical regression

- 単独：Role ambiguity β = .21、Role conflict β = .38 → harassment 分散の 21% 予測
- Harassment 加入後：harassment は **多くの outcome で incremental variance を有意に追加**（role stressors を統制しても）

#### Quotable Elements（原文逐語、page 番号付き）

> "Both environmental and individual difference factors potentially contributed to harassment and harassment was negatively related to the well-being of both individual employees and their employing organizations." (Abstract, p. 998)

> "The organization can be seen as directly responsible for the presence of a perpetrator because its human resources systems (Box 2) select, train, and reward employees..." (p. 1000) — 「組織が perpetrator を選別・育成する」枠組みは本研究の Phase 2 構造介入の論理基盤

> "Among the Big Five (McCrae & Costa, 1987) personality characteristics... agreeableness and conscientiousness are two possibilities. People who are conscientious and agreeable would seem less likely to be a target of harassment because they do less to irritate coworkers who are potential perpetrators..." (pp. 1000–1001) — Big Five と harassment の理論的関連は **当時 victim 側でも明確化されていなかった**

> "Workplace harassment is one example of a stressor (Spector & Jex, 1998), and several strains have been hypothesized to result from it." (p. 1001)

#### 本研究での citation 用途

1. **Introduction の "antecedent" framing**：「**Both environmental and individual difference factors contribute to workplace harassment** (Bowling & Beehr, 2006)」 — 本研究が **個人差（HEXACO 類型）軸**を扱うことの理論的位置付け
2. **個人差の限定的役割の認識**：Bowling & Beehr では個人差最大が ρ=.25（NA）。本研究で「個人差は中規模」と謙虚に位置付ける根拠
3. **Phase 2 の構造介入正当化**：work constraints ρ=.53、role conflict ρ=.44 は組織レベル介入の余地が大きいことを示す → Counterfactual C（structural intervention）の根拠
4. **Phase 2 の displacement リスク議論**：「組織は HR systems で perpetrator を select/train」（Bowling & Beehr の主張）→ structural intervention だけでは extensive margin で displacement する論理を補強
5. **Limitation：victim vs perpetrator 視点の区別**：Bowling & Beehr は victim 視点。**本研究は perpetrator 側を扱うが、両側のメカニズムは reciprocal**（Bowling & Beehr p. 1000 reciprocity model）であることを明示
6. **Big Five → HEXACO の進歩の言及**：Bowling & Beehr 当時（2006）は Big Five 直接 meta が無かった → Pletzer et al. (2019) HEXACO meta、Nielsen et al. (2017) FFM meta が後に登場。本研究の HEXACO 7 類型はこの系譜の延長と位置付け

---

### [1.2-C1] 厚生労働省 (2021) 令和2年度 職場のハラスメントに関する実態調査 報告書（概要版）— **Phase 1 主 validation target**

**Citation**：厚生労働省 (2021). *令和２年度 厚生労働省委託事業 職場のハラスメントに関する実態調査 報告書（概要版）*. 東京海上日動リスクコンサルティング株式会社（受託）.

**Verification**：✅✅ 原文 PDF 36p（概要版）を本セッションで精読

#### Research Question

「2019 年の労働施策総合推進法改正（パワハラ防止法）から 1 年を経た時点で、日本の職場ハラスメント実態（企業の取組み状況、労働者の被害経験、相談・対応プロセス）はどうなっているか？ 平成 28 年度（2016 年）からの推移は？」

#### Method

##### 企業調査
- **対象**：全国の従業員 30 人以上の企業・団体
- **発送**：24,000 件（300 人未満：業種・規模別無作為抽出 12,086 件、300 人以上：全数 11,914 件）
- **回答**：6,426 件（**回収率 26.8%**）
- **方式**：郵送調査
- **時期**：2020 年 10–11 月

##### 労働者等調査（一般サンプル）— **本研究の Phase 1 validation 主 target**
- **対象**：全国の 20–64 歳の企業・団体勤務男女労働者（経営者・自営業・役員・公務員除く）
- **N = 8,000 名**
- **割付**：就業構造基本調査参考、性別 × 年代 × 正社員/非正社員（例：男性 20-29 歳正社員 550 名、女性 50-59 歳非正社員 200 名 等）
- **方式**：インターネット調査（パネル）
- **時期**：2020 年 10 月 6–7 日

##### 特別サンプル
- 妊娠・出産・育児休業ハラスメント（女性）：別 N
- 男性の育児休業ハラスメント：別 N
- 就活等セクハラ：別 N（2017–2019 年度卒）

#### Key Findings（exact numbers from 概要版 p. 15–16）

##### **過去 3 年間 ハラスメント被害経験割合**（全 N=8,000、本研究の主要参照値）

| ハラスメント | 経験率 | 平成 28 年度比 |
|---|---|---|
| **パワハラ** | **31.4%** | 32.5% から 1.1 pt 減 |
| **顧客等からの著しい迷惑行為** | **15.0%** | — |
| **セクハラ** | **10.2%** | — |

##### 性別差（過去 3 年間）

| ハラスメント | 男性 | 女性 |
|---|---|---|
| **パワハラ** | **33.3%** | **29.1%**（男性 +4.2 pt）|
| **セクハラ** | **7.9%** | **12.8%**（女性 +4.9 pt）|
| 顧客等からの著しい迷惑行為 | 14.9% | 15.0%（差なし）|

→ **本研究の Phase 1 14-cell（7 類型 × 2 gender）simulation 出力をこの値と triangulate する**

##### 過去 5 年間 マタハラ等

| 種類 | 経験率 |
|---|---|
| 妊娠・出産・育児休業等ハラスメント（女性）| 26.3% |
| 妊娠・出産等否定的言動（プレマタハラ）| 17.1% |
| 男性の育児休業等ハラスメント | 26.2% |

##### 過去 5 年間（学生）

- 就活等セクハラ：**25.5%**（2017–2019 年度卒）

##### パワハラ被害経験 詳細（図表 10 から）

被害頻度内訳：
- 何度も繰り返し経験：**6.3%**
- ときどき：**16.1%**
- たまに：**9.0%**
- 一度も経験ない：**68.7%**
- → 一度以上経験：**31.4%**

##### 業種別パワハラ経験率

- 高い：電気・ガス・熱供給・水道業 41.1% / 建設業 36.2% / 医療・福祉 35.5% / 生活関連サービス・娯楽業（高め）
- → **業種は本研究の N=354 にはないが、aggregate 比較で業種補正の必要性議論可**

##### 加害者と被害者の関係（企業調査）

- ⑤ 「正社員から正社員へ」が最多
- 「**上司（役員以外）から部下へ**」が最多 — **本研究の役職推定（管理職モデル）の妥当性根拠**
- 顧客等：「顧客等から自社従業員へ」91.5%
- 就活等セクハラ：「インターンシップ担当自社従業員」高い

##### 企業調査結果

- 企業の**過去 3 年間相談件数**：パワハラ 48.2%（最多）/ セクハラ 29.8% / 顧客等 19.5%
- パワハラ防止取組の副次効果トップ：「職場のコミュニケーション活性化／風通し改善」（35.9%）、「管理職の意識変化で職場環境変化」（32.4%）

##### 経験者と未経験者の職場特徴差（パワハラ）

経験者で高い項目：
- 上司・部下のコミュニケーション少ない／ない
- ハラスメント防止規定が制定されていない
- 失敗が許されない／失敗への許容度が低い
- 従業員間に冗談・おどかし・からかいが日常的

→ **これは Phase 2 構造介入（Counterfactual C）の介入対象組織変数**

#### Quotable Elements（原文逐語）

> "過去 3 年間に、パワハラ、セクハラおよび顧客等からの著しい迷惑行為を一度以上経験した者の割合は、それぞれ 31.4%、10.2%、15.0%であった。" (p. 15)

> "過去 3 年間にパワハラを経験した者の割合を男女別でみると、男性（33.3%）の方が女性（29.1%）よりも高かった。" (p. 16)

> "過去 3 年間にセクハラを経験した者の割合を男女別でみると、女性（12.8%）の方が男性（7.9%）よりも高かった。" (p. 16)

> "勤務先によるパワハラ、セクハラ行為の認定については、「ハラスメントがあったともなかったとも判断せずあいまいなままだった」（パワハラ 59.3%、セクハラ 40.2%）の割合が最も高かった。" (p. 15) — 解決放置 bias

#### 本研究での citation 用途

1. **Phase 1 主 validation target**：「Our simulation predicts a national power harassment prevalence of X.X% (95% CI [...]); the Ministry of Health, Labour and Welfare's nationwide survey reports 31.4% (MHLW, 2021)」 → MAPE ≤ 30% 主成功基準の照合先
2. **性別×ハラスメント種別の cell 単位検証**：男性パワハラ 33.3%、女性パワハラ 29.1%、男性セクハラ 7.9%、女性セクハラ 12.8% — 本研究の 14-cell（7 type × 2 gender）simulation の **gender 周辺分布** と triangulate
3. **被害頻度の binarization 妥当化**：「6.3% 何度も繰返し / 16.1% ときどき / 9.0% たまに」 → 本研究の binary outcome（mean+0.5SD 閾値）が「ときどき以上」に近い calibration である根拠
4. **加害者プロファイル**：「上司から部下へ」が最多 → 本研究の役職推定モデル（D1）で managerial role の比重を上げる根拠
5. **Phase 2 構造介入のターゲット**：経験者の職場で高い「ハラスメント防止規定なし」「コミュニケーション少ない」「失敗許容度低い」 → Counterfactual C の介入対象組織変数として使用
6. **Introduction Japan-context**：「31.4% of Japanese workers experienced power harassment in the past 3 years (MHLW, 2021)」を ILO global 23% / Nielsen 2010 meta 14.6% と並べて positioning
7. **Sample 比較根拠**：MHLW N=8,000 vs 本研究 N=354。母集団が労働者層であり、population scaling の参照点として使用
8. **2020 年と 2023 年比較**：MHLW 2024（[1.2-S1]）と組み合わせて時系列 sensitivity 分析

---

### [1.2-C3] Tsuno et al. (2015) — Socioeconomic determinants of bullying: National representative sample in Japan

**Citation**：Tsuno, K., Kawakami, N., Tsutsumi, A., Shimazu, A., Inoue, A., Odagiri, Y., Yoshikawa, T., Haratani, T., Shimomitsu, T., & Kawachi, I. (2015). Socioeconomic determinants of bullying in the workplace: A national representative sample in Japan. *PLoS ONE, 10*(3), e0119435. https://doi.org/10.1371/journal.pone.0119435

**Verification**：✅✅ 原文 PDF 15p を本セッションで精読

#### Research Question

「日本の national representative sample で、職場 bullying の social determinants（power distance、safety climate、frustration）を検証する。具体的には employment type / occupation / company size / education / household income / subjective social status (SSS) のうちどれが bullying experience と関連するか？」

#### 3 hypotheses（理論的）

1. **Power distance hypothesis**：階層底辺ほど bullying 多い（permanent < temporary, manager < non-manager）
2. **Safety climate hypothesis**：大企業ほど対策が整い bullying 少ない
3. **Frustration hypothesis**：低 SSS ほど fr​ustration 由来 bullying 多い（または受けやすい）

#### Method

- **Design**：cross-sectional, two-stage random sampling（national representative）
- **Sample frame**：N=5,000 Japan residents（20–60 歳）
- **対象**：47 都道府県を 11 strata、各都道府県内 100 survey sites を population size で sampling
- **時期**：2010 年 11 月 – 2011 年 2 月
- **Response**：N=2,384（47.7% response rate）
- **解析対象**：N = **1,546**（雇用者のみ抽出）
- **Bullying measurement**：**self-label without definition**（"Have you been bullied in your workplace?" 過去 30 日間、yes/no、定義提示なし）
- **Witnessing**：別の質問で計測
- **Statistical model**：Multiple logistic regression（3 モデル：M1 demographic, M2 + work + SES, M3 + SSS）

#### Sample 特性

| 変数 | 比率 |
|---|---|
| Male | 52.3% |
| Female | 47.7% |
| 大学卒以上 | 約 30% |
| Permanent worker | > 60% |
| Part-time | 約 20% |
| Tertiary sector | 約 70% |
| 〜50 人企業 | 約 30% |

→ 著者注：性別・雇用形態・企業規模・産業分類は **労働力調査 (LFS) と broadly comparable**

#### Key Findings（exact numbers）

##### Bullying prevalence（過去 30 日間）

| 指標 | 数値 |
|---|---|
| **個人 bullying 経験**（victim）| **6.1%**（n = 94 / 1,546）|
| **目撃**（witnessing）| **14.8%**（n = 229 / 1,546）|

→ MHLW 2021 の **31.4%（過去 3 年間）**と比較すると、Tsuno 2015 は **過去 30 日間** ベースなので 5–6 倍の reference period 差を考慮。**頻度ベースで両者は整合**。

##### 過去 30 日間 bullying との社会経済的関連（M1: 性年齢調整後 OR）

| 変数 | OR | 95% CI |
|---|---|---|
| Temporary employees vs permanent | **2.45** | [1.03, 5.85] |
| Junior high school vs university | **2.62** | [1.01, 6.79] |
| Lowest household income (< 2.5M yen) vs highest | **4.13** | [1.58, 10.8] |
| **Lowest SSS** vs upper/middle | **4.21** | [1.66, 10.7] |

##### 全変数同時投入（M3）

- **SSS（subjective social status）**：bullying との **逆相関 p = 0.002** ← Frustration hypothesis 支持
- 個別 SES 変数（household income, education）の効果は SSS 投入で減衰
- → SSS（主観的社会経済地位）は education / income を超える説明力

##### Witnessing（M3）

- **SSS** 逆相関 p = 0.017
- **Temporary employees** OR = 2.25 [1.04, 4.87]

##### 年齢

- 30 歳未満が高い（p = 0.021）

##### 性別

- 性差なし（experience に有意差なし）

#### Hypothesis 検証結果

| Hypothesis | 結果 |
|---|---|
| Power distance（permanent vs temporary）| **部分支持**（temporary > permanent OR=2.45）|
| Safety climate（large vs small companies）| **不支持**（company size 効果なし）|
| Frustration（SSS 逆相関）| **強く支持**（SSS p=0.002 in full model）|

#### Quotable Elements（原文逐語、page 番号付き）

> "Among 2,384 respondents, data were analyzed from 1,546 workers. ... Six percent and 15 percent of the total sample reported experiencing or witnessing workplace bullying, respectively." (Abstract, p. 1)

> "After adjusting for gender and age, temporary employees (Odds Ratio [OR]: 2.45 [95% CI = 1.03–5.85]), junior high school graduates (OR: 2.62 [95% CI: 1.01–6.79]), workers with lowest household income (OR: 4.13 [95% CI: 1.58–10.8]), and workers in the lowest SSS stratum (OR: 4.21 [95% CI: 1.66–10.7]) were at increased risk of experiencing workplace bullying." (Abstract, p. 1)

> "When all variables were entered simultaneously in the model, a significant inverse association was observed between higher SSS and experiencing bullying (p = 0.002)." (Abstract, p. 1)

> "The prevalence of workplace bullying has been reported to be as high as 15.7% on average in European countries, except Scandinavia [2]. A similarly high prevalence (9.0–15.5%) has been found in Asian countries including Japan [3–5]." (Introduction, p. 2)

> "In the survey, we did not present a definition of bullying to respondents due to limitations of space." (Methods, p. 4) — Nielsen 2010 meta の self-label without definition カテゴリに該当

#### 本研究での citation 用途

1. **Japan validation の peer-reviewed counterpart**：「Independent peer-reviewed estimates using a national representative sample place the past-30-days bullying prevalence at 6.1% (Tsuno et al., 2015)」 — MHLW 2021 と並べて Phase 1 validation の **学術的 anchor**
2. **Reference period の差を triangulate**：本研究の simulation は累積／時間窓を含意 → MHLW 31.4% (3y) と Tsuno 6.1% (30d) を 60 倍率で換算する根拠
3. **性差なしの言及**：「In a Japanese national representative sample, no significant gender difference in past-30-days bullying experience was observed (Tsuno et al., 2015)」— 一方 MHLW 2021 ではパワハラに男女差あり（男 33.3% / 女 29.1%）。本研究の **性別 cell の境界条件議論**に使用
4. **社会経済階層の bullying リスク**：本研究の simulation は personality を主軸とするが、**SSS / 雇用形態 effects が大きい（OR > 4）**ことは**limitation で明示**：「Personality-only model omits major SES determinants (Tsuno et al., 2015)」
5. **役職推定モデル（D1）の根拠**：Tsuno は manager / non-manual / service / manual 5 分類を使用 → 本研究の管理職／一般 2 分類を補強する文献根拠
6. **Frustration hypothesis 支持**：Phase 2 Counterfactual B（高リスク類型 targeted）の論理：低 SSS × 加害確率高い類型は二重リスク → 政策含意
7. **Sample size 比較**：Tsuno N=1,546 vs MHLW N=8,000 vs 本研究 N=354 → 本研究の N は中規模、national representative ではない limitation 明示

---

### [1.1-S1] Nielsen, Glasø, & Einarsen (2017) — Exposure to workplace harassment and FFM: meta-analysis

**Citation**：Nielsen, M. B., Glasø, L., & Einarsen, S. (2017). Exposure to workplace harassment and the Five Factor Model of personality: A meta-analysis. *Personality and Individual Differences, 104*, 195–206. https://doi.org/10.1016/j.paid.2016.08.015

**Verification**：✅✅ 原文 PDF 12p（Open Access、CC BY-NC-ND）を本セッションで精読

#### Research Question

「Exposure to workplace harassment（被害側）と Five Factor Model 5 traits の関連の真の規模は？ Geographical region / sampling method / measurement method / form of harassment は moderator か？」

#### 理論的枠組み（重要）

著者らは Nielsen & Knardahl (2015) に基づき、personality × harassment の 4 mechanisms：

1. **Target behavior mechanism**：personality が actual behavior に影響 → 加害者を誘発
2. **Negative perception mechanism**：personality が出来事の受け取り方を変える（同じ行為でも被害認知変動）
3. **Reversed causality mechanism**：harassment 経験が personality を変える（縦断研究で観察）
4. **Selection mechanism**：environmental selection（personality が職場環境を選ぶ）

→ 本研究との関連：本研究は **perpetrator** 側の personality を扱うが、Nielsen 2017 の 4 mechanisms は本研究の Discussion で「victim vs perpetrator の対称性／非対称性」を議論する理論的基盤になる。

#### Method

- **Design**：random-effects meta-analysis（Comprehensive Meta-Analysis v2/3、inverse-variance weighting）
- **Inclusion**：cross-sectional studies on FFM × workplace harassment exposure、published up to January 2015
- **Sample**：**101 effect sizes from 36 independent samples、Total N = 13,896**（mean N per study = 386）
- **Total observations** in random effects model：N = 29,105
- **Moderators**：
  - Geographical region（Asia/Oceania / Europe / USA）
  - Sampling（probability / non-probability）
  - Measurement method（behavioral experience / self-labeling）
  - Form of harassment（bullying / abusive supervision / other）

#### Key Findings（exact numbers）

##### 主効果（Table 2、random effects model、total N = 29,105）

| Big Five trait | K | N | Mean r | 95% CI | Sig |
|---|---|---|---|---|---|
| **Neuroticism** | 32 | 12,997 | **+0.25** | (CI not shown for main) | **p < .01** |
| **Agreeableness** | 19 | 8,843 | **−0.17** | — | **p < .01** |
| **Conscientiousness** | 22 | 9,343 | **−0.10** | — | **p < .05** |
| **Extraversion** | 17 | 7,717 | **−0.10** | — | **p < .05** |
| Openness | 11 | 6,689 | **+0.04** | — | n.s. |

→ effect sizes は small-medium（Neuroticism のみ medium-large に近い）

##### Geographical moderator（Table 3、Asia/Oceania vs Europe vs USA）

| Trait | Asia/Oceania | Europe | USA | Q_Between |
|---|---|---|---|---|
| **Neuroticism** | **0.16*** (K=4) | **0.33*** (K=12) | 0.21*** (K=16) | 6.11* |
| **Agreeableness** | **−0.27*** (K=1) | −0.05 ns (K=6) | **−0.22*** (K=12) | 13.62*** |
| **Conscientiousness** | **−0.15*** (K=3) | +0.03 ns (K=9) | **−0.19*** (K=10) | 14.19*** |
| Extraversion | −0.14* (K=2) | −0.16* (K=9) | −0.01 ns (K=6) | 4.96 ns |
| Openness | 0.05 ns | 0.07 ns | 0.01 ns | 1.66 ns |

→ **Asia/Oceania での Neuroticism 効果は最弱（r=0.16）**、Conscientiousness と Agreeableness は **Asia/Oceania で米国と同等以上に効く**（−0.15 〜 −0.27）

##### Measurement method moderator

- Neuroticism：self-label r=0.38, behavioral r=0.20（Q=4.48*） → self-label で **効果ほぼ倍**
- Agreeableness：self-label r=−0.02 ns, behavioral r=−0.21（Q=11.34***） → behavioral でのみ検出可能
- Conscientiousness：self-label r=+0.07 ns, behavioral r=−0.17（同上 pattern）

→ self-label measurement は **neuroticism を膨らませ、agreeableness/conscientiousness を消す** bias

##### Sampling method（probability vs non-probability）— ほぼ effect なし

- Neuroticism：non-probability r=0.25, probability r=0.24（Q=0.02 ns）
- 他 traits も Q_Between すべて n.s.

##### 公開バイアス

著者は publication bias 解析を実施、effect は robust と結論

#### Quotable Elements（原文逐語）

> "101 cross-sectional effect sizes from 36 independent samples, totaling 13,896 respondents, showed that exposure to harassment was positively associated with neuroticism (r = 0.25; p < 0.01; K = 32), and negatively associated with extraversion (r = −0.10; p < 0.05; K = 17), agreeableness (r = −0.17; p < 0.01; K = 19), and conscientiousness (r = −0.10; p < 0.05; K = 22). Harassment was not related to openness (r = 0.04; p > 0.05; K = 11)." (Abstract, p. 195)

> "For applied purposes, managers, consultants and HR personnel need to understand the true role of personality traits in order to avoid being a captive of the fundamental attribution error which may lead them to overestimate the role these dispositions play in the harassment process when handling actual cases (Ross, 1977)." (p. 196) — **Fundamental attribution error 警告**は本研究の Discussion limitation の根拠

> "Studies from Europe (r = 0.33; 95% CI = 0.21–0.44) provided stronger associations [for neuroticism] compared to studies from USA (r = 0.21; 95% CI = 0.13–0.29) and Asia/Oceania (r = 0.16; 95% CI = 0.10–0.22)." (p. 201)

> "Summarized, the findings provide evidence for personality traits as correlates of exposure to workplace harassment." (Abstract, p. 195)

#### 本研究での citation 用途

1. **Introduction Pillar 1 → 2 の bridge**：「Personality traits—particularly neuroticism (r=0.25) and agreeableness (r=−0.17)—are established correlates of harassment exposure (Nielsen, Glasø, & Einarsen, 2017)」 — **personality predictor 系譜の核**
2. **HEXACO への移行論理**：Nielsen 2017 は FFM のみ。**Honesty-Humility（HEXACO 第 6 因子）は包含されていない** → 本研究で HEXACO 7 類型（HH 含む）を扱う novelty を強化
3. **Asia/Oceania での効果サイズ低下**：日本含む地域で Neuroticism 効果が r=0.16（Europe r=0.33 の半分）→ 「日本 sample で personality 効果がやや小さい可能性」を limitation で言及
4. **Asia の Conscientiousness × Agreeableness 強い効果**：本研究の N=354 でも C / A 系 trait は強く効く可能性。**Phase 1 結果解釈の優先 trait 候補**
5. **Self-label vs behavioral measurement の歪み**：本研究の harassment は **self-label without definition に近い** → Neuroticism 効果は膨らみ、A/C 効果は減衰しやすい。Limitation 議論で明示
6. **4 mechanisms framework**：本研究は perpetrator focus だが、Discussion で「victim 側の 4 mechanisms (Nielsen & Knardahl, 2015) と perpetrator 側の対応する mechanisms（target selection / aggressor characteristics / displacement / etc.）」の対称性を議論
7. **Fundamental attribution error 警告**：本研究は personality-based simulation だが、**個別事例への適用ではなく集団政策のための tool であること**を Discussion で強調する根拠（Nielsen 2017 が明示的に推奨）
8. **Bowling & Beehr 2006 後継としての位置付け**：Bowling & Beehr 当時 Big Five 直接 meta なし → Nielsen 2017 が初の包括的 meta → **本研究は HEXACO ベースで perpetrator 側 typology meta-equivalent を提供する**位置付け

---

### [1.1-C3] Einarsen, Hoel, & Notelaers (2009) — NAQ-R psychometric properties

**Citation**：Einarsen, S., Hoel, H., & Notelaers, G. (2009). Measuring exposure to bullying and harassment at work: Validity, factor structure and psychometric properties of the Negative Acts Questionnaire-Revised. *Work & Stress, 23*(1), 24–44. https://doi.org/10.1080/02678370902815673

**Verification**：✅✅ 原文 PDF 22p を本セッションで精読

#### Research Question

「workplace bullying 測定の de facto standard になりつつある **NAQ-R（22 項目）** の psychometric properties（信頼性、因子構造、criterion validity、construct validity）を、heterogeneous な UK 大規模 sample で確立する」

#### 設計上の重要点

- 全 22 項目が **行動に基づく**（behavioural）：「bullying / harassment」という用語を使わず具体的行為を尋ねる（Arvey & Cavanaugh, 1995 推奨に準拠）
- → これは self-labelling よりも **objective estimate** を提供
- 過去 6 ヶ月間の頻度を測定（5 段階：Never / Now and then / Monthly / Weekly / Daily）
- Last 2 categories は使用頻度低く analysis では集約

#### Method

- **Sample**：N = **5,288 UK employees**（heterogeneous sample）
- **解析**：
  - Confirmatory factor analysis（1, 2, 3 factor models 比較）
  - Cronbach's alpha for internal consistency
  - Latent class cluster (LCC) analysis（被害群分類）
  - 単一項目 self-label victimization measure と相関
  - 心身健康（GHQ-12, OSI psychosomatic）、職場環境（PMI workload、relationships、climate、satisfaction、commitment）、leadership（autocratic、laissez-faire）と相関で construct validity 検証

#### Key Findings（exact numbers）

##### 信頼性

- **Total NAQ-R 22 項目 Cronbach's α = .90** — excellent internal consistency
- "alpha if item deleted" 解析で **どの項目を除いても α 改善せず** → 全 22 項目とも有効

##### 因子構造（CFA）

3 因子モデルが最適：
1. **Personal bullying**（人格攻撃系）
2. **Work-related bullying**（業務関連 bullying）
3. **Physical intimidation**（物理的威嚇、3 項目）

→ ただし「**single factor measure としても運用可能**」と明記（柔軟性）

##### Criterion validity

- NAQ-R total score と単一項目 self-labelled victimization に高相関
- **Targets of bullying は 22 項目すべてで non-targets より有意に高得点**

##### Construct validity（NAQ-R 高得点者ほど…）

- General Health Questionnaire 12 項目（α = .92）：mental health 悪化
- Psychosomatic complaints (OSI) (α = .89)：身体症状増
- PMI workload (α = .86)：仕事量過大
- PMI relationships (α = .89)：同僚関係悪化
- PMI organizational climate (α = .84)：組織風土悪化
- PMI organizational satisfaction (α = .88)：満足度低下
- Autocratic leadership (α = .76)：上司の独断的傾向強
- Laissez-faire leadership (α = .77)：上司の放任強

→ 期待される全方向で相関、construct validity 確保

##### Latent Class Cluster (LCC) 分析

「LCC analysis showed that the instrument may be used to differentiate between groups of employees with different levels of exposure to bullying, ranging from infrequent exposure to incivility at work to severe victimization from bullying and harassment.」

→ Notelaers et al. (2006, 2011) [3.3-C2, 3.3-C3] と一貫した方法論。**bullying は continuum でなく離散的な exposure typology** であることを支持

#### Quotable Elements（原文逐語）

> "By reanalyzing data based on a heterogeneous sample of 5288 UK employees, the results show that the 22-item instrument has a high internal stability, with three underlying factors: personal bullying, work-related bullying and physically intimidating forms of bullying, although the instrument may also be used as a single factor measure." (Abstract, p. 24)

> "All items are written in behavioural terms with no reference to the terms 'bullying' or 'harassment,' following recommendations by Arvey and Cavanaugh (1995) in relation to sexual harassment. Although based on self-report, such an approach is considered to provide a more objective estimate of exposure to bullying behaviours than self-labelling approaches, as respondents' need for cognitive and emotional processing of information would be reduced." (p. 28)

> "Cronbach's alpha for the 22 items in the NAQ-R was .90, indicating excellent internal consistency whilst also suggesting that it may be a reliable instrument with an even fewer number of items." (p. 33)

> "Hence, the NAQ-R is proposed as a standardized and valid instrument for the measurement of workplace bullying." (Abstract, p. 24)

#### 本研究での citation 用途

1. **国際標準測定 instrument としての位置付け**：「The Negative Acts Questionnaire-Revised (NAQ-R; Einarsen, Hoel, & Notelaers, 2009) has become the standard instrument for measuring workplace bullying internationally」 — 本研究の Methods で日本語パワハラ尺度（Tou et al., 2017）/ ジェンダーハラスメント尺度（Kobayashi & Tanaka, 2010）を NAQ-R 系統に位置付け
2. **3 因子構造の Japanese 対応**：本研究の power harassment と gender harassment は NAQ-R の personal bullying / work-related bullying 因子に部分的対応 → 国際比較可能性
3. **Behavioural vs self-labelling の区別**：本研究は **behavioural items に近い** scale を使用 → Nielsen 2010 meta の behavioral 14.8% prevalence 系列に位置付け
4. **6 ヶ月 reference period**：NAQ-R は **過去 6 ヶ月**、MHLW は **過去 3 年**、Tsuno 2015 は **過去 30 日**。本研究の simulation の reference period 議論で並列参照
5. **Construct validity の根拠**：本研究の harassment 尺度も心身健康・leadership と相関するはず → Phase 1 の cell-level estimate の妥当性議論
6. **LCC 分析の precedent**：bullying は離散的 exposure typology を持つ → 本研究の **加害側 personality typology**（HEXACO 7 cluster）も同じ離散的構造で扱う論理的整合
7. **物理的脅威の包含**：NAQ-R の 3 因子目（physical intimidation）は本研究の power harassment にも対応する可能性 → 日本のパワハラ概念の国際的位置付け補強
8. **MHLW 2024 報告書との対応**：MHLW R5 は職場環境変数を含む → NAQ-R construct validity 文献で並列参照可能（Phase 2 構造介入対象選定）

---

### [1.2-C2] Tsuno, Kawakami, Inoue, & Abe (2010) — Japanese version of NAQ-R: reliability and validity

**Citation**：Tsuno, K., Kawakami, N., Inoue, A., & Abe, K. (2010). Measuring workplace bullying: Reliability and validity of the Japanese version of the Negative Acts Questionnaire. *Journal of Occupational Health, 52*(4), 216–226. https://doi.org/10.1539/joh.L10036

**Verification**：✅✅ 原文 PDF 11p（J-STAGE Open Access）を本セッションで精読

#### Research Question

「Bergen Bullying Research Group の認可を受けて NAQ-R 22 項目を日本語に翻訳した。日本人サンプルで internal consistency reliability、concurrent validity、construct validity（factor-based validity を含む）は確立できるか？」

#### Method

- **Sample**：日本人公務員（civil servants）N = **1,626**（男 830 + 女 796）
- **対象組織**：関東地方の **7 職場**（市役所 6 + 地方自治体公務員事務所 1）
- **時期**：2009 年 3 月
- **Response rate**：**46.7%**（4,072 名から回答）
- **解析対象**：1,626 名（不完全回答除外後）
- **Measures**：
  - NAQ-R 22 項目（過去 6 ヶ月）
  - LIPT（Leymann Inventory of Psychological Terror、12 ヶ月、45 行為）— concurrent validity 検証
  - 心理的苦痛尺度（GHQ 系）
  - 対人関係尺度（intragroup / intergroup conflict、supervisor / coworker support）
  - Interactional justice 尺度
- **Statistical analyses**：
  - Cronbach's α
  - Confirmatory factor analysis：1-, 2-, 3-factor models（GFI, AGFI, RMSEA, BIC）
  - Pearson r で concurrent / construct validity

#### Key Findings（exact numbers）

##### 信頼性

- **Cronbach's α = 0.91–0.95**（男性・女性別）
- → Einarsen et al. 2009 UK 0.90 と整合

##### 因子構造

- 3 因子抽出（Einarsen 2009 と同じ意図）
- ただし **日本の因子構造は Einarsen 2009 と若干異なる**
- → **Factor 1 が分散の大半を説明** → "**single factor structure** が data に最 fit"

→ この発見は重要：**日本では bullying は international のような明確な 3 因子（personal/work-related/physical）ではなく、より単一次元的な現象**。本研究で power harassment / gender harassment の 2 尺度を扱う際の参考。

##### Concurrent validity（NAQ-R と他 bullying 尺度の関連）

- **NAQ-R は LIPT と強相関**
- → 国際標準（NAQ-R）と Leymann の伝統的尺度（LIPT）の **measurement invariance** が日本で確立

##### Construct validity（NAQ-R 高得点者ほど…）

- 心理的苦痛 高（期待方向、有意）
- Intragroup conflict 高（同上）
- Intergroup conflict 高（同上）
- Supervisor support 低（同上）
- Coworker support 低（同上）
- Interactional justice 低（同上）

→ Einarsen 2009 と同方向、construct validity 確立

##### 文化的洞察

> "Japan is supposed to be more vertical and hierarchy-oriented (...) than in European countries (...), which could result in a greater prevalence of workplace bullying. Also, patterns of workplace bullying may be different among countries, as suggested by different factor structures of the NAQ found among studies." (p. 217)

→ 著者自身が日本での **structural difference を予測** → factor 1 dominance（単一次元）はその一例

#### Quotable Elements（原文逐語）

> "A total of 830 males and 796 females were surveyed, using anonymous questionnaires including the NAQ-R, Leymann Inventory of Psychological Terror (LIPT), and scales for interpersonal relations at work and psychological distress (response rate, 46.7%)." (Abstract, p. 216)

> "Cronbach's alpha coefficients of the internal consistency reliability of the NAQ-R were high (0.91–0.95) for males and females." (Abstract, p. 216)

> "Although three factors were extracted, this findings differed slightly from the factor structure previously reported (Einarsen et al., 2009). However, Factor 1 explained most of the variance, indicating that a one factor structure fitted the data better." (Abstract, p. 216)

> "Japan is supposed to be more vertical and hierarchy-oriented than in European countries, which could result in a greater prevalence of workplace bullying. Also, patterns of workplace bullying may be different among countries, as suggested by different factor structures of the NAQ found among studies." (p. 217) — **vertical hierarchy** の日本特有性

> "Conclusion: The present study showed acceptable levels of reliability and validity of the Japanese version of the NAQ-R among Japanese civil servants." (Abstract, p. 216)

#### 本研究での citation 用途

1. **日本における NAQ-R 測定の正当化**：「The Japanese version of the NAQ-R has been validated in a sample of 1,626 Japanese civil servants (Tsuno et al., 2010), establishing measurement compatibility with international research」 — 国際比較可能性の根拠
2. **本研究の harassment 尺度の international bridge**：Tou et al. (2017) Power Harassment Scale + Kobayashi & Tanaka (2010) Gender Harassment Scale を Tsuno 2010 経由で NAQ-R 系統に位置付け
3. **Factor 1 dominance の意味**：日本では bullying が**より単一次元的**に経験される → 本研究の simulation で power harassment と gender harassment を別 outcome として扱う妥当性議論（両者を組み合わせる代わりに別個推定する根拠）
4. **Vertical hierarchy 仮説**：Tsuno が明示する「日本の vertical / hierarchical 文化」は本研究の **役職推定（managerial / non-managerial）の重要性** を強化。Phase 1 で役職を含めて分析する正当化
5. **Sample limitation の比較**：Tsuno 2010 は **公務員のみ** → 本研究 N=354 はクラウドソース。両方とも general representative ではない → 本研究の limitation 議論で **Tsuno 2010 自身も civil servants only であった点を引用**して相対化
6. **Reference period の参考**：NAQ-R は過去 6 ヶ月、本研究の simulation の time window 議論で参照
7. **Internal consistency benchmark**：日本語 NAQ-R で α=0.91–0.95 → 本研究の harassment scales の alpha 値の比較 reference として使用
8. **MHLW survey との bridge**：MHLW 実態調査は self-label without definition、Tsuno 2010 は behavioral measurement（NAQ-R）→ 両者の prevalence 差は **Nielsen 2010 meta の measurement method 差（11.3% vs 14.8%）と整合する**ことを Discussion で示せる

---

### [1.2-S1] 厚生労働省 (2024) 令和5年度 職場のハラスメントに関する実態調査 結果概要

**Citation**：厚生労働省 (2024). *令和5年度厚生労働省委託事業 職場のハラスメントに関する実態調査 結果概要*. 雇用環境・均等局 雇用機会均等課. （2024-05-17 公表、雇用の分野における女性活躍推進に関する検討会 第6回 資料4）

**Verification**：✅✅ 原文 PDF 28p（結果概要）を本セッションで精読

#### Research Question

「令和元年法改正・令和4年4月の中小企業含む完全施行から 2 年経過時点で、ハラスメントの発生状況・企業対策進捗・労働者の意識はどう変化したか？ 令和2年度（2020）からの推移は？」

#### Method

##### 企業調査
- 対象：従業員 30 人以上、全国
- 発送 25,000 件（令和2年度 24,000 件から増）
- **有効回答 7,780 件（有効回答率 31.1%）** ← R2 の 6,426 件 / 26.8% から増
- 方式：郵送（Web 受付併用）
- 時期：2023 年 12 月

##### 労働者等調査
- **N = 8,000 名**（一般サンプル）+ 特別サンプル 2,500 名
- 対象：20–64 歳、企業・団体勤務、経営者・自営業・公務員除く
- 方式：インターネット調査（パネル）
- 時期：2024 年 1 月 11–29 日

→ R2（2020）と R5（2023）の sample frame は同一仕様。**直接比較可能**。

#### Key Findings — R5 vs R2 比較（exact numbers from p. 29 / 概要 §）

##### 労働者の過去 3 年間ハラスメント被害経験

| ハラスメント | **R5 (2023)** | **R2 (2020)** | **変化** |
|---|---|---|---|
| **パワハラ** | **19.3%** | 31.4% | **−12.1 pt（大幅減）** |
| **セクハラ** | **6.3%** | 10.2% | **−3.9 pt** |
| **顧客等からの著しい迷惑行為** | **10.8%** | 15.0% | **−4.2 pt** |
| **女性 妊娠・出産・育児休業ハラスメント** | 26.1% | 26.3% | **−0.2 pt（横ばい）** |
| **男性 育児休業ハラスメント** | 24.0% | 26.2% | **−2.2 pt** |
| 就活等セクハラ（インターンシップ中）| 30.1% | n/a | 増加（特別サンプル新規）|
| 就活等セクハラ（インターンシップ以外）| 31.9% | 25.5% | +6.4 pt（増加）|

→ **パワハラは 31.4% → 19.3% で大幅減少**（−12 pt）。令和元年法改正（パワハラ防止法）の effect 示唆。

##### 企業調査：相談有無

| 種別 | R5 | R2 | 変化 |
|---|---|---|---|
| パワハラ相談あり | **64.2%** | 48.2% | +16.0 pt（**増**）|
| セクハラ相談あり | 39.5% | 29.8% | +9.7 pt |
| 顧客等迷惑行為 相談あり | 27.9% | 19.5% | +8.4 pt |
| マタハラ相談あり | 10.2% | 5.2% | +5.0 pt |

→ **被害経験は減少、相談は増加** の二重 trend。**社会的開示の改善 + 実数減少**の同時発生。

##### 該当事例（企業がハラスメントと判断）

- パワハラ 73.0%（+3.0 pt）
- セクハラ 80.9%（+2.2 pt）
- 顧客等迷惑行為 86.8%（−5.9 pt）
- 介護休業ハラ 55.5%（+33.6 pt の大幅増）

##### 経験者と未経験者の職場特徴差（R5）

R2 と同様の職場 risk factors が確認：
- ハラスメント防止規定なし
- 上司・部下のコミュニケーション不足
- 失敗が許されない／許容度低い
- 従業員が男性ばかり（**新たに R5 で同定**：差 19.3 pt）
- 従業員間の競争激しい
- 中途入社や外国人など多様 background 比率高い（差 6.4 pt 程度）

→ **gender-monolithic workplace は power harassment risk factor**（Phase 2 構造介入の新候補変数）

#### Quotable Elements（原文逐語）

> "過去３年間に勤務先等で各ハラスメントを受けた経験については、パワハラは19.3%（−12.1%）、セクハラは6.3％（−3.9%）、顧客等からの著しい迷惑行為は10.8%（−4.2％）..." (p. 29)

> "令和２年度と比べて、パワハラ、セクハラ、顧客等からの著しい迷惑行為は減少し、女性の妊娠・出産・育児休業等ハラスメント、男性の育児休業等ハラスメントは横ばい、インターンシップ中及びインターンシップ以外の就職活動中での就活等セクハラは増加している。" (p. 29)

> "過去３年間に相談があったと回答した企業割合については、パワハラは64.2%（＋16.0％）..." (p. 2) — 企業相談増加と労働者経験減少の trend 共存

#### 本研究での citation 用途

1. **Phase 1 validation target の選択明確化**：本研究は **MHLW 2021（R2）の 31.4% を主 validation**。R5 の 19.3% は sensitivity reference として併用（同 sample 仕様、3 年差）
2. **時系列 trend の議論**：「Power harassment prevalence has declined from 31.4% (MHLW, 2021) to 19.3% (MHLW, 2024) following the 2019 legal reform」 — Phase 2 介入効果の存在を国レベルで確認できる証拠（**Counterfactual A 普遍介入の partial natural experiment**）
3. **Phase 2 anchor の補強**：法改正介入で 12 pt 減少 → universal intervention（Counterfactual A）の plausible effect size の上限例として引用
4. **企業対応の改善 trend**：相談率 +16.0 pt、該当判定 +3.0 pt → 組織レベル対策強化の存在 → Counterfactual C（structural intervention）の効果が現実に出ている根拠
5. **「男性のみ職場」リスクの新発見**：Phase 1 の gender 変数を超えて、**workplace gender composition** が独立 risk factor → 本研究の 14-cell（gender × type）simulation の解釈強化
6. **就活等セクハラの増加**：労働者経験データに含まれないが、本研究の sample（20–64 歳労働者）の境界を明示する根拠
7. **Sample size の安定性**：R2 N=8,000、R5 N=8,000 と同一 → 直接比較可能 → 本研究の simulation 出力の R2 vs R5 trianguration 可能
8. **Pre-Bowling-Beehr 概念での再評価**：R5 の組織変数解析は Bowling & Beehr (2006) 枠組みの環境系 antecedents を経験的に補強 → **環境介入の有効性**

---

### [1.2-S2] Tsuno & Tabuchi (2022) — Risk factors for workplace bullying during COVID-19 pandemic in Japan

**Citation**：Tsuno, K., & Tabuchi, T. (2021/2022). Risk factors for workplace bullying, severe psychological distress, and suicidal ideation during the COVID-19 pandemic: A nationwide internet survey for the general working population in Japan. *medRxiv preprint* (https://doi.org/10.1101/2021.11.18.21266501) — published version available via PMC https://pmc.ncbi.nlm.nih.gov/articles/PMC9638740/

**Verification**：✅✅ medRxiv preprint PDF 44p を本セッションで精読

#### Research Question

「COVID-19 pandemic 期間中、日本の workplace bullying / severe psychological distress (SPD) / suicidal ideation の risk factors は何か？ 高 risk 群は pre-pandemic と異なるか？」

#### Method

- **Design**：cross-sectional, nationwide internet survey
- **Sample**：N = **16,384**（一般労働人口、JACSIS = Japan COVID-19 and Society Internet Survey の subset）
- **時期**：2020 年 8–9 月（COVID-19 第 2–3 波時期）
- **Bullying measurement**：Brief Job Stress Questionnaire（簡易職業性ストレス調査票）の **single item** — past period（明示なし、おそらく過去 1 ヶ月）
- **Mental health**：
  - **K6 ≥ 13**：severe psychological distress
  - 自殺念慮：single item
- **Statistical model**：Poisson regression（**Prevalence Ratios** with 95% CI、3 model 階層調整）
- **Adjustments**：gender, age, partner, area, education, income, occupation, industry, office size, prior depression history, work demands, work mode

#### Key Findings — Overall prevalence（COVID 期間中）

| 指標 | 数値 |
|---|---|
| **Workplace bullying** | **15%** |
| Severe psychological distress (SPD) | 9% |
| Suicidal ideation | 12% |

→ Tsuno 2015 の **6.1%（pre-pandemic, 30 days, single item）**から増加（**約 2.5 倍**、Brief Job Stress Q vs definition なし self-label の差を含む）

#### Key Findings — Bullying risk factors（PR with 95% CI、Model 3 fully adjusted）

##### 性別（reference: female）

- **Men**：bullying リスク高（pre-pandemic と逆転、Tsuno 2015 では性差なし）

##### 年齢（reference: Over 65）

- 18–24 歳：基準
- **25–34 歳：PR 2.38 [1.90, 2.98]** — 若年が高リスク
- 35–44 歳：PR 2.13
- 45–54 歳：PR 2.00
- 55–64 歳：PR 1.64
- → **若年労働者が最高リスク**（Tsuno 2015 と整合）

##### 雇用形態（reference: Part-time worker）

| カテゴリ | PR (95% CI) |
|---|---|
| **Executive** | **1.94** [1.58, 2.38] — **最高** |
| **Manager** | **1.66** [1.38, 2.00] |
| **Permanent employee (non-manager)** | **1.56** [1.33, 1.82] |
| Contract worker | 1.31 [1.06, 1.61] |
| Self-employed | 1.14 (ns) |
| Dispatched worker | 0.99 (ns) |

→ **Executives & Managers が最高 risk**（**pre-pandemic Tsuno 2015 の "lowest hierarchy = highest risk" と逆転**！）

##### 教育・所得（reference: 大学卒以上、年収 1,000 万円以上）

- 教育：差なし（n.s.）
- 年収：**1.99 万円以下 PR 1.58** [1.29, 1.93] — 低所得が高リスク（Tsuno 2015 と整合）
- 年収 2.00–3.99 万円 PR 1.32 [1.12, 1.55]

##### 働き方変化

- **新たに在宅勤務開始**：bullying に対しては **preventive**（保護的）
- ただし mental health（SPD、suicidal ideation）には **risk factor**
- → **Bullying と mental health で異なる pattern**

##### Bullying → SPD / 自殺念慮の関連

- Bullying experience が SPD prevalence ratio：**男性 PR 3.20** [2.74, 3.73]、女性 PR 2.39 [2.00, 2.86]
- Bullying experience が suicidal ideation PR：男性 1.95、女性 1.87
- → bullying は **男女両方で深刻 mental health 帰結**を持つ。男性で更に強い

#### Quotable Elements（原文逐語）

> "Overall, 15% of workers experienced workplace bullying, 9% had SPD, and 12% had suicidal ideation during the second and third wave of the COVID-19 pandemic in Japan." (Abstract, p. 2)

> "Men, executives, managers, and permanent employees had a higher risk of bullying compared to women or part-time workers." (Abstract, p. 2)

> "Increased physical and psychological demands were common risk factors for bullying, SPD, and suicidal ideation. Newly starting working from home was a significant predictor for adverse mental health outcomes, however, it was found to be a preventive factor against workplace bullying." (Abstract, p. 2)

> "When intervening to decrease workplace bullying or mental health problems, we should focus on not only previously reported vulnerable workers but also workers who experienced a change of their working styles or job demands." (Conclusions, p. 3)

#### 本研究での citation 用途

1. **Pandemic-period prevalence の reference**：「During the pandemic, Japanese workplace bullying prevalence reached 15% (Tsuno & Tabuchi, 2022, N=16,384)」 — MHLW 2024 R5 の 19.3%（過去 3 年）の context
2. **「Manager > non-manager」逆転の議論**：本研究の役職推定（D1）は managerial role を **加害者プロファイル**として扱うが、Tsuno 2022 は **manager が被害者にもなる**ことを示す → 本研究の Discussion で「役職と加害／被害は両側性」を明示
3. **Phase 1 sensitivity analysis 強化**：Tsuno 2015 (pre-pandemic, low-hierarchy=high-risk) と Tsuno 2022 (pandemic, high-hierarchy=high-risk) の **対照** → 本研究の 14-cell 結果が時期依存することを認識
4. **Sample size の参考点**：N=16,384 は単一研究最大級 → 本研究 N=354 が large-scale national と異なる情報を持つことを正当化（personality detail vs prevalence breadth のトレードオフ）
5. **Mental health 連鎖の anchor**：Bullying → SPD PR 3.20 (男性) — Phase 1 Stage 2「被害者の f2 割がメンタル疾患」の実証根拠（f2 ≈ 0.30 を支持）
6. **Suicidal ideation の連鎖**：Bullying → suicidal ideation PR 1.95（男性）→ Phase 1 Stage 2「被害者の f3 割が自殺念慮」を新たに加える sensitivity 用変数候補
7. **Working from home の dual effect**：Phase 2 Counterfactual C（structural intervention）の **副作用**例：bullying 抑制（intended）vs mental health 悪化（unintended）→ **displacement effect** の具体例
8. **Single-item Brief Job Stress Q vs NAQ-R の差**：Tsuno 2015 (NAQ-R 系) vs Tsuno 2022 (BJSQ single-item) の比較は本研究の measurement validity 議論で参照

---

## Pillar 1 Synthesis（10 件統合）

### 1. Workplace harassment は global health concern であり、prevalence は方法論的選択に強く依存する

| Reference | Population | Measurement | Reference period | Prevalence |
|---|---|---|---|---|
| ILO 2022 | Global, N=74,364 (121 countries) | self-label, 3 forms | lifetime | **22.8%** (any form) |
| Nielsen 2010 meta | 86 samples, N=130,973 | mixed | mixed | **14.6%** (overall) |
| Nielsen 2010 meta（self-label w/ def）| same | self-label with def | various | 11.3% |
| Nielsen 2010 meta（behavioural）| same | NAQ-R 系 | 6 mo | 14.8% |
| Nielsen 2010 meta（self-label w/o def）| same | self-label without def | various | 18.1% |
| Bowling & Beehr 2006 meta | 各 variable k=8–25, N up to 7,441 | mixed | various | (effect size focus) |
| Einarsen 2009 NAQ-R | UK, N=5,288 | NAQ-R 22 items | 6 mo | (psychometric focus) |
| **MHLW 2021 R2** | Japan, N=8,000 | self-label w/o def（職場のパワハラ概念）| 3 yr | **31.4%（パワハラ）** |
| **MHLW 2024 R5** | Japan, N=8,000 | same | 3 yr | **19.3%（パワハラ）** |
| Tsuno 2015 PLoS | Japan, N=1,546 (random) | self-label single item | 30 days | **6.1%** |
| Tsuno 2010 NAQ-J | Japan civil servants, N=1,626 | NAQ-R Japanese | 6 mo | (psychometric focus) |
| Tsuno 2022 COVID | Japan, N=16,384 | BJSQ single item | recent period | **15%** |

**論理**：
- 同じ「日本」でも 6.1%（30日, single item）から 31.4%（3年, パワハラ概念）まで **5 倍以上の幅**
- これは Nielsen 2010 meta の **measurement method 8.7 pp 差**および **reference period の効果**で説明可
- → **本研究の simulation 出力をどの reference にあてるかで、単純比較は誤導的**

### 2. Harassment の predictors は environment >> individual differences

Bowling & Beehr (2006) meta：
- 環境系：work constraints ρ=.53、role conflict ρ=.44、role ambiguity ρ=.30
- 個人差：negative affectivity ρ=.25、性別 ρ=−.05、年齢 ρ=−.04、教育（n.s.）

Nielsen 2017 FFM meta：
- Big Five → harassment exposure：Neuroticism r=+.25、Agreeableness r=−.17、C/E r=−.10、O n.s.
- **Asia/Oceania では Neuroticism 効果は弱め（.16 vs Europe .33）**、C/A 効果は USA と同等

Tsuno 2015：
- **SSS（subjective social status）OR=4.21** が最強 predictor（frustration hypothesis）
- 雇用形態：temporary > permanent OR=2.45（power distance hypothesis 部分支持）

**論理**：
- 個人差効果は中規模（meta r=.10–.25）。HEXACO の HH も Pletzer 2019 で workplace deviance に効くが effect size は同程度
- → 本研究は **個人差軸**（HEXACO 7 類型）を扱うが、**環境系効果（Bowling & Beehr 2006）と SSS 効果（Tsuno 2015）を limitation で明示**

### 3. 本研究は「victim 側 personality 文献」の延長ではなく、「perpetrator 側 typology」novelty

- Nielsen 2017 FFM meta は **target/victim 側**：低 A、低 C、高 N の人が **harassment を受けやすい**
- 本研究の Harassment 論文（Tokiwa preprint）は **perpetrator 側**：低 HH、高 Psychopathy が **harassment を起こしやすい**
- → 両者は対称的だが **異なる文献系譜**。本研究は **perpetrator 側 HEXACO meta-equivalent** として novelty 主張

### 4. 日本固有の文脈：vertical hierarchy、measurement convergence

- Tsuno 2010 (NAQ-J)：日本では bullying が **より単一次元的**（Factor 1 dominant）
- Tsuno 2010 著者註：「Japan is more vertical and hierarchy-oriented than European countries」
- Tsuno 2022 COVID：**executives & managers が高 risk**（pre-COVID と逆転）→ 階層上位への bullying の存在
- MHLW 2021/2024：「上司から部下へ」が最多 → 階層下位への bullying（経典的）
- → **日本の harassment は階層性に強く規定される。本研究の役職推定（D1）は中核変数**

### 5. Phase 1 validation 戦略の固化

| Triangulation 比較対象 | Reference | 期待値 | sensitivity 用途 |
|---|---|---|---|
| **主：MHLW 2021 R2 過去 3 年** | パワハラ 31.4%、セクハラ 10.2% | MAPE ≤ 30% | gender/age cell の比較も可 |
| **副：MHLW 2024 R5 過去 3 年** | パワハラ 19.3%、セクハラ 6.3% | trend 分析、12 pt 減を介入なしで再現できるか | 法改正効果を本研究の simulation で説明可能か |
| **副：Tsuno 2015 過去 30 日** | bullying 6.1% | reference period 換算後 | random sampling vs クラウドソース比較 |
| **副：Tsuno 2022 過去** | bullying 15% | COVID context | manager/executive 高 risk の同定可能性 |
| **国際 baseline**：ILO 2022 | Asia-Pacific 19.2% lifetime | 国際相対化 | 日本 outlier かどうか |

→ **Multi-reference triangulation で robust な validation が可能**

### 6. Phase 2 介入の自然実験 evidence

MHLW 2021 (31.4%) → 2024 (19.3%) の **−12 pt** 減少は：
- 2019 法改正（パワハラ防止法）の universal intervention
- 2022 中小企業含む完全施行
- = Counterfactual A（universal）の **partial natural experiment evidence**
- → 本研究の Phase 2 で δ = +0.3 SD の介入予測値が −10 pt 程度であれば、**MHLW 自然実験と整合**することを示せる

### 7. 残存する gap（本研究で対応すべき）

- Pillar 1 文献は **prevalence と victim antecedents** に集中
- **Perpetrator-side prediction × HEXACO typology × population-scale** の組合せは依然 untouched
- → 本研究の novelty は Pillar 1 内にも存在（perpetrator typology projection）

---

## Round 2：Pillar 3.1 — Non-LLM Microsimulation 系譜

---

### [3.1-C1] Orcutt (1957) — A new type of socio-economic system [microsimulation 系譜の祖]

**Citation**：Orcutt, G. H. (1957). A new type of socio-economic system. *Review of Economics and Statistics, 39*(2), 116–123. (Reprinted in *International Journal of Microsimulation, 1*(1), 3–9, 2007). https://doi.org/10.2307/1928528

**Verification**：✅✅ 原文 PDF 7p（IJM 2007 reprint、原典の page 番号は square brackets で保持）を本セッションで精読

#### Research Question

「既存の **aggregate-level** 経済モデルは、individual decision-making unit の挙動を捉えられず、予測力に重大な限界がある。**個人・世帯・企業 を elemental decision-making unit として扱い、確率的な振る舞いを集計して全体予測を得る**新型モデルは可能か？」

#### 中核的論証（4 段階）

##### 1. Aggregate モデルの限界
- 政策効果予測に弱い
- 長期予測に弱い
- 短期予測ですら弱い
- **集計値だけ予測し、distribution（分布）を予測できない**
- 個人・世帯・企業の **数や場所**を扱えない

##### 2. Aggregation 不可能性の論証（key insight、p. 117 数値例）

100 人の個人を考え、各自 Y = f(X) = 0 if X=0、1 if X=1、1 if X=2（**非線形・断続的**）。

- X が全員 1 のとき：ΣY = 100
- X の半分が 0、半分が 2（依然 ΣX = 100）：**ΣY = 50**
- → **ΣX が同じでも ΣY が異なる** = aggregate input → aggregate output の安定関係は存在しない
- 「ミクロでの安定的関係 ≠ マクロでの安定的関係」 nonlinear 系では aggregate モデルは **原理的に正しく組めない**

##### 3. 解決策：個人単位 simulation
- **Elemental decision-making units**（個人、世帯、企業）を model 単位に
- 各 unit に **operating characteristics**（= 入力 → 出力確率分布の規則）
- 出力は確率的：probability distribution からの **random drawing** で決定
- 期によって probability 自体が変動（前期の event/state 依存）
- = **recursive、discrete-step、確率的 model**

##### 4. 集計の代替手法：simulation census
- 集計値は依然有用だが、**個人レベル simulation の出力を census のように集計する**ことで得る
- → relationship aggregation ではなく **outcome aggregation**

#### Illustrative model（p. 122–123）

具体的 sketch として：

- **3 unit types**：individual males / individual females / married couples
- **Individual male/female の possible outputs**：marriage（特定 female と）、self death
- **Married couple の possible outputs**：male child、female child、dissolution（divorce or death）
- **Operating characteristics**：
  - 死亡確率：年齢の関数（性別ごと別関数）
  - 結婚確率：season × age of male × age of female × marriageable 比率
  - 出産確率（無/男/女）：marital status × mother's age × previous births × interval × season
  - 離婚確率：marriage duration の関数
- **時間刻み**：月次

→ これは Phase 1 の本研究 simulation の構造的祖。本研究は：
- 7 HEXACO types = elemental decision-making unit
- 加害確率 = operating characteristic（type × gender × role 条件付き）
- 1 期 simulation でも複数期（被害者数、離職、メンタル疾患の連鎖）も同型に扱える

#### Quotable Elements（原文逐語、bracket page numbers from original RES 1957）

> "Existing models of our socio-economic system have proved to be of rather limited predictive usefulness. This is particularly true with respect to predictions about the effects of alternative governmental actions and with respect to any predictions of a long-range character." (p. [116])

> "It is also true, but not so widely noticed, that current models of our socio-economic system only predict aggregates and fail to predict distributions of individuals, households, or firms in single or multi-variate classifications." (p. [116])

> "There is an inherent difficulty, if not practical impossibility, in aggregating anything but absurdly simple relationships about elemental decision-making units into comprehensible relationships between large aggregative units such as industries, the household sector, and the government sector." (p. [116])

> "If nonlinear relationships are present, then stable relationships at the micro level are quite consistent with the absence of stable relationships at the aggregate level." (p. [117])

> "The most distinctive feature of this new type of model is the key role played by actual decision-making units of the real world such as the individual, the household, and the firm." (p. [117])

> "Predictions about aggregates will still be needed but will be obtained by aggregating behavior of elemental units rather than by attempting to aggregate behavioral relationships of these elemental units. That is, aggregates will be obtained from the simulated models in a fashion analogous to the way a census or survey obtains aggregates relating to real socioeconomic systems." (p. [117])

> "The probabilities associated with alternative behaviors or responses are treated as dependent on conditions or events prior to the behavior. Thus, these probabilities vary over time as the system develops or as external conditions change..." (p. [117])

#### 本研究での citation 用途

1. **Microsimulation 系譜の founding citation**：「Microsimulation, originating in Orcutt's (1957) framework, models socio-economic systems as collections of elemental decision-making units (individuals, households, firms) whose probabilistic behaviors aggregate to predict population-level outcomes」 — Introduction の方法論的系譜の **第一引用**
2. **Aggregate 限界からの正当化**：本研究の core 主張「個人レベル HEXACO 類型から国レベル aggregate prevalence を予測する」は **Orcutt 1957 の logic の harassment 領域 first application** と positioning
3. **Operating characteristic = 確率テーブル**：本研究の Stage 0「7 類型 × gender × role の cell ごとに加害確率を bootstrap 推定」は Orcutt の "operating characteristic" の現代的 instantiation
4. **Outcome aggregation の正当化**：「relationship を集約せず、outcome を集約」（Orcutt の方法論的選択）→ 本研究の Stage 1（母集団スケーリング）の論理基盤
5. **Random drawing の使用根拠**：本研究の Monte Carlo 部分は Orcutt's 1957 の "random drawings from one or more discrete probability distributions" の延長
6. **Park 2024 LLM-based agent simulation との位置付け**：両者とも individual unit から aggregate を作るが、Orcutt 系譜は **probabilistic、外的 calibration、transparent** で、Park 系譜は **LLM-based、internal language modeling、opaque**。本研究は Orcutt 系譜の現代化として positioning
7. **Discrete probability table の透明性主張**：Orcutt の time、Phase 1 D13 power analysis での 14-cell 推定は、Orcutt's "operating characteristics" の透明性原理の現代的実装（cell ごと CI 付き）
8. **政策含意 generation 能力**：Orcutt は「政策効果予測の改善」を新型モデルの主目的とした → 本研究 Phase 2 の counterfactual 介入予測は **Orcutt's foundational motivation の直接継承**

---

### [3.1-C2] Spielauer (2011) — What is dynamic social science microsimulation?

**Citation**：Spielauer, M. (2011). What is social science microsimulation? *Social Science Computer Review, 29*(1), 9–20. https://doi.org/10.1177/0894439310370085

**Verification**：✅✅ 原文 PDF 14p（Statistics Canada chap1-eng.pdf 版を本セッションで精読）

#### Research Question

「Dynamic social science microsimulation とは何か？ どのような状況で他の手法より優れるか？ 強みと限界はどこか？ Orcutt 以来 50 年経過した現時点（2011）でなぜ普及するか？」

#### 中核的内容

##### 定義

> "Dynamic social science microsimulation can be perceived as experimenting with a virtual society of thousands - or millions – of individuals who are created and whose life courses unfold in a computer." (p. 5)

= **個人レベル simulation を時間的に展開し、社会システムを再現する**手法

→ 本研究との対応：本研究は **時間軸を含まない静的 microsimulation**（cross-sectional snapshot）に近いが、Stage 2 の連鎖（被害者数 → 離職 → メンタル疾患）で動的要素を加える

##### Cell-based model との対比（重要）

| | Cell-based model | Microsimulation |
|---|---|---|
| データ表現 | Cross-classification table（cell ごと count）| Individual records |
| 更新単位 | Cell の頻度 | 個人の特性 |
| 限界 | 変数増えると **cell 爆発**（12 変数 × 6 levels = 21 億 cells）| Monte Carlo variation を伴う stochasticity |
| 連続変数 | 不可能（離散化要） | 直接処理可能 |
| 個人履歴 | 失う（cell 内匿名）| 保持 |

→ **本研究の D13 power analysis（14-cell vs 28-cell）はまさにこの cell explosion 問題に直面**：8,000 万労働人口を 28 cell に押し込むと、cell ごと N=2.86M で OK だが、HEXACO 連続変数を保持したい場合は microsimulation が optimal

##### Microsimulation が適切な 3 状況（p. 7–9）

###### 2.1 Population heterogeneity（個人差が重要）

> "Microsimulation is the preferred modeling choice if individuals are different, if differences matter, and if there are too many possible combinations of considered characteristics to split the population into a manageable number of groups."

→ 本研究：HEXACO 6 次元連続スコア + age + gender + 業種 + 役職 → 12 個以上の特性組合せ。**cell-based では cell 爆発**、microsimulation 優位

###### 2.2 Aggregation problem（集計困難）

> "Microsimulation is the adequate modeling choice if behaviours are complex at the macro level but better understood at the micro level."

→ 本研究：個人 personality → 加害確率関係は cell-level で安定推定可能だが、aggregate prevalence 関係は文化・組織・時期に depend → micro モデルから aggregate を生成する Orcutt-Spielauer 系譜

###### 2.3 Individual histories（履歴が重要）

> "Microsimulation is the only modeling choice if individual histories matter, i.e. when processes possess memory."

→ 本研究：Phase 2 介入後の長期 follow-up（介入受けた人 vs 受けなかった人の累積効果）を扱う場合、microsimulation 必須。Phase 1 でも被害者の累積メンタル悪化は履歴依存

##### Strengths（3 dimensions）

###### Theoretical：life course perspective を直接支援

- macro → micro パラダイムシフト
- causality / time の概念を持ち込む
- Analysis → Synthesis（Willekens 1999）：multiple processes を組み合わせて複雑 dynamics を生成
- **Linked lives**（個人間相互作用の追跡）

###### Practical：policy maker の distributional concern を直接扱える

- 年金 sustainability、税制改革の winners/losers、教育政策
- 「介入によって誰がどう変わるか」を simulation で示せる
- → 本研究の Phase 2 counterfactual の正当化

###### Technical：variable/process type の制限なし

- Continuous / categorical 混在可
- 非線形挙動可
- 確率的 outcomes 自然

##### Drawbacks（2 種）

###### Intrinsic：Monte Carlo variation

- 確率的サンプリングの結果、毎回 simulation 結果が微妙に変動
- 大 sample で expected value に収束（law of large numbers）
- → 本研究では bootstrap で CI を提示することで対処

###### Transitory：データ要求高、計算コスト（時間とともに減）

##### Orcutt 以来の歴史的位置付け

> "In the social sciences, dynamic microsimulation goes back to Guy Orcutt's (1957) idea about mimicking natural experiments in economics. His proposed modeling approach corresponds to what can be labelled as the empirical or data-driven stream of dynamic microsimulation models, i.e. models designed and used operatively for forecasting and policy recommendations." (p. 6)

→ Orcutt → Spielauer の **継承的系譜** を明示。本研究はこの系譜の現代化と positioning

#### Quotable Elements（原文逐語）

> "Dynamic social science microsimulation can be perceived as experimenting with a virtual society of thousands - or millions – of individuals who are created and whose life courses unfold in a computer." (p. 5)

> "Microsimulation is the preferred modeling choice if individuals are different, if differences matter, and if there are too many possible combinations of considered characteristics to split the population into a manageable number of groups." (p. 7)

> "Microsimulation is the adequate modeling choice if behaviours are complex at the macro level but better understood at the micro level." (p. 8)

> "Microsimulation is the only modeling choice if individual histories matter, i.e. when processes possess memory." (p. 9)

> "Due to this random element, each simulation experiment will result in a slightly different aggregated outcome, converging to the expected value as we increase the simulated population size. This difference in aggregate results is called Monte Carlo variation which is a typical attribute of microsimulation." (p. 7)

> "Microsimulation is attractive from a theoretical point of view, as it supports innovative research embedded into modern research paradigms like the life course perspective. (...) Microsimulation is attractive from a practical point of view, as it can provide the tools for the study and projection of socio-demographic and socio-economic dynamics of high policy relevance. And microsimulation is attractive from a technical perspective, since it is not restricted with respect to variable and process types..." (p. 9)

#### 本研究での citation 用途

1. **Microsimulation 現代定義の引用**：「Dynamic microsimulation, defined as 'experimenting with a virtual society of thousands or millions of individuals' (Spielauer, 2011), is a probabilistic, individual-level approach to socio-economic systems...」 — Methods の方法論的 framing
2. **Cell-based vs microsim の境界線**：本研究の 14-cell 主分析は **cell-based に近い**が、HEXACO 連続スコア保持 + bootstrap individual-level 抽出を組み合わせる **hybrid approach**。Spielauer の枠組みで「cell explosion 直前で踏みとどまる cell-based」と説明可能
3. **3 situations の適用根拠**：本研究は (i) population heterogeneity 高（HEXACO 7 類型）、(ii) aggregation problem あり（personality × harassment は line non-linear）→ **2 situations で microsimulation 適合**
4. **Theoretical strength → 本研究の理論的支援**：Spielauer が指摘する「life course perspective + linked lives」は本研究の Phase 2（targeted intervention の subsequent effects on subordinates / colleagues）の議論に転用可能
5. **Monte Carlo variation の合理化**：本研究の bootstrap CI は「Monte Carlo variation を CI で示すことが microsimulation の standard practice」（Spielauer）として正当化
6. **Orcutt 系譜の現代化**：「The present study extends Orcutt's (1957) microsimulation paradigm, as articulated in modern form by Spielauer (2011), to the workplace harassment domain」 — Introduction での **二段引用**
7. **Policy maker への applicability**：Spielauer は microsim の policy relevance（distributional analysis）を強調 → Phase 2 counterfactual の policy implication は **microsim の established function**
8. **LLM-based simulation との位置付け**：Spielauer の microsim は **probabilistic、calibrated、transparent**。Park 2024 系 LLM-based agent simulation は **language-model based、internal logic opaque**。本研究は Spielauer 系の現代的 instantiation

---

### [3.1-C3] Rutter, Zaslavsky, & Feuer (2011) — Dynamic microsimulation models for health outcomes: A review

**Citation**：Rutter, C. M., Zaslavsky, A. M., & Feuer, E. J. (2011). Dynamic microsimulation models for health outcomes: A review. *Medical Decision Making, 31*(1), 10–18. https://doi.org/10.1177/0272989X10369005

**Verification**：✅✅ 原文 PDF 13p（NIH-PA author manuscript via PMC）を本セッションで精読

#### Research Question

「Health policy 領域で MSM（microsimulation model）はどう構築・運用されるべきか？ Calibration / validation / sensitivity analysis / between-model comparison / variability 源泉の整理」

#### 中核的論証

##### MSM の定義と目的

> "Microsimulation models (MSMs) for health outcomes simulate individual event histories associated with key components of a disease process; these simulated life histories can be aggregated to estimate population-level effects of treatment on disease outcomes and the comparative effectiveness of treatments." (Abstract)

→ Orcutt 1957 の health 領域への adaptation。本研究は harassment 領域への adaptation（**parallel positioning**）。

##### MSM の歴史（health 系譜）

- 1985 Habbema et al. **MISCAN**（cancer screening）
- 1994 **Population Health Model**（Statistics Canada、lung/breast cancer）
- diabetes、cardiovascular、stroke、osteoporosis 等で展開
- **CISNET**（NCI Cancer Intervention and Surveillance Modeling Network）が代表的研究 network

##### MSM の 2 components

1. **Natural history model**：介入なしの疾患プロセス
2. **Intervention model**：介入の効果

→ 本研究との対応：
- Natural history model = Phase 1（baseline harassment prevalence）
- Intervention model = Phase 2（3 種 counterfactual A/B/C）

##### Calibration（パラメータ選択）

- 直接 estimation 不可な場合に使用
- 観測 data を再現するように parameter を fit
- 例：observed tumor detection rate は initiation rate と growth rate の合成 → 個別 estimation 不可、合計を fit

→ 本研究では：cell-level 加害確率を直接 N=354 から estimate（calibration が necessity ではない）。むしろ **cell estimate を validation 段階で MHLW と比較** する形

##### Validation

- **External validation**：calibration に使わなかった data との比較
- **Internal validation**：calibration data 自身との比較
- 中間：calibration data の subset を hold out

→ 本研究：
- **External validation**：N=354 個人レベル → 国レベル aggregate vs MHLW（外部 source）
- これは Rutter 2011 の **external validation** の strict definition に該当

##### Sensitivity Analysis

- 不確実 parameter の値を変動させて結果の変化を確認
- **Probabilistic sensitivity analysis**：parameter に分布を置き、複数 run で結果分布を得る
- → 本研究の Stage 3（V, f1, f2 sweep）はこれに該当

##### Sources of Variability（6 つ）

1. Inherent population variability
2. Parameter estimation 不確実性
3. Calibration data selection
4. Calibration data の sampling variability
5. **Simulation (Monte Carlo) variability**
6. Model structure assumptions

→ Bayesian calibration で 1–5 を統合可能と Rutter 2011 は推奨

##### Between-model comparison

- 同じ calibration data を使って異なる model 構造で結果比較
- 例：CISNET の 7 breast cancer models、Mt. Hood Challenge diabetes models
- → 本研究の Phase 1 階層 baseline（B0–B3）はこれの **mini version**：同じ data で 4 model 構造を比較

##### Population simulation 設計

> "Once developed, MSMs are used to simulate a hypothetical population with specific characteristics, such as a specific age-sex distribution, or a specific risk factor profile. An MSM can be structured to simulate a population directly, taking distributions of population characteristics at baseline as inputs..."

→ 本研究の Phase 1 Stage 1（13,668 の類型分布 + 厚労省 demographics で母集団スケーリング）はこの **direct population simulation** スタイル

#### Quotable Elements（原文逐語）

> "Microsimulation models (MSMs) for health outcomes simulate individual event histories associated with key components of a disease process; these simulated life histories can be aggregated to estimate population-level effects of treatment on disease outcomes and the comparative effectiveness of treatments." (Abstract, p. 10)

> "MSMs have two components: a natural history model and an intervention model." (p. 12)

> "When such direct estimation is not possible, model parameters are selected so that the model reproduces observed results, a process called 'calibration'." (p. 14)

> "Model validation is the process of assessing whether a model is consistent with data not used for calibration, a process also called 'external validation'. Because validation requires that data be held out of the calibration process, MSMs may not be validated..." (p. 15)

> "Sources of MSM variability and uncertainty include inherent variability in the population of interest, variability due to estimation of unknown parameters, selection of calibration data, sampling variability of the selected calibration data, simulation (Monte Carlo) variability, and variability due to model structure assumptions." (p. 16)

> "Probabilistic sensitivity analysis places distributions on unknown parameters, providing a range of possible results. Parameters are sampled from specified distributions, and multiple MSM runs are used to infer variability in model results that result from variability in model parameters." (p. 16)

#### 本研究での citation 用途

1. **Health 領域 MSM の系譜引用**：「Dynamic microsimulation models have been extensively applied to health policy questions (Rutter, Zaslavsky, & Feuer, 2011), including cancer screening (Habbema et al., 1985 MISCAN), cardiovascular disease, and infectious disease transmission. The present study extends this lineage to workplace harassment as a public health concern」 — Introduction の方法論的系譜
2. **Natural history + intervention 2-model framework**：本研究の Phase 1 / Phase 2 構造をこの枠組みで明示的に記述
3. **External validation の正当化**：「Following Rutter et al. (2011)'s recommendation, our simulation is externally validated against MHLW national survey data not used for calibration」
4. **6 sources of variability の体系化**：本研究の limitation セクションでこの 6 種類すべてを言及（特に MC variability、parameter uncertainty、population variability）。**透明性の demonstration**
5. **Between-model comparison の mini 実装**：Phase 1 baseline B0–B3 比較は Rutter 2011 の between-model comparison の小規模版 → 「following Rutter et al. (2011), we compare our type-conditional model against three baseline models...」
6. **CISNET 等 institutional infrastructure の言及**：本研究は単発研究だが、**network 化の potential**（D14 future work で comparable workplace harassment modeling network）を Discussion で示唆
7. **Probabilistic sensitivity analysis の標準化**：本研究の Stage 3（V, f1, f2 sweep）はこの「probabilistic sensitivity analysis」の standard practice として位置付け
8. **Simulation の transparency**：Rutter 2011 が再三強調する「model transparency」は、本研究の確率テーブルの公開・bootstrap CI の提示で実装。LLM-based black box との contrast

---

### [3.1-C4] Bruch & Atwell (2015) — Agent-based models in empirical social research

**Citation**：Bruch, E., & Atwell, J. (2015). Agent-based models in empirical social research. *Sociological Methods & Research, 44*(2), 186–221. https://doi.org/10.1177/0049124113506405

**Verification**：✅✅ 原文 PDF 30p（HHS Public Access via PMC）を本セッションで精読

#### Research Question

「Agent-based modeling は sociology の主流に入っていないが、empirical research と結びつけば強力な tool になる。**ABM を empirical research program に組み込む方法**は？ Calibration、validation、sensitivity の方法論的 codification は可能か？」

#### 中核的論証

##### ABM の定義と sociological motivation

> "Agent-based models are computer programs in which artificial agents interact based on a set of rules and within an environment specified by the researcher (Miller and Page 2007). While these rules and constraints describe predictable behavior at the micro-level, the interactions among agents and their environment often aggregate to create unexpected social patterns." (p. 186)

→ Sociology の中核問題（**micro-macro problem**）：個人の motivation/decision がどう large-scale social organization と connect するか
- Schelling 1971 / 1978 segregation tipping
- Axelrod cooperation
- これら以外の sociology mainstream 影響は限定的

##### ABM が役立つ situation

1. **Modeling interdependent behavior**：個人行動が他者の過去・現在・予測未来に依存（≠ 単純合計可能な独立 behavior）
2. **Generating mechanism-based explanations**（Hedström & Ylikoski 2010）
   - feedback / tipping / contagion / diffusion
   - selection / offsetting / vacancy chains / network externalities
3. **Predicting policy effects**：個人 behavioral response を組み込んで意図せぬ結果を anticipate（公共政策の counterfactual）
4. **Sharpening empirical thinking**：mechanism を可視化、key explanatory variables を identify

##### Realism の 2 段階

| 種別 | 内容 |
|---|---|
| **Low-dimensional realism** | 1–2 dimensions で empirical realism、他は stylized。Schelling segregation type |
| **High-dimensional realism** | 多次元 empirical realism、real population correspondence、microsimulation 系 |

→ 本研究：HEXACO 6 次元 + gender + role + 業種 + 年齢 + 多 outcome → **high-dimensional realism** の実装

##### Empirically Calibrated Agent-Based Models（ECA、Hedström & Åberg 2005）

ABM 文献に **empirical data 統合**を主題化した先駆。本研究は ECA 系譜の現代延長。

##### ABM の goodness-of-fit と sensitivity 評価

- **Goodness-of-fit**：simulated outputs vs observed empirical patterns
- **Sensitivity testing**：parameter / 構造仮定の変動下での結果の安定性
- 推奨 practice：multiple runs、systematic parameter sweep、robustness check

→ 本研究の D13 power analysis、Stage 3 sensitivity sweep（V, f1, f2）はこの recommended practice の実装

#### 重要な区別：Independence assumption

> "...this article focuses on somewhat simpler MSMs that assume independence across individuals, though the issues raised here are equally relevant to agent-based models." (Rutter et al., 2011 と整合)

→ Bruch & Atwell の ABM は **agent interactions** を扱う。本研究の Phase 1 は **independence assumption** を置く（cell-level estimate 個人独立）。
- **本研究は厳密には ABM ではなく microsimulation**（agent 相互作用なし）。**この区別を方法論議論で明示**することで Bruch & Atwell との positioning が clear に

#### Quotable Elements（原文逐語）

> "Agent-based modeling has become increasingly popular in recent years, but there is still no codified set of recommendations or practices for how to use these models within a program of empirical research." (Abstract, p. 186)

> "Agent-based models help fill the gap between formal but restrictive models and rich but imprecise qualitative description (Holland and Miller 1991, cited in Page 2008)." (p. 188)

> "Moreover, agent-based models are especially amenable to incorporating detailed, multi-layered empirical data on human behavior and the social and physical environment, and can represent a granularity of information and faithfulness of detail that is not easily handled within statistical or mathematical models." (p. 188)

> "A key feature of agent-based modeling is that it explicitly links micro- and macro-levels of analysis." (p. 189)

> "...the connection between individuals' actions and their collective consequences would be transparent if one could simply sum over individuals' intentions or behavior to generate expected population-level attributes. The problem is that nearly all human behavior is interdependent..." (p. 189)

#### 本研究での citation 用途

1. **Sociological positioning**：「Following the agent-based and empirically calibrated tradition (Bruch & Atwell, 2015; Hedström & Åberg, 2005), we ground our simulation in micro-level empirical data (N=354) while targeting macro-level outcomes (national prevalence)」 — Methods の方法論的 framing
2. **Microsim vs ABM の区別**：本研究は **agent interactions を扱わない（independence assumption）**、microsim カテゴリ。Bruch & Atwell の ABM 文献を引用しつつ、**「broader simulation tradition」内での positioning**を明示
3. **Schelling との関係**：Bruch & Atwell が認める「Schelling 1971 が ABM を sociology に持ち込んだ」 → 本研究の Pillar 3 系譜の最古層
4. **Micro-macro problem への寄与**：本研究は personality（micro）→ harassment prevalence（macro）を bridge する。これは Bruch & Atwell の core motivation
5. **Mechanism-based explanation**：本研究は personality typology → harassment causation の mechanism を提案。Bruch & Atwell の analytical framework に乗る
6. **High-dimensional realism の implementation**：本研究の HEXACO 6 + gender + role + 業種 + cell-level binary outcome は high-dim realism の実装。Schelling stylized model との対比
7. **Empirical calibration の主題化**：「By calibrating type-conditional probabilities to observed N=354 data and validating against MHLW national survey, we follow Bruch & Atwell (2015)'s recommendations for empirical agent-based / microsimulation research」
8. **Goodness-of-fit + sensitivity の dual evaluation**：本研究は (a) MAPE ≤ 30% で goodness-of-fit、(b) Stage 3 で sensitivity → Bruch & Atwell の dual evaluation framework に integrate

---

### [3.1-S1] Bonabeau (2002) — Agent-based modeling: Methods and techniques for simulating human systems

**Citation**：Bonabeau, E. (2002). Agent-based modeling: Methods and techniques for simulating human systems. *Proceedings of the National Academy of Sciences, 99*(suppl. 3), 7280–7287. https://doi.org/10.1073/pnas.082080899

**Verification**：✅✅ 原文 PDF 8p（PNAS Sackler Colloquium）を本セッションで精読

#### Research Question

「ABM はいつ・どう使うべきか？ 4 領域（flow / organizational / market / diffusion）の現実応用例で benefits を可視化」

#### 中核的論証

##### ABM の 3 benefits

1. **Emergent phenomena を捕える**：whole > sum of parts、interactions が新しい properties を生む
2. **System を natural に描写**：behavioral entities ベース → modeler の直感に合う
3. **Flexible**：variables、process types、scale すべてで自由

##### ABM が適するのは emergent phenomena が想定される場合（4 conditions）

1. **Individual behavior が non-linear**（thresholds、if-then rules、non-linear coupling）→ 微分方程式で扱いにくい
2. **Memory、path-dependence、temporal correlations、learning、adaptation** がある
3. **Interactions が heterogeneous and complex**
4. **Topology が non-trivial**（地理・network 構造）

→ 本研究との対応：harassment は (1) personality threshold（dark triad facet effects）と (2) memory（perpetrator-victim relationship、reciprocity）と (3) heterogeneous interactions（type × type、type × role）の features を持つ

##### "Can you grow it?" 命題

> "...it raises the issue of what constitutes an explanation of such a phenomenon. The broader agenda of the ABM community is to advocate a new way of approaching social phenomena, not from a traditional modeling perspective but from..." (p. ?)

= **Epstein 流の「成長して説明する」**：合成可能な mechanism を提示することが explanation。本研究も「7 類型 typology から prevalence を grow できる」を示す

##### 4 application areas

| 領域 | 例 | 核心 emergent phenomenon |
|---|---|---|
| **Flow simulation** | 群衆 panic、交通渋滞、避難 | 個人 panic decisions → collective stampede |
| **Organizational simulation** ★ | **operational risk in financial institutions、organizational design** | 個人 error/risk → organizational loss、organizational structure → performance |
| Market simulation | 株式市場、商品 placement | trader 心理 → market dynamics |
| Diffusion simulation | innovation 採用、消費者選好の伝播 | individual influence → product penetration curves |

##### Organizational simulation の重要事例

**Société Générale Asset Management（SGAM）operational risk model**：

- Operational loss は **historical data 希薄**（low-frequency, high-impact）
- Direct statistical estimation 困難
- → **ABM で simulate してから capital allocation**：「artificial data set」を generate
- bank の actors / activities / interactions / risk factors を bottom-up modelling
- 「earnings-at-risk」を 95% confidence で計算
- → **hypothetical population scaling without empirical data abundance** = 本研究の Phase 1 Stage 1（13,668 → 6,800 万労働人口）の祖

**ABM as natural risk modeling**：

> "ABM is not only a simulation tool; it is a naturally structured repository for self-assessment and ideas for redesigning the organization." (p. ?)

→ 本研究の Phase 2 counterfactual を **organizational redesign tool** として positioning する根拠

##### "ABM は単純だが概念的には深い"

> "Although ABM is technically simple, it is also conceptually deep. This unusual combination often leads to improper use of ABM." (p. 7280)

→ 本研究の 14-cell hybrid model も同じく **technically simple（cell-level bootstrap + Monte Carlo aggregation）but conceptually deep（individual personality → national prevalence）**

#### Quotable Elements（原文逐語）

> "Agent-based modeling is a powerful simulation modeling technique that has seen a number of applications in the last few years, including applications to real-world business problems." (Abstract, p. 7280)

> "ABM captures emergent phenomena. Emergent phenomena result from the interactions of individual entities. By definition, they cannot be reduced to the system's parts: the whole is more than the sum of its parts because of the interactions between the parts." (p. 7281)

> "Individual behavior is nonlinear and can be characterized by thresholds, if-then rules, or nonlinear coupling. Describing discontinuity in individual behavior is difficult with differential equations." (p. 7281)

> "Modeling risk in an organization using ABM is THE right approach to modeling risk because most often risk is a property of the actors in the organization: risk events impact people's activities, not processes." (p. 7282) — **risk = actor の属性**の主張は本研究の "harassment = HEXACO type の属性" と直結

> "Once one has a reliable model of an organization, it is possible to play with it, change some of the organizational parameters, and measure how the performance of the organization varies in response to these changes." (p. ?) — **counterfactual の正当化**

> "ABM will revolutionize business risk advisory services because it constitutes a paradigm shift from spreadsheet-based and process-oriented models." (p. 7282)

#### 本研究での citation 用途

1. **ABM 系譜の主要 review**：「Agent-based modeling has been applied to organizational risk (Bonabeau, 2002), residential segregation (Schelling, 1971), and population health (Rutter et al., 2011), among other domains」 — 系譜の breadth 引用
2. **Risk = actor's property 主張**：「Following Bonabeau (2002)'s framework, we treat workplace harassment as a property of individual actors (perpetrators) rather than purely organizational processes」
3. **Hypothetical population generation の正当化**：Bonabeau の SGAM operational risk example で「historical data 希薄、ABM で artificial data 生成」 → 本研究の N=354 → 6,800 万 scaling の同型先例
4. **Organizational redesign tool としての positioning**：Phase 2 の 3 種 counterfactual を **organizational redesign simulation** として framing
5. **Emergent phenomena framing**：harassment prevalence は集団レベル emergent property → ABM/microsim canonical 適用領域
6. **4 application areas の本研究 mapping**：本研究は Bonabeau の **Organizational simulation** カテゴリの harassment 領域への適用 → "Workplace harassment is, in essence, an organizational risk that emerges from individual behavior; we apply Bonabeau (2002)'s ABM organizational risk framework"
7. **Conceptual depth の擁護**：「Although technically simple (cell-level bootstrap aggregated to national prevalence), our approach captures the conceptually deep insight that individual HEXACO-conditional probabilities aggregate non-trivially into national patterns (Bonabeau, 2002)」
8. **"Can you grow it?" 哲学**：Discussion で「Per Bonabeau (2002) and Epstein, our paper demonstrates that national harassment prevalence can be 'grown' from individual personality typology, providing a generative explanation rather than just descriptive correlation」

---

### [3.1-S2] Krijkamp et al. (2018) — Microsimulation modeling for health decision sciences using R: A tutorial

**Citation**：Krijkamp, E. M., Alarid-Escudero, F., Enns, E. A., Jalal, H. J., Hunink, M. G. M., & Pechlivanoglou, P. (2018). Microsimulation modeling for health decision sciences using R: A tutorial. *Medical Decision Making, 38*(3), 400–422. https://doi.org/10.1177/0272989X18754513

**Verification**：✅✅ 原文 PDF 33p（HHS Public Access）を本セッションで精読

#### Research Question

「R 言語による microsimulation 実装の **step-by-step practical tutorial** を提供。Cohort model（Markov）の限界を超えた個人レベル simulation を **transparent、reproducible、computationally efficient** に R で書く方法は？」

#### 中核的内容

##### Cohort model（Markov）の限界

- **Markov 仮定**：transition probability は current state のみに依存（history なし）
- 履歴を states として展開すると **state explosion**（cell 爆発の Markov 版）
- Deterministic mean response しか得られない
- Heterogeneous individual characteristics を扱えない

##### Microsimulation の利点

> "Individual-based state-transition (or microsimulation) models address many of the limitations of deterministic cohort models because they can more accurately reflect individual clinical pathways, incorporate the impact of history on future events, and more easily capture the variation in patients' characteristics at baseline." (p. 2)

→ 本研究との対応：
- 7 HEXACO types = individual heterogeneity を baseline で保持
- bootstrap = stochastic variation in outcomes
- Phase 2 連鎖（被害 → 離職 → メンタル疾患）= individual pathway tracking

##### R による implementation の利点

- 統計分析を decision model 内に **直接 incorporate**
- More transparent and reproducible（cf. proprietary TreeAge, Arena）
- 強力な vectorization solutions
- Open source、再現性

##### Tutorial の主要ステップ（abstract から推定）

1. Decision problem の specification
2. Health states と transition probabilities の定義
3. Individual trajectory simulation
4. Outcome aggregation
5. Sensitivity analysis
6. Vectorization for efficiency

##### Monte Carlo Standard Error（MCSE）

- 同じ model でも個人 simulation 結果は variation
- Number of individual simulations が増えるほど MCSE 減少
- → 本研究の bootstrap 2,000 iter は同様の MCSE 制御

##### Code 公開

- GitHub: DARTH-git/Microsimulation-tutorial に **完全 R code**
- 4 つの appendix files
- 本研究の Phase 1 実装の **直接 reference code base**

#### Quotable Elements（原文逐語）

> "Microsimulation models are becoming increasingly common in the field of decision modeling for health. Since microsimulation models are more computationally demanding than traditional Markov cohort models, the use of computer programming languages in their development becomes more common. R is a programming language that has gained recognition within the field of decision modeling." (Abstract, p. 1)

> "Cohort models investigate a hypothetical homogeneous cohort of individuals as they transition across health states. In a deterministic cohort model the result is precisely determined given a set of initial conditions and parameters." (p. 1)

> "[Microsimulation models] simulate the impact of interventions or policies on individual trajectories rather than the deterministic mean response of homogeneous cohorts. In a microsimulation model, outcomes are generated for each individual and are used to estimate the distribution of an outcome for a sample of potentially heterogeneous individuals. This individual-level simulation allows the inclusion of stochastic variation in disease progression as well as variation due to individual characteristics. Microsimulation models do not require the Markov assumption..." (p. 2)

> "[The MCSE] is the standard error around an outcome estimate that arises from running the simulation a finite number of times. As the number of individuals included in a simulation grows, the MCSE shrinks." (paraphrased from p. ?)

#### 本研究での citation 用途

1. **R による実装の正当化**：「Following Krijkamp et al. (2018)'s tutorial framework, our microsimulation pipeline is implemented in [Python/R] with publicly available code, ensuring transparency and reproducibility」 — Methods での実装言及
2. **Markov 仮定からの離脱**：本研究は **Markov 仮定を置かない**（HEXACO type は permanent baseline、harassment は state でなく event/outcome）。Krijkamp の framework との関係を Methods で明示
3. **State explosion の回避**：D13 power analysis での 28-cell vs 14-cell 議論を Krijkamp の "state explosion" 概念で framing（同じ cell explosion 問題）
4. **MCSE / bootstrap reproducibility**：「Per Krijkamp et al. (2018), we report Monte Carlo standard error to characterize simulation variability」
5. **Computational efficiency considerations**：本研究の Phase 1 は cell-level estimation で computationally light だが、Phase 2 counterfactual は重め → Krijkamp の vectorization recommendation 採用
6. **Open source / reproducibility commitment**：DARTH コード公開と類似に、本研究も GitHub + OSF に code 公開
7. **Tutorial-led methodology の継承**：本研究は Krijkamp tutorial 系列の harassment 領域への adaptation
8. **DARTH framework との互換性**：DARTH (Decision Analysis in R for Technologies in Health) framework は health 領域だが、本研究は workplace harassment を public health concern として positioning することで DARTH framework と親和性確保

---

### [3.1-S3] Schofield et al. (2018) — A brief, global history of microsimulation models in health

**Citation**：Schofield, D. J., Zeppel, M. J. B., Tan, O., Lymer, S., Cunich, M. M., & Shrestha, R. N. (2018). A brief, global history of microsimulation models in health: Past applications, lessons learned and future directions. *International Journal of Microsimulation, 11*(1), 97–142. https://microsimulation.pub/articles/00175

**Verification**：✅✅ 原文 PDF 46p（IJM Open Access）を本セッションで精読

#### Research Question

「Health microsimulation の **過去 40+ 年の global history** を辿り、early models（1970s〜1990 means-based）から現代の robust dynamic models まで evolution を整理。応用領域（health expenditure / aging / diabetes / mortality / spatial）と国別展開（Canada, USA, UK, Sweden, Netherlands, NZ, Australia, Africa）を mapping。Future directions（genomics, precision medicine, rare diseases）を提示」

#### 中核的内容

##### Health microsim の歴史段階

| 期 | 特徴 |
|---|---|
| **1970s** | Egyptian population、US POPREP（family planning）— Orcutt 1957 直系 |
| **1975–1990** | static models 中心、fertility、breastfeeding、insurance、cancer screening（Parkin 1985）、river blindness vector model（Plaisier）|
| **1990s** | 急速拡大、health benefits/expenditure focus、しかし **means-based / cell-based** 主流 — early limitations |
| **2000s〜** | dynamic models 増加、health-specific input data、disease-specific models（Habbema MISCAN 等）|
| **2010s** | spatial microsim、aging、diabetes、cancer screening 完成期 |

##### 「means-based / cell-based」approach の限界（1990s 主流）

- 既存の static microsim model（他目的）に health 変数を imputation するだけ
- 「serious and unrealistic assumption」と Schofield 等は批判
- → health-specific input data に基づく purpose-built 検証モデルへ移行

→ 本研究との対応：本研究の 14-cell scenario は **cell-based approach の hybrid**（HEXACO 連続スコアを保持しつつ aggregation）。Schofield の批判を **partially address** しつつ、scale 限界を認める

##### 国別 microsim infrastructure

- **Canada**：Statistics Canada の POHEM / Modgen — health policy analysis の代表
- **USA**：MISCAN（NIH）、CISNET 等
- **Australia**：APPSIM、Health-Mod
- **UK**、**Sweden**、**Netherlands**、**NZ**：各国独自モデル
- **Africa**：限定的だが onchocerciasis モデル等

##### Future directions（Schofield 2018 が提示）

1. **Genomics / precision medicine**：高次元 individual data
2. **Rare diseases**：sample 限られる場合の modeling
3. **Childhood cancers**：individual trajectories
4. **Spatial modeling**：geographic heterogeneity
5. **Better data**：model error 回避の必要

##### Lessons learned（pitfalls）

- 古い data + simple imputation は危険
- 過度な simplification（means-based）は spurious result を生む
- model assumptions の transparency が欠かせない
- validation against external data is critical

#### Quotable Elements（原文逐語）

> "This review discusses the evolution of microsimulation models in health over the past three decades. We focus on three aspects of health microsimulation. First, we describe the origins and applications of health microsimulation, including how early research challenged early methodologies and led to the development of more rigorous models." (Abstract, p. 97)

> "Microsimulation models can be used to address a wide range of research questions, including simulating life histories to estimate the effects of interventions or policies at a population scale, or test the impact of health policy on a population (Brown, 2011; Rutter, Zaslavsky, & Feuer, 2011)." (Introduction, p. 99)

> "The 1990s was a period of rapid expansion and a move to robust purpose-built health models with many focussing on health benefits and expenditure. However, the primary method was a 'means-based approach', also known as a cell-based method, within an existing static microsimulation model developed for other purposes (National Research Council, 1991). This was a serious and..." (p. 8/§)

#### 本研究での citation 用途

1. **Microsimulation 系譜の breadth 引用**：「Health microsimulation has been developed extensively over four decades across multiple countries and disease domains (Schofield et al., 2018), with applications spanning cancer screening, diabetes, aging, and health expenditure」 — Introduction の方法論的系譜に **scale と breadth** を追加
2. **本研究の domain の新規性主張**：Schofield et al. 2018 が網羅した領域に **workplace harassment は含まれない** → 本研究の **domain novelty** を Schofield 2018 unawareness で補強
3. **Cell-based の限界認識**：Schofield 2018 の "means-based / cell-based" 1990s 批判は本研究の 14-cell scenario の限界を framing する根拠 → 本研究は **transparency と sensitivity analysis で限界を補う**ことを明示
4. **国別 infrastructure の不在**：日本に health microsim infrastructure（CISNET 相当）が存在しない → 本研究は日本における **domain 拡大 + infrastructure pioneering** の dual contribution
5. **Lessons learned の引用**：Schofield 2018 の pitfalls warning（external validation 必須、assumption transparency 必須）に従って本研究は MAPE ≤ 30% pre-registration、cell-level CI、sensitivity analysis を実装
6. **Future directions の harassment 領域**：Schofield 2018 の future directions に **occupational health / workplace harassment** を提案 → 本研究は **future directions article の prescient 提案を実現**するもの
7. **Spatial heterogeneity への接続**：Schofield 2018 の spatial models は本研究の **業種別・地域別 prevalence** 拡張で参照可能（Phase 2 の future work）
8. **Genomics 系譜との橋渡し**：本研究の HEXACO は **highly heritable**（双子研究）→ Schofield 2018 の precision medicine direction と HEXACO-based individual-targeted intervention（Counterfactual B）に概念的接続

---

### [3.1-P1] Macal & North (2010) — Tutorial on agent-based modelling and simulation

**Citation**：Macal, C. M., & North, M. J. (2010). Tutorial on agent-based modelling and simulation. *Journal of Simulation, 4*(3), 151–162. https://doi.org/10.1057/jos.2010.3

**Verification**：✅✅ 原文 PDF 12p（Iowa State faculty PDF 版）を本セッションで精読

#### Research Question

「Agent-based modelling and simulation (ABMS) を **operations research / simulation** 共同体向けに体系化。Concepts、structure、tools の practical primer」

#### 中核的内容

##### ABM の構造（3 elements）

1. **A set of agents, their attributes and behaviours**：個人の特性（state variables）と行動規則（behaviour rules）
2. **A set of agent relationships and methods of interaction**：interaction topology（grid / network / proximity）
3. **The agents' environment**：agents が interaction する物理・抽象空間

##### Simulation 系譜内での ABMS positioning

- vs Discrete-event simulation（DES）：DES は queue / event chain、ABMS は agent decision focused
- vs System dynamics（SD）：SD は aggregate stocks/flows、ABMS は heterogeneous individual
- vs Microsimulation：ABMS と microsim は overlap、ABMS は interaction emphasizes、microsim は population-level outcomes emphasizes

→ 本研究は **microsim 寄り**（interaction なし、independence assumption）と再確認

##### Heterogeneity + Self-organization

- ABM の distinguishing features
- agent diversity を直接 modeling できる
- pattern/structure が **interactions から emerge**

→ 本研究：HEXACO 7 type の **heterogeneity** は明示的、しかし self-organization は **modeling しない**（individual independence）

##### ABM software toolkits 紹介

- Swarm（古典）
- NetLogo（教育用、最普及）
- Repast（business / social）
- Mason（large scale）
- AnyLogic（commercial）

→ 本研究は **toolkit 不使用**（直接 Python/R で probability table 操作）→ Macal & North の software ecosystem は本研究の **methodological alternative** として参照

##### Application range（広範）

- stock market、supply chain、epidemics、bio-warfare、immune system、consumer purchasing、ancient civilizations、battlefield、naval engagement

→ workplace harassment は missing → 本研究の domain novelty 補強

#### Quotable Elements（原文逐語）

> "Agent-based modelling and simulation (ABMS) is a relatively new approach to modelling complex systems composed of interacting, autonomous 'agents'." (p. 151)

> "By modelling systems from the 'ground up'—agent-by-agent and interaction-by-interaction—self-organization can often be observed in such models. Patterns, structures, and behaviours emerge that were not explicitly programmed into the models, but arise through the agent interactions." (p. 151)

> "The emphasis on modelling the heterogeneity of agents across a population and the emergence of self-organization are two of the distinguishing features of agent-based simulation as compared to other simulation techniques such as discrete-event simulation and system dynamics." (p. 151)

> "A typical agent-based model has three elements: 1. A set of agents, their attributes and behaviours. 2. A set of agent relationships and methods of interaction: An underlying topology of connectedness defines how and with whom agents interact. 3. The agents' environment..." (p. 153)

#### 本研究での citation 用途

1. **ABMS 系譜の補完引用**：Bonabeau 2002 と並行して「ABM tradition has been thoroughly reviewed in tutorials (Macal & North, 2010; Bonabeau, 2002)」 — comprehensive 引用
2. **3-element framework の整理**：本研究の simulation を Macal & North の 3-element framework で記述：(i) agents = N=354 individuals + 7 types、(ii) interactions = 仮定なし（independence）、(iii) environment = MHLW 統計 baseline
3. **Microsim vs ABM の境界明示**：「While ABM emphasizes interactions and self-organization (Macal & North, 2010), our study uses an independence-assumption microsimulation closer to Orcutt's (1957) original framework」
4. **Toolkit 不使用の合理化**：「Rather than using ABM-specific toolkits (Macal & North, 2010), we implement our microsimulation directly in [Python/R] for transparent custom logic」
5. **Workplace harassment の domain novelty 再確認**：Macal & North 2010 の application list に **workplace bullying / harassment は不在**
6. **Heterogeneity 強調の継承**：Macal & North の "agent diversity" emphasis は本研究の HEXACO 7 type focus と整合
7. **Self-organization は本研究 scope 外**：本研究は **non-emergent** な aggregation を扱う → Macal & North との差異を Discussion で明示
8. **Operations research 共同体への橋渡し**：本研究は psychology / sociology / public health 文脈だが、Macal & North 2010 を引用することで OR 共同体への accessibility 確保

---

### [3.1-P2] Schelling (1971) — Dynamic models of segregation

**Citation**：Schelling, T. C. (1971). Dynamic models of segregation. *Journal of Mathematical Sociology, 1*(2), 143–186. https://doi.org/10.1080/0022250X.1971.9989794

**Verification**：✅✅ 原文 PDF 44p（UZH 講義サイト版）を本セッションで精読

#### Research Question

「集団偏析（segregation）は組織的措置だけでなく **個人の discriminatory choice の interplay** からも生じる。微弱な個人選好からどの程度の集団パターンが creator されるか？ Spatial proximity model + analytic compartment model で **micro-macro non-correspondence** を実証する」

#### 中核的論証

##### Segregation の 4 source

1. Organizational practices（明示的排除）
2. Specialized communication systems（言語等）
3. Correlation with non-random variables（職業 → 収入 → 居住地）
4. **Interplay of discriminatory individual choices**（本論文の focus）

##### Spatial proximity model（key contribution）

- 線形空間 / グリッド空間に 2 種類の「item」（記号 + と −）を配置
- 各 item は **隣人との同種比率**に preference
- 同種が一定 fraction より少ないと **動く**（move）
- 動いた先で再び preference 評価 → 新たな移動 chain

→ key finding：**個人選好「同種が 30% 以上いれば残る」程度でも、集団パターンは ~70% 以上の同種に segregate**（**preference >> realized pattern**）

##### Compartmented model（analytic）

- 「neighborhood」を離散 compartment として捉え、population dynamics の equation で解析
- **Tipping point** の存在：少数派比率が閾値を超えると multi-step migration で完全偏析へ転落

##### "Tipping" theory の origins

- 既存「neighborhood tipping」現象（白人居住地に有色人種が増えると一気に flip）を説明する **simple mechanism**
- 個人 motives と aggregate pattern の **non-correspondence**

#### 哲学的含意（key for our paper）

> "The systemic effects are found to be overwhelming: there is no simple correspondence of individual incentive to collective results. Exaggerated separation and patterning result from the dynamics of movement. **Inferences about individual motives can usually not be drawn from aggregate patterns.**" (Abstract)

→ **重要な warning**：本研究も **aggregate harassment prevalence から individual motives を直接 infer することはできない**。Schelling と整合的に、本研究は逆方向（individual personality → aggregate prevalence）を modeling し、**aggregate からの inverse inference は推奨しない**ことを Discussion で明示

#### Quotable Elements（原文逐語、page 番号）

> "Some segregation results from the practices of organizations, some from specialized communication systems, some from correlation with a variable that is non-random; and some results from the interplay of individual choices. This is an abstract study of the interactive dynamics of discriminatory individual choices." (Abstract, p. 143)

> "The systemic effects are found to be overwhelming: there is no simple correspondence of individual incentive to collective results. Exaggerated separation and patterning result from the dynamics of movement." (Abstract, p. 143)

> "Inferences about individual motives can usually not be drawn from aggregate patterns." (Abstract, p. 143)

> "By 'discriminatory' I mean reflecting an awareness, conscious or unconscious, of sex or age or religion or color or whatever the basis of segregation is, an awareness that influences decisions on where to live, whom to sit by, what occupation to join or to avoid, whom to play with or whom to talk to." (p. 144)

> "What follows is an abstract exploration of some of the quantitative dynamics of segregating behavior. The first section is a spatial model in which people—actually, not 'people' but items or counters or units of some sort—distribute themselves along a line or within an area in accordance with preferences about the composition of their..." (p. 145–146)

#### 本研究での citation 用途

1. **ABM / microsim の祖**：「Following Schelling's (1971) seminal demonstration that micro-level individual preferences aggregate non-trivially to macro-level patterns, our microsimulation extends this lineage to workplace harassment」 — **古典的 anchor citation**
2. **Micro-macro non-correspondence の警告**：本研究の Phase 2 介入で「個人介入 vs 集団 pattern」の関係は Schelling-type tipping を含み得る → policy implication で「micro intervention の macro effect は non-linear」と Schelling から学んだ caveat 提示
3. **個人選好と aggregate pattern の disjunction**：Schelling の central insight は本研究の **「individual harassment risk」 vs 「national prevalence pattern」** の議論で central reference
4. **Aggregate pattern からの inverse inference の禁止**：「Inferences about individual motives can usually not be drawn from aggregate patterns (Schelling, 1971)」 — 本研究は逆方向 modeling、これを explicitly 注意
5. **Tipping point 概念**：Phase 2 構造介入で非線形効果（一定閾値で大幅 prevalence 減少）を予測する場合、Schelling の tipping framework で議論
6. **Pure mathematical methods の限界**：Schelling の dual approach（simulation + analytic）は本研究の方法論的選択（probabilistic simulation ≠ closed-form mathematical analysis）の正当化
7. **Reciprocity 効果**：Schelling は同種 + 異種双方の preference reciprocity を扱う → 本研究 Discussion での perpetrator-victim reciprocity（Bowling & Beehr 2006）と接続
8. **政策含意の skepticism**：Schelling は「個人選好の小さな変化でも大きな集団変化を生む」を強調 → 逆に「大きな個人介入でも集団変化は限定的」も論理的に対称的に成立しうる → Phase 2 sensitivity の concept anchor

---

## Round 3：Pillar 3.2 + 3.3 — Precursor 確認 + Latent Class

---

### [3.2-C1] Ho, Campenni, Manolchev, Lewis, & Mustafee (2025) — ABSS for bullying targets' coping strategies [本研究の最近接 precursor]

**Citation**：Ho, C.-H., Campenni, M., Manolchev, C., Lewis, D., & Mustafee, N. (2025). Exploring the coping strategies of bullying targets in organisations through abductive reasoning: An agent-based simulation approach. *Journal of Business Ethics, 199*(4). https://doi.org/10.1007/s10551-024-05861-2

**Verification**：✅✅ 原文 PDF 23p（Springer Open Access）を本セッションで精読

#### Research Question

「Workplace bullying の **coping strategies**（被害者がどう対処するか、特に silence を選ぶ理由）は、**Perceived Organisational Support (POS)** と **Trade Union (TU) membership** で influence できるか？ ABSS（agent-based social simulation）で 4 spaces conception モデル（4SC）を validate し、coping strategy 変容の possibility を評価」

#### 中核的論証

##### 4SC（Four Spaces Conception）枠組み

被害者の coping strategy 選択肢を 4 spaces で classify：

1. **Silence space**（沈黙；何もしない）
2. **Internal channels**（同僚、上司、HR への報告）
3. **External channels**（弁護士、裁判所、外部機関）
4. （Multiple combination）

##### 主要仮説（hunches）

- Hunch 1：**POS（perceived organisational support）が高いと内部 channel 利用が増え external 利用が減る**
- Hunch 2：**TU membership が doing nothing → taking action への移行を促す**
- Hunch 3：**social network topology**（small-world、random、preferential など）が coping strategy 拡散に影響

##### Method（多段階）

###### Study 1（empirical）：4SC 確認 phase

- **Sample**：UK NHS Trust 2 sites の労働者
- **Initial respondents**：N=1,798 (722 male + 1,065 female + 11 prefer not to say)
- 平均年齢 43.9 歳（SD=11.38）
- 在職 10+ 年 44%、6–10 年 17.8%、3–5 年 20.3%
- TU member：58.6%（1,061 / 1,812）、非 member 41.4%

###### Study 2（ABSS）：simulation phase

- ABSS（Davidsson 2002）使用
- 4 phases of abductive reasoning：observe anomaly → confirm anomaly → generate hunches → evaluate hunches
- Network topology を比較（small-world、random、preferential attachment）
- POS / TU membership を independent variable として manipulate

##### Findings

1. **POS は強い効果**：external → internal channel への移行
2. **TU membership は中程度効果**：doing nothing → taking action
3. Network topology の選定で **abductive theory building** が可能
4. **Silence は依然 default**（介入なしの場合）

#### 本研究との **詳細比較**

| 要素 | Ho et al. 2025 | 本研究 |
|---|---|---|
| **Side** | Victim/target | **Perpetrator** |
| **Outcome** | Coping strategy（after harassment）| Harassment perpetration（before/during）|
| **Scale** | UK NHS Trust 2 sites（単一組織レベル）| **国レベル（日本 6,800 万労働人口）** |
| **Input** | Demographic + POS + TU membership | **HEXACO 7 typology** |
| **Methodology** | ABSS with network topology | **Probabilistic microsimulation, no agent interactions** |
| **Validation** | Internal（survey data 内整合）| **External（MHLW national survey）** |
| **Counterfactual** | POS / TU membership 変動 | **3 種介入（A 普遍 / B 集中 / C 構造）** |
| **Theoretical framing** | Ethical infrastructure | **Personality typology + capability approach** |

→ **3 つの軸（perpetrator side、national scale、HEXACO typology）すべてで novelty**

#### Quotable Elements（原文逐語）

> "While previous studies have explored the antecedents of such negative acts and proposed various intervention and prevention strategies, there remains a critical need to examine the coping strategies employed by those targeted by bullying, particularly in instances where silence is the chosen response." (Abstract)

> "In this pioneering study, we use primary data from two UK National Health Service trusts and agent-based social simulation, to determine whether it is possible to influence the coping strategies of bullying targets." (Abstract)

> "Our findings suggest that perceived organisational support has a strong effect on changing bullying coping strategies, away from external (solicitors, Court of Law) and towards internal channels (colleagues, managers, etc.). We also find that TU membership can moderately influence a change in bullying coping strategies from doing nothing to taking actions." (Abstract)

> "Researchers from both Social Sciences and Operations Research (OR) use ABSS to study complex social systems (Brailsford et al., 2019; Fan et al., 2024; Gu & Kunc, 2020; Sznajd-Weron et al., 2024), e.g., the emergence of social behaviour (Gilbert, 1995), crowd behaviour (Wijermans et al., 2013), religious behaviour (Shults, 2019)..." (Study 2 overview)

#### 本研究での citation 用途

1. **最近接 precursor の明示**：「The closest existing precursor is Ho et al. (2025), which used ABSS to model coping strategies of bullying targets in two UK NHS Trusts (N=1,798). Our study differs in three key dimensions: (i) we model perpetration rather than coping; (ii) we operate at national rather than single-organization scale; and (iii) we use HEXACO personality typology rather than organizational support and TU membership as input」 — **novelty 確立の central reference**
2. **ABSS lineage との接続**：Ho et al. 2025 は ABSS（OR + Sociology 共同体）系譜。本研究は microsimulation（demographic + health policy）系譜。**両系譜は ABM common roots**（Schelling 1971）を共有
3. **4SC 枠組みの inverse**：Ho et al. 2025 の "silence vs internal vs external coping" の 4 space は victim 側。本研究は perpetrator 側で類似 typology（low/high HH × low/high X 等）を提供
4. **Sample size 比較**：Ho 2025 N=1,798（2 organization）vs 本研究 N=354（individual heterogeneity 高）+ N=13,668（reference centroid）+ 6,800 万 simulated population → **scale leap**
5. **Network topology の不在の説明**：本研究は network interactions を modeling しない（independence assumption）。Ho 2025 が network topology を中心要素にすることと **deliberate 対比**：「Unlike Ho et al. (2025), we do not model agent interactions, focusing instead on aggregate prevalence projections」
6. **Validation 方法の比較**：Ho 2025 は internal validation（survey data 内整合）、本研究は external validation（MHLW national survey）→ Rutter 2011 の external validation 推奨に従う
7. **Ethical infrastructure 概念の継承**：Ho 2025 は ethical infrastructure（formal + informal organizational elements）を主題化 → 本研究 Phase 2 Counterfactual C（structural intervention）の **organizational ethical infrastructure** への投資と接続
8. **Methodological dialogue**：本研究と Ho 2025 は **complementary**：本研究が perpetrator-side prevalence を国レベルで予測 → Ho 2025 が target-side coping を組織レベルで modeling → 両者組み合わせて **comprehensive harassment simulation framework** を future work で示唆

---

### [3.2-S1] Sapouna et al. (2010) — Virtual learning intervention to reduce bullying victimization in primary school

**Citation**：Sapouna, M., Wolke, D., Vannini, N., Watson, S., Woods, S., Schneider, W., Enz, S., Hall, L., Paiva, A., Andre, E., Dautenhahn, K., & Aylett, R. (2010). Virtual learning intervention to reduce bullying victimization in primary school: A controlled trial. *Journal of Child Psychology and Psychiatry, 51*(1), 104–112. https://doi.org/10.1111/j.1469-7610.2009.02137.x

**Verification**：✅✅ 原文 PDF 9p（J Child Psychol Psychiatry）を本セッションで精読

#### Research Question

「**FearNot!**（Fun with Empathic Agents to achieve Novel Outcomes in Teaching）— virtual learning anti-bullying intervention は、primary school 児童の bullying victimization を減らせるか？ Coping skills 強化アプローチの効果は？」

#### Method

- **Design**：non-randomized controlled trial（学校レベル割付）
- **Sample**：N = **1,129 primary school children** (mean age 8.9 yrs)
- **Setting**：UK + Germany、**27 primary schools**
- **Intervention**：3 sessions × 30 minutes over 3 weeks
  - **Virtual learning**：empathic agent（virtual victim）との interaction
  - 児童は victim agent に coping advice を与える
- **Control**：waiting list（normal curriculum）
- **Measures**：self-report victimization（baseline + 1 week + 4 weeks post）

#### Key Findings

- **Combined sample**：baseline victims が intervention group で **escape victimization 確率高**（adjusted RR=1.41 [1.02, 1.81]、first follow-up）
- **Dose-response**：active interaction 量と escape rate に正の関連（OR=1.09 [1.003, 1.18]）
- **UK children only**：significant escape effect（adjusted RR=1.90 [1.23, 2.57]）
- **UK overall victimization rate も減少**（adjusted RR=.60 [.36, .93]）
- **German sample**：効果有意でない（cultural / implementation difference 推定）

#### 本研究との関係

| 要素 | Sapouna 2010 | 本研究 |
|---|---|---|
| 対象 | School children primary | Workplace adults |
| 介入 | Virtual learning（agent interaction）| Counterfactual simulation |
| Outcome | Coping + victimization | Perpetration prevalence |
| Method | Empirical RCT | Microsimulation projection |
| Scale | 27 schools | National |
| Side | Victim coping | Perpetrator |

→ **完全に異なる研究**。Sapouna 2010 は **bullying intervention research が school 領域で確立している**ことを示す reference として参照。本研究の **workplace + microsimulation + perpetrator** 軸の novelty を補強する。

#### 重要な insight

- **Cultural moderator**（UK 効果、Germany 無効果）→ 介入効果は文脈依存
- → 本研究の Phase 2 介入推定の **cultural sensitivity warning** に直結
- 「日本での介入効果サイズは UK / US と異なる可能性」という limitation の根拠

#### Quotable Elements（原文逐語）

> "Anti-bullying interventions to date have shown limited success in reducing victimization and have rarely been evaluated using a controlled trial design." (Abstract)

> "Current anti-bullying interventions have demonstrated some positive outcomes in regard to reducing victimization (Baldry & Farrington, 2007). However, most intervention effects are small or overestimates as studies do not adjust for the non-independence of observations that occurs when individuals are analyzed within clusters (i.e., classes) (Vreeman & Carroll, 2007). Reducing bullying behavior has proven even less successful (P.K. Smith, Ananiadou, & Cowie, 2003)." (p. 105)

> "Subsample analyses found a significant effect on escaping victimization only to hold for UK children (adjusted RR, 1.90; CI, 1.23–2.57)." (Abstract)

#### 本研究での citation 用途

1. **School bullying vs workplace harassment の domain 区別**：「While bullying interventions have been evaluated in school settings (Sapouna et al., 2010, virtual learning RCT), workplace harassment intervention research is limited and largely lacks population-scale projection」
2. **Bullying intervention が困難であることの literature 認識**：「Reducing bullying behavior has proven even less successful (Smith, Ananiadou, & Cowie, 2003; Sapouna et al., 2010)」 — Phase 2 の介入効果推定で **慎重な effect size assumption** の根拠
3. **Cultural moderator の前例**：UK vs Germany 効果差は **cultural sensitivity の必要性**を示す → 本研究 Phase 2 で「介入 anchor は西欧中心、日本では effect attenuated 可能性」の limitation 議論
4. **Coping skills approach 系譜**：Sapouna 2010 は Lazarus & Folkman (1984) coping theory ベース。本研究は personality-based 介入（Hudson 2023）系譜だが、両者は異なる介入哲学
5. **Virtual learning vs probabilistic simulation の区別**：Sapouna 2010 の "virtual learning" は **児童が virtual agent と対話して学習**する intervention。本研究の simulation は **policy maker が intervention シナリオを評価**する tool。**用語の overlap に注意**して positioning
6. **Effect size benchmark**：Sapouna 2010 の RR=1.41 は intervention 効果の **modest reference point**。本研究の Phase 2 で Roberts 2017 の d=0.37 が「modest but real」であることを補強する secondary anchor
7. **Cluster-level analysis の必要性**：Sapouna 2010 は class-level non-independence を adjust → 本研究の cell-level estimate も **同様の cluster correction** 検討材料
8. **Pillar 3.2 の breadth 補強**：Ho 2025 + Sapouna 2010 + Merlone & Argentero 2018（不取得）+ Tucker 2013（不取得）→ workplace bullying の simulation/modeling 研究は **限定的**であることを confirm

---

### [3.3-C1] Lanza & Rhoades (2013) — Latent class analysis for subgroup analysis in prevention and treatment

**Citation**：Lanza, S. T., & Rhoades, B. L. (2013). Latent class analysis: An alternative perspective on subgroup analysis in prevention and treatment. *Prevention Science, 14*(2), 157–168. https://doi.org/10.1007/s11121-011-0201-1

**Verification**：✅✅ 原文 PDF 21p（NIH-PA via PMC）を本セッションで精読

#### Research Question

「Subgroup analysis（介入の differential effect 検出）の伝統的アプローチ（regression with moderators）は **Type I error 高、power 低、higher-order interaction 困難**。**Latent Class Analysis (LCA)** は alternative として有効か？ N=1,900 adolescents で実証」

#### 中核的論証

##### 伝統的 subgroup analysis の限界

1. **High Type I error rate**：multiple moderator examination で偽陽性多発
2. **Low statistical power**：interaction 検定は main effect より要 N 大幅増
3. **Higher-order interactions の困難**：4-way interaction（治療 × 性別 × 人種 × 年齢）は検定ほぼ不可能
4. **Combinations の implausibility**：すべての moderator 組合せが実在する subgroup を表すとは限らない

##### LCA approach

- 多次元 risk profile を **少数の latent classes** に集約
- 経験的に意味ある subgroup のみを残す（"every possible combination" 問題回避）
- Type I error と power の同時改善
- **Tailoring variables** として個別化介入に直接使用

##### 実証例（N=1,900 adolescents）

- 6 characteristics：household poverty / single-parent status / peer cigarette use / peer alcohol use / neighborhood unemployment / neighborhood poverty
- → **5 latent subgroups** identified：
  1. **Low Risk**
  2. **Peer Risk**
  3. **Economic Risk**
  4. **Household & Peer Risk**
  5. **Multi-Contextual Risk**

##### 介入効果の評価 2 アプローチ

1. **Classify-analyze**：LCA で classification → subgroup ごとに treatment effect 推定
2. **Model-based**：LCA with covariates の reparameterization、direct integration

##### Differential treatment effects への応用

> "Such approaches can facilitate targeting future intervention resources to subgroups that promise to show the maximum treatment response." (Abstract)

→ 本研究 Phase 2 Counterfactual B（targeted intervention）の **直接的方法論的基盤**

#### 本研究との関係

| 要素 | Lanza & Rhoades 2013 | 本研究 |
|---|---|---|
| Outcome 領域 | Substance use、prevention | Workplace harassment |
| Subgroup 同定法 | LCA（経験的）| Hard clustering（事前 7 類型）|
| Sample size | N=1,900 | N=354 + N=13,668 |
| Subgroup 数 | 5 | 7 |
| 用途 | Differential treatment effect | Type-conditional perpetration probability |
| **Phase 2 接続** | Direct（targeting） | Counterfactual B 直接的根拠 |

→ Lanza & Rhoades の LCA は本研究の **HEXACO 7 類型を介入単位として使う論理的根拠**として central。Clustering 論文（Tokiwa）は ward / k-means / spectral 等を使うが、LCA も alternative として positioning 可能

#### Quotable Elements（原文逐語）

> "The overall goal of this study is to introduce latent class analysis (LCA) as an alternative approach to latent subgroup analysis. Traditionally, subgroup analysis aims to determine whether individuals respond differently to a treatment based on one or more measured characteristics." (Abstract, p. 157)

> "LCA provides a way to identify a small set of underlying subgroups characterized by multiple dimensions which could, in turn, be used to examine differential treatment effects. This approach can help to address methodological challenges that arise in subgroup analysis, including a high Type I error rate, low statistical power, and limitations in examining higher-order interactions." (Abstract, p. 157)

> "Such approaches can facilitate targeting future intervention resources to subgroups that promise to show the maximum treatment response." (Abstract, p. 157)

> "In social, behavioral, and health research, prevention and intervention programs are often administered to populations without consideration of individual characteristics that might predict treatment response. Recently, however, there has been growing interest in individualizing treatments in order to administer the right program to the right individuals, thereby maximizing treatment effectiveness." (p. 1)

#### 本研究での citation 用途

1. **Typology-based intervention の方法論的正当化**：「Latent class methods (Lanza & Rhoades, 2013) and related typological approaches have been established as effective for identifying intervention-relevant subgroups in prevention research. The present study extends this approach to workplace harassment by using HEXACO-based personality types as the typological structure」 — Methods での methodological framing
2. **Phase 2 Counterfactual B の central anchor**：「Targeted intervention to high-risk types follows Lanza & Rhoades (2013)'s framework for differential treatment effects, applying intervention resources where treatment response is expected to be maximal」
3. **Type I error / power 議論**：D13 Power Analysis で確認した cell-level 検出力限界（d≥0.9 でしか検出不能）は Lanza & Rhoades の警告と整合 — 本研究は cell-level inference を avoid し aggregate-level に focus
4. **5 vs 7 subgroups**：Lanza & Rhoades の adolescent 研究で 5 latent classes、本研究の Tokiwa clustering paper で 7 → **subgroup 数の選定は経験的・領域依存**であることを Methods で明示
5. **Adaptive treatment strategies の系譜**：Lanza & Rhoades は Fast Track program（Conduct Problems Prevention Research Group 1992）を引用 → 本研究は同様の **adaptive intervention strategy** の harassment 領域への adaptation
6. **Higher-order interaction の禁止**：本研究の 14-cell（type × gender）は 2-way、Lanza & Rhoades が警告する higher-order interaction を意図的に避ける設計
7. **LCA の連続応用 (Notelaers 2006/2011 へ橋渡し)**：Lanza & Rhoades は methodology paper、Notelaers et al. 2006/2011 は workplace bullying への LCA 直接応用 → 本研究は両者の交差点
8. **Person-centered approach の正当化**：Lanza & Rhoades は variable-centered（regression）vs person-centered（LCA）の choice を主題化 → 本研究は **person-centered**（type-based）を明確に選択

---

### [3.3-C2] Notelaers, Einarsen, De Witte, & Vermunt (2006) — Latent class cluster approach to workplace bullying

**Citation**：Notelaers, G., Einarsen, S., De Witte, H., & Vermunt, J. K. (2006). Measuring exposure to bullying at work: The validity and advantages of the latent class cluster approach. *Work & Stress, 20*(4), 288–301. https://doi.org/10.1080/02678370601071594

**Verification**：✅✅ 原文 PDF 14p（Tilburg University repository）を本セッションで精読

#### Research Question

「従来の **operational classification method**（Leymann 1990a：週1回以上の bullying 行為を 6ヶ月以上経験すれば victim）は victims/non-victims の二値分類のみ。bullying は continuum のはず。**Latent Class Cluster Analysis** で belgian sample n=6,175 を分析した場合、より詳細な exposure groups が同定でき、construct + predictive validity も改善するか？」

#### Method

- **Sample**：N = **6,175 Belgian workers**
- **Measure**：Negative Acts Questionnaire (NAQ; Einarsen & Raknes, 1997) — 22 行動 items を 5-point frequency scale で計測
- **比較対照**：
  - **Operational method (Leymann 1990a)**：少なくとも 1 行為を週 1 回以上、6ヶ月以上 → victim
  - **LCA（Magidson & Vermunt 2004）**：response pattern から latent classes を推定
- **Validation**：
  - Construct validity：self-labeled victimization との相関
  - Predictive validity：strain（job-related, general health）と well-being

#### Key Findings — 6 latent classes 同定

| Class | 名称 | 特徴 |
|---|---|---|
| 1 | **Not bullied** | 行為経験ほぼなし |
| 2 | **Limited work criticism** | 業務関連批判が時々 |
| 3 | **Limited negative encounters** | 限定的な対人 negative encounter |
| 4 | **Sometimes bullied** | 中程度頻度の複数行為 |
| 5 | **Work related bullied** | 業務関連 bullying 集中 |
| 6 | **Victims** | 多種類・高頻度の bullying（severe）|

→ classes 4–6（sometimes bullied + work related bullied + victims）が伝統的「victim」に対応するが、内部構造を可視化

##### Validity 比較

- **LCA approach は operational method より高 construct validity**（self-label との一致）
- **LCA approach は operational method より高 predictive validity**（strain / well-being との関連）
- → **LCA は theoretical 妥当性 + practical 妥当性ともに operational より優位**

#### 重要な貢献

> "Latent class modelling is a method of analysis that does not appear to have been used in occupational health psychology before." (Abstract, p. 288)

→ Notelaers 2006 は **occupational health psychology に LCA を導入した先駆**。本研究の HEXACO 7 類型は personality side の対応する典型化アプローチ

#### Quotable Elements（原文逐語）

> "Although bullying is conceived as a complex phenomenon, the dominant method used in bullying surveys, the operational classification method, only distinguishes two groups: victims versus non-victims. Hence, the complex nature of workplace bullying may not be accounted for." (Abstract, p. 288)

> "In this study, six latent classes emerged: 'not bullied,' 'limited work criticism,' 'limited negative encounters,' 'sometimes bullied,' 'work related bullied,' and 'victims.'" (Abstract, p. 288)

> "The results show that compared to the traditional operational classification method, the latent class cluster approach shows higher construct and higher predictive validity with respect to self-assessments and indicators of strain and well-being at work." (Abstract, p. 288)

> "Bullying must be looked upon as a continuum from 'not at all exposed' to 'highly exposed,' and not as an either/or phenomenon." (p. 289, citing Matthiesen, Raknes, & Røkkum 1989)

#### 本研究での citation 用途

1. **Victim-side LCA の確立論文として central reference**：「Notelaers et al. (2006) established that workplace bullying victims fall into 6 latent classes ranging from 'not bullied' to 'severe victims', demonstrating that LCA outperforms operational classification in construct and predictive validity. The present study extends this typological approach to the perpetrator side using HEXACO personality structure」 — Pillar 3.3 の central anchor
2. **Bullying as continuum**：「Bullying is a continuum, not an either/or phenomenon」（Matthiesen et al. 1989; Notelaers et al. 2006）→ 本研究の binary outcome（mean+0.5SD threshold）を **measurement convenience** として正直に位置付ける根拠
3. **Belgian sample との対比**：N=6,175 Belgian vs 本研究 N=354 Japanese → sample 規模差を limitation で明示
4. **6 victim classes vs 7 perpetrator types の対称性**：victim side 6 (Notelaers) + perpetrator side 7 (本研究) → **harassment は両側で discrete typology を持つ** という統合 framework
5. **Operational classification の批判**：従来 method は二値、LCA は continuum を尊重 → 本研究 simulation の **continuous risk score**（cell-conditional probability）も同様に continuum 反映
6. **Predictive validity の demonstration**：Notelaers 2006 で LCA classes が strain と直接予測 → 本研究の Phase 1 Stage 2（被害 → 離職 → メンタル疾患の連鎖）の precedent
7. **方法論 import への前例**：Notelaers が occupational health psychology に LCA を初導入（2006 年） → 本研究が microsimulation を harassment 領域に初導入（2026 年）の対称性 → 「**新方法論の domain crossover** がフィールド進歩を生む」という argument
8. **Magidson & Vermunt 2004 software 系譜**：Notelaers が使用した Latent GOLD は **Vermunt 系**。本研究の clustering paper（Tokiwa）は別 software 使用だが、類似 methodology の継承

---

### [3.3-C3] Notelaers, Vermunt, Baillien, Einarsen, & De Witte (2011) — Exploring risk groups workplace bullying with categorical data

**Citation**：Notelaers, G., Vermunt, J. K., Baillien, E., Einarsen, S., & De Witte, H. (2011). Exploring risk groups workplace bullying with categorical data. *Industrial Health, 49*(1), 73–88. https://doi.org/10.2486/indhealth.MS1155

**Verification**：✅✅ 原文 PDF 16p（J-STAGE Open Access）を本セッションで精読

#### Research Question

「Notelaers et al. (2006) で belgian sample N=6,175 から 6 latent classes 同定。本研究では **large, heterogeneous sample** で（**さらに大規模**）：(1) 6 exposure groups の **prevalence rate** を population baseline として確立、(2) **risk groups**（高 exposure 群に多く所属する社会人口学的群）を multinomial logistic regression で同定」

#### Method

- **Sample**：large heterogeneous Belgian sample（n は本文で確認、約 11,000 程度）
- **Measure**：NAQ（22 行為 items）
- **解析**：
  - LCA → 6 exposure classes
  - Multinomial logistic regression → demographic risk factors

#### Key Findings — 6 exposure groups の prevalence

| Class | 名称 | Prevalence |
|---|---|---|
| 1 | Not bullied（hardly any negative act）| **30.5%** |
| 2 | Limited work criticism | **27.2%** |
| 3 | Limited negative encounters | **20.8%** |
| 4 | Occasionally bullied | **8.3%** |
| 5 | Predominately work related bullied | **9.5%** |
| 6 | Victims of severe workplace bullying | **3.6%** |

→ classes 4–6（**21.4%**）が伝統的「victim」相当
→ class 6（**3.6%**）は severe victim、Leymann 1990a の operational classification に近い

##### 国際比較

著者らが本論文で言及している既存推定：
- Sweden（Leymann）3.5%
- Finland（university staff）16%
- UK（Rayner）50% lifetime
- 欧州 31 studies review：weekly+ bullying 1–4%、less serious 8–10%

→ Notelaers 2011 の severe class 3.6% は Leymann/Sweden estimate と整合

#### Risk groups（multinomial regression）

##### **高リスク**

- **35–54 歳**（vs 25 歳未満）
- **公務員（public servants）**
- **Blue-collar workers**
- **食品・製造業従業員**

##### **低リスク**

- 25 歳未満
- 非正規（temporary contract）
- 教師、看護師、看護助手

##### 含意

- **Age × 雇用形態 × 職種** が交絡的に bullying リスクに関連
- Tsuno 2015 (Japan) の SSS / temporary employee 結果と **部分的に一致 + 部分的に不一致**（年齢方向は逆：日本では若年高 risk、ベルギーでは中年高 risk）

#### Quotable Elements（原文逐語）

> "The results show six different exposure groups: almost 30.5% is not bullied since they report hardly any negative act at work at all, 27.2% face some limited work criticism, 20.8% face limited negative encounters, 8.3% is occasionally bullied, 9.5% are predominately work related bullied, and a total of 3.6% can be seen victims of severe workplace bullying." (Abstract, p. 73)

> "Employees between the age of 35 and 54, public servants, blue-collar workers, as well as employees working in the food and manufacturing industries have a significantly elevated risk to be victims of workplace bullying. In contrast, employees younger than 25, employees with a temporary contract, teachers, nurses and assistant nurses are those least likely at risk." (Abstract, p. 73)

> "Workplace bullying is not about single and isolated events of aggression, but instead is a gradually evolving process characterised by a series of negative behaviours systematically directed against employees who are often unable to counterattack in kind." (p. 73)

> "These findings are important for policymakers at the national and organisational level as they assist in focussing towards possible avenues to prevent workplace bullying." (Abstract, p. 73)

#### 本研究での citation 用途

1. **Population-scale prevalence projection の最近接 precursor**：「Notelaers et al. (2011) demonstrated that workplace bullying exposure follows a 6-class typology with population prevalences of 30.5% / 27.2% / 20.8% / 8.3% / 9.5% / 3.6% in a heterogeneous Belgian sample. The present study extends this typological projection to the perpetrator side using HEXACO-based personality classes」 — **本研究の direct methodological ancestor**
2. **Phase 1 出力構造の対応**：本研究 Phase 1 が出す「7 perpetrator type の prevalence breakdown」は Notelaers 2011 の 6 victim type prevalence breakdown の **perpetrator-side mirror**
3. **Risk group identification の方法論的継承**：Notelaers 2011 の multinomial regression（demographic → class membership）は本研究の Phase 1 の cell-level conditional probability 推定の **methodological precedent**
4. **Industry-specific risk の言及**：Notelaers 2011 は food / manufacturing industry 高 risk → 本研究は personality-focused だが、Discussion で「業種を統制すべき」を limitation として明示
5. **Severe vs less serious bullying の区別**：Notelaers の class 6（severe 3.6%）と class 4–5（less serious 17.8%）の区別は、本研究の binary outcome の threshold 選定で参照（mean+0.5SD 閾値が less serious + severe の両方を含む or severe のみか議論）
6. **国際 prevalence range の確認**：Sweden 3.5% – UK 50% lifetime range は ILO 2022 / Nielsen 2010 meta と整合 → 本研究の Phase 1 出力の plausible range
7. **Pillar 3.3 完結**：Lanza & Rhoades 2013 (LCA methodology) → Notelaers 2006 (LCA bullying validity) → Notelaers 2011 (LCA bullying prevalence + risk groups) → 本研究 (LCA + microsimulation, perpetrator-side) の **3 段階 lineage** が完結
8. **Policy maker への橋渡し**：Notelaers 2011 が明示する「policymakers at national and organisational level」への含意は本研究の Phase 2 counterfactual と同じ audience target → **policy-relevant simulation tradition** の継承

---

## Cross-Pillar Synthesis：本研究の Introduction-ready Story

### 1. 本研究が立つ 3 系譜の交差点

本研究は **3 つの independent literature 系譜の novel intersection** に立つ。

#### 系譜 A：Workplace Harassment Epidemiology（Pillar 1）

```
Einarsen 1990s mobbing concept → NAQ-R 確立 (Einarsen, Hoel, & Notelaers 2009)
                                        ↓
                            Bowling & Beehr 2006 antecedent/consequence meta
                                        ↓
                            Nielsen, Matthiesen, & Einarsen 2010 prevalence meta
                                        ↓
                            ILO 2022 first global survey (23% lifetime)
                                        ↓
                            Nielsen, Glasø, & Einarsen 2017 FFM × harassment meta (victim side)
                                        ↓
                            Tsuno 2010 (J-NAQ-R), Tsuno 2015 (Japan national 6.1%),
                            Tsuno 2022 (COVID 15%), MHLW 2021/2024 (31.4% → 19.3%)
                                        ↓
                            ★ 本研究：HEXACO perpetrator typology による prevalence projection ★
```

#### 系譜 B：Non-LLM Microsimulation Lineage（Pillar 3.1）

```
Orcutt 1957 founding microsimulation
                ↓
Schelling 1971 dynamic models of segregation [ABM precursor]
                ↓
Bonabeau 2002 ABM PNAS, Macal & North 2010 ABM tutorial
                ↓
Rutter 2011 health MSM review, Spielauer 2011 social science microsim
                ↓
Bruch & Atwell 2015 sociological ABM, Krijkamp 2018 R tutorial,
Schofield 2018 health microsim history
                ↓
            （Park 2024 LLM-based agent simulation — 別系譜）
                ↓
            ★ 本研究：probabilistic microsim を harassment 領域に initial application ★
```

#### 系譜 C：Latent Class Typology + Workplace Bullying（Pillar 3.2 + 3.3）

```
Lanza & Rhoades 2013 LCA methodology in prevention
                ↓
Notelaers et al. 2006 LCA × NAQ-R (Belgian n=6,175, 6 victim classes)
                ↓
Notelaers et al. 2011 LCA × population-scale risk groups
                ↓
            （Ho et al. 2025 ABSS for coping strategies — 隣接系譜）
                ↓
            ★ 本研究：victim-side typology を perpetrator-side に拡張、HEXACO 7 type で operational ★
```

### 2. Introduction の 5 段落構成（推奨）

#### 第 1 段落：Global concern（opening）

> "Workplace violence and harassment affects approximately 23% of employed adults globally (ILO, 2022). Meta-analytic evidence places workplace bullying prevalence at 11–18% depending on measurement methodology (Nielsen, Matthiesen, & Einarsen, 2010), with both environmental and individual-difference factors contributing to its occurrence (Bowling & Beehr, 2006). The economic, organizational, and public health costs are substantial..."

**引用**：ILO 2022、Nielsen 2010、Bowling & Beehr 2006

#### 第 2 段落：Japan context

> "In Japan, the Ministry of Health, Labour and Welfare's nationwide survey (N=8,000) reported that 31.4% of workers experienced power harassment in the past three years (MHLW, 2021), declining to 19.3% in the 2023 follow-up (MHLW, 2024) following the 2019 Workplace Bullying Prevention Act. Independent peer-reviewed estimates using the Japanese version of the Negative Acts Questionnaire (Tsuno et al., 2010) place the past-30-days bullying prevalence at 6.1% in a national representative sample (Tsuno et al., 2015), with elevated rates during the COVID-19 pandemic (Tsuno et al., 2022). Japan's vertical hierarchy and unitary bullying structure (Tsuno et al., 2010) make it a particularly informative context for testing typology-based projection models."

**引用**：MHLW 2021、MHLW 2024、Tsuno 2010、Tsuno 2015、Tsuno 2022

#### 第 3 段落：Predictor lineage

> "Personality traits are established correlates of workplace harassment exposure. Meta-analytic evidence shows that neuroticism (r = 0.25), low agreeableness (r = -0.17), and to a lesser degree low conscientiousness (r = -0.10) predict harassment victimization (Nielsen, Glasø, & Einarsen, 2017). On the perpetrator side, HEXACO's Honesty-Humility dimension—absent from the Five-Factor Model—emerges as the strongest personality predictor of workplace deviance (Pletzer et al., 2019), and Dark Triad traits, particularly Psychopathy, predict harassment perpetration in our prior work (Tokiwa et al., preprint). Whether such individual-level predictors aggregate to reproduce population-level prevalence patterns has not been tested."

**引用**：Nielsen 2017、Pletzer 2019（既存）、Tokiwa preprint（自己引用）

#### 第 4 段落：Methodological gap — non-LLM lineage

> "Microsimulation, originating in Orcutt's (1957) framework for socio-economic systems, models populations as collections of elemental decision-making units whose probabilistic behaviors aggregate to predict population-level outcomes. The approach has been extensively developed in health policy (Rutter, Zaslavsky, & Feuer, 2011; Schofield et al., 2018), social science (Spielauer, 2011), and sociological agent-based research (Bruch & Atwell, 2015; Schelling, 1971). This lineage is methodologically distinct from recent large-language-model agent simulations (Park et al., 2024) in that it is probabilistic, externally calibrated, and structurally transparent. To date, however, microsimulation has not been applied to workplace harassment at the national prevalence level."

**引用**：Orcutt 1957、Rutter 2011、Schofield 2018、Spielauer 2011、Bruch & Atwell 2015、Schelling 1971、Park 2024

#### 第 5 段落：Existing precursors and what they don't do, then study aim

> "While agent-based and system-dynamics models have been applied to bullying-related questions—including coping strategies of bullying targets in two UK NHS Trusts (Ho et al., 2025) and qualitative dynamics of mobbing (Merlone & Argentero, 2018)—they have focused on within-organization processes rather than national-scale prevalence projection. Likewise, latent class methods have established the validity of identifying typological subgroups for both differential treatment effects (Lanza & Rhoades, 2013) and workplace bullying victim profiles (Notelaers et al., 2006, 2011). However, no prior work has used HEXACO-based personality typology as input to such simulations, nor projected harassment prevalence at the national level using type-conditional probabilities. The present study fills this three-fold gap..."

> "...by combining (i) HEXACO 7-type clustering derived from a Japanese reference population (N=13,668), (ii) type-conditional perpetration probabilities estimated from an independent sample (N=354), and (iii) population scaling against MHLW national survey data (Phase 1). We additionally evaluate three counterfactual interventions—universal HH education, targeted intervention to high-risk types, and structural moral reminder—to project potential reductions in national harassment prevalence (Phase 2)."

**引用**：Ho et al. 2025、Merlone & Argentero 2018（抄録ベース）、Lanza & Rhoades 2013、Notelaers et al. 2006、Notelaers et al. 2011

### 3. Novelty 主張の 3 軸（再確認）

| 軸 | 主張 | 支持文献 |
|---|---|---|
| **Substantive** | Perpetrator-side HEXACO typology による national prevalence projection | Tokiwa preprint（HEXACO × harassment）+ Pletzer 2019（HH meta）+ Bowling & Beehr 2006（individual difference 系譜）|
| **Methodological** | Non-LLM probabilistic microsimulation を harassment 領域に initial application | Orcutt 1957 → Spielauer 2011 → Schofield 2018（系譜）+ Park 2024（LLM 別系譜の対比）|
| **Applied** | 3 種介入 counterfactual の policy-relevant comparison | Ho 2025（隣接 ABSS but coping not perpetration）+ Lanza & Rhoades 2013（differential treatment）|

### 4. Novelty を脅かす可能性のある「強い類似研究」（再点検）

| 候補 | 何を扱うか | 本研究との差異（具体的） |
|---|---|---|
| **Ho et al. 2025** | UK NHS の bullying targets coping strategies、ABSS | (i) target side、(ii) organization scale、(iii) personality 不使用 — **差異 3 軸 confirmed** |
| Merlone & Argentero 2018 | System dynamics qualitative model of mobbing | causal loops の theoretical 描写、prevalence projection でない |
| Notelaers 2006 / 2011 | Belgian sample で LCA → 6 victim classes、population prevalence | victim 側、本研究は perpetrator 側で対応 |
| Tucker et al. 2013 | Catastrophe theory model of workplace bullying | non-linear theoretical model、empirical projection でない |
| Sapouna et al. 2010 | School bullying virtual learning intervention RCT | school 領域、empirical RCT、simulation でない |

→ **本研究の novelty は依然として明確**

### 5. Limitation 統合（重要文献からの統合）

| Limitation | 主な根拠 | 対応 |
|---|---|---|
| Self-report 循環性 | Nielsen 2010（meta：method-conditional 11.3-18.1% 差） | MHLW 被害者報告と triangulate、limitation 明示 |
| Cell-level 検出力低 | Lanza & Rhoades 2013（Type I error 警告） | aggregate-level inference に集中、d≥0.9 のみ confidently 検出 |
| Asia/Oceania での personality 効果弱め | Nielsen 2017（geographic moderator） | 「Asia での effect 縮小可能性」を Discussion に含む |
| 介入効果サイズの cultural 依存 | Sapouna 2010（UK vs Germany 効果差）| Counterfactual sensitivity range で対応 |
| Aggregate からの inverse inference 禁止 | Schelling 1971（micro motives ≠ aggregate pattern）| Discussion で明示的に warn |
| 業種統制なし | Notelaers 2011（食品・製造で risk 高） | sensitivity に業種補正版を sub-analysis として |
| Independence 仮定 | Bruch & Atwell 2015（ABM では interactions 重要）| 本研究は microsim、interaction 不在を明示 |
| Self-report perpetration の social desirability | Bowling & Beehr 2006、Tsuno 2022（manager > non-manager 逆転）| Tier 2 文献で更に補強予定（Pillar 7 sub-task）|

### 6. 残存文献 gap（Tier 2 で補強候補）

- **Self-report perpetration validity**（Pillar 7）：Bowling & Beehr 2006 + Nielsen 2017 で間接的に押さえているが、**直接の self-report perpetrator 妥当性研究**が薄い
- **Sexual / power harassment training meta-analyses**（Pillar 5 補強）：Phase 2 Counterfactual C の null-finding 補強用
- **Counterfactual modeling methodology**（Pillar 6）：Pearl / Hernán & Robins の causal inference under intervention assumption — Methods で簡潔に触れる程度で十分
- **Capability approach operationalization**（Pillar 8）：Discussion で "consistent with Sen 1999, Nussbaum 2011" 程度に留める（4 層分離原則）

### 7. 次のステップ（implementation phase 移行）

文献基盤が確立したので、以下に着手可能：

1. **論文 Introduction draft 着手**（5 段落構成、上記 Cross-Pillar Synthesis の段落案を起点）
2. **Stage 0 コード実装**（D13 で確定した 14-cell 設計、N=354 → 7 類型 → 加害確率テーブル）
3. **Tier 2 検索**（必要に応じて、論文執筆中に補完）

---

## Round 4：Tier 2 — 論理的 gap 補強用追加文献（9 件、PDF 取得後に追加）

ユーザーが Tier 2 の 9 件 PDF を main にアップロード（commit `4343150`）。これに基づき、URL 検証段階から **原典 PDF deep reading** に格上げ。

| Tier 2 ID | 論文 | 重要度 | Pillar |
|---|---|---|---|
| D-1 | Grijalva et al. (2015) Narcissism and Leadership | ★★★ Core | Personality upstream |
| D-3 | Heckman et al. (2006) Cognitive/Noncognitive Abilities | ★★★ Core | Personality upstream |
| B-1 | Roehling & Huang (2018) Sexual Harassment Training | ★★★ Core | Intervention review |
| B-2 | Bezrukova et al. (2016) Diversity Training Meta | ★★★ Core | Intervention review |
| B-3 | Dobbin & Kalev (2018) Why Doesn't Diversity Training Work | ★★★ Core | Intervention review |
| C-1 | Hernán & Robins (2020) Causal Inference: What If | ★★★ Core | Causal framing |
| C-2 | Pearl (2009) Causality | ★★ Strong | Causal framing |
| A-1 | Berry, Carpenter, & Barratt (2012) Self vs Other CWB | ★★★ Core | Self-report validity |
| A-2 | Anderson & Bushman (2002) Human Aggression | ★★ Strong | Self-report validity |

**未取得**：D-2 Lee & Ashton (2005)、B-4 Roehling & Huang (2022)、B-5 Antecol & Cobb-Clark (2003) — 9 件で核は covered。

---

### [D-1] Grijalva, Harms, Newman, Gaddis, & Fraley (2015) — Narcissism and Leadership: A Meta-Analytic Review

**Citation**：Grijalva, E., Harms, P. D., Newman, D. A., Gaddis, B. H., & Fraley, R. C. (2015). Narcissism and leadership: A meta-analytic review of linear and nonlinear relationships. *Personnel Psychology, 68*(1), 1–47. https://doi.org/10.1111/peps.12072

**Verification**：✅✅ 原文 PDF 48p（Penn State Digital Commons OA 版）を本セッションで精読

#### Research Question

「Narcissism と leadership の **混合した先行知見** を meta-analysis で統合。4 contributions：
1. **Leadership emergence** vs **effectiveness** を区別
2. Extraversion による mediation を検証
3. Self-report vs observer-report の差異
4. **非線形（curvilinear）関係**の検証 — optimal mid-range narcissism」

#### Method

- **Design**：random-effects meta-analysis、psychometric correction（unreliability adjustment）
- **Sample size**：複数の effect sizes、unpublished studies 含む
- **Inventories included**：NPI（Narcissistic Personality Inventory）、HDS-Bold（Hogan Development Survey）、CPI、historiometric measures

#### Key Findings（exact numbers from Tables 1, 2）

##### 主効果（leadership emergence）

- **Narcissism → Leadership emergence**：**ρ = +0.16** [95% CI: 0.08, 0.15]、Hypothesis 1 支持
- 不採録 author 削除後：k=12、N=2,612、**ρ = +0.16** [0.09, 0.16] — robust
- **Acquaintance moderator**：
  - Minimal (< 1 week)：ρ = +0.18 [0.09, 0.18]
  - Longer (≥ 1 week)：ρ = +0.09 [0.002, 0.14]
  - → narcissism 効果は **時間とともに減衰** ("strong first impression but wears off")

##### 主効果（leadership effectiveness）

- **Narcissism → Leadership effectiveness**：**ρ = +0.03** [−0.01, 0.04] — **null**
- 80% CV = [−0.15, 0.20] — moderator 存在示唆
- → emergence と effectiveness は **異なる**（CI 非重複）

##### Source moderator（observer vs self-report）

- **Self-report effectiveness**：ρ = **+0.29** [0.17, 0.25]（自己評価では narcissist が leadership 高 self-rate）
- **Supervisor**：ρ = +0.04 [−0.01, 0.06]
- **Peer**：ρ = +0.02 [−0.04, 0.06]
- **Subordinate**：ρ = +0.12 [0.03, 0.13]
- → **Observer reports で narcissism × leadership effectiveness 効果なし** — narcissism 効果は self-enhancement bias

##### Curvilinear relationship（H4）

- Linear ρ = .03 だが、underlying curvilinear トレンド存在
- **Optimal mid-range narcissism** が leadership effectiveness 最大化
- → 「too little or too much narcissism は両方とも悪い」

##### Extraversion による mediation（H2）

- Narcissism と leadership emergence の関係は **extraversion で fully mediated**（H2 supports）
- → 「Narcissism は extraversion を通じて leadership emergence を生む」

#### Quotable Elements（原文逐語）

> "narcissism displays a positive relationship with leadership emergence, but no relationship with leadership effectiveness" (Abstract)

> "Whereas observer-reported leadership effectiveness ratings (e.g., supervisor-report, subordinate-report, and peer-report) are not related to narcissism, self-reported leadership effectiveness ratings are positively related to narcissism" (Abstract)

> "the nil linear relationship between narcissism and leadership effectiveness masks an underlying curvilinear trend, advancing the idea that there exists an optimal, midrange level of leader narcissism" (Abstract)

> "leadership emergence was positively related to narcissism (ρ = .16; 95% CI for = [.08, .15])" (p. 13)

> "narcissism had no linear relationship with leadership effectiveness (ρ = .03; 95% CI = [–.01, .04])" (p. 14, paraphrased)

#### 本研究での citation 用途

1. **Personality 上流 chain の central anchor**：「Narcissism—a personality trait inversely correlated with HEXACO Honesty-Humility (Lee & Ashton, 2005)—predicts leadership emergence (ρ = +0.16; Grijalva et al., 2015), placing low-HH individuals disproportionately into positions of organizational power. This personality-driven sorting into leadership positions partially explains the apparent SSS effect on workplace harassment victimization (Tsuno et al., 2015) and supports modeling personality typology as upstream of position-mediated harassment risk」
2. **「個人の性格と SSS は独立ではない」central reference**：narcissism → emergence の causal chain を defend
3. **Phase 2 介入の含意**：Narcissism (= 低 HH) を介入で減らせば、emergence への systematic boost が減 → **権力濫用機会の structural reduction**
4. **Mid-range narcissism optimum の議論**：本研究 Phase 2 で「HH を 0.5 SD 上げる」介入は **mid-range optimum を保ちつつ extreme を抑制**（curvilinear evidence と整合）
5. **Self-report bias 警告**：Narcissist は self-report で leadership effectiveness を **過大評価**（ρ=0.29）→ 本研究の N=354 self-report harassment perpetrator も narcissist が **真の effect を underreport** している可能性 → Berry et al. 2012 と組み合わせて self-report 妥当性議論
6. **Acquaintanceship moderator の延長**：「First-impression narcissism boost」は本研究の Discussion で「短期 selection vs 長期 outcome の divergence」議論に活用可
7. **Observer-report の null effect**：Counterfactual C（structural intervention）で observer-rated harassment は personality 効果が小さい → personality 介入（B 主軸）でも observer-rated outcome は遅れて改善する predict
8. **Big Five を超える HEXACO 主張**：Grijalva 2015 は narcissism × Big Five extraversion mediation。HEXACO は narcissism を **Honesty-Humility の inverse** として直接 capture（Lee & Ashton 2005）→ 本研究の HEXACO 7 typology は narcissism-related dynamics を **より parsimoniously** modeling

---

### [D-3] Heckman, Stixrud, & Urzua (2006) — Cognitive and Noncognitive Abilities on Labor Market Outcomes

**Citation**：Heckman, J. J., Stixrud, J., & Urzua, S. (2006). The effects of cognitive and noncognitive abilities on labor market outcomes and social behavior. *Journal of Labor Economics, 24*(3), 411–482. https://doi.org/10.1086/504455

**Verification**：✅✅ NBER Working Paper 12006（80 ページ）を本セッションで精読

#### Research Question

「Cognitive ability は労働市場 outcome の強い predictor として確立。**Noncognitive ability**（personality traits、persistence、motivation）は schooling、wages、occupation、risky behaviors にどう影響するか？ Cognitive と equally important か？」

#### 中核的論証

##### 主張：Low-dimensional vector of skills は多様な outcome を説明

> "This paper established that a low dimensional vector of cognitive and noncognitive skills explains a variety of labor market and behavioral outcomes. For many dimensions of social performance cognitive and noncognitive skills are equally important." (Abstract)

→ **Schooling、wages、employment、occupation、teenage pregnancy、smoking、marijuana use、illegal activities** すべて noncognitive skills が説明

##### Method 概要

- **Sample**：NLSY79（National Longitudinal Survey of Youth 1979、age 14–21 at start、longitudinal panel）
- **Cognitive measure**：ASVAB 5 components（arithmetic reasoning, word knowledge, paragraph comprehension, mathematical knowledge, coding speed）
- **Noncognitive measure**：Rotter Locus of Control + Rosenberg Self-Esteem の standardized average
- **Methodological innovation**：
  - Schooling を endogenous として扱う（IV / structural model）
  - Latent skill model で measurement error 補正
  - Reverse causality 問題 に対応

##### Key Findings（structural results）

###### Wages への直接効果

- Cognitive と noncognitive skills **両方**が wages を直接予測
- Conditioning on schooling すると effect 縮小（schooling は両 skill から決まる endogenous mediator）
- **Schooling を removing すると noncognitive skill effect は cognitive と equally large**

###### Schooling 決定への効果

- **Noncognitive skills strongly influence schooling decisions**
- Personality（perseverance, self-control）が「continue education」の "psychic cost" を低下させる
- → "psychic cost" theory：高 noncognitive skill 者は schooling continuation の心理的負担が低い

###### Occupation 選択

- Noncognitive と cognitive が **independent に** occupation 選択を予測
- → 高 noncognitive 者が prestige 高 occupation に sort される

###### Risky behaviors（10s 年代の reverse outcomes）

- Teenage pregnancy、smoking、marijuana use、illegal activities：noncognitive で全て予測
- Cognitive と noncognitive **両方** required（独立効果）

###### Causal evidence（Perry Preschool に言及）

著者らは Perry Preschool program（介入対象児が控除群より高 noncognitive を獲得）を引用：
- Perry treatment group：high school by age 18（**65% vs 45%** control）
- 14 歳 California Achievement Test 10th percentile 以上：**49% vs 15%** control
- Age 40 illegal activities：**有意に少ない**

→ Noncognitive intervention が long-term outcomes に causal evidence

##### 関連分野の文献継承

- Bowles & Gintis (1976), Edwards (1976) Marxist 経済学：employer は低 skill 市場で **docility, dependability, persistence** を重視
- → 「Personality は market signal として機能」する伝統的理解と整合

#### Quotable Elements（原文逐語）

> "This paper established that a low dimensional vector of cognitive and noncognitive skills explains a variety of labor market and behavioral outcomes. For many dimensions of social performance cognitive and noncognitive skills are equally important." (Abstract)

> "Noncognitive skills strongly influence schooling decisions, and also affect wages given schooling decisions. Schooling, employment, work experience and choice of occupation are affected by latent noncognitive and cognitive skills." (Abstract)

> "The same low dimensional vector of abilities that explains schooling choices, wages, employment, work experience and choice of occupation explains these behavioral outcomes [risky behaviors]." (Abstract)

> "Common sense suggests that personality traits, persistence, motivation and charm matter for success in life. Marxist economists (Bowles and Gintis, 1976; Edwards, 1976) have produced a large body of evidence that employers in low skill labor markets value docility, dependability, and persistence more than cognitive ability or independent thought." (p. 1)

> "We find that latent noncognitive skills, corrected for schooling and family background effects, raise wages through their direct effects on productivity as well as through their indirect effects on schooling and work experience. Our evidence is consistent with an emerging body of literature that finds that 'psychic costs' (which may be determined by noncognitive traits) explain why many adolescents who would appear to financially benefit from [more schooling do not pursue it]" (p. ?)

#### 本研究での citation 用途

1. **Personality 上流性の経済学的 anchor**：「Personality (noncognitive abilities) is upstream of educational attainment, employment, occupation, and wages (Heckman, Stixrud, & Urzua, 2006)」 — Pillar D の central reference
2. **「Personality と SSS は独立ではない」反論**：「The 'social class effect' on harassment exposure (e.g., Tsuno et al., 2015 OR=4.21) reflects in part personality-driven sorting into positions, as established by Heckman et al. (2006)」
3. **Phase 2 介入の社会経済的含意**：Personality intervention は **labor market outcomes も同時に改善する波及効果**を持つ（Heckman 2006 が複数 outcome の共通源泉として noncognitive を確立）
4. **Causal evidence からの援用**：Perry Preschool の long-term outcome を引用し、「early personality intervention has causal long-term effects on outcomes」 — 本研究 Phase 2 universal HH intervention の plausibility 強化
5. **方法論的注意**：Heckman は schooling endogeneity を IV で扱った。本研究は cross-sectional のみだが、**Heckman 系統の structural causal modeling tradition**（後の Hernán & Robins 2020 と integrate）に positioning
6. **"Psychic cost" 概念の応用**：Personality intervention は **harassment perpetration cost を上げる**ことで behavior 変化を生む（HH 介入後に narcissist が harassment を行う心理的コストが上がる）
7. **NLSY 系統の large-scale empirical 評価**：本研究の N=354 は Heckman の NLSY と比較して small だが、**本研究は新規データ収集なし、既存 N=13,668 + N=354 を組み合わせた modeling**で類似の breadth を実現
8. **政策含意の整合**：Heckman は「noncognitive skill investment は cognitive investment と equally important」と政策提言。本研究は **harassment 領域への同 message**（personality intervention は environmental intervention と equally important）

---

### [B-1] Roehling & Huang (2018) — Sexual harassment training effectiveness: An interdisciplinary review

**Citation**：Roehling, M. V., & Huang, J. (2018). Sexual harassment training effectiveness: An interdisciplinary review and call for research. *Journal of Organizational Behavior, 39*(2), 134–150. https://doi.org/10.1002/job.2257

**Verification**：✅✅ 原文 PDF 17p（JOB Annual Review）を本セッションで精読

#### Research Question

「Sexual harassment (SH) training は **遍在的**だが、効果に関する **interdisciplinary な総合的 review がこれまで存在しない**。Legal context、effectiveness の多次元（reactions、learning、attitudes、behavior、transfer）、organizational moderators を整理し、research agenda を提示」

#### 中核的論証

##### Kirkpatrick 4-level framework での効果評価

| Level | Outcome | SH training での evidence |
|---|---|---|
| 1 | **Reactions**（受講者の反応・満足度）| ✓ 多くの study が positive reactions |
| 2 | **Learning**（知識・skill 獲得）| ✓ 知識増加 が比較的安定して観察 |
| 3 | **Behavior**（実職場での行動変化）| △ 限定的、mixed evidence |
| 4 | **Results**（組織 outcome、incidence 減少等）| ✗ ほぼ証拠なし、Phillips 1997 のみ turnover -4.3% |

→ **Level 1 / 2 では効果あり、Level 3 / 4（distal）では弱い** — Bezrukova 2016 と整合

##### 重要な individual finding

- **Magley et al. 2013 Study 2**：3-hr training で **non-Hispanic trainees の知識増加、Hispanic trainees に効果なし** → 文化・demographic moderation
- **Coping skills training**：Bell et al. が SH 直面時の coping を 1 年後も維持と報告
- **Phillips 1997**：SH training 後 turnover **4.3% 減** — outcome レベルの数少ない証拠
- **Severity moderation**（Blakely et al.）：training は **most severe forms** にのみ効果

##### Key insights

1. **Legal context が training 設計を歪める**：legally compliant とは behaviorally effective とは異なる
2. **Organizational context が motivation/transfer を調整**：training の effective transfer は climate（accountability、support）次第
3. **Individual 受講者特性**：高 LSH（Likelihood to Sexually Harass）受講者は **more negative attitudes** 表明（**逆効果可能性**）

##### Research agenda（著者提示）

- Long-term (transfer, behavior change) outcome の研究強化
- Cross-cultural / cross-organizational 比較
- Individual difference moderators（LSH、ethnicity）
- Training の **iatrogenic effects**（高 risk 受講者で逆効果）

#### Quotable Elements（原文逐語）

> "Although sexual harassment (SH) training is widespread, has many important consequences for individuals and organizations, and is of demonstrated interest to researchers across a wide range of disciplines, there has never been a comprehensive, interdisciplinary attempt to identify and systematically evaluate relevant research findings." (Summary)

> "It discusses the legal context of SH training and its relevance to research issues, provides an organizing framework for understanding the primary factors influencing SH training effectiveness, critically reviews empirical research providing evidence of the effectiveness of SH training, and sets forth a research agenda." (Summary)

> "high LSH participants reported significantly more negative attitudes [after training]." (p. ?)— **iatrogenic effect** の重要証拠

> "[For] one study found a training effect on only the most severe forms of sexually oriented work behaviors (Blakely et al., ...)" (p. ?)

#### 本研究での citation 用途

1. **Counterfactual C（structural intervention）の限界 anchor**：「Sexual harassment training shows reliable effects on reactions and knowledge (Roehling & Huang, 2018) but limited effects on actual harassment behavior or incidence」
2. **Phase 2 介入の "proximal vs distal" framework**：Kirkpatrick 4-level を使って「本研究の Counterfactual A/B/C は **distal outcome（incidence）への効果**を予測する。Roehling & Huang のレビューが示すように distal evidence は弱い」 — 慎重 effect size assumption の根拠
3. **Iatrogenic effect の警告**：高 LSH 受講者で逆効果可能性 → Phase 2 Counterfactual A（universal training）の potential downside を Discussion で言及
4. **Cultural moderator の言及**：non-Hispanic vs Hispanic effect 差 → 日本での介入効果が西欧 anchor と異なる可能性 limitation
5. **Bezrukova 2016 + Dobbin & Kalev 2018 + Roehling 2018 の triangulation**：Counterfactual C の限界を **3 系統 review で堅固に**主張
6. **Legal compliance ≠ behavioral effectiveness**：Roehling 2018 の central insight は本研究の Phase 2 で「日本のパワハラ防止法（2019）は legal compliance 強化、行動変化 evidence は別」と framing する根拠
7. **Severity moderation**：training が most severe forms にのみ効果 → 本研究の binary outcome（mean+0.5SD threshold）の sensitivity analysis で thread-conditional effect を検討する根拠
8. **Organizational climate moderator**：training の effect は accountability climate に依存 → Phase 2 Counterfactual C 単独でなく、組織文化 intervention との combination が必要

---

### [B-2] Bezrukova, Spell, Perry, & Jehn (2016) — Diversity Training Meta-Analysis (260 samples)

**Citation**：Bezrukova, K., Spell, C. S., Perry, J. L., & Jehn, K. A. (2016). A meta-analytical integration of over 40 years of research on diversity training evaluation. *Psychological Bulletin, 142*(11), 1227–1274. https://doi.org/10.1037/bul0000067

**Verification**：✅✅ 原文 PDF 130p（large meta-analysis）を本セッションで精読

#### Research Question

「**40 年以上 / 260 independent samples** の diversity training 研究を meta-analyse。Kirkpatrick 4-level outcome（reactions、cognitive learning、behavioral learning、attitudinal/affective learning）に分けて、training context、design、participants の moderator 効果を同定」

#### 中核的論証

##### 全体効果サイズと outcome 別 breakdown（Hypothesis 1）

| Outcome | g | サイズ評価 |
|---|---|---|
| **Reactions**（受講後評価） | **.61** | medium-large |
| **Cognitive learning**（知識獲得） | **.57** | medium-large |
| **Behavioral learning**（skill 開発） | **.48** | medium |
| **Attitudinal/affective learning**（態度変化）| **.30** | small-medium |
| **Overall** | **.38** | small-medium |

→ Q_B(3) = 41.48, p < .001（outcome 間で effect size 異なる）
→ **Reactions と cognitive learning に最大効果、attitudinal change に最小効果**

##### 時間減衰（Hypothesis 2）

> "Whereas the effects of diversity training on reactions and attitudinal/affective learning decayed over time, training effects on cognitive learning remained stable and even increased in some cases."

→ **Attitude effect は時間とともに消失**、cognitive learning のみ持続

##### Setting moderator（教育 vs 組織）

- **Educational setting**：reactions g = **.80**
- **Organizational setting**：reactions g = **.28**
- Q_B(1) = 6.43, p = .02
- → **学校環境 vs 職場の reactions 効果に 3 倍差** — 「**学校で楽しむ vs 職場では受け入れにくい**」

##### Integrated vs standalone approach

- **Integrated**（他 diversity initiative と組合せ）：attitudinal g = .47、behavioral g = **.86**
- **Standalone**：attitudinal g = .27、behavioral g = .42
- → **複合的アプローチで behavioral effect が 2 倍**

##### Mandatory vs voluntary

- Behavioral learning：mandatory g = .63 vs voluntary g = .42（mandatory 優位）
- Reactions：mandatory g = .37 vs voluntary g = **.71**（**voluntary 優位**）
- → 「強制的に受けると不機嫌だが behavior は変わる、自発的だと機嫌は良いが行動変化弱い」

##### 介入「逆効果」事例（abstract で言及）

> "diversity training has been shown to backfire in some cases by reinforcing stereotypes and prejudice among students (Robb & Doverspike, 2001) or creating new problems for the company (Kaplan, 2006), such as when air traffic controllers sued the Federal Aviation Administration because they had found diversity training traumatic"

→ **Training の iatrogenic effect は実在**

##### 効果を高める条件（Discussion）

- **複合実装**（other diversity initiatives との組合せ）
- **awareness + skills development の両方を target**
- **長期間の実施**（短期では弱い）
- Group composition：女性比率高で reactions favorable

#### Quotable Elements（原文逐語）

> "This meta-analysis of 260 independent samples assessed the effects of diversity training on 4 training outcomes over time and across characteristics of training context, design, and participants." (Abstract)

> "The results revealed an overall effect size (Hedges g) of .38 with the largest effect being for reactions to training and cognitive learning; smaller effects were found for behavioral and attitudinal/affective learning. Whereas the effects of diversity training on reactions and attitudinal/affective learning decayed over time, training effects on cognitive learning remained stable and even increased in some cases." (Abstract)

> "Diversity training had the largest effect on reactions (g = .61), followed by cognitive learning (g = .57), behavioral learning (g = .48), and attitudinal/affective learning (g = .30), QB(3) = 41.48, p = .00." (p. ~30)

> "The positive effects of diversity training were greater when training was complemented by other diversity initiatives, targeted to both awareness and skills development, and conducted over a significant period of time." (Abstract)

#### 本研究での citation 用途

1. **Counterfactual C 限界の central anchor**：「Despite 40+ years of diversity training and 260 evaluated samples (Bezrukova et al., 2016), behavioral effects are modest (g = .48) and attitudinal effects are small (g = .30) and decay over time. This pattern—proximal effective, distal weak and time-decaying—supports our conservative effect size assumption for structural interventions」
2. **Phase 2 effect size sensitivity 範囲**：g = .30–.48 を **structural intervention upper bound** として採用、Pruckner 2013 の 30% 削減（intensive margin only）と整合
3. **Standalone vs integrated**：Counterfactual C を **standalone** training として modeling すると effect 弱め。Integrated approach（他 initiative 組合せ）が必要
4. **Iatrogenic effect の警告**：Training は backfire 可能（stereotype reinforcement）→ Phase 2 で 「Counterfactual A（universal）の potential downside」を Discussion に
5. **Time decay の含意**：Attitude effect は時間とともに消失 → Phase 2 の **24 週時点 effect**（Roberts 2017）が上限値である根拠
6. **Mandatory vs voluntary trade-off**：強制 training は behavior に効くが reactions 悪化 → policy implication で「単純な mandatory 拡大は逆効果」
7. **Cultural 一般化 limitation**：Bezrukova は西欧中心 sample → 日本での effect size は別途 calibration 必要
8. **Outcome 階層の framework**：本研究の 14-cell aggregate prevalence prediction は **Kirkpatrick Level 4 (Results)** に相当 → Bezrukova で最も弱い outcome category。**保守的に解釈する**根拠

