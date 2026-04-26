# Reference Index — Simulation Paper

**目的**: シミュレーション論文（Big Five → generative agent → 大学入試結果予測）で引用候補となる文献を一元管理。執筆時は本ファイルを参照して citation の正確性を担保する。

**検証状況**:
- ✅ = 原文ノート、PDF、または arXiv/journal page で著者・年・タイトル・venue・DOI を直接確認済
- ✅✅ = 一次 PDF 全文を本セッションで完読、本文逐語引用が可能
- 📖 = 標準知識（PDF 非所持、執筆前に再確認推奨）
- 🔴 = AI 生成由来の誤引用が判明（要修正）
- ⚠️ = サブエージェント要約のみ、未検証（引用前に一次確認必須）

**最終更新**: 2026-04-26

---

## ⚡ 重要な方法論的フレーミング（2026-04 追加）

シミュレーション論文の主要評価指標は **2 つの異なるレベル** を持ち、近年の論文間で明示的に区別されるようになった。Toubia et al. (2026) が Park et al. (2024) を引用しつつ次のように指摘している:

> "Note that [Park et al. 2024] report a much higher correlation in their digital twin study. However, **they compute the correlation across questions for each participant**. In contrast, **we compute the correlation across participants for each outcome**. This is more often the measure of interest, as social scientists often ask questions such as who is more likely to vote for a candidate or who is more likely to purchase a specific product."
> (Toubia et al., 2026, p. 11)

| 指標 | 別名 | Park 2024 数値 | Toubia 2026 数値 |
|---|---|---|---|
| Within-person × across-questions | Park's normalized accuracy | 0.83 (interview), 0.82 (surveys), 0.86 (combined) | — |
| Between-person × per-outcome | Toubia's correlation | — | 0.197 (full persona), 0.232 (best) |

**含意**: 本研究で「Big Five → 入試偏差値の個人レベル予測」を問う場合、**Toubia の between-person 指標**が直接対応する。Park の 0.83 を直接ベンチマークとして使うのは指標誤用にあたる。

---

# Part I: CS 222 Reading List（25 本、原文 PDF + 詳細ノート確認済）

## I-A. 中核 5 本（本研究の手法アンカー）

### CS-05_1. Park et al. (2023) — Generative Agents ★ ✅
Park, J. S., O'Brien, J. C., Cai, C. J., Morris, M. R., Liang, P., & Bernstein, M. S. (2023). Generative agents: Interactive simulacra of human behavior. In *Proceedings of the 36th Annual ACM Symposium on User Interface Software and Technology* (Article 2, pp. 1–22). Association for Computing Machinery. https://doi.org/10.1145/3586183.3606763

ノート: [paper_05_1.md](docs/notes/cs222/papers/paper_05_1.md) | arXiv: 2304.03442

### CS-03_1. Park et al. (2022) — Social Simulacra ✅
Park, J. S., Popowski, L., Cai, C. J., Morris, M. R., Liang, P., & Bernstein, M. S. (2022). Social simulacra: Creating populated prototypes for social computing systems. In *Proceedings of the 35th Annual ACM Symposium on User Interface Software and Technology*. ACM.

ノート: [paper_03_1.md](docs/notes/cs222/papers/paper_03_1.md)

### CS-03_2. Argyle et al. (2023) — Out of One, Many ✅
Argyle, L. P., Busby, E. C., Fulda, N., Gubler, J. R., Rytting, C., & Wingate, D. (2023). Out of one, many: Using language models to simulate human samples. *Political Analysis, 31*(3), 337–351. https://doi.org/10.1017/pan.2023.2

ノート: [paper_03_2.md](docs/notes/cs222/papers/paper_03_2.md)

### CS-07_2. Hewitt et al. (2024) — Predicting Social Science ✅
Hewitt, L., Ashokkumar, A., Ghezae, I., & Willer, R. (2024). *Predicting results of social science experiments using large language models* [Preprint, August 8, 2024]. Stanford University.

ノート: [paper_07_2.md](docs/notes/cs222/papers/paper_07_2.md) | Web demo: https://treatmenteffect.app

### CS-05_2. Sumers et al. (2024) — CoALA ✅
Sumers, T. R., Yao, S., Narasimhan, K., & Griffiths, T. L. (2024). Cognitive architectures for language agents. *Transactions on Machine Learning Research*. https://openreview.net/forum?id=1i6ZCvflQJ

ノート: [paper_05_2.md](docs/notes/cs222/papers/paper_05_2.md) | arXiv: 2309.02427

## I-B. 認知アーキテクチャ・古典 ABM・believability

### CS-04_1. Newell (1992) — Unified Theories of Cognition ✅
Newell, A. (1992). Précis of *Unified theories of cognition*. *Behavioral and Brain Sciences, 15*(3), 425–492.

### CS-04_2. Lehman, Laird, & Rosenbloom (2006) — SOAR ✅
Lehman, J. F., Laird, J. E., & Rosenbloom, P. S. (2006). A gentle introduction to Soar, an architecture for human cognition: 2006 update.

### CS-07_1. Bates (1994) — Believable Agents ✅
Bates, J. (1994). The role of emotion in believable agents. *Communications of the ACM, 37*(7), 122–125. https://doi.org/10.1145/176789.176803

### CS-09_1. Bruch & Atwell (2015) — ABMs in Empirical Social Research ✅
Bruch, E., & Atwell, J. (2015). Agent-based models in empirical social research. *Sociological Methods & Research, 44*(2), 186–221. https://doi.org/10.1177/0049124113506405

## I-C. シミュレーション応用・社会ネットワーク

### CS-06_1. Chang et al. (2025) — LLM Social Networks ✅
Chang, S., Chaszczewicz, A., Wang, E., Josifovska, M., Pierson, E., & Leskovec, J. (2025). LLMs generate structurally realistic social networks but overestimate political homophily [AAAI 2025]. arXiv:2408.16629.

### CS-06_2. Louie et al. (2024) — Roleplay-doh ✅
Louie, R., Nandi, A., Fang, W., Chang, C., Brunskill, E., & Yang, D. (2024). *Roleplay-doh: Enabling domain-experts to create LLM-simulated patients via eliciting and adhering to principles* [EMNLP 2024]. arXiv:2407.00870.

### CS-16_1. Chang et al. (2021) — COVID Mobility Networks ✅
Chang, S., Pierson, E., Koh, P. W., Gerardin, J., Redbird, B., Grusky, D., & Leskovec, J. (2021). Mobility network models of COVID-19 explain inequities and inform reopening. *Nature, 589*(7840), 82–87. https://doi.org/10.1038/s41586-020-2923-3

### CS-16_2. Chang et al. (2025) — Supply Chain GNN ✅
Chang, S., Lin, Z., Yan, B., Bembde, S., Xiu, Q., Wong, C. H., Qin, Y., Kloster, F., Luo, X., Palleti, R., & Leskovec, J. (2025). *Learning production functions for supply chains with graph neural networks* [AAAI 2025]. arXiv:2407.18772.

## I-D. Wicked problems・古典 ABM・哲学的基盤

### CS-02_1. Rittel & Webber (1973) — Wicked Problems ✅
Rittel, H. W. J., & Webber, M. M. (1973). Dilemmas in a general theory of planning. *Policy Sciences, 4*(2), 155–169.

### CS-02_2. Schelling (1978) — Micromotives ✅
Schelling, T. C. (1978). *Micromotives and macrobehavior*. W. W. Norton.

### CS-08_1. Ansolabehere, Rodden, & Snyder (2008) — Strength of Issues ✅
Ansolabehere, S., Rodden, J., & Snyder, J. M., Jr. (2008). The strength of issues: Using multiple measures to gauge preference stability, ideological constraint, and issue voting. *American Political Science Review, 102*(2), 215–232. https://doi.org/10.1017/S0003055408080210

### CS-08_2. Resnick et al. (1994) — GroupLens ✅
Resnick, P., Iacovou, N., Suchak, M., Bergstrom, P., & Riedl, J. (1994). GroupLens: An open architecture for collaborative filtering of netnews. In *Proceedings of the 1994 ACM Conference on Computer Supported Cooperative Work* (pp. 175–186). ACM. https://doi.org/10.1145/192844.192905

## I-E. メタアナロジー・科学史的参照

### CS-10_1. IHGSC (2001) — Human Genome ✅
International Human Genome Sequencing Consortium. (2001). Initial sequencing and analysis of the human genome. *Nature, 409*(6822), 860–921.

### CS-11_1. Lorenz (1993) — Essence of Chaos ✅
Lorenz, E. N. (1993). *The essence of chaos*. University of Washington Press.

### CS-11_2. Holt & Roth (2004) — Nash Equilibrium ✅
Holt, C. A., & Roth, A. E. (2004). The Nash equilibrium: A perspective. *Proceedings of the National Academy of Sciences, 101*(12), 3999–4002.

### CS-12_1. Bostock, Ogievetsky, & Heer (2011) — D3 ✅
Bostock, M., Ogievetsky, V., & Heer, J. (2011). D³: Data-driven documents. *IEEE Transactions on Visualization and Computer Graphics, 17*(12), 2301–2309.

## I-F. Ethics・LLM-as-participant の限界

### CS-13_1. Wang, Morgenstern, & Dickerson (2024) — Flatten Identity ✅
Wang, A., Morgenstern, J., & Dickerson, J. P. (2024). *Large language models that replace human participants can harmfully misportray and flatten identity groups* [Preprint]. arXiv:2402.01908. https://doi.org/10.48550/arXiv.2402.01908

### CS-13_2. Santurkar et al. (2023) — Whose Opinions ✅
Santurkar, S., Durmus, E., Ladhak, F., Lee, C., Liang, P., & Hashimoto, T. (2023). Whose opinions do language models reflect? In *Proceedings of the 40th International Conference on Machine Learning*. arXiv:2303.17548.

### CS-15_1. Morris & Brubaker (2024) — Generative Ghosts ✅
Morris, M. R., & Brubaker, J. R. (2024). *Generative ghosts: Anticipating benefits and risks of AI afterlives*. arXiv:2402.01662.

### CS-15_2. Manzini et al. (2024) — Code That Binds Us ✅
Manzini, A., Keeling, G., Alberts, L., Vallor, S., Morris, M. R., & Gabriel, I. (2024). The code that binds us: Navigating the appropriateness of human-AI assistant relationships. In *Proceedings of the Seventh AAAI/ACM Conference on AI, Ethics, and Society (AIES 2024)*.

---

# Part II: 個人レベル予測の理論的・経験的アンカー（検証済）

## II-A. Park 2024 — 1,000 People → Self-Reports Grounded ✅✅

**最新版（2026-04-22 改訂、本セッションで一次確認）— 改題＋著者追加**:
Park, J. S., Zou, C. Q., Kamphorst, J., Egan, N., Shaw, A., Hill, B. M., Cai, C., Morris, M. R., Liang, P., Willer, R., & Bernstein, M. S. (2024–2026). *LLM agents grounded in self-reports enable general-purpose simulation of individuals* [Preprint]. arXiv:2411.10109. https://doi.org/10.48550/arXiv.2411.10109

**v1（2024-11、初出題名）— アーカイブ版**:
Park, J. S., Zou, C. Q., Shaw, A., Hill, B. M., Cai, C., Morris, M. R., Willer, R., Liang, P., & Bernstein, M. S. (2024). *Generative agent simulations of 1,000 people* (v1) [Preprint]. arXiv:2411.10109.

**主要数値（v2 abstract / Fig. 2 より）**:
- N = 1,052 米国成人（age, gender, race, region, education, party identification で stratified）
- 評価: General Social Survey (GSS) 150 項目, BFI-44, 行動経済ゲーム 5 種, 行動実験 5 種
- 正規化分母 = 個人の 2 週間後の test-retest 一致率
- **Normalized accuracy（within-person × across-questions 指標）**:
  - Interview only (2h semi-structured): **0.83**
  - Surveys only (GSS + BFI-44): **0.82**
  - Survey + Interview combined: **0.86**
  - Demographics only (age/gender/race/ideology): **0.74**
  - Persona-paragraph baseline: **0.71**
- Big Five 正規化相関: interview 0.80, surveys-only 0.65, combined 0.77, demographics 0.61
- Ablation: インタビュー 80% 削除しても GSS 0.79 を維持 → 言語的特徴より情報内容が支配的

**指標の性質**: across-questions correlation per participant（個人内一貫性）。Toubia et al. (2026) が指摘するように between-person 個人差予測の指標とは異なる。

ノート: [paper_07_2.md ではなく 1000 People 専用ノート未作成、PDF 内蔵テキストは `simulation/prior_research/_text/LLM Agents Grounded in Self-Reports....txt`]

## II-B. Park (2024) Lecture Notes ✅
Park, J. S. (2024). *CS 222: AI agents and simulations* [Lecture notes]. Stanford University. https://joonspk-research.github.io/cs222-fall24/

## II-C. Lundberg et al. (2024) — Origins of Unpredictability ✅✅
Lundberg, I., Brown-Weinstock, R., Clampet-Lundquist, S., Pachman, S., Nelson, T. J., Yang, V., Edin, K., & Salganik, M. J. (2024). The origins of unpredictability in life outcome prediction tasks. *Proceedings of the National Academy of Sciences, 121*(24), e2322973121. https://doi.org/10.1073/pnas.2322973121

本セッションで一次完読。重要内容:
- 予測誤差の **2 成分分解**: irreducible error（観測上同一の個人内の outcome 分散）+ learning error（推定された予測関数と真の条件付き平均のズレ）
- Irreducible error の 3 源: ① consequential intervening events（測定窓と outcome の間の事象）② unmeasured features ③ imperfectly measured features（粗 Likert 等）
- **§4.1 短期 horizon は例外**: "natural low-dimensional representations and short time horizons are the exception rather than the norm"
- Fragile Families Challenge の最良 R²ₕₒₗ𝒹ₒᵤₜ = **0.19**（GPA 予測）

注: 旧 introduction.md 第 11 段落に "Lundberg, Brand, & Jeon (2024)" のハルシネーションがあったが、introduction.md は 2026-04-26 に削除済み。新規執筆時は本エントリの正しい著者リストを使用。

## II-D. Gordon et al. (2022) — Jury Learning ✅
Gordon, M. L., Lam, M. S., Park, J. S., Patel, K., Hancock, J. T., Hashimoto, T., & Bernstein, M. S. (2022). Jury learning: Integrating dissenting voices into machine learning models. In *Proceedings of the 2022 CHI Conference on Human Factors in Computing Systems* (Article 115, pp. 1–19). ACM. https://doi.org/10.1145/3491102.3502004

arXiv: 2202.02950

## II-E. Schelling (1971) — Dynamic Models of Segregation ✅
Schelling, T. C. (1971). Dynamic models of segregation. *Journal of Mathematical Sociology, 1*(2), 143–186. https://doi.org/10.1080/0022250X.1971.9989794

## II-F. John & Srivastava (1999) — Big Five 📖
John, O. P., & Srivastava, S. (1999). The Big Five trait taxonomy: History, measurement, and theoretical perspectives. In L. A. Pervin & O. P. John (Eds.), *Handbook of personality: Theory and research* (2nd ed., pp. 102–138). Guilford Press.

書籍章のため DOI なし。著者・タイトル・出版社は標準知識として確認済。

## II-G. Peng, Toubia et al. (2026) — Digital Twins are Funhouse Mirrors ✅✅ ★
Peng, T., Gui, G., Brucks, M., Merlau, D. J., Fan, G. J., Ben Sliman, M., Johnson, E. J., Althenayyan, A., Bellezza, S., Donati, D., Fong, H., Friedman, E., Guevara, A., Hussein, M., Jerath, K., Kogut, B., Kumar, A., Lane, K., Li, H., … Toubia, O. (2026). *Digital twins are funhouse mirrors: Five systematic distortions* [Preprint, arXiv:2509.19088 v5, 2026-04-19]. https://doi.org/10.48550/arXiv.2509.19088

別題: *A Mega-Study of Digital Twins Reveals Strengths, Weaknesses and Opportunities for Further Improvement*

SSRN: https://doi.org/10.2139/ssrn.5518418
データ: https://huggingface.co/datasets/LLM-Digital-Twin/Twin-2K-500-Mega-Study
コード: https://github.com/TianyiPeng/Twin-2K-500-Mega-Study

本セッションで一次完読。本研究にとって最重要の方法論アンカー。

**設計**:
- 22 著者（Columbia 中心、Yale, Yeshiva）の協働 mega-study
- **19 pre-registered substudies**, **164 outcomes**, 13,506 participants（うち 1,784 unique）
- Twin-2K-500 dataset（既存）に対して **新規 19 サブスタディを追加実行**
- Twin-2K-500 ベース データ: 14 demographic + 279 personality (19 tests, 26 constructs) + 85 cognitive + 34 economic + 48 heuristics + 40 pricing = **500+ 質問**, 4 wave longitudinal
- 検証モデル: GPT4.1 (default 0.7), GPT-5, Deepseek, Gemini, Gemini-3 Pro, fine-tuned GPT4.1, Centaur

**主要数値**（本文 pp. 8–11 より直接抽出）:

| 条件 | Individual accuracy = 1−(MAD/range) | Correlation (between-person, per-outcome) |
|---|---|---|
| Random baseline | 0.629 | 0.001 |
| Empty persona (LLM のみ) | 0.734 | 0.080 |
| Demographics only (14 vars) | 0.746 | 0.145 |
| **Full persona (500+ items)** | **0.748** | **0.197** |
| Best (GPT4.1 temp=0) | 0.752 | 0.232 |

**Park 2024 との指標差の明示**（p. 11 逐語）:
> "Note that (17) report a much higher correlation in their digital twin study. However, they compute the correlation across questions for each participant. In contrast, we compute the correlation across participants for each outcome."

**Big Five 5 数値で十分との重要発見**（p. 11 逐語）:
> "We also experimented with replacing the full persona information with a concise (approximately 13K characters), statement-based summary of the questions and responses with distributional information (e.g., Big 5 personality scores with percentile ranks rather than 44 detailed questions and answers on which the scores are based). We find that this simpler version performs very similarly to the full persona, offering a viable lower-cost alternative."

**XGBoost 比較（pp. 23–24）**:
- Full-persona digital twin の correlation ≈ XGBoost (full persona, N≈180 で訓練)
- Full-persona digital twin の accuracy ≈ XGBoost (full persona, N≈75 で訓練)
- → 私たちの N=103 で従来型 ML が agent と同等以上の可能性大

**5 Distortions（各 diagnostic benchmark あり）**:

| # | 歪み | 診断方法 | Toubia の数値 |
|---|---|---|---|
| 1 | Insufficient individuation | twin SD vs human SD | twin SD < human SD in **154/164 (93.9%)** outcomes |
| 2 | Stereotyping | full persona vs demographics-only twin の MAD 比較 | MAD_full-vs-demo = 0.132 < MAD_full-vs-empty = 0.175 < MAD_full-vs-humans = 0.252 |
| 3 | Representation bias | demographic group 別 accuracy の Partial Dependence Plot | 高教育・高所得・moderate political views で精度高 |
| 4 | Ideological biases | 系統的方向性バイアスの検査 | pro-human + pro-technology + 「algorithmic hiring に寛容」「target online ads を低侵入と評価」 |
| 5 | Hyper-rationality | 客観正解問題で twin の正答率が人間より高いか | 客観正解問題で twin が perfect knowledge を示す傾向 |

**ドメイン依存性**:
- **強い**: cognitive, human-tech interaction, response scales, conflict, pro-social, social cognition, personality
- **弱い**: socially desirable contexts, political domain, valenced evaluations, prompt-level variation across participants

→ 本研究の outcome（入試偏差値）は cognitive + personality 寄りで**有利な領域**。ただし「学業達成」は社会的望ましさと連動するので潜在的に **distortion #4** リスクあり。

---

# Part III: Gap-Filling Literature（未調査領域、完全検証済）

CS 222 reading list は HCI / 政治学寄りで、以下 4 領域の補完が必要。各文献は WebFetch / WebSearch で原文を直接確認済。

## III-A. LLM-as-economic-agent

### Horton, Filippas, & Manning (2023) ✅
Horton, J. J., Filippas, A., & Manning, B. S. (2023). *Large language models as simulated economic agents: What can we learn from Homo Silicus?* [Preprint]. arXiv:2301.07543. https://doi.org/10.48550/arXiv.2301.07543

NBER Working Paper No. 31122 としても流通。

### Aher, Arriaga, & Kalai (2023) ✅
Aher, G. V., Arriaga, R. I., & Kalai, A. T. (2023). Using large language models to simulate multiple humans and replicate human subject studies. In *Proceedings of the 40th International Conference on Machine Learning* (PMLR, Vol. 202). https://proceedings.mlr.press/v202/aher23a.html

arXiv: 2208.10264 | ICML 2023 Oral

### Manning, Zhu, & Horton (2024) ✅
Manning, B. S., Zhu, K., & Horton, J. J. (2024). *Automated social science: Language models as scientist and subjects* [Preprint]. NBER Working Paper No. 32381. https://www.nber.org/papers/w32381

arXiv: 2404.11794

### Tjuatja, Chen, Wu, Talwalkar, & Neubig (2024) ✅
Tjuatja, L., Chen, V., Wu, T., Talwalkar, A., & Neubig, G. (2024). Do LLMs exhibit human-like response biases? A case study in survey design. *Transactions of the Association for Computational Linguistics, 12*, 1011–1026. https://doi.org/10.1162/tacl_a_00685

arXiv: 2311.04076

## III-B. Personality-conditioned LLMs

### Serapio-García et al. (2025) ✅ — Nature MI 査読版
Serapio-García, G., Safdari, M., Crepy, C., Sun, L., Fitz, S., Romero, P., Abdulhai, M., Faust, A., & Matarić, M. (2025). A psychometric framework for evaluating and shaping personality traits in large language models. *Nature Machine Intelligence, 7*(12), 1954–1968. https://doi.org/10.1038/s42256-025-01115-6

arXiv 前身（同著者・別タイトル）: Serapio-García et al. (2023). *Personality traits in large language models*. arXiv:2307.00184.

### Salecha et al. (2024) ✅
Salecha, A., Ireland, M. E., Subrahmanya, S., Sedoc, J., Ungar, L. H., & Eichstaedt, J. C. (2024). Large language models display human-like social desirability biases in Big Five personality surveys. *PNAS Nexus, 3*(12), pgae533. https://doi.org/10.1093/pnasnexus/pgae533

arXiv: 2405.06058

### Jiang et al. (2024) — PersonaLLM ✅
Jiang, H., Zhang, X., Cao, X., Breazeal, C., Roy, D., & Kabbara, J. (2024). PersonaLLM: Investigating the ability of large language models to express personality traits. In *Findings of the Association for Computational Linguistics: NAACL 2024*. https://aclanthology.org/2024.findings-naacl.229/

arXiv: 2305.02547

## III-C. LLM validation as research instrument

### Dillion, Tandon, Gu, & Gray (2023) ✅
Dillion, D., Tandon, N., Gu, Y., & Gray, K. (2023). Can AI language models replace human participants? *Trends in Cognitive Sciences, 27*(7), 597–600. https://doi.org/10.1016/j.tics.2023.04.008

### Bisbee, Clinton, Dorff, Kenkel, & Larson (2024) ✅
Bisbee, J., Clinton, J. D., Dorff, C., Kenkel, B., & Larson, J. M. (2024). Synthetic replacements for human survey data? The perils of large language models. *Political Analysis, 32*(4), 401–416.

Replication: https://doi.org/10.7910/DVN/VPN481

## III-D. LLM agents in education

### Markel, Opferman, Landay, & Piech (2023) — GPTeach ✅
Markel, J. M., Opferman, S. G., Landay, J. A., & Piech, C. (2023). GPTeach: Interactive TA training with GPT-based students. In *Proceedings of the Tenth ACM Conference on Learning @ Scale (L@S '23)* (pp. 226–236). ACM. https://doi.org/10.1145/3573051.3593393

### Shetye (2024) — Khanmigo evaluation ✅
Shetye, S. (2024). An evaluation of Khanmigo, a generative AI tool, as a computer-assisted language learning app. *Studies in Applied Linguistics & TESOL, 24*(1), 38–53.

ERIC: EJ1435677 | 単著の application study、有用だが foundational ではない。

### Xiao & Shen (2026) — Personality-Driven Student Agent ✅✅ ★
Xiao, B., & Shen, Q. (2026). *Personality-driven student agent-based modeling in mathematics education: How well do student agents align with human learners?* [Preprint, arXiv:2603.21358v1, 2026-03-22]. https://doi.org/10.48550/arXiv.2603.21358

著者: University of Florida 所属 2 名（査読 venue 未確定）。コード: https://github.com/kitayamachingtak/bigfivestudent.git

本セッションで一次完読。**本研究の最近接競合候補だったが、検証の結果競合度は低いと判断**。

**設計**:
- データ: NuminaMath-CoT（LI et al. 2024）の数学問題を 4 領域に分類（Algebra 1,341 / Geometry 672 / Counting & Probability 547 / Number Theory 484、計 3,044 問）。**実学生データは使用なし**
- Backbone: gpt-oss-120b（OpenAI 2025）
- 性格表現: Big Five 5 traits の **各 "high" 1 種類のみ計 5 persona**（連続値ではない）
- Pipeline: 学習 round（teacher-interaction / self-study / rest を agent が選択）→ 試験 round（100 問）
- 学習 round 数: 10 / 20 / 50 を独立 3 回試行
- 教師 agent には memory なし、学生 personality は事前通知

**評価方法（致命的限界）**:
- **実学生データではなく文献基準のみで評価**
- 13 件の Big Five × 学習研究を収集、14 評価項目に蒸留
- 各項目を Accurate (1) / Partial (0.5) / Not Met (0) で**手動採点**
- 「71.4% alignment」 = 10/14 項目（合計 10.0 / 14）

**主要結果**:
- 学習で全 4 領域・全 personality で F1 改善
- 50 round では over-learning で性能低下
- **High-extraversion が最高得点**（高 conscientiousness ではない）
- High-neuroticism は最低（毎 round 教師に質問するパターン）
- High-conscientiousness は教師接触最少・timestamp 消費最少

**著者自身が述べる限界**（本文 Limitations）:
- "lack of a well-defined quantitative framework for evaluating how closely agent behavior mirrors that of human learners"
- "Personality prompts are static, but real human personalities change dynamically"
- "While we can confirm a correlation between academic and exam performance and the established personality traits, **the mechanism remains unclear**"

**本研究との差別化（明確）**:
| 軸 | Xiao & Shen 2026 | 本研究 |
|---|---|---|
| ground truth | 13 文献基準のみ | N=103 実学生 + 実入試結果 |
| personality | 5 種類（high のみ） | 連続値 Big Five |
| outcome | NuminaMath F1 (生成テスト) | 実大学入試偏差値 |
| 検証粒度 | 集団レベル定性 | 個人レベル定量 |
| 行動データ | 合成 (agent 内部のみ) | StudySapuri 実ログ |
| 言語・文脈 | 英語数学問題 | 日本高校生・大学受験 |

## III-E. 個人予測の限界（Lundberg 2024 の元論文）

### Salganik et al. (2020) — Fragile Families Challenge ✅
Salganik, M. J., Lundberg, I., Kindel, A. T., Ahearn, C. E., Al-Ghoneim, K., Almaatouq, A., Altschul, D. M., Brand, J. E., Carnegie, N. B., Compton, R. J., Datta, D., Davidson, T., Filippova, A., Gilroy, C., Goode, B. J., Jahani, E., Kashyap, R., Kirchner, A., McKay, S., ... McLanahan, S. (2020). Measuring the predictability of life outcomes with a scientific mass collaboration. *Proceedings of the National Academy of Sciences, 117*(15), 8398–8403. https://doi.org/10.1073/pnas.1915006117

100+ author の mass collaboration（APA 7th: 最初 19 + 省略記号 + 最終著者）。Fragile Families Challenge の主結果を初出。Park (2024) lecture が個人予測困難の論拠として引用。**Lundberg 2024（intro 既引用）の元論文**として必須。

## III-F. 自己制御学習（Big Five → 学習行動 → 成果の中間機構）

### Zimmerman (2008) — SRL Investigation ✅
Zimmerman, B. J. (2008). Investigating self-regulation and motivation: Historical background, methodological developments, and future prospects. *American Educational Research Journal, 45*(1), 166–183. https://doi.org/10.3102/0002831207312909

Tokiwa (2025) も引用、Big Five (特に C) → 学習行動 mediation の理論的枠組み。

## III-G. 性格変化介入（Part 2 反実仮想介入の根拠）

### Roberts et al. (2017) — Personality Trait Change Meta-Analysis ✅
Roberts, B. W., Luo, J., Briley, D. A., Chow, P. I., Su, R., & Hill, P. L. (2017). A systematic review of personality trait change through intervention. *Psychological Bulletin, 143*(2), 117–141. https://doi.org/10.1037/bul0000088

207 件のメタ分析。「Big Five は介入で変化可能」を実証 (d = 0.37, 24 週間)。Part 2 で C を +1 SD に底上げする反実仮想シミュレーションの **ecological validity** を支える。

---

# Part IV: 既存所有（metaanalysis/ 経由）

メタ分析論文の `metaanalysis/reference_index.md` Part I (44 PDF) と Part II（測定尺度・統計手法）は本シミュレーション論文でも再利用可能。

主要再利用候補:

- **Tokiwa (2025)** ✅ — 自己引用、BFI-2-J × StudySapuri (Frontiers in Psychology, 16, 1420996)
- **Tokiwa (2026)** ✅ — 自己メタ分析（OSF DOI 10.17605/OSF.IO/E5W47）
- **Mammadov (2022)** ✅ — Big Five × academic achievement meta-analysis
- **Poropat (2009)** ✅ — 古典メタ分析
- **Soto & John (2017)** ✅ — BFI-2 原典
- **Yoshino et al. (2022)** ✅ — BFI-2-J 日本語版

詳細は [`metaanalysis/reference_index.md`](../metaanalysis/reference_index.md) Part I + II 参照。

---

# 過去のハルシネーション記録（教訓として保持）

旧 `simulation/paper/introduction.md` には AI ハルシネーション由来の誤引用が確認されていた:

| 場所 | 誤（旧 intro.md） | 正（検証済） |
|---|---|---|
| §11 | Lundberg, I., Brand, J. E., & Jeon, N. (2024) | Lundberg, I., Brown-Weinstock, R., Clampet-Lundquist, S., Pachman, S., Nelson, T. J., Yang, V., Edin, K., & Salganik, M. J. (2024) |

introduction.md は 2026-04-26 に**全削除**（方向性再検討のため）。次回の Introduction 執筆時には **本ファイルの正しいエントリのみ**から引用候補を選択すること。

---

# ⚠️ 引用前の必須一次確認リスト（未読の重要候補）

以下は本セッションのサブエージェント調査で「重要」と判定されたが、**一次資料未確認**のため引用前に必ず PDF / 元ページで再確認が必要:

| arXiv ID / 出典 | 第一著者 | 主張 | 重要度 |
|---|---|---|---|
| 2508.02679 | Yan et al. (2025) | StudentLife smartphone sensing × Big Five で行動 grounding agent | 高（本研究の最近接候補） |
| 2509.13397 | (Anonymous) | Prompt 構成で LLM simulation 結果が大きく揺れる | 高（方法論リスク） |
| 2509.03730 | Wen et al. (2025) | LLM の自己報告と行動の乖離 | 中（行動 grounding の理論的根拠） |
| Stachl et al. (2020) PNAS | Stachl | Smartphone behavior → Big Five 推定 r ≈ 0.37 | 高（情報重複検証） |
| Binz et al. (2025) Nature | Binz | Centaur foundation model（Toubia が試した代替） | 中 |
| 2510.07230 | Wang et al. (Customer-R1) | RL ベースの persona × clickstream 個別化 | 中（EC ドメインの行動 grounding） |
| 2503.20749 | Lu et al. | LLM agent の行動精度 11.86%（Park 系の限界批判） | 中 |
| Komarraju et al. (2011) | Komarraju | Big Five → 学習方略 → GPA mediation の古典 | 中（mediation 理論的支柱） |
| 2404.07963 | Xu & Zhang (EduAgent) | 認知 prior + persona から学習 trajectory 生成 | 中 |
| 2501.10332 | Liu et al. (Agent4Edu, AAAI 2025) | persona-memory-action の 3 段モジュール | 中 |
| 2502.02780 | Xu et al. (Classroom Simulacra) | N=60 の 6 週ログで Transferable Iterative Reflection | 中 |

---

# 検証手順（再現用）

各 ✅ マーク付き引用は以下のいずれかで原文確認済:
1. `simulation/prior_research/_text/*.txt` — 一次 PDF を本セッションで pdftotext 抽出（✅✅）
2. `simulation/docs/notes/cs222/papers/paper_*.md` — CS 222 受講中の精読ノート
3. `simulation/docs/stanford univ AI Agents and Simulations/*.pdf` — CS 222 配布 PDF
4. arXiv abstract page / WebSearch スニペット
5. journal landing page — DOI 解決経由
6. PubMed / PMC entry — author list と article number 検証

執筆時の手順:
1. 本ファイルから引用候補を選択
2. ✅✅ または ✅ ならそのまま APA 7th 形式で使用可
3. ⚠️ マーク付きは**必ず一次資料で再確認してから ✅ に昇格させる**
4. 新規追加候補は一次資料で確認 → ✅ で登録してから引用
5. AI ハルシネーション混入を防ぐため、サブエージェント要約からの直接転記は禁止
