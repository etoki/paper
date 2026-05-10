# Hallucination Check Protocol

**目的**: マニュスクリプトのハルシネーションチェックを毎回ゼロから全文走査する代わりに、
タスクを MECE 分割して個別に実行できるようにする。

**対象**: `metaanalysis/paper_v2/build_docx.py` (および `references_data.py`)

**真実の根拠**: `metaanalysis/analysis/CANONICAL_RESULTS.md` および以下の派生ファイル
- `analysis/pooling_results.csv` — primary pooled effects
- `analysis/moderator_results.csv` — region/era/outcome moderators
- `analysis/sensitivity_results.md` — sensitivity analyses
- `analysis/data_extraction_populated.csv` — 31 件の study metadata
- `deep_reading_notes.md` — 引用著者・年・ページ等

**呼び出し方**: ユーザーは「T1 やって」「auto check 実行」などで個別タスクを指定できる。

---

## タスク一覧 (MECE)

| ID | タスク名 | 自動化 | 所要 | チェック内容 |
|----|---------|------|------|-------------|
| **T1** | Tables 2-5 数値チェック | 完全自動 | ~30秒 | 全 60+ セルの r/CI/I²/Q/τ²/k/N を canonical と照合 |
| **T2** | 本文 pooled effects チェック | 完全自動 | ~30秒 | Abstract/Results/Discussion の trait pooled 値が一貫しているか |
| **T3** | 寄与研究 (contributor) チェック | 半自動 | ~3分 | β-converted 寄与 = Yu+Kaspar のみ等の指示語ベース検証 |
| **T4** | 集計値 (counts/breakdowns) チェック | 完全自動 | ~30秒 | retained=25, primary=10, education/region/era 内訳 |
| **T5** | 引用文献 (references) audit | 完全自動 | ~1分 | 引用著者 ⇔ references_data.py 完全対応 |
| **T6** | PRISMA flow 算術チェック | 完全自動 | ~10秒 | 38-11=27, 27+4=31, 31-6=25 の算術整合 |
| **T7** | Prior meta-analyses ベンチマーク | 半自動 | ~1分 | Mammadov/Meyer/Chen 等の k, N, ρ 値が deep_reading_notes と一致 |

**自動化可能 (T1, T2, T4, T6)**: `python3 analysis/check_hallucinations.py` で一括実行可能。

**半自動 (T3, T5, T7)**: 機械検出後、文脈判断が必要な箇所を Claude が手動確認。

---

## T1: Tables 2-5 数値チェック

**スクリプト**: `analysis/check_hallucinations.py --task t1`

**チェック対象**:
- Table 2 (Pooled effects): 5 traits × 8 columns (k, N, r, CI, PI, I², τ², Q(df), p)
- Table 3 (Moderators): 30 行 (3 moderators × 5 traits × 2 levels)
- Table 4 (Sensitivity): 30 行 (5 traits × 6 analyses)
- Table 5 (GRADE): 5 行 × 8 列

**真実の根拠**: `pooling_results.csv`, `moderator_results.csv`, `sensitivity_results.md`

**典型的失敗パターン**: CI 境界値の hand-typing による捏造（例: `[−.134, .060]` vs canonical `[−.115, .039]`）

---

## T2: 本文 pooled effects チェック

**スクリプト**: `analysis/check_hallucinations.py --task t2`

**チェック対象**:
- Abstract で言及される r 値・CI 値が canonical と一致
- Results part 2 (per-trait sections) が canonical と一致
- Discussion summary 段落が canonical と一致
- 全箇所で同じ trait に同じ値を使っているか (一貫性)

**真実の根拠**: `pooling_results.csv`

**典型的失敗パターン**: 同じ値を複数箇所で hand-type して微妙にズレる

---

## T3: 寄与研究 (contributor) チェック

**スクリプト**: `analysis/check_hallucinations.py --task t3` (機械検出部分)

**機械チェック**:
- "β-converted" の文脈で Mustafa, Wang, Bhagat, Audet 等が cited されていないか
- "primary pool" 言及で含まれる study ID が pooling_summary に存在するか
- Leave-one-out の Δr 値が sensitivity_results と一致するか

**手動確認 (Claude)**:
- "driven by", "principally", "the largest contribution" 等の主張が pooling_summary の contributing studies と整合しているか
- Discussion で primary pool 外の研究を "drove the pool" として誤って引用していないか

**真実の根拠**: `pooling_summary.md` (contributing studies per trait), `sensitivity_results.md`

**典型的失敗パターン**: 二次アウトカム研究 (A-12 Baruth, A-26 Wang) を primary pool driver と誤記

---

## T4: 集計値 (counts/breakdowns) チェック

**スクリプト**: `analysis/check_hallucinations.py --task t4`

**チェック対象**:
- "31 catalogued primary studies" の言及
- "25 retained for qualitative synthesis" の言及
- "10 contributing to primary quantitative pool" の言及
- Education 内訳: K-12 (2), UG (15), UG/Grad (5), Grad (2), Sec/Post (1) = 25
- Region 内訳: Asia (12), Europe (7), NA (6) = 25
- Era 内訳: pre-COVID (8), COVID (12), post-COVID (4), mixed (1) = 25
- Modality: fully online (24), not online-specific (1) = 25
- Instrument: BFI (13), NEO (2), IPIP (3), HEXACO (1), TIPI (2), Other (4) = 25
- RoB: mean=5.44, SD=0.87, range 4-7, ≥5: 21, <5: 4

**真実の根拠**: `data_extraction_populated.csv` (`inclusion_status` 列でフィルタ)

**典型的失敗パターン**: 25 vs 31 の取り違え、または旧 v1 値 (UG=22, Asia=13, RoB mean=5.6) の残存

---

## T5: 引用文献 (references) audit

**スクリプト**: `analysis/check_hallucinations.py --task t5`

**チェック対象**:
1. 本文で `Author (YYYY)` 形式で引用されている著者が `references_data.py` に存在するか
2. `references_data.py` の各 entry の年・巻号・ページ・DOI が `deep_reading_notes.md` の "Citation:" 行と一致するか
3. APA 7 整合: reference list の全 entry が本文で 1 回以上 cite されているか

**典型的失敗パターン**:
- 引用しているが reference list に無い (前回の Mustafa, Quigley, Kaspar など)
- reference に巻号・ページを fabricate する (前回の Kaspar "42, 33153–33170")
- 著者名のスペル違い (Bahçekapılı の特殊文字)

**真実の根拠**: `deep_reading_notes.md` の各 A-XX セクションの "**Citation**:" 行

---

## T6: PRISMA flow 算術チェック

**スクリプト**: `analysis/check_hallucinations.py --task t6`

**チェック対象**:
- 38 (full-text assessed) - 11 (excluded) = 27 (passed eligibility)
- 27 + 4 (newly added: A-29, A-30, A-31, A-37) = 31 (catalogued in Table 1)
- 31 - 6 (post-eligibility excluded: A-05, A-09, A-10, A-16, A-24, A-27) = 25 (retained)
- 25 = 10 (primary pool) + 15 (secondary-only)
- Exclusion 内訳: 5 non-BFI + 4 face-to-face (A-09, A-10, A-14, A-16) + 1 overlap (A-05) + 1 not-extractable (A-24) = 11

**典型的失敗パターン**: "Seven reports were excluded" など合計が合わない記述

---

## T7: Prior meta-analyses ベンチマーク

**スクリプト**: `analysis/check_hallucinations.py --task t7`

**チェック対象** (各引用が deep_reading_notes と一致):

| 引用 | k | N | C ρ | O ρ | Notes |
|------|---|---|-----|-----|-------|
| Poropat (2009) | 138 | 70,926 | .22 | .12 | Tertiary |
| McAbee & Oswald (2013) | 57 | 26,382 | .26 | — | GPA |
| Vedel (2014) | 21 | 17,717 | .26 | — | Tertiary |
| Stajkovic et al. (2018) | 5 samples | 875 | .21 | — | β SE→Perf .24-.33 |
| Mammadov (2022) | 267 | 413,074 | .27 | .16 | Asia C=.35, A=.23, N=-.19 |
| Meyer et al. (2023) | 110 | 500,218 | .24 | .21 | K-12 only |
| Zell & Lesick (2022) | 54 | >500K | .28 (academic) | .14 | umbrella |
| Chen et al. (2025) | 84 articles, 370 corr | 46,729 | .206 | .081 | University N≥200 |
| Hunter et al. (2025) | 23 (from 848) | 1,542 | qualitative | — | Narrative review |

**典型的失敗パターン**: k と N の取り違え、Asian-amplified 値の混同

---

## 推奨運用

### 通常時 (マニュスクリプト編集後)

```bash
python3 metaanalysis/analysis/check_hallucinations.py
```

これで T1, T2, T4, T6 を自動実行し、不一致を report。

### 詳細チェック時 (preprint/journal 提出前)

ユーザー → Claude:
> 「T3, T5, T7 をチェック」

Claude が半自動チェック (機械 + 文脈判断) を実行。

### 個別タスク

ユーザー → Claude:
> 「T1 だけ実行」「T5 だけ実行」

該当タスクのみ実行。

---

## 履歴

- 2026-05-01 初版: 7 タスクに分割
- 過去発見した hallucination パターン:
  - Round 1 (`adaf88b`): 17 件欠落引用 + Mustafa β寄与誤記 + PRISMA 算術不整合(合計 7 種)
  - Round 2 (`f60760f`): Table 3 outcome_type CI 全 10 行捏造 + Table 4 sensitivity CI 全 10 行捏造 + Leave-one-out 数値誤記(合計 5 種)
  - Round 3 (`e0f0c89`): post-COVID k=2 vs canonical k=0(1 件)
  - **Round 4 (`5cef6e7` → `2168a94`, 2026-05-10)**: A-25 Tokiwa (2025) の citation を「Frontiers in Psychology 16, 1420996(実刊行)」ではなく「Manuscript in preparation, SUNBLAZE Co., Ltd.」と誤記。誤りが親 preprint の `references_data.py` に潜伏し、reference_index.md A-25 → 4 conference paper References → cover letters → 親 Research Square v1 PDF(deposited 2026-04-27)reference list 32 番まで全コピー先に伝播。

これらのパターンは新しいハルシネーションが発生する可能性のある場所で、各タスクで重点的にチェックする。

---

## ⚠️ ルール: 著者自身の論文 entry を引用する場合(Round 4 後追加)

**著者自身の刊行物**(Tokiwa の場合は `online_learning/`、`clustering/`、`harassment/`、`metaanalysis/`、`simulation/` の各 directory に PDF を所持)を citation に含める場合は、**reference_index.md / references_data.py / 任意の引用個所のいずれを「正」と思って使う前に、必ず該当 directory の corpus PDF と直接突き合わせる**こと。

理由: reference_index は **secondary source**(どこかから合成した index)であり、source-of-truth(実 PDF)が別の場所にある場合、index 側に潜伏した hallucination がそのまま伝播する。Round 4 では:

```
metaanalysis/data_extraction.csv      (truth, A-25 Frontiers DOI 正記)
metaanalysis/literature_review.md     (truth, A-25 Frontiers DOI 正記)
metaanalysis/pdf_download_urls.md     (truth, A-25 Frontiers DOI 正記)
                                                ↓
                                      references_data.py
                                        (synthesised:
                                         「Manuscript in preparation」 ← regression)
                                                ↓
                                      reference_index.md A-25
                                                ↓
                                      4 conference paper References
                                                ↓
                                      4 cover letters
                                                ↓
                                      Research Square v1 PDF
```

**チェック手順**(著者の刊行物を新規 citation する時):

1. リポジトリ内で `find . -name "*.pdf" -path "*/<著者の paper directory>*"` で当該論文の PDF が corpus にあるか確認。
2. PDF を `pdftotext -l 1` で 1 ページ目を抽出し、**タイトル・誌名・巻号・DOI・掲載日**を target source として確定。
3. 同一 author の `[Manuscript in preparation]` / `[unpublished]` / `[under review]` 等の placeholder 表記が `reference_index.md` や source code に既存している場合、**warn flag を立てて step 2 の verification を再実行**するまで該当 entry を信用しない。
4. submission 前に `metaanalysis/conference_submissions/scripts/check_dois.py` で全 DOI の syntax と(可能なら)resolution を verify。
5. 「Manuscript in preparation」「[unpublished]」を citation に書く前に **本当に未刊行か** を該当 directory PDF + DOI 検索で再確認。著者本人の業績は刊行ステータスを誤りやすい。

このルールは `metaanalysis/conference_submissions/scripts/check_numbers.py` の "(A) Cross-paper canonical-value consistency" にも組み込み済(Tokiwa 2025 Frontiers DOI `10.3389/fpsyg.2025.1420996` を canonical anchor として全 paper 横断で検証)。
