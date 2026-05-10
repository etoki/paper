# 国際学会投稿戦略ダッシュボード

**ベース日付**: 2026-05-10(日) / **ゴール**: 2026 年中に 2 件採択(SFC 論文博士要件)
**Branch**: `claude/conference-submission-strategy-ZNJ6X`(ZDn3N をマージ済)

---

## 🚨 直近締切カウントダウン

| 残日数 | 日付 | イベント | 学会 | アクション |
|---:|---|---|---|---|
| **D-4** | 2026-05-14 | Excellence Awards abstract 締切 | ECEL | 検討 → 投稿 or skip |
| **D-11** | 2026-05-21 | Full Paper 査読提出締切 | ECEL | **投稿(Frontiers Joe Fares 返信待ち)** |
| **D-51** | 2026-06-30 | Paper 締切 | ICEEL(東京) | **投稿** |
| **D-60** | 2026-07-09 | Paper/Abstract 締切 | ICERI(Seville、virtual) | **保険投稿** |
| TBD | TBD | CFP 公開待ち | AAOU 2026 | 週次チェック |
| TBD | TBD | CFP 公開待ち | AACE E-Learn | 月次チェック |

---

## 📋 学会別ステータス

| 学会 | Tier | Decision | Status | Paper | Action |
|---|---|---|---|---|---|
| **ECEL 2026** | S | ✅ GO | full paper draft + cover letter ready / Awards abstract ready | P3 modality moderator | **ECEL 委員会延長返信待ち**(5/13 まで)→ 承認後 5/14 Awards / 5/21 Full Paper |
| **ICEEL 2026** | A | ✅ **SUBMITTED (2026-05-10)** | zmeeting 投稿済 / Hofstede direction-match (3/5; N+ headline) Option 2 採用版 | P3 Hofstede + Japan | acceptance 通知待ち(7/30) |
| **ICERI 2026** | A | ✅ **SUBMITTED (2026-05-10)** | myIATED abstract 投稿済 | P3 cross-tab | acceptance 通知待ち(9/1)→ 採択時 full paper 提出 |
| **IEEE TALE 2026** | S | ❌ **DROP** | — | — | Research Square 不可 + 無移行方針 → 投稿経路なし |
| AAOU 2026 | A | 🟡 WATCH | CFP 待ち | 別 chat | 週次 CFP 確認 |
| AACE E-Learn 2026 | A | 🟡 WATCH | CFP 待ち | 別 chat | 月次 CFP 確認 |
| WORLDCTE 2026 | B | 🟡 OPTION | mid-tier 保険 | 別 chat | 7/3 判断 |
| ICTLE 2026 | B | ❌ SKIP | WORLDCTE と重複 | — | — |
| EDULEARN 2026 | C | ❌ SKIP | 締切経過 | — | — |
| ASCILITE 2026 | B | ❌ DROP | live 必須 | — | — |

凡例: ✅ GO / ⚠️ HOLD / 🟡 OPTION / ❌ SKIP

---

## 🔬 P3 サブペーパー解析パイプライン(2026-05-10 時点)

`metaanalysis/conference_submissions/{ecel,iceel,iceri}/` に各学会別の analysis_plan / abstract / paper_outline / scripts / results を配置。

| 学会 | 新規分析 | スクリプト | 結果 |
|---|---|---|---|
| **ECEL** | Modality moderator + interaction | `ecel/scripts/run_modality_meta.py` | ✅ 実行済 / Q_b for E×modality 17.60 (p<.001) / interaction Wald χ²(4) = 13.64, p = .0085 |
| **ICEEL** | Hofstede 文化次元 moderator + Japan focus | `iceel/scripts/run_hofstede_meta.py` | ✅ 実行済 / k = 2 Asian primary、df_resid = 0、slopes descriptive only / Asian C = 0.111, E = -0.131, N = 0.089 |
| **ICERI** | Education-level × Discipline 4×4 cross-tab | `iceri/scripts/run_cross_tab_meta.py` | ✅ 実行済 / Wald χ²(6) = 1.638, p = .9498 / UG×Psychology C = 0.292 |

詳細: 各 venue の `results/summary.md` を参照。

### Cross-paper consistency / DOI checks

| Script | Coverage | Last result |
|---|---|---|
| `scripts/check_numbers.py` | 3 paper 間で重複数値 + 各 paper body vs results CSV | ✅ 0 failures |
| `scripts/check_dois.py --offline` | 全 38 DOI の syntax | ✅ 0 failures |
| `scripts/check_dois.py` (online) | DOI resolution | ⚠ sandbox 内では `host_not_allowed`、ローカル実行で確認 |

---

## ✅ TODO リスト(今後 6 か月)

### 5/10 〜 5/14
- [x] Frontiers (Joe Fares) 宛 follow-up email 送信(A-25 reference 訂正、Frontiers in Education v2 件、ECEL とは別件)
- [x] **ICEEL 2026 投稿**(zmeeting、Hofstede direction-match Option 2 版)
- [x] **ICERI 2026 投稿**(myIATED、cross-tab 版)
- [ ] **ECEL 委員会(`info@academic-conferences.org`)延長承認返信受領待ち**(〜5/13)
- [x] ECEL Excellence Awards 投稿判断 → **GO**(`awards_abstract.md` ready)

### 5/15 〜 5/21(ECEL 延長承認時のみ)
- [ ] 5/14: ECEL Awards abstract 投稿
- [ ] 5/21: ECEL Full Paper 投稿(`helen@academic-conferences.org`)
- [ ] 延長不承認時 → ECEL 取り下げ → AAOU / AACE / WORLDCTE の保険判断

### 6 月
- [ ] AAOU 2026 / AACE E-Learn CFP 公開チェック(週次・月次)

### 7-9 月(査読結果待ち期間)
- [ ] 7/20: ECEL 採択結果確認(投稿していれば)
- [ ] 7/30: **ICEEL 採択結果通知**
- [ ] 9/1: **ICERI 採択結果通知** → 採択時 full paper 執筆
- [ ] 採択結果に応じて保険投稿(AAOU/AACE/WORLDCTE)判断

### 10 月以降
- [ ] ECEL 最終版(8/27)+ author payment(9/17)※採択時
- [ ] ICERI full paper 提出(9 月以降、myIATED で deadline 別途通知)
- [ ] ICEEL カメラレディ + early-bird 登録(8/30)
- [ ] 11/9-11 ICERI 参加(virtual)
- [ ] 11/27-29 ICEEL 参加(東京 in-person)
- [ ] SFC 論文博士 提出書類作成

---

## 🔒 Hard rules(変更不可)

1. **Research Square preprint stays up.**(2026-05-09 確定)各学会への投稿は preprint DOI を必ず disclose。`templates/preprint_disclosure_template.md` Version B を ECEL/ICEEL/ICERI で使用。
2. **Research Square 不可の学会は drop**(投稿せず)。Migration は永続的に実施しない。
3. **各 conference paper は preprint に「無い」新規分析を含む**(self-plagiarism firewall)。
4. **メールは自動送信しない。** 全 outgoing は `templates/` で起草、投稿者本人がレビューしてから送信。
5. **Single author only.** 共同研究者なし。
6. **論文の数値は CSV に必ず traceable**(`<venue>/results/`)。手打ち禁止。`scripts/check_numbers.py` で自動検証。
7. **DOI は捏造禁止。** `reference_index.md` 未収録は引用しない。submission 前に `scripts/check_dois.py` 実行。

---

## 🎯 成功基準(2026 年末まで)

- [ ] **2 件以上の採択獲得**(SFC 論文博士要件) ← ECEL + ICEEL で達成見込
- [ ] うち最低 1 件は Scopus/WoS 索引付き proceedings(ECEL → Scopus + WoS CPCI)
- [ ] 全て virtual presentation または東京現地で参加可能
- [ ] 参加費合計 < $2,000

---

## 過去の重要訂正(audit trail)

- **2026-05-10**: A-25 Tokiwa (2025) reference を全 paper / 親 preprint source code から訂正。旧 "Manuscript in preparation, SUNBLAZE Co., Ltd." → 正 *Frontiers in Psychology, 16*, 1420996, DOI 10.3389/fpsyg.2025.1420996(CC BY)。詳細は PR #14 commit message。
- **2026-05-10**: IEEE TALE 2026 を portfolio から DROP。Research Square 不許可 + 「RS 維持」方針で投稿経路なし。
- **2026-05-10**: 全 conference 関連ファイルを `metaanalysis/conference_submissions/` 配下に集約。
- **2026-05-10**: ICEEL paper Option 2 採用 — Hofstede direction-match (3/5 trait, N+ headline) を 8 件 primary citation で armoring。詳細は PR #19。
- **2026-05-10**: ICEEL + ICERI 投稿完了。ECEL は委員会延長返信待ち。

最終更新: 2026-05-10(2 件投稿完了後)
