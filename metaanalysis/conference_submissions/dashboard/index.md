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
| **ECEL 2026** | S | ✅ GO | full paper draft + cover letter ready | P3 modality moderator | ECEL 延長返信 → 5/14 Awards / 5/21 Full Paper |
| **ICEEL 2026** | A | ✅ GO | full paper draft + cover letter ready | P3 Hofstede + Japan | 6/30 投稿(東京 in-person) |
| **ICERI 2026** | A | ✅ GO | full paper draft + cover letter ready | P3 cross-tab | 7/9 投稿(virtual 申請) |
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

## ✅ TODO リスト(今後 30 日)

### 5/10 〜 5/14
- [x] Frontiers (Joe Fares) 宛 follow-up email 送信(A-25 reference 訂正、Frontiers in Education v2 件、ECEL とは別件)
- [ ] **ECEL 委員会(`info@academic-conferences.org`)延長承認返信受領待ち** — 元 abstract 締切 4/22 既経過、延長承認が来てから Awards / Full Paper が submit 可能
- [x] ECEL Excellence Awards 投稿判断 → **GO**(`awards_abstract.md` draft 完成、5/14 投稿は延長承認後)
- [ ] PRISMA flow diagram 各 paper への埋め込み(figures は `figures/prisma_flow_*.png` に生成済)

### 5/15 〜 5/21
- [ ] ECEL 延長承認後 → ECEL Full Paper 投稿(`helen@academic-conferences.org`)
- [ ] ECEL cover letter 最終 review
- [ ] 延長不承認 or 5/13 まで返信なし時 → ECEL 投稿全体取り下げ判断

### 5/22 〜 6/30
- [ ] ICEEL paper 最終 review → 6/30 投稿(zmeeting)

### 7 月
- [ ] ICERI abstract / paper 投稿(7/9, IATED)
- [ ] ECEL 採択結果確認(7/20)

### 8 月以降
- [ ] ECEL 最終版(8/27)+ author payment(9/17)
- [ ] ICERI 採択結果(8 月)
- [ ] ICEEL 採択結果(9 月)
- [ ] 各学会への参加・発表(virtual or 東京現地)
- [ ] SFC 提出書類作成

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

最終更新: 2026-05-10
