# 国際学会投稿戦略ダッシュボード

**ベース日付:** 2026-05-09(土) / **ゴール:** 2026年中に2件採択
**Branch:** `claude/conference-submission-strategy-ZNJ6X`(ZDn3N をマージ済み)

---

## 🚨 直近締切カウントダウン(本日 = 2026-05-09)

| 残日数 | 日付 | イベント | 学会 | アクション |
|---:|---|---|---|---|
| **D-5** | 2026-05-14 | Excellence Awards abstract 締切 | ECEL | 検討 → 投稿 or skip |
| **D-12** | 2026-05-21 | Full Paper 査読提出締切 | ECEL | **延長申請済 + 投稿** |
| **D-34** | 2026-06-12 | Abstract 締切 | ICTLE | 投稿判断 |
| **D-50** | 2026-06-29 | Paper 締切 | ASCILITE | (DROP予定) |
| **D-51** | 2026-06-30 | Paper 締切 | ICEEL (東京) | **投稿** |
| **D-51** | 2026-06-30 | Paper 締切 | IEEE TALE | **投稿** |
| **D-54** | 2026-07-03 | Abstract 締切 | WORLDCTE | 保険判断 |
| **D-60** | 2026-07-09 | Abstract 締切 | ICERI | **保険投稿** |
| TBD | TBD | CFP公開待ち | AAOU 2026 | 週次チェック |
| TBD | TBD | CFP公開待ち | AACE E-Learn | 月次チェック |

---

## 📋 学会別ステータス

| 学会 | Tier | Decision | Status | Paper | Action |
|---|---|---|---|---|---|
| ECEL 2026 | S | ✅ GO | **投稿準備中(モダリティ解析実施済)** | P3 (モダリティ moderator 追加) | 5/14 abstract / 5/21 full paper |
| IEEE TALE 2026 | S | ⚠️ HOLD | virtual確認待ち | P3 (ML pipeline) または P5 | 5/12週: virtual照会メール |
| ICEEL 2026 | A | ✅ GO | 投稿準備中 | P3 (Hofstede + Japan) または P1派生 | 6/30: 投稿 |
| ICERI 2026 | A | 🟡 SAVE | 保険として | P3 (Edu x Discipline) または P5 | 7/9: 保険投稿 |
| AAOU 2026 | A | 🟡 WATCH | CFP待ち | P1派生 | 週次CFP確認 |
| AACE E-Learn 2026 | A | 🟡 WATCH | CFP待ち | P5 または P1派生 | 月次CFP確認 |
| WORLDCTE 2026 | B | 🟡 OPTION | mid-tier保険 | P3または P1派生 | 7/3: 判断 |
| ICTLE 2026 | B | ❌ SKIP | WORLDCTEと重複 | - | - |
| EDULEARN 2026 | C | ❌ SKIP | 締切経過 | - | - |
| ASCILITE 2026 | B | ❌ DROP | live必須 | - | - |

凡例: ✅ GO / ⚠️ HOLD / 🟡 OPTION / ❌ SKIP

---

## 🔬 P3 サブペーパー解析パイプライン(本日 5/9 時点)

`metaanalysis/conference_submissions/` 配下に各学会別の analysis_plan / abstract / paper_outline と実行スクリプトを配置。

| 学会 | 新規分析 | スクリプト | 実行状況 | 結果 |
|---|---|---|---|---|
| ECEL | Modality moderator + interaction | `ecel/scripts/run_modality_meta.py` | ✅ 実行済 | E × modality Q_b=15.52, p<.001 / 交互作用 Wald χ²(8)=14.27, p=.075 |
| IEEE TALE | ML (LR/RF/XGB) + SHAP + fairlearn | `ieee_tale/scripts/run_ml_pipeline.py` | ⚙️ 骨格(sklearn 動作確認済) | N=10 で predictive性能は弱い、interpretability 用途に caveat 付き |
| ICEEL | Hofstede 文化次元 moderator + Japan focus | (要実装) | 🔜 5月後半 | — |
| ICERI | Education-level × Discipline 3×3 cross-tab | (要実装) | 🔜 6月 | — |

詳細結果: `metaanalysis/conference_submissions/ecel/results/summary.md` ほか CSV 一式。

---

## ✅ TODO リスト(今後30日)

### 5月9日(本日)
- [x] ECEL 延長要請メール送信(`info@academic-conferences.org`)
- [x] **リポジトリ統合**(ZDn3N + ZNJ6X マージ → ZNJ6X 一本化)
- [x] ECEL 用 `studies.csv` 派生 + モダリティ実解析
- [ ] **意思決定: P3 preprint migration**(OSF / arXiv / 維持) — 投稿者判断待ち

### 5月10日〜14日
- [ ] ECEL Excellence Awards abstract 投稿判断(締切 5/14)
- [ ] ECEL 用 P3 abstract 数値の最終確認(`metaanalysis/conference_submissions/ecel/abstract.md`)
- [ ] sensitivity 4シナリオでロック(drop_beta_converted / drop_coi / drop_unspecified_modality)
- [ ] preprint migration 実行(意思決定後)
- [ ] AAOU 2026 CFP 公開チェック

### 5月15日〜21日
- [ ] ECEL Full Paper(10p, ACI Word テンプレ)執筆 → `helen@academic-conferences.org` に提出
- [ ] IEEE TALE への virtual presentation 照会メール送信(`templates/ieee_tale_virtual_inquiry.md`)
- [ ] IEEE TALE への P5 preprint server 適合性照会メール

### 5月22日〜31日
- [ ] IEEE TALE 4-6ページ conference paper 着手(P5 ベース or P3 ML)
- [ ] ICEEL 用 8-10ページ paper 着手(P3 Hofstede or P1派生)
- [ ] ECEL 査読結果待ち(7/20通知)

### 6月
- [ ] IEEE TALE paper 完成 → 6/30 EDAS 投稿
- [ ] ICEEL paper 完成 → 6/30 投稿
- [ ] ICERI abstract 着手

### 7月
- [ ] ICERI abstract 投稿(7/9)
- [ ] WORLDCTE 投稿判断(7/3)
- [ ] ECEL 採択結果確認(7/20)

### 8〜9月
- [ ] ECEL 最終版 (8/27) + author payment (9/17)
- [ ] IEEE TALE 採択結果(9/1)
- [ ] ICERI 採択結果

### 10月以降
- [ ] ICEEL 採択結果
- [ ] 各学会への参加・発表(virtual or in-person)
- [ ] SFC 提出書類作成(2件採択分)

---

## 🔬 Preprint Migration ステータス

| Paper | 現在 | 候補移行先 | 状態 |
|---|---|---|---|
| P3 metaanalysis | Research Square ⚠️ | OSF / arXiv | **意思決定待ち** |
| P5 simulation | OSF + SocArXiv ✅ | (既に non-profit) | 移行不要 |
| P4 harassment | 要確認 | - | メタデータ確認待ち |

詳細: `preprint_migration/README.md`

---

## 🔒 Hard rules(変更不可)

1. **Preprint stays up.**(Strategy 2 確定)各学会への投稿は preprint DOI を必ず disclose。`templates/preprint_disclosure_template.md` の Version A〜D を venue ごとに使い分ける。
2. **各 conference paper は preprint に「無い」新規分析を含むこと**(self-plagiarism firewall)。
3. **メールは自動送信しない。** 全 outgoing は `templates/` で起草、投稿者本人がレビューしてから送信。送信後は `_sent_<date>.md` 形式で履歴保存。
4. **Single author only.** 共同研究者なし。
5. **論文の数値は CSV に必ず traceable**(`metaanalysis/conference_submissions/<venue>/results/`)。手打ち禁止。

---

## 🎯 成功基準(2026年末まで)

- [ ] **2件以上の採択獲得**(SFC論文博士要件)
- [ ] うち最低1件は Scopus/WoS 索引付き proceedings
- [ ] 全て virtual presentation または東京現地で参加可能
- [ ] 参加費合計 < $2,000(目標)

---

最終更新: 2026-05-09(ZDn3N + ZNJ6X 統合後)
