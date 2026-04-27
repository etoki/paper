# OSF Project Sync Guide

**目的**: GitHub 上の analysis ファイル（コード・CSV・プロット・原稿）を OSF Project `79m5j` の 7 Components に同期する。

**登録済み Registration**（`e5w47`）は不変。本ガイドは **Project 本体**（編集可能）への追加手順。

**Preprint**: 本原稿は Research Square で公開済（2026-04-27, DOI 10.21203/rs.3.rs-9513298/v1）。詳細は `paper/preprint_info.md` を参照。

**最終更新**: 2026-04-27

---

## 前提: 3 つの同期方式の選択

| 方式 | 労力 | 同期性 | 推奨 |
|------|------|--------|------|
| **A. Manual upload (一括)** | 中（30分） | 一回限り | ✅ 最も簡単、本ガイドのメイン |
| **B. GitHub add-on** | 小（5分） | 継続自動同期 | ✅ 以降のcommitが自動反映 |
| **C. osfclient CLI** | 大（要 token 設定） | スクリプト化可能 | ⚠ 上級者向け |

**最短ルート**: まず **A（Manual upload）で 1 回同期** → 必要に応じて **B（GitHub add-on）で以降自動化**。

---

## Option A: Manual Upload（コンポーネント別）

### 手順（共通）

1. ブラウザで https://osf.io/79m5j/ を開く
2. 各 Component（`01_protocol` 等）をクリック → 左サイドバー `Files` → `OSF Storage` を展開
3. ローカルの該当ファイルを **drag & drop** でアップロード
4. 失敗時は `Upload` ボタンからファイル選択

---

### 01_protocol — プロトコル + 原稿

| File (local path) | 備考 |
|-------------------|------|
| `metaanalysis/osf_registration_draft.md` | ✅ **既 upload 済**（Phase 2）|
| `metaanalysis/meta_analysis_plan.md` | ✅ **既 upload 済** |
| `metaanalysis/literature_review.md` | ✅ **既 upload 済** |
| `metaanalysis/prospero_draft.md` | ✅ **既 upload 済**（アーカイブ） |
| `metaanalysis/deep_reading_notes.md` | 🆕 **要 upload**（30 PDFs 精読記録） |
| `metaanalysis/reference_index.md` | 🆕 **要 upload**（95 refs index）|
| `metaanalysis/paper/manuscript_preprint.docx` | 🆕 **要 upload**（Preprint 版）|
| `metaanalysis/paper/manuscript_journal.docx` | 🆕 **要 upload**（Journal 版）|
| `metaanalysis/paper/introduction.md` | 🆕 **要 upload**（Markdown draft）|
| `metaanalysis/paper/references_data.py` | 🆕 **要 upload**（37 refs data）|
| `metaanalysis/paper/build_docx.py` | 🆕 **要 upload**（生成スクリプト）|
| `metaanalysis/paper/reference_audit.md` | 🆕 **要 upload**（引用監査報告）|

**小計: 新規 8 files**

---

### 02_search — 検索戦略 + ログ

| File | 備考 |
|------|------|
| `metaanalysis/search_log.md` | 🆕 **要 upload** |
| `metaanalysis/search_results/candidate_studies.md` | 🆕 **要 upload**（28 候補）|
| `metaanalysis/search_results/pdf_download_tier1.md` | 🆕 **要 upload**（Tier 1 DL list）|

**小計: 新規 3 files**

---

### 03_screening — スクリーニング記録

**現時点で該当ファイルなし**（将来、Rayyan exports や PRISMA flow 数値を追加予定）

---

### 04_extraction — データ抽出

| File | 備考 |
|------|------|
| `metaanalysis/data_extraction.csv` | ✅ **既 upload 済**（template） |
| `metaanalysis/data_extraction_README.md` | ✅ **既 upload 済** |
| `metaanalysis/analysis/data_extraction_populated.csv` | 🆕 **要 upload**（31 studies 実データ）|
| `metaanalysis/analysis/extract_data.py` | 🆕 **要 upload**（抽出スクリプト）|

**小計: 新規 2 files**

---

### 05_risk_of_bias — JBI risk of bias 評価

**現時点で該当ファイルなし**。RoB は `data_extraction_populated.csv` の `risk_of_bias_score` 列に統合済みのため、04_extraction で代替。将来 RoB を独立 CSV として切り出す場合はここへ。

---

### 06_analysis — R/Python コード + 結果 + プロット

| File | 備考 |
|------|------|
| `metaanalysis/analysis/pool.py` | 🆕 **要 upload**（REML+HKSJ pooling）|
| `metaanalysis/analysis/sensitivity.py` | 🆕 **要 upload**（LOO+COI+β除外+low quality）|
| `metaanalysis/analysis/plots.py` | 🆕 **要 upload**（forest/funnel 生成）|
| `metaanalysis/analysis/prisma.py` | 🆕 **要 upload**（PRISMA flow 生成）|
| `metaanalysis/analysis/pooling_results.csv` | 🆕 **要 upload**（primary results）|
| `metaanalysis/analysis/moderator_results.csv` | 🆕 **要 upload**（moderator results）|
| `metaanalysis/analysis/pooling_summary.md` | 🆕 **要 upload**（統計サマリ）|
| `metaanalysis/analysis/sensitivity_results.md` | 🆕 **要 upload**（sensitivity 報告）|

**小計: 新規 8 files + figures/ subfolder**

**figures/ サブフォルダ**（`06_analysis` 内に `figures` フォルダを作成後、以下を upload）:

| File | 備考 |
|------|------|
| `figures/prisma_flow.png` | Figure 1 |
| `figures/forest_O.png` | Figure 2 |
| `figures/forest_C.png` | Figure 3 |
| `figures/forest_E.png` | Figure 4 |
| `figures/forest_A.png` | Figure 5 |
| `figures/forest_N.png` | Figure 6 |
| `figures/funnel_O.png` | Figure 7 |
| `figures/funnel_C.png` | Figure 8 |
| `figures/funnel_E.png` | Figure 9 |
| `figures/funnel_A.png` | Figure 10 |
| `figures/funnel_N.png` | Figure 11 |

**小計: 11 PNG files**

---

### 07_pdf_index — 論文 DOI インデックス

| File | 備考 |
|------|------|
| `metaanalysis/pdf_download_urls.md` | ✅ **既 upload 済** |

**小計: 新規 0 files**

---

## アップロード件数 Summary

| Component | 既 upload | 新規要 upload |
|-----------|----------|-------------|
| 01_protocol | 4 | **8** |
| 02_search | 0 | **3** |
| 03_screening | 0 | 0 |
| 04_extraction | 2 | **2** |
| 05_risk_of_bias | 0 | 0 |
| 06_analysis | 0 | **8 + 11 PNG** |
| 07_pdf_index | 1 | 0 |
| **Total** | **7** | **32 files** |

---

## Option B: GitHub Add-on（推奨、継続同期）

Manual upload の代替 / 以降の自動同期用。

### 手順

1. OSF Project `79m5j` → Settings → **Add-ons**
2. **GitHub** を探し、`Enable` をクリック
3. OSF に GitHub アクセス権を付与（初回のみ OAuth）
4. Repository 選択: `etoki/paper`、Branch: `main`
5. Save

有効化後、GitHub repo の `metaanalysis/` 配下のファイルが **OSF Project 上で自動表示**される（download も可能、OSF ファイル tree に混在）。

### 各 Component で設定可能

各 Component（01_protocol 等）についても同じ手順で GitHub add-on を有効にすると、Component 内の Files タブに GitHub ファイルが表示される。

### メリット

- **今後の commit が自動反映**（手動 re-upload 不要）
- GitHub 上で修正 → OSF 即座に反映
- **Version control 履歴**が GitHub 経由で OSF からも見える

### デメリット

- OSF Registration（スナップショット）には GitHub add-on 経由のファイルは **含まれない**（OSF Storage ネイティブのみ）
- → **登録済 Registration `e5w47` の補足資料は必ず OSF Storage に直接 upload**

**推奨**: 
1. **まず Option A で OSF Storage に直接 upload**（Registration と紐付く確実な保全）
2. **次に Option B で GitHub add-on 追加**（継続開発用）

---

## Option C: osfclient CLI（上級者向け）

`osfclient` は OSF の Python 製 CLI。Personal Access Token を事前発行すれば、スクリプト化可能。

### インストールと設定

```bash
pip install osfclient
# OSF → Settings → Personal Access Tokens → Create new
osf init   # project id と token を .osfcli.config に保存
```

### Upload 例（06_analysis へ）

```bash
cd /home/user/paper/metaanalysis/analysis
osf upload -r pool.py osfstorage/pool.py
osf upload -r figures/*.png osfstorage/figures/
```

`-r` flag で directory 再帰 upload 可能。

### Sync スクリプト例

```bash
#!/bin/bash
# sync_osf.sh — 定期実行で OSF と同期
OSF_PROJECT=79m5j  # 実際は各 component の GUID
osf upload -U analysis/*.py osfstorage/code/
osf upload -U analysis/*.csv osfstorage/data/
osf upload -U analysis/figures/*.png osfstorage/figures/
```

`-U` = update existing files（上書き）。

---

## 推奨実施順序

### 本日の作業（Option A 一括、30–45 分）

1. ブラウザで OSF を 7 タブ開く（Component ごとに 1 タブ）
2. 各タブで Files → OSF Storage を展開
3. 本ガイド **§01_protocol** の 🆕 8 files を drag & drop
4. **§02_search** の 3 files 同上
5. **§04_extraction** の 2 files
6. **§06_analysis** の 8 files + `figures` フォルダを作って 11 PNG
7. 最後に一旦全 Component を再読み込みして反映を目視確認

### 将来の継続同期（Option B、1 回のみ設定）

1. OSF Settings → Add-ons → GitHub 有効化
2. `etoki/paper` を指定

以降は `git push` すれば OSF Project でも自動で最新が見える。

---

## 後続タスク候補

- `03_screening` Component への Rayyan export 追加（実際の screening 実施後）
- `05_risk_of_bias` への JBI 詳細 CSV 追加（独立ファイルに切り出す場合）
- 各 Component の Wiki 更新（files upload に合わせて "Files section" に追記）
