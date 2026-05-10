# OSF Preprints 投稿手順 (P3 移行)

## 前提

- 投稿者は既に OSF アカウント保有(関連プロジェクト `osf.io/79m5j`, `osf.io/e5w47`, `osf.io/3y54u` など)
- 移行対象: P3 メタアナリシス
- ローカル原稿: `metaanalysis/paper/manuscript_preprint.docx` (英語)

## 手順

### 1. ログイン
- https://osf.io/preprints にアクセス
- 既存アカウント(eisuke.tokiwa@sunblaze.jp)でログイン

### 2. 新規 preprint 作成
- "Add a Preprint" をクリック
- Provider 選択: **OSF Preprints**(general)
   - サブドメインの選択肢: PsyArXiv (心理学系)、SocArXiv (社会学系)、EdArXiv (教育系) も検討可
   - **推奨: PsyArXiv または EdArXiv** (Big Five × 教育のフィット度)

### 3. メタデータ入力

| 項目 | 値 |
|---|---|
| Title | Big Five Personality Traits and Academic Achievement in Online Learning Environments: A Systematic Review and Meta-Analysis |
| Author | Eisuke Tokiwa (ORCID 0009-0009-7124-6669) |
| Affiliation | SUNBLAZE Co., Ltd., Tokyo, Japan |
| Abstract | (`metaanalysis/paper/manuscript_preprint.docx` の Abstract セクションをコピー) |
| Subjects | Education > Educational Psychology / Educational Technology |
| Keywords | Big Five, online learning, meta-analysis, academic achievement, personality |
| License | CC BY 4.0 (Research Square と同じ) |
| Conflict of interest | None to declare |
| Preprint DOI option | "Generate DOI" をチェック |

### 4. ファイルアップロード
- `manuscript_preprint.docx` または PDF版
- データファイル(可能なら separate component として `data_extraction.csv` など)

### 5. 既存 Research Square preprint との関係
- **Disclosure section:** "An earlier version of this manuscript was previously posted on Research Square (DOI 10.21203/rs.3.rs-9513298/v1, posted 2026-04-27). The Research Square version is being withdrawn."

### 6. 公開
- 投稿後、自動で DOI が発行される(数分〜数時間)
- 新DOIをメモ → `metaanalysis/conference_submissions/portfolio_status.mdstatus.md` を更新

### 7. ローカル記録更新
- `metaanalysis/paper/preprint_info.md` を更新(新DOI、新URL)
- `metaanalysis/conference_submissions/portfolio_status.mdstatus.md` を更新

## サブドメイン選択ガイド

| Subdomain | フィット度 | コメント |
|---|---|---|
| **PsyArXiv** | ★★★ | Personality Psychology の中心、Big Five研究多数 |
| **EdArXiv** | ★★★ | Educational Psychology / Online Learning の中心 |
| **SocArXiv** | ★★ | 社会学・教育社会学。投稿者は既に Simulation 論文で使用 |
| OSF Preprints (general) | ★★ | カテゴリ未定の場合の汎用 |

**推奨: PsyArXiv**(personality研究主軸)または **EdArXiv**(online learning主軸)。両方に同時投稿は不可なので一方を選ぶ。

## チェックリスト

- [ ] OSF アカウントでログイン
- [ ] サブドメイン選択(PsyArXiv 推奨)
- [ ] メタデータ入力完了
- [ ] manuscript ファイルアップロード
- [ ] Research Square 関係を disclosure 文に記載
- [ ] 新DOI取得
- [ ] `metaanalysis/conference_submissions/portfolio_status.mdstatus.md` を新DOIで更新
- [ ] `metaanalysis/paper/preprint_info.md` を更新
- [ ] **その後** Research Square 撤回手続き(`research_square_withdrawal_steps.md`)
