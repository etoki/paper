# arXiv 投稿手順 (P3 移行 — 代替案)

## 前提

- arXiv は **endorsement 制度** あり。初回投稿は既存研究者からの推薦が必要。
- IEEE は arXiv を明示的に許容しているため、IEEE TALE にとっては最も安全。

## 手順

### 1. アカウント作成
- https://arxiv.org/user/register
- 投稿者既存 ORCID と紐付け

### 2. Endorsement 申請

**該当カテゴリ候補:**
- `cs.CY` (Computers and Society) — オンライン学習・教育テクノロジー
- `stat.AP` (Statistics - Applications) — メタアナリシスの統計的観点

**Endorser を探す:**
- Frontiers Psychology 共著者・査読者経由
- IEEE Xplore 既論文(P2) の引用関係から
- 慶應SFC教員(指導教員候補)
- 過去に arXiv 投稿経験のある日本の心理学研究者

**Endorsement リクエスト:**
- arXiv login 画面で endorsement code を取得
- 該当研究者にメールで依頼:
  > "I would like to submit my first paper to arXiv (category: cs.CY). Could you provide an endorsement? My endorsement code is XXXX. The paper is a meta-analysis of Big Five personality traits and academic achievement in online learning environments (already preprinted on Research Square; migrating to arXiv for IEEE conference compliance)."

### 3. 投稿

| 項目 | 値 |
|---|---|
| Primary category | cs.CY |
| Secondary category | stat.AP |
| Title | Big Five Personality Traits and Academic Achievement in Online Learning Environments: A Systematic Review and Meta-Analysis |
| Author | Eisuke Tokiwa |
| Affiliation | SUNBLAZE Co., Ltd. |
| Abstract | (manuscript_preprint.docx より) |
| Comments | "An earlier version was posted on Research Square (DOI 10.21203/rs.3.rs-9513298/v1, 2026-04-27); migrating to arXiv. Withdrawal of Research Square version is in process." |
| License | arXiv non-exclusive license (推奨) |

### 4. 形式
- arXiv は **PDF 直接アップロード可** だが、LaTeX ソースが推奨
- 既存 docx を Pandoc で .tex に変換 → 再フォーマット必要
   ```bash
   pandoc manuscript_preprint.docx -o manuscript.tex
   ```
- アップロード後、processing に数時間〜1日

### 5. DOI
- arXiv は arXiv ID(例: 2605.12345)を発行
- DataCite経由で DOI も付与可能(オプション)

## arXiv vs OSF 比較

| 項目 | arXiv | OSF Preprints |
|---|---|---|
| 即時投稿 | ❌ endorsement待ち | ✅ 即時 |
| IEEE 明示許容 | ✅ | ⚠️ PSPB approved 確認推奨 |
| 投稿難易度 | 中(LaTeX 推奨) | 低(docx/pdf 直接) |
| 学術慣行(教育・心理学系) | 限定的 | 主流 |
| 移行所要時間 | 1〜2週間 | 1日以内 |

## 推奨: OSF を選択

教育心理学分野では **OSF (PsyArXiv / EdArXiv)** が arXiv より広く使われている。endorsement待ちのリスクと比較して OSF が現実的。

ただし IEEE TALE への投稿確実性を最優先するなら arXiv も検討余地あり。

## チェックリスト(arXivを選ぶ場合)

- [ ] arXiv アカウント登録
- [ ] Endorser 候補を 3 名以上リスト化
- [ ] Endorsement リクエスト送信
- [ ] Endorsement 取得
- [ ] LaTeX 化(または PDF 単体投稿)
- [ ] メタデータ入力 + アップロード
- [ ] arXiv ID 取得
- [ ] DataCite DOI 取得(オプション)
- [ ] Research Square 撤回手続きへ
