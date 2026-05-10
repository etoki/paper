# Research Square 撤回手順 (P3)

## 重要原則

⚠️ **必ず先に新しい preprint server に移行完了 + 新 DOI 取得してから Research Square 撤回を申請する。**

## 撤回の現実的な仕組み

Research Square は「完全削除」ではなく **withdrawn notice の付与** で対応するのが通例:
- preprint ページは残るが、目立つ "WITHDRAWN" バナーが付く
- 全文 PDF はダウンロード不可になる
- 引用済みの DOI は無効にならない(リンクは生きる)
- 検索結果には withdrawn として表示される

これは preprint 業界の標準慣行であり、問題なし。

## 手順

### 1. 事前準備(撤回申請前に必ず完了させる)

- [ ] OSF または arXiv に preprint アップロード完了
- [ ] 新DOI取得済(URLとしてアクセス可能であることを確認)
- [ ] 新DOIを `metaanalysis/conference_submissions/portfolio_status.mdstatus.md` に記録
- [ ] `metaanalysis/paper/preprint_info.md` 更新

### 2. Research Square ダッシュボード確認

- https://www.researchsquare.com にログイン
- "My Manuscripts" で対象 preprint を探す
- 通常、author dashboard から直接 withdraw できる場合と、support 経由が必要な場合がある

### 3. 直接撤回が可能な場合

- "Withdraw" ボタンクリック
- 理由欄に: "Migrated to OSF Preprints (DOI: [NEW_DOI]) for compliance with target conference's preprint policy."
- 確認ダイアログで承認

### 4. Support 経由が必要な場合

- `metaanalysis/conference_submissions/templates/research_square_withdrawal.md` のメールを Research Square Support に送信
- 通常 1〜2週間で処理完了
- 確認メールが届くまで他の作業を進める

### 5. 撤回完了確認

- Research Square preprint URL にアクセス → "WITHDRAWN" 表示確認
- DOI Resolver (https://doi.org/10.21203/rs.3.rs-9513298/v1) で landing page が withdrawn 表示

### 6. 各種記録更新

- [ ] `metaanalysis/paper/preprint_info.md` を新 preprint 情報で完全置換
- [ ] `metaanalysis/paper/preprint_info.md` の旧記録を historical section として保存
- [ ] `metaanalysis/conference_submissions/dashboard/index.md` の Preprint Migration ステータスを完了に更新

## 撤回しない選択肢(オプションC を選んだ場合)

Research Square を維持する場合、撤回不要。ただし:
- IEEE TALE への P3 投稿は不可(P5 など他論文を回す)
- ECEL/ICERI/ICEEL等への投稿時は cover letter で disclosure
- preprint disclosure テンプレート Version B を使用

## 注意

- 撤回申請メール送信は **投稿者本人** が実施
- このスクリプト/Claude Code が自動実行することはない
