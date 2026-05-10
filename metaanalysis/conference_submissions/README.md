# 国際学会投稿戦略ワークスペース(P3 メタ分析)

**ベース日付**: 2026-05-09 / **ゴール**: 2026 年中に 2 件採択(慶應 SFC 論文博士要件)
**スコープ**: メタ分析(P3)を中心とした国際学会投稿。他論文(P1/P2/P4/P5)のステータスは参考情報として `paper_portfolio/` に同居。

## ディレクトリ構成

```
metaanalysis/conference_submissions/
├── README.md                  ← 本ファイル
├── ecel/                      ECEL 2026 投稿(5/21 deadline)
├── iceel/                     ICEEL 2026 投稿(6/30 deadline、東京開催)
├── iceri/                     ICERI 2026 投稿(7/9 deadline、Seville、virtual 可)
├── inputs/                    studies.csv + 抽出スクリプト(3 venue 共有)
├── preprint_audit.md          venue 別 disclosure scope 監査
├── portfolio_status.md        P3 status (ex papers/P3_metaanalysis/status.md)
├── conferences/               10 venue の JSON ファクト集
├── decision_matrix/           paper × venue スコアリング + 投稿割当
├── dashboard/                 締切カウントダウン + TODO
├── templates/                 メールテンプレート(ECEL extension / ICEEL inquiry / preprint disclosure)
├── preprint_migration/        ※ DEAD: 「Research Square 維持」決定により移行は実施しない
└── paper_portfolio/           参考: P1/P2/P4/P5 status トラッカー
```

## 現状(2026-05-10 時点)

- **ECEL 2026**(5/21 deadline) — full paper draft + cover letter ready。Frontiers Joe Fares 宛 follow-up 送信済 → 返信待ち。
- **ICEEL 2026**(6/30 deadline) — full paper draft + cover letter ready。東京開催 + 在住者なので travel cost 0。
- **ICERI 2026**(7/9 deadline) — full paper draft + cover letter ready。virtual 申請予定。
- **IEEE TALE 2026** — **DROP 確定**。`research_square_allowed: false` policy + 「Research Square 維持」方針で投稿経路なし。
- **Drop 確定**: ASCILITE(live 必須), EDULEARN(締切経過), ICTLE(WORLDCTE と重複), 移行系全般。

## 推奨戦略(2 件確保 + 保険)

```
第一候補(必須 2 件):
  ① ECEL 2026 (5/21 締切) ← P3 メタアナリシス、modality × trait interaction
  ② ICEEL 2026 東京 (6/30 締切) ← P3 派生、Hofstede within-Asia + Japan focus

保険:
  ③ ICERI 2026 (7/9 締切) ← P3 派生、education level × discipline cross-tab、virtual

DROP:
  ❌ IEEE TALE (Research Square 不可)
  ❌ ASCILITE (live 必須)
  ❌ EDULEARN (締切経過)
  ❌ ICTLE (WORLDCTE と重複)
```

## クイックリンク

- [ダッシュボード](dashboard/index.md) — 全体状況
- [Decision Matrix](decision_matrix/scoring.md) — 学会スコアリング
- [論文 × 学会割当](decision_matrix/paper_to_venue_mapping.md)
- [学会 venue facts](conferences/README.md) — 10 venue JSON ファクト集
- [P3 portfolio status](portfolio_status.md)
- [Preprint disclosure テンプレ](templates/preprint_disclosure_template.md)

## 投稿者意思決定事項

1. **ECEL Excellence Awards 応募**(5/14 締切)— 応募する/しない
2. **ICEEL での発表方法**(東京現地 — 既定で in-person)
3. **ICERI virtual 申請**(cover letter で要請済)
4. **メール送信タイミング**(Claude Code は自動送信しない)

## Out of scope(本ワークスペースでは扱わない)

- 他論文(P1/P2/P4/P5)のハルシネーションチェックや学会選定 — 別チャットで対応
- Research Square から OSF/arXiv/TechRxiv への移行 — 「Research Square 維持」方針で永続的に DROP
- IEEE TALE 投稿そのもの — 上記理由で DROP
