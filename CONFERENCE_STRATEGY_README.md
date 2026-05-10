# 国際学会投稿戦略ワークスペース

**ベース日付:** 2026-05-09 / **ゴール:** 2026年中に2件採択(慶應SFC論文博士要件)
**Branch:** `claude/conference-submission-strategy-ZDn3N`

## ディレクトリ構成

```
.
├── conferences/         # 10学会の JSON ファクト集 + README
├── papers/              # 既存5論文のステータストラッカー
│   ├── P1_online_learning/  # Frontiers Psychology 2025
│   ├── P2_clustering/       # IEEE Xplore HEXACO Cluster
│   ├── (P3 status moved to metaanalysis/conference_submissions/portfolio_status.md)
│   ├── P4_harassment/       # preprint
│   └── P5_simulation/       # OSF + SocArXiv preprint ✅
├── dashboard/           # 締切カウントダウン + TODO
├── templates/           # メールテンプレート 5本
├── preprint_migration/  # OSF / arXiv 移行手順 + RS 撤回
└── decision_matrix/     # 学会スコアリング + 論文割当
```

## 🚨 本日(2026-05-09)の最優先アクション

1. **`templates/ecel_extension_request.md` をレビューして本日中に送信**(ECEL 5/21 締切に間に合わせるため)
2. **意思決定: P3 preprint migration**(オプションA OSF / B arXiv / C 維持)
3. **P5 simulation の OSF + SocArXiv DOI を `papers/P5_simulation/status.md` に記録**(IEEE TALE 適合確認のため)

## クイックリンク

- [ダッシュボード](dashboard/index.md) — 全体状況
- [Decision Matrix](decision_matrix/scoring.md) — 学会スコアリング
- [論文 × 学会割当](decision_matrix/paper_to_venue_mapping.md)
- [Preprint Migration ガイド](preprint_migration/README.md)
- [学会一覧](conferences/README.md)

## 推奨戦略(2 件確保 + 保険)

```
第一候補(必須2件):
  ① ECEL 2026 (5/21締切) ← P3 メタアナリシス
  ② ICEEL 2026 東京 (6/30締切) ← P1派生 または P5

攻め追加(SFC評価最大化):
  ③ IEEE TALE 2026 (6/30締切) ← P5 (OSF preprint 適合)

保険:
  ④ ICERI 2026 (7/9締切) ← virtual + WoS CPCI
  ⑤ AAOU 2026 / AACE E-Learn 2026 (CFP公開待ち)

DROP:
  ❌ ASCILITE (live必須)
  ❌ EDULEARN (締切経過)
  ❌ ICTLE (WORLDCTEと重複)
```

## 📞 投稿者(Eisuke Tokiwa)に確認が必要な意思決定事項

このワークスペースで自動進行できる作業は完了しているが、以下は **本人の意思決定** を待つ:

1. **P3 preprint migration** をするか / どこへ
2. **IEEE TALE への P5 投稿** に進むか(virtual確認後)
3. **ICEEL での発表方法**(東京現地 / virtual 確認)
4. **EXCELLENCE Awards に応募するか**(ECEL 5/14締切)
5. **メール送信タイミング**(Claude Code は自動送信しない)

意思決定後、`dashboard/index.md` の todo 項目を更新する。
