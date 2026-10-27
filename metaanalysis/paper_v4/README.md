# paper_v4 — 本文の修正版

Heliyon で不採択になった v3（`paper_v3_scoping/`）を出発点に、本文を書き直した版です。v3 は投稿した版の記録としてそのまま残しています。

## ビルド

```
python metaanalysis/paper_v4/build_docx_v4.py
```

出力: `manuscript_journal_v4.docx`、`manuscript_preprint_v4.docx`、`split/`（タイトル・本文・表・図の分割版）。

## 数字の出どころ

本文と表の数字は、ビルドのたびに `values.py` が分析パイプライン（`analysis/pool.py`、`analysis/sensitivity.py`、抽出データ）から計算して埋め込みます。手で書き写した数字はありません。分析を直せば、本文と表も自動で揃います。証拠評価（表5）の判定も、Methods に書いた規則から計算しています。

## v3 からの主な変更

- 分析の訂正（2026-10-27）: 正確な REML の τ²、Knapp–Hartung のモデレーター検定（Holm 補正つき）、Peterson–Brown の正しい換算式、標準的な trim-and-fill。事前登録していたのに実施していなかった感度分析（N < 50 の除外、DerSimonian–Laird）を実施し、統合から外した A-37 を r = 0 で戻す分析を追加
- A-37 Zheng & Zheng (2023) を統合から除外（2026-10-27 の判断。コロナ前後は対面・コロナ中のみオンラインの学生を合わせたデータで、相関も |r| < .10 としか報告されていないため）。スコーピングの地図には残し、本文の逸脱(7)に「結果を見た後の判断」と明記。A-37 を戻した感度分析も表4に載せている。除外は `analysis/pool.py` の `POOL_EXCLUSIONS` で管理
- Methods を、実際に行ったことだけに書き直し（κ による信頼性確認、Zotero、著者への問い合わせ、RVE、p-curve、Peters 検定などは「実施しなかった」と逸脱に記載）
- PRISMA 図を、記録から確認できる件数で作り直し（全文段階より前は「記録なし」と明記）
- データ公開先を GitHub に変更（OSF は事前登録のみ）、生成 AI 利用の申告を追加
- Heliyon の不採択理由にあった文章の不備（"offers , to"、"a exploratory"、モデレーター数、A-27 の注記、図番号の参照）を修正
- 参考文献を 5 本追加（Higgins 2009、Holm 1979、Knapp & Hartung 2003、Sterne 2011、Viechtbauer 2015）、引用されなくなった 4 本を削除し、著者名順に並べ直し

## 投稿前に必要なこと

- 追加した参考文献 5 本の DOI をブラウザで確認する（作業環境から doi.org に接続できなかった）
- 生成 AI 利用の申告文（Declarations）と、Methods の「選別に LLM の支援を受けた」という記述が実際と合っているか確認する
- PRISMA-ScR チェックリスト（v3 の `build_prisma_scr_checklist.py`）を v4 の節構成に合わせて作り直す
