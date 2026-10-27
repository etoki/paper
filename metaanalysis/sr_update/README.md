# sr_update — 検索をやり直してシステマティックレビューにするための準備

2026年4月の検索は、作業環境から登録したデータベースに接続できず、ウェブ検索8回（約80件）で代用しました。このフォルダは、事前登録したデータベースで検索をやり直すための道具一式です。まだ一度も実行していません（作業環境から API に接続できないため）。

## ファイル

| ファイル | 中身 |
|---|---|
| `protocol_amendment.md` | OSF に登録する修正（Amendment 1）の草案。上が日本語の要約、下が登録する英語の本文 |
| `search_strategy.py` | 検索式。事前登録した語（`PREREGISTERED`）と追加する語（`ADDED`）を分けて持っている |
| `run_search.py` | 検索を実行し、すべての応答と記録を保存するスクリプト |
| `seeds.csv` | 引用をたどる起点。組み入れた25本と先行レビュー6本 |

## 進め方

### 1. 環境のネットワーク設定で、次の3つを許可する（常盤さん）

- `eutils.ncbi.nlm.nih.gov`（PubMed）
- `api.ies.ed.gov`（ERIC）
- `api.openalex.org`（OpenAlex）

設定場所は、クラウド環境の設定の Network access → Allowed domains です（https://code.claude.com/docs/en/cloud-environments#network-access）。

### 2. OSF に修正を登録する（常盤さん）

- 必ず検索の前に登録します。後だと「結果を見てから決めた」ことになるためです。
- `protocol_amendment.md` の「登録前に確認してほしいこと」の3点を決めてから登録してください。
- 登録するのは英語の部分だけです。
- 事前登録の §4 に「方針変更は OSF に新しい version を登録する」と書いてあるので、登録ページの Update から新しい version として登録します。それができない場合は、OSF プロジェクトにファイルとして置いても日付は残ります。

### 3. 検索を実行する（Claude）

```
python metaanalysis/sr_update/run_search.py --check   # 接続確認
python metaanalysis/sr_update/run_search.py           # 本番
```

保存されるもの:
- `records_dedup.csv`: 重複を除いた全件（選別の欄つき）
- `search_log.json`: 検索日、検索式、件数（PRISMA 図の数字）
- `raw/`: すべての応答。大きいので git には入れず、zip にして OSF に置く

Google Scholar の上位200件は、ブラウザでの手作業が必要です（やる場合だけ）。

### 4. 選別する（常盤さん、Claude）

**題名・要旨の段階**:
- 常盤さんと Claude が別々に判定し、食い違いは常盤さんが決めます。
- 1週間以上あけて10%を選び直し、一致度（κ）を出します。これは事前登録どおりの手順です。

**全文の段階**:
- Claude が PDF の DOI リストを作り、常盤さんがダウンロードします。
- 判定は常盤さんが行い、20%を選び直します。

### 5. データを抽出し、分析をやり直す

- 新しい研究は `analysis/data_extraction_populated.csv` に足します。
- `paper_v4/build_docx_v4.py` を実行すれば、本文と表の数字は自動で入れ替わります。

## 件数の見込み

実行するまでわかりません。データベースで数百〜千件程度、引用をたどる分で数千件になる可能性があります。件数が多すぎる場合は、検索語を絞る前に相談します。絞ると修正の登録内容と違ってしまうためです。

## 見つけたデータの誤り（直した）

`analysis/data_extraction_populated.csv` の A-12（Baruth & Cohen, 2023）の DOI が、A-07（Cohen & Baruth, 2017）の DOI になっていました。参考文献リストにある正しい DOI（10.1007/s10639-022-11199-x）に直しました。分析には使っていない列なので、数字は変わりません。
