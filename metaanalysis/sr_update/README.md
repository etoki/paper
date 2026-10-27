# sr_update — 検索をやり直してシステマティックレビューにするための準備

2026年4月の検索は、作業環境から登録したデータベースに接続できず、ウェブ検索8回（約80件）で代用しました。このフォルダは、事前登録したデータベースで検索をやり直すための道具一式です。まだ一度も実行していません（作業環境から API に接続できないため）。

## ファイル

| ファイル | 中身 |
|---|---|
| `protocol_amendment.md` | OSF に登録する修正（Amendment 1）の草案。上が日本語の要約、下が登録する英語の本文 |
| `search_strategy.py` | 検索式。事前登録した語（`PREREGISTERED`）と追加する語（`ADDED`）を分けて持っている |
| `run_search.py` | 検索を実行し、すべての応答と記録を保存するスクリプト |
| `seeds.csv` | 引用をたどる起点。組み入れた25本と先行レビュー6本 |
| `fetch_fulltext.py` | 全文段階の PDF を、無料で公開されている版から自動で取ってくるスクリプト |
| `fulltext/pdf/` | PDF の保存先。ファイル名は `<record_id>.pdf`。著作権があるので git には入れない |

## 進め方

### 1. 接続できる環境を用意する

ローカルの Mac で実行するなら設定は不要です。

クラウド環境で実行する場合は、環境の設定の Network access → Allowed domains で次の3つを許可します（https://code.claude.com/docs/en/cloud-environments#network-access）。2026-10-27 に許可を試しましたが、そのセッションの中では 403 のままでした。
- `eutils.ncbi.nlm.nih.gov`（PubMed）
- `api.ies.ed.gov`（ERIC）
- `api.openalex.org`（OpenAlex）

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

### 4. 選別する（常盤さん）

事前登録どおり、常盤さん1人で選別します。Claude は2人目の選別者にしません（2026-10-27 決定）。

**題名・要旨の段階**:
- 常盤さんが全件を判定します。
- 1週間以上あけて10%を選び直し、一致度（κ）を出します。

題名・要旨の判定は `records_dedup.csv` の `ta_decision` 欄に `include`（全文を読む）、`maybe`（迷う。全文を読む）、`exclude` のどれかで書きます。

**全文の段階**:
- Claude が次のコマンドで PDF を取ります。`ta_decision` が `include` か `maybe` の行が対象です。

  ```
  python metaanalysis/sr_update/fetch_fulltext.py
  ```

- 取れなかったものは `fulltext/to_download_by_hand.csv` に、題名・DOI・DOI のリンクつきで並びます。常盤さんがダウンロードし、`save_as` 欄の名前（`fulltext/pdf/<record_id>.pdf`）で保存してください。
- もう一度同じコマンドを実行すると、置いた PDF は飛ばし、まだ無いものだけがリストに残ります。
- 判定は常盤さんが行い、20%を選び直します。

取得の経路は、無料で公開されている版だけです（OpenAlex の OA 情報、PubMed Central の公開データ、ERIC、OSTI の順）。出版社に断られたら、ブラウザを装うなどの回避はせず、手作業のリストに回します。Unpaywall は使いません（OpenAlex が同じデータを持っており、追加でメールアドレスを送らずに済むため）。

2026-10-27 に組み入れ済みの25本と先行レビュー6本で試したところ、31本中15本が自動で取れ、16本が手作業のリストに回りました。手作業になったのは、Elsevier・Springer・Wiley・APA の有料論文と DOI のない論文です。新しく見つかる論文でも、半分くらいは手作業になると見込んでください。

### 5. データを抽出し、分析をやり直す

- 新しい研究は `analysis/data_extraction_populated.csv` に足します。
- `paper_v4/build_docx_v4.py` を実行すれば、本文と表の数字は自動で入れ替わります。

## 件数の見込み

実行するまでわかりません。データベースで数百〜千件程度、引用をたどる分で数千件になる可能性があります。件数が多すぎる場合は、検索語を絞る前に相談します。絞ると修正の登録内容と違ってしまうためです。

## 見つけたデータの誤り（直した）

`analysis/data_extraction_populated.csv` の A-12（Baruth & Cohen, 2023）の DOI が、A-07（Cohen & Baruth, 2017）の DOI になっていました。参考文献リストにある正しい DOI（10.1007/s10639-022-11199-x）に直しました。分析には使っていない列なので、数字は変わりません。
