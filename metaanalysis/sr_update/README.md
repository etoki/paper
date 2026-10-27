# sr_update — 検索をやり直してシステマティックレビューにするための準備

2026年4月の検索は、作業環境から登録したデータベースに接続できず、ウェブ検索8回（約80件）で代用しました。このフォルダは、事前登録したデータベースで検索をやり直すための道具一式です。まだ一度も実行していません（作業環境から API に接続できないため）。

## ファイル

| ファイル | 中身 |
|---|---|
| `protocol_amendment.md` | OSF に登録する修正（Amendment 1）の草案。上が日本語の要約、下が登録する英語の本文 |
| `search_strategy.py` | 検索式。事前登録した語（`PREREGISTERED`）と追加する語（`ADDED`）を分けて持っている |
| `run_search.py` | 検索を実行し、すべての応答と記録を保存するスクリプト |
| `records_index.csv` | 選別の対象の一覧（書誌情報だけ。要旨つきの `records_dedup.csv` は手元と OSF のみ） |
| `seeds.csv` | 引用をたどる起点。組み入れた25本と先行レビュー6本 |
| `scholar/pop_export.csv` | Publish or Perish で書き出した Google Scholar の結果（常盤さんが保存） |
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

**済み（2026-10-27）**: Amendment 1 を OSF の新しい版として登録した（12:29 UTC 提出・承認）。変えたのは Data collection procedures、Starting and stopping rules、Missing data の3欄で、どれも先頭に修正を足し、元の文面は残した。入れた文面は `osf_amendment1_fields.md`。以下は登録前の手順の記録。


- 必ず検索の前に登録します。後だと「結果を見てから決めた」ことになるためです。
- `protocol_amendment.md` の「登録前に確認してほしいこと」の3点を決めてから登録してください。
- 登録するのは英語の部分だけです。
- 事前登録の §4 に「方針変更は OSF に新しい version を登録する」と書いてあるので、登録ページの Update から新しい version として登録します。それができない場合は、OSF プロジェクトにファイルとして置いても日付は残ります。

### 3. 検索を実行する（Claude）

**OpenAlex の API キー（常盤さん）**: OpenAlex はキーなしだと、同じ回線の全員で共有する1日の無料枠しか使えません。2026-10-27 の1回目の本番実行は、PubMed（33件）と ERIC（176件）を取ったところで OpenAlex の枠が尽きて止まりました（HTTP 429）。どの情報源も同じ日に検索する約束なので、この回の結果は使わず、キーを用意してから全部を同じ日に取り直します。

1. https://openalex.org でアカウントを作り（無料）、API キーを発行する。
2. キーだけを1行で書いたファイルを `metaanalysis/sr_update/.openalex_key` として保存する。このファイルは git に入りません。Claude はファイルの中身を見ません。
3. Claude が `--check` で接続と残り枠を確かめてから、本番を実行します。


```
python metaanalysis/sr_update/run_search.py --check   # 接続確認
python metaanalysis/sr_update/run_search.py           # 本番
```

保存されるもの:
- `records_dedup.csv`: 重複を除いた全件（選別の欄つき）
- `search_log.json`: 検索日、検索式、件数（PRISMA 図の数字）
- `raw/`: すべての応答。大きいので git には入れず、zip にして OSF に置く

**Google Scholar（常盤さん。API の検索と同じ日に）**: Google Scholar には API がないので、Publish or Perish で行います。

1. Publish or Perish で「Google Scholar」の検索を開き、Keywords 欄に次の式をそのまま入れる（`search_strategy.py` の `GOOGLE_SCHOLAR_QUERY`）。

   ```
   ("Big Five" OR "five factor" OR conscientiousness OR personality) ("online learning" OR "e-learning" OR "distance learning" OR MOOC) ("academic achievement" OR "academic performance" OR grades OR GPA)
   ```

2. 取得件数の上限（Maximum number of results）を 200 にして検索する。年の範囲などほかの条件は入れない。
3. 結果を CSV で書き出し（Save Results → Results as CSV）、`metaanalysis/sr_update/scholar/pop_export.csv` に保存する。
4. Claude が次を実行して、ほかの情報源と合わせて重複を除き直す。記録番号が振り直されるので、選別を始める前に行う。

   ```
   python metaanalysis/sr_update/run_search.py --merge
   ```

**実行結果（2026-10-27、すべて同じ日）**

| 情報源 | 件数 | 登録した語だけの件数 |
|---|---|---|
| PubMed | 33 | 12 |
| ERIC | 176 | 72 |
| OpenAlex | 5,601 | 3,127 |
| 引用追跡（30起点） | 3,818 | − |
| Google Scholar（Publish or Perish、上位200件） | 200 | − |
| 合計 | 9,831 | |
| 重複除去後（選別の対象） | 8,559 → **8,670** | |

- DOI のない起点のうち、A-02 と A-21 は OpenAlex の ID、A-03 は見つかった DOI で引用追跡に加えた（`seeds.csv`）。A-19（修士論文）は OpenAlex に無いので、手作業で引用をたどる。
- API の検索は3回実行した（すべて同日）。1回目は OpenAlex の共有枠切れで止まり、2回目は A-02・A-03・A-21 を起点に加える前だった。上の数字は3回目。
- Google Scholar は常盤さんが Publish or Perish で3回試した（22:12 に16件、22:23 に40件、22:43 に200件。前の2回は Google が途中で返さなくなった）。合流したのは3回目の200件で、1回目のファイルは `scholar/pop_export_run1_16records.csv` として残した（2回目は上書きされて残っていない）。設定（Keywords 欄、特許を除く、CITATION records を含む、年の制限なし）と試行の記録は `search_strategy.py` と `search_log.json` にある。200件のうち111件は、ほかの情報源には無かった。

**検索がどれだけ拾えたかの確認**: v4 で扱った31本のうち28本が見つかった。データベースと Google Scholar の検索だけで見つかったのは20本で、残りは引用追跡で見つかった。統合した9本では8本（検索だけでは7本）。見つからなかったのは、対面授業のみで除外した A-09 と A-10、統合した A-15 Elvers (2003) の3本。A-15 は検索式にも、ほかの起点との引用関係にも、Google Scholar の上位200件にも掛からなかった。v4 の37本は修正どおり元の判定を引き継ぐので、A-15 が統合から落ちることはない。データベース検索だけだと拾える割合が低く、引用追跡が欠かせないことを論文に書く。

注: 2回目の実行のコミットメッセージ（261da4c）に「31本はすべて見つかった」とあるのは誤り。照合のスクリプトが題名の空の記録と突き合わせていたためで、正しくは上のとおり。

**記録の置き場所**: リポジトリは公開なので、出版社に著作権がある要旨は git に入れない。git に入れるのは `search_log.json`、`seeds.csv`、`records_index.csv`（選別の対象 8,670 件の書誌情報だけ。要旨なし）。要旨つきの `records_dedup.csv`（選別に使う）、情報源ごとの `records_*.csv`、Publish or Perish の書き出し `scholar/`、API の応答 `raw/`（約170MB）は手元に置き、zip にして OSF に置く。

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
