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

判定するのは常盤さん1人です。Claude は2人目の選別者ではなく、全件に「こう判定するのが筋だ」という案を出すだけです（2026-10-27 決定）。常盤さんが全件を最終確認して初めて判定になります。

**題名・要旨の段階（2026-10-28 に案を作成済み）**

手順:

1. 条件と判定の規則を `screening/criteria.md` に書く。版番号をつけ、変えたら番号を上げる。いまは version 2。
2. 試験をする。`make_batches.py --pilot` で、v4 の既知28本と無作為72件を混ぜた100件を作り、案を出させ、`merge_suggestions.py --pilot` で既知28本が残るか見る。version 1 では28本中26本しか残らなかった（授業形態が要旨に書かれていない研究を落としていた）。規則4を直して version 2 とし、28本すべてが残った。
3. 全件を分ける。`make_batches.py` が 8,670 件を200件ずつ44束にする。並び順は、人格の語とオンラインの語の両方を含むもの、どちらかを含むもの、どちらも含まないもの。
4. 束ごとに、同じ指示文で Claude（Opus 5）に案を出させる。見せるのは題名・要旨・年・掲載誌だけ。出力は `screening/suggestions/batch_NNN.jsonl`（record_id、suggestion、reason、note）。要旨を含まないので git に入れる。
   束001は最初に別のモデル（Opus 5.5）で走らせたため、やり直しました。捨てた分は `screening/superseded/batch_001_opus-5-5.jsonl` に残してあり、使っていません。
5. `merge_suggestions.py` で検証する。全件に案が1つずつあるか、値と除外コードが許したものだけか、を確かめる。通ると `screening/screening_ta.csv`（全件）と `screening/screening_ta_shortlist.csv`（案が include か maybe のものだけ）ができる。

案の内訳（8,670 件）:

| 案 | 件数 |
|---|---|
| include | 161 |
| maybe | 681 |
| exclude | 7,828 |

除外コードの内訳: E1（Big Five の尺度なし）4,679、E5（量的な一次研究でない）1,899、E3（対象が学生でない）522、E7（報告の種類が不適）233、E2（対面のみ、または学習の研究でない）204、E4（該当する結果変数なし）198、E6（英語でない）93。

既知の28本はすべて include か maybe に残りました（11本が maybe。授業形態や結果変数が要旨から読み取れないもの）。

**常盤さんがやること**: まず `screening/screening_ta_shortlist.csv`（案が include か maybe の842件、約1MB）を開き、`et_decision` 欄に `include` / `maybe` / `exclude` を入れます。ここで全文を読む対象が決まります。除外の7,828件は `screening/screening_ta.csv`（全8,670件、約21MB）に入っています。どちらも `claude_suggestion` がそのまま妥当なら同じ値を書けば済みます。変える場合は `et_reason`（除外なら E1〜E7）と `et_note` に理由を書いてください。どちらのファイルも要旨を含むので git に入れません。

**再現性の確認**: 常盤さんの判定が終わったら、1週間以上あけて10%を選び直し、一致度（κ）を出します。これは常盤さんの判定どうしの一致度で、Claude の案とは比べません。

**英語以外の93件**: 登録どおり除外します（E6）。中身が条件に合うものもありますが（例: R05763、スペイン語、遠隔の心理学専攻の学生）、件数が少ないため登録を変えずに除外で通します（2026-10-28 決定）。

**全文の段階**:
全文は842件すべてを読みます（2026-10-28 決定）。maybe を全文で確認する方針で、題名・要旨の段階では絞りません。

- Claude が次のコマンドで PDF を取ります。

  対象は `screening/screening_ta.csv` で `et_decision` が `include` か `maybe` の行です。`et_decision` が空の行は `claude_suggestion` を使うので、常盤さんの記入を待たずに取得を始められます。記入済みの行では常盤さんの判定が優先されます。

  ```
  python metaanalysis/sr_update/fetch_fulltext.py
  ```

- 取れなかったものは `fulltext/to_download_by_hand.csv` に並びます。出版社ごと（件数の多い順）、その中で案（`decision`）は `include` が先です。同じサイトの中で続けて作業できるようにしてあります。`link` 欄を開き、`save_as` 欄の名前（`fulltext/pdf/<record_id>.pdf`）で保存してください。DOI がない記録では `link` が題名での検索になります。

  2026-10-28 時点の内訳は、Elsevier 152件、DOI なし146件、Springer 89件、Taylor and Francis 43件、SAGE 24件、APA 24件、Wiley 40件、以下は10件台以下です。掲載誌は340種あり、253種は1件だけです。
- もう一度同じコマンドを実行すると、置いた PDF は飛ばし、まだ無いものだけがリストに残ります。
- 判定は常盤さんが行い、20%を選び直します。

取得の経路は、無料で公開されている版だけです（OpenAlex の OA 情報、PubMed Central の公開データ、ERIC、OSTI の順）。出版社に断られたら、ブラウザを装うなどの回避はせず、手作業のリストに回します。Unpaywall は使いません（OpenAlex が同じデータを持っており、追加でメールアドレスを送らずに済むため）。

**2026-10-28 の実行結果**: 842件のうち163件が自動で取れ、679件が手作業に回りました。

| 経路 | 件数 |
|---|---|
| OpenAlex の OA 情報 | 131 |
| PubMed Central | 28 |
| ERIC | 4 |
| 手作業 | 679 |

手作業の679件のうち、112件が案 `include`、567件が案 `maybe` です。146件は DOI がなく、題名で探す必要があります。取れなかった理由は、無料で公開された版が見つからないもの597件、出版社に 403 で断られたもの47件、PDF ではなく案内ページが返ったもの19件、残りが ERIC の 404 と通信エラーです。通信エラーの4件（R02008、R06264、R07167、R08437）は時間をおいて再試行しましたが、やはり取れませんでした。

2026-10-27 に組み入れ済みの25本と先行レビュー6本で試したときは、31本中15本が自動で取れました。本番では自動で取れた割合がこれより低く（19%）、無料で読める版がない論文が大半です。

### 5. データを抽出し、分析をやり直す

- 新しい研究は `analysis/data_extraction_populated.csv` に足します。
- `paper_v4/build_docx_v4.py` を実行すれば、本文と表の数字は自動で入れ替わります。

## 件数の見込み

実行するまでわかりません。データベースで数百〜千件程度、引用をたどる分で数千件になる可能性があります。件数が多すぎる場合は、検索語を絞る前に相談します。絞ると修正の登録内容と違ってしまうためです。

## 見つけたデータの誤り（直した）

`analysis/data_extraction_populated.csv` の A-12（Baruth & Cohen, 2023）の DOI が、A-07（Cohen & Baruth, 2017）の DOI になっていました。参考文献リストにある正しい DOI（10.1007/s10639-022-11199-x）に直しました。分析には使っていない列なので、数字は変わりません。
