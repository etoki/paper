# ECEL 2026 EL-095 正誤表

公開済みの ECEL 論文 EL-095 の統計の誤りを訂正するための一式です。

| ファイル | 内容 |
|---|---|
| `recompute_moderator_tests.py` | 公開版の数値を読み込み、正しい方法で計算し直して比較する |
| `results/summary.md` | 公開版と訂正後の全数値の比較（上のスクリプトの出力） |
| `results/*.csv` | 同じ内容の CSV |
| `correction_EL-095.md` | 正誤表本文（英語・原稿） |
| `build_correction_docx.py` | 上の Markdown から Word 版を作る |
| `correction_EL-095.docx` | 主催者に送る正誤表（Word） |
| `correction_EL-095_ja.md` | 正誤表の日本語参考訳（送信しない） |
| `email_to_acpil.md` | 主催者宛てメール（英語＋日本語訳）と送信前チェック |

再実行:

```
python metaanalysis/conference_submissions/ecel/erratum/recompute_moderator_tests.py
python metaanalysis/conference_submissions/ecel/erratum/build_correction_docx.py
```

PyMARE（`pip install pymare`）があれば、REML の τ² と Knapp–Hartung の p 値を PyMARE と自動で照合します。
