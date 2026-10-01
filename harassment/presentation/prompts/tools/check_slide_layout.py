#!/usr/bin/env python3
"""スライドのレイアウト崩れを、PDF書き出しから機械的に検出する。

Googleスライドの API が返す座標・行高は、実際のレンダリングと一致しない
（セルが折り返すと表は API の行高より高くなる／テキストボックスは
はみ出しても切り詰められない）。そのため検証は必ず PDF を見て行う。

PDF の作り方（Drive コネクタ）:
    download_file_content(fileId=<デッキID>, exportMimeType="application/pdf")
    → 返ってきた JSON の "content"（base64）をデコードして .pdf に保存
      （--from-export を使えばこの JSON を直接渡せる）

使い方:
    python3 check_slide_layout.py deck.pdf
    python3 check_slide_layout.py export.json --from-export --pdf-out deck.pdf
    python3 check_slide_layout.py deck.pdf --footer-top 5.42 --render-dir out/

必要: pip install pymupdf
"""
import argparse
import base64
import json
import sys

try:
    import pymupdf
except ImportError:
    sys.exit("pymupdf が必要です:  pip install pymupdf")


def load_pdf(path, from_export, pdf_out):
    if not from_export:
        return path
    raw = json.load(open(path, encoding="utf-8"))
    while isinstance(raw, dict) and "content" in raw and not isinstance(raw["content"], str):
        raw = raw["content"]
    b64 = raw["content"] if isinstance(raw, dict) else raw
    open(pdf_out, "wb").write(base64.b64decode(b64))
    return pdf_out


def blocks(page, page_w_in, page_h_in, ignore):
    out = []
    for x0, y0, x1, y1, txt, _no, _type in page.get_text("blocks"):
        t = " ".join(txt.split())
        if not t or any(t.startswith(p) for p in ignore):
            continue
        out.append((
            y0 / page.rect.height * page_h_in,
            y1 / page.rect.height * page_h_in,
            x0 / page.rect.width * page_w_in,
            x1 / page.rect.width * page_w_in,
            t,
        ))
    out.sort()
    return out


def table_bottom(page, page_h_in):
    """表の罫線（細く長い矩形）の最下端を返す。無ければ None。"""
    ys = [d["rect"].y1 / page.rect.height * page_h_in
          for d in page.get_drawings()
          if d["rect"].width > 50 and d["rect"].height < 2]
    return max(ys) if ys else None


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("pdf", help="PDF、または --from-export なら書き出しJSON")
    ap.add_argument("--from-export", action="store_true", help="Drive の書き出しJSONを渡す")
    ap.add_argument("--pdf-out", default="deck.pdf", help="--from-export のときの保存先")
    ap.add_argument("--footer-top", type=float, default=5.42,
                    help="フッタ帯の上端（インチ）。これより下に本文があれば崩れ")
    ap.add_argument("--right-margin", type=float, default=0.25,
                    help="右端に残すべき余白（インチ）")
    ap.add_argument("--table-gap", type=float, default=0.10,
                    help="表の下端と次の本文のあいだに要る余白（インチ）")
    ap.add_argument("--ignore-prefix", action="append", default=["Copyright"],
                    help="無視する行の先頭文字列（マスタのフッタなど）")
    ap.add_argument("--render-dir", help="指定すると全ページを PNG で書き出す")
    ap.add_argument("--dpi", type=int, default=100)
    a = ap.parse_args()

    doc = pymupdf.open(load_pdf(a.pdf, a.from_export, a.pdf_out))
    page_w_in = doc[0].rect.width / 72.0
    page_h_in = doc[0].rect.height / 72.0
    print(f"{doc.page_count} ページ / {page_w_in:.2f} × {page_h_in:.2f} in\n")

    bad = []
    for i in range(doc.page_count):
        page = doc[i]
        bl = blocks(page, page_w_in, page_h_in, a.ignore_prefix)
        problems = []

        low = max((b[1] for b in bl), default=0.0)
        if low > a.footer_top:
            problems.append(f"画面外/フッタ侵入 (最下端 {low:.2f} > {a.footer_top:.2f})")

        right = max((b[3] for b in bl), default=0.0)
        if right > page_w_in - a.right_margin:
            problems.append(f"右端はみ出し ({right:.2f} > {page_w_in - a.right_margin:.2f})")

        for x in range(len(bl)):
            for y in range(x + 1, len(bl)):
                ay0, ay1, ax0, ax1, at = bl[x]
                by0, by1, bx0, bx1, bt = bl[y]
                if ay0 < by1 - 0.02 and by0 < ay1 - 0.02 and ax0 < bx1 - 0.02 and bx0 < ax1 - 0.02:
                    problems.append(f"要素の重なり: {at[:24]!r} ↔ {bt[:24]!r}")

        tb = table_bottom(page, page_h_in)
        if tb:
            below = [b for b in bl if b[0] > tb - 0.30]
            if below:
                gap = below[0][0] - tb
                if gap < a.table_gap:
                    problems.append(
                        f"表の直下が詰まりすぎ (余白 {gap:+.2f} < {a.table_gap:.2f}): {below[0][4][:28]!r}")

        status = "NG" if problems else "ok"
        print(f"[{i + 1:2d}] {status}  最下端={low:.2f} 右端={right:.2f}"
              + (f" 表下端={tb:.2f}" if tb else ""))
        for p in problems:
            print(f"       - {p}")
        if problems:
            bad.append(i + 1)

        if a.render_dir:
            import os
            os.makedirs(a.render_dir, exist_ok=True)
            page.get_pixmap(dpi=a.dpi).save(f"{a.render_dir}/page-{i + 1:02d}.png")

    print()
    if bad:
        print(f"要修正: {bad}")
        sys.exit(1)
    print("全ページ、はみ出し・重なり・表下の詰まりなし")


if __name__ == "__main__":
    main()
