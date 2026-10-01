#!/usr/bin/env python3
"""読み上げスクリプトの所要時間を実測する。

想定するスクリプトの書式（20261004_script.md と同じ）:

    # 背景                 ← ブロック見出し（任意）
    ## ▶ S7               ← スライド見出し
    本文。／区切り。〈間〉   ← 話す文（空行で段落を分ける）

話速モデル:
    全角1文字 = 1/CPM 分、〈間〉= 1.0 秒、／ = 0.3 秒
    太字記号 ** と、表・箇条書き・引用行は数えない。

使い方:
    python3 measure_script_timing.py 20261004_script.md
    python3 measure_script_timing.py 20261004_script.md --cpm 300
    python3 measure_script_timing.py 20261004_script.md --stop-at 時間配分
"""
import argparse
import re
import sys

SLIDE_RE = re.compile(r"^##\s*▶\s*(S\d+)")
HEAD_RE = re.compile(r"^(#{1,6})\s+(.*)$")
SKIP_RE = re.compile(r"^(\||[-*]\s|>)")


def fmt(sec):
    return f"{int(sec // 60)}分{int(round(sec % 60)):02d}秒"


def duration(text, cpm, pause, slash):
    """1行の所要秒数を返す。"""
    n_pause = text.count("〈間〉")
    n_slash = text.count("／")
    body = text.replace("〈間〉", "").replace("／", "").replace("**", "")
    chars = len(re.sub(r"\s", "", body))
    return chars / cpm * 60 + n_pause * pause + n_slash * slash


def parse(path, cpm, pause, slash, stop_at):
    """(順序つきスライド一覧, スライド→秒, スライド→ブロック名) を返す。"""
    order, per_slide, block_of = [], {}, {}
    slide = None
    block = "(no section)"
    for line in open(path, encoding="utf-8").read().split("\n"):
        m = SLIDE_RE.match(line)
        if m:
            slide = m.group(1)
            order.append(slide)
            per_slide[slide] = 0.0
            block_of[slide] = block
            continue
        h = HEAD_RE.match(line)
        if h:
            if stop_at and h.group(2).strip().startswith(stop_at):
                break
            if len(h.group(1)) == 1:          # 「# 背景」などのブロック見出し
                block = h.group(2).strip()
            slide = None                       # 見出しが来たらスライド外に出る
            continue
        if slide is None:
            continue
        t = line.strip()
        if not t or t == "---" or SKIP_RE.match(t):
            continue
        per_slide[slide] += duration(t, cpm, pause, slash)
    return order, per_slide, block_of


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("script")
    ap.add_argument("--cpm", type=float, default=280.0, help="1分あたりの字数（既定 280＝ゆっくりめ）")
    ap.add_argument("--pause", type=float, default=1.0, help="〈間〉の秒数")
    ap.add_argument("--slash", type=float, default=0.3, help="／の秒数")
    ap.add_argument("--stop-at", default="時間配分",
                    help="この見出しより後ろは数えない（既定: 時間配分）")
    ap.add_argument("--budget", type=float, default=None, help="発表の持ち時間（分）。超過分を表示する")
    a = ap.parse_args()

    order, per, block_of = parse(a.script, a.cpm, a.pause, a.slash, a.stop_at)
    if not order:
        sys.exit(f"スライド見出し（## ▶ S1 形式）が {a.script} に見つかりません")

    print(f"# スライド別（{a.cpm:.0f}字/分、〈間〉{a.pause}秒、／{a.slash}秒）\n")
    for s in order:
        print(f"  {s:4s} {per[s]:6.1f}s  {fmt(per[s])}")

    total = sum(per.values())
    print(f"\n# ブロック別\n")
    seen, cum = [], 0.0
    for s in order:
        b = block_of[s]
        if not seen or seen[-1][0] != b:
            seen.append([b, [], 0.0])
        seen[-1][1].append(s)
        seen[-1][2] += per[s]
    print("| ブロック | スライド | 実測 |")
    print("|---|---|---|")
    for b, slides, sec in seen:
        cum += sec
        print(f"| {b} | {slides[0]}–{slides[-1]} | {fmt(sec)} |")
    print(f"\n**合計 {fmt(total)}**")
    if a.budget:
        diff = total - a.budget * 60
        verdict = f"{fmt(abs(diff))} {'超過' if diff > 0 else '余裕'}"
        print(f"持ち時間 {a.budget:.0f}分 に対して {verdict}")

    print(f"\n# リハーサルのチェックポイント（各スライドを話し終えた時点）\n")
    c = 0.0
    for s in order:
        c += per[s]
        print(f"  {fmt(c)}  {s} まで終了")

    print(f"\n# 長い順（削る候補）\n")
    for s in sorted(order, key=lambda x: -per[x])[:8]:
        print(f"  {fmt(per[s])}  {s}")


if __name__ == "__main__":
    main()
