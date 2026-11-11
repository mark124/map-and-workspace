"""Figure 4 for "The Map and the Workspace": the Gemma 4 reading test, without and with the note (dumbbell chart).

Every number drawn here is computed from the probe result files; nothing is typed in.
Reads (relative to the project root, two levels above this file):
  kaggle_probe/out_test/probe_results.jsonl   E2B
  kaggle_probe/out_full/probe_results.jsonl   E4B, 12B, 31B
Per model and note setting (flag): out-of-date version answered correctly ("recognised"), up-to-date version
answered correctly ("trusted"), and both versions of the same case answered correctly (paired by item and
flag, as in kaggle_probe/paired.py).
Writes fig4_reading_test.png (200 dpi) and fig4_reading_test.svg next to this script.

Run:  python -I paper/figures/make_fig_dumbbell.py
"""
import json
import os
import sys
from collections import defaultdict

HERE = os.path.dirname(os.path.abspath(__file__))
sys.dont_write_bytecode = True
sys.path.insert(0, HERE)
import make_figures as mf  # noqa: E402  (same matplotlib setup and palette as the other paper-1 figures)

plt = mf.plt
INK, INK2, MUTED, GRID, AXIS, SURFACE = mf.INK, mf.INK2, mf.MUTED, mf.GRID, mf.AXIS, mf.SURFACE
UP, DOWN = "#2a78d6", "#eb6834"      # reference palette slots 1 and 2: more correct with the note / fewer

ROOT = mf.ROOT
FILES = [("out_test", {"e2b"}), ("out_full", {"e4b", "12b", "31b"})]
ORDER = ["E2B", "E4B", "12B", "31B"]


def short(model):
    # "gemma-4-e4b-it-qat-w4a16-ct" -> "E4B"
    return model.split("-")[2].upper()


def load():
    rows = []
    for folder, expect in FILES:
        path = os.path.join(ROOT, "kaggle_probe", folder, "probe_results.jsonl")
        got = set()
        with open(path, encoding="utf-8") as fh:
            for line in fh:
                if line.strip():
                    r = json.loads(line)
                    got.add(short(r["model"]).lower())
                    rows.append(r)
        assert got == expect, (folder, got)
    return rows


def tally(rows):
    by = defaultdict(dict)                       # (model, item, flag) -> {fresh: row}
    for r in rows:
        key = (short(r["model"]), r["item"], r["flag"])
        assert r["fresh"] not in by[key], key   # one answer per version
        by[key][r["fresh"]] = r
    out = {}
    for m in ORDER:
        for flag in (False, True):
            keys = [k for k in by if k[0] == m and k[2] == flag]
            assert all(len(by[k]) == 2 for k in keys), (m, flag)
            out[(m, flag)] = {
                "items": len(keys),
                "recognised": sum(bool(by[k][False]["correct"]) for k in keys),
                "trusted": sum(bool(by[k][True]["correct"]) for k in keys),
                "both": sum(bool(by[k][False]["correct"] and by[k][True]["correct"]) for k in keys),
                "same_verdict": sum(by[k][False]["meaning"] == by[k][True]["meaning"] for k in keys),
            }
    return out


PANELS = [
    ("recognised", "Out-of-date reply recognised", "said the reply is from before the edit"),
    ("trusted", "Up-to-date reply trusted", "said the reply is current"),
    ("both", "Both versions judged correctly", "same case, out-of-date and up-to-date"),
]


def main():
    rows = load()
    t = tally(rows)
    n_items = {v["items"] for v in t.values()}
    assert len(n_items) == 1
    N = n_items.pop()

    W, H = 8.2, 4.15
    fig = plt.figure(figsize=(W, H))
    x0, x1, gap = 0.085, 0.985, 0.035
    pw = (x1 - x0 - 2 * gap) / 3
    yb, yt = 0.17, 0.66
    for p, (key, title, gloss) in enumerate(PANELS):
        ax = fig.add_axes([x0 + p * (pw + gap), yb, pw, yt - yb])
        for i, m in enumerate(ORDER):
            y = len(ORDER) - 1 - i
            a, b = t[(m, False)][key], t[(m, True)][key]
            if i % 2 == 0:
                ax.axhspan(y - 0.5, y + 0.5, color="#f6f6f3", lw=0, zorder=0)
            if a == b:
                ax.scatter([a], [y], s=52, color=INK2, zorder=4, lw=0)
                ax.text(a, y + 0.28, f"{a}", ha="center", va="bottom", fontsize=8.2, color=INK)
                continue
            col = UP if b > a else DOWN
            if abs(b - a) >= 5:      # room for an arrowhead between the dots
                ax.annotate("", xy=(b, y), xytext=(a, y),
                            arrowprops=dict(arrowstyle="-|>,head_length=0.45,head_width=0.22", color=col, lw=2.0,
                                            shrinkA=4, shrinkB=4), zorder=3)
            else:                    # too close for a head: a plain bar; the dot fill still shows direction
                ax.plot([a, b], [y, y], color=col, lw=2.0, zorder=3, solid_capstyle="butt")
            ax.scatter([a], [y], s=46, facecolor=SURFACE, edgecolor=col, linewidth=1.6, zorder=4)
            ax.scatter([b], [y], s=46, color=col, zorder=4, lw=0)
            lo, hi = (a, b) if a < b else (b, a)
            ax.text(lo - 2.0, y, str(lo), ha="right", va="center", fontsize=8.2,
                    color=INK2 if lo == a else INK)
            ax.text(hi + 2.0, y, str(hi), ha="left", va="center", fontsize=8.2,
                    color=INK2 if hi == a else INK, weight="bold" if hi == b else "normal")
        ax.set_xlim(-3, N + 3)
        ax.set_ylim(-0.6, len(ORDER) - 0.4)
        ax.set_xticks([0, 12, 24, 36, 48])
        ax.tick_params(axis="x", labelsize=8, length=2.5, color=AXIS)
        ax.xaxis.grid(True, color=GRID, lw=0.7, zorder=0)
        ax.set_axisbelow(True)
        ax.set_yticks(range(len(ORDER)))
        ax.set_yticklabels(ORDER[::-1] if p == 0 else [], fontsize=9.6, color=INK)
        ax.tick_params(axis="y", length=0, pad=6)
        for s in ("top", "right", "left"):
            ax.spines[s].set_visible(False)
        ax.spines["bottom"].set_color(AXIS)
        ax.spines["bottom"].set_bounds(0, N)
        ax.set_title(title, loc="left", fontsize=9.8, weight="bold", color=INK, pad=16)
        ax.text(0.0, 1.015, gloss, transform=ax.transAxes, ha="left", va="bottom", fontsize=7.9, color=INK2)
        ax.set_xlabel(f"cases, of {N}", fontsize=8.2, color=INK2, labelpad=3)

    fig.text(0.03, 0.965, "With a two-field note on the map's age, Gemma 4 31B trusts up-to-date replies more often",
             fontsize=12.2, weight="bold", va="top", color=INK)
    fig.text(0.03, 0.895, f"{N} cases built from real tool replies, each shown in an out-of-date and an up-to-date "
             "version, without the note (open dot) and with it (filled dot).\nThe arrow points to the result "
             "with the note: blue = more cases correct, orange = fewer; a single dot = no change.",
             fontsize=8.7, va="top", color=INK2, linespacing=1.4)
    fig.text(0.03, 0.015, "Official 4-bit quantized models, temperature 0, thinking off; multiple-choice question "
             "with three answers in shuffled order.", fontsize=7.6, color=MUTED, va="bottom")
    mf.save(fig, "fig4_reading_test")
    print(json.dumps({f"{m} {'note' if f else 'no note'}": v for (m, f), v in t.items()}, indent=1))


if __name__ == "__main__":
    main()
