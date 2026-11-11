"""Figure 2b for "The Map and the Workspace": which out-of-date-map problems occur together (UpSet plot).

Every number drawn here is computed from stale/census.jsonl and stale/replay.jsonl; nothing is typed in.
The per-task tests are the same ones make_figures.py (and stale/headline.py) use for the Section 3 table, and the
script checks that its per-type totals equal make_figures.prevalence() before drawing.
All edited files are counted (not library code only), as in the first column of the Section 3 table.
Writes fig2b_upset.png (200 dpi) and fig2b_upset.svg next to this script.

Run:  python -I paper/figures/make_fig_upset.py
"""
import json
import os
import sys
from collections import Counter

HERE = os.path.dirname(os.path.abspath(__file__))
sys.dont_write_bytecode = True
sys.path.insert(0, HERE)
import make_figures as mf  # noqa: E402  (sets matplotlib up the same way, incl. the user-site fallback)

plt = mf.plt
INK, INK2, MUTED, GRID, AXIS, SURFACE, BLUE = mf.INK, mf.INK2, mf.MUTED, mf.GRID, mf.AXIS, mf.SURFACE, mf.BLUE
DOT_OFF = "#e1e0d9"

# key, short label for the matrix rows (same wording as the Section 3 table, shortened)
TYPES = [
    ("old_text", "Changed function shown with its earlier source"),
    ("stale_search", "Search shows another changed function's earlier source"),
    ("new_unseen", "Added function or class not in the map yet"),
    ("missing_call", "Call list misses a call the fix added"),
    ("subs", "Lookup of an added function answered by an older one"),
    ("phantom_call", "Call list keeps a call the fix removed"),
    ("removed_listed", "Deleted function or class still listed"),
]
FIRST_THREE = ("old_text", "new_unseen", "subs")


def membership(census, replay):
    """Per task: the set of types that apply. Same tests as make_figures.prevalence (all edited files)."""
    syms = {x["symbol"] for r in census for x in r["rows"]}
    out = {}
    for r in census:
        rows = r["rows"]
        p = replay[r["id"]]
        s = set()
        if any(x["change"] == "body_changed" and x["in_graph"] and x["kind"] == "func" for x in rows):
            s.add("old_text")
        if any(x["change"] == "added" for x in rows):
            s.add("new_unseen")
        if any(x["change"] == "removed" and x["in_graph"] for x in rows):
            s.add("removed_listed")
        if any(any(x["symbol"].endswith(sub["asked"]) and x["symbol"] in syms for x in rows)
               for sub in p["substitutions"]):
            s.add("subs")
        if p["stale_code_in_search"]:
            s.add("stale_search")
        if p["missing_callees"]:
            s.add("missing_call")
        if p["phantom_callees"]:
            s.add("phantom_call")
        out[r["id"]] = frozenset(s)
    return out


def main():
    census, replay, _, _ = mf.load()
    N = len(census)
    mem = membership(census, replay)
    sizes = {k: sum(k in s for s in mem.values()) for k, _ in TYPES}
    # reconcile with the figure-2 / Section 3 numbers
    ref = mf.prevalence(census, replay)["all"]
    for k, _ in TYPES:
        assert sizes[k] == ref[k], (k, sizes[k], ref[k])
    any3 = sum(bool(s & set(FIRST_THREE)) for s in mem.values())
    assert any3 == ref["any3"], (any3, ref["any3"])
    none_all = sum(not s for s in mem.values())
    none3 = N - any3

    combos = Counter(mem.values())
    combos_nonempty = [(c, n) for c, n in combos.items() if c]
    combos_nonempty.sort(key=lambda cn: (-cn[1], len(cn[0]), sorted(cn[0])))
    MIN_TASKS = 2                       # show every combination shared by at least two tasks
    shown = [cn for cn in combos_nonempty if cn[1] >= MIN_TASKS]
    rest = [cn for cn in combos_nonempty if cn[1] < MIN_TASKS]
    rest_tasks = sum(n for _, n in rest)
    any7 = N - none_all
    multi = sum(len(s) >= 2 for s in mem.values())
    cols = shown + [(frozenset(), none_all)]      # last column: tasks with none of the seven

    types = sorted(TYPES, key=lambda t: -sizes[t[0]])   # matrix rows, largest set on top
    nT = len(types)

    W, H = 8.2, 5.6
    fig = plt.figure(figsize=(W, H))
    xLab, xS0, xS1, xM0, xM1 = 0.395, 0.405, 0.50, 0.535, 0.985
    yB, yM, yT = 0.085, 0.43, 0.755
    ax_bar = fig.add_axes([xM0, yM + 0.015, xM1 - xM0, yT - yM - 0.015])
    ax_mat = fig.add_axes([xM0, yB, xM1 - xM0, yM - yB])
    ax_set = fig.add_axes([xS0, yB, xS1 - xS0, yM - yB])

    ncol = len(cols)
    xs = list(range(ncol))
    vals = [n for _, n in cols]
    for x, (c, v) in zip(xs, cols):
        ax_bar.bar([x], [v], width=0.62, color=BLUE if c else MUTED, zorder=3)
        ax_bar.text(x, v + max(vals) * 0.02, str(v), ha="center", va="bottom", fontsize=8.6, color=INK)
    ax_bar.text(xs[-1], vals[-1] + max(vals) * 0.09, "none", ha="center", va="bottom", fontsize=7.8, color=INK2)
    ax_bar.set_xlim(-0.6, ncol - 0.4)
    ax_bar.set_ylim(0, max(vals) * 1.12)
    ax_bar.axis("off")
    ax_bar.plot([-0.6, ncol - 0.4], [0, 0], color=AXIS, lw=0.8)
    ax_bar.text(0.7, max(vals) * 0.9, "Tasks with exactly this combination", ha="left", va="top", fontsize=8.6,
                color=INK2)

    for i, (k, _) in enumerate(types):
        y = nT - 1 - i
        if i % 2 == 0:
            ax_mat.axhspan(y - 0.5, y + 0.5, color="#f6f6f3", zorder=0, lw=0)
        for x, (c, _) in zip(xs, cols):
            ax_mat.scatter([x], [y], s=44, color=INK if k in c else DOT_OFF, zorder=3, lw=0)
    for x, (c, _) in zip(xs, cols):
        ys = [nT - 1 - i for i, (k, _) in enumerate(types) if k in c]
        if len(ys) > 1:
            ax_mat.plot([x, x], [min(ys), max(ys)], color=INK, lw=1.6, zorder=2)
    ax_mat.set_xlim(-0.6, ncol - 0.4)
    ax_mat.set_ylim(-0.5, nT - 0.5)
    ax_mat.axis("off")

    for i, (k, label) in enumerate(types):
        y = nT - 1 - i
        v = sizes[k]
        ax_set.axhspan(y - 0.5, y + 0.5, color="#f6f6f3" if i % 2 == 0 else SURFACE, zorder=0, lw=0)
        ax_set.barh(y, v, height=0.56, color=BLUE, zorder=3)
        ax_set.text(v + N * 0.03, y, str(v), ha="left", va="center", fontsize=8.3, color=INK2)
        ax_set.text(-N * 0.06, y, label + (" †" if k in FIRST_THREE else ""), ha="right", va="center",
                    fontsize=8.6, color=INK, clip_on=False)
    ax_set.set_xlim(0, N)
    ax_set.set_ylim(-0.5, nT - 0.5)
    ax_set.axis("off")
    ax_set.text(0, nT - 0.35, f"Tasks (of {N})", ha="left", va="bottom", fontsize=8, color=INK2)

    fig.text(0.035, 0.965, f"After one correct fix, {multi} of the {any7} affected tasks show two or more\n"
             "kinds of out-of-date tool answer", fontsize=12.6, weight="bold", va="top", color=INK,
             linespacing=1.2)
    import textwrap
    sub = (f"Each of the {N} public tasks, after its reference fix is applied: which kinds of out-of-date answer "
           f"it shows (dots, one row per kind) and how many tasks share each exact combination (bars). Shown: "
           f"every combination found in two or more tasks; the other {len(rest)} occur once each "
           f"({rest_tasks} tasks).")
    fig.text(0.035, 0.865, textwrap.fill(sub, 118), fontsize=8.9, va="top", color=INK2, linespacing=1.35)
    fig.text(0.035, 0.70, f"{none_all} of {N} tasks show none of the seven kinds.\n"
             f"† At least one of the three marked kinds:\n   {any3} of {N} tasks ({none3} show none of those three).",
             fontsize=8.9, va="top", color=INK, linespacing=1.45)
    fig.text(0.035, 0.015, "All edited files counted. A task counts once per kind, however many functions it "
             "affects.", fontsize=7.6, color=MUTED, va="bottom")
    mf.save(fig, "fig2b_upset")

    report = {"N": N, "set_sizes": sizes, "any_first_three": any3, "none_of_first_three": none3,
              "none_of_seven": none_all, "n_combinations": len(combos_nonempty), "two_or_more_kinds": multi, "any_of_seven": any7,
              "shown": [(sorted(c), n) for c, n in shown], "not_shown_combinations": len(rest),
              "not_shown_tasks": rest_tasks}
    print(json.dumps(report, indent=1))


if __name__ == "__main__":
    main()
