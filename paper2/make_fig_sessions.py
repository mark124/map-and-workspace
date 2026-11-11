"""Figure 0 for paper 2: every tool call of all 160 sessions, one row per session ("barcode" plot).

Every number drawn here is computed from dataset/tool_calls.jsonl and dataset/sessions.csv; nothing is typed in.
Writes figures/fig0_sessions.png (200 dpi) and figures/fig0_sessions.svg.

Run:  python -I paper2/make_fig_sessions.py
"""
import csv
import json
import site
import statistics
import sys
from collections import Counter, defaultdict
from pathlib import Path

try:
    import matplotlib
except ModuleNotFoundError:
    # `python -I` hides the per-user site-packages, which is where matplotlib lives on this machine.
    sys.path.append(site.getusersitepackages())
    import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from matplotlib.colors import ListedColormap  # noqa: E402
from matplotlib.patches import Patch  # noqa: E402

HERE = Path(__file__).resolve().parent
DATA = HERE / "dataset"
FIG = HERE / "figures"
FIG.mkdir(exist_ok=True)

# ---- palette: dataviz reference palette. The four saturated hues (blue, red, yellow, aqua) pass the
# validator with --pairs all (worst CVD dE 9.1, normal-vision 22.9); reads and shell calls are kept as quiet
# neutrals/tints on purpose so the coloured events stand out. -------------------------------------------
INK, INK2, MUTED = "#0b0b0b", "#52514e", "#898781"
GRID, AXIS, SURFACE = "#e1e0d9", "#c3c2b7", "#ffffff"
GOOD = "#0ca30c"

plt.rcParams.update({
    "font.family": "sans-serif",
    "font.sans-serif": ["Segoe UI", "DejaVu Sans"],
    "svg.fonttype": "path",
    "figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "savefig.facecolor": SURFACE,
    "text.color": INK, "axes.labelcolor": INK2, "xtick.color": INK2, "ytick.color": INK2,
    "axes.edgecolor": AXIS, "axes.linewidth": 0.8,
})

RUNS = ["stock_12b", "check_12b", "checknote_12b", "check_31b"]
RUN_LABEL = {
    "stock_12b": "12B, sample instructions",
    "check_12b": "12B, + line asking it to check its edits",
    "checknote_12b": "12B, + check line, + note in graph replies",
    "check_31b": "31B, + line asking it to check its edits",
}

# category order = legend order; value = (label, colour)
CATS = [
    ("read", "Read a file", "#c6d9f1"),
    ("shell", "Shell command: search, view, other", "#e1e0d9"),
    ("run", "Ran a test or Python script", "#eda100"),
    ("edit_ok", "Edit or file write that worked", "#2a78d6"),
    ("edit_nf", "Edit failed: text to replace not found", "#d03b3b"),
    ("edit_other", "Edit refused for another reason", "#f2b3b3"),
    ("graph", "Code-graph lookup", "#1baf7a"),
    ("submit", "Submit the patch", INK),
    ("misc", "Start sub-agent, check status", MUTED),
]
CAT_IDX = {c[0]: i for i, c in enumerate(CATS)}


def category(r):
    t = r["tool"]
    if t == "read_file":
        return "read"
    if t == "run_command":
        return "run" if r.get("cmd_kind") in ("python script", "test") else "shell"
    if t == "edit_file":
        e = r.get("edit_result")
        if e == "ok":
            return "edit_ok"
        if e == "old_string not found":
            return "edit_nf"
        return "edit_other"
    if t == "write_file":
        return "edit_ok" if r.get("reply_status") == "ok" else "edit_other"
    if t in ("search_similar_code", "get_code_neighbors", "get_code_subgraph"):
        return "graph"
    if t == "submit_patch":
        return "submit"
    return "misc"


SHORT = {"solved": "\nsolved", "submitted, not solved": "\nsubmitted,\nnot solved",
         "hit a time, call or turn limit": "\nhit a limit", "stopped early or prompt overflowed": "\nother end"}


def outcome_group(s):
    if s["resolved"] == "True":
        return 0, "solved"
    if s["session_end"] == "submitted":
        return 1, "submitted, not solved"
    if s["session_end"] in ("time limit", "tool-call limit", "turn limit"):
        return 3, "hit a time, call or turn limit"
    return 2, "stopped early or prompt overflowed"


def main():
    seqs = defaultdict(dict)       # (run, task) -> {call_index: cat}
    n_rows = 0
    nf_by_model = Counter()
    for line in open(DATA / "tool_calls.jsonl", encoding="utf-8"):   # streamed, one row at a time
        r = json.loads(line)
        seqs[(r["run"], r["task"])][r["call_index"]] = category(r)
        n_rows += 1
        if r["tool"] == "edit_file" and r.get("edit_result") == "old_string not found":
            nf_by_model[r["model"]] += 1
    sessions = {(s["run"], s["task"]): s for s in csv.DictReader(open(DATA / "sessions.csv", encoding="utf-8"))}
    assert set(sessions) == set(seqs), "sessions.csv and tool_calls.jsonl disagree on the session list"
    for k, d in seqs.items():
        assert sorted(d) == list(range(len(d))), k
    max_len = max(len(d) for d in seqs.values())

    # streaks of back-to-back "not found" edits (other calls in between break the streak)
    def longest_nf_streak(cats):
        best = cur = 0
        for c in cats:
            cur = cur + 1 if c == "edit_nf" else 0
            best = max(best, cur)
        return best

    stats = {"total_calls": n_rows, "sessions": len(seqs), "not_found_by_model": dict(nf_by_model),
             "max_len": max_len, "runs": {}}

    fig = plt.figure(figsize=(8.2, 8.2))
    # 2 x 2 panels; each panel keeps a narrow strip on its right for the outcome-group labels
    L, B, T = 0.035, 0.16, 0.80
    lab_w, hgap, vgap = 0.085, 0.03, 0.095
    pw = (0.985 - L - 2 * lab_w - hgap) / 2
    ph = (T - B - vgap) / 2
    cmap = ListedColormap([c[2] for c in CATS])
    cmap.set_bad(SURFACE)

    for k, run in enumerate(RUNS):
        col, row = k % 2, k // 2
        ax = fig.add_axes([L + col * (pw + lab_w + hgap), T - (row + 1) * ph - row * vgap, pw, ph])
        keys = [key for key in seqs if key[0] == run]
        keys.sort(key=lambda key: (outcome_group(sessions[key])[0], len(seqs[key]), key[1]))
        n = len(keys)
        M = np.full((n, max_len), np.nan)
        for i, key in enumerate(keys):
            d = seqs[key]
            for j in range(len(d)):
                M[i, j] = CAT_IDX[d[j]]
        ax.pcolormesh(np.arange(max_len + 1), np.arange(n + 1), np.ma.masked_invalid(M), cmap=cmap,
                      vmin=-0.5, vmax=len(CATS) - 0.5, edgecolors=SURFACE, linewidth=0.35)
        ax.set_xlim(-3.2, max_len + 0.2)
        ax.set_ylim(n + 0.1, -0.1)
        # group separators + group labels on the right
        groups = [outcome_group(sessions[key]) for key in keys]
        start = 0
        glabels = []
        for i in range(1, n + 1):
            if i == n or groups[i] != groups[start]:
                if start > 0:
                    ax.plot([-3.2, max_len], [start, start], color=INK2, lw=0.7, clip_on=False)
                glabels.append((start, i, groups[start][1]))
                start = i
        for a, b, lab in glabels:
            cnt = b - a
            if cnt > 2:
                ax.text(max_len + 1.2, (a + b) / 2, f"{cnt}{SHORT[lab]}", ha="left",
                        va="center", fontsize=7.6, color=INK2, linespacing=1.15)
            else:   # a tiny group: label each row at its own end, inside the panel
                for i in range(a, b):
                    end = sessions[keys[i]]["session_end"]
                    txt = "prompt overflowed" if end == "context overflow" else "stopped without submitting"
                    ax.text(len(seqs[keys[i]]) + 0.8, i + 0.5, txt, ha="left", va="center", fontsize=6.3,
                            color=INK2, zorder=5, bbox=dict(boxstyle="square,pad=0.04", fc=SURFACE, ec="none"))
        # solved tick at the left of each solved row
        for i, key in enumerate(keys):
            if sessions[key]["resolved"] == "True":
                ax.text(-1.6, i + 0.5, "✓", ha="center", va="center", fontsize=6.6, color=GOOD, family="DejaVu Sans",
                        weight="bold")
        ax.set_yticks([])
        ax.set_xticks([0, 10, 20, 30, 40, 50, 60])
        ax.tick_params(axis="x", labelsize=8, length=2.5, color=AXIS)
        for s in ("top", "right", "left"):
            ax.spines[s].set_visible(False)
        ax.spines["bottom"].set_bounds(0, max_len)
        if row == 1:
            ax.set_xlabel("Tool call number within the session", fontsize=8.6, color=INK2, labelpad=3)

        # per-run numbers
        lens = [len(seqs[key]) for key in keys]
        cats_all = Counter(c for key in keys for c in seqs[key].values())
        solved = sum(sessions[key]["resolved"] == "True" for key in keys)
        limit = sum(outcome_group(sessions[key])[0] == 3 for key in keys)
        streaks = [longest_nf_streak([seqs[key][j] for j in range(len(seqs[key]))]) for key in keys]
        st = {"sessions": n, "calls": sum(lens), "median_len": statistics.median(lens), "solved": solved,
              "hit_limit": limit, "not_found": cats_all["edit_nf"], "graph": cats_all["graph"],
              "sessions_with_3plus_nf_in_a_row": sum(s >= 3 for s in streaks), "longest_nf_streak": max(streaks),
              "groups": [(lab, b - a) for a, b, lab in glabels], "cats": dict(cats_all)}
        stats["runs"][run] = st
        ax.text(-3.2, -2.9, RUN_LABEL[run], ha="left", va="bottom", fontsize=9.6, weight="bold", color=INK)
        ax.text(-3.2, -0.9, f"solved {solved} of {n}  ·  {st['not_found']} red (text not found)  ·  "
                f"median {st['median_len']:g} calls  ·  {limit} hit a limit", ha="left", va="bottom",
                fontsize=7.9, color=INK2)

    # title
    fig.text(0.03, 0.975, "Figure 1. Sessions in the 12B runs run longer and more often repeat failed edits or hit a limit;\n"
             "the 31B run's sessions are shorter and mostly end with a patch", fontsize=12.6, linespacing=1.2, weight="bold",
             va="top", color=INK)
    sub = (f"All {n_rows:,} tool calls in {len(seqs)} Gemma 4 agent sessions: one row per session, one cell per call, "
           f"read left to right. Red marks an edit whose text to replace "
           f"was not in the file: {nf_by_model['12B']} in the three 12B runs, {nf_by_model['31B']} in the 31B run. "
           f"A green tick marks a solved task. Rows are sorted by outcome, then by length.")
    import textwrap
    fig.text(0.03, 0.905, textwrap.fill(sub, 128), fontsize=8.6, va="top", color=INK2, linespacing=1.35)

    # legend
    handles = [Patch(facecolor=c[2], edgecolor=AXIS if c[0] in ("shell", "read") else c[2], lw=0.5, label=c[1])
               for c in CATS]
    fig.legend(handles=handles, loc="lower left", bbox_to_anchor=(0.03, 0.008), ncol=3, frameon=False,
               fontsize=8.2, handlelength=1.1, handleheight=0.9, columnspacing=1.4, labelspacing=0.45)
    fig.savefig(FIG / "fig0_sessions.png", dpi=200)
    fig.savefig(FIG / "fig0_sessions.svg")
    plt.close(fig)
    print(json.dumps(stats, indent=1))


if __name__ == "__main__":
    main()
