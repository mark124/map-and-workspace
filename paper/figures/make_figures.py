"""Figures for "The Map and the Workspace" (Gemma 4 Developer Agent, paper track).

Every number drawn here is computed from the data files; nothing is typed in.
Reads (relative to the project root, two levels above this file):
  stale/census.jsonl              per task, what the reference patch changed and whether the graph knows it
  stale/replay.jsonl              per task, what the re-implemented graph tools answer after the edit
  stale/probe_items.json          real post-edit tool replies (figure 1 uses the FileProxy.isatty item)
  kaggle_refresh/out/equivalence.json   per task, refresh vs full rebuild (equal? timings)
  data/graphs/<repo>_<commit>.json      optional: the competition graph for the figure-1 task, used only
                                        to show the source of the node that really answered
Writes fig1_short_name_match, fig2_prevalence, fig3_refresh as PNG (200 dpi) and SVG next to this script.

Run:  python -I paper/figures/make_figures.py
"""
import json
import math
import os
import site
import statistics
import sys
import textwrap

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.normpath(os.path.join(HERE, os.pardir, os.pardir))

try:
    import matplotlib
except ModuleNotFoundError:
    # `python -I` hides the per-user site-packages, which is where matplotlib lives on this machine.
    sys.path.append(site.getusersitepackages())
    import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.lines import Line2D  # noqa: E402
from matplotlib.patches import Circle, FancyBboxPatch, Patch  # noqa: E402
from matplotlib.transforms import blended_transform_factory  # noqa: E402

# ---- palette (dataviz reference palette; validated with its validator before use) ----------------
INK, INK2, MUTED = "#0b0b0b", "#52514e", "#898781"
GRID, AXIS, SURFACE, PLANE = "#e1e0d9", "#c3c2b7", "#ffffff", "#f6f6f3"
BLUE, BLUE_DARK = "#2a78d6", "#104281"      # one hue, two steps (all files / library only)
CRIT, GOOD = "#d03b3b", "#0ca30c"           # status colours: the wrong answer / the correct edit

plt.rcParams.update({
    "font.family": "sans-serif",
    "font.sans-serif": ["Segoe UI", "DejaVu Sans"],
    "font.monospace": ["Consolas", "DejaVu Sans Mono"],
    "svg.fonttype": "path",
    "figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "savefig.facecolor": SURFACE,
    "text.color": INK, "axes.labelcolor": INK2, "xtick.color": INK2, "ytick.color": INK2,
    "axes.edgecolor": AXIS, "axes.linewidth": 0.8,
})

REPO_SHORT = {"fastapi/fastapi": "fastapi", "Textualize/rich": "rich", "psf/requests": "requests",
              "encode/httpx": "httpx"}
NONLIB = ("docs_src/", "scripts/", "tests/", "docs/")


def pct(n, N):
    return f"{100 * n / N:.0f}%"


def load():
    def jl(path):
        return [json.loads(l) for l in open(path, encoding="utf-8") if l.strip()]
    census = jl(os.path.join(ROOT, "stale", "census.jsonl"))
    replay = {r["id"]: r for r in jl(os.path.join(ROOT, "stale", "replay.jsonl"))}
    probe = json.load(open(os.path.join(ROOT, "stale", "probe_items.json"), encoding="utf-8"))
    equiv = json.load(open(os.path.join(ROOT, "kaggle_refresh", "out", "equivalence.json"), encoding="utf-8"))
    return census, replay, probe, equiv


def save(fig, name):
    fig.savefig(os.path.join(HERE, name + ".png"), dpi=200)
    fig.savefig(os.path.join(HERE, name + ".svg"))
    plt.close(fig)
    print("wrote", name + ".png", name + ".svg")


# =====================================================================================================
# Figure 1: one real silent swap
# =====================================================================================================
def graph_facts(item):
    """Facts about the node that answered, read from the competition graph if it is on disk."""
    repo = REPO_SHORT[item["repo"]]
    path = os.path.join(ROOT, "data", "graphs", f"{repo}_{item['base_commit']}.json")
    if not os.path.exists(path):
        return None
    sys.dont_write_bytecode = True        # leave no cache file behind in stale/
    sys.path.insert(0, os.path.join(ROOT, "stale"))
    from toolsim import resolve_node_name  # the line-by-line reimplementation of the harness resolver
    g = json.load(open(path, encoding="utf-8"))
    ids = [n["id"] for n in g["nodes"]]
    text = {n["id"]: n.get("text") or "" for n in g["nodes"]}
    short = item["asked"]
    cands = [i for i in ids if i.endswith("." + short)]
    got = resolve_node_name(short, ids)
    cls_form = ".".join(item["symbol"].split(".")[-2:])
    nb = sorted({e["target"] for e in g["edges"] if e["source"] == got} |
                {e["source"] for e in g["edges"] if e["target"] == got})
    return {"candidates": cands, "resolved": got, "resolved_text": text[got], "neighbors": nb,
            "class_form": cls_form, "class_form_resolves_to": resolve_node_name(cls_form, ids),
            "full_name_in_graph": item["symbol"] in text}


def fig1(probe, replay):
    item = next(i for i in probe if i["kind"] == "S" and i["symbol"] == "rich.file_proxy.FileProxy.isatty")
    asked, got, reply = item["asked"], item["substituted"], item["reply"]
    task = item["task"]
    assert reply["node"] == asked and got != item["symbol"]
    facts = graph_facts(item)
    if facts is not None:
        assert facts["resolved"] == got, (facts["resolved"], got)
        assert [f"{item['repo']};{n}" for n in facts["neighbors"]] == reply["neighbors"]
        assert not facts["full_name_in_graph"]
    # The Class.method form: replay.jsonl records a substitution only when the resolver returned a node,
    # so its absence means the tool found nothing for that form.
    cls_form = ".".join(item["symbol"].split(".")[-2:])
    cls_form_found = any(s["form"] == "class_method" and s["asked"] == cls_form
                         for s in replay[task]["substitutions"])
    if facts is not None:
        assert cls_form_found == (facts["class_form_resolves_to"] is not None)

    got_file = "/".join(got.split(".")[:-2]) + ".py"          # rich.progress._Reader.isatty -> rich/progress.py
    got_class = got.split(".")[-2]
    new_file = item["file"]
    new_class = item["symbol"].split(".")[-2]

    W, H = 7.4, 7.55
    fig = plt.figure(figsize=(W, H))
    ax = fig.add_axes([0, 0, 1, 1])
    ax.set_xlim(0, W); ax.set_ylim(0, H); ax.set_aspect("equal"); ax.axis("off")

    FS_CODE, FS_TXT = 8.6, 9.3
    LH = FS_CODE * 1.45 / 72            # code line height, inches
    CW = 0.5498 * FS_CODE / 72          # Consolas advance width, inches per character
    X_CODE, W_CODE = 0.28, 3.3
    X_TXT = X_CODE + W_CODE + 0.28
    WRAP = 50

    def badge(y, n, color, title):
        ax.add_patch(Circle((X_CODE + 0.12, y), 0.115, fc=color, ec="none"))
        ax.text(X_CODE + 0.12, y - 0.004, str(n), ha="center", va="center", fontsize=8.5, color="white",
                weight="bold")
        ax.text(X_CODE + 0.34, y, title, ha="left", va="center", fontsize=10.4, weight="bold", color=INK)

    def code_box(y_top, lines, marks=None):
        """lines: list of str. marks: dict line_index -> ('add' | 'crit'). Returns (box_bottom, {i: y_center})."""
        marks = marks or {}
        pad = 0.11
        h = pad * 2 + len(lines) * LH
        box = FancyBboxPatch((X_CODE, y_top - h), W_CODE, h, boxstyle="round,pad=0,rounding_size=0.07",
                             fc=PLANE, ec=GRID, lw=0.8)
        ax.add_patch(box)
        centers = {}
        for i, ln in enumerate(lines):
            yc = y_top - pad - (i + 0.5) * LH
            centers[i] = yc
            kind = marks.get(i)
            if kind == "add":
                ax.add_patch(FancyBboxPatch((X_CODE + 0.08, yc - LH / 2), W_CODE - 0.16, LH,
                                            boxstyle="round,pad=0,rounding_size=0.02", fc=GOOD, alpha=0.12,
                                            ec="none"))
            if kind == "crit":
                wtxt = len(ln.rstrip()) * CW + 0.08
                ax.add_patch(FancyBboxPatch((X_CODE + 0.11, yc - LH / 2 + 0.01), wtxt, LH - 0.02,
                                            boxstyle="round,pad=0,rounding_size=0.03", fc=CRIT, alpha=0.10,
                                            ec=CRIT, lw=1.0))
            ax.text(X_CODE + 0.15, yc, ln, family="monospace", fontsize=FS_CODE, ha="left", va="center",
                    color=INK)
        return y_top - h, centers

    def prose(y_top, s, fs=FS_TXT, color=INK, width=WRAP, x=None):
        t = ax.text(x if x is not None else X_TXT, y_top, textwrap.fill(s, width), ha="left", va="top",
                    fontsize=fs, color=color, linespacing=1.32)
        return t

    def callout(yc, s, color=CRIT, width=WRAP - 2, x_from=None):
        x0 = (x_from if x_from is not None else X_CODE + W_CODE) + 0.02
        ax.add_line(Line2D([x0, X_TXT - 0.07], [yc, yc], color=color, lw=1.0))
        ax.add_patch(Circle((x0, yc), 0.028, fc=color, ec="none"))
        ax.text(X_TXT, yc + 0.07, textwrap.fill(s, width), ha="left", va="top", fontsize=FS_TXT - 0.3,
                color=INK, linespacing=1.3)

    # ---- title ----
    ax.text(0.28, H - 0.30, "A new method is not in the map yet, so a lookup by its\n"
            "short name is answered by an older method with the same name", ha="left", va="top", fontsize=12.6,
            weight="bold", color=INK, linespacing=1.25)
    ax.text(0.28, H - 0.86, f"One real case from the public task {task} ({item['repo']}). The code map behind "
            "the tools is built once, at the task's starting commit, which keeps runs fast and repeatable.",
            ha="left", va="top", fontsize=9.2, color=INK2, wrap=True, linespacing=1.3)
    # wrap the subtitle manually (ax.text wrap is unreliable at this size)
    ax.texts[-1].set_text(textwrap.fill(ax.texts[-1].get_text(), 112))

    # ---- step 1: the edit ----
    y = H - 1.38
    badge(y, 1, BLUE, "The agent adds a method (the reference fix)")
    edit_lines = [l for l in item["edit_code"].rstrip("\n").splitlines()]
    dedent = min(len(l) - len(l.lstrip()) for l in edit_lines if l.strip())
    shown = [f"{new_file}  -  class {new_class}"] + ["+ " + l[dedent:] for l in edit_lines]
    bottom1, _ = code_box(y - 0.2, shown, marks={i + 1: "add" for i in range(len(edit_lines))})
    prose(y - 0.2, f"The edit is correct: {new_class} gains an isatty method that passes the call on to the "
          f"wrapped file. It is now in the file on disk, but not yet in the map, which describes the starting commit.")

    # ---- step 2: the call and the reply ----
    y = bottom1 - 0.33
    badge(y, 2, BLUE, "Then it asks the graph tool about the new method")
    call = f'{item["tool"]}(node="{asked}")'
    reply_lines = json.dumps(reply, indent=2).splitlines()
    lines2 = [call, "", "reply as the agent sees it:"] + reply_lines
    i_node = next(i for i, l in enumerate(lines2) if l.strip().startswith('"node"'))
    i_nb = next(i for i, l in enumerate(lines2) if l.strip().startswith('"' + item["repo"]))
    bottom2, centers = code_box(y - 0.2, lines2, marks={i_node: "crit", i_nb: "crit"})
    callout(centers[i_node], "Shows the name the agent typed; the reply does not say "
            "another method answered.")
    callout(centers[i_nb], f"The one neighbour listed is a class in {got_file}. It belongs to "
            f"{got_class}.isatty, not to the new {new_class}.isatty.")
    if not cls_form_found:
        prose(centers[i_nb] - 0.64, f'Asked as "{cls_form}" instead, the tool reports nothing found. '
              f'Only the short name is matched to another method.', fs=FS_TXT - 0.5, color=INK2)

    # ---- step 3: what actually answered ----
    y = bottom2 - 0.33
    badge(y, 3, CRIT, "The method that answered")
    if facts is not None:
        src = facts["resolved_text"].rstrip("\n").splitlines()
        dd = min(len(l) - len(l.lstrip()) for l in src[1:] if l.strip()) if len(src) > 1 else 0
        src = [src[0]] + ["    " + l[dd:] for l in src[1:]]   # keep the body indented under its def line
        lines3 = [f"{got_file}  -  class {got_class}"] + ["  " + l for l in src]
    else:
        lines3 = [f"{got_file}  -  class {got_class}", f"  {got}"]
    bottom3, _ = code_box(y - 0.2, lines3)
    if facts is not None:
        cands = facts["candidates"]
        lib = [c for c in cands if not c.startswith("tests.")]
        tests = [c for c in cands if c.startswith("tests.")]
        cand_txt = ", ".join(".".join(c.split(".")[-2:]) for c in lib)
        if tests:
            cand_txt += f" and {len(tests)} test double{'s' if len(tests) > 1 else ''} " \
                        f"({', '.join('.'.join(c.split('.')[-2:]) for c in tests)})"
        why = (f'"{asked}" on its own is not a full name in the map. To help with short names, the tool looks '
               f'for names ending in ".{asked}" ({len(cands)} here: {cand_txt}) and picks the shortest, '
               f'{got}. Its only neighbour in the map is its own class, and that is the list the agent received.')
    else:
        why = (f'"{asked}" on its own is not a full name in the map. To help with short names, the tool looks '
               f'for names ending in ".{asked}" and picks the shortest, {got}. Its only neighbour in the map is '
               f'its own class, and that is the list the agent received.')
    prose(y - 0.2, why)

    # ---- flow line through the badges ----
    ax.add_line(Line2D([X_CODE + 0.12, X_CODE + 0.12], [H - 1.38 - 0.13, y + 0.13], color=AXIS, lw=0.9,
                       zorder=0))

    ax.text(0.28, 0.17, "Reply produced by a line-by-line reproduction of the harness lookup code "
            "(swegemma 0.2.11), checked against the official functions on Kaggle: same method, same "
            "neighbour list.", ha="left", va="bottom", fontsize=7.6, color=MUTED, wrap=True)
    ax.texts[-1].set_text(textwrap.fill(ax.texts[-1].get_text(), 150))
    save(fig, "fig1_short_name_match")
    return {"task": task, "asked": asked, "got": got, "reply": reply, "class_form_found": cls_form_found,
            "graph_candidates": facts["candidates"] if facts else None}


# =====================================================================================================
# Figure 2: prevalence per failure type
# =====================================================================================================
def prevalence(census, replay):
    """The Section 3 table, recomputed with the logic of stale/headline.py and stale/replay2.py."""
    N = len(census)
    out = {}
    for lib_only in (False, True):
        def keep(x):
            return not (lib_only and x["file"].startswith(NONLIB))
        syms = {x["symbol"] for r in census for x in r["rows"] if keep(x)}
        c = dict(old_text=0, new_unseen=0, removed_listed=0, subs=0, any3=0, any_headline=0)
        for r in census:
            rows = [x for x in r["rows"] if keep(x)]
            a = any(x["change"] == "body_changed" and x["in_graph"] and x["kind"] == "func" for x in rows)
            b = any(x["change"] == "added" for x in rows)
            d = any(x["change"] == "removed" and x["in_graph"] for x in rows)
            s = any(any(x["symbol"].endswith(sub["asked"]) and x["symbol"] in syms for x in r["rows"])
                    for sub in replay[r["id"]]["substitutions"])
            c["old_text"] += a; c["new_unseen"] += b; c["removed_listed"] += d; c["subs"] += s
            c["any3"] += (a or b or s)            # "any of the first three" rows of the paper's table
            c["any_headline"] += (a or b or d)    # headline.py's "any"
        out["lib" if lib_only else "all"] = c
    # tool-level rows from the replay (the paper does not split these by file type)
    out["all"]["stale_search"] = sum(1 for r in replay.values() if r["stale_code_in_search"])
    out["all"]["missing_call"] = sum(1 for r in replay.values() if r["missing_callees"])
    out["all"]["phantom_call"] = sum(1 for r in replay.values() if r["phantom_callees"])
    out["queries"] = sum(2 * r["added"] for r in replay.values())
    out["subs_total"] = sum(len(r["substitutions"]) for r in replay.values())
    out["N"] = N
    return out


ROWS = [  # key, label, gloss, counted in "any of the three marked types"
    ("old_text", "Changed function is shown with its earlier source", "the map holds the text from before the edit", True),
    ("stale_search", "Search shows another changed function's earlier source", "search_similar_code", False),
    ("new_unseen", "Added function or class is not in the map yet", "added by the fix after the map was built", True),
    ("missing_call", "Call list misses a call the fix added", "get_code_neighbors", False),
    ("subs", "Lookup of an added function is answered by an older one", "get_code_neighbors, by name match", True),
    ("phantom_call", "Call list keeps a call the fix removed", "get_code_neighbors", False),
    ("removed_listed", "Deleted function or class is still listed", "removed by the fix, still in the map", False),
]


def fig2(stats):
    N = stats["N"]
    rows = sorted(ROWS, key=lambda r: -stats["all"][r[0]])
    n = len(rows)
    W, H = 7.6, 5.55
    fig = plt.figure(figsize=(W, H))
    ax = fig.add_axes([0.475, 0.115, 0.495, 0.665])
    bh = 0.30
    for i, (key, label, gloss, marked) in enumerate(rows):
        y = n - 1 - i
        v = stats["all"][key]
        ax.barh(y + 0.17, 100 * v / N, height=bh, color=BLUE, zorder=3)
        ax.text(100 * v / N + 1.2, y + 0.17, f"{v} ({pct(v, N)})", va="center", ha="left", fontsize=9, color=INK)
        lv = stats["lib"].get(key)
        if lv is not None:
            ax.barh(y - 0.17, 100 * lv / N, height=bh, color=BLUE_DARK, zorder=3)
            ax.text(100 * lv / N + 1.2, y - 0.17, f"{lv} ({pct(lv, N)})", va="center", ha="left",
                    fontsize=8.3, color=INK2)
        tr = blended_transform_factory(ax.transAxes, ax.transData)
        ax.text(-0.025, y + 0.13, label + (" †" if marked else ""), transform=tr, ha="right", va="center",
                fontsize=9.4, color=INK)
        ax.text(-0.025, y - 0.21, gloss, transform=tr, ha="right", va="center", fontsize=7.9, color=MUTED)
    ax.set_xlim(0, 100); ax.set_ylim(-0.6, n - 0.4)
    ax.set_xticks([0, 25, 50, 75, 100]); ax.set_xticklabels(["0%", "25%", "50%", "75%", "100%"], fontsize=8.5)
    ax.set_yticks([])
    ax.xaxis.grid(True, color=GRID, lw=0.7, zorder=0); ax.set_axisbelow(True)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    ax.spines["left"].set_color(AXIS); ax.spines["bottom"].set_color(AXIS)
    ax.tick_params(axis="x", length=0, pad=4)
    ax.set_xlabel(f"Share of the {N} public tasks affected", fontsize=9, color=INK2, labelpad=6)
    ax.legend(handles=[Patch(color=BLUE, label="All edited files"),
                       Patch(color=BLUE_DARK, label="Library code only (no docs, examples, scripts, tests)")],
              loc="lower right", bbox_to_anchor=(1.0, 1.0), ncol=2, fontsize=8.3, frameon=False,
              handlelength=1.2, handleheight=0.8, borderaxespad=0.2, columnspacing=1.6)
    a3, l3 = stats["all"]["any3"], stats["lib"]["any3"]
    fig.text(0.03, 0.955, "After a correct fix, how often the pre-built map is out of date", fontsize=12.6, weight="bold",
             va="top", color=INK)
    fig.text(0.03, 0.905, "Each public reference fix is applied, then the three graph tools are asked about the "
             "code it changed.\n"
             f"† At least one of the three marked types: {a3} of {N} tasks ({pct(a3, N)}); "
             f"{l3} ({pct(l3, N)}) counting library code only.",
             fontsize=9.2, va="top", color=INK2, linespacing=1.35)
    fig.text(0.03, 0.025, "Rows with one bar were not split by file type. A task counts once per type, however "
             "many symbols it affects.", fontsize=7.8, color=MUTED, va="bottom")
    save(fig, "fig2_prevalence")


# =====================================================================================================
# Figure 3: refresh vs full rebuild
# =====================================================================================================
def fig3(equiv, census):
    repo_of = {r["id"]: REPO_SHORT[r["repo"]] for r in census}
    n = len(equiv)
    n_applied = sum(1 for e in equiv if e["applied"])
    n_equal = sum(1 for e in equiv if e["equal"])
    xs = [e["full_s"] * 1000 for e in equiv]
    ys = [e["refresh_s"] * 1000 for e in equiv]
    med_x, med_y = statistics.median(xs), statistics.median(ys)
    ratios = [x / y for x, y in zip(xs, ys)]
    r_lo, r_hi = min(ratios), max(ratios)
    repos = {}
    for e, x, y in zip(equiv, xs, ys):
        repos.setdefault(repo_of.get(e["id"], e["id"].split("_")[0]), []).append((x, y))

    W, H = 7.6, 4.75
    fig = plt.figure(figsize=(W, H))
    ax = fig.add_axes([0.085, 0.13, 0.50, 0.63])
    ax.set_xscale("log"); ax.set_yscale("log")
    XL, YL = (20, 2000), (2, 200)
    ax.set_xlim(*XL); ax.set_ylim(*YL); ax.set_aspect("equal", adjustable="box")
    # reference diagonals: same time, 10x faster, 100x faster
    for k, lab in ((1, "same time"), (10, "10× faster"), (100, "100× faster")):
        ax.plot([XL[0], XL[1]], [XL[0] / k, XL[1] / k], color=GRID if k != 10 else AXIS, lw=0.8, zorder=1)
        x_lab = min(XL[1] * 0.78, YL[1] * k * 0.78)
        ax.text(x_lab, x_lab / k * 1.12, lab, rotation=45, rotation_mode="anchor", ha="center", va="bottom",
                fontsize=7.8, color=MUTED, zorder=2)
    # medians as hairlines
    ax.axvline(med_x, color=INK2, lw=0.7, zorder=1); ax.axhline(med_y, color=INK2, lw=0.7, zorder=1)
    ax.text(med_x * 1.06, YL[0] * 1.12, f"median rebuild {med_x:.0f} ms", fontsize=7.8, color=INK2, ha="left",
            va="bottom")
    ax.text(XL[0] * 1.08, med_y * 1.12, f"median refresh {med_y:.0f} ms", fontsize=7.8, color=INK2, ha="left",
            va="bottom")
    ax.scatter(xs, ys, s=30, color=BLUE, edgecolor=SURFACE, linewidth=1.0, alpha=0.9, zorder=4)
    # one text label per repository cluster, placed by hand around its centre
    place = {"fastapi": (-0.42, 0.40, "right"), "rich": (-0.05, -0.55, "right"), "requests": (-0.30, 0.0, "right"),
             "httpx": (0.24, 0.36, "left")}
    for name, pts in repos.items():
        cx = math.exp(sum(math.log(x) for x, _ in pts) / len(pts))
        cy = math.exp(sum(math.log(y) for _, y in pts) / len(pts))
        dx, dy, ha = place.get(name, (0.3, 0.3, "left"))
        lx, ly = cx * 10 ** dx, cy * 10 ** dy
        ax.annotate(f"{name}\n{len(pts)} task{'s' if len(pts) > 1 else ''}", xy=(cx, cy), xytext=(lx, ly),
                    ha=ha, va="center", fontsize=8.6, color=INK, linespacing=1.2,
                    arrowprops=dict(arrowstyle="-", color=MUTED, lw=0.7, shrinkA=0, shrinkB=5), zorder=5)
    ax.set_xticks([20, 100, 1000]); ax.set_xticklabels(["20", "100", "1,000"], fontsize=8.5)
    ax.set_yticks([2, 10, 100]); ax.set_yticklabels(["2", "10", "100"], fontsize=8.5)
    ax.tick_params(which="minor", length=0); ax.tick_params(which="major", length=3, color=AXIS)
    ax.set_xlabel("Full rebuild of the graph, ms (log scale)", fontsize=9, color=INK2, labelpad=5)
    ax.set_ylabel("Refresh of changed files only, ms (log scale)", fontsize=9, color=INK2, labelpad=5)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)

    # right-hand figures
    X0 = 0.655
    def stat(y, big, small, color=INK, fs=20):
        fig.text(X0, y, big, fontsize=fs, weight="bold", color=color, va="top")
        fig.text(X0, y - (fs * 1.55) / 72 / H, textwrap.fill(small, 40), fontsize=8.4, color=INK2, va="top",
                 linespacing=1.3)
    stat(0.80, f"{n_equal} / {n}", "patched trees where the refreshed index equals a full rebuild: same nodes, "
         "same source texts, same edges", fs=24)
    stat(0.555, f"{med_y:.0f} ms  vs  {med_x:.0f} ms", "median refresh vs median full rebuild (CPU notebook)",
         fs=15)
    stat(0.385, f"{math.floor(r_lo)}× to {math.floor(r_hi)}× faster",
         f"on every one of the {n} tasks; dots below the “10×” line took less than a tenth of "
         "the rebuild time", fs=15)
    fig.text(0.03, 0.955, "Refreshing only the changed files gives the same graph as a full rebuild",
             fontsize=12.6, weight="bold", va="top", color=INK)
    fig.text(0.03, 0.905, textwrap.fill(f"One dot per public task ({n}; {n_applied} patches applied). After the "
             "reference patch, the index is refreshed in place and compared with a graph rebuilt from scratch.", 118),
             fontsize=9.2, va="top", color=INK2, linespacing=1.35)
    save(fig, "fig3_refresh")
    return {"n": n, "applied": n_applied, "equal": n_equal, "median_refresh_ms": med_y, "median_full_ms": med_x,
            "ratio_min": r_lo, "ratio_max": r_hi, "per_repo": {k: len(v) for k, v in repos.items()}}


def main():
    census, replay, probe, equiv = load()
    f1 = fig1(probe, replay)
    st = prevalence(census, replay)
    fig2(st)
    f3 = fig3(equiv, census)
    print(json.dumps({"fig1": f1, "fig2": st, "fig3": f3}, indent=1))


if __name__ == "__main__":
    main()
