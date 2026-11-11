"""Headline table: all edited files vs library code only (excluding docs_src/, scripts/, tests/, docs/)."""
import json, sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
R = [json.loads(l) for l in open(sys.argv[1], encoding="utf-8")]
P = {json.loads(l)["id"]: json.loads(l) for l in open(sys.argv[2], encoding="utf-8")}
NONLIB = ("docs_src/", "scripts/", "tests/", "docs/")


def table(lib_only):
    N = len(R)
    def keep(x): return not (lib_only and x["file"].startswith(NONLIB))
    c = dict.fromkeys(["old_text", "new_unseen", "removed_listed", "calls_changed", "any"], 0)
    for r in R:
        rows = [x for x in r["rows"] if keep(x)]
        a = any(x["change"] == "body_changed" and x["in_graph"] and x["kind"] == "func" for x in rows)
        b = any(x["change"] == "added" for x in rows)
        d = any(x["change"] == "removed" and x["in_graph"] for x in rows)
        e = any(x["change"] == "body_changed" and x["in_graph"] and x["kind"] == "func" and (x["calls_added"] or x["calls_removed"]) for x in rows)
        c["old_text"] += a; c["new_unseen"] += b; c["removed_listed"] += d; c["calls_changed"] += e; c["any"] += (a or b or d)
    syms = {x["symbol"] for r in R for x in r["rows"] if keep(x)}
    subs = sum(1 for p in P.values() if any(s for s in p["substitutions"] if any(
        x["symbol"].endswith(s["asked"]) and x["symbol"] in syms for r in R if r["id"] == p["id"] for x in r["rows"])))
    return N, c, subs


for lib in (False, True):
    N, c, subs = table(lib)
    print("LIBRARY ONLY" if lib else "ALL EDITED FILES", {k: f"{v}/{N} ({100*v/N:.0f}%)" for k, v in c.items()},
          "silent substitution tasks", f"{subs}/{N}")
