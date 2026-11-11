import json, sys
from collections import Counter
R = [json.loads(l) for l in open(sys.argv[1], encoding="utf-8")]
N = len(R)
def pct(k, n=N): return f"{k}/{n} ({100*k/n:.0f}%)"

funcs = lambda r: [x for x in r["rows"] if x["kind"] == "func"]
t_any = sum(1 for r in R if funcs(r))
t_body_in = sum(1 for r in R if any(x["change"] == "body_changed" and x["in_graph"] for x in funcs(r)))
t_body_out = sum(1 for r in R if any(x["change"] == "body_changed" and not x["in_graph"] for x in funcs(r)))
t_added = sum(1 for r in R if any(x["change"] == "added" for x in r["rows"]))
t_removed = sum(1 for r in R if any(x["change"] == "removed" and x["in_graph"] for x in r["rows"]))
t_calls = sum(1 for r in R if any(x["change"] == "body_changed" and x["in_graph"] and (x["calls_added"] or x["calls_removed"]) for x in funcs(r)))
t_wrongres = sum(1 for r in R if any(x["change"] == "added" and x["resolve_short"][1] != "none" for x in r["rows"]))

print("tasks", N)
print("touch >=1 function/method:", pct(t_any))
print("edit a function the graph knows (tools return pre-edit text):", pct(t_body_in))
print("edit a function the graph lacks (e.g. async):", pct(t_body_out))
print("add >=1 new def/class (tools can't see it):", pct(t_added))
print("  ...and asking for it by short name lands on a DIFFERENT existing node:", pct(t_wrongres))
print("delete a symbol the graph still lists:", pct(t_removed))
print("change the calls inside a known function (neighbors stale):", pct(t_calls))
print("ANY of the above:", pct(sum(1 for r in R if any(
    (x["change"] == "body_changed" and x["in_graph"]) or x["change"] in ("added",) or (x["change"] == "removed" and x["in_graph"]) for x in r["rows"]))))

rows = [x for r in R for x in r["rows"]]
bc = [x for x in rows if x["change"] == "body_changed" and x["in_graph"] and x["kind"] == "func"]
print("\nchanged known functions:", len(bc), "graph text == base source:", sum(x["graph_text_matches_base"] for x in bc))
ad = [x for x in rows if x["change"] == "added"]
print("added symbols:", len(ad), "kinds", Counter(x["kind"] for x in ad), "resolve tiers", Counter(x["resolve_short"][1] for x in ad))
print("examples of wrong resolution:")
for x in [x for x in ad if x["resolve_short"][1] != "none"][:12]:
    print("  ", x["symbol"], "->", x["resolve_short"])
print("per repo:", Counter(r["repo"] for r in R))
