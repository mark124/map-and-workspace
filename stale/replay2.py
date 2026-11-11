"""After each reference edit, ask the (faithfully re-implemented) graph tools about the touched symbols.
Usage: python replay2.py DATA_DIR CENSUS_JSONL OUT_JSONL"""
import json, os, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from toolsim import Repo

DATA, CENSUS, OUT = sys.argv[1:4]
RS = {"fastapi/fastapi": "fastapi", "Textualize/rich": "rich", "psf/requests": "requests", "encode/httpx": "httpx"}
T = {}
for l in open(os.path.join(DATA, "tasks.jsonl"), encoding="utf-8"):
    d = json.loads(l); T[d["instance_id"]] = d

out = open(OUT, "w", encoding="utf-8")
for line in open(CENSUS, encoding="utf-8"):
    r = json.loads(line)
    t = T[r["id"]]
    key = f"{RS[t['repo']]}_{t['base_commit']}"
    R = Repo(os.path.join(DATA, "graphs", key + ".json"), os.path.join(DATA, "embeddings", key + ".npz"))
    short_index = {}
    for i in R.ids:
        short_index.setdefault(i.rsplit(".", 1)[-1], set()).add(i)
    edited = {x["symbol"] for x in r["rows"] if x["change"] == "body_changed" and x["in_graph"]}
    rec = {"id": r["id"], "repo": t["repo"], "n_edited_known": len(edited)}
    subs, phantom, missing, stale_code_shown, added_n, removed_listed = [], 0, 0, 0, 0, 0
    for x in r["rows"]:
        q = x["symbol"]; parts = q.split(".")
        forms = {"class_method": ".".join(parts[-2:]), "bare": parts[-1]}
        if x["change"] == "added":
            added_n += 1
            for f, qq in forms.items():
                res, nb = R.neighbors(qq)
                if res is not None:  # the new symbol is not in the graph: any hit is a different node
                    subs.append({"asked": qq, "form": f, "got": res, "neighbors": len(nb)})
        elif x["change"] == "removed" and x["in_graph"]:
            removed_listed += 1
        elif x["change"] == "body_changed" and x["in_graph"] and x["kind"] == "func":
            _, nb = R.neighbors(q)
            outs = {v for v, _ in R.out.get(q, [])}
            for c in x.get("calls_removed", []):
                if any(v.rsplit(".", 1)[-1] == c for v in outs):
                    phantom += 1
            for c in x.get("calls_added", []):
                if c in short_index and not any(v.rsplit(".", 1)[-1] == c for v in outs):
                    missing += 1
        if x["kind"] == "func" and x["change"] in ("body_changed", "added"):
            for f, qq in forms.items():
                _, mode, res = R.similar(qq, k=10)
                stale_code_shown += sum(1 for nm, _ in res if nm in edited and nm != q)
    rec.update(substitutions=subs, phantom_callees=phantom, missing_callees=missing,
               stale_code_in_search=stale_code_shown, added=added_n, removed_still_listed=removed_listed)
    out.write(json.dumps(rec) + "\n")
out.close()

R_ = [json.loads(l) for l in open(OUT, encoding="utf-8")]
N = len(R_)
def p(k): return f"{k}/{N} ({100*k/N:.0f}%)"
print("tasks", N)
print("asking about a NEW symbol silently returns a DIFFERENT node:", p(sum(1 for r in R_ if r["substitutions"])),
      "| substitutions:", sum(len(r["substitutions"]) for r in R_), "of", sum(2 * r["added"] for r in R_), "queries")
print("  by form:", {f: sum(1 for r in R_ for s in r["substitutions"] if s["form"] == f) for f in ("class_method", "bare")})
print("neighbors still list a callee the fix removed:", p(sum(1 for r in R_ if r["phantom_callees"])), "total", sum(r["phantom_callees"] for r in R_))
print("neighbors miss a known callee the fix added:", p(sum(1 for r in R_ if r["missing_callees"])), "total", sum(r["missing_callees"] for r in R_))
print("search results show PRE-edit code of another edited function:", p(sum(1 for r in R_ if r["stale_code_in_search"])))
print("removed symbol still listed:", p(sum(1 for r in R_ if r["removed_still_listed"])))
