"""Build probe items from real post-edit tool answers.

Kind S (substitution): the agent has just added a new function with edit_file; it then asks
get_code_neighbors about it by name; the tool silently answers for a different, existing node.
Kind T (stale text): the agent has just edited function F; a later search_similar_code call returns F
among its results with F's PRE-edit source.

Each item gets a multiple-choice question with three readings; option order is rotated per item.
Usage: python build_probe.py DATA_DIR REPOS_DIR CENSUS_JSONL OUT_JSON
"""
import json, os, sys, random, subprocess, tempfile, shutil
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import census as C
from toolsim import Repo

DATA, REPOS, CENSUS, OUT = sys.argv[1:5]
RS = {"fastapi/fastapi": "fastapi", "Textualize/rich": "rich", "psf/requests": "requests", "encode/httpx": "httpx"}
T = {}
for l in open(os.path.join(DATA, "tasks.jsonl"), encoding="utf-8"):
    d = json.loads(l); T[d["instance_id"]] = d
rng = random.Random(20261110)


def post_tree(t):
    tmp = tempfile.mkdtemp()
    subprocess.run(["git", "init", "-q", tmp], check=True)
    repo = os.path.join(REPOS, RS[t["repo"]])
    for f in C.patch_files(t["patch"]):
        r = subprocess.run(["git", "-C", repo, "show", f"{t['base_commit']}:{f}"], capture_output=True)
        if r.returncode == 0:
            os.makedirs(os.path.dirname(os.path.join(tmp, f)) or tmp, exist_ok=True)
            open(os.path.join(tmp, f), "wb").write(r.stdout)
    p = t["patch"] if t["patch"].endswith("\n") else t["patch"] + "\n"
    open(os.path.join(tmp, "_p.patch"), "w", encoding="utf-8", newline="\n").write(p)
    subprocess.run(["git", "-C", tmp, "apply", "--whitespace=nowarn", "_p.patch"], check=True, capture_output=True)
    return tmp


def clip(s, n=1400):
    return s if len(s) <= n else s[:n] + "\n    ... (truncated)"


S_OPTS = {
    "trust": "The listed callers/callees belong to the helper I just wrote, so I can rely on them for my next step.",
    "stale": "The code graph does not contain my new helper yet; this answer describes a different, older symbol, so I should read the file directly instead.",
    "lost": "My new helper was not saved; I should write it again.",
}
T_OPTS = {
    "lost": "My edit to {f} did not take effect; I should apply it again.",
    "stale": "The search index was built before my edit, so it shows the old code of {f}; my edit is still in the file (I can re-read the file to confirm).",
    "current": "The code shown for {f} is the current version, so I should base my next change on it.",
}

items = []
for line in open(CENSUS, encoding="utf-8"):
    r = json.loads(line)
    t = T[r["id"]]
    key = f"{RS[t['repo']]}_{t['base_commit']}"
    R = Repo(os.path.join(DATA, "graphs", key + ".json"), os.path.join(DATA, "embeddings", key + ".npz"))
    tmp = post_tree(t)
    try:
        post, files_of = {}, {}
        for f in r["files"]:
            pth = os.path.join(tmp, f)
            if not os.path.exists(pth):
                continue
            full = C.module_name(f)
            cands = [full] + [".".join(full.split(".")[k:]) for k in range(1, len(full.split(".")))]
            mod = next((c for c in cands if any(i == c or i.startswith(c + ".") for i in R.ids)), full)
            syms = C.symbols(open(pth, encoding="utf-8", errors="replace").read(), mod) or {}
            post.update(syms)
            for q in syms:
                files_of[q] = f
        edited = [x for x in r["rows"] if x["change"] == "body_changed" and x["in_graph"] and x["kind"] == "func"]
        # Kind S
        for x in r["rows"]:
            if x["change"] != "added" or x["kind"] != "func" or x["symbol"] not in post:
                continue
            bare = x["symbol"].rsplit(".", 1)[-1]
            got, nb = R.neighbors(bare)
            if got is None or not nb:
                continue
            reply = {"status": "ok", "node": bare, "neighbors": [f"{t['repo']};{n}" for n in nb], "count": len(nb), "is_truncated": False}
            # what an up-to-date graph would say: callees the helper really calls + edited callers that call it
            fresh = set()
            for c in post[x["symbol"]].calls:
                for i in R.ids:
                    if i.rsplit(".", 1)[-1] == c and not i.startswith("tests."):
                        fresh.add(i)
            for q2, s2 in post.items():
                if bare in s2.calls and q2 != x["symbol"]:
                    fresh.add(q2)
            fresh = sorted(fresh)[:50]
            if not fresh:
                continue
            fresh_reply = {"status": "ok", "node": bare, "neighbors": [f"{t['repo']};{n}" for n in fresh], "count": len(fresh), "is_truncated": False}
            items.append({"kind": "S", "task": r["id"], "repo": t["repo"], "file": files_of.get(x["symbol"], ""),
                          "symbol": x["symbol"], "asked": bare, "substituted": got,
                          "edit_code": clip(post[x["symbol"]].src), "tool": "get_code_neighbors",
                          "tool_args": {"node": bare}, "reply": reply, "fresh_reply": fresh_reply,
                          "base_commit": t["base_commit"], "problem": t["problem_statement"][:1200]})
            break  # one S item per task
        # Kind T
        done = False
        for x in r["rows"]:
            if done or x["kind"] != "func" or x["change"] not in ("body_changed", "added"):
                continue
            for qq in (".".join(x["symbol"].split(".")[-2:]), x["symbol"].rsplit(".", 1)[-1]):
                _, mode, res = R.similar(qq, k=5)
                hits = [(nm, txt) for nm, txt in res if nm in {e["symbol"] for e in edited} and nm != x["symbol"] and nm in post]
                if hits:
                    f_name, old_txt = hits[0]
                    total = sum(len(tx) for _, tx in res)
                    reply = {"status": "ok", "query": qq, "results": [
                        {"node_name": nm, "code": clip(tx, 900)} for nm, tx in res], "count": len(res), "is_truncated": total > 5000}
                    fresh_reply = {"status": "ok", "query": qq, "results": [
                        {"node_name": nm, "code": clip(post[nm].src if nm == f_name else tx, 900)} for nm, tx in res],
                        "count": len(res), "is_truncated": total > 5000}
                    items.append({"kind": "T", "task": r["id"], "repo": t["repo"], "file": files_of.get(f_name, ""),
                                  "symbol": f_name, "edit_code": clip(post[f_name].src), "old_code": old_txt,
                                  "tool": "search_similar_code", "tool_args": {"query": qq, "k": 5}, "reply": reply,
                                  "fresh_reply": fresh_reply, "search_mode": mode, "base_commit": t["base_commit"],
                                  "problem": t["problem_statement"][:1200]})
                    done = True
                    break
    finally:
        shutil.rmtree(tmp, ignore_errors=True)

for i, it in enumerate(items):
    opts = S_OPTS if it["kind"] == "S" else {k: v.format(f=it["symbol"].rsplit(".", 1)[-1]) for k, v in T_OPTS.items()}
    keys = list(opts)
    rng.shuffle(keys)
    it["options"] = [{"letter": "ABC"[j], "meaning": k, "text": opts[k]} for j, k in enumerate(keys)]
    it["correct_stale"] = next(o["letter"] for o in it["options"] if o["meaning"] == "stale")
    it["correct_fresh"] = next(o["letter"] for o in it["options"] if o["meaning"] == ("trust" if it["kind"] == "S" else "current"))
    it["item_id"] = f"{it['kind']}{i:03d}"
json.dump(items, open(OUT, "w", encoding="utf-8"), indent=1)
from collections import Counter
print(len(items), Counter(i["kind"] for i in items), Counter((i["kind"], i["repo"]) for i in items))
