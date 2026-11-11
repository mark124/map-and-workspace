"""Ask the OFFICIAL swegemma 0.2.11 graph tools, after each reference edit, about every symbol
the edit touched, and record whether the answer describes the edited code.

Usage: python replay.py DATA_DIR HARNESS_SRC CENSUS_JSONL REPOS_DIR OUT_JSONL
"""
import json, os, sys, subprocess, ast, tempfile, shutil
from types import SimpleNamespace

DATA, HSRC, CENSUS, REPOS, OUT = sys.argv[1:6]
sys.path.insert(0, HSRC)
from swegemma.tools import graph as tg  # official tool wrappers

RS = {"fastapi/fastapi": "fastapi", "Textualize/rich": "rich", "psf/requests": "requests", "encode/httpx": "httpx"}
TASKS = {}
for l in open(os.path.join(DATA, "tasks.jsonl"), encoding="utf-8"):
    d = json.loads(l); TASKS[d["instance_id"]] = d


def ctx_for(t):
    task = SimpleNamespace(repo=t["repo"], base_commit=t["base_commit"], instance_id=t["instance_id"])
    return SimpleNamespace(task=task, repo=None, graph_repo_key=None,
                           graph_dir=os.path.join(DATA, "graphs"), embeddings_dir=os.path.join(DATA, "embeddings"),
                           harness=SimpleNamespace(max_stdout_chars=5000), max_stdout_chars=5000,
                           budget=None, tool_calls=0)


def call(fn, ctx, **kw):
    f = getattr(fn, "__wrapped__", fn)  # skip the budget gate; the tool body is unchanged
    try:
        return json.loads(f(ctx, **kw))
    except Exception as e:
        return {"status": "exception", "error": repr(e)[:200]}


def norm(t):
    return "\n".join(l.rstrip() for l in (t or "").strip().splitlines())


def post_sources(t, files):
    """symbol -> source text after applying the reference patch (no decorators, like the graph)."""
    sys.path.insert(0, os.path.dirname(__file__))
    import census as C
    tmp = tempfile.mkdtemp()
    out = {}
    try:
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
    except Exception:
        shutil.rmtree(tmp, ignore_errors=True)
        raise


def main():
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    import census as C
    out = open(OUT, "w", encoding="utf-8")
    for line in open(CENSUS, encoding="utf-8"):
        r = json.loads(line)
        t = TASKS[r["id"]]
        ctx = ctx_for(t)
        tmp = post_sources(t, r["files"])
        try:
            g = json.load(open(os.path.join(DATA, "graphs", f"{RS[t['repo']]}_{t['base_commit']}.json"), encoding="utf-8"))
            ids = {n["id"] for n in g["nodes"]}
            post = {}
            for f in r["files"]:
                pth = os.path.join(tmp, f)
                if not os.path.exists(pth):
                    continue
                full = C.module_name(f)
                cands = [full] + [".".join(full.split(".")[k:]) for k in range(1, len(full.split(".")))]
                mod = next((c for c in cands if any(i == c or i.startswith(c + ".") for i in ids)), full)
                post.update(C.symbols(open(pth, encoding="utf-8", errors="replace").read(), mod) or {})
            for x in r["rows"]:
                if x["kind"] != "func" and x["change"] != "added":
                    continue
                q_full = x["symbol"]
                parts = q_full.split(".")
                q_short = parts[-1]
                q_cls = ".".join(parts[-2:])
                res = {"id": r["id"], "symbol": q_full, "change": x["change"], "in_graph": x["in_graph"]}
                for form, q in (("dotted", q_full), ("class_method", q_cls), ("bare", q_short)):
                    nb = call(tg.get_code_neighbors, ctx, node=q)
                    ss = call(tg.search_similar_code, ctx, query=q, k=5)
                    codes = [it.get("code", "") for it in ss.get("results", [])]
                    names = [it.get("node_name", "") for it in ss.get("results", [])]
                    ps = post.get(q_full)
                    rec = {"nb_status": nb.get("status"), "nb_count": nb.get("count", 0), "nb_echo": nb.get("node"),
                           "ss_status": ss.get("status"), "ss_count": ss.get("count", 0), "ss_top": names[:3]}
                    if ps is not None:
                        rec["ss_has_post_text"] = any(norm(c) == norm(ps.src) for c in codes)
                    if x["change"] == "body_changed" and x["in_graph"]:
                        node_txt = next((n["text"] for n in g["nodes"] if n["id"] == q_full), None)
                        rec["ss_returns_old_text_of_symbol"] = any(
                            nm.endswith(q_full) and node_txt is not None and norm(c) and norm(node_txt).startswith(norm(c)[:200])
                            for nm, c in zip(names, codes))
                    res[form] = rec
                out.write(json.dumps(res) + "\n")
        finally:
            shutil.rmtree(tmp, ignore_errors=True)
        out.flush()
    out.close()


if __name__ == "__main__":
    main()
