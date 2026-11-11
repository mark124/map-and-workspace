"""Stale-map census.

For each task: take the files the reference patch touches, at base_commit (from the public git
clone) and after applying the patch. Parse both with Python's ast (sync AND async defs).
Compare with the released graph (built at base_commit). For every symbol the fix touches,
record what the provided graph tools would answer AFTER the edit:

  body_changed   symbol exists in graph, its source changed  -> tools return the old text
  added          new symbol                                  -> not in graph; resolve may hit a different node
  removed        symbol deleted by the fix                   -> tools still return it
  calls_added    callee names added inside a changed symbol  -> get_code_neighbors misses them
  calls_removed  callee names removed                        -> get_code_neighbors still lists them

Usage: python census.py DATA_DIR REPOS_DIR OUT_JSONL
"""
import ast, json, os, subprocess, sys, tempfile, shutil
from collections import defaultdict

DATA, REPOS, OUT = sys.argv[1], sys.argv[2], sys.argv[3]
REPO_DIR = {"fastapi/fastapi": "fastapi", "Textualize/rich": "rich",
            "psf/requests": "requests", "encode/httpx": "httpx"}


def git(repo, *args, check=True):
    r = subprocess.run(["git", "-C", repo, *args], capture_output=True)
    if check and r.returncode:
        raise RuntimeError(r.stderr.decode(errors="replace"))
    return r.stdout


def patch_files(patch):
    files = []
    for line in patch.splitlines():
        if line.startswith("+++ ") or line.startswith("--- "):
            p = line[4:].split("\t")[0].strip()
            if p == "/dev/null":
                continue
            if p[:2] in ("a/", "b/"):
                p = p[2:]
            if p not in files:
                files.append(p)
    return files


def module_name(path):
    p = path[:-3] if path.endswith(".py") else path
    if p.startswith("src/"):
        p = p[4:]
    parts = p.split("/")
    if parts[-1] == "__init__":
        parts = parts[:-1]
    return ".".join(parts)


class Sym:
    __slots__ = ("qual", "src", "calls", "is_async", "kind")


def symbols(source, mod):
    """qualified name -> Sym for every module-level and nested def/class."""
    out = {}
    try:
        tree = ast.parse(source)
    except SyntaxError:
        return None
    lines = source.splitlines(keepends=True)

    def seg(n):
        # the released graph stores a definition WITHOUT its decorators
        return "".join(lines[n.lineno - 1:n.end_lineno])

    def calls_in(n):
        c = set()
        for x in ast.walk(n):
            if isinstance(x, ast.Call):
                f = x.func
                if isinstance(f, ast.Name):
                    c.add(f.id)
                elif isinstance(f, ast.Attribute):
                    c.add(f.attr)
        return c

    def visit(node, prefix):
        for ch in ast.iter_child_nodes(node):
            if isinstance(ch, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
                q = prefix + "." + ch.name
                s = Sym()
                s.qual, s.src = q, seg(ch)
                s.calls = calls_in(ch) if not isinstance(ch, ast.ClassDef) else set()
                s.is_async = isinstance(ch, ast.AsyncFunctionDef)
                s.kind = "class" if isinstance(ch, ast.ClassDef) else "func"
                out[q] = s
                visit(ch, q)
    visit(tree, mod)
    return out


def norm(t):
    return "\n".join(l.rstrip() for l in (t or "").strip().splitlines())


def resolve(name, ids, lower):
    """The harness's documented 4-tier resolution: exact -> . ; / suffix -> case-insensitive -> substring."""
    if name in ids:
        return name, "exact"
    suf = [i for i in ids if i.endswith("." + name) or i.endswith(";" + name) or i.endswith("/" + name)]
    if suf:
        return sorted(suf)[0], "suffix"
    if name.lower() in lower:
        return lower[name.lower()], "case"
    sub = [i for i in ids if name in i]
    if sub:
        return sorted(sub)[0], "substring"
    return None, "none"


def main():
    tasks = [json.loads(l) for l in open(os.path.join(DATA, "tasks.jsonl"), encoding="utf-8")]
    out = open(OUT, "w", encoding="utf-8")
    for t in tasks:
        rid, repo_short = t["instance_id"], REPO_DIR[t["repo"]]
        repo = os.path.join(REPOS, repo_short)
        rec = {"id": rid, "repo": t["repo"]}
        gpath = os.path.join(DATA, "graphs", f"{repo_short}_{t['base_commit']}.json")
        if not os.path.exists(gpath):
            gpath = os.path.join(DATA, "graphs", rid + ".json")
        if not os.path.exists(gpath):
            rec["error"] = "no graph"; out.write(json.dumps(rec) + "\n"); continue
        g = json.load(open(gpath, encoding="utf-8"))
        nodes = {n["id"]: n for n in g["nodes"]}
        ids = list(nodes)
        lower = {i.lower(): i for i in ids}
        succ = defaultdict(set)
        for e in g.get("edges", g.get("links", [])):
            succ[e["source"]].add(e["target"])
        rec["graph_nodes"], rec["graph_edges"] = len(nodes), sum(len(v) for v in succ.values())

        all_files = patch_files(t["patch"])
        files = [f for f in all_files if f.endswith(".py")]
        tmp = tempfile.mkdtemp()
        try:
            subprocess.run(["git", "init", "-q", tmp], check=True)
            for f in all_files:
                try:
                    data = git(repo, "show", f"{t['base_commit']}:{f}")
                except RuntimeError:
                    data = None  # new file
                if data is not None:
                    os.makedirs(os.path.dirname(os.path.join(tmp, f)) or tmp, exist_ok=True)
                    open(os.path.join(tmp, f), "wb").write(data)
            pf = os.path.join(tmp, "_fix.patch")
            ptxt = t["patch"] if t["patch"].endswith("\n") else t["patch"] + "\n"
            open(pf, "w", encoding="utf-8", newline="\n").write(ptxt)
            r = subprocess.run(["git", "-C", tmp, "apply", "--whitespace=nowarn", "_fix.patch"], capture_output=True)
            if r.returncode:
                rec["error"] = "apply failed: " + r.stderr.decode(errors="replace")[:200]
                out.write(json.dumps(rec) + "\n"); continue
            rows = []
            for f in files:
                full = module_name(f)
                # the graph names some trees without their top folder (e.g. docs_src/x/y.py -> x.y)
                cands = [full] + [".".join(full.split(".")[k:]) for k in range(1, len(full.split(".")))]
                mod = next((c for c in cands if any(i == c or i.startswith(c + ".") for i in ids)), full)
                try:
                    base_src = git(repo, "show", f"{t['base_commit']}:{f}").decode("utf-8", "replace")
                except RuntimeError:
                    base_src = ""
                p = os.path.join(tmp, f)
                post_src = open(p, encoding="utf-8", errors="replace").read() if os.path.exists(p) else ""
                b, a = symbols(base_src, mod) or {}, symbols(post_src, mod) or {}
                for q in sorted(set(b) | set(a)):
                    sb, sa = b.get(q), a.get(q)
                    if sb and sa and norm(sb.src) == norm(sa.src):
                        continue
                    # only innermost changes: skip a class whose change is entirely inside a changed method
                    row = {"file": f, "symbol": q, "in_graph": q in nodes,
                           "async": bool((sa or sb).is_async), "kind": (sa or sb).kind}
                    if sb and sa:
                        row["change"] = "body_changed"
                        row["graph_text_matches_base"] = q in nodes and norm(nodes[q].get("text")) == norm(sb.src)
                        row["calls_added"] = sorted(sa.calls - sb.calls)
                        row["calls_removed"] = sorted(sb.calls - sa.calls)
                    elif sa:
                        row["change"] = "added"
                        short = q.rsplit(".", 1)[-1]
                        hit, tier = resolve(short, ids, lower)
                        row["resolve_short"] = [hit, tier]
                        row["calls"] = sorted(sa.calls)
                    else:
                        row["change"] = "removed"
                        row["graph_callers"] = sum(1 for s, ts in succ.items() if q in ts)
                    rows.append(row)
            rec["files"] = files
            rec["rows"] = rows
        finally:
            shutil.rmtree(tmp, ignore_errors=True)
        out.write(json.dumps(rec) + "\n")
        out.flush()
    out.close()


if __name__ == "__main__":
    main()
