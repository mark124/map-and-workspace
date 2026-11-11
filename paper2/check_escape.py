"""For failed edit_file calls ("old_string not found"), test whether undoing one level of escaping
(\\" -> ", \\n -> newline, \\t -> tab, \\\\ -> \\) makes old_string appear in the file at the task's base commit.
Usage: python check_escape.py TRACES_ROOT REPOS_DIR DATA_DIR"""
import glob, json, os, subprocess, sys
ROOT, REPOS, DATA = sys.argv[1:4]
RS = {"fastapi/fastapi": "fastapi", "Textualize/rich": "rich", "psf/requests": "requests", "encode/httpx": "httpx"}
T = {json.loads(l)["instance_id"]: json.loads(l) for l in open(os.path.join(DATA, "tasks.jsonl"), encoding="utf-8")}
cache = {}
def base_file(tid, path):
    k = (tid, path)
    if k not in cache:
        t = T[tid]
        r = subprocess.run(["git", "-C", os.path.join(REPOS, RS[t["repo"]]), "show", f"{t['base_commit']}:{path}"],
                           capture_output=True)
        cache[k] = r.stdout.decode("utf-8", "replace") if r.returncode == 0 else None
    return cache[k]
def unescape(s):
    return s.replace('\\"', '"').replace("\\'", "'").replace("\\n", "\n").replace("\\t", "\t").replace("\\\\", "\\")
stats = {}
for run in sorted(glob.glob(os.path.join(ROOT, "*"))):
    rn = os.path.basename(run)
    st = stats.setdefault(rn, {"failed": 0, "file_found": 0, "raw_in_base": 0, "unescaped_in_base": 0, "has_backslash": 0,
                               "stray_eol_backslash": 0, "fixed_by_dropping_eol_backslash": 0})
    for tf in glob.glob(os.path.join(run, "traces", "trace_*.json")):
        tid = os.path.basename(tf)[6:-5]
        steps = json.load(open(tf, encoding="utf-8")).get("steps", [])
        for s in steps:
            calls = s.get("tool_calls") or []
            obs = json.dumps(s.get("observation") or "")
            for c in calls:
                if c.get("function_name") != "edit_file":
                    continue
                if "not found" not in obs.lower():
                    continue
                a = c.get("arguments") or {}
                old, path = a.get("old_string"), a.get("filepath") or a.get("path")
                if not old or not path:
                    continue
                st["failed"] += 1
                st["has_backslash"] += "\\" in old
                src = base_file(tid, path.lstrip("/").replace("workspace/", "", 1))
                if src is None:
                    continue
                st["file_found"] += 1
                st["raw_in_base"] += old in src
                st["unescaped_in_base"] += ("\\" in old) and (unescape(old) in src)
                eol = "\\\n" in old and "\\\n" not in src
                st["stray_eol_backslash"] += eol
                st["fixed_by_dropping_eol_backslash"] += eol and (old.replace("\\\n", "\n") in src)
for rn, st in stats.items():
    print(rn, st)
