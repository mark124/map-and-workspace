"""For every task: build(base) -> apply reference patch -> refresh(); compare with build(post). Times both.
Usage: python check_equivalence.py DATA_DIR REPOS_DIR OUT_JSON"""
import json, os, shutil, subprocess, sys, tempfile, time
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from graphrefresh import Index

DATA, REPOS, OUT = sys.argv[1:4]
RS = {"fastapi/fastapi": "fastapi", "Textualize/rich": "rich", "psf/requests": "requests", "encode/httpx": "httpx"}
res = []
for l in open(os.path.join(DATA, "tasks.jsonl"), encoding="utf-8"):
    t = json.loads(l)
    repo = os.path.join(REPOS, RS[t["repo"]])
    tmp = tempfile.mkdtemp()
    try:
        # export the base tree (python files only keeps it small)
        arch = subprocess.run(["git", "-C", repo, "archive", "--format=tar", t["base_commit"]], capture_output=True, check=True).stdout
        import io, tarfile
        wt = os.path.join(tmp, "wt"); os.makedirs(wt)
        with tarfile.open(fileobj=io.BytesIO(arch)) as tf:
            for m in tf.getmembers():
                if not m.isfile() or ".." in m.name or m.name.startswith("/"):
                    continue  # regular files only
                dest = os.path.join(wt, *m.name.split("/"))
                os.makedirs(os.path.dirname(dest), exist_ok=True)
                with open(dest, "wb") as fh:
                    fh.write(tf.extractfile(m).read())
        t0 = time.perf_counter(); idx = Index().build(wt); t_base = time.perf_counter() - t0
        p = t["patch"] if t["patch"].endswith("\n") else t["patch"] + "\n"
        pf = os.path.join(tmp, "_p.patch"); open(pf, "w", encoding="utf-8", newline="\n").write(p)
        subprocess.run(["git", "apply", "--whitespace=nowarn", "--directory=", pf], cwd=wt, capture_output=True)
        r = subprocess.run(["git", "apply", "--whitespace=nowarn", "--check", "-R", pf], cwd=wt, capture_output=True)
        applied = r.returncode == 0
        t0 = time.perf_counter(); changed = idx.refresh(wt); t_ref = time.perf_counter() - t0
        t0 = time.perf_counter(); full = Index().build(wt); t_full = time.perf_counter() - t0
        n1, e1 = idx.snapshot(); n2, e2 = full.snapshot()
        same = (n1 == n2) and (e1 == e2)
        res.append({"id": t["instance_id"], "applied": applied, "changed_files": len(changed), "nodes": len(n2),
                    "edges": sum(len(v) for v in e2.values()), "equal": same, "refresh_s": round(t_ref, 4),
                    "full_s": round(t_full, 4), "version_differs": idx.version() != Index().build(wt).version() if False else None})
        print(t["instance_id"], "applied", applied, "equal", same, f"refresh {t_ref:.3f}s full {t_full:.3f}s", flush=True)
    finally:
        shutil.rmtree(tmp, ignore_errors=True)
json.dump(res, open(OUT, "w"), indent=1)
ok = [r for r in res if r["applied"]]
print("applied", len(ok), "of", len(res), "| equal", sum(r["equal"] for r in ok),
      "| median refresh", sorted(r["refresh_s"] for r in ok)[len(ok) // 2], "s | median full", sorted(r["full_s"] for r in ok)[len(ok) // 2], "s")
