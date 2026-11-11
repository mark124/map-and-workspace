"""Build a CPU Kaggle kernel that checks graphrefresh on every task using the competition snapshots."""
import json, os
here = os.path.dirname(os.path.abspath(__file__))
root = os.path.dirname(here)
lib = open(os.path.join(root, "graphrefresh", "graphrefresh.py"), encoding="utf-8").read()
body = r'''
import glob, io, json, os, shutil, subprocess, tarfile, tempfile, time
from pathlib import Path
DATA = Path('/kaggle/input/competitions/gemma-4-developer-agent')
if not DATA.exists():
    DATA = Path(next(iter(glob.glob('/kaggle/input/**/gemma-4-developer-agent', recursive=True))))
res = []
for l in open(DATA / 'tasks.jsonl'):
    t = json.loads(l)
    snap = DATA / 'snapshots' / (t['instance_id'] + '.tgz')
    tmp = tempfile.mkdtemp()
    try:
        with tarfile.open(snap) as tf:
            tf.extractall(tmp, filter='data')
        # the archive holds a repo directory; find the folder that contains .git
        cands = [p for p in Path(tmp).rglob('.git') if p.is_dir()]
        wt = str(cands[0].parent) if cands else tmp
        t0 = time.perf_counter(); idx = Index().build(wt); tb = time.perf_counter() - t0
        p = t['patch'] if t['patch'].endswith('\n') else t['patch'] + '\n'
        pf = os.path.join(tmp, '_p.patch'); open(pf, 'w').write(p)
        r = subprocess.run(['git', 'apply', '--whitespace=nowarn', pf], cwd=wt, capture_output=True)
        applied = r.returncode == 0
        os.remove(pf)
        t0 = time.perf_counter(); changed = idx.refresh(wt); tr = time.perf_counter() - t0
        t0 = time.perf_counter(); full = Index().build(wt); tf_ = time.perf_counter() - t0
        n1, e1 = idx.snapshot(); n2, e2 = full.snapshot()
        same = n1 == n2 and e1 == e2
        res.append({'id': t['instance_id'], 'applied': applied, 'changed_files': len(changed), 'nodes': len(n2),
                    'edges': sum(len(v) for v in e2.values()), 'equal': same, 'base_build_s': round(tb, 4),
                    'refresh_s': round(tr, 4), 'full_s': round(tf_, 4), 'err': r.stderr.decode()[:200] if not applied else ''})
        print(t['instance_id'], applied, same, f'{tr:.3f}s vs {tf_:.3f}s', flush=True)
    finally:
        shutil.rmtree(tmp, ignore_errors=True)
json.dump(res, open('/kaggle/working/equivalence.json', 'w'), indent=1)
ok = [r for r in res if r['applied']]
med = lambda xs: sorted(xs)[len(xs) // 2] if xs else None
print('SUMMARY applied', len(ok), 'of', len(res), 'equal', sum(r['equal'] for r in ok),
      'median refresh', med([r['refresh_s'] for r in ok]), 'median full', med([r['full_s'] for r in ok]))
'''
out = os.path.join(here, "k")
os.makedirs(out, exist_ok=True)
open(os.path.join(out, "check.py"), "w", encoding="utf-8").write(lib + "\n\n" + body)
json.dump({"id": "markrowset/graphrefresh-equivalence", "title": "graphrefresh equivalence", "code_file": "check.py",
           "language": "python", "kernel_type": "script", "is_private": True, "enable_gpu": False, "enable_tpu": False,
           "enable_internet": False, "dataset_sources": [], "competition_sources": ["gemma-4-developer-agent"],
           "kernel_sources": [], "model_sources": []}, open(os.path.join(out, "kernel-metadata.json"), "w"), indent=1)
print("ok")
