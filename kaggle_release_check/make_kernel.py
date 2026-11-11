"""CPU Kaggle kernel: run stale/replay2.py with the official-swegemma toolsim (release version) and compare
with our original replay.jsonl. Usage: python make_kernel.py"""
import base64, json, os, zlib
here = os.path.dirname(os.path.abspath(__file__)); root = os.path.dirname(here)
enc = lambda p: base64.b64encode(zlib.compress(open(os.path.join(root, p), "rb").read(), 9)).decode()
ts = "release_toolsim.py" if os.path.exists(os.path.join(root, "release_toolsim.py")) else "stale/toolsim.py"
files = {"toolsim.py": ts, "replay2.py": "stale/replay2.py",
         "census.jsonl": "stale/census.jsonl", "replay_ours.jsonl": "stale/replay.jsonl"}
blob = json.dumps({k: enc(v) for k, v in files.items()})
body = r'''
import base64, glob, importlib, json, os, shutil, subprocess, sys, zlib
from pathlib import Path
FILES = json.loads(%r)
W = Path('/kaggle/working/rc'); W.mkdir(parents=True, exist_ok=True)
for k, v in FILES.items():
    (W / k).write_bytes(zlib.decompress(base64.b64decode(v)))
WH = Path(next(iter(glob.glob('/kaggle/input/**/gemma-4-developer-agent-wheelhouse', recursive=True))))
tmp = Path('/tmp/wh'); tmp.mkdir(exist_ok=True)
for w in WH.glob('*.whl'):
    if 'cutlass' in w.name.lower(): continue
    nm = w.name.replace('cu128', '+cu128') if ('cu128' in w.name and '+' not in w.name) else w.name
    if not (tmp / nm).exists(): shutil.copyfile(w, tmp / nm)
for pat in ('/usr/local/lib/python*/dist-packages/*cutlass*.pth', '/usr/local/lib/python*/site-packages/*cutlass*.pth'):
    for p in glob.glob(pat):
        try: os.unlink(p)
        except OSError: pass
subprocess.run([sys.executable, '-m', 'pip', 'install', '-q', '--no-deps', '--force-reinstall', *sorted(str(p) for p in tmp.glob('*.whl'))], check=True)
importlib.invalidate_caches()
import swegemma; print('swegemma', getattr(swegemma, '__version__', '?'), flush=True)
DATA = Path(next(iter(glob.glob('/kaggle/input/**/gemma-4-developer-agent', recursive=True))))
r = subprocess.run([sys.executable, str(W / 'replay2.py'), str(DATA), str(W / 'census.jsonl'), '/kaggle/working/replay_official.jsonl'],
                   capture_output=True, text=True, cwd=str(W))
print(r.stdout[-3000:]); print(r.stderr[-3000:])
A = {json.loads(l)['id']: json.loads(l) for l in open(W / 'replay_ours.jsonl')}
B = {json.loads(l)['id']: json.loads(l) for l in open('/kaggle/working/replay_official.jsonl')}
keys = ['substitutions', 'phantom_callees', 'missing_callees', 'stale_code_in_search', 'added', 'removed_still_listed']
diff = {k: sum(1 for i in A if i in B and A[i][k] != B[i][k]) for k in keys}
print('TASKS ours', len(A), 'official', len(B), 'DIFFERING TASKS PER FIELD', json.dumps(diff), flush=True)
json.dump({'tasks': len(B), 'differing': diff}, open('/kaggle/working/release_check.json', 'w'))
''' % blob
out = os.path.join(here, "k"); os.makedirs(out, exist_ok=True)
open(os.path.join(out, "check.py"), "w", encoding="utf-8").write(body)
json.dump({"id": "markrowset/release-toolsim-check", "title": "release toolsim check", "code_file": "check.py",
           "language": "python", "kernel_type": "script", "is_private": True, "enable_gpu": False, "enable_tpu": False,
           "enable_internet": False, "dataset_sources": ["metric/gemma-4-developer-agent-wheelhouse"],
           "competition_sources": ["gemma-4-developer-agent"], "kernel_sources": [], "model_sources": [],
           "docker_image": "gcr.io/kaggle-private-byod/python@sha256:37c64f7dd9c54116ecd1bcc88817c5469b88387388fade02bfa8bf3fc647d461"},
          open(os.path.join(out, "kernel-metadata.json"), "w"), indent=1)
print("ok", len(body))
