"""Download everything except the 21 GB snapshots (we rebuild those from the public GitHub repos)."""
import json, os, sys, zipfile
from kaggle.api.kaggle_api_extended import KaggleApi
api = KaggleApi(); api.authenticate()
root = sys.argv[1]
files = json.load(open(os.path.join(root, "filelist.json")))
want = [n for n, s in files if not n.startswith("snapshots/")]
for i, n in enumerate(want):
    dest_dir = os.path.join(root, os.path.dirname(n))
    os.makedirs(dest_dir, exist_ok=True)
    final = os.path.join(root, n)
    if os.path.exists(final) and os.path.getsize(final) > 0:
        continue
    api.competition_download_file("gemma-4-developer-agent", n, path=dest_dir, quiet=True)
    z = final + ".zip"
    if os.path.exists(z):
        with zipfile.ZipFile(z) as zf:
            zf.extractall(dest_dir)
        os.remove(z)
    if i % 25 == 0:
        print(i, "/", len(want), n, flush=True)
print("done", len(want))
