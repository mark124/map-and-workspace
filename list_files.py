import json, sys
from kaggle.api.kaggle_api_extended import KaggleApi
api = KaggleApi(); api.authenticate()
out, tok = [], None
while True:
    r = api.competition_list_files("gemma-4-developer-agent", page_token=tok, page_size=200)
    for f in r.files:
        out.append((f.name, f.total_bytes))
    tok = r.next_page_token
    if not tok:
        break
json.dump(out, open(sys.argv[1], "w"), indent=0)
from collections import defaultdict
agg = defaultdict(lambda: [0, 0])
for n, s in out:
    k = n.split("/")[0] if "/" in n else n
    agg[k][0] += 1; agg[k][1] += s or 0
for k, (c, s) in sorted(agg.items()):
    print(f"{k:30s} {c:5d} files {s/1e6:10.1f} MB")
