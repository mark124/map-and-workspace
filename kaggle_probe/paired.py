"""Per item, no note: does the model give the same answer to the out-of-date and the up-to-date version?
Usage: python paired.py results1.jsonl [results2.jsonl ...]"""
import json, sys, collections
R = [json.loads(l) for f in sys.argv[1:] for l in open(f, encoding="utf-8")]
by = collections.defaultdict(dict)
for r in R:
    by[(r["model"], r["item"], r["flag"])][r["fresh"]] = r
for flag in (False, True):
    print("NOTE" if flag else "NO NOTE")
    for m in sorted({r["model"] for r in R}):
        keys = [k for k in by if k[0] == m and k[2] == flag and len(by[k]) == 2]
        same = sum(by[k][False]["meaning"] == by[k][True]["meaning"] for k in keys)
        both = sum(by[k][False]["correct"] and by[k][True]["correct"] for k in keys)
        stale_ok = sum(by[k][False]["correct"] for k in keys)
        fresh_ok = sum(by[k][True]["correct"] for k in keys)
        print(f"  {m}: items {len(keys)} same-answer {same} both-correct {both} stale-correct {stale_ok} fresh-correct {fresh_ok}")
