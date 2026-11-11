"""Tally probe answers: which option meaning the model picked, per kind x fresh x flag."""
import json, sys, collections
R = [json.loads(l) for l in open(sys.argv[1], encoding="utf-8")]
c = collections.defaultdict(collections.Counter)
for r in R:
    c[(r["model"], r["kind"], "fresh" if r["fresh"] else "stale", "flag" if r["flag"] else "plain")][r["meaning"]] += 1
for k in sorted(c):
    print(k, dict(c[k]))
# what counts as correct in each condition
cc = collections.defaultdict(collections.Counter)
for r in R:
    cc[(r["kind"], r["fresh"])][(r["meaning"], r["correct"])] += 1
print()
for k in sorted(cc):
    print(k, dict(cc[k]))
