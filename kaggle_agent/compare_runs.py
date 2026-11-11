"""Compare agent runs. Usage: python compare_runs.py OUTDIR [OUTDIR ...]
Each OUTDIR is a downloaded kernel output with graph_calls_<cond>.jsonl and <cond>_<prompt>_task_results.jsonl."""
import glob, json, os, sys
for d in sys.argv[1:]:
    calls = [json.loads(l) for f in glob.glob(os.path.join(d, "graph_calls_*.jsonl")) for l in open(f, encoding="utf-8")]
    tr = [json.loads(l) for f in glob.glob(os.path.join(d, "*_task_results.jsonl")) for l in open(f, encoding="utf-8")]
    res = {t.get("instance_id") or t.get("task_id"): bool(t.get("resolved")) for t in tr}
    after = [c for c in calls if c["changed_files"] and not str(c["changed_files"][0]).startswith("<error")]
    hit = [c for c in after if c.get("query_in_changed") or c.get("stale_results") or c.get("nodes_in_changed")]
    by_tool = {}
    for c in calls:
        by_tool.setdefault(c["tool"], [0, 0])[0] += 1
    for c in after:
        by_tool[c["tool"]][1] += 1
    print(f"== {os.path.basename(d)}")
    print(f"  tasks {len(res)}  resolved {sum(res.values())}  tasks with any graph call {len({c['task'] for c in calls})}")
    print(f"  graph calls {len(calls)}  after first edit {len(after)} (in {len({c['task'] for c in after})} tasks)")
    print(f"  after-edit calls touching changed code {len(hit)} (in {len({c['task'] for c in hit})} tasks)")
    print(f"  name-match substitutions {sum(1 for c in calls if c.get('substituted'))}")
    print(f"  by tool [all, after edit]: {by_tool}")
    hit_tasks = {c['task'] for c in hit}
    print(f"  resolved among tasks with an out-of-date after-edit answer: {sum(res.get(t, False) for t in hit_tasks)}/{len(hit_tasks)}")
    for c in hit[:5]:
        print("   e.g.", c["task"], c["tool"], c.get("query"), "| changed:", c["changed_files"][:3],
              "| stale results:", (c.get("stale_results") or [])[:3])
