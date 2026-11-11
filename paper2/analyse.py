# Paper 2: how Gemma 4 agents use their tools in the swegemma harness.
# Reads the 160 ATIF session records (4 runs x 40 tasks) one file at a time, prints every number
# used in DRAFT.md, and writes the derived dataset to paper2/dataset/.
# Run:  python -I analyse.py            (read-only on the run outputs)
import csv, json, os, re, statistics, sys
from collections import Counter, defaultdict
from pathlib import Path

sys.stdout.reconfigure(encoding='utf-8')
HERE = Path(__file__).resolve().parent
KA = HERE.parent / 'kaggle_agent'
RUNS = [  # (run id, results dir, model, prompt, graph-reply note)
    ('stock_12b', KA / 'out_stock_plain' / 'results_plain', '12B', 'stock', False),
    ('check_12b', KA / 'out_check_plain' / 'results_plain', '12B', 'check', False),
    ('checknote_12b', KA / 'out_check_note' / 'results_note', '12B', 'check', True),
    ('check_31b', KA / 'out_31b_check_plain' / 'results_plain', '31B', 'check', False),
]
OUT = HERE / 'dataset'; OUT.mkdir(exist_ok=True)

GRAPH = {'get_code_neighbors', 'search_similar_code', 'get_code_subgraph'}
UNCOUNTED = {'submit_patch', 'get_status', 'code_analyzer_agent'}   # not budget-gated per HARNESS_README 6.1 (sub-agent: checked below)
TESTRE = re.compile(r'\b(pytest|unittest|tox|nox)\b')

def med(xs):
    xs = [x for x in xs if x is not None]
    return round(statistics.median(xs), 1) if xs else None

def pct(a, b):
    return f'{a}/{b} ({100*a/b:.0f}%)' if b else f'{a}/0'

def parse_reply(ob):
    if not ob: return None, ''
    c = ob.get('content')
    raw = c if isinstance(c, str) else json.dumps(c)
    try: j = json.loads(raw)
    except Exception: j = None
    if isinstance(j, dict) and 'status' not in j:
        if 'error' in j:   # argument check by the agent framework, before the tool runs
            e = str(j['error'])
            miss = re.findall(r'^\s*([A-Za-z_]+)\s*$', e.split('not present:', 1)[1].split('You could', 1)[0], re.M) if 'not present:' in e else []
            j = {'status': 'error', 'error_type': 'MissingArgument' if 'mandatory input parameters' in e else 'FrameworkError', 'missing': miss}
        elif 'raw' in j:   # sub-agent's text answer
            j = {'status': 'text'}
    return j, raw

def edit_error_kind(j):
    if not j: return 'no reply'
    if j.get('status') == 'ok': return 'ok'
    et = j.get('error_type'); msg = str(j.get('error_message', ''))
    if et == 'FileEditError':
        if 'not found' in msg: return 'old_string not found'
        if 'occurs' in msg: return 'old_string matches several places'
        if 'identical' in msg: return 'no change'
        return 'other edit error'
    return {'MissingArgument': 'missing argument', 'BudgetExceeded': 'call budget used up'}.get(et, str(et))

def is_scratch(path):
    p = path.replace('\\', '/').lstrip('/')
    if p.startswith('workspace/'): p = p[len('workspace/'):]
    b = p.rsplit('/', 1)[-1]
    return (p.startswith(('tests/', 'test/', 'testing/', 'tmp/', '/tmp')) or b.startswith(('test_', 'repro', 'reproduce', 'debug', 'check_', 'verify'))
            or b.endswith('_test.py') or '/tests/' in p or b in ('conftest.py',))

def norm_path(p):
    p = str(p).replace('\\', '/')
    for pre in ('/workspace/', 'workspace/', './', '/'):
        if p.startswith(pre): p = p[len(pre):]
    return p

def path_mods(p):
    p = norm_path(p)
    if not p.endswith('.py'): return set()
    m = p[:-3].replace('/', '.')
    if m.endswith('.__init__'): m = m[:-9]
    out = {m}
    if '.' in m: out.add(m.split('.', 1)[1])
    return out

def node_in_mods(node, mods):
    n = str(node).split(';', 1)[-1]
    return any(n == m or n.startswith(m + '.') for m in mods)

def query_shape(q):
    q = str(q).strip()
    if re.search(r'\s', q): return 'multi-word keywords'
    if re.fullmatch(r'[A-Za-z_][A-Za-z0-9_]*', q): return 'bare name'
    if re.fullmatch(r'[A-Za-z_][A-Za-z0-9_]*\.[A-Za-z_][A-Za-z0-9_]*', q):
        return 'full dotted path' if q.split('.')[0] in ('fastapi', 'rich', 'requests', 'docs_src', 'tests') else 'Class.method'
    if re.fullmatch(r'[A-Za-z_][A-Za-z0-9_]*(\.[A-Za-z_][A-Za-z0-9_]*){2,}', q): return 'full dotted path'
    return 'other (path, number, symbol)'

def cmd_kind(c):
    c = str(c).strip()
    if TESTRE.search(c): return 'test'
    if re.search(r'sed\s+-i|perl\s+-pi|\bcat\s*>|>\s*\S+\.py|\btee\b|\bpatch\b', c): return 'shell edit'
    if re.match(r'(cd\s+\S+\s*&&\s*)?python3?\b', c): return 'python script'
    if re.match(r'(cd\s+\S+\s*&&\s*)?git\b', c): return 'git'
    if re.match(r'(cd\s+\S+\s*&&\s*)?(grep|rg|find|ls|tree|cat|head|tail|sed|awk|wc|ag)\b', c): return 'search/view'
    if re.match(r'(cd\s+\S+\s*&&\s*)?pip\b', c): return 'pip'
    return 'other'

def bare_pytest(c):
    c = str(c)
    if 'pytest' not in c: return False
    seg = c.split('pytest', 1)[1].split('|')[0].split('&&')[0]
    toks = [t for t in seg.split() if not t.startswith('-')]
    return not any(('/' in t or '.py' in t or '::' in t or t.startswith('test')) for t in toks)

calls_f = open(OUT / 'tool_calls.jsonl', 'w', encoding='utf-8')
sessions = []
graph_rows = []        # per graph call details for section 3
sub_invocations = []   # per sub-agent call
nudge_kinds = defaultdict(Counter)
overflow_sizes = Counter()   # (who, prompt tokens, requested answer tokens, window) from the refusal message
budget_check = defaultdict(Counter)

for run, rdir, model, prompt, note in RUNS:
    tr = {}
    for line in open(rdir / 'task_results.jsonl', encoding='utf-8'):
        r = json.loads(line); tr[r['instance_id']] = r
    for tid in sorted(tr):
        res = tr[tid]
        d = json.load(open(rdir / 'traces' / f'trace_{tid}.json', encoding='utf-8'))
        steps = d['steps']
        calls = []   # flat list of tool calls in order
        out_cap = 0; nudges = 0; ctx_err_steps = 0; llm_calls = 0; max_prompt = 0
        pending_sub = None
        for si, s in enumerate(steps):
            ex = s.get('extra') or {}
            m = s.get('metrics') or {}
            if isinstance(m, dict):
                if (m.get('completion_tokens') or 0) >= 8192: out_cap += 1
                max_prompt = max(max_prompt, m.get('prompt_tokens') or 0)
            llm_calls += int(s.get('llm_call_count') or 0)
            if ex.get('event_type') == 'continuation_nudge':
                nudges += 1
                msg = s.get('message') or ''
                k = 'token limit while thinking' if 'while thinking' in msg else ('tool call cut off' if 'cut off' in msg else 'plain continue')
                nudge_kinds[run][k] += 1
            ob = s.get('observation')
            if s['source'] == 'system' and ob:
                j, raw = parse_reply(ob)
                if j and j.get('error_type') == 'ContextWindowExceededError':
                    ctx_err_steps += 1
                    who = (ob.get('extra') or {}).get('tool_name')
                    mm = re.search(r'maximum context length is (\d+).*?requested (\d+) output tokens.*?at least (\d+) input tokens', raw, re.S)
                    overflow_sizes[(who, mm.group(3), mm.group(2), mm.group(1)) if mm else (who, '?', '?', '?')] += 1
                    if who == 'code_analyzer_agent' and pending_sub is not None:
                        pending_sub['ended'] = 'context overflow'
            tcs = s.get('tool_calls') or []
            for k, t in enumerate(tcs):
                te = t.get('extra') or {}
                author = te.get('author') or ex.get('author')
                j, raw = parse_reply(ob) if (k == len(tcs) - 1) else (None, '')
                c = {'tool': t['function_name'], 'args': t.get('arguments') or {}, 'agent': 'sub' if author == 'code_analyzer_agent' else 'main',
                     'elapsed_s': te.get('elapsed_s', ex.get('elapsed_s')), 'step_id': s.get('step_id'), 'reply': j, 'reply_chars': len(raw),
                     'turn_prompt_tokens': m.get('prompt_tokens') if isinstance(m, dict) else None}
                calls.append(c)
                if c['tool'] == 'code_analyzer_agent':
                    pending_sub = {'run': run, 'task': tid, 'ended': 'returned', 'inner_calls': 0, 'resolved': res['resolved']}
                    sub_invocations.append(pending_sub)
                elif c['agent'] == 'sub' and pending_sub is not None:
                    pending_sub['inner_calls'] += 1
        # ---- per-call derived fields
        first_edit = first_any_edit = None
        for i, c in enumerate(calls):
            if c['tool'] in ('edit_file', 'write_file') and c['reply'] and c['reply'].get('status') == 'ok':
                if first_any_edit is None: first_any_edit = i
                if first_edit is None and not is_scratch(c['args'].get('filepath', '')): first_edit = i
        submit_idx = [i for i, c in enumerate(calls) if c['tool'] == 'submit_patch']
        ok_submit = [i for i in submit_idx if calls[i]['reply'] and calls[i]['reply'].get('status') == 'ok']
        err = res.get('error') or ''
        if 'session timeout' in err: end = 'time limit'
        elif 'tool call budget' in err: end = 'tool-call limit'
        elif 'turns budget' in err: end = 'turn limit'
        elif 'ContextWindow' in err: end = 'context overflow'
        elif ok_submit: end = 'submitted'
        else: end = 'stopped without submitting'
        # budget-gated calls that actually ran: leave out refusals after the limit, framework argument errors, and calls cut off by the time limit
        ran = [c for c in calls if c['tool'] not in UNCOUNTED and c['reply'] and c['reply'].get('error_type') not in ('BudgetExceeded', 'MissingArgument', 'FrameworkError')]
        ran_main = [c for c in ran if c['agent'] == 'main']
        budget_check[run]['match (main + sub-agent)' if len(ran) == res['tool_calls'] else ('match (main only)' if len(ran_main) == res['tool_calls'] else f'off by {len(ran) - res["tool_calls"]}')] += 1
        tests = [i for i, c in enumerate(calls) if c['tool'] == 'run_command' and cmd_kind(c['args'].get('command', '')) == 'test']
        tests_after = [i for i in tests if first_edit is not None and i > first_edit]
        pass_after = [i for i in tests_after if calls[i]['reply'] and calls[i]['reply'].get('status') == 'ok']
        checks_after = [i for i, c in enumerate(calls) if c['tool'] == 'run_command' and first_edit is not None and i > first_edit
                        and cmd_kind(c['args'].get('command', '')) in ('test', 'python script')]
        checks_ok = [i for i in checks_after if calls[i]['reply'] and calls[i]['reply'].get('status') == 'ok']
        tests_before_submit = [i for i in tests_after if not ok_submit or i < ok_submit[0]]
        files_read = {norm_path(c['args'].get('filepath', '')) for c in calls if c['tool'] == 'read_file'}
        edit_fail = sum(1 for c in calls if c['tool'] == 'edit_file' and c['reply'] and c['reply'].get('status') != 'ok')
        # graph calls
        for i, c in enumerate(calls):
            if c['tool'] not in GRAPH: continue
            q = c['args'].get('node') if c['tool'] == 'get_code_neighbors' else (c['args'].get('query') if c['tool'] == 'search_similar_code' else json.dumps(c['args'].get('nodes')))
            j = c['reply'] or {}
            if c['tool'] == 'get_code_neighbors': nodes = list(j.get('neighbors') or []) + [j.get('node') or '']; n = j.get('count')
            elif c['tool'] == 'search_similar_code': nodes = [x.get('node_name', '') for x in (j.get('results') or [])]; n = j.get('count')
            else: nodes = list(j.get('nodes') or []); n = j.get('node_count')
            follow = [x for x in calls[i + 1:i + 4] if x['agent'] == c['agent'] and x['tool'] == 'read_file']
            follow_match = any(node_in_mods(nd, path_mods(x['args'].get('filepath', ''))) for x in follow for nd in nodes if nd)
            graph_rows.append({'run': run, 'model': model, 'task': tid, 'tool': c['tool'], 'agent': c['agent'], 'query': q, 'shape': query_shape(q),
                               'status': j.get('status', 'none'), 'empty': (n == 0), 'n': n, 'after_edit': first_edit is not None and i > first_edit,
                               'read_next3': bool(follow), 'read_returned_file': follow_match, 'resolved': res['resolved']})
        # write call rows
        for i, c in enumerate(calls):
            a = c['args']; j = c['reply'] or {}
            row = {'run': run, 'model': model, 'prompt': prompt, 'graph_note': note, 'task': tid, 'repo': res['repo'], 'call_index': i,
                   'step_id': c['step_id'], 'agent': c['agent'], 'tool': c['tool'], 'elapsed_s': round(c['elapsed_s'], 2) if c['elapsed_s'] is not None else None,
                   'phase': ('before_first_edit' if (first_edit is None or i < first_edit) else ('first_edit' if i == first_edit else 'after_first_edit')),
                   'reply_status': j.get('status') if c['reply'] else None, 'reply_error_type': j.get('error_type'), 'reply_chars': c['reply_chars'],
                   'budget_warning': 'budget_warning' in j, 'turn_prompt_tokens': c['turn_prompt_tokens'],
                   'session_resolved': res['resolved'], 'session_end': end}
            if c['tool'] == 'run_command':
                cmd = str(a.get('command', ''))
                row.update(arg_text=cmd[:400], arg_chars=len(cmd), cmd_kind=cmd_kind(cmd),
                           exit_code=(j.get('exit_code') if 'exit_code' in j else (j.get('details') or {}).get('exit_code')))
            elif c['tool'] in GRAPH:
                q = a.get('node') if c['tool'] == 'get_code_neighbors' else (a.get('query') if c['tool'] == 'search_similar_code' else a.get('nodes'))
                row.update(arg_text=str(q)[:400], arg_chars=len(str(q)), query_shape=query_shape(q),
                           result_count=j.get('count', j.get('node_count')), is_truncated=j.get('is_truncated'))
            elif c['tool'] == 'read_file':
                row.update(filepath=norm_path(a.get('filepath', '')), start_line=a.get('start_line'), end_line=a.get('end_line'), is_truncated=j.get('is_truncated'))
            elif c['tool'] == 'edit_file':
                ostr = str(a.get('old_string', ''))
                row.update(old_has_backslash=chr(92) in ostr, old_has_backslash_before_linebreak=(chr(92) + chr(10)) in ostr,
                           old_has_escaped_quote=(chr(92) + chr(34)) in ostr, old_lines=ostr.count(chr(10)) + 1 if ostr else 0,
                           missing_args=j.get('missing') if j.get('error_type') == 'MissingArgument' else None)
                row.update(filepath=norm_path(a.get('filepath', '')), old_chars=len(str(a.get('old_string', ''))), new_chars=len(str(a.get('new_string', ''))),
                           allow_multiple=a.get('allow_multiple'), edit_strategy=j.get('strategy'), edit_result=edit_error_kind(c['reply']),
                           scratch_file=is_scratch(a.get('filepath', '')))
            elif c['tool'] == 'write_file':
                row.update(filepath=norm_path(a.get('filepath', '')), content_chars=len(str(a.get('content', ''))), scratch_file=is_scratch(a.get('filepath', '')))
            elif c['tool'] == 'code_analyzer_agent':
                row.update(arg_chars=len(json.dumps(a)))
            elif c['tool'] == 'submit_patch':
                row.update(patch_size=j.get('patch_size'), files_changed=j.get('files_changed'))
            calls_f.write(json.dumps(row) + '\n')
        fm = d.get('final_metrics') or {}
        tc = Counter(c['tool'] for c in calls)
        sessions.append({
            'run': run, 'model': model, 'prompt': prompt, 'graph_note': note, 'task': tid, 'repo': res['repo'], 'resolved': res['resolved'],
            'session_end': end, 'harness_error': re.sub(r'/tmp/\S+', '<sandbox path>', err)[:80], 'submitted': bool(ok_submit), 'n_submit_calls': len(submit_idx),
            'patch_chars': res.get('agent_patch_size'), 'harness_tool_calls': res['tool_calls'], 'calls_logged': len(calls),
            'calls_main': sum(c['agent'] == 'main' for c in calls), 'calls_sub': sum(c['agent'] == 'sub' for c in calls),
            **{f'n_{k}': tc.get(k, 0) for k in ['run_command', 'read_file', 'edit_file', 'write_file', 'get_status', 'submit_patch',
                                               'get_code_neighbors', 'search_similar_code', 'get_code_subgraph', 'code_analyzer_agent']},
            'graph_calls': sum(tc.get(k, 0) for k in GRAPH),
            'graph_after_edit': sum(1 for i, c in enumerate(calls) if c['tool'] in GRAPH and first_edit is not None and i > first_edit),
            'calls_to_first_edit': first_edit, 'secs_to_first_edit': round(calls[first_edit]['elapsed_s'], 1) if first_edit is not None else None,
            'calls_to_first_any_edit': first_any_edit,
            'test_runs': len(tests), 'test_runs_after_edit': len(tests_after), 'test_pass_after_edit': len(pass_after),
            'tested_before_submit': bool(tests_before_submit) if ok_submit else None,
            'checks_after_edit': len(checks_after), 'checks_ok_after_edit': len(checks_ok),
            'checked_before_submit': (any(i < ok_submit[0] for i in checks_after)) if ok_submit else None,
            'edit_errors_not_found': sum(1 for c in calls if c['tool'] == 'edit_file' and edit_error_kind(c['reply']) == 'old_string not found'),
            'edit_errors_missing_arg': sum(1 for c in calls if c['tool'] in ('edit_file', 'write_file') and edit_error_kind(c['reply']) == 'missing argument'),
            'bare_pytest': sum(1 for i in tests if bare_pytest(calls[i]['args'].get('command', ''))),
            'files_read': len(files_read), 'edit_failures': edit_fail,
            'scratch_writes': sum(1 for c in calls if c['tool'] == 'write_file' and is_scratch(c['args'].get('filepath', ''))),
            'output_cap_hits': out_cap, 'nudges': nudges, 'context_errors': ctx_err_steps, 'llm_calls': llm_calls,
            'prompt_tokens': fm.get('total_prompt_tokens'), 'completion_tokens': fm.get('total_completion_tokens'), 'total_tokens': fm.get('total_tokens'),
            'max_prompt_tokens_one_call': max_prompt, 'duration_s': res.get('duration_seconds'),
            'agent_secs': round(max((c['elapsed_s'] or 0) for c in calls), 1) if calls else 0,
        })
        del d, steps, calls
calls_f.close()

# per-session table
with open(OUT / 'sessions.csv', 'w', newline='', encoding='utf-8') as f:
    w = csv.DictWriter(f, fieldnames=list(sessions[0].keys())); w.writeheader(); w.writerows(sessions)

RUNIDS = [r[0] for r in RUNS]
by = lambda run: [s for s in sessions if s['run'] == run]
P = print
P('=== 0. Sanity')
P('sessions', len(sessions), 'resolved per run', {r: sum(s['resolved'] for s in by(r)) for r in RUNIDS})
P('harness tool_calls vs our count of budget-gated calls (all agents / main only / neither):', {r: dict(budget_check[r]) for r in RUNIDS})
P('resolved by repo', {r: dict(Counter(s['repo'] for s in by(r) if s['resolved'])) for r in RUNIDS})

P('\n=== 1. Tool mix (all calls, main + sub-agent)')
tools = ['run_command', 'read_file', 'edit_file', 'write_file', 'search_similar_code', 'get_code_neighbors', 'get_code_subgraph', 'get_status', 'submit_patch', 'code_analyzer_agent']
for r in RUNIDS:
    S = by(r); tot = sum(s['calls_logged'] for s in S)
    P(r, 'total calls', tot, 'median/session', med([s['calls_logged'] for s in S]), 'sub-agent inner calls', sum(s['calls_sub'] for s in S))
    P('   ', {t: f"{sum(s['n_'+t] for s in S)} ({100*sum(s['n_'+t] for s in S)/tot:.1f}%)" for t in tools})
    P('    graph share', pct(sum(s['graph_calls'] for s in S), tot), '| sessions with any graph call', pct(sum(s['graph_calls'] > 0 for s in S), 40),
      '| sessions using sub-agent', pct(sum(s['n_code_analyzer_agent'] > 0 for s in S), 40))
allg = Counter();
for s in sessions: allg['total'] += s['calls_logged']; allg['graph'] += s['graph_calls']; allg['subgraph'] += s['n_get_code_subgraph']
P('ALL 160: calls', allg['total'], 'graph', pct(allg['graph'], allg['total']), 'get_code_subgraph', allg['subgraph'])
gw = Counter((g['run'], g['agent']) for g in graph_rows)
P('graph calls by who made them', {r: {'main': gw[(r, 'main')], 'sub': gw[(r, 'sub')]} for r in RUNIDS})
P('\nrun_command kinds (all runs)')
kinds = defaultdict(Counter)
for line in open(OUT / 'tool_calls.jsonl', encoding='utf-8'):
    row = json.loads(line)
    if row['tool'] == 'run_command': kinds[row['run']][row['cmd_kind']] += 1
for r in RUNIDS: P('  ', r, dict(kinds[r].most_common()))
P('sub-agent invocations', len(sub_invocations), 'by run', dict(Counter(x['run'] for x in sub_invocations)),
  'ended', dict(Counter(x['ended'] for x in sub_invocations)), 'median inner calls', med([x['inner_calls'] for x in sub_invocations]),
  'in resolved sessions', sum(x['resolved'] for x in sub_invocations))
submix = Counter(); subfull = Counter()
for line in open(OUT / 'tool_calls.jsonl', encoding='utf-8'):
    row = json.loads(line)
    if row['agent'] == 'sub':
        submix[row['tool']] += 1
        if row['tool'] == 'read_file': subfull['whole file (no start_line)' if row.get('start_line') in (None, 1, '1') else 'from a line'] += 1
P('sub-agent inner calls by tool', dict(submix), '| sub-agent reads', dict(subfull))
P('run_command share that is search/view', {r: pct(kinds[r]['search/view'], sum(kinds[r].values())) for r in RUNIDS})
P('read_file + run_command share', {r: pct(sum(s['n_read_file'] + s['n_run_command'] for s in by(r)), sum(s['calls_logged'] for s in by(r))) for r in RUNIDS})

P('\n=== 2. Phases')
phase = defaultdict(lambda: defaultdict(Counter))
for line in open(OUT / 'tool_calls.jsonl', encoding='utf-8'):
    row = json.loads(line)
    ph = 'before' if row['phase'] == 'before_first_edit' else 'after'
    t = row['tool'] if row['tool'] != 'run_command' else 'run_command:' + row['cmd_kind']
    phase[row['model']][ph][t] += 1
for mdl in ('12B', '31B'):
    for ph in ('before', 'after'):
        tot = sum(phase[mdl][ph].values())
        P(mdl, ph, 'first source edit: total', tot, {k: f'{v} ({100*v/tot:.0f}%)' for k, v in phase[mdl][ph].most_common(8)})
for r in RUNIDS:
    S = by(r); E = [s for s in S if s['calls_to_first_edit'] is not None]
    P(r, 'sessions with a source edit', pct(len(E), 40), 'median calls to first edit', med([s['calls_to_first_edit'] for s in E]),
      'median secs to first edit', med([s['secs_to_first_edit'] for s in E]),
      '| ran tests at all', pct(sum(s['test_runs'] > 0 for s in S), 40), '| ran a test after first edit', pct(sum(s['test_runs_after_edit'] > 0 for s in S), 40),
      '| a test after edit exited 0', pct(sum(s['test_pass_after_edit'] > 0 for s in S), 40),
      '| submitted sessions that tested after edit first', pct(sum(1 for s in S if s['tested_before_submit']), sum(1 for s in S if s['submitted'])),
      '| bare-pytest commands', sum(s['bare_pytest'] for s in S), '| scratch-file writes', sum(s['scratch_writes'] for s in S),
      '| edit_file failures', sum(s['edit_failures'] for s in S), 'of', sum(s['n_edit_file'] for s in S))
strat = defaultdict(Counter); eres = defaultdict(Counter); retry = defaultdict(Counter)
prev = None
for line in open(OUT / 'tool_calls.jsonl', encoding='utf-8'):
    row = json.loads(line)
    if row['tool'] == 'edit_file':
        strat[row['model']][row.get('edit_strategy') or '-'] += 1
        eres[row['model']][row['edit_result']] += 1
        eres[row['run']][row['edit_result']] += 1
    if prev and prev['tool'] == 'edit_file' and prev['edit_result'] == 'old_string not found' and (prev['run'], prev['task']) == (row['run'], row['task']):
        retry[prev['model']][row['tool'] + (':' + row['edit_result'] if row['tool'] == 'edit_file' else '')] += 1
    prev = row
P('edit_file results', {m: dict(eres[m].most_common()) for m in eres})
P('edit_file match strategy when ok', {m: dict(strat[m].most_common()) for m in strat})
P('call right after an old_string-not-found error', {m: dict(retry[m].most_common(6)) for m in retry}, 'totals', {m: sum(retry[m].values()) for m in retry})
bs = defaultdict(Counter)
for line in open(OUT / 'tool_calls.jsonl', encoding='utf-8'):
    row = json.loads(line)
    if row['tool'] == 'edit_file' and row['edit_result'] in ('ok', 'old_string not found'):
        k = (row['model'], row['edit_result'])
        bs[k]['n'] += 1; bs[k]['backslash'] += row['old_has_backslash']; bs[k]['esc_quote'] += row['old_has_escaped_quote']; bs[k]['lines'] += row['old_lines']
P('old_string with a backslash, and mean lines, by result', {f'{k[0]} {k[1]}': f"backslash {v['backslash']}/{v['n']}, escaped double quote {v['esc_quote']}/{v['n']}, mean lines {v['lines']/v['n']:.1f}" for k, v in sorted(bs.items())})
for r in RUNIDS:
    S = by(r)
    P(r, 'sessions with >=1 not-found edit error', pct(sum(s['edit_errors_not_found'] > 0 for s in S), 40),
      '| verification run (test or python) after edit', pct(sum(s['checks_after_edit'] > 0 for s in S), 40),
      '| one exited 0', pct(sum(s['checks_ok_after_edit'] > 0 for s in S), 40),
      '| submitted sessions with a verification run before submit', pct(sum(1 for s in S if s['checked_before_submit']), sum(1 for s in S if s['submitted'])))

P('\n=== 3. Graph tools')
P('graph calls total', len(graph_rows), 'by tool', dict(Counter(g['tool'] for g in graph_rows)))
for tool in ('search_similar_code', 'get_code_neighbors'):
    G = [g for g in graph_rows if g['tool'] == tool]
    P(tool, 'n', len(G))
    for shp, n in Counter(g['shape'] for g in G).most_common():
        H = [g for g in G if g['shape'] == shp]
        P('   ', shp, n, 'empty', sum(g['empty'] for g in H), 'error', sum(g['status'] != 'ok' for g in H))
    P('    any read_file in next 3 calls', pct(sum(g['read_next3'] for g in G), len(G)), '| read the file of a returned node', pct(sum(g['read_returned_file'] for g in G), len(G)))
    P('    after first edit', sum(g['after_edit'] for g in G))
P('example queries by shape:')
ex = defaultdict(list)
for g in graph_rows:
    if len(ex[(g['tool'], g['shape'])]) < 6: ex[(g['tool'], g['shape'])].append(g['query'])
for k, v in ex.items(): P('  ', k, v)
P('repeat queries (same run, task, tool, query asked again):', sum(v - 1 for v in Counter((g['run'], g['task'], g['tool'], g['query']) for g in graph_rows).values() if v > 1))
for r in RUNIDS:
    S = by(r)
    P(r, 'sessions with graph call after first edit', pct(sum(s['graph_after_edit'] > 0 for s in S), 40), 'graph calls after edit', sum(s['graph_after_edit'] for s in S))

P('\n=== 4. How sessions end')
for r in RUNIDS:
    S = by(r)
    P(r, dict(Counter(s['session_end'] for s in S)), '| submitted', pct(sum(s['submitted'] for s in S), 40),
      '| >1 submit', sum(s['n_submit_calls'] > 1 for s in S), '| non-empty patch', pct(sum((s['patch_chars'] or 0) > 0 for s in S), 40),
      '| resolved w/o submit', sum(s['resolved'] and not s['submitted'] for s in S))
    P('    tokens/session median total', med([s['total_tokens'] for s in S]), 'prompt', med([s['prompt_tokens'] for s in S]), 'completion', med([s['completion_tokens'] for s in S]),
      '| max prompt in one call median', med([s['max_prompt_tokens_one_call'] for s in S]), 'max', max(s['max_prompt_tokens_one_call'] for s in S),
      '| LLM calls median', med([s['llm_calls'] for s in S]), '| agent secs median', med([s['agent_secs'] for s in S]))
    P('    answers that hit the 8,192-token cap', sum(s['output_cap_hits'] for s in S), 'in sessions', pct(sum(s['output_cap_hits'] > 0 for s in S), 40),
      '| nudges', dict(nudge_kinds[r]), '| context-overflow errors', sum(s['context_errors'] for s in S), 'in sessions', sum(s['context_errors'] > 0 for s in S))
    sec_per_call = [s['agent_secs'] / s['llm_calls'] for s in S if s['llm_calls']]
    P('    secs per LLM call (median of session means)', med(sec_per_call))

P('\n=== 5. Resolved vs unresolved (descriptive; small samples)')
def describe(S, label):
    n = len(S)
    if not n: P(label, 'n=0'); return
    P(f'{label} n={n}: calls med {med([s["calls_logged"] for s in S])} | calls to first edit med {med([s["calls_to_first_edit"] for s in S])}'
      f' | secs to first edit med {med([s["secs_to_first_edit"] for s in S])} | files read med {med([s["files_read"] for s in S])}'
      f' | test after edit {pct(sum(s["test_runs_after_edit"] > 0 for s in S), n)} | passing test after edit {pct(sum(s["test_pass_after_edit"] > 0 for s in S), n)}'
      f' | verify run after edit {pct(sum(s["checks_after_edit"] > 0 for s in S), n)} | not-found edit errors {pct(sum(s["edit_errors_not_found"] > 0 for s in S), n)}'
      f' | submitted {pct(sum(s["submitted"] for s in S), n)} | hit time/call limit {pct(sum(s["session_end"] in ("time limit", "tool-call limit", "turn limit") for s in S), n)}'
      f' | any graph call {pct(sum(s["graph_calls"] > 0 for s in S), n)} | sub-agent {pct(sum(s["n_code_analyzer_agent"] > 0 for s in S), n)}'
      f' | output-cap hit {pct(sum(s["output_cap_hits"] > 0 for s in S), n)} | tokens med {med([s["total_tokens"] for s in S])}'
      f' | edit failures med {med([s["edit_failures"] for s in S])} | n_edit_file med {med([s["n_edit_file"] for s in S])}')
for mdl in ('12B', '31B'):
    S = [s for s in sessions if s['model'] == mdl]
    describe([s for s in S if s['resolved']], f'{mdl} resolved')
    describe([s for s in S if not s['resolved']], f'{mdl} unresolved')
describe([s for s in sessions if s['resolved']], 'ALL resolved')
describe([s for s in sessions if not s['resolved']], 'ALL unresolved')
P('resolution by end type (all 160):', {e: f"{sum(s['resolved'] for s in sessions if s['session_end']==e)}/{sum(1 for s in sessions if s['session_end']==e)}" for e in sorted({s['session_end'] for s in sessions})})
P('resolution by graph use (all 160): used', f"{sum(s['resolved'] for s in sessions if s['graph_calls']>0)}/{sum(1 for s in sessions if s['graph_calls']>0)}",
  'not used', f"{sum(s['resolved'] for s in sessions if s['graph_calls']==0)}/{sum(1 for s in sessions if s['graph_calls']==0)}")
P('resolution by passing test after edit (all 160): yes', f"{sum(s['resolved'] for s in sessions if s['test_pass_after_edit']>0)}/{sum(1 for s in sessions if s['test_pass_after_edit']>0)}",
  'no', f"{sum(s['resolved'] for s in sessions if s['test_pass_after_edit']==0)}/{sum(1 for s in sessions if s['test_pass_after_edit']==0)}")
P('resolution by output-cap hit (all 160): hit', f"{sum(s['resolved'] for s in sessions if s['output_cap_hits']>0)}/{sum(1 for s in sessions if s['output_cap_hits']>0)}",
  'none', f"{sum(s['resolved'] for s in sessions if s['output_cap_hits']==0)}/{sum(1 for s in sessions if s['output_cap_hits']==0)}")

P('\n=== 5b. Same task, different runs')
tasks = sorted({s['task'] for s in sessions})
solved = {r: {s['task'] for s in by(r) if s['resolved']} for r in RUNIDS}
twelve = [solved[r] for r in RUNIDS[:3]]
cnt = Counter(sum(t in x for x in twelve) for t in tasks)
P('12B: tasks solved in k of 3 runs', dict(sorted(cnt.items())), '| union', len(set().union(*twelve)), '| in all three', len(set.intersection(*twelve)))
P('31B solved', len(solved['check_31b']), '| also solved by some 12B run', len(solved['check_31b'] & set().union(*twelve)),
  '| solved by 12B but not 31B', len(set().union(*twelve) - solved['check_31b']))
P('solved sets', {r: sorted(solved[r]) for r in RUNIDS})
P('distinct tasks solved in any run', len(set().union(*solved.values())), 'from', sum(len(v) for v in solved.values()), 'solves')
P('sessions where the main agent overflowed the window (harness error)', sum('ContextWindow' in s['harness_error'] for s in sessions))

P('\n=== 6. 12B (check_12b) vs 31B (check_31b), same instructions')
for r in ('check_12b', 'check_31b'):
    S = by(r)
    P(r, 'resolved', sum(s['resolved'] for s in S), '| median calls', med([s['calls_logged'] for s in S]), '| edit_file', sum(s['n_edit_file'] for s in S),
      'write_file', sum(s['n_write_file'] for s in S), 'read_file', sum(s['n_read_file'] for s in S), 'run_command', sum(s['n_run_command'] for s in S),
      '| tests run', sum(s['test_runs'] for s in S), '| submitted', sum(s['submitted'] for s in S), '| median completion tokens', med([s['completion_tokens'] for s in S]),
      '| median files read', med([s['files_read'] for s in S]))

P('\n=== 7. Setup facts and extra numbers used in the paper')
import random
ROOT = HERE.parent
tasks_all = [json.loads(l)['instance_id'] for l in open(ROOT / 'data' / 'tasks.jsonl', encoding='utf-8')]
tmpl = (ROOT / 'kaggle_agent' / 'agent_template.py').read_text(encoding='utf-8')
seed = int(re.search(r'^SEED = (\d+)', tmpl, re.M).group(1))
chosen = sorted(random.Random(seed).sample(sorted(tasks_all), 40))
P('public tasks', len(tasks_all), '| seed', seed, '| seed-7 sample equals the 40 run tasks:', chosen == sorted({s['task'] for s in sessions}))
P('repo split of the 40 tasks', dict(Counter(s['repo'] for s in by('stock_12b'))))
P('run limits in agent_template.py:', re.search(r"evaluation:\\n(.*?)'", tmpl).group(1).replace('\\n', ' | '),
  '| answer cap', re.search(r"max_output_tokens: (\d+)'", tmpl).group(1), '| context window', re.search(r'max_model_len=(\d+)', tmpl).group(1))
P('sample sampling.yaml:', (ROOT / 'data' / 'sample_submission' / 'configs' / 'sampling.yaml').read_text(encoding='utf-8').split('\n')[2])
for ad in sorted((ROOT / 'data' / 'sample_submission' / 'adapters').iterdir()):
    cfg = json.load(open(ad / 'adapter_config.json', encoding='utf-8'))
    P('sample adapter', ad.name, 'rank', cfg['r'], 'layers', cfg['layers_to_transform'], 'modules', cfg['target_modules'],
      'file bytes', (ad / 'adapter_model.safetensors').stat().st_size)
for mdl in ('12B', '31B'):
    E = [s['secs_to_first_edit'] for s in sessions if s['model'] == mdl and s['secs_to_first_edit'] is not None]
    meds = [med([s['secs_to_first_edit'] for s in by(r) if s['secs_to_first_edit'] is not None]) for r in RUNIDS if by(r)[0]['model'] == mdl]
    P(mdl, 'median minutes to first edit per run', [round(x / 60, 1) for x in meds])
    ph = phase[mdl]['before']; tot = sum(ph.values()); g = ph['search_similar_code'] + ph['get_code_neighbors'] + ph['get_code_subgraph']
    P(mdl, 'graph calls before first edit', pct(g, tot), f'= {100*g/tot:.1f}%')
ended_sub = [s for s in sessions if s['session_end'] == 'submitted']
ever_sub = [s for s in sessions if s['submitted']]
P('sessions ENDED by submitting', len(ended_sub), '| sessions that ever called submit_patch successfully', len(ever_sub),
  '| called submit_patch but ended another way', dict(Counter(s['session_end'] for s in ever_sub if s['session_end'] != 'submitted')))
P('ever submitted per run', {r: sum(s['submitted'] for s in by(r)) for r in RUNIDS}, '| ended by submitting per run', {r: sum(s['session_end'] == 'submitted' for s in by(r)) for r in RUNIDS})
for mdl in ('12B', '31B'):
    S = [s for s in sessions if s['model'] == mdl]
    P(mdl, 'sessions with >=1 answer at the cap', pct(sum(s['output_cap_hits'] > 0 for s in S), len(S)), '| answers at cap', sum(s['output_cap_hits'] for s in S),
      '| ended at a limit', pct(sum(s['session_end'] in ('time limit', 'tool-call limit', 'turn limit') for s in S), len(S)))
P('context-overflow refusals inside sessions (who, prompt tokens, answer tokens requested, window):', dict(overflow_sizes))
miss = Counter(); bsl = Counter()
for line in open(OUT / 'tool_calls.jsonl', encoding='utf-8'):
    row = json.loads(line)
    if row['tool'] == 'edit_file':
        if row.get('missing_args'): miss[(row['model'], ','.join(row['missing_args']))] += 1
        if row['edit_result'] == 'old_string not found':
            bsl[(row['run'], 'not found')] += 1; bsl[(row['run'], 'backslash before line break')] += row['old_has_backslash_before_linebreak']
P('edit_file missing-argument errors by missing argument', dict(miss))
P('old_string-not-found edits containing a backslash right before a line break (presence only; check_escape.py compares with the files)',
  {r: f"{bsl[(r, 'backslash before line break')]}/{bsl[(r, 'not found')]}" for r in RUNIDS})
P('files_read = distinct paths passed to read_file; calls_to_first_edit = number of calls made before the first-edit call')

P('\nwrote dataset/tool_calls.jsonl and dataset/sessions.csv')
