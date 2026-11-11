# Stale code-graph probe for the Gemma 4 Developer Agent paper track.
# Part 1: re-check the post-edit tool answers with the OFFICIAL swegemma graph code.
# Part 2: ask each Gemma 4 size (official w4a16 checkpoints, served with vLLM exactly as the
#         host's getting-started notebook does) to interpret real stale / fresh tool answers,
#         with and without a two-field freshness note in the tool reply.
import base64, glob, importlib, json, os, re, shutil, subprocess, sys, time, zlib
from pathlib import Path

os.environ.update({
    'TRANSFORMERS_NO_TF': '1', 'TRANSFORMERS_NO_FLAX': '1', 'TRANSFORMERS_OFFLINE': '1',
    'VLLM_NO_USAGE_STATS': '1', 'VLLM_WORKER_MULTIPROC_METHOD': 'spawn', 'VLLM_ENGINE_READY_TIMEOUT_S': '1200',
    'VLLM_MEMORY_PROFILER_ESTIMATE_CUDAGRAPHS': '1', 'NCCL_DEBUG': 'WARN', 'PYTORCH_CUDA_ALLOC_CONF': 'expandable_segments:True',
    'OTEL_SDK_DISABLED': 'true', 'LITELLM_LOCAL_MODEL_COST_MAP': 'True', 'LITELLM_TELEMETRY': 'False',
})
OUT = Path('/kaggle/working'); OUT.mkdir(parents=True, exist_ok=True)
LOG = open(OUT / 'log.txt', 'a')
def log(*a):
    s = ' '.join(str(x) for x in a); print(s, flush=True); LOG.write(s + '\n'); LOG.flush()

ITEMS = json.loads(zlib.decompress(base64.b64decode('__ITEMS__')).decode('utf-8'))
CENSUS = [json.loads(l) for l in zlib.decompress(base64.b64decode('__CENSUS__')).decode('utf-8').splitlines() if l.strip()]
SYSTEM = zlib.decompress(base64.b64decode('__SYSTEM__')).decode('utf-8')
log('items', len(ITEMS), 'census tasks', len(CENSUS))

# ---------- install the official wheelhouse (same steps as the host notebook) ----------
WH = Path('/kaggle/input/datasets/metric/gemma-4-developer-agent-wheelhouse')
if not WH.exists():
    WH = Path(next(iter(glob.glob('/kaggle/input/**/gemma-4-developer-agent-wheelhouse', recursive=True)), str(WH)))
for pat in ('/usr/local/lib/python*/dist-packages/*cutlass*.pth', '/usr/local/lib/python*/site-packages/*cutlass*.pth'):
    for p in glob.glob(pat):
        try: os.unlink(p)
        except OSError: pass
tmp_whl = Path('/tmp/wheelhouse'); tmp_whl.mkdir(parents=True, exist_ok=True)
for w in WH.glob('*.whl'):
    if 'cutlass' in w.name.lower(): continue
    name = w.name.replace('cu128', '+cu128') if ('cu128' in w.name and '+' not in w.name) else w.name
    t = tmp_whl / name
    if not t.exists():
        try: os.symlink(w, t)
        except OSError: shutil.copyfile(w, t)
subprocess.run([sys.executable, '-m', 'pip', 'install', '-q', '--no-deps', '--force-reinstall', *sorted(str(w) for w in tmp_whl.glob('*.whl'))], check=True)
importlib.invalidate_caches()
log('wheelhouse installed')

DATA = Path('/kaggle/input/competitions/gemma-4-developer-agent')
if not DATA.exists():
    DATA = Path(next(iter(glob.glob('/kaggle/input/**/gemma-4-developer-agent', recursive=True)), str(DATA)))
GRAPHS, EMBS = str(DATA / 'graphs'), str(DATA / 'embeddings')

# ---------- Part 1: official tool answers ----------
from swegemma import graph as sg
check = {'S_items': 0, 'S_same_neighbors': 0, 'S_resolved_elsewhere': 0, 'T_items': 0, 'T_stale_node_returned': 0,
         'added_queries': 0, 'added_resolved_to_other_node': 0}
by_task = {}
for it in ITEMS:
    g = sg.get_graph(repo_name=it['repo'], graph_dir=GRAPHS, embeddings_dir=EMBS, base_commit=it['base_commit'])
    if it['kind'] == 'S':
        check['S_items'] += 1
        nb = sg.get_neighbor(node=it['asked'], graph=g, max_neighbors=50)
        mine = sorted(n.split(';', 1)[1] for n in it['reply']['neighbors'])
        theirs = sorted(str(n).split(';', 1)[-1] for n in nb)
        check['S_same_neighbors'] += int(mine == theirs)
        r = sg.resolve_node_name(it['asked'], g)
        check['S_resolved_elsewhere'] += int(r is not None and r != it['symbol'])
    else:
        check['T_items'] += 1
        res = sg.get_similar_nodes(node=it['tool_args']['query'], repo_name=it['repo'], k=5, graph=g,
                                   graph_dir=GRAPHS, embeddings_dir=EMBS, base_commit=it['base_commit'])
        hit = [x for x in res if x['node_name'] == it['symbol']]
        check['T_stale_node_returned'] += int(bool(hit) and hit[0]['code'].strip() == it['old_code'].strip())
TASKS = {json.loads(l)['instance_id']: json.loads(l) for l in open(DATA / 'tasks.jsonl')}
for r in CENSUS:
    t = TASKS[r['id']]
    g = sg.get_graph(repo_name=t['repo'], graph_dir=GRAPHS, embeddings_dir=EMBS, base_commit=t['base_commit'])
    for x in r['rows']:
        if x['change'] == 'added':
            for q in (x['symbol'].rsplit('.', 1)[-1], '.'.join(x['symbol'].split('.')[-2:])):
                check['added_queries'] += 1
                rr = sg.resolve_node_name(q, g)
                check['added_resolved_to_other_node'] += int(rr is not None)
log('OFFICIAL CHECK', json.dumps(check))
json.dump(check, open(OUT / 'official_check.json', 'w'), indent=1)

# ---------- Part 2: model probe ----------
import requests, torch
from adk_submission import VllmConfig, VllmServer

NOTE_MODE = '__NOTE_MODE__'   # 'commit' = index_commit + files_changed_since_index; 'matched' = name the node that answered
if NOTE_MODE == 'matched':
    ITEMS = [it for it in ITEMS if it['kind'] == 'S']
    log('matched-node mode: S items only', len(ITEMS))

def note(it, fresh):
    if NOTE_MODE == 'matched':
        return {'matched_node': it['symbol'] if fresh else it['substituted']}
    if fresh:
        return {'index_commit': 'working tree (re-indexed after your last edit)', 'files_changed_since_index': []}
    return {'index_commit': it['base_commit'][:7] + ' (task base commit)', 'files_changed_since_index': [it['file']]}

def prompt(it, fresh, flag):
    reply = dict(it['fresh_reply'] if fresh else it['reply'])
    if flag:
        reply.update(note(it, fresh))
    opts = '\n'.join(f"{o['letter']}) {o['text']}" for o in it['options'])
    args = json.dumps(it['tool_args'])
    user = (f"You are evaluating a software engineering task for repository {it['repo']}.\n\nProblem Statement:\n{it['problem']}\n\n"
            f"## Session so far\n"
            f"1. You called edit_file on {it['file']}. Result: {{\"status\": \"ok\"}}. After the edit, the code reads:\n```python\n{it['edit_code']}\n```\n"
            f"2. You called {it['tool']}({args}). Result:\n```json\n{json.dumps(reply, indent=1)}\n```\n\n"
            f"## Question\nBefore your next action, which statement is most likely true?\n{opts}\n\n"
            f"Answer with the single letter on the first line, then one short sentence.")
    return [{'role': 'system', 'content': SYSTEM}, {'role': 'user', 'content': user}]

def letter(text):
    m = re.search(r'\b([ABC])\b', (text or '').strip()[:40])
    return m.group(1) if m else None

MODELS = [m for m in os.environ.get('PROBE_MODELS', '__MODELS__').split(',') if m]
EXTRA = {'gemma-4-12b-it-qat-w4a16-ct': ('--hf-overrides', '{"architectures": ["Gemma4ForCausalLM"], "model_type": "gemma4"}')}
res_f = open(OUT / 'probe_results.jsonl', 'a')
gpu = torch.cuda.device_count() if torch.cuda.is_available() else 1
for mname in MODELS:
    mp = Path(f'/kaggle/input/models/google/gemma-4/other/{mname}/2')
    if not mp.exists():
        cand = glob.glob(f'/kaggle/input/**/{mname}/*', recursive=True)
        mp = Path(cand[0]) if cand else mp
    if not mp.exists():
        log('MISSING MODEL', mname); continue
    tp = 4 if gpu >= 4 else (2 if gpu >= 2 else 1)
    cfg = VllmConfig(model=str(mp), port=8000, host='127.0.0.1', tool_call_parser='gemma4', reasoning_parser='gemma4',
                     max_model_len=16384, dtype='bfloat16' if torch.cuda.is_bf16_supported() else 'auto',
                     gpu_memory_utilization=0.90, enable_auto_tool_choice=True, enable_lora=False, max_loras=1,
                     max_lora_rank=128, scheduler_reserve_full_isl=False, tensor_parallel_size=tp,
                     startup_timeout=60 * 20, extra_args=EXTRA.get(mname, ()))
    t0 = time.time()
    srv = VllmServer(cfg); srv.start()
    base = srv.base_url.rstrip('/')
    if not base.endswith('/v1'): base += '/v1'
    served = requests.get(base + '/models', timeout=60).json()['data'][0]['id']
    log('SERVED', mname, served, 'tp', tp, f'{time.time()-t0:.0f}s')
    for it in ITEMS:
        for fresh in (False, True):
            for flag in (False, True):
                body = {'model': served, 'messages': prompt(it, fresh, flag), 'temperature': 0.0, 'max_tokens': 1536}
                try:
                    j = requests.post(base + '/chat/completions', json=body, timeout=600).json()
                    msg = j['choices'][0]['message']
                    content, reasoning = msg.get('content') or '', msg.get('reasoning_content') or ''
                except Exception as e:
                    content, reasoning = 'ERROR ' + repr(e)[:200], ''
                L = letter(content)
                meaning = next((o['meaning'] for o in it['options'] if o['letter'] == L), None)
                correct = it['correct_fresh'] if fresh else it['correct_stale']
                res_f.write(json.dumps({'model': mname, 'item': it['item_id'], 'kind': it['kind'], 'repo': it['repo'],
                                        'fresh': fresh, 'flag': flag, 'letter': L, 'meaning': meaning,
                                        'correct': L == correct, 'content': content[:600], 'reasoning_chars': len(reasoning)}) + '\n')
                res_f.flush()
    log('DONE', mname, f'{time.time()-t0:.0f}s')
    srv.stop()
    time.sleep(10)

# ---------- summary ----------
from collections import defaultdict
rows = [json.loads(l) for l in open(OUT / 'probe_results.jsonl')]
agg = defaultdict(lambda: [0, 0])
for r in rows:
    k = (r['model'], r['kind'], 'fresh' if r['fresh'] else 'stale', 'flag' if r['flag'] else 'plain')
    agg[k][0] += int(r['correct']); agg[k][1] += 1
for k in sorted(agg):
    log('ACC', *k, f'{agg[k][0]}/{agg[k][1]}')
