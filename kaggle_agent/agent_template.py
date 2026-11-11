# Real-agent run for the stale code-graph paper (Gemma 4 Developer Agent paper track).
# Follows the host's getting-started notebook step by step (wheelhouse install, sample_submission,
# vLLM server, Evaluator with the subprocess sandbox). Our only change: the three graph tools are
# wrapped. In BOTH conditions the wrapper logs every graph call together with the files the agent has
# changed so far. In condition "note" it also adds two fields to the reply (index_commit,
# files_changed_since_index), names the node that actually answered when the resolver fell back to a
# different node, and marks search results whose source file the agent has changed.
import glob, importlib, json, logging, os, random, re, shutil, subprocess, sys, time, asyncio
from pathlib import Path

COND = '__COND__'            # 'plain' or 'note'
MODEL_NAME = '__MODEL__'
N_TASKS = int('__NTASKS__')
SEED = 7

os.environ.update({
    'TRANSFORMERS_NO_TF': '1', 'TRANSFORMERS_NO_FLAX': '1', 'TRANSFORMERS_OFFLINE': '1', 'VLLM_NO_USAGE_STATS': '1',
    'VLLM_WORKER_MULTIPROC_METHOD': 'spawn', 'VLLM_ENGINE_READY_TIMEOUT_S': '1200', 'VLLM_MEMORY_PROFILER_ESTIMATE_CUDAGRAPHS': '1',
    'NCCL_DEBUG': 'WARN', 'LITELLM_LOCAL_MODEL_COST_MAP': 'True', 'LITELLM_TELEMETRY': 'False', 'LITELLM_LOG': 'WARNING',
    'OPENAI_API_KEY': os.environ.get('OPENAI_API_KEY', 'EMPTY'), 'LOCAL_API_KEY': os.environ.get('LOCAL_API_KEY', 'EMPTY'),
    'PYTORCH_CUDA_ALLOC_CONF': 'expandable_segments:True', 'OTEL_SDK_DISABLED': 'true', 'OTEL_PYTHON_DISABLED': 'true',
})
for name in ('LiteLLM', 'LiteLLM Router', 'LiteLLM Proxy', 'litellm', 'httpx', 'httpcore', 'openai', 'urllib3', 'opentelemetry'):
    logging.getLogger(name).setLevel(logging.WARNING)

WORK = Path('/kaggle/working'); WORK.mkdir(parents=True, exist_ok=True)
LOG = open(WORK / 'log.txt', 'a')
def log(*a):
    s = ' '.join(str(x) for x in a); print(s, flush=True); LOG.write(s + '\n'); LOG.flush()
log('COND', COND, 'MODEL', MODEL_NAME, 'N', N_TASKS)

# ---------- wheelhouse (host notebook cell 2) ----------
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
    nm = w.name.replace('cu128', '+cu128') if ('cu128' in w.name and '+' not in w.name) else w.name
    t = tmp_whl / nm
    if not t.exists():
        try: os.symlink(w, t)
        except OSError: shutil.copyfile(w, t)
subprocess.run([sys.executable, '-m', 'pip', 'install', '-q', '--no-deps', '--force-reinstall', *sorted(str(w) for w in tmp_whl.glob('*.whl'))], check=True)
importlib.invalidate_caches()
log('wheelhouse installed')

# ---------- submission (host notebook cell 4) ----------
from swegemma.models import load_tasks
DATA = Path('/kaggle/input/competitions/gemma-4-developer-agent')
if not DATA.exists():
    DATA = Path(next(iter(glob.glob('/kaggle/input/**/gemma-4-developer-agent', recursive=True))))
AGENT_DIR = WORK / 'sample_submission'
shutil.rmtree(AGENT_DIR, ignore_errors=True)
shutil.copytree(DATA / 'sample_submission', AGENT_DIR)
if MODEL_NAME != 'gemma-4-31b-it-qat-w4a16-ct':
    for yp in [AGENT_DIR / 'agent.yaml', *sorted((AGENT_DIR / 'sub_agents').glob('*.yaml'))]:
        tx = yp.read_text(encoding='utf-8')
        tx = re.sub(r'^(\s*model:\s*).*$', rf'\g<1>{MODEL_NAME}', tx, flags=re.MULTILINE)
        tx = re.sub(r'^\s*adapter:\s*.*\n?', '', tx, flags=re.MULTILINE)
        yp.write_text(tx, encoding='utf-8')
    shutil.rmtree(AGENT_DIR / 'adapters', ignore_errors=True)
PROMPT = '__PROMPT__'        # 'stock' or 'check' (adds one post-edit impact-check instruction)
if PROMPT == 'check':
    sp = AGENT_DIR / 'prompts' / 'system.md'
    sp.write_text(sp.read_text(encoding='utf-8').rstrip() + '\n- After you edit a function, call `get_code_neighbors` on it (and `search_similar_code` if useful) to check its callers and related code before you submit.\n', encoding='utf-8')
# cap output tokens so prompt + output fits the 32,768-token window (sample asks for 16,384, which overflowed in our smoke test)
for yp in (AGENT_DIR / 'configs').glob('*.yaml'):
    yp.write_text(re.sub(r'max_output_tokens:\s*\d+', 'max_output_tokens: 8192', yp.read_text(encoding='utf-8')), encoding='utf-8')
log('prompt', PROMPT)
# realistic budgets (the sample's eval_config is a 1-minute smoke test)
(AGENT_DIR / 'eval_config.yaml').write_text('evaluation:\n  timeout_seconds: 120\n  max_tool_calls: 50\n  max_time_minutes: 20\n  max_turns: 60\n', encoding='utf-8')
TASKS_PATH = DATA / 'tasks.jsonl'
tasks = load_tasks(TASKS_PATH)
ids = sorted(t.instance_id for t in tasks)
chosen = sorted(random.Random(SEED).sample(ids, min(N_TASKS, len(ids))))
log('tasks', len(chosen), json.dumps(chosen))
GRAPH_DIR, EMB_DIR = str(DATA / 'graphs'), str(DATA / 'embeddings')

# ---------- the wrapper (our only change) ----------
import swegemma.tools.graph as tg
from swegemma import graph as sg
CALLS = open(WORK / f'graph_calls_{COND}.jsonl', 'a')
_orig = {'get_code_neighbors': tg.get_code_neighbors, 'search_similar_code': tg.search_similar_code,
         'get_code_subgraph': tg.get_code_subgraph}

def _task_info(ctx):
    task = getattr(ctx, 'task', None)
    g = (lambda k: task.get(k) if isinstance(task, dict) else getattr(task, k, None)) if task else (lambda k: None)
    return (getattr(ctx, 'repo', None) or g('repo')), g('base_commit'), (g('instance_id') or getattr(ctx, 'task_id', ''))

def _changed(ctx):
    try:
        r = ctx.exec('cd /workspace && (git diff --name-only HEAD 2>/dev/null; git ls-files --others --exclude-standard 2>/dev/null) || true', timeout=30)
        return sorted({l.strip() for l in (getattr(r, 'stdout', '') or '').splitlines() if l.strip().endswith('.py')})
    except Exception as e:
        return ['<error ' + repr(e)[:80] + '>']

def _mods(files):
    out = set()
    for f in files:
        if not f.endswith('.py'): continue
        m = f[:-3].replace('/', '.')
        if m.endswith('.__init__'): m = m[:-9]
        out.add(m)
        if '.' in m: out.add(m.split('.', 1)[1])      # docs_src/... is indexed without its top folder
    return out

def _in_changed(name, mods):
    n = str(name).split(';', 1)[-1]
    return any(n == m or n.startswith(m + '.') for m in mods)

def _resolved(ctx, repo, base, q):
    try:
        gr = sg.get_graph(repo_name=repo, graph_dir=GRAPH_DIR, embeddings_dir=EMB_DIR, base_commit=base)
        r = sg.resolve_node_name(q if ';' in str(q) else f'{repo};{q}', gr)
        return None if r is None else str(r).split(';', 1)[-1]
    except Exception:
        return None

def _wrap(tool):
    orig = _orig[tool]
    def w(ctx, *a, **k):
        raw = orig(ctx, *a, **k)
        try:
            repo, base, tid = _task_info(ctx)
            files = _changed(ctx); mods = _mods(files)
            rec = {'cond': COND, 'task': tid, 't': time.time(), 'tool': tool, 'args': [str(x)[:200] for x in a],
                   'changed_files': files}
            try: j = json.loads(raw)
            except Exception: j = None
            if isinstance(j, dict) and j.get('status') == 'ok':
                if tool == 'get_code_neighbors':
                    q = str(a[0]) if a else str(k.get('node'))
                    rn = _resolved(ctx, repo, base, q)
                    exact = rn is not None and (rn == q.split(';', 1)[-1])
                    rec.update(query=q, resolved=rn, substituted=bool(rn) and not exact and not rn.endswith('.' + q.split(';', 1)[-1]),
                               query_in_changed=_in_changed(rn or q, mods))
                    if COND == 'note' and rn and not exact:
                        j['resolved_node'] = rn
                elif tool == 'search_similar_code':
                    hits = [x.get('node_name') for x in j.get('results', []) if _in_changed(x.get('node_name', ''), mods)]
                    rec.update(query=str(a[0]) if a else str(k.get('query')), stale_results=hits, n_results=len(j.get('results', [])))
                    if COND == 'note':
                        for x in j.get('results', []):
                            if _in_changed(x.get('node_name', ''), mods):
                                x['source_may_be_outdated'] = True
                else:
                    ns = j.get('nodes', [])
                    rec.update(nodes_in_changed=sum(_in_changed(n, mods) for n in ns), n_nodes=len(ns))
                if COND == 'note':
                    j['index_commit'] = (str(base)[:7] if base else '?') + ' (task base commit; does not include your edits)'
                    j['files_changed_since_index'] = files
                raw = json.dumps(j)
            CALLS.write(json.dumps(rec) + '\n'); CALLS.flush()
        except Exception as e:
            log('WRAP ERROR', tool, repr(e)[:200])
        return raw
    w.__name__ = getattr(orig, '__name__', tool); w.__doc__ = getattr(orig, '__doc__', None)
    return w

for tname in _orig:
    setattr(tg, tname, _wrap(tname))
log('graph tools wrapped, cond', COND)

# ---------- vLLM (host notebook cell 8) ----------
import litellm, torch
from adk_submission import VllmConfig, VllmServer, discover_adapters
from adk_submission.discovery import discover_declared_models
from swegemma.config import ALLOWED_ADAPTER_EXTENSIONS
litellm.drop_params = True
declared = next(iter(discover_declared_models(AGENT_DIR)))
MODEL_PATH = Path(f'/kaggle/input/models/google/gemma-4/other/{declared}/2')
if not MODEL_PATH.exists():
    c = glob.glob(f'/kaggle/input/**/{declared}/*', recursive=True); MODEL_PATH = Path(c[0]) if c else MODEL_PATH
adapters = discover_adapters(str(AGENT_DIR), adapter_extensions=ALLOWED_ADAPTER_EXTENSIONS)
amap = getattr(adapters, 'adapters', None) or {}
ranks = [r for i in amap.values() if (r := getattr(i, 'rank', None)) is not None and r > 0]
gpu = torch.cuda.device_count() if torch.cuda.is_available() else 1
log('gpus', gpu, [torch.cuda.get_device_name(i) for i in range(gpu)] if torch.cuda.is_available() else 'cpu')
tp = 4 if gpu >= 4 else (2 if gpu >= 2 else 1)
EXTRA = {'gemma-4-12b-it-qat-w4a16-ct': ('--hf-overrides', '{"architectures": ["Gemma4ForCausalLM"], "model_type": "gemma4"}')}
cfg = VllmConfig(model=str(MODEL_PATH), port=8000, host='127.0.0.1', tool_call_parser='gemma4', reasoning_parser='gemma4',
                 max_model_len=32768, dtype='bfloat16' if (torch.cuda.is_available() and torch.cuda.is_bf16_supported()) else 'auto',
                 gpu_memory_utilization=0.90, enable_auto_tool_choice=True, enable_lora=bool(amap), max_loras=max(1, len(amap)),
                 max_lora_rank=max(ranks) if ranks else 128, scheduler_reserve_full_isl=False, tensor_parallel_size=tp,
                 startup_timeout=60 * 20, extra_args=EXTRA.get(declared, ()))
t0 = time.time()
srv = VllmServer(cfg, adapter_manifest=adapters); srv.start()
models = srv.create_model_registry(aliases=[declared], model_prefix='openai/', api_key=os.environ['LOCAL_API_KEY'])
log('vLLM up', declared, f'{time.time()-t0:.0f}s tp', tp)

# ---------- evaluate (host notebook cell 10, with our task subset and concurrency) ----------
import yaml
from google.adk.agents.context_cache_config import ContextCacheConfig
from google.adk.apps._configs import EventsCompactionConfig
from swegemma.config import EvalConfig, build_submission_limits
from swegemma.evaluate import Evaluator
ev = yaml.safe_load((AGENT_DIR / 'eval_config.yaml').read_text())['evaluation']
limits, gen_constraints = build_submission_limits()
RES = WORK / f'results_{COND}'
ec = EvalConfig(tasks_path=TASKS_PATH, snapshots_dir=DATA / 'snapshots', results_dir=RES, submission_dir=AGENT_DIR,
                models=models, sandbox='subprocess', timeout_seconds=int(ev['timeout_seconds']),
                verification_timeout_seconds=300, max_time_minutes=float(ev['max_time_minutes']),
                max_tool_calls=int(ev['max_tool_calls']), max_turns=int(ev['max_turns']), limits=limits,
                generation_constraints=gen_constraints, adapter_manifest=adapters,
                context_cache_config=ContextCacheConfig(min_tokens=2048, ttl_seconds=1800, cache_intervals=10),
                events_compaction_config=EventsCompactionConfig(compaction_interval=15, overlap_size=2, token_threshold=14336, event_retention_size=5),
                graph_dir=GRAPH_DIR, embeddings_dir=EMB_DIR, wheels_dir=DATA / 'wheels', verbose=False,
                task_ids=chosen, concurrency=int('__CONC__'), display_mode='quiet')
t0 = time.time()
result = asyncio.run(Evaluator(ec).run())
log('EVAL DONE', COND, f'{result.resolved}/{result.total}', f'{time.time()-t0:.0f}s')

# ---------- summary of graph calls ----------
rows = [json.loads(l) for l in open(WORK / f'graph_calls_{COND}.jsonl')]
after = [r for r in rows if r['changed_files'] and not str(r['changed_files'][0]).startswith('<error')]
summ = {'cond': COND, 'prompt': PROMPT, 'model': declared, 'tasks': len(chosen), 'resolved': result.resolved,
        'graph_calls': len(rows), 'graph_calls_after_first_edit': len(after),
        'tasks_with_graph_call_after_edit': len({r['task'] for r in after}),
        'neighbors_substituted': sum(1 for r in rows if r.get('substituted')),
        'after_edit_touching_changed_code': sum(1 for r in after if r.get('query_in_changed') or r.get('stale_results') or r.get('nodes_in_changed')),
        'tasks_with_stale_answer_after_edit': len({r['task'] for r in after if r.get('query_in_changed') or r.get('stale_results') or r.get('nodes_in_changed')})}
log('SUMMARY', json.dumps(summ))
json.dump(summ, open(WORK / f'summary_{COND}.json', 'w'), indent=1)
# copy per-task results to the top level and pack traces so the output download includes them
for fn in ('summary.json', 'task_results.jsonl'):
    for p in RES.rglob(fn):
        shutil.copyfile(p, WORK / f'{COND}_{PROMPT}_{fn}'); break
try:
    shutil.make_archive(str(WORK / f'{COND}_{PROMPT}_results'), 'gztar', root_dir=str(RES))
except Exception as e:
    log('archive error', repr(e)[:200])
srv.stop()
