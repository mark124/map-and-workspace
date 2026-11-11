# Figures for paper 2, drawn from dataset/tool_calls.jsonl (written by analyse.py).
import json, site, sys
from collections import Counter, defaultdict
from pathlib import Path
try:
    import matplotlib
except ImportError:
    sys.path.append(site.getusersitepackages()); import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

HERE = Path(__file__).resolve().parent
FIG = HERE / 'figures'; FIG.mkdir(exist_ok=True)
RUNS = ['stock_12b', 'check_12b', 'checknote_12b', 'check_31b']
LABEL = {'stock_12b': '12B, sample prompt', 'check_12b': '12B, + check line', 'checknote_12b': '12B, + check line + note', 'check_31b': '31B, + check line'}
PAL = ['#2a78d6', '#eb6834', '#1baf7a', '#eda100', '#e87ba4', '#008300', '#6250d6', '#e34948']
SURF, INK, INK2 = '#fcfcfb', '#0b0b0b', '#52514e'

def group(r):
    t = r['tool']
    if t == 'read_file': return 'read_file'
    if t == 'run_command': return 'run_command: search/view' if r.get('cmd_kind') == 'search/view' else 'run_command: scripts, tests, other'
    if t in ('edit_file', 'write_file'): return 'edit_file / write_file'
    if t in ('search_similar_code', 'get_code_neighbors', 'get_code_subgraph'): return 'graph tools'
    return 'submit, status, sub-agent'

GROUPS = ['read_file', 'run_command: search/view', 'run_command: scripts, tests, other', 'edit_file / write_file', 'graph tools', 'submit, status, sub-agent']
EDIT = ['ok', 'old_string not found', 'missing argument', 'other']
mix = defaultdict(Counter); edits = defaultdict(Counter)
for line in open(HERE / 'dataset' / 'tool_calls.jsonl', encoding='utf-8'):
    r = json.loads(line)
    mix[r['run']][group(r)] += 1
    if r['tool'] == 'edit_file':
        e = r.get('edit_result'); edits[r['run']][e if e in EDIT else 'other'] += 1

def stacked(ax, data, cats, title, xlabel, pct=True):
    ys = list(range(len(RUNS)))[::-1]; seen = set()
    for y, run in zip(ys, RUNS):
        tot = sum(data[run].values()); left = 0
        for i, c in enumerate(cats):
            v = data[run][c]; w = 100 * v / tot if pct else v
            if w <= 0: continue
            ax.barh(y, w, left=left, height=0.62, color=PAL[i], edgecolor=SURF, linewidth=2, label=None if c in seen else c); seen.add(c)
            if w >= 7:
                ax.text(left + w / 2, y, f'{w:.0f}%' if pct else str(v), ha='center', va='center', fontsize=8.5, color='white' if i != 3 else INK)
            left += w
        ax.text(left + 1, y, f'n={tot}', va='center', fontsize=8.5, color=INK2)
    ax.set_yticks(ys); ax.set_yticklabels([LABEL[r] for r in RUNS], fontsize=9, color=INK)
    ax.set_xlim(0, 112); ax.set_xlabel(xlabel, fontsize=9, color=INK2)
    ax.set_title(title, fontsize=10.5, color=INK, loc='left')
    for s in ('top', 'right', 'left'): ax.spines[s].set_visible(False)
    ax.spines['bottom'].set_color('#c3c2b7'); ax.tick_params(axis='x', colors=INK2, labelsize=8.5); ax.tick_params(axis='y', length=0)
    ax.set_facecolor(SURF)
    h, l = ax.get_legend_handles_labels(); o = [l.index(c) for c in cats if c in l]
    ax.legend([h[i] for i in o], [l[i] for i in o], loc='upper center', bbox_to_anchor=(0.45, -0.22), ncol=3, frameon=False, fontsize=8.5)

fig, ax = plt.subplots(figsize=(8.2, 3.6), facecolor=SURF)
stacked(ax, mix, GROUPS, 'Figure 2. Share of all tool calls by tool (main agent + sub-agent)', 'percent of tool calls in the run')
fig.tight_layout(); fig.savefig(FIG / 'fig1_tool_mix.png', dpi=200); fig.savefig(FIG / 'fig1_tool_mix.svg')

fig, ax = plt.subplots(figsize=(8.2, 3.3), facecolor=SURF)
stacked(ax, edits, EDIT, 'Figure 3. What happened to each edit_file call', 'percent of edit_file calls in the run')
fig.tight_layout(); fig.savefig(FIG / 'fig2_edit_outcomes.png', dpi=200); fig.savefig(FIG / 'fig2_edit_outcomes.svg')
print('figures written', {r: dict(edits[r]) for r in RUNS})
