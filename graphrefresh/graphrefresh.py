"""graphrefresh: keep a Python code graph in step with an editing agent.

The graph has one node per module-level or nested def/class (id = dotted path, text = source without
decorators, like the competition graphs) and 'calls' edges from a definition to every definition whose
short name it calls (a deliberately simple, documented name-based rule; it is OUR extractor, not a claim
about the competition's).

Index.build(root)           full build from a source tree
Index.refresh(root)         re-hash every .py file; re-parse only changed / added / deleted files;
                            recompute edges for (a) definitions in changed files and (b) definitions
                            elsewhere that call a short name whose set of targets changed.
Index.version()             content hash of the indexed tree, to put in every tool reply
Index.changed_since_build   files modified since the last full build or refresh

The invariant tested in tests: build(post_tree) == refresh(build(base_tree), post_tree).
"""
import ast, hashlib, os
from collections import defaultdict

SKIP_DIRS = {".git", "__pycache__", ".venv", "venv", "node_modules", ".tox", "build", "dist"}


def module_name(rel):
    p = rel[:-3]
    if p.startswith("src/"):
        p = p[4:]
    parts = p.split("/")
    if parts[-1] == "__init__":
        parts = parts[:-1]
    return ".".join(parts)


def parse_file(src, mod):
    """-> {id: (text, frozenset(called short names))} for every def/class in the file."""
    try:
        tree = ast.parse(src)
    except (SyntaxError, ValueError):
        return {}
    lines = src.splitlines(keepends=True)
    out = {}

    def calls(n):
        c = set()
        for x in ast.walk(n):
            if isinstance(x, ast.Call):
                f = x.func
                if isinstance(f, ast.Name):
                    c.add(f.id)
                elif isinstance(f, ast.Attribute):
                    c.add(f.attr)
        return frozenset(c)

    def visit(node, prefix):
        for ch in ast.iter_child_nodes(node):
            if isinstance(ch, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
                q = prefix + "." + ch.name if prefix else ch.name
                text = "".join(lines[ch.lineno - 1:ch.end_lineno])
                out[q] = (text, frozenset() if isinstance(ch, ast.ClassDef) else calls(ch))
                visit(ch, q)
    visit(tree, mod)
    return out


class Index:
    def __init__(self):
        self.files = {}      # rel path -> sha256
        self.defs = {}       # rel path -> {id: (text, calls)}
        self.by_short = defaultdict(set)
        self.edges = {}      # id -> frozenset(target ids)
        self.changed_since_build = []

    # ---- helpers
    @staticmethod
    def _scan(root):
        found = {}
        for d, dirs, fs in os.walk(root):
            dirs[:] = [x for x in dirs if x not in SKIP_DIRS]
            for f in fs:
                if f.endswith(".py"):
                    p = os.path.join(d, f)
                    rel = os.path.relpath(p, root).replace(os.sep, "/")
                    found[rel] = p
        return found

    def _all_defs(self):
        for rel, ds in self.defs.items():
            for q, v in ds.items():
                yield q, v

    def _edge_for(self, calls):
        t = set()
        for c in calls:
            t |= self.by_short.get(c, set())
        return frozenset(t)

    # ---- API
    def build(self, root):
        self.__init__()
        for rel, p in self._scan(root).items():
            b = open(p, "rb").read()
            self.files[rel] = hashlib.sha256(b).hexdigest()
            self.defs[rel] = parse_file(b.decode("utf-8", "replace"), module_name(rel))
        for q, _ in self._all_defs():
            self.by_short[q.rsplit(".", 1)[-1]].add(q)
        for q, (_, calls) in self._all_defs():
            self.edges[q] = self._edge_for(calls)
        return self

    def refresh(self, root):
        found = self._scan(root)
        changed = []
        for rel, p in found.items():
            b = open(p, "rb").read()
            h = hashlib.sha256(b).hexdigest()
            if self.files.get(rel) != h:
                changed.append((rel, b))
                self.files[rel] = h
        deleted = [rel for rel in list(self.files) if rel not in found]
        touched_short = set()
        touched_ids = set()
        for rel in deleted:
            for q in self.defs.get(rel, {}):
                touched_short.add(q.rsplit(".", 1)[-1]); self.by_short[q.rsplit(".", 1)[-1]].discard(q)
                self.edges.pop(q, None)
            self.defs.pop(rel, None); self.files.pop(rel, None)
        for rel, b in changed:
            old = self.defs.get(rel, {})
            new = parse_file(b.decode("utf-8", "replace"), module_name(rel))
            for q in old:
                if q not in new:
                    touched_short.add(q.rsplit(".", 1)[-1]); self.by_short[q.rsplit(".", 1)[-1]].discard(q)
                    self.edges.pop(q, None)
            for q in new:
                if q not in old:
                    touched_short.add(q.rsplit(".", 1)[-1]); self.by_short[q.rsplit(".", 1)[-1]].add(q)
            self.defs[rel] = new
            touched_ids |= set(new)
        for s in [s for s, v in self.by_short.items() if not v]:
            del self.by_short[s]
        # (a) definitions in changed files, (b) callers of any short name whose targets changed
        for q, (_, calls) in self._all_defs():
            if q in touched_ids or (calls & touched_short):
                self.edges[q] = self._edge_for(calls)
        self.changed_since_build = sorted([r for r, _ in changed] + deleted)
        return self.changed_since_build

    def version(self):
        h = hashlib.sha256()
        for rel in sorted(self.files):
            h.update(rel.encode()); h.update(self.files[rel].encode())
        return h.hexdigest()[:12]

    def snapshot(self):
        nodes = {q: v[0] for q, v in self._all_defs()}
        return nodes, {q: e for q, e in self.edges.items() if q in nodes}
