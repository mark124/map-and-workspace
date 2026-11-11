"""Graph-tool lookups for the measurement scripts, done with the OFFICIAL swegemma functions.

This public version calls swegemma (install it from the competition's wheelhouse dataset,
metric/gemma-4-developer-agent-wheelhouse) instead of re-implementing its lookup code:
  resolve_node_name, get_neighbor, get_similar_nodes  from swegemma.graph
The competition graph JSON is read here only for the node list, stored source text and raw edges.
"""
import json, os
from swegemma import graph as sg

_REPOS = {"fastapi": "fastapi/fastapi", "rich": "Textualize/rich", "requests": "psf/requests", "httpx": "encode/httpx"}


def _bare(x):
    x = str(x)
    return x.split(";", 1)[1] if ";" in x else x


class Repo:
    def __init__(self, graph_path, emb_path):
        key = os.path.splitext(os.path.basename(graph_path))[0]          # e.g. rich_<commit>
        short, commit = key.split("_", 1)
        self.repo = _REPOS[short]
        self.g = sg.get_graph(repo_name=self.repo, graph_dir=os.path.dirname(graph_path),
                              embeddings_dir=os.path.dirname(emb_path), base_commit=commit)
        self._dirs = (os.path.dirname(graph_path), os.path.dirname(emb_path), commit)
        data = json.load(open(graph_path, encoding="utf-8"))
        self.ids = [n["id"] for n in data["nodes"]]
        self.text = {n["id"]: n.get("text") or "" for n in data["nodes"]}
        self.out, self.inn = {}, {}
        for e in data["edges"]:
            self.out.setdefault(e["source"], []).append((e["target"], e.get("type")))
            self.inn.setdefault(e["target"], []).append((e["source"], e.get("type")))

    def neighbors(self, node, max_neighbors=50):
        """(resolved node or None, neighbour list), as get_code_neighbors answers."""
        r = sg.resolve_node_name(f"{self.repo};{_bare(node)}", self.g)
        if r is None:
            return None, []
        nb = sg.get_neighbor(node=r, graph=self.g, max_neighbors=max_neighbors)
        return _bare(r), sorted(_bare(n) for n in nb)

    def similar(self, query, k=10):
        """(resolved or None, mode, [(node, stored source)]), as search_similar_code answers before its size cap."""
        r = sg.resolve_node_name(f"{self.repo};{_bare(query)}", self.g)
        gd, ed, commit = self._dirs
        res = sg.get_similar_nodes(node=query, repo_name=self.repo, k=k, graph=self.g,
                                   graph_dir=gd, embeddings_dir=ed, base_commit=commit)
        mode = "embedding" if r is not None else "lexical"
        return (_bare(r) if r is not None else None), mode, [(_bare(x["node_name"]), x.get("code") or "") for x in res]
