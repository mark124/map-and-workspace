# The Map and the Workspace: code

Code for the paper "The Map and the Workspace: What a Coding Agent's Code Map Shows After Its Own Edits" (Gemma 4 Developer Agent Paper Track, Kaggle, 2026).

Licensed under the Apache License 2.0 (see `LICENSE`). Competition data is not included; scripts read it from the competition input on Kaggle or from a local download made with `list_files.py` and `fetch.py` (needs your own Kaggle API credentials).

## What is here

| Folder | What it does | Paper section |
|---|---|---|
| `stale/` | Applies each public reference fix, compares functions before and after, and reproduces the harness lookup code (`toolsim.py`) to see what the graph tools return about the changed code. `census.py` then `replay2.py`, then `headline.py` for the table. `build_probe.py` builds the 48 reading-test cases. | 2, 3, 4 |
| `graphrefresh/` | `graphrefresh.py`: a small index that re-reads only changed Python files and repairs the call links that depend on them. `check_equivalence.py`: refresh versus full rebuild, run locally. | 5 |
| `kaggle_refresh/` | Builds the CPU Kaggle notebook that runs the refresh-versus-rebuild check on the competition snapshots. | 5 |
| `kaggle_probe/` | Builds the GPU Kaggle notebook for the reading test (official `swegemma` check, then four Gemma 4 sizes with and without the note). `tally.py` and `paired.py` summarise the answers. | 2, 4 |
| `kaggle_agent/` | Builds the GPU Kaggle notebook for full agent sessions, following the host's getting-started notebook. It wraps the three graph tools to log every call and, in the "note" condition, adds the version fields. `compare_runs.py` summarises runs. | 4.1 |
| `paper/figures/` | `make_figures.py` redraws Figures 1 to 3 from the data files. | Figures |

## Notes

- `stale/toolsim.py` calls the official `swegemma` 0.2.11 functions (`resolve_node_name`, `get_neighbor`, `get_similar_nodes`); install `swegemma` from the competition's wheelhouse dataset. For the paper we used our own reproduction of these functions, which is not included here; `kaggle_release_check/` runs `stale/replay2.py` with this official-code version on Kaggle and compares it with our results.
- `graphrefresh` uses its own simple extractor (one node per function or class; a call to `foo()` links to every function named `foo`). It is not the competition's graph builder.
- Kaggle notebooks are written for 4 x L4 GPUs and the official `gemma-4-*-it-qat-w4a16-ct` checkpoints.
