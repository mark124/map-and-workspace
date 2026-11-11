"""Fill agent_template.py and write a private GPU Kaggle kernel.
Usage: python make_kernel.py OUTDIR COND MODEL NTASKS CONC SLUG"""
import json, os, sys
here = os.path.dirname(os.path.abspath(__file__))
out, cond, model, n, conc, slug, prompt = sys.argv[1:8]
assert cond in ("plain", "note") and prompt in ("stock", "check")
os.makedirs(out, exist_ok=True)
src = open(os.path.join(here, "agent_template.py"), encoding="utf-8").read()
for k, v in {"__COND__": cond, "__MODEL__": model, "__NTASKS__": n, "__CONC__": conc, "__PROMPT__": prompt}.items():
    src = src.replace(k, v)
open(os.path.join(out, "run.py"), "w", encoding="utf-8").write(src)
meta = {
    "id": f"markrowset/{slug}", "title": slug.replace("-", " "), "code_file": "run.py", "language": "python",
    "kernel_type": "script", "is_private": True, "enable_gpu": True, "enable_tpu": False, "enable_internet": False,
    "dataset_sources": ["metric/gemma-4-developer-agent-wheelhouse"], "competition_sources": ["gemma-4-developer-agent"],
    "kernel_sources": [], "model_sources": [f"google/gemma-4/Other/{model}/2"],
    "docker_image": "gcr.io/kaggle-private-byod/python@sha256:37c64f7dd9c54116ecd1bcc88817c5469b88387388fade02bfa8bf3fc647d461",
    "machine_shape": "NvidiaL4",
}
json.dump(meta, open(os.path.join(out, "kernel-metadata.json"), "w"), indent=1)
print("written", out)
