"""Fill the probe template and write a private Kaggle script kernel. Usage: python make_kernel.py OUTDIR MODELS SLUG"""
import base64, json, os, sys, zlib
here = os.path.dirname(os.path.abspath(__file__))
root = os.path.dirname(here)
out, models, slug = sys.argv[1], sys.argv[2], sys.argv[3]
os.makedirs(out, exist_ok=True)
enc = lambda b: base64.b64encode(zlib.compress(b, 9)).decode()
src = open(os.path.join(here, "probe_template.py"), encoding="utf-8").read()
src = src.replace("__ITEMS__", enc(open(os.path.join(root, "stale", "probe_items.json"), "rb").read()))
src = src.replace("__CENSUS__", enc(open(os.path.join(root, "stale", "census.jsonl"), "rb").read()))
src = src.replace("__SYSTEM__", enc(open(os.path.join(root, "data", "sample_submission", "prompts", "system.md"), "rb").read()))
src = src.replace("__MODELS__", models)
src = src.replace("__NOTE_MODE__", sys.argv[4] if len(sys.argv) > 4 else "commit")
open(os.path.join(out, "probe.py"), "w", encoding="utf-8").write(src)
mid = {"e2b": "gemma-4-e2b-it-qat-w4a16-ct", "e4b": "gemma-4-e4b-it-qat-w4a16-ct",
       "12b": "gemma-4-12b-it-qat-w4a16-ct", "31b": "gemma-4-31b-it-qat-w4a16-ct"}
meta = {
    "id": f"markrowset/{slug}", "title": slug.replace("-", " "), "code_file": "probe.py", "language": "python",
    "kernel_type": "script", "is_private": True, "enable_gpu": True, "enable_tpu": False, "enable_internet": False,
    "dataset_sources": ["metric/gemma-4-developer-agent-wheelhouse"], "competition_sources": ["gemma-4-developer-agent"],
    "kernel_sources": [], "model_sources": [f"google/gemma-4/Other/{v}/2" for v in mid.values()],
    "docker_image": "gcr.io/kaggle-private-byod/python@sha256:37c64f7dd9c54116ecd1bcc88817c5469b88387388fade02bfa8bf3fc647d461",
    "machine_shape": "NvidiaL4",
}
json.dump(meta, open(os.path.join(out, "kernel-metadata.json"), "w"), indent=1)
print("written", out, len(src), "chars")

