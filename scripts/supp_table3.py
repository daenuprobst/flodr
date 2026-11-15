import ast
import json
import os
import pickle
import shutil
import subprocess
import sys
import tempfile

import numpy as np

from common import ROOT, memo, out
from openproblems import DIR, embed, fetch

# version 1 of the task, the leaderboard of the OpenProblems website. Its results.json,
# method_info.json and metric_info.json go into V1, and its metric functions are those
# of github.com/openproblems-bio/openproblems at commit 65efdc8, checked out at REPO
V1, REPO = os.path.join(DIR, "v1"), os.path.join(DIR, "openproblems")
# an interpreter pinned as the image of the task was: python 3.8, numpy 1.23,
# scipy 1.9, scikit-learn 1.1, anndata 0.8, scanpy 1.9, umap-learn 0.5.3, numba
PYTHON = os.environ.get(
    "OPENPROBLEMS_PYTHON", os.path.join(ROOT, ".venv-openproblems", "bin", "python")
)
# the files of the data sets on figshare
DATASETS = {
    "olsson_2016_mouse_blood": "36872214",
    "mouse_hspc_nestorowa2016": "36088649",
    "tenx_5k_pbmc": "25555739",
    "zebrafish_labs": "24566651?private_link=e3921450ec1bd0587870",
}
# FloDR at its defaults and with the density term, by its weight
VARIANTS = {"flodr_logCP10k": 0.0, "flodr_density_logCP10k": 2.0}


def shim():
    # a minimal openproblems package: the metric files of the task unchanged and, of the
    # modules they import, the functions they call (the rest imports R and scprep)
    src, dst = os.path.join(REPO, "openproblems"), os.path.join(V1, "shim")
    task = "tasks/dimensionality_reduction"
    if os.path.exists(dst):
        return dst

    def functions(path, names):
        text = open(os.path.join(src, path)).read()
        found = []
        for node in ast.parse(text).body:
            if isinstance(node, ast.FunctionDef) and node.name in names:
                first = node.decorator_list[0] if node.decorator_list else node
                lines = text.splitlines()[first.lineno - 1 : node.end_lineno]
                found.append("\n".join(lines))
        assert len(found) == len(names), (path, names)
        return "\n\n\n".join(found) + "\n"

    def write(path, text=""):
        path = os.path.join(tmp, "openproblems", path)
        os.makedirs(os.path.dirname(path), exist_ok=True)
        open(path, "w").write(text)

    tmp = dst + ".part"
    shutil.rmtree(tmp, ignore_errors=True)
    for d in ("", "data", "tools", "tasks", task, f"{task}/metrics", f"{task}/methods"):
        write(os.path.join(d, "__init__.py"))
    files = [f"metrics/{f}.py" for f in ("density", "distance_correlation")]
    files += ["metrics/nn_ranking.py", "metrics/trustworthiness.py"]
    for f in files + ["methods/diffusion_map.py"]:
        write(f"{task}/{f}", open(os.path.join(src, task, f)).read())
    head = "import anndata\nimport functools\nimport logging\n\n"
    head += 'log = logging.getLogger("openproblems")\n\n\n'
    head += "def _backport_code_version(apply_method, code_version):\n"
    head += '    apply_method.metadata["code_version"] = code_version\n'
    head += "    return apply_method\n\n\n"
    names = ["normalizer", "method", "metric"]
    write("tools/decorators.py", head + functions("tools/decorators.py", names))
    head = "from . import decorators\n\nimport anndata as ad\nimport logging\n\n"
    head += 'log = logging.getLogger("openproblems")\n\n\n'
    names = ["_cp10k", "log_cp10k"]
    write("tools/normalize.py", head + functions("tools/normalize.py", names))
    write("tools/utils.py", 'def check_version(pkg):\n    return "shim"\n')
    head = "import numpy as np\nimport scipy.sparse\n\n\n"
    names = ["_fix_matrix_format", "_fix_adata", "filter_genes_cells"]
    write("data/utils.py", head + functions("data/utils.py", names))
    os.replace(tmp, dst)
    return dst


# runs in this interpreter to read a data set and in that of the task to score it
_TASK = """
import hashlib, json, pickle, sys
import anndata, numpy as np
mode, shim, data, emb, out, metrics = json.loads(sys.argv[1])
sys.path.insert(0, shim)
from openproblems.data.utils import _fix_adata, filter_genes_cells
from openproblems.tools.normalize import log_cp10k
# the data set as the task hands it to methods and metrics
adata = anndata.read_h5ad(data)
filter_genes_cells(adata)
_fix_adata(adata)
adata.uns["n_genes"] = adata.shape[1]
adata = log_cp10k(adata)
adata.strings_to_categoricals()
names = np.asarray(adata.obs_names, dtype=str)
if mode == "read":
    pickle.dump((adata.X, names), open(out, "wb"))
    sys.exit()
from openproblems.tasks.dimensionality_reduction.metrics import density
from openproblems.tasks.dimensionality_reduction.metrics import distance_correlation
from openproblems.tasks.dimensionality_reduction.metrics import nn_ranking
from openproblems.tasks.dimensionality_reduction.metrics import trustworthiness
modules = (density, distance_correlation, trustworthiness, nn_ranking)
fns = {k: getattr(m, k) for m in modules for k in metrics if hasattr(m, k)}
# the six co-ranking metrics each fit the same pair of matrices
fit, last = nn_ranking._fit, {}
def digest(a):
    return hashlib.sha1(np.ascontiguousarray(a).tobytes()).hexdigest()
def once(X, E):
    key = digest(X), digest(np.asarray(E))
    if key not in last:
        last.clear()
        last[key] = fit(X, E)
    return last[key]
nn_ranking._fit = once
z = np.load(emb)
assert (names == z["obs_names"]).all()
res = {}
for m in z.files:
    if m != "obs_names":
        adata.obsm["X_emb"] = z[m]
        res[m] = {k: float(fns[k](adata)) for k in metrics}
json.dump(res, open(out, "w"))
"""


def run(name, metrics):
    # FloDR on one data set: fitted here on the task's normalised counts, then scored by
    # the task's metric functions in its interpreter
    url = "https://ndownloader.figshare.com/files/" + DATASETS[name]
    data = fetch(url, os.path.join(V1, "data", f"{name}.h5ad"))
    emb = os.path.join(V1, "emb", f"{name}.npz")
    with tempfile.TemporaryDirectory() as tmp:
        f = os.path.join(tmp, "out")

        def task(python, mode):
            args = json.dumps([mode, shim(), data, emb, f, metrics])
            subprocess.run([python, "-c", _TASK, args], check=True)

        if not os.path.exists(emb):
            task(sys.executable, "read")
            X, names = pickle.load(open(f, "rb"))
            maps = {m: embed(X, w) for m, w in VARIANTS.items()}
            os.makedirs(os.path.dirname(emb), exist_ok=True)
            np.savez(emb, obs_names=names, **maps)
        task(PYTHON, "score")
        return json.load(open(f))


def published(name):
    return json.load(open(os.path.join(V1, f"{name}.json")))


results, info = published("results"), published("method_info")
METRICS = [m["metric_id"] for m in published("metric_info")]
baselines = {m["method_id"] for m in info if m.get("is_baseline")}
pub = {(r["method_id"], r["dataset_id"]): r["metric_values"] for r in results}
# FloDR on the metrics the published results have for the data set, four on zebrafish
ours = {}
for d in DATASETS:
    have = sorted({k for (_, ds), v in pub.items() if ds == d for k in v})
    ours[d] = memo(os.path.join(V1, "scores.json"), d, lambda: run(d, have))

# each metric is scaled per data set between the lowest and the highest baseline
bounds = {}
for d in DATASETS:
    for k in METRICS:
        bv = [pub[b, d][k] for b in baselines if k in pub[b, d]]
        bounds[d, k] = (min(bv), max(bv)) if len(bv) >= 2 else None
# which reproduces the published scaled scores
for r in results:
    for k, v in r["scaled_scores"].items():
        lo, hi = bounds[r["dataset_id"], k]
        assert abs((r["metric_values"][k] - lo) / (hi - lo) - v) < 1e-12


def cells(values):
    # as on the website: clipped to [0, 1], a missing value counts as zero
    S = np.zeros((len(DATASETS), len(METRICS)))
    for i, d in enumerate(DATASETS):
        for j, k in enumerate(METRICS):
            if values[d].get(k) is not None and bounds[d, k]:
                lo, hi = bounds[d, k]
                S[i, j] = np.clip((values[d][k] - lo) / (hi - lo), 0, 1)
    return S


# FloDR, then the published methods under their published names, by their mean
names = {"flodr_logCP10k": "FloDR", "flodr_density_logCP10k": "FloDR, density term"}
names |= {m["method_id"]: m["method_name"] for m in info}
S = {m: cells({d: ours[d][m] for d in DATASETS}) for m in VARIANTS}
rest = sorted({m for m, _ in pub} - baselines)
rest = {m: cells({d: pub[m, d] for d in DATASETS}) for m in rest}
S.update(sorted(rest.items(), key=lambda t: -t[1].mean()))

lines = [r"Method & Myeloid & HSPC & PBMC & Zebrafish & Mean \\", r"\midrule"]
for m, s in S.items():
    means = [f"{x:.3f}" for x in (*s.mean(1), s.mean())]
    name = names[m].replace("gamma=0", r"$\gamma=0$")
    lines.append(f"{name} & " + " & ".join(means) + r" \\")
    if m == "flodr_density_logCP10k":
        lines.append(r"\midrule")

open(out("supp_table3.tex"), "w").write("\n".join(lines) + "\n")
print("\n".join(lines))
