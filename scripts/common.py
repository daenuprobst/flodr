import io
import json
import os
import subprocess
import tempfile
import time
import urllib.request

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA = os.environ.get("DATA", os.path.join(ROOT, "data"))
CACHE = os.environ.get("CACHE", os.path.join(ROOT, "scripts", "cache"))
OUT = os.environ.get("OUT", os.path.join(ROOT, "scripts", "out"))
# Parametric UMAP needs TensorFlow 2.15 and umap-learn 0.5.5 in an interpreter of its own
DREAMS_PYTHON = os.environ.get(
    "DREAMS_PYTHON", os.path.join(ROOT, ".venv-dreams", "bin", "python")
)
PCUMAP_PYTHON = os.environ.get(
    "PCUMAP_PYTHON", os.path.join(ROOT, ".venv-pumap", "bin", "python")
)

K = 15
N_CAP = 5000
SEEDS = (0, 1, 2)
BENCHMARKS = ("mnist", "fmnist", "paul15", "drfp")
ATLASES = ("bmarrow", "cerebellum", "plant")
NAMES_2L = {
    "bmarrow": "Human fetal\nbone marrow",
    "cerebellum": "Mouse developing\ncerebellum",
    "plant": "Arabidopsis\ngerminating seed",
}
# deposited labels that are code identifiers, shown as words
PRETTY = {
    "B_lineage": "B lineage",
    "T_NK": "T/NK",
    "HSC/MPP and pro": "HSC/MPP and prog.",
}
NAMES = {
    "mnist": "MNIST",
    "fmnist": "Fashion-MNIST",
    "paul15": "paul15",
    "drfp": "Schneider 50k",
    "bmarrow": "Human fetal bone marrow",
    "cerebellum": "Mouse developing cerebellum",
    "plant": "Arabidopsis germinating seed",
}
ATLAS_FILES = {
    "bmarrow": ("fetal_bone_marrow.h5ad", ("broad_fig1_cell.labels", "cell_type")),
    "cerebellum": ("cerebellum.h5ad", ("cell_type", "celltype", "annotation")),
    "plant": ("arabidopsis_seed.h5ad", ("CellType", "reduced_CellType", "cell_type")),
}


def cached(path, make):
    # a dict of arrays computed once
    if not os.path.exists(path):
        os.makedirs(os.path.dirname(path), exist_ok=True)
        np.savez_compressed(path, **make())
    return dict(np.load(path, allow_pickle=True))


def memo(path, key, make):
    # per-key results in one json, computed once
    d = json.load(open(path)) if os.path.exists(path) else {}
    if key not in d:
        d[key] = make()
        os.makedirs(os.path.dirname(path), exist_ok=True)
        json.dump(d, open(path, "w"), indent=1)
    return d[key]


def out(name):
    os.makedirs(OUT, exist_ok=True)
    return os.path.join(OUT, name)


def standardise(X):
    X = np.asarray(X, np.float64)
    return ((X - X.mean(0)) / (X.std(0) + 1e-9)).astype(np.float32)


def pca50(X):
    from sklearn.decomposition import PCA

    X = np.asarray(X.todense() if hasattr(X, "todense") else X, np.float64)
    return standardise(
        PCA(50, svd_solver="randomized", random_state=0).fit_transform(X)
    )


def _openml(name):
    from sklearn.datasets import fetch_openml

    X, y = fetch_openml(
        name, version=1, return_X_y=True, as_frame=False, parser="liac-arff"
    )
    sub = np.random.default_rng(0).permutation(len(X))[:N_CAP]
    X = X[sub].astype(np.float32)
    return (X - X.mean()) / X.std(), np.unique(y, return_inverse=True)[1][sub]


def _paul15():
    import scanpy as sc

    sc.settings.datasetdir = DATA
    a = sc.datasets.paul15()
    y = np.unique(a.obs["paul15_clusters"], return_inverse=True)[1]
    sc.pp.filter_genes(a, min_counts=1)
    sc.pp.normalize_total(a, target_sum=1e4)
    sc.pp.log1p(a)
    sc.pp.highly_variable_genes(a, n_top_genes=1000)
    a = a[:, a.var.highly_variable].copy()
    sc.pp.scale(a, max_value=10)
    return pca50(a.X), y


def _drfp():
    import pandas as pd
    from drfp import DrfpEncoder

    url = "https://raw.githubusercontent.com/reymond-group/drfp/main/data/schneider50k.tsv"
    df = pd.read_csv(
        io.BytesIO(urllib.request.urlopen(url, timeout=600).read()),
        sep="\t",
        index_col=0,
    )
    sup = df.rxn_class.astype(str).str.split(".").str[0]
    df = df.groupby(sup, group_keys=False).apply(
        lambda g: g.sample(max(1, round(N_CAP * len(g) / len(sup))), random_state=0)
    )
    sup = df.rxn_class.astype(str).str.split(".").str[0]
    y = sup.map({c: i for i, c in enumerate(sorted(sup.unique(), key=int))}).to_numpy(
        int
    )
    X = np.array(DrfpEncoder.encode(df.rxn.tolist(), n_folded_length=2048)) > 0
    return X.astype(np.float32), y


def _atlas(name):
    import scanpy as sc

    file, cols = ATLAS_FILES[name]
    a = sc.read_h5ad(os.path.join(DATA, file))
    if float(a.X.max()) < 50 and a.raw is not None:
        a = a.raw.to_adata()
    y = np.asarray(a.obs[next(c for c in cols if c in a.obs)].astype(str))
    # the plant file holds scaled values only
    if float(a.X.min()) < -0.01:
        return pca50(a.X), y
    sc.pp.filter_genes(a, min_counts=1)
    sc.pp.normalize_total(a, target_sum=1e4)
    sc.pp.log1p(a)
    sc.pp.highly_variable_genes(a, n_top_genes=2000, subset=True)
    sc.pp.scale(a, max_value=10)
    return pca50(a.X), y


def _subset(X, y, n):
    rows = np.sort(np.random.default_rng(1).choice(len(X), n, replace=False))
    return X[rows], y[rows]


def _mnist10k():
    from sklearn.datasets import fetch_openml

    X, y = fetch_openml(
        "mnist_784", version=1, as_frame=False, return_X_y=True, parser="auto"
    )
    X, y = _subset(X, np.unique(y, return_inverse=True)[1], 10_000)
    X = X.astype(np.float32) / 255.0
    return (X - X.mean()) / X.std(), y


def _hnoca():
    # the scPoli latent and its labels, range-read out of the 18.7 GB h5ad (about 80 MB)
    import fsspec
    import h5py
    from anndata.io import read_elem

    url = "https://zenodo.org/records/14161275/files/hnoca_cleanedmeta.h5ad?download=1"
    fs = fsspec.filesystem("http", block_size=16 * 1024 * 1024)
    with fs.open(url, "rb") as fh, h5py.File(fh, "r") as f:
        Z = np.asarray(read_elem(f["obsm"]["X_scpoli"]), np.float32)
        return Z, np.asarray(read_elem(f["obs"]["annot_level_2"])).astype(str)


def _zebrafish():
    # CELLxGENE Discover dataset bcd4103c-afdc-4233-af6e-554589d70cd3
    import scanpy as sc

    a = sc.read_h5ad(os.path.join(DATA, "zebrafish", "dev_zebrafish.h5ad"))
    y = np.asarray(a.obs["cell_type"]).astype(str)
    sc.pp.normalize_total(a, target_sum=1e4)
    sc.pp.log1p(a)
    sc.pp.highly_variable_genes(a, n_top_genes=2000, subset=True)
    sc.pp.scale(a, max_value=10)
    return pca50(a.X), y


def paul15_pcs():
    # the 50 principal components without per-component scaling, the labels, and the
    # linear readout of the 1000 highly variable genes (log expression) from them
    import scanpy as sc
    from sklearn.decomposition import PCA

    sc.settings.datasetdir = DATA
    a = sc.datasets.paul15()
    labels = np.asarray(a.obs["paul15_clusters"]).astype(str)
    sc.pp.filter_genes(a, min_counts=1)
    sc.pp.normalize_total(a, target_sum=1e4)
    sc.pp.log1p(a)
    sc.pp.highly_variable_genes(a, n_top_genes=1000, subset=True)
    expr = np.asarray(a.X, np.float64)
    sc.pp.scale(a, max_value=10)
    pca = PCA(50, svd_solver="randomized", random_state=0)
    P = pca.fit_transform(np.asarray(a.X, np.float64)).astype(np.float32)
    sd, mean = a.var["std"].to_numpy(), a.var["mean"].to_numpy()
    W, b = pca.components_ * sd, pca.mean_ * sd + mean
    return P, labels, dict(genes=list(a.var_names), expr=expr, W=W, b=b)


def split(n, test=0.2):
    # the 80/20 split of the held-out experiments: (train, test)
    perm = np.random.default_rng(0).permutation(n)
    k = round(test * n)
    return np.sort(perm[k:]), np.sort(perm[:k])


def load(name):
    path = (
        os.path.join(CACHE, "scrna", f"{name}.npz")
        if name in ATLASES
        else os.path.join(CACHE, "data", f"{name}.npz")
    )
    if not os.path.exists(path):
        X, y = {
            "mnist": lambda: _openml("mnist_784"),
            "fmnist": lambda: _openml("Fashion-MNIST"),
            "paul15": _paul15,
            "drfp": _drfp,
            "bmarrow20k": lambda: _subset(*load("bmarrow"), 20_000),
            "mnist10k": _mnist10k,
            "hnoca": _hnoca,
            "zebrafish": _zebrafish,
        }.get(name, lambda: _atlas(name))()
        os.makedirs(os.path.dirname(path), exist_ok=True)
        np.savez_compressed(path, P=X, labels=y)
    z = np.load(path, allow_pickle=True)
    return np.ascontiguousarray(z["P"], np.float32), np.asarray(z["labels"])


def metric(ds, method=""):
    return (
        "jaccard"
        if ds == "drfp"
        and method.split(" ")[0] in ("FloDR", "UMAP", "openTSNE", "Free")
        else "euclid"
    )


def _flodr(w):
    def go(X, seed, metric):
        from flodr import FloDR

        return FloDR(w=w, random_state=seed, verbose=False).fit(X).embedding_

    return go


def _umap(X, seed, metric):
    import umap

    return umap.UMAP(
        n_neighbors=K,
        min_dist=0.1,
        random_state=seed,
        init="pca",
        metric="jaccard" if metric == "jaccard" else "euclidean",
    ).fit_transform(X)


def _opentsne(X, seed, metric):
    from openTSNE import TSNE

    return np.asarray(
        TSNE(
            perplexity=30,
            random_state=seed,
            n_jobs=8,
            metric="jaccard" if metric == "jaccard" else "euclidean",
        ).fit(X)
    )


def _trimap(X, seed, metric):
    import trimap

    np.random.seed(seed)
    return trimap.TRIMAP(verbose=False).fit_transform(X)


def _pacmap(X, seed, metric, cls="PaCMAP"):
    import pacmap

    return getattr(pacmap, cls)(
        n_neighbors=K, random_state=seed, verbose=False
    ).fit_transform(X)


def _umato(X, seed, metric):
    import umato

    return umato.UMATO(
        n_neighbors=K, n_components=2, random_state=seed, verbose=False
    ).fit_transform(X)


def _paramrepulsor(X, seed, metric):
    from parampacmap import ParamPaCMAP

    return np.asarray(
        ParamPaCMAP(
            n_components=2, n_neighbors=K, seed=seed, verbose=False
        ).fit_transform(X)
    )


def _phate(X, seed, metric):
    import phate

    return phate.PHATE(random_state=seed, verbose=0, n_jobs=8).fit_transform(X)


def _pymde(X, seed, metric):
    import pymde
    import torch

    pymde.seed(seed)
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    mde = pymde.preserve_neighbors(
        torch.as_tensor(X), embedding_dim=2, n_neighbors=K, verbose=False, device=dev
    )
    return mde.embed(verbose=False).cpu().numpy()


# Parametric UMAP runs in an interpreter of its own that has none of this module's imports
_PCUMAP = """
import json, sys, numpy as np, tensorflow as tf
from umap.parametric_umap import ParametricUMAP
fin, fout, seed = json.loads(sys.argv[1])
tf.random.set_seed(seed)
np.random.seed(seed)
z = np.load(fin)
est = ParametricUMAP(n_neighbors=15, min_dist=0.1, random_state=seed, verbose=False)
Y = est.fit_transform(z["X"])
np.savez(fout, Y=Y, Yte=est.transform(z["Xte"]) if "Xte" in z else Y[:0])
"""


def pcumap(X, seed, Xte=None):
    with tempfile.TemporaryDirectory() as tmp:
        fin, fout = os.path.join(tmp, "in.npz"), os.path.join(tmp, "out.npz")
        np.savez(fin, X=X, **({} if Xte is None else {"Xte": Xte}))
        cmd = [PCUMAP_PYTHON, "-c", _PCUMAP, json.dumps([fin, fout, seed])]
        env = dict(os.environ, TF_CPP_MIN_LOG_LEVEL="3")
        subprocess.run(cmd, env=env, check=True, stdout=subprocess.DEVNULL)
        z = np.load(fout)
        return np.asarray(z["Y"], np.float32), np.asarray(z["Yte"], np.float32)


# DREAMS runs in an interpreter of its own, with the authors' openTSNE fork (tag tmlr)
_DREAMS = """
import json, sys, numpy as np
import openTSNE.tsne as ot
from openTSNE import TSNE
from sklearn.decomposition import PCA
f, seed = json.loads(sys.argv[1])
# the tag passes X to an optimiser that no longer takes it
call = ot.gradient_descent.__call__
ot.gradient_descent.__call__ = lambda self, *a, X=None, **kw: call(self, *a, **kw)
X = np.load(f)
# start and target: the first two PCs, the first at sd 1e-4
R = PCA(2, random_state=0).fit_transform(X.astype(np.float64))
R = np.ascontiguousarray(R / R[:, 0].std() * 1e-4)
kw = dict(regularization=True, reg_lambda=0.15, reg_embedding=R, n_jobs=8)
Y = TSNE(initialization=R, random_state=seed, **kw).fit(X)
np.save(f, np.asarray(Y, np.float32))
"""


def _dreams(X, seed, metric):
    with tempfile.TemporaryDirectory() as tmp:
        f = os.path.join(tmp, "X.npy")
        np.save(f, X)
        cmd = [DREAMS_PYTHON, "-c", _DREAMS, json.dumps([f, seed])]
        subprocess.run(cmd, check=True, stdout=subprocess.DEVNULL)
        return np.load(f)


def _pca2(X, seed, metric):
    from sklearn.decomposition import PCA

    return PCA(2, svd_solver="randomized", random_state=seed).fit_transform(X)


def _squadmds(X, seed, metric, n_iter=4000):
    # Lambert et al. 2021, from PCA at unit scale
    import torch
    from sklearn.decomposition import PCA

    dev = "cuda" if torch.cuda.is_available() else "cpu"
    g = torch.Generator().manual_seed(seed)
    Xt = torch.as_tensor(X, device=dev)
    Y = torch.as_tensor(
        PCA(2, random_state=seed).fit_transform(X), device=dev, dtype=torch.float32
    )
    Y = (Y / Y.std()).requires_grad_(True)
    opt = torch.optim.Adam([Y], lr=2e-2)
    m = (len(X) // 4) * 4
    a, b = zip(*[(0, 1), (0, 2), (0, 3), (1, 2), (1, 3), (2, 3)])

    def profile(Z, q):
        d = torch.stack(
            [(Z[q[:, i]] - Z[q[:, j]]).pow(2).sum(1).sqrt() for i, j in zip(a, b)], 1
        )
        return d / d.sum(1, keepdim=True).clamp_min(1e-12)

    for it in range(n_iter):
        q = torch.randperm(len(X), generator=g)[:m].reshape(-1, 4).to(dev)
        loss = (profile(Xt, q) - profile(Y, q)).pow(2).sum(1).mean()
        opt.zero_grad()
        loss.backward()
        opt.step()
        for gp in opt.param_groups:
            gp["lr"] = 2e-2 * (1 - it / n_iter)
    return Y.detach().cpu().numpy()


def fit_free(X, seed, metric="euclid", steps=6400, lr=0.05):
    # the FloDR objective at w = 2 on free coordinates from the whitened PCA start, Adam
    # with a linear decay; the same graph, kernel and terms as the library
    import torch
    from flodr import find_ab, fuzzy_knn_graph, raw_coords
    from flodr.data import knn_query
    from flodr.estimator import prune_fraction

    dev = "cuda" if torch.cuda.is_available() else "cpu"
    gen = torch.Generator(device=dev).manual_seed(seed)
    n = len(X)
    pre = knn_query(X, None, K, "jaccard") if metric == "jaccard" else None
    _, ei, ej, wa, _ = fuzzy_knn_graph(X, K, precomputed=pre, return_dist=True)
    if prune_fraction(n) < 1.0:
        sel = np.sort(np.argsort(-wa)[: round(prune_fraction(n) * len(wa))])
        ei, ej, wa = ei[sel], ej[sel], wa[sel]
    ei, ej = torch.as_tensor(ei, device=dev), torch.as_tensor(ej, device=dev)
    wa = torch.as_tensor(np.asarray(wa, np.float32), device=dev)
    a, b = find_ab(0.01)
    Y0 = raw_coords(X, 50, seed)[0][:, :2]
    Y = torch.nn.Parameter(torch.as_tensor((Y0 - Y0.mean(0)) / Y0.std(), device=dev))
    opt = torch.optim.Adam([Y], lr=lr)
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda i: 1.0 - i / steps)
    xs = torch.as_tensor(np.asarray(X, np.float32), device=dev)
    rs = xs.sum(1)

    def log_dx(r, s):
        if metric == "jaccard":
            inter = (xs[r] * xs[s]).sum(1)
            jac = 1.0 - inter / (rs[r] + rs[s] - inter).clamp_min(1.0)
            return torch.log(jac.clamp_min(1e-6))
        return 0.5 * torch.log(((xs[r] - xs[s]) ** 2).sum(1).clamp_min(1e-12))

    def draw(size, high=n):
        return torch.randint(0, high, (size,), device=dev, generator=gen)

    def d2(p, q):
        return ((Y[p] - Y[q]) ** 2).sum(1)

    def q(x):
        return 1.0 / (1.0 + a * (x + 1e-6) ** b)

    for it in range(steps):
        e = draw(262_144, len(ei)) if len(ei) > 262_144 else slice(None)
        l_attr = -(wa[e] * torch.log(q(d2(ei[e], ej[e])) + 1e-6)).sum() / wa[e].sum()
        m = len(wa[e])
        l_rep = -torch.log(1.0 - q(d2(draw(48 * m), draw(48 * m))) + 1e-6).mean()
        dg = d2(draw(3 * n), draw(3 * n))
        pa, pb, pc, pd = (draw(4 * n) for _ in range(4))
        mask = (((pa + pb) % 5 != 0) & ((pc + pd) % 5 != 0)).float()
        la = 0.5 * torch.log(d2(pa, pb).clamp_min(1e-12))
        lb = 0.5 * torch.log(d2(pc, pd).clamp_min(1e-12))
        with torch.no_grad():
            sign = torch.sign(log_dx(pc, pd) - log_dx(pa, pb))
        l_ord = (mask * torch.relu(sign * (la - lb) + 0.1)).sum() / mask.sum()
        ramp = min(1.0, (it + 1) / (0.05 * steps))
        loss = l_attr + 15 * l_rep + ramp * (0.6 * (dg / (1 + dg)).mean() + 2 * l_ord)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()
        sched.step()
    return Y.detach().cpu().numpy()


FIT = {
    **{f"FloDR (w={w})": _flodr(float(w)) for w in (0, 1, 2, 3)},
    "openTSNE": _opentsne,
    "UMAP": _umap,
    "TriMap": _trimap,
    "PaCMAP": _pacmap,
    "LocalMAP": lambda X, seed, metric: _pacmap(X, seed, metric, "LocalMAP"),
    "UMATO": _umato,
    "ParamRepulsor": _paramrepulsor,
    "PHATE": _phate,
    "PyMDE": _pymde,
    "PCUMAP": lambda X, seed, metric: pcumap(X, seed)[0],
    "SQuadMDS": _squadmds,
    "PCA-2": _pca2,
    "Free layout (pca)": fit_free,
    "DREAMS": _dreams,
}


def fit(ds, method, seed, fn=None, X=None):
    # one cached layout per (data set, method, seed), with its wall time
    safe = "".join(c for c in method if c not in " ()=")
    path = os.path.join(CACHE, "layouts", ds, f"{safe}_s{seed}.npz")
    if not os.path.exists(path):
        X = load(ds)[0] if X is None else X
        t0 = time.perf_counter()
        Y = np.asarray((fn or FIT[method])(X, seed, metric(ds, method)), np.float32)
        secs = time.perf_counter() - t0
        assert Y.shape == (len(X), 2) and np.isfinite(Y).all(), f"{method} on {ds}"
        os.makedirs(os.path.dirname(path), exist_ok=True)
        np.savez_compressed(path, Y=Y, secs=secs)
    z = np.load(path)
    return z["Y"], float(z["secs"])


def knn(X, k=K, metric="euclid"):
    from sklearn.neighbors import NearestNeighbors

    Z, kw = (X.astype(bool), {"metric": "jaccard"}) if metric == "jaccard" else (X, {})
    idx = (
        NearestNeighbors(n_neighbors=k + 1, **kw)
        .fit(Z)
        .kneighbors(Z, return_distance=False)
    )
    return np.array([r[r != i][:k] for i, r in enumerate(idx)])


def knn_faiss(ref, query, k=K):
    import faiss

    index = faiss.IndexFlatL2(ref.shape[1])
    index.add(np.ascontiguousarray(ref, np.float32))
    return index.search(np.ascontiguousarray(query, np.float32), k + 1)[1]


class Scorer:
    # recall@15 and CPD (Spearman of pairwise distances) on all pairs and on the held-out pairs
    # (index sum divisible by five, which the ordinal term never samples); atlases from 50 000
    # anchors and 400 000 pairs, the same for every method
    def __init__(
        self, X, metric="euclid", large=False, n_anchor=50_000, n_pairs=400_000
    ):
        self.large = large
        if not large:
            from scipy.spatial.distance import pdist

            self.anchors = np.arange(len(X))
            self.hd = knn(X, K, metric)
            self.i, self.j = np.triu_indices(len(X), 1)
            self.dh = (
                pdist(X.astype(bool), "jaccard") if metric == "jaccard" else pdist(X)
            )
        else:
            self.anchors = np.sort(
                np.random.default_rng(777).permutation(len(X))[:n_anchor]
            )
            self.hd = np.array(
                [
                    r[r != i][:K]
                    for i, r in zip(self.anchors, knn_faiss(X, X[self.anchors]))
                ]
            )
            rng = np.random.default_rng(778)
            i, j = rng.integers(0, len(X), n_pairs), rng.integers(0, len(X), n_pairs)
            self.i, self.j = i[i != j], j[i != j]
            self.dh = np.linalg.norm(X[self.i] - X[self.j], axis=1)
        self.held = (self.i + self.j) % 5 == 0

    def __call__(self, Y):
        from scipy.stats import spearmanr

        Y = np.ascontiguousarray(Y, np.float32)
        if self.large:
            ld = np.array(
                [
                    r[r != i][:K]
                    for i, r in zip(self.anchors, knn_faiss(Y, Y[self.anchors]))
                ]
            )
        else:
            ld = knn(Y)
        recall = np.mean([len(set(a) & set(b)) for a, b in zip(self.hd, ld)]) / K
        Z = Y if self.large else Y.astype(np.float64)
        dl = np.linalg.norm(Z[self.i] - Z[self.j], axis=1)
        cpd = spearmanr(self.dh[self.held], dl[self.held]).statistic
        return float(recall), float(cpd), float(spearmanr(self.dh, dl).statistic)


_scorers = {}


def score(ds, Y):
    if ds not in _scorers:
        _scorers[ds] = Scorer(load(ds)[0], metric(ds, "FloDR"), ds in ATLASES)
    return _scorers[ds](Y)


def scores(ds, method, seeds=SEEDS):
    # (recall, CPD) per seed, cached with the layouts
    path = os.path.join(CACHE, "scores.json")
    res = json.load(open(path)) if os.path.exists(path) else {}
    for s in seeds:
        key = f"{ds}/{method}/{s}"
        if key not in res:
            res[key] = score(ds, fit(ds, method, s)[0])
            json.dump(res, open(path, "w"), indent=1)
    return np.array([res[f"{ds}/{method}/{s}"] for s in seeds])


# figure style: Helvetica throughout, sizes at print width (183 mm)
TEXT_W = 7.2
RC = {
    "font.size": 8,
    "text.usetex": True,
    "font.family": "sans-serif",
    "axes.unicode_minus": False,
    "axes.titlepad": 4.0,
    "text.latex.preamble": r"\usepackage{amsmath}\usepackage[T1]{fontenc}\usepackage[scaled=0.95]{helvet}"
    r"\renewcommand{\familydefault}{\sfdefault}\usepackage{sansmath}\sansmath",
}
OURS, OURS_DARK, OURS_LIGHT = "#AA4499", "#882255", "#E3B6D6"
PARTNER = "#44AA99"
TOL = (
    "#CC6677",
    "#332288",
    "#DDCC77",
    "#117733",
    "#88CCEE",
    "#882255",
    "#44AA99",
    "#999933",
    "#AA4499",
    "#77AADD",
    "#99DDFF",
    "#BBCC33",
    "#EE8866",
    "#FFAABB",
    "#AAAA00",
    "#EEDD88",
    "#44BB99",
    "#BBBBBB",
)
METHOD_STYLE = {
    "FloDR": (OURS, "o"),
    "FloDR + optA": (OURS_DARK, "D"),
    "openTSNE": ("#4477AA", "o"),
    "UMAP": ("#117733", "s"),
    "TriMap": ("#999933", "^"),
    "PaCMAP": ("#332288", "D"),
    "LocalMAP": ("#88CCEE", "v"),
    "UMATO": ("#44AA99", "P"),
    "ParamRepulsor": ("#C9B458", "X"),
    "PHATE": ("#77AADD", "*"),
    "PyMDE": ("#AA7744", "h"),
    "PCUMAP": ("#EE8866", ">"),
    "SQuadMDS": ("#44546A", "<"),
    "PCA-2": ("#888888", "p"),
    "Parametric UMAP": ("#EE8866", ">"),
    "kNN interp.": ("#222222", "v"),
    "DREAMS": ("#225555", "d"),
    "densMAP": ("#BBCC33", "x"),
}


def cmap(stops):
    from matplotlib.colors import LinearSegmentedColormap

    return LinearSegmentedColormap.from_list("c", stops)


SEQ = [(0.0, "#2D1E4A"), (0.35, "#6E2066"), (0.65, "#B9479B"), (1.0, "#F2A7D2")]


def class_colours(labels):
    # the same class gets the same colour in every figure of an atlas
    import matplotlib.pyplot as plt

    labels = np.asarray(labels).astype(str)
    uniq = np.unique(labels)
    base = list(TOL) + [
        plt.matplotlib.colors.to_hex(c) for c in plt.get_cmap("tab20b").colors
    ]
    pal = {c: base[i % len(base)] for i, c in enumerate(uniq)}
    return np.array([pal[c] for c in labels]), pal


def square(ax, Y, pad=0.03, crop=None):
    # a square box at 1:1 data scale
    Y = np.asarray(Y, float)[:, :2]
    lo, hi = (Y.min(0), Y.max(0)) if crop is None else np.percentile(Y, crop, axis=0)
    rad, (cx, cy) = float((hi - lo).max()) * (0.5 + pad), (lo + hi) / 2
    ax.set_box_aspect(1)
    ax.set_aspect("equal", adjustable="datalim")
    ax.set_xlim(cx - rad, cx + rad)
    ax.set_ylim(cy - rad, cy + rad)
    ax.set_xticks([])
    ax.set_yticks([])


def letter(ax, s, dx=-6):
    ax.annotate(
        r"\textbf{" + s + "}",
        xy=(0, 1),
        xycoords="axes fraction",
        xytext=(dx, 4),
        textcoords="offset points",
        fontsize=9,
        ha="right",
        va="baseline",
        annotation_clip=False,
    )


def save(fig, name):
    fig.savefig(out(name), dpi=300, bbox_inches="tight", pad_inches=0.02)
