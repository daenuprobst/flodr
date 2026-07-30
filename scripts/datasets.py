import io
import os
import urllib.request

import numpy as np
from sklearn.datasets import fetch_openml, make_swiss_roll
from sklearn.decomposition import PCA

from flodr import FloDR

CACHE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "cache")
os.makedirs(CACHE, exist_ok=True)
N_CAP = 5000


def _std_global(X):
    return ((X - X.mean()) / X.std()).astype(np.float32)


def _std_perfeat(X):
    X = np.asarray(X, np.float64)
    return ((X - X.mean(0)) / (X.std(0) + 1e-9)).astype(np.float32)


def _openml(name, n=N_CAP, seed=0):
    X, y = fetch_openml(name, version=1, return_X_y=True, as_frame=False, parser="liac-arff")
    X = X.astype(np.float32)
    y = np.unique(y, return_inverse=True)[1].astype(int)
    if len(X) > n:
        subset = np.random.default_rng(seed).permutation(len(X))[:n]
        X, y = X[subset], y[subset]
    return X, y


def get_mnist():
    X, y = _openml("mnist_784")
    return _std_global(X), y, "MNIST", "raw pixels"


def get_fmnist():
    X, y = _openml("Fashion-MNIST")
    return _std_global(X), y, "Fashion-MNIST", "raw pixels"


def get_paul15():
    import scanpy as sc
    sc.settings.verbosity = 0
    adata = sc.datasets.paul15()
    labels = np.unique(adata.obs["paul15_clusters"], return_inverse=True)[1]
    sc.pp.filter_genes(adata, min_counts=1)
    sc.pp.normalize_total(adata, target_sum=1e4)
    sc.pp.log1p(adata)
    sc.pp.highly_variable_genes(adata, n_top_genes=1000)
    adata = adata[:, adata.var.highly_variable].copy()
    sc.pp.scale(adata, max_value=10)
    pcs = PCA(n_components=50, svd_solver="randomized", random_state=0).fit_transform(
        np.asarray(adata.X, np.float64))
    return _std_perfeat(pcs), labels, "paul15 (scRNA-seq)", "HVG-1000 -> PCA-50"


def get_drfp():
    import pandas as pd
    from drfp import DrfpEncoder
    npz = f"{CACHE}/drfp_schneider50k.npz"
    if os.path.exists(npz):
        cached = np.load(npz)
        return cached["X"], cached["y"], "Schneider 50k (reactions)", "DRFP, 2048 bits"
    url = "https://raw.githubusercontent.com/reymond-group/drfp/main/data/schneider50k.tsv"
    df = pd.read_csv(io.BytesIO(urllib.request.urlopen(url, timeout=600).read()),
                     sep="\t", index_col=0)
    supercls = df.rxn_class.astype(str).str.split(".").str[0]
    df = df.groupby(supercls, group_keys=False).apply(
        lambda grp: grp.sample(max(1, round(N_CAP * len(grp) / len(supercls))), random_state=0))
    supercls = df.rxn_class.astype(str).str.split(".").str[0]
    classes = sorted(supercls.unique(), key=int)
    labels = supercls.map({cls: i for i, cls in enumerate(classes)}).to_numpy(dtype=int)
    X = np.array(DrfpEncoder.encode(df.rxn.tolist(), n_folded_length=2048), dtype=np.float32)
    X = (X > 0).astype(np.float32)
    np.savez_compressed(npz, X=X, y=labels)
    return X, labels, "Schneider 50k (reactions)", "DRFP, 2048 bits"


PAPER = {"mnist": get_mnist, "fmnist": get_fmnist, "paul15": get_paul15, "drfp": get_drfp}
# everything else is Euclidean
METRIC = {"drfp": "jaccard"}
# these cannot consume a non-Euclidean metric, fitted in Euclidean and scored in
# the metric the row is measured in
EUCLID_ONLY = ("TriMap", "PaCMAP", "PCA-2", "LocalMAP", "PHATE", "PyMDE", "PCUMAP",
               "SQuadMDS")


def make_ring(n=6000, d_noise=16, seed=0, r_lo=0.5, r_hi=3.0, noise=0.02):
    rng = np.random.default_rng(seed)
    base = rng.uniform(-4, 4, (n, 2)).astype(np.float64)
    radius = r_lo + (r_hi - r_lo) * (base[:, 0] + 4) / 8
    theta = rng.uniform(0, 2 * np.pi, n)
    fiber = np.column_stack([radius * np.cos(theta), radius * np.sin(theta)])
    eps = rng.normal(0, noise, (n, 2 + 2 + d_noise))
    X = np.concatenate([base * 2.0, fiber, np.zeros((n, d_noise))], 1) + eps
    return X.astype(np.float32), np.sqrt(radius ** 2 + noise ** 2 * (4 + d_noise))


def make_hetero(n=6000, k=8, d_fiber=28, seed=0, s_lo=0.1, s_hi=3.0, sep=9.0,
                base_noise=0.6):
    rng = np.random.default_rng(seed)
    labels = rng.integers(0, k, n)
    angles = 2 * np.pi * np.arange(k) / k
    centers = sep * np.column_stack([np.cos(angles), np.sin(angles)])
    sigma = np.geomspace(s_lo, s_hi, k)
    base = centers[labels] + rng.normal(0, base_noise, (n, 2))
    fiber = rng.normal(0, 1.0, (n, d_fiber)) * sigma[labels, None]
    X = np.concatenate([base, fiber], 1)
    return X.astype(np.float32), (sigma[labels] * np.sqrt(d_fiber)).astype(np.float64), labels


def make_blob(n=6000, d=30, seed=0):
    return np.random.default_rng(seed).normal(0, 1.0, (n, d)).astype(np.float32)


def make_hidden(n=6000, d_fiber=28, lam=0.0, seed=0, sep_base=4.0, s_fiber=(0.7, 1.5),
                u_noise=1.0):
    rng = np.random.default_rng(seed)
    C = np.array([[-sep_base / 2, 0.0], [sep_base / 2, 0.0]])
    g = rng.integers(0, 2, n)
    u = (1 - lam) * C[g] + rng.normal(0, u_noise, (n, 2))
    # per-class variance
    s2 = 1.0 + lam * (np.asarray(s_fiber, float) ** 2 - 1.0)
    f = rng.normal(0, 1.0, (n, d_fiber)) * np.sqrt(s2[g])[:, None]
    X = np.concatenate([u, f], 1).astype(np.float32)
    var_mat = np.repeat(s2[:, None], d_fiber, axis=1)
    truth = dict(C=C, V=var_mat, d_fiber=d_fiber, lam=lam, u_noise=u_noise)
    return X, u.astype(np.float64), g, truth


def _post_ll(u, f, truth):
    C, var_mat = truth["C"], truth["V"]
    mu = (1 - truth["lam"]) * C
    ll = -0.5 * ((u[:, None, :] - mu[None]) ** 2).sum(-1) / truth.get("u_noise", 1.0) ** 2
    if f is not None:
        ll = ll - 0.5 * (np.log(var_mat).sum(1) + (f[:, None, :] ** 2 / var_mat[None]).sum(-1))
    return ll


def hidden_truth(u, truth, n_mc=512, seed=0):
    var_mat = truth["V"]
    ll_u = _post_ll(u, None, truth)
    lu = ll_u - ll_u.max(1, keepdims=True)
    # p(g|u)
    post = np.exp(lu); post /= post.sum(1, keepdims=True)
    H_u = -(post * np.log(np.maximum(post, 1e-300))).sum(1)
    rng = np.random.default_rng(seed)
    ent_acc = np.zeros(len(u))
    # E_{f|u} H(G|u,f)
    for _ in range(n_mc):
        # g ~ p(g|u)
        cls = (rng.random(len(u)) >= post[:, 0]).astype(int)
        f = rng.normal(0, 1.0, (len(u), var_mat.shape[1])) * np.sqrt(var_mat[cls])
        ll = _post_ll(u, f, truth)
        ll -= ll.max(1, keepdims=True)
        p = np.exp(ll); p /= p.sum(1, keepdims=True)
        ent_acc += -(p * np.log(np.maximum(p, 1e-300))).sum(1)
    return H_u - ent_acc / n_mc


def hidden_total_mi(truth, n_mc=200_000, seed=0):
    C, var_mat, d_fiber = truth["C"], truth["V"], truth["d_fiber"]
    lam = truth["lam"]
    rng = np.random.default_rng(seed)
    g = rng.integers(0, 2, n_mc)
    u = (1 - lam) * C[g] + rng.normal(0, truth.get("u_noise", 1.0), (n_mc, 2))
    f = rng.normal(0, 1.0, (n_mc, d_fiber)) * np.sqrt(var_mat[g])

    def mi(ll):
        ll = ll - ll.max(1, keepdims=True)
        p = np.exp(ll); p /= p.sum(1, keepdims=True)
        return float(np.log(2) + (p * np.log(np.maximum(p, 1e-300))).sum(1).mean())

    return mi(_post_ll(u, None, truth)), mi(_post_ll(u, f, truth))


def _toy_hetero():
    X, _, labels = make_hetero(seed=0)
    return X, labels, "clustered synthetic", "8 clusters, 30x spread range"


def _toy_ring():
    X, _ = make_ring(seed=0)
    return X, np.digitize(X[:, 0], np.quantile(X[:, 0], np.linspace(0, 1, 9)[1:-1])), \
        "fibered ring", "thin circle fibers"


def _toy_swiss():
    X, t = make_swiss_roll(6000, noise=0.05, random_state=0)
    return X.astype(np.float32), \
        np.digitize(t, np.quantile(t, np.linspace(0, 1, 9)[1:-1])), "swiss roll", "3D"


TOY = {"hetero": _toy_hetero, "ring": _toy_ring, "swiss": _toy_swiss}
ALL = {**PAPER, **TOY}


def load_mnist(n=20000, seed=0):
    return _openml("mnist_784", n=n, seed=seed)


def _ds_mnist20k():
    X, y = load_mnist()
    return X, y, "MNIST", "raw pixels, n=20000"


DATASETS = {"mnist": _ds_mnist20k, "hetero": _toy_hetero, "ring": _toy_ring,
            "swiss": _toy_swiss}


def fit_flodr(X, seed=0, w=2.0, device=None, density=True, gate_max=0.5):
    return FloDR(w=w, random_state=seed, device=device, density=density,
                 advanced=dict(gate_max=gate_max) if gate_max else None).fit(X)
