import functools
import math
import os

import numpy as np

import common
from common import ATLASES, BENCHMARKS, K

DIR = os.path.join(common.CACHE, "fig6")
FORWARD, OPTA = "FloDR (forward pass)", "FloDR + optA"
S = common.METHOD_STYLE
STYLE = {
    FORWARD: S["FloDR"],
    OPTA: S["FloDR + optA"],
    "openTSNE transform": S["openTSNE"],
    "UMAP transform": S["UMAP"],
    **{m: S[m] for m in ("Parametric UMAP", "ParamRepulsor", "kNN interp.", "PCA-2")},
}


@functools.lru_cache
def donors(ds):
    import anndata

    a = anndata.read_h5ad(f"{common.DATA}/{common.ATLAS_FILES[ds][0]}", backed="r")
    # the seed has no donor column; its barcodes end in one of six sample ids
    if ds == "plant":
        return np.array([s.rsplit("_", 1)[1] for s in a.obs_names])
    return a.obs["donor_id"].astype(str).to_numpy()


def split(ds, rep, n):
    if ds in BENCHMARKS:
        perm = np.random.default_rng(10_000 + rep).permutation(n)
        return perm[n // 5 :], perm[: n // 5]
    # whole donors held out, so no held-out cell has a same-donor twin in training
    g = donors(ds)
    uniq = np.unique(g)
    test = uniq[np.random.default_rng(20_000 + rep).permutation(len(uniq))]
    test = np.isin(g, test[: math.ceil(len(uniq) / 5)])
    return np.flatnonzero(~test), np.flatnonzero(test)


# one fit per job gives (training layout, held-out layout) for each method it serves
def flodr(Xtr, Xte, rep, metric):
    from flodr import FloDR

    est = FloDR(w=2.0, random_state=rep, verbose=False).fit(Xtr)
    Y = est.embedding_
    return {FORWARD: (Y, est.transform(Xte)), OPTA: (Y, est.transform_opt(Xte))}


def umap_(Xtr, Xte, rep, metric):
    import umap
    from sklearn.neighbors import NearestNeighbors

    kw = dict(n_neighbors=K, min_dist=0.1, init="pca", metric=metric)
    est = umap.UMAP(random_state=rep, **kw).fit(Xtr)
    Y = est.embedding_
    # the floor: a held-out point at the inverse-distance mean of its training
    # neighbours on UMAP's own training map
    Ztr, Zte = (Xtr > 0, Xte > 0) if metric == "jaccard" else (Xtr, Xte)
    dist, idx = NearestNeighbors(n_neighbors=K, metric=metric).fit(Ztr).kneighbors(Zte)
    w = 1.0 / (dist + 1e-9)
    knn = (w[..., None] * Y[idx]).sum(1) / w.sum(1, keepdims=True)
    return {"UMAP transform": (Y, est.transform(Xte)), "kNN interp.": (Y, knn)}


def opentsne(Xtr, Xte, rep, metric):
    from openTSNE import TSNE

    Y = TSNE(perplexity=30, random_state=rep, n_jobs=8, metric=metric).fit(Xtr)
    return {"openTSNE transform": (np.asarray(Y), np.asarray(Y.transform(Xte)))}


def pcumap(Xtr, Xte, rep, metric):
    return {"Parametric UMAP": common.pcumap(Xtr, rep, Xte)}


def paramrepulsor(Xtr, Xte, rep, metric):
    from parampacmap import ParamPaCMAP

    est = ParamPaCMAP(n_components=2, n_neighbors=K, seed=rep, verbose=False)
    Y = est.fit_transform(Xtr)
    return {"ParamRepulsor": (Y, est.transform(Xte))}


def pca2(Xtr, Xte, rep, metric):
    from sklearn.decomposition import PCA

    est = PCA(2, svd_solver="randomized", random_state=rep).fit(Xtr)
    return {"PCA-2": (est.transform(Xtr), est.transform(Xte))}


def held_nbrs(X, te, metric="euclid"):
    # neighbours in the whole set, train and test together, the point itself removed
    if metric == "jaccard":
        return common.knn(X, K, "jaccard")[te]
    return np.array([r[r != i][:K] for i, r in zip(te, common.knn_faiss(X, X[te]))])


def scores(ds, rep):
    from scipy.stats import spearmanr

    X, metric = common.load(ds)[0], common.metric(ds, "UMAP")
    tr, te = split(ds, rep, len(X))
    layouts = {}
    for job in (flodr, umap_, opentsne, pcumap, paramrepulsor, pca2):

        def make():
            print(f"{ds} {job.__name__} r{rep}", flush=True)
            lib = "jaccard" if metric == "jaccard" else "euclidean"
            out = {}
            for m, (Ytr, Yte) in job(X[tr], X[te], rep, lib).items():
                out[m] = np.empty((len(X), 2), np.float32)
                out[m][tr], out[m][te] = Ytr, Yte
            return out

        layouts |= common.cached(f"{DIR}/{ds}_{job.__name__}_r{rep}.npz", make)
    # CPD on 20 random partners per held-out point, at most 100 000 pairs
    rng = np.random.default_rng(777)
    idx = te if len(te) * 20 <= 100_000 else rng.choice(te, 5000, replace=False)
    i = np.repeat(idx, 20)
    j = rng.integers(0, len(X), len(i))
    if metric == "jaccard":
        a, b = X[i].astype(bool), X[j].astype(bool)
        dx = 1.0 - (a & b).sum(1) / np.maximum((a | b).sum(1), 1)
    else:
        dx = np.linalg.norm(X[i] - X[j], axis=1)
    nn = held_nbrs(X, te, metric)
    out = {}
    for m, Y in layouts.items():
        recall = np.mean([len(set(a) & set(b)) for a, b in zip(nn, held_nbrs(Y, te))])
        dy = np.linalg.norm(Y[i] - Y[j], axis=1)
        out[m] = recall / K, spearmanr(dx, dy).statistic
    return out


if __name__ == "__main__":
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from flodr.viz import plot_frontier

    plt.rcParams.update({**common.RC, "axes.linewidth": 0.6})
    plt.rcParams.update({"xtick.major.width": 0.6, "ytick.major.width": 0.6})
    fig, axes = plt.subplots(2, 4, figsize=(common.TEXT_W, 3.9))
    for n, ds in enumerate(BENCHMARKS + ATLASES):
        runs = [scores(ds, r) for r in range(5 if ds in BENCHMARKS else 3)]
        results = {m: np.array([s[m] for s in runs]).T for m in STYLE}
        for m, (r, c) in results.items():
            print(f"{ds:10s} {m:20s} recall {r.mean():.4f} CPD {c.mean():.4f}")
        xlabel = "held-out recall@15" if n >= 4 else ""
        ylabel = "" if n % 4 else "held-out CPD"
        ax = plot_frontier(results, STYLE, axes.flat[n], xlabel, ylabel)
        ax.set_title({**common.NAMES, **common.NAMES_2L}[ds], fontsize=8, loc="left")
        common.letter(ax, "abcdefg"[n])
    axes.flat[7].axis("off")
    kw = dict(frameon=False, fontsize=7, handletextpad=0.4, labelspacing=0.55)
    axes.flat[7].legend(*ax.get_legend_handles_labels(), loc="center", **kw)
    fig.tight_layout(w_pad=0.6, h_pad=0.7)
    common.save(fig, "fig6.pdf")
