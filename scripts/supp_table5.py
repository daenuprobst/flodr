import os

import numpy as np
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import dijkstra
from scipy.stats import rankdata

import common
from common import K, SEEDS

TITLE = {
    "mnist": "MNIST",
    "fmnist": "Fashion-MNIST",
    "paul15": "paul15",
    "drfp": "Schneider50k",
    "bmarrow": "bone marrow",
    "cerebellum": "cerebellum",
    "plant": "seed",
}


def tuned(method, **kw):
    # a baseline of common.FIT at other settings
    def go(X, seed, metric):
        m = "jaccard" if metric == "jaccard" else "euclidean"
        if method == "UMAP":
            import umap

            return umap.UMAP(
                n_neighbors=K, random_state=seed, init="pca", metric=m, **kw
            ).fit_transform(X)
        if method == "openTSNE":
            from openTSNE import TSNE

            return np.asarray(
                TSNE(perplexity=30, random_state=seed, n_jobs=8, metric=m, **kw).fit(X)
            )
        import pacmap

        return pacmap.PaCMAP(
            n_neighbors=K, random_state=seed, verbose=False, **kw
        ).fit_transform(X)

    return go


GRID = (
    [
        ("UMAP", dict(min_dist=md, negative_sample_rate=nsr))
        for md in (0.001, 0.01, 0.1, 0.5)
        for nsr in (5, 15)
        if (md, nsr) != (0.1, 5)
    ]
    + [("openTSNE", dict(exaggeration=e)) for e in (2, 4, 8, 12)]
    + [
        ("PaCMAP", dict(MN_ratio=mn, FP_ratio=fp))
        for mn in (0.5, 2)
        for fp in (1, 2, 4)
        if (mn, fp) != (0.5, 2)
    ]
)
TUNED = {
    m + " " + " ".join(f"{k}={v:g}" for k, v in kw.items()): tuned(m, **kw)
    for m, kw in GRID
}
UMAPS = ["UMAP"] + [m for m in TUNED if m.startswith("UMAP")]
# the baselines of the main comparison and their tuned settings
OTHERS = [m for m in common.FIT if not m.startswith(("FloDR", "Free", "DREAMS"))]
OTHERS += list(TUNED)


def label(m):
    # md: minimum distance, nsr: negative samples per edge
    kw = dict(p.split("=") for p in m.split()[1:])
    if m == "UMAP":
        return "default"
    if m.startswith("UMAP"):
        return f"md ${kw['min_dist']}$, nsr {kw['negative_sample_rate']}"
    if "exaggeration" in kw:
        return rf"openTSNE ex.\,{kw['exaggeration']}"
    return m


def cranks(v):
    # spearman as the dot product of centred, normalised average ranks
    r = rankdata(v)
    r -= r.mean()
    return r / np.linalg.norm(r)


def scores(ds):
    # rank correlation of map distances with shortest paths along the symmetrised 15-NN
    # graph of the input, from 300 fixed sources to every reachable cell
    def make():
        print("geodesic CPD on", ds, flush=True)
        X = common.load(ds)[0]
        n = len(X)
        if common.metric(ds, "FloDR") == "jaccard":
            idx = common.knn(X, K, "jaccard")
            B = X.astype(bool)
            inter, union = (B[:, None] & B[idx]).sum(2), (B[:, None] | B[idx]).sum(2)
            dist = (union - inter) / union
        else:
            idx = np.array(
                [r[r != i][:K] for i, r in enumerate(common.knn_faiss(X, X))]
            )
            X = X.astype(np.float64)
            dist = np.stack(
                [np.linalg.norm(X - X[idx[:, k]], axis=1) for k in range(K)], 1
            )
        # the offset keeps duplicates connected
        rows = np.repeat(np.arange(n), K)
        A = coo_matrix((dist.ravel() + 1e-9, (rows, idx.ravel())), shape=(n, n)).tocsr()
        src = np.sort(np.random.default_rng(0).choice(n, 300, replace=False))
        G = dijkstra(A.maximum(A.T), directed=False, indices=src)
        ok = np.isfinite(G)
        ok[np.arange(300), src] = False
        k, b = np.nonzero(ok)
        a, rg = src[k], cranks(G[k, b])
        res = {}
        for m in ["FloDR (w=2)", "FloDR (w=0)", "Free layout (pca)", *OTHERS]:
            Ys = [
                common.fit(ds, m, s, fn=TUNED.get(m))[0].astype(np.float64)
                for s in SEEDS
            ]
            res[m] = [
                float(rg @ cranks(np.sqrt(((Y[a] - Y[b]) ** 2).sum(1)))) for Y in Ys
            ]
        return res

    path = os.path.join(common.CACHE, "supp_table5", "scores.json")
    return {m: np.array(v) for m, v in common.memo(path, ds, make).items()}


if __name__ == "__main__":
    lines = []
    for ds, title in TITLE.items():
        sc = scores(ds)
        fl, free = sc["FloDR (w=2)"], sc["Free layout (pca)"]
        best = max(UMAPS, key=lambda m: sc[m].mean())
        other = max(OTHERS, key=lambda m: sc[m].mean())
        d = fl - sc[best]
        cells = [
            title,
            f"${fl.mean():.3f}\\pm{fl.std(ddof=1):.3f}$",
            f"${sc['FloDR (w=0)'].mean():.3f}$",
            label(best),
            f"${sc[best].mean():.3f}$",
            f"${d.mean() / d.std(ddof=1):.1f}$",
            f"${free.mean():.3f}$",
            f"{label(other)} ${sc[other].mean():.3f}$",
        ]
        lines.append(" & ".join(cells) + r" \\")
    open(common.out("supp_table5.tex"), "w").write("\n".join(lines) + "\n")
    print("\n".join(lines))
