from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import scanpy as sc
from openTSNE import TSNE
from scipy.stats import spearmanr
from sklearn.datasets import fetch_openml
from sklearn.decomposition import PCA
from sklearn.neighbors import NearestNeighbors
from umap import UMAP

from flodr import FloDR, viz

OUT = Path(__file__).resolve().parents[1] / "docs" / "docs" / "images"
SEEDS = (0, 1, 2)


def standardised_pcs(X):
    P = PCA(50, svd_solver="randomized", random_state=0).fit_transform(X)
    return ((P - P.mean(0)) / P.std(0)).astype(np.float32)


def paul15():
    a = sc.datasets.paul15()
    sc.pp.filter_genes(a, min_counts=1)
    sc.pp.normalize_total(a, target_sum=1e4)
    sc.pp.log1p(a)
    sc.pp.highly_variable_genes(a, n_top_genes=1000, subset=True)
    sc.pp.scale(a, max_value=10)
    return standardised_pcs(np.asarray(a.X, np.float64))


def fashion_mnist(n=10000):
    X = fetch_openml(
        "Fashion-MNIST", version=1, return_X_y=True, as_frame=False, parser="liac-arff"
    )[0]
    return standardised_pcs(X[np.random.default_rng(0).permutation(len(X))[:n]])


METHODS = {
    "FloDR": lambda X, s: FloDR(random_state=s, verbose=False).fit_transform(X),
    "FloDR, w = 0": lambda X, s: FloDR(
        w=0, random_state=s, verbose=False
    ).fit_transform(X),
    "UMAP": lambda X, s: UMAP(init="pca", random_state=s).fit_transform(X),
    "openTSNE": lambda X, s: np.asarray(TSNE(random_state=s, n_jobs=8).fit(X)),
    "PCA": lambda X, s: PCA(2).fit_transform(X),
}
STYLES = {
    "FloDR": ("#c25e08", "*"),
    "FloDR, w = 0": ("#f0a35e", "*"),
    "UMAP": ("#4477AA", "o"),
    "openTSNE": ("#228833", "s"),
    "PCA": ("0.5", "D"),
}


def scores(X, Y, k=15, pairs=500_000):
    # recall@15 of input neighbours, and the rank correlation of distances (CPD) over
    # random pairs
    a, b = (
        NearestNeighbors(n_neighbors=k + 1).fit(Z).kneighbors(Z)[1][:, 1:]
        for Z in (X, Y)
    )
    recall = np.mean([len(set(p) & set(q)) for p, q in zip(a, b)]) / k
    i, j = np.random.default_rng(0).integers(0, len(X), (2, pairs))
    d = [np.linalg.norm(Z[i] - Z[j], axis=1) for Z in (X, Y)]
    return recall, spearmanr(*d)[0]


plt.rcParams.update(viz.RC)
fig, axes = plt.subplots(1, 2, figsize=(10, 4.6))
for ax, (name, X) in zip(
    axes, (("paul15", paul15()), ("Fashion-MNIST, 10,000 images", fashion_mnist()))
):
    results = {}
    for method, fit in METHODS.items():
        runs = [scores(X, fit(X, s)) for s in SEEDS]
        results[method] = tuple(zip(*runs))
        recall, cpd = np.mean(runs, 0)
        print(f"{name}, {method}: recall {recall:.3f}, CPD {cpd:.3f}")
    viz.plot_frontier(results, STYLES, ax)
    ax.set_title(name)
axes[0].legend(frameon=False, fontsize=9)
viz.save(fig, OUT / "frontier.png", dpi=200)
