from functools import partial
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import scanpy as sc
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
from scipy.spatial import cKDTree
from scipy.spatial.distance import pdist
from scipy.stats import spearmanr
from sklearn.decomposition import PCA

from flodr import FloDR, audit, viz
from flodr.conformal import conformal_bands, sample_fibre

OUT = Path(__file__).resolve().parents[1] / "docs" / "docs" / "images"
# Paul et al. number the erythroid clusters from the most mature (1) to the least
PATHS = [
    (["7MEP", "6Ery", "5Ery", "4Ery", "3Ery", "2Ery", "1Ery"], ["Gata1", "Hba-a2"]),
    ([("9GMP", "10GMP"), ("16Neu", "17Neu")], ["Mpo", "Elane"]),
]

# log expression of the 1,000 most variable genes and its 50 principal components,
# unscaled: on 2,730 cells the trailing ones are mostly noise, and standardising would
# weigh them like the leading ones
adata = sc.datasets.paul15()
clusters = np.asarray(adata.obs["paul15_clusters"]).astype(str)
types = np.char.lstrip(clusters, "0123456789")
sc.pp.filter_genes(adata, min_counts=1)
sc.pp.normalize_total(adata, target_sum=1e4)
sc.pp.log1p(adata)
sc.pp.highly_variable_genes(adata, n_top_genes=1000, subset=True)
sc.pp.scale(adata, max_value=10)
pca = PCA(50, svd_solver="randomized", random_state=0)
X = pca.fit_transform(np.asarray(adata.X, np.float64)).astype(np.float32)
genes = list(adata.var_names)
mu, sd = adata.var["mean"].to_numpy(), adata.var["std"].to_numpy()
perm = np.random.default_rng(0).permutation(len(X))
test, train = perm[: len(X) // 5], perm[len(X) // 5 :]


def expression(X, names):
    # log expression of genes, read linearly from the components
    j = [genes.index(g) for g in names]
    return X @ (pca.components_[:, j] * sd[j]) + pca.mean_[j] * sd[j] + mu[j]


def scores(X, Y, k=15):
    # recall@15 of input neighbours, and the rank correlation of all pairwise distances
    a = cKDTree(X).query(X, k + 1)[1][:, 1:]
    b = cKDTree(Y).query(Y, k + 1)[1][:, 1:]
    recall = np.mean([len(set(p) & set(q)) for p, q in zip(a, b)]) / k
    return recall, spearmanr(pdist(X), pdist(Y))[0]


# the distance-order term (w) and the heat start, on the same cells
plt.rcParams.update(viz.RC)
pal = viz.palette(types)
fits = {
    "w = 0": dict(w=0),
    "w = 2 (default)": dict(),
    'w = 2, axes="heat"': dict(axes="heat"),
}
models = {}
fig, axes = plt.subplots(1, 3, figsize=(9, 3.3))
fig.subplots_adjust(left=0.02, right=0.98, top=0.92, bottom=0.07, wspace=0.08)
for ax, (name, kw) in zip(axes, fits.items()):
    m = models[name] = FloDR(random_state=0, verbose=False, **kw).fit(X[train])
    recall, cpd = scores(X[train], m.embedding_)
    print(f"{name}: recall {recall:.3f}, CPD {cpd:.3f}")
    text = f"recall@15 {recall:.3f}\nCPD {cpd:.3f}"
    viz.plot_embedding(m.embedding_, types[train], ax=ax, colours=pal, s=6, text=text)
    ax.set_title(name)
handles = [
    Line2D([], [], ls="", marker="o", ms=5, c=c, label=t) for t, c in pal.items()
]
fig.legend(
    handles=handles,
    loc="upper center",
    bbox_to_anchor=(0.5, 0.07),
    ncol=10,
    frameon=False,
)
viz.save(fig, OUT / "paul15_maps.png", dpi=200)
model = models["w = 2 (default)"]
Y = model.embedding_
tree = cKDTree(Y)


def medoid(groups):
    i = np.flatnonzero(np.isin(clusters[train], groups))
    return Y[i[np.argmin(((Y[i] - np.median(Y[i], 0)) ** 2).sum(1))]]


def route(stages, n=160):
    # straight segments through the stage medoids, at equal arc length
    Q = np.array([medoid(s) for s in stages])
    s = np.r_[0, np.cumsum(np.linalg.norm(np.diff(Q, axis=0), axis=1))]
    t = np.linspace(0, s[-1], n)
    return np.stack([np.interp(t, s, Q[:, j]) for j in (0, 1)], 1).astype(np.float32)


# genes decoded along two lineages: exact draws from p(x | y) at each position, their
# 90% conformal band calibrated on held-out cells, and the mean of the 30 nearest cells
paths = [route(stages) for stages, _ in PATHS]
gap = np.percentile(tree.query(Y, k=2)[0][:, 1], 99)
far = [tree.query(p)[0] > gap for p in paths]
ends = [
    (("MEP", pal["MEP"]), ("Ery", pal["Ery"])),
    (("GMP", pal["GMP"]), ("Neu", pal["Neu"])),
]
on = np.isin(types[train], ["MEP", "Ery", "GMP", "Neu"])
fig = plt.figure(figsize=(9, 3.9))
gs = fig.add_gridspec(1, 2, width_ratios=(1, 1.4), wspace=0.15)
viz.plot_paths(Y, paths, types[train], pal, ends, far, on, fig.add_subplot(gs[0]))
bands = gs[1].subgridspec(2, 2, hspace=0.35, wspace=0.25)
for p, (path, (_, names)) in enumerate(zip(paths, PATHS)):
    read = partial(expression, names=names)
    band = conformal_bands(model, X[test], path, readout=read)
    draws = expression(sample_fibre(model, path, n_draws=256)[0], names).mean(0)
    near = expression(X[train][tree.query(path, k=30)[1]], names).mean(1)
    for g, name in enumerate(names):
        ax = fig.add_subplot(bands[g, p])
        lo, hi = band["lo"][:, g], band["hi"][:, g]
        viz.plot_band(lo, hi, draws[:, g], near[:, g], far[p], name, ax)
        ax.set_xticklabels([ends[p][0][0], ends[p][1][0]])
key = [
    Patch(color="#DCD3DC", label="90% conformal band"),
    Line2D([], [], c="#AA4499", label="mean of 256 exact draws"),
    Line2D(
        [], [], ls="", marker="_", c="#2E8B7A", label="mean of the 30 nearest cells"
    ),
]
fig.legend(
    handles=key, loc="upper center", bbox_to_anchor=(0.66, 0.02), ncol=3, frameon=False
)
viz.save(fig, OUT / "paul15_paths.png", dpi=200)

# the coverage certificate on the held-out cells
cert = model.coverage_certificate(X[test])
print(f"coverage certificate: {cert['certifies']}")
fig, ax = plt.subplots(figsize=(4, 4))
ranks = [cert["fibre"]["ranks"], cert["map"]["ranks"]]
viz.plot_ranks(ranks, ax=ax, colours=["#c25e08", "#4477AA"])
ax.legend(ax.lines[-2:], ["fibre", "map"], frameon=False, loc="upper left")
ax.set(ylabel="fraction of cells", title=f"{len(test)} held-out cells")
viz.save(fig, OUT / "paul15_certificate.png", dpi=130)

# what a map hides about the cell type, for this map and for the first two principal
# components, on one colour scale
maps = {"FloDR": Y, "PCA": X[train][:, :2]}
res = {
    k: audit(v, X=X[train], labels=types[train], seed=0, verbose=False)
    for k, v in maps.items()
}
top = max(np.quantile(np.clip(r.lower, 0, None), 0.99) for r in res.values())
fig, axes = plt.subplots(1, 2, figsize=(8, 3.7))
for ax, (name, r) in zip(axes, res.items()):
    print(name, r)
    viz.plot_audit(r, axes=[ax], show_labels=False, vmax=max(top, 0.05))
    ax.set_title(f"{name}\n{ax.get_title()}", fontsize=10)
viz.save(fig, OUT / "paul15_audit.png", dpi=200)
