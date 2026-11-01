import time
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from scipy.spatial import cKDTree
from sklearn.datasets import fetch_openml
from sklearn.decomposition import PCA

from flodr import FloDR, viz

OUT = Path(__file__).resolve().parents[1] / "docs" / "docs" / "images"
CLASSES = ["T-shirt", "Trouser", "Pullover", "Dress", "Coat"]
CLASSES += ["Sandal", "Shirt", "Sneaker", "Bag", "Ankle boot"]
ROUTES = [["Sandal", "Sneaker", "Ankle boot"], ["Trouser", "Dress", "Coat", "Bag"]]

# the 60,000 training and 10,000 test images as 50 standardised principal components
pixels, y = fetch_openml(
    "Fashion-MNIST", version=1, return_X_y=True, as_frame=False, parser="liac-arff"
)
y = y.astype(int)
labels = np.array(CLASSES)[y]
train, test = np.arange(60000), np.arange(60000, 70000)
pca = PCA(50, random_state=0).fit(pixels[train])
P = pca.transform(pixels)
mean, sd = P[train].mean(0), P[train].std(0)
X = ((P - mean) / sd).astype(np.float32)

model = FloDR(random_state=0).fit(X[train])
Y = model.embedding_
latent = model.transform_latent(X[train])
tree = cKDTree(Y)
pal = viz.palette(labels)
plt.rcParams.update(viz.RC)


def decode(positions):
    # each position with the residual of the 15 training images nearest it on the map
    nn = tree.query(positions, k=15)[1]
    Z = latent[nn].mean(1)
    Z[:, :2] = positions
    x = pca.inverse_transform(model.inverse_latent(Z) * sd + mean)
    return np.clip(x / 255, 0, 1).reshape(-1, 28, 28), labels[train][nn[:, 0]]


def tile(ax, img, xy, half, colour, alpha=1.0):
    # an image in the colour of its class, ink as opacity
    rgba = np.zeros((28, 28, 4))
    rgba[..., :3], rgba[..., 3] = colour, img * alpha
    x, y = xy
    ax.imshow(rgba, extent=(x - half, x + half, y - half, y + half), zorder=3)


# the map, and images decoded on a grid over it wherever training images are near
fig, (a, b) = plt.subplots(1, 2, figsize=(11, 5.5))
viz.plot_embedding(Y, labels[train], ax=a, colours=pal)
box = dict(boxstyle="round,pad=0.2", fc="white", ec="none", alpha=0.85)
for c in CLASSES:
    xy = np.median(Y[labels[train] == c], 0)
    a.text(*xy, c, ha="center", va="center", fontsize=9, bbox=box)
viz.frame(b, Y)
(x0, x1), (y0, y1), n = b.get_xlim(), b.get_ylim(), 28
step = (x1 - x0) / n
gx, gy = np.meshgrid(x0 + step * (np.arange(n) + 0.5), y0 + step * (np.arange(n) + 0.5))
grid = np.c_[gx.ravel(), gy.ravel()].astype(np.float32)
grid = grid[tree.query(grid)[0] < 0.6 * step]
for img, c, xy in zip(*decode(grid), grid):
    tile(b, img, xy, 0.48 * step, pal[c])
viz.frame(b, Y)
a.set_title("60,000 Fashion-MNIST images")
b.set_title("decoded from the map")
viz.save(fig, OUT / "fashion_map.png", dpi=200)

# new images: one forward pass each, scored by the labels of their map neighbours
model.transform(X[test[:100]])
t = time.perf_counter()
Y_test = model.transform(X[test])
ms = 1e3 * (time.perf_counter() - t)
votes = y[train][tree.query(Y_test, k=15)[1]]
majority = np.array([np.bincount(v).argmax() for v in votes])
agree = np.mean(majority == y[test])
print(f"transform: {len(test)} images in {ms:.0f} ms, {agree:.1%} agree")
fig, ax = plt.subplots(figsize=(5.5, 5.5))
ax.scatter(*Y.T, s=0.3, c="0.88", lw=0, rasterized=True)
viz.plot_embedding(Y_test, labels[test], ax=ax, colours=pal, s=1.5)
viz.frame(ax, Y)
ax.set_title(f"10,000 new images placed in {ms:.0f} ms")
text = f"{agree:.0%} land among images of their own class"
ax.text(0.03, 0.03, text, transform=ax.transAxes, fontsize=8, bbox=box)
viz.save(fig, OUT / "fashion_new_points.png", dpi=200)


def medoid(c):
    i = np.flatnonzero(labels[train] == c)
    return Y[i[np.argmin(((Y[i] - np.median(Y[i], 0)) ** 2).sum(1))]]


def route(names, n=160):
    # straight segments through the class medoids, at equal arc length
    Q = np.array([medoid(c) for c in names])
    s = np.r_[0, np.cumsum(np.linalg.norm(np.diff(Q, axis=0), axis=1))]
    t = np.linspace(0, s[-1], n)
    return np.stack([np.interp(t, s, Q[:, j]) for j in (0, 1)], 1).astype(np.float32)


# paths across the map, decoded at nine steps; faded where no training image is near
paths = [route(r) for r in ROUTES]
gap = np.percentile(tree.query(Y, k=2)[0][:, 1], 99)
far = [tree.query(p)[0] > gap for p in paths]
ends = [((r[0], pal[r[0]]), (r[-1], pal[r[-1]])) for r in ROUTES]
on = np.isin(labels[train], sum(ROUTES, []))
fig = plt.figure(figsize=(12, 5))
gs = fig.add_gridspec(1, 2, width_ratios=(1, 1.45), wspace=0.08)
ax = fig.add_subplot(gs[0])
viz.plot_paths(Y, paths, labels[train], pal, ends, far, on, ax)
strips = gs[1].subgridspec(2, 9, hspace=0.35, wspace=0.06)
for p, (path, names) in enumerate(zip(paths, ROUTES)):
    k = np.linspace(0, len(path) - 1, 9).round().astype(int)
    ax.scatter(*path[k].T, s=16, fc="w", ec="k", lw=0.8, zorder=6)
    for i, (img, c) in enumerate(zip(*decode(path[k]))):
        a = fig.add_subplot(strips[p, i])
        tile(a, img, (0, 0), 0.5, pal[c], 0.35 if far[p][k[i]] else 1.0)
        a.set(xlim=(-0.5, 0.5), ylim=(-0.5, 0.5), xticks=[], yticks=[])
        a.spines[:].set_color("0.85")
        if i == 0:
            a.set_title(", ".join(names), loc="left", fontsize=10)
viz.save(fig, OUT / "fashion_paths.png", dpi=200)

# the density of the test images: the map coloured by log p(x), and the most and the
# least likely images
logp = model.score_samples(X[test])
order = np.argsort(logp)
cert = model.coverage_certificate(X[test])
print(f"log density median {np.median(logp):.1f}, coverage: {cert['certifies']}")
fig = plt.figure(figsize=(11, 5.2))
gs = fig.add_gridspec(1, 2, width_ratios=(1, 1.1), wspace=0.15)
ax = fig.add_subplot(gs[0])
norm = plt.Normalize(*np.percentile(logp, [2, 98]))
viz.plot_embedding(Y_test, values=logp, ax=ax, cmap="magma", norm=norm, order="high")
fig.colorbar(plt.cm.ScalarMappable(norm, "magma"), ax=ax, shrink=0.7, label="log p(x)")
ax.set_title("test images by their log density")
grid = gs[1].subgridspec(4, 6, hspace=0.25, wspace=0.06)
for r, (idx, name) in enumerate(
    ((order[::-1], "most likely"), (order, "least likely"))
):
    for i, j in enumerate(idx[:12]):
        a = fig.add_subplot(grid[2 * r + i // 6, i % 6])
        a.imshow(pixels[test[j]].reshape(28, 28), cmap="gray_r")
        a.set(xticks=[], yticks=[])
        a.spines[:].set_color("0.85")
        if i == 0:
            a.set_title(name, loc="left", fontsize=10)
viz.save(fig, OUT / "fashion_density.png", dpi=200)
