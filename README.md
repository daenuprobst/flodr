# FloDR

Dimensionality reduction built on an invertible normalising flow. The first
`n_components` output coordinates of the flow are the embedding and the rest are kept as
a residual, so the trained map is a bijection with an exact inverse. The diagnostics use
that inverse to measure what the layout leaves undetermined at each position. The API
follows the scikit-learn estimator interface.

![](assets/flow_morph_bmarrow_white.gif)

## Installation

Python 3.11 or newer.

```bash
uv pip install -e .
```

## Usage

```python
import numpy as np
from flodr import FloDR, viz

X = np.random.default_rng(0).normal(size=(2000, 50)).astype("float32")
labels = (X[:, 0] > 0).astype(int)

model = FloDR(w=2.0, random_state=0, density=True).fit(X)
Y = model.embedding_                            # (n, 2)

Y_new = model.transform(X[:10])                 # one forward pass through the flow
Y_opt = model.transform_opt(X[:10])             # per-point placement, UMAP-style
X_back = model.inverse_transform(Y[:10])        # embedding back to input space
logp = model.score_samples(X[:10])              # exact log density

sigma = model.conditional_spread()              # what the layout leaves undetermined
field, cert = model.hidden_contrast(labels)     # label structure the layout hides

ax = viz.plot_embedding(Y, labels)
viz.save(ax.figure, "embedding.pdf")
```

Training runs on the GPU whenever CUDA is available; pass `device="cpu"` to force the
CPU. `hidden_contrast` fits a classifier per permutation replica and is by far the
slowest call above.

## Parameters

| Parameter | Default | Description |
|---|---|---|
| `w` | `2.0` | Weight of the ordinal term, the local-to-global dial. Higher keeps more global distance. |
| `random_state` | `0` | Seed for the fit. |
| `device` | GPU if there is one | Resolved by `flodr.default_device()`. Pass `"cpu"` to force it. |
| `density` | `False` | Fit the residual density. Required by `score_samples` and the diagnostics. |
| `n_components` | `2` | Embedding dimension. Must be at least 2 and below the latent dimension. |
| `advanced` | `None` | Dict of `TrainConfig` overrides, e.g. `dict(gate_max=0.5, edge_batch=32768)`. |

## Methods

| Method | Returns | Description |
|---|---|---|
| `fit(X, iters=800, progress=True)` / `fit_transform(X, ...)` | self / `(n, k)` | Fit the flow; the layout also lands in `embedding_`. `iters` is the training budget, `progress` draws a tqdm bar. |
| `transform(X)` | `(m, k)` | Embed new points in one forward pass. |
| `transform_opt(X, steps=200, lr=0.05)` | `(m, k)` | Placement transform: per-point optimisation against the frozen train embedding, in the same class as UMAP's `transform`. |
| `inverse_transform(Y)` | `(m, d)` | Decode screen positions, residual set to zero. |
| `inverse_coords(Z)` | `(m, d)` | Decode full latent coordinates. |
| `score_samples(X)` | `(m,)` | Log density under the flow. |
| `conditional_spread(Y=None, n_samples=64)` | `(m,)` | Spread of the fiber over each position, in input units. |
| `atypicality(n_samples=128)` | `(n,)` | Per-point tail probability under its own fiber, in `[0, 1]`. |
| `spread_calibration(n_bins=14, min_pts=12, n_samples=64)` | dict | Certificate for `conditional_spread`, with `passed` and `confidence`. |
| `hidden_contrast(G, ...)` | `(field, cert)` | Label information present in the input but not in the layout, in nats, against a within-bin permutation null (`n_perm=99` by default). |
| `diagnostics(G=None, **kw)` | dict | Both fields with their certificates, units and labels. |

## Attributes

| Attribute | Description |
|---|---|
| `embedding_` | `(n, k)` layout of the training data. |
| `roundtrip_` | Max absolute error of the latent round trip. |
| `flow_` | The trained `Flow`. |
| `knn_idx_`, `knn_dist_` | The 15-NN graph the fit was built on. |

## `flodr.viz`

| Function | Description |
|---|---|
| `plot_embedding(Y, labels)` | Scatter of the layout. |
| `plot_field(Y, v)` | One scalar field over the layout. |
| `bivariate_field(Y, x, y)` | Two fields in one panel, with a corner key. |
| `occupancy_field(Y)` | Grid axes and a soft mask of where the layout has data. |
| `depth_fog(Y)` | Draw order, colour weight and depth, for 3D scatters. |
| `residual(flow, X)`, `residual_fraction(flow, X)` | The residual coordinates and their share of the latent norm. |
| `save(fig, path)` | Save at 300 dpi with tight bounds. |
| `RC`, `RC_PAPER` | Matplotlib rc dicts for screen and for print. |

### Training budget

`iters` defaults to 800, the value every number in the paper was produced with. It counts
optimiser steps, and each step draws a fixed `edge_batch` of kNN edges, so the number of
times the average edge is trained on falls as the dataset grows: roughly 490 passes at the
paper's `n=4000`, 22 at a 90k-cell atlas, under 2 at a million cells. Raising it is cheap
at scale, since the kNN graph build dominates the wall clock there rather than the
optimisation. On the 1.22M-cell zebrafish atlas, doubling to 1600 moved CPD from 0.632 to
0.645 and left recall@15 within noise, so a larger budget buys global structure rather than
local neighbourhoods.

`progress=False` turns the tqdm bar off outright. Left on, it draws only when stderr is a
terminal, so piping to a log does not fill it with carriage returns.

Low-level pieces are exported for custom training: `Flow`, `Coupling`, `TrainConfig`,
`train_flodr`, `raw_recipe`, plus the data helpers `preprocess`, `raw_coords`,
`fuzzy_knn_graph`, `knn_search`, `global_pairs`, `find_ab`.

## Example notebook

`examples/quickstart.ipynb` runs every call above on Fashion-MNIST, 40000 images reduced to
a standardised PCA-50, and takes about two minutes on a GPU. It fits on four fifths of the
images, embeds the rest with both transforms, sweeps `inverse_transform` over a grid of
screen positions to decode the layout back into pictures, and ends on the two diagnostics
with their certificates, both of which pass at that sample size.

## Reproducing the paper

The `scripts` directory reconstructs every figure and number in the paper. See
`scripts/README.md`.
