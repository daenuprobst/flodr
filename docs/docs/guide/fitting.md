---
title: Fitting
sidebar_position: 1
---

# Fitting

```python
from flodr import FloDR

model = FloDR(n_components=2, w=2.0, density=True, random_state=0).fit(X)
```

`X` is a two-dimensional array or DataFrame of numbers, one row per point, at least two rows and two columns.

## What `fit` does

1. It whitens the input with PCA. The leading components, up to 50, are scaled to unit variance and the rest of
   the input is kept alongside them, so this first step is itself an exact bijection.
2. Inputs with fewer than 30 columns are padded with noise columns up to 30. See
   [Choosing the input](inputs.md#narrow-inputs).
3. It builds a fuzzy graph of the 15 nearest neighbours of every point, on the input as given. Binary input uses
   Jaccard distances.
4. It trains the flow. The first coordinates are pulled towards their graph neighbours and pushed away from
   random other points, as in UMAP. An ordinal term compares the order of distances between pairs of points in
   the input with their order in the layout. With `density=True` a likelihood term trains the whole map as a
   density model at the same time.

The flow starts at the identity, so training starts from the first two principal components.

## Parameters

### `n_components`

The dimension of the embedding, 2 by default. It must be at least 2 and smaller than the latent dimension, which
equals the number of input columns, or 30 for padded inputs.

### `w`

The weight of the ordinal term, 2.0 by default. It sets the balance between local and global structure. At 0 the
layout keeps neighbourhoods like a neighbour embedding and arranges groups freely. Larger values keep more of the
input's distance order between groups at a small cost in local neighbourhoods. The default was chosen on
benchmark and single-cell data, and values from 0 to 3 have been tested.

### `density`

Whether to train and fit the density, `True` by default. `score_samples`, `score`, `conditional_spread`,
`atypicality` and `spread_calibration` need it. `hidden_contrast` does not. `density=False` fits faster.

### `max_iter`

The number of training steps. The default `None` runs 6,400 steps at a learning rate of 0.0025. A different
number keeps the product of the two at 16, so `max_iter=1600` runs at 0.01. Below 800 steps the rate stays at
0.02, since larger steps damage the inverse. Fewer steps give a rougher layout sooner.

### `random_state`

The seed, 0 by default. An integer, a `numpy.random.Generator`, a `RandomState` or `None` for a fresh seed. On the
CPU a seed reproduces a fit exactly. On CUDA the default training path is not bitwise reproducible, see
[Performance](performance.md#reproducibility).

### `device`

`"cuda"`, `"cpu"` or any PyTorch device string. `None` picks the GPU when PyTorch sees one.

### `verbose`

Whether to show a progress bar, `True` by default. The bar only draws when the output is a terminal.

### `advanced`

A dictionary of overrides for the training configuration, `None` by default. Keys must be fields of
[`TrainConfig`](../api/training.md#trainconfig), and unknown keys raise a `ValueError`. The table lists the ones
most often useful.

| Key | Default | Effect |
|---|---|---|
| `pad` | 30 | width that narrower inputs are padded to, 0 turns padding off |
| `fine_eps` | 0.25 | weight of the Fourier branch in the couplings that write the layout, 0 turns it off |
| `edge_batch` | -1 | graph edges per step, -1 for the whole graph up to 262,144 edges and a random batch of that many above |
| `neg` | 48 | negative samples per edge |
| `n_layers` | 4 | coupling layers |
| `hid` | 256 | width of the coupling networks |
| `compile_edges` | `True` | compile the training step on CUDA |
| `half_edges` | `"fp16"` | half precision for the hidden layers of the training step on CUDA, `""` for full precision |

## Attributes after fitting

`embedding_` holds the layout of the training data. `n_iter_` is the number of steps run, `roundtrip_` the largest
absolute error of the latent round trip on the training data, and `knn_idx_` and `knn_dist_` the neighbour graph.
The full list is in the [API reference](../api/flodr.md#attributes).
