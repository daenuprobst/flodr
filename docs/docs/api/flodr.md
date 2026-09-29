---
title: FloDR
sidebar_position: 1
---

# `flodr.FloDR`

```python
FloDR(n_components=2, *, w=2.0, density=True, max_iter=None, random_state=0, device=None, verbose=True,
      advanced=None)
```

A dimensionality reduction estimator built on an invertible normalising flow. The guide explains the parameters in
[Fitting](../guide/fitting.md).

## Parameters

| Parameter | Default | Description |
|---|---|---|
| `n_components` | `2` | dimension of the embedding, at least 2 and below the latent dimension |
| `w` | `2.0` | weight of the ordinal term, 0 turns it off |
| `density` | `True` | train and fit the density, needed by `score_samples`, `score` and the spread diagnostics |
| `max_iter` | `None` | training steps, `None` for 6,400 |
| `random_state` | `0` | integer, `Generator`, `RandomState` or `None` |
| `device` | `None` | PyTorch device, `None` for the GPU when there is one |
| `verbose` | `True` | show a progress bar while training |
| `advanced` | `None` | dictionary of [`TrainConfig`](training.md#trainconfig) overrides |

## Attributes

| Attribute | Description |
|---|---|
| `embedding_` | `(n, n_components)` layout of the training data |
| `n_iter_` | training steps run |
| `n_features_in_` | number of input columns |
| `feature_names_in_` | input column names, when fitted on a DataFrame |
| `roundtrip_` | largest absolute error of the latent round trip on the training data |
| `flow_` | the trained [`Flow`](training.md#flow) |
| `knn_idx_` | `(n, 15)` indices of each point's nearest neighbours, the point itself excluded |
| `knn_dist_` | `(n, 15)` distances to those neighbours |

## Fitting and transforming

### `fit(X, y=None)`

Fits the model to `X` of shape `(n, d)` and returns it. `y` is ignored.

### `fit_transform(X, y=None)`

Fits the model and returns `embedding_`.

### `transform(X)`

Returns the layout coordinates of `X`, shape `(m, n_components)`, from one pass through the flow.

### `transform_opt(X, steps=200, lr=0.05)`

Returns layout coordinates of `X` refined against the training layout, shape `(m, n_components)`. Each point
starts from `transform` and moves for `steps` Adam steps at rate `lr` towards the positions of its 15 nearest
training points. See [New points](../guide/new-points.md).

### `transform_latent(X)`

Returns the full latent coordinates of `X`, shape `(m, D)`, the layout first and then the residual. `D` is the
number of input columns, or 30 for padded inputs.

### `inverse_latent(Z)`

Decodes latent coordinates of shape `(m, D)` to inputs of shape `(m, d)`. The inverse of `transform_latent` up to
float precision. Accepts a NumPy array or a PyTorch tensor.

### `inverse_transform(Y)`

Decodes layout positions of shape `(m, n_components)` to inputs of shape `(m, d)`, with the residual set to zero.

## Density

### `score_samples(X)`

Returns the log density of each row of `X`, shape `(m,)`. Needs `density=True`. See [Density](../guide/density.md).

### `score(X, y=None)`

Returns the mean of `score_samples(X)`. Needs `density=True`.

## Diagnostics

### `conditional_spread(Y=None, n_samples=64)`

Returns the spread of the inputs the model places at each position, shape `(m,)`, in input units. `Y` defaults to
`embedding_`. Needs `density=True`.

### `atypicality(n_samples=128)`

Returns, for each training point, the fraction of decoded samples at its position that lie closer to their mean
than the point itself, shape `(n,)`. Needs `density=True`.

### `spread_calibration(n_bins=14, min_pts=12, n_samples=64)`

Tests `conditional_spread` on held-out points and returns a dictionary. Needs `density=True`.

| Key | Description |
|---|---|
| `passed` | whether the field passed |
| `certifies` | `"magnitude+shape"` or `"magnitude-only"` |
| `confidence` | smallest bootstrap pass rate among the tested parts |
| `confidence_parts` | pass rates of the level, and of slope and R squared when tested |
| `slope`, `r2` | log-log regression of measured on predicted variance per grid cell |
| `ratio_quartiles` | quartiles of the ratio of predicted to measured variance |
| `dynamic_range` | ratio of the 95th to the 5th percentile of the measured variance |
| `n_bins` | grid cells used |

### `hidden_contrast(G, n_bins=14, min_pts=16, k_local=10, hid=128, iters=3000, n_perm=99, alpha=0.01)`

Returns `(field, cert)`. `field`, shape `(n,)`, is the information about the labels `G` that the input holds and the
layout does not, at each training point, in nats. `hid` and `iters` set the width and training steps of the
classifiers, `n_perm` the number of shuffled replicas and `alpha` the level of the test.

| Key | Description |
|---|---|
| `passed` | whether the field passed, `None` when undecided |
| `certifies` | `"magnitude+shape"`, or `"undecided"` with too few grid cells |
| `mean_gap` | mean held-out gap in nats, after subtracting the null |
| `p_level`, `null_level` | permutation p value and the replicas' gaps |
| `slope`, `r2`, `shape_ok` | regression of the measured on the predicted gap per grid cell |
| `level` | ratio of the mean measured gap to the mean predicted gap |
| `confidence`, `confidence_parts` | bootstrap pass rates of slope, R squared and level |
| `field_null` | a shuffled replica's field, for drawing beside the real one |
| `n_bins`, `n_perm`, `dynamic_range` | grid cells used, replicas, spread of the measured gap |

### `diagnostics(G=None, **kw)`

Returns a dictionary with `"spread"` when the density is on and `"hidden_contrast"` when `G` is given. Each entry
holds `field`, `cert`, `units` and `label`. Keyword arguments go to `hidden_contrast`.

## scikit-learn

`get_params`, `set_params`, `get_feature_names_out` (`flodr0`, `flodr1`, and so on) and
`set_output(transform="pandas")` behave as in any scikit-learn transformer. See
[scikit-learn](../guide/scikit-learn.md).
