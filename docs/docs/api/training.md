---
title: Training internals
sidebar_position: 3
---

# Training internals

The estimator covers normal use. These parts are exported for custom training and may change between versions.

## `TrainConfig`

A dataclass with every setting of a training run. `FloDR(advanced=...)` overrides its fields by name. The fields most
often changed are listed in [Fitting](../guide/fitting.md#advanced).

## `raw_recipe(seed=0, threads=None, **overrides)`

Returns the `TrainConfig` the estimator starts from. Overriding `iters` without `lr` rescales the learning rate so
that their product stays at 16, with the rate capped at 0.02.

## `train_flodr(Xp, ei, ej, cfg, rng, return_model=False, w=None, stress_X=None, ...)`

Trains a flow on whitened coordinates `Xp` and graph edges `ei`, `ej`. Returns the layout, the round-trip error and
reconstruction errors, and the flow when `return_model=True`.

## `Flow`

The normalising flow, a stack of affine couplings. `flow(x)` maps whitened inputs to latent coordinates and
`flow.inverse(z)` maps them back. `flow.log_prob(x)` gives the log density once the density is fitted.

## `Coupling`

One affine coupling layer of the flow.

## Data helpers

| Function | Description |
|---|---|
| `raw_coords(X, n_components=50, seed=0)` | the PCA whitening, with its forward and inverse maps |
| `fuzzy_knn_graph(Xp, k, ...)` | the fuzzy neighbour graph used for training |
| `knn_search(Xp, k, exact=None)` | nearest neighbours of every point |
| `global_pairs(Xp, count, rng)` | random pairs for the global terms |
| `find_ab(min_dist=0.1, spread=1.0)` | the parameters of the layout kernel |
| `preprocess(X, n_components=50, seed=0, ...)` | PCA to at most `n_components` components scaled to unit variance, optionally with its inverse and transform |

## Device helpers

| Function | Description |
|---|---|
| `default_device()` | `"cuda"` when PyTorch sees a GPU, otherwise `"cpu"` |
| `native_bf16(device=None)` | whether the device computes in bfloat16 natively |
| `perf_cores()` | the number of performance cores, which FloDR uses on the CPU |
