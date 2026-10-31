---
title: Training internals
---

The estimator covers normal use. These parts are exported for custom training and may change between versions.

## `TrainConfig`

A dataclass with every setting of a training run. `FloDR(advanced=...)` overrides its fields by name.

```python source
flodr.TrainConfig
```

## `raw_recipe`

```python source
flodr.raw_recipe
```

Returns the `TrainConfig` the estimator starts from. Overriding `iters` without `lr` rescales the learning rate so
that their product stays at 16, with the rate capped at 0.02.

## `train_flodr`

```python source
flodr.train_flodr
```

Trains a flow on whitened coordinates `Xp` and graph edges `ei`, `ej`. Returns the layout, the round-trip error and
reconstruction errors, and the flow when `return_model=True`.

## `Flow`

```python source
flodr.Flow
```

The normalising flow, a stack of affine couplings. `flow(x)` maps whitened inputs to latent coordinates and
`flow.inverse(z)` maps them back. `flow.log_prob(x)` gives the log density once the density is fitted.

## `Coupling`

```python source
flodr.Coupling
```

One affine coupling layer of the flow.

## Data helpers

```python source
flodr.raw_coords
flodr.fuzzy_knn_graph
flodr.knn_search
flodr.global_pairs
flodr.find_ab
flodr.preprocess
```

| Function | Description |
|---|---|
| `raw_coords` | the PCA whitening, with its forward and inverse maps |
| `fuzzy_knn_graph` | the fuzzy neighbour graph used for training |
| `knn_search` | nearest neighbours of every point, exact on the GPU |
| `global_pairs` | random pairs for the global terms |
| `find_ab` | the parameters of the layout kernel |
| `preprocess` | PCA to at most `n_components` components scaled to unit variance, optionally with its inverse and transform |

## Device helpers

```python source
flodr.default_device
flodr.native_bf16
flodr.perf_cores
```

| Function | Description |
|---|---|
| `default_device` | `"cuda"` when PyTorch sees a GPU, otherwise `"cpu"` |
| `native_bf16` | whether the device computes in bfloat16 natively |
| `perf_cores` | the number of performance cores, which FloDR uses on the CPU |
