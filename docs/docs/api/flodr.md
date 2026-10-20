---
title: FloDR
sidebar_position: 1
---

# `flodr.FloDR`

```python
FloDR(n_components=2, *, w=2.0, density=True, max_iter=None, random_state=0, device=None, verbose=True,
      advanced=None, axes="variance")
```

Parameters are described in [Fitting](../guide/fitting.md#parameters).

## Methods

| Method | Returns |
|---|---|
| `fit(X, y=None)` | the fitted model |
| `fit_transform(X, y=None)` | `embedding_` |
| `transform(X)` | `(m, n_components)` positions from one pass through the flow |
| `transform_opt(X, steps=200, lr=0.05)` | `(m, n_components)` positions optimised against the training map |
| `transform_latent(X)` | `(m, D)` latent coordinates, the map first, then the residual |
| `inverse_latent(Z)` | `(m, d)` inputs, the exact inverse of `transform_latent` |
| `inverse_transform(Y)` | `(m, d)` inputs decoded with the residual set to zero |
| `score_samples(X)` | `(m,)` log density |
| `score(X, y=None)` | mean log density |
| `coverage_certificate(X=None, n_draws=199, margin=0.1, alpha=0.05)` | dict with `certifies`, `passed`, `fibre` and `map` (`ranks`, `ks`, `band`), `flagged` regions |
| `energy_certificate(X, n_draws=199, margin=0.1, alpha=0.05)` | dict with `certifies`, `passed`, `energy` (`ranks`), `precision` (`wild`) |
| `faithful_scale(X=None, h=(0.125, 0.25, 0.5, 1.0, 2.0))` | dict with `rho` `(m, len(h))`, `hstar` `(m,)`, `faithful` |

`D` is the number of input columns, or 30 for padded inputs. `score_samples`, `score` and the certificates need
`density=True`.

## Attributes

| Attribute | |
|---|---|
| `embedding_` | `(n, n_components)` map of the training points |
| `n_iter_` | training steps run |
| `n_features_in_`, `feature_names_in_` | input width, and column names when fitted on a DataFrame |
| `roundtrip_` | largest absolute error of the latent round trip on the training points |
| `knn_idx_`, `knn_dist_` | `(n, 15)` neighbour graph |
| `knn_backend_` | the neighbour search used |
| `step_mode_` | how the training step ran: `"graph"` (one CUDA graph) or `"eager"` |
| `density_sigma_` | sd of the density's input noise |
| `flow_` | the trained [`Flow`](training.md#flow) |

# `flodr.audit`

```python
audit(data, basis="X_umap", use_rep="X_pca", key=None, *, X=None, labels=None, test=True, seed=0, n_threads=4,
      inplace=True, verbose=True)
```

`data` is a 2-D map `(n, 2)` with `X` and `labels`, or an AnnData object read through `basis`, `use_rep` and `key`.
Returns an `AuditResult` with `level` (nats), `p_global`, `certified`, the accuracies `acc_map`, `acc_input`,
`bacc_map`, `bacc_input`, and per point `field` and `lower`. `plot_audit(res, axes=None)` draws it. See
[Hidden information](../guide/audit.md).

# `flodr.conformal`

| Function | Returns |
|---|---|
| `sample_fibre(est, Y, n_draws=64, seed=0)` | exact draws from p(x &#124; y): inputs `(m, n_draws, d)`, their log densities, their positions |
| `conformal_bands(est, calib_X, Y, alpha=0.1, n_draws=256, q=(5, 95), readout=None)` | dict with `lo`, `hi` per position, calibrated on held-out `calib_X` |
| `FibreConformal(est, alpha=0.1).fit(calib_X)` | `.contains(X)` per point, `.bands(Y, ...)` as above |
