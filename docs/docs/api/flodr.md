---
title: FloDR
---

```python source
flodr.FloDR
```

Parameters are described in [Fitting](../guide/fitting.md#parameters).

## Methods

```python source
flodr.FloDR.fit
flodr.FloDR.fit_transform
flodr.FloDR.transform
flodr.FloDR.transform_opt
flodr.FloDR.transform_latent
flodr.FloDR.inverse_latent
flodr.FloDR.inverse_transform
flodr.FloDR.score_samples
flodr.FloDR.score
flodr.FloDR.coverage_certificate
flodr.FloDR.energy_certificate
flodr.FloDR.faithful_scale
```

| Method | Returns |
|---|---|
| `fit` | the fitted model |
| `fit_transform` | `embedding_` |
| `transform` | `(m, n_components)` positions from one pass through the flow |
| `transform_opt` | `(m, n_components)` positions optimised against the training map |
| `transform_latent` | `(m, D)` latent coordinates, the map first, then the residual |
| `inverse_latent` | `(m, d)` inputs, the exact inverse of `transform_latent` |
| `inverse_transform` | `(m, d)` inputs decoded with the residual set to zero |
| `score_samples` | `(m,)` log density |
| `score` | mean log density |
| `coverage_certificate` | dict with `certifies`, `passed`, `fibre` and `map` (`ranks`, `ks`, `band`), `flagged` regions |
| `energy_certificate` | dict with `certifies`, `passed`, `energy` (`ranks`), `precision` (`wild`) |
| `faithful_scale` | dict with `rho` `(m, len(h))`, `hstar` `(m,)`, `faithful` |

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

## `flodr.audit`

```python source
flodr.audit
flodr.plot_audit
```

`data` is a 2-D map `(n, 2)` with `X` and `labels`, or an AnnData object read through `basis`, `use_rep` and `key`.
Returns an `AuditResult` with `level` (nats), `p_global`, `certified`, the accuracies `acc_map`, `acc_input`,
`bacc_map`, `bacc_input`, and per point `field` and `lower`. `plot_audit` draws it. See
[Hidden information](../guide/audit.md).

## `flodr.conformal`

```python source
flodr.conformal.sample_fibre
flodr.conformal.conformal_bands
flodr.conformal.FibreConformal
flodr.conformal.FibreConformal.fit
flodr.conformal.FibreConformal.contains
flodr.conformal.FibreConformal.bands
```

| Function | Returns |
|---|---|
| `sample_fibre` | exact draws from p(x &#124; y): inputs `(n_draws, m, d)`, their log densities, their positions |
| `conformal_bands` | dict with `lo`, `hi` per position, calibrated on held-out `calib_X` |
| `FibreConformal` | `.contains(X)` per point, `.bands(Y, ...)` as above |
