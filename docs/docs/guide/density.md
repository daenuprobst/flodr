---
title: Density
sidebar_position: 4
---

# Density

```python
logp = model.score_samples(X)               # log p(x), per point
model.score(X)                              # the mean, for GridSearchCV
```

The density is fitted after the layout, under a small input noise. Its values are worth reading only where it is a
law of the data, so test it on points the model has not seen.

## Certificates

```python
cert = model.coverage_certificate(X_test)
cert["certifies"]                           # "certified", "refused" or "undecided"
energy = model.energy_certificate(X_test)
scale = model.faithful_scale(X_test)
```

`coverage_certificate` ranks each held-out point among exact draws of the density at its own position (`fibre`) and
among draws of positions (`map`). Both ranks are uniform when the density is right; the verdict is an equivalence test
at `margin=0.1`. Without `X` it refits the density on half the training points and tests it on the other half.
`cert["flagged"]` lists the map regions where coverage falls short. `viz.plot_ranks` draws the ranks.

`energy_certificate` ranks held-out points by their input distance to the draws at their position, which also
catches draws that are too wide. `faithful_scale` reports, per point, how far the decode stays near-linear at the
density's input noise (`rho`, `hstar`).

## Bands along the map

```python
from flodr.conformal import conformal_bands

bands = conformal_bands(model, X_calib, Y_path, readout=W)  # lo, hi per position
```

The bands hold the inputs at each position at level `1 - alpha` (0.9 by default), calibrated split-conformally on
held-out points `X_calib`. `readout` maps inputs to what you want to read, a matrix (e.g. PCA loadings back to genes)
or a function. `viz.plot_band` draws them along a path.
