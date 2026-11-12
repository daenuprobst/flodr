---
title: The inverse
---

```python
Z = model.transform_latent(X)       # (n, d): the map, then the residual
X_same = model.inverse_latent(Z)    # exact up to float precision
X_rep = model.inverse_transform(Y)  # positions only, residual set to zero
X_dec = model.decode(Y)             # positions only, from the fitted points around them
```

Every layer of the flow is an affine coupling, so the map is a bijection and `inverse_latent` undoes
`transform_latent` exactly. `inverse_transform` decodes a position with the residual set to zero, one input per
position; the zero of the residual has no meaning of its own.

`decode` takes the mean of the 15 fitted points nearest a position on the map and moves it onto the fibre of the
position: to a nearby input that the map places exactly there. On held-out points it is as accurate as that mean
and as a decoder trained on the layout, and it needs neither.

By hand, with the residual of the same points instead, which is less accurate on large data sets:

```python
from scipy.spatial import cKDTree

Z_train = model.transform_latent(X_train)
nn = cKDTree(Z_train[:, :2]).query(Y_query, k=15)[1]
Z = Z_train[nn].mean(1)
Z[:, :2] = Y_query
X_decoded = model.inverse_latent(Z)
```

![Two walks across the Fashion-MNIST map, decoded at nine steps](../images/fashion_paths.png)

Two walks across the [Fashion-MNIST](../examples/fashion-mnist.md) map, each the shortest path through the graph of
15 nearest neighbours on the map between the densest points of two classes, decoded at nine steps with the borrowed
residual. Where no training image is near (dashed on the map) the decoded images are faded: there the map has
nothing to decode from.

Many inputs share a position. With the density, `flodr.conformal.sample_fibre(model, Y, n_draws=64)` draws them
exactly from p(x | y), and `flodr.conformal.conformal_bands(model, X_calib, Y)` gives split-conformal bands of the
inputs at each position, calibrated on held-out points (see [Density](density.md)).
