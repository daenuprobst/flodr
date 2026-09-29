---
title: New points
sidebar_position: 3
---

# New points

A fitted model places new points in two ways.

```python
Y_new = model.transform(X_new)
Y_opt = model.transform_opt(X_new, steps=200, lr=0.05)
```

## `transform`

`transform` runs the new points through the trained flow once and keeps the first coordinates. It is fast,
deterministic and part of the bijection, so the full latent of the same points from `transform_latent` decodes
back exactly.

The forward pass places new points well relative to the global arrangement of the layout but finds their nearest
neighbours poorly. On held-out MNIST points, 7% of the 15 nearest neighbours in the input were among the 15 nearest
in the layout, against 35% with `transform_opt`.

## `transform_opt`

`transform_opt` starts from the forward pass and then moves each new point, for `steps` Adam steps at rate `lr`,
towards the layout positions of its 15 nearest training points in the input. The training layout stays fixed. This
is the same kind of placement that UMAP's `transform` does, and it takes about two seconds for 50,000 points on a
GPU.

The positions it returns are no longer outputs of the flow. `inverse_transform` decodes them like any other
position, but they have no residual for `inverse_latent`.

## Which to use

Use `transform_opt` when the neighbours of new points matter, for example to label them by their neighbours.
Use `transform` when you need the map itself, for example to compare new points with the training data in the
latent space or to decode them exactly.
