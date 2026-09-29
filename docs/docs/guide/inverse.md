---
title: Latent space and inverse
sidebar_position: 4
---

# Latent space and inverse

The flow maps every input to a latent vector with one coordinate per input column. The first `n_components`
coordinates are the layout. The rest are the residual, which holds what the layout does not.

```python
Z = model.transform_latent(X)       # (n, D), layout first, then the residual
X_back = model.inverse_latent(Z)    # (n, d), the input again
```

## Exact round trip

`inverse_latent` undoes `transform_latent` up to float precision. `roundtrip_` reports the largest absolute error
of that round trip on the training data after fitting, typically between 1e-6 and 1e-4 in whitened units.

`inverse_latent` accepts any latent vector, not only those of real points. For inputs without padding, editing the
residual of a point while keeping its layout coordinates gives another input that the model places at the same
position.

## Decoding a position

```python
X_rep = model.inverse_transform(Y)  # Y has n_components columns
```

`inverse_transform` decodes positions of the layout, with the residual set to zero. The result is one
representative input for each position. It is not the mean of the inputs placed there, and many different inputs
map to the same position. [`conditional_spread`](diagnostics.md#conditional-spread) measures how different they
are.

## Why the inverse is exact

Every layer of the flow is an affine coupling. It changes some coordinates by a scale and a shift computed from the
others, which pass through unchanged, so each layer can be undone exactly, and so can the whole flow. The PCA whitening in front of it is an invertible linear map.
