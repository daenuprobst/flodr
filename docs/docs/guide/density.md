---
title: Density
sidebar_position: 5
---

# Density

With `density=True`, the default, a fitted model is also a density model of its input.

```python
logp = model.score_samples(X)   # (n,) log density of each row
mean_logp = model.score(X)      # the mean of score_samples
```

## How the density is built

The change of variables formula gives the density of an input from the density of its latent vector and the
Jacobian of the flow. The latent density has two parts. One is a density over positions of the layout. The other
is a Gaussian over the residual whose mean and variance depend on the position, preceded by a few extra coupling
layers that leave the layout coordinates unchanged.

Training includes a likelihood term, so the whole map is shaped as a density model while it learns the layout. The
position-dependent part is fitted after training, at the first call that needs it, which is why the first density
call after `fit` takes longer than the ones after it.

## Units and comparisons

`score_samples` returns the natural logarithm of the density of the whitened input, which differs from the density
of the raw input by a constant for a given training set. Values compare between points of one model, and between
models fitted to the same data. They do not compare between models fitted to different data.

For padded inputs the density includes the padding columns, set to zero for the evaluated points. See
[Choosing the input](inputs.md#narrow-inputs).

## Model selection

`score` is the mean log density, so scikit-learn model selection uses held-out likelihood when no other scorer is
given.

```python
from sklearn.model_selection import GridSearchCV

search = GridSearchCV(FloDR(max_iter=1600), {"w": [1.0, 2.0, 3.0]}, cv=3).fit(X)
search.best_params_
```

Each candidate is fitted once per fold, so a search costs as many fits as candidates times folds.

## Without the density

`FloDR(density=False)` skips the likelihood term and the conditional fit. `score_samples` and `score` are then
absent, and `hasattr(model, "score")` is `False`. `conditional_spread`, `atypicality` and `spread_calibration` raise a
`ValueError`. `hidden_contrast` still works.
