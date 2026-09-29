---
title: Quickstart
sidebar_position: 3
---

# Quickstart

This fits FloDR to a matrix with one row per point, then uses each part of the fitted model once.

```python
import numpy as np
from flodr import FloDR, viz

rng = np.random.default_rng(0)
X = rng.normal(size=(5000, 50)).astype("float32")
labels = (X[:, 0] > 0).astype(int)

model = FloDR(random_state=0).fit(X)
Y = model.embedding_                        # (5000, 2) layout

Y_new = model.transform(X[:100])            # one pass through the flow
Y_opt = model.transform_opt(X[:100])        # placement refined against the training layout

Z = model.transform_latent(X[:100])         # layout followed by the residual
X_back = model.inverse_latent(Z)            # equal to X[:100] up to float precision
X_rep = model.inverse_transform(Y[:100])    # one representative input per position

logp = model.score_samples(X[:100])         # log density of each point

sigma = model.conditional_spread()          # spread of the inputs behind each position
cert = model.spread_calibration()           # held-out test of that field
field, hidden = model.hidden_contrast(labels)

ax = viz.plot_embedding(Y, labels)
viz.save(ax.figure, "embedding.png")
```

A fit of 5,000 points takes about two minutes on a consumer GPU. `hidden_contrast` trains a classifier for
every permutation of its null and is the slowest call here.

## Next

The layout depends on the input more than on any parameter, so read [Choosing the input](guide/inputs.md) before
fitting your own data. [Fitting](guide/fitting.md) explains the parameters, and
[Diagnostics](guide/diagnostics.md) explains how to read the fields and their tests.
