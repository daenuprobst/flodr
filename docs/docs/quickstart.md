---
title: Quickstart
---

```python
import matplotlib.pyplot as plt
import numpy as np
from flodr import FloDR, audit, viz

# X: one row per point, e.g. 50 standardised principal components; labels: one per row
train, test = np.arange(len(X)) % 5 != 0, np.arange(len(X)) % 5 == 0

model = FloDR(random_state=0).fit(X[train])
Y = model.embedding_                              # the map of the training points

Y_new = model.transform(X[test])                  # one pass through the flow
Y_opt = model.transform_opt(X[test])              # optimised against the training map
X_back = model.inverse_transform(Y[:10])          # positions back to the input
Z = model.transform_latent(X[test])               # the map, then the residual
X_same = model.inverse_latent(Z)                  # exact up to float precision

logp = model.score_samples(X[test])               # log density
cert = model.coverage_certificate(X[test])        # is the density a law of new points?
print(cert["certifies"])                          # "certified", "refused" or "undecided"

res = audit(Y, X=X[train], labels=labels[train])  # what the map hides about the labels
print(res)

fig, ax = plt.subplots(1, 2, figsize=(8, 4))
viz.plot_embedding(Y, labels[train], ax=ax[0])
viz.plot_ranks([cert["fibre"]["ranks"], cert["map"]["ranks"]], ax=ax[1])
viz.save(fig, "flodr.pdf")
```

Fitting uses the GPU when PyTorch sees one. `examples/quickstart.ipynb` runs these calls on Fashion-MNIST.
