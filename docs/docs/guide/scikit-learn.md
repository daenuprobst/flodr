---
title: scikit-learn
sidebar_position: 7
---

# scikit-learn

`FloDR` is a scikit-learn transformer. It clones, takes part in pipelines and model selection, and returns
DataFrames on request.

## Pipelines

```python
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

pipe = make_pipeline(StandardScaler(), FloDR(random_state=0)).fit(X)
Y_new = pipe.transform(X_new)
```

Only scale the input when that is the geometry you want. See [Choosing the input](inputs.md).

## DataFrames

```python
model = FloDR(random_state=0).set_output(transform="pandas").fit(df)
model.transform(df)                 # DataFrame with columns flodr0, flodr1
model.feature_names_in_             # the column names of df
model.get_feature_names_out()       # ["flodr0", "flodr1"]
```

## Model selection

`score` returns the mean log density, so `GridSearchCV` and `cross_val_score` select on held-out likelihood when
no scorer is given. See [Density](density.md#model-selection).

## Pickling

A fitted model pickles with its tensors on the CPU. A model fitted on a GPU therefore loads on a machine without
one and moves back to the GPU where one is available.

```python
import pickle

pickle.dump(model, open("model.pkl", "wb"))
model = pickle.load(open("model.pkl", "rb"))
```

## Estimator checks

`FloDR` passes scikit-learn's `check_estimator` apart from two groups of checks. Several checks force
`n_components=1`, and a one-dimensional layout leaves the flow no residual to route through. Two checks expect
`transform` on the training data to reproduce `fit_transform`, which does not hold for padded inputs. The test
suite lists both groups as expected failures.

## Deprecated arguments

`fit(X, iters=...)`, `fit(X, progress=...)` and `inverse_coords` still work and warn. Use `max_iter`, `verbose` and
`inverse_latent` instead.
