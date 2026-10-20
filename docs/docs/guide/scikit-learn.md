---
title: scikit-learn
sidebar_position: 7
---

# scikit-learn

`FloDR` is a scikit-learn transformer: it clones, sits in a `Pipeline` and returns DataFrames on request.

```python
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

pipe = make_pipeline(StandardScaler(), FloDR(random_state=0))
Y = pipe.fit_transform(X)
```

- `set_output(transform="pandas")` returns DataFrames with columns `flodr0`, `flodr1`, ...
- `score` is the mean log density, so `GridSearchCV` selects on held-out likelihood when no scorer is given.
- A fitted model pickles (`pickle`, `joblib`) with its tensors on the CPU; a model fitted on a GPU moves back to
  it when loaded on a machine with one.
