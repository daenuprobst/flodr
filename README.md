# FloDR

FloDR is a dimensionality reduction method built on an invertible normalising flow. It reads the first two output coordinates of the flow as a two-dimensional embedding and keeps the remaining coordinates as a residual, so the trained map is a bijection with an exact inverse. The diagnostics built on that inverse measure what the layout leaves undetermined at each position. The API follows the scikit-learn estimator interface.

## Installation

FloDR requires Python 3.11 or newer.

```bash
uv pip install -e .
```

## Usage

```python
import numpy as np
from flodr import FloDR

X = np.random.default_rng(0).normal(size=(2000, 50)).astype("float32")

model = FloDR(w=2.0, random_state=0, density=True)
Y = model.fit_transform(X)                 # (n, 2)
Y_new = model.transform(X[:10])            # new points in one forward pass
X_back = model.inverse_transform(Y[:10])   # decode screen positions to input space

sigma = model.conditional_spread()         # per-point hidden spread, input units
report = model.diagnostics(G=labels)       # both fields with their certificates
```

| Parameter | Default | Description |
|---|---|---|
| `w` | `2.0` | Weight of the ordinal term, the local to global dial. Higher keeps more global distance. |
| `random_state` | `0` | Seed for the fit. |
| `device` | `"cpu"` | `"cuda"` to train on a GPU. |
| `density` | `False` | Fit the residual density, required for the diagnostics. |
| `n_components` | `2` | Embedding dimension. Only `2` is supported. |

## Reproducing the paper

The `scripts` directory reconstructs every figure and number in the paper. See `scripts/README.md`.
