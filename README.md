# FloDR

**[Documentation](https://daenuprobst.github.io/flodr/)**

Dimensionality reduction built on an invertible normalising flow. The first `n_components` coordinates of the flow
are the map and the rest are a residual, so the fitted map has an exact inverse and an exact density. FloDR follows
the scikit-learn estimator interface, like UMAP.

![](assets/flow_morph_bmarrow_white.gif)

## Installation

Python 3.11 or newer. Training uses the GPU when PyTorch sees one.

```bash
pip install flodr
```

## Usage

```python
import matplotlib.pyplot as plt
from flodr import FloDR, audit, viz

model = FloDR(random_state=0).fit(X)
Y = model.embedding_                         # the map

Y_new = model.transform(X_new)               # one pass through the flow
Y_opt = model.transform_opt(X_new)           # optimised against the training map, as UMAP's transform
X_back = model.decode(Y[:10])                # positions back to inputs, from the fitted points around them
Z = model.transform_latent(X_new)            # the map, then the residual
X_same = model.inverse_latent(Z)             # exact up to float precision

logp = model.score_samples(X_new)            # log density
cert = model.coverage_certificate(X_new)     # is the density a law of new points?

res = audit(Y, X=X, labels=labels)           # what the map hides about the labels

fig, ax = plt.subplots(1, 3, figsize=(12, 4))
viz.plot_embedding(Y, labels, ax=ax[0])
viz.plot_ranks([cert["fibre"]["ranks"], cert["map"]["ranks"]], ax=ax[1])
viz.plot_audit(res, axes=ax[2], show_labels=False)
```

| Parameter | Default | |
|---|---|---|
| `n_components` | `2` | dimension of the map |
| `w` | `2.0` | weight of the distance order: 0 keeps neighbourhoods only, larger keeps more global structure |
| `axes` | `"hinge"` | start from the plane that best keeps the input's distance order, the same for every seed; `"variance"` from the leading principal components, or `"heat"` from the heat geometry of the neighbour graph |
| `density` | `True` | fit the density, needed by `score_samples` and the certificates |
| `max_iter` | `None` | training steps, 6,400 by default |
| `fast` | `False` | a quarter of the steps on two of the four coupling layers: about six times faster, fewer neighbours kept on large data |
| `random_state` | `0` | seed |
| `device` | `None` | the GPU when there is one |

`examples/quickstart.ipynb` runs these calls on Fashion-MNIST.

## Development

```bash
git clone https://github.com/daenuprobst/flodr
cd flodr
uv sync --extra dev
uv run pytest
```

The documentation is a Starlight site built from `docs/docs/*.md` and the signatures in `src/flodr`:
`cd docs && npm install && npm run dev`.

## Reproducing the paper

`scripts/` holds one script per figure and table of the paper, named after it (`fig2.py`, `table1.py`,
`ed_table1.py`, `supp_table1.py`, ...), with `scripts/common.py` for the data, the methods and the scores they share
and `scripts/openproblems.py` for the OpenProblems scores, shared in the same way. Each script fits what it needs,
caches it under `scripts/cache`, so scripts that share fits reuse them, and writes its figure or table to
`scripts/out`. The tests against unseen measurements (Fig. 5 and Supplementary Tables 4 and 5) are not among these
scripts.

```bash
uv sync --extra paper
uv pip install git+https://github.com/hyhuang00/ParamRepulsor.git
uv run python scripts/fig2.py
```

Raw data go into `data/` (or the folder named by `DATA`); the data availability statement of the paper lists the
sources, and each script names the files it reads. Two tools need interpreters of their own, named by
`PCUMAP_PYTHON` (Parametric UMAP: TensorFlow 2.15 with umap-learn 0.5.5) and `DREAMS_PYTHON` (DREAMS: the authors'
openTSNE fork). The OpenProblems scripts (`fig1.py`, `table1.py`, `ed_table3.py`, `supp_table3.py`) read the published
scores from `scripts/cache/openproblems` (or the folder named by `OPENPROBLEMS`) and run FloDR through the task
where its scores are missing: version 2 with `viash`, Docker and a checkout of `task_dimensionality_reduction` in
that folder, version 1 with a checkout of `openproblems` there and its Python 3.8 interpreter, named by
`OPENPROBLEMS_PYTHON`.
