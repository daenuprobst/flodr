---
title: Performance
sidebar_position: 8
---

# Performance

## Time

Times on an RTX 4070 Ti with the default settings.

| Points | Input | Fit |
|---|---|---|
| 5,000 | 50 to 2,048 columns | about 2 minutes |
| 100,000 | 50 principal components | about 4 minutes |
| 250,000 | 10-dimensional latent, padded to 30 | about 6 minutes |

Each step trains on the whole neighbour graph up to 262,144 edges, and on a random batch of that many above, so the
cost of a step stops growing with the number of points. Between 50,000 and 500,000 points only the heaviest quarter
of the graph edges is kept.

`max_iter` trades time for quality. Fewer steps train faster and give a rougher layout.

## Memory

A fit of 250,000 points peaks at about 2.7 GB of GPU memory. In version 0.1.0 the last step of `fit` passes all
points through the flow at once, which on a 12 GB GPU ran out of memory at 1.77 million points.

## CPU

Training runs on the CPU when no GPU is available or with `device="cpu"`, several times slower than on a GPU. FloDR
limits PyTorch to the performance cores of the CPU.

## Reproducibility

On the CPU a fixed `random_state` reproduces a fit exactly. On CUDA the training step is compiled and runs its hidden
layers in half precision, which is about twice as fast but not bitwise reproducible between runs. The eager path in
full precision runs with

```python
FloDR(advanced={"compile_edges": False, "half_edges": ""})
```

## Checking a fit

In rare cases a fit returns coordinates that are not finite. This was seen once, at `w=3`, and a refit with the same
seed was finite. Check the layout and refit with another seed if it happens.

```python
import numpy as np

assert np.isfinite(model.embedding_).all()
```
