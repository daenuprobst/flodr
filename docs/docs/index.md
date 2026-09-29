---
slug: /
title: FloDR
sidebar_position: 1
---

# FloDR

FloDR reduces high-dimensional data to two dimensions through an invertible normalising flow. The flow maps each
point to as many coordinates as it came in with. The first two are the embedding and the rest are kept as a
residual. Because the trained map is a bijection, the same model that draws the layout can decode positions back
to the input space and give an exact log density for any point.

```python
from flodr import FloDR

model = FloDR(random_state=0).fit(X)
Y = model.embedding_
```

FloDR follows the scikit-learn estimator interface. It fits, transforms, clones, pickles and runs inside a
`Pipeline` like any other transformer.

## What a fitted model offers

| Attribute or method | What it gives |
|---|---|
| `embedding_` | the layout of the training data |
| `transform`, `transform_opt` | place new points on the layout |
| `inverse_latent` | decode full latent coordinates back to inputs, exactly |
| `inverse_transform` | decode a position of the layout to one representative input |
| `score_samples` | the log density of any input under the model |
| `conditional_spread`, `hidden_contrast` | diagnostic fields that report what the layout does not show, each with a held-out test |

## Where to go next

Start with [Installation](installation.md) and the [Quickstart](quickstart.md). The user guide explains
[fitting](guide/fitting.md), [which input to use](guide/inputs.md) and each part of a fitted model in turn. The
[API reference](api/flodr.md) lists every parameter, attribute and method.
