---
title: Diagnostics
sidebar_position: 6
---

# Diagnostics

A two-dimensional layout cannot hold everything in its input. FloDR offers two fields that show, position by
position, what the layout leaves out. Each comes with a test on held-out points, and a field that fails its test
should not be read.

## Conditional spread

```python
sigma = model.conditional_spread()        # at every training point
sigma_at = model.conditional_spread(Y)    # at any positions, Y has n_components columns
```

At a position of the layout, the model has a conditional density over the residual. `conditional_spread` draws
`n_samples` residuals from it, 64 by default, decodes each one together with the position, and returns the square
root of the total variance of the decoded inputs. It is measured in the units of the input. A large value means the
model places very different inputs at that position. It needs `density=True`.

```python
from flodr import viz

ax = viz.plot_field(model.embedding_, sigma)
```

## Atypicality

```python
a = model.atypicality()   # (n,) values between 0 and 1
```

For each training point, `atypicality` decodes `n_samples` inputs at its position, 128 by default, and returns the
fraction of them that lie closer to their mean than the point itself. Values near 1 mark points that are unusual
for the position they were given. It needs `density=True`.

## Testing the spread

```python
cert = model.spread_calibration()
cert["passed"], cert["confidence"]
```

`spread_calibration` fits the conditional density again on half of the points and predicts the spread at the
positions of the other half. It divides the layout into a 14 by 14 grid and, in every cell with at least 12
held-out points, compares the predicted variance with the variance of the inputs actually placed there.

The field passes when the median ratio of predicted to measured variance lies between 0.5 and 2. If the measured
variance changes at least threefold across the grid, the log of the measured variance is also regressed on the log
of the predicted one, and the slope must lie between 0.7 and 1.3 with R squared of at least 0.6. With less
variation there is too little range to test the shape, only the level is tested, and `cert["certifies"]` reads
`"magnitude-only"`.

`confidence` is the smallest pass rate among these parts over 2,000 bootstrap resamples of the grid cells.

## Hidden contrast

```python
field, cert = model.hidden_contrast(labels)
```

The hidden contrast measures, at each position, how much information about a label is present in the input but
lost in the layout. It is measured in nats.

The points are split into three parts. On the first, two classifiers learn to predict the label, one from the input
and one from the layout position. On the other parts, the difference between their log likelihoods of the true
label measures what the input knows and the layout does not. The field is that difference averaged over the 10
nearest points of the second part, which makes it a function of position.

The same classifiers are also trained on labels shuffled within the cells of a 14 by 14 grid, 99 times by default.
Shuffling within a cell keeps what the layout shows and destroys what it hides, so these replicas give the null.
Their mean is subtracted from the field.

The field passes when two conditions hold on the third part. Its mean gap must beat the shuffled replicas at level
`alpha`, 0.01 by default, which with 99 replicas means exceeding every one of them. The field of the second part must
predict the gap measured per grid cell, with a slope between 0.7 and 1.3 and R squared of at least 0.6. With fewer than three grid cells of at least 16 points the result
is `{"passed": None, "certifies": "undecided"}`.

The hidden contrast reads the input and the layout. It does not use the inverse or the density, so it also works
with `density=False`. It trains 100 classifiers per view in two batched passes and takes under a minute for 250,000
points on a GPU.

## Both at once

```python
out = model.diagnostics(labels)
out["spread"]["field"], out["spread"]["cert"]
out["hidden_contrast"]["field"], out["hidden_contrast"]["cert"]
```

`diagnostics` returns both fields with their tests, units and plot labels. Without labels it returns the spread
alone, and without the density the hidden contrast alone. Keyword arguments go to `hidden_contrast`.

## Reading the results

A field that passes predicted held-out data within the tolerances above. It can still be wrong in ways the test
does not probe. A field that fails should not be read. Both tests divide the layout into grid cells and need enough
points in each, so on a few thousand points they are often undecided or fail for lack of data rather than because
the field is wrong.
