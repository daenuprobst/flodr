---
title: flodr.viz
sidebar_position: 2
---

# `flodr.viz`

Plotting helpers built on matplotlib, and the functions behind the diagnostics.

## Plotting

### `plot_embedding(Y, labels=None, ax=None, cmap="tab10", s=6, title=None)`

Draws the layout `Y` as a scatter, coloured by `labels` when given, and returns the axes.

### `plot_field(Y, v, ax=None, cmap="Reds", s=8)`

Draws the layout coloured by one value per point, such as `conditional_spread`, with a colour bar, and returns the
axes.

### `bivariate_field(Y, x, y, ax=None, scheme="accessible", ...)`

Draws two fields over one layout with a two-dimensional colour key. Useful for the spread and the hidden contrast
together.

### `occupancy_field(Y, n_grid=100, scale=5.0, pad=0.06)`

Returns grid axes over the layout and a soft mask of where the layout has points.

### `depth_fog(Y, elev=22.0, azim=-60.0, strength=0.9, gamma=2.4)`

For three-dimensional layouts. Returns a drawing order, a colour weight and a depth per point for a view from
`elev` and `azim`.

### `save(fig, path, dpi=300)`

Saves a figure with tight bounds.

### `RC`, `RC_PAPER`

Matplotlib settings for the screen and for print, for use with `matplotlib.rc_context`.

## Diagnostics

These take a trained flow directly and are what the estimator's methods call.

### `conditional_spread(flow, Y, to_input=None, n_samples=64, seed=0, chunk=1024)`

The spread of decoded inputs at the positions `Y`.

### `conditional_moments(flow, Y, to_input=None, n_samples=64, seed=0, chunk=1024)`

The mean and the spread of decoded inputs at the positions `Y`.

### `conditional_atypicality(flow, Z, to_input=None, n_samples=128, seed=0, chunk=512)`

The atypicality of the points with latent coordinates `Z`.

### `residual(flow, X)`, `residual_fraction(flow, X)`

The residual coordinates of whitened inputs `X`, and the median share of the latent norm they carry.
