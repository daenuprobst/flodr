---
title: flodr.viz
sidebar_position: 2
---

# `flodr.viz`

Every plotting function takes `ax=None` (a new axes) and returns the axes. See [Visualisation](../guide/visualisation.md).

| Function | Draws |
|---|---|
| `plot_embedding(Y, labels=None, values=None, colours=None, cmap="viridis", norm=None, s=None, order="random", crop=(1, 99), text=None)` | a map coloured by label (`colours`: label to colour) or by value; `order="high"` draws high values last |
| `plot_frontier(results, styles=None, xlabel="recall@15", ylabel="CPD")` | per method the mean and sd of `(recalls, cpds)`; `styles`: method to `(colour, marker)` |
| `plot_ranks(ranks, colours=None, margin=0.1, alpha=0.05)` | the empirical law of certificate ranks against the diagonal, with the DKW band and the margin |
| `plot_paths(Y, paths, labels=None, colours=None, ends=None, far=None, highlight=None, crop=(1, 99))` | paths over a map; `ends` names their ends, `far` dashes steps far from the training points |
| `plot_band(lo, hi, mean, near=None, far=None, label=None)` | a value decoded along a path, with its band |
| `plot_audit(res, axes=None, show_labels=True)` | the result of `flodr.audit`: the labels and the hidden information |
| `frame(ax, Y, crop=(1, 99), pad=0.03)` | sets a square 1:1 window around the map |
| `save(fig, path, dpi=300)` | saves with tight bounds |

`RC` is a matplotlib style for screen figures.
