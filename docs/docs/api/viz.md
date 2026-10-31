---
title: flodr.viz
---

Every plotting function takes `ax=None` (a new axes) and returns the axes. See [Visualisation](../guide/visualisation.md).

```python source
flodr.viz.plot_embedding
flodr.viz.plot_frontier
flodr.viz.plot_ranks
flodr.viz.plot_paths
flodr.viz.plot_band
flodr.viz.plot_audit
flodr.viz.frame
flodr.viz.palette
flodr.viz.save
```

| Function | Draws |
|---|---|
| `plot_embedding` | a map coloured by label (`colours`: label to colour) or by value; `order="high"` draws high values last |
| `plot_frontier` | per method the mean and sd of `(recalls, cpds)`; `styles`: method to `(colour, marker)` |
| `plot_ranks` | the empirical law of certificate ranks against the diagonal, with the DKW band and the margin |
| `plot_paths` | paths over a map; `ends` names their ends, `far` dashes steps far from the training points |
| `plot_band` | a value decoded along a path, with its band |
| `plot_audit` | the result of `flodr.audit`: the labels and the hidden information |
| `palette` | a colour per label, tab10 for up to ten labels, else tab20 |
| `frame` | sets a square 1:1 window around the map |
| `save` | saves with tight bounds |

`RC` is a matplotlib style for screen figures.
