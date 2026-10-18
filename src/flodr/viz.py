import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import to_rgb

from .audit import plot_audit

RC = {
    "font.size": 11,
    "axes.titlesize": 12,
    "figure.titlesize": 15,
    "legend.fontsize": 8,
    "font.family": "sans-serif",
    "pdf.fonttype": 42,
    "ps.fonttype": 42,
    "savefig.facecolor": "white",
}

__all__ = [
    "RC",
    "frame",
    "plot_embedding",
    "plot_frontier",
    "plot_ranks",
    "plot_paths",
    "plot_band",
    "plot_audit",
    "save",
]


def frame(ax, Y, crop=(1, 99), pad=0.03):
    # a square box at 1:1 data scale around the given percentiles, so a few far points
    # do not shrink the map
    lo, hi = np.percentile(np.asarray(Y, float)[:, :2], crop, axis=0)
    rad, (cx, cy) = (hi - lo).max() * (0.5 + pad), (lo + hi) / 2
    ax.set_box_aspect(1)
    ax.set_aspect("equal", adjustable="datalim")
    ax.set(xlim=(cx - rad, cx + rad), ylim=(cy - rad, cy + rad), xticks=[], yticks=[])

    for spine in ax.spines.values():
        spine.set(linewidth=0.6, color="0.6")

    return ax


def palette(labels):
    uniq = np.unique(np.asarray(labels).astype(str))
    cycle = plt.get_cmap("tab10" if len(uniq) <= 10 else "tab20").colors

    return {c: cycle[i % len(cycle)] for i, c in enumerate(uniq)}


def plot_embedding(
    Y,
    labels=None,
    values=None,
    ax=None,
    colours=None,
    cmap="viridis",
    norm=None,
    s=None,
    order="random",
    crop=(1, 99),
    text=None,
):
    # a map coloured by labels (colours maps label to colour) or by values (on cmap);
    # order "random" shuffles the points so no class hides another, "high" and "low"
    # draw the high or low values last
    Y = np.asarray(Y, float)
    ax = ax or plt.figure().add_subplot()
    o = np.random.default_rng(0).permutation(len(Y))
    kw = dict(s=max(0.5, 7000 / len(Y)) if s is None else s, lw=0, rasterized=True)

    if values is not None:
        values = np.asarray(values, float)
        o = (
            o
            if order == "random"
            else np.argsort(values if order == "high" else -values)
        )
        kw.update(c=values[o], cmap=cmap, norm=norm)
    elif labels is not None:
        labels = np.asarray(labels).astype(str)
        colours = colours or palette(labels)
        kw.update(c=[colours[c] for c in labels[o]])
    else:
        kw.update(c="0.3")

    ax.scatter(*Y[o].T, **kw)
    frame(ax, Y, crop)

    if text:
        box = dict(boxstyle="round,pad=0.25", fc="white", ec="none")
        ax.text(
            0.03, 0.03, text, transform=ax.transAxes, fontsize=6.5, bbox=box, zorder=10
        )

    return ax


def plot_frontier(results, styles=None, ax=None, xlabel="recall@15", ylabel="CPD"):
    # the local-to-global trade-off: per method the mean over runs with one-sd bars;
    # results maps a method to its (recalls, CPDs), styles to its (colour, marker)
    ax = ax or plt.figure().add_subplot()

    for i, (m, (r, c)) in enumerate(results.items()):
        colour, marker = (styles or {}).get(m, (f"C{i}", "o"))
        (x, dx), (y, dy) = (np.mean(r), np.std(r)), (np.mean(c), np.std(c))
        kw = dict(color=colour, lw=0, zorder=3)
        ax.errorbar(x, y, dy, dx, elinewidth=0.6, alpha=0.5, **kw)
        ms = 7.5 if marker == "*" else 6
        ax.plot(x, y, marker=marker, ms=ms, mew=0.6, mec="w", label=m, **kw)

    ax.grid(True, lw=0.4, color="0.88")
    ax.set_axisbelow(True)
    ax.margins(0.13)
    ax.set(xlabel=xlabel, ylabel=ylabel)

    return ax


def plot_ranks(ranks, ax=None, colours=None, margin=0.1, alpha=0.05):
    # the empirical law of certificate ranks (cert["fibre"]["ranks"], ...) against the
    # diagonal, with the DKW band at their count and the certificate's margin
    ranks = [np.asarray(ranks)] if np.ndim(ranks[0]) == 0 else list(ranks)
    ax = ax or plt.figure().add_subplot()
    g = np.linspace(0, 1, 201)
    eps = np.sqrt(np.log(2.0 / alpha) / (2.0 * len(ranks[0])))
    ax.fill_between(
        g, np.clip(g - eps, 0, 1), np.clip(g + eps, 0, 1), color="0.88", lw=0
    )

    for m in (-margin, margin):
        ax.plot(g, np.clip(g + m, 0, 1), color="0.55", lw=0.5, ls=(0, (2, 2)))

    ax.plot([0, 1], [0, 1], color="0.2", lw=0.6)

    for i, r in enumerate(ranks):
        c = colours[i] if colours is not None else f"C{i}"
        y = np.arange(1, len(r) + 1) / len(r)
        ax.step(np.sort(r), y, where="post", color=c, lw=0.8)

    ax.set(
        xlim=(0, 1), ylim=(0, 1), aspect="equal", xticks=[0, 0.5, 1], yticks=[0, 0.5, 1]
    )
    ax.set_xlabel("held-out rank")

    return ax


def plot_paths(
    Y,
    paths,
    labels=None,
    colours=None,
    ends=None,
    far=None,
    highlight=None,
    ax=None,
    crop=(1, 99),
):
    # paths drawn over the map, each from a dot to an arrowhead; far marks the steps with
    # no training cell nearby (white dashes), ends names the two ends of each path as
    # ((text, colour), (text, colour)), and cells outside highlight are grey
    ax = ax or plt.figure().add_subplot()
    on = np.ones(len(Y), bool) if highlight is None else np.asarray(highlight)
    c = "0.3"

    if labels is not None:
        labels = np.asarray(labels).astype(str)
        colours = colours or palette(labels)
        c = np.array([colours[x] for x in labels[on]])

    ax.scatter(*Y[~on].T, s=0.6, c="0.85", lw=0, rasterized=True)
    ax.scatter(*Y[on].T, s=0.6, c=c, lw=0, rasterized=True)
    frame(ax, np.vstack([Y[on], *paths]), crop, pad=0.1)
    span = np.diff(ax.get_xlim())[0]
    arrow = dict(arrowstyle="-|>", color="k", lw=1.0, mutation_scale=7)
    box = dict(boxstyle="round,pad=0.1", fc="w", ec="none")

    for p, path in enumerate(paths):
        ax.plot(*path.T, c="k", lw=1.0, solid_capstyle="round", zorder=4)

        if far is not None:
            dashed = np.where(np.asarray(far[p])[:, None], path, np.nan)
            ax.plot(*dashed.T, c="w", lw=0.6, ls=(0, (1.5, 1.5)), zorder=5)

        ax.scatter(*path[0], s=10, c="k", lw=0, zorder=5)
        ax.annotate("", xy=path[-1], xytext=path[-4], arrowprops=arrow, zorder=5)

        for (text, colour), xy, nb in zip(
            ends[p] if ends else (), path[[0, -1]], path[[6, -7]]
        ):
            u = (xy - nb) / (np.linalg.norm(xy - nb) + 1e-9)
            # darkened, so a light class colour still reads as text on white
            dark = 0.55 * np.array(to_rgb(colour))
            kw = dict(ha="center", va="center", zorder=6, bbox=box, fontsize=7)
            ax.text(*(xy + 0.07 * span * u), text, color=dark, **kw)

    return ax


def plot_band(
    lo,
    hi,
    mean,
    near=None,
    far=None,
    label=None,
    ax=None,
    colours=("#DCD3DC", "#AA4499", "#2E8B7A"),
):
    # a quantity decoded along a path: the band from lo to hi, the mean of the draws,
    # ticks for the mean of the nearest training cells, grey where none is near
    ax = ax or plt.figure().add_subplot()
    band, colour, near_colour = colours
    x = np.linspace(0, 1, len(lo))

    if far is not None:
        edges = np.flatnonzero(np.diff(np.r_[0, np.asarray(far, int), 0]))

        for s, e in zip(edges[::2], edges[1::2]):
            ax.axvspan(x[s], x[e - 1], color="0.93", lw=0, zorder=0)

    ax.fill_between(x, lo, hi, color=band, lw=0, zorder=1)
    ax.plot(x, mean, c=colour, lw=1.0, zorder=3)
    ymin, ymax = np.nanmin(lo), np.nanmax(hi)

    if near is not None:
        t = np.linspace(0, len(x) - 1, 18).round().astype(int)
        ax.scatter(x[t], near[t], marker="_", s=14, lw=0.9, c=near_colour, zorder=4)
        ymin, ymax = min(ymin, np.min(near)), max(ymax, np.max(near))

    pad = 0.06 * (ymax - ymin)
    ax.set(xlim=(0, 1), ylim=(ymin - pad, ymax + pad), xticks=[0, 1])
    ax.set_xticklabels(["start", "end"])
    ax.spines[["top", "right"]].set_visible(False)

    if label:
        ax.text(0.99, 0.97, label, transform=ax.transAxes, ha="right", va="top")

    return ax


def save(fig, path, dpi=300):
    fig.savefig(path, dpi=dpi, bbox_inches="tight")
