import matplotlib.pyplot as plt
import numpy as np
from flodr.viz import plot_embedding
from matplotlib.colors import Normalize, to_rgb
from matplotlib.lines import Line2D
from scipy.ndimage import gaussian_filter
from scipy.spatial.distance import cdist
from scipy.stats import spearmanr

import common

# the class the differentiation axis starts from, and its short name
PROGENITOR = {
    "bmarrow": ("HSC/MPP and pro", "HSC/MPP"),
    "cerebellum": ("cerebellar granule cell precursor", "GC precursor"),
    "plant": ("Dividing", "dividing cells"),
}
LIM = (-0.03, 1.16)


def ranks(X):
    return np.argsort(np.argsort(cdist(X, X), axis=1, kind="stable"), axis=1)


def rnx(rp, ry):
    # the co-ranking block sums are the cumulative histogram of the larger rank of each pair
    n = len(rp)
    t = np.maximum(rp, ry)
    np.fill_diagonal(t, 0)
    k = np.arange(1, n // 2 + 1)
    q = np.cumsum(np.bincount(t.ravel(), minlength=n)[1:])[: len(k)] / (k * n)
    curve = ((n - 1) * q - k) / (n - 1 - k)
    return k, curve, np.sum(curve / k) / np.sum(1 / k)


def density_lines(ax, x, y, colour, levels=(0.5, 0.8, 0.95)):
    # outlines of the smallest regions holding these shares of the points, innermost darkest
    H = gaussian_filter(np.histogram2d(x, y, bins=160, range=[LIM, LIM])[0], 2.6)
    flat = np.sort(H.ravel())[::-1]
    mass = np.cumsum(flat) / flat.sum()
    cuts = [flat[min(np.searchsorted(mass, q), len(flat) - 1)] for q in levels[::-1]]
    base = np.asarray(to_rgb(colour))
    shades = [tuple(base + (1 - base) * f) for f in np.linspace(0.62, 0, len(levels))]
    c = np.linspace(*LIM, 321)[1::2]  # the bin centres
    ax.contour(c, c, H.T, levels=cuts, colors=shades, linewidths=0.9)


def draw(atlas, name, ncol, legend_size=7.5):
    plt.rcParams.update({**common.RC, "axes.axisbelow": True})
    P, labels = common.load(atlas)
    layouts = {"FloDR": common.fit(atlas, "FloDR (w=2)", 0)[0]}
    layouts["UMAP"] = common.fit(atlas, "UMAP", 0)[0]
    rng = np.random.default_rng(0)
    i, j = rng.integers(0, len(P), (2, 60_000))
    i, j = i[i != j], j[i != j]
    sub = np.sort(rng.permutation(len(P))[:6000])
    dh, rp = np.linalg.norm(P[i] - P[j], axis=1), ranks(P[sub])
    curves = {m: rnx(rp, ranks(Y[sub])) for m, Y in layouts.items()}
    axis = np.linalg.norm(P - P[labels == PROGENITOR[atlas][0]].mean(0), axis=1)
    lo, hi = np.percentile(axis, [1, 99])
    kw = dict(cmap=common.cmap(common.SEQ), norm=Normalize(lo, hi), s=0.5)
    pal = common.class_colours(labels)[1]
    uniq, counts = np.unique(labels, return_counts=True)
    order = uniq[np.argsort(-counts)]

    # geometry in inches: the two maps, the legend, then one analysis row per method
    x0, span, row, top, gap = 0.27, 6.89, 1.43, 3.38, 0.484
    cx2 = x0 + row + 0.876
    cw = (x0 + span - cx2 - gap) / 2
    rows_y = {"FloDR": 0.51 + row + 0.16, "UMAP": 0.51}
    leg_y = rows_y["FloDR"] + row + 0.45 + -(-len(uniq) // ncol) * 0.155
    W, H = common.TEXT_W, leg_y + 0.10 + top + 0.52
    fig = plt.figure(figsize=(W, H))

    def box(x, y, w, h):
        return fig.add_axes([x / W, y / H, w / W, h / H])

    def text(x, y, s, **kw):
        fig.text(x / W, y / H, s, ha="center", **kw)

    text(x0 + span / 2, H - 0.21, f"{common.NAMES[atlas]} ({len(P):,} cells)", size=9)
    for k, (m, Y) in enumerate(layouts.items()):
        ax = box(x0 + k * (top + 0.13), leg_y + 0.10, top, top)
        plot_embedding(Y, labels, ax=ax, colours=pal, s=0.8).set_title(m, fontsize=9)
        common.letter(ax, "ab"[k])
    fig.legend(
        [Line2D([], [], marker="o", ls="", ms=4, color=pal[c]) for c in order],
        [common.PRETTY.get(c, c) for c in order],
        ncol=ncol,
        loc="upper center",
        bbox_to_anchor=((x0 + span / 2) / W, leg_y / H),
        frameon=False,
        fontsize=legend_size,
        handletextpad=0.3,
        columnspacing=1.4,
        labelspacing=0.45,
    )
    y = rows_y["FloDR"] + row + 4 / 72
    head = f"differentiation axis\n(dist. from {PROGENITOR[atlas][1]})"
    text(x0 + (row + 0.15) / 2, y, head, va="baseline", size=7.5, linespacing=1.4)
    text(cx2 + cw / 2, y, "pairwise distances", va="baseline")
    text(cx2 + 1.5 * cw + gap, y, "neighbourhood preservation", va="baseline")

    for r, (m, Y) in enumerate(layouts.items()):
        y0, colour = rows_y[m], common.METHOD_STYLE[m][0]
        text(x0 - 0.21, y0 + row / 2, m, rotation=90, va="center", size=9)
        ax = plot_embedding(Y, values=axis, ax=box(x0, y0, row, row), **kw)
        common.letter(ax, "cd"[r])
        cax = box(x0 + row + 0.045, y0, 0.105, row)
        fig.colorbar(ax.collections[0], cax=cax, extend="both").outline.set_lw(0.6)
        cax.set_yticks([lo, hi], [f"near\n{lo:.0f}", f"far\n{hi:.0f}"], fontsize=7)
        cax.tick_params(length=0, pad=1.5)

        dl = np.linalg.norm(Y[i] - Y[j], axis=1)
        x, y = dh / np.percentile(dh, 99), dl / np.percentile(dl, 99)
        rho = spearmanr(dh, dl)[0]
        sh = box(cx2, y0, cw, row)
        density_lines(sh, x, y, colour)
        e = [0, LIM[1]]
        sh.plot(e, e, ls=":", lw=0.8, color="0.25")
        sh.plot(e, np.polyval(np.polyfit(x, y, 1), e), lw=1, color="0.15")
        sh.set(xlim=LIM, ylim=LIM, xticks=[0, 0.5, 1], yticks=[0, 0.5, 1])
        sh.set_ylabel("embedded distance")
        sh.text(0.03, 0.965, rf"$\rho = {rho:.2f}$", transform=sh.transAxes, va="top")

        ax = box(cx2 + cw + gap, y0, cw, row)
        for o in ("UMAP" if m == "FloDR" else "FloDR", m):
            st = dict(lw=1.6, c=colour) if o == m else dict(ls="--", lw=1, c="0.55")
            ax.plot(*curves[o][:2], label=f"{o} ({curves[o][2]:.3f})", **st)
        ax.set(xscale="log", ylim=(0, 0.63), ylabel=r"$R_{NX}(k)$")
        ax.grid(True, lw=0.4, color="0.9")
        ax.legend(frameon=False, fontsize=7, loc="upper left", handlelength=1.6)
        for p in (sh, ax):
            p.tick_params(labelsize=7.5, labelbottom=r == 1)
    sh.set_xlabel("true distance (PCA-50)")
    ax.set_xlabel("neighbourhood size $k$")
    common.save(fig, name)


if __name__ == "__main__":
    draw("bmarrow", "fig4.pdf", ncol=5)
