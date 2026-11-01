import matplotlib.pyplot as plt
from flodr.viz import plot_frontier

from common import ATLASES, BENCHMARKS, METHOD_STYLE, NAMES, NAMES_2L, OURS, RC
from common import TEXT_W, letter, save, scores

BASELINES = ["openTSNE", "UMAP", "TriMap", "PaCMAP", "LocalMAP", "UMATO"]
BASELINES += ["ParamRepulsor", "PHATE", "PyMDE", "PCUMAP", "SQuadMDS", "PCA-2"]
# where the w labels sit, in points from their dot
W_LABEL = {1: (5, -8, "top"), 2: (6, -2, "top"), 3: (5, 3, "bottom")}

plt.rcParams.update(
    {**RC, "axes.linewidth": 0.6, "xtick.major.width": 0.6, "ytick.major.width": 0.6}
)
fig, axes = plt.subplots(2, 4, figsize=(TEXT_W, 3.9))
for i, ds in enumerate(BENCHMARKS + ATLASES):
    ax = axes.flat[i]
    results = {
        {"PCUMAP": "Parametric UMAP"}.get(m, m): scores(ds, m)[:, :2].T
        for m in BASELINES
    }
    xlabel = "neighbourhood recall@15" if i >= 3 else ""
    plot_frontier(results, METHOD_STYLE, ax, xlabel, "CPD" if i % 4 == 0 else "")
    ax.set_title({**NAMES, **NAMES_2L}[ds], fontsize=8, loc="left")
    letter(ax, "abcdefg"[i])
    ax.margins(x=0.24, y=0.13)

    # the FloDR sweep over w is one curve, w=2 the large dot and w=0 the hollow one
    sweep = [scores(ds, f"FloDR (w={w})")[:, :2] for w in range(4)]
    ax.plot(*zip(*[s.mean(0) for s in sweep]), color=OURS, lw=1.1, zorder=4)
    for w, s in enumerate(sweep):
        (x, y), (dx, dy) = s.mean(0), s.std(0)
        ax.errorbar(
            x,
            y,
            dy,
            dx,
            color=OURS,
            lw=0,
            elinewidth=0.8,
            capsize=1.5,
            marker="o",
            ms=7.5 if w == 2 else 5.0,
            mew=0.9 if w == 2 else 0.6,
            mec=OURS if w == 0 else "white",
            mfc="white" if w == 0 else OURS,
            zorder=6 if w == 2 else 5,
        )
        if w:
            dxt, dyt, va = W_LABEL[w]
            kw = dict(textcoords="offset points", color=OURS, fontsize=7, va=va)
            ax.annotate(f"$w = {w}$", (x, y), xytext=(dxt, dyt), **kw)

axes[1, 3].axis("off")
handles, labels = axes[0, 0].get_legend_handles_labels()
flodr = axes[1, 3].plot([], [], "o", color=OURS, ms=6, mew=0.6, mec="w")[0]
axes[1, 3].legend(
    [flodr] + handles,
    ["FloDR"] + labels,
    ncol=2,
    loc="center",
    frameon=False,
    fontsize=7,
    handletextpad=0.4,
    columnspacing=1.0,
    labelspacing=0.5,
)
fig.tight_layout(w_pad=0.6, h_pad=0.7)
save(fig, "fig3.pdf")
