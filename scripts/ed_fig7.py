import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from flodr import viz
from matplotlib.colors import LogNorm, PowerNorm, TwoSlopeNorm

import common
from ed_fig6 import COVERAGE, coverage_field, fitted, model


def readings(ds):
    # per training cell of the seed-0 map: rho(1), the decoded error of a noise-sized
    # step over the step, and the lower band of what the input keeps about cell type
    def make():
        from flodr.audit import audit

        print(f"faithful scale and audit, {ds}", flush=True)
        est, (X, labels) = model(ds, 0), common.load(ds)
        tr = common.split(len(X))[0]
        res = audit(est.embedding_, X=X[tr], labels=labels[tr].astype(str), test=False)
        return dict(
            rho=est.faithful_scale()["rho"][:, 3], lower=res.lower, level=res.level
        )

    return common.cached(os.path.join(common.CACHE, "ed_fig7", f"{ds}.npz"), make)


# rho(1) on a log scale from 0.25 to 10, white where the decoded error equals the step
mid = np.log(4) / np.log(40)
RHO = [
    (0, "#1F5E54"),
    (mid / 2, "#44AA99"),
    (mid, "#F2F0F1"),
    ((1 + mid) / 2, common.OURS),
]
HIDDEN = [(0, "#F2EEF1"), (0.15, "#E8D0E2"), (0.45, "#C98AB9"), (0.75, common.OURS)]
CMAPS = [COVERAGE] + [common.cmap(c + [(1, "#5E1B4F")]) for c in (RHO, HIDDEN)]
NORMS = (TwoSlopeNorm(0.9, 0.5, 1.0), LogNorm(0.25, 10), PowerNorm(0.5, 0, 1))
TITLES = ("held-out coverage", r"faithful scale $\rho(1)$", "hidden about cell type")
TICKS = ([0.5, 0.7, 0.9, 1], [0.25, 0.5, 1, 2, 5, 10], [0, 0.1, 0.25, 0.5, 1])
LABELS = (
    "share of held-out cells inside their 90\\% region",
    r"$\rho(1)$, decoded error over step",
    "nats, lower 95\\% band",
)

plt.rcParams.update(common.RC)
fig = plt.figure(figsize=(common.TEXT_W, common.TEXT_W * 0.86))
gs = fig.add_gridspec(4, 4, height_ratios=(1, 1, 1, 0.055), hspace=0.06, wspace=0.04)
gs.update(left=0.04, right=0.995, top=0.965, bottom=0.075)
for r, ds in enumerate(common.ATLASES):
    fit, read, labels = fitted(ds, 0), readings(ds), common.load(ds)[1].astype(str)
    labels = labels[common.split(len(labels))[0]]
    print(
        f"{ds}: hides {float(read['level']):.3f} nats about cell type, lower band"
        f" above zero at {(read['lower'] > 0).mean():.0%} of positions"
    )
    axes = [fig.add_subplot(gs[r, c]) for c in range(4)]
    colours = common.class_colours(labels)[1]
    viz.plot_embedding(fit["Y"], labels, ax=axes[0], colours=colours, s=0.5)
    axes[0].set_ylabel(common.NAMES_2L[ds], fontsize=7.5)
    values = (coverage_field(fit), read["rho"], np.clip(read["lower"], 0, None))
    for ax, v, norm, cmap, o in zip(
        axes[1:], values, NORMS, CMAPS, ("low", "high", "high")
    ):
        viz.plot_embedding(
            fit["Y"], values=v, ax=ax, cmap=cmap, norm=norm, s=0.5, order=o
        )
    for c, title in enumerate(("cell type",) + TITLES if r == 0 else ()):
        axes[c].set_title(title, fontsize=8)
        common.letter(axes[c], "abcd"[c])
for c in range(3):
    cax = fig.add_subplot(gs[3, c + 1].subgridspec(1, 3, width_ratios=(1, 8, 1))[0, 1])
    sm = plt.cm.ScalarMappable(NORMS[c], CMAPS[c])
    cb = fig.colorbar(
        sm, cax, orientation="horizontal", extend=("min", "both", "max")[c]
    )
    labels = [f"{t:g}" for t in TICKS[c]]
    cb.set_ticks(TICKS[c], labels=["$\\le$0.5"] + labels[1:] if c == 0 else labels)
    cb.ax.minorticks_off()
    cb.ax.tick_params(labelsize=6.5, length=2, pad=1.5)
    cb.outline.set_linewidth(0.4)
    cb.set_label(LABELS[c], fontsize=7, labelpad=2)
common.save(fig, "ed_fig7.pdf")
