"""Local-global frontier figure (frontier_all.pdf), read from the caches written by
evaluate.py (benchmarks) and scrna_bench.py (atlases). Nothing here recomputes an embedding.

Run: .venv/bin/python scripts/plotting/figures.py
"""
import json
import os
import sys

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__))
SCRIPTS = os.path.dirname(HERE)
ROOT = os.path.dirname(SCRIPTS)
sys.path.insert(0, os.path.join(ROOT, "src"))
sys.path.insert(0, SCRIPTS)
sys.path.insert(0, HERE)
from datasets import CACHE  # noqa: E402
from flodr import viz  # noqa: E402

FIG = os.path.join(ROOT, "figures")
EVAL = os.path.join(CACHE, "eval")
os.makedirs(FIG, exist_ok=True)

DS = [("mnist", "MNIST"), ("fmnist", "Fashion-MNIST"), ("paul15", "paul15"),
      ("drfp", "Schneider 50k")]
DS_SCRNA = [("bmarrow", "Fetal bone marrow"), ("cerebellum", "Cerebellum"),
            ("plant", "Arabidopsis seed")]
FLODR = ["FloDR (w=3)", "FloDR (w=2)", "FloDR (w=1)", "FloDR (no ordinal)"]
# Okabe-Ito derived; marker shape duplicates hue so the figure survives greyscale and CVD
BASE = [("openTSNE", "o", "#0072B2"), ("UMAP", "s", "#009E73"),
        ("TriMap", "^", "#A6761D"), ("PaCMAP", "D", "#7570B3"),
        ("LocalMAP", "v", "#CC79A7"), ("PHATE", "*", "#56B4E9"),
        ("PyMDE", "h", "#8C564B"), ("PCUMAP", "<", "#E69F00"),
        ("SQuadMDS", ">", "#2F4F4F")]
FLO_C, REF_C = "#D55E00", "#7a7a7a"


def save(fig, name):
    fig.savefig(f"{FIG}/{name}.pdf", dpi=300, bbox_inches="tight")
    fig.savefig(f"{FIG}/{name}.png", dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"saved {name}", flush=True)


def meta(key):
    path = f"{EVAL}/{key}_metrics.json"
    return json.load(open(path)) if os.path.exists(path) else None


def scrna_meta(key):
    path = os.path.join(CACHE, "scrna", f"{key}_metrics.json")
    return json.load(open(path)) if os.path.exists(path) else None


def val(met, name, field):
    ms = met["methods"].get(name, {}).get("mean_std", {}).get(field)
    return (None, None) if not ms or ms[0] is None else (ms[0], ms[1] or 0.0)


def _frontier_panel(ax, met, title, xlabel=True, ylabel=True):
    xs, ys, xe, ye = [], [], [], []
    for name in FLODR:
        x, x_std = val(met, name, "recall")
        y, y_std = val(met, name, "gspear")
        if x is not None and y is not None:
            xs.append(x); ys.append(y); xe.append(x_std); ye.append(y_std)
    if xs:
        ax.plot(xs, ys, "-", color=FLO_C, lw=1.0, zorder=3)
        ax.errorbar(xs, ys, xerr=xe, yerr=ye, fmt="o", ms=4.0, color=FLO_C,
                    elinewidth=0.6, capsize=1.4, zorder=4, label="FloDR")
        if len(xs) > 1:                       # scRNA panels carry the w=2 point only
            for label, i in (("$w{=}3$", 0), ("no ord.", len(xs) - 1)):
                ax.annotate(label, (xs[i], ys[i]), textcoords="offset points", xytext=(4, 1),
                            fontsize=6.5, color=FLO_C, zorder=5)
    for name, mark, color in BASE:
        x, x_std = val(met, name, "recall")
        y, y_std = val(met, name, "gspear")
        if x is None:
            continue
        ax.errorbar([x], [y], xerr=[x_std], yerr=[y_std], fmt=mark, ms=4.5, color=color,
                    elinewidth=0.6, capsize=1.4, zorder=4, label=name)
    x, x_std = val(met, "PCA-2", "recall")
    y, y_std = val(met, "PCA-2", "gspear")
    if x is not None:
        ax.errorbar([x], [y], xerr=[x_std], yerr=[y_std], fmt="x", ms=4.5, color=REF_C,
                    elinewidth=0.6, capsize=1.4, zorder=4, label="PCA-2")
    ax.set_title(title, fontsize=8)
    ax.set_box_aspect(1)
    ax.grid(True, lw=0.3, color="0.88", zorder=0)
    ax.set_axisbelow(True)
    for spine in ("top", "right"):
        ax.spines[spine].set_visible(False)
    if xlabel:
        ax.set_xlabel("recall@15")
    if ylabel:
        ax.set_ylabel(r"CPD ($\rho$ of pairwise distances)")


def frontier_all():
    """Every dataset in one grid -- benchmark rows then scRNA -- with one shared legend
    occupying the spare cell."""
    have = [(k, t, meta(k)) for k, t in DS if meta(k)]
    have += [(k, t, scrna_meta(k)) for k, t in DS_SCRNA if scrna_meta(k)]
    if len(have) < 2:
        return
    ncol = 4
    nrow = int(np.ceil((len(have) + 1) / ncol))          # +1 cell for the legend
    with plt.rc_context(viz.RC_PAPER):
        fig, axgrid = plt.subplots(nrow, ncol, figsize=(2.05 * ncol, 2.25 * nrow))
        axs = np.atleast_1d(axgrid).ravel()
        for i, (key, title, met) in enumerate(have):
            ax = axs[i]
            _frontier_panel(ax, met, title, xlabel=i + ncol >= len(have),
                            ylabel=i % ncol == 0)
            if len(met["methods"]) < 3:
                # two-method panels: autoscale would blow a 0.01 recall gap up to the full
                # axis and read as a large one. Anchor at zero so the spacing is honest.
                ax.set_xlim(0, max(ax.get_xlim()[1], 1e-9) * 1.15)
                ax.set_ylim(0, max(ax.get_ylim()[1], 1e-9) * 1.12)
        # union of handles across panels: the scRNA panels carry FloDR and UMAP only
        seen, handles, labels = set(), [], []
        for ax in axs[: len(have)]:
            for handle, label in zip(*ax.get_legend_handles_labels()):
                if label not in seen:
                    seen.add(label); handles.append(handle); labels.append(label)
        for ax in axs[len(have):]:
            ax.axis("off")
        axs[len(have)].legend(handles, labels, loc="center", frameon=False, ncol=2,
                              fontsize=6.4, handletextpad=0.35, labelspacing=0.55,
                              columnspacing=1.0, borderaxespad=0.0)
        fig.tight_layout(h_pad=0.9, w_pad=0.8)
        save(fig, "frontier_all")


if __name__ == "__main__":
    frontier_all()
    print("FIGURES DONE", flush=True)
