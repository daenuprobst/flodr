import json
import os
import sys

import numpy as np
import matplotlib
import matplotlib.pyplot as plt

matplotlib.use("Agg")

HERE = os.path.dirname(os.path.abspath(__file__))
SCRIPTS = os.path.dirname(HERE)
ROOT = os.path.dirname(SCRIPTS)
sys.path.insert(0, os.path.join(ROOT, "src"))
from flodr import viz  # noqa: E402

FIG = os.path.join(ROOT, "figures")
CACHE = os.path.join(SCRIPTS, "cache")
os.makedirs(FIG, exist_ok=True)

ROW1 = [("mnist", "MNIST"), ("fmnist", "Fashion-MNIST"), ("paul15", "paul15"),
        ("drfp", "Schneider 50k")]
ROW2 = [("bmarrow", "Bone marrow"), ("cerebellum", "Cerebellum"),
        ("plant", "Arabidopsis seed")]
# (label, marker, colour, benchmark json key, atlas json key); Okabe-Ito, FloDR warm
ARMS = [("FloDR (forward pass)", "h", "#D55E00", "flodr", "flodr"),
        ("FloDR + optA", "D", "#CC79A7", "flodr_optA", "opta"),
        ("openTSNE transform", "o", "#0072B2", "opentsne", "opentsne"),
        ("UMAP transform", "s", "#009E73", "umap", "umap"),
        ("Parametric UMAP", "P", "#E69F00", "parametric_umap", None),
        ("kNN interp.", "^", "#A6761D", "knnmap", "knnmap"),
        ("PCA-2", "x", "#7a7a7a", "pca2", "pca2")]
# no error bars on row 2
ATLAS_REP0_ONLY = {"opentsne", "parametric_umap"}


def _points_bench(scores, ds):
    for lbl, mk, c, key, _ in ARMS:
        a = scores[ds].get(key)
        if a:
            yield (a["recall15_mean"], a["recall15_std"], a["cpd_mean"], a["cpd_std"],
                   lbl, mk, c)


def _points_atlas(atlas_res, pumap, ds):
    for lbl, mk, c, _, key in ARMS:
        # parametric_umap, from showcase json
        if key is None:
            p = pumap.get(ds)
            if p:
                yield (p["recall15"], 0.0, p["cpd"], 0.0, lbl, mk, c)
            continue
        cell = atlas_res.get(ds, {}).get(key, {}).get("mean_std", {})
        if cell.get("recall15"):
            r, rs = cell["recall15"]
            cp, cps = cell["cpd"]
            # single rep: no error bar
            if key in ATLAS_REP0_ONLY:
                rs = cps = 0.0
            yield (r, rs, cp, cps, lbl, mk, c)


def _panel(ax, points, title, xlabel, ylabel):
    for x, sx, y, sy, lbl, mk, c in points:
        ax.errorbar([x], [y], xerr=[sx], yerr=[sy], fmt=mk, ms=4.5, color=c,
                    elinewidth=0.6, capsize=1.4, zorder=4, label=lbl)
        print(f"    {lbl}: recall {x:.4f} +/- {sx:.4f}, cpd {y:.4f} +/- {sy:.4f}",
              flush=True)
    ax.set_title(title, fontsize=8)
    ax.set_box_aspect(1)
    ax.grid(True, lw=0.3, color="0.88", zorder=0)
    ax.set_axisbelow(True)
    for sp in ("top", "right"):
        ax.spines[sp].set_visible(False)
    if xlabel:
        ax.set_xlabel("held-out recall@15")
    if ylabel:
        ax.set_ylabel(r"held-out CPD ($\rho$ of pairwise distances)")


def main():
    scores = json.load(open(os.path.join(CACHE, "heldout_benchmark.json")))
    atlas_res = json.load(open(os.path.join(CACHE, "atlas_frontier.json")))
    pumap = {ds: json.load(open(os.path.join(CACHE, f"showcase_{ds}_pumap.json")))
             for ds, _ in ROW2}

    with plt.rc_context(viz.RC_PAPER):
        fig, axs = plt.subplots(2, 4, figsize=(2.05 * 4, 2.25 * 2))
        for i, (ds, title) in enumerate(ROW1):
            print(f"  {title}:", flush=True)
            _panel(axs[0][i], _points_bench(scores, ds), title,
                   xlabel=False, ylabel=i == 0)
        for i, (ds, title) in enumerate(ROW2):
            print(f"  {title}:", flush=True)
            _panel(axs[1][i], _points_atlas(atlas_res, pumap, ds), title,
                   xlabel=True, ylabel=i == 0)
        axs[1][3].axis("off")
        seen, handles, labels = set(), [], []
        for ax in axs.ravel():
            for h, l in zip(*ax.get_legend_handles_labels()):
                if l not in seen:
                    seen.add(l); handles.append(h); labels.append(l)
        axs[1][3].legend(handles, labels, loc="center", frameon=False, fontsize=7,
                         handletextpad=0.5, labelspacing=0.8, borderaxespad=0.0)
        fig.tight_layout(h_pad=0.9, w_pad=0.8)
        for ext in ("pdf", "png"):
            fig.savefig(f"{FIG}/heldout_opta.{ext}", dpi=300, bbox_inches="tight")
        plt.close(fig)
    print(f"saved {FIG}/heldout_opta.{{pdf,png}}", flush=True)


if __name__ == "__main__":
    main()
