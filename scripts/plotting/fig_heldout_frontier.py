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
SCORES = os.path.join(SCRIPTS, "cache", "heldout_benchmark.json")
os.makedirs(FIG, exist_ok=True)

DS = [("mnist", "MNIST"), ("fmnist", "Fashion-MNIST"), ("paul15", "paul15"),
      ("drfp", "Schneider 50k")]
# (json key, legend label, w) in curve order, where w=2 is the benchmark's `flodr` arm
FLODR = [("flodr_w0", "FloDR", 0), ("flodr_w1", "FloDR", 1),
         ("flodr", "FloDR", 2), ("flodr_w3", "FloDR", 3)]
# Okabe-Ito derived, matching figures.py wherever a method appears in both
BASE = [("opentsne", "openTSNE transform", "o", "#0072B2"),
        ("umap", "UMAP transform", "s", "#009E73"),
        ("parametric_umap", "Parametric UMAP", "P", "#E69F00"),
        ("knnmap", "kNN interp.", "^", "#A6761D")]
FLO_C, REF_C = "#D55E00", "#7a7a7a"


def val(m, key):
    a = m.get(key)
    if not a:
        return None
    return (a["recall15_mean"], a["recall15_std"], a["cpd_mean"], a["cpd_std"])


def _panel(ax, m, title, ylabel=True):
    curve = [(w, val(m, key)) for key, _, w in FLODR]
    curve = [(w, v) for w, v in curve if v]
    if curve:
        xs = [v[0] for _, v in curve]
        ys = [v[2] for _, v in curve]
        ax.plot(xs, ys, "-", color=FLO_C, lw=1.0, zorder=3)
        ax.errorbar(xs, ys, xerr=[v[1] for _, v in curve],
                    yerr=[v[3] for _, v in curve], fmt="o", ms=4.0, color=FLO_C,
                    elinewidth=0.6, capsize=1.4, zorder=4, label="FloDR")
        for (w, _), x, y in zip(curve, xs, ys):
            ax.annotate(f"$w{{=}}{w}$", (x, y), textcoords="offset points",
                        xytext=(4, -9) if w == 0 else (4, 1),
                        fontsize=6.5, color=FLO_C, zorder=5)
        for (w, v) in curve:
            print(f"    flodr w={w}: recall {v[0]:.4f} +/- {v[1]:.4f}, "
                  f"cpd {v[2]:.4f} +/- {v[3]:.4f}", flush=True)
    for key, lbl, mk, c in BASE:
        v = val(m, key)
        if v is None:
            continue
        ax.errorbar([v[0]], [v[2]], xerr=[v[1]], yerr=[v[3]], fmt=mk, ms=4.5, color=c,
                    elinewidth=0.6, capsize=1.4, zorder=4, label=lbl)
        print(f"    {lbl}: recall {v[0]:.4f} +/- {v[1]:.4f}, "
              f"cpd {v[2]:.4f} +/- {v[3]:.4f}", flush=True)
    v = val(m, "pca2")
    if v is not None:
        ax.errorbar([v[0]], [v[2]], xerr=[v[1]], yerr=[v[3]], fmt="x", ms=4.5,
                    color=REF_C, elinewidth=0.6, capsize=1.4, zorder=4, label="PCA-2")
        print(f"    PCA-2: recall {v[0]:.4f} +/- {v[1]:.4f}, "
              f"cpd {v[2]:.4f} +/- {v[3]:.4f}", flush=True)
    ax.set_title(title, fontsize=8)
    ax.set_box_aspect(1)
    ax.grid(True, lw=0.3, color="0.88", zorder=0)
    ax.set_axisbelow(True)
    for sp in ("top", "right"):
        ax.spines[sp].set_visible(False)
    ax.set_xlabel("held-out recall@15")
    if ylabel:
        ax.set_ylabel(r"held-out CPD ($\rho$ of pairwise distances)")


def main():
    scores = json.load(open(SCORES))
    with plt.rc_context(viz.RC_PAPER):
        fig, axs = plt.subplots(1, len(DS), figsize=(2.05 * len(DS), 2.25))
        for i, (key, title) in enumerate(DS):
            if key not in scores:
                axs[i].axis("off")
                continue
            print(f"  {title}:", flush=True)
            _panel(axs[i], scores[key], title, ylabel=i == 0)
        seen, handles, labels = set(), [], []
        for ax in axs:
            for h, l in zip(*ax.get_legend_handles_labels()):
                if l not in seen:
                    seen.add(l); handles.append(h); labels.append(l)
        fig.tight_layout(h_pad=0.9, w_pad=0.8)
        fig.legend(handles, labels, loc="lower center", frameon=False,
                   ncol=len(labels), fontsize=6.4, handletextpad=0.35,
                   columnspacing=1.0, borderaxespad=0.0)
        fig.subplots_adjust(bottom=0.28)
        for ext in ("pdf", "png"):
            fig.savefig(f"{FIG}/heldout_frontier.{ext}", dpi=300, bbox_inches="tight")
        plt.close(fig)
    print(f"saved {FIG}/heldout_frontier.{{pdf,png}}", flush=True)


if __name__ == "__main__":
    main()
