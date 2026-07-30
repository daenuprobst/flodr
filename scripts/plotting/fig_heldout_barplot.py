import json

import matplotlib.pyplot as plt
import numpy as np

DATA = "scripts/cache/heldout_benchmark.json"
OUT = "figures/heldout_barplot"

ORDER = [
    ("flodr", "base", "#D55E00"),
    ("flodr_w0", "w=0", "#E69F00"),
    ("flodr_w1", "w=1", "#E69F00"),
    ("flodr_w3", "w=3", "#E69F00"),
    ("flodr_cons0.5", "cons.5", "#999999"),
    ("flodr_cons2", "cons2", "#999999"),
    ("flodr_chart0.5", "chart.5", "#999999"),
    ("flodr_chart1", "chart1", "#999999"),
    ("flodr_val", "val", "#BBBBBB"),
    ("flodr_val0", "val0", "#BBBBBB"),
    ("flodr_mix", "mix", "#BBBBBB"),
    ("flodr_skhead", "skhead", "#F0E442"),
    ("knnmap_flodr", "knnF", "#56B4E9"),
    ("otmap0.25db", "OT", "#0072B2"),
    ("flodr_opt", "opt", "#CC79A7"),
    ("flodr_optA", "optA", "#CC79A7"),
    ("optA_long", "optA500", "#CC79A7"),
    ("skhead_opt", "sk+opt", "#882255"),
    ("skhead_optA", "sk+optA", "#882255"),
    ("pca2", "PCA-2", "#666666"),
    ("knnmap", "knnU", "#A6761D"),
    ("parametric_umap", "pUMAP", "#E69F00"),
    ("umap", "UMAP", "#009E73"),
    ("opentsne", "openTSNE", "#0072B2"),
]
DATASETS = [("mnist", "MNIST"), ("fmnist", "Fashion-MNIST"), ("paul15", "paul15"), ("drfp", "Schneider 50k")]
METRICS = [("recall15", "held-out recall@15"), ("cpd", "held-out CPD")]

res = json.load(open(DATA))
plt.rcParams.update({"font.size": 7, "axes.linewidth": 0.6})
fig, axes = plt.subplots(2, 4, figsize=(13, 3.6), sharey="row")
for col, (ds, title) in enumerate(DATASETS):
    for row, (mk, mlabel) in enumerate(METRICS):
        ax = axes[row, col]
        vals = [(lab, res[ds][key][f"{mk}_mean"], res[ds][key][f"{mk}_std"], c)
                for key, lab, c in ORDER if key in res[ds]]
        xs = np.arange(len(vals))
        ax.bar(xs, [v[1] for v in vals], yerr=[v[2] for v in vals],
               color=[v[3] for v in vals], error_kw=dict(lw=0.6, capsize=1),
               width=0.75, edgecolor="none")
        ax.axvline(len(ORDER) - 5.5, color="k", lw=0.5, ls=":")
        if row == 0:
            ax.set_title(title)
        if col == 0:
            ax.set_ylabel(mlabel)
        ax.set_xticks(xs)
        ax.set_xticklabels([v[0] for v in vals], rotation=90)
        ax.spines[["top", "right"]].set_visible(False)
        ax.grid(axis="y", lw=0.3, alpha=0.4)
        ax.set_axisbelow(True)
fig.tight_layout()
for ext in ("pdf", "png"):
    fig.savefig(f"{OUT}.{ext}", dpi=300)
print("wrote", OUT)
