import matplotlib.pyplot as plt
from flodr.viz import plot_embedding

from common import RC, TEXT_W, class_colours, fit, letter, load, metric, save, scores

ROWS = [
    ("mnist", "MNIST", "raw pixels", "Euclidean"),
    ("fmnist", "Fashion-MNIST", "raw pixels", "Euclidean"),
    ("paul15", "paul15 (scRNA)", r"HVG-1000$\rightarrow$PCA-50", "Euclidean"),
    ("drfp", "Schneider 50k", "reaction fingerprints, 2048 bits", "Jaccard"),
]
TITLE = {
    "FloDR (w=2)": "FloDR (ours)\n$w$=2",
    "FloDR (w=0)": "FloDR (ours)\nno ordinal",
    "PCUMAP": "Parametric UMAP",
}


def draw(methods, name):
    plt.rcParams.update(RC)
    cols = len(methods)
    fig, axes = plt.subplots(4, cols, figsize=(TEXT_W, TEXT_W / cols * 4 + 0.5))
    for r, (ds, title, representation, dist) in enumerate(ROWS):
        X, y = load(ds)
        colours = class_colours(y)[1]
        # seven narrow columns leave no room for the representation
        head = [title, f"$n$={len(X)}, $d$={X.shape[1]}"]
        head += [representation, dist] if cols <= 5 else [dist]
        axes[r, 0].set_ylabel(
            "\n".join(head), fontsize=6.5, labelpad=5, linespacing=1.25
        )
        letter(axes[r, 0], "abcd"[r])
        for ax, method in zip(axes[r], methods):
            # the map is seed 0, recall and CPD the means over three seeds
            rec, cpd = scores(ds, method)[:, :2].mean(0)
            text = f"recall@15 {rec:.3f}\nCPD {cpd:.3f}"
            if metric(ds, method) != metric(ds, "FloDR"):
                text += "\nEuclidean-only"
            Y = fit(ds, method, 0)[0]
            plot_embedding(Y, y, ax=ax, colours=colours, crop=(0.5, 99.5), text=text)
            if r == 0:
                ax.set_title(TITLE.get(method, method), fontsize=8, loc="left")
    fig.tight_layout(w_pad=0.4, h_pad=0.5)
    save(fig, name)


if __name__ == "__main__":
    draw(["FloDR (w=2)", "FloDR (w=0)", "UMAP", "openTSNE", "PCA-2"], "fig2.pdf")
