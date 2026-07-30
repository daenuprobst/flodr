import ast
import json
import os
import re
import sys

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import LinearSegmentedColormap, Normalize
from scipy.spatial import cKDTree

matplotlib.use("Agg")

HERE = os.path.dirname(os.path.abspath(__file__))
SCRIPTS = os.path.dirname(HERE)
ROOT = os.path.dirname(SCRIPTS)
sys.path.insert(0, os.path.join(ROOT, "src"))
sys.path.insert(0, SCRIPTS)

from flodr import viz  # noqa: E402

CACHE = os.path.join(SCRIPTS, "cache")
FIG = os.path.join(ROOT, "figures")

# Sequential grey to deep red, for a magnitude with a meaningful zero. The knots are
# placed rather than evenly spaced, chroma stays near-neutral in the lower half and
# the saturated reds are held back past 0.8, so a mostly small field reads as mostly
# quiet. Lightness is strictly monotonic, so it survives greyscale. The floor is a
# definite grey rather than near-white, or a quiet field reads as no field at all.
GREY_RED = LinearSegmentedColormap.from_list("grey_red", [
    (0.00, "#dcdcdc"), (0.28, "#d4c6c0"), (0.50, "#ccae9f"),
    (0.70, "#c4886f"), (0.85, "#b64c33"), (0.94, "#8b190c"), (1.00, "#4d0300")])
CMAP = GREY_RED if os.environ.get("CMAP") is None else plt.get_cmap(os.environ["CMAP"])
OK, BAD = "#1a6b3a", "#b3242b"
SETS = [("bmarrow", "Human fetal bone marrow"),
        ("cerebellum", "Mouse developing cerebellum"),
        ("plant", "Arabidopsis germinating seed")]


def wash(ax, Y, v, norm, cmap, res=560):
    lo = Y.min(0) - 0.03 * np.ptp(Y, 0)
    hi = Y.max(0) + 0.03 * np.ptp(Y, 0)
    grid_x, grid_y = np.meshgrid(np.linspace(lo[0], hi[0], res),
                                 np.linspace(lo[1], hi[1], res))
    tree = cKDTree(Y)
    dist, nbr = tree.query(np.column_stack([grid_x.ravel(), grid_y.ravel()]), k=8)
    r_nn = float(np.median(tree.query(Y, k=16)[0][:, 15]))
    alpha = np.clip(1.0 - dist[:, 0].reshape(res, res) / (2.5 * r_nn), 0, 1) ** 1.2
    ax.imshow(v[nbr].mean(1).reshape(res, res), origin="lower",
              extent=(lo[0], hi[0], lo[1], hi[1]), cmap=cmap, norm=norm,
              alpha=alpha, interpolation="bilinear", zorder=1)
    # square the data window
    ctr, half = 0.5 * (lo + hi), 0.5 * float(np.max(hi - lo))
    ax.set_xlim(ctr[0] - half, ctr[0] + half)
    ax.set_ylim(ctr[1] - half, ctr[1] + half)
    ax.set_xticks([]); ax.set_yticks([]); ax.set_aspect("equal", adjustable="box")
    for spine in ax.spines.values():
        spine.set_color("0.75")


def parse_cert(v):
    text = str(v)
    try:
        return json.loads(text)
    except Exception:
        pass
    # np.True_ and np.False_ repr without parentheses, unlike the numeric scalars
    text = re.sub(r"np\.True_", "True", text)
    text = re.sub(r"np\.False_", "False", text)
    text = re.sub(r"np\.\w+\(([^()]*)\)", r"\1", text)
    return ast.literal_eval(text)


def note(ax, lines, color):
    ax.text(0.035, 0.022, "\n".join(lines), transform=ax.transAxes, fontsize=5.9,
            color=color, va="bottom", linespacing=1.35)


def main():
    have = []
    for ds, title in SETS:
        path = os.path.join(CACHE, f"hidden_contrast_{ds}.npz")
        if os.path.exists(path):
            npz = np.load(path, allow_pickle=True)
            if "sigma" not in npz.files:
                raise SystemExit(f"{path} predates the sigma pass -- rerun hidden_contrast.py")
            have.append((ds, title, npz))
    if not have:
        raise SystemExit("run scripts/ablations/hidden_contrast.py first")

    hid_max = max(float(np.percentile(npz["field"], 99)) for _, _, npz in have)
    sig_max = max(float(np.percentile(npz["sigma"], 99)) for _, _, npz in have)
    sig_min = min(float(np.percentile(npz["sigma"], 1)) for _, _, npz in have)
    # sigma shares the hidden contrast's ramp, so red means "more" in both rows. It has a
    # natural zero but never approaches it, so the ramp starts at the pooled 1st
    # percentile, since from zero it would waste its lower half and wash every panel pale.
    # Pooled bounds keep the three atlases comparable to each other.
    hid_norm = Normalize(0.0, hid_max)
    sig_norm = Normalize(sig_min, sig_max)

    with plt.rc_context(viz.RC_PAPER):
        n_atlas = len(have)
        n_row = 4
        panel_w, gap_x, gap_y, cbar_w, pad = 1.72, 0.09, 0.09, 0.062, 0.10
        fig_w = n_atlas * panel_w + (n_atlas - 1) * gap_x + pad + cbar_w + 0.52
        fig_h = n_row * panel_w + (n_row - 1) * gap_y + 0.28
        fig = plt.figure(figsize=(fig_w, fig_h))
        row_y = [(n_row - 1 - row) * (panel_w + gap_y) / fig_h for row in range(n_row)]

        for col, (ds, title, npz) in enumerate(have):
            emb = np.asarray(npz["Y"], float)
            hid_cert = parse_cert(npz["cert"])
            sig_cert = parse_cert(npz["sigma_cert"])
            left = col * (panel_w + gap_x) / fig_w

            # the map itself, for reference. tab20 matches the other scRNA figures, and
            # with no legend the hues mark grouping rather than identity
            ax = fig.add_axes([left, row_y[0], panel_w / fig_w, panel_w / fig_h])
            labs = np.asarray(npz["labels"]).astype(str)
            cats = np.unique(labs)
            palette = plt.get_cmap("tab20")(np.linspace(0, 1, 20))
            for cat_i, cat in enumerate(cats):
                mask = labs == cat
                ax.scatter(emb[mask, 0], emb[mask, 1], s=0.30, color=palette[cat_i % 20],
                           linewidths=0, alpha=0.75, rasterized=True)
            ctr = 0.5 * (emb.min(0) + emb.max(0))
            half = 0.53 * float(np.max(emb.max(0) - emb.min(0)))
            ax.set_xlim(ctr[0] - half, ctr[0] + half)
            ax.set_ylim(ctr[1] - half, ctr[1] + half)
            ax.set_xticks([]); ax.set_yticks([]); ax.set_aspect("equal", adjustable="box")
            for spine in ax.spines.values():
                spine.set_color("0.75")
            ax.set_title(title, fontsize=7.2, pad=3)
            note(ax, [f"{len(emb):,} cells, {len(cats)} classes"], "0.30")

            ax = fig.add_axes([left, row_y[1], panel_w / fig_w, panel_w / fig_h])
            wash(ax, emb, np.asarray(npz["sigma"], float), sig_norm, CMAP)
            note(ax, [f"median {np.median(npz['sigma']):.1f} input units",
                      (f"certified, magnitude {sig_cert.get('level')}$\\times$, "
                       f"conf {sig_cert.get('confidence'):.2f}") if sig_cert.get("passed") else
                      f"refused, magnitude {sig_cert.get('level')}$\\times$"],
                 OK if sig_cert.get("passed") else BAD)

            ax = fig.add_axes([left, row_y[2], panel_w / fig_w, panel_w / fig_h])
            wash(ax, emb, np.asarray(npz["field"], float), hid_norm, CMAP)
            # the field's mean, which is what the caption promises. mean_gap is the raw
            # validation-fold gap the certificate tests, a slightly different quantity
            note(ax, [f"{np.asarray(npz['field'], float).mean():.2f} nats hidden",
                      (f"certified, $p$ = {hid_cert.get('p_level'):.2f}" if hid_cert.get("passed")
                       else "refused")], OK if hid_cert.get("passed") else BAD)

            ax = fig.add_axes([left, row_y[3], panel_w / fig_w, panel_w / fig_h])
            wash(ax, emb, np.asarray(npz["field_null"], float), hid_norm, CMAP)
            note(ax, ["label shuffled within bins"], "0.30")

            if col == 0:
                for row, lbl in enumerate(("class", r"spread $\sigma(y)$",
                                           "hidden contrast", "null")):
                    fig.text(left - gap_x / fig_w, row_y[row] + 0.5 * panel_w / fig_h, lbl,
                             fontsize=7.0, rotation=90, va="center", ha="right")

        cbar_x = (n_atlas * panel_w + (n_atlas - 1) * gap_x + pad) / fig_w
        # both bars one panel tall. The spread bar sits beside its single row, and the
        # contrast bar serves rows 3 and 4 and is centred on the pair rather than
        # stretched, which would read as a taller scale carrying more range than it does
        for bot, norm, cmap, label in (
                (row_y[1], sig_norm, CMAP, r"$\sigma(y)$  (input units)"),
                (row_y[3] + 0.5 * (panel_w + gap_y) / fig_h, hid_norm, CMAP,
                 r"$I(G;R \mid Y{=}y)$  (nats)")):
            cax = fig.add_axes([cbar_x, bot, cbar_w / fig_w, panel_w / fig_h])
            cbar = fig.colorbar(plt.cm.ScalarMappable(norm=norm, cmap=cmap), cax=cax)
            cbar.set_label(label, fontsize=6.4)
            cbar.ax.tick_params(labelsize=5.8, length=2)
            cbar.outline.set_edgecolor("0.75")

        for ext in ("png", "pdf"):
            fig.savefig(os.path.join(FIG, f"diagnostics.{ext}"), dpi=300,
                        bbox_inches="tight", pad_inches=0.01)
        plt.close(fig)
    print("wrote figures/diagnostics.png / .pdf", flush=True)


if __name__ == "__main__":
    main()
