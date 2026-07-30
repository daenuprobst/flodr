import json
import os
import sys

import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
from matplotlib.lines import Line2D

matplotlib.use("Agg")

HERE = os.path.dirname(os.path.abspath(__file__))
SCRIPTS = os.path.dirname(HERE)
ROOT = os.path.dirname(SCRIPTS)
sys.path.insert(0, os.path.join(ROOT, "src"))

from flodr import viz  # noqa: E402

CACHE_DIR = os.path.join(SCRIPTS, "cache")
METRICS = os.path.join(CACHE_DIR, "hnoca_metrics.json")
# "raw" pairs FloDR with the atlas's published UMAP, both given the latent as it ships.
# "standardised" pairs the two fits that were both given the standardised latent. Either
# way input and scoring space agree, which at d=10 they have to. see hnoca_metrics.py
PROTOCOL = os.environ.get("PROTOCOL", "raw")
# W1=1 adds the w=1 arm, which shows the ordinal dial between w=2 and UMAP
W1 = os.environ.get("W1", "0") == "1"
LAYOUT_SRC = {
    "raw": ((("flodr", "hnoca_atlas.npz", "Y", "FloDR (w=2)"),
             ("flodr_w1", "hnoca_atlas_w1.npz", "Y", "FloDR (w=1)"),
             ("umap_published", "hnoca_atlas.npz", "umap", "UMAP (published)"))
            if W1 else
            (("flodr", "hnoca_atlas.npz", "Y", "FloDR"),
             ("umap_published", "hnoca_atlas.npz", "umap", "UMAP (published)"))),
    "standardised": (("flodr_std", "hnoca_atlas_std.npz", "Y", "FloDR"),
                     ("umap_std", "hnoca_umap_std.npz", "umap_std", "UMAP")),
}[PROTOCOL]
# cell type, age and cell count come from the raw cache, which every run writes
CACHE = os.path.join(CACHE_DIR, "hnoca_atlas.npz")
FIG = os.path.join(ROOT, "figures", "hnoca_flodr")
SUFFIX = {"raw": "", "standardised": "_std"}
CMAP = os.environ.get("CMAP", "turbo")
MAX_PTS = int(os.environ.get("MAX_PTS", "0"))
# age spans 7 to 450 days with 90% of cells below 166, so a linear ramp would spend most
# of its range on the sparse tail
AGE_TICKS = (7, 15, 30, 60, 120, 250, 450)


def frame(Y, pad=0.035):
    # the full extent, not a percentile window, so no cell may fall outside its panel
    lo, hi = Y.min(0), Y.max(0)
    margin = pad * float((hi - lo).max())
    return lo - margin, hi + margin


def fit_box(ax, lo, hi):
    # the box takes the data's own aspect rather than a forced square, so a layout with
    # one long thin filament still fills its slot. Matching box aspect to limit ratio
    # keeps the units equal on both axes, so nothing is stretched.
    ax.set_xlim(lo[0], hi[0])
    ax.set_ylim(lo[1], hi[1])
    ax.set_box_aspect((hi[1] - lo[1]) / (hi[0] - lo[0]))
    ax.set_xticks([])
    ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_color("0.75")
        spine.set_linewidth(0.6)


def main():
    if not os.path.exists(CACHE):
        raise SystemExit(f"no {CACHE}; run scripts/hnoca_atlas.py first")

    z = np.load(CACHE, allow_pickle=True)
    age = np.asarray(z["age"], np.float64)
    kinds = np.asarray(z["cell_type"]).astype(str)
    n_all = len(age)
    met = json.load(open(METRICS)) if os.path.exists(METRICS) else {}

    rows_spec = []
    for key, fname, field, name in LAYOUT_SRC:
        path = os.path.join(CACHE_DIR, fname)
        if not os.path.exists(path):
            raise SystemExit(f"no {path}, needed for the {name} row of protocol "
                             f"'{PROTOCOL}'")
        src = z if fname == os.path.basename(CACHE) else np.load(path, allow_pickle=True)
        rows_spec.append((key, np.asarray(src[field], np.float32), name))

    # one shuffle shared by every panel, so a cell sits in the same draw position in all
    # four and no single protocol or cell type ends up on top everywhere
    rng = np.random.default_rng(0)
    order = rng.permutation(n_all)
    if MAX_PTS and n_all > MAX_PTS:
        order = order[:MAX_PTS]
    age, kinds = age[order], kinds[order]
    ROWS = tuple((key, name) for key, _, name in rows_spec)
    layouts = {key: Y[order] for key, Y, _ in rows_spec}

    cats, code = np.unique(kinds, return_inverse=True)
    # legend follows abundance, which is how the atlas papers order these
    by_size = np.argsort(np.bincount(code, minlength=len(cats)))[::-1]
    pal = plt.get_cmap("tab20")(np.linspace(0, 1, 20))
    rank = np.empty(len(cats), int)
    rank[by_size] = np.arange(len(cats))
    col = pal[rank[code] % 20]

    n_leg_col = 3
    n_leg_row = int(np.ceil(len(cats) / n_leg_col))
    norm = LogNorm(vmin=max(age.min(), 1.0), vmax=age.max())

    with plt.rc_context(viz.RC):
        # laid out in inches, so nothing needs a fudge factor to clear anything else
        panel_in, gap_x, gap_y = 3.95, 0.14, 0.16
        left_in = 1.22
        suptitle_in, coltitle_in = 0.36, 0.30
        key_in = max(0.19 * n_leg_row + 0.16, 0.80)
        bot_in = 0.10
        fig_w = left_in + 2 * panel_in + gap_x
        n_row = len(ROWS)
        fig_h = (suptitle_in + coltitle_in + n_row * panel_in
                 + (n_row - 1) * gap_y + key_in + bot_in)
        fig = plt.figure(figsize=(fig_w, fig_h))

        row_top = fig_h - suptitle_in - coltitle_in
        scat = None

        for r, (key, name) in enumerate(ROWS):
            Y = layouts[key]
            lo, hi = frame(Y)
            y_bot = (row_top - (r + 1) * panel_in - r * gap_y) / fig_h

            for c in range(2):
                x0 = (left_in + c * (panel_in + gap_x)) / fig_w
                ax = fig.add_axes([x0, y_bot, panel_in / fig_w, panel_in / fig_h])
                if c == 0:
                    ax.scatter(Y[:, 0], Y[:, 1], s=0.05, c=col, linewidths=0,
                               rasterized=True)
                else:
                    scat = ax.scatter(Y[:, 0], Y[:, 1], s=0.05, c=age, cmap=CMAP,
                                      norm=norm, linewidths=0, rasterized=True)
                fit_box(ax, lo, hi)
                if r == 0:
                    ax.set_title(("cell type", "organoid age")[c], fontsize=11, pad=5)

                inside = ((Y >= lo) & (Y <= hi)).all(1)
                if not inside.all():
                    raise AssertionError(
                        f"{name}: {int((~inside).sum())} cells outside the panel")

            # method name and its scores, right-aligned into the left margin. Scored in
            # the same space it was fit in, which is the protocol this figure draws
            cell = met.get("layouts", {}).get(key, {}).get("scores", {}).get(PROTOCOL)
            lines = [name]
            if cell:
                lines += [f"recall@15  {cell['recall15']:.4f}",
                          f"CPD  {cell['cpd']:.4f}"]
            fig.text((left_in - 0.16) / fig_w, y_bot + 0.5 * panel_in / fig_h,
                     "\n".join(lines), fontsize=10.5, ha="right", va="center",
                     linespacing=1.9)

        handles = [Line2D([0], [0], marker="o", linestyle="", markersize=4.0,
                          markerfacecolor=pal[i % 20], markeredgewidth=0,
                          label=cats[by_size[i]]) for i in range(len(cats))]
        leg_ax = fig.add_axes([left_in / fig_w, bot_in / fig_h, panel_in / fig_w,
                               key_in / fig_h])
        leg_ax.axis("off")
        leg_ax.legend(handles=handles, loc="upper center", ncol=n_leg_col, frameon=False,
                      fontsize=7.0, handletextpad=0.3, columnspacing=0.8,
                      labelspacing=0.30, borderaxespad=0.0)

        cbar_w_in, cbar_h_in = panel_in - 1.5, 0.10
        cax = fig.add_axes([
            (left_in + panel_in + gap_x + 0.5 * (panel_in - cbar_w_in)) / fig_w,
            (bot_in + key_in - 0.42) / fig_h, cbar_w_in / fig_w, cbar_h_in / fig_h])
        cbar = fig.colorbar(scat, cax=cax, orientation="horizontal")
        cbar.set_ticks(AGE_TICKS)
        cbar.set_ticklabels([str(t) for t in AGE_TICKS])
        cbar.ax.tick_params(labelsize=7.5, length=2.4, pad=1.6)
        cbar.set_label("days in culture", fontsize=8.5, labelpad=2.0)
        cbar.outline.set_edgecolor("0.75")
        cbar.outline.set_linewidth(0.6)
        cbar.solids.set_rasterized(True)

        fig.text(0.5, 1 - 0.5 * suptitle_in / fig_h,
                 f"Human Neural Organoid Cell Atlas, {n_all:,} cells, "
                 f"on the scPoli latent space",
                 fontsize=12.5, ha="center", va="center")

        stem = FIG + SUFFIX[PROTOCOL] + ("_w1" if W1 else "")
        os.makedirs(os.path.dirname(stem), exist_ok=True)
        for ext in ("png", "pdf"):
            fig.savefig(f"{stem}.{ext}", dpi=300, bbox_inches="tight")
        plt.close(fig)

    print(f"wrote {stem}.png / .pdf  (protocol '{PROTOCOL}', {len(age):,} of "
          f"{n_all:,} cells drawn, {len(cats)} cell types, "
          f"age {age.min():.0f}-{age.max():.0f} d, all points inside every panel)",
          flush=True)


if __name__ == "__main__":
    main()
