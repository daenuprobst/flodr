import json
import os
import sys

import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

matplotlib.use("Agg")

HERE = os.path.dirname(os.path.abspath(__file__))
SCRIPTS = os.path.dirname(HERE)
ROOT = os.path.dirname(SCRIPTS)
sys.path.insert(0, os.path.join(ROOT, "src"))

from flodr import viz  # noqa: E402

CACHE = os.path.join(SCRIPTS, "cache")
FIG = os.path.join(ROOT, "figures", "zebrafish_flodr")
CMAP = os.environ.get("CMAP", "turbo")
N_TYPES = int(os.environ.get("N_TYPES", "17"))
MAX_PTS = int(os.environ.get("MAX_PTS", "0"))
# larger than the HNOCA figure uses, because the same cell count spreads over many
# smaller islands here, so 0.05 leaves them washed out
PT = float(os.environ.get("PT", "0.13"))
OTHER = "other"
GREY = np.array([0.72, 0.72, 0.72, 1.0])
# W1=1 inserts the w=1 arm between w=2 and UMAP, which is where it lands on the frontier
W1 = os.environ.get("W1", "0") == "1"
ROWS = ((("flodr", "zebrafish_atlas.npz", "Y", "FloDR (w=2)"),
         ("flodr_w1", "zebrafish_atlas_w1.npz", "Y", "FloDR (w=1)"),
         ("umap", "zebrafish_umap.npz", "umap", "UMAP"))
        if W1 else
        (("flodr", "zebrafish_atlas.npz", "Y", "FloDR"),
         ("umap", "zebrafish_umap.npz", "umap", "UMAP")))


def frame(Y, pad=0.035):
    # the full extent, so no cell falls outside its panel
    lo, hi = Y.min(0), Y.max(0)
    margin = pad * float((hi - lo).max())
    return lo - margin, hi + margin


def fit_box(ax, lo, hi):
    ax.set_xlim(lo[0], hi[0])
    ax.set_ylim(lo[1], hi[1])
    # box takes the data's aspect, so equal units without wasted panel
    ax.set_box_aspect((hi[1] - lo[1]) / (hi[0] - lo[0]))
    ax.set_xticks([])
    ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_color("0.75")
        spine.set_linewidth(0.6)


def main():
    base = os.path.join(CACHE, ROWS[0][1])
    if not os.path.exists(base):
        raise SystemExit(f"no {base}; run scripts/zebrafish_atlas.py first")
    z0 = np.load(base, allow_pickle=True)
    kinds = np.asarray(z0["cell_type"]).astype(str)
    age = np.asarray(z0["timepoint"], np.float64)
    n_all = len(age)
    met = json.load(open(os.path.join(CACHE, "zebrafish_metrics.json"))) \
        if os.path.exists(os.path.join(CACHE, "zebrafish_metrics.json")) else {}

    # draw whatever rows are cached rather than refusing outright, so the FloDR panel is
    # available while a UMAP fit at this n is still running
    rows_spec, layouts = [], {}
    for key, fname, field, name in ROWS:
        path = os.path.join(CACHE, fname)
        if not os.path.exists(path):
            print(f"no {fname}, skipping the {name} row", flush=True)
            continue
        src = z0 if fname == ROWS[0][1] else np.load(path, allow_pickle=True)
        layouts[key] = np.asarray(src[field], np.float32)
        rows_spec.append((key, name))
    if not rows_spec:
        raise SystemExit("no cached layouts; run scripts/zebrafish_atlas.py first")

    # one shuffle shared by every panel, so a cell sits in the same draw position in all
    # four and no cell type ends up on top everywhere
    rng = np.random.default_rng(0)
    order = rng.permutation(n_all)
    if MAX_PTS and n_all > MAX_PTS:
        order = order[:MAX_PTS]
    kinds, age = kinds[order], age[order]
    layouts = {k: v[order] for k, v in layouts.items()}

    # one cell of 1.22M has no recorded timepoint, so nan-safe limits leave it undrawn in
    # the age panel rather than collapsing the colour scale
    n_nan = int(np.isnan(age).sum())
    age_lo, age_hi = float(np.nanmin(age)), float(np.nanmax(age))
    if n_nan:
        print(f"{n_nan} cell(s) without a timepoint, omitted from the age panel",
              flush=True)

    # 101 annotated types is far past what a legend can carry, so name the most abundant
    # and pool the tail into one neutral class rather than recycling colours
    cats, code = np.unique(kinds, return_inverse=True)
    counts = np.bincount(code, minlength=len(cats))
    top = np.argsort(counts)[::-1][:N_TYPES]
    pal = plt.get_cmap("tab20")(np.linspace(0, 1, 20))
    col = np.repeat(GREY[None], len(kinds), axis=0)
    named = []
    for rank, ci in enumerate(top):
        sel = code == ci
        col[sel] = pal[rank % 20]
        named.append((cats[ci], pal[rank % 20], int(counts[ci])))
    shown = int(counts[top].sum())
    print(f"{len(cats)} annotated types; naming {len(named)} covering "
          f"{100 * shown / len(kinds):.0f}% of cells", flush=True)

    n_leg_col = 3
    n_leg_row = int(np.ceil((len(named) + 1) / n_leg_col))

    with plt.rc_context(viz.RC):
        panel_in, gap_x, gap_y = 3.95, 0.14, 0.16
        left_in = 1.22
        suptitle_in, coltitle_in = 0.36, 0.30
        key_in = max(0.19 * n_leg_row + 0.16, 0.80)
        bot_in = 0.10
        fig_w = left_in + 2 * panel_in + gap_x
        n_row = len(rows_spec)
        fig_h = (suptitle_in + coltitle_in + n_row * panel_in
                 + (n_row - 1) * gap_y + key_in + bot_in)
        fig = plt.figure(figsize=(fig_w, fig_h))
        row_top = fig_h - suptitle_in - coltitle_in
        scat = None

        for r, (key, name) in enumerate(rows_spec):
            Y = layouts[key]
            lo, hi = frame(Y)
            y_bot = (row_top - (r + 1) * panel_in - r * gap_y) / fig_h

            for c in range(2):
                x0 = (left_in + c * (panel_in + gap_x)) / fig_w
                ax = fig.add_axes([x0, y_bot, panel_in / fig_w, panel_in / fig_h])
                if c == 0:
                    ax.scatter(Y[:, 0], Y[:, 1], s=PT, c=col, linewidths=0,
                               rasterized=True)
                else:
                    scat = ax.scatter(Y[:, 0], Y[:, 1], s=PT, c=age, cmap=CMAP,
                                      vmin=age_lo, vmax=age_hi, linewidths=0,
                                      rasterized=True)
                fit_box(ax, lo, hi)
                if r == 0:
                    ax.set_title(("cell type", "developmental stage")[c], fontsize=11,
                                 pad=5)

                if not ((Y >= lo) & (Y <= hi)).all():
                    raise AssertionError(f"{name}: cells outside the panel")

            cell = met.get("layouts", {}).get(key, {})
            lines = [name]
            if cell:
                lines += [f"recall@15  {cell['recall15']:.4f}",
                          f"CPD  {cell['cpd']:.4f}"]
            fig.text((left_in - 0.16) / fig_w, y_bot + 0.5 * panel_in / fig_h,
                     "\n".join(lines), fontsize=10.5, ha="right", va="center",
                     linespacing=1.9)

        handles = [Line2D([0], [0], marker="o", linestyle="", markersize=4.0,
                          markerfacecolor=c, markeredgewidth=0, label=lbl[:30])
                   for lbl, c, _ in named]
        handles.append(Line2D([0], [0], marker="o", linestyle="", markersize=4.0,
                              markerfacecolor=GREY, markeredgewidth=0,
                              label=f"{OTHER} ({len(cats) - len(named)} types)"))
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
        cbar.ax.tick_params(labelsize=7.5, length=2.4, pad=1.6)
        cbar.set_label("hours post-fertilisation", fontsize=8.5, labelpad=2.0)
        cbar.outline.set_edgecolor("0.75")
        cbar.outline.set_linewidth(0.6)
        cbar.solids.set_rasterized(True)

        fig.text(0.5, 1 - 0.5 * suptitle_in / fig_h,
                 f"Developmental Zebrafish Atlas, {n_all:,} cells, "
                 f"on a standardised PCA-50",
                 fontsize=12.5, ha="center", va="center")

        os.makedirs(os.path.dirname(FIG), exist_ok=True)
        stem = FIG + ("_w1" if W1 else "")
        for ext in ("png", "pdf"):
            fig.savefig(f"{stem}.{ext}", dpi=300, bbox_inches="tight")
        plt.close(fig)

    print(f"wrote {stem}.png / .pdf  ({len(age):,} of {n_all:,} cells drawn, "
          f"{age_lo:.0f}-{age_hi:.0f} hpf, all points inside every panel)",
          flush=True)


if __name__ == "__main__":
    main()
