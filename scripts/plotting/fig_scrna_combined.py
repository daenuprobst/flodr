"""scRNA showpiece: the two class-coloured maps, then three lenses (differentiation axis,
pairwise Shepard, R_NX) as one row per method. Reads the scrna_embed.py cache.

Run: [DATASET=bmarrow|cerebellum|plant] .venv/bin/python scripts/fig_scrna_combined.py
"""
import os
import sys
import warnings

warnings.filterwarnings("ignore")
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from scipy.stats import linregress, spearmanr

HERE = os.path.dirname(os.path.abspath(__file__))
SCRIPTS = os.path.dirname(HERE)
ROOT = os.path.dirname(SCRIPTS)
sys.path.insert(0, os.path.join(ROOT, "src"))
from flodr import viz  # noqa: E402

DATASET = os.environ.get("DATASET", "bmarrow")
TITLE = {"bmarrow": "Human fetal bone marrow", "cerebellum": "Mouse developing cerebellum",
         "plant": "Arabidopsis germinating seed"}[DATASET]
PROG_DEFAULT = {"bmarrow": "HSC/MPP and pro", "cerebellum": "cerebellar granule cell precursor",
                "plant": "Dividing"}
CACHE = os.path.join(SCRIPTS, "cache", "scrna", f"{DATASET}.npz")
FIG = os.path.join(ROOT, "figures", f"scrna_{DATASET}_combined")
NPAIR = 60000
NSUB = 6000                                  # cells for the O(n^2) co-ranking matrix
FIGW = 6.9                                   # A4 text width with narrow margins


def palette(ncat):
    if ncat <= 10:
        return plt.get_cmap("tab10")(np.arange(10))[:ncat]
    if ncat <= 20:
        return plt.get_cmap("tab20")(np.arange(20))[:ncat]
    base = plt.get_cmap("tab20")(np.linspace(0, 1, 20))
    reps = int(np.ceil(ncat / 20))
    pal = np.concatenate([base] * reps)[:ncat]
    pal[:, :3] = (pal[:, :3] + np.repeat(np.linspace(0, 0.12, reps), 20)[:ncat, None]) % 1.0
    return pal


def rect_lims(Y, pad=0.03):
    lo, hi = Y.min(0), Y.max(0)
    margin = (hi - lo) * pad
    return (lo[0] - margin[0], hi[0] + margin[0]), (lo[1] - margin[1], hi[1] + margin[1])


def rnx_curve(Xs, Ys):
    """R_NX(k) from the co-ranking matrix (Lee & Verleysen); 1 is perfect at that scale,
    0 is random. Subsampled -- the matrix is O(n^2). AUC is log-weighted, as published."""
    def ranks(A):
        dist = np.sqrt(((A[:, None, :] - A[None, :, :]) ** 2).sum(-1))
        rank = np.empty(dist.shape, np.int32)
        n_pts = len(A)
        rank[np.arange(n_pts)[:, None], np.argsort(dist, axis=1)] = np.arange(n_pts)[None, :]
        return rank
    n = len(Xs)
    rank_max = np.maximum(ranks(Xs), ranks(Ys)).ravel()  # pairs enter at max(rank_hi, rank_lo)
    cum_pairs = np.cumsum(np.bincount(rank_max, minlength=n)[1:n])  # both ranks <= k
    ks = np.arange(1, n - 1)
    qnx = cum_pairs[: n - 2] / (n * ks)
    rnx = ((n - 1) * qnx - ks) / (n - 1 - ks)
    weight = 1.0 / ks
    return ks, rnx, float((rnx * weight).sum() / weight.sum())


def rnx_panel(ax, ks, own, other, own_name, other_name, own_auc, other_auc, color,
              top=None):
    """One method's R_NX curve, the other faint behind it. Drawn to k = n/2: the (n-1-k)
    denominator makes the last decade unstable."""
    half = len(ks) // 2
    ks, own, other = ks[:half], own[:half], other[:half]
    ax.semilogx(ks, other, lw=0.8, color="0.62", ls="--", zorder=2,
                label=f"{other_name} ({other_auc:.3f})")
    ax.semilogx(ks, own, lw=1.4, color=color, zorder=3, label=f"{own_name} ({own_auc:.3f})")
    ax.set_ylim(0, max(own.max(), other.max()) * 1.12)
    ax.set_xlim(1, ks[-1])
    ax.set_box_aspect(1)
    ax.grid(True, lw=0.3, alpha=0.35)
    ax.set_axisbelow(True)
    ax.tick_params(labelsize=6)
    ax.set_ylabel(r"$R_{NX}(k)$", fontsize=6.5, labelpad=1.5)
    ax.legend(fontsize=5.6, frameon=False, loc="upper right", handlelength=1.4,
              handletextpad=0.4, labelspacing=0.25, borderaxespad=0.2)
    if top:
        ax.set_title(top, fontsize=7)
    for sp in ("top", "right"):
        ax.spines[sp].set_visible(False)


def corr_panel(ax, x, y, kind, xlabel, ylabel, top=None):
    """Shepard panel, axes normalised to the 99.5th percentile so both methods share a
    scale. Dotted line is identity, solid is the fit."""
    xn, yn = x / np.percentile(x, 99.5), y / np.percentile(y, 99.5)
    fit = linregress(xn, yn)
    rho = spearmanr(x, y).statistic
    lim = 1.08
    if kind == "hex":
        ax.hexbin(xn, yn, gridsize=40, bins="log", cmap="viridis", mincnt=1, linewidths=0,
                  extent=(0, lim, 0, lim), rasterized=True)
    else:
        ax.scatter(xn, yn, s=7, c="#3b6ea5", alpha=0.85, edgecolors="white", linewidths=0.2)
    ax.plot([0, lim], [0, lim], color="0.55", lw=0.7, ls=":", zorder=1)
    x_fit = np.linspace(0, lim, 50)
    ax.plot(x_fit, fit.intercept + fit.slope * x_fit, color="crimson", lw=1.1, zorder=3)
    ax.set_xlim(0, lim); ax.set_ylim(0, lim); ax.set_box_aspect(1)
    ax.set_xticks(np.arange(0, 1.01, 0.5)); ax.set_yticks(np.arange(0, 1.01, 0.5))
    ax.tick_params(labelsize=6)
    ax.set_xlabel(xlabel, fontsize=6.5, labelpad=1.5)
    ax.set_ylabel(ylabel, fontsize=6.5, labelpad=1.5)
    # rho sits inside: a per-row title would collide once the rows are packed tight
    ax.text(0.04, 0.965, rf"$\rho={rho:.2f}$", transform=ax.transAxes, fontsize=7,
            va="top", ha="left",
            bbox=dict(boxstyle="round,pad=0.16", fc="white", ec="none", alpha=0.75))
    if top:
        ax.set_title(top, fontsize=7)
    for sp in ("top", "right"):
        ax.spines[sp].set_visible(False)


def main():
    npz = np.load(CACHE, allow_pickle=True)
    pcs, y_flodr, y_umap, labels = (npz["P"].astype(np.float32), npz["flodr"], npz["umap"],
                                    npz["labels"])
    n = len(pcs)
    cats, lab_idx = np.unique(labels, return_inverse=True)
    ncat = len(cats)
    order = np.argsort(np.bincount(lab_idx))[::-1]
    pal = palette(ncat)
    centers = np.stack([pcs[labels == c].mean(0) for c in cats])

    prog = os.environ.get("PROGENITOR") or PROG_DEFAULT.get(DATASET)
    if prog not in set(cats):
        from scipy.spatial.distance import cdist
        prog = cats[cdist(centers, centers).mean(1).argmin()]
    axis = np.linalg.norm(pcs - centers[list(cats).index(prog)], axis=1)
    vlo, vhi = np.percentile(axis, [2, 98])

    rng = np.random.default_rng(0)
    pair_i, pair_j = rng.integers(0, n, NPAIR), rng.integers(0, n, NPAIR)
    distinct = pair_i != pair_j
    pair_i, pair_j = pair_i[distinct], pair_j[distinct]
    dist_true = np.linalg.norm(pcs[pair_i] - pcs[pair_j], axis=1)
    tri = np.triu_indices(ncat, 1)
    cent_dist = np.linalg.norm(centers[tri[0]] - centers[tri[1]], axis=1)
    shuf = rng.permutation(n)
    # cached: the co-ranking matrix is the only expensive part here. Both methods see the
    # same cells.
    rnx_path = os.path.join(os.path.dirname(CACHE), f"{DATASET}_rnx.npz")
    if os.path.exists(rnx_path):
        cached = np.load(rnx_path)
        ks, rnx = cached["ks"], {"FloDR": cached["flodr"], "UMAP": cached["umap"]}
        auc = {"FloDR": float(cached["auc_flodr"]), "UMAP": float(cached["auc_umap"])}
    else:
        sub = rng.permutation(n)[:NSUB]
        ks, rnx_f, auc_f = rnx_curve(pcs[sub], y_flodr[sub])
        _, rnx_u, auc_u = rnx_curve(pcs[sub], y_umap[sub])
        rnx, auc = {"FloDR": rnx_f, "UMAP": rnx_u}, {"FloDR": auc_f, "UMAP": auc_u}
        np.savez_compressed(rnx_path, ks=ks, flodr=rnx_f, umap=rnx_u,
                            auc_flodr=auc_f, auc_umap=auc_u)
    print(f"  AUC R_NX: FloDR {auc['FloDR']:.3f}  UMAP {auc['UMAP']:.3f}", flush=True)

    # cap the columns rather than the rows: with 22 cell types a 3-row legend needs 8
    # columns and widens the whole figure past the text block
    ncol = min(6, ncat)
    nrows_leg = int(np.ceil(ncat / ncol))
    left, right, gapx = 0.075, 0.985, 0.012
    wspace, hspace = 0.34, 0.02
    # laid out in inches so no square panel floats inside an oversized slot
    map_w_in = (right - left - gapx) / 2 * FIGW
    map_h_in = 0.82 * map_w_in
    panel_in = (right - left) * FIGW / (3 + 2 * wspace)          # square analytics panel
    legend_in = 0.15 * nrows_leg + 0.04
    top_in, bot_in = 0.30, 0.06                                  # suptitle / x-label
    gap_above_leg, gap_below_leg = 0.05, 0.34   # legend hugs its maps, sits clear of the row
    analytics_in = 2 * panel_in + hspace * panel_in + 0.40       # + row titles and labels
    fig_h = (top_in + map_h_in + gap_above_leg + legend_in + gap_below_leg
             + analytics_in + bot_in)
    with plt.rc_context(viz.RC_PAPER):
        fig = plt.figure(figsize=(FIGW, fig_h))
        fig.suptitle(f"{TITLE} ({n:,} cells)", fontsize=9.5, y=1 - 0.06 / fig_h)
        y_top = 1 - top_in / fig_h
        map_w = (right - left - gapx) / 2
        map_h = map_h_in / fig_h
        for col, (emb, name) in enumerate(((y_flodr, "FloDR"), (y_umap, "UMAP"))):
            ax = fig.add_axes([left + col * (map_w + gapx), y_top - map_h, map_w, map_h])
            ax.scatter(emb[shuf, 0], emb[shuf, 1], c=pal[lab_idx][shuf], s=0.7, alpha=0.8,
                       linewidths=0, rasterized=True)
            (x0, x1), (y0, y1) = rect_lims(emb)
            ax.set_xlim(x0, x1); ax.set_ylim(y0, y1)
            ax.set_aspect("equal", adjustable="datalim")
            ax.set_xticks([]); ax.set_yticks([])
            ax.set_title(name, fontsize=9, fontweight="bold", pad=2.5)
            for spine in ax.spines.values():
                spine.set_visible(True); spine.set_color("0.45"); spine.set_linewidth(0.6)

        leg_band = legend_in / fig_h
        leg_y = y_top - map_h - gap_above_leg / fig_h - leg_band
        leg_ax = fig.add_axes([left, leg_y, right - left, leg_band])
        leg_ax.axis("off")
        handles = [Line2D([0], [0], marker="o", linestyle="", markersize=3.4,
                          markerfacecolor=pal[c], markeredgewidth=0, label=cats[c][:26])
                   for c in order]
        leg_ax.legend(handles=handles, loc="center", ncol=ncol, fontsize=6.2, frameon=False,
                      handletextpad=0.25, columnspacing=0.9, labelspacing=0.35)

        a_top = leg_y - gap_below_leg / fig_h
        gs = fig.add_gridspec(2, 3, left=left, right=right, top=a_top,
                              bottom=bot_in / fig_h, hspace=hspace, wspace=wspace)
        for row, (emb, name) in enumerate(((y_flodr, "FloDR"), (y_umap, "UMAP"))):
            ax_map = fig.add_subplot(gs[row, 0])
            scat = ax_map.scatter(emb[shuf, 0], emb[shuf, 1], c=axis[shuf], s=0.6, cmap="turbo",
                                  vmin=vlo, vmax=vhi, linewidths=0, rasterized=True)
            (x0, x1), (y0, y1) = rect_lims(emb)
            ax_map.set_xlim(x0, x1); ax_map.set_ylim(y0, y1)
            ax_map.set_aspect("equal", adjustable="datalim")
            ax_map.set_box_aspect(1)
            ax_map.set_xticks([]); ax_map.set_yticks([])
            ax_map.set_ylabel(name, fontsize=8, fontweight="bold", labelpad=2)
            if row == 0:
                ax_map.set_title(f"differentiation axis\n(dist. from {prog[:22]})", fontsize=7)
            cax = ax_map.inset_axes([1.02, 0.0, 0.035, 1.0])
            cbar = fig.colorbar(scat, cax=cax)
            cbar.set_ticks([vlo, vhi])
            cbar.set_ticklabels(["prog.", "mature"])
            cax.tick_params(length=0, labelsize=5.5)
            cbar.outline.set_linewidth(0.3)

            dist_emb = np.linalg.norm(emb[pair_i] - emb[pair_j], axis=1)
            ax_shep = fig.add_subplot(gs[row, 1])
            corr_panel(ax_shep, dist_true, dist_emb, "hex", "true distance (PCA-50)",
                       "embedded distance", top="pairwise distances" if row == 0 else None)
            ax_rnx = fig.add_subplot(gs[row, 2])
            rnx_panel(ax_rnx, ks, rnx[name], rnx["UMAP" if name == "FloDR" else "FloDR"],
                      name, "UMAP" if name == "FloDR" else "FloDR",
                      auc[name], auc["UMAP" if name == "FloDR" else "FloDR"],
                      "#D55E00" if name == "FloDR" else "#0072B2",
                      top="neighbourhood preservation" if row == 0 else None)
            ax_rnx.set_xlabel("neighbourhood size $k$", fontsize=6.5, labelpad=1.5)
            if row == 0:                     # shared x with the row below: drop duplicate labels
                for ax in (ax_shep, ax_rnx):
                    ax.set_xlabel("")
                    ax.set_xticklabels([])
        for ext in ("png", "pdf"):
            fig.savefig(f"{FIG}.{ext}", dpi=300, bbox_inches="tight")
    print(f"-> wrote {FIG}.png / .pdf  (n={n:,}, progenitor={prog})", flush=True)


if __name__ == "__main__":
    main()
