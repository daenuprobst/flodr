import glob
import gzip
import os

import anndata as ad
import h5py
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scanpy as sc
import umap
from flodr import FloDR
from flodr.viz import frame
from matplotlib.colors import to_rgb
from scipy.io import mmread
from scipy.stats import gaussian_kde, rankdata
from sklearn.metrics import roc_auc_score

import common as C
from ed_fig5 import legend, panel, popgen, recall

LSK_MAPS = ("FloDR (w=2)", "UMAP", "openTSNE", "PaCMAP", "densMAP", "DREAMS", "PHATE")
LSK_MAPS += ("PCA-2",)
LARRY_MAPS = LSK_MAPS + ("FloDR (w=0)", "FloDR heat start", "UMAP heat start")
HERE = os.path.join(C.CACHE, "fig5")


def lanes(paths, name, qc, day):
    # CellTag-multi lane matrices, cells named as the authors' and kept if they passed
    # their QC, without the construct transcripts that mark clones
    parts = []
    for p in paths:
        a = sc.read_10x_h5(p)
        a.var_names_make_unique()
        a.obs_names = [name(p, b) for b in a.obs_names]
        a = a[a.obs_names.isin(qc)].copy()
        a.obs["day"] = day(p)
        parts.append(a)
    a = ad.concat(parts, join="inner", merge="same")
    g = a.var_names
    drop = g.isin(["CellTag.UTR", "GFP.CDS", "FoxA1.HNF4a"]) | g.str.startswith("mt-")
    return a[:, ~drop].copy()


def pca50(a):
    sc.pp.normalize_total(a, target_sum=1e4)
    sc.pp.log1p(a)
    sc.pp.highly_variable_genes(a, flavor="seurat", n_top_genes=2000)
    a = a[:, a.var["highly_variable"]].copy()
    sc.pp.scale(a, max_value=10)
    sc.tl.pca(a, n_comps=50, svd_solver="arpack", random_state=0)
    return np.ascontiguousarray(a.obsm["X_pca"], np.float32)


def _lsk():
    # CellTag-multi, LSK arm (Jindal et al. 2023): GEO GSE216606 lanes, the authors' QC
    # cells and labels (figshare 26367055) and clones (github
    # morris-lab/CellTag-multi-2023, clone_tables)
    raw = os.path.join(C.DATA, "celltagmulti_lsk")
    with h5py.File(os.path.join(raw, "lsk_full_mult_coembed.h5ad")) as f:
        obs = ad.io.read_elem(f["obs"])
    rna, atac = obs[obs["tech"] == "rna"], obs[obs["tech"] == "atac"]

    def name(p, b):
        # day-5 lanes 1-4 are replicate 1, 5-8 replicate 2, suffixed by the lane in it
        k, b = int(os.path.basename(p).split("_")[2]), b.split("-")[0]
        if "_d2_5_" in p:
            return f"d2_5-rna-{b}-1"
        return f"d5r{1 + (k > 4)}-rna-{b}-{(k - 1) % 4 + 1}"

    paths = sorted(glob.glob(os.path.join(raw, "GSM*_d*_feature_bc_matrix.h5")))
    a = lanes(paths, name, rna.index, lambda p: 2.5 if "_d2_5_" in p else 5.0)
    t = pd.read_csv(os.path.join(raw, "hsc.rna_atac.r1_r2_master_v2.github.csv"))
    label = pd.concat([rna["cell_type2"], atac["predictedGroup_Un"]]).astype(str)
    t = t[t["cell.bc"].isin(label.index)]
    late = t[t["day"] == "d5"]
    # day-5 neutrophil and monocyte siblings per clone, RNA and ATAC cells together
    sib = pd.crosstab(late["clone.id"], label.reindex(late["cell.bc"]).to_numpy())
    clone = t[t["assay"] == "rna"].set_index("cell.bc")["clone.id"]
    clone = clone.reindex(a.obs_names).fillna(-1).to_numpy(np.int64)
    sib = sib.reindex(index=clone, columns=["Neutro", "Mono"], fill_value=0).to_numpy()
    return dict(
        X=pca50(a),
        y=rna["cell_type2"].reindex(a.obs_names).to_numpy(str),
        day=a.obs["day"].to_numpy(),
        clone=clone,
        sib=sib * (clone >= 0)[:, None],
        sib_fates=np.array(["Neutro", "Mono"]),
    )


def _larry():
    # LARRY in vitro (Weinreb et al. 2020), kleintools.hms.harvard.edu/paper_websites
    raw = lambda part: os.path.join(C.DATA, "larry", f"stateFate_inVitro_{part}")
    meta = pd.read_csv(raw("metadata.txt.gz"), sep="\t", keep_default_na=False)
    a = ad.AnnData(mmread(raw("normed_counts.mtx.gz")).tocsr().astype(np.float32))
    a.var_names = [g.strip() for g in gzip.open(raw("gene_names.txt.gz"), "rt")]
    clones = mmread(raw("clone_matrix.mtx.gz")).tocsr()
    clone = np.where(clones.getnnz(1) > 0, np.asarray(clones.argmax(1)).ravel(), -1)
    return dict(
        X=pca50(a),
        y=meta["Cell type annotation"].to_numpy(str),
        day=meta["Time point"].to_numpy(float),
        clone=clone,
        start=meta["Starting population"].to_numpy(str),
    )


def inputs(ds):
    make = {"celltag_lsk": _lsk, "larry": _larry}[ds]
    return C.cached(os.path.join(HERE, f"{ds}.npz"), make)


def readout(ds, z):
    # the barcoded progenitors, whether most of their clone's later siblings took the
    # first fate, and the cells of the two fates
    y, day, bar = z["y"].astype(str), z["day"], z["clone"] >= 0
    if ds == "larry":
        late = day > 2
        n = lambda t: np.bincount(z["clone"][late & bar & (y == t)], minlength=len(y))
        sa, sb, need = n("Neutrophil")[z["clone"]], n("Monocyte")[z["clone"]], 3
        early = (day == 2) & bar & (y == "Undifferentiated")
        a, b = late & (y == "Neutrophil"), late & (y == "Monocyte")
    elif ds == "celltag_lsk":
        f = list(z["sib_fates"].astype(str))
        sa, sb, need = z["sib"][:, f.index("Neutro")], z["sib"][:, f.index("Mono")], 1
        early = (day == 2.5) & bar & np.isin(y, ("MPP", "MPP/GMP", "MEP"))
        a, b = (day == 5) & (y == "Neutro"), (day == 5) & (y == "Mono")
    else:
        sa, sb, need = z["sib_reprogramming"], z["sib_dead_end"], 1
        early, late = (day == 3) & bar, (day == 21) & bar
        a, b = late & (y == "reprogramming"), late & (y == "dead-end")
    prog = np.flatnonzero(early & (sa != sb) & (sa + sb >= need))
    return prog, (sa > sb)[prog], a, b


def fate_score(Y, prog, a, b):
    # how much nearer the median of the first fate than that of the second
    Y = np.asarray(Y, np.float64)
    da = np.linalg.norm(Y[prog] - np.median(Y[a], 0), axis=1)
    return np.linalg.norm(Y[prog] - np.median(Y[b], 0), axis=1) - da


def p2(Y, r):
    return float(roc_auc_score(r[1], fate_score(Y, r[0], r[2], r[3])))


def lineage(ds, z, maps, seeds):
    # (recall@15, progenitor-fate AUROC) per map, seed means, and the input
    def heat(X, s, _):
        return FloDR(axes="heat", random_state=s, verbose=False).fit(X).embedding_

    def umap_heat(X, s, _):
        # from FloDR(axes="heat")'s start, the first two columns of its rotated head
        kw = dict(axes="heat", max_iter=2, density=False, verbose=False, random_state=s)
        start = lambda: dict(Y=FloDR(**kw).fit(X)._coords[:, :2])
        Y0 = C.cached(os.path.join(HERE, f"heat_start_{ds}_s{s}.npz"), start)["Y"]
        est = umap.UMAP(n_neighbors=C.K, min_dist=0.1, random_state=s, init=Y0)
        return est.fit_transform(X)

    def densmap(X, s, _):
        kw = dict(densmap=True, dens_lambda=2.0, init="pca", random_state=s)
        return umap.UMAP(n_neighbors=C.K, min_dist=0.1, **kw).fit_transform(X)

    fits = {**C.FIT, "densMAP": densmap, "FloDR heat start": heat}
    fits["UMAP heat start"] = umap_heat
    r = readout(ds, z)

    def one(m, s):
        Y = C.fit(ds, m, s, fits[m], z["X"])[0]
        return [recall(ds, z["X"], Y), p2(Y, r)]

    path = os.path.join(HERE, "scores.json")
    out = {
        m: np.mean([C.memo(path, f"{ds}/{m}/{s}", lambda: one(m, s)) for s in seeds], 0)
        for m in maps
    }
    return {**out, "input": (np.nan, p2(z["X"], r))}


def points():
    lsk = lineage("celltag_lsk", inputs("celltag_lsk"), LSK_MAPS, range(3))
    return lsk, lineage("larry", inputs("larry"), LARRY_MAPS, range(5)), popgen()


def light(c, f):
    return (1 - f) * np.asarray(to_rgb(c)) + f


def plot_fate_map(Y, a, b, prog, fate, ax, colours):
    # the map turned so that the median of fate a lies left of that of fate b, framed on
    # the two fates with every progenitor in view; light, the cells of each fate; dots,
    # the progenitors by the fate their clone took; crosses, the fate medians
    ca, cb = colours
    t = -np.arctan2(*(np.median(Y[b], 0) - np.median(Y[a], 0))[::-1])
    Y = Y @ np.array([[np.cos(t), np.sin(t)], [-np.sin(t), np.cos(t)]])
    cp, *ends = (np.median(Y[m], 0) for m in (prog, a, b))
    for m, c, f in ((~(a | b), "#dcdcdc", 0), (b, cb, 0.72), (a, ca, 0.45)):
        ax.scatter(*Y[m].T, s=0.3, color=light(c, f), lw=0, rasterized=True)
    for e, c in zip(ends, colours):
        ax.plot(*np.c_[cp, e], color="0.3", lw=0.6, zorder=3)
        ax.scatter(*e, s=30, marker="X", color=c, edgecolor="k", lw=0.6, zorder=6)
    ax.scatter(*Y[prog[~fate]].T, s=4, color=cb, lw=0, zorder=4)
    ax.scatter(*Y[prog[fate]].T, s=4, color=ca, lw=0, zorder=5)
    box = np.percentile(Y[a | b], [1, 99], axis=0)
    frame(ax, np.vstack([box, Y[prog]]), crop=(0, 100), pad=0.05)
    return cp, ends


def plot_fate_ranks(score, fate, ax, colours):
    # smoothed densities of the progenitors' ranks by fate_score, per fate, those
    # relatively nearer fate a to the left
    x, grid = 1 - (rankdata(score) - 0.5) / len(score), np.linspace(0, 1, 200)
    da, db = gaussian_kde(x[fate], 0.25)(grid), gaussian_kde(x[~fate], 0.25)(grid)
    ax.fill_between(grid, 0, da, color=light(colours[0], 0.55), lw=0)
    ax.plot(grid, da, color=colours[0], lw=0.9)
    ax.plot(grid, db, color=colours[1], lw=0.9)
    ax.set(xlim=(0, 1), ylim=(0, max(da.max(), db.max()) * 1.32), xticks=[], yticks=[])
    ax.spines[["top", "right", "left"]].set_visible(False)
    ax.spines["bottom"].set_linewidth(0.5)


def main():
    plt.rcParams.update(C.RC)
    lsk, pts = inputs("celltag_lsk"), points()
    W, H = C.TEXT_W, 5.1
    fig = plt.figure(figsize=(W, H))
    at = lambda x, y, w, h: fig.add_axes([x / W, y / H, w / W, h / H])
    # a-d, every map at its seed mean against the measurement
    top = [at(0.52 + 1.78 * k, 3.62, 1.24, 1.32) for k in range(4)]
    clones, geo = "progenitor fate (AUROC)", "agreement with geography"
    panel(top[0], pts[0], 1, clones, "CellTag-multi LSK, clones")
    panel(top[1], pts[1], 1, clones, "LARRY, clones")
    panel(top[2], pts[2], 1, geo, "HGDP genomes, map")
    panel(top[3], pts[2], 2, geo, "HGDP, placed populations")
    for ax, s in zip(top, "abcd"):
        C.letter(ax, s)
    legend(fig, top, 2.72 / H)
    # e-h, the LSK arm at seed 0
    prog, fate, a, b = readout("celltag_lsk", lsk)
    col, fates = (C.PARTNER, C.OURS), ("neutrophils", "monocytes")
    for k, m in reversed(list(enumerate(("FloDR (w=2)", "openTSNE", "UMAP", "PCA-2")))):
        Y, x0 = C.fit("celltag_lsk", m, 0)[0].astype(float), 0.2 + k * 1.77
        ax, sx = at(x0, 0.97, 1.6, 1.6), at(x0, 0.48, 1.6, 0.42)
        cp, ends = plot_fate_map(Y, a, b, prog, fate, ax, col)
        plot_fate_ranks(fate_score(Y, prog, a, b), fate, sx, col)
        s = (r"\textbf{%s}" if k == 0 else "%s") % f"AUROC {pts[0][m][1]:.2f}"
        sx.text(0.98, 0.97, s, transform=sx.transAxes, ha="right", va="top", size=6.5)
        ax.set_title(r"\textbf{FloDR}" if k == 0 else m, fontsize=7.5)
        C.letter(ax, "efgh"[k])
    # the last drawn is FloDR's, which names the groups
    kw = dict(textcoords="offset points", ha="center", fontsize=6, zorder=7)
    kw["bbox"] = dict(boxstyle="round,pad=0.12", fc="white", ec="none")
    ax.annotate("progenitors, day 2.5", cp, xytext=(0, 12), color="0.15", **kw)
    h, counts = [], (fate.sum(), (~fate).sum())
    for i, (e, c, s, n) in enumerate(zip(ends, col, fates, counts)):
        ax.annotate(s, e, xytext=(0, -10), va="top", color=0.6 * light(c, 0), **kw)
        t = sx.text(i, -0.06, "relatively nearer\n" + s, transform=sx.transAxes)
        t.set(ha=("left", "right")[i], va="top", fontsize=6, color="0.3", linespacing=1)
        s = f"progenitor, siblings mostly {s} ({n})"
        h.append(plt.Line2D([], [], ls="", marker="o", ms=3.5, color=c, label=s))
    s, x = "median of the day-5 cells of each fate", dict(mec="k", mew=0.6, ls="")
    h.append(plt.Line2D([], [], color="0.55", marker="X", ms=5, label=s, **x))
    kw = dict(handletextpad=0.2, columnspacing=1.4, frameon=False, fontsize=6.5)
    fig.legend(handles=h, loc="lower center", ncol=3, bbox_to_anchor=(0.5, 0), **kw)
    for name, p in zip(("LSK", "LARRY", "HGDP"), pts):
        print(name, {m: np.round(v, 3).tolist() for m, v in p.items()})
    C.save(fig, "fig5.pdf")


if __name__ == "__main__":
    main()
