import glob
import os

import anndata as ad
import h5py
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import common as C
from fig5 import inputs, lanes, lineage, p2, pca50, readout

KS = (2, 3, 5, 10, 20, 50)
SETS = {"LARRY": "larry", "CellTag LSK": "celltag_lsk"}
SETS.update({f"CellTag iEP {r}": f"celltag_iep_{r}" for r in ("r1", "r2")})
SUBSETS = ["all cells", "cells grown from LSK", "cells grown from LK"]
SUBSETS += ["without Baso, Mast, Meg, Ery"]


def _iep(rep):
    # CellTag-multi fibroblast-to-iEP reprogramming (Jindal et al. 2023), one replicate:
    # GEO GSE216518 lanes of days 3, 12 and 21, the authors' QC cells (Zenodo 22715159)
    # and clones (github morris-lab/CellTag-multi-2023, clone_tables)
    raw = os.path.join(C.DATA, "celltagmulti")
    with h5py.File(os.path.join(raw, "b4_full_coembed.h5ad")) as f:
        obs = ad.io.read_elem(f["obs"])
    qc = obs.index[
        (obs["assay"] == "RNA") & obs["day"].isin(["B4D3", "B4D12", "B4D21"])
    ]
    lane = lambda p: os.path.basename(p).split("_")[1]
    paths = glob.glob(os.path.join(raw, f"GSM*_B4D*-RNA-{rep}-*_feature_bc_matrix.h5"))
    day = lambda p: float(lane(p).split("-")[0][3:])
    a = lanes(sorted(paths), lambda p, b: f"{lane(p)}-{b}", qc, day)
    t = pd.read_csv(
        os.path.join(raw, "b4.rna_atac.r1_r2_master_rev1_annotated.github.csv")
    )
    late = t[t["day"].isin(["D12", "D21"])]
    t = t[t["assay"] == "RNA"].set_index("cell.bc").reindex(a.obs_names)
    clone = pd.factorize(t["clone_id"])[0]
    # day-12 and day-21 siblings per clone, RNA and ATAC cells together
    sib = lambda f: late.loc[late["cell_type"] == f, "clone_id"].value_counts()
    n = lambda f: sib(f).reindex(t["clone_id"]).fillna(0).to_numpy(int) * (clone >= 0)
    y = t["cell_type"].fillna("unlabelled").to_numpy(str)
    return dict(
        X=pca50(a),
        y=y,
        day=a.obs["day"].to_numpy(),
        clone=clone,
        sib_reprogramming=n("reprogramming"),
        sib_dead_end=n("dead-end"),
    )


def main():
    plt.rcParams.update(C.RC)
    kw = dict(width_ratios=(1.4, 1), wspace=0.55)
    fig, (a, b) = plt.subplots(1, 2, figsize=(C.TEXT_W, 2.3), gridspec_kw=kw)
    # a, the progenitor-fate AUROC of the first k principal components and of the FloDR
    # maps (seed means)
    edge = dict(edgecolor="k", lw=0.6)
    for (name, ds), c in zip(SETS.items(), (C.OURS, C.PARTNER, "#332288", "#999933")):
        iep = os.path.join(C.CACHE, "ed_fig4", f"{ds}.npz")
        z = C.cached(iep, lambda: _iep(ds[-2:])) if "iep" in ds else inputs(ds)
        r = readout(ds, z)
        pcs = [p2(z["X"][:, :k], r) for k in KS]
        pts = lineage(
            ds, z, ("FloDR (w=2)", "FloDR heat start"), range(5 if ds == "larry" else 3)
        )
        maps = pts["FloDR (w=2)"][1], pts["FloDR heat start"][1]
        a.plot(KS, pcs, "-o", ms=2.5, lw=1, color=c, label=name)
        a.scatter([1.6], [maps[0]], s=14, marker="o", color=c, **edge)
        a.scatter([1.8], [maps[1]], s=16, marker="D", color=c, **edge)
        print(name, np.round(pcs, 3).tolist(), np.round(maps, 3).tolist())
    a.scatter([], [], s=14, marker="o", color="w", label="FloDR map", **edge)
    a.scatter([], [], s=16, marker="D", color="w", label="FloDR heat map", **edge)
    a.set_xscale("log", base=2)
    a.set(xticks=KS, xticklabels=KS, xlim=(1.5, 64))
    a.set_xlabel("principal components of the input", fontsize=7)
    a.set_ylabel("progenitor fate (AUROC)", fontsize=7)
    kw = dict(fontsize=6.5, frameon=False, handlelength=1.6, columnspacing=1.0)
    a.legend(loc="lower right", ncol=2, **kw)
    # b, LARRY with its first two axes refitted on subsets of cells, all cells projected
    z = inputs("larry")
    r, X, y, start = readout("larry", z), z["X"].astype(np.float64), z["y"], z["start"]
    blood = ~np.isin(y, ["Baso", "Mast", "Meg", "Erythroid"])
    comp = []
    for m in (
        np.ones_like(blood),
        start == "Lin-Kit+Sca1+",
        start == "Lin-Kit+Sca1-",
        blood,
    ):
        axes = np.linalg.svd(X[m] - X[m].mean(0), full_matrices=False)[2][:2]
        comp.append(p2((X - X[m].mean(0)) @ axes.T, r))
    print("LARRY, first two axes", dict(zip(SUBSETS, np.round(comp, 3).tolist())))
    b.barh(range(4), comp, color=C.OURS, height=0.6)
    b.set(yticks=range(4), xlim=(0.3, 0.8))
    b.set_yticklabels(SUBSETS, fontsize=7)
    b.invert_yaxis()
    b.set_xlabel("LARRY, AUROC on the first two axes", fontsize=7)
    a.axhline(0.5, color="0.6", lw=0.5, ls=(0, (1, 1.5)))
    b.axvline(0.5, color="0.6", lw=0.5, ls=(0, (1, 1.5)))
    for ax, s, dx in ((a, "a", -6), (b, "b", -62)):
        ax.tick_params(labelsize=7, length=2)
        ax.spines[["top", "right"]].set_visible(False)
        C.letter(ax, s, dx)
    C.save(fig, "ed_fig4.pdf")


if __name__ == "__main__":
    main()
