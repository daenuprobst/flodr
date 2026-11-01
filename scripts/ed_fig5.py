import itertools
import os
import subprocess

import anndata as ad
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import phate
import umap
from flodr import FloDR
from openTSNE import TSNE
from scipy.sparse.csgraph import shortest_path
from scipy.stats import spearmanr
from sklearn.decomposition import PCA

import common as C

MAPS = ("FloDR (w=2)", "FloDR heat start", "FloDR (w=0)", "UMAP", "openTSNE", "PHATE")
MAPS += ("Parametric UMAP", "PCA-2", "PaCMAP", "DREAMS")
# controls, drawn hollow
HOLLOW = {
    "FloDR (w=0)": ("FloDR, $w=0$", "o"),
    "FloDR heat start": ("FloDR, heat start", "D"),
    "UMAP heat start": ("UMAP, heat start", "s"),
}
WAYPOINTS = dict(Cairo=(30.0, 31.2), Istanbul=(41.0, 28.9), Phnom_Penh=(11.5, 104.9))
WAYPOINTS.update(Anadyr=(64.7, 177.5), Prince_Rupert=(54.3, -130.3))
# regions joined over land, and the routes through waypoints (Ramachandran et al. 2005)
LINKS = [
    set(p.split("-"))
    for p in "MIDDLE_EAST-EUROPE MIDDLE_EAST-CENTRAL_SOUTH_ASIA EUROPE-CENTRAL_SOUTH_ASIA"
    " CENTRAL_SOUTH_ASIA-EAST_ASIA Cairo-AFRICA Cairo-MIDDLE_EAST Cairo-Istanbul"
    " Istanbul-MIDDLE_EAST Istanbul-EUROPE Phnom_Penh-EAST_ASIA Phnom_Penh-OCEANIA"
    " Anadyr-EAST_ASIA Anadyr-Prince_Rupert Prince_Rupert-AMERICA".split()
]


def spearman(a, b):
    return float(spearmanr(a, b).statistic)


def _spatial():
    # SEA-AD MTG MERFISH (Gabitto et al. 2024): glutamatergic neurons with a measured
    # depth from the pia, 180 genes, normalised per cell; scaling and PCA-50 fitted on
    # the reference donors, the others projected. Within each Alzheimer's stage, donors
    # sorted by ID, every third from the third is placed.
    a = ad.read_h5ad(os.path.join(C.DATA, "SEAAD_MTG_MERFISH.2024-12-11.h5ad"))
    o = a.obs
    neuron = o["Class"].to_numpy(str) == "Neuronal: Glutamatergic"
    keep = np.flatnonzero(np.isfinite(o["Depth from pia"].to_numpy()) & neuron)
    o, X = o.iloc[keep], np.asarray(a.X[keep], np.float64)
    X = np.log1p(X / np.maximum(X.sum(1, keepdims=True), 1) * 1e4)
    donor = o["Donor ID"].to_numpy(str)
    stage = o["Overall AD neuropathological Change"].to_numpy(str)
    placed = [d for s in set(stage) for d in sorted(set(donor[stage == s]))[2::3]]
    ref = ~np.isin(donor, placed)
    sd = X[ref].std(0, ddof=1)
    Z = np.minimum((X - X[ref].mean(0)) / np.where(sd == 0, 1, sd), 10)
    X = PCA(50, svd_solver="arpack", random_state=0).fit(Z[ref]).transform(Z)
    depth = o["Normalized depth from pia"].to_numpy(float)
    return dict(X=X.astype(np.float32), ref=ref, donor=donor, ndepth=depth)


def _popgen():
    # HGDP-CEPH whole genomes (Bergstrom et al. 2020), the PLINK 2 release
    # (cog-genomics.org/plink/2.0/resources): autosomal biallelic SNPs, MAF >= 0.05,
    # missingness <= 0.02, LD-pruned, one of each pair of second-degree relatives out
    raw, tmp = os.path.join(C.DATA, "hgdp"), os.path.join(C.CACHE, "ed_fig5", "hgdp")
    if not os.path.exists(os.path.join(tmp, "pruned.bed")):
        os.makedirs(tmp, exist_ok=True)
        plink = [os.environ.get("PLINK2", "plink2"), "--threads", "8"]
        with open(os.path.join(tmp, "all.pgen"), "wb") as f:
            zst = os.path.join(raw, "hgdp_all.pgen.zst")
            subprocess.run([*plink, "--zst-decompress", zst], stdout=f, check=True)
        for args in (
            f"--pgen all.pgen --pvar {raw}/hgdp_all_noannot.pvar.zst --psam "
            f"{raw}/hgdp.psam --autosome --snps-only just-acgt --max-alleles 2 "
            "--maf 0.05 --geno 0.02 --set-all-var-ids @:#:$r:$a "
            "--new-id-max-allele-len 10 truncate --make-pgen --out step1",
            "--pfile step1 --indep-pairwise 200 50 0.2 --out prune",
            "--pfile step1 --extract prune.prune.in --king-cutoff 0.0884 --out king",
            "--pfile step1 --extract prune.prune.in --keep king.king.cutoff.in.id "
            "--make-bed --out pruned",
        ):
            subprocess.run([*plink, *args.split()], cwd=tmp, check=True)
    iid = [ln.split()[1] for ln in open(os.path.join(tmp, "pruned.fam"))]
    b = np.fromfile(os.path.join(tmp, "pruned.bed"), np.uint8)[3:]
    b = b.reshape(-1, (len(iid) + 3) // 4)
    # two bits per individual, lowest first: two copies of A1, missing, one, none
    codes = np.stack([(b >> (2 * k)) & 3 for k in range(4)], 2).reshape(len(b), -1)
    G = np.array([2, np.nan, 1, 0], np.float32)[codes[:, : len(iid)]].T
    m = pd.read_csv(os.path.join(raw, "hgdp.psam"), sep="\t", index_col=0).loc[iid]
    pop, region = m["population"].to_numpy(str), m["region"].to_numpy(str)
    # held out: per region the population of median size, ties alphabetical
    held = []
    for r in sorted(set(region)):
        sizes = sorted((int((pop == p).sum()), p) for p in set(pop[region == r]))
        held.append(sizes[(len(sizes) - 1) // 2][1])
    ref = ~np.isin(pop, held)
    # standardised per SNP as in smartpca, frequencies of the reference, missing at 0
    p = np.nanmean(G[ref], 0) / 2
    keep = (p > 0) & (p < 1)
    Z = np.nan_to_num((G[:, keep] - 2 * p[keep]) / np.sqrt(2 * p[keep] * (1 - p[keep])))
    X = PCA(50, svd_solver="randomized", random_state=0).fit(Z[ref]).transform(Z)
    lat, lon = m["latitude"].to_numpy(float), m["longitude"].to_numpy(float)
    return dict(X=X, ref=ref, pop=pop, region=region, lat=lat, lon=lon)


def popgen_units(z, Y):
    # per population, the rank correlation of its map distances to the reference
    # populations with their great-circle distances along the routes; the layout over
    # the reference populations, the placement over the held-out individuals
    ref, pop = z["ref"], z["pop"]
    pops = sorted(set(pop))
    first = [np.flatnonzero(pop == p)[0] for p in pops]
    nodes = [(z["region"][i], (z["lat"][i], z["lon"][i])) for i in first]
    nodes += list(WAYPOINTS.items())
    A = np.zeros((len(nodes), len(nodes)))
    for i, (u, a) in enumerate(nodes):
        for j, (v, b) in enumerate(nodes[:i]):
            if u == v or {u, v} in LINKS:
                (la1, lo1), (la2, lo2) = np.radians(a), np.radians(b)
                h = np.sin((la2 - la1) / 2) ** 2
                h += np.cos(la1) * np.cos(la2) * np.sin((lo2 - lo1) / 2) ** 2
                A[i, j] = A[j, i] = 2 * 6371 * np.arcsin(np.sqrt(h))
    W = shortest_path(A, directed=False)
    rp = [i for i, p in enumerate(pops) if ref[pop == p].all()]
    cen = np.array([np.median(Y[ref & (pop == pops[i])], 0) for i in rp])
    d = np.linalg.norm(cen[:, None] - cen[None], axis=2)
    lay = [
        spearman(np.delete(d[k], k), np.delete(W[i, rp], k)) for k, i in enumerate(rp)
    ]
    place = []
    for i, p in enumerate(pops):
        y = Y[~ref & (pop == p)]
        if len(y) and np.isfinite(y).all():
            d = np.linalg.norm(y[:, None] - cen[None], axis=2)
            place.append(np.mean([spearman(r, W[i, rp]) for r in d]))
    return float(np.mean(lay)), float(np.mean(place)) if place else np.nan


def spatial_units(z, Y):
    # per donor, the rank correlation of the map distances between the medians of ten
    # bins of normalised depth (at least 20 neurons, at least four bins) with their
    # separation in depth; the reference donors, then the placed donors
    out = {True: [], False: []}
    for donor in sorted(set(z["donor"])):
        rows = np.flatnonzero((z["donor"] == donor) & np.isfinite(z["ndepth"]))
        b = np.minimum((np.clip(z["ndepth"][rows], 0, 1) * 10).astype(int), 9)
        bins = [k for k in range(10) if (b == k).sum() >= 20]
        if len(bins) >= 4 and np.isfinite(Y[rows]).all():
            cen = {k: np.median(Y[rows[b == k]], 0) for k in bins}
            ij = list(itertools.combinations(bins, 2))
            d = [np.linalg.norm(cen[i] - cen[j]) for i, j in ij]
            out[bool(z["ref"][rows[0]])].append(spearman(d, [j - i for i, j in ij]))
    return tuple(float(np.mean(v)) if v else np.nan for v in out.values())


def placed(m, X, ref, seed):
    # fitted on the reference, the held-out rows placed by the method's own step
    Xr, Xq = X[ref], X[~ref]
    if m.startswith("FloDR"):
        w, axes = (0.0 if "w=0" in m else 2.0), ("heat" if "heat" in m else "variance")
        est = FloDR(w=w, axes=axes, random_state=seed, verbose=False).fit(Xr)
        Yr, Yq = est.embedding_, est.transform_opt(Xq)
    elif m == "UMAP":
        est = umap.UMAP(n_neighbors=C.K, min_dist=0.1, random_state=seed, init="pca")
        Yr, Yq = est.fit(Xr).embedding_, est.transform(Xq)
    elif m == "openTSNE":
        Yr = TSNE(perplexity=30, random_state=seed, n_jobs=8).fit(Xr)
        Yq = Yr.transform(Xq)
    elif m == "PHATE":
        est = phate.PHATE(random_state=seed, verbose=0, n_jobs=8).fit(Xr)
        Yr, Yq = est.transform(), est.transform(Xq)
    else:
        Yr, Yq = C.pcumap(Xr, seed, Xq)
    Y = np.empty((len(X), 2), np.float32)
    Y[ref], Y[~ref] = Yr, Yq
    return Y


def domain(ds, make, units, seeds):
    # (recall@15 on the reference, agreement of the layout, of the placement) per map,
    # seed means; PaCMAP and DREAMS have no placement step, PCA-2 is the projection
    z = C.cached(os.path.join(C.CACHE, "ed_fig5", f"{ds}.npz"), make)
    X, ref = z["X"], z["ref"]

    def one(m, s):
        if m == "PCA-2":
            Y = X[:, :2].astype(np.float64)
        elif m in ("PaCMAP", "DREAMS"):
            Y = np.full((len(X), 2), np.nan)
            Y[ref] = C.fit(ds, m, s, X=X[ref])[0]
        else:
            Y = C.fit(ds, m, s, lambda X, s, _: placed(m, X, ref, s), X)[0]
        return [recall(ds, X[ref], Y[ref]), *units(z, Y)]

    path = os.path.join(C.CACHE, "ed_fig5", "scores.json")
    out = {
        m: np.mean([C.memo(path, f"{ds}/{m}/{s}", lambda: one(m, s)) for s in seeds], 0)
        for m in MAPS
    }
    return {**out, "input": (np.nan, *units(z, X.astype(np.float64)))}


_scorers = {}


def recall(ds, X, Y):
    if ds not in _scorers:
        _scorers[ds] = C.Scorer(X, "euclid", len(X) > 20_000)
    return _scorers[ds](Y)[0]


def popgen():
    return domain("popgen", _popgen, popgen_units, range(5))


def spatial():
    return domain("spatial", _spatial, spatial_units, range(3))


def panel(ax, pts, k, ylabel, title):
    # neighbours kept against agreement with the truth, the input as a dashed line
    for m, v in pts.items():
        if m == "input":
            ax.axhline(v[k], ls="--", lw=0.7, color="0.45", label="input, no map")
            continue
        label = "FloDR" if m.startswith("FloDR") else m.replace(" heat start", "")
        c, mk = C.METHOD_STYLE[label]
        kw = dict(color=c, ms=4.5, zorder=2)
        if m in HOLLOW:
            label, mk = HOLLOW[m]
            kw = dict(ms=5, mfc="none", mec=c, mew=1, zorder=3)
        elif label == "FloDR":
            kw.update(ms=6, zorder=4)
        ax.plot(*v[[0, k]], mk, label=label, **kw)
    ax.set(xlabel="recall@15", ylabel=ylabel)
    ax.set_title(title, fontsize=8)
    ax.spines[["top", "right"]].set_visible(False)


def legend(fig, axes, y):
    seen = {s: h for ax in axes for h, s in zip(*ax.get_legend_handles_labels())}
    kw = dict(loc="lower center", ncol=6, frameon=False, fontsize=6.5)
    fig.legend(seen.values(), seen.keys(), bbox_to_anchor=(0.5, y), **kw)


def main():
    plt.rcParams.update(C.RC)
    pts = spatial()
    fig, axs = plt.subplots(1, 2, figsize=(C.TEXT_W * 0.55, 2.35))
    panel(axs[0], pts, 1, "agreement with depth", "SEA-AD cortex, map")
    panel(axs[1], pts, 2, "agreement with depth", "placed donors")
    for ax, s in zip(axs, "ab"):
        C.letter(ax, s)
    fig.tight_layout(rect=(0, 0.2, 1, 1), w_pad=1.2)
    legend(fig, axs, -0.02)
    print({m: np.round(v, 3).tolist() for m, v in pts.items()})
    C.save(fig, "ed_fig5.pdf")


if __name__ == "__main__":
    main()
