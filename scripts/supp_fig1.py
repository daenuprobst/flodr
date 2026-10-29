import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from flodr import viz
from scipy.ndimage import gaussian_filter1d
from scipy.spatial import cKDTree

import common
from ed_fig6 import model

N_PATH, N_DRAWS, K_NEAR, ALPHA = 160, 256, 30, 0.1
# a path runs through the medoids of its stages on the training map, and genes are read
# along it; Paul et al. number the erythroid clusters from the most mature (1) to the least
ERY = ["7MEP", "6Ery", "5Ery", "4Ery", "3Ery", "2Ery", "1Ery"]
NEU = [("9GMP", "10GMP"), "16Neu"]
HSC = ["HSC/MPP", "MEP", "early erythroid", "mid erythroid", "late erythroid"]
PATHS = {
    "paul15": [
        ("MEP to erythroid", ERY, "Gata1 Klf1 Hba-a2 Pf4 Itga2b".split()),
        ("GMP to neutrophil", NEU, "Mpo Elane Cebpe Irf8 Csf1r".split()),
    ],
    "bmarrow": [("HSC/MPP to erythroid", HSC, "BLVRB PRDX2 FECH PRSS57".split())],
}
GENES = {ds: sum((p[2] for p in paths), []) for ds, paths in PATHS.items()}


def inputs(ds):
    # cells, broad types, the labels the paths run through, and a linear readout of the
    # genes' log expression from the cells: the PCA loadings on paul15, least squares on
    # the training cells on bone marrow, where two of the genes are not highly variable
    if ds == "paul15":
        P, clusters, g = common.paul15_pcs()
        c = [g["genes"].index(x) for x in GENES[ds]]
        types = np.char.lstrip(clusters, "0123456789")
        return P, types, clusters, g["W"][:, c], g["b"][c]
    import scanpy as sc

    P, types = common.load(ds)
    a = sc.read_h5ad(f"{common.DATA}/{common.ATLAS_FILES[ds][0]}").raw.to_adata()
    sc.pp.normalize_total(a, target_sum=1e4)
    sc.pp.log1p(a)
    cols = [list(a.var["feature_name"]).index(x) for x in GENES[ds]]
    tr = common.split(len(P))[0]
    A = np.c_[np.asarray(P, np.float64)[tr], np.ones(len(tr))]
    expr = a.X[:, cols].toarray()[tr].astype(np.float64)
    coef = np.linalg.lstsq(A, expr, rcond=None)[0]
    fine = a.obs["cell.labels"].astype(str).to_numpy()
    return P, types.astype(str), fine, coef[:-1], coef[-1]


def stage_path(Y, nodes, smooth=2.0):
    # straight segments between the medoids at equal arc length, corners rounded
    def resample(Q):
        seg = np.r_[0.0, np.cumsum(np.sqrt((np.diff(Q, axis=0) ** 2).sum(1)))]
        t = np.linspace(0, seg[-1], N_PATH)
        return np.stack([np.interp(t, seg, Q[:, j]) for j in range(2)], 1)

    P = resample(Y[nodes].astype(np.float64))
    if len(nodes) > 2:
        ends = np.minimum(np.arange(N_PATH), N_PATH - 1 - np.arange(N_PATH))
        w = np.clip(ends / (3 * smooth), 0, 1)[:, None]
        P = resample(w * gaussian_filter1d(P, smooth, 0, mode="nearest") + (1 - w) * P)
    return P.astype(np.float32)


def decoded(ds):
    # along each path: the 5th to 95th percentile of exact draws from p(x | y) inside
    # the 90% conformal set, the mean of all draws and of the 30 nearest training cells;
    # and coverage on held-out cells, cross-fitted on two halves
    def make():
        from flodr.conformal import FibreConformal, sample_fibre

        print(f"decode {ds}", flush=True)
        P, types, groups, W, b = inputs(ds)
        est, (tr, te) = model(ds, 0), common.split(len(P))
        vals = P.astype(np.float64) @ W + b
        Y, tree = est.embedding_, cKDTree(est.embedding_)
        far = np.percentile(tree.query(Y, k=2)[0][:, 1], 99)
        cp = FibreConformal(est, ALPHA).fit(P[te])
        out = dict(Y=Y, types=types[tr])
        for p, (_, stages, _) in enumerate(PATHS[ds]):
            # through the cell of each stage nearest its median position
            idx = [np.flatnonzero(np.isin(groups[tr], s)) for s in stages]
            nodes = [
                i[np.argmin(((Y[i] - np.median(Y[i], 0)) ** 2).sum(1))] for i in idx
            ]
            Yp = stage_path(Y, nodes)
            bd = cp.bands(Yp, n_draws=N_DRAWS, seed=0, readout=lambda x: x @ W + b)
            xs = sample_fibre(est, Yp, n_draws=N_DRAWS, seed=1)[0]
            near = tree.query(Yp, k=K_NEAR)[1]
            # spread of the draws over that of the nearest cells, per PC, in the input
            spread = xs.std(0) / (P[tr][near].std(1) + 1e-9)
            out |= {
                f"p{p}_path": Yp,
                f"p{p}_far": tree.query(Yp)[0] > far,
                f"p{p}_lo": bd["lo"],
                f"p{p}_hi": bd["hi"],
                f"p{p}_mean": (xs.astype(np.float64) @ W + b).mean(0),
                f"p{p}_near_mean": vals[tr][near].mean(1),
                f"p{p}_spread": np.median(spread.mean(1)),
            }
        halves = np.array_split(np.random.default_rng(1).permutation(te), 2)
        res = []
        for cal, ev in ((halves[0], halves[1]), (halves[1], halves[0])):
            cpf = FibreConformal(est, ALPHA).fit(P[cal])
            y = est.transform(P[ev])
            bd = cpf.bands(y, n_draws=N_DRAWS, seed=2, readout=lambda x: x @ W + b)
            near = vals[tr][tree.query(y, k=K_NEAR)[1]]
            lo, hi = np.percentile(near, [5, 95], axis=1)
            v = vals[ev]
            res.append(
                ((v >= bd["lo"]) & (v <= bd["hi"]), (v >= lo) & (v <= hi))
                + (bd["hi"] - bd["lo"], hi - lo, cpf.contains(P[ev])[:, None])
            )
        cov, cov_near, width, width_near, joint = map(np.concatenate, zip(*res))
        sd = vals[te].std(0)
        out |= dict(cov=cov.mean(0), cov_near=cov_near.mean(0), joint=joint.mean())
        out |= dict(width=np.nanmean(width, 0) / sd, width_near=width_near.mean(0) / sd)
        return out

    return common.cached(os.path.join(common.CACHE, "supp_fig1", f"{ds}.npz"), make)


plt.rcParams.update({**common.RC, "font.size": 7, "axes.linewidth": 0.5})
for k, v in zip(
    ("labelsize", "major.size", "major.width", "major.pad"), (6.5, 2, 0.4, 1)
):
    plt.rcParams[f"xtick.{k}"] = plt.rcParams[f"ytick.{k}"] = v
W, H = common.TEXT_W, 5.25
fig = plt.figure(figsize=(W, H))


def axes(x, y, w, h):
    return fig.add_axes([x / W, y / H, w / W, h / H])


rows = []
for ds, y0, side, x1, width, ends, crop in (
    ("paul15", 2.7, 2.3, 2.95, 1.86, ("MEP", "Ery", "GMP", "Neu"), (0.5, 99.5)),
    ("bmarrow", 0.32, 2.1, 2.72, 1.75, ("HSC/MPP and pro", "erythroid"), (1, 99)),
):
    # the map with its paths at (0.12, y0) and the gene strips from x1, in inches
    d = decoded(ds)
    pal = common.class_colours(d["types"])[1]
    short = {"HSC/MPP and pro": "HSC/MPP", "erythroid": "late ery."}
    named = [(short.get(t, t), pal[t]) for t in ends]
    paths = [d[f"p{p}_path"] for p in range(len(PATHS[ds]))]
    far = [d[f"p{p}_far"] for p in range(len(PATHS[ds]))]
    on, ax = np.isin(d["types"], ends), axes(0.12, y0, side, side)
    named = [named[:2], named[2:]]
    viz.plot_paths(d["Y"], paths, d["types"], pal, named, far, on, ax, crop)
    common.letter(ax, "ac"[ds == "bmarrow"])
    top = y0 + side - 0.08
    for p, (name, _, genes) in enumerate(PATHS[ds]):
        x = x1 + p * (width + 0.42)
        fig.text((x + width / 2) / W, (top + 0.02) / H, name, ha="center", va="bottom")
        print(f"{name}: draws spread {d[f'p{p}_spread']:.1f} times the nearest 30")
        for r, g in enumerate(genes):
            j = GENES[ds].index(g)
            lo, hi, mean, near = (
                d[f"p{p}_{k}"][:, j] for k in ("lo", "hi", "mean", "near_mean")
            )
            ax = axes(x, top - (r + 1) * 0.39 - r * 0.085, width, 0.39)
            viz.plot_band(lo, hi, mean, near, d[f"p{p}_far"], r"\textit{" + g + "}", ax)
            ymin, ymax = min(np.nanmin(lo), near.min()), max(np.nanmax(hi), near.max())
            ax.set_yticks([np.ceil(ymin * 2) / 2, np.floor(ymax * 2) / 2])
            if r < len(genes) - 1:
                ax.set_xticks([])
            if p == r == 0:
                common.letter(ax, "bd"[ds == "bmarrow"], dx=-14)
    for j, g in enumerate(GENES[ds]):
        c, cn, w, wn = (d[k][j] for k in ("cov", "cov_near", "width", "width_near"))
        print(
            f"{g}: band holds {c:.3f} at {w:.2f} sd, {w / wn:.1f} times the 5-95% "
            f"range of the nearest 30, which holds {cn:.3f}"
        )
        rows.append((r"\textit{" + g + "}", c, cn, w, wn))
    print(f"{ds}: the joint 90% set holds {float(d['joint']):.3f}")
    rows.append(("joint set", float(d["joint"]), np.nan, np.nan, np.nan))

# e: held-out coverage and width of the gene bands and of the nearest training cells
n = len(GENES["paul15"]) + 1
yy = np.arange(len(rows))[::-1] - 0.6 * (np.arange(len(rows)) >= n)
ax_c, ax_w = axes(5.3, 0.54, 0.86, 1.8), axes(6.38, 0.54, 0.7, 1.8)
for ax, (ours, near) in ((ax_c, (1, 2)), (ax_w, (3, 4))):
    ax.scatter([r[near] for r in rows], yy, s=9, c="#2E8B7A", lw=0, zorder=3)
    ax.scatter([r[ours] for r in rows], yy, s=9, c=common.OURS, lw=0, zorder=4)
    ax.set(ylim=(yy.min() - 0.7, yy.max() + 0.7), yticks=yy, yticklabels=[])
    ax.spines[["top", "right"]].set_visible(False)
    ax.axhline(yy[n - 1] - 0.8, c="0.8", lw=0.4)
ax_c.axvline(1 - ALPHA, c="0.5", lw=0.5, ls=(0, (2, 2)), zorder=0)
ax_c.set_yticklabels([r[0] for r in rows], fontsize=6.5)
ax_c.set(xlim=(0.78, 1.0), xticks=[0.8, 0.9, 1.0])
ax_c.set_xlabel("held-out coverage", labelpad=1)
ax_w.set_xlim(0, None)
ax_w.xaxis.set_major_locator(matplotlib.ticker.MaxNLocator(3, integer=True))
ax_w.set_xlabel("width / s.d.", labelpad=1)
common.letter(ax_c, "e", dx=-40)
common.save(fig, "supp_fig1.pdf")
