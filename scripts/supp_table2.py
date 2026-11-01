import functools
import os
import subprocess
import zlib

import numpy as np
import torch
from scipy import stats
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import dijkstra
from scipy.stats import rankdata

import common
from common import K

DIR = os.path.join(common.CACHE, "supp_table2")
SCORES = os.path.join(DIR, "scores.json")
# E2's zebrafish latent: the 10-D scVI model on 250,000 cells, in an interpreter with
# scvi-tools, trained on the GPU
SCVI_PYTHON = os.environ.get(
    "SCVI_PYTHON", os.path.join(common.ROOT, ".venv-scvi", "bin", "python")
)
_SCVI = """
import sys, anndata as ad, numpy as np, scanpy as sc, scvi
h5, out = sys.argv[1:]
scvi.settings.seed = 0
a = ad.read_h5ad(h5)
rows = np.sort(np.random.default_rng(0).permutation(a.n_obs)[:250_000])
a = a[rows].copy()
sc.pp.highly_variable_genes(a, n_top_genes=2000, flavor="seurat_v3", subset=True)
scvi.model.SCVI.setup_anndata(a, batch_key="expt")
m = scvi.model.SCVI(a, n_latent=10)
kw = dict(max_epochs=None, early_stopping=True, plan_kwargs={"lr": 1e-3})
m.train(accelerator="gpu", **kw)
lat10 = np.asarray(m.get_latent_representation(), np.float32)
ctype = np.asarray(a.obs["cell_type_broad"]).astype(str)
np.savez_compressed(out, lat10=lat10, ctype=ctype, rows=rows)
"""
SUBSETS = ("paul15", "bmarrow20k", "mnist10k")
MAPS = ("flodr", "mlp")
PLACE = {"forward", "opt", "interp", "umap"}


def data(ds):
    # E1, E5, E6: the subsets; E4: two atlases; E3: bone marrow with two donors held
    # out; E2: 20,000 cells of two 10-D latents
    if ds == "donors":
        z = common.cached(os.path.join(DIR, "donors.npz"), donors)
        return z["X"], z["y"]
    if ds == "scvi10":
        path = os.path.join(common.CACHE, "scvi_protocol.npz")
        if not os.path.exists(path):
            h5 = os.path.join(common.DATA, "zebrafish", "dev_zebrafish.h5ad")
            subprocess.run([SCVI_PYTHON, "-c", _SCVI, h5, path], check=True)
        z = np.load(path, allow_pickle=True)
        X, y = z["lat10"], z["ctype"]
    else:
        X, y = common.load("hnoca" if ds == "scpoli10" else ds)
    if ds in ("scvi10", "scpoli10"):
        rows = np.sort(np.random.default_rng(1).choice(len(X), 20000, replace=False))
        X, y = X[rows], y[rows]
    return np.asarray(X, np.float32), np.asarray(y).astype(str)


def donors():
    # gene filter, highly variable genes, scaling and PCA see the reference donors only
    import anndata as ad
    import scanpy as sc
    import scipy.sparse as sp
    from sklearn.decomposition import PCA

    a = ad.read_h5ad(os.path.join(common.DATA, "fetal_bone_marrow.h5ad"))
    donor = np.asarray(a.obs["donor_id"]).astype(str)
    if float(a.X.max()) < 50 and a.raw is not None:
        a = a.raw.to_adata()
    X = sp.csr_matrix(a.X, dtype=np.float32)
    ref = ~np.isin(donor, ("F29", "F45"))
    X = X[:, np.asarray(X[ref].sum(0)).ravel() >= 1]
    X = sp.diags((1e4 / np.asarray(X.sum(1)).ravel()).astype(np.float32)) @ X
    X.data = np.log1p(X.data)
    r = ad.AnnData(X[ref])
    sc.pp.highly_variable_genes(r, n_top_genes=2000)
    S = np.asarray(X[:, np.asarray(r.var["highly_variable"])].todense(), np.float64)
    sd = S[ref].std(0, ddof=1)
    S = np.clip((S - S[ref].mean(0)) / np.where(sd > 0, sd, 1.0), -10.0, 10.0)
    P = PCA(50, svd_solver="randomized", random_state=0).fit(S[ref]).transform(S)
    return dict(X=(P - P[ref].mean(0)) / (P[ref].std(0) + 1e-9), y=donor)


def split(part, y):
    # (fitted, placed) cells
    n = len(y)
    if part == "full":
        return np.arange(n), np.arange(0)
    if part == "tr":
        return common.split(n)
    if part in ("A", "B"):
        perm = np.random.default_rng(0).permutation(n)
        a, b = np.sort(perm[: n // 2]), np.sort(perm[n // 2 :])
        return (a, b) if part == "A" else (b, a)
    # the two held-out donors, or the type whose share is closest to 5% (E5)
    names, counts = np.unique(y, return_counts=True)
    held = names[[np.argmin(np.abs(counts / n - 0.05))]]
    held = np.isin(y, ("F29", "F45") if part == "ref" else held)
    return np.flatnonzero(~held), np.flatnonzero(held)


class Encoder(torch.nn.Module):
    # the non-invertible control, swapped in for flodr.train.Flow so that all else is
    # FloDR's: three tanh layers of 512, a skip onto the first two whitened coordinates
    # and a zero last layer, so it starts at the PCA layout as the flow does
    def __init__(self, d, *args, k=2, **kwargs):
        super().__init__()
        L, T = torch.nn.Linear, torch.nn.Tanh
        self.k, self._density_args = k, None
        self.net = torch.nn.Sequential(
            L(d, 512), T(), L(512, 512), T(), L(512, 512), T(), L(512, k)
        )
        torch.nn.init.zeros_(self.net[-1].weight)
        torch.nn.init.zeros_(self.net[-1].bias)
        self.layers, self.tail = torch.nn.ModuleList(), torch.nn.ModuleList()

    def forward(self, x):
        from flodr.model import _mlp

        # _mlp keeps the last layer in fp32 on the half-precision step, as for the flow
        return torch.cat([x[:, : self.k] + _mlp(self.net, x), x[:, self.k :]], 1)

    def forward_with_logdet(self, x):
        return self(x), x.new_zeros(len(x))

    def inverse(self, z):
        return torch.full_like(z, float("nan"))

    forward_main, inverse_main = forward, inverse


def fit_map(arm, X, seed):
    import flodr.train
    from flodr import FloDR

    if arm == "flodr":
        return FloDR(random_state=seed, verbose=False).fit(X)
    flow, flodr.train.Flow = flodr.train.Flow, Encoder
    # "mlp" reads narrow inputs without FloDR's noise columns, "mlp_pad" with them
    adv = None if arm == "mlp_pad" else {"pad": 0}
    try:
        return FloDR(random_state=seed, verbose=False, density=False, advanced=adv).fit(
            X
        )
    finally:
        flodr.train.Flow = flow


def knn(ref, query, k=K):
    import faiss

    index = faiss.IndexFlatL2(ref.shape[1])
    index.add(np.ascontiguousarray(ref, np.float32))
    d2, idx = index.search(np.ascontiguousarray(query, np.float32), k)
    return np.sqrt(np.clip(d2, 0, None)), idx


def self_nn(Y):
    # the 15 nearest within Y, self removed by index since ties can put it anywhere
    idx = knn(Y, Y, K + 1)[1]
    keep = idx != np.arange(len(Y))[:, None]
    keep[keep.all(1), -1] = False
    return idx[keep].reshape(len(Y), K)


def smin(est, Z, Zref, nn):
    # smallest singular value of the map's 2 x D Jacobian on the local input tangent
    # (top five directions of a cell's 15 input neighbours), in whitened coordinates
    P = Zref[nn].astype(np.float64)
    U = np.linalg.svd(P - P.mean(1, keepdims=True), full_matrices=False)[2][:, :5]
    U, dev, out = torch.as_tensor(U), next(est.flow_.parameters()).device, []
    for lo in range(0, len(Z), 2048):
        z = torch.as_tensor(Z[lo : lo + 2048], device=dev).requires_grad_(True)
        y = est.flow_(z)[:, :2]
        g = [torch.autograd.grad(y[:, i].sum(), z, retain_graph=not i) for i in (0, 1)]
        J = torch.stack([x[0] for x in g], 1).double().cpu()
        out.append(torch.linalg.svdvals(J @ U[lo : lo + 2048].transpose(1, 2))[:, -1])
    return torch.cat(out).numpy()


def cell(arm, ds, part, seed):
    # one fit and its placement of the other cells: FloDR ("flodr"), the encoder ("mlp",
    # "mlp_pad"), free coordinates ("free") or UMAP ("umap")
    def make():
        X, y = data(ds)
        own, other = split(part, y)
        Xo, Xc = X[own], X[other]
        print(f"fitting {arm} {ds} {part} s{seed}", flush=True)
        if arm == "umap":
            import umap

            est = umap.UMAP(n_neighbors=K, min_dist=0.1, random_state=seed, init="pca")
            return dict(Y=est.fit_transform(Xo), umap=est.transform(Xc))
        if arm == "free":
            Y = common.fit_free(Xo, seed)
            if not len(other):
                return dict(Y=Y)
            # inverse-distance weighting of the 15 input neighbours, then the step of
            # transform_opt from there
            from flodr import find_ab

            d, nn = knn(Xo, Xc)
            w = 1.0 / (d + 1e-9)
            y0 = ((w / w.sum(1, keepdims=True))[..., None] * Y[nn]).sum(1)
            a, b = find_ab(0.01)
            y, Yn = torch.tensor(y0, requires_grad=True), torch.as_tensor(Y[nn])
            opt = torch.optim.Adam([y], lr=0.05)
            for _ in range(200):
                d2 = ((y[:, None] - Yn) ** 2).sum(-1)
                loss = -torch.log(1.0 / (1.0 + a * (d2 + 1e-6) ** b) + 1e-6).mean()
                opt.zero_grad()
                loss.backward()
                opt.step()
            return dict(Y=Y, interp=y0, opt=y.detach().numpy())
        est = fit_map(arm, Xo, seed)
        out = dict(Y=est.embedding_, rt=est.roundtrip_)
        if len(other):
            ref = np.random.default_rng(5).choice(len(Xo), min(4000, len(Xo)), False)
            ref, Z = np.sort(ref), est._coords
            out.update(
                forward=est.transform(Xc),
                opt=est.transform_opt(Xc),
                jt=smin(est, Z[ref], Z, self_nn(Xo)[ref]),
                jo=smin(est, est._to_coords(Xc), Z, knn(Xo, Xc)[1]),
            )
        return out

    path = os.path.join(DIR, arm, f"{ds}_{part}_s{seed}.npz")
    return {k: np.float32(v) for k, v in common.cached(path, make).items()}


def collapse(c):
    # placed cells whose tangent singular value is below 0.1 of the training median
    return float(np.mean(c["jo"] < 0.1 * np.median(c["jt"])))


def overlap(A, B):
    return (A[:, :, None] == B[:, None, :]).any(2)


def ranks(v):
    # Spearman correlation as the dot product of centred, normalised average ranks
    r = rankdata(v)
    r -= r.mean()
    return r / np.linalg.norm(r)


def dist(Y, i, j, c=100_000):
    d = [
        np.linalg.norm(Y[i[s : s + c]] - Y[j[s : s + c]], axis=1)
        for s in range(0, len(i), c)
    ]
    return np.concatenate(d)


def spearman(X, seed, draws, cap=None, far=False):
    # map against input distances on random pairs that the ordinal term never samples,
    # or on the farthest fifth of them
    rng = np.random.default_rng(seed)
    i, j = rng.integers(0, len(X), draws), rng.integers(0, len(X), draws)
    keep = (i != j) & ((i + j) % 5 == 0)
    i, j = i[keep][:cap], j[keep][:cap]
    dh = dist(X, i, j)
    if far:
        f = dh >= np.quantile(dh, 0.8)
        i, j, dh = i[f], j[f], dh[f]
    r = ranks(dh)
    return lambda Y: float(r @ ranks(dist(Y, i, j)))


def geodesic(X):
    # map distance against shortest paths along the symmetrised 15-NN graph of the
    # input, from 300 sources; the offset keeps duplicates connected
    n, nn, X = len(X), self_nn(X), X.astype(np.float64)
    d = [
        np.linalg.norm(X[s : s + 5000, None] - X[nn[s : s + 5000]], axis=2)
        for s in range(0, n, 5000)
    ]
    w = (np.concatenate(d).ravel() + 1e-9, (np.repeat(np.arange(n), K), nn.ravel()))
    A = coo_matrix(w, (n, n)).tocsr()
    src = np.sort(np.random.default_rng(0).choice(n, 300, replace=False))
    G = dijkstra(A.maximum(A.T), directed=False, indices=src)
    G[np.arange(300), src] = np.inf
    k, b = np.nonzero(np.isfinite(G))
    a, r = src[k], ranks(G[k, b])
    return lambda Y: float(r @ ranks(dist(np.asarray(Y, np.float64), a, b)))


def placed(hd, Ytr, Yc):
    # new cells: their 15 input neighbours among the training cells against their 15 map
    # neighbours among them
    return float(overlap(hd, knn(Ytr, Yc)[1]).mean())


def bench(ds):
    # E1 and E6: layout of the fits on all cells (M1), placement of the 20% held out
    # (M2r), and agreement across seeds (M3), of all cells with 80% (M4x) and of halves
    # (M5x): shared 15 map neighbours and the Spearman of distances on 500,000 pairs
    X, y = data(ds)
    n, (tr, te), hd = len(X), split("tr", y), self_nn(X)
    m1 = {
        k: spearman(X, 777, 6_000_000, 2_000_000, k == "far5") for k in ("cpd", "far5")
    }
    m1["geo"] = geodesic(X)
    hd_ref, pairs = knn(X[tr], X[te])[1], {}
    for m in (n, len(tr)):
        rng = np.random.default_rng(779)
        i, j = rng.integers(0, m, 500_000), rng.integers(0, m, 500_000)
        pairs[m] = i[i != j], j[i != j]

    def view(Y):
        return self_nn(Y), ranks(dist(Y, *pairs[len(Y)]))

    out = {}
    for arm in ("flodr", "mlp", "free", "umap"):
        seeds, r = range(10 if arm in MAPS else 3), out.setdefault(arm, {})

        def put(key, s, v):
            r.setdefault(key, {})[str(s)] = float(v)

        def agree(key, A, B, ordered=True):
            for a in seeds:
                for b in seeds:
                    if a < b or (ordered and a != b):
                        put(f"{key} knn", f"{a}-{b}", overlap(A[a][0], B[b][0]).mean())
                        put(f"{key} dist", f"{a}-{b}", A[a][1] @ B[b][1])

        T = {s: cell(arm, ds, "tr", s) for s in seeds}
        for s, c in T.items():
            for tf in set(c) & PLACE:
                put(f"M2r {tf} recall", s, placed(hd_ref, c["Y"], c[tf]))
        if arm == "umap":
            continue
        F = {s: cell(arm, ds, "full", s)["Y"] for s in seeds}
        for s, Y in F.items():
            put("M1 recall", s, overlap(hd, self_nn(Y)).mean())
            for k, f in m1.items():
                put(f"M1 {k}", s, f(Y))
        v = {s: view(Y) for s, Y in F.items()}
        agree("M3", v, v, ordered=False)
        vt = {s: view(c["Y"]) for s, c in T.items()}
        agree("M4x", {s: view(Y[tr]) for s, Y in F.items()}, vt)
        tf, H = "interp" if arm == "free" else "forward", {}
        for p in "AB":
            own, other = split(p, y)
            for s in seeds:
                c, Y = cell(arm, ds, p, s), np.empty((n, 2), np.float32)
                Y[own], Y[other] = c["Y"], c[tf]
                H[p, s] = view(Y)
        agree(f"M5x {tf}", {s: H["A", s] for s in seeds}, {s: H["B", s] for s in seeds})
        print(f"scored {ds} {arm}", flush=True)
    return out


def held(ds, part, arms, layout=None):
    # per arm and seed: placement of the other cells and the collapse of the map there,
    # with the layout scored by recall ("recall"), or by the atlas scorer, the farthest
    # fifth of its held-out pairs and geodesic CPD ("atlas")
    X, y = data(ds)
    own, other = split(part, y)
    Xo, out = X[own], {}
    hd = knn(Xo, X[other])[1] if len(other) else None
    nn = self_nn(Xo) if layout == "recall" else None
    if layout == "atlas":
        sc, geo = common.Scorer(Xo, large=True), geodesic(Xo)
        far = spearman(Xo, 778, 400_000, far=True)

    def score(Y):
        if layout != "atlas":
            return {"M1 recall": overlap(nn, self_nn(Y)).mean()} if layout else {}
        keys = ("M1 recall", "M1 cpd", "M1 far5", "M1 geo")
        return dict(zip(keys, (*sc(Y)[:2], far(Y), geo(Y))))

    for arm in arms:
        for s in range(3):
            c = cell(arm, ds, part, s)
            row = {f"M2r {t} recall": placed(hd, c["Y"], c[t]) for t in set(c) & PLACE}
            row.update(
                score(c["Y"]), **({"collapse": collapse(c)} if "jo" in c else {})
            )
            out.setdefault(arm, {})[str(s)] = row
    return out


@functools.cache
def E(name):
    # the scores of one experiment, computed once
    latents, both = ("scvi10", "scpoli10"), MAPS + ("free",)
    make = {
        "E1": lambda: {d: bench(d) for d in SUBSETS},
        "E2": lambda: {
            d: held(d, "tr", MAPS + ("mlp_pad", "free"), "recall") for d in latents
        },
        "E3": lambda: {"donors": held("donors", "ref", MAPS + ("umap",))},
        "E4": lambda: {
            "bmarrow": held("bmarrow", "tr", both, "atlas"),
            "cerebellum": held("cerebellum", "full", MAPS, "atlas"),
        },
        "E5": lambda: {d: held(d, "notype", MAPS) for d in SUBSETS},
    }[name]
    return common.memo(SCORES, name, make)


def interval(parts):
    # Welch t interval of the equal-weight mean of a - b over data sets; parts: (mean,
    # squared standard error, seeds) of a and of b
    m = len(parts)
    comps = [(v / m**2, k) for p in parts for _, v, k in p]
    v = sum(c for c, _ in comps)
    df = v**2 / sum(c**2 / (k - 1) for c, k in comps) if v > 0 else np.inf
    d = float(np.mean([a[0] - b[0] for a, b in parts]))
    h = stats.t.ppf(0.975, df) * np.sqrt(v)
    return d, (d - h, d + h)


def compare(rows, fl):
    # rows: {data set: (a, b)}, each (mean, squared standard error, seeds); the verdict
    # is "a" or "b" when the pooled interval favours one and no data set the other
    per = {ds: (p[0][0], p[1][0], interval([p])[1]) for ds, p in rows.items()}
    d, (lo, hi) = interval(list(rows.values()))
    for_a = any(c[0] > 0 for *_, c in per.values())
    for_b = any(c[1] < 0 for *_, c in per.values())
    within = -fl < lo and hi < fl
    if lo > 0 and not for_b:
        v = "a"
    elif hi < 0 and not for_a:
        v = "b"
    else:
        v = "equivalent" if within and not (for_a and for_b) else "unresolved"
    per = {ds: dict(a=a, b=b, diff=a - b, ci=c) for ds, (a, b, c) in per.items()}
    return dict(diff=d, ci=(lo, hi), verdict=v, within=within, per=per)


def floor(name):
    # tie margins
    return 0.01 if "recall" in name else 0.02


def mean_over(node, C, sym):
    # the mean over fits, or over pairs of fits with distinct seeds, each seed counted
    # as often as in C
    if "-" not in next(iter(node)):
        V = np.array([node[str(s)] for s in range(10)])
        return (C @ V) / C.sum(1)
    V = np.full((10, 10), np.nan)
    for p, x in node.items():
        a, b = map(int, p.split("-"))
        V[a, b] = x
        V[b, a] = x if sym else V[b, a]
    M, V = np.isfinite(V).astype(float), np.nan_to_num(V)
    with np.errstate(invalid="ignore"):
        return np.einsum("bi,ij,bj->b", C, V, C) / np.einsum("bi,ij,bj->b", C, M, C)


def e1(name):
    # FloDR minus the encoder over ten seeds, each arm's standard error from 10,000
    # replicates that resample seeds; pairs of fits that share a seed stay dependent
    rows = {}
    for ds in SUBSETS:
        rows[ds] = []
        for arm in MAPS:
            node = E("E1")[ds][arm][name]
            rng = np.random.default_rng(zlib.crc32(f"{ds}/{arm}/{name}".encode()))
            draws = rng.integers(0, 10, (10_000, 10))
            C = np.stack([(draws == s).sum(1) for s in range(10)], 1).astype(float)
            reps = mean_over(node, C, name.startswith("M3"))
            point = mean_over(node, np.ones((1, 10)), name.startswith("M3"))[0]
            rows[ds].append((point, np.nanvar(reps, ddof=1) * 10 / 9, 10))
    return compare(rows, floor(name))


def e6(ds, name, ref, comp, name_comp=None):
    # comparator minus reference over seeds 0-2 (agreements 0-1, 0-2, 1-2 across
    # seeds, s with s + 1 mod 3 otherwise), t interval with 4 degrees of freedom
    def vals(arm, key):
        ks = {"M1": "012", "M2": "012", "M3": ("0-1", "0-2", "1-2")}.get(key[:2])
        return np.array([E("E1")[ds][arm][key][k] for k in ks or ("0-1", "1-2", "2-0")])

    f, c = vals(ref, name), vals(comp, name_comp or name)
    d, se = c.mean() - f.mean(), np.sqrt(c.var(ddof=1) / 3 + f.var(ddof=1) / 3)
    t = stats.t.ppf(0.975, 4)
    return dict(diff=d, ci=(d - t * se, d + t * se))


def welch(R, key, a, b, key_b=None):
    # a minus b over seeds per data set, pooled over those where both arms have scores
    rows = {}
    for ds in R:
        x, z = (
            [v[k] for v in R[ds].get(arm, {}).values() if k in v]
            for arm, k in ((a, key), (b, key_b or key))
        )
        if x and z:
            rows[ds] = [
                (np.mean(v), np.var(v, ddof=1) / len(v), len(v)) for v in (x, z)
            ]
    return compare(rows, floor(key))


def e4(key, a, b, key_b=None):
    return welch(E("E4"), key, a, b, key_b)


def within(*diffs):
    # an upper bound on the differences at three decimals
    return np.ceil(max(map(abs, diffs)) * 1000 - 1e-9) / 1000


def rows():
    def bm(key, ds="bmarrow"):
        # FloDR, the encoder and free coordinates on an atlas, seed means
        return [np.mean([v[key] for v in E("E4")[ds][a].values()]) for a in E("E4")[ds]]

    def e1_within(name):
        return within(*(r["diff"] for r in e1(name)["per"].values()))

    every, F2 = common.BENCHMARKS + common.ATLASES, "FloDR (w=2)"
    C = {
        (d, m): common.scores(d, m).mean(0)
        for d in every
        for m in (F2, "UMAP", "openTSNE")
    }
    tsne = "zero one two three four five six seven".split()
    tsne = tsne[sum(C[d, F2][0] < C[d, "openTSNE"][0] for d in every)]
    free = [common.scores(d, "Free layout (pca)")[:, 0].mean() for d in common.ATLASES]
    free = np.mean(free) - np.mean([C[d, F2][0] for d in common.ATLASES])
    fn = -np.mean(
        [e6(d, "M1 recall", a, "free")["diff"] for d in SUBSETS[:2] for a in MAPS]
    )
    cpd = [-e6("mnist10k", "M1 cpd", a, "free")["diff"] for a in MAPS]
    rec, cb, cc = bm("M1 recall"), bm("M1 cpd"), bm("M1 cpd", "cerebellum")
    geo, far, enc = bm("M1 geo"), bm("M1 far5"), e4("M1 recall", "flodr", "mlp")["per"]
    dfree = rec[2] - rec[0]
    w0 = common.scores("bmarrow", "FloDR (w=0)")[:, 1].mean()
    w0 = C["bmarrow", F2][1], w0, C["bmarrow", "UMAP"][1]
    text = rf"""
Neighbours kept (recall) & above UMAP on the four benchmarks and three atlases, below it
on full HNOCA and zebrafish, and below openTSNE on {tsne} of seven data sets.
Benchmarks: both functions about ${fn:.3f}$ above free coordinates on paul15 and the
bone marrow subsample, with intervals that exclude zero but lie inside the tie margin of
E6, and tied on MNIST (E6); encoder within ${e1_within("M1 recall"):.3f}$ of FloDR (E1).
Atlas: encoder ${-enc["bmarrow"]["diff"]:.3f}$ above FloDR on the bone marrow atlas and
${-enc["cerebellum"]["diff"]:.3f}$ on the cerebellum, free coordinates ${dfree:.3f}$
above FloDR on the bone marrow atlas (E4); free coordinates fitted on the whole atlases
were about ${free:.2f}$ above FloDR on all three (Results) \\

Global structure (CPD) & benchmarks: both functions above free coordinates on the bone
marrow subsample and on MNIST, there by ${cpd[0]:.3f}$ and ${cpd[1]:.3f}$ inside the tie
margin, and tied on paul15 (E6); encoder within ${e1_within("M1 cpd"):.3f}$ (E1). Atlas:
FloDR ${cb[0]:.3f}$, encoder ${cb[1]:.3f}$, free coordinates ${cb[2]:.3f}$ on the bone
marrow atlas, FloDR ${cc[0]:.3f}$ and encoder ${cc[1]:.3f}$ on the cerebellum (E4).
Outside these rules, the ordinal term is needed: without it CPD on the whole bone marrow
atlas falls from ${w0[0]:.2f}$ to ${w0[1]:.2f}$ (UMAP ${w0[2]:.2f}$), and FloDR is above
UMAP on every data set scored \\

Distances along the neighbour graph & both functions above free coordinates on the bone
marrow subsample, FloDR also on MNIST and the encoder also on paul15 (E6), and on the
bone marrow atlas, ${geo[0]:.2f}$ and ${geo[1]:.2f}$ against ${geo[2]:.2f}$ (E4);
encoder within ${e1_within("M1 geo"):.2f}$ (E1); FloDR above the best of eight UMAP
settings on 6 of 7 data sets \\

Order of the largest distances & farthest fifth of pairs, atlas: FloDR ${far[0]:.2f}$,
encoder ${far[1]:.2f}$, free coordinates ${far[2]:.2f}$ (E4); benchmarks: both functions
above free coordinates on the bone marrow subsample, FloDR below them on MNIST (E6) \\

Same neighbours after refitting & both functions above free coordinates after removing a
fifth of the points, and across seeds FloDR on all three and the encoder only on paul15,
mostly on neighbours that are not input neighbours; between disjoint halves, free
coordinates above both on MNIST (E6); FloDR within the margin of the encoder (E1) \\

Map distances after refitting & both functions above free coordinates after removing a
fifth of the points; across seeds above them on paul15 and below on the bone marrow
subsample; between halves below them on MNIST (E6); FloDR within the margin of the
encoder (E1) \\
"""
    return [" ".join(r.split()) for r in text.strip().split("\n\n")]


if __name__ == "__main__":
    text = "\n".join(rows())
    open(common.out("supp_table2.tex"), "w").write(text + "\n")
    print(text)
