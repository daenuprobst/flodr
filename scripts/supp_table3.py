import functools
import os

import numpy as np
import torch
from scipy.spatial import cKDTree

import common
import supp_table2 as T


def no_collapse(R):
    # the flow's if the encoder collapses more placed cells than FloDR, pooled, by more
    # than 1% of them, with no data set against
    c = T.welch(R, "collapse", "mlp", "flodr")
    against = any(p["ci"][1] < 0 for p in c["per"].values())
    return "flow" if c["ci"][0] > 0 and c["diff"] > 0.01 and not against else "none"


def stable_data(ds):
    # the follow-up reads paul15 with its components unscaled
    X, y = common.paul15_pcs()[:2] if ds == "paul15" else T.data(ds)
    return X, np.unique(y, return_inverse=True)[1]


def queries(y):
    # 300 held-out cells to place; the other held-out cells fill three random batches
    # and one batch of each of the two commonest types among them
    te = common.split(len(y))[1]
    q = np.sort(np.random.default_rng(11).choice(te, 300, replace=False))
    others = np.setdiff1d(te, q)
    n, rng = min(700, len(others) // 2), np.random.default_rng(12)
    batches = [np.sort(rng.choice(others, n, replace=False)) for _ in range(3)]
    names, counts = np.unique(y[others], return_counts=True)
    for t in names[np.argsort(-counts)[:2]]:
        pool = others[y[others] == t]
        batches.append(np.sort(rng.choice(pool, min(n, len(pool)), replace=False)))
    return q, batches


def stable_cell(ds, arm, seed):
    # the queries placed inside the first batch and, at seed 0, alone, inside each
    # batch, at one and eight CPU threads and on the GPU
    def make():
        import numba
        import umap
        from flodr.estimator import _flow_to

        X, y = stable_data(ds)
        tr = common.split(len(X))[0]
        q, batches = queries(y)
        if arm == "umap":
            est = umap.UMAP(n_neighbors=15, min_dist=0.1, init="pca", random_state=seed)
            est.fit(X[tr])
        else:
            est = T.fit_map(arm, X[tr], seed)
        out = dict(Y=est.embedding_)
        threads = numba.get_num_threads(), torch.get_num_threads()
        for tf in ("transform", "transform_opt") if arm == "flodr" else ("transform",):
            place = getattr(est, tf)
            out[tf] = place(X[np.r_[q, batches[0]]])[:300]
            if seed:
                continue
            P = [np.concatenate([place(X[[i]]) for i in q])]
            P += [place(X[np.r_[q, b]])[:300] for b in batches]
            for t in (1, 8):
                if arm == "umap":
                    numba.set_num_threads(t)
                else:
                    torch.set_num_threads(t)
                    _flow_to(est.flow_, "cpu")
                P.append(place(X[q]))
            if arm != "umap" and torch.cuda.is_available():
                _flow_to(est.flow_, "cuda")
                P.append(place(X[q]))
            numba.set_num_threads(threads[0])
            torch.set_num_threads(threads[1])
            out[tf + "_conditions"] = np.stack(P)
        return out

    return common.cached(
        os.path.join(T.DIR, "stability", f"{ds}_{arm}_s{seed}.npz"), make
    )


def majority(L):
    # the commonest code along the last axis, ties to the lower code
    return np.stack([(L == t).sum(-1) for t in range(L.max() + 1)], -1).argmax(-1)


@functools.cache
def placement_test():
    # per query: spread over the conditions in median map spacings, flips of the region
    # label (majority of the 15 nearest training cells) over the conditions and over
    # seeds 0-4, recall@15, and agreement of the region label with the input neighbours.
    # Registered: FloDR's transform flips less than UMAP.transform over conditions and
    # over seeds, by two paired standard errors, and loses at most 0.02 in recall and
    # label agreement (95% bound), on both data sets
    res = {}
    for ds in ("paul15", "bmarrow20k"):
        X, y = stable_data(ds)
        tr, q = common.split(len(X))[0], queries(y)[0]
        yt, in15 = y[tr], cKDTree(X[tr]).query(X[q], k=15)[1]
        for arm, tf in (
            ("flodr", "transform"),
            ("flodr", "transform_opt"),
            ("mlp", "transform"),
            ("umap", "transform"),
        ):
            c = [stable_cell(ds, arm, s) for s in range(5)]
            E0, P = c[0]["Y"], c[0][tf + "_conditions"]
            sp = np.median(cKDTree(E0).query(E0, k=2)[0][:, 1])
            L = [majority(yt[cKDTree(E0).query(p, k=15)[1]]) for p in P]
            nn = [cKDTree(x["Y"]).query(x[tf], k=15)[1] for x in c]
            Ls = np.stack([majority(yt[i]) for i in nn])
            hits = [[len(np.intersect1d(a, b)) for a, b in zip(in15, i)] for i in nn]
            res[ds, arm if tf == "transform" else "flodr_opt"] = dict(
                sd=np.sqrt(((P - P.mean(0)) ** 2).sum(-1).mean(0)) / sp,
                flip_proc=(np.stack(L) != L[0]).any(0),
                flip_seed=(Ls != majority(Ls.T)).any(0),
                recall=np.mean(hits, 0) / 15,
                agree=(Ls == majority(yt[in15])).mean(0),
            )

    def passes(key, sign, z, low):
        # the bound mean - z SE of sign * (FloDR - UMAP.transform) above low, both sets
        for ds in ("paul15", "bmarrow20k"):
            d = sign * (res[ds, "flodr"][key] - res[ds, "umap"][key].astype(float))
            if d.mean() - z * d.std(ddof=1) / np.sqrt(len(d)) <= low:
                return False
        return True

    tests = {
        "accuracy": passes("recall", 1, 1.96, -0.02)
        and passes("agree", 1, 1.96, -0.02),
        "stability over conditions": passes("flip_proc", -1, 2, 0),
        "seed stability": passes("flip_seed", -1, 2, 0),
    }
    return res, [name for name, ok in tests.items() if not ok]


def sci(x):
    m, e = f"{x:.0e}".split("e")
    return rf"{m}\times10^{{{int(e)}}}"


def rows():
    fwd = T.e1("M2r forward recall")["per"]
    opt = T.within(
        *(r["diff"] for r in T.e1("M2r opt recall")["per"].values()),
        T.welch(T.E("E3"), "M2r opt recall", "flodr", "mlp")["diff"],
        T.e4("M2r opt recall", "flodr", "mlp")["diff"],
    )
    free = [
        -T.e6("bmarrow20k", "M2r opt recall", a, "free")["diff"]
        for a in ("flodr", "mlp")
    ]
    fo = T.e4("M2r opt recall", "flodr", "free")["per"]["bmarrow"]
    atlas = T.e4("M2r forward recall", "flodr", "mlp")["diff"]
    donor = T.welch(T.E("E3"), "M2r forward recall", "flodr", "mlp")["diff"]
    E = {
        ds: [
            np.mean([v["M1 recall"] for v in r.values()])
            for r in T.E("E2")[ds].values()
        ]
        for ds in T.E("E2")
    }
    pad = -T.welch(T.E("E2"), "M1 recall", "flodr", "mlp_pad")["diff"]
    rt = [
        T.cell("flodr", ds, p, s)["rt"]
        for ds in T.SUBSETS
        for p in ("full", "tr", "A", "B")
        for s in range(10)
    ]
    res, failed = placement_test()
    sd = {
        a: sorted(res[ds, a]["sd"].mean() for ds in ("paul15", "bmarrow20k"))
        for a in ("flodr_opt", "umap")
    }
    fails = " and ".join(failed)
    text = rf"""
Placement by one forward pass & FloDR minus encoder ${fwd["paul15"]["diff"]:+.3f}$ on
paul15, ${fwd["bmarrow20k"]["diff"]:+.3f}$ on the bone marrow subsample and
${fwd["mnist10k"]["diff"]:+.3f}$ on MNIST (E1); both functions above interpolated free
coordinates on paul15 and the bone marrow subsample, below on MNIST; FloDR ties
UMAP.transform on paul15 and the bone marrow subsample and loses on MNIST (E6). On the
atlas, FloDR is above the encoder by ${atlas:.3f}$ (E4), which the code does not score
as a property \\

Placement by the optimised transform & benchmarks: both functions above free coordinates
only on the bone marrow subsample, by ${free[0]:.3f}$ and ${free[1]:.3f}$, and tied on
paul15 and MNIST, so neither the function nor the objective rule is met (E6). Atlas:
free coordinates ${-fo["diff"]:.3f}$ above FloDR, ${fo["b"]:.3f}$ against
${fo["a"]:.3f}$ (E4). FloDR within ${opt:.3f}$ of the encoder (E1, E3, E4); both above
UMAP.transform on held-out donors (E3) \\

Same placement alone, in any batch, thread count or device & FloDR and the encoder place
every cell at the same position, the optimised transform to within
${sci(sd["flodr_opt"][1])}$ map spacings; UMAP.transform moves cells by
${sd["umap"][0]:.1f}$ to ${sd["umap"][1]:.1f}$ spacings. Free-coordinate placement was
not an arm, and the registered test failed on {fails} (follow-up on the library
4801f79192a1e7ac) \\

Cells from held-out donors & FloDR above the encoder by ${donor:.3f}$ in recall of the
forward pass; both below UMAP.transform (E3) \\

No collapse of new inputs & neither map collapsed a held-out cell type (E5) \\

Narrow inputs & FloDR, the encoder with and without padding and free coordinates at
${min(E["scpoli10"]):.3f}$ to ${max(E["scpoli10"]):.3f}$ in recall on the scPoli latent
and ${min(E["scvi10"]):.3f}$ to ${max(E["scvi10"]):.3f}$ on the scVI latent, three
seeds, with the padded encoder above FloDR by ${pad:.3f}$ (E2) \\

Exact inverse and density & largest absolute round-trip error in whitened coordinates
${sci(min(rt))}$ to ${sci(max(rt))}$ over the {len(rt)} FloDR fits of E1, and up to
$1.5\times10^{{-4}}$ on the atlas fits of Extended Data Fig.~\ref{{fig:diagnostics}};
the density is exact and normalised, and describes the data only where the map is
near-linear at the noise scale \\

Inverse, density and placement agree & every decoded draw is placed at its own position
without a solver; a separate decoder gains the same consistency by projection onto the
fibres, with a solver (Supplementary Note~\ref{{sec:flowonly}}) \\

Hidden-information audit & needs only the inputs, a map and the labels \\
"""
    return [" ".join(r.split()) for r in text.strip().split("\n\n")]


if __name__ == "__main__":
    text = "\n".join(rows())
    open(common.out("supp_table3.tex"), "w").write(text + "\n")
    print(text)
