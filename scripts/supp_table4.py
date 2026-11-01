import os
from functools import lru_cache

import numpy as np
from scipy.stats import spearmanr

import common
from common import K

# w_lin per data set; 0 is the base the other rows are compared with at the same seed
WEIGHTS = {
    "bmarrow20k": (0, 1, 0.1, 0.03),
    "paul15": (0, 1, 0.1),
    "mnist10k": (0, 1, 0.1),
    "bmarrow": (0, 1, 0.3, 0.1),
    "cerebellum": (0, 0.3, 0.1, 0.03),
}
TITLE = {
    "bmarrow20k": "bone marrow 20k",
    "paul15": r"paul15$^\ast$",
    "mnist10k": r"MNIST$^\ast$",
    "bmarrow": r"bone marrow 82k$^\ast$",
    "cerebellum": r"cerebellum$^\ast$",
}
CACHE = os.path.join(common.CACHE, "supp_table4")


@lru_cache(None)
def data(ds):
    # paul15 on its unscaled principal components; 80/20 split
    X = common.paul15_pcs()[0] if ds == "paul15" else common.load(ds)[0]
    return (X, *common.split(len(X)))


def measure(est, X, tr, te):
    # rho(1) of 2,000 training cells: the decode deviation of an input step at the density's noise
    from flodr.fidelity import faithful_scale

    rows = np.random.default_rng(0).choice(len(tr), 2000, replace=False)
    hs = (0.125, 0.25, 0.5, 1.0, 2.0)
    fs = faithful_scale(
        est, est._coords[rows], est.flow_._density_noise, hs, 4, 0.25, 0
    )
    return dict(Y=est.embedding_, Yte=est.transform(X[te]), rho=fs["rho"][:, 3])


def fit(ds, w, seed):
    def make():
        from flodr import FloDR

        print("fit", ds, f"w_lin={w:g}", "seed", seed, flush=True)
        X, tr, te = data(ds)
        # at weight 1 the small sets take the term at every step
        every = 1 if w == 1 and ds != "bmarrow" else 2
        adv = dict(w_lin=w, lin_every=every) if w else None
        est = FloDR(random_state=seed, verbose=False, advanced=adv).fit(X[tr])
        return measure(est, X, tr, te)

    return common.cached(os.path.join(CACHE, f"{ds}_w{w:g}_s{seed}.npz"), make)


def overlap(a, b):
    return np.mean([len(set(p) & set(q)) for p, q in zip(a, b)]) / K


@lru_cache(None)
def score(ds, w, seed):
    # rho(1) median, near-linear share, train recall@15 and CPD (200,000 random pairs), and
    # recall@15 of held-out cells placed by the forward pass among the training cells
    X, tr, te = data(ds)
    z = fit(ds, w, seed)
    Xtr, Y = X[tr], z["Y"]
    recall = overlap(common.knn_faiss(Xtr, Xtr)[:, 1:], common.knn_faiss(Y, Y)[:, 1:])
    i, j = np.random.default_rng(0).integers(0, len(Xtr), (2, 200_000))
    dx, dy = (np.linalg.norm(A[i] - A[j], axis=1) for A in (Xtr, Y))
    place = overlap(
        common.knn_faiss(Xtr, X[te])[:, :K], common.knn_faiss(Y, z["Yte"])[:, :K]
    )
    rho = z["rho"]
    return np.array(
        [np.median(rho), np.mean(rho <= 0.3), recall, spearmanr(dx, dy)[0], place]
    )


if __name__ == "__main__":
    lines = []
    for ds, ws in WEIGHTS.items():
        for w in ws:
            sds = (0, 1, 2) if ds == "bmarrow20k" and w in (0, 1) else (0,)
            s = np.mean([score(ds, w, sd) for sd in sds], 0)
            base = np.mean([score(ds, 0, sd) for sd in sds], 0)
            star = r"$^\ast$" if ds == "bmarrow20k" and len(sds) == 1 else ""
            # base rows give recall and CPD, the others their change
            d = s[2:4] - base[2:4]
            rc = [f"${x:+.3f}$" if round(x, 4) else "$0.000$" for x in d]
            rc = rc if w else [f"${x:.3f}$" for x in s[2:4]]
            head = ["" if w else TITLE[ds], f"{w:g}{star}"] + [
                f"${x:.2f}$" for x in s[:2]
            ]
            lines.append(" & ".join(head + rc + [f"${s[4]:.3f}$"]) + r" \\")
    open(common.out("supp_table4.tex"), "w").write("\n".join(lines) + "\n")
    print("\n".join(lines))
