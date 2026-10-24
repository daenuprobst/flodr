import os
import subprocess
import sys

import numpy as np
from scipy.spatial.distance import pdist, squareform
from scipy.stats import pearsonr, spearmanr

from common import BENCHMARKS, CACHE, K, SEEDS, Scorer, cached, fit, knn, load
from common import memo, metric, out

LABEL = {"FloDR (w=2)": "FloDR, $w=2$", "FloDR (w=0)": "FloDR, $w=0$"}
LABEL["PCUMAP"] = r"Param.\ UMAP"
METHODS = ["FloDR (w=2)", "FloDR (w=0)", "UMAP", "openTSNE", "PCA-2", "TriMap"]
METHODS += ["PaCMAP", "LocalMAP", "PHATE", "PyMDE", "PCUMAP", "SQuadMDS"]
DATA = {
    "mnist": ("MNIST", "raw pixels"),
    "fmnist": ("Fashion-MNIST", "raw pixels"),
    "paul15": ("paul15 (scRNA-seq)", r"HVG-1000 $\to$ PCA-50"),
    "drfp": ("Schneider 50k (reactions)", "DRFP, 2048 bits"),
}
HEAD = r"Method & recall@15 & trust & cont. & CPD ($\rho$) & Shepard $r$ & "
HEAD += r"SNS $\downarrow$ & centroid & time (s) \\"
# per column, +1 where higher is better; and whether it shows a standard deviation
SIGN, SD = np.array([1, 1, 1, 1, 1, -1, 1]), [1, 0, 0, 1, 0, 1, 0]
TIMES = os.path.join(CACHE, "ed_table1", "times.json")


def ranks(D):
    # the rank of every point as seen from every other, self last
    np.fill_diagonal(D, np.inf)
    R = np.empty(D.shape, np.int64)
    np.put_along_axis(R, np.argsort(D, 1), np.arange(1, len(D) + 1)[None], 1)
    return R


def agree(R, nn):
    # trustworthiness: the neighbours of one space ranked in the other (Venna and Kaski)
    n, excess = len(R), np.take_along_axis(R, nn, 1) - K
    return 1 - excess[excess > 0].sum() * 2 / (n * K * (2 * n - 3 * K - 1))


def measure(ds):
    X, y = load(ds)
    jaccard = metric(ds, "FloDR") == "jaccard"
    sc = Scorer(X, metric(ds, "FloDR"))
    Rh = ranks(squareform(sc.dh))
    groups = [np.flatnonzero(y == c) for c in np.unique(y)]
    if jaccard:
        # a mean of fingerprints is not a fingerprint, so a class is its medoid
        groups = [
            g[[squareform(pdist(X[g] > 0, "jaccard")).sum(1).argmin()]] for g in groups
        ]

    def centres(Z):
        return np.array([Z[g].mean(0) for g in groups])

    ch = pdist(centres(X) > 0, "jaccard") if jaccard else pdist(centres(X))
    res = {m: [] for m in METHODS}
    for m in METHODS:
        for s in SEEDS:
            Y = fit(ds, m, s)[0]
            rec, cpd, _ = sc(Y)
            dl = pdist(Y)
            fitted = sc.dh @ dl / (dl @ dl) * dl
            res[m].append(
                [
                    rec,
                    agree(Rh, knn(Y)),
                    agree(ranks(squareform(dl)), sc.hd),
                    cpd,
                    pearsonr(sc.dh, dl).statistic,
                    np.sqrt(np.sum((fitted - sc.dh) ** 2) / np.sum(sc.dh**2)),
                    spearmanr(ch, pdist(centres(Y))).statistic,
                ]
            )
    return res


def clock(ds, m):
    # one fit at seed 0 in a fresh interpreter, with nothing else running
    code = (
        f"import time, common as c; X = c.load({ds!r})[0]; t = time.perf_counter(); "
        f"c.FIT[{m!r}](X, 0, c.metric({ds!r}, {m!r})); print(time.perf_counter() - t)"
    )
    here = os.path.dirname(os.path.abspath(__file__))
    run = subprocess.run(
        [sys.executable, "-c", code],
        cwd=here,
        capture_output=True,
        text=True,
        check=True,
    )
    return float(run.stdout.split()[-1])


lines = [HEAD, r"\midrule"]
for ds in BENCHMARKS:
    X = load(ds)[0]
    name, representation = DATA[ds]
    dist = "Jaccard" if metric(ds, "FloDR") == "jaccard" else "Euclidean"
    if ds != BENCHMARKS[0]:
        lines.append(r"\midrule")
    lines.append(
        rf"\multicolumn{{9}}{{@{{}}l}}{{\emph{{{name}}} ($n={len(X)}$, "
        rf"$d={X.shape[1]}$, {representation}; {dist})}} \\"
    )
    v = cached(os.path.join(CACHE, "ed_table1", f"{ds}.npz"), lambda: measure(ds))
    # bold the best, underline the second best per column
    order = np.argsort([-SIGN * v[m].mean(0) for m in METHODS], 0, kind="stable")
    mark = np.full((len(METHODS), 7), "", object)
    mark[order[0], range(7)], mark[order[1], range(7)] = r"\mathbf", r"\underline"
    for i, m in enumerate(METHODS):
        cells = []
        for c in range(7):
            body = f"{v[m][:, c].mean():.3f}"
            body += f"{{\\pm}}{v[m][:, c].std():.3f}" if SD[c] else ""
            cells.append(f"${mark[i, c]}{{{body}}}$" if mark[i, c] else f"${body}$")
        label = LABEL.get(m, m)
        if metric(ds, m) != metric(ds, "FloDR"):
            label += r"$^{\dagger}$"
        secs = memo(TIMES, f"{ds}/{m}", lambda: clock(ds, m))
        lines.append(f"{label} & " + " & ".join(cells) + f" & ${secs:.1f}$ \\\\")

open(out("ed_table1.tex"), "w").write("\n".join(lines) + "\n")
print("\n".join(lines))
