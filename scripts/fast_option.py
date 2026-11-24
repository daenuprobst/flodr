import os

import numpy as np

from common import ATLASES, BENCHMARKS, CACHE, K, SEEDS, Scorer, fit, load, memo
from common import out, scores

# The fast option outside Extended Data Fig. 7, which holds its fit times: recall and CPD on
# the seven data sets and on all cells of the zebrafish atlas, and the placement of new points
FAST, DEFAULT = "FloDR fast", "FloDR (w=2)"
PATH = os.path.join(CACHE, "fast_option", "numbers.json")
scorer = []


def zebrafish(method):
    # one fit of all cells, scored as the atlases are
    if not scorer:
        scorer.append(Scorer(load("zebrafish")[0], "euclid", large=True))
    return [float(v) for v in scorer[0](fit("zebrafish", method, 0)[0])[:2]]


def placed(ds, method, seed):
    # a fifth of the points is left out of the fit and placed by one pass through the flow.
    # Of the K nearest fitted points of a placed point in the input, the share that is among
    # its K nearest on the map, over up to 5,000 placed points
    from flodr import FloDR
    from sklearn.neighbors import NearestNeighbors

    X = load(ds)[0]
    rng = np.random.default_rng(100 + seed)
    new = np.sort(rng.choice(len(X), len(X) // 5, replace=False))
    old = np.setdiff1d(np.arange(len(X)), new)
    est = FloDR(random_state=seed, verbose=False, fast=method == FAST).fit(X[old])
    new = new[:5000]

    def nearest(ref, query):
        nn = NearestNeighbors(n_neighbors=K).fit(ref)
        return nn.kneighbors(query, return_distance=False)

    a, b = nearest(X[old], X[new]), nearest(est.embedding_, est.transform(X[new]))
    return float(np.mean([len(set(i) & set(j)) for i, j in zip(a, b)]) / K)


lines = ["recall and CPD, fast option against default"]
for ds in BENCHMARKS + ATLASES:
    f, d = scores(ds, FAST).mean(0), scores(ds, DEFAULT).mean(0)
    lines.append(f"{ds:11s} {f[0]:.3f} {f[1]:.3f} against {d[0]:.3f} {d[1]:.3f}")
lines.append("all cells of the zebrafish atlas, one fit each")
for m in (FAST, DEFAULT, "UMAP"):
    r = memo(PATH, f"zebrafish/{m}", lambda: zebrafish(m))
    lines.append(f"{m:11s} {r[0]:.4f} {r[1]:.3f}")
lines.append("placed points whose nearest fitted points are kept, fast against default")
for ds, seeds in (("mnist", SEEDS), ("bmarrow", SEEDS[:1])):
    v = {}
    for m in (FAST, DEFAULT):
        key = f"placed/{ds}/{m}/"
        v[m] = np.mean(
            [memo(PATH, key + str(s), lambda: placed(ds, m, s)) for s in seeds]
        )
    lines.append(
        f"{ds:11s} {v[FAST]:.3f} against {v[DEFAULT]:.3f}, {len(seeds)} seed(s)"
    )
open(out("fast_option.txt"), "w").write("\n".join(lines) + "\n")
print("\n".join(lines))
