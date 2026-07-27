"""Scores the held-out embeddings produced by heldout_benchmark.py (no training here):
test points only, against the full high-dimensional distance matrix (Euclidean, Jaccard
for drfp; computed once per dataset, cached as scripts/cache/heldout_hd_{ds}.npy).
Metrics: recall15 (2D/HD 15-NN overlap), trust15 (k=15), cpd (Spearman HD-vs-2D over
20 partners per test point, rng seed 777 fixed across methods). Aggregates mean/std
over reps per (dataset, method) into scripts/cache/heldout_benchmark.json. Idempotent.

Usage: .venv/bin/python scripts/heldout_score.py
"""
import glob
import json
import os

import numpy as np
from scipy.spatial import cKDTree
from scipy.stats import spearmanr
from sklearn.metrics import pairwise_distances

from datasets import PAPER, METRIC

CACHE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "cache")
EMBEDS = os.path.join(CACHE, "heldout_embeds")
OUT = os.path.join(CACHE, "heldout_benchmark.json")
K = 15
N_PAIRS = 20
PAIR_SEED = 777
METHODS = ("flodr", "flodr_w0", "flodr_w1", "flodr_w3",
           "umap", "opentsne", "pca2", "knnmap")             # preferred order


def method_order(found):
    """Known methods first in METHODS order, then any extras alphabetically."""
    return [m for m in METHODS if m in found] + sorted(set(found) - set(METHODS))


def hd_matrix(ds):
    """Full (n, n) float32 input-distance matrix, cached on disk."""
    path = os.path.join(CACHE, f"heldout_hd_{ds}.npy")
    if os.path.exists(path):
        return np.load(path)
    X, y, title, note = PAPER[ds]()
    X = np.asarray(X, np.float32)
    if METRIC.get(ds) == "jaccard":
        D = pairwise_distances(X.astype(bool), metric="jaccard")
    else:
        D = pairwise_distances(X, metric="euclidean")
    D = D.astype(np.float32)
    np.save(path, D)
    return D


def score_cell(D, Y_full, train_idx, test_idx):
    """All three metrics for one (ds, method, rep). Scores test points only."""
    n, m = len(Y_full), len(test_idx)

    # 2D kNN of each test point among all n points, self excluded
    _, knn_2d = cKDTree(Y_full).query(Y_full[test_idx], k=K + 1)
    y_nn = np.array([row[row != i][:K] for row, i in zip(knn_2d, test_idx)])

    # HD kNN and HD ranks, test rows only
    D_test = D[test_idx].astype(np.float32, copy=True)
    D_test[np.arange(m), test_idx] = np.inf
    hd_nn = np.argpartition(D_test, K, axis=1)[:, :K]
    order = np.argsort(D_test, axis=1)
    ranks = np.empty((m, n), np.int32)
    ranks[np.arange(m)[:, None], order] = np.arange(n)

    recall = float(np.mean([len(np.intersect1d(row_2d, row_hd)) / K
                            for row_2d, row_hd in zip(y_nn, hd_nn)]))

    nn_ranks = ranks[np.arange(m)[:, None], y_nn]        # HD rank of each 2D neighbour
    penalty = np.maximum(nn_ranks - K, 0).sum()
    trust = float(1.0 - 2.0 * penalty / (m * K * (2 * n - 3 * K - 1)))

    rng = np.random.default_rng(PAIR_SEED)               # same pairs for every method
    hd_dists, ld_dists = [], []
    for i in test_idx:
        partners = rng.integers(0, n - 1, N_PAIRS)
        partners = partners + (partners >= i)            # uniform over all points != i
        hd_dists.append(D[i, partners])
        ld_dists.append(np.linalg.norm(Y_full[i] - Y_full[partners], axis=1))
    cpd = float(spearmanr(np.concatenate(hd_dists), np.concatenate(ld_dists)).statistic)
    return dict(recall15=recall, trust15=trust, cpd=cpd)


def main():
    cells = {}
    for path in sorted(glob.glob(os.path.join(EMBEDS, "*.npz"))):
        method, ds, rep = os.path.basename(path)[:-4].rsplit("_", 2)
        cells.setdefault(ds, {}).setdefault(method, {})[int(rep)] = path
    if not cells:
        print(f"no npz files in {EMBEDS}")
        return

    out = {}
    for ds in PAPER:
        if ds not in cells:
            continue
        D = hd_matrix(ds)
        print(f"{ds}: HD matrix {D.shape} ready", flush=True)
        for method in method_order(cells[ds]):
            per_rep = []
            for rep in sorted(cells[ds][method]):
                z = np.load(cells[ds][method][rep])
                train_idx, test_idx = z["train_idx"], z["test_idx"]
                Y_full = np.empty((len(train_idx) + len(test_idx), 2), np.float32)
                Y_full[train_idx] = z["Y_train"]
                Y_full[test_idx] = z["Y_test"]
                per_rep.append(score_cell(D, Y_full, train_idx, test_idx))
                print(f"  {method} rep{rep}: {per_rep[-1]}", flush=True)
            agg = {"n_reps": len(per_rep)}
            for key in ("recall15", "trust15", "cpd"):
                vals = [r[key] for r in per_rep]
                agg[f"{key}_mean"], agg[f"{key}_std"] = float(np.mean(vals)), float(np.std(vals))
            out.setdefault(ds, {})[method] = agg

    with open(OUT, "w") as f:
        json.dump(out, f, indent=2, sort_keys=True)

    print(f"\nwrote {OUT}\n")
    hdr = f"{'dataset':<10} {'method':<9} {'recall@15':>16} {'trust@15':>16} {'cpd':>16}"
    print(hdr)
    print("-" * len(hdr))
    for ds in out:
        for method in method_order(out[ds]):
            stats = out[ds][method]
            row = [f"{stats[k + '_mean']:.4f} +/- {stats[k + '_std']:.4f}"
                   for k in ("recall15", "trust15", "cpd")]
            print(f"{ds:<10} {method:<9} {row[0]:>16} {row[1]:>16} {row[2]:>16}")


if __name__ == "__main__":
    main()
