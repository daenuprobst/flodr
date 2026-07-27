"""PCA-initialisation confound ablation on paul15, FloDR's clearest global-structure win
(CPD 0.709 vs PCA-2 0.497 vs UMAP 0.045). Only the whitened PCA head is rotated by a random
orthogonal Q, so the layout starts at a random 2D plane of PCA-50 space while the input
information, the knn graph, and the ordinal term's input-distance targets are unchanged.

Run: [DEVICE=cuda] [SEEDS=0,1,2] .venv/bin/python scripts/ablations/pca_init_confound.py
"""
import os
import sys
import json
import time

import numpy as np
from scipy.stats import spearmanr
from scipy.linalg import null_space
from sklearn.decomposition import PCA

HERE = os.path.dirname(os.path.abspath(__file__))
SCRIPTS = os.path.dirname(HERE)
ROOT = os.path.dirname(SCRIPTS)
sys.path.insert(0, os.path.join(ROOT, "src"))
sys.path.insert(0, SCRIPTS)

from datasets import PAPER                                    # noqa: E402
from flodr.data import fuzzy_knn_graph                        # noqa: E402
from flodr.train import raw_recipe, train_flodr              # noqa: E402

K = 15
DEVICE = os.environ.get("DEVICE", "cpu")
SEEDS = tuple(int(s) for s in os.environ.get("SEEDS", "0,1,2").split(","))
OUT = os.path.join(SCRIPTS, "cache", "pca_init_confound.json")


def whitened_coords(X_raw):
    """Reproduce evaluate.py's flow input: whitened PCA head + floored-whitened tail."""
    n_dim = X_raw.shape[1]
    pca = PCA(n_components=min(50, n_dim - 1), svd_solver="randomized", random_state=0).fit(X_raw)
    proj = pca.transform(X_raw)
    v_null = null_space(pca.components_).astype(np.float32)
    z_tail = (X_raw - pca.mean_) @ v_null
    tail_sd = np.maximum(z_tail.std(0), 1e-2 * z_tail.std(0).max() + 1e-12)
    z_white = np.hstack([(proj - proj.mean(0)) / (proj.std(0) + 1e-9),
                         z_tail / tail_sd]).astype(np.float32)
    return z_white, proj.shape[1]


def cpd(Y, pair_i, pair_j, d_targ):
    d_emb = np.linalg.norm(Y[pair_i] - Y[pair_j], axis=1)
    return float(spearmanr(d_emb, d_targ).statistic)


def rand_rotation(d, seed):
    """A random orthogonal d x d matrix from the QR of a Gaussian, sign-fixed for determinism."""
    gauss = np.random.default_rng(seed).standard_normal((d, d))
    q, r = np.linalg.qr(gauss)
    return (q * np.sign(np.diag(r))).astype(np.float32)


def fit(z_white, edge_i, edge_j, w_attr, X_raw, seed):
    cfg = raw_recipe(seed=seed, device=DEVICE, w_stress=2.0, stress_ordinal=1,
                     stress_metric="euclid")
    Y, roundtrip, _ = train_flodr(z_white, edge_i, edge_j, cfg, np.random.default_rng(seed),
                                  w=w_attr, stress_X=X_raw)
    return np.asarray(Y), float(roundtrip)


def main():
    X_raw, ylab = PAPER["paul15"]()[:2]
    X_raw = np.asarray(X_raw, np.float32)
    n, n_dim = X_raw.shape
    print(f"paul15: n={n}, d={n_dim}, device={DEVICE}, seeds={SEEDS}", flush=True)

    z_white, d_head = whitened_coords(X_raw)
    _, edge_i, edge_j, w_attr, _ = fuzzy_knn_graph(X_raw, K, return_dist=True)

    rng = np.random.default_rng(1000)                         # identical to evaluate.py
    pair_i, pair_j = rng.integers(0, n, 1_200_000), rng.integers(0, n, 1_200_000)
    mask = (pair_i != pair_j) & (((pair_i + pair_j) % 5) == 0)  # held out from the ordinal term
    pair_i, pair_j = pair_i[mask][:300_000], pair_j[mask][:300_000]
    d_targ = np.linalg.norm(X_raw[pair_i] - X_raw[pair_j], axis=1)

    # linear reference: PCA-2 is the initialisation the confound is about
    pca2 = PCA(n_components=2, svd_solver="randomized", random_state=0).fit_transform(X_raw)
    cpd_pca2 = cpd(pca2, pair_i, pair_j, d_targ)
    print(f"\nPCA-2 (the initialisation)         CPD {cpd_pca2:.3f}", flush=True)
    print("paper table: FloDR w=2 = 0.709, PCA-2 = 0.497, UMAP = 0.045\n", flush=True)

    rows = {"pca_init": [], "rand_init": []}
    for seed in SEEDS:
        t0 = time.perf_counter()
        # unchanged: layout starts at top-2 PCs
        Y_pca, _ = fit(z_white, edge_i, edge_j, w_attr, X_raw, seed)
        cpd_p = cpd(Y_pca, pair_i, pair_j, d_targ)

        Q = rand_rotation(d_head, 1000 + seed)                # rotate only the whitened head
        z_rot = np.hstack([z_white[:, :d_head] @ Q.T, z_white[:, d_head:]]).astype(np.float32)
        # layout starts at a random 2D plane
        Y_rand, _ = fit(z_rot, edge_i, edge_j, w_attr, X_raw, seed)
        cpd_r = cpd(Y_rand, pair_i, pair_j, d_targ)

        rows["pca_init"].append(cpd_p)
        rows["rand_init"].append(cpd_r)
        print(f"seed {seed}: pca-init CPD {cpd_p:.3f}   rand-init CPD {cpd_r:.3f}   "
              f"({time.perf_counter()-t0:.0f}s)", flush=True)

    def stat(vals):
        arr = np.array(vals)
        return float(arr.mean()), float(arr.std(ddof=1) if len(arr) > 1 else 0.0)

    mean_p, sd_p = stat(rows["pca_init"])
    mean_r, sd_r = stat(rows["rand_init"])
    print(f"\npca-init   CPD {mean_p:.3f} +/- {sd_p:.3f}", flush=True)
    print(f"rand-init  CPD {mean_r:.3f} +/- {sd_r:.3f}", flush=True)
    print(f"retained {mean_r/mean_p*100:.0f}% of the pca-init CPD; PCA-2 floor {cpd_pca2:.3f}",
          flush=True)

    json.dump({"n": n, "d": n_dim, "device": DEVICE, "seeds": list(SEEDS),
               "cpd_pca2": cpd_pca2, "pca_init": rows["pca_init"], "rand_init": rows["rand_init"],
               "pca_init_mean": mean_p, "pca_init_sd": sd_p,
               "rand_init_mean": mean_r, "rand_init_sd": sd_r},
              open(OUT, "w"), indent=1)
    print(f"-> {OUT}", flush=True)


if __name__ == "__main__":
    main()
