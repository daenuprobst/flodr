import json
import os
import sys
import time

import numpy as np
import torch
from scipy.stats import spearmanr
from sklearn.neighbors import NearestNeighbors

HERE = os.path.dirname(os.path.abspath(__file__))
SCRIPTS = os.path.dirname(HERE)
ROOT = os.path.dirname(SCRIPTS)
sys.path.insert(0, os.path.join(ROOT, "src"))
sys.path.insert(0, SCRIPTS)

from datasets import PAPER, METRIC  # noqa: E402
from flodr import FloDR  # noqa: E402

# every arm recompiles train_flodr's step(), and the full arm x seed grid blows through
# the default limit of 8 inside one process
torch._dynamo.config.recompile_limit = 64

K = 15
N_PAIR = 60_000
W = 2.0
DEVICE = os.environ.get("DEVICE", "cuda")
SEEDS = tuple(int(s) for s in os.environ.get("SEEDS", "0,1,2").split(","))
DATASETS = os.environ.get("DATASETS", "mnist,fmnist,paul15,drfp").split(",")
OUT = os.path.join(SCRIPTS, "cache", "component_ablation.json")

ARMS = {
    "base":    dict(density=False, advanced=dict(gate_max=0.5)),
    "layers2": dict(density=False, advanced=dict(gate_max=0.5, n_layers=2)),
    "layers8": dict(density=False, advanced=dict(gate_max=0.5, n_layers=8)),
    "nosketch": dict(density=False, advanced=dict(gate_max=0.5, sketch=0)),
    "nll":     dict(density=True, advanced=dict(gate_max=0.5)),
}


def knn_indices(A, k=K, metric="euclidean"):
    nn = NearestNeighbors(n_neighbors=k + 1, metric=metric).fit(A)
    return nn.kneighbors(A, return_distance=False)[:, 1:]


def pair_dist(X, pair_i, pair_j, metric):
    if metric == "jaccard":
        bits_i, bits_j = X[pair_i] > 0, X[pair_j] > 0
        inter = (bits_i & bits_j).sum(1)
        union = (bits_i | bits_j).sum(1)
        return 1.0 - inter / np.maximum(union, 1)
    return np.linalg.norm(X[pair_i] - X[pair_j], axis=1)


def eval_embedding(X, Y, metric, nn_x, pairs, d_in):
    nn_y = knn_indices(Y, metric="euclidean")
    rec = float(np.mean([len(set(nn_x[row]) & set(nn_y[row])) for row in range(len(Y))]) / K)
    cpd = float(spearmanr(np.linalg.norm(Y[pairs[0]] - Y[pairs[1]], axis=1), d_in).statistic)
    return rec, cpd


def fit_arm(X, arm, seed):
    for device in (DEVICE, "cpu") if DEVICE != "cpu" else ("cpu",):
        try:
            t0 = time.perf_counter()
            model = FloDR(w=W, random_state=seed, device=device,
                          density=ARMS[arm]["density"], advanced=ARMS[arm]["advanced"]).fit(X)
            secs = time.perf_counter() - t0
            if device != DEVICE:
                print(f"    [{arm} seed {seed}: {DEVICE} failed, refit on cpu]", flush=True)
            return np.asarray(model.embedding_, np.float32), secs
        except RuntimeError as e:
            if device == "cpu" or "cuda" not in str(e).lower():
                raise
            print(f"    [{arm} seed {seed}: {DEVICE} error ({e}); retrying on cpu]", flush=True)
    raise RuntimeError("unreachable")


def stat(vals):
    arr = np.asarray(vals, float)
    return float(arr.mean()), float(arr.std(ddof=1) if len(arr) > 1 else 0.0)


def main():
    res = json.load(open(OUT)) if os.path.exists(OUT) else {}
    t0 = time.perf_counter()

    for ds in DATASETS:
        X = np.asarray(PAPER[ds]()[0], np.float32)
        n, d = X.shape
        metric = METRIC.get(ds, "euclidean")
        print(f"\n{ds}: n={n}, d={d}, metric={metric}, device={DEVICE}", flush=True)

        nn_x = knn_indices(X, metric=metric)
        rng = np.random.default_rng(5)
        pair_i, pair_j = rng.integers(0, n, 8 * N_PAIR), rng.integers(0, n, 8 * N_PAIR)
        pair_mask = (pair_i != pair_j) & (((pair_i + pair_j) % 5) == 0)
        pair_i, pair_j = pair_i[pair_mask], pair_j[pair_mask]
        if len(pair_i) > N_PAIR:
            sel = rng.choice(len(pair_i), N_PAIR, replace=False)
            pair_i, pair_j = pair_i[sel], pair_j[sel]
        d_in = pair_dist(X, pair_i, pair_j, metric)
        pairs = (pair_i, pair_j)

        # the sketch conditioner only means anything where d > 64, and paul15 has d=50
        arms = [a for a in ARMS if not (a == "nosketch" and d <= 64)]
        for arm in arms:
            cell = res.setdefault(ds, {}).setdefault(arm, {})
            for seed in SEEDS:
                if str(seed) in cell:
                    continue
                Y, secs = fit_arm(X, arm, seed)
                rec, cpd = eval_embedding(X, Y, metric, nn_x, pairs, d_in)
                cell[str(seed)] = dict(recall=rec, cpd=cpd, secs=secs)
                print(f"  {arm:9s} seed {seed}  {secs:6.1f}s  recall {rec:.4f}  "
                      f"CPD {cpd:.4f}", flush=True)
                try:
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()
                except Exception:
                    pass
                json.dump(res, open(OUT, "w"), indent=1)

        # mean/std over seeds, stored next to the per-seed cells
        for arm in arms:
            cell = res[ds][arm]
            done = [cell[str(s)] for s in SEEDS if str(s) in cell]
            if done:
                for key in ("recall", "cpd", "secs"):
                    mean, std = stat([entry[key] for entry in done])
                    cell[f"{key}_mean"], cell[f"{key}_std"] = mean, std
        json.dump(res, open(OUT, "w"), indent=1)

    total = time.perf_counter() - t0
    print(f"\ntotal wall time {total/60:.1f} min (fits only per cell in 'secs'), wrote {OUT}")

    print(f"\n{'dataset':10s} {'arm':9s} {'recall@15':>15s} {'CPD':>15s} {'secs':>7s}")
    for ds in DATASETS:
        if ds not in res:
            continue
        for arm in ARMS:
            cell = res[ds].get(arm)
            if not cell or "recall_mean" not in cell:
                continue
            print(f"{ds:10s} {arm:9s} "
                  f"{cell['recall_mean']:7.4f} +/- {cell['recall_std']:5.4f} "
                  f"{cell['cpd_mean']:7.4f} +/- {cell['cpd_std']:5.4f} {cell['secs_mean']:7.1f}")
        print()


if __name__ == "__main__":
    main()
