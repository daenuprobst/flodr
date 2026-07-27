"""Held-out (out-of-sample) embedding benchmark: FloDR.transform against the baselines'
own transform procedures, on deterministic train/test splits of every PAPER dataset.
Writes scripts/cache/heldout_embeds/*.npz (existing cells are skipped) and
heldout_meta.json; scores are computed separately by heldout_score.py.

Usage: .venv/bin/python scripts/heldout_benchmark.py
"""
import json
import os
import time

import numpy as np
from scipy.spatial import cKDTree
from sklearn.decomposition import PCA
from sklearn.neighbors import NearestNeighbors

from datasets import PAPER, METRIC

CACHE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "cache")
EMBEDS = os.path.join(CACHE, "heldout_embeds")
META = os.path.join(CACHE, "heldout_meta.json")
os.makedirs(EMBEDS, exist_ok=True)
N_REPS = 5
K_NN = 15

DEVICE = "cuda"
try:
    import torch
    import torch.func
    if not torch.cuda.is_available():
        DEVICE = "cpu"
    # each fit compiles fresh closures; the default recompile limit (8) is exhausted by
    # a full sweep in one process and fullgraph=True turns that into a hard failure
    torch._dynamo.config.cache_size_limit = 128
except ImportError:
    torch = None
    DEVICE = "cpu"

from flodr import FloDR
from flodr.data import fuzzy_knn_graph


def split(n, rep):
    """The exact split every producer of heldout embeddings must use."""
    perm = np.random.default_rng(10_000 + rep).permutation(n)
    return perm[n // 5:], perm[: n // 5]          # train_idx, test_idx


def load_meta():
    if os.path.exists(META):
        with open(META) as f:
            return json.load(f)
    return {}


def save_meta(meta):
    tmp = META + ".tmp"
    with open(tmp, "w") as f:
        json.dump(meta, f, indent=2, sort_keys=True)
    os.replace(tmp, META)


def cell_path(method, ds, rep):
    return os.path.join(EMBEDS, f"{method}_{ds}_{rep}.npz")


def save_cell(method, ds, rep, train_idx, test_idx, Y_train, Y_test):
    np.savez(cell_path(method, ds, rep), train_idx=train_idx, test_idx=test_idx,
             Y_train=np.asarray(Y_train, np.float32), Y_test=np.asarray(Y_test, np.float32))


def record_timing(meta, ds, method, rep, fit_secs, transform_secs, device=None):
    rec = dict(fit_secs=round(fit_secs, 3), transform_secs=round(transform_secs, 3))
    if device:
        rec["device"] = device
    meta.setdefault("timings", {}).setdefault(ds, {}).setdefault(method, {})[str(rep)] = rec


def run_flodr(X_train, X_test, seed, device, w=2.0):
    est = FloDR(w=w, random_state=seed, device=device,
                advanced=dict(gate_max=0.5))
    t0 = time.perf_counter()
    est.fit(X_train)
    fit_secs = time.perf_counter() - t0
    t0 = time.perf_counter()
    Y_test = est.transform(X_test)
    transform_secs = time.perf_counter() - t0
    return est, est.embedding_, Y_test, fit_secs, transform_secs


def fit_flodr_safe(X_train, X_test, seed, w=2.0):
    """Prefer cuda; on CUDA OOM (shared GPU) fall back to cpu."""
    try:
        return run_flodr(X_train, X_test, seed, DEVICE, w)
    except RuntimeError as e:
        if DEVICE == "cpu" or "memory" not in str(e).lower():
            raise
        print(f"    cuda failed ({e}); retrying on cpu", flush=True)
        if torch is not None:
            torch.cuda.empty_cache()
        return run_flodr(X_train, X_test, seed, "cpu", w)


def run_umap(X_train, X_test, seed, metric):
    import umap
    est = umap.UMAP(n_neighbors=15, random_state=seed, metric=metric)
    t0 = time.perf_counter()
    est.fit(X_train)
    fit_secs = time.perf_counter() - t0
    t0 = time.perf_counter()
    Y_test = est.transform(X_test)
    transform_secs = time.perf_counter() - t0
    return np.asarray(est.embedding_), np.asarray(Y_test), fit_secs, transform_secs


def run_opentsne(X_train, X_test, seed, metric):
    import openTSNE
    kw = dict(neighbors="exact") if metric == "jaccard" else {}
    est = openTSNE.TSNE(random_state=seed, metric=metric, **kw)
    t0 = time.perf_counter()
    emb = est.fit(X_train)
    fit_secs = time.perf_counter() - t0
    t0 = time.perf_counter()
    Y_test = emb.transform(X_test)
    transform_secs = time.perf_counter() - t0
    return np.asarray(emb), np.asarray(Y_test), fit_secs, transform_secs


def run_pca2(X_train, X_test, seed, metric):
    est = PCA(n_components=2, random_state=seed)
    t0 = time.perf_counter()
    est.fit(X_train)
    fit_secs = time.perf_counter() - t0
    t0 = time.perf_counter()
    Y_test = est.transform(X_test)
    transform_secs = time.perf_counter() - t0
    return est.transform(X_train), Y_test, fit_secs, transform_secs


def umap_train_embedding(ds, rep, X_train, X_test, train_idx, test_idx, metric, meta):
    """The UMAP train embedding for this split, from cache or a fresh fit (cached)."""
    path = cell_path("umap", ds, rep)
    if os.path.exists(path):
        return np.load(path)["Y_train"]
    Y_train, Y_test, fit_secs, transform_secs = run_umap(X_train, X_test, rep, metric)
    save_cell("umap", ds, rep, train_idx, test_idx, Y_train, Y_test)
    record_timing(meta, ds, "umap", rep, fit_secs, transform_secs)
    save_meta(meta)
    return np.asarray(Y_train, np.float32)


def run_knnmap(X_train, X_test, Y_train_umap, metric):
    """15-NN of each test point among the train points in the input metric; the test
    embedding is the inverse-distance weighted average of their UMAP positions."""
    t0 = time.perf_counter()
    if metric == "jaccard":
        nn = NearestNeighbors(n_neighbors=K_NN, metric="jaccard").fit(X_train.astype(bool))
        dists, idx = nn.kneighbors(X_test.astype(bool))
    else:
        dists, idx = cKDTree(X_train).query(X_test, k=K_NN)
    fit_secs = time.perf_counter() - t0
    t0 = time.perf_counter()
    weights = 1.0 / (dists + 1e-9)
    weights /= weights.sum(1, keepdims=True)
    Y_test = (weights[..., None] * Y_train_umap[idx]).sum(1)
    transform_secs = time.perf_counter() - t0
    return Y_train_umap, Y_test, fit_secs, transform_secs


def invertibility(est, X_train):
    """Round-trip through the exact bijection: per-point relative L2 of
    inverse_coords(_forward(X)) vs X. Returns (median, max)."""
    Z = est._forward(X_train)
    X_rec = est.inverse_coords(Z)
    num = np.linalg.norm(X_rec - X_train, axis=1)
    den = np.maximum(np.linalg.norm(X_train, axis=1), 1e-12)
    rel = num / den
    return float(np.median(rel)), float(rel.max())


def jacobian_diag(est, X_train, n_points=32, chunk=4):
    """Condition number and log|det J| of the trained flow's forward, at n_points train
    points in latent space. Chunked so at most `chunk` Jacobians (D x D each) are live.
    Runs on the flow's device, falling back to a CPU copy on CUDA failure."""
    z = torch.from_numpy(est._to_coords(X_train[:n_points])).float()
    flow = est.flow_

    def compute(flow_, z_):
        def fwd(z1):
            return flow_(z1[None])[0]
        jac_fn = torch.func.vmap(torch.func.jacrev(fwd))
        kappas, logdets = [], []
        for start in range(0, len(z_), chunk):
            J = jac_fn(z_[start:start + chunk])            # (b, D, D)
            svals = torch.linalg.svdvals(J)
            kappas.append((svals[:, 0] / svals[:, -1]).cpu())
            logdets.append(torch.linalg.slogdet(J)[1].cpu())
            del J, svals
        return torch.cat(kappas), torch.cat(logdets)

    device = next(flow.parameters()).device
    try:
        kappa, logdet = compute(flow, z.to(device))
    except RuntimeError as e:
        if device.type == "cpu" or "memory" not in str(e).lower():
            raise
        import copy
        print(f"    jacobian on cuda failed ({e}); retrying on cpu", flush=True)
        torch.cuda.empty_cache()
        kappa, logdet = compute(copy.deepcopy(flow).cpu(), z)
    return dict(kappa_median=float(kappa.median()), kappa_max=float(kappa.max()),
                logdet_min=float(logdet.min()), logdet_max=float(logdet.max()),
                n_points=n_points)


def main():
    meta = load_meta()
    datasets = {}
    for ds in PAPER:
        X, y, title, note = PAPER[ds]()
        datasets[ds] = np.asarray(X, np.float32)
        print(f"loaded {ds}: X{list(X.shape)} ({note})", flush=True)

    for ds, X in datasets.items():
        metric = METRIC.get(ds, "euclidean")
        n = len(X)
        for rep in range(N_REPS):
            train_idx, test_idx = split(n, rep)
            X_train, X_test = X[train_idx], X[test_idx]

            # fuzzy kNN graph timing, once per dataset (it lives inside FloDR.fit)
            if rep == 0 and ds not in meta.get("fuzzy_knn_secs", {}):
                t0 = time.perf_counter()
                fuzzy_knn_graph(X_train, K_NN)
                meta.setdefault("fuzzy_knn_secs", {})[ds] = round(time.perf_counter() - t0, 3)
                save_meta(meta)
                print(f"[{ds}] fuzzy kNN graph: {meta['fuzzy_knn_secs'][ds]}s", flush=True)

            for method in ("flodr", "flodr_w0", "flodr_w1", "flodr_w3",
                           "umap", "opentsne", "pca2", "knnmap"):
                path = cell_path(method, ds, rep)
                if os.path.exists(path):
                    print(f"[{ds} rep{rep}] {method}: cached, skip", flush=True)
                    continue
                t_start = time.perf_counter()
                if method.startswith("flodr"):
                    w = float(method[7:]) if method.startswith("flodr_w") else 2.0
                    est, Y_train, Y_test, fit_secs, xform_secs = fit_flodr_safe(
                        X_train, X_test, rep, w)
                    device = next(est.flow_.parameters()).device.type
                    save_cell(method, ds, rep, train_idx, test_idx, Y_train, Y_test)
                    med_rel, max_rel = invertibility(est, X_train)
                    meta.setdefault("invertibility", {}).setdefault(
                        ds + (f"_w{method[7:]}" if w != 2.0 else ""), {})[str(rep)] = \
                        dict(median_rel=med_rel, max_rel=max_rel)
                    record_timing(meta, ds, method, rep, fit_secs, xform_secs, device)
                    save_meta(meta)
                    if rep == 0 and method == "flodr":
                        meta.setdefault("jacobian", {})[ds] = jacobian_diag(est, X_train)
                    del est
                    if torch is not None and device == "cuda":
                        torch.cuda.empty_cache()
                elif method == "knnmap":
                    Y_umap = umap_train_embedding(ds, rep, X_train, X_test,
                                                  train_idx, test_idx, metric, meta)
                    Y_train, Y_test, fit_secs, xform_secs = run_knnmap(
                        X_train, X_test, Y_umap, metric)
                    save_cell(method, ds, rep, train_idx, test_idx, Y_train, Y_test)
                    record_timing(meta, ds, method, rep, fit_secs, xform_secs)
                else:
                    fn = {"umap": run_umap, "opentsne": run_opentsne,
                          "pca2": run_pca2}[method]
                    Y_train, Y_test, fit_secs, xform_secs = fn(
                        X_train, X_test, rep, metric)
                    save_cell(method, ds, rep, train_idx, test_idx, Y_train, Y_test)
                    record_timing(meta, ds, method, rep, fit_secs, xform_secs)
                save_meta(meta)
                print(f"[{ds} rep{rep}] {method}: fit {fit_secs:.1f}s, transform "
                      f"{xform_secs:.2f}s (total {time.perf_counter() - t_start:.1f}s)",
                      flush=True)

    # CPU-only FloDR fit timing, last: mnist train split of rep 0. May take tens of minutes.
    if "flodr_cpu_fit_mnist_secs" not in meta:
        X = datasets["mnist"]
        train_idx, _ = split(len(X), 0)
        t0 = time.perf_counter()
        FloDR(w=2.0, random_state=0, device="cpu",
              advanced=dict(gate_max=0.5)).fit(X[train_idx])
        meta["flodr_cpu_fit_mnist_secs"] = round(time.perf_counter() - t0, 1)
        save_meta(meta)
        print(f"CPU FloDR fit (mnist rep0 train): {meta['flodr_cpu_fit_mnist_secs']}s",
              flush=True)

    print("done", flush=True)


if __name__ == "__main__":
    main()
