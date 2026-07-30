import copy
import json
import os
import time

import numpy as np
from scipy.spatial import cKDTree
from sklearn.decomposition import PCA
from sklearn.neighbors import NearestNeighbors

from datasets import PAPER, METRIC
from flodr import FloDR
from flodr.data import find_ab, fuzzy_knn_graph

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
    # every fit compiles fresh closures, and a full sweep in one process blows through
    # the default limit of 8, which fullgraph=True turns into a hard failure
    torch._dynamo.config.cache_size_limit = 128
except ImportError:
    torch = None
    DEVICE = "cpu"


def split(n, rep):
    perm = np.random.default_rng(10_000 + rep).permutation(n)
    # train_idx, test_idx
    return perm[n // 5:], perm[: n // 5]


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
    path = cell_path("umap", ds, rep)
    if os.path.exists(path):
        return np.load(path)["Y_train"]
    Y_train, Y_test, fit_secs, transform_secs = run_umap(X_train, X_test, rep, metric)
    save_cell("umap", ds, rep, train_idx, test_idx, Y_train, Y_test)
    record_timing(meta, ds, "umap", rep, fit_secs, transform_secs)
    save_meta(meta)
    return np.asarray(Y_train, np.float32)


def knn_train(X_train, X_test, metric):
    if metric == "jaccard":
        nn = NearestNeighbors(n_neighbors=K_NN, metric="jaccard").fit(X_train.astype(bool))
        return nn.kneighbors(X_test.astype(bool))
    return cKDTree(X_train).query(X_test, k=K_NN)


def run_knnmap(X_train, X_test, Y_train_umap, metric):
    t0 = time.perf_counter()
    dists, idx = knn_train(X_train, X_test, metric)
    fit_secs = time.perf_counter() - t0
    t0 = time.perf_counter()
    weights = 1.0 / (dists + 1e-9)
    weights /= weights.sum(1, keepdims=True)
    Y_test = (weights[..., None] * Y_train_umap[idx]).sum(1)
    transform_secs = time.perf_counter() - t0
    return Y_train_umap, Y_test, fit_secs, transform_secs


# name -> (anchor cell, repulsion, steps, lr, datasets to restrict to or None)
OPT_ARMS = {"flodr_opt": ("flodr", True, 200, 0.05, None),
            "flodr_optA": ("flodr", False, 200, 0.05, None),
            "optA_long": ("flodr", False, 500, 0.025, ("mnist", "fmnist"))}


def run_transform_opt(Y_train, y0, nn_idx, seed, repulsion=True, steps=200, lr=0.05,
                      n_neg=5):
    a, b = find_ab(0.01)
    dev = torch.device(DEVICE if torch is not None else "cpu")
    Y_train_t = torch.from_numpy(np.ascontiguousarray(Y_train, np.float32)).to(dev)
    y = torch.from_numpy(np.ascontiguousarray(y0, np.float32)).to(dev).requires_grad_(True)
    # (m, 15)
    nn_t = torch.from_numpy(np.asarray(nn_idx)).long().to(dev)
    m = len(y)
    gen = torch.Generator(device=dev).manual_seed(20_000 + seed)
    opt = torch.optim.Adam([y], lr=lr)

    def q(d2):
        return 1.0 / (1.0 + a * (d2 + 1e-6) ** b)

    for _ in range(steps):
        # (m, 15)
        d2 = ((y[:, None, :] - Y_train_t[nn_t]) ** 2).sum(-1)
        loss = -torch.log(q(d2) + 1e-6).mean()
        if repulsion:
            neg = torch.randint(0, len(Y_train_t), (m, n_neg), device=dev, generator=gen)
            d2n = ((y[:, None, :] - Y_train_t[neg]) ** 2).sum(-1)
            loss = loss - torch.log(1.0 - q(d2n) + 1e-6).mean()
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()
    return y.detach().cpu().numpy()


def invertibility(est, X_train):
    Z = est._forward(X_train)
    X_rec = est.inverse_coords(Z)
    num = np.linalg.norm(X_rec - X_train, axis=1)
    den = np.maximum(np.linalg.norm(X_train, axis=1), 1e-12)
    rel = num / den
    return float(np.median(rel)), float(rel.max())


def jacobian_diag(est, X_train, n_points=32, chunk=4):
    z = torch.from_numpy(est._to_coords(X_train[:n_points])).float()
    flow = est.flow_

    def compute(flow_, z_):
        def fwd(z1):
            return flow_(z1[None])[0]
        jac_fn = torch.func.vmap(torch.func.jacrev(fwd))
        kappas, logdets = [], []
        for start in range(0, len(z_), chunk):
            # (b, D, D)
            J = jac_fn(z_[start:start + chunk])
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

            # time the fuzzy kNN graph once per dataset, since it lives inside FloDR.fit
            if rep == 0 and ds not in meta.get("fuzzy_knn_secs", {}):
                t0 = time.perf_counter()
                fuzzy_knn_graph(X_train, K_NN)
                meta.setdefault("fuzzy_knn_secs", {})[ds] = round(time.perf_counter() - t0, 3)
                save_meta(meta)
                print(f"[{ds}] fuzzy kNN graph: {meta['fuzzy_knn_secs'][ds]}s", flush=True)

            for method in ("flodr", "flodr_w0", "flodr_w1", "flodr_w3",
                           "umap", "opentsne", "pca2", "knnmap", *OPT_ARMS):
                path = cell_path(method, ds, rep)
                if os.path.exists(path):
                    print(f"[{ds} rep{rep}] {method}: cached, skip", flush=True)
                    continue
                t_start = time.perf_counter()
                if method in OPT_ARMS:
                    # placement against a cached FloDR cell's frozen train embedding,
                    # initialised at that cell's forward pass, and never refits
                    src, repu, stp, lr_, only = OPT_ARMS[method]
                    if only and ds not in only:
                        continue
                    zf = np.load(cell_path(src, ds, rep))
                    t0 = time.perf_counter()
                    _, idx_nn = knn_train(X_train, X_test, metric)
                    fit_secs = time.perf_counter() - t0
                    t0 = time.perf_counter()
                    Y_test = run_transform_opt(zf["Y_train"], zf["Y_test"], idx_nn,
                                               seed=rep, repulsion=repu, steps=stp, lr=lr_)
                    xform_secs = time.perf_counter() - t0
                    save_cell(method, ds, rep, train_idx, test_idx, zf["Y_train"], Y_test)
                    record_timing(meta, ds, method, rep, fit_secs, xform_secs)
                elif method.startswith("flodr"):
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

    # CPU-only fit timing, last because it can take tens of minutes
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
