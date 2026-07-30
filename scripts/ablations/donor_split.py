import json
import math
import os
import sys
import time

import numpy as np
import torch
from scipy.stats import spearmanr
from sklearn.decomposition import PCA

HERE = os.path.dirname(os.path.abspath(__file__))
SCRIPTS = os.path.dirname(HERE)
ROOT = os.path.dirname(SCRIPTS)
sys.path.insert(0, os.path.join(ROOT, "src"))

from flodr import FloDR  # noqa: E402

CACHE = os.path.join(SCRIPTS, "cache")
OUT = os.path.join(CACHE, "donor_split.json")
ATLAS = {
    "bmarrow": os.path.join(ROOT, "data", "scrna", "fetal_bone_marrow.h5ad"),
    "cerebellum": os.path.join(ROOT, "data", "scrna", "cerebellum.h5ad"),
}
REPS = (0, 1, 2)
TRANSFORM_CHUNK = 20_000


def preprocess(atlas):
    npz = os.path.join(CACHE, f"donor_split_preproc_{atlas}.npz")
    if os.path.exists(npz):
        data = np.load(npz, allow_pickle=True)
        return data["P"].astype(np.float32), data["donors"].astype(str)
    import scanpy as sc

    adata = sc.read_h5ad(ATLAS[atlas])
    donors = adata.obs["donor_id"].astype(str).to_numpy()
    sc.pp.normalize_total(adata, target_sum=1e4)
    sc.pp.log1p(adata)
    sc.pp.highly_variable_genes(adata, n_top_genes=2000)
    adata = adata[:, adata.var.highly_variable].copy()
    sc.pp.scale(adata, max_value=10)
    pcs = PCA(n_components=50, svd_solver="randomized", random_state=0).fit_transform(adata.X)
    pcs = pcs.astype(np.float32)
    pcs = (pcs - pcs.mean(0)) / (pcs.std(0) + 1e-12)
    P = pcs.astype(np.float32)
    np.savez_compressed(npz, P=P, donors=donors)
    return P, donors


def donor_split(donors, rep):
    uniq = np.array(sorted(set(donors.tolist())))
    rng = np.random.default_rng(20_000 + rep)
    perm = uniq[rng.permutation(len(uniq))]
    n_test = math.ceil(len(uniq) / 5)
    test_donors = set(perm[:n_test].tolist())
    test_mask = np.array([donor in test_donors for donor in donors])
    return ~test_mask, test_mask, sorted(test_donors)


def knn15(X, query):
    import faiss

    X = np.ascontiguousarray(X, dtype=np.float32)
    index = faiss.IndexFlatL2(X.shape[1])
    index.add(X)
    _, nbrs = index.search(np.ascontiguousarray(query, dtype=np.float32), 16)
    # the query row is in X at distance 0, but not always first when there are ties
    row_idx = np.arange(len(query))[:, None]
    keep = nbrs != row_idx
    out = np.empty((len(query), 15), dtype=np.int64)
    for row in range(len(query)):
        out[row] = nbrs[row][keep[row]][:15]
    return out


def recall_at_15(P, emb2d, test_idx):
    nn_in = knn15(P, P[test_idx])
    nn_em = knn15(emb2d, emb2d[test_idx])
    inter = [len(set(in_nn) & set(em_nn)) for in_nn, em_nn in zip(nn_in, nn_em)]
    return float(np.mean(inter) / 15.0)


def cpd(P, emb2d, test_idx):
    rng = np.random.default_rng(777)
    n = P.shape[0]
    if len(test_idx) * 20 > 100_000:
        test_idx = rng.choice(test_idx, size=100_000 // 20, replace=False)
    pair_i = np.repeat(test_idx, 20)
    pair_j = rng.integers(0, n, size=len(pair_i))
    d_in = np.linalg.norm(P[pair_i] - P[pair_j], axis=1)
    d_em = np.linalg.norm(emb2d[pair_i] - emb2d[pair_j], axis=1)
    return float(spearmanr(d_in, d_em).statistic)


def chunked_transform(model, X):
    outs = [model.transform(X[start:start + TRANSFORM_CHUNK])
            for start in range(0, len(X), TRANSFORM_CHUNK)]
    return np.concatenate(outs).astype(np.float32)


def run_cell(atlas, method, rep, P, train_mask, test_mask):
    X_train, X_test = P[train_mask], P[test_mask]
    if method == "flodr":
        def build(device):
            return FloDR(w=2.0, random_state=rep, device=device,
                         advanced=dict(gate_max=0.5, edge_batch=32768))

        model = build("cuda")
        t0 = time.perf_counter()
        try:
            model.fit(X_train)
        except RuntimeError as e:
            if "out of memory" not in str(e).lower():
                raise
            print(f"[{atlas}/flodr/{rep}] CUDA OOM, retrying on cpu", flush=True)
            torch.cuda.empty_cache()
            model = build("cpu")
            t0 = time.perf_counter()
            model.fit(X_train)
        fit_secs = time.perf_counter() - t0
    else:
        import umap

        model = umap.UMAP(n_neighbors=15, random_state=rep)
        t0 = time.perf_counter()
        model.fit(X_train)
        fit_secs = time.perf_counter() - t0
    t0 = time.perf_counter()
    emb_test = chunked_transform(model, X_test)
    transform_secs = time.perf_counter() - t0

    emb_train = model.embedding_.astype(np.float32)
    emb_all = np.concatenate([emb_train, emb_test])
    # same row order as emb_all
    P_all = np.concatenate([X_train, X_test])
    assert P_all.shape[0] == emb_all.shape[0]
    test_rows = np.arange(len(emb_train), len(emb_all))
    return {
        "fit_secs": round(fit_secs, 2),
        "transform_secs": round(transform_secs, 2),
        "recall@15": round(recall_at_15(P_all, emb_all, test_rows), 4),
        "cpd": round(cpd(P_all, emb_all, test_rows), 4),
    }


def main():
    res = json.load(open(OUT)) if os.path.exists(OUT) else {}
    for atlas in ("bmarrow", "cerebellum"):
        P, donors = preprocess(atlas)
        for rep in REPS:
            train_mask, test_mask, test_donors = donor_split(donors, rep)
            for method in ("flodr", "umap"):
                key = f"{atlas}/{method}/{rep}"
                if key in res:
                    print(f"[{key}] cached, skipping", flush=True)
                    continue
                print(f"[{key}] n_train={train_mask.sum()} n_test={test_mask.sum()}",
                      flush=True)
                result = run_cell(atlas, method, rep, P, train_mask, test_mask)
                result.update(n_train=int(train_mask.sum()), n_test=int(test_mask.sum()),
                              n_donors_total=int(len(set(donors.tolist()))),
                              test_donors=test_donors)
                res[key] = result
                json.dump(res, open(OUT, "w"), indent=1)
                print(f"[{key}] {result}", flush=True)

    print("\nsummary (mean +/- std over reps)")
    for atlas in ("bmarrow", "cerebellum"):
        for method in ("flodr", "umap"):
            cells = [res[f"{atlas}/{method}/{rep}"] for rep in REPS]
            for metric in ("recall@15", "cpd"):
                vals = [cell[metric] for cell in cells]
                print(f"{atlas:10s} {method:5s} {metric:9s} "
                      f"{np.mean(vals):.4f} +/- {np.std(vals):.4f}   {vals}")
            fits = [cell["fit_secs"] for cell in cells]
            trans = [cell["transform_secs"] for cell in cells]
            print(f"{atlas:10s} {method:5s} fit {np.mean(fits):.1f}s, "
                  f"transform {np.mean(trans):.1f}s (means over reps)")


if __name__ == "__main__":
    main()
