import os
import shutil
import subprocess
import urllib.request

import numpy as np

import common

# the published scores, the data sets and FloDR's embeddings and scores
DIR = os.environ.get("OPENPROBLEMS", os.path.join(common.CACHE, "openproblems"))
DIR = os.path.abspath(DIR)
# version 2 of the task: github.com/openproblems-bio/task_dimensionality_reduction at
# commit 360499d, its components run by viash in the docker images of the published run
TASK = os.path.join(DIR, "task_dimensionality_reduction")
S3 = "https://openproblems-data.s3.amazonaws.com/resources"
S3 += "/task_dimensionality_reduction"
RUN = "run_2024-12-19_15-35-59"
TAG = '.engines[.type == "docker"].target_tag := "build_main"'
# the data sets of the published run with at least two control values for every metric
DATASETS = [
    "openproblems_v1/mouse_blood_olsson_labelled",
    "openproblems_v1/mouse_hspc_nestorowa2016",
    "openproblems_v1/pancreas",
    "openproblems_v1/allen_brain_atlas",
    "openproblems_v1/zebrafish",
]
CONTROLS = ("random_features", "spectral_features", "true_features")
# the metrics by the component that computes them
FAMILIES = {
    "clustering": ["adjusted_rand_index", "normalized_mutual_information"],
    "co-ranking": [
        "continuity_at_k30",
        "trustworthiness_at_k30",
        "qnx_at_k30",
        "lcmc_at_k30",
        "qnx_auc",
        "qlocal",
        "qglobal",
    ],
    "density": ["density_preservation"],
    "distance correlations": [
        "centroid_distance_correlation",
        "label_distance_correlation",
        "waypoint_distance_correlation",
    ],
    "spectral": ["spectral_distance_correlation"],
}
COMPONENTS = ["clustering_performance", "coranking", "density_preservation"]
COMPONENTS += ["distance_correlation", "spectral_distance_correlation"]
METRICS = [k for ks in FAMILIES.values() for k in ks]
# FloDR at its defaults and with the density term, by its weight
VARIANTS = {"flodr": 0.0, "flodr_density": 2.0}
NAMES = {
    "flodr": "FloDR",
    "flodr_density": "FloDR, density term",
    "neuralee": "NeuralEE",
    "phate": "PHATE",
    "ivis": "ivis",
    "densmap": "densMAP",
    "tsne": "t-SNE",
    "lmds": "LMDS",
    "pymde": "PyMDE",
    "umap": "UMAP",
    "pca": "PCA",
    "diffusion_map": "Diffusion map",
    "simlr": "SIMLR",
}


def fetch(url, path):
    if not os.path.exists(path):
        os.makedirs(os.path.dirname(path), exist_ok=True)
        req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
        with urllib.request.urlopen(req) as r, open(path + ".part", "wb") as f:
            shutil.copyfileobj(r, f)
        os.replace(path + ".part", path)
    return path


def embed(X, w_dens):
    # the method as submitted to the task: FloDR on min(500, n - 1) principal components
    # of the normalised counts, enough for their distances to follow those of all genes
    import scanpy as sc
    from flodr import FloDR

    P = sc.tl.pca(X, n_comps=min(500, min(X.shape) - 1), svd_solver="arpack")
    advanced = {"w_dens": w_dens} if w_dens else None
    est = FloDR(random_state=0, verbose=False, advanced=advanced)
    return est.fit_transform(np.ascontiguousarray(P, dtype=np.float32))


def viash(component, **args):
    config = os.path.join(TASK, "src", component, "config.vsh.yaml")
    cmd = ["viash", "run", "-c", TAG, config, "--"]
    for k, v in args.items():
        cmd += [f"--{k}", v]
    # the R metric's reticulate downloads its own Python unless told to use the image's
    if component == "metrics/coranking":
        cmd += ["---docker_run_args", "-e RETICULATE_PYTHON=/usr/bin/python3"]
    cmd += ["---cpus", "8"]
    subprocess.run(cmd, check=True, cwd=TASK, stdout=subprocess.DEVNULL)


def _ranks(D, a):
    # the rank of every point as seen from points a, a + 1, ..., each of them first
    o = np.argsort(D, axis=1, kind="stable")
    rows = np.arange(D.shape[0])
    self_pos = np.argmax(o == (a + rows)[:, None], axis=1)
    R = np.empty_like(o)
    for r in rows:
        oi = np.concatenate(([a + r], np.delete(o[r], self_pos[r])))
        R[r, oi] = np.arange(D.shape[1])
    return R


def coranking(Dh, De, K=30, block=512):
    # the seven values of the task's co-ranking component (coRanking 0.2.5 in R), built
    # one column at a time so that only the co-ranking matrix is held. Ties between
    # distances are broken by index here and by std::sort in R
    names = FAMILIES["co-ranking"]
    if np.isnan(De).any():
        return dict.fromkeys(names, 0.0)
    N = len(Dh)
    Q = np.zeros((N - 1, N - 1), np.int32)
    ar = np.arange(N)
    for a in range(0, N, block):
        b = min(N, a + block)
        rh = _ranks(np.ascontiguousarray(Dh[:, a:b].T), a)
        rl = _ranks(np.ascontiguousarray(De[:, a:b].T), a)
        for r in range(b - a):
            keep = ar != a + r
            Q[rh[r, keep] - 1, rl[r, keep] - 1] += 1
    # the sum of the leading k x k block, grown one row and column at a time
    QT = np.ascontiguousarray(Q.T)
    blk, s = np.empty(N - 1, np.int64), 0
    for k in range(N - 1):
        s += int(Q[k, : k + 1].sum(dtype=np.int64)) + int(QT[k, :k].sum(dtype=np.int64))
        blk[k] = s
    ks = np.arange(1, N)
    qnx = blk / ks / N
    lcmc = qnx - ks / (N - 1)
    kmax = int(np.argmax(lcmc)) + 1
    G = N * K * (2 * N - 3 * K - 1) if K < N / 2 else N * (N - K) * (N - K - 1)
    far = np.arange(K + 1, N) - K
    trust = 1 - 2 / G * float(np.sum(far[:, None] * Q[K:, :K].astype(np.float64)))
    cont = 1 - 2 / G * float(np.sum(far[None, :] * Q[:K, K:].astype(np.float64)))
    values = [cont, trust, qnx[K - 1], lcmc[K - 1], qnx.mean(), qnx[:kmax].mean()]
    return dict(zip(names, map(float, values + [qnx[kmax:].mean()])))


def run(ds, method):
    # FloDR through the task on one data set: the fit, then the task's processing of the
    # embedding and its five metric components
    import anndata as ad
    import h5py

    data = os.path.join(DIR, "v2", "data", ds)
    work = os.path.join(DIR, "v2", "work", ds, method)
    for f in ("dataset.h5ad", "solution.h5ad"):
        fetch(f"{S3}/datasets/{ds}/log_cp10k/{f}", os.path.join(data, f))
    solution = os.path.join(data, "solution.h5ad")
    emb = os.path.join(work, "embedding.h5ad")
    processed = os.path.join(work, "processed.h5ad")
    if not os.path.exists(emb):
        a = ad.read_h5ad(os.path.join(data, "dataset.h5ad"))
        uns = {k: a.uns[k] for k in ("dataset_id", "normalization_id")}
        obsm = {"X_emb": embed(a.layers["normalized"], VARIANTS[method])}
        out = ad.AnnData(obs=a.obs[[]], obsm=obsm, uns={**uns, "method_id": "flodr"})
        os.makedirs(work, exist_ok=True)
        out.write_h5ad(emb, compression="gzip")
    if not os.path.exists(processed):
        kw = dict(input_embedding=emb, input_solution=solution, output=processed)
        viash("data_processors/process_embedding", **kw)
    with h5py.File(processed) as h:
        D = h["uns/between_waypoint_distances"][()]
    # the R component needs more than 62 GB from about 26 000 waypoints and stops on
    # distances that are not exactly symmetric; coranking() is within 1e-6 of it
    port = len(D) > 20_000 or not np.array_equal(D, D.T)
    del D
    out = {}
    for c in COMPONENTS:
        if c == "coranking" and port:
            with h5py.File(solution) as hs, h5py.File(processed) as hp:
                key = "uns/between_waypoint_distances"
                out.update(coranking(hs[key][()], hp[key][()]))
            continue
        f = os.path.join(work, f"score_{c}.h5ad")
        if not os.path.exists(f):
            kw = dict(input_embedding=processed, input_solution=solution, output=f)
            viash(f"metrics/{c}", **kw)
        u = ad.read_h5ad(f).uns
        out.update(zip(list(u["metric_ids"]), map(float, u["metric_values"])))
    return out


def scores():
    # per method, in the order of the tables, its scores per data set (rows) and metric
    # (columns), scaled between the lowest and the highest control as the task's report
    # does, nan where the method has no value. FloDR is run where its scores are
    # missing, the other methods are those of the published run
    import yaml

    path = os.path.join(DIR, "v2", "score_uns.yaml")
    fetch(f"{S3}/results/{RUN}/score_uns.yaml", path)
    values = {}
    for r in yaml.load(open(path), Loader=yaml.BaseLoader):
        per = values.setdefault(r["method_id"], {}).setdefault(r["dataset_id"], {})
        per.update(zip(r["metric_ids"], map(float, r["metric_values"])))
    path = os.path.join(DIR, "v2", "scores.json")
    for ds in DATASETS:
        ours = common.memo(path, ds, lambda: {m: run(ds, m) for m in VARIANTS})
        for m in VARIANTS:
            values.setdefault(m, {})[ds] = ours[m]

    def scaled(m, ds, k):
        v = values[m].get(ds, {}).get(k)
        cv = [values[c][ds][k] for c in CONTROLS if k in values[c].get(ds, {})]
        if v is None or not cv:
            return np.nan
        # as in R, a zero range gives +-inf, clipped later, or nan, counted as zero
        with np.errstate(divide="ignore", invalid="ignore"):
            return np.float64(v - min(cv)) / np.float64(max(cv) - min(cv))

    T = {
        m: np.array([[scaled(m, ds, k) for k in METRICS] for ds in DATASETS])
        for m in sorted(values)
        if m not in CONTROLS
    }
    rest = [m for m in T if m not in VARIANTS]
    rest.sort(key=lambda m: -official(T[m]).mean(1).mean())
    return {NAMES[m]: T[m] for m in (*VARIANTS, *rest)}


def official(S):
    # as on the leaderboard: clipped to [0, 1], a missing value counts as zero
    return np.clip(np.nan_to_num(S, nan=0.0, posinf=1.0, neginf=0.0), 0, 1)


def completed(S):
    # the mean over the values a method has, per data set and then over its data sets
    S = np.clip(np.nan_to_num(S, nan=np.nan, posinf=1.0, neginf=0.0), 0, 1)
    return float(np.mean([np.nanmean(r) for r in S if not np.isnan(r).all()]))
