import warnings

import numpy as np
from scipy.linalg import null_space
from scipy.optimize import curve_fit
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import laplacian
from scipy.sparse.linalg import eigsh
from sklearn.decomposition import PCA
from usearch.index import Index

warnings.filterwarnings("ignore")


def preprocess(
    X, n_components=50, seed=0, return_inverse=False, return_transform=False, whiten=1.0
):
    n_pcs = min(n_components, X.shape[1])
    pca = PCA(n_components=n_pcs, random_state=seed).fit(X.astype(np.float64))
    proj = pca.transform(X.astype(np.float64))
    mu, scale = proj.mean(0), (proj.std(0) + 1e-9) ** whiten
    X_std = ((proj - mu) / scale).astype(np.float32)
    out = [X_std]

    if return_inverse:
        out.append(lambda Z: pca.inverse_transform(np.asarray(Z) * scale + mu))

    if return_transform:
        out.append(
            lambda Xnew: (
                (pca.transform(np.asarray(Xnew).astype(np.float64)) - mu) / scale
            ).astype(np.float32)
        )

    return out[0] if len(out) == 1 else tuple(out)


def raw_coords(X, n_components=50, seed=0):
    X = np.asarray(X, dtype=np.float32)
    n_head = min(n_components, X.shape[1] - 1)

    pca = PCA(n_components=n_head, svd_solver="randomized", random_state=seed).fit(X)

    tail_v = null_space(pca.components_).astype(np.float32)
    proj = pca.transform(X)

    tail = (X - pca.mean_) @ tail_v

    head_mu, head_sd = proj.mean(0), proj.std(0) + 1e-9
    tail_sd = np.maximum(tail.std(0), 1e-2 * tail.std(0).max() + 1e-12)

    Z = np.hstack([(proj - head_mu) / head_sd, tail / tail_sd]).astype(np.float32)

    def to_raw(Zn):
        Zn = np.asarray(Zn, dtype=np.float32)
        head = Zn[:, :n_head] * head_sd + head_mu
        tail = Zn[:, n_head:] * tail_sd

        return (pca.mean_ + head @ pca.components_ + tail @ tail_v.T).astype(np.float32)

    def to_coords(Xn):
        Xn = np.asarray(Xn, dtype=np.float32)
        head = (pca.transform(Xn) - head_mu) / head_sd
        tail = ((Xn - pca.mean_) @ tail_v) / tail_sd

        return np.hstack([head, tail]).astype(np.float32)

    return Z, to_raw, to_coords


def spectral_init(ei, ej, w, n, scale=1.0, seed=0):
    graph = coo_matrix(
        (np.asarray(w, dtype=np.float64), (ei, ej)), shape=(n, n)
    ).tocsr()
    graph = graph.maximum(graph.T)

    lap = laplacian(graph, normed=True)

    try:
        k = 3
        _, vecs = eigsh(
            lap, k=k, sigma=-1e-3, which="LM", v0=np.ones(n) / np.sqrt(n), maxiter=n * 5
        )
        layout = vecs[:, 1:3]
    except Exception:
        layout = np.random.default_rng(seed).standard_normal((n, 2))

    layout = np.asarray(layout, dtype=np.float32)
    layout = layout - layout.mean(0)

    return (layout / (layout.std(0) + 1e-12) * scale).astype(np.float32)


EXACT_KNN_MAX = (
    50_000  # below this, brute force is both faster to trust and cheap enough
)


def knn_search(Xp, k, exact=None):
    X_f32 = np.ascontiguousarray(Xp, dtype=np.float32)
    n, n_dim = X_f32.shape

    if exact is None:
        exact = n <= EXACT_KNN_MAX

    if exact:
        try:  # faiss brute force: exact and deterministic
            import faiss  # even multithreaded; usearch fallback if absent

            index = faiss.IndexFlatL2(n_dim)
            index.add(X_f32)
            dist2, keys = index.search(X_f32, k + 1)
            keys, dist2 = keys.astype(np.int64), dist2.astype(np.float64)
            order = np.argsort(keys == np.arange(n)[:, None], axis=1, kind="stable")
            keys = np.take_along_axis(keys, order, axis=1)[:, :k]
            dist2 = np.take_along_axis(dist2, order, axis=1)[:, :k]
            return np.sqrt(np.clip(dist2, 0, None)), keys
        except ImportError:
            pass

    index = Index(ndim=n_dim, metric="l2sq", dtype="f32")

    # single-thread pin is for HNSW nondeterminism; the exact path is thread-count-independent
    # (verified bitwise-identical against threads=1) so it may use all cores
    index.add(np.arange(n), X_f32, threads=1 if not exact else 0)
    res = index.search(
        X_f32, k + 1, threads=1 if not exact else 0, exact=exact
    )  # k+1: drop self
    keys, dist2 = res.keys.astype(np.int64), res.distances.astype(np.float64)

    if exact:
        # lexsort to canonical (distance, index) tie order: multithreaded exact search returns
        # ties in nondeterministic order (0.03% of MNIST rows), flipping graph edges run-to-run
        order = np.lexsort((keys, dist2), axis=1)
        keys = np.take_along_axis(keys, order, axis=1)
        dist2 = np.take_along_axis(dist2, order, axis=1)

    order = np.argsort(
        keys == np.arange(n)[:, None], axis=1, kind="stable"
    )  # push self last
    keys = np.take_along_axis(keys, order, axis=1)[:, :k]
    dist2 = np.take_along_axis(dist2, order, axis=1)[:, :k]

    return np.sqrt(dist2), keys


def fuzzy_knn_graph(Xp, k, n_iter=64, precomputed=None, return_dist=False):
    n = len(Xp)

    if precomputed is not None:
        dist, idx = precomputed
        if dist.shape != (n, k) or idx.shape != (n, k):
            raise ValueError(
                f"precomputed must be two ({n},{k}) arrays, got "
                f"{dist.shape} and {idx.shape}"
            )
    else:
        dist, idx = knn_search(Xp, k)

    rho = dist[:, 0]
    d_clip = np.clip(dist - rho[:, None], 0, None)
    target = np.log2(k)
    lo, hi, sigma = np.zeros(n), np.full(n, np.inf), np.ones(n)

    for _ in range(n_iter):
        over = np.exp(-d_clip / sigma[:, None]).sum(1) > target
        hi[over], lo[~over] = sigma[over], sigma[~over]
        sigma = np.where(over | np.isfinite(hi), (lo + hi) / 2, sigma * 2)

    w_attr = np.exp(-d_clip / sigma[:, None])
    rows = np.repeat(np.arange(n), k)
    A = coo_matrix((w_attr.ravel(), (rows, idx.ravel())), shape=(n, n)).tocsr()
    B = (A + A.T - A.multiply(A.T)).tocoo()
    mask = B.row < B.col  # unique undirected edges
    out = (
        idx,
        B.row[mask].astype(np.int64),
        B.col[mask].astype(np.int64),
        B.data[mask].astype(np.float32),
    )

    return out + (dist,) if return_dist else out


def find_ab(min_dist=0.1, spread=1.0):
    x_grid = np.linspace(0, spread * 3, 300)
    y_prof = np.where(x_grid < min_dist, 1.0, np.exp(-(x_grid - min_dist) / spread))
    (a, b), _ = curve_fit(
        lambda x, a, b: 1.0 / (1.0 + a * x ** (2 * b)),
        x_grid,
        y_prof,
        p0=[1.0, 1.0],
        maxfev=10000,
    )

    return float(a), float(b)


def global_pairs(Xp, count, rng):
    n = len(Xp)
    pair_i = rng.integers(0, n, count)
    pair_j = rng.integers(0, n, count)

    mask = pair_i != pair_j
    pair_i, pair_j = pair_i[mask], pair_j[mask]
    dist_hd = np.sqrt(((Xp[pair_i] - Xp[pair_j]) ** 2).sum(1))

    return pair_i, pair_j, dist_hd
