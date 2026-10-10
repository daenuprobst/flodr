import warnings

import numpy as np
from scipy.linalg import null_space
from scipy.optimize import curve_fit
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import laplacian
from scipy.sparse.linalg import eigsh
from scipy.spatial import cKDTree
from sklearn.decomposition import PCA
from sklearn.neighbors import NearestNeighbors
from usearch.index import Index


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


# a class rather than closures, so a fitted model pickles
class Whitening:
    def __init__(self, pca, n_head, head_mu, head_sd, tail_v, tail_sd):
        self.pca, self.n_head = pca, n_head
        self.head_mu, self.head_sd = head_mu, head_sd
        self.tail_v, self.tail_sd = tail_v, tail_sd

    def to_raw(self, Zn):
        Zn = np.asarray(Zn, dtype=np.float32)
        head = Zn[:, : self.n_head] * self.head_sd + self.head_mu
        tail = Zn[:, self.n_head :] * self.tail_sd

        return (
            self.pca.mean_ + head @ self.pca.components_ + tail @ self.tail_v.T
        ).astype(np.float32)

    def to_coords(self, Xn):
        Xn = np.asarray(Xn, dtype=np.float32)
        head = (self.pca.transform(Xn) - self.head_mu) / self.head_sd
        tail = ((Xn - self.pca.mean_) @ self.tail_v) / self.tail_sd

        return np.hstack([head, tail]).astype(np.float32)


# R is orthogonal, so to_raw undoes the rotation and the inverse stays exact
class RotatedWhitening:
    def __init__(self, whitening, R):
        self.whitening, self.R = whitening, np.asarray(R, dtype=np.float32)
        self.n_head = self.R.shape[0]

    def to_raw(self, Zn):
        Zn = np.asarray(Zn, dtype=np.float32)
        h = self.n_head

        return self.whitening.to_raw(np.hstack([Zn[:, :h] @ self.R.T, Zn[:, h:]]))

    def to_coords(self, Xn):
        Z = self.whitening.to_coords(Xn)
        h = self.n_head

        return np.hstack([Z[:, :h] @ self.R, Z[:, h:]]).astype(np.float32)


def raw_coords(X, n_components=50, seed=0):
    X = np.asarray(X, dtype=np.float32)

    # PCA cannot return more components than it has samples
    n_head = min(n_components, X.shape[1] - 1, X.shape[0] - 1)

    pca = PCA(n_components=n_head, svd_solver="randomized", random_state=seed).fit(X)

    tail_v = null_space(pca.components_).astype(np.float32)
    proj = pca.transform(X)

    tail = (X - pca.mean_) @ tail_v

    head_mu, head_sd = proj.mean(0), proj.std(0) + 1e-9
    tail_sd = np.maximum(tail.std(0), 1e-2 * tail.std(0).max() + 1e-12)

    Z = np.hstack([(proj - head_mu) / head_sd, tail / tail_sd]).astype(np.float32)
    wh = Whitening(pca, n_head, head_mu, head_sd, tail_v, tail_sd)

    return Z, wh.to_raw, wh.to_coords


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


EXACT_KNN_MAX = 50_000


def knn_query(Xref, Xq, k, metric="euclid"):
    self_q = Xq is None

    if metric == "jaccard":
        nn = NearestNeighbors(n_neighbors=k, metric="jaccard").fit(
            np.asarray(Xref, dtype=bool)
        )
        # kneighbors(None) already leaves each point out of its own list
        dist, idx = nn.kneighbors(None if self_q else np.asarray(Xq, dtype=bool))

        return np.asarray(dist, dtype=np.float64), np.asarray(idx, dtype=np.int64)
    else:
        # exact, so the thread count cannot change the answer; the fit's thread count,
        # not every logical CPU
        dist, idx = cKDTree(Xref).query(
            Xref if self_q else Xq, k=k + self_q, workers=_threads()
        )

    if self_q:
        dist, idx = dist[:, 1:], idx[:, 1:]

    return np.asarray(dist, dtype=np.float64), np.asarray(idx, dtype=np.int64)


def _canonical(keys, dist2, k):
    # (distance, index) order, then self pushed last and dropped
    n = len(keys)
    order = np.lexsort((keys, dist2), axis=1)
    keys = np.take_along_axis(keys, order, axis=1)
    dist2 = np.take_along_axis(dist2, order, axis=1)
    order = np.argsort(keys == np.arange(n)[:, None], axis=1, kind="stable")
    keys = np.take_along_axis(keys, order, axis=1)[:, :k]
    dist2 = np.take_along_axis(dist2, order, axis=1)[:, :k]

    return np.sqrt(dist2), keys


def _gpu_blocks(X, Xc, sq, c, cells, rq):
    import torch

    n = len(X)
    br = min(n, max(256, cells // 4096))
    bq = min(n, max(1, cells // br))
    keys = np.empty((n, c), np.int64)
    dist2 = np.empty((n, c), np.float64)

    for qs in range(0, n, bq):
        q = Xc[qs : qs + bq]
        best_d = best_i = None

        for rs in range(0, n, br):
            # |q|^2 is the same along a row, so it is left out of the ranking
            blk = torch.addmm(sq[None, rs : rs + br], q, Xc[rs : rs + br].T, alpha=-2.0)
            v, i = torch.topk(
                blk, min(c, blk.shape[1]), dim=1, largest=False, sorted=False
            )
            i = i + rs

            if best_d is not None:
                v, i = torch.cat([best_d, v], 1), torch.cat([best_i, i], 1)
                v, j = torch.topk(v, c, dim=1, largest=False, sorted=False)
                i = torch.gather(i, 1, j)

            best_d, best_i = v, i

        for s in range(0, len(q), rq):
            rows = slice(qs + s, qs + s + min(rq, len(q) - s))
            cand = best_i[s : s + rq]
            diff = X[rows][:, None, :].double() - X[cand].double()
            keys[rows] = cand.cpu().numpy()
            dist2[rows] = (diff**2).sum(-1).cpu().numpy()

    return keys, dist2


def knn_gpu(Xp, k, device="cuda", extra=16, cells=None):
    # fp32 GEMM on centred data picks k + 1 + extra candidates, re-ranked in float64:
    # exact unless fp32 rounding exceeds the gap from the k-th to the last candidate
    import torch

    dev = torch.device(device)
    X = torch.as_tensor(np.ascontiguousarray(Xp, dtype=np.float32), device=dev)
    n, n_dim = X.shape
    c = min(n, k + 1 + extra)
    Xc = X - X.mean(0, dtype=torch.float64).float()
    sq = (Xc * Xc).sum(1)

    # buffers take at most a quarter of free memory (24 bytes: the re-rank's three
    # float64 arrays); out of memory, retry at a quarter of the size
    for attempt in range(4):
        free = torch.cuda.mem_get_info(dev)[0] if dev.type == "cuda" else 2**32
        budget = max(2**24, free // 4) >> (2 * attempt)
        block = cells or max(2**20, min(budget // 2, 2**30) // 4)
        rq = max(1, min(2**28, budget // 2) // (24 * c * n_dim))

        try:
            with torch.no_grad():
                tf32 = torch.backends.cuda.matmul.allow_tf32
                torch.backends.cuda.matmul.allow_tf32 = False

                try:
                    keys, dist2 = _gpu_blocks(X, Xc, sq, c, block, rq)
                finally:
                    torch.backends.cuda.matmul.allow_tf32 = tf32

            return _canonical(keys, dist2, k)
        except torch.cuda.OutOfMemoryError:
            if attempt == 3 or cells:
                raise

            torch.cuda.empty_cache()


def _on_gpu(device):
    import torch

    if device is None:
        return torch.cuda.is_available()

    return str(device).startswith("cuda")


def _threads():
    from .train import default_threads

    return default_threads()


def knn_search(Xp, k, exact=None, device=None, info=None):
    X_f32 = np.ascontiguousarray(Xp, dtype=np.float32)
    n, n_dim = X_f32.shape
    info = {} if info is None else info

    if exact is not False and _on_gpu(device):
        try:
            out = knn_gpu(X_f32, k, "cuda" if device is None else device)
            info["knn"] = "gpu_exact"
            return out
        except RuntimeError as err:
            warnings.warn(
                f"the GPU kNN failed ({str(err).splitlines()[0][:160]}); searching on the CPU, "
                f"exact up to {EXACT_KNN_MAX} points and approximate (HNSW) above",
                RuntimeWarning,
                stacklevel=2,
            )

    if exact is None:
        exact = n <= EXACT_KNN_MAX

    if exact:
        try:
            # exact and deterministic even multithreaded
            import faiss

            index = faiss.IndexFlatL2(n_dim)
            index.add(X_f32)
            dist2, keys = index.search(X_f32, k + 1)
            keys, dist2 = keys.astype(np.int64), dist2.astype(np.float64)
            order = np.argsort(keys == np.arange(n)[:, None], axis=1, kind="stable")
            keys = np.take_along_axis(keys, order, axis=1)[:, :k]
            dist2 = np.take_along_axis(dist2, order, axis=1)[:, :k]
            info["knn"] = "faiss_exact"
            return np.sqrt(np.clip(dist2, 0, None)), keys
        except ImportError:
            pass

    index = Index(ndim=n_dim, metric="l2sq", dtype="f32")
    info["knn"] = "usearch_exact" if exact else "hnsw"

    # HNSW is nondeterministic across threads, so it runs on one
    threads = _threads() if exact else 1
    index.add(np.arange(n), X_f32, threads=threads)
    res = index.search(X_f32, k + 1, threads=threads, exact=exact)
    keys, dist2 = res.keys.astype(np.int64), res.distances.astype(np.float64)

    if exact:
        # multithreaded exact search returns ties in any order, which flips graph edges
        # from run to run
        order = np.lexsort((keys, dist2), axis=1)
        keys = np.take_along_axis(keys, order, axis=1)
        dist2 = np.take_along_axis(dist2, order, axis=1)

    # push self last
    order = np.argsort(keys == np.arange(n)[:, None], axis=1, kind="stable")
    keys = np.take_along_axis(keys, order, axis=1)[:, :k]
    dist2 = np.take_along_axis(dist2, order, axis=1)[:, :k]

    return np.sqrt(dist2), keys


def fuzzy_knn_graph(
    Xp, k, n_iter=64, precomputed=None, return_dist=False, device=None, info=None
):
    n = len(Xp)

    if precomputed is not None:
        dist, idx = precomputed
        if dist.shape != (n, k) or idx.shape != (n, k):
            raise ValueError(
                f"precomputed must be two ({n},{k}) arrays, got "
                f"{dist.shape} and {idx.shape}"
            )
    else:
        dist, idx = knn_search(Xp, k, device=device, info=info)

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

    mask = B.row < B.col
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
