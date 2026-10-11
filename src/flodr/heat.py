import numpy as np

# HeatGeo (Huguet et al., arXiv:2305.19043) at its class defaults; only the first few
# coordinates are used, to pick the directions the layout starts from
TAU, ORDER, LANDMARKS, BANDWIDTH, DECAY = 10.0, 32, 500, 4, 40.0


def _landmarks(dist, idx, n_land, seed, rule="random", X=None):
    # uniform landmarks follow cell mass; farthest points also cover rare branches
    from scipy.sparse import csr_matrix
    from scipy.sparse.csgraph import connected_components

    n, k = idx.shape
    G = csr_matrix(
        (np.ones(n * k), (np.repeat(np.arange(n), k), idx.ravel())), shape=(n, n)
    )
    _, lab = connected_components(G.maximum(G.T), directed=False)
    pool = np.flatnonzero(lab == np.bincount(lab).argmax())
    rng = np.random.default_rng(seed)
    n_land = min(n_land, len(pool))

    if rule == "random" or X is None:
        return rng.choice(pool, n_land, replace=False)

    Z = np.asarray(X, dtype=np.float64)[pool]
    pick = [int(rng.integers(len(pool)))]
    dmin = ((Z - Z[pick[0]]) ** 2).sum(1)

    for _ in range(n_land - 1):
        pick.append(int(np.argmax(dmin)))
        dmin = np.minimum(dmin, ((Z - Z[pick[-1]]) ** 2).sum(1))

    return pool[np.array(pick)]


def _coefficients():
    from scipy.special import ive

    # exp(-tau x) on [0, 2] in Chebyshev polynomials of x - 1: 2 e^-tau (-1)^k I_k(tau)
    return 2 * ive(np.arange(ORDER + 1), TAU) * (-1.0) ** np.arange(ORDER + 1)


def _kernel(dist):
    # the 5th neighbour, or the nearest positive distance where exact copies make it 0
    bw = dist[:, min(BANDWIDTH, dist.shape[1] - 1)]
    pos = np.where(dist > 0, dist, np.inf).min(1)
    bw = np.where(bw > 0, bw, np.where(np.isfinite(pos), pos, 1.0))

    with np.errstate(over="ignore"):
        w = np.exp(-((dist / bw[:, None]) ** DECAY))
    w[w < 1e-4] = 0

    return w


def _heat_cpu(dist, idx, lm, cols=64):
    from scipy.sparse import csr_matrix

    n, k = idx.shape
    rows = np.repeat(np.arange(n), k)
    K = csr_matrix((_kernel(dist).ravel(), (rows, idx.ravel())), shape=(n, n))
    K = ((K + K.T) / 2).tocsr()
    K.setdiag(1.0)
    s = 1 / np.sqrt(np.asarray(K.sum(1)).ravel())
    # L - I for the normalised Laplacian L
    Y = (-(K.multiply(s[:, None]).multiply(s[None, :]))).tocsr()
    c = _coefficients()
    H = np.empty((len(lm), n), dtype=np.float64)

    for a in range(0, len(lm), cols):
        b = min(len(lm), a + cols)
        T0 = np.zeros((n, b - a))
        T0[lm[a:b], np.arange(b - a)] = 1.0
        T1 = Y @ T0
        acc = 0.5 * c[0] * T0 + c[1] * T1

        for j in range(2, ORDER + 1):
            T0, T1 = T1, 2 * (Y @ T1) - T0
            acc += c[j] * T1

        H[a:b] = np.maximum(acc.T, 0)

    return H


def _heat_gpu(dist, idx, lm, device, cols=64):
    import torch

    dev = torch.device(device)
    n, k = idx.shape
    c = _coefficients()

    with torch.no_grad():
        w = torch.as_tensor(_kernel(dist).ravel() / 2, device=dev, dtype=torch.float64)
        r = torch.as_tensor(np.repeat(np.arange(n), k), device=dev)
        j = torch.as_tensor(idx.ravel(), device=dev)
        K = torch.sparse_coo_tensor(
            torch.stack([torch.cat([r, j]), torch.cat([j, r])]),
            torch.cat([w, w]),
            (n, n),
        ).coalesce()
        # the + 1 is the unit diagonal, as setdiag(1.0) in _heat_cpu
        s = (torch.sparse.sum(K, 1).to_dense() + 1).rsqrt()[:, None]
        K = K.to_sparse_csr()
        lm_t = torch.as_tensor(lm, device=dev)
        H = np.empty((len(lm), n), dtype=np.float64)

        def step(B):
            return -(s * torch.sparse.mm(K, s * B) + s * s * B)

        for a in range(0, len(lm), cols):
            b = min(len(lm), a + cols)
            T0 = torch.zeros((n, b - a), dtype=torch.float64, device=dev)
            T0[lm_t[a:b], torch.arange(b - a, device=dev)] = 1
            T1 = step(T0)
            acc = 0.5 * c[0] * T0 + c[1] * T1

            for q in range(2, ORDER + 1):
                T0, T1 = T1, 2 * step(T1) - T0
                acc += c[q] * T1

            H[a:b] = acc.clamp_min(0).T.cpu().numpy()

    return H


def _lmds(D2, lm, m):
    # D2 is overwritten in place to save a copy of an (n, landmarks) array
    sub = D2[:, lm]
    J = np.eye(len(lm)) - 1.0 / len(lm)
    ev, V = np.linalg.eigh(-0.5 * J @ sub @ J)
    o = np.argsort(ev)[::-1][:m]
    ev, V = ev[o], V[:, o]
    keep = ev > 0
    ev, V = ev[keep], V[:, keep]
    D2 -= sub.mean(1, keepdims=True)

    return -0.5 * ((V / np.sqrt(ev)).T @ D2).T


def heat_coordinates(dist, idx, m=2, device=None, seed=0, landmarks="random", X=None):
    # dist, idx: (n, k) kNN lists without self, nearest first
    dist = np.asarray(dist, dtype=np.float64)
    idx = np.asarray(idx, dtype=np.int64)
    lm = _landmarks(dist, idx, LANDMARKS, seed, landmarks, X)
    gpu = device is not None and str(device).startswith("cuda")
    H = _heat_gpu(dist, idx, lm, device) if gpu else _heat_cpu(dist, idx, lm)
    # squared heat-geodesic distances d^2 = -4 tau log(h), in place
    H += 1e-16
    np.log(H, out=H)
    H *= -4 * TAU
    np.maximum(H, 0, out=H)

    return _lmds(H, lm, m)


def heat_rotation(Z_head, targets):
    # first directions fit the heat coordinates by least squares; the rest is completed
    # orthonormally, so the map stays invertible
    Z_head = np.asarray(Z_head, dtype=np.float64)
    T = np.asarray(targets, dtype=np.float64)
    coef, *_ = np.linalg.lstsq(Z_head, T - T.mean(0), rcond=None)
    R, _ = np.linalg.qr(np.c_[coef, np.eye(Z_head.shape[1])])

    return R[:, : Z_head.shape[1]].astype(np.float32)
