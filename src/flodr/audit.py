# per position of a 2-D map Y, a 95% lower bound on I(G; X | Y): the nats about a
# label G that the input X keeps and the map drops

import time
import warnings
from dataclasses import dataclass
from dataclasses import field as dfield

import numpy as np
import torch
from scipy.spatial import cKDTree

KS = (15, 50, 150)
ALPHA_KNN = 1.0
CAP_KNN = 5.0
HID = 64
TAUS = torch.exp(torch.linspace(-1.5, 1.5, 61, dtype=torch.float64))

MAX_FIELD = 80_000
MIN_CELLS = 20_000
TEST_CELLS = 5_000
# the null scores TEST_R replicas but fits TEST_REPS: all share one batched fit (its
# initialisation and early stop), so fitting fewer would change the scored ones
TEST_REPS = 199
TEST_R = 99
LEAF = 10
SPLITS = 5
K_FIELD = 150
Z95 = 1.6448536


def _split(Y, roots, done):
    out = []
    stack = list(roots)

    while stack:
        idx = stack.pop()

        if done(idx):
            out.append(idx)
            continue

        ax = int(np.argmax(np.ptp(Y[idx], 0)))
        o = idx[np.argsort(Y[idx, ax], kind="stable")]
        half = len(o) // 2
        stack += [o[:half], o[half:]]

    return out


def _extent(Y, idx):
    return float(np.hypot(*np.ptp(Y[idx], 0))) if len(idx) > 1 else 0.0


def kd_leaves(Y, leaf=10, max_ext=3.0):
    # splitting stops below 1.5 leaf, so leaves hold 8-15 cells
    leaves = _split(Y, [np.arange(len(Y))], lambda idx: len(idx) <= 1.5 * leaf)
    # a leaf that straddles a gap breaks the swap's flat-p(g|y) assumption, so wide
    # leaves are split until they are compact
    cap = max_ext * np.median([_extent(Y, idx) for idx in leaves])
    leaves = _split(Y, leaves, lambda idx: len(idx) == 1 or _extent(Y, idx) <= cap)
    out = np.empty(len(Y), np.int64)

    for lid, idx in enumerate(leaves):
        out[idx] = lid

    return out


def leaf_replicas(g, leaves, R, seed):
    rng = np.random.default_rng(seed)
    n = len(g)
    base = np.lexsort((np.arange(n), leaves))
    out = np.empty((R + 1, n), np.int64)
    out[0] = g

    for r in range(R):
        order = np.lexsort((rng.random(n), leaves))
        out[r + 1, base] = g[order]

    return out


def _std(A):
    A = np.asarray(A, np.float64)

    return ((A - A.mean(0)) / (A.std(0) + 1e-12)).astype(np.float32)


class _Nets:
    def __init__(self, R, d, hid, C, seed, fan_in=None):
        gen = torch.Generator().manual_seed(seed)
        gen_in = torch.Generator().manual_seed(seed + 7)
        fan_in = fan_in or d

        def uni(*shape, fan, g=gen):
            b = 1.0 / np.sqrt(fan)
            return (torch.rand(*shape, generator=g) * 2 - 1) * b

        # input rows come from their own stream, so q_x's net starts with q_y's
        # weights on the shared map features and differs only on the extra x rows
        W1 = uni(d, R, hid, fan=fan_in, g=gen_in).permute(1, 0, 2).contiguous()
        self.p = [
            W1.requires_grad_(),
            uni(R, 1, hid, fan=fan_in).requires_grad_(),
            uni(R, hid, hid, fan=hid).requires_grad_(),
            uni(R, 1, hid, fan=hid).requires_grad_(),
            # zero last layer: both nets start as the calibrated kNN they are stacked on
            torch.zeros(R, hid, C, requires_grad=True),
            torch.zeros(R, 1, C, requires_grad=True),
        ]

    def __call__(self, xb, p=None):
        W1, b1, W2, b2, W3, b3 = self.p if p is None else p
        h = torch.relu(torch.einsum("bd,rdh->rbh", xb, W1) + b1)
        h = torch.relu(torch.bmm(h, W2) + b2)

        return torch.bmm(h, W3) + b3


def _nll_rows(net, p, Xt, Gt, off, rows, chunk=4096):
    R = Gt.shape[0]
    tot = torch.zeros(R, dtype=torch.float64)

    with torch.no_grad():
        for s in range(0, len(rows), chunk):
            rr = rows[s : s + chunk]
            lg = net(Xt[rr], p)

            if off is not None:
                lg = lg + off[:, rr]

            lp = torch.log_softmax(lg, 2).gather(2, Gt[:, rr].unsqueeze(2)).squeeze(2)
            tot -= lp.sum(1).double()

    return tot / len(rows)


def _train(
    Xt,
    Gt,
    off,
    fit,
    stop,
    C,
    seed,
    lr=2e-3,
    wd=1e-4,
    batch=256,
    max_epochs=200,
    patience=12,
    fan_in=None,
):
    R = Gt.shape[0]
    net = _Nets(R, Xt.shape[1], HID, C, seed, fan_in)
    opt = torch.optim.Adam(net.p, lr=lr, weight_decay=wd)
    fit_t = torch.as_tensor(fit)
    stop_t = torch.as_tensor(stop)
    gen = torch.Generator().manual_seed(seed + 1)
    best = [q.detach().clone() for q in net.p]
    best_loss = _nll_rows(net, None, Xt, Gt, off, stop_t)
    best_ep = np.zeros(R, np.int64)
    wait = np.zeros(R, np.int64)
    bs = min(batch, len(fit))
    steps = max(1, len(fit) // bs)

    for ep in range(1, max_epochs + 1):
        perm = fit_t[torch.randperm(len(fit_t), generator=gen)]

        for s in range(steps):
            rr = perm[s * bs : (s + 1) * bs]
            lg = net(Xt[rr])

            if off is not None:
                lg = lg + off[:, rr]

            loss = torch.nn.functional.cross_entropy(
                lg.reshape(-1, C), Gt[:, rr].reshape(-1), reduction="sum"
            ) / len(rr)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()

        cur = _nll_rows(net, None, Xt, Gt, off, stop_t)
        imp = (cur < best_loss - 1e-6).numpy()

        if imp.any():
            m = torch.as_tensor(imp)

            with torch.no_grad():
                for b, q in zip(best, net.p):
                    b[m] = q[m]

            best_loss = torch.where(m, cur, best_loss)
            best_ep[imp] = ep

        wait = np.where(imp, 0, wait + 1)

        if (wait >= patience).all():
            break

    return net, best, best_ep


def _predict(net, p, Xt, rows, off=None, chunk=4096):
    outs = []

    with torch.no_grad():
        for s in range(0, len(rows), chunk):
            rr = rows[s : s + chunk]
            lg = net(Xt[rr], p)

            if off is not None:
                lg = lg + off[:, s : s + chunk]

            outs.append(lg)

    return torch.cat(outs, 1)


def _temp(L, Gsub):
    best_nll = torch.full((L.shape[0],), np.inf, dtype=torch.float64)
    best_tau = torch.ones(L.shape[0], dtype=torch.float64)
    idx = Gsub.unsqueeze(2)

    for tau in TAUS:
        nll = (
            -torch.log_softmax(L / tau.float(), 2)
            .gather(2, idx)
            .squeeze(2)
            .double()
            .mean(1)
        )
        better = nll < best_nll
        best_nll = torch.where(better, nll, best_nll)
        best_tau = torch.where(better, tau, best_tau)

    return best_tau.float(), best_nll


def _knn_logp(Gp, nbr, dist, caps, C, prior):
    # neighbours past the cap are dropped, so a small island is predicted from its own
    # cells, not from distant ones, which would read as hidden information
    R = Gp.shape[0]
    m = nbr.shape[0]
    out = {}
    rowid = np.arange(m)[:, None] * (C + 1)

    for k in KS:
        valid = dist[:, :k] <= caps[k]
        keff = valid.sum(1)[:, None].astype(np.float64)
        arr = np.empty((R, m, C), np.float32)

        for r in range(R):
            lab = np.where(valid, Gp[r][nbr[:, :k]], C)
            cnt = np.bincount((rowid + lab).ravel(), minlength=m * (C + 1))
            cnt = cnt.reshape(m, C + 1)[:, :C]
            arr[r] = np.log((cnt + ALPHA_KNN * prior[r]) / (keff + ALPHA_KNN))

        out[k] = torch.from_numpy(arr)

    return out


def _fold(Ys, Fy, Xs, G, T, E, C, seed):
    rng = np.random.default_rng(seed)
    R, n = G.shape
    nT = len(T)
    Gt = torch.as_tensor(G)
    # a class with no cell in T gets half a count; at zero its log q is -inf on both
    # sides and the gap d is NaN
    cnt = np.stack([np.bincount(G[r, T], minlength=C) for r in range(R)]).astype(
        np.float64
    )
    prior = np.where(cnt == 0, 0.5, cnt)
    prior /= prior.sum(1, keepdims=True)

    # leave-one-out on T, so the base logits are out of sample where g and h train
    tree = cKDTree(Ys[T])
    dT, nbT = tree.query(Ys[T], k=KS[-1] + 1)
    is_self = nbT == np.arange(nT)[:, None]
    self_pos = np.where(is_self.any(1), is_self.argmax(1), KS[-1])
    keep = np.arange(KS[-1] + 1)[None, :] != self_pos[:, None]
    nbT = nbT[keep].reshape(nT, KS[-1])
    dT = dT[keep].reshape(nT, KS[-1])
    dE, nbE = tree.query(Ys[E], k=KS[-1])
    caps = {k: CAP_KNN * float(np.median(dT[:, k - 1])) for k in KS}
    GT = G[:, T]
    knnT = _knn_logp(GT, nbT, dT, caps, C, prior)
    knnE = _knn_logp(GT, nbE, dE, caps, C, prior)

    pT = rng.permutation(nT)
    temp_loc = pT[: nT // 5]
    stop_loc = pT[nT // 5 : 2 * nT // 5]
    fit_loc = pT[2 * nT // 5 :]
    sel_loc = np.sort(np.concatenate([stop_loc, fit_loc]))
    temp_loc = np.sort(temp_loc)

    GTt = torch.as_tensor(GT)

    best_sel = torch.full((R,), np.inf, dtype=torch.float64)
    choice = np.zeros(R, np.int64)
    baseT = torch.empty((R, nT, C))
    baseE = torch.empty((R, len(E), C))

    for ci, k in enumerate(KS):
        tau, _ = _temp(knnT[k][:, temp_loc], GTt[:, temp_loc])
        cT = torch.log_softmax(knnT[k] / tau[:, None, None], 2)
        sel = (
            -cT[:, sel_loc]
            .gather(2, GTt[:, sel_loc].unsqueeze(2))
            .squeeze(2)
            .double()
            .mean(1)
        )
        better = sel < best_sel

        if better.any():
            baseT[better] = cT[better]
            baseE[better] = torch.log_softmax(
                knnE[k][better] / tau[better][:, None, None], 2
            )
            best_sel = torch.where(better, sel, best_sel)
            choice[better.numpy()] = ci

    # q_y = base + g(map features), q_x = base + h(map features, x): one learner, from
    # zero, early-stopped on the same cells, so x is the only difference
    off = torch.zeros((R, n, C))
    off[:, T] = baseT
    d_f = Fy.shape[1]
    outs, eps, taus = [], [], []

    for feats in (Fy, np.concatenate([Fy, Xs], 1)):
        Ft = torch.as_tensor(np.ascontiguousarray(feats))
        net, p, ep = _train(
            Ft, Gt, off, T[fit_loc], T[stop_loc], C, seed + 10, fan_in=d_f
        )
        lT = _predict(net, p, Ft, torch.as_tensor(T[temp_loc]), off=baseT[:, temp_loc])
        tau, _ = _temp(lT, GTt[:, temp_loc])
        lE = _predict(net, p, Ft, torch.as_tensor(E), off=baseE)
        outs.append(torch.log_softmax(lE / tau[:, None, None], 2))
        eps.append(ep)
        taus.append(tau)

    GE = Gt[:, E].unsqueeze(2)
    out_y = outs[0].gather(2, GE).squeeze(2).numpy().astype(np.float64)
    out_x = outs[1].gather(2, GE).squeeze(2).numpy().astype(np.float64)
    dist = tuple(o[0].numpy().astype(np.float64) for o in outs)
    notes = dict(
        k=KS[choice[0]],
        k_all=np.bincount(choice, minlength=len(KS)).tolist(),
        ep_y=int(eps[0][0]),
        ep_x=int(eps[1][0]),
        ep_y_mean=float(eps[0].mean()),
        ep_x_mean=float(eps[1].mean()),
        tau_y=float(taus[0][0]),
        tau_x=float(taus[1][0]),
    )

    return out_y, out_x, dist, notes


def cross_fit(Y, X, G, n_folds=5, seed=0, smooth_k=15):
    # G (R, n): row 0 the observed labels, the rest permuted replicas
    G = np.atleast_2d(np.asarray(G, np.int64))
    R, n = G.shape
    C = int(G.max()) + 1
    Ys, Xs = _std(Y), _std(X)
    # q_y also sees the mean input of its nearest map neighbours (self left out), a
    # label-free function of position, as the map folds even linear functions of x
    Fy = Ys

    if smooth_k:
        _, nb = cKDTree(Ys).query(Ys, k=smooth_k + 1)
        Fy = np.concatenate([Ys, _std(Xs[nb[:, 1:]].mean(1))], 1)

    folds = np.array_split(np.random.default_rng(seed).permutation(n), n_folds)
    ly = np.empty((R, n))
    lx = np.empty((R, n))
    qy = np.full((n, C), np.nan)
    qx = np.full((n, C), np.nan)
    notes = []

    for f, E in enumerate(folds):
        E = np.sort(E)
        T = np.sort(np.concatenate([folds[j] for j in range(n_folds) if j != f]))
        a, b, (dy, dx), nt = _fold(Ys, Fy, Xs, G, T, E, C, seed * 100 + f)
        ly[:, E], lx[:, E] = a, b
        qy[E], qx[E] = dy, dx
        notes.append(nt)

    D = lx - ly

    if not np.isfinite(D).all():
        raise FloatingPointError(f"{int((~np.isfinite(D)).sum())} non-finite gaps")

    return D, ly, lx, (qy, qx), notes


def local_field(Y, d, k=150, z=Z95):
    _, nb = cKDTree(Y).query(Y, k=min(k, len(Y)))
    w = d[nb]
    mu = w.mean(1)
    se = w.std(1, ddof=1) / np.sqrt(w.shape[1])

    return mu, mu - z * se


def _bacc(pred, g, C):
    return float(
        np.mean([np.mean(pred[g == c] == c) for c in range(C) if np.any(g == c)])
    )


def _missing(lab):
    try:
        import pandas as pd

        return np.asarray(pd.isna(lab), bool)
    except ImportError:
        flat = np.asarray(lab, object).ravel()
        return np.array([v is None or (isinstance(v, float) and v != v) for v in flat])


@dataclass
class AuditResult:
    key: str
    classes: list
    n: int
    n_field: int
    n_test: int
    level: float
    level_splits: list
    p_global: float
    certified: object
    null_mean: float
    null_sd: float
    level_test: float
    band_positive: float
    entropy: float
    ce_map: float
    ce_input: float
    share_hidden: float
    acc_map: float
    acc_input: float
    bacc_map: float
    bacc_input: float
    acc_majority: float
    secs: float
    # per cell, NaN where the label is missing; subsampled-out cells take their nearest
    # scored cell's field and band
    field: np.ndarray = dfield(repr=False)
    lower: np.ndarray = dfield(repr=False)
    gap: np.ndarray = dfield(repr=False)
    scored: np.ndarray = dfield(repr=False)
    Y: np.ndarray = dfield(repr=False)
    labels: np.ndarray = dfield(repr=False)
    notes: list = dfield(default_factory=list)

    def summary(self):
        skip = {"field", "lower", "gap", "scored", "Y", "labels"}

        return {k: v for k, v in self.__dict__.items() if k not in skip}

    def __str__(self):
        verdict = {True: "certified", False: "not certified", None: "not tested"}[
            self.certified
        ]
        share = (
            f"{100 * self.share_hidden:.0f}% of what the profile recovers"
            if np.isfinite(self.share_hidden)
            else "the profile recovers nothing held out"
        )

        return (
            f"{self.key}: level {self.level:.3f} nats ({share}), global p "
            f"{self.p_global:.2f} ({verdict}), lower band > 0 at "
            f"{100 * self.band_positive:.0f}% of positions; held out, guessed right "
            f"{100 * self.acc_map:.0f}% from position, {100 * self.acc_input:.0f}% from "
            f"profile (balanced {100 * self.bacc_map:.0f}% vs {100 * self.bacc_input:.0f}%, "
            f"majority {100 * self.acc_majority:.0f}%)"
        )


def _resolve(data, basis, use_rep, key, X, labels):
    if hasattr(data, "obsm"):
        Y = np.asarray(data.obsm[basis], np.float64)
        Xin = data.X if use_rep in (None, "X") else data.obsm[use_rep]
        Xin = np.asarray(Xin.toarray() if hasattr(Xin, "toarray") else Xin, np.float64)
        lab = data.obs[key].to_numpy() if labels is None else np.asarray(labels)

        return Y, Xin, lab, key

    if X is None or labels is None:
        raise ValueError("with a plain array map, pass X= and labels=")

    return (
        np.asarray(data, np.float64),
        np.asarray(X, np.float64),
        np.asarray(labels),
        key or "label",
    )


def audit(
    data,
    basis="X_umap",
    use_rep="X_pca",
    key=None,
    *,
    X=None,
    labels=None,
    test=True,
    seed=0,
    n_threads=4,
    inplace=True,
    verbose=True,
):
    t0 = time.perf_counter()
    threads = torch.get_num_threads()
    torch.set_num_threads(n_threads)

    try:
        res = _audit(data, basis, use_rep, key, X, labels, test, seed)
    finally:
        torch.set_num_threads(threads)

    res.secs = time.perf_counter() - t0

    if inplace and hasattr(data, "obsm"):
        data.obs[f"hidden_{res.key}"] = res.field
        data.obs[f"hidden_{res.key}_lower"] = res.lower
        data.uns.setdefault("hidden_audit", {})[res.key] = {
            k: v
            for k, v in res.summary().items()
            if np.isscalar(v) or isinstance(v, (list, str))
        }

    if verbose:
        print(res, flush=True)

    return res


def _audit(data, basis, use_rep, key, X, labels, test, seed):
    Y, Xin, lab, key = _resolve(data, basis, use_rep, key, X, labels)

    if Y.ndim != 2 or Y.shape[1] != 2:
        raise ValueError(f"the map must be 2-D, got shape {Y.shape}")

    if len(Y) != len(Xin) or len(Y) != len(lab):
        raise ValueError("map, input and labels must have one row per cell")

    notes = []

    has = ~_missing(lab)
    has &= np.isfinite(Y).all(1) & np.isfinite(Xin).all(1)
    cells = np.flatnonzero(has)
    classes, g_all = np.unique(np.asarray(lab[cells]).astype(str), return_inverse=True)
    C = len(classes)

    if C < 2:
        raise ValueError(
            "the label needs at least two classes among the labelled cells"
        )

    n = len(cells)

    if n < MIN_CELLS:
        msg = (
            f"{n} labelled cells, fewer than 20,000: the level can read high for labels "
            "that are simple in the input"
        )
        warnings.warn(msg, stacklevel=3)
        notes.append(msg)

    rng = np.random.default_rng(seed)
    sub = (
        np.sort(rng.choice(n, MAX_FIELD, replace=False))
        if n > MAX_FIELD
        else np.arange(n)
    )

    if n > MAX_FIELD:
        notes.append(f"field on a uniform {MAX_FIELD}-cell subsample of {n}")

    Ys, Xs, g = Y[cells[sub]], Xin[cells[sub]], g_all[sub]
    m = len(sub)
    k = min(K_FIELD, m)

    # median over fold splits, with the between-split spread in the variance
    # (DML-style), since one split's band depends on the split
    fs, ses, levels, gaps, accs, ce_y, ce_x = [], [], [], [], [], [], []

    for b in range(SPLITS):
        D, _, _, (qy, qx), _ = cross_fit(
            Ys, Xs, g[None], seed=seed if b == 0 else seed * 10 + b
        )
        d = D[0]
        mu, lo = local_field(Ys, d, k=k)
        fs.append(mu)
        ses.append((mu - lo) / Z95)
        levels.append(float(d.mean()))
        gaps.append(d)
        ce_y.append(-qy[np.arange(m), g].mean())
        ce_x.append(-qx[np.arange(m), g].mean())
        py, px = qy.argmax(1), qx.argmax(1)
        accs.append(
            (np.mean(py == g), np.mean(px == g), _bacc(py, g, C), _bacc(px, g, C))
        )

    fs, ses = np.array(fs), np.array(ses)
    mu = np.median(fs, 0)
    lo = mu - Z95 * np.sqrt(np.median(ses**2 + (fs - mu) ** 2, 0))
    accs = np.median(np.array(accs), 0)
    prior = np.bincount(g, minlength=C) / m
    entropy = float(-(prior[prior > 0] * np.log(prior[prior > 0])).sum())
    level = float(np.median(levels))
    info_x = entropy - float(np.median(ce_x))

    p_global, null_mean, null_sd, level_test, n_test = np.nan, np.nan, np.nan, np.nan, 0

    if test:
        tsub = (
            np.sort(rng.choice(m, TEST_CELLS, replace=False))
            if m > TEST_CELLS
            else np.arange(m)
        )
        n_test = len(tsub)

        if m < TEST_CELLS:
            msg = f"global test on all {m} cells, fewer than 5,000: it has less power"
            warnings.warn(msg, stacklevel=3)
            notes.append(msg)

        Yt, Xt = Ys[tsub], Xs[tsub]
        _, gt = np.unique(g[tsub], return_inverse=True)
        leaves = kd_leaves(Yt, LEAF)
        # within a leaf the map cannot tell cells apart, so swapping labels there keeps
        # what the map shows and breaks what only the input holds
        G = leaf_replicas(gt, leaves, TEST_REPS, seed=2000 + seed)
        D = cross_fit(Yt, Xt, G, seed=seed)[0]
        obs, null = D[0].mean(), D[1 : TEST_R + 1].mean(1)
        p_global = float((1 + np.sum(null >= obs)) / (1 + TEST_R))
        null_mean, null_sd, level_test = (
            float(null.mean()),
            float(null.std()),
            float(obs),
        )

    N = len(Y)
    f_all, lo_all, gap_all = np.full(N, np.nan), np.full(N, np.nan), np.full(N, np.nan)
    scored = np.zeros(N, bool)
    scored[cells[sub]] = True
    f_all[cells[sub]], lo_all[cells[sub]] = mu, lo
    gap_all[cells[sub]] = np.mean(gaps, 0)
    rest = cells[~np.isin(np.arange(n), sub)]

    if len(rest):
        _, nn = cKDTree(Ys).query(Y[rest])
        f_all[rest], lo_all[rest] = mu[nn], lo[nn]

    lab_all = np.full(N, None, object)
    lab_all[cells] = classes[g_all]

    return AuditResult(
        key=key,
        classes=classes.tolist(),
        n=n,
        n_field=m,
        n_test=n_test,
        level=level,
        level_splits=levels,
        p_global=p_global,
        certified=bool(p_global <= 0.05) if test else None,
        null_mean=null_mean,
        null_sd=null_sd,
        level_test=level_test,
        band_positive=float(np.mean(lo > 0)),
        entropy=entropy,
        ce_map=float(np.median(ce_y)),
        ce_input=float(np.median(ce_x)),
        share_hidden=float(level / info_x) if info_x > 0 else np.nan,
        acc_map=float(accs[0]),
        acc_input=float(accs[1]),
        bacc_map=float(accs[2]),
        bacc_input=float(accs[3]),
        acc_majority=float(prior.max()),
        secs=0.0,
        field=f_all,
        lower=lo_all,
        gap=gap_all,
        scored=scored,
        Y=Y,
        labels=lab_all,
        notes=notes,
    )


_RAMP = [
    (0.0, "#e8e8e8"),
    (0.15, "#d3dbe6"),
    (0.45, "#7ea6cc"),
    (0.75, "#3b7cba"),
    (1.0, "#173e6b"),
]


def plot_audit(res, axes=None, size=3.0, s=None, vmax=None, show_labels=True):
    import matplotlib.pyplot as plt
    from matplotlib.colors import LinearSegmentedColormap, to_rgba

    ncol = 2 if show_labels else 1

    if axes is None:
        _, axes = plt.subplots(1, ncol, figsize=(size * ncol, size), squeeze=False)
        axes = axes[0]

    axes = np.atleast_1d(axes)
    Y, ok = res.Y, np.isfinite(res.lower)
    s = s if s is not None else max(0.3, min(4.0, 3e4 / max(len(Y), 1)))
    order = np.random.default_rng(0).permutation(np.flatnonzero(ok))
    i = 0

    if show_labels:
        ax = axes[0]
        lab = np.asarray(res.labels[order]).astype(str)
        pal = plt.get_cmap("tab10").colors
        cols = np.array([to_rgba(pal[res.classes.index(c) % 10]) for c in lab])
        ax.scatter(Y[order, 0], Y[order, 1], c=cols, s=s, lw=0, rasterized=True)

        for j, c in enumerate(res.classes[:10]):
            ax.scatter([], [], color=pal[j % 10], s=12, label=c)

        ax.legend(
            frameon=False, fontsize=7, loc="best", handletextpad=0.1, markerscale=1.5
        )
        i = 1

    ax = axes[i]
    # a lower bound at or below 0 says nothing, so those share the floor colour
    v = np.clip(res.lower[order], 0, None)
    # the floor keeps a near-zero field pale instead of stretching noise to full colour
    top = vmax if vmax is not None else max(float(np.quantile(v, 0.99)), 0.05)
    sc = ax.scatter(
        Y[order, 0],
        Y[order, 1],
        c=v,
        s=s,
        lw=0,
        vmin=0,
        vmax=top,
        cmap=LinearSegmentedColormap.from_list("hidden", _RAMP),
        rasterized=True,
    )
    cb = plt.colorbar(sc, ax=ax, fraction=0.046, pad=0.02)
    cb.set_label("hidden nats, 95% lower bound", fontsize=7)
    cb.ax.tick_params(labelsize=6)
    p = f", p {res.p_global:.2f}" if np.isfinite(res.p_global) else ""
    ax.set_title(
        f"{max(res.level, 0):.2f} nats{p}; acc. {100 * res.acc_map:.0f}% map, "
        f"{100 * res.acc_input:.0f}% profile",
        fontsize=7,
    )

    for ax in axes[: i + 1]:
        ax.set_aspect("equal", adjustable="datalim")
        ax.set_xticks([])
        ax.set_yticks([])

    return axes
