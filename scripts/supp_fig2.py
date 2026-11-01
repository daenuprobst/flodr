import os
from functools import lru_cache

import matplotlib.pyplot as plt
import numpy as np
import torch
from matplotlib.cm import ScalarMappable
from matplotlib.colors import PowerNorm
from scipy.spatial import cKDTree

import common
from flodr import audit, raw_coords
from flodr.viz import plot_embedding

RUNS = {2500: range(1, 6), 5000: range(1, 6), 20000: range(1, 6), 80000: range(1, 3)}
MARKER = {"S1": "o", "S2": "s", "S3": "^", "V": "D"}
SHADE = dict(zip(RUNS, ("#E3B6D6", "#C98AB9", "#AA4499", "#5E1B4F")))
STOPS = (0, 0.15, 0.45, 0.75, 1)
CONTRAST = common.cmap(
    list(zip(STOPS, ("#F2EEF1", "#E8D0E2", "#C98AB9", "#AA4499", "#5E1B4F")))
)
CACHE = os.path.join(common.CACHE, "supp_fig2")
# label/cells/seed of every audit run, and of the earlier permutation contrast
KEYS = [f"{k}/{n}/{s}" for k in MARKER for n in RUNS for s in RUNS[n]]
KEYS += [f"old/{k}/5000/{s}" for k in ("S1", "S2", "S3") for s in (1, 2, 3)]


def sig(a):
    return 1.0 / (1.0 + np.exp(-a))


@lru_cache(None)
def population():
    # the FloDR map of the bone marrow training cells; per label, p(g | x) and the hidden
    # information of every cell, KL(p(g|x) || p(g|y)) with p(g|y) the leave-self-out
    # local-linear fit of p(g|x) over the 50 nearest map neighbours
    P = common.load("bmarrow")[0]
    train = common.split(len(P))[0]
    X = P[train].astype(np.float64)
    Y = common.fit("bmarrow_train", "FloDR (w=2)", 0, X=P[train])[0].astype(np.float64)
    # hidden directions s: most variance around the 50-NN map mean; visible v: most explained
    nb = cKDTree(Y).query(Y, k=50)[1]
    Ex = np.concatenate([X[nb[i : i + 10000]].mean(1) for i in range(0, len(X), 10000)])
    r, v = np.linalg.eigh(np.cov((X - Ex).T))[1], np.linalg.eigh(np.cov(Ex.T))[1]
    p = [X @ w for w in (r[:, -1], r[:, -2], v[:, -1], v[:, -2])]
    s1, s2, v1, v2 = ((q - q.mean()) / q.std() for q in p)
    e = np.array([[1, 1], [1, -1], [-1, 1], [-1, -1]], np.float64)
    lg = 1.2 * (s1[:, None] * e[:, 0] + s2[:, None] * e[:, 1])
    lg += 0.6 * (v1[:, None] * e[:, 0] + v2[:, None] * e[:, 1])
    p2 = np.exp(lg - lg.max(1, keepdims=True))
    Ys = (Y - Y.mean(0)) / Y.std(0)
    probs = dict(
        S1=sig(0.8 * v1 + 3.0 * sig(2.0 * v2) * s1),
        S2=p2 / p2.sum(1, keepdims=True),
        S3=sig(0.8 * v1 + 0.6 * s1),
        V=sig(1.5 * np.sin(2.0 * Ys[:, 0]) + Ys[:, 1]),
    )
    probs = {k: q if q.ndim == 2 else np.stack([1 - q, q], 1) for k, q in probs.items()}
    nb = cKDTree(Y).query(Y, k=51)[1][:, 1:]
    D = Y[nb] - Y[:, None, :]
    A = np.concatenate([np.ones(D.shape[:2] + (1,)), D], 2)
    AtA = np.einsum("nki,nkj->nij", A, A) + 1e-9 * np.eye(3)
    truth = {}
    for k, q in probs.items():
        pbar = np.linalg.solve(AtA, np.einsum("nki,nkc->nic", A, q[nb]))[:, 0]
        pbar = np.clip(pbar, 1e-6, None)
        pbar /= pbar.sum(1, keepdims=True)
        truth[k] = (q * (np.log(q + 1e-12) - np.log(pbar))).sum(1)
    return X, Y, probs, truth


def permutation_contrast(Y, Z, g, seed, hid=128):
    # the earlier estimate: held-out log-likelihood of the label under a tanh MLP on the
    # input minus one on the map, fitted on a third of the cells, minus its mean over 99
    # shuffles of the labels within 14 x 14 map bins, averaged over another third
    rng = np.random.default_rng(seed + 17)
    perm = rng.permutation(len(Y))
    ix = np.clip(((Y - Y.min(0)) / (np.ptp(Y, 0) + 1e-9) * 14).astype(int), 0, 13)
    bin_id = ix[:, 0] * 14 + ix[:, 1]
    bins = [np.nonzero(bin_id == b)[0] for b in np.unique(bin_id)]
    G = [g]
    for _ in range(99):
        G.append(g.copy())
        for rows in bins:
            G[-1][rows] = G[-1][rng.permutation(rows)]
    G, fit = torch.as_tensor(np.array(G)), torch.as_tensor(perm[: len(Y) // 3])
    R, C = len(G), int(G.max()) + 1

    def logp(X):
        # one network per label replica, all trained in one batch
        gen = torch.Generator().manual_seed(seed + 21)
        X = torch.as_tensor(X, dtype=torch.float32)

        def par(*shape, fan):
            u = torch.rand(*shape, generator=gen) * 2 - 1
            return torch.nn.Parameter(u * (1.0 / np.sqrt(fan)))

        d = X.shape[1]
        W1, b1 = par(R, d, hid, fan=d), par(R, hid, fan=d)
        W2, b2 = par(R, hid, hid, fan=hid), par(R, hid, fan=hid)
        W3, b3 = par(R, hid, C, fan=hid), par(R, C, fan=hid)
        opt = torch.optim.Adam([W1, b1, W2, b2, W3, b3], lr=1e-3, weight_decay=1e-4)

        def fwd(xb):
            h = torch.tanh(torch.einsum("bi,rih->rbh", xb, W1) + b1[:, None, :])
            h = torch.tanh(torch.einsum("rbh,rhg->rbg", h, W2) + b2[:, None, :])
            return torch.einsum("rbg,rgc->rbc", h, W3) + b3[:, None, :]

        for _ in range(3000):
            sel = fit[torch.randint(0, len(fit), (1024,), generator=gen)]
            loss = torch.nn.functional.cross_entropy(
                fwd(X[sel]).reshape(-1, C), G[:, sel].reshape(-1)
            )
            opt.zero_grad()
            loss.backward()
            opt.step()
        with torch.no_grad():
            return torch.log_softmax(fwd(X), 2).gather(2, G[..., None])[..., 0].numpy()

    gap = logp(Z) - logp(Y)
    return float((gap[0] - gap[1:].mean(0))[perm[2 * len(Y) // 3 :]].mean())


def run(key):
    # (true, estimated level) of one subsample and one draw of the labels
    print(key, flush=True)
    X, Y, probs, truth = population()
    k, n, s = key.split("/")[-3:]
    rng = np.random.default_rng(1000 + int(s))
    idx = np.sort(rng.choice(len(Y), int(n), replace=False))
    P = probs[k][idx]
    g = (rng.random(len(P))[:, None] > np.cumsum(P, 1)).sum(1).clip(0, P.shape[1] - 1)
    if key.startswith("old"):
        est = permutation_contrast(Y[idx], raw_coords(X, 50, 0)[0][idx], g, int(s))
    else:
        a = audit(Y[idx], X=X[idx], labels=g, test=False, seed=int(s), verbose=False)
        est = a.level
    if key == "S2/20000/1":
        nb = cKDTree(Y[idx]).query(Y[idx], k=150)[1]
        t = truth[k][idx][nb].mean(1)
        panel = dict(Y=Y[idx], field=a.field, lower=a.lower, truth=t)
        common.cached(os.path.join(CACHE, "S2_20000_1.npz"), lambda: panel)
    return [float(truth[k][idx].mean()), est]


def draw(res, z):
    fig = plt.figure(figsize=(common.TEXT_W, 2.05))
    fig.subplots_adjust(left=0.07, right=0.99, top=0.88, bottom=0.2)
    gs = fig.add_gridspec(1, 5, width_ratios=(1.25, 0.08, 1, 1, 1), wspace=0.08)
    ax = fig.add_subplot(gs[0, 0])
    t = np.linspace(0, 0.35, 50)
    tol = np.maximum(0.02, 0.2 * t)
    ax.fill_between(t, t - tol, t + tol, color="0.92", lw=0, zorder=0)
    ax.plot(t, t, color="0.4", lw=0.6, ls=(0, (2, 2)), zorder=1)
    for key, (true, est) in res.items():
        k, n = key.split("/")[-3:-1]
        if key.startswith("old"):
            kw = dict(facecolor="none", edgecolor=common.PARTNER, lw=0.7, zorder=2)
        else:
            kw = dict(color=SHADE[int(n)], edgecolor="white", lw=0.3, zorder=3)
        ax.scatter(true, est, marker=MARKER[k], s=11, **kw)
    ax.set(xlim=(-0.01, 0.35), ylim=(-0.03, 0.62))
    ax.set(xlabel="true level (nats)", ylabel="estimated level (nats)")
    ax.spines[["top", "right"]].set_visible(False)
    common.letter(ax, "a")
    dots = [
        dict(color=c, label=f"{n:,} cells".replace(",", r"\,"))
        for n, c in SHADE.items()
    ]
    dots.append(dict(mfc="none", mec=common.PARTNER, label="permutation contrast"))
    handles = [plt.Line2D([], [], ls="", marker="o", ms=4, **d) for d in dots]
    legend = dict(fontsize=6, frameon=False, handletextpad=0.2, borderaxespad=0.2)
    ax.legend(handles=handles, loc="upper left", **legend)
    norm = PowerNorm(0.5, 0.0, float(np.percentile(z["truth"], 99.5)))
    titles = ("true hidden information", "estimated field", r"lower 95\% band")
    for c, (key, title) in enumerate(zip(("truth", "field", "lower"), titles)):
        a = fig.add_subplot(gs[0, 2 + c])
        v = np.clip(z[key], 0, None)
        plot_embedding(
            z["Y"], values=v, ax=a, cmap=CONTRAST, norm=norm, s=0.6, order="high"
        )
        a.set_title(title, fontsize=8)
        common.letter(a, "bcd"[c])
    cax = fig.add_axes([0.62, 0.1, 0.25, 0.035])
    cb = fig.colorbar(
        ScalarMappable(norm, CONTRAST), cax=cax, orientation="horizontal", extend="max"
    )
    cb.ax.tick_params(labelsize=6.5, length=2, pad=1.5)
    cb.outline.set_linewidth(0.4)
    cb.set_label("nats, averaged over 150 map neighbours", fontsize=7, labelpad=2)
    return fig


if __name__ == "__main__":
    plt.rcParams.update(common.RC)
    path = os.path.join(CACHE, "runs.json")
    res = {key: common.memo(path, key, lambda: run(key)) for key in KEYS}
    z = np.load(os.path.join(CACHE, "S2_20000_1.npz"))
    common.save(draw(res, z), "supp_fig2.pdf")
