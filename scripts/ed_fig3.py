import json
import os
import subprocess
import tempfile

import numpy as np

import common
import ed_fig1
from common import ATLASES, K

DIR = os.path.join(common.CACHE, "ed_fig3")
# the reaction fingerprints are left out: UMAP has no inverse under the Jaccard distance
SETS = ("mnist", "fmnist", "paul15") + ATLASES
SHORT = dict(
    zip(
        SETS,
        ("MNIST", "Fashion-\nMNIST", "paul15", "Bone\nmarrow", "Cere-\nbellum", "Seed"),
    )
)
REPS = {ds: 3 if ds in ATLASES else 5 for ds in SETS}
# held-out points decoded per split; UMAP's inverse takes over a minute per thousand
# images
N_EVAL = 5000
# the split whose decoded images are drawn
SHOWN = ("fmnist", 0)
# drawn with (colour, marker, filled, label): the inverse each method comes with, then
# what a layout allows when the fitted points or a trained decoder are used instead
S = common.METHOD_STYLE
ARMS = {
    "flodr": (*S["FloDR"], True, "FloDR"),
    "umap": (*S["UMAP"], True, "UMAP"),
    "pcumap": (*S["Parametric UMAP"], True, "Parametric UMAP"),
    "pca": (*S["PCA-2"], True, "PCA"),
    "nbr_flodr": (S["FloDR"][0], "o", False, "neighbour mean, FloDR map"),
    "dec_flodr": (S["FloDR"][0], "D", False, "trained decoder, FloDR map"),
    "nbr_umap": (S["UMAP"][0], "s", False, "neighbour mean, UMAP map"),
    "dec_umap": (S["UMAP"][0], "D", False, "trained decoder, UMAP map"),
}


def data(ds, rep):
    X, y = common.load(ds)
    tr, te = ed_fig1.split(ds, rep, len(X))
    if len(te) > N_EVAL:
        te = np.sort(np.random.default_rng(30_000 + rep).permutation(te)[:N_EVAL])
    return X[tr], X[te], y[te]


def error(Xhat, Xte, mean):
    # squared error over that of the training mean: 0 is exact, 1 no better than it
    return float(((Xhat - Xte) ** 2).sum() / ((Xte - mean) ** 2).sum())


def to_fibre(est, X0, Y, tol, steps=50):
    # an input near X0 that the map places at Y: Gauss-Newton on the two map
    # coordinates, each step the smallest in input units that closes the gap, halved
    # while it widens it. Also returns which rows met the tolerance; the last lines set
    # the position itself, so the map places every returned input at Y either way
    import torch

    flow = est.flow_
    dev = next(flow.parameters()).device
    wh = est._raw_of.__self__
    sd = [np.ravel(wh.head_sd), np.ravel(wh.tail_sd), np.ones(est._n_pad)]
    w = torch.as_tensor(np.concatenate(sd) ** -2.0, dtype=torch.float32, device=dev)
    z = torch.as_tensor(est._to_coords(np.asarray(X0, np.float32)), device=dev)
    y = torch.as_tensor(np.asarray(Y, np.float32), device=dev)
    eye = torch.eye(2, device=dev)
    for _ in range(steps):
        z.requires_grad_(True)
        out = flow(z)[:, :2]
        rows = [
            torch.autograd.grad(out[:, c].sum(), z, retain_graph=True)[0]
            for c in (0, 1)
        ]
        J, e, z = torch.stack(rows, 1), (out - y).detach(), z.detach()
        size = e.norm(dim=1)
        if float(size.max()) < tol:
            break
        A = (J * w) @ J.transpose(1, 2)
        A = A + 1e-10 * A.diagonal(dim1=1, dim2=2).sum(1)[:, None, None] * eye
        dz = ((J * w).transpose(1, 2) @ torch.linalg.solve(A, e[..., None]))[..., 0]
        step = torch.ones(len(z), 1, device=dev)
        with torch.no_grad():
            for _ in range(6):
                worse = (flow(z - step * dz)[:, :2] - y).norm(dim=1) > size
                step[worse] = step[worse] / 2
            z = z - step * dz
    with torch.no_grad():
        lat = flow(z)
        met = ((lat[:, :2] - y).norm(dim=1) < tol).cpu().numpy()
        lat[:, :2] = y
        return est._to_raw(flow.inverse(lat).cpu().numpy()), met


def decode(est, Xtr, tree, Y, tol):
    # the mean of the fitted points nearest to each position, and that mean moved onto
    # the fibre of the position
    near = Xtr[tree.query(Y, k=K)[1]].mean(1)
    return (*to_fibre(est, near, Y, tol), near)


def trained_decoder(Ytr, Xtr, Yq, seed):
    # NNInv: four hidden layers of 1 024, squared error, early stopping on a tenth of
    # the points
    import torch

    torch.manual_seed(seed)
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    perm = np.random.default_rng(seed).permutation(len(Ytr))
    val, fit = perm[: len(perm) // 10], perm[len(perm) // 10 :]
    ym, ys, xm, xs = Ytr.mean(0), Ytr.std(0), Xtr.mean(0), float(Xtr.std())

    def t(a):
        return torch.from_numpy(np.ascontiguousarray(a, np.float32)).to(dev)

    A, B = t((Ytr - ym) / ys), t((Xtr - xm) / xs)
    dims = [2, 1024, 1024, 1024, 1024]
    layers = [
        m
        for a, b in zip(dims, dims[1:])
        for m in (torch.nn.Linear(a, b), torch.nn.ReLU())
    ]
    net = torch.nn.Sequential(*layers, torch.nn.Linear(1024, Xtr.shape[1])).to(dev)
    opt = torch.optim.Adam(net.parameters(), lr=1e-3)
    batch = 128 if len(fit) < 20_000 else 1024
    fit_t, val_t = t(fit).long(), t(val).long()
    best, state, bad = np.inf, None, 0
    for _ in range(300):
        order = fit_t[torch.randperm(len(fit_t), device=dev)]
        for s in range(0, len(order), batch):
            i = order[s : s + batch]
            loss = ((net(A[i]) - B[i]) ** 2).mean()
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
        with torch.no_grad():
            v = float(((net(A[val_t]) - B[val_t]) ** 2).mean())
        if v < best - 1e-6:
            best, state, bad = v, {n: w.clone() for n, w in net.state_dict().items()}, 0
        else:
            bad += 1
            if bad == 20:
                break
    net.load_state_dict(state)
    with torch.no_grad():
        return net(t((Yq - ym) / ys)).cpu().numpy() * xs + xm


def first_of_each_class(y):
    return np.array([np.flatnonzero(y == c)[0] for c in np.unique(y)])


def run(ds, rep):
    def make():
        import faiss
        import umap
        from flodr import FloDR
        from scipy.spatial import cKDTree
        from sklearn.decomposition import PCA

        print(f"{ds} r{rep}", flush=True)
        Xtr, Xte, yte = data(ds, rep)
        mean = Xtr.mean(0)
        out = {}

        def err(name, Xhat):
            out[name] = error(Xhat, Xte, mean)
            if (ds, rep) == SHOWN:
                out["x_" + name] = np.asarray(
                    Xhat[first_of_each_class(yte)], np.float32
                )

        est = FloDR(w=2.0, random_state=rep, verbose=False).fit(Xtr)
        Ytr = est.embedding_
        tree = cKDTree(Ytr)
        index = faiss.IndexFlatL2(Xtr.shape[1])
        index.add(Xtr)
        # the units: the spacing of neighbouring fitted points, on the map and in the
        # input
        spacing = np.median(tree.query(Ytr, k=2)[0][:, 1])
        some = np.random.default_rng(40_000 + rep).permutation(len(Xtr))[:2000]
        unit = np.median(np.sqrt(index.search(Xtr[some], 2)[0][:, 1]))
        Yo, Yf = est.transform_opt(Xte), est.transform(Xte)
        decoded, met, near = decode(est, Xtr, tree, Yo, 1e-3 * spacing)
        err("flodr", decoded)
        err("nbr_flodr", near)
        err("flodr_fwd", decode(est, Xtr, tree, Yf, 1e-3 * spacing)[0])
        err("dec_flodr", trained_decoder(Ytr, Xtr, Yo, rep))
        # how far the mean was moved, the share of positions whose steps did not meet
        # the tolerance, and where the map puts the decoded points
        out["moved"] = float(np.median(np.linalg.norm(decoded - near, axis=1)) / unit)
        out["unmet"] = float(1 - met.mean())
        off = np.linalg.norm(est.transform(decoded) - Yo, axis=1) / spacing
        out["land_median"], out["land_max"] = float(np.median(off)), float(off.max())
        # three shortcuts through the residuals of the same neighbours: each run back
        # through the flow at the position and the results averaged, their mean, none
        Ztr, nn = est.transform_latent(Xtr), tree.query(Yo, k=K)[1]
        each = np.zeros(Xte.shape)
        for j in range(K):
            Z = Ztr[nn[:, j]].copy()
            Z[:, :2] = Yo
            each += est.inverse_latent(Z)
        err("flodr_each", each / K)
        Z = Ztr[nn].mean(1)
        Z[:, :2] = Yo
        err("flodr_meanres", est.inverse_latent(Z))
        err("flodr_zero", est.inverse_transform(Yo))

        um = umap.UMAP(n_neighbors=K, min_dist=0.1, init="pca", random_state=rep).fit(
            Xtr
        )
        Yu = um.transform(Xte)
        err("umap", um.inverse_transform(Yu))
        nn = cKDTree(um.embedding_).query(Yu, k=K)[1]
        err("nbr_umap", Xtr[nn].mean(1))
        err("dec_umap", trained_decoder(um.embedding_, Xtr, Yu, rep))

        pca = PCA(2, svd_solver="randomized", random_state=rep).fit(Xtr)
        err("pca", pca.inverse_transform(pca.transform(Xte)))
        # no map at all: the mean of the nearest training points in the input
        err("nbr_input", Xtr[index.search(Xte, K)[1]].mean(1))
        if (ds, rep) == SHOWN:
            out["x_truth"] = Xte[first_of_each_class(yte)]
        return out

    return common.cached(f"{DIR}/{ds}_r{rep}.npz", make)


# Parametric UMAP with its decoder, in the interpreter of its own; the inputs are not in
# [0, 1], so the decoder is fitted on the squared error, not the default cross-entropy
_PCUMAP = """
import json, sys, numpy as np, tensorflow as tf
from umap.parametric_umap import ParametricUMAP
fin, fout, seed = json.loads(sys.argv[1])
tf.random.set_seed(seed)
np.random.seed(seed)
z = np.load(fin)
mse = tf.keras.losses.MeanSquaredError()
est = ParametricUMAP(n_neighbors=15, min_dist=0.1, random_state=seed, verbose=False,
                     parametric_reconstruction=True,
                     parametric_reconstruction_loss_fcn=mse)
est.fit(z["X"])
np.save(fout, est.inverse_transform(est.transform(z["Xte"])))
"""


def pcumap(ds, rep):
    def make():
        print(f"{ds} Parametric UMAP r{rep}", flush=True)
        Xtr, Xte, yte = data(ds, rep)
        with tempfile.TemporaryDirectory() as tmp:
            fin, fout = os.path.join(tmp, "in.npz"), os.path.join(tmp, "out.npy")
            np.savez(fin, X=Xtr, Xte=Xte)
            cmd = [common.PCUMAP_PYTHON, "-c", _PCUMAP, json.dumps([fin, fout, rep])]
            env = dict(os.environ, TF_CPP_MIN_LOG_LEVEL="3")
            subprocess.run(cmd, env=env, check=True, stdout=subprocess.DEVNULL)
            Xhat = np.load(fout)
        out = {"pcumap": error(Xhat, Xte, Xtr.mean(0))}
        if (ds, rep) == SHOWN:
            out["x_pcumap"] = np.asarray(Xhat[first_of_each_class(yte)], np.float32)
        return out

    return common.cached(f"{DIR}/{ds}_pcumap_r{rep}.npz", make)


def sampled(ds, seed):
    # the other way to decode: residuals drawn from the fitted density. On the 80/20
    # split of the density figure, the verdict of the coverage certificate and how wide
    # the draws are in the input, against the fitted points next to each held-out cell
    def make():
        from flodr import FloDR
        from flodr.conformal import sample_fibre
        from scipy.spatial import cKDTree

        print(f"{ds} draws s{seed}", flush=True)
        X = common.load(ds)[0]
        tr, te = common.split(len(X))
        est = FloDR(random_state=seed, verbose=False).fit(X[tr])
        cert = est.coverage_certificate(X[te])
        sub = te[np.random.default_rng(seed).permutation(len(te))[:2000]]
        Y = est.transform(X[sub])
        D = sample_fibre(est, Y, n_draws=64, seed=seed)[0]
        near = X[tr][cKDTree(est.embedding_).query(Y, k=30)[1]]
        span = np.subtract(*np.percentile(D, [95, 5], axis=0))
        ref = np.subtract(*np.percentile(near, [95, 5], axis=1))
        return {
            "verdict": cert["certifies"],
            "cov90": cert["fibre"]["cov90"],
            "width": float(np.median(span / ref)),
        }

    return common.cached(f"{DIR}/draws_{ds}_s{seed}.npz", make)


def draw():
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D

    plt.rcParams.update({**common.RC, "axes.linewidth": 0.6})
    plt.rcParams.update({"xtick.major.width": 0.6, "ytick.major.width": 0.6})
    res = {
        ds: [dict(run(ds, r)) | dict(pcumap(ds, r)) for r in range(REPS[ds])]
        for ds in SETS
    }
    W, H = common.TEXT_W, 2.42
    fig = plt.figure(figsize=(W, H))

    def box(x, y, w, h):
        return fig.add_axes([x / W, y / H, w / W, h / H])

    # a: the first held-out image of each class and what each inverse returns for its
    # position
    shown = res[SHOWN[0]][SHOWN[1]]
    rows = {"x_truth": "held out", "x_flodr": "FloDR", "x_umap": "UMAP"}
    rows |= {"x_pcumap": "Parametric\nUMAP", "x_pca": "PCA"}
    side, x0, top = 0.232, 0.80, H - 0.20
    lo, hi = shown["x_truth"].min(), np.percentile(shown["x_truth"], 99.5)
    for r, (key, name) in enumerate(rows.items()):
        y0 = top - (r + 1) * side - (0.07 if r else 0)
        for c, img in enumerate(shown[key]):
            ax = box(x0 + c * side, y0, side - 0.014, side - 0.014)
            ax.imshow(img.reshape(28, 28), cmap="gray_r", vmin=lo, vmax=hi)
            ax.set(xticks=[], yticks=[])
            for sp in ax.spines.values():
                sp.set(linewidth=0.3, color="0.6")
            if r == 0 and c == 0:
                common.letter(ax, "a", dx=-46)
        kw = dict(ha="right", va="center", fontsize=7.5, linespacing=0.95)
        fig.text((x0 - 0.07) / W, (y0 + side / 2) / H, name, **kw)

    # b: the squared error of every decoder on the held-out points, over that of the
    # training mean
    ax = box(3.72, 0.45, W - 3.72 - 0.05, top - 0.45)
    # the four own decodes, a gap, the four references on the two maps
    offs = np.r_[np.linspace(-0.43, -0.07, 4), np.linspace(0.07, 0.43, 4)]
    for g, ds in enumerate(SETS):
        for o, (key, (colour, marker, filled, _)) in zip(offs, ARMS.items()):
            v = np.array([r[key] for r in res[ds] if key in r], float)
            if len(v):
                kw = dict(
                    mfc=colour if filled else "w", mec=colour, ms=3.4 if filled else 2.9
                )
                ax.errorbar(
                    g + o,
                    v.mean(),
                    v.std(),
                    marker=marker,
                    color=colour,
                    mew=0.6,
                    elinewidth=0.6,
                    ls="",
                    zorder=3 if filled else 2,
                    **kw,
                )
        v = np.mean([r["nbr_input"] for r in res[ds]])
        ax.plot([g - 0.47, g + 0.47], [v, v], color="k", lw=0.8, zorder=1)
    ax.axhline(1, color="0.7", lw=0.6, ls=":", zorder=0)
    for g in range(1, len(SETS)):
        ax.axvline(g - 0.5, color="0.9", lw=0.5, zorder=0)
    ax.set(xlim=(-0.5, len(SETS) - 0.5), ylim=(0, 1.1), ylabel="relative squared error")
    ax.set_xticks(range(len(SETS)), [SHORT[d] for d in SETS], fontsize=7)
    ax.tick_params(axis="x", length=0)
    ax.tick_params(axis="y", labelsize=7)
    ax.spines[["top", "right"]].set_visible(False)
    common.letter(ax, "b", dx=-24)
    handles = [
        Line2D(
            [],
            [],
            marker=m,
            ls="",
            ms=3.8,
            mew=0.7,
            mec=c,
            mfc=c if f else "w",
            label=lab,
        )
        for c, m, f, lab in ARMS.values()
    ] + [Line2D([], [], color="k", lw=0.8, label="neighbour mean, input")]
    kw = dict(ncol=2, frameon=False, fontsize=6.5, handletextpad=0.3, columnspacing=1.0)
    fig.legend(
        handles=handles,
        loc="lower left",
        bbox_to_anchor=(0.02 / W, 0.0),
        labelspacing=0.35,
        **kw,
    )
    common.save(fig, "ed_fig3.pdf")
    return res


def report(res):
    # the numbers quoted in the text
    for ds in SETS:
        keys = [k for k in res[ds][0] if not k.startswith("x_")]
        print(ds, len(res[ds]), "splits")
        for k in keys:
            v = np.array([r[k] for r in res[ds] if k in r], float)
            print(f"  {k:14s} {v.mean():.4g} (sd {v.std():.2g}, n {len(v)})")
    for ds in ("paul15", "bmarrow"):
        for s in (0, 1, 2):
            r = sampled(ds, s)
            print(
                ds,
                s,
                {
                    k: str(v) if v.dtype.kind in "US" else round(float(v), 3)
                    for k, v in r.items()
                },
            )


if __name__ == "__main__":
    import matplotlib

    matplotlib.use("Agg")
    report(draw())
