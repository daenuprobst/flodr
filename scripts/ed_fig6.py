import copy
import os
import pickle

import numpy as np

import common

DIR = os.path.join(common.CACHE, "ed_fig6")
DATASETS = {"paul15": "paul15", "bmarrow": "Bone marrow", "mnist10k": "MNIST"}
VERDICT = {"certified": "#44AA99", "undecided": "#9A9A9A", "refused": "#332288"}
CLEAN = "#5b6b7a"
COVERAGE = common.cmap([(0.0, common.OURS_DARK), (0.5, "#dcdcdc"), (1.0, "#44AA99")])


def gap(est, Xtr, Xte):
    # mean log p(x) - log p(y) of training minus held-out cells, each with one draw
    # of the density's input noise from a generator on the model's device
    import torch

    flow, sd = est.flow_, est.flow_._density_noise
    dev, ell = next(flow.parameters()).device, []
    for X, seed in ((Xtr, 4321), (Xte, 4322)):
        z = torch.from_numpy(est._to_coords(X)).to(dev)
        if sd is not None:
            gen = torch.Generator(dev).manual_seed(seed)
            z = z + sd * torch.randn(z.shape, device=dev, generator=gen)
        with torch.no_grad():
            l = [
                flow.log_prob(c)
                - flow.cond.y_log_prob(flow.forward_main(c)[:, : flow.k])
                for c in z.split(16384)
            ]
        ell.append(float(torch.cat(l).mean()))
    return ell[0] - ell[1]


def fitted(ds, seed):
    # an 80/20 fit and the coverage certificate of its density on the held-out cells,
    # at the default input noise and refitted without it
    path = os.path.join(DIR, f"{ds}_s{seed}")

    def make():
        import torch
        from flodr import FloDR

        print(f"fit {ds} seed {seed}", flush=True)
        X = common.paul15_pcs()[0] if ds == "paul15" else common.load(ds)[0]
        tr, te = common.split(len(X))
        est = FloDR(random_state=seed, verbose=False).fit(X[tr])
        out = dict(Y=est.embedding_)
        for cond in ("noisy", "clean"):
            e = est if cond == "noisy" else copy.deepcopy(est)
            if cond == "clean":
                # the same layout, its density refitted on clean cells (private attributes)
                e.flow_.tail, e.flow_._cond = torch.nn.ModuleList(), None
                e.flow_._density_noise = e.flow_._density_args = None
            c = e.coverage_certificate(X[te])
            res = c["fibre"]["ranks"], c["map"]["ranks"], c["positions"]
            res += (gap(e, X[tr], X[te]),)
            out |= {
                f"{k}_{cond}": v for k, v in zip(("fibre", "map", "pos", "gap"), res)
            }
        with open(path + ".pkl", "wb") as f:
            pickle.dump(est, f)
        return out

    return common.cached(path + ".npz", make)


def model(ds, seed):
    fitted(ds, seed)
    with open(os.path.join(DIR, f"{ds}_s{seed}.pkl"), "rb") as f:
        return pickle.load(f)


def coverage_field(fit, k=100):
    # per training cell, the share of the k held-out cells nearest on the map that lie
    # inside the 90% region of their own fibre (knn_faiss returns k + 1)
    nn = common.knn_faiss(fit["pos_noisy"], fit["Y"], k - 1)
    return (fit["fibre_noisy"] >= 0.1)[nn].mean(1)


if __name__ == "__main__":
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from flodr import viz
    from matplotlib.colors import TwoSlopeNorm
    from matplotlib.lines import Line2D
    from scipy.stats import kstest

    plt.rcParams.update({**common.RC, "axes.titlesize": 8, "axes.labelsize": 7.5})
    plt.rcParams.update({"xtick.labelsize": 6.5, "ytick.labelsize": 6.5})
    plt.rcParams.update({"xtick.major.size": 2, "ytick.major.size": 2})
    W, H = common.TEXT_W, 5.6
    fig = plt.figure(figsize=(W, H))

    def axes(x, y, w, h):
        return fig.add_axes([x / W, y / H, w / W, h / H])

    # a: held-out coverage over the bone marrow map, seed 0
    fit = fitted("bmarrow", 0)
    field = coverage_field(fit)
    print(
        f"coverage above 0.95 at {(field > 0.95).mean():.0%} of cells, below 0.8 at "
        f"{(field < 0.8).mean():.0%}"
    )
    norm, ax = TwoSlopeNorm(0.9, 0.5, 1.0), axes(0.12, 3.09, 2.3, 2.3)
    kw = dict(cmap=COVERAGE, norm=norm, s=0.25, order="low")
    viz.plot_embedding(fit["Y"], values=field, ax=ax, **kw).set_title("Bone marrow")
    common.letter(ax, "a")
    cb = fig.colorbar(
        plt.cm.ScalarMappable(norm, COVERAGE), axes(2.48, 3.39, 0.07, 1.7)
    )
    cb.set_ticks([0.5, 0.7, 0.9, 1], labels=["$\\le$0.5", "0.7", "0.9", "1"])
    cb.outline.set_linewidth(0.5)

    # b, c: verdict and likelihood gap of every certificate, the larger KS distance of
    # the fibre and map ranks to uniform against the DKW band at margin 0.1; d-f: ranks
    axb, axc = axes(3.55, 3.09, 1.62, 2.3), axes(5.37, 3.09, 1.62, 2.3)
    y, yt = 0.0, []
    for i, (ds, name) in enumerate(DATASETS.items()):
        top, ranks = y, []
        for cond, colour in (("clean", CLEAN), ("noisy", common.OURS)):
            for s in (0, 1, 2):
                f = fitted(ds, s)
                ranks.append(f[f"fibre_{cond}"])
                ks = max(
                    kstest(f[f"{r}_{cond}"], "uniform")[0] for r in ("fibre", "map")
                )
                eps = np.sqrt(np.log(2 / 0.05) / (2 * len(ranks[-1])))
                v = "certified" if ks + eps <= 0.1 else "undecided"
                v = "refused" if ks - eps > 0.1 else v
                print(
                    f"{ds} seed {s} {cond}: n {len(ranks[-1])} KS {ks:.4f} band "
                    f"{eps:.4f} {v}, gap {f[f'gap_{cond}']:.3f} nats"
                )
                axb.hlines(y, ks - eps, ks + eps, color=VERDICT[v], lw=1)
                mk = dict(ms=3.4, mew=0.8, **(dict(mfc="w") if cond == "clean" else {}))
                axb.plot(ks, y, "o", color=VERDICT[v], **mk)
                axc.plot(f[f"gap_{cond}"], y, "o", color=colour, **mk)
                y -= 1
            y -= 0.5
        yt.append((top + y + 1.5) / 2)
        y -= 0.8
        ax = axes(0.55 + 2.34 * i, 0.4, 1.72, 1.72)
        viz.plot_ranks(ranks, ax, [CLEAN] * 3 + [common.OURS] * 3).set_title(name)
        ax.set(xlabel="held-out fibre rank", ylabel="" if i else "fraction of cells")
        ax.spines[["top", "right"]].set_visible(False)
        common.letter(ax, "def"[i])
    axb.axvline(0.1, color="0.3", lw=0.6, ls=(0, (3, 2)), zorder=1)
    axb.set(xlim=(0, 0.36), xlabel="largest KS distance")
    axc.axvline(0, color="0.3", lw=0.6, zorder=1)
    axc.set_xscale("symlog", linthresh=1.0, linscale=0.6)
    axc.set(xlim=(-1.5, 300), xlabel="train $-$ held-out (nats)")
    axc.set_xticks([-1, 0, 1, 10, 100], labels=["$-1$", "0", "1", "10", "100"])
    for a, s, title in ((axb, "b", "verdict"), (axc, "c", "likelihood gap")):
        a.set(title=title, ylim=(y + 0.3, 0.7), yticks=yt, yticklabels=[])
        a.tick_params(axis="y", length=0)
        a.grid(True, axis="x", lw=0.4, color="0.9", zorder=0)
        a.set_axisbelow(True)
        a.spines[["top", "right"]].set_visible(False)
        common.letter(a, s)
    axb.set_yticklabels(DATASETS.values(), fontsize=7)

    kw = dict(lw=0.9, marker="o", ms=3.4)
    handles = [
        Line2D([], [], color=CLEAN, mfc="w", label="clean density", **kw),
        Line2D([], [], color=common.OURS, label="density at input noise 0.5", **kw),
    ] + [Line2D([], [], color=c, lw=1, label=v) for v, c in VERDICT.items()]
    kw = dict(ncol=5, frameon=False, fontsize=7, handletextpad=0.4, columnspacing=1.2)
    fig.legend(handles=handles, loc="center", bbox_to_anchor=(0.5, 2.54 / H), **kw)
    common.save(fig, "ed_fig6.pdf")
