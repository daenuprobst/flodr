import copy
import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D
from matplotlib.transforms import Bbox

import common

DIR = os.path.join(common.CACHE, "ed_fig2")
# training with and without the likelihood term
ARMS = {"base": {}, "nll": {"w_nll": 0.5}}
# p(r | y) as selected by the calibration of held-out cells, as a diagonal Gaussian, and
# as selected by held-out likelihood, which adds two couplings
TAILS = {"default": {}, "diag_gauss": {"grid": ((0, 128, 1500, 0.0, 0.0),)}}
TAILS["likelihood"] = {"select": "likelihood"}
TAIL_NAME = dict(zip(TAILS, ("scale mixture", "Gaussian", "two couplings")))
TEAL, MUTED, GREY, GLOSS = "#2E8B7A", "#DCD3DC", "0.55", "0.35"
COLOUR = {"base": common.OURS, "nll": TEAL, "default": common.OURS}
COLOUR.update(diag_gauss=TEAL, likelihood=TEAL)


def paul15(arm, seed):
    # fitted on 80% of the cells: the spread of each residual coordinate around the mean
    # of the 15 nearest training cells on the map, for training and held-out cells, and
    # log p(x) of both, on the cells and with one draw of the input noise
    import torch
    from flodr import FloDR
    from sklearn.neighbors import NearestNeighbors

    X = common.load("paul15")[0]
    perm = np.random.default_rng(seed).permutation(len(X))
    cut = int(0.8 * len(X))
    est = FloDR(random_state=seed, verbose=False, advanced=ARMS[arm] or None)
    est.fit(X[perm[:cut]])
    est.score_samples(X[perm[:2]])
    dev = next(est.flow_.parameters()).device
    out, y, r = {}, {}, {}
    with torch.no_grad():
        for part, rows in (("train", perm[:cut]), ("heldout", perm[cut:])):
            z = est._to_coords(X[rows]).astype(np.float32)
            v = est.flow_.forward_main(torch.as_tensor(z, device=dev)).cpu().numpy()
            y[part], r[part] = v[:, :2], v[:, 2:]
            for tag, zz in (("clean", z), ("noised", est._noised(z))):
                logp = est.flow_.log_prob(torch.as_tensor(zz, device=dev))
                out[f"logp_{part}_{tag}"] = logp.cpu().numpy()
    nn = NearestNeighbors(n_neighbors=common.K + 1).fit(y["train"])
    near = nn.kneighbors(y["train"], return_distance=False)[:, 1:]
    out["sd_train"] = (r["train"] - r["train"][near].mean(1)).std(0)
    near = nn.kneighbors(y["heldout"], return_distance=False)[:, : common.K]
    out["sd_heldout"] = (r["heldout"] - r["train"][near].mean(1)).std(0)
    return out


def bmarrow():
    # fitted on 80% of the cells: the certificate's ranks of the held-out cells under
    # the three tails, two of the cells against exact draws from their fibres, and p(y)
    import torch
    from flodr import FloDR

    X, labels = common.load("bmarrow")
    tr, te = common.split(len(X))
    est = FloDR(random_state=0, verbose=False).fit(X[tr])
    out, fits = dict(Y=est.embedding_), {}
    for name, kw in TAILS.items():
        fits[name] = v = copy.deepcopy(est) if kw else est
        if kw:
            # the same map, its density refitted (private attributes)
            v.flow_._density_args = None
            v.flow_.tail, v.flow_._cond = torch.nn.ModuleList(), None
            noise = v._density_sd()
            v.flow_.fit_cond_tail_cv(v._coords, seed=v._seed, noise_sd=noise, **kw)
        cert = v.coverage_certificate(X[te])
        out[f"fibre_{name}"] = cert["fibre"]["ranks"]
        if name == "default":
            out["map"], out["band"] = cert["map"]["ranks"], cert["fibre"]["band"]

    # an erythroid cell at the rank nearest 0.5 among those the Gaussian ranks above
    # 0.97, and the B-lineage cell nearest 0.05 among those ranked 0.03 to 0.07, and
    # below 0.005 by the Gaussian
    u, g, kind = out["fibre_default"], out["fibre_diag_gauss"], labels[te].astype(str)

    def nearest(mask, rank):
        rows = np.flatnonzero(mask)
        return rows[np.abs(u[rows] - rank).argmin()]

    cells = [nearest((kind == "erythroid") & (g >= 0.97), 0.5)]
    mask = (kind == "B_lineage") & (u >= 0.03) & (u <= 0.07) & (g < 0.005)
    cells.append(nearest(mask, 0.05))
    for name in ("default", "diag_gauss"):
        v = fits[name]
        flow, cond = v.flow_, v.flow_.cond
        dev = next(flow.parameters()).device
        # the certificate's noised copies, drawn for all held-out cells at once
        x = torch.as_tensor(v._noised(v._to_coords(X[te]))[cells], device=dev)
        gen = torch.Generator(device=dev).manual_seed(31)

        def ell(z, logdet):
            y, r = z[:, :2], z[:, 2:]
            return cond.log_prob(y, r) - cond.y_log_prob(y) + logdet

        with torch.no_grad():
            z, logdet = flow.forward_with_logdet(x)
            y, draws = z[:, :2], []
            for _ in range(199):
                xm = flow.inverse(torch.cat([y, cond.sample_r(y, gen)], 1))
                draws.append(ell(*flow.forward_with_logdet(xm)))
            out[f"{name}_cell"] = ell(z, logdet).cpu().numpy()
            out[f"{name}_draws"] = torch.stack(draws, 1).cpu().numpy()
            out["y_cells"] = y.cpu().numpy()

    # p(y) is the same under every tail; its 50, 80 and 95% regions from its own draws
    cond = est.flow_.cond
    dev = cond.ymu.device
    lo, hi = out["Y"].min(0), out["Y"].max(0)
    rad, c = float((hi - lo).max()) * 0.53, (lo + hi) / 2
    gx = np.linspace(c[0] - rad, c[0] + rad, 600, dtype=np.float32)
    gy = np.linspace(c[1] - rad, c[1] + rad, 600, dtype=np.float32)
    grid = torch.as_tensor(np.stack(np.meshgrid(gx, gy), -1).reshape(-1, 2), device=dev)
    gen = torch.Generator(device=dev).manual_seed(7)
    with torch.no_grad():
        logp = torch.cat([cond.y_log_prob(g) for g in grid.split(65536)])
        w = torch.softmax(cond.ylogw, 0)
        comp = torch.multinomial(w, 200_000, replacement=True, generator=gen)
        noise = torch.randn((200_000, 2), device=dev, generator=gen)
        yd = cond.ymu[comp] + torch.exp(0.5 * cond.ylogvar[comp]) * noise
        ld = torch.cat([cond.y_log_prob(g) for g in yd.split(65536)]).cpu().numpy()
    out.update(gx=gx, gy=gy, logp_y=logp.cpu().numpy().reshape(600, 600))
    out["levels"] = np.quantile(ld, [0.05, 0.2, 0.5])
    return out


W, H, TOP = common.TEXT_W, 5.55, 1.55
# in inches: the tops of the two rows of data panels under a, the left edge of c and e,
# and that of their letters
ROW_BC, ROW_DE, RIGHT, LETTER_R = 0.12, 2.02, 4.28, 3.80
plt.rcParams.update(common.RC)
fig = plt.figure(figsize=(W, H))


def axes(left, top, w, h, strip=False):
    # from the top of the data panels, or of the figure for the strip of a
    top += 0 if strip else TOP
    return fig.add_axes([left / W, 1 - (top + h) / H, w / W, h / H])


def text(x, top, s, **kw):
    kw = dict(ha="left", va="baseline") | kw
    return fig.text(x / W, 1 - top / H, s, **kw)


def letter(s, x, top):
    text(x, top, r"\textbf{" + s + "}", fontsize=9)


def frame(ax):
    ax.spines[["top", "right"]].set_visible(False)
    ax.spines[["left", "bottom"]].set_linewidth(0.5)
    ax.tick_params(length=2, width=0.5, pad=1.5)


# a, a toy with two input dimensions and one map coordinate: the input on a crescent and
# the same cells unrolled into (y, r)
rng = np.random.default_rng(3)
y = np.where(
    rng.random(60) < 0.55, rng.normal(0.32, 0.07, 60), rng.normal(0.70, 0.06, 60)
)
y = y.clip(0.05, 0.95)
r = rng.choice([0.5, 1.0, 2.2], 60, p=[0.3, 0.5, 0.2]) * rng.standard_normal(60)
r = r.clip(-2.4, 2.4)
ys, r_draw, r_cell = 0.62, np.array([-2.3, -1.0, -0.45, 0.05, 0.5, 1.1, 2.2]), 0.75
rr, fibres = np.linspace(-2.5, 2.5, 60), np.linspace(0.1, 0.9, 7)
open_dot = dict(s=11, facecolor="white", edgecolor=common.OURS, lw=0.7, zorder=4)


def to_x(yy, rr):
    phi, rad = np.deg2rad(150 - 120 * yy + 4 * rr), 2.2 + 0.32 * rr
    return rad * np.cos(phi), rad * np.sin(phi)


ax = axes(0.12, 0.10, 1.85, 1.40, strip=True)
for yf in fibres:
    ax.plot(*to_x(yf, rr), color="0.82", lw=0.4, zorder=1)
ax.plot(*to_x(ys, rr), color=common.OURS, lw=0.9, zorder=2)
ax.scatter(*to_x(y, r), s=5, color=GREY, lw=0, zorder=3)
ax.scatter(*to_x(ys, r_draw), **open_dot)
ax.scatter(*to_x(ys, r_cell), s=11, color="black", lw=0, zorder=5)
arrow = dict(arrowstyle="-", color=common.OURS, lw=0.5, shrinkB=1.5)
kw = dict(color=common.OURS, fontsize=7, ha="center", va="center", arrowprops=arrow)
ax.annotate(r"fibre of $y^*$", to_x(ys, -1.5), xytext=(-0.1, 0.95), **kw)
ax.set(aspect="equal", xlim=(-2.9, 2.9), ylim=(0.45, 3.1))
ax.axis("off")
kw = dict(transform=ax.transAxes, ha="center", va="top", fontsize=7, color=GLOSS)
ax.text(0.5, -0.04, r"input $x$", **kw)

# the unrolled view, p(y) under it and p(r | y*) on its side
ax2 = axes(2.70, 0.12, 1.62, 1.00, strip=True)
for yf in fibres:
    ax2.plot([yf, yf], [-2.5, 2.5], color="0.82", lw=0.4, zorder=1)
ax2.plot([ys, ys], [-2.5, 2.5], color=common.OURS, lw=0.9, zorder=2)
ax2.scatter(y, r, s=5, color=GREY, lw=0, zorder=3)
ax2.scatter(np.full(7, ys), r_draw, **open_dot)
ax2.scatter([ys], [r_cell], s=11, color="black", lw=0, zorder=5)
rg = np.linspace(-2.5, 2.5, 200)
mix = ((0.3, 0.5), (0.5, 1.0), (0.2, 2.2))
pr = sum(w * np.exp(-0.5 * (rg / s) ** 2) / s for w, s in mix)
pr = 0.12 * pr / pr.max()
ax2.fill_betweenx(rg, ys - pr, ys, color=common.OURS_LIGHT, lw=0, zorder=0)
ax2.plot(ys - pr, rg, color=common.OURS, lw=0.6, zorder=2)
kw = dict(color=common.OURS, fontsize=7, ha="right", va="top")
ax2.text(ys - 0.035, 3.05, r"$p(r \mid y^*)$", **kw)
ax2.set(xlim=(0, 1), ylim=(-2.8, 3.1), xticks=[], yticks=[])
for sp in ax2.spines.values():
    sp.set(linewidth=0.5, color="0.4")
ax2.set_ylabel(r"residual $r$", fontsize=7, color=GLOSS, labelpad=2)
ax3 = axes(2.70, 1.14, 1.62, 0.26, strip=True)
yg = np.linspace(0, 1, 200)
py = 0.55 * np.exp(-0.5 * ((yg - 0.32) / 0.07) ** 2) / 0.07
py += 0.45 * np.exp(-0.5 * ((yg - 0.70) / 0.06) ** 2) / 0.06
ax3.fill_between(yg, 0, py, color=common.OURS_LIGHT, lw=0)
ax3.plot(yg, py, color=common.OURS, lw=0.6)
ax3.axvline(ys, color=common.OURS, lw=0.9)
ax3.set(xlim=(0, 1), ylim=(0, py.max() * 1.1))
ax3.axis("off")
kw = dict(transform=ax3.transAxes, fontsize=7, va="top")
ax3.text(0.02, 0.98, r"$p(y)$", color=common.OURS, ha="left", **kw)
ax3.text(0.5, -0.08, r"map $y$", color=GLOSS, ha="center", **kw)

# f above its arrow and the inverse below its own, centred between what is drawn of the
# two views. The axes of the input view are invisible, so its extent is that of its
# curves, labels and points, the points from their offsets
fig.canvas.draw()
rend, inch = fig.canvas.get_renderer(), fig.dpi_scale_trans.inverted()
drawn = [*ax.lines, *(t for t in ax.texts if type(t) is matplotlib.text.Text)]
pts = ax.transData.transform(np.vstack([c.get_offsets() for c in ax.collections]))
boxes = [a.get_window_extent(rend) for a in drawn] + [Bbox([pts.min(0), pts.max(0)])]
bl = Bbox.union(boxes).transformed(inch)
br = Bbox.union([a.get_tightbbox(rend) for a in (ax2, ax3)]).transformed(inch)
xm, half = (bl.x1 + br.x0) / 2, 0.3 * (br.x0 - bl.x1)
ym = H - (bl.y0 + bl.y1 + br.y0 + br.y1) / 4
for sign, label in ((1, r"$f_\theta$"), (-1, r"$f_\theta^{-1}$")):
    x0, x1, yy = xm - sign * half, xm + sign * half, ym - sign * 0.13
    kw = dict(arrowstyle="-|>", mutation_scale=7, lw=0.7, color="0.2")
    ends = (x0 / W, 1 - yy / H), (x1 / W, 1 - yy / H)
    arrow = matplotlib.patches.FancyArrowPatch(*ends, transform=fig.transFigure, **kw)
    fig.add_artist(arrow)
    va = "bottom" if sign > 0 else "top"
    text(xm, yy - sign * 0.05, label, ha="center", va=va)

# the split of the density
x, top, xl = 4.55, 0.38, 6.00
gloss = dict(fontsize=7, color=GLOSS)
text(x, top, r"$\log p(x) \;=\; \log p(y)$")
text(xl, top, "on the map", **gloss)
text(x + 0.53, top + 0.24, r"$+\;\log p(r \mid y)$")
text(xl, top + 0.24, "residual at that position", **gloss)
text(x + 0.53, top + 0.48, r"$+\;\log \lvert \det J \rvert$")
text(xl, top + 0.48, "volume change", **gloss)
text(x, top + 0.86, r"$\ell \;=\; \log p(x) - \log p(y)$")
text(xl, top + 0.86, "ranked among fibre draws", **gloss)
letter("a", 0.02, 0.12)

# b, c: paul15 with and without the likelihood term in training, three seeds
fits = {}
for arm in ARMS:
    for s in common.SEEDS:
        path = os.path.join(DIR, f"paul15_{arm}_s{s}.npz")
        fits[arm, s] = common.cached(path, lambda: paul15(arm, s))
ax = axes(0.58, ROW_BC, 3.02, 1.30)
for (arm, s), z in fits.items():
    ratio = np.sort(z["sd_train"] / z["sd_heldout"])
    ax.plot(np.arange(1, len(ratio) + 1), ratio, color=COLOUR[arm], lw=0.8)
ax.axhline(0.1, color="black", lw=0.6, ls=(0, (3, 2)))
ax.set(yscale="log", ylim=(2e-4, 2), xlim=(0.5, 48.5), xticks=[1, 24, 48])
ax.set_xlabel("residual coordinate, sorted", labelpad=1.5)
ax.set_ylabel("training / held-out spread", labelpad=2)
ax.text(2, 0.95, "default", color=COLOUR["base"], ha="left", va="bottom")
ax.text(2, 1.5e-3, "+NLL", color=COLOUR["nll"], ha="left", va="bottom")
frame(ax)
ax.tick_params(axis="y", which="minor", length=1, width=0.4)
letter("b", 0.02, TOP + ROW_BC - 0.06)
text(0.07, TOP + ROW_BC + 0.65, "paul15", rotation=90, ha="center", va="center")

ax = axes(RIGHT, ROW_BC, W - 0.08 - RIGHT, 1.30)
for (arm, s), z in fits.items():
    clean = z["logp_train_clean"].mean() - z["logp_heldout_clean"].mean()
    noised = z["logp_train_noised"].mean() - z["logp_heldout_noised"].mean()
    print(f"paul15 {arm} seed {s}: {clean:.1f} nats, {noised:.1f} with the noise")
    at, kw = list(ARMS).index(arm) + 0.12 * (s - 1), dict(s=13, color=COLOUR[arm])
    ax.scatter(clean, at, lw=0, zorder=4, **kw)
    ax.scatter(noised, at, facecolor="none", lw=0.8, zorder=3, **kw)
ax.axvline(0, color="0.6", lw=0.5, zorder=1)
ax.set(xlim=(-2, 31), ylim=(1.55, -0.55))
ax.set_yticks([0, 1], ["default", "+NLL"])
for t, arm in zip(ax.get_yticklabels(), ARMS):
    t.set_color(COLOUR[arm])
ax.set_xlabel(r"mean $\log p$, training minus held-out (nats)", labelpad=1.5)
frame(ax)
ax.tick_params(axis="y", length=0)
kw = dict(ls="", marker="o", color="0.3")
handles = [Line2D([], [], ms=3.6, mew=0, label="on the cells", **kw)]
kw.update(ms=3.4, mfc="none", mew=0.8, label="with the input noise")
handles.append(Line2D([], [], **kw))
kw = dict(frameon=False, fontsize=7, handletextpad=0.2, labelspacing=0.3)
ax.legend(handles=handles, loc="center", bbox_to_anchor=(0.5, 0.5), **kw)
letter("c", LETTER_R, TOP + ROW_BC - 0.06)

# d: two held-out bone marrow cells on the map, and l of the draws from their fibres
z = common.cached(os.path.join(DIR, "bmarrow.npz"), bmarrow)
ax = axes(0.30, ROW_DE, 1.55, 1.55)
ax.scatter(*z["Y"].T, s=0.5, color="0.75", lw=0, rasterized=True, zorder=1)
base = np.asarray(matplotlib.colors.to_rgb(common.OURS))
shades = [tuple(base + (1 - base) * f) for f in (0.62, 0.31, 0.0)]
kw = dict(levels=z["levels"], colors=shades, linewidths=0.5, zorder=2)
ax.contour(z["gx"], z["gy"], z["logp_y"], **kw)
ax.scatter(*z["y_cells"].T, s=9, color="black", lw=0, zorder=4)
for i, dx in enumerate((6, -6)):
    kw = dict(textcoords="offset points", ha="left" if dx > 0 else "right", zorder=5)
    kw["bbox"] = dict(boxstyle="square,pad=0.08", fc="white", ec="none")
    ax.annotate(str(i + 1), z["y_cells"][i], xytext=(dx, 0), va="center", **kw)
common.square(ax, z["Y"], crop=(5, 95))
for sp in ax.spines.values():
    sp.set(linewidth=0.5, color="0.4")
letter("d", 0.02, TOP + ROW_DE - 0.06)
kw = dict(rotation=90, ha="center", va="center")
text(0.07, TOP + ROW_DE + 0.78, common.NAMES["bmarrow"], **kw)

ax = axes(2.02, ROW_DE, 1.40, 1.55)
# where the draws of each cell sit, the selected tail above the Gaussian
lo, hi, ROWS = -46, 86, (3.2, 2.45, 1.0, 0.25)
for cell, tail, yy in zip((0, 0, 1, 1), ("default", "diag_gauss") * 2, ROWS):
    d = z[f"{tail}_draws"][cell] - z[f"{tail}_cell"][cell]
    # shaded, the lowest tenth of the draws
    box = (lo, yy - 0.3), np.quantile(d, 0.1) - lo, 0.6
    ax.add_patch(matplotlib.patches.Rectangle(*box, color=MUTED, lw=0))
    ax.vlines(d, yy - 0.27, yy + 0.27, color=COLOUR[tail], lw=0.35, zorder=2)
    # the share of these draws below the cell, ties at one half as in the certificate
    u = ((d < 0).sum() + 0.5) / (len(d) + 1)
    kw = dict(color=COLOUR[tail], fontsize=7.5, ha="left", va="center")
    ax.text(hi + 3, yy, f"{u:.2f}" if u >= 0.01 else f"{u:.3f}", **kw)
ax.axvline(0, color="black", lw=0.8, zorder=3)
ax.set(xlim=(lo, hi), xticks=[-40, 0, 40, 80], ylim=(-0.25, 3.7))
ax.set_yticks([2.825, 0.625], ["1", "2"])
ax.text(hi + 3, 3.62, "rank", fontsize=7, color=GLOSS, ha="left", va="bottom")
ax.set_xlabel(r"$\ell$ of draw minus $\ell$ of cell (nats)", labelpad=1.5)
frame(ax)
ax.spines["left"].set_visible(False)
ax.tick_params(axis="y", length=0)

# e: the ranks of all held-out cells against the uniform, in the fibre and on the map
ax = axes(RIGHT, ROW_DE, W - 0.08 - 0.78 - 0.12 - RIGHT, 1.55)
axm = axes(W - 0.08 - 0.78, ROW_DE, 0.78, 1.55)
for name, ls in zip(TAILS, ("-", "-", (0, (3, 1.6)))):
    u = np.sort(z[f"fibre_{name}"])
    kw = dict(color=COLOUR[name], ls=ls, lw=0.9, label=TAIL_NAME[name])
    ax.plot(u, np.arange(1, len(u) + 1) / len(u) - u, **kw)
u = np.sort(z["map"])
axm.plot(u, np.arange(1, len(u) + 1) / len(u) - u, color=common.OURS, lw=0.9)
for a in (ax, axm):
    for c in (0.1, -0.1):
        a.axhspan(c - z["band"], c + z["band"], color=MUTED, lw=0, zorder=0)
    a.axhline(0, color="0.6", lw=0.5, zorder=0)
    a.axvline(0.1, color="0.3", lw=0.6, ls=(0, (1, 1.5)), zorder=1)
    a.set(xlim=(0, 1), ylim=(-0.30, 0.12), yticks=[-0.3, -0.2, -0.1, 0, 0.1])
    frame(a)
ax.set_xticks([0, 0.5, 1], ["0", "0.5", "1"])
axm.set_xticks([0.5, 1], ["0.5", "1"])
axm.set_yticklabels([])
ax.set_xlabel("rank in the fibre", labelpad=1.5)
axm.set_xlabel("on the map", labelpad=1.5)
ax.set_ylabel("ECDF minus rank", labelpad=2)
kw = dict(frameon=False, fontsize=7, handlelength=1.8, handletextpad=0.4)
ax.legend(loc="lower left", labelspacing=0.3, borderaxespad=0.2, **kw)
letter("e", LETTER_R, TOP + ROW_DE - 0.06)
fig.savefig(common.out("ed_fig2.pdf"), dpi=300)
