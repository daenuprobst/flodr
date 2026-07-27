import numpy as np
import torch
import matplotlib.pyplot as plt
from matplotlib.colors import to_rgb
from scipy.spatial import cKDTree

RC = {
    "font.size": 11,
    "axes.titlesize": 12,
    "figure.titlesize": 15,
    "legend.fontsize": 8,
    "font.family": "sans-serif",
    "pdf.fonttype": 42,
    "ps.fonttype": 42,
    "savefig.facecolor": "white",
}

# publication style (NeurIPS-sized: 5.5in text width, serif)
RC_PAPER = {
    "font.size": 8,
    "axes.titlesize": 8,
    "legend.fontsize": 7,
    "xtick.labelsize": 7,
    "ytick.labelsize": 7,
    "font.family": "serif",
    "mathtext.fontset": "stix",
    "font.serif": ["Times New Roman", "STIXGeneral", "DejaVu Serif"],
    "pdf.fonttype": 42,
    "ps.fonttype": 42,
    "savefig.facecolor": "white",
    "axes.linewidth": 0.6,
}


def residual(flow, X):
    device = next(flow.parameters()).device

    with torch.no_grad():
        return (
            flow.forward_main(torch.as_tensor(X, dtype=torch.float32, device=device))[
                :, flow.k :
            ]
            .cpu()
            .numpy()
        )


def residual_fraction(flow, X):
    device = next(flow.parameters()).device
    with torch.no_grad():
        z = flow.forward_main(torch.as_tensor(X, dtype=torch.float32, device=device))

        return float(
            (z[:, flow.k :].norm(dim=1) / z.norm(dim=1).clamp(min=1e-12)).median()
        )


def occupancy_field(Y, n_grid=100, scale=5.0, pad=0.06):
    Y = np.asarray(Y, float)
    lo, hi = Y.min(0) - pad * np.ptp(Y, 0), Y.max(0) + pad * np.ptp(Y, 0)
    xs, ys = np.linspace(lo[0], hi[0], n_grid), np.linspace(lo[1], hi[1], n_grid)
    grid_x, grid_y = np.meshgrid(xs, ys)
    tree = cKDTree(Y)
    med_nn = float(np.median(tree.query(Y, k=2)[0][:, 1]))
    dist = tree.query(np.column_stack([grid_x.ravel(), grid_y.ravel()]), k=1)[
        0
    ].reshape(n_grid, n_grid)
    return xs, ys, np.clip(1.0 - (dist / (scale * med_nn + 1e-12)) ** 2, 0.0, 1.0)


def _sample_fiber(flow, yb, n_samples, gen, to_input):
    cond = flow.cond
    mu, logvar = cond.r_params(yb)
    sigma = torch.exp(0.5 * logvar)
    out = []

    for _ in range(n_samples):
        r = mu + sigma * torch.randn(mu.shape, device=yb.device, generator=gen)
        x = flow.inverse(torch.cat([yb, r], 1)).cpu().numpy()
        out.append(np.asarray(to_input(x) if to_input is not None else x, np.float64))

    return np.stack(out)


def conditional_spread(flow, Y, to_input=None, n_samples=64, seed=0, chunk=1024):
    return conditional_moments(
        flow, Y, to_input=to_input, n_samples=n_samples, seed=seed, chunk=chunk
    )[1]


def conditional_moments(flow, Y, to_input=None, n_samples=64, seed=0, chunk=1024):
    if flow.cond is None:
        raise ValueError(
            "no conditional base: fit with density enabled (cond_tail > 0) first"
        )

    device = next(flow.parameters()).device
    y_torch = torch.as_tensor(np.asarray(Y, np.float32), device=device)
    gen = torch.Generator(device=device)
    gen.manual_seed(seed)
    sigma = np.empty(y_torch.shape[0])
    means = []

    with torch.no_grad():
        for start in range(0, y_torch.shape[0], chunk):
            samples = _sample_fiber(
                flow, y_torch[start : start + chunk], n_samples, gen, to_input
            )
            mu = samples.mean(0)
            sigma[start : start + chunk] = np.sqrt(((samples - mu) ** 2).sum(2).mean(0))
            means.append(mu)

    return np.concatenate(means), sigma


def conditional_atypicality(flow, Z, to_input=None, n_samples=128, seed=0, chunk=512):
    if flow.cond is None:
        raise ValueError(
            "no conditional base: fit with density enabled (cond_tail > 0) first"
        )

    device = next(flow.parameters()).device
    z_torch = torch.as_tensor(np.asarray(Z, np.float32), device=device)
    gen = torch.Generator(device=device)
    gen.manual_seed(seed)

    x = np.asarray(
        to_input(np.asarray(Z, np.float32)) if to_input is not None else Z, np.float64
    )

    out = np.empty(z_torch.shape[0])

    with torch.no_grad():
        y_all = flow(z_torch)[:, : flow.k]

        for start in range(0, z_torch.shape[0], chunk):
            samples = _sample_fiber(
                flow, y_all[start : start + chunk], n_samples, gen, to_input
            )
            mu = samples.mean(0)
            d_samp = np.sqrt(((samples - mu) ** 2).sum(2))  # (n_samples, m)
            d_x = np.sqrt(((x[start : start + chunk] - mu) ** 2).sum(1))  # (m,)
            out[start : start + chunk] = (d_samp < d_x).mean(0)

    return out


def plot_field(Y, v, ax=None, cmap="Reds", s=8):
    Y = np.asarray(Y, float)
    ax = ax or plt.gca()
    sc = ax.scatter(
        Y[:, 0], Y[:, 1], c=np.asarray(v, float), cmap=cmap, s=s, linewidths=0
    )
    ax.figure.colorbar(sc, ax=ax, fraction=0.046, pad=0.04)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_aspect("equal")

    for spine in ax.spines.values():
        spine.set_color("0.8")

    return ax


def plot_embedding(Y, labels=None, ax=None, cmap="tab10", s=6, title=None):
    Y = np.asarray(Y, float)
    ax = ax or plt.gca()

    if labels is None:
        ax.scatter(Y[:, 0], Y[:, 1], s=s, c="0.3", alpha=0.6, linewidths=0)
    else:
        pal = plt.get_cmap(cmap)
        ax.scatter(
            Y[:, 0],
            Y[:, 1],
            s=s,
            alpha=0.7,
            linewidths=0,
            c=[pal(int(v) % 10) for v in np.asarray(labels)],
        )

    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_aspect("equal")

    for spine in ax.spines.values():
        spine.set_color("0.8")

    if title:
        ax.set_title(title)

    return ax


BIVARIATE_SCHEMES = {
    "accessible": ["#e7e7e3", "#2a78d6", "#e34948", "#4a3aa7"],  # CVD-validated
    # biscale / bivariatechoropleth palettes, given as corners [low-low, high-x, high-y, high-both]
    "Bluegill": ["#d3d3d3", "#74a7a3", "#976020", "#534c19"],
    "BlueGold": ["#d3d3d3", "#488fb0", "#dea301", "#4c6e01"],
    "BlueOr": ["#d3d3d3", "#dd6a29", "#169dd0", "#174f28"],
    "BlueYl": ["#d3d3d3", "#d9be00", "#0088d9", "#007b00"],
    "Brown2": ["#d3d3d3", "#b6a352", "#8b689f", "#78503e"],
    "DkBlue2": ["#d3d3d3", "#52b6b6", "#ad5b9c", "#434e87"],
    "DkCyan2": ["#d3d3d3", "#6277a5", "#699e74", "#31595b"],
    "DkViolet2": ["#d3d3d3", "#9e3547", "#4279b0", "#311e3b"],
    "GrPink2": ["#d3d3d3", "#b65252", "#5b9cad", "#4e3d43"],
    "PinkGrn": ["#d3d3d3", "#459b22", "#bc177d", "#3e1114"],
    "PurpleGrn": ["#d3d3d3", "#027a2e", "#6f2d85", "#011a1d"],
    "PurpleOr": ["#d3d3d3", "#d25601", "#563787", "#551601"],
}


def _scheme_cols(scheme):
    hexes = BIVARIATE_SCHEMES.get(scheme, scheme) if isinstance(scheme, str) else scheme
    return np.array([to_rgb(h) for h in hexes], float)


def _quantize(v, steps):
    if steps <= 1:
        return np.zeros_like(v)

    edges = np.quantile(v, np.linspace(0, 1, steps + 1))[1:-1]

    return np.clip(np.digitize(v, edges), 0, steps - 1) / (steps - 1)


def _norm01(v, log=False, floor0=False, lo=2, hi=98):
    v = np.asarray(v, float)

    if floor0:
        v = np.clip(v, 0.0, None)

    p_lo, p_hi = np.percentile(v, lo), np.percentile(v, hi)

    if log:
        p_lo = max(p_lo, 1e-9)
        return np.clip(
            (np.log(np.clip(v, p_lo, p_hi)) - np.log(p_lo))
            / (np.log(p_hi) - np.log(p_lo) + 1e-12),
            0,
            1,
        )

    return np.clip((v - p_lo) / (p_hi - p_lo + 1e-12), 0.0, 1.0)


def _bare(ax):
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_aspect("equal")

    for spine in ax.spines.values():
        spine.set_color("0.8")


def _bare3d(ax):
    ax.set_axis_off()


def depth_fog(Y, elev=22.0, azim=-60.0, strength=0.9, gamma=2.4):
    Y = np.asarray(Y, float)
    elev_r, azim_r = np.radians(elev), np.radians(azim)
    view = np.array(
        [
            np.cos(elev_r) * np.cos(azim_r),
            np.cos(elev_r) * np.sin(azim_r),
            np.sin(elev_r),
        ]
    )
    depth = (Y - Y.mean(0)) @ view  # signed distance along the line of sight

    near = (depth - depth.min()) / (
        np.ptp(depth) + 1e-12
    )  # 1 nearest the camera, 0 farthest

    weight = (1.0 - strength * (1.0 - near) ** gamma)[:, None]

    return np.argsort(depth), weight, near


def _emptiest_corner(Y, ax, box=0.30):
    Y = np.asarray(Y, float)
    (x0, x1), (y0, y1) = ax.get_xlim(), ax.get_ylim()
    frac_x = (Y[:, 0] - x0) / (x1 - x0 + 1e-12)
    frac_y = (Y[:, 1] - y0) / (y1 - y0 + 1e-12)
    anchors = [(0.0, 1 - box), (1 - box, 1 - box), (0.0, 0.0), (1 - box, 0.0)]
    counts = [
        int(
            (
                (frac_x >= cx)
                & (frac_x < cx + box)
                & (frac_y >= cy)
                & (frac_y < cy + box)
            ).sum()
        )
        for cx, cy in anchors
    ]

    return int(np.argmin(counts))


def _corner_xy(i, w, h=None, pad=0.02):
    h = w if h is None else h

    return [
        (pad, 1 - h - pad),
        (1 - w - pad, 1 - h - pad),
        (pad, pad),
        (1 - w - pad, pad),
    ][i]


def _bivar_mix(sx, sy, cols):
    sx = np.clip(np.asarray(sx, float), 0, 1)
    sy = np.clip(np.asarray(sy, float), 0, 1)

    if len(cols) == 4:
        c00, c10, c01, c11 = cols

        return np.clip(
            (1 - sx)[:, None] * (1 - sy)[:, None] * c00
            + sx[:, None] * (1 - sy)[:, None] * c10
            + (1 - sx)[:, None] * sy[:, None] * c01
            + sx[:, None] * sy[:, None] * c11,
            0,
            1,
        )

    grid = cols.reshape(3, 3, 3)  # [y, x, rgb]

    gx, gy = sx * 2, sy * 2

    xi = np.clip(np.floor(gx).astype(int), 0, 1)
    yi = np.clip(np.floor(gy).astype(int), 0, 1)
    tx, ty = (gx - xi)[:, None], (gy - yi)[:, None]

    return np.clip(
        (1 - tx) * (1 - ty) * grid[yi, xi]
        + tx * (1 - ty) * grid[yi, xi + 1]
        + (1 - tx) * ty * grid[yi + 1, xi]
        + tx * ty * grid[yi + 1, xi + 1],
        0,
        1,
    )


def _bivar_key(ax, mode, cols, steps, ci, xlabel, ylabel):
    ramp = (
        np.arange(steps) / (steps - 1) if steps and steps > 1 else np.linspace(0, 1, 40)
    )

    zero = np.zeros_like(ramp)

    if mode == "both":
        x0, y0 = _corner_xy(ci, 0.22, pad=0.055)

        inset = ax.inset_axes([x0, y0, 0.22, 0.22])
        inset.set_box_aspect(1)

        gx, gy = np.meshgrid(ramp, ramp)
        img = _bivar_mix(gx.ravel(), gy.ravel(), cols).reshape(len(ramp), len(ramp), 3)

        inset.set_xlabel(xlabel, fontsize=8, labelpad=1)
        inset.set_ylabel(ylabel, fontsize=8, labelpad=1)
    elif mode == "x":
        x0, y0 = _corner_xy(ci, 0.24, 0.05, pad=0.055)

        inset = ax.inset_axes([x0, y0, 0.24, 0.05])
        img = _bivar_mix(ramp, zero, cols).reshape(1, len(ramp), 3)
        inset.set_xlabel(xlabel, fontsize=8, labelpad=1)
    else:
        x0, y0 = _corner_xy(ci, 0.05, 0.24, pad=0.055)
        inset = ax.inset_axes([x0, y0, 0.05, 0.24])
        img = _bivar_mix(zero, ramp, cols).reshape(len(ramp), 1, 3)
        inset.set_ylabel(ylabel, fontsize=8, labelpad=1)

    inset.imshow(
        img, origin="lower", extent=[0, 1, 0, 1], aspect="auto", interpolation="nearest"
    )

    inset.set_xticks([])
    inset.set_yticks([])

    for spine in inset.spines.values():
        spine.set_color("0.55")
        spine.set_linewidth(0.6)

    return inset


def bivariate_field(
    Y,
    x,
    y,
    ax=None,
    scheme="accessible",
    steps=None,
    x_pass=True,
    y_pass=True,
    highlight=None,
    threed=False,
    fog=True,
    elev=22.0,
    azim=-60.0,
    fog_strength=0.9,
    shade=None,
    x_log=True,
    y_sqrt=True,
    s=6,
    legend=True,
    xlabel=r"$\sigma \rightarrow$",
    ylabel=r"$h \rightarrow$",
):
    Y = np.asarray(Y, float)
    cols = _scheme_cols(scheme)
    n = len(Y)

    sx = _norm01(x, log=x_log) if x_pass else np.zeros(n)
    sy = (
        (np.sqrt(_norm01(y, floor0=True)) if y_sqrt else _norm01(y, floor0=True))
        if y_pass
        else np.zeros(n)
    )

    mode = "both" if x_pass and y_pass else "x" if x_pass else "y" if y_pass else "none"
    sev = {"both": np.maximum(sx, sy), "x": sx, "y": sy, "none": np.zeros(n)}[mode]

    qx = (
        _quantize(sx, steps) if steps and x_pass else sx
    )  # a dropped axis stays 0, never
    qy = (
        _quantize(sy, steps) if steps and y_pass else sy
    )  # quantised (that would read as top bin)

    col = _bivar_mix(qx, qy, cols)

    if shade is not None:  # ambient-occlusion shading (per cell)
        col = col * np.clip(np.asarray(shade, float), 0.0, 1.0)[:, None]

    if highlight is not None and mode != "none":
        thresh, low = (
            np.percentile(sev, highlight),
            np.percentile(sev, max(highlight - 25, 0)),
        )
        fade = np.clip((sev - low) / (thresh - low + 1e-9), 0, 1)[:, None]
        col = col * fade + np.array([0.93, 0.93, 0.92]) * (1 - fade)

    order = np.argsort(sev)  # worst cells drawn last, on top

    if threed:
        if ax is None:
            ax = plt.gcf().add_subplot(projection="3d")

        if fog:  # atmospheric depth: fade the far tail to bg,
            order, weight, near = depth_fog(
                Y, elev, azim, fog_strength
            )  # draw back-to-front, near cells bigger
            col = col * weight + np.array([1.0, 1.0, 1.0]) * (1 - weight)
            size = s * (0.3 + 1.3 * near)  # steep size gradient reinforces depth
            ax.view_init(elev=elev, azim=azim)
        else:
            size = np.full(len(Y), float(s))

        ax.scatter(
            Y[order, 0],
            Y[order, 1],
            Y[order, 2],
            s=size[order],
            c=col[order],
            linewidths=0,
            depthshade=False,
            rasterized=True,
        )

        _bare3d(ax)

        if legend and mode != "none":
            _bivar_key(
                ax, mode, cols, steps, 0, xlabel, ylabel
            )  # top-left: 3D bbox has no data-count

        return ax

    ax = ax or plt.gca()
    ax.scatter(
        Y[order, 0], Y[order, 1], s=s, c=col[order], linewidths=0, rasterized=True
    )
    _bare(ax)

    if legend and mode != "none":
        _bivar_key(ax, mode, cols, steps, _emptiest_corner(Y, ax), xlabel, ylabel)

    return ax


def save(fig, path, dpi=300):
    fig.savefig(path, dpi=dpi, bbox_inches="tight")
