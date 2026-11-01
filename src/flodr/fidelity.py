import math
import warnings

import numpy as np
import torch
from scipy.stats import beta

from .conformal import _sample_r


def _dev(est):
    return next(est.flow_.parameters()).device


def _raw_affine(est):
    # _to_raw is affine, z A + b; padding columns get zero rows, as _to_raw drops them
    d = est.flow_.d
    b = est._to_raw(np.zeros((1, d), np.float32))[0].astype(np.float64)
    A = est._to_raw(np.eye(d, dtype=np.float32)).astype(np.float64) - b

    return A, b


def _raw_map(est, dev):
    A, b = _raw_affine(est)
    A = torch.as_tensor(A, dtype=torch.float32, device=dev)
    b = torch.as_tensor(b, dtype=torch.float32, device=dev)

    return lambda z: z @ A + b


def _rank(below, ties, n, rng):
    return (below + rng.random(np.shape(below)) * (ties + 1.0)) / (n + 1.0)


def band(u, margin=0.1, alpha=0.05):
    # an equivalence test, not a p-value: the KS distance from uniform and its DKW
    # band at level alpha against margin
    n = len(u)
    srt = np.sort(u)
    ks = float(
        max((np.arange(1, n + 1) / n - srt).max(), (srt - np.arange(n) / n).max())
    )
    eps = math.sqrt(math.log(2.0 / alpha) / (2.0 * n))

    if ks + eps <= margin:
        passed, certifies = True, "certified"
    elif ks - eps > margin:
        passed, certifies = False, "refused"
    else:
        passed, certifies = None, "undecided"

    return dict(
        ks=ks, band=eps, mean=float(np.mean(u)), passed=passed, certifies=certifies
    )


def energy_ranks(own, draws, rng, scores=False):
    # the cell's rank among the M + 1 mean distances is uniform when the law is right,
    # near 0 when the draws are too wide, near 1 when too narrow
    M = draws.shape[1]
    rs = torch.cdist(draws, draws).sum(2)
    dpd = torch.cdist(own[:, None], draws)
    rho_own = dpd.sum(2) / M
    rho_draw = (rs[:, None, :] + dpd) / M
    below = (rho_draw < rho_own[..., None]).sum(2).float()
    ties = (rho_draw == rho_own[..., None]).sum(2).float()
    u = _rank(below.cpu().numpy(), ties.cpu().numpy(), M, rng)[:, 0]

    if not scores:
        return u

    # energy score E|X - x| - E|X - X'| / 2, lower is better
    es = rho_own[:, 0] - rs.sum(1) / (2 * M * (M - 1))

    return u, es.cpu().numpy()


@torch.no_grad()
def _draws(flow, y, n_draws, gen, raw):
    n = len(y)
    yb = y.repeat(n_draws, 1)
    z = flow.inverse(torch.cat([yb, _sample_r(flow.cond, yb, gen)], 1))

    return raw(z).view(n_draws, n, -1).transpose(0, 1)


@torch.no_grad()
def nearest_distance(Q, ref, chunk=None):
    chunk = chunk or max(64, (1 << 25) // max(len(ref), 1))

    return torch.cat(
        [
            torch.cdist(Q[i : i + chunk], ref).min(1).values
            for i in range(0, len(Q), chunk)
        ]
    )


@torch.no_grad()
def energy_certificate(
    est, X, n_draws=199, margin=0.1, alpha=0.05, precision_margin=0.02
):
    flow, dev, k = est.flow_, _dev(est), est.flow_.k
    raw = _raw_map(est, dev)
    gen = torch.Generator(device=dev)
    gen.manual_seed(est._seed + 53)
    rng = np.random.default_rng(est._seed + 59)
    clean = np.ascontiguousarray(est._to_coords(X), np.float32)

    # held-out cells carry one draw of the density's input noise and are ranked among
    # draws at their own noised position
    xc = np.ascontiguousarray(est._noised(clean), np.float32)
    chunk = max(8, (1 << 22) // (n_draws * max(flow.d, 64)))
    u, pos = [], []

    for i in range(0, len(xc), chunk):
        x = torch.as_tensor(xc[i : i + chunk], device=dev)
        y = flow.forward_main(x)[:, :k]
        u.append(energy_ranks(raw(x), _draws(flow, y, n_draws, gen, raw), rng))
        pos.append(y.cpu().numpy())

    u, pos = np.concatenate(u), np.concatenate(pos)

    # rank tests barely see a minority of wild draws, hence the precision test; it
    # uses noised cells, as clean ones sit nearer the data than the law of the draws
    ref = torch.as_tensor(np.asarray(est._X, np.float32), device=dev)
    xt, ct = (torch.as_tensor(a, device=dev) for a in (xc, clean))
    # the noise in input units, exactly 0 without it
    shift = raw(xt) - raw(ct)
    d_cell = nearest_distance(torch.as_tensor(X, device=dev) + shift, ref).cpu().numpy()
    d_draw = []

    for i in range(0, len(xc), 64):
        y = flow.forward_main(torch.as_tensor(xc[i : i + 64], device=dev))[:, :k]
        D = _draws(flow, y, n_draws, gen, raw)
        d_draw.append(nearest_distance(D.reshape(-1, D.shape[-1]), ref).cpu().numpy())

    d_draw = np.concatenate(d_draw)
    q99 = float(np.quantile(d_cell, 0.99))
    wild = float((d_draw > q99).mean())
    limit = 0.01 + precision_margin
    prec_ok = wild <= limit

    # one-sided Clopper-Pearson upper bound, reported without a verdict
    n = len(u)
    exc = int((u > 0.95).sum())
    upper = float(beta.ppf(1 - alpha, exc + 1, n - exc)) if exc < n else 1.0

    e = band(u, margin, alpha)

    if e["passed"] is False or not prec_ok:
        passed, certifies = False, "refused"
    elif e["passed"]:
        passed, certifies = True, "certified"
    else:
        passed, certifies = None, "undecided"

    return dict(
        energy=dict(ranks=u, **e),
        precision=dict(
            wild=wild,
            limit=limit,
            q99=q99,
            draw_median=float(np.median(d_draw)),
            cell_median=float(np.median(d_cell)),
            passed=bool(prec_ok),
        ),
        exceedance=dict(share=exc / n, upper=upper, alpha=alpha),
        positions=pos,
        passed=passed,
        certifies=certifies,
        margin=margin,
        n_cells=n,
        n_draws=n_draws,
    )


def _jvp(fn, z, e):
    # jvp works under no_grad; torch loads its jvp decompositions on first use through
    # the deprecated torch.jit.script
    with warnings.catch_warnings():
        warnings.filterwarnings(
            "ignore", message="`torch.jit.script`", category=DeprecationWarning
        )

        return torch.func.jvp(fn, (z,), (e,))[1]


@torch.no_grad()
def faithful_scale(est, Z, sd, h, n_steps, threshold, seed, chunk=None):
    # rho(h) = ||F^-1(F(z) + J e) - (z + e)|| / ||e|| for steps e ~ N(0, h^2 S),
    # median over n_steps; J e by forward-mode AD, so no d x d Jacobian
    flow, dev, k = est.flow_, _dev(est), est.flow_.k
    hs = np.asarray(h, np.float64)
    gen = torch.Generator(device=dev)
    gen.manual_seed(seed)
    chunk = chunk or max(256, (1 << 21) // flow.d)
    rho = np.empty((len(Z), len(hs)))
    pos = np.empty((len(Z), k), np.float32)

    for i in range(0, len(Z), chunk):
        z = torch.as_tensor(np.asarray(Z[i : i + chunk], np.float32), device=dev)
        T = flow.forward_main(z)
        pos[i : i + chunk] = T[:, :k].cpu().numpy()

        for j, hj in enumerate(hs):
            r = []

            for _ in range(n_steps):
                e = float(hj) * sd * torch.randn(z.shape, device=dev, generator=gen)
                t = _jvp(flow.forward_main, z, e)
                x = flow.inverse_main(T + t)
                r.append(((x - z - e).norm(dim=1) / e.norm(dim=1)).cpu().numpy())

            rho[i : i + chunk, j] = np.median(np.stack(r), 0)

    ok = rho <= threshold
    hstar = np.where(ok, hs[None], 0.0).max(1)

    return dict(
        h=hs,
        rho=rho,
        hstar=hstar,
        rho_median=np.median(rho, 0),
        hstar_median=float(np.median(hstar)),
        faithful=ok.mean(0),
        positions=pos,
        threshold=threshold,
        n_steps=n_steps,
        n_cells=len(Z),
    )
