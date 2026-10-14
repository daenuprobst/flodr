import math

import numpy as np
import torch
from scipy.spatial import cKDTree


def _dev(est):
    return next(est.flow_.parameters()).device


def _l_and_y(flow, z):
    # l = log p(x) - log p(y) in whitened coordinates; the whitening is linear, so it
    # shifts every l by one constant and moves no threshold
    out, logdet = flow.forward_with_logdet(z)
    y = out[:, : flow.k]

    return flow.base_log_prob(out) + logdet - flow.cond.y_log_prob(y), y


def fibre_score(est, X, chunk=8192, noised=False):
    # noised adds one draw of the input noise the density is fitted under
    est._need_density()
    z_all = np.ascontiguousarray(est._to_coords(np.asarray(X, np.float32)), np.float32)

    if noised:
        z_all = np.ascontiguousarray(est._noised(z_all), np.float32)
    flow, dev = est.flow_, _dev(est)
    l_out = np.empty(len(z_all))
    y_out = np.empty((len(z_all), flow.k), np.float32)

    with torch.no_grad():
        for s in range(0, len(z_all), chunk):
            l_b, y_b = _l_and_y(flow, torch.from_numpy(z_all[s : s + chunk]).to(dev))
            l_out[s : s + chunk] = l_b.cpu().numpy()
            y_out[s : s + chunk] = y_b.cpu().numpy()

    return l_out, y_out


def _sample_r(cond, y, gen):
    return cond.sample_r(y, gen)


def _fibre_chunks(est, Y, n_draws, seed, width):
    est._need_density()
    flow, dev = est.flow_, _dev(est)
    Y = np.ascontiguousarray(Y, np.float32)
    gen = torch.Generator(device=dev)
    gen.manual_seed(seed)

    step = max(1, (1 << 24) // (n_draws * max(width, flow.d)))

    with torch.no_grad():
        for s in range(0, len(Y), step):
            c = min(step, len(Y) - s)
            # draw-major rows, so a reshape to (n_draws, c, .) lines them up
            yb = torch.from_numpy(Y[s : s + c]).to(dev).repeat(n_draws, 1)
            z = flow.inverse(torch.cat([yb, _sample_r(flow.cond, yb, gen)], 1))
            l_d, y_back = _l_and_y(flow, z)

            # cells are scored with zeros in the padding columns, so a draw is scored as
            # the input row it becomes; on a padded fit it re-embeds near y, not at it
            if est._n_pad:
                l_d, _ = _l_and_y(
                    flow, torch.cat([z[:, : est._d_z], 0 * z[:, est._d_z :]], 1)
                )

            yield (
                s,
                z.cpu().numpy().reshape(n_draws, c, -1),
                l_d.cpu().numpy().reshape(n_draws, c),
                y_back.cpu().numpy().reshape(n_draws, c, -1),
            )


def sample_fibre(est, Y, n_draws=64, seed=0):
    # exact draws from p(x|y): r ~ p(r|y), x = f^-1(y, r)
    Y = np.asarray(Y, np.float32)
    xs, ls, ys = [], [], []

    for _, z, l_d, y_back in _fibre_chunks(est, Y, n_draws, seed, est.n_features_in_):
        m, c, d = z.shape
        xs.append(est._to_raw(z.reshape(m * c, d)).reshape(m, c, -1))
        ls.append(l_d)
        ys.append(y_back)

    return np.concatenate(xs, 1), np.concatenate(ls, 1), np.concatenate(ys, 1)


def split_threshold(scores, alpha):
    # k = ceil((n+1) alpha) - 1: a new exchangeable score falls below the k-th smallest
    # with probability k/(n+1) <= alpha; the 1e-9 stops float rounding adding a rank
    s = np.sort(np.asarray(scores, np.float64))
    k = math.ceil((len(s) + 1) * alpha - 1e-9) - 1

    return s[k - 1] if k >= 1 else -np.inf


class MapRegions:
    def __init__(self, lo, hi, n_bins=14, min_cells=20):
        self.lo, self.hi = np.asarray(lo, np.float64), np.asarray(hi, np.float64)
        self.n_bins, self.min_cells = n_bins, min_cells

    def _bins(self, Y):
        # bins fixed by the training map, not by the calibration cells
        Y = np.asarray(Y, np.float64)
        ix = ((Y - self.lo) / (self.hi - self.lo + 1e-9) * self.n_bins).astype(int)
        ix = np.clip(ix, 0, self.n_bins - 1)

        return np.ravel_multi_index(ix.T, (self.n_bins,) * Y.shape[1])

    def fit(self, Y_cal):
        Y_cal = np.asarray(Y_cal, np.float64)
        occ, inv = np.unique(self._bins(Y_cal), return_inverse=True)
        inv = inv.ravel()
        cnt = np.bincount(inv, minlength=len(occ)).astype(np.int64)
        sums = np.zeros((len(occ), Y_cal.shape[1]))
        np.add.at(sums, inv, Y_cal)
        owner = np.arange(len(occ))
        alive = list(range(len(occ)))

        while len(alive) > 1:
            j = int(np.argmin(cnt[alive]))

            if cnt[alive[j]] >= self.min_cells:
                break

            cent = sums[alive] / cnt[alive][:, None]
            dist = ((cent - cent[j]) ** 2).sum(1)
            dist[j] = np.inf
            src, dst = alive[j], alive[int(np.argmin(dist))]
            sums[dst] += sums[src]
            cnt[dst] += cnt[src]
            owner[owner == src] = dst
            alive.pop(j)

        relabel = {r: i for i, r in enumerate(alive)}
        self.bin_region_ = np.full(self.n_bins ** Y_cal.shape[1], -1, np.int64)
        self.bin_region_[occ] = [relabel[o] for o in owner]
        self.n_regions_ = len(alive)
        self.labels_ = self.bin_region_[occ][inv]
        self._tree = cKDTree(Y_cal)

        return self

    def assign(self, Y):
        Y = np.asarray(Y, np.float64)
        reg = self.bin_region_[self._bins(Y)]
        miss = reg < 0

        if miss.any():
            reg[miss] = self.labels_[self._tree.query(Y[miss])[1]]

        return reg


def _kept_quantiles(v, keep, q):
    # numpy's linear percentile over each query's kept draws, nan where none is kept
    n_keep = keep.sum(0)
    v = np.sort(np.where(keep[..., None], v, np.inf), 0)
    out = []

    for p in q:
        pos = np.maximum(p / 100.0 * (n_keep - 1), 0.0)
        i0 = np.floor(pos).astype(np.int64)
        i1 = np.minimum(i0 + 1, np.maximum(n_keep - 1, 0))
        a = np.take_along_axis(v, i0[None, :, None], 0)[0]
        b = np.take_along_axis(v, i1[None, :, None], 0)[0]
        with np.errstate(invalid="ignore"):
            val = a + (b - a) * (pos - i0)[:, None]

        out.append(np.where(n_keep[:, None] > 0, val, np.nan))

    return out


def _apply_readout(x, readout):
    if readout is None:
        return x

    if callable(readout):
        return np.asarray(readout(x))

    return x @ np.asarray(readout, x.dtype)


class FibreConformal:
    def __init__(self, est, alpha=0.1, n_bins=14, min_cells=20):
        self.est, self.alpha = est, alpha
        self.n_bins, self.min_cells = n_bins, min_cells

    def fit(self, calib_X):
        # calibration cells must be held out from the layout and the tail, or q_hat is
        # too high; one draw of input noise scores them as draws of the fitted law
        l_cal, y_cal = fibre_score(self.est, calib_X, noised=True)
        # a slim model keeps the layout's extent but not the layout
        lo, hi = self.est._layout_bounds()
        self.regions_ = MapRegions(lo, hi, self.n_bins, self.min_cells).fit(y_cal)
        reg = self.regions_.labels_
        self.n_cal_ = np.bincount(reg, minlength=self.regions_.n_regions_)
        self.q_hat_ = np.array(
            [
                split_threshold(l_cal[reg == g], self.alpha)
                for g in range(self.regions_.n_regions_)
            ]
        )
        self.l_cal_, self.y_cal_ = l_cal, y_cal

        return self

    def threshold(self, Y):
        return self.q_hat_[self.regions_.assign(Y)]

    def contains(self, X):
        # the coverage statement is about cells drawn like the calibration cells
        l_x, y_x = fibre_score(self.est, X, noised=True)

        return l_x >= self.threshold(y_x)

    def bands(self, Y, n_draws=256, q=(5, 95), seed=0, readout=None):
        Y = np.asarray(Y, np.float32)
        q_hat = self.threshold(Y)
        width = getattr(readout, "shape", (0, self.est.n_features_in_))[1]
        lo, hi, kept = [], [], np.empty(len(Y))

        for s, z, l_d, _ in _fibre_chunks(self.est, Y, n_draws, seed, width):
            m, c, d = z.shape
            x = self.est._to_raw(z.reshape(m * c, d))
            v = _apply_readout(x, readout).reshape(m, c, -1)
            keep = l_d >= q_hat[s : s + c]
            kept[s : s + c] = keep.mean(0)
            a, b = _kept_quantiles(v, keep, q)
            lo.append(a)
            hi.append(b)

        return dict(
            lo=np.concatenate(lo),
            hi=np.concatenate(hi),
            kept=kept,
            q_hat=q_hat,
            region=self.regions_.assign(Y),
        )


def conformal_bands(
    est,
    calib_X,
    Y_query,
    alpha=0.1,
    n_bins=14,
    min_cells=20,
    n_draws=256,
    q=(5, 95),
    seed=0,
    readout=None,
):
    cp = FibreConformal(est, alpha, n_bins, min_cells).fit(calib_X)

    return cp.bands(Y_query, n_draws=n_draws, q=q, seed=seed, readout=readout)
