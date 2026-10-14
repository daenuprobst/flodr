import contextlib
import copy
import functools
import math
import os
import warnings

import numpy as np
import torch
from sklearn.mixture import GaussianMixture
from threadpoolctl import threadpool_limits

from .fidelity import _draws, band, energy_ranks, nearest_distance

# the tail fits train on a few hundred rows at a time, where more threads only add
# barrier waits
SMALL_FIT_THREADS = 4


def graph_loop(iters, draw, body, dev, warmup=3):
    done = 0
    if (
        dev.type == "cuda"
        and iters > warmup + 1
        and os.environ.get("FLODR_NO_GRAPHS") != "1"
    ):
        try:
            stream = torch.cuda.Stream()
            stream.wait_stream(torch.cuda.current_stream())

            with torch.cuda.stream(stream):
                for _ in range(warmup):
                    draw()
                    body()
                    done += 1

            torch.cuda.current_stream().wait_stream(stream)
            graph = torch.cuda.CUDAGraph()
            draw()

            with torch.cuda.graph(graph):
                body()

            # the capture records without executing
            graph.replay()
            done += 1
            for _ in range(iters - done):
                draw()
                graph.replay()

            return
        except RuntimeError:
            # capture unsupported
            pass

    for _ in range(iters - done):
        draw()
        body()


@contextlib.contextmanager
def deterministic():
    # the mode is process-wide, so it is put back afterwards
    prev = (
        torch.are_deterministic_algorithms_enabled(),
        torch.is_deterministic_algorithms_warn_only_enabled(),
    )
    torch.use_deterministic_algorithms(True, warn_only=True)

    try:
        yield
    finally:
        torch.use_deterministic_algorithms(prev[0], warn_only=prev[1])


def _few_threads(fn):
    @functools.wraps(fn)
    def wrapped(self, *args, **kw):
        if self.base_logvar.device.type != "cpu":
            return fn(self, *args, **kw)

        prev = torch.get_num_threads()

        # torch last, as leaving threadpool_limits resets the OpenMP pool torch runs on
        try:
            with threadpool_limits(SMALL_FIT_THREADS), deterministic():
                torch.set_num_threads(min(prev, SMALL_FIT_THREADS))

                return fn(self, *args, **kw)
        finally:
            torch.set_num_threads(prev)

    return wrapped


def _mlp(net, x):
    # under autocast the output layer stays fp32: it writes the layout, and bf16 rounds
    # that coarser than the 1-NN spacing
    if not (Coupling.fp32_head and torch.is_autocast_enabled(x.device.type)):
        return net(x)

    h = net[:-1](x)

    with torch.autocast(x.device.type, enabled=False):
        return net[-1](h.float())


class Coupling(torch.nn.Module):
    # set by train_flodr, see _mlp
    fp32_head = False

    def __init__(
        self,
        d,
        mask,
        hid,
        sketch=0,
        act="tanh",
        shallow=False,
        sketch_head=0,
        gate_max=0.0,
        gate_display=0.0,
        k=2,
    ):
        super().__init__()
        self.register_buffer("mask", mask)
        self.gate_max = float(gate_max)

        gate_vec = None

        if gate_display > 0:
            gate_vec = torch.full((d,), float(gate_max))
            gate_vec[:k] = float(gate_display)

        self.register_buffer("gmax_vec", gate_vec)
        self.gate = torch.nn.Parameter(torch.ones(1 if gate_vec is None else d))
        d_out = 2 * d

        # a fixed random projection of the masked dims is enough, since a coupling stays
        # invertible for any function of the conditioning set
        use_sketch = bool(sketch) and d > sketch
        sketch_m = torch.randn(sketch, d) / d**0.5 if use_sketch else None

        # reading only the PCA head starves the null-space per-point fingerprint
        if sketch_m is not None and sketch_head:
            sketch_m[:, sketch_head:] = 0.0

        self.register_buffer("S", sketch_m)
        d_in = sketch if use_sketch else d

        act_cls = {"tanh": torch.nn.Tanh, "relu": torch.nn.ReLU, "silu": torch.nn.SiLU}[
            act
        ]
        mid = [] if shallow else [torch.nn.Linear(hid, hid), act_cls()]
        self.net = torch.nn.Sequential(
            torch.nn.Linear(d_in, hid), act_cls(), *mid, torch.nn.Linear(hid, d_out)
        )

        # zero head: each coupling starts as the identity
        with torch.no_grad():
            self.net[-1].weight.mul_(0.0)
            self.net[-1].bias.mul_(0.0)

    def _st(self, xm):
        free = 1 - self.mask
        inp = xm @ self.S.T if self.S is not None else xm
        net = self.net
        head = _mlp(net, inp) if isinstance(net, torch.nn.Sequential) else net(inp)
        s_raw, t_raw = head.chunk(2, dim=1)

        if self.gate_max <= 0:
            gate = self.gate
        else:
            bound = self.gate_max if self.gmax_vec is None else self.gmax_vec
            gate = bound * torch.tanh(self.gate)

        return gate * torch.tanh(s_raw) * free, t_raw * free

    def forward(self, x):
        x_masked = x * self.mask
        scale, shift = self._st(x_masked)

        return x_masked + (1 - self.mask) * (x * torch.exp(scale) + shift)

    def forward_logdet(self, x):
        x_masked = x * self.mask
        scale, shift = self._st(x_masked)

        return x_masked + (1 - self.mask) * (x * torch.exp(scale) + shift), scale.sum(1)

    def inverse(self, y):
        y_masked = y * self.mask
        scale, shift = self._st(y_masked)

        return y_masked + (1 - self.mask) * ((y - shift) * torch.exp(-scale))


class FourierShift(torch.nn.Module):
    # shift only, so the coupling's log-det is untouched
    def __init__(
        self, base, d_in, hid, d_out, n_freq=256, lo=2.0, hi=6.0, eps=0.25, gen=None
    ):
        super().__init__()
        self.base = base
        shift_mask = torch.zeros(d_out)
        shift_mask[d_out // 2 :] = 1.0
        self.register_buffer("shift_mask", shift_mask)

        mag = torch.exp(
            torch.empty(n_freq).uniform_(math.log(lo), math.log(hi), generator=gen)
        )
        dirs = torch.randn(d_in, n_freq, generator=gen)
        self.register_buffer("W", dirs / dirs.norm(dim=0, keepdim=True) * mag)
        self.register_buffer("b", torch.rand(n_freq, generator=gen) * 2 * math.pi)

        # off until train_flodr switches it on
        self.eps, self.on = float(eps), False
        self.high = torch.nn.Sequential(
            torch.nn.Linear(2 * n_freq, hid),
            torch.nn.Tanh(),
            torch.nn.Linear(hid, d_out),
        )

        with torch.no_grad():
            self.high[-1].weight.mul_(0.0)
            self.high[-1].bias.mul_(0.0)

    def forward(self, x):
        out = _mlp(self.base, x)

        if not self.on:
            return out

        # the phase reaches |f| ~ 20, where a bf16 step is 0.125 rad
        if Coupling.fp32_head:
            with torch.autocast(x.device.type, enabled=False):
                f = x.float() @ self.W + self.b
        else:
            f = x @ self.W + self.b

        shift = _mlp(self.high, torch.cat([f.cos(), f.sin()], -1)) * self.shift_mask

        return out + self.eps * shift


# (n_tail, cond_hid, iters, noise_floor, weight_decay[, n_scale]) for
# fit_cond_tail_cv; under input noise the rows with a floor drop out
COVERAGE_GRID = (
    (0, 32, 1500, 0.0, 0.0),
    (0, 128, 300, 0.0, 0.0),
    (2, 128, 1500, 0.0, 0.0),
    (2, 64, 800, 0.05, 1e-3),
    (2, 128, 800, 0.1, 1e-2),
    (0, 32, 1500, 0.05, 0.0),
    (0, 128, 1500, 0.0, 0.0, 20),
    (0, 64, 800, 0.0, 1e-3, 20),
)
# (n_tail, cond_hid, iters, n_scale) for fit_cond_tail_energy; the depth is searched
# too, since deep tails overfit small samples
ENERGY_GRID = (
    (0, 128, 1500, 20),
    (2, 128, 1500, 20),
    (4, 256, 3000, 20),
    (8, 256, 1500, 20),
    (8, 256, 3000, 20),
    (8, 256, 6000, 20),
    (8, 128, 3000, 20),
)


@torch.no_grad()
def _energy_val(flow, zc, ref, to_in, sd, n_draws, seed, margin, alpha, chunk=25):
    k, dev = flow.k, zc.device
    gen = torch.Generator(device=dev).manual_seed(seed + 53)
    rng = np.random.default_rng(seed + 59)
    zn = (
        zc if sd is None else zc + sd * torch.randn(zc.shape, device=dev, generator=gen)
    )
    u, es, d_draw = [], [], []

    for i in range(0, len(zn), chunk):
        z = zn[i : i + chunk]
        D = _draws(flow, flow.forward_main(z)[:, :k], n_draws, gen, to_in)
        r, e = energy_ranks(to_in(z), D, rng, scores=True)
        u.append(r)
        es.append(e)
        d_draw.append(nearest_distance(D.reshape(-1, D.shape[-1]), ref).cpu().numpy())

    d_cell = nearest_distance(to_in(zn), ref).cpu().numpy()
    b = band(np.concatenate(u), margin, alpha)

    return dict(
        ks=b["ks"],
        refused=b["passed"] is False,
        wild=float((np.concatenate(d_draw) > np.quantile(d_cell, 0.99)).mean()),
        score=float(np.concatenate(es).mean()),
    )


def _fit_y_gmm(Y, ks=(16, 32, 64, 128, 256), seed=0):
    Y = np.asarray(Y, dtype=np.float64)
    rng = np.random.default_rng(seed)
    perm = rng.permutation(len(Y))
    tr_idx, va_idx = perm[: len(Y) // 2], perm[len(Y) // 2 :]
    best, best_ll = None, -np.inf

    for k in ks:
        if k >= len(tr_idx):
            continue

        gmm = GaussianMixture(
            k, covariance_type="diag", random_state=seed, reg_covar=1e-6
        ).fit(Y[tr_idx])

        loglik = float(gmm.score(Y[va_idx]))

        if loglik > best_ll:
            best, best_ll = k, loglik

    # fewer points than the smallest candidate: one component per four points
    if best is None:
        best = max(1, len(Y) // 4)

    gmm = GaussianMixture(
        best, covariance_type="diag", random_state=seed, reg_covar=1e-6
    ).fit(Y)

    return (
        torch.tensor(np.log(gmm.weights_), dtype=torch.float32),
        torch.tensor(gmm.means_, dtype=torch.float32),
        torch.tensor(np.log(gmm.covariances_), dtype=torch.float32),
    )


# the passes after training run over every cell at once, so outside autograd they go in
# chunks to bound memory
_PASS_CHUNK = 131072


def _chunked(fn):
    @functools.wraps(fn)
    def run(self, x):
        if torch.is_grad_enabled() or x.shape[0] <= _PASS_CHUNK:
            return fn(self, x)

        return torch.cat(
            [
                fn(self, x[i : i + _PASS_CHUNK])
                for i in range(0, x.shape[0], _PASS_CHUNK)
            ]
        )

    return run


class CondBase(torch.nn.Module):
    def __init__(self, d, hid=128, k=2, n_scale=0, scale_lo=0.2, scale_hi=3.0):
        super().__init__()
        self.k = k
        self.m = d - k
        self.n_scale = n_scale
        self.y_mu = torch.nn.Parameter(torch.zeros(k))
        self.y_logvar = torch.nn.Parameter(torch.zeros(k))

        self.net = torch.nn.Sequential(
            torch.nn.Linear(k, hid),
            torch.nn.Tanh(),
            torch.nn.Linear(hid, hid),
            torch.nn.Tanh(),
            torch.nn.Linear(hid, 2 * (d - k) + n_scale),
        )

        # a Gaussian scale mixture on a fixed log grid, weights depending on y, since
        # affine couplings cannot change the tail class of a Gaussian base
        if n_scale:
            self.register_buffer(
                "log_s", torch.linspace(math.log(scale_lo), math.log(scale_hi), n_scale)
            )

        # start at the unconditional standard normal
        with torch.no_grad():
            self.net[-1].weight.mul_(0.0)
            self.net[-1].bias.mul_(0.0)

    def r_params(self, y):
        out = self.net(y)
        m = self.m

        return out[:, :m], out[:, m : 2 * m].clamp(-8.0, 8.0)

    def scale_logw(self, y):
        if not self.n_scale:
            return None

        return torch.log_softmax(self.net(y)[:, 2 * self.m :], 1)

    def sample_r(self, y, gen):
        mu, logvar = self.r_params(y)
        eps = torch.randn(mu.shape, device=y.device, generator=gen)
        logw = self.scale_logw(y)

        if logw is None:
            return mu + torch.exp(0.5 * logvar) * eps

        j = torch.multinomial(logw.exp(), 1, generator=gen)[:, 0]

        return mu + torch.exp(0.5 * logvar + self.log_s[j][:, None]) * eps

    def fit_y_density(self, Y, ks=(16, 32, 64, 128, 256), seed=0):
        dev = self.y_mu.device

        for name, val in zip(
            ("ylogw", "ymu", "ylogvar"), _fit_y_gmm(Y, ks=ks, seed=seed)
        ):
            self.register_buffer(name, val.to(dev))

        return self

    # its (n, K, k) temporaries are too large for every cell at once
    @_chunked
    def y_log_prob(self, y):
        l2pi = math.log(2 * math.pi)

        if getattr(self, "ylogw", None) is not None:
            diff = y[:, None, :] - self.ymu[None, :, :]
            log_p = -0.5 * (
                diff**2 * torch.exp(-self.ylogvar[None]) + self.ylogvar[None] + l2pi
            ).sum(2)

            return torch.logsumexp(self.ylogw[None] + log_p, dim=1)

        return -0.5 * (
            (y - self.y_mu) ** 2 * torch.exp(-self.y_logvar) + self.y_logvar + l2pi
        ).sum(1)

    def log_prob(self, y, r):
        l2pi = math.log(2 * math.pi)
        mu, logvar = self.r_params(y)
        logw = self.scale_logw(y)

        if logw is None:
            log_p_r = -0.5 * ((r - mu) ** 2 * torch.exp(-logvar) + logvar + l2pi).sum(1)
        else:
            q = ((r - mu) ** 2 * torch.exp(-logvar)).sum(1)
            per_scale = (
                -0.5 * q[:, None] * torch.exp(-2 * self.log_s)[None]
                - self.m * self.log_s[None]
            )
            log_p_r = torch.logsumexp(logw + per_scale, 1) - 0.5 * (logvar + l2pi).sum(
                1
            )

        return self.y_log_prob(y) + log_p_r


class Flow(torch.nn.Module):
    def __init__(
        self,
        d,
        n_layers=4,
        hid=256,
        sketch=0,
        act="tanh",
        shallow=False,
        sketch_head=0,
        gate_max=0.0,
        gate_display=0.0,
        k=2,
    ):
        super().__init__()
        layers = []
        res = torch.arange(k, d)

        # alternating masks, so every second layer writes the display from the residual
        # and the others write half the residual; a random draw can miss a display axis
        for i in range(n_layers):
            mask = torch.zeros(d)

            if (i + 1) % 2 == 0:
                mask[k:] = 1.0
            else:
                mask[:k] = 1.0
                mask[res[(i // 2) % 2 :: 2]] = 1.0

            layers.append(
                Coupling(
                    d,
                    mask,
                    hid,
                    sketch=sketch,
                    act=act,
                    shallow=shallow,
                    sketch_head=sketch_head,
                    gate_max=gate_max,
                    gate_display=gate_display,
                    k=k,
                )
            )

        self.layers = torch.nn.ModuleList(layers)

        self.tail = torch.nn.ModuleList()

        self._cond = None

        # set by train_flodr for the lazy density fit
        self._density_args = None

        # per-dim sd of the input noise the density is fitted under, set by FloDR.fit
        self._density_noise = None
        self.base_logvar = torch.nn.Parameter(torch.zeros(d))
        self.d, self.hid, self.k = d, hid, k

    @_chunked
    def forward(self, x):
        for layer in self.layers:
            x = layer(x)

        # the tail leaves the first k dims, the embedding, unchanged
        for layer in self.tail:
            x = layer(x)

        return x

    def forward_with_logdet(self, x):
        logdet = x.new_zeros(x.shape[0])

        for layer in self.layers:
            x, ld_layer = layer.forward_logdet(x)
            logdet = logdet + ld_layer

        for layer in self.tail:
            x, ld_layer = layer.forward_logdet(x)
            logdet = logdet + ld_layer

        return x, logdet

    @_chunked
    def inverse(self, z):
        for layer in reversed(self.tail):
            z = layer.inverse(z)

        for layer in reversed(self.layers):
            z = layer.inverse(z)

        return z

    @_chunked
    def forward_main(self, x):
        for layer in self.layers:
            x = layer(x)

        return x

    @_chunked
    def inverse_main(self, z):
        for layer in reversed(self.layers):
            z = layer.inverse(z)

        return z

    @torch.no_grad()
    def _main_latents(self, x):
        zs, lds = [], []

        for i in range(0, x.shape[0], _PASS_CHUNK):
            z = x[i : i + _PASS_CHUNK]
            logdet = z.new_zeros(z.shape[0])

            for layer in self.layers:
                z, ld_layer = layer.forward_logdet(z)
                logdet = logdet + ld_layer

            zs.append(z)
            lds.append(logdet)

        return torch.cat(zs), torch.cat(lds)

    @_few_threads
    def fit_cond_tail(
        self,
        x,
        n_tail=2,
        iters=1500,
        batch=512,
        cond_hid=128,
        lr=1e-3,
        seed=0,
        noise_floor=0.0,
        weight_decay=0.0,
        n_scale=0,
        _latents=None,
        _ydensity=None,
    ):
        if len(self.tail):
            raise RuntimeError("fit_cond_tail was already called on this flow")

        dev = self.base_logvar.device

        if _latents is not None:
            z_main, ld_main = _latents
        else:
            with torch.no_grad():
                # x may already be a device tensor, and np.asarray raises on those
                x_tens = x if torch.is_tensor(x) else torch.as_tensor(np.asarray(x))
                z_main, ld_main = self.forward_with_logdet(
                    x_tens.to(device=dev, dtype=torch.float32)
                )

        torch.manual_seed(seed + 1000)
        d, k = self.d, self.k
        tail = torch.nn.ModuleList()

        for _ in range(n_tail):
            mask = torch.zeros(d)

            # y always in the conditioning (identity) set
            mask[:k] = 1.0
            mask[k + torch.randperm(d - k)[: (d - k) // 2]] = 1.0
            tail.append(Coupling(d, mask, self.hid))

        tail.to(dev)
        cond = CondBase(d, cond_hid, k=k, n_scale=n_scale).to(dev)
        params = list(tail.parameters()) + list(cond.parameters())

        opt = torch.optim.Adam(
            params, lr=lr, weight_decay=weight_decay, capturable=dev.type == "cuda"
        )

        gen = torch.Generator(device=dev)
        gen.manual_seed(seed + 1000)
        n = z_main.shape[0]
        n_batch = min(batch, n)
        rows = torch.zeros(n_batch, dtype=torch.long, device=dev)
        noise = torch.zeros(n_batch, d - k, device=dev) if noise_floor else None

        def draw():
            rows.copy_(torch.randint(0, n, (n_batch,), device=dev, generator=gen))

            if noise_floor:
                noise.copy_(torch.randn((n_batch, d - k), device=dev, generator=gen))

        def body():
            z, logdet = z_main[rows], ld_main[rows]

            if noise_floor:
                z = torch.cat([z[:, :k], z[:, k:] + noise_floor * noise], 1)

            for layer in tail:
                z, ld_layer = layer.forward_logdet(z)
                logdet = logdet + ld_layer

            loss = -(cond.log_prob(z[:, :k], z[:, k:]) + logdet).mean() / d
            opt.zero_grad(set_to_none=False)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(params, 5.0)
            opt.step()

        # the lazy fit may run inside a caller's no_grad
        stack = torch.enable_grad()
        stack.__enter__()
        graph_loop(iters, draw, body, dev)
        stack.__exit__(None, None, None)
        self.tail = tail
        self._cond = cond
        self._density_args = None

        # p(y) as a GMM, as a single Gaussian does not know where the data is; the
        # main flow is frozen by now
        if _ydensity is not None:
            for name, val in zip(("ylogw", "ymu", "ylogvar"), _ydensity):
                cond.register_buffer(name, val.to(dev))
        else:
            with torch.no_grad():
                y_frozen = (
                    self.forward_main(
                        torch.as_tensor(x, dtype=torch.float32, device=cond.y_mu.device)
                    )[:, :k]
                    .cpu()
                    .numpy()
                )

            cond.fit_y_density(y_frozen)

        return self

    @_few_threads
    def fit_cond_tail_cv(
        self,
        x,
        seed=0,
        grid=None,
        select="coverage",
        n_draws=99,
        max_score_cells=2000,
        noise_sd=None,
        copies=4,
        raw=None,
    ):
        if select == "energy":
            return self.fit_cond_tail_energy(
                x,
                seed=seed,
                grid=grid or ENERGY_GRID,
                noise_sd=noise_sd,
                copies=copies,
                raw=raw,
            )

        grid = grid or COVERAGE_GRID
        n = len(x)
        rng = np.random.default_rng(seed)
        perm = rng.permutation(n)
        tr_idx, va_idx = perm[: n // 2], perm[n // 2 :]
        dev = self.base_logvar.device

        x_all = torch.as_tensor(
            np.asarray(x.cpu() if torch.is_tensor(x) else x),
            dtype=torch.float32,
            device=dev,
        )

        # with noise_sd the law is the cells convolved with input noise: fits see noisy
        # copies, the split stays by cell, and the noise replaces the grid's floor
        if noise_sd is not None:
            gen_n = torch.Generator(device=dev).manual_seed(seed + 5)
            sd = torch.as_tensor(noise_sd, dtype=torch.float32, device=dev)
            copies = int(np.clip(2_000_000 // max(n, 1), 1, copies))

            def noisy(rows, m):
                z = x_all[rows].repeat(m, 1)

                return z + sd * torch.randn(z.shape, device=dev, generator=gen_n)

            x_fit, x_val = noisy(tr_idx, copies), noisy(va_idx, 1)
            x_final = noisy(np.arange(n), copies)
            grid = tuple(c for c in grid if len(c) < 4 or not c[3])
        else:
            x_fit, x_val, x_final = x_all[tr_idx], x_all[va_idx], x

        # the candidates share the frozen main flow, so the latents and the p(y) GMM are
        # computed once
        z_tr, ld_tr = self._main_latents(x_fit)
        z_va, ld_va = self._main_latents(x_val)

        y_gmm = _fit_y_gmm(z_tr[:, : self.k].cpu().numpy(), seed=seed)
        best, best_v = None, np.inf

        for cand in grid:
            cand = tuple(cand)
            n_tail, hid, iters, floor, w_decay, n_scale = (
                cand + (0.0, 0.0, 0)[len(cand) - 3 :]
            )[:6]
            n_scale = int(n_scale)
            cand_flow = copy.deepcopy(self)
            cand_flow._density_args = None
            cand_flow.tail, cand_flow._cond = torch.nn.ModuleList(), None

            cand_flow.fit_cond_tail(
                x_fit,
                n_tail=n_tail,
                iters=iters,
                cond_hid=hid,
                seed=seed,
                noise_floor=floor,
                weight_decay=w_decay,
                n_scale=n_scale,
                _latents=(z_tr, ld_tr),
                _ydensity=y_gmm,
            )

            # tail-only forward, identical to cand_flow.log_prob(x[va_idx])
            with torch.no_grad():
                z, logdet = z_va.clone(), ld_va.clone()
                for layer in cand_flow.tail:
                    z, ld_layer = layer.forward_logdet(z)
                    logdet = logdet + ld_layer

                nll = -float((cand_flow.base_log_prob(z) + logdet).mean()) / cand_flow.d

            # held-out likelihood can prefer a tail whose fibre regions miss new cells,
            # so coverage ranks the held-out cells among exact fibre draws instead
            if select == "coverage":
                u = cand_flow.fibre_ranks(
                    x_val[:max_score_cells], n_draws=n_draws, seed=seed
                )
                score = float(
                    np.abs(np.sort(u) - (np.arange(len(u)) + 0.5) / len(u)).max()
                )
            else:
                score = nll

            if score < best_v:
                best, best_v = (n_tail, hid, iters, floor, w_decay, n_scale), score

        n_tail, hid, iters, floor, w_decay, n_scale = best
        self._density_args = None
        self.tail, self._cond = torch.nn.ModuleList(), None
        self.cond_choice_ = dict(
            n_tail=n_tail,
            cond_hid=hid,
            iters=iters,
            noise_floor=floor,
            weight_decay=w_decay,
            n_scale=n_scale,
            select=select,
            val_score=best_v,
            noise=noise_sd is not None,
        )

        # the main pass in chunks, which fit_cond_tail's own pass is not
        return self.fit_cond_tail(
            x_final,
            n_tail=n_tail,
            iters=iters,
            cond_hid=hid,
            seed=seed,
            noise_floor=floor,
            weight_decay=w_decay,
            n_scale=n_scale,
            _latents=self._main_latents(x_all if noise_sd is None else x_final),
        )

    @_few_threads
    def fit_cond_tail_energy(
        self,
        x,
        seed=0,
        grid=ENERGY_GRID,
        noise_sd=None,
        copies=4,
        raw=None,
        n_val=1000,
        n_draws=199,
        margin=0.1,
        alpha=0.05,
        precision_margin=0.02,
    ):
        # the energy certificate, as l-rank coverage passes draws far too wide;
        # raw = (A, b) maps these coords to the input units it measures in, z A + b
        dev = self.base_logvar.device
        n = len(x)
        x_all = torch.as_tensor(
            np.asarray(x.cpu() if torch.is_tensor(x) else x),
            dtype=torch.float32,
            device=dev,
        )
        perm = np.random.default_rng(seed + 31).permutation(n)
        tr_idx, va_idx = np.sort(perm[: n // 2]), np.sort(perm[n // 2 :])
        va_idx = va_idx[
            np.random.default_rng(seed + 37).choice(
                len(va_idx), min(n_val, len(va_idx)), replace=False
            )
        ]
        sd = None
        copies = int(np.clip(2_000_000 // max(n, 1), 1, copies))

        if noise_sd is not None:
            sd = torch.as_tensor(noise_sd, dtype=torch.float32, device=dev)

        # a fresh generator per call, so the final noise does not depend on the choice
        def noisy(rows):
            z = x_all[rows]

            if sd is None:
                return z

            z = z.repeat(copies, 1)
            gen = torch.Generator(device=dev).manual_seed(seed + 5)

            return z + sd * torch.randn(z.shape, device=dev, generator=gen)

        if raw is None:
            to_in = lambda z: z  # noqa: E731
        else:
            A, b = (torch.as_tensor(v, dtype=torch.float32, device=dev) for v in raw)
            to_in = lambda z: z @ A + b  # noqa: E731

        x_fit = noisy(tr_idx)
        ref = to_in(x_all[tr_idx])
        z_tr, ld_tr = self._main_latents(x_fit)
        y_gmm = _fit_y_gmm(z_tr[:, : self.k].cpu().numpy(), seed=seed)
        rows = []

        for n_tail, hid, iters, n_scale in grid:
            cand_flow = copy.deepcopy(self)
            cand_flow._density_args = None
            cand_flow.tail, cand_flow._cond = torch.nn.ModuleList(), None
            cand_flow.fit_cond_tail(
                x_fit,
                n_tail=n_tail,
                iters=iters,
                cond_hid=hid,
                seed=seed,
                n_scale=n_scale,
                _latents=(z_tr, ld_tr),
                _ydensity=y_gmm,
            )
            v = _energy_val(
                cand_flow, x_all[va_idx], ref, to_in, sd, n_draws, seed, margin, alpha
            )
            v.update(cfg=(n_tail, hid, iters, n_scale))
            v["eligible"] = not v["refused"] and v["wild"] <= 0.01 + precision_margin
            rows.append(v)

        # the map is fitted on these cells too, so the checks are optimistic
        best = min([v for v in rows if v["eligible"]] or rows, key=lambda v: v["score"])

        if not best["eligible"]:
            warnings.warn(
                "no density tail passed the energy certificate on held-out cells; kept the "
                "one with the best energy score (cond_choice_['passed'] is False)",
                RuntimeWarning,
                stacklevel=3,
            )

        n_tail, hid, iters, n_scale = best["cfg"]
        self._density_args = None
        self.tail, self._cond = torch.nn.ModuleList(), None
        del z_tr, ld_tr, x_fit
        x_final = noisy(np.arange(n))
        self.fit_cond_tail(
            x_final,
            n_tail=n_tail,
            iters=iters,
            cond_hid=hid,
            seed=seed,
            n_scale=n_scale,
            _latents=self._main_latents(x_final),
        )
        self.cond_choice_ = dict(
            n_tail=n_tail,
            cond_hid=hid,
            iters=iters,
            n_scale=n_scale,
            select="energy",
            passed=best["eligible"],
            val_score=best["score"],
            val_ks=best["ks"],
            val_wild=best["wild"],
            candidates=rows,
            noise=noise_sd is not None,
        )

        return self

    @torch.no_grad()
    def fibre_ranks(self, x, n_draws=199, seed=0, chunk=512):
        # l ranked among exact draws from the cell's own fibre; randomised ties make
        # the rank uniform on (0, 1) when the fitted law is right, in finite samples
        cond = self.cond
        dev = self.base_logvar.device
        x = torch.as_tensor(
            np.asarray(x.cpu() if torch.is_tensor(x) else x),
            dtype=torch.float32,
            device=dev,
        )
        gen = torch.Generator(device=dev)
        gen.manual_seed(seed + 31)
        rng = np.random.default_rng(seed + 37)
        k = self.k

        def ell(z, logdet):
            y, r = z[:, :k], z[:, k:]
            return cond.log_prob(y, r) - cond.y_log_prob(y) + logdet

        out = []

        for start in range(0, x.shape[0], chunk):
            z, logdet = self.forward_with_logdet(x[start : start + chunk])
            y = z[:, :k]
            l_cell = ell(z, logdet)
            below = torch.zeros_like(l_cell)
            ties = torch.zeros_like(l_cell)

            for _ in range(n_draws):
                xm = self.inverse(torch.cat([y, cond.sample_r(y, gen)], 1))
                l_m = ell(*self.forward_with_logdet(xm))
                below += (l_m < l_cell).float()
                ties += (l_m == l_cell).float()

            u = rng.random(len(l_cell))
            b, t = below.cpu().numpy(), ties.cpu().numpy()
            out.append((b + u * (t + 1.0)) / (n_draws + 1.0))

        return np.concatenate(out)

    @property
    def cond(self):
        if self._cond is None and self._density_args is not None:
            # a fixed-capacity tail memorises small samples, so the deferred fit
            # selects its tail on held-out cells
            x, _, _, seed = self._density_args
            self._density_args = None
            self.fit_cond_tail_cv(
                x,
                seed=seed,
                noise_sd=self._density_noise,
                **getattr(self, "_tail_kw", {}),
            )

        return self._cond

    def base_log_prob(self, z):
        l2pi = math.log(2 * math.pi)
        if self.cond is not None:
            return self.cond.log_prob(z[:, : self.k], z[:, self.k :])

        return -0.5 * (
            z**2 * torch.exp(-self.base_logvar) + self.base_logvar + l2pi
        ).sum(1)

    def log_prob(self, x):
        # the deferred tail has to exist before the forward pass
        _ = self.cond
        z, logdet = self.forward_with_logdet(x)

        return self.base_log_prob(z) + logdet
