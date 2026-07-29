import copy
import math
import os

import numpy as np
import torch
from sklearn.cluster import KMeans
from sklearn.mixture import GaussianMixture


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

            graph.replay()  # capture records without executing

            done += 1
            for _ in range(iters - done):
                draw()
                graph.replay()

            return
        except RuntimeError:
            pass  # capture unsupported: finish eagerly

    for _ in range(iters - done):
        draw()
        body()


class Coupling(torch.nn.Module):
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
    ):
        super().__init__()
        self.register_buffer("mask", mask)
        self.gate = torch.nn.Parameter(torch.ones(1))
        self.gate_max = float(gate_max)
        d_out = 2 * d

        # fixed sketch conditioner: condition on a k-dim random projection of the masked dims.
        # A coupling stays invertible for any function of the conditioning set.
        use_sketch = bool(sketch) and d > sketch
        sketch_m = torch.randn(sketch, d) / d**0.5 if use_sketch else None

        # sketch_head>0: zero the sketch's tail columns so the conditioner reads only the PCA
        # head, starving the null-space per-point fingerprint (candidate out-of-sample collapse)
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

        with torch.no_grad():  # near-identity init: zero the head
            self.net[-1].weight.mul_(0.0)
            self.net[-1].bias.mul_(0.0)

    def _st(self, xm):
        free = 1 - self.mask
        inp = xm @ self.S.T if self.S is not None else xm
        s_raw, t_raw = self.net(inp).chunk(2, dim=1)

        gate = (
            self.gate if self.gate_max <= 0 else self.gate_max * torch.tanh(self.gate)
        )

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

    gmm = GaussianMixture(
        best, covariance_type="diag", random_state=seed, reg_covar=1e-6
    ).fit(Y)

    return (
        torch.tensor(np.log(gmm.weights_), dtype=torch.float32),
        torch.tensor(gmm.means_, dtype=torch.float32),
        torch.tensor(np.log(gmm.covariances_), dtype=torch.float32),
    )


class CondBase(torch.nn.Module):
    def __init__(self, d, hid=128, k=2):
        super().__init__()
        self.k = k
        self.y_mu = torch.nn.Parameter(torch.zeros(k))
        self.y_logvar = torch.nn.Parameter(torch.zeros(k))

        self.net = torch.nn.Sequential(
            torch.nn.Linear(k, hid),
            torch.nn.Tanh(),
            torch.nn.Linear(hid, hid),
            torch.nn.Tanh(),
            torch.nn.Linear(hid, 2 * (d - k)),
        )

        with torch.no_grad():  # start at the unconditional standard normal
            self.net[-1].weight.mul_(0.0)
            self.net[-1].bias.mul_(0.0)

    def r_params(self, y):
        out = self.net(y)
        half = out.shape[1] // 2

        return out[:, :half], out[:, half:].clamp(-8.0, 8.0)

    def fit_y_density(self, Y, ks=(16, 32, 64, 128, 256), seed=0):
        dev = self.y_mu.device

        for name, val in zip(
            ("ylogw", "ymu", "ylogvar"), _fit_y_gmm(Y, ks=ks, seed=seed)
        ):
            self.register_buffer(name, val.to(dev))
        return self

    def y_log_prob(self, y):
        l2pi = math.log(2 * math.pi)

        if getattr(self, "ylogw", None) is not None:  # fitted GMM (preferred)
            diff = y[:, None, :] - self.ymu[None, :, :]  # (n, K, k)
            log_p = -0.5 * (
                diff**2 * torch.exp(-self.ylogvar[None]) + self.ylogvar[None] + l2pi
            ).sum(2)  # (n, K)

            return torch.logsumexp(self.ylogw[None] + log_p, dim=1)

        return -0.5 * (
            (y - self.y_mu) ** 2 * torch.exp(-self.y_logvar) + self.y_logvar + l2pi
        ).sum(1)

    def log_prob(self, y, r):
        l2pi = math.log(2 * math.pi)
        mu, logvar = self.r_params(y)
        log_p_r = -0.5 * ((r - mu) ** 2 * torch.exp(-logvar) + logvar + l2pi).sum(1)

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
        k=2,
    ):
        super().__init__()
        layers = []

        for i in range(n_layers):
            mask = torch.zeros(d)
            mask[torch.randperm(d)[: d // 2]] = 1.0
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
                )
            )

        self.layers = torch.nn.ModuleList(layers)
        self.tail = (
            torch.nn.ModuleList()
        )  # y-conditioned density couplings (fit_cond_tail)

        self._cond = None  # CondBase, set by fit_cond_tail
        self._density_args = None  # set by train_flodr for lazy density fitting
        self.base_logvar = torch.nn.Parameter(torch.zeros(d))
        self.d, self.hid, self.k = d, hid, k

        for name in (
            "gmm_logw",
            "gmm_W",
            "gmm_muW",
            "gmm_logdet",
        ):  # set by fit_gmm_base
            self.register_buffer(name, None)

    def forward(self, x):
        for layer in self.layers:
            x = layer(x)

        for layer in self.tail:  # identity on dims 0-1: embedding unchanged
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

    def inverse(self, z):
        for layer in reversed(self.tail):
            z = layer.inverse(z)

        for layer in reversed(self.layers):
            z = layer.inverse(z)

        return z

    def forward_main(self, x):
        for layer in self.layers:
            x = layer(x)

        return x

    def inverse_main(self, z):
        for layer in reversed(self.layers):
            z = layer.inverse(z)

        return z

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
        _latents=None,
        _ydensity=None,
    ):
        if len(self.tail):
            raise RuntimeError("fit_cond_tail was already called on this flow")

        dev = self.base_logvar.device

        if _latents is not None:
            z_main, ld_main = _latents  # precomputed by fit_cond_tail_cv
        else:
            with torch.no_grad():
                # x may already be a device tensor; np.asarray on a cuda tensor raises
                x_tens = x if torch.is_tensor(x) else torch.as_tensor(np.asarray(x))
                z_main, ld_main = self.forward_with_logdet(
                    x_tens.to(device=dev, dtype=torch.float32)
                )

        torch.manual_seed(seed + 1000)
        d, k = self.d, self.k
        tail = torch.nn.ModuleList()

        for _ in range(n_tail):
            mask = torch.zeros(d)
            mask[:k] = 1.0  # y always in the conditioning (identity) set
            mask[k + torch.randperm(d - k)[: (d - k) // 2]] = 1.0
            tail.append(Coupling(d, mask, self.hid))

        tail.to(dev)
        cond = CondBase(d, cond_hid, k=k).to(dev)
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

        stack = torch.enable_grad()  # lazy fit may run inside a caller's no_grad
        stack.__enter__()
        graph_loop(iters, draw, body, dev)
        stack.__exit__(None, None, None)
        self.tail = tail
        self._cond = cond
        self._density_args = None

        # fit the GMM p(y) so the default density knows where the data is; the single Gaussian
        # is not usable for this. The main flow is already frozen.
        if _ydensity is not None:  # precomputed by fit_cond_tail_cv
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

    def fit_cond_tail_cv(
        self,
        x,
        seed=0,
        grid=(
            (0, 32, 1500, 0.0, 0.0),
            (0, 128, 300, 0.0, 0.0),
            (2, 128, 1500, 0.0, 0.0),
            (2, 64, 800, 0.05, 1e-3),
            (2, 128, 800, 0.1, 1e-2),
            (0, 32, 1500, 0.05, 0.0),
        ),
    ):
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

        # candidates share the frozen main flow: compute the latents and p(y) GMM once
        # (bitwise-identical to per-candidate recomputation)
        with torch.no_grad():

            def _main_latents(x_in):
                z, logdet = x_in, x_in.new_zeros(x_in.shape[0])
                for layer in self.layers:
                    z, ld_layer = layer.forward_logdet(z)
                    logdet = logdet + ld_layer
                return z, logdet

            z_tr, ld_tr = _main_latents(x_all[tr_idx])
            z_va, ld_va = _main_latents(x_all[va_idx])

        y_gmm = _fit_y_gmm(z_tr[:, : self.k].cpu().numpy(), seed=seed)
        best, best_v = None, np.inf

        for cand in grid:
            n_tail, hid, iters, floor, w_decay = (tuple(cand) + (0.0, 0.0))[:5]
            cand_flow = copy.deepcopy(self)
            cand_flow._density_args = None
            cand_flow.tail, cand_flow._cond = (
                torch.nn.ModuleList(),
                None,
            )  # idempotent: strip any prior fit

            cand_flow.fit_cond_tail(
                x[tr_idx],
                n_tail=n_tail,
                iters=iters,
                cond_hid=hid,
                seed=seed,
                noise_floor=floor,
                weight_decay=w_decay,
                _latents=(z_tr, ld_tr),
                _ydensity=y_gmm,
            )

            with (
                torch.no_grad()
            ):  # tail-only forward, identical to cand_flow.log_prob(x[va_idx])
                z, logdet = z_va.clone(), ld_va.clone()
                for layer in cand_flow.tail:
                    z, ld_layer = layer.forward_logdet(z)
                    logdet = logdet + ld_layer
                nll = -float((cand_flow.base_log_prob(z) + logdet).mean()) / cand_flow.d

            if nll < best_v:
                best, best_v = (n_tail, hid, iters, floor, w_decay), nll

        n_tail, hid, iters, floor, w_decay = best
        self._density_args = None
        self.tail, self._cond = torch.nn.ModuleList(), None
        self.cond_choice_ = dict(
            n_tail=n_tail,
            cond_hid=hid,
            iters=iters,
            noise_floor=floor,
            weight_decay=w_decay,
            val_nll=best_v,
        )

        return self.fit_cond_tail(
            x,
            n_tail=n_tail,
            iters=iters,
            cond_hid=hid,
            seed=seed,
            noise_floor=floor,
            weight_decay=w_decay,
        )

    @property
    def cond(self):
        if self._cond is None and self._density_args is not None:
            x, n_tail, iters, seed = self._density_args
            self.fit_cond_tail(x, n_tail=n_tail, iters=iters, seed=seed)

        return self._cond

    def fit_density(self):
        _ = self.cond

        return self

    def fit_gmm_base(
        self, x, n_components=20, seed=0, max_iter=100, tol=1e-3, reg_covar=1e-4
    ):
        with torch.no_grad():
            z32 = self.forward(
                torch.as_tensor(x, dtype=torch.float32, device=self.base_logvar.device)
            )

        n_comp = max(1, min(n_components, z32.shape[0] // (2 * self.d)))

        if not z32.is_cuda:
            gmm = GaussianMixture(
                n_comp,
                covariance_type="full",
                reg_covar=reg_covar,
                max_iter=max_iter,
                tol=tol,
                random_state=seed,
            ).fit(z32.double().cpu().numpy())

            weights = torch.from_numpy(gmm.weights_)
            means = torch.from_numpy(gmm.means_)
            w_chol = torch.from_numpy(gmm.precisions_cholesky_)
        else:
            z = z32.double()
            n, d = z.shape
            eye = torch.eye(d, dtype=torch.float64, device=z.device)
            labels = KMeans(n_comp, n_init=1, random_state=seed).fit_predict(
                z.cpu().numpy()
            )
            resp = torch.zeros(n, n_comp, dtype=torch.float64, device=z.device)
            resp[
                torch.arange(n, device=z.device),
                torch.as_tensor(labels, device=z.device),
            ] = 1.0

            def m_step(resp):
                n_k = resp.sum(0) + 1e-12
                means = (resp.T @ z) / n_k[:, None]
                cov = torch.empty(n_comp, d, d, dtype=torch.float64, device=z.device)

                for comp in range(n_comp):
                    diff = z - means[comp]
                    cov[comp] = (diff * resp[:, comp : comp + 1]).T @ diff / n_k[
                        comp
                    ] + reg_covar * eye

                return n_k / n, means, cov

            def prec_chol(cov):
                chol = torch.linalg.cholesky(cov)

                return torch.linalg.solve_triangular(
                    chol, eye.expand(n_comp, d, d).contiguous(), upper=False
                ).transpose(1, 2)  # precision = W W^T

            weights, means, cov = m_step(resp)
            lb = -torch.inf

            for _ in range(max_iter):  # E-step; stop on lower-bound convergence
                w_chol = prec_chol(cov)
                y = (
                    torch.einsum("nd,kde->kne", z, w_chol)
                    - torch.einsum("kd,kde->ke", means, w_chol)[:, None, :]
                )
                log_p = (
                    -0.5 * ((y**2).sum(-1) + d * math.log(2 * math.pi))
                    + torch.diagonal(w_chol, dim1=1, dim2=2).log().sum(1)[:, None]
                    + weights.log()[:, None]
                )
                loglik = torch.logsumexp(log_p, 0)
                lb_new = float(loglik.mean())

                if abs(lb_new - lb) < tol:
                    break

                lb = lb_new
                weights, means, cov = m_step((log_p - loglik).T.exp())
            w_chol = prec_chol(cov)

        dev = z32.device
        self.gmm_logw = weights.float().log().to(dev)
        self.gmm_W = w_chol.float().to(dev)
        self.gmm_muW = torch.einsum("kd,kde->ke", means, w_chol).float().to(dev)
        self.gmm_logdet = (
            torch.diagonal(w_chol, dim1=1, dim2=2).log().sum(1).float().to(dev)
        )

        return self

    def base_log_prob(self, z):
        l2pi = math.log(2 * math.pi)
        if self.cond is not None:
            return self.cond.log_prob(z[:, : self.k], z[:, self.k :])

        if self.gmm_logw is not None:
            y = torch.einsum("nd,kde->kne", z, self.gmm_W) - self.gmm_muW[:, None, :]
            log_p = -0.5 * ((y**2).sum(-1) + self.d * l2pi) + self.gmm_logdet[:, None]
            return torch.logsumexp(self.gmm_logw[:, None] + log_p, 0)

        return -0.5 * (
            z**2 * torch.exp(-self.base_logvar) + self.base_logvar + l2pi
        ).sum(1)

    def log_prob(self, x):
        _ = self.cond  # deferred tail must exist before the forward pass
        z, logdet = self.forward_with_logdet(x)

        return self.base_log_prob(z) + logdet
