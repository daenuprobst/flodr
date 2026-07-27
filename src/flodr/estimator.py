import copy

import numpy as np
import torch
from scipy.spatial import cKDTree

from .data import fuzzy_knn_graph, raw_coords
from .model import graph_loop
from .train import raw_recipe, train_flodr
from . import viz


class NotFittedError(ValueError):
    pass


class FloDR:
    # frozen at the recipe behind the paper's main comparison; not user knobs on purpose
    _FROZEN = dict(n_neighbors=15, pca_head=50, stress_ordinal=True)

    def __init__(
        self,
        w=2.0,
        *,
        random_state=0,
        device="cpu",
        density=False,
        n_components=2,
        w_stress=None,
        advanced=None,
    ):
        self.w = (
            float(w_stress) if w_stress is not None else float(w)
        )  # w_stress: legacy alias
        self.random_state = random_state
        self.device = device
        self.density = density
        self.n_components = n_components
        self.advanced = dict(advanced or {})

    # legacy alias, so existing callers keep working
    @property
    def w_stress(self):
        return self.w

    @w_stress.setter
    def w_stress(self, v):
        self.w = float(v)

    def get_params(self, deep=True):
        return {
            k: getattr(self, k)
            for k in (
                "w",
                "random_state",
                "device",
                "density",
                "n_components",
                "advanced",
            )
        }

    def set_params(self, **kw):
        for k, v in kw.items():
            if k == "w_stress":
                self.w = float(v)
            elif k in (
                "w",
                "random_state",
                "device",
                "density",
                "n_components",
                "advanced",
            ):
                setattr(self, k, v)
            else:
                self.advanced[k] = v  # unsupported, but do not silently drop it
        return self

    @staticmethod
    def _metric_for(X):
        uniq = np.unique(X[: min(len(X), 1000)])

        return (
            "jaccard"
            if uniq.size <= 2 and set(uniq.tolist()) <= {0.0, 1.0}
            else "euclid"
        )

    def _config(self, X):
        kw = dict(self.advanced)

        if self.w:
            kw.update(
                w_stress=self.w,
                stress_ordinal=int(self._FROZEN["stress_ordinal"]),
                stress_metric=kw.pop("stress_metric", None) or self._metric_for(X),
            )

        if self.density:
            kw.setdefault("w_nll", 0.5)
            kw.setdefault("cond_tail", 2)

        kw.setdefault("k", self.n_components)
        return raw_recipe(seed=self.random_state, device=self.device, **kw)

    def _check(self):
        if not hasattr(self, "flow_"):
            raise NotFittedError("call fit first")

    def fit(self, X, y=None):
        if not (isinstance(self.n_components, int) and self.n_components >= 2):
            raise ValueError(
                "n_components must be an int >= 2 (the embedding is the flow's first "
                "n_components output dims)"
            )

        X = np.asarray(X, dtype=np.float32)
        self.n_features_in_ = X.shape[1]
        Z, self._to_raw, self._to_coords = raw_coords(
            X, self._FROZEN["pca_head"], self.random_state
        )

        if self.n_components >= Z.shape[1]:
            raise ValueError(
                f"n_components ({self.n_components}) must be < the latent dimension "
                f"({Z.shape[1]}); the residual needs at least one dim to stay invertible"
            )

        self.knn_idx_, edge_i, edge_j, w_attr, self.knn_dist_ = fuzzy_knn_graph(
            X, self._FROZEN["n_neighbors"], return_dist=True
        )
        Y, roundtrip, _, flow = train_flodr(
            Z,
            edge_i,
            edge_j,
            self._config(X),
            np.random.default_rng(self.random_state),
            return_model=True,
            w=w_attr,
            stress_X=X,
        )

        self.embedding_, self.roundtrip_, self.flow_ = (
            np.asarray(Y),
            float(roundtrip),
            flow,
        )

        self._coords = Z

        return self

    def fit_transform(self, X, y=None):
        return self.fit(X).embedding_

    def _forward(self, X):
        z_new = self._to_coords(np.asarray(X, dtype=np.float32))
        t = torch.from_numpy(z_new).float().to(next(self.flow_.parameters()).device)

        with torch.no_grad():
            return self.flow_(t)

    def transform(self, X):
        self._check()
        return self._forward(X)[:, : self.n_components].cpu().numpy()

    def inverse_transform(self, Y):
        self._check()
        Y = np.asarray(Y, dtype=np.float32)
        dev = next(self.flow_.parameters()).device
        z = torch.zeros((len(Y), self.flow_.d), dtype=torch.float32, device=dev)
        z[:, : self.n_components] = torch.from_numpy(Y).to(dev)

        with torch.no_grad():
            return self._to_raw(self.flow_.inverse(z).cpu().numpy())

    def inverse_coords(self, Zlat):
        self._check()

        with torch.no_grad():
            return self._to_raw(self.flow_.inverse(torch.as_tensor(Zlat)).cpu().numpy())

    def score_samples(self, X):
        self._check()
        self._need_density()
        with torch.no_grad():
            return self.flow_.log_prob(self._forward(X)).cpu().numpy()

    def _need_density(self):
        if not self.density:
            raise ValueError(
                "density is off. conditional_spread/atypicality are functionals of the fitted "
                "p(r|y), which the raw recipe does not train (w_nll=0, cond_tail=0). Refit with "
                "FloDR(density=True)."
            )

        if self.flow_._cond is None:
            # capacity-selected conditional (fit_cond_tail_cv), NOT the deferred fixed-capacity
            # fit: the fixed default memorises small samples (digits n=898: -1.3 nats/dim
            # in-sample, +154 held-out), which turns sigma(y) into noise between train points
            self.flow_._density_args = None
            self.flow_.fit_cond_tail_cv(self._coords, seed=self.random_state)

    def conditional_spread(self, Y=None, n_samples=64):
        self._check()
        self._need_density()
        Y = self.embedding_ if Y is None else np.asarray(Y, dtype=np.float32)

        return viz.conditional_spread(
            self.flow_,
            Y,
            to_input=self._to_raw,
            n_samples=n_samples,
            seed=self.random_state,
        )

    def atypicality(self, n_samples=128):
        self._check()
        self._need_density()

        return viz.conditional_atypicality(
            self.flow_,
            self._coords,
            to_input=self._to_raw,
            n_samples=n_samples,
            seed=self.random_state,
        )

    def diagnostics(self, G=None, **kw):
        self._check()
        out = {}

        if self.density:
            sigma = np.asarray(self.conditional_spread())
            out["spread"] = dict(
                field=sigma,
                cert=self.spread_calibration(),
                units="input units",
                label=r"$\sigma(y)$",
            )
        if G is not None:
            field, cert = self.hidden_contrast(G, **kw)
            field_null = cert.pop(
                "field_null", None
            )  # absent on the undecided early return
            out["hidden_contrast"] = dict(
                field=field,
                field_null=None if field_null is None else np.asarray(field_null),
                cert=cert,
                units="nats",
                label=r"$I(G;R \mid Y{=}y)$",
            )
        return out

    def _class_logp_batched(
        self, Xin, gRN, rows_fit, n_cls, hid, iters, seed, wd=1e-4, batch=1024
    ):
        dev = next(self.flow_.parameters()).device
        R = len(gRN)
        d_in = Xin.shape[1]
        gen = torch.Generator(device=dev)
        gen.manual_seed(seed)

        def par(*shape, fan):
            b = 1.0 / np.sqrt(fan)
            return torch.nn.Parameter(
                (torch.rand(*shape, generator=gen, device=dev) * 2 - 1) * b
            )

        W1, b1 = par(R, d_in, hid, fan=d_in), par(R, hid, fan=d_in)
        W2, b2 = par(R, hid, hid, fan=hid), par(R, hid, fan=hid)
        W3, b3 = par(R, hid, n_cls, fan=hid), par(R, n_cls, fan=hid)
        params = [W1, b1, W2, b2, W3, b3]

        Xt = torch.as_tensor(np.asarray(Xin, np.float32), device=dev)
        gt = torch.as_tensor(np.asarray(gRN, np.int64), device=dev)  # (R, n)
        fit_rows = torch.as_tensor(np.asarray(rows_fit, np.int64), device=dev)
        opt = torch.optim.Adam(
            params, lr=1e-3, weight_decay=wd, capturable=dev.type == "cuda"
        )
        bs = min(batch, len(rows_fit))
        sel = torch.zeros(bs, dtype=torch.long, device=dev)

        def fwd(xb):
            h = torch.tanh(torch.einsum("bi,rih->rbh", xb, W1) + b1[:, None, :])
            h = torch.tanh(torch.einsum("rbh,rhg->rbg", h, W2) + b2[:, None, :])

            return torch.einsum("rbg,rgc->rbc", h, W3) + b3[:, None, :]

        def draw():
            sel.copy_(
                fit_rows[
                    torch.randint(0, len(fit_rows), (bs,), device=dev, generator=gen)
                ]
            )

        def body():
            logits = fwd(Xt[sel])  # (R, bs, n_cls)
            loss = torch.nn.functional.cross_entropy(
                logits.reshape(-1, n_cls), gt[:, sel].reshape(-1)
            )
            opt.zero_grad(set_to_none=False)
            loss.backward()
            opt.step()

        graph_loop(iters, draw, body, dev)
        out = np.empty((R, len(Xt)), np.float64)

        with torch.no_grad():
            for start in range(0, len(Xt), 8192):  # chunked: R x n x C is large
                stop = min(start + 8192, len(Xt))
                lp = torch.log_softmax(fwd(Xt[start:stop]), 2)
                idx = gt[:, start:stop].unsqueeze(2)
                out[:, start:stop] = lp.gather(2, idx).squeeze(2).cpu().numpy()

        return out

    def _class_logp(
        self, Xin, g, rows_fit, rows_eval, n_cls, hid, iters, seed, wd=1e-4
    ):
        dev = next(self.flow_.parameters()).device
        torch.manual_seed(seed)

        net = torch.nn.Sequential(
            torch.nn.Linear(Xin.shape[1], hid),
            torch.nn.Tanh(),
            torch.nn.Linear(hid, hid),
            torch.nn.Tanh(),
            torch.nn.Linear(hid, n_cls),
        ).to(dev)

        Xt = torch.as_tensor(np.asarray(Xin, np.float32), device=dev)
        gt = torch.as_tensor(np.asarray(g, np.int64), device=dev)
        opt = torch.optim.Adam(
            net.parameters(), lr=1e-3, weight_decay=wd, capturable=dev.type == "cuda"
        )
        gen = torch.Generator(device=dev)
        gen.manual_seed(seed)
        fit_rows = torch.as_tensor(np.asarray(rows_fit, np.int64), device=dev)
        bs = min(1024, len(rows_fit))
        sel = torch.zeros(bs, dtype=torch.long, device=dev)
        loss_fn = torch.nn.functional.cross_entropy

        def draw():
            sel.copy_(
                fit_rows[
                    torch.randint(0, len(fit_rows), (bs,), device=dev, generator=gen)
                ]
            )

        def body():
            loss = loss_fn(net(Xt[sel]), gt[sel])
            opt.zero_grad(set_to_none=False)
            loss.backward()
            opt.step()

        graph_loop(iters, draw, body, dev)
        eval_rows = torch.as_tensor(np.asarray(rows_eval, np.int64), device=dev)

        with torch.no_grad():
            lp = torch.log_softmax(net(Xt[eval_rows]), 1)

            return (
                lp[torch.arange(len(eval_rows), device=dev), gt[eval_rows]]
                .cpu()
                .numpy()
            )

    def hidden_contrast(
        self,
        G,
        n_bins=14,
        min_pts=16,
        k_local=10,
        hid=128,
        iters=3000,
        n_perm=99,
        alpha=0.01,
    ):
        self._check()
        Y = np.asarray(self.embedding_, np.float64)
        n = len(Y)
        g = np.asarray(G)
        cls, g_idx = np.unique(g, return_inverse=True)
        n_cls = len(cls)
        if n_cls < 2:
            raise ValueError("G must take at least two values")

        rng = np.random.default_rng(self.random_state + 17)
        perm = rng.permutation(n)
        fold_c, fold_a, fold_b = (
            perm[: n // 3],
            perm[n // 3 : 2 * n // 3],
            perm[2 * n // 3 :],
        )

        lo, hi = Y.min(0), Y.max(0)
        bin_ix = np.clip(
            ((Y - lo) / (hi - lo + 1e-9) * n_bins).astype(int), 0, n_bins - 1
        )
        bin_id = bin_ix[:, 0] * n_bins + bin_ix[:, 1]
        bins = [np.nonzero(bin_id == b)[0] for b in np.unique(bin_id)]

        x_in = np.asarray(self._coords, np.float64)
        tree_a = cKDTree(Y[fold_a])
        _, nn_a = tree_a.query(Y, k=min(k_local, len(fold_a)))

        # replica 0 is the real contrast; 1..n_perm are within-bin shuffles. All are fitted in
        # one batched pass per view, so the permutation null costs little more than the signal.
        g_rn = np.empty((n_perm + 1, n), np.int64)
        g_rn[0] = g_idx
        for perm_i in range(n_perm):
            g_perm = g_idx.copy()
            for bin_rows in bins:  # shuffle within bins: contrast dies, marginal lives
                g_perm[bin_rows] = g_perm[rng.permutation(bin_rows)]
            g_rn[perm_i + 1] = g_perm

        lp_x = self._class_logp_batched(
            x_in, g_rn, fold_c, n_cls, hid, iters, self.random_state + 21
        )
        lp_y = self._class_logp_batched(
            Y, g_rn, fold_c, n_cls, hid, iters, self.random_state + 21
        )
        gaps = lp_x - lp_y  # (R, n) nats recoverable from x but not from y
        gap, null_gaps = gaps[0], list(gaps[1:])
        null_gap = np.mean(null_gaps, 0)

        # field: fold-A gap smoothed over screen position, read out anywhere. A was not seen
        # by either classifier, so the field carries no in-sample optimism; subtracting the
        # null's field removes the capacity-mismatch bias, which is not zero.
        field = (gap - null_gap)[fold_a][nn_a].mean(1)
        gap = gap - null_gap
        null_level = [float(np.mean((ng - null_gap)[fold_b])) for ng in null_gaps]

        # a visual null for the figure: one permutation debiased by the OTHERS, so it is a
        # genuine held-out null rather than a residual that is zero by construction
        field_null = (
            ((null_gaps[0] - np.mean(null_gaps[1:], 0))[fold_a][nn_a].mean(1))
            if n_perm >= 2
            else np.zeros_like(field)
        )

        # certificate: does the fold-A field predict fold-B's own gap, per screen bin?
        bin_pairs = []
        for b in np.unique(bin_id[fold_b]):
            sel = fold_b[bin_id[fold_b] == b]
            if len(sel) < min_pts:
                continue
            bin_pairs.append((float(field[sel].mean()), float(gap[sel].mean())))
        if len(bin_pairs) < 3:
            return field, dict(
                passed=None, certifies="undecided", n_bins=len(bin_pairs)
            )
        pred, tgt = (
            np.array([pair[0] for pair in bin_pairs]),
            np.array([pair[1] for pair in bin_pairs]),
        )
        # linear, not log: the gap is a difference of log-likelihoods and may be negative
        design = np.column_stack([pred, np.ones_like(pred)])
        coef, *_ = np.linalg.lstsq(design, tgt, rcond=None)
        r2 = float(
            1
            - ((tgt - design @ coef) ** 2).sum()
            / max(((tgt - tgt.mean()) ** 2).sum(), 1e-12)
        )
        dyn_range = float(np.percentile(tgt, 95) - np.percentile(tgt, 5))
        level = (
            float(tgt.mean() / pred.mean()) if abs(pred.mean()) > 1e-9 else float("nan")
        )
        rng_b = np.random.default_rng(self.random_state + 23)
        n_pairs = len(pred)
        slope_pass, r2_pass = [], []

        for _ in range(2000):
            boot_ix = rng_b.integers(0, n_pairs, n_pairs)
            design_b = np.column_stack([pred[boot_ix], np.ones(n_pairs)])

            try:
                coef_b, *_ = np.linalg.lstsq(design_b, tgt[boot_ix], rcond=None)
                r2_b = 1 - ((tgt[boot_ix] - design_b @ coef_b) ** 2).sum() / max(
                    ((tgt[boot_ix] - tgt[boot_ix].mean()) ** 2).sum(), 1e-12
                )
            except np.linalg.LinAlgError:
                continue

            slope_pass.append(0.7 <= coef_b[0] <= 1.3)
            r2_pass.append(r2_b >= 0.6)
        # level: does the debiased signal clear its own permutation null, one-sided
        obs = float(np.mean(gap[fold_b]))
        p_level = (1.0 + sum(lvl >= obs for lvl in null_level)) / (
            1.0 + len(null_level)
        )

        parts = {
            "slope": float(np.mean(slope_pass)),
            "r2": float(np.mean(r2_pass)),
            "level": float(1.0 - p_level),
        }

        shape_ok = bool(0.7 <= coef[0] <= 1.3 and r2 >= 0.6)
        passed = bool(shape_ok and p_level <= alpha)

        return field, dict(
            slope=float(coef[0]),
            r2=r2,
            level=level,
            n_bins=n_pairs,
            dynamic_range=dyn_range,
            mean_gap=obs,
            null_level=[round(lvl, 5) for lvl in null_level],
            p_level=float(p_level),
            n_perm=int(n_perm),
            shape_ok=shape_ok,
            field_null=field_null,
            confidence=float(min(parts.values())),
            confidence_parts=parts,
            passed=passed,
            certifies="magnitude+shape",
        )

    def spread_calibration(self, n_bins=14, min_pts=12, n_samples=64):
        self._check()

        if not self.density:
            raise ValueError("density is off; refit with FloDR(density=True)")

        n = len(self._coords)
        rng = np.random.default_rng(self.random_state)
        perm = rng.permutation(n)

        fold_a, fold_b = perm[: n // 2], perm[n // 2 :]
        flow_a = copy.deepcopy(self.flow_)
        flow_a._density_args = None
        flow_a.fit_cond_tail_cv(self._coords[fold_a], seed=self.random_state)

        mu, sigma = viz.conditional_moments(
            flow_a,
            self.embedding_[fold_b],
            to_input=self._to_raw,
            n_samples=n_samples,
            seed=self.random_state,
        )

        sigma2 = sigma**2
        X = self._to_raw(self._coords)
        Y = self.embedding_
        lo, hi = Y.min(0), Y.max(0)

        bin_ix = np.clip(
            ((Y[fold_b] - lo) / (hi - lo + 1e-9) * n_bins).astype(int), 0, n_bins - 1
        )

        bin_id = bin_ix[:, 0] * n_bins + bin_ix[:, 1]
        emp_var, mod_var = [], []

        for b in np.unique(bin_id):
            sel = bin_id == b

            if sel.sum() < min_pts:
                continue

            x_bin = X[fold_b][sel].astype(np.float64)
            emp_var.append(((x_bin - x_bin.mean(0)) ** 2).sum(1).mean())

            # Var(mu) term: without it, bin smearing biases the slope up (measured 1.74 on
            # a field whose sigma/truth = 1.00)
            mu_bin = mu[sel].astype(np.float64)
            mod_var.append(
                float(
                    sigma2[sel].mean() + ((mu_bin - mu_bin.mean(0)) ** 2).sum(1).mean()
                )
            )

        emp_var, mod_var = np.array(emp_var), np.array(mod_var)

        # log space: bin variances span decades, raw regression is dominated by the top bins
        log_emp, log_mod = (
            np.log(np.maximum(emp_var, 1e-12)),
            np.log(np.maximum(mod_var, 1e-12)),
        )

        design = np.column_stack([log_mod, np.ones_like(log_mod)])
        coef, *_ = np.linalg.lstsq(design, log_emp, rcond=None)

        r2 = float(
            1
            - ((log_emp - design @ coef) ** 2).sum()
            / ((log_emp - log_emp.mean()) ** 2).sum()
        )

        ratio = mod_var / np.maximum(emp_var, 1e-12)

        dyn_range = float(
            np.percentile(emp_var, 95) / max(np.percentile(emp_var, 5), 1e-12)
        )

        level_ok = bool(
            0.5 <= np.median(ratio) <= 2.0
        )  # the log intercept absorbs constant

        passed = level_ok and (
            dyn_range < 3.0  # bias, so level is gated separately
            or (0.7 <= coef[0] <= 1.3 and r2 >= 0.6)
        )

        # bootstrap over bins, intersection-union: scarce data gives low confidence on its
        # own, separating "insufficient data" from "measured miscalibration"
        rng_b = np.random.default_rng(self.random_state + 7)
        n_pairs = len(emp_var)
        slope_pass, r2_pass, level_pass = [], [], []

        for _ in range(2000):
            boot_ix = rng_b.integers(0, n_pairs, n_pairs)
            le_b, lm_b, ratio_b = log_emp[boot_ix], log_mod[boot_ix], ratio[boot_ix]
            design_b = np.column_stack([lm_b, np.ones_like(lm_b)])

            try:
                coef_b, *_ = np.linalg.lstsq(design_b, le_b, rcond=None)
                r2_b = 1 - ((le_b - design_b @ coef_b) ** 2).sum() / max(
                    ((le_b - le_b.mean()) ** 2).sum(), 1e-12
                )
            except np.linalg.LinAlgError:
                continue

            slope_pass.append(0.7 <= coef_b[0] <= 1.3)
            r2_pass.append(r2_b >= 0.6)
            level_pass.append(0.5 <= np.median(ratio_b) <= 2.0)

        conf_parts = {"level": float(np.mean(level_pass))}

        if dyn_range >= 3.0:
            conf_parts["slope"] = float(np.mean(slope_pass))
            conf_parts["r2"] = float(np.mean(r2_pass))

        confidence = float(min(conf_parts.values()))

        return dict(
            slope=float(coef[0]),
            r2=r2,
            ratio_quartiles=tuple(np.percentile(ratio, [25, 50, 75]).round(3)),
            dynamic_range=dyn_range,
            n_bins=len(emp_var),
            passed=passed,
            confidence=confidence,
            confidence_parts=conf_parts,
            certifies="magnitude+shape" if dyn_range >= 3.0 else "magnitude-only",
        )
