import copy
import dataclasses
import numbers
import warnings

import numpy as np
import torch
from scipy.spatial import cKDTree
from sklearn.base import (
    BaseEstimator,
    ClassNamePrefixFeaturesOutMixin,
    TransformerMixin,
)
from sklearn.utils import check_array, check_random_state
from sklearn.utils.metaestimators import available_if
from sklearn.utils.validation import check_is_fitted, validate_data

from . import viz
from .data import find_ab, fuzzy_knn_graph, knn_query, raw_coords
from .model import graph_loop
from .train import TrainConfig, default_device, raw_recipe, train_flodr

# keeping the heaviest quarter of the fuzzy graph is +32 to +41% recall between 50k and
# 500k points, flat past a million, and costs recall below 10k
PRUNE_KEEP = 0.25
PRUNE_RANGE = (50_000, 500_000)


def prune_fraction(n):
    lo, hi = PRUNE_RANGE

    return PRUNE_KEEP if lo <= n <= hi else 1.0


def _flow_to(flow, dev):
    # parameters and buffers, the plain tensors some modules keep, and the deferred density input
    flow.to(dev)

    for m in flow.modules():
        for k, v in list(vars(m).items()):
            if isinstance(v, torch.Tensor):
                setattr(m, k, v.to(dev))

    args = getattr(flow, "_density_args", None)

    if args is not None:
        flow._density_args = (args[0].to(dev), *args[1:])

    return flow


def _has_density(est):
    # hides score and score_samples from hasattr without a density, as sklearn expects
    if not est.density:
        raise AttributeError(
            "score and score_samples need the density; refit with FloDR(density=True)"
        )

    return True


class FloDR(ClassNamePrefixFeaturesOutMixin, TransformerMixin, BaseEstimator):
    # the recipe behind the paper's main comparison, deliberately not user knobs
    _FROZEN = dict(n_neighbors=15, pca_head=50, stress_ordinal=True)

    def __init__(
        self,
        n_components=2,
        *,
        w=2.0,
        density=True,
        max_iter=None,
        random_state=0,
        device=None,
        verbose=True,
        advanced=None,
    ):
        self.n_components = n_components
        self.w = w
        self.density = density
        self.max_iter = max_iter
        self.random_state = random_state
        self.device = device
        self.verbose = verbose
        self.advanced = advanced

    def __sklearn_tags__(self):
        tags = super().__sklearn_tags__()
        tags.transformer_tags.preserves_dtype = ["float32"]
        # the compiled CUDA step is not bitwise reproducible
        tags.non_deterministic = str(self.device or default_device()).startswith("cuda")

        return tags

    # a GPU fit pickles with its tensors on the CPU, so it loads on a machine without one
    def __getstate__(self):
        state = super().__getstate__()

        if "flow_" in state:
            state = dict(state, flow_=_flow_to(copy.deepcopy(self.flow_), "cpu"))

        return state

    def __setstate__(self, state):
        super().__setstate__(state)
        cuda = str(getattr(self, "_device", "cpu")).startswith("cuda")

        if "flow_" in state and cuda and torch.cuda.is_available():
            _flow_to(self.flow_, self._device)

    @staticmethod
    def _metric_for(X):
        uniq = np.unique(X[: min(len(X), 1000)])

        return (
            "jaccard"
            if uniq.size <= 2 and set(uniq.tolist()) <= {0.0, 1.0}
            else "euclid"
        )

    def _check_params(self, max_iter):
        n = self.n_components

        if not (isinstance(n, numbers.Integral) and not isinstance(n, bool) and n >= 2):
            raise ValueError(
                f"n_components must be an int >= 2 (the embedding is the flow's first "
                f"n_components output dims), got {n!r}"
            )

        if not (isinstance(self.w, numbers.Real) and self.w >= 0):
            raise ValueError(f"w must be a number >= 0, got {self.w!r}")

        if max_iter is not None and not (
            isinstance(max_iter, numbers.Integral) and max_iter >= 1
        ):
            raise ValueError(f"max_iter must be None or an int >= 1, got {max_iter!r}")

        if self.advanced is not None and not isinstance(self.advanced, dict):
            raise ValueError(f"advanced must be None or a dict, got {self.advanced!r}")

        fields = {f.name for f in dataclasses.fields(TrainConfig)}
        bad = sorted(set(self.advanced or {}) - fields)

        if bad:
            raise ValueError(f"advanced keys {bad} are not TrainConfig fields")

    def _seed_from(self):
        rs = self.random_state

        if isinstance(rs, numbers.Integral):
            return int(rs)

        if isinstance(rs, np.random.Generator):
            return int(rs.integers(2**31 - 1))

        # None draws a fresh seed, as in sklearn
        return int(check_random_state(rs).randint(2**31 - 1))

    def _config(self, X, max_iter):
        kw = dict(self.advanced or {})

        if max_iter is not None:
            kw["iters"] = int(max_iter)

        if self.w:
            kw.update(
                w_stress=float(self.w),
                stress_ordinal=int(self._FROZEN["stress_ordinal"]),
                stress_metric=kw.pop("stress_metric", None) or self._metric_for(X),
            )

        if self.density:
            kw.setdefault("w_nll", 0.5)
            kw.setdefault("cond_tail", 2)

        kw.setdefault("k", self.n_components)
        return raw_recipe(seed=self._seed, device=self._device, **kw)

    def _fit_args(self, iters, progress):
        # fit(iters=..., progress=...) predate max_iter and verbose
        if iters != "deprecated":
            warnings.warn(
                "fit(iters=...) is deprecated; use FloDR(max_iter=...)",
                FutureWarning,
                stacklevel=3,
            )

        if progress != "deprecated":
            warnings.warn(
                "fit(progress=...) is deprecated; use FloDR(verbose=...)",
                FutureWarning,
                stacklevel=3,
            )

        return (
            self.max_iter if iters == "deprecated" else iters,
            self.verbose if progress == "deprecated" else progress,
        )

    def fit(self, X, y=None, iters="deprecated", progress="deprecated"):
        max_iter, verbose = self._fit_args(iters, progress)
        self._check_params(max_iter)
        X = validate_data(
            self, X, dtype=np.float32, ensure_min_samples=2, ensure_min_features=2
        )
        self._seed = self._seed_from()
        self._device = default_device() if self.device is None else self.device

        # below 16 points every other point is a neighbour
        self._k = min(self._FROZEN["n_neighbors"], len(X) - 1)
        self._cfg = self._config(X, max_iter)
        Z, self._raw_of, self._coords_of = raw_coords(
            X, self._FROZEN["pca_head"], self._seed
        )

        # noise columns widen the residual the layout is routed through on narrow inputs.
        # new points get zeros there, and _to_raw drops them. padding a 50-D input
        # breaks the spread certificate
        self._d_z = Z.shape[1]
        self._n_pad = max(0, self._cfg.pad - self._d_z)
        noise = np.random.default_rng(self._seed + 4242).standard_normal(
            (len(Z), self._n_pad)
        )
        Z = np.hstack([Z, noise]).astype(np.float32)

        if self.n_components >= Z.shape[1]:
            raise ValueError(
                f"n_components ({self.n_components}) must be < the latent dimension "
                f"({Z.shape[1]}); the residual needs at least one dim to stay invertible"
            )

        self._metric = self._metric_for(X)

        # on binary data the graph is built in Jaccard, like the stress term
        pre = (
            knn_query(X, None, self._k, "jaccard")
            if self._metric == "jaccard"
            else None
        )
        self.knn_idx_, edge_i, edge_j, w_attr, self.knn_dist_ = fuzzy_knn_graph(
            X, self._k, precomputed=pre, return_dist=True
        )
        keep = prune_fraction(len(X))

        if keep < 1.0:
            sel = np.sort(np.argsort(-w_attr)[: round(keep * len(w_attr))])
            edge_i, edge_j, w_attr = edge_i[sel], edge_j[sel], w_attr[sel]

        Y, roundtrip, _, flow = train_flodr(
            Z,
            edge_i,
            edge_j,
            self._cfg,
            np.random.default_rng(self._seed),
            return_model=True,
            w=w_attr,
            stress_X=X,
            progress=verbose,
        )

        self.embedding_, self.roundtrip_, self.flow_ = (
            np.asarray(Y),
            float(roundtrip),
            flow,
        )
        self.n_iter_ = self._cfg.iters
        self._n_features_out = self.n_components

        self._coords = Z
        self._X = X

        return self

    def fit_transform(self, X, y=None, iters="deprecated", progress="deprecated"):
        return self.fit(X, iters=iters, progress=progress).embedding_

    # the fitted whitening; narrow inputs drop their noise columns on the way out and get
    # zeros there on the way in
    def _to_raw(self, Zn):
        return self._raw_of(np.asarray(Zn)[:, : self._d_z])

    def _to_coords(self, Xn):
        head = self._coords_of(Xn)

        return np.hstack([head, np.zeros((len(head), self._n_pad), np.float32)])

    def _forward(self, X):
        z_new = self._to_coords(np.asarray(X, dtype=np.float32))
        t = torch.from_numpy(z_new).float().to(next(self.flow_.parameters()).device)

        with torch.no_grad():
            return self.flow_(t)

    def transform(self, X):
        check_is_fitted(self)
        X = validate_data(self, X, dtype=np.float32, reset=False)

        return self._forward(X)[:, : self.n_components].cpu().numpy()

    def transform_opt(self, X, steps=200, lr=0.05):
        check_is_fitted(self)
        X = validate_data(self, X, dtype=np.float32, reset=False)
        _, nn = knn_query(self._X, X, self._k, self._metric)
        a, b = find_ab(self._cfg.min_dist)
        dev = next(self.flow_.parameters()).device
        Ytr = torch.from_numpy(np.ascontiguousarray(self.embedding_, np.float32)).to(
            dev
        )
        nn_t = torch.from_numpy(nn).to(dev)
        y = self._forward(X)[:, : self.n_components].clone().requires_grad_(True)
        opt = torch.optim.Adam([y], lr=lr)

        def q(d2):
            return 1.0 / (1.0 + a * (d2 + 1e-6) ** b)

        for _ in range(steps):
            d2 = ((y[:, None, :] - Ytr[nn_t]) ** 2).sum(-1)
            loss = -torch.log(q(d2) + 1e-6).mean()
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()

        return y.detach().cpu().numpy()

    # the full latent, layout first and then the residual; inverse_latent undoes it exactly
    def transform_latent(self, X):
        check_is_fitted(self)
        X = validate_data(self, X, dtype=np.float32, reset=False)

        return self._forward(X).cpu().numpy()

    def inverse_transform(self, Y):
        check_is_fitted(self)
        Y = check_array(Y, dtype=np.float32)

        if Y.shape[1] != self.n_components:
            raise ValueError(
                f"Y has {Y.shape[1]} columns but the layout has {self.n_components}"
            )

        dev = next(self.flow_.parameters()).device
        z = torch.zeros((len(Y), self.flow_.d), dtype=torch.float32, device=dev)
        z[:, : self.n_components] = torch.from_numpy(Y).to(dev)

        with torch.no_grad():
            return self._to_raw(self.flow_.inverse(z).cpu().numpy())

    def inverse_latent(self, Z):
        check_is_fitted(self)

        if isinstance(Z, torch.Tensor):
            Z = Z.detach().cpu().numpy()

        Z = check_array(Z, dtype=np.float32)

        if Z.shape[1] != self.flow_.d:
            raise ValueError(
                f"Z has {Z.shape[1]} columns but the latent has {self.flow_.d}"
            )

        dev = next(self.flow_.parameters()).device

        with torch.no_grad():
            z = torch.from_numpy(Z).to(dev)

            return self._to_raw(self.flow_.inverse(z).cpu().numpy())

    def inverse_coords(self, Zlat):
        warnings.warn(
            "inverse_coords is deprecated; use inverse_latent",
            FutureWarning,
            stacklevel=2,
        )

        return self.inverse_latent(Zlat)

    @available_if(_has_density)
    def score_samples(self, X):
        check_is_fitted(self)
        X = validate_data(self, X, dtype=np.float32, reset=False)
        self._need_density()

        # log_prob runs the flow itself, so it takes latent coords. passing _forward(X)
        # would apply the flow twice and return the wrong Jacobian with it
        dev = next(self.flow_.parameters()).device
        z = torch.from_numpy(self._to_coords(X)).float().to(dev)

        with torch.no_grad():
            return self.flow_.log_prob(z).cpu().numpy()

    # mean log density per sample, so GridSearchCV can select on held-out likelihood
    @available_if(_has_density)
    def score(self, X, y=None):
        return float(np.mean(self.score_samples(X)))

    def _need_density(self):
        if not self.density:
            raise ValueError(
                "density is off. conditional_spread/atypicality are functionals of the fitted "
                "p(r|y), which the raw recipe does not train (w_nll=0, cond_tail=0). Refit with "
                "FloDR(density=True)."
            )

        if self.flow_._cond is None:
            # capacity-selected, not the deferred fixed-capacity fit, because the fixed default
            # memorises small samples and sigma(y) becomes noise between train points
            self.flow_._density_args = None
            self.flow_.fit_cond_tail_cv(self._coords, seed=self._seed)

    def conditional_spread(self, Y=None, n_samples=64):
        check_is_fitted(self)
        self._need_density()
        Y = self.embedding_ if Y is None else np.asarray(Y, dtype=np.float32)

        return viz.conditional_spread(
            self.flow_,
            Y,
            to_input=self._to_raw,
            n_samples=n_samples,
            seed=self._seed,
        )

    def atypicality(self, n_samples=128):
        check_is_fitted(self)
        self._need_density()

        return viz.conditional_atypicality(
            self.flow_,
            self._coords,
            to_input=self._to_raw,
            n_samples=n_samples,
            seed=self._seed,
        )

    def diagnostics(self, G=None, **kw):
        check_is_fitted(self)
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
            # absent when undecided
            field_null = cert.pop("field_null", None)
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
        # (R, n)
        gt = torch.as_tensor(np.asarray(gRN, np.int64), device=dev)
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
            # (R, bs, n_cls)
            logits = fwd(Xt[sel])
            loss = torch.nn.functional.cross_entropy(
                logits.reshape(-1, n_cls), gt[:, sel].reshape(-1)
            )
            opt.zero_grad(set_to_none=False)
            loss.backward()
            opt.step()

        graph_loop(iters, draw, body, dev)
        out = np.empty((R, len(Xt)), np.float64)

        with torch.no_grad():
            # R x n x C is large
            for start in range(0, len(Xt), 8192):
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
        check_is_fitted(self)

        Y = np.asarray(self.embedding_, np.float64)
        n = len(Y)
        g = np.asarray(G)
        cls, g_idx = np.unique(g, return_inverse=True)

        n_cls = len(cls)
        if n_cls < 2:
            raise ValueError("G must take at least two values")

        rng = np.random.default_rng(self._seed + 17)
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

        # replica 0 is the real contrast, 1..n_perm are within-bin shuffles, and one batched
        # pass per view fits them all, so the null costs little more than the signal
        g_rn = np.empty((n_perm + 1, n), np.int64)
        g_rn[0] = g_idx
        for perm_i in range(n_perm):
            g_perm = g_idx.copy()

            # within bins the contrast dies, the marginal lives
            for bin_rows in bins:
                g_perm[bin_rows] = g_perm[rng.permutation(bin_rows)]

            g_rn[perm_i + 1] = g_perm

        lp_x = self._class_logp_batched(
            x_in, g_rn, fold_c, n_cls, hid, iters, self._seed + 21
        )
        lp_y = self._class_logp_batched(
            Y, g_rn, fold_c, n_cls, hid, iters, self._seed + 21
        )
        # (R, n) nats recoverable from x but not from y
        gaps = lp_x - lp_y
        gap, null_gaps = gaps[0], list(gaps[1:])
        null_gap = np.mean(null_gaps, 0)

        # the fold-A gap smoothed over screen position. Neither classifier saw A, so there
        # is no in-sample optimism, and subtracting the null removes the capacity-mismatch bias
        field = (gap - null_gap)[fold_a][nn_a].mean(1)
        gap = gap - null_gap
        null_level = [float(np.mean((ng - null_gap)[fold_b])) for ng in null_gaps]

        # one permutation debiased by the others, so the figure's null is held out rather
        # than a residual that is zero by construction
        field_null = (
            ((null_gaps[0] - np.mean(null_gaps[1:], 0))[fold_a][nn_a].mean(1))
            if n_perm >= 2
            else np.zeros_like(field)
        )

        # the certificate asks whether the fold-A field predicts fold-B's gap, per screen bin
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

        # linear, not log, the gap is a log-likelihood difference and can be negative
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
        rng_b = np.random.default_rng(self._seed + 23)
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

        # one-sided, asking whether the debiased signal clears its own permutation null
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
        check_is_fitted(self)

        if not self.density:
            raise ValueError("density is off; refit with FloDR(density=True)")

        n = len(self._coords)
        rng = np.random.default_rng(self._seed)
        perm = rng.permutation(n)

        fold_a, fold_b = perm[: n // 2], perm[n // 2 :]
        flow_a = copy.deepcopy(self.flow_)
        flow_a._density_args = None
        flow_a.fit_cond_tail_cv(self._coords[fold_a], seed=self._seed)

        mu, sigma = viz.conditional_moments(
            flow_a,
            self.embedding_[fold_b],
            to_input=self._to_raw,
            n_samples=n_samples,
            seed=self._seed,
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

            # without the Var(mu) term, bin smearing biases the slope up
            mu_bin = mu[sel].astype(np.float64)
            mod_var.append(
                float(
                    sigma2[sel].mean() + ((mu_bin - mu_bin.mean(0)) ** 2).sum(1).mean()
                )
            )

        emp_var, mod_var = np.array(emp_var), np.array(mod_var)

        # in log space, because bin variances span decades and the top bins would dominate
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

        # gated separately from the shape, because the log intercept absorbs a constant bias
        level_ok = bool(0.5 <= np.median(ratio) <= 2.0)

        passed = level_ok and (dyn_range < 3.0 or (0.7 <= coef[0] <= 1.3 and r2 >= 0.6))

        # intersection-union bootstrap over bins, so scarce data reads as low confidence
        # rather than as measured miscalibration
        rng_b = np.random.default_rng(self._seed + 7)
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
