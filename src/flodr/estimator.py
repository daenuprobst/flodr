import copy
import dataclasses
import numbers

import numpy as np
import torch
from sklearn.base import (
    BaseEstimator,
    ClassNamePrefixFeaturesOutMixin,
    TransformerMixin,
)
from sklearn.utils import check_array, check_random_state
from sklearn.utils.metaestimators import available_if
from sklearn.utils.validation import check_is_fitted, validate_data

from .data import find_ab, fuzzy_knn_graph, knn_query, raw_coords
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

    def fit(self, X, y=None):
        max_iter, verbose = self.max_iter, self.verbose
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

    def fit_transform(self, X, y=None):
        return self.fit(X).embedding_

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
