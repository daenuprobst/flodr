import copy
import dataclasses
import numbers
import warnings

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

from . import fidelity
from .data import (
    RotatedWhitening,
    Whitening,
    find_ab,
    fuzzy_knn_graph,
    knn_query,
    raw_coords,
)
from .heat import heat_coordinates, heat_rotation
from .model import ENERGY_GRID
from .train import TrainConfig, default_device, raw_recipe, train_flodr

# in this size range the heaviest quarter of the fuzzy graph keeps more neighbours
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


def _pushed_sd(coords_of, X, sigma):
    # per-axis sd of isotropic input noise after coords_of, from the map's Jacobian by central
    # differences at the data mean; exact for a linear map
    x0 = np.asarray(X, np.float64).mean(0)
    step = sigma * np.sqrt(len(x0))
    up = (x0 + step * np.eye(len(x0))).astype(np.float32)
    down = (x0 - step * np.eye(len(x0))).astype(np.float32)
    # the steps as the float32 inputs carry them
    width = np.diag(up.astype(np.float64) - down.astype(np.float64))
    J = np.asarray(coords_of(up), np.float64) - np.asarray(coords_of(down), np.float64)
    J /= width[:, None]

    return sigma * np.sqrt((J**2).sum(0))


def _axis_sd(Z, chunk=65536):
    # float64 for axes with an sd near 5e-7, in chunks to bound memory
    mu = np.zeros(Z.shape[1])

    for i in range(0, len(Z), chunk):
        mu += Z[i : i + chunk].sum(0, dtype=np.float64)

    mu /= len(Z)
    ss = np.zeros(Z.shape[1])

    for i in range(0, len(Z), chunk):
        ss += ((Z[i : i + chunk] - mu) ** 2).sum(0)

    return np.sqrt(ss / len(Z))


def _has_density(est):
    # hides score and score_samples from hasattr without a density, as sklearn expects
    if not est.density:
        raise AttributeError(
            "score and score_samples need the density; refit with FloDR(density=True)"
        )

    return True


class FloDR(ClassNamePrefixFeaturesOutMixin, TransformerMixin, BaseEstimator):
    # fixed parts of the recipe, not user settings; farthest-point landmarks cover rare branches
    _FROZEN = dict(
        n_neighbors=15,
        pca_head=50,
        stress_ordinal=True,
        density_noise_cap=0.5,
        heat_landmarks="farthest",
    )

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
        axes="variance",
    ):
        self.n_components = n_components
        self.w = w
        self.density = density
        self.max_iter = max_iter
        self.random_state = random_state
        self.device = device
        self.verbose = verbose
        self.advanced = advanced
        self.axes = axes

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

    # a model fitted on CUDA moves to the GPU while it is unpickled, when there is one
    def __setstate__(self, state):
        super().__setstate__(state)

        cuda = str(self._device).startswith("cuda") if "flow_" in state else False

        if cuda and torch.cuda.is_available():
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

        if self.axes not in ("variance", "heat"):
            raise ValueError(f"axes must be 'variance' or 'heat', got {self.axes!r}")

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

        select = (self.advanced or {}).get("tail_select", "auto")

        if select not in ("auto", "energy", "coverage"):
            raise ValueError(
                f"tail_select must be 'auto', 'energy' or 'coverage', got {select!r}"
            )

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
            # the density is fitted after the layout: a likelihood term in the layout fit
            # flattens the training cells' residual without improving the layout
            kw.setdefault("w_nll", 0.0)
            kw.setdefault("cond_tail", 2)

        kw.setdefault("k", self.n_components)
        return raw_recipe(seed=self._seed, device=self._device, **kw)

    def fit(self, X, y=None):
        self._check_params(self.max_iter)
        X = validate_data(
            self, X, dtype=np.float32, ensure_min_samples=2, ensure_min_features=2
        )
        self._seed = self._seed_from()
        self._device = default_device() if self.device is None else self.device

        # below 16 points every other point is a neighbour
        self._k = min(self._FROZEN["n_neighbors"], len(X) - 1)
        self._cfg = self._config(X, self.max_iter)
        Z, self._raw_of, self._coords_of = raw_coords(
            X, self._FROZEN["pca_head"], self._seed
        )

        # noise columns widen the residual of narrow inputs; new points get zeros there and
        # _to_raw drops them
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
        # gpu_exact, faiss_exact, usearch_exact, hnsw (approximate) or jaccard_exact
        info = dict(knn="jaccard_exact" if pre is not None else "unrecorded")
        self.knn_idx_, edge_i, edge_j, w_attr, self.knn_dist_ = fuzzy_knn_graph(
            X,
            self._k,
            precomputed=pre,
            return_dist=True,
            device=self._device,
            info=info,
        )
        self.knn_backend_ = info["knn"]

        # axes="heat" starts the flow from the heat geometry of the graph instead of the leading
        # principal components; the layout keeps the arrangement it starts from. The rotation is
        # orthogonal and undone on the way back, so the inverse stays exact
        wh = getattr(self._raw_of, "__self__", None)

        if self.axes == "heat" and isinstance(wh, Whitening):
            P = heat_coordinates(
                self.knn_dist_,
                self.knn_idx_,
                self.n_components,
                self._device,
                self._seed,
                self._FROZEN["heat_landmarks"],
                X,
            )
            R = heat_rotation(Z[:, : wh.n_head], P)
            Z[:, : wh.n_head] = Z[:, : wh.n_head] @ R
            rot = RotatedWhitening(wh, R)
            self._raw_of, self._coords_of = rot.to_raw, rot.to_coords

        keep = prune_fraction(len(X))

        if keep < 1.0:
            sel = np.sort(np.argsort(-w_attr)[: round(keep * len(w_attr))])
            edge_i, edge_j, w_attr = edge_i[sel], edge_j[sel], w_attr[sel]

        # the density's input noise: isotropic in X, diagonal in the latent; none on the padding
        # columns or on binary inputs
        sd = None

        if self.density and self._cfg.density_noise and self._metric == "euclid":
            wh = getattr(self._raw_of, "__self__", None)
            sigma = self._cfg.density_noise * float(np.median(self.knn_dist_[:, 0]))
            sigma /= np.sqrt(X.shape[1])
            if isinstance(wh, Whitening):
                sd = np.concatenate(
                    [sigma / wh.head_sd, sigma / wh.tail_sd, np.zeros(self._n_pad)]
                )
            else:
                # a rotated whitening, so the noise goes through the map itself
                sd = np.concatenate(
                    [_pushed_sd(self._coords_of, X, sigma), np.zeros(self._n_pad)]
                )
            # at most half of each axis's own spread, so near-constant axes get no large noise
            sd = np.minimum(sd, self._FROZEN["density_noise_cap"] * _axis_sd(Z))

        # the linearity term steps at that noise
        lin = dict(noise_sd=sd) if sd is not None and self._cfg.w_lin else {}

        Y, roundtrip, _, flow = train_flodr(
            Z,
            edge_i,
            edge_j,
            self._cfg,
            np.random.default_rng(self._seed),
            return_model=True,
            w=w_attr,
            stress_X=X,
            progress=self.verbose,
            **lin,
        )

        self.embedding_, self.roundtrip_, self.flow_ = (
            np.asarray(Y),
            float(roundtrip),
            flow,
        )
        self.step_mode_ = getattr(flow, "_step_mode", "eager")
        bad = int((~np.isfinite(self.embedding_)).any(1).sum())

        # CUDA fits vary between runs, so a refit at the same seed can succeed
        if bad:
            warnings.warn(
                f"the fit returned non-finite coordinates for {bad} of {len(X)} points; "
                "refit, at the same or another random_state",
                RuntimeWarning,
                stacklevel=2,
            )

        self.n_iter_ = self._cfg.iters
        self._n_features_out = self.n_components

        self._coords = Z
        self._X = X

        self.density_sigma_ = 0.0

        if sd is not None:
            self.flow_._density_noise = torch.as_tensor(
                sd, dtype=torch.float32, device=next(self.flow_.parameters()).device
            )
            self.density_sigma_ = float(sigma)

        # on the flow, so a deferred density fit makes the same choice
        if self.density:
            self.flow_._tail_kw = self._tail_kw()

        return self

    def _tail_kw(self):
        cfg = self._cfg

        # the energy-selected tail only with the linearity term
        if cfg.tail_select == "coverage" or (
            cfg.tail_select == "auto" and not cfg.w_lin
        ):
            return dict(select="coverage")

        A, b = fidelity._raw_affine(self)

        return dict(
            select="energy",
            grid=ENERGY_GRID,
            raw=(A.astype(np.float32), b.astype(np.float32)),
        )

    def fit_transform(self, X, y=None):
        return self.fit(X).embedding_

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

    # decodes [y, 0]: the residual set to zero
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

        # log_prob runs the flow itself, so it takes the whitened coordinates
        dev = next(self.flow_.parameters()).device
        z = torch.from_numpy(self._to_coords(X)).float().to(dev)

        with torch.no_grad():
            if not self._n_pad:
                return self.flow_.log_prob(z).cpu().numpy()

            return self._pad_marginal(z, 128).cpu().numpy()

    # the density was fitted with N(0, 1) noise in the padding columns, so a point's density is
    # the marginal over them by importance sampling from that law:
    # log p(x) = logmeanexp_k [log p(x, e_k) - log phi(e_k)]. The draws are fixed by
    # random_state and shared by all points
    def _pad_marginal(self, z, n_draws):
        gen = torch.Generator().manual_seed(self._seed + 4244)
        e = torch.randn((n_draws, self._n_pad), generator=gen).to(z.device)
        log_phi = -0.5 * (e**2).sum(1) - 0.5 * self._n_pad * np.log(2 * np.pi)
        rows = max(1, 131072 // n_draws)
        out = []

        for s in range(0, len(z), rows):
            zc = z[s : s + rows]
            zz = zc[None].repeat(n_draws, 1, 1)
            zz[:, :, self._d_z :] = e[:, None, :]
            lp = self.flow_.log_prob(zz.reshape(-1, zz.shape[-1])).reshape(n_draws, -1)
            out.append(torch.logsumexp(lp - log_phi[:, None], 0) - np.log(n_draws))

        return torch.cat(out)

    # mean log density per sample, so GridSearchCV can select on held-out likelihood
    @available_if(_has_density)
    def score(self, X, y=None):
        return float(np.mean(self.score_samples(X)))

    def _need_density(self):
        if not self.density:
            raise ValueError("this needs the density; refit with FloDR(density=True)")

        if self.flow_._cond is None:
            # selected on held-out cells, since a fixed capacity memorises small samples
            self.flow_._density_args = None
            self.flow_.fit_cond_tail_cv(
                self._coords,
                seed=self._seed,
                noise_sd=self._density_sd(),
                **getattr(self.flow_, "_tail_kw", {}),
            )

    def _layout_bounds(self):
        return self.embedding_.min(0), self.embedding_.max(0)

    def _density_sd(self):
        return getattr(self.flow_, "_density_noise", None)

    # held-out cells are drawn from the density's own law: the cells plus one draw of its input noise
    def _noised(self, xc):
        sd = self._density_sd()

        if sd is None:
            return xc

        rng = np.random.default_rng(self._seed + 47)

        return (xc + sd.cpu().numpy() * rng.standard_normal(xc.shape)).astype(
            np.float32
        )

    def coverage_certificate(
        self, X=None, n_draws=199, margin=0.1, alpha=0.05, n_bins=14, min_pts=20
    ):
        check_is_fitted(self)
        self._need_density()
        from scipy.stats import beta

        # held-out cells ranked among exact draws of the fitted law: each cell's
        # l = log p(x) - log p(y) among draws from the fibre of its own pixel, and log p(y)
        # among draws of the map density. Both ranks are uniform when the law is right,
        # so the certificate is an equivalence band around the diagonal, not a p-value
        k = self.n_components
        rng = np.random.default_rng(self._seed + 41)

        if X is None:
            # a tail refitted on half the cells, certified on the other half
            perm = rng.permutation(len(self._coords))
            fold_a, fold_b = perm[: len(perm) // 2], perm[len(perm) // 2 :]
            flow = copy.deepcopy(self.flow_)
            flow._density_args = None
            flow.tail, flow._cond = torch.nn.ModuleList(), None
            flow.fit_cond_tail_cv(
                self._coords[fold_a],
                seed=self._seed,
                noise_sd=self._density_sd(),
                **getattr(flow, "_tail_kw", {}),
            )
            xc = np.asarray(self._coords[fold_b], np.float32)
        else:
            X = validate_data(self, X, dtype=np.float32, reset=False)
            flow, xc = self.flow_, np.asarray(self._to_coords(X), np.float32)

        xc = self._noised(xc)

        dev = next(flow.parameters()).device
        cond = flow.cond

        with torch.no_grad():
            xt = torch.as_tensor(xc, device=dev)
            y_cell = flow.forward_main(xt)[:, :k]
            lp_cell = cond.y_log_prob(y_cell).cpu().numpy()
            gen = torch.Generator(device=dev)
            gen.manual_seed(self._seed + 43)
            w = torch.softmax(cond.ylogw, 0)
            comp = torch.multinomial(w, 20000, replacement=True, generator=gen)
            y_draw = cond.ymu[comp] + torch.exp(0.5 * cond.ylogvar[comp]) * torch.randn(
                (20000, k), device=dev, generator=gen
            )
            lp_draw = np.sort(cond.y_log_prob(y_draw).cpu().numpy())

        u = flow.fibre_ranks(xc, n_draws=n_draws, seed=self._seed)
        below = np.searchsorted(lp_draw, lp_cell, side="left")
        ties = np.searchsorted(lp_draw, lp_cell, side="right") - below
        v = (below + rng.random(len(lp_cell)) * (ties + 1.0)) / (len(lp_draw) + 1.0)

        def band(r):
            n = len(r)
            srt = np.sort(r)
            ks = float(
                max(
                    (np.arange(1, n + 1) / n - srt).max(),
                    (srt - np.arange(n) / n).max(),
                )
            )
            eps = float(np.sqrt(np.log(2.0 / alpha) / (2.0 * n)))
            inside = int((r >= 0.1).sum())
            lo = beta.ppf(alpha / 2, inside, n - inside + 1) if inside else 0.0
            hi = beta.ppf(1 - alpha / 2, inside + 1, n - inside) if inside < n else 1.0

            if ks + eps <= margin:
                passed, certifies = True, "certified"
            elif ks - eps > margin:
                passed, certifies = False, "refused"
            else:
                passed, certifies = None, "undecided"

            return dict(
                ks=ks,
                band=eps,
                cov90=inside / n,
                cov90_ci=(float(lo), float(hi)),
                passed=passed,
                certifies=certifies,
            )

        # flags the map bins whose 90% fibre regions hold demonstrably fewer than 0.9 - margin
        # of the held-out cells
        Y = y_cell.cpu().numpy()
        lo_y, hi_y = self._layout_bounds()
        bix = np.clip(
            ((Y - lo_y) / (hi_y - lo_y + 1e-9) * n_bins).astype(int), 0, n_bins - 1
        )
        bin_id = bix[:, 0] * n_bins + bix[:, 1]
        regions, flagged = [], []

        for b in np.unique(bin_id):
            sel = bin_id == b
            n_b = int(sel.sum())

            if n_b < min_pts:
                continue

            inside = int((u[sel] >= 0.1).sum())
            hi = (
                beta.ppf(1 - alpha / 2, inside + 1, n_b - inside)
                if inside < n_b
                else 1.0
            )
            regions.append(
                dict(bin=int(b), n=n_b, cov90=inside / n_b, cov90_hi=float(hi))
            )

            if hi < 0.9 - margin:
                flagged.append(int(b))

        fibre, dmap = band(u), band(v)
        both = (fibre["passed"], dmap["passed"])

        # the share of cells less likely than every draw, with its one-sided Clopper-Pearson
        # bound and the nominal 1 / (M + 1); reported, not part of the verdict
        def tail(r, m):
            n, k_out = len(r), int((r < 1.0 / (m + 1)).sum())
            hi = beta.ppf(1 - alpha, k_out + 1, n - k_out) if k_out < n else 1.0
            return dict(share=k_out / n, upper=float(hi), nominal=1.0 / (m + 1), n=n)

        return dict(
            fibre=dict(ranks=u, **fibre),
            map=dict(ranks=v, **dmap),
            tail=dict(fibre=tail(u, n_draws), map=tail(v, len(lp_draw))),
            regions=regions,
            flagged=flagged,
            bin_id=bin_id,
            positions=Y,
            passed=False if False in both else (None if None in both else True),
            certifies=(
                "certified"
                if both == (True, True)
                else ("refused" if False in both else "undecided")
            ),
            margin=margin,
            n_cells=len(u),
            n_draws=n_draws,
            held_out="tail refitted on half the cells" if X is None else "new cells",
        )

    # ranks held-out cells by input distance to the draws, which also catches draws that are too
    # wide in input space
    def energy_certificate(
        self, X, n_draws=199, margin=0.1, alpha=0.05, precision_margin=0.02
    ):
        check_is_fitted(self)
        self._need_density()
        X = validate_data(self, X, dtype=np.float32, reset=False)

        return fidelity.energy_certificate(
            self, X, n_draws, margin, alpha, precision_margin
        )

    # where the decode is near-linear at the density's input noise
    def faithful_scale(
        self, X=None, h=(0.125, 0.25, 0.5, 1.0, 2.0), n_steps=4, threshold=0.25
    ):
        check_is_fitted(self)
        sd = self._density_sd()

        if sd is None:
            raise ValueError(
                "faithful_scale steps at the density's input noise, which this model does not "
                "have (density=False, density_noise=0 or a binary input)"
            )

        if X is None:
            Z = self._coords
        else:
            X = validate_data(self, X, dtype=np.float32, reset=False)
            Z = self._to_coords(X)

        return fidelity.faithful_scale(
            self, Z, sd, h, int(n_steps), threshold, self._seed + 61
        )
