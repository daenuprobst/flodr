import functools
import os
import re
from dataclasses import dataclass, field
from typing import Tuple

import numpy as np
import torch

from .data import find_ab
from .model import Flow

# deterministic GEMM workspace, needed before the first cuBLAS handle exists
os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")


def _read(path):
    try:
        with open(path) as f:
            return f.read().strip()
    except OSError:
        return ""


def _cpu_list(text):
    cpus = set()
    for part in filter(None, text.split(",")):
        if "-" in part:
            lo, hi = part.split("-")
            cpus.update(range(int(lo), int(hi) + 1))
        elif part.isdigit():
            cpus.add(int(part))
    return cpus


@functools.lru_cache(maxsize=1)
def default_device() -> str:
    return "cuda" if torch.cuda.is_available() else "cpu"


@functools.lru_cache(maxsize=8)
def native_bf16(device=None) -> bool:
    if str(device or default_device()).startswith("cuda"):
        return torch.cuda.is_available() and torch.cuda.is_bf16_supported()

    return bool(re.search(r"\b(avx512_bf16|amx_bf16)\b", _read("/proc/cpuinfo")))


@functools.lru_cache(maxsize=1)
def perf_cores() -> int:
    # Intel hybrid, P-cores only
    cpus = _cpu_list(_read("/sys/devices/cpu_core/cpus"))

    if not cpus:
        cpus = _cpu_list(_read("/sys/devices/system/cpu/online")) or set(
            range(os.cpu_count() or 1)
        )
    # collapse SMT siblings
    groups = {
        frozenset(
            _cpu_list(
                _read(f"/sys/devices/system/cpu/cpu{c}/topology/thread_siblings_list")
            )
            or {c}
        )
        for c in cpus
    }

    return max(1, len(groups) if groups else (os.cpu_count() or 2) // 2)


class Lion(torch.optim.Optimizer):
    def __init__(self, params, lr=1e-4, betas=(0.9, 0.99)):
        super().__init__(params, dict(lr=lr, betas=betas))

    @torch.no_grad()
    def step(self):
        for group in self.param_groups:
            b1, b2 = group["betas"]

            for param in group["params"]:
                if param.grad is None:
                    continue

                mom = self.state.setdefault(param, {}).setdefault(
                    "m", torch.zeros_like(param)
                )
                param.add_(
                    torch.sign(mom.mul(b1).add(param.grad, alpha=1 - b1)),
                    alpha=-group["lr"],
                )
                mom.mul_(b2 / b1 if b1 else b2).add_(param.grad, alpha=1 - b2)


def _ns5(G):
    a, b, c = 3.4445, -4.7750, 2.0315
    X = G / (G.norm() + 1e-7)
    transposed = X.shape[0] > X.shape[1]
    X = X.T if transposed else X

    for _ in range(5):
        A = X @ X.T
        X = a * X + (b * A + c * A @ A) @ X

    return X.T if transposed else X


class Muon(torch.optim.Optimizer):
    def __init__(self, params, lr=0.02, momentum=0.95, adam_ratio=0.05):
        super().__init__(params, dict(lr=lr, momentum=momentum, adam_ratio=adam_ratio))

    @torch.no_grad()
    def step(self):
        for group in self.param_groups:
            for param in group["params"]:
                if param.grad is None:
                    continue

                state = self.state.setdefault(param, {})
                mom = state.setdefault("m", torch.zeros_like(param))
                mom.mul_(group["momentum"]).add_(param.grad)

                if param.ndim == 2:
                    update = _ns5(mom) * max(
                        1.0, (param.shape[0] / param.shape[1]) ** 0.5
                    )
                    param.add_(update, alpha=-group["lr"])
                # 1D, RMS-normalized momentum step
                else:
                    var = state.setdefault("v", torch.zeros_like(param))
                    var.mul_(0.99).addcmul_(param.grad, param.grad, value=0.01)
                    param.addcdiv_(
                        mom,
                        var.sqrt().add_(1e-8),
                        value=-group["lr"] * group["adam_ratio"] / 0.05 * 0.05,
                    )


# Largest edge batch the auto mode will use. Below this the graph is taken whole.
#
# Measured across all seven paper data sets against the previous edge_batch=16384 with
# iters=6400, it's 3.6-4.4x faster on the four benchmark sets at flat quality, and +14-21%
# recall with CPD also up on the three atlases. The old constant was set to fix a GPU
# OOM at 500k points and became a global default. It over-trains small data with tiny
# steps and starves large data. What matters is passes over the edge set,
# iters * edge_batch / n_edges, which the old setting left at ~700 for n=5000 (saturates
# near 350) and ~34 for n=100k.
EDGE_BATCH_CAP = 262_144


@dataclass
class TrainConfig:
    iters: int = 700
    n_layers: int = 4
    hid: int = 256
    lr: float = 1e-3
    # negative samples per edge
    neg: int = 12
    # output kernel min_dist, smaller packs tighter
    min_dist: float = 0.01
    w_rep: float = 5.0
    w_recon: float = 2.0
    # mid-near attraction
    w_global: float = 0.6
    w_stress: float = 0.0
    # 1 = ordinal hinge on ranks, 0 = log-ratio stress
    stress_ordinal: int = 1
    # "euclid" | "jaccard" (needs stress_X in {0, 1})
    stress_metric: str = "euclid"
    w_nll: float = 0.5
    # chart consistency on kNN edges, ||f^-1(y_j, r_i) - x_j||^2
    w_chart: float = 0.0
    # encoder-Jacobian validity, keep 0, it collapses recall
    w_jac: float = 0.0
    # "vec" | "lognorm" (size only) | "centered"
    jac_mode: str = "vec"
    # kNN edges per iteration for the Jacobian term
    jac_edges: int = 4096
    # fraction of kNN edges permanently held out
    jac_hold: float = 0.0
    # keep recon from reshaping the layout
    recon_detach: bool = True
    # ramp recon/nll/global in over this fraction of iters
    warmup_frac: float = 0.3
    # rows per iter for recon + NLL, 0 = full batch
    aux_batch: int = 512
    # trailing fraction of iters run full-batch
    aux_full_frac: float = 0.2
    # kNN edges per iter. 0 = full batch, -1 = auto (see EDGE_BATCH_CAP): full batch
    # while the graph fits, capped above that. No cudagraphs when set.
    edge_batch: int = -1
    # couplings of the post-hoc conditional density tail
    cond_tail: int = 2
    cond_iters: int = 1500
    # GMM base components, only when cond_tail=0
    gmm_base: int = 20
    # >0 restricts the sketch to the first N dims (the PCA head)
    sketch_head: int = 0
    # fixed k-dim sketch conditioner (0 = off)
    sketch: int = 0
    # warm-start iters regressing dims 0-1 onto y_init
    init_iters: int = 0
    # the GPU where there is one
    device: str = field(default_factory=default_device)
    # None = auto, cuda and n*D >= 5e6. Not bitwise-eager
    cudagraphs: bool = None
    # checkpoint the recon-inverse pass, less memory, slower
    checkpoint: bool = False
    # recompute the conditioner MLP in backward. The (N, hid) activations it holds are
    # most of a fit's memory; exact, costs about a third more flow compute
    ckpt_net: bool = False
    # recompute the repulsive pointwise chain in backward. Exact, nearly free
    ckpt_neg: bool = False
    # embedding dimension
    k: int = 2
    # nested-dropout cuts
    cuts: Tuple[int, ...] = (2, 2, 2, 4, 8, 16, 24, 32)
    # torch CPU threads (0 = leave untouched)
    threads: int = 0
    # OneCycleLR at max_lr = 3*lr, pair with iters ~400
    onecycle: bool = False
    # torch.compile(step) on CPU, not for edge_batch
    cpu_compile: bool = False
    # bf16 matmuls while training, fp32 everywhere else
    bf16: bool = False
    # "adam" | "nadam" | "lion" (lr ~1e-4) | "muon" (lr ~0.02)
    optim: str = "adam"
    # "tanh" | "relu" | "silu"
    act: str = "tanh"
    # 1-hidden-layer conditioners
    shallow: bool = False
    # >0 bounds |log scale| per coupling
    gate_max: float = 0.0
    # >0 makes the bound per dimension: this on the display, gate_max on the residual
    gate_display: float = 0.0
    seed: int = 0


def raw_recipe(seed=0, threads=None, **overrides):
    dev = str(overrides.get("device") or default_device())
    if threads is None:
        threads = 0 if dev.startswith("cuda") else perf_cores()

    # 1600 iterations at lr 0.01 against the previous 800 at 0.02, holding
    # lr * iters = 16, which is the invariant that matters because Newton-Schulz makes
    # Muon's step magnitude-blind. Paired with edge_batch=-1 this is 3.6-4.4x faster on
    # the benchmark sets and +14-21% recall on the atlases; see EDGE_BATCH_CAP.
    kw = dict(
        seed=seed,
        threads=threads,
        sketch=64,
        onecycle=True,
        iters=1600,
        aux_full_frac=0.05,
        w_nll=0.0,
        cond_tail=0,
        gmm_base=0,
        # 17x penalty if bf16 is emulated rather than native
        bf16=native_bf16(dev),
        optim="muon",
        lr=0.01,
        # the scale bound is per dimension: 1.0 on the display, 0.5 on the residual
        gate_max=0.5,
        gate_display=1.0,
        # 0.3 spends a third of a 1600-iteration run ramping, which at this schedule
        # is most of the useful training
        warmup_frac=0.05,
        # 48 rather than the 12 of TrainConfig's own default. Measured under this
        # schedule against neg=12, the four benchmark sets lose about 1% recall, but
        # the atlases lose 13-20% (plant 0.0799 to 0.0643, bone marrow 0.1728 to
        # 0.1501). Same story as the schedule itself (at n=5000 the fit is past
        # saturation and the extra negatives are wasted, at n=100k it is starved and
        # they convert straight into neighbour retention). The benchmark cost is 20-27%
        # wall clock on fits that already take under 30 s, which is the cheaper side of
        # the trade.
        neg=48,
        w_rep=15.0,
        # raw mode defers recon anyway
        w_recon=0.0,
    )
    kw.update(overrides)

    return TrainConfig(**kw)


def train_flodr(
    Xp,
    ei,
    ej,
    cfg: TrainConfig,
    rng,
    return_model=False,
    w=None,
    stress_X=None,
    y_init=None,
    progress=False,
):
    n, D = Xp.shape
    k = cfg.k

    if cfg.threads:
        torch.set_num_threads(cfg.threads)
        torch.use_deterministic_algorithms(True, warn_only=True)

    torch.manual_seed(cfg.seed)
    dev = torch.device(cfg.device)
    # output kernel 1 / (1 + a d^(2b))
    a, b = find_ab(cfg.min_dist)
    x_t = torch.from_numpy(Xp).float().to(dev)

    edge_i = torch.from_numpy(ei).to(dev)
    edge_j = torch.from_numpy(ej).to(dev)

    w_attr = torch.from_numpy(np.asarray(w)).float().to(dev) if w is not None else None
    dim_idx = torch.arange(D, device=dev)

    # floored at k, because a cut below the embedding would zero part of the map
    cuts = tuple(sorted({min(max(c, k), D) for c in cfg.cuts} | {D}))

    # class-level so it also reaches couplings built outside Flow (the density tail)
    from .model import Coupling as _Coupling

    _Coupling.ckpt_net = bool(cfg.ckpt_net)

    flow = Flow(
        D,
        cfg.n_layers,
        cfg.hid,
        sketch=cfg.sketch,
        act=cfg.act,
        sketch_head=cfg.sketch_head,
        gate_max=cfg.gate_max,
        gate_display=cfg.gate_display,
        shallow=cfg.shallow,
        k=k,
    ).to(dev)

    if y_init is not None and cfg.init_iters:
        y_init_t = torch.from_numpy(np.asarray(y_init)).float().to(dev)

        opt_init = torch.optim.Adam(
            flow.parameters(), lr=1e-3, fused=dev.type == "cuda"
        )

        gen_init = torch.Generator(device=dev).manual_seed(cfg.seed + 7)

        for _ in range(cfg.init_iters):
            rows = torch.randint(0, n, (min(2048, n),), device=dev, generator=gen_init)
            loss_ws = ((flow(x_t[rows])[:, :k] - y_init_t[rows]) ** 2).mean()
            opt_init.zero_grad(set_to_none=True)
            loss_ws.backward()
            opt_init.step()

    opt = {
        "adam": lambda: torch.optim.Adam(
            flow.parameters(), lr=cfg.lr, fused=dev.type == "cuda"
        ),
        "nadam": lambda: torch.optim.NAdam(flow.parameters(), lr=cfg.lr),
        "lion": lambda: Lion(flow.parameters(), lr=cfg.lr),
        "muon": lambda: Muon(flow.parameters(), lr=cfg.lr),
    }[cfg.optim]()

    n_edges = edge_i.shape[0]

    x_stress = (
        torch.from_numpy(stress_X).float().to(dev) if stress_X is not None else x_t
    )

    if cfg.stress_metric not in ("euclid", "jaccard"):
        raise ValueError(
            f"stress_metric must be euclid|jaccard, got {cfg.stress_metric!r}"
        )

    if cfg.w_stress and cfg.stress_metric == "jaccard":
        vals = torch.unique(x_stress)

        if vals.numel() > 2 or not torch.all((vals == 0) | (vals == 1)):
            raise ValueError("stress_metric='jaccard' needs binary stress_X")

        # |A|, precomputed once
        row_sums = x_stress.sum(1)

    def _lh(row_a, row_b):
        if cfg.stress_metric == "jaccard":
            inter = (x_stress[row_a] * x_stress[row_b]).sum(1)
            union = (row_sums[row_a] + row_sums[row_b] - inter).clamp_min(1.0)
            return torch.log((1.0 - inter / union).clamp_min(1e-6))

        return 0.5 * torch.log(
            ((x_stress[row_a] - x_stress[row_b]) ** 2).sum(1).clamp_min(1e-12)
        )

    aux_n = n if cfg.aux_batch <= 0 else min(cfg.aux_batch, n)

    jac_rows = None

    if cfg.w_jac:
        gen_hold = torch.Generator(device="cpu")
        gen_hold.manual_seed(cfg.seed + 90210)
        held = torch.rand(n_edges, generator=gen_hold) < cfg.jac_hold
        jac_pool = torch.nonzero(~held).squeeze(1).to(dev)
        jac_held = torch.nonzero(held).squeeze(1).to(dev)

        if cfg.jac_mode == "centered":
            ei_np = edge_i.detach().cpu().numpy().astype(np.int64)
            keep_np = (~held).numpy()
            order = np.argsort(ei_np, kind="stable")
            order = order[keep_np[order]]
            src = ei_np[order]
            counts = np.bincount(src, minlength=n)
            kmax = int(max(counts.max(), 2))
            pad = np.zeros((n, kmax), np.int64)
            mask = np.zeros((n, kmax), np.float32)
            pos = np.arange(len(src)) - np.repeat(np.cumsum(counts) - counts, counts)
            pad[src, pos] = order
            mask[src, pos] = 1.0
            jac_rows = (
                torch.as_tensor(pad, device=dev),
                torch.as_tensor(mask, device=dev),
            )
    else:
        jac_pool = jac_held = None

    def _jac_terms(jac_sel):
        pair_i, pair_j = edge_i[jac_sel].long(), edge_j[jac_sel].long()

        x_i, x_j = x_t[pair_i], x_t[pair_j]
        y_i, jv = torch.func.jvp(
            lambda u: flow.forward_main(u)[:, :k], (x_i,), (x_j - x_i,)
        )

        return jv, y_i, pair_i, pair_j

    def step(
        neg_a,
        neg_b,
        glob_a,
        glob_b,
        rows,
        chart_e,
        keep,
        ramp,
        str_a,
        str_b,
        str_c,
        str_d,
        jac_sel,
    ):
        Z, logdet = flow.forward_with_logdet(x_t)
        Y = Z[:, :k]

        d2_edge = ((Y[edge_i] - Y[edge_j]) ** 2).sum(1)
        w_edge = 1.0 / (1.0 + a * (d2_edge + 1e-6) ** b)

        if w_attr is None:
            L_attr = -torch.log(w_edge + 1e-6).mean()
        else:
            L_attr = -(w_attr * torch.log(w_edge + 1e-6)).sum() / w_attr.sum()

        d2_neg = ((Y[neg_a] - Y[neg_b]) ** 2).sum(1)
        w_neg = 1.0 / (1.0 + a * (d2_neg + 1e-6) ** b)
        neg_loss = -torch.log(1.0 - w_neg + 1e-6)
        L_rep = neg_loss.mean()
        L_local = L_attr + cfg.w_rep * L_rep
        d2_glob = ((Y[glob_a] - Y[glob_b]) ** 2).sum(1)
        L_glob = (d2_glob / (1.0 + d2_glob)).mean()

        if cfg.w_stress:
            mask_a = (((str_a + str_b) % 5) != 0).float()
            ldist_a = 0.5 * torch.log(
                ((Y[str_a] - Y[str_b]) ** 2).sum(1).clamp_min(1e-12)
            )
            mdist_a = _lh(str_a, str_b)

            if cfg.stress_ordinal:
                mask_b = (((str_c + str_d) % 5) != 0).float()
                ldist_b = 0.5 * torch.log(
                    ((Y[str_c] - Y[str_d]) ** 2).sum(1).clamp_min(1e-12)
                )
                mdist_b = _lh(str_c, str_d)
                # +1 if pair a truly nearer
                sign = torch.sign(mdist_b - mdist_a)
                hinge = torch.relu(sign * (ldist_a - ldist_b) + 0.1)
                mask = mask_a * mask_b
                L_stress = (mask * hinge).sum() / mask.sum().clamp_min(1.0)
            else:
                resid = ldist_a - mdist_a
                res_mean = (mask_a * resid).sum() / mask_a.sum().clamp_min(1.0)
                L_stress = (
                    mask_a * (resid - res_mean) ** 2
                ).sum() / mask_a.sum().clamp_min(1.0)
        else:
            L_stress = 0.0

        rows_ = rows
        z_sub, logdet_s = Z[rows_], logdet[rows_]

        # the inverse pass is 25-35% of the step, so gate it
        if cfg.w_recon:
            y_head = z_sub[:, :k].detach() if cfg.recon_detach else z_sub[:, :k]
            z_trunc = torch.cat([y_head, z_sub[:, k:]], 1) * keep

            if cfg.checkpoint:
                x_rec = z_trunc

                for c in reversed(flow.layers):
                    x_rec = torch.utils.checkpoint.checkpoint(
                        c.inverse, x_rec, use_reentrant=False
                    )
            else:
                x_rec = flow.inverse(z_trunc)

            L_recon = ((x_rec - x_t[rows_]) ** 2).mean()
        else:
            L_recon = 0.0

        L_nll = -(flow.base_log_prob(z_sub) + logdet_s).mean() / D if cfg.w_nll else 0.0

        if cfg.w_chart:
            src_i, dst_j = edge_i[chart_e], edge_j[chart_e]
            z_mix = torch.cat([Z[dst_j, :k], Z[src_i, k:]], 1)
            L_chart = ((flow.inverse(z_mix) - x_t[dst_j]) ** 2).mean()
        else:
            L_chart = 0.0

        if cfg.w_jac:
            with torch.autocast(dev.type, enabled=False):
                if cfg.jac_mode == "centered":
                    pad_rows, mask_rows = jac_rows
                    # (M, kmax)
                    pad_sel, mask_sel = (
                        pad_rows[jac_sel],
                        mask_rows[jac_sel],
                    )
                    jv, y_i, pair_i, pair_j = _jac_terms(pad_sel.reshape(-1))
                    dy = Z[pair_j, :k] - y_i
                    resid = (
                        0.5 * torch.log((jv**2).sum(1).clamp_min(1e-12))
                        - 0.5 * torch.log((dy**2).sum(1).clamp_min(1e-12))
                    ).reshape(pad_sel.shape)
                    counts = mask_sel.sum(1, keepdim=True).clamp_min(1.0)
                    res_mean = (resid * mask_sel).sum(1, keepdim=True) / counts
                    L_jac = (
                        (resid - res_mean) ** 2 * mask_sel
                    ).sum() / mask_sel.sum().clamp_min(1.0)
                    return L_local + ramp * (
                        cfg.w_recon * L_recon
                        + cfg.w_global * L_glob
                        + cfg.w_nll * L_nll
                        + cfg.w_chart * L_chart
                        + cfg.w_stress * L_stress
                        + cfg.w_jac * L_jac
                    )
                jv, y_i, pair_i, pair_j = _jac_terms(jac_sel)
                dy = Z[pair_j, :k] - y_i

                if cfg.jac_mode == "lognorm":
                    L_jac = (
                        0.5 * torch.log((jv**2).sum(1).clamp_min(1e-12))
                        - 0.5 * torch.log((dy**2).sum(1).clamp_min(1e-12))
                    ) ** 2
                    L_jac = L_jac.mean()
                # "vec" is the relative vector error
                else:
                    L_jac = (
                        ((jv - dy) ** 2).sum(1) / (dy**2).sum(1).clamp_min(1e-12)
                    ).mean()
        else:
            L_jac = 0.0
        return L_local + ramp * (
            cfg.w_recon * L_recon
            + cfg.w_global * L_glob
            + cfg.w_nll * L_nll
            + cfg.w_chart * L_chart
            + cfg.w_stress * L_stress
            + cfg.w_jac * L_jac
        )

    # logdet feeds only L_nll, and z_sub/rows feed only L_nll and L_recon. The raw
    # recipe leaves both weights at zero, so on that path the log-determinant
    # accumulation and its backward are computed and then multiplied by nothing.
    needs_logdet = bool(cfg.w_nll or cfg.w_recon)

    def step_edge_batch(edge_idx, keep, ramp):
        pts = torch.cat([edge_i[edge_idx], edge_j[edge_idx]])
        uniq, inv = torch.unique(pts, return_inverse=True)

        n_uniq, n_edge = uniq.shape[0], edge_idx.shape[0]
        # gather once (was doubled for recon)
        x_uniq = x_t[uniq]

        if needs_logdet:
            Z, logdet = flow.forward_with_logdet(x_uniq)
        else:
            Z, logdet = flow.forward_main(x_uniq), None

        Y = Z[:, :k]

        d2_edge = ((Y[inv[:n_edge]] - Y[inv[n_edge:]]) ** 2).sum(1)
        w_edge = 1.0 / (1.0 + a * (d2_edge + 1e-6) ** b)

        if w_attr is None:
            L_attr = -torch.log(w_edge + 1e-6).mean()
        else:
            # gathered once, not twice
            wa = w_attr[edge_idx]
            L_attr = -(wa * torch.log(w_edge + 1e-6)).sum() / wa.sum()

        # int32 throughout: n is far below 2^31 and the full-batch path's _ri helper
        # already draws int32 and indexes Y with it, so this is the same trick applied
        # to the branch that was left out. Halves index traffic and the sort key width
        # in the scatter backward.
        neg_a = torch.randint(
            0, n_uniq, (cfg.neg * n_edge,), device=dev, dtype=torch.int32, generator=gen
        )
        neg_b = torch.randint(
            0, n_uniq, (cfg.neg * n_edge,), device=dev, dtype=torch.int32, generator=gen
        )

        # The repulsive block is a long pointwise chain over neg*edge_batch pairs, and
        # every link of it is retained for backward. At neg=48, edge_batch=262144 that
        # is ~12.6M elements a dozen times over. Recomputing it in backward costs one
        # extra pass over cheap 2D arithmetic and holds only the two index tensors.
        # Exact: the indices are drawn outside, so the recompute sees the same input.
        def _rep(Yv, ia, ib):
            d2 = ((Yv[ia] - Yv[ib]) ** 2).sum(1)

            return -torch.log(1.0 - 1.0 / (1.0 + a * (d2 + 1e-6) ** b) + 1e-6).mean()

        if cfg.ckpt_neg and torch.is_grad_enabled():
            L_rep = torch.utils.checkpoint.checkpoint(
                _rep, Y, neg_a, neg_b, use_reentrant=False
            )
        else:
            L_rep = _rep(Y, neg_a, neg_b)

        glob_a = torch.randint(
            0, n_uniq, (3 * n_uniq,), device=dev, dtype=torch.int32, generator=gen
        )

        glob_b = torch.randint(
            0, n_uniq, (3 * n_uniq,), device=dev, dtype=torch.int32, generator=gen
        )

        d2_glob = ((Y[glob_a] - Y[glob_b]) ** 2).sum(1)
        L_glob = (d2_glob / (1.0 + d2_glob)).mean()

        if cfg.w_stress:
            n_pairs = 4 * n_uniq
            idx_a = torch.randint(
                0, n_uniq, (n_pairs,), device=dev, dtype=torch.int32, generator=gen
            )
            idx_b = torch.randint(
                0, n_uniq, (n_pairs,), device=dev, dtype=torch.int32, generator=gen
            )
            row_a, row_b = uniq[idx_a], uniq[idx_b]
            mask_a = (((row_a + row_b) % 5) != 0).float()

            ldist_a = 0.5 * torch.log(
                ((Y[idx_a] - Y[idx_b]) ** 2).sum(1).clamp_min(1e-12)
            )

            mdist_a = _lh(row_a, row_b)

            if cfg.stress_ordinal:
                idx_c = torch.randint(
                    0, n_uniq, (n_pairs,), device=dev, dtype=torch.int32, generator=gen
                )

                idx_d = torch.randint(
                    0, n_uniq, (n_pairs,), device=dev, dtype=torch.int32, generator=gen
                )

                row_c, row_d = uniq[idx_c], uniq[idx_d]
                mask_b = (((row_c + row_d) % 5) != 0).float()
                ldist_b = 0.5 * torch.log(
                    ((Y[idx_c] - Y[idx_d]) ** 2).sum(1).clamp_min(1e-12)
                )
                mdist_b = _lh(row_c, row_d)
                sign = torch.sign(mdist_b - mdist_a)
                hinge = torch.relu(sign * (ldist_a - ldist_b) + 0.1)
                mask = mask_a * mask_b
                L_stress = (mask * hinge).sum() / mask.sum().clamp_min(1.0)
            else:
                resid = ldist_a - mdist_a
                res_mean = (mask_a * resid).sum() / mask_a.sum().clamp_min(1.0)
                L_stress = (
                    mask_a * (resid - res_mean) ** 2
                ).sum() / mask_a.sum().clamp_min(1.0)
        else:
            L_stress = 0.0

        # rows, z_sub and logdet_s feed only L_recon and L_nll. With both weights at
        # zero, as the raw recipe leaves them, this whole block is gathered and then
        # discarded.

        # The draw stays unconditional even when nothing consumes it. It is one cheap
        # randint, and skipping it would advance the generator differently and change
        # every subsequent sample. This is ar eseed, which quietly moves results without
        # improving anything. Only the gather it feeds is skipped.
        rows = torch.randint(
            0, n_uniq, (min(aux_n, n_uniq),), device=dev, generator=gen
        )
        z_sub, logdet_s = (Z[rows], logdet[rows]) if needs_logdet else (None, None)

        if cfg.w_recon:
            y_head = z_sub[:, :k].detach() if cfg.recon_detach else z_sub[:, :k]
            z_trunc = torch.cat([y_head, z_sub[:, k:]], 1) * keep

            if cfg.checkpoint:
                x_rec = z_trunc

                for c in reversed(flow.layers):
                    x_rec = torch.utils.checkpoint.checkpoint(
                        c.inverse, x_rec, use_reentrant=False
                    )
            else:
                x_rec = flow.inverse(z_trunc)

            L_recon = ((x_rec - x_uniq[rows]) ** 2).mean()
        else:
            L_recon = 0.0

        L_nll = -(flow.base_log_prob(z_sub) + logdet_s).mean() / D if cfg.w_nll else 0.0

        if cfg.w_chart:
            chart_e = torch.randint(
                0, n_edge, (min(aux_n, n_edge),), device=dev, generator=gen
            )
            z_mix = torch.cat(
                [Z[inv[n_edge:][chart_e], :k], Z[inv[:n_edge][chart_e], k:]], 1
            )
            L_chart = ((flow.inverse(z_mix) - x_t[pts[n_edge:][chart_e]]) ** 2).mean()
        else:
            L_chart = 0.0

        return (
            L_attr
            + cfg.w_rep * L_rep
            + ramp
            * (
                cfg.w_recon * L_recon
                + cfg.w_global * L_glob
                + cfg.w_nll * L_nll
                + cfg.w_chart * L_chart
                + cfg.w_stress * L_stress
            )
        )

    use_graphs = cfg.cudagraphs

    if use_graphs is None:
        use_graphs = dev.type == "cuda" and n * D >= 5_000_000 and not cfg.edge_batch

    compiled = use_graphs or (cfg.cpu_compile and not cfg.edge_batch)

    if use_graphs:
        torch.use_deterministic_algorithms(True, warn_only=True)
        # compile once per shape, on disk
        torch._inductor.config.fx_graph_cache = True
        step = torch.compile(step, mode="reduce-overhead", fullgraph=True)
    elif compiled:
        step = torch.compile(step, dynamic=False)

    if compiled:
        # ramp enters the graph as a tensor, a fresh python float would recompile every iter
        ramps = torch.tensor(
            [
                (
                    1.0
                    if cfg.warmup_frac <= 0
                    else min(1.0, (i + 1) / (cfg.warmup_frac * cfg.iters))
                )
                for i in range(cfg.iters)
            ],
            dtype=torch.float32,
            device=dev,
        )

    sched = (
        torch.optim.lr_scheduler.OneCycleLR(
            opt, max_lr=3 * cfg.lr, total_steps=cfg.iters
        )
        if cfg.onecycle
        else None
    )

    gen = torch.Generator(device=dev)
    gen.manual_seed(cfg.seed)

    # device-side draws, no H2D copy on the critical path
    def _ri(size):
        return torch.randint(
            0, n, (size,), device=dev, dtype=torch.int32, generator=gen
        )

    if cfg.edge_batch < 0:
        # full batch while the graph fits, capped above that
        eb_size = 0 if n_edges <= EDGE_BATCH_CAP else EDGE_BATCH_CAP
    else:
        eb_size = min(cfg.edge_batch, n_edges) if cfg.edge_batch else 0

    steps = range(cfg.iters)

    if progress:
        from tqdm.auto import tqdm

        steps = tqdm(steps, desc=f"flodr n={n}", unit="it", leave=False, disable=None)

    for it in steps:
        ramp = (
            1.0
            if cfg.warmup_frac <= 0
            else min(1.0, (it + 1) / (cfg.warmup_frac * cfg.iters))
        )

        # keep feeds only the recon branch; skip the host-side draw when it is off
        if cfg.w_recon:
            keep = (dim_idx < int(rng.choice(cuts))).to(torch.float32)
        else:
            keep = None

        if eb_size:
            edge_idx = torch.randint(
                0, n_edges, (eb_size,), device=dev, dtype=torch.int32, generator=gen
            )
            loss = step_edge_batch(edge_idx, keep, ramp)
        else:
            neg_a, neg_b = _ri(cfg.neg * n_edges), _ri(cfg.neg * n_edges)
            glob_a, glob_b = _ri(3 * n), _ri(3 * n)

            # full batch for the last aux_full_frac, once the layout has settled
            if it >= (1.0 - cfg.aux_full_frac) * cfg.iters:
                rows = torch.arange(n, device=dev)
            else:
                rows = _ri(aux_n)

            # drawn only when used, so w_chart=0 leaves the RNG stream untouched
            chart_e = (
                torch.randint(
                    0, n_edges, (min(aux_n, n_edges),), device=dev, generator=gen
                )
                if cfg.w_chart
                else neg_a[:1]
            )

            # likewise
            if cfg.w_stress:
                str_a, str_b = _ri(4 * n), _ri(4 * n)
                str_c, str_d = (
                    (_ri(4 * n), _ri(4 * n))
                    if (cfg.w_stress and cfg.stress_ordinal)
                    else (str_a[:1], str_a[:1])
                )
            else:
                str_a = str_b = str_c = str_d = neg_a[:1]

            if cfg.w_jac and cfg.jac_mode == "centered":
                # here jac_edges is a point budget, edges ~ jac_edges * mean degree
                jac_sel = torch.randint(
                    0, n, (max(1, cfg.jac_edges // 12),), device=dev, generator=gen
                )
            elif cfg.w_jac:
                jac_sel = jac_pool[
                    torch.randint(
                        0,
                        jac_pool.shape[0],
                        (min(cfg.jac_edges, jac_pool.shape[0]),),
                        device=dev,
                        generator=gen,
                    )
                ]
            else:
                jac_sel = neg_a[:1]

            with torch.autocast(dev.type, dtype=torch.bfloat16, enabled=cfg.bf16):
                loss = step(
                    neg_a,
                    neg_b,
                    glob_a,
                    glob_b,
                    rows,
                    chart_e,
                    keep,
                    ramps[it] if compiled else ramp,
                    str_a,
                    str_b,
                    str_c,
                    str_d,
                    jac_sel,
                )

        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(flow.parameters(), 5.0)
        opt.step()

        if sched is not None:
            sched.step()

    if cfg.cond_tail:
        # deferred to the first density use, or to an explicit fit_density()
        flow._density_args = (x_t, cfg.cond_tail, cfg.cond_iters, cfg.seed)
    elif cfg.gmm_base:
        flow.fit_gmm_base(x_t, cfg.gmm_base, seed=cfg.seed)

    with torch.no_grad():
        Z = flow(x_t)
        Y = Z[:, :k].cpu().numpy()
        roundtrip = float(np.abs(flow.inverse(Z).cpu().numpy() - Xp).max())
        scale = np.sqrt((Xp**2).sum(1)).mean()
        # truncation lives on the main latent, the density tail re-coordinates dims 2 onward
        z_main = flow.forward_main(x_t)
        recon = {}

        # the embedding, plus a deeper cut
        for cut_d in sorted({k, min(16, D)}):
            z_cut = z_main.clone()
            z_cut[:, cut_d:] = 0.0
            recon[cut_d] = float(
                np.sqrt(
                    ((flow.inverse_main(z_cut).cpu().numpy() - Xp) ** 2).sum(1)
                ).mean()
                / scale
            )

    return (Y, roundtrip, recon, flow) if return_model else (Y, roundtrip, recon)
