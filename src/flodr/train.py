import functools
import math
import os
import re
import warnings
from dataclasses import dataclass, field
from typing import Tuple

import numpy as np
import torch

from .data import find_ab
from .model import Coupling, Flow, FourierShift, deterministic

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
    # P-cores only on Intel hybrid CPUs
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


def _cgroup_cpus():
    # cgroup v2 cpu.max holds "quota period", or "max period" without a limit
    try:
        quota, period = _read("/sys/fs/cgroup/cpu.max").split()[:2]
        return max(1, math.ceil(int(quota) / int(period)))
    except ValueError:
        return None


def default_threads():
    counts = [perf_cores(), torch.get_num_threads(), _cgroup_cpus() or perf_cores()]

    if hasattr(os, "sched_getaffinity"):
        counts.append(len(os.sched_getaffinity(0)))

    omp = os.environ.get("OMP_NUM_THREADS", "").strip()

    if omp.isdigit() and int(omp) > 0:
        counts.append(int(omp))

    return max(1, min(counts))


def _ns5(G):
    a, b, c = 3.4445, -4.7750, 2.0315
    X = G / (G.norm() + 1e-7)
    transposed = X.shape[0] > X.shape[1]
    X = X.T if transposed else X

    for _ in range(5):
        A = X @ X.T
        X = a * X + (b * A + c * A @ A) @ X

    return X.T if transposed else X


def _ns5_stack(G):
    a, b, c = 3.4445, -4.7750, 2.0315
    X = G / (G.norm(dim=(1, 2), keepdim=True) + 1e-7)
    transposed = X.shape[1] > X.shape[2]
    X = X.mT if transposed else X

    for _ in range(5):
        A = X @ X.mT
        X = torch.baddbmm(X, torch.baddbmm(A, A, A, beta=b, alpha=c), X, beta=a)

    return X.mT if transposed else X


def _kept(ok, x, other=0.0):
    return x if ok is None else torch.where(ok, x, other)


class Muon(torch.optim.Optimizer):
    def __init__(self, params, lr=0.02, momentum=0.95, adam_ratio=0.05):
        super().__init__(params, dict(lr=lr, momentum=momentum, adam_ratio=adam_ratio))

    # ok, a device bool, skips the step where False without a host sync; where True the
    # arithmetic is the plain step's, bit for bit
    @torch.no_grad()
    def step(self, ok=None):
        for group in self.param_groups:
            for param in group["params"]:
                if param.grad is None:
                    continue

                state = self.state.setdefault(param, {})
                mom = state.setdefault("m", torch.zeros_like(param))

                if ok is None:
                    new = mom.mul_(group["momentum"]).add_(param.grad)
                else:
                    new = mom.mul(group["momentum"]).add_(param.grad)
                    mom.copy_(torch.where(ok, new, mom))

                if param.ndim == 2:
                    update = _ns5(new) * max(
                        1.0, (param.shape[0] / param.shape[1]) ** 0.5
                    )
                    param.add_(_kept(ok, update), alpha=-group["lr"])
                # 1-D: an RMS-normalised momentum step
                else:
                    var = state.setdefault("v", torch.zeros_like(param))

                    if ok is None:
                        var_new = var.mul_(0.99)
                    else:
                        var_new = var.mul(0.99)

                    var_new.addcmul_(param.grad, param.grad, value=0.01)

                    if ok is not None:
                        var.copy_(torch.where(ok, var_new, var))

                    param.addcdiv_(
                        _kept(ok, new),
                        _kept(ok, var_new.sqrt().add_(1e-8), 1.0),
                        value=-group["lr"] * group["adam_ratio"],
                    )

    def _stacks(self):
        # each state is a view into its shape's moment stack, so the eager and the
        # graphed step share moments; parameters without a gradient are left out
        live = [
            (gi, p)
            for gi, group in enumerate(self.param_groups)
            for p in group["params"]
            if p.grad is not None
        ]
        key = tuple(id(p) for _, p in live)

        if getattr(self, "_stacked", (None,))[0] != key:
            by_shape = {}

            for gi, param in live:
                by_shape.setdefault((gi, tuple(param.shape)), []).append(param)

            stacked = []

            for (gi, shape), params in by_shape.items():
                p0 = params[0]
                M = torch.zeros(
                    (len(params),) + shape, device=p0.device, dtype=p0.dtype
                )
                V = torch.zeros_like(M) if len(shape) == 1 else None

                for i, param in enumerate(params):
                    state = self.state.setdefault(param, {})
                    M[i].copy_(state.get("m", M[i]))
                    state["m"] = M[i]

                    if V is not None:
                        V[i].copy_(state.get("v", V[i]))
                        state["v"] = V[i]

                stacked.append((self.param_groups[gi], params, M, V))

            self._stacked = (key, stacked)

        return self._stacked[1]

    # replayed by a CUDA graph: lr and momentum are device tensors filled before each
    # replay, as a python float would keep its capture value; ok False leaves all as is
    @torch.no_grad()
    def step_graph(self, ok, lr, momentum):
        if len(self.param_groups) != 1:
            raise RuntimeError("the graphed Muon step takes one parameter group")

        for group, params, M, V in self._stacks():
            G = torch.where(ok, torch.stack([p.grad for p in params]), 0.0)
            M.mul_(torch.where(ok, momentum, 1.0)).add_(G)

            if V is None:
                U = _ns5_stack(M) * max(1.0, (M.shape[1] / M.shape[2]) ** 0.5)
                rate = lr
            else:
                V.mul_(torch.where(ok, 0.99, 1.0)).addcmul_(G, G, value=0.01)
                U = M / (V.sqrt() + 1e-8)
                rate = lr * group["adam_ratio"]

            torch._foreach_add_(params, list((U * (-rate * ok)).unbind(0)))


class _Guarded:
    # steps on a device-side flag without reading it: where ok is False, device state
    # is put back at once and host state (Adam's step count) once the flag arrives
    def __init__(self, opt, dev):
        self.opt = opt
        self.cuda = dev.type == "cuda"
        self.flag = torch.ones((), dtype=torch.bool, pin_memory=self.cuda)
        self.event = torch.cuda.Event() if self.cuda else None
        self.pending = None

    def settle(self):
        if self.pending is None:
            return

        if self.event is not None:
            self.event.synchronize()

        if not bool(self.flag):
            for param, host in self.pending:
                if host is None:
                    # the step created the state; the optimizer makes it afresh
                    self.opt.state.pop(param, None)
                else:
                    for key, val in host.items():
                        self.opt.state[param][key].copy_(val)

        self.pending = None

    @torch.no_grad()
    def step(self, ok):
        self.settle()
        # the flag leaves first, so the next settle waits on the clip only
        self.flag.copy_(ok, non_blocking=self.cuda)

        if self.event is not None:
            self.event.record()

        params = [
            p for g in self.opt.param_groups for p in g["params"] if p.grad is not None
        ]
        before = []

        for param in params:
            state = self.opt.state.get(param) or {}
            saved = {
                key: val.clone() for key, val in state.items() if torch.is_tensor(val)
            }
            before.append((param, param.clone(), saved if state else None))

        self.opt.step()
        self.pending = []

        for param, old, saved in before:
            param.copy_(torch.where(ok, param, old))
            host = {}

            for key, val in self.opt.state[param].items():
                if not torch.is_tensor(val):
                    continue

                if val.device != param.device:
                    if saved is not None:
                        host[key] = saved[key]
                elif saved is not None:
                    val.copy_(torch.where(ok, val, saved[key]))

            self.pending.append((param, host if saved is not None else None))

        if self.event is None:
            self.settle()


# Adam on a device-side step count, which a graph can replay; a skipped step leaves
# the count, the moments and the parameters as they were
@torch.no_grad()
def _adam_graph(opt, ok):
    for group in opt.param_groups:
        b1, b2 = group["betas"]

        for param in group["params"]:
            if param.grad is None:
                continue

            state = opt.state[param]

            if not state:
                state.update(
                    step=torch.zeros((), device=param.device),
                    exp_avg=torch.zeros_like(param),
                    exp_avg_sq=torch.zeros_like(param),
                )
            elif state["step"].device != param.device:
                state["step"] = state["step"].to(param.device)

            g = torch.where(ok, param.grad, 0.0)
            state["step"].add_(ok)
            state["exp_avg"].lerp_(g, torch.where(ok, 1 - b1, 0.0))
            state["exp_avg_sq"].mul_(torch.where(ok, b2, 1.0)).addcmul_(
                g, g, value=1 - b2
            )
            bc1 = (1 - b1 ** state["step"]).clamp_min(1e-12)
            bc2 = (1 - b2 ** state["step"]).clamp_min(1e-12)
            den = state["exp_avg_sq"].sqrt() / bc2.sqrt() + group["eps"]
            param.add_(state["exp_avg"] / den * (-group["lr"] / bc1 * ok))


def _graphs_to_eager(opt_fine, scaler, scale):
    # what the graphed step kept on the device goes where the eager one keeps it
    if opt_fine is not None:
        for state in opt_fine.state.values():
            if "step" in state:
                state["step"] = state["step"].cpu()

    if scale is not None:
        scaler._scale, scaler._growth_tracker = scale


def _padded_unique(pts, cap):
    # torch.unique(pts, return_inverse=True) padded to cap rows, on the device; past
    # the cap the ranks are clamped and wrong, and the caller skips that step
    srt, perm = torch.sort(pts)
    first = torch.ones_like(srt, dtype=torch.bool)
    first[1:] = srt[1:] != srt[:-1]
    rank = torch.cumsum(first, 0) - 1
    count = rank[-1] + 1
    rank = rank.clamp_max(cap - 1)
    uniq = torch.zeros(cap, dtype=pts.dtype, device=pts.device).scatter_(0, rank, srt)

    return uniq, torch.empty_like(rank).scatter_(0, perm, rank), count


def _unique_bounds(edge_i, edge_j, n, eb):
    # a cell is among the endpoints of eb edges with probability 1 - (1 - deg / m)^eb;
    # the independent-indicator sd overstates the true one (negative correlation)
    m = edge_i.shape[0]
    deg = torch.bincount(torch.cat([edge_i, edge_j]), minlength=n).double()
    p = 1 - (1 - deg / m) ** eb
    mu, var = float(p.sum()), float((p * (1 - p)).sum())

    return max(1, round(mu)), min(2 * eb, n, math.ceil(mu + 10 * math.sqrt(var)) + 64)


def _graphs_on(cfg, dev, flow, lin_on):
    if (
        cfg.cudagraphs is False
        or dev.type != "cuda"
        or os.environ.get("FLODR_NO_GRAPHS") == "1"
    ):
        return False

    # python hooks would run at capture only
    hooked = any(
        p._backward_hooks or p._post_accumulate_grad_hooks for p in flow.parameters()
    ) or any(
        m._forward_hooks
        or m._forward_pre_hooks
        or m._backward_hooks
        or m._backward_pre_hooks
        for m in flow.modules()
    )
    eager = lin_on or cfg.checkpoint or cfg.optim != "muon" or hooked

    if eager and cfg.cudagraphs:
        warnings.warn(
            "cudagraphs=True: the linearity term, checkpointing, an optimizer other "
            "than Muon or a hook on the flow keeps the step eager",
            RuntimeWarning,
            stacklevel=3,
        )

    return not eager


class _StepGraphs:
    # one CUDA graph per phase (Fourier branch off or on, full-batch tail); warm-up
    # and capture are the fit's own steps, and a new phase drops the last graph
    def __init__(self, body, params, warmup=3):
        self.body, self.params, self.warmup = body, params, warmup
        self.key, self.graph, self.done = None, None, 0
        self.side = torch.cuda.Stream()
        self.loss = None

    def __call__(self, key):
        if key == self.key and self.graph is not None:
            self.graph.replay()
            return

        if key != self.key:
            self.key, self.graph, self.done = key, None, 0

        for param in self.params:
            param.grad = None

        if self.done < self.warmup:
            self.side.wait_stream(torch.cuda.current_stream())

            with torch.cuda.stream(self.side):
                self.loss = self.body(*key)

            torch.cuda.current_stream().wait_stream(self.side)
            self.done += 1
            return

        graph = torch.cuda.CUDAGraph()

        with torch.cuda.graph(graph):
            self.loss = self.body(*key)

        # the capture records the step without running it
        graph.replay()
        self.graph = graph

    def close(self):
        self.graph = None

        for param in self.params:
            param.grad = None


# largest edge batch for edge_batch=-1, below it the graph is taken whole
EDGE_BATCH_CAP = 262_144


@dataclass
class TrainConfig:
    iters: int = 700
    n_layers: int = 4
    hid: int = 256
    lr: float = 1e-3
    # negative samples per edge
    neg: int = 12
    min_dist: float = 0.01
    w_rep: float = 5.0
    w_recon: float = 2.0
    # mid-near attraction
    w_global: float = 0.6
    w_stress: float = 0.0
    # densMAP's density term: the correlation of each cell's log neighbour radius, input
    # vs layout; trades local recall for density preservation, so off by default
    w_dens: float = 0.0
    # 1 = ordinal hinge on ranks, 0 = log-ratio stress
    stress_ordinal: int = 1
    # "euclid" | "jaccard" (needs stress_X in {0, 1})
    stress_metric: str = "euclid"
    w_nll: float = 0.5
    # keep recon from reshaping the layout
    recon_detach: bool = True
    # ramp recon/nll/global in over this fraction of iters
    warmup_frac: float = 0.3
    # rows per iter for recon + NLL, 0 = full batch
    aux_batch: int = 512
    # trailing fraction of iters run full-batch
    aux_full_frac: float = 0.2
    # kNN edges per iter (0 = full batch, -1 = full batch up to EDGE_BATCH_CAP edges and
    # capped above)
    edge_batch: int = 0
    # couplings of the post-hoc conditional density tail
    cond_tail: int = 2
    cond_iters: int = 1500
    # input noise for the density, norm about this times the median 1-NN distance
    # (0 = none); with it the density cannot gain by putting cells on a sheet
    density_noise: float = 0.5
    # "energy": tail chosen by the energy certificate on held-out cells; "coverage": by
    # l-rank coverage, which keeps draws too wide; "auto": energy only with w_lin
    tail_select: str = "auto"
    # >0 restricts the sketch to the first N dims (the PCA head)
    sketch_head: int = 0
    # fixed k-dim sketch conditioner (0 = off)
    sketch: int = 0
    device: str = field(default_factory=default_device)
    # the step as one CUDA graph: None = wherever supported, False = eager; not bitwise
    # equal to eager. FLODR_NO_GRAPHS=1 turns it off too
    cudagraphs: bool = None
    # torch.compile the full-batch loss inside the graph
    graph_compile: bool = True
    # checkpoint the recon-inverse pass, less memory, slower
    checkpoint: bool = False
    # embedding dimension
    k: int = 2
    # nested-dropout cuts
    cuts: Tuple[int, ...] = (2, 2, 2, 4, 8, 16, 24, 32)
    # torch CPU threads (0 = leave untouched)
    threads: int = 0
    # OneCycleLR at max_lr = 3 * lr
    onecycle: bool = False
    # torch.compile(step) on CPU, not for edge_batch
    cpu_compile: bool = False
    # torch.compile the edge-batched loss on CUDA; the draws stay eager
    compile_edges: bool = False
    # "bf16" or "fp16" hidden layers on the edge-batched step, output layers and Fourier
    # phase in fp32; fp16 with loss scaling
    half_edges: str = ""
    # bf16 matmuls while training, fp32 everywhere else
    bf16: bool = False
    # "adam" | "nadam" | "muon" (lr ~0.02)
    optim: str = "adam"
    # "tanh" | "relu" | "silu"
    act: str = "tanh"
    # 1-hidden-layer conditioners
    shallow: bool = False
    # >0 bounds |log scale| per coupling
    gate_max: float = 0.0
    # >0 makes the bound per dim, this on the display and gate_max on the residual
    gate_display: float = 0.0
    # FloDR pads narrower inputs with unit-variance noise columns up to this width, the
    # graph and the stress stay on X
    pad: int = 0
    # >0 adds a FourierShift to the couplings that write the display, at this weight
    fine_eps: float = 0.0
    # its own Adam step size, and the fraction of iters before it switches on
    fine_lr: float = 1e-3
    fine_start: float = 0.5
    # log(1 + rho^2), rho = |F^-1(F(x) + J e) - (x + e)| / |e|, e the density's noise
    # times h, so the draws are that noise; opt-in as it costs recall; needs noise_sd
    w_lin: float = 0.0
    # off until the layout has formed, then a linear ramp over lin_ramp of the iters
    lin_start: float = 0.5
    lin_ramp: float = 0.1
    lin_h: Tuple[float, ...] = (0.5, 1.0)
    lin_batch: int = 1024
    # every lin_every-th step at lin_every times the weight: the same push through the
    # momentum for a fraction of the kernel launches
    lin_every: int = 2
    # log(1 + rho^2 / lin_tau^2)
    lin_tau: float = 1.0
    seed: int = 0


def raw_recipe(seed=0, threads=None, **overrides):
    dev = str(overrides.get("device") or default_device())
    if threads is None:
        threads = 0 if dev.startswith("cuda") else default_threads()

    kw = dict(
        seed=seed,
        threads=threads,
        sketch=64,
        onecycle=True,
        iters=6400,
        aux_full_frac=0.05,
        w_nll=0.0,
        cond_tail=0,
        # bf16 only reaches the full-batch step; the edge-batched one has half_edges
        bf16=False,
        optim="muon",
        lr=0.0025,
        w_rep=15.0,
        # raw mode defers recon anyway
        w_recon=0.0,
        neg=48,
        warmup_frac=0.05,
        edge_batch=-1,
        gate_max=0.5,
        gate_display=1.0,
        pad=30,
        fine_eps=0.25,
        compile_edges=True,
        half_edges="fp16",
    )
    kw.update(overrides)

    # an iters override keeps lr * iters = 16, as Muon's step is magnitude-blind, and
    # fine_lr * iters = 6.4; below 800 iters both stay at their 800-iter values
    if "iters" in overrides and "lr" not in overrides:
        kw["lr"] = min(16.0 / kw["iters"], 0.02)

    if "iters" in overrides and "fine_lr" not in overrides:
        kw["fine_lr"] = min(6.4 / kw["iters"], 0.008)

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
    progress=False,
    noise_sd=None,
):
    # cfg.threads and deterministic kernels are process-wide, so hold for this fit only
    if not cfg.threads:
        return _train_flodr(
            Xp, ei, ej, cfg, rng, return_model, w, stress_X, progress, noise_sd
        )

    prev = torch.get_num_threads()
    torch.set_num_threads(cfg.threads)

    try:
        with deterministic():
            return _train_flodr(
                Xp,
                ei,
                ej,
                cfg,
                rng,
                return_model,
                w,
                stress_X,
                progress,
                noise_sd,
            )
    finally:
        torch.set_num_threads(prev)


def _train_flodr(Xp, ei, ej, cfg, rng, return_model, w, stress_X, progress, noise_sd):
    n, D = Xp.shape
    k = cfg.k

    torch.manual_seed(cfg.seed)
    dev = torch.device(cfg.device)

    # the compiled steps are closures redefined every fit, so their cache entries pile
    # up on one code object; the fifth fit on new shapes then hits the recompile limit,
    # an error under fullgraph
    # this clears every compiled function in the process; evicting per code object would
    # spare users' own torch.compile models
    if dev.type == "cuda" or cfg.cpu_compile:
        torch.compiler.reset()

    # output kernel 1 / (1 + a d^(2b))
    a, b = find_ab(cfg.min_dist)
    x_t = torch.from_numpy(Xp).float().to(dev)

    edge_i = torch.from_numpy(ei).to(dev)
    edge_j = torch.from_numpy(ej).to(dev)

    w_attr = torch.from_numpy(np.asarray(w)).float().to(dev) if w is not None else None
    dim_idx = torch.arange(D, device=dev)

    # floored at k, because a cut below the embedding would zero part of the map
    cuts = tuple(sorted({min(max(c, k), D) for c in cfg.cuts} | {D}))

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
    )
    fine = []

    if cfg.fine_eps:
        gen_fine = torch.Generator().manual_seed(cfg.seed + 9000)

        # only the couplings that write the display and nothing else
        for layer in flow.layers:
            free = 1 - layer.mask

            if bool(free[:k].all()) and not bool(free[k:].any()):
                fine.append(layer)

        for layer in fine:
            d_in = layer.S.shape[0] if layer.S is not None else D
            layer.net = FourierShift(
                layer.net, d_in, cfg.hid, 2 * D, eps=cfg.fine_eps, gen=gen_fine
            )

        fine = [
            layer.net for layer in flow.layers if isinstance(layer.net, FourierShift)
        ]

    flow = flow.to(dev)

    fine_params = [p for m in fine for p in m.high.parameters()]
    fine_ids = {id(p) for p in fine_params}
    params = [p for p in flow.parameters() if id(p) not in fine_ids]

    opt = {
        "adam": lambda: torch.optim.Adam(params, lr=cfg.lr, fused=dev.type == "cuda"),
        "nadam": lambda: torch.optim.NAdam(params, lr=cfg.lr),
        "muon": lambda: Muon(params, lr=cfg.lr),
    }[cfg.optim]()
    opt_fine = torch.optim.Adam(fine_params, lr=cfg.fine_lr) if fine else None

    n_edges = edge_i.shape[0]

    # its own generator, so every other draw is the one a fit without it makes
    lin_on = bool(cfg.w_lin) and noise_sd is not None

    if lin_on:
        lin_sd = torch.as_tensor(np.asarray(noise_sd), dtype=torch.float32, device=dev)
        lin_h = torch.as_tensor(cfg.lin_h, dtype=torch.float32, device=dev)
        gen_lin = torch.Generator(device=dev).manual_seed(cfg.seed + 4711)
        lin_t0 = int(cfg.lin_start * cfg.iters)
        lin_n = min(cfg.lin_batch, n)

    def lin_loss(it):
        if it < lin_t0 or it % cfg.lin_every:
            return None

        ramp_lin = min(1.0, (it - lin_t0 + 1) / max(1.0, cfg.lin_ramp * cfg.iters))
        x = x_t[torch.randint(0, n, (lin_n,), device=dev, generator=gen_lin)]

        def pick(m):
            return torch.randint(0, m, (lin_n, 1), device=dev, generator=gen_lin)

        def noise():
            return lin_sd * torch.randn(x.shape, device=dev, generator=gen_lin)

        # half the cells clean, half with one draw of the noise, one h per cell
        x = (x + pick(2) * noise()).requires_grad_(True)
        e = lin_h[pick(len(lin_h))] * noise()
        T = flow.forward_main(x)
        # J e by two reverse passes, exact and fewer launches than forward mode in eager
        a = torch.zeros_like(T, requires_grad=True)
        g = torch.autograd.grad((T * a).sum(), x, create_graph=True)[0]
        t = torch.autograd.grad((g * e).sum(), a, create_graph=True)[0]
        rho2 = ((flow.inverse_main(T + t) - x - e) ** 2).sum(1) / (e**2).sum(1)
        pen = torch.log1p(rho2 / cfg.lin_tau**2).mean()

        return cfg.w_lin * cfg.lin_every * ramp_lin * pen

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

        row_sums = x_stress.sum(1)

    def _lh(row_a, row_b):
        if cfg.stress_metric == "jaccard":
            inter = (x_stress[row_a] * x_stress[row_b]).sum(1)
            union = (row_sums[row_a] + row_sums[row_b] - inter).clamp_min(1.0)
            return torch.log((1.0 - inter / union).clamp_min(1e-6))

        return 0.5 * torch.log(
            ((x_stress[row_a] - x_stress[row_b]) ** 2).sum(1).clamp_min(1e-12)
        )

    if cfg.w_dens:
        w_e = (
            w_attr
            if w_attr is not None
            else torch.ones_like(edge_i, dtype=torch.float32)
        )
        mu = (
            torch.zeros(n, device=dev)
            .index_add(0, edge_i, w_e)
            .index_add(0, edge_j, w_e)
        )

        # densMAP's local radius, the weighted mean squared edge length, as a unit
        # vector of centred logs, so a dot product of two is their correlation
        def log_radius(d2):
            r = torch.zeros(n, device=dev).index_add(0, edge_i, w_e * d2)
            r = r.index_add(0, edge_j, w_e * d2)
            r = torch.log(1e-8 + r / mu.clamp_min(1e-12))
            r = r - r.mean()

            return r / r.norm().clamp_min(1e-12)

        # in the input's own metric
        ro = log_radius(torch.exp(2 * _lh(edge_i, edge_j)))

    aux_n = n if cfg.aux_batch <= 0 else min(cfg.aux_batch, n)

    def step(
        neg_a,
        neg_b,
        glob_a,
        glob_b,
        rows,
        keep,
        ramp,
        str_a,
        str_b,
        str_c,
        str_d,
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

        z_sub, logdet_s = Z[rows], logdet[rows]

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

            L_recon = ((x_rec - x_t[rows]) ** 2).mean()
        else:
            L_recon = 0.0

        L_nll = -(flow.base_log_prob(z_sub) + logdet_s).mean() / D if cfg.w_nll else 0.0
        L_dens = -(log_radius(d2_edge) @ ro) if cfg.w_dens else 0.0

        return L_local + ramp * (
            cfg.w_recon * L_recon
            + cfg.w_global * L_glob
            + cfg.w_nll * L_nll
            + cfg.w_stress * L_stress
            + cfg.w_dens * L_dens
        )

    # a step's draws, outside the loss, so a compiled loss sees the eager indices
    def draw_edge_batch(edge_idx):
        pts = torch.cat([edge_i[edge_idx], edge_j[edge_idx]])
        uniq, inv = torch.unique(pts, return_inverse=True)

        n_uniq, n_edge = uniq.shape[0], edge_idx.shape[0]

        def draw(high, size):
            return torch.randint(0, high, (size,), device=dev, generator=gen)

        neg_a = draw(n_uniq, cfg.neg * n_edge)
        neg_b = draw(n_uniq, cfg.neg * n_edge)
        glob_a = draw(n_uniq, 3 * n_uniq)
        glob_b = draw(n_uniq, 3 * n_uniq)
        idx_a = idx_b = idx_c = idx_d = mask_a = mask_b = mdist_a = mdist_b = None

        if cfg.w_stress:
            n_pairs = 4 * n_uniq
            idx_a, idx_b = draw(n_uniq, n_pairs), draw(n_uniq, n_pairs)
            row_a, row_b = uniq[idx_a], uniq[idx_b]
            mask_a = (((row_a + row_b) % 5) != 0).float()
            mdist_a = _lh(row_a, row_b)

            if cfg.stress_ordinal:
                idx_c, idx_d = draw(n_uniq, n_pairs), draw(n_uniq, n_pairs)
                row_c, row_d = uniq[idx_c], uniq[idx_d]
                mask_b = (((row_c + row_d) % 5) != 0).float()
                mdist_b = _lh(row_c, row_d)

        rows = draw(n_uniq, min(aux_n, n_uniq))
        wa = w_attr[edge_idx] if w_attr is not None else None

        return (
            x_t[uniq],
            inv,
            wa,
            neg_a,
            neg_b,
            glob_a,
            glob_b,
            idx_a,
            idx_b,
            idx_c,
            idx_d,
            mask_a,
            mask_b,
            mdist_a,
            mdist_b,
            rows,
            edge_idx,
        )

    def loss_edge_batch(
        x_uniq,
        inv,
        wa,
        neg_a,
        neg_b,
        glob_a,
        glob_b,
        idx_a,
        idx_b,
        idx_c,
        idx_d,
        mask_a,
        mask_b,
        mdist_a,
        mdist_b,
        rows,
        edge_idx,
        keep,
        ramp,
    ):
        n_edge = inv.shape[0] // 2
        Z, logdet = flow.forward_with_logdet(x_uniq)
        Y = Z[:, :k]

        d2_edge = ((Y[inv[:n_edge]] - Y[inv[n_edge:]]) ** 2).sum(1)
        w_edge = 1.0 / (1.0 + a * (d2_edge + 1e-6) ** b)

        if wa is None:
            L_attr = -torch.log(w_edge + 1e-6).mean()
        else:
            L_attr = -(wa * torch.log(w_edge + 1e-6)).sum() / wa.sum()

        d2_neg = ((Y[neg_a] - Y[neg_b]) ** 2).sum(1)
        w_neg = 1.0 / (1.0 + a * (d2_neg + 1e-6) ** b)

        L_rep = -torch.log(1.0 - w_neg + 1e-6).mean()

        d2_glob = ((Y[glob_a] - Y[glob_b]) ** 2).sum(1)
        L_glob = (d2_glob / (1.0 + d2_glob)).mean()

        if cfg.w_stress:
            ldist_a = 0.5 * torch.log(
                ((Y[idx_a] - Y[idx_b]) ** 2).sum(1).clamp_min(1e-12)
            )

            if cfg.stress_ordinal:
                ldist_b = 0.5 * torch.log(
                    ((Y[idx_c] - Y[idx_d]) ** 2).sum(1).clamp_min(1e-12)
                )
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

        z_sub, logdet_s = Z[rows], logdet[rows]

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

        if cfg.w_dens:
            d2_e = d2_edge.float()
            # the batch's edges at their current length, every other edge as last seen
            L_dens = -(log_radius(d2_buf.index_put((edge_idx,), d2_e)) @ ro)
            # a skipped over-cap batch still writes lengths between wrong cells; they
            # are finite and redrawn within a few steps
            d2_buf.index_put_((edge_idx,), d2_e.detach())
        else:
            L_dens = 0.0

        return (
            L_attr
            + cfg.w_rep * L_rep
            + ramp
            * (
                cfg.w_recon * L_recon
                + cfg.w_global * L_glob
                + cfg.w_nll * L_nll
                + cfg.w_stress * L_stress
                + cfg.w_dens * L_dens
            )
        )

    compiled_edges = cfg.compile_edges and dev.type == "cuda" and not cfg.w_recon
    loss_edge_eager = loss_edge_batch

    if compiled_edges:
        # shapes change every step, so dynamic; the atomic scatters in backward are not
        # bitwise reproducible across runs
        loss_edge_eager = torch.compile(loss_edge_batch, dynamic=True)

    def step_edge_batch(edge_idx, keep, ramp):
        if compiled_edges:
            ramp = torch.tensor(ramp, dtype=torch.float32, device=dev)

        return loss_edge_eager(*draw_edge_batch(edge_idx), keep, ramp)

    if cfg.edge_batch < 0:
        eb_size = EDGE_BATCH_CAP if n_edges > EDGE_BATCH_CAP else 0
    else:
        eb_size = min(cfg.edge_batch, n_edges) if cfg.edge_batch else 0

    # a batch holds a few edges, so each cell's radius takes every other edge's last
    # length
    d2_buf = None

    if cfg.w_dens and eb_size:
        with torch.no_grad():
            Y0 = torch.cat(
                [
                    flow.forward_with_logdet(x_t[s : s + 8192])[0][:, :k]
                    for s in range(0, n, 8192)
                ]
            )

        d2_buf = ((Y0[edge_i] - Y0[edge_j]) ** 2).sum(1).float()

    graphs = _graphs_on(cfg, dev, flow, lin_on)
    step_mode = "graph" if graphs else "eager"
    compiled = cfg.cpu_compile and not eb_size and not graphs

    if compiled:
        step = torch.compile(step, dynamic=False)
        # ramp enters as a tensor, as a fresh python float would recompile every iter
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

    fp16 = cfg.half_edges == "fp16" and dev.type == "cuda"
    half_dtype = torch.float16 if fp16 else torch.bfloat16
    half_on = fp16 or (cfg.half_edges == "bf16" and cfg.bf16)
    Coupling.fp32_head = half_on and bool(eb_size)
    # a no-op unless fp16: scale, unscale_ and update pass through
    scaler = torch.amp.GradScaler(dev.type, enabled=fp16 and bool(eb_size))

    # the non-finite guard decides on the device: Muon masks its own step, any other
    # optimizer is put back
    main_step = opt.step if isinstance(opt, Muon) else _Guarded(opt, dev).step
    fine_guard = _Guarded(opt_fine, dev) if opt_fine is not None else None

    if graphs:
        # what changes between replays sits in fixed buffers filled before each: the
        # draws, the dropout mask, the ramp, and Muon's lr and momentum (OneCycle)
        lr_t = torch.zeros((), device=dev)
        mom_t = torch.zeros((), device=dev)
        ramp_t = torch.zeros((), device=dev)
        keep_t = torch.zeros(D, device=dev)
        keeps = {c: (dim_idx < c).to(torch.float32) for c in cuts}
        rows_used = bool(cfg.w_recon or cfg.w_nll)
        buf = {}

        if eb_size:
            # endpoints padded to a fixed count, uniforms scaled by the true count, as
            # torch.unique's data-dependent size would sync
            u_mean, u_cap = _unique_bounds(edge_i, edge_j, n, eb_size)

            # float64, as a float32 uniform times a large count favours some indices
            def uniform(*size):
                return torch.zeros(size, device=dev, dtype=torch.float64)

            buf.update(
                neg_a=uniform(cfg.neg * eb_size),
                neg_b=uniform(cfg.neg * eb_size),
                glob_a=uniform(3 * u_mean),
                glob_b=uniform(3 * u_mean),
                rows=uniform(min(aux_n, u_mean)),
                pairs=(
                    uniform(4 if cfg.stress_ordinal else 2, 4 * u_mean)
                    if cfg.w_stress
                    else None
                ),
            )
            edge_idx_t = torch.zeros(eb_size, dtype=torch.int64, device=dev)

            def fill(late):
                torch.randint(0, n_edges, (eb_size,), generator=gen, out=edge_idx_t)

                for val in buf.values():
                    if val is not None:
                        torch.rand(val.shape, generator=gen, out=val)

            def loss_edges_fixed(
                edge_idx, neg_a, neg_b, glob_a, glob_b, rows, pairs, keep, ramp
            ):
                pts = torch.cat([edge_i[edge_idx], edge_j[edge_idx]])
                uniq, inv, n_uniq = _padded_unique(pts, u_cap)
                n_in = n_uniq.clamp_max(u_cap)

                def pick(u):
                    return torch.minimum((u * n_in).long(), n_in - 1)

                idx = [None] * 4
                mask_a = mask_b = mdist_a = mdist_b = None

                if cfg.w_stress:
                    idx[: len(pairs)] = [pick(u) for u in pairs]
                    row_a, row_b = uniq[idx[0]], uniq[idx[1]]
                    mask_a = (((row_a + row_b) % 5) != 0).float()
                    mdist_a = _lh(row_a, row_b)

                    if cfg.stress_ordinal:
                        row_c, row_d = uniq[idx[2]], uniq[idx[3]]
                        mask_b = (((row_c + row_d) % 5) != 0).float()
                        mdist_b = _lh(row_c, row_d)

                loss = loss_edge_batch(
                    x_t[uniq],
                    inv,
                    w_attr[edge_idx] if w_attr is not None else None,
                    pick(neg_a),
                    pick(neg_b),
                    pick(glob_a),
                    pick(glob_b),
                    *idx,
                    mask_a,
                    mask_b,
                    mdist_a,
                    mdist_b,
                    pick(rows),
                    edge_idx,
                    keep,
                    ramp,
                )

                # a batch with more endpoints than the cap is skipped
                return loss, n_uniq <= u_cap

            loss_fixed = (
                torch.compile(loss_edges_fixed, dynamic=False, fullgraph=True)
                if compiled_edges
                else loss_edges_fixed
            )

            def graph_loss(late):
                return loss_fixed(edge_idx_t, *buf.values(), keep_t, ramp_t)

        else:

            def index(size):
                return torch.zeros(size, dtype=torch.int32, device=dev)

            # in the order the eager step draws them, so both see the same indices
            buf.update(
                neg_a=index(cfg.neg * n_edges),
                neg_b=index(cfg.neg * n_edges),
                glob_a=index(3 * n),
                glob_b=index(3 * n),
                rows=index(aux_n),
            )

            if cfg.w_stress:
                buf.update(str_a=index(4 * n), str_b=index(4 * n))

                if cfg.stress_ordinal:
                    buf.update(str_c=index(4 * n), str_d=index(4 * n))

            rows_all = torch.arange(n, device=dev)
            spare = buf["neg_a"][:1]

            def fill(late):
                for key, val in buf.items():
                    if key == "rows" and late:
                        continue

                    torch.randint(0, n, val.shape, generator=gen, out=val)

            step_fixed = (
                torch.compile(step, dynamic=False, fullgraph=True)
                if cfg.graph_compile
                else step
            )

            def graph_loss(late):
                pairs = [buf.get(key, spare) for key in ("str_a", "str_b")]
                pairs += (
                    [buf.get(key, spare) for key in ("str_c", "str_d")]
                    if "str_c" in buf
                    else [pairs[0][:1]] * 2
                )

                return (
                    step_fixed(
                        buf["neg_a"],
                        buf["neg_b"],
                        buf["glob_a"],
                        buf["glob_b"],
                        rows_all if late else buf["rows"],
                        keep_t,
                        ramp_t,
                        *pairs,
                    ),
                    None,
                )

        all_params = list(flow.parameters())
        scale_t = None

        if scaler.is_enabled():
            # GradScaler's own step reads its inf flag on the host; the graph keeps the
            # same scale, backoff and growth on the device
            scale_t = torch.full((), 2.0**16, device=dev)
            tracker = torch.zeros((), dtype=torch.int32, device=dev)
            found = torch.zeros((), device=dev)

        def body(fine_on, late):
            with torch.autocast(
                dev.type,
                dtype=half_dtype if eb_size else torch.bfloat16,
                enabled=half_on if eb_size else cfg.bf16,
                cache_enabled=False,
            ):
                loss, fits = graph_loss(late)

            (loss if scale_t is None else loss * scale_t).backward()

            live = [p for p in all_params if p.grad is not None]

            if scale_t is not None:
                found.zero_()
                torch._amp_foreach_non_finite_check_and_unscale_(
                    [p.grad for p in live], found, scale_t.double().reciprocal().float()
                )

            total = torch.nn.utils.clip_grad_norm_(live, 5.0, foreach=True)
            ok = torch.isfinite(total) if fits is None else torch.isfinite(total) & fits
            opt.step_graph(ok, lr_t, mom_t)

            if fine_on:
                _adam_graph(opt_fine, ok)

            if scale_t is not None:
                torch._amp_update_scale_(scale_t, tracker, found, 2.0, 0.5, 2000)

            return loss.detach()

        runner = _StepGraphs(body, all_params)

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
        cut = int(rng.choice(cuts))

        for m in fine:
            m.on = it >= cfg.fine_start * cfg.iters

        # the branch has no gradients before it switches on, and the scaler needs some
        fine_on = opt_fine is not None and fine[0].on

        if graphs:
            late = it >= (1.0 - cfg.aux_full_frac) * cfg.iters and not eb_size
            fill(late)
            keep_t.copy_(keeps[cut])
            ramp_t.fill_(ramp)
            lr_t.fill_(opt.param_groups[0]["lr"])
            mom_t.fill_(opt.param_groups[0]["momentum"])

            try:
                runner((fine_on, late and rows_used))
            except RuntimeError as err:
                warnings.warn(
                    f"the CUDA-graph step failed ({str(err).splitlines()[0][:200]}); "
                    "the fit continues eager",
                    RuntimeWarning,
                    stacklevel=2,
                )
                graphs = False
                step_mode = f"graph to step {it}, then eager"
                runner.close()
                _graphs_to_eager(
                    opt_fine, scaler, None if scale_t is None else (scale_t, tracker)
                )
            else:
                if sched is not None:
                    # the scheduler wants to see the optimizer's own step called first
                    opt._opt_called = True
                    sched.step()

                continue

        keep = (dim_idx < cut).to(torch.float32)

        if eb_size:
            edge_idx = torch.randint(0, n_edges, (eb_size,), device=dev, generator=gen)
            with torch.autocast(dev.type, dtype=half_dtype, enabled=half_on):
                loss = step_edge_batch(edge_idx, keep, ramp)
        else:
            neg_a, neg_b = _ri(cfg.neg * n_edges), _ri(cfg.neg * n_edges)
            glob_a, glob_b = _ri(3 * n), _ri(3 * n)

            if it >= (1.0 - cfg.aux_full_frac) * cfg.iters:
                rows = torch.arange(n, device=dev)
            else:
                rows = _ri(aux_n)

            # drawn only when used, so w_stress=0 leaves the RNG stream untouched
            if cfg.w_stress:
                str_a, str_b = _ri(4 * n), _ri(4 * n)
                str_c, str_d = (
                    (_ri(4 * n), _ri(4 * n))
                    if cfg.stress_ordinal
                    else (str_a[:1], str_a[:1])
                )
            else:
                str_a = str_b = str_c = str_d = neg_a[:1]

            with torch.autocast(dev.type, dtype=torch.bfloat16, enabled=cfg.bf16):
                loss = step(
                    neg_a,
                    neg_b,
                    glob_a,
                    glob_b,
                    rows,
                    keep,
                    ramps[it] if compiled else ramp,
                    str_a,
                    str_b,
                    str_c,
                    str_d,
                )

        # outside autocast, in fp32
        if lin_on:
            pen = lin_loss(it)

            if pen is not None:
                loss = loss + pen

        opt.zero_grad(set_to_none=True)

        if opt_fine is not None:
            opt_fine.zero_grad(set_to_none=True)

        scaler.scale(loss).backward()
        scaler.unscale_(opt)

        if fine_on:
            scaler.unscale_(opt_fine)

        # a NaN in one parameter set makes the shared clip norm NaN for both, so both
        # skip; the flag stays on the device, a host read every step stalls queueing
        total = torch.nn.utils.clip_grad_norm_(flow.parameters(), 5.0)
        ok = torch.isfinite(total)
        main_step(ok)

        if fine_on:
            fine_guard.step(ok)

        scaler.update()

        if sched is not None:
            sched.step()

    if graphs:
        runner.close()

    # what ran, for the estimator to report
    flow._step_mode = step_mode

    if fine_guard is not None:
        fine_guard.settle()

    if hasattr(main_step, "__self__") and isinstance(main_step.__self__, _Guarded):
        main_step.__self__.settle()

    if cfg.cond_tail:
        # deferred to the first density use
        flow._density_args = (x_t, cfg.cond_tail, cfg.cond_iters, cfg.seed)

    with torch.no_grad():
        Z = flow(x_t)
        Y = Z[:, :k].cpu().numpy()
        roundtrip = float(np.abs(flow.inverse(Z).cpu().numpy() - Xp).max())
        scale = np.sqrt((Xp**2).sum(1)).mean()
        # truncation lives on the main latent; the density tail re-coordinates dims k on
        z_main = flow.forward_main(x_t)
        recon = {}

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
