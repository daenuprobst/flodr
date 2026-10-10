# The fast paths: the non-finite guard decided on the device, the whole-step CUDA graph and the GPU kNN.
# On the CPU, the guarded steps are bitwise the plain ones where the step is finite and leave every
# value as it was where it is not; the graphed optimizers and the padded unique set agree with the eager
# ones; the brute-force kNN is the exact one. On a GPU, a graphed fit skips poisoned steps, falls back
# to eager when a capture fails, and lands near the eager fit.

import copy
import os
import warnings

import numpy as np
import pytest
import torch
from scipy.spatial import cKDTree
from scipy.stats import ks_2samp

import flodr.train as ft
from flodr import FloDR
from flodr.data import fuzzy_knn_graph, knn_gpu, knn_search

GPU = pytest.mark.skipif(not torch.cuda.is_available(), reason="no GPU")
DEVICES = ["cpu", pytest.param("cuda", marks=GPU)]


def blobs(n, d, k=4, seed=0):
    rng = np.random.default_rng(seed)
    c = rng.normal(size=(k, d)) * 4
    return (c[rng.integers(0, k, n)] + rng.normal(size=(n, d))).astype(np.float32)


def params(device, seed=0):
    g = torch.Generator().manual_seed(seed)
    shapes = [(32, 12), (32, 12), (12, 32), (32,), (32,), (5,)]
    return [torch.randn(s, generator=g).to(device).requires_grad_() for s in shapes]


def set_grads(ps, seed, poison=False):
    g = torch.Generator().manual_seed(seed)
    for p in ps:
        p.grad = torch.randn(p.shape, generator=g).to(p.device)
    if poison:
        ps[0].grad[0, 0] = float("nan")


def snapshot(opt, ps):
    out = [p.detach().clone() for p in ps]
    for p in ps:
        for v in opt.state.get(p, {}).values():
            if torch.is_tensor(v):
                out.append(v.detach().clone())
    return out


def same(a, b):
    return len(a) == len(b) and all(torch.equal(x.cpu(), y.cpu()) for x, y in zip(a, b))


@pytest.mark.parametrize("device", DEVICES)
def test_muon_guard_is_bitwise(device):
    torch.set_num_threads(2)
    pa, pb = params(device), params(device)
    a, b = ft.Muon(pa, lr=0.01), ft.Muon(pb, lr=0.01)
    yes = torch.ones((), dtype=torch.bool, device=device)

    for s in range(4):
        set_grads(pa, s)
        set_grads(pb, s)
        a.step()
        b.step(yes)
        assert same(snapshot(a, pa), snapshot(b, pb))

    before = snapshot(b, pb)
    set_grads(pb, 9, poison=True)
    b.step(torch.isfinite(sum(p.grad.sum() for p in pb)))
    assert same(before, snapshot(b, pb))


@pytest.mark.parametrize("device", DEVICES)
def test_guarded_adam_skips_exactly(device):
    # a skipped step is as if it never happened: the next steps match an Adam that never saw it,
    # step count included, whether the count lives on the host (wound back a step late) or not
    torch.set_num_threads(2)
    pa, pb = params(device), params(device)
    a = torch.optim.Adam(pa, lr=1e-2)
    b = ft._Guarded(torch.optim.Adam(pb, lr=1e-2), torch.device(device))

    def ok():
        return torch.isfinite(sum(p.grad.sum() for p in pb))

    # a skip at the very first step leaves no state behind
    set_grads(pb, 99, poison=True)
    b.step(ok())

    for s in range(6):
        set_grads(pa, s)
        set_grads(pb, s)
        a.step()
        b.step(ok())

        if s == 2:
            set_grads(pb, 77, poison=True)
            b.step(ok())

    b.settle()
    assert same(snapshot(a, pa), snapshot(b.opt, pb))


def test_graph_optimizers_match_eager():
    # the graphed steps on the CPU, under OneCycle as in a default fit: Muon batched by shape and Adam
    # on a device-side count agree with the eager ones to float rounding, and a skipped step leaves
    # every value bitwise as it was. The graphed Muon may read lr and momentum only from its tensor
    # arguments: a capture would freeze any group value it reads, so those are NaN while it steps
    torch.set_num_threads(2)
    pa, pb = params("cpu"), params("cpu")
    a, b = ft.Muon(pa, lr=0.01), ft.Muon(pb, lr=0.01)
    sa, sb = (
        torch.optim.lr_scheduler.OneCycleLR(o, max_lr=0.03, total_steps=60)
        for o in (a, b)
    )
    # as train_flodr does, since the graphed step bypasses the optimizer's own step
    b._opt_called = True
    fa, fb = params("cpu", 1), params("cpu", 1)
    adam_a, adam_b = torch.optim.Adam(fa, lr=1e-2), torch.optim.Adam(fb, lr=1e-2)
    yes = torch.ones((), dtype=torch.bool)
    moms = []

    def graph_step(ok):
        group = b.param_groups[0]
        lr, mom = torch.tensor(group["lr"]), torch.tensor(group["momentum"])
        moms.append(group["momentum"])
        group["lr"] = group["momentum"] = float("nan")
        try:
            b.step_graph(ok, lr, mom)
        finally:
            group["lr"], group["momentum"] = float(lr), float(mom)

    for s in range(40):
        for ps in (pa, pb, fa, fb):
            set_grads(ps, s)
        a.step()
        graph_step(yes)
        adam_a.step()
        ft._adam_graph(adam_b, yes)
        sa.step()
        sb.step()

    # OneCycle moved the momentum, which the graphed step followed
    assert max(moms) - min(moms) > 0.05
    for x, y in zip(snapshot(a, pa), snapshot(b, pb)):
        torch.testing.assert_close(x, y, rtol=1e-3, atol=1e-4)
    for x, y in zip(snapshot(adam_a, fa), snapshot(adam_b, fb)):
        torch.testing.assert_close(x, y, rtol=1e-4, atol=1e-6)

    before = snapshot(b, pb) + snapshot(adam_b, fb)
    set_grads(pb, 9, poison=True)
    set_grads(fb, 9, poison=True)
    no = torch.isfinite(pb[0].grad.sum())
    graph_step(no)
    ft._adam_graph(adam_b, no)
    assert same(before, snapshot(b, pb) + snapshot(adam_b, fb))
    # the eager step continues from the moments the graphed one wrote
    set_grads(pa, 5)
    set_grads(pb, 5)
    a.step()
    b.step(yes)
    for x, y in zip(snapshot(a, pa), snapshot(b, pb)):
        torch.testing.assert_close(x, y, rtol=1e-3, atol=1e-4)


@GPU
@pytest.mark.parametrize("frozen", [False, True], ids=["buffers", "frozen"])
def test_captured_step_follows_the_schedule(frozen):
    # the graphed optimizers inside a real CUDA graph, replayed under OneCycle from the same states and
    # gradients as the eager ones: lr and momentum come from buffers filled before each replay, so the
    # replays follow the eager steps. The frozen arm passes momentum as a constant tensor, as reading the
    # group at capture did (review B1), and must not
    pa, pb = params("cuda"), params("cuda")
    fa, fb = params("cuda", 1), params("cuda", 1)
    a, b = ft.Muon(pa, lr=0.01), ft.Muon(pb, lr=0.01)
    sa, sb = (
        torch.optim.lr_scheduler.OneCycleLR(o, max_lr=0.03, total_steps=60)
        for o in (a, b)
    )
    b._opt_called = True
    adam_a, adam_b = torch.optim.Adam(fa, lr=1e-2), torch.optim.Adam(fb, lr=1e-2)
    lr, mom = torch.zeros((), device="cuda"), torch.zeros((), device="cuda")
    held = torch.full((), 0.95, device="cuda")
    ok = torch.ones((), dtype=torch.bool, device="cuda")
    set_grads(pb, 0)
    set_grads(fb, 0)

    def body():
        b.step_graph(ok, lr, held if frozen else mom)
        ft._adam_graph(adam_b, ok)

    graph, side = None, torch.cuda.Stream()

    for s in range(40):
        set_grads(pa, s)
        set_grads(fa, s)
        for x, y in zip(pb + fb, pa + fa):
            x.grad.copy_(y.grad)
        lr.fill_(b.param_groups[0]["lr"])
        mom.fill_(b.param_groups[0]["momentum"])
        a.step()
        adam_a.step()

        if s < 2:
            side.wait_stream(torch.cuda.current_stream())
            with torch.cuda.stream(side):
                body()
            torch.cuda.current_stream().wait_stream(side)
        else:
            if graph is None:
                graph = torch.cuda.CUDAGraph()
                with torch.cuda.graph(graph):
                    body()
            graph.replay()

        sa.step()
        sb.step()

    torch.cuda.synchronize()
    near = all(
        torch.allclose(x, y, rtol=1e-3, atol=1e-4)
        for x, y in zip(
            snapshot(a, pa) + snapshot(adam_a, fa),
            snapshot(b, pb) + snapshot(adam_b, fb),
        )
    )
    assert near != frozen


def test_padded_unique():
    g = torch.Generator().manual_seed(0)
    pts = torch.randint(0, 500, (2000,), generator=g)
    want, want_inv = torch.unique(pts, return_inverse=True)
    uniq, inv, count = ft._padded_unique(pts, 600)
    assert int(count) == len(want)
    assert torch.equal(uniq[: len(want)], want) and torch.equal(inv, want_inv)
    assert torch.equal(uniq[inv], pts)
    # over the cap the count says so, and nothing is written out of bounds
    _, inv, count = ft._padded_unique(pts, 100)
    assert int(count) == len(want) and int(inv.max()) == 99


def test_unique_bounds():
    # the expected count of a batch's endpoints matches the drawn one, and the cap sits above every draw
    rng = np.random.default_rng(0)
    X = blobs(3000, 10)
    _, ei, ej, _ = fuzzy_knn_graph(X, 15, device="cpu")
    ei, ej = torch.as_tensor(ei), torch.as_tensor(ej)
    eb = 4096
    mean, cap = ft._unique_bounds(ei, ej, len(X), eb)
    drawn = [
        len(torch.unique(torch.cat([ei[s], ej[s]])))
        for s in (torch.as_tensor(rng.integers(0, len(ei), eb)) for _ in range(20))
    ]
    assert abs(np.mean(drawn) - mean) < 4 * np.std(drawn) + 2
    assert cap > max(drawn) and cap <= min(2 * eb, len(X))


def test_graphs_only_where_supported(monkeypatch):
    cuda = torch.device("cuda")
    flow = ft.Flow(12, 2, 16)
    cfg = ft.raw_recipe(device="cpu")
    assert ft._graphs_on(cfg, cuda, flow, False)
    assert not ft._graphs_on(cfg, torch.device("cpu"), flow, False)
    assert not ft._graphs_on(cfg, cuda, flow, True)
    assert not ft._graphs_on(ft.raw_recipe(cudagraphs=False), cuda, flow, False)
    assert not ft._graphs_on(ft.raw_recipe(optim="adam"), cuda, flow, False)
    monkeypatch.setenv("FLODR_NO_GRAPHS", "1")
    assert not ft._graphs_on(cfg, cuda, flow, False)
    monkeypatch.delenv("FLODR_NO_GRAPHS")
    # a hook would run at capture only
    flow.layers[0].net[0].weight.register_hook(lambda g: g)
    assert not ft._graphs_on(cfg, cuda, flow, False)
    with pytest.warns(RuntimeWarning, match="keeps the step eager"):
        ft._graphs_on(ft.raw_recipe(cudagraphs=True), cuda, flow, False)


def exact_knn(X, k):
    d, i = cKDTree(X).query(X, k=k + 1)
    return d[:, 1:], i[:, 1:]


@pytest.mark.parametrize("device", DEVICES)
def test_knn_gpu_is_exact(device):
    torch.set_num_threads(2)
    X = blobs(3000, 20)
    # duplicated rows tie at distance 0, in index order
    X[100] = X[7]
    X[2500] = X[7]
    d, i = knn_gpu(X, 15, device)
    dr, ir = exact_knn(X, 15)
    np.testing.assert_allclose(d, dr, rtol=1e-6, atol=1e-6)
    assert (i == ir).mean() > 0.999
    assert list(i[7, :2]) == [100, 2500] and list(i[100, :2]) == [7, 2500]
    # small blocks merge across the reference chunks to the same graph
    d2, i2 = knn_gpu(X, 15, device, cells=2**18)
    assert np.array_equal(i2, i) and np.array_equal(d2, d)


@pytest.mark.parametrize("device", DEVICES)
def test_knn_gpu_uncentred(device):
    # an offset of 30 moved half the rows' neighbour sets when the fp32 GEMM ran on the data as given
    torch.set_num_threads(2)
    X = blobs(4000, 50) + 30.0
    d, i = knn_gpu(X, 15, device)
    dr, ir = exact_knn(X, 15)
    np.testing.assert_allclose(d, dr, rtol=1e-6)
    differ = [r for r in range(len(X)) if set(i[r]) != set(ir[r])]
    assert all(np.allclose(d[r], dr[r], rtol=1e-6) for r in differ)
    assert len(differ) < 5


def test_knn_gpu_failure_is_visible(monkeypatch):
    import flodr.data as fd

    def broken(*a, **kw):
        raise RuntimeError("CUDA out of memory")

    monkeypatch.setattr(fd, "knn_gpu", broken)
    info = {}
    with pytest.warns(RuntimeWarning, match="GPU kNN failed"):
        d, i = knn_search(blobs(500, 12), 15, device="cuda", info=info)
    assert info["knn"] in ("faiss_exact", "usearch_exact") and i.shape == (500, 15)


def test_backends_recorded():
    m = FloDR(device="cpu", max_iter=20, verbose=False, density=False).fit(
        blobs(300, 12)
    )
    assert (
        m.knn_backend_ in ("faiss_exact", "usearch_exact") and m.step_mode_ == "eager"
    )
    info = {}
    knn_search(blobs(300, 12), 15, exact=False, device="cpu", info=info)
    assert info["knn"] == "hnsw"


def test_fit_restores_threads_and_determinism(monkeypatch):
    # a CPU fit's thread count and deterministic kernels hold for that fit only
    prev = torch.get_num_threads()
    torch.set_num_threads(3)
    torch.use_deterministic_algorithms(False)

    try:
        FloDR(
            device="cpu",
            max_iter=10,
            verbose=False,
            density=False,
            advanced=dict(threads=2),
        ).fit(blobs(200, 12))
        assert torch.get_num_threads() == 3
        assert not torch.are_deterministic_algorithms_enabled()
    finally:
        torch.set_num_threads(prev)

    # the default count stays within OMP_NUM_THREADS and the process's CPU affinity
    monkeypatch.setenv("OMP_NUM_THREADS", "1")
    assert ft.default_threads() == 1
    monkeypatch.delenv("OMP_NUM_THREADS")
    assert ft.default_threads() <= len(os.sched_getaffinity(0))


def test_cpu_knn_unchanged():
    # on the CPU the graph is still exact FAISS (or usearch exact) up to EXACT_KNN_MAX cells
    X = blobs(800, 12)
    d, i = knn_search(X, 15, device="cpu")
    dr, ir = exact_knn(X, 15)
    assert (i == ir).mean() > 0.999 and np.allclose(d, dr, rtol=1e-4, atol=1e-4)


def short_gpu(**adv):
    return FloDR(
        device="cuda",
        random_state=0,
        max_iter=60,
        verbose=False,
        density=False,
        advanced=dict(compile_edges=False, graph_compile=False, **adv),
    )


def recall(X, Y, k=15):
    a, b = exact_knn(X, k)[1], exact_knn(Y, k)[1]
    return np.mean([len(set(p) & set(q)) for p, q in zip(a, b)]) / k


@GPU
@pytest.mark.parametrize("edge_batch", [0, 2048], ids=["full", "edges"])
def test_graph_fit_matches_eager(edge_batch, monkeypatch):
    replays = []
    real = ft._StepGraphs.__call__

    def counted(self, key):
        replays.append(self.graph is not None and key == self.key)
        return real(self, key)

    monkeypatch.setattr(ft._StepGraphs, "__call__", counted)
    X = blobs(1500, 20)
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        g = short_gpu(edge_batch=edge_batch).fit(X)
    assert not [w for w in caught if "CUDA-graph" in str(w.message)]
    assert sum(replays) >= 40
    e = short_gpu(edge_batch=edge_batch, cudagraphs=False).fit(X)
    assert np.isfinite(g.embedding_).all() and g.roundtrip_ < 1e-3
    assert abs(recall(X, g.embedding_) - recall(X, e.embedding_)) < 0.05


@GPU
@pytest.mark.parametrize("edge_batch", [0, 2048], ids=["full", "edges"])
def test_graph_skips_nonfinite_steps(edge_batch, monkeypatch):
    # NaN through a buffer the graph reads, on a replayed step before and after the Fourier branch
    # switches on and on an eager warm-up step: each step leaves the parameters bitwise as they were
    poison_at, seen = {10, 31, 45}, {}

    class Poisoned(ft.Flow):
        def __init__(self, *a, **kw):
            super().__init__(*a, **kw)
            self.register_buffer("poison", torch.ones(()))

        def forward_with_logdet(self, x):
            return super().forward_with_logdet(x * self.poison)

    flows = []

    class Rng:
        def __init__(self, rng):
            self.rng, self.it = rng, 0

        def choice(self, *a, **kw):
            fl = flows[-1]
            snap = [p.detach().clone() for p in fl.parameters()]
            seen[self.it] = snap
            fl.poison.fill_(float("nan") if self.it in poison_at else 1.0)
            self.it += 1
            return self.rng.choice(*a, **kw)

    real_flow = Poisoned.__init__

    def record(self, *a, **kw):
        real_flow(self, *a, **kw)
        flows.append(self)

    monkeypatch.setattr(Poisoned, "__init__", record)
    monkeypatch.setattr(ft, "Flow", Poisoned)
    import flodr.estimator as fe

    real = fe.train_flodr
    monkeypatch.setattr(
        fe,
        "train_flodr",
        lambda Z, ei, ej, cfg, rng, **kw: real(Z, ei, ej, cfg, Rng(rng), **kw),
    )
    m = short_gpu(edge_batch=edge_batch).fit(blobs(1500, 20))
    flows[-1].poison.fill_(1.0)
    assert np.isfinite(m.embedding_).all()
    for t in poison_at:
        assert same(seen[t], seen[t + 1]), t
    assert not same(seen[20], seen[21])


@GPU
def test_graph_capture_failure_falls_back(monkeypatch):
    def broken(*a, **kw):
        raise RuntimeError("capture unsupported")

    monkeypatch.setattr(torch.cuda, "graph", broken)
    X = blobs(1500, 20)
    with pytest.warns(RuntimeWarning, match="continues eager"):
        m = short_gpu().fit(X)
    assert np.isfinite(m.embedding_).all() and m.roundtrip_ < 1e-3


@GPU
def test_gpu_fit_graph_is_exact_knn():
    X = blobs(3000, 20)
    m = short_gpu().fit(X)
    _, ir = exact_knn(X, 15)
    assert (m.knn_idx_ == ir).mean() > 0.999
    assert copy.deepcopy(m).knn_dist_.dtype == np.float64


def test_score_unpadded_is_the_joint():
    # unpadded fits score as before, bit for bit: the flow's log density at the point's coordinates
    m = FloDR(device="cpu", max_iter=20, verbose=False, advanced=dict(threads=2)).fit(
        blobs(300, 40)
    )
    assert m._n_pad == 0
    z = torch.from_numpy(m._to_coords(blobs(50, 40, seed=1))).float()
    with torch.no_grad():
        want = m.flow_.log_prob(z).numpy()
    assert np.array_equal(m.score_samples(blobs(50, 40, seed=1)), want)


def test_score_marginalises_the_padding():
    # a padded fit scores every point, training cell or new, through the same marginal over its padding
    # columns: training and held-out cells of one population score alike, and a point's score does not
    # depend on what it is scored with
    X = blobs(1200, 12)
    m = FloDR(device="cpu", max_iter=60, verbose=False, advanced=dict(threads=2)).fit(
        X[:600]
    )
    assert m._n_pad == 18
    train, held = m.score_samples(X[:600]), m.score_samples(X[600:])
    assert ks_2samp(train, held).pvalue > 0.05
    # the same draws for every point; only BLAS blocking moves the last bit with the batch
    np.testing.assert_allclose(held[:30], m.score_samples(X[600:630]), rtol=1e-6)


@GPU
def test_pad_marginal_at_default_length():
    # training and held-out cells score alike, and 128 padding draws sit near 2,048
    X = blobs(2000, 12)
    m = FloDR(device="cuda", random_state=0, verbose=False).fit(X[:1500])
    train, held = m.score_samples(X[:1500]), m.score_samples(X[1500:])
    assert ks_2samp(train, held).pvalue > 0.05
    z = torch.from_numpy(m._to_coords(X[1500:])).float().to("cuda")
    with torch.no_grad():
        many = m._pad_marginal(z, 2048).cpu().numpy()
    assert abs((held - many).mean()) < 0.8


def step_losses(monkeypatch, graphs, frozen=False):
    # the loss of every step of a short default fit: from the scaler on the eager path, from the graph's
    # own output on the graphed one
    losses = []
    scale = torch.amp.GradScaler.scale

    def eager(self, outputs):
        losses.append(outputs.detach().clone())
        return scale(self, outputs)

    call = ft._StepGraphs.__call__

    def graphed(self, key):
        call(self, key)
        losses.append(self.loss.clone())

    monkeypatch.setattr(torch.amp.GradScaler, "scale", eager)
    monkeypatch.setattr(ft._StepGraphs, "__call__", graphed)

    if frozen:
        real = ft.Muon.step_graph
        monkeypatch.setattr(
            ft.Muon,
            "step_graph",
            lambda self, ok, lr, mom: real(self, ok, lr, torch.full_like(mom, 0.95)),
        )

    adv = None if graphs else dict(cudagraphs=False)
    FloDR(
        device="cuda",
        random_state=0,
        max_iter=200,
        verbose=False,
        density=False,
        advanced=adv,
    ).fit(blobs(1500, 20))
    monkeypatch.undo()

    return torch.stack([v.float().reshape(()) for v in losses]).cpu().numpy()


@GPU
@pytest.mark.parametrize("frozen", [False, True], ids=["buffers", "frozen"])
def test_graph_loss_follows_eager(monkeypatch, frozen):
    # the default fit's graphed step draws what the eager one draws, so the two losses agree to float
    # rounding for tens of steps before chaos parts them (paul15: 2e-6 over 25 steps). A scheduled
    # scalar frozen at capture, as momentum was (review B1), parts them within twenty
    eager = step_losses(monkeypatch, False)
    graph = step_losses(monkeypatch, True, frozen)
    rel = np.abs(graph[:25] - eager[:25]) / np.abs(eager[:25])
    assert (rel.max() < 1e-4) != frozen


def gpu_fit(X, **adv):
    return FloDR(
        device="cuda",
        random_state=0,
        max_iter=60,
        verbose=False,
        density=False,
        advanced=adv or None,
    ).fit(X)


@GPU
@pytest.mark.parametrize(
    "adv",
    [
        dict(cudagraphs=False),
        dict(cudagraphs=False, edge_batch=2048, compile_edges=False),
    ],
    ids=["full", "edges"],
)
def test_eager_gpu_fit_repeats(adv):
    # the graphed step is not bitwise reproducible between runs; the eager one is, given the edge-batched
    # step uncompiled
    X = blobs(1500, 20)
    a, b = gpu_fit(X, **adv), gpu_fit(X, **adv)
    assert a.step_mode_ == "eager" and np.array_equal(a.embedding_, b.embedding_)


@GPU
def test_no_graphs_switch_repeats(monkeypatch):
    monkeypatch.setenv("FLODR_NO_GRAPHS", "1")
    X = blobs(1500, 20)
    a, b = gpu_fit(X), gpu_fit(X)
    assert a.step_mode_ == "eager" and np.array_equal(a.embedding_, b.embedding_)
