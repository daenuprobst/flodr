# The energy certificate and the decode-faithfulness scale on the CPU. A fitted model with its
# main flow set to the identity (or an affine map) and a known Gaussian law over the residual
# gives held-out cells of exactly known law: the right law certifies, and a law three times too
# wide along a few axes, too narrow by a matching amount along the rest, keeps the l-rank
# certified and is refused by the energy certificate. The density tail chosen by the energy
# certificate on such cells certifies.

import copy

import numpy as np
import pytest
import torch

from flodr import FloDR, fidelity
from flodr.model import CondBase, Flow

N, D, SD = 600, 40, 0.1
N_CELLS, Q = 1000, 4


def short(**kw):
    return FloDR(
        device="cpu",
        random_state=0,
        max_iter=40,
        verbose=False,
        advanced=dict(threads=4),
        **kw,
    )


@pytest.fixture(scope="module")
def base():
    # isotropic input, so the whitening and input-space distances stay near isotropic
    X = np.random.default_rng(0).standard_normal((N, D)).astype(np.float32)

    return short().fit(X)


def heads(layer):
    net = layer.net

    return [net] if isinstance(net, torch.nn.Sequential) else [net.base, net.high]


def known_law(m, var_r, affine=False):
    t = copy.deepcopy(m)
    fl = t.flow_
    g = torch.Generator().manual_seed(1)

    with torch.no_grad():
        # zero output weights make every coupling constant in its input: the identity, or
        # with random biases an affine map
        for layer in fl.layers:
            for i, net in enumerate(heads(layer)):
                net[-1].weight.zero_()
                b = net[-1].bias
                b.copy_(
                    0.3 * torch.randn(b.shape, generator=g)
                    if affine and i == 0
                    else 0 * b
                )

        k, d = fl.k, fl.d
        cond = CondBase(d, hid=16, k=k)
        cond.net[-1].bias[d - k :] = torch.log(
            torch.as_tensor(var_r, dtype=torch.float32)
        )
        Y = fl.forward_main(torch.as_tensor(t._coords))[:, :k].numpy()

    cond.fit_y_density(Y, ks=(4,))
    fl.tail, fl._cond, fl._density_args = torch.nn.ModuleList(), cond, None
    fl._density_noise = torch.full((d,), SD)

    return t


def cells(t, var_r, seed=2, n=N_CELLS):
    # the identity flow makes the latent the whitened input, so a cell is a training position
    # and a residual whose variance after the certificate's noise draw is var_r
    rng = np.random.default_rng(seed)
    k, d = t.flow_.k, t.flow_.d
    y = t._coords[rng.integers(0, N, n), :k]
    r = rng.standard_normal((n, d - k)) * np.sqrt(np.asarray(var_r) - SD**2)

    return t._to_raw(np.hstack([y, r]).astype(np.float32))


def test_correct_law_certified(base):
    v = np.ones(D - 2)
    t = known_law(base, v)
    X = cells(t, v)
    np.testing.assert_allclose(t._to_raw(t._to_coords(X)), X, atol=1e-4)
    cert = t.energy_certificate(X)
    assert cert["energy"]["certifies"] == "certified", cert["energy"]
    assert cert["precision"]["passed"] and cert["certifies"] == "certified"
    assert cert["energy"]["ranks"].shape == (N_CELLS,) and cert["n_cells"] == N_CELLS
    assert 0.02 < cert["exceedance"]["share"] < 0.09
    assert t.coverage_certificate(X)["fibre"]["certifies"] == "certified"


def test_inflated_law_refused_by_energy_not_by_lrank(base):
    # the law 3x the cells' sd along Q residual axes; along the others the cells are wider by
    # the amount that leaves their mean Mahalanobis norm, and so the l-rank, unchanged
    m = D - 2
    law = np.ones(m)
    law[:Q] = 9.0
    var_cells = np.full(m, 1.0 + Q * (1 - 1 / 9) / (m - Q))
    var_cells[:Q] = 1.0
    assert (var_cells / law).sum() == pytest.approx(m)
    t = known_law(base, law)
    X = cells(t, var_cells)
    cert = t.energy_certificate(X)
    assert cert["energy"]["certifies"] == "refused" and cert["certifies"] == "refused"
    # held-out cells sit too centrally among the draws
    assert cert["energy"]["mean"] < 0.3
    assert t.coverage_certificate(X)["fibre"]["certifies"] != "refused"


def test_affine_flow_is_faithful(base):
    t = known_law(base, np.ones(D - 2), affine=True)
    z = torch.as_tensor(t._coords[:5])
    with torch.no_grad():
        assert (t.flow_.forward_main(z) - z).abs().max() > 0.05
    out = t.faithful_scale()
    assert out["rho"].shape == (N, 5) and out["rho"].max() < 1e-3
    assert (out["hstar"] == 2.0).all() and out["hstar_median"] == 2.0


# a padded fit with a short scale-mixture tail, so the draws run through the tail couplings
@pytest.fixture(scope="module")
def fitted():
    rng = np.random.default_rng(3)
    c = rng.normal(size=(4, 10)) * 4
    X = (c[rng.integers(0, 4, 500)] + rng.normal(size=(500, 10))).astype(np.float32)
    m = short().fit(X[:400])
    m.flow_._density_args = None
    m.flow_.fit_cond_tail_cv(
        m._coords,
        seed=m._seed,
        grid=((2, 32, 200, 0, 0, 20),),
        noise_sd=m._density_sd(),
    )

    return m, X[400:]


def test_faithful_scale_shapes(fitted):
    m, Xh = fitted
    out = m.faithful_scale()
    assert out["rho"].shape == (400, 5) and out["hstar"].shape == (400,)
    assert out["positions"].shape == (400, 2) and out["rho_median"].shape == (5,)
    assert np.isfinite(out["rho"]).all() and (out["rho"] >= 0).all()
    h = (0.5, 1.0, 2.0)
    new = m.faithful_scale(Xh, h=h, n_steps=2)
    assert new["rho"].shape == (100, 3) and new["faithful"].shape == (3,)
    assert set(np.unique(new["hstar"])) <= {0.0, *h}
    np.testing.assert_allclose(new["positions"], m.transform(Xh), atol=1e-5)
    nn = copy.deepcopy(m)
    nn.flow_._density_noise = None
    with pytest.raises(ValueError, match="input noise"):
        nn.faithful_scale(Xh)


def test_energy_certificate_runs(fitted):
    m, Xh = fitted
    cert = m.energy_certificate(Xh, n_draws=49)
    u = cert["energy"]["ranks"]
    assert u.shape == (100,) and (u > 0).all() and (u < 1).all()
    assert cert["certifies"] in {"certified", "refused", "undecided"}
    assert 0 <= cert["precision"]["wild"] <= 1 and cert["positions"].shape == (100, 2)
    with pytest.raises(ValueError):
        m.energy_certificate(Xh[:, :9])


# twice the unit variance on the residual: a candidate trained for one step keeps the unit-variance
# start, too narrow, and loses to the trained one, whose tail certifies on fresh cells. In 38
# residual dims the energy ranks resolve a few percent of variance, hence 4,000 fitting cells
def test_energy_selection_certifies(base):
    v = np.full(D - 2, 2.0)
    t = known_law(base, v)
    fl = t.flow_
    fl.tail, fl._cond = torch.nn.ModuleList(), None
    fl.fit_cond_tail_cv(
        t._to_coords(cells(t, v, seed=3, n=4000)),
        seed=0,
        grid=((0, 16, 1, 0), (0, 16, 3000, 0)),
        select="energy",
        noise_sd=fl._density_noise,
        raw=fidelity._raw_affine(t),
    )
    ch = fl.cond_choice_
    assert ch["select"] == "energy" and ch["iters"] == 3000 and ch["passed"]
    assert ch["val_ks"] < 0.1 and ch["val_wild"] <= 0.03
    assert [c["eligible"] for c in ch["candidates"]] == [False, True]
    cert = t.energy_certificate(cells(t, v, seed=4))
    assert cert["certifies"] == "certified", (cert["energy"], cert["precision"])


# when no candidate passes, the best energy score is kept and the fit says so
def test_energy_selection_none_passes(base):
    v = np.full(D - 2, 2.0)
    t = known_law(base, v)
    fl = t.flow_
    fl.tail, fl._cond = torch.nn.ModuleList(), None

    with pytest.warns(RuntimeWarning, match="no density tail passed"):
        fl.fit_cond_tail_cv(
            t._to_coords(cells(t, v, seed=3)),
            seed=0,
            grid=((0, 16, 1, 0),),
            select="energy",
            noise_sd=fl._density_noise,
            raw=fidelity._raw_affine(t),
        )

    assert fl.cond_choice_["passed"] is False and fl.cond_choice_["iters"] == 1


# the precision test on the exact law: noised held-out cells and draws at their positions are
# exchangeable, so about 1% of the draws lie beyond the cells' 99th percentile
def test_precision_reference_calibrated(base):
    v = np.ones(D - 2)
    t = known_law(base, v)
    cert = t.energy_certificate(cells(t, v, seed=6))
    assert 0.003 < cert["precision"]["wild"] < 0.02, cert["precision"]
    assert cert["precision"]["passed"]


# the density's full-size passes go in chunks: within one chunk they are the single pass, and on a
# forced small chunk they agree with it
def test_chunked_passes(monkeypatch):
    import flodr.model as M

    torch.manual_seed(0)
    fl = Flow(6, n_layers=4, hid=16, k=2)

    with torch.no_grad():
        for layer in fl.layers:
            layer.net[-1].weight.normal_(0, 0.1)
            layer.net[-1].bias.normal_(0, 0.1)

    x = torch.randn(300, 6)

    with torch.no_grad():
        z_one, ld_one = fl.forward_with_logdet(x)

    cond = CondBase(6, hid=8, k=2).fit_y_density(z_one[:, :2].numpy(), ks=(4,))

    with torch.no_grad():
        lp_one = cond.y_log_prob(z_one[:, :2])

    z, ld = fl._main_latents(x)
    assert torch.equal(z, z_one) and torch.equal(ld, ld_one)
    monkeypatch.setattr(M, "_PASS_CHUNK", 64)
    z, ld = fl._main_latents(x)

    with torch.no_grad():
        lp = cond.y_log_prob(z_one[:, :2])

    torch.testing.assert_close(z, z_one, rtol=1e-6, atol=1e-6)
    torch.testing.assert_close(ld, ld_one, rtol=1e-6, atol=1e-6)
    torch.testing.assert_close(lp, lp_one, rtol=1e-6, atol=1e-6)
