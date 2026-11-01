# The linearity term (smooth mode, opt-in) on the CPU. Off, the default, it leaves no trace in a
# fit; on, it brings the decode deviation at the density's noise scale down, as faithful_scale
# reads it. That the default fit is bitwise the frozen release's is checked against the
# release's own source, which cannot be imported next to this one, outside the suite.

import numpy as np
import pytest

from flodr import FloDR, raw_recipe
from flodr.model import ENERGY_GRID


def blobs(n, d, k=4, seed=0):
    rng = np.random.default_rng(seed)
    c = rng.normal(size=(k, d)) * 4
    return (c[rng.integers(0, k, n)] + rng.normal(size=(n, d))).astype(np.float32)


def fit(X, max_iter, density=True, **adv):
    return FloDR(
        device="cpu",
        random_state=0,
        max_iter=max_iter,
        density=density,
        verbose=False,
        advanced=dict(threads=4, **adv),
    ).fit(X)


def test_defaults():
    cfg = raw_recipe()
    assert cfg.w_lin == 0.0 and cfg.lin_start == 0.5 and cfg.lin_ramp == 0.1
    assert cfg.lin_h == (0.5, 1.0) and cfg.lin_batch == 1024
    assert cfg.lin_every == 2 and cfg.lin_tau == 1.0
    assert cfg.tail_select == "auto"
    # (n_tail, cond_hid, iters, n_scale), the depth among the choices
    assert all(len(c) == 4 and c[3] == 20 for c in ENERGY_GRID)
    assert len({c[0] for c in ENERGY_GRID}) > 1


# a term that never reaches its start draws nothing from the fit's generators, and without the
# density there is no noise to step at, so no term
def test_off_leaves_no_trace():
    X = blobs(300, 20)
    off = fit(X, 40).embedding_
    np.testing.assert_array_equal(off, fit(X, 40, w_lin=1.0, lin_start=1.0).embedding_)
    assert not np.array_equal(off, fit(X, 40, w_lin=1.0).embedding_)
    np.testing.assert_array_equal(
        fit(X, 40, density=False).embedding_,
        fit(X, 40, density=False, w_lin=1.0).embedding_,
    )


# 200 steps on blobs: without the term the map bends inside the noise, rho(1) about 0.7 at the
# median; with it about 0.1, every cell at or below 0.3
@pytest.mark.parametrize(
    "lin",
    [{}, dict(lin_every=4, lin_batch=256, lin_h=(1.0,), lin_tau=0.5)],
    ids=["default", "every4"],
)
def test_lowers_rho(lin):
    X = blobs(500, 40)
    base = fit(X, 200).faithful_scale(h=(1.0,), n_steps=4)
    on = fit(X, 200, w_lin=1.0, **lin).faithful_scale(h=(1.0,), n_steps=4)
    assert base["rho_median"][0] > 0.4
    assert on["rho_median"][0] < 0.25 and (on["rho"][:, 0] <= 0.3).mean() > 0.9
