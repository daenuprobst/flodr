# conformal fibre bands on a tiny CPU fit: exact fibre draws, the finite-sample rank, the
# region merge and Mondrian coverage on fresh exchangeable cells

import numpy as np
import pytest
from scipy.spatial import cKDTree

from flodr import FloDR
from flodr.conformal import (
    FibreConformal,
    MapRegions,
    conformal_bands,
    fibre_score,
    sample_fibre,
    split_threshold,
)

D, ALPHA = 40, 0.1
CENTRES = np.random.default_rng(0).normal(size=(4, D)) * 4


def blobs(n, seed):
    # fixed centres, so every seed is a fresh draw from one law
    rng = np.random.default_rng(seed)
    lab = rng.integers(0, len(CENTRES), n)

    return (CENTRES[lab] + rng.normal(size=(n, D))).astype(np.float32)


def short_fit(X):
    m = FloDR(
        device="cpu",
        random_state=0,
        max_iter=40,
        verbose=False,
        advanced=dict(threads=4),
    ).fit(X)

    # one short scale-mixture candidate instead of the full CV grid, which takes a minute
    m.flow_._density_args = None
    m.flow_.fit_cond_tail_cv(m._coords, seed=m._seed, grid=((0, 32, 300, 0, 0, 20),))

    return m


@pytest.fixture(scope="module")
def est():
    m = short_fit(blobs(300, 1))
    m.score_samples(blobs(5, 2))

    return m


def test_draws_reembed_at_their_pixel(est):
    emb = est.embedding_
    lo, hi = emb.min(0), emb.max(0)
    spacing = np.median(cKDTree(emb).query(emb, k=2)[0][:, 1])

    # pixel centres of the occupied 14 x 14 bins
    reg = MapRegions(lo, hi, 14)
    occ = np.unique(reg._bins(emb))
    ix = np.stack(np.unravel_index(occ, (14, 14)), 1)
    Y = (lo + (ix + 0.5) / 14 * (hi - lo + 1e-9)).astype(np.float32)

    Xd, l_d, Yd = sample_fibre(est, Y, n_draws=32, seed=3)
    assert Xd.shape == (32, len(Y), D) and np.isfinite(l_d).all()

    err = np.sqrt(((Yd - Y[None]) ** 2).sum(-1))
    assert err.max() <= 1e-4 * spacing
    assert (reg._bins(Yd.reshape(-1, 2)).reshape(32, -1) == occ[None]).all()

    # back through the float32 whitening, as a user would re-embed them
    Yt = est.transform(Xd.reshape(-1, D)).reshape(Yd.shape)
    assert np.sqrt(((Yt - Y[None]) ** 2).sum(-1)).max() <= 1e-3 * spacing

    # a draw's l is the same l the estimator gives an input row
    l_x, _ = fibre_score(est, Xd[0])
    assert np.allclose(l_x, l_d[0], atol=1e-3)


def test_threshold_rank():
    s = np.arange(40.0)[::-1]

    # k = ceil((n+1) alpha) - 1, lower tail
    assert split_threshold(s[:19], 0.1) == np.sort(s[:19])[0]
    assert split_threshold(s[:20], 0.1) == np.sort(s[:20])[1]
    assert split_threshold(s[:29], 0.1) == np.sort(s[:29])[1]
    assert split_threshold(s[:8], 0.1) == -np.inf

    # leave-one-out over every rotation is the exact finite-sample coverage
    rng = np.random.default_rng(0)
    for n in range(9, 80):
        pool = rng.normal(size=n + 1)
        cov = np.mean(
            [
                pool[i] >= split_threshold(np.delete(pool, i), ALPHA)
                for i in range(n + 1)
            ]
        )
        assert 1 - ALPHA <= cov <= 1 - ALPHA + 1 / (n + 1) + 1e-12


def test_region_merge_and_thresholds(est):
    X_cal = blobs(400, 5)
    cp = FibreConformal(est, ALPHA, min_cells=20).fit(X_cal)
    reg = cp.regions_

    assert reg.n_regions_ > 1 and (cp.n_cal_ >= 20).all()
    assert (reg.assign(cp.y_cal_) == reg.labels_).all()

    for g in range(reg.n_regions_):
        assert cp.q_hat_[g] == split_threshold(cp.l_cal_[reg.labels_ == g], ALPHA)

    # a position in a bin without calibration cells takes its nearest cell's region
    emb = est.embedding_
    lo, hi = emb.min(0), emb.max(0)
    empty = np.nonzero(reg.bin_region_ < 0)[0]
    assert len(empty)
    ix = np.stack(np.unravel_index(empty, (14, 14)), 1)
    Y_empty = lo + (ix + 0.5) / 14 * (hi - lo + 1e-9)
    nearest = cKDTree(cp.y_cal_).query(Y_empty)[1]
    assert (reg.assign(Y_empty) == reg.labels_[nearest]).all()

    # fewer cells than min_cells makes one region
    assert MapRegions(emb.min(0), emb.max(0), 14, 1000).fit(emb).n_regions_ == 1


def test_coverage_on_exchangeable_cells(est):
    # test cells scored like the calibration cells, with one draw of the density's input noise
    pool = blobs(1000, 7)
    l_pool, y_pool = fibre_score(est, pool, noised=True)
    rng = np.random.default_rng(0)
    covs = []

    for _ in range(40):
        perm = rng.permutation(len(pool))
        cal, tst = perm[:500], perm[500:]
        cp = FibreConformal(est, ALPHA).fit(pool[cal])
        covs.append(np.mean(l_pool[tst] >= cp.threshold(y_pool[tst])))

    assert 1 - ALPHA - 0.015 <= np.mean(covs) <= 1 - ALPHA + 0.05


def test_bands_are_kept_draw_quantiles(est):
    X_cal, Y = blobs(300, 9), est.embedding_[:25]
    cp = FibreConformal(est, ALPHA).fit(X_cal)
    out = cp.bands(Y, n_draws=64, q=(0, 100), seed=4)

    Xd, l_d, _ = sample_fibre(est, Y, n_draws=64, seed=4)
    keep = l_d >= cp.threshold(Y)[None]
    assert np.allclose(out["kept"], keep.mean(0))

    for j in np.nonzero(keep.any(0))[0]:
        kept = Xd[keep[:, j], j]
        assert np.allclose(out["lo"][j], kept.min(0), atol=1e-5)
        assert np.allclose(out["hi"][j], kept.max(0), atol=1e-5)

    inner = cp.bands(Y, n_draws=64, q=(5, 95), seed=4)
    ok = keep.any(0)
    assert (inner["lo"][ok] >= out["lo"][ok] - 1e-6).all()
    assert (inner["hi"][ok] <= out["hi"][ok] + 1e-6).all()

    # a linear readout is applied to the draws before the quantiles
    W = np.random.default_rng(1).normal(size=(D, 3)).astype(np.float32)
    gene = conformal_bands(est, X_cal, Y, n_draws=64, q=(0, 100), seed=4, readout=W)
    for j in np.nonzero(ok)[0]:
        g = Xd[keep[:, j], j] @ W
        assert np.allclose(gene["lo"][j], g.min(0), atol=1e-4)
        assert np.allclose(gene["hi"][j], g.max(0), atol=1e-4)


def test_padded_draws_are_scored_as_input_rows():
    # 10 features are padded to 30 with noise columns that new rows see as zeros
    X = blobs(300, 11)[:, :10]
    m = short_fit(X)
    assert m._n_pad == 20

    Xd, l_d, Yd = sample_fibre(m, m.embedding_[:20], n_draws=32, seed=0)
    assert Xd.shape == (32, 20, 10)
    assert np.abs(Yd - m.embedding_[:20][None]).max() <= 1e-4

    l_x, _ = fibre_score(m, Xd.reshape(-1, 10))
    assert np.allclose(l_x, l_d.ravel(), atol=1e-3)

    cp = FibreConformal(m, ALPHA).fit(blobs(300, 12)[:, :10])
    assert cp.bands(m.embedding_[:20], n_draws=64)["kept"].mean() > 0.5
