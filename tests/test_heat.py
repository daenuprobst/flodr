# FloDR(axes="heat") on the CPU with short fits: the rotated start keeps the latent map exact, transform consistent
# with the layout, pickling and the density intact, and the heat coordinates agree between the CPU and GPU paths
import pickle

import numpy as np
import pytest
import torch

from flodr import FloDR
from flodr.data import RotatedWhitening
from flodr.heat import heat_coordinates, heat_rotation

N, ITERS = 600, 40


def branches(n, d, seed=0):
    # three branches leaving a common root, plus noise
    rng = np.random.default_rng(seed)
    t = rng.uniform(0, 1, n)
    b = rng.integers(0, 3, n)
    dirs = rng.normal(size=(3, d)) * 3
    return (t[:, None] * dirs[b] + rng.normal(size=(n, d)) * 0.3).astype(np.float32), b


def short(**kw):
    return FloDR(
        device="cpu", random_state=0, max_iter=ITERS, verbose=False, axes="heat", **kw
    )


@pytest.fixture(scope="module", params=[40, 10], ids=["wide", "padded"])
def fitted(request):
    X, b = branches(N, request.param)
    return short().fit(X), X


def test_fit_and_rotation(fitted):
    m, X = fitted
    assert m.embedding_.shape == (N, 2) and np.isfinite(m.embedding_).all()
    rot = m._raw_of.__self__
    assert isinstance(rot, RotatedWhitening)
    np.testing.assert_allclose(rot.R.T @ rot.R, np.eye(rot.R.shape[0]), atol=1e-5)


def test_transform_reproduces_the_layout(fitted):
    m, X = fitted
    Y = m.transform(X[:100])
    assert Y.shape == (100, 2) and np.isfinite(Y).all()
    if X.shape[1] >= 30:
        np.testing.assert_allclose(Y, m.embedding_[:100], atol=1e-3)


def test_latent_round_trip_is_exact(fitted):
    m, X = fitted
    np.testing.assert_allclose(
        m.inverse_latent(m.transform_latent(X[:30])), X[:30], atol=1e-3
    )
    Xi = m.inverse_transform(m.embedding_[:30])
    assert Xi.shape == (30, X.shape[1]) and np.isfinite(Xi).all()


def test_density_runs(fitted):
    m, X = fitted
    s = m.score_samples(X[:50])
    assert s.shape == (50,) and np.isfinite(s).all()


def test_pickle(fitted):
    m, X = fitted
    m2 = pickle.loads(pickle.dumps(m))
    np.testing.assert_allclose(m2.transform(X[:20]), m.transform(X[:20]), atol=1e-5)


def test_three_components():
    X, _ = branches(N, 20)
    m = short(n_components=3).fit(X)
    assert m.embedding_.shape == (N, 3) and np.isfinite(m.embedding_).all()


# heat is an option: on the benchmarks it costs class-centroid structure (2026-10-07), so variance stays the default
def test_variance_is_the_default():
    assert FloDR().axes == "variance"


def test_rotation_is_orthogonal_and_aligned():
    rng = np.random.default_rng(1)
    Z = rng.normal(size=(500, 12))
    Z -= Z.mean(0)
    T = Z[:, [3, 7]] + 0.01 * rng.normal(size=(500, 2))
    R = heat_rotation(Z, T)
    np.testing.assert_allclose(R.T @ R, np.eye(12), atol=1e-5)
    # the first two rotated directions span the target directions
    W = Z @ R[:, :2]
    c = np.linalg.lstsq(W, T - T.mean(0), rcond=None)[0]
    assert np.abs(W @ c - (T - T.mean(0))).max() < 0.1


@pytest.mark.skipif(not torch.cuda.is_available(), reason="no GPU")
def test_heat_coordinates_cpu_matches_gpu():
    from flodr.data import knn_query

    X, _ = branches(2000, 20)
    d, i = knn_query(X, None, 15)
    a = heat_coordinates(d, i, 2, None)
    b = heat_coordinates(d, i, 2, "cuda")
    s = np.sign((a * b).sum(0))
    np.testing.assert_allclose(a * s, b, atol=1e-6 * np.abs(a).max() + 1e-6)
