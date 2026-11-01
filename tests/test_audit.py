# The hidden-information audit on a small synthetic map, CPU: a label carried by an input axis the
# map drops is found and certified, a label the map shows carries about nothing, unlabelled cells
# are skipped, and an AnnData-style object gets its fields written back. The validated levels are
# in the audit's own pre-registration, not checked here.

import warnings

import matplotlib
import numpy as np
import pandas as pd
import pytest

from flodr import AuditResult, audit, plot_audit

matplotlib.use("Agg")
N = 1500


def cells(seed=0):
    rng = np.random.default_rng(seed)
    X = rng.normal(size=(N, 6))
    # the map keeps the first two axes; the third is dropped
    hidden = (X[:, 2] + 0.3 * rng.normal(size=N) > 0).astype(int)
    shown = np.where(X[:, 0] + 0.3 * rng.normal(size=N) > 0, "a", "b")
    return X[:, :2].copy(), X, hidden, shown


@pytest.fixture(scope="module")
def hidden_run():
    Y, X, hidden, _ = cells()
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        res = audit(Y, X=X, labels=hidden, verbose=False)
    return res, w


def test_hidden_label_found(hidden_run):
    res, w = hidden_run
    assert isinstance(res, AuditResult) and res.n == N and res.n_test == N
    # below the validated sizes it says so, and records it
    assert any("20,000" in str(x.message) for x in w) and len(res.notes) == 2
    assert res.level > 0.2 and len(res.level_splits) == 5
    assert res.certified is True and res.p_global <= 0.05
    assert res.band_positive > 0.8 and res.acc_input > res.acc_map + 0.15
    assert res.field.shape == (N,) and np.isfinite(res.lower).all()
    assert "nats" in str(res) and "field" not in res.summary()


def test_shown_label_carries_little():
    Y, X, _, shown = cells(1)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        res = audit(Y, X=X, labels=shown, test=False, verbose=False)
    assert res.certified is None and np.isnan(res.p_global)
    assert abs(res.level) < 0.05 and res.classes == ["a", "b"]


class _Adata:
    def __init__(self, Y, X, lab):
        self.obsm = {"X_umap": Y, "X_pca": X}
        self.obs = pd.DataFrame({"cond": lab})
        self.uns = {}


def test_anndata_and_missing_labels():
    Y, X, _, shown = cells(2)
    lab = shown.astype(object)
    lab[:100] = None
    ad = _Adata(Y, X, lab)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        res = audit(ad, key="cond", test=False, verbose=False)
    assert res.n == N - 100 and not res.scored[:100].any()
    assert np.isnan(ad.obs["hidden_cond"].to_numpy()[:100]).all()
    assert np.isfinite(ad.obs["hidden_cond_lower"].to_numpy()[100:]).all()
    assert ad.uns["hidden_audit"]["cond"]["n"] == N - 100
    with pytest.raises(ValueError):
        audit(Y[:, :1], X=X, labels=shown, verbose=False)


def test_plot(hidden_run):
    axes = plot_audit(hidden_run[0])
    assert len(axes) == 2
