# scikit-learn's own estimator checks. The expected failures are properties of the method, not
# gaps in the wrapper: several checks force n_components=1, and a one-dimensional layout leaves
# the flow nothing to route the residual through; inputs under 30 features carry per-cell noise
# columns at fit that new points do not, so transform does not reproduce fit_transform there.

import warnings

from sklearn.utils.estimator_checks import parametrize_with_checks

from flodr import FloDR

ONE_D = "the check forces n_components=1; a 1-D layout is not invertible"
PADDED = "inputs under 30 features carry per-cell noise columns at fit that new points do not"
XFAIL = {
    **dict.fromkeys(
        [
            "check_fit2d_1sample",
            "check_fit2d_1feature",
            "check_fit2d_predict1d",
            "check_methods_subset_invariance",
            "check_methods_sample_order_invariance",
            "check_dont_overwrite_parameters",
        ],
        ONE_D,
    ),
    **dict.fromkeys(
        ["check_transformer_general", "check_transformer_data_not_an_array"], PADDED
    ),
}


@parametrize_with_checks(
    [FloDR(max_iter=20, device="cpu", verbose=False)],
    expected_failed_checks=lambda est: XFAIL,
)
def test_sklearn_checks(estimator, check):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        check(estimator)
