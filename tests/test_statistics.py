import numpy as np
import pytest
from patsy import dmatrix
from isovita import statistics as s, splines, DiscreteConfig


def test_eligible_cuts_require_distinct_age_and_minimum():
    cuts, minimum = s._valid_change_point_positions(
        np.array([1, 1, 2, 2, 3, 3, 4, 4]), min_samples_side=2, min_fraction_side=0.3
    )
    np.testing.assert_array_equal(cuts, [4])
    assert minimum == 3
    cuts, _ = s._valid_change_point_positions(np.arange(5), min_samples_side=3, min_fraction_side=0)
    assert not len(cuts)


def test_step_f_and_changepoint_independent_ols():
    age = np.arange(12, dtype=float)
    y = np.array([0.1, -0.2, 0.3, 0.0, 0.1, -0.1, 2.0, 2.2, 1.9, 2.3, 2.1, 2.0])
    config = DiscreteConfig(min_samples_side=3, min_fraction_side=0)
    cuts = np.arange(3, 10)
    rss0 = ((y - y.mean()) ** 2).sum()
    rss = np.array(
        [((y[:j] - y[:j].mean()) ** 2).sum() + ((y[j:] - y[j:].mean()) ** 2).sum() for j in cuts]
    )
    fs = (rss0 - rss) / (rss / (len(y) - 2))
    out = s._step_stats_for_sorted_values(y, cuts)
    np.testing.assert_allclose(out["f_stat"], fs, rtol=1e-11)
    shuffled = np.random.default_rng(1).permutation(len(age))
    fit = s._fit_discrete_step_statistic(age[shuffled], y[shuffled], config=config)
    j = cuts[np.argmax(fs)]
    assert fit["best_change_age"] == (age[j - 1] + age[j]) / 2
    assert fit["n_left"] == j and fit["n_right"] == len(y) - j
    np.testing.assert_allclose(fit["max_f_stat"], max(fs), rtol=1e-11)
    np.testing.assert_allclose(
        s._max_step_f_statistics_from_sorted_matrix(np.vstack([y, -y]), cuts), max(fs), rtol=1e-11
    )


@pytest.mark.parametrize(
    "age,y",
    [(np.ones(12), np.arange(12)), (np.arange(12), np.ones(12)), (np.arange(2), np.arange(2))],
)
def test_untestable(age, y):
    assert not s._fit_discrete_step_statistic(age, y, config=DiscreteConfig())["tested"]
    assert np.isnan(s._fit_nested_spline_test(age, y, spline_df=3, spline_degree=3)["f_stat"])


def test_continuous_independent_projection():
    age = np.linspace(0, 80, 30)
    y = np.sin(age / 25) + np.random.default_rng(9).normal(0, 0.15, 30)
    X = np.asarray(dmatrix("bs(age, df=3, degree=3, include_intercept=False)", {"age": age}))
    Q, _ = np.linalg.qr(X)
    rss0 = ((y - y.mean()) ** 2).sum()
    rss1 = np.sum((y - Q @ (Q.T @ y)) ** 2)
    rank = np.linalg.matrix_rank(X)
    f = ((rss0 - rss1) / (rank - 1)) / (rss1 / (len(age) - rank))
    actual = s._fit_nested_spline_test(age, y, spline_df=3, spline_degree=3)
    assert actual["df_diff"] == 3 and actual["df_resid_full"] == 26
    np.testing.assert_allclose(actual["f_stat"], f, rtol=1e-12)
    w = splines._build_fixed_age_spline_workspace(age, spline_df=3, spline_degree=3)
    np.testing.assert_allclose(s._spline_f_statistics_from_sorted_matrix(y, w), [f], rtol=1e-12)


def test_constant_and_perfect_vectorized_statistics():
    age = np.arange(20, dtype=float)
    w = splines._build_fixed_age_spline_workspace(age, spline_df=3, spline_degree=3)
    out = s._spline_f_statistics_from_sorted_matrix(np.vstack([np.ones(20), age]), w)
    assert out[0] == 0 and np.isinf(out[1])
    assert np.isinf(
        s._max_step_f_statistics_from_sorted_matrix(np.r_[np.zeros(10), np.ones(10)], [10])[0]
    )
