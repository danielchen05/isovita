import numpy as np
import pytest
from patsy import dmatrix
from isovita import statistics as s, splines, DiscreteConfig


def test_eligible_cuts_require_distinct_time_and_minimum():
    cuts, minimum = s._valid_change_point_positions(
        np.array([1, 1, 2, 2, 3, 3, 4, 4]), min_samples_side=2, min_fraction_side=0.3
    )
    np.testing.assert_array_equal(cuts, [4])
    assert minimum == 3
    cuts, _ = s._valid_change_point_positions(np.arange(5), min_samples_side=3, min_fraction_side=0)
    assert not len(cuts)


def test_step_f_and_changepoint_independent_ols():
    time = np.arange(12, dtype=float)
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
    shuffled = np.random.default_rng(1).permutation(len(time))
    fit = s._fit_discrete_step_statistic(time[shuffled], y[shuffled], config=config)
    j = cuts[np.argmax(fs)]
    assert fit["best_change_time"] == (time[j - 1] + time[j]) / 2
    assert fit["n_left"] == j and fit["n_right"] == len(y) - j
    np.testing.assert_allclose(fit["max_f_stat"], max(fs), rtol=1e-11)
    np.testing.assert_allclose(
        s._max_step_f_statistics_from_sorted_matrix(np.vstack([y, -y]), cuts), max(fs), rtol=1e-11
    )


@pytest.mark.parametrize(
    "time,y",
    [(np.ones(12), np.arange(12)), (np.arange(12), np.ones(12)), (np.arange(2), np.arange(2))],
)
def test_untestable(time, y):
    assert not s._fit_discrete_step_statistic(time, y, config=DiscreteConfig())["tested"]
    assert np.isnan(s._fit_nested_spline_test(time, y, spline_df=3, spline_degree=3)["f_stat"])


def test_continuous_independent_projection():
    time = np.linspace(0, 80, 30)
    y = np.sin(time / 25) + np.random.default_rng(9).normal(0, 0.15, 30)
    X = np.asarray(dmatrix("bs(time, df=3, degree=3, include_intercept=False)", {"time": time}))
    Q, _ = np.linalg.qr(X)
    rss0 = ((y - y.mean()) ** 2).sum()
    rss1 = np.sum((y - Q @ (Q.T @ y)) ** 2)
    rank = np.linalg.matrix_rank(X)
    f = ((rss0 - rss1) / (rank - 1)) / (rss1 / (len(time) - rank))
    actual = s._fit_nested_spline_test(time, y, spline_df=3, spline_degree=3)
    assert actual["df_diff"] == 3 and actual["df_resid_full"] == 26
    np.testing.assert_allclose(actual["f_stat"], f, rtol=1e-12)
    w = splines._build_fixed_time_spline_workspace(time, spline_df=3, spline_degree=3)
    np.testing.assert_allclose(s._spline_f_statistics_from_sorted_matrix(y, w), [f], rtol=1e-12)


def test_constant_and_perfect_vectorized_statistics():
    time = np.arange(20, dtype=float)
    w = splines._build_fixed_time_spline_workspace(time, spline_df=3, spline_degree=3)
    out = s._spline_f_statistics_from_sorted_matrix(np.vstack([np.ones(20), time]), w)
    assert out[0] == 0 and np.isinf(out[1])
    assert np.isinf(
        s._max_step_f_statistics_from_sorted_matrix(np.r_[np.zeros(10), np.ones(10)], [10])[0]
    )
