from dataclasses import replace
import numpy as np
import pandas as pd
import pytest
from isovita import balances as b, permutation as p, splines, statistics as s
from isovita import validate_core_inputs, run_isovita_pipeline


@pytest.mark.parametrize(
    "observed,null,expected",
    [
        (3, [1, 2, 3, 4], 3 / 5),
        (5, [1, 2, 3, 4], 1 / 5),
        (0, [1, 2], 1),
        (np.inf, [1, np.inf], 2 / 3),
        (np.nan, [1, 2], np.nan),
    ],
)
def test_plus_one_p_value(observed, null, expected):
    np.testing.assert_allclose(
        p._selection_aware_permutation_p_value(observed, np.array(null)), expected, equal_nan=True
    )


@pytest.mark.parametrize("null", [[], [np.nan]])
def test_invalid_null_stats(null):
    with pytest.raises(ValueError):
        p._selection_aware_permutation_p_value(1, null)


def test_synchronized_age_permutations():
    ages = np.array([3.0, 1.0, 1.0, 2.0, 5.0])
    idx = p._cached_permutation_indices(5, 9, 10)
    for row in idx:
        np.testing.assert_array_equal(np.sort(row), np.arange(5))
    expected = np.argsort(ages[idx], axis=1, kind="mergesort")
    actual = p._get_cached_permutation_orders(ages, n_permutations=9, random_seed=10)
    np.testing.assert_array_equal(actual, expected)
    np.testing.assert_array_equal(idx, p._cached_permutation_indices(5, 9, 10))
    assert not np.array_equal(idx, p._cached_permutation_indices(5, 9, 11))


def test_relearn_each_permutation_against_scalar(inputs, config):
    expr, meta, samples = validate_core_inputs(*inputs)
    tx = expr.loc[expr.gene_id.eq("smooth"), "transcript_id"]
    prepared = b.make_effective_isoform_composition(
        "smooth", expr, tx, samples, meta, pseudocount=1e-8, min_samples=20
    )
    ages = prepared["ages"]
    clr = b.clr_transform(prepared["composition"])
    idx = p._cached_permutation_indices(len(ages), 39, 17)
    orders = p._get_cached_permutation_orders(ages, n_permutations=39, random_seed=17)
    w = splines._build_fixed_age_spline_workspace(
        ages, spline_df=4, spline_degree=3, age_grid_size=100
    )
    batch = b._learn_permuted_balance_features_batch(
        clr,
        orders,
        config=config.balance,
        spline_workspace=w,
        log_composition=np.log(prepared["composition"]),
    )
    assignments = []
    for i in range(39):
        perm_ages = ages[idx[i]]
        observed = splines._build_observed_age_spline_workspace(
            perm_ages, spline_df=4, spline_degree=3, age_grid_size=100
        )
        scalar = b._learn_balance_from_prepared(
            prepared,
            perm_ages,
            config=config.balance,
            spline_workspace=observed,
            clr_values=clr,
            sort_order=observed["order"],
        )
        np.testing.assert_allclose(batch[i], scalar["feature"], rtol=1e-11, atol=1e-12)
        assignments.append(tuple(scalar["assignment"]))
    assert len(set(assignments)) > 1, "Fixture must exercise actual feature reselection"
    # Independently recompute both null statistics and final p-values.
    observed_summary, _, feature = b.fit_gene_balance("smooth", expr, tx, samples, meta)
    assert observed_summary["status"] == "ok"
    actual = p._run_selection_aware_gene_tests(
        "smooth",
        prepared,
        feature.to_numpy(),
        branches=("discrete", "continuous"),
        balance_config=config.balance,
        discrete_config=config.discrete,
        continuous_config=config.continuous,
        windows=None,
        inference_config=config.inference,
        spline_workspace_cache={},
        permutation_orders=orders,
    )
    null_d = []
    null_c = []
    for i in range(39):
        perm_age = ages[idx[i]]
        null_d.append(
            s._fit_discrete_step_statistic(perm_age, batch[i], config=config.discrete)["max_f_stat"]
        )
        null_c.append(
            s._fit_nested_spline_test(perm_age, batch[i], spline_df=3, spline_degree=3)["f_stat"]
        )
    for branch, key, null in [("discrete", "max_f_stat", null_d), ("continuous", "f_stat", null_c)]:
        expected = (1 + np.count_nonzero(np.array(null) >= actual[branch][key])) / 40
        assert actual[branch]["p_value"] == expected


@pytest.mark.parametrize("jobs,batch", [(1, 1), (1, 13), (2, 7), (-1, 7)])
def test_parallel_and_batch_consistency(inputs, config, result, jobs, batch):
    c = replace(config, inference=replace(config.inference, batch_size=batch))
    r = run_isovita_pipeline(tpm=inputs[0], metadata=inputs[1], config=c, n_jobs=jobs)
    for name in [
        "balance_feature_matrix",
        "discrete_trajectory_results",
        "continuous_trajectory_results",
        "final_features",
    ]:
        pd.testing.assert_frame_equal(getattr(r, name), getattr(result, name), check_exact=True)


@pytest.mark.slow
def test_higher_permutation_reproducibility(inputs, config):
    c = replace(config, inference=replace(config.inference, n_permutations=199))
    a = run_isovita_pipeline(tpm=inputs[0], metadata=inputs[1], config=c)
    b_ = run_isovita_pipeline(tpm=inputs[0], metadata=inputs[1], config=c, n_jobs=2)
    pd.testing.assert_frame_equal(
        a.discrete_trajectory_results, b_.discrete_trajectory_results, check_exact=True
    )
    pd.testing.assert_frame_equal(
        a.continuous_trajectory_results, b_.continuous_trajectory_results, check_exact=True
    )
