"""Windowed tests preserve global feature learning and selection-aware inference."""

import json
from dataclasses import fields, replace
import numpy as np
import pandas as pd
import pytest
import isovita as iv
from isovita import balances as b, permutation as p, statistics as s, splines
from conftest import FIXTURES, assert_frame


@pytest.fixture(scope="module")
def window_config(config):
    return replace(config, windows=iv.WindowConfig(edges=(0, 42.5, 85)))


@pytest.fixture(scope="module")
def window_result(inputs, window_config):
    return iv.run_isovita_pipeline(
        tpm=inputs[0],
        metadata=inputs[1],
        config=window_config,
        branches=("discrete", "continuous", "localized"),
    )


@pytest.mark.parametrize(
    "branches",
    [
        ("discrete", "continuous", "localized"),
        ("continuous", "localized"),
        ("discrete", "localized"),
    ],
)
def test_original_window_pipeline_all_fields(inputs, window_config, branches):
    expected = json.loads((FIXTURES / "windows_baseline.json").read_text())["+".join(branches)]
    actual = iv.run_isovita_pipeline(
        tpm=inputs[0], metadata=inputs[1], branches=branches, config=window_config
    )
    assert [f.name for f in fields(actual)] == list(expected)
    for name, value in expected.items():
        if name == "selected_branches":
            assert list(actual.selected_branches) == value
        else:
            table = getattr(actual, name)
            assert_frame(table, value)
            for col in ["p_value", "q_value", "selection_aware_p_value"]:
                if col in table:
                    j = value["columns"].index(col)
                    np.testing.assert_array_equal(table[col], [row[j] for row in value["data"]])


@pytest.mark.parametrize(
    "edges", [(), (1,), (1, 1), (3, 2), (0, np.inf), (0, np.nan), (-np.inf, 1)]
)
def test_invalid_window_edges(edges):
    with pytest.raises(ValueError):
        iv.WindowConfig(edges=edges)


@pytest.mark.parametrize("include_upper,expected", [(False, [1, 2, 3]), (True, [1, 2, 3, 4, 5])])
def test_window_boundaries_tied_ages_and_sorted_positions(include_upper, expected):
    age = np.array([-1, 0, 0, 1, 2, 2, 3])
    mask = p._window_mask(age, 0, 2, include_upper=include_upper)
    np.testing.assert_array_equal(np.flatnonzero(mask), expected)
    np.testing.assert_array_equal(
        p._window_sorted_positions(age, 0, 2, include_upper=include_upper), expected
    )
    a = p._window_mask(age, 0, 2, include_upper=False)
    z = p._window_mask(age, 2, 3, include_upper=True)
    assert not np.any(a & z)
    np.testing.assert_array_equal(np.flatnonzero(a | z), np.arange(1, 7))


def test_missing_explicit_windows_rejected(inputs, config):
    with pytest.raises(ValueError, match="explicit"):
        iv.run_isovita_pipeline(
            tpm=inputs[0], metadata=inputs[1], branches=("localized",), config=config
        )
    with pytest.raises(ValueError, match="explicit"):
        iv.run_selection_aware_temporal_tests(pd.DataFrame(), {}, branches=("localized",))


def test_window_results_do_not_change_global_results_or_feature(window_result, result):
    for name in [
        "global_metrics",
        "balance_summary",
        "balance_loadings",
        "balance_feature_matrix",
        "discrete_trajectory_results",
        "continuous_trajectory_results",
        "gene_trajectory_summary",
        "final_features",
    ]:
        pd.testing.assert_frame_equal(
            getattr(window_result, name), getattr(result, name), check_exact=True
        )
    for name in ["localized_discrete_results", "localized_continuous_results"]:
        table = getattr(window_result, name)
        assert len(table) == 6
        assert table.tested.all()
        assert set(table.window) == {"0-42.5", "42.5-85"}
        assert table.n.eq(20).all()
        assert table.n_permutations.eq(39).all()
        assert table.inference_method.str.startswith(
            "full_pipeline_age_permutation_windowed_"
        ).all()


def test_one_full_range_window_matches_global(inputs, config):
    r = iv.run_isovita_pipeline(
        tpm=inputs[0],
        metadata=inputs[1],
        config=replace(config, windows=iv.WindowConfig((0, 85))),
        branches=("discrete", "continuous", "localized"),
    )
    for branch, stat in [("discrete", "max_f_stat"), ("continuous", "f_stat")]:
        g = getattr(r, f"{branch}_trajectory_results").set_index("gene_id").sort_index()
        w = getattr(r, f"localized_{branch}_results").set_index("gene_id").sort_index()
        pd.testing.assert_frame_equal(
            g[[stat, "p_value", "q_value", "fdr_significant"]],
            w[[stat, "p_value", "q_value", "fdr_significant"]],
            check_exact=True,
        )


@pytest.mark.parametrize("jobs,batch", [(1, 1), (2, 13)])
def test_window_batch_and_thread_consistency(inputs, window_config, window_result, jobs, batch):
    config = replace(window_config, inference=replace(window_config.inference, batch_size=batch))
    r = iv.run_isovita_pipeline(
        tpm=inputs[0],
        metadata=inputs[1],
        config=config,
        branches=("discrete", "continuous", "localized"),
        n_jobs=jobs,
    )
    for name in ["localized_discrete_results", "localized_continuous_results", "final_features"]:
        pd.testing.assert_frame_equal(
            getattr(r, name), getattr(window_result, name), check_exact=True
        )


@pytest.mark.parametrize("edges", [(90, 100), (0, 1, 85), (0, 40, 40.1, 85)])
def test_empty_or_tiny_windows_are_untestable(inputs, config, edges):
    r = iv.run_isovita_pipeline(
        tpm=inputs[0],
        metadata=inputs[1],
        config=replace(config, windows=iv.WindowConfig(edges)),
        branches=("localized",),
    )
    for name in ["localized_discrete_results", "localized_continuous_results"]:
        table = getattr(r, name)
        invalid = table.loc[~table.tested]
        assert len(invalid) > 0
        assert invalid.p_value.isna().all() and invalid.q_value.isna().all()
        assert not invalid.fdr_significant.any()
    assert r.final_features.empty


def test_window_continuous_minimum_sample_size(inputs, window_config):
    c = replace(window_config, continuous=replace(window_config.continuous, min_samples=21))
    r = iv.run_isovita_pipeline(
        tpm=inputs[0], metadata=inputs[1], config=c, branches=("localized",)
    )
    assert r.localized_discrete_results.tested.all()
    assert not r.localized_continuous_results.tested.any()
    assert r.localized_continuous_results.p_value.isna().all()


def test_windowed_bh_pools_gene_window_pairs_per_detector():
    # Unequal per-window values distinguish pooled correction from per-window BH.
    frame = pd.DataFrame(
        {
            "gene_id": ["a", "b", "a", "b"],
            "window_index": [0, 0, 1, 1],
            "p_value": [0.01, 0.04, 0.3, np.nan],
            "tested": [True, True, True, False],
            "abs_step_effect": [1.0, 1.0, 1.0, 1.0],
            "abs_pearson_r": [0.5] * 4,
            "fitted_robust_amp": [1.0] * 4,
        }
    )
    d = p._finalize_discrete_table(
        frame,
        config=iv.DiscreteConfig(),
        inference_config=iv.PermutationInferenceConfig(),
        windowed=True,
    )
    c = p._finalize_continuous_table(
        frame.assign(p_value=[0.2, 0.5, 0.8, np.nan]),
        config=iv.ContinuousConfig(),
        inference_config=iv.PermutationInferenceConfig(),
        windowed=True,
    )
    np.testing.assert_allclose(d.q_value, [0.03, 0.06, 0.3, np.nan], equal_nan=True)
    np.testing.assert_allclose(c.q_value, [0.6, 0.75, 0.8, np.nan], equal_nan=True)
    assert d.fdr_significant.tolist() == [True, False, False, False]
    assert not c.fdr_significant.any()


@pytest.mark.parametrize("gene", ["step", "smooth"])
def test_window_permutations_against_scalar_global_relearning(inputs, config, gene):
    expr, meta, samples = iv.validate_core_inputs(*inputs)
    tx = expr.loc[expr.gene_id.eq(gene), "transcript_id"]
    prepared = b.make_effective_isoform_composition(
        gene, expr, tx, samples, meta, pseudocount=1e-8, min_samples=20
    )
    age = prepared["ages"]
    clr = b.clr_transform(prepared["composition"])
    _, _, observed = b.fit_gene_balance(gene, expr, tx, samples, meta)
    windows = iv.WindowConfig((0, 42.5, 85))
    idx = p._cached_permutation_indices(len(age), 39, 17)
    orders = p._get_cached_permutation_orders(age, n_permutations=39, random_seed=17)
    actual = p._run_selection_aware_gene_tests(
        gene,
        prepared,
        observed.to_numpy(),
        branches=("localized",),
        balance_config=config.balance,
        discrete_config=config.discrete,
        continuous_config=config.continuous,
        windows=windows,
        inference_config=config.inference,
        spline_workspace_cache={},
        permutation_orders=orders,
    )
    null_d = [[], []]
    null_c = [[], []]
    for perm in idx:
        perm_age = age[perm]
        workspace = splines._build_observed_age_spline_workspace(
            perm_age, spline_df=4, spline_degree=3, age_grid_size=100
        )
        # Relearn on the complete permuted age vector, before choosing window samples.
        learned = b._learn_balance_from_prepared(
            prepared,
            perm_age,
            config=config.balance,
            spline_workspace=workspace,
            clr_values=clr,
            sort_order=workspace["order"],
        )
        for i, (lo, hi) in enumerate(windows.intervals):
            mask = (perm_age >= lo) & ((perm_age <= hi) if i == 1 else (perm_age < hi))
            null_d[i].append(
                s._fit_discrete_step_statistic(
                    perm_age[mask], learned["feature"][mask], config=config.discrete
                )["max_f_stat"]
            )
            null_c[i].append(
                s._fit_nested_spline_test(
                    perm_age[mask], learned["feature"][mask], spline_df=3, spline_degree=3
                )["f_stat"]
            )
    for branch, key, nulls in [
        ("discrete", "max_f_stat", null_d),
        ("continuous", "f_stat", null_c),
    ]:
        for i, row in enumerate(actual[f"localized_{branch}"]):
            mask = (age >= windows.edges[i]) & (
                (age <= windows.edges[i + 1]) if i == 1 else (age < windows.edges[i + 1])
            )
            if branch == "discrete":
                expected_stat = s._fit_discrete_step_statistic(
                    age[mask], observed.to_numpy()[mask], config=config.discrete
                )[key]
            else:
                expected_stat = s._fit_nested_spline_test(
                    age[mask], observed.to_numpy()[mask], spline_df=3, spline_degree=3
                )[key]
            np.testing.assert_allclose(row[key], expected_stat, rtol=1e-12)
            expected_p = (1 + np.count_nonzero(np.array(nulls[i]) >= expected_stat)) / 40
            assert row["p_value"] == expected_p


def test_window_exports(window_result, tmp_path):
    iv.save_isovita_pipeline_outputs(window_result, tmp_path)
    for name in [
        "localized_discrete_results",
        "localized_continuous_results",
        "localized_discrete_candidates",
        "localized_continuous_candidates",
    ]:
        expected = getattr(window_result, name)
        loaded = pd.read_csv(tmp_path / f"{name}.csv")
        assert loaded.columns.tolist() == expected.columns.tolist()
        assert len(loaded) == len(expected)
    assert (tmp_path / "selected_branches.txt").read_text() == "discrete\ncontinuous\nlocalized\n"


def test_no_eligible_genes_with_windows(inputs, window_config):
    tpm = inputs[0].groupby("gene_id", sort=False).head(1)
    r = iv.run_isovita_pipeline(
        tpm=tpm, metadata=inputs[1], config=window_config, branches=("localized",)
    )
    assert r.localized_discrete_results.empty and r.localized_continuous_results.empty
    assert r.final_features.empty


def test_window_only_matches_original_window_tables(inputs, window_config, window_result):
    # The original script computed these tables but crashed building the global
    # summary for window-only calls. This regression covers the small summary fix.
    r = iv.run_isovita_pipeline(
        tpm=inputs[0], metadata=inputs[1], config=window_config, branches=("localized",)
    )
    for name in [
        "localized_discrete_results",
        "localized_continuous_results",
        "localized_discrete_candidates",
        "localized_continuous_candidates",
    ]:
        pd.testing.assert_frame_equal(
            getattr(r, name), getattr(window_result, name), check_exact=True
        )
    assert r.discrete_trajectory_results.empty and r.continuous_trajectory_results.empty
    assert r.final_features.empty
    assert r.gene_trajectory_summary.n_detection_branches.eq(0).all()
    assert r.gene_trajectory_summary.trajectory_signature.eq("not_selected").all()
