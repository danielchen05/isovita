import inspect
from dataclasses import replace
import numpy as np
import pytest
import isovita as iv


def test_public_entrypoint_signature():
    s = inspect.signature(iv.run_isovita_pipeline)
    assert list(s.parameters) == [
        "tpm",
        "metadata",
        "branches",
        "config",
        "progress_every",
        "n_jobs",
    ]
    assert all(v.kind == inspect.Parameter.KEYWORD_ONLY for v in s.parameters.values())
    assert s.parameters["branches"].default == ("discrete", "continuous")
    assert s.parameters["progress_every"].default is None
    assert s.parameters["n_jobs"].default == 1
    assert iv.__version__ == "0.1.0"


def test_real_isoplex_and_result_contract(result):
    assert result.selected_branches == ("discrete", "continuous")
    assert result.balance_feature_matrix.shape == (3, 40)
    assert result.balance_summary.status.eq("ok").all()
    assert result.discrete_trajectory_results.tested.all()
    assert result.continuous_trajectory_results.tested.all()
    assert set(result.global_metrics.columns) >= {"effective", "gene_id", "transcript_id"}
    expected = set(result.discrete_trajectory_candidates.gene_id) | set(
        result.continuous_trajectory_candidates.gene_id
    )
    assert set(result.final_features.gene_id) == expected
    for n in [
        "localized_discrete_results",
        "localized_continuous_results",
        "localized_discrete_candidates",
        "localized_continuous_candidates",
    ]:
        assert getattr(result, n).empty


@pytest.mark.parametrize("branches", [[], ["bad"], ["discrete", "discrete"], ["localized"]])
def test_invalid_branch_selection(inputs, branches):
    with pytest.raises(ValueError):
        iv.run_isovita_pipeline(tpm=inputs[0], metadata=inputs[1], branches=branches)


@pytest.mark.parametrize("kwargs", [{"n_permutations": 0}, {"batch_size": 0}])
def test_invalid_permutation_config(kwargs):
    with pytest.raises(ValueError):
        iv.PermutationInferenceConfig(**kwargs)


@pytest.mark.parametrize("jobs", [0, -2])
def test_invalid_jobs(inputs, config, jobs):
    with pytest.raises(ValueError, match="n_jobs"):
        iv.run_isovita_pipeline(tpm=inputs[0], metadata=inputs[1], config=config, n_jobs=jobs)


def test_windows_are_opt_in(inputs, config, result):
    import pandas as pd

    window_config = replace(config, windows=iv.WindowConfig(edges=(0, 40, 85)))
    actual = iv.run_isovita_pipeline(tpm=inputs[0], metadata=inputs[1], config=window_config)
    assert actual.localized_discrete_results.empty
    assert actual.localized_continuous_results.empty
    pd.testing.assert_frame_equal(actual.final_features, result.final_features, check_exact=True)


def test_continuous_minimum_sample_size(inputs, config):
    c = replace(config, continuous=replace(config.continuous, min_samples=41))
    r = iv.run_isovita_pipeline(tpm=inputs[0], metadata=inputs[1], config=c)
    assert not r.continuous_trajectory_results.tested.any()
    assert r.continuous_trajectory_results.p_value.isna().all()


def test_top_n_ranking(inputs, config):
    c = replace(config, selection=iv.SelectionConfig(top_n_genes=1))
    r = iv.run_isovita_pipeline(tpm=inputs[0], metadata=inputs[1], config=c)
    assert len(r.final_features) <= 1
    np.testing.assert_array_equal(
        r.final_features.rank_gene, np.arange(1, len(r.final_features) + 1)
    )


def test_no_eligible_balances_returns_empty_result(inputs, config):
    tpm, meta = inputs
    tpm = tpm.groupby("gene_id", sort=False).head(1)
    r = iv.run_isovita_pipeline(tpm=tpm, metadata=meta, config=config)
    assert r.gene_routes.route.eq("exclude_lt2").all()
    assert r.balance_summary.empty
    assert r.balance_feature_matrix.empty
    assert r.discrete_trajectory_results.empty and r.continuous_trajectory_results.empty
    assert r.gene_trajectory_summary.empty and r.final_features.empty


def test_all_zero_expression_raises(inputs):
    tpm, meta = inputs
    zero = tpm.copy()
    zero.iloc[:, 2:] = 0
    with pytest.raises(ValueError, match="No transcripts remain"):
        iv.run_isovita_pipeline(tpm=zero, metadata=meta)


def test_constant_expression_is_untestable(inputs, config):
    tpm, meta = inputs
    constant = tpm.copy()
    constant.iloc[:, 2:] = 10.0
    r = iv.run_isovita_pipeline(tpm=constant, metadata=meta, config=config)
    assert not r.discrete_trajectory_results.tested.any()
    assert not r.continuous_trajectory_results.tested.any()
    assert r.final_features.empty


def test_config_defaults():
    c = iv.IsoVitaConfig()
    assert c.balance == iv.BalanceConfig(4, 3, 100, 1e-8, 20, 0.01)
    assert c.discrete == iv.DiscreteConfig(10, 0.15, 0.05)
    assert c.continuous == iv.ContinuousConfig(3, 3, 0.05, 20, 0.05, 0.95)
    assert c.inference == iv.PermutationInferenceConfig(1000, 0, 128)
    assert c.isoplex == iv.IsoPlexConfig(0.5, 5, True)
    assert c.selection.top_n_genes is None and c.windows is None
