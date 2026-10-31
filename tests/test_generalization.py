"""General covariates, optional filtering, and preparation CLI contracts."""
import builtins
from dataclasses import replace
import subprocess
import sys

import numpy as np
import pandas as pd
import pytest
from isovita import (
    IsoVitaConfig, WindowConfig, run_isovita_pipeline, save_isovita_pipeline_outputs,
    prepare_isovita_inputs_from_expression, validate_core_inputs, validate_prepared_inputs,
)
from isovita.cli import build_argument_parser


@pytest.mark.parametrize("value", [np.inf, -np.inf, np.nan, "invalid"])
def test_reject_nonfinite_coordinate(inputs, value):
    tpm, meta = inputs
    meta = meta.astype({"time": object}).copy()
    meta.loc[0, "time"] = value
    for validate in [validate_core_inputs, validate_prepared_inputs]:
        with pytest.raises(ValueError):
            validate(tpm, meta)


def test_generic_preparation_retains_negative_values(inputs):
    tpm, meta = inputs
    meta = meta.rename(columns={"time": "exposure"}).assign(exposure=lambda x: x.exposure - 50)
    prepared = prepare_isovita_inputs_from_expression(
        expression=tpm.drop(columns="gene_id"), metadata=meta,
        tx2gene=tpm[["gene_id", "transcript_id"]], metadata_sample_col="sample_id",
        time_col="exposure", expression_transcript_col="transcript_id",
    )
    np.testing.assert_array_equal(prepared.metadata.time, meta.exposure)
    assert (prepared.metadata.time < 0).any()
    assert len(prepared.metadata) == len(meta)
    assert set(prepared.summary.metric) >= {"time_min", "time_max", "n_negative_time"}


@pytest.mark.parametrize("mode", ["none", "isoplex"])
def test_negative_coordinate_windows_and_exports(inputs, config, tmp_path, mode):
    tpm, meta = inputs
    meta = meta.assign(time=meta.time - 40)
    c = replace(config, filtering=mode, windows=WindowConfig((-40, 2.5, 45)))
    r = run_isovita_pipeline(tpm=tpm, metadata=meta, config=c,
                             branches=("discrete", "continuous", "localized"))
    assert r.balance_summary.status.eq("ok").all()
    assert r.discrete_trajectory_results.tested.all()
    assert r.localized_continuous_results.tested.all()
    assert "best_change_time" in r.discrete_trajectory_results
    assert "discrete_change_time" in r.gene_trajectory_summary
    save_isovita_pipeline_outputs(r, tmp_path)
    assert (tmp_path / "filtering_method.txt").read_text().strip() == mode
    assert (tmp_path / "component_membership.csv").exists()
    assert (tmp_path / "isoplex_global_metrics.csv").exists() == (mode == "isoplex")
    if mode == "none":
        assert r.sample_metrics.empty
        assert r.global_metrics.effective.all()
        pd.testing.assert_frame_equal(r.filtered_expression_df, tpm)


def test_none_matches_isoplex_on_identical_membership(inputs, config, result):
    tpm, meta = inputs
    retained = result.global_metrics.loc[result.global_metrics.effective, "transcript_id"]
    r = run_isovita_pipeline(tpm=tpm.loc[tpm.transcript_id.isin(retained)], metadata=meta,
                             config=replace(config, filtering="none"))
    for name in ["balance_summary", "balance_loadings", "balance_feature_matrix",
                 "discrete_trajectory_results", "continuous_trajectory_results", "final_features"]:
        pd.testing.assert_frame_equal(getattr(r, name), getattr(result, name), check_exact=True)


def test_none_does_not_import_isoplex(inputs, config, monkeypatch):
    original_import = builtins.__import__
    def no_isoplex(name, *args, **kwargs):
        if name == "isoplex":
            raise ImportError("intentionally unavailable")
        return original_import(name, *args, **kwargs)
    monkeypatch.setattr(builtins, "__import__", no_isoplex)
    r = run_isovita_pipeline(tpm=inputs[0], metadata=inputs[1],
                             config=replace(config, filtering="none"))
    assert not r.balance_feature_matrix.empty
    with pytest.raises(ImportError, match=r'isovita\[isoplex\]'):
        run_isovita_pipeline(tpm=inputs[0], metadata=inputs[1], config=config)


@pytest.mark.parametrize("bad", [-1, np.inf, np.nan])
def test_none_validates_abundance(inputs, config, bad):
    tpm = inputs[0].copy()
    tpm.iloc[0, 2] = bad
    with pytest.raises(ValueError):
        run_isovita_pipeline(tpm=tpm, metadata=inputs[1], config=replace(config, filtering="none"))


def test_none_requires_positive_totals(inputs, config):
    tpm = inputs[0].copy()
    tpm.iloc[:, 2] = 0
    with pytest.raises(ValueError, match="positive group totals"):
        run_isovita_pipeline(tpm=tpm, metadata=inputs[1], config=replace(config, filtering="none"))


def test_none_keeps_small_abundances_and_excludes_singletons(inputs, config):
    tpm = inputs[0].copy()
    tpm.iloc[:, 2:] *= 1e-5
    r = run_isovita_pipeline(tpm=tpm, metadata=inputs[1], config=replace(config, filtering="none"))
    assert r.global_metrics.effective.all()
    assert not r.balance_feature_matrix.empty
    singletons = tpm.groupby("gene_id", sort=False).head(1)
    r = run_isovita_pipeline(tpm=singletons, metadata=inputs[1],
                             config=replace(config, filtering="none"))
    assert r.gene_routes.route.eq("exclude_lt2").all()
    assert r.balance_feature_matrix.empty and r.final_features.empty


def test_invalid_filter_mode():
    with pytest.raises(ValueError, match="filtering"):
        IsoVitaConfig(filtering="typo")


def test_cli_end_to_end(inputs, tmp_path):
    import anndata as ad
    tpm, meta = inputs
    a = ad.AnnData(tpm.iloc[:, 2:].to_numpy().T,
                   obs=pd.DataFrame(index=meta.sample_id),
                   var=pd.DataFrame(index=tpm.transcript_id))
    a.obsm["abundance"] = a.X.copy()
    a.write_h5ad(tmp_path / "input.h5ad")
    meta.rename(columns={"time": "exposure"}).assign(
        exposure=meta.time - 40).to_csv(tmp_path / "metadata.tsv", sep="\t", index=False)
    tpm[["gene_id", "transcript_id"]].to_csv(tmp_path / "mapping.csv", index=False)
    argv = ["--adata", str(tmp_path / "input.h5ad"), "--metadata", str(tmp_path / "metadata.tsv"),
            "--tx2gene", str(tmp_path / "mapping.csv"), "--time-col", "exposure",
            "--output-dir", str(tmp_path / "out")]
    assert build_argument_parser().parse_args(argv).time_col == "exposure"
    subprocess.run([sys.executable, "-c", "from isovita.cli import main; main()", *argv],
                   check=True, capture_output=True, text=True)
    actual = pd.read_csv(tmp_path / "out/metadata.tsv", sep="\t")
    np.testing.assert_allclose(actual.time, meta.time - 40)
