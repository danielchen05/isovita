from dataclasses import fields
import numpy as np
import pytest
from isovita import (
    run_isovita_pipeline,
    prepare_isovita_inputs_from_expression,
    validate_core_inputs,
)
from isovita import balances, splines
from conftest import assert_frame


@pytest.mark.parametrize("branches", [("discrete", "continuous"), ("discrete",), ("continuous",)])
def test_original_pipeline_all_fields(inputs, config, baseline, branches):
    r = run_isovita_pipeline(tpm=inputs[0], metadata=inputs[1], branches=branches, config=config)
    expected = baseline["+".join(branches)]
    assert [f.name for f in fields(r)] == list(expected)
    for f in fields(r):
        actual = getattr(r, f.name)
        if f.name == "selected_branches":
            assert list(actual) == expected[f.name]
        else:
            assert_frame(actual, expected[f.name])
            for col in ["p_value", "q_value", "selection_aware_p_value", "permutation_p_value"]:
                if col in actual:
                    j = expected[f.name]["columns"].index(col)
                    np.testing.assert_array_equal(
                        actual[col], [row[j] for row in expected[f.name]["data"]]
                    )


def test_original_preprocessing(inputs, baseline):
    tpm, meta = inputs
    prepared = prepare_isovita_inputs_from_expression(
        expression=tpm.drop(columns="gene_id").rename(columns={"transcript_id": "tx_id"}),
        metadata=meta.iloc[::-1],
        tx2gene=tpm[["gene_id", "transcript_id"]],
        metadata_sample_col="sample_id",
        age_col="age",
    )
    for name in ["tpm", "metadata", "summary"]:
        assert_frame(getattr(prepared, name), baseline["preprocessing"][name], exact=True)


@pytest.mark.parametrize("gene", ["step", "smooth", "noise"])
def test_original_intermediate_values(inputs, config, baseline, gene):
    expr, meta, samples = validate_core_inputs(*inputs)
    tx = expr.loc[expr.gene_id.eq(gene), "transcript_id"]
    p = balances.make_effective_isoform_composition(
        gene, expr, tx, samples, meta, pseudocount=1e-8, min_samples=20
    )
    clr = balances.clr_transform(p["composition"])
    w = splines._build_observed_age_spline_workspace(
        p["ages"], spline_df=4, spline_degree=3, age_grid_size=100
    )
    learned = balances._learn_balance_from_prepared(
        p,
        p["ages"],
        config=config.balance,
        spline_workspace=w,
        clr_values=clr,
        sort_order=w["order"],
    )
    expected = baseline["intermediates"][gene]
    for name, actual in {
        "clr": clr,
        "composition": p["composition"],
        **{
            k: learned[k]
            for k in [
                "pc1",
                "assignment",
                "direction",
                "feature",
                "singular_values",
                "fitted_grid",
                "centered_grid",
            ]
        },
    }.items():
        np.testing.assert_allclose(actual, expected[name], rtol=1e-9, atol=1e-11)


def test_all_45_original_public_signatures():
    import inspect
    import json
    import isovita
    from conftest import FIXTURES

    expected = json.loads((FIXTURES / "api.json").read_text())
    assert len(expected) == 45
    for name, signature in expected.items():
        assert str(inspect.signature(getattr(isovita, name))) == signature
