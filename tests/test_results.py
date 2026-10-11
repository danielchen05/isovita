import numpy as np
import pandas as pd
from isovita import add_bh_fdr, save_isovita_pipeline_outputs
from isovita.permutation import _finalize_discrete_table, _finalize_continuous_table
from isovita import DiscreteConfig, ContinuousConfig, PermutationInferenceConfig


def test_bh_hand_computed_and_missing():
    frame = pd.DataFrame({"p": [0.01, 0.04, 0.03, 0.8, np.nan]})
    out = add_bh_fdr(frame, "p")
    np.testing.assert_allclose(
        out.q_value, [0.04, 4 * 0.04 / 3, 4 * 0.04 / 3, 0.8, np.nan], equal_nan=True
    )
    assert out.fdr_significant.tolist() == [True, False, False, False, False]
    assert frame.columns.tolist() == ["p"]
    masked = add_bh_fdr(frame, "p", valid_mask=pd.Series([True, False, False, False, False]))
    assert masked.q_value.iloc[0] == 0.01 and masked.q_value.iloc[1:].isna().all()


def test_bh_families_are_separate_and_untested_excluded():
    d = pd.DataFrame(
        {
            "gene_id": ["a", "b", "c"],
            "p_value": [0.01, 0.04, 0.001],
            "tested": [True, True, False],
            "abs_step_effect": [1, 2, 3],
        }
    )
    c = pd.DataFrame(
        {
            "gene_id": ["a", "b"],
            "p_value": [0.5, 0.6],
            "tested": [True, True],
            "abs_pearson_r": [0.2, 0.3],
            "fitted_robust_amp": [1.0, 2.0],
        }
    )
    dout = _finalize_discrete_table(
        d, config=DiscreteConfig(), inference_config=PermutationInferenceConfig(), windowed=False
    ).set_index("gene_id")
    cout = _finalize_continuous_table(
        c, config=ContinuousConfig(), inference_config=PermutationInferenceConfig(), windowed=False
    ).set_index("gene_id")
    np.testing.assert_allclose(dout.loc[["a", "b"], "q_value"], [0.02, 0.04])
    assert np.isnan(dout.loc["c", "q_value"]) and not dout.loc["c", "fdr_significant"]
    np.testing.assert_allclose(cout.q_value, [0.6, 0.6])


def test_legacy_out_of_range_p_values_are_not_validated():
    # Existing helper delegates to statsmodels without clipping/checking its input.
    out = add_bh_fdr(pd.DataFrame({"p": [-0.1, 0.2]}), "p")
    assert out.q_value.iloc[0] < 0


def test_output_files(result, tmp_path):
    save_isovita_pipeline_outputs(result, tmp_path)
    assert (tmp_path / "selected_branches.txt").read_text() == "discrete\ncontinuous\n"
    actual = pd.read_csv(tmp_path / "balance_feature_matrix.csv", index_col=0)
    np.testing.assert_allclose(actual, result.balance_feature_matrix)
    assert not list(tmp_path.glob("localized*"))
    assert (tmp_path / "final_features.csv").exists()
