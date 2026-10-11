from types import SimpleNamespace
import numpy as np
import pandas as pd
import pytest
from isovita import preprocessing as p, validate_core_inputs, filter_tpm_for_isoplex, IsoPlexConfig


def test_metadata_filter_age_and_order():
    meta = pd.DataFrame(
        {
            "id": ["a", "b", "c", "d", "e"],
            "years": ["2", "bad", -1, 8, None],
            "region": ["A", "A", "A", "B", "A"],
        }
    )
    out = p.prepare_metadata(meta, sample_col="id", age_col="years", filters={"region": "A"})
    assert out.sample_id.tolist() == ["a"] and out.age.tolist() == [2.0]
    out = p.prepare_metadata(meta, sample_col="id", age_col="years", postnatal_only=False)
    assert out.sample_id.tolist() == ["a", "c", "d"]


def test_mapping_alignment_and_unmapped_drop():
    expression = pd.DataFrame(
        {"tx_id": ["t1", "t2", "unmapped"], "b": [2, 3, 4], "a": [5, 6, 7], "extra": [8, 9, 10]}
    )
    metadata = pd.DataFrame({"sample_id": ["a", "missing", "b"], "age": [1, 2, 3]})
    mapping = pd.DataFrame({"transcript": ["t1", "t2"], "gene": ["g", "g"]})
    out = p.prepare_isovita_inputs_from_expression(
        expression=expression,
        metadata=metadata,
        tx2gene=mapping,
        metadata_sample_col="sample_id",
        age_col="age",
    )
    assert out.tpm.columns.tolist() == ["gene_id", "transcript_id", "a", "b"]
    assert out.metadata.sample_id.tolist() == ["a", "b"]
    assert out.tpm.transcript_id.tolist() == ["t1", "t2"]
    assert out.tpm.gene_id.tolist() == ["g", "g"]
    assert out.tpm.a.tolist() == [5, 6]


@pytest.mark.parametrize(
    "kind",
    [
        "duplicate_tx",
        "duplicate_sample",
        "bad_age",
        "missing_age",
        "negative_expression",
        "infinite_expression",
        "missing_expression",
        "wrong_order",
        "missing_gene_column",
    ],
)
def test_invalid_core_inputs(inputs, kind):
    tpm, meta = (x.copy() for x in inputs)
    if kind == "duplicate_tx":
        tpm.loc[1, "transcript_id"] = tpm.loc[0, "transcript_id"]
    elif kind == "duplicate_sample":
        meta.loc[1, "sample_id"] = meta.loc[0, "sample_id"]
    elif kind == "bad_age":
        meta["age"] = meta.age.astype(object)
        meta.loc[0, "age"] = "bad"
    elif kind == "missing_age":
        meta.loc[0, "age"] = np.nan
    elif kind == "negative_expression":
        tpm.iloc[0, 2] = -1
    elif kind == "infinite_expression":
        tpm.iloc[0, 2] = np.inf
    elif kind == "missing_expression":
        tpm.iloc[0, 2] = np.nan
    elif kind == "wrong_order":
        meta = meta.iloc[::-1]
    else:
        tpm = tpm.drop(columns="gene_id")
    with pytest.raises(ValueError):
        validate_core_inputs(tpm, meta)


def test_duplicate_mapping_and_metadata():
    with pytest.raises(ValueError, match="duplicate"):
        p.normalize_tx2gene(pd.DataFrame({"tx_id": ["t", "t"], "gene_id": ["g", "g"]}))
    with pytest.raises(ValueError, match="Duplicate"):
        p.prepare_metadata(
            pd.DataFrame({"id": ["s", "s"], "age": [1, 1]}), sample_col="id", age_col="age"
        )


def test_tpm_threshold_is_strict_and_gene_zero_filter():
    tpm = pd.DataFrame(
        {
            "gene_id": ["a", "a", "b", "b"],
            "transcript_id": ["a1", "a2", "b1", "b2"],
            "s1": [0.5, 1, 0, 0],
            "s2": [0.5, 1, 1, 2],
        }
    )
    out, summary = filter_tpm_for_isoplex(
        tpm, ["s1", "s2"], config=IsoPlexConfig(min_samples_above_tpm=1)
    )
    assert out.transcript_id.tolist() == ["a2"]
    assert summary.n_transcripts.tolist() == [4, 3, 1]
    out, _ = filter_tpm_for_isoplex(
        tpm,
        ["s1", "s2"],
        config=IsoPlexConfig(
            min_samples_above_tpm=1, require_positive_gene_total_every_sample=False
        ),
    )
    assert out.transcript_id.tolist() == ["a2", "b1", "b2"]
    with pytest.raises(ValueError, match="No transcripts"):
        filter_tpm_for_isoplex(tpm, ["s1", "s2"], config=IsoPlexConfig(tpm_threshold=100))


def test_anndata_adapter_sparse_and_parser():
    import anndata as ad
    from scipy.sparse import csr_matrix

    a = ad.AnnData(
        np.zeros((3, 2)),
        obs=pd.DataFrame(index=["s1_x", "s2_x", "s3_x"]),
        var=pd.DataFrame(index=["t1", "t2"]),
    )
    a.obsm["abundance"] = csr_matrix([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]])
    out = p.extract_transcript_abundance(a, sample_id_parser=p.make_underscore_prefix_parser(1))
    assert out.columns.tolist() == ["tx_id", "s1", "s2", "s3"]
    assert out.s2.tolist() == [3.0, 4.0]
    with pytest.raises(ValueError, match="not unique"):
        p.extract_transcript_abundance(a, sample_id_parser=lambda x: "same")
    with pytest.raises(ValueError, match="shape"):
        p.extract_transcript_abundance(
            SimpleNamespace(obsm={"abundance": np.zeros((3, 4))}, n_obs=3, n_vars=2)
        )


def test_save_and_reload_preparation(inputs, tmp_path):
    tpm, meta = inputs
    p.save_prepared_inputs(
        p.PreparedIsoVitaInputs(tpm, meta, p.build_preparation_summary(tpm=tpm, metadata=meta)),
        tmp_path,
    )
    pd.testing.assert_frame_equal(p.read_table(tmp_path / "tpm.tsv"), tpm)
    assert p.normalize_separator("tab") == "\t"


def test_known_legacy_validation_gaps(inputs):
    # Compatibility assertions, not endorsements: documented for future fixes.
    tpm, meta = (x.copy() for x in inputs)
    meta.loc[0, "age"] = np.inf
    _, validated, _ = validate_core_inputs(tpm, meta)
    assert np.isinf(validated.age.iloc[0])
    mapping = p.normalize_tx2gene(pd.DataFrame({"tx_id": [None], "gene_id": [None]}))
    assert mapping.tx_id.iloc[0] == "None"
