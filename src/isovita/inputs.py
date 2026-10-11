"""Canonical input validation and IsoPlex eligibility filters."""

from __future__ import annotations

from typing import List, Sequence, Tuple
import numpy as np
import pandas as pd

from .config import (
    IsoPlexConfig,
    VALID_BRANCHES,
)


def normalize_branches(branches: Sequence[str]) -> Tuple[str, ...]:
    """Validate branch selection and return it in canonical algorithm order."""
    if isinstance(branches, str):
        requested = [branches]
    else:
        requested = [str(branch) for branch in branches]

    if not requested:
        raise ValueError(
            f"At least one IsoVita branch must be selected from: {', '.join(VALID_BRANCHES)}."
        )

    unknown = sorted(set(requested) - set(VALID_BRANCHES))
    if unknown:
        raise ValueError(
            f"Unknown IsoVita branch(es): {unknown}. Choose from: {', '.join(VALID_BRANCHES)}."
        )

    if len(requested) != len(set(requested)):
        raise ValueError("Duplicate branch names are not allowed.")

    return tuple(branch for branch in VALID_BRANCHES if branch in requested)


def validate_core_inputs(
    tpm: pd.DataFrame,
    metadata: pd.DataFrame,
) -> Tuple[pd.DataFrame, pd.DataFrame, List[str]]:
    """Validate standardized TPM/metadata without performing dataset filtering."""
    required_tpm = {"gene_id", "transcript_id"}
    missing_tpm = required_tpm - set(tpm.columns)
    if missing_tpm:
        raise ValueError(
            "TPM input must already be gene-annotated. Missing columns: "
            f"{sorted(missing_tpm)}. Run isovita.preprocessing first."
        )

    required_meta = {"sample_id", "age"}
    missing_meta = required_meta - set(metadata.columns)
    if missing_meta:
        raise ValueError(
            "Metadata must already contain canonical columns sample_id and age. "
            f"Missing: {sorted(missing_meta)}."
        )

    expr = tpm.copy()
    expr["gene_id"] = expr["gene_id"].astype(str)
    expr["transcript_id"] = expr["transcript_id"].astype(str)

    if expr["transcript_id"].duplicated().any():
        duplicated = (
            expr.loc[expr["transcript_id"].duplicated(keep=False), "transcript_id"]
            .unique()
            .tolist()
        )
        raise ValueError(f"TPM transcript_id values must be unique. Examples: {duplicated[:10]}")

    meta = metadata.copy()
    meta["sample_id"] = meta["sample_id"].astype(str)
    meta["age"] = pd.to_numeric(meta["age"], errors="coerce")
    if meta["sample_id"].duplicated().any():
        raise ValueError("Metadata sample_id values must be unique.")
    if meta["age"].isna().any():
        raise ValueError("Metadata age must be numeric and non-missing.")

    sample_cols = [
        str(column) for column in expr.columns if column not in {"gene_id", "transcript_id"}
    ]
    if not sample_cols:
        raise ValueError("TPM input contains no sample columns.")

    rename_map = {
        column: str(column) for column in expr.columns if column not in {"gene_id", "transcript_id"}
    }
    expr = expr.rename(columns=rename_map)

    if len(sample_cols) != len(set(sample_cols)):
        raise ValueError("TPM sample columns are not unique after string conversion.")

    metadata_samples = meta["sample_id"].tolist()
    if sample_cols != metadata_samples:
        raise ValueError(
            "TPM sample columns must exactly match metadata sample_id values in "
            "the same order. Harmonize/filter samples in isovita.preprocessing."
        )

    numeric = expr[sample_cols].apply(pd.to_numeric, errors="coerce")
    if numeric.isna().any().any():
        raise ValueError("TPM sample values must be finite numeric values.")
    values = numeric.to_numpy(dtype=float)
    if np.any(~np.isfinite(values)):
        raise ValueError("TPM sample values must be finite numeric values.")
    if np.any(values < 0):
        raise ValueError("TPM values must be non-negative.")
    expr[sample_cols] = numeric

    meta = meta.set_index("sample_id", drop=True)
    meta.index.name = "sample"
    return expr, meta, sample_cols


def filter_tpm_for_isoplex(
    tpm: pd.DataFrame,
    sample_cols: Sequence[str],
    *,
    config: IsoPlexConfig = IsoPlexConfig(),
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Apply the algorithm's TPM eligibility filters before running IsoPlex."""
    expr = tpm.copy()
    sample_cols = list(sample_cols)

    summary_rows = [
        {
            "step": "core_tpm_input",
            "n_transcripts": int(expr["transcript_id"].nunique()),
            "n_genes": int(expr["gene_id"].nunique()),
        }
    ]

    keep_tx = (expr[sample_cols] > config.tpm_threshold).sum(axis=1) >= config.min_samples_above_tpm
    expr = expr.loc[keep_tx].copy()
    summary_rows.append(
        {
            "step": "transcript_filter",
            "n_transcripts": int(expr["transcript_id"].nunique()),
            "n_genes": int(expr["gene_id"].nunique()),
        }
    )

    if config.require_positive_gene_total_every_sample and not expr.empty:
        gene_total = expr.groupby("gene_id")[sample_cols].transform("sum")
        keep_gene = (gene_total > 0).all(axis=1)
        expr = expr.loc[keep_gene].copy()

    summary_rows.append(
        {
            "step": "gene_total_filter",
            "n_transcripts": int(expr["transcript_id"].nunique()),
            "n_genes": int(expr["gene_id"].nunique()),
        }
    )

    if expr.empty:
        raise ValueError("No transcripts remain after the configured IsoPlex TPM filters.")

    return expr, pd.DataFrame(summary_rows)
