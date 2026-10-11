"""IsoPlex adapter and effective-isoform routing."""

from __future__ import annotations

from typing import Dict, List
import pandas as pd


def run_isoplex(expression_df: pd.DataFrame):
    """Run IsoPlex on a preprocessed expression table."""
    try:
        import isoplex
    except ImportError as exc:
        raise ImportError("IsoPlex is required for run_isoplex().") from exc

    return isoplex.compute_multi_sample_isoform_metrics(
        expression_df,
        gene_col="gene_id",
        feature_col="transcript_id",
        expression_type="tpm",
    )


def get_effective_isoforms(
    global_metrics: pd.DataFrame,
) -> Dict[str, List[str]]:
    """Map each gene to its IsoPlex-effective transcript IDs."""
    required = {"gene_id", "transcript_id", "effective"}
    missing = required - set(global_metrics.columns)
    if missing:
        raise ValueError(f"IsoPlex global metrics missing: {sorted(missing)}")

    eff = global_metrics.loc[
        global_metrics["effective"].astype(bool),
        ["gene_id", "transcript_id"],
    ].copy()
    eff["gene_id"] = eff["gene_id"].astype(str)
    eff["transcript_id"] = eff["transcript_id"].astype(str)

    return eff.groupby("gene_id")["transcript_id"].apply(lambda x: list(dict.fromkeys(x))).to_dict()


def route_genes_by_effective_isoform_count(
    global_metrics: pd.DataFrame,
) -> pd.DataFrame:
    """
    Route genes under the unified condensation design.

    There is intentionally NO statistical distinction between K=2 and K>2.
    Both use the same CLR -> trajectory PC1 -> sparse-balance definition.
    """
    effective_map = get_effective_isoforms(global_metrics)
    all_genes = pd.unique(global_metrics["gene_id"].astype(str))

    rows = []
    for gene_id in all_genes:
        isoforms = effective_map.get(str(gene_id), [])
        n_eff = len(isoforms)
        route = "exclude_lt2" if n_eff < 2 else "balance_ge2"

        rows.append(
            {
                "gene_id": str(gene_id),
                "n_effective": n_eff,
                "route": route,
                "effective_isoforms": tuple(isoforms),
            }
        )

    return pd.DataFrame(rows).sort_values("gene_id").reset_index(drop=True)
