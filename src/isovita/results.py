"""Result containers, multiplicity, selection, and CSV export."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from statsmodels.stats.multitest import multipletests
from typing import Dict, Optional, Sequence, Tuple
import numpy as np
import pandas as pd

from .config import (
    PRODUCTION_BRANCHES,
    SelectionConfig,
)
from .inputs import (
    normalize_branches,
)


@dataclass
class IsoVitaPipelineResult:
    """Container for shared intermediates and requested analysis outputs."""

    selected_branches: Tuple[str, ...]
    metadata: pd.DataFrame
    tpm: pd.DataFrame
    isoplex_expression_df: pd.DataFrame
    isoplex_filter_summary: pd.DataFrame
    sample_metrics: pd.DataFrame
    global_metrics: pd.DataFrame
    gene_routes: pd.DataFrame
    balance_summary: pd.DataFrame
    balance_loadings: pd.DataFrame
    balance_feature_matrix: pd.DataFrame
    discrete_trajectory_results: pd.DataFrame
    continuous_trajectory_results: pd.DataFrame
    localized_discrete_results: pd.DataFrame
    localized_continuous_results: pd.DataFrame
    gene_trajectory_summary: pd.DataFrame
    discrete_trajectory_candidates: pd.DataFrame
    continuous_trajectory_candidates: pd.DataFrame
    localized_discrete_candidates: pd.DataFrame
    localized_continuous_candidates: pd.DataFrame
    final_features: pd.DataFrame


def add_bh_fdr(
    df: pd.DataFrame,
    p_col: str,
    *,
    alpha: float = 0.05,
    valid_mask: Optional[pd.Series] = None,
    q_col: str = "q_value",
    reject_col: str = "fdr_significant",
) -> pd.DataFrame:
    """Return a copy with BH-adjusted p-values and rejection flags."""
    out = df.copy()

    if valid_mask is None:
        valid_mask = out[p_col].notna()
    else:
        valid_mask = pd.Series(valid_mask, index=out.index).fillna(False)
        valid_mask = valid_mask & out[p_col].notna()

    out[q_col] = np.nan
    out[reject_col] = False

    if int(valid_mask.sum()) > 0:
        reject, qvals, _, _ = multipletests(
            out.loc[valid_mask, p_col].to_numpy(dtype=float),
            alpha=alpha,
            method="fdr_bh",
        )
        out.loc[valid_mask, q_col] = qvals
        out.loc[valid_mask, reject_col] = reject

    return out


def _production_branches(branches: Sequence[str]) -> Tuple[str, ...]:
    requested = normalize_branches(branches)
    return tuple(branch for branch in PRODUCTION_BRANCHES if branch in requested)


def _trajectory_signature(row: pd.Series, branches: Sequence[str]) -> str:
    # Windowed-only execution has no selected global detectors.
    if not branches:
        return "not_selected"
    labels = [
        branch
        for branch in _production_branches(branches)
        if bool(row.get(f"{branch}_fdr_significant", False))
    ]
    return "+".join(labels) if labels else "not_selected"


def summarize_gene_trajectories(
    balance_summary: pd.DataFrame,
    discrete_results: pd.DataFrame,
    continuous_results: pd.DataFrame,
    *,
    branches: Sequence[str] = PRODUCTION_BRANCHES,
) -> pd.DataFrame:
    """Merge global evidence from selected production branches only."""
    production_branches = _production_branches(branches)
    if balance_summary.empty:
        return balance_summary.copy()
    ok_balance = balance_summary.loc[balance_summary["status"].eq("ok")].copy()
    if ok_balance.empty:
        return ok_balance

    summary = ok_balance.copy()

    if "discrete" in production_branches and not discrete_results.empty:
        discrete_keep = discrete_results.rename(
            columns={
                "p_value": "discrete_p_value",
                "q_value": "discrete_q_value",
                "fdr_significant": "discrete_fdr_significant",
                "max_f_stat": "discrete_max_f_stat",
                "step_r_squared": "discrete_step_r_squared",
                "step_effect": "discrete_step_effect",
                "abs_step_effect": "discrete_abs_step_effect",
                "best_change_age": "discrete_change_age",
                "left_mean": "discrete_left_mean",
                "right_mean": "discrete_right_mean",
                "n_left": "discrete_n_left",
                "n_right": "discrete_n_right",
                "n_candidate_cuts": "discrete_n_candidate_cuts",
            }
        )
        keep_cols = [
            "gene_id",
            "discrete_p_value",
            "discrete_q_value",
            "discrete_fdr_significant",
            "discrete_max_f_stat",
            "discrete_step_r_squared",
            "discrete_step_effect",
            "discrete_abs_step_effect",
            "discrete_change_age",
            "discrete_left_mean",
            "discrete_right_mean",
            "discrete_n_left",
            "discrete_n_right",
            "discrete_n_candidate_cuts",
        ]
        summary = summary.merge(
            discrete_keep[[c for c in keep_cols if c in discrete_keep.columns]],
            on="gene_id",
            how="left",
        )

    if "continuous" in production_branches and not continuous_results.empty:
        continuous_keep = continuous_results.rename(
            columns={
                "p_value": "continuous_p_value",
                "q_value": "continuous_q_value",
                "fdr_significant": "continuous_fdr_significant",
                "f_stat": "continuous_f_stat",
                "r_squared": "continuous_r_squared",
                "fitted_robust_amp": "continuous_fitted_robust_amp",
                "pearson_r": "continuous_pearson_r",
                "spearman_r": "continuous_spearman_r",
                "abs_pearson_r": "continuous_abs_pearson_r",
            }
        )
        keep_cols = [
            "gene_id",
            "continuous_p_value",
            "continuous_q_value",
            "continuous_fdr_significant",
            "continuous_f_stat",
            "continuous_r_squared",
            "continuous_fitted_robust_amp",
            "continuous_pearson_r",
            "continuous_spearman_r",
            "continuous_abs_pearson_r",
            "score_predictive_hybrid",
        ]
        summary = summary.merge(
            continuous_keep[[c for c in keep_cols if c in continuous_keep.columns]],
            on="gene_id",
            how="left",
        )

    for branch in PRODUCTION_BRANCHES:
        col = f"{branch}_fdr_significant"
        if col not in summary.columns:
            summary[col] = False
        summary[col] = summary[col].fillna(False).astype(bool)

    if production_branches:
        sig_cols = [f"{branch}_fdr_significant" for branch in production_branches]
        summary["n_detection_branches"] = summary[sig_cols].astype(int).sum(axis=1)
    else:
        summary["n_detection_branches"] = 0

    summary["trajectory_signature"] = summary.apply(
        lambda row: _trajectory_signature(row, production_branches),
        axis=1,
    )

    q_col_map = {
        "discrete": "discrete_q_value",
        "continuous": "continuous_q_value",
    }
    q_cols = [q_col_map[b] for b in production_branches if q_col_map[b] in summary.columns]
    summary["min_branch_q_value"] = summary[q_cols].min(axis=1, skipna=True) if q_cols else np.nan

    return summary.sort_values(
        ["n_detection_branches", "min_branch_q_value", "gene_id"],
        ascending=[False, True, True],
        na_position="last",
    ).reset_index(drop=True)


def select_final_gene_features(
    gene_summary: pd.DataFrame,
    *,
    branches: Sequence[str] = PRODUCTION_BRANCHES,
    config: SelectionConfig = SelectionConfig(),
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Return production-branch candidates and their unique gene-level union."""
    production_branches = _production_branches(branches)
    if gene_summary.empty:
        empty = gene_summary.copy()
        return empty, empty, empty

    candidates: Dict[str, pd.DataFrame] = {}
    for branch in PRODUCTION_BRANCHES:
        sig_col = f"{branch}_fdr_significant"
        if branch in production_branches and sig_col in gene_summary.columns:
            candidates[branch] = gene_summary.loc[gene_summary[sig_col].fillna(False)].copy()
        else:
            candidates[branch] = gene_summary.iloc[0:0].copy()

    if production_branches:
        sig_cols = [f"{branch}_fdr_significant" for branch in production_branches]
        selected_mask = gene_summary[sig_cols].fillna(False).any(axis=1)
    else:
        selected_mask = pd.Series(False, index=gene_summary.index)

    selected = gene_summary.loc[selected_mask].copy()
    selected = selected.sort_values(
        ["n_detection_branches", "min_branch_q_value", "gene_id"],
        ascending=[False, True, True],
        na_position="last",
    )

    if config.top_n_genes is not None:
        selected = selected.head(int(config.top_n_genes)).copy()

    selected.insert(0, "rank_gene", np.arange(1, len(selected) + 1))

    return (
        candidates["discrete"].reset_index(drop=True),
        candidates["continuous"].reset_index(drop=True),
        selected.reset_index(drop=True),
    )


def _significant_rows(table: pd.DataFrame) -> pd.DataFrame:
    """Return FDR-significant rows from an already finalized result table."""
    if table.empty or "fdr_significant" not in table.columns:
        return table.iloc[0:0].copy()
    return table.loc[table["fdr_significant"].fillna(False)].reset_index(drop=True)


def save_isovita_pipeline_outputs(
    result: IsoVitaPipelineResult,
    output_dir: str | Path,
) -> None:
    """Write shared outputs plus only the analyses that were requested."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    tables = {
        "isoplex_filter_summary.csv": result.isoplex_filter_summary,
        "isoplex_sample_metrics.csv": result.sample_metrics,
        "isoplex_global_metrics.csv": result.global_metrics,
        "gene_routes.csv": result.gene_routes,
        "balance_summary.csv": result.balance_summary,
        "balance_loadings.csv": result.balance_loadings,
        "balance_feature_matrix.csv": result.balance_feature_matrix,
        "gene_trajectory_summary.csv": result.gene_trajectory_summary,
        "final_features.csv": result.final_features,
    }

    if "discrete" in result.selected_branches:
        tables["discrete_trajectory_results.csv"] = result.discrete_trajectory_results
        tables["discrete_trajectory_candidates.csv"] = result.discrete_trajectory_candidates

    if "continuous" in result.selected_branches:
        tables["continuous_trajectory_results.csv"] = result.continuous_trajectory_results
        tables["continuous_trajectory_candidates.csv"] = result.continuous_trajectory_candidates

    if "localized" in result.selected_branches:
        tables["localized_discrete_results.csv"] = result.localized_discrete_results
        tables["localized_continuous_results.csv"] = result.localized_continuous_results
        tables["localized_discrete_candidates.csv"] = result.localized_discrete_candidates
        tables["localized_continuous_candidates.csv"] = result.localized_continuous_candidates

    for filename, table in tables.items():
        table.to_csv(
            output_dir / filename,
            index=(filename == "balance_feature_matrix.csv"),
        )

    (output_dir / "selected_branches.txt").write_text("\n".join(result.selected_branches) + "\n")
