"""End-to-end production orchestration."""

from __future__ import annotations

from typing import Dict, Optional, Sequence, Tuple
import pandas as pd

from .balances import (
    _build_gene_row_index,
    build_one_feature_per_gene,
)
from .config import (
    IsoVitaConfig,
    PRODUCTION_BRANCHES,
)
from .inputs import (
    filter_tpm_for_isoplex,
    normalize_branches,
    validate_core_inputs,
)
from .isoplex import (
    route_genes_by_effective_isoform_count,
    run_isoplex,
)
from .permutation import (
    run_selection_aware_temporal_tests,
)
from .results import (
    IsoVitaPipelineResult,
    _significant_rows,
    select_final_gene_features,
    summarize_gene_trajectories,
)


def run_isovita_pipeline(
    *,
    tpm: pd.DataFrame,
    metadata: pd.DataFrame,
    branches: Sequence[str] = PRODUCTION_BRANCHES,
    config: IsoVitaConfig = IsoVitaConfig(),
    progress_every: Optional[int] = None,
    n_jobs: int = 1,
) -> IsoVitaPipelineResult:
    """Run IsoVita using the requested analysis modes.

    Parameters
    ----------
    tpm
        Standardized transcript-level TPM table with ``gene_id``,
        ``transcript_id``, then one column per retained sample.
    metadata
        Standardized sample metadata with columns ``sample_id`` and ``age``.
        It must already reflect all dataset/cohort filtering.
    branches
        Defaults to the production workflow ``("discrete", "continuous")``.
        ``"localized"`` may be explicitly added for optional windowed analyses,
        but it never contributes to ``final_features``.
    config
        Top-level IsoVita configuration. ``config.windows`` must be explicitly
        supplied whenever ``"localized"`` is requested.
    n_jobs
        Number of shared-memory worker threads for selection-aware gene-level
        permutation inference. Default 1 preserves serial behavior. Use -1 for
        all available logical CPUs.
    """
    selected_branches = normalize_branches(branches)
    if "localized" in selected_branches and config.windows is None:
        raise ValueError(
            "Localized/windowed analysis requires explicit user-defined windows. "
            "Set config.windows=WindowConfig(edges=(...))."
        )

    full_tpm, meta, sample_cols = validate_core_inputs(tpm, metadata)

    isoplex_expression_df, isoplex_filter_summary = filter_tpm_for_isoplex(
        full_tpm,
        sample_cols,
        config=config.isoplex,
    )
    sample_metrics, global_metrics = run_isoplex(isoplex_expression_df)
    gene_routes = route_genes_by_effective_isoform_count(global_metrics)

    # Run-local acceleration structures. They do not alter statistical results.
    gene_row_index = _build_gene_row_index(full_tpm)
    prepared_cache: Dict[str, Optional[Dict[str, object]]] = {}
    spline_workspace_cache: Dict[Tuple[object, ...], Dict[str, object]] = {}

    # IsoPlex determines effective-isoform membership. Feature values are then
    # computed from the original prepared TPM values for those effective isoforms.
    balance_summary, balance_loadings, balance_feature_matrix = build_one_feature_per_gene(
        expression_df=full_tpm,
        global_metrics=global_metrics,
        sample_cols=sample_cols,
        metadata=meta,
        config=config.balance,
        progress_every=progress_every,
        gene_row_index=gene_row_index,
        prepared_cache=prepared_cache,
        spline_workspace_cache=spline_workspace_cache,
    )

    branch_results = run_selection_aware_temporal_tests(
        observed_feature_matrix=balance_feature_matrix,
        prepared_cache=prepared_cache,
        branches=selected_branches,
        balance_config=config.balance,
        discrete_config=config.discrete,
        continuous_config=config.continuous,
        windows=config.windows,
        inference_config=config.inference,
        progress_every=progress_every,
        n_jobs=n_jobs,
        spline_workspace_cache=spline_workspace_cache,
    )

    discrete_results = branch_results["discrete"]
    continuous_results = branch_results["continuous"]
    localized_discrete_results = branch_results["localized_discrete"]
    localized_continuous_results = branch_results["localized_continuous"]

    # Production summary/final selection intentionally ignores windowed results.
    gene_summary = summarize_gene_trajectories(
        balance_summary,
        discrete_results,
        continuous_results,
        branches=selected_branches,
    )

    (
        discrete_candidates,
        continuous_candidates,
        final_features,
    ) = select_final_gene_features(
        gene_summary,
        branches=selected_branches,
        config=config.selection,
    )

    localized_discrete_candidates = _significant_rows(localized_discrete_results)
    localized_continuous_candidates = _significant_rows(localized_continuous_results)

    return IsoVitaPipelineResult(
        selected_branches=selected_branches,
        metadata=meta,
        tpm=full_tpm,
        isoplex_expression_df=isoplex_expression_df,
        isoplex_filter_summary=isoplex_filter_summary,
        sample_metrics=sample_metrics,
        global_metrics=global_metrics,
        gene_routes=gene_routes,
        balance_summary=balance_summary,
        balance_loadings=balance_loadings,
        balance_feature_matrix=balance_feature_matrix,
        discrete_trajectory_results=discrete_results,
        continuous_trajectory_results=continuous_results,
        localized_discrete_results=localized_discrete_results,
        localized_continuous_results=localized_continuous_results,
        gene_trajectory_summary=gene_summary,
        discrete_trajectory_candidates=discrete_candidates,
        continuous_trajectory_candidates=continuous_candidates,
        localized_discrete_candidates=localized_discrete_candidates,
        localized_continuous_candidates=localized_continuous_candidates,
        final_features=final_features,
    )
