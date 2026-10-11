"""IsoVita: interpretable temporal variation in isoform composition."""

__version__ = "0.1.0"

from .balances import (
    build_one_feature_per_gene,
    clr_transform,
    compute_balance_feature,
    find_sparse_balance,
    fit_gene_balance,
    make_effective_isoform_composition,
)
from .config import (
    BalanceConfig,
    ContinuousConfig,
    DiscreteConfig,
    IsoPlexConfig,
    IsoVitaConfig,
    PRODUCTION_BRANCHES,
    PermutationInferenceConfig,
    SelectionConfig,
    VALID_BRANCHES,
    WindowConfig,
)
from .inputs import (
    filter_tpm_for_isoplex,
    normalize_branches,
    validate_core_inputs,
)
from .isoplex import (
    get_effective_isoforms,
    route_genes_by_effective_isoform_count,
    run_isoplex,
)
from .permutation import (
    run_selection_aware_temporal_tests,
)
from .pipeline import (
    run_isovita_pipeline,
)
from .preprocessing import (
    PreparedIsoVitaInputs,
    annotate_tpm_with_genes,
    apply_metadata_filters,
    build_preparation_summary,
    extract_transcript_abundance,
    identity_sample_id,
    make_underscore_prefix_parser,
    match_expression_and_metadata,
    normalize_separator,
    normalize_tx2gene,
    prepare_isovita_inputs,
    prepare_isovita_inputs_from_expression,
    prepare_metadata,
    read_table,
    resolve_sample_id_parser,
    save_prepared_inputs,
    validate_prepared_inputs,
)
from .results import (
    IsoVitaPipelineResult,
    add_bh_fdr,
    save_isovita_pipeline_outputs,
    select_final_gene_features,
    summarize_gene_trajectories,
)
from .statistics import (
    robust_amplitude,
)

__all__ = [
    "BalanceConfig",
    "ContinuousConfig",
    "DiscreteConfig",
    "IsoPlexConfig",
    "IsoVitaConfig",
    "IsoVitaPipelineResult",
    "PRODUCTION_BRANCHES",
    "PermutationInferenceConfig",
    "PreparedIsoVitaInputs",
    "SelectionConfig",
    "VALID_BRANCHES",
    "WindowConfig",
    "add_bh_fdr",
    "annotate_tpm_with_genes",
    "apply_metadata_filters",
    "build_one_feature_per_gene",
    "build_preparation_summary",
    "clr_transform",
    "compute_balance_feature",
    "extract_transcript_abundance",
    "filter_tpm_for_isoplex",
    "find_sparse_balance",
    "fit_gene_balance",
    "get_effective_isoforms",
    "identity_sample_id",
    "make_effective_isoform_composition",
    "make_underscore_prefix_parser",
    "match_expression_and_metadata",
    "normalize_branches",
    "normalize_separator",
    "normalize_tx2gene",
    "prepare_isovita_inputs",
    "prepare_isovita_inputs_from_expression",
    "prepare_metadata",
    "read_table",
    "resolve_sample_id_parser",
    "robust_amplitude",
    "route_genes_by_effective_isoform_count",
    "run_isoplex",
    "run_isovita_pipeline",
    "run_selection_aware_temporal_tests",
    "save_isovita_pipeline_outputs",
    "save_prepared_inputs",
    "select_final_gene_features",
    "summarize_gene_trajectories",
    "validate_core_inputs",
    "validate_prepared_inputs",
    "__version__",
]
