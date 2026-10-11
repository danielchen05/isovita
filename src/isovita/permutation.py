"""Batched selection-aware inference and branch-specific FDR."""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from functools import lru_cache
from typing import Dict, List, Optional, Sequence, Tuple
import numpy as np
import os
import pandas as pd
import warnings

from .balances import (
    _learn_permuted_balance_features_batch,
)
from .config import (
    BalanceConfig,
    ContinuousConfig,
    DiscreteConfig,
    PRODUCTION_BRANCHES,
    PermutationInferenceConfig,
    WindowConfig,
)
from .inputs import (
    normalize_branches,
)
from .results import (
    add_bh_fdr,
)
from .splines import (
    _array_cache_token,
    _get_cached_fixed_age_spline_workspace,
)
from .statistics import (
    _feature_correlations,
    _fit_discrete_step_statistic,
    _fit_nested_spline_test,
    _max_step_f_statistics_from_sorted_matrix,
    _spline_f_statistics_from_sorted_matrix,
    _valid_change_point_positions,
)


@lru_cache(maxsize=32)
def _cached_permutation_indices(
    n: int,
    n_permutations: int,
    random_seed: int,
) -> np.ndarray:
    """Return reproducible synchronized permutations for a given sample size."""
    # Mixing n into the seed makes cached permutations deterministic even when
    # genes with different amounts of missingness are encountered in a
    # different order across runs.
    seed = np.random.SeedSequence([int(random_seed), int(n), int(n_permutations)])
    rng = np.random.default_rng(seed)
    return np.vstack([rng.permutation(n) for _ in range(n_permutations)]).astype(
        np.int32,
        copy=False,
    )


def _get_cached_permutation_orders(
    ages: np.ndarray,
    *,
    n_permutations: int,
    random_seed: int,
    cache: Optional[Dict[Tuple[object, ...], np.ndarray]] = None,
) -> np.ndarray:
    """Return stable sort orders for all synchronized age permutations.

    Many genes share the same sample/age vector. Computing these B age sorts
    once per unique age vector avoids repeating one of the hottest pieces of
    Python/NumPy work for every gene.
    """
    ages = np.asarray(ages, dtype=float)
    key = (
        "perm_orders",
        int(n_permutations),
        int(random_seed),
        *_array_cache_token(ages),
    )

    if cache is not None:
        cached = cache.get(key)
        if cached is not None:
            return cached

    perm_idx = _cached_permutation_indices(
        len(ages),
        int(n_permutations),
        int(random_seed),
    )

    permuted_ages = ages[perm_idx]
    orders = np.argsort(
        permuted_ages,
        axis=1,
        kind="mergesort",
    ).astype(np.int32, copy=False)

    if cache is not None:
        cache[key] = orders
    return orders


def _selection_aware_permutation_p_value(
    observed_stat: float,
    null_stats: np.ndarray,
) -> float:
    """Monte Carlo upper-tail p-value with the standard +1 correction."""
    null_stats = np.asarray(null_stats, dtype=float)
    if np.isnan(observed_stat):
        return np.nan
    if null_stats.size == 0 or np.any(np.isnan(null_stats)):
        raise ValueError("Full-pipeline permutation statistics must not contain NaN.")
    return float((1 + np.count_nonzero(null_stats >= float(observed_stat))) / (len(null_stats) + 1))


def _annotate_discrete_no_inference(
    obs: Dict[str, object],
    *,
    n_permutations: int,
    inference_method: str,
) -> None:
    obs["selection_aware_p_value"] = np.nan
    obs["permutation_p_value"] = np.nan
    obs["p_value"] = np.nan
    obs["n_permutations"] = int(n_permutations)
    obs["inference_method"] = inference_method


def _annotate_continuous_no_inference(
    obs: Dict[str, object],
    *,
    n_permutations: int,
    inference_method: str,
) -> None:
    obs["selection_aware_p_value"] = np.nan
    obs["p_value"] = np.nan
    obs["n_permutations"] = int(n_permutations)
    obs["inference_method"] = inference_method


def _run_selection_aware_gene_tests(
    gene_id: str,
    prepared: Dict[str, object],
    observed_feature: np.ndarray,
    *,
    branches: Sequence[str],
    balance_config: BalanceConfig,
    discrete_config: DiscreteConfig,
    continuous_config: ContinuousConfig,
    windows: Optional[WindowConfig],
    inference_config: PermutationInferenceConfig,
    spline_workspace_cache: Optional[Dict[Tuple[object, ...], Dict[str, object]]],
    permutation_orders: np.ndarray,
) -> Dict[str, object]:
    """Run requested global and optional windowed tests for one gene.

    ``localized`` is only an API flag for windowed analysis. Within every
    configured interval the exact same discrete and continuous hypotheses are
    evaluated as in the global analysis.
    """
    branches = normalize_branches(branches)
    gene_id = str(gene_id)

    if "localized" in branches and windows is None:
        raise ValueError(
            "Localized/windowed analysis requires explicit config.windows. "
            "Example: WindowConfig(edges=(20, 40, 60, 85))."
        )

    ages = np.asarray(prepared["ages"], dtype=float)
    observed_feature = np.asarray(observed_feature, dtype=float)
    if len(observed_feature) != len(ages):
        raise ValueError(f"Observed feature for {gene_id} does not align with prepared samples.")

    result: Dict[str, object] = {
        "localized_discrete": [],
        "localized_continuous": [],
    }

    # ------------------------------------------------------------------
    # Observed global statistics
    # ------------------------------------------------------------------
    if "discrete" in branches:
        discrete_obs = _fit_discrete_step_statistic(
            ages,
            observed_feature,
            config=discrete_config,
        )
        discrete_obs["gene_id"] = gene_id
        result["discrete"] = discrete_obs

    if "continuous" in branches:
        continuous_obs = _fit_nested_spline_test(
            ages,
            observed_feature,
            spline_df=continuous_config.spline_df,
            spline_degree=continuous_config.spline_degree,
        )
        continuous_obs.update(_feature_correlations(ages, observed_feature))
        continuous_obs["gene_id"] = gene_id
        continuous_obs["tested"] = bool(
            continuous_obs["n"] >= continuous_config.min_samples
            and not np.isnan(continuous_obs["f_stat"])
        )
        continuous_obs["analytic_p_value"] = continuous_obs["p_value"]
        if not continuous_obs["tested"]:
            continuous_obs["p_value"] = np.nan
        result["continuous"] = continuous_obs

    # ------------------------------------------------------------------
    # Observed windowed statistics: same two hypotheses within each interval
    # ------------------------------------------------------------------
    window_specs: List[Dict[str, object]] = []
    if "localized" in branches:
        assert windows is not None
        n_windows = len(windows.intervals)

        for window_index, ((lower, upper), label) in enumerate(
            zip(windows.intervals, windows.labels)
        ):
            include_upper = window_index == n_windows - 1
            mask = _window_mask(
                ages,
                lower,
                upper,
                include_upper=include_upper,
            )
            ages_w = ages[mask]
            feature_w = observed_feature[mask]

            discrete_obs = _fit_discrete_step_statistic(
                ages_w,
                feature_w,
                config=discrete_config,
            )
            discrete_obs.update(
                {
                    "gene_id": gene_id,
                    "window": label,
                    "window_index": int(window_index),
                    "window_start": float(lower),
                    "window_end": float(upper),
                }
            )
            result["localized_discrete"].append(discrete_obs)

            continuous_obs = _fit_nested_spline_test(
                ages_w,
                feature_w,
                spline_df=continuous_config.spline_df,
                spline_degree=continuous_config.spline_degree,
            )
            continuous_obs.update(_feature_correlations(ages_w, feature_w))
            continuous_obs.update(
                {
                    "gene_id": gene_id,
                    "window": label,
                    "window_index": int(window_index),
                    "window_start": float(lower),
                    "window_end": float(upper),
                }
            )
            continuous_obs["tested"] = bool(
                continuous_obs["n"] >= continuous_config.min_samples
                and not np.isnan(continuous_obs["f_stat"])
            )
            continuous_obs["analytic_p_value"] = continuous_obs["p_value"]
            if not continuous_obs["tested"]:
                continuous_obs["p_value"] = np.nan
            result["localized_continuous"].append(continuous_obs)

            window_specs.append(
                {
                    "window_index": int(window_index),
                    "lower": float(lower),
                    "upper": float(upper),
                    "include_upper": bool(include_upper),
                }
            )

    B = int(inference_config.n_permutations)
    global_inference_label = "full_pipeline_age_permutation"
    window_discrete_label = "full_pipeline_age_permutation_windowed_discrete"
    window_continuous_label = "full_pipeline_age_permutation_windowed_continuous"

    any_tested = False
    if "discrete" in branches:
        any_tested |= bool(result["discrete"]["tested"])
    if "continuous" in branches:
        any_tested |= bool(result["continuous"]["tested"])
    if "localized" in branches:
        any_tested |= any(bool(row["tested"]) for row in result["localized_discrete"])
        any_tested |= any(bool(row["tested"]) for row in result["localized_continuous"])

    if not any_tested:
        if "discrete" in branches:
            _annotate_discrete_no_inference(
                result["discrete"],
                n_permutations=B,
                inference_method=global_inference_label,
            )
        if "continuous" in branches:
            _annotate_continuous_no_inference(
                result["continuous"],
                n_permutations=B,
                inference_method=global_inference_label,
            )
        if "localized" in branches:
            for row in result["localized_discrete"]:
                _annotate_discrete_no_inference(
                    row,
                    n_permutations=B,
                    inference_method=window_discrete_label,
                )
            for row in result["localized_continuous"]:
                _annotate_continuous_no_inference(
                    row,
                    n_permutations=B,
                    inference_method=window_continuous_label,
                )
        return result

    # ------------------------------------------------------------------
    # Age-only reusable objects
    # ------------------------------------------------------------------
    reference_order = np.argsort(ages, kind="mergesort")
    reference_ages_sorted = ages[reference_order]

    continuous_workspace = None
    if "continuous" in branches and bool(result["continuous"]["tested"]):
        continuous_workspace = _get_cached_fixed_age_spline_workspace(
            ages,
            spline_df=continuous_config.spline_df,
            spline_degree=continuous_config.spline_degree,
            cache=spline_workspace_cache,
        )

    discrete_cut_positions = None
    if "discrete" in branches and bool(result["discrete"]["tested"]):
        discrete_cut_positions, _ = _valid_change_point_positions(
            reference_ages_sorted,
            min_samples_side=discrete_config.min_samples_side,
            min_fraction_side=discrete_config.min_fraction_side,
        )

    localized_positions: Dict[int, np.ndarray] = {}
    localized_discrete_cuts: Dict[int, np.ndarray] = {}
    localized_continuous_workspaces: Dict[int, Dict[str, object]] = {}

    if "localized" in branches:
        for spec in window_specs:
            i = int(spec["window_index"])
            positions = _window_sorted_positions(
                reference_ages_sorted,
                float(spec["lower"]),
                float(spec["upper"]),
                include_upper=bool(spec["include_upper"]),
            )
            localized_positions[i] = positions
            ages_w_sorted = reference_ages_sorted[positions]

            if bool(result["localized_discrete"][i]["tested"]):
                cuts, _ = _valid_change_point_positions(
                    ages_w_sorted,
                    min_samples_side=discrete_config.min_samples_side,
                    min_fraction_side=discrete_config.min_fraction_side,
                )
                localized_discrete_cuts[i] = cuts

            if bool(result["localized_continuous"][i]["tested"]):
                localized_continuous_workspaces[i] = _get_cached_fixed_age_spline_workspace(
                    ages_w_sorted,
                    spline_df=continuous_config.spline_df,
                    spline_degree=continuous_config.spline_degree,
                    cache=spline_workspace_cache,
                )

    # ------------------------------------------------------------------
    # Balance relearning setup
    # ------------------------------------------------------------------
    k = len(prepared["isoforms"])
    clr_values = None
    log_composition = None
    balance_workspace = None

    if k > 2:
        composition = np.asarray(prepared["composition"], dtype=float)
        log_composition = np.log(composition)
        clr_values = log_composition - log_composition.mean(
            axis=1,
            keepdims=True,
        )
        balance_workspace = _get_cached_fixed_age_spline_workspace(
            ages,
            spline_df=balance_config.spline_df,
            spline_degree=balance_config.spline_degree,
            age_grid_size=balance_config.age_grid_size,
            cache=spline_workspace_cache,
        )

    batch_size = min(int(inference_config.batch_size), B)
    permutation_orders = np.asarray(permutation_orders, dtype=np.int32)
    if permutation_orders.shape != (B, len(ages)):
        raise ValueError(
            "Precomputed permutation_orders have the wrong shape for "
            f"{gene_id}: expected {(B, len(ages))}, "
            f"got {permutation_orders.shape}."
        )

    discrete_null = (
        np.zeros(B, dtype=float)
        if "discrete" in branches and bool(result["discrete"]["tested"])
        else None
    )
    continuous_null = (
        np.zeros(B, dtype=float)
        if "continuous" in branches and bool(result["continuous"]["tested"])
        else None
    )
    localized_discrete_null = {i: np.zeros(B, dtype=float) for i in localized_discrete_cuts}
    localized_continuous_null = {
        i: np.zeros(B, dtype=float) for i in localized_continuous_workspaces
    }

    # ------------------------------------------------------------------
    # Batched full-pipeline permutations
    # ------------------------------------------------------------------
    for start in range(0, B, batch_size):
        stop = min(start + batch_size, B)
        orders = permutation_orders[start:stop]

        # K=2 has one balance up to sign. All test statistics are sign-invariant,
        # so explicit balance relearning is exactly unnecessary in this case.
        if k == 2:
            sorted_features = observed_feature[orders]
        else:
            permuted_features = _learn_permuted_balance_features_batch(
                clr_values,
                orders,
                config=balance_config,
                spline_workspace=balance_workspace,
                log_composition=log_composition,
            )
            sorted_features = np.take_along_axis(
                permuted_features,
                orders,
                axis=1,
            )

        if discrete_null is not None:
            stats = _max_step_f_statistics_from_sorted_matrix(
                sorted_features,
                discrete_cut_positions,
            )
            if np.any(np.isnan(stats)):
                raise RuntimeError(f"Discrete permutation became untestable for {gene_id}.")
            discrete_null[start:stop] = stats

        if continuous_null is not None:
            stats = _spline_f_statistics_from_sorted_matrix(
                sorted_features,
                continuous_workspace,
            )
            if np.any(np.isnan(stats)):
                raise RuntimeError(f"Continuous permutation became untestable for {gene_id}.")
            continuous_null[start:stop] = stats

        for i, null in localized_discrete_null.items():
            stats = _max_step_f_statistics_from_sorted_matrix(
                sorted_features[:, localized_positions[i]],
                localized_discrete_cuts[i],
            )
            if np.any(np.isnan(stats)):
                label = result["localized_discrete"][i]["window"]
                raise RuntimeError(
                    f"Windowed discrete permutation became untestable for "
                    f"{gene_id}, window {label}."
                )
            null[start:stop] = stats

        for i, null in localized_continuous_null.items():
            stats = _spline_f_statistics_from_sorted_matrix(
                sorted_features[:, localized_positions[i]],
                localized_continuous_workspaces[i],
            )
            if np.any(np.isnan(stats)):
                label = result["localized_continuous"][i]["window"]
                raise RuntimeError(
                    f"Windowed continuous permutation became untestable for "
                    f"{gene_id}, window {label}."
                )
            null[start:stop] = stats

    # ------------------------------------------------------------------
    # Monte Carlo p-values
    # ------------------------------------------------------------------
    if "discrete" in branches:
        obs = result["discrete"]
        p = (
            _selection_aware_permutation_p_value(
                float(obs["max_f_stat"]),
                discrete_null,
            )
            if discrete_null is not None
            else np.nan
        )
        obs["selection_aware_p_value"] = p
        obs["permutation_p_value"] = p
        obs["p_value"] = p
        obs["n_permutations"] = B
        obs["inference_method"] = global_inference_label

    if "continuous" in branches:
        obs = result["continuous"]
        p = (
            _selection_aware_permutation_p_value(
                float(obs["f_stat"]),
                continuous_null,
            )
            if continuous_null is not None
            else np.nan
        )
        obs["selection_aware_p_value"] = p
        obs["p_value"] = p
        obs["n_permutations"] = B
        obs["inference_method"] = global_inference_label

    if "localized" in branches:
        for i, obs in enumerate(result["localized_discrete"]):
            null = localized_discrete_null.get(i)
            p = (
                _selection_aware_permutation_p_value(
                    float(obs["max_f_stat"]),
                    null,
                )
                if null is not None
                else np.nan
            )
            obs["selection_aware_p_value"] = p
            obs["permutation_p_value"] = p
            obs["p_value"] = p
            obs["n_permutations"] = B
            obs["inference_method"] = window_discrete_label

        for i, obs in enumerate(result["localized_continuous"]):
            null = localized_continuous_null.get(i)
            p = (
                _selection_aware_permutation_p_value(
                    float(obs["f_stat"]),
                    null,
                )
                if null is not None
                else np.nan
            )
            obs["selection_aware_p_value"] = p
            obs["p_value"] = p
            obs["n_permutations"] = B
            obs["inference_method"] = window_continuous_label

    return result


def _warn_permutation_resolution(
    analysis_name: str,
    family_size: int,
    *,
    alpha: float,
    n_permutations: int,
) -> None:
    """Warn when Monte Carlo resolution is coarse relative to the first BH cutoff."""
    if family_size <= 0:
        return
    min_possible_p = 1.0 / (int(n_permutations) + 1.0)
    first_bh_threshold = float(alpha) / int(family_size)
    if min_possible_p > first_bh_threshold:
        warnings.warn(
            f"{analysis_name} permutation resolution may be coarse for BH: "
            f"minimum p={min_possible_p:.3g}, first BH threshold="
            f"{first_bh_threshold:.3g}. Increase inference.n_permutations "
            "when precise tail resolution is required.",
            RuntimeWarning,
            stacklevel=3,
        )


def _finalize_discrete_table(
    table: pd.DataFrame,
    *,
    config: DiscreteConfig,
    inference_config: PermutationInferenceConfig,
    windowed: bool,
) -> pd.DataFrame:
    if table.empty:
        return table

    tested = table["tested"].fillna(False).astype(bool)
    _warn_permutation_resolution(
        "Windowed discrete" if windowed else "Discrete",
        int(tested.sum()),
        alpha=config.alpha_fdr,
        n_permutations=inference_config.n_permutations,
    )
    table = add_bh_fdr(
        table,
        "p_value",
        alpha=config.alpha_fdr,
        valid_mask=tested,
        q_col="q_value",
        reject_col="fdr_significant",
    )

    if windowed:
        sort_cols = [
            "q_value",
            "p_value",
            "abs_step_effect",
            "window_index",
            "gene_id",
        ]
        ascending = [True, True, False, True, True]
    else:
        sort_cols = ["q_value", "p_value", "abs_step_effect", "gene_id"]
        ascending = [True, True, False, True]

    return table.sort_values(
        sort_cols,
        ascending=ascending,
        na_position="last",
    ).reset_index(drop=True)


def _finalize_continuous_table(
    table: pd.DataFrame,
    *,
    config: ContinuousConfig,
    inference_config: PermutationInferenceConfig,
    windowed: bool,
) -> pd.DataFrame:
    if table.empty:
        return table

    tested = table["tested"].fillna(False).astype(bool)
    _warn_permutation_resolution(
        "Windowed continuous" if windowed else "Continuous",
        int(tested.sum()),
        alpha=config.alpha_fdr,
        n_permutations=inference_config.n_permutations,
    )
    table = add_bh_fdr(
        table,
        "p_value",
        alpha=config.alpha_fdr,
        valid_mask=tested,
        q_col="q_value",
        reject_col="fdr_significant",
    )
    table["score_predictive_hybrid"] = table["abs_pearson_r"].fillna(0.0) * table[
        "fitted_robust_amp"
    ].fillna(0.0)

    if windowed:
        sort_cols = [
            "q_value",
            "p_value",
            "score_predictive_hybrid",
            "window_index",
            "gene_id",
        ]
        ascending = [True, True, False, True, True]
    else:
        sort_cols = [
            "q_value",
            "p_value",
            "score_predictive_hybrid",
            "gene_id",
        ]
        ascending = [True, True, False, True]

    return table.sort_values(
        sort_cols,
        ascending=ascending,
        na_position="last",
    ).reset_index(drop=True)


def run_selection_aware_temporal_tests(
    observed_feature_matrix: pd.DataFrame,
    prepared_cache: Dict[str, Optional[Dict[str, object]]],
    *,
    branches: Sequence[str] = PRODUCTION_BRANCHES,
    balance_config: BalanceConfig = BalanceConfig(),
    discrete_config: DiscreteConfig = DiscreteConfig(),
    continuous_config: ContinuousConfig = ContinuousConfig(),
    windows: Optional[WindowConfig] = None,
    inference_config: PermutationInferenceConfig = PermutationInferenceConfig(),
    progress_every: Optional[int] = None,
    n_jobs: int = 1,
    spline_workspace_cache: Optional[Dict[Tuple[object, ...], Dict[str, object]]] = None,
    permutation_order_cache: Optional[Dict[Tuple[object, ...], np.ndarray]] = None,
) -> Dict[str, pd.DataFrame]:
    """Run production and optional windowed selection-aware inference."""
    branches = normalize_branches(branches)
    if "localized" in branches and windows is None:
        raise ValueError(
            "Localized/windowed analysis requires explicit windows. Set "
            "config.windows=WindowConfig(edges=(...))."
        )

    if spline_workspace_cache is None:
        spline_workspace_cache = {}
    if permutation_order_cache is None:
        permutation_order_cache = {}

    genes = [
        str(gene)
        for gene in observed_feature_matrix.index
        if str(gene) in prepared_cache and prepared_cache[str(gene)] is not None
    ]

    rows: Dict[str, List[Dict[str, object]]] = {
        "discrete": [],
        "continuous": [],
        "localized_discrete": [],
        "localized_continuous": [],
    }

    payloads: List[Tuple[str, Dict[str, object], np.ndarray, np.ndarray]] = []
    for gene_id in genes:
        prepared = prepared_cache[gene_id]
        if prepared is None:
            continue

        samples = np.asarray(prepared["samples"], dtype=str)
        feature_row = observed_feature_matrix.loc[gene_id]
        missing = [sample for sample in samples if sample not in feature_row.index]
        if missing:
            raise ValueError(
                f"Observed feature matrix is missing samples for {gene_id}: {missing[:5]}"
            )
        observed_feature = feature_row.loc[samples].to_numpy(dtype=float)

        permutation_orders = _get_cached_permutation_orders(
            np.asarray(prepared["ages"], dtype=float),
            n_permutations=int(inference_config.n_permutations),
            random_seed=int(inference_config.random_seed),
            cache=permutation_order_cache,
        )
        payloads.append((gene_id, prepared, observed_feature, permutation_orders))

    def _worker(payload):
        gene_id, prepared, observed_feature, permutation_orders = payload
        return _run_selection_aware_gene_tests(
            gene_id,
            prepared,
            observed_feature,
            branches=branches,
            balance_config=balance_config,
            discrete_config=discrete_config,
            continuous_config=continuous_config,
            windows=windows,
            inference_config=inference_config,
            spline_workspace_cache=spline_workspace_cache,
            permutation_orders=permutation_orders,
        )

    if n_jobs == 0 or n_jobs < -1:
        raise ValueError("n_jobs must be -1 or a positive integer.")
    resolved_n_jobs = max(1, int(os.cpu_count() or 1)) if int(n_jobs) == -1 else int(n_jobs)

    if resolved_n_jobs == 1:
        result_iter = map(_worker, payloads)
        executor = None
    else:
        executor = ThreadPoolExecutor(max_workers=resolved_n_jobs)
        result_iter = executor.map(_worker, payloads)

    try:
        for i, gene_result in enumerate(result_iter):
            if progress_every and i % progress_every == 0:
                print(
                    f"Selection-aware inference: {i}/{len(payloads)}",
                    flush=True,
                )

            if "discrete" in branches:
                rows["discrete"].append(dict(gene_result["discrete"]))
            if "continuous" in branches:
                rows["continuous"].append(dict(gene_result["continuous"]))
            if "localized" in branches:
                rows["localized_discrete"].extend(
                    dict(row) for row in gene_result["localized_discrete"]
                )
                rows["localized_continuous"].extend(
                    dict(row) for row in gene_result["localized_continuous"]
                )
    finally:
        if executor is not None:
            executor.shutdown(wait=True)

    out = {name: pd.DataFrame(values) for name, values in rows.items()}

    if "discrete" in branches:
        out["discrete"] = _finalize_discrete_table(
            out["discrete"],
            config=discrete_config,
            inference_config=inference_config,
            windowed=False,
        )
    if "continuous" in branches:
        out["continuous"] = _finalize_continuous_table(
            out["continuous"],
            config=continuous_config,
            inference_config=inference_config,
            windowed=False,
        )
    if "localized" in branches:
        # FDR families are detector-specific across every tested gene x window.
        out["localized_discrete"] = _finalize_discrete_table(
            out["localized_discrete"],
            config=discrete_config,
            inference_config=inference_config,
            windowed=True,
        )
        out["localized_continuous"] = _finalize_continuous_table(
            out["localized_continuous"],
            config=continuous_config,
            inference_config=inference_config,
            windowed=True,
        )

    return out


def _window_mask(
    ages: np.ndarray,
    lower: float,
    upper: float,
    *,
    include_upper: bool,
) -> np.ndarray:
    """Return the deterministic sample mask for one configured age interval."""
    ages = np.asarray(ages, dtype=float)
    if include_upper:
        return (ages >= float(lower)) & (ages <= float(upper))
    return (ages >= float(lower)) & (ages < float(upper))


def _window_sorted_positions(
    ages_sorted: np.ndarray,
    lower: float,
    upper: float,
    *,
    include_upper: bool,
) -> np.ndarray:
    """Return positions for one interval in an already age-sorted vector."""
    ages_sorted = np.asarray(ages_sorted, dtype=float)
    left = int(np.searchsorted(ages_sorted, float(lower), side="left"))
    right = int(
        np.searchsorted(
            ages_sorted,
            float(upper),
            side="right" if include_upper else "left",
        )
    )
    return np.arange(left, right, dtype=np.int64)
