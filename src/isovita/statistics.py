"""Scalar and vectorized step and spline test statistics."""

from __future__ import annotations

from patsy import dmatrix
from scipy.stats import f as f_dist, pearsonr, spearmanr
from typing import Dict, Optional, Tuple
import numpy as np

from .config import (
    DiscreteConfig,
)


def robust_amplitude(
    values: np.ndarray,
    q_low: float = 0.05,
    q_high: float = 0.95,
) -> float:
    values = np.asarray(values, dtype=float)
    values = values[np.isfinite(values)]
    if values.size == 0:
        return np.nan
    return float(np.quantile(values, q_high) - np.quantile(values, q_low))


def _spline_f_statistics_from_sorted_matrix(
    y_sorted_matrix: np.ndarray,
    workspace: Dict[str, object],
) -> np.ndarray:
    """Vectorized spline F statistics for rows already ordered by fixed ages."""
    Y = np.asarray(y_sorted_matrix, dtype=float)
    if Y.ndim == 1:
        Y = Y[None, :]
    if Y.ndim != 2:
        raise ValueError("y_sorted_matrix must be 1D or 2D.")

    X = np.asarray(workspace["X"], dtype=float)
    X_pinv = np.asarray(workspace["X_pinv"], dtype=float)
    if Y.shape[1] != X.shape[0]:
        raise ValueError("Feature matrix does not match the spline workspace.")

    rank1 = int(workspace["rank"])
    df_diff = rank1 - 1
    df_resid = Y.shape[1] - rank1
    if df_diff <= 0 or df_resid <= 0:
        return np.full(Y.shape[0], np.nan, dtype=float)

    # beta: batch x p; fit: batch x n
    beta = Y @ X_pinv.T
    fit1 = beta @ X.T

    means = Y.mean(axis=1, keepdims=True)
    rss0 = np.sum((Y - means) ** 2, axis=1)
    rss1 = np.sum((Y - fit1) ** 2, axis=1)
    delta = np.maximum(rss0 - rss1, 0.0)

    eps = np.finfo(float).eps * np.maximum(1.0, rss0) * 32.0
    out = np.empty(Y.shape[0], dtype=float)

    constant = rss0 <= eps
    perfect = (~constant) & (rss1 <= eps)
    regular = (~constant) & (~perfect)

    out[constant] = 0.0
    out[perfect] = np.where(delta[perfect] > eps[perfect], np.inf, 0.0)

    if np.any(regular):
        numerator = delta[regular] / df_diff
        denominator = rss1[regular] / df_resid
        out[regular] = np.divide(
            numerator,
            denominator,
            out=np.full_like(numerator, np.nan),
            where=denominator > 0,
        )

    return out


def _max_step_f_statistics_from_sorted_matrix(
    y_sorted_matrix: np.ndarray,
    cut_positions: np.ndarray,
) -> np.ndarray:
    """Vectorized maximized pure-step F statistic for sorted feature rows."""
    Y = np.asarray(y_sorted_matrix, dtype=float)
    if Y.ndim == 1:
        Y = Y[None, :]
    cuts = np.asarray(cut_positions, dtype=int)
    if Y.ndim != 2:
        raise ValueError("y_sorted_matrix must be 1D or 2D.")
    if cuts.size == 0:
        return np.zeros(Y.shape[0], dtype=float)

    n = Y.shape[1]
    if n < 3:
        return np.full(Y.shape[0], np.nan, dtype=float)

    csum = np.cumsum(Y, axis=1)
    cssq = np.cumsum(Y * Y, axis=1)

    total_sum = csum[:, -1][:, None]
    total_ssq = cssq[:, -1][:, None]

    left_idx = cuts - 1
    sum_left = csum[:, left_idx]
    ssq_left = cssq[:, left_idx]
    sum_right = total_sum - sum_left
    ssq_right = total_ssq - ssq_left

    n_left = cuts[None, :].astype(float)
    n_right = (n - cuts)[None, :].astype(float)

    rss_left = np.maximum(
        ssq_left - (sum_left * sum_left) / n_left,
        0.0,
    )
    rss_right = np.maximum(
        ssq_right - (sum_right * sum_right) / n_right,
        0.0,
    )
    rss1 = np.maximum(rss_left + rss_right, 0.0)

    rss0 = np.maximum(
        total_ssq - (total_sum * total_sum) / n,
        0.0,
    )
    delta = np.maximum(rss0 - rss1, 0.0)
    denom = rss1 / (n - 2)

    eps = np.finfo(float).eps * np.maximum(1.0, rss0) * 32.0
    regular = denom > eps

    f_stats = np.zeros_like(rss1, dtype=float)
    f_stats[regular] = delta[regular] / denom[regular]
    f_stats[(~regular) & (delta > eps)] = np.inf

    return np.max(f_stats, axis=1)


def _valid_change_point_positions(
    ages_sorted: np.ndarray,
    *,
    min_samples_side: int,
    min_fraction_side: float,
) -> Tuple[np.ndarray, int]:
    """Return valid split positions j (left has j samples) between distinct ages."""
    ages = np.asarray(ages_sorted, dtype=float)
    n = len(ages)
    min_side = max(
        int(min_samples_side),
        int(np.ceil(float(min_fraction_side) * n)),
    )

    if n < 2 * min_side:
        return np.asarray([], dtype=int), min_side

    # Position j means ages[:j] are left and ages[j:] are right.
    positions = np.arange(1, n, dtype=int)
    distinct_boundary = ages[:-1] < ages[1:]
    enough_left = positions >= min_side
    enough_right = (n - positions) >= min_side
    valid = distinct_boundary & enough_left & enough_right
    return positions[valid], min_side


def _step_stats_for_sorted_values(
    y_sorted: np.ndarray,
    cut_positions: np.ndarray,
) -> Dict[str, np.ndarray | float]:
    """Evaluate constant-vs-step statistics at all supplied split positions."""
    y = np.asarray(y_sorted, dtype=float)
    cuts = np.asarray(cut_positions, dtype=int)
    n = len(y)

    if n < 3 or cuts.size == 0:
        raise ValueError("At least one valid cut and three samples are required.")

    csum = np.cumsum(y)
    cssq = np.cumsum(y * y)
    total_sum = float(csum[-1])
    total_ssq = float(cssq[-1])

    rss0 = max(total_ssq - total_sum * total_sum / n, 0.0)

    j = cuts.astype(float)
    sum_left = csum[cuts - 1]
    ssq_left = cssq[cuts - 1]
    sum_right = total_sum - sum_left
    ssq_right = total_ssq - ssq_left

    rss_left = np.maximum(ssq_left - (sum_left * sum_left) / j, 0.0)
    n_right = n - cuts
    rss_right = np.maximum(
        ssq_right - (sum_right * sum_right) / n_right,
        0.0,
    )
    rss1 = np.maximum(rss_left + rss_right, 0.0)
    delta_rss = np.maximum(rss0 - rss1, 0.0)

    denom = rss1 / (n - 2)
    eps = np.finfo(float).eps * max(1.0, rss0) * 32.0
    f_stats = np.empty_like(rss1, dtype=float)
    regular = denom > eps
    f_stats[regular] = delta_rss[regular] / denom[regular]
    f_stats[~regular & (delta_rss > eps)] = np.inf
    f_stats[~regular & (delta_rss <= eps)] = 0.0

    left_mean = sum_left / j
    right_mean = sum_right / n_right
    step_effect = right_mean - left_mean
    step_r2 = 1.0 - rss1 / rss0 if rss0 > eps else np.full_like(rss1, np.nan, dtype=float)

    return {
        "rss0": float(rss0),
        "rss1": rss1,
        "delta_rss": delta_rss,
        "f_stat": f_stats,
        "left_mean": left_mean,
        "right_mean": right_mean,
        "step_effect": step_effect,
        "step_r_squared": np.asarray(step_r2, dtype=float),
    }


def _fit_discrete_step_statistic(
    age: np.ndarray,
    y: np.ndarray,
    *,
    config: DiscreteConfig,
    sort_order: Optional[np.ndarray] = None,
    precomputed_cut_positions: Optional[np.ndarray] = None,
    precomputed_min_side: Optional[int] = None,
) -> Dict[str, object]:
    """Fit one unknown pure step and return the maximized observed statistic."""
    age = np.asarray(age, dtype=float)
    y = np.asarray(y, dtype=float)
    ok = np.isfinite(age) & np.isfinite(y)
    age = age[ok]
    y = y[ok]
    n = len(y)

    base = {
        "n": int(n),
        "tested": False,
        "n_candidate_cuts": 0,
        "best_change_age": np.nan,
        "n_left": np.nan,
        "n_right": np.nan,
        "left_mean": np.nan,
        "right_mean": np.nan,
        "step_effect": np.nan,
        "abs_step_effect": np.nan,
        "rss_null": np.nan,
        "rss_step": np.nan,
        "delta_rss": np.nan,
        "step_r_squared": np.nan,
        "max_f_stat": np.nan,
        "min_samples_side_used": np.nan,
    }

    if n < 3 or np.allclose(np.std(age), 0) or np.allclose(np.std(y), 0):
        return base

    # Precomputed orders are only valid when no values were dropped by the
    # finite mask. This is always true for the validated end-to-end pipeline.
    if sort_order is not None and bool(np.all(ok)):
        order = np.asarray(sort_order, dtype=np.int64)
    else:
        order = np.argsort(age, kind="mergesort")

    age_sorted = age[order]
    y_sorted = y[order]

    if precomputed_cut_positions is not None and bool(np.all(ok)):
        cut_positions = np.asarray(precomputed_cut_positions, dtype=int)
        min_side = (
            int(precomputed_min_side)
            if precomputed_min_side is not None
            else max(
                int(config.min_samples_side),
                int(np.ceil(float(config.min_fraction_side) * n)),
            )
        )
    else:
        cut_positions, min_side = _valid_change_point_positions(
            age_sorted,
            min_samples_side=config.min_samples_side,
            min_fraction_side=config.min_fraction_side,
        )

    base["min_samples_side_used"] = int(min_side)
    base["n_candidate_cuts"] = int(len(cut_positions))

    if cut_positions.size == 0:
        return base

    stats = _step_stats_for_sorted_values(y_sorted, cut_positions)
    f_stats = np.asarray(stats["f_stat"], dtype=float)
    if f_stats.size == 0 or np.all(np.isnan(f_stats)):
        return base

    # np.argmax is deterministic; exact ties resolve to the earliest age cut.
    best_idx = int(np.nanargmax(f_stats))
    best_pos = int(cut_positions[best_idx])
    best_f = float(f_stats[best_idx])
    change_age = float((age_sorted[best_pos - 1] + age_sorted[best_pos]) / 2.0)

    base.update(
        {
            "tested": True,
            "best_change_age": change_age,
            "n_left": best_pos,
            "n_right": n - best_pos,
            "left_mean": float(np.asarray(stats["left_mean"])[best_idx]),
            "right_mean": float(np.asarray(stats["right_mean"])[best_idx]),
            "step_effect": float(np.asarray(stats["step_effect"])[best_idx]),
            "abs_step_effect": float(abs(np.asarray(stats["step_effect"])[best_idx])),
            "rss_null": float(stats["rss0"]),
            "rss_step": float(np.asarray(stats["rss1"])[best_idx]),
            "delta_rss": float(np.asarray(stats["delta_rss"])[best_idx]),
            "step_r_squared": float(np.asarray(stats["step_r_squared"])[best_idx]),
            "max_f_stat": best_f,
        }
    )
    return base


def _fit_nested_spline_test(
    age: np.ndarray,
    y: np.ndarray,
    *,
    spline_df: int,
    spline_degree: int,
) -> Dict[str, float]:
    """Compare intercept-only versus spline(age) with an OLS nested F-test."""
    age = np.asarray(age, dtype=float)
    y = np.asarray(y, dtype=float)
    ok = np.isfinite(age) & np.isfinite(y)
    age = age[ok]
    y = y[ok]
    n = len(y)

    if n < 3 or np.allclose(np.std(age), 0) or np.allclose(np.std(y), 0):
        return {
            "n": n,
            "f_stat": np.nan,
            "p_value": np.nan,
            "df_diff": np.nan,
            "df_resid_full": np.nan,
            "r_squared": np.nan,
            "fitted_robust_amp": np.nan,
        }

    X0 = np.ones((n, 1), dtype=float)
    X1 = np.asarray(
        dmatrix(
            (f"bs(age, df={int(spline_df)}, degree={int(spline_degree)}, include_intercept=False)"),
            {"age": age},
            return_type="dataframe",
        ),
        dtype=float,
    )

    beta0 = np.linalg.lstsq(X0, y, rcond=None)[0]
    beta1 = np.linalg.lstsq(X1, y, rcond=None)[0]
    fit0 = X0 @ beta0
    fit1 = X1 @ beta1

    rss0 = float(np.sum((y - fit0) ** 2))
    rss1 = float(np.sum((y - fit1) ** 2))
    rank0 = int(np.linalg.matrix_rank(X0))
    rank1 = int(np.linalg.matrix_rank(X1))
    df_diff = rank1 - rank0
    df_resid = n - rank1

    if df_diff <= 0 or df_resid <= 0 or rss1 <= 0:
        f_stat = np.nan
        p_value = np.nan
    else:
        numerator = max(rss0 - rss1, 0.0) / df_diff
        denominator = rss1 / df_resid
        f_stat = numerator / denominator if denominator > 0 else np.nan
        p_value = float(f_dist.sf(f_stat, df_diff, df_resid)) if np.isfinite(f_stat) else np.nan

    tss = float(np.sum((y - np.mean(y)) ** 2))
    r2 = 1.0 - rss1 / tss if tss > 0 else np.nan

    return {
        "n": n,
        "f_stat": float(f_stat) if np.isfinite(f_stat) else np.nan,
        "p_value": p_value,
        "df_diff": float(df_diff),
        "df_resid_full": float(df_resid),
        "r_squared": float(r2) if np.isfinite(r2) else np.nan,
        "fitted_robust_amp": robust_amplitude(fit1),
    }


def _feature_correlations(age: np.ndarray, y: np.ndarray) -> Dict[str, float]:
    age = np.asarray(age, dtype=float)
    y = np.asarray(y, dtype=float)
    ok = np.isfinite(age) & np.isfinite(y)
    age = age[ok]
    y = y[ok]

    if len(y) < 3 or np.allclose(np.std(age), 0) or np.allclose(np.std(y), 0):
        return {
            "pearson_r": np.nan,
            "spearman_r": np.nan,
            "abs_pearson_r": np.nan,
        }

    pr = float(pearsonr(y, age)[0])
    sr = float(spearmanr(y, age)[0])
    return {
        "pearson_r": pr,
        "spearman_r": sr,
        "abs_pearson_r": abs(pr),
    }
