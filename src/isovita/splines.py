"""Reusable observed and permutation spline workspaces."""

from __future__ import annotations

from patsy import build_design_matrices, dmatrix
from typing import Dict, Optional, Tuple
import numpy as np


def _array_cache_token(values: np.ndarray) -> Tuple[int, bytes]:
    """Hashable exact token for a small numeric vector used in run-local caches."""
    arr = np.ascontiguousarray(np.asarray(values, dtype=np.float64))
    return int(arr.size), arr.tobytes()


def _build_observed_age_spline_workspace(
    ages: np.ndarray,
    *,
    spline_df: int,
    spline_degree: int,
    age_grid_size: int,
) -> Dict[str, object]:
    """Build observed-pass spline objects once per age vector."""
    ages = np.asarray(ages, dtype=float)

    # Keep the original observed-fit ordering behavior for numerical continuity.
    order = np.argsort(ages)
    ages_sorted = ages[order]

    formula = f"bs(age, df={int(spline_df)}, degree={int(spline_degree)}, include_intercept=False)"
    X_df = dmatrix(
        formula,
        {"age": ages_sorted},
        return_type="dataframe",
    )
    X = np.asarray(X_df, dtype=float)

    age_grid = np.linspace(
        float(np.min(ages_sorted)),
        float(np.max(ages_sorted)),
        int(age_grid_size),
    )
    X_grid = np.asarray(
        build_design_matrices(
            [X_df.design_info],
            {"age": age_grid},
        )[0],
        dtype=float,
    )

    return {
        "order": order,
        "ages_sorted": ages_sorted,
        "X": X,
        "age_grid": age_grid,
        "X_grid": X_grid,
    }


def _get_cached_observed_age_spline_workspace(
    ages: np.ndarray,
    *,
    spline_df: int,
    spline_degree: int,
    age_grid_size: int,
    cache: Optional[Dict[Tuple[object, ...], Dict[str, object]]],
) -> Dict[str, object]:
    """Reuse observed-feature Patsy bases across genes sharing the same age vector."""
    if cache is None:
        return _build_observed_age_spline_workspace(
            ages,
            spline_df=spline_df,
            spline_degree=spline_degree,
            age_grid_size=age_grid_size,
        )

    key = (
        "observed",
        int(spline_df),
        int(spline_degree),
        int(age_grid_size),
        *_array_cache_token(ages),
    )
    workspace = cache.get(key)
    if workspace is None:
        workspace = _build_observed_age_spline_workspace(
            ages,
            spline_df=spline_df,
            spline_degree=spline_degree,
            age_grid_size=age_grid_size,
        )
        cache[key] = workspace
    return workspace


def _get_cached_fixed_age_spline_workspace(
    ages: np.ndarray,
    *,
    spline_df: int,
    spline_degree: int,
    age_grid_size: Optional[int] = None,
    cache: Optional[Dict[Tuple[object, ...], Dict[str, object]]] = None,
) -> Dict[str, object]:
    """Reuse permutation-inference spline bases across genes with the same age multiset."""
    ages = np.asarray(ages, dtype=float)
    ages_sorted = np.sort(ages, kind="mergesort")

    if cache is None:
        return _build_fixed_age_spline_workspace(
            ages,
            spline_df=spline_df,
            spline_degree=spline_degree,
            age_grid_size=age_grid_size,
        )

    key = (
        "fixed",
        int(spline_df),
        int(spline_degree),
        None if age_grid_size is None else int(age_grid_size),
        *_array_cache_token(ages_sorted),
    )
    workspace = cache.get(key)
    if workspace is None:
        workspace = _build_fixed_age_spline_workspace(
            ages,
            spline_df=spline_df,
            spline_degree=spline_degree,
            age_grid_size=age_grid_size,
        )
        cache[key] = workspace
    return workspace


def _build_fixed_age_spline_workspace(
    ages: np.ndarray,
    *,
    spline_df: int,
    spline_degree: int,
    age_grid_size: Optional[int] = None,
) -> Dict[str, object]:
    """Precompute a spline basis for a fixed multiset of age values.

    Age permutations preserve the multiset of ages. Therefore the spline basis,
    its rank, and (for feature learning) the prediction grid are unchanged; only
    the expression/feature values assigned to those ages change.
    """
    ages = np.asarray(ages, dtype=float)
    order = np.argsort(ages, kind="mergesort")
    ages_sorted = ages[order]
    formula = f"bs(age, df={int(spline_df)}, degree={int(spline_degree)}, include_intercept=False)"
    X_df = dmatrix(
        formula,
        {"age": ages_sorted},
        return_type="dataframe",
    )
    X = np.asarray(X_df, dtype=float)
    workspace: Dict[str, object] = {
        "ages_sorted": ages_sorted,
        "X": X,
        "X_pinv": np.linalg.pinv(X),
        "rank": int(np.linalg.matrix_rank(X)),
    }

    if age_grid_size is not None:
        age_grid = np.linspace(
            float(np.min(ages_sorted)),
            float(np.max(ages_sorted)),
            int(age_grid_size),
        )
        X_grid = np.asarray(
            build_design_matrices(
                [X_df.design_info],
                {"age": age_grid},
            )[0],
            dtype=float,
        )
        workspace["age_grid"] = age_grid
        workspace["X_grid"] = X_grid
        workspace["X_grid_centered"] = X_grid - X_grid.mean(axis=0, keepdims=True)

    return workspace
