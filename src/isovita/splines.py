"""Reusable observed and permutation spline workspaces."""

from __future__ import annotations

from patsy import build_design_matrices, dmatrix
from typing import Dict, Optional, Tuple
import numpy as np


def _array_cache_token(values: np.ndarray) -> Tuple[int, bytes]:
    """Hashable exact token for a small numeric vector used in run-local caches."""
    arr = np.ascontiguousarray(np.asarray(values, dtype=np.float64))
    return int(arr.size), arr.tobytes()


def _build_observed_time_spline_workspace(
    times: np.ndarray,
    *,
    spline_df: int,
    spline_degree: int,
    time_grid_size: int,
) -> Dict[str, object]:
    """Build observed-pass spline objects once per time vector."""
    times = np.asarray(times, dtype=float)

    # Keep the original observed-fit ordering behavior for numerical continuity.
    order = np.argsort(times)
    times_sorted = times[order]

    formula = f"bs(time, df={int(spline_df)}, degree={int(spline_degree)}, include_intercept=False)"
    X_df = dmatrix(
        formula,
        {"time": times_sorted},
        return_type="dataframe",
    )
    X = np.asarray(X_df, dtype=float)

    time_grid = np.linspace(
        float(np.min(times_sorted)),
        float(np.max(times_sorted)),
        int(time_grid_size),
    )
    X_grid = np.asarray(
        build_design_matrices(
            [X_df.design_info],
            {"time": time_grid},
        )[0],
        dtype=float,
    )

    return {
        "order": order,
        "times_sorted": times_sorted,
        "X": X,
        "time_grid": time_grid,
        "X_grid": X_grid,
    }


def _get_cached_observed_time_spline_workspace(
    times: np.ndarray,
    *,
    spline_df: int,
    spline_degree: int,
    time_grid_size: int,
    cache: Optional[Dict[Tuple[object, ...], Dict[str, object]]],
) -> Dict[str, object]:
    """Reuse observed-feature Patsy bases across genes sharing the same time vector."""
    if cache is None:
        return _build_observed_time_spline_workspace(
            times,
            spline_df=spline_df,
            spline_degree=spline_degree,
            time_grid_size=time_grid_size,
        )

    key = (
        "observed",
        int(spline_df),
        int(spline_degree),
        int(time_grid_size),
        *_array_cache_token(times),
    )
    workspace = cache.get(key)
    if workspace is None:
        workspace = _build_observed_time_spline_workspace(
            times,
            spline_df=spline_df,
            spline_degree=spline_degree,
            time_grid_size=time_grid_size,
        )
        cache[key] = workspace
    return workspace


def _get_cached_fixed_time_spline_workspace(
    times: np.ndarray,
    *,
    spline_df: int,
    spline_degree: int,
    time_grid_size: Optional[int] = None,
    cache: Optional[Dict[Tuple[object, ...], Dict[str, object]]] = None,
) -> Dict[str, object]:
    """Reuse permutation-inference spline bases across genes with the same time multiset."""
    times = np.asarray(times, dtype=float)
    times_sorted = np.sort(times, kind="mergesort")

    if cache is None:
        return _build_fixed_time_spline_workspace(
            times,
            spline_df=spline_df,
            spline_degree=spline_degree,
            time_grid_size=time_grid_size,
        )

    key = (
        "fixed",
        int(spline_df),
        int(spline_degree),
        None if time_grid_size is None else int(time_grid_size),
        *_array_cache_token(times_sorted),
    )
    workspace = cache.get(key)
    if workspace is None:
        workspace = _build_fixed_time_spline_workspace(
            times,
            spline_df=spline_df,
            spline_degree=spline_degree,
            time_grid_size=time_grid_size,
        )
        cache[key] = workspace
    return workspace


def _build_fixed_time_spline_workspace(
    times: np.ndarray,
    *,
    spline_df: int,
    spline_degree: int,
    time_grid_size: Optional[int] = None,
) -> Dict[str, object]:
    """Precompute a spline basis for a fixed multiset of time values.

    Time permutations preserve the multiset of times. Therefore the spline basis,
    its rank, and (for feature learning) the prediction grid are unchanged; only
    the expression/feature values assigned to those times change.
    """
    times = np.asarray(times, dtype=float)
    order = np.argsort(times, kind="mergesort")
    times_sorted = times[order]
    formula = f"bs(time, df={int(spline_df)}, degree={int(spline_degree)}, include_intercept=False)"
    X_df = dmatrix(
        formula,
        {"time": times_sorted},
        return_type="dataframe",
    )
    X = np.asarray(X_df, dtype=float)
    workspace: Dict[str, object] = {
        "times_sorted": times_sorted,
        "X": X,
        "X_pinv": np.linalg.pinv(X),
        "rank": int(np.linalg.matrix_rank(X)),
    }

    if time_grid_size is not None:
        time_grid = np.linspace(
            float(np.min(times_sorted)),
            float(np.max(times_sorted)),
            int(time_grid_size),
        )
        X_grid = np.asarray(
            build_design_matrices(
                [X_df.design_info],
                {"time": time_grid},
            )[0],
            dtype=float,
        )
        workspace["time_grid"] = time_grid
        workspace["X_grid"] = X_grid
        workspace["X_grid_centered"] = X_grid - X_grid.mean(axis=0, keepdims=True)

    return workspace
