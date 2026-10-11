"""Compositions, exact sparse balances, and age-informed features."""

from __future__ import annotations

from typing import Dict, List, Optional, Sequence, Tuple
import numpy as np
import pandas as pd

from .config import (
    BalanceConfig,
)
from .isoplex import (
    get_effective_isoforms,
)
from .splines import (
    _get_cached_observed_age_spline_workspace,
)
from .statistics import (
    robust_amplitude,
)


def _stabilize_composition(
    fractions: np.ndarray,
    pseudocount: float,
) -> np.ndarray:
    """Add a small positive floor and renormalize each sample composition."""
    x = np.asarray(fractions, dtype=float)
    if x.ndim != 2:
        raise ValueError("fractions must be a 2D sample x isoform matrix.")
    if np.any(~np.isfinite(x)) or np.any(x < 0):
        raise ValueError("fractions must be finite and non-negative.")

    x = x + float(pseudocount)
    row_sums = x.sum(axis=1, keepdims=True)
    if np.any(row_sums <= 0):
        raise ValueError("Composition has a non-positive row sum after stabilization.")
    return x / row_sums


def clr_transform(composition: np.ndarray) -> np.ndarray:
    """Centered log-ratio transform for a strictly positive composition."""
    x = np.asarray(composition, dtype=float)
    if np.any(x <= 0):
        raise ValueError("CLR requires strictly positive compositions.")
    log_x = np.log(x)
    return log_x - log_x.mean(axis=1, keepdims=True)


def make_effective_isoform_composition(
    gene_id: str,
    expression_df: pd.DataFrame,
    effective_isoforms: Sequence[str],
    sample_cols: Sequence[str],
    metadata: pd.DataFrame,
    *,
    pseudocount: float,
    min_samples: int,
    gene_row_index: Optional[Dict[str, np.ndarray]] = None,
) -> Optional[Dict[str, object]]:
    """
    Build the effective-isoform composition for one gene.

    Only IsoPlex-effective isoforms are retained, and they are renormalized to
    sum to one within that effective set for every usable sample.

    ``gene_row_index`` is an optional positional-row lookup created once for the
    expression table. Supplying it avoids a full-table boolean scan for every
    gene while leaving the resulting composition unchanged.
    """
    gene_id = str(gene_id)
    effective_isoforms = list(dict.fromkeys(map(str, effective_isoforms)))

    if len(effective_isoforms) < 2:
        return None

    if gene_row_index is not None:
        rows = gene_row_index.get(gene_id)
        if rows is None:
            return None
        gene = expression_df.iloc[np.asarray(rows, dtype=np.int64)].copy()
    else:
        # Backward-compatible fallback for direct external calls.
        gene = expression_df.loc[expression_df["gene_id"].astype(str).eq(gene_id)].copy()

    if gene.empty:
        return None

    gene["transcript_id"] = gene["transcript_id"].astype(str)
    gene = gene.loc[gene["transcript_id"].isin(effective_isoforms)].copy()
    if gene["transcript_id"].nunique() < 2:
        return None

    # Preserve the effective-isoform order from IsoPlex.
    present = set(gene["transcript_id"])
    ordered_isoforms = [tx for tx in effective_isoforms if tx in present]
    if len(ordered_isoforms) < 2:
        return None

    mat = gene.set_index("transcript_id").loc[ordered_isoforms, list(sample_cols)].astype(float).T
    mat.index = mat.index.astype(str)

    common_samples = [sample for sample in mat.index if sample in metadata.index]
    mat = mat.loc[common_samples]

    totals = mat.sum(axis=1)
    valid = np.isfinite(totals.to_numpy()) & (totals.to_numpy() > 0)
    mat = mat.loc[valid]

    if len(mat) < min_samples:
        return None

    raw_fractions = mat.div(mat.sum(axis=1), axis=0).to_numpy(dtype=float)
    composition = _stabilize_composition(raw_fractions, pseudocount)

    return {
        "gene_id": gene_id,
        "isoforms": ordered_isoforms,
        "samples": mat.index.to_numpy(dtype=str),
        "ages": metadata.loc[mat.index, "age"].to_numpy(dtype=float),
        "raw_fractions": raw_fractions,
        "composition": composition,
    }


def _build_gene_row_index(expression_df: pd.DataFrame) -> Dict[str, np.ndarray]:
    """Build a one-time gene -> positional-row lookup for a transcript table."""
    return {
        str(gene_id): np.asarray(rows, dtype=np.int64)
        for gene_id, rows in expression_df.groupby("gene_id", sort=False).indices.items()
    }


def _orient_pc1(pc1: np.ndarray) -> np.ndarray:
    """Give the SVD loading vector a deterministic sign."""
    v = np.asarray(pc1, dtype=float).copy()

    if len(v) == 2:
        # Two-isoform CLR loadings have equal absolute magnitude.
        # Always orient the first loading positively.
        if v[0] < 0:
            v *= -1.0
    else:
        idx = int(np.argmax(np.abs(v)))
        if v[idx] < 0:
            v *= -1.0

    norm = np.linalg.norm(v)
    if norm == 0:
        raise ValueError("PC1 has zero norm.")
    return v / norm


def _assignment_to_balance_direction(assignment: np.ndarray) -> np.ndarray:
    """
    Convert {-1,0,+1} group labels to a zero-sum unit-norm balance direction.

    Group A receives equal positive weights, Group B equal negative weights,
    and unused isoforms receive zero weight. Dividing each group by its size
    enforces a zero sum; unit normalization gives a cosine-comparable vector.
    """
    z = np.asarray(assignment, dtype=np.int8)
    a = np.flatnonzero(z == 1)
    b = np.flatnonzero(z == -1)

    if len(a) == 0 or len(b) == 0:
        raise ValueError("A valid balance requires non-empty A and B groups.")

    direction = np.zeros(len(z), dtype=float)
    direction[a] = 1.0 / len(a)
    direction[b] = -1.0 / len(b)

    norm = np.linalg.norm(direction)
    if norm == 0:
        raise ValueError("Balance direction has zero norm.")

    return direction / norm


def _find_sparse_balance_exact(
    pc1: np.ndarray,
    *,
    near_optimal_cosine_tolerance: float,
) -> Dict[str, object]:
    """
    Exact polynomial-time sparse-balance search for K >= 3.

    For fixed group sizes (a, b), the cosine-maximizing balance places:
        - the a largest PC1 loadings in Group A
        - the b smallest PC1 loadings in Group B

    Therefore we only need to search group sizes rather than all
    {-1, 0, +1} assignments.

    This exactly preserves the balance-selection rule:
        1. find the globally best cosine;
        2. define the near-optimal set using the configured tolerance;
        3. choose the smallest total number of used isoforms;
        4. within that complexity, choose the highest cosine.

    Notes
    -----
    The selected balance, best cosine, chosen cosine, and n_used are exact.

    The exhaustive diagnostic ``n_near_optimal`` is not computed because
    counting every individual near-optimal assignment is itself combinatorial.
    It is diagnostic only and is not used in feature construction or inference.
    """
    v = np.asarray(pc1, dtype=float)

    norm = np.linalg.norm(v)
    if norm == 0:
        raise ValueError("PC1 has zero norm.")

    v = v / norm
    k = len(v)

    if k < 3:
        raise ValueError("Exact optimizer is defined here for K >= 3.")

    # Sort once. For fixed (a, b), A uses the a largest loadings and B uses
    # the b smallest. Stable sorting makes exact loading ties deterministic.
    order = np.argsort(v, kind="mergesort")
    v_sorted = v[order]

    # prefix[j] = sum of the j smallest sorted loadings.
    prefix = np.empty(k + 1, dtype=float)
    prefix[0] = 0.0
    np.cumsum(v_sorted, out=prefix[1:])
    total_sum = float(prefix[k])

    # For each total complexity m = a + b, record the best cosine attainable
    # by any balance using exactly m isoforms.
    best_by_used = np.full(k + 1, -np.inf, dtype=float)
    best_a_by_used = np.full(k + 1, -1, dtype=int)

    for m in range(2, k + 1):
        a = np.arange(1, m, dtype=int)
        b = m - a

        a_float = a.astype(float)
        b_float = b.astype(float)

        # Sum of the a largest and b smallest loadings.
        sum_a = total_sum - prefix[k - a]
        sum_b = prefix[b]

        mean_difference = sum_a / a_float - sum_b / b_float
        balance_norm = np.sqrt(1.0 / a_float + 1.0 / b_float)
        cosine = mean_difference / balance_norm

        # Deterministic first-tie behavior within a fixed total complexity.
        idx = int(np.argmax(cosine))
        best_by_used[m] = float(cosine[idx])
        best_a_by_used[m] = int(a[idx])

    # Original IsoVita rule: global optimum -> tolerance band -> simplest
    # complexity -> best cosine within that complexity.
    best = float(np.max(best_by_used[2:]))
    tolerance = float(near_optimal_cosine_tolerance)
    threshold = best - tolerance - 1e-12

    eligible_complexities = np.flatnonzero(best_by_used >= threshold)
    eligible_complexities = eligible_complexities[eligible_complexities >= 2]
    if eligible_complexities.size == 0:
        raise RuntimeError("No near-optimal balance complexity found.")

    n_used = int(eligible_complexities[0])
    a_size = int(best_a_by_used[n_used])
    b_size = int(n_used - a_size)

    assignment = np.zeros(k, dtype=np.int8)
    b_idx = order[:b_size]
    a_idx = order[k - a_size :]
    assignment[a_idx] = 1
    assignment[b_idx] = -1

    direction = _assignment_to_balance_direction(assignment)
    chosen_cosine = float(abs(np.dot(direction, v)))

    # Orient A/B toward the deterministically oriented PC1.
    if np.dot(direction, v) < 0:
        assignment = -assignment
        direction = -direction

    return {
        "assignment": assignment.astype(np.int8),
        "direction": direction.astype(float),
        "cosine_similarity": chosen_cosine,
        "best_cosine_similarity": best,
        "n_used": n_used,
        "n_near_optimal": np.nan,
    }


def _find_sparse_balance_assignments_batch(
    pc1_batch: np.ndarray,
    *,
    near_optimal_cosine_tolerance: float,
) -> np.ndarray:
    """Vectorized exact sparse-balance assignments for a batch of PC1 vectors.

    This applies the same optimization rule as ``find_sparse_balance`` but
    returns only the assignment matrix needed during permutation inference.
    """
    V = np.asarray(pc1_batch, dtype=float)
    if V.ndim == 1:
        V = V[None, :]
    if V.ndim != 2:
        raise ValueError("pc1_batch must be 1D or 2D.")

    batch, k = V.shape
    if k < 2:
        raise ValueError("At least two isoforms are required for a balance.")

    norms = np.linalg.norm(V, axis=1)
    if np.any(norms == 0):
        raise ValueError("PC1 has zero norm.")

    V = V / norms[:, None]

    # Match the scalar _orient_pc1() sign convention.
    if k == 2:
        flip = V[:, 0] < 0
    else:
        pivot = np.argmax(np.abs(V), axis=1)
        flip = V[np.arange(batch), pivot] < 0

    if np.any(flip):
        V[flip] *= -1.0

    if k == 2:
        assignments = np.tile(
            np.asarray([1, -1], dtype=np.int8),
            (batch, 1),
        )
        # The scalar implementation orients the balance toward PC1.
        # For K=2 this is equivalent to flipping when v0-v1 < 0.
        flip_balance = (V[:, 0] - V[:, 1]) < 0
        assignments[flip_balance] *= -1
        return assignments

    order = np.argsort(V, axis=1, kind="mergesort")
    V_sorted = np.take_along_axis(V, order, axis=1)

    prefix = np.zeros((batch, k + 1), dtype=float)
    prefix[:, 1:] = np.cumsum(V_sorted, axis=1)
    total_sum = prefix[:, -1]

    best_by_used = np.full((batch, k + 1), -np.inf, dtype=float)
    best_a_by_used = np.full((batch, k + 1), -1, dtype=np.int16)

    row_idx = np.arange(batch)
    for m in range(2, k + 1):
        a = np.arange(1, m, dtype=int)
        b = m - a

        sum_a = total_sum[:, None] - prefix[:, k - a]
        sum_b = prefix[:, b]

        mean_difference = sum_a / a[None, :] - sum_b / b[None, :]
        balance_norm = np.sqrt(1.0 / a[None, :] + 1.0 / b[None, :])
        cosine = mean_difference / balance_norm

        best_idx = np.argmax(cosine, axis=1)
        best_by_used[:, m] = cosine[row_idx, best_idx]
        best_a_by_used[:, m] = a[best_idx]

    best = np.max(best_by_used[:, 2:], axis=1)
    threshold = best - float(near_optimal_cosine_tolerance) - 1e-12

    eligible = best_by_used[:, 2:] >= threshold[:, None]
    if np.any(~eligible.any(axis=1)):
        raise RuntimeError("No near-optimal balance complexity found.")

    n_used = np.argmax(eligible, axis=1) + 2
    a_size = best_a_by_used[row_idx, n_used].astype(int)
    b_size = n_used - a_size

    # ranks[b, isoform] gives that isoform's ascending loading rank.
    ranks = np.empty_like(order)
    np.put_along_axis(
        ranks,
        order,
        np.broadcast_to(np.arange(k), order.shape),
        axis=1,
    )

    assignments = np.zeros((batch, k), dtype=np.int8)
    assignments[ranks < b_size[:, None]] = -1
    assignments[ranks >= (k - a_size)[:, None]] = 1

    return assignments


def _balance_features_from_assignments_batch(
    log_composition: np.ndarray,
    assignments: np.ndarray,
) -> np.ndarray:
    """Compute log-GM balance features for many assignments at once."""
    log_x = np.asarray(log_composition, dtype=float)
    A = np.asarray(assignments, dtype=np.int8)
    if log_x.ndim != 2 or A.ndim != 2:
        raise ValueError("log_composition and assignments must both be 2D.")
    if log_x.shape[1] != A.shape[1]:
        raise ValueError("Assignment width must equal the number of isoforms.")

    n_a = np.sum(A == 1, axis=1)
    n_b = np.sum(A == -1, axis=1)
    if np.any(n_a == 0) or np.any(n_b == 0):
        raise ValueError("Every balance requires non-empty A and B groups.")

    # Row-wise weights are +1/|A| for Group A and -1/|B| for Group B.
    weights = np.where(
        A == 1,
        1.0 / n_a[:, None],
        np.where(A == -1, -1.0 / n_b[:, None], 0.0),
    )

    return weights @ log_x.T


def _learn_permuted_balance_features_batch(
    clr_values: np.ndarray,
    permutation_orders: np.ndarray,
    *,
    config: BalanceConfig,
    spline_workspace: Dict[str, object],
    log_composition: np.ndarray,
) -> np.ndarray:
    """Relearn sparse balances for a batch of age permutations."""
    orders = np.asarray(permutation_orders, dtype=np.int64)
    if orders.ndim != 2:
        raise ValueError("permutation_orders must be a 2D matrix.")

    clr = np.asarray(clr_values, dtype=float)
    log_x = np.asarray(log_composition, dtype=float)
    if clr.ndim != 2 or clr.shape[0] != orders.shape[1]:
        raise ValueError("CLR values do not align with permutation orders.")
    if log_x.shape != clr.shape:
        raise ValueError("log_composition must have the same shape as CLR values.")

    X_pinv = np.asarray(spline_workspace["X_pinv"], dtype=float)
    X_grid_centered = np.asarray(spline_workspace["X_grid_centered"], dtype=float)

    # batch x sample x isoform
    clr_sorted = clr[orders]

    # batch x basis x isoform
    beta = np.matmul(X_pinv[None, :, :], clr_sorted)

    # batch x age_grid x isoform
    centered_grid = np.matmul(X_grid_centered[None, :, :], beta)

    # Batched SVD; only the first right singular vector is needed.
    _, _, vt = np.linalg.svd(centered_grid, full_matrices=False)
    assignments = _find_sparse_balance_assignments_batch(
        vt[:, 0, :],
        near_optimal_cosine_tolerance=config.near_optimal_cosine_tolerance,
    )

    return _balance_features_from_assignments_batch(log_x, assignments)


def find_sparse_balance(
    pc1: np.ndarray,
    *,
    near_optimal_cosine_tolerance: float = 0.01,
) -> Dict[str, object]:
    """Find the simplest near-optimal A/B/unused balance exactly.

    K = 2 has one balance up to sign and is handled analytically. For every
    K >= 3, IsoVita uses the exact polynomial-time optimizer over group sizes.

    """
    pc1 = np.asarray(pc1, dtype=float)
    norm = np.linalg.norm(pc1)
    if norm == 0:
        raise ValueError("PC1 has zero norm.")

    pc1 = pc1 / norm
    k = len(pc1)
    if k < 2:
        raise ValueError("At least two isoforms are required for a balance.")

    if k == 2:
        assignment = np.asarray([1, -1], dtype=np.int8)
        direction = _assignment_to_balance_direction(assignment)
        cosine = float(abs(np.dot(direction, pc1)))
        if np.dot(direction, pc1) < 0:
            assignment = -assignment
            direction = -direction
        return {
            "assignment": assignment,
            "direction": direction,
            "cosine_similarity": cosine,
            "best_cosine_similarity": cosine,
            "n_used": 2,
            "n_near_optimal": 1,
        }

    return _find_sparse_balance_exact(
        pc1,
        near_optimal_cosine_tolerance=float(near_optimal_cosine_tolerance),
    )


def compute_balance_feature(
    composition: np.ndarray,
    assignment: np.ndarray,
) -> np.ndarray:
    """Compute log(GM(Group A) / GM(Group B)) for every sample."""
    x = np.asarray(composition, dtype=float)
    z = np.asarray(assignment, dtype=np.int8)

    a = np.flatnonzero(z == 1)
    b = np.flatnonzero(z == -1)
    if len(a) == 0 or len(b) == 0:
        raise ValueError("A valid balance requires non-empty A and B groups.")
    if np.any(x <= 0):
        raise ValueError("Balance feature requires strictly positive composition.")

    log_x = np.log(x)
    return log_x[:, a].mean(axis=1) - log_x[:, b].mean(axis=1)


def _learn_balance_from_prepared(
    prepared: Dict[str, object],
    ages: np.ndarray,
    *,
    config: BalanceConfig,
    spline_workspace: Dict[str, object],
    clr_values: np.ndarray,
    sort_order: np.ndarray,
) -> Dict[str, object]:
    """Learn the observed age-informed sparse balance from a prepared gene."""
    isoforms = list(prepared["isoforms"])
    samples = np.asarray(prepared["samples"], dtype=str)
    composition = np.asarray(prepared["composition"], dtype=float)
    ages = np.asarray(ages, dtype=float)
    clr = np.asarray(clr_values, dtype=float)
    order = np.asarray(sort_order, dtype=np.int64)

    if len(ages) != len(samples):
        raise ValueError("ages must align one-to-one with prepared samples.")
    if np.any(~np.isfinite(ages)):
        raise ValueError("ages must be finite.")
    if clr.shape[0] != len(ages):
        raise ValueError("CLR values must align one-to-one with prepared samples.")

    clr_sorted = clr[order]
    X = np.asarray(spline_workspace["X"], dtype=float)
    if X.shape[0] != len(ages):
        raise ValueError("Age vector does not match the feature-learning spline workspace.")

    # Observed pass retains the original least-squares numerical path.
    beta = np.linalg.lstsq(X, clr_sorted, rcond=None)[0]
    age_grid = np.asarray(spline_workspace["age_grid"], dtype=float)
    X_grid = np.asarray(spline_workspace["X_grid"], dtype=float)
    fitted_grid = X_grid @ beta
    centered_grid = fitted_grid - fitted_grid.mean(axis=0, keepdims=True)

    _, singular_values, vt = np.linalg.svd(centered_grid, full_matrices=False)
    pc1 = _orient_pc1(vt[0])

    balance = find_sparse_balance(
        pc1,
        near_optimal_cosine_tolerance=config.near_optimal_cosine_tolerance,
    )

    assignment = np.asarray(balance["assignment"], dtype=np.int8)
    direction = np.asarray(balance["direction"], dtype=float)
    feature = compute_balance_feature(composition, assignment)

    return {
        "isoforms": isoforms,
        "samples": samples,
        "ages": ages,
        "composition": composition,
        "age_grid": age_grid,
        "fitted_grid": fitted_grid,
        "centered_grid": centered_grid,
        "singular_values": singular_values,
        "pc1": pc1,
        "balance": balance,
        "assignment": assignment,
        "direction": direction,
        "feature": feature,
    }


def fit_gene_balance(
    gene_id: str,
    expression_df: pd.DataFrame,
    effective_isoforms: Sequence[str],
    sample_cols: Sequence[str],
    metadata: pd.DataFrame,
    *,
    config: BalanceConfig = BalanceConfig(),
    gene_row_index: Optional[Dict[str, np.ndarray]] = None,
    prepared: Optional[Dict[str, object]] = None,
    spline_workspace_cache: Optional[Dict[Tuple[object, ...], Dict[str, object]]] = None,
) -> Tuple[Dict[str, object], pd.DataFrame, pd.Series]:
    """Fit the complete one-feature-per-gene IsoVita balance for one gene."""
    gene_id = str(gene_id)

    if prepared is None:
        prepared = make_effective_isoform_composition(
            gene_id,
            expression_df,
            effective_isoforms,
            sample_cols,
            metadata,
            pseudocount=config.clr_pseudocount,
            min_samples=config.min_samples,
            gene_row_index=gene_row_index,
        )

    if prepared is None:
        summary = {
            "gene_id": gene_id,
            "status": "failed_input",
            "n_effective": len(effective_isoforms),
        }
        return summary, pd.DataFrame(), pd.Series(dtype=float, name=gene_id)

    isoforms = list(prepared["isoforms"])
    samples = np.asarray(prepared["samples"], dtype=str)
    ages = np.asarray(prepared["ages"], dtype=float)

    try:
        observed_workspace = _get_cached_observed_age_spline_workspace(
            ages,
            spline_df=config.spline_df,
            spline_degree=config.spline_degree,
            age_grid_size=config.age_grid_size,
            cache=spline_workspace_cache,
        )
        learned = _learn_balance_from_prepared(
            prepared,
            ages,
            config=config,
            spline_workspace=observed_workspace,
            clr_values=clr_transform(np.asarray(prepared["composition"], dtype=float)),
            sort_order=np.asarray(observed_workspace["order"], dtype=np.int64),
        )
        singular_values = np.asarray(learned["singular_values"], dtype=float)
        pc1 = np.asarray(learned["pc1"], dtype=float)
        balance = learned["balance"]
        assignment = np.asarray(learned["assignment"], dtype=np.int8)
        direction = np.asarray(learned["direction"], dtype=float)
        feature = np.asarray(learned["feature"], dtype=float)

        a_idx = np.flatnonzero(assignment == 1)
        b_idx = np.flatnonzero(assignment == -1)
        u_idx = np.flatnonzero(assignment == 0)

        group_a = [isoforms[i] for i in a_idx]
        group_b = [isoforms[i] for i in b_idx]
        unused = [isoforms[i] for i in u_idx]

        total_s2 = float(np.sum(singular_values**2))
        pc1_explained = float(singular_values[0] ** 2 / total_s2) if total_s2 > 0 else np.nan

        summary = {
            "gene_id": gene_id,
            "status": "ok",
            "n_effective": len(isoforms),
            "n_samples": len(samples),
            "group_A": "|".join(group_a),
            "group_B": "|".join(group_b),
            "unused_isoforms": "|".join(unused),
            "n_group_A": len(group_a),
            "n_group_B": len(group_b),
            "n_unused": len(unused),
            "n_used": int(balance["n_used"]),
            "cosine_similarity": float(balance["cosine_similarity"]),
            "best_cosine_similarity": float(balance["best_cosine_similarity"]),
            "cosine_gap_from_best": float(
                balance["best_cosine_similarity"] - balance["cosine_similarity"]
            ),
            "n_near_optimal": balance["n_near_optimal"],
            "pc1_explained_variance_ratio": pc1_explained,
            "feature_robust_amplitude": robust_amplitude(feature),
            "feature_sd": float(np.std(feature, ddof=1)) if len(feature) > 1 else 0.0,
            "balance_method": "CLR_smooth_PC1_sparse_balance",
        }

        loadings = pd.DataFrame(
            {
                "gene_id": gene_id,
                "transcript_id": isoforms,
                "pc1_loading": pc1,
                "balance_assignment": assignment.astype(int),
                "balance_direction": direction,
            }
        )

        feature_series = pd.Series(feature, index=samples, name=gene_id, dtype=float)
        feature_series.index.name = "sample"

        return summary, loadings, feature_series

    except Exception as exc:
        summary = {
            "gene_id": gene_id,
            "status": f"failed: {exc}",
            "n_effective": len(isoforms),
            "n_samples": len(samples),
        }
        return summary, pd.DataFrame(), pd.Series(dtype=float, name=gene_id)


def build_one_feature_per_gene(
    expression_df: pd.DataFrame,
    global_metrics: pd.DataFrame,
    sample_cols: Sequence[str],
    metadata: pd.DataFrame,
    *,
    config: BalanceConfig = BalanceConfig(),
    progress_every: Optional[int] = None,
    gene_row_index: Optional[Dict[str, np.ndarray]] = None,
    prepared_cache: Optional[Dict[str, Optional[Dict[str, object]]]] = None,
    spline_workspace_cache: Optional[Dict[Tuple[object, ...], Dict[str, object]]] = None,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Build one sparse-balance feature for every gene with >=2 effective isoforms."""
    effective_map = get_effective_isoforms(global_metrics)
    genes = [gene for gene, txs in effective_map.items() if len(txs) >= 2]

    if gene_row_index is None:
        gene_row_index = _build_gene_row_index(expression_df)
    if prepared_cache is None:
        prepared_cache = {}

    summary_rows: List[Dict[str, object]] = []
    loading_tables: List[pd.DataFrame] = []
    feature_series: List[pd.Series] = []

    for i, gene_id in enumerate(genes):
        if progress_every and i % progress_every == 0:
            print(f"Balance: {i}/{len(genes)}", flush=True)

        if gene_id in prepared_cache:
            prepared = prepared_cache[gene_id]
        else:
            prepared = make_effective_isoform_composition(
                gene_id,
                expression_df,
                effective_map[gene_id],
                sample_cols,
                metadata,
                pseudocount=config.clr_pseudocount,
                min_samples=config.min_samples,
                gene_row_index=gene_row_index,
            )
            prepared_cache[gene_id] = prepared

        summary, loadings, feature = fit_gene_balance(
            gene_id,
            expression_df,
            effective_map[gene_id],
            sample_cols,
            metadata,
            config=config,
            gene_row_index=gene_row_index,
            prepared=prepared,
            spline_workspace_cache=spline_workspace_cache,
        )
        summary_rows.append(summary)

        if not loadings.empty:
            loading_tables.append(loadings)
        if not feature.empty:
            feature_series.append(feature)

    balance_summary = pd.DataFrame(summary_rows)
    if not balance_summary.empty:
        balance_summary = balance_summary.sort_values(
            ["status", "gene_id"],
            kind="stable",
        ).reset_index(drop=True)

    balance_loadings = (
        pd.concat(loading_tables, ignore_index=True)
        if loading_tables
        else pd.DataFrame(
            columns=[
                "gene_id",
                "transcript_id",
                "pc1_loading",
                "balance_assignment",
                "balance_direction",
            ]
        )
    )

    if feature_series:
        balance_feature_matrix = pd.concat(feature_series, axis=1).T
        balance_feature_matrix.index.name = "gene_id"
        balance_feature_matrix.columns.name = "sample"
    else:
        balance_feature_matrix = pd.DataFrame()
        balance_feature_matrix.index.name = "gene_id"

    return balance_summary, balance_loadings, balance_feature_matrix
