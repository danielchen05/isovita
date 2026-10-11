import itertools
import numpy as np
import pytest
from isovita import balances as b, BalanceConfig, validate_core_inputs


def test_stabilization_clr_and_log_gm():
    x = b._stabilize_composition(np.array([[0.0, 1.0, 3.0], [2.0, 2.0, 0.0]]), 0.1)
    expected = np.array([[0.1, 1.1, 3.1], [2.1, 2.1, 0.1]]) / 4.3
    np.testing.assert_allclose(x, expected)
    clr = b.clr_transform(x)
    np.testing.assert_allclose(clr.sum(axis=1), 0, atol=1e-14)
    np.testing.assert_allclose(clr, np.log(x) - np.log(x).mean(axis=1, keepdims=True))
    z = np.array([1, 1, -1])
    np.testing.assert_allclose(
        b.compute_balance_feature(x, z), np.log(np.sqrt(x[:, 0] * x[:, 1]) / x[:, 2])
    )
    d = b._assignment_to_balance_direction(z)
    np.testing.assert_allclose(d.sum(), 0, atol=1e-14)
    np.testing.assert_allclose(np.linalg.norm(d), 1)
    # Feature is the unnormalized log-GM contrast, not the unit direction projection.
    np.testing.assert_allclose(b.compute_balance_feature(x, z), np.log(x) @ d * np.sqrt(1 / 2 + 1))


@pytest.mark.parametrize("k", [2, 3, 4, 5, 6])
@pytest.mark.parametrize("tolerance", [0.0, 0.01, 0.2])
def test_exact_optimizer_against_exhaustive(k, tolerance):
    v = np.random.default_rng(20 + k).normal(size=k)
    v -= v.mean()
    v = b._orient_pc1(v)
    possibilities = []
    for z in itertools.product([-1, 0, 1], repeat=k):
        if 1 not in z or -1 not in z:
            continue
        z = np.array(z)
        d = b._assignment_to_balance_direction(z)
        possibilities.append((abs(d @ v), np.count_nonzero(z)))
    best = max(c for c, m in possibilities)
    used = min(m for c, m in possibilities if c >= best - tolerance - 1e-12)
    chosen = max(c for c, m in possibilities if m == used)
    actual = b.find_sparse_balance(v, near_optimal_cosine_tolerance=tolerance)
    assert actual["n_used"] == used
    np.testing.assert_allclose(actual["best_cosine_similarity"], best)
    np.testing.assert_allclose(actual["cosine_similarity"], chosen)
    assert actual["direction"] @ v >= 0


@pytest.mark.parametrize("k", [2, 3, 4, 8])
def test_batch_optimizer_and_feature_match_scalar(k):
    v = np.random.default_rng(k).normal(size=(8, k))
    # Scalar caller supplies oriented PC1; batched routine performs orientation internally.
    v = np.array([b._orient_pc1(x) for x in v])
    a = b._find_sparse_balance_assignments_batch(v, near_optimal_cosine_tolerance=0.01)
    x = b._stabilize_composition(np.random.default_rng(k + 1).uniform(size=(20, k)), 1e-8)
    features = b._balance_features_from_assignments_batch(np.log(x), a)
    for i, row in enumerate(v):
        expected = b.find_sparse_balance(row)
        np.testing.assert_array_equal(a[i], expected["assignment"])
        np.testing.assert_allclose(features[i], b.compute_balance_feature(x, a[i]))


def test_composition_order_and_index_cache(inputs):
    expr, meta, samples = validate_core_inputs(*inputs)
    tx = list(expr.loc[expr.gene_id.eq("smooth"), "transcript_id"])[::-1]
    p = b.make_effective_isoform_composition(
        "smooth", expr, tx, samples, meta, pseudocount=1e-8, min_samples=20
    )
    q = b.make_effective_isoform_composition(
        "smooth",
        expr,
        tx,
        samples,
        meta,
        pseudocount=1e-8,
        min_samples=20,
        gene_row_index=b._build_gene_row_index(expr),
    )
    assert p["isoforms"] == tx
    np.testing.assert_array_equal(p["composition"], q["composition"])
    np.testing.assert_allclose(p["composition"].sum(axis=1), 1)
    assert (
        b.make_effective_isoform_composition(
            "smooth", expr, tx[:1], samples, meta, pseudocount=1e-8, min_samples=20
        )
        is None
    )
    assert (
        b.make_effective_isoform_composition(
            "smooth", expr, tx, samples, meta, pseudocount=1e-8, min_samples=100
        )
        is None
    )


@pytest.mark.parametrize(
    "call",
    [
        lambda: b.find_sparse_balance([0, 0]),
        lambda: b.find_sparse_balance([1]),
        lambda: b.clr_transform(np.array([[0.0, 1.0]])),
        lambda: b._assignment_to_balance_direction([1, 0]),
        lambda: b._stabilize_composition(np.array([1, 2]), 1e-8),
        lambda: b._stabilize_composition(np.array([[-1, 2]]), 1e-8),
        lambda: BalanceConfig(clr_pseudocount=0),
    ],
)
def test_invalid_balances(call):
    with pytest.raises(ValueError):
        call()
