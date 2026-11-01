"""Configuration objects and supported production branches."""

from __future__ import annotations

from dataclasses import dataclass, field
from numbers import Integral
from typing import Optional, Tuple
import numpy as np


PRODUCTION_BRANCHES: Tuple[str, ...] = ("discrete", "continuous")
VALID_BRANCHES: Tuple[str, ...] = PRODUCTION_BRANCHES + ("localized",)


def _require_integer(name: str, value: object) -> None:
    """Accept Python/NumPy integers, but never silently truncate or accept bools."""
    if isinstance(value, bool) or not isinstance(value, Integral):
        raise ValueError(f"{name} must be an integer.")


def _format_edge(x: float) -> str:
    x = float(x)
    return str(int(x)) if x.is_integer() else str(x)


@dataclass(frozen=True)
class WindowConfig:
    """Explicit time-window specification for optional windowed analysis.

    ``edges=(20, 40, 60, 85)`` defines the intervals 20-40, 40-60, and 60-85.
    Intervals are left-closed/right-open, except the final interval also includes
    its upper edge. No windows are supplied by IsoVita by default.
    """

    edges: Tuple[float, ...]

    def __post_init__(self) -> None:
        edges = tuple(float(x) for x in self.edges)
        if len(edges) < 2:
            raise ValueError("At least two window edges are required.")
        if any(not np.isfinite(x) for x in edges):
            raise ValueError("Window edges must be finite numeric values.")
        if any(b <= a for a, b in zip(edges[:-1], edges[1:])):
            raise ValueError("Window edges must be strictly increasing.")
        object.__setattr__(self, "edges", edges)

    @property
    def labels(self) -> Tuple[str, ...]:
        return tuple(
            f"{_format_edge(a)}-{_format_edge(b)}" for a, b in zip(self.edges[:-1], self.edges[1:])
        )

    @property
    def intervals(self) -> Tuple[Tuple[float, float], ...]:
        return tuple(zip(self.edges[:-1], self.edges[1:]))


@dataclass(frozen=True)
class IsoPlexConfig:
    """Preprocessing settings used before IsoPlex."""

    tpm_threshold: float = 0.5
    min_samples_above_tpm: int = 5
    require_positive_gene_total_every_sample: bool = True

    def __post_init__(self) -> None:
        _require_integer("min_samples_above_tpm", self.min_samples_above_tpm)


@dataclass(frozen=True)
class BalanceConfig:
    """One-feature-per-gene CLR -> trajectory PC1 -> sparse balance settings."""

    spline_df: int = 4
    spline_degree: int = 3
    time_grid_size: int = 100
    clr_pseudocount: float = 1e-8
    min_samples: int = 20
    near_optimal_cosine_tolerance: float = 0.01

    def __post_init__(self) -> None:
        for name in ("spline_df", "spline_degree", "time_grid_size", "min_samples"):
            _require_integer(name, getattr(self, name))
        if self.spline_df < self.spline_degree:
            raise ValueError("spline_df must be >= spline_degree.")
        if self.time_grid_size < 10:
            raise ValueError("time_grid_size must be at least 10.")
        if not np.isfinite(self.clr_pseudocount) or self.clr_pseudocount <= 0:
            raise ValueError("clr_pseudocount must be finite and > 0.")
        if self.min_samples < 3:
            raise ValueError("min_samples must be at least 3.")
        if not (0 <= self.near_optimal_cosine_tolerance < 1):
            raise ValueError("near_optimal_cosine_tolerance must lie in [0, 1).")


@dataclass(frozen=True)
class DiscreteConfig:
    """Pure-step change-point settings for the discrete branch."""

    min_samples_side: int = 10
    min_fraction_side: float = 0.15
    alpha_fdr: float = 0.05

    def __post_init__(self) -> None:
        _require_integer("min_samples_side", self.min_samples_side)
        if self.min_samples_side < 2:
            raise ValueError("min_samples_side must be at least 2.")
        if not (0.0 <= self.min_fraction_side < 0.5):
            raise ValueError("min_fraction_side must lie in [0, 0.5).")
        if not (0.0 < self.alpha_fdr < 1.0):
            raise ValueError("alpha_fdr must lie in (0, 1).")


@dataclass(frozen=True)
class ContinuousConfig:
    """Spline-test settings for the continuous branch."""

    spline_df: int = 3
    spline_degree: int = 3
    alpha_fdr: float = 0.05
    min_samples: int = 20
    robust_amp_q_low: float = 0.05
    robust_amp_q_high: float = 0.95

    def __post_init__(self) -> None:
        for name in ("spline_df", "spline_degree", "min_samples"):
            _require_integer(name, getattr(self, name))
        if self.spline_df < self.spline_degree:
            raise ValueError("spline_df must be >= spline_degree.")
        if self.min_samples < 3:
            raise ValueError("min_samples must be at least 3.")
        if not (0.0 < self.alpha_fdr < 1.0):
            raise ValueError("alpha_fdr must lie in (0, 1).")
        if not (0.0 <= self.robust_amp_q_low < self.robust_amp_q_high <= 1.0):
            raise ValueError("robust amplitude quantiles must satisfy 0 <= low < high <= 1.")


@dataclass(frozen=True)
class PermutationInferenceConfig:
    """Selection-aware full-pipeline time-permutation inference settings.

    ``batch_size`` controls how many time permutations are processed together
    during the vectorized pass. It changes memory/runtime only; it does not
    change the permutation set or inferential definition.
    """

    n_permutations: int = 1000
    random_seed: int = 0
    batch_size: int = 128

    def __post_init__(self) -> None:
        for name in ("n_permutations", "random_seed", "batch_size"):
            _require_integer(name, getattr(self, name))
        if self.random_seed < 0:
            raise ValueError("random_seed must be >= 0.")
        if self.n_permutations < 1:
            raise ValueError("n_permutations must be >= 1.")
        if self.batch_size < 1:
            raise ValueError("batch_size must be >= 1.")


@dataclass(frozen=True)
class SelectionConfig:
    """Final union/ranking settings for production branches."""

    top_n_genes: Optional[int] = None

    def __post_init__(self) -> None:
        if self.top_n_genes is not None:
            _require_integer("top_n_genes", self.top_n_genes)
        if self.top_n_genes is not None and self.top_n_genes < 1:
            raise ValueError("top_n_genes must be None or >= 1.")


@dataclass(frozen=True)
class IsoVitaConfig:
    """Top-level configuration for IsoVita.

    ``filtering`` selects "isoplex" (default) or "none" for prefiltered
    components. ``isoplex`` settings are ignored in the latter mode.
    ``time`` denotes any finite continuous coordinate, not necessarily time.

    ``windows`` is intentionally ``None`` by default. It is required only when
    the optional ``localized`` analysis mode is explicitly requested.
    """

    isoplex: IsoPlexConfig = field(default_factory=IsoPlexConfig)
    balance: BalanceConfig = field(default_factory=BalanceConfig)
    discrete: DiscreteConfig = field(default_factory=DiscreteConfig)
    continuous: ContinuousConfig = field(default_factory=ContinuousConfig)
    inference: PermutationInferenceConfig = field(default_factory=PermutationInferenceConfig)
    selection: SelectionConfig = field(default_factory=SelectionConfig)
    windows: Optional[WindowConfig] = None
    filtering: str = "isoplex"

    def __post_init__(self) -> None:
        if self.filtering not in ("isoplex", "none"):
            raise ValueError("filtering must be 'isoplex' or 'none'.")
