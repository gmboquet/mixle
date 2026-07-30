"""Per-edge premise checks for cross-modal transport.

Every real modality edge should prove that a plain conditional transport is
usable and calibrated on that edge's own data before the edge is trusted in a
belief graph. Calibration is checked against held-out truth for the edge rather
than transferred from unrelated examples.

This module exposes a reusable per-edge check with two decisions:

* premise fails: the edge should not be used for cross-modal purposes;
* premise passes: the plain conditional transport may be composed as-is.

The check combines held-out coverage, conditional subgroup coverage, and
interval sharpness. A low-power goodness-of-fit test or a trivially wide
interval is not affirmative evidence.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field
from datetime import UTC, datetime
from enum import StrEnum

import numpy as np
from scipy.stats import beta, binomtest

from mixle.inference import optimize
from mixle.models.mixture_density import NeuralConditionalDensity, build_mdn

ALPHA = 0.10  # 90% nominal credible-interval coverage
COVERAGE_P_FLOOR = 0.01  # p-value floor below which coverage is inconsistent with nominal
COVERAGE_TOLERANCE = 0.10
SUBGROUP_TOLERANCE = 0.20
MAX_RELATIVE_INTERVAL_WIDTH = 2.0
COVERAGE_BOUND_ERROR_RATE = 0.05  # one-sided error rate of the coverage lower confidence bound


def _smallest_decidable_holdout(limit: int = 10_000) -> int:
    """Smallest holdout size at which this module's own coverage gate can return PASS.

    The gate requires a one-sided lower confidence bound on the coverage rate to reach
    ``1 - ALPHA - COVERAGE_TOLERANCE``. That bound tightens with sample size, so below some size it
    cannot reach the threshold *even for a sampler whose coverage is exactly nominal* -- the verifier
    would be accepting input on which its only possible verdict is FAIL. At the published constants
    that size is 44; a hardcoded 30 admitted 30-43, a band where a flawless edge is rejected for being
    measured too little. Deriving the minimum from the gate keeps admission and decision consistent
    when either constant changes.
    """
    threshold = 1.0 - ALPHA - COVERAGE_TOLERANCE
    for total in range(2, limit + 1):
        hits = int(round((1.0 - ALPHA) * total))
        if hits and float(beta.ppf(COVERAGE_BOUND_ERROR_RATE, hits, total - hits + 1)) >= threshold:
            return total
    raise ValueError(  # pragma: no cover -- unreachable for any sane ALPHA/COVERAGE_TOLERANCE pair
        f"no holdout size up to {limit} can satisfy a coverage bound of {threshold}; "
        "ALPHA and COVERAGE_TOLERANCE are jointly unsatisfiable."
    )


MIN_HOLDOUT_ROWS = _smallest_decidable_holdout()


class PremiseStatus(StrEnum):
    """Evidence state for an edge premise."""

    PASS = "pass"
    FAIL = "fail"
    INCONCLUSIVE = "inconclusive"
    INVALID = "invalid"


@dataclass(frozen=True)
class PremiseReceipt:
    """Tamper-evident identity and outcome of one held-out edge check."""

    edge_name: str
    verifier: str
    dataset_digest: str
    metric: str
    evaluated_at: str
    status: PremiseStatus

    def __post_init__(self) -> None:
        for field_name in ("edge_name", "verifier", "metric"):
            value = getattr(self, field_name)
            if not isinstance(value, str) or not value.strip():
                raise ValueError(f"{field_name} must be a non-empty string.")
        if not isinstance(self.dataset_digest, str):
            raise TypeError("dataset_digest must be a string.")
        prefix, separator, hex_digest = self.dataset_digest.partition(":")
        if prefix != "sha256" or separator != ":" or len(hex_digest) != 64:
            raise ValueError("dataset_digest must be a sha256:<64 lowercase hex> content digest.")
        try:
            if hex_digest != hex_digest.lower():
                raise ValueError
            int(hex_digest, 16)
        except ValueError as exc:
            raise ValueError("dataset_digest must be a sha256:<64 lowercase hex> content digest.") from exc
        try:
            timestamp = datetime.fromisoformat(self.evaluated_at)
        except (TypeError, ValueError) as exc:
            raise ValueError("evaluated_at must be an ISO-8601 timestamp.") from exc
        if timestamp.tzinfo is None:
            raise ValueError("evaluated_at must include a timezone.")
        if not isinstance(self.status, PremiseStatus):
            raise TypeError("status must be a PremiseStatus.")

    @property
    def passed(self) -> bool:
        """Whether the premise supplied affirmative evidence for composition."""
        return self.status is PremiseStatus.PASS


def fit_conditional_transport(
    data,
    *,
    x_dim: int,
    y_dim: int,
    k: int = 3,
    max_its: int = 30,
    m_steps: int = 80,
    lr: float = 3e-3,
    seed: int = 0,
    delta: float | None = 1.0e-9,
    reuse_estep_ll: bool = True,
):
    """Fit ``p(cond | target)`` and return a sampler with ``sample_given``.

    Uses :func:`mixle.models.mixture_density.build_mdn` and
    :class:`~mixle.models.mixture_density.NeuralConditionalDensity`, fit through :func:`optimize`.
    Pass ``delta=None, reuse_estep_ll=False`` for an edge whose relationship needs the full iteration
    budget rather than early stopping. The seed is applied before module
    construction, to optimizer data order, and to the returned sampler. Exact
    bitwise replay can still depend on device-specific nondeterministic kernels.
    """
    import torch

    if isinstance(seed, bool) or not isinstance(seed, (int, np.integer)):
        raise ValueError("seed must be an integer.")
    torch.manual_seed(int(seed))
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(int(seed))
    module = build_mdn(x_dim=x_dim, y_dim=y_dim, k=k, hidden=32, layers=2)
    leaf = NeuralConditionalDensity(module, m_steps=m_steps, lr=lr)
    fitted = optimize(
        data,
        leaf.estimator(),
        max_its=max_its,
        delta=delta,
        reuse_estep_ll=reuse_estep_ll,
        out=None,
        rng=np.random.RandomState(int(seed)),
    )
    return fitted.sampler(seed=int(seed))


def marginal_coverage(sampler, x_test, y_test, *, n_draws: int = 200):
    """Return per-dimension credible-interval coverage flags."""
    x_test, y_test, n_draws = _validated_holdout(x_test, y_test, n_draws)
    d = x_test.shape[1]
    covered = [[] for _ in range(d)]
    for i in range(len(x_test)):
        # one batched forward pass for all n_draws of THIS point, instead of n_draws individual calls
        y_batch = np.repeat(np.atleast_2d(np.asarray(y_test[i], dtype=float)), n_draws, axis=0)
        draws = np.asarray(sampler.sample_given_batch(y_batch), dtype=float)
        if draws.shape != (n_draws, d) or not np.isfinite(draws).all():
            raise ValueError(f"edge sampler must return finite draws with shape {(n_draws, d)}, got {draws.shape}")
        lo = np.quantile(draws, ALPHA / 2, axis=0)
        hi = np.quantile(draws, 1 - ALPHA / 2, axis=0)
        for dim in range(d):
            covered[dim].append(bool(lo[dim] <= x_test[i, dim] <= hi[dim]))
    return covered


def coverage_consistent_with_nominal(covered_flags) -> tuple[float, float]:
    """``(observed_rate, p_value)`` for a two-sided binomial test of coverage against ``1 - ALPHA``."""
    n = len(covered_flags)
    hits = int(sum(covered_flags))
    p = float(binomtest(hits, n, 1.0 - ALPHA).pvalue)
    return hits / n, p


@dataclass
class EdgeTransportVerdict:
    """Premise decision for one real modality edge, computed on that edge."""

    edge_name: str
    status: PremiseStatus
    receipt: PremiseReceipt
    coverage_rates: list[float] = field(default_factory=list)
    coverage_lower_bounds: list[float] = field(default_factory=list)
    relative_interval_widths: list[float] = field(default_factory=list)
    subgroup_coverage_rates: list[list[float]] = field(default_factory=list)
    p_values: list[float] = field(default_factory=list)
    reason: str = ""

    @property
    def usable(self) -> bool:
        """Compatibility alias for an affirmative premise verdict."""
        return self.status is PremiseStatus.PASS


def verify_edge_transport(edge_name: str, sampler, x_test, y_test, *, n_draws: int = 200) -> EdgeTransportVerdict:
    """Check calibration *and* sharpness on aligned held-out edge data.

    A passing result requires enough holdout rows, a one-sided confidence lower
    bound above the declared coverage floor, acceptable conditional subgroup
    coverage, and intervals narrower than a data-only baseline. Merely failing
    to reject a binomial null is never treated as evidence of calibration.
    """
    if not isinstance(edge_name, str) or not edge_name.strip():
        raise ValueError("edge_name must be a non-empty string.")
    x_test, y_test, n_draws = _validated_holdout(x_test, y_test, n_draws)
    dataset_digest = _holdout_digest(x_test, y_test)
    evaluated_at = datetime.now(UTC).isoformat()
    metric = "marginal:coverage_lower_bound+conditional_subgroups+relative_interval_width"
    if len(x_test) < MIN_HOLDOUT_ROWS:
        return _verdict(
            edge_name,
            PremiseStatus.INCONCLUSIVE,
            dataset_digest,
            metric,
            evaluated_at,
            reason=f"need at least {MIN_HOLDOUT_ROWS} held-out rows, got {len(x_test)}",
        )

    d = x_test.shape[1]
    covered = [[] for _ in range(d)]
    widths = [[] for _ in range(d)]
    for i in range(len(x_test)):
        y_batch = np.repeat(y_test[i : i + 1], n_draws, axis=0)
        draws = np.asarray(sampler.sample_given_batch(y_batch), dtype=float)
        if draws.shape != (n_draws, d) or not np.isfinite(draws).all():
            return _verdict(
                edge_name,
                PremiseStatus.INVALID,
                dataset_digest,
                metric,
                evaluated_at,
                reason=f"sampler returned non-finite or mis-shaped draws: expected {(n_draws, d)}, got {draws.shape}",
            )
        lo = np.quantile(draws, ALPHA / 2.0, axis=0)
        hi = np.quantile(draws, 1.0 - ALPHA / 2.0, axis=0)
        for dim in range(d):
            covered[dim].append(bool(lo[dim] <= x_test[i, dim] <= hi[dim]))
            widths[dim].append(float(hi[dim] - lo[dim]))

    rates, lower_bounds, relative_widths, p_values = [], [], [], []
    for dim_covered in covered:
        rate, p = coverage_consistent_with_nominal(dim_covered)
        rates.append(rate)
        p_values.append(p)
        lower_bounds.append(_binomial_lower_bound(int(sum(dim_covered)), len(dim_covered)))
    for dim in range(d):
        baseline_width = float(
            np.quantile(x_test[:, dim], 1.0 - ALPHA / 2.0) - np.quantile(x_test[:, dim], ALPHA / 2.0)
        )
        scale = max(baseline_width, float(np.std(x_test[:, dim])), np.finfo(float).eps)
        relative_widths.append(float(np.median(widths[dim])) / scale)

    subgroup_rates = _conditional_subgroup_coverage(covered, y_test)
    low_dims = [i for i, bound in enumerate(lower_bounds) if bound < 1.0 - ALPHA - COVERAGE_TOLERANCE]
    wide_dims = [i for i, width in enumerate(relative_widths) if width > MAX_RELATIVE_INTERVAL_WIDTH]
    bad_subgroups = [
        (group, dim)
        for group, rates_for_group in enumerate(subgroup_rates)
        for dim, rate in enumerate(rates_for_group)
        if rate < 1.0 - ALPHA - SUBGROUP_TOLERANCE
    ]
    failures = []
    if low_dims:
        failures.append(f"coverage confidence bound below tolerance on dim(s): {low_dims}")
    if wide_dims:
        failures.append(f"intervals uninformative on dim(s): {wide_dims}")
    if bad_subgroups:
        failures.append(f"conditional subgroup coverage below tolerance: {bad_subgroups}")
    status = PremiseStatus.FAIL if failures else PremiseStatus.PASS
    return _verdict(
        edge_name,
        status,
        dataset_digest,
        metric,
        evaluated_at,
        coverage_rates=rates,
        coverage_lower_bounds=lower_bounds,
        relative_interval_widths=relative_widths,
        subgroup_coverage_rates=subgroup_rates,
        p_values=p_values,
        reason="; ".join(failures),
    )


def _validated_holdout(x_test, y_test, n_draws: int) -> tuple[np.ndarray, np.ndarray, int]:
    if isinstance(n_draws, bool) or not isinstance(n_draws, (int, np.integer)) or int(n_draws) <= 0:
        raise ValueError("n_draws must be a positive integer.")
    x = np.asarray(x_test, dtype=float)
    y = np.asarray(y_test, dtype=float)
    if x.ndim != 2 or y.ndim != 2 or not len(x) or len(x) != len(y):
        raise ValueError("x_test and y_test must be non-empty aligned two-dimensional arrays.")
    if not np.isfinite(x).all() or not np.isfinite(y).all():
        raise ValueError("held-out arrays must contain only finite values.")
    return x, y, int(n_draws)


def _holdout_digest(x_test: np.ndarray, y_test: np.ndarray) -> str:
    digest = hashlib.sha256()
    for name, value in (("x", x_test), ("y", y_test)):
        header = json.dumps(
            {"name": name, "shape": value.shape, "dtype": value.dtype.str},
            sort_keys=True,
            separators=(",", ":"),
        ).encode()
        digest.update(len(header).to_bytes(8, "big"))
        digest.update(header)
        digest.update(np.ascontiguousarray(value).tobytes())
    return f"sha256:{digest.hexdigest()}"


def _binomial_lower_bound(hits: int, total: int, *, error_rate: float = COVERAGE_BOUND_ERROR_RATE) -> float:
    if hits == 0:
        return 0.0
    return float(beta.ppf(error_rate, hits, total - hits + 1))


def _conditional_subgroup_coverage(covered: list[list[bool]], y_test: np.ndarray) -> list[list[float]]:
    order = np.argsort(y_test[:, 0], kind="stable")
    groups = np.array_split(order, 2)
    return [[float(np.mean(np.asarray(dim_covered, dtype=bool)[group])) for dim_covered in covered] for group in groups]


def _verdict(
    edge_name: str,
    status: PremiseStatus,
    dataset_digest: str,
    metric: str,
    evaluated_at: str,
    **kwargs,
) -> EdgeTransportVerdict:
    receipt = PremiseReceipt(
        edge_name=edge_name,
        verifier="mixle.reason.transport_edge.verify_edge_transport@0.8",
        dataset_digest=dataset_digest,
        metric=metric,
        evaluated_at=evaluated_at,
        status=status,
    )
    return EdgeTransportVerdict(edge_name=edge_name, status=status, receipt=receipt, **kwargs)


__all__ = [
    "ALPHA",
    "COVERAGE_P_FLOOR",
    "COVERAGE_TOLERANCE",
    "EdgeTransportVerdict",
    "MAX_RELATIVE_INTERVAL_WIDTH",
    "MIN_HOLDOUT_ROWS",
    "PremiseReceipt",
    "PremiseStatus",
    "coverage_consistent_with_nominal",
    "fit_conditional_transport",
    "marginal_coverage",
    "verify_edge_transport",
]
