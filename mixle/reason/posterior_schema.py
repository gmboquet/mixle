"""Semantic schema for IC-1 posteriors -- the missing layer under the frozen ``Posterior`` protocol.

``mixle.reason.posterior_protocol.Posterior`` fixes the *mechanics* of a posterior (``.samples`` /
``.mean`` / ``.cov`` over a ``d``-vector) but says nothing about what the ``d`` axes *mean* -- their
names, units, or coordinate space. That gap is a real, observed source of silent bugs: two posteriors
can both "satisfy IC-1" while their axes mean different things (the concrete case that motivated this:
G2's ``invert_source`` reports ``(x, y, rate)`` while G6's ``design_monitoring_network`` expects
``(x, y, log_rate, onset)``, and composing them needed a hand-written transform bridge that nothing
validated -- if a future edit reordered an axis or changed a unit, no error would fire; the numbers
would just be wrong).

This module adds, additively and opt-in (it changes no existing posterior), three things:

  * :class:`PosteriorSchema` / :class:`AxisSpec` -- declare what each latent axis *is*.
  * :func:`adapt` -- convert a posterior from one schema's convention to another's (reorder,
    coordinate-transform under an explicit supported probability model, marginalize dropped axes), and
    **raise loudly** when a target axis has no source axis to build it from, instead of silently
    producing a wrong number. This is the general, validated replacement for hand-written bridges.
  * :func:`join_independent` -- block-diagonally combine independent posteriors (e.g. an ``(x, y,
    rate)`` inversion posterior plus a separately-estimated ``onset`` belief with no cross-covariance
    to it) into one schematized posterior. This is the honest way to *append* a belief that genuinely
    isn't in another posterior's covariance -- distinct from :func:`adapt`, which only rearranges
    information already present.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

__all__ = [
    "AdaptationReceipt",
    "AxisSpec",
    "PosteriorSchema",
    "SchematizedPosterior",
    "adapt",
    "join_independent",
]

_SPACES = ("linear", "log")


@dataclass(frozen=True)
class AxisSpec:
    """One latent axis's meaning: the natural quantity ``name``, its ``unit``, and the coordinate
    ``space`` the posterior's mean/cov are expressed in for this axis (``"linear"`` or ``"log"``).

    Example: G2 stores the release rate directly -> ``AxisSpec("rate", "kg/s", "linear")``; G6 wants
    it log-transformed -> ``AxisSpec("rate", "kg/s", "log")``. Same ``name``/``unit`` (it's the same
    physical quantity), different ``space`` -- which is exactly the information :func:`adapt` needs,
    together with an explicit probability model, to build the transform.
    """

    name: str
    unit: str = ""
    space: str = "linear"

    def __post_init__(self) -> None:
        if not isinstance(self.name, str) or not self.name.strip():
            raise ValueError("AxisSpec.name must be a non-empty string.")
        if not isinstance(self.unit, str):
            raise TypeError("AxisSpec.unit must be a string.")
        if self.space not in _SPACES:
            raise ValueError(f"AxisSpec.space must be one of {_SPACES}; got {self.space!r}")


@dataclass(frozen=True)
class PosteriorSchema:
    """An ordered list of :class:`AxisSpec` -- the semantic contract for a posterior's ``d`` axes."""

    axes: tuple[AxisSpec, ...]

    def __post_init__(self) -> None:
        axes = tuple(self.axes)
        if not axes:
            raise ValueError("PosteriorSchema requires at least one semantic axis.")
        if any(not isinstance(axis, AxisSpec) for axis in axes):
            raise TypeError("PosteriorSchema.axes must contain AxisSpec values.")
        object.__setattr__(self, "axes", axes)
        names = [a.name for a in self.axes]
        if len(set(names)) != len(names):
            raise ValueError(f"duplicate axis names in schema: {names}")

    @property
    def names(self) -> list[str]:
        return [a.name for a in self.axes]

    @property
    def arity(self) -> int:
        return len(self.axes)

    def index(self, name: str) -> int:
        return self.names.index(name)

    def validate(self, mean: np.ndarray, cov: np.ndarray) -> None:
        """Raise if ``mean``/``cov`` don't match this schema's arity -- the check that turns a silent
        convention mismatch into a loud, early error."""
        try:
            mean = np.asarray(mean, dtype=float)
            cov = np.asarray(cov, dtype=float)
        except (TypeError, ValueError) as exc:
            raise ValueError("posterior mean and covariance must be numeric") from exc
        if mean.shape != (self.arity,):
            raise ValueError(f"mean has shape {mean.shape} but schema declares {self.arity} axes {self.names}")
        if cov.shape != (self.arity, self.arity):
            raise ValueError(f"cov has shape {cov.shape} but schema declares {self.arity} axes {self.names}")
        if not np.all(np.isfinite(mean)) or not np.all(np.isfinite(cov)):
            raise ValueError("posterior mean and covariance must contain only finite values")
        if not np.allclose(cov, cov.T, rtol=1e-10, atol=1e-12):
            raise ValueError("posterior covariance must be symmetric")
        if np.any(np.diag(cov) < 0.0):
            raise ValueError("posterior covariance cannot contain negative marginal variance")
        scale = np.sqrt(np.maximum(np.diag(cov), np.finfo(float).tiny))
        correlation = cov / np.outer(scale, scale)
        eigenvalues = np.linalg.eigvalsh((correlation + correlation.T) / 2.0)
        tolerance = np.finfo(float).eps * max(1, self.arity) * 100.0
        if float(np.min(eigenvalues)) < -tolerance:
            raise ValueError("posterior covariance must be positive semidefinite")


@dataclass(frozen=True)
class AdaptationReceipt:
    """Assumption and numerical status of a coordinate-space adaptation."""

    transformation_model: str
    transformed_axes: tuple[str, ...]
    approximation_error: float
    assumption: str


@dataclass
class SchematizedPosterior:
    """A Gaussian posterior (mean + cov) carrying its :class:`PosteriorSchema`. Satisfies the IC-1
    ``Posterior`` protocol (``.samples`` / ``.mean`` / ``.cov`` / ``.credible_interval`` /
    ``.derived_quantity``) so it's a drop-in wherever a ``Posterior`` is expected, and additionally
    exposes ``.schema`` so downstream code can check conventions instead of assuming them."""

    mean: np.ndarray
    cov: np.ndarray
    schema: PosteriorSchema
    prior_dominated: bool | None = None
    adaptation_receipt: AdaptationReceipt | None = None

    def __post_init__(self) -> None:
        self.mean = np.asarray(self.mean, dtype=float)
        self.cov = np.asarray(self.cov, dtype=float)
        self.schema.validate(self.mean, self.cov)
        if self.prior_dominated is not None and not isinstance(self.prior_dominated, (bool, np.bool_)):
            raise TypeError("prior_dominated must be True, False, or None when the diagnostic is unavailable.")
        if self.prior_dominated is not None:
            self.prior_dominated = bool(self.prior_dominated)

    def samples(self, n: int, rng: np.random.Generator) -> np.ndarray:
        count = _positive_count(n, "n")
        if not hasattr(rng, "multivariate_normal") or not callable(rng.multivariate_normal):
            raise TypeError("rng must provide multivariate_normal(mean, cov, size).")
        draws = np.asarray(rng.multivariate_normal(self.mean, self.cov, size=count), dtype=float)
        if draws.shape != (count, self.schema.arity) or not np.isfinite(draws).all():
            raise ValueError(
                f"rng must return finite posterior draws with shape {(count, self.schema.arity)}, got {draws.shape}."
            )
        return draws

    def credible_interval(self, level: float) -> tuple[np.ndarray, np.ndarray]:
        from scipy.stats import norm

        if not np.isfinite(level) or not 0.0 < level < 1.0:
            raise ValueError("credible interval level must be finite and strictly between 0 and 1")
        z = float(norm.ppf(0.5 + level / 2.0))
        sd = np.sqrt(np.diag(self.cov))
        return self.mean - z * sd, self.mean + z * sd

    def derived_quantity(self, fn: Any, n: int, rng: np.random.Generator) -> Any:
        from mixle.reason.posterior_protocol import DerivedQuantity  # noqa: F401 (documents intent)

        if not callable(fn):
            raise TypeError("derived quantity function must be callable.")
        draws = self.samples(n, rng)
        samples = np.asarray(fn(draws), dtype=float)
        if samples.ndim not in (1, 2) or samples.shape[0] != len(draws):
            raise ValueError("derived quantity must return one scalar or vector row per posterior draw.")
        if samples.ndim == 2 and samples.shape[1] == 0:
            raise ValueError("derived quantity cannot return empty vectors.")
        if not np.isfinite(samples).all():
            raise ValueError("derived quantity samples must contain only finite values.")
        return _SchemaDerivedQuantity(samples=samples, prior_dominated=self.prior_dominated)


@dataclass
class _SchemaDerivedQuantity:
    samples: np.ndarray
    prior_dominated: bool | None

    def require_prior_dominance(self) -> bool:
        """Return the diagnostic or fail closed when inference did not provide it."""
        if self.prior_dominated is None:
            raise RuntimeError("prior-dominance state is unknown; inference supplied no validated diagnostic.")
        return self.prior_dominated

    def credible_interval(self, level: float) -> tuple[np.ndarray, np.ndarray]:
        if not np.isfinite(level) or not 0.0 < level < 1.0:
            raise ValueError("credible interval level must be finite and strictly between 0 and 1")
        a = (1.0 - level) / 2.0
        return np.quantile(self.samples, a, axis=0), np.quantile(self.samples, 1.0 - a, axis=0)


def adapt(
    mean: np.ndarray,
    cov: np.ndarray,
    source: PosteriorSchema,
    target: PosteriorSchema,
    *,
    transformation_model: str | None = None,
    prior_dominated: bool | None = None,
) -> SchematizedPosterior:
    """Convert a Gaussian posterior from ``source`` convention to ``target`` convention.

    Matches target axes to source axes **by name**, applies any supported
    coordinate change under an explicit probability model, reorders to the target order, and marginalizes
    (drops) any source axis not in the target. Raises :class:`KeyError` if a ``target`` axis has no
    source axis of the same name -- the case that must never silently produce a wrong number, because
    the information to build that axis genuinely isn't present (use :func:`join_independent` to append
    a belief that legitimately comes from elsewhere). Raises :class:`ValueError` on a unit mismatch
    for a matched axis -- same name, different unit is a real inconsistency, not an adaptation.

    This is the general, validated replacement for hand-written convention bridges (e.g.
    ``SourcePosterior.to_doe_prior``'s rate->log_rate step).
    """
    source.validate(mean, cov)
    mean = np.asarray(mean, dtype=float)
    cov = np.asarray(cov, dtype=float)

    source_by_name = {a.name: (i, a) for i, a in enumerate(source.axes)}
    missing = [a.name for a in target.axes if a.name not in source_by_name]
    if missing:
        raise KeyError(
            f"target axes {missing} have no source axis to build from (source has {source.names}); "
            f"adapt() never invents information -- use join_independent() to append a belief that "
            f"legitimately comes from a separate posterior."
        )

    conversions: list[tuple[int, str, str]] = []
    for tgt_axis in target.axes:
        si, src_axis = source_by_name[tgt_axis.name]
        if src_axis.unit != tgt_axis.unit:
            raise ValueError(
                f"axis {tgt_axis.name!r} unit mismatch: source {src_axis.unit!r} vs target {tgt_axis.unit!r}"
            )
        if src_axis.space != tgt_axis.space:
            conversions.append((si, src_axis.space, tgt_axis.space))

    receipt = None
    work_mean, work_cov = mean.copy(), cov.copy()
    if conversions:
        directions = {(from_space, to_space) for _, from_space, to_space in conversions}
        if transformation_model != "joint_log_normal":
            raise ValueError(
                "linear/log adaptation requires transformation_model='joint_log_normal'; "
                "Gaussian linear moments alone do not establish positive support."
            )
        if len(directions) != 1:
            raise ValueError("mixed linear-to-log and log-to-linear conversions are not supported in one adaptation.")
        indices = [index for index, _, _ in conversions]
        direction = next(iter(directions))
        if direction == ("linear", "log"):
            work_mean, work_cov = _linear_moments_to_log(work_mean, work_cov, indices)
        elif direction == ("log", "linear"):
            work_mean, work_cov = _log_moments_to_linear(work_mean, work_cov, indices)
        else:
            raise ValueError(f"unsupported coordinate change {direction!r}.")
        receipt = AdaptationReceipt(
            transformation_model=transformation_model,
            transformed_axes=tuple(source.axes[index].name for index in indices),
            approximation_error=0.0,
            assumption="moments are exact only under the declared joint log-normal/normal model",
        )

    order = [source_by_name[a.name][0] for a in target.axes]
    new_mean = work_mean[order]
    new_cov = work_cov[np.ix_(order, order)]
    return SchematizedPosterior(
        new_mean,
        new_cov,
        target,
        prior_dominated=prior_dominated,
        adaptation_receipt=receipt,
    )


def _linear_moments_to_log(
    mean: np.ndarray,
    cov: np.ndarray,
    indices: list[int],
) -> tuple[np.ndarray, np.ndarray]:
    """Recover joint normal moments from declared log-normal/normal moments."""
    source_mean = mean.copy()
    source_cov = cov.copy()
    result_mean = mean.copy()
    result_cov = cov.copy()
    transformed = set(indices)
    for index in indices:
        value = source_mean[index]
        if value <= 0.0:
            raise ValueError(f"cannot log-transform axis {index}: declared log-normal mean {value} is not positive.")
        variance = source_cov[index, index]
        log_variance = float(np.log1p(variance / value**2))
        result_mean[index] = float(np.log(value) - 0.5 * log_variance)
    for i in indices:
        for j in range(len(mean)):
            if j in transformed:
                ratio = source_cov[i, j] / (source_mean[i] * source_mean[j])
                if ratio <= -1.0:
                    raise ValueError("linear moments are incompatible with a joint log-normal covariance.")
                value = float(np.log1p(ratio))
            else:
                value = float(source_cov[i, j] / source_mean[i])
            result_cov[i, j] = value
            result_cov[j, i] = value
    return result_mean, result_cov


def _log_moments_to_linear(
    mean: np.ndarray,
    cov: np.ndarray,
    indices: list[int],
) -> tuple[np.ndarray, np.ndarray]:
    """Compute exact log-normal/normal moments from joint normal moments."""
    source_mean = mean.copy()
    source_cov = cov.copy()
    result_mean = mean.copy()
    result_cov = cov.copy()
    transformed = set(indices)
    linear_means = {
        index: float(np.exp(source_mean[index] + 0.5 * source_cov[index, index]))
        for index in indices
    }
    for index, value in linear_means.items():
        result_mean[index] = value
    for i in indices:
        for j in range(len(mean)):
            if j in transformed:
                value = linear_means[i] * linear_means[j] * float(np.expm1(source_cov[i, j]))
            else:
                value = linear_means[i] * float(source_cov[i, j])
            result_cov[i, j] = value
            result_cov[j, i] = value
    return result_mean, result_cov


def _positive_count(value: Any, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, (int, np.integer)) or int(value) <= 0:
        raise ValueError(f"{name} must be a positive integer.")
    return int(value)


def join_independent(*blocks: tuple[np.ndarray, np.ndarray, PosteriorSchema]) -> SchematizedPosterior:
    """Block-diagonally combine independent Gaussian posteriors into one schematized posterior.

    Each block is ``(mean, cov, schema)``; the result stacks the means and places the covariances on
    the block diagonal (zero cross-covariance between blocks -- the explicit statement that these
    beliefs are independent, which is exactly the honest way to *append* an axis like ``onset`` that a
    separate estimate provides and is genuinely uncorrelated with the rest, rather than pretending a
    single posterior's covariance contained it). Raises on a duplicate axis name across blocks.
    """
    if not blocks:
        raise ValueError("join_independent requires at least one block")
    means, covs, all_axes = [], [], []
    for mean, cov, schema in blocks:
        schema.validate(mean, cov)
        means.append(np.asarray(mean, dtype=float))
        covs.append(np.asarray(cov, dtype=float))
        all_axes.extend(schema.axes)
    names = [a.name for a in all_axes]
    if len(set(names)) != len(names):
        raise ValueError(f"join_independent: duplicate axis name across blocks: {names}")

    d = sum(m.shape[0] for m in means)
    joined_cov = np.zeros((d, d))
    offset = 0
    for c in covs:
        k = c.shape[0]
        joined_cov[offset : offset + k, offset : offset + k] = c
        offset += k
    return SchematizedPosterior(np.concatenate(means), joined_cov, PosteriorSchema(tuple(all_axes)))
