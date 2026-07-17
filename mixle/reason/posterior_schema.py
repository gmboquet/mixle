"""Semantic schema for IC-1 posteriors -- the missing layer under the frozen ``Posterior`` protocol.

``mixle.reason.posterior_protocol.Posterior`` fixes the *mechanics* of a posterior (``.samples`` /
``.mean`` / ``.cov`` over a ``d``-vector) but says nothing about what the ``d`` axes *mean* -- their
names, units, or coordinate space. That gap is a real, observed source of silent bugs: two posteriors
can both "satisfy IC-1" while their axes mean different things (the concrete case that motivated this:
G2's ``invert_source`` reports ``(x, y, rate)`` while G6's ``design_monitoring_network`` expects
``(x, y, log_rate, onset)``, and composing them needed a hand-written delta-method bridge that nothing
validated -- if a future edit reordered an axis or changed a unit, no error would fire; the numbers
would just be wrong).

This module adds, additively and opt-in (it changes no existing posterior), three things:

  * :class:`PosteriorSchema` / :class:`AxisSpec` -- declare what each latent axis *is*.
  * :func:`adapt` -- convert a posterior from one schema's convention to another's (reorder,
    coordinate-transform e.g. linear<->log via the delta method, marginalize dropped axes), and
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

__all__ = ["AxisSpec", "PosteriorSchema", "SchematizedPosterior", "adapt", "join_independent"]

_SPACES = ("linear", "log")


@dataclass(frozen=True)
class AxisSpec:
    """One latent axis's meaning: the natural quantity ``name``, its ``unit``, and the coordinate
    ``space`` the posterior's mean/cov are expressed in for this axis (``"linear"`` or ``"log"``).

    Example: G2 stores the release rate directly -> ``AxisSpec("rate", "kg/s", "linear")``; G6 wants
    it log-transformed -> ``AxisSpec("rate", "kg/s", "log")``. Same ``name``/``unit`` (it's the same
    physical quantity), different ``space`` -- which is exactly the information :func:`adapt` needs to
    build the (delta-method) transform automatically instead of by hand.
    """

    name: str
    unit: str = ""
    space: str = "linear"

    def __post_init__(self) -> None:
        if self.space not in _SPACES:
            raise ValueError(f"AxisSpec.space must be one of {_SPACES}; got {self.space!r}")


@dataclass(frozen=True)
class PosteriorSchema:
    """An ordered list of :class:`AxisSpec` -- the semantic contract for a posterior's ``d`` axes."""

    axes: tuple[AxisSpec, ...]

    def __post_init__(self) -> None:
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
        eigenvalues = np.linalg.eigvalsh((cov + cov.T) / 2.0)
        tolerance = 1e-12 * max(1.0, float(np.max(np.abs(eigenvalues))))
        if float(np.min(eigenvalues)) < -tolerance:
            raise ValueError("posterior covariance must be positive semidefinite")


@dataclass
class SchematizedPosterior:
    """A Gaussian posterior (mean + cov) carrying its :class:`PosteriorSchema`. Satisfies the IC-1
    ``Posterior`` protocol (``.samples`` / ``.mean`` / ``.cov`` / ``.credible_interval`` /
    ``.derived_quantity``) so it's a drop-in wherever a ``Posterior`` is expected, and additionally
    exposes ``.schema`` so downstream code can check conventions instead of assuming them."""

    mean: np.ndarray
    cov: np.ndarray
    schema: PosteriorSchema

    def __post_init__(self) -> None:
        self.mean = np.asarray(self.mean, dtype=float)
        self.cov = np.asarray(self.cov, dtype=float)
        self.schema.validate(self.mean, self.cov)

    def samples(self, n: int, rng: np.random.Generator) -> np.ndarray:
        return rng.multivariate_normal(self.mean, self.cov, size=int(n))

    def credible_interval(self, level: float) -> tuple[np.ndarray, np.ndarray]:
        from scipy.stats import norm

        if not np.isfinite(level) or not 0.0 < level < 1.0:
            raise ValueError("credible interval level must be finite and strictly between 0 and 1")
        z = float(norm.ppf(0.5 + level / 2.0))
        sd = np.sqrt(np.diag(self.cov))
        return self.mean - z * sd, self.mean + z * sd

    def derived_quantity(self, fn: Any, n: int, rng: np.random.Generator) -> Any:
        from mixle.reason.posterior_protocol import DerivedQuantity  # noqa: F401 (documents intent)

        draws = self.samples(n, rng)
        samples = np.asarray(fn(draws))
        return _SchemaDerivedQuantity(samples=samples)


@dataclass
class _SchemaDerivedQuantity:
    samples: np.ndarray
    prior_dominated: bool = False

    def credible_interval(self, level: float) -> tuple[np.ndarray, np.ndarray]:
        if not np.isfinite(level) or not 0.0 < level < 1.0:
            raise ValueError("credible interval level must be finite and strictly between 0 and 1")
        a = (1.0 - level) / 2.0
        return np.quantile(self.samples, a, axis=0), np.quantile(self.samples, 1.0 - a, axis=0)


def _transform_axis(
    mean: np.ndarray, cov: np.ndarray, i: int, from_space: str, to_space: str
) -> tuple[np.ndarray, np.ndarray]:
    """Delta-method coordinate change of a single axis ``i`` from ``from_space`` to ``to_space``,
    updating both the mean and the full covariance row/column for that axis. Only linear<->log is
    implemented (the case that actually arises); anything else raises rather than silently no-op'ing."""
    if from_space == to_space:
        return mean, cov
    mean = mean.copy()
    cov = cov.copy()
    if from_space == "linear" and to_space == "log":
        x = mean[i]
        if x <= 0:
            raise ValueError(f"cannot log-transform axis {i}: mean {x} is not positive")
        # y = log x: dy/dx = 1/x. mean_y = log(mean_x); Cov scales by the Jacobian on axis i.
        j = 1.0 / x
        mean[i] = np.log(x)
    elif from_space == "log" and to_space == "linear":
        # x = exp(y): dx/dy = exp(y) = mean_x. mean_x = exp(mean_y).
        j = np.exp(mean[i])
        mean[i] = j
    else:
        raise ValueError(f"unsupported coordinate change {from_space!r} -> {to_space!r} (only linear<->log)")
    cov[i, :] *= j
    cov[:, i] *= j
    return mean, cov


def adapt(mean: np.ndarray, cov: np.ndarray, source: PosteriorSchema, target: PosteriorSchema) -> SchematizedPosterior:
    """Convert a Gaussian posterior from ``source`` convention to ``target`` convention.

    Matches target axes to source axes **by name**, applies any per-axis coordinate change (linear
    <-> log, via the delta method on the covariance), reorders to the target order, and marginalizes
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

    work_mean, work_cov = mean.copy(), cov.copy()
    for tgt_axis in target.axes:
        si, src_axis = source_by_name[tgt_axis.name]
        if src_axis.unit != tgt_axis.unit:
            raise ValueError(
                f"axis {tgt_axis.name!r} unit mismatch: source {src_axis.unit!r} vs target {tgt_axis.unit!r}"
            )
        if src_axis.space != tgt_axis.space:
            work_mean, work_cov = _transform_axis(work_mean, work_cov, si, src_axis.space, tgt_axis.space)

    order = [source_by_name[a.name][0] for a in target.axes]
    new_mean = work_mean[order]
    new_cov = work_cov[np.ix_(order, order)]
    return SchematizedPosterior(new_mean, new_cov, target)


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
