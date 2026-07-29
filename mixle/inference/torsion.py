"""Twisted composition for mixture components with a shared base density.

Mixture components can share one base density modulo a declared group action,
instead of each independently learning a density from scratch on its own slice
of data. Concretely, :class:`CyclicGroup`
acts on a periodic coordinate by rotation of its ``(cos, sin)`` embedding (an exact, Jacobian-1 change of
variables, so a fitted embedding density scores identically whichever group element aligned a point into it);
:func:`fit_twisted_mixture` pools every group's data into ONE shared base density after undoing each group's
twist, so the shared density is fit on the union -- effectively ``|groups|`` times the data for the same
parameter count as fitting one group alone.

The *rotation* is Jacobian-1, but the ``(cos, sin)`` **embedding** is not a change of variables at all: it
maps the 1-D coordinate onto a measure-zero circle in ``R^2``. An ambient 2-D density evaluated on that
circle is therefore NOT a density over ``x`` -- it integrates over one period to an arbitrary constant that
differs from fit to fit. Every scoring entry point here (:meth:`TwistedMixtureResult.log_density`,
:func:`independent_log_density`) divides out that constant via :func:`log_circular_normalizer`, so the
returned values are genuine log densities on ``[0, period)`` and the twisted-vs-independent held-out
likelihood comparison this module exists to make is between comparable quantities.

Use this as an experimental modeling option. If the shared-base model does not
beat independently fit per-group models at matched per-component capacity on
held-out per-group log likelihood, keep the independent baseline as the default.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import Any

import numpy as np

from mixle.utils.special import logsumexp

#: Quadrature points per period used by :func:`log_circular_normalizer`. The integrand is smooth and
#: periodic, so the uniform-grid (periodic trapezoid) rule converges spectrally -- a few thousand nodes
#: pin the normalizer far below any difference the twisted/independent comparison turns on.
NORMALIZER_GRID = 4096


def _exact_int(value: Any) -> int | None:
    """``value`` as a Python ``int`` when it is an exact, non-Boolean integer; ``None`` otherwise.

    ``bool`` is excluded deliberately: ``True == 1`` would make it a silent alias for element 1, and a
    ``{True: ..., 1: ...}`` group mapping would collapse to a single key without anyone noticing.
    """
    if isinstance(value, bool):
        return None
    if isinstance(value, (int, np.integer)):
        return int(value)
    return None


@dataclass(frozen=True)
class CyclicGroup:
    """Z_``order`` acting on a periodic real-valued coordinate of period ``period`` by rotation.

    Each group element ``k`` in ``{0, ..., order - 1}`` is realized concretely as a rotation of the
    coordinate's ``(cos, sin)`` embedding by angle ``2*pi*k/order`` -- an orthogonal (norm- and
    Jacobian-preserving) transform, so composing group elements is exactly addition mod ``order``
    (:meth:`compose`), and a density fit on the embedding is unaffected by which element aligned a point into
    it (the twist is undone before scoring, not baked into the density).

    ``order`` must be an exact positive integer and ``period`` a finite positive number: a zero order
    divides by zero in the rotation law, a negative or fractional one describes no cyclic group at all,
    and a zero/non-finite period makes every embedding NaN. ``{0, ..., order - 1}`` is the whole element
    set, and it is checked -- an out-of-range ``k`` used to be silently wrapped into a *different*
    element, so ``-1``, ``order``, ``order + 1`` and ``True`` all acted as undeclared aliases.
    """

    order: int
    period: float = 1.0

    def __post_init__(self) -> None:
        order = _exact_int(self.order)
        if order is None or order < 1:
            raise ValueError(f"CyclicGroup order must be an exact integer >= 1; got {self.order!r}.")
        period = float(self.period)
        if not np.isfinite(period) or period <= 0.0:
            raise ValueError(f"CyclicGroup period must be finite and positive; got {self.period!r}.")
        # Store what was validated. Both values were coerced, checked, and then dropped, so the
        # instance kept the raw input: CyclicGroup(3, "1.0") constructed happily and then failed
        # somewhere else entirely the first time the period was used as a number. Validation that
        # does not commit its result only moves the error.
        object.__setattr__(self, "order", order)
        object.__setattr__(self, "period", period)

    def element(self, k: Any) -> int:
        """Validate ``k`` as a declared group element and return it as a plain ``int``."""
        idx = _exact_int(k)
        if idx is None or not (0 <= idx < self.order):
            raise ValueError(
                f"{k!r} is not an element of Z_{self.order}; elements are the integers 0..{self.order - 1}."
            )
        return idx

    def embed(self, x: Sequence[float]) -> np.ndarray:
        """The periodic coordinate's ``(cos, sin)`` embedding, shape ``(..., 2)``."""
        theta = 2.0 * np.pi * np.asarray(x, dtype=np.float64) / self.period
        return np.stack([np.cos(theta), np.sin(theta)], axis=-1)

    def _rotation(self, k: int) -> np.ndarray:
        angle = 2.0 * np.pi * (k % self.order) / self.order
        c, s = np.cos(angle), np.sin(angle)
        return np.array([[c, -s], [s, c]])

    def act(self, embedded: np.ndarray, k: int) -> np.ndarray:
        """Rotate an ``(..., 2)`` embedding by group element ``k`` (the forward twist)."""
        return np.asarray(embedded, dtype=np.float64) @ self._rotation(self.element(k)).T

    def inverse_act(self, embedded: np.ndarray, k: int) -> np.ndarray:
        """Undo group element ``k``'s twist -- ``act(inverse_act(v, k), k) == v``."""
        # ``-k`` is the group inverse; name it as the declared element ``(order - k) mod order`` so the
        # rotation is requested by a real element rather than relying on ``act`` wrapping a negative one.
        return self.act(embedded, (-self.element(k)) % self.order)

    def compose(self, k1: int, k2: int) -> int:
        """The group element equivalent to applying ``k1`` then ``k2`` -- addition mod ``order``."""
        return (self.element(k1) + self.element(k2)) % self.order


def _embedding_log_density(density: Any, embedded: np.ndarray) -> np.ndarray:
    """Raw ambient ``log q(v)`` of the 2-D base density at embedded points ``(..., 2)``."""
    enc = density.dist_to_encoder().seq_encode([row for row in embedded])
    return np.asarray(density.seq_log_density(enc), dtype=np.float64)


def log_circular_normalizer(density: Any, group: CyclicGroup, *, grid: int = NORMALIZER_GRID) -> float:
    """``log ∫_0^period q(embed(t)) dt`` -- what turns an ambient embedding score into a density on ``x``.

    ``embed`` places the periodic coordinate on a measure-zero circle in ``R^2``, so an ambient 2-D density
    ``q`` restricted to it does not integrate to one over a period; it integrates to a fit-specific constant.
    Subtracting this log constant from ``log q(embed(x))`` yields a normalized density on ``[0, period)``.

    The value does not depend on the group element: ``inverse_act(embed(x), k) == embed(x - k*period/order)``,
    and integrating over a whole period is invariant to that shift -- so one normalizer serves every ``k``.
    """
    if int(grid) < 2:
        raise ValueError("log_circular_normalizer needs at least 2 quadrature nodes")
    ts = np.arange(int(grid), dtype=np.float64) * (group.period / int(grid))
    log_q = _embedding_log_density(density, group.embed(ts))
    return float(logsumexp(log_q) - np.log(int(grid)) + np.log(abs(float(group.period))))


@dataclass
class TwistedMixtureResult:
    """A single shared base density plus the group whose elements twist it into each group's local factor."""

    base_density: Any
    group: CyclicGroup
    _log_normalizer: float | None = field(default=None, init=False, repr=False, compare=False)

    def log_normalizer(self) -> float:
        """The (cached, ``k``-independent) log constant that normalizes the shared base density over a period."""
        if self._log_normalizer is None:
            self._log_normalizer = log_circular_normalizer(self.base_density, self.group)
        return self._log_normalizer

    def log_density(self, x: Sequence[float], k: int) -> np.ndarray:
        """``log p(x | group=k)``: undo ``k``'s twist, score under the shared base density, normalize.

        A normalized density on the periodic coordinate -- ``exp`` of this integrates to 1 over one period
        -- so values are comparable against :func:`independent_log_density` and across fits.
        """
        aligned = self.group.inverse_act(self.group.embed(x), k)
        return _embedding_log_density(self.base_density, aligned) - self.log_normalizer()


def _fit_density(rows: list[np.ndarray], *, n_components: int, seed: int, max_its: int) -> Any:
    import mixle.stats as st
    from mixle.inference import optimize

    est = st.MixtureEstimator([st.DiagonalGaussianEstimator(dim=2)] * n_components)
    return optimize(rows, est, max_its=max_its, rng=np.random.RandomState(seed), out=None)


def fit_twisted_mixture(
    group: CyclicGroup,
    data_by_group: dict[int, Sequence[float]],
    *,
    n_components: int = 2,
    seed: int = 0,
    max_its: int = 50,
) -> TwistedMixtureResult:
    """Fit ONE base density on every group's data pooled together after undoing each group's twist.

    ``data_by_group`` maps a group element ``k`` to that group's (small) sample of the periodic coordinate.
    Every sample is embedded and rotated back by its own group's ``inverse_act`` before pooling -- so the
    fitted ``n_components``-component density sees ``sum(len(v) for v in data_by_group.values())`` points,
    not just one group's slice, for the same parameter count as :func:`fit_independent_mixtures` spends on a
    SINGLE group.
    """
    rows: list[np.ndarray] = []
    for k, xs in data_by_group.items():
        aligned = group.inverse_act(group.embed(xs), k)
        rows.extend(list(aligned))
    return TwistedMixtureResult(
        base_density=_fit_density(rows, n_components=n_components, seed=seed, max_its=max_its),
        group=group,
    )


def fit_independent_mixtures(
    group: CyclicGroup,
    data_by_group: dict[int, Sequence[float]],
    *,
    n_components: int = 2,
    seed: int = 0,
    max_its: int = 50,
) -> dict[int, Any]:
    """The untwisted baseline: one independently-fit ``n_components``-component density per group.

    Same per-group parameter count as the shared base density in :func:`fit_twisted_mixture`, but ``|groups|``
    times the total parameters overall, and each fit sees only its own group's (small) sample -- the
    comparison :func:`fit_twisted_mixture` is measured against.
    """
    out: dict[int, Any] = {}
    for raw_k, xs in data_by_group.items():
        # keyed by the validated element, so the returned dict is indexable by the same declared
        # elements callers pass to :func:`independent_log_density` (and cannot carry an alias key).
        k = group.element(raw_k)
        rows = list(group.embed(xs))
        out[k] = _fit_density(rows, n_components=n_components, seed=seed + k, max_its=max_its)
    return out


def independent_log_density(models: dict[int, Any], group: CyclicGroup, x: Sequence[float], k: int) -> np.ndarray:
    """Score ``x`` under group ``k``'s independently-fit density (the baseline sibling of
    :meth:`TwistedMixtureResult.log_density`).

    Normalized over one period by that model's own :func:`log_circular_normalizer` -- each independent fit
    has a different embedding normalizer, so without this the baseline comparison would be decided by
    normalization rather than by fit quality.
    """
    model = models[group.element(k)]
    scores = _embedding_log_density(model, group.embed(x))
    return scores - log_circular_normalizer(model, group)
