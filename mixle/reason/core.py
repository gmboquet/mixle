"""Reasoning front door for fusing modality evidence into a belief.

``reason(prior, evidence)`` folds a sequence of linear-Gaussian observations into a belief state
by exact Kalman assimilation, tracking how many nats of uncertainty each modality removed. The
returned :class:`ReasonedAnswer` is the posterior belief plus the tools a scientific answer needs:
credible intervals, per-modality attribution, and an epistemic/aleatoric split of any prediction.
"""

from __future__ import annotations

import itertools
import math
import warnings
from dataclasses import dataclass
from typing import Any

import numpy as np

from mixle.inference.belief import BeliefState, GaussianBelief
from mixle.inference.uncertainty import UncertaintyDecomposition

_PSD_EIGENVALUE_RATIO = 1e-9
"""Relative-tolerance bound (matching ``mixle.inference.belief.GaussianBelief``'s own PSD gate and
``mixle.doe.batch._safe_cholesky``'s) on how negative a covariance's worst eigenvalue may be, relative
to its own eigenvalue scale, before it is refused outright as not a valid covariance at all."""


def _validated_covariance(cov: Any, d: int, name: str) -> np.ndarray:
    """Return ``cov`` as a validated ``(d, d)`` covariance: finite, symmetric, and PSD.

    Symmetrized defensively before the finiteness/PSD checks run (matching
    ``GaussianBelief.__init__`` and ``mixle.doe.batch._safe_cholesky``'s established convention) so
    the matrix that gets validated is the SAME one that gets used -- an asymmetric-but-otherwise-valid
    input is silently harmonized rather than having its off-diagonal disagreement silently ignored by
    whichever triangle a downstream ``eigvalsh``/diagonal read happens to prefer.

    Raises:
        ValueError: if ``cov`` is not exactly ``(d, d)``, is non-finite, or (after symmetrizing) is
            not positive semi-definite within :data:`_PSD_EIGENVALUE_RATIO` of its own eigenvalue scale.
    """
    P = np.atleast_2d(np.asarray(cov, dtype=float))
    if P.shape != (d, d):
        raise ValueError(f"{name} must have shape ({d}, {d}); got {P.shape}")
    if not np.isfinite(P).all():
        raise ValueError(f"{name} must be finite (no NaN or inf)")
    P = 0.5 * (P + P.T)
    evals = np.linalg.eigvalsh(P)
    scale = float(np.abs(evals).max()) if evals.size else 0.0
    if evals.min() < -_PSD_EIGENVALUE_RATIO * max(scale, 1e-300):
        raise ValueError(
            f"{name} must be positive semi-definite (worst eigenvalue {evals.min():.6g} vs "
            f"eigenvalue scale {scale:.6g})"
        )
    return P


def _aleatoric_from_noise(R: Any, k: int) -> np.ndarray:
    """Validate a prediction's noise ``R`` and return its ``k`` per-coordinate variances.

    Accepts a scalar (homoscedastic noise shared by all ``k`` outputs), a length-``k`` vector
    (per-output variances), or a ``(k, k)`` covariance matrix (validated and symmetrized via
    :func:`_validated_covariance`; only its diagonal is used, since :meth:`ReasonedAnswer.predict`'s
    law-of-total-variance split is per-coordinate). Every form must be finite, and every variance
    must be non-negative -- a negative entry cannot be a variance under any interpretation.

    Raises:
        ValueError: if ``R``'s shape doesn't match ``k``, it is non-finite, contains a negative
            variance, or (matrix form) fails :func:`_validated_covariance`.
    """
    Rm = np.asarray(R, dtype=float)
    if Rm.ndim == 0:
        if not np.isfinite(Rm):
            raise ValueError("R must be finite (no NaN or inf)")
        if Rm < 0.0:
            raise ValueError(f"R must be non-negative (a variance); got {float(Rm):.6g}")
        return np.full(k, float(Rm))
    if Rm.ndim == 1:
        if Rm.shape != (k,):
            raise ValueError(f"R must have length {k} (one variance per predicted output); got shape {Rm.shape}")
        if not np.isfinite(Rm).all():
            raise ValueError("R must be finite (no NaN or inf)")
        if np.any(Rm < 0.0):
            raise ValueError(f"R must be non-negative (a variance per output); got {Rm}")
        return Rm.copy()
    return np.diag(_validated_covariance(Rm, k, "R")).copy()


def _exact_int(value: Any, name: str) -> int:
    """Return ``value`` as an exact integer, rejecting fractional input (no silent ``int()`` truncation).

    The fractional check was made through ``float(value)``, which is not a type check at all:
    ``float(True)`` is ``1.0`` and ``float("3")`` is ``3.0``, so a Boolean and a numeric string both
    passed and became dimensions and step counts (MXR-080-1898). A dimension read from configuration
    text arrives as a string, and ``True`` as a dimension is the kind of value that produces a
    silently wrong-shaped computation rather than an error.
    """
    if isinstance(value, (bool, np.bool_)):
        raise TypeError(f"{name} must be an integer, not a Boolean; got {value!r}")
    if not isinstance(value, (int, float, np.integer, np.floating)):
        raise TypeError(f"{name} must be a real number; got {value!r} ({type(value).__name__})")
    f = float(value)
    if not np.isfinite(f) or f != int(f):
        raise ValueError(f"{name} must be an integer; got {value!r}")
    return int(f)


def _positive_int(value: Any, name: str) -> int:
    """Return ``value`` as an exact positive integer (``>= 1``), rejecting fractional or non-positive input."""
    n = _exact_int(value, name)
    if n < 1:
        raise ValueError(f"{name} must be a positive integer; got {value!r}")
    return n


class Latent:
    """Factories for the shared latent prior used at the start of assimilation."""

    @staticmethod
    def gaussian(mean: Any, cov: Any) -> GaussianBelief:
        """A Gaussian prior ``N(mean, cov)`` over the latent."""
        return GaussianBelief(mean, cov)

    @staticmethod
    def vector(dim: int, *, mean: float = 0.0, var: float = 1.0) -> GaussianBelief:
        """An isotropic Gaussian prior over a ``dim``-vector latent: ``N(mean*1, var*I)``.

        Raises:
            ValueError: if ``dim`` is not an exact positive integer (MXR-080-0274) -- a fractional
                ``dim`` used to be silently truncated with ``int()`` rather than rejected.
        """
        d = _positive_int(dim, "dim")
        return GaussianBelief(np.full(d, float(mean)), np.eye(d) * float(var))

    @staticmethod
    def mechanistic(
        A: Any,
        steps: int,
        *,
        x0_mean: Any = None,
        x0_cov: Any = None,
        process_cov: Any = None,
    ) -> GaussianBelief:
        """Return a linear-dynamics prior over ``z_0 .. z_{steps-1}``.

        The trajectory follows ``z_{t+1} = A z_t + w_t`` with
        ``w_t ~ N(0, Q)``. The returned belief is the joint Gaussian over the
        stacked trajectory ``(steps * d,)``. Because the states are coupled,
        evidence at one time can inform other times through the dynamics, so
        fusing observations via :func:`reason` performs exact Kalman smoothing
        for this linear-Gaussian model.

        Args:
            A: ``(d, d)`` linear state-transition operator (one discrete step).
            steps: number of time steps ``T`` in the trajectory.
            x0_mean: mean of ``z_0`` (default zeros).
            x0_cov: covariance of ``z_0`` (default identity).
            process_cov: process-noise covariance ``Q`` (default zeros -- deterministic dynamics).

        Raises:
            ValueError: if ``steps`` is not an exact positive integer; if ``x0_mean`` doesn't have
                shape ``(d,)`` or isn't finite; or if ``x0_cov`` / ``process_cov`` isn't a finite,
                symmetric, positive-semidefinite ``(d, d)`` covariance (MXR-080-0274) -- previously
                ``steps`` was silently truncated with ``int()`` and the covariances were never
                validated at all, so a malformed one either propagated silently into the joint prior
                or surfaced only as a confusing low-level error far from its actual cause.
        """
        A = np.atleast_2d(np.asarray(A, dtype=float))
        d = A.shape[0]
        if A.shape != (d, d):
            raise ValueError(f"A must be square (d, d); got {A.shape}")
        T = _positive_int(steps, "steps")
        m0 = np.zeros(d) if x0_mean is None else np.atleast_1d(np.asarray(x0_mean, dtype=float))
        if m0.shape != (d,):
            raise ValueError(f"x0_mean must have shape ({d},) to match A; got {m0.shape}")
        if not np.isfinite(m0).all():
            raise ValueError("x0_mean must be finite (no NaN or inf)")
        P0 = np.eye(d) if x0_cov is None else _validated_covariance(x0_cov, d, "x0_cov")
        Q = np.zeros((d, d)) if process_cov is None else _validated_covariance(process_cov, d, "process_cov")

        # forward marginals: mean_{t+1} = A mean_t, P_{t+1} = A P_t Aᵀ + Q
        means = [m0]
        margs = [P0]
        for _ in range(1, T):
            means.append(A @ means[-1])
            margs.append(A @ margs[-1] @ A.T + Q)

        # joint covariance: Cov(z_t, z_s) = A^{t-s} P_s for t >= s (noise after s is independent of z_s)
        big = np.zeros((T * d, T * d))
        for s in range(T):
            for t in range(s, T):
                block = np.linalg.matrix_power(A, t - s) @ margs[s]
                big[t * d : (t + 1) * d, s * d : (s + 1) * d] = block
                big[s * d : (s + 1) * d, t * d : (t + 1) * d] = block.T
        return GaussianBelief(np.concatenate(means), big)


@dataclass(frozen=True)
class LinearGaussianEvidence:
    """One modality's evidence about the latent ``z``: ``y = H z + noise``, ``noise ~ N(0, R)``.

    ``H`` is the (possibly linearized) forward operator mapping the latent to this modality's
    measurement space, ``y`` the observed data, ``R`` its noise covariance (matrix, diagonal, or
    scalar). Application forward models (e.g. ``mixle_pde`` geophysics operators) produce these.
    """

    H: Any
    y: Any
    R: Any
    name: str = ""


#: Short alias -- ``Evidence(H, y, R, name)``.
Evidence = LinearGaussianEvidence


@dataclass
class NonlinearEvidence:
    """One modality's evidence through a nonlinear forward model.

    Assimilated by (iterated) extended-Kalman linearization: at the current belief mean ``m`` the
    forward is replaced by its tangent ``h(z) ~ h(m) + J(m)(z - m)`` and the exact linear update runs
    on that tangent; with ``iterations > 1`` the linearization point is refined at the updated mean and
    the update repeats from the pre-update belief, which matters when the prior mean
    is far from the truth. ``jacobian`` is analytic when you have it; otherwise a central finite
    difference is used. This is a Gaussian approximation around the
    linearization point; for strongly multimodal posteriors it reports one
    mode's belief, not the full mixture.
    """

    h: Any  # callable z -> predicted measurement (m,)
    y: Any
    R: Any
    jacobian: Any = None  # callable z -> (m, d) Jacobian; finite-difference when None
    iterations: int = 2
    name: str = ""


def _fd_jacobian(h: Any, z: np.ndarray, eps: float = 1.0e-6) -> np.ndarray:
    z = np.asarray(z, dtype=np.float64).reshape(-1)
    base = np.asarray(h(z), dtype=np.float64).reshape(-1)
    J = np.empty((base.shape[0], z.shape[0]), dtype=np.float64)
    for j in range(z.shape[0]):
        dz = np.zeros_like(z)
        dz[j] = eps * max(1.0, abs(float(z[j])))
        J[:, j] = (
            np.asarray(h(z + dz), dtype=np.float64).reshape(-1) - np.asarray(h(z - dz), dtype=np.float64).reshape(-1)
        ) / (2.0 * dz[j])
    return J


def _assimilate_nonlinear(belief: Any, e: NonlinearEvidence) -> Any:
    """Iterated-EKF fold of one nonlinear observation into a Gaussian belief."""
    y = np.asarray(e.y, dtype=np.float64).reshape(-1)
    point = np.asarray(belief.mean(), dtype=np.float64).reshape(-1)
    updated = belief
    for _ in range(max(1, int(e.iterations))):
        J = np.asarray(e.jacobian(point) if e.jacobian is not None else _fd_jacobian(e.h, point), dtype=np.float64)
        # tangent measurement: y - h(point) + J point plays the role of the linear y for H = J
        y_lin = y - np.asarray(e.h(point), dtype=np.float64).reshape(-1) + J @ point
        updated = belief.update(J, y_lin, e.R)  # each pass restarts from the PRE-update belief (IEKF)
        point = np.asarray(updated.mean(), dtype=np.float64).reshape(-1)
    return updated


def block_selector(step: int, n_blocks: int, block_dim: int, within: Any = None) -> np.ndarray:
    """An observation matrix that reads time-block ``step`` of a stacked trajectory latent.

    For a latent built by :meth:`Latent.mechanistic` (shape ``(n_blocks * block_dim,)``), returns the
    ``H`` selecting block ``step`` -- use it to build :class:`LinearGaussianEvidence` for an
    observation at that time. ``within`` optionally reads only part of the block (a
    ``(k, block_dim)`` local readout); by default the whole block is read (identity).

    Raises:
        ValueError: if ``n_blocks`` / ``block_dim`` is not an exact positive integer; if ``step`` is
            not an exact integer or falls outside ``[0, n_blocks)``; or if ``within`` doesn't have
            ``block_dim`` columns (MXR-080-0274). ``step`` used to accept Python-style negative
            indexing by accident (silently selecting a DIFFERENT, in-range block with no signal that
            anything unusual happened) while a step at or beyond ``n_blocks`` fell through to a
            confusing low-level "could not broadcast" error instead of a clear one.
    """
    n_blocks = _positive_int(n_blocks, "n_blocks")
    block_dim = _positive_int(block_dim, "block_dim")
    step = _exact_int(step, "step")
    if not (0 <= step < n_blocks):
        raise ValueError(f"step must be in [0, {n_blocks}); got {step}")
    local = np.eye(block_dim) if within is None else np.atleast_2d(np.asarray(within, dtype=float))
    if local.shape[1] != block_dim:
        raise ValueError(f"within must have {block_dim} columns to match block_dim; got shape {local.shape}")
    H = np.zeros((local.shape[0], n_blocks * block_dim))
    H[:, step * block_dim : (step + 1) * block_dim] = local
    return H


_SHAPLEY_EXACT_MAX_SOURCES = 7
"""Above this many evidence items, :meth:`ReasonedAnswer.attribution`'s ``method="shapley"`` switches
from exact permutation enumeration (``n!`` fold sequences -- 5040 for 7 items, each a handful of cheap
Kalman updates) to Monte Carlo sampling, since ``n!`` grows too fast to enumerate exactly beyond this.
Realistic evidence counts in this codebase (per-modality or per-model fusion) are a handful of sources,
comfortably under this threshold, so exact computation is the common case in practice."""

_SHAPLEY_DEFAULT_SAMPLES = 200
"""Default number of sampled permutations for ``method="shapley"`` beyond
:data:`_SHAPLEY_EXACT_MAX_SOURCES` sources -- a standard permutation-sampling Monte Carlo estimator of
Shapley value, whose standard error shrinks as ``O(1 / sqrt(n_permutations))``. Override via
``attribution(method="shapley", n_permutations=...)`` for a tighter (more samples) or cheaper (fewer)
estimate."""


class ReasonedAnswer:
    """A posterior belief about a query, with the UQ a scientific answer needs.

    Beyond ``mean`` / ``interval`` / ``entropy`` (delegated to the belief), it exposes
    :meth:`attribution` -- the nats of uncertainty each modality removed -- and :meth:`predict`,
    which splits a *prediction's* uncertainty into epistemic (from latent uncertainty) and
    aleatoric (observation noise) via the law of total variance.
    """

    def __init__(
        self,
        belief: BeliefState,
        prior: BeliefState,
        trace: list[tuple[str, BeliefState]],
        *,
        full_prior: BeliefState | None = None,
        evidence: list[Any] | None = None,
        query_idx: Any = None,
    ) -> None:
        self.belief = belief
        # The prior belief IN THIS ANSWER'S OWN COORDINATE SPACE: the full prior for an unqueried
        # answer, or that same prior's matching marginal after :meth:`marginal` (MXR-080-0272). Every
        # entropy this class reports is computed from `self._prior` and `self.belief`/`self._trace`
        # together -- never a full-state entropy mixed against a lower-dimensional marginal's.
        self._prior = prior
        # (name, belief-after-this-fold) for every evidence item, in fold order, ALSO projected into
        # this answer's own coordinate space -- lets attribution() recompute each source's nats
        # entirely within the queried subspace instead of reusing the full-state gain.
        self._trace = list(trace)
        # The UNTOUCHED original prior and evidence list (always in the FULL latent space -- never
        # re-marginalized) plus the coordinate indices (into that full space) this answer currently
        # represents. Retained across repeated :meth:`marginal` calls purely so order-invariant
        # (Shapley, MXR-080-0275) attribution can still be recomputed exactly in the query's own
        # subspace by re-folding subsets of the ORIGINAL evidence in a different order -- something
        # `self._trace` alone (one fixed fold order) cannot answer.
        self._full_prior = prior if full_prior is None else full_prior
        self._evidence = [] if evidence is None else list(evidence)
        full_dim = int(np.size(self._full_prior.mean()))
        self._query_idx = np.arange(full_dim) if query_idx is None else np.atleast_1d(np.asarray(query_idx, dtype=int))
        contributions: dict[str, float] = {}
        before = self._prior
        for name, after in self._trace:
            gain = before.entropy() - after.entropy()
            contributions[name] = contributions.get(name, 0.0) + gain
            before = after
        self._contributions = contributions

    @property
    def mean(self) -> np.ndarray:
        """Return posterior mean from the underlying belief."""
        return self.belief.mean()

    def cov(self) -> np.ndarray:
        """Return posterior covariance from the underlying belief."""
        return self.belief.cov()

    def sd(self) -> np.ndarray:
        """Return posterior standard deviations from the underlying belief."""
        return self.belief.sd()

    def entropy(self) -> float:
        """Return posterior entropy from the underlying belief."""
        return self.belief.entropy()

    def interval(self, level: float = 0.9) -> np.ndarray:
        """Per-coordinate central credible interval at ``level`` (an ``(d, 2)`` array of ``[lo, hi]``)."""
        return self.belief.interval(level)

    def information_gain(self) -> float:
        """Total nats of uncertainty the evidence removed from the prior (``H[prior] - H[posterior]``).

        Both entropies are always taken from this answer's OWN coordinate space -- the full prior for
        an unqueried answer, or that prior's matching marginal after :meth:`marginal` -- so querying a
        coordinate the evidence never touched correctly reports zero gain (MXR-080-0272), rather than
        a full-state prior entropy mixed against a lower-dimensional posterior marginal's.
        """
        return self._prior.entropy() - self.belief.entropy()

    def attribution(self, *, normalize: bool = False, method: str = "sequential", **kwargs: Any) -> dict[str, float]:
        """Per-modality information gain in nats -- which modality sharpened the belief, and by how much.

        Two allocation rules are available, and they generally disagree for REDUNDANT sources (two
        items that both constrain overlapping coordinates) -- pick deliberately, not by default
        (MXR-080-0275):

        * ``method="sequential"`` (default): each item's nats are ``H[belief before it] - H[belief
          after it]``, in the order ``evidence`` was given to :func:`reason`. Cheap (one pass over the
          fold trace) and always sums exactly to :meth:`information_gain`, but for redundant sources
          it is a CONDITIONAL, fold-order-dependent credit split, not an intrinsic measure of modality
          importance: whichever redundant source is folded first is credited with uncertainty a
          different order would have credited to another source. The FINAL POSTERIOR does not depend
          on order for linear-Gaussian evidence (sequential exact Kalman assimilation commutes -- see
          :func:`reason`); this per-source SPLIT of it still does. Treat the result as "credit under
          the order evidence happened to be listed in," not as "how important is this modality."
        * ``method="shapley"``: the order-INVARIANT Shapley value -- each source's nats averaged over
          (up to) every permutation of fold order, satisfying the same efficiency property (values
          sum to :meth:`information_gain`) without depending on which order the caller listed evidence
          in. Exact for up to :data:`_SHAPLEY_EXACT_MAX_SOURCES` sources (``O(n!)`` re-folds of the
          full evidence sequence); beyond that it averages ``n_permutations`` uniformly random
          permutations instead (``O(n_permutations * n)`` re-folds), a standard permutation-sampling
          Monte Carlo estimator whose standard error shrinks as ``O(1 / sqrt(n_permutations))``. Pass
          ``n_permutations=`` / ``rng=`` to control the sampling estimate. Each re-fold repeats the
          full (possibly nonlinear, possibly expensive) assimilation from the original prior, so this
          costs ``O(n)`` to ``O(n!)`` times a single :func:`reason` call -- fine for the small modality
          counts this module is used with, not intended for hundreds of sources.

        With ``normalize=True``, values are the fraction of the total gain (they then sum to ~1),
        computed after the chosen allocation rule.
        """
        if method == "sequential":
            contrib = dict(self._contributions)
        elif method == "shapley":
            contrib = self._shapley_attribution(**kwargs)
        else:
            raise ValueError(f"attribution method must be 'sequential' or 'shapley'; got {method!r}")
        if normalize:
            total = sum(contrib.values())
            if total > 0:
                contrib = {k: v / total for k, v in contrib.items()}
        return contrib

    def _shapley_attribution(self, *, n_permutations: int | None = None, rng: Any = None) -> dict[str, float]:
        """Order-invariant per-source attribution: nats averaged over permutations of fold order.

        Re-folds the ORIGINAL evidence (never the already-fixed ``self._trace``) from
        ``self._full_prior``, projecting each intermediate belief to ``self._query_idx`` -- the SAME
        coordinates :meth:`information_gain` uses -- before taking its entropy, so this is correct
        for a marginal query too (MXR-080-0272 applies here as well: never a full-state entropy mixed
        against a marginal one). See :meth:`attribution` for the exactness/cost tradeoff.
        """
        n = len(self._evidence)
        names = [e.name or f"evidence[{i}]" for i, e in enumerate(self._evidence)]
        contrib = {name: 0.0 for name in names}
        if n == 0:
            return contrib
        prior_entropy = self._full_prior.marginal(self._query_idx).entropy()
        if n <= _SHAPLEY_EXACT_MAX_SOURCES:
            perms = itertools.permutations(range(n))
            count = math.factorial(n)
        else:
            rgen = rng if isinstance(rng, np.random.Generator) else np.random.default_rng(rng)
            count = int(n_permutations) if n_permutations else _SHAPLEY_DEFAULT_SAMPLES
            perms = (rgen.permutation(n) for _ in range(count))
        for perm in perms:
            belief = self._full_prior
            before_entropy = prior_entropy
            for i in perm:
                e = self._evidence[i]
                if isinstance(e, NonlinearEvidence):
                    belief = _assimilate_nonlinear(belief, e)
                else:
                    belief = belief.update(e.H, e.y, e.R)
                after_entropy = belief.marginal(self._query_idx).entropy()
                contrib[names[i]] += before_entropy - after_entropy
                before_entropy = after_entropy
        return {name: value / count for name, value in contrib.items()}

    def predict(self, H: Any, R: Any = 0.0) -> UncertaintyDecomposition:
        """Split the uncertainty of a new prediction ``y* = H z + noise(R)`` (law of total variance).

        ``epistemic = diag(H P Hᵀ)`` (from the latent's remaining uncertainty, reducible by more
        data) and ``aleatoric = diag(R)`` (irreducible observation noise). Exact for the Gaussian
        belief.

        Args:
            H: ``(k, d)`` observation operator, where ``d`` is this belief's own latent dimension.
                Its shape must match exactly -- it is never reshaped from a looser guess, since a
                wrong-width ``H`` can have a total element count that happens to divide evenly by
                ``d`` and so silently describe a DIFFERENT (still valid-looking) operator instead.
            R: the prediction's noise covariance: a scalar (shared by all ``k`` outputs), a
                length-``k`` vector of per-output variances, or a ``(k, k)`` covariance matrix
                (symmetrized, and required finite and positive semi-definite).

        Raises:
            ValueError: if ``H``'s width doesn't match this belief's latent dimension, or ``R`` is
                the wrong shape, non-finite, or not a valid (non-negative / positive semi-definite)
                noise covariance (MXR-080-0273).
        """
        P = np.atleast_2d(self.belief.cov())
        Hm = np.atleast_2d(np.asarray(H, dtype=float))
        if Hm.shape[1] != P.shape[0]:
            raise ValueError(
                f"H must have {P.shape[0]} columns to match this belief's latent dimension; got shape "
                f"{Hm.shape}. predict() no longer reshapes a wrong-width H from its element count alone, "
                "since that can silently match a different, unintended operator by coincidence."
            )
        epistemic = np.diag(Hm @ P @ Hm.T).copy()
        aleatoric = _aleatoric_from_noise(R, epistemic.shape[0])
        total = epistemic + aleatoric
        return UncertaintyDecomposition(total=total, aleatoric=aleatoric, epistemic=epistemic, kind="variance")

    def marginal(self, indices: Any) -> ReasonedAnswer:
        """Restrict the answer to a subset of latent coordinates (query a specific variable).

        The prior and every fold step's before/after belief are re-marginalized to the SAME
        coordinates as the posterior (MXR-080-0272), so :meth:`information_gain` and the default
        :meth:`attribution` on the result are computed entirely within the queried subspace -- a
        coordinate the evidence never touched now correctly reports zero gain instead of leaking in
        the full-state prior entropy. ``indices`` are local to THIS answer's own coordinates (so
        repeated/nested :meth:`marginal` calls compose correctly); the original full-space prior and
        evidence are carried through unchanged so ``attribution(method="shapley")`` keeps working
        after marginalizing.
        """
        idx = np.atleast_1d(np.asarray(indices, dtype=int))
        sub_belief = self.belief.marginal(idx)
        sub_prior = self._prior.marginal(idx)
        sub_trace = [(name, after.marginal(idx)) for name, after in self._trace]
        return ReasonedAnswer(
            sub_belief,
            sub_prior,
            sub_trace,
            full_prior=self._full_prior,
            evidence=self._evidence,
            query_idx=self._query_idx[idx],
        )

    def __repr__(self) -> str:
        return (
            f"ReasonedAnswer(dim={np.size(self.mean)}, "
            f"info_gain={self.information_gain():.3f} nats, "
            f"modalities={list(self._contributions)})"
        )


def _to_belief(prior: Any) -> GaussianBelief:
    if isinstance(prior, GaussianBelief):
        return prior
    if isinstance(prior, BeliefState):
        return GaussianBelief(prior.mean(), prior.cov())
    raise TypeError(f"prior must be a GaussianBelief (or BeliefState with cov), got {type(prior).__name__}")


def reason(prior: Any, evidence: Any, *, query: Any = None) -> ReasonedAnswer:
    """Fuse ``evidence`` into ``prior`` by exact Kalman assimilation; return the queried posterior.

    Args:
        prior: the latent's prior belief (:class:`GaussianBelief`; build one with :class:`Latent`).
        evidence: a sequence of :class:`LinearGaussianEvidence` and/or :class:`NonlinearEvidence`
            -- one per modality / observation. Nonlinear items assimilate by iterated-EKF
            linearization (a Gaussian approximation; see :class:`NonlinearEvidence`). They are
            folded in one at a time, and the nats each removes are recorded for
            :meth:`ReasonedAnswer.attribution`.

            **Order independence holds only when every item is** :class:`LinearGaussianEvidence`:
            sequential exact Kalman assimilation of linear-Gaussian observations commutes, so folding
            them in one at a time then equals conditioning on all of them at once, in any order. That
            guarantee does NOT extend to :class:`NonlinearEvidence`: each nonlinear item linearizes
            at the CURRENT belief mean, which itself depends on whatever was already folded in before
            it. Mixing nonlinear evidence with anything else -- other nonlinear evidence included --
            therefore makes both the posterior and the per-source :meth:`ReasonedAnswer.attribution`
            depend on the order ``evidence`` is given in (a :class:`RuntimeWarning` is raised when
            this applies); pass evidence in a fixed, deliberate order rather than relying on
            permutation invariance.

            **The POSTERIOR's order independence does not extend to** ``attribution()``'s default
            credit split, even for pure :class:`LinearGaussianEvidence` (MXR-080-0275): two sources
            that both constrain overlapping coordinates ("redundant" evidence) split their combined
            information gain differently depending on fold order, because the default allocation is
            sequential (whichever redundant source is folded first is credited with uncertainty a
            different order would credit to the other). Use ``attribution(method="shapley")`` for an
            order-invariant split when that matters more than the cost of recomputing it.
        query: optional latent coordinate indices to restrict the answer to.

    Returns:
        A :class:`ReasonedAnswer` -- the posterior belief plus attribution and prediction UQ.
    """
    evidence = list(evidence)
    if len(evidence) > 1 and any(isinstance(e, NonlinearEvidence) for e in evidence):
        warnings.warn(
            "reason(): evidence mixes NonlinearEvidence with other evidence. Unlike pure "
            "LinearGaussianEvidence, the result is ORDER-DEPENDENT: each NonlinearEvidence item "
            "linearizes at the belief mean at the time it is folded in, which depends on whatever "
            "was folded in before it, so the posterior and per-source attribution() can differ "
            "substantially between orderings of the same evidence list. Pass evidence in a fixed, "
            "deliberate order.",
            RuntimeWarning,
            stacklevel=2,
        )
    belief = _to_belief(prior)
    prior_belief = belief
    trace: list[tuple[str, GaussianBelief]] = []
    for i, e in enumerate(evidence):
        name = e.name or f"evidence[{i}]"
        if isinstance(e, NonlinearEvidence):
            belief = _assimilate_nonlinear(belief, e)
        else:
            belief = belief.update(e.H, e.y, e.R)
        trace.append((name, belief))
    answer = ReasonedAnswer(belief, prior_belief, trace, full_prior=prior_belief, evidence=evidence)
    if query is not None:
        answer = answer.marginal(query)
    return answer
