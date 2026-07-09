"""Reasoning front door for fusing modality evidence into a belief.

``reason(prior, evidence)`` folds a sequence of linear-Gaussian observations into a belief state
by exact Kalman assimilation, tracking how many nats of uncertainty each modality removed. The
returned :class:`ReasonedAnswer` is the posterior belief plus the tools a scientific answer needs:
credible intervals, per-modality attribution, and an epistemic/aleatoric split of any prediction.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

from mixle.inference.belief import BeliefState, GaussianBelief
from mixle.inference.uncertainty import UncertaintyDecomposition


class Latent:
    """Factories for the shared latent prior used at the start of assimilation."""

    @staticmethod
    def gaussian(mean: Any, cov: Any) -> GaussianBelief:
        """A Gaussian prior ``N(mean, cov)`` over the latent."""
        return GaussianBelief(mean, cov)

    @staticmethod
    def vector(dim: int, *, mean: float = 0.0, var: float = 1.0) -> GaussianBelief:
        """An isotropic Gaussian prior over a ``dim``-vector latent: ``N(mean*1, var*I)``."""
        return GaussianBelief(np.full(int(dim), float(mean)), np.eye(int(dim)) * float(var))

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
        """
        A = np.atleast_2d(np.asarray(A, dtype=float))
        d = A.shape[0]
        if A.shape != (d, d):
            raise ValueError(f"A must be square (d, d); got {A.shape}")
        T = int(steps)
        if T < 1:
            raise ValueError("steps must be >= 1")
        m0 = np.zeros(d) if x0_mean is None else np.atleast_1d(np.asarray(x0_mean, dtype=float))
        P0 = np.eye(d) if x0_cov is None else np.atleast_2d(np.asarray(x0_cov, dtype=float))
        Q = np.zeros((d, d)) if process_cov is None else np.atleast_2d(np.asarray(process_cov, dtype=float))

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
    """
    local = np.eye(block_dim) if within is None else np.atleast_2d(np.asarray(within, dtype=float))
    H = np.zeros((local.shape[0], n_blocks * block_dim))
    H[:, step * block_dim : (step + 1) * block_dim] = local
    return H


class ReasonedAnswer:
    """A posterior belief about a query, with the UQ a scientific answer needs.

    Beyond ``mean`` / ``interval`` / ``entropy`` (delegated to the belief), it exposes
    :meth:`attribution` -- the nats of uncertainty each modality removed -- and :meth:`predict`,
    which splits a *prediction's* uncertainty into epistemic (from latent uncertainty) and
    aleatoric (observation noise) via the law of total variance.
    """

    def __init__(self, belief: BeliefState, prior_entropy: float, contributions: dict[str, float]) -> None:
        self.belief = belief
        self._prior_entropy = float(prior_entropy)
        self._contributions = dict(contributions)

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
        """Total nats of uncertainty the evidence removed from the prior (``H[prior] - H[posterior]``)."""
        return self._prior_entropy - self.belief.entropy()

    def attribution(self, *, normalize: bool = False) -> dict[str, float]:
        """Per-modality information gain in nats -- which modality sharpened the belief, and by how much.

        With ``normalize=True``, values are the fraction of the total gain (they then sum to ~1).
        """
        contrib = dict(self._contributions)
        if normalize:
            total = sum(contrib.values())
            if total > 0:
                contrib = {k: v / total for k, v in contrib.items()}
        return contrib

    def predict(self, H: Any, R: Any = 0.0) -> UncertaintyDecomposition:
        """Split the uncertainty of a new prediction ``y* = H z + noise(R)`` (law of total variance).

        ``epistemic = diag(H P Hᵀ)`` (from the latent's remaining uncertainty, reducible by more
        data) and ``aleatoric = diag(R)`` (irreducible observation noise). Exact for the Gaussian
        belief.
        """
        P = np.atleast_2d(self.belief.cov())
        Hm = np.atleast_2d(np.asarray(H, dtype=float))
        if Hm.shape[1] != P.shape[0]:
            Hm = Hm.reshape(-1, P.shape[0])
        epistemic = np.diag(Hm @ P @ Hm.T).copy()
        Rm = np.asarray(R, dtype=float)
        if Rm.ndim == 0:
            aleatoric = np.full(epistemic.shape, float(Rm))
        elif Rm.ndim == 1:
            aleatoric = Rm.copy()
        else:
            aleatoric = np.diag(Rm).copy()
        total = epistemic + aleatoric
        return UncertaintyDecomposition(total, aleatoric, epistemic, "variance")

    def marginal(self, indices: Any) -> ReasonedAnswer:
        """Restrict the answer to a subset of latent coordinates (query a specific variable)."""
        sub = self.belief.marginal(indices)
        return ReasonedAnswer(sub, self._prior_entropy, self._contributions)

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
            linearization (a Gaussian approximation; see :class:`NonlinearEvidence`).
            They are folded in one at a time (order does not affect the result), and the nats each
            removes are recorded for :meth:`ReasonedAnswer.attribution`.
        query: optional latent coordinate indices to restrict the answer to.

    Returns:
        A :class:`ReasonedAnswer` -- the posterior belief plus attribution and prediction UQ.
    """
    belief = _to_belief(prior)
    prior_entropy = belief.entropy()
    contributions: dict[str, float] = {}
    for i, e in enumerate(evidence):
        name = e.name or f"evidence[{i}]"
        before = belief.entropy()
        if isinstance(e, NonlinearEvidence):
            belief = _assimilate_nonlinear(belief, e)
        else:
            belief = belief.update(e.H, e.y, e.R)
        gain = before - belief.entropy()
        contributions[name] = contributions.get(name, 0.0) + gain
    answer = ReasonedAnswer(belief, prior_entropy, contributions)
    if query is not None:
        answer = answer.marginal(query)
    return answer
