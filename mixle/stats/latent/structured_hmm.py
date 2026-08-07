"""Structured HMMs: a composable transition layer (dense / low-rank / combinators) + forward-backward.

A standard HMM stores a dense K x K transition matrix and the forward-backward does O(K^2) work per step.
Rich structure -- a low-rank transition, a block of independent chains, a factorial (Kronecker) product --
is hard to express that way. This module factors the transition behind a small :class:`TransitionOperator`
interface so the forward-backward only needs two primitives:

    forward(alpha)  = alpha @ A         (push a state-belief forward one step)
    backward(v)     = A @ v             (pull an emission-weighted belief back one step)

and an M-step that re-estimates the operator from expected transition mass. Any operator that implements
those plugs into the SAME forward-backward / EM. Implementations:

    * :class:`DenseTransition`     -- the usual K x K matrix (O(K^2)).
    * :class:`LowRankTransition`   -- A = G @ Phi with an inner rank r (K x r, r x K row-stochastic): each
      state mixes over r shared "transition profiles". Forward/backward and the M-step are O(K r), and the
      parameter count drops from K^2 to 2 K r. (Combinators -- block-diagonal, Kronecker/factorial -- are
      the same interface; see TransitionOperator subclasses.)

The forgetting / mixing property of an ergodic chain (beliefs forget the distant past) is what lets the
forward-backward be split into chunks and run in parallel; see ``parallel`` in this package's estimation.
"""

from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass
from typing import Any

import numpy as np

from mixle.stats.compute.mixture_evidence import (
    validated_probability_vector,
    validated_row_probability_matrix,
)
from mixle.stats.latent.effective_sample import (
    heal_pooled_statistics,
    require_finite_count_totals,
    restore_accumulator_statistics,
    snapshot_accumulator_statistics,
    validate_effective_sample_mass,
    validated_count_array,
    validated_observation_weight,
    validated_statistic_tuple,
)
from mixle.stats.latent.markov_stopping import (
    require_terminal_reached,
    validate_terminal_reachability,
    validated_state_ids,
    validated_terminal_states,
)
from mixle.utils.vector import ImpossibleEvidenceError, require_possible_log_evidence


class TransitionOperator:
    """A row-stochastic state-transition operator behind the HMM forward-backward.

    Subclasses provide the two linear maps the recursions need plus an M-step from expected transition
    mass. ``forward``/``backward`` must be consistent with ``as_matrix`` (``forward(a) == a @ A``,
    ``backward(v) == A @ v``); the low-overhead operators never materialize ``A``.
    """

    n_states: int

    # Operators are required components of the serializable StructuredHMM / InputOutputHMM but are not
    # distributions or estimators themselves, so they opt in to mixle JSON serialization explicitly
    # (state round-trips via __dict__; SparseTransition's csr matrix uses the sparse codec).
    __pysp_serializable__ = True

    def to_dict(self) -> dict[str, Any]:
        """Return a safe JSON-compatible representation of this operator."""
        from mixle.utils.serialization import to_serializable

        return to_serializable(self)

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> TransitionOperator:
        """Reconstruct an operator from ``to_dict`` output."""
        from mixle.utils.serialization import from_serializable

        rv = from_serializable(payload)
        if not isinstance(rv, cls):
            raise TypeError("decoded object is %s, not %s" % (type(rv).__name__, cls.__name__))
        return rv

    def forward(self, alpha: np.ndarray) -> np.ndarray:  # alpha @ A
        """Push a state-belief row vector one transition forward."""
        raise NotImplementedError

    def backward(self, v: np.ndarray) -> np.ndarray:  # A @ v
        """Pull an emission-weighted belief vector one transition backward."""
        raise NotImplementedError

    def as_matrix(self) -> np.ndarray:
        """Materialize the transition matrix when an explicit matrix is needed."""
        raise NotImplementedError

    # --- M-step: accumulate expected transition mass over a sequence, then re-estimate ---
    def new_accumulator(self) -> Any:
        """Create an empty transition sufficient-statistic accumulator."""
        raise NotImplementedError

    def accumulate(self, acc: Any, alpha_t: np.ndarray, w_next: np.ndarray, scale: float) -> None:
        """Add one transition's expected mass. ``alpha_t`` is the (normalized) forward belief at t,
        ``w_next = b_{t+1} * beta_{t+1}`` the emission-weighted backward belief at t+1, ``scale`` the
        forward normalizer ``c_{t+1}`` (so the per-step posterior transition mass is exact)."""
        raise NotImplementedError

    def estimate(self, acc: Any) -> TransitionOperator:
        """Estimate a transition operator of the same structural family from accumulated statistics."""
        raise NotImplementedError

    def random_accumulator(self, rng) -> Any:
        """A randomly-filled accumulator whose ``estimate`` yields a random (structured) transition --
        used to seed EM when there is no warm start. Fills ``new_accumulator`` shapes (nested) with noise."""

        def fill(a):
            return a + rng.random(a.shape) if isinstance(a, np.ndarray) else [fill(x) for x in a]

        return fill(self.new_accumulator())


def _exact_positive_integer(value: Any, label: str) -> int:
    """Return an exact positive integer."""
    if isinstance(value, bool) or not isinstance(value, (int, np.integer)):
        raise TypeError(f"{label} must be an integer.")
    result = int(value)
    if result <= 0:
        raise ValueError(f"{label} must be positive.")
    return result


def _exact_nonnegative_integer(value: Any, label: str) -> int:
    """Return an exact non-negative integer."""
    if isinstance(value, bool) or not isinstance(value, (int, np.integer)):
        raise TypeError(f"{label} must be an integer.")
    result = int(value)
    if result < 0:
        raise ValueError(f"{label} must be non-negative.")
    return result


def _finite_nonnegative_real(value: Any, label: str) -> float:
    """Return a finite non-negative real without accepting strings or booleans."""
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, float, np.integer, np.floating)):
        raise TypeError(f"{label} must be a real number.")
    result = float(value)
    if not np.isfinite(result) or result < 0.0:
        raise ValueError(f"{label} must be finite and non-negative.")
    return result


def _validated_fit_controls(max_its: Any, tol: Any) -> tuple[int, float]:
    """Validate common EM iteration and convergence controls."""
    return (
        _exact_positive_integer(max_its, "max_its"),
        _finite_nonnegative_real(tol, "tol"),
    )


def _validated_sequences(values: Any, label: str) -> list[list[Any]]:
    """Materialize a non-empty batch of non-empty observation sequences."""
    try:
        sequences = [list(sequence) for sequence in values]
    except TypeError as exc:
        raise TypeError(f"{label} must be an iterable of observation sequences.") from exc
    if not sequences:
        raise ValueError(f"{label} must contain at least one sequence.")
    empty = [index for index, sequence in enumerate(sequences) if not sequence]
    if empty:
        raise ValueError(f"{label} contains empty sequence rows {empty}.")
    return sequences


def _validated_weights(
    values: Any,
    size: int,
    label: str = "weights",
    *,
    require_positive_total: bool = True,
) -> np.ndarray:
    """Return an owned finite non-negative weight vector with positive total mass."""
    if values is None:
        weights = np.ones(size, dtype=np.float64)
    else:
        try:
            weights = np.asarray(values, dtype=np.float64)
        except (TypeError, ValueError, OverflowError) as exc:
            raise TypeError(f"{label} must be a numeric vector.") from exc
        if weights.shape != (size,):
            raise ValueError(f"{label} must have shape ({size},).")
        if np.any(~np.isfinite(weights)) or np.any(weights < 0.0):
            raise ValueError(f"{label} must contain finite non-negative values.")
        weights = weights.copy()
    if require_positive_total and float(weights.sum()) <= 0.0:
        raise ValueError(f"{label} must contain positive total mass.")
    return weights


def _validated_input_symbols(values: Any, n_inputs: int, label: str) -> list[int]:
    """Return exact in-range discrete IOHMM input symbols."""
    try:
        raw_symbols = list(values)
    except TypeError as exc:
        raise TypeError(f"{label} must be an iterable of input symbols.") from exc
    symbols: list[int] = []
    for index, symbol in enumerate(raw_symbols):
        if isinstance(symbol, (bool, np.bool_)) or not isinstance(symbol, (int, np.integer)):
            raise TypeError(f"{label}[{index}] must be an integer.")
        normalized = int(symbol)
        if normalized < 0 or normalized >= n_inputs:
            raise ValueError(f"{label}[{index}]={normalized} is outside [0, {n_inputs}).")
        symbols.append(normalized)
    return symbols


def _validated_io_record(values: Any, n_inputs: int, label: str) -> tuple[list[Any], list[int]]:
    """Split and validate a non-empty IOHMM record of observation/input pairs."""
    try:
        record = list(values)
    except TypeError as exc:
        raise TypeError(f"{label} must be an iterable of observation/input pairs.") from exc
    if not record:
        raise ValueError(f"{label} must not be empty.")
    observations: list[Any] = []
    raw_inputs: list[Any] = []
    for index, pair in enumerate(record):
        try:
            observation, symbol = pair
        except (TypeError, ValueError) as exc:
            raise TypeError(f"{label}[{index}] must contain exactly one observation and input symbol.") from exc
        observations.append(observation)
        raw_inputs.append(symbol)
    return observations, _validated_input_symbols(raw_inputs, n_inputs, f"{label} inputs")


@dataclass(frozen=True)
class HMMFitDiagnostics:
    """Machine-readable receipt for a structured-family EM fit."""

    algorithm: str
    converged: bool
    iterations: int
    termination_reason: str
    log_likelihood_trace: tuple[float, ...]
    initial_log_likelihood: float
    final_log_likelihood: float
    final_absolute_delta: float | None
    final_relative_delta: float | None
    monotone: bool
    n_sequences: int
    total_weight: float
    approximate: bool


class HMMFitResult(tuple):
    """Backward-compatible ``(model, trace)`` result with a fit diagnostics receipt."""

    diagnostics: HMMFitDiagnostics

    def __new__(cls, model: Any, trace: list[float], diagnostics: HMMFitDiagnostics):
        result = super().__new__(cls, (model, trace))
        result.diagnostics = diagnostics
        return result

    @property
    def model(self):
        """Return the fitted model."""
        return self[0]

    @property
    def log_likelihood_trace(self):
        """Return the accepted model log-likelihood trajectory."""
        return self[1]


def _fit_delta(previous: float, current: float) -> tuple[float, float]:
    """Return absolute and scale-relative likelihood changes."""
    absolute = current - previous
    return absolute, absolute / max(1.0, abs(previous))


def _weighted_fit_log_likelihood(scores: Any, weights: np.ndarray, context: str) -> float:
    """Validate per-record evidence and return its finite weighted sum."""
    likelihoods = require_possible_log_evidence(scores, context=context)
    if likelihoods.shape != weights.shape:
        raise RuntimeError(f"{context} returned {likelihoods.size} scores for {weights.size} weights.")
    total = float(np.dot(weights, likelihoods))
    if not np.isfinite(total):
        raise RuntimeError(f"{context} produced a non-finite weighted log likelihood.")
    return total


def _fit_receipt(
    *,
    algorithm: str,
    trace: list[float],
    converged: bool,
    iterations: int,
    termination_reason: str,
    n_sequences: int,
    total_weight: float,
    approximate: bool,
) -> HMMFitDiagnostics:
    """Build a validated diagnostics receipt from an accepted likelihood trajectory."""
    likelihoods = np.asarray(trace, dtype=np.float64)
    if likelihoods.ndim != 1 or likelihoods.size == 0 or np.any(~np.isfinite(likelihoods)):
        raise RuntimeError(f"{algorithm} did not produce a finite likelihood trajectory.")
    differences = np.diff(likelihoods)
    monotonicity_allowance = 1.0e-8 * np.maximum(1.0, np.abs(likelihoods[:-1]))
    monotone = bool(np.all(differences >= -monotonicity_allowance))
    if not monotone and not approximate:
        raise RuntimeError(f"{algorithm} accepted a non-monotone likelihood trajectory.")
    if len(trace) > 1:
        absolute, relative = _fit_delta(trace[-2], trace[-1])
    else:
        absolute = relative = None
    return HMMFitDiagnostics(
        algorithm=algorithm,
        converged=converged,
        iterations=iterations,
        termination_reason=termination_reason,
        log_likelihood_trace=tuple(float(value) for value in trace),
        initial_log_likelihood=float(trace[0]),
        final_log_likelihood=float(trace[-1]),
        final_absolute_delta=None if absolute is None else float(absolute),
        final_relative_delta=None if relative is None else float(relative),
        monotone=monotone,
        n_sequences=n_sequences,
        total_weight=total_weight,
        approximate=approximate,
    )


def _numeric_matrix(values: Any, label: str) -> np.ndarray:
    """Return an owned finite non-negative two-dimensional matrix."""
    try:
        matrix = np.asarray(values, dtype=np.float64)
    except (TypeError, ValueError, OverflowError) as exc:
        raise TypeError(f"{label} must be a numeric matrix.") from exc
    if matrix.ndim != 2 or 0 in matrix.shape:
        raise ValueError(f"{label} must be a non-empty two-dimensional matrix.")
    if np.any(~np.isfinite(matrix)) or np.any(matrix < 0.0):
        raise ValueError(f"{label} must contain finite non-negative values.")
    return matrix.copy()


def _row_normalize(m: np.ndarray, fallback: np.ndarray | None = None) -> np.ndarray:
    """Normalize finite non-negative rows, retaining explicit fallback rows when unobserved."""
    matrix = _numeric_matrix(m, "transition counts")
    row_sums = matrix.sum(axis=1, keepdims=True)
    empty = row_sums[:, 0] == 0.0
    if np.any(empty):
        if fallback is None:
            raise ValueError("transition counts contain an empty row and no fallback law was supplied.")
        fallback_matrix = validated_row_probability_matrix(
            fallback,
            "transition fallback",
            shape=matrix.shape,
        )
        matrix[empty, :] = fallback_matrix[empty, :]
        row_sums = matrix.sum(axis=1, keepdims=True)
    return matrix / row_sums


def _log_probabilities(values: np.ndarray) -> np.ndarray:
    """Take an exact probability log while preserving structural zeros as ``-inf``."""
    with np.errstate(divide="ignore"):
        return np.log(values)


def _validated_transition_operator(operator: Any, n_states: int, label: str) -> TransitionOperator:
    """Validate a transition operator's shape, stochastic matrix, and linear maps."""
    if not isinstance(operator, TransitionOperator):
        raise TypeError(f"{label} must be a TransitionOperator.")
    declared_states = _exact_positive_integer(operator.n_states, f"{label} n_states")
    if declared_states != n_states:
        raise ValueError(f"{label} declares {declared_states} states, expected {n_states}.")
    matrix = validated_row_probability_matrix(
        operator.as_matrix(),
        f"{label} matrix",
        shape=(n_states, n_states),
    )
    probe = np.arange(1, n_states + 1, dtype=np.float64)
    probe /= probe.sum()
    forward = np.asarray(operator.forward(probe), dtype=np.float64)
    backward = np.asarray(operator.backward(probe), dtype=np.float64)
    if forward.shape != (n_states,) or not np.allclose(forward, probe @ matrix, rtol=1.0e-10, atol=1.0e-12):
        raise ValueError(f"{label}.forward is inconsistent with as_matrix().")
    if backward.shape != (n_states,) or not np.allclose(backward, matrix @ probe, rtol=1.0e-10, atol=1.0e-12):
        raise ValueError(f"{label}.backward is inconsistent with as_matrix().")
    return operator


class DenseTransition(TransitionOperator):
    """The usual dense K x K row-stochastic transition (O(K^2) forward-backward).

    ``prior`` (a K x K pseudocount matrix) is added to the expected counts before each M-step
    re-normalization -- a Dirichlet/MAP transition. A diagonal prior is a *sticky* self-transition bias
    (see :func:`sticky_transition`); a flat prior is symmetric-Dirichlet smoothing.
    """

    def __init__(self, a: np.ndarray, prior: np.ndarray | None = None) -> None:
        raw = _numeric_matrix(a, "DenseTransition matrix")
        if raw.shape[0] != raw.shape[1]:
            raise ValueError("DenseTransition matrix must be square.")
        self.a = validated_row_probability_matrix(
            raw,
            "DenseTransition matrix",
            shape=raw.shape,
        )
        self.n_states = self.a.shape[0]
        if prior is None:
            self.prior = None
        else:
            self.prior = _numeric_matrix(prior, "DenseTransition prior")
            if self.prior.shape != self.a.shape:
                raise ValueError(f"DenseTransition prior must have shape {self.a.shape}.")

    def forward(self, alpha):
        """Push a state-belief row vector forward with the dense matrix."""
        return alpha @ self.a

    def backward(self, v):
        """Pull a vector backward with the dense matrix."""
        return self.a @ v

    def as_matrix(self):
        """Return the dense row-stochastic transition matrix."""
        return self.a

    def new_accumulator(self):
        """Create dense expected-transition-count storage."""
        return np.zeros_like(self.a)

    def accumulate(self, acc, alpha_t, w_next, scale):
        """Accumulate one dense expected-transition-count contribution."""
        acc += np.outer(alpha_t, w_next) * (self.a / max(scale, 1e-300))

    def estimate(self, acc):
        """Estimate a row-normalized dense transition from expected counts."""
        counts = _numeric_matrix(acc, "DenseTransition expected counts")
        if counts.shape != self.a.shape:
            raise ValueError(f"DenseTransition expected counts must have shape {self.a.shape}.")
        effective = counts if self.prior is None else counts + self.prior
        return DenseTransition(_row_normalize(effective, self.a), self.prior)


def sticky_transition(a, kappa: float) -> DenseTransition:
    """A dense transition with a STICKY self-transition prior: ``kappa`` pseudocounts on the diagonal
    favor staying in a state (longer dwell times, cleaner segmentation -- the sticky-HMM idea)."""
    transition = DenseTransition(a)
    try:
        strength = float(kappa)
    except (TypeError, ValueError, OverflowError) as exc:
        raise TypeError("sticky transition kappa must be numeric.") from exc
    if not np.isfinite(strength) or strength < 0.0:
        raise ValueError("sticky transition kappa must be finite and non-negative.")
    return DenseTransition(transition.a, prior=strength * np.eye(transition.n_states))


def dirichlet_transition(a, alpha: float) -> DenseTransition:
    """A dense transition with a symmetric Dirichlet(``alpha``) smoothing prior on every row (MAP)."""
    transition = DenseTransition(a)
    try:
        strength = float(alpha)
    except (TypeError, ValueError, OverflowError) as exc:
        raise TypeError("Dirichlet transition alpha must be numeric.") from exc
    if not np.isfinite(strength) or strength < 0.0:
        raise ValueError("Dirichlet transition alpha must be finite and non-negative.")
    return DenseTransition(
        transition.a,
        prior=np.full((transition.n_states, transition.n_states), strength),
    )


def kron_initial(pi1, pi2) -> np.ndarray:
    """Factorized initial distribution ``pi1 (x) pi2`` for a factorial (Kronecker) HMM -- the two chains
    start independently. Matches a :class:`KroneckerTransition` so the joint initial respects the factors."""
    first = validated_probability_vector(pi1, "first Kronecker initial probabilities")
    second = validated_probability_vector(pi2, "second Kronecker initial probabilities")
    return np.kron(first, second)


class LowRankTransition(TransitionOperator):
    """A = G @ Phi: each state's next-state distribution is a mix of ``r`` shared transition profiles.

    ``G`` is K x r row-stochastic (state -> profile mixing), ``Phi`` is r x K row-stochastic (profile ->
    next-state). ``A = G @ Phi`` is K x K row-stochastic with rank <= r. Forward (``(alpha @ G) @ Phi``),
    backward (``G @ (Phi @ v)``) and the M-step are all O(K r) -- never forming A -- and the parameter
    count is 2 K r instead of K^2.
    """

    def __init__(self, g: np.ndarray, phi: np.ndarray) -> None:
        raw_g = _numeric_matrix(g, "LowRankTransition g")
        raw_phi = _numeric_matrix(phi, "LowRankTransition phi")
        if raw_g.shape[1] != raw_phi.shape[0] or raw_g.shape[0] != raw_phi.shape[1]:
            raise ValueError(
                "LowRankTransition requires g shape (K, r) and phi shape (r, K); "
                f"got {raw_g.shape} and {raw_phi.shape}."
            )
        self.g = validated_row_probability_matrix(raw_g, "LowRankTransition g", shape=raw_g.shape)
        self.phi = validated_row_probability_matrix(raw_phi, "LowRankTransition phi", shape=raw_phi.shape)
        self.n_states = self.g.shape[0]
        self.rank = self.g.shape[1]

    def forward(self, alpha):
        """Push a state-belief row vector through the low-rank transition."""
        return (alpha @ self.g) @ self.phi  # (alpha^T A)

    def backward(self, v):
        """Pull a vector backward through the low-rank transition."""
        return self.g @ (self.phi @ v)  # (A v)

    def as_matrix(self):
        """Materialize the implied dense transition matrix."""
        return self.g @ self.phi

    def new_accumulator(self):
        """Create state-profile and profile-next sufficient-statistic storage."""
        return [np.zeros_like(self.g), np.zeros_like(self.phi)]  # [n (K,r), m (r,K)]

    def accumulate(self, acc, alpha_t, w_next, scale):
        """Accumulate one exact low-rank transition contribution."""
        # exact expected mass of the latent profile r on this transition, in O(K r) (no K x K matrix):
        #   u[r]   = sum_i alpha_t[i] G[i,r]            (alpha into profiles)
        #   v[r]   = sum_j Phi[r,j] w_next[j]           (profiles' emission-weighted reach)
        #   n[i,r] += alpha_t[i] G[i,r] v[r] / scale    (state->profile counts, for G)
        #   m[r,j] += u[r] Phi[r,j] w_next[j] / scale   (profile->next counts, for Phi)
        inv = 1.0 / max(scale, 1e-300)
        u = alpha_t @ self.g  # (r,)
        v = self.phi @ w_next  # (r,)
        acc[0] += (alpha_t[:, None] * self.g) * v[None, :] * inv
        acc[1] += (u[:, None] * self.phi) * w_next[None, :] * inv

    def estimate(self, acc):
        """Estimate low-rank transition factors from accumulated statistics."""
        if not isinstance(acc, (list, tuple)) or len(acc) != 2:
            raise ValueError("LowRankTransition expected counts must contain g and phi matrices.")
        return LowRankTransition(
            _row_normalize(acc[0], self.g),
            _row_normalize(acc[1], self.phi),
        )


class SparseTransition(TransitionOperator):
    """Only the given ``(from, to)`` edges are allowed (left-to-right / banded HMMs). Forward, backward
    and the M-step are O(#edges) -- transitions outside the edge set stay exactly zero through EM, so the
    structure is preserved. Build edges yourself or with :func:`left_to_right_edges` / :func:`banded_edges`."""

    def __init__(self, n_states: int, edges, values=None) -> None:
        from scipy.sparse import csr_matrix

        self.n_states = _exact_positive_integer(n_states, "SparseTransition n_states")
        try:
            edge_list = list(edges)
        except TypeError as exc:
            raise TypeError("SparseTransition edges must be an iterable of integer pairs.") from exc
        if not edge_list:
            raise ValueError("SparseTransition edges must not be empty.")
        normalized_edges: list[tuple[int, int]] = []
        for index, edge in enumerate(edge_list):
            try:
                source, target = edge
            except (TypeError, ValueError) as exc:
                raise TypeError(f"SparseTransition edge {index} must contain exactly two state IDs.") from exc
            if (
                isinstance(source, bool)
                or not isinstance(source, (int, np.integer))
                or isinstance(target, bool)
                or not isinstance(target, (int, np.integer))
            ):
                raise TypeError(f"SparseTransition edge {index} state IDs must be integers.")
            pair = (int(source), int(target))
            if any(state < 0 or state >= self.n_states for state in pair):
                raise ValueError(f"SparseTransition edge {index} {pair} is outside the state space.")
            normalized_edges.append(pair)
        if len(set(normalized_edges)) != len(normalized_edges):
            raise ValueError("SparseTransition edges must be unique.")
        self.rows = np.asarray([edge[0] for edge in normalized_edges], dtype=int)
        self.cols = np.asarray([edge[1] for edge in normalized_edges], dtype=int)
        if values is None:
            vals = np.ones(len(self.rows), dtype=np.float64)
        else:
            try:
                vals = np.asarray(values, dtype=np.float64)
            except (TypeError, ValueError, OverflowError) as exc:
                raise TypeError("SparseTransition values must be numeric.") from exc
            if vals.shape != (len(self.rows),):
                raise ValueError(f"SparseTransition values must have shape ({len(self.rows)},).")
            if np.any(~np.isfinite(vals)) or np.any(vals < 0.0):
                raise ValueError("SparseTransition values must be finite and non-negative.")
        a = csr_matrix((vals, (self.rows, self.cols)), shape=(self.n_states, self.n_states))
        rs = np.asarray(a.sum(axis=1)).ravel()
        empty_rows = np.flatnonzero(rs == 0.0)
        if len(empty_rows):
            raise ValueError(
                f"SparseTransition every state needs positive outgoing mass; empty rows {empty_rows.tolist()}."
            )
        from scipy.sparse import diags

        self.a = diags(1.0 / rs) @ a  # row-normalized csr
        self._edge_vals = np.asarray(self.a[self.rows, self.cols]).ravel()

    def forward(self, alpha):
        """Push a state-belief row vector through the sparse transition."""
        return alpha @ self.a

    def backward(self, v):
        """Pull a vector backward through the sparse transition."""
        return self.a @ v

    def as_matrix(self):
        """Materialize the sparse transition as a dense matrix."""
        return np.asarray(self.a.todense())

    def new_accumulator(self):
        """Create one expected-count slot per allowed edge."""
        return np.zeros(len(self.rows))  # one expected count per allowed edge

    def accumulate(self, acc, alpha_t, w_next, scale):
        """Accumulate one sparse expected-transition contribution."""
        acc += alpha_t[self.rows] * w_next[self.cols] * self._edge_vals / max(scale, 1e-300)

    def estimate(self, acc):
        """Estimate a sparse transition over the same allowed edge set."""
        try:
            counts = np.asarray(acc, dtype=np.float64)
        except (TypeError, ValueError, OverflowError) as exc:
            raise TypeError("SparseTransition expected counts must be numeric.") from exc
        if counts.shape != self._edge_vals.shape:
            raise ValueError(f"SparseTransition expected counts must have shape {self._edge_vals.shape}.")
        if np.any(~np.isfinite(counts)) or np.any(counts < 0.0):
            raise ValueError("SparseTransition expected counts must be finite and non-negative.")
        effective = counts.copy()
        for state in range(self.n_states):
            edge_indices = np.flatnonzero(self.rows == state)
            if effective[edge_indices].sum() == 0.0:
                effective[edge_indices] = self._edge_vals[edge_indices]
        return SparseTransition(
            self.n_states,
            list(zip(self.rows.tolist(), self.cols.tolist())),
            effective,
        )


def left_to_right_edges(n_states: int, skip: int = 1):
    """Edges for a left-to-right (Bakis) HMM: each state may stay or advance up to ``skip`` states."""
    return [(i, j) for i in range(n_states) for j in range(i, min(n_states, i + skip + 1))]


def banded_edges(n_states: int, bandwidth: int = 1):
    """Edges for a banded transition: state i connects to i-bandwidth .. i+bandwidth (local time-series)."""
    return [(i, j) for i in range(n_states) for j in range(max(0, i - bandwidth), min(n_states, i + bandwidth + 1))]


def _final_state_enumerate(hmm, len_dist, max_results=50):
    """Best-first enumeration of observation sequences in descending marginal probability for a
    StructuredHMM whose sequences must END in a ``final_states`` state (e.g. an HSMM expansion).
    Admissible A*: a prefix's forward log-vector + a backward upper bound UB[r][s] (logsumexp of r further
    steps from s using each state's best emission, ending in a final state) bounds every completion, so a
    popped complete sequence is in true descending order. Needs discrete (Categorical) emissions + a
    Categorical-like ``len_dist`` (a .pmap over lengths)."""
    import heapq

    from scipy.special import logsumexp

    from mixle.enumeration import EnumerationError

    try:
        symbols = sorted(set().union(*(set(e.pmap.keys()) for e in hmm.emissions)))
        lengths = sorted(int(x) for x in len_dist.pmap.keys())
    except AttributeError as exc:
        raise EnumerationError(hmm, reason="final-state enumeration needs Categorical emissions + len_dist") from exc
    final_mask = hmm.final_mask
    log_emit = {v: np.array([float(e.log_density(v)) for e in hmm.emissions]) for v in symbols}
    max_emit = np.max(np.stack([log_emit[v] for v in symbols]), axis=0)
    log_pi = _log_probabilities(hmm.pi)
    log_a = _log_probabilities(hmm.transition.as_matrix())
    log_len = {x: float(len_dist.log_density(x)) for x in lengths}
    l_max = max(lengths)
    ub = [np.where(final_mask, 0.0, -np.inf)]
    for _ in range(1, l_max):
        ub.append(logsumexp(log_a + (max_emit + ub[-1])[None, :], axis=1))

    def complete_score(fwd, t):
        return (logsumexp(fwd[final_mask]) + log_len[t]) if (t in log_len and final_mask.any()) else -np.inf

    def extend_ub(fwd, t):
        best = -np.inf
        for x in lengths:
            r = x - t
            if r == 0:
                best = max(best, complete_score(fwd, t))
            elif r > 0:
                best = max(best, logsumexp(fwd + ub[r]) + log_len[x])
        return best

    heap = []
    counter = 0
    for v in symbols:
        fwd = log_pi + log_emit[v]
        pri = extend_ub(fwd, 1)
        if np.isfinite(pri):
            heapq.heappush(heap, (-pri, counter, "p", (v,), fwd, 1))
            counter += 1
    out = []
    while heap and len(out) < max_results:
        neg, _, kind, prefix, fwd, t = heapq.heappop(heap)
        if kind == "c":
            out.append((list(prefix), -neg))
            continue
        sc = complete_score(fwd, t)
        if np.isfinite(sc):
            heapq.heappush(heap, (-sc, counter, "c", prefix, fwd, t))
            counter += 1
        if t < l_max:
            base = logsumexp(fwd[:, None] + log_a, axis=0)
            for v in symbols:
                fwd2 = base + log_emit[v]
                pri = extend_ub(fwd2, t + 1)
                if np.isfinite(pri):
                    heapq.heappush(heap, (-pri, counter, "p", prefix + (v,), fwd2, t + 1))
                    counter += 1
    return out


class FinalStateEnumeration:
    """Result of :func:`_final_state_enumerate`: ``top_k(k)`` -> [(sequence, log_prob), ...] descending."""

    def __init__(self, hmm, len_dist):
        self._hmm, self._len_dist = hmm, len_dist

    def top_k(self, k):
        """Return the top ``k`` final-state-constrained sequences."""
        return _final_state_enumerate(self._hmm, self._len_dist, max_results=int(k))


_DENSE_FB_NUMBA = None


def _dense_fb_numba():
    """Lazily build a numba-jitted scaled dense forward-backward returning (loglik, gamma, xi_sum)."""
    global _DENSE_FB_NUMBA
    if _DENSE_FB_NUMBA is not None:
        return _DENSE_FB_NUMBA
    from numba import njit

    @njit(cache=True)
    def fb(log_b, pi, a):  # log_b (T,K), pi (K,), a (K,K)
        t_len, k = log_b.shape
        b = np.empty((t_len, k))
        mxsum = 0.0
        for t in range(t_len):
            mx = log_b[t].max()
            mxsum += mx
            for j in range(k):
                b[t, j] = np.exp(log_b[t, j] - mx)
        alpha = np.empty((t_len, k))
        c = np.empty(t_len)
        s = 0.0
        for j in range(k):
            alpha[0, j] = pi[j] * b[0, j]
            s += alpha[0, j]
        c[0] = s
        for j in range(k):
            alpha[0, j] /= s
        for t in range(1, t_len):
            s = 0.0
            for j in range(k):
                acc = 0.0
                for i in range(k):
                    acc += alpha[t - 1, i] * a[i, j]
                alpha[t, j] = acc * b[t, j]
                s += alpha[t, j]
            c[t] = s
            for j in range(k):
                alpha[t, j] /= s
        loglik = mxsum
        for t in range(t_len):
            loglik += np.log(c[t])
        beta = np.empty((t_len, k))
        for j in range(k):
            beta[t_len - 1, j] = 1.0
        for t in range(t_len - 2, -1, -1):
            for i in range(k):
                acc = 0.0
                for j in range(k):
                    acc += a[i, j] * b[t + 1, j] * beta[t + 1, j]
                beta[t, i] = acc / c[t + 1]
        gamma = np.empty((t_len, k))
        for t in range(t_len):
            gs = 0.0
            for j in range(k):
                gamma[t, j] = alpha[t, j] * beta[t, j]
                gs += gamma[t, j]
            for j in range(k):
                gamma[t, j] /= gs
        xi = np.zeros((k, k))
        for t in range(t_len - 1):
            for i in range(k):
                ai = alpha[t, i]
                if ai == 0.0:
                    continue
                for j in range(k):
                    xi[i, j] += ai * a[i, j] * b[t + 1, j] * beta[t + 1, j] / c[t + 1]
        return loglik, gamma, xi

    _DENSE_FB_NUMBA = fb
    return fb


class StructuredHMM:
    """An HMM whose transition is a :class:`TransitionOperator` (dense / low-rank / a combinator).

    ``emissions`` is one observation distribution per state; ``pi`` the initial-state distribution;
    ``transition`` any ``TransitionOperator``. The scaled forward-backward and EM call the operator's
    ``forward``/``backward``/``accumulate``/``estimate``, so a low-rank or factorial transition runs the
    SAME inference at its own cost (O(K r) for low-rank). ``emission_estimators`` (one per state) drives
    the emission M-step; default reuses ``emissions[k].estimator()``.
    """

    def __init__(
        self,
        emissions,
        pi,
        transition: TransitionOperator,
        emission_estimators=None,
        keys=(None, None),
        name=None,
        len_dist=None,
        terminal_states=None,
        final_states=None,
    ) -> None:
        try:
            self.emissions = list(emissions)
        except TypeError as exc:
            raise TypeError("StructuredHMM emissions must be an iterable.") from exc
        if not self.emissions:
            raise ValueError("StructuredHMM requires at least one emission distribution.")
        self.K = len(self.emissions)
        self.pi = validated_probability_vector(pi, "StructuredHMM initial probabilities", size=self.K)
        self.transition = _validated_transition_operator(transition, self.K, "StructuredHMM transition")
        self.keys = tuple(keys)  # (init_key, trans_key) for parameter tying across models
        if len(self.keys) != 2:
            raise ValueError("StructuredHMM keys must contain exactly two entries.")
        self.name = name
        self.len_dist = len_dist  # optional distribution over sequence length (needed for enumeration)
        # terminal (absorbing) states: when set, the sequence length is a STOPPING TIME -- the chain only
        # transitions FROM non-terminal states and the sequence must END in a terminal state (no len_dist).
        self.terminal_states = validated_terminal_states(
            terminal_states,
            self.K,
            context="StructuredHMM",
        )
        self.term_mask = None
        if self.terminal_states is not None:
            self.term_mask = np.zeros(self.K, dtype=bool)
            self.term_mask[list(self.terminal_states)] = True
        # final states: the sequence may END only in one of these (a NON-absorbing boundary -- unlike
        # terminal_states, the chain still transitions through them mid-sequence). Used by the HSMM->HMM
        # expansion to require the final segment to complete. terminal_states takes precedence if both set.
        self.final_states = validated_state_ids(
            final_states,
            self.K,
            context="StructuredHMM",
            field="final_states",
        )
        self.final_mask = None
        if self.final_states is not None:
            self.final_mask = np.zeros(self.K, dtype=bool)
            self.final_mask[list(self.final_states)] = True
        if self.terminal_states is not None and self.final_states is not None:
            raise ValueError("StructuredHMM cannot combine terminal_states and final_states")
        validate_terminal_reachability(
            self.pi,
            self.transition.as_matrix(),
            self.terminal_states,
            context="StructuredHMM",
        )
        if emission_estimators is None:
            self._emit_est = [emission.estimator() for emission in self.emissions]
        else:
            self._emit_est = list(emission_estimators)
            if len(self._emit_est) != self.K:
                raise ValueError(f"StructuredHMM requires exactly {self.K} emission estimators.")

    def _log_b(self, seq) -> np.ndarray:
        return np.array([[float(e.log_density(x)) for e in self.emissions] for x in seq])

    def _forward_backward(self, log_b, pi=None):
        if self.term_mask is not None:
            return self._terminal_forward_backward(log_b, pi=pi)
        if self.final_mask is not None:
            return self._final_forward_backward(log_b, pi=pi)
        T, _ = log_b.shape
        op = self.transition
        mx = log_b.max(axis=1, keepdims=True)
        b = np.exp(log_b - np.where(np.isfinite(mx), mx, 0.0))  # (T,K) scaled; all-impossible row -> b=0

        alpha = np.zeros((T, self.K))
        c = np.zeros(T)
        alpha[0] = (self.pi if pi is None else pi) * b[0]
        c[0] = alpha[0].sum()
        alpha[0] = alpha[0] / c[0] if c[0] > 0 else alpha[0]
        for t in range(1, T):
            alpha[t] = op.forward(alpha[t - 1]) * b[t]
            c[t] = alpha[t].sum()
            alpha[t] = alpha[t] / c[t] if c[t] > 0 else alpha[t]
        with np.errstate(divide="ignore"):  # an impossible observation gives c=0 -> loglik -inf, not NaN
            loglik = float(np.sum(np.log(c)) + np.sum(mx))  # add the per-step maxima back
        beta = np.zeros((T, self.K))
        beta[T - 1] = 1.0
        for t in range(T - 2, -1, -1):
            beta[t] = op.backward(b[t + 1] * beta[t + 1]) / c[t + 1] if c[t + 1] > 0 else np.zeros(self.K)
        gamma = alpha * beta
        gsum = gamma.sum(axis=1, keepdims=True)
        gamma = np.divide(gamma, gsum, out=np.zeros_like(gamma), where=gsum > 0)
        return alpha, beta, c, b, gamma, loglik

    def _final_forward_backward(self, log_b, pi=None):
        """Standard forward, but the sequence may end only in a ``final_states`` state (non-absorbing): the
        likelihood sums the FINAL position over final_states and the backward boundary is the final mask.
        Transitions are unrestricted (the chain passes through final states normally mid-sequence)."""
        T, _ = log_b.shape
        op = self.transition
        mx = log_b.max(axis=1, keepdims=True)
        b = np.exp(log_b - np.where(np.isfinite(mx), mx, 0.0))  # all-impossible row -> b=0, not NaN
        alpha = np.zeros((T, self.K))
        c = np.zeros(T)
        alpha[0] = (self.pi if pi is None else pi) * b[0]
        c[0] = alpha[0].sum()
        alpha[0] = alpha[0] / c[0] if c[0] > 0 else alpha[0]
        for t in range(1, T):
            alpha[t] = op.forward(alpha[t - 1]) * b[t]
            c[t] = alpha[t].sum()
            alpha[t] = alpha[t] / c[t] if c[t] > 0 else alpha[t]
        final_mass = float(alpha[T - 1][self.final_mask].sum())
        with np.errstate(divide="ignore"):
            loglik = float(np.sum(np.log(c)) + np.sum(mx) + np.log(final_mass))
        beta = np.zeros((T, self.K))
        beta[T - 1] = np.where(self.final_mask, 1.0, 0.0)
        for t in range(T - 2, -1, -1):
            backward = op.backward(b[t + 1] * beta[t + 1])
            beta[t] = backward / c[t + 1] if c[t + 1] > 0 else np.zeros_like(backward)
        gamma = alpha * beta
        gs = gamma.sum(axis=1, keepdims=True)
        gamma = np.divide(gamma, gs, out=np.zeros_like(gamma), where=gs > 0)
        return alpha, beta, c, b, gamma, loglik

    def _terminal_forward_backward(self, log_b, pi=None):
        """Forward-backward when states are terminal (absorbing): the chain transitions only FROM
        non-terminal states (mask the belief at terminal states before each step) and the sequence must end
        in a terminal state. Works through the operator interface, so any transition structure is supported.
        Returns the SAME tuple as the standard path; the alpha returned is already terminal-masked for the
        transition accumulate, and the loglik is the terminal-stopping-time likelihood."""
        T, _ = log_b.shape
        op = self.transition
        nonterm = ~self.term_mask
        mx = log_b.max(axis=1, keepdims=True)
        b = np.exp(log_b - np.where(np.isfinite(mx), mx, 0.0))  # all-impossible row -> b=0, not NaN
        alpha = np.zeros((T, self.K))
        c = np.zeros(T)
        alpha[0] = (self.pi if pi is None else pi) * b[0]
        c[0] = alpha[0].sum()
        alpha[0] = alpha[0] / c[0] if c[0] > 0 else alpha[0]
        for t in range(1, T):
            alpha[t] = op.forward(np.where(nonterm, alpha[t - 1], 0.0)) * b[t]  # only leave non-terminal states
            c[t] = alpha[t].sum()
            alpha[t] = alpha[t] / c[t] if c[t] > 0 else alpha[t]
        term_mass = float(alpha[T - 1][self.term_mask].sum())  # must end in a terminal state
        with np.errstate(divide="ignore"):
            loglik = float(np.sum(np.log(c)) + np.sum(mx) + np.log(term_mass))
        beta = np.zeros((T, self.K))
        beta[T - 1] = np.where(self.term_mask, 1.0, 0.0)  # only terminal states close a sequence
        for t in range(T - 2, -1, -1):
            backward = np.where(nonterm, op.backward(b[t + 1] * beta[t + 1]), 0.0)
            beta[t] = backward / c[t + 1] if c[t + 1] > 0 else np.zeros_like(backward)
        gamma = alpha * beta
        gs = gamma.sum(axis=1, keepdims=True)
        gamma = np.divide(gamma, gs, out=np.zeros_like(gamma), where=gs > 0)
        alpha_masked = alpha * nonterm[None, :]  # terminal states contribute no outgoing transition mass
        return alpha_masked, beta, c, b, gamma, loglik

    def viterbi(self, seq):
        """Most-likely state path (Viterbi / max-product). Uses the transition matrix, so it works for any
        operator; O(T K^2) -- a read-out, not the EM hot loop."""
        log_b = self._log_b(seq)
        require_possible_log_evidence(
            self._forward_backward(log_b)[5],
            context="StructuredHMM.viterbi",
        )
        log_a = _log_probabilities(self.transition.as_matrix())
        log_pi = _log_probabilities(self.pi)
        t_len, k = log_b.shape
        delta = np.zeros((t_len, k))
        psi = np.zeros((t_len, k), dtype=int)
        delta[0] = log_pi + log_b[0]
        for t in range(1, t_len):
            previous = delta[t - 1]
            if self.term_mask is not None:
                previous = np.where(~self.term_mask, previous, -np.inf)
            m = previous[:, None] + log_a  # (from, to)
            psi[t] = np.argmax(m, axis=0)
            delta[t] = m[psi[t], np.arange(k)] + log_b[t]
        path = np.zeros(t_len, dtype=int)
        ending = delta[-1]
        if self.term_mask is not None:
            ending = np.where(self.term_mask, ending, -np.inf)
        elif self.final_mask is not None:
            ending = np.where(self.final_mask, ending, -np.inf)
        path[-1] = int(np.argmax(ending))
        for t in range(t_len - 2, -1, -1):
            path[t] = psi[t + 1, path[t + 1]]
        return path

    def posterior_decode(self, seq):
        """Per-position MAP state argmax_k P(z_t = k | x) from the forward-backward posteriors gamma."""
        return np.argmax(self.state_posteriors(seq), axis=1)

    def enumerator(self):
        """Enumerate observation sequences in descending marginal probability (top_k / rank / seek /
        nucleus / certified estimates). Enumeration depends only on pi, the transition MATRIX, the
        emissions and a length distribution -- not on the operator's internal structure -- so it reuses
        the built-in HMM enumerator (an A*-style best-first search over the trellis) on the dense matrix.
        Requires ``len_dist`` (a distribution over sequence length) and enumerable (discrete) emissions."""
        from mixle.enumeration import EnumerationError
        from mixle.stats.latent.hidden_markov import HiddenMarkovModelDistribution

        if self.len_dist is None:
            raise EnumerationError(self, reason="StructuredHMM needs a len_dist to enumerate sequence length")
        if self.final_mask is not None:
            # sequences must end in a final state (e.g. an HSMM expansion) -- the built-in enumerator does
            # not honor a final-state mask, so use the dedicated final-state best-first enumerator.
            return FinalStateEnumeration(self, self.len_dist)
        dense = HiddenMarkovModelDistribution(
            self.emissions,
            w=self.pi.tolist(),
            transitions=self.transition.as_matrix().tolist(),
            len_dist=self.len_dist,
        )
        return dense.enumerator()

    def dist_to_enumerator(self):
        """Return the sequence enumerator for this HMM."""
        return self.enumerator()

    def state_posteriors(self, seq):
        """The full smoothing posteriors gamma[t,k] = P(z_t = k | x)."""
        result = self._forward_backward(self._log_b(seq))
        require_possible_log_evidence(result[5], context="StructuredHMM.state_posteriors")
        return result[4]

    def seq_log_density(self, seqs) -> np.ndarray:
        """Score a batch of observation sequences."""
        return np.array([self._forward_backward(self._log_b(s))[5] for s in seqs])

    def sampler(self, seed=None):
        """Return a sampler for observation sequences."""
        return _StructuredHMMSampler(self, seed)

    def _can_fast_fb(self):
        """The numba dense forward-backward applies for a plain dense transition with no terminal states."""
        from mixle.utils.optional_deps import HAS_NUMBA

        return (
            HAS_NUMBA
            and self.term_mask is None
            and self.final_mask is None  # the fast kernel is unconstrained; final_states needs the masked backward
            and type(self.transition) is DenseTransition
        )

    def fit(self, seqs, *, max_its: int = 50, tol: float = 1e-6, fast: bool = True, weights=None):
        """Fit by Baum-Welch through the transition operator.

        ``fast=True`` uses the numba-jitted dense forward-backward when possible. The returned
        :class:`HMMFitResult` remains unpackable as ``(fitted_hmm, loglik_trace)`` and also exposes a
        validated ``diagnostics`` receipt.
        """
        max_its, tol = _validated_fit_controls(max_its, tol)
        if not isinstance(fast, (bool, np.bool_)):
            raise TypeError("fast must be a boolean.")
        seqs = _validated_sequences(seqs, "StructuredHMM fit data")
        weights = _validated_weights(weights, len(seqs))
        active_indices = np.flatnonzero(weights > 0.0)
        use_fast = require_exact_bool(fast, "fast") and self._can_fast_fb()
        fb = _dense_fb_numba() if use_fast else None
        ll_trace: list[float] = []
        iterations = 0
        converged = False
        termination_reason = "max_iterations"
        rollback = None

        for _ in range(max_its):
            active = [seqs[index] for index in active_indices]
            try:
                total_ll = _weighted_fit_log_likelihood(
                    self.seq_log_density(active),
                    weights[active_indices],
                    "StructuredHMM.fit",
                )
            except (ImpossibleEvidenceError, ValueError, RuntimeError, FloatingPointError):
                if rollback is None:
                    raise
                self.pi, self.transition, self.emissions = rollback
                iterations -= 1
                termination_reason = "invalid_update_rejected"
                break
            if ll_trace:
                absolute, _ = _fit_delta(ll_trace[-1], total_ll)
                allowance = 1.0e-8 * max(1.0, abs(ll_trace[-1]))
                if absolute < -allowance:
                    if rollback is not None:
                        self.pi, self.transition, self.emissions = rollback
                        iterations -= 1
                    termination_reason = "non_monotone_update_rejected"
                    break
                ll_trace.append(total_ll)
                if abs(absolute) <= tol * max(1.0, abs(ll_trace[-2])):
                    converged = True
                    termination_reason = "converged"
                    break
            else:
                ll_trace.append(total_ll)

            trans_acc = self.transition.new_accumulator()
            pi_acc = np.zeros(self.K)
            emit_accs = [est.accumulator_factory().make() for est in self._emit_est]
            nk = np.zeros(self.K)
            a_mat = self.transition.as_matrix() if use_fast else None
            for index in active_indices:
                seq = seqs[index]
                weight = weights[index]
                log_b = self._log_b(seq)
                if use_fast:
                    _, gamma, xi = fb(log_b, self.pi, a_mat)
                    trans_acc += weight * xi
                else:
                    alpha, beta, c, b, gamma, _ = self._forward_backward(log_b)
                    for t in range(len(seq) - 1):
                        self.transition.accumulate(
                            trans_acc,
                            alpha[t],
                            weight * b[t + 1] * beta[t + 1],
                            c[t + 1],
                        )
                pi_acc += weight * gamma[0]
                for k in range(self.K):
                    enc = self.emissions[k].dist_to_encoder().seq_encode(seq)
                    state_weights = weight * gamma[:, k]
                    emit_accs[k].seq_update(enc, state_weights, self.emissions[k])
                    nk[k] += state_weights.sum()
            rollback = (self.pi.copy(), self.transition, list(self.emissions))
            self.transition = self.transition.estimate(trans_acc)
            self.pi = pi_acc / pi_acc.sum()
            self.emissions = [self._emit_est[k].estimate(float(nk[k]), emit_accs[k].value()) for k in range(self.K)]
            iterations += 1

        if iterations and termination_reason == "max_iterations":
            try:
                final_ll = _weighted_fit_log_likelihood(
                    self.seq_log_density([seqs[index] for index in active_indices]),
                    weights[active_indices],
                    "StructuredHMM.fit final model",
                )
            except (ImpossibleEvidenceError, ValueError, RuntimeError, FloatingPointError):
                self.pi, self.transition, self.emissions = rollback
                iterations -= 1
                termination_reason = "invalid_update_rejected"
            else:
                absolute, _ = _fit_delta(ll_trace[-1], final_ll)
                allowance = 1.0e-8 * max(1.0, abs(ll_trace[-1]))
                if absolute < -allowance:
                    self.pi, self.transition, self.emissions = rollback
                    iterations -= 1
                    termination_reason = "non_monotone_update_rejected"
                else:
                    ll_trace.append(final_ll)
                    if abs(absolute) <= tol * max(1.0, abs(ll_trace[-2])):
                        converged = True
                        termination_reason = "converged"

        diagnostics = _fit_receipt(
            algorithm="structured-baum-welch",
            trace=ll_trace,
            converged=converged,
            iterations=iterations,
            termination_reason=termination_reason,
            n_sequences=len(active_indices),
            total_weight=float(weights.sum()),
            approximate=False,
        )
        return HMMFitResult(self, ll_trace, diagnostics)


class _StructuredHMMSampler:
    def __init__(self, hmm: StructuredHMM, seed=None):
        self.hmm = hmm
        self.rng = np.random.RandomState(seed)

    def sample(self, length: int):
        h = self.hmm
        a = h.transition.as_matrix()
        s = self.rng.choice(h.K, p=h.pi)
        out = []
        terminated = False
        for _ in range(int(length)):
            out.append(h.emissions[s].sampler(seed=int(self.rng.randint(1, 2**31))).sample())
            if h.term_mask is not None and h.term_mask[s]:
                terminated = True
                break  # terminal (absorbing) state ends the sequence -- length is the stopping time
            row = a[s]
            rs = row.sum()
            s = self.rng.choice(h.K, p=row / rs) if rs > 0 else s
        if h.term_mask is not None:
            require_terminal_reached(
                terminated,
                mode="structured terminal-state",
                max_steps=int(length),
                last_state=int(s),
            )
        return out


class BlockDiagonalTransition(TransitionOperator):
    """Independent sub-chains: the states partition into blocks and transitions stay within a block.

    A model whose initial state picks a block and then evolves inside it -- a mixture of regimes that do
    not switch. Build it from any sub-operators (each block can itself be dense or low-rank). Exact,
    block-local forward-backward and M-step.
    """

    def __init__(self, blocks) -> None:
        try:
            self.blocks = list(blocks)
        except TypeError as exc:
            raise TypeError("BlockDiagonalTransition blocks must be an iterable of transition operators.") from exc
        if not self.blocks:
            raise ValueError("BlockDiagonalTransition requires at least one block.")
        if any(not isinstance(block, TransitionOperator) for block in self.blocks):
            raise TypeError("BlockDiagonalTransition blocks must all be TransitionOperator instances.")
        self.sizes = [
            _exact_positive_integer(block.n_states, f"BlockDiagonalTransition block {index} n_states")
            for index, block in enumerate(self.blocks)
        ]
        self.blocks = [
            _validated_transition_operator(block, size, f"BlockDiagonalTransition block {index}")
            for index, (block, size) in enumerate(zip(self.blocks, self.sizes))
        ]
        self.offsets = np.cumsum([0] + self.sizes)
        self.n_states = int(self.offsets[-1])

    def _slices(self):
        return [slice(int(self.offsets[i]), int(self.offsets[i + 1])) for i in range(len(self.blocks))]

    def forward(self, alpha):
        """Push each block's belief mass forward independently."""
        out = np.zeros(self.n_states)
        for b, sl in zip(self.blocks, self._slices()):
            out[sl] = b.forward(alpha[sl])
        return out

    def backward(self, v):
        """Pull each block's vector backward independently."""
        out = np.zeros(self.n_states)
        for b, sl in zip(self.blocks, self._slices()):
            out[sl] = b.backward(v[sl])
        return out

    def as_matrix(self):
        """Materialize the block-diagonal transition matrix."""
        from scipy.linalg import block_diag

        return block_diag(*[b.as_matrix() for b in self.blocks])

    def new_accumulator(self):
        """Create one transition accumulator per block."""
        return [b.new_accumulator() for b in self.blocks]

    def accumulate(self, acc, alpha_t, w_next, scale):
        """Accumulate block-local expected transition mass."""
        for b, a, sl in zip(self.blocks, acc, self._slices()):
            b.accumulate(a, alpha_t[sl], w_next[sl], scale)

    def estimate(self, acc):
        """Estimate each block and return a new block-diagonal transition."""
        if not isinstance(acc, (list, tuple)) or len(acc) != len(self.blocks):
            raise ValueError(
                f"BlockDiagonalTransition expected counts must contain exactly {len(self.blocks)} block values."
            )
        return BlockDiagonalTransition([b.estimate(a) for b, a in zip(self.blocks, acc)])


class KroneckerTransition(TransitionOperator):
    """Factorial HMM: the state is the pair ``(s1, s2)`` of two chains evolving in parallel, with
    ``A = A1 (x) A2`` (Kronecker). State index is ``i1 * K2 + i2``.

    Forward-backward uses the reshape identity (``alpha @ (A1 (x) A2)`` reshapes to ``A1^T @ M @ A2``),
    so a step is O(K1 K2 (K1 + K2)) instead of O((K1 K2)^2) -- the whole point of a factorial HMM. The
    E-step is *exact* over the joint state; the M-step is the standard factorial marginal update (each
    factor re-estimated from the marginalized joint transition mass), verified to keep EM monotone.
    """

    def __init__(self, op1: TransitionOperator, op2: TransitionOperator) -> None:
        if not isinstance(op1, TransitionOperator) or not isinstance(op2, TransitionOperator):
            raise TypeError("KroneckerTransition factors must be TransitionOperator instances.")
        self.k1 = _exact_positive_integer(op1.n_states, "KroneckerTransition first factor n_states")
        self.k2 = _exact_positive_integer(op2.n_states, "KroneckerTransition second factor n_states")
        self.op1 = _validated_transition_operator(op1, self.k1, "KroneckerTransition first factor")
        self.op2 = _validated_transition_operator(op2, self.k2, "KroneckerTransition second factor")
        self.n_states = self.k1 * self.k2

    def _a1(self):
        return self.op1.as_matrix()

    def _a2(self):
        return self.op2.as_matrix()

    def forward(self, alpha):
        """Push a joint belief through the Kronecker-factorized transition."""
        m = alpha.reshape(self.k1, self.k2)
        return (self._a1().T @ m @ self._a2()).reshape(-1)  # alpha @ (A1 (x) A2)

    def backward(self, v):
        """Pull a joint vector backward through the Kronecker transition."""
        m = v.reshape(self.k1, self.k2)
        return (self._a1() @ m @ self._a2().T).reshape(-1)  # (A1 (x) A2) @ v

    def as_matrix(self):
        """Materialize the dense Kronecker transition matrix."""
        return np.kron(self._a1(), self._a2())

    def new_accumulator(self):
        """Create marginal transition-count accumulators for both factors."""
        return [np.zeros((self.k1, self.k1)), np.zeros((self.k2, self.k2))]  # marginal factor counts

    def accumulate(self, acc, alpha_t, w_next, scale):
        """Accumulate exact marginalized factor transition statistics."""
        a1, a2 = self._a1(), self._a2()
        am = alpha_t.reshape(self.k1, self.k2)
        wm = w_next.reshape(self.k1, self.k2)
        inv = 1.0 / max(scale, 1e-300)
        acc[0] += a1 * (am @ (a2 @ wm.T)) * inv  # n1[i1,j1] = A1[i1,j1] * sum_{i2,j2} xi
        acc[1] += a2 * (am.T @ (a1 @ wm)) * inv  # n2[i2,j2] = A2[i2,j2] * sum_{i1,j1} xi

    def estimate(self, acc):
        """Estimate both Kronecker factors from marginalized expected counts."""
        if not isinstance(acc, (list, tuple)) or len(acc) != 2:
            raise ValueError("KroneckerTransition expected counts must contain exactly two factor matrices.")
        return KroneckerTransition(
            DenseTransition(_row_normalize(acc[0], self._a1())),
            DenseTransition(_row_normalize(acc[1], self._a2())),
        )


def _chunk_spans(t_len: int, chunk: int, overlap: int):
    """Yield (ctx_lo, ctx_hi, keep_lo, keep_hi): a window [ctx_lo:ctx_hi] whose interior [keep_lo:keep_hi]
    (relative to the window) is kept; the ``overlap`` context on each side is run only to forget the
    boundary, then discarded."""
    for start in range(0, t_len, chunk):
        end = min(start + chunk, t_len)
        ctx_lo, ctx_hi = max(0, start - overlap), min(t_len, end + overlap)
        yield ctx_lo, ctx_hi, start - ctx_lo, end - ctx_lo


def chunked_state_posteriors(hmm: StructuredHMM, seq, *, chunk: int, overlap: int) -> np.ndarray:
    """State posteriors gamma for one long sequence via overlapping chunks, each run INDEPENDENTLY
    (embarrassingly parallel). The first chunk uses the model's pi; interior chunks start from the uniform
    belief and the ``overlap`` context lets the chain *forget* that wrong boundary -- so the kept interior
    matches the exact forward-backward up to an error that decays at the mixing rate in ``overlap``."""
    chunk = _exact_positive_integer(chunk, "chunk")
    overlap = _exact_nonnegative_integer(overlap, "overlap")
    if overlap >= chunk:
        raise ValueError("overlap must be smaller than chunk.")
    sequence = _validated_sequences([seq], "chunked posterior data")[0]
    # seq_log_density is _forward_backward(_log_b(seq)), so guarding with it scored every emission a
    # second time -- the line below scores them all again. Same evidence, one scoring pass.
    log_b_full = hmm._log_b(sequence)
    require_possible_log_evidence(hmm._forward_backward(log_b_full)[5], context="chunked_state_posteriors")
    t_len = len(sequence)
    out = np.zeros((t_len, hmm.K))
    uniform = np.ones(hmm.K) / hmm.K
    for ctx_lo, ctx_hi, keep_lo, keep_hi in _chunk_spans(t_len, chunk, overlap):
        pi = hmm.pi if ctx_lo == 0 else uniform
        _, _, _, _, gamma, _ = hmm._forward_backward(log_b_full[ctx_lo:ctx_hi], pi=pi)
        out[ctx_lo + keep_lo : ctx_lo + keep_hi] = gamma[keep_lo:keep_hi]
    return out


def fit_chunked(
    hmm: StructuredHMM,
    seqs,
    *,
    chunk: int,
    overlap: int,
    max_its: int = 50,
    workers: int = 0,
    tol: float = 1e-6,
    weights=None,
):
    """Baum-Welch where each long sequence's forward-backward is split into overlapping chunks run in
    PARALLEL (the forgetting property bounds the boundary error). ``workers>0`` runs the per-chunk E-steps
    on a thread pool (NumPy releases the GIL in its array kernels); ``workers=0`` runs them serially. The
    interior suff-statistics are accumulated exactly as in :meth:`StructuredHMM.fit`; this only changes
    *how* the E-step is computed, trading a small, overlap-controlled approximation for intra-sequence
    parallelism. The returned :class:`HMMFitResult` remains unpackable as ``(fitted_hmm,
    loglik_trace)`` and marks the receipt's objective as approximate."""
    from concurrent.futures import ThreadPoolExecutor

    if not isinstance(hmm, StructuredHMM):
        raise TypeError("fit_chunked requires a StructuredHMM.")
    chunk = _exact_positive_integer(chunk, "chunk")
    overlap = _exact_nonnegative_integer(overlap, "overlap")
    if overlap >= chunk:
        raise ValueError("overlap must be smaller than chunk.")
    workers = _exact_nonnegative_integer(workers, "workers")
    max_its, tol = _validated_fit_controls(max_its, tol)
    seqs = _validated_sequences(seqs, "chunked StructuredHMM fit data")
    weights = _validated_weights(weights, len(seqs))
    active_indices = np.flatnonzero(weights > 0.0)
    uniform = np.ones(hmm.K) / hmm.K
    ll_trace: list[float] = []
    iterations = 0
    converged = False
    termination_reason = "max_iterations"
    rollback = None

    def chunk_estep(args):
        seq_index, seq, weight, log_b_full, ctx_lo, ctx_hi, keep_lo, keep_hi = args
        log_b = log_b_full[ctx_lo:ctx_hi]
        pi = hmm.pi if ctx_lo == 0 else uniform
        alpha, beta, c, b, gamma, ll = hmm._forward_backward(log_b, pi=pi)
        require_possible_log_evidence(ll, context="fit_chunked")
        # transition mass over kept interior transitions only; c is in the mx-scaled frame of
        # _forward_backward, so the kept window's per-position emission maxima must be added back
        # for the REPORTED trace (the full-fit loglik does the same) -- without them the returned
        # ll_trace is offset by a parameter-dependent amount. The convergence break below keeps
        # using the scaled-frame quantity so stopping behavior is unchanged.
        win = slice(keep_lo, max(keep_lo + 1, keep_hi))
        with np.errstate(divide="ignore"):
            contrib_scaled = float(np.sum(np.log(c[win])))
            contrib_ll = contrib_scaled + float(np.sum(log_b[win].max(axis=1)))
        return (
            seq_index,
            seq,
            weight,
            ctx_lo,
            keep_lo,
            keep_hi,
            alpha,
            beta,
            c,
            b,
            gamma,
            contrib_ll,
        )

    def run_estep():
        # _log_b scores every emission at every position of the whole sequence, so it is computed
        # once per sequence and sliced per chunk. Calling it inside chunk_estep re-scored the entire
        # sequence for every chunk, making emission cost grow with the chunk count -- the very split
        # this function exists to parallelize.
        log_b_by_index = {int(index): hmm._log_b(seqs[index]) for index in active_indices}
        tasks = [
            (index, seqs[index], weights[index], log_b_by_index[int(index)], lo, hi, keep_lo, keep_hi)
            for index in active_indices
            for (lo, hi, keep_lo, keep_hi) in _chunk_spans(len(seqs[index]), chunk, overlap)
        ]
        if workers > 1:
            with ThreadPoolExecutor(max_workers=workers) as ex:
                return list(ex.map(chunk_estep, tasks))
        return [chunk_estep(task) for task in tasks]

    for _ in range(max_its):
        try:
            results = run_estep()
        except (ImpossibleEvidenceError, ValueError, RuntimeError, FloatingPointError):
            if rollback is None:
                raise
            hmm.pi, hmm.transition, hmm.emissions = rollback
            iterations -= 1
            termination_reason = "invalid_update_rejected"
            break
        total_ll = float(sum(result[2] * result[-1] for result in results))
        if not np.isfinite(total_ll):
            if rollback is None:
                raise RuntimeError("fit_chunked produced a non-finite weighted approximate log likelihood.")
            hmm.pi, hmm.transition, hmm.emissions = rollback
            iterations -= 1
            termination_reason = "invalid_update_rejected"
            break
        if ll_trace:
            absolute, _ = _fit_delta(ll_trace[-1], total_ll)
            allowance = 1.0e-8 * max(1.0, abs(ll_trace[-1]))
            ll_trace.append(total_ll)
            if absolute >= -allowance and abs(absolute) <= tol * max(1.0, abs(ll_trace[-2])):
                converged = True
                termination_reason = "converged"
                break
        else:
            ll_trace.append(total_ll)
        trans_acc = hmm.transition.new_accumulator()
        pi_acc = np.zeros(hmm.K)
        emit_accs = [est.accumulator_factory().make() for est in hmm._emit_est]
        nk = np.zeros(hmm.K)
        for _, seq, weight, ctx_lo, keep_lo, keep_hi, alpha, beta, c, b, gamma, _ in results:
            if ctx_lo == 0:
                pi_acc += weight * gamma[0]
            for t in range(keep_lo, keep_hi):  # kept interior transitions
                if t + 1 < len(c):
                    hmm.transition.accumulate(
                        trans_acc,
                        alpha[t],
                        weight * b[t + 1] * beta[t + 1],
                        c[t + 1],
                    )
            seg = seq[ctx_lo + keep_lo : ctx_lo + keep_hi]
            for k in range(hmm.K):
                enc = hmm.emissions[k].dist_to_encoder().seq_encode(seg)
                state_weights = weight * gamma[keep_lo:keep_hi, k]
                emit_accs[k].seq_update(enc, state_weights, hmm.emissions[k])
                nk[k] += state_weights.sum()
        rollback = (hmm.pi.copy(), hmm.transition, list(hmm.emissions))
        hmm.transition = hmm.transition.estimate(trans_acc)
        hmm.pi = pi_acc / pi_acc.sum()
        hmm.emissions = [hmm._emit_est[k].estimate(float(nk[k]), emit_accs[k].value()) for k in range(hmm.K)]
        iterations += 1

    if iterations and termination_reason == "max_iterations":
        try:
            final_results = run_estep()
            final_ll = float(sum(result[2] * result[-1] for result in final_results))
            if not np.isfinite(final_ll):
                raise RuntimeError("fit_chunked final model produced a non-finite objective.")
        except (ImpossibleEvidenceError, ValueError, RuntimeError, FloatingPointError):
            hmm.pi, hmm.transition, hmm.emissions = rollback
            iterations -= 1
            termination_reason = "invalid_update_rejected"
        else:
            absolute, _ = _fit_delta(ll_trace[-1], final_ll)
            allowance = 1.0e-8 * max(1.0, abs(ll_trace[-1]))
            ll_trace.append(final_ll)
            if absolute >= -allowance and abs(absolute) <= tol * max(1.0, abs(ll_trace[-2])):
                converged = True
                termination_reason = "converged"

    diagnostics = _fit_receipt(
        algorithm="chunked-structured-baum-welch",
        trace=ll_trace,
        converged=converged,
        iterations=iterations,
        termination_reason=termination_reason,
        n_sequences=len(active_indices),
        total_weight=float(weights.sum()),
        approximate=True,
    )
    return HMMFitResult(hmm, ll_trace, diagnostics)


# ===================================================================================================
# The 5-part estimator contract: makes StructuredHMM a SequenceEncodableProbabilityDistribution that
# optimize()/run_em() can fit directly (optimize(seqs, hmm.estimator())). The E-step (forward-backward
# per sequence) lives in the accumulator; the M-step (pi / transition-operator / emission re-estimation)
# in the estimator. Keys (init_key, trans_key) let two HMMs TIE their initial / transition parameters.
# ===================================================================================================
from mixle.stats.compute.pdist import (  # noqa: E402
    DataSequenceEncoder,
    ParameterEstimator,
    SequenceEncodableProbabilityDistribution,
    SequenceEncodableStatisticAccumulator,
    StatisticAccumulatorFactory,
)
from mixle.utils.exact import require_exact_bool


def _add_nested(a, b):
    return a + b if isinstance(a, np.ndarray) else [_add_nested(x, y) for x, y in zip(a, b)]


def _scale_nested(a, f):
    return a * f if isinstance(a, np.ndarray) else [_scale_nested(x, f) for x in a]


def _copy_nested(value):
    """Copy an arbitrarily nested transition sufficient statistic."""
    return deepcopy(value)


def _require_finite_nested(value, *, name, label):
    """Finiteness postcondition over every ndarray leaf of a nested transition statistic."""
    if isinstance(value, np.ndarray):
        require_finite_count_totals(((name, value),), label=label)
    else:
        for part in value:
            _require_finite_nested(part, name=name, label=label)


def _validated_structured_hmm_statistics(values, states, *, label, transition_count=None):
    """Validate common structured/IO HMM statistic geometry."""
    pi_acc, trans_acc, emit_vals, nk = validated_statistic_tuple(values, 4, label)
    pi_acc = validated_count_array(pi_acc, (states,), f"{label} initial counts")
    nk = validated_count_array(nk, (states,), f"{label} emission counts")
    if not isinstance(emit_vals, (tuple, list)) or len(emit_vals) != states:
        raise ValueError(f"{label} emission statistics must have one item per state")
    if transition_count is not None and (
        not isinstance(trans_acc, (tuple, list)) or len(trans_acc) != transition_count
    ):
        raise ValueError(f"{label} transition statistics must have {transition_count} entries")
    return pi_acc, _copy_nested(trans_acc), tuple(emit_vals), nk


def _validated_edhmm_statistics(values, states, durations, *, label):
    """Validate explicit-duration HMM statistics and segment-mass laws."""
    pi_acc, trans_acc, dur_acc, emit_vals, nk = validated_statistic_tuple(values, 5, label)
    pi_acc = validated_count_array(pi_acc, (states,), f"{label} initial counts")
    trans_acc = validated_count_array(trans_acc, (states, states), f"{label} transition counts")
    dur_acc = validated_count_array(dur_acc, (states, durations), f"{label} duration counts")
    nk = validated_count_array(nk, (states,), f"{label} emission counts")
    if not isinstance(emit_vals, (tuple, list)) or len(emit_vals) != states:
        raise ValueError(f"{label} emission statistics must have one item per state")
    validate_effective_sample_mass(
        dur_acc.sum(),
        pi_acc.sum() + trans_acc.sum(),
        label=f"{label} segment mass",
    )
    return pi_acc, trans_acc, dur_acc, tuple(emit_vals), nk


class StructuredHMMDataEncoder(DataSequenceEncoder):
    """Sequences pass through as lists -- the structured forward-backward scores raw observations through
    the per-state emission ``log_density`` (no flattened columnar encoding; composability over raw speed)."""

    def seq_encode(self, x):
        """Encode sequences as lists without changing their observations."""
        return [list(s) for s in x]

    def row_count(self, x):
        """Return the number of pass-through structured-HMM records."""
        if not isinstance(x, list):
            raise ValueError("structured HMM encoding must be a list of sequence records")
        return len(x)

    def __eq__(self, other):
        return isinstance(other, StructuredHMMDataEncoder)

    def __hash__(self):
        return hash("StructuredHMMDataEncoder")


class StructuredHMMAccumulator(SequenceEncodableStatisticAccumulator):
    """Baum-Welch E-step accumulator: per-sequence forward-backward, accumulating initial-state mass,
    transition-operator mass, and per-state weighted emission statistics."""

    def __init__(self, emission_accumulators, transition_proto, keys=(None, None)) -> None:
        self.emit = list(emission_accumulators)
        self.K = len(self.emit)
        self.transition_proto = transition_proto
        self.pi_acc = np.zeros(self.K)
        self.trans_acc = transition_proto.new_accumulator()
        self.nk = np.zeros(self.K)
        self.init_key, self.trans_key = keys

    def update(self, x, weight, estimate):
        """Accumulate sufficient statistics from one weighted sequence."""
        weight = validated_observation_weight(weight, "structured-HMM observation weight")
        self.seq_update([x], np.array([weight], dtype=float), estimate)

    def seq_update(self, x, weights, estimate):
        """Run forward-backward and accumulate weighted sufficient statistics for a batch."""
        sequences = list(x)
        weights = _validated_weights(
            weights,
            len(sequences),
            "StructuredHMM accumulator weights",
            require_positive_total=False,
        )
        active_indices = np.flatnonzero(weights > 0.0)
        active = [sequences[index] for index in active_indices]
        if any(not sequence for sequence in active):
            raise ValueError("StructuredHMM accumulator data contains an empty positive-weight sequence.")
        if not active:
            return
        require_possible_log_evidence(
            estimate.seq_log_density(active),
            context="StructuredHMMAccumulator.seq_update",
        )
        for index in active_indices:
            seq = sequences[index]
            w = weights[index]
            log_b = estimate._log_b(seq)
            alpha, beta, c, b, gamma, _ = estimate._forward_backward(log_b)
            self.pi_acc += w * gamma[0]
            for t in range(len(seq) - 1):
                estimate.transition.accumulate(self.trans_acc, alpha[t], b[t + 1] * beta[t + 1] * w, c[t + 1])
            for k in range(self.K):
                enc = estimate.emissions[k].dist_to_encoder().seq_encode(seq)
                wk = gamma[:, k] * w
                self.emit[k].seq_update(enc, wk, estimate.emissions[k])
                self.nk[k] += float(wk.sum())

    def seq_initialize(self, x, weights, rng):
        """Initialize sufficient statistics with random soft state responsibilities."""
        sequences = list(x)
        weights = _validated_weights(
            weights,
            len(sequences),
            "StructuredHMM initializer weights",
            require_positive_total=False,
        )
        active_indices = np.flatnonzero(weights > 0.0)
        if any(not sequences[index] for index in active_indices):
            raise ValueError("StructuredHMM initializer data contains an empty positive-weight sequence.")
        if not len(active_indices):
            return
        # no model yet: seed with random soft responsibilities + a random transition accumulator
        self.trans_acc = _add_nested(self.trans_acc, self.transition_proto.random_accumulator(rng))
        for index in active_indices:
            seq = sequences[index]
            w = weights[index]
            g = rng.dirichlet(np.ones(self.K), len(seq))
            self.pi_acc += w * g[0]
            for k in range(self.K):
                enc = self.emit[k].acc_to_encoder().seq_encode(seq)
                wk = g[:, k] * w
                self.emit[k].seq_initialize(enc, wk, rng)
                self.nk[k] += float(wk.sum())

    def combine(self, suff_stat):
        """Merge serialized HMM sufficient statistics."""
        pi_acc, trans_acc, emit_vals, nk = _validated_structured_hmm_statistics(
            suff_stat,
            self.K,
            label="structured-HMM sufficient statistics",
        )
        # The ENTIRE combine is transactional with a finiteness postcondition: a child
        # rejecting its part mid-loop used to leave the counts and earlier children merged,
        # and individually valid statistics can sum to an infinite aggregate (measured in the
        # latent-family mutator audit; STAT-RR8-1/RR9-1 classes).
        _snapshot = snapshot_accumulator_statistics(self, count_attrs=("pi_acc", "nk"), child_attrs=("emit",))
        # _add_nested/_scale_nested rebind (they allocate); the original nested statistic is
        # never mutated in place, so restoring the original REFERENCE preserves both values
        # and the identity an external alias observes (STAT-RR11-3)
        _previous_trans = self.trans_acc
        self.pi_acc += pi_acc
        self.trans_acc = _add_nested(self.trans_acc, trans_acc)
        self.nk += nk
        try:
            require_finite_count_totals(
                (("initial counts", self.pi_acc), ("emission counts", self.nk)),
                label="combined structured-HMM",
            )
            _require_finite_nested(self.trans_acc, name="transition counts", label="combined structured-HMM")
            for k in range(self.K):
                self.emit[k].combine(emit_vals[k])
        except Exception:
            restore_accumulator_statistics(self, _snapshot)
            self.trans_acc = _previous_trans
            raise
        return self

    def value(self):
        """Return serialized HMM sufficient statistics."""
        return (
            self.pi_acc.copy(),
            _copy_nested(self.trans_acc),
            [e.value() for e in self.emit],
            self.nk.copy(),
        )

    def from_value(self, x):
        """Restore accumulator state from serialized sufficient statistics."""
        # Candidates validated before ANY assignment; children restore transactionally
        # (measured; STAT-RR9-1 class).
        candidate_pi, candidate_trans, emit_vals, candidate_nk = _validated_structured_hmm_statistics(
            x,
            self.K,
            label="structured-HMM sufficient statistics",
        )
        _snapshot = snapshot_accumulator_statistics(self, count_attrs=("pi_acc", "nk"), child_attrs=("emit",))
        # _add_nested/_scale_nested rebind (they allocate); the original nested statistic is
        # never mutated in place, so restoring the original REFERENCE preserves both values
        # and the identity an external alias observes (STAT-RR11-3)
        _previous_trans = self.trans_acc
        self.pi_acc, self.trans_acc, self.nk = candidate_pi, candidate_trans, candidate_nk
        try:
            for k in range(self.K):
                self.emit[k].from_value(emit_vals[k])
        except Exception:
            restore_accumulator_statistics(self, _snapshot)
            self.trans_acc = _previous_trans
            raise
        return self

    def scale(self, factor):
        """Multiply the running statistics by ``factor`` -- the decay primitive online/streaming
        Baum-Welch (StreamingEstimator) uses to fold a new batch into a forgetting running estimate."""
        f = validated_observation_weight(factor, "structured-HMM scale factor")
        # Parent statistics and children scale as ONE transaction with the scaled result
        # validated as a postcondition (measured; STAT-RR8-1/RR10-1 classes).
        _snapshot = snapshot_accumulator_statistics(self, count_attrs=("pi_acc", "nk"), child_attrs=("emit",))
        # _add_nested/_scale_nested rebind (they allocate); the original nested statistic is
        # never mutated in place, so restoring the original REFERENCE preserves both values
        # and the identity an external alias observes (STAT-RR11-3)
        _previous_trans = self.trans_acc
        self.pi_acc *= f
        self.trans_acc = _scale_nested(self.trans_acc, f)
        self.nk *= f
        try:
            require_finite_count_totals(
                (("initial counts", self.pi_acc), ("emission counts", self.nk)),
                label="scaled structured-HMM",
            )
            _require_finite_nested(self.trans_acc, name="transition counts", label="scaled structured-HMM")
            for e in self.emit:
                if hasattr(e, "scale"):
                    e.scale(f)
        except Exception:
            restore_accumulator_statistics(self, _snapshot)
            self.trans_acc = _previous_trans
            raise
        return self

    def acc_to_encoder(self):
        """Return the encoder associated with this accumulator."""
        return StructuredHMMDataEncoder()

    # parameter tying: pool initial / transition counts across accumulators sharing a key
    def key_merge(self, store):
        """Merge tied initial or transition sufficient statistics into ``store``."""
        # Transactional against the mapping, healed in place on failure: a later pool failing
        # used to leave the initial pool already merged (measured; STAT-RR9-1/RR10-1 classes),
        # and pooling reaches overflow by addition exactly as combine() does.
        _snapshot = deepcopy(store)
        try:
            if self.init_key is not None:
                if self.init_key in store:
                    pooled_pi = self.pi_acc + store[self.init_key]
                    require_finite_count_totals(
                        (("pooled initial counts", pooled_pi),), label="structured-HMM key merge"
                    )
                    store[self.init_key] = pooled_pi
                else:
                    # Copy on adoption: store must never alias this accumulator's own live pi_acc
                    # array. The "already present" branch above is safe (`+` always allocates a
                    # new array), but pi_acc IS mutated in place elsewhere (seq_update's `+=`,
                    # combine's `+=`, scale's `*=`), so without this copy a second tied
                    # accumulator's key_replace would still leave both accumulators pointing at
                    # this accumulator's own original, in-place-mutable array.
                    store[self.init_key] = self.pi_acc.copy()
            if self.trans_key is not None:
                if self.trans_key in store:
                    pooled_trans = _add_nested(store[self.trans_key], self.trans_acc)
                    _require_finite_nested(
                        pooled_trans, name="pooled transition counts", label="structured-HMM key merge"
                    )
                    store[self.trans_key] = pooled_trans
                else:
                    store[self.trans_key] = _copy_nested(self.trans_acc)
            for e in self.emit:
                if hasattr(e, "key_merge"):
                    e.key_merge(store)
        except Exception:
            heal_pooled_statistics(store, _snapshot)
            raise

    def key_replace(self, store):
        """Replace tied initial or transition statistics from ``store``."""
        # Candidates validated BEFORE assignment (a replacement used to land with no shape or
        # finiteness checks at all -- [inf, 0] went straight into pi_acc), and the whole
        # replace rolls back on any later failure (measured; STAT-RR8-1/RR9-1 classes).
        candidate_pi = None
        if self.init_key is not None and self.init_key in store:
            candidate_pi = validated_count_array(
                store[self.init_key],
                np.shape(self.pi_acc),
                "structured-HMM replacement initial counts",
            )
            require_finite_count_totals((("initial counts", candidate_pi),), label="structured-HMM key replace")
        candidate_trans = None
        if self.trans_key is not None and self.trans_key in store:
            candidate_trans = _copy_nested(store[self.trans_key])
            _require_finite_nested(
                candidate_trans, name="replacement transition counts", label="structured-HMM key replace"
            )

        _snapshot = snapshot_accumulator_statistics(self, count_attrs=("pi_acc",), child_attrs=("emit",))
        # the keyed-pooling protocol tests build accumulators via __new__ with only the fields
        # under test, so an absent trans_acc is skipped exactly as the snapshot helper skips it
        _previous_trans = self.trans_acc if hasattr(self, "trans_acc") else None
        if candidate_pi is not None:
            # Copy on replace too: without it, every tied accumulator ends up pointing at the
            # SAME array object, so any one of them later accumulating new local data (pi_acc
            # is mutated in place via += in seq_update/combine and *= in scale) would silently
            # corrupt every other tied accumulator's counts.
            self.pi_acc = candidate_pi.copy()
        if candidate_trans is not None:
            self.trans_acc = candidate_trans
        try:
            for e in self.emit:
                if hasattr(e, "key_replace"):
                    e.key_replace(store)
        except Exception:
            restore_accumulator_statistics(self, _snapshot)
            if _previous_trans is not None:
                self.trans_acc = _previous_trans
            raise


class StructuredHMMAccumulatorFactory(StatisticAccumulatorFactory):
    """Factory for structured HMM accumulators."""

    def __init__(self, emission_estimators, transition_proto, keys):
        self.emission_estimators = emission_estimators
        self.transition_proto = transition_proto
        self.keys = keys

    def make(self):
        """Create a fresh structured HMM accumulator."""
        emit = [est.accumulator_factory().make() for est in self.emission_estimators]
        return StructuredHMMAccumulator(emit, self.transition_proto, self.keys)


class StructuredHMMEstimator(ParameterEstimator):
    """Estimator (M-step) for a :class:`StructuredHMM`: re-estimates pi, the transition OPERATOR (any
    structure -- dense/low-rank/combinator), and each state's emission from the Baum-Welch statistics.
    ``keys=(init_key, trans_key)`` tie the initial / transition parameters across HMMs that share them."""

    def __init__(
        self, emission_estimators, transition_proto, keys=(None, None), name=None, len_dist=None, terminal_states=None
    ):
        self.emission_estimators = list(emission_estimators)
        self.transition_proto = transition_proto
        self.keys = tuple(keys)
        self.name = name
        self.len_dist = len_dist  # carried (not fit) so fitted models retain it for enumeration
        self.terminal_states = terminal_states

    def accumulator_factory(self):
        """Return the accumulator factory used by this estimator."""
        return StructuredHMMAccumulatorFactory(self.emission_estimators, self.transition_proto, self.keys)

    def estimate(self, nobs, suff_stat):
        """Estimate an HMM from Baum-Welch sufficient statistics."""
        pi_acc, trans_acc, emit_vals, nk = _validated_structured_hmm_statistics(
            suff_stat,
            len(self.emission_estimators),
            label="structured-HMM sufficient statistics",
        )
        validate_effective_sample_mass(nobs, pi_acc.sum(), label="structured-HMM effective sample")
        pi = pi_acc / pi_acc.sum() if pi_acc.sum() > 0 else np.ones(len(pi_acc)) / len(pi_acc)
        transition = self.transition_proto.estimate(trans_acc)
        emissions = [self.emission_estimators[k].estimate(float(nk[k]), emit_vals[k]) for k in range(len(emit_vals))]
        return StructuredHMM(
            emissions,
            pi,
            transition,
            self.emission_estimators,
            self.keys,
            self.name,
            self.len_dist,
            self.terminal_states,
        )


# --- make StructuredHMM satisfy the distribution side of the contract -------------------------------
def _structured_hmm_log_density(self, x):
    return self._forward_backward(self._log_b(x))[5]


def _structured_hmm_dist_to_encoder(self):
    return StructuredHMMDataEncoder()


def _structured_hmm_estimator(self, pseudo_count=None):
    return StructuredHMMEstimator(
        self._emit_est, self.transition, self.keys, self.name, self.len_dist, self.terminal_states
    )


StructuredHMM.log_density = _structured_hmm_log_density
StructuredHMM.dist_to_encoder = _structured_hmm_dist_to_encoder
StructuredHMM.estimator = _structured_hmm_estimator
SequenceEncodableProbabilityDistribution.register(StructuredHMM)


def stationary_initial(op: TransitionOperator, *, iters: int = 2000, tol: float = 1e-13) -> np.ndarray:
    """The transition's stationary distribution (pi @ A == pi), by power iteration through ``op.forward``
    -- so it is O(K r) for a low-rank op, never forming A. Use it to COUPLE a StructuredHMM's initial
    state to its transition (``pi = stationary_initial(transition)``): the chain starts in its long-run
    distribution instead of a free, separately-estimated pi. Answers "do the initial states match the
    transition?" -- they can, by construction."""
    pi = np.ones(op.n_states) / op.n_states
    for _ in range(int(iters)):
        nxt = np.maximum(op.forward(pi), 0.0)
        s = nxt.sum()
        nxt = nxt / s if s > 0 else pi
        if np.max(np.abs(nxt - pi)) < tol:
            return nxt
        pi = nxt
    return pi


class InputOutputHMM:
    """Input-output HMM (IOHMM): an exogenous discrete input ``u_t`` selects which transition governs each
    step. Holds one :class:`TransitionOperator` per input symbol; the emission is per-state. Data is
    ``(obs_seq, input_seq)`` pairs where ``input_seq[t]`` in {0..M-1} drives the transition from t to t+1.

    Lets a covariate steer the dynamics -- regime switching driven by an observed control, the difference
    between a plain HMM and a controlled Markov model. (Input-dependent emissions are a natural extension;
    here emissions depend on state only.)
    """

    def __init__(self, emissions, pi, transitions, emission_estimators=None, name=None, terminal_states=None) -> None:
        try:
            self.emissions = list(emissions)
        except TypeError as exc:
            raise TypeError("InputOutputHMM emissions must be an iterable.") from exc
        if not self.emissions:
            raise ValueError("InputOutputHMM requires at least one emission distribution.")
        self.K = len(self.emissions)
        self.pi = validated_probability_vector(pi, "InputOutputHMM initial probabilities", size=self.K)
        try:
            raw_transitions = list(transitions)
        except TypeError as exc:
            raise TypeError("InputOutputHMM transitions must be an iterable of transition operators.") from exc
        if not raw_transitions:
            raise ValueError("InputOutputHMM requires at least one input transition.")
        self.transitions = [
            _validated_transition_operator(transition, self.K, f"InputOutputHMM transition {index}")
            for index, transition in enumerate(raw_transitions)
        ]
        self.M = len(self.transitions)
        self.name = name
        self.terminal_states = validated_terminal_states(
            terminal_states,
            self.K,
            context="InputOutputHMM",
        )
        self.term_mask = None
        if self.terminal_states is not None:
            self.term_mask = np.zeros(self.K, dtype=bool)
            self.term_mask[list(self.terminal_states)] = True
        for index, transition in enumerate(self.transitions):
            validate_terminal_reachability(
                self.pi,
                transition.as_matrix(),
                self.terminal_states,
                context="InputOutputHMM transition %d" % index,
            )
        if emission_estimators is None:
            self._emit_est = [emission.estimator() for emission in self.emissions]
        else:
            self._emit_est = list(emission_estimators)
            if len(self._emit_est) != self.K:
                raise ValueError(f"InputOutputHMM requires exactly {self.K} emission estimators.")

    def _log_b(self, seq):
        return np.array([[float(e.log_density(x)) for e in self.emissions] for x in seq])

    def _forward_backward(self, log_b, inputs):
        t_len, _ = log_b.shape
        term = self.term_mask
        nonterm = None if term is None else ~term
        mx = log_b.max(axis=1, keepdims=True)
        b = np.exp(log_b - np.where(np.isfinite(mx), mx, 0.0))  # all-impossible row -> b=0, not NaN
        alpha = np.zeros((t_len, self.K))
        c = np.zeros(t_len)
        alpha[0] = self.pi * b[0]
        c[0] = alpha[0].sum()
        alpha[0] = alpha[0] / c[0] if c[0] > 0 else alpha[0]
        for t in range(1, t_len):
            prev = alpha[t - 1] if nonterm is None else np.where(nonterm, alpha[t - 1], 0.0)
            alpha[t] = self.transitions[inputs[t - 1]].forward(prev) * b[t]
            c[t] = alpha[t].sum()
            alpha[t] = alpha[t] / c[t] if c[t] > 0 else alpha[t]
        if term is None:
            with np.errstate(divide="ignore"):  # impossible observation: c=0 -> -inf, not NaN
                loglik = float(np.sum(np.log(c)) + np.sum(mx))
        else:
            tm = float(alpha[t_len - 1][term].sum())
            with np.errstate(divide="ignore"):
                loglik = float(np.sum(np.log(c)) + np.sum(mx) + np.log(tm))
        beta = np.zeros((t_len, self.K))
        beta[t_len - 1] = 1.0 if term is None else np.where(term, 1.0, 0.0)
        for t in range(t_len - 2, -1, -1):
            back = self.transitions[inputs[t]].backward(b[t + 1] * beta[t + 1])
            scaled = back if nonterm is None else np.where(nonterm, back, 0.0)
            beta[t] = scaled / c[t + 1] if c[t + 1] > 0 else scaled
        gamma = alpha * beta
        gs = gamma.sum(axis=1, keepdims=True)
        gamma = np.divide(gamma, gs, out=np.zeros_like(gamma), where=gs > 0)
        if nonterm is not None:
            alpha = alpha * nonterm[None, :]  # mask outgoing transition mass from terminal states
        return alpha, beta, c, b, gamma, loglik

    def seq_log_density(self, x, input_seqs=None):
        """Per-sequence forward log-likelihood. Two call forms:
        - ``seq_log_density(obs_seqs, input_seqs)`` -- the explicit two-list API; or
        - ``seq_log_density(records)`` -- one list of ``(obs, input)``-pair sequences (the 5-part contract)."""
        if input_seqs is None:
            records = list(x)
            split = [
                _validated_io_record(record, self.M, f"InputOutputHMM record {index}")
                for index, record in enumerate(records)
            ]
        else:
            observations = _validated_sequences(x, "InputOutputHMM observation data")
            try:
                raw_inputs = list(input_seqs)
            except TypeError as exc:
                raise TypeError("InputOutputHMM input data must be an iterable of input sequences.") from exc
            if len(raw_inputs) != len(observations):
                raise ValueError(
                    "InputOutputHMM observation and input batches must contain the same number of sequences."
                )
            split = []
            for index, (obs, values) in enumerate(zip(observations, raw_inputs)):
                inputs = _validated_input_symbols(values, self.M, f"InputOutputHMM inputs row {index}")
                if len(inputs) != len(obs):
                    raise ValueError(f"InputOutputHMM inputs row {index} must contain exactly {len(obs)} symbols.")
                split.append((obs, inputs))
        return np.array([self._forward_backward(self._log_b(obs), inputs)[5] for obs, inputs in split])

    def _obs_inputs(self, seq, inputs):
        """Split one record into (observations, inputs), accepting both call forms the scoring API uses:
        ``(record)`` -- one sequence of ``(observation, input)`` pairs -- or ``(obs_seq, input_seq)``."""
        if inputs is None:
            return _validated_io_record(seq, self.M, "InputOutputHMM record")
        observations = _validated_sequences([seq], "InputOutputHMM observation data")[0]
        symbols = _validated_input_symbols(inputs, self.M, "InputOutputHMM inputs")
        if len(symbols) != len(observations):
            raise ValueError(f"InputOutputHMM inputs must contain exactly {len(observations)} symbols.")
        return observations, symbols

    def viterbi(self, seq, inputs=None):
        """Most-likely state path (Viterbi / max-product), conditioned on the input/control sequence:
        step t -> t+1 maximizes over the transition ``inputs[t]`` selects. Call as ``viterbi(record)``
        on one ``(observation, input)``-pair sequence or as ``viterbi(obs_seq, input_seq)``. Uses the
        per-input transition matrices, so it works for any operator; O(T K^2) -- a read-out, not the
        EM hot loop."""
        obs, u = self._obs_inputs(seq, inputs)
        log_b = self._log_b(obs)
        require_possible_log_evidence(
            self._forward_backward(log_b, u)[5],
            context="InputOutputHMM.viterbi",
        )
        log_as = [_log_probabilities(t.as_matrix()) for t in self.transitions]
        log_pi = _log_probabilities(self.pi)
        t_len, k = log_b.shape
        delta = np.zeros((t_len, k))
        psi = np.zeros((t_len, k), dtype=int)
        delta[0] = log_pi + log_b[0]
        for t in range(1, t_len):
            previous = delta[t - 1]
            if self.term_mask is not None:
                previous = np.where(~self.term_mask, previous, -np.inf)
            m = previous[:, None] + log_as[u[t - 1]]  # (from, to) under the input driving this step
            psi[t] = np.argmax(m, axis=0)
            delta[t] = m[psi[t], np.arange(k)] + log_b[t]
        path = np.zeros(t_len, dtype=int)
        ending = delta[-1]
        if self.term_mask is not None:
            ending = np.where(self.term_mask, ending, -np.inf)
        path[-1] = int(np.argmax(ending))
        for t in range(t_len - 2, -1, -1):
            path[t] = psi[t + 1, path[t + 1]]
        return path

    def posterior_decode(self, seq, inputs=None):
        """Per-position MAP state argmax_k P(z_t = k | x, u) from the forward-backward posteriors gamma."""
        return np.argmax(self.state_posteriors(seq, inputs), axis=1)

    def state_posteriors(self, seq, inputs=None):
        """The full smoothing posteriors gamma[t,k] = P(z_t = k | x, u), conditioned on the inputs."""
        obs, u = self._obs_inputs(seq, inputs)
        result = self._forward_backward(self._log_b(obs), u)
        require_possible_log_evidence(result[5], context="InputOutputHMM.state_posteriors")
        return result[4]

    def sampler(self, seed=None):
        """Return a sampler for ``(observation, input)``-pair records along a given control sequence."""
        return _IOHMMSampler(self, seed)

    def fit(self, obs_seqs, input_seqs, *, max_its: int = 50, tol: float = 1e-6, weights=None):
        """Fit the IOHMM by weighted Baum-Welch and return a diagnostics-bearing result."""
        max_its, tol = _validated_fit_controls(max_its, tol)
        obs_seqs = _validated_sequences(obs_seqs, "InputOutputHMM fit observations")
        try:
            raw_inputs = list(input_seqs)
        except TypeError as exc:
            raise TypeError("InputOutputHMM fit inputs must be an iterable of input sequences.") from exc
        if len(raw_inputs) != len(obs_seqs):
            raise ValueError("InputOutputHMM fit observations and inputs must contain the same number of sequences.")
        input_seqs = []
        for index, (observations, values) in enumerate(zip(obs_seqs, raw_inputs)):
            symbols = _validated_input_symbols(values, self.M, f"InputOutputHMM fit inputs row {index}")
            if len(symbols) != len(observations):
                raise ValueError(
                    f"InputOutputHMM fit inputs row {index} must contain exactly {len(observations)} symbols."
                )
            input_seqs.append(symbols)
        weights = _validated_weights(weights, len(obs_seqs))
        active_indices = np.flatnonzero(weights > 0.0)
        ll_trace: list[float] = []
        iterations = 0
        converged = False
        termination_reason = "max_iterations"
        rollback = None

        for _ in range(max_its):
            active_obs = [obs_seqs[index] for index in active_indices]
            active_inputs = [input_seqs[index] for index in active_indices]
            try:
                total_ll = _weighted_fit_log_likelihood(
                    self.seq_log_density(active_obs, active_inputs),
                    weights[active_indices],
                    "InputOutputHMM.fit",
                )
            except (ImpossibleEvidenceError, ValueError, RuntimeError, FloatingPointError):
                if rollback is None:
                    raise
                self.pi, self.transitions, self.emissions = rollback
                iterations -= 1
                termination_reason = "invalid_update_rejected"
                break
            if ll_trace:
                absolute, _ = _fit_delta(ll_trace[-1], total_ll)
                allowance = 1.0e-8 * max(1.0, abs(ll_trace[-1]))
                if absolute < -allowance:
                    if rollback is not None:
                        self.pi, self.transitions, self.emissions = rollback
                        iterations -= 1
                    termination_reason = "non_monotone_update_rejected"
                    break
                ll_trace.append(total_ll)
                if abs(absolute) <= tol * max(1.0, abs(ll_trace[-2])):
                    converged = True
                    termination_reason = "converged"
                    break
            else:
                ll_trace.append(total_ll)

            trans_accs = [t.new_accumulator() for t in self.transitions]
            pi_acc = np.zeros(self.K)
            emit_accs = [est.accumulator_factory().make() for est in self._emit_est]
            nk = np.zeros(self.K)
            for index in active_indices:
                o = obs_seqs[index]
                u = input_seqs[index]
                weight = weights[index]
                log_b = self._log_b(o)
                alpha, beta, c, b, gamma, _ = self._forward_backward(log_b, u)
                pi_acc += weight * gamma[0]
                for t in range(len(o) - 1):
                    m = u[t]
                    self.transitions[m].accumulate(
                        trans_accs[m],
                        alpha[t],
                        weight * b[t + 1] * beta[t + 1],
                        c[t + 1],
                    )
                for k in range(self.K):
                    enc = self.emissions[k].dist_to_encoder().seq_encode(o)
                    state_weights = weight * gamma[:, k]
                    emit_accs[k].seq_update(enc, state_weights, self.emissions[k])
                    nk[k] += state_weights.sum()
            rollback = (self.pi.copy(), list(self.transitions), list(self.emissions))
            self.transitions = [self.transitions[m].estimate(trans_accs[m]) for m in range(self.M)]
            self.pi = pi_acc / pi_acc.sum()
            self.emissions = [self._emit_est[k].estimate(float(nk[k]), emit_accs[k].value()) for k in range(self.K)]
            iterations += 1

        if iterations and termination_reason == "max_iterations":
            try:
                final_ll = _weighted_fit_log_likelihood(
                    self.seq_log_density(
                        [obs_seqs[index] for index in active_indices],
                        [input_seqs[index] for index in active_indices],
                    ),
                    weights[active_indices],
                    "InputOutputHMM.fit final model",
                )
            except (ImpossibleEvidenceError, ValueError, RuntimeError, FloatingPointError):
                self.pi, self.transitions, self.emissions = rollback
                iterations -= 1
                termination_reason = "invalid_update_rejected"
            else:
                absolute, _ = _fit_delta(ll_trace[-1], final_ll)
                allowance = 1.0e-8 * max(1.0, abs(ll_trace[-1]))
                if absolute < -allowance:
                    self.pi, self.transitions, self.emissions = rollback
                    iterations -= 1
                    termination_reason = "non_monotone_update_rejected"
                else:
                    ll_trace.append(final_ll)
                    if abs(absolute) <= tol * max(1.0, abs(ll_trace[-2])):
                        converged = True
                        termination_reason = "converged"

        diagnostics = _fit_receipt(
            algorithm="input-output-baum-welch",
            trace=ll_trace,
            converged=converged,
            iterations=iterations,
            termination_reason=termination_reason,
            n_sequences=len(active_indices),
            total_weight=float(weights.sum()),
            approximate=False,
        )
        return HMMFitResult(self, ll_trace, diagnostics)

    # --- 5-part contract: a record is one (obs, input) sequence = a list of (observation, input) pairs ---
    def log_density(self, seq):
        """Return the log likelihood of one ``(observation, input)`` sequence."""
        obs, inputs = _validated_io_record(seq, self.M, "InputOutputHMM record")
        return self._forward_backward(self._log_b(obs), inputs)[5]

    def dist_to_encoder(self):
        """Return the pass-through IOHMM sequence encoder."""
        return IOHMMDataEncoder()

    def estimator(self, pseudo_count=None):
        """Return the estimator for this IOHMM structure."""
        return IOHMMEstimator(self._emit_est, list(self.transitions), self.name)

    def to_dict(self) -> dict[str, Any]:
        """Return a safe JSON-compatible representation of this IOHMM (decodes via ``load_models``)."""
        from mixle.utils.serialization import to_serializable

        return to_serializable(self)


class _IOHMMSampler:
    """Samples one IOHMM record -- a list of ``(observation, input)`` pairs -- along a given control
    sequence. The inputs are exogenous, so the caller supplies them; the sampler draws the state path
    through the per-input transitions and one emission per state visited."""

    def __init__(self, hmm: InputOutputHMM, seed=None):
        self.hmm = hmm
        self.rng = np.random.RandomState(seed)

    def sample(self, inputs):
        h = self.hmm
        u = _validated_input_symbols(inputs, h.M, "InputOutputHMM sampler inputs")
        if not u:
            raise ValueError("InputOutputHMM sampler inputs must not be empty.")
        mats = [t.as_matrix() for t in h.transitions]
        s = self.rng.choice(h.K, p=h.pi)
        out = []
        terminated = False
        for t, m in enumerate(u):
            out.append((h.emissions[s].sampler(seed=int(self.rng.randint(1, 2**31))).sample(), m))
            if h.term_mask is not None and h.term_mask[s]:
                terminated = True
                break  # terminal (absorbing) state ends the sequence -- length is the stopping time
            if t + 1 < len(u):
                row = mats[m][s]
                rs = row.sum()
                s = self.rng.choice(h.K, p=row / rs) if rs > 0 else s
        if h.term_mask is not None:
            require_terminal_reached(
                terminated,
                mode="input-output terminal-state",
                max_steps=len(u),
                last_state=int(s),
            )
        return out


class IOHMMDataEncoder(DataSequenceEncoder):
    """An IOHMM record is one ``(obs, input)`` sequence -- a list of ``(observation, input_symbol)`` pairs."""

    def seq_encode(self, x):
        """Encode IOHMM records as lists of ``(observation, input)`` pairs."""
        return [list(s) for s in x]

    def row_count(self, x):
        """Return the number of pass-through input-output HMM records."""
        if not isinstance(x, list):
            raise ValueError("input-output HMM encoding must be a list of sequence records")
        return len(x)

    def __eq__(self, other):
        return isinstance(other, IOHMMDataEncoder)

    def __hash__(self):
        return hash("IOHMMDataEncoder")


class IOHMMAccumulator(SequenceEncodableStatisticAccumulator):
    """Accumulator for IOHMM Baum-Welch sufficient statistics."""

    def __init__(self, emission_accumulators, transition_protos):
        self.emit = list(emission_accumulators)
        self.K = len(self.emit)
        self.transition_protos = list(transition_protos)
        self.M = len(self.transition_protos)
        self.pi_acc = np.zeros(self.K)
        self.trans_accs = [t.new_accumulator() for t in self.transition_protos]
        self.nk = np.zeros(self.K)

    def update(self, x, weight, estimate):
        """Accumulate sufficient statistics from one weighted IOHMM record."""
        weight = validated_observation_weight(weight, "IOHMM observation weight")
        self.seq_update([x], np.array([weight], dtype=float), estimate)

    def seq_update(self, x, weights, estimate):
        """Accumulate weighted sufficient statistics from a batch of IOHMM records."""
        records = list(x)
        weights = _validated_weights(
            weights,
            len(records),
            "IOHMM accumulator weights",
            require_positive_total=False,
        )
        active_indices = np.flatnonzero(weights > 0.0)
        active = [records[index] for index in active_indices]
        if not active:
            return
        require_possible_log_evidence(
            estimate.seq_log_density(active),
            context="IOHMMAccumulator.seq_update",
        )
        for index in active_indices:
            seq = records[index]
            w = weights[index]
            obs, inputs = _validated_io_record(seq, self.M, f"IOHMM accumulator record {index}")
            log_b = estimate._log_b(obs)
            alpha, beta, c, b, gamma, _ = estimate._forward_backward(log_b, inputs)
            self.pi_acc += w * gamma[0]
            for t in range(len(seq) - 1):
                m = inputs[t]
                estimate.transitions[m].accumulate(self.trans_accs[m], alpha[t], b[t + 1] * beta[t + 1] * w, c[t + 1])
            for k in range(self.K):
                enc = estimate.emissions[k].dist_to_encoder().seq_encode(obs)
                wk = gamma[:, k] * w
                self.emit[k].seq_update(enc, wk, estimate.emissions[k])
                self.nk[k] += float(wk.sum())

    def seq_initialize(self, x, weights, rng):
        """Initialize IOHMM sufficient statistics with random soft responsibilities."""
        records = list(x)
        weights = _validated_weights(
            weights,
            len(records),
            "IOHMM initializer weights",
            require_positive_total=False,
        )
        active_indices = np.flatnonzero(weights > 0.0)
        if not len(active_indices):
            return
        for m in range(self.M):
            self.trans_accs[m] = _add_nested(self.trans_accs[m], self.transition_protos[m].random_accumulator(rng))
        for index in active_indices:
            seq = records[index]
            w = weights[index]
            obs, _ = _validated_io_record(seq, self.M, f"IOHMM initializer record {index}")
            g = rng.dirichlet(np.ones(self.K), len(seq))
            self.pi_acc += w * g[0]
            for k in range(self.K):
                enc = self.emit[k].acc_to_encoder().seq_encode(obs)
                wk = g[:, k] * w
                self.emit[k].seq_initialize(enc, wk, rng)
                self.nk[k] += float(wk.sum())

    def combine(self, suff_stat):
        """Merge serialized IOHMM sufficient statistics."""
        pi_acc, trans_accs, emit_vals, nk = _validated_structured_hmm_statistics(
            suff_stat,
            self.K,
            label="IOHMM sufficient statistics",
            transition_count=self.M,
        )
        # Transactional with a finiteness postcondition (measured on the family; STAT-RR8-1/
        # RR9-1 classes).
        _snapshot = snapshot_accumulator_statistics(self, count_attrs=("pi_acc", "nk"), child_attrs=("emit",))
        # _add_nested/_scale_nested rebind (they allocate); restoring the original REFERENCE
        # preserves both values and alias-observed identity (STAT-RR11-3)
        _previous_trans = self.trans_accs
        self.pi_acc += pi_acc
        self.trans_accs = [_add_nested(a, b) for a, b in zip(self.trans_accs, trans_accs)]
        self.nk += nk
        try:
            require_finite_count_totals(
                (("initial counts", self.pi_acc), ("emission counts", self.nk)),
                label="combined IOHMM",
            )
            _require_finite_nested(self.trans_accs, name="transition counts", label="combined IOHMM")
            for k in range(self.K):
                self.emit[k].combine(emit_vals[k])
        except Exception:
            restore_accumulator_statistics(self, _snapshot)
            self.trans_accs = _previous_trans
            raise
        return self

    def value(self):
        """Return serialized IOHMM sufficient statistics."""
        return (
            self.pi_acc.copy(),
            _copy_nested(self.trans_accs),
            [e.value() for e in self.emit],
            self.nk.copy(),
        )

    def from_value(self, x):
        """Restore accumulator state from serialized IOHMM statistics."""
        # Candidates validated before ANY assignment; children restore transactionally
        # (measured on the family; STAT-RR9-1 class).
        candidate_pi, candidate_trans, emit_vals, candidate_nk = _validated_structured_hmm_statistics(
            x,
            self.K,
            label="IOHMM sufficient statistics",
            transition_count=self.M,
        )
        _snapshot = snapshot_accumulator_statistics(self, count_attrs=("pi_acc", "nk"), child_attrs=("emit",))
        # _add_nested/_scale_nested rebind (they allocate); restoring the original REFERENCE
        # preserves both values and alias-observed identity (STAT-RR11-3)
        _previous_trans = self.trans_accs
        self.pi_acc, self.trans_accs, self.nk = candidate_pi, candidate_trans, candidate_nk
        try:
            for k in range(self.K):
                self.emit[k].from_value(emit_vals[k])
        except Exception:
            restore_accumulator_statistics(self, _snapshot)
            self.trans_accs = _previous_trans
            raise
        return self

    def scale(self, factor):
        """Scale IOHMM transition, emission, and initial-state statistics."""
        factor = validated_observation_weight(factor, "IOHMM scale factor")
        # One transaction with a scaled-result postcondition (measured on the family;
        # STAT-RR8-1/RR10-1 classes).
        _snapshot = snapshot_accumulator_statistics(self, count_attrs=("pi_acc", "nk"), child_attrs=("emit",))
        # _add_nested/_scale_nested rebind (they allocate); restoring the original REFERENCE
        # preserves both values and alias-observed identity (STAT-RR11-3)
        _previous_trans = self.trans_accs
        self.pi_acc *= factor
        self.trans_accs = _scale_nested(self.trans_accs, factor)
        self.nk *= factor
        try:
            require_finite_count_totals(
                (("initial counts", self.pi_acc), ("emission counts", self.nk)),
                label="scaled IOHMM",
            )
            _require_finite_nested(self.trans_accs, name="transition counts", label="scaled IOHMM")
            for accumulator in self.emit:
                accumulator.scale(factor)
        except Exception:
            restore_accumulator_statistics(self, _snapshot)
            self.trans_accs = _previous_trans
            raise
        return self

    def acc_to_encoder(self):
        """Return the encoder associated with this accumulator."""
        return IOHMMDataEncoder()


class IOHMMAccumulatorFactory(StatisticAccumulatorFactory):
    """Factory for IOHMM accumulators."""

    def __init__(self, emission_estimators, transition_protos):
        self.emission_estimators = emission_estimators
        self.transition_protos = transition_protos

    def make(self):
        """Create a fresh IOHMM accumulator."""
        emit = [est.accumulator_factory().make() for est in self.emission_estimators]
        return IOHMMAccumulator(emit, self.transition_protos)


class IOHMMEstimator(ParameterEstimator):
    """Estimator (M-step) for an :class:`InputOutputHMM`: re-estimates pi, one transition operator per input
    symbol (from the per-input expected counts), and each state's emission."""

    def __init__(self, emission_estimators, transition_protos, name=None):
        self.emission_estimators = list(emission_estimators)
        self.transition_protos = list(transition_protos)
        self.name = name

    def accumulator_factory(self):
        """Return the accumulator factory used by this estimator."""
        return IOHMMAccumulatorFactory(self.emission_estimators, self.transition_protos)

    def estimate(self, nobs, suff_stat):
        """Estimate an IOHMM from Baum-Welch sufficient statistics."""
        pi_acc, trans_accs, emit_vals, nk = _validated_structured_hmm_statistics(
            suff_stat,
            len(self.emission_estimators),
            label="IOHMM sufficient statistics",
            transition_count=len(self.transition_protos),
        )
        validate_effective_sample_mass(nobs, pi_acc.sum(), label="IOHMM effective sample")
        pi = pi_acc / pi_acc.sum() if pi_acc.sum() > 0 else np.ones(len(pi_acc)) / len(pi_acc)
        transitions = [self.transition_protos[m].estimate(trans_accs[m]) for m in range(len(trans_accs))]
        emissions = [self.emission_estimators[k].estimate(float(nk[k]), emit_vals[k]) for k in range(len(emit_vals))]
        return InputOutputHMM(emissions, pi, transitions, self.emission_estimators, self.name)


SequenceEncodableProbabilityDistribution.register(InputOutputHMM)


class ExplicitDurationHMM:
    """Hidden semi-Markov model (explicit-duration HMM): each state emits for a random *duration* drawn
    from a per-state duration distribution, then switches state (the transition matrix has a zero diagonal
    -- dwell time is modeled explicitly, not as a self-loop). This captures non-geometric state durations a
    plain HMM cannot.

    ``durations`` is one length-``max_duration`` probability vector per state (over d = 1..max_duration).
    A sequence has an exogenous fixed observation horizon. Its final latent duration is right-censored when
    it extends beyond that horizon, exactly matching ``sample(length)``. Forward/EM are O(T * K *
    max_duration).
    """

    def __init__(self, emissions, pi, transition_matrix, durations, max_duration, name=None) -> None:
        try:
            self.emissions = list(emissions)
        except TypeError as exc:
            raise TypeError("ExplicitDurationHMM emissions must be an iterable.") from exc
        if len(self.emissions) < 2:
            raise ValueError("ExplicitDurationHMM requires at least two states because segments must switch state.")
        self.K = len(self.emissions)
        self.pi = validated_probability_vector(pi, "ExplicitDurationHMM initial probabilities", size=self.K)
        self.a = validated_row_probability_matrix(
            transition_matrix,
            "ExplicitDurationHMM transition matrix",
            shape=(self.K, self.K),
        )
        if np.any(np.diag(self.a) != 0.0):
            raise ValueError("ExplicitDurationHMM transition diagonal must be exactly zero.")
        self.D = _exact_positive_integer(max_duration, "ExplicitDurationHMM max_duration")
        self.dur = validated_row_probability_matrix(
            durations,
            "ExplicitDurationHMM duration probabilities",
            shape=(self.K, self.D),
        )
        self.name = name
        self._emit_est = [e.estimator() for e in self.emissions]

    def _log_b(self, seq):
        return np.array([[float(e.log_density(x)) for e in self.emissions] for x in seq])

    def _seg_loglik(self, log_b):
        """seg[t, d, j] = log P(obs_{t-d+1 .. t} | state j) for a length-(d+1) segment ENDING at t."""
        t_len = log_b.shape[0]
        seg = np.full((t_len, self.D, self.K), -np.inf)
        if t_len:
            seg[:, 0, :] = log_b
        for duration_index in range(1, self.D):
            for end in range(duration_index, t_len):
                seg[end, duration_index] = seg[end - 1, duration_index - 1] + log_b[end]
        return seg

    def _log_duration_survival(self):
        """Return log P(duration >= d) for duration indices d=0..D-1."""
        survival = np.flip(np.cumsum(np.flip(self.dur, axis=1), axis=1), axis=1)
        return _log_probabilities(survival)

    def _forward(self, log_b):
        t_len = log_b.shape[0]
        seg = self._seg_loglik(log_b)
        log_dur = _log_probabilities(self.dur)
        log_a = _log_probabilities(self.a)
        log_pi = _log_probabilities(self.pi)
        log_alpha = np.full((t_len, self.K), -np.inf)  # segment ends at t in j
        log_e = np.full((t_len + 1, self.K), -np.inf)  # entry into j at time tau (segment starts at tau)
        log_e[0] = log_pi
        for t in range(t_len):
            for j in range(self.K):
                terms = [
                    log_e[t - d, j] + log_dur[j, d] + seg[t, d, j]
                    for d in range(min(t + 1, self.D))
                    if np.isfinite(log_e[t - d, j])
                ]
                if terms:
                    log_alpha[t, j] = _logsumexp(terms)
            for j in range(self.K):
                log_e[t + 1, j] = _logsumexp(log_alpha[t] + log_a[:, j])
        return log_alpha, log_e, seg

    def forward_loglik(self, seq):
        """Fixed-horizon log likelihood, marginalizing a possibly right-censored final duration."""
        sequence = _validated_sequences([seq], "ExplicitDurationHMM scoring data")[0]
        _, log_e, seg = self._forward(self._log_b(sequence))
        t_len = len(sequence)
        log_survival = self._log_duration_survival()
        final_terms = []
        for observed_index in range(min(t_len, self.D)):
            start = t_len - observed_index - 1
            final_terms.extend(log_e[start] + log_survival[:, observed_index] + seg[-1, observed_index])
        return float(_logsumexp(final_terms))

    def _backward(self, log_b, seg):
        t_len = log_b.shape[0]
        log_dur = _log_probabilities(self.dur)
        log_survival = self._log_duration_survival()
        log_a = _log_probabilities(self.a)
        log_beta = np.full((t_len, self.K), -np.inf)  # P(obs_{t+1:} | segment ends at t in j)
        log_bstar = np.full((t_len, self.K), -np.inf)  # P(obs_{tau:} | segment starts at tau in j)
        for tau in range(t_len - 1, -1, -1):
            remaining = t_len - tau
            for j in range(self.K):
                terms = []
                if remaining <= self.D:
                    terms.append(log_survival[j, remaining - 1] + seg[t_len - 1, remaining - 1, j])
                for duration in range(1, min(self.D, remaining - 1) + 1):
                    end = tau + duration - 1
                    terms.append(log_dur[j, duration - 1] + seg[end, duration - 1, j] + log_beta[end, j])
                log_bstar[tau, j] = _logsumexp(terms) if terms else -np.inf
            if tau > 0:
                for j in range(self.K):
                    log_beta[tau - 1, j] = _logsumexp(log_a[j, :] + log_bstar[tau, :])
        return log_beta, log_bstar

    def _estep(self, seq):
        """Per-sequence E-step. Returns (loglik, pi_contrib (K,), trans_contrib (K,K), dur_contrib (K,D),
        occ (T,K)). The last segment's latent duration is marginalized over every duration at least as
        long as its observed right-censored portion."""
        log_b = self._log_b(seq)
        t_len = len(seq)
        log_dur = _log_probabilities(self.dur)
        log_a = _log_probabilities(self.a)
        log_pi = _log_probabilities(self.pi)
        log_alpha, log_e, seg = self._forward(log_b)
        log_beta, log_bstar = self._backward(log_b, seg)
        z = _logsumexp(log_pi + log_bstar[0])
        require_possible_log_evidence(z, context="ExplicitDurationHMM._estep")
        pi_contrib = np.exp(log_pi + log_bstar[0] - z)
        dur_contrib = np.zeros((self.K, self.D))
        trans_contrib = np.zeros((self.K, self.K))
        occ = np.zeros((t_len, self.K))

        # Completed, non-final segments have an exact latent duration and transition onward.
        for t in range(t_len - 1):
            for j in range(self.K):
                for d in range(min(t + 1, self.D)):
                    lp = log_e[t - d, j] + log_dur[j, d] + seg[t, d, j] + log_beta[t, j] - z
                    if np.isfinite(lp):
                        p = np.exp(lp)
                        dur_contrib[j, d] += p
                        occ[t - d : t + 1, j] += p
            for i in range(self.K):
                trans_contrib[i] += np.exp(log_alpha[t, i] + log_a[i, :] + log_bstar[t + 1, :] - z)

        # The final segment is observed for r positions but its actual duration may be any q >= r.
        for observed_index in range(min(t_len, self.D)):
            start = t_len - observed_index - 1
            for j in range(self.K):
                emission_loglik = seg[t_len - 1, observed_index, j]
                for duration_index in range(observed_index, self.D):
                    lp = log_e[start, j] + log_dur[j, duration_index] + emission_loglik - z
                    if np.isfinite(lp):
                        probability = np.exp(lp)
                        dur_contrib[j, duration_index] += probability
                        occ[start:t_len, j] += probability
        return float(z), pi_contrib, trans_contrib, dur_contrib, occ

    def fit(self, seqs, *, max_its: int = 50, tol: float = 1e-6, weights=None):
        """Fit the explicit-duration HMM by weighted Baum-Welch with a diagnostics receipt."""
        max_its, tol = _validated_fit_controls(max_its, tol)
        seqs = _validated_sequences(seqs, "ExplicitDurationHMM fit data")
        weights = _validated_weights(weights, len(seqs))
        active_indices = np.flatnonzero(weights > 0.0)
        ll_trace: list[float] = []
        iterations = 0
        converged = False
        termination_reason = "max_iterations"
        rollback = None

        for _ in range(max_its):
            active = [seqs[index] for index in active_indices]
            try:
                total_ll = _weighted_fit_log_likelihood(
                    self.seq_log_density(active),
                    weights[active_indices],
                    "ExplicitDurationHMM.fit",
                )
            except (ImpossibleEvidenceError, ValueError, RuntimeError, FloatingPointError):
                if rollback is None:
                    raise
                self.pi, self.a, self.dur, self.emissions = rollback
                iterations -= 1
                termination_reason = "invalid_update_rejected"
                break
            if ll_trace:
                absolute, _ = _fit_delta(ll_trace[-1], total_ll)
                allowance = 1.0e-8 * max(1.0, abs(ll_trace[-1]))
                if absolute < -allowance:
                    if rollback is not None:
                        self.pi, self.a, self.dur, self.emissions = rollback
                        iterations -= 1
                    termination_reason = "non_monotone_update_rejected"
                    break
                ll_trace.append(total_ll)
                if abs(absolute) <= tol * max(1.0, abs(ll_trace[-2])):
                    converged = True
                    termination_reason = "converged"
                    break
            else:
                ll_trace.append(total_ll)

            dur_acc = np.zeros((self.K, self.D))
            trans_acc = np.zeros((self.K, self.K))
            pi_acc = np.zeros(self.K)
            emit_accs = [est.accumulator_factory().make() for est in self._emit_est]
            nk = np.zeros(self.K)
            for index in active_indices:
                seq = seqs[index]
                weight = weights[index]
                _, pic, trc, drc, occ = self._estep(seq)
                pi_acc += weight * pic
                trans_acc += weight * trc
                dur_acc += weight * drc
                for k in range(self.K):
                    enc = self.emissions[k].dist_to_encoder().seq_encode(seq)
                    state_weights = weight * occ[:, k]
                    emit_accs[k].seq_update(enc, state_weights, self.emissions[k])
                    nk[k] += state_weights.sum()
            rollback = (self.pi.copy(), self.a.copy(), self.dur.copy(), list(self.emissions))
            self.pi = pi_acc / pi_acc.sum()
            np.fill_diagonal(trans_acc, 0.0)
            self.a = _row_normalize(trans_acc, self.a)
            self.dur = _row_normalize(dur_acc, self.dur)
            self.emissions = [self._emit_est[k].estimate(float(nk[k]), emit_accs[k].value()) for k in range(self.K)]
            iterations += 1

        if iterations and termination_reason == "max_iterations":
            try:
                final_ll = _weighted_fit_log_likelihood(
                    self.seq_log_density([seqs[index] for index in active_indices]),
                    weights[active_indices],
                    "ExplicitDurationHMM.fit final model",
                )
            except (ImpossibleEvidenceError, ValueError, RuntimeError, FloatingPointError):
                self.pi, self.a, self.dur, self.emissions = rollback
                iterations -= 1
                termination_reason = "invalid_update_rejected"
            else:
                absolute, _ = _fit_delta(ll_trace[-1], final_ll)
                allowance = 1.0e-8 * max(1.0, abs(ll_trace[-1]))
                if absolute < -allowance:
                    self.pi, self.a, self.dur, self.emissions = rollback
                    iterations -= 1
                    termination_reason = "non_monotone_update_rejected"
                else:
                    ll_trace.append(final_ll)
                    if abs(absolute) <= tol * max(1.0, abs(ll_trace[-2])):
                        converged = True
                        termination_reason = "converged"

        diagnostics = _fit_receipt(
            algorithm="explicit-duration-baum-welch",
            trace=ll_trace,
            converged=converged,
            iterations=iterations,
            termination_reason=termination_reason,
            n_sequences=len(active_indices),
            total_weight=float(weights.sum()),
            approximate=False,
        )
        return HMMFitResult(self, ll_trace, diagnostics)

    # --- 5-part contract: a record is one observation sequence ---
    def log_density(self, seq):
        """Return the explicit-duration HMM log likelihood for one sequence."""
        return self.forward_loglik(seq)

    def seq_log_density(self, x):
        """Score a batch of observation sequences."""
        return np.array([self.forward_loglik(list(seq)) for seq in x])

    def dist_to_encoder(self):
        """Return the pass-through explicit-duration sequence encoder."""
        return EDHMMDataEncoder()

    def estimator(self, pseudo_count=None):
        """Return the estimator for this explicit-duration HMM structure."""
        return EDHMMEstimator(self._emit_est, self.K, self.D, self.name)

    def to_structured_hmm(self, len_dist=None):
        """The HSMM as an EQUIVALENT StructuredHMM via the remaining-duration expansion: K*D sub-states
        (k, r) = "state k with r steps left in the segment". The expanded chain emits from state k at every
        sub-state, decrements deterministically (k,r)->(k,r-1), and at (k,1) switches segment with
        A[k,k']*dur[k'](d'). Reading the first T expanded-chain emissions exactly represents the
        right-censored fixed horizon, so no final-state constraint is applied. This hands the HSMM the full
        StructuredHMM read-out API and, with ``len_dist``, enumeration. O(K*D) states."""

        def idx(k, r):  # r in 1..D
            return k * self.D + (r - 1)

        n = self.K * self.D
        emissions = [self.emissions[k] for k in range(self.K) for _ in range(self.D)]
        pi = np.zeros(n)
        for k in range(self.K):
            for d in range(1, self.D + 1):
                pi[idx(k, d)] = self.pi[k] * self.dur[k, d - 1]
        a = np.zeros((n, n))
        for k in range(self.K):
            for r in range(2, self.D + 1):
                a[idx(k, r), idx(k, r - 1)] = 1.0  # decrement remaining duration
            for kp in range(self.K):
                if kp == k:
                    continue
                for dp in range(1, self.D + 1):
                    a[idx(k, 1), idx(kp, dp)] = self.a[k, kp] * self.dur[kp, dp - 1]  # switch segment
        return StructuredHMM(emissions, pi, DenseTransition(a), len_dist=len_dist)

    def enumerator(self, len_dist):
        """Enumerate fixed-horizon observation sequences in descending marginal probability, given a
        ``len_dist`` over total sequence length. Built on the exact right-censored HMM expansion;
        ``.top_k(k)`` -> [(sequence, log_prob), ...]."""
        return self.to_structured_hmm(len_dist=len_dist).enumerator()

    def state_posteriors(self, seq):
        """Per-position smoothing posteriors gamma[t, j] = P(z_t = j | obs), marginalizing the durations
        (sum the posterior of every segment that covers position t). Rows sum to 1."""
        sequence = _validated_sequences([seq], "ExplicitDurationHMM posterior data")[0]
        return self._estep(sequence)[4]

    def posterior_decode(self, seq):
        """Per-position MAP state argmax_j P(z_t = j | obs)."""
        return np.argmax(self.state_posteriors(seq), axis=1)

    def viterbi_segments(self, seq):
        """Most-likely segmentation (max-product over the segment lattice): a list of (state, start,
        observed_duration) segments covering the sequence, O(T K D). The last tuple describes the
        observed portion of a right-censored latent duration."""
        sequence = _validated_sequences([seq], "ExplicitDurationHMM Viterbi data")[0]
        require_possible_log_evidence(self.forward_loglik(sequence), context="ExplicitDurationHMM.viterbi_segments")
        log_b = self._log_b(sequence)
        t_len = len(sequence)
        seg = self._seg_loglik(log_b)
        log_dur = _log_probabilities(self.dur)
        log_a = _log_probabilities(self.a)
        log_pi = _log_probabilities(self.pi)
        delta = np.full((t_len, self.K), -np.inf)  # best score of a segment ending at t in j
        bp_d = np.zeros((t_len, self.K), dtype=int)  # chosen duration index
        entry = np.full((t_len, self.K), -np.inf)  # best score to START a segment at t in j
        bp_prev = np.full((t_len, self.K), -1, dtype=int)  # previous state at a segment boundary
        entry[0] = log_pi
        for t in range(t_len):
            for j in range(self.K):
                best, bd = -np.inf, 0
                for d in range(min(t + 1, self.D)):
                    e = entry[t - d, j]
                    if np.isfinite(e):
                        val = e + log_dur[j, d] + seg[t, d, j]
                        if val > best:
                            best, bd = val, d
                delta[t, j], bp_d[t, j] = best, bd
            if t + 1 < t_len:
                for j in range(self.K):
                    vals = delta[t] + log_a[:, j]
                    entry[t + 1, j], bp_prev[t + 1, j] = float(vals.max()), int(vals.argmax())

        best_score = -np.inf
        final_state = 0
        final_start = 0
        final_observed_duration = 1
        for observed_index in range(min(t_len, self.D)):
            start = t_len - observed_index - 1
            observed_duration = observed_index + 1
            for state in range(self.K):
                emission_loglik = seg[-1, observed_index, state]
                for duration_index in range(observed_index, self.D):
                    score = entry[start, state] + log_dur[state, duration_index] + emission_loglik
                    if score > best_score:
                        best_score = score
                        final_state = state
                        final_start = start
                        final_observed_duration = observed_duration

        segments = [(int(final_state), int(final_start), int(final_observed_duration))]
        start = final_start
        state = final_state
        while start > 0:
            state = int(bp_prev[start, state])
            end = start - 1
            duration_index = int(bp_d[end, state])
            start = end - duration_index
            segments.append((state, int(start), duration_index + 1))
        segments.reverse()
        return segments

    def sampler(self, seed=None):
        """Return a sampler for explicit-duration HMM sequences."""
        return _EDHMMSampler(self, seed)


def _logsumexp(v):
    v = np.asarray(v, dtype=float)
    if v.size == 0:
        return -np.inf
    m = v.max()
    if not np.isfinite(m):
        return -np.inf
    return float(m + np.log(np.sum(np.exp(v - m))))


class _EDHMMSampler:
    def __init__(self, hmm, seed=None):
        self.hmm = hmm
        self.rng = np.random.RandomState(seed)

    def sample(self, length):
        h = self.hmm
        horizon = _exact_positive_integer(length, "ExplicitDurationHMM sample length")
        out = []
        s = self.rng.choice(h.K, p=h.pi)
        while len(out) < horizon:
            d = self.rng.choice(h.D, p=h.dur[s]) + 1
            for _ in range(d):
                if len(out) >= horizon:
                    break
                out.append(h.emissions[s].sampler(seed=int(self.rng.randint(1, 2**31))).sample())
            if len(out) >= horizon:
                break
            s = self.rng.choice(h.K, p=h.a[s])
        return out


class EDHMMDataEncoder(DataSequenceEncoder):
    """An ExplicitDurationHMM record is one observation sequence (the durations are latent)."""

    def seq_encode(self, x):
        """Encode EDHMM records as observation-sequence lists."""
        return [list(s) for s in x]

    def row_count(self, x):
        """Return the number of pass-through explicit-duration HMM records."""
        if not isinstance(x, list):
            raise ValueError("explicit-duration HMM encoding must be a list of sequence records")
        return len(x)

    def __eq__(self, other):
        return isinstance(other, EDHMMDataEncoder)

    def __hash__(self):
        return hash("EDHMMDataEncoder")


class EDHMMAccumulator(SequenceEncodableStatisticAccumulator):
    """E-step accumulator for an explicit-duration HMM: per-sequence segment posteriors -> initial /
    transition / per-state DURATION counts + emission occupancy statistics."""

    def __init__(self, emission_accumulators, k, d):
        self.emit = list(emission_accumulators)
        self.K = _exact_positive_integer(k, "EDHMMAccumulator state count")
        if self.K < 2:
            raise ValueError("EDHMMAccumulator requires at least two states.")
        self.D = _exact_positive_integer(d, "EDHMMAccumulator maximum duration")
        if len(self.emit) != self.K:
            raise ValueError(f"EDHMMAccumulator requires exactly {self.K} emission accumulators.")
        self.pi_acc = np.zeros(self.K)
        self.trans_acc = np.zeros((self.K, self.K))
        self.dur_acc = np.zeros((self.K, self.D))
        self.nk = np.zeros(self.K)

    def update(self, x, weight, estimate):
        """Accumulate sufficient statistics from one weighted sequence."""
        weight = validated_observation_weight(weight, "explicit-duration HMM observation weight")
        self.seq_update([x], np.array([weight], dtype=float), estimate)

    def seq_update(self, x, weights, estimate):
        """Accumulate weighted explicit-duration HMM statistics from a batch."""
        sequences = list(x)
        weights = _validated_weights(
            weights,
            len(sequences),
            "ExplicitDurationHMM accumulator weights",
            require_positive_total=False,
        )
        active_indices = np.flatnonzero(weights > 0.0)
        active = [sequences[index] for index in active_indices]
        if any(not sequence for sequence in active):
            raise ValueError("ExplicitDurationHMM accumulator data contains an empty positive-weight sequence.")
        if not active:
            return
        require_possible_log_evidence(
            estimate.seq_log_density(active),
            context="EDHMMAccumulator.seq_update",
        )
        for index in active_indices:
            seq = sequences[index]
            w = weights[index]
            _, pic, trc, drc, occ = estimate._estep(list(seq))
            self.pi_acc += w * pic
            self.trans_acc += w * trc
            self.dur_acc += w * drc
            for k in range(self.K):
                enc = estimate.emissions[k].dist_to_encoder().seq_encode(seq)
                wk = occ[:, k] * w
                self.emit[k].seq_update(enc, wk, estimate.emissions[k])
                self.nk[k] += float(wk.sum())

    def seq_initialize(self, x, weights, rng):
        """Initialize explicit-duration HMM statistics with random soft responsibilities."""
        sequences = list(x)
        weights = _validated_weights(
            weights,
            len(sequences),
            "ExplicitDurationHMM initializer weights",
            require_positive_total=False,
        )
        active_indices = np.flatnonzero(weights > 0.0)
        if any(not sequences[index] for index in active_indices):
            raise ValueError("ExplicitDurationHMM initializer data contains an empty positive-weight sequence.")
        if not len(active_indices):
            return
        for index in active_indices:
            seq = sequences[index]
            w = weights[index]
            state_mass = rng.dirichlet(np.ones(self.K))
            self.pi_acc += w * state_mass
            self.dur_acc[:, min(len(seq), self.D) - 1] += w * state_mass
            for k in range(self.K):
                enc = self.emit[k].acc_to_encoder().seq_encode(seq)
                wk = np.full(len(seq), state_mass[k] * w)
                self.emit[k].seq_initialize(enc, wk, rng)
                self.nk[k] += float(wk.sum())

    def combine(self, suff_stat):
        """Merge serialized explicit-duration HMM sufficient statistics."""
        pi_acc, trans_acc, dur_acc, emit_vals, nk = _validated_edhmm_statistics(
            suff_stat,
            self.K,
            self.D,
            label="explicit-duration HMM sufficient statistics",
        )
        # Transactional with a finiteness postcondition (measured; STAT-RR8-1/RR9-1 classes).
        _snapshot = snapshot_accumulator_statistics(
            self, count_attrs=("pi_acc", "trans_acc", "dur_acc", "nk"), child_attrs=("emit",)
        )
        self.pi_acc += pi_acc
        self.trans_acc += trans_acc
        self.dur_acc += dur_acc
        self.nk += nk
        try:
            require_finite_count_totals(
                (
                    ("initial counts", self.pi_acc),
                    ("transition counts", self.trans_acc),
                    ("duration counts", self.dur_acc),
                    ("emission counts", self.nk),
                ),
                label="combined explicit-duration HMM",
            )
            for k in range(self.K):
                self.emit[k].combine(emit_vals[k])
        except Exception:
            restore_accumulator_statistics(self, _snapshot)
            raise
        return self

    def value(self):
        """Return serialized explicit-duration HMM sufficient statistics."""
        return (
            self.pi_acc.copy(),
            self.trans_acc.copy(),
            self.dur_acc.copy(),
            [e.value() for e in self.emit],
            self.nk.copy(),
        )

    def from_value(self, x):
        """Restore accumulator state from serialized EDHMM statistics."""
        # Candidates validated before ANY assignment; children restore transactionally
        # (measured; STAT-RR9-1 class).
        candidate_pi, candidate_trans, candidate_dur, emit_vals, candidate_nk = _validated_edhmm_statistics(
            x,
            self.K,
            self.D,
            label="explicit-duration HMM sufficient statistics",
        )
        _snapshot = snapshot_accumulator_statistics(
            self, count_attrs=("pi_acc", "trans_acc", "dur_acc", "nk"), child_attrs=("emit",)
        )
        self.pi_acc, self.trans_acc, self.dur_acc, self.nk = (
            candidate_pi,
            candidate_trans,
            candidate_dur,
            candidate_nk,
        )
        try:
            for k in range(self.K):
                self.emit[k].from_value(emit_vals[k])
        except Exception:
            restore_accumulator_statistics(self, _snapshot)
            raise
        return self

    def scale(self, factor):
        """Scale explicit-duration HMM statistics."""
        factor = validated_observation_weight(factor, "explicit-duration HMM scale factor")
        # One transaction with a scaled-result postcondition (measured; STAT-RR8-1/RR10-1).
        _snapshot = snapshot_accumulator_statistics(
            self, count_attrs=("pi_acc", "trans_acc", "dur_acc", "nk"), child_attrs=("emit",)
        )
        self.pi_acc *= factor
        self.trans_acc *= factor
        self.dur_acc *= factor
        self.nk *= factor
        try:
            require_finite_count_totals(
                (
                    ("initial counts", self.pi_acc),
                    ("transition counts", self.trans_acc),
                    ("duration counts", self.dur_acc),
                    ("emission counts", self.nk),
                ),
                label="scaled explicit-duration HMM",
            )
            for accumulator in self.emit:
                accumulator.scale(factor)
        except Exception:
            restore_accumulator_statistics(self, _snapshot)
            raise
        return self

    def acc_to_encoder(self):
        """Return the encoder associated with this accumulator."""
        return EDHMMDataEncoder()


class EDHMMAccumulatorFactory(StatisticAccumulatorFactory):
    """Factory for explicit-duration HMM accumulators."""

    def __init__(self, emission_estimators, k, d):
        self.emission_estimators = emission_estimators
        self.k, self.d = k, d

    def make(self):
        """Create a fresh explicit-duration HMM accumulator."""
        emit = [est.accumulator_factory().make() for est in self.emission_estimators]
        return EDHMMAccumulator(emit, self.k, self.d)


class EDHMMEstimator(ParameterEstimator):
    """Estimator (M-step) for an :class:`ExplicitDurationHMM`: re-estimates pi, the zero-diagonal transition,
    the per-state DURATION distributions, and each state's emission from the segment-posterior statistics."""

    def __init__(self, emission_estimators, k, d, name=None):
        self.emission_estimators = list(emission_estimators)
        self.k = _exact_positive_integer(k, "EDHMMEstimator state count")
        if self.k < 2:
            raise ValueError("EDHMMEstimator requires at least two states.")
        self.d = _exact_positive_integer(d, "EDHMMEstimator maximum duration")
        if len(self.emission_estimators) != self.k:
            raise ValueError(f"EDHMMEstimator requires exactly {self.k} emission estimators.")
        self.name = name

    def accumulator_factory(self):
        """Return the accumulator factory used by this estimator."""
        return EDHMMAccumulatorFactory(self.emission_estimators, self.k, self.d)

    def estimate(self, nobs, suff_stat):
        """Estimate an explicit-duration HMM from segment posterior statistics."""
        pi_acc, trans_acc, dur_acc, emit_vals, nk = _validated_edhmm_statistics(
            suff_stat,
            self.k,
            self.d,
            label="explicit-duration HMM sufficient statistics",
        )
        validate_effective_sample_mass(
            nobs,
            pi_acc.sum(),
            label="explicit-duration HMM effective sample",
        )
        pi = pi_acc / pi_acc.sum() if pi_acc.sum() > 0 else np.ones(self.k) / self.k
        a = trans_acc.copy()
        np.fill_diagonal(a, 0.0)
        transition_fallback = np.full((self.k, self.k), 1.0 / (self.k - 1))
        np.fill_diagonal(transition_fallback, 0.0)
        a = _row_normalize(a, transition_fallback)
        dur = _row_normalize(dur_acc, np.full((self.k, self.d), 1.0 / self.d))
        emissions = [self.emission_estimators[i].estimate(float(nk[i]), emit_vals[i]) for i in range(self.k)]
        return ExplicitDurationHMM(emissions, pi, a, dur, self.d, name=self.name)


SequenceEncodableProbabilityDistribution.register(ExplicitDurationHMM)


def jit_forward_loglik(hmm: StructuredHMM):
    """Compile the scaled forward log-likelihood recursion to a single jax.jit XLA program (lax.scan over
    time). Returns a callable ``score(seq) -> float``: emission log-densities are evaluated on the host
    (arbitrary emissions), then the forward scan runs jitted on the transition matrix. Works for any
    operator (uses ``as_matrix()``); the win is large T / K. Requires the JAX optional extra.

    Precision follows the caller's ambient JAX policy rather than mixle's: with the default
    ``jax_enable_x64=False`` the scan runs in float32, so the returned score agrees with
    ``seq_log_density`` to roughly float32 relative precision, not to float64. Enable x64 in your own
    process if you need more -- mixle does not set it for you (MXR-080-0147 removed the import-time
    global mutation that used to make this path silently float64)."""
    import jax
    import jax.numpy as jnp

    a_mat = jnp.asarray(hmm.transition.as_matrix())
    pi = jnp.asarray(hmm.pi)

    @jax.jit
    def _fwd(log_b):
        mx = log_b.max(axis=1, keepdims=True)
        b = jnp.exp(log_b - mx)
        a0 = pi * b[0]
        c0 = a0.sum()

        def step(alpha, bt):
            a2 = (alpha @ a_mat) * bt
            c = a2.sum()
            return a2 / c, jnp.log(c)

        _, logc = jax.lax.scan(step, a0 / c0, b[1:])
        return jnp.sum(logc) + jnp.log(c0) + jnp.sum(mx)

    def score(seq):
        return float(_fwd(jnp.asarray(hmm._log_b(seq))))

    return score
