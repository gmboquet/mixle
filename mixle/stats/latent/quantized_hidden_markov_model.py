"""Quantized hidden Markov models with shared-base probability exponents.

Data type: Sequence[T] where T is a categorical emission value drawn from a finite set of levels.

A quantized HMM is an HMM with finite categorical emissions in which every probability in the model
is a power of a single shared base theta in (0, 1):

    (1) State transitions:      p_mat(Z_t = j | Z_{t-1} = i) = theta^k_trans[i, j] / Z_trans[i]
    (2) Emissions:              P(X_t = v | Z_t = i)         = theta^k_emit[i, v]  / Z_emit[i]
    (3) Initial states:         p_mat(Z_1 = j)               = theta^k_init[j]     / Z_init
                                (or the stationary distribution of (1), see init_mode)

where every exponent k is a non-negative integer and Z_row = sum_j theta^k[row, j] is the per-row
normalizer required because integer exponents with a single shared theta cannot sum to exactly one.
A negative exponent entry (-1) marks a structural zero: that transition/emission has probability 0.

Log-probabilities therefore take the form k * log(theta) - log(Z_row): a single integer multiply
plus one cached per-row float, so log <-> probability conversion is fast and the maximum-likelihood
parameters are naturally quantized (at theta = 1/2 the exponents are code lengths in bits).

Estimation fits within EM: the E-step is the ordinary Baum-Welch pass (reused unchanged from
mixle.stats.latent.hidden_markov), while the M-step performs coordinate ascent on the expected
complete-data log-likelihood, alternating between (a) quantizing the unconstrained row MLEs to
integer exponents, k = round(log(p_hat) / log(theta)), and (b) a 1-d bounded maximization of

    f(theta) = sum_cells N * k * log(theta) - sum_rows N_row * log(Z_row(theta))

over theta. theta is shared by the transition, emission, and (when init_mode='quantized') initial
state blocks. Note that without a k_max cap the likelihood always improves as theta -> 1 (the
quantization grid becomes arbitrarily fine), so supply k_max (or fixed_theta) when a coarse,
meaningful quantization is wanted.

Because the M-step rounds probabilities to the theta^k grid, near-symmetric states produced by a
random initialization can round onto (nearly) identical grid points and leave EM at an exact fixed
point (dense HMM EM escapes such saddles by amplifying tiny count asymmetries, which quantization
erases). The raw expected counts still carry those asymmetries, so by default the estimator checks
after each M-step for state pairs whose quantized log-probabilities differ by less than split_nats
everywhere and pushes them split_nats apart along the strongest raw-count asymmetry
(split_collapsed=True), restoring a hill-climbing direction for the next E-step; an unwarranted
split is simply rounded back by the next M-step. Random restarts
(e.g. mixle.inference.estimation.best_of) remain useful for multimodality, as with any HMM.

If included, the length of the sequences is modeled through a length distribution with support on
non-negative integers.
"""

import heapq
import itertools
import math
from collections.abc import Iterator, Sequence
from typing import Any

import numpy as np
from scipy.optimize import minimize_scalar
from scipy.special import logsumexp

from mixle.capability import Neutral, supports
from mixle.enumeration.algorithms import BufferedStream, LengthFrontierMerge
from mixle.stats.combinator.null_dist import NullDistribution, NullEstimator
from mixle.stats.compute.pdist import (
    DistributionEnumerator,
    EnumerationError,
    ParameterEstimator,
    SequenceEncodableProbabilityDistribution,
    child_enumerator,
)
from mixle.stats.latent.hidden_markov import HiddenMarkovAccumulatorFactory, HiddenMarkovModelDistribution
from mixle.stats.sequences.markov_chain import stationary_distribution
from mixle.stats.univariate.discrete.categorical import CategoricalDistribution, CategoricalEstimator
from mixle.utils.optional_deps import HAS_NUMBA

STRUCTURAL_ZERO = -1


def _exponent_log_probs(exponents: np.ndarray, log_theta: float) -> np.ndarray:
    """Row-normalized log-probabilities k*log(theta) - log(Z_row) for an integer exponent matrix.

    Entries with a negative exponent (structural zeros) map to -inf.

    Args:
        exponents (np.ndarray): 2-d integer exponent matrix; negative entries are structural zeros.
        log_theta (float): Log of the shared base theta.

    Returns:
        2-d numpy array of log-probabilities with rows summing to one in probability space.

    """
    mask = exponents >= 0
    log_p = np.where(mask, exponents * log_theta, -np.inf)
    log_z = logsumexp(log_p, axis=-1, keepdims=True)
    return log_p - log_z


def _quantize_counts(counts: np.ndarray, log_theta: float, k_max: int | None) -> np.ndarray:
    """Quantize per-row counts to integer exponents for a fixed theta.

    Each row is normalized to the unconstrained MLE p_hat and quantized to
    k = round(log(p_hat) / log(theta)), anchored so the smallest exponent in each row is zero
    (the row objective is invariant to a common exponent shift). Cells with zero count become
    structural zeros (-1); rows with zero total count become uniform (all exponents zero).

    Args:
        counts (np.ndarray): 2-d non-negative count matrix.
        log_theta (float): Log of the shared base theta.
        k_max (Optional[int]): If set, exponents are clipped to [0, k_max].

    Returns:
        2-d integer numpy array of exponents with -1 marking structural zeros.

    """
    counts = np.asarray(counts, dtype=np.float64)
    rv = np.full(counts.shape, STRUCTURAL_ZERO, dtype=np.int64)

    for r in range(counts.shape[0]):
        row = counts[r, :]
        row_total = row.sum()
        pos = row > 0

        if row_total <= 0 or not np.any(pos):
            rv[r, :] = 0
            continue

        k = np.rint(np.log(row[pos] / row_total) / log_theta).astype(np.int64)
        k -= k.min()
        if k_max is not None:
            np.clip(k, 0, k_max, out=k)
        rv[r, pos] = k

    return rv


def _quantized_expected_ll(count_blocks: Sequence[np.ndarray], exp_blocks: Sequence[np.ndarray], theta: float) -> float:
    """Expected complete-data log-likelihood of count blocks under exponent blocks and theta.

    Args:
        count_blocks (Sequence[np.ndarray]): 2-d expected-count matrices (transitions, emissions,
            and optionally the initial-state row).
        exp_blocks (Sequence[np.ndarray]): Matching 2-d integer exponent matrices.
        theta (float): Shared base in (0, 1).

    Returns:
        Expected complete-data log-likelihood (up to terms constant in theta and the exponents).

    """
    log_theta = math.log(theta)
    rv = 0.0

    for counts, exponents in zip(count_blocks, exp_blocks):
        mask = exponents >= 0
        log_p = np.where(mask, exponents * log_theta, -np.inf)
        log_z = logsumexp(log_p, axis=-1)
        rv += log_theta * float(np.sum(counts[mask] * exponents[mask]))
        rv -= float(np.sum(counts.sum(axis=-1) * log_z))

    return rv


def _optimize_theta(
    count_blocks: Sequence[np.ndarray], exp_blocks: Sequence[np.ndarray], current_theta: float
) -> float:
    """Maximize the expected complete-data log-likelihood over theta for fixed exponents.

    Args:
        count_blocks (Sequence[np.ndarray]): 2-d expected-count matrices.
        exp_blocks (Sequence[np.ndarray]): Matching 2-d integer exponent matrices.
        current_theta (float): Returned unchanged when the objective is constant in theta (all
            finite exponents zero).

    Returns:
        Optimal theta in (0, 1).

    """
    if all(np.all(k[k >= 0] == 0) for k in exp_blocks):
        return current_theta

    res = minimize_scalar(
        lambda t: -_quantized_expected_ll(count_blocks, exp_blocks, t), bounds=(1.0e-6, 1.0 - 1.0e-6), method="bounded"
    )

    return float(res.x)


def _fit_quantized_parameters(
    count_blocks: Sequence[np.ndarray], fixed_theta: float | None, k_max: int | None, max_its: int
) -> tuple[float, list[np.ndarray]]:
    """Coordinate ascent over (theta, integer exponents) for the quantized M-step.

    Alternates exponent quantization at the current theta with the 1-d theta maximization, from
    several theta starting points, keeping the (theta, exponents) pair with the best expected
    complete-data log-likelihood seen (the rounding step is not guaranteed monotone).

    Args:
        count_blocks (Sequence[np.ndarray]): 2-d expected-count matrices sharing theta.
        fixed_theta (Optional[float]): If set, theta is not optimized and a single quantization
            pass is performed.
        k_max (Optional[int]): Optional cap on the integer exponents.
        max_its (int): Maximum coordinate-ascent iterations per starting point.

    Returns:
        Tuple of (theta, list of integer exponent matrices matching count_blocks).

    """
    if fixed_theta is not None:
        log_theta = math.log(fixed_theta)
        return fixed_theta, [_quantize_counts(u, log_theta, k_max) for u in count_blocks]

    best_ll = -np.inf
    best: tuple[float, list[np.ndarray]] | None = None

    for theta0 in (0.25, 0.5, 0.75, 0.9):
        theta = theta0
        prev_exps: list[np.ndarray] | None = None

        for _ in range(max_its):
            exps = [_quantize_counts(u, math.log(theta), k_max) for u in count_blocks]
            theta = _optimize_theta(count_blocks, exps, theta)

            ll = _quantized_expected_ll(count_blocks, exps, theta)
            if ll > best_ll:
                best_ll = ll
                best = (theta, exps)

            if prev_exps is not None and all(np.array_equal(u, v) for u, v in zip(prev_exps, exps)):
                break
            prev_exps = exps

    return best


def _swap_perm(num_states: int, i: int, j: int) -> np.ndarray:
    """Permutation of range(num_states) exchanging i and j."""
    perm = np.arange(num_states)
    perm[i], perm[j] = j, i
    return perm


def _states_nearly_collapsed(
    trans_exp: np.ndarray, emit_exp: np.ndarray, i: int, j: int, log_theta: float, tol_nats: float
) -> bool:
    """True when states i and j are effectively indistinguishable for the EM dynamics.

    Compares the normalized log-probabilities of the emission rows and the swap-aware transition
    rows (state j's row with columns i and j exchanged): the pair counts as collapsed when no cell
    differs by more than tol_nats. Structural-zero patterns must match exactly.

    Args:
        trans_exp (np.ndarray): Integer transition exponent matrix.
        emit_exp (np.ndarray): Integer emission exponent matrix.
        i (int): First state index.
        j (int): Second state index.
        log_theta (float): Log of the shared base theta.
        tol_nats (float): Maximum per-cell log-probability difference (in nats) below which the
            states are considered collapsed.

    Returns:
        True if the two states' quantized parameters are within tol_nats of exchangeable.

    """
    perm = _swap_perm(trans_exp.shape[0], i, j)

    for row_i, row_j in ((emit_exp[i, :], emit_exp[j, :]), (trans_exp[i, :], trans_exp[j, perm])):
        zero_i = row_i < 0
        zero_j = row_j < 0
        if not np.array_equal(zero_i, zero_j):
            return False
        finite = ~zero_i
        if np.any(finite):
            log_p_i = _exponent_log_probs(row_i[None, :], log_theta)[0, finite]
            log_p_j = _exponent_log_probs(row_j[None, :], log_theta)[0, finite]
            if np.max(np.abs(log_p_i - log_p_j)) > tol_nats:
                return False

    return True


def _split_collapsed_pair(
    trans_exp: np.ndarray,
    emit_exp: np.ndarray,
    trans_counts: np.ndarray,
    emit_counts: np.ndarray,
    i: int,
    j: int,
    k_max: int | None,
    log_theta: float,
    split_nats: float,
) -> bool:
    """Push states i and j apart along their strongest raw-count asymmetry.

    The quantized M-step can round two nearly-identical states onto (nearly) the same theta^k grid
    points, leaving EM at an exact fixed point even though the raw expected counts still differ
    between the states. This locates the cell (emission, or swap-aware transition) with the
    largest log-probability difference between the raw count rows and sets the exponent gap at
    that cell to about split_nats of separation (ceil(split_nats / |log theta|) quanta), making
    the less-likely state less likely there. The next E-step then sees distinguishable states and
    ordinary hill climbing resumes; if the split was unwarranted, the next quantization simply
    rounds it back.

    Args:
        trans_exp (np.ndarray): Integer transition exponent matrix (mutated in place).
        emit_exp (np.ndarray): Integer emission exponent matrix (mutated in place).
        trans_counts (np.ndarray): Raw expected transition counts from the E-step.
        emit_counts (np.ndarray): Raw expected emission counts from the E-step.
        i (int): First state index.
        j (int): Second state index.
        k_max (Optional[int]): Cap on the integer exponents.
        log_theta (float): Log of the shared base theta.
        split_nats (float): Target log-probability separation (in nats) at the split cell.

    Returns:
        True if an exponent was adjusted, False when the raw counts carry no asymmetry.

    """
    perm = _swap_perm(trans_exp.shape[0], i, j)

    # cells: (signal, exponents, (row_i, col_i), (row_j, col_j)); signal > 0 means state i is
    # relatively more likely there than state j under the raw counts
    cells = []
    signals = []

    for counts_i, counts_j, exps, cell_i, cell_j in (
        (emit_counts[i, :], emit_counts[j, :], emit_exp, lambda c: (i, c), lambda c: (j, c)),
        (trans_counts[i, :], trans_counts[j, perm], trans_exp, lambda c: (i, c), lambda c: (j, perm[c])),
    ):
        tot_i = counts_i.sum()
        tot_j = counts_j.sum()
        if tot_i <= 0 or tot_j <= 0:
            continue

        for c in range(len(counts_i)):
            if exps[cell_i(c)] < 0 or exps[cell_j(c)] < 0:
                continue
            if counts_i[c] <= 0 or counts_j[c] <= 0:
                continue
            signal = math.log(counts_i[c] / tot_i) - math.log(counts_j[c] / tot_j)
            cells.append((exps, cell_i(c), cell_j(c)))
            signals.append(signal)

    if len(cells) == 0:
        return False

    signals = np.asarray(signals)
    c = int(np.argmax(np.abs(signals)))
    if signals[c] == 0.0:
        return False

    exps, cell_i, cell_j = cells[c]
    low_cell, high_cell = (cell_i, cell_j) if signals[c] < 0.0 else (cell_j, cell_i)

    gap = max(1, int(math.ceil(split_nats / abs(log_theta))))
    exps[low_cell] = exps[high_cell] + gap
    if k_max is not None and exps[low_cell] > k_max:
        exps[low_cell] = k_max
        exps[high_cell] = max(0, k_max - gap)

    return True


def _split_collapsed_states(
    trans_exp: np.ndarray,
    emit_exp: np.ndarray,
    trans_counts: np.ndarray,
    emit_counts: np.ndarray,
    k_max: int | None,
    log_theta: float,
    split_nats: float,
) -> int:
    """Separate every nearly-collapsed state pair; returns the number of splits applied."""
    num_states = trans_exp.shape[0]
    num_split = 0

    for i in range(num_states):
        for j in range(i + 1, num_states):
            if _states_nearly_collapsed(trans_exp, emit_exp, i, j, log_theta, split_nats):
                if _split_collapsed_pair(
                    trans_exp, emit_exp, trans_counts, emit_counts, i, j, k_max, log_theta, split_nats
                ):
                    num_split += 1

    return num_split


class QuantizedHiddenMarkovModelDistribution(HiddenMarkovModelDistribution):
    """Hidden Markov model distribution with quantized observation summaries."""

    def compute_declaration(self):
        """Return the generated-compute declaration for the quantized HMM."""
        from mixle.stats.compute.declarations import (
            DistributionDeclaration,
            ParameterSpec,
            StatisticSpec,
            declaration_for,
        )

        length = None if supports(self.len_dist, Neutral) else declaration_for(self.len_dist)
        children = () if length is None else (length,)
        return DistributionDeclaration(
            name="quantized_hidden_markov",
            distribution_type=type(self),
            parameters=(
                ParameterSpec("theta", constraint="unit_interval"),
                ParameterSpec("levels", constraint="metadata", differentiable=False),
                ParameterSpec("transition_exponents", constraint="integer_matrix", differentiable=False),
                ParameterSpec("emission_exponents", constraint="integer_matrix", differentiable=False),
                ParameterSpec("initial_exponents", constraint="integer_vector", differentiable=False),
                ParameterSpec("init_mode", constraint="metadata", differentiable=False),
                ParameterSpec("k_max", constraint="optional_integer", differentiable=False),
            ),
            statistics=(
                StatisticSpec("num_states", kind="metadata", additive=False, scales=False),
                StatisticSpec("initial_counts"),
                StatisticSpec("state_counts"),
                StatisticSpec("transition_counts"),
                StatisticSpec("emissions", kind="tuple"),
                StatisticSpec("length", kind="child_stat"),
            ),
            support="quantized_hidden_state_sequence",
            children=children,
            child_roles=("length",) if length is not None else (),
            differentiable=False,
        )

    def __init__(
        self,
        theta: float,
        levels: Sequence[Any],
        transition_exponents: Sequence[Sequence[int]] | np.ndarray,
        emission_exponents: Sequence[Sequence[int]] | np.ndarray,
        initial_exponents: Sequence[int] | np.ndarray | None = None,
        init_mode: str = "quantized",
        k_max: int | None = None,
        len_dist: SequenceEncodableProbabilityDistribution | None = NullDistribution(),
        name: str | None = None,
        terminal_values: set | None = None,
        use_numba: bool | None = None,
        terminal_states: set[int] | Sequence[int] | None = None,
    ) -> None:
        """QuantizedHiddenMarkovModelDistribution: an HMM whose probabilities are powers of theta.

        Every transition, emission, and (when init_mode='quantized') initial-state probability is
        theta^k / Z_row for a shared base theta and per-cell non-negative integer exponents k.
        Exponent entries equal to -1 are structural zeros (probability 0). Each exponent row must
        contain at least one non-negative entry.

        Args:
            theta (float): Shared base in (0, 1).
            levels (Sequence[Any]): Emission support values; column v of emission_exponents is the
                exponent of levels[v].
            transition_exponents (Union[Sequence[Sequence[int]], np.ndarray]): n_states by n_states
                integer exponent matrix for state transitions.
            emission_exponents (Union[Sequence[Sequence[int]], np.ndarray]): n_states by len(levels)
                integer exponent matrix for emissions.
            initial_exponents (Optional[Union[Sequence[int], np.ndarray]]): Length n_states integer
                exponents for the initial state distribution. Required when init_mode='quantized'.
            init_mode (str): 'quantized' (initial states are theta^k / Z as well) or 'stationary'
                (initial states are the stationary distribution of the transition matrix).
            k_max (Optional[int]): Exponent cap used when re-estimating this distribution; carried
                through estimator().
            len_dist (Optional[SequenceEncodableProbabilityDistribution]): Distribution with support
                on non-negative integers for the sequence lengths.
            name (Optional[str]): Optional distribution name.
            terminal_values (Optional[set]): Define terminating emission outputs of the HMM.
            use_numba (Optional[bool]): Whether to use the numba encoding and vectorized operations.
                ``None`` (default) selects the compiled path when numba is installed.

        Attributes:
            theta (float): Shared base.
            log_theta (float): Log of the shared base; log-probabilities are k*log_theta - log Z.
            levels (List[Any]): Emission support values.
            transition_exponents (np.ndarray): Integer transition exponents (-1 = structural zero).
            emission_exponents (np.ndarray): Integer emission exponents (-1 = structural zero).
            initial_exponents (Optional[np.ndarray]): Integer initial-state exponents, or None when
                init_mode='stationary'.
            init_mode (str): Initial-state parameterization mode.
            k_max (Optional[int]): Exponent cap carried into estimator().

        All attributes of HiddenMarkovModelDistribution (transitions, w, topics, ...) are populated
        with the corresponding dense probabilities, so density evaluation, Viterbi, sampling, and
        sequence encoding are inherited unchanged.

        """
        if not (0.0 < theta < 1.0):
            raise ValueError("QuantizedHiddenMarkovModelDistribution requires theta in (0, 1).")
        if init_mode not in ("quantized", "stationary"):
            raise ValueError("init_mode must be 'quantized' or 'stationary'.")

        self.theta = float(theta)
        self.log_theta = math.log(self.theta)
        self.levels = list(levels)
        self.init_mode = init_mode
        self.k_max = k_max

        self.transition_exponents = np.asarray(transition_exponents, dtype=np.int64)
        self.emission_exponents = np.asarray(emission_exponents, dtype=np.int64)

        num_states = self.transition_exponents.shape[0]
        num_levels = len(self.levels)

        if self.transition_exponents.shape != (num_states, num_states):
            raise ValueError("transition_exponents must be a square matrix.")
        if self.emission_exponents.shape != (num_states, num_levels):
            raise ValueError("emission_exponents must have shape (n_states, len(levels)).")
        if not np.all(np.any(self.transition_exponents >= 0, axis=1)):
            raise ValueError("Each transition_exponents row needs a non-negative entry.")
        if not np.all(np.any(self.emission_exponents >= 0, axis=1)):
            raise ValueError("Each emission_exponents row needs a non-negative entry.")

        transitions = np.exp(_exponent_log_probs(self.transition_exponents, self.log_theta))
        emission_probs = np.exp(_exponent_log_probs(self.emission_exponents, self.log_theta))

        if init_mode == "quantized":
            if initial_exponents is None:
                raise ValueError("initial_exponents is required when init_mode='quantized'.")
            self.initial_exponents = np.reshape(np.asarray(initial_exponents, dtype=np.int64), num_states)
            if not np.any(self.initial_exponents >= 0):
                raise ValueError("initial_exponents needs a non-negative entry.")
            w = np.exp(_exponent_log_probs(self.initial_exponents[None, :], self.log_theta))[0]
        else:
            self.initial_exponents = None
            w = stationary_distribution(transitions)

        topics = [
            CategoricalDistribution(dict(zip(self.levels, emission_probs[i, :].tolist()))) for i in range(num_states)
        ]

        super().__init__(
            topics=topics,
            w=w,
            transitions=transitions,
            taus=None,
            len_dist=len_dist,
            name=name,
            terminal_values=terminal_values,
            use_numba=use_numba,
            terminal_states=terminal_states,
        )

    def __str__(self) -> str:
        """Return a constructor-style representation of the distribution."""
        s_init = repr(None if self.initial_exponents is None else [int(u) for u in self.initial_exponents])

        return (
            "QuantizedHiddenMarkovModelDistribution(%s, %s, %s, %s, initial_exponents=%s, "
            "init_mode=%s, k_max=%s, len_dist=%s, name=%s, terminal_values=%s, use_numba=%s)"
            % (
                repr(self.theta),
                repr(self.levels),
                repr([[int(v) for v in u] for u in self.transition_exponents]),
                repr([[int(v) for v in u] for u in self.emission_exponents]),
                s_init,
                repr(self.init_mode),
                repr(self.k_max),
                str(self.len_dist),
                repr(self.name),
                repr(self.terminal_values),
                repr(self.use_numba),
            )
        )

    @classmethod
    def left_to_right(
        cls,
        theta: float,
        levels: Sequence[Any],
        transition_exponents: Sequence[Sequence[int]] | np.ndarray,
        emission_exponents: Sequence[Sequence[int]] | np.ndarray,
        initial_exponents: Sequence[int] | np.ndarray | None = None,
        **kwargs: Any,
    ) -> "QuantizedHiddenMarkovModelDistribution":
        """Construct a left-to-right (upper-triangular) quantized HMM.

        ``transition_exponents`` must be upper triangular: every entry strictly below the diagonal is a
        structural zero (negative exponent), so the hidden-state path is monotone non-decreasing (a Bakis
        chain). This makes a sentence's state paths exactly its monotone *segmentations* -- only
        polynomially many in the length (``O(L^{n-1})``) rather than the ``n^L`` of a general HMM -- which
        bounds the path/sequence ambiguity. When the per-state emission supports are additionally disjoint
        the model is *unambiguous* (one path per sentence); then the structural descending-probability
        seek/unrank coincides with the exact marginal order (up to quantization granularity, no path
        over-count), which a general HMM's structural seek cannot.

        Raises:
            ValueError: if ``transition_exponents`` is not square or not upper triangular.
        """
        t = np.asarray(transition_exponents)
        if t.ndim != 2 or t.shape[0] != t.shape[1]:
            raise ValueError("transition_exponents must be a square matrix")
        below = t[np.tril_indices(t.shape[0], k=-1)]
        if below.size and np.any(below >= 0):
            raise ValueError(
                "left_to_right requires upper-triangular transition_exponents: every entry below the "
                "diagonal must be a structural zero (negative exponent)"
            )
        return cls(
            theta, levels, transition_exponents, emission_exponents, initial_exponents=initial_exponents, **kwargs
        )

    def to_fisher(self, **kwargs):
        """Forward-backward Fisher view for the quantized HMM."""
        if hasattr(self, "topics") and hasattr(self, "transitions"):
            from mixle.stats.latent.hidden_markov import HiddenMarkovFisherView

            return HiddenMarkovFisherView(self)
        return super().to_fisher(**kwargs)

    def estimator(self, pseudo_count: float | None = None) -> "QuantizedHiddenMarkovEstimator":
        """Return an estimator matching this quantized HMM configuration.

        Args:
            pseudo_count (Optional[float]): Per-cell pseudo count for the initial, transition, and
                emission expected counts. When None, unobserved cells become structural zeros.

        Returns:
            QuantizedHiddenMarkovEstimator object.

        """
        len_est = (
            NullEstimator() if supports(self.len_dist, Neutral) else self.len_dist.estimator(pseudo_count=pseudo_count)
        )

        return QuantizedHiddenMarkovEstimator(
            self.n_states,
            levels=self.levels,
            pseudo_count=pseudo_count,
            k_max=self.k_max,
            init_mode=self.init_mode,
            len_estimator=len_est,
            name=self.name,
            use_numba=self.use_numba,
        )

    def enumerator(self):
        """Return an exact descending-probability enumerator over observation sequences.

        For the ordinary (length-distribution) case this is the quantized-HMM-specialized enumerator,
        which avoids constructing per-state categorical streams and uses the cached quantized emission
        log-probability matrix directly. For the ``terminal_values`` stopping-time case it delegates to
        :class:`HiddenMarkovModelEnumerator`, which implements that support (the quantized specialization
        only covers the length-distribution path).
        """
        if self.terminal_values is not None:
            from mixle.stats.latent.hidden_markov import HiddenMarkovModelEnumerator

            return HiddenMarkovModelEnumerator(self)
        return QuantizedHiddenMarkovModelEnumerator(self)

    def determinize(self, max_states: int = 1 << 16):
        """Weighted determinization (Mohri 1997; Mohri & Riley 2002) of this terminal-value quantized HMM
        into a :class:`~mixle.stats.latent.hmm_determinize.DeterminizedSequenceDistribution`.

        Rebuilds the machine over belief states (exact rational arithmetic) so each sequence has a single
        path and edge weights multiply to the exact marginal -- yielding exact, duplicate-free n-best
        *sequences* (not n-best paths). Requires terminal_values. Raises EnumerationError if the belief
        expansion exceeds ``max_states`` (the twins property fails -- not finitely determinizable; keep the
        original HMM's exact O(index) enumerate-and-bin path instead)."""
        from mixle.stats.latent.hmm_determinize import determinize_quantized_terminal

        return determinize_quantized_terminal(self, max_states=max_states)


class _QuantizedHmmPrefix:
    """Concrete observation prefix used by QuantizedHiddenMarkovModelEnumerator."""

    __slots__ = ("t", "values", "proj")

    def __init__(self, t: int, values: tuple[Any, ...], proj: np.ndarray) -> None:
        self.t = t
        self.values = values
        self.proj = proj


class QuantizedHiddenMarkovModelEnumerator(DistributionEnumerator):
    """Exact best-first enumerator specialized for QuantizedHiddenMarkovModelDistribution."""

    def __init__(self, dist: QuantizedHiddenMarkovModelDistribution) -> None:
        """Create an enumerator for quantized HMM paths.

        Args:
            dist (QuantizedHiddenMarkovModelDistribution): Distribution whose support is enumerated.

        """
        super().__init__(dist)
        if dist.has_topics:
            raise EnumerationError(dist, reason="taus/topics parameterization is not supported")
        if dist.terminal_values is not None:
            raise EnumerationError(dist, reason="terminal_values semantics are not supported")
        if supports(dist.len_dist, Neutral):
            raise EnumerationError(dist, reason="no length distribution is modeled (len_dist is Null)")

        self._levels = list(dist.levels)
        self._n_states = dist.n_states
        self._log_w = np.asarray(dist.log_w, dtype=np.float64)
        self._log_a = np.asarray(dist.log_transitions, dtype=np.float64)
        self._emit_lp = _exponent_log_probs(dist.emission_exponents, dist.log_theta)
        self._head_max = np.max(self._emit_lp, axis=1)

        pool = []
        for rank, value in enumerate(self._levels):
            emis = self._emit_lp[:, rank]
            pool_lp = float(np.max(emis))
            if pool_lp > -np.inf:
                pool.append((value, pool_lp, emis))
        pool.sort(key=lambda u: -u[1])
        self._pool = pool

        # UB[r][s] bounds r further (transition + emission) steps out of state s.
        self._ub: list[np.ndarray] = [np.zeros(self._n_states, dtype=np.float64)]

        len_stream = BufferedStream(child_enumerator(dist.len_dist, "QuantizedHiddenMarkovModelDistribution.len_dist"))
        self._merge = LengthFrontierMerge(len_stream, self._kbest_sequences)

    def _emissions(self, rank: int) -> np.ndarray | None:
        if rank >= len(self._pool):
            return None
        return self._pool[rank][2]

    def _ub_for(self, r: int) -> np.ndarray:
        while len(self._ub) <= r:
            prev = self._ub[-1]
            step = self._log_a + (self._head_max + prev)[None, :]
            self._ub.append(logsumexp(step, axis=1))
        return self._ub[r]

    def _kbest_sequences(self, n: int, lp_len: float) -> Iterator[tuple[list[Any], float]]:
        if n == 0:
            yield ([], lp_len)
            return
        if len(self._pool) == 0:
            return

        counter = itertools.count()
        heap = []  # entries: (-score, counter, kind, payload)

        def push_candidate(parent: "_QuantizedHmmPrefix", rank: int) -> None:
            if rank >= len(self._pool):
                return
            pool_lp = self._pool[rank][1]
            remaining = n - parent.t - 1
            bound = logsumexp(parent.proj + self._ub_for(remaining)) + pool_lp + lp_len
            if bound > -np.inf:
                heapq.heappush(heap, (-bound, next(counter), "cand", (parent, rank)))

        root = _QuantizedHmmPrefix(0, (), self._log_w)
        push_candidate(root, 0)

        while heap:
            _, _, kind, payload = heapq.heappop(heap)
            if kind == "done":
                yield payload
                continue

            parent, rank = payload
            push_candidate(parent, rank + 1)
            x = self._pool[rank][0]
            alpha = parent.proj + self._pool[rank][2]
            t = parent.t + 1
            if np.max(alpha) == -np.inf:
                continue
            if t == n:
                exact = float(logsumexp(alpha) + lp_len)
                if exact > -np.inf:
                    heapq.heappush(heap, (-exact, next(counter), "done", (list(parent.values) + [x], exact)))
            else:
                proj = logsumexp(alpha[:, None] + self._log_a, axis=0)
                child = _QuantizedHmmPrefix(t, parent.values + (x,), proj)
                push_candidate(child, 0)

    def __next__(self) -> tuple[list[Any], float]:
        return next(self._merge)


class QuantizedHiddenMarkovEstimator(ParameterEstimator):
    """Estimate quantized hidden Markov models from accumulated sufficient statistics."""

    def __init__(
        self,
        num_states: int,
        levels: Sequence[Any] | None = None,
        pseudo_count: float | None = None,
        k_max: int | None = None,
        fixed_theta: float | None = None,
        init_mode: str = "quantized",
        len_estimator: ParameterEstimator | None = NullEstimator(),
        name: str | None = None,
        keys: tuple[str | None, str | None, str | None] | None = (None, None, None),
        use_numba: bool | None = None,
        max_quant_its: int = 50,
        split_collapsed: bool = True,
        split_nats: float = math.log(2.0),
    ) -> None:
        """QuantizedHiddenMarkovEstimator for estimating QuantizedHiddenMarkovModelDistribution.

        The E-step accumulates the standard Baum-Welch expected counts (reusing
        HiddenMarkovAccumulator with categorical emission accumulators). The M-step jointly fits
        the shared base theta and the integer exponent matrices by coordinate ascent on the
        expected complete-data log-likelihood.

        Args:
            num_states (int): Number of hidden states.
            levels (Optional[Sequence[Any]]): Emission support values to include in addition to the
                values observed in the data. Levels supplied here but never observed become
                structural zeros when pseudo_count is None.
            pseudo_count (Optional[float]): Per-cell pseudo count added to the initial, transition,
                and emission expected counts. When None (or 0), zero-count cells become structural
                zeros (probability 0).
            k_max (Optional[int]): Cap on the integer exponents; zero-count cells under a pseudo
                count and very rare events land on the probability floor theta^k_max. Without
                k_max, a free theta drifts toward 1 across EM iterations (ever finer quantization).
            fixed_theta (Optional[float]): If set, theta is held fixed (e.g. 0.5 for bit-length
                semantics) and only the integer exponents are estimated.
            init_mode (str): 'quantized' (initial states get their own theta^k row sharing theta)
                or 'stationary' (initial states are the stationary distribution of the fitted
                transition matrix).
            len_estimator (Optional[ParameterEstimator]): Optional ParameterEstimator for the
                sequence length distribution.
            name (Optional[str]): Optional name assigned to estimated distributions.
            keys (Optional[Tuple[Optional[str], Optional[str], Optional[str]]]): Set keys for
                initial states, transition counts, and emission accumulators.
            use_numba (Optional[bool]): If True, Numba is used for sequence encoding and vectorized functions. If
                None (default), numba is used automatically when installed (HAS_NUMBA); the paths are bit-identical.
            max_quant_its (int): Maximum coordinate-ascent iterations per theta starting point in
                the M-step.
            split_collapsed (bool): After each M-step, check for state pairs whose quantized
                parameters differ by less than split_nats everywhere and push them apart along the
                strongest raw expected-count asymmetry. Quantization otherwise rounds
                nearly-symmetric states onto (nearly) identical grid points, leaving EM at an
                exact fixed point that the raw counts say is escapable; an unwarranted split is
                rounded back by the next M-step.
            split_nats (float): Collapse tolerance and target separation (in nats of per-cell
                log-probability difference) for split_collapsed. Defaults to log(2).

        Attributes:
            num_states (int): Number of hidden states.
            levels (Optional[List[Any]]): Additional emission support values.
            pseudo_count (Optional[float]): Per-cell pseudo count.
            k_max (Optional[int]): Cap on the integer exponents.
            fixed_theta (Optional[float]): Fixed shared base, or None to optimize it.
            init_mode (str): Initial-state parameterization mode.
            len_estimator (ParameterEstimator): ParameterEstimator for the length distribution.
            name (Optional[str]): Optional name assigned to estimated distributions.
            keys (Tuple[Optional[str], Optional[str], Optional[str]]): Keys for initial states,
                transition counts, and emission accumulators.
            use_numba (bool): If True, Numba is used for sequence encoding.
            max_quant_its (int): Maximum coordinate-ascent iterations per starting point.
            split_collapsed (bool): If True, separate nearly-collapsed state pairs after each
                M-step.
            split_nats (float): Collapse tolerance and target separation in nats.

        """
        if init_mode not in ("quantized", "stationary"):
            raise ValueError("init_mode must be 'quantized' or 'stationary'.")
        if fixed_theta is not None and not (0.0 < fixed_theta < 1.0):
            raise ValueError("fixed_theta must be in (0, 1).")
        if k_max is not None and k_max < 1:
            raise ValueError("k_max must be a positive integer.")

        self.num_states = num_states
        self.levels = None if levels is None else list(levels)
        self.pseudo_count = pseudo_count
        self.k_max = k_max
        self.fixed_theta = fixed_theta
        self.init_mode = init_mode
        self.len_estimator = len_estimator if len_estimator is not None else NullEstimator()
        self.name = name
        self.keys = keys if keys is not None else (None, None, None)
        self.use_numba = HAS_NUMBA if use_numba is None else use_numba
        self.max_quant_its = max_quant_its
        self.split_collapsed = split_collapsed
        self.split_nats = split_nats

    def accumulator_factory(self) -> "HiddenMarkovAccumulatorFactory":
        """Returns a HiddenMarkovAccumulatorFactory with categorical emission accumulators."""
        est_factories = [CategoricalEstimator().accumulator_factory() for _ in range(self.num_states)]
        len_factory = self.len_estimator.accumulator_factory()

        return HiddenMarkovAccumulatorFactory(est_factories, len_factory, self.use_numba, self.keys, self.name)

    def estimate(
        self,
        nobs: float | None,
        suff_stat: tuple[int, np.ndarray, np.ndarray, np.ndarray, Sequence[dict[Any, float]], Any | None],
    ) -> "QuantizedHiddenMarkovModelDistribution":
        """Estimate a QuantizedHiddenMarkovModelDistribution from Baum-Welch expected counts.

        Sufficient statistics in arg 'suff_stat' are the HiddenMarkovAccumulator value:
            suff_stat[0] (int): Number of hidden states.
            suff_stat[1] (np.ndarray): Initial state counts.
            suff_stat[2] (np.ndarray): State counts.
            suff_stat[3] (np.ndarray): State transition counts.
            suff_stat[4] (Sequence[Dict[Any, float]]): Per-state categorical emission counts.
            suff_stat[5] (Optional[Any]): Optional sufficient statistics of the length distribution.

        Args:
            nobs (Optional[float]): Number of observations used in estimation.
            suff_stat: See above for details.

        Returns:
            QuantizedHiddenMarkovModelDistribution object.

        """
        num_states, init_counts, state_counts, trans_counts, topic_ss, len_ss = suff_stat

        len_dist = self.len_estimator.estimate(nobs, len_ss)

        vocab = set() if self.levels is None else set(self.levels)
        for state_counts_map in topic_ss:
            vocab.update(state_counts_map.keys())

        if len(vocab) == 0:
            raise ValueError(
                "QuantizedHiddenMarkovEstimator.estimate() requires observed emission values or estimator levels."
            )

        try:
            levels = sorted(vocab)
        except TypeError:
            levels = sorted(vocab, key=str)
        level_index = {v: i for i, v in enumerate(levels)}

        emit_counts = np.zeros((num_states, len(levels)), dtype=np.float64)
        for i, state_counts_map in enumerate(topic_ss):
            for v, cnt in state_counts_map.items():
                emit_counts[i, level_index[v]] += cnt

        init_counts = np.asarray(init_counts, dtype=np.float64).copy()
        trans_counts = np.asarray(trans_counts, dtype=np.float64).copy()

        if self.pseudo_count is not None and self.pseudo_count > 0:
            init_counts += self.pseudo_count
            trans_counts += self.pseudo_count
            emit_counts += self.pseudo_count

        count_blocks = [trans_counts, emit_counts]
        if self.init_mode == "quantized":
            count_blocks.append(init_counts[None, :])

        theta, exp_blocks = _fit_quantized_parameters(count_blocks, self.fixed_theta, self.k_max, self.max_quant_its)

        if self.split_collapsed and num_states > 1:
            _split_collapsed_states(
                exp_blocks[0], exp_blocks[1], trans_counts, emit_counts, self.k_max, math.log(theta), self.split_nats
            )

        initial_exponents = exp_blocks[2][0, :] if self.init_mode == "quantized" else None

        return QuantizedHiddenMarkovModelDistribution(
            theta,
            levels,
            exp_blocks[0],
            exp_blocks[1],
            initial_exponents=initial_exponents,
            init_mode=self.init_mode,
            k_max=self.k_max,
            len_dist=len_dist,
            name=self.name,
            use_numba=self.use_numba,
        )


# --- Backward-compatible API naming aliases ---
QuantizedHiddenMarkovModelEstimator = QuantizedHiddenMarkovEstimator
