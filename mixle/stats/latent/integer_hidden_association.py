"""Integer hidden association models over grouped-count word sets.

An observation is a pair ``(previous_counts, emitted_counts)`` where both sides
are sparse integer grouped counts. The model is a low-rank hidden association:
words in the emitted set are generated through hidden states conditioned on the
words in the previous set.

For previous words ``S1`` and emitted words ``S2``, ``cond_weights`` models
``p(state | word_in_S1)`` and ``state_prob_mat`` models
``p(word_in_S2 | state)``. The ``alpha`` parameter mixes this learned
association with a uniform background over emitted values.

The grouped-count representation keeps repeated integer words compact while
preserving the same density as an expanded iid sequence with counts.
"""

import math
from collections.abc import Sequence
from typing import Any, TypeVar

import numpy as np

from mixle.capability import Neutral, supports
from mixle.engines.arithmetic import *
from mixle.engines.arithmetic import maxrandint
from mixle.enumeration.algorithms import BufferedStream, frontier_merge
from mixle.stats.combinator.null_dist import (
    NullAccumulator,
    NullAccumulatorFactory,
    NullDistribution,
    NullEstimator,
)
from mixle.stats.compute.pdist import (
    DataSequenceEncoder,
    DistributionEnumerator,
    DistributionSampler,
    EnumerationError,
    ParameterEstimator,
    SequenceEncodableProbabilityDistribution,
    SequenceEncodableStatisticAccumulator,
    StatisticAccumulatorFactory,
    child_enumerator,
)
from mixle.stats.latent.integer_probabilistic_latent_semantic_indexing import multinomial_bag_stream
from mixle.utils.optional_deps import HAS_NUMBA, numba
from mixle.utils.optsutil import count_by_value

E0 = tuple[tuple[list[tuple[np.ndarray, ...]], Any | None, Any | None], None]

E1 = TypeVar("E1")  # Encoded prev
E2 = TypeVar("E2")  # Encoded lengths
E3 = tuple[tuple[list[tuple[np.ndarray, ...]], E2 | None, E1 | None], None]
E4 = tuple[None, tuple[tuple[np.ndarray, ...], E1 | None, E2 | None]]
E = E3 | E4

SS1 = TypeVar("SS1")  # suff stat prev
SS2 = TypeVar("SS2")  # suff stat len


class IntegerHiddenAssociationDistribution(SequenceEncodableProbabilityDistribution):
    """Integer hidden association model: words of a second set are emitted through hidden states conditioned
    on words of a first set."""

    def __init__(
        self,
        state_prob_mat: list[list[float]] | np.ndarray,
        cond_weights: list[list[float]] | np.ndarray,
        alpha: float = 0.0,
        prev_dist: SequenceEncodableProbabilityDistribution | None = NullDistribution(),
        len_dist: SequenceEncodableProbabilityDistribution | None = NullDistribution(),
        name: str | None = None,
        keys: tuple[str | None, str | None] = (None, None),
        use_numba: bool | None = None,
    ) -> None:
        """Create an integer hidden association distribution.

        Args:
            state_prob_mat: Emitted-word probabilities by hidden state.
            cond_weights: Hidden-state probabilities by previous-set word.
            alpha: Uniform-background mixing probability.
            prev_dist: Optional distribution for previous-set grouped counts.
            len_dist: Optional distribution for emitted-set length.
            name: Optional diagnostic name.
            keys: Optional merge keys for weight and state counts.
            use_numba: Whether to use Numba encodings and vectorized kernels.

        Attributes:
            cond_weights: Hidden-state probabilities by previous-set word.
            state_prob_mat: Emitted-word probabilities by hidden state.
            len_dist: Distribution for emitted-set length.
            prev_dist: Distribution for previous-set grouped counts.
            has_prev_dist: Whether ``prev_dist`` is non-null.
            num_vals2: Number of emitted-set values.
            num_vals1: Number of previous-set values.
            num_states: Number of hidden states.
            alpha: Uniform-background mixing probability.
            name: Optional diagnostic name.
            keys: Optional sufficient-statistic merge keys.
            init_prob_vec: Uniform background probability vector.
            use_numba: Whether Numba encoding/kernels are enabled.

        """
        self.cond_weights = np.asarray(cond_weights, dtype=np.float64)
        self.state_prob_mat = np.asarray(state_prob_mat, dtype=np.float64)
        self.len_dist = len_dist if len_dist is not None else NullDistribution()
        self.prev_dist = prev_dist if prev_dist is not None else NullDistribution()
        self.has_prev_dist = not supports(self.prev_dist, Neutral)
        self.num_vals2 = self.state_prob_mat.shape[1]
        self.num_vals1 = self.cond_weights.shape[0]
        self.num_states = self.state_prob_mat.shape[0]
        self.alpha = alpha
        self.name = name
        self.keys = keys
        self.init_prob_vec = np.empty(0, dtype=np.float64)
        self.use_numba = HAS_NUMBA if use_numba is None else use_numba

    def compute_capabilities(self):
        """Return backend capability metadata for this concrete integer association model."""
        from mixle.stats.compute.capabilities import DistributionCapabilities, intersect_engine_ready

        if self.use_numba:
            return DistributionCapabilities(engine_ready=("numpy",), kernel_status="legacy_numpy")
        return DistributionCapabilities(
            engine_ready=intersect_engine_ready((self.prev_dist, self.len_dist)), kernel_status="generic_latent"
        )

    def compute_declaration(self):
        """Return the generated-compute declaration for integer hidden association."""
        from mixle.stats.compute.declarations import (
            DistributionDeclaration,
            ParameterSpec,
            StatisticSpec,
            declaration_for,
        )

        previous = None if supports(self.prev_dist, Neutral) else declaration_for(self.prev_dist)
        length = None if supports(self.len_dist, Neutral) else declaration_for(self.len_dist)
        children = tuple(child for child in (previous, length) if child is not None)
        roles = ()
        if previous is not None:
            roles += ("previous",)
        if length is not None:
            roles += ("length",)
        return DistributionDeclaration(
            name="integer_hidden_association",
            distribution_type=type(self),
            parameters=(
                ParameterSpec("state_prob_mat", constraint="row_simplex_matrix"),
                ParameterSpec("cond_weights", constraint="row_simplex_matrix"),
                ParameterSpec("alpha", constraint="unit_interval"),
            ),
            statistics=(
                StatisticSpec("initial_counts"),
                StatisticSpec("weight_counts"),
                StatisticSpec("state_counts"),
                StatisticSpec("previous", kind="child_stat"),
                StatisticSpec("length", kind="child_stat"),
            ),
            support="integer_hidden_association_grouped_counts",
            children=children,
            child_roles=roles,
            differentiable=False,
        )

    def __str__(self) -> str:
        """Return a readable distribution summary."""
        s1 = ",".join(
            ["[" + ",".join(map(str, self.state_prob_mat[i, :])) + "]" for i in range(len(self.state_prob_mat))]
        )
        s2 = ",".join(["[" + ",".join(map(str, self.cond_weights[i, :])) + "]" for i in range(len(self.cond_weights))])
        s3 = str(self.alpha)
        s4 = repr(self.prev_dist) if self.prev_dist is None else str(self.prev_dist)
        s5 = str(self.len_dist)
        s6 = repr(self.name)
        s7 = repr(self.keys)

        return (
            "IntegerHiddenAssociationDistribution([%s], [%s], alpha=%s, prev_dist=%s, len_dist=%s, name=%s, "
            "keys=%s)" % (s1, s2, s3, s4, s5, s6, s7)
        )

    def log_density(self, x: tuple[list[tuple[int, float]], list[tuple[int, float]]]) -> float:
        """Log-density of the integer hidden association model at observation x.

        For each emitted word in x[1], marginalizes over the given words in x[0] (weighted by count)
        and the hidden states, mixing with a uniform density with probability alpha. Adds the
        log-density of x[0] under prev_dist and of the total emission count under len_dist.

        Args:
            x (Tuple[List[Tuple[int, float]], List[Tuple[int, float]]]): Grouped-count observation
                ([(S1 word, count)], [(S2 word, count)]).

        Returns:
            Log-density at observation x.

        """
        nw = self.num_vals2
        a = self.alpha / nw
        b = 1 - self.alpha

        cx = np.asarray([u[1] for u in x[0]], dtype=float)
        vx = np.asarray([u[0] for u in x[0]], dtype=int)
        cy = np.asarray([u[1] for u in x[1]], dtype=float)
        vy = np.asarray([u[0] for u in x[1]], dtype=int)

        n1 = np.sum(cx)
        n2 = np.sum(cy)

        ll = self.cond_weights[vx, :].T * (cx / np.sum(cx))
        ll = np.dot(ll.T, self.state_prob_mat[:, vy])
        with np.errstate(divide="ignore"):
            log_sum_x = np.log(b * np.sum(ll, axis=0) + a)
        rv = float(np.dot(log_sum_x, cy))
        # rv += np.dot(np.log(self.init_prob_vec[vx]), cx)

        rv += self.prev_dist.log_density(x[0])
        rv += self.len_dist.log_density(n2)

        return rv

    def seq_log_density(self, x: E) -> np.ndarray:
        """Vectorized evaluation of log-density at sequence encoded input x.

        Args:
            x (E): Sequence encoded observations from IntegerHiddenAssociationDataEncoder.seq_encode().
                Uses the numba kernel when the encoding was produced with use_numba=True.

        Returns:
            Numpy array of log-density values, one per encoded observation.

        """
        nw = self.num_vals2
        a = self.alpha / nw
        b = 1 - self.alpha

        if x[1] is None:
            xx = x[0]
            rv = np.zeros(len(xx[0]), dtype=np.float64)

            for i, entry in enumerate(xx[0]):
                vx, cx, vy, cy = entry

                x_mat = self.cond_weights[vx, :].T * (cx / np.sum(cx))
                x_mat = np.dot(x_mat.T, self.state_prob_mat[:, vy])
                with np.errstate(divide="ignore"):
                    rv[i] = np.dot(np.log(b * np.sum(x_mat, axis=0) + a), cy)
                # rv[i] += np.dot(np.log(self.init_prob_vec[vx]), cx)

            rv += self.prev_dist.seq_log_density(xx[1])
            rv += self.len_dist.seq_log_density(xx[2])

        else:
            (s0, s1, x0, x1, c0, c1, w0), xv, nn = x[1]

            rv = np.zeros(len(s0), dtype=np.float64)
            t0 = np.concatenate([[0], s0]).cumsum().astype(np.int32)
            t1 = np.concatenate([[0], s1]).cumsum().astype(np.int32)
            max_len = s0.max()
            numba_seq_log_density(
                self.num_states,
                max_len,
                t0,
                t1,
                x0,
                x1,
                c0,
                c1,
                w0,
                self.cond_weights,
                self.state_prob_mat,
                self.init_prob_vec,
                a,
                b,
                rv,
            )

            rv += self.prev_dist.seq_log_density(xv)
            rv += self.len_dist.seq_log_density(nn)

        return rv

    def backend_seq_log_density(self, x: E, engine: Any) -> Any:
        """Evaluate encoded log-densities using a backend-neutral compute engine."""
        from mixle.stats.compute.backend import BackendScoringError, backend_seq_log_density

        nw = self.num_vals2
        a = self.alpha / nw
        b = 1 - self.alpha

        if x[1] is not None:
            if getattr(engine, "name", None) == "numpy":
                return self.seq_log_density(x)
            raise BackendScoringError("IntegerHiddenAssociation numba-encoded scoring is NumPy-only.")

        entries, prev_enc, len_enc = x[0]
        if not entries:
            rv = engine.zeros(0)
        else:
            cond_weights = engine.asarray(self.cond_weights)
            state_probs = engine.asarray(self.state_prob_mat)
            scores = []

            for vx, cx, vy, cy in entries:
                given_ids = engine.asarray(vx)
                given_counts = engine.asarray(cx)
                emitted_ids = engine.asarray(vy)
                emitted_counts = engine.asarray(cy)

                given_weights = given_counts / engine.sum(given_counts)
                given_state = cond_weights[given_ids, :] * given_weights[:, None]
                emitted_given = engine.matmul(given_state, state_probs[:, emitted_ids])
                emitted_probs = b * engine.sum(emitted_given, axis=0) + a
                scores.append(engine.sum(engine.log(emitted_probs) * emitted_counts))

            rv = engine.stack(scores)

        rv = rv + backend_seq_log_density(self.prev_dist, prev_enc, engine)
        rv = rv + backend_seq_log_density(self.len_dist, len_enc, engine)
        return rv

    def conditional_word_log_probs(self, s1: list[tuple[int, float]]) -> np.ndarray | None:
        """Log of the per-emission word distribution ``q(.|S1)`` for a given S1 bag, or None if empty.

        ``q(w|S1) = (1-alpha) * sum_u (c_u/n1) * sum_s cond_weights[u,s] * state_prob_mat[s,w] + alpha/W``
        -- the smoothed mixture the model uses to score each emitted word. Returns None for an empty S1
        (``n1 = 0``), whose conditional is degenerate (the model's own density is undefined there).
        """
        if not s1:
            return None
        vx = np.asarray([u[0] for u in s1], dtype=int)
        cx = np.asarray([u[1] for u in s1], dtype=float)
        n1 = float(cx.sum())
        if n1 <= 0.0:
            return None
        a = self.alpha / self.num_vals2
        b = 1.0 - self.alpha
        state_weight = (self.cond_weights[vx, :] * (cx / n1)[:, None]).sum(axis=0)  # (num_states,)
        q = b * (state_weight @ self.state_prob_mat) + a  # (num_vals2,)
        with np.errstate(divide="ignore"):
            return np.log(q)

    def enumerator(self) -> DistributionEnumerator:
        """Enumerate ``(S1, S2)`` observations in descending probability order.

        The model factors as ``prev_dist(S1) * [prod_w q(w|S1)^{c_w}] * P_len(n2)``: the emitted bag S2
        is a trial-count multinomial whose word distribution ``q(.|S1)`` depends on the given bag S1.
        Enumeration is a conditional product -- the outer stream enumerates S1 from ``prev_dist`` and,
        for each S1, the inner stream enumerates S2 by the multinomial bag search under ``len_dist``,
        merged by descending total score with ``prev_dist(S1)`` as the outer frontier bound. Requires an
        enumerable, non-null ``prev_dist`` so the S1 support is defined.
        """
        if not self.has_prev_dist:
            raise EnumerationError(self, reason="enumeration requires a non-null prev_dist over the S1 bags")
        return IntegerHiddenAssociationEnumerator(self)

    def sampler(self, seed: int | None = None) -> "IntegerHiddenAssociationSampler":
        """Return a sampler for iid integer hidden-association observations.

        Requires non-null prev_dist and len_dist.

        Args:
            seed: Optional random seed.

        Returns:
            A configured integer hidden-association sampler.

        """
        if supports(self.prev_dist, Neutral):
            raise ValueError("HiddenAssociationSampler requires attribute dist.prev_dist.")
        if supports(self.len_dist, Neutral):
            raise ValueError("HiddenAssociationSampler requires attribute dist.size_dist.")
        return IntegerHiddenAssociationSampler(self, seed)

    def estimator(self, pseudo_count: float | None = None) -> "IntegerHiddenAssociationEstimator":
        """Return an estimator with matching dimensions and child estimators.

        Args:
            pseudo_count: Unused; accepted for protocol consistency.

        Returns:
            A configured integer hidden-association estimator.

        """
        n_vals = (self.num_vals1, self.num_vals2)
        prev_est = self.prev_dist.estimator()
        len_est = self.len_dist.estimator()

        return IntegerHiddenAssociationEstimator(
            num_vals=n_vals,
            num_states=self.num_states,
            alpha=self.alpha,
            prev_estimator=prev_est,
            len_estimator=len_est,
            name=self.name,
            keys=self.keys,
            use_numba=self.use_numba,
        )

    def dist_to_encoder(self) -> "IntegerHiddenAssociationDataEncoder":
        """Return an encoder for integer hidden-association observations."""
        prev_encoder = self.prev_dist.dist_to_encoder()
        len_encoder = self.len_dist.dist_to_encoder()
        return IntegerHiddenAssociationDataEncoder(prev_encoder, len_encoder, self.use_numba)


class IntegerHiddenAssociationEnumerator(DistributionEnumerator):
    """Enumerate integer hidden-association support pairs in best-first order."""

    def __init__(self, dist: "IntegerHiddenAssociationDistribution") -> None:
        """Conditional-product enumeration of ``(S1, S2)`` (S1 from prev_dist, S2 multinomial given S1).

        Args:
            dist (IntegerHiddenAssociationDistribution): Distribution whose support is enumerated.
        """
        super().__init__(dist)
        len_dist = dist.len_dist

        def make_inner(s1, lp1):
            log_q = dist.conditional_word_log_probs(s1)
            if log_q is None:
                return iter(())  # degenerate empty S1: the model density itself is undefined there

            def combine(pairs, s1=s1):
                return (s1, [(int(w), int(c)) for w, c in sorted(pairs)])

            return ((value, lp1 + lp2) for value, lp2 in multinomial_bag_stream(log_q, 0, len_dist, combine))

        outer = BufferedStream(child_enumerator(dist.prev_dist, "IntegerHiddenAssociationDistribution.prev_dist"))
        self._merge = frontier_merge(outer, make_inner)

    def __next__(self):
        return next(self._merge)


class IntegerHiddenAssociationSampler(DistributionSampler):
    """Sampler for grouped-count word-set pairs from an integer hidden-association distribution."""

    def __init__(self, dist: IntegerHiddenAssociationDistribution, seed: int | None = None) -> None:
        """Create a sampler for an integer hidden-association distribution.

        Args:
            dist (IntegerHiddenAssociationDistribution): Distribution to sample from. Must have non-null
                ``prev_dist`` and ``len_dist``.
            seed (Optional[int]): Seed for random number generator.

        Attributes:
            rng (RandomState): Random state initialized from ``seed`` when supplied.
            dist (IntegerHiddenAssociationDistribution): Distribution to sample from.
            prev_sampler (DistributionSampler): Sampler for the previous word set.
            size_sampler (DistributionSampler): Sampler for the number of emitted words.

        """
        self.rng = np.random.RandomState(seed)
        self.dist = dist

        if supports(self.dist.prev_dist, Neutral):
            raise ValueError("HiddenAssociationSampler requires attribute dist.prev_dist.")
        else:
            self.prev_sampler = self.dist.prev_dist.sampler(seed=self.rng.randint(0, maxrandint))

        if supports(self.dist.len_dist, Neutral):
            raise ValueError("HiddenAssociationSampler requires attribute dist.size_dist.")
        else:
            self.size_sampler = self.dist.len_dist.sampler(seed=self.rng.randint(0, maxrandint))

    def sample_given(self, x: list[tuple[int, float]]) -> list[tuple[int, float]]:
        """Draw an emitted grouped-count word set conditioned on the given word set x.

        Args:
            x (List[Tuple[int, float]]): Given word set as (word, count) pairs.

        Returns:
            List of (emitted word, count) pairs.

        """
        slen = self.size_sampler.sample()
        rng = np.random.RandomState(self.rng.randint(0, maxrandint))

        x0 = np.asarray([xx[0] for xx in x])
        x1 = np.asarray([xx[1] for xx in x], dtype=float)
        s1 = np.sum(x1)

        if s1 > 0:
            x1 /= s1
        else:
            return []

        v2 = []
        z1 = rng.choice(len(x0), p=x1, replace=True, size=slen)
        ns = self.dist.num_states
        nw = self.dist.num_vals2

        for zz1 in z1:
            if rng.rand() >= self.dist.alpha:
                u = rng.choice(ns, p=self.dist.cond_weights[x0[zz1], :])
                v2.append(rng.choice(nw, p=self.dist.state_prob_mat[u, :]))
            else:
                v2.append(rng.choice(nw))

        return list(count_by_value(v2).items())

    def sample(self, size: int | None = None) -> Sequence[list[tuple[int, float]]] | list[tuple[int, float]]:
        """Draw iid grouped-count observations from the integer hidden association model.

        Args:
            size (Optional[int]): Number of observations to draw. If None, a single observation is returned.

        Returns:
            A ([(S1 word, count)], [(S2 word, count)]) tuple if size is None, else a list of such tuples
            of length size.

        """
        if size is None:
            x = self.prev_sampler.sample()
            return x, self.sample_given(x)
        else:
            return [self.sample() for i in range(size)]


class IntegerHiddenAssociationAccumulator(SequenceEncodableStatisticAccumulator):
    """Accumulator for hidden-state and emission counts from observed word-set pairs."""

    def __init__(
        self,
        num_vals1: int,
        num_vals2: int,
        num_states: int,
        prev_acc: SequenceEncodableStatisticAccumulator | None = NullAccumulator(),
        size_acc: SequenceEncodableStatisticAccumulator | None = NullAccumulator(),
        use_numba: bool | None = None,
        keys: tuple[str | None, str | None] | None = (None, None),
    ) -> None:
        """Create an accumulator for integer hidden-association sufficient statistics.

        Args:
            num_vals1 (int): Number of words in S1.
            num_vals2 (int): Number of words in S2.
            num_states (int): Number of hidden states.
            prev_acc (Optional[SequenceEncodableStatisticAccumulator]): Accumulator for the previous word set.
            size_acc (Optional[SequenceEncodableStatisticAccumulator]): Accumulator for the emission count.
            use_numba (Optional[bool]): Whether numba encodings are used for vectorized updates.
                ``None`` (default) selects the compiled path when numba is installed.
            keys (Optional[Tuple[Optional[str], Optional[str]]]): Keys for the weight and state counts.

        Attributes:
            init_count (np.ndarray): Weighted counts of S1 words.
            weight_count (np.ndarray): num_vals1 by num_states matrix of weighted state counts per S1 word.
            state_count (np.ndarray): num_states by num_vals2 matrix of weighted emission counts per state.
            size_accumulator (SequenceEncodableStatisticAccumulator): Accumulator for the emission count.
            prev_accumulator (SequenceEncodableStatisticAccumulator): Accumulator for the previous word set.
            num_vals1 (int): Number of words in S1.
            num_vals2 (int): Number of words in S2.
            num_states (int): Number of hidden states.
            use_numba (Optional[bool]): Whether numba encodings are used for vectorized updates.
                ``None`` (default) selects the compiled path when numba is installed.
            weight_key (Optional[str]): Key for merging weight counts.
            state_key (Optional[str]): Key for merging state counts.

        """
        self.init_count = np.zeros(num_vals1, dtype=np.float64)
        self.weight_count = np.zeros((num_vals1, num_states), dtype=np.float64)
        self.state_count = np.zeros((num_states, num_vals2), dtype=np.float64)
        self.size_accumulator = size_acc if size_acc is not None else NullAccumulator()
        self.prev_accumulator = prev_acc if prev_acc is not None else NullAccumulator()
        self.num_vals1 = num_vals1
        self.num_vals2 = num_vals2
        self.num_states = num_states
        self.use_numba = HAS_NUMBA if use_numba is None else use_numba
        self.weight_key, self.state_key = keys if keys is not None else (None, None)

        # Data log-likelihood accumulated as a byproduct of the E-step (the per-observation log_density),
        # only when _track_ll is enabled. Used by the fused-EM fast path in
        # optimize(reuse_estep_ll=True); not part of value(). Off by default so the standard path pays
        # nothing. Only the pure (non-numba) blocked branch reports it; the numba branch sets it to
        # None to force the caller's separate-scoring fallback.
        self._track_ll = False
        self._seq_ll = 0.0

        self._init_rng = False
        self._rng_prev = None
        self._rng_size = None
        self._rng_weight = None
        self._rng_state = None

    def update(
        self,
        x: tuple[list[tuple[int, float]], list[tuple[int, float]]],
        weight: float,
        estimate: IntegerHiddenAssociationDistribution,
    ) -> None:
        """Update sufficient statistics with posterior word/state assignments for the observation.

        Args:
            x (Tuple[List[Tuple[int, float]], List[Tuple[int, float]]]): Grouped-count observation
                ([(S1 word, count)], [(S2 word, count)]).
            weight (float): Weight for the observation.
            estimate (IntegerHiddenAssociationDistribution): Previous estimate used to compute posteriors.

        """
        vx = np.asarray([u[0] for u in x[0]], dtype=int)
        cx = np.asarray([u[1] for u in x[0]], dtype=float)
        vy = np.asarray([u[0] for u in x[1]], dtype=int)
        cy = np.asarray([u[1] for u in x[1]], dtype=float)
        nx = np.sum(cx)

        a = estimate.alpha / estimate.num_vals2
        b = 1 - estimate.alpha

        x_mat = (estimate.cond_weights[vx, :].T * (cx / nx)).T
        y_mat = estimate.state_prob_mat[:, vy]
        z_mat = x_mat[:, :, None] * y_mat[None, :, :]

        # [old word] x [state] x [new word]

        ss = np.sum(np.sum(z_mat, axis=0, keepdims=True), axis=1, keepdims=True)
        denom = ss * b + a
        scale = np.zeros_like(denom)
        np.divide(b, denom, out=scale, where=denom > 0.0)
        z_mat *= scale

        self.weight_count[vx, :] += np.dot(z_mat, cy) * weight
        self.state_count[:, vy] += np.sum(z_mat, axis=0) * cy * weight
        self.init_count[vx] += cx * weight

        self.prev_accumulator.update(x[0], weight, None if estimate is None else estimate.prev_dist)
        self.size_accumulator.update(cy.sum(), weight, None if estimate is None else estimate.len_dist)

    def _rng_initialize(self, rng: np.random.RandomState) -> None:
        if not self._init_rng:
            seeds = rng.randint(low=0, high=maxrandint, size=4)
            self._rng_state = np.random.RandomState(seed=seeds[0])
            self._rng_weight = np.random.RandomState(seed=seeds[1])
            self._rng_prev = np.random.RandomState(seed=seeds[2])
            self._rng_size = np.random.RandomState(seed=seeds[3])
            self._init_rng = True

    def initialize(
        self, x: tuple[list[tuple[int, float]], list[tuple[int, float]]], weight: float, rng: np.random.RandomState
    ) -> None:
        """Initialize sufficient statistics with random (Dirichlet) state assignments.

        Args:
            x (Tuple[List[Tuple[int, float]], List[Tuple[int, float]]]): Grouped-count observation
                ([(S1 word, count)], [(S2 word, count)]).
            weight (float): Weight for the observation.
            rng (np.random.RandomState): Random number generator for the random assignments.

        """
        if not self._init_rng:
            self._rng_initialize(rng)

        vx = np.asarray([u[0] for u in x[0]], dtype=int)
        cx = np.asarray([u[1] for u in x[0]], dtype=float)
        vy = np.asarray([u[0] for u in x[1]], dtype=int)
        cy = np.asarray([u[1] for u in x[1]], dtype=float)

        self.weight_count[vx, :] += self._rng_weight.dirichlet(np.ones(self.num_states), size=len(vx)) * weight
        self.state_count[:, vy] += self._rng_state.dirichlet(np.ones(self.num_states), size=len(vy)).T * cy * weight
        self.init_count[vx] += cx * weight

        self.prev_accumulator.initialize(x[0], weight, self._rng_prev)
        self.size_accumulator.initialize(cy.sum(), weight, self._rng_size)

    def seq_initialize(self, x: E, weights: np.ndarray, rng: np.random.RandomState) -> None:
        """Vectorized initialization of sufficient statistics from sequence encoded observations.

        Args:
            x (E): Sequence encoded observations from IntegerHiddenAssociationDataEncoder.seq_encode().
            weights (np.ndarray): Weights, one per encoded observation.
            rng (np.random.RandomState): Random number generator for the random assignments.

        """
        if not self._init_rng:
            self._rng_initialize(rng)

        if x[1] is None:
            xx = x[0]

            for i, (entry, weight) in enumerate(zip(xx[0], weights)):
                vx, cx, vy, cy = entry

                self.weight_count[vx, :] += self._rng_weight.dirichlet(np.ones(self.num_states), size=len(vx)) * weight
                self.state_count[:, vy] += (
                    self._rng_state.dirichlet(np.ones(self.num_states), size=len(vy)).T * cy * weight
                )
                self.init_count[vx] += cx * weight

            self.prev_accumulator.seq_initialize(xx[1], weights, self._rng_prev)
            self.size_accumulator.seq_initialize(xx[2], weights, self._rng_size)

        else:
            (s0, s1, x0, x1, c0, c1, w0), xv, nn = x[1]
            weights_0 = []
            weights_1 = []

            for i in range(len(s0)):
                weights_0.extend([weights[i]] * s0[i] * self.num_states)
                weights_1.extend([weights[i]] * s1[i])

            weights_0 = np.asarray(weights_0)
            weights_1 = np.asarray(weights_1)
            ww0 = self._rng_weight.dirichlet(np.ones(self.num_states), size=len(x0)).flatten() * weights_0
            ww0 = np.reshape(ww0, (len(x0), self.num_states))

            self.weight_count += vec_bincount1(x=x0, w=ww0, out=np.zeros_like(self.weight_count, dtype=np.float64))

            ww1 = self._rng_state.dirichlet(np.ones(self.num_states), size=len(x1)).T
            ww1 *= np.reshape(c1 * weights_1, (-1, len(x1)))

            self.state_count += vec_bincount2(x=x1, w=ww1, out=np.zeros_like(self.state_count, dtype=np.float64))

            self.init_count += np.bincount(
                x0,
                weights=c0 * weights_0[np.arange(0, len(weights_0), self.num_states)],
                minlength=len(self.init_count),
            )

            self.prev_accumulator.seq_initialize(xv, weights, self._rng_prev)
            self.size_accumulator.seq_initialize(nn, weights, self._rng_size)

    def seq_update(self, x: E, weights: np.ndarray, estimate: IntegerHiddenAssociationDistribution) -> None:
        """Vectorized update of sufficient statistics from sequence encoded observations.

        Args:
            x (E): Sequence encoded observations from IntegerHiddenAssociationDataEncoder.seq_encode().
                Uses the numba kernel when the encoding was produced with use_numba=True.
            weights (np.ndarray): Weights, one per encoded observation.
            estimate (IntegerHiddenAssociationDistribution): Previous estimate used to compute posteriors.

        """
        if x[1] is None:
            xx = x[0]
            a = estimate.alpha / estimate.num_vals2
            b = 1 - estimate.alpha
            track = self._track_ll
            obs_ll = np.zeros(len(xx[0]), dtype=np.float64) if track else None

            for i, (entry, weight) in enumerate(zip(xx[0], weights)):
                vx, cx, vy, cy = entry
                nx = np.sum(cx)
                x_mat = (estimate.cond_weights[vx, :].T * (cx / nx)).T
                y_mat = estimate.state_prob_mat[:, vy]
                z_mat = x_mat[:, :, None] * y_mat[None, :, :]

                # [old word] x [state] x [new word]

                ss = np.sum(np.sum(z_mat, axis=0, keepdims=True), axis=1, keepdims=True)
                denom = ss * b + a
                if track:
                    # Per-observation log-density (== IntegerHiddenAssociation.log_density assoc term),
                    # reusing ``denom`` (the per-emitted-word mixture mass) before it is consumed by the
                    # responsibility normalization below.
                    with np.errstate(divide="ignore"):
                        obs_ll[i] = float(np.dot(np.log(denom.reshape(-1)), cy))
                scale = np.zeros_like(denom)
                np.divide(b, denom, out=scale, where=denom > 0.0)
                z_mat *= scale

                self.weight_count[vx, :] += np.dot(z_mat, cy) * weight
                self.state_count[:, vy] += np.sum(z_mat, axis=0) * cy * weight
                self.init_count[vx] += cx * weight

            self.prev_accumulator.seq_update(xx[1], weights, None if estimate is None else estimate.prev_dist)
            self.size_accumulator.seq_update(xx[2], weights, None if estimate is None else estimate.len_dist)

            if track:
                obs_ll += estimate.prev_dist.seq_log_density(xx[1])
                obs_ll += estimate.len_dist.seq_log_density(xx[2])
                self._seq_ll += float(np.dot(np.asarray(weights, dtype=np.float64), obs_ll))
        else:
            if self._track_ll:
                # The numba kernel does not expose the per-observation normalizer; signal the
                # fused-EM caller to fall back to a separate scoring pass instead of reporting a
                # wrong value.
                self._seq_ll = None

            (s0, s1, x0, x1, c0, c1, w0), xv, nn = x[1]

            t0 = np.concatenate([[0], s0]).cumsum().astype(np.int32)
            t1 = np.concatenate([[0], s1]).cumsum().astype(np.int32)
            max_len = s0.max()

            a = estimate.alpha / estimate.num_vals2
            b = 1 - estimate.alpha

            numba_seq_update(
                self.num_states,
                max_len,
                t0,
                t1,
                x0,
                x1,
                c0,
                c1,
                w0,
                estimate.cond_weights,
                estimate.state_prob_mat,
                self.weight_count,
                self.state_count,
                self.init_count,
                weights,
                a,
                b,
            )

            self.prev_accumulator.seq_update(xv, weights, None if estimate is None else estimate.prev_dist)
            self.size_accumulator.seq_update(nn, weights, None if estimate is None else estimate.len_dist)

    def seq_update_engine(
        self, x: E, weights: np.ndarray, estimate: IntegerHiddenAssociationDistribution, engine: Any
    ) -> None:
        """Engine-resident E-step for the pure (non-numba) blocked encoding.

        Mirrors the numpy branch of ``seq_update``: for each observation the (given word x state x
        emitted word) responsibility tensor is built and normalized with the alpha smoothing on the
        active engine (numpy or torch), and the initial/weight/state counts are scattered into
        engine-resident accumulators via ``index_add``. Only the per-observation orchestration runs
        in Python; all tensor arithmetic and accumulation are on the engine.
        """
        if x[1] is not None or x[0] is None:
            # numba encoding -> defer to the host numba path
            self.seq_update(x, weights, estimate)
            return

        xx = x[0]
        weights_np = np.asarray(engine.to_numpy(weights) if hasattr(engine, "to_numpy") else weights, dtype=np.float64)

        num_states = estimate.num_states
        num_vals = estimate.cond_weights.shape[0]
        num_vals2 = estimate.state_prob_mat.shape[1]
        a = float(estimate.alpha) / estimate.num_vals2
        b = 1.0 - float(estimate.alpha)

        cond_weights = engine.asarray(estimate.cond_weights)  # (num_vals, S)
        state_prob_mat = engine.asarray(estimate.state_prob_mat)  # (S, num_vals2)
        weight_acc = engine.zeros((num_vals, num_states))
        state_acc_t = engine.zeros((num_vals2, num_states))  # transposed for axis-0 scatter
        init_acc = engine.zeros(num_vals)
        a_e = engine.asarray(a)
        b_e = engine.asarray(b)
        one = engine.asarray(1.0)
        zero = engine.asarray(0.0)

        for i, entry in enumerate(xx[0]):
            vx, cx, vy, cy = entry
            weight = float(weights_np[i])
            vx_e = engine.asarray(np.asarray(vx, dtype=np.int64))
            vy_e = engine.asarray(np.asarray(vy, dtype=np.int64))
            cx_e = engine.asarray(np.asarray(cx, dtype=np.float64))
            cy_e = engine.asarray(np.asarray(cy, dtype=np.float64))
            nx = engine.sum(cx_e)

            x_mat = cond_weights[vx_e, :] * (cx_e / nx).reshape((-1, 1))  # (gx, S)
            y_mat = state_prob_mat[:, vy_e]  # (S, gy)
            z = x_mat[:, :, None] * y_mat[None, :, :]  # (gx, S, gy)

            ss = engine.sum(engine.sum(z, axis=0), axis=0)  # (gy,)
            denom = ss * b_e + a_e  # (gy,)
            pos = denom > zero
            scale = engine.where(pos, b_e / engine.where(pos, denom, one), zero)  # (gy,)
            z = z * scale[None, None, :]

            wc_contrib = engine.sum(z * cy_e[None, None, :], axis=2)  # (gx, S)
            weight_acc = engine.index_add(weight_acc, vx_e, wc_contrib * engine.asarray(weight))

            sc_contrib = engine.sum(z, axis=0) * cy_e[None, :] * engine.asarray(weight)  # (S, gy)
            state_acc_t = engine.index_add(state_acc_t, vy_e, sc_contrib.T)  # (gy, S)

            init_acc = engine.index_add(init_acc, vx_e, cx_e * engine.asarray(weight))

        self.weight_count += np.asarray(engine.to_numpy(weight_acc))
        self.state_count += np.asarray(engine.to_numpy(state_acc_t)).T
        self.init_count += np.asarray(engine.to_numpy(init_acc))

        self.prev_accumulator.seq_update(xx[1], weights_np, None if estimate is None else estimate.prev_dist)
        self.size_accumulator.seq_update(xx[2], weights_np, None if estimate is None else estimate.len_dist)

    def combine(
        self, suff_stat: tuple[np.ndarray, np.ndarray, np.ndarray, SS1 | None, SS2 | None]
    ) -> "IntegerHiddenAssociationAccumulator":
        """Merge sufficient statistics of suff_stat into this accumulator.

        Args:
            suff_stat (Tuple[np.ndarray, np.ndarray, np.ndarray, Optional[SS1], Optional[SS2]]): Init counts,
                weight counts, state counts, prev suff stats, and size suff stats.

        Returns:
            This IntegerHiddenAssociationAccumulator.

        """
        init_count, weight_count, state_count, prev_acc, size_acc = suff_stat

        self.prev_accumulator.combine(prev_acc)
        self.size_accumulator.combine(size_acc)

        self.init_count += init_count
        self.weight_count += weight_count
        self.state_count += state_count

        return self

    def value(self) -> tuple[np.ndarray, np.ndarray, np.ndarray, Any | None, Any | None]:
        """Returns the sufficient statistics: (init counts, weight counts, state counts, prev, size)."""
        pval = self.prev_accumulator.value()
        sval = self.size_accumulator.value()

        return self.init_count, self.weight_count, self.state_count, pval, sval

    def from_value(
        self, x: tuple[np.ndarray, np.ndarray, np.ndarray, SS1 | None, SS2 | None]
    ) -> "IntegerHiddenAssociationAccumulator":
        """Set the sufficient statistics of this accumulator from x.

        Args:
            x (Tuple[np.ndarray, np.ndarray, np.ndarray, Optional[SS1], Optional[SS2]]): Init counts,
                weight counts, state counts, prev suff stats, and size suff stats.

        Returns:
            This IntegerHiddenAssociationAccumulator.

        """
        init_count, weight_count, state_count, prev_acc, size_acc = x

        self.init_count = init_count
        self.weight_count = weight_count
        self.state_count = state_count

        self.prev_accumulator.from_value(prev_acc)
        self.size_accumulator.from_value(size_acc)

        return self

    def scale(self, c: float) -> "IntegerHiddenAssociationAccumulator":
        """Scale linear association counts and delegate child accumulators."""
        self.init_count *= c
        self.weight_count *= c
        self.state_count *= c
        self.prev_accumulator.scale(c)
        self.size_accumulator.scale(c)
        return self

    def key_merge(self, stats_dict: dict[str, Any]) -> None:
        """Merge this accumulator's weight and state counts into stats_dict under their keys, if keyed.

        Args:
            stats_dict (Dict[str, Any]): Maps keys to merged sufficient statistics.

        """
        if self.weight_key is not None:
            if self.weight_key in stats_dict:
                stats_dict[self.weight_key] += self.weight_count
            else:
                stats_dict[self.weight_key] = self.weight_count.copy()

        if self.state_key is not None:
            if self.state_key in stats_dict:
                stats_dict[self.state_key] += self.state_count
            else:
                stats_dict[self.state_key] = self.state_count.copy()

        self.prev_accumulator.key_merge(stats_dict)
        self.size_accumulator.key_merge(stats_dict)

    def key_replace(self, stats_dict: dict[str, Any]) -> None:
        """Replace this accumulator's weight and state counts with the keyed statistics in stats_dict, if keyed.

        Args:
            stats_dict (Dict[str, Any]): Maps keys to merged sufficient statistics.

        """
        if self.weight_key is not None:
            if self.weight_key in stats_dict:
                self.weight_count = stats_dict[self.weight_key].copy()

        if self.state_key is not None:
            if self.state_key in stats_dict:
                self.state_count = stats_dict[self.state_key].copy()

        self.prev_accumulator.key_replace(stats_dict)
        self.size_accumulator.key_replace(stats_dict)

    def acc_to_encoder(self) -> "DataSequenceEncoder":
        """Return an encoder compatible with integer hidden-association observations."""
        prev_encoder = self.prev_accumulator.acc_to_encoder()
        len_encoder = self.size_accumulator.acc_to_encoder()
        return IntegerHiddenAssociationDataEncoder(prev_encoder, len_encoder, self.use_numba)


class IntegerHiddenAssociationAccumulatorFactory(StatisticAccumulatorFactory):
    """Factory for integer hidden-association accumulators."""

    def __init__(
        self,
        num_vals1: int,
        num_vals2: int,
        num_states: int,
        prev_factory: StatisticAccumulatorFactory | None = NullAccumulatorFactory(),
        len_factory: StatisticAccumulatorFactory | None = NullAccumulatorFactory(),
        use_numba: bool | None = None,
        keys: tuple[str | None, str | None] = (None, None),
    ) -> None:
        """Create an accumulator factory.

        Args:
            num_vals1: Number of previous-set values.
            num_vals2: Number of emitted-set values.
            num_states: Number of hidden states.
            prev_factory: Factory for the previous-set accumulator.
            len_factory: Factory for the emitted-count accumulator.
            use_numba: Whether Numba encodings are used for vectorized updates.
            keys: Optional merge keys for weight and state counts.

        Attributes:
            len_factory: Factory for emitted-count accumulation.
            prev_factory: Factory for previous-set accumulation.
            keys: Optional sufficient-statistic merge keys.
            use_numba: Whether Numba encodings are enabled.
            num_vals1: Number of previous-set values.
            num_vals2: Number of emitted-set values.
            num_states: Number of hidden states.

        """
        self.len_factory = len_factory if len_factory is not None else NullAccumulatorFactory()
        self.prev_factory = prev_factory if prev_factory is not None else NullAccumulatorFactory()
        self.keys = keys
        self.use_numba = HAS_NUMBA if use_numba is None else use_numba
        self.num_vals1 = num_vals1
        self.num_vals2 = num_vals2
        self.num_states = num_states

    def make(self) -> "IntegerHiddenAssociationAccumulator":
        """Return a fresh integer hidden-association accumulator."""
        len_acc = self.len_factory.make()
        prev_acc = self.prev_factory.make()
        return IntegerHiddenAssociationAccumulator(
            num_vals1=self.num_vals1,
            num_vals2=self.num_vals2,
            num_states=self.num_states,
            prev_acc=prev_acc,
            size_acc=len_acc,
            use_numba=self.use_numba,
            keys=self.keys,
        )


class IntegerHiddenAssociationEstimator(ParameterEstimator):
    """Estimator for integer hidden-association distributions."""

    def __init__(
        self,
        num_vals: list[int] | tuple[int, int] | int,
        num_states: int,
        alpha: float = 0.0,
        prev_estimator: ParameterEstimator | None = NullEstimator(),
        len_estimator: ParameterEstimator | None = NullEstimator(),
        suff_stat: Any | None = None,
        pseudo_count: float | None = None,
        use_numba: bool | None = None,
        name: str | None = None,
        keys: tuple[str | None, str | None] | None = (None, None),
    ) -> None:
        """Create an estimator for integer hidden-association sufficient statistics.

        Args:
            num_vals: Number of values in each set, either a shared integer or
                a two-entry ``(num_vals1, num_vals2)`` sequence.
            num_states: Number of hidden states.
            alpha: Uniform-background mixing probability.
            prev_estimator: Estimator for previous-set grouped counts.
            len_estimator: Estimator for emitted-set length.
            suff_stat: Optional prior sufficient statistics, accepted for protocol consistency.
            pseudo_count: Optional smoothing count, accepted for protocol consistency.
            use_numba: Whether to use Numba encodings. ``None`` enables it when installed.
            name: Optional diagnostic name.
            keys: Optional merge keys for weight and state counts.

        Attributes:
            num_vals: Original value-count specification.
            num_states: Number of hidden states.
            alpha: Uniform-background mixing probability.
            prev_estimator: Estimator for previous-set grouped counts.
            len_estimator: Estimator for emitted-set length.
            suff_stat: Optional prior sufficient statistics.
            pseudo_count: Optional smoothing count.
            use_numba: Whether Numba encoding/kernels are enabled.
            name: Optional diagnostic name.
            keys: Optional sufficient-statistic merge keys.
            num_vals1: Number of previous-set values.
            num_vals2: Number of emitted-set values.

        """
        self.prev_estimator = prev_estimator if prev_estimator is not None else NullEstimator()
        self.len_estimator = len_estimator if len_estimator is not None else NullEstimator()
        self.pseudo_count = pseudo_count
        self.suff_stat = suff_stat
        self.num_vals = num_vals
        self.num_states = num_states
        self.alpha = alpha
        self.use_numba = HAS_NUMBA if use_numba is None else use_numba
        self.name = name
        self.keys = keys if keys is not None else (None, None)

        if isinstance(num_vals, (tuple, list)):
            if len(num_vals) >= 2:
                self.num_vals1 = num_vals[0]
                self.num_vals2 = num_vals[1]
            elif len(num_vals) == 1:
                self.num_vals1 = num_vals[0]
                self.num_vals2 = num_vals[0]
        else:
            self.num_vals1 = num_vals
            self.num_vals2 = num_vals

    def accumulator_factory(self) -> "IntegerHiddenAssociationAccumulatorFactory":
        """Return an accumulator factory matching this estimator."""
        len_factory = self.len_estimator.accumulator_factory()
        prev_factory = self.prev_estimator.accumulator_factory()

        return IntegerHiddenAssociationAccumulatorFactory(
            self.num_vals1, self.num_vals2, self.num_states, prev_factory, len_factory, self.use_numba, self.keys
        )

    def estimate(
        self, nobs: float | None, suff_stat: tuple[np.ndarray, np.ndarray, np.ndarray, SS1 | None, SS2 | None]
    ) -> "IntegerHiddenAssociationDistribution":
        """Estimate an IntegerHiddenAssociationDistribution from aggregated sufficient statistics.

        Args:
            nobs (Optional[float]): Number of observations, passed to the prev and length estimators.
            suff_stat (Tuple[np.ndarray, np.ndarray, np.ndarray, Optional[SS1], Optional[SS2]]): Init counts,
                weight counts, state counts, prev suff stats, and size suff stats.

        Returns:
            IntegerHiddenAssociationDistribution object.

        """
        init_count, weight_count, state_count, prev_stats, size_stats = suff_stat

        len_dist = self.len_estimator.estimate(nobs, size_stats)
        prev_dist = self.prev_estimator.estimate(nobs, prev_stats)

        if self.pseudo_count is not None:
            init_count += self.pseudo_count / len(init_count)
            state_count += self.pseudo_count / (self.num_states * self.num_vals2)
            weight_count += self.pseudo_count / (self.num_states * self.num_vals1)

        # init_prob = init_count / np.sum(init_count)

        wsum = np.sum(weight_count, axis=1, keepdims=True)
        ssum = np.sum(state_count, axis=1, keepdims=True)
        ssum[ssum == 0] = 1.0
        wsum[wsum == 0] = 1.0

        weight_prob = weight_count / wsum
        state_prob = state_count / ssum

        # return IntegerHiddenAssociationDistribution(init_prob, state_prob, weight_prob, self.alpha, len_dist)
        return IntegerHiddenAssociationDistribution(
            state_prob_mat=state_prob,
            cond_weights=weight_prob,
            alpha=self.alpha,
            prev_dist=prev_dist,
            use_numba=self.use_numba,
            len_dist=len_dist,
            name=self.name,
            keys=self.keys,
        )


class IntegerHiddenAssociationDataEncoder(DataSequenceEncoder):
    """Encode grouped-count integer hidden-association observations."""

    def __init__(self, prev_encoder: DataSequenceEncoder, len_encoder: DataSequenceEncoder, use_numba: bool) -> None:
        """Create an encoder for grouped-count word-set pairs.

        Args:
            prev_encoder: Encoder for previous-set grouped counts.
            len_encoder: Encoder for emitted-set lengths.
            use_numba: Whether to encode flattened arrays for Numba kernels.

        Attributes:
            prev_encoder: Encoder for previous-set grouped counts.
            len_encoder: Encoder for emitted-set lengths.
            use_numba: Whether Numba encoding is enabled.

        """
        self.prev_encoder = prev_encoder
        self.len_encoder = len_encoder
        self.use_numba = use_numba

    def __str__(self) -> str:
        """Return a readable encoder summary."""
        s = "IntegerHiddenAssociationDataEncoder(prev_encoder=" + str(self.prev_encoder) + ",len_encoder="
        s += str(self.len_encoder) + ",use_numba=" + str(self.use_numba) + ")"
        return s

    def __eq__(self, other: object) -> bool:
        """Return whether ``other`` uses the same child encoders and Numba setting."""
        if isinstance(other, IntegerHiddenAssociationDataEncoder):
            cond0 = self.prev_encoder == other.prev_encoder
            cond1 = self.len_encoder == other.len_encoder
            cond2 = self.use_numba == other.use_numba
            return cond0 and cond1 and cond2
        else:
            return False

    def _seq_encode(
        self, x: Sequence[tuple[list[tuple[int, float]], list[tuple[int, float]]]]
    ) -> tuple[tuple[list[tuple[np.ndarray, ...]], Any | None, Any | None], None]:
        """Encode observations for the non-Numba sequence path.

        Returns 'rv' Tuple of
            rv[0] (List[Tuple[ndarray[int], ndarray[float], ndarray[int], ndarray[float]]]): List of Tuples containing
                Flattened numpy arrays of x0 values, x0 counts, x1 values, x1 counts.
            rv[1] (E1): Sequence encoded output from list of Tuples containing sum of counts for
                x0 and x1.
            rv[2] (E2): Sequence encoding of x0 from prev_encoder.

        Args:
            x: Sequence of iid integer hidden association observations.

        Returns:
            See rv above.

        """
        rv = []
        nn = []

        for xx in x:
            rv0 = []
            for c_vec in xx:
                rv0.append(np.asarray([v for v, c in c_vec], dtype=int))
                rv0.append(np.asarray([c for v, c in c_vec], dtype=float))
            nn0 = np.sum(rv0[-1])

            rv.append(tuple(rv0))
            nn.append(nn0)

        nn = self.len_encoder.seq_encode(nn)
        xv = self.prev_encoder.seq_encode([x[0] for x in x])

        return (rv, xv, nn), None

    def seq_encode(
        self, x: Sequence[tuple[list[tuple[int, float]], list[tuple[int, float]]]]
    ) -> (
        tuple[tuple[list[tuple[np.ndarray, ...]], Any | None, Any | None], None]
        | tuple[None, tuple[np.ndarray, ...], Any | None, Any | None]
    ):
        """Sequence encoding for integer hidden association observations.

        If numba is not used see _seq_encode(). Else the following is returned a Tuple of the following form is returned
        None, ((s0, s1, x0, x1, c0, c1, w0), xv, nn) with,

            s0 (np.ndarray): Numpy array of lengths for length of x[i][0]
            s1 (np.ndarray): Numpy array of lengths for length of x[i][1].
            x0 (np.ndarray): Flattened numpy array of values from x[i][0].
            x1 (np.ndarray): Flattened numpy array of values from x[i][1].
            c0 (np.ndarray): Flattened numpy array of counts from x[i][0].
            c1 (np.ndarray): Flattened numpy array of counts from x[i][1].
            w0 (np.ndarray): Numpy array of sum of counts for each x[i][0].
            xv (E1): Sequence encoded flattened values of x[i][0].
            nn (E2): Sequence encoded values of lengths (counts).

        Args:
            x: Sequence of iid integer hidden association observations.

        Returns:
            See above.

        """

        if not self.use_numba:
            enc_rv = self._seq_encode(x)
        else:
            x1 = []
            x0 = []
            s1 = []
            s0 = []
            c0 = []
            c1 = []
            w0 = []
            nn = []

            for i, xx in enumerate(x):
                xx0 = [v for v, c in xx[0]]
                cc0 = [c for v, c in xx[0]]
                xx1 = [v for v, c in xx[1]]
                cc1 = [c for v, c in xx[1]]

                x0.extend(xx0)
                x1.extend(xx1)
                c0.extend(cc0)
                c1.extend(cc1)
                w0.append(sum(cc0))
                s1.append(len(xx1))
                s0.append(len(xx0))
                nn.append(sum(cc1))

            nn = self.len_encoder.seq_encode(nn)
            xv = self.prev_encoder.seq_encode([x[0] for x in x])

            x0 = np.asarray(x0, dtype=np.int32)
            x1 = np.asarray(x1, dtype=np.int32)
            c0 = np.asarray(c0, dtype=np.float64)
            c1 = np.asarray(c1, dtype=np.float64)
            s0 = np.asarray(s0, dtype=np.int32)
            s1 = np.asarray(s1, dtype=np.int32)
            w0 = np.asarray(w0, dtype=np.float64)

            enc_rv = tuple([None, ((s0, s1, x0, x1, c0, c1, w0), xv, nn)])

        return enc_rv


@numba.njit(
    "void(int64, int64, int32[:], int32[:], int32[:], int32[:], float64[:], float64[:], float64[:], float64[:,:], "
    "float64[:,:], float64[:], float64, float64, float64[:])",
    cache=True,
)
def numba_seq_log_density(
    num_states, max_len1, t0, t1, x0, x1, c0, c1, w0, cond_weights, state_prob_mat, init_prob_vec, a, b, out
):
    """Numba kernel computing per-observation log-densities into out from flattened encodings."""
    x_mat = np.zeros((max_len1, num_states), dtype=np.float64)

    for i in range(len(t0) - 1):
        vx = x0[t0[i] : t0[i + 1]]
        cx = c0[t0[i] : t0[i + 1]]
        vy = x1[t1[i] : t1[i + 1]]
        cy = c1[t1[i] : t1[i + 1]]
        sx = w0[i]

        l1 = t0[i + 1] - t0[i]
        l2 = t1[i + 1] - t1[i]

        for j in range(l1):
            temp = cx[j] / sx
            # out[i] += math.log(init_prob_vec[vx[j]])*cx[j]
            for k in range(num_states):
                x_mat[j, k] = cond_weights[vx[j], k] * temp

        for w in range(l2):
            wid = vy[w]
            temp_sum = 0
            for j in range(l1):
                for k in range(num_states):
                    temp_sum += x_mat[j, k] * state_prob_mat[k, wid]
            prob = temp_sum * b + a
            if prob > 0.0:
                out[i] += math.log(prob) * cy[w]
            else:
                out[i] = -math.inf


@numba.njit(
    "void(int64, int64, int32[:], int32[:], int32[:], int32[:], float64[:], float64[:], float64[:], float64[:,:], "
    "float64[:,:], float64[:,:], float64[:,:], float64[:], float64[:], float64, float64)",
    cache=True,
)
def numba_seq_update(
    num_states,
    max_len1,
    t0,
    t1,
    x0,
    x1,
    c0,
    c1,
    w0,
    cond_weights,
    state_prob_mat,
    weight_count,
    state_count,
    init_count,
    weights,
    a,
    b,
):
    """Numba kernel accumulating posterior weight/state counts in place from flattened encodings."""
    x_mat = np.zeros((max_len1, num_states), dtype=np.float64)
    z_mat = np.zeros((max_len1, num_states), dtype=np.float64)

    for i in range(len(t0) - 1):
        weight = weights[i]
        vx = x0[t0[i] : t0[i + 1]]
        cx = c0[t0[i] : t0[i + 1]]
        vy = x1[t1[i] : t1[i + 1]]
        cy = c1[t1[i] : t1[i + 1]]

        l1 = t0[i + 1] - t0[i]
        l2 = t1[i + 1] - t1[i]

        nx = w0[i]

        for j in range(l1):
            temp = cx[j] / nx
            init_count[vx[j]] += cx[j] * weight
            for k in range(num_states):
                x_mat[j, k] = cond_weights[vx[j], k] * temp

        for w in range(l2):
            wid = vy[w]
            temp_sum = 0
            for j in range(l1):
                for k in range(num_states):
                    temp = x_mat[j, k] * state_prob_mat[k, wid]
                    z_mat[j, k] = temp
                    temp_sum += temp

            denom = temp_sum * b + a
            if denom > 0.0:
                temp_weight = cy[w] * weight * b / denom
            else:
                temp_weight = 0.0
            for j in range(l1):
                for k in range(num_states):
                    temp = temp_weight * z_mat[j, k]
                    weight_count[vx[j], k] += temp
                    state_count[k, wid] += temp


@numba.njit("float64[:,:](int32[:], float64[:,:], float64[:,:])", cache=True)
def vec_bincount1(x, w, out):
    """Numba bincount on the rows of matrix w for groups x.

    Args:
        x (np.ndarray[np.float64]): Group ids of rows
        w (np.ndarray[np.float64]): N by S numpy array with rows corresponding to x
        out (np.ndarray[np.float64]): Unique values in support of x by S.

    Returns:
        Numpy 2-d array.

    """
    for i in range(len(x)):
        out[x[i], :] += w[i, :]
    return out


@numba.njit("float64[:,:](int32[:], float64[:,:], float64[:,:])", cache=True)
def vec_bincount2(x, w, out):
    """Numba bincount on the rows of matrix w for groups x.

    N = len(x)
    S = number of states.
    U = unique values in x can take on.

    Args:
        x (np.ndarray[np.float64]): Group ids of columns of w.
        w (np.ndarray[np.float64]): S by N numpy array with cols corresponding to x
        out (np.ndarray[np.float64]): S by U matrix.

    Returns:
        Numpy 2-d array.

    """
    for j in range(len(x)):
        out[:, x[j]] += w[:, j]
    return out


def _register_int_hidden_association_engine_kernel():
    """Register the engine-resident integer-hidden-association kernel (idempotent; called at import)."""
    from mixle.stats.compute.kernel import GenericKernel, GenericKernelFactory, KernelFactory, register_kernel_factory

    class IntegerHiddenAssociationKernel(GenericKernel):
        def accumulate(self, enc, weights):
            if self.estimator is None:
                raise ValueError("IntegerHiddenAssociationKernel.accumulate requires an estimator.")
            if not getattr(self.engine, "resident_estep", True):
                return super().accumulate(enc, weights)
            host_enc = getattr(enc, "host_payload", enc)
            accumulator = self.estimator.accumulator_factory().make()
            accumulator.seq_update_engine(host_enc, weights, self.dist, self.engine)
            return accumulator.value()

    class IntegerHiddenAssociationKernelFactory(KernelFactory):
        def build(self, dist, engine, estimator=None):
            if not dist.supports_engine(engine):
                return GenericKernelFactory().build(dist, engine, estimator=estimator)
            return IntegerHiddenAssociationKernel(dist, engine=engine, estimator=estimator)

    register_kernel_factory(IntegerHiddenAssociationDistribution, IntegerHiddenAssociationKernelFactory())


_register_int_hidden_association_engine_kernel()
