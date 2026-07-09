"""Hidden Markov models whose latent states live on rooted trees.

Data type: Sequence[Tuple[Tuple[int, int], T]]. A single observation is a rooted tree given as a sequence
of ((node_id, parent_id), emission) tuples, where node_id is an integer in 0,1,...,n-1, the root node has
parent_id = -1, and the emission has the data type T of the emission (topic) distributions.

Each node u carries a hidden state Z_u in {0,...,K-1}. The root state is drawn from the initial state
weights w, the state of every child is drawn from the transition probability matrix A conditioned on its
parent's state (children of a node are conditionally independent given the parent state), and the observed
emission at node u is drawn from topics[Z_u]. The number of children of each node is modeled by an optional
length distribution (len_dist) with support on the non-negative integers, which is required for sampling.

Inference uses an upward-downward (beta/eta) message passing recursion over tree levels. Two equivalent
vectorized implementations are provided and selected with the use_numba flag: numba-compiled kernels that
parallelize over trees, and a pure-numpy implementation that batches nodes level by level.

"""

import math
from collections.abc import Sequence
from typing import Any, TypeVar

import numpy as np
from numpy.random import RandomState

import mixle.utils.vector as vec
from mixle.capability import Neutral, supports
from mixle.engines.arithmetic import *
from mixle.engines.arithmetic import maxrandint
from mixle.stats.combinator.null_dist import (
    NullAccumulator,
    NullAccumulatorFactory,
    NullDataEncoder,
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
)
from mixle.utils.aliasing import MISSING, coalesce_alias, require
from mixle.utils.optional_deps import numba

D = tuple[int, int | None]
T = TypeVar("T")  # Type for emissions
SS0 = TypeVar("SS0")  # Type for suff stat of emissions
SS1 = TypeVar("SS1")  # Type for suff-stat of length dist

E1 = tuple[int, np.ndarray, np.ndarray, np.ndarray]
E2 = tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]
E3 = TypeVar("E3")  # Encoded emissions
E4 = TypeVar("E4")  # encoded lengths of children
E5 = tuple[np.ndarray, np.ndarray, np.ndarray]
E6 = tuple[
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    list[np.ndarray],
    list[np.ndarray],
    list[np.ndarray],
    list[np.ndarray],
    np.ndarray,
]
E01 = tuple[np.ndarray, E1, E2, E3, tuple[np.ndarray, E4] | None]
E02 = tuple[int, np.ndarray, E5, E6, E3, tuple[np.ndarray, E4] | None]
E = tuple[E01 | None, E02 | None]


def find_level(parents: np.ndarray) -> list[int]:
    """Find the level in the tree for nodes, given an array of parents.

    Args:
        parents (np.ndarray): Numpy array of integers with first entry -1.

    Returns:
        Level of each node in the free excluding the first entry which is the root (level = 0).

    """
    n = len(parents)
    if n == 1:
        return []
    out = np.zeros(n, dtype=np.int32)
    for i in range(1, n):
        out[i] = out[parents[i]] + 1
    return list(out[1:])


class TreeHiddenMarkovModelDistribution(SequenceEncodableProbabilityDistribution):
    """Hidden Markov model on a rooted tree with emission distributions of type T.

    Data type: Sequence[Tuple[Tuple[int, int], T]] (((node_id, parent_id), emission) per node, root parent -1).
    """

    def compute_capabilities(self):
        """Declare generated-compute support from emission and length children."""
        from mixle.stats.compute.capabilities import DistributionCapabilities, intersect_engine_ready

        if self.use_numba:
            return DistributionCapabilities(engine_ready=("numpy",), kernel_status="legacy_numpy")
        children = tuple(self.topics) + ((self.len_dist,) if self.len_dist is not None else ())
        return DistributionCapabilities(engine_ready=intersect_engine_ready(children), kernel_status="generic_latent")

    def compute_declaration(self):
        """Return the generated-compute declaration for the tree HMM."""
        from mixle.stats.compute.declarations import (
            DistributionDeclaration,
            ParameterSpec,
            StatisticSpec,
            declaration_for,
        )

        topic_children = tuple(declaration_for(topic) for topic in self.topics)
        length = None if supports(self.len_dist, Neutral) else declaration_for(self.len_dist)
        children = tuple(
            child for child in topic_children + ((length,) if length is not None else ()) if child is not None
        )
        roles = tuple("state_%d_emission" % i for i, child in enumerate(topic_children) if child is not None)
        if length is not None:
            roles += ("length",)
        return DistributionDeclaration(
            name="tree_hidden_markov",
            distribution_type=type(self),
            parameters=(
                ParameterSpec("w", constraint="simplex_vector"),
                ParameterSpec("transitions", constraint="row_simplex_matrix"),
            ),
            statistics=(
                StatisticSpec("num_states", kind="metadata", additive=False, scales=False),
                StatisticSpec("initial_counts"),
                StatisticSpec("state_counts"),
                StatisticSpec("transition_counts"),
                StatisticSpec("emissions", kind="tuple"),
                StatisticSpec("length", kind="child_stat"),
            ),
            support="tree_hidden_state_sequence",
            children=children,
            child_roles=roles,
            differentiable=False,
        )

    def __init__(
        self,
        topics: Sequence[SequenceEncodableProbabilityDistribution],
        w: Sequence[float] | np.ndarray = MISSING,
        transitions: list[list[float]] | np.ndarray = MISSING,
        len_dist: SequenceEncodableProbabilityDistribution | None = NullDistribution(),
        terminal_level: int = 10,
        name: str | None = None,
        use_numba: bool = False,
        weights: Sequence[float] | np.ndarray = MISSING,
    ) -> None:
        """TreeHiddenMarkovModelDistribution for specifying an HMM on a rooted tree.

        Args:
            topics (Sequence[SequenceEncodableProbabilityDistribution]): Emission distributions having type T.
            w (Union[Sequence[float], np.ndarray]): Initial state weights. Must sum to 1 and have same length as topics.
            transitions (Union[List[List[float]], np.ndarray]): Define the TPM for HMM. Dim is len(topics) by
                len(topics).
            len_dist (Optional[SequenceEncodableProbabilityDistribution]): Distribution for the number of children
                a node in the tree will have. Must have support on non-negative integers.
            terminal_level (int): Level of tree to terminate sampling. Default to 10.
            name (Optional[str]): Optional distribution name.
            use_numba (bool): If true Numba is used for vectorized calculations.

        Note on terminal *states*: the absorbing-hidden-state ``terminal_states`` supported by the
        sequence HMM variants (base/quantized/lookback/semi-supervised/segmental) does not apply here.
        That concept makes the *sequence length* a stopping time via an absorbing-state forward
        algorithm -- but a tree has no single length or linear forward (inference is an upward-downward
        recursion over many branches/leaves). The tree's native termination is structural:
        ``terminal_level`` bounds the depth and ``len_dist`` governs per-node branching (a node with 0
        children is a leaf). A tree analogue of absorbing states ("a terminal state forces 0 children")
        would be a *different* model -- a state-coupled child-count distribution requiring a rewrite of
        the numba upward-downward kernels -- and is intentionally not conflated with the sequence
        variants' ``terminal_states``.

        Attributes:
            topics (Sequence[SequenceEncodableProbabilityDistribution]): Emission distributions having type T.
            num_states (int): Number of states in HMM.
            w (np.ndarray): Initial state distribution. Sums to 1.
            log_w (np.ndarray): Log of above.
            transitions (np.ndarray): TPM with dimensions num_states by num_states.
            log_transitions (np.ndarray): Log of TPM.
            len_dist (SequenceEncodableProbabilityDistribution): Distribution for number of children for a node.
                Defaults to NullDistribution.
            terminal_level (int): Level in tree to terminate sampling.
            use_numba (bool): If true Numba used for computations.

        """
        w = coalesce_alias("w", w, "weights", weights, default=MISSING)
        transitions = require("transitions", transitions, default=MISSING)

        with np.errstate(divide="ignore"):
            self.topics = topics
            self.num_states = len(w)
            self.w = vec.make(w)
            self.log_w = np.log(self.w)

            if not isinstance(transitions, np.ndarray):
                transitions = np.asarray(transitions, dtype=float)

            self.transitions = np.reshape(transitions, (self.num_states, self.num_states))
            self.log_transitions = np.log(self.transitions)
            self.name = name
            self.len_dist = len_dist if len_dist is not None else NullDistribution()
            self.terminal_level = terminal_level
            self.use_numba = use_numba

        # Cache for the parameter-only per-level marginal state probabilities
        # (init_prob @ transitions^k). Keyed by the (w, transitions) identities so it
        # is rebuilt if the parameters are ever replaced. See _get_p_level.
        self._p_level_cache: tuple[int, int, np.ndarray] | None = None

    def _get_p_level(self, levels: int) -> np.ndarray:
        """Return per-level marginal state probabilities for the first ``levels`` levels.

        ``level_state_prob`` depends only on the parameters (w, transitions), so the
        result is memoized across ``seq_log_density`` calls. A larger cached table covers
        smaller requests (each row k only depends on rows <= k), so we grow the cache
        monotonically and slice it.
        """
        key_w = id(self.w)
        key_t = id(self.transitions)
        cache = self._p_level_cache
        if cache is not None and cache[0] == key_w and cache[1] == key_t and cache[2].shape[0] >= levels:
            return cache[2][:levels]

        p_level = np.zeros((levels, self.num_states), dtype=np.float64)
        level_state_prob(levels, self.num_states, self.transitions, self.w, p_level)
        self._p_level_cache = (key_w, key_t, p_level)
        return p_level

    def __str__(self) -> str:
        """Return a constructor-style representation of the distribution."""
        s1 = ",".join(map(str, self.topics))
        s2 = repr(list(self.w))
        s3 = repr([list(u) for u in self.transitions])
        s4 = str(self.len_dist)
        s5 = repr(self.name)
        s6 = repr(self.use_numba)

        return (
            "TreeHiddenMarkovModelDistribution(topics=[%s], w=%s, transitions=%s, len_dist=%s, name=%s, "
            "use_numba=%s)" % (s1, s2, s3, s4, s5, s6)
        )

    def density(self, x: Sequence[tuple[D, T]]) -> float:
        """Density of tree HMM at a single observed tree x.

        See log_density() for details.

        Args:
            x (Sequence[Tuple[D, T]]): Tree as ((node_id, parent_id), emission) tuples (root parent -1).

        Returns:
            Density at observation x.

        """
        return exp(self.log_density(x))

    def log_density(self, x: Sequence[tuple[D, T]]) -> float:
        """Log-density of tree HMM at a single observed tree x.

        The hidden states are marginalized out with an upward (beta) message passing recursion over the
        tree. When a non-null length distribution (len_dist) is set, its contribution to the likelihood
        (the sum over nodes of len_dist.log_density(num_children)) is included; a NullDistribution length
        contributes nothing.

        Args:
            x (Sequence[Tuple[D, T]]): Tree as ((node_id, parent_id), emission) tuples (root parent -1).

        Returns:
            Log-density at observation x.

        """
        enc_x = self.dist_to_encoder().seq_encode([x])
        return self.seq_log_density(enc_x)[0]

    def seq_log_density(self, x: E) -> np.ndarray:
        """Vectorized evaluation of log-density at sequence encoded input x.

        Dispatches to numba kernels or to the pure-numpy level-by-level recursion depending on which
        encoding (use_numba) the input was created with.

        Args:
            x (E): Sequence encoded trees from TreeHiddenMarkovDataEncoder.seq_encode().

        Returns:
            Numpy array of log-density (float), one entry per encoded tree.

        """

        if x[0] is not None:
            tz, (max_level, xln, xlnl, tlnz), (xbi, xp, xc, xl, txz, tp, tpz), enc_x, len_enc = x[0]

            num_states = self.num_states
            w = self.w
            a_mat = self.transitions
            tot_cnt = tz[-1]
            num_trees = len(tz) - 1

            p_level = self._get_p_level(max_level + 1)

            pr_obs = np.zeros((tot_cnt, num_states), dtype=np.float64)
            ll_ret = np.zeros(num_trees, dtype=np.float64)

            # Compute state likelihood vectors and scale the max to one
            for i in range(num_states):
                pr_obs[:, i] = self.topics[i].seq_log_density(enc_x)

            pr_max0 = pr_obs.max(axis=1)
            with np.errstate(invalid="ignore"):  # impossible rows have max -inf -> NaN; zeroed below
                pr_obs -= pr_max0[:, None]
                np.exp(pr_obs, out=pr_obs)
            pr_obs[np.isnan(pr_obs).any(axis=1), :] = 0.0  # impossible observation -> zero emission row

            betas = np.ones_like(pr_obs, dtype=np.float64)
            etas = np.zeros((len(xbi), num_states), dtype=np.float64)

            numba_seq_log_density(
                num_states,
                tz,
                txz,
                tp,
                tpz,
                tlnz,
                xp,
                xc,
                xl,
                xbi,
                xln,
                xlnl,
                pr_obs,
                p_level,
                a_mat,
                pr_max0,
                betas,
                etas,
                ll_ret,
            )

            #  Childless-root trees have no leaf/parent entries and never enter the kernel.
            single = np.flatnonzero(np.diff(tz) == 1)
            if single.size > 0:
                r = tz[single]
                ll_ret[single] += np.log(np.dot(pr_obs[r, :], w)) + pr_max0[r]

            if len_enc is not None and len_enc[1] is not None:
                len_ll = self.len_dist.seq_log_density(len_enc[1])
                ll_ret += np.bincount(len_enc[0], weights=len_ll, minlength=num_trees)

            return ll_ret

        else:
            cnt, tz, (xln, xlnl, xlni), (idx, xbi, xp, xc, level_idx, p_nxt, eta_p, i_nxt, _, _), enc_x, len_enc = x[1]

            num_states = self.num_states
            max_level = len(level_idx)
            a_mat = self.transitions
            w = self.w
            num_trees = len(tz) - 1

            betas = np.ones((cnt, num_states), dtype=np.float64)
            etas = np.zeros((len(xbi), num_states), dtype=np.float64)

            p_level = np.zeros((max_level + 1, num_states), dtype=np.float64)
            p_level[0, :] += w

            for level in range(1, max_level + 1):
                p_level[level, :] += np.matmul(p_level[level - 1, :], a_mat)

            pr_obs = np.zeros((cnt, num_states), dtype=np.float64)
            ll_ret = np.zeros(num_trees, dtype=np.float64)

            # Compute state likelihood vectors and scale the max to one
            for i in range(num_states):
                pr_obs[:, i] = self.topics[i].seq_log_density(enc_x)

            pr_max0 = pr_obs.max(axis=1)
            with np.errstate(invalid="ignore"):  # impossible rows have max -inf -> NaN; zeroed below
                pr_obs -= pr_max0[:, None]
                np.exp(pr_obs, out=pr_obs)
            pr_obs[np.isnan(pr_obs).any(axis=1), :] = 0.0  # impossible observation -> zero emission row

            #  set the leaf nodes
            betas[xln, :] *= pr_obs[xln, :] * p_level[xlnl, :]
            betas_sum = np.sum(betas[xln, :], axis=1, keepdims=True)
            # divide by a clamped sum (impossible leaf -> betas_sum 0 -> keep beta 0, avoid 0/0), but keep
            # the true betas_sum for the log below so an impossible leaf correctly drives ll to -inf.
            betas[xln, :] /= np.where(betas_sum > 0.0, betas_sum, 1.0)

            with np.errstate(divide="ignore"):
                ll_ret += np.bincount(xlni, weights=np.log(betas_sum.flatten()) + pr_max0[xln], minlength=num_trees)

            #  upward pass on betas
            for level in range(len(level_idx) - 1, -1, -1):
                lidx = level_idx[level]
                idxs, xbis, xps, xcs = idx[lidx], xbi[lidx], xp[lidx], xc[lidx]

                #  Get etas
                temp = np.reshape(betas[xcs, :], (-1, num_states, 1))
                temp /= np.reshape(p_level[level + 1, :], (1, num_states, 1))
                temp = np.sum(a_mat.T * temp, axis=1)
                etas[xbis, :] += temp

                # within-segment sums (batch-independent, unlike a cumsum-difference)
                with np.errstate(divide="ignore"):  # an impossible subtree gives etas 0 -> log -inf -> beta 0
                    log_etas = np.add.reduceat(np.log(etas[xbis, :]), eta_p[level][:-1], axis=0)

                betas[p_nxt[level], :] *= np.exp(log_etas) * pr_obs[p_nxt[level], :]
                betas[p_nxt[level], :] *= p_level[level, :]
                betas_sum = np.sum(betas[p_nxt[level], :], axis=1, keepdims=True)

                # clamp the divisor (impossible node -> betas_sum 0 -> keep beta 0, avoid 0/0), keeping the
                # true betas_sum for the log so an impossible subtree correctly drives ll to -inf.
                betas[p_nxt[level], :] /= np.where(betas_sum > 0.0, betas_sum, 1.0)

                with np.errstate(divide="ignore"):
                    ll_ret += np.bincount(
                        i_nxt[level], weights=np.log(betas_sum.flatten()) + pr_max0[p_nxt[level]], minlength=num_trees
                    )

            if len_enc is not None and len_enc[1] is not None:
                len_ll = self.len_dist.seq_log_density(len_enc[1])
                ll_ret += np.bincount(len_enc[0], weights=len_ll, minlength=num_trees)

            return ll_ret

    def backend_seq_log_density(self, x: E, engine: Any) -> Any:
        """Engine-neutral tree-HMM scoring for the pure non-numba encoded layout."""
        from mixle.stats.compute.backend import BackendScoringError, backend_seq_log_density

        if x[0] is not None:
            if getattr(engine, "name", None) == "numpy":
                return self.seq_log_density(x)
            raise BackendScoringError("Tree HMM backend scoring requires the pure non-numba encoding.")
        if x[1] is None:
            raise BackendScoringError("Tree HMM backend scoring received an empty encoded layout.")

        cnt, tz, (xln, xlnl, xlni), (idx, xbi, xp, xc, level_idx, p_nxt, eta_p, i_nxt, _, _), enc_x, len_enc = x[1]

        num_states = self.num_states
        max_level = len(level_idx)
        num_trees = len(tz) - 1

        betas = engine.zeros((cnt, num_states)) + engine.asarray(1.0)
        etas = engine.zeros((len(xbi), num_states))

        a_mat = engine.asarray(self.transitions)
        a_mat_t = engine.asarray(self.transitions.T)
        p_levels = [engine.asarray(self.w)]
        for _ in range(1, max_level + 1):
            p_levels.append(engine.matmul(p_levels[-1], a_mat))
        p_level = engine.stack(p_levels, axis=0)

        emission_scores = [backend_seq_log_density(topic, enc_x, engine) for topic in self.topics]
        log_pr_obs = engine.stack(emission_scores, axis=1)
        pr_max0 = engine.max(log_pr_obs, axis=1)
        pr_obs = engine.exp(log_pr_obs - pr_max0[:, None])
        ll_ret = engine.zeros(num_trees)

        if len(xln):
            leaf_idx = engine.asarray(xln)
            leaf_level = engine.asarray(xlnl)
            leaf_beta = pr_obs[leaf_idx, :] * p_level[leaf_level, :]
            betas[leaf_idx, :] = leaf_beta
            betas_sum = engine.sum(betas[leaf_idx, :], axis=1, keepdims=True)
            betas[leaf_idx, :] = betas[leaf_idx, :] / betas_sum
            ll_ret = ll_ret + engine.bincount(
                engine.asarray(xlni), weights=engine.log(betas_sum[:, 0]) + pr_max0[leaf_idx], minlength=num_trees
            )

        for level in range(max_level - 1, -1, -1):
            xbis = xbi[level_idx[level]]
            xcs = xc[level_idx[level]]
            if len(xbis) == 0:
                continue

            xbis_idx = engine.asarray(xbis)
            child_idx = engine.asarray(xcs)
            child_beta = betas[child_idx, :] / p_level[level + 1, :]
            temp = engine.sum(a_mat_t[None, :, :] * child_beta[:, :, None], axis=1)
            etas[xbis_idx, :] = etas[engine.asarray(xbis), :] + temp

            log_eta_rows = engine.log(etas[xbis_idx, :])
            log_eta_parts = []
            for start, stop in zip(eta_p[level][:-1], eta_p[level][1:]):
                log_eta_parts.append(engine.sum(log_eta_rows[int(start) : int(stop), :], axis=0))
            log_etas = engine.stack(log_eta_parts, axis=0) if log_eta_parts else engine.zeros((0, num_states))

            parent_idx = engine.asarray(p_nxt[level])
            parent_beta = betas[parent_idx, :] * engine.exp(log_etas) * pr_obs[parent_idx, :] * p_level[level, :]
            betas[parent_idx, :] = parent_beta
            betas_sum = engine.sum(betas[parent_idx, :], axis=1, keepdims=True)
            betas[parent_idx, :] = betas[parent_idx, :] / betas_sum
            ll_ret = ll_ret + engine.bincount(
                engine.asarray(i_nxt[level]),
                weights=engine.log(betas_sum[:, 0]) + pr_max0[parent_idx],
                minlength=num_trees,
            )

        if len_enc is not None and len_enc[1] is not None:
            len_ll = backend_seq_log_density(self.len_dist, len_enc[1], engine)
            ll_ret = ll_ret + engine.bincount(engine.asarray(len_enc[0]), weights=len_ll, minlength=num_trees)

        return ll_ret

    def seq_posterior(self, x: E) -> list[np.ndarray] | None:
        """Posterior state membership probabilities for each node of each encoded tree.

        Args:
            x (E): Sequence encoded trees from TreeHiddenMarkovDataEncoder.seq_encode().

        Returns:
            List with one numpy array per tree, each of shape (num_nodes, num_states), containing the
            posterior probability of each hidden state at each node.

        """

        if x[0] is not None:
            tz, (max_level, xln, xlnl, tlnz), (xbi, xp, xc, xl, txz, tp, tpz), enc_x, _ = x[0]

            num_states = self.num_states
            w = self.w
            a_mat = self.transitions
            tot_cnt = tz[-1]

            p_level = self._get_p_level(max_level + 1)

            pr_obs = np.zeros((tot_cnt, num_states), dtype=np.float64)

            # Compute state likelihood vectors and scale the max to one
            for i in range(num_states):
                pr_obs[:, i] = self.topics[i].seq_log_density(enc_x)

            pr_max0 = pr_obs.max(axis=1)
            pr_obs -= pr_max0[:, None]
            np.exp(pr_obs, out=pr_obs)

            betas = np.zeros_like(pr_obs, dtype=np.float64)
            etas = np.zeros((len(xbi), num_states), dtype=np.float64)

            ### Need to do upward and downward, then read back the gammas
            numba_posteriors(
                num_states, tz, txz, tp, tpz, tlnz, xp, xc, xl, xbi, xln, xlnl, pr_obs, p_level, a_mat, betas, etas
            )

            #  Childless-root trees have no leaf/parent entries and never enter the kernel.
            single = np.flatnonzero(np.diff(tz) == 1)
            if single.size > 0:
                r = tz[single]
                gam = pr_obs[r, :] * w[None, :]
                betas[r, :] = gam / gam.sum(axis=1, keepdims=True)

            return [betas[tz[i] : tz[i + 1], :] for i in range(len(tz) - 1)]

        else:
            cnt, tz, (xln, xlnl, xlni), (idx, xbi, xp, xc, level_idx, p_nxt, eta_p, i_nxt, _, _), enc_x, len_enc = x[1]

            num_states = self.num_states
            max_level = len(level_idx)
            a_mat = self.transitions
            w = self.w
            num_trees = len(tz) - 1

            betas = np.ones((cnt, num_states), dtype=np.float64)
            etas = np.zeros((len(xbi), num_states), dtype=np.float64)

            p_level = np.zeros((max_level + 1, num_states), dtype=np.float64)
            p_level[0, :] += w

            for level in range(1, max_level + 1):
                p_level[level, :] += np.matmul(p_level[level - 1, :], a_mat)

            pr_obs = np.zeros((cnt, num_states), dtype=np.float64)

            # Compute state likelihood vectors and scale the max to one
            for i in range(num_states):
                pr_obs[:, i] = self.topics[i].seq_log_density(enc_x)

            pr_max0 = pr_obs.max(axis=1)
            pr_obs -= pr_max0[:, None]
            np.exp(pr_obs, out=pr_obs)

            #  set the leaf nodes
            betas[xln, :] *= pr_obs[xln, :] * p_level[xlnl, :]
            betas_sum = np.sum(betas[xln, :], axis=1, keepdims=True)
            betas[xln, :] /= betas_sum

            #  upward pass on betas
            for level in range(len(level_idx) - 1, -1, -1):
                lidx = level_idx[level]
                idxs, xbis, xps, xcs = idx[lidx], xbi[lidx], xp[lidx], xc[lidx]

                #  Get etas
                temp = np.reshape(betas[xcs, :], (-1, num_states, 1))
                temp /= np.reshape(p_level[level + 1, :], (1, num_states, 1))
                temp = np.sum(a_mat.T * temp, axis=1)
                etas[xbis, :] += temp

                # within-segment sums (batch-independent, unlike a cumsum-difference)
                log_etas = np.add.reduceat(np.log(etas[xbis, :]), eta_p[level][:-1], axis=0)

                betas[p_nxt[level], :] *= np.exp(log_etas) * pr_obs[p_nxt[level], :]
                betas[p_nxt[level], :] *= p_level[level, :]
                betas_sum = np.sum(betas[p_nxt[level], :], axis=1, keepdims=True)

                betas[p_nxt[level], :] /= betas_sum

            #  Return betas by observed sequence need tz
            return [betas[tz[i] : tz[i + 1], :] for i in range(len(tz) - 1)]

    def viterbi(self, x: Sequence[tuple[D, T]]) -> np.ndarray:
        """Most likely hidden state assignment for each node of a single observed tree.

        Args:
            x (Sequence[Tuple[D, T]]): Tree as ((node_id, parent_id), emission) tuples (root parent -1).

        Returns:
            Numpy array of int states, one per node of the tree.

        """
        enc_x = self.dist_to_encoder().seq_encode([x])
        return self.seq_viterbi(enc_x)[0]

    def seq_viterbi(self, x: E) -> list[np.ndarray]:
        """Vectorized Viterbi state assignments for sequence encoded trees.

        Args:
            x (E): Sequence encoded trees from TreeHiddenMarkovDataEncoder.seq_encode().

        Returns:
            List with one numpy array of int states per tree (one state per node).

        """
        if x[0] is not None:
            tz, (max_level, xln, xlnl, tlnz), (xbi, xp, xc, xl, txz, tp, tpz), enc_x, _ = x[0]

            num_states = self.num_states
            log_w = self.log_w
            log_a_mat = self.log_transitions
            tot_cnt = tz[-1]

            log_pr_obs = np.zeros((tot_cnt, num_states), dtype=np.float64)

            for i in range(num_states):
                log_pr_obs[:, i] = self.topics[i].seq_log_density(enc_x)

            betas = np.ones_like(log_pr_obs, dtype=np.float64)
            etas = np.ones((len(xbi), num_states), dtype=np.float64)
            out = np.zeros(tot_cnt, dtype=np.int32)

            numba_viterbi(
                num_states, tz, txz, tp, tpz, tlnz, xp, xc, xl, xbi, xln, log_pr_obs, log_w, log_a_mat, betas, etas, out
            )

            return [out[tz[i] : tz[i + 1]] for i in range(len(tz) - 1)]

        else:
            cnt, tz, (xln, xlnl, xlni), (idx, xbi, xp, xc, level_idx, p_nxt, eta_p, i_nxt, rns, rni), enc_x, _ = x[1]

            num_states = self.num_states
            max_level = len(level_idx)
            log_a_mat = self.log_transitions
            log_w = self.log_w

            log_delta = np.ones((cnt, num_states), dtype=np.float64)
            log_eta = np.zeros((len(xbi), num_states), dtype=np.float64)
            state_tracker = np.zeros(cnt, dtype=np.int32)

            # Compute state likelihood vectors, and initialize the deltas for each state
            for i in range(num_states):
                log_delta[:, i] += self.topics[i].seq_log_density(enc_x)

            state_tracker[xln] += np.argmax(log_delta[xln, :], axis=1).flatten()

            #  upward pass on deltas
            for level in range(max_level - 1, -1, -1):
                lidx = level_idx[level]
                idxs, xbis, xps, xcs = idx[lidx], xbi[lidx], xp[lidx], xc[lidx]

                #  Get log_etas
                log_eta[xbis, :] += np.max(np.reshape(log_delta[xcs, :], (-1, 1, num_states)) + log_a_mat, axis=2)
                temp = np.zeros((len(xbis) + 1, num_states), dtype=np.float64)
                temp[1:, :] += np.cumsum(log_eta[xbis, :], axis=0)
                temp = temp[eta_p[level][1:], :] - temp[eta_p[level][:-1], :]
                log_delta[p_nxt[level], :] += temp
                state_tracker[p_nxt[level]] += np.argmax(log_delta[p_nxt[level], :], axis=1, keepdims=False)

            #  Set the init for leaf nodes
            log_delta[rns, :] += log_w
            state_tracker[rns] += np.argmax(log_delta[rns, :], axis=1).flatten()

            return [state_tracker[tz[i] : tz[i + 1]] for i in range(len(tz) - 1)]

    def density_semantics(self):
        """Return exact-or-approximate density semantics joined from child models."""
        from mixle.stats.compute.pdist import DensitySemantics, join_density_semantics

        children = list(self.topics) + ([] if self.len_dist is None else [self.len_dist])
        sems = [c.density_semantics() for c in children if hasattr(c, "density_semantics")]
        return join_density_semantics(sems) if sems else DensitySemantics.EXACT

    def sampler(self, seed: int | None = None) -> "TreeHiddenMarkovSampler":
        """Create a sampler for this tree hidden Markov distribution.

        Args:
            seed (Optional[int]): Used to set seed in random sampler.

        Returns:
            TreeHiddenMarkovSampler: Sampler bound to this distribution.

        Raises:
            Exception: If len_dist is a NullDistribution (a length distribution with support on the
                non-negative integers is required for sampling).

        """
        if supports(self.len_dist, Neutral):
            raise Exception("TreeHiddenMarkovSampler requires len_dist with support on non-negative integers")
        return TreeHiddenMarkovSampler(self, seed)

    def enumerator(self) -> DistributionEnumerator:
        """Not supported: observations are rooted trees, not chains.

        The chain best-first enumerator (:class:`HiddenMarkovModelEnumerator`) does not apply -- the
        marginal sums over hidden states on a branching tree whose shape is itself governed by the
        per-node child-count ``len_dist``, so the support is a set of trees rather than sequences.
        Use :meth:`sampler` or the exact ``log_density`` instead.
        """
        raise EnumerationError(
            self,
            reason="tree-structured (branching) observations are not supported by the chain enumerator",
        )

    def estimator(self, pseudo_count: float | None = None) -> "TreeHiddenMarkovEstimator":
        """Create a TreeHiddenMarkovEstimator with estimators for the topics and length distribution.

        Args:
            pseudo_count (Optional[float]): Used to inflate sufficient statistics of the initial state
                weights, transition matrix, topics, and length distribution.

        Returns:
            TreeHiddenMarkovEstimator object.

        """
        len_est = None if self.len_dist is None else self.len_dist.estimator(pseudo_count=pseudo_count)
        comp_ests = [u.estimator(pseudo_count=pseudo_count) for u in self.topics]
        return TreeHiddenMarkovEstimator(
            comp_ests,
            pseudo_count=(pseudo_count, pseudo_count),
            len_estimator=len_est,
            name=self.name,
            use_numba=self.use_numba,
        )

    def dist_to_encoder(self) -> "TreeHiddenMarkovDataEncoder":
        """Return a data encoder for iid tree HMM observations."""
        emission_encoder = self.topics[0].dist_to_encoder()
        len_encoder = self.len_dist.dist_to_encoder()

        return TreeHiddenMarkovDataEncoder(
            emission_encoder=emission_encoder, len_encoder=len_encoder, use_numba=self.use_numba
        )


class TreeHiddenMarkovSampler(DistributionSampler):
    """Sampler for the TreeHiddenMarkovModelDistribution. Draws rooted trees of state-emitted observations."""

    def __init__(self, dist: "TreeHiddenMarkovModelDistribution", seed: int | None = None) -> None:
        """Create a sampler for a tree hidden Markov model.

        Args:
            dist (TreeHiddenMarkovModelDistribution): Distribution to sample from. Must have a len_dist
                with support on the non-negative integers.
            seed (Optional[int]): Seed for random number generator.

        Attributes:
            num_states (int): Number of hidden states.
            dist (TreeHiddenMarkovModelDistribution): Distribution to sample from.
            rng (RandomState): Random number generator.
            obs_samplers (List[DistributionSampler]): Sampler for each topic (emission) distribution.
            init_w (np.ndarray): Initial state weights.
            transitions (np.ndarray): Transition probability matrix.
            len_sampler (Optional[DistributionSampler]): Sampler for the number of children of a node.

        """
        self.num_states = dist.num_states
        self.dist = dist
        self.rng = RandomState(seed)
        self.obs_samplers = [topic.sampler(seed=self.rng.randint(maxrandint)) for topic in dist.topics]
        self.init_w = dist.w
        self.transitions = dist.transitions

        if dist.len_dist is not None:
            self.len_sampler = dist.len_dist.sampler(seed=self.rng.randint(0, maxrandint))
        else:
            self.len_sampler = None

        # Guard against pathological exponential tree growth. sample_tree() branches len_dist-many
        # children at every level up to terminal_level, so a len_dist whose mean exceeds 1 child
        # explodes (e.g. len_dist={4:1.0}, terminal_level=10 -> ~4**10 nodes per tree, which looks
        # like a hang). Estimate the expected per-tree node count from the branching mean and fail
        # fast with an actionable message. A throwaway sampler is used so the real RNG is untouched.
        if self.len_sampler is not None:
            levels = int(self.dist.terminal_level)
            try:
                draws = np.asarray(dist.len_dist.sampler(seed=0).sample(size=4096), dtype=float)
                mean_children = float(np.mean(draws))
            except (TypeError, ValueError):
                mean_children = float("nan")  # len_dist yields no numeric child counts -> cannot estimate
            if np.isfinite(mean_children) and mean_children > 1.0 and levels > 0:
                expected_nodes = (mean_children ** (levels + 1) - 1.0) / (mean_children - 1.0)
                if expected_nodes > 1.0e6:
                    raise ValueError(
                        "TreeHiddenMarkovSampler would generate ~%.2g nodes per tree: len_dist has mean "
                        "%.3g children per node (>1) with terminal_level=%d, an exponential blow-up. Use a "
                        "len_dist with mass on 0 so the mean is <= 1 child (branching terminates), and/or a "
                        "smaller terminal_level." % (expected_nodes, mean_children, levels)
                    )

    def sample_state(self, given_state: int, size: int | None = None) -> int | np.ndarray:
        """Draw child state(s) from the transition matrix row of a given parent state.

        Args:
            given_state (int): State of the parent node.
            size (Optional[int]): Number of child states to draw. If None, a single int is returned.

        Returns:
            Single state (int) if size is None, else numpy array of size states.

        """
        return self.rng.choice(self.num_states, p=self.transitions[given_state, :], replace=True, size=size)

    def sample_tree(self, size: int | None = None):
        """Sample rooted tree(s) of ((node_id, parent_id), emission) tuples.

        The root state is drawn from the initial weights, each child state from the transition matrix,
        and the number of children of each node from the length sampler. Sampling stops once the
        terminal_level of the distribution is reached.

        Args:
            size (Optional[int]): Number of trees to draw. If None, a single tree is returned.

        Returns:
            A single tree (list of ((node_id, parent_id), emission) tuples with root parent -1) if size
            is None, else a list of size trees.

        """
        if size is None:
            seq = []
            xi = 0
            zi = self.rng.choice(self.num_states, p=self.init_w)
            ni = self.len_sampler.sample()
            nodes = [(xi, zi, ni)]
            y0 = self.obs_samplers[zi].sample()

            seq.append(((0, -1), y0))
            iter_cond = True if ni > 0 else False

            cnt = 1
            lvl_cnt = 0

            while iter_cond and lvl_cnt < self.dist.terminal_level:
                nodes_next = []
                for node in nodes:
                    xi, zi, ni = node

                    zj = self.sample_state(given_state=zi, size=ni)
                    nj = self.len_sampler.sample(size=ni)

                    for j in range(ni):
                        if nj[j] > 0:
                            nodes_next.append((cnt + j, zj[j], nj[j]))
                        seq.append(((cnt + j, xi), self.obs_samplers[zj[j]].sample()))
                    cnt += ni
                if len(nodes_next) == 0:
                    iter_cond = False
                else:
                    nodes = [xx for xx in nodes_next]

                lvl_cnt += 1

            return seq

        else:
            return [self.sample_tree() for xx in range(size)]

    def sample(self, size: int | None = None):
        """Draw iid tree observations from the tree HMM.

        Args:
            size (Optional[int]): Number of trees to draw. If None, a single tree is returned.

        Returns:
            A single tree if size is None, else a list of size trees. See sample_tree().

        Raises:
            RuntimeError: If no length sampler is available for the number of children.

        """
        if self.len_sampler is not None:
            return self.sample_tree(size=size)
        else:
            raise RuntimeError("TreeHiddenMarkovSampler requires either a length distribution for number of children.")


class TreeHiddenMarkovAccumulator(SequenceEncodableStatisticAccumulator):
    """Accumulator for the tree HMM. Tracks initial-state, state, and transition counts plus emission stats."""

    def __init__(
        self,
        accumulators: Sequence[SequenceEncodableStatisticAccumulator],
        len_accumulator: SequenceEncodableStatisticAccumulator | None = NullAccumulator(),
        keys: tuple[str | None, str | None, str | None] = (None, None, None),
        name: str | None = None,
        use_numba: bool = True,
    ) -> None:
        """Create an accumulator for tree-HMM sufficient statistics.

        Args:
            accumulators (Sequence[SequenceEncodableStatisticAccumulator]): Accumulator for the emission
                distribution of each hidden state.
            len_accumulator (Optional[SequenceEncodableStatisticAccumulator]): Accumulator for the
                number-of-children distribution. Defaults to NullAccumulator.
            keys (Tuple[Optional[str], Optional[str], Optional[str]]): Keys for merging the initial state
                counts, transition counts, and state (emission) accumulators respectively.
            name (Optional[str]): Optional accumulator name.
            use_numba (bool): If True, encoders created from this accumulator use the numba encoding.

        Attributes:
            accumulators (Sequence[SequenceEncodableStatisticAccumulator]): Emission accumulators.
            num_states (int): Number of hidden states.
            init_counts (np.ndarray): Expected root-state counts.
            trans_counts (np.ndarray): Expected state transition counts (num_states by num_states).
            state_counts (np.ndarray): Expected state membership counts.
            len_accumulator (SequenceEncodableStatisticAccumulator): Accumulator for number of children.
            init_key (Optional[str]): Key for merging initial state counts.
            trans_key (Optional[str]): Key for merging transition counts.
            state_key (Optional[str]): Key for merging emission accumulators.
            name (Optional[str]): Optional accumulator name.
            use_numba (bool): If True, encoders created from this accumulator use the numba encoding.

        """
        self.accumulators = accumulators
        self.num_states = len(accumulators)
        self.init_counts = np.zeros(self.num_states, dtype=np.float64)
        self.trans_counts = np.zeros((self.num_states, self.num_states), dtype=np.float64)
        self.state_counts = np.zeros(self.num_states, dtype=np.float64)
        self.len_accumulator = len_accumulator if len_accumulator is not None else NullAccumulator()

        self.init_key = keys[0]
        self.trans_key = keys[1]
        self.state_key = keys[2]

        self.name = name
        self.use_numba = use_numba

        # When _track_ll is enabled, seq_update accumulates the per-tree data
        # log-likelihood into _seq_ll. Used by the fused-EM fast path in
        # optimize(reuse_estep_ll=True); default path is unchanged and zero-cost.
        self._track_ll = False
        self._seq_ll = 0.0

        # protected for initialization.
        self._init_rng: bool = False
        self._len_rng: RandomState | None = None
        self._acc_rng: list[RandomState] | None = None
        self._idx_rng: RandomState | None = None

    def update(self, x: Sequence[tuple[D, T]], weight: float, estimate: TreeHiddenMarkovModelDistribution) -> None:
        """Update sufficient statistics with a single weighted tree observation.

        Encodes the tree and delegates to seq_update().

        Args:
            x (Sequence[Tuple[D, T]]): Tree as ((node_id, parent_id), emission) tuples (root parent -1).
            weight (float): Weight for observation.
            estimate (TreeHiddenMarkovModelDistribution): Previous estimate used for the E-step.

        """
        enc_x = estimate.dist_to_encoder().seq_encode([x])
        self.seq_update(enc_x, np.asarray([weight]), estimate)

    def _rng_initialize(self, rng: RandomState) -> None:
        """Seed the member random number generators used by initialize()/seq_initialize().

        Args:
            rng (RandomState): Random number generator used to draw the member seeds.

        """
        rng_seeds = rng.randint(maxrandint, size=2 + self.num_states)
        self._idx_rng = RandomState(seed=rng_seeds[0])
        self._len_rng = RandomState(seed=rng_seeds[1])
        self._acc_rng = [RandomState(seed=rng_seeds[2 + i]) for i in range(self.num_states)]
        self._w_rng = RandomState(seed=rng.randint(2**30))
        self._init_rng = True

    def initialize(self, x: Sequence[tuple[D, T]], weight: float, rng: RandomState) -> None:
        """Initialize sufficient statistics with a single weighted tree observation.

        Args:
            x (Sequence[Tuple[D, T]]): Tree as ((node_id, parent_id), emission) tuples (root parent -1).
            weight (float): Weight for observation.
            rng (RandomState): Random number generator used to draw random state assignments.

        """
        if not self._init_rng:
            self._rng_initialize(rng)

        enc_x = self.acc_to_encoder().seq_encode([x])
        self.seq_initialize(enc_x, weights=np.asarray([weight]), rng=rng)

    def seq_initialize(self, x: E, weights: np.ndarray, rng: np.random.RandomState) -> None:
        """Vectorized initialization of sufficient statistics from sequence encoded trees.

        Hidden states are assigned uniformly at random and the initial-state, state, transition, emission,
        and length statistics are accumulated under that random assignment.

        Args:
            x (E): Sequence encoded trees from TreeHiddenMarkovDataEncoder.seq_encode().
            weights (np.ndarray): Weights for each encoded tree.
            rng (np.random.RandomState): Random number generator used to draw state assignments.

        """

        if not self._init_rng:
            self._rng_initialize(rng)

        if x[0] is not None:
            tz, _, (xbi, xp, xc, xl, txz, tp, tpz), enc_x, len_enc = x[0]

            states = self._idx_rng.choice(self.num_states, replace=True, size=tz[-1])

            numba_initialize(
                tz, txz, tp, tpz, xp, xc, states, weights, self.init_counts, self.state_counts, self.trans_counts
            )

            idx = len_enc[0]

            for i in range(self.num_states):
                w = weights[idx].copy()
                w[states != i] = 0.0
                self.accumulators[i].seq_initialize(enc_x, w, self._acc_rng[i])

            if len_enc is not None:
                self.len_accumulator.seq_initialize(len_enc[1], weights[len_enc[0]], self._len_rng)

        else:
            cnt, tz, _, (idx, xbi, xp, xc, level_idx, p_nxt, eta_p, i_nxt, rns, rni), enc_x, len_enc = x[1]

            num_states = self.num_states
            states = self._idx_rng.choice(self.num_states, replace=True, size=cnt)

            #  Get root node states
            root_states = np.bincount(states[rns], weights=weights[rni], minlength=num_states)
            self.init_counts += root_states
            self.state_counts += root_states

            # count state transitions by the levels
            ns2 = num_states**2
            for level in range(len(level_idx) - 1, -1, -1):
                lidx = level_idx[level]
                idxs, xps, xcs = idx[lidx], xp[lidx], xc[lidx]

                _, xps_cnt = np.unique(xps, return_counts=True)
                bin_weights = []
                bin_weights.extend([weights[kk] for kk in idxs])

                arr = np.asarray([states[xps], states[xcs]], dtype=np.int32)
                multi_idx = np.ravel_multi_index(arr, (num_states, num_states))

                trans_cnts = np.bincount(multi_idx, weights=bin_weights, minlength=ns2)
                self.trans_counts += np.reshape(trans_cnts, (num_states, num_states))

            obs_idx = len_enc[0]

            for i in range(self.num_states):
                w = weights[obs_idx].copy()
                w[states != i] = 0.0
                self.accumulators[i].seq_initialize(enc_x, w, self._acc_rng[i])

            if len_enc is not None:
                self.len_accumulator.seq_initialize(len_enc[1], weights[len_enc[0]], self._len_rng)

    def seq_update(self, x: E, weights: np.ndarray, estimate: TreeHiddenMarkovModelDistribution) -> None:
        """Vectorized E-step update of sufficient statistics from sequence encoded trees.

        Runs the upward-downward (Baum-Welch style) recursion under the previous estimate to obtain
        expected initial-state, state membership, and transition counts, and passes the node posteriors
        as weights to the emission accumulators (and tree weights to the length accumulator).

        Args:
            x (E): Sequence encoded trees from TreeHiddenMarkovDataEncoder.seq_encode().
            weights (np.ndarray): Weights for each encoded tree.
            estimate (TreeHiddenMarkovModelDistribution): Previous estimate used for the E-step.

        """
        if x[0] is not None:
            tz, (max_level, xln, xlnl, tlnz), (xbi, xp, xc, xl, txz, tp, tpz), enc_x, len_enc = x[0]

            tot_cnt = tz[-1]
            num_states = estimate.num_states
            w = estimate.w
            a_mat = estimate.transitions
            num_trees = len(tz) - 1

            p_level = np.zeros((max_level + 1, num_states), dtype=np.float64)

            level_state_prob(max_level + 1, num_states, a_mat, w, p_level)
            pr_obs = np.zeros((tot_cnt, num_states), dtype=np.float64)

            # Compute state likelihood vectors and scale the max to one
            for i in range(num_states):
                pr_obs[:, i] = estimate.topics[i].seq_log_density(enc_x)

            pr_max0 = pr_obs.max(axis=1)
            pr_obs -= pr_max0[:, None]
            np.exp(pr_obs, out=pr_obs)

            # When the fused-EM fast path requests it, compute the per-tree data log-likelihood
            # from the already-scored emissions via the (read-only) forward kernel, reusing pr_obs
            # so no emissions are re-scored. Done before Baum-Welch. Matches seq_log_density exactly.
            if self._track_ll:
                ll_ret = np.zeros(num_trees, dtype=np.float64)
                ll_betas = np.ones_like(pr_obs, dtype=np.float64)
                ll_etas = np.zeros((len(xbi), num_states), dtype=np.float64)
                numba_seq_log_density(
                    num_states,
                    tz,
                    txz,
                    tp,
                    tpz,
                    tlnz,
                    xp,
                    xc,
                    xl,
                    xbi,
                    xln,
                    xlnl,
                    pr_obs,
                    p_level,
                    a_mat,
                    pr_max0,
                    ll_betas,
                    ll_etas,
                    ll_ret,
                )
                single_ll = np.flatnonzero(np.diff(tz) == 1)
                if single_ll.size > 0:
                    r = tz[single_ll]
                    ll_ret[single_ll] += np.log(np.dot(pr_obs[r, :], w)) + pr_max0[r]
                if len_enc is not None and len_enc[1] is not None:
                    len_ll = estimate.len_dist.seq_log_density(len_enc[1])
                    ll_ret = ll_ret + np.bincount(len_enc[0], weights=len_ll, minlength=num_trees)
                self._seq_ll += float(np.dot(weights, ll_ret))

            betas = np.zeros((tot_cnt, num_states), dtype=np.float64)
            etas = np.zeros((len(xbi), num_states), dtype=np.float64)
            alphas = np.zeros((tot_cnt, num_states), dtype=np.float64)
            xi_acc = np.zeros((num_trees, num_states, num_states), dtype=np.float64)
            pi_acc = np.zeros((num_trees, num_states), dtype=np.float64)

            numba_baum_welch(
                num_states,
                tz,
                txz,
                tp,
                tpz,
                tlnz,
                xp,
                xc,
                xl,
                xbi,
                xln,
                xlnl,
                pr_obs,
                p_level,
                a_mat,
                weights,
                betas,
                etas,
                alphas,
                xi_acc,
                pi_acc,
            )

            #  Childless-root trees have no leaf/parent entries and never enter the kernel.
            single = np.flatnonzero(np.diff(tz) == 1)
            if single.size > 0:
                r = tz[single]
                gam = pr_obs[r, :] * w[None, :]
                gam /= gam.sum(axis=1, keepdims=True)
                gam *= weights[single][:, None]
                alphas[r, :] = gam
                pi_acc[single, :] = gam

            self.init_counts += pi_acc.sum(axis=0)
            self.trans_counts += xi_acc.sum(axis=0)

            for i in range(num_states):
                self.accumulators[i].seq_update(enc_x, alphas[:, i], estimate.topics[i])

            self.state_counts += alphas.sum(axis=0)

            if len_enc is not None:
                self.len_accumulator.seq_update(len_enc[1], weights[len_enc[0]], estimate.len_dist)

        else:
            ## numpy calculation from encoding
            cnt, tz, (xln, xlnl, xlni), (idx, xbi, xp, xc, level_idx, p_nxt, eta_p, i_nxt, rns, rni), enc_x, len_enc = (
                x[1]
            )

            num_states = estimate.num_states
            max_level = len(level_idx)
            a_mat = estimate.transitions
            w = estimate.w
            num_trees = len(tz) - 1

            betas = np.ones((cnt, num_states), dtype=np.float64)
            etas = np.zeros((len(xbi), num_states), dtype=np.float64)
            alphas = np.zeros((cnt, num_states), dtype=np.float64)

            p_level = np.zeros((max_level + 1, num_states), dtype=np.float64)
            p_level[0, :] += w

            for level in range(1, max_level + 1):
                p_level[level, :] += np.matmul(p_level[level - 1, :], a_mat)

            pr_obs = np.zeros((cnt, num_states), dtype=np.float64)

            # Compute state likelihood vectors and scale the max to one
            for i in range(num_states):
                pr_obs[:, i] = estimate.topics[i].seq_log_density(enc_x)

            pr_max0 = pr_obs.max(axis=1)
            pr_obs -= pr_max0[:, None]
            np.exp(pr_obs, out=pr_obs)

            # When the fused-EM fast path requests it, accumulate the per-tree data log-likelihood
            # inline during the (existing) upward beta pass from the betas_sum normalizers + emission
            # max, matching seq_log_density exactly. The standard path skips this entirely.
            track_ll = self._track_ll
            ll_ret = np.zeros(num_trees, dtype=np.float64) if track_ll else None

            #  set the leaf nodes
            betas[xln, :] *= pr_obs[xln, :] * p_level[xlnl, :]
            betas_sum = np.sum(betas[xln, :], axis=1, keepdims=True)
            betas[xln, :] /= betas_sum

            if track_ll:
                ll_ret += np.bincount(xlni, weights=np.log(betas_sum.flatten()) + pr_max0[xln], minlength=num_trees)

            #  upward pass on betas
            for level in range(len(level_idx) - 1, -1, -1):
                lidx = level_idx[level]
                idxs, xbis, xps, xcs = idx[lidx], xbi[lidx], xp[lidx], xc[lidx]

                #  Get etas
                temp = np.reshape(betas[xcs, :], (-1, num_states, 1))
                temp /= np.reshape(p_level[level + 1, :], (1, num_states, 1))
                temp = np.sum(a_mat.T * temp, axis=1)
                etas[xbis, :] += temp

                # within-segment sums (batch-independent, unlike a cumsum-difference)
                log_etas = np.add.reduceat(np.log(etas[xbis, :]), eta_p[level][:-1], axis=0)

                betas[p_nxt[level], :] *= np.exp(log_etas) * pr_obs[p_nxt[level], :]
                betas[p_nxt[level], :] *= p_level[level, :]
                betas_sum = np.sum(betas[p_nxt[level], :], axis=1, keepdims=True)

                betas[p_nxt[level], :] /= betas_sum

                if track_ll:
                    ll_ret += np.bincount(
                        i_nxt[level], weights=np.log(betas_sum.flatten()) + pr_max0[p_nxt[level]], minlength=num_trees
                    )

            if track_ll:
                if len_enc is not None and len_enc[1] is not None:
                    len_ll = estimate.len_dist.seq_log_density(len_enc[1])
                    ll_ret = ll_ret + np.bincount(len_enc[0], weights=len_ll, minlength=num_trees)
                self._seq_ll += float(np.dot(weights, ll_ret))

            ## alpha (upward pass) set the root nodes
            alphas[rns, :] += betas[rns, :]

            for level in range(len(level_idx)):
                lidx = level_idx[level]
                idxs, xbis, xps, xcs = idx[lidx], xbi[lidx], xp[lidx], xc[lidx]
                weights_loc = np.reshape(weights[idxs], (-1, 1, 1))

                xi0 = np.reshape(alphas[xps, :] / etas[xbis, :], (-1, num_states, 1)) * a_mat
                xi1 = np.reshape(betas[xcs, :] / p_level[level + 1, :], (-1, 1, num_states))
                xi_loc = xi0 * xi1

                xi_loc_sum = xi_loc.sum(axis=1, keepdims=True).sum(axis=2, keepdims=True)
                xi_loc_sum[xi_loc_sum == 0] = 1.0

                temp = xi_loc.sum(axis=1)
                temp_sum = temp.sum(axis=1, keepdims=True)
                temp_sum[temp_sum == 0] = 1.0
                temp /= temp_sum

                xi_loc *= weights_loc / xi_loc_sum

                self.trans_counts += xi_loc.sum(axis=0)
                alphas[xcs, :] += temp

            for i in range(num_states):
                alphas[:, i] *= weights[len_enc[0]]

            self.init_counts += np.sum(alphas[rns, :], axis=0)
            self.state_counts += alphas.sum(axis=0)

            for i in range(num_states):
                self.accumulators[i].seq_update(enc_x, alphas[:, i], estimate.topics[i])

            if len_enc is not None:
                self.len_accumulator.seq_update(len_enc[1], weights[len_enc[0]], estimate.len_dist)

    def seq_update_engine(
        self, x: E, weights: np.ndarray, estimate: TreeHiddenMarkovModelDistribution, engine: Any
    ) -> None:
        """Engine-resident E-step for the pure (non-numba) tree encoding.

        Runs the upward-downward recursion entirely with engine ops (numpy or torch): emission
        scoring, the level-by-level beta/eta upward pass, and the alpha/xi downward pass all live
        on the active engine. Node posteriors and aggregated initial/state/transition counts are
        produced on the engine and converted to host arrays only to feed the child accumulators.
        Mirrors the numpy branch of ``seq_update``.
        """
        from mixle.stats.compute.backend import backend_seq_log_density

        if x[1] is None:
            return
        cnt, tz, (xln, xlnl, xlni), (idx, xbi, xp, xc, level_idx, p_nxt, eta_p, i_nxt, rns, rni), enc_x, len_enc = x[1]

        num_states = estimate.num_states
        max_level = len(level_idx)
        weights_np = np.asarray(engine.to_numpy(weights) if hasattr(engine, "to_numpy") else weights, dtype=np.float64)
        w_eng = engine.asarray(weights_np)

        a_mat = engine.asarray(estimate.transitions)
        a_mat_t = engine.asarray(estimate.transitions.T)
        p_levels = [engine.asarray(estimate.w)]
        for _ in range(1, max_level + 1):
            p_levels.append(engine.matmul(p_levels[-1], a_mat))
        p_level = engine.stack(p_levels, axis=0)

        emission_scores = [backend_seq_log_density(topic, enc_x, engine) for topic in estimate.topics]
        log_pr_obs = engine.stack(emission_scores, axis=1)
        pr_max0 = engine.max(log_pr_obs, axis=1)
        pr_obs = engine.exp(log_pr_obs - pr_max0[:, None])

        betas = engine.zeros((cnt, num_states)) + engine.asarray(1.0)
        etas = engine.zeros((len(xbi), num_states))

        # --- upward pass: betas (node->root) and etas (branch messages) ---
        if len(xln):
            leaf_idx = engine.asarray(xln)
            leaf_level = engine.asarray(xlnl)
            betas[leaf_idx, :] = pr_obs[leaf_idx, :] * p_level[leaf_level, :]
            betas_sum = engine.sum(betas[leaf_idx, :], axis=1, keepdims=True)
            betas[leaf_idx, :] = betas[leaf_idx, :] / betas_sum

        for level in range(max_level - 1, -1, -1):
            xbis = xbi[level_idx[level]]
            xcs = xc[level_idx[level]]
            if len(xbis) == 0:
                continue
            xbis_idx = engine.asarray(xbis)
            child_idx = engine.asarray(xcs)
            child_beta = betas[child_idx, :] / p_level[level + 1, :]
            temp = engine.sum(a_mat_t[None, :, :] * child_beta[:, :, None], axis=1)
            etas[xbis_idx, :] = etas[xbis_idx, :] + temp

            log_eta_rows = engine.log(etas[xbis_idx, :])
            log_eta_parts = [
                engine.sum(log_eta_rows[int(start) : int(stop), :], axis=0)
                for start, stop in zip(eta_p[level][:-1], eta_p[level][1:])
            ]
            log_etas = engine.stack(log_eta_parts, axis=0) if log_eta_parts else engine.zeros((0, num_states))

            parent_idx = engine.asarray(p_nxt[level])
            betas[parent_idx, :] = (
                betas[parent_idx, :] * engine.exp(log_etas) * pr_obs[parent_idx, :] * p_level[level, :]
            )
            betas_sum = engine.sum(betas[parent_idx, :], axis=1, keepdims=True)
            betas[parent_idx, :] = betas[parent_idx, :] / betas_sum

        # --- downward pass: alphas (node posteriors) and xi (transition) counts ---
        alphas = engine.zeros((cnt, num_states))
        rns_idx = engine.asarray(rns)
        alphas[rns_idx, :] = alphas[rns_idx, :] + betas[rns_idx, :]
        trans_acc = engine.zeros((num_states, num_states))
        one = engine.asarray(1.0)
        zero = engine.asarray(0.0)

        for level in range(max_level):
            lidx = level_idx[level]
            idxs, xbis, xps, xcs = idx[lidx], xbi[lidx], xp[lidx], xc[lidx]
            if len(xbis) == 0:
                continue
            xbis_idx = engine.asarray(xbis)
            xps_idx = engine.asarray(xps)
            xcs_idx = engine.asarray(xcs)
            weights_loc = w_eng[engine.asarray(idxs)].reshape((-1, 1, 1))

            xi0 = (alphas[xps_idx, :] / etas[xbis_idx, :]).reshape((-1, num_states, 1)) * a_mat
            xi1 = (betas[xcs_idx, :] / p_level[level + 1, :]).reshape((-1, 1, num_states))
            xi_loc = xi0 * xi1

            xi_loc_sum = engine.sum(engine.sum(xi_loc, axis=1), axis=1).reshape((-1, 1, 1))
            xi_loc_sum = engine.where(xi_loc_sum == 0, one, xi_loc_sum)

            temp = engine.sum(xi_loc, axis=1)
            temp_sum = engine.sum(temp, axis=1).reshape((-1, 1))
            temp_sum = engine.where(temp_sum == 0, one, temp_sum)
            temp = temp / temp_sum

            xi_loc = xi_loc * (weights_loc / xi_loc_sum)
            trans_acc = trans_acc + engine.sum(xi_loc, axis=0)
            alphas[xcs_idx, :] = alphas[xcs_idx, :] + temp

        node_w = w_eng[engine.asarray(np.asarray(len_enc[0], dtype=np.int64))]
        alphas = alphas * node_w[:, None]

        init_acc = engine.sum(alphas[rns_idx, :], axis=0)
        state_acc = engine.sum(alphas, axis=0)

        self.init_counts += np.asarray(engine.to_numpy(init_acc))
        self.trans_counts += np.asarray(engine.to_numpy(trans_acc))
        self.state_counts += np.asarray(engine.to_numpy(state_acc))

        alphas_np = np.asarray(engine.to_numpy(alphas))
        for i in range(num_states):
            self.accumulators[i].seq_update(enc_x, alphas_np[:, i], estimate.topics[i])

        if len_enc is not None:
            self.len_accumulator.seq_update(len_enc[1], weights_np[np.asarray(len_enc[0])], estimate.len_dist)

    def combine(
        self, suff_stat: tuple[int, np.ndarray, np.ndarray, np.ndarray, Sequence[SS0], SS1 | None]
    ) -> "TreeHiddenMarkovAccumulator":
        """Combine sufficient statistics from another accumulator into this one.

        Args:
            suff_stat (Tuple): Tuple of number of states, initial-state counts, state counts, transition
                counts, emission sufficient statistics per state, and length sufficient statistics.

        Returns:
            Self, with aggregated sufficient statistics.

        """
        num_states, init_counts, state_counts, trans_counts, acc_values, len_acc_value = suff_stat

        self.init_counts += init_counts
        self.state_counts += state_counts
        self.trans_counts += trans_counts

        for i in range(self.num_states):
            self.accumulators[i].combine(acc_values[i])

        if len_acc_value is not None:
            self.len_accumulator.combine(len_acc_value)

        return self

    def value(self) -> tuple[int, np.ndarray, np.ndarray, np.ndarray, Sequence[Any], Any | None]:
        """Returns sufficient statistics as a Tuple of number of states, initial-state counts, state
        counts, transition counts, emission sufficient statistics per state, and length statistics."""
        len_val = self.len_accumulator.value()

        return (
            self.num_states,
            self.init_counts,
            self.state_counts,
            self.trans_counts,
            tuple([u.value() for u in self.accumulators]),
            len_val,
        )

    def from_value(
        self, x: tuple[int, np.ndarray, np.ndarray, np.ndarray, Sequence[SS0], SS1 | None]
    ) -> "TreeHiddenMarkovAccumulator":
        """Set sufficient statistics of accumulator from value x.

        Args:
            x (Tuple): Tuple of number of states, initial-state counts, state counts, transition counts,
                emission sufficient statistics per state, and length sufficient statistics.

        Returns:
            Self, with sufficient statistics set to x.

        """
        num_states, init_counts, state_counts, trans_counts, accumulators, len_acc = x
        self.num_states = num_states
        self.init_counts = init_counts
        self.state_counts = state_counts
        self.trans_counts = trans_counts

        for i, v in enumerate(accumulators):
            self.accumulators[i].from_value(v)

        if self.len_accumulator is not None:
            self.len_accumulator.from_value(len_acc)

        return self

    def scale(self, c: float) -> "TreeHiddenMarkovAccumulator":
        """Scale all accumulated tree-HMM sufficient statistics in place."""
        self.init_counts *= c
        self.state_counts *= c
        self.trans_counts *= c
        for acc in self.accumulators:
            acc.scale(c)
        if self.len_accumulator is not None:
            self.len_accumulator.scale(c)
        return self

    def key_merge(self, stats_dict: dict[str, Any]) -> None:
        """Merge keyed sufficient statistics into stats_dict (and recurse into member accumulators).

        Args:
            stats_dict (Dict[str, Any]): Dictionary mapping keys to shared sufficient statistics.

        """
        if self.init_key is not None:
            if self.init_key in stats_dict:
                stats_dict[self.init_key] += self.init_counts
            else:
                stats_dict[self.init_key] = self.init_counts

        if self.trans_key is not None:
            if self.trans_key in stats_dict:
                stats_dict[self.trans_key] += self.trans_counts
            else:
                stats_dict[self.trans_key] = self.trans_counts

        if self.state_key is not None:
            if self.state_key in stats_dict:
                acc = stats_dict[self.state_key]
                for i in range(len(acc)):
                    acc[i] = acc[i].combine(self.accumulators[i].value())
            else:
                stats_dict[self.state_key] = self.accumulators

        for u in self.accumulators:
            u.key_merge(stats_dict)

        if self.len_accumulator is not None:
            self.len_accumulator.key_merge(stats_dict)

        return None

    def key_replace(self, stats_dict: dict[str, Any]) -> None:
        """Replace keyed sufficient statistics with values from stats_dict (and recurse into members).

        Args:
            stats_dict (Dict[str, Any]): Dictionary mapping keys to shared sufficient statistics.

        """
        if self.init_key is not None:
            if self.init_key in stats_dict:
                self.init_counts = stats_dict[self.init_key]

        if self.trans_key is not None:
            if self.trans_key in stats_dict:
                self.trans_counts = stats_dict[self.trans_key]

        if self.state_key is not None:
            if self.state_key in stats_dict:
                self.accumulators = stats_dict[self.state_key]

        for u in self.accumulators:
            u.key_replace(stats_dict)

        if self.len_accumulator is not None:
            self.len_accumulator.key_replace(stats_dict)

        return None

    def acc_to_encoder(self) -> "TreeHiddenMarkovDataEncoder":
        """Return a data encoder built from the emission and length accumulators."""
        emission_encoder = self.accumulators[0].acc_to_encoder()
        len_encoder = self.len_accumulator.acc_to_encoder()

        return TreeHiddenMarkovDataEncoder(
            emission_encoder=emission_encoder, len_encoder=len_encoder, use_numba=self.use_numba
        )


class TreeHiddenMarkovAccumulatorFactory(StatisticAccumulatorFactory):
    """Factory for tree hidden Markov accumulators."""

    def __init__(
        self,
        factories: Sequence[StatisticAccumulatorFactory],
        len_factory: StatisticAccumulatorFactory = NullAccumulatorFactory(),
        keys: tuple[str | None, str | None, str | None] | None = (None, None, None),
        name: str | None = None,
        use_numba: bool = True,
    ) -> None:
        """Create a factory for tree hidden Markov accumulators.

        Args:
            factories (Sequence[StatisticAccumulatorFactory]): Factory for the emission accumulator of
                each hidden state.
            len_factory (StatisticAccumulatorFactory): Factory for the number-of-children accumulator.
                Defaults to NullAccumulatorFactory.
            keys (Optional[Tuple[Optional[str], Optional[str], Optional[str]]]): Keys for merging the
                initial state counts, transition counts, and state (emission) accumulators respectively.
            name (Optional[str]): Optional accumulator name.
            use_numba (bool): If True, created accumulators use the numba encoding.

        """
        self.factories = factories
        self.keys = keys if keys is not None else (None, None, None)
        self.len_factory = len_factory
        self.name = name
        self.use_numba = use_numba

    def make(self) -> "TreeHiddenMarkovAccumulator":
        """Return a new tree hidden Markov accumulator."""
        len_acc = self.len_factory.make() if self.len_factory is not None else None
        return TreeHiddenMarkovAccumulator(
            [self.factories[i].make() for i in range(len(self.factories))],
            len_accumulator=len_acc,
            keys=self.keys,
            name=self.name,
            use_numba=self.use_numba,
        )


class TreeHiddenMarkovEstimator(ParameterEstimator):
    """Estimator for the TreeHiddenMarkovModelDistribution from aggregated sufficient statistics."""

    def __init__(
        self,
        estimators: list[ParameterEstimator],
        len_estimator: ParameterEstimator | None = NullEstimator(),
        pseudo_count: tuple[float | None, float | None] | None = (None, None),
        name: str | None = None,
        keys: tuple[str | None, str | None, str | None] | None = (None, None, None),
        use_numba: bool = True,
    ) -> None:
        """Create an estimator for a tree hidden Markov model.

        Args:
            estimators (List[ParameterEstimator]): Estimator for the emission distribution of each hidden
                state. The number of states is set to len(estimators).
            len_estimator (Optional[ParameterEstimator]): Estimator for the number-of-children
                distribution. Defaults to NullEstimator.
            pseudo_count (Optional[Tuple[Optional[float], Optional[float]]]): Pseudo-counts used to smooth
                the initial state weights and the transition matrix rows respectively.
            name (Optional[str]): Optional name assigned to estimated distributions.
            keys (Optional[Tuple[Optional[str], Optional[str], Optional[str]]]): Keys for merging the
                initial state counts, transition counts, and state (emission) accumulators respectively.
            use_numba (bool): If True, the estimated distribution and accumulators use the numba encoding.

        """
        self.num_states = len(estimators)
        self.estimators = estimators
        self.pseudo_count = pseudo_count if pseudo_count is not None else (None, None)
        self.keys = keys if keys is not None else (None, None, None)
        self.len_estimator = len_estimator if len_estimator is not None else NullEstimator()
        self.name = name
        self.use_numba = use_numba

    def accumulator_factory(self) -> TreeHiddenMarkovAccumulatorFactory:
        """Return an accumulator factory configured from this estimator."""
        est_factories = [u.accumulator_factory() for u in self.estimators]
        len_factory = self.len_estimator.accumulator_factory()
        return TreeHiddenMarkovAccumulatorFactory(
            factories=est_factories, len_factory=len_factory, keys=self.keys, name=self.name, use_numba=self.use_numba
        )

    def estimate(
        self,
        nobs: float | None,
        suff_stat: tuple[int, np.ndarray, np.ndarray, np.ndarray, Sequence[SS0], SS1 | None],
    ) -> "TreeHiddenMarkovModelDistribution":
        """Estimate a TreeHiddenMarkovModelDistribution from sufficient statistics (M-step).

        Initial state weights and transition rows are normalized counts, optionally smoothed with the
        pseudo_count pair. Rows of the transition matrix with no observed transitions are left as zeros
        when no pseudo-count is given. Topics and the length distribution are estimated from their
        respective sufficient statistics.

        Args:
            nobs (Optional[float]): Number of observations.
            suff_stat (Tuple): Tuple of number of states, initial-state counts, state counts, transition
                counts, emission sufficient statistics per state, and length sufficient statistics.

        Returns:
            TreeHiddenMarkovModelDistribution: Estimated distribution.

        """
        num_states, init_counts, state_counts, trans_counts, topic_ss, len_ss = suff_stat

        len_dist = self.len_estimator.estimate(nobs, len_ss)
        topics = [self.estimators[i].estimate(state_counts[i], topic_ss[i]) for i in range(num_states)]

        if self.pseudo_count[0] is not None:
            p1 = self.pseudo_count[0] / float(num_states)
            w = init_counts + p1
            w /= w.sum()
        else:
            init_sum = init_counts.sum()
            w = np.full(num_states, 1.0 / num_states) if init_sum <= 0.0 else init_counts / init_sum

        if self.pseudo_count[1] is not None:
            p2 = self.pseudo_count[1] / float(num_states * num_states)
            transitions = trans_counts + p2
            row_sum = transitions.sum(axis=1, keepdims=True)
            transitions /= row_sum
        else:
            row_sum = trans_counts.sum(axis=1, keepdims=True)
            bad_rows = row_sum.flatten() == 0.0

            if np.any(bad_rows):
                good_rows = ~bad_rows
                transitions = np.zeros_like(trans_counts, dtype=np.float64)
                transitions[good_rows, :] += trans_counts[good_rows, :] / row_sum[good_rows]
            else:
                transitions = trans_counts / row_sum

        return TreeHiddenMarkovModelDistribution(
            topics=topics, w=w, transitions=transitions, len_dist=len_dist, name=self.name, use_numba=self.use_numba
        )


class TreeHiddenMarkovDataEncoder(DataSequenceEncoder):
    """Data encoder for sequences of tree HMM observations (flattens trees into level-indexed arrays)."""

    def __init__(
        self,
        emission_encoder: DataSequenceEncoder,
        len_encoder: DataSequenceEncoder | None = NullDataEncoder(),
        use_numba: bool = True,
    ) -> None:
        """Create an encoder for tree HMM observations.

        Args:
            emission_encoder (DataSequenceEncoder): Encoder for the node emissions (data type T).
            len_encoder (Optional[DataSequenceEncoder]): Encoder for the number of children of each node.
                Defaults to NullDataEncoder.
            use_numba (bool): If True, seq_encode() produces the numba-kernel encoding, else the
                pure-numpy level-batched encoding.

        """
        self.emission_encoder = emission_encoder
        self.len_encoder = len_encoder if len_encoder is not None else NullDataEncoder()
        self.use_numba = use_numba

    def __str__(self) -> str:
        """Return a constructor-style representation of the encoder."""
        s1 = repr(self.emission_encoder)
        s2 = repr(self.len_encoder)
        s3 = repr(self.use_numba)
        return "TreeHiddenMarkovDataEncoder(emission_encoder=%s, len_encoder=%s, use_numba=%s)" % (s1, s2, s3)

    def __eq__(self, other: object) -> bool:
        """Return true when ``other`` is a tree-HMM encoder with a matching length encoder.

        Args:
            other (object): Object to compare against.

        Returns:
            True if other is a TreeHiddenMarkovDataEncoder with an equal len_encoder, else False/None.

        """
        if isinstance(other, TreeHiddenMarkovDataEncoder):
            if self.len_encoder == other.len_encoder:
                return True
        else:
            return False

    def _seq_encode(self, x: Sequence[Sequence[tuple[D, T]]]) -> tuple[int, np.ndarray, E5, E6, Any, Any | None]:
        """Encode trees for the pure-numpy implementation (nodes batched across trees by level).

        Args:
            x (Sequence[Sequence[Tuple[D, T]]]): Sequence of trees of ((node_id, parent_id), emission)
                tuples (root parent -1).

        Returns:
            Tuple of total node count, tree slice offsets, leaf-node arrays, level-indexed parent/child
            arrays, encoded emissions, and the optional (tree index, encoded child counts) pair.

        """
        xs = []  # flattened values of nodes in order encoded
        obs_idx = []  #  tree seq idx for observed flattened nodes
        idx = []  # idx for node observation by tree in seq used in betas
        tz = [0]  #  Track entries in beta by observation.
        #  Encodings for the beta pass
        xln = []  # leaf nodes
        xlnl = []  # levels for the leaf nodes
        xlni = []
        root_id = []
        root_node = []  # flattened index of each (non-empty) tree's root node

        xbi = []  # Use this to track beta_j(p(u), u)
        xp = []  # parents, repeated for each child
        xl = []  # level of xc below
        xc = []  # children of xp

        nc = []  # number of children for a given node.

        cnt = 0
        eta_cnt = 0
        for i, xx in enumerate(x):
            n = len(xx)
            tz.append(n)
            if n > 0:
                root_id.append(i)
                root_node.append(cnt)  # flattened index of this tree's root node

            xi0 = np.asarray([v[0][0] for v in xx], dtype=np.int32)
            xp0 = np.asarray([v[0][1] for v in xx], dtype=np.int32)

            p_sort = np.argsort(xp0)

            xc0 = np.asarray([xx[i][0][0] for i in p_sort[1:]], dtype=np.int32)
            ## relabel entries to be 0,1,2,3,....,n-1
            xi0 = xi0[p_sort] + cnt
            xp0 = xp0[p_sort]

            xs.extend([xx[i][1] for i in p_sort])

            u0, u1 = np.unique(xp0[1:], return_counts=True)

            #  beta parent/child combos
            if len(u1) > 0:
                for j in range(len(u1)):
                    xp.extend([u0[j] + cnt] * u1[j])
                    xc.extend(cnt + xc0[np.flatnonzero(xp0[1:] == u0[j])])

            if len(xp0) > 1:
                xbi.extend([kk + eta_cnt for kk in range(len(xp0) - 1)])
                eta_cnt += len(xp0) - 1

                xl_temp = find_level(xp0)
                xl.extend(xl_temp)
                xln_temp = np.delete(np.arange(n), u0)
                xlnl.extend([xl_temp[np.flatnonzero(xc0 == x)[0]] for x in xln_temp])
                xlni.extend([i] * len(xln_temp))
                xln.extend(xln_temp + cnt)
                idx.extend([i] * len(xl_temp))

            elif n == 1:
                #  Childless root: score it as a level-0 leaf (its marginal uses p_level[0] = w).
                xln.append(cnt)
                xlnl.append(0)
                xlni.append(i)

            #  Length distribution
            nc_temp = np.zeros(n, dtype=np.int32)
            nc_temp[u0] = u1
            nc.extend(nc_temp)
            obs_idx.extend([i] * n)

            cnt += n

        idx = np.asarray(idx, dtype=np.int32)
        xbi = np.asarray(xbi, dtype=np.int32)
        xp = np.asarray(xp, dtype=np.int32)
        xc = np.asarray(xc, dtype=np.int32)
        xl = np.asarray(xl, dtype=np.int32)
        xln = np.asarray(xln, dtype=np.int32)
        xlnl = np.asarray(xlnl, dtype=np.int32)
        xlni = np.asarray(xlni, dtype=np.int32)
        root_idx = np.asarray(root_id, dtype=np.int32)
        #  Root node (flattened index) and owning tree index for every non-empty tree, including the
        #  childless single-node trees that never appear as a parent in the level arrays.
        rns = np.asarray(root_node, dtype=np.int32)
        rni = root_idx

        level_idx = []
        eta_p = []
        p_nxt = []
        i_nxt = []

        max_level = int(np.max(xl)) if len(xl) > 0 else 0
        for level in range(1, max_level + 1):
            level_idx.append(np.flatnonzero(xl == level))
            #  Unique parents at this level (sorted, grouped by parent) and the tree index of each.
            u0, first, u1 = np.unique(xp[level_idx[-1]], return_index=True, return_counts=True)
            eta_p.append(np.cumsum(np.append([0], u1)))
            p_nxt.append(u0)
            i_nxt.append(idx[level_idx[-1]][first])  # tree index aligned to p_nxt (per parent)

        enc_x = self.emission_encoder.seq_encode(xs)
        len_enc = self.len_encoder.seq_encode(nc)

        tz = np.cumsum(tz).astype(np.int32)
        obs_idx = np.asarray(obs_idx, dtype=np.int32)

        if len_enc is not None:
            return (
                cnt,
                tz,
                (xln, xlnl, xlni),
                (idx, xbi, xp, xc, level_idx, p_nxt, eta_p, i_nxt, rns, rni),
                enc_x,
                (obs_idx, len_enc),
            )
        else:
            return cnt, tz, (xln, xlnl, xlni), (idx, xbi, xp, xc, level_idx, p_nxt, eta_p, i_nxt, rns, rni), enc_x, None

    def seq_encode(
        self, x: Sequence[Sequence[tuple[D, T]]]
    ) -> tuple[
        tuple[np.ndarray, E1, E2, Any, tuple[np.ndarray, Any] | None] | None,
        tuple[int, np.ndarray, E5, E6, Any, tuple[np.ndarray, Any] | None] | None,
    ]:
        """Encode a sequence of tree observations for vectorized functions.

        Returns a pair (numba_encoding, numpy_encoding) with exactly one entry set, depending on
        use_numba: the numba encoding stores per-tree slice offsets consumed by the numba kernels, while
        the numpy encoding (see _seq_encode()) batches nodes across trees by level.

        Args:
            x (Sequence[Sequence[Tuple[D, T]]]): Sequence of trees of ((node_id, parent_id), emission)
                tuples (root parent -1).

        Returns:
            Tuple (E) with the numba encoding in slot 0 (numpy slot None), or vice versa.

        """
        if self.use_numba:
            xs = []  # flattened values of nodes in order encoded
            tz = [0]  # slice entries for a given observed tree

            #  Encodings for the beta pass
            xln = []  # leaf nodes
            xlnl = []  # levels for the leaf nodes
            tlnz = [0]  # slice leaf nodes for given tree observation
            xbi = []  # Use this to track beta_j(p(u), u)
            xp = []  # parents, repeated for each child
            xl = []  # level of xc below
            xc = []  # children of xp
            txz = [0]  # slice xp, xc, and xl for observed tree
            tp = []  # partition couples of (p, c) for all of a parents children.
            tpz = [0]  # slice tp for an observed tree

            nc = []  # number of children for a given node.

            for i, xx in enumerate(x):
                n = len(xx)

                xi0 = np.asarray([v[0][0] for v in xx], dtype=np.int32)
                xp0 = np.asarray([v[0][1] for v in xx], dtype=np.int32)

                p_sort = np.argsort(xp0)

                xc0 = np.asarray([xx[i][0][0] for i in p_sort[1:]], dtype=np.int32)
                #  relabel entries to be 0,1,2,3,....,n-1
                xi0 = xi0[p_sort]
                xp0 = xp0[p_sort]
                xs.extend([xx[i][1] for i in p_sort])

                u0, u1 = np.unique(xp0[1:], return_counts=True)

                #  beta parent/child combos
                if len(u1) > 0:
                    for j in range(len(u1)):
                        xp.extend([u0[j]] * u1[j])
                        xc.extend(xc0[np.flatnonzero(xp0[1:] == u0[j])])

                    txz.append(np.sum(u1))
                    tp.extend(np.cumsum([0] + list(u1)))
                    tpz.append(len(u1) + 1)

                else:
                    txz.append(0)
                    tp.append(0)
                    tpz.append(1)

                if len(xp0) > 1:
                    xbi.extend([kk for kk in range(len(xp0) - 1)])

                    xl_temp = find_level(xp0)
                    xl.extend(xl_temp)
                    xln_temp = [yy for yy in np.delete(np.arange(n), u0)]
                    xlnl.extend([xl_temp[np.flatnonzero(xc0 == x)[0]] for x in xln_temp])
                    xln.extend(xln_temp)

                    tlnz.append(len(xln_temp))
                else:
                    tlnz.append(0)

                tz.append(n)

                nc_temp = np.zeros(n, dtype=np.int32)
                nc_temp[u0] = u1
                nc.extend(nc_temp)

            #  Length distribution: one weight index per node, grouped by observed tree.
            idx = np.repeat(np.arange(len(x), dtype=np.int32), tz[1:])

            tz = np.cumsum(tz).astype(np.int32)

            xln = np.asarray(xln, dtype=np.int32)
            xlnl = np.asarray(xlnl, dtype=np.int32)
            tlnz = np.cumsum(tlnz).astype(np.int32)

            xbi = np.asarray(xbi, dtype=np.int32)
            xp = np.asarray(xp, dtype=np.int32)
            xc = np.asarray(xc, dtype=np.int32)
            xl = np.asarray(xl, dtype=np.int32)
            txz = np.cumsum(txz).astype(np.int32)
            tp = np.asarray(tp, dtype=np.int32)
            tpz = np.cumsum(tpz).astype(np.int32)

            enc_x = self.emission_encoder.seq_encode(xs)
            len_enc = self.len_encoder.seq_encode(nc)

            # if len_enc is not None:
            len_enc = tuple([np.asarray(idx, np.int32), len_enc])

            max_xln = int(np.max(xln)) if len(xln) > 0 else 0
            return (tz, (max_xln, xln, xlnl, tlnz), (xbi, xp, xc, xl, txz, tp, tpz), enc_x, len_enc), None

        else:
            return None, self._seq_encode(x)


@numba.njit(
    "void(int32, int32[:], int32[:], int32[:], int32[:], int32[:], int32[:], int32[:], int32[:], int32[:], int32[:], "
    "int32[:], float64[:,:], float64[:, :], float64[:, :], float64[:], float64[:,:], float64[:,:], float64[:])",
    fastmath=True,
    parallel=True,
    cache=True,
)
def numba_seq_log_density(
    num_states, tz, txz, tp, tpz, tlnz, xp, xc, xl, xbi, xln, xlnl, pr_obs, p_level, tr_mat, pr_max0, betas, etas, out
):
    """Numba kernel: per-tree log-likelihood via the upward (beta) recursion, written to out."""
    for n in numba.prange(len(tz) - 1):
        #  Observed value slice (xs)
        s0, s1 = tz[n], tz[n + 1]

        if s0 == s1:
            out[n] = 0
            continue

        #  Slice the upward pass
        i0, i1 = txz[n], txz[n + 1]
        if i0 == i1:
            #  Only root node in tree
            beta_sum = 0
            for i in range(num_states):
                temp = pr_obs[s0, i] * p_level[0, i]
                beta_sum += temp
            out[n] = (math.log(beta_sum) if beta_sum > 0.0 else -np.inf) + pr_max0[s0]

        ll_sum = 0.0
        beta_mat = betas[s0:s1, :]
        eta_mat = etas[i0:i1, :]
        b = pr_obs[s0:s1, :]
        b_max = pr_max0[s0:s1]

        #  Start with the leaf nodes (non-parent-nodes).
        j0, j1 = tlnz[n], tlnz[n + 1]
        xlns = xln[j0:j1]
        xlnls = xlnl[j0:j1]

        for k in range(len(xlns)):
            leaf_node = xlns[k]
            leaf_level = xlnls[k]
            beta_sum = 0
            for i in range(num_states):
                temp = b[leaf_node, i] * p_level[leaf_level, i]
                beta_mat[leaf_node, i] *= temp
                beta_sum += temp

            ll_sum += (math.log(beta_sum) if beta_sum > 0.0 else -np.inf) + b_max[leaf_node]
            if beta_sum <= 0.0:  # impossible observation: log above gave -inf; keep beta 0, avoid 0/0 -> NaN
                beta_sum = 1.0

            for i in range(num_states):
                beta_mat[leaf_node, i] /= beta_sum

        #  Slice the upward pass
        xps = xp[i0:i1]
        xcs = xc[i0:i1]
        xls = xl[i0:i1]
        xbis = xbi[i0:i1]

        #  Partitions for the groupings on the betas
        tps = tp[tpz[n] : tpz[n + 1]]

        for nn in range(len(tps) - 2, -1, -1):
            t0, t1 = tps[nn], tps[nn + 1]
            p, level = xps[t0], xls[t0]

            #  Get eta(p, u)_i and sum then get beta_i(p)
            beta_sum = 0
            for i in range(num_states):
                beta_mat[p, i] *= b[p, i] * p_level[level - 1, i]

                for k in range(t0, t1):
                    c = xcs[k]
                    eta_idx = xbis[k]
                    eta_sum = 0

                    for j in range(num_states):
                        eta_sum += beta_mat[c, j] * tr_mat[i, j] / p_level[level, j]

                    eta_mat[eta_idx, i] += eta_sum
                    beta_mat[p, i] *= eta_sum

                beta_sum += beta_mat[p, i]

            ll_sum += (math.log(beta_sum) if beta_sum > 0.0 else -np.inf) + b_max[p]
            if beta_sum <= 0.0:  # impossible subtree: keep beta 0, avoid 0/0 -> NaN
                beta_sum = 1.0

            for i in range(num_states):
                beta_mat[p, i] /= beta_sum

        out[n] = ll_sum


@numba.njit(
    "void(int32, int32[:], int32[:], int32[:], int32[:], int32[:], int32[:], int32[:], int32[:], int32[:], int32[:], "
    "int32[:], float64[:,:], float64[:, :], float64[:, :], float64[:], float64[:,:], float64[:,:], float64[:,:], "
    "float64[:,:, :], float64[:,:])",
    parallel=True,
    cache=True,
)
def numba_baum_welch(
    num_states,
    tz,
    txz,
    tp,
    tpz,
    tlnz,
    xp,
    xc,
    xl,
    xbi,
    xln,
    xlnl,
    pr_obs,
    p_level,
    tr_mat,
    weights,
    betas,
    etas,
    alphas,
    xi_acc,
    pi_acc,
):
    """Numba kernel: upward-downward E-step writing node posteriors (alphas), per-tree expected
    transition counts (xi_acc), and root-state counts (pi_acc)."""
    for n in numba.prange(len(tz) - 1):
        #  Observed value slice (xs)
        s0, s1 = tz[n], tz[n + 1]
        weight_loc = weights[n]

        if s0 == s1:
            continue

        #  Slice the upward pass
        i0, i1 = txz[n], txz[n + 1]

        if i0 == i1:
            #  Only one node with no children, need to handle this. No transition updates just pi_acc
            alpha_sum = 0
            for i in range(num_states):
                temp = pr_obs[s0, i] * p_level[0, i]

                alphas[s0, i] = temp * weight_loc
                alpha_sum += temp

            for i in range(num_states):
                alphas[s0, i] /= alpha_sum
                pi_acc[n, i] += alphas[s0, i]

            continue

        beta_mat = betas[s0:s1, :]
        eta_mat = etas[i0:i1, :]
        b = pr_obs[s0:s1, :]

        #  Start with the leaf nodes (non-parent-nodes).
        j0, j1 = tlnz[n], tlnz[n + 1]
        xlns = xln[j0:j1]
        xlnls = xlnl[j0:j1]

        for k in range(len(xlns)):
            leaf_node = xlns[k]
            leaf_level = xlnls[k]
            beta_sum = 0
            for i in range(num_states):
                temp = b[leaf_node, i] * p_level[leaf_level, i]
                beta_mat[leaf_node, i] = temp
                beta_sum += temp

            for i in range(num_states):
                beta_mat[leaf_node, i] /= beta_sum

        #  Slice the upward pass
        xps = xp[i0:i1]
        xcs = xc[i0:i1]
        xls = xl[i0:i1]
        xbis = xbi[i0:i1]

        #  Partitions for the groupings on the betas
        tps = tp[tpz[n] : tpz[n + 1]]

        for nn in range(len(tps) - 2, -1, -1):
            t0, t1 = tps[nn], tps[nn + 1]
            p, level = xps[t0], xls[t0]

            #  Get eta(p, u)_i and sum then get beta_i(p)
            beta_sum = 0
            for i in range(num_states):
                beta_mat[p, i] = b[p, i] * p_level[level - 1, i]

                for k in range(t0, t1):
                    c = xcs[k]
                    eta_idx = xbis[k]
                    eta_sum = 0

                    for j in range(num_states):
                        eta_sum += beta_mat[c, j] * tr_mat[i, j] / p_level[level, j]

                    eta_mat[eta_idx, i] = eta_sum
                    beta_mat[p, i] *= eta_sum

                beta_sum += beta_mat[p, i]

            for i in range(num_states):
                beta_mat[p, i] /= beta_sum

        ### do the alpha pass
        alpha_mat = alphas[s0:s1, :]
        xi_buff = np.zeros((num_states, num_states), dtype=np.float64)

        #  set the root
        for i in range(num_states):
            alpha_mat[0, i] += beta_mat[0, i] * weight_loc

        for nn in range(0, len(tps) - 1):
            t0, t1 = tps[nn], tps[nn + 1]
            p, level = xps[t0], xls[t0]

            for k in range(t0, t1):
                c, eta_idx = xcs[k], xbis[k]
                xi_buff_sum = 0

                gamma_sum = 0
                for i in range(num_states):
                    alpha_sum = 0
                    for j in range(num_states):
                        temp = tr_mat[j, i] * alpha_mat[p, j] / eta_mat[eta_idx, j]
                        alpha_sum += temp

                        temp *= beta_mat[c, i]
                        temp /= p_level[level, i]

                        xi_buff_sum += temp
                        xi_buff[j, i] = temp

                    alpha_sum *= beta_mat[c, i]
                    alpha_sum /= p_level[level, i]

                    alpha_mat[c, i] += alpha_sum
                    gamma_sum += alpha_sum

                if gamma_sum > 0:
                    gamma_sum = weight_loc / gamma_sum
                if xi_buff_sum > 0:
                    xi_buff_sum = weight_loc / xi_buff_sum
                for i in range(num_states):
                    alpha_mat[c, i] *= gamma_sum
                    for j in range(num_states):
                        xi_acc[n, i, j] += xi_buff[i, j] * xi_buff_sum

        for i in range(num_states):
            pi_acc[n, i] += alpha_mat[0, i]


@numba.njit(
    "void(int32, int32[:], int32[:], int32[:], int32[:], int32[:], int32[:], int32[:], int32[:], int32[:], int32[:], "
    "int32[:], float64[:,:], float64[:, :], float64[:,:], float64[:,:], float64[:,:])",
    fastmath=True,
    parallel=True,
    cache=True,
)
def numba_posteriors(
    num_states, tz, txz, tp, tpz, tlnz, xp, xc, xl, xbi, xln, xlnl, pr_obs, p_level, tr_mat, betas, etas
):
    """Numba kernel: upward (beta) recursion writing normalized per-node state posteriors into betas."""
    for n in numba.prange(len(tz) - 1):
        #  Observed value slice (xs)
        s0, s1 = tz[n], tz[n + 1]

        if s0 == s1:
            continue

        #  Slice the upward pass
        i0, i1 = txz[n], txz[n + 1]

        if i0 == i1:
            #  Only one node with no children, need to handle this. No transition updates just pi_acc
            beta_sum = 0
            for i in range(num_states):
                temp = pr_obs[s0, i] * p_level[0, i]

                betas[s0, i] += temp
                beta_sum += temp

            for i in range(num_states):
                betas[s0, i] /= beta_sum

        beta_mat = betas[s0:s1, :]
        eta_mat = etas[i0:i1, :]
        b = pr_obs[s0:s1, :]

        #  Start with the leaf nodes (non-parent-nodes).
        j0, j1 = tlnz[n], tlnz[n + 1]
        xlns = xln[j0:j1]
        xlnls = xlnl[j0:j1]

        for k in range(len(xlns)):
            leaf_node = xlns[k]
            leaf_level = xlnls[k]
            beta_sum = 0
            for i in range(num_states):
                temp = b[leaf_node, i] * p_level[leaf_level, i]
                beta_mat[leaf_node, i] = temp
                beta_sum += temp

            for i in range(num_states):
                beta_mat[leaf_node, i] /= beta_sum

        #  Slice the upward pass
        xps = xp[i0:i1]
        xcs = xc[i0:i1]
        xls = xl[i0:i1]
        xbis = xbi[i0:i1]

        #  Partitions for the groupings on the betas
        tps = tp[tpz[n] : tpz[n + 1]]

        for nn in range(len(tps) - 2, -1, -1):
            t0, t1 = tps[nn], tps[nn + 1]
            p, level = xps[t0], xls[t0]

            #  Get eta(p, u)_i and sum then get beta_i(p)
            beta_sum = 0
            for i in range(num_states):
                beta_mat[p, i] = b[p, i] * p_level[level - 1, i]

                for k in range(t0, t1):
                    c = xcs[k]
                    eta_idx = xbis[k]
                    eta_sum = 0

                    for j in range(num_states):
                        eta_sum += beta_mat[c, j] * tr_mat[i, j] / p_level[level, j]

                    eta_mat[eta_idx, i] = eta_sum
                    beta_mat[p, i] *= eta_sum

                beta_sum += beta_mat[p, i]

            for i in range(num_states):
                beta_mat[p, i] /= beta_sum


@numba.jit(
    "void(int32[:], int32[:], int32[:], int32[:], int32[:], int32[:], int64[:], float64[:], float64[:], "
    "float64[:], float64[:,:])",
    parallel=True,
    nopython=True,
    cache=True,
)
def numba_initialize(tz, txz, tp, tpz, xp, xc, states, weights, init_counts, state_counts, trans_counts):
    """Numba kernel: accumulate initial-state, state, and transition counts for random state assignments."""
    for n in numba.prange(len(tz) - 1):
        s0, s1 = tz[n], tz[n + 1]

        if s0 == s1:
            continue

        weight_loc = weights[n]
        ss = states[s0:s1]
        init_counts[ss[0]] += weight_loc
        state_counts[ss[0]] += weight_loc

        i0, i1 = txz[n], txz[n + 1]

        if i0 == i1:
            continue

        xps = xp[i0:i1]
        xcs = xc[i0:i1]
        tps = tp[tpz[n] : tpz[n + 1]]

        for nn in range(len(tps) - 1):
            j0, j1 = tps[nn], tps[nn + 1]
            p = ss[xps[j0]]
            for k in range(j0, j1):
                c = ss[xcs[k]]
                trans_counts[p, c] += weight_loc
                state_counts[c] += weight_loc


@numba.njit(
    "void(int32, int32[:], int32[:], int32[:], int32[:], int32[:], int32[:], int32[:], int32[:], int32[:], "
    "int32[:], float64[:,:], float64[:], float64[:,:], float64[:,:], float64[:,:], int32[:])",
    parallel=True,
    cache=True,
)
def numba_viterbi(
    num_states, tz, txz, tp, tpz, tlnz, xp, xc, xl, xbi, xln, log_pr_obs, log_init_p, log_tr_mat, betas, etas, out
):
    """Numba kernel: max-product upward recursion writing the most likely state per node into out."""
    for n in numba.prange(len(tz) - 1):
        #  Observed value slice (xs)
        s0, s1 = tz[n], tz[n + 1]

        if s0 == s1:
            continue

        #  Slice the upward pass
        i0, i1 = txz[n], txz[n + 1]
        outs = out[s0:s1]

        if i0 == i1:
            #  Only one node with no children, need to handle this. No transition updates just pi_acc
            beta_max = None
            beta_max_i = 0
            for i in range(num_states):
                temp = log_pr_obs[s0, i] + log_init_p[i]
                if beta_max is None:
                    beta_max = temp
                    beta_max_i = i
                else:
                    if beta_max < temp:
                        beta_max = temp
                        beta_max_i = i

            outs[0] = beta_max_i

        beta_mat = betas[s0:s1, :]
        eta_mat = etas[i0:i1, :]
        log_b = log_pr_obs[s0:s1, :]

        #  Start with the leaf nodes (non-parent-nodes).
        j0, j1 = tlnz[n], tlnz[n + 1]
        xlns = xln[j0:j1]

        for k in range(len(xlns)):
            leaf_node = xlns[k]
            temp = log_b[leaf_node, 0]
            beta_mat[leaf_node, 0] += temp
            max_leaf_v = temp
            max_leaf_i = 0
            for i in range(1, num_states):
                temp = log_b[leaf_node, i]
                beta_mat[leaf_node, i] += temp

                if max_leaf_v < temp:
                    max_leaf_v = temp
                    max_leaf_i = i

            outs[leaf_node] = max_leaf_i

        #  Slice the upward pass
        xps = xp[i0:i1]
        xcs = xc[i0:i1]
        xls = xl[i0:i1]
        xbis = xbi[i0:i1]

        #  Partitions for the groupings on the betas
        tps = tp[tpz[n] : tpz[n + 1]]

        for nn in range(len(tps) - 2, -1, -1):
            t0, t1 = tps[nn], tps[nn + 1]
            p, level = xps[t0], xls[t0]
            beta_max_v = None
            beta_max_i = None
            #  Get eta(p, u)_i and sum then get beta_i(p)
            for i in range(0, num_states):
                for k in range(t0, t1):
                    c = xcs[k]
                    eta_idx = xbis[k]
                    eta_max = beta_mat[c, 0] + log_tr_mat[i, 0]

                    for j in range(1, num_states):
                        temp = beta_mat[c, j] + log_tr_mat[i, j]
                        eta_max = max(eta_max, temp)

                    eta_mat[eta_idx, i] += eta_max
                    beta_mat[p, i] += log_b[p, i]
                    if beta_max_v is None:
                        beta_max_v = beta_mat[p, i]
                        beta_max_i = i
                    else:
                        if beta_max_v < beta_mat[p, i]:
                            beta_max_v = beta_mat[p, i]
                            beta_max_i = i

            outs[p] = beta_max_i


@numba.njit("float64[:](int32[:], float64[:], float64[:])", parallel=True, cache=True)
def vec_bincount(idx, ll, out):
    """Numba kernel: scatter-add ll into out at positions idx (weighted bincount)."""
    for i in numba.prange(len(idx)):
        out[idx[i]] += ll[i]
    return out


@numba.njit("void(int32, int32, float64[:, :], float64[:], float64[:, :])", cache=True)
def level_state_prob(levels, num_states, tr_mat, init_prob, out):
    """Numba kernel: marginal state probabilities per tree level, out[k] = init_prob @ tr_mat^k."""
    for i in range(num_states):
        out[0, i] = init_prob[i]

    for k in range(1, levels):
        for i in range(num_states):
            for j in range(num_states):
                out[k, i] += out[k - 1, i] * tr_mat[i, j]


def _register_tree_hmm_engine_kernel():
    """Register the engine-resident tree-HMM kernel (idempotent; called at import)."""
    from mixle.stats.compute.kernel import GenericKernel, GenericKernelFactory, KernelFactory, register_kernel_factory

    class TreeHiddenMarkovKernel(GenericKernel):
        def accumulate(self, enc, weights):
            if self.estimator is None:
                raise ValueError("TreeHiddenMarkovKernel.accumulate requires an estimator.")
            if not getattr(self.engine, "resident_estep", True):
                return super().accumulate(enc, weights)
            host_enc = getattr(enc, "host_payload", enc)
            accumulator = self.estimator.accumulator_factory().make()
            accumulator.seq_update_engine(host_enc, weights, self.dist, self.engine)
            return accumulator.value()

    class TreeHiddenMarkovKernelFactory(KernelFactory):
        def build(self, dist, engine, estimator=None):
            if not dist.supports_engine(engine):
                return GenericKernelFactory().build(dist, engine, estimator=estimator)
            return TreeHiddenMarkovKernel(dist, engine=engine, estimator=estimator)

    register_kernel_factory(TreeHiddenMarkovModelDistribution, TreeHiddenMarkovKernelFactory())


_register_tree_hmm_engine_kernel()


# --- Backward-compatible API naming aliases ---
TreeHiddenMarkovModelAccumulator = TreeHiddenMarkovAccumulator
TreeHiddenMarkovModelAccumulatorFactory = TreeHiddenMarkovAccumulatorFactory
TreeHiddenMarkovModelDataEncoder = TreeHiddenMarkovDataEncoder
TreeHiddenMarkovModelEstimator = TreeHiddenMarkovEstimator
TreeHiddenMarkovModelSampler = TreeHiddenMarkovSampler
