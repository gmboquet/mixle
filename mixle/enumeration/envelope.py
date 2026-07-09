"""Approximate deep enumeration for autoregressive models (LLMs): the per-depth envelope index.

The exact autoregressive count index (:mod:`~mixle.enumeration.autoregressive`) is a *tree* recursion:
each prefix has its own next-token distribution, so counting to a bit budget ``B`` must expand every live
prefix -- ``Theta(count / V)`` work. That is the right tool for the head (ranks up to ~1e6 on a real LM),
and provably the end of the road for deep ranks: reaching rank 1e15 exactly would visit ~1e13 prefixes.

This module trades that wall for a **mean-field approximation with an explicit contract**:

* **Precompute** (the envelope): ancestral-sample ``n_paths`` contexts per depth and average their
  next-token *fine-bucket histograms* into one envelope ``E_d`` per depth (depth 0 is the real root
  context, so it is exact). Suffix-convolve them once -- ``S_d = E_d (*) S_{d+1}`` -- with float64
  counts at C speed. ``S_0`` approximates the full count histogram the way a
  :class:`~mixle.stats.combinator.sequence.SequenceDistribution` computes its own **exactly**: the
  approximation is precisely "treat the steps as independent draws from the per-depth aggregate".
* **Query**: ``count`` / ``threshold`` / ``mass_above`` read ``S_0``. ``unrank(i)`` descends the *real*
  model -- one forward per step, V real step buckets at each depth -- apportioning the target offset
  among tokens by their envelope-estimated subtree counts. O(L) forwards per query, never Theta(count).

Contract: the returned sequences are real model outputs and every reported ``log_prob`` is the
**exact** model log-probability; only the *rank coordinate* is approximate. For a prefix-independent
(iid-step) model the envelope equals the true per-step histogram and everything here is exact (tested).
For a context-dependent model the envelope is an estimate averaged over typical (ancestrally sampled)
contexts -- the same high-probability contexts that dominate the enumeration head; ``rank_bracket``
returns the induced bucket bracket so downstream users can carry the uncertainty. Fixed-length models
only (a terminating/eos model needs an absorbing channel -- raise, do not guess).
"""

from __future__ import annotations

import math
from typing import Any

import numpy as np

from mixle.enumeration.quantization.core import _TOL, CountHistogram

__all__ = ["AREnvelopeIndex", "LatticeEnvelopeIndex"]

_LOG2 = math.log(2.0)


class AREnvelopeIndex:
    """Envelope (mean-field) seek index over a fixed-length :class:`AutoregressiveEnumerable`.

    Args:
        model: the autoregressive adapter (``max_len`` set; terminating/eos models are rejected).
        n_paths: contexts sampled per depth for the envelope calibration (more = sharper envelope;
            the root depth is always exact). The calibration forwards are memoized on the model, so
            they are shared with any exact index built later.
        seed: calibration sampling seed (the index is deterministic given it).
        budget_bits: initial depth of the suffix tables; queries deepen geometrically as needed.
        calibration_sequences: optional typical sequences (a corpus, a provider's fast generations) to
            calibrate the envelope from INSTEAD of ancestral sampling. Each is harvested through
            ``model.harvest`` -- with the ``all_position_logprobs`` contract that is ONE forward per
            sequence for all its per-depth contexts, ~L-times cheaper than sampling token by token.
    """

    def __init__(
        self,
        model: Any,
        *,
        n_paths: int = 64,
        seed: int = 0,
        budget_bits: float = 64.0,
        calibration_sequences: list[tuple] | None = None,
    ) -> None:
        if getattr(model, "terminating", False):
            raise ValueError(
                "AREnvelopeIndex supports fixed-length models only; a terminating (eos) model needs an "
                "absorbing-length envelope that is not implemented -- use the exact count index."
            )
        self.model = model
        self.length = int(model._depth)
        self.quantizer = model._quantizer()
        self.n_paths = int(n_paths)
        self.seed = int(seed)
        self._budget_fb = max(1, int(math.ceil(float(budget_bits) * self.quantizer.fine_per_bit())))
        self._envelopes: list[CountHistogram] = []
        self._suffix: list[CountHistogram] = []
        if calibration_sequences is not None:
            self._calibrate_from_sequences([tuple(s) for s in calibration_sequences])
        else:
            self._calibrate()
        self._rebuild_suffix()

    # -- precompute ---------------------------------------------------------------------------------------

    def _step_hist(self, prefix: tuple) -> CountHistogram:
        """The fine-bucket histogram of the real next-token log-probs at ``prefix`` (float counts)."""
        _tokens, lps = self.model._steps_np(prefix)
        scale = self.quantizer.oversample / self.quantizer.bin_width_bits
        sb = np.floor(np.maximum(0.0, -lps / _LOG2) * scale + _TOL).astype(np.int64)
        if sb.size == 0:
            return CountHistogram.empty()
        base = int(sb.min())
        return CountHistogram(base, np.bincount(sb - base).astype(np.float64).tolist())

    def _calibrate(self) -> None:
        """Average sampled contexts' step histograms into one envelope per depth (root depth exact)."""
        rng = np.random.RandomState(self.seed)
        # ancestral prefixes: paths[j] grows token by token; depth d uses the length-d prefixes
        prefixes: list[tuple] = [() for _ in range(self.n_paths)]
        envelopes: list[CountHistogram] = []
        for d in range(self.length):
            if d == 0:
                envelopes.append(self._step_hist(()))  # one real root context: exact
            else:
                acc = CountHistogram.empty()
                seen: dict[tuple, int] = {}
                for p in prefixes:
                    seen[p] = seen.get(p, 0) + 1
                for p, mult in seen.items():  # distinct contexts once; weight by multiplicity
                    h = self._step_hist(p)
                    if mult != 1:
                        h = CountHistogram(h.base, [c * mult for c in h.data])
                    acc = acc.add(h)
                total_paths = float(len(prefixes))
                envelopes.append(CountHistogram(acc.base, [c / total_paths for c in acc.data]))
            if d == self.length - 1:
                break  # no need to extend the sampled paths past the last scored depth
            for j in range(self.n_paths):
                tokens, lps = self.model._steps_np(prefixes[j])
                p = np.exp(lps - lps.max())
                p /= p.sum()
                prefixes[j] = prefixes[j] + (tokens[int(rng.choice(tokens.size, p=p))].item(),)
        self._envelopes = envelopes

    def _calibrate_from_sequences(self, sequences: list[tuple]) -> None:
        """Envelope per depth from user-supplied typical sequences (corpus calibration).

        Each sequence is harvested first (one ``all_position_logprobs`` forward when the model has that
        contract), then depth ``d``'s envelope averages the step histograms of the sequences' length-``d``
        prefixes -- depth 0 stays the exact root context.
        """
        if not sequences:
            raise ValueError("calibration_sequences must be non-empty")
        if any(len(s) < self.length for s in sequences):
            raise ValueError("every calibration sequence must cover the model length %d" % self.length)
        harvest = getattr(self.model, "harvest", None)
        if callable(harvest):
            for seq in sequences:
                harvest(seq[: self.length])
        envelopes: list[CountHistogram] = [self._step_hist(())]  # one real root context: exact
        for d in range(1, self.length):
            acc = CountHistogram.empty()
            seen: dict[tuple, int] = {}
            for seq in sequences:
                p = seq[:d]
                seen[p] = seen.get(p, 0) + 1
            for p, mult in seen.items():
                h = self._step_hist(p)
                if mult != 1:
                    h = CountHistogram(h.base, [c * mult for c in h.data])
                acc = acc.add(h)
            envelopes.append(CountHistogram(acc.base, [c / float(len(sequences)) for c in acc.data]))
        self._envelopes = envelopes

    def _rebuild_suffix(self) -> None:
        """``S_d = E_d (*) S_{d+1}`` (float64, capped at the budget) -- the reusable seek tables."""
        suffix: list[CountHistogram] = [CountHistogram.empty()] * (self.length + 1)
        suffix[self.length] = CountHistogram.delta(0, 1)
        for d in range(self.length - 1, -1, -1):
            suffix[d] = self._envelopes[d].convolve_float(suffix[d + 1], max_fine_bucket=self._budget_fb)
        self._suffix = suffix

    def ensure_bits(self, depth_bits: float) -> AREnvelopeIndex:
        """Deepen the suffix tables to cover ``depth_bits`` (geometric, low-overhead: L capped convolutions)."""
        needed = max(1, int(math.ceil(float(depth_bits) * self.quantizer.fine_per_bit())))
        if needed > self._budget_fb:
            self._budget_fb = max(needed, self._budget_fb * 2)
            self._rebuild_suffix()
        return self

    # -- whole-support estimates off S_0 --------------------------------------------------------------------

    def total(self) -> float:
        """Estimated number of sequences within the built depth (exact for an iid-step model)."""
        return float(self._suffix[0].total())

    def count(self, min_log_prob: float) -> float:
        """Estimated number of sequences with ``log_density >= min_log_prob`` (mean-field; iid-exact)."""
        self.ensure_bits(self.quantizer.bits(min_log_prob) + self.quantizer.bin_width_bits)
        fb = self.quantizer.fine_bucket(min_log_prob)
        hist = self._suffix[0]
        total = 0.0
        for j, c in enumerate(hist.data):
            if hist.base + j > fb:
                break
            total += c
        return float(total)

    def mass_above(self, min_log_prob: float) -> tuple[float, float]:
        """A mean-field ``(lower, upper)`` estimate of the head mass above ``min_log_prob``.

        Same bucket arithmetic as the exact index's ``mass_above`` (each bucket of ``c`` sequences holds
        between ``c * 2**-hi_bits`` and ``c * 2**-lo_bits`` of mass), applied to the envelope histogram --
        so the bracket carries the envelope's estimation error on top of the quantization smear.
        """
        self.ensure_bits(self.quantizer.bits(min_log_prob) + self.quantizer.bin_width_bits)
        cutoff = self.quantizer.fine_bucket(min_log_prob)
        per_bit = self.quantizer.fine_per_bit()
        hist = self._suffix[0]
        lo = hi = 0.0
        for j, c in enumerate(hist.data):
            fb = hist.base + j
            if fb > cutoff:
                break
            if not c:
                continue
            lo += c * 2.0 ** (-(fb + self.length) / per_bit)
            hi += c * 2.0 ** (-fb / per_bit)
        return lo, hi

    # -- unrank: descend the REAL model, apportion by envelope counts ---------------------------------------

    def unrank(self, i: int) -> tuple[tuple, float]:
        """The approximately-``i``-th most probable sequence and its **exact** log-probability.

        Costs one model forward per step (L total, memoized) -- never a tree expansion. The rank
        coordinate inherits the envelope approximation; the returned sequence and its log-probability
        are exact model quantities. Raises ``IndexError`` past the estimated support size.
        """
        if i < 0:
            raise IndexError("rank must be >= 0")
        self.ensure_bits(math.log2(float(i) + 2.0) + 1.0)
        hist = self._suffix[0]
        target = float(i)
        bucket = None
        for j, c in enumerate(hist.data):
            if target < c:
                bucket = hist.base + j
                break
            target -= c
        if bucket is None:
            raise IndexError("rank %d beyond the estimated support (size %.6g)" % (i, self.total()))

        prefix: tuple = ()
        remaining = int(bucket)
        offset = target
        for d in range(self.length):
            tokens, lps = self.model._steps_np(prefix)
            scale = self.quantizer.oversample / self.quantizer.bin_width_bits
            sb = np.floor(np.maximum(0.0, -lps / _LOG2) * scale + _TOL).astype(np.int64)
            if d == self.length - 1:
                # real leaf: the tokens whose own bucket is the remaining budget, in the model's
                # descending-probability order; envelope drift can leave none -- clamp to the nearest.
                exact = np.flatnonzero(sb == remaining)
                if exact.size:
                    j = min(int(offset), exact.size - 1)
                    choice = int(exact[j])
                else:
                    choice = int(np.argmin(np.abs(sb - remaining)))
                prefix = prefix + (tokens[choice].item(),)
                break
            nxt = self._suffix[d + 1]
            chosen = None
            for t_idx in range(tokens.size):  # model order (descending lp): deterministic apportioning
                c = nxt.count_at(remaining - int(sb[t_idx]))
                if offset < c:
                    chosen = t_idx
                    break
                offset -= c
            if chosen is None:
                # envelope over-estimated this bucket: fall into the most probable viable branch
                viable = [t for t in range(tokens.size) if nxt.count_at(remaining - int(sb[t])) > 0]
                chosen = viable[-1] if viable else 0
                offset = 0.0
            remaining -= int(sb[chosen])
            prefix = prefix + (tokens[chosen].item(),)
        return prefix, float(self.model.log_density(prefix))

    def threshold(self, rank: int) -> float:
        """Exact log-probability of the sequence the envelope places at ``rank`` (approximate boundary)."""
        if rank < 1:
            raise ValueError("rank must be >= 1")
        _seq, lp = self.unrank(rank - 1)
        return lp

    def rank_bracket(self, sequence: Any) -> tuple[float, float]:
        """Estimated ``[lo, hi]`` rank bracket of ``sequence`` -- its envelope bucket's rank span.

        ``lo`` counts the estimated sequences in strictly shallower buckets; ``hi`` adds the sequence's
        own bucket. Exact for iid-step models; otherwise a mean-field estimate (floats, not certificates).
        """
        seq = tuple(sequence)
        fb_total = 0
        prefix: tuple = ()
        scale = self.quantizer.oversample / self.quantizer.bin_width_bits
        for token in seq:
            tokens, lps = self.model._steps_np(prefix)
            match = np.flatnonzero(tokens == token)
            if match.size == 0:
                raise ValueError("sequence leaves the model's support at %r" % (token,))
            lp = float(lps[int(match[0])])
            fb_total += int(math.floor(max(0.0, -lp / _LOG2) * scale + _TOL))
            prefix = prefix + (token,)
        self.ensure_bits((fb_total + 1) / self.quantizer.fine_per_bit())
        hist = self._suffix[0]
        lo = 0.0
        for j, c in enumerate(hist.data):
            if hist.base + j >= fb_total:
                break
            lo += c
        return lo, lo + max(float(hist.count_at(fb_total)) - 1.0, 0.0)


_ROOT = object()  # the lattice's synthetic root cluster (the one exactly-known context)


class LatticeEnvelopeIndex:
    """Cluster-conditioned envelope index: the Markov refinement between mean-field and exact.

    :class:`AREnvelopeIndex` averages one envelope per depth -- exact only for iid steps. This index
    conditions each depth's envelope on the **cluster of the last token** (``cluster_fn``): the
    calibration histograms are kept per ``(depth, cluster)`` and *split by the next token's cluster*, and
    the suffix tables become the lattice DP ``S_d[c] = sum_c' H_d[c][c'] (*) S_{d+1}[c']`` -- the same
    shape as :class:`~mixle.enumeration.hmm_paths.HMMPathIndex` with clusters as states. The result is
    **exact for any order-1 Markov model** whose next-token distribution depends only on
    ``(depth, cluster_fn(last token))`` -- with ``cluster_fn`` the identity, that is every last-token
    Markov model, where the mean-field envelope is provably lossy. Coarser clusterings interpolate:
    ``m = 1`` recovers mean-field, ``m = V`` conditions on the full last token.

    Queries mirror :class:`AREnvelopeIndex`: counts and thresholds off the root table, ``unrank`` descends
    the REAL model apportioning by cluster-conditioned subtree estimates (O(L) forwards), every returned
    log-probability exact. A ``(depth, cluster)`` never visited in calibration borrows the depth's
    aggregate envelope (mean-field fallback for that pocket) -- raise ``n_paths`` to shrink the pockets.
    Fixed-length models only.
    """

    def __init__(
        self,
        model: Any,
        *,
        n_clusters: int | None = None,
        cluster_fn: Any = None,
        n_paths: int = 128,
        seed: int = 0,
        budget_bits: float = 64.0,
    ) -> None:
        if getattr(model, "terminating", False):
            raise ValueError("LatticeEnvelopeIndex supports fixed-length models only (see AREnvelopeIndex).")
        if cluster_fn is None:
            if n_clusters is None:
                raise ValueError("give cluster_fn (token -> cluster id) or n_clusters (int tokens hashed mod m)")
            m = int(n_clusters)
            cluster_fn = lambda t: int(t) % m  # noqa: E731 - the documented default for integer tokens
        self.model = model
        self.length = int(model._depth)
        self.quantizer = model._quantizer()
        self.cluster_fn = cluster_fn
        self.n_paths = int(n_paths)
        self.seed = int(seed)
        self._budget_fb = max(1, int(math.ceil(float(budget_bits) * self.quantizer.fine_per_bit())))
        self._split: dict[tuple[int, Any], dict[Any, CountHistogram]] = {}  # (depth, cluster) -> c' -> hist
        self._agg: list[dict[Any, CountHistogram]] = []  # depth -> c' -> hist (fallback for unseen clusters)
        self._suffix: list[dict[Any, CountHistogram]] = []  # depth -> cluster -> completion histogram
        self._clusters: set[Any] = set()
        self._calibrate()
        self._rebuild_suffix()

    # -- precompute ----------------------------------------------------------------------------------------

    def _split_hist(self, prefix: tuple) -> dict[Any, CountHistogram]:
        """Next-token bucket histograms at ``prefix``, split by the next token's cluster."""
        tokens, lps = self.model._steps_np(prefix)
        scale = self.quantizer.oversample / self.quantizer.bin_width_bits
        sb = np.floor(np.maximum(0.0, -lps / _LOG2) * scale + _TOL).astype(np.int64)
        out: dict[Any, CountHistogram] = {}
        by_cluster: dict[Any, list[int]] = {}
        for j in range(tokens.size):
            by_cluster.setdefault(self.cluster_fn(tokens[j].item()), []).append(int(sb[j]))
        for c, buckets in by_cluster.items():
            arr = np.asarray(buckets, dtype=np.int64)
            base = int(arr.min())
            out[c] = CountHistogram(base, np.bincount(arr - base).astype(np.float64).tolist())
        return out

    @staticmethod
    def _merge_scaled(acc: dict[Any, CountHistogram], part: dict[Any, CountHistogram], w: float) -> None:
        for c, h in part.items():
            scaled = CountHistogram(h.base, [x * w for x in h.data]) if w != 1.0 else h
            acc[c] = scaled if c not in acc else acc[c].add(scaled)

    def _calibrate(self) -> None:
        rng = np.random.RandomState(self.seed)
        prefixes: list[tuple] = [() for _ in range(self.n_paths)]
        agg: list[dict[Any, CountHistogram]] = []
        for d in range(self.length):
            if d == 0:
                root = self._split_hist(())  # the one real root context: exact
                self._split[(0, _ROOT)] = root
                agg.append(root)
            else:
                # group sampled contexts by their last token's cluster; average split-hists per group
                groups: dict[Any, dict[tuple, int]] = {}
                for p in prefixes:
                    g = groups.setdefault(self.cluster_fn(p[-1]), {})
                    g[p] = g.get(p, 0) + 1
                # coverage completion: a cluster no sampled path ended in still needs an envelope (one
                # context suffices for the Markov-exactness contract) -- synthesize it by appending one
                # token of that cluster to a sampled parent context. <= m extra forwards per depth.
                parent = prefixes[0][: d - 1]
                ptokens, _plps = self.model._steps_np(parent)
                for tok in ptokens.tolist():
                    c = self.cluster_fn(tok)
                    if c not in groups:
                        groups[c] = {parent + (tok,): 1}
                        self._clusters.add(c)
                depth_agg: dict[Any, CountHistogram] = {}
                for c, members in groups.items():
                    acc: dict[Any, CountHistogram] = {}
                    total = float(sum(members.values()))
                    for p, mult in members.items():
                        self._merge_scaled(acc, self._split_hist(p), float(mult))
                    self._split[(d, c)] = {
                        k: CountHistogram(h.base, [x / total for x in h.data]) for k, h in acc.items()
                    }
                    self._merge_scaled(depth_agg, self._split[(d, c)], float(total))
                n_all = float(len(prefixes))
                agg.append({k: CountHistogram(h.base, [x / n_all for x in h.data]) for k, h in depth_agg.items()})
            if d == self.length - 1:
                break
            for j in range(self.n_paths):
                tokens, lps = self.model._steps_np(prefixes[j])
                p = np.exp(lps - lps.max())
                p /= p.sum()
                tok = tokens[int(rng.choice(tokens.size, p=p))].item()
                prefixes[j] = prefixes[j] + (tok,)
                self._clusters.add(self.cluster_fn(tok))
        self._agg = agg

    def _envelope_at(self, d: int, cluster: Any) -> dict[Any, CountHistogram]:
        got = self._split.get((d, cluster))
        return got if got is not None else self._agg[d]  # unseen pocket: depth aggregate (mean-field there)

    @staticmethod
    def _agg_tail(table: dict[Any, CountHistogram]) -> CountHistogram:
        """Average completion histogram over a level's clusters -- the tail for an unseen cluster."""
        pool = CountHistogram.empty()
        n = 0
        for h in table.values():
            pool = pool.add(h)
            n += 1
        return pool if n <= 1 else CountHistogram(pool.base, [x / n for x in pool.data])

    def _rebuild_suffix(self) -> None:
        clusters = sorted(self._clusters, key=repr)
        suffix: list[dict[Any, CountHistogram]] = [dict() for _ in range(self.length + 1)]
        suffix[self.length] = {c: CountHistogram.delta(0, 1) for c in [*clusters, _ROOT]}
        for d in range(self.length - 1, 0, -1):
            level: dict[Any, CountHistogram] = {}
            for c in clusters:
                acc = CountHistogram.empty()
                for c2, h in self._envelope_at(d, c).items():
                    tail = suffix[d + 1].get(c2)
                    if tail is None:  # a cluster never reached in calibration: average-tail fallback
                        tail = self._agg_tail(suffix[d + 1])
                    acc = acc.add(h.convolve_float(tail, max_fine_bucket=self._budget_fb))
                level[c] = acc
            suffix[d] = level
        root_acc = CountHistogram.empty()
        root_level = suffix[1] if self.length > 1 else suffix[self.length]
        for c2, h in self._split[(0, _ROOT)].items():
            tail = root_level.get(c2)
            if tail is None:
                tail = self._agg_tail(root_level) if root_level else CountHistogram.delta(0, 1)
            root_acc = root_acc.add(h.convolve_float(tail, max_fine_bucket=self._budget_fb))
        suffix[0] = {_ROOT: root_acc}
        self._suffix = suffix

    def ensure_bits(self, depth_bits: float) -> LatticeEnvelopeIndex:
        """Ensure suffix envelopes are deep enough for the requested bit depth."""
        needed = max(1, int(math.ceil(float(depth_bits) * self.quantizer.fine_per_bit())))
        if needed > self._budget_fb:
            self._budget_fb = max(needed, self._budget_fb * 2)
            self._rebuild_suffix()
        return self

    # -- queries (mirror AREnvelopeIndex, cluster-conditioned) -----------------------------------------------

    def _root(self) -> CountHistogram:
        return self._suffix[0][_ROOT]

    def total(self) -> float:
        """Estimated sequences within the built depth (exact for depth+last-cluster Markov models)."""
        return float(self._root().total())

    def count(self, min_log_prob: float) -> float:
        """Estimated number of sequences with ``log_density >= min_log_prob`` (Markov-refined estimate)."""
        self.ensure_bits(self.quantizer.bits(min_log_prob) + self.quantizer.bin_width_bits)
        fb = self.quantizer.fine_bucket(min_log_prob)
        hist = self._root()
        total = 0.0
        for j, c in enumerate(hist.data):
            if hist.base + j > fb:
                break
            total += c
        return float(total)

    def unrank(self, i: int) -> tuple[tuple, float]:
        """The approximately-``i``-th most probable sequence with its **exact** log-probability."""
        if i < 0:
            raise IndexError("rank must be >= 0")
        self.ensure_bits(math.log2(float(i) + 2.0) + 1.0)
        hist = self._root()
        target = float(i)
        bucket = None
        for j, c in enumerate(hist.data):
            if target < c:
                bucket = hist.base + j
                break
            target -= c
        if bucket is None:
            raise IndexError("rank %d beyond the estimated support (size %.6g)" % (i, self.total()))

        prefix: tuple = ()
        remaining = int(bucket)
        offset = target
        scale = self.quantizer.oversample / self.quantizer.bin_width_bits
        for d in range(self.length):
            tokens, lps = self.model._steps_np(prefix)
            sb = np.floor(np.maximum(0.0, -lps / _LOG2) * scale + _TOL).astype(np.int64)
            if d == self.length - 1:
                exact = np.flatnonzero(sb == remaining)
                if exact.size:
                    j = min(int(offset), exact.size - 1)
                    choice = int(exact[j])
                else:
                    choice = int(np.argmin(np.abs(sb - remaining)))
                prefix = prefix + (tokens[choice].item(),)
                break
            nxt = self._suffix[d + 1]
            chosen = None
            for t_idx in range(tokens.size):
                c2 = self.cluster_fn(tokens[t_idx].item())
                tail = nxt.get(c2)
                c = tail.count_at(remaining - int(sb[t_idx])) if tail is not None else 0.0
                if offset < c:
                    chosen = t_idx
                    break
                offset -= c
            if chosen is None:
                viable = [
                    t
                    for t in range(tokens.size)
                    if (nxt.get(self.cluster_fn(tokens[t].item())) or CountHistogram.empty()).count_at(
                        remaining - int(sb[t])
                    )
                    > 0
                ]
                chosen = viable[-1] if viable else 0
                offset = 0.0
            remaining -= int(sb[chosen])
            prefix = prefix + (tokens[chosen].item(),)
        return prefix, float(self.model.log_density(prefix))

    def threshold(self, rank: int) -> float:
        """Return the log-density threshold at a one-based rank."""
        if rank < 1:
            raise ValueError("rank must be >= 1")
        _seq, lp = self.unrank(rank - 1)
        return lp

    def rank_bracket(self, sequence: Any) -> tuple[float, float]:
        """Estimated ``[lo, hi]`` rank bracket -- exact for depth+last-cluster Markov models."""
        seq = tuple(sequence)
        fb_total = 0
        prefix: tuple = ()
        scale = self.quantizer.oversample / self.quantizer.bin_width_bits
        for token in seq:
            tokens, lps = self.model._steps_np(prefix)
            match = np.flatnonzero(tokens == token)
            if match.size == 0:
                raise ValueError("sequence leaves the model's support at %r" % (token,))
            lp = float(lps[int(match[0])])
            fb_total += int(math.floor(max(0.0, -lp / _LOG2) * scale + _TOL))
            prefix = prefix + (token,)
        self.ensure_bits((fb_total + 1) / self.quantizer.fine_per_bit())
        hist = self._root()
        lo = 0.0
        for j, c in enumerate(hist.data):
            if hist.base + j >= fb_total:
                break
            lo += c
        return lo, lo + max(float(hist.count_at(fb_total)) - 1.0, 0.0)
