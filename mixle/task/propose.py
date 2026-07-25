"""Budgeted propose-verify-retrain loop over a discrete structured design space.

This module handles structured and discrete candidates such as short protein-like sequences or compact
program sketches. It uses a proposal distribution over short fixed-length symbol sequences, with one
:class:`~mixle.stats.univariate.discrete.categorical.CategoricalDistribution` per position, then
samples K candidates per round, verifies every one against a
:class:`~mixle.doe.oracle.VerifiableOracle`, keeps the verifiably-better ones, and refits the proposal
on the winners (reweighted MLE through the shared :func:`mixle.inference.optimize` EM driver -- no
hand-rolled counting) for a fixed number of rounds under a hard oracle-call budget
(``k_per_round * rounds``). Every candidate tried is retained in the round log, dead ends included;
only ``keep_frac`` of each round feeds the refit.

Same "no verifiable objective, no optimization" precondition as ``optimize_under_oracle``:
``oracle=None`` refuses immediately rather than fabricating a candidate.

An abstained oracle call (``OracleResult.abstained`` -- e.g. a timeout; see
:class:`~mixle.doe.oracle.VerifiableOracle`) is never selected as a "winner" to refit on, and never
becomes ``best_result``: its placeholder ``-inf`` score is not a real observation of the oracle's
objective, only a record that the oracle did not answer. Every attempted candidate is still logged in
its round's :class:`RoundLog`, abstained or not, for the full receipted record -- only the *selection*
and *best-result* bookkeeping exclude abstentions, mirroring
:func:`mixle.doe.oracle.optimize_under_oracle`'s own "always record, selectively use" pattern.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np

from mixle.doe.oracle import OracleResult, VerifiableOracle
from mixle.inference import optimize
from mixle.stats.univariate.discrete.categorical import CategoricalDistribution


@dataclass
class SequenceProposal:
    """A position-independent categorical proposal over fixed-length sequences from ``alphabet``."""

    alphabet: tuple[Any, ...]
    length: int
    pseudo_count: float = 1.0
    position_models: list[CategoricalDistribution] = field(default_factory=list)

    def __post_init__(self) -> None:
        if self.length <= 0:
            raise ValueError("length must be positive.")
        if not self.position_models:
            uniform = {sym: 1.0 / len(self.alphabet) for sym in self.alphabet}
            self.position_models = [CategoricalDistribution(pmap=dict(uniform)) for _ in range(self.length)]
        elif len(self.position_models) != self.length:
            raise ValueError("position_models must have exactly one entry per sequence position.")

    def sample(self, k: int, rng: np.random.Generator) -> list[tuple]:
        """Draw ``k`` i.i.d. sequences (each ``length`` symbols) from the current proposal."""
        sequences = []
        for _ in range(k):
            symbols = [model.sampler(seed=int(rng.integers(0, 2**31 - 1))).sample() for model in self.position_models]
            sequences.append(tuple(symbols))
        return sequences

    def refit(self, sequences: list[tuple], weights: np.ndarray) -> SequenceProposal:
        """Reweighted MLE: refit each position's categorical on ``sequences``, replicated in that
        position's training multiset proportional to ``weights``, through the shared ``optimize`` EM
        driver -- never a hand-rolled frequency count."""
        w = np.asarray(weights, dtype=float)
        if w.shape != (len(sequences),):
            raise ValueError("weights must have one entry per sequence.")
        w = np.clip(w, 0.0, None)
        if not np.any(w > 0.0):
            raise ValueError("refit requires at least one strictly positive weight.")
        counts = np.maximum(np.round(w / w.max() * 100.0), 1).astype(int)

        new_models = []
        for i in range(self.length):
            data: list[Any] = []
            for seq, count in zip(sequences, counts):
                data.extend([seq[i]] * int(count))
            uniform = {sym: 1.0 / len(self.alphabet) for sym in self.alphabet}
            estimator = CategoricalDistribution(pmap=dict(uniform)).estimator(pseudo_count=self.pseudo_count)
            fitted = optimize(data, estimator, max_its=1, out=None)
            # Canonicalize pmap key order to `self.alphabet`: the accumulator's internal dict order
            # is not guaranteed stable across interpreter processes (Python's per-process string hash
            # randomization), which would otherwise change which sampled index maps to which symbol.
            new_models.append(CategoricalDistribution(pmap={sym: fitted.pmap[sym] for sym in self.alphabet}))
        return SequenceProposal(
            alphabet=self.alphabet, length=self.length, pseudo_count=self.pseudo_count, position_models=new_models
        )


@dataclass
class RoundLog:
    """One round's full record: every candidate tried and its oracle result, plus which were kept.

    ``candidates``/``results`` include every attempted candidate, abstained or not -- dead ends and
    abstentions alike are never dropped from the receipted record. ``kept_indices`` never includes an
    abstained candidate's index (see ``propose_verify_retrain``): an abstention is not a real observation
    of the oracle's objective, so it is excluded from the ranking that produces ``kept_indices`` even
    when there are too few genuine candidates in the round to fill the usual quota.
    """

    round_index: int
    candidates: list[tuple]
    results: list[OracleResult]
    kept_indices: list[int]


@dataclass
class ProposeVerifyResult:
    """The full receipted history of a propose-verify-retrain run."""

    proposal: SequenceProposal
    rounds: list[RoundLog] = field(default_factory=list)
    best_candidate: tuple | None = None
    best_result: OracleResult | None = None

    @property
    def oracle_calls(self) -> int:
        """Return the total number of candidate evaluations sent to the oracle."""
        return sum(len(r.candidates) for r in self.rounds)

    def all_candidates(self) -> list[tuple[tuple, OracleResult]]:
        """Every candidate tried across every round, in order -- dead ends included, none dropped."""
        out: list[tuple[tuple, OracleResult]] = []
        for r in self.rounds:
            out.extend(zip(r.candidates, r.results))
        return out


def propose_verify_retrain(
    proposal: SequenceProposal,
    oracle: VerifiableOracle | None,
    *,
    k_per_round: int,
    rounds: int,
    keep_frac: float = 0.25,
    seed: int | None = None,
) -> ProposeVerifyResult:
    """Sample, verify, keep, and refit under a fixed oracle-call budget.

    Each round draws ``k_per_round`` candidates from ``proposal``, verifies every one with ``oracle``,
    keeps the top ``keep_frac`` by oracle score, and refits ``proposal`` on the kept winners weighted
    by score. The exact oracle-call budget is ``k_per_round * rounds``. ``oracle=None`` raises
    immediately because this routine requires a verifiable objective rather than fabricating one.

    An abstained oracle result (``OracleResult.abstained``) never wins a "kept" slot and never becomes
    ``best_result``: both are ranked/compared only among that round's GENUINE candidates. A round in
    which every candidate abstains leaves ``proposal`` unchanged for that round rather than refitting on
    a fabricated all-abstained weight vector; every attempted candidate, abstained or not, still appears
    in that round's ``RoundLog``.
    """
    if oracle is None:
        raise ValueError(
            "no verifiable objective; cannot optimize. propose_verify_retrain requires a "
            "VerifiableOracle -- this is a hard precondition, not a missing default."
        )
    if not 0.0 < keep_frac <= 1.0:
        raise ValueError("keep_frac must be in (0, 1].")
    if k_per_round <= 0 or rounds <= 0:
        raise ValueError("k_per_round and rounds must be positive.")

    rng = np.random.default_rng(seed)
    result = ProposeVerifyResult(proposal=proposal)
    current = proposal
    for round_index in range(rounds):
        candidates = current.sample(k_per_round, rng)
        results = [oracle(c) for c in candidates]
        scores = np.asarray([r.score for r in results], dtype=float)

        # Rank only GENUINE (non-abstained) candidates for the top-n_keep selection: an abstained result
        # (OracleResult.abstained -- e.g. a timeout) is a placeholder -inf, not a real observation of the
        # oracle's objective, so it must never be picked as a "winner" to refit on, nor treated as
        # "verifiably terrible but real." Restricting the ranking pool to genuine indices up front (rather
        # than ranking every index and relying on -inf sorting last) means an abstained candidate is
        # excluded even when there are too few genuine candidates to fill n_keep, instead of being pulled
        # in just to pad out the count.
        n_keep = max(1, int(np.ceil(keep_frac * len(candidates))))
        genuine_indices = np.asarray([i for i, r in enumerate(results) if not r.abstained], dtype=int)
        if genuine_indices.size:
            order = np.argsort(-scores[genuine_indices])[:n_keep]
            kept_indices = [int(genuine_indices[i]) for i in order]
        else:
            kept_indices = []
        result.rounds.append(RoundLog(round_index, candidates, results, kept_indices))

        for candidate, oracle_result in zip(candidates, results):
            if oracle_result.abstained:
                continue  # a timeout's -inf placeholder must never win "best", genuine or not yet seen
            if result.best_result is None or oracle_result.score > result.best_result.score:
                result.best_candidate, result.best_result = candidate, oracle_result

        if kept_indices:
            winners = [candidates[i] for i in kept_indices]
            winner_scores = scores[kept_indices]
            current = current.refit(winners, winner_scores)
        # else: every candidate this round abstained -- nothing genuine to refit on, so the proposal is
        # left unchanged for this round rather than fed a fabricated all-abstained weight vector (mirrors
        # optimize_under_oracle skipping opt.tell() for an abstention rather than corrupting the fit).

    result.proposal = current
    return result
