"""Reasoning as a belief walk across a chain of verified transports.

A multi-hop reasoning path, such as ``binding -> structure -> activity``,
transports a belief at each hop. This module composes fitted conditional
transports from :func:`~mixle.reason.cycle_consistency.fit_cycle_transport` by
Monte Carlo forward simulation: draw from the belief at hop 0, push the sample
through hop 1's transport, and continue through the chain. The result is an
empirical posterior over the final variable whose spread reflects uncertainty
from every intervening hop.

Composition is gated on the edge premise: a transport that has not been
verified usable and calibrated on its own edge is refused before composition.
:func:`coverage_by_hop_count` checks calibration by hop count with a two-sided
binomial test against nominal coverage, so degradation across composed hops is
measured rather than assumed.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any

import numpy as np
from scipy.stats import binomtest


@dataclass
class HopTransport:
    """One edge of the belief walk and its own calibration verdict.

    ``premise_passed`` records whether this transport was independently
    verified usable and calibrated on this edge.
    """

    name: str
    fit: Any
    premise_passed: bool = True

    def sampler(self, seed: int | None = None) -> Any:
        """Return the sampler for this hop's fitted transport."""
        return self.fit.sampler(seed)


@dataclass
class WalkResult:
    """The belief walk's outcome: an empirical posterior over the final hop's variable."""

    hop_names: list[str]
    samples: np.ndarray  # (n_draws, dim)

    @property
    def mean(self) -> np.ndarray:
        """Return posterior sample mean for the final hop."""
        return self.samples.mean(axis=0)

    @property
    def std(self) -> np.ndarray:
        """Return posterior sample standard deviation for the final hop."""
        return self.samples.std(axis=0)

    def credible_interval(self, alpha: float = 0.1) -> tuple[np.ndarray, np.ndarray]:
        """Return marginal credible interval bounds from walk samples."""
        lo = np.quantile(self.samples, alpha / 2.0, axis=0)
        hi = np.quantile(self.samples, 1.0 - alpha / 2.0, axis=0)
        return lo, hi


def belief_walk(hops: Sequence[HopTransport], x0: Any, *, n_draws: int = 200, seed: int = 0) -> WalkResult:
    """Propagate a belief forward through a chain of hops, starting from a single value ``x0``.

    Each hop's transport is applied by drawing ``n_draws`` samples of the
    current belief and pushing each through the hop's ``sample_given`` method.
    Raises if any hop's ``premise_passed`` flag is ``False``.
    """
    unverified = [h.name for h in hops if not h.premise_passed]
    if unverified:
        raise ValueError(f"hop(s) {unverified} did not pass their F2 premise check; refusing to compose them")

    rng = np.random.RandomState(seed)
    x0_arr = np.atleast_1d(np.asarray(x0, dtype=np.float64))
    current = np.tile(x0_arr, (n_draws, 1))
    for hop in hops:
        sampler = hop.sampler(seed=int(rng.randint(0, 2**31 - 1)))
        current = np.asarray(sampler.sample_given_batch(current), dtype=np.float64)
    return WalkResult([h.name for h in hops], current)


def coverage_by_hop_count(
    hops: Sequence[HopTransport],
    x0_test: np.ndarray,
    true_final: dict[int, np.ndarray],
    *,
    alpha: float = 0.1,
    n_draws: int = 150,
    seed: int = 0,
) -> dict[int, dict[str, float]]:
    """Return empirical calibration by hop count.

    For ``k = 1 .. len(hops)``, walks the first ``k`` hops for every test point
    in ``x0_test`` and checks credible-interval coverage of ``true_final[k]``
    against the nominal ``1 - alpha`` rate with a two-sided binomial test.
    ``true_final`` must supply ground truth for each checked hop count.
    """
    out: dict[int, dict[str, float]] = {}
    for k in range(1, len(hops) + 1):
        truth_k = np.atleast_2d(np.asarray(true_final[k], dtype=np.float64))
        covered = 0
        for i in range(len(x0_test)):
            result = belief_walk(hops[:k], x0_test[i], n_draws=n_draws, seed=seed + i)
            lo, hi = result.credible_interval(alpha)
            covered += int(np.all((lo <= truth_k[i]) & (truth_k[i] <= hi)))
        rate = covered / len(x0_test)
        p = float(binomtest(covered, len(x0_test), 1.0 - alpha).pvalue)
        out[k] = {"coverage": rate, "p_value": p, "consistent_with_nominal": p >= 0.01}
    return out
