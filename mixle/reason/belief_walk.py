"""Reasoning as a belief walk across a chain of verified transports (workstream F3).

A multi-hop reasoning path (e.g. binding -> structure -> activity) transports a BELIEF at each hop,
with uncertainty compounding honestly rather than assumed. This module composes a chain of fitted
conditional transports (:func:`~mixle.reason.cycle_consistency.fit_cycle_transport`, the same MDN
family CARD TRANSPORT-a's F0 gate proved usable and calibrated) by Monte Carlo forward simulation:
draw a sample of the belief at hop 0, push it through hop 1's transport to get a sample at hop 1, and
so on. The result is an empirical posterior over the final hop's variable whose spread reflects every
intervening hop's genuine uncertainty -- not a point estimate chained through point estimates.

Per the plan, composition is gated on the F2 premise: a transport that has not itself been verified
usable/calibrated on its own edge must not be composed into a walk -- an unverified edge silently
corrupts every downstream calibration claim built on top of it. :func:`calibration_by_hop_count`
checks calibration AS A FUNCTION OF HOP COUNT (a two-sided binomial test against nominal coverage,
mirroring :mod:`mixle.task.solve`'s own use of the same test) rather than assuming it holds -- if
composed calibration degrades faster than the per-hop errors alone would predict, that degradation
curve is the thing to report, not a blind "uncertainty compounds honestly" claim.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any

import numpy as np
from scipy.stats import binomtest


@dataclass
class HopTransport:
    """One edge of the belief walk: a fitted conditional transport plus its own F2 premise verdict.

    ``premise_passed`` records whether THIS transport, on THIS edge, was independently verified usable
    and calibrated (CARD F2-a) -- never assumed true because F0 passed on an unrelated toy problem.
    """

    name: str
    fit: Any
    premise_passed: bool = True

    def sampler(self, seed: int | None = None) -> Any:
        return self.fit.sampler(seed)


@dataclass
class WalkResult:
    """The belief walk's outcome: an empirical posterior over the final hop's variable."""

    hop_names: list[str]
    samples: np.ndarray  # (n_draws, dim)

    @property
    def mean(self) -> np.ndarray:
        return self.samples.mean(axis=0)

    @property
    def std(self) -> np.ndarray:
        return self.samples.std(axis=0)

    def credible_interval(self, alpha: float = 0.1) -> tuple[np.ndarray, np.ndarray]:
        lo = np.quantile(self.samples, alpha / 2.0, axis=0)
        hi = np.quantile(self.samples, 1.0 - alpha / 2.0, axis=0)
        return lo, hi


def belief_walk(hops: Sequence[HopTransport], x0: Any, *, n_draws: int = 200, seed: int = 0) -> WalkResult:
    """Propagate a belief forward through a chain of hops, starting from a single value ``x0``.

    Each hop's transport is applied by drawing ``n_draws`` independent samples of the CURRENT belief
    and pushing each through the hop's ``sample_given`` -- the empirical spread at the end is the
    walk's honestly-compounded uncertainty. Raises if any hop's ``premise_passed`` is ``False``:
    composing an unverified edge is refused rather than silently producing an uncheckable posterior.
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
    """Calibration AS A FUNCTION OF HOP COUNT: for ``k = 1 .. len(hops)``, walk the first ``k`` hops
    for every test point in ``x0_test`` and check the empirical credible-interval coverage of
    ``true_final[k]`` (the true value at hop ``k`` for the SAME test points) against the nominal
    ``1 - alpha`` rate, via a two-sided binomial test.

    Returns ``{k: {"coverage": observed_rate, "p_value": ..., "consistent_with_nominal": bool}}`` --
    the degradation curve the card requires, measured, never assumed. ``true_final`` must supply the
    ground truth at every hop count checked (e.g. from a known generative process on held-out data).
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
