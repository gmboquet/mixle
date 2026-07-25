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

from mixle.reason.transport_edge import PremiseReceipt


@dataclass
class HopTransport:
    """One edge of the belief walk and its independently produced premise receipt.

    Explicit dimensions prevent a sampler from silently changing its batch
    contract while the receipt binds the edge to a verifier and held-out data.
    """

    name: str
    fit: Any
    receipt: PremiseReceipt
    input_dim: int
    output_dim: int

    def __post_init__(self) -> None:
        if not isinstance(self.name, str) or not self.name.strip():
            raise ValueError("hop name must be a non-empty string.")
        if not isinstance(self.receipt, PremiseReceipt):
            raise TypeError("hop receipt must be a PremiseReceipt produced by an edge verifier.")
        if self.receipt.edge_name != self.name:
            raise ValueError("hop name must match its premise receipt edge_name.")
        for field_name in ("input_dim", "output_dim"):
            value = getattr(self, field_name)
            if isinstance(value, bool) or not isinstance(value, (int, np.integer)) or int(value) <= 0:
                raise ValueError(f"{field_name} must be a positive integer.")
            setattr(self, field_name, int(value))
        if not hasattr(self.fit, "sampler") or not callable(self.fit.sampler):
            raise TypeError("hop fit must provide sampler(seed).")

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
        if not np.isfinite(alpha) or not 0.0 < alpha < 1.0:
            raise ValueError("alpha must be finite and strictly between zero and one.")
        lo = np.quantile(self.samples, alpha / 2.0, axis=0)
        hi = np.quantile(self.samples, 1.0 - alpha / 2.0, axis=0)
        return lo, hi

    def simultaneous_credible_interval(self, alpha: float = 0.1) -> tuple[np.ndarray, np.ndarray]:
        """Return Bonferroni intervals with declared joint target ``1 - alpha``."""
        if self.samples.ndim != 2 or self.samples.shape[1] <= 0:
            raise ValueError("walk samples must be a non-empty two-dimensional table.")
        return self.credible_interval(alpha / self.samples.shape[1])


def belief_walk(hops: Sequence[HopTransport], x0: Any, *, n_draws: int = 200, seed: int = 0) -> WalkResult:
    """Propagate a belief forward through a chain of hops, starting from a single value ``x0``.

    Each hop's transport is applied by drawing ``n_draws`` samples of the
    current belief and pushing each through the hop's ``sample_given`` method.
    Raises unless every hop carries an affirmative premise receipt.
    """
    if isinstance(n_draws, bool) or not isinstance(n_draws, (int, np.integer)) or int(n_draws) <= 0:
        raise ValueError("n_draws must be a positive integer.")
    if isinstance(seed, bool) or not isinstance(seed, (int, np.integer)):
        raise ValueError("seed must be an integer.")
    if not hops:
        raise ValueError("belief_walk requires at least one hop.")
    unverified = [h.name for h in hops if not h.receipt.passed]
    if unverified:
        raise ValueError(f"hop(s) {unverified} lack an affirmative premise receipt; refusing to compose them")

    rng = np.random.RandomState(int(seed))
    x0_arr = np.asarray(x0, dtype=np.float64)
    if x0_arr.ndim == 0:
        x0_arr = x0_arr.reshape(1)
    if x0_arr.ndim != 1 or x0_arr.shape != (hops[0].input_dim,) or not np.isfinite(x0_arr).all():
        raise ValueError(f"x0 must be a finite vector with shape {(hops[0].input_dim,)}.")
    current = np.tile(x0_arr, (int(n_draws), 1))
    for hop in hops:
        if current.shape[1] != hop.input_dim:
            raise ValueError(
                f"hop {hop.name!r} expects width {hop.input_dim}, but the preceding belief has width {current.shape[1]}."
            )
        sampler = hop.sampler(seed=int(rng.randint(0, 2**31 - 1)))
        current = np.asarray(sampler.sample_given_batch(current), dtype=np.float64)
        expected = (int(n_draws), hop.output_dim)
        if current.shape != expected or not np.isfinite(current).all():
            raise ValueError(f"hop {hop.name!r} must return finite samples with shape {expected}, got {current.shape}.")
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
    if not np.isfinite(alpha) or not 0.0 < alpha < 1.0:
        raise ValueError("alpha must be finite and strictly between zero and one.")
    if not hops:
        raise ValueError("coverage_by_hop_count requires at least one hop.")
    x0_table = np.asarray(x0_test, dtype=np.float64)
    if (
        x0_table.ndim != 2
        or not len(x0_table)
        or x0_table.shape[1] != hops[0].input_dim
        or not np.isfinite(x0_table).all()
    ):
        raise ValueError("x0_test must be a non-empty finite table aligned to the first hop input.")
    expected_keys = set(range(1, len(hops) + 1))
    if set(true_final) != expected_keys:
        raise ValueError(f"true_final must provide exactly hop counts {sorted(expected_keys)}.")
    truth_tables: dict[int, np.ndarray] = {}
    for k, hop in enumerate(hops, start=1):
        truth = np.asarray(true_final[k], dtype=np.float64)
        expected_shape = (len(x0_table), hop.output_dim)
        if truth.shape != expected_shape or not np.isfinite(truth).all():
            raise ValueError(f"true_final[{k}] must be a finite table with shape {expected_shape}.")
        truth_tables[k] = truth

    out: dict[int, dict[str, float]] = {}
    for k in range(1, len(hops) + 1):
        truth_k = truth_tables[k]
        covered = 0
        for i in range(len(x0_table)):
            result = belief_walk(hops[:k], x0_table[i], n_draws=n_draws, seed=seed + i)
            lo, hi = result.simultaneous_credible_interval(alpha)
            covered += int(np.all((lo <= truth_k[i]) & (truth_k[i] <= hi)))
        rate = covered / len(x0_table)
        target = 1.0 - alpha
        p = float(binomtest(covered, len(x0_table), target).pvalue)
        out[k] = {
            "coverage": rate,
            "joint_coverage_target": target,
            "p_value": p,
            "consistent_with_nominal": p >= 0.01,
        }
    return out
