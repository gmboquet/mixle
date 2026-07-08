"""Runtime observability for distributed EM backends (MP/MPI/Spark).

Companion to :mod:`mixle.utils.parallel.balance`: ``balance`` produces a *static* worker grid
from a FLOPs-under-memory model before the fit starts; this module captures what *actually*
happened per rank per round once the fit is running, so stragglers and data-volume skew that
the static plan could not see (heterogeneous hardware, cold caches, a lopsided shard) are
caught within a single round instead of silently eating wall-clock for the whole fit.

The unlock is the same one that makes shard recomputation a corruption audit and accumulators
checkpoints: EM's sufficient statistics are additive, so per-rank *records* are additive too --
a round's ``RankRecord`` list is a complete, replayable receipt of what each worker did, and
folding/analyzing them needs no coordination beyond what already flows back to the driver.

Pieces:
    * :class:`RankRecord` -- one worker-rank's timing/bytes/accumulator-size receipt for one round.
    * :func:`record_rank_round` -- append a :class:`RankRecord` as a :mod:`mixle.telemetry` event.
    * :func:`detect_stragglers` -- robust (median/MAD) outlier test over one round's rank times.
    * :func:`imbalance_receipt` -- quantitative data-volume skew across ranks for one round.
    * :func:`plan_rebalance_weights` -- turn observed timings into a feedback signal for
      :func:`mixle.utils.parallel.balance.balance_plan`'s next static plan.
    * :func:`fit_report` -- one human/machine-readable summary of a whole fit's collected records.
"""

from __future__ import annotations

import statistics
from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import Any

from mixle.telemetry.core import Event, Telemetry

__all__ = [
    "FitReport",
    "ImbalanceReceipt",
    "RankRecord",
    "StragglerReport",
    "detect_stragglers",
    "fit_report",
    "imbalance_receipt",
    "plan_rebalance_weights",
    "record_rank_round",
    "records_from_events",
]


@dataclass(frozen=True)
class RankRecord:
    """One worker-rank's structured receipt for one EM round.

    ``e_step_seconds``/``m_step_seconds`` are wall-clock durations measured on the rank itself
    (E-step = the accumulate/score pass over the shard, M-step = folding+``estimate`` when a rank
    does its own partial M-step, ``0.0`` when the M-step is driver-side only). ``bytes_processed``
    is the raw/encoded data volume the rank touched this round; ``accumulator_bytes`` is the size
    of the sufficient-statistics payload it shipped back -- both additive across ranks, so a
    round's totals are a plain sum.
    """

    rank: int
    round: int
    e_step_seconds: float
    m_step_seconds: float = 0.0
    bytes_processed: int = 0
    accumulator_bytes: int = 0
    n_obs: int = 0
    extra: dict[str, Any] = field(default_factory=dict)

    @property
    def total_seconds(self) -> float:
        return self.e_step_seconds + self.m_step_seconds

    def as_features(self) -> dict[str, Any]:
        """Flat dict form used as a :class:`~mixle.telemetry.core.Event` ``features`` payload."""
        return {
            "rank": self.rank,
            "round": self.round,
            "e_step_seconds": self.e_step_seconds,
            "m_step_seconds": self.m_step_seconds,
            "total_seconds": self.total_seconds,
            "bytes_processed": self.bytes_processed,
            "accumulator_bytes": self.accumulator_bytes,
            "n_obs": self.n_obs,
            **self.extra,
        }

    @classmethod
    def from_features(cls, features: dict[str, Any]) -> RankRecord:
        """Inverse of :meth:`as_features` (round-trips through a :class:`~mixle.telemetry.core.Event`)."""
        known = {"rank", "round", "e_step_seconds", "m_step_seconds", "bytes_processed", "accumulator_bytes", "n_obs"}
        extra = {k: v for k, v in features.items() if k not in known and k != "total_seconds"}
        return cls(
            rank=int(features["rank"]),
            round=int(features["round"]),
            e_step_seconds=float(features.get("e_step_seconds", 0.0)),
            m_step_seconds=float(features.get("m_step_seconds", 0.0)),
            bytes_processed=int(features.get("bytes_processed", 0)),
            accumulator_bytes=int(features.get("accumulator_bytes", 0)),
            n_obs=int(features.get("n_obs", 0)),
            extra=extra,
        )


def record_rank_round(
    telemetry: Telemetry,
    *,
    rank: int,
    round: int,
    e_step_seconds: float,
    m_step_seconds: float = 0.0,
    bytes_processed: int = 0,
    accumulator_bytes: int = 0,
    n_obs: int = 0,
    run_id: str | None = None,
    extra: dict[str, Any] | None = None,
) -> Event:
    """Append one rank's per-round receipt to ``telemetry`` as an ``"em_round"`` event.

    Thin wrapper around :meth:`Telemetry.record` -- reuses the existing typed-event/JSONL
    recorder rather than inventing a parallel logging object. ``run_id`` (if given) tags the
    event so records from concurrent fits sharing one recorder can be told apart.
    """
    rec = RankRecord(
        rank=rank,
        round=round,
        e_step_seconds=e_step_seconds,
        m_step_seconds=m_step_seconds,
        bytes_processed=bytes_processed,
        accumulator_bytes=accumulator_bytes,
        n_obs=n_obs,
        extra=dict(extra or {}),
    )
    tags = {"run_id": run_id} if run_id is not None else {}
    return telemetry.record("em_round", features=rec.as_features(), choice=f"rank={rank}", tags=tags)


def records_from_events(events: Sequence[Event]) -> list[RankRecord]:
    """Recover :class:`RankRecord` objects from ``"em_round"`` :class:`~mixle.telemetry.core.Event` rows."""
    return [RankRecord.from_features(ev.features) for ev in events if ev.kind == "em_round"]


def _records_for_round(records: Sequence[RankRecord], round: int | None) -> tuple[int, list[RankRecord]]:
    if round is None:
        round = max((r.round for r in records), default=0)
    return round, [r for r in records if r.round == round]


@dataclass(frozen=True)
class StragglerReport:
    """Straggler/imbalance verdict for one EM round, from a robust median/MAD outlier test."""

    round: int
    rank_seconds: dict[int, float]  # total_seconds per rank this round
    median_seconds: float
    mad_seconds: float  # median absolute deviation (unscaled)
    ratios: dict[int, float]  # rank_time / median_time
    z_scores: dict[int, float]  # robust z-score: (t - median) / (1.4826 * MAD)
    threshold_ratio: float
    z_threshold: float
    slow_ranks: tuple[int, ...]

    @property
    def has_stragglers(self) -> bool:
        return len(self.slow_ranks) > 0


def detect_stragglers(
    records: Sequence[RankRecord],
    *,
    round: int | None = None,
    threshold_ratio: float = 1.5,
    z_threshold: float = 3.0,
) -> StragglerReport:
    """Flag ranks that are meaningfully slower than the rest for one round.

    A robust (median / median-absolute-deviation) outlier test rather than a mean/stddev test,
    since a single straggler should not be allowed to inflate the scale it is measured against.
    A rank is flagged when it is BOTH ``threshold_ratio``x slower than the median rank time AND
    (when the round has enough spread to have a nonzero MAD) ``z_threshold`` robust-z above the
    median -- the ratio catches the practically-significant case, the z-score guards against
    flagging normal jitter when every rank is close together.
    """
    r, round_records = _records_for_round(records, round)
    if not round_records:
        return StragglerReport(
            round=r,
            rank_seconds={},
            median_seconds=0.0,
            mad_seconds=0.0,
            ratios={},
            z_scores={},
            threshold_ratio=threshold_ratio,
            z_threshold=z_threshold,
            slow_ranks=(),
        )

    times = {rec.rank: rec.total_seconds for rec in round_records}
    values = list(times.values())
    median = statistics.median(values)
    mad = statistics.median(abs(v - median) for v in values)
    # 1.4826 makes MAD a consistent estimator of the standard deviation under normality.
    scale = 1.4826 * mad

    ratios = {rank: (t / median if median > 0 else (0.0 if t == 0 else float("inf"))) for rank, t in times.items()}
    z_scores = {}
    for rank, t in times.items():
        if scale > 0:
            z_scores[rank] = (t - median) / scale
        else:
            z_scores[rank] = float("inf") if t > median else 0.0

    slow = tuple(
        sorted(
            rank
            for rank in times
            if ratios[rank] >= threshold_ratio and (scale == 0.0 or z_scores[rank] >= z_threshold)
        )
    )

    return StragglerReport(
        round=r,
        rank_seconds=times,
        median_seconds=median,
        mad_seconds=mad,
        ratios=ratios,
        z_scores=z_scores,
        threshold_ratio=threshold_ratio,
        z_threshold=z_threshold,
        slow_ranks=slow,
    )


@dataclass(frozen=True)
class ImbalanceReceipt:
    """Quantitative data-volume skew across ranks for one round (bytes and observation counts)."""

    round: int
    bytes_per_rank: dict[int, int]
    n_obs_per_rank: dict[int, int]
    mean_bytes: float
    max_bytes_ratio: float  # max(bytes) / mean(bytes) -- >1 means at least one rank is above-average loaded
    skew_by_rank: dict[int, float]  # bytes[rank] / mean(bytes) per rank -- matches a "rank got Nx the data" claim


def imbalance_receipt(records: Sequence[RankRecord], *, round: int | None = None) -> ImbalanceReceipt:
    """Measure how unevenly data volume was actually spread across ranks for one round.

    ``skew_by_rank[rank]`` is that rank's ``bytes_processed`` divided by the mean across ranks --
    a planted "rank 0 gets 10x the data" skew shows up directly as ``skew_by_rank[0] ~= 10 * (P /
    (P - 1 + 10))`` scaling (exactly proportional to the planted ratio for the standard case of
    one heavy rank among otherwise-equal peers; see the test for the exact algebra), so this is a
    real correctness check against the planted skew, not just a boolean "imbalanced" flag.
    """
    r, round_records = _records_for_round(records, round)
    bytes_per_rank = {rec.rank: rec.bytes_processed for rec in round_records}
    n_obs_per_rank = {rec.rank: rec.n_obs for rec in round_records}
    values = list(bytes_per_rank.values())
    mean_bytes = statistics.fmean(values) if values else 0.0
    max_bytes = max(values) if values else 0.0
    max_ratio = (max_bytes / mean_bytes) if mean_bytes > 0 else 0.0
    skew = {rank: (b / mean_bytes if mean_bytes > 0 else 0.0) for rank, b in bytes_per_rank.items()}

    return ImbalanceReceipt(
        round=r,
        bytes_per_rank=bytes_per_rank,
        n_obs_per_rank=n_obs_per_rank,
        mean_bytes=mean_bytes,
        max_bytes_ratio=max_ratio,
        skew_by_rank=skew,
    )


def plan_rebalance_weights(records: Sequence[RankRecord], *, round: int | None = None) -> dict[int, float]:
    """Turn one round's observed rank timings into a feedback signal for the next static plan.

    ``balance_plan`` (:mod:`mixle.utils.parallel.balance`) picks a static ``(D, M)`` worker grid
    and, when ``M > 1``, splits the model's units across shards with :func:`_balance_units`
    weighted by a *predicted* per-unit FLOP cost (``best.unit_works``). This function produces the
    *observed* analogue: a per-rank weight inversely proportional to how long that rank actually
    took, normalized to sum to the rank count -- a rank that ran 2x slower than average gets ~0.5x
    the weight, meaning "give this rank half as much data/model next round". The weights are in
    exactly the shape ``_balance_units`` consumes (a sequence of relative per-unit works, here one
    "unit" per rank), so a re-planning loop can plug them in directly; wiring that end-to-end
    (re-running ``balance_plan`` with per-worker throughput overrides derived from these weights)
    is future work -- this function only produces the signal, documented here as the integration
    point.
    """
    r, round_records = _records_for_round(records, round)
    times = {rec.rank: rec.total_seconds for rec in round_records}
    if not times:
        return {}
    positive = {rank: t for rank, t in times.items() if t > 0}
    if not positive:
        # every rank reported zero time (e.g. a degenerate/empty round): weight uniformly.
        n = len(times)
        return dict.fromkeys(times, 1.0 / n if n else 0.0)
    inv = {rank: (1.0 / t if t > 0 else 0.0) for rank, t in times.items()}
    total_inv = sum(inv.values())
    n = len(times)
    return {rank: (w / total_inv) * n if total_inv > 0 else 0.0 for rank, w in inv.items()}


@dataclass(frozen=True)
class FitReport:
    """A whole-fit summary rolled up from per-rank, per-round records -- the EM-side ``run.report()``.

    Mirrors the shape F4's training-health ``run.report()`` is expected to have (per-round health,
    an overall verdict, a printable render) without depending on F4, which had not landed when
    this was written; aligning field names/shape once F4 exists is a natural, tracked follow-up.
    """

    n_rounds: int
    n_ranks: int
    total_seconds: float
    total_bytes: int
    total_obs: int
    rounds: tuple[int, ...]
    stragglers_by_round: dict[int, StragglerReport]
    imbalance_by_round: dict[int, ImbalanceReceipt]
    worst_straggler_ratio: float
    worst_imbalance_ratio: float

    @property
    def any_stragglers(self) -> bool:
        return any(s.has_stragglers for s in self.stragglers_by_round.values())

    def render(self) -> str:
        """A short human-readable summary, e.g. for a CLI/log line."""
        lines = [
            f"fit_report: {self.n_ranks} ranks x {self.n_rounds} rounds, "
            f"{self.total_seconds:.3f}s total, {self.total_bytes} bytes, {self.total_obs} obs",
        ]
        for r in self.rounds:
            s = self.stragglers_by_round[r]
            im = self.imbalance_by_round[r]
            flag = f"STRAGGLERS={s.slow_ranks}" if s.has_stragglers else "ok"
            lines.append(
                f"  round {r}: median={s.median_seconds:.4f}s max_ratio={max(s.ratios.values(), default=0.0):.2f}x "
                f"imbalance_ratio={im.max_bytes_ratio:.2f}x [{flag}]"
            )
        return "\n".join(lines)

    def as_dict(self) -> dict[str, Any]:
        """Machine-readable form (JSON-friendly)."""
        return {
            "n_rounds": self.n_rounds,
            "n_ranks": self.n_ranks,
            "total_seconds": self.total_seconds,
            "total_bytes": self.total_bytes,
            "total_obs": self.total_obs,
            "rounds": list(self.rounds),
            "worst_straggler_ratio": self.worst_straggler_ratio,
            "worst_imbalance_ratio": self.worst_imbalance_ratio,
            "any_stragglers": self.any_stragglers,
            "by_round": {
                r: {
                    "slow_ranks": list(self.stragglers_by_round[r].slow_ranks),
                    "rank_seconds": self.stragglers_by_round[r].rank_seconds,
                    "imbalance_max_ratio": self.imbalance_by_round[r].max_bytes_ratio,
                    "skew_by_rank": self.imbalance_by_round[r].skew_by_rank,
                }
                for r in self.rounds
            },
        }


def fit_report(
    records: Sequence[RankRecord],
    *,
    threshold_ratio: float = 1.5,
    z_threshold: float = 3.0,
) -> FitReport:
    """Build one summary report from a completed (or in-progress) distributed EM fit's records.

    Groups ``records`` by round, runs :func:`detect_stragglers` and :func:`imbalance_receipt` per
    round, and rolls the results into one :class:`FitReport`. Safe to call mid-fit with only the
    rounds collected so far -- an empty ``records`` sequence yields a zeroed report rather than
    raising.
    """
    rounds = sorted({rec.round for rec in records})
    ranks = sorted({rec.rank for rec in records})

    stragglers_by_round: dict[int, StragglerReport] = {}
    imbalance_by_round: dict[int, ImbalanceReceipt] = {}
    worst_straggler_ratio = 0.0
    worst_imbalance_ratio = 0.0

    for r in rounds:
        s = detect_stragglers(records, round=r, threshold_ratio=threshold_ratio, z_threshold=z_threshold)
        im = imbalance_receipt(records, round=r)
        stragglers_by_round[r] = s
        imbalance_by_round[r] = im
        if s.ratios:
            worst_straggler_ratio = max(worst_straggler_ratio, max(s.ratios.values()))
        worst_imbalance_ratio = max(worst_imbalance_ratio, im.max_bytes_ratio)

    return FitReport(
        n_rounds=len(rounds),
        n_ranks=len(ranks),
        total_seconds=sum(rec.total_seconds for rec in records),
        total_bytes=sum(rec.bytes_processed for rec in records),
        total_obs=sum(rec.n_obs for rec in records),
        rounds=tuple(rounds),
        stragglers_by_round=stragglers_by_round,
        imbalance_by_round=imbalance_by_round,
        worst_straggler_ratio=worst_straggler_ratio,
        worst_imbalance_ratio=worst_imbalance_ratio,
    )
