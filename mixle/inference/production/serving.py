"""Production scoring with activity + computation logging and health/problem reporting.

A :class:`Service` wraps a fitted model (loaded directly or from a :class:`Registry` alias) and
scores production batches, recording every computation -- record count, wall time, mean log-likelihood,
how many records were *unscorable* (outside the model's support), and how many were *unavailable* (the
model could not be reached to ask) -- to an in-memory activity log (and optionally a JSONL file).
:meth:`health` summarizes recent activity so problems (rising unscorable rate, model unavailability,
falling log-likelihood, slow batches) are visible; with a reference sample set it can also flag drift.

Unscorable and unavailable are deliberately kept distinct. A record legitimately outside the model's
support scores as a non-finite log-density that the model itself computes and returns, no exception
involved. An exception raised while *asking* the model -- a network call failing, a backend timing out,
the model erroring outright -- means the model was never actually reached: an outage, not a scored
observation. :meth:`Service.score` isolates that failure mode with
:func:`mixle.system.fault.with_fallback` (mode ``"model_unavailable"``) so an outage is never silently
folded into an ordinary ``-inf`` indistinguishable from a genuine one -- see :mod:`mixle.system.fault`.

The batch call failing is itself a distinct, separately-tracked degradation (mode
``"batch_unavailable"``). When ``seq_log_density`` raises, every record in that call is retried
individually -- and a record can recover (score normally through the per-record path) even though the
batch path failed for it. Left untracked, that retry would be invisible on precisely the calls where it
matters most: a batch endpoint that fails on every call but whose per-record fallback always succeeds
would leave no trace in the activity log or :meth:`health` at all, since every record still ends up
scored and finite -- nothing looks unscorable or unavailable, even though every call is silently paying
for a full per-record retry. ``n_batch_unavailable`` / ``batch_unavailable_rate`` count records exposed
to that fallback regardless of whether the retry itself succeeded, so a degrading batch endpoint is
visible before it starts actually losing data -- which is what ``unavailable_rate`` covers.
"""

from __future__ import annotations

import json
import time
from typing import Any

import numpy as np

from mixle.system.fault import DegradedResult, with_fallback


class Service:
    """A deployed model that scores batches and logs each computation for monitoring."""

    def __init__(
        self,
        model: Any,
        *,
        name: str | None = None,
        reference: Any = None,
        log_path: str | None = None,
        keep: int = 1000,
    ) -> None:
        self.model = model
        self.name = name
        self.reference = list(reference) if reference is not None else None
        self.log_path = log_path
        self.keep = keep
        self.activity: list[dict] = []
        self.header = getattr(model, "header", None)

    @classmethod
    def from_registry(cls, registry: Any, name: str, *, alias: str = "production", **kw: Any) -> Service:
        """Load the model an alias points at in ``registry`` and serve it (carrying its provenance header)."""
        model, header = registry.current(name, alias)
        svc = cls(model, name=name, **kw)
        if header is not None and svc.header is None:  # the registry stores the header separately
            svc.header = header
        return svc

    def _log(self, event: dict) -> None:
        self.activity.append(event)
        if len(self.activity) > self.keep:
            self.activity = self.activity[-self.keep :]
        if self.log_path is not None:
            with open(self.log_path, "a") as f:
                f.write(json.dumps(event) + "\n")

    def score(self, records: Any) -> np.ndarray:
        """Return per-record log-densities and log the computation (timing, mean log-lik, unscorable
        count, unavailable count, batch-unavailable count).

        Records affected by a ``model_unavailable`` failure (see module docstring) still score ``-inf``
        in the returned array -- preserving its shape/dtype contract for callers -- but are counted
        separately as ``n_unavailable`` in the logged event and surfaced as ``unavailable_rate`` by
        :meth:`health`, rather than being indistinguishable from a genuinely-scored ``-inf``.

        The batch call itself failing is tracked separately again, as ``batch_unavailable``: every
        record in a batch whose ``seq_log_density`` call raised is retried individually, and is counted
        in ``n_batch_unavailable`` / ``batch_unavailable_rate`` regardless of whether that per-record
        retry recovers a real score or itself fails into ``n_unavailable`` -- so a batch endpoint that
        is failing over to the (much slower) per-record path on every call remains visible even when
        every individual retry happens to succeed and no record is actually lost.

        Raises:
            ValueError: if the model's batch output does not have exactly one log-density per input
                record -- a cardinality mismatch is a contract violation to reject, not a shape to
                silently accept. Raised only for a batch call that *succeeded* with the wrong shape;
                it is not itself treated as a reason to retry per-record.
        """
        recs = list(records)
        n_expected = len(recs)
        t0 = time.time()
        n_unavailable = 0

        def _batch_call() -> np.ndarray:
            enc = self.model.dist_to_encoder().seq_encode(recs)
            return np.asarray(self.model.seq_log_density(enc), dtype=float)

        def _batch_fallback(exc: Exception) -> np.ndarray:
            # the batch call raised -- not a cardinality bug (that requires a *successful* call, and is
            # validated below, outside this fallback) but the batch path being unreachable. Retry
            # per-record; each record's own outcome (recovered or itself unavailable) is tracked via
            # _safe_logd/n_unavailable, independently of the fact that this batch call is degraded.
            nonlocal n_unavailable
            lp = np.empty(n_expected, dtype=float)
            for i, r in enumerate(recs):
                outcome = self._safe_logd(r)
                lp[i] = outcome.value
                if outcome.degraded:
                    n_unavailable += 1
            return lp

        outcome = with_fallback(_batch_call, _batch_fallback, mode="batch_unavailable")
        lp = outcome.value
        if not outcome.degraded and lp.shape != (n_expected,):
            raise ValueError(
                f"model {self.name!r} returned log-densities with shape {lp.shape} for "
                f"{n_expected} input records; seq_log_density must return exactly one "
                f"log-density per record (expected shape ({n_expected},))"
            )

        dt = time.time() - t0
        finite = np.isfinite(lp)
        self._log(
            {
                "time": time.time(),
                "op": "score",
                "model": self.name,
                "n": len(recs),
                "duration_s": round(dt, 6),
                "mean_loglik": float(lp[finite].mean()) if finite.any() else None,
                "n_unscorable": int((~finite).sum()),
                "n_unavailable": n_unavailable,
                "n_batch_unavailable": n_expected if outcome.degraded else 0,
            }
        )
        return lp

    def _safe_logd(self, r: Any) -> DegradedResult:
        """Score one record, isolating any exception -- the model being unreachable or erroring, not the
        model judging the record impossible -- as an explicit ``model_unavailable`` degradation instead
        of an indistinguishable ``-inf``."""
        return with_fallback(
            lambda: float(self.model.log_density(r)), lambda exc: float("-inf"), mode="model_unavailable"
        )

    def check_drift(self, records: Any) -> Any:
        """Drift of ``records`` versus the service's reference sample (requires a ``reference``)."""
        if self.reference is None:
            raise ValueError("Service has no reference sample; pass reference= to enable drift checks")
        from mixle.inference.production.drift import detect_drift

        report = detect_drift(self.model, self.reference, list(records))
        self._log({"time": time.time(), "op": "drift", "model": self.name, "drift": report.drift})
        return report

    def health(self, *, window: int = 100) -> dict:
        """Summary of the most recent ``window`` scoring events -- throughput, mean log-likelihood, the
        unscorable rate (the production problem signal), the unavailable rate (the subset of unscorable
        records caused by the model being unreachable -- an outage -- rather than legitimately scored),
        and the batch-unavailable rate (records whose batch call failed and had to be retried
        per-record, whether or not that retry itself recovered a score -- always ``>=`` the unavailable
        rate; see :meth:`score`)."""
        drift_events = sum(1 for e in self.activity if e["op"] == "drift" and e.get("drift"))
        scores = [e for e in self.activity if e["op"] == "score"][-window:]
        if not scores:
            return {"events": 0, "drift_events": drift_events}
        n = sum(e["n"] for e in scores)
        unscor = sum(e["n_unscorable"] for e in scores)
        unavail = sum(e.get("n_unavailable", 0) for e in scores)
        batch_unavail = sum(e.get("n_batch_unavailable", 0) for e in scores)
        lls = [e["mean_loglik"] for e in scores if e["mean_loglik"] is not None]
        return {
            "events": len(scores),
            "records": n,
            "records_per_s": round(n / max(sum(e["duration_s"] for e in scores), 1e-9), 1),
            "mean_loglik": float(np.mean(lls)) if lls else None,
            "unscorable_rate": round(unscor / n, 6) if n else 0.0,
            "unavailable_rate": round(unavail / n, 6) if n else 0.0,
            "batch_unavailable_rate": round(batch_unavail / n, 6) if n else 0.0,
            "drift_events": drift_events,
        }
