"""``mixle.Model`` -- the model lifecycle as one object with consistent verbs.

Everything here exists elsewhere in the library; this facade makes the lifecycle *discoverable* without
knowing which subpackage owns which verb::

    m = mixle.propose(data)          # a model shape recommended from the data (with confidence + caveats)
    m.fit(data)                      # inference chosen from the structure (EM / MLE / closed form)
    m.evaluate(holdout)              # held-out scores
    m.sample(5)                      # draw new records
    m.enumerate().top_k(3)           # most-probable support (discrete/structured families)
    m.posterior(x)                   # latent posteriors (mixtures, HMMs, ...)
    m.distill(teacher, inputs)       # tiny deployable student in front of the teacher (task spine)
    m.deploy("artifacts/m")          # durable artifact directory; Model.load() restores it
    m.explain()                      # what it is, what it supports, and how it was proposed
    m(x)                             # use it: log-density of an observation

``Model`` wraps a prototype distribution, an estimator, or nothing (the estimator is inferred from the
data); verbs delegate to :func:`mixle.inference.optimize`, ``dist.enumerator()``, ``mixle.task.solve``,
and :func:`mixle.describe`. It adds no new inference -- only one place to stand.
"""

from __future__ import annotations

import json
import pickle
import time
from pathlib import Path
from typing import Any

import numpy as np


class Model:
    """One object over the model lifecycle: build / fit / evaluate / enumerate / distill / deploy / use."""

    def __init__(self, spec: Any = None, *, notes: list[str] | None = None) -> None:
        """``spec`` is a prototype distribution, an estimator, or ``None`` (infer from data at fit time)."""
        self.spec = spec
        self.fitted: Any = None
        self.notes: list[str] = list(notes or [])
        self.frontier: list[dict[str, Any]] | None = None  # candidate ranking when built by propose()
        self._fit_info: dict[str, Any] = {}

    # --- fit / use -------------------------------------------------------------------------------
    def fit(self, data: Any, **optimize_kw: Any) -> Model:
        """Fit via :func:`mixle.inference.optimize`; the algorithm follows from the model's structure."""
        from mixle.inference import optimize

        optimize_kw.setdefault("out", None)
        self.fitted = optimize(data, self.spec, **optimize_kw)
        self._fit_info = {"n": len(data) if hasattr(data, "__len__") else None, "when": time.time()}
        return self

    def _require_fitted(self) -> Any:
        if self.fitted is None:
            raise RuntimeError("fit(data) first -- this Model has no fitted distribution yet")
        return self.fitted

    def __call__(self, x: Any) -> float:
        """The model as a scorer: ``log p(x)`` of one observation under the fitted distribution."""
        return float(self._require_fitted().log_density(x))

    def evaluate(self, data: Any) -> dict[str, Any]:
        """Held-out fit quality: total and mean log-density over ``data``."""
        d = self._require_fitted()
        enc = d.dist_to_encoder().seq_encode(list(data))
        ll = np.asarray(d.seq_log_density(enc), dtype=np.float64)
        return {"n": int(ll.size), "mean_log_density": float(ll.mean()), "total_log_density": float(ll.sum())}

    def sample(self, size: int | None = None, *, seed: int | None = None) -> Any:
        return self._require_fitted().sampler(seed=seed).sample(size)

    # --- structure verbs -------------------------------------------------------------------------
    def enumerate(self) -> Any:
        """The fitted distribution's enumerator (top-k / top-p / rank / seek), where supported."""
        return self._require_fitted().enumerator()

    def posterior(self, x: Any) -> Any:
        """Latent posterior for one observation (mixtures, HMMs, ...), where supported."""
        return self._require_fitted().posterior(x)

    # --- distill / deploy ------------------------------------------------------------------------
    def distill(self, teacher: Any = None, inputs: Any = None, **solve_kw: Any):
        """Distill a tiny deployable student via :func:`mixle.task.solve`.

        With ``teacher=None`` the *fitted model itself* teaches: inputs are labeled by their most-probable
        latent component (``posterior`` argmax), so a fitted mixture becomes a fast, calibrated classifier
        of its own clusters. Returns a :class:`mixle.task.Solution` (call it, ``report()``, ``improve()``).
        """
        from mixle.task import solve

        if inputs is None:
            raise ValueError("distill needs the example inputs to label and train on")
        if teacher is None:
            fitted = self._require_fitted()

            def teacher(x: Any) -> str:  # label = most probable latent component under this model
                return str(int(np.argmax(np.asarray(fitted.posterior(x)))))

        return solve(teacher, inputs, **solve_kw)

    def deploy(self, path: str) -> str:
        """Persist a durable artifact directory (model + manifest); :meth:`Model.load` restores it."""
        d = self._require_fitted()
        out = Path(path)
        out.mkdir(parents=True, exist_ok=True)
        with open(out / "model.pkl", "wb") as f:
            pickle.dump(d, f)
        manifest = {
            "family": type(d).__name__,
            "created_at": time.time(),
            "fit": self._fit_info,
            "notes": self.notes,
            "mixle_artifact": "lifecycle.Model/v1",
        }
        (out / "manifest.json").write_text(json.dumps(manifest, indent=2, default=str))
        return str(out)

    @classmethod
    def load(cls, path: str) -> Model:
        p = Path(path)
        with open(p / "model.pkl", "rb") as f:
            fitted = pickle.load(f)
        m = cls(fitted)
        m.fitted = fitted
        try:
            m.notes = list(json.loads((p / "manifest.json").read_text()).get("notes", []))
        except (OSError, ValueError):
            pass
        return m

    # --- introspection ---------------------------------------------------------------------------
    def explain(self) -> str:
        """What this model is, what it supports, and how it was proposed."""
        from mixle.capability import describe

        target = self.fitted if self.fitted is not None else self.spec
        head = "unfitted" if self.fitted is None else "fitted"
        body = describe(target) if target is not None else "(no spec: the estimator is inferred at fit time)"
        notes = ("\nproposal notes:\n  - " + "\n  - ".join(self.notes)) if self.notes else ""
        return f"Model ({head})\n{body}{notes}"

    def __repr__(self) -> str:
        inner = type(self.fitted or self.spec).__name__ if (self.fitted or self.spec) is not None else "auto"
        return f"Model({inner}, fitted={self.fitted is not None})"


def propose(
    data: Any,
    *,
    fit: bool = False,
    llm: Any = None,
    holdout: float = 0.25,
    seed: int = 0,
    max_its: int = 25,
    **recommend_kw: Any,
) -> Model:
    """Propose a model for ``data`` from a *verified frontier* of candidates and return the winner.

    Candidates come from every proposer the library has — the heuristic recommendation
    (:func:`mixle.task.recommend.recommend_model`, dependency-aware), the plain independence baseline
    (:func:`mixle.utils.automatic.get_estimator`), and, when an ``llm`` handle is given, an LLM-designed
    structure (:func:`mixle.task.design.design_model`, allowlisted-spec, fit-validated). Each candidate is
    **fitted on a train split and scored on held-out data**, so the ranking is out-of-sample, not a guess.
    The winner becomes the returned :class:`Model`; the full ranking lands in ``Model.frontier`` and the
    per-field confidence / dependency / candidate notes in ``Model.notes`` (shown by ``explain()``).
    Pass ``fit=True`` to also fit the winner to all of ``data`` before returning.
    """
    from mixle.inference import optimize
    from mixle.task import recommend_model

    rows = list(data)
    rec = recommend_model(rows, **recommend_kw)
    candidates: list[tuple[str, Any]] = [("recommended", rec.estimator)]
    try:  # the independence baseline the frontier has to beat (skip when identical to the recommendation)
        from mixle.utils.automatic import get_estimator

        indep = get_estimator(rows)
        if repr(indep) != repr(rec.estimator):
            candidates.append(("independent", indep))
    except Exception:  # noqa: BLE001 - a baseline that can't build is just absent from the frontier
        pass
    if llm is not None:
        from mixle.task import design_model

        designed = design_model(rows, llm)
        if designed.source == "llm":
            candidates.append(("llm-designed", designed.estimator))

    rng = np.random.RandomState(seed)
    order = rng.permutation(len(rows))
    n_val = max(2, int(round(len(rows) * holdout)))
    val = [rows[i] for i in order[:n_val]]
    train = [rows[i] for i in order[n_val:]]

    frontier: list[dict[str, Any]] = []
    for name, est in candidates:
        try:
            fitted = optimize(train, est, max_its=max_its, out=None)
            enc = fitted.dist_to_encoder().seq_encode(val)
            score = float(np.mean(np.asarray(fitted.seq_log_density(enc), dtype=np.float64)))
            frontier.append({"name": name, "estimator": est, "heldout_mean_log_density": score})
        except Exception as exc:  # noqa: BLE001 - a failing candidate is reported, never silently dropped
            frontier.append({"name": name, "estimator": est, "error": f"{type(exc).__name__}: {exc}"})
    scored = sorted(
        (f for f in frontier if "heldout_mean_log_density" in f), key=lambda f: -f["heldout_mean_log_density"]
    )
    frontier = scored + [f for f in frontier if "error" in f]
    winner = scored[0]["estimator"] if scored else rec.estimator

    notes = [
        f"field {c.path}: {c.family}"
        + (
            f" (runner-up {c.runner_up}, gap {c.gap_bits:.1f} bits)"
            if c.runner_up is not None and c.gap_bits is not None
            else ""
        )
        for c in rec.fields
    ]
    notes += [f"dependency: {a} <-> {b} ({bits:.1f} bits for joint modeling)" for a, b, bits in rec.dependencies]
    notes += list(rec.warnings)
    notes += [
        f"candidate {f['name']}: "
        + (
            f"held-out mean log-density {f['heldout_mean_log_density']:.3f}"
            if "error" not in f
            else f"failed ({f['error']})"
        )
        for f in frontier
    ]
    m = Model(winner, notes=notes)
    m.frontier = frontier
    return m.fit(rows) if fit else m
