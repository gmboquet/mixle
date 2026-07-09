"""``mixle.Model`` -- the model lifecycle as one object with consistent verbs.

Everything here exists elsewhere in the library; this facade makes the lifecycle *discoverable* without
knowing which subpackage owns which verb::

    m = mixle.propose(data)          # a model shape recommended from the data (with confidence + caveats)
    m.fit(data)                      # inference chosen from the structure (EM / MLE / closed form)
    m.evaluate(holdout)              # held-out scores
    m.sample(5)                      # draw new records
    m.enumerate().top_k(3)           # most-probable support (discrete/structured families)
    m.posterior(x)                   # latent posteriors (mixtures, HMMs, ...)
    m.distill(teacher, inputs)       # compact deployable student in front of the teacher (task spine)
    m.deploy("artifacts/m")          # durable artifact directory; Model.load() restores it
    m.explain()                      # what it is, what it supports, and how it was proposed
    m.explain_prediction(x)          # exact per-part attribution of one score
    m.forecast(history, h)           # horizon predictions with calibrated intervals (HMMs)
    m.do({field: value})             # graph-surgery intervention (learned Bayesian networks)
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


def saddle_suspect(fitted: Any, data: Any, *, sample: int = 200, tol: float = 0.02) -> bool:
    """Family-agnostic symmetric-saddle check for latent-variable fits.

    At the symmetric saddle every component is identical, so every observation's component posterior
    is (numerically) uniform. Suspect when, over a data sample, NO observation's posterior deviates
    from uniform by more than ``tol``. Non-latent models (no ``posterior``/``components``) return False.
    """
    if not (hasattr(fitted, "posterior") and hasattr(fitted, "components")):
        return False
    rows = list(data)[: int(sample)]
    if not rows:
        return False
    k = len(fitted.components)
    if k < 2:
        return False
    try:
        dev = 0.0
        for x in rows:
            post = np.asarray(fitted.posterior(x), dtype=np.float64).reshape(-1)
            dev = max(dev, float(np.max(np.abs(post - 1.0 / k))))
        return dev < tol
    except Exception:  # noqa: BLE001 - a family whose posterior we cannot read is not "suspect"
        return False


class Model:
    """One object over the model lifecycle: build / fit / evaluate / enumerate / distill / deploy / use."""

    def __init__(self, spec: Any = None, *, notes: list[str] | None = None) -> None:
        """``spec`` is a prototype distribution, an estimator, or ``None`` (infer from data at fit time)."""
        self.spec = spec
        self.fitted: Any = None
        self.notes: list[str] = list(notes or [])
        self.frontier: list[dict[str, Any]] | None = None  # candidate ranking when built by propose()
        self.certificate: Any = None  # EstimationCertificate attached by fit() -- how each block was solved
        self.calibration: Any = None  # CalibrationReport attached by fit(calibrate=...) -- UQ validation
        self._fit_info: dict[str, Any] = {}

    # --- fit / use -------------------------------------------------------------------------------
    def fit(self, data: Any, *, restarts: Any = "auto", calibrate: float | bool = False, **optimize_kw: Any) -> Model:
        """Fit via :func:`mixle.inference.optimize`; the algorithm follows from the model's structure.

        ``restarts="auto"`` (default) makes latent-variable fitting genuinely automatic: after the
        plain fit, a family-agnostic saddle check runs (a mixture stuck at the symmetric saddle gives
        every observation a ~uniform component posterior), and on suspicion the fit silently reruns as
        multi-restart EM (:func:`mixle.inference.best_of`), keeping the better log-likelihood and
        recording what happened in ``notes``. Pass an int to force that many restarts up front, or
        ``restarts=None`` for the raw single fit.

        ``calibrate`` (opt-in, default off): reserve a holdout slice (a fraction, or ``True`` for
        25%), fit on the rest, and attach a :class:`~mixle.inference.CalibrationReport` on
        ``self.calibration`` measuring calibration quality on held-out data
        (PIT test + held-out log-density). Off by default because it costs training data."""
        from mixle.inference import certify, optimize

        optimize_kw.setdefault("out", None)

        cal_frac = 0.25 if calibrate is True else float(calibrate or 0.0)
        cal_holdout: list[Any] = []
        fit_data = data
        if cal_frac > 0.0 and hasattr(data, "__len__") and len(data) >= 8:
            rows = list(data)
            rng = optimize_kw.get("rng") or np.random.RandomState(0)
            order = rng.permutation(len(rows))
            n_cal = max(2, int(round(len(rows) * cal_frac)))
            cal_holdout = [rows[i] for i in order[:n_cal]]
            fit_data = [rows[i] for i in order[n_cal:]]

        self.fitted = optimize(fit_data, self.spec, **optimize_kw)
        self._fit_info = {"n": len(fit_data) if hasattr(fit_data, "__len__") else None, "when": time.time()}

        escape_tested = False
        want = 4 if restarts == "auto" else restarts
        if want and (restarts != "auto" or saddle_suspect(self.fitted, data)):
            better, delta_ll, how = self._refit_symmetry_broken(data, int(want), optimize_kw)
            if better is not None:
                self.fitted = better
                escape_tested = True
                why = "saddle suspected" if restarts == "auto" else "restarts requested"
                self.notes.append(f"{why}: {how} kept (log-lik +{delta_ll:.3f})")
            elif restarts == "auto":
                self.notes.append("saddle suspected: symmetry-broken refits did not improve — inspect the fit")
        # the estimation certificate: which method solved each block, how strong the guarantee, and
        # exactly where (if anywhere) gradient descent was unavoidable. Low-overhead inspection, computed once.
        try:
            self.certificate = certify(self.fitted, escape_tested=escape_tested)
        except Exception:  # noqa: BLE001 - certification is a report; never let it break a fit
            self.certificate = None
        if cal_holdout:
            from mixle.inference import calibration_report

            try:
                self.calibration = calibration_report(self.fitted, cal_holdout)
            except Exception:  # noqa: BLE001 - a calibration report never breaks a fit
                self.calibration = None
        return self

    def _refit_symmetry_broken(self, data: Any, trials: int, optimize_kw: dict) -> tuple[Any, float, str]:
        """Escape the symmetric saddle by construction, not by re-rolling the same init.

        For a mixture estimator, each attempt fits every component on its OWN random disjoint shard of
        the data (a hard-partition init: components start different because they saw different data),
        then runs full EM from that start. Falls back to :func:`mixle.inference.best_of` when the
        estimator's components are not accessible. Returns ``(better, ll_gain, description)``."""
        from mixle.inference import best_of, optimize

        rng = optimize_kw.get("rng") or np.random.RandomState(1)
        max_its = int(optimize_kw.get("max_its", 20))
        base = self.evaluate(data)["total_log_density"]
        rows = list(data)

        comp_ests = getattr(self.spec, "estimators", None)
        best_ll, best_model, how = base, None, ""
        if comp_ests:
            from mixle.stats import MixtureDistribution

            k = len(comp_ests)
            # Random disjoint shards are exchangeable samples of the SAME mixture — each component would
            # refit the pooled law and the symmetry survives. Sort by the current (pooled/saddled) fit's
            # log-density instead: contiguous quantile blocks live in different density regions, so the
            # components start genuinely different. Trials differ by rotating the sorted order.
            enc0 = self.fitted.dist_to_encoder().seq_encode(rows)
            scores = np.asarray(self.fitted.seq_log_density(enc0), dtype=np.float64)
            sorted_order = np.argsort(scores)
            for _ in range(int(trials)):
                order = np.roll(sorted_order, int(rng.randint(len(rows))))
                shards = np.array_split(order, k)
                try:
                    comps = [
                        optimize([rows[i] for i in shard], comp_ests[j], max_its=2, out=None)
                        for j, shard in enumerate(shards)
                    ]
                    init = MixtureDistribution(comps, [1.0 / k] * k)
                    cand = optimize(rows, self.spec, max_its=max_its, prev_estimate=init, out=None)
                except Exception:  # noqa: BLE001 - a failed attempt is just not an improvement
                    continue
                enc = cand.dist_to_encoder().seq_encode(rows)
                ll = float(np.sum(np.asarray(cand.seq_log_density(enc), dtype=np.float64)))
                if ll > best_ll + 1e-6:
                    best_ll, best_model, how = ll, cand, f"hard-partition init x{trials}"
        if best_model is None:  # estimator shape unknown (or partitions didn't help): plain multi-restart
            ll_new, cand = best_of(
                rows,
                None,
                self.spec,
                trials=int(trials),
                max_its=max_its,
                init_p=0.1,
                delta=optimize_kw.get("delta", 1.0e-9),
                rng=rng,
                out=None,
            )
            if np.isfinite(ll_new) and ll_new > best_ll + 1e-6:
                best_ll, best_model, how = float(ll_new), cand, f"best-of-{trials} restart"
        if best_model is not None:
            return best_model, float(best_ll - base), how
        return None, 0.0, ""

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
        """Draw samples from the fitted distribution."""
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
        """Distill a compact deployable student via :func:`mixle.task.solve`.

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
        """Restore a :class:`Model` from an artifact directory created by :meth:`deploy`."""
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

    # --- the analysis verbs (delegate to the inference front doors) -------------------------------
    def explain_prediction(self, x: Any):
        """Exact per-part attribution of ``log p(x)`` — :func:`mixle.inference.explain`."""
        from mixle.inference import explain

        return explain(self._require_fitted(), x)

    def forecast(self, history: Any, horizon: int, **kw: Any):
        """Horizon predictions with calibrated intervals — :func:`mixle.inference.forecast` (HMMs)."""
        from mixle.inference import forecast

        return forecast(self._require_fitted(), history, horizon, **kw)

    def do(self, interventions: dict, **kw: Any):
        """Graph-surgery intervention — :func:`mixle.inference.do` (M0's generic engine: dependency
        trees, Bayesian networks, composites, mixtures; reduces to :func:`mixle.inference.bn_do`'s
        exact behavior for a fitted ``HeterogeneousBayesianNetwork``)."""
        from mixle.inference import do

        return do(self._require_fitted(), interventions, **kw)

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
