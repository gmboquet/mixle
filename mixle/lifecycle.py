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

import contextlib
import hashlib
import itertools
import json
import os
import pickle
import tempfile
import time
from pathlib import Path
from typing import Any

import numpy as np


def _sha256_file(path: Path) -> str:
    """The SHA-256 of a file's bytes, streamed, as ``sha256:<hex>``."""
    digest = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            digest.update(chunk)
    return f"sha256:{digest.hexdigest()}"


def _write_text_atomically(path: Path, text: str) -> None:
    """Write ``text`` to ``path`` via a private temp file in the same directory, then rename.

    The manifest is what makes an artifact directory readable at all -- it names the model file,
    its format and its digest -- so it is written last and promoted in one step. A crash partway
    through leaves the previous manifest (or none), never a truncated one that resolves to the
    wrong file.
    """
    fd, tmp_name = tempfile.mkstemp(dir=path.parent, prefix=path.name + ".", suffix=".tmp")
    try:
        with os.fdopen(fd, "w") as f:
            f.write(text)
            f.flush()
            os.fsync(f.fileno())
        os.replace(tmp_name, path)
    except BaseException:
        with contextlib.suppress(OSError):
            os.unlink(tmp_name)
        raise


def saddle_suspect(fitted: Any, data: Any, *, sample: int = 200, tol: float = 0.02) -> bool:
    """Family-agnostic symmetric-saddle check for latent-variable fits.

    At the symmetric saddle every component is identical, so every observation's component posterior
    is (numerically) uniform. Suspect when, over a data sample, NO observation's posterior deviates
    from uniform by more than ``tol``. Non-latent models (no ``posterior``/``components``) return False.
    """
    if not (hasattr(fitted, "posterior") and hasattr(fitted, "components")):
        return False
    if isinstance(sample, (bool, np.bool_)) or not isinstance(sample, (int, np.integer)) or sample < 1:
        raise ValueError(f"sample must be a positive integer, got {sample!r}")
    if isinstance(tol, (bool, np.bool_)) or not isinstance(tol, (int, float, np.integer, np.floating)):
        raise ValueError(f"tol must be a finite number in [0, 1), got {tol!r}")
    tol = float(tol)
    if not np.isfinite(tol) or not 0.0 <= tol < 1.0:
        raise ValueError(f"tol must be a finite number in [0, 1), got {tol!r}")
    rows = list(itertools.islice(data, int(sample)))
    if not rows:
        return False
    k = len(fitted.components)
    if k < 2:
        return False
    dev = 0.0
    for x in rows:
        post = np.asarray(fitted.posterior(x), dtype=np.float64)
        if post.shape != (k,):
            raise ValueError(f"posterior must have shape ({k},), got {post.shape}")
        if not np.isfinite(post).all() or np.any(post < 0.0):
            raise ValueError("posterior must contain finite, non-negative probabilities")
        total = float(post.sum())
        if not np.isclose(total, 1.0, rtol=1e-7, atol=1e-9):
            raise ValueError(f"posterior probabilities must sum to one, got {total!r}")
        dev = max(dev, float(np.max(np.abs(post - 1.0 / k))))
    return dev < tol


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
        source = data.records() if hasattr(data, "records") and callable(data.records) else data
        rows = list(source)
        if not rows:
            raise ValueError("fit requires at least one training record")
        if restarts not in ("auto", None) and (
            isinstance(restarts, (bool, np.bool_)) or not isinstance(restarts, (int, np.integer)) or int(restarts) < 1
        ):
            raise ValueError(f"restarts must be 'auto', None, or a positive integer, got {restarts!r}")

        if isinstance(calibrate, (bool, np.bool_)):
            cal_frac = 0.25 if calibrate else 0.0
        elif isinstance(calibrate, (int, float, np.integer, np.floating)):
            cal_frac = float(calibrate)
        else:
            raise ValueError(f"calibrate must be Boolean or a finite fraction in [0, 1), got {calibrate!r}")
        if not np.isfinite(cal_frac) or not 0.0 <= cal_frac < 1.0:
            raise ValueError(f"calibrate must be Boolean or a finite fraction in [0, 1), got {calibrate!r}")
        cal_holdout: list[Any] = []
        fit_data = rows
        if cal_frac > 0.0 and len(rows) >= 8:
            rng = optimize_kw.get("rng") or np.random.RandomState(0)
            order = rng.permutation(len(rows))
            n_cal = max(2, int(round(len(rows) * cal_frac)))
            cal_holdout = [rows[i] for i in order[:n_cal]]
            fit_data = [rows[i] for i in order[n_cal:]]

        self.fitted = optimize(fit_data, self.spec, **optimize_kw)
        self._fit_info = {"n": len(fit_data) if hasattr(fit_data, "__len__") else None, "when": time.time()}

        escape_tested = False
        want = 4 if restarts == "auto" else restarts
        # fit_data, never data: fit_data IS data when calibrate is off (or too little data to hold
        # anything out) -- the assignment above only diverges when cal_holdout was actually carved out.
        # A restart/saddle-check against the caller's original data would (a) let calibration rows sway
        # the saddle-suspicion verdict and (b) let _refit_symmetry_broken refit on them outright; the
        # replacement model would then get "evaluated held-out" against rows it was just trained on.
        if want and (restarts != "auto" or saddle_suspect(self.fitted, fit_data)):
            better, delta_ll, how = self._refit_symmetry_broken(fit_data, int(want), optimize_kw)
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
            self.certificate = certify(self.fitted, data=fit_data)
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
        estimator's components are not accessible. Returns ``(better, ll_gain, description)``.

        ``data`` must be exactly what the initial fit trained on (:meth:`fit`'s ``fit_data`` --
        the training-only partition when ``calibrate`` reserved a holdout), never the caller's original
        full dataset: every row seen here can end up back in ``self.fitted`` when a candidate is kept,
        so the caller's ``calibrate`` holdout would silently stop being held out."""
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
        rows = list(data)
        if not rows:
            raise ValueError("evaluate requires at least one held-out record")
        enc = d.dist_to_encoder().seq_encode(rows)
        ll = np.asarray(d.seq_log_density(enc), dtype=np.float64)
        if ll.shape != (len(rows),):
            raise ValueError(f"scorer returned shape {ll.shape}; expected one score for each of {len(rows)} records")
        if np.isnan(ll).any() or np.isposinf(ll).any():
            raise ValueError("scorer returned NaN or positive-infinite log density")
        return {"n": len(rows), "mean_log_density": float(ll.mean()), "total_log_density": float(ll.sum())}

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
        """Persist a durable artifact directory (model + manifest); :meth:`Model.load` restores it.

        A registry-serializable model is written as safe, type-tagged JSON (``model.json``); a model the
        serialization registry cannot represent (e.g. a torch-backed leaf) falls back to a pickle
        (``model.pkl``). The ``format`` recorded in the manifest tells :meth:`Model.load` which to read, so
        the common pure-model path never needs an unsafe pickle load.

        **This is a model export, not an evidence export.** The artifact carries the fitted
        distribution, ``notes``, and the fit record (``n``/``when``/``artifact_hash``) -- and nothing
        else. The :class:`~mixle.inference.EstimationCertificate` on ``certificate``, the
        :class:`~mixle.inference.CalibrationReport` on ``calibration`` and the candidate ``frontier``
        are *not* serialized and do not survive a round trip; a deployed model comes back able to
        predict but unable to show how it was chosen or how well it was calibrated. Whatever evidence
        was present at deploy time is listed in the manifest's ``evidence_not_exported`` and reported
        as a note on the loaded model, so the loss is visible rather than silent. Keep those receipts
        alongside the artifact yourself if a consumer needs to judge the model, not just run it.

        The model file is written first, then hashed, and the manifest naming it (``model_file``) and
        binding its ``model_sha256`` is promoted atomically last. A redeployment removes the previous
        generation's other-format model file, so an artifact directory never holds a stale
        ``model.pkl`` next to a fresh ``model.json``, and :meth:`load` re-checks the digest before
        deserializing anything.
        """
        d = self._require_fitted()
        out = Path(path)
        out.mkdir(parents=True, exist_ok=True)
        fmt = self._write_model(out, d)
        model_file = "model.json" if fmt == "json" else "model.pkl"
        # A redeploy that switches format used to leave the old generation's file in place, so the
        # directory held two models and only the manifest said which one was real -- and a manifest
        # lost or rolled back to a mixed state resolved to the stale one.
        stale = out / ("model.pkl" if fmt == "json" else "model.json")
        if stale.exists():
            stale.unlink()
        evidence_not_exported = [
            name
            for name, present in (
                ("certificate", self.certificate is not None),
                ("calibration", self.calibration is not None),
                ("frontier", self.frontier is not None),
            )
            if present
        ]
        manifest = {
            "family": type(d).__name__,
            "created_at": time.time(),
            "fit": self._fit_info,
            "notes": self.notes,
            "format": fmt,
            "model_file": model_file,
            "model_sha256": _sha256_file(out / model_file),
            "evidence_not_exported": evidence_not_exported,
            "mixle_artifact": "lifecycle.Model/v1",
        }
        _write_text_atomically(out / "manifest.json", json.dumps(manifest, indent=2, default=str))
        return str(out)

    @staticmethod
    def _write_model(out: Path, d: Any) -> str:
        """Write ``d`` as safe JSON when the registry can represent it, else pickle. Returns the format used.

        Only the registry's explicit "this type has no JSON form" answer (:class:`SerializationError`)
        selects the pickle path. A blanket ``except Exception`` also caught encoder bugs, registry
        initialization failures and JSON encoding faults, so an internal ``TypeError`` silently turned
        an ordinary JSON-serializable Gaussian into an executable ``model.pkl`` -- a programming
        failure quietly downgrading the artifact's security and portability contract. Those errors now
        propagate.
        """
        from mixle.utils.serialization import (
            SerializationError,
            ensure_pysp_serialization_registry,
            to_serializable,
        )

        ensure_pysp_serialization_registry()  # a registry that cannot initialize is not "needs pickle"
        try:
            payload = to_serializable(d)
        except SerializationError:  # not registry-serializable (torch leaf, custom object): use pickle
            with open(out / "model.pkl", "wb") as f:
                pickle.dump(d, f)
            return "pickle"
        (out / "model.json").write_text(json.dumps(payload))
        return "json"

    @classmethod
    def load(cls, path: str, *, trust_code: bool = False) -> Model:
        """Restore a :class:`Model` from an artifact directory created by :meth:`deploy`.

        .. warning::

           A ``pickle``-format artifact (a torch-backed model, or one written by an older mixle)
           executes arbitrary code from the file when loaded, exactly like ``pickle.load`` on any
           untrusted input. JSON-format artifacts are deserialized through a type-tagged registry that
           never imports or executes code from the payload -- **unless** the model contains a
           NeuralLeaf-family component, whose weights are embedded as a pickle blob inside the
           otherwise-safe JSON (see :mod:`mixle.models._neural_serial`); loading one of those also
           executes code. There is no way to tell which case applies without attempting the load.

           Because of this, ``load`` refuses to run any code-executing path unless the caller passes
           ``trust_code=True`` -- an explicit statement that the artifact's source is trusted. Without
           it, a pickle-format artifact raises immediately, and a JSON artifact raises only if it
           actually contains an embedded module (a pure-statistical model still loads normally).

        When the manifest records a ``model_sha256`` (every artifact written by this version does),
        the named model file is hashed and checked against it *before* anything is deserialized, so a
        model file edited, truncated or swapped after deployment is refused rather than loaded. An
        older manifest without that field is loaded unverified, exactly as before -- there is nothing
        to check it against.

        The restored :class:`Model` carries the fitted distribution, ``notes`` and the fit record.
        The certificate, calibration report and candidate frontier are not in the artifact at all
        (see :meth:`deploy`); when the deploying model had any of them, a note saying so is appended.
        """
        from mixle.utils.serialization import SerializationError, trusted_deserialization

        p = Path(path)
        try:
            manifest = json.loads((p / "manifest.json").read_text())
        except (OSError, ValueError):
            manifest = {}
        fmt = manifest.get("format", "pickle")  # artifacts predating the format field are pickle-only
        model_file = str(manifest.get("model_file") or ("model.json" if fmt == "json" else "model.pkl"))
        if Path(model_file).name != model_file:
            raise SerializationError(f"manifest model_file {model_file!r} must be a plain file name")
        expected_digest = manifest.get("model_sha256")
        if expected_digest is not None:
            actual = _sha256_file(p / model_file)
            if actual != expected_digest:
                raise SerializationError(
                    f"{path!r}: {model_file} does not match the digest recorded in its manifest "
                    f"(expected {expected_digest}, found {actual}); the artifact is incomplete or altered."
                )
        with trusted_deserialization() if trust_code else contextlib.nullcontext():
            if fmt == "json":
                from mixle.utils.serialization import ensure_pysp_serialization_registry, from_serializable

                ensure_pysp_serialization_registry()
                fitted = from_serializable(json.loads((p / model_file).read_text()))
            else:
                if not trust_code:
                    raise SerializationError(
                        f"{path!r} is a pickle-format artifact: loading it executes arbitrary code from "
                        "the file. Pass trust_code=True to Model.load() only if you trust its source."
                    )
                with open(p / model_file, "rb") as f:
                    fitted = pickle.load(f)  # noqa: S301 - trust_code=True required by the caller above
        m = cls(fitted)
        m.fitted = fitted
        m.notes = list(manifest.get("notes", []))
        m._fit_info = dict(manifest.get("fit") or {})
        missing = manifest.get("evidence_not_exported") or []
        if missing:
            m.notes.append(
                f"artifact export dropped: {', '.join(str(name) for name in missing)} "
                "(deploy() persists the model, notes and fit record only)"
            )
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
    max_candidates: int | None = None,
    timeout: float | None = None,
    **recommend_kw: Any,
) -> Model:
    """Propose a model for ``data`` from a *verified frontier* of candidates and return the winner.

    ``max_candidates=0`` or ``timeout=0.0`` is the one exception to "verified": every candidate is then
    skipped rather than fitted/scored, and the returned winner falls back to the raw, unverified heuristic
    recommendation -- honestly disclosed (never silently), via a ``"search budget: skipped ..."`` line in
    ``Model.notes`` and a ``"skipped": "search budget reached (...)"`` entry per candidate in
    ``Model.frontier``, both readable through ``explain()``. Any other ``max_candidates``/``timeout`` still
    verifies every candidate it has budget for before choosing a winner.

    Candidates come from every proposer the library has — the heuristic recommendation
    (:func:`mixle.task.recommend.recommend_model`, dependency-aware in the narrow sense of a joint
    multivariate-Gaussian candidate for fully-observed numeric vector rows), a structured-search candidate
    (:func:`mixle.inference.optimize` called with no pre-built estimator — its own no-estimator
    auto-structure-search, the fuller copula/learned-Bayesian-network dependence upgrade described in
    :doc:`/automatic-modeling-contract`'s "Dependence between fields"), the plain independence baseline
    (:func:`mixle.utils.automatic.get_estimator`), and, when an ``llm`` handle is given, an LLM-designed
    structure (:func:`mixle.task.design.design_model`, allowlisted-spec, fit-validated). Each candidate is
    **fitted on a train split and scored on held-out data**, so the ranking is out-of-sample, not a guess.
    The winner becomes the returned :class:`Model`; the full ranking lands in ``Model.frontier`` and the
    per-field confidence / dependency / candidate notes in ``Model.notes`` (shown by ``explain()``). Pass
    ``fit=True`` to also fit the winner to all of ``data`` before returning.

    The frontier search is bounded by ``max_candidates`` (evaluate at most this many candidates, in
    proposer order — the heuristic recommendation is always first) and ``timeout`` (stop starting new
    candidate fits once this many wall-clock seconds have elapsed). Both default to ``None`` (unbounded).
    A candidate skipped for budget is **recorded** in ``Model.frontier`` and ``Model.notes``, never silently
    dropped, so a bounded search reports exactly what it did not evaluate.
    """
    from mixle.inference import optimize
    from mixle.task import recommend_model

    # Negative values here have no reasonable interpretation (verify with a negative amount of time,
    # hold out a negative fraction) and must not silently fail open. max_candidates=0 / timeout=0.0 are
    # different: an intentional, already-tested "skip verification, return the fast heuristic" escape
    # hatch (see propose_budget_test.py's test_zero_timeout_skips_and_falls_back_to_recommendation) --
    # every skipped candidate is still recorded in notes/frontier ("search budget reached"), so this
    # stays honest about what it did not evaluate rather than needing to be disallowed outright.
    if isinstance(max_candidates, (bool, np.bool_)) or (
        max_candidates is not None and (not isinstance(max_candidates, (int, np.integer)) or int(max_candidates) < 0)
    ):
        raise ValueError(f"max_candidates must be a non-negative integer or None, got {max_candidates!r}")
    if isinstance(timeout, (bool, np.bool_)) or (
        timeout is not None
        and (
            not isinstance(timeout, (int, float, np.integer, np.floating))
            or not np.isfinite(float(timeout))
            or float(timeout) < 0.0
        )
    ):
        raise ValueError(f"timeout must be a finite non-negative number of seconds or None, got {timeout!r}")
    if isinstance(holdout, (bool, np.bool_)) or not isinstance(holdout, (int, float, np.integer, np.floating)):
        raise ValueError(f"holdout must be a finite number in (0, 1), got {holdout!r}")
    holdout = float(holdout)
    if not np.isfinite(holdout) or not 0.0 < holdout < 1.0:
        raise ValueError(f"holdout must be a finite number in (0, 1), got {holdout!r}")
    if isinstance(seed, (bool, np.bool_)) or not isinstance(seed, (int, np.integer)) or not 0 <= seed < 2**32:
        raise ValueError(f"seed must be an integer in [0, 2**32), got {seed!r}")
    if isinstance(max_its, (bool, np.bool_)) or not isinstance(max_its, (int, np.integer)) or max_its < 1:
        raise ValueError(f"max_its must be a positive integer, got {max_its!r}")

    rows = list(data)
    if len(rows) < 3:
        raise ValueError("propose requires at least three records for a non-empty train/holdout split")
    rec = recommend_model(rows, **recommend_kw)
    candidates: list[tuple[str, Any]] = [("recommended", rec.estimator)]
    # optimize()'s own no-estimator auto-structure-search (structure="auto", the default, reached by
    # passing estimator=None) is the only route to a copula or learned-Bayesian-network dependence
    # upgrade -- see docs/automatic-modeling-contract.rst's "Dependence between fields". Model already
    # treats spec=None as a first-class state ("the estimator is inferred at fit time" -- see Model's
    # own docstring/__repr__), and Model.fit()'s optimize(fit_data, self.spec, ...) already re-runs
    # structure search correctly on a later re-fit, so this candidate needs no special-casing in the
    # scoring loop below or in what "winner" becomes: it participates in max_candidates/timeout exactly
    # like every other candidate, one slot each, regardless of how many structures it tries internally.
    # Safe unconditionally: non-tuple (fixed-length numeric-vector) rows fall through to a plain fit
    # rather than erroring, and a failure either way is caught and reported by this loop's own
    # try/except below, never silently dropped.
    candidates.append(("structured", None))
    try:  # the independence baseline the frontier has to beat (skip when identical to the recommendation)
        from mixle.utils.automatic import get_estimator

        indep = get_estimator(rows)
        # ParameterEstimator has no value-based __repr__ (it falls back to the default object repr,
        # keyed on identity/memory address), so a repr() comparison here never matches even when the
        # two estimators are structurally identical -- the "skip when identical" intent silently
        # never fires, doubling the frontier's fit cost. to_dict() is the real structural comparison.
        if indep.to_dict() != rec.estimator.to_dict():
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
    evaluated = 0
    budget_start = time.monotonic()
    for candidate_index, (name, est) in enumerate(candidates):
        over_count = max_candidates is not None and evaluated >= max_candidates
        over_time = timeout is not None and (time.monotonic() - budget_start) > timeout
        if over_count or over_time:
            reason = "max_candidates" if over_count else "timeout"
            frontier.append({"name": name, "estimator": est, "skipped": f"search budget reached ({reason})"})
            continue
        try:
            candidate_rng = np.random.RandomState((int(seed) + candidate_index) % (2**32))
            fitted = optimize(train, est, max_its=int(max_its), rng=candidate_rng, out=None)
            enc = fitted.dist_to_encoder().seq_encode(val)
            scores = np.asarray(fitted.seq_log_density(enc), dtype=np.float64)
            if scores.shape != (len(val),):
                raise ValueError(
                    f"candidate scorer returned shape {scores.shape}; expected one score for each "
                    f"of {len(val)} held-out records"
                )
            if not np.isfinite(scores).all():
                raise ValueError("candidate scorer returned a non-finite held-out log density")
            score = float(np.mean(scores))
            entry = {
                "name": name,
                "estimator": est,
                "heldout_mean_log_density": score,
                "candidate_index": candidate_index,
            }
            try:
                from mixle.data.hashing import model_hash

                entry["fitted_artifact_hash"] = model_hash(fitted)
            except Exception:  # noqa: BLE001 - artifact hashing must not discard a valid candidate
                entry["fitted_artifact_hash"] = None
            frontier.append(entry)
            evaluated += 1
        except Exception as exc:  # noqa: BLE001 - a failing candidate is reported, never silently dropped
            frontier.append({"name": name, "estimator": est, "error": f"{type(exc).__name__}: {exc}"})
            evaluated += 1
    scored = sorted(
        (f for f in frontier if "heldout_mean_log_density" in f),
        key=lambda f: (-f["heldout_mean_log_density"], f["candidate_index"], f["name"]),
    )
    frontier = scored + [f for f in frontier if "error" in f or "skipped" in f]
    winner = scored[0]["estimator"] if scored else rec.estimator
    skipped_names = [f["name"] for f in frontier if "skipped" in f]

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

    def _candidate_note(f: dict[str, Any]) -> str:
        if "skipped" in f:
            return f"candidate {f['name']}: {f['skipped']}"
        if "error" in f:
            return f"candidate {f['name']}: failed ({f['error']})"
        return f"candidate {f['name']}: held-out mean log-density {f['heldout_mean_log_density']:.3f}"

    notes += [_candidate_note(f) for f in frontier]
    if skipped_names:
        notes.append(f"search budget: skipped {len(skipped_names)} candidate(s) unevaluated: {skipped_names}")
    m = Model(winner, notes=notes)
    m.frontier = frontier
    if fit:
        m.fit(rows, restarts=None, max_its=int(max_its), rng=np.random.RandomState(int(seed)))
        try:
            from mixle.data.hashing import model_hash

            m._fit_info["artifact_hash"] = model_hash(m.fitted)
        except Exception:  # noqa: BLE001 - artifact hashing must not discard a valid final fit
            m._fit_info["artifact_hash"] = None
    return m
