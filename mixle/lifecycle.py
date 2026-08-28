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
    m.forecast(history, h)           # horizon predictions, model-predictive MC intervals (HMMs)
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
import warnings
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import numpy as np

from mixle.utils.exact import require_explicit_true


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

    The promoted file carries the process-default permissions (``0o666`` masked by the umask --
    ``0o644`` under the usual ``022``), exactly what ``open()`` gives every other artifact file.
    ``mkstemp`` alone creates ``0o600`` temp files, so the rename used to leave the manifest
    owner-only next to a world-readable ``model.json``: a serving process running as another user
    could read the model but not the manifest that names it. Neither file is more sensitive than
    the other -- the manifest describes the model -- so they deliberately share one policy, the
    caller's umask (a ``077`` umask still yields a consistently private ``0o600`` pair).
    """
    fd, tmp_name = tempfile.mkstemp(dir=path.parent, prefix=path.name + ".", suffix=".tmp")
    try:
        with os.fdopen(fd, "w") as f:
            f.write(text)
            f.flush()
            os.fsync(f.fileno())
        # os.umask can only be read by setting it; set-and-restore is the standard idiom. (Racy in
        # a program mutating its umask concurrently -- worst case the manifest gets that thread's
        # mask, which is still a policy the process asked for.)
        umask = os.umask(0o022)
        os.umask(umask)
        os.chmod(tmp_name, 0o666 & ~umask)
        os.replace(tmp_name, path)
    except BaseException:
        with contextlib.suppress(OSError):
            os.unlink(tmp_name)
        raise


def json_read_back_failure(text: str) -> str | None:
    """Why the JSON in ``text`` would fail to decode, or ``None`` when it reads back cleanly.

    Encoding is not evidence of a round trip. ``to_serializable`` accepts any registered class's
    ``__dict__``, while the decoder additionally requires that state to reconstruct through the
    class's own constructor -- so a family whose estimator pins a fit annotation onto the fitted
    object, or whose state names a parameter differently from the constructor's, encodes cleanly
    and then refuses to load. The only honest test of "can this be read back" is to read it back,
    against the exact text about to be written.

    The decode runs inside ``trusted_deserialization`` on purpose. This payload was encoded from a
    live object in this very process moments ago; the trust gate exists to stop an artifact that
    arrived from somewhere else from executing code, and it has nothing to say about our own
    in-memory model. Probing untrusted would report every model carrying an embedded torch module
    as unreadable and demote a perfectly good JSON artifact to a pickle.
    """
    from mixle.utils.serialization import from_serializable, trusted_deserialization

    try:
        payload = json.loads(text)
    except ValueError as exc:
        return f"{type(exc).__name__}: {exc}"
    try:
        # Reconstruction re-runs constructor-time repairs (a covariance re-ridged, a boundary
        # clamped). The caller already saw those at fit time, and they describe the model rather
        # than the write, so re-emitting them here would be noise attributed to the wrong step.
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            with trusted_deserialization():
                from_serializable(payload)
    except Exception as exc:  # noqa: BLE001 - the question is "does it read back", not "how does it fail"
        # Deliberately blind: whatever the reader raises here is exactly what it will raise later in
        # a serving process, so every type of failure is equally disqualifying. Nothing is swallowed
        # -- the caller reports this string in the manifest and in its warning.
        return f"{type(exc).__name__}: {exc}"
    return None


#: The one manifest schema this module writes and reads. Written into every manifest as
#: ``mixle_artifact`` and CHECKED on load: an artifact stamped with a different schema tag is
#: refused with its tag named, instead of being parsed as if it were this schema and failing
#: somewhere misleading. Manifests without the tag predate it and load exactly as before.
_ARTIFACT_TAG = "lifecycle.Model/v1"


class DeployedArtifact(str):
    """The artifact path :meth:`Model.deploy` returns, annotated with what was written there.

    A plain ``str`` (every ``os.path`` / ``Path`` use keeps working), plus the two facts a caller
    needs at the call site rather than by re-opening the manifest: ``format`` (``"json"`` or
    ``"pickle"``) and ``format_fallback`` (``None`` for JSON; otherwise the sentence saying why
    this artifact is a pickle and therefore costs its readers ``trust_code=True``). A deploy
    pipeline can assert ``result.format == "json"`` to enforce a no-pickle policy in one line.
    """

    format: str
    format_fallback: str | None

    def __new__(cls, path: str, *, format: str, format_fallback: str | None) -> DeployedArtifact:
        self = super().__new__(cls, path)
        self.format = format
        self.format_fallback = format_fallback
        return self


def _tabular_records(data: Any) -> list:
    """``data`` as a list of observation records -- always one record per ROW, never per column.

    A pandas ``DataFrame`` iterates as its column labels and a mapping iterates as its keys, so a bare
    ``list(data)`` silently modeled the five HEADER STRINGS of an 891-row table (and stamped
    ``n_rows=5``) while :func:`mixle.inference.optimize` handled the same frame correctly. Convert the
    tabular inputs into the row records the estimation path expects (DataFrame -> one record per row
    via :func:`mixle.data.sources.pandas_source.dataframe_records`, exactly the shape ``optimize``'s
    ``fields`` path produces; DataSource -> ``records()``; mapping-of-columns expanded here), and
    leave already-row-shaped sequences byte-identical to the historical ``list(data)``.
    """
    if hasattr(data, "records") and callable(data.records) and hasattr(data, "structure"):
        return list(data.records())  # a mixle DataSource
    if hasattr(data, "columns") and hasattr(data, "itertuples"):  # a pandas DataFrame (duck-typed)
        from mixle.data.sources.pandas_source import dataframe_records

        return dataframe_records(data)
    if isinstance(data, Mapping):
        # {field: column}: iterating the mapping would model the field-name strings. Build one
        # record per row across the columns (scalar for a single column, tuple otherwise) --
        # the same shape a DataFrame of those columns produces.
        if not data:
            raise ValueError(
                "received an empty mapping; pass records (a list of observations) or a mapping of "
                "equal-length columns keyed by field name"
            )
        lengths: dict[Any, int] = {}
        columns: list[list[Any]] = []
        for name, column in data.items():
            if isinstance(column, (str, bytes)) or not hasattr(column, "__len__"):
                raise ValueError(
                    f"a mapping is read as {{field: column}}, but field {name!r} is not a sized "
                    f"column (got {type(column).__name__}); for row-shaped data pass a list of "
                    "records instead of a single mapping"
                )
            from mixle.data.sources.pandas_source import column_records

            columns.append(column_records(column))
            lengths[name] = len(columns[-1])
        if len(set(lengths.values())) > 1:
            raise ValueError(f"mapping-of-columns input needs equal-length columns, got lengths {lengths}")
        return columns[0] if len(columns) == 1 else [tuple(row) for row in zip(*columns, strict=True)]
    if type(data).__name__ == "Series" and type(data).__module__.startswith("pandas"):
        # A bare Series carries pandas' own missing-value convention rather than the row-shaped
        # sentinel Model.fit/evaluate/propose expect, and the generic list(data) fallthrough below
        # does not normalize it, so a Model built from a Series meets the same dtype-dependent
        # sentinel mismatch the auto-inference path had (campaign four, T2-02, the Series half).
        from mixle.data.sources.pandas_source import column_records

        return column_records(data)
    try:
        return list(data)
    except TypeError as exc:
        # A single observation (m.fit(0.5), m.evaluate(0.5)) died here as a bare
        # "TypeError: 'float' object is not iterable" that named neither the expectation nor the
        # verb that does take one observation.
        raise ValueError(
            f"data must be a collection of observation records (a list, array, DataFrame, "
            f"DataSource, or mapping of columns); got a single {type(data).__name__}. To score one "
            "observation use model(x); to fit or evaluate on it, wrap it in a list."
        ) from exc


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
        if isinstance(spec, type):
            # A class here used to travel all the way into the estimation engine and die as
            # ``AttributeError: type object ... has no attribute 'accumulator_factory'``.
            raise TypeError(
                f"Model spec must be a distribution or estimator INSTANCE (or None to infer the "
                f"estimator from the data at fit time); got the class {spec.__name__} itself -- "
                f"instantiate it first, e.g. Model({spec.__name__}(...))."
            )
        self.spec = spec
        self.fitted: Any = None
        self.notes: list[str] = list(notes or [])
        self.frontier: list[dict[str, Any]] | None = None  # candidate ranking when built by propose()
        self.certificate: Any = None  # EstimationCertificate attached by fit() -- how each block was solved
        self.calibration: Any = None  # CalibrationReport attached by fit(calibrate=...) -- UQ validation
        # Why each evidence artifact above is or is not present. Both producing steps swallow every
        # exception and leave their attribute at None, so an INTERNAL FAILURE used to be
        # indistinguishable from a caller who never requested (or whose model never supported) the
        # check: a fit whose certification raised came back with certificate=None, calibration=None and
        # empty notes, exactly like an ordinary uncertified fit. A fit may remain usable when its
        # evidence could not be produced, but it must not erase WHY the evidence is absent. Keyed by
        # step name; see _record_evidence for the record shape.
        self.evidence: dict[str, dict[str, Any]] = {}
        self._fit_info: dict[str, Any] = {}

    def _record_evidence(self, step: str, status: str, *, error: BaseException | None = None, **detail: Any) -> None:
        """Record one evidence step's typed outcome: ``attempted``/``succeeded``/``failed``/``not_applicable``.

        A ``failed`` record also names the failure class and message and appends a note, so the reason
        an evidence artifact is missing survives into ``explain()`` rather than vanishing.
        """
        record: dict[str, Any] = {"status": status, "error": None, "error_type": None, **detail}
        if error is not None:
            record["error_type"] = type(error).__name__
            record["error"] = str(error)
            self.notes.append(f"{step} failed: {type(error).__name__}: {error}")
        self.evidence[step] = record

    # --- fit / use -------------------------------------------------------------------------------
    def fit(self, data: Any, *, restarts: Any = "auto", calibrate: float | bool = False, **optimize_kw: Any) -> Model:
        """Fit via :func:`mixle.inference.optimize`; the algorithm follows from the model's structure.

        The fit iterates EM to its tolerance: unless the caller passes ``max_its=``, the iteration
        cap defaults to 500 here (``optimize``'s own default of 10 regularly stops a mixture fit on
        the initialization plateau -- 58 nats short of the optimum on Old Faithful -- with no signal).
        The ``delta`` stopping rule ends a converged fit long before the cap; a fit that does hit the
        cap without converging is disclosed in ``notes`` and in ``self._fit_info`` (``n_iter``,
        ``converged``, read from the model's :meth:`~mixle.stats.compute.pdist.FitProvenance` receipt).
        Passing ``delta=None`` through ``**optimize_kw`` requests a fixed iteration count instead --
        see :func:`mixle.inference.optimize`'s own ``delta`` docs for the caveat that a rejected
        (non-improving) EM step still ends the loop early even then, which ``optimize`` discloses via
        a ``UserWarning`` (not ``notes``) when it happens.

        A ``spec`` that is a prototype *distribution* also supplies the EM starting point: its
        parameter values are honored as the initialization (``optimize``'s bare prototype coercion
        keeps only the structure and draws a random subsample start). Pass ``prev_estimate=`` yourself
        to override the start explicitly.

        ``restarts="auto"`` (default) makes latent-variable fitting genuinely automatic: after the
        plain fit, a family-agnostic saddle check runs (a mixture stuck at the symmetric saddle gives
        every observation a ~uniform component posterior) -- and a latent fit that exhausted its
        iteration cap without converging is treated as equally suspect. On suspicion the fit reruns
        from diversified initializations (hard-partition component starts, falling back to
        :func:`mixle.inference.best_of`), keeping the better log-likelihood and recording what
        happened in ``notes``. Pass an int to force that many restarts up front, or ``restarts=None``
        for the raw single fit.

        ``calibrate`` (opt-in, default off): reserve a holdout slice (a fraction, or ``True`` for
        25%), fit on the rest, and attach a :class:`~mixle.inference.CalibrationReport` on
        ``self.calibration`` measuring calibration quality on held-out data
        (PIT test + held-out log-density). Off by default because it costs training data.

        Neither evidence step can break a fit, but neither silently disappears either: ``self.evidence``
        carries a typed ``succeeded``/``failed``/``not_applicable`` record per step, naming the failure
        class and message when one failed, and a failure also lands in ``notes`` (so ``explain()``
        shows it). ``certificate=None`` alone cannot tell you whether certification was never
        applicable or raised."""
        from mixle.inference import certify, optimize
        from mixle.stats.compute.pdist import SequenceEncodableProbabilityDistribution

        optimize_kw.setdefault("out", None)
        # optimize()'s stable default (max_its=10) exists for its own callers; a lifecycle fit is
        # "fit this model", so run EM until the delta tolerance actually stops it. 500 is a safety
        # cap, not a target -- converged fits exit on delta far earlier, and hitting the cap is
        # disclosed below via the fit-provenance receipt.
        optimize_kw.setdefault("max_its", 500)
        source = data.records() if hasattr(data, "records") and callable(data.records) else data
        rows = _tabular_records(source)
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

        # A prototype DISTRIBUTION carries parameter values the caller chose; honor them as the EM
        # start. optimize() alone coerces a prototype to its estimator (structure only) and then
        # initializes from a fixed-seed random subsample, so the supplied mu/sig2/w were silently
        # ignored. Estimators and spec=None keep the automatic initialization, and an explicit
        # init_estimator= keeps its own initialization role.
        if isinstance(self.spec, SequenceEncodableProbabilityDistribution) and "init_estimator" not in optimize_kw:
            optimize_kw.setdefault("prev_estimate", self.spec)

        self.fitted = optimize(fit_data, self.spec, **optimize_kw)
        self._fit_info = {"n": len(fit_data) if hasattr(fit_data, "__len__") else None, "when": time.time()}

        # Surface the run's own receipt: a fit that stopped on the iteration cap instead of the
        # delta tolerance is an under-converged fit, and silence here is exactly how a 58-nat
        # deficit shipped with empty notes.
        provenance = self.fitted.fit_provenance() if callable(getattr(self.fitted, "fit_provenance", None)) else None
        hit_cap = False
        if provenance is not None:
            self._fit_info["n_iter"] = int(provenance.iterations)
            self._fit_info["converged"] = bool(provenance.converged)
            hit_cap = not provenance.converged and provenance.delta is not None
            if hit_cap:
                self.notes.append(
                    f"EM stopped at the iteration cap ({provenance.iterations} iterations) before "
                    f"reaching its tolerance; the fit may be under-converged -- raise max_its to continue"
                )

        escape_tested = False
        want = 4 if restarts == "auto" else restarts
        is_latent = hasattr(self.fitted, "posterior") and hasattr(self.fitted, "components")
        # fit_data, never data: fit_data IS data when calibrate is off (or too little data to hold
        # anything out) -- the assignment above only diverges when cal_holdout was actually carved out.
        # A restart/saddle-check against the caller's original data would (a) let calibration rows sway
        # the saddle-suspicion verdict and (b) let _refit_symmetry_broken refit on them outright; the
        # replacement model would then get "evaluated held-out" against rows it was just trained on.
        saddle = saddle_suspect(self.fitted, fit_data) if restarts == "auto" else False
        # An unconverged latent fit is suspect for the same reason a saddle is: the returned
        # parameters are wherever the budget ran out, not a chosen optimum. Only latent models get
        # the diversified refit -- restarting a closed-form fit re-runs the identical computation.
        if want and (restarts != "auto" or saddle or (hit_cap and is_latent)):
            better, delta_ll, how = self._refit_symmetry_broken(fit_data, int(want), optimize_kw)
            why = (
                "restarts requested"
                if restarts != "auto"
                else ("saddle suspected" if saddle else "iteration cap reached on a latent fit")
            )
            if better is not None:
                self.fitted = better
                escape_tested = True
                self.notes.append(f"{why}: {how} kept (log-lik +{delta_ll:.3f})")
            elif restarts == "auto":
                self.notes.append(f"{why}: symmetry-broken refits did not improve — inspect the fit")
        # the estimation certificate: which method solved each block, how strong the guarantee, and
        # exactly where (if anywhere) gradient descent was unavoidable. Low-overhead inspection, computed once.
        # A failure here still does not break the fit, but it is RECORDED (see _record_evidence): a
        # certificate that is absent because certification raised is a different fact from one that was
        # never asked for, and both used to look identical (certificate=None, no note).
        try:
            self.certificate = certify(self.fitted, data=fit_data)
        except Exception as exc:  # noqa: BLE001 - certification is a report; never let it break a fit
            self.certificate = None
            self._record_evidence("certificate", "failed", error=exc, n_rows=len(fit_data))
        else:
            self._record_evidence("certificate", "succeeded", n_rows=len(fit_data))
        if cal_holdout:
            from mixle.inference import calibration_report

            try:
                self.calibration = calibration_report(self.fitted, cal_holdout)
            except Exception as exc:  # noqa: BLE001 - a calibration report never breaks a fit
                self.calibration = None
                self._record_evidence("calibration", "failed", error=exc, n_holdout=len(cal_holdout))
            else:
                self._record_evidence("calibration", "succeeded", n_holdout=len(cal_holdout))
        else:
            self._record_evidence(
                "calibration", "not_applicable", n_holdout=0, reason="no calibration holdout requested"
            )
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
            # A frontier is set only by propose(), so the hint names the exact remedy for the model
            # most likely to arrive here unfitted (propose() verifies candidates on a train/holdout
            # split internally but returns the WINNER unfitted unless fit=True was passed).
            hint = (
                " (propose() returned this winner unfitted: call fit(data), or pass fit=True to propose)"
                if self.frontier is not None
                else ""
            )
            raise RuntimeError(f"fit(data) first -- this Model has no fitted distribution yet{hint}")
        return self.fitted

    def __call__(self, x: Any) -> float:
        """The model as a scorer: ``log p(x)`` of one observation under the fitted distribution."""
        return float(self._require_fitted().log_density(x))

    def evaluate(self, data: Any) -> dict[str, Any]:
        """Held-out fit quality: total and mean log-density over ``data``."""
        d = self._require_fitted()
        rows = _tabular_records(data)
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
        """Latent posterior probabilities for one observation (mixtures, HMMs, ...), where supported.

        A component-latent model (a mixture) answers with its own ``posterior``: the ``(k,)``
        component probabilities for ``x``. A chain-latent model (an HMM), whose ``x`` is one
        observation *sequence*, has no ``posterior`` method -- the module docstring's promise used
        to die there as a bare ``AttributeError`` -- so it answers through ``latent_posterior``:
        the ``(T, k)`` forward-backward smoothing marginals, one state distribution per timestep.
        The richer chain-posterior object (Viterbi ``mode()``, FFBS ``sample()``, ``entropy()``)
        stays available as ``model.fitted.latent_posterior(x)``.

        A model with no latent state (a plain Gaussian) raises ``AttributeError`` naming the two
        supported shapes instead of the bare delegation failure.
        """
        d = self._require_fitted()
        if callable(getattr(d, "posterior", None)):
            return d.posterior(x)
        latent = getattr(d, "latent_posterior", None)
        if callable(latent):
            q = latent(x)
            marginals = getattr(q, "marginals", None)
            # Marginals, not the posterior object: posterior() answers with probabilities for every
            # family (arrays in, arrays out), and the full object remains one attribute away.
            return np.asarray(marginals()) if callable(marginals) else q
        raise AttributeError(
            f"posterior() needs a latent-variable model: one with posterior(x) (mixtures and other "
            f"component-latent families) or latent_posterior(x) (HMMs and other chain-latent "
            f"families). {type(d).__name__} has neither -- it models no latent state."
        )

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

    def deploy(self, path: str) -> DeployedArtifact:
        """Persist a durable artifact directory (model + manifest); :meth:`Model.load` restores it.

        A registry-serializable model is written as safe, type-tagged JSON (``model.json``); a model
        the serialization registry cannot represent falls back to a pickle (``model.pkl``).
        **"Pure statistical" does not guarantee "JSON":** besides the expected torch-backed leaves,
        any base-install family the registry has no JSON form for takes the pickle path too (the
        Bernoulli-set, Thurstone, and Spearman ranking families -- models :func:`propose` itself
        selects for ordinary set-valued and ranking data -- deployed that way until their 0.8.0
        codec repair; they now write JSON). Never assume the
        format: read it from the returned path's ``format`` attribute (the return value is a
        :class:`DeployedArtifact`, a plain ``str`` annotated with ``format`` and
        ``format_fallback``), from the warning a fallback raises here, or from the manifest.

        "Cannot represent" is decided by trying it, not by predicting it: the JSON is decoded again
        before the artifact is accepted, so a model that encodes cleanly and then refuses to load --
        the state whole families were in, deployed successfully and readable by nothing -- takes the
        pickle path instead of becoming a write-only artifact. Any such fallback is disclosed three
        ways rather than taken silently: a warning here, ``format_fallback`` in the manifest naming
        the model and the exact read-back error, and a note on the model :meth:`load` returns. It
        costs the reader a ``trust_code=True``, so it should be visible; if that is unacceptable for
        a given family, the fix is a serialization hook on the family, not a quieter deploy.

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
        try:
            out.mkdir(parents=True, exist_ok=True)
        except FileExistsError as exc:
            # exist_ok=True tolerates an existing DIRECTORY only; a plain file at the path used to
            # surface as a bare "[Errno 17] File exists" naming neither the expectation nor a way out.
            raise FileExistsError(
                f"deploy path {str(out)!r} already exists as a plain file; deploy() writes an "
                "artifact DIRECTORY (model file + manifest.json). Point it at a directory path, or "
                "remove the file first."
            ) from exc
        except NotADirectoryError as exc:
            raise NotADirectoryError(
                f"deploy path {str(out)!r} has a plain file where a parent directory is needed "
                f"({exc}); deploy() writes an artifact DIRECTORY. Point it somewhere a directory "
                "can be created."
            ) from exc
        except OSError as exc:
            # Same class re-raised (a PermissionError stays a PermissionError for callers matching
            # on it), with the path and the remedy the bare errno message lacked.
            raise type(exc)(
                f"deploy() could not create the artifact directory {str(out)!r} "
                f"({type(exc).__name__}: {exc}); pass a path where a writable directory can exist."
            ) from exc
        fmt, format_fallback = self._write_model(out, d)
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
        import mixle  # function-scope: lifecycle is imported while mixle/__init__ is still executing

        try:
            from mixle.data.hashing import model_hash

            # A content-level hash of what the model file DESERIALIZES to, distinct from
            # model_sha256 (the file's bytes). load() recomputes it and warns on divergence, so a
            # same-family model file swapped in with a recomputed byte digest no longer serves
            # silently under this manifest's fit record (see load's trust-model note).
            content_hash = model_hash(d)
        except Exception:  # noqa: BLE001 - a model that cannot content-hash still deploys; it just goes uncross-checked
            content_hash = None
        manifest = {
            "family": type(d).__name__,
            "created_at": time.time(),
            # Producer identity: the version whose registry/codecs wrote this artifact. A load-time
            # failure elsewhere can then name what produced the artifact instead of guessing.
            "mixle_version": getattr(mixle, "__version__", "unknown"),
            "fit": self._fit_info,
            "notes": self.notes,
            "format": fmt,
            "format_fallback": format_fallback,
            "model_file": model_file,
            "model_sha256": _sha256_file(out / model_file),
            "model_content_hash": content_hash,
            "evidence_not_exported": evidence_not_exported,
            "mixle_artifact": _ARTIFACT_TAG,
        }
        _write_text_atomically(out / "manifest.json", json.dumps(manifest, indent=2, default=str))
        if format_fallback is not None:
            # Said once, here, while the caller is still at the keyboard and the model is still in
            # memory: this artifact costs its readers something they did not ask for. Discovering
            # that at load time means discovering it in a serving process, on another day, from an
            # error that names no remedy.
            warnings.warn(
                f"Model.deploy({path!r}) wrote a pickle artifact rather than safe JSON because "
                f"{format_fallback}. Model.load() will refuse it unless the caller passes "
                "trust_code=True, since loading a pickle executes arbitrary code from the file. "
                "The reason is recorded in the manifest's 'format_fallback'.",
                stacklevel=2,
            )
        return DeployedArtifact(str(out), format=fmt, format_fallback=format_fallback)

    @staticmethod
    def _write_model(out: Path, d: Any) -> tuple[str, str | None]:
        """Write ``d`` in the first format that actually round-trips. Returns ``(format, fallback_reason)``.

        Safe JSON is preferred and is used only when the JSON *reads back*; otherwise the artifact
        falls back to pickle, and ``fallback_reason`` says in words why -- ``None`` means JSON was
        used and nothing needs disclosing.

        The two candidate formats are tried against different evidence, deliberately:

        * **Encoding.** Only the registry's explicit "this type has no JSON form" answer
          (:class:`SerializationError`) selects the pickle path. A blanket ``except Exception``
          also caught encoder bugs, registry initialization failures and JSON encoding faults, so
          an internal ``TypeError`` silently turned an ordinary JSON-serializable Gaussian into an
          executable ``model.pkl`` -- a programming failure quietly downgrading the artifact's
          security and portability contract. Those errors still propagate.
        * **Decoding.** Encoding successfully proved only that the encoder accepted the object, and
          that is a weaker claim than the one this artifact makes: whole families encoded cleanly
          here and were then refused by :meth:`load`, so ``deploy`` reported success on artifacts
          nothing could read. Whatever the read back raises, it is the same thing :meth:`load` will
          raise later in another process, so any failure disqualifies JSON -- but it is recorded
          verbatim rather than swallowed, which is what separates this from the blanket
          ``except`` above.
        """
        from mixle.utils.serialization import (
            SerializationError,
            ensure_pysp_serialization_registry,
            to_serializable,
        )

        ensure_pysp_serialization_registry()  # a registry that cannot initialize is not "needs pickle"
        try:
            payload = to_serializable(d)
        except SerializationError as exc:  # no JSON form at all (torch leaf, custom object)
            reason = f"the serialization registry has no JSON form for {type(d).__name__} ({exc})"
        else:
            text = json.dumps(payload)
            failure = json_read_back_failure(text)
            if failure is None:
                (out / "model.json").write_text(text)
                return "json", None
            reason = (
                f"{type(d).__name__} encodes to JSON that cannot be read back again ({failure}); "
                "the JSON artifact would have been write-only"
            )
        try:
            with open(out / "model.pkl", "wb") as f:
                pickle.dump(d, f)
        except Exception as exc:
            # Neither format works, so there is no artifact to write and no correct answer to
            # compute. Report both halves: the pickle error alone would send the caller looking in
            # the wrong place for a model whose real problem is the JSON path.
            raise SerializationError(
                f"cannot persist {type(d).__name__}: {reason}; and pickling it failed too "
                f"({type(exc).__name__}: {exc}). This model cannot be deployed as an artifact."
            ) from exc
        return "pickle", reason

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
           ``trust_code=False``, ``None`` and ``0`` are all read as "no trust" (an artifact that needs
           no trust loads under any of them); any other value -- a truthy string, ``1`` -- is rejected
           loudly, because only the exact ``True`` may authorize code execution.

        When the manifest records a ``model_sha256`` (every artifact written by this version does),
        the named model file is hashed and checked against it *before* anything is deserialized, so a
        model file edited, truncated or swapped after deployment -- while the manifest stays intact
        -- is refused rather than loaded. An older manifest without that field is loaded unverified,
        exactly as before -- there is nothing to check it against.

        **What these checks are, and are not.** All of them read the manifest itself as ground
        truth: the digest binds the manifest to the model file's bytes, the ``family`` check binds
        its self-description to what those bytes deserialize to, and the ``model_content_hash``
        cross-check (when recorded) compares the deserialized model's content hash against the one
        stamped at deploy time -- a mismatch there means the manifest's fit record and the model
        being served may not come from the same deployment, and it is disclosed as a warning and a
        note rather than a refusal (a content-hash recomputed by a different mixle version can
        legitimately drift; refusing would brick valid artifacts). None of this *authenticates* the
        manifest: a writer with write access to the artifact directory can rewrite the manifest and
        model together. Integrity against that threat needs a signature kept outside the directory
        (sign the artifact, or compare ``model_sha256`` against a digest recorded elsewhere).

        The restored :class:`Model` carries the fitted distribution, ``notes`` and the fit record.
        The certificate, calibration report and candidate frontier are not in the artifact at all
        (see :meth:`deploy`); when the deploying model had any of them, a note saying so is appended.
        """
        from mixle.utils.serialization import SerializationError, trusted_deserialization

        p = Path(path)
        # Every unreadable-manifest state used to collapse into "is a pickle-format artifact ...
        # Pass trust_code=True": a nonexistent path, a plain file, an empty directory and the
        # directory an interrupted deploy() leaves behind all got a false message whose remedy was
        # to enable arbitrary code execution. Name each actual problem instead.
        if not p.exists():
            raise SerializationError(
                f"no artifact at {path!r}: the path does not exist. Model.load expects the artifact "
                "directory Model.deploy() created."
            )
        if not p.is_dir():
            raise SerializationError(
                f"{path!r} is a file, not an artifact directory. Model.load expects the directory "
                "Model.deploy() created (manifest.json next to the model file)."
            )
        manifest_path = p / "manifest.json"
        if not manifest_path.exists():
            found = [name for name in ("model.json", "model.pkl") if (p / name).exists()]
            if found:
                raise SerializationError(
                    f"{path!r} contains {found[0]} but no manifest.json -- the state an interrupted "
                    "deploy() leaves behind (the manifest is promoted last). Re-run deploy() to "
                    "produce a complete artifact; without the manifest that names and digests the "
                    "model file, it cannot be loaded."
                )
            raise SerializationError(
                f"{path!r} is not a mixle artifact: it contains no manifest.json and no model file. "
                "Point Model.load at the directory Model.deploy() created."
            )
        try:
            manifest_text = manifest_path.read_text()
        except (OSError, UnicodeDecodeError) as exc:
            raise SerializationError(
                f"{path!r}: manifest.json exists but could not be read ({exc}); fix the file or re-deploy the artifact."
            ) from exc
        try:
            manifest = json.loads(manifest_text)
        except ValueError as exc:
            raise SerializationError(
                f"{path!r}: manifest.json is not valid JSON ({exc}); the manifest is corrupt or "
                "truncated. Re-deploy the artifact or restore the manifest from its source."
            ) from exc
        if not isinstance(manifest, dict):
            raise SerializationError(
                f"{path!r}: manifest.json must hold a JSON object describing the artifact, got "
                f"{type(manifest).__name__}. Re-deploy the artifact; this manifest was not written "
                "by Model.deploy()."
            )
        tag = manifest.get("mixle_artifact")
        if tag is not None and tag != _ARTIFACT_TAG:
            # The manifest says, itself, that it follows some other schema (a task/solve artifact,
            # or a later lifecycle schema revision). Reading it as this one would fail somewhere
            # misleading -- or worse, succeed with reinterpreted fields. Manifests without the tag
            # predate it and keep loading as before.
            producer = manifest.get("mixle_version")
            raise SerializationError(
                f"{path!r} declares artifact schema {tag!r}, but this mixle reads {_ARTIFACT_TAG!r}. "
                f"It was written by {'mixle ' + str(producer) if producer else 'a different producer'} "
                "under a different schema; load it with the mixle version that wrote it, or "
                "re-deploy the model with this one."
            )
        fmt = manifest.get("format", "pickle")  # manifests predating the format field described pickle artifacts
        if fmt not in ("json", "pickle"):
            # Every unrecognized format value ("JSON", "json ", "msgpack") used to fall into the
            # pickle branch, whose refusal misdescribed the artifact AND advised trust_code=True --
            # enabling arbitrary-code execution as the remedy for a manifest typo. Name the actual
            # value instead. deploy() only ever writes the two exact strings, so nothing this
            # version produced is refused here.
            raise SerializationError(
                f"{path!r}: the manifest records format {fmt!r}, which this mixle does not read "
                "(it reads 'json' and 'pickle'). The manifest was edited or written by a different "
                "producer; fix its 'format' to name the model file's actual format, or re-deploy."
            )
        model_file = str(manifest.get("model_file") or ("model.json" if fmt == "json" else "model.pkl"))
        if Path(model_file).name != model_file:
            raise SerializationError(f"manifest model_file {model_file!r} must be a plain file name")
        if not (p / model_file).exists():
            raise SerializationError(
                f"{path!r}: the manifest names {model_file} but that file is not in the artifact "
                "directory; the artifact is incomplete -- re-deploy it."
            )
        expected_digest = manifest.get("model_sha256")
        if expected_digest is not None:
            try:
                actual = _sha256_file(p / model_file)
            except OSError as exc:
                raise SerializationError(
                    f"{path!r}: {model_file} could not be read to verify its digest ({exc}); fix the "
                    "file or re-deploy the artifact."
                ) from exc
            if actual != expected_digest:
                raise SerializationError(
                    f"{path!r}: {model_file} does not match the digest recorded in its manifest "
                    f"(expected {expected_digest}, found {actual}); the artifact is incomplete or altered."
                )
        # Truthiness gated both the trusted scope below and the raw pickle.load beneath it, so
        # trust_code="false" -- the string, straight out of a config file or CLI argument -- opened
        # both. Not named by the audit, which cited the same defect in Embedder.load and Registry.get;
        # it is the same gate and is closed the same way (MXR-080-1881).
        # None and a plain integer 0 are unambiguous "no trust" answers (cfg.get("trust_code") is the
        # ordinary way a pipeline says no); rejecting them pushed callers toward True on artifacts
        # that need no trust at all. Only ambiguous/truthy non-True values still get the loud gate.
        says_no = trust_code is False or trust_code is None or (type(trust_code) is int and trust_code == 0)
        if trust_code is not True and not says_no:
            require_explicit_true(
                trust_code,
                "Model.load trust_code",
                because="It authorizes unpickling this artifact, which executes arbitrary code from the file.",
            )
        trusted = trust_code is True
        with trusted_deserialization() if trusted else contextlib.nullcontext():
            if fmt == "json":
                from mixle.utils.serialization import ensure_pysp_serialization_registry, from_serializable

                ensure_pysp_serialization_registry()
                try:
                    payload_text = (p / model_file).read_text()
                except (OSError, UnicodeDecodeError) as exc:
                    raise SerializationError(
                        f"{path!r}: {model_file} could not be read as text ({exc}); the model file is "
                        "corrupt or not the JSON artifact its manifest describes -- re-deploy it."
                    ) from exc
                try:
                    payload = json.loads(payload_text)
                except ValueError as exc:
                    raise SerializationError(
                        f"{path!r}: {model_file} is not valid JSON ({exc}); the model file is corrupt "
                        "or truncated -- re-deploy the artifact."
                    ) from exc
                fitted = from_serializable(payload)
            else:
                if not trusted:
                    raise SerializationError(
                        f"{path!r} is a pickle-format artifact: loading it executes arbitrary code from "
                        "the file. Pass trust_code=True to Model.load() only if you trust its source."
                    )
                try:
                    f = open(p / model_file, "rb")
                except OSError as exc:
                    raise SerializationError(
                        f"{path!r}: {model_file} could not be opened ({exc}); the artifact is "
                        "incomplete -- re-deploy it."
                    ) from exc
                with f:
                    fitted = pickle.load(f)  # noqa: S301 - trust_code=True required by the caller above  # nosec B301 # MXR-080-1881: Model.load runs require_explicit_true(trust_code) above, so only the True singleton opens this path -- a truthy string out of a config file no longer does
        family = manifest.get("family")
        if family is not None and type(fitted).__name__ != str(family):
            # The digest binds the manifest to the model file's BYTES; this binds its self-description
            # to what those bytes deserialize to, so a swapped model.json with a recomputed digest
            # cannot carry another model's provenance.
            raise SerializationError(
                f"{path!r}: the manifest describes a {family} but {model_file} deserialized to "
                f"{type(fitted).__name__}; the manifest and model file do not belong to the same "
                "deployment. Re-deploy so the manifest matches its model."
            )
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
        fallback = manifest.get("format_fallback")
        if fallback:
            # Carries deploy()'s disclosure to whoever ends up holding the model, who is usually
            # not whoever ran deploy() and never saw the warning.
            m.notes.append(f"artifact written as {fmt} rather than safe JSON: {fallback}")
        recorded_content_hash = manifest.get("model_content_hash")
        if recorded_content_hash:
            try:
                from mixle.data.hashing import model_hash

                actual_content_hash = model_hash(fitted)
            except Exception:  # noqa: BLE001 - a model that cannot content-hash goes uncross-checked, like a pre-hash artifact
                actual_content_hash = None
            if actual_content_hash is not None and actual_content_hash != recorded_content_hash:
                # Disclosed, not refused: the byte digest and family already passed, so this state
                # is either a same-family model file swapped in alongside a rewritten manifest, or
                # a content-hash algorithm drift across mixle versions. Only the first is an attack,
                # and load cannot tell them apart -- refusing would brick legitimately migrated
                # artifacts (fail-closed overreach), while silence re-creates the defect this check
                # exists for. The docstring's trust-model note states exactly this.
                note = (
                    f"integrity note: the loaded model's content hash ({actual_content_hash[:16]}...) "
                    f"does not match the manifest's model_content_hash ({str(recorded_content_hash)[:16]}...); "
                    "the manifest's fit record may describe a different model than this artifact "
                    "serves (a swapped model file, or a content-hash change between mixle versions). "
                    "Re-deploy from the original model if in doubt."
                )
                m.notes.append(note)
                warnings.warn(f"Model.load({path!r}): {note}", stacklevel=2)
        return m

    # --- the analysis verbs (delegate to the inference front doors) -------------------------------
    def explain_prediction(self, x: Any):
        """Exact per-part attribution of ``log p(x)`` — :func:`mixle.inference.explain`."""
        from mixle.inference import explain

        return explain(self._require_fitted(), x)

    def forecast(self, history: Any, horizon: int, **kw: Any):
        """Horizon predictions with MODEL-PREDICTIVE Monte Carlo intervals — :func:`mixle.inference.forecast`.

        The intervals are the fitted model's own predictive quantiles, not held-out-calibrated
        bands (STAT-P20-04); ``forecast_price`` is the recalibrated route."""
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


def _dtype_universe_note(rows: list) -> str | None:
    """The candidate-universe disclosure for scalar data: which hypothesis class the element type chose.

    The automatic profiler deliberately branches on element type (an ``int`` column is modeled by
    discrete families, a ``float`` column by continuous densities -- the two universes score in
    incommensurable units and cannot share a leaderboard), but nothing at the propose() level said
    so. Returns ``None`` for non-scalar or mixed-type data (the composite path discloses per field).
    """
    if not rows or isinstance(rows[0], (tuple, list, dict, str, bytes)):
        return None
    if all(isinstance(v, (int, np.integer)) and not isinstance(v, (bool, np.bool_)) for v in rows):
        return (
            "integer dtype: discrete candidate families considered "
            "(cast the column to float to model it as a continuous quantity)"
        )
    if all(isinstance(v, (float, np.floating)) for v in rows):
        return (
            "float dtype: continuous candidate families considered "
            "(an integer column arriving as float is modeled as continuous; keep integer dtype for "
            "discrete families)"
        )
    return None


#: Reject a frontier candidate as a likelihood spike only past BOTH of these (see
#: :func:`_degenerate_likelihood_spike`); values chosen an order of magnitude beyond anything a sane
#: fit was measured to produce, so the guard condemns exactly the collapsed-scale state.
_SPIKE_SPREAD_NATS = 20.0
_PATHOLOGICAL_PIT = 1.0


def _degenerate_likelihood_spike(fitted: Any, val: list, scores: np.ndarray) -> str | None:
    """The reason a candidate's held-out win is a degenerate likelihood spike, or ``None`` when sound.

    A continuous family whose scale collapses onto a repeated value (a near-Dirac fit) *wins* mean
    held-out log-density -- the unbounded density at the atom is exactly what the criterion rewards --
    while sampling a constant and reporting infinite moments. Rejected only when BOTH of these hold,
    so a legitimate fit is never caught:

    * some held-out log-density is positive (pointwise density above one -- possible for a sound fit
      only on very small scales, where it then holds roughly uniformly);
    * the PIT calibration is pathological: total-variation error at least ``_PATHOLOGICAL_PIT`` (of a
      1.8 maximum at ten bins). A model with no scalar predictive CDF is never rejected here -- the
      spike signals alone cannot distinguish a legitimate high-density multivariate fit.

    The max-minus-median spread against ``_SPIKE_SPREAD_NATS`` distinguishes the two collapse shapes
    for the note but no longer GATES the PIT check: with a MAJORITY of held-out rows sitting on the
    atom, the median is pulled up to the spike itself and a spread precondition let the MORE
    degenerate fit through as a verified winner (wave-3 adversarial check). Deterministic (the
    randomized PIT is seeded); the calibration pass runs only on candidates whose pointwise density
    already exceeded one somewhere.
    """
    finite = scores[np.isfinite(scores)]
    if finite.size == 0:
        return None
    top = float(np.max(finite))
    mid = float(np.median(finite))
    if top <= 0.0:
        return None
    try:
        from mixle.inference import calibration_report

        report = calibration_report(fitted, val)
    except Exception:  # noqa: BLE001 - what cannot be measured must not be condemned
        return None
    if report.pit_error is None or report.pit_error < _PATHOLOGICAL_PIT:
        return None
    if top - mid > _SPIKE_SPREAD_NATS:
        return (
            f"held-out log-density spikes to +{top:.1f} (median {mid:.1f}) with pathological PIT "
            f"calibration ({report.pit_error:.2f} of {report.max_pit_error():.1f} max): the fitted scale "
            "collapsed toward a point mass at a repeated value"
        )
    # No spread: the median itself sits at the spike. That is THIS defect class only when a genuine
    # atom carries it -- a single repeated value holding the majority of the held-out rows. Without
    # an atom, a flat high profile with bad PIT is a different pathology (e.g. the variance floor
    # over-widening a legitimate tiny-scale fit, disclosed via numerical_repairs()), and condemning
    # it here would reject sound tiny-scale data (found while validating the reorder on
    # normal(0, 1e-6) draws: floored sigma2=1e-8 gives PIT 1.60 with zero repeated values).
    try:
        values, counts = np.unique(np.asarray(val, dtype=float), return_counts=True)
    except (TypeError, ValueError):
        return None  # non-scalar rows: no atom evidence obtainable here
    if counts.size == 0 or counts.max() <= 0.5 * len(val):
        return None
    atom = float(values[int(np.argmax(counts))])
    return (
        f"held-out log-density is positive across the atom (max +{top:.1f}, median {mid:.1f}) with "
        f"pathological PIT calibration ({report.pit_error:.2f} of {report.max_pit_error():.1f} max): "
        f"the fitted scale collapsed toward a point mass on the repeated value {atom:g} carried by a "
        "MAJORITY of held-out rows"
    )


def _top_level_field_paths(rec: Any) -> list[str]:
    """One path label per TOP-LEVEL field of ``rec.estimator``, matching the granularity of a
    composite's ``dists`` -- NOT ``rec.fields``, which is LEAF-level and gives a sequence/composite
    field (one top-level column) multiple entries (e.g. "$[0]['element']", "$[0]['length']" for one
    sequence column).

    ``_unseen_label_rescue`` names the excluded field by indexing ``field_paths`` with a top-level
    ``dists`` index. Handing it ``rec.fields``' flat leaf list works only by coincidence, when every
    field up to the culprit happens to own exactly one leaf; the moment an earlier column expands to
    more than one leaf (any sequence/composite field), every later top-level index reads the wrong
    leaf's name (T2-01, third occurrence). ``rec.profile.fields`` carries the same entries as
    ``rec.fields``, index-aligned, but with the RAW tuple path (e.g. ``(0, "length")``) instead of
    the pre-formatted string -- grouping those by the path prefix that ``DatumNode.get_estimator``
    adds per top-level child (one positional element for tuple/list rows, ``("key", k)`` for dict
    rows; see ``_extract_field_series``/``get_estimator`` in mixle/utils/automatic/profiling.py)
    recovers exactly the one-entry-per-top-level-child list ``dists`` needs.
    """
    from mixle.utils.automatic.profiling import format_path

    fields = rec.fields
    profile = getattr(rec, "profile", None)
    raw_paths = [getattr(f, "path", None) for f in profile.fields] if profile is not None else None
    if raw_paths is None or len(raw_paths) != len(fields) or any(p is None for p in raw_paths):
        return [c.path for c in fields]  # can't safely group -- fall back to the flat leaf list
    groups: list[str] = []
    prev_key: tuple[Any, ...] | None = None
    for raw in raw_paths:
        key = tuple(raw[:2]) if tuple(raw[:1]) == ("key",) else tuple(raw[:1])
        if key != prev_key:
            groups.append(format_path(key))
            prev_key = key
    return groups


def _unseen_label_rescue(
    fitted: Any, enc: Any, scores: np.ndarray, field_paths: list[str]
) -> tuple[np.ndarray, str] | None:
    """Recover a field-decomposable candidate whose non-finite held-out score is fully explained by
    CategoricalDistribution's documented ``default_value=0.0`` -inf-on-unseen-label behavior.

    Without this, propose()'s scoring loop treats ANY non-finite entry anywhere in a candidate's
    per-row score array as fatal for the WHOLE candidate -- so one identifier-like/high-cardinality
    field (which legitimately meets held-out labels its training split never saw; see
    ``CategoricalDistribution.__init__``'s ``default_value`` docstring) silently voids verification
    for every OTHER, well-behaved field in the same joint model, collapsing the whole candidate to
    "failed" and eventually the whole frontier to the unverified fallback.

    When every field responsible for a non-finite row score is provably this documented, benign
    cause -- not some other numerical problem this check would otherwise be catching -- this drops
    just those fields' contribution from the AGGREGATE held-out score and returns the rest, so the
    candidate can still be verified and ranked on the fields that DO score finitely. Returns
    ``None`` (keep failing the candidate, as before) whenever the candidate isn't a
    field-decomposable composite, or the non-finite scores are not fully explained by this specific
    cause -- an unexplained non-finite score must still fail the candidate outright, not be papered
    over.
    """
    from mixle.stats.combinator.optional import OptionalDistribution
    from mixle.stats.univariate.discrete.categorical import CategoricalDistribution

    dists = getattr(fitted, "dists", None)
    n = scores.shape[0]
    if not dists or not isinstance(enc, (tuple, list)) or len(enc) != len(dists):
        return None  # not a per-field composite (or arity mismatch) -- nothing to decompose
    field_scores: list[np.ndarray] = []
    for i, d in enumerate(dists):
        try:
            fs = np.asarray(d.seq_log_density(enc[i]), dtype=np.float64)
        except Exception:  # noqa: BLE001 - can't decompose this field; the whole-candidate check still applies
            return None
        if fs.shape != (n,):
            return None
        field_scores.append(fs)
    bad = [i for i, fs in enumerate(field_scores) if not np.isfinite(fs).all()]
    if not bad:
        return None  # the aggregate's non-finiteness isn't localized to any one field -- unexplained
    for i in bad:
        leaf = dists[i]
        seen = 0
        while hasattr(leaf, "dist") and seen < 8:  # unwrap OptionalDistribution/IgnoredDistribution wrappers
            if isinstance(leaf, OptionalDistribution) and not (np.isfinite(leaf.log_p) and np.isfinite(leaf.log_pn)):
                # This wrapper's OWN score is -inf whenever a held-out row is missing and p==0 (or
                # present and p==1) -- indistinguishable, from the aggregate score alone, from the
                # leaf's unseen-label -inf this rescue exists to explain. An adversarial review
                # caught this: a field fit with p==0 (no missing rows in the training split) that
                # meets a genuinely MISSING held-out value scores -inf from THIS wrapper, not from
                # any unseen categorical label, and the field is silently (and wrongly) excused as
                # "unseen in training" -- masking a real missing-value generalization failure. Only
                # when 0 < p < 1 can this wrapper never itself produce -inf, making any -inf here
                # unambiguously the child's; otherwise refuse, matching the function's own rule that
                # an unexplained non-finite score must fail the candidate outright.
                return None
            leaf = leaf.dist
            seen += 1
        if not isinstance(leaf, CategoricalDistribution) or leaf.default_value != 0.0:
            return None  # not the documented unseen-label case -- don't paper over an unexplained failure
    reduced = np.zeros(n, dtype=np.float64)
    for i, fs in enumerate(field_scores):
        if i not in bad:
            reduced = reduced + fs
    if not np.isfinite(reduced).all():
        return None  # excluding the bad fields didn't fix it -- some other field is also non-finite
    detail = "; ".join(
        f"{field_paths[i] if i < len(field_paths) else f'field {i}'} "
        f"({int(np.sum(~np.isfinite(field_scores[i])))}/{n} held-out rows carry a label unseen in training)"
        for i in bad
    )
    note = (
        f"excluded from this candidate's held-out verification score: {detail} -- "
        "CategoricalDistribution's documented default_value=0.0 scores an unseen label at -inf; the "
        "field is still part of the fitted model, it just doesn't count toward this candidate's "
        "verified held-out density"
    )
    return reduced, note


def _refresh_frozen_identifier_leaves(estimator: Any, rows: list, field_paths: list[str]) -> Any:
    """Rebuild a winning candidate's frozen (Ignored) leaves against ALL of ``rows``, not just train.

    ``_get_identifier_estimator`` (mixle/utils/automatic/factories.py) freezes an identifier-like,
    unrecognized-scalar-type, or ambiguous bool/numeric field to the empirical categorical observed
    when the field was profiled -- deliberately finite on every value that profiling pass saw, -inf
    on anything it didn't (the same finite-support convention every automatically fitted categorical
    column gets; see that function's docstring). propose()'s candidates are profiled and fitted on
    the TRAINING split only (STAT-RR18-01), so a frozen leaf's support only ever covers train's
    values -- and the winning candidate's SAME estimator object is then reused, unchanged, for the
    final full-data refit (``fit=True``). A fully-unique-per-row identifier column guarantees every
    held-out row's value is absent from that frozen support, and ``IgnoredEstimator``'s accumulator
    never re-estimates its child, so the refit's initial model -- and every later iteration, since
    nothing about this field ever changes -- scores those rows at -inf forever. ``_em_loop`` then
    raises "EM did not produce a finite objective from its non-finite initial model" on data the
    frontier just finished verifying.

    Rebuilding each frozen leaf's support from ``rows`` (train UNION held-out, exactly what the final
    refit is about to run on) fixes this without touching the leaf's finite-support semantics or
    reopening the STAT-RR18-01 leak: unlike a family/structure choice, a frozen leaf's support never
    participates in candidate ranking (T2-01's ``_unseen_label_rescue`` already excludes it from
    every candidate's held-out score whenever it is the sole source of non-finiteness there), so
    widening it only after the winner is already chosen leaks nothing back into that choice.

    Scoped to the shapes this can be done safely for: a flat, per-field ``CompositeEstimator`` whose
    child count matches ``field_paths`` (so position ``i`` unambiguously means ``rows[*][i]``), or its
    dict-keyed equivalent, a flat ``RecordEstimator`` whose child count matches ``field_paths`` (so
    child ``i`` unambiguously means ``rows[*][estimator.sources[i]]`` -- propose() builds a
    RecordEstimator, not a CompositeEstimator, for dict-shaped rows; T3-01 caught this function
    silently no-op'ing for that shape via the ``isinstance(estimator, CompositeEstimator)`` guard,
    reproducing the exact crash this function exists to prevent whenever the winning candidate came
    from dict rows). Anything else -- including the "structured"/``None`` candidate, which already
    re-profiles fresh against the full ``rows`` at fit time and never carries a stale leaf -- is
    returned unchanged, matching this module's existing rule that an unexplained shape must not be
    papered over.
    """
    from mixle.stats.combinator.composite import CompositeEstimator
    from mixle.stats.combinator.ignored import IgnoredEstimator
    from mixle.stats.combinator.record import RecordEstimator
    from mixle.stats.univariate.discrete.categorical import CategoricalDistribution
    from mixle.utils.automatic.factories import _get_identifier_estimator

    if isinstance(estimator, CompositeEstimator):
        children = estimator.estimators
        if len(children) != len(field_paths):
            return estimator

        def _value(row: Any, i: int) -> Any:
            return row[i]

        def _rebuild(refreshed: list[Any]) -> Any:
            return CompositeEstimator(refreshed, keys=estimator.keys)

    elif isinstance(estimator, RecordEstimator):
        children = estimator.estimators
        if len(children) != len(field_paths):
            return estimator
        sources = estimator.sources

        def _value(row: Any, i: int) -> Any:
            return row[sources[i]]

        def _rebuild(refreshed: list[Any]) -> Any:
            return RecordEstimator(tuple(zip(estimator.fields, estimator.sources)), refreshed)

    else:
        return estimator

    try:
        refreshed = list(children)
        changed = False
        for i, child in enumerate(refreshed):
            if not (isinstance(child, IgnoredEstimator) and isinstance(child.dist, CategoricalDistribution)):
                continue
            vdict: dict[Any, float] = {}
            for row in rows:
                value = _value(row, i)
                vdict[value] = vdict.get(value, 0.0) + 1.0
            if not vdict:
                continue
            refreshed[i] = _get_identifier_estimator(vdict)
            changed = True
    except Exception:  # noqa: BLE001 - a shape this can't safely widen is left exactly as it was
        return estimator
    return _rebuild(refreshed) if changed else estimator


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

    **The winner comes back UNFITTED unless ``fit=True`` is passed.** Candidate ranking fits every
    candidate internally (on the training split, scored held-out), but what returns by default is
    the winning *specification*: ``Model.fitted`` is ``None`` and the scoring verbs raise until
    ``fit(data)`` runs. ``propose(data, fit=True)`` is the one-call propose-and-fit form.

    ``data`` is anything row-shaped: a list of records, a numpy array, a pandas ``DataFrame`` (one
    record per row across its columns -- never a model of the column names), a mapping of equal-length
    columns, a ``DataSource``, or a one-shot iterator (materialized once).

    Two paths return an *unverified* winner, and both are honestly disclosed (never silently).
    ``max_candidates=0`` or ``timeout=0.0`` skips every candidate: the winner falls back to the raw
    heuristic recommendation, with a ``"search budget: skipped ..."`` line in ``Model.notes`` and a
    ``"skipped": "search budget reached (...)"`` entry per candidate in ``Model.frontier``, both
    readable through ``explain()``. And when every candidate *fails* to verify (scoring errors, or
    every fit rejected as degenerate), the same fallback applies with a ``"no candidate could be
    verified: ..."`` note naming each failure. On BOTH routes, with ``fit=True`` the
    ``evidence["certificate"]`` record is downgraded from ``succeeded`` to ``attempted`` -- a
    certificate produced for a winner nothing verified out-of-sample is not verification, whether
    the candidates failed or the budget skipped them. Any other ``max_candidates``/``timeout``
    still verifies every candidate it has budget for before choosing a winner.

    The verified frontier is guarded against degenerate wins: a candidate whose held-out score is a
    likelihood spike -- a continuous fit whose scale collapsed onto a repeated value, detected by a
    positive log-density spike (max over held-out rows above 0, whether localized or median-wide, per
    the median) *combined with* pathological PIT calibration (total-variation error >= 1.0 of the 1.8
    maximum) -- is rejected from the frontier with a note, and the win falls to the best
    non-degenerate candidate. The criterion is deterministic and never fires on a model without a
    scalar predictive CDF (see ``_degenerate_likelihood_spike``).

    A joint (multi-field) candidate's held-out score is also guarded against one field wrongly
    voiding every other, well-behaved field: when a field legitimately meets a held-out label its
    training split never saw, ``CategoricalDistribution``'s documented ``default_value=0.0`` scores
    that row at ``-inf`` (an identifier-like/high-cardinality field with a long tail is the common
    real-world trigger). That field's contribution is excluded from the candidate's held-out score
    -- not the whole candidate -- with a note naming the field and how many held-out rows it
    affected (see ``_unseen_label_rescue``); an unexplained non-finite score still fails the
    candidate outright.

    For scalar data the candidate universe follows the ELEMENT TYPE, and the choice is recorded in
    ``Model.notes``: integer-typed values are modeled by discrete families (categorical / Poisson /
    binomial ...), float-typed values by continuous densities -- the two universes score in different
    units (log-mass vs log-density) and cannot be ranked against each other, so passing
    ``.astype(float)`` on a count column is an affirmative statement that it is continuous.

    Candidates come from every proposer the library has — the heuristic recommendation
    (:func:`mixle.task.recommend.recommend_model`, dependency-aware in the narrow sense of a joint
    multivariate-Gaussian candidate for fully-observed numeric vector rows), a structured-search candidate
    (:func:`mixle.inference.optimize` called with no pre-built estimator — its own no-estimator
    auto-structure-search, the fuller copula/learned-Bayesian-network dependence upgrade described in
    :doc:`/automatic-modeling-contract`'s "Dependence between fields"), the plain independence baseline
    (:func:`mixle.utils.automatic.get_estimator`), and, when an ``llm`` handle is given, an LLM-designed
    structure (:func:`mixle.task.design.design_model`, allowlisted-spec, fit-validated). Each candidate is
    **generated from the training split only, fitted on it, and scored on held-out data** -- the
    outer holdout is invisible to every proposer until final ranking (STAT-RR18-01), so the
    ranking is out-of-sample, not a guess.
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

    rows = _tabular_records(data)
    if len(rows) < 3:
        raise ValueError("propose requires at least three records for a non-empty train/holdout split")
    # STAT-RR18-01: the outer split happens BEFORE any candidate generation, and every proposer
    # sees TRAINING rows only. Letting recommend_model/get_estimator/design_model inspect all
    # rows meant holdout-informed model-FAMILY selection before candidates were ranked on that
    # same holdout (measured: with 25 adverse holdout rows visible, full-data recommendation
    # chose Laplace where train-only chose HalfNormal) -- fitting on train afterward does not
    # undo a family choice the holdout already steered.
    rng = np.random.RandomState(seed)
    order = rng.permutation(len(rows))
    n_val = max(2, int(round(len(rows) * holdout)))
    val = [rows[i] for i in order[:n_val]]
    train = [rows[i] for i in order[n_val:]]
    if not train:
        raise ValueError("holdout leaves no training rows; lower holdout or provide more records")
    rec = recommend_model(train, **recommend_kw)
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

        indep = get_estimator(train)
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

        designed = design_model(train, llm)
        if designed.source == "llm":
            candidates.append(("llm-designed", designed.estimator))

    frontier: list[dict[str, Any]] = []
    evaluated = 0
    budget_start = time.monotonic()
    top_level_field_paths = _top_level_field_paths(rec)  # one per composite's `dists` index -- see docstring
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
            rescue_note = None
            if not np.isfinite(scores).all():
                rescued = _unseen_label_rescue(fitted, enc, scores, top_level_field_paths)
                if rescued is None:
                    raise ValueError("candidate scorer returned a non-finite held-out log density")
                scores, rescue_note = rescued
            degenerate = _degenerate_likelihood_spike(fitted, val, scores)
            if degenerate is not None:
                # A likelihood spike would WIN the mean-log-density ranking; record it as a failed
                # candidate (same disclosure plumbing as a scoring error) so the win falls to the
                # best non-degenerate candidate instead.
                frontier.append({"name": name, "estimator": est, "error": f"degenerate fit rejected: {degenerate}"})
                evaluated += 1
                continue
            score = float(np.mean(scores))
            entry = {
                "name": name,
                "estimator": est,
                "heldout_mean_log_density": score,
                "candidate_index": candidate_index,
            }
            if rescue_note is not None:
                entry["partial_verification"] = rescue_note
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
    # Every candidate FAILED (scoring error or degenerate rejection): the winner below is the raw
    # heuristic recommendation, which nothing verified out-of-sample. The budget-skip path already
    # discloses its fallback; this path gets the same honesty (a roll-up note here, and the
    # certificate downgrade after the fit).
    unverified_fallback = not scored and any("error" in f for f in frontier)

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
    universe_note = _dtype_universe_note(rows)
    if universe_note is not None:
        notes.append(universe_note)

    def _candidate_note(f: dict[str, Any]) -> str:
        if "skipped" in f:
            return f"candidate {f['name']}: {f['skipped']}"
        if "error" in f:
            return f"candidate {f['name']}: failed ({f['error']})"
        base = f"candidate {f['name']}: held-out mean log-density {f['heldout_mean_log_density']:.3f}"
        if "partial_verification" in f:
            base += f" [{f['partial_verification']}]"
        return base

    notes += [_candidate_note(f) for f in frontier]
    if skipped_names:
        notes.append(f"search budget: skipped {len(skipped_names)} candidate(s) unevaluated: {skipped_names}")
    if unverified_fallback:
        reasons = "; ".join(f"{f['name']}: {f['error']}" for f in frontier if "error" in f)
        notes.append(
            f"no candidate could be verified: {reasons}; returning the unverified heuristic "
            f"recommendation ({type(winner).__name__})"
        )
    m = Model(winner, notes=notes)
    m.frontier = frontier
    if fit:
        m.spec = _refresh_frozen_identifier_leaves(m.spec, rows, top_level_field_paths)
        m.fit(rows, restarts=None, max_its=int(max_its), rng=np.random.RandomState(int(seed)))
        nothing_verified = unverified_fallback or (
            skipped_names and not any("heldout_mean_log_density" in f for f in frontier)
        )
        if nothing_verified and m.evidence.get("certificate", {}).get("status") == "succeeded":
            # certify() ran, but a certificate produced for a winner nothing verified out-of-sample
            # must not read 'succeeded' -- 'attempted' is the honest status the evidence schema
            # already supports (see _record_evidence). The same rule covers BOTH documented
            # unverified routes: candidates all failed/rejected, and the search budget skipping
            # every candidate (max_candidates=0 / timeout=0.0) -- the wave-3 check found the two
            # identically-unverified winners carrying different certificate statuses.
            reason = (
                "no candidate could be verified on held-out data; the winner is the unverified heuristic recommendation"
                if unverified_fallback
                else "the search budget skipped every candidate unevaluated; the winner is the "
                "unverified heuristic recommendation"
            )
            m._record_evidence(
                "certificate",
                "attempted",
                n_rows=m.evidence["certificate"].get("n_rows"),
                reason=reason,
            )
        if m.spec is None and m.fitted is not None and callable(getattr(m.fitted, "estimator", None)):
            # The structured candidate's estimator slot is None ("infer at fit time"), which made the
            # returned proposal non-reusable: Model(m.spec).fit(other_data) silently re-inferred a
            # different family. Carry the winning family's estimator so the proposal round-trips.
            try:
                m.spec = m.fitted.estimator()
            except Exception as exc:  # noqa: BLE001 - a proposal without a reusable spec is still a valid fit
                m.notes.append(
                    f"winning spec could not be reconstructed for reuse "
                    f"({type(exc).__name__}: {exc}); Model.spec stays None"
                )
        try:
            from mixle.data.hashing import model_hash

            m._fit_info["artifact_hash"] = model_hash(m.fitted)
        except Exception:  # noqa: BLE001 - artifact hashing must not discard a valid final fit
            m._fit_info["artifact_hash"] = None
    return m
