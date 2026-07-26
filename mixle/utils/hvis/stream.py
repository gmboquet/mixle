"""Streaming HViS: place arriving points into a frozen model-based embedding, honestly.

Classic t-SNE/UMAP cannot stream: they are transductive (no out-of-sample map), their input
probabilities are normalized over the whole dataset, and re-running on n+1 points produces an
arbitrarily rotated/rearranged layout -- the picture "jumps" and a monitoring view is useless.

HViS escapes all three because its affinities come from a FITTED MODEL, not from raw pairwise
distances: every observation has a fixed encoder (``x -> posterior / field evidence``), so a new
point's affinity row against a frozen set of landmarks is self-contained -- no other stream data
required. That turns streaming into three separable, individually-checkable pieces:

1. **Atlas** -- embed a landmark reservoir once (:func:`~mixle.utils.hvis.embed.htsne`, or any
   coordinates you supply, e.g. a ``humap`` layout). The atlas is FROZEN: existing coordinates never
   move while streaming, so the view is stable by construction rather than by hope.
2. **Placement** -- each arriving point gets its perplexity-calibrated affinity row over the
   landmarks only (reusing the exact factor/calibration machinery of the batch path), then minimizes
   its OWN row-KL against the frozen atlas under the same heavy-tailed kernel. One moving point per
   objective (the out-of-sample "transform" trick), O(landmarks) per point, vectorized across the
   whole batch since placed points do not interact.
3. **Drift** -- placement is only trustworthy while the model still fits the stream. The model gives
   a free, principled drift signal: the mean log-density of arrivals versus the landmark reference.
   When it trips, :meth:`StreamingHvis.refresh` re-embeds warm-started from the current coordinates
   and Procrustes-aligns the result back onto the old atlas -- and REPORTS the alignment residual,
   so a genuine geometry change is surfaced instead of being animated away.

With an ``estimator``, the MODEL streams too, closing the loop with mixle's native
sufficient-statistic machinery: each arriving batch is E-stepped once at arrival time under the
then-current model (``accumulator.seq_update``), and :meth:`StreamingHvis.refresh` performs the
M-step over reservoir + accumulated stream statistics before re-embedding -- incremental EM
(Neal & Hinton 1998), one honest sweep per refresh, never a silent full re-fit.

The one thing this deliberately does not promise: a placement is a projection into the atlas's
geometry. A point unlike anything in the reservoir gets a low-affinity row (and drags the drift
score down) rather than a secretly-wrong confident position -- check :meth:`drift_score` before
trusting a batch of placements.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from mixle.utils.hvis.affinity import (
    _affinity_factors,
    _calibrate_row,
    _posteriors_and_loglikes,
    _resolve_affinity,
    log_affinity_block,
)
from mixle.utils.vector import ImpossibleEvidenceError

__all__ = ["StreamingHvis", "place_in_atlas"]


def _cross_log_affinity(factors, row_idx: np.ndarray, col_idx: np.ndarray, evidence_cap: float | None) -> np.ndarray:
    """Rectangular log-affinity block; the shared implementation lives in
    :func:`mixle.utils.hvis.affinity.log_affinity_block` (also used by ``affinity_health``)."""
    return log_affinity_block(factors, row_idx, col_idx, evidence_cap)


def _row_probabilities(log_aff: np.ndarray, perplexity: float | None) -> np.ndarray:
    """Per-row conditional probabilities over the landmark columns (calibrated when perplexity set).

    Row calibration only ever needs the row itself -- this is precisely why out-of-sample placement
    is well-posed here while global t-SNE symmetrization is not.
    """
    log_aff = np.asarray(log_aff, dtype=np.float64)
    if log_aff.ndim != 2 or log_aff.shape[0] == 0 or log_aff.shape[1] < 2:
        raise ValueError("log_aff must be a non-empty matrix with at least two landmarks.")
    if np.any(np.isnan(log_aff)) or np.any(np.isposinf(log_aff)):
        raise ValueError("log_aff must contain only finite values or -inf.")
    b, n_landmarks = log_aff.shape
    if perplexity is not None:
        try:
            perplexity = float(perplexity)
        except (TypeError, ValueError) as exc:
            raise TypeError("perplexity must be a finite real number or None.") from exc
        if not np.isfinite(perplexity) or not 1.0 <= perplexity <= n_landmarks:
            raise ValueError(f"perplexity must be between 1 and {n_landmarks}.")
    p = np.zeros_like(log_aff)
    target = None if perplexity is None else np.log(perplexity)
    for i in range(b):
        row = log_aff[i]
        finite = np.isfinite(row)
        if not np.any(finite):
            raise ImpossibleEvidenceError(f"the model provides no landmark evidence for placement row {i}.")
        if perplexity is not None and perplexity > int(finite.sum()):
            raise ImpossibleEvidenceError(
                f"placement row {i} has only {int(finite.sum())} possible landmarks, "
                f"fewer than perplexity={perplexity}."
            )
        if target is None:
            shifted = row[finite] - row[finite].max()
            q = np.exp(shifted)
            p[i, finite] = q / q.sum()
        else:
            p[i, finite] = _calibrate_row(row[finite].copy(), target)
    return p


def place_in_atlas(
    p_rows: np.ndarray,
    atlas: np.ndarray,
    *,
    alpha: float = 1.0,
    max_its: int = 250,
    eta: float | None = None,
    momentum: float = 0.8,
    tol: float = 1.0e-7,
) -> np.ndarray:
    """Place each row's point into a FROZEN atlas by minimizing its own row-KL under the t-kernel.

    ``p_rows`` is ``(B, L)`` row-stochastic (each arriving point's calibrated affinities over the
    ``L`` landmarks); ``atlas`` is ``(L, d)``. Each point's objective involves only itself and the
    frozen landmarks, so the whole batch optimizes as one vectorized gradient descent. Initialized
    at the affinity-weighted barycenter of landmark coordinates.
    """
    p_rows = np.asarray(p_rows, dtype=np.float64)
    atlas = np.asarray(atlas, dtype=np.float64)
    if p_rows.ndim != 2 or p_rows.shape[0] == 0:
        raise ValueError("p_rows must be a non-empty two-dimensional probability matrix.")
    if atlas.ndim != 2 or atlas.shape[0] < 2 or atlas.shape[1] == 0:
        raise ValueError("atlas must be a two-dimensional coordinate matrix with at least two landmarks.")
    if p_rows.shape[1] != atlas.shape[0]:
        raise ValueError("p_rows columns must align with atlas landmarks.")
    if not np.all(np.isfinite(p_rows)) or np.any(p_rows < 0.0):
        raise ValueError("p_rows must contain only finite non-negative probabilities.")
    if not np.allclose(p_rows.sum(axis=1), 1.0, rtol=1.0e-7, atol=1.0e-10):
        raise ValueError("p_rows must be row-stochastic.")
    if not np.all(np.isfinite(atlas)):
        raise ValueError("atlas must contain only finite coordinates.")
    if not np.isfinite(alpha) or alpha <= 0.0:
        raise ValueError("alpha must be finite and positive.")
    if isinstance(max_its, bool) or not isinstance(max_its, (int, np.integer)) or max_its <= 0:
        raise ValueError("max_its must be a positive integer.")
    if eta is not None and (not np.isfinite(eta) or eta <= 0.0):
        raise ValueError("eta must be finite and positive or None.")
    if not np.isfinite(momentum) or not 0.0 <= momentum < 1.0:
        raise ValueError("momentum must be finite and in [0, 1).")
    if not np.isfinite(tol) or tol < 0.0:
        raise ValueError("tol must be finite and non-negative.")
    y = p_rows @ atlas  # barycentric init: already in the right neighborhood for sharp rows
    if eta is None:
        spread = float(atlas.std())
        eta = 0.5 * (spread if spread > 0 else 1.0)
    velocity = np.zeros_like(y)
    c = (alpha + 1.0) / alpha

    for _ in range(int(max_its)):
        d2 = np.square(y[:, None, :] - atlas[None, :, :]).sum(axis=2)  # (B, L)
        u = 1.0 / (1.0 + d2 / alpha)
        w = u ** ((alpha + 1.0) / 2.0)
        q = w / np.maximum(w.sum(axis=1, keepdims=True), 1.0e-300)
        coeff = (p_rows - q) * u  # (B, L)
        grad = c * (y * coeff.sum(axis=1, keepdims=True) - coeff @ atlas)
        if not np.all(np.isfinite(grad)):
            raise FloatingPointError("atlas placement gradient became non-finite.")
        velocity = momentum * velocity - eta * grad
        y = y + velocity
        if float(np.abs(grad).max()) < tol:
            break
    if not np.all(np.isfinite(y)):
        raise FloatingPointError("atlas placement produced non-finite coordinates.")
    return y


def _procrustes_align(source: np.ndarray, target: np.ndarray) -> tuple[np.ndarray, float, float]:
    """Align ``source`` onto ``target`` by rotation/reflection + translation + UNIFORM scale (full
    Procrustes). A t-SNE layout expands globally as it converges, so isotropic scale is nuisance for
    the continuity question; what the residual measures is SHAPE change. The scale factor is returned
    (and reported) rather than hidden -- a large one says the layout re-expanded even if the shape
    held. Returns ``(aligned, rms_residual, scale)``."""
    from scipy.linalg import orthogonal_procrustes

    source = np.asarray(source, dtype=np.float64)
    target = np.asarray(target, dtype=np.float64)
    if source.ndim != 2 or source.shape != target.shape or source.shape[0] < 2 or source.shape[1] == 0:
        raise ValueError("source and target must be aligned coordinate matrices with at least two rows.")
    if not np.all(np.isfinite(source)) or not np.all(np.isfinite(target)):
        raise ValueError("Procrustes coordinates must contain only finite values.")
    mu_s, mu_t = source.mean(axis=0), target.mean(axis=0)
    src, tgt = source - mu_s, target - mu_t
    rotation, trace = orthogonal_procrustes(src, tgt)
    denom = float(np.sum(src * src))
    scale = float(trace) / denom if denom > 0 else 1.0
    aligned = scale * (src @ rotation) + mu_t
    residual = float(np.sqrt(np.mean(np.sum((aligned - target) ** 2, axis=1))))
    if not np.all(np.isfinite(aligned)) or not np.isfinite(residual) or not np.isfinite(scale):
        raise FloatingPointError("Procrustes alignment produced a non-finite result.")
    return aligned, residual, scale


def _mean_log_density(mix_model, data) -> float:
    """Mean per-observation log-density under the mixture -- the drift reference/score input."""
    data = list(data)
    if not data:
        raise ValueError("log-density monitoring requires at least one observation.")
    if hasattr(mix_model, "dist_to_encoder") and hasattr(mix_model, "seq_log_density"):
        enc = mix_model.dist_to_encoder().seq_encode(data)
        value = float(np.mean(np.asarray(mix_model.seq_log_density(enc), dtype=np.float64)))
    else:
        _, ll_mat = _posteriors_and_loglikes(mix_model, data=data)
        log_w = np.asarray(mix_model.log_w, dtype=np.float64).reshape(1, -1)
        joint = ll_mat + log_w
        mx = joint.max(axis=1, keepdims=True)
        value = float(np.mean(mx[:, 0] + np.log(np.exp(joint - mx).sum(axis=1))))
    if not np.isfinite(value):
        raise ImpossibleEvidenceError("the model assigns non-finite mean log density to the observations.")
    return value


class StreamingHvis:
    """A frozen model-based atlas that arriving points are placed into, with drift accounting.

    Args:
        mix_model: the fitted mixture the affinities come from (any model ``htsne`` accepts).
        landmark_data: the reservoir the atlas is built over. The model's components make this easy
            to keep representative -- e.g. sample a quota per component.
        atlas: optional precomputed ``(len(landmark_data), emb_dim)`` coordinates (e.g. a ``humap``
            layout). When omitted, the atlas is built here with :func:`htsne`.
        affinity: any named HViS affinity. Note ``'local'`` learns component-local metrics from the
            data the factors are built over, which during streaming is ``landmarks + batch`` -- with
            a reasonably sized reservoir the landmarks dominate, but ``'balanced'`` (the default) is
            a pure per-point function of the model and has no such coupling.
        estimator: optional ``ParameterEstimator`` consistent with ``mix_model``. When given, the
            MODEL streams too, by incremental EM (Neal & Hinton 1998): every :meth:`add` batch is
            E-stepped once, at arrival time, under the model current at that moment, and its
            sufficient statistics accumulate; :meth:`refresh` then performs one M-step over the
            reservoir's statistics (E-stepped under the current model) combined with the accumulated
            stream statistics, adopting the re-estimated model before re-embedding. One honest EM
            sweep per refresh -- NOT full-batch EM to convergence; pass ``refresh(mix_model=...)``
            with your own fully re-fit model when that is what you want (an explicit model always
            wins, and discards the pending stream statistics).
        drift_threshold_nats: how far (in nats) the arrivals' mean log-density may fall below the
            landmark reference before :attr:`drifted` trips.
        htsne_kwargs: forwarded to :func:`htsne` for atlas builds and :meth:`refresh`.
    """

    def __init__(
        self,
        mix_model: Any,
        landmark_data: list,
        *,
        atlas: np.ndarray | None = None,
        emb_dim: int = 2,
        alpha: float = 1.0,
        perplexity: float | None = 30.0,
        affinity: str = "balanced",
        evidence_cap: float | None = 1.0,
        field_weights=None,
        estimator: Any = None,
        drift_threshold_nats: float = 2.0,
        seed: int | None = None,
        **htsne_kwargs: Any,
    ) -> None:
        if mix_model is None:
            raise ValueError("mix_model is required.")
        self.mix_model = mix_model
        self.landmark_data = list(landmark_data)
        if len(self.landmark_data) < 2:
            raise ValueError("landmark_data must contain at least two observations.")
        if isinstance(emb_dim, bool) or not isinstance(emb_dim, (int, np.integer)) or emb_dim <= 0:
            raise ValueError("emb_dim must be a positive integer.")
        self.emb_dim = int(emb_dim)
        if not np.isfinite(alpha) or alpha <= 0.0:
            raise ValueError("alpha must be finite and positive.")
        self.alpha = float(alpha)
        if perplexity is not None and (
            not np.isfinite(perplexity) or not 1.0 <= float(perplexity) <= len(self.landmark_data)
        ):
            raise ValueError(f"perplexity must be between 1 and {len(self.landmark_data)}, or None.")
        self.perplexity = perplexity
        self.affinity = affinity
        self.evidence_cap = evidence_cap
        self.field_weights = field_weights
        if not np.isfinite(drift_threshold_nats) or drift_threshold_nats < 0.0:
            raise ValueError("drift_threshold_nats must be finite and non-negative.")
        self.drift_threshold_nats = float(drift_threshold_nats)
        self.seed = seed
        self._htsne_kwargs = dict(htsne_kwargs)
        self.estimator = estimator
        self._stream_acc = estimator.accumulator_factory().make() if estimator is not None else None
        self._stream_nobs = 0.0

        if atlas is not None:
            atlas = np.asarray(atlas, dtype=np.float64)
            if atlas.shape != (len(self.landmark_data), self.emb_dim):
                raise ValueError(f"atlas must have shape ({len(self.landmark_data)}, {self.emb_dim}).")
            if not np.all(np.isfinite(atlas)):
                raise ValueError("atlas must contain only finite coordinates.")
            self.atlas = atlas.copy()
        else:
            self.atlas = self._embed_landmarks(Y=None)

        self._reference_log_density = _mean_log_density(self.mix_model, self.landmark_data)
        self._recent_log_density: float | None = None

    # -- atlas ------------------------------------------------------------------------------------

    def _embed_landmarks(self, Y: np.ndarray | None, *, mix_model=None) -> np.ndarray:
        import io

        from mixle.utils.hvis.embed import htsne

        kwargs = dict(self._htsne_kwargs)
        kwargs.setdefault("out", io.StringIO())  # quiet by default; pass out=sys.stdout for progress
        if Y is not None:  # warm start: continuity comes from here, exaggeration would wreck it
            kwargs.setdefault("early_exaggeration", 1.0)
        embedded = np.asarray(
            htsne(
                self.landmark_data,
                emb_dim=self.emb_dim,
                alpha=self.alpha,
                perplexity=self.perplexity,
                mix_model=self.mix_model if mix_model is None else mix_model,
                affinity=self.affinity,
                evidence_cap=self.evidence_cap,
                field_weights=self.field_weights,
                seed=self.seed,
                Y=Y,
                **kwargs,
            ),
            dtype=np.float64,
        )
        if embedded.shape != (len(self.landmark_data), self.emb_dim):
            raise ValueError(
                f"landmark embedding must have shape ({len(self.landmark_data)}, {self.emb_dim})."
            )
        if not np.all(np.isfinite(embedded)):
            raise FloatingPointError("landmark embedding produced non-finite coordinates.")
        return embedded

    def _placement_rows(self, batch: list) -> np.ndarray:
        combined = self.landmark_data + list(batch)
        n_landmarks = len(self.landmark_data)
        resolved = _resolve_affinity(self.affinity, self.mix_model, combined, self.field_weights)
        if isinstance(resolved, str):
            z, ll = _posteriors_and_loglikes(self.mix_model, data=combined)
            factors = _affinity_factors(z, ll, resolved)
        else:
            factors = _affinity_factors(None, None, resolved)
        row_idx = np.arange(n_landmarks, n_landmarks + len(batch), dtype=np.int64)
        col_idx = np.arange(n_landmarks, dtype=np.int64)
        log_aff = _cross_log_affinity(factors, row_idx, col_idx, self.evidence_cap)
        return _row_probabilities(log_aff, self.perplexity)

    # -- streaming --------------------------------------------------------------------------------

    def add(self, batch: list, *, max_its: int = 250, eta: float | None = None) -> np.ndarray:
        """Place a batch of arriving observations into the frozen atlas; returns ``(B, emb_dim)``.

        Landmark coordinates are guaranteed unchanged by this call -- stability is structural, not
        a tuning outcome. Also updates the running drift score from the batch's log-density.
        """
        batch = list(batch)
        if not batch:
            return np.zeros((0, self.emb_dim))
        p_rows = self._placement_rows(batch)
        coords = place_in_atlas(p_rows, self.atlas, alpha=self.alpha, max_its=max_its, eta=eta)
        batch_ll = _mean_log_density(self.mix_model, batch)
        candidate_acc = None
        if self._stream_acc is not None:  # build a replacement; never partially mutate live evidence
            candidate_acc = self.estimator.accumulator_factory().make()
            candidate_acc.combine(self._stream_acc.value())
            enc = self.mix_model.dist_to_encoder().seq_encode(batch)
            candidate_acc.seq_update(enc, np.ones(len(batch)), self.mix_model)
        if self._recent_log_density is None:
            recent_log_density = batch_ll
        else:  # EWMA so one odd batch informs but does not own the verdict
            recent_log_density = 0.7 * self._recent_log_density + 0.3 * batch_ll
        if candidate_acc is not None:
            self._stream_acc = candidate_acc
            self._stream_nobs += len(batch)
        self._recent_log_density = recent_log_density
        return coords

    def extend_landmarks(self, data: list, coords: np.ndarray | None = None) -> None:
        """Promote observations into the landmark reservoir (typically recent arrivals), placing
        them first if coordinates are not supplied. Grows the atlas without moving anything."""
        data = list(data)
        if coords is None:
            coords = self.add(data)
        coords = np.asarray(coords, dtype=np.float64)
        if coords.shape != (len(data), self.emb_dim):
            raise ValueError(f"coords must have shape ({len(data)}, {self.emb_dim}).")
        if not np.all(np.isfinite(coords)):
            raise ValueError("coords must contain only finite coordinates.")
        candidate_data = self.landmark_data + data
        candidate_atlas = np.vstack([self.atlas, coords])
        candidate_reference = _mean_log_density(self.mix_model, candidate_data)
        self.landmark_data = candidate_data
        self.atlas = candidate_atlas
        self._reference_log_density = candidate_reference

    # -- drift ------------------------------------------------------------------------------------

    def drift_score(self) -> float:
        """Nats of mean log-density the recent stream sits BELOW the landmark reference (>=0-ish;
        near zero or negative means the stream fits the model at least as well as the reservoir)."""
        if self._recent_log_density is None:
            return 0.0
        return self._reference_log_density - self._recent_log_density

    @property
    def drifted(self) -> bool:
        return self.drift_score() > self.drift_threshold_nats

    # -- refresh ----------------------------------------------------------------------------------

    def refresh(self, mix_model: Any = None) -> dict[str, Any]:
        """Re-embed the landmark reservoir (optionally under an updated model), warm-started from
        the current coordinates and rigidly aligned back onto them.

        With an ``estimator`` configured and stream statistics pending, the model is re-estimated
        first (one incremental-EM M-step over reservoir + stream statistics) and the re-embed runs
        under the NEW model. An explicit ``mix_model`` argument always wins and discards the pending
        stream statistics -- passing both a stream-updated posture and an external model would make
        the vintage of the statistics unaccountable.

        Returns ``{"alignment_residual_rms", "alignment_scale", "atlas_spread", "n_landmarks",
        "model_updated", "n_stream_obs_consumed"}``. A residual small relative to the spread means
        visual continuity is real; a large one means the embedding geometry genuinely changed and
        the report says so rather than hiding it in the alignment. Resets the drift accumulator
        (a refresh is the response to drift, so scoring restarts).
        """
        model_updated = None
        n_consumed = 0.0
        candidate_model = self.mix_model
        clear_pending = False
        if mix_model is not None:
            candidate_model = mix_model
            model_updated = "explicit"
            clear_pending = self._stream_acc is not None
        elif self._stream_acc is not None and self._stream_nobs > 0:
            acc = self.estimator.accumulator_factory().make()
            enc = self.mix_model.dist_to_encoder().seq_encode(self.landmark_data)
            acc.seq_update(enc, np.ones(len(self.landmark_data)), self.mix_model)
            acc.combine(self._stream_acc.value())
            stats_dict: dict[Any, Any] = {}
            acc.key_merge(stats_dict)
            acc.key_replace(stats_dict)
            candidate_model = self.estimator.estimate(None, acc.value())
            model_updated = "stream_em"
            n_consumed = self._stream_nobs
            clear_pending = True
        candidate_acc = self.estimator.accumulator_factory().make() if clear_pending else self._stream_acc
        old = self.atlas.copy()
        new = self._embed_landmarks(Y=old.copy(), mix_model=candidate_model)
        aligned, residual, scale = _procrustes_align(new, old)
        reference_log_density = _mean_log_density(candidate_model, self.landmark_data)
        atlas_spread = float(old.std())
        if not np.isfinite(atlas_spread):
            raise FloatingPointError("atlas spread is non-finite.")

        # Commit only after model estimation, embedding, alignment, reference
        # scoring, and replacement-accumulator construction have all succeeded.
        self.mix_model = candidate_model
        self.atlas = aligned
        self._reference_log_density = reference_log_density
        self._recent_log_density = None
        self._stream_acc = candidate_acc
        if clear_pending:
            self._stream_nobs = 0.0
        return {
            "alignment_residual_rms": residual,
            "alignment_scale": scale,
            "atlas_spread": atlas_spread,
            "n_landmarks": len(self.landmark_data),
            "model_updated": model_updated,
            "n_stream_obs_consumed": float(n_consumed),
        }
