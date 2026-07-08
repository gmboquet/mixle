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
    b, n_landmarks = log_aff.shape
    p = np.zeros_like(log_aff)
    target = None if perplexity is None else np.log(min(float(perplexity), max(1.0, n_landmarks - 1.0)))
    for i in range(b):
        row = log_aff[i]
        finite = np.isfinite(row)
        if not np.any(finite):
            p[i] = 1.0 / n_landmarks  # the model offers no information: honest uniform, not a crash
            continue
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
        velocity = momentum * velocity - eta * grad
        y = y + velocity
        if float(np.abs(grad).max()) < tol:
            break
    return y


def _procrustes_align(source: np.ndarray, target: np.ndarray) -> tuple[np.ndarray, float, float]:
    """Align ``source`` onto ``target`` by rotation/reflection + translation + UNIFORM scale (full
    Procrustes). A t-SNE layout expands globally as it converges, so isotropic scale is nuisance for
    the continuity question; what the residual measures is SHAPE change. The scale factor is returned
    (and reported) rather than hidden -- a large one says the layout re-expanded even if the shape
    held. Returns ``(aligned, rms_residual, scale)``."""
    from scipy.linalg import orthogonal_procrustes

    mu_s, mu_t = source.mean(axis=0), target.mean(axis=0)
    src, tgt = source - mu_s, target - mu_t
    rotation, trace = orthogonal_procrustes(src, tgt)
    denom = float(np.sum(src * src))
    scale = float(trace) / denom if denom > 0 else 1.0
    aligned = scale * (src @ rotation) + mu_t
    residual = float(np.sqrt(np.mean(np.sum((aligned - target) ** 2, axis=1))))
    return aligned, residual, scale


def _mean_log_density(mix_model, data) -> float:
    """Mean per-observation log-density under the mixture -- the drift reference/score input."""
    if hasattr(mix_model, "dist_to_encoder") and hasattr(mix_model, "seq_log_density"):
        enc = mix_model.dist_to_encoder().seq_encode(list(data))
        return float(np.mean(np.asarray(mix_model.seq_log_density(enc), dtype=np.float64)))
    _, ll_mat = _posteriors_and_loglikes(mix_model, data=list(data))
    log_w = np.asarray(mix_model.log_w, dtype=np.float64).reshape(1, -1)
    joint = ll_mat + log_w
    mx = joint.max(axis=1, keepdims=True)
    return float(np.mean(mx[:, 0] + np.log(np.exp(joint - mx).sum(axis=1))))


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
        self.mix_model = mix_model
        self.landmark_data = list(landmark_data)
        self.emb_dim = int(emb_dim)
        self.alpha = float(alpha)
        self.perplexity = perplexity
        self.affinity = affinity
        self.evidence_cap = evidence_cap
        self.field_weights = field_weights
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
            self.atlas = atlas.copy()
        else:
            self.atlas = self._embed_landmarks(Y=None)

        self._reference_log_density = _mean_log_density(self.mix_model, self.landmark_data)
        self._recent_log_density: float | None = None

    # -- atlas ------------------------------------------------------------------------------------

    def _embed_landmarks(self, Y: np.ndarray | None) -> np.ndarray:
        import io

        from mixle.utils.hvis.embed import htsne

        kwargs = dict(self._htsne_kwargs)
        kwargs.setdefault("out", io.StringIO())  # quiet by default; pass out=sys.stdout for progress
        if Y is not None:  # warm start: continuity comes from here, exaggeration would wreck it
            kwargs.setdefault("early_exaggeration", 1.0)
        return np.asarray(
            htsne(
                self.landmark_data,
                emb_dim=self.emb_dim,
                alpha=self.alpha,
                perplexity=self.perplexity,
                mix_model=self.mix_model,
                affinity=self.affinity,
                evidence_cap=self.evidence_cap,
                field_weights=self.field_weights,
                seed=self.seed,
                Y=Y,
                **kwargs,
            ),
            dtype=np.float64,
        )

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
        if self._recent_log_density is None:
            self._recent_log_density = batch_ll
        else:  # EWMA so one odd batch informs but does not own the verdict
            self._recent_log_density = 0.7 * self._recent_log_density + 0.3 * batch_ll
        if self._stream_acc is not None:  # incremental EM: E-step now, under the model of this moment
            enc = self.mix_model.dist_to_encoder().seq_encode(batch)
            self._stream_acc.seq_update(enc, np.ones(len(batch)), self.mix_model)
            self._stream_nobs += len(batch)
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
        self.landmark_data.extend(data)
        self.atlas = np.vstack([self.atlas, coords])
        self._reference_log_density = _mean_log_density(self.mix_model, self.landmark_data)

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
        if mix_model is not None:
            self.mix_model = mix_model
            model_updated = "explicit"
            if self._stream_acc is not None:  # stale-vintage stats must not survive an external model
                n_consumed = 0.0
                self._stream_acc = self.estimator.accumulator_factory().make()
                self._stream_nobs = 0.0
        elif self._stream_acc is not None and self._stream_nobs > 0:
            acc = self.estimator.accumulator_factory().make()
            enc = self.mix_model.dist_to_encoder().seq_encode(self.landmark_data)
            acc.seq_update(enc, np.ones(len(self.landmark_data)), self.mix_model)
            acc.combine(self._stream_acc.value())
            stats_dict: dict[Any, Any] = {}
            acc.key_merge(stats_dict)
            acc.key_replace(stats_dict)
            self.mix_model = self.estimator.estimate(None, acc.value())
            model_updated = "stream_em"
            n_consumed = self._stream_nobs
            self._stream_acc = self.estimator.accumulator_factory().make()
            self._stream_nobs = 0.0
        old = self.atlas
        new = self._embed_landmarks(Y=old.copy())
        aligned, residual, scale = _procrustes_align(new, old)
        self.atlas = aligned
        self._reference_log_density = _mean_log_density(self.mix_model, self.landmark_data)
        self._recent_log_density = None
        return {
            "alignment_residual_rms": residual,
            "alignment_scale": scale,
            "atlas_spread": float(old.std()),
            "n_landmarks": len(self.landmark_data),
            "model_updated": model_updated,
            "n_stream_obs_consumed": float(n_consumed),
        }
