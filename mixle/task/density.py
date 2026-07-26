"""``DensityGate`` -- a real ``p(x)`` over inputs, so the model can escalate on "I've never seen this".

Conformal sets calibrate *which label* is plausible, but a softmax over a ReLU net has no ``p(x)`` -- it cannot
tell a typical input from a wildly novel one, and will hand back a confident singleton for both. That residual
is exactly what a *describable random process* fixes: fit a generative density over the input features (a
diagonal-Gaussian mixture by EM -- mixle's home turf), and an input whose ``log p(x)`` falls below a calibrated
floor is out-of-distribution -> escalate, regardless of how confident the classifier looks.

Pair it with :class:`mixle.task.calibrate.CalibratedTaskModel` (which accepts a ``density_gate=``): the cascade
then escalates when the conformal set is ambiguous **or** the input is atypical -- the union of "unsure which
label" and "unlike anything I trained on". The density is a fitted mixle distribution, so it serializes into the
artifact and reloads identically.
"""

from __future__ import annotations

from collections.abc import Sequence
from hashlib import sha256
from typing import Any

import numpy as np

from mixle.task.model import HashedNGram, HashedRecord


class DensityGate:
    """A generative density over featurized inputs with a calibrated out-of-distribution floor on ``log p(x)``.

    The featurizer is any ``transform(list) -> matrix``: :class:`HashedNGram` for text, or
    :class:`HashedRecord` for dict/tuple records (so record models get the same OOD protection).
    """

    def __init__(
        self,
        featurizer: Any,
        density: Any = None,
        log_threshold: float | None = None,
        calibration_receipt: dict[str, Any] | None = None,
    ) -> None:
        if log_threshold is not None and not np.isfinite(log_threshold):
            raise ValueError("log_threshold must be finite")
        self.featurizer = featurizer
        self.density = density
        self.log_threshold = log_threshold
        self.calibration_receipt = dict(calibration_receipt or {})

    def _rows(self, texts: Sequence[Any]) -> list[np.ndarray]:
        # str-coerce only for the text featurizer: a record featurizer must see the raw dict/tuple,
        # not its repr, or the gate silently scores invalid feature representations.
        items = [str(t) for t in texts] if isinstance(self.featurizer, HashedNGram) else list(texts)
        rows = [np.asarray(r, dtype=np.float64) for r in self.featurizer.transform(items)]
        if not rows:
            raise ValueError("density data must contain at least one example")
        width = rows[0].size
        if rows[0].ndim != 1 or width == 0:
            raise ValueError("density features must be nonempty one-dimensional vectors")
        if any(row.ndim != 1 or row.size != width or not np.all(np.isfinite(row)) for row in rows):
            raise ValueError("density features must have a consistent width and contain only finite values")
        return rows

    def fit(
        self,
        texts: Sequence[Any],
        *,
        calibration_texts: Sequence[Any] | None = None,
        calibration_frac: float = 0.2,
        n_components: int = 4,
        alpha: float = 0.02,
        max_its: int = 60,
        min_covar: float = 1e-3,
        seed: int = 0,
    ) -> DensityGate:
        """Fit density on one slice and calibrate its OOD floor on a disjoint slice.

        When ``calibration_texts`` is omitted, a reproducible internal split is made before fitting. An
        explicit calibration sequence lets a caller share an existing held-out split. The receipt records
        counts and content digests without retaining potentially sensitive examples.
        """
        import mixle.stats as st
        from mixle.inference import optimize

        if isinstance(n_components, bool) or not isinstance(n_components, int) or n_components <= 0:
            raise ValueError("n_components must be a positive integer")
        if isinstance(max_its, bool) or not isinstance(max_its, int) or max_its <= 0:
            raise ValueError("max_its must be a positive integer")
        if isinstance(seed, bool) or not isinstance(seed, int):
            raise ValueError("seed must be an integer")
        if not np.isfinite(alpha) or not 0.0 < alpha < 1.0:
            raise ValueError("alpha must be finite and in (0, 1)")
        if not np.isfinite(min_covar) or min_covar <= 0.0:
            raise ValueError("min_covar must be finite and positive")

        source = list(texts)
        if calibration_texts is None:
            if not np.isfinite(calibration_frac) or not 0.0 < calibration_frac < 1.0:
                raise ValueError("calibration_frac must be finite and in (0, 1)")
            if len(source) < 2:
                raise ValueError("at least two examples are required for disjoint density calibration")
            permutation = np.random.RandomState(seed).permutation(len(source))
            n_cal = min(len(source) - 1, max(1, int(round(len(source) * calibration_frac))))
            cal_indices = [int(i) for i in permutation[:n_cal]]
            fit_indices = [int(i) for i in permutation[n_cal:]]
            fit_items = [source[i] for i in fit_indices]
            cal_items = [source[i] for i in cal_indices]
            split_kind = "seeded_internal"
        else:
            fit_items = source
            cal_items = list(calibration_texts)
            if not fit_items or not cal_items:
                raise ValueError("explicit density fitting and calibration slices must both be nonempty")
            fit_indices = list(range(len(fit_items)))
            cal_indices = list(range(len(cal_items)))
            split_kind = "explicit"

        rows = self._rows(fit_items)
        cal_rows = self._rows(cal_items)
        if rows[0].size != cal_rows[0].size:
            raise ValueError("density fitting and calibration features must have the same width")
        dim = rows[0].shape[0]
        est = st.MixtureEstimator([st.DiagonalGaussianEstimator(dim=dim, min_covar=min_covar)] * n_components)
        self.density = optimize(rows, est, max_its=max_its, rng=np.random.RandomState(seed), out=None)
        ld = self._log_density_rows(cal_rows)
        if ld.shape != (len(cal_rows),) or not np.all(np.isfinite(ld)):
            raise ValueError("fitted density returned invalid calibration scores")
        self.log_threshold = float(np.quantile(ld, alpha, method="lower"))
        self.calibration_receipt = {
            "kind": split_kind,
            "seed": seed,
            "alpha": float(alpha),
            "fit_count": len(fit_items),
            "calibration_count": len(cal_items),
            "fit_indices": fit_indices if split_kind == "seeded_internal" else None,
            "calibration_indices": cal_indices if split_kind == "seeded_internal" else None,
            "fit_digest": sha256(repr(fit_items).encode("utf-8")).hexdigest(),
            "calibration_digest": sha256(repr(cal_items).encode("utf-8")).hexdigest(),
        }
        return self

    def _log_density_rows(self, rows: list[np.ndarray]) -> np.ndarray:
        enc = self.density.dist_to_encoder().seq_encode(rows)
        scores = np.asarray(self.density.seq_log_density(enc), dtype=np.float64)
        if scores.shape != (len(rows),) or not np.all(np.isfinite(scores)):
            raise ValueError("density returned invalid log-density scores")
        return scores

    def log_density(self, texts: Sequence[str]) -> np.ndarray:
        """``log p(x)`` of each input under the fitted density (higher = more typical of training data)."""
        if self.density is None:
            raise RuntimeError("call fit(...) (or load a fitted gate) before scoring")
        return self._log_density_rows(self._rows(texts))

    def is_ood(self, text: str) -> bool:
        """True when the input is atypical: ``log p(x)`` below the calibrated floor."""
        if self.log_threshold is None:
            raise RuntimeError("density gate has not been calibrated")
        return bool(self.log_density([text])[0] < self.log_threshold)

    def ood_mask(self, texts: Sequence[str]) -> np.ndarray:
        """Return a boolean mask marking inputs below the calibrated density floor."""
        if self.log_threshold is None:
            raise RuntimeError("density gate has not been calibrated")
        return self.log_density(texts) < self.log_threshold

    def to_spec(self) -> dict[str, Any]:
        """Serialize the featurizer, fitted density, and threshold for task artifacts."""
        from mixle.utils.serialization import ensure_pysp_serialization_registry, to_serializable

        ensure_pysp_serialization_registry()
        return {
            "featurizer": self.featurizer.to_spec(),
            "featurizer_kind": "record" if isinstance(self.featurizer, HashedRecord) else "text",
            "density": to_serializable(self.density),
            "log_threshold": self.log_threshold,
            "calibration_receipt": self.calibration_receipt,
        }

    @classmethod
    def from_spec(cls, spec: dict[str, Any]) -> DensityGate:
        """Rebuild a density gate from :meth:`to_spec` output."""
        from mixle.utils.serialization import ensure_pysp_serialization_registry, from_serializable

        ensure_pysp_serialization_registry()
        feat_cls = HashedRecord if spec.get("featurizer_kind") == "record" else HashedNGram
        return cls(
            feat_cls.from_spec(spec["featurizer"]),
            density=from_serializable(spec["density"]),
            log_threshold=spec["log_threshold"],
            calibration_receipt=spec.get("calibration_receipt"),
        )
