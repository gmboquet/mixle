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
from typing import Any

import numpy as np

from mixle.task.model import HashedNGram, HashedRecord


class DensityGate:
    """A generative density over featurized inputs with a calibrated out-of-distribution floor on ``log p(x)``.

    The featurizer is any ``transform(list) -> matrix``: :class:`HashedNGram` for text, or
    :class:`HashedRecord` for dict/tuple records (so record models get the same OOD protection).
    """

    def __init__(self, featurizer: Any, density: Any = None, log_threshold: float | None = None) -> None:
        self.featurizer = featurizer
        self.density = density
        self.log_threshold = log_threshold

    def _rows(self, texts: Sequence[Any]) -> list[np.ndarray]:
        # str-coerce only for the text featurizer: a record featurizer must see the raw dict/tuple,
        # not its repr, or the gate silently scores invalid feature representations.
        items = [str(t) for t in texts] if isinstance(self.featurizer, HashedNGram) else list(texts)
        return [np.asarray(r, dtype=np.float64) for r in self.featurizer.transform(items)]

    def fit(
        self,
        texts: Sequence[str],
        *,
        n_components: int = 4,
        alpha: float = 0.02,
        max_its: int = 60,
        min_covar: float = 1e-3,
        seed: int = 0,
    ) -> DensityGate:
        """Fit a diagonal-Gaussian mixture to the features and set the OOD floor at the ``alpha`` density quantile."""
        import mixle.stats as st
        from mixle.inference import optimize

        rows = self._rows(texts)
        dim = rows[0].shape[0]
        est = st.MixtureEstimator([st.DiagonalGaussianEstimator(dim=dim, min_covar=min_covar)] * n_components)
        self.density = optimize(rows, est, max_its=max_its, rng=np.random.RandomState(seed), out=None)
        ld = self._log_density_rows(rows)
        self.log_threshold = float(np.quantile(ld, alpha))
        return self

    def _log_density_rows(self, rows: list[np.ndarray]) -> np.ndarray:
        enc = self.density.dist_to_encoder().seq_encode(rows)
        return np.asarray(self.density.seq_log_density(enc), dtype=np.float64)

    def log_density(self, texts: Sequence[str]) -> np.ndarray:
        """``log p(x)`` of each input under the fitted density (higher = more typical of training data)."""
        if self.density is None:
            raise RuntimeError("call fit(...) (or load a fitted gate) before scoring")
        return self._log_density_rows(self._rows(texts))

    def is_ood(self, text: str) -> bool:
        """True when the input is atypical: ``log p(x)`` below the calibrated floor."""
        return bool(self.log_density([text])[0] < self.log_threshold)

    def ood_mask(self, texts: Sequence[str]) -> np.ndarray:
        """Return a boolean mask marking inputs below the calibrated density floor."""
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
        )
