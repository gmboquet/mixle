"""Neural conditional models for ``mixle.ppl`` -- a :class:`~mixle.ppl.core.Net` predictor in a slot.

The nonlinear sibling of :mod:`mixle.ppl.regression`. A ``Net`` in an outer family's slot makes a neural
conditional model; the outer family sets the link::

    Categorical(logits=Net(out=K)).fit(y, given={"x": X})   # softmax link  -> classification  (SoftmaxNeuralLeaf)
    Normal(Net(out=1), free).fit(y, given={"x": X})         # identity link -> neural mean + learned noise (the blend)

The objective is the leaf's own log-density; fitting routes to the standard
:func:`mixle.inference.estimate` loop -- there is no loss function and no training loop in user code.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from mixle.ppl.core import _NeuralPredictor


class NeuralResult:
    """A fitted neural conditional model. ``predict(given={"x": X})`` returns class labels (Categorical) or the
    conditional mean (Normal) at new covariates -- the same shape of interface as ``RegressionResult.predict``.
    ``.dist`` is the underlying mixle leaf (composes into mixtures / composites like any distribution)."""

    def __init__(self, dist: Any, field: str, kind: str) -> None:
        self.dist = dist
        self.field = field
        self.kind = kind

    def _design(self, given: dict) -> np.ndarray:
        if self.field not in given:
            raise ValueError(f"needs the covariates: given={{{self.field!r}: X}}")
        # keep the natural shape: (N, D) for an MLP, (N, C, H, W) for a conv net -- the module handles it
        return np.asarray(given[self.field], dtype="float32")

    def predict(self, given: dict) -> np.ndarray:
        """Class labels (Categorical) or the conditional mean (Normal) at covariates ``given``."""
        x = self._design(given)
        return self.dist.predict(x) if self.kind == "categorical" else self.dist._forward(x)

    def score(self, data: Any, given: dict) -> float:
        """Held-out accuracy (Categorical) or R^2 (Normal) on ``(data, given)``."""
        pred = self.predict(given)
        if self.kind == "categorical":
            return float(np.mean(pred == np.asarray(data, dtype=int).reshape(-1)))
        y = np.asarray(data, dtype=float).reshape(len(pred), -1)
        ss = ((y - pred) ** 2).sum()
        return float(1.0 - ss / (((y - y.mean(0)) ** 2).sum() + 1e-12))


def neural_fit(
    rv: Any,
    data: Any,
    *,
    given: dict | None = None,
    epochs: int = 200,
    lr: float = 0.01,
    batch_size: int | None = None,
    device: str = "cpu",
    init: Any = None,
    weights: Any = None,
    ewc: Any = None,
    **_: Any,
) -> NeuralResult:
    """Fit a neural-headed conditional RV. ``data`` is the response ``y``; ``given`` carries the covariates.

    ``epochs`` is the number of passes; ``batch_size`` (None = full batch) and ``device`` ("mps"/"cuda") let a
    conv net train on image tensors on the selected device. The input keeps its natural shape -- (N, D) or
    (N, C, H, W).

    Multi-stage pipeline (one module across stages):
    ``init=`` continues a previous fit's module (CPT/SFT) instead of building a fresh one; ``weights=`` are
    per-observation loss weights (e.g. an SFT prompt mask: 0 on prompt tokens, 1 on the completion); ``ewc=``
    is an ``(anchor, fisher, lambda)`` EWC penalty for continued pretraining without forgetting.
    """
    from mixle.inference import estimate

    net = next(a for a in rv._args if isinstance(a, _NeuralPredictor))
    given = given or {}
    if net.field not in given:
        raise ValueError(f"neural fit needs covariates: .fit(y, given={{{net.field!r}: X}})")
    x = np.asarray(given[net.field], dtype="float32")
    module = init.dist.module if init is not None else net.build(tuple(x.shape[1:]))  # continue, or build fresh
    fam = rv._family.name

    if fam == "Categorical":
        from mixle.models.softmax_leaf import SoftmaxNeuralLeafEstimator

        y = np.asarray(data, dtype=int).reshape(-1)
        est = SoftmaxNeuralLeafEstimator(
            module, m_steps=int(epochs), lr=float(lr), batch_size=batch_size, device=device, ewc=ewc
        )
        if weights is None and ewc is None:
            fitted = estimate(list(zip(x, y)), est)
        else:  # per-observation loss weights (SFT mask) and/or the EWC penalty -> drive the accumulator directly
            acc = est.accumulator_factory().make()
            enc = est.accumulator_factory().make().acc_to_encoder().seq_encode(list(zip(x, y)))
            w = np.ones(len(y)) if weights is None else np.asarray(weights, dtype=float)
            acc.seq_update(enc, w, None)
            fitted = est.estimate(None, acc.value())
        return NeuralResult(fitted, net.field, "categorical")

    if fam in ("Normal", "Gaussian"):
        from mixle.models.neural_leaf import NeuralLeaf

        y = np.asarray(data, dtype=float).reshape(len(x), -1)
        leaf = NeuralLeaf(module, m_steps=int(epochs), lr=float(lr))
        fitted = estimate(list(zip(x, y)), leaf.estimator())
        return NeuralResult(fitted, net.field, "normal")

    raise NotImplementedError(f"a Net slot is not supported for the {fam!r} family yet.")
