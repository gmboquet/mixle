"""Neural conditional models for ``mixle.ppl`` -- a :class:`~mixle.ppl.core.Net` predictor in a slot.

The nonlinear sibling of :mod:`mixle.ppl.regression`. A ``Net`` in an outer family's slot makes a neural
conditional model; the outer family sets the link::

    Categorical(logits=Net(out=K)).fit(y, given={"x": X})   # softmax link  -> classification  (SoftmaxNeuralLeaf)
    Normal(Net(out=1), free).fit(y, given={"x": X})         # identity link -> neural mean + learned noise (the blend)

The objective is the leaf's own log-density; fitting routes to the standard
:func:`mixle.inference.estimate` loop -- there is no loss function and no training loop in user code.
"""

from __future__ import annotations

from numbers import Integral, Real
from typing import Any

import numpy as np

from mixle.ppl.core import _NeuralPredictor


def _positive_int(value: Any, name: str) -> int:
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, Integral):
        raise TypeError(f"{name} must be a positive integer")
    result = int(value)
    if result <= 0:
        raise ValueError(f"{name} must be a positive integer")
    return result


def _positive_finite(value: Any, name: str) -> float:
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, Real):
        raise TypeError(f"{name} must be a positive finite real number")
    result = float(value)
    if not np.isfinite(result) or result <= 0.0:
        raise ValueError(f"{name} must be a positive finite real number")
    return result


def _features(value: Any) -> np.ndarray:
    try:
        result = np.asarray(value, dtype="float32")
    except (TypeError, ValueError) as error:
        raise TypeError("neural covariates must be a finite numeric array") from error
    if result.ndim < 2 or result.shape[0] == 0 or any(size == 0 for size in result.shape[1:]):
        raise ValueError("neural covariates must have shape (rows, ...non-empty feature axes)")
    if not np.all(np.isfinite(result)):
        raise ValueError("neural covariates must contain only finite values")
    return result


def _weights(value: Any, rows: int) -> np.ndarray:
    try:
        result = np.asarray(value, dtype=float)
    except (TypeError, ValueError) as error:
        raise TypeError("weights must be a finite non-negative vector") from error
    if result.ndim != 1 or result.shape != (rows,):
        raise ValueError(f"weights must be one-dimensional with exactly {rows} entries")
    if not np.all(np.isfinite(result)) or np.any(result < 0.0):
        raise ValueError("weights must contain finite non-negative values")
    if not np.any(result > 0.0):
        raise ValueError("weights must contain at least one positive value")
    return result


def _labels(value: Any, rows: int, classes: int) -> np.ndarray:
    result = np.asarray(value)
    if result.ndim != 1 or result.shape != (rows,):
        raise ValueError(f"categorical responses must be one-dimensional with exactly {rows} entries")
    if result.dtype.kind not in {"i", "u"}:
        raise ValueError("categorical responses must contain exact integer class labels")
    if np.any(result < 0) or np.any(result >= classes):
        raise ValueError(f"categorical class labels must lie in [0, {classes})")
    return result.astype(np.int64, copy=False)


def _responses(value: Any, rows: int) -> np.ndarray:
    try:
        result = np.asarray(value, dtype=float)
    except (TypeError, ValueError) as error:
        raise TypeError("normal responses must be a finite numeric array") from error
    if result.ndim == 1:
        result = result[:, None]
    if result.ndim != 2 or result.shape[0] != rows or result.shape[1] == 0:
        raise ValueError(
            f"normal responses must have shape ({rows}, outputs) with a non-empty output axis"
        )
    if not np.all(np.isfinite(result)):
        raise ValueError("normal responses must contain only finite values")
    return result


def _validated_device(value: Any, torch: Any) -> str:
    try:
        device = torch.device(value)
    except (TypeError, RuntimeError) as error:
        raise ValueError(f"invalid torch device {value!r}") from error
    if device.type not in {"cpu", "cuda", "mps"}:
        raise ValueError(f"unsupported torch device type {device.type!r}; use 'cpu', 'cuda', or 'mps'")
    if device.type == "cuda" and not torch.cuda.is_available():
        raise ValueError("CUDA device requested but CUDA is unavailable")
    if device.type == "mps" and not getattr(torch.backends, "mps", None):
        raise ValueError("MPS device requested but this torch build has no MPS backend")
    if device.type == "mps" and not torch.backends.mps.is_available():
        raise ValueError("MPS device requested but MPS is unavailable")
    return str(device)


def _validated_module(
    net: _NeuralPredictor,
    x: np.ndarray,
    expected_outputs: int,
    *,
    init: Any,
    family: str,
    device: str,
) -> Any:
    import torch

    if init is None:
        module = net.build(tuple(x.shape[1:]))
    else:
        if not isinstance(init, NeuralResult):
            raise TypeError("init must be a NeuralResult from a compatible prior fit")
        expected_kind = "categorical" if family == "Categorical" else "normal"
        if init.kind != expected_kind or init.field != net.field:
            raise ValueError(
                f"init is for kind={init.kind!r}, field={init.field!r}; "
                f"expected kind={expected_kind!r}, field={net.field!r}"
            )
        module = getattr(init.dist, "module", None)
    if not isinstance(module, torch.nn.Module):
        raise TypeError("neural predictor must build a torch.nn.Module")
    parameters = list(module.parameters())
    if not parameters:
        raise ValueError("neural predictor module must expose at least one trainable parameter")
    if not any(parameter.requires_grad for parameter in parameters):
        raise ValueError("neural predictor module must expose at least one trainable parameter")
    try:
        module.to(device=device, dtype=torch.float32)
    except (RuntimeError, TypeError, ValueError) as error:
        raise ValueError(f"neural predictor cannot be moved to device {device!r}: {error}") from error
    was_training = module.training
    try:
        module.eval()
        with torch.no_grad():
            probe = module(torch.as_tensor(x[: min(2, len(x))], dtype=torch.float32, device=device))
    except Exception as error:
        raise ValueError(
            f"neural predictor cannot evaluate covariates with shape {tuple(x.shape[1:])}: {error}"
        ) from error
    finally:
        module.train(was_training)
    if not isinstance(probe, torch.Tensor):
        raise TypeError("neural predictor module must return a torch.Tensor")
    if probe.ndim != 2 or probe.shape[0] != min(2, len(x)) or probe.shape[1] != expected_outputs:
        raise ValueError(
            "neural predictor output must have shape "
            f"(rows, {expected_outputs}); got {tuple(probe.shape)}"
        )
    if not bool(torch.isfinite(probe).all()):
        raise ValueError("neural predictor output must contain only finite values")
    return module


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

    ``epochs`` is the number of passes and ``device`` selects CPU, MPS, or CUDA execution. Categorical heads
    also accept ``batch_size`` (None = full batch); Normal heads reject it until their estimator has a real
    minibatch implementation. The input keeps its natural shape -- (N, D) or (N, C, H, W).

    Multi-stage pipeline (one module across stages):
    ``init=`` continues a previous fit's module (CPT/SFT) instead of building a fresh one; ``weights=`` are
    per-observation loss weights (e.g. an SFT prompt mask: 0 on prompt tokens, 1 on the completion); ``ewc=``
    is an ``(anchor, fisher, lambda)`` EWC penalty for continued pretraining without forgetting.
    """
    import torch

    from mixle.inference import estimate

    predictors = [argument for argument in rv._args if isinstance(argument, _NeuralPredictor)]
    if len(predictors) != 1:
        raise ValueError(f"neural fit requires exactly one neural predictor slot; found {len(predictors)}")
    net = predictors[0]
    if given is None:
        given = {}
    if not isinstance(given, dict):
        raise TypeError("given must be a dictionary of aligned covariate arrays")
    if net.field not in given:
        raise ValueError(f"neural fit needs covariates: .fit(y, given={{{net.field!r}: X}})")
    x = _features(given[net.field])
    epochs = _positive_int(epochs, "epochs")
    lr = _positive_finite(lr, "lr")
    if batch_size is not None:
        batch_size = _positive_int(batch_size, "batch_size")
    device = _validated_device(device, torch)
    fam = rv._family.name

    if fam == "Categorical":
        from mixle.models.softmax_leaf import SoftmaxNeuralLeafEstimator

        expected_outputs = _positive_int(getattr(net, "out", None), "neural predictor output width")
        y = _labels(data, len(x), expected_outputs)
        w = np.ones(len(y)) if weights is None else _weights(weights, len(y))
        module = _validated_module(
            net,
            x,
            expected_outputs,
            init=init,
            family=fam,
            device=device,
        )
        est = SoftmaxNeuralLeafEstimator(
            module,
            m_steps=epochs,
            lr=lr,
            batch_size=batch_size,
            device=device,
            ewc=ewc,
            optimizer_state=None if init is None else getattr(init.dist, "optimizer_state", None),
        )
        if weights is None and ewc is None:
            fitted = estimate(list(zip(x, y)), est)
        else:  # per-observation loss weights (SFT mask) and/or the EWC penalty -> drive the accumulator directly
            acc = est.accumulator_factory().make()
            enc = acc.acc_to_encoder().seq_encode(list(zip(x, y)))
            acc.seq_update(enc, w, None)
            fitted = est.estimate(None, acc.value())
        return NeuralResult(fitted, net.field, "categorical")

    if fam in ("Normal", "Gaussian"):
        from mixle.models.neural_leaf import NeuralLeaf

        if batch_size is not None:
            raise NotImplementedError(
                "Normal neural fitting does not yet support batch_size; omit it or use a categorical head"
            )
        if ewc is not None:
            raise NotImplementedError(
                "Normal neural fitting does not yet support EWC; omit ewc or use a categorical head"
            )
        y = _responses(data, len(x))
        expected_outputs = int(y.shape[1])
        declared_outputs = _positive_int(getattr(net, "out", None), "neural predictor output width")
        if declared_outputs != expected_outputs:
            raise ValueError(
                f"Normal neural predictor declares {declared_outputs} outputs but responses have {expected_outputs}"
            )
        w = None if weights is None else _weights(weights, len(y))
        module = _validated_module(
            net,
            x,
            expected_outputs,
            init=init,
            family=fam,
            device=device,
        )
        leaf = NeuralLeaf(module, m_steps=epochs, lr=lr, device=device)
        estimator = leaf.estimator()
        if w is None:
            fitted = estimate(list(zip(x, y)), estimator)
        else:
            accumulator = estimator.accumulator_factory().make()
            encoded = accumulator.acc_to_encoder().seq_encode(list(zip(x, y)))
            accumulator.seq_update(encoded, w, None)
            fitted = estimator.estimate(None, accumulator.value())
        return NeuralResult(fitted, net.field, "normal")

    raise NotImplementedError(f"a Net slot is not supported for the {fam!r} family yet.")
