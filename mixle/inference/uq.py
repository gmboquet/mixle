"""Unified uncertainty dispatch for models, predictors, ensembles, and LLM-style callables.

``uq(thing, data)`` inspects the object it receives and routes to a compatible
uncertainty method, returning a :class:`UQResult` with the method name and the
quantities needed for downstream checks.

  * a fitted mixle model (has ``seq_log_density``) -> a Laplace likelihood-curvature approximation;
    sample fitted models and get an explicitly approximate parameter interval.
  * a torch module / any point predictor callable over arrays -> split-conformal calibration from a
    held-out ``(X, y)``; ``interval(x)`` returns a prediction interval with finite-sample coverage.
    Scope of that statement: the ``1 - alpha`` coverage is MARGINAL over the calibration draw and
    the query jointly, under exchangeability of the calibration rows and incoming queries; it is
    not a per-query statement, and distribution shift voids it silently -- re-calibrate on drifted
    traffic. Give a LIST of predictors instead and it becomes a deep ensemble (epistemic spread +
    conformal).
  * an LLM-style callable over prompts (returns a string, or samples of strings) -> semantic entropy
    over meaning classes; ``confident(prompt)`` abstains when the model disagrees with itself.

The method is chosen from observed capability rather than a caller-supplied
mode string.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

import numpy as np

__all__ = ["UQResult", "uq"]


@dataclass
class UQResult:
    """The uncertainty of a predictor, with the method that produced it and receipts to check it."""

    kind: str
    method: str  # human-readable method name
    payload: dict[str, Any]

    # -- mixle model: Laplace parameter approximation ----------------------------------------------
    def sample_models(self, n: int = 200, *, seed: int | None = None) -> list[Any]:
        """``n`` fitted models drawn from the parameter approximation."""
        if isinstance(n, (bool, np.bool_)) or not isinstance(n, (int, np.integer)) or n < 1:
            raise ValueError("n must be a positive integer")
        post = self.payload.get("posterior", self.payload.get("approximation"))
        if post is None:
            raise TypeError("this UQ result has no parameter approximation")
        rng = np.random.RandomState(seed) if seed is not None else None
        return post.sample(int(n), rng=rng)

    def credible_interval(
        self, readout: Callable[[Any], float], alpha: float = 0.1, *, n: int = 400, seed: int = 0
    ) -> tuple[float, float]:
        """A ``1-alpha`` credible interval on ``readout(model)`` over the parameter posterior."""
        approximation = self.payload.get("posterior", self.payload.get("approximation"))
        if approximation is None or not bool(getattr(approximation, "is_posterior", False)):
            raise ValueError(
                "credible_interval requires an explicit-prior posterior; use parameter_interval for "
                "a likelihood-curvature approximation"
            )
        return self.parameter_interval(readout, alpha=alpha, n=n, seed=seed)

    def parameter_interval(
        self, readout: Callable[[Any], float], alpha: float = 0.1, *, n: int = 400, seed: int = 0
    ) -> tuple[float, float]:
        """A central interval over draws from the disclosed parameter approximation."""
        _validate_alpha(alpha)
        if not callable(readout):
            raise TypeError("readout must be callable")
        vals = np.asarray([float(readout(m)) for m in self.sample_models(n, seed=seed)], dtype=float)
        if not np.all(np.isfinite(vals)):
            raise ValueError("readout must return a finite value for every posterior model")
        lo, hi = np.quantile(vals, [alpha / 2.0, 1.0 - alpha / 2.0])
        return float(lo), float(hi)

    # -- point predictor: split conformal (single or ensemble) -------------------------------------
    def interval(self, x: Any, alpha: float | None = None) -> tuple[np.ndarray, np.ndarray]:
        """Calibrated prediction interval(s) at ``x``. ``alpha`` overrides the calibrated level."""
        from mixle.inference.conformal import split_conformal

        predict = self.payload["predict"]
        q = self.payload["qhat"] if alpha is None else None
        pred = _finite_prediction(predict(x), "predict")
        if q is not None:
            return pred - q, pred + q
        lo, hi = split_conformal(self.payload["cal_pred"], self.payload["cal_y"], pred, alpha=float(alpha))
        return lo, hi

    def epistemic_std(self, x: Any) -> np.ndarray:
        """Ensemble disagreement (std across members) at ``x`` -- 0.0 for a single predictor."""
        members = self.payload.get("members")
        if not members:
            return np.zeros(_finite_prediction(self.payload["predict"](x), "predict").shape)
        predictions = [_finite_prediction(member(x), "ensemble member") for member in members]
        if any(prediction.shape != predictions[0].shape for prediction in predictions[1:]):
            raise ValueError("ensemble members must return predictions with matching shapes")
        preds = np.stack(predictions)
        return preds.std(axis=0)

    # -- LLM callable: semantic entropy ------------------------------------------------------------
    def semantic_entropy(self, prompt: Any, *, n: int = 8) -> float:
        """Entropy (nats) over the meaning classes of ``n`` sampled generations for ``prompt``."""
        from mixle.inference.uncertainty import semantic_entropy as _se

        gen = self.payload["generate"]
        equivalent = self.payload.get("equivalent")
        if isinstance(n, (bool, np.bool_)) or not isinstance(n, (int, np.integer)) or n < 1:
            raise ValueError("n must be a positive integer")
        samples = [gen(prompt) for _ in range(int(n))]
        entropy = float(_se(samples, equivalent))
        if not np.isfinite(entropy):
            raise ValueError("semantic entropy calculation returned a non-finite value")
        return entropy

    def confident(self, prompt: Any, *, n: int = 8, max_entropy: float | None = None) -> bool:
        """True when semantic entropy is below the threshold -- else the model disagrees with itself."""
        thr = self.payload["max_entropy"] if max_entropy is None else float(max_entropy)
        if thr is None:
            raise ValueError("confident() requires calibrated prompts or an explicit max_entropy")
        if not np.isfinite(thr) or thr < 0.0:
            raise ValueError("max_entropy must be a finite non-negative value")
        return self.semantic_entropy(prompt, n=n) <= thr

    def report(self) -> dict[str, Any]:
        """Return uncertainty-quantification metadata and scalar payload fields."""
        r = {"kind": self.kind, "method": self.method}
        r.update({k: v for k, v in self.payload.items() if isinstance(v, (int, float, str, bool))})
        return r


# --------------------------------------------------------------------------------------------------
# dispatch
# --------------------------------------------------------------------------------------------------


def _is_mixle_model(obj: Any) -> bool:
    return hasattr(obj, "seq_log_density") and hasattr(obj, "dist_to_encoder")


def _is_torch_module(obj: Any) -> bool:
    if callable(getattr(obj, "parameters", None)):
        try:
            import torch.nn as nn

            return isinstance(obj, nn.Module)
        except Exception:  # noqa: BLE001
            return True
    return False


def _as_predict(model: Any) -> Callable[[Any], np.ndarray]:
    """A point-prediction callable ``x -> yhat`` for a torch module or a plain array predictor."""
    if _is_torch_module(model):
        import torch

        def predict(x: Any) -> np.ndarray:
            modes = {module: bool(module.training) for module in model.modules()}
            try:
                model.eval()
                with torch.no_grad():
                    xt = torch.as_tensor(np.atleast_2d(np.asarray(x, dtype=float)), dtype=torch.float32)
                    out = model(xt)
                    values = out.detach().cpu().numpy() if hasattr(out, "detach") else np.asarray(out)
                    return _finite_prediction(values, "torch model")
            finally:
                for module, training in modes.items():
                    module.training = training

        return predict
    return lambda x: _finite_prediction(model(x), "predictor")


def _finite_prediction(values: Any, source: str) -> np.ndarray:
    try:
        prediction = np.atleast_1d(np.asarray(values, dtype=float)).reshape(-1)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{source} must return finite numeric predictions") from exc
    if prediction.size == 0 or not np.all(np.isfinite(prediction)):
        raise ValueError(f"{source} must return at least one finite prediction")
    return prediction


def _validate_alpha(alpha: float) -> float:
    if (
        isinstance(alpha, (bool, np.bool_))
        or not isinstance(alpha, (int, float, np.integer, np.floating))
        or not np.isfinite(alpha)
        or not 0.0 < float(alpha) < 1.0
    ):
        raise ValueError("alpha must be a finite number strictly between 0 and 1")
    return float(alpha)


def _uq_mixle(model: Any, data: Any) -> UQResult:
    from mixle.inference.blackbox import laplace_posterior

    if data is None:
        raise ValueError("uq(mixle_model, data): fitting data is needed to build the Laplace approximation")
    post = laplace_posterior(model, list(data))
    return UQResult(
        kind="parameter_likelihood_approximation",
        method="laplace likelihood curvature (unconstrained Gaussian over parameters)",
        payload={"approximation": post, "n_params": int(len(post.u_mode)), "metadata": post.summary()},
    )


def _uq_point(predictor: Any, data: Any, alpha: float) -> UQResult:
    from mixle.inference.conformal import split_conformal

    if not (isinstance(data, tuple) and len(data) == 2):
        raise ValueError("uq(predictor, data): pass data=(X_cal, y_cal) -- calibration inputs and responses")
    x_cal, y_cal = data
    members = predictor if isinstance(predictor, (list, tuple)) else None
    if members:
        predicts = [_as_predict(m) for m in members]

        def predict(x: Any) -> np.ndarray:
            return np.mean([p(x) for p in predicts], axis=0)
    else:
        predict = _as_predict(predictor)

    x_cal = list(x_cal)
    cal_y = np.asarray(list(y_cal), dtype=float)
    if len(x_cal) == 0 or cal_y.ndim != 1 or len(cal_y) != len(x_cal):
        raise ValueError("X_cal and y_cal must be non-empty and contain the same number of rows")
    if not np.all(np.isfinite(cal_y)):
        raise ValueError("y_cal must contain only finite values")
    rows = [predict(xi) for xi in x_cal]
    if any(len(row) != 1 for row in rows):
        raise ValueError("regression predictors must return exactly one value per calibration row")
    cal_pred = np.asarray([row[0] for row in rows], dtype=float)
    lo, hi = split_conformal(cal_pred, cal_y, cal_pred, alpha=alpha)
    qhat = float((hi - cal_pred).mean())
    return UQResult(
        kind="ensemble_regressor" if members else "conformal_regressor",
        method=("deep ensemble + split conformal" if members else "split conformal"),
        payload={
            "predict": predict,
            "members": [_as_predict(m) for m in members] if members else None,
            "cal_pred": cal_pred,
            "cal_y": cal_y,
            "qhat": qhat,
            "alpha": alpha,
            # IN-SAMPLE by construction: this is the band's hit rate on the very rows whose
            # residual quantile set qhat, so it is >= 1 - alpha mechanically and is NOT evidence
            # of serving coverage (the MXR-080-1891 class); it is kept only as a sanity receipt
            # that the band was assembled correctly. Measure real coverage on disjoint rows.
            "coverage_cal_in_sample_mechanical": float(
                np.mean((cal_y >= cal_pred - qhat) & (cal_y <= cal_pred + qhat))
            ),
        },
    )


def _uq_llm(
    generate: Callable[[Any], Any], data: Any, alpha: float, equivalent: Callable[[Any, Any], bool] | None
) -> UQResult:
    from mixle.inference.uncertainty import semantic_entropy

    # calibrate an abstention threshold from example prompts, if given: the (1-alpha) quantile of
    # semantic entropy over the calibration prompts becomes the "too uncertain" cutoff.
    max_entropy = None
    if data is not None:
        ents = [semantic_entropy([generate(p) for _ in range(8)], equivalent) for p in data]
        if not ents:
            raise ValueError("LLM uncertainty calibration requires at least one prompt")
        if not np.all(np.isfinite(ents)):
            raise ValueError("semantic entropy calibration produced a non-finite value")
        max_entropy = float(np.quantile(ents, 1.0 - alpha))
    return UQResult(
        kind="llm_semantic",
        method="semantic entropy over meaning classes",
        payload={"generate": generate, "equivalent": equivalent, "max_entropy": max_entropy, "alpha": alpha},
    )


def uq(
    thing: Any,
    data: Any = None,
    *,
    alpha: float = 0.1,
    equivalent: Callable[[Any, Any], bool] | None = None,
) -> UQResult:
    """Quantify the uncertainty of ``thing``, choosing the method from what ``thing`` is.

    Args:
        thing: a fitted mixle model, a torch module / point-predictor callable (or a list of them for
            a deep ensemble), or an LLM-style callable that maps a prompt to a generation.
        data: for a mixle model, the fitting data (builds a Laplace likelihood-curvature approximation); for a point
            predictor, ``(X_cal, y_cal)`` calibration data; for an LLM, optional example prompts used
            to calibrate an abstention threshold.
        alpha: target miscoverage / abstention level (``1 - alpha`` coverage).
        equivalent: for the LLM path, an optional meaning-equivalence predicate over generations
            (default: exact string match after stripping).

    Returns:
        A :class:`UQResult` exposing the method-appropriate accessors and its own calibration numbers.
    """
    alpha = _validate_alpha(alpha)
    if _is_mixle_model(thing):
        return _uq_mixle(thing, data)
    if isinstance(thing, (list, tuple)) and thing and (_is_torch_module(thing[0]) or callable(thing[0])):
        # a list of predictors -> ensemble, UNLESS it is plainly (X, y) calibration data mistakenly passed here
        return _uq_point(thing, data, alpha)
    if _is_torch_module(thing):
        return _uq_point(thing, data, alpha)
    if callable(thing):
        # a bare callable is ambiguous: an array point-predictor (data is (X, y)) vs an LLM generator.
        if isinstance(data, tuple) and len(data) == 2:
            return _uq_point(thing, data, alpha)
        return _uq_llm(thing, data if data is None else list(data), alpha, equivalent)
    raise TypeError(
        f"uq() does not know how to quantify uncertainty for {type(thing).__name__}; pass a fitted "
        "mixle model, a torch module / predictor callable, or an LLM-style generation callable"
    )
