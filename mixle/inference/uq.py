"""Unified uncertainty dispatch for models, predictors, ensembles, and LLM-style callables.

``uq(thing, data)`` inspects the object it receives and routes to a compatible
uncertainty method, returning a :class:`UQResult` with the method name and the
quantities needed for downstream checks.

  * a fitted mixle model (has ``seq_log_density``) -> a Laplace parameter posterior; sample fitted
    models, read any summary, get a credible interval (epistemic uncertainty over parameters).
  * a torch module / any point predictor callable over arrays -> split-conformal calibration from a
    held-out ``(X, y)``; ``interval(x)`` returns a prediction interval with finite-sample coverage.
    Give a LIST of predictors instead and it becomes a deep ensemble (epistemic spread + conformal).
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

    kind: str  # 'parameter_posterior' | 'conformal_regressor' | 'ensemble_regressor' | 'llm_semantic'
    method: str  # human-readable method name
    payload: dict[str, Any]

    # -- mixle model: Laplace parameter posterior --------------------------------------------------
    def sample_models(self, n: int = 200, *, seed: int | None = None) -> list[Any]:
        """``n`` fitted models drawn from the parameter posterior (epistemic ensemble)."""
        post = self.payload["posterior"]
        rng = np.random.RandomState(seed) if seed is not None else None
        return post.sample(int(n), rng=rng)

    def credible_interval(
        self, readout: Callable[[Any], float], alpha: float = 0.1, *, n: int = 400, seed: int = 0
    ) -> tuple[float, float]:
        """A ``1-alpha`` credible interval on ``readout(model)`` over the parameter posterior."""
        vals = np.asarray([float(readout(m)) for m in self.sample_models(n, seed=seed)], dtype=float)
        lo, hi = np.quantile(vals, [alpha / 2.0, 1.0 - alpha / 2.0])
        return float(lo), float(hi)

    # -- point predictor: split conformal (single or ensemble) -------------------------------------
    def interval(self, x: Any, alpha: float | None = None) -> tuple[np.ndarray, np.ndarray]:
        """Calibrated prediction interval(s) at ``x``. ``alpha`` overrides the calibrated level."""
        from mixle.inference.conformal import split_conformal

        predict = self.payload["predict"]
        q = self.payload["qhat"] if alpha is None else None
        pred = np.atleast_1d(predict(x))
        if q is not None:
            return pred - q, pred + q
        lo, hi = split_conformal(self.payload["cal_pred"], self.payload["cal_y"], pred, alpha=float(alpha))
        return lo, hi

    def epistemic_std(self, x: Any) -> np.ndarray:
        """Ensemble disagreement (std across members) at ``x`` -- 0.0 for a single predictor."""
        members = self.payload.get("members")
        if not members:
            return np.zeros(np.atleast_1d(self.payload["predict"](x)).shape)
        preds = np.stack([np.atleast_1d(m(x)) for m in members])
        return preds.std(axis=0)

    # -- LLM callable: semantic entropy ------------------------------------------------------------
    def semantic_entropy(self, prompt: Any, *, n: int = 8) -> float:
        """Entropy (nats) over the meaning classes of ``n`` sampled generations for ``prompt``."""
        from mixle.inference.uncertainty import semantic_entropy as _se

        gen = self.payload["generate"]
        equivalent = self.payload.get("equivalent")
        samples = [gen(prompt) for _ in range(int(n))]
        return float(_se(samples, equivalent))

    def confident(self, prompt: Any, *, n: int = 8, max_entropy: float | None = None) -> bool:
        """True when semantic entropy is below the threshold -- else the model disagrees with itself."""
        thr = self.payload["max_entropy"] if max_entropy is None else float(max_entropy)
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
            model.eval()
            with torch.no_grad():
                xt = torch.as_tensor(np.atleast_2d(np.asarray(x, dtype=float)), dtype=torch.float32)
                out = model(xt)
                return np.asarray(out).reshape(-1) if not hasattr(out, "numpy") else out.cpu().numpy().reshape(-1)

        return predict
    return lambda x: np.atleast_1d(np.asarray(model(x), dtype=float)).reshape(-1)


def _uq_mixle(model: Any, data: Any) -> UQResult:
    from mixle.inference.blackbox import laplace_posterior

    if data is None:
        raise ValueError("uq(mixle_model, data): the fitting data is needed to build the Laplace posterior")
    post = laplace_posterior(model, list(data))
    return UQResult(
        kind="parameter_posterior",
        method="laplace (unconstrained Gaussian over parameters)",
        payload={"posterior": post, "n_params": int(len(post.u_mode))},
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

    cal_pred = np.asarray([float(predict(xi)[0]) for xi in x_cal], dtype=float)
    cal_y = np.asarray([float(v) for v in y_cal], dtype=float)
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
            "coverage_cal": float(np.mean((cal_y >= cal_pred - qhat) & (cal_y <= cal_pred + qhat))),
        },
    )


def _uq_llm(
    generate: Callable[[Any], Any], data: Any, alpha: float, equivalent: Callable[[Any, Any], bool] | None
) -> UQResult:
    from mixle.inference.uncertainty import semantic_entropy

    # calibrate an abstention threshold from example prompts, if given: the (1-alpha) quantile of
    # semantic entropy over the calibration prompts becomes the "too uncertain" cutoff.
    max_entropy = float("inf")
    if data is not None:
        ents = [semantic_entropy([generate(p) for _ in range(8)], equivalent) for p in data]
        if ents:
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
        data: for a mixle model, the fitting data (builds the Laplace posterior); for a point
            predictor, ``(X_cal, y_cal)`` calibration data; for an LLM, optional example prompts used
            to calibrate an abstention threshold.
        alpha: target miscoverage / abstention level (``1 - alpha`` coverage).
        equivalent: for the LLM path, an optional meaning-equivalence predicate over generations
            (default: exact string match after stripping).

    Returns:
        A :class:`UQResult` exposing the method-appropriate accessors and its own calibration numbers.
    """
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
