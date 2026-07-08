"""Compatibility wrapper for Torch-backed local model evaluation.

Historically this module contained one Torch implementation class per
distribution family.  That made Torch support an omni-file and duplicated the
math already owned by the distributions.  The remaining ``TorchMixture`` class
is a small adapter over the modular compute-engine stack:

* data are encoded with the model's normal ``dist_to_encoder`` protocol;
* scoring and accumulation dispatch through ``dist.kernel(engine=...)`` when a
  Torch kernel is available;
* unsupported object-valued models fall back to the legacy ``seq_*`` protocol
  as fixed, CPU-scored compatibility paths;
* gradient fitting delegates to the declaration/objective-based generic
  optimizers in ``mixle.inference.estimation``.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from mixle.engines import TorchEngine
from mixle.inference.gradient_fit import fit_map as _fit_map
from mixle.inference.gradient_fit import fit_mle as _fit_mle
from mixle.stats.compute.pdist import ParameterEstimator, SequenceEncodableProbabilityDistribution

__all__ = ["TorchMixture"]


class TorchMixture:
    """Thin compatibility adapter over ``ComputeEngine`` kernels.

    New code should prefer ``dist.kernel(engine=TorchEngine(...))``,
    ``optimize(..., engine=...)``, ``fit_mle`` / ``fit_map``, or
    ``mixle.inference.objectives`` directly.  This class exists so older code and
    compatibility tests importing ``mixle.stats.compute.torch_mixture.TorchMixture`` keep
    working while distribution math remains distribution-owned.
    """

    def __init__(self, model: SequenceEncodableProbabilityDistribution, device: str = "cpu", dtype: Any = None) -> None:
        self.model = model
        self.engine = TorchEngine(device=device, dtype=dtype)
        self.device = self.engine.device
        self.dtype = self.engine.dtype
        self.is_mixture = _is_mixture_model(model)
        self.K = model.num_components if self.is_mixture else 1

    def encode(self, data: Any) -> tuple[int, Any]:
        """Encode observations using the model's canonical sequence encoder."""
        payload = self.model.dist_to_encoder().seq_encode(data)
        return len(data), payload

    def seq_component_log_density(self, enc: tuple[int, Any], model: Any | None = None) -> np.ndarray:
        """Return component log-density matrix as a NumPy array."""
        model = self.model if model is None else model
        self._validate_ignored_structure(model)
        payload = enc[1]
        if _is_mixture_model(model):
            try:
                scores = model.kernel(engine=self.engine).component_scores(payload)
                return np.asarray(self.engine.to_numpy(scores), dtype=np.float64)
            except Exception:
                # Fall back to the legacy numpy path if the model has no modular kernel for this
                # engine (or the kernel raises). This intentionally swallows kernel errors; set a
                # breakpoint here when debugging a silently-degraded engine path.
                return np.asarray(model.seq_component_log_density(payload), dtype=np.float64)
        scores = self._score(model, payload)
        return scores.reshape((-1, 1))

    def seq_log_density(self, enc: tuple[int, Any], model: Any | None = None) -> np.ndarray:
        """Return per-row model log densities as a NumPy array."""
        model = self.model if model is None else model
        self._validate_ignored_structure(model)
        return self._score(model, enc[1])

    def posteriors(self, enc: tuple[int, Any], model: Any | None = None) -> Any:
        """Return posterior component weights as a Torch tensor."""
        model = self.model if model is None else model
        self._validate_ignored_structure(model)
        n = int(enc[0])
        if not _is_mixture_model(model):
            return self.engine.asarray(np.ones((n, 1), dtype=np.float64))
        payload = enc[1]
        try:
            kernel = model.kernel(engine=self.engine)
            if callable(getattr(kernel, "posteriors", None)):
                return kernel.posteriors(payload)
            comp = kernel.component_scores(payload) + self.engine.asarray(model.log_w)[None, :]
            denom = self.engine.logsumexp(comp, axis=1)
            return self.engine.exp(comp - denom[:, None])
        except Exception:
            # Fall back to the legacy numpy posterior if no modular kernel is available (or it
            # raises). Intentionally swallows kernel errors to keep the engine path optional.
            return self.engine.asarray(model.seq_posterior(payload))

    def weighted_suff_stats(self, enc: tuple[int, Any], gamma: Any, model: Any | None = None) -> Any:
        """Return legacy-format sufficient statistics for posterior weights."""
        model = self.model if model is None else model
        if _is_mixture_model(model):
            gamma_np = np.asarray(self.engine.to_numpy(gamma), dtype=np.float64)
            comp_counts = gamma_np.sum(axis=0)
            comp_stats = []
            for i, component in enumerate(model.components):
                acc = component.estimator().accumulator_factory().make()
                acc.seq_update(enc[1], gamma_np[:, i], component)
                comp_stats.append(acc.value())
            return comp_counts, tuple(comp_stats)

        weights = np.asarray(self.engine.to_numpy(gamma), dtype=np.float64)
        if weights.ndim == 2:
            weights = weights[:, 0]
        acc = model.estimator().accumulator_factory().make()
        acc.seq_update(enc[1], weights, model)
        return acc.value()

    def em_step(
        self,
        enc: tuple[int, Any],
        estimator: ParameterEstimator,
        model: Any | None = None,
        weights: Any | None = None,
    ) -> Any:
        """Run one EM M-step using modular kernels when possible."""
        model = self.model if model is None else model
        n, payload = enc
        row_weights = (
            np.ones(int(n), dtype=np.float64)
            if weights is None
            else np.asarray(self.engine.to_numpy(weights), dtype=np.float64)
        )
        try:
            stats = model.kernel(engine=self.engine, estimator=estimator).accumulate(
                payload, self.engine.asarray(row_weights)
            )
        except Exception:
            # Fall back to the legacy accumulator M-step if no modular kernel exists for this
            # (model, estimator, engine) combination (or it raises). Swallows kernel errors by design.
            acc = estimator.accumulator_factory().make()
            acc.seq_update(payload, row_weights, model)
            stats = acc.value()
        return estimator.estimate(n, stats)

    def initialize(
        self, enc: tuple[int, Any], estimator: ParameterEstimator, rng: np.random.RandomState, p: float = 0.1
    ) -> Any:
        """Initialize through the standard sequence-initialization protocol."""
        from mixle.inference import seq_initialize

        return seq_initialize([enc], estimator, rng, p)

    def fit(
        self,
        enc: tuple[int, Any],
        estimator: ParameterEstimator,
        max_its: int = 100,
        delta: float = 1.0e-8,
        rng: np.random.RandomState | None = None,
        init_p: float = 0.1,
        model: Any | None = None,
        out: Any | None = None,
    ) -> tuple[Any, float]:
        """Run local EM to convergence and return ``(model, log_likelihood)``."""
        model = self.initialize(enc, estimator, rng or np.random.RandomState(0), p=init_p) if model is None else model
        old_ll = float(self.seq_log_density(enc, model=model).sum())
        for i in range(max(1, int(max_its))):
            model = self.em_step(enc, estimator, model=model)
            ll = float(self.seq_log_density(enc, model=model).sum())
            if out is not None:
                out.write("Iteration %d: ln[p(Data|Model)]=%e, delta=%e\n" % (i + 1, ll, ll - old_ll))
            if abs(ll - old_ll) < delta:
                old_ll = ll
                break
            old_ll = ll
        return model, old_ll

    def fit_mle(
        self,
        enc: tuple[int, Any],
        model: Any | None = None,
        max_its: int = 500,
        lr: float = 0.05,
        optimizer: str = "adam",
        tol: float = 1.0e-7,
        out: Any | None = None,
        print_iter: int = 100,
        return_result: bool = False,
    ) -> Any:
        """Delegate gradient MLE to the generic declaration-backed fitter."""
        return _fit_mle(
            enc[1],
            self.model if model is None else model,
            engine=self.engine,
            max_its=max_its,
            lr=lr,
            optimizer=optimizer,
            tol=tol,
            out=out,
            print_iter=print_iter,
            return_result=return_result,
        )

    def fit_map(
        self,
        enc: tuple[int, Any],
        model: Any | None = None,
        priors: Any | None = None,
        prior_strength: float = 1.0,
        w_alpha: float | None = None,
        max_its: int = 500,
        lr: float = 0.05,
        optimizer: str = "adam",
        tol: float = 1.0e-7,
        out: Any | None = None,
        print_iter: int = 100,
        return_result: bool = False,
    ) -> Any:
        """Delegate MAP fitting to the generic objective/declaration path.

        ``priors`` and ``w_alpha`` are accepted for source compatibility; rich
        conjugate priors now belong in distribution/objective declarations
        rather than this compatibility shim.
        """
        return _fit_map(
            enc[1],
            self.model if model is None else model,
            engine=self.engine,
            prior_strength=prior_strength,
            priors=priors,
            max_its=max_its,
            lr=lr,
            optimizer=optimizer,
            tol=tol,
            out=out,
            print_iter=print_iter,
            return_result=return_result,
        )

    def _score(self, model: Any, payload: Any) -> np.ndarray:
        try:
            scores = model.kernel(engine=self.engine).score(payload)
            return np.asarray(self.engine.to_numpy(scores), dtype=np.float64)
        except Exception:
            # Fall back to the legacy numpy scorer when no modular kernel is available (or it
            # raises). Intentionally swallows kernel errors to keep the engine path optional.
            return np.asarray(model.seq_log_density(payload), dtype=np.float64)

    def _validate_ignored_structure(self, model: Any) -> None:
        if _ignored_signature(model) != _ignored_signature(self.model):
            raise ValueError("TorchMixture encoded data are tied to the wrapped IgnoredDistribution structure.")


def _ignored_signature(model: Any) -> Any:
    if _has_type_name(model, "IgnoredDistribution") and hasattr(model, "dist"):
        return ("ignored", str(model.dist))
    if _has_type_name(model, "CompositeDistribution") and hasattr(model, "dists"):
        return ("composite", tuple(_ignored_signature(d) for d in model.dists))
    if _is_mixture_model(model):
        return ("mixture", tuple(_ignored_signature(d) for d in model.components))
    if _has_type_name(model, "OptionalDistribution") and hasattr(model, "dist"):
        return ("optional", _ignored_signature(model.dist))
    if _has_type_name(model, "SequenceDistribution") and hasattr(model, "dist") and hasattr(model, "len_dist"):
        return ("sequence", _ignored_signature(model.dist), _ignored_signature(model.len_dist))
    return None


def _is_mixture_model(model: Any) -> bool:
    return bool(
        hasattr(model, "components")
        and hasattr(model, "log_w")
        and hasattr(model, "num_components")
        and callable(getattr(model, "seq_component_log_density", None))
        and callable(getattr(model, "seq_posterior", None))
    )


def _has_type_name(model: Any, name: str) -> bool:
    return any(cls.__name__ == name for cls in type(model).mro())
