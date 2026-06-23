"""The ``Posterior`` algebra: parameter / predictive posteriors + the ``posterior()`` factory.

*Inference produces posteriors; you draw from them through one interface* -- the
:class:`~pysp.stats.compute.posterior.Posterior` base in the compute layer. The latent ``q(z | x)``
realizations live there with the base (they need no inference machinery); the realizations here are
the ones that *do* need it:

* :class:`ParameterPosterior` -- ``q(theta | data)``, closed-form when the family is conjugate and
  MCMC otherwise, behind one ``sample`` / ``samples`` / ``mean`` / ``interval`` interface;
* :class:`PredictivePosterior` -- draws of *new* data from a fitted model (plug-in), or with
  parameter uncertainty integrated in via :meth:`PredictivePosterior.from_parameter_posterior`.

:func:`posterior` is the front door that picks the right realization from ``over=`` and the family.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

import numpy as np
from numpy.random import RandomState

from pysp.engines.arithmetic import maxrandint
from pysp.inference.mcmc.parameter_bridge import sample_parameter_posterior
from pysp.stats.bayes.conjugate import ConjugatePosterior, conjugate_posterior, is_conjugate_family
from pysp.stats.compute.posterior import Posterior

__all__ = ["ParameterPosterior", "PredictivePosterior", "posterior"]


def _as_rng(rng: Any) -> RandomState:
    return rng if isinstance(rng, RandomState) else RandomState(rng)


def _seed_from(rng: Any) -> int:
    return int(_as_rng(rng).randint(maxrandint))


class ParameterPosterior(Posterior):
    """``q(theta | data)`` over a model family's parameters -- exact (conjugate) or MCMC.

    Built by ``posterior(model, data, over="params")``; the exact-vs-MCMC distinction is hidden behind
    the shared :class:`Posterior` interface. A single :meth:`sample` returns one parameter set, and
    :meth:`samples` returns ``n`` of them; :meth:`mean` and :meth:`interval` summarize the posterior.
    """

    def __init__(
        self,
        draw_one: Callable[[Any], Any],
        draw_many: Callable[[int, Any], Any],
        *,
        mean_fn: Callable[[], Any] | None = None,
        chain: np.ndarray | None = None,
        kind: str = "",
    ) -> None:
        self._draw_one = draw_one
        self._draw_many = draw_many
        self._mean_fn = mean_fn
        self._chain = chain  # (n_samples, dim) numeric chain for summaries (MCMC); None for conjugate
        self.kind = kind

    @classmethod
    def from_conjugate(cls, cp: ConjugatePosterior) -> ParameterPosterior:
        """Wrap a closed-form :class:`ConjugatePosterior` (each draw is a parameter dict)."""

        def one(rng: Any) -> Any:
            return cp.sampler(_seed_from(rng)).sample(None)

        def many(n: int, rng: Any) -> Any:
            return cp.sample(int(n), _as_rng(rng))

        return cls(one, many, mean_fn=cp.mean, kind="conjugate")

    @classmethod
    def from_mcmc(cls, result: Any) -> ParameterPosterior:
        """Wrap an MCMC ``MCMCResult``; draws resample the retained chain, summaries use it directly."""
        samples = list(result.samples)
        if not samples:
            raise ValueError("MCMC result has no retained samples to form a posterior.")
        try:
            chain = np.asarray(samples, dtype=float)  # (n_samples, dim) parameter-space chain
        except (ValueError, TypeError):
            chain = None  # non-numeric samples (e.g. rebuilt distributions): summaries unavailable

        def one(rng: Any) -> Any:
            return samples[_as_rng(rng).randint(len(samples))]

        def many(n: int, rng: Any) -> Any:
            idx = _as_rng(rng).randint(len(samples), size=int(n))
            return [samples[i] for i in idx]

        return cls(one, many, chain=chain, kind="mcmc")

    def sample(self, rng: Any = None) -> Any:
        """One parameter draw from the posterior."""
        return self._draw_one(rng)

    def samples(self, n: int, rng: Any = None) -> Any:
        """``n`` parameter draws (a dict of length-``n`` arrays for conjugate; a list for MCMC)."""
        return self._draw_many(n, rng)

    def mean(self) -> Any:
        """The posterior mean of the parameters."""
        if self._mean_fn is not None:
            return self._mean_fn()
        if self._chain is None:
            raise NotImplementedError("mean() unavailable for non-numeric MCMC samples")
        return self._chain.mean(axis=0)

    def interval(self, level: float = 0.9) -> Any:
        """Central credible interval at ``level`` -- ``[lo, hi]`` over the chain (MCMC) or 2000 draws."""
        lo, hi = (1.0 - level) / 2.0, 1.0 - (1.0 - level) / 2.0
        if self._chain is not None:
            return np.quantile(self._chain, [lo, hi], axis=0)
        draws = self._draw_many(2000, 0)
        return {k: np.quantile(np.asarray(v, dtype=float), [lo, hi], axis=0) for k, v in draws.items()}


class PredictivePosterior(Posterior):
    """The posterior-predictive: draws of *new* data from a fitted model.

    :meth:`plug_in` wraps a single fitted model's sampler. :meth:`from_parameter_posterior` integrates
    parameter uncertainty -- each predictive draw first samples ``theta ~ q(theta | data)``, rebuilds
    the model, then draws data -- so the spread reflects both sampling and parameter uncertainty.
    """

    def __init__(self, draw_one: Callable[[Any], Any], draw_many: Callable[[int, Any], Any]) -> None:
        self._draw_one = draw_one
        self._draw_many = draw_many

    @classmethod
    def plug_in(cls, model: Any) -> PredictivePosterior:
        """Plug-in predictive: draw new data from ``model`` at its fitted parameters."""
        if not callable(getattr(model, "sampler", None)):
            raise TypeError(f"{type(model).__name__} is not samplable (no .sampler()).")

        def one(rng: Any) -> Any:
            return model.sampler(_seed_from(rng)).sample()

        def many(n: int, rng: Any) -> Any:
            return model.sampler(_seed_from(rng)).sample(int(n))

        return cls(one, many)

    @classmethod
    def from_parameter_posterior(
        cls, param_post: ParameterPosterior, build: Callable[[Any], Any]
    ) -> PredictivePosterior:
        """Posterior-predictive integrating parameter uncertainty.

        ``build`` maps one parameter draw (the object :meth:`ParameterPosterior.sample` returns) to a
        distribution; each predictive draw rebuilds the model from a fresh ``theta`` and samples it.
        """

        def one(rng: Any) -> Any:
            r = _as_rng(rng)
            model = build(param_post.sample(r))
            return model.sampler(_seed_from(r)).sample()

        def many(n: int, rng: Any) -> Any:
            r = _as_rng(rng)
            return [one(r) for _ in range(int(n))]

        return cls(one, many)

    def sample(self, rng: Any = None) -> Any:
        """One predictive draw of new data."""
        return self._draw_one(rng)

    def samples(self, n: int, rng: Any = None) -> Any:
        """``n`` predictive draws of new data."""
        return self._draw_many(n, rng)


def posterior(
    model: Any,
    data: Any = None,
    *,
    over: str = "predictive",
    prior: Any = None,
    method: str = "auto",
    **kwargs: Any,
) -> Posterior:
    """Build the :class:`Posterior` of ``model`` over the requested variables.

    Args:
        model: a fitted pysp distribution (or a latent-variable model for ``over='latent'``).
        data: observations -- required for ``over='params'`` and for ``over='latent'`` (the ``x`` the
            latent posterior conditions on); ignored for plug-in ``over='predictive'``.
        over: ``'latent'`` -> ``q(z | x)`` (needs the ``latent_posterior`` capability);
            ``'params'`` -> ``q(theta | data)``; ``'predictive'`` -> draws of new data.
        prior: prior over parameters for ``over='params'`` (see ``conjugate_posterior`` /
            ``sample_parameter_posterior``).
        method: for ``over='params'`` -- ``'auto'`` (conjugate when the family supports it, else MCMC),
            ``'conjugate'``, or ``'mcmc'``.
        **kwargs: forwarded to ``sample_parameter_posterior`` for the MCMC path (``sampler``, ``steps``...).

    Returns:
        A :class:`Posterior` -- a ``LatentPosterior``, :class:`ParameterPosterior`, or
        :class:`PredictivePosterior`.
    """
    if over == "latent":
        if not callable(getattr(model, "latent_posterior", None)):
            raise TypeError(f"{type(model).__name__} has no latent_posterior(x); over='latent' needs it.")
        return model.latent_posterior(data)

    if over == "params":
        if data is None:
            raise ValueError("over='params' requires data to form q(theta | data).")
        if method not in ("auto", "conjugate", "mcmc"):
            raise ValueError(f"unknown method {method!r}; expected 'auto', 'conjugate', or 'mcmc'.")
        conj = is_conjugate_family(model)
        if method == "conjugate" and not conj:
            raise TypeError(f"{type(model).__name__} is not a conjugate family; use method='mcmc'.")
        if method == "conjugate" or (method == "auto" and conj):
            return ParameterPosterior.from_conjugate(conjugate_posterior(model, data, prior=prior))
        return ParameterPosterior.from_mcmc(sample_parameter_posterior(model, data, prior=prior, **kwargs))

    if over == "predictive":
        return PredictivePosterior.plug_in(model)

    raise ValueError(f"unknown over={over!r}; expected 'latent', 'params', or 'predictive'.")
