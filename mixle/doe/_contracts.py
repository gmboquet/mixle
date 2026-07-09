"""Structural (duck-typed) contracts for the DOE layer, as runtime-checkable Protocols (WS-E).

The DOE layer follows a "register, don't branch" pattern: acquisition functions go through
:func:`mixle.doe.bayesopt.register_acquisition`, optimality criteria through
:func:`mixle.doe.optimal.register_criterion`, and the GP surrogate is passed in as a duck-typed
``gp=`` argument. This module formalizes those three contracts as ``@runtime_checkable`` Protocols
so the registry value types and the ``gp=`` parameters can be annotated precisely (instead of bare
``Callable[..., ...]`` / ``Any``), and so a surrogate can be validated with ``isinstance``.

These are typing-level only: they add no behavior and the registries / call sites are unchanged.
"""

from __future__ import annotations

from typing import Any, Protocol, runtime_checkable

import numpy as np


@runtime_checkable
class Acquisition(Protocol):
    """The acquisition-function contract registered via ``register_acquisition`` (EI/PI/UCB).

    An acquisition scores candidate points from their surrogate posterior moments and returns a
    *merit* array that the proposal loop maximizes over the candidate set. It is called as
    ``fn(mean, std, best, *, maximize, **params)`` where ``mean`` / ``std`` are the predictive mean
    and standard deviation at the candidates, ``best`` is the incumbent objective value, ``maximize``
    selects the optimization sense, and ``**params`` carries per-acquisition knobs (e.g. ``xi`` for
    EI/PI, ``kappa`` for UCB). Built-ins :func:`mixle.doe.bayesopt.expected_improvement`,
    :func:`probability_of_improvement`, and :func:`upper_confidence_bound` satisfy this contract.
    """

    def __call__(self, mean: Any, std: Any, best: float, *, maximize: bool = ..., **params: Any) -> np.ndarray: ...


@runtime_checkable
class Surrogate(Protocol):
    """The GP-surrogate contract passed as ``gp=`` to the Bayesian-optimization loops.

    A surrogate is fit to the observed ``(x, y)`` and queried for the posterior predictive moments at
    new candidate points. The call convention is the one used by
    :class:`mixle.models.gaussian_process.GaussianProcessRegressor`: ``fit`` trains in place (it may
    return diagnostics, which the loops ignore), and ``predict`` takes the training data alongside the
    query points, returning the posterior mean (``return_cov=False``) or ``(mean, cov)`` pair
    (``return_cov=True``).
    """

    def fit(self, x: Any, y: Any, **kwargs: Any) -> Any:
        """Fit or update the surrogate from observed design points ``x`` and responses ``y``."""
        ...

    def predict(self, x_train: Any, y_train: Any, x_new: Any, return_cov: bool = ...) -> Any:
        """Return predictive moments at ``x_new`` using the observed training data."""
        ...


@runtime_checkable
class Criterion(Protocol):
    """The optimality-criterion contract registered via ``register_criterion`` (D/A/I-optimality).

    A criterion maps the information matrix ``M = F.T @ F`` to a scalar *merit* that
    :func:`mixle.doe.optimal.optimal_design` maximizes over candidate designs. It is called as
    ``fn(info, *, ref)`` where ``ref`` is an optional reference model matrix (used by I-optimality).
    Built-ins :func:`mixle.doe.optimal.d_criterion`, :func:`a_criterion`, and :func:`i_criterion`
    satisfy this contract.
    """

    def __call__(self, info: np.ndarray, *, ref: np.ndarray | None = ...) -> float: ...


__all__ = ["Acquisition", "Surrogate", "Criterion"]
