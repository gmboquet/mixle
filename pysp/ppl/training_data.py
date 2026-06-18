"""Generate (dataset -> pysp.ppl model code) training examples for the model-writing LLM (WS-I).

Because pysp can *sample* from any model, the generative library is a source of **free, verifiable
labels**: draw a model family, sample a dataset from a concrete instance, and emit the canonical
``pysp.ppl`` code (with ``free`` parameters) that a model-writing assistant should produce for that
dataset. :func:`build_model_from_code` executes such code back into a fittable model, so every
generated pair can be checked to round-trip (the emitted model fits its own data), and the same
helper doubles as the scorer for an evaluation harness.

This is the data foundation for WS-I; the fine-tuning / serving layer is a separate later phase.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

from numpy.random import RandomState


@dataclass(frozen=True)
class ModelingExample:
    """One supervised example: a dataset and the target ``pysp.ppl`` model code that explains it."""

    family: str
    task: str
    code: str
    data: list[Any]


def _seed(rng: RandomState) -> int:
    return int(rng.randint(0, 2**31 - 1))


def _gaussian(rng: RandomState, n: int) -> ModelingExample:
    from pysp.stats import GaussianDistribution

    mu = float(rng.uniform(-5.0, 5.0))
    sigma = float(rng.uniform(0.5, 3.0))
    data = list(GaussianDistribution(mu, sigma * sigma).sampler(_seed(rng)).sample(n))
    code = "from pysp.ppl import Normal, free\nmodel = Normal(free, free)"
    return ModelingExample("gaussian", f"Fit a univariate Gaussian to {n} real-valued observations.", code, data)


def _exponential(rng: RandomState, n: int) -> ModelingExample:
    from pysp.stats import ExponentialDistribution

    rate = float(rng.uniform(0.3, 3.0))
    data = list(ExponentialDistribution(rate).sampler(_seed(rng)).sample(n))
    code = "from pysp.ppl import Exponential, free\nmodel = Exponential(free)"
    return ModelingExample("exponential", f"Fit an exponential distribution to {n} positive observations.", code, data)


def _poisson(rng: RandomState, n: int) -> ModelingExample:
    from pysp.stats import PoissonDistribution

    lam = float(rng.uniform(0.5, 12.0))
    data = [int(v) for v in PoissonDistribution(lam).sampler(_seed(rng)).sample(n)]
    code = "from pysp.ppl import Poisson, free\nmodel = Poisson(free)"
    return ModelingExample("poisson", f"Fit a Poisson distribution to {n} non-negative integer counts.", code, data)


def _gamma(rng: RandomState, n: int) -> ModelingExample:
    from pysp.stats import GammaDistribution

    k = float(rng.uniform(1.5, 6.0))
    theta = float(rng.uniform(0.5, 2.5))
    data = list(GammaDistribution(k, theta).sampler(_seed(rng)).sample(n))
    code = "from pysp.ppl import Gamma, free\nmodel = Gamma(free, free)"
    return ModelingExample("gamma", f"Fit a gamma distribution to {n} positive, right-skewed observations.", code, data)


def _gaussian_mixture(rng: RandomState, n: int) -> ModelingExample:
    from pysp.stats import GaussianDistribution, MixtureDistribution

    centers = sorted(rng.uniform(-7.0, 7.0, size=2))
    while centers[1] - centers[0] < 3.0:
        centers = sorted(rng.uniform(-7.0, 7.0, size=2))
    w0 = float(rng.uniform(0.3, 0.7))
    dist = MixtureDistribution(
        [GaussianDistribution(centers[0], 1.0), GaussianDistribution(centers[1], 1.0)], w=[w0, 1.0 - w0]
    )
    data = list(dist.sampler(_seed(rng)).sample(n))
    code = "from pysp.ppl import Mix, Normal, free\nmodel = Mix([Normal(free, free), Normal(free, free)])"
    return ModelingExample(
        "gaussian_mixture", f"Fit a two-component Gaussian mixture to {n} real-valued observations.", code, data
    )


_TEMPLATES: list[Callable[[RandomState, int], ModelingExample]] = [
    _gaussian,
    _exponential,
    _poisson,
    _gamma,
    _gaussian_mixture,
]


def families() -> list[str]:
    """Return the model families the generator currently covers."""
    return ["gaussian", "exponential", "poisson", "gamma", "gaussian_mixture"]


def generate_examples(
    num_examples: int, seed: int | None = None, n_obs: tuple[int, int] = (50, 300)
) -> list[ModelingExample]:
    """Generate ``num_examples`` (dataset, target-code) pairs, cycling through the model families.

    ``n_obs`` is the inclusive range for the per-example sample size.
    """
    rng = RandomState(seed)
    lo, hi = int(n_obs[0]), int(n_obs[1])
    examples = []
    for i in range(int(num_examples)):
        template = _TEMPLATES[i % len(_TEMPLATES)]
        n = int(rng.randint(lo, hi + 1))
        examples.append(template(rng, n))
    return examples


def build_model_from_code(code: str):
    """Execute generated ``pysp.ppl`` model code and return the bound ``model`` RandomVariable.

    The code is expected to import from ``pysp.ppl`` and assign a ``model`` variable (as emitted by
    :func:`generate_examples`). Used to verify that an example round-trips and as the scorer in an
    evaluation harness (``build_model_from_code(code).fit(data)``).
    """
    namespace: dict[str, Any] = {}
    exec(code, namespace)  # noqa: S102 - controlled, generator-emitted pysp.ppl code
    if "model" not in namespace:
        raise ValueError("generated code must assign a `model` variable.")
    return namespace["model"]


def fit_example(example: ModelingExample):
    """Fit ``example``'s target model to its data (verifies the pair round-trips)."""
    return build_model_from_code(example.code).fit(example.data)
