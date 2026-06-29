"""B4 — the extensibility seams: add inference / families / backends without editing dispatch.

mixle is built around a few small public registries. A new inference algorithm, distribution family,
or data backend is *added*, never patched into a central switch. This file demonstrates the inference
seam end-to-end (runnable) and documents the others with accurate references.

  1. register_fitter        -- a new `how=` inference route            (mixle.ppl.core)   [RUNNABLE here]
  2. register_family        -- a new PPL distribution (5-part contract) (mixle.ppl.core)   [see _lowering.py]
  3. capability ABCs         -- declare what a family supports           (mixle.stats.compute)
  4. register_encoded_data_backend -- a new encoded-data backend         (mixle.utils.parallel)

Run: ``python examples/extensibility_seams_example.py``
"""

from __future__ import annotations

import numpy as np

from mixle.ppl import Normal, free
from mixle.ppl.core import RandomVariable, register_fitter


# ---- Seam 1: register_fitter -- add a custom inference route reachable via fit(how=...) -------------
@register_fitter("mom")
def _method_of_moments(rv: RandomVariable, data, **_):
    """A method-of-moments fitter for Normal(free, free): match the sample mean and variance.

    A fitter receives the RandomVariable being fit and the data, and returns a *bound* RandomVariable
    (the fitted distribution + an optional posterior result). Registering it makes ``fit(how='mom')``
    work for any model -- no change to the auto-router or the fit dispatch."""
    from mixle.stats import GaussianDistribution

    x = np.asarray(data, dtype=float)
    return RandomVariable._bound(GaussianDistribution(x.mean(), x.var(), name=rv._name), name=rv._name)


def demo_register_fitter():
    rng = np.random.RandomState(0)
    data = list(rng.normal(5.0, 2.0, 4000))
    m = Normal(free, free).fit(data, how="mom")  # dispatches to the freshly registered fitter
    print(
        "1. register_fitter  -> custom how='mom' fitted:",
        round(float(m.dist.mu), 2),
        round(float(m.dist.sigma2) ** 0.5, 2),
        "(true 5, 2)",
    )


# ---- Seams 2-4: documented with accurate references ------------------------------------------------
SEAMS_DOC = """
2. register_family (mixle.ppl.core) -- expose a stats distribution (whose 5-part contract --
   Distribution / Sampler / Estimator / Accumulator / DataEncoder -- already exists) as a PPL family.
   The PPL's own families register themselves this way; see mixle/ppl/_lowering.py, e.g. (real call):

       register_family(
           "Normal",
           GaussianDistribution,                                   # the Distribution class
           GaussianEstimator,                                      # its Estimator
           lambda mean, sd: {"mu": float(mean), "sigma2": float(sd) ** 2},  # args -> dist kwargs
           arity=2,
           seed_at=lambda v, s: {"mu": float(v), "sigma2": (float(s) ** 2) or 1.0},
           positive=(False, True),                                 # which slots must be > 0
           read=lambda d: {"mean": d.mu, "sd": float(np.sqrt(d.sigma2))},
       )

   A new family is added by one register_family(...) call; nothing in the dispatch path changes.

3. capability ABCs (mixle.stats.compute) -- a family advertises what it supports by subclassing the
   capability protocols (ExactDensity, ConjugateUpdatable, EngineResidentEStep, ...) and/or declaring
   compute_capabilities(). describe(x) / supports(x, Cap) read these, and the auto-router uses them.
   This is how `engine_ready=(...)` and the conjugate/route choices are surfaced -- see how
   describe(model) reports the fit route.

4. register_encoded_data_backend (mixle.utils.parallel) -- register a new encoded-data backend (an
   alternative to the local/mp/mpi/spark/dask/ray backends) so optimize(..., backend='yours') routes
   to it. The planner consumes it through the same interface; no fit-loop changes.
"""


def main():
    print("# mixle extensibility seams (add, don't patch)\n")
    demo_register_fitter()
    print(SEAMS_DOC)
    print(
        "Each seam is a registry: new inference, families, capabilities, and backends are *registered*, "
        "so dispatch (the auto-router, the fit loop, the planner) never needs editing to extend mixle."
    )


if __name__ == "__main__":
    main()
