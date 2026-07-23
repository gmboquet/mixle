"""The capability layer: what can this object actually do, not what class is it?

mixle's families are deliberately heterogeneous -- a finite-support categorical, a countably-infinite
Poisson, a continuous Gaussian, a multivariate Gaussian that can condition and marginalize -- and none of
them share a common rich base class you could ``isinstance()`` against to ask "does this have a mean?" or
"can I condition on a coordinate?". The capability layer answers those questions by inspecting an object's
actual surface instead of its class, so generic code (and code operating on families it has never seen)
can dispatch on behaviour instead of a concrete type name.

``examples/quickstart_example.py`` already covers the single-call summary, ``mixle.describe(x)``. This
file covers the rest of the introspection surface (``mixle.capability``, re-exported at the top level):

  * ``capabilities(obj)``        -- the frozenset of capability names ``obj`` provides
  * ``supports(obj, Cap)``       -- does ``obj`` provide one specific capability
  * ``require(obj, Cap, op)``    -- raise a clear ``CapabilityError`` if it doesn't
  * ``catalog()``                -- the full capability vocabulary, as data
  * ``what_supports(Cap, pool)`` -- filter a heterogeneous collection down to the ones that qualify
  * ``summarize(obj)``           -- every closed-form statistic obj's capabilities make available

Run: ``python examples/capability_layer_example.py``
"""

from __future__ import annotations

import numpy as np

import mixle
from mixle.capability import CapabilityError, Conditionable, Enumerable
from mixle.stats.multivariate.multivariate_gaussian import MultivariateGaussianDistribution
from mixle.stats.univariate.continuous.gaussian import GaussianDistribution
from mixle.stats.univariate.discrete.categorical import CategoricalDistribution
from mixle.stats.univariate.discrete.poisson import PoissonDistribution


def build_pool():
    """Four distributions, deliberately picked so each has a different capability profile.

    Categorical has finite support, so it's enumerable AND rankable by index -- but no mean/variance/cdf
    methods. Poisson has countably infinite support, so it's enumerable but NOT finite/rankable -- and
    unlike Categorical it does have closed-form moments. Gaussian is continuous: no enumeration at all,
    but the richest closed-form summary. MultivariateGaussian is the only one of the four with real
    ``condition()`` / ``marginal()`` methods, and (like Categorical) has no moment/cdf methods of its own.
    """
    return {
        "CategoricalDistribution": CategoricalDistribution({"a": 0.5, "b": 0.3, "c": 0.2}),
        "PoissonDistribution": PoissonDistribution(2.0),
        "GaussianDistribution": GaussianDistribution(0.0, 1.0),
        "MultivariateGaussianDistribution": MultivariateGaussianDistribution(np.zeros(2), np.eye(2)),
    }


def demo_capabilities(pool):
    print("1. capabilities(obj) -- one call, the full capability set, per object:")
    for name, dist in pool.items():
        print(f"   {name:<34} {sorted(mixle.capabilities(dist))}")


def demo_catalog():
    rows = mixle.catalog()
    facets = [s for s in rows if s.kind == "distribution facet"]
    print(f"\n2. catalog() -- the whole capability vocabulary, as data: {len(rows)} rows total ({len(facets)} of")
    print("   kind 'distribution facet'). A few, verbatim -- the same names used in the demo above:")
    highlight = ("Enumerable", "Conditionable", "ConjugateUpdatable", "HasMoments")
    for spec in rows:
        if spec.name in highlight:
            print(f"   - {spec.name:<19} {spec.summary}  [backed by {spec.backed_by}]")


def demo_what_supports(pool):
    names = mixle.what_supports(Enumerable, list(pool.values()))
    print(f"\n3. what_supports(Enumerable, pool) -- filters the 4-object pool down to: {names}")


def demo_require(pool):
    print("\n4. require(obj, Conditionable, op) -- enforce a capability before relying on it:")
    mvn = pool["MultivariateGaussianDistribution"]
    mixle.require(mvn, Conditionable, "condition on coordinate 0")
    print("   MultivariateGaussianDistribution: passes silently -- mvn.condition({0: 1.0}) is safe to call")
    try:
        mixle.require(pool["GaussianDistribution"], Conditionable, "condition on coordinate 0")
    except CapabilityError as exc:
        print(f"   GaussianDistribution:             raises CapabilityError: {exc}")


def demo_summarize(pool):
    print("\n5. summarize(obj) -- every closed-form statistic obj's capabilities make available:")
    for name, dist in pool.items():
        stats = mixle.summarize(dist)
        shown = stats if stats else "{}  (no HasMoments/HasCDF/HasEntropy on this family -- not an error)"
        print(f"   {name:<34} {shown}")


def main():
    print("# mixle capability layer: introspect what an object can do, not what class it is\n")
    pool = build_pool()
    demo_capabilities(pool)
    demo_catalog()
    demo_what_supports(pool)
    demo_require(pool)
    demo_summarize(pool)


if __name__ == "__main__":
    main()
