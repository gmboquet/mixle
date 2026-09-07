"""Gallery: random-graph families, each built / sampled / re-estimated.

Erdos-Renyi (i.i.d. edges), stochastic block (edge probability by node block), random dot-product
(edge probability from latent node positions), and a knowledge graph (scored entity/relation
triples). Observations are adjacency matrices; each true model exposes a matching estimator via
``.estimator()``. Random data only.
"""

import numpy as np

from mixle.inference import estimate
from mixle.stats import (
    ErdosRenyiGraphDistribution,
    KnowledgeGraphDistribution,
    RandomDotProductGraphDistribution,
    StochasticBlockGraphDistribution,
)

if __name__ == "__main__":
    rng = np.random.RandomState(0)

    print("# ErdosRenyiGraph (i.i.d. edges over a fixed node set)")
    d = ErdosRenyiGraphDistribution(0.3, num_nodes=6)
    fit = estimate(list(d.sampler(1).sample(2000)), d.estimator())
    print("  true p 0.30 -> fit %s" % fit)

    print("# StochasticBlockGraph (edge probability depends on node block membership)")
    assignments = [0, 0, 0, 1, 1, 1]
    d = StochasticBlockGraphDistribution([[0.8, 0.1], [0.1, 0.7]], block_assignments=assignments)
    fit = estimate(list(d.sampler(1).sample(2000)), d.estimator())
    print("  fit: %s" % fit)

    print("# RandomDotProductGraph (edge probability from latent node positions)")
    # Positions scaled so every inner product lands in [0, 1]: uniform draws on [0, 1]^2 put a few
    # pairs above 1, which the distribution has to clip -- and a clipped model is no longer the
    # rank-2 dot-product model these positions define, so the rank-2 fit below would be chasing a
    # different truth than the one that generated the graphs. numerical_repairs() says so when it
    # happens; here there is nothing to say, which is what makes the comparison meaningful.
    positions = rng.rand(8, 2) / np.sqrt(2.0)
    d = RandomDotProductGraphDistribution(positions)
    assert d.numerical_repairs() == ()
    fit = estimate(list(d.sampler(1).sample(2000)), d.estimator())
    print("  fit: %s" % fit)
    print(
        "  mean |P_fit - P_true| over off-diagonal pairs: %.4f"
        % float(np.mean(np.abs(fit.edge_marginals() - d.edge_marginals())[~np.eye(8, dtype=bool)]))
    )

    print("# KnowledgeGraph (scored entity/relation triples)")
    d = KnowledgeGraphDistribution(rng.randn(6, 3), rng.randn(2, 3))
    fit = estimate(list(d.sampler(1).sample(2000)), d.estimator())
    print("  fit: %s" % fit)
