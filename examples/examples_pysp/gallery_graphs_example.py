"""Gallery: random-graph families, each built / sampled / re-estimated.

Erdos-Renyi (i.i.d. edges), stochastic block (edge probability by node block), random dot-product
(edge probability from latent node positions), and a knowledge graph (scored entity/relation
triples). Observations are adjacency matrices; each true model exposes a matching estimator via
``.estimator()``. Random data only.
"""

import numpy as np

from pysp.stats import (
    ErdosRenyiGraphDistribution,
    KnowledgeGraphDistribution,
    RandomDotProductGraphDistribution,
    StochasticBlockGraphDistribution,
)
from pysp.inference import estimate

if __name__ == '__main__':
    rng = np.random.RandomState(0)

    print('# ErdosRenyiGraph (i.i.d. edges over a fixed node set)')
    d = ErdosRenyiGraphDistribution(0.3, num_nodes=6)
    fit = estimate(list(d.sampler(1).sample(2000)), d.estimator())
    print('  true p 0.30 -> fit %s' % fit)

    print('# StochasticBlockGraph (edge probability depends on node block membership)')
    assignments = [0, 0, 0, 1, 1, 1]
    d = StochasticBlockGraphDistribution([[0.8, 0.1], [0.1, 0.7]], block_assignments=assignments)
    fit = estimate(list(d.sampler(1).sample(2000)), d.estimator())
    print('  fit: %s' % fit)

    print('# RandomDotProductGraph (edge probability from latent node positions)')
    d = RandomDotProductGraphDistribution(rng.rand(8, 2))
    fit = estimate(list(d.sampler(1).sample(2000)), d.estimator())
    print('  fit: %s' % fit)

    print('# KnowledgeGraph (scored entity/relation triples)')
    d = KnowledgeGraphDistribution(rng.randn(6, 3), rng.randn(2, 3))
    fit = estimate(list(d.sampler(1).sample(2000)), d.estimator())
    print('  fit: %s' % fit)
