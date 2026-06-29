"""mixle.enumeration: exact descending-probability enumeration of discrete models.

For any distribution with enumerable support, ``top_k`` returns the k most probable outcomes in
exact descending order without normalizing the whole support, and ``supports_enumeration`` reports
whether a model qualifies. (For harder latent models, ``sound_top_k`` gives a certified top-k.)
"""

import numpy as np

from mixle.stats import (
    CategoricalDistribution,
    CompositeDistribution,
    IntegerCategoricalDistribution,
    MixtureDistribution,
)
from mixle.enumeration import supports_enumeration, top_k

if __name__ == '__main__':
    cat = CategoricalDistribution({'a': 0.5, 'b': 0.3, 'c': 0.2})

    # A heterogeneous record: an integer feature and a categorical feature.
    comp = CompositeDistribution([
        IntegerCategoricalDistribution(0, [0.6, 0.4]),
        CategoricalDistribution({'x': 0.7, 'y': 0.3}),
    ])

    # A discrete mixture (the latent component is marginalized out).
    mix = MixtureDistribution(
        [IntegerCategoricalDistribution(0, [0.8, 0.2]), IntegerCategoricalDistribution(0, [0.2, 0.8])],
        [0.5, 0.5],
    )

    for label, dist in [('Categorical', cat), ('Composite', comp), ('Mixture', mix)]:
        print('%-12s supports_enumeration=%s' % (label, supports_enumeration(dist)))
        for outcome, log_p in top_k(dist, 3):
            print('   p=%.3f  %r' % (np.exp(log_p), outcome))
