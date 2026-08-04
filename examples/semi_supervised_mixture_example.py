"""SemiSupervisedMixture: fit one mixture from a few labelled rows plus a lot of unlabelled ones.

Each observation is a pair ``(value, label_prior)``. ``label_prior`` is either ``None`` (this row is
unlabelled -- EM infers its component as usual) or a list of ``(component_index, weight)`` pairs
expressing what is known about the row's component. A hard label is ``[(k, 1.0)]``; a partial belief
like ``[(0, 0.7), (1, 0.3)]`` is equally valid, which is the part a "just filter to the labelled rows"
workflow cannot express.

Here 3 hand-written labelled exemplars -- one per component, each an unambiguous prototype -- anchor
the component identities, and 1000 unlabelled samples supply the density. Without the anchors the
three components are recovered only up to a permutation; with them, component ``k`` is the one the
exemplar named.

Takeaway: supervision is a per-observation PRIOR over components, not a separate training mode. The
same ``best_of`` restart loop that fits an unsupervised mixture fits this, and the labelled and
unlabelled rows travel together in one dataset.

KNOWN ISSUE (0.8.0): this script currently raises
``NotImplementedError: SemiSupervisedMixtureDataEncoder must implement row_count()`` at the
``best_of`` call below, for any component family whose encoded payload is not a plain array (the
Composite/Sequence components used here). The encoder's payload already carries the row count as its
first element but does not override ``row_count()``. This is a library defect, not an example defect,
and the example is left as-is so it keeps reproducing it.
"""

import numpy as np

from mixle.inference import best_of
from mixle.stats import *
from mixle.stats import SemiSupervisedMixtureDistribution, SemiSupervisedMixtureEstimator

if __name__ == "__main__":
    seq_samp = 10
    c1 = CompositeDistribution(
        (
            SequenceDistribution(
                CategoricalDistribution({"a": 0.8, "b": 0.1, "c": 0.1}),
                len_dist=CategoricalDistribution({seq_samp: 1.0}),
            ),
            GaussianDistribution(0.0, 1.0),
        )
    )
    c2 = CompositeDistribution(
        (
            SequenceDistribution(
                CategoricalDistribution({"a": 0.1, "b": 0.8, "c": 0.1}),
                len_dist=CategoricalDistribution({seq_samp: 1.0}),
            ),
            GaussianDistribution(1.0, 1.0),
        )
    )
    c3 = CompositeDistribution(
        (
            SequenceDistribution(
                CategoricalDistribution({"a": 0.1, "b": 0.1, "c": 0.8}),
                len_dist=CategoricalDistribution({seq_samp: 1.0}),
            ),
            GaussianDistribution(2.0, 1.0),
        )
    )

    dist = SemiSupervisedMixtureDistribution([c1, c2, c3], [0.6, 0.3, 0.1])

    data = dist.sampler(seed=1).sample(1000)
    rng = np.random.RandomState(1)

    data = [
        ((["a"] * seq_samp, 0.0), [(0, 1.0)]),
        ((["b"] * seq_samp, 1.0), [(1, 1.0)]),
        ((["c"] * seq_samp, 2.0), [(2, 1.0)]),
    ] + [(u, None) for u in data]
    suff_stat = {"a": 1.0 / 3.0, "b": 1.0 / 3.0, "c": 1.0 / 3.0}

    est = SemiSupervisedMixtureEstimator(
        [
            CompositeEstimator(
                (SequenceEstimator(CategoricalEstimator(pseudo_count=1.0e-3, suff_stat=suff_stat)), GaussianEstimator())
            )
        ]
        * 3,
        pseudo_count=1.0e-6,
    )

    _, model = best_of(data, data, est, 10, 1000, 0.05, 1.0e-8, rng, print_iter=1000)

    print(str(model.w))
    print("\n".join(map(str, model.components)))
