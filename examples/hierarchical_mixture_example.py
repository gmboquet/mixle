"""HierarchicalMixture: an admixture -- a mixture over per-document MIXING PROFILES of shared topics.

Each observation is a bag of tokens. A plain mixture would assign the whole bag to one component; a
hierarchical mixture instead draws an outer component (a "mixing profile", ``taus``) and then draws
every token in the bag from the shared topics under that profile. Three of the four profiles here are
near-degenerate (all mass on one topic) and the fourth is a genuine blend ``[0.3, 0.4, 0.3]`` -- so the
model can represent both "this document is about topic 2" and "this document mixes all three".

``posterior(bag)`` is shown before any fitting: hand it a bag drawn purely from topic 1/2/3 and it
returns the outer-component responsibilities, which concentrate on the matching degenerate profile.

Takeaway: this is the general admixture base -- LDA is its Categorical-topic special case, and the
same class takes Gaussian, Poisson, or heterogeneous topics (see ``latent_variable_models_example.py``).
The long comment on ``max_its`` below documents why this particular configuration is capped by
iteration count rather than by a convergence delta. Runtime is on the order of one to two minutes.
"""

import numpy as np

from mixle.inference import optimize
from mixle.stats import *

if __name__ == "__main__":
    # Create data distribution

    topic1 = CategoricalDistribution({"a": 0.50, "b": 0.25, "c": 0.25})
    topic2 = CategoricalDistribution({"a": 0.25, "b": 0.50, "c": 0.25})
    topic3 = CategoricalDistribution({"a": 0.25, "b": 0.25, "c": 0.50})

    w = [0.25, 0.25, 0.25, 0.25]
    taus = [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0], [0.3, 0.4, 0.3]]

    len_dist = CategoricalDistribution({8: 0.1, 9: 0.2, 10: 0.7})
    dist = HierarchicalMixtureDistribution([topic1, topic2, topic3], w, taus, len_dist=len_dist)

    # Mixture posteriors for bags of samples

    t1_sample = topic1.sampler(1).sample(10)
    t2_sample = topic2.sampler(2).sample(10)
    t3_sample = topic3.sampler(3).sample(10)

    print(dist.posterior(t1_sample))
    print(dist.posterior(t2_sample))
    print(dist.posterior(t3_sample))

    # Sample data
    data = dist.sampler(1).sample(2000)

    # Estimate model parameters
    num_topics = 3
    num_mixtures = 4
    est0 = CategoricalEstimator()
    est1 = CategoricalEstimator()
    est = HierarchicalMixtureEstimator([est0] * num_topics, num_mixtures, len_estimator=est1)

    # This configuration is weakly identified (4 outer components -- 3 near-degenerate
    # single-topic, one blended -- over only 3 mildly-skewed categorical symbols, 8-10 tokens per
    # document), so the estimator's exact delta=1e-9 default tolerance is never reached in
    # practice: a direct per-iteration trace shows log-likelihood plateaus for ~1000 iterations,
    # jumps sharply between iterations 1000-1500 (EM escaping a saddle), then creeps for
    # thousands more iterations without crossing delta<1e-9 (still ~8.6e-6 at iteration 10000).
    # That is long-standing behavior, not a recent regression (confirmed unchanged back to
    # v0.7.0; see docs/example-execution-manifest.rst). max_its is capped well past the
    # iteration-1500 escape -- capturing >99.99% of the log-likelihood improvement a full
    # 10000-iteration run achieves -- so this example finishes quickly and reliably instead of
    # grinding through thousands of iterations of marginal refinement.
    model = optimize(data, est, max_its=2000, print_iter=500, rng=np.random.RandomState(2))
    print(str(model))
