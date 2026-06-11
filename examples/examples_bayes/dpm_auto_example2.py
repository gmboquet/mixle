from pysp.utils.automatic import get_dpm_mixture, get_estimator
from pysp.bstats import *
import numpy as np


if __name__ == '__main__':

    rng = np.random.RandomState(2)
    m = 10
    n = 7
    cc = 2.0
    ss = 1.5
    components = []
    w = np.zeros(m)
    pvec = np.ones(m) * 0.5
    for i in range(m):
        len_dist = IntegerCategoricalDistribution([0.2, 0.3, 0.3, 0.2], min_index=3)
        dist1 = GaussianDistribution((i + 1) * ss, 1)
        dist2 = IntegerCategoricalDistribution((np.eye(m)[i, :] + cc) / (m * cc + 1))
        dist3 = CategoricalDistribution({str(j): ((1.0 + cc) if i == j else cc) / (m * cc + 1) for j in range(m)})
        dist4 = OptionalDistribution(PoissonDistribution((i + 1) * ss), p=0.1)
        dist = SequenceDistribution(CompositeDistribution((dist1, dist2, dist3, dist4)), len_dist)
        components.append(dist)
        w[i] = np.prod(1 - pvec[:i]) * pvec[i]

    w[n:] = 1.0e-16
    w /= w.sum()

    dist = MixtureDistribution(components, w)
    data = dist.sampler(seed=1).sample(300)

    est = get_estimator(data, use_bstats=True)
    model = get_dpm_mixture(data, rng=np.random.RandomState(1))

    print(str(model))
    print(model.num_components)
    for u in model.components:
        print(str(u))