import os
#os.environ['NUMBA_DISABLE_JIT'] = '1'

from pysp.stats.pdist import SequenceEncodableProbabilityDistribution
from pysp.stats import *
from pysp.utils.estimation import empirical_kl_divergence
import numpy as np
import unittest

from pysp.stats.int_spike import *

class BaseDistributionTestCase(unittest.TestCase):

    def setUp(self) -> None:
        dists = []
        # dists.append(BinomialDistribution(p=0.4,n=10,min_val=1,name='a',keys='test_keys'))
        # dists.append(CategoricalDistribution({'a': 0.4, 'b': 0.3, 'c': 0.2, 'd': 0.1}, default_value=0.0, name='a'))
        # dists.append(MultinomialDistribution(IntegerCategoricalDistribution(0, [0.1, 0.4, 0.3, 0.2]),
        #                                      CategoricalDistribution({5: 1.0}), name='a'))
        # dists.append(CompositeDistribution((ExponentialDistribution(3.1), PoissonDistribution(3.2))))
        # given_dist = IntegerCategoricalDistribution(min_val=1, p_vec=np.ones(5)/5, name='a')
        # dists.append(ConditionalDistribution(dmap={k: ExponentialDistribution(beta=k*2.0) for k in range(1, 6)}, given_dist=given_dist,name='b', keys='test_key'))
        # dists.append(DirichletDistribution([1.1, 2.8, 4.5], name='a'))
        # dists.append(DiagonalGaussianDistribution([1.8, 4.3, -1.5], [1.1, 4.8, 9.1], name='a'))
        # dists.append(ExponentialDistribution(10.8, name='a'))
        # dists.append(GammaDistribution(k=1.0, theta=10.0, name='a'))
        # dists.append(GaussianDistribution(mu=1.0, sigma2=1.0,name='a'))
        #dists.append(GeometricDistribution(p=0.20,name='a'))

        #### needs more samples on estimate tests
        # comps = [GaussianDistribution(mu=100, sigma2=1.0), ExponentialDistribution(beta=1.0)]
        # dists.append(HeterogeneousMixtureDistribution(components=comps, w=np.ones(2)/2, name='a'))

        #### slow since seq_ is just call to update()
        # aa = 0.90
        # bb = (1.0 - aa) / 2
        # dist1 = CategoricalDistribution({'a': aa, 'b': bb, 'c': bb}, name='a0')
        # dist2 = CategoricalDistribution({'a': bb, 'b': aa, 'c': bb}, name='a1')
        # dist3 = CategoricalDistribution({'a': bb, 'b': bb, 'c': aa}, name='a2')
        # cond_dist = ConditionalDistribution({'a': dist1, 'b': dist2, 'c': dist3}, name='b0')
        # given_dist = MultinomialDistribution(CategoricalDistribution({'a': 0.3, 'b': 0.2, 'c': 0.5}),
        #                                      len_dist=CategoricalDistribution({5: 1.0}), name='b1')
        # len_dist = CategoricalDistribution({7: 1.0}, name='b2')
        # dists.append(HiddenAssociationDistribution(cond_dist=cond_dist, given_dist=given_dist, len_dist=len_dist, name='c'))

        #### HMM needs samples raised for seq_est
        # topics = [GaussianDistribution(mu=100, sigma2=1.0), GaussianDistribution(mu=-100, sigma2=1.0)]
        # w = np.ones(2)/2
        # transitions = np.ones((2, 2)) / 2
        # len_dist = CategoricalDistribution({3: 1.0})
        # hmm = HiddenMarkovModelDistribution(topics=topics, w=w, transitions=transitions, taus=None, use_numba=False,
        #                                     name='a', terminal_values=None, len_dist=len_dist)
        # dists.append(hmm)

        ### need samples raised for seq_est
        # topic1 = CategoricalDistribution({'a': 0.50, 'b': 0.25, 'c': 0.25})
        # topic2 = CategoricalDistribution({'a': 0.25, 'b': 0.50, 'c': 0.25})
        # topic3 = CategoricalDistribution({'a': 0.25, 'b': 0.25, 'c': 0.50})
        #
        # w = [0.25, 0.25, 0.25, 0.25]
        # taus = [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0], [0.3, 0.4, 0.3]]
        #
        # len_dist = CategoricalDistribution({8: 0.1, 9: 0.2, 10: 0.7})
        # dists.append(HierarchicalMixtureDistribution([topic1, topic2, topic3], w, taus, len_dist=len_dist, name='a'))

        # dependency_list = [(0, None), (1, 0), (2, 1), (3, 2), (4, 3), (5, 4), (6, 5), (7, 6)]
        # conditional_log_densities = np.ones(len(dependency_list))/len(dependency_list)
        # dists.append(ICLTreeDistribution(dependency_list=dependency_list, conditional_log_densities=conditional_log_densities))

        #dists.append(IgnoredDistribution(GeometricDistribution(0.8)))
        # dists.append(
        #     IntegerBernoulliSetDistribution(np.log([0.9, 0.8, 0.7, 0.6, 0.5]), np.log([0.1, 0.2, 0.3, 0.4, 0.5]),
        #                                     name='a'))
        #
        #
        # ### slow since seq_ is just call to update()
        # ### slow since seq_ is just call to update()
        # cond_probs = np.ones((5 ** 2, 5)) / 5
        # len_dist = IntegerCategoricalDistribution(min_val=0, p_vec=np.ones(5) / 5)
        # init = SequenceDistribution(dist=IntegerCategoricalDistribution(min_val=0, p_vec=np.ones(5) / 5),
        #                             len_dist=CategoricalDistribution({2: 1.0}))
        #
        # dists.append(
        #     IntegerMarkovChainDistribution(num_values=5, cond_dist=cond_probs, lag=2, init_dist=init, len_dist=len_dist,
        #                                    name='a', keys='test_keys'))
        # rng = np.random.RandomState(1)
        # authors = 4
        # states = 3
        # words = 10
        # state_word_mat = rng.dirichlet(alpha=np.ones(words), size=states).T
        # doc_state_mat = rng.dirichlet(alpha=np.ones(states), size=authors)
        # doc_vec = rng.dirichlet(alpha=np.ones(authors), size=1)[0]
        # len_dist = CategoricalDistribution({8: 0.1, 9: 0.2, 10: 0.7})
        # dists.append(
        #     IntegerPLSIDistribution(state_word_mat=state_word_mat, doc_state_mat=doc_state_mat, doc_vec=doc_vec,
        #                             len_dist=len_dist, name='a'))
        # dists.append(IntegerMultinomialDistribution(0, [0.1, 0.4, 0.3, 0.2], len_dist=CategoricalDistribution({4: 1.0}),
        #                                             name='a'))
        # dists.append(IntegerCategoricalDistribution(0, [0.1, 0.4, 0.3, 0.2], name='a'))
        #dists.append(IntegerBernoulliSetDistribution(np.log([0.9, 0.8, 0.7, 0.6, 0.5]), name='a'))

        ### need to increase number of samples in est comparison
        # d11 = CompositeDistribution(
        #     [CategoricalDistribution({'a': 1.0, 'b': 0.0, 'c': 0.0}), GaussianDistribution(mu=-6.0, sigma2=1.0)])
        # d12 = CompositeDistribution(
        #     [CategoricalDistribution({'a': 0.0, 'b': 1.0, 'c': 0.0}), GaussianDistribution(mu=0.0, sigma2=1.0)])
        # d13 = CompositeDistribution(
        #     [CategoricalDistribution({'a': 0.0, 'b': 0.0, 'c': 1.0}), GaussianDistribution(mu=6.0, sigma2=1.0)])
        #
        # d21 = SequenceDistribution(
        #     CompositeDistribution([GaussianDistribution(mu=-6.0, sigma2=1.0), GammaDistribution(1.0, 3.0)]),
        #     PoissonDistribution(3.0), len_normalized=True)
        # d22 = SequenceDistribution(
        #     CompositeDistribution([GaussianDistribution(mu=0.0, sigma2=1.0), GammaDistribution(3.0, 3.0)]),
        #     PoissonDistribution(3.0), len_normalized=True)
        # d23 = SequenceDistribution(
        #     CompositeDistribution([GaussianDistribution(mu=6.0, sigma2=1.0), GammaDistribution(1.0, 3.0)]),
        #     PoissonDistribution(3.0), len_normalized=True)
        #
        # taus12 = [[0.8, 0.1, 0.1], [0.1, 0.8, 0.1], [0.1, 0.1, 0.8]]
        # taus21 = [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]]
        # w1 = [0.6, 0.3, 0.1]
        # w2 = [0.7, 0.2, 0.1]
        # dists.append(JointMixtureDistribution([d11, d12, d13], [d21, d22, d23], w1, w2, taus12, taus21, name='a'))

        # dists.append(LogGaussianDistribution(mu=10.0, sigma2=1.0, name='a'))
        # dists.append(MarkovChainDistribution({'a': 0.1, 'b': 0.5, 'c': 0.4},
        #                                      {'a': {'a': 0.8, 'b': 0.1, 'c': 0.1},
        #                                       'b': {'a': 0.1, 'b': 0.8, 'c': 0.1},
        #                                       'c': {'a': 0.1, 'b': 0.1, 'c': 0.8}},
        #                                      len_dist=CategoricalDistribution({5:1.0}), name='a'))

        # #### needs more samples on estimate tests
        # comps = [GaussianDistribution(mu=100, sigma2=1.0), GaussianDistribution(mu=0, sigma2=1.0)]
        # dists.append(MixtureDistribution(components=comps, w=np.ones(2)/2, name='a'))
        # dists.append(
        #     MultivariateGaussianDistribution([1.0, 3.3, 2.2], [[3.0, 2.0, 1.0], [2.0, 3.0, 2.0], [1.0, 2.0, 3.0]],
        #                                      name='a'))

        # dists.append(OptionalDistribution(PoissonDistribution(4.7), p=0.1, name='a'))
        # dists.append(OptionalDistribution(PoissonDistribution(4.7), p=0.1, missing_value='asdf', name='a'))
        # dists.append(OptionalDistribution(BinomialDistribution(0.25, 5, name='a'), p=0.2, missing_value=float("nan"), name='a'))
        # dists.append(PoissonDistribution(lam=10.0, name='a'))
        # dists.append(
        #     SequenceDistribution(GeometricDistribution(0.8), len_dist=CategoricalDistribution({5: 1.0}), name='a'))
        # dists.append(BernoulliSetDistribution({'a': 0.8, 'b': 0.1, 'c': 0.7}, name='a'))
        # dists.append(BernoulliSetDistribution({'a': 0.8, 'b': 0.1, 'c': 0.0}, name='a'))
        # dists.append(BernoulliSetDistribution({'a': 1.0, 'b': 0.1, 'c': 0.7}, name='a'))
        # dists.append(BernoulliSetDistribution({'a': 0.8, 'b': 0.2, 'c': 0.6}, min_prob=1.0e-128, name='a'))

        #### Need to increase sample size on tests
        # seq_samp = 10
        # c1 = CompositeDistribution((SequenceDistribution(CategoricalDistribution({'a': 0.8, 'b': 0.1, 'c': 0.1}),
        #                                                  len_dist=CategoricalDistribution({seq_samp: 1.0})),
        #                             GaussianDistribution(0.0, 1.0)))
        # c2 = CompositeDistribution((SequenceDistribution(CategoricalDistribution({'a': 0.1, 'b': 0.8, 'c': 0.1}),
        #                                                  len_dist=CategoricalDistribution({seq_samp: 1.0})),
        #                             GaussianDistribution(1.0, 1.0)))
        # c3 = CompositeDistribution((SequenceDistribution(CategoricalDistribution({'a': 0.1, 'b': 0.1, 'c': 0.8}),
        #                                                  len_dist=CategoricalDistribution({seq_samp: 1.0})),
        #                             GaussianDistribution(2.0, 1.0)))
        # dist = SemiSupervisedMixtureDistribution([c1, c2, c3], [0.6, 0.3, 0.1], name='a')
        #
        # dists.append(VonMisesFisherDistribution([1.1,2.1,3.1,4.1,5.1], 2.0, name='a'))
        #dists.append(IntegerUniformSpikeDistribution(k=3, min_val=0, num_vals=10, p=0.6, name='a'))
        # dists.append(NegativeBinomialDistribution(r=3, p=0.45, name='a'))

        num_states = 3
        rng = np.random.RandomState(1)

        p = [[0.7, 0.20, .10], [0.10, 0.70, .20], [0.20, 0.10, .70]]
        topics = []

        for s in range(num_states):
            topics.append(GaussianDistribution(mu=0 + s * 10, sigma2=0.10 ** 2))

        len_probs = np.array([0, 0, 1], dtype=np.float64)
        len_probs /= np.sum(len_probs)

        trans_mat = np.asarray(p)  # np.asarray([[0.1, 0.90], [0.90, 0.1]])

        w = np.ones(num_states) / num_states
        len_dist = IntegerCategoricalDistribution(min_val=0, p_vec=len_probs)

        d = TreeHiddenMarkovModelDistribution(topics=topics, w=w, transitions=trans_mat, len_dist=len_dist,
                                              terminal_level=4)
        dists.append(d)


        self.dists = dists

    def test_sampler_repeat(self):
        for dist in self.dists:
            res = sampler_repeat_test(dist)
            self.assertTrue(res[0], str(res[1]))

    def test_string_match(self):
        for dist in self.dists:
            res = string_match_test(dist)
            self.assertTrue(res[0], str(res[1]))

    def test_eval(self):
        for dist in self.dists:
            res = string_eval_test(dist)
            self.assertTrue(res[0], str(res[1]))

    def test_log_density(self):
        for dist in self.dists:
            res = log_density_test(dist)
            if not res[0]:
                print(str(dist))
            self.assertTrue(res[0], str(res[1]))

    def test_estimation(self):
        for dist in self.dists:
            res = estimation_test(dist)
            self.assertTrue(res[0], str(res[1]))

    def test_seq_estimation(self):
        for dist in self.dists:
            res = seq_estimation_test(dist)
            self.assertTrue(res[0], str(res[1]))

    def test_estimation_same_name(self):
        for dist in self.dists:
            res = estimation_same_name_test(dist)
            #if not res[0]:
            #    print(str(dist))
            self.assertTrue(res[0], str((dist,res[1])))

def sampler_repeat_test(dist):

    seeds = [1, 2, 3]
    sz = 20
    rv = []
    for seed in seeds:

        s  = dist.sampler(seed)
        d1 = s.sample(size=sz)
        s  = dist.sampler(seed)
        d2 = s.sample(size=sz)

        is_same = [u[0] == u[1] for u in zip(map(str, d1), map(str, d2))]

        rv.append(all(is_same))

    return all(rv), rv

def string_match_test(dist):
    sdist = eval(str(dist))
    return str(sdist) == str(dist), '__str__ is not idempotent.'

def string_eval_test(dist):

    seeds = [1, 2]
    sz = 5
    rv = []
    for seed in seeds:

        s  = dist.sampler(seed)
        data = s.sample(size=sz)

        sdist = eval(str(dist))

        enc_data = dist.dist_to_encoder().seq_encode(data)
        seq_ll0  = dist.seq_log_density(enc_data)
        seq_ll1  = sdist.seq_log_density(enc_data)
        seq_dll  = np.zeros(sz, dtype=np.float64)

        for i in range(sz):

            if seq_ll0[i] == 0:
                seq_dll[i] = np.abs(seq_ll1[i])
            else:
                seq_dll[i] = np.abs(seq_ll0[i] - seq_ll1[i])/np.abs(seq_ll0[i])

        rv.append(np.max(seq_dll))

    return max(rv) < 1.0e-15, max(rv)

def log_density_test(dist):

    seeds = [1, 2, 3]
    sz = 20
    rv = []
    for seed in seeds:

        s  = dist.sampler(seed)
        data = s.sample(size=sz)

        seq_ll = dist.seq_log_density(dist.dist_to_encoder().seq_encode(data))
        for i in range(sz):

            if seq_ll[i] == 0:
                seq_ll[i] = np.abs(dist.log_density(data[i]))
            else:
                seq_ll[i] = np.abs(seq_ll[i] - dist.log_density(data[i]))/np.abs(seq_ll[i])

        rv.append(max(seq_ll))

    return max(rv) < 1.0e-14, max(rv)


def estimation_test(dist):

    seeds = [1, 2, 3, 4]
    szs = [50, 500, 1000]
    rv  = []

    akld = []
    for seed in seeds:

        kld = []
        better = []
        for sz in szs:

            data  = dist.sampler(seed).sample(size=sz)
            est   = dist.estimator()
            enc_data = seq_encode(data, encoder=dist.dist_to_encoder())
            init = initialize(data, est, rng=np.random.RandomState(1), p=1.0)
            est_dist = estimate(data, est, init)

            emp_kld, _, _ = empirical_kl_divergence(dist, est_dist, enc_data)

            if len(kld) > 0:
                better.append(kld[-1] >= emp_kld)

            kld.append(emp_kld)
        akld.append(kld)
        rv.append(all(better))

    akld_mean = np.mean(akld, axis=0)
    rv = np.all(akld_mean[1:] <= akld_mean[:-1])

    return rv, akld

def seq_estimation_test(dist):

    seeds = [1, 2, 3, 4]
    szs = [2000]
    rv  = []

    akld = []
    for seed in seeds:

        kld = []
        better = []
        for sz in szs:

            data  = dist.sampler(seed).sample(size=sz)
            est   = dist.estimator()
            enc_data = seq_encode(data, model=dist)
            init = seq_initialize(enc_data, est, np.random.RandomState(1), p=1.0)
            est_dist = seq_estimate(enc_data, est, init)

            emp_kld, _, _ = empirical_kl_divergence(dist, est_dist, enc_data)

            if len(kld) > 0:
                better.append(kld[-1] >= emp_kld)

            kld.append(emp_kld)
        akld.append(kld)
        rv.append(all(better))

    akld_mean = np.mean(akld, axis=0)
    rv = np.all(akld_mean[1:] <= akld_mean[:-1])

    return rv, akld

def estimation_same_name_test(dist):

    if not hasattr(dist, 'name'):
        return True, ''

    seed = 1
    data = dist.sampler(seed).sample(50)
    est = dist.estimator()
    enc_data = seq_encode(data, model=dist)
    init = seq_initialize(enc_data=enc_data, estimator=est, rng=np.random.RandomState(1), p=1.0)
    model = seq_estimate(enc_data, est, init)

    return model.name is dist.name, ''

def evaluate_dists(dists):

    tests = [sampler_repeat_test, log_density_test, estimation_test, string_match_test, string_eval_test]

    for dist in dists:
        print(str(dist))
        all_res1 = []
        all_res2 = []
        for test in tests:
            res = test(dist)
            all_res1.append(res[0])
            all_res2.append((test.__name__, res))
        passed = all(all_res1)
        if passed:
            print('Passed All Tests')
        else:
            for t in all_res2:
                if not t[1][0]:
                    print(t)
        print('-'*10)


if __name__ == '__main__':
    unittest.main()