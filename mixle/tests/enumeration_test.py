"""Tests for the smart enumeration API.

Every enumerable distribution is run through the generic invariants: descending order,
consistency with log_density, uniqueness, completeness (finite supports), and agreement
with brute-force enumeration for small models. Error cases verify the fail-fast
EnumerationError contract.
"""

import itertools
import math
import unittest

import numpy as np

from mixle.enumeration.algorithms import SoundTopKResult, freeze, sound_top_k, supports_enumeration
from mixle.stats import *
from mixle.stats.compute.pdist import EnumerationError

TOL = 1e-9


def make_cases():
    """Returns a list of (name, dist, n_to_take, expected_count_or_None) cases."""
    cat = CategoricalDistribution({"a": 0.7, "b": 0.3})
    cat3 = CategoricalDistribution({"a": 0.5, "b": 0.3, "c": 0.2})
    intcat = IntegerCategoricalDistribution(2, [0.1, 0.0, 0.6, 0.3])
    geo = GeometricDistribution(0.3)
    mix_geo = MixtureDistribution([GeometricDistribution(0.5), GeometricDistribution(0.1)], [0.6, 0.4])

    hmm = HiddenMarkovModelDistribution(
        topics=[CategoricalDistribution({"a": 0.8, "b": 0.2}), CategoricalDistribution({"b": 0.6, "c": 0.4})],
        w=[0.7, 0.3],
        transitions=[[0.9, 0.1], [0.4, 0.6]],
        len_dist=IntegerCategoricalDistribution(0, [0.1, 0.3, 0.4, 0.2]),
    )

    mc = MarkovChainDistribution(
        {"x": 0.6, "y": 0.4},
        {"x": {"x": 0.8, "y": 0.2}, "y": {"x": 0.5, "y": 0.5}},
        len_dist=IntegerCategoricalDistribution(0, [0.1, 0.3, 0.4, 0.2]),
    )

    seg_hmm = SegmentalHiddenMarkovModelDistribution(
        emissions=[CategoricalDistribution({"a": 0.8, "b": 0.2}), CategoricalDistribution({"b": 0.6, "c": 0.4})],
        w=[0.7, 0.3],
        transitions=[[0.9, 0.1], [0.4, 0.6]],
        len_dist=IntegerCategoricalDistribution(0, [0.1, 0.3, 0.4, 0.2]),
    )

    return [
        ("categorical", cat3, 10, 3),
        ("intcat_zero_prob", intcat, 10, 3),
        ("binomial", BinomialDistribution(p=0.4, n=6, min_val=3), 20, 7),
        ("geometric", geo, 40, None),
        ("poisson", PoissonDistribution(lam=4.7), 40, None),
        ("poisson_int_lam", PoissonDistribution(lam=3.0), 40, None),
        ("poisson_small_lam", PoissonDistribution(lam=0.2), 40, None),
        ("composite", CompositeDistribution((cat3, intcat)), 30, 9),
        ("composite_nested", CompositeDistribution((cat, CompositeDistribution((cat, intcat)))), 30, 12),
        ("composite_with_mixture", CompositeDistribution((cat, mix_geo)), 50, None),
        ("record", RecordDistribution({"u": cat3, "v": intcat}), 30, 9),
        ("record_nested", RecordDistribution({"u": cat, "w": CompositeDistribution((cat, intcat))}), 30, 12),
        ("record_with_mixture", RecordDistribution({"u": cat, "g": mix_geo}), 50, None),
        (
            "mixture_overlap",
            MixtureDistribution(
                [IntegerCategoricalDistribution(0, [0.7, 0.2, 0.1]), IntegerCategoricalDistribution(1, [0.5, 0.5])],
                [0.6, 0.4],
            ),
            10,
            3,
        ),
        ("mixture_geometrics", mix_geo, 40, None),
        (
            "mixture_zero_weight_gaussian",
            MixtureDistribution([intcat, GaussianDistribution(0.0, 1.0)], [1.0, 0.0]),
            10,
            3,
        ),
        (
            "het_mixture",
            HeterogeneousMixtureDistribution([cat3, CategoricalDistribution({"b": 0.5, "d": 0.5})], [0.5, 0.5]),
            10,
            4,
        ),
        ("optional", OptionalDistribution(cat3, p=0.2, missing_value="MISSING"), 10, 4),
        ("optional_collision", OptionalDistribution(cat3, p=0.2, missing_value="a"), 10, 3),
        ("weighted", WeightedDistribution(cat3), 10, 3),
        ("point_mass", PointMassDistribution("atom"), 5, 1),
        (
            "transform_categorical",
            TransformDistribution(
                CategoricalDistribution({0: 0.7, 1: 0.3}), transform=AffineTransform(loc=10.0, scale=2.0)
            ),
            10,
            2,
        ),
        ("null", NullDistribution(), 5, 1),
        ("sequence", SequenceDistribution(cat, len_dist=IntegerCategoricalDistribution(0, [0.2, 0.5, 0.3])), 30, 7),
        ("sequence_geo_len", SequenceDistribution(cat, len_dist=GeometricDistribution(0.5)), 25, None),
        ("intsetdist", IntegerBernoulliSetDistribution(np.log([0.8, 0.4, 0.01])), 10, 8),
        ("markov_chain", mc, 30, 15),
        ("hmm", hmm, 60, 40),
        ("segmental_hmm", seg_hmm, 60, 40),
    ]


class EnumerationInvariantTestCase(unittest.TestCase):
    def setUp(self):
        self.cases = make_cases()

    def test_ordering_non_increasing(self):
        for name, dist, n, _ in self.cases:
            items = dist.enumerator().top_k(n)
            lps = [lp for _, lp in items]
            for i in range(len(lps) - 1):
                self.assertGreaterEqual(lps[i], lps[i + 1] - TOL, "%s: order violated at %d" % (name, i))

    def test_log_prob_matches_log_density(self):
        for name, dist, n, _ in self.cases:
            with np.errstate(divide="ignore"):
                for v, lp in dist.enumerator().top_k(n):
                    self.assertAlmostEqual(lp, dist.log_density(v), delta=TOL, msg="%s: lp mismatch at %r" % (name, v))

    def test_values_unique(self):
        for name, dist, n, _ in self.cases:
            items = dist.enumerator().top_k(n)
            keys = [freeze(v) for v, _ in items]
            self.assertEqual(len(keys), len(set(keys)), "%s: duplicate values yielded" % name)

    def test_finite_completeness_and_total_mass(self):
        for name, dist, n, expected in self.cases:
            if expected is None:
                continue
            items = list(dist.enumerator())
            self.assertEqual(len(items), expected, "%s: wrong support size" % name)
            total = np.logaddexp.reduce([lp for _, lp in items])
            if name == "optional_collision":
                # When missing_value collides with a base support value, log_density routes
                # it to the missing branch and the base mass of that value is unreachable —
                # the distribution itself sums to 0.2 + 0.8*(0.3+0.2) = 0.6, and the
                # enumeration faithfully reflects that.
                self.assertAlmostEqual(total, np.log(0.6), delta=1e-8, msg=name)
            else:
                self.assertAlmostEqual(total, 0.0, delta=1e-8, msg="%s: total mass != 1" % name)

    def test_top_k(self):
        for name, dist, n, expected in self.cases:
            top3 = dist.enumerator().top_k(3)
            fresh = dist.enumerator().top_k(max(3, n))[:3]
            self.assertEqual([freeze(v) for v, _ in top3], [freeze(v) for v, _ in fresh], name)
            if expected is not None:
                self.assertEqual(len(dist.enumerator().top_k(expected + 100)), expected, name)

    def test_top_p_is_minimal_nucleus(self):
        # top_p(p) must be the smallest descending-probability prefix whose mass reaches p:
        # it agrees with the corresponding prefix of the full enumeration, its mass is >= p, and
        # dropping its last item falls below p (minimality). max_items keeps infinite supports finite.
        for name, dist, n, _ in self.cases:
            for p in (0.5, 0.9):
                nucleus = dist.enumerator().top_p(p, max_items=n)
                prefix = dist.enumerator().top_k(len(nucleus))
                self.assertEqual(
                    [freeze(v) for v, _ in nucleus], [freeze(v) for v, _ in prefix], "%s: top_p != prefix" % name
                )
                mass = sum(math.exp(lp) for _, lp in nucleus)
                if mass >= p and len(nucleus) >= 1 and len(nucleus) < n:
                    mass_without_last = mass - math.exp(nucleus[-1][1])
                    self.assertLess(mass_without_last, p + TOL, "%s: top_p not minimal" % name)
        self.assertEqual(CategoricalDistribution({"a": 0.6, "b": 0.4}).enumerator().top_p(0.0), [])


class CapabilityMatrixTestCase(unittest.TestCase):
    """The four capabilities -- enumeration, arbitrary-index unranking, rank, CDF -- are reachable
    for every family through the shared entry points: ``enumerator()`` for enumeration, its
    ``quantized_index`` for unranking, and ``density_rank`` for rank + CDF (exact head when the family
    enumerates, Monte-Carlo CDF otherwise). This pins that contract across the family matrix."""

    def test_every_enumerable_family_supports_all_four(self):
        from mixle.enumeration.density_rank import density_rank

        cases = make_cases()
        cases.append(("erdos_renyi", ErdosRenyiGraphDistribution(0.3, num_nodes=4), 10, None))
        cases.append(
            (
                "stochastic_block",
                StochasticBlockGraphDistribution(np.array([[0.7, 0.2], [0.2, 0.5]]), block_assignments=[0, 0, 1, 1]),
                10,
                None,
            )
        )
        for name, dist, _, _ in cases:
            # cap 1: enumeration yields a most-probable value
            first = next(iter(dist.enumerator()))
            value = first[0]
            # cap 2: arbitrary-index unranking index builds from the enumeration
            dist.enumerator().quantized_index(max_bits=10.0)
            # caps 3 + 4: rank ("index of") and CDF via the exact head (enumerable => exact)
            r = density_rank(dist, value, n_samples=200, seed=0)
            self.assertIn(r.method, ("exact-head", "exact-exhausted"), "%s: rank not exact" % name)
            self.assertTrue(0.0 <= r.cumulative_probability <= 1.0 + TOL, "%s: cdf out of range" % name)
            self.assertEqual(r.rank, 0, "%s: most-probable value should rank 0" % name)

    def test_nonenumerable_families_have_sampling_cdf(self):
        # Continuous / coupled families cannot enumerate (uncountable or non-decomposable support), so
        # enumeration + unranking + exact rank are N/A -- but CDF ("probability-ordered cumulative")
        # is still available for any samplable family via density_rank's Monte-Carlo fallback.
        from mixle.enumeration.density_rank import density_rank
        from mixle.stats.bayes.dirichlet import DirichletDistribution
        from mixle.stats.bayes.normal_gamma import NormalGammaDistribution
        from mixle.stats.bayes.symmetric_dirichlet import SymmetricDirichletDistribution
        from mixle.stats.directional.von_mises_fisher import VonMisesFisherDistribution
        from mixle.stats.latent.gaussian_mixture import GaussianMixtureDistribution
        from mixle.stats.multivariate.multivariate_gaussian import MultivariateGaussianDistribution

        # (name, dist, value, expected density_rank method): families with no enumerator still expose a
        # CDF -- exact-analytic where a closed-form probability-ordered cumulative exists (the
        # multivariate Gaussian via chi-square, von Mises-Fisher via the cosine marginal), Monte-Carlo
        # otherwise (continuous leaves' probability-ordered CDF, parameter priors, MVN mixtures).
        cases = [
            ("gaussian", GaussianDistribution(0.0, 1.0), 0.5, "sampling"),
            ("gamma", GammaDistribution(2.0, 2.0), 3.0, "sampling"),
            ("student_t", StudentTDistribution(5.0, 0.0, 1.0), 0.5, "sampling"),
            ("mvn", MultivariateGaussianDistribution(np.zeros(2), np.eye(2)), np.array([0.5, 0.5]), "exact-analytic"),
            (
                "vmf",
                VonMisesFisherDistribution(np.array([1.0, 0.0, 0.0]), 3.0),
                np.array([0.0, 1.0, 0.0]),
                "exact-analytic",
            ),
            ("dirichlet", DirichletDistribution(np.array([2.0, 3.0, 1.5])), np.array([0.3, 0.4, 0.3]), "sampling"),
            ("symdirichlet", SymmetricDirichletDistribution(2.0, 4), np.array([0.25, 0.25, 0.25, 0.25]), "sampling"),
            ("normgamma", NormalGammaDistribution(0.0, 1.0, 2.0, 1.0), (0.0, 1.0), "sampling"),
            (
                "mvnmixture",
                GaussianMixtureDistribution([np.zeros(2), np.ones(2)], [np.ones(2), np.ones(2)], [0.6, 0.4]),
                np.array([0.2, 0.3]),
                "sampling",
            ),
            ("erdos_renyi_unsized", ErdosRenyiGraphDistribution(0.3), None, None),
        ]
        for name, dist, x, expected_method in cases:
            self.assertFalse(supports_enumeration(dist), "%s: should not enumerate" % name)
            if x is None:
                continue
            r = density_rank(dist, x, n_samples=3000, seed=1)
            self.assertEqual(r.method, expected_method, "%s: unexpected CDF method" % name)
            self.assertTrue(0.0 <= r.cumulative_probability <= 1.0, "%s: cdf out of range" % name)

    def test_nonenumerable_families_have_arbitrary_index_and_enumeration(self):
        # The remaining two capabilities -- arbitrary-index and (descending-density) enumeration -- are
        # reachable for every samplable family through the base-class density_quantile /
        # density_enumeration (exact where overridden, e.g. MVN; Monte-Carlo representative otherwise).
        from mixle.stats.bayes.dirichlet import DirichletDistribution
        from mixle.stats.latent.gaussian_mixture import GaussianMixtureDistribution
        from mixle.stats.multivariate.multivariate_gaussian import MultivariateGaussianDistribution

        # Monte-Carlo representatives: density falls (or is non-increasing) as q -> 1, enumeration is
        # descending and the requested size.
        for dist in (
            DirichletDistribution(np.array([2.0, 3.0, 1.5])),
            GaussianDistribution(0.0, 1.0),
            GaussianMixtureDistribution([np.zeros(2), np.ones(2)], [np.ones(2), np.ones(2)], [0.6, 0.4]),
        ):
            lps = [float(dist.log_density(dist.density_quantile(q, n_samples=6000, seed=3))) for q in (0.05, 0.5, 0.95)]
            self.assertTrue(all(lps[i] >= lps[i + 1] - 0.5 for i in range(len(lps) - 1)))
            enum = dist.density_enumeration(6, n_samples=6000, seed=3)
            self.assertEqual(len(enum), 6)
            self.assertTrue(all(enum[i][1] >= enum[i + 1][1] for i in range(len(enum) - 1)))

        # The exact override (MVN) takes precedence over the sampling default: deterministic + inverts.
        mvn = MultivariateGaussianDistribution(np.zeros(2), np.eye(2))
        np.testing.assert_array_equal(mvn.density_quantile(0.5), mvn.density_quantile(0.5))
        self.assertAlmostEqual(mvn.density_cumulative(mvn.density_quantile(0.4)), 0.4, delta=1e-9)


class BruteForceCrossCheckTestCase(unittest.TestCase):
    """Independently constructs small supports, scores them with log_density, and compares."""

    def assert_matches_brute(self, dist, support, name):
        with np.errstate(divide="ignore"):
            brute = [(v, dist.log_density(v)) for v in support]
        brute = [(v, lp) for v, lp in brute if lp > -np.inf]
        brute.sort(key=lambda u: -u[1])
        items = list(dist.enumerator())
        self.assertEqual(len(items), len(brute), "%s: size mismatch" % name)
        np.testing.assert_allclose(
            [lp for _, lp in items], [lp for _, lp in brute], atol=TOL, err_msg="%s: score sequence mismatch" % name
        )

        # Compare value sets per strictly-descending tier to avoid tie-order false failures.
        def tiers(pairs):
            out = {}
            for v, lp in pairs:
                out.setdefault(round(lp, 8), set()).add(freeze(v))
            return out

        self.assertEqual(tiers(items), tiers(brute), "%s: tier values mismatch" % name)

    def test_composite(self):
        cat = CategoricalDistribution({"a": 0.5, "b": 0.3, "c": 0.2})
        intcat = IntegerCategoricalDistribution(0, [0.6, 0.4])
        dist = CompositeDistribution((cat, intcat))
        support = [(s, i) for s in "abc" for i in (0, 1)]
        self.assert_matches_brute(dist, support, "composite")

    def test_record(self):
        cat = CategoricalDistribution({"a": 0.5, "b": 0.3, "c": 0.2})
        intcat = IntegerCategoricalDistribution(0, [0.6, 0.4])
        dist = RecordDistribution({"u": cat, "v": intcat})
        support = [{"u": s, "v": i} for s in "abc" for i in (0, 1)]
        self.assert_matches_brute(dist, support, "record")

    def test_segmental_hmm(self):
        # Segments here are structured composite tuples scored by different emission classes per
        # state -- exercising the "arbitrary segment" generality on top of the standard HMM forward.
        e0 = CompositeDistribution(
            (CategoricalDistribution({"x": 0.7, "y": 0.3}), IntegerCategoricalDistribution(0, [0.6, 0.4]))
        )
        e1 = CompositeDistribution(
            (CategoricalDistribution({"x": 0.2, "y": 0.8}), IntegerCategoricalDistribution(0, [0.5, 0.5]))
        )
        dist = SegmentalHiddenMarkovModelDistribution(
            emissions=[e0, e1],
            w=[0.6, 0.4],
            transitions=[[0.7, 0.3], [0.5, 0.5]],
            len_dist=IntegerCategoricalDistribution(0, [0.2, 0.5, 0.3]),
        )
        seg_alpha = [("x", 0), ("x", 1), ("y", 0), ("y", 1)]
        support = [[]]
        for length in (1, 2):
            support += [list(t) for t in itertools.product(seg_alpha, repeat=length)]
        self.assert_matches_brute(dist, support, "segmental_hmm")

    def test_conditional_enumeration_matches_brute(self):
        # Most-probable-completion query: enumerating with some fields/positions fixed must equal the
        # brute-force list of consistent full outcomes sorted by joint log_density, and each yielded
        # log_prob is the full joint log_density (fixed parts enter as a constant offset).
        cat = CategoricalDistribution({"a": 0.5, "b": 0.3, "c": 0.2})
        cat2 = CategoricalDistribution({"p": 0.7, "q": 0.3})
        intcat = IntegerCategoricalDistribution(0, [0.6, 0.3, 0.1])

        rec = RecordDistribution({"u": cat, "v": cat2, "n": intcat})
        rec_support = [{"u": u, "v": v, "n": n} for u in "abc" for v in "pq" for n in (0, 1, 2)]
        for given in ({"v": "q"}, {"u": "a", "n": 1}, {"u": "b", "v": "p", "n": 0}):
            got = list(rec.conditional_enumerator(given))
            cons = sorted(
                (r for r in rec_support if all(r[k] == val for k, val in given.items())),
                key=lambda r: -rec.log_density(r),
            )
            self.assertEqual([v for v, _ in got], cons, "record given=%r" % given)
            for v, lp in got:
                self.assertAlmostEqual(lp, rec.log_density(v), delta=TOL, msg="record lp given=%r" % given)

        comp = CompositeDistribution((cat, cat2, intcat))
        comp_support = [(u, v, n) for u in "abc" for v in "pq" for n in (0, 1, 2)]
        for given in ({1: "q"}, {0: "a", 2: 1}):
            got = list(comp.conditional_enumerator(given))
            cons = sorted(
                (t for t in comp_support if all(t[k] == val for k, val in given.items())),
                key=lambda t: -comp.log_density(t),
            )
            self.assertEqual([v for v, _ in got], cons, "composite given=%r" % given)

        # impossible fixed value -> empty; bad key -> ValueError
        self.assertEqual(list(rec.conditional_enumerator({"u": "ZZ"})), [])
        self.assertEqual(list(comp.conditional_enumerator({0: "ZZ"})), [])
        with self.assertRaises(ValueError):
            rec.conditional_enumerator({"nope": 1})
        with self.assertRaises(ValueError):
            comp.conditional_enumerator({9: "a"})

    def test_record_nested_matches_composite_ranks(self):
        # A Record over the same children as a Composite scores identically (sum of field
        # log-densities), so count_dp_rank must agree row-for-row through the dict relabelling.
        from mixle.enumeration.density_rank import count_dp_rank

        cat = CategoricalDistribution({"a": 0.5, "b": 0.3, "c": 0.2})
        intcat = IntegerCategoricalDistribution(0, [0.7, 0.2, 0.1])
        rec = RecordDistribution({"u": cat, "w": CompositeDistribution((cat, intcat))})
        comp = CompositeDistribution((cat, CompositeDistribution((cat, intcat))))
        for s in "abc":
            for s2 in "abc":
                for i in (0, 1, 2):
                    row = {"u": s, "w": (s2, i)}
                    tup = (s, (s2, i))
                    rr = count_dp_rank(rec, row, oversample=64)
                    cr = count_dp_rank(comp, tup, oversample=64)
                    self.assertEqual((rr.window_lower, rr.window_upper), (cr.window_lower, cr.window_upper), row)

    def test_sound_top_k_correct_for_nested_mixture(self):
        # Mixture of nested composite/sequence models: the tropical seek order is badly displaced,
        # so sound_top_k's mass certificate (not the ordering) must yield the exact true-descending
        # top-k AND an arbitrary [start, start+k) slice, matching the exact best-first enumerator.
        from mixle.stats.combinator.composite import CompositeDistribution
        from mixle.stats.combinator.sequence import SequenceDistribution

        def nested(seed):
            r = np.random.RandomState(seed)
            seq = SequenceDistribution(
                IntegerCategoricalDistribution(0, list(r.dirichlet(np.ones(4)))),
                len_dist=IntegerCategoricalDistribution(1, list(r.dirichlet(np.ones(4)))),
            )
            return CompositeDistribution((seq, IntegerCategoricalDistribution(0, list(r.dirichlet(np.ones(5))))))

        rng = np.random.RandomState(0)
        mix = MixtureDistribution([nested(s) for s in (1, 2, 3)], list(rng.dirichlet(np.ones(3))))
        exact = list(itertools.islice(mix.enumerator(), 20))

        def tiers(pairs):
            out = {}
            for v, lp in pairs:
                out.setdefault(round(lp, 6), set()).add(freeze(v))
            return out

        first = sound_top_k(mix, 8, budget_bits=30)
        second = sound_top_k(mix, 5, start=10, budget_bits=30)
        # A genuinely-completable query on a well-behaved (finite-mass, fast-concentrating) mixture
        # must come back certified -- not just correct -- since that is the whole point of the API.
        self.assertEqual(first.status, "certified")
        self.assertTrue(first.sound)
        self.assertEqual(second.status, "certified")
        self.assertTrue(second.sound)
        self.assertEqual(tiers(first.items), tiers(exact[:8]))
        self.assertEqual(tiers(second.items), tiers(exact[10:15]))

    def test_mixture_overlapping(self):
        dist = MixtureDistribution(
            [IntegerCategoricalDistribution(0, [0.7, 0.2, 0.1]), IntegerCategoricalDistribution(1, [0.5, 0.5])],
            [0.6, 0.4],
        )
        self.assert_matches_brute(dist, list(range(-1, 5)), "mixture")

    def test_sequence(self):
        cat = CategoricalDistribution({"a": 0.7, "b": 0.3})
        dist = SequenceDistribution(cat, len_dist=IntegerCategoricalDistribution(0, [0.2, 0.5, 0.3]))
        support = [[]] + [list(t) for L in (1, 2) for t in itertools.product("ab", repeat=L)]
        self.assert_matches_brute(dist, support, "sequence")

    def test_markov_chain(self):
        dist = MarkovChainDistribution(
            {"x": 0.6, "y": 0.4},
            {"x": {"x": 0.8, "y": 0.2}, "y": {"x": 0.5, "y": 0.5}},
            len_dist=IntegerCategoricalDistribution(0, [0.1, 0.3, 0.4, 0.2]),
        )
        support = [[]] + [list(t) for L in (1, 2, 3) for t in itertools.product("xy", repeat=L)]
        self.assert_matches_brute(dist, support, "markov_chain")

    def test_hmm(self):
        dist = HiddenMarkovModelDistribution(
            topics=[CategoricalDistribution({"a": 0.8, "b": 0.2}), CategoricalDistribution({"b": 0.6, "c": 0.4})],
            w=[0.7, 0.3],
            transitions=[[0.9, 0.1], [0.4, 0.6]],
            len_dist=IntegerCategoricalDistribution(0, [0.1, 0.3, 0.4, 0.2]),
        )
        support = [[]] + [list(t) for L in (1, 2, 3) for t in itertools.product("abc", repeat=L)]
        self.assert_matches_brute(dist, support, "hmm")

    def test_intsetdist(self):
        dist = IntegerBernoulliSetDistribution(np.log([0.8, 0.4, 0.01]))
        support = [sorted(c) for r in range(4) for c in itertools.combinations(range(3), r)]
        support = [list(c) for c in support]
        self.assert_matches_brute(dist, support, "intsetdist")


class SoundTopKCertificateTestCase(unittest.TestCase):
    """Regression coverage for MXR-080-0211/0212: sound_top_k must never present a truncated,
    uncertified prefix as though it were the proven-complete top-k."""

    def test_geometric_infinite_support_deepens_instead_of_false_exhaustion(self):
        # MXR-080-0211 repro: GeometricDistribution has infinite support, so no finite prefix is
        # ever "the complete answer". A small initial budget_bits builds a heavily truncated
        # count-budget index; the prior implementation inferred "support exhausted" straight from
        # `len(seen) < need` without first checking `index.truncated`, so it returned that
        # truncated handful as though it were the true (complete) top 10. At these exact
        # parameters it used to silently return only the first 2 values instead of 10.
        geo = GeometricDistribution(0.5)
        result = sound_top_k(geo, 10, budget_bits=1.0)
        exact = geo.enumerator().top_k(10)

        self.assertIsInstance(result, SoundTopKResult)
        self.assertNotEqual(result.status, "exhausted")  # infinite support can never be "exhausted"
        self.assertEqual(result.status, "certified")
        self.assertTrue(result.sound)
        self.assertEqual(len(result.items), 10)
        self.assertEqual([v for v, _ in result.items], [v for v, _ in exact])
        for (_, lp), (_, elp) in zip(result.items, exact):
            self.assertAlmostEqual(lp, float(elp), places=9)
        # It had to deepen past the tiny starting budget to reach a certified answer -- the old
        # code returned on the very first (truncated) pass instead.
        self.assertGreater(result.budget_bits, 1.0)

    def test_finite_support_reports_true_exhaustion_without_overcorrecting(self):
        # Negative control: a distribution with a genuinely small, FINITE support (3 outcomes)
        # must still promptly report status="exhausted" with fewer than `k` items when that is the
        # honest answer -- the fix must not overcorrect into deepening forever regardless.
        cat = CategoricalDistribution({"a": 0.5, "b": 0.3, "c": 0.2})
        result = sound_top_k(cat, 10, budget_bits=1.0)
        self.assertEqual(result.status, "exhausted")
        self.assertTrue(result.sound)
        self.assertEqual(len(result.items), 3)
        self.assertEqual({v for v, _ in result.items}, {"a", "b", "c"})
        # Stopped well short of the budget cap -- true exhaustion was detected, not forced by it.
        self.assertLess(result.budget_bits, 512.0)

    def test_result_is_structured_with_explicit_status_for_certified_and_incomplete(self):
        # MXR-080-0212: the return type must let a caller distinguish a proof from a budget-cap
        # guess -- exercise both outcomes explicitly.
        geo = GeometricDistribution(0.5)

        certified = sound_top_k(geo, 2, budget_bits=10.0)
        self.assertIsInstance(certified, SoundTopKResult)
        self.assertEqual(certified.status, "certified")
        self.assertTrue(certified.sound)
        self.assertGreaterEqual(certified.remaining_mass_bound, 0.0)
        self.assertEqual([v for v, _ in certified.items], [1, 2])

        # max_budget_bits == budget_bits forbids any deepening, and two items is not enough for
        # the mass certificate to fire (accumulated 0.75 leaves exactly the worst kept item's own
        # 0.25 of probability unaccounted for -- not strictly less), so this must fail closed.
        incomplete = sound_top_k(geo, 2, budget_bits=1.0, max_budget_bits=1.0)
        self.assertIsInstance(incomplete, SoundTopKResult)
        self.assertEqual(incomplete.status, "incomplete")
        self.assertFalse(incomplete.sound)
        self.assertGreaterEqual(incomplete.remaining_mass_bound, 0.0)
        self.assertEqual(len(incomplete.items), 2)  # best-effort prefix, still returned -- just marked unsound

    def test_total_mass_validation_rejects_nonsensical_bounds(self):
        # MXR-080-0212: the mass proof trusts total_mass as a caller-supplied bound, but must
        # still reject values that cannot possibly be a valid probability-mass upper bound.
        geo = GeometricDistribution(0.5)
        for bad in (0.0, -1.0, -0.5, float("nan"), float("inf"), float("-inf")):
            with self.assertRaises(ValueError):
                sound_top_k(geo, 3, total_mass=bad)

    def test_mass_lower_bound_never_overstates_the_true_sum(self):
        # MXR-080-0212: the accumulated-mass bookkeeping must never OVERSTATE the true accumulated
        # mass (an overstatement would understate the derived remaining-mass bound, which is the
        # direction that could make the certificate fire when it should not).
        from fractions import Fraction

        from mixle.enumeration.best_first import _fold_mass_lower_bound

        # Classic ties-to-even construction where plain nearest-rounded addition OVERSTATES the
        # exact sum: nextafter(1.0, inf) has an odd last mantissa bit and 2**-53 is exactly half a
        # ULP at that magnitude, so the exact sum sits exactly halfway between two representable
        # doubles and ties-to-even rounds to the larger (even-mantissa) one, past the true value.
        a = math.nextafter(1.0, math.inf)
        b = 2.0**-53
        naive = a + b
        exact = Fraction(a) + Fraction(b)
        self.assertGreater(Fraction(naive), exact, "test construction must itself overstate")

        lower = _fold_mass_lower_bound(_fold_mass_lower_bound(0.0, a), b)
        self.assertLessEqual(Fraction(lower), exact)

        # Broader property check across random descending-order sequences (sound_top_k always
        # accumulates largest-probability-first): the fixed accumulator must never exceed the
        # exact sum, regardless of how any individual naive addition would have rounded.
        rng = np.random.RandomState(0)
        for _ in range(200):
            terms = sorted(rng.uniform(0.0, 1.0, size=int(rng.randint(1, 40))).tolist(), reverse=True)
            exact_sum = sum(Fraction(t) for t in terms)
            lower = 0.0
            for t in terms:
                lower = _fold_mass_lower_bound(lower, t)
            self.assertLessEqual(Fraction(lower), exact_sum)


class InfiniteSupportMassDominanceTestCase(unittest.TestCase):
    """The first N items of an infinite enumeration must contain every value that is
    strictly more probable than the N-th item."""

    def test_poisson_and_geometric(self):
        # Geometric support starts at 1 (its log_density applies the pmf formula outside
        # the support too, so scanning from 0 would test values the distribution excludes).
        for dist, lo in ((PoissonDistribution(4.7), 0), (PoissonDistribution(3.0), 0), (GeometricDistribution(0.3), 1)):
            items = dist.enumerator().top_k(40)
            cutoff = items[-1][1]
            seen = set(v for v, _ in items)
            for x in range(lo, 200):
                if dist.log_density(x) > cutoff + TOL:
                    self.assertIn(x, seen, "%s missing %d" % (dist, x))

    def test_mixture_of_geometrics(self):
        dist = MixtureDistribution([GeometricDistribution(0.5), GeometricDistribution(0.1)], [0.6, 0.4])
        items = dist.enumerator().top_k(40)
        cutoff = items[-1][1]
        seen = set(v for v, _ in items)
        for x in range(1, 300):
            if dist.log_density(x) > cutoff + TOL:
                self.assertIn(x, seen, "mixture missing %d" % x)


class HmmTerminalValuesEnumerationTestCase(unittest.TestCase):
    """Enumeration of the terminal-VALUE stopping-time support (sequences ending at the first
    terminal emission), scored by the plain forward likelihood."""

    def _dist(self):
        return HiddenMarkovModelDistribution(
            [
                CategoricalDistribution({"a": 0.5, "b": 0.3, ".": 0.2}),
                CategoricalDistribution({"a": 0.2, "b": 0.3, ".": 0.5}),
            ],
            w=[0.6, 0.4],
            transitions=[[0.7, 0.3], [0.4, 0.6]],
            terminal_values={"."},
        )

    def _brute(self, d, max_len=10):
        out = []
        for length in range(1, max_len + 1):
            for pre in itertools.product(["a", "b"], repeat=length - 1):
                seq = list(pre) + ["."]
                out.append((seq, d.log_density(seq)))
        out.sort(key=lambda kv: -kv[1])
        return out

    def test_top_k_matches_brute_force(self):
        d = self._dist()
        top = d.enumerator().top_k(30)
        brute = self._brute(d)[:30]
        for (seq, lp), (bseq, blp) in zip(top, brute):
            self.assertEqual(seq, bseq)
            self.assertAlmostEqual(lp, blp, places=10)

    def test_scores_equal_log_density(self):
        d = self._dist()
        for seq, lp in d.enumerator().top_k(30):
            self.assertEqual(seq[-1], ".")  # ends at the terminal value
            self.assertNotIn(".", seq[:-1])  # no interior terminal value
            self.assertAlmostEqual(lp, d.log_density(seq), places=10)

    def test_descending_order_and_mass_to_one(self):
        d = self._dist()
        items = d.enumerator().top_k(500)
        lps = [lp for _, lp in items]
        self.assertTrue(all(lps[i] >= lps[i + 1] - 1e-12 for i in range(len(lps) - 1)))
        self.assertAlmostEqual(sum(math.exp(lp) for lp in lps), 1.0, delta=0.05)

    def test_seek_and_from_index(self):
        d = self._dist()
        full = d.enumerator().top_k(20)
        self.assertEqual([s for s, _ in d.enumerator().from_index(5, 10)], [s for s, _ in full[5:10]])
        for k in (0, 3, 7):
            self.assertEqual(d.enumerator().seek(k).value, full[k][0])

    def test_terminal_values_with_length_dist_raises(self):
        # terminal_values + a non-Null length distribution is ambiguous and is rejected.
        with self.assertRaises(EnumerationError):
            HiddenMarkovModelDistribution(
                [CategoricalDistribution({"a": 0.5, ".": 0.5})],
                w=[1.0],
                transitions=[[1.0]],
                len_dist=GeometricDistribution(0.5),
                terminal_values={"."},
            ).enumerator()


class EnumerationErrorTestCase(unittest.TestCase):
    def test_continuous_raises(self):
        with self.assertRaises(EnumerationError):
            GaussianDistribution(0.0, 1.0).enumerator()

    def test_composite_error_names_child(self):
        cat = CategoricalDistribution({"a": 1.0})
        with self.assertRaises(EnumerationError) as cm:
            CompositeDistribution((cat, GaussianDistribution(0.0, 1.0))).enumerator()
        self.assertIn("dists[1]", str(cm.exception))
        self.assertIn("GaussianDistribution", str(cm.exception))

    def test_nested_error_path(self):
        cat = CategoricalDistribution({"a": 1.0})
        inner = MixtureDistribution([GaussianDistribution(0.0, 1.0), cat], [0.5, 0.5])
        with self.assertRaises(EnumerationError) as cm:
            CompositeDistribution((cat, inner)).enumerator()
        msg = str(cm.exception)
        self.assertIn("dists[1]", msg)
        self.assertIn("components[0]", msg)
        self.assertIn("GaussianDistribution", msg)

    def test_categorical_default_value_raises(self):
        with self.assertRaises(EnumerationError):
            CategoricalDistribution({"a": 0.9}, default_value=0.1).enumerator()

    def test_sequence_without_len_dist_raises(self):
        with self.assertRaises(EnumerationError):
            SequenceDistribution(CategoricalDistribution({"a": 1.0})).enumerator()

    def test_sequence_len_normalized_raises(self):
        with self.assertRaises(EnumerationError):
            SequenceDistribution(
                CategoricalDistribution({"a": 1.0}), len_dist=GeometricDistribution(0.5), len_normalized=True
            ).enumerator()

    def test_ignored_raises(self):
        with self.assertRaises(EnumerationError):
            IgnoredDistribution(CategoricalDistribution({"a": 1.0})).enumerator()

    def test_optional_without_p_raises(self):
        with self.assertRaises(EnumerationError):
            OptionalDistribution(CategoricalDistribution({"a": 1.0})).enumerator()

    # terminal_values enumeration (with a Null length distribution) is supported; see
    # HmmTerminalValuesEnumerationTestCase. The non-Null-length case is rejected there too.

    def test_markov_chain_default_value_raises(self):
        with self.assertRaises(EnumerationError):
            MarkovChainDistribution(
                {"x": 1.0}, {"x": {"x": 1.0}}, default_value=0.1, len_dist=GeometricDistribution(0.5)
            ).enumerator()

    def test_zero_weight_type_incompatible_component(self):
        # A zero-weight Gaussian mixed with a string categorical must never be evaluated,
        # neither for stream generation nor for exact re-scoring.
        dist = MixtureDistribution(
            [CategoricalDistribution({"a": 0.6, "b": 0.4}), GaussianDistribution(0.0, 1.0)], [1.0, 0.0]
        )
        items = list(dist.enumerator())
        self.assertEqual([v for v, _ in items], ["a", "b"])
        np.testing.assert_allclose([lp for _, lp in items], np.log([0.6, 0.4]), atol=TOL)

    def test_supports_enumeration_predicate(self):
        self.assertTrue(supports_enumeration(CategoricalDistribution({"a": 1.0})))
        self.assertFalse(supports_enumeration(GaussianDistribution(0.0, 1.0)))


class FlagshipCompositionTestCase(unittest.TestCase):
    """The motivating case: a composite of a categorical and a mixture of geometrics."""

    def test_composite_of_categorical_and_mixture(self):
        m = MixtureDistribution([GeometricDistribution(0.5), GeometricDistribution(0.1)], [0.6, 0.4])
        c = CompositeDistribution((CategoricalDistribution({"a": 0.7, "b": 0.3}), m))
        items = c.enumerator().top_k(50)
        lps = [lp for _, lp in items]
        self.assertEqual(len(items), 50)
        for i in range(len(lps) - 1):
            self.assertGreaterEqual(lps[i], lps[i + 1] - TOL)
        for v, lp in items:
            self.assertAlmostEqual(lp, c.log_density(v), delta=TOL)
        # mass dominance against a brute scan of the head of the support
        cutoff = lps[-1]
        seen = set(freeze(v) for v, _ in items)
        for s in "ab":
            for x in range(1, 100):
                if c.log_density((s, x)) > cutoff + TOL:
                    self.assertIn(freeze((s, x)), seen)


if __name__ == "__main__":
    unittest.main()
