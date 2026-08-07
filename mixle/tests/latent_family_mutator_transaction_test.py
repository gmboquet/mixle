"""Latent-family mutator audit: every accumulator's mutators are wholly transactional.

Review passes eight through ten established the family contract on the chain/tree HMMs (see
D-0150..D-0152): a mutator validates candidates before assigning anything, validates its RESULT
for element and aggregate finiteness, and on any failure restores every object it mutated --
counts, children, adopted containers, and pooled-mapping entries, in place and
identity-preserving. This audit measured the same defect classes alive in eleven more
accumulators (mixture, semi-supervised mixture, joint mixture, hierarchical mixture, lookback /
segmental / semi-supervised / scheduled HMMs, structured HMM, IOHMM, explicit-duration HMM):
partial mutation on child failure in combine/from_value/scale, silent infinite aggregates from
finite elements, silent scale overflow, unvalidated key_replace ingestion, partial key_merge
pools -- and, on the segmental and lookback HMMs, the overreach direction: their unconditional
state-mass equality rejected a KEYED accumulator's own value() round-trip and every keyed fit's
M-step (the chain twin learned that in STAT-RR5-2). Every probe here reproduces a measured
defect; the legitimacy controls pin that the repaired guards accept what the library
legitimately produces.
"""

import unittest

import numpy as np
from numpy.random import RandomState

import mixle.stats as S
from mixle.stats import GaussianEstimator, SequenceEstimator
from mixle.stats.latent.hierarchical_mixture import HierarchicalMixtureEstimator
from mixle.stats.latent.joint_mixture import JointMixtureEstimator
from mixle.stats.latent.lookback_hidden_markov_model import LookbackHiddenMarkovModelEstimator
from mixle.stats.latent.mixture import MixtureEstimator
from mixle.stats.latent.scheduled_hidden_markov_model import Homogeneous, ScheduledHMMEstimator
from mixle.stats.latent.segmental_hidden_markov_model import SegmentalHiddenMarkovEstimator
from mixle.stats.latent.semi_supervised_hidden_markov_model import SemiSupervisedHiddenMarkovEstimator
from mixle.stats.latent.semi_supervised_mixture import SemiSupervisedMixtureEstimator
from mixle.stats.latent.structured_hmm import (
    DenseTransition,
    ExplicitDurationHMM,
    InputOutputHMM,
    StructuredHMM,
)

SCALARS = [0.1, 2.3, -1.0, 5.5, 3.3, 0.7, -0.4, 4.1]
SEQS = [[0.1, 2.3, -1.0], [5.5, 3.3], [0.7, -0.4, 4.1, 1.0], [2.0, 2.5]]


def _gaussians(n=2):
    return [GaussianEstimator() for _ in range(n)]


def _initialize_seeder(estimator, data):
    def seed():
        acc = estimator.accumulator_factory().make()
        rng = RandomState(5)
        for x in data:
            acc.initialize(x, 1.0, rng)
        return acc

    return seed


def _update_seeder(estimator, distribution, data):
    def seed():
        acc = estimator.accumulator_factory().make()
        for x in data:
            acc.update(x, 1.0, distribution)
        return acc

    return seed


def _structured_family():
    dist = StructuredHMM(
        [S.GaussianDistribution(-2.0, 1.0), S.GaussianDistribution(2.0, 1.0)],
        [0.5, 0.5],
        DenseTransition(np.array([[0.7, 0.3], [0.3, 0.7]])),
    )
    est = dist.estimator()
    return {"seed": _update_seeder(est, dist, SEQS), "make": est.accumulator_factory().make, "child_slot": 2}


def _iohmm_family():
    dist = InputOutputHMM(
        [S.GaussianDistribution(-2.0, 1.0), S.GaussianDistribution(2.0, 1.0)],
        [0.5, 0.5],
        [DenseTransition(np.array([[0.7, 0.3], [0.3, 0.7]])), DenseTransition(np.array([[0.4, 0.6], [0.6, 0.4]]))],
    )
    est = dist.estimator()
    records = [[(0.1, 0), (2.3, 1), (-1.0, 0)], [(5.5, 1), (3.3, 0)]]
    return {"seed": _update_seeder(est, dist, records), "make": est.accumulator_factory().make, "child_slot": 2}


def _edhmm_family():
    dist = ExplicitDurationHMM(
        [S.GaussianDistribution(-2.0, 1.0), S.GaussianDistribution(2.0, 1.0)],
        [0.6, 0.4],
        np.array([[0.0, 1.0], [1.0, 0.0]]),
        np.array([[0.2, 0.5, 0.3], [0.5, 0.3, 0.2]]),
        3,
    )
    est = dist.estimator()
    return {"seed": _update_seeder(est, dist, SEQS), "make": est.accumulator_factory().make, "child_slot": 3}


def _scheduled_family():
    est = ScheduledHMMEstimator(
        2,
        Homogeneous(),
        S.IntegerCategoricalDistribution(min_val=0, p_vec=[0.5, 0.5]).estimator(),
        pseudo_count=0.2,
    )

    def seed():
        acc = est.accumulator_factory().make()
        acc.seq_initialize([[0, 1, 0], [1, 1], [0, 1, 1, 0], [1, 0]], np.ones(4), RandomState(4))
        return acc

    return {"seed": seed, "make": est.accumulator_factory().make, "child_slot": 3}


def _families():
    """name -> {seed, make, child_slot} for every audited family (unkeyed)."""
    fams = {}
    for name, est, data in (
        ("mixture", MixtureEstimator(_gaussians()), SCALARS),
        ("semi_supervised_mixture", SemiSupervisedMixtureEstimator(_gaussians()), [(x, None) for x in SCALARS]),
        ("joint_mixture", JointMixtureEstimator(_gaussians(), _gaussians()), [(x, x + 1.0) for x in SCALARS]),
        ("hierarchical_mixture", HierarchicalMixtureEstimator(_gaussians(), num_mixtures=2), SEQS),
        (
            "lookback_hmm",
            LookbackHiddenMarkovModelEstimator([SequenceEstimator(GaussianEstimator()) for _ in range(2)], lag=0),
            SEQS,
        ),
        ("segmental_hmm", SegmentalHiddenMarkovEstimator(_gaussians()), SEQS),
        ("semi_supervised_hmm", SemiSupervisedHiddenMarkovEstimator(_gaussians()), [(list(s), None) for s in SEQS]),
    ):
        slot = {
            "mixture": 1,
            "semi_supervised_mixture": 1,
            "joint_mixture": 3,
            "hierarchical_mixture": 2,
            "lookback_hmm": 5,
            "segmental_hmm": 4,
            "semi_supervised_hmm": 2,
        }[name]
        fams[name] = {
            "seed": _initialize_seeder(est, data),
            "make": est.accumulator_factory().make,
            "child_slot": slot,
        }
    fams["structured_hmm"] = _structured_family()
    fams["iohmm"] = _iohmm_family()
    fams["edhmm"] = _edhmm_family()
    fams["scheduled_hmm"] = _scheduled_family()
    return fams


def _describe(x):
    """Structural fingerprint of a value tuple, for exact before/after comparison."""
    if isinstance(x, np.ndarray):
        return ("nd", x.shape, x.tobytes())
    if isinstance(x, (tuple, list)):
        return (type(x).__name__, tuple(_describe(v) for v in x))
    return ("obj", repr(x))


def _poison_child(value, slot):
    out = list(value)
    seq = list(out[slot])
    seq[-1] = None
    out[slot] = tuple(seq) if isinstance(value[slot], tuple) else seq
    return tuple(out)


def _scaled_value(value, factor):
    from mixle.stats.compute.pdist import scale_suff_stat

    return scale_suff_stat(value, factor)


def _rescaled_preserving_ints(value, factor):
    scaled = list(_scaled_value(value, factor))
    for i, part in enumerate(value):
        if isinstance(part, (int, np.integer)) and not isinstance(part, bool):
            scaled[i] = part
    return tuple(scaled)


class FamilyMutatorTransactionTest(unittest.TestCase):
    """Every audited family: transactional mutators with finiteness postconditions."""

    @classmethod
    def setUpClass(cls):
        cls.families = _families()

    def test_combine_child_failure_rolls_the_whole_accumulator_back(self):
        for name, fx in self.families.items():
            with self.subTest(family=name):
                a, b = fx["seed"](), fx["seed"]()
                before = _describe(a.value())
                with self.assertRaises((TypeError, ValueError, AttributeError)):
                    a.combine(_poison_child(b.value(), fx["child_slot"]))
                self.assertEqual(_describe(a.value()), before)

    def test_from_value_child_failure_rolls_the_whole_accumulator_back(self):
        for name, fx in self.families.items():
            with self.subTest(family=name):
                a, b = fx["seed"](), fx["seed"]()
                before = _describe(a.value())
                # candidate at 3x mass so a partial assignment is visible, not a same-value no-op
                candidate = _poison_child(_rescaled_preserving_ints(b.value(), 3.0), fx["child_slot"])
                with self.assertRaises((TypeError, ValueError, AttributeError)):
                    a.from_value(candidate)
                self.assertEqual(_describe(a.value()), before)

    def test_combine_refuses_a_finite_elements_infinite_aggregate_result(self):
        for name, fx in self.families.items():
            with self.subTest(family=name):
                a = fx["seed"]()
                arrays = [v for v in a.value() if isinstance(v, np.ndarray)]
                total = max(float(np.abs(v).sum()) for v in arrays)
                vbig = _rescaled_preserving_ints(a.value(), 9.2e307 / total)
                fresh = fx["make"]()
                try:
                    fresh.from_value(vbig)
                except Exception:  # noqa: BLE001 - any family-specific contract rejection
                    # a family whose own contracts already refuse the huge-but-finite seed
                    # (lookback's sequence-length laws) cannot reach the overflow site at all
                    continue
                before = _describe(fresh.value())
                with self.assertRaises(ValueError):
                    fresh.combine(vbig)
                self.assertEqual(_describe(fresh.value()), before)

    def test_scale_refuses_an_overflowing_product_and_rolls_back(self):
        for name, fx in self.families.items():
            with self.subTest(family=name):
                a = fx["seed"]()
                arrays = [v for v in a.value() if isinstance(v, np.ndarray)]
                max_el = max(float(np.abs(v).max()) for v in arrays)
                vbig = _rescaled_preserving_ints(a.value(), 9.0e307 / max_el)
                fresh = fx["make"]()
                try:
                    fresh.from_value(vbig)
                except Exception:  # noqa: BLE001 - any family-specific contract rejection
                    continue
                before = _describe(fresh.value())
                with self.assertRaises(ValueError):
                    fresh.scale(3.0)
                self.assertEqual(_describe(fresh.value()), before)

    def test_legitimate_combine_and_scale_still_apply(self):
        for name, fx in self.families.items():
            with self.subTest(family=name):
                a, b = fx["seed"](), fx["seed"]()
                arrays_before = [v.copy() for v in a.value() if isinstance(v, np.ndarray)]
                a.combine(b.value())
                a.scale(0.5)
                arrays_after = [v for v in a.value() if isinstance(v, np.ndarray)]
                for pre, post in zip(arrays_before, arrays_after):
                    np.testing.assert_allclose(post, pre, rtol=1e-9)


class KeyedFamilyTransactionTest(unittest.TestCase):
    """Keyed protocol: validated replacements, healed pools, and accepted legitimate pooling."""

    @staticmethod
    def _keyed():
        return {
            "mixture": (lambda: MixtureEstimator(_gaussians(), keys=("wk", "ck")), SCALARS, "wk", "comp_counts"),
            "semi_supervised_mixture": (
                lambda: SemiSupervisedMixtureEstimator(_gaussians(), keys=("wk", "ck")),
                [(x, None) for x in SCALARS],
                "wk",
                "comp_counts",
            ),
            "hierarchical_mixture": (
                lambda: HierarchicalMixtureEstimator(_gaussians(), num_mixtures=2, keys=("wk", "ck")),
                SEQS,
                "wk",
                "comp_counts",
            ),
            "lookback_hmm": (
                lambda: LookbackHiddenMarkovModelEstimator(
                    [SequenceEstimator(GaussianEstimator()) for _ in range(2)], lag=0, keys=("ik", "tk", "sk")
                ),
                SEQS,
                "ik",
                "init_counts",
            ),
            "segmental_hmm": (
                lambda: SegmentalHiddenMarkovEstimator(_gaussians(), keys=("ik", "tk", "sk")),
                SEQS,
                "ik",
                "init_counts",
            ),
            "semi_supervised_hmm": (
                lambda: SemiSupervisedHiddenMarkovEstimator(_gaussians(), keys=("tk", "sk")),
                [(list(s), None) for s in SEQS],
                "tk",
                "trans_counts",
            ),
        }

    def _seeded_pair(self, est_mk, data):
        est = est_mk()
        first, second = est.accumulator_factory().make(), est.accumulator_factory().make()
        rng = RandomState(5)
        for acc in (first, second):
            for x in data:
                acc.initialize(x, 1.0, rng)
        return est, first, second

    def test_key_replace_validates_the_incoming_replacement(self):
        for name, (est_mk, data, key, attr) in self._keyed().items():
            with self.subTest(family=name):
                _, acc, _ = self._seeded_pair(est_mk, data)
                counts_before = np.asarray(getattr(acc, attr)).copy()
                # hierarchical pools (comp, w) tuples; wrap to match each family's pool shape
                bad = np.full(np.shape(counts_before), np.inf)
                payloads = [
                    bad,
                    (bad, np.full(np.shape(acc.w_counts), np.inf)) if name == "hierarchical_mixture" else bad,
                ]
                rejected = False
                for payload in payloads:
                    try:
                        acc.key_replace({key: payload})
                    except (TypeError, ValueError):
                        rejected = True
                self.assertTrue(rejected)
                np.testing.assert_array_equal(np.asarray(getattr(acc, attr)), counts_before)

    def test_key_merge_failure_heals_the_earlier_pool_through_aliases(self):
        for name, (est_mk, data, key, attr) in self._keyed().items():
            with self.subTest(family=name):
                _, first, second = self._seeded_pair(est_mk, data)
                pool = {}
                first.key_merge(pool)
                described = {k: _describe(v) for k, v in pool.items()}
                # a caller-held alias to a pooled array must never observe a partial merge --
                # in-place pools are healed element-wise, out-of-place pools never mutate it
                aliases = {k: (v, v.copy()) for k, v in pool.items() if isinstance(v, np.ndarray)}
                # poison the LAST key's pool so earlier pools merge first and must be healed
                last_key = [k for k in pool][-1] if pool else key
                pool[last_key] = np.zeros((9, 9))
                with self.assertRaises((TypeError, ValueError, AttributeError)):
                    second.key_merge(pool)
                for k, fingerprint in described.items():
                    if k == last_key:
                        continue
                    self.assertEqual(_describe(pool[k]), fingerprint, f"{name}: pool {k!r} not healed")
                for k, (alias, values_before) in aliases.items():
                    if k == last_key:
                        continue
                    np.testing.assert_array_equal(
                        alias, values_before, err_msg=f"{name}: alias to pool {k!r} observed the failed merge"
                    )

    def test_a_keyed_accumulators_own_round_trip_and_m_step_are_accepted(self):
        # the overreach direction: segmental and lookback rejected these outright before the
        # repair (their unconditional equality never learned pooled parts break it)
        for name, (est_mk, data, _key, _attr) in self._keyed().items():
            with self.subTest(family=name):
                est, first, second = self._seeded_pair(est_mk, data)
                pool = {}
                first.key_merge(pool)
                second.key_merge(pool)
                first.key_replace(pool)
                second.key_replace(pool)
                replay = est.accumulator_factory().make()
                replay.from_value(first.value())  # must not raise
                est.estimate(8.0, first.value())  # must not raise


if __name__ == "__main__":
    unittest.main()
