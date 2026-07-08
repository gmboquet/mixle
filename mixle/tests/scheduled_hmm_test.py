"""Length-/position-conditional ("scheduled") HMM: forward correctness, homogeneous parity, EM, serialization.

Verifies the phase-indexed HMM against brute force (exact forward over all state paths), that a Homogeneous
schedule reduces to the ordinary HMM, that the sampler draws lengths from len_dist, that EM is monotone and a
position/length-aware schedule beats a homogeneous one on held-out data with genuine position/length
structure, and that the schedules index and serialize correctly.
"""

import itertools
import unittest

import numpy as np

from mixle.stats import CategoricalDistribution as DCat
from mixle.stats import IntegerCategoricalDistribution as Cat
from mixle.stats.latent.hidden_markov import HiddenMarkovModelDistribution as HMM
from mixle.stats.latent.scheduled_hidden_markov_model import (
    ByLength,
    ByPosition,
    ByRelativePosition,
    Homogeneous,
    PhaseSchedule,
    ScheduledHMMEstimator,
)
from mixle.stats.latent.scheduled_hidden_markov_model import (
    ScheduledHiddenMarkovModelDistribution as SHMM,
)


def _random_model(rng, k=2, v=3, schedule=None):
    schedule = schedule or ByRelativePosition(2)
    p = schedule.n_phases
    inits = rng.dirichlet(np.ones(k), size=p)
    trans = np.stack([rng.dirichlet(np.ones(k), size=k) for _ in range(p)])
    emis = [[Cat(0, rng.dirichlet(np.ones(v))) for _ in range(k)] for _ in range(p)]
    return SHMM(inits, trans, emis, schedule)


class ScheduledHMMTest(unittest.TestCase):
    def test_forward_matches_brute(self):
        rng = np.random.RandomState(0)
        d = _random_model(rng, k=2, v=3, schedule=ByRelativePosition(2))
        inits, trans, emis = d.inits, d.transitions, d.emissions

        def brute(x):
            length, tot = len(x), -np.inf
            for path in itertools.product(range(d.n_states), repeat=length):
                p0 = d.schedule.phase(0, length)
                lp = np.log(inits[p0][path[0]]) + emis[p0][path[0]].log_density(x[0])
                for t in range(1, length):
                    lp += np.log(trans[d.schedule.phase(t - 1, length)][path[t - 1], path[t]])
                    lp += emis[d.schedule.phase(t, length)][path[t]].log_density(x[t])
                tot = np.logaddexp(tot, lp)
            return tot

        for length in (1, 2, 3, 4, 5):
            x = list(rng.randint(0, 3, size=length))
            self.assertAlmostEqual(d.log_density(x), brute(x), places=9)

    def test_homogeneous_reduces_to_base_hmm(self):
        rng = np.random.RandomState(1)
        k, v = 3, 4
        w = rng.dirichlet(np.ones(k))
        a = rng.dirichlet(np.ones(k), size=k)
        em = [Cat(0, rng.dirichlet(np.ones(v))) for _ in range(k)]
        sh = SHMM(w[None, :], a[None, :, :], [em], Homogeneous())
        hm = HMM(em, w, a)
        for length in (1, 2, 3, 5):
            x = list(rng.randint(0, v, size=length))
            self.assertAlmostEqual(sh.log_density(x), hm.log_density(x), places=9)

    def test_sampler_draws_lengths_from_len_dist(self):
        rng = np.random.RandomState(2)
        d = _random_model(rng, schedule=ByRelativePosition(2))
        d = SHMM(d.inits, d.transitions, d.emissions, d.schedule, len_dist=DCat({4: 0.5, 7: 0.5}))
        lengths = {len(s) for s in d.sampler(0).sample(50)}
        self.assertTrue(lengths <= {4, 7})

    def test_em_monotone_and_relative_beats_homogeneous(self):
        # Truth has position-dependent emissions (early -> low symbols, late -> high), so a relative-position
        # schedule should fit it and a homogeneous one cannot.
        k, v = 2, 4
        truth = SHMM(
            np.array([[0.7, 0.3], [0.5, 0.5]]),
            np.array([[[0.8, 0.2], [0.3, 0.7]], [[0.6, 0.4], [0.4, 0.6]]]),
            [
                [Cat(0, [0.7, 0.1, 0.1, 0.1]), Cat(0, [0.1, 0.7, 0.1, 0.1])],
                [Cat(0, [0.1, 0.1, 0.7, 0.1]), Cat(0, [0.1, 0.1, 0.1, 0.7])],
            ],
            ByRelativePosition(2),
            len_dist=DCat({6: 0.5, 8: 0.5}),
        )
        # n=200/iters=8 (down from 400/12): verified the monotone-EM and rel-beats-homogeneous margin
        # (~200-390 nats, well clear of 0) holds across 20 independent data/init seeds at this size.
        data = truth.sampler(0).sample(200)
        held = truth.sampler(99).sample(200)

        def fit(schedule, iters=8):
            est = ScheduledHMMEstimator(
                k, schedule, Cat(0, [0.25] * v).estimator(), DCat({6: 0.5, 8: 0.5}).estimator(), pseudo_count=0.2
            )
            acc = est.accumulator_factory().make()
            acc.seq_initialize(data, np.ones(len(data)), np.random.RandomState(1))
            model, lls = est.estimate(None, acc.value()), []
            for _ in range(iters):
                a = est.accumulator_factory().make()
                a.seq_update(data, np.ones(len(data)), model)
                model = est.estimate(None, a.value())
                lls.append(float(model.seq_log_density(data).sum()))
            return model, lls

        m_rel, lls = fit(ByRelativePosition(2))
        self.assertTrue(all(lls[i] <= lls[i + 1] + 1e-4 for i in range(len(lls) - 1)))  # monotone
        m_hom, _ = fit(Homogeneous())
        self.assertGreater(m_rel.seq_log_density(held).sum(), m_hom.seq_log_density(held).sum())

    def test_length_conditional_beats_homogeneous(self):
        # Short and long sequences have different emission alphabets -> a ByLength schedule captures it.
        k, v = 2, 4
        short = SHMM(
            np.array([[0.5, 0.5]]),
            np.array([[[0.5, 0.5], [0.5, 0.5]]]),
            [[Cat(0, [0.8, 0.1, 0.05, 0.05]), Cat(0, [0.1, 0.8, 0.05, 0.05])]],
            Homogeneous(),
            len_dist=DCat({3: 1.0}),
        )
        long = SHMM(
            np.array([[0.5, 0.5]]),
            np.array([[[0.5, 0.5], [0.5, 0.5]]]),
            [[Cat(0, [0.05, 0.05, 0.8, 0.1]), Cat(0, [0.05, 0.05, 0.1, 0.8])]],
            Homogeneous(),
            len_dist=DCat({9: 1.0}),
        )
        # 100/class (down from 200) and iters=6 (down from 10): verified the by-length-beats-homogeneous
        # margin (~300-390 nats) holds across 20 independent data/init seeds at this size.
        rng = np.random.RandomState(3)
        data = short.sampler(0).sample(100) + long.sampler(1).sample(100)
        rng.shuffle(data)
        held = short.sampler(5).sample(100) + long.sampler(6).sample(100)

        def fit(schedule, iters=6):
            est = ScheduledHMMEstimator(
                k, schedule, Cat(0, [0.25] * v).estimator(), DCat({3: 0.5, 9: 0.5}).estimator(), pseudo_count=0.2
            )
            acc = est.accumulator_factory().make()
            acc.seq_initialize(data, np.ones(len(data)), np.random.RandomState(2))
            model = est.estimate(None, acc.value())
            for _ in range(iters):
                a = est.accumulator_factory().make()
                a.seq_update(data, np.ones(len(data)), model)
                model = est.estimate(None, a.value())
            return model

        by_len = fit(ByLength([5]))  # split at length 5: short (<=5) vs long (>5)
        hom = fit(Homogeneous())
        self.assertGreater(by_len.seq_log_density(held).sum(), hom.seq_log_density(held).sum())

    def test_phase_indices(self):
        self.assertEqual([Homogeneous().phase(t, 5) for t in range(5)], [0] * 5)
        self.assertEqual([ByPosition(3).phase(t, 9) for t in range(5)], [0, 1, 2, 2, 2])
        self.assertEqual([ByRelativePosition(2).phase(t, 4) for t in range(4)], [0, 0, 1, 1])
        self.assertEqual([ByLength([5]).phase(0, length) for length in (3, 5, 6, 9)], [0, 0, 1, 1])

    def test_schedule_serialization_roundtrip(self):
        for s in (Homogeneous(), ByPosition(4), ByRelativePosition(3), ByLength([5, 10])):
            r = PhaseSchedule.from_dict(s.to_dict())
            self.assertEqual(r.to_dict(), s.to_dict())
            self.assertEqual(r.n_phases, s.n_phases)


if __name__ == "__main__":
    unittest.main()
