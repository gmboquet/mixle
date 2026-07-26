"""Tests for the resilient multiprocessing EM backend (K4): retry, rank blacklisting, mid-fit
checkpointing, and the chaos-test determinism receipt."""

import io
import time
import unittest

import numpy as np

from mixle.inference import seq_estimate
from mixle.inference.estimation import optimize
from mixle.stats import seq_encode
from mixle.tests.parallel_test import make_data, make_estimator, make_start_model
from mixle.utils.parallel.resilient_em import (
    ResilientMPEncodedData,
    _canonical_shard_payloads,
    checkpointed_fold,
)


def _model_signature(model):
    """A fully-expanded tuple of the model's floats, for EXACT (bit-identical) comparison."""
    sig = []
    for w, comp in zip(model.w, model.components):
        gauss, cat = comp.dists
        sig.append((float(w), float(gauss.mu), float(gauss.sigma2), tuple(sorted(cat.pmap.items()))))
    return tuple(sig)


class CanonicalShardFoldTestCase(unittest.TestCase):
    def test_migrated_shards_are_interleaved_by_logical_identity(self):
        self.assertEqual(
            _canonical_shard_payloads([(0, b"zero"), (3, b"three"), (1, b"one"), (2, b"two")], {0, 1, 2, 3}),
            [b"zero", b"one", b"two", b"three"],
        )

    def test_duplicate_or_missing_logical_shards_are_rejected(self):
        with self.assertRaisesRegex(RuntimeError, "duplicate"):
            _canonical_shard_payloads([(0, b"a"), (0, b"b")], {0})
        with self.assertRaisesRegex(RuntimeError, "no payload"):
            _canonical_shard_payloads([(0, b"a")], {0, 1})


class WorkerDeadlineAndOperationRecoveryTestCase(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.data = make_data()
        cls.est = make_estimator()
        cls.m_start = make_start_model()

    def test_stalled_live_connection_has_a_bounded_receive(self):
        observed_timeouts: list[float] = []

        class _StalledConnection:
            def poll(self, timeout):
                observed_timeouts.append(timeout)
                return False

        class _LiveProcess:
            def is_alive(self):
                return True

        encoded = ResilientMPEncodedData.__new__(ResilientMPEncodedData)
        encoded.worker_timeout_s = 0.01
        encoded._conns = {0: _StalledConnection()}
        encoded._procs = {0: _LiveProcess()}
        started = time.monotonic()
        with self.assertRaisesRegex(TimeoutError, "did not reply"):
            encoded._recv_raw(0)
        self.assertLess(time.monotonic() - started, 0.1)
        self.assertEqual(len(observed_timeouts), 1)
        encoded._conns = {}
        encoded._procs = {}

    def test_initialize_and_scoring_recover_a_worker_dead_before_the_operation(self):
        seed = 17
        with ResilientMPEncodedData(self.data, estimator=self.est, num_workers=3) as reference:
            expected_initial = reference.pysp_seq_initialize(self.est, np.random.RandomState(seed), 0.5)
            expected_score = reference.pysp_seq_log_density_sum(self.m_start)

        with ResilientMPEncodedData(self.data, estimator=self.est, num_workers=3) as recovered:
            recovered._procs[1].kill()
            recovered._procs[1].join(timeout=5)
            actual_initial = recovered.pysp_seq_initialize(self.est, np.random.RandomState(seed), 0.5)
            self.assertEqual(_model_signature(actual_initial), _model_signature(expected_initial))
            self.assertIn(1, recovered._conns)

            recovered._procs[2].kill()
            recovered._procs[2].join(timeout=5)
            actual_score = recovered.pysp_seq_log_density_sum(self.m_start)
            self.assertEqual(actual_score, expected_score)
            self.assertIn(2, recovered._conns)


class ChaosDeterminismTestCase(unittest.TestCase):
    """K4 acceptance criterion: kill a worker mid-E-step; fit resumes and reaches the SAME
    result (determinism receipt) with only the lost shard recomputed."""

    @classmethod
    def setUpClass(cls):
        cls.data = make_data()
        cls.est = make_estimator()
        cls.m_start = make_start_model()

    def test_single_round_recovers_only_the_lost_shard_and_is_bit_identical(self):
        with ResilientMPEncodedData(self.data, estimator=self.est, num_workers=4) as ref_enc:
            model_ref = ref_enc.pysp_seq_estimate(self.est, self.m_start)

        killed = {"done": False}

        def hook(worker_id, proc):
            if worker_id == 1 and not killed["done"]:
                killed["done"] = True
                proc.kill()
                proc.join(timeout=5)

        with ResilientMPEncodedData(self.data, estimator=self.est, num_workers=4) as chaos_enc:
            chaos_enc.arm_kill(hook)
            model_chaos = chaos_enc.pysp_seq_estimate(self.est, self.m_start)

            # the kill actually happened, and recovery only touched the lost shard
            self.assertTrue(killed["done"])
            self.assertEqual(chaos_enc.last_round_failed_workers, {1})
            self.assertEqual(chaos_enc.last_round_recomputed_shards, {1})
            self.assertEqual(chaos_enc.last_round_reused_shards, {0, 2, 3})
            # the killed rank was retried (respawned), not blacklisted, on a single failure
            self.assertEqual(chaos_enc._blacklist, set())
            self.assertIn(1, chaos_enc._conns)

        # the determinism receipt: exact equality, not just close
        self.assertEqual(_model_signature(model_chaos), _model_signature(model_ref))

    def test_simultaneous_failures_are_recovered_only_on_known_survivors(self):
        with ResilientMPEncodedData(self.data, estimator=self.est, num_workers=4) as ref_enc:
            model_ref = ref_enc.pysp_seq_estimate(self.est, self.m_start)

        killed: set[int] = set()

        def hook(worker_id, proc):
            if worker_id in {1, 2}:
                killed.add(worker_id)
                proc.kill()
                proc.join(timeout=5)

        with ResilientMPEncodedData(self.data, estimator=self.est, num_workers=4) as chaos_enc:
            chaos_enc.arm_kill(hook)
            model_chaos = chaos_enc.pysp_seq_estimate(self.est, self.m_start)
            self.assertEqual(killed, {1, 2})
            self.assertEqual(chaos_enc.last_round_failed_workers, {1, 2})
            self.assertEqual(chaos_enc.last_round_recomputed_shards, {1, 2})
            self.assertEqual(chaos_enc.last_round_reused_shards, {0, 3})

        self.assertEqual(_model_signature(model_chaos), _model_signature(model_ref))

    def test_full_fit_resumes_and_reaches_bit_identical_result(self):
        """Run a REAL multi-iteration fit via optimize(); kill one worker on the very first
        E-step. The fit must complete and land on the exact same model as a failure-free
        reference fit from the same starting point and the same data."""
        with ResilientMPEncodedData(self.data, estimator=self.est, num_workers=4) as ref_enc:
            model_ref = optimize(
                None,
                self.est,
                enc_data=ref_enc,
                max_its=15,
                delta=None,
                reuse_estep_ll=False,
                prev_estimate=self.m_start,
                out=io.StringIO(),
            )

        killed = {"done": False}

        def hook(worker_id, proc):
            if worker_id == 2 and not killed["done"]:
                killed["done"] = True
                proc.kill()
                proc.join(timeout=5)

        with ResilientMPEncodedData(self.data, estimator=self.est, num_workers=4) as chaos_enc:
            chaos_enc.arm_kill(hook)
            model_chaos = optimize(
                None,
                self.est,
                enc_data=chaos_enc,
                max_its=15,
                delta=None,
                reuse_estep_ll=False,
                prev_estimate=self.m_start,
                out=io.StringIO(),
            )

        self.assertTrue(killed["done"], "chaos hook never fired -- test didn't exercise the failure path")
        self.assertEqual(_model_signature(model_chaos), _model_signature(model_ref))

        # sanity: the fit actually moved from the (deliberately bad) starting point
        enc_local = seq_encode(self.data, estimator=self.est)
        m_serial = seq_estimate(enc_local, self.est, self.m_start)
        mus_chaos = sorted(c.dists[0].mu for c in model_chaos.components)
        self.assertAlmostEqual(mus_chaos[0], -3.0, delta=0.5)
        self.assertAlmostEqual(mus_chaos[1], 3.0, delta=0.5)
        del m_serial


class RetryAndBlacklistTestCase(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.data = make_data()
        cls.est = make_estimator()
        cls.m_start = make_start_model()

    def test_repeated_failures_blacklist_the_rank(self):
        with ResilientMPEncodedData(self.data, estimator=self.est, num_workers=4, max_retries=2) as enc:
            model = self.m_start

            def kill_worker_1(worker_id, proc):
                if worker_id == 1:
                    proc.kill()
                    proc.join(timeout=5)

            # round 1: worker 1 dies -> failures[1] == 1 < max_retries(2) -> respawned, not blacklisted
            enc.arm_kill(kill_worker_1)
            model = enc.pysp_seq_estimate(self.est, model)
            self.assertEqual(enc.last_round_failed_workers, {1})
            self.assertEqual(enc._blacklist, set())
            self.assertIn(1, enc._conns)

            # round 2: worker 1 dies again -> failures[1] == 2 >= max_retries -> blacklisted
            enc.arm_kill(kill_worker_1)
            model = enc.pysp_seq_estimate(self.est, model)
            self.assertEqual(enc.last_round_failed_workers, {1})
            self.assertEqual(enc._blacklist, {1})
            self.assertNotIn(1, enc._conns)

            # round 3: no failures injected -- worker 1 must not be assigned any new shard,
            # since it's blacklisted; its shard was migrated onto a survivor instead.
            model = enc.pysp_seq_estimate(self.est, model)
            self.assertEqual(enc.last_round_failed_workers, set())
            self.assertNotIn(1, enc._conns)
            self.assertNotIn(1, enc._worker_shards)
            # all 4 original shards are still covered by the surviving workers
            covered = set()
            for shard_ids in enc._worker_shards.values():
                covered |= shard_ids
            self.assertEqual(covered, {0, 1, 2, 3})

    def test_single_failure_then_success_does_not_blacklist(self):
        with ResilientMPEncodedData(self.data, estimator=self.est, num_workers=4, max_retries=2) as enc:
            model = self.m_start

            killed_once = {"done": False}

            def kill_worker_3_once(worker_id, proc):
                if worker_id == 3 and not killed_once["done"]:
                    killed_once["done"] = True
                    proc.kill()
                    proc.join(timeout=5)

            # round 1: worker 3 dies once -> retried (respawned), not blacklisted
            enc.arm_kill(kill_worker_3_once)
            model = enc.pysp_seq_estimate(self.est, model)
            self.assertTrue(killed_once["done"])
            self.assertEqual(enc.last_round_failed_workers, {3})
            self.assertEqual(enc._blacklist, set())

            # round 2, 3: no failures -- worker 3 keeps participating normally
            for _ in range(2):
                model = enc.pysp_seq_estimate(self.est, model)
                self.assertEqual(enc.last_round_failed_workers, set())

            self.assertEqual(enc._blacklist, set())
            self.assertIn(3, enc._conns)
            self.assertEqual(enc._failures[3], 1)


class CheckpointRoundTripTestCase(unittest.TestCase):
    """K4 acceptance criterion 3: serialize an accumulator's checkpoint mid-fit, discard the
    in-memory state, restore FROM the checkpoint via from_value(), and confirm continuing the
    fit from the restored checkpoint gives the identical final result as never checkpointing."""

    @classmethod
    def setUpClass(cls):
        cls.data = make_data()
        cls.est = make_estimator()
        cls.m_start = make_start_model()

    def _shard_payloads(self):
        import pickle

        estimator = self.est
        model = self.m_start
        n = len(self.data)
        num_shards = 4
        payloads = []
        for i in range(num_shards):
            shard = [self.data[j] for j in range(i, n, num_shards)]
            enc = estimator.accumulator_factory().make().acc_to_encoder().seq_encode(shard)
            local_acc = estimator.accumulator_factory().make()
            local_acc.seq_update(enc, np.ones(len(shard)), model)
            payloads.append(pickle.dumps((float(len(shard)), local_acc.value()), protocol=pickle.HIGHEST_PROTOCOL))
        return payloads

    def test_checkpoint_midway_matches_uninterrupted_fold(self):
        payloads = self._shard_payloads()

        nobs_plain, value_plain = checkpointed_fold(self.est, payloads, checkpoint_after=None)
        nobs_ckpt, value_ckpt = checkpointed_fold(self.est, payloads, checkpoint_after=1)

        self.assertEqual(nobs_plain, nobs_ckpt)
        m_plain = self.est.estimate(nobs_plain, value_plain)
        m_ckpt = self.est.estimate(nobs_ckpt, value_ckpt)
        self.assertEqual(_model_signature(m_plain), _model_signature(m_ckpt))

    def test_from_value_round_trip_is_a_real_reconstruction(self):
        """value()/from_value() actually round-trips through bytes -- not just object identity."""
        import pickle

        payloads = self._shard_payloads()
        accumulator = self.est.accumulator_factory().make()
        count, stats = pickle.loads(payloads[0])
        accumulator.combine(stats)
        checkpoint_bytes = pickle.dumps(accumulator.value(), protocol=pickle.HIGHEST_PROTOCOL)

        del accumulator  # the in-memory accumulator is genuinely gone

        restored_value = pickle.loads(checkpoint_bytes)
        restored = self.est.accumulator_factory().make()
        restored.from_value(restored_value)

        # continuing the restored accumulator with the remaining shards must match folding
        # everything from scratch with no checkpoint at all
        for raw in payloads[1:]:
            c, s = pickle.loads(raw)
            count += c
            restored.combine(s)
        stats_dict = {}
        restored.key_merge(stats_dict)
        restored.key_replace(stats_dict)

        nobs_plain, value_plain = checkpointed_fold(self.est, payloads)
        self.assertEqual(count, nobs_plain)
        m_restored = self.est.estimate(count, restored.value())
        m_plain = self.est.estimate(nobs_plain, value_plain)
        self.assertEqual(_model_signature(m_restored), _model_signature(m_plain))


if __name__ == "__main__":
    unittest.main()
