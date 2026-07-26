"""Tests for the SDC (silent-data-corruption) audit layer (K5): injected-bit-flip catch rate,
audit overhead, false-positive rate on clean runs, the NaN/Inf combine()-boundary watchdog, and
end-to-end quarantine + receipt behavior built on top of K4's retry/blacklist machinery."""

import pickle
import time
import unittest
from dataclasses import dataclass

import numpy as np

from mixle.tests.parallel_test import make_data, make_estimator, make_start_model
from mixle.utils.parallel.sdc_audit import (
    AuditedMPEncodedData,
    _assert_finite_value,
    _compare_statistic_payloads,
    finite_guarded_fold,
    inject_bit_flip,
)


class _CountingAccumulator:
    """Minimal StatisticAccumulator-protocol stand-in with a combine() call counter, used to
    prove the finite watchdog fires at the FIRST offending combine() and does not keep going."""

    def __init__(self):
        self.total = 0.0
        self.combine_calls = 0

    def combine(self, x):
        self.combine_calls += 1
        self.total += x
        return self

    def value(self):
        return self.total

    def from_value(self, x):
        self.total = x
        return self

    def key_merge(self, stats_dict):
        pass

    def key_replace(self, stats_dict):
        pass


class _CountingFactory:
    def __init__(self):
        self.made = []

    def make(self):
        acc = _CountingAccumulator()
        self.made.append(acc)
        return acc


class _FakeEstimator:
    def __init__(self):
        self._factory = _CountingFactory()

    def accumulator_factory(self):
        return self._factory


class _KeyTyingAccumulator(_CountingAccumulator):
    def key_replace(self, stats_dict):
        self.total = {"tied": complex(float("inf"), 0.0)}


class _KeyTyingFactory(_CountingFactory):
    def make(self):
        acc = _KeyTyingAccumulator()
        self.made.append(acc)
        return acc


class _KeyTyingEstimator(_FakeEstimator):
    def __init__(self):
        self._factory = _KeyTyingFactory()


@dataclass
class _NestedStatistic:
    values: set[complex]


class FiniteWatchdogTestCase(unittest.TestCase):
    """K5 acceptance criterion 4: the NaN/Inf watchdog fires at the FIRST combine() boundary
    where a non-finite value appears, not at the end of the fold and not on some later shard."""

    def _payloads(self, values):
        return [pickle.dumps((1.0, v), protocol=pickle.HIGHEST_PROTOCOL) for v in values]

    def test_watchdog_fires_at_first_offending_combine_not_later(self):
        est = _FakeEstimator()
        # payload index 2 (0-based) introduces the NaN; two more clean payloads follow it.
        payloads = self._payloads([1.0, 2.0, float("nan"), 4.0, 5.0])

        with self.assertRaises(ValueError) as ctx:
            finite_guarded_fold(est, payloads)

        self.assertIn("payload index 2", str(ctx.exception))
        # exactly 3 combine() calls happened (indices 0, 1, 2) -- the watchdog stopped the fold
        # immediately after the corrupting combine(), it did not process indices 3 and 4.
        acc = est._factory.made[0]
        self.assertEqual(acc.combine_calls, 3)

    def test_watchdog_fires_on_inf_too(self):
        est = _FakeEstimator()
        payloads = self._payloads([1.0, float("inf"), 3.0])
        with self.assertRaises(ValueError):
            finite_guarded_fold(est, payloads)
        self.assertEqual(est._factory.made[0].combine_calls, 2)

    def test_clean_fold_is_unaffected_and_matches_plain_sum(self):
        est = _FakeEstimator()
        payloads = self._payloads([1.0, 2.0, 3.0, 4.0])
        nobs, value = finite_guarded_fold(est, payloads)
        self.assertEqual(nobs, 4.0)
        self.assertEqual(value, 10.0)

    def test_final_value_is_checked_after_key_tying(self):
        with self.assertRaisesRegex(ValueError, "after key_merge/key_replace"):
            finite_guarded_fold(_KeyTyingEstimator(), self._payloads([1.0]))

    def test_complex_sets_dataclasses_and_object_arrays_are_traversed(self):
        nested = np.array([_NestedStatistic({1.0 + 0.0j, complex(float("nan"), 1.0)})], dtype=object)
        with self.assertRaises(ValueError):
            _assert_finite_value(nested, "nested statistic")


class BitFlipInjectionTestCase(unittest.TestCase):
    def test_inject_bit_flip_changes_exactly_one_bit(self):
        payload = bytes([0b00000000, 0b11110000, 0b10101010])
        corrupted = inject_bit_flip(payload, bit_offset=4)  # 5th bit of byte 0
        self.assertNotEqual(payload, corrupted)
        diff_bits = sum(bin(a ^ b).count("1") for a, b in zip(payload, corrupted))
        self.assertEqual(diff_bits, 1)


class TypedStatisticComparisonTestCase(unittest.TestCase):
    def _payload(self, statistic):
        return pickle.dumps((3.0, statistic), protocol=pickle.HIGHEST_PROTOCOL)

    def test_serialization_order_is_not_treated_as_corruption(self):
        primary = self._payload({"a": np.array([1.0, 2.0]), "b": 4})
        audit = self._payload({"b": 4, "a": np.array([1.0, 2.0])})
        self.assertNotEqual(primary, audit)

        comparison = _compare_statistic_payloads(primary, audit, rtol=1.0e-12, atol=1.0e-12)
        self.assertTrue(comparison.equivalent)
        self.assertEqual(comparison.primary_canonical_sha256, comparison.audit_canonical_sha256)

    def test_numeric_tolerance_is_explicit_and_receiptable(self):
        primary = self._payload(np.array([1.0, 2.0]))
        within = self._payload(np.array([1.0 + 1.0e-13, 2.0]))
        outside = self._payload(np.array([1.0 + 1.0e-6, 2.0]))

        close = _compare_statistic_payloads(primary, within, rtol=1.0e-12, atol=1.0e-12)
        far = _compare_statistic_payloads(primary, outside, rtol=1.0e-12, atol=1.0e-12)
        self.assertTrue(close.equivalent)
        self.assertGreater(close.max_abs_error, 0.0)
        self.assertNotEqual(close.primary_canonical_sha256, close.audit_canonical_sha256)
        self.assertFalse(far.equivalent)
        self.assertIn("exceeds tolerance", far.reason)


class QuarantinePlanningTestCase(unittest.TestCase):
    def test_no_survivor_leaves_all_workers_and_shards_untouched(self):
        encoded = AuditedMPEncodedData.__new__(AuditedMPEncodedData)
        encoded._conns = {0: object(), 1: object()}
        encoded._procs = {}
        encoded._blacklist = set()
        encoded._failures = {0: 0, 1: 0}
        encoded._worker_shards = {0: {0}, 1: {1}}
        encoded.max_retries = 2

        status = encoded._quarantine({0, 1})
        self.assertEqual(status, "deferred:no_safe_survivor")
        self.assertEqual(encoded._blacklist, set())
        self.assertEqual(encoded._worker_shards, {0: {0}, 1: {1}})
        self.assertEqual(set(encoded._conns), {0, 1})
        encoded._conns = {}


class SDCAuditTestCase(unittest.TestCase):
    """Real end-to-end tests against a live multiprocessing worker pool. A single pool is
    reused across many independent audit rounds/trials within each test (spawning worker
    processes dominates wall-clock time; each individual `_audit_shards` round call itself
    takes ~1ms once the pool is up), which is what makes a statistically meaningful number of
    trials affordable."""

    @classmethod
    def setUpClass(cls):
        cls.data = make_data()
        cls.est = make_estimator()
        cls.m_start = make_start_model()
        cls.num_workers = 10

    def test_end_to_end_quarantine_and_receipt_on_a_real_bit_flip(self):
        """A single injected bit-flip on one shard's primary rank, discovered during a real
        pysp_seq_estimate round, produces a receipt and uses a third witness to quarantine only
        the worker that disagrees with the two matching computations."""
        with AuditedMPEncodedData(
            self.data, estimator=self.est, num_workers=6, audit_rate=1.0, rng=np.random.RandomState(42)
        ) as enc:
            target_shard = 2
            hit = {"n": 0}

            def hook(worker_id, shard_id, role, payload):
                if shard_id == target_shard and role == "primary":
                    hit["n"] += 1
                    return inject_bit_flip(payload)
                return payload

            enc.arm_corruption(hook)
            enc.pysp_seq_estimate(self.est, self.m_start)

            self.assertGreaterEqual(hit["n"], 1)
            self.assertEqual(len(enc.last_round_audit_mismatches), 1)
            receipt = enc.last_round_audit_mismatches[0]
            self.assertEqual(receipt.shard_id, target_shard)
            self.assertIsNotNone(receipt.first_diff_byte_offset)
            self.assertNotEqual(receipt.primary_sha256, receipt.audit_sha256)
            self.assertIsNotNone(receipt.witness_worker)
            self.assertEqual(receipt.suspected_workers, (receipt.primary_worker,))
            self.assertIn(receipt.primary_worker, enc._blacklist)
            self.assertNotIn(receipt.audit_worker, enc._blacklist)
            self.assertEqual(receipt.quarantine_status, "committed")
            self.assertEqual(enc.audit_receipts, [receipt])

    def test_catch_rate_matches_audit_rate_per_single_round(self):
        """Injected bit-flips are caught with empirical probability ~= audit_rate per round."""
        num_workers = self.num_workers
        audit_rate = 0.3
        n_audit = round(audit_rate * num_workers)
        expected_p = n_audit / num_workers  # exact per-round probability (sampling w/o replacement)
        n_trials = 400

        rng = np.random.RandomState(7)
        with AuditedMPEncodedData(
            self.data, estimator=self.est, num_workers=num_workers, audit_rate=audit_rate, rng=rng
        ) as enc:
            catches = 0
            for _ in range(n_trials):
                target = int(rng.randint(num_workers))

                def hook(worker_id, shard_id, role, payload, _target=target):
                    if shard_id == _target and role == "primary":
                        return inject_bit_flip(payload)
                    return payload

                enc.arm_corruption(hook)
                enc._audit_shards(self.est, self.m_start, set(range(num_workers)), quarantine_on_mismatch=False)
                caught = any(r.shard_id == target for r in enc.last_round_audit_mismatches)
                catches += int(caught)
                enc.arm_corruption(None)

            empirical_rate = catches / n_trials
            print(
                "\n[K5 receipt] single-round catch rate: audit_rate=%.3f expected_p=%.3f "
                "empirical=%.3f (n=%d trials, %d catches)" % (audit_rate, expected_p, empirical_rate, n_trials, catches)
            )
            # binomial std at p=0.3, n=400 is ~0.023; allow a generous 5-sigma band
            self.assertAlmostEqual(empirical_rate, expected_p, delta=0.12)

    def test_cumulative_catch_rate_approaches_one_as_rounds_accumulate(self):
        """If the SAME corrupted shard persists across multiple audit rounds, the CUMULATIVE
        catch probability over N rounds approaches 100%, tracking 1 - (1 - p) ** N."""
        num_workers = self.num_workers
        audit_rate = 0.3
        n_audit = round(audit_rate * num_workers)
        p = n_audit / num_workers
        max_rounds = 8
        n_sequences = 250

        rng = np.random.RandomState(11)
        with AuditedMPEncodedData(
            self.data, estimator=self.est, num_workers=num_workers, audit_rate=audit_rate, rng=rng
        ) as enc:
            catch_round = []  # round index (1-based) the corruption was first caught, or None
            for _ in range(n_sequences):
                target = int(rng.randint(num_workers))

                def hook(worker_id, shard_id, role, payload, _target=target):
                    if shard_id == _target and role == "primary":
                        return inject_bit_flip(payload)
                    return payload

                enc.arm_corruption(hook)
                caught_at = None
                for r in range(1, max_rounds + 1):
                    enc._audit_shards(self.est, self.m_start, set(range(num_workers)), quarantine_on_mismatch=False)
                    if any(rec.shard_id == target for rec in enc.last_round_audit_mismatches):
                        caught_at = r
                        break
                catch_round.append(caught_at)
                enc.arm_corruption(None)

            cumulative = []
            for k in range(1, max_rounds + 1):
                empirical = sum(1 for c in catch_round if c is not None and c <= k) / n_sequences
                theoretical = 1.0 - (1.0 - p) ** k
                cumulative.append((k, empirical, theoretical))

            print("\n[K5 receipt] cumulative catch rate (p=%.3f per round, %d sequences):" % (p, n_sequences))
            for k, empirical, theoretical in cumulative:
                print("  round %d: empirical=%.3f theoretical=%.3f" % (k, empirical, theoretical))

            # loose per-point tolerance (binomial noise), but the trend must clearly approach 1
            for k, empirical, theoretical in cumulative:
                self.assertAlmostEqual(empirical, theoretical, delta=0.15)
            self.assertGreater(cumulative[-1][1], 0.9)

    def test_overhead_scales_linearly_with_audit_rate(self):
        """The extra redundant shard computations the audit performs scale linearly with
        audit_rate -- not some hidden larger/quadratic cost -- measured both by the exact
        evaluation count and by real wall-clock time."""
        num_workers = self.num_workers
        rates = [0.1, 0.3, 0.6]
        reps = 60

        rng = np.random.RandomState(3)
        with AuditedMPEncodedData(
            self.data, estimator=self.est, num_workers=num_workers, audit_rate=rates[0], rng=rng
        ) as enc:
            results = []
            for r in rates:
                enc.audit_rate = r
                # warm-up rep (excluded from timing) to avoid first-call jitter
                enc._audit_shards(self.est, self.m_start, set(range(num_workers)), quarantine_on_mismatch=False)
                t0 = time.perf_counter()
                total_eval = 0
                for _ in range(reps):
                    enc._audit_shards(self.est, self.m_start, set(range(num_workers)), quarantine_on_mismatch=False)
                    total_eval += enc.last_round_audit_eval_count
                elapsed = time.perf_counter() - t0
                expected_eval_per_round = 2 * round(r * num_workers)
                results.append((r, elapsed, total_eval, expected_eval_per_round))

            print("\n[K5 receipt] audit overhead vs. audit_rate (%d reps/rate, num_workers=%d):" % (reps, num_workers))
            for r, elapsed, total_eval, expected_eval_per_round in results:
                print(
                    "  audit_rate=%.2f: wall_clock=%.4fs (%.5fs/round) eval_count=%d (expected/round=%d)"
                    % (r, elapsed, elapsed / reps, total_eval, expected_eval_per_round)
                )

            # eval_count is EXACTLY linear in audit_rate by construction (2 recomputes per
            # audited shard); confirm that directly.
            for r, _elapsed, total_eval, expected_eval_per_round in results:
                self.assertEqual(total_eval, expected_eval_per_round * reps)

            # and confirm real wall-clock time is not some hidden larger-than-linear cost:
            # time-per-eval should be roughly constant across audit rates (a doubling of the
            # audit rate should roughly double the wall-clock overhead, not 4x or 10x it).
            per_eval_times = [elapsed / total_eval for _r, elapsed, total_eval, _e in results if total_eval > 0]
            self.assertGreater(len(per_eval_times), 1)
            self.assertLess(max(per_eval_times) / min(per_eval_times), 4.0)

    def test_no_mismatches_on_clean_local_workers_within_declared_envelope(self):
        """Clean workers in this recorded local environment agree under the configured envelope."""
        num_workers = self.num_workers
        n_rounds = 300

        rng = np.random.RandomState(23)
        with AuditedMPEncodedData(
            self.data, estimator=self.est, num_workers=num_workers, audit_rate=0.5, rng=rng
        ) as enc:
            model = self.m_start
            total_audited = 0
            total_mismatches = 0
            for _ in range(n_rounds):
                enc._audit_shards(self.est, model, set(range(num_workers)), quarantine_on_mismatch=False)
                total_audited += len(enc.last_round_audited_shards)
                total_mismatches += len(enc.last_round_audit_mismatches)

            print(
                "\n[K5 receipt] clean-run false-positive check: %d rounds, %d total shard audits, "
                "%d mismatches" % (n_rounds, total_audited, total_mismatches)
            )
            self.assertGreater(total_audited, 0)
            self.assertEqual(total_mismatches, 0)
            self.assertEqual(enc.audit_receipts, [])


if __name__ == "__main__":
    unittest.main()
