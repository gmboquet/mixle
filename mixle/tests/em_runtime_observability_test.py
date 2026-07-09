"""Tests for runtime EM observability: straggler detection, imbalance receipts, fit_report."""

import time
import unittest
from concurrent.futures import ThreadPoolExecutor

from mixle.telemetry.core import Telemetry
from mixle.utils.parallel.em_observability import (
    RankRecord,
    detect_stragglers,
    fit_report,
    imbalance_receipt,
    plan_rebalance_weights,
    record_rank_round,
    records_from_events,
)


def _simulate_rank(telemetry, *, rank, round, sleep_seconds, n_obs, run_id):
    """Simulate one rank's E-step: a real time.sleep so the slow rank is genuinely slower wall-clock."""
    start = time.perf_counter()
    time.sleep(sleep_seconds)
    elapsed = time.perf_counter() - start
    record_rank_round(
        telemetry,
        rank=rank,
        round=round,
        e_step_seconds=elapsed,
        bytes_processed=n_obs * 64,
        accumulator_bytes=256,
        n_obs=n_obs,
        run_id=run_id,
    )


class StragglerDetectionTest(unittest.TestCase):
    def test_induced_slow_rank_identified_within_one_round(self):
        # 4 ranks running "in parallel" (real threads, real time.sleep) for ONE EM round; rank 2 is
        # deliberately made slow. The straggler test must flag exactly rank 2 from this single round.
        telemetry = Telemetry()
        fast = 0.02
        slow = 0.25
        num_ranks = 4
        slow_rank = 2

        with ThreadPoolExecutor(max_workers=num_ranks) as pool:
            futures = [
                pool.submit(
                    _simulate_rank,
                    telemetry,
                    rank=rank,
                    round=0,
                    sleep_seconds=slow if rank == slow_rank else fast,
                    n_obs=100,
                    run_id="induced-slow",
                )
                for rank in range(num_ranks)
            ]
            for f in futures:
                f.result()

        records = records_from_events(list(telemetry.events(kind="em_round")))
        self.assertEqual(len(records), num_ranks)

        report = detect_stragglers(records, round=0)
        self.assertEqual(report.slow_ranks, (slow_rank,))
        self.assertGreaterEqual(report.ratios[slow_rank], report.threshold_ratio)
        for rank in range(num_ranks):
            if rank != slow_rank:
                self.assertNotIn(rank, report.slow_ranks)

    def test_no_stragglers_when_ranks_are_even(self):
        records = [RankRecord(rank=r, round=0, e_step_seconds=1.0 + 0.01 * r, bytes_processed=1000) for r in range(6)]
        report = detect_stragglers(records, round=0)
        self.assertEqual(report.slow_ranks, ())
        self.assertFalse(report.has_stragglers)

    def test_empty_round_is_handled(self):
        report = detect_stragglers([], round=5)
        self.assertEqual(report.round, 5)
        self.assertEqual(report.slow_ranks, ())


class ImbalanceReceiptTest(unittest.TestCase):
    def test_planted_skew_matches_measured_skew(self):
        # rank 0 gets 10x the raw bytes of each of the other 3 (otherwise-equal) ranks.
        base = 1000
        skew_factor = 10
        num_ranks = 4
        records = [RankRecord(rank=0, round=0, e_step_seconds=1.0, bytes_processed=base * skew_factor, n_obs=100)]
        records += [
            RankRecord(rank=r, round=0, e_step_seconds=1.0, bytes_processed=base, n_obs=10) for r in range(1, num_ranks)
        ]

        receipt = imbalance_receipt(records, round=0)

        mean_bytes = (base * skew_factor + base * (num_ranks - 1)) / num_ranks
        expected_skew_rank0 = (base * skew_factor) / mean_bytes
        self.assertAlmostEqual(receipt.mean_bytes, mean_bytes)
        self.assertAlmostEqual(receipt.skew_by_rank[0], expected_skew_rank0, places=9)
        self.assertAlmostEqual(receipt.max_bytes_ratio, expected_skew_rank0, places=9)

        # rank 0's measured skew, relative to a peer rank, exactly recovers the planted 10x ratio.
        peer_skew = receipt.skew_by_rank[1]
        self.assertAlmostEqual(receipt.skew_by_rank[0] / peer_skew, skew_factor, places=9)

    def test_balanced_bytes_gives_no_skew(self):
        records = [RankRecord(rank=r, round=0, e_step_seconds=1.0, bytes_processed=500) for r in range(5)]
        receipt = imbalance_receipt(records, round=0)
        for rank in range(5):
            self.assertAlmostEqual(receipt.skew_by_rank[rank], 1.0)
        self.assertAlmostEqual(receipt.max_bytes_ratio, 1.0)


class RebalanceWeightsTest(unittest.TestCase):
    def test_slower_rank_gets_less_weight(self):
        records = [
            RankRecord(rank=0, round=0, e_step_seconds=2.0),
            RankRecord(rank=1, round=0, e_step_seconds=1.0),
        ]
        weights = plan_rebalance_weights(records, round=0)
        self.assertLess(weights[0], weights[1])
        self.assertAlmostEqual(sum(weights.values()), 2.0, places=9)  # normalized to sum to rank count


class FitReportTest(unittest.TestCase):
    def test_fit_report_smoke(self):
        records = []
        for round in range(3):
            for rank in range(4):
                slow = rank == 1 and round == 2
                records.append(
                    RankRecord(
                        rank=rank,
                        round=round,
                        e_step_seconds=1.0 if not slow else 5.0,
                        m_step_seconds=0.1,
                        bytes_processed=1000,
                        accumulator_bytes=128,
                        n_obs=100,
                    )
                )

        report = fit_report(records)

        self.assertEqual(report.n_rounds, 3)
        self.assertEqual(report.n_ranks, 4)
        self.assertEqual(report.rounds, (0, 1, 2))
        self.assertEqual(report.total_obs, 3 * 4 * 100)
        self.assertTrue(report.any_stragglers)
        self.assertIn(1, report.stragglers_by_round[2].slow_ranks)
        self.assertEqual(report.stragglers_by_round[0].slow_ranks, ())

        text = report.render()
        self.assertIn("fit_report", text)
        self.assertIn("STRAGGLERS=(1,)", text)

        as_dict = report.as_dict()
        self.assertEqual(as_dict["n_rounds"], 3)
        self.assertTrue(as_dict["any_stragglers"])
        self.assertIn(2, as_dict["by_round"])

    def test_fit_report_empty_records(self):
        report = fit_report([])
        self.assertEqual(report.n_rounds, 0)
        self.assertEqual(report.n_ranks, 0)
        self.assertFalse(report.any_stragglers)
        self.assertIn("fit_report", report.render())


if __name__ == "__main__":
    unittest.main()
