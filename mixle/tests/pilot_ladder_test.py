"""F7 pilot ladder (mixle.task.pilot_ladder): rung-by-rung GO/NO-GO orchestration, at tiny simulated scale.

The real roadmap rungs (1B/8k/8-GPU -> 8B/128k/256-GPU -> 8B/10M/1000-GPU -> headline) are unmeasurable
here -- no such hardware exists in this environment. What IS tested here is the orchestration/gating logic
itself: does the ladder actually collect MFU/loss-curve/forgetting-curve artifacts per rung, does the
decision journal record a real (replayable, tamper-evident) entry per rung, does the gate correctly let a
passing rung through, and does it correctly HALT at a rung that fails its own stated criteria.
"""

import unittest

import pytest

pytest.importorskip("torch")
pytest.importorskip("safetensors")

from mixle.task.pilot_ladder import (  # noqa: E402
    PILOT_LADDER_UNAVAILABLE_PIECES,
    Rung,
    run_pilot_ladder,
)


def _tiny_rung(name: str, real_target: str, decision_pieces: tuple[str, ...], **overrides) -> Rung:
    defaults = dict(
        vocab=24,
        d_model=8,
        n_layer=2,
        n_head=2,
        block=6,
        n_workers=2,
        steps=30,
        batch_size=8,
        lr=5e-2,
        seed=0,
        max_final_loss=4.0,
        max_forgetting_gap=4.0,
    )
    defaults.update(overrides)
    return Rung(name=name, real_target=real_target, decision_pieces=decision_pieces, **defaults)


class PilotLadderOrchestrationTest(unittest.TestCase):
    def test_ladder_collects_artifacts_and_progresses_through_passing_rungs(self):
        rungs = [
            _tiny_rung("rung_i_shakeout", "1B params / 8k context / 8 GPUs", ("F1", "F4")),
            _tiny_rung(
                "rung_ii_bakeoff",
                "8B params / 128k context / 256 GPUs",
                ("E7", "H2", "F9"),
                exercise_mup_transfer=True,
                mup_base_width=8,
                exercise_moe_decision=True,
                moe_experts=4,
                d_model=8,
            ),
            _tiny_rung(
                "rung_iii_context",
                "8B params / 10M context / 1000 GPUs",
                ("E8", "F5"),
            ),
        ]
        result = run_pilot_ladder(rungs, peak_flops_per_sec=1.0e9)

        self.assertEqual(len(result.outcomes), 3)
        self.assertIsNone(result.halted_at)
        self.assertEqual(result.passed_rungs(), [r.name for r in rungs])

        for outcome in result.outcomes:
            artifacts = outcome.artifacts
            # MFU artifact
            self.assertIn("mfu", artifacts.health_report)
            self.assertGreaterEqual(artifacts.health_report["mfu"]["n_samples"], 1)
            self.assertIsNotNone(artifacts.mfu_mean)
            # loss-curve artifact
            self.assertEqual(len(artifacts.loss_curve), 30)
            self.assertTrue(all(isinstance(x, float) for x in artifacts.loss_curve))
            # forgetting-curve artifact
            self.assertGreater(len(artifacts.forgetting_curve), 1)
            self.assertIsNotNone(artifacts.forgetting_gap)
            # decision-journal entry
            self.assertTrue(outcome.passed)
            self.assertEqual(outcome.decision_record.rationale, outcome.reason)
            self.assertEqual(outcome.decision_record.action_chosen, "advance_to_next_rung")

        # rung ii actually exercised F9 (real transferred lr) and H2 (real MoE-vs-dense receipt)
        rung_ii = result.outcomes[1].artifacts
        self.assertIn("F9_mup_transfer", rung_ii.exercised_receipts)
        self.assertIn("H2_moe_vs_dense", rung_ii.exercised_receipts)
        self.assertIn("relative_output_diff", rung_ii.exercised_receipts["H2_moe_vs_dense"])
        self.assertIn("decision", rung_ii.exercised_receipts["H2_moe_vs_dense"])
        # E7 (opted out) is honestly noted as skipped, with the real reason
        self.assertIn("E7", rung_ii.skipped_pieces)
        self.assertEqual(rung_ii.skipped_pieces["E7"], PILOT_LADDER_UNAVAILABLE_PIECES["E7"])

        # rung iii honestly notes both E8 and F5 as unreachable-here, with the real reasons
        rung_iii = result.outcomes[2].artifacts
        self.assertEqual(rung_iii.skipped_pieces["E8"], PILOT_LADDER_UNAVAILABLE_PIECES["E8"])
        self.assertEqual(rung_iii.skipped_pieces["F5"], PILOT_LADDER_UNAVAILABLE_PIECES["F5"])

        # journal has exactly one entry per rung and is internally consistent
        self.assertEqual(len(result.journal), 3)
        self.assertTrue(result.journal.verify())

    def test_gate_halts_the_ladder_at_a_rung_that_fails_its_own_criteria(self):
        rungs = [
            _tiny_rung("rung_i_shakeout", "1B params / 8k context / 8 GPUs", ("F1", "F4")),
            # deliberately unachievable target for a 30-step, 8-wide toy model: this rung MUST fail
            _tiny_rung(
                "rung_ii_impossible",
                "8B params / 128k context / 256 GPUs",
                ("H2",),
                max_final_loss=1.0e-6,
                steps=10,
            ),
            _tiny_rung("rung_iii_unreached", "8B params / 10M context / 1000 GPUs", ("E8", "F5")),
        ]
        result = run_pilot_ladder(rungs, peak_flops_per_sec=1.0e9)

        self.assertEqual(result.halted_at, "rung_ii_impossible")
        # only the first two rungs actually ran; the third was never attempted
        self.assertEqual(len(result.outcomes), 2)
        self.assertEqual([o.artifacts.rung for o in result.outcomes], ["rung_i_shakeout", "rung_ii_impossible"])

        first, second = result.outcomes
        self.assertTrue(first.passed)
        self.assertFalse(second.passed)
        self.assertIn("final loss", second.reason)
        self.assertEqual(second.decision_record.action_chosen, "halt_ladder")

        # the journal only has entries for rungs that actually ran
        self.assertEqual(len(result.journal), 2)
        self.assertTrue(result.journal.verify())

    def test_unavailable_piece_raises_notimplementederror_not_a_silent_skip(self):
        rung = _tiny_rung("rung_needs_f5", "headline run", ("F5",), exercise_scaling_law_fit=True)
        with self.assertRaises(NotImplementedError):
            run_pilot_ladder([rung])

    def test_unavailable_fault_tolerance_opt_in_raises(self):
        rung = _tiny_rung("rung_i_shakeout", "1B/8k/8 GPUs", ("F1", "F2"), exercise_fault_tolerance=True)
        with self.assertRaises(NotImplementedError):
            run_pilot_ladder([rung])


class PilotLadderJournalIntegrityTest(unittest.TestCase):
    def test_journal_replays_the_whole_ladder_run(self):
        rungs = [
            _tiny_rung("rung_i", "1B/8k/8 GPUs", ("F1", "F4")),
            _tiny_rung("rung_ii", "8B/128k/256 GPUs", ("H2",), exercise_moe_decision=True),
        ]
        result = run_pilot_ladder(rungs, peak_flops_per_sec=1.0e9)

        # every record's action matches the outcome it was journaled for, in the same order as rungs ran
        self.assertEqual(len(result.journal), len(result.outcomes))
        for record, outcome in zip(result.journal, result.outcomes):
            expected_action = "advance_to_next_rung" if outcome.passed else "halt_ladder"
            self.assertEqual(record.action_chosen, expected_action)
            self.assertEqual(record.step_index, result.outcomes.index(outcome))

        # replay() reconstructs a full belief trajectory, one portfolio per journaled rung
        trajectory = result.journal.replay()
        self.assertEqual(len(trajectory), len(rungs))
        for portfolio in trajectory:
            self.assertEqual(len(portfolio), 2)  # {"go", "no_go"} hypotheses
            self.assertAlmostEqual(float(portfolio.weights.sum()) + portfolio.w_open, 1.0, places=6)

        # tamper-evidence: mutating a stored snapshot breaks verify()
        self.assertTrue(result.journal.verify())
        result.journal.records[0].portfolio_snapshot["w_open"] += 0.25
        self.assertFalse(result.journal.verify())

        # round-trips through JSON exactly, so the whole ladder run is a durable, replayable artifact
        restored = type(result.journal).from_json(type(result.journal)(result.journal.records[1:]).to_json())
        self.assertEqual(len(restored), 1)


if __name__ == "__main__":
    unittest.main()
