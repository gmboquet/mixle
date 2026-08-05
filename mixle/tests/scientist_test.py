"""The assembled laptop scientist: real encoders (CLIP/MiniLM/SmolLM2) + certified heads + verified QA.

Marked optional+slow: needs the open-weight models in the local HF cache. Excluded from the fast gate;
part of the full correctness run. Each test is a RECEIPT that a frontier-relevant claim actually holds
on a laptop with no network -- not an assertion.
"""

import unittest

import pytest

pytestmark = [pytest.mark.optional, pytest.mark.slow, pytest.mark.integration]

transformers = pytest.importorskip("transformers")
datasets = pytest.importorskip("datasets")


class ScientistConstructorTest(unittest.TestCase):
    """Scientist's constructor surface matches what it actually does -- no dead knobs."""

    def test_no_max_entropy_parameter(self):
        # max_entropy used to be accepted and stored but never read anywhere: ask()'s abstention
        # comes from retrieval confidence and the factuality check (see ask()'s own docstring), not
        # from the local model's self-assessed uncertainty -- a real, separate mechanism
        # (mixle.inference.uq, wrapped by substrate.interop.ExternalModel) that Scientist never
        # used. A parameter that silently does nothing is worse than no parameter at all, so it was
        # removed rather than wired up to a feature this class deliberately doesn't use.
        from mixle.scientist import Scientist

        Scientist()  # still constructs fine with no arguments
        with self.assertRaises(TypeError):
            Scientist(max_entropy=0.5)
        self.assertFalse(hasattr(Scientist(), "max_entropy"))


class VerifiedReasoningTest(unittest.TestCase):
    """Grounded QA through the local LLM: answers only what the substrate supports, abstains otherwise."""

    def setUp(self):
        from mixle.scientist import Scientist

        self.sci = Scientist()
        self.sci.learn(
            [
                "Uranium-238 decays to lead-206 with a half-life of 4.468 billion years, the basis of U-Pb dating.",
                "The Cretaceous-Paleogene (K-Pg) boundary is dated to approximately 66.0 million years ago.",
                "Carbon-14 has a half-life of 5730 years, useful for dating materials younger than 50,000 years.",
            ]
        )

    def test_answers_supported_questions_and_grounds_them(self):
        inv = self.sci.ask("what is the half-life of uranium-238")
        self.assertFalse(inv.abstained)
        self.assertGreaterEqual(inv.factuality.grounded_fraction, 0.5)
        self.assertIn("4.468", inv.answer)  # the real number, extracted from evidence

    def test_abstains_on_unsupported_questions(self):
        # raw SmolLM2 confidently hallucinates these; the scientist refuses without provenance
        for q in ["what is the boiling point of tungsten", "who discovered the electron"]:
            self.assertTrue(self.sci.ask(q).abstained, q)

    def test_every_answer_carries_citations(self):
        inv = self.sci.ask("when is the K-Pg boundary dated to")
        self.assertFalse(inv.abstained)
        self.assertTrue(inv.factuality.verdicts)  # per-claim receipt exists

    def test_generate_handles_whatever_apply_chat_template_returns(self):
        # generate()'s LM-wrapping leaf: apply_chat_template() returns a bare id tensor on some
        # transformers versions and a BatchEncoding (.input_ids/.attention_mask) on others, and this
        # package's pin spans both. This calls it directly -- independent of retrieval/factuality --
        # so a regression here fails on the leaf function itself rather than only surfacing
        # indirectly through ask()/wonder() quietly abstaining.
        from mixle.scientist import generate

        text = generate("Reply with just the single word: OK", max_new_tokens=8)
        self.assertIsInstance(text, str)
        self.assertTrue(text.strip())


class ProposeAndWonderTest(unittest.TestCase):
    """The don't-know-but-here's-how half: abstention becomes a plan, and curiosity generates conjectures."""

    def setUp(self):
        from mixle.scientist import Scientist
        from mixle.substrate.act import Action

        self.sci = Scientist()
        self.sci.learn(
            [
                "Uranium-238 decays to lead-206 with a half-life of 4.468 billion years, the basis of U-Pb dating.",
                "Zircon crystals incorporate uranium but reject lead at crystallization, so lead is radiogenic.",
                "The Cretaceous-Paleogene (K-Pg) boundary is dated to approximately 66.0 million years ago.",
            ]
        )
        self.sci.add_action(
            Action(
                "halflife_calc",
                "compute",
                run=lambda q: ["x"],
                cost=1.0,
                description="compute decay ages from isotope half-life measurements",
            )
        )

    def test_abstention_returns_a_ranked_research_proposal(self):
        inv = self.sci.investigate("what is the half-life of potassium-40")
        self.assertTrue(inv.abstained)
        self.assertIsNotNone(inv.proposal)
        self.assertTrue(inv.proposal.options)  # concrete ways to find out
        # the mounted, topically-relevant compute capability is ranked at the top (cheapest relevant)
        self.assertEqual(inv.proposal.best()["kind"], "compute")
        self.assertIn("half-life", inv.proposal.render())

    def test_proposal_names_the_nearest_knowledge_as_the_gap(self):
        prop = self.sci.propose("what is the half-life of potassium-40")
        self.assertTrue(prop.nearest_knowledge)  # it says what it ALMOST knows
        self.assertIn("don't know", prop.render().lower())

    def test_answered_question_needs_no_proposal(self):
        inv = self.sci.investigate("when is the K-Pg boundary dated to")
        self.assertFalse(inv.abstained)
        self.assertIsNone(getattr(inv, "proposal", None))

    def test_wonder_generates_labeled_conjectures_it_does_not_already_know(self):
        conjectures = self.sci.wonder(topic="dating", n=2, seed=1)
        self.assertTrue(conjectures)  # curiosity produced something
        for c in conjectures:
            self.assertEqual(c.status, "conjecture")  # never asserted as fact
            self.assertTrue(self.sci.ask(c.question).abstained)  # genuinely open, not rediscovery
            self.assertIsNotNone(c.proposal)  # each carries a proposed test


if __name__ == "__main__":
    unittest.main()
