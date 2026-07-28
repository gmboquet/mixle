"""The assembled laptop scientist: real encoders (CLIP/MiniLM/SmolLM2) + certified heads + verified QA.

Marked optional+slow: needs the open-weight models in the local HF cache. Excluded from the fast gate;
part of the full correctness run. Each test is a RECEIPT that a frontier-relevant claim actually holds
on a laptop with no network -- not an assertion.
"""

import unittest

import numpy as np
import pytest

pytestmark = [pytest.mark.optional, pytest.mark.slow, pytest.mark.integration]

transformers = pytest.importorskip("transformers")
datasets = pytest.importorskip("datasets")

CIFAR10_ID = "uoft-cs/cifar10"
CIFAR10_REVISION = "0b2714987fa478483af9968de7c934580d0bb9a2"
BANKING77_ID = "PolyAI/banking77"
BANKING77_REVISION = "90d4e2ee5521c04fc1488f065b8b083658768c57"


def _cifar(n_train=1500, n_test=600):
    from datasets import load_dataset

    tr = load_dataset(CIFAR10_ID, split=f"train[:{n_train}]", revision=CIFAR10_REVISION)
    te = load_dataset(CIFAR10_ID, split=f"test[:{n_test}]", revision=CIFAR10_REVISION)
    return tr, te


def skip_if_assets_unavailable(build):
    """Run ``build``, turning a missing pretrained asset into a skip rather than an error.

    These are ``optional``/``slow``/``integration`` tests: they need real CIFAR-10, Banking77 and
    CLIP weights, none of which ship with the package. The module already skips when transformers or
    datasets are absent, but the *weights* can be missing independently -- offline, behind a proxy,
    or cached in a format the installed transformers refuses (the observed failure is
    "openai/clip-vit-base-patch32 does not appear to have a file named model.safetensors"). That is
    a statement about the machine, not about mixle, and it was surfacing as seven setup ERRORs that
    are indistinguishable at a glance from real breakage. Skip on the asset failure only; anything
    else still propagates.
    """
    try:
        return build()
    except OSError as exc:  # HuggingFace raises OSError for unfetchable/unreadable weights
        raise unittest.SkipTest(f"pretrained asset unavailable on this machine: {exc}") from exc
    except RuntimeError as exc:
        # `datasets` >= 4 refuses script-based loaders ("Dataset scripts are no longer supported"),
        # which is how Banking77 is published. Also an environment statement, but it arrives as a
        # RuntimeError, so match the message rather than the type -- any other RuntimeError is a
        # real failure and must still propagate.
        if "no longer supported" not in str(exc) and "requires huggingface_hub" not in str(exc):
            raise
        raise unittest.SkipTest(f"dataset loader unsupported by the installed stack: {exc}") from exc


class CertifiedPerceptionTest(unittest.TestCase):
    """CLIP image latents + a closed-form mixle head: accurate, CERTIFIED, and calibrated on real CIFAR-10."""

    @classmethod
    def setUpClass(cls):
        from mixle.scientist import Scientist, encode_images

        def build():
            tr, te = _cifar()
            cls.ztr = encode_images([r["img"] for r in tr])
            cls.zte = encode_images([r["img"] for r in te])
            cls.ytr = [r["label"] for r in tr]
            cls.yte = np.array([r["label"] for r in te])
            cls.model = Scientist.study(cls.ztr, cls.ytr, alpha=0.1, seed=0)

        skip_if_assets_unavailable(build)

    def test_accuracy_is_high_and_fit_is_closed_form(self):
        acc = float((self.model.predict(self.zte) == self.yte).mean())
        self.assertGreater(acc, 0.85)  # real CLIP + a closed-form head, no gradient descent
        self.assertEqual(self.model.certificate.guarantee.name, "GLOBAL_UNIQUE")
        self.assertEqual(len(self.model.certificate.gradient_blocks), 0)

    def test_conformal_sets_cover_at_the_stated_level(self):
        sets = self.model.prediction_sets(self.zte)
        coverage = float(np.mean([y in s for y, s in zip(self.yte, sets)]))
        self.assertGreater(coverage, 0.85)  # 90% target, honest sampling slack

    def test_confident_predictions_are_more_accurate_than_overall(self):
        pred = self.model.predict(self.zte)
        confident = ~self.model.abstains(self.zte)
        acc_all = float((pred == self.yte).mean())
        acc_conf = float((pred[confident] == self.yte[confident]).mean())
        self.assertGreater(acc_conf, acc_all)  # abstention buys accuracy where it matters

    def test_trains_in_seconds(self):
        self.assertLess(self.model.train_seconds, 5.0)  # the head fit itself is near-instant

    def test_image_embeddings_are_plain_arrays_with_the_documented_shape(self):
        # pins encode_images()'s actual return contract: transformers wraps CLIP's image features in a
        # BaseModelOutputWithPooling now, not a raw tensor -- this must be unpacked (.pooler_output)
        # inside encode_images() rather than leaking the wrapper (or its .numpy()-less object) here.
        for z, n in ((self.ztr, len(self.ytr)), (self.zte, len(self.yte))):
            self.assertIsInstance(z, np.ndarray)
            self.assertEqual(z.shape, (n, 512))  # ViT-B/32's joint embedding width, per the docstring
            self.assertTrue(np.isfinite(z).all())


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


class EdgeDistillationTest(unittest.TestCase):
    """Foundation-on-laptop (a foundation capability, CPU) compressed to a torch-free edge artifact."""

    @classmethod
    def setUpClass(cls):
        from datasets import load_dataset

        from mixle.scientist import encode_texts, study

        def build():
            ds = load_dataset(BANKING77_ID, split="train", revision=BANKING77_REVISION)
            te = load_dataset(BANKING77_ID, split="test", revision=BANKING77_REVISION)
            tr = [(r["text"], r["label"]) for r in ds if r["label"] < 20][:1200]
            ts = [(r["text"], r["label"]) for r in te if r["label"] < 20][:400]
            cls.xtr = [t for t, _ in tr]
            cls.xte = [t for t, _ in ts]
            cls.yte = [lab for _, lab in ts]
            return study(encode_texts(cls.xtr), [lab for _, lab in tr], alpha=0.1)

        head = skip_if_assets_unavailable(build)
        cache: dict = {}

        def teacher(x):
            if x not in cache:
                cache[x] = int(head.predict(encode_texts([x]))[0])
            return cache[x]

        cls.teacher = staticmethod(teacher)  # keep it a plain callable, not a bound method

    def test_edge_student_is_torch_free_and_tiny(self):
        from mixle.scientist import distill_to_edge

        art = distill_to_edge(self.teacher, self.xtr, self.xte, self.yte, max_bytes=500_000, seed=0)
        self.assertTrue(art.torch_free)  # deploys with NO torch and NO foundation model
        self.assertLess(art.bytes, 500_000)  # kilobyte-scale, fits the device budget

    def test_edge_student_retains_most_of_the_capability(self):
        from mixle.scientist import distill_to_edge

        art = distill_to_edge(self.teacher, self.xtr, self.xte, self.yte, max_bytes=500_000, seed=0)
        self.assertGreater(art.teacher_accuracy, 0.85)  # the foundation capability is strong
        self.assertGreater(art.retention, 0.9)  # the edge student keeps >90% of it
        self.assertGreater(art.agreement, 0.85)  # and it actually mimics the teacher


if __name__ == "__main__":
    unittest.main()
