"""Deploy/load round-trip contract: a write this library accepts is a write it can read back.

Campaign finding T3-01. ``Model.deploy`` returned success for 13 distribution families whose
artifacts ``Model.load`` then refused, and ``dump_models`` returned JSON ``load_models`` could not
read. Both surfaces had checked only that the model *encoded*, which is a weaker claim than the one
an artifact makes -- the decoder additionally requires the state to reconstruct through the class's
own constructor, and a fitted von Mises (whose estimator pins ``fit_metadata`` onto it) does not.
The failure surfaced at read time, in whatever process later tried to load the model.

These tests pin the contract rather than the codec: whichever format an artifact ends up in, it must
load, and any downgrade taken to achieve that must be disclosed. They must keep passing once the
underlying codecs are fixed and these families go back to being plain JSON -- so nothing here
asserts that von Mises deploys as pickle, only that what deploy writes, load reads.
"""

import json
import os
import tempfile
import unittest
import warnings

import numpy as np
import pytest

try:
    import torch  # noqa: F401

    _HAS_TORCH = True
except ImportError:
    _HAS_TORCH = False


def _angles(n=400, seed=7):
    """Circular data a von Mises fit is the natural model for."""
    return [float(a) for a in np.random.default_rng(seed).vonmises(1.1, 4.0, n) + np.pi]


def _fit_von_mises():
    import mixle
    from mixle.stats import VonMisesEstimator

    return mixle.Model(VonMisesEstimator()).fit(_angles())


class DeployRoundTripTest(unittest.TestCase):
    def test_deployed_directional_model_can_be_loaded_again(self):
        # T3-01: this deploy reported success and wrote a well-formed model.json with a manifest
        # digest, and Model.load on that exact directory raised SerializationError -- for every
        # directional family, plus Dirichlet, the matrix families and MultivariateStudentT. The
        # artifact was write-only and nothing said so at write time.
        import mixle

        m = _fit_von_mises()
        probe = _angles(n=20, seed=11)
        with tempfile.TemporaryDirectory() as d:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                path = m.deploy(d + "/vm")
            back = mixle.Model.load(path, trust_code=True)
            self.assertEqual(type(back.fitted).__name__, type(m.fitted).__name__)
            for x in probe:
                self.assertAlmostEqual(back.fitted.log_density(x), m.fitted.log_density(x), places=12)

    def test_deploy_never_writes_an_artifact_it_cannot_read_back(self):
        # The contract, stated over the families that actually broke: whatever format deploy
        # chooses, loading the result must return the model. trust_code=True is passed because a
        # disclosed pickle fallback is a legal outcome here; the point is that SOMETHING loads.
        import mixle
        from mixle.stats import DirichletEstimator, MultivariateStudentTEstimator, VonMisesEstimator

        rng = np.random.default_rng(5)
        simplex = [list(map(float, r / r.sum())) for r in rng.gamma(2.0, size=(300, 3))]
        rows = [list(map(float, r)) for r in rng.normal(size=(400, 2))]
        cases = {
            "von_mises": (VonMisesEstimator(), _angles()),
            "dirichlet": (DirichletEstimator(), simplex),
            "mv_student_t": (MultivariateStudentTEstimator(), rows),
        }
        with tempfile.TemporaryDirectory() as d:
            for name, (estimator, data) in cases.items():
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    m = mixle.Model(estimator).fit(data)
                    path = m.deploy(os.path.join(d, name))
                back = mixle.Model.load(path, trust_code=True)
                self.assertEqual(type(back.fitted).__name__, type(m.fitted).__name__, name)

    def test_a_broken_component_no_longer_poisons_the_enclosing_model(self):
        # One unreadable leaf used to make the whole composite unreadable, so a record model with a
        # single angular field deployed successfully and could never be served.
        import mixle
        from mixle.stats import CompositeEstimator, GaussianEstimator, VonMisesEstimator

        rng = np.random.default_rng(4)
        angles = _angles(n=300)
        rows = [(float(v), a) for v, a in zip(rng.normal(size=300), angles, strict=True)]
        with tempfile.TemporaryDirectory() as d:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                m = mixle.Model(CompositeEstimator([GaussianEstimator(), VonMisesEstimator()])).fit(rows)
                path = m.deploy(d + "/composite")
            back = mixle.Model.load(path, trust_code=True)
            self.assertAlmostEqual(back.fitted.log_density(rows[0]), m.fitted.log_density(rows[0]), places=12)


class DeployDisclosureTest(unittest.TestCase):
    def test_a_format_downgrade_is_warned_recorded_and_carried_to_the_reader(self):
        # A pickle artifact costs its readers a trust_code=True they did not ask for, so taking that
        # fallback silently would trade one undisclosed surprise for another. Three disclosures,
        # because the person who runs deploy() and the person who holds the model are rarely the
        # same person: a warning at the keyboard, the reason in the manifest, a note on the model.
        import mixle

        m = _fit_von_mises()
        with tempfile.TemporaryDirectory() as d:
            with warnings.catch_warnings(record=True) as caught:
                warnings.simplefilter("always")
                path = m.deploy(d + "/vm")
            messages = [str(w.message) for w in caught if issubclass(w.category, UserWarning)]
            self.assertTrue(any("trust_code=True" in text for text in messages), messages)
            self.assertTrue(any("VonMisesDistribution" in text for text in messages), messages)

            with open(os.path.join(path, "manifest.json")) as f:
                manifest = json.loads(f.read())
            self.assertIsNotNone(manifest["format_fallback"])
            self.assertIn("VonMisesDistribution", manifest["format_fallback"])

            back = mixle.Model.load(path, trust_code=True)
            self.assertTrue(any("rather than safe JSON" in note for note in back.notes), back.notes)

    def test_a_model_neither_format_can_persist_names_both_failures(self):
        # No correct answer exists for this one, so raising is right -- but the pickle error alone
        # would send the caller hunting for a pickling problem in a model whose JSON path failed
        # first. An unregistered local class fails both routes without any monkeypatching.
        import mixle
        from mixle.utils.serialization import SerializationError

        class Unpersistable:  # local classes have no importable qualname, so pickle refuses them
            def __init__(self) -> None:
                self.value = 1.0

        m = mixle.Model(None)
        m.fitted = Unpersistable()
        with tempfile.TemporaryDirectory() as d:
            with self.assertRaises(SerializationError) as ctx:
                m.deploy(d + "/nope")
        message = str(ctx.exception)
        self.assertIn("Unpersistable", message)
        self.assertIn("cannot be deployed", message)


class DeployNoOverreachTest(unittest.TestCase):
    """The read-back probe must refuse only what genuinely cannot be read back."""

    def test_ordinary_json_families_are_untouched_and_silent(self):
        # The probe runs on every deploy, so a false positive here would demote the whole stable
        # centre of the library to pickle and make every deploy warn.
        import mixle
        from mixle.stats import CategoricalEstimator, GaussianEstimator, PoissonEstimator

        rng = np.random.default_rng(3)
        cases = {
            "categorical": (CategoricalEstimator(), ["a", "b", "a", "a", "c", "b"]),
            "gaussian": (GaussianEstimator(), [float(v) for v in rng.normal(size=200)]),
            "poisson": (PoissonEstimator(), [int(v) for v in rng.poisson(3.0, size=200)]),
        }
        with tempfile.TemporaryDirectory() as d:
            for name, (estimator, data) in cases.items():
                m = mixle.Model(estimator).fit(data)
                with warnings.catch_warnings(record=True) as caught:
                    warnings.simplefilter("always")
                    path = m.deploy(os.path.join(d, name))
                self.assertEqual([str(w.message) for w in caught if issubclass(w.category, UserWarning)], [], name)
                with open(os.path.join(path, "manifest.json")) as f:
                    manifest = json.loads(f.read())
                self.assertEqual(manifest["format"], "json", name)
                self.assertIsNone(manifest["format_fallback"], name)
                self.assertNotIn("model.pkl", os.listdir(path))
                mixle.Model.load(path)  # still loads with no trust asserted at all

    @pytest.mark.torch
    @unittest.skipUnless(_HAS_TORCH, "torch not installed")
    def test_an_embedded_module_does_not_look_like_an_unreadable_artifact(self):
        # The sharpest false positive available: a NeuralLeaf's JSON refuses to decode OUTSIDE a
        # trusted scope, by design. Probing untrusted would read that refusal as "unreadable" and
        # demote every neural model to a pickle. The probe is trusted because it is re-reading a
        # payload this process encoded from its own live object seconds earlier -- the trust gate
        # exists for artifacts of unknown origin, and this one has none.
        import mixle
        from mixle.models.neural import make_mlp
        from mixle.models.neural_leaf import NeuralGaussian
        from mixle.utils.serialization import SerializationError

        dist = NeuralGaussian(make_mlp(input_dim=1, hidden_dims=[8], output_dim=2))
        m = mixle.Model(dist)
        m.fitted = dist
        with tempfile.TemporaryDirectory() as d:
            with warnings.catch_warnings(record=True) as caught:
                warnings.simplefilter("always")
                path = m.deploy(d + "/neural")
            self.assertEqual([str(w.message) for w in caught if issubclass(w.category, UserWarning)], [])
            with open(os.path.join(path, "manifest.json")) as f:
                self.assertEqual(json.loads(f.read())["format"], "json")
            with self.assertRaises(SerializationError):
                mixle.Model.load(path)  # the embedded module still demands trust from its reader
            self.assertIsInstance(mixle.Model.load(path, trust_code=True).fitted, NeuralGaussian)


class DumpModelsRoundTripTest(unittest.TestCase):
    def test_dump_models_refuses_json_that_load_models_cannot_read(self):
        # T3-01's other half: dump_models returned 861 bytes of "safe strict JSON" for a fitted von
        # Mises and load_models could not read a byte of it. There is no fallback format for a
        # function that returns a string, so refusing at write time is the only honest answer.
        import mixle.stats as stats
        from mixle.utils.serialization import SerializationError

        m = _fit_von_mises()
        with self.assertRaises(SerializationError) as ctx:
            stats.dump_models(m.fitted)
        message = str(ctx.exception)
        self.assertIn("VonMisesDistribution", message)
        self.assertIn("verify=False", message)  # the escape hatch is named in the refusal itself

    def test_dump_models_still_round_trips_what_it_always_could(self):
        import mixle
        import mixle.stats as stats
        from mixle.stats import GaussianEstimator

        rng = np.random.default_rng(2)
        m = mixle.Model(GaussianEstimator()).fit([float(v) for v in rng.normal(size=200)])
        restored = stats.load_models(stats.dump_models(m.fitted))
        self.assertAlmostEqual(restored.log_density(0.3), m.fitted.log_density(0.3), places=12)

    def test_verify_false_returns_the_identical_text_for_inspection(self):
        # The written state is complete even when the reader refuses it -- the fitted parameters are
        # all in there. Reading them out is a legitimate use, and the verified path must not be the
        # only way to get the text, or this check would destroy the one recovery route that exists
        # while the codecs are still broken.
        import mixle.stats as stats
        from mixle.utils.serialization import to_json

        m = _fit_von_mises()
        text = stats.dump_models(m.fitted, verify=False)
        self.assertEqual(text, to_json(m.fitted))
        state = dict(json.loads(text)["state"]["items"])
        self.assertAlmostEqual(state["kappa"], m.fitted.kappa, places=12)


if __name__ == "__main__":
    unittest.main()
