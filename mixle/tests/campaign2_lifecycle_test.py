"""Campaign-2 lifecycle findings: every claim deploy/load/posterior makes must be one a user can bank on.

Second repair wave over ``mixle.lifecycle``. The common thread of these findings is documented
guarantees that were false and diagnoses that named the wrong problem:

* T3-10 -- deploy()'s docstring promised "the common pure-model path never needs an unsafe pickle
  load" while pure-statistical base-install families (Bernoulli-set, Thurstone) deploy as pickle.
  The docstring is now true (it names the exception class), and the format is disclosed per
  artifact in the RETURN VALUE as well as the manifest and a warning.
* T3-03 -- any manifest ``format`` other than exactly ``"json"`` was read as a pickle artifact,
  and the refusal advised ``trust_code=True`` (arbitrary-code execution) as the remedy for what
  might be a manifest typo.
* T3-04 -- the manifest recorded nothing about its producer and its own schema tag was never read.
* T3-08 -- manifest.json was 0600 (a mkstemp accident) next to a 0644 model.json.
* T3-09 -- the module docstring advertises ``m.posterior(x)`` for HMMs; HMMs raised bare
  ``AttributeError`` although the forward-backward machinery exists (``latent_posterior``).
* T3-11 -- deploy() onto a non-directory path raised a bare errno message.
* T3-02 -- a same-family model file swapped in alongside a recomputed manifest digest served
  silently under the original fit record; divergence is now disclosed (not refused: a content-hash
  algorithm drift across versions must not brick a legitimate artifact).
* T2-11 -- propose() returns the winner UNFITTED by default while the README says it fits; the
  docstring now says so up front and the unfitted-use error names the remedy.

These tests pin behavior (what loads, what warns, what an error names), not incidental digits.
"""

import json
import os
import shutil
import stat
import tempfile
import unittest
import warnings

import numpy as np

import mixle
from mixle.lifecycle import DeployedArtifact, Model, propose
from mixle.utils.serialization import SerializationError


def _fit_gaussian(mean=0.0, seed=0, n=60):
    from mixle.stats import GaussianDistribution

    rng = np.random.RandomState(seed)
    return Model(GaussianDistribution(float(mean), 1.0)).fit(list(rng.normal(mean, 1, n)))


from mixle.stats import GaussianDistribution


class UnserializableDistribution(GaussianDistribution):
    """Fitted-model stand-in the JSON registry has no codec for (see DeployFormatDisclosureTest)."""

    def __pysp_getstate__(self):
        raise TypeError("this family deliberately has no JSON form (test stand-in)")


class DeployFormatDisclosureTest(unittest.TestCase):
    """T3-10: pickle is never silent, and the docstring makes no false 'never pickle' promise."""

    def test_pickle_fallback_is_disclosed_three_ways_plus_return(self):
        # Originally pinned with BernoulliSetDistribution -- the pure-statistical family whose
        # silent pickle deployment WAS finding T3-10 -- but the same repair wave then gave that
        # family (and Thurstone/Spearman) real JSON codecs, so no shipped base-install family
        # still takes the fallback. The DISCLOSURE machinery is the contract under test, and it
        # must hold for whatever future family lacks a codec: pin it with a distribution the
        # registry deliberately cannot represent (module-level, because the pickle path needs an
        # importable class).
        m = _fit_gaussian()
        m.fitted = UnserializableDistribution(0.0, 1.0)
        with tempfile.TemporaryDirectory() as d:
            with warnings.catch_warnings(record=True) as caught:
                warnings.simplefilter("always")
                result = m.deploy(os.path.join(d, "sets"))
            # the return value discloses the format at the call site
            self.assertIsInstance(result, DeployedArtifact)
            self.assertIsInstance(result, str)  # still a plain path for os.path/Path callers
            self.assertEqual(result.format, "pickle")
            self.assertIn("UnserializableDistribution", result.format_fallback)
            # the warning names the family and the load-time implication
            messages = [str(w.message) for w in caught if issubclass(w.category, UserWarning)]
            self.assertTrue(any("UnserializableDistribution" in t for t in messages), messages)
            self.assertTrue(any("trust_code=True" in t for t in messages), messages)
            # the manifest records it, and the note reaches whoever loads the artifact
            with open(os.path.join(result, "manifest.json")) as f:
                manifest = json.load(f)
            self.assertEqual(manifest["format"], "pickle")
            self.assertIn("UnserializableDistribution", manifest["format_fallback"])
            back = Model.load(result, trust_code=True)
            self.assertTrue(any("rather than safe JSON" in n for n in back.notes), back.notes)

    def test_the_t3_10_families_now_deploy_as_json(self):
        # The T3-10 root-cause repair: the family that used to be the silent-pickle example now
        # writes safe JSON, silently, like any ordinary family.
        from mixle.stats import BernoulliSetEstimator

        data = [{"a", "b"}, {"a"}, {"b", "c"}, {"a", "c"}, {"a", "b", "c"}, {"b"}] * 3
        m = Model(BernoulliSetEstimator()).fit(data)
        with tempfile.TemporaryDirectory() as d:
            with warnings.catch_warnings(record=True) as caught:
                warnings.simplefilter("always")
                result = m.deploy(os.path.join(d, "sets"))
            self.assertEqual(result.format, "json")
            self.assertIsNone(result.format_fallback)
            self.assertEqual([str(w.message) for w in caught if issubclass(w.category, UserWarning)], [])

    def test_json_deploy_returns_annotated_path_and_stays_silent(self):
        # The disclosure machinery must cost the ordinary JSON path nothing: no warning, and the
        # return value says so affirmatively (format == "json", fallback None).
        m = _fit_gaussian()
        with tempfile.TemporaryDirectory() as d:
            with warnings.catch_warnings(record=True) as caught:
                warnings.simplefilter("always")
                result = m.deploy(os.path.join(d, "g"))
            self.assertEqual([str(w.message) for w in caught if issubclass(w.category, UserWarning)], [])
            self.assertEqual(result.format, "json")
            self.assertIsNone(result.format_fallback)

    def test_deploy_docstring_no_longer_promises_pure_models_never_pickle(self):
        # The exact sentence the campaign falsified. The replacement must both drop the false
        # guarantee and say affirmatively that pure-statistical families can take the pickle path.
        doc = Model.deploy.__doc__
        self.assertNotIn("never needs an unsafe pickle load", doc)
        self.assertIn("does not guarantee", doc)


class LoadFormatDispatchTest(unittest.TestCase):
    """T3-03: dispatch on the manifest's actual format value, and name it when unrecognized."""

    def _deployed(self, d):
        path = _fit_gaussian().deploy(os.path.join(d, "g"))
        with open(os.path.join(path, "manifest.json")) as f:
            return path, json.load(f)

    def test_unknown_format_is_named_and_never_diagnosed_as_pickle(self):
        # format "JSON" (a case typo away from valid) used to produce "is a pickle-format artifact
        # ... Pass trust_code=True": a misdiagnosis whose advice enables code execution.
        with tempfile.TemporaryDirectory() as d:
            path, manifest = self._deployed(d)
            manifest["format"] = "JSON"
            with open(os.path.join(path, "manifest.json"), "w") as f:
                json.dump(manifest, f)
            with self.assertRaises(SerializationError) as ctx:
                Model.load(path)
            message = str(ctx.exception)
            self.assertIn("'JSON'", message)
            self.assertNotIn("pickle-format artifact", message)
            self.assertNotIn("trust_code=True", message)

    def test_exact_json_and_missing_format_still_dispatch_as_before(self):
        # The new refusal must not catch what already worked: exact "json" loads trust-free, and a
        # legacy manifest with no format field still defaults to the pickle path (which refuses
        # without trust, as always).
        import pickle

        with tempfile.TemporaryDirectory() as d:
            path, _ = self._deployed(d)
            self.assertIsInstance(Model.load(path).fitted, object)
            legacy = os.path.join(d, "legacy")
            os.mkdir(legacy)
            with open(os.path.join(legacy, "model.pkl"), "wb") as f:
                pickle.dump(_fit_gaussian().fitted, f)
            with open(os.path.join(legacy, "manifest.json"), "w") as f:
                json.dump({"notes": []}, f)
            with self.assertRaises(SerializationError):
                Model.load(legacy)  # pickle still demands trust; the point is it is not misnamed
            self.assertIsNotNone(Model.load(legacy, trust_code=True).fitted)


class ManifestProvenanceTest(unittest.TestCase):
    """T3-04: the manifest names its producer, and its schema tag is actually checked."""

    def test_manifest_records_producer_version_and_schema_tag(self):
        with tempfile.TemporaryDirectory() as d:
            path = _fit_gaussian().deploy(os.path.join(d, "g"))
            with open(os.path.join(path, "manifest.json")) as f:
                manifest = json.load(f)
            self.assertEqual(manifest["mixle_version"], mixle.__version__)
            self.assertEqual(manifest["mixle_artifact"], "lifecycle.Model/v1")

    def test_foreign_schema_tag_is_refused_with_both_tags_named(self):
        # "lifecycle.Model/v99" loaded silently before: the tag existed only to be written. A
        # different-schema artifact must be refused with a message naming what was found and what
        # this reader speaks -- not parsed as v1 and failed somewhere misleading.
        with tempfile.TemporaryDirectory() as d:
            path = _fit_gaussian().deploy(os.path.join(d, "g"))
            with open(os.path.join(path, "manifest.json")) as f:
                manifest = json.load(f)
            manifest["mixle_artifact"] = "lifecycle.Model/v99"
            with open(os.path.join(path, "manifest.json"), "w") as f:
                json.dump(manifest, f)
            with self.assertRaises(SerializationError) as ctx:
                Model.load(path)
            message = str(ctx.exception)
            self.assertIn("lifecycle.Model/v99", message)
            self.assertIn("lifecycle.Model/v1", message)

    def test_manifest_without_tag_still_loads(self):
        # Guard-overreach check: hand-written and pre-tag manifests carry no mixle_artifact and are
        # legitimate; only a tag that AFFIRMS a different schema may be refused.
        with tempfile.TemporaryDirectory() as d:
            path = _fit_gaussian().deploy(os.path.join(d, "g"))
            with open(os.path.join(path, "manifest.json")) as f:
                manifest = json.load(f)
            del manifest["mixle_artifact"]
            with open(os.path.join(path, "manifest.json"), "w") as f:
                json.dump(manifest, f)
            self.assertIsNotNone(Model.load(path).fitted)


class ArtifactFileModeTest(unittest.TestCase):
    """T3-08: one deliberate permission policy for both artifact files (the process umask)."""

    def test_manifest_and_model_file_share_the_umask_mode(self):
        # mkstemp's private 0600 leaked onto the manifest while model.json got the open() default,
        # so a serving user could read the model but not the manifest that names it.
        with tempfile.TemporaryDirectory() as d:
            path = _fit_gaussian().deploy(os.path.join(d, "g"))
            manifest_mode = stat.S_IMODE(os.stat(os.path.join(path, "manifest.json")).st_mode)
            model_mode = stat.S_IMODE(os.stat(os.path.join(path, "model.json")).st_mode)
            self.assertEqual(manifest_mode, model_mode)
            umask = os.umask(0o022)
            os.umask(umask)
            self.assertEqual(manifest_mode, 0o666 & ~umask)


class PosteriorSupportTest(unittest.TestCase):
    """T3-09: the module docstring's `m.posterior(x)` promise holds for HMMs, mixtures, and says
    what it needs when it cannot hold."""

    def test_hmm_posterior_returns_per_timestep_state_marginals(self):
        from mixle.stats import GaussianEstimator, HiddenMarkovEstimator

        rng = np.random.RandomState(0)
        seqs = [[float(v) for v in rng.normal(0, 1, 10)] for _ in range(20)]
        m = Model(HiddenMarkovEstimator([GaussianEstimator(), GaussianEstimator()]))
        m.fit(seqs, restarts=None, max_its=5)
        post = np.asarray(m.posterior(seqs[0]))
        self.assertEqual(post.shape, (10, 2))  # one state distribution per timestep
        self.assertTrue(np.all(post >= 0.0))
        self.assertTrue(np.allclose(post.sum(axis=1), 1.0))

    def test_mixture_posterior_is_unchanged(self):
        from mixle.stats import GaussianEstimator, MixtureEstimator

        rng = np.random.RandomState(0)
        m = Model(MixtureEstimator([GaussianEstimator(), GaussianEstimator()]))
        m.fit([float(v) for v in rng.normal(0, 1, 50)], restarts=None, max_its=5)
        post = np.asarray(m.posterior(0.5))
        self.assertEqual(post.shape, (2,))
        self.assertAlmostEqual(float(post.sum()), 1.0, places=9)

    def test_non_latent_model_error_names_the_supported_shapes(self):
        m = _fit_gaussian()
        with self.assertRaises(AttributeError) as ctx:  # same class as before, no longer bare
            m.posterior(0.5)
        message = str(ctx.exception)
        self.assertIn("GaussianDistribution", message)
        self.assertIn("latent_posterior", message)
        self.assertIn("mixture", message.lower())


class DeployPathErrorTest(unittest.TestCase):
    """T3-11: a bad deploy path names the path and the remedy, keeping the exception class."""

    def test_deploy_onto_plain_file_names_path_and_remedy(self):
        m = _fit_gaussian()
        fd, plain = tempfile.mkstemp()
        os.close(fd)
        try:
            with self.assertRaises(FileExistsError) as ctx:
                m.deploy(plain)
            message = str(ctx.exception)
            self.assertIn(plain, message)
            self.assertIn("DIRECTORY", message)
        finally:
            os.unlink(plain)

    def test_deploy_under_plain_file_parent_names_path_and_remedy(self):
        m = _fit_gaussian()
        fd, plain = tempfile.mkstemp()
        os.close(fd)
        try:
            target = os.path.join(plain, "sub")
            with self.assertRaises(NotADirectoryError) as ctx:
                m.deploy(target)
            message = str(ctx.exception)
            self.assertIn(target, message)
            self.assertIn("DIRECTORY", message)
        finally:
            os.unlink(plain)


class ManifestSwapDisclosureTest(unittest.TestCase):
    """T3-02: a same-family model swap under a rewritten manifest is disclosed, not silent."""

    def test_two_edit_same_family_swap_is_disclosed_on_load(self):
        # Swap in another deployment's same-family model file and recompute the manifest's byte
        # digest (the two edits the campaign measured). The family check passes (same family), the
        # digest check passes (recomputed) -- before this wave the load was completely silent while
        # the manifest kept advertising the original fit record.
        with tempfile.TemporaryDirectory() as d:
            victim = _fit_gaussian(mean=0.0, seed=0).deploy(os.path.join(d, "a"))
            donor = _fit_gaussian(mean=100.0, seed=1).deploy(os.path.join(d, "b"))
            shutil.copy(os.path.join(donor, "model.json"), os.path.join(victim, "model.json"))
            with open(os.path.join(victim, "manifest.json")) as f:
                manifest = json.load(f)
            with open(os.path.join(donor, "manifest.json")) as f:
                manifest["model_sha256"] = json.load(f)["model_sha256"]
            with open(os.path.join(victim, "manifest.json"), "w") as f:
                json.dump(manifest, f)
            with warnings.catch_warnings(record=True) as caught:
                warnings.simplefilter("always")
                loaded = Model.load(victim)
            # Disclosed, not refused: load still returns the model (a content-hash algorithm change
            # between versions must not brick a legitimate artifact), but says what it saw.
            self.assertIsNotNone(loaded.fitted)
            self.assertTrue(any("integrity note" in n for n in loaded.notes), loaded.notes)
            messages = [str(w.message) for w in caught if issubclass(w.category, UserWarning)]
            self.assertTrue(any("integrity note" in t for t in messages), messages)

    def test_untampered_artifact_loads_without_integrity_noise(self):
        # The check must never smear a clean round trip.
        with tempfile.TemporaryDirectory() as d:
            path = _fit_gaussian().deploy(os.path.join(d, "g"))
            with warnings.catch_warnings(record=True) as caught:
                warnings.simplefilter("always")
                loaded = Model.load(path)
            self.assertFalse(any("integrity note" in n for n in loaded.notes), loaded.notes)
            self.assertEqual([str(w.message) for w in caught if issubclass(w.category, UserWarning)], [])


class ProposeUnfittedDisclosureTest(unittest.TestCase):
    """T2-11: the unfitted-by-default contract is stated where a user will actually meet it."""

    def test_propose_default_returns_unfitted_and_error_names_the_remedy(self):
        rng = np.random.RandomState(0)
        m = propose([float(v) for v in rng.normal(0, 1, 30)])
        self.assertIsNone(m.fitted)  # the documented (now prominently) default
        with self.assertRaises(RuntimeError) as ctx:
            m(0.5)
        message = str(ctx.exception)
        self.assertIn("propose()", message)
        self.assertIn("fit=True", message)

    def test_hand_built_model_error_keeps_the_generic_message(self):
        # The propose-specific hint must not leak onto models propose() never built.
        from mixle.stats import GaussianEstimator

        with self.assertRaises(RuntimeError) as ctx:
            Model(GaussianEstimator())(0.5)
        self.assertNotIn("propose()", str(ctx.exception))

    def test_propose_docstring_states_the_unfitted_default_up_front(self):
        self.assertIn("UNFITTED unless ``fit=True``", propose.__doc__)


class ScalarDataPapercutTest(unittest.TestCase):
    """T2-12 (the one grounded item): a single observation gets a named error, not a bare TypeError."""

    def test_single_observation_names_the_expectation_and_the_scoring_verb(self):
        m = _fit_gaussian()
        with self.assertRaises(ValueError) as ctx:
            m.evaluate(0.5)
        message = str(ctx.exception)
        self.assertIn("float", message)
        self.assertIn("model(x)", message)


if __name__ == "__main__":
    unittest.main()
