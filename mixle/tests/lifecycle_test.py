"""mixle.Model / mixle.propose — the lifecycle facade: one object, consistent verbs, no new inference."""

import tempfile
import unittest

import numpy as np
import pytest

try:
    import torch  # noqa: F401

    _HAS_TORCH = True
except ImportError:
    _HAS_TORCH = False


def _records(n, seed=0):
    rng = np.random.RandomState(seed)
    out = []
    for _ in range(n):
        z = rng.randint(0, 2)
        cat = ["a", "b"][z]
        val = float(rng.normal(-3.0 if z == 0 else 3.0, 1.0))
        out.append((cat, val))
    return out


class LifecycleTest(unittest.TestCase):
    def test_top_level_lazy_exports(self):
        import mixle

        self.assertTrue(callable(mixle.propose))
        self.assertTrue(isinstance(mixle.Model, type))
        self.assertIn("Model", dir(mixle))

    def test_propose_fit_evaluate_sample_explain(self):
        import mixle

        data = _records(300)
        m = mixle.propose(data, fit=True)
        self.assertIsNotNone(m.fitted)
        self.assertTrue(m.notes)  # per-field choices / dependencies / warnings surfaced

        ev = m.evaluate(_records(100, seed=1))
        self.assertEqual(ev["n"], 100)
        self.assertTrue(np.isfinite(ev["mean_log_density"]))

        draws = m.sample(5, seed=0)
        self.assertEqual(len(draws), 5)

        text = m.explain()
        self.assertIn("fitted", text)
        self.assertIn("field", text)

        self.assertTrue(np.isfinite(m(data[0])))  # use it: log p(x)

    def test_propose_builds_a_verified_frontier(self):
        import mixle

        m = mixle.propose(_records(300), fit=True)
        self.assertIsNotNone(m.frontier)
        scored = [f for f in m.frontier if "heldout_mean_log_density" in f]
        self.assertGreaterEqual(len(scored), 1)
        scores = [f["heldout_mean_log_density"] for f in scored]
        self.assertEqual(scores, sorted(scores, reverse=True))  # ranked out-of-sample, best first
        self.assertTrue(any(n.startswith("candidate ") for n in m.notes))
        self.assertIs(m.spec, scored[0]["estimator"])  # the winner is the returned model
        self.assertIsNotNone(m.fitted)

    def test_propose_skips_independent_baseline_when_structurally_identical(self):
        # _records(300)'s recommended (dependency-aware) and independent (plain) estimators come back
        # structurally identical here -- confirmed via to_dict(), the real structural comparison.
        # repr() differs regardless (default object repr is identity/memory-address-based), so a
        # repr()-based dedup check would never skip the redundant candidate; to_dict() must.
        import mixle
        from mixle.task import recommend_model
        from mixle.utils.automatic import get_estimator

        rows = _records(300)
        rec_estimator = recommend_model(rows).estimator
        indep_estimator = get_estimator(rows)
        self.assertNotEqual(repr(indep_estimator), repr(rec_estimator))
        self.assertEqual(indep_estimator.to_dict(), rec_estimator.to_dict())

        m = mixle.propose(rows, fit=True)
        names = [f["name"] for f in m.frontier]
        self.assertEqual(names.count("independent"), 0)
        self.assertEqual(names.count("recommended"), 1)

    def test_propose_frontier_reaches_the_copula_dependence_upgrade(self):
        # propose()'s candidates include "structured" -- optimize() called with estimator=None, i.e.
        # its own no-estimator auto-structure-search, the only route to a copula/Bayesian-network
        # dependence upgrade (see docs/automatic-modeling-contract.rst's "Dependence between fields").
        # Pin this directly: the SAME strongly-dependent, heterogeneous data that optimize() alone
        # upgrades to a CopulaDistribution also produces one via propose(), scored on held-out data
        # and able to win the frontier on its own out-of-sample merit -- not just present as an
        # unused option. (History: this candidate was deliberately absent for one release-prep pass,
        # docs narrowed to match -- D-0016 -- then added after an external review scored the gap as
        # the most serious release blocker, quantified on examples/quickstart_example.py's own
        # dependent dataset; see D-0017.)
        import mixle
        from mixle.inference import optimize
        from mixle.stats.combinator.copula import CopulaDistribution

        rng = np.random.RandomState(0)
        z = rng.multivariate_normal([0.0, 0.0], [[1.0, 0.85], [0.85, 1.0]], size=1500)
        try:
            from scipy.stats import gamma as spgamma
            from scipy.stats import norm
        except ImportError:
            self.skipTest("scipy required for this copula-dependence fixture")
        u = norm.cdf(z)
        x0 = spgamma.ppf(u[:, 0], a=2.0, scale=2.0)
        x1 = norm.ppf(u[:, 1], loc=5.0, scale=2.0)
        data = [(float(a), float(b)) for a, b in zip(x0, x1)]

        direct = optimize(data, out=None)
        self.assertIsInstance(direct, CopulaDistribution)  # optimize() alone correctly detects the dependence

        m = mixle.propose(data, fit=True)
        self.assertIn("structured", [f["name"] for f in m.frontier])
        self.assertIsInstance(m.fitted, CopulaDistribution)  # the structured candidate wins on held-out score

    def test_fit_with_explicit_spec_and_enumerate(self):
        import mixle
        from mixle.stats import CategoricalEstimator

        m = mixle.Model(CategoricalEstimator()).fit(["a", "b", "a", "a", "c", "a", "b"])
        top = m.enumerate().top_k(2)
        self.assertEqual(top[0][0], "a")  # most probable value first

    def test_posterior_and_deploy_roundtrip(self):
        import mixle
        from mixle.stats import GaussianDistribution, GaussianEstimator, MixtureDistribution, MixtureEstimator

        reals = np.concatenate(
            [np.random.RandomState(0).normal(-3, 1, 300), np.random.RandomState(1).normal(3, 1, 300)]
        ).tolist()
        init = MixtureDistribution([GaussianDistribution(-1.0, 1.0), GaussianDistribution(1.0, 1.0)], [0.5, 0.5])
        m = mixle.Model(MixtureEstimator([GaussianEstimator(), GaussianEstimator()]))
        m.fit(reals, prev_estimate=init, max_its=25)

        post = np.asarray(m.posterior(-3.0))
        self.assertEqual(post.shape[-1], 2)
        self.assertAlmostEqual(float(np.sum(post)), 1.0, places=5)

        with tempfile.TemporaryDirectory() as d:
            path = m.deploy(d + "/gmm")
            back = mixle.Model.load(path)
            self.assertAlmostEqual(back(-3.0), m(-3.0), places=10)

    def test_pure_model_deploys_as_safe_json_without_pickle(self):
        # A registry-serializable model must not be persisted as an executable pickle: loading a deployed
        # artifact from an untrusted source would otherwise be arbitrary code execution.
        import json
        import os

        import mixle
        from mixle.stats import CategoricalEstimator

        m = mixle.Model(CategoricalEstimator()).fit(["a", "b", "a", "a", "c", "a", "b"])
        with tempfile.TemporaryDirectory() as d:
            path = m.deploy(d + "/cat")
            files = os.listdir(path)
            self.assertNotIn("model.pkl", files)  # no pickle artifact for a pure model
            self.assertIn("model.json", files)
            self.assertEqual(json.loads(open(os.path.join(path, "manifest.json")).read())["format"], "json")
            back = mixle.Model.load(path)
            self.assertAlmostEqual(back.fitted.log_density("a"), m.fitted.log_density("a"), places=10)

    @pytest.mark.torch
    @unittest.skipUnless(_HAS_TORCH, "torch not installed")
    def test_neural_json_artifact_requires_trust_code(self):
        # A "json"-format artifact whose model embeds a NeuralLeaf carries a pickle-backed torch module
        # inside otherwise-safe JSON; loading it must require the same explicit trust as a pickle
        # artifact, or the "JSON does not execute code" claim above would be false for this case.
        import os

        import mixle
        from mixle.models.neural import make_mlp
        from mixle.models.neural_leaf import NeuralGaussian
        from mixle.utils.serialization import SerializationError

        module = make_mlp(input_dim=1, hidden_dims=[8], output_dim=2)
        dist = NeuralGaussian(module)
        m = mixle.Model(dist)
        m.fitted = dist
        with tempfile.TemporaryDirectory() as d:
            path = m.deploy(d + "/neural")
            self.assertIn("model.json", os.listdir(path))  # registry-serializable, so still "json" format
            with self.assertRaises(SerializationError):
                mixle.Model.load(path)
            back = mixle.Model.load(path, trust_code=True)
            self.assertIsInstance(back.fitted, NeuralGaussian)

    def test_internal_serializer_failure_never_becomes_a_pickle_artifact(self):
        # MXR-080-1716: _write_model caught every exception across registry initialization,
        # conversion and JSON encoding and read all of them as "this model needs pickle". An
        # internal TypeError therefore turned an ordinary JSON-serializable model into an
        # executable model.pkl -- a programming failure silently downgrading the artifact's
        # security and portability contract. Only the registry's explicit unsupported-type answer
        # may select pickle.
        import os

        import mixle
        import mixle.utils.serialization as ser
        from mixle.stats import CategoricalEstimator

        m = mixle.Model(CategoricalEstimator()).fit(["a", "b", "a"])
        original = ser.to_serializable

        def broken(_value):
            raise TypeError("internal serializer bug")

        with tempfile.TemporaryDirectory() as d:
            ser.to_serializable = broken
            try:
                with self.assertRaises(TypeError):
                    m.deploy(d + "/cat")
            finally:
                ser.to_serializable = original
            self.assertNotIn("model.pkl", os.listdir(d + "/cat"))

    def test_deploy_declares_the_evidence_it_does_not_export(self):
        # MXR-080-1717: a round trip turned a model carrying a certificate, calibration report,
        # candidate frontier and {"n": 7} fit record into one with None, None, None and {} -- the
        # artifact kept the prediction code and dropped everything needed to judge it, silently.
        import json
        import os

        import mixle
        from mixle.stats import CategoricalEstimator

        m = mixle.Model(CategoricalEstimator()).fit(["a", "b", "a"])
        m.certificate = "certificate-object"
        m.calibration = "calibration-report"
        m.frontier = [{"name": "recommended"}]
        m._fit_info = {"n": 7}

        with tempfile.TemporaryDirectory() as d:
            path = m.deploy(d + "/cat")
            with open(os.path.join(path, "manifest.json")) as f:
                manifest = json.loads(f.read())
            self.assertEqual(manifest["evidence_not_exported"], ["certificate", "calibration", "frontier"])

            back = mixle.Model.load(path)
            self.assertEqual(back._fit_info, {"n": 7})  # the fit record survives the round trip
            self.assertIsNone(back.certificate)  # these do not, and the model says so
            self.assertIsNone(back.calibration)
            self.assertIsNone(back.frontier)
            self.assertTrue(any("artifact export dropped" in note for note in back.notes))
            self.assertIn("not an evidence export", mixle.Model.deploy.__doc__)

    def test_redeployment_leaves_no_stale_generation_and_binds_the_model_digest(self):
        # MXR-080-1719: the model was written straight into the final directory with no digest and
        # no cleanup, so redeploying in a different format left a stale model.pkl beside the fresh
        # model.json, and any later byte change to the model file was undetectable.
        import json
        import os

        import mixle
        import mixle.utils.serialization as ser
        from mixle.stats import CategoricalEstimator
        from mixle.utils.serialization import SerializationError

        m = mixle.Model(CategoricalEstimator()).fit(["a", "b", "a"])
        original = ser.to_serializable

        def unsupported(_value):
            raise SerializationError("not registry-serializable")

        with tempfile.TemporaryDirectory() as d:
            path = d + "/cat"
            ser.to_serializable = unsupported
            try:
                m.deploy(path)  # generation 1: pickle
            finally:
                ser.to_serializable = original
            self.assertIn("model.pkl", os.listdir(path))

            m.deploy(path)  # generation 2: json -- the pickle generation must not survive it
            self.assertEqual(sorted(os.listdir(path)), ["manifest.json", "model.json"])

            with open(os.path.join(path, "manifest.json")) as f:
                manifest = json.loads(f.read())
            self.assertEqual(manifest["model_file"], "model.json")
            self.assertTrue(manifest["model_sha256"].startswith("sha256:"))
            self.assertTrue(mixle.Model.load(path).fitted is not None)

            with open(os.path.join(path, "model.json")) as f:
                payload = json.loads(f.read())
            with open(os.path.join(path, "model.json"), "w") as f:
                f.write(json.dumps(payload) + " ")  # one byte, still valid JSON, still decodable
            with self.assertRaises(SerializationError):
                mixle.Model.load(path)

    def test_legacy_pickle_artifact_still_loads(self):
        # Artifacts written before the JSON format (manifest without a "format" field) still load via
        # pickle -- but only when the caller explicitly trusts the source (pickle execs arbitrary code).
        import json
        import os
        import pickle

        import mixle
        from mixle.stats import CategoricalEstimator
        from mixle.utils.serialization import SerializationError

        fitted = mixle.Model(CategoricalEstimator()).fit(["a", "b", "a"]).fitted
        with tempfile.TemporaryDirectory() as d:
            with open(os.path.join(d, "model.pkl"), "wb") as f:
                pickle.dump(fitted, f)
            with open(os.path.join(d, "manifest.json"), "w") as f:
                f.write(json.dumps({"notes": ["legacy"]}))  # no "format" -> defaults to pickle
            with self.assertRaises(SerializationError):
                mixle.Model.load(d)  # refuses without an explicit trust_code=True
            back = mixle.Model.load(d, trust_code=True)
            self.assertEqual(back.notes, ["legacy"])
            self.assertAlmostEqual(back.fitted.log_density("a"), fitted.log_density("a"), places=10)

    @pytest.mark.torch
    @unittest.skipUnless(_HAS_TORCH, "torch not installed")
    def test_distill_self_teacher_labels_own_clusters(self):
        import mixle
        from mixle.stats import (
            CategoricalEstimator,
            CompositeEstimator,
            GaussianEstimator,
            MixtureEstimator,
        )

        data = _records(240)
        comp = lambda: CompositeEstimator((CategoricalEstimator(), GaussianEstimator()))  # noqa: E731
        m = mixle.Model(MixtureEstimator([comp(), comp()])).fit(data, max_its=25)

        sol = m.distill(inputs=data, epochs=150, seed=0)  # teacher=None -> the model's posterior argmax
        self.assertGreater(sol.holdout_agreement, 0.8)  # clusters are well separated; student matches them
        self.assertIn(sol(data[0]), ("0", "1"))


@pytest.mark.torch
@unittest.skipUnless(_HAS_TORCH, "torch not installed")
class AnalysisVerbsTest(unittest.TestCase):
    def test_explain_prediction_forecast_and_do_delegate(self):
        import mixle
        from mixle.stats import (
            CategoricalEstimator,
            CompositeEstimator,
            GaussianDistribution,
            GaussianEstimator,
            HiddenMarkovModelDistribution,
            MixtureEstimator,
        )

        # explain_prediction on a fitted mixture-of-composites
        data = _records(200)
        comp = lambda: CompositeEstimator((CategoricalEstimator(), GaussianEstimator()))  # noqa: E731
        m = mixle.Model(MixtureEstimator([comp(), comp()])).fit(data, max_its=15)
        ex = m.explain_prediction(data[0])
        self.assertAlmostEqual(ex.total, m(data[0]), places=9)
        self.assertTrue(ex.parts)

        # forecast on a fitted HMM held by a Model
        hmm = HiddenMarkovModelDistribution(
            [GaussianDistribution(-4.0, 1.0), GaussianDistribution(4.0, 1.0)],
            [0.5, 0.5],
            [[0.9, 0.1], [0.1, 0.9]],
        )
        mh = mixle.Model(hmm)
        mh.fitted = hmm  # already-fitted model adopted by the facade
        f = mh.forecast([3.9, 4.1, 4.0], horizon=3, n=2000, seed=0)
        self.assertEqual(f.state_probs.shape, (3, 2))

        # do() rejects models that are not learned Bayesian networks
        with self.assertRaises(TypeError):
            mh.do({0: 1.0})


class AutoRestartTest(unittest.TestCase):
    """restarts='auto': the newcomer's first mixture fit escapes the symmetric saddle by itself."""

    def _data(self):
        rng = np.random.RandomState(0)
        return np.concatenate([rng.normal(-3, 1, 400), rng.normal(3, 1, 400)]).tolist()

    def test_gamma_mixture_saddle_is_detected_and_escaped(self):
        # THE known repro ([[mixture-init-em-saddle]]): positive-support leaves collapse to the
        # symmetric saddle under the default random init. Every saddling seed must escape via the
        # sorted-block hard-partition init (random shards are exchangeable and would NOT escape).
        import mixle
        from mixle.lifecycle import saddle_suspect
        from mixle.stats import GammaDistribution, GammaEstimator, MixtureEstimator

        data = np.concatenate(
            [GammaDistribution(2.0, 0.5).sampler(1).sample(400), GammaDistribution(20.0, 1.0).sampler(2).sample(400)]
        ).tolist()
        est = MixtureEstimator([GammaEstimator(), GammaEstimator()])

        saddled = escaped = 0
        for seed in range(6):
            raw = mixle.Model(est).fit(data, restarts=None, rng=np.random.RandomState(seed), max_its=40)
            if saddle_suspect(raw.fitted, data):
                saddled += 1
                auto = mixle.Model(est).fit(data, restarts="auto", rng=np.random.RandomState(seed), max_its=40)
                if not saddle_suspect(auto.fitted, data):
                    escaped += 1
                    self.assertTrue(any("kept" in n for n in auto.notes))
        self.assertGreater(saddled, 0)  # the repro must actually reproduce
        self.assertEqual(escaped, saddled)  # and every saddle must be escaped

    def test_good_fit_is_untouched(self):
        import mixle
        from mixle.stats import GaussianDistribution, GaussianEstimator, MixtureDistribution, MixtureEstimator

        data = self._data()
        init = MixtureDistribution([GaussianDistribution(-1.0, 1.0), GaussianDistribution(1.0, 1.0)], [0.5, 0.5])
        m = mixle.Model(MixtureEstimator([GaussianEstimator(), GaussianEstimator()]))
        m.fit(data, restarts="auto", prev_estimate=init, max_its=30)
        self.assertFalse(any("saddle" in n for n in m.notes))  # detector stayed quiet on a healthy fit
        mus = sorted(c.mu for c in m.fitted.components)
        self.assertLess(mus[0], -2.5)


class CalibrateRestartLeakTest(unittest.TestCase):
    """fit(calibrate=...) reserves a holdout slice; a restart (explicit restarts=N, or an automatic
    saddle-triggered one) must train on the SAME reduced partition the initial fit used -- never claw
    the calibration rows back into training just because a restart fires."""

    def test_explicit_restart_never_retrains_on_calibration_holdout(self):
        # THE known repro: an 8-row fit with restarts=1, calibrate=.25 used to pass fit()'s ORIGINAL
        # full `data` (not the already-computed training-only `fit_data`) into _refit_symmetry_broken,
        # so the restart silently retrained on the very rows calibration had just reserved as holdout:
        #   initial fit rows: [1, 7, 3, 0, 5, 4]
        #   restart rows:     [0, 1, 2, 3, 4, 5, 6, 7]   <- includes calibration rows 6, 2
        #   calibration rows: [6, 2]
        from unittest import mock

        import mixle
        from mixle.lifecycle import Model
        from mixle.stats import GaussianEstimator

        data = [float(100 + i) for i in range(8)]  # row identity encoded in the value (100 + index)

        def as_idx(values):
            return {int(v) - 100 for v in values}

        from mixle.inference import optimize as real_optimize

        captured_initial: dict = {}

        def optimize_spy(data_arg, spec, **kw):
            captured_initial.setdefault("rows", as_idx(list(data_arg)))
            return real_optimize(data_arg, spec, **kw)

        real_refit = Model._refit_symmetry_broken
        seen: dict = {}

        def refit_spy(self, data_arg, trials, optimize_kw):
            seen["rows"] = as_idx(list(data_arg))
            return real_refit(self, data_arg, trials, optimize_kw)

        with (
            mock.patch("mixle.inference.optimize", side_effect=optimize_spy),
            mock.patch.object(Model, "_refit_symmetry_broken", refit_spy),
        ):
            m = mixle.Model(GaussianEstimator())
            m.fit(data, restarts=1, calibrate=0.25, rng=np.random.RandomState(0))

        self.assertIn("rows", captured_initial)  # the initial fit must actually have run
        self.assertIn("rows", seen)  # the restart must actually have fired

        self.assertEqual(captured_initial["rows"], {1, 7, 3, 0, 5, 4})  # pins the audit's exact repro
        cal_rows = set(range(8)) - captured_initial["rows"]
        self.assertEqual(cal_rows, {6, 2})

        self.assertEqual(seen["rows"], captured_initial["rows"])  # restart trained on the SAME partition
        self.assertTrue(
            seen["rows"].isdisjoint(cal_rows), f"restart retrained on calibration rows {seen['rows'] & cal_rows}"
        )
        self.assertIsNotNone(m.calibration)  # calibration was in fact evaluated (not silently skipped)

    def test_automatic_saddle_restart_never_retrains_on_calibration_holdout(self):
        # Same leak, via restarts="auto"'s own saddle-triggered path instead of an explicit request.
        # Reuses AutoRestartTest's known Gamma-mixture symmetric-saddle repro; calibrate=.25 shrinks the
        # training set, which shifts which seeds saddle, so this searches for one rather than assuming
        # AutoRestartTest's own seeds still apply.
        from unittest import mock

        import mixle
        from mixle.lifecycle import Model, saddle_suspect
        from mixle.stats import GammaDistribution, GammaEstimator, MixtureEstimator

        data = np.concatenate(
            [GammaDistribution(2.0, 0.5).sampler(1).sample(400), GammaDistribution(20.0, 1.0).sampler(2).sample(400)]
        ).tolist()
        est = MixtureEstimator([GammaEstimator(), GammaEstimator()])

        seed = None
        for candidate in range(30):
            raw = mixle.Model(est).fit(
                data, restarts=None, calibrate=0.25, rng=np.random.RandomState(candidate), max_its=40
            )
            if saddle_suspect(raw.fitted, data):
                seed = candidate
                break
        self.assertIsNotNone(seed, "no seed reproduced the saddle within 30 tries -- repro drifted")

        from mixle.inference import optimize as real_optimize

        captured_initial: dict = {}

        def optimize_spy(data_arg, spec, **kw):
            captured_initial.setdefault("rows", frozenset(data_arg))
            return real_optimize(data_arg, spec, **kw)

        real_refit = Model._refit_symmetry_broken
        seen: dict = {}

        def refit_spy(self, data_arg, trials, optimize_kw):
            seen["rows"] = frozenset(data_arg)
            return real_refit(self, data_arg, trials, optimize_kw)

        with (
            mock.patch("mixle.inference.optimize", side_effect=optimize_spy),
            mock.patch.object(Model, "_refit_symmetry_broken", refit_spy),
        ):
            m = mixle.Model(est)
            m.fit(data, restarts="auto", calibrate=0.25, rng=np.random.RandomState(seed), max_its=40)

        self.assertIn("rows", captured_initial)
        self.assertIn("rows", seen)  # the automatic restart must actually have fired

        cal_rows = frozenset(data) - captured_initial["rows"]
        self.assertTrue(cal_rows)  # calibrate=.25 on 800 rows must have actually reserved a holdout
        self.assertEqual(seen["rows"], captured_initial["rows"])  # restart trained on the SAME partition
        self.assertTrue(seen["rows"].isdisjoint(cal_rows), "automatic restart retrained on calibration rows")
        self.assertIsNotNone(m.calibration)


class EvidenceRecordTest(unittest.TestCase):
    """MXR-080-1715: both evidence-producing steps catch every exception and replace the result with
    None, without status, reason or note -- so an injected internal certification failure returned a
    fitted model with certificate=None, calibration=None and empty notes, indistinguishable from a
    caller who never requested (or whose model never supported) those checks."""

    def _fit(self, **kw):
        import mixle
        import mixle.stats as st

        return mixle.Model(st.GaussianEstimator()).fit([float(i) for i in range(8)], **kw)

    def test_a_fit_that_was_never_asked_to_calibrate_says_so(self):
        m = self._fit()
        self.assertEqual(m.evidence["certificate"]["status"], "succeeded")
        self.assertEqual(m.evidence["calibration"]["status"], "not_applicable")
        self.assertIsNone(m.calibration)

    def test_a_successful_calibration_is_recorded_with_its_holdout_size(self):
        m = self._fit(calibrate=0.25)
        self.assertEqual(m.evidence["calibration"]["status"], "succeeded")
        self.assertGreater(m.evidence["calibration"]["n_holdout"], 0)
        self.assertIsNotNone(m.calibration)

    def test_an_internal_certification_failure_is_not_erased(self):
        from unittest import mock

        def boom(*args, **kwargs):
            raise RuntimeError("injected certification failure")

        with mock.patch("mixle.inference.certify", side_effect=boom):
            m = self._fit(calibrate=0.25)

        self.assertIsNone(m.certificate)  # the fit still succeeds and stays usable
        record = m.evidence["certificate"]
        self.assertEqual(record["status"], "failed")  # but NOT indistinguishable from never-attempted
        self.assertEqual(record["error_type"], "RuntimeError")
        self.assertIn("injected certification failure", record["error"])
        self.assertTrue(any("certificate failed" in note for note in m.notes))
        self.assertIn("certificate failed", m.explain())

    def test_an_internal_calibration_failure_is_not_erased(self):
        from unittest import mock

        def boom(*args, **kwargs):
            raise RuntimeError("injected calibration failure")

        with mock.patch("mixle.inference.calibration_report", side_effect=boom):
            m = self._fit(calibrate=0.25)

        self.assertIsNone(m.calibration)
        record = m.evidence["calibration"]
        self.assertEqual(record["status"], "failed")
        self.assertEqual(record["error_type"], "RuntimeError")
        self.assertTrue(any("calibration failed" in note for note in m.notes))


if __name__ == "__main__":
    unittest.main()
