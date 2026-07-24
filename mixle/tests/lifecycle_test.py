"""mixle.Model / mixle.propose — the lifecycle facade: one object, consistent verbs, no new inference."""

import tempfile
import unittest

import numpy as np

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


if __name__ == "__main__":
    unittest.main()
