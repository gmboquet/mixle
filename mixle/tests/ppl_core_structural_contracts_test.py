"""Release contracts for PPL structural grouping, fixed models, registries, and moments."""

import pickle
import unittest

import numpy as np

from mixle.ppl import AR1, MVN, Field, Flow, Net, Normal, constrain, free
from mixle.ppl.core import register_composite, register_family, register_fitter


class FixedModelContractTest(unittest.TestCase):
    def test_fitting_a_fully_specified_model_preserves_every_fixed_parameter(self):
        model = Normal(7.0, 2.5, name="fixed")
        fitted = model.fit([-3.0, 0.0, 20.0])
        self.assertEqual(fitted.params, {"mean": 7.0, "sd": 2.5})
        self.assertEqual(fitted.parameter_dimension, 0)
        with self.assertRaisesRegex(ValueError, "no inferable parameters"):
            model.fit([1.0], how="map")
        with self.assertRaisesRegex(ValueError, "print_iter"):
            model.fit([1.0], print_iter=1)


class StructuralFitControlContractTest(unittest.TestCase):
    def test_specialized_routes_reject_controls_they_do_not_implement(self):
        regression = Normal(free * Field("x") + free, free)
        with self.assertRaisesRegex(NotImplementedError, "local execution"):
            regression.fit([0.0, 1.0], given={"x": [0.0, 1.0]}, backend="mp")
        with self.assertRaisesRegex(TypeError, "constraints"):
            regression.fit([0.0, 1.0], given={"x": [0.0, 1.0]}, constraints=[])

        indexed = Normal(free(2, name="theta")[Field("group")], free)
        with self.assertRaisesRegex(NotImplementedError, "local execution"):
            indexed.fit([0.0, 1.0], given={"group": [0, 1]}, backend="mp")
        # the shared control validator names the bound explicitly ("... must be an integer >= 1")
        # rather than saying "positive integer"
        with self.assertRaisesRegex(ValueError, r"max_iter must be an integer >= 1"):
            indexed.fit([0.0, 1.0], given={"group": [0, 1]}, max_its=0)

        with self.assertRaisesRegex(NotImplementedError, "how='map'"):
            AR1().fit([0.0, 0.1], how="map")
        with self.assertRaisesRegex(NotImplementedError, "local execution"):
            AR1().fit([0.0, 0.1], backend="mp")
        with self.assertRaisesRegex(NotImplementedError, "how='map'"):
            Flow(2).fit([[0.0, 1.0]], how="map")

        neural = Normal(Net(out=1), free)
        with self.assertRaisesRegex(NotImplementedError, "how='auto'"):
            neural.fit([0.0], given={"x": [[0.0]]}, how="map")
        with self.assertRaisesRegex(NotImplementedError, "local execution"):
            neural.fit([0.0], given={"x": [[0.0]]}, backend="mp")


class IndexedGroupContractTest(unittest.TestCase):
    def test_group_order_and_identity_are_first_seen_stable_and_serialized(self):
        prior = Normal(0.0, 2.0).each(by="group")
        model = Normal(prior, 1.0)
        labels = ["beta", "alpha", "beta", "gamma", "alpha", "gamma"]
        fitted = model.fit([2.0, -1.0, 2.2, 4.0, -0.8, 4.2], given={"group": labels})
        self.assertEqual(fitted.group_labels, ("beta", "alpha", "gamma"))
        self.assertEqual(fitted.group_index, {"beta": 0, "alpha": 1, "gamma": 2})
        restored = pickle.loads(pickle.dumps(fitted))
        self.assertEqual(restored.group_labels, fitted.group_labels)
        self.assertEqual(restored.group_index, fitted.group_index)

    def test_invalid_group_labels_are_rejected_without_sorting_or_dropping(self):
        prior = Normal(0.0, 2.0).each(by="group")
        model = Normal(prior, 1.0)
        with self.assertRaisesRegex(ValueError, "finite"):
            model.fit([1.0, 2.0], given={"group": [0.0, np.nan]})
        with self.assertRaisesRegex(TypeError, "homogeneous"):
            model.fit([1.0, 2.0], given={"group": [1, "1"]})


class RegistryAndIntrospectionContractTest(unittest.TestCase):
    def test_registry_collisions_are_rejected(self):
        with self.assertRaisesRegex(ValueError, "already registered"):
            register_family("Normal", object, object, lambda *args: {}, 0)
        with self.assertRaisesRegex(ValueError, "already registered"):
            register_composite("Mixture", lambda *args: None, lambda *args: None)
        with self.assertRaisesRegex(ValueError, "already registered"):
            register_fitter("map")(lambda *args, **kwargs: None)

    def test_columns_use_declared_event_shape_without_sampling(self):
        vector = MVN(2, mean=np.zeros(2), cov=np.eye(2), name="v")
        self.assertEqual(constrain(vector > 0.0).columns, ["v[0]", "v[1]"])
        self.assertEqual(MVN(2).parameter_dimension, 5)
        self.assertEqual(vector.parameter_dimension, 0)

    def test_analytic_moments_precede_bounded_monte_carlo(self):
        model = Normal(3.0, 2.0)
        self.assertEqual(model.mean(samples=1), 3.0)
        self.assertEqual(model.var(samples=1), 4.0)
        self.assertEqual((model + model).mean(samples=1), 6.0)
        self.assertEqual((model + model).var(samples=1), 16.0)
        with self.assertRaisesRegex(ValueError, "must not exceed"):
            model.mean(samples=1_000_001)


if __name__ == "__main__":
    unittest.main()
