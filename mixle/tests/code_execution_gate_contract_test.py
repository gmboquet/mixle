"""A gate that authorizes code execution requires the ``True`` singleton (MXR-080-1881).

``bool("false")`` is ``True``. Three code-execution gates read their flag with truthiness -- ``if
trust_code:`` or ``trust_code or ...`` -- so the string ``"false"``, which is exactly the form a flag
arrives in from a config file, an environment variable, or a CLI argument, OPENED the gate it names
the closing of. ``load_encoded`` had already been repaired to require ``trusted=True`` (MXR-080-1873);
these three now share that contract.

``Model.load`` was not named by the audit, which cited ``Embedder.load`` and ``Registry.get``. It is
the same gate with the same defect, gating both a trusted-deserialization scope and a bare
``pickle.load``, so it is closed here too.
"""

import tempfile
import unittest

import numpy as np

from mixle.stats import GaussianDistribution
from mixle.utils.exact import require_exact_bool, require_explicit_true

# Values that are truthy, falsy-but-not-False, or merely equal to True. None is consent.
NOT_CONSENT = ("false", "no", "True", "0", 0, 1, [], {}, None, 2, np.True_)


class ExplicitTrueTest(unittest.TestCase):
    def test_only_the_singleton_authorizes(self):
        for value in NOT_CONSENT:
            with self.subTest(value=repr(value)):
                with self.assertRaisesRegex(ValueError, "must be exactly True"):
                    require_explicit_true(value, "gate", because="It executes code.")

    def test_true_authorizes(self):
        self.assertIsNone(require_explicit_true(True, "gate", because="It executes code."))

    def test_the_reason_reaches_the_caller(self):
        # A caller who hits this should learn what they would be agreeing to.
        with self.assertRaisesRegex(ValueError, "It unpickles the artifact"):
            require_explicit_true("false", "gate", because="It unpickles the artifact.")


class ExactBoolTest(unittest.TestCase):
    def test_a_numpy_bool_is_a_real_boolean(self):
        # Refusing it would reject values the library's own array paths produce.
        self.assertIs(require_exact_bool(np.True_, "flag"), True)
        self.assertIs(require_exact_bool(np.False_, "flag"), False)

    def test_non_booleans_are_refused_rather_than_coerced(self):
        for value in ("false", "true", 0, 1, None, [], 2.0):
            with self.subTest(value=repr(value)):
                with self.assertRaisesRegex(TypeError, "must be an actual Boolean"):
                    require_exact_bool(value, "flag")

    def test_real_booleans_pass(self):
        self.assertIs(require_exact_bool(True, "flag"), True)
        self.assertIs(require_exact_bool(False, "flag"), False)


class RegistryGateTest(unittest.TestCase):
    def _registry(self):
        from mixle.inference.production.registry import Registry

        registry = Registry(tempfile.mkdtemp())
        registry.register(GaussianDistribution(0.0, 1.0), "m")
        return registry

    def test_a_truthy_string_no_longer_opens_the_trusted_scope(self):
        registry = self._registry()
        for value in ("false", "no", 1, []):
            with self.subTest(value=repr(value)):
                with self.assertRaisesRegex(ValueError, "must be exactly True"):
                    registry.get("m", trust_code=value)

    def test_both_real_booleans_still_work(self):
        registry = self._registry()
        self.assertIsInstance(registry.get("m", trust_code=True)[0], GaussianDistribution)
        self.assertIsInstance(registry.get("m", trust_code=False)[0], GaussianDistribution)

    def test_the_default_is_still_the_closed_gate(self):
        self.assertIsInstance(self._registry().get("m")[0], GaussianDistribution)


class ModelLoadGateTest(unittest.TestCase):
    def test_a_truthy_string_no_longer_authorizes_unpickling(self):
        from mixle.lifecycle import Model

        with tempfile.TemporaryDirectory() as directory:
            model = Model(GaussianDistribution(0.0, 1.0))
            model.fitted = GaussianDistribution(0.0, 1.0)
            model.deploy(directory)
            for value in ("false", 1, []):
                with self.subTest(value=repr(value)):
                    with self.assertRaisesRegex(ValueError, "must be exactly True"):
                        Model.load(directory, trust_code=value)
            # The closed gate is still the default, and a JSON artifact still loads through it.
            self.assertIsNotNone(Model.load(directory))


class EmbedderGateTest(unittest.TestCase):
    def test_a_truthy_string_no_longer_authorizes_unpickling(self):
        from mixle.represent.api import Embedder

        with tempfile.TemporaryDirectory() as directory:
            for value in ("false", 1, []):
                with self.subTest(value=repr(value)):
                    # The flag is checked before the path is, so a missing artifact cannot mask it.
                    with self.assertRaisesRegex(ValueError, "must be exactly True"):
                        Embedder.load(directory, trust_code=value)


if __name__ == "__main__":
    unittest.main()
