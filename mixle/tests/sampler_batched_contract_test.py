"""Every ``DistributionSampler`` subclass must accept the ABC's ``batched`` keyword.

``DistributionSampler.sample``'s abstract signature is ``sample(self, size=None, *, batched=True)``
(``mixle/stats/compute/pdist.py``). A concrete sampler that doesn't accept ``batched`` breaks its own
base-class contract: any generic caller that does ``sampler.sample(n, batched=True)`` -- exactly what
the vectorized-fast-path callers in ``markov_chain.py``/``sequence.py``/``hidden_markov.py`` already
do against samplers they construct internally -- gets a ``TypeError`` instead of a value, the moment
that caller is handed a sampler it didn't special-case. This walks every module under ``mixle.stats``
and ``mixle.experimental``, collects every concrete (non-abstract) ``DistributionSampler`` subclass,
and asserts each one's ``sample`` accepts ``batched`` as a keyword. It does not assert anything about
*what* ``batched`` does for a given sampler -- per the ABC docstring, a sampler that is already fully
vectorized (or has only one possible draw order) may legitimately accept and ignore the flag.
"""

import importlib
import inspect
import pkgutil
import unittest
from abc import ABC

import mixle.experimental
import mixle.stats
from mixle.stats.compute.pdist import DistributionSampler


def _iter_submodules(package):
    prefix = package.__name__ + "."
    for _finder, name, _is_pkg in pkgutil.walk_packages(package.__path__, prefix):
        yield name


def _collect_sampler_subclasses():
    modules = list(_iter_submodules(mixle.stats)) + list(_iter_submodules(mixle.experimental))
    classes = {}
    for name in modules:
        try:
            module = importlib.import_module(name)
        except ImportError:
            # an optional dependency this environment doesn't have (mirrors the skip behavior in
            # test_public_api_manifest_test.py's dynamic-package resolution).
            continue
        for attr_name, obj in vars(module).items():
            if (
                inspect.isclass(obj)
                and issubclass(obj, DistributionSampler)
                and obj is not DistributionSampler
                and ABC not in obj.__bases__
                and not inspect.isabstract(obj)
            ):
                classes[f"{obj.__module__}.{obj.__qualname__}"] = obj
    return classes


class SamplerBatchedContractTest(unittest.TestCase):
    def test_every_concrete_sampler_accepts_batched(self):
        classes = _collect_sampler_subclasses()
        self.assertGreater(len(classes), 100, "sampler discovery found suspiciously few classes -- check the walk")

        missing = []
        for qualname, cls in sorted(classes.items()):
            sig = inspect.signature(cls.sample)
            params = sig.parameters
            accepts = "batched" in params or any(p.kind == inspect.Parameter.VAR_KEYWORD for p in params.values())
            if not accepts:
                missing.append(qualname)

        self.assertEqual(
            missing,
            [],
            f"{len(missing)} DistributionSampler subclass(es) don't accept the ABC's `batched` keyword: {missing}",
        )


if __name__ == "__main__":
    unittest.main()
