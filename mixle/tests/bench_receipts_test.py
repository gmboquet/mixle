"""Volume-scale performance receipts (roadmap item A7).

Each test prints a real, measured wall-clock number into the CI log and asserts it against a
GENEROUS floor (>=5x a locally measured baseline): these are regression guards against an
accidental O(n^2)/materialize-the-whole-array bug, not micro-benchmarks -- the floors are pinned
loose on purpose so ordinary machine-to-machine variance never flakes them.

Covers: a GradLeaf (mixle.models.grad_leaf) gradient fit on 1e6x4 float rows; the mixture E-step
engine=torch vs engine=numpy comparison on CUDA (skips cleanly without a CUDA device -- there is no
"assert torch is faster" claim to make on CPU-only hardware); and an A3 patch-streamed mixture fit
over a memmap volume (roadmap item A3: mixle.data.sources.array_source), timing the sample-and-fit
path that never materializes the full volume.
"""

from __future__ import annotations

import io
import os
import shutil
import tempfile
import time
import unittest

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from mixle.data.sources.array_source import MemmapArraySource, PatchSampler  # noqa: E402
from mixle.engines.numpy_engine import NumpyEngine  # noqa: E402
from mixle.inference.estimation import optimize  # noqa: E402
from mixle.models import GradLeaf  # noqa: E402
from mixle.stats import (  # noqa: E402
    GaussianDistribution,
    GaussianEstimator,
    MixtureDistribution,
    MixtureEstimator,
    seq_encode,
)

try:
    from mixle.engines import TorchEngine

    _HAS_TORCH_ENGINE = True
except Exception:  # noqa: BLE001
    _HAS_TORCH_ENGINE = False


class _DiagGauss(torch.nn.Module):
    """Smallest honest density module for the GradLeaf benchmark: a learnable diagonal Gaussian."""

    def __init__(self, dim: int, mu0: float = 0.0) -> None:
        super().__init__()
        self.mu = torch.nn.Parameter(torch.full((dim,), float(mu0)))
        self.log_sigma = torch.nn.Parameter(torch.zeros(dim))

    def _dist(self):
        return torch.distributions.Normal(self.mu, torch.exp(self.log_sigma))

    def log_density(self, x):
        return self._dist().log_prob(x).sum(-1)

    def sample(self, n: int):
        return self._dist().sample((n,))


class GradLeafVolumeScaleBenchmarkTest(unittest.TestCase):
    """Acceptance test A7-1: GradLeaf fit on 1e6x4 float rows finishes within a generous floor."""

    def test_fit_1e6_by_4_within_generous_floor(self):
        rng = np.random.RandomState(0)
        n, dim = 1_000_000, 4
        data = rng.normal(loc=2.0, scale=1.0, size=(n, dim)).astype("float32")

        torch.manual_seed(0)
        module = _DiagGauss(dim=dim)
        start = time.perf_counter()
        fitted = optimize(data, module, max_its=5, out=None)
        elapsed = time.perf_counter() - start
        print("A7 receipt: GradLeaf fit on 1e6x4 float32 rows took %.3f s" % elapsed)

        self.assertIsInstance(fitted, GradLeaf)
        # Pinned floor: measured ~11-13 s under this suite's single-threaded-torch test fixture
        # (max_its=5, default m_steps=60 -> 300 gradient steps over the full 1e6-row batch each
        # call; ~5.4 s standalone with full thread parallelism). 5x the single-threaded measurement
        # is a loose ceiling that only trips if the fit path regresses to materially superlinear
        # cost, not on ordinary machine-to-machine noise.
        self.assertLess(elapsed, 65.0, "GradLeaf fit on 1e6x4 rows took %.3f s, floor is 65 s" % elapsed)


@unittest.skipUnless(_HAS_TORCH_ENGINE, "torch engine unavailable")
class MixtureEStepEngineBenchmarkTest(unittest.TestCase):
    """Acceptance test A7-2: mixture E-step engine=torch vs engine=numpy, GPU only.

    There is no honest "torch beats numpy" claim to assert on CPU-only hardware -- CPU numpy is
    frequently faster than a CPU torch dispatch for small dense E-steps. So this receipt only runs
    (and only asserts torch <= numpy) when a CUDA device is actually present; it skips cleanly
    everywhere else, matching the roadmap card.
    """

    @unittest.skipUnless(torch.cuda.is_available(), "no CUDA device available")
    def test_torch_e_step_is_not_slower_than_numpy_on_cuda(self):
        rng = np.random.RandomState(0)
        n = 400_000
        half = n // 2
        data = [float(v) for v in np.concatenate([rng.normal(-3, 1, half), rng.normal(3, 1, n - half)])]

        estimator = MixtureEstimator([GaussianEstimator()] * 2)
        init = MixtureDistribution([GaussianDistribution(-1.0, 4.0), GaussianDistribution(1.0, 4.0)], [0.5, 0.5])

        start = time.perf_counter()
        numpy_model = optimize(data, estimator, max_its=20, prev_estimate=init, out=None, engine=NumpyEngine())
        numpy_elapsed = time.perf_counter() - start

        start = time.perf_counter()
        torch_model = optimize(
            data, estimator, max_its=20, prev_estimate=init, out=None, engine=TorchEngine(device="cuda")
        )
        torch_elapsed = time.perf_counter() - start

        print(
            "A7 receipt: mixture E-step on 4e5 rows -- numpy %.3f s, torch/cuda %.3f s" % (numpy_elapsed, torch_elapsed)
        )
        self.assertLessEqual(torch_elapsed, numpy_elapsed)

        numpy_mus = sorted(c.mu for c in numpy_model.components)
        torch_mus = sorted(c.mu for c in torch_model.components)
        self.assertAlmostEqual(numpy_mus[0], torch_mus[0], delta=0.1)
        self.assertAlmostEqual(numpy_mus[1], torch_mus[1], delta=0.1)


def _write_two_region_memmap(path: str, shape: tuple[int, int, int], *, block: int = 30) -> None:
    """Write a ``(depth, height, width)`` float32 memmap volume: first half N(-3, 1), second N(3, 1).

    Written in ``block``-sized depth slabs so the writer itself never holds the full volume resident
    -- matching the "never materialize the full volume" contract the receipt below is measuring.
    """
    depth, height, width = shape
    mm = np.memmap(path, dtype="float32", mode="w+", shape=shape)
    rng = np.random.default_rng(12345)
    half = depth // 2
    for start in range(0, depth, block):
        stop = min(start + block, depth)
        mean = -3.0 if start < half else 3.0
        mm[start:stop] = rng.normal(loc=mean, scale=1.0, size=(stop - start, height, width)).astype("float32")
    mm.flush()
    del mm


class PatchStreamedFitBenchmarkTest(unittest.TestCase):
    """Acceptance test A7-3: A3 patch-streamed mixture fit over a memmap volume, timed.

    Companion to ``array_data_sources_test.py::PeakRssPatchStreamingTest`` (which pins peak RSS);
    this receipt pins wall-clock time for the same sample-off-disk-then-fit path so a regression
    that starts reading whole chunks/planes per patch (rather than the requested patch slice) shows
    up as a timing blowup even on a run that doesn't happen to trip the RSS ceiling.
    """

    def test_patch_streamed_fit_within_generous_floor(self):
        shape = (300, 300, 300)  # ~97 MiB float32 volume on disk
        tmpdir = tempfile.mkdtemp(prefix="mixle_bench_patch_")
        try:
            path = os.path.join(tmpdir, "volume.dat")
            _write_two_region_memmap(path, shape)

            start = time.perf_counter()
            source = MemmapArraySource(path, dtype="float32", shape=shape)
            sampler = PatchSampler(source.array, patch_size=(8, 8, 8), num_patches=800, seed=7)
            means = [float(np.mean(patch)) for patch, _coords in sampler.records()]

            estimator = MixtureEstimator([GaussianEstimator()] * 2)
            enc = seq_encode(means, estimator=estimator)
            init = MixtureDistribution([GaussianDistribution(-1.0, 4.0), GaussianDistribution(1.0, 4.0)], [0.5, 0.5])
            model = optimize(None, estimator, enc_data=enc, max_its=30, prev_estimate=init, out=io.StringIO())
            elapsed = time.perf_counter() - start
            print(
                "A7 receipt: A3 patch-streamed mixture fit (800 8^3 patches off a 300^3 volume) took %.3f s" % elapsed
            )

            # Pinned floor: measured 0.02-0.13 s locally across repeated runs (small volume, few
            # small patches -- dominated by process/import noise, not by real work). 5x the observed
            # worst case would be ~0.6 s; we use a much looser 3.0 s floor since this test's whole
            # point is to catch a correctness regression that turns "read 800 8x8x8 patches" into
            # "read the whole 300^3 volume", which would blow past 3 s by orders of magnitude.
            self.assertLess(elapsed, 3.0, "patch-streamed fit took %.3f s, floor is 3.0 s" % elapsed)

            mus = sorted(c.mu for c in model.components)
            self.assertAlmostEqual(mus[0], -3.0, delta=0.75)
            self.assertAlmostEqual(mus[1], 3.0, delta=0.75)
        finally:
            shutil.rmtree(tmpdir, ignore_errors=True)


if __name__ == "__main__":
    unittest.main()
