"""Eval/train discipline and fan-in integrity for the neural leaves (review findings G-1, G-2, G-5).

Scoring a leaf must be a pure read: with the module left in train mode, a Dropout submodule scored
stochastically (log-density diffs up to ~2 between two passes of the SAME leaf on the SAME data) and a
BatchNorm submodule had its running statistics MUTATED by mere scoring -- so E-step responsibilities
were random and scoring corrupted the model. The M-step is the converse contract: a module the user
pre-set to ``eval()`` must still optimize under train-mode semantics, and either way the caller's
training flag must come back as it was left. ``DataBufferAccumulator.combine`` must adopt a worker's
field arity on a fresh root (the mp/mpi/dask/ray fan-in path) instead of silently dropping every field
past the first for conditional leaves. And ``NeuralGaussian`` must follow an active torch engine's
float dtype instead of hardcoding fp32 against the substrate's precision plan.
"""

import unittest

import numpy as np
import pytest

torch = pytest.importorskip("torch")

pytestmark = pytest.mark.torch

from mixle.models.dpo_leaf import DPOModel  # noqa: E402
from mixle.models.grad_leaf import DataBufferAccumulatorFactory, GradEstimator, GradLeafEncoder  # noqa: E402
from mixle.models.neural_leaf import NeuralGaussian  # noqa: E402
from mixle.models.softmax_leaf import NeuralCategorical  # noqa: E402
from mixle.models.streaming_transformer_leaf import StreamingTransformer  # noqa: E402


def _dropout_regressor() -> torch.nn.Module:
    return torch.nn.Sequential(torch.nn.Linear(2, 16), torch.nn.Dropout(p=0.5), torch.nn.Linear(16, 1))


def _batchnorm_regressor() -> torch.nn.Module:
    return torch.nn.Sequential(torch.nn.Linear(2, 8), torch.nn.BatchNorm1d(8), torch.nn.Linear(8, 1))


def _dropout_classifier(classes: int = 3) -> torch.nn.Module:
    return torch.nn.Sequential(torch.nn.Linear(2, 16), torch.nn.Dropout(p=0.5), torch.nn.Linear(16, classes))


def _batchnorm_classifier(classes: int = 3) -> torch.nn.Module:
    return torch.nn.Sequential(torch.nn.Linear(2, 8), torch.nn.BatchNorm1d(8), torch.nn.Linear(8, classes))


def _xy(n: int = 24, dim: int = 2, seed: int = 0) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.RandomState(seed)
    return rng.randn(n, dim), rng.randn(n, 1)


def _xc(n: int = 24, dim: int = 2, classes: int = 3, seed: int = 0) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.RandomState(seed)
    return rng.randn(n, dim), rng.randint(0, classes, size=n)


class NeuralGaussianScoringPurityTest(unittest.TestCase):
    """G-1: scoring a NeuralGaussian is deterministic, side-effect free, and mode-restoring."""

    def test_dropout_module_scores_identically_across_passes(self):
        module = _dropout_regressor()
        module.train()  # the state a freshly built / mid-EM module is naturally in
        leaf = NeuralGaussian(module=module, noise=1.0)
        x, y = _xy()
        first = leaf.seq_log_density((x, y))
        second = leaf.seq_log_density((x, y))
        np.testing.assert_array_equal(first, second)
        self.assertTrue(module.training)  # the caller's mode flag comes back as it was left

    def test_scoring_neither_mutates_batchnorm_stats_nor_uses_batch_statistics(self):
        module = _batchnorm_regressor()
        module.train()
        bn = module[1]
        leaf = NeuralGaussian(module=module, noise=1.0)
        x, y = _xy()
        before_mean = bn.running_mean.clone()
        before_var = bn.running_var.clone()
        scored = leaf.seq_log_density((x, y))
        torch.testing.assert_close(bn.running_mean, before_mean, rtol=0.0, atol=0.0)
        torch.testing.assert_close(bn.running_var, before_var, rtol=0.0, atol=0.0)
        # scoring must use eval statistics: match an explicit eval-mode forward's Gaussian log-density
        module.eval()
        with torch.no_grad():
            mean = module(torch.as_tensor(x, dtype=torch.float32)).numpy()
        expected = -0.5 * ((y - mean) ** 2).sum(axis=1) - 0.5 * np.log(2.0 * np.pi)
        np.testing.assert_allclose(scored, expected, rtol=1e-6, atol=1e-6)


class NeuralCategoricalScoringPurityTest(unittest.TestCase):
    """G-1: same purity contract for the discriminative sibling."""

    def test_dropout_module_scores_identically_across_passes(self):
        module = _dropout_classifier()
        module.train()
        leaf = NeuralCategorical(module=module)
        x, c = _xc()
        first = leaf.seq_log_density((x, c))
        second = leaf.seq_log_density((x, c))
        np.testing.assert_array_equal(first, second)
        self.assertTrue(module.training)

    def test_scoring_neither_mutates_batchnorm_stats_nor_uses_batch_statistics(self):
        module = _batchnorm_classifier()
        module.train()
        bn = module[1]
        leaf = NeuralCategorical(module=module)
        x, c = _xc()
        before_mean = bn.running_mean.clone()
        before_var = bn.running_var.clone()
        scored = leaf.seq_log_density((x, c))
        torch.testing.assert_close(bn.running_mean, before_mean, rtol=0.0, atol=0.0)
        torch.testing.assert_close(bn.running_var, before_var, rtol=0.0, atol=0.0)
        module.eval()
        with torch.no_grad():
            logp = torch.log_softmax(module(torch.as_tensor(x, dtype=torch.float32)), dim=1).numpy()
        np.testing.assert_allclose(scored, logp[np.arange(len(c)), c], rtol=1e-6, atol=1e-6)


class SiblingLeafScoringPurityTest(unittest.TestCase):
    """G-1: the streaming-transformer and DPO scoring paths share the same eval discipline."""

    def test_streaming_transformer_dropout_scores_identically(self):
        module = _dropout_classifier(classes=5)
        module.train()
        leaf = StreamingTransformer(module=module)
        x, c = _xc(classes=5)
        first = leaf.seq_log_density((x, c))
        second = leaf.seq_log_density((x, c))
        np.testing.assert_array_equal(first, second)
        self.assertTrue(module.training)
        pred_a = leaf.predict(x)
        pred_b = leaf.predict(x)
        np.testing.assert_array_equal(pred_a, pred_b)

    def test_dpo_dropout_policy_scores_identically(self):
        policy = _dropout_classifier(classes=4)
        policy.train()
        ref = _dropout_classifier(classes=4)
        ref.train()
        leaf = DPOModel(policy=policy, ref=ref)
        rng = np.random.RandomState(0)
        x = rng.randn(16, 2)
        ch = rng.randint(0, 4, size=16)
        rj = (ch + 1) % 4
        first = leaf.seq_log_density((x, ch, rj))
        second = leaf.seq_log_density((x, ch, rj))
        np.testing.assert_array_equal(first, second)
        self.assertTrue(policy.training)
        self.assertTrue(ref.training)


class _ModeRecorder(torch.nn.Module):
    """A linear module that records ``self.training`` at every forward -- what mode the M-step really ran in."""

    def __init__(self, out_dim: int):
        super().__init__()
        self.lin = torch.nn.Linear(2, out_dim)
        self.modes: list = []

    def forward(self, x):
        self.modes.append(self.training)
        return self.lin(x)


class FitModeRestoreTest(unittest.TestCase):
    """G-1: the M-step optimizes under train-mode semantics yet restores the caller's flag, both ways."""

    def _assert_fit_trains_and_restores(self, module: _ModeRecorder, fit) -> None:
        for pre_training in (False, True):
            module.train(pre_training)
            module.modes.clear()
            fit()
            self.assertTrue(module.modes)  # the M-step really ran forwards ...
            self.assertTrue(all(module.modes))  # ... under train-mode semantics, even for a pre-eval()ed module
            self.assertEqual(module.training, pre_training)  # and the caller's flag comes back as it was left
            self.assertTrue(all(m.training == pre_training for m in module.modules()))

    def test_neural_gaussian_fit_trains_and_restores_training_flag(self):
        module = _ModeRecorder(out_dim=1)
        leaf = NeuralGaussian(module=module, noise=1.0, m_steps=3, lr=0.01)
        est = leaf.estimator()
        acc = est.accumulator_factory().make()
        x, y = _xy()
        acc.seq_update(leaf.dist_to_encoder().seq_encode(list(zip(x, y))), np.ones(len(x)), leaf)
        self._assert_fit_trains_and_restores(module, lambda: est.estimate(None, acc.value()))

    def test_neural_categorical_fit_trains_and_restores_training_flag(self):
        module = _ModeRecorder(out_dim=3)
        leaf = NeuralCategorical(module=module, m_steps=3, lr=0.01)
        est = leaf.estimator()
        acc = est.accumulator_factory().make()
        x, c = _xc()
        acc.seq_update(leaf.dist_to_encoder().seq_encode(list(zip(x, c))), np.ones(len(x)), leaf)
        self._assert_fit_trains_and_restores(module, lambda: est.estimate(None, acc.value()))

    def test_dpo_fit_trains_policy_and_restores_training_flag(self):
        policy = _ModeRecorder(out_dim=4)
        ref = _dropout_classifier(classes=4)
        leaf = DPOModel(policy=policy, ref=ref, beta=0.1, m_steps=3, lr=1e-3)
        est = leaf.estimator()
        acc = est.accumulator_factory().make()
        rng = np.random.RandomState(0)
        x = rng.randn(12, 2)
        ch = rng.randint(0, 4, size=12)
        rj = (ch + 1) % 4
        acc.seq_update((x, ch, rj), np.ones(len(x)), leaf)
        self._assert_fit_trains_and_restores(policy, lambda: est.estimate(None, acc.value()))

    def test_streaming_transformer_train_step_trains_and_restores_training_flag(self):
        module = _ModeRecorder(out_dim=5)
        est = StreamingTransformer(module=module).estimator()
        acc = est.accumulator_factory().make()
        x, c = _xc(classes=5)
        self._assert_fit_trains_and_restores(module, lambda: acc.seq_update((x, c), np.ones(len(x)), None))


class _CondGauss(torch.nn.Module):
    """The smallest conditional density module: ``p(y | x) = N(y; w x + b, 1)`` (arity-2 log_density)."""

    def __init__(self):
        super().__init__()
        self.lin = torch.nn.Linear(1, 1)

    def log_density(self, x, y):
        return (-0.5 * (y - self.lin(x)) ** 2 - 0.5 * float(np.log(2.0 * np.pi))).sum(-1)


class DataBufferFanInTest(unittest.TestCase):
    """G-2: a fresh root's combine() adopts the worker's field arity instead of dropping fields."""

    @staticmethod
    def _worker_value() -> tuple:
        rng = np.random.RandomState(0)
        x = rng.uniform(-2.0, 2.0, size=40)
        y = 1.5 * x + 0.1 * rng.randn(40)
        data = list(zip(x[:, None], y[:, None]))
        factory = DataBufferAccumulatorFactory(GradLeafEncoder(), n_fields=1)
        worker = factory.make()
        worker.seq_update(GradLeafEncoder().seq_encode(data), np.ones(len(data)), None)
        return worker.value()

    def test_fresh_root_combine_preserves_conditional_arity_and_shapes(self):
        worker_value = self._worker_value()
        root = DataBufferAccumulatorFactory(GradLeafEncoder(), n_fields=1).make()
        root.combine(worker_value)
        root_value = root.value()
        self.assertEqual(len(root_value), len(worker_value))  # (x, y, w) survives the fan-in intact
        for got, expected in zip(root_value, worker_value):
            np.testing.assert_array_equal(got, expected)

    def test_fan_in_fit_matches_single_worker_fit(self):
        worker_value = self._worker_value()
        root = DataBufferAccumulatorFactory(GradLeafEncoder(), n_fields=1).make()
        root.combine(worker_value)

        torch.manual_seed(7)
        direct_module = _CondGauss()
        torch.manual_seed(7)
        fanned_module = _CondGauss()
        direct = GradEstimator(module=direct_module, m_steps=5, lr=0.05).estimate(None, worker_value)
        fanned = GradEstimator(module=fanned_module, m_steps=5, lr=0.05).estimate(None, root.value())
        torch.testing.assert_close(fanned.module.lin.weight, direct.module.lin.weight, rtol=0.0, atol=0.0)
        torch.testing.assert_close(fanned.module.lin.bias, direct.module.lin.bias, rtol=0.0, atol=0.0)


class _DtypeProbe(torch.nn.Module):
    """A linear regressor that records the dtype of every forward input."""

    def __init__(self):
        super().__init__()
        self.lin = torch.nn.Linear(2, 1)
        self.seen: list = []

    def forward(self, x):
        self.seen.append(x.dtype)
        return self.lin(x)


class NeuralGaussianEngineDtypeTest(unittest.TestCase):
    """G-5: NeuralGaussian follows an active torch engine's float dtype instead of hardcoding fp32."""

    def test_forward_follows_a_float64_engine_and_default_stays_float32(self):
        from mixle.engines import TorchEngine
        from mixle.engines.base import using_active_engine

        x, y = _xy()
        default_probe = _DtypeProbe()
        NeuralGaussian(module=default_probe, noise=1.0).seq_log_density((x, y))
        self.assertEqual(default_probe.seen[-1], torch.float32)  # no engine: the historical default

        engine_probe = _DtypeProbe()
        leaf = NeuralGaussian(module=engine_probe, noise=1.0)
        with using_active_engine(TorchEngine(dtype=torch.float64)):
            leaf.seq_log_density((x, y))
        self.assertEqual(engine_probe.seen[-1], torch.float64)
        self.assertEqual(next(engine_probe.parameters()).dtype, torch.float64)
        # scoring after the engine context stays consistent with the engine-cast module (the
        # estimation loop activates the engine only around estimate(), never around E-step scoring)
        leaf.seq_log_density((x, y))
        self.assertEqual(engine_probe.seen[-1], torch.float64)

    def test_m_step_follows_a_float64_engine(self):
        from mixle.engines import TorchEngine
        from mixle.engines.base import using_active_engine

        probe = _DtypeProbe()
        leaf = NeuralGaussian(module=probe, noise=1.0, m_steps=2, lr=0.01)
        est = leaf.estimator()
        acc = est.accumulator_factory().make()
        x, y = _xy()
        acc.seq_update(leaf.dist_to_encoder().seq_encode(list(zip(x, y))), np.ones(len(x)), leaf)
        with using_active_engine(TorchEngine(dtype=torch.float64)):
            fitted = est.estimate(None, acc.value())
        self.assertEqual(probe.seen[-1], torch.float64)
        self.assertTrue(np.isfinite(fitted.noise))


if __name__ == "__main__":
    unittest.main()
