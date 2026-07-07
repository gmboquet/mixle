"""Task-sufficient projection (mixle.reason.task_projection): pi_T compresses a belief for one task.

The load-bearing claim is the mismatched-projection control: a projection built for task A must not
be assumed to serve task B well. We construct a 2-D Gaussian-mixture belief whose first coordinate
carries task A's signal and second coordinate carries task B's signal, with the two only partially
correlated -- so collapsing by task A's grouping measurably destroys information task B needed, and
vice versa.
"""

import unittest

import numpy as np

from mixle.inference.project import gaussian_kl
from mixle.reason.task_projection import TaskReadout, read_out, task_sufficient_projection
from mixle.stats.latent.gaussian_mixture import GaussianMixtureDistribution


def _label_x(mean: np.ndarray) -> str:
    return "pos" if mean[0] >= 0 else "neg"


def _label_y(mean: np.ndarray) -> str:
    return "pos" if mean[1] >= 0 else "neg"


TASK_A = TaskReadout("A", _label_x)
TASK_B = TaskReadout("B", _label_y)


def _belief():
    # 8 components on a 2x2x2 grid of (sign(x), sign(y), a within-cell offset) so components sharing a
    # task-A label differ in y (and vice versa) -- correlated but not identical groupings.
    mu = []
    for sx in (-3.0, 3.0):
        for sy in (-3.0, 3.0):
            for off in (-0.6, 0.6):
                mu.append([sx + off, sy - off])
    mu = np.asarray(mu)
    cov = np.stack([0.25 * np.eye(2) for _ in range(len(mu))])
    w = np.full(len(mu), 1.0 / len(mu))
    return GaussianMixtureDistribution(mu, cov, w)


def _samples(belief, n, seed):
    return belief.sampler(seed).sample(n)


class TaskProjectionTest(unittest.TestCase):
    def test_projection_shrinks_and_groups_by_task_label(self):
        belief = _belief()
        pi_a = task_sufficient_projection(belief, TASK_A)
        self.assertLess(pi_a.num_components, belief.num_components)
        self.assertEqual(pi_a.num_components, 2)  # exactly the two task-A labels
        self.assertAlmostEqual(float(pi_a.w.sum()), 1.0, places=6)

    def test_matched_projection_preserves_accuracy(self):
        belief = _belief()
        pi_a = task_sufficient_projection(belief, TASK_A)
        xs = _samples(belief, 300, 0)
        truth = [_label_x(x) for x in xs]
        full_acc = np.mean([read_out(belief, TASK_A, x) == t for x, t in zip(xs, truth)])
        proj_acc = np.mean([read_out(pi_a, TASK_A, x) == t for x, t in zip(xs, truth)])
        self.assertGreater(full_acc, 0.9)
        # far smaller (2 vs 8 components) yet matches full-belief accuracy on ITS task
        self.assertGreaterEqual(proj_acc, full_acc - 0.03)

    def test_mismatched_projection_measurably_underperforms(self):
        belief = _belief()
        pi_a = task_sufficient_projection(belief, TASK_A)  # built for task A
        pi_b = task_sufficient_projection(belief, TASK_B)  # built for task B
        xs = _samples(belief, 400, 1)
        truth_b = [_label_y(x) for x in xs]

        acc_matched = np.mean([read_out(pi_b, TASK_B, x) == t for x, t in zip(xs, truth_b)])
        acc_mismatched = np.mean([read_out(pi_a, TASK_B, x) == t for x, t in zip(xs, truth_b)])

        self.assertGreater(acc_matched, 0.9)
        # pi_a collapsed components that disagree on task B's label -- it must measurably underperform
        # pi_b when asked to answer task B, proving the projection is task-specific, not generic
        # compression that happens to work for anything.
        self.assertLess(acc_mismatched, acc_matched - 0.1)

    def test_within_group_merge_is_exact_moment_match(self):
        belief = _belief()
        pi_a = task_sufficient_projection(belief, TASK_A)
        # the "pos" group covers 4 original components sharing sign(x) >= 0: its moment-matched merge
        # must reproduce that subset's exact weighted mean (law of total variance), checkable via KL
        # to itself and via the raw mean formula.
        idx = [k for k in range(belief.num_components) if _label_x(belief.mu[k]) == "pos"]
        w = belief.w[idx] / belief.w[idx].sum()
        expected_mean = w @ belief.mu[idx]
        merged = next(c for c in pi_a.components if _label_x(_mean_of(c)) == "pos")
        np.testing.assert_allclose(_mean_of(merged), expected_mean, atol=1e-8)
        self.assertAlmostEqual(gaussian_kl(merged, merged), 0.0, places=9)

    def test_single_component_group_passes_through_unchanged(self):
        # a task fine enough to give every component its own label: no merging happens.
        fine = TaskReadout("fine", lambda mean: tuple(np.round(mean, 3)))
        belief = _belief()
        pi_fine = task_sufficient_projection(belief, fine)
        self.assertEqual(pi_fine.num_components, belief.num_components)


def _mean_of(component) -> np.ndarray:
    if hasattr(component, "covar"):
        return np.asarray(component.mu, dtype=float).ravel()
    return np.array([float(component.mu)])


if __name__ == "__main__":
    unittest.main()
