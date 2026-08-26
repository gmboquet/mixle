"""0.8.0 release-wave regressions for the solve()/doe headline path (B13, B14, base-install refusals).

B13: ``HashedRecord`` encoded every numeric field as a single raw ``tanh(value)``, which saturates to
exactly +/-1 for |value| > ~19 -- ordinary tabular records (masses, lengths, prices) all mapped to ONE
identical feature vector, so the README-quickstart student was information-free (agreement at chance,
100% escalation) while ``report()`` still said ``promoted=True``. Fixed by multiscale numeric features
(plus a saturation-free magnitude feature), and by demoting a Solution whose real answer-or-escalate
rule answered zero selection rows.

B14: the density gate's floor is itself a calibration score, and ``seq_log_density`` reduces in a
batch-shape-dependent order, so an input sitting exactly ON the floor flipped between the batch path
(offline metrics: ~7% escalation) and the one-at-a-time serving path (100% escalation) on 1 ULP of
noise. Fixed by a shared guard-banded compare used by both ``is_ood`` and ``ood_mask``.

Base install: ``solve()``'s default student and ``mixle.doe.minimize``'s GP surrogate need torch, but
used to surface a bare ``ModuleNotFoundError`` only AFTER the teacher had labeled every input /
``n_init`` objective evaluations were spent. Both now refuse up front with the extra named.
"""

import hashlib
import os
import subprocess
import sys
import unittest
import warnings
from pathlib import Path

import numpy as np

try:
    import torch  # noqa: F401

    _HAS_TORCH = True
except ImportError:
    _HAS_TORCH = False

REPO_ROOT = Path(__file__).resolve().parents[2]

# penguin-like magnitudes: three classes over four large, same-signed numeric fields -- exactly the
# shape that saturated the legacy single-tanh encoding into one constant vector for every row
_CENTERS = ((39.0, 18.5, 190.0, 3700.0), (46.0, 17.5, 196.0, 3730.0), (47.5, 15.0, 217.0, 5080.0))
_SPREADS = (2.5, 1.0, 6.0, 350.0)
_SPECIES = ("Adelie", "Chinstrap", "Gentoo")


def _penguinish(n=342, seed=0):
    """Deterministic (rows, labels) with per-row class recoverable from the row alone."""
    rng = np.random.RandomState(seed)
    rows, labels = [], []
    for i in range(n):
        k = i % 3
        c = _CENTERS[k]
        rows.append(tuple(float(c[j] + _SPREADS[j] * rng.randn()) for j in range(4)))
        labels.append(_SPECIES[k])
    return rows, labels


def _lookup_teacher(rows, labels):
    truth = dict(zip(rows, labels))

    def teacher(xs):
        if isinstance(xs, list):
            return [truth[x] for x in xs]
        return truth[xs]

    return teacher


class RecordFeaturizationCollapseTest(unittest.TestCase):
    """B13 root cause: large same-signed numerics must not collapse to one feature vector."""

    def test_large_same_signed_numerics_stay_distinguishable(self):
        from mixle.task.model import HashedRecord

        rows, _ = _penguinish()
        feat = HashedRecord.for_records(rows)
        x = feat.transform(rows)
        self.assertEqual(np.unique(x, axis=0).shape[0], len(rows))
        # and the defect shape is pinned: the legacy encoding really did collapse these rows
        legacy = HashedRecord.from_spec({"dim": 256, "seed": 0, "record_kind": "sequence", "record_width": 4})
        self.assertEqual(legacy.numeric_encoding, "tanh")
        x_old = legacy.transform(rows)
        self.assertEqual(np.unique(x_old, axis=0).shape[0], 1)

    def test_magnitudes_beyond_the_scale_ladder_stay_distinguishable(self):
        # unix-timestamp scale (~1.7e9) saturates every ladder scale; the arcsinh magnitude
        # feature must still separate rows a day apart
        from mixle.task.model import HashedRecord

        rows = [(1.7e9 + 86400.0 * i,) for i in range(30)]
        feat = HashedRecord.for_records(rows)
        x = feat.transform(rows)
        self.assertEqual(np.unique(x, axis=0).shape[0], len(rows))

    def test_spec_round_trip_keeps_the_encoding_and_legacy_specs_stay_legacy(self):
        from mixle.task.model import HashedRecord

        rows, _ = _penguinish(n=12)
        feat = HashedRecord.for_records(rows)
        spec = feat.to_spec()
        self.assertEqual(spec["numeric_encoding"], "tanh-multiscale")
        rebuilt = HashedRecord.from_spec(spec)
        np.testing.assert_array_equal(rebuilt.transform(rows), feat.transform(rows))
        # a pre-0.8.0 spec has no numeric_encoding key: it must rebuild the exact features the
        # artifact's model was trained on (single tanh), not the new ladder
        legacy_spec = {k: v for k, v in spec.items() if k != "numeric_encoding"}
        legacy = HashedRecord.from_spec(legacy_spec)
        self.assertEqual(legacy.numeric_encoding, "tanh")
        row = rows[0]
        vec = legacy.transform([row])[0]
        by_hand = np.zeros(256, dtype=np.float32)
        for i, value in enumerate(row):
            by_hand[legacy._bucket(f"num:{i}")] += float(np.tanh(value))
            by_hand[legacy._bucket(f"has:{i}")] += 1.0
        by_hand /= np.linalg.norm(by_hand)
        np.testing.assert_allclose(vec, by_hand, rtol=0, atol=1e-6)

    def test_unknown_numeric_encoding_is_refused_by_name(self):
        from mixle.task.model import HashedRecord

        with self.assertRaisesRegex(ValueError, "numeric_encoding"):
            HashedRecord(record_kind="scalar", numeric_encoding="fourier")


class DensityGateThresholdParityTest(unittest.TestCase):
    """B14: batch and one-at-a-time gate decisions must agree, including exactly at the floor."""

    def test_degenerate_density_scores_never_fire_the_gate_on_either_path(self):
        from mixle.task.density import DensityGate
        from mixle.task.model import HashedRecord

        same = [(1.0, 2.0, 3.0)] * 40
        feat = HashedRecord.for_records(same)
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            gate = DensityGate(feat).fit(same, alpha=0.05, seed=0)
        # every calibration score sits exactly ON the floor: the batch path scored 0% OOD while the
        # serving path scored 100% on 1 ULP of reduction-order noise. Both must now say 0%.
        self.assertEqual(float(np.mean(gate.ood_mask(same))), 0.0)
        self.assertEqual(sum(gate.is_ood(x) for x in same), 0)
        # and the degenerate featurization is called out, not silently accepted
        self.assertTrue(any("identical" in str(w.message) for w in caught))

    def test_a_genuinely_novel_input_still_escalates_on_both_paths(self):
        from mixle.task.density import DensityGate
        from mixle.task.model import HashedRecord

        rows, _ = _penguinish(n=200)
        feat = HashedRecord.for_records(rows)
        gate = DensityGate(feat).fit(rows, alpha=0.05, seed=0)
        novel = (-4000.0, 0.001, -250.0, 0.5)  # wildly off the training manifold
        self.assertTrue(gate.is_ood(novel))
        self.assertTrue(bool(gate.ood_mask([novel])[0]))

    def test_decide_and_batch_decide_agree_on_training_distribution(self):
        from mixle.task import solve

        rows, labels = _penguinish()
        sol = solve(_lookup_teacher(rows, labels), rows, student="generative", seed=0)
        cal = sol.cascade.model
        live = rows[:100]
        self.assertEqual([cal.decide(x) for x in live], cal.batch_decide(live))

    def test_live_escalation_tracks_the_reported_offline_rate(self):
        from mixle.task import solve
        from mixle.task.calibrate import ESCALATE

        rows, labels = _penguinish()
        sol = solve(_lookup_teacher(rows, labels), rows, student="generative", seed=0)
        live = rows[:100]
        for x in live:
            sol(x)
        rep = sol.report()
        live_rate = rep["live_escalated"] / rep["requests"]
        # B14's failure mode was 100% live against a reported 7%: on training-distribution queries
        # the two must agree to within a few points (measured: reported 0.186, live 0.15, seed 0)
        self.assertLessEqual(abs(live_rate - rep["holdout_escalation_rate"]), 0.10)
        self.assertLess(live_rate, 0.5)
        # the monitor agrees with serving again as well
        self.assertFalse(sol.health()["drifted"])
        # decide() is what serving runs; the harvested counts must reflect the same rule
        self.assertEqual(sum(sol.cascade.model.decide(x) is ESCALATE for x in live), rep["live_escalated"])


@unittest.skipUnless(_HAS_TORCH, "torch not installed")
class SolveHonestReportingTest(unittest.TestCase):
    """B13 at the solve() level: the README default path learns, and promotion never lies."""

    def test_default_student_learns_large_magnitude_tabular_records(self):
        from mixle.task import solve

        rows, labels = _penguinish()
        sol = solve(_lookup_teacher(rows, labels), rows, seed=0)
        rep = sol.report()
        # majority class on this data is 1/3; the legacy encoding pinned agreement at chance with
        # 100% escalation (measured 0.30 / 1.0 / answered_slice None before the fix)
        self.assertGreaterEqual(rep["holdout_agreement"], 0.6)
        self.assertLess(rep["holdout_escalation_rate"], 1.0)
        self.assertIsInstance(rep["answered_slice"], dict)
        self.assertGreater(rep["answered_slice"]["n_answered"], 0)
        for key in ("agreement", "n_answered", "n_evaluated", "ci95"):
            self.assertIn(key, rep["answered_slice"])
        self.assertTrue(rep["promoted"])
        # and a promoted student really answers some live training-distribution traffic
        for x in rows[:100]:
            sol(x)
        self.assertLess(sol.report()["live_escalated"], 100)

    def test_student_that_answers_nothing_is_demoted_not_promoted(self):
        from mixle.task import solve

        def unlearnable(xs):
            def one(x):
                digest = hashlib.blake2b(repr(x).encode(), digest_size=4).digest()
                return "wxyz"[digest[0] % 4]

            return [one(x) for x in xs] if isinstance(xs, list) else one(xs)

        rng = np.random.RandomState(3)
        rows = [tuple(float(v) for v in rng.uniform(0, 50, size=3)) for _ in range(160)]
        sol = solve(unlearnable, rows, seed=0, epochs=60)
        rep = sol.report()
        # the gated rule answered zero selection rows: promoted=True here was B13's lie
        self.assertIsNone(rep["answered_slice"])
        self.assertFalse(rep["promoted"])
        # a demoted Solution still serves correct answers -- straight from the teacher
        self.assertEqual(sol(rows[0]), unlearnable(rows[0]))


class BaseInstallRefusalTest(unittest.TestCase):
    """Torch-needing entry points refuse up front, with the extra named, before spending budget."""

    @staticmethod
    def _run_without_torch(code: str) -> subprocess.CompletedProcess:
        # same worktree-pinning rationale as doe_task_models_clean_import_test._clean_import; the
        # prelude simulates a base install by refusing every (re)import of torch in the subprocess
        prelude = (
            "import sys\n"
            "class _BlockTorch:\n"
            "    def find_spec(self, name, path=None, target=None):\n"
            "        if name == 'torch' or name.startswith('torch.'):\n"
            "            raise ModuleNotFoundError(f'No module named {name!r}', name=name)\n"
            "        return None\n"
            "sys.meta_path.insert(0, _BlockTorch())\n"
            "for _m in [m for m in sys.modules if m == 'torch' or m.startswith('torch.')]:\n"
            "    del sys.modules[_m]\n"
        )
        env = dict(os.environ)
        env["PYTHONPATH"] = str(REPO_ROOT)
        return subprocess.run(
            [sys.executable, "-c", prelude + code],
            cwd=REPO_ROOT,
            env=env,
            capture_output=True,
            text=True,
            timeout=180,
        )

    def test_solve_default_student_refuses_before_any_teacher_call(self):
        result = self._run_without_torch(
            "calls = {'n': 0}\n"
            "def teacher(xs):\n"
            "    calls['n'] += len(xs) if isinstance(xs, list) else 1\n"
            "    return ['a'] * len(xs) if isinstance(xs, list) else 'a'\n"
            "rows = [(float(i), float(i % 3)) for i in range(24)]\n"
            "from mixle.task import solve\n"
            "try:\n"
            "    solve(teacher, rows)\n"
            "except ImportError as e:\n"
            "    print('REFUSED', calls['n'], 'mixle[torch]' in str(e))\n"
        )
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn("REFUSED 0 True", result.stdout)

    def test_minimize_refuses_before_any_objective_evaluation(self):
        result = self._run_without_torch(
            "calls = {'n': 0}\n"
            "def objective(x):\n"
            "    calls['n'] += 1\n"
            "    return float((x ** 2).sum())\n"
            "from mixle.doe import minimize\n"
            "try:\n"
            "    minimize(objective, [(-5.0, 10.0), (0.0, 15.0)], n_init=10, n_iter=60, seed=0)\n"
            "except ImportError as e:\n"
            "    print('REFUSED', calls['n'], 'mixle[torch]' in str(e))\n"
            "res = minimize(objective, [(-5.0, 10.0), (0.0, 15.0)], n_init=6, n_iter=0, seed=0)\n"
            "print('DESIGN_ONLY', res.n_evaluations, res.stopped_reason)\n"
        )
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn("REFUSED 0 True", result.stdout)
        # a pure n_init design needs no surrogate and must stay runnable on a base install
        self.assertIn("DESIGN_ONLY 6 budget_exhausted", result.stdout)

    def test_generative_student_still_solves_end_to_end_without_torch(self):
        result = self._run_without_torch(
            "import numpy as np\n"
            "rng = np.random.RandomState(0)\n"
            "centers = ((39.0, 18.5, 190.0, 3700.0), (46.0, 17.5, 196.0, 3730.0), (47.5, 15.0, 217.0, 5080.0))\n"
            "spreads = (2.5, 1.0, 6.0, 350.0)\n"
            "rows = [tuple(float(centers[i % 3][j] + spreads[j] * rng.randn()) for j in range(4))"
            " for i in range(120)]\n"
            "labels = ['ACG'[i % 3] for i in range(120)]\n"
            "truth = dict(zip(rows, labels))\n"
            "teacher = lambda xs: [truth[x] for x in xs] if isinstance(xs, list) else truth[xs]\n"
            "from mixle.task import solve\n"
            "sol = solve(teacher, rows, student='generative', seed=0)\n"
            "for x in rows[:60]:\n"
            "    sol(x)\n"
            "rep = sol.report()\n"
            "print('OK', rep['promoted'], rep['live_escalated'] < 60)\n"
        )
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn("OK True True", result.stdout)


if __name__ == "__main__":
    unittest.main()
