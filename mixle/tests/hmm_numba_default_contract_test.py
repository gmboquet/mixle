"""Contract: every ``use_numba`` parameter in the latent family defaults to ``None``.

``None`` resolves to ``HAS_NUMBA`` at construction (the pattern established by
``HiddenMarkovModelDistribution``), so numba engages exactly when it is installed. A hardcoded
``False`` default silently forces the ~6x-slower numpy Baum-Welch even with numba present (the
recurring bug this test exists to kill); a hardcoded ``True`` silently runs numba-shaped kernels
in pure Python when numba is absent. Deliberate force-offs (terminal-state layouts, the
semi-supervised custom updater) assign ``self.use_numba = False`` in the body, never through a
parameter default, so they pass this scan by construction.
"""

import ast
import unittest
from pathlib import Path

import mixle.stats.latent as latent_pkg
from mixle.utils.optional_deps import HAS_NUMBA

LATENT_DIR = Path(latent_pkg.__file__).parent


def _bad_defaults():
    """Return every (file, function, default) where a use_numba parameter defaults to non-None."""
    offenders = []
    for path in sorted(LATENT_DIR.glob("*.py")):
        tree = ast.parse(path.read_text())
        for node in ast.walk(tree):
            if not isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                continue
            args = node.args
            params = args.posonlyargs + args.args + args.kwonlyargs
            defaults = [None] * (len(args.posonlyargs) + len(args.args) - len(args.defaults))
            defaults += list(args.defaults) + list(args.kw_defaults)
            for param, default in zip(params, defaults):
                if param.arg != "use_numba" or default is None:
                    continue
                if not (isinstance(default, ast.Constant) and default.value is None):
                    offenders.append((path.name, node.name, ast.unparse(default)))
    return offenders


class UseNumbaDefaultContractTest(unittest.TestCase):
    def test_every_use_numba_parameter_defaults_to_none(self):
        offenders = _bad_defaults()
        self.assertEqual(
            offenders,
            [],
            "use_numba parameter defaults must be None (resolved to HAS_NUMBA at construction); "
            "hardcoded defaults silently pin the slow or the broken path: %r" % (offenders,),
        )

    def test_entry_points_resolve_none_to_has_numba(self):
        from mixle.stats import GaussianDistribution
        from mixle.stats.latent.hidden_markov import HiddenMarkovModelDistribution
        from mixle.stats.latent.tree_hidden_markov_model import TreeHiddenMarkovEstimator

        hmm = HiddenMarkovModelDistribution(
            topics=[GaussianDistribution(0.0, 1.0), GaussianDistribution(3.0, 1.0)],
            w=[0.5, 0.5],
            transitions=[[0.9, 0.1], [0.1, 0.9]],
        )
        self.assertEqual(hmm.use_numba, HAS_NUMBA)

        from mixle.stats import GaussianEstimator
        from mixle.stats.latent.hidden_markov import HiddenMarkovEstimator

        est = HiddenMarkovEstimator([GaussianEstimator(), GaussianEstimator()])
        self.assertEqual(est.use_numba, HAS_NUMBA)
        # the resolved value must propagate unchanged through the accumulator pipeline
        factory = est.accumulator_factory()
        self.assertEqual(factory.use_numba, HAS_NUMBA)
        self.assertEqual(factory.make().use_numba, HAS_NUMBA)
        self.assertEqual(hmm.dist_to_encoder().use_numba, HAS_NUMBA)

        tree_est = TreeHiddenMarkovEstimator([GaussianEstimator(), GaussianEstimator()])
        self.assertEqual(tree_est.use_numba, HAS_NUMBA)

    def test_terminal_states_still_force_the_per_sequence_layout(self):
        from mixle.stats import GaussianDistribution
        from mixle.stats.latent.hidden_markov import HiddenMarkovModelDistribution

        hmm = HiddenMarkovModelDistribution(
            topics=[GaussianDistribution(0.0, 1.0), GaussianDistribution(3.0, 1.0)],
            w=[0.5, 0.5],
            transitions=[[0.9, 0.1], [0.1, 0.9]],
            terminal_states=[1],
        )
        self.assertFalse(hmm.use_numba)


if __name__ == "__main__":
    unittest.main()
