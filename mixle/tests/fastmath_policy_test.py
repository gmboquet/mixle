"""Fastmath policy + nested-fusion gating for the -inf-semantics kernel modules.

The generated fused kernels and the HMM numba kernels rely on ``-np.inf`` as a SEMANTIC value:
out-of-support scores, log mixture weights, the all--inf log-sum-exp guard, and the HMM
``llsum += -np.inf`` impossible-observation branches. Compiling them with full ``fastmath=True``
sets LLVM's ``ninf``/``nnan`` flags, which declare those values undefined behavior -- and really
miscompiled: a fused Pareto-mixture scorer returned POSITIVE log-densities on rows where one
component is out of support. Policy pinned here: these modules may use the safe reassociation
subset, never full ``fastmath=True`` (no ``ninf``/``nnan``).

Also pinned (source-level, so the fast gate carries the signal without a numba compile):

* ``wants_minmax`` leaf templates (Pareto) must DECLINE nested fusion -- the nested emitter has no
  ``to_value_g`` / support-min plumbing, so admitting them crashed mid-fit with a ``TypeError``.
* the nested emitters guard every mixture node's log-sum-exp and responsibility pushdown against
  all-children--inf rows (no nested-fusible leaf can currently score -inf, so the emitted source
  is the only place the guard can be asserted without one).

Deliberately numba-free: everything here inspects source or the pure-python analyzers.
"""

import re
import unittest
from pathlib import Path

import mixle.stats as stats
from mixle.stats.compute import fused_codegen, fused_nested

# every module whose kernels carry -inf semantics (review findings D-1 / D-7)
_MINUS_INF_KERNEL_MODULES = (
    "mixle.stats.compute.fused_codegen",
    "mixle.stats.compute.fused_nested",
    "mixle.stats.latent._hidden_markov_numba_kernels",
    "mixle.stats.latent.tree_hidden_markov_model",
)


def _module_source_without_comments(dotted: str) -> str:
    import importlib

    mod = importlib.import_module(dotted)
    src = Path(mod.__file__).read_text()
    # crude comment strip (no '#' appears inside string literals near fastmath usage in these files);
    # keeps the policy check from tripping on prose that MENTIONS fastmath=True
    return "\n".join(line.split("#", 1)[0] for line in src.splitlines())


class FastmathPolicyTest(unittest.TestCase):
    def test_no_full_fastmath_in_minus_inf_kernel_modules(self):
        for dotted in _MINUS_INF_KERNEL_MODULES:
            code = _module_source_without_comments(dotted)
            self.assertIsNone(
                re.search(r"fastmath\s*=\s*True", code),
                f"{dotted} compiles a kernel with full fastmath=True (ninf/nnan): -inf semantics break",
            )

    def test_fastmath_flag_sets_never_enable_ninf_or_nnan(self):
        for dotted in _MINUS_INF_KERNEL_MODULES:
            code = _module_source_without_comments(dotted)
            for flag in ("ninf", "nnan"):
                self.assertNotIn(f"'{flag}'", code, f"{dotted} enables fastmath {flag}")
                self.assertNotIn(f'"{flag}"', code, f"{dotted} enables fastmath {flag}")


class NestedMinmaxExclusionTest(unittest.TestCase):
    def _pareto_mixture_of_mixtures(self):
        inner1 = stats.MixtureDistribution(
            components=[
                stats.ParetoDistribution(xm=1.0, alpha=2.0),
                stats.ParetoDistribution(xm=2.0, alpha=3.0),
            ],
            w=[0.6, 0.4],
        )
        inner2 = stats.MixtureDistribution(
            components=[
                stats.ParetoDistribution(xm=1.5, alpha=1.5),
                stats.ParetoDistribution(xm=3.0, alpha=2.5),
            ],
            w=[0.5, 0.5],
        )
        return stats.MixtureDistribution(components=[inner1, inner2], w=[0.7, 0.3])

    def test_minmax_leaves_decline_nested_fusion_on_every_gate(self):
        # wants_minmax templates have to_value=None; the nested emitter would call it mid-fit
        mm = self._pareto_mixture_of_mixtures()
        self.assertFalse(fused_nested.fusible_nested(mm))
        self.assertFalse(fused_codegen.fusible(mm, bare_bridge=False))
        self.assertFalse(fused_codegen.fusible_estep(mm, bare_bridge=False))
        # the shape is still COVERED, by the bare-bridge last resort: per-component native
        # scoring/accumulation has no to_value problem (parity pinned in fused_out_of_support_test)
        self.assertTrue(fused_codegen.fusible(mm))
        self.assertTrue(fused_codegen.fusible_estep(mm))

    def test_flat_pareto_mixture_still_fuses(self):
        # the exclusion is scoped to NESTING: the flat compiler has the minmax plumbing
        flat = stats.MixtureDistribution(
            components=[
                stats.ParetoDistribution(xm=1.0, alpha=2.5),
                stats.ParetoDistribution(xm=2.0, alpha=1.5),
            ],
            w=[0.5, 0.5],
        )
        self.assertTrue(fused_codegen.fusible(flat))
        self.assertTrue(fused_codegen.fusible_estep(flat))


class NestedEmitterGuardTest(unittest.TestCase):
    """The nested emitters must guard EVERY mixture node against all-children--inf rows.

    Without the guard, an impossible subtree turns ``-inf - (-inf)`` into NaN in both the score
    (log-sum-exp) and the E-step responsibility pushdown, which then poisons the model LL and every
    sufficient statistic. The numeric behavior is pinned in fused_out_of_support_test (numba); this
    pins the emitted source itself.
    """

    def _built_tree(self):
        inner1 = stats.MixtureDistribution(
            components=[
                stats.GaussianDistribution(mu=-1.0, sigma2=1.0),
                stats.GaussianDistribution(mu=0.0, sigma2=2.0),
            ],
            w=[0.6, 0.4],
        )
        inner2 = stats.MixtureDistribution(
            components=[stats.GaussianDistribution(mu=3.0, sigma2=1.0), stats.GaussianDistribution(mu=5.0, sigma2=0.5)],
            w=[0.5, 0.5],
        )
        model = stats.MixtureDistribution(components=[inner1, inner2], w=[0.7, 0.3])
        built = fused_nested.analyze_nested(model)
        self.assertIsNotNone(built)
        return built

    def test_score_emitter_guards_every_mixture_node(self):
        root, _ = self._built_tree()
        lines: list[str] = []
        fused_nested._emit_score(root, lines)
        src = "\n".join(lines)
        for mx in fused_nested._mixtures(root):
            nid = mx.node_id
            self.assertIn(
                f"if mx{nid} > -np.inf else -np.inf",
                src,
                f"mixture node {nid} lacks the all-children--inf log-sum-exp guard",
            )

    def test_backward_emitter_falls_back_to_prior_weights(self):
        root, _ = self._built_tree()
        fwd: list[str] = []
        fused_nested._emit_score(root, fwd)
        bwd: list[str] = []
        fused_nested._emit_backward(root, "wi", bwd)
        src = "\n".join(bwd)
        for mx in fused_nested._mixtures(root):
            nid = mx.node_id
            for j in range(len(mx.children)):
                self.assertIn(
                    f"if ns{nid} > -np.inf else np.exp(logw{nid}[{j}])",
                    src,
                    f"mixture node {nid} child {j} lacks the prior-weight responsibility fallback",
                )


if __name__ == "__main__":
    unittest.main()
