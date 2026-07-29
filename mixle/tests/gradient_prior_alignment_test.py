"""Structural gradient priors must align exactly unless intent is receipted."""

import unittest

import numpy as np

from mixle.inference.priors import (
    CompositePrior,
    GammaPrior,
    MarkovChainPrior,
    MixturePrior,
    PriorAlignmentReceipt,
    RecordPrior,
    as_prior_dict,
    broadcast,
    partial_alignment,
)
from mixle.stats.compute.gradient import (
    composite_child_priors,
    conditional_priors,
    dirichlet_alpha_tensor,
    markov_chain_priors,
    mixture_priors,
    normal_gamma_log_prior,
    record_child_priors,
    sequence_priors,
    validated_prior_scalar,
)


class GradientPriorAlignmentTest(unittest.TestCase):
    def test_alignment_receipts_require_supported_mode_and_rationale(self):
        with self.assertRaises(ValueError):
            PriorAlignmentReceipt("implicit", "legacy behavior")
        with self.assertRaises(ValueError):
            PriorAlignmentReceipt("partial", " ")

    def test_composite_is_exact_by_default_and_partial_is_explicit(self):
        with self.assertRaisesRegex(ValueError, "exact alignment requires 2"):
            composite_child_priors(as_prior_dict(CompositePrior([{"family": "gamma"}])), 2)
        with self.assertRaisesRegex(ValueError, "structure has only 2"):
            composite_child_priors(
                as_prior_dict(
                    CompositePrior(
                        [1, 2, 3],
                        alignment_receipt=partial_alignment("only supplied positions are regularized"),
                    )
                ),
                2,
            )

        aligned = composite_child_priors(
            as_prior_dict(
                CompositePrior(
                    [{"family": "gamma"}],
                    alignment_receipt=partial_alignment("leave the second child unregularized"),
                )
            ),
            2,
        )
        self.assertEqual(aligned, [{"family": "gamma"}, None])

    def test_broadcast_requires_and_preserves_a_receipt(self):
        prior = as_prior_dict(broadcast(GammaPrior(2.0, 3.0), "share one calibrated prior across children"))
        children = composite_child_priors(prior, 3)
        self.assertEqual(len(children), 3)
        self.assertTrue(all(child["family"] == "gamma" for child in children))

        with self.assertRaisesRegex(ValueError, "requires exact children or an explicit"):
            composite_child_priors({"family": "gamma", "shape": 2.0, "rate": 3.0}, 3)

    def test_keyed_structures_reject_missing_extra_and_unknown_keys(self):
        with self.assertRaisesRegex(ValueError, "missing keys"):
            conditional_priors({"a": 1}, ("a", "b"))
        with self.assertRaisesRegex(ValueError, "unknown keys"):
            conditional_priors({"a": 1, "b": 2, "typo": 3}, ("a", "b"))
        with self.assertRaisesRegex(ValueError, "missing keys"):
            record_child_priors(as_prior_dict(RecordPrior({"x": 1})), ("x", "y"), 2)

        record = RecordPrior(
            {"x": 1},
            alignment_receipt=partial_alignment("field y intentionally uses no prior"),
        )
        self.assertEqual(record_child_priors(as_prior_dict(record), ("x", "y"), 2), [1, None])

    def test_mixture_and_markov_partial_routes_are_receipted(self):
        with self.assertRaisesRegex(ValueError, "exact alignment requires 2"):
            mixture_priors(as_prior_dict(MixturePrior(weights={"family": "dirichlet"})), 2)
        components, weights = mixture_priors(
            as_prior_dict(
                MixturePrior(
                    weights={"family": "dirichlet"},
                    alignment_receipt=partial_alignment("regularize weights but not components"),
                )
            ),
            2,
        )
        self.assertEqual(components, [None, None])
        self.assertEqual(weights["family"], "dirichlet")

        with self.assertRaisesRegex(ValueError, "missing keys"):
            markov_chain_priors(
                as_prior_dict(MarkovChainPrior(transitions={"a": 1})),
                ("a", "b"),
            )
        _, rows, _ = markov_chain_priors(
            as_prior_dict(
                MarkovChainPrior(
                    transitions={"a": 1},
                    alignment_receipt=partial_alignment("transition row b intentionally uses no prior"),
                )
            ),
            ("a", "b"),
        )
        self.assertEqual(rows, {"a": 1, "b": None})

    def test_sequence_and_dirichlet_labels_do_not_silently_pad(self):
        with self.assertRaisesRegex(ValueError, "exact alignment requires 2"):
            sequence_priors((1,), has_length=True)
        with self.assertRaisesRegex(ValueError, "missing labels"):
            dirichlet_alpha_tensor({"a": 2.0}, ("a", "b"), np.zeros(2), np, None)
        with self.assertRaisesRegex(ValueError, "unknown keys"):
            dirichlet_alpha_tensor({"a": 2.0, "b": 2.0, "typo": 1.0}, ("a", "b"), np.zeros(2), np, None)

    def test_dirichlet_concentrations_require_exact_positive_finite_shape(self):
        logits = np.zeros(2)
        for alpha in ([1.0], [1.0, 0.0], [1.0, -1.0], [1.0, np.inf], [1.0, np.nan]):
            with self.subTest(alpha=repr(alpha)):
                with self.assertRaises(ValueError):
                    dirichlet_alpha_tensor(alpha, None, logits, np, np)
        with self.assertRaises(TypeError):
            dirichlet_alpha_tensor([True, True], None, logits, np, np)
        np.testing.assert_allclose(
            dirichlet_alpha_tensor([2.0, 3.0], None, logits, np, np),
            [2.0, 3.0],
        )

    def test_scalar_hyperparameters_reject_invalid_and_unreceipted_improper_values(self):
        for value in (0.0, -1.0, np.inf, np.nan):
            with self.subTest(value=repr(value)):
                with self.assertRaises(ValueError):
                    validated_prior_scalar({"family": "gamma", "shape": value}, "shape")
        with self.assertRaises(TypeError):
            validated_prior_scalar({"family": "gamma", "shape": True}, "shape")
        with self.assertRaises(ValueError):
            validated_prior_scalar({"family": "gamma", "shape": [1.0]}, "shape")
        with self.assertRaisesRegex(ValueError, "aliases.*disagree"):
            validated_prior_scalar({"family": "normalgamma", "alpha": 2.0, "a": 3.0}, "alpha", aliases=("a",))

        acknowledged = {
            "family": "gamma",
            "shape": 0.0,
            "proper": False,
            "improper_receipt": {
                "status": "improper_limit_acknowledged",
                "rationale": "evaluate a documented limiting posterior",
            },
        }
        self.assertEqual(validated_prior_scalar(acknowledged, "shape"), 0.0)

    def test_normal_gamma_requires_all_finite_proper_controls(self):
        valid = {
            "family": "normalgamma",
            "alpha": 2.0,
            "beta": 1.0,
            "kappa": 0.5,
            "mu0": 0.0,
        }
        self.assertTrue(np.isfinite(normal_gamma_log_prior(1.0, 2.0, valid, np)))
        for key, value in (("alpha", 0.0), ("beta", -1.0), ("kappa", np.inf), ("mu0", np.nan)):
            malformed = dict(valid)
            malformed[key] = value
            with self.subTest(key=repr(key), value=repr(value)):
                with self.assertRaises(ValueError):
                    normal_gamma_log_prior(1.0, 2.0, malformed, np)
        missing = dict(valid)
        del missing["kappa"]
        with self.assertRaisesRegex(ValueError, "requires hyperparameter 'kappa'"):
            normal_gamma_log_prior(1.0, 2.0, missing, np)


if __name__ == "__main__":
    unittest.main()
