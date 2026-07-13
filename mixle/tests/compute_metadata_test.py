import ast
import unittest
from pathlib import Path

import numpy as np

from mixle.stats import (
    AffineTransform,
    BernoulliDistribution,
    BernoulliEstimator,
    BernoulliSetDistribution,
    BetaDistribution,
    BetaEstimator,
    BinomialDistribution,
    BinomialEstimator,
    CategoricalDistribution,
    CategoricalEstimator,
    CompositeDistribution,
    CompositeEstimator,
    ConditionalDistribution,
    ConditionalDistributionEstimator,
    DiagonalGaussianDistribution,
    DiagonalGaussianEstimator,
    DiracLengthMixtureDistribution,
    DirichletDistribution,
    DirichletEstimator,
    ExponentialDistribution,
    ExponentialEstimator,
    GammaDistribution,
    GammaEstimator,
    GaussianDistribution,
    GaussianEstimator,
    GeometricDistribution,
    GeometricEstimator,
    HeterogeneousMixtureDistribution,
    HiddenAssociationDistribution,
    HiddenMarkovEstimator,
    HiddenMarkovModelDistribution,
    HierarchicalMixtureDistribution,
    HierarchicalMixtureEstimator,
    IgnoredDistribution,
    IgnoredEstimator,
    IndianBuffetProcessDistribution,
    IntegerBernoulliEditDistribution,
    IntegerBernoulliEditEstimator,
    IntegerBernoulliSetDistribution,
    IntegerCategoricalDistribution,
    IntegerCategoricalEstimator,
    IntegerHiddenAssociationDistribution,
    IntegerMarkovChainDistribution,
    IntegerMultinomialDistribution,
    IntegerProbabilisticLatentSemanticIndexingDistribution,
    IntegerStepBernoulliEditDistribution,
    IntegerStepBernoulliEditEstimator,
    IntegerUniformSpikeDistribution,
    JointMixtureDistribution,
    JointMixtureEstimator,
    LaplaceDistribution,
    LaplaceEstimator,
    LDADistribution,
    LogGaussianDistribution,
    LogGaussianEstimator,
    LogisticDistribution,
    LogisticEstimator,
    MarkovChainDistribution,
    MarkovChainEstimator,
    MixtureDistribution,
    MixtureEstimator,
    MultinomialDistribution,
    MultinomialEstimator,
    MultivariateGaussianDistribution,
    MultivariateGaussianEstimator,
    NegativeBinomialDistribution,
    NegativeBinomialEstimator,
    NullDistribution,
    OptionalDistribution,
    OptionalEstimator,
    ParetoDistribution,
    ParetoEstimator,
    PointMassDistribution,
    PoissonDistribution,
    PoissonEstimator,
    QuantizedHiddenMarkovModelDistribution,
    RayleighDistribution,
    RayleighEstimator,
    RecordDistribution,
    RecordEstimator,
    SegmentalHiddenMarkovEstimator,
    SegmentalHiddenMarkovModelDistribution,
    SelectDistribution,
    SelectEstimator,
    SemiSupervisedMixtureDistribution,
    SequenceDistribution,
    SequenceEstimator,
    SpearmanRankingDistribution,
    StudentTDistribution,
    StudentTEstimator,
    TransformDistribution,
    TransformEstimator,
    TreeHiddenMarkovModelDistribution,
    UniformDistribution,
    UniformEstimator,
    VonMisesFisherDistribution,
    WeibullDistribution,
    WeibullEstimator,
    WeightedDistribution,
    WeightedEstimator,
    capabilities_for,
    declaration_for,
    declaration_issues,
    declared_distribution_types,
    generated_log_density_diagnostics,
    generated_stacked_strategy,
    numpy_only_distribution_types,
    stacked_component_strategy,
    statistic_layout_issues,
    validate_declaration,
    validate_statistic_layout,
)
from mixle.stats.latent.semi_supervised_hidden_markov_model import SemiSupervisedHiddenMarkovModelDistribution


def _assert_suff_close(test_case, actual, expected):
    if isinstance(actual, dict):
        test_case.assertEqual(set(actual.keys()), set(expected.keys()))
        for key in actual:
            _assert_suff_close(test_case, actual[key], expected[key])
        return
    if isinstance(actual, (tuple, list)):
        test_case.assertEqual(len(actual), len(expected))
        for a, e in zip(actual, expected):
            _assert_suff_close(test_case, a, e)
        return
    if actual is None or expected is None:
        test_case.assertEqual(actual, expected)
        return
    if isinstance(actual, np.ndarray) or isinstance(expected, np.ndarray):
        np.testing.assert_allclose(
            np.asarray(actual, dtype=float), np.asarray(expected, dtype=float), rtol=1.0e-12, atol=1.0e-12
        )
        return
    if isinstance(actual, (str, bytes, bool)):
        test_case.assertEqual(actual, expected)
        return
    np.testing.assert_allclose(
        np.asarray(actual, dtype=float), np.asarray(expected, dtype=float), rtol=1.0e-12, atol=1.0e-12
    )


def _sign_choice(x):
    return 0 if float(x) < 0.0 else 1


def _assert_models_equivalent(test_case, actual, expected, data):
    test_case.assertEqual(type(actual), type(expected))
    actual_scores = actual.seq_log_density(actual.dist_to_encoder().seq_encode(data))
    expected_scores = expected.seq_log_density(expected.dist_to_encoder().seq_encode(data))
    np.testing.assert_allclose(
        np.asarray(actual_scores, dtype=float), np.asarray(expected_scores, dtype=float), rtol=1.0e-10, atol=1.0e-10
    )


class ComputeMetadataTestCase(unittest.TestCase):
    class FakeTorchEngine:
        name = "torch"

    def test_backend_scoring_family_capabilities_are_registered(self):
        caps = capabilities_for(GaussianDistribution)
        # Gaussian also declares the (verified) jax scoring engine; composition does NOT propagate it
        # (combinators/wrappers cap to COMPOSITION_ENGINES -- see the combinator/wrapper tests below).
        self.assertEqual(caps.engine_ready, ("numpy", "torch", "jax"))
        self.assertEqual(caps.kernel_status, "numba_adapter")
        self.assertFalse(caps.is_permanently_numpy_only)

    def test_leaf_only_jax_engine_does_not_propagate_through_composition(self):
        # A leaf may declare a scoring-only engine (jax) for direct fitting, but a model that *wraps* or
        # *combines* it must not claim jax unless its own kernel supports it. This guards the
        # delegated_engine_ready / intersect_engine_ready caps against a false-capability regression.
        self.assertIn("jax", capabilities_for(GaussianDistribution).engine_ready)
        combos = [
            CompositeDistribution((GaussianDistribution(0.0, 1.0), PoissonDistribution(2.0))),
            MixtureDistribution([GaussianDistribution(-1.0, 1.0), GaussianDistribution(1.0, 1.0)], [0.5, 0.5]),
            WeightedDistribution(GaussianDistribution(0.0, 1.0)),
            IgnoredDistribution(GaussianDistribution(0.0, 1.0)),
            TransformDistribution(GaussianDistribution(0.0, 1.0), transform=AffineTransform(loc=2.0, scale=3.0)),
        ]
        for d in combos:
            self.assertNotIn("jax", capabilities_for(d).engine_ready, f"{type(d).__name__} over-claims jax")

    def test_object_valued_categorical_supports_torch_lookup_scoring(self):
        caps = capabilities_for(CategoricalDistribution)
        self.assertEqual(caps.engine_ready, ("numpy", "torch"))
        self.assertEqual(caps.kernel_status, "numba_adapter")
        self.assertFalse(caps.is_permanently_numpy_only)

    def test_combinator_capabilities_intersect_child_engines(self):
        tensor_comp = CompositeDistribution(
            (
                GaussianDistribution(0.0, 1.0),
                PoissonDistribution(2.0),
            )
        )
        self.assertEqual(capabilities_for(tensor_comp).engine_ready, ("numpy", "torch"))

        object_comp = CompositeDistribution(
            (
                GaussianDistribution(0.0, 1.0),
                CategoricalDistribution({"a": 0.7, "b": 0.3}),
            )
        )
        self.assertEqual(capabilities_for(object_comp).engine_ready, ("numpy", "torch"))

        tensor_mix = MixtureDistribution(
            [GaussianDistribution(-1.0, 1.0), GaussianDistribution(1.0, 1.0)],
            [0.5, 0.5],
        )
        self.assertEqual(capabilities_for(tensor_mix).engine_ready, ("numpy", "torch"))

        record = RecordDistribution(
            {
                "x": GaussianDistribution(0.0, 1.0),
                "y": PoissonDistribution(2.0),
            }
        )
        self.assertEqual(capabilities_for(record).engine_ready, ("numpy", "torch"))

        sequence = SequenceDistribution(
            GaussianDistribution(0.0, 1.0),
            len_dist=IntegerCategoricalDistribution(0, [0.2, 0.3, 0.5]),
        )
        self.assertEqual(capabilities_for(sequence).engine_ready, ("numpy", "torch"))

        multinomial = MultinomialDistribution(
            CategoricalDistribution({"a": 0.6, "b": 0.4}),
            len_dist=IntegerCategoricalDistribution(0, [0.2, 0.3, 0.5]),
        )
        self.assertEqual(capabilities_for(multinomial).engine_ready, ("numpy", "torch"))
        self.assertEqual(capabilities_for(multinomial).kernel_status, "generic_table")

        transform = TransformDistribution(GaussianDistribution(0.0, 1.0), transform=AffineTransform(loc=2.0, scale=3.0))
        self.assertEqual(capabilities_for(transform).engine_ready, ("numpy", "torch"))

        select = SelectDistribution(
            [GaussianDistribution(-1.0, 0.5), GaussianDistribution(1.0, 0.7)],
            _sign_choice,
        )
        self.assertEqual(capabilities_for(select).engine_ready, ("numpy", "torch"))

        conditional = ConditionalDistribution(
            {"a": GaussianDistribution(-1.0, 0.5), "b": GaussianDistribution(1.0, 0.7)},
            default_dist=GaussianDistribution(0.0, 2.0),
            given_dist=CategoricalDistribution({"a": 0.4, "b": 0.5, "c": 0.1}),
        )
        self.assertEqual(capabilities_for(conditional).engine_ready, ("numpy", "torch"))

        markov = MarkovChainDistribution(
            {"a": 0.7, "b": 0.3},
            {"a": {"a": 0.2, "b": 0.8}, "b": {"a": 0.6, "b": 0.4}},
            len_dist=IntegerCategoricalDistribution(0, [0.1, 0.2, 0.4, 0.3]),
        )
        self.assertEqual(capabilities_for(markov).engine_ready, ("numpy", "torch"))

        hmm = HiddenMarkovModelDistribution(
            [GaussianDistribution(-1.0, 0.8), GaussianDistribution(2.0, 1.5)],
            [0.6, 0.4],
            [[0.75, 0.25], [0.20, 0.80]],
            len_dist=IntegerCategoricalDistribution(0, [0.1, 0.2, 0.4, 0.3]),
            use_numba=False,
        )
        self.assertEqual(capabilities_for(hmm).engine_ready, ("numpy", "torch"))
        self.assertEqual(capabilities_for(hmm).kernel_status, "generic_latent")

        numba_hmm = HiddenMarkovModelDistribution(
            [GaussianDistribution(-1.0, 0.8), GaussianDistribution(2.0, 1.5)],
            [0.6, 0.4],
            [[0.75, 0.25], [0.20, 0.80]],
            len_dist=IntegerCategoricalDistribution(0, [0.1, 0.2, 0.4, 0.3]),
            use_numba=True,
        )
        # use_numba selects the tuned numpy HOST kernel but no longer gates torch: the engine consumes
        # the numba encoding for both scoring and the E-step, so the default HMM is engine-ready too.
        self.assertEqual(capabilities_for(numba_hmm).engine_ready, ("numpy", "torch"))
        self.assertEqual(capabilities_for(numba_hmm).kernel_status, "generic_latent")

        joint_mix = JointMixtureDistribution(
            components1=[GaussianDistribution(-1.0, 0.8), GaussianDistribution(2.0, 1.5)],
            components2=[PoissonDistribution(1.5), PoissonDistribution(4.0)],
            w1=[0.4, 0.6],
            w2=[0.5, 0.5],
            taus12=[[0.8, 0.2], [0.25, 0.75]],
            taus21=[[0.64, 0.18181818181818182], [0.36, 0.8181818181818182]],
        )
        self.assertEqual(capabilities_for(joint_mix).engine_ready, ("numpy", "torch"))
        self.assertEqual(capabilities_for(joint_mix).kernel_status, "generic_latent")

        hetero_mix = HeterogeneousMixtureDistribution(
            [ExponentialDistribution(1.5), GammaDistribution(2.0, 0.75)], [0.35, 0.65]
        )
        self.assertEqual(capabilities_for(hetero_mix).engine_ready, ("numpy", "torch"))
        self.assertEqual(capabilities_for(hetero_mix).kernel_status, "generic_latent")

        semi_mix = SemiSupervisedMixtureDistribution(
            [GaussianDistribution(-1.0, 0.8), GaussianDistribution(2.0, 1.5)], [0.4, 0.6]
        )
        self.assertEqual(capabilities_for(semi_mix).engine_ready, ("numpy", "torch"))
        self.assertEqual(capabilities_for(semi_mix).kernel_status, "generic_latent")

        hier_mix = HierarchicalMixtureDistribution(
            topics=[GaussianDistribution(-1.0, 0.8), GaussianDistribution(2.0, 1.5)],
            mixture_weights=[0.45, 0.55],
            topic_weights=[[0.8, 0.2], [0.25, 0.75]],
            len_dist=IntegerCategoricalDistribution(0, [0.1, 0.2, 0.4, 0.3]),
        )
        self.assertEqual(capabilities_for(hier_mix).engine_ready, ("numpy", "torch"))
        self.assertEqual(capabilities_for(hier_mix).kernel_status, "generic_latent")

        dirac_len = DiracLengthMixtureDistribution(
            len_dist=IntegerCategoricalDistribution(0, [0.2, 0.5, 0.3]), p=0.7, v=0
        )
        self.assertEqual(capabilities_for(dirac_len).engine_ready, ("numpy", "torch"))
        self.assertEqual(capabilities_for(dirac_len).kernel_status, "generic_latent")

        segmental = SegmentalHiddenMarkovModelDistribution(
            [GaussianDistribution(-1.0, 0.8), GaussianDistribution(2.0, 1.5)],
            [0.6, 0.4],
            [[0.75, 0.25], [0.20, 0.80]],
            len_dist=IntegerCategoricalDistribution(0, [0.1, 0.2, 0.4, 0.3]),
        )
        self.assertEqual(capabilities_for(segmental).engine_ready, ("numpy", "torch"))
        self.assertEqual(capabilities_for(segmental).kernel_status, "generic_latent")

        tree_hmm = TreeHiddenMarkovModelDistribution(
            [GaussianDistribution(-1.0, 0.8), GaussianDistribution(2.0, 1.5)],
            [0.6, 0.4],
            [[0.75, 0.25], [0.20, 0.80]],
            len_dist=IntegerCategoricalDistribution(0, [0.1, 0.2, 0.4]),
            terminal_level=4,
            use_numba=False,
        )
        self.assertEqual(capabilities_for(tree_hmm).engine_ready, ("numpy", "torch"))
        self.assertEqual(capabilities_for(tree_hmm).kernel_status, "generic_latent")

        semi_sup_hmm = SemiSupervisedHiddenMarkovModelDistribution(
            [GaussianDistribution(-1.0, 0.8), GaussianDistribution(2.0, 1.5)],
            [[0.75, 0.25], [0.20, 0.80]],
            len_dist=IntegerCategoricalDistribution(0, [0.1, 0.2, 0.4, 0.3]),
        )
        # the semi-supervised HMM (per-observation state priors) is numpy-only
        self.assertEqual(capabilities_for(semi_sup_hmm).engine_ready, ("numpy",))
        self.assertEqual(capabilities_for(semi_sup_hmm).kernel_status, "legacy_numpy")

        quantized_hmm = QuantizedHiddenMarkovModelDistribution(
            theta=0.5,
            levels=["a", "b"],
            transition_exponents=[[0, 2], [1, 0]],
            emission_exponents=[[0, 1], [2, 0]],
            initial_exponents=[0, 1],
            len_dist=IntegerCategoricalDistribution(0, [0.1, 0.2, 0.4, 0.3]),
            use_numba=False,
        )
        self.assertEqual(capabilities_for(quantized_hmm).engine_ready, ("numpy", "torch"))
        self.assertEqual(capabilities_for(quantized_hmm).kernel_status, "generic_latent")

        self.assertEqual(capabilities_for(IntegerUniformSpikeDistribution).engine_ready, ("numpy", "torch"))
        self.assertEqual(capabilities_for(IntegerUniformSpikeDistribution).kernel_status, "generic_table")
        self.assertEqual(capabilities_for(IntegerBernoulliSetDistribution).engine_ready, ("numpy", "torch"))
        self.assertEqual(capabilities_for(IntegerBernoulliSetDistribution).kernel_status, "generic_table")
        self.assertEqual(capabilities_for(BernoulliSetDistribution).engine_ready, ("numpy", "torch"))
        self.assertEqual(capabilities_for(BernoulliSetDistribution).kernel_status, "generic_table")
        self.assertEqual(capabilities_for(IndianBuffetProcessDistribution).engine_ready, ("numpy", "torch"))
        self.assertEqual(capabilities_for(IndianBuffetProcessDistribution).kernel_status, "generic_table")
        int_multinomial = IntegerMultinomialDistribution(
            1, [0.50, 0.30, 0.20], len_dist=IntegerCategoricalDistribution(0, [0.05, 0.10, 0.30, 0.35, 0.20])
        )
        self.assertEqual(capabilities_for(int_multinomial).engine_ready, ("numpy", "torch"))
        self.assertEqual(capabilities_for(int_multinomial).kernel_status, "generic_table")
        int_markov = IntegerMarkovChainDistribution(
            3,
            [[0.70, 0.20, 0.10], [0.10, 0.60, 0.30], [0.25, 0.25, 0.50]],
            lag=1,
            init_dist=SequenceDistribution(
                IntegerCategoricalDistribution(0, [0.25, 0.45, 0.30]), len_dist=IntegerCategoricalDistribution(1, [1.0])
            ),
            len_dist=IntegerCategoricalDistribution(0, [0.10, 0.20, 0.30, 0.40]),
        )
        self.assertEqual(capabilities_for(int_markov).engine_ready, ("numpy", "torch"))
        self.assertEqual(capabilities_for(int_markov).kernel_status, "generic_table")

        plsi = IntegerProbabilisticLatentSemanticIndexingDistribution(
            [[0.70, 0.10], [0.20, 0.30], [0.10, 0.60]],
            [[0.80, 0.20], [0.25, 0.75]],
            [0.55, 0.45],
            len_dist=IntegerCategoricalDistribution(0, [0.05, 0.15, 0.30, 0.30, 0.20]),
        )
        self.assertEqual(capabilities_for(plsi).engine_ready, ("numpy", "torch"))
        self.assertEqual(capabilities_for(plsi).kernel_status, "generic_latent")

        int_assoc = IntegerHiddenAssociationDistribution(
            state_prob_mat=[[0.70, 0.20, 0.10], [0.10, 0.30, 0.60]],
            cond_weights=[[0.80, 0.20], [0.30, 0.70], [0.50, 0.50]],
            alpha=0.15,
            len_dist=IntegerCategoricalDistribution(0, [0.10, 0.20, 0.30, 0.40]),
            use_numba=False,
        )
        self.assertEqual(capabilities_for(int_assoc).engine_ready, ("numpy", "torch"))
        self.assertEqual(capabilities_for(int_assoc).kernel_status, "generic_latent")

        hidden_assoc = HiddenAssociationDistribution(
            cond_dist=ConditionalDistribution(
                {
                    "a": CategoricalDistribution({"x": 0.80, "y": 0.20}),
                    "b": CategoricalDistribution({"x": 0.25, "y": 0.75}),
                }
            ),
            len_dist=CategoricalDistribution({0.0: 0.10, 2.0: 0.30, 3.0: 0.60}),
        )
        self.assertEqual(capabilities_for(hidden_assoc).engine_ready, ("numpy", "torch"))
        self.assertEqual(capabilities_for(hidden_assoc).kernel_status, "generic_latent")

        lda = LDADistribution(
            [
                IntegerCategoricalDistribution(0, [0.70, 0.20, 0.10]),
                IntegerCategoricalDistribution(0, [0.10, 0.30, 0.60]),
            ],
            [0.8, 1.3],
            len_dist=IntegerCategoricalDistribution(0, [0.05, 0.15, 0.30, 0.30, 0.20]),
            gamma_threshold=1.0e-10,
        )
        self.assertEqual(capabilities_for(lda).engine_ready, ("numpy", "torch"))
        self.assertEqual(capabilities_for(lda).kernel_status, "generic_latent")

        vmf = VonMisesFisherDistribution([1.0, 0.0, 0.0], 2.0)
        self.assertEqual(capabilities_for(vmf).engine_ready, ("numpy", "torch"))
        self.assertEqual(capabilities_for(vmf).kernel_status, "generic")

        spearman = SpearmanRankingDistribution([0, 1, 2], rho=0.8)
        self.assertEqual(capabilities_for(spearman).engine_ready, ("numpy", "torch"))
        self.assertEqual(capabilities_for(spearman).kernel_status, "generic")

        edit = IntegerBernoulliEditDistribution(
            np.log([[0.20, 0.75], [0.45, 0.30], [0.65, 0.55]]),
            init_dist=IntegerBernoulliSetDistribution(np.log([0.30, 0.55, 0.75])),
        )
        self.assertEqual(capabilities_for(edit).engine_ready, ("numpy", "torch"))
        self.assertEqual(capabilities_for(edit).kernel_status, "generic_table")

        step_edit = IntegerStepBernoulliEditDistribution(
            np.log([[0.20, 0.75], [0.45, 0.30], [0.65, 0.55]]),
            init_dist=IntegerBernoulliSetDistribution(np.log([0.30, 0.55, 0.75])),
        )
        self.assertEqual(capabilities_for(step_edit).engine_ready, ("numpy", "torch"))
        self.assertEqual(capabilities_for(step_edit).kernel_status, "generic_table")

    def test_permanent_numpy_only_capabilities_are_registered(self):
        # No distribution remains permanently NumPy-only: HeterogeneousPCFG (CKY inside) and
        # SparseMarkovAssociation (sparse-gather + engine reductions) were both routed through
        # ComputeEngine ops (numpy + torch).
        self.assertEqual(set(numpy_only_distribution_types()), set())

    def test_latent_family_capabilities_are_legacy_numpy_not_permanent(self):
        for dist_type in (
            HiddenMarkovModelDistribution,
            JointMixtureDistribution,
            HierarchicalMixtureDistribution,
            SegmentalHiddenMarkovModelDistribution,
        ):
            with self.subTest(dist=dist_type.__name__):
                caps = capabilities_for(dist_type)
                self.assertEqual(caps.engine_ready, ("numpy",))
                self.assertEqual(caps.kernel_status, "legacy_numpy")
                self.assertFalse(caps.is_permanently_numpy_only)
                self.assertIsNone(caps.numpy_only_reason)

    def test_tensor_ready_family_accepts_generic_torch_kernel(self):
        dist = DirichletDistribution([1.0, 2.0, 3.0])
        kernel = dist.kernel(engine=self.FakeTorchEngine())
        self.assertEqual(type(kernel).__name__, "GenericKernel")

    def test_high_level_compute_utilities_do_not_import_concrete_distributions(self):
        pysp_root = Path(__file__).resolve().parents[1]
        targets = [
            pysp_root / "stats" / "compute" / "torch_mixture.py",
            # the fitting machinery now lives in mixle.inference (was mixle.utils)
            pysp_root / "inference" / "estimation.py",
            pysp_root / "inference" / "objectives.py",
            pysp_root / "inference" / "em.py",
            pysp_root / "inference" / "fisher.py",
            pysp_root / "utils" / "automatic" / "factories.py",
            pysp_root / "utils" / "automatic" / "profiling.py",
            pysp_root / "utils" / "parallel" / "planner.py",  # moved here with the parallel backends it serves
        ] + sorted((pysp_root / "engines").glob("*.py"))
        allowed_modules = {
            "mixle.stats",
            "mixle.stats.compute.backend",
            "mixle.stats.compute.capabilities",
            "mixle.stats.compute.declarations",
            "mixle.stats.compute.gradient",
            "mixle.stats.compute.pdist",
            # the vectorized seq_* drivers — pure pdist-protocol dispatch, not concrete distributions
            "mixle.stats.compute.sequence",
            # SQUAREM's parameter packing: it exists PRECISELY to keep concrete-family knowledge out of
            # inference/em.py -- its own module level imports only numpy (the family handlers import
            # lazily inside the builder), and em.py imports just the packer function.
            "mixle.stats.parameter_packing",
        }
        concrete_suffixes = (
            "Accumulator",
            "AccumulatorFactory",
            "DataEncoder",
            "Distribution",
            "Estimator",
            "Sampler",
        )

        violations = []
        for path in targets:
            tree = ast.parse(path.read_text(), filename=str(path))
            for node in ast.walk(tree):
                if isinstance(node, ast.Import):
                    for alias in node.names:
                        name = alias.name
                        if name.startswith("mixle.stats.") and name not in allowed_modules:
                            violations.append("%s imports %s" % (path.relative_to(pysp_root), name))
                elif isinstance(node, ast.ImportFrom) and node.level == 0:
                    module = node.module or ""
                    if module == "mixle.stats":
                        for alias in node.names:
                            if alias.name.endswith(concrete_suffixes):
                                violations.append(
                                    "%s imports concrete mixle.stats.%s" % (path.relative_to(pysp_root), alias.name)
                                )
                    elif module.startswith("mixle.stats.") and module not in allowed_modules:
                        violations.append("%s imports %s" % (path.relative_to(pysp_root), module))
        self.assertEqual(violations, [])

    def test_torch_mixture_remains_a_compatibility_shim_not_an_omni_file(self):
        torch_mixture_path = Path(__file__).resolve().parents[1] / "stats" / "compute" / "torch_mixture.py"
        tree = ast.parse(torch_mixture_path.read_text(), filename=str(torch_mixture_path))
        class_names = [node.name for node in tree.body if isinstance(node, ast.ClassDef)]
        exports = []
        for node in tree.body:
            if isinstance(node, ast.Assign):
                for target in node.targets:
                    if isinstance(target, ast.Name) and target.id == "__all__" and isinstance(node.value, ast.List):
                        exports = [
                            item.value
                            for item in node.value.elts
                            if isinstance(item, ast.Constant) and isinstance(item.value, str)
                        ]
        self.assertEqual(class_names, ["TorchMixture"])
        self.assertEqual(exports, ["TorchMixture"])

    def test_leaf_declarations_are_registered(self):
        expected = {
            GaussianDistribution,
            PoissonDistribution,
            ExponentialDistribution,
            BernoulliDistribution,
            CategoricalDistribution,
            GammaDistribution,
            LogGaussianDistribution,
            BinomialDistribution,
            NegativeBinomialDistribution,
            GeometricDistribution,
            DiagonalGaussianDistribution,
            StudentTDistribution,
            LogisticDistribution,
            WeibullDistribution,
            RayleighDistribution,
            ParetoDistribution,
            UniformDistribution,
            IntegerCategoricalDistribution,
            BetaDistribution,
            DirichletDistribution,
            LaplaceDistribution,
            MultivariateGaussianDistribution,
            NullDistribution,
            PointMassDistribution,
            BernoulliSetDistribution,
            IndianBuffetProcessDistribution,
            IntegerUniformSpikeDistribution,
            IntegerBernoulliSetDistribution,
            SpearmanRankingDistribution,
            VonMisesFisherDistribution,
        }
        self.assertTrue(expected.issubset(set(declared_distribution_types())))

    def test_registered_declarations_pass_generic_schema_validation(self):
        for dist_type in declared_distribution_types():
            with self.subTest(dist=dist_type.__name__):
                self.assertEqual(declaration_issues(dist_type), ())
                self.assertEqual(validate_declaration(dist_type).name, declaration_for(dist_type).name)

    def test_instance_declarations_pass_generic_schema_validation(self):
        instances = [
            RecordDistribution(
                {
                    "x": GaussianDistribution(0.0, 1.0),
                    "y": PoissonDistribution(2.0),
                }
            ),
            MixtureDistribution([GaussianDistribution(-1.0, 1.0), GaussianDistribution(1.0, 1.0)], [0.4, 0.6]),
            MarkovChainDistribution(
                {"a": 0.7, "b": 0.3},
                {"a": {"a": 0.2, "b": 0.8}, "b": {"a": 0.6, "b": 0.4}},
                len_dist=IntegerCategoricalDistribution(0, [0.2, 0.8]),
            ),
            HiddenMarkovModelDistribution(
                [GaussianDistribution(-1.0, 1.0), GaussianDistribution(1.0, 1.0)],
                [0.5, 0.5],
                [[0.8, 0.2], [0.3, 0.7]],
                len_dist=IntegerCategoricalDistribution(0, [0.1, 0.9]),
                use_numba=False,
            ),
        ]

        for dist in instances:
            with self.subTest(dist=type(dist).__name__):
                self.assertEqual(declaration_issues(dist), ())
                self.assertEqual(validate_declaration(dist).name, declaration_for(dist).name)

    def test_declaration_validation_reports_structural_errors(self):
        from mixle.stats import DistributionDeclaration, ParameterSpec, StatisticSpec

        bad = DistributionDeclaration(
            name="bad",
            distribution_type=GaussianDistribution,
            parameters=(
                ParameterSpec("high", constraint="greater_than:low"),
                ParameterSpec("high", constraint="not_a_constraint"),
            ),
            statistics=(StatisticSpec("count"), StatisticSpec("count")),
            support="real",
            children=(None,),
            child_roles=("child", "extra"),
        )

        issues = declaration_issues(bad)
        self.assertTrue(any("anchor low" in issue for issue in issues))
        self.assertTrue(any("unknown constraint not_a_constraint" in issue for issue in issues))
        self.assertTrue(any("duplicate parameter high" in issue for issue in issues))
        self.assertTrue(any("duplicate statistic count" in issue for issue in issues))
        self.assertTrue(any("child roles" in issue for issue in issues))
        with self.assertRaises(ValueError):
            validate_declaration(bad)

    def test_gaussian_declaration_extracts_parameters(self):
        dist = GaussianDistribution(1.5, 2.5)
        declaration = declaration_for(dist)
        self.assertEqual(declaration.name, "gaussian")
        self.assertEqual(declaration.parameter_names, ("mu", "sigma2"))
        self.assertEqual(declaration.statistic_names, ("sum", "sum2", "count", "count2"))
        self.assertEqual(declaration.support, "real")
        self.assertTrue(declaration.has_exponential_family)
        self.assertEqual(declaration.parameter_values(dist), {"mu": 1.5, "sigma2": 2.5})

    def test_gaussian_declaration_maps_accumulator_value(self):
        dist = GaussianDistribution(1.0, 2.0)
        enc = dist.dist_to_encoder().seq_encode(np.asarray([-1.0, 0.0, 2.0]))
        weights = np.asarray([0.5, 1.0, 1.5])
        acc = GaussianEstimator().accumulator_factory().make()
        acc.seq_update(enc, weights, dist)

        declaration = declaration_for(dist)
        stats = declaration.statistic_values(acc.value())
        self.assertEqual(tuple(stats.keys()), declaration.statistic_names)
        self.assertAlmostEqual(stats["sum"], float(np.dot(enc, weights)))
        self.assertAlmostEqual(stats["count"], float(weights.sum()))

    def test_discrete_declarations_capture_constraints(self):
        poisson = declaration_for(PoissonDistribution(3.0))
        self.assertEqual(poisson.parameter_names, ("lam",))
        self.assertEqual(poisson.parameters[0].constraint, "positive")
        self.assertTrue(poisson.has_exponential_family)
        self.assertIsNotNone(poisson.exponential_family.base_measure)

        exponential = declaration_for(ExponentialDistribution(2.0))
        self.assertEqual(exponential.parameter_names, ("beta",))
        self.assertEqual(exponential.parameters[0].constraint, "positive")
        self.assertTrue(exponential.has_exponential_family)
        self.assertIsNotNone(exponential.exponential_family.base_measure)

        gamma = declaration_for(GammaDistribution(2.0, 1.5))
        self.assertEqual(gamma.parameter_names, ("k", "theta"))
        self.assertTrue(gamma.has_exponential_family)
        self.assertIsNotNone(gamma.exponential_family.base_measure)

        log_gaussian = declaration_for(LogGaussianDistribution(0.0, 1.0))
        self.assertEqual(log_gaussian.parameter_names, ("mu", "sigma2"))
        self.assertTrue(log_gaussian.has_exponential_family)
        self.assertIsNotNone(log_gaussian.exponential_family.base_measure)

        rayleigh = declaration_for(RayleighDistribution(1.2))
        self.assertEqual(rayleigh.parameter_names, ("sigma",))
        self.assertTrue(rayleigh.has_exponential_family)
        self.assertIsNotNone(rayleigh.exponential_family.base_measure)

        beta = declaration_for(BetaDistribution(2.0, 5.0))
        self.assertEqual(beta.parameter_names, ("a", "b"))
        self.assertTrue(beta.has_exponential_family)

        binomial = declaration_for(BinomialDistribution(0.4, 5, min_val=2))
        self.assertEqual(binomial.parameter_names, ("p", "n", "min_val"))
        self.assertFalse(binomial.parameters[1].differentiable)
        self.assertFalse(binomial.parameters[2].differentiable)
        self.assertTrue(binomial.has_exponential_family)
        self.assertIsNotNone(binomial.exponential_family.base_measure_from_params)

        bernoulli = declaration_for(BernoulliDistribution(0.4))
        self.assertEqual(bernoulli.parameter_names, ("p",))
        self.assertEqual(bernoulli.parameters[0].constraint, "unit_interval")
        self.assertTrue(bernoulli.has_exponential_family)

        categorical = declaration_for(CategoricalDistribution({"a": 0.7, "b": 0.3}))
        self.assertTrue(categorical.differentiable)
        self.assertEqual(categorical.parameters[0].constraint, "simplex_map")

        full_mvn = declaration_for(MultivariateGaussianDistribution([0.0, 1.0], [[1.0, 0.2], [0.2, 2.0]]))
        self.assertFalse(full_mvn.differentiable)
        self.assertEqual(full_mvn.parameters[1].constraint, "positive_matrix")

    def test_generated_stacked_strategy_reports_dispatch_source(self):
        self.assertEqual(generated_stacked_strategy(GaussianDistribution), "exp_family")
        self.assertEqual(generated_stacked_strategy(BinomialDistribution), "exp_family")
        self.assertEqual(generated_stacked_strategy(GeometricDistribution), "exp_family")
        self.assertEqual(generated_stacked_strategy(DiagonalGaussianDistribution), "exp_family")
        self.assertEqual(generated_stacked_strategy(IntegerCategoricalDistribution), "none")
        self.assertEqual(generated_stacked_strategy(MultivariateGaussianDistribution), "exp_family")
        self.assertEqual(generated_stacked_strategy(CategoricalDistribution), "none")

        self.assertEqual(stacked_component_strategy(GaussianDistribution), "generated_exp_family")
        self.assertEqual(stacked_component_strategy(BinomialDistribution), "generated_exp_family")
        self.assertEqual(stacked_component_strategy(GeometricDistribution), "generated_exp_family")
        self.assertEqual(stacked_component_strategy(LaplaceDistribution), "generated_backend_hook")
        self.assertEqual(stacked_component_strategy(UniformDistribution), "generated_backend_hook")
        self.assertEqual(stacked_component_strategy(DiagonalGaussianDistribution), "generated_exp_family")
        self.assertEqual(stacked_component_strategy(MultivariateGaussianDistribution), "generated_exp_family")
        self.assertEqual(stacked_component_strategy(IntegerCategoricalDistribution), "explicit_stacked")
        self.assertEqual(stacked_component_strategy(CategoricalDistribution), "explicit_stacked")
        self.assertEqual(stacked_component_strategy(NullDistribution), "explicit_stacked")
        self.assertEqual(stacked_component_strategy(DirichletDistribution), "explicit_stacked")
        self.assertEqual(stacked_component_strategy(IgnoredDistribution), "explicit_stacked")
        self.assertEqual(stacked_component_strategy(RecordDistribution), "explicit_stacked")
        self.assertEqual(stacked_component_strategy(SelectDistribution), "explicit_stacked")
        self.assertEqual(stacked_component_strategy(SequenceDistribution), "explicit_stacked")
        self.assertEqual(stacked_component_strategy(ConditionalDistribution), "explicit_stacked")
        self.assertEqual(stacked_component_strategy(MultinomialDistribution), "explicit_stacked")
        self.assertEqual(stacked_component_strategy(IntegerMultinomialDistribution), "explicit_stacked")
        self.assertEqual(stacked_component_strategy(PointMassDistribution), "explicit_stacked")
        self.assertEqual(stacked_component_strategy(IntegerUniformSpikeDistribution), "explicit_stacked")
        self.assertEqual(stacked_component_strategy(IntegerBernoulliSetDistribution), "explicit_stacked")
        self.assertEqual(stacked_component_strategy(BernoulliSetDistribution), "explicit_stacked")
        self.assertEqual(stacked_component_strategy(IndianBuffetProcessDistribution), "explicit_stacked")
        self.assertEqual(stacked_component_strategy(IntegerBernoulliEditDistribution), "explicit_stacked")
        self.assertEqual(stacked_component_strategy(IntegerStepBernoulliEditDistribution), "explicit_stacked")
        self.assertEqual(stacked_component_strategy(DiracLengthMixtureDistribution), "explicit_stacked")
        self.assertEqual(stacked_component_strategy(MarkovChainDistribution), "explicit_stacked")
        self.assertEqual(stacked_component_strategy(IntegerMarkovChainDistribution), "explicit_stacked")
        self.assertEqual(stacked_component_strategy(SpearmanRankingDistribution), "generated_backend_hook")
        self.assertEqual(stacked_component_strategy(VonMisesFisherDistribution), "generated_backend_hook")

    def test_generated_log_density_diagnostics_trace_symbolic_formulas(self):
        gaussian = generated_log_density_diagnostics(GaussianDistribution)
        self.assertEqual(gaussian["strategy"], "exp_family")
        self.assertEqual(gaussian["encoded_symbols"], ("x",))
        self.assertEqual(gaussian["parameter_symbols"], ("mu", "sigma2"))
        self.assertEqual(gaussian["symbols"], ("mu", "sigma2", "x"))
        self.assertEqual(gaussian["op_counts"]["log"], 1)
        self.assertIn("sigma2", gaussian["expression"])

        poisson = generated_log_density_diagnostics(PoissonDistribution)
        self.assertEqual(poisson["strategy"], "exp_family")
        self.assertEqual(poisson["encoded_symbols"], ("vals", "log_fact"))
        self.assertEqual(poisson["parameter_symbols"], ("lam",))
        self.assertEqual(poisson["symbols"], ("lam", "log_fact", "vals"))
        self.assertEqual(poisson["op_counts"]["where"], 1)
        self.assertEqual(poisson["op_counts"]["ge"], 1)
        self.assertEqual(poisson["op_counts"]["eq"], 1)

        beta = generated_log_density_diagnostics(
            BetaDistribution,
            encoded_symbols=("log_x", "log1m_x", "sum", "sum2"),
        )
        self.assertEqual(beta["strategy"], "exp_family")
        self.assertEqual(beta["parameter_symbols"], ("a", "b"))
        self.assertEqual(beta["symbols"], ("a", "b", "log1m_x", "log_x"))
        self.assertEqual(beta["op_counts"]["betaln"], 1)

        diagonal = generated_log_density_diagnostics(DiagonalGaussianDistribution)
        self.assertEqual(diagonal["strategy"], "backend_log_density_from_params")
        self.assertEqual(diagonal["encoded_symbols"], ("x",))
        self.assertEqual(diagonal["parameter_symbols"], ("mu", "covar"))
        self.assertEqual(diagonal["symbols"], ("covar_0", "covar_1", "mu_0", "mu_1", "x_0", "x_1"))
        self.assertEqual(diagonal["op_counts"]["log"], 1)

    def test_record_declaration_uses_named_child_roles(self):
        dist = RecordDistribution(
            {
                "x": GaussianDistribution(0.0, 1.0),
                "y": PoissonDistribution(2.0),
            }
        )
        declaration = declaration_for(dist)
        self.assertEqual(declaration.name, "record")
        self.assertEqual(declaration.child_roles, ("x", "y"))

    def test_wrapper_declarations_delegate_to_children(self):
        weighted = declaration_for(WeightedDistribution(GaussianDistribution(0.0, 1.0)))
        self.assertEqual(weighted.name, "weighted")
        self.assertEqual(weighted.child_roles, ("value",))
        self.assertEqual(tuple(child.name for child in weighted.children), ("gaussian",))
        self.assertFalse(weighted.differentiable)

        ignored = declaration_for(IgnoredDistribution(PoissonDistribution(2.0)))
        self.assertEqual(ignored.name, "ignored")
        self.assertEqual(ignored.child_roles, ("ignored",))
        self.assertEqual(tuple(child.name for child in ignored.children), ("poisson",))
        self.assertFalse(ignored.differentiable)

        transform = declaration_for(
            TransformDistribution(GaussianDistribution(0.0, 1.0), transform=AffineTransform(loc=2.0, scale=3.0))
        )
        self.assertEqual(transform.name, "transform")
        self.assertEqual(transform.child_roles, ("base",))
        self.assertEqual(tuple(child.name for child in transform.children), ("gaussian",))
        self.assertTrue(transform.differentiable)

        select = declaration_for(
            SelectDistribution(
                [GaussianDistribution(-1.0, 0.5), GaussianDistribution(1.0, 0.7)],
                _sign_choice,
            )
        )
        self.assertEqual(select.name, "select")
        self.assertEqual(select.child_roles, ("choice_0", "choice_1"))
        self.assertTrue(select.differentiable)

        conditional = declaration_for(
            ConditionalDistribution(
                {"a": GaussianDistribution(-1.0, 0.5), "b": GaussianDistribution(1.0, 0.7)},
                default_dist=GaussianDistribution(0.0, 2.0),
                given_dist=CategoricalDistribution({"a": 0.4, "b": 0.5, "c": 0.1}),
            )
        )
        self.assertEqual(conditional.name, "conditional")
        self.assertEqual(conditional.child_roles, ("condition_'a'", "condition_'b'", "default", "given"))
        self.assertTrue(conditional.differentiable)

        markov = declaration_for(
            MarkovChainDistribution(
                {"a": 0.7, "b": 0.3},
                {"a": {"a": 0.2, "b": 0.8}, "b": {"a": 0.6, "b": 0.4}},
                len_dist=IntegerCategoricalDistribution(0, [0.1, 0.2, 0.4, 0.3]),
            )
        )
        self.assertEqual(markov.name, "markov_chain")
        self.assertEqual(markov.child_roles, ("length",))
        self.assertTrue(markov.differentiable)

        null = declaration_for(NullDistribution())
        self.assertEqual(null.name, "null")
        self.assertEqual(null.parameter_names, ())
        self.assertFalse(null.differentiable)

        point = declaration_for(PointMassDistribution("fixed"))
        self.assertEqual(point.name, "point_mass")
        self.assertEqual(point.parameter_names, ("value",))
        self.assertFalse(point.differentiable)

        self.assertEqual(
            capabilities_for(WeightedDistribution(GaussianDistribution(0.0, 1.0))).engine_ready, ("numpy", "torch")
        )
        self.assertEqual(
            capabilities_for(IgnoredDistribution(GaussianDistribution(0.0, 1.0))).engine_ready, ("numpy", "torch")
        )

    def test_count_table_declarations_describe_payloads_and_children(self):
        multinomial = MultinomialDistribution(
            CategoricalDistribution({"a": 0.6, "b": 0.4}),
            len_dist=IntegerCategoricalDistribution(0, [0.2, 0.3, 0.5]),
        )
        m_decl = declaration_for(multinomial)
        self.assertEqual(m_decl.name, "multinomial")
        self.assertEqual(m_decl.child_roles, ("value", "length"))
        self.assertEqual(m_decl.statistic_names, ("values", "length"))
        self.assertFalse(m_decl.differentiable)

        int_multinomial = IntegerMultinomialDistribution(
            1, [0.5, 0.3, 0.2], len_dist=IntegerCategoricalDistribution(0, [0.2, 0.3, 0.5])
        )
        im_decl = declaration_for(int_multinomial)
        self.assertEqual(im_decl.name, "integer_multinomial")
        self.assertEqual(im_decl.parameter_names, ("min_val", "p_vec"))
        self.assertEqual(im_decl.statistic_names, ("min_val", "count_vec", "length"))
        self.assertEqual(im_decl.child_roles, ("length",))
        self.assertFalse(im_decl.differentiable)

        bern_set = declaration_for(BernoulliSetDistribution({"a": 0.7, "b": 0.2}, min_prob=0.0))
        self.assertEqual(bern_set.name, "bernoulli_set")
        self.assertEqual(bern_set.parameter_names, ("pmap", "min_prob"))
        self.assertEqual(bern_set.statistic_names, ("inclusion_counts", "total_weight"))
        self.assertFalse(bern_set.differentiable)

        ibp = declaration_for(IndianBuffetProcessDistribution(3, alpha=1.2, feature_probs=[0.2, 0.5, 0.8]))
        self.assertEqual(ibp.name, "indian_buffet_process")
        self.assertEqual(ibp.parameter_names, ("num_features", "alpha", "beta_params"))
        self.assertEqual(ibp.statistic_names, ("feature_counts", "total_count", "alpha"))
        self.assertFalse(ibp.differentiable)

        int_spike = declaration_for(IntegerUniformSpikeDistribution(k=2, num_vals=5, p=0.35, min_val=0))
        self.assertEqual(int_spike.name, "integer_uniform_spike")
        self.assertEqual(int_spike.parameter_names, ("k", "num_vals", "p", "min_val"))
        self.assertEqual(int_spike.statistic_names, ("min_val", "count_vec"))
        self.assertFalse(int_spike.differentiable)

        int_set = declaration_for(IntegerBernoulliSetDistribution(np.log([0.2, 0.6, 0.8])))
        self.assertEqual(int_set.name, "integer_bernoulli_set")
        self.assertEqual(int_set.parameter_names, ("log_pvec", "log_nvec"))
        self.assertEqual(int_set.statistic_names, ("inclusion_counts", "total_weight"))
        self.assertFalse(int_set.differentiable)

        int_markov = IntegerMarkovChainDistribution(
            3,
            [[0.70, 0.20, 0.10], [0.10, 0.60, 0.30], [0.25, 0.25, 0.50]],
            lag=1,
            init_dist=SequenceDistribution(
                IntegerCategoricalDistribution(0, [0.25, 0.45, 0.30]), len_dist=IntegerCategoricalDistribution(1, [1.0])
            ),
            len_dist=IntegerCategoricalDistribution(0, [0.10, 0.20, 0.30, 0.40]),
        )
        imc_decl = declaration_for(int_markov)
        self.assertEqual(imc_decl.name, "integer_markov_chain")
        self.assertEqual(imc_decl.parameter_names, ("num_values", "cond_dist", "lag"))
        self.assertEqual(imc_decl.statistic_names, ("transition_counts", "initial", "length"))
        self.assertEqual(imc_decl.child_roles, ("initial", "length"))
        self.assertFalse(imc_decl.differentiable)

    def test_latent_mixture_declarations_describe_payloads_and_children(self):
        hetero = HeterogeneousMixtureDistribution(
            [ExponentialDistribution(1.5), GammaDistribution(2.0, 0.75)], [0.35, 0.65]
        )
        hetero_decl = declaration_for(hetero)
        self.assertEqual(hetero_decl.name, "heterogeneous_mixture")
        self.assertEqual(hetero_decl.parameter_names, ("w",))
        self.assertEqual(hetero_decl.statistic_names, ("component_counts", "components"))
        self.assertEqual(hetero_decl.child_roles, ("component_0", "component_1"))
        self.assertFalse(hetero_decl.differentiable)

        semi = SemiSupervisedMixtureDistribution(
            [GaussianDistribution(-1.0, 0.8), GaussianDistribution(2.0, 1.5)], [0.4, 0.6]
        )
        semi_decl = declaration_for(semi)
        self.assertEqual(semi_decl.name, "semi_supervised_mixture")
        self.assertEqual(semi_decl.parameter_names, ("w",))
        self.assertEqual(semi_decl.statistic_names, ("component_counts", "components"))
        self.assertEqual(semi_decl.child_roles, ("component_0", "component_1"))
        self.assertFalse(semi_decl.differentiable)

        joint = JointMixtureDistribution(
            components1=[GaussianDistribution(-1.0, 1.0), GaussianDistribution(2.0, 1.5)],
            components2=[PoissonDistribution(1.5), PoissonDistribution(4.0)],
            w1=[0.4, 0.6],
            w2=[0.5, 0.5],
            taus12=[[0.8, 0.2], [0.25, 0.75]],
            taus21=[[0.64, 0.18181818181818182], [0.36, 0.8181818181818182]],
        )
        joint_decl = declaration_for(joint)
        self.assertEqual(joint_decl.name, "joint_mixture")
        self.assertEqual(joint_decl.parameter_names, ("w1", "w2", "taus12", "taus21"))
        self.assertEqual(
            joint_decl.statistic_names,
            ("component_counts1", "component_counts2", "joint_counts", "components1", "components2"),
        )
        self.assertEqual(
            joint_decl.child_roles, ("x1_component_0", "x1_component_1", "x2_component_0", "x2_component_1")
        )
        self.assertFalse(joint_decl.differentiable)

        hier = HierarchicalMixtureDistribution(
            topics=[GaussianDistribution(-1.0, 0.8), GaussianDistribution(2.0, 1.5)],
            mixture_weights=[0.45, 0.55],
            topic_weights=[[0.8, 0.2], [0.25, 0.75]],
            len_dist=IntegerCategoricalDistribution(0, [0.1, 0.2, 0.4, 0.3]),
        )
        hier_decl = declaration_for(hier)
        self.assertEqual(hier_decl.name, "hierarchical_mixture")
        self.assertEqual(hier_decl.parameter_names, ("w", "taus"))
        self.assertEqual(hier_decl.statistic_names, ("component_counts", "outer_weight_counts", "topics", "length"))
        self.assertEqual(hier_decl.child_roles, ("topic_0", "topic_1", "length"))
        self.assertFalse(hier_decl.differentiable)

        dirac = DiracLengthMixtureDistribution(len_dist=IntegerCategoricalDistribution(0, [0.2, 0.5, 0.3]), p=0.7, v=0)
        dirac_decl = declaration_for(dirac)
        self.assertEqual(dirac_decl.name, "dirac_length_mixture")
        self.assertEqual(dirac_decl.parameter_names, ("p", "v"))
        self.assertEqual(dirac_decl.statistic_names, ("component_counts", "length"))
        self.assertEqual(dirac_decl.child_roles, ("length",))
        self.assertFalse(dirac_decl.differentiable)

    def test_document_association_declarations_describe_payloads_and_children(self):
        plsi = IntegerProbabilisticLatentSemanticIndexingDistribution(
            [[0.70, 0.20], [0.20, 0.50], [0.10, 0.30]],
            [[0.65, 0.35], [0.25, 0.75]],
            [0.40, 0.60],
            len_dist=IntegerCategoricalDistribution(0, [0.1, 0.2, 0.4, 0.3]),
        )
        plsi_decl = declaration_for(plsi)
        self.assertEqual(plsi_decl.name, "integer_plsi")
        self.assertEqual(plsi_decl.parameter_names, ("prob_mat", "state_mat", "doc_vec"))
        self.assertEqual(plsi_decl.statistic_names, ("word_counts", "state_counts", "document_counts", "length"))
        self.assertEqual(plsi_decl.child_roles, ("length",))
        self.assertFalse(plsi_decl.differentiable)

        previous = MultinomialDistribution(
            IntegerCategoricalDistribution(0, [0.6, 0.4]), len_dist=IntegerCategoricalDistribution(0, [0.2, 0.5, 0.3])
        )
        int_assoc = IntegerHiddenAssociationDistribution(
            [[0.60, 0.30, 0.10], [0.20, 0.50, 0.30]],
            [[0.80, 0.20], [0.40, 0.60]],
            alpha=0.1,
            prev_dist=previous,
            len_dist=IntegerCategoricalDistribution(0, [0.1, 0.2, 0.4, 0.3]),
            use_numba=False,
        )
        int_assoc_decl = declaration_for(int_assoc)
        self.assertEqual(int_assoc_decl.name, "integer_hidden_association")
        self.assertEqual(int_assoc_decl.parameter_names, ("state_prob_mat", "cond_weights", "alpha"))
        self.assertEqual(
            int_assoc_decl.statistic_names, ("initial_counts", "weight_counts", "state_counts", "previous", "length")
        )
        self.assertEqual(int_assoc_decl.child_roles, ("previous", "length"))
        self.assertFalse(int_assoc_decl.differentiable)

        conditional = ConditionalDistribution(
            {"a": CategoricalDistribution({"x": 0.8, "y": 0.2}), "b": CategoricalDistribution({"x": 0.3, "y": 0.7})},
            default_dist=CategoricalDistribution({"x": 0.5, "y": 0.5}),
        )
        given = MultinomialDistribution(
            CategoricalDistribution({"a": 0.6, "b": 0.4}), len_dist=IntegerCategoricalDistribution(0, [0.2, 0.5, 0.3])
        )
        hidden_assoc = HiddenAssociationDistribution(
            conditional, given_dist=given, len_dist=IntegerCategoricalDistribution(0, [0.1, 0.2, 0.4, 0.3])
        )
        hidden_assoc_decl = declaration_for(hidden_assoc)
        self.assertEqual(hidden_assoc_decl.name, "hidden_association")
        self.assertEqual(hidden_assoc_decl.parameter_names, ())
        self.assertEqual(hidden_assoc_decl.statistic_names, ("conditional", "given", "length"))
        self.assertEqual(hidden_assoc_decl.child_roles, ("conditional", "given", "length"))
        self.assertFalse(hidden_assoc_decl.differentiable)

        lda = LDADistribution(
            [CategoricalDistribution({"a": 0.70, "b": 0.30}), CategoricalDistribution({"a": 0.20, "b": 0.80})],
            [0.8, 1.3],
            len_dist=IntegerCategoricalDistribution(0, [0.1, 0.2, 0.4, 0.3]),
            gamma_threshold=1.0e-10,
        )
        lda_decl = declaration_for(lda)
        self.assertEqual(lda_decl.name, "lda")
        self.assertEqual(lda_decl.parameter_names, ("alpha", "gamma_threshold"))
        self.assertEqual(
            lda_decl.statistic_names,
            ("previous_alpha", "sum_of_logs", "document_count", "topic_counts", "topics", "length"),
        )
        self.assertEqual(lda_decl.child_roles, ("topic_0", "topic_1", "length"))
        self.assertFalse(lda_decl.differentiable)

    def test_hmm_declarations_describe_parameters_payloads_and_children(self):
        hmm = HiddenMarkovModelDistribution(
            [GaussianDistribution(-1.0, 0.8), GaussianDistribution(2.0, 1.5)],
            [0.6, 0.4],
            [[0.75, 0.25], [0.20, 0.80]],
            len_dist=IntegerCategoricalDistribution(0, [0.1, 0.2, 0.4, 0.3]),
            use_numba=False,
        )
        hmm_decl = declaration_for(hmm)
        self.assertEqual(hmm_decl.name, "hidden_markov")
        self.assertEqual(hmm_decl.parameter_names, ("w", "transitions", "taus"))
        self.assertEqual(
            hmm_decl.statistic_names,
            ("num_states", "initial_counts", "state_counts", "transition_counts", "emissions", "length"),
        )
        self.assertEqual(hmm_decl.child_roles, ("state_0_emission", "state_1_emission", "length"))
        self.assertFalse(hmm_decl.differentiable)
        hmm_params = hmm_decl.parameter_values(hmm)
        np.testing.assert_allclose(hmm_params["w"], np.asarray([0.6, 0.4]))
        self.assertIsNone(hmm_params["taus"])

        quantized = QuantizedHiddenMarkovModelDistribution(
            theta=0.5,
            levels=["a", "b"],
            transition_exponents=[[0, 2], [1, 0]],
            emission_exponents=[[0, 1], [2, 0]],
            initial_exponents=[0, 1],
            len_dist=IntegerCategoricalDistribution(0, [0.1, 0.2, 0.4, 0.3]),
            use_numba=False,
        )
        quant_decl = declaration_for(quantized)
        self.assertEqual(quant_decl.name, "quantized_hidden_markov")
        self.assertEqual(
            quant_decl.parameter_names,
            (
                "theta",
                "levels",
                "transition_exponents",
                "emission_exponents",
                "initial_exponents",
                "init_mode",
                "k_max",
            ),
        )
        self.assertEqual(quant_decl.statistic_names, hmm_decl.statistic_names)
        self.assertEqual(quant_decl.child_roles, ("length",))
        self.assertFalse(quant_decl.differentiable)
        quant_params = quant_decl.parameter_values(quantized)
        self.assertEqual(quant_params["theta"], 0.5)
        self.assertEqual(quant_params["levels"], ["a", "b"])
        np.testing.assert_array_equal(quant_params["initial_exponents"], np.asarray([0, 1]))

        segmental = SegmentalHiddenMarkovModelDistribution(
            [GaussianDistribution(-1.0, 0.8), GaussianDistribution(2.0, 1.5)],
            [0.6, 0.4],
            [[0.75, 0.25], [0.20, 0.80]],
            len_dist=IntegerCategoricalDistribution(0, [0.1, 0.2, 0.4, 0.3]),
        )
        segmental_decl = declaration_for(segmental)
        self.assertEqual(segmental_decl.name, "segmental_hidden_markov")
        self.assertEqual(segmental_decl.parameter_names, ("w", "transitions"))
        self.assertEqual(segmental_decl.statistic_names, hmm_decl.statistic_names)
        self.assertEqual(segmental_decl.child_roles, ("state_0_emission", "state_1_emission", "length"))
        self.assertFalse(segmental_decl.differentiable)
        segmental_params = segmental_decl.parameter_values(segmental)
        np.testing.assert_allclose(segmental_params["w"], np.asarray([0.6, 0.4]))

        tree_hmm = TreeHiddenMarkovModelDistribution(
            [GaussianDistribution(-1.0, 0.8), GaussianDistribution(2.0, 1.5)],
            [0.6, 0.4],
            [[0.75, 0.25], [0.20, 0.80]],
            len_dist=IntegerCategoricalDistribution(0, [0.1, 0.2, 0.4]),
            terminal_level=4,
            use_numba=False,
        )
        tree_decl = declaration_for(tree_hmm)
        self.assertEqual(tree_decl.name, "tree_hidden_markov")
        self.assertEqual(tree_decl.parameter_names, ("w", "transitions"))
        self.assertEqual(tree_decl.statistic_names, hmm_decl.statistic_names)
        self.assertEqual(tree_decl.child_roles, ("state_0_emission", "state_1_emission", "length"))
        self.assertFalse(tree_decl.differentiable)
        tree_params = tree_decl.parameter_values(tree_hmm)
        np.testing.assert_allclose(tree_params["transitions"], np.asarray([[0.75, 0.25], [0.20, 0.80]]))

        # (the semi-supervised HMM is numpy-only and exposes no compute declaration)

    def test_declared_leaf_statistics_match_accumulator_layouts(self):
        cases = [
            (GaussianDistribution(0.0, 1.0), GaussianEstimator(), np.asarray([-1.0, 0.0, 1.0])),
            (PoissonDistribution(3.0), PoissonEstimator(), [0, 2, 4]),
            (ExponentialDistribution(2.0), ExponentialEstimator(), np.asarray([0.2, 1.0, 3.0])),
            (BernoulliDistribution(0.4), BernoulliEstimator(), [False, True, True, False]),
            (CategoricalDistribution({"a": 0.6, "b": 0.4}), None, ["a", "b", "a"]),
            (GammaDistribution(2.0, 1.5), GammaEstimator(), np.asarray([0.5, 1.0, 2.0])),
            (LogGaussianDistribution(0.0, 0.5), LogGaussianEstimator(), np.asarray([0.5, 1.0, 2.0])),
            (BinomialDistribution(0.4, 5), BinomialEstimator(), [0, 2, 4]),
            (NegativeBinomialDistribution(2.0, 0.4), NegativeBinomialEstimator(r=2.0), [0, 1, 3]),
            (GeometricDistribution(0.4), GeometricEstimator(), [1, 2, 3]),
            (
                DiagonalGaussianDistribution([0.0, 1.0], [1.0, 2.0]),
                DiagonalGaussianEstimator(dim=2),
                np.asarray([[0.0, 1.0], [2.0, 3.0], [-1.0, 0.0]]),
            ),
            (StudentTDistribution(5.0), StudentTEstimator(df=5.0), np.asarray([-1.0, 0.0, 2.0])),
            (LogisticDistribution(0.0, 1.0), LogisticEstimator(), np.asarray([-1.0, 0.0, 2.0])),
            (WeibullDistribution(1.5, 2.0), WeibullEstimator(), np.asarray([0.2, 1.0, 2.0])),
            (RayleighDistribution(1.0), RayleighEstimator(), np.asarray([0.5, 1.0, 2.0])),
            (ParetoDistribution(1.0, 2.0), ParetoEstimator(), np.asarray([1.1, 2.0, 3.0])),
            (UniformDistribution(0.0, 3.0), UniformEstimator(), np.asarray([0.5, 1.0, 2.5])),
            (
                IntegerCategoricalDistribution(0, [0.2, 0.5, 0.3]),
                IntegerCategoricalEstimator(min_val=0, max_val=2),
                [0, 1, 2, 1],
            ),
            (BetaDistribution(2.0, 5.0), BetaEstimator(), np.asarray([0.1, 0.25, 0.6, 0.8])),
            (
                DirichletDistribution([2.0, 3.0, 4.0]),
                DirichletEstimator(dim=3),
                np.asarray([[0.2, 0.3, 0.5], [0.4, 0.4, 0.2], [0.1, 0.7, 0.2]]),
            ),
            (LaplaceDistribution(0.5, 1.7), LaplaceEstimator(), np.asarray([-2.0, 0.0, 0.5, 3.0])),
            (
                MultivariateGaussianDistribution([0.5, -1.0], [[1.5, 0.3], [0.3, 2.0]]),
                MultivariateGaussianEstimator(dim=2),
                np.asarray([[-1.0, 0.0], [0.5, -1.0], [2.0, 1.5]]),
            ),
            (
                VonMisesFisherDistribution([1.0, 0.0, 0.0], 3.0),
                None,
                np.asarray(
                    [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0], [1.0 / np.sqrt(2.0), 1.0 / np.sqrt(2.0), 0.0]]
                ),
            ),
            (SpearmanRankingDistribution([0, 1, 2], rho=0.8), None, [[0, 1, 2], [0, 2, 1], [1, 0, 2], [2, 1, 0]]),
            (NullDistribution(), None, [None, "anything", 3.0]),
            (PointMassDistribution("fixed"), None, ["fixed", "other", "fixed"]),
            (BernoulliSetDistribution({"a": 0.7, "b": 0.2}, min_prob=0.0), None, [[], ["a"], ["a", "b"]]),
            (IndianBuffetProcessDistribution(3, alpha=1.2, feature_probs=[0.2, 0.5, 0.8]), None, [[], [0, 2], [1]]),
            (IntegerUniformSpikeDistribution(k=2, num_vals=5, p=0.6, min_val=0), None, [0, 1, 2, 3, 4, 5]),
            (IntegerBernoulliSetDistribution(np.log([0.2, 0.5, 0.8])), None, [[], [0], [1, 2], [0, 2]]),
        ]

        for dist, estimator, data in cases:
            with self.subTest(dist=type(dist).__name__):
                declaration = declaration_for(dist)
                self.assertIsNotNone(declaration)
                self.assertEqual(tuple(declaration.parameter_values(dist).keys()), declaration.parameter_names)

                est = dist.estimator() if estimator is None else estimator
                enc = dist.dist_to_encoder().seq_encode(data)
                weights = np.linspace(0.5, 1.5, len(data))
                acc = est.accumulator_factory().make()
                acc.seq_update(enc, weights, dist)

                stats = declaration.statistic_values(acc.value())
                self.assertEqual(tuple(stats.keys()), declaration.statistic_names)
                self.assertEqual(statistic_layout_issues(declaration, acc.value()), ())
                self.assertIs(validate_statistic_layout(declaration, acc.value()), declaration)

    def test_instance_declaration_statistics_match_accumulator_layouts(self):
        cases = [
            (
                IgnoredDistribution(GaussianDistribution(0.0, 1.0)),
                IgnoredEstimator(GaussianDistribution(0.0, 1.0)),
                [-1.0, 0.0, 1.0],
            ),
            (
                OptionalDistribution(GaussianDistribution(0.0, 1.0), p=0.25),
                OptionalEstimator(GaussianEstimator()),
                [None, -1.0, 0.0, 1.0],
            ),
            (
                SequenceDistribution(
                    GaussianDistribution(0.0, 1.0), len_dist=IntegerCategoricalDistribution(0, [0.2, 0.3, 0.5])
                ),
                SequenceEstimator(GaussianEstimator(), len_estimator=IntegerCategoricalEstimator(min_val=0, max_val=2)),
                [[], [0.0], [-1.0, 0.5]],
            ),
            (
                MultinomialDistribution(
                    CategoricalDistribution({"a": 0.6, "b": 0.4}),
                    len_dist=IntegerCategoricalDistribution(0, [0.2, 0.3, 0.5]),
                ),
                MultinomialEstimator(
                    CategoricalEstimator(), len_estimator=IntegerCategoricalEstimator(min_val=0, max_val=2)
                ),
                [[], [("a", 1)], [("a", 1), ("b", 1)]],
            ),
            (
                WeightedDistribution(UniformDistribution(0.0, 3.0)),
                WeightedEstimator(UniformEstimator()),
                [(0.5, 0.25), (1.0, 2.0), (2.5, 0.5)],
            ),
            (
                CompositeDistribution((UniformDistribution(0.0, 3.0), BinomialDistribution(0.4, 5))),
                CompositeEstimator((UniformEstimator(), BinomialEstimator())),
                [(0.5, 0), (1.0, 2), (2.5, 4)],
            ),
            (
                RecordDistribution({"u": UniformDistribution(0.0, 3.0), "b": BinomialDistribution(0.4, 5)}),
                RecordEstimator({"u": UniformEstimator(), "b": BinomialEstimator()}),
                [{"u": 0.5, "b": 0}, {"u": 1.0, "b": 2}, {"u": 2.5, "b": 4}],
            ),
            (
                TransformDistribution(
                    GaussianDistribution(0.0, 1.0),
                    transform=AffineTransform(loc=2.0, scale=3.0),
                    density_correction=True,
                ),
                TransformEstimator(
                    GaussianEstimator(), transform=AffineTransform(loc=2.0, scale=3.0), density_correction=True
                ),
                np.asarray([-1.0, 2.0, 5.0]),
            ),
            (
                SelectDistribution([GaussianDistribution(-1.0, 0.5), GaussianDistribution(1.0, 0.7)], _sign_choice),
                SelectEstimator([GaussianEstimator(), GaussianEstimator()], _sign_choice),
                np.asarray([-2.0, -0.5, 0.25, 1.5]),
            ),
            (
                ConditionalDistribution(
                    {"a": GaussianDistribution(-1.0, 0.5), "b": GaussianDistribution(1.0, 0.7)},
                    default_dist=GaussianDistribution(0.0, 2.0),
                    given_dist=CategoricalDistribution({"a": 0.4, "b": 0.5, "c": 0.1}),
                ),
                ConditionalDistributionEstimator(
                    {"a": GaussianEstimator(), "b": GaussianEstimator()},
                    default_estimator=GaussianEstimator(),
                    given_estimator=CategoricalEstimator(),
                ),
                [("a", -1.5), ("b", 1.25), ("c", 0.0), ("a", -0.5)],
            ),
            (
                MarkovChainDistribution(
                    {"a": 0.7, "b": 0.3},
                    {"a": {"a": 0.2, "b": 0.8}, "b": {"a": 0.6, "b": 0.4}},
                    len_dist=IntegerCategoricalDistribution(0, [0.1, 0.2, 0.4, 0.3]),
                ),
                MarkovChainEstimator(len_estimator=IntegerCategoricalEstimator(min_val=0, max_val=3)),
                [[], ["a"], ["a", "b", "a"], ["b", "a"]],
            ),
            (
                MixtureDistribution([UniformDistribution(0.0, 3.0), UniformDistribution(1.0, 5.0)], [0.4, 0.6]),
                MixtureEstimator([UniformEstimator(), UniformEstimator()]),
                np.asarray([0.5, 1.2, 2.5, 4.0]),
            ),
            (
                HeterogeneousMixtureDistribution(
                    [ExponentialDistribution(1.5), GammaDistribution(2.0, 0.75)], [0.35, 0.65]
                ),
                None,
                np.asarray([0.2, 0.8, 1.5, 3.0]),
            ),
            (
                SemiSupervisedMixtureDistribution(
                    [GaussianDistribution(-1.0, 0.8), GaussianDistribution(2.0, 1.5)], [0.4, 0.6]
                ),
                None,
                [(-1.2, None), (0.0, [(0, 0.8), (1, 0.2)]), (2.6, [(1, 1.0)]), (1.9, None)],
            ),
            (
                JointMixtureDistribution(
                    components1=[GaussianDistribution(-1.0, 1.0), GaussianDistribution(2.0, 1.5)],
                    components2=[GaussianDistribution(0.0, 2.0), GaussianDistribution(3.0, 0.75)],
                    w1=[0.4, 0.6],
                    w2=[0.5, 0.5],
                    taus12=[[0.8, 0.2], [0.25, 0.75]],
                    taus21=[[0.64, 0.18181818181818182], [0.36, 0.8181818181818182]],
                ),
                JointMixtureEstimator(
                    [GaussianEstimator(), GaussianEstimator()], [GaussianEstimator(), GaussianEstimator()]
                ),
                [(-1.2, -0.3), (0.0, 2.1), (2.6, 3.5), (1.9, 2.8)],
            ),
            (
                HiddenMarkovModelDistribution(
                    [CategoricalDistribution({"a": 0.8, "b": 0.2}), CategoricalDistribution({"a": 0.3, "b": 0.7})],
                    [0.6, 0.4],
                    [[0.7, 0.3], [0.2, 0.8]],
                    len_dist=IntegerCategoricalDistribution(0, [0.1, 0.2, 0.4, 0.3]),
                ),
                HiddenMarkovEstimator(
                    [CategoricalEstimator(), CategoricalEstimator()],
                    len_estimator=IntegerCategoricalEstimator(min_val=0, max_val=3),
                ),
                [[], ["a"], ["a", "b", "a"], ["b", "a"]],
            ),
            (
                HierarchicalMixtureDistribution(
                    topics=[GaussianDistribution(0.0, 1.0), GaussianDistribution(4.0, 2.0)],
                    mixture_weights=[0.5, 0.5],
                    topic_weights=[[0.8, 0.2], [0.3, 0.7]],
                    len_dist=IntegerCategoricalDistribution(1, [0.3, 0.5, 0.2]),
                ),
                HierarchicalMixtureEstimator(
                    [GaussianEstimator(), GaussianEstimator()],
                    2,
                    len_estimator=IntegerCategoricalEstimator(min_val=1, max_val=3),
                ),
                [[0.1], [3.5, 4.2], [0.0, 4.0, 4.5], [1.0, 1.2]],
            ),
            (
                SegmentalHiddenMarkovModelDistribution(
                    [GaussianDistribution(-2.0, 1.0), StudentTDistribution(5.0, loc=2.0, scale=1.5)],
                    [0.6, 0.4],
                    [[0.7, 0.3], [0.2, 0.8]],
                    len_dist=IntegerCategoricalDistribution(0, [0.0, 0.0, 1.0, 0.0]),
                ),
                SegmentalHiddenMarkovEstimator(
                    [GaussianEstimator(), StudentTEstimator(df=5.0)],
                    len_estimator=IntegerCategoricalEstimator(min_val=0, max_val=3),
                    pseudo_count=(1.0, 1.0),
                ),
                [[-2.0, 1.0], [2.5, 2.0], [-1.0, -2.5]],
            ),
            (
                DiracLengthMixtureDistribution(len_dist=IntegerCategoricalDistribution(0, [0.2, 0.5, 0.3]), p=0.7, v=0),
                None,
                [0, 1, 2],
            ),
            (
                IntegerMultinomialDistribution(
                    min_val=0,
                    p_vec=[0.55, 0.30, 0.15],
                    len_dist=IntegerCategoricalDistribution(0, [0.05, 0.20, 0.35, 0.40]),
                ),
                None,
                [[], [(0, 2.0), (1, 1.0)], [(2, 2.0)], [(0, 1.0), (1, 2.0), (2, 1.0)]],
            ),
            (
                IntegerMarkovChainDistribution(
                    3,
                    [[0.70, 0.20, 0.10], [0.10, 0.60, 0.30], [0.25, 0.25, 0.50]],
                    lag=1,
                    init_dist=SequenceDistribution(
                        IntegerCategoricalDistribution(0, [0.25, 0.45, 0.30]),
                        len_dist=IntegerCategoricalDistribution(1, [1.0]),
                    ),
                    len_dist=IntegerCategoricalDistribution(0, [0.10, 0.20, 0.30, 0.40]),
                ),
                None,
                [[], [0], [0, 1, 2], [2, 2, 1]],
            ),
            (
                IntegerProbabilisticLatentSemanticIndexingDistribution(
                    [[0.70, 0.10], [0.20, 0.30], [0.10, 0.60]],
                    [[0.80, 0.20], [0.25, 0.75]],
                    [0.55, 0.45],
                    len_dist=IntegerCategoricalDistribution(0, [0.05, 0.15, 0.30, 0.30, 0.20]),
                ),
                None,
                [(0, [(0, 2.0), (1, 1.0)]), (1, [(2, 3.0)]), (1, []), (0, [(1, 2.0), (2, 1.0)])],
            ),
            (
                IntegerHiddenAssociationDistribution(
                    state_prob_mat=[[0.70, 0.20, 0.10], [0.10, 0.30, 0.60]],
                    cond_weights=[[0.80, 0.20], [0.30, 0.70], [0.50, 0.50]],
                    alpha=0.15,
                    len_dist=IntegerCategoricalDistribution(0, [0.10, 0.20, 0.30, 0.40]),
                    use_numba=False,
                ),
                None,
                [([(0, 2.0), (1, 1.0)], [(0, 1.0), (2, 2.0)]), ([(2, 1.0)], [(1, 3.0)]), ([(1, 1.0)], [])],
            ),
            (
                HiddenAssociationDistribution(
                    cond_dist=ConditionalDistribution(
                        {
                            "a": CategoricalDistribution({"x": 0.80, "y": 0.20}),
                            "b": CategoricalDistribution({"x": 0.25, "y": 0.75}),
                        }
                    ),
                    len_dist=CategoricalDistribution({0.0: 0.10, 2.0: 0.30, 3.0: 0.60}),
                ),
                None,
                [
                    ([("a", 2.0), ("b", 1.0)], [("x", 1.0), ("y", 2.0)]),
                    ([("b", 3.0)], [("y", 2.0)]),
                    ([("a", 1.0)], []),
                ],
            ),
            (
                LDADistribution(
                    [
                        IntegerCategoricalDistribution(0, [0.70, 0.20, 0.10]),
                        IntegerCategoricalDistribution(0, [0.10, 0.30, 0.60]),
                    ],
                    [0.8, 1.3],
                    len_dist=IntegerCategoricalDistribution(0, [0.05, 0.15, 0.30, 0.30, 0.20]),
                    gamma_threshold=1.0e-10,
                ),
                None,
                [[(0, 2.0), (1, 1.0)], [(2, 3.0)], [(1, 1.0), (2, 1.0)]],
            ),
        ]

        for dist, estimator, data in cases:
            with self.subTest(dist=type(dist).__name__):
                declaration = declaration_for(dist)
                self.assertIsNotNone(declaration)
                est = dist.estimator() if estimator is None else estimator
                enc = dist.dist_to_encoder().seq_encode(data)
                weights = np.linspace(0.5, 1.5, len(data))
                acc = est.accumulator_factory().make()
                acc.seq_update(enc, weights, dist)

                stats = declaration.statistic_values(acc.value())
                self.assertEqual(tuple(stats.keys()), declaration.statistic_names)
                self.assertEqual(set(stats.keys()), set(declaration.statistic_names))
                self.assertEqual(statistic_layout_issues(declaration, acc.value()), ())
                self.assertIs(validate_statistic_layout(declaration, acc.value()), declaration)

    def test_statistic_layout_validation_reports_nested_child_errors(self):
        dist = MixtureDistribution(
            [GaussianDistribution(-1.0, 1.0), GaussianDistribution(1.0, 1.5)],
            [0.4, 0.6],
        )
        declaration = declaration_for(dist)
        bad_stats = (np.asarray([1.0, 2.0]), ((1.0, np.asarray([0.0])),))

        issues = statistic_layout_issues(declaration, bad_stats)
        self.assertTrue(any("components expected 2 child statistics, got 1" in issue for issue in issues))
        with self.assertRaises(ValueError):
            validate_statistic_layout(declaration, bad_stats)

    def test_accumulator_scale_matches_reweighted_seq_update_for_declared_families(self):
        cases = [
            (GaussianDistribution(0.0, 1.0), GaussianEstimator(), np.asarray([-1.0, 0.0, 1.0])),
            (PoissonDistribution(3.0), PoissonEstimator(), [0, 2, 4]),
            (ExponentialDistribution(2.0), ExponentialEstimator(), np.asarray([0.2, 1.0, 3.0])),
            (BernoulliDistribution(0.4), BernoulliEstimator(), [False, True, True, False]),
            (CategoricalDistribution({"a": 0.6, "b": 0.4}), CategoricalEstimator(), ["a", "b", "a"]),
            (GammaDistribution(2.0, 1.5), GammaEstimator(), np.asarray([0.5, 1.0, 2.0])),
            (LogGaussianDistribution(0.0, 0.5), LogGaussianEstimator(), np.asarray([0.5, 1.0, 2.0])),
            (BinomialDistribution(0.4, 5), BinomialEstimator(), [0, 2, 4]),
            (NegativeBinomialDistribution(2.0, 0.4), NegativeBinomialEstimator(r=2.0), [0, 1, 3]),
            (GeometricDistribution(0.4), GeometricEstimator(), [1, 2, 3]),
            (
                DiagonalGaussianDistribution([0.0, 1.0], [1.0, 2.0]),
                DiagonalGaussianEstimator(dim=2),
                np.asarray([[0.0, 1.0], [2.0, 3.0], [-1.0, 0.0]]),
            ),
            (StudentTDistribution(5.0), StudentTEstimator(df=5.0), np.asarray([-1.0, 0.0, 2.0])),
            (LogisticDistribution(0.0, 1.0), LogisticEstimator(), np.asarray([-1.0, 0.0, 2.0])),
            (WeibullDistribution(1.5, 2.0), WeibullEstimator(), np.asarray([0.2, 1.0, 2.0])),
            (RayleighDistribution(1.0), RayleighEstimator(), np.asarray([0.5, 1.0, 2.0])),
            (ParetoDistribution(1.0, 2.0), ParetoEstimator(), np.asarray([1.1, 2.0, 3.0])),
            (UniformDistribution(0.0, 3.0), UniformEstimator(), np.asarray([0.5, 1.0, 2.5])),
            (
                IntegerCategoricalDistribution(0, [0.2, 0.5, 0.3]),
                IntegerCategoricalEstimator(min_val=0, max_val=2),
                [0, 1, 2, 1],
            ),
            (BetaDistribution(2.0, 5.0), BetaEstimator(), np.asarray([0.1, 0.25, 0.6, 0.8])),
            (LaplaceDistribution(0.5, 1.7), LaplaceEstimator(), np.asarray([-2.0, 0.0, 0.5, 3.0])),
            (
                MultivariateGaussianDistribution([0.5, -1.0], [[1.5, 0.3], [0.3, 2.0]]),
                MultivariateGaussianEstimator(dim=2),
                np.asarray([[-1.0, 0.0], [0.5, -1.0], [2.0, 1.5]]),
            ),
            (
                DirichletDistribution([2.0, 3.0, 4.0]),
                DirichletEstimator(dim=3),
                np.asarray([[0.2, 0.3, 0.5], [0.4, 0.4, 0.2], [0.1, 0.7, 0.2]]),
            ),
            (
                VonMisesFisherDistribution([1.0, 0.0, 0.0], 3.0),
                None,
                np.asarray(
                    [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0], [1.0 / np.sqrt(2.0), 1.0 / np.sqrt(2.0), 0.0]]
                ),
            ),
            (SpearmanRankingDistribution([0, 1, 2], rho=0.8), None, [[0, 1, 2], [0, 2, 1], [1, 0, 2], [2, 1, 0]]),
            (NullDistribution(), NullDistribution().estimator(), [None, "anything", 3.0]),
            (PointMassDistribution("fixed"), PointMassDistribution("fixed").estimator(), ["fixed", "other", "fixed"]),
            (
                IgnoredDistribution(GaussianDistribution(0.0, 1.0)),
                IgnoredEstimator(GaussianDistribution(0.0, 1.0)),
                [-1.0, 0.0, 1.0],
            ),
            (
                OptionalDistribution(GaussianDistribution(0.0, 1.0), p=0.25),
                OptionalEstimator(GaussianEstimator()),
                [None, -1.0, 0.0, 1.0],
            ),
            (
                SequenceDistribution(
                    GaussianDistribution(0.0, 1.0), len_dist=IntegerCategoricalDistribution(0, [0.2, 0.3, 0.5])
                ),
                SequenceEstimator(GaussianEstimator(), len_estimator=IntegerCategoricalEstimator(min_val=0, max_val=2)),
                [[], [0.0], [-1.0, 0.5]],
            ),
            (
                MultinomialDistribution(
                    CategoricalDistribution({"a": 0.6, "b": 0.4}),
                    len_dist=IntegerCategoricalDistribution(0, [0.2, 0.3, 0.5]),
                ),
                MultinomialEstimator(
                    CategoricalEstimator(), len_estimator=IntegerCategoricalEstimator(min_val=0, max_val=2)
                ),
                [[], [("a", 1)], [("a", 1), ("b", 1)]],
            ),
            (
                WeightedDistribution(UniformDistribution(0.0, 3.0)),
                WeightedEstimator(UniformEstimator()),
                [(0.5, 0.25), (1.0, 2.0), (2.5, 0.5)],
            ),
            (
                CompositeDistribution((UniformDistribution(0.0, 3.0), BinomialDistribution(0.4, 5))),
                CompositeEstimator((UniformEstimator(), BinomialEstimator())),
                [(0.5, 0), (1.0, 2), (2.5, 4)],
            ),
            (
                RecordDistribution({"u": UniformDistribution(0.0, 3.0), "b": BinomialDistribution(0.4, 5)}),
                RecordEstimator({"u": UniformEstimator(), "b": BinomialEstimator()}),
                [{"u": 0.5, "b": 0}, {"u": 1.0, "b": 2}, {"u": 2.5, "b": 4}],
            ),
            (
                TransformDistribution(
                    GaussianDistribution(0.0, 1.0),
                    transform=AffineTransform(loc=2.0, scale=3.0),
                    density_correction=True,
                ),
                TransformEstimator(
                    GaussianEstimator(), transform=AffineTransform(loc=2.0, scale=3.0), density_correction=True
                ),
                np.asarray([-1.0, 2.0, 5.0]),
            ),
            (
                SelectDistribution([GaussianDistribution(-1.0, 0.5), GaussianDistribution(1.0, 0.7)], _sign_choice),
                SelectEstimator([GaussianEstimator(), GaussianEstimator()], _sign_choice),
                np.asarray([-2.0, -0.5, 0.25, 1.5]),
            ),
            (
                ConditionalDistribution(
                    {"a": GaussianDistribution(-1.0, 0.5), "b": GaussianDistribution(1.0, 0.7)},
                    default_dist=GaussianDistribution(0.0, 2.0),
                    given_dist=CategoricalDistribution({"a": 0.4, "b": 0.5, "c": 0.1}),
                ),
                ConditionalDistributionEstimator(
                    {"a": GaussianEstimator(), "b": GaussianEstimator()},
                    default_estimator=GaussianEstimator(),
                    given_estimator=CategoricalEstimator(),
                ),
                [("a", -1.5), ("b", 1.25), ("c", 0.0), ("a", -0.5)],
            ),
            (
                MarkovChainDistribution(
                    {"a": 0.7, "b": 0.3},
                    {"a": {"a": 0.2, "b": 0.8}, "b": {"a": 0.6, "b": 0.4}},
                    len_dist=IntegerCategoricalDistribution(0, [0.1, 0.2, 0.4, 0.3]),
                ),
                MarkovChainEstimator(len_estimator=IntegerCategoricalEstimator(min_val=0, max_val=3)),
                [[], ["a"], ["a", "b", "a"], ["b", "a"]],
            ),
            (
                MixtureDistribution([UniformDistribution(0.0, 3.0), UniformDistribution(1.0, 5.0)], [0.4, 0.6]),
                MixtureEstimator([UniformEstimator(), UniformEstimator()]),
                np.asarray([0.5, 1.2, 2.5, 4.0]),
            ),
            (
                JointMixtureDistribution(
                    components1=[GaussianDistribution(-1.0, 1.0), GaussianDistribution(2.0, 1.5)],
                    components2=[GaussianDistribution(0.0, 2.0), GaussianDistribution(3.0, 0.75)],
                    w1=[0.4, 0.6],
                    w2=[0.5, 0.5],
                    taus12=[[0.8, 0.2], [0.25, 0.75]],
                    taus21=[[0.64, 0.18181818181818182], [0.36, 0.8181818181818182]],
                ),
                JointMixtureEstimator(
                    [GaussianEstimator(), GaussianEstimator()], [GaussianEstimator(), GaussianEstimator()]
                ),
                [(-1.2, -0.3), (0.0, 2.1), (2.6, 3.5), (1.9, 2.8)],
            ),
            (
                HiddenMarkovModelDistribution(
                    [CategoricalDistribution({"a": 0.8, "b": 0.2}), CategoricalDistribution({"a": 0.3, "b": 0.7})],
                    [0.6, 0.4],
                    [[0.7, 0.3], [0.2, 0.8]],
                    len_dist=IntegerCategoricalDistribution(0, [0.1, 0.2, 0.4, 0.3]),
                ),
                HiddenMarkovEstimator(
                    [CategoricalEstimator(), CategoricalEstimator()],
                    len_estimator=IntegerCategoricalEstimator(min_val=0, max_val=3),
                ),
                [[], ["a"], ["a", "b", "a"], ["b", "a"]],
            ),
            (
                HierarchicalMixtureDistribution(
                    topics=[GaussianDistribution(0.0, 1.0), GaussianDistribution(4.0, 2.0)],
                    mixture_weights=[0.5, 0.5],
                    topic_weights=[[0.8, 0.2], [0.3, 0.7]],
                    len_dist=IntegerCategoricalDistribution(1, [0.3, 0.5, 0.2]),
                ),
                HierarchicalMixtureEstimator(
                    [GaussianEstimator(), GaussianEstimator()],
                    2,
                    len_estimator=IntegerCategoricalEstimator(min_val=1, max_val=3),
                ),
                [[0.1], [3.5, 4.2], [0.0, 4.0, 4.5], [1.0, 1.2]],
            ),
            (
                SegmentalHiddenMarkovModelDistribution(
                    [GaussianDistribution(-2.0, 1.0), StudentTDistribution(5.0, loc=2.0, scale=1.5)],
                    [0.6, 0.4],
                    [[0.7, 0.3], [0.2, 0.8]],
                    len_dist=IntegerCategoricalDistribution(0, [0.0, 0.0, 1.0, 0.0]),
                ),
                SegmentalHiddenMarkovEstimator(
                    [GaussianEstimator(), StudentTEstimator(df=5.0)],
                    len_estimator=IntegerCategoricalEstimator(min_val=0, max_val=3),
                    pseudo_count=(1.0, 1.0),
                ),
                [[-2.0, 1.0], [2.5, 2.0], [-1.0, -2.5]],
            ),
            (IntegerUniformSpikeDistribution(k=2, num_vals=5, p=0.6, min_val=0), None, [0, 1, 2, 3, 4, 5]),
            (IntegerBernoulliSetDistribution(np.log([0.2, 0.5, 0.8])), None, [[], [0], [1, 2], [0, 2]]),
            (
                IndianBuffetProcessDistribution(
                    4, alpha=1.2, feature_probs=[0.15, 0.45, 0.75, 0.35], data_format="sparse"
                ),
                None,
                [[], [0, 2], [1], [0, 1, 3]],
            ),
            (
                IntegerBernoulliEditDistribution(
                    np.log([[0.20, 0.75], [0.45, 0.30], [0.65, 0.55]]),
                    init_dist=IntegerBernoulliSetDistribution(np.log([0.30, 0.55, 0.75])),
                ),
                IntegerBernoulliEditEstimator(
                    3, init_estimator=IntegerBernoulliSetDistribution(np.log([0.30, 0.55, 0.75])).estimator()
                ),
                [([0], [0, 1]), ([1, 2], [2]), ([], [0, 2]), ([0, 2], [])],
            ),
            (
                IntegerStepBernoulliEditDistribution(
                    np.log([[0.20, 0.75], [0.45, 0.30], [0.65, 0.55]]),
                    init_dist=IntegerBernoulliSetDistribution(np.log([0.30, 0.55, 0.75])),
                ),
                IntegerStepBernoulliEditEstimator(
                    3, init_estimator=IntegerBernoulliSetDistribution(np.log([0.30, 0.55, 0.75])).estimator()
                ),
                [([0], [0, 1]), ([1, 2], [2]), ([], [0, 2]), ([0, 2], [])],
            ),
            (
                IntegerProbabilisticLatentSemanticIndexingDistribution(
                    [[0.70, 0.10], [0.20, 0.30], [0.10, 0.60]],
                    [[0.80, 0.20], [0.25, 0.75]],
                    [0.55, 0.45],
                    len_dist=IntegerCategoricalDistribution(0, [0.05, 0.15, 0.30, 0.30, 0.20]),
                ),
                None,
                [(0, [(0, 2.0), (1, 1.0)]), (1, [(2, 3.0)]), (1, []), (0, [(1, 2.0), (2, 1.0)])],
            ),
            (
                IntegerHiddenAssociationDistribution(
                    state_prob_mat=[[0.70, 0.20, 0.10], [0.10, 0.30, 0.60]],
                    cond_weights=[[0.80, 0.20], [0.30, 0.70], [0.50, 0.50]],
                    alpha=0.15,
                    len_dist=IntegerCategoricalDistribution(0, [0.10, 0.20, 0.30, 0.40]),
                    use_numba=False,
                ),
                None,
                [([(0, 2.0), (1, 1.0)], [(0, 1.0), (2, 2.0)]), ([(2, 1.0)], [(1, 3.0)]), ([(1, 1.0)], [])],
            ),
            (
                HiddenAssociationDistribution(
                    cond_dist=ConditionalDistribution(
                        {
                            "a": CategoricalDistribution({"x": 0.80, "y": 0.20}),
                            "b": CategoricalDistribution({"x": 0.25, "y": 0.75}),
                        }
                    ),
                    len_dist=CategoricalDistribution({0.0: 0.10, 2.0: 0.30, 3.0: 0.60}),
                ),
                None,
                [
                    ([("a", 2.0), ("b", 1.0)], [("x", 1.0), ("y", 2.0)]),
                    ([("b", 3.0)], [("y", 2.0)]),
                    ([("a", 1.0)], []),
                ],
            ),
            (
                LDADistribution(
                    [
                        IntegerCategoricalDistribution(0, [0.70, 0.20, 0.10]),
                        IntegerCategoricalDistribution(0, [0.10, 0.30, 0.60]),
                    ],
                    [0.8, 1.3],
                    len_dist=IntegerCategoricalDistribution(0, [0.05, 0.15, 0.30, 0.30, 0.20]),
                    gamma_threshold=1.0e-10,
                ),
                None,
                [[(0, 2.0), (1, 1.0)], [(2, 3.0)], [(1, 1.0), (2, 1.0)]],
            ),
        ]
        c = 0.37

        for dist, estimator, data in cases:
            with self.subTest(dist=type(dist).__name__):
                enc = dist.dist_to_encoder().seq_encode(data)
                weights = np.linspace(0.5, 1.5, len(data))
                estimator = dist.estimator() if estimator is None else estimator
                acc = estimator.accumulator_factory().make()
                acc.seq_update(enc, weights, dist)
                scaled = acc.scale(c)
                self.assertIs(scaled, acc)

                expected = estimator.accumulator_factory().make()
                expected.seq_update(enc, weights * c, dist)
                _assert_suff_close(self, acc.value(), expected.value())

                nobs = float(weights.sum() * c)
                scaled_model = estimator.estimate(nobs, acc.value())
                expected_model = estimator.estimate(nobs, expected.value())
                _assert_models_equivalent(self, scaled_model, expected_model, data)

    def test_composite_declaration_carries_child_roles(self):
        dist = CompositeDistribution(
            (
                GaussianDistribution(0.0, 1.0),
                CategoricalDistribution({"a": 0.6, "b": 0.4}),
            )
        )
        declaration = declaration_for(dist)
        self.assertEqual(declaration.name, "composite")
        self.assertEqual(declaration.child_roles, ("field_0", "field_1"))
        self.assertEqual(tuple(child.name for child in declaration.children), ("gaussian", "categorical"))
        self.assertTrue(declaration.differentiable)

    def test_mixture_declaration_maps_legacy_stat_tuple(self):
        dist = MixtureDistribution(
            [GaussianDistribution(-1.0, 1.0), GaussianDistribution(1.0, 1.5)],
            [0.4, 0.6],
        )
        est = MixtureEstimator([GaussianEstimator(), GaussianEstimator()])
        enc = dist.dist_to_encoder().seq_encode(dist.sampler(seed=1).sample(size=12))
        acc = est.accumulator_factory().make()
        acc.seq_update(enc, np.ones(12), dist)

        declaration = declaration_for(dist)
        self.assertEqual(declaration.parameter_names, ("w",))
        self.assertEqual(declaration.child_roles, ("component_0", "component_1"))
        stats = declaration.statistic_values(acc.value())
        self.assertEqual(tuple(stats.keys()), ("component_counts", "components"))
        self.assertEqual(len(stats["components"]), 2)

    def test_optional_and_sequence_declarations_describe_children(self):
        optional = declaration_for(OptionalDistribution(GaussianDistribution(0.0, 1.0), p=0.2))
        self.assertEqual(optional.name, "optional")
        self.assertEqual(optional.child_roles, ("observed",))
        self.assertEqual(optional.statistic_names, ("missing_observed_counts", "observed"))

        sequence = declaration_for(SequenceDistribution(GaussianDistribution(0.0, 1.0)))
        self.assertEqual(sequence.name, "sequence")
        self.assertEqual(sequence.child_roles, ("element",))
        self.assertEqual(sequence.statistic_names, ("elements", "lengths"))

        sequence_with_length = declaration_for(
            SequenceDistribution(
                GaussianDistribution(0.0, 1.0),
                len_dist=IntegerCategoricalDistribution(0, [0.2, 0.3, 0.5]),
            )
        )
        self.assertEqual(sequence_with_length.child_roles, ("element", "length"))

        normalized = declaration_for(SequenceDistribution(GaussianDistribution(0.0, 1.0), len_normalized=True))
        self.assertTrue(normalized.differentiable)


if __name__ == "__main__":
    unittest.main()
