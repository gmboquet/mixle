import importlib
import tempfile
import unittest

import numpy as np

from mixle.engines import NUMPY_ENGINE, NumpyEngine, TorchEngine
from mixle.stats import (
    AffineTransform,
    BernoulliDistribution,
    BernoulliSetDistribution,
    BetaDistribution,
    BinomialDistribution,
    CategoricalDistribution,
    CompositeDistribution,
    ConditionalDistribution,
    DiagonalGaussianDistribution,
    DiracLengthMixtureDistribution,
    DirichletDistribution,
    ExponentialDistribution,
    GammaDistribution,
    GaussianDistribution,
    GeneralizedExtremeValueDistribution,
    GeneralizedParetoDistribution,
    GeometricDistribution,
    HeterogeneousMixtureDistribution,
    HiddenAssociationDistribution,
    HiddenMarkovModelDistribution,
    HierarchicalMixtureDistribution,
    IgnoredDistribution,
    IndianBuffetProcessDistribution,
    IntegerBernoulliEditDistribution,
    IntegerBernoulliSetDistribution,
    IntegerCategoricalDistribution,
    IntegerHiddenAssociationDistribution,
    IntegerMarkovChainDistribution,
    IntegerMultinomialDistribution,
    IntegerProbabilisticLatentSemanticIndexingDistribution,
    IntegerStepBernoulliEditDistribution,
    IntegerUniformSpikeDistribution,
    JointMixtureDistribution,
    LaplaceDistribution,
    LDADistribution,
    LogGaussianDistribution,
    LogisticDistribution,
    MarkovChainDistribution,
    MixtureDistribution,
    MultinomialDistribution,
    MultivariateGaussianDistribution,
    NakagamiDistribution,
    NegativeBinomialDistribution,
    NullDistribution,
    OptionalDistribution,
    ParetoDistribution,
    PointMassDistribution,
    PoissonDistribution,
    QuantizedHiddenMarkovModelDistribution,
    RayleighDistribution,
    RecordDistribution,
    SegmentalHiddenMarkovModelDistribution,
    SelectDistribution,
    SemiSupervisedMixtureDistribution,
    SequenceDistribution,
    SpearmanRankingDistribution,
    StackedMixtureKernel,
    StackedMixtureResidentStats,
    StackedMixtureShardEstimate,
    StudentTDistribution,
    TransformDistribution,
    TreeHiddenMarkovModelDistribution,
    UniformDistribution,
    VonMisesFisherDistribution,
    WeibullDistribution,
    WeightedDistribution,
    backend_seq_component_log_density,
    backend_seq_log_density,
    generated_stacked_log_density,
    generated_stacked_params,
    generated_stacked_sufficient_statistics,
)

HAS_TORCH = importlib.util.find_spec("torch") is not None
if HAS_TORCH:
    import torch
else:
    torch = None


def _single_rank_mesh():
    import torch.distributed as dist
    from torch.distributed.tensor import DeviceMesh

    if not dist.is_initialized():
        path = tempfile.NamedTemporaryFile(delete=False).name
        dist.init_process_group("gloo", rank=0, world_size=1, init_method="file://" + path)
    return DeviceMesh("cpu", [0])


def _sign_choice(x):
    return 0 if float(x) < 0.0 else 1


class BackendScoringTestCase(unittest.TestCase):
    @staticmethod
    def backend_leaf_cases():
        return [
            (GaussianDistribution(0.5, 1.7), np.asarray([-1.0, 0.0, 2.0])),
            (ExponentialDistribution(2.0), np.asarray([0.2, 1.0, 3.0])),
            (PoissonDistribution(3.0), [0, 2, 5]),
            (BernoulliDistribution(0.4), [False, True, True, False]),
            (GammaDistribution(2.0, 1.5), np.asarray([0.5, 1.0, 2.0])),
            (LogGaussianDistribution(0.1, 0.7), np.asarray([0.5, 1.0, 2.5])),
            (BinomialDistribution(0.4, 5), [0, 2, 4]),
            (NegativeBinomialDistribution(2.0, 0.4), [0, 1, 3]),
            (GeometricDistribution(0.4), [1, 2, 3]),
            (DiagonalGaussianDistribution([0.0, 1.0], [1.0, 2.0]), [[-1.0, 0.5], [0.0, 1.0], [2.0, -1.0]]),
            (StudentTDistribution(5.0, 0.25, 1.5), np.asarray([-1.0, 0.0, 2.0])),
            (LogisticDistribution(0.25, 1.5), np.asarray([-2.0, 0.0, 3.0])),
            (WeibullDistribution(1.5, 2.0), np.asarray([0.2, 1.0, 2.5])),
            (RayleighDistribution(1.2), np.asarray([0.2, 1.0, 2.5])),
            (NakagamiDistribution(2.0, 4.0), np.asarray([0.5, 1.0, 2.5])),
            (GeneralizedParetoDistribution(2.0, 0.3), np.asarray([0.1, 1.0, 5.0])),
            (GeneralizedExtremeValueDistribution(0.5, 2.0, 0.2), np.asarray([-1.0, 0.5, 4.0])),
            (ParetoDistribution(1.0, 2.5), np.asarray([1.1, 2.0, 4.0])),
            (UniformDistribution(-1.0, 3.0), np.asarray([-0.5, 0.0, 2.5])),
            (IntegerCategoricalDistribution(0, [0.2, 0.5, 0.3]), [0, 1, 2, 1]),
            (BetaDistribution(2.0, 5.0), np.asarray([0.1, 0.25, 0.6, 0.8])),
            (DirichletDistribution([2.0, 3.0, 4.0]), np.asarray([[0.2, 0.3, 0.5], [0.4, 0.4, 0.2], [0.1, 0.7, 0.2]])),
            (LaplaceDistribution(0.5, 1.7), np.asarray([-2.0, 0.0, 0.5, 3.0])),
            (
                MultivariateGaussianDistribution([0.5, -1.0], [[1.5, 0.3], [0.3, 2.0]]),
                [[-1.0, 0.0], [0.5, -1.0], [2.0, 1.5]],
            ),
            (
                VonMisesFisherDistribution([1.0, 0.0, 0.0], 3.0),
                np.asarray(
                    [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0], [1.0 / np.sqrt(2.0), 1.0 / np.sqrt(2.0), 0.0]]
                ),
            ),
            (SpearmanRankingDistribution([0, 1, 2], rho=0.8), [[0, 1, 2], [0, 2, 1], [1, 0, 2], [2, 1, 0]]),
            (NullDistribution(), [None, "anything", 3.0]),
            (PointMassDistribution("fixed"), ["fixed", "other", "fixed"]),
            (IgnoredDistribution(GaussianDistribution(0.5, 1.7)), np.asarray([-1.0, 0.0, 2.0])),
            (WeightedDistribution(GaussianDistribution(0.5, 1.7)), [(-1.0, 0.25), (0.0, 2.0), (2.0, 0.5)]),
            (
                SequenceDistribution(
                    GaussianDistribution(0.5, 1.7), len_dist=IntegerCategoricalDistribution(0, [0.2, 0.3, 0.5])
                ),
                [[], [0.0], [-1.0, 0.5]],
            ),
            (SequenceDistribution(GaussianDistribution(0.5, 1.7), len_normalized=True), [[], [0.0], [-1.0, 0.5, 2.0]]),
            (
                MultinomialDistribution(
                    CategoricalDistribution({"a": 0.55, "b": 0.30, "c": 0.15}),
                    len_dist=CategoricalDistribution({0.0: 0.05, 2.0: 0.20, 3.0: 0.35, 4.0: 0.40}),
                ),
                [[], [("a", 2.0), ("b", 1.0)], [("c", 2.0)], [("a", 1.0), ("b", 2.0), ("c", 1.0)]],
            ),
            (
                MultinomialDistribution(
                    CategoricalDistribution({"a": 0.55, "b": 0.30, "c": 0.15}), len_normalized=True
                ),
                [[], [("a", 2.0), ("b", 1.0)], [("c", 2.0)], [("a", 1.0), ("b", 2.0), ("c", 1.0)]],
            ),
            (
                TransformDistribution(
                    GaussianDistribution(0.5, 1.7),
                    transform=AffineTransform(loc=2.0, scale=3.0),
                    density_correction=True,
                ),
                np.asarray([-1.0, 2.0, 5.0, 8.0]),
            ),
            (
                SelectDistribution(
                    [GaussianDistribution(-1.0, 0.5), GaussianDistribution(1.0, 0.7)],
                    _sign_choice,
                ),
                np.asarray([-2.0, -0.5, 0.25, 1.5]),
            ),
            (
                ConditionalDistribution(
                    {
                        "a": GaussianDistribution(-1.0, 0.5),
                        "b": GaussianDistribution(1.0, 0.7),
                    },
                    default_dist=GaussianDistribution(0.0, 2.0),
                    given_dist=CategoricalDistribution({"a": 0.4, "b": 0.5, "c": 0.1}),
                ),
                [("a", -1.5), ("b", 1.25), ("c", 0.0), ("a", -0.5)],
            ),
            (
                MarkovChainDistribution(
                    {"a": 0.7, "b": 0.3},
                    {"a": {"a": 0.2, "b": 0.8}, "b": {"a": 0.6, "b": 0.4}},
                    len_dist=IntegerCategoricalDistribution(0, [0.1, 0.2, 0.4, 0.3]),
                ),
                [[], ["a"], ["a", "b", "a"], ["b", "a"]],
            ),
            (
                HiddenMarkovModelDistribution(
                    [GaussianDistribution(-1.0, 0.8), GaussianDistribution(2.0, 1.5)],
                    [0.6, 0.4],
                    [[0.75, 0.25], [0.20, 0.80]],
                    len_dist=IntegerCategoricalDistribution(0, [0.1, 0.2, 0.4, 0.3]),
                    use_numba=False,
                ),
                [[], [-1.2], [-0.5, 1.5], [2.0, 2.5, -0.2]],
            ),
            (
                JointMixtureDistribution(
                    components1=[GaussianDistribution(-1.0, 0.8), GaussianDistribution(2.0, 1.5)],
                    components2=[PoissonDistribution(1.5), PoissonDistribution(4.0)],
                    w1=[0.4, 0.6],
                    w2=[0.5, 0.5],
                    taus12=[[0.8, 0.2], [0.25, 0.75]],
                    taus21=[[0.64, 0.18181818181818182], [0.36, 0.8181818181818182]],
                ),
                [(-1.2, 0), (0.0, 2), (2.6, 5), (1.9, 3)],
            ),
            (
                HeterogeneousMixtureDistribution(
                    [ExponentialDistribution(1.5), GammaDistribution(2.0, 0.75)], [0.35, 0.65]
                ),
                np.asarray([0.2, 0.8, 1.5, 3.0]),
            ),
            (
                SemiSupervisedMixtureDistribution(
                    [GaussianDistribution(-1.0, 0.8), GaussianDistribution(2.0, 1.5)], [0.4, 0.6]
                ),
                [(-1.2, None), (0.0, [(0, 0.8), (1, 0.2)]), (2.6, [(1, 1.0)]), (1.9, None)],
            ),
            (
                HierarchicalMixtureDistribution(
                    topics=[GaussianDistribution(-1.0, 0.8), GaussianDistribution(2.0, 1.5)],
                    mixture_weights=[0.45, 0.55],
                    topic_weights=[[0.8, 0.2], [0.25, 0.75]],
                    len_dist=IntegerCategoricalDistribution(0, [0.1, 0.2, 0.4, 0.3]),
                ),
                [[], [-1.2], [-0.5, 1.5], [2.0, 2.5, -0.2]],
            ),
            (
                SegmentalHiddenMarkovModelDistribution(
                    [GaussianDistribution(-1.0, 0.8), GaussianDistribution(2.0, 1.5)],
                    [0.6, 0.4],
                    [[0.75, 0.25], [0.20, 0.80]],
                    len_dist=IntegerCategoricalDistribution(0, [0.1, 0.2, 0.4, 0.3]),
                ),
                [[], [-1.2], [-0.5, 1.5], [2.0, 2.5, -0.2]],
            ),
            (
                TreeHiddenMarkovModelDistribution(
                    [GaussianDistribution(-1.0, 0.8), GaussianDistribution(2.0, 1.5)],
                    [0.6, 0.4],
                    [[0.75, 0.25], [0.20, 0.80]],
                    len_dist=IntegerCategoricalDistribution(0, [0.1, 0.2, 0.4]),
                    terminal_level=4,
                    use_numba=False,
                ),
                [[((0, -1), -1.2)], [((0, -1), -0.5), ((1, 0), 1.5)], [((0, -1), 2.0), ((1, 0), 2.5), ((2, 0), -0.2)]],
            ),
            (
                QuantizedHiddenMarkovModelDistribution(
                    theta=0.5,
                    levels=["a", "b"],
                    transition_exponents=[[0, 2], [1, 0]],
                    emission_exponents=[[0, 1], [2, 0]],
                    initial_exponents=[0, 1],
                    len_dist=IntegerCategoricalDistribution(0, [0.1, 0.2, 0.4, 0.3]),
                    use_numba=False,
                ),
                [[], ["a"], ["a", "b"], ["b", "b", "a"]],
            ),
            (
                DiracLengthMixtureDistribution(len_dist=IntegerCategoricalDistribution(0, [0.2, 0.5, 0.3]), p=0.7, v=0),
                [0, 1, 2],
            ),
            (IntegerUniformSpikeDistribution(k=2, num_vals=5, p=0.6, min_val=0), [0, 1, 2, 3, 4, 5]),
            (IntegerBernoulliSetDistribution(np.log([0.2, 0.5, 0.8])), [[], [0], [1, 2], [0, 2]]),
            (
                BernoulliSetDistribution({"a": 1.0, "b": 0.25, "c": 0.0}, min_prob=0.0),
                [["a"], ["a", "b"], ["b"], ["a", "c"]],
            ),
            (
                IndianBuffetProcessDistribution(
                    4, alpha=1.2, feature_probs=[0.15, 0.45, 0.75, 0.35], data_format="sparse"
                ),
                [[], [0, 2], [1], [0, 1, 3]],
            ),
            (
                IntegerBernoulliEditDistribution(
                    np.log([[0.20, 0.75], [0.45, 0.30], [0.65, 0.55]]),
                    init_dist=IntegerBernoulliSetDistribution(np.log([0.30, 0.55, 0.75])),
                ),
                [([0], [0, 1]), ([1, 2], [2]), ([], [0, 2]), ([0, 2], [])],
            ),
            (
                IntegerStepBernoulliEditDistribution(
                    np.log([[0.20, 0.75], [0.45, 0.30], [0.65, 0.55]]),
                    init_dist=IntegerBernoulliSetDistribution(np.log([0.30, 0.55, 0.75])),
                ),
                [([0], [0, 1]), ([1, 2], [2]), ([], [0, 2]), ([0, 2], [])],
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
                [[], [0], [0, 1, 2], [2, 2, 1]],
            ),
            (
                IntegerMultinomialDistribution(
                    1, [0.50, 0.30, 0.20], len_dist=IntegerCategoricalDistribution(0, [0.05, 0.10, 0.30, 0.35, 0.20])
                ),
                [[], [(1, 2.0), (2, 1.0)], [(3, 3.0)], [(1, 1.0), (2, 2.0), (3, 1.0)]],
            ),
            (
                IntegerProbabilisticLatentSemanticIndexingDistribution(
                    [[0.70, 0.10], [0.20, 0.30], [0.10, 0.60]],
                    [[0.80, 0.20], [0.25, 0.75]],
                    [0.55, 0.45],
                    len_dist=IntegerCategoricalDistribution(0, [0.05, 0.15, 0.30, 0.30, 0.20]),
                ),
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
                [[(0, 2.0), (1, 1.0)], [(2, 3.0)], [(1, 1.0), (2, 1.0)]],
            ),
            (OptionalDistribution(GaussianDistribution(0.5, 1.2), p=0.25), [None, -1.0, 0.0, 2.0]),
        ]

    @staticmethod
    def stacked_mixture_cases():
        def choose_sign(x):
            return 0 if x < 0.0 else 1

        return [
            (
                "exponential",
                MixtureDistribution(
                    [
                        ExponentialDistribution(0.8),
                        ExponentialDistribution(2.0),
                        ExponentialDistribution(4.0),
                    ],
                    [0.25, 0.45, 0.30],
                ),
                np.asarray([0.1, 0.4, 1.0, 2.5, 4.0]),
            ),
            (
                "poisson",
                MixtureDistribution(
                    [
                        PoissonDistribution(1.2),
                        PoissonDistribution(4.0),
                        PoissonDistribution(8.0),
                    ],
                    [0.30, 0.50, 0.20],
                ),
                [0, 1, 2, 5, 8, 11],
            ),
            (
                "bernoulli",
                MixtureDistribution(
                    [
                        BernoulliDistribution(0.2),
                        BernoulliDistribution(0.8),
                    ],
                    [0.55, 0.45],
                ),
                [False, True, True, False, True, False],
            ),
            (
                "gamma",
                MixtureDistribution(
                    [
                        GammaDistribution(1.5, 0.7),
                        GammaDistribution(3.0, 1.1),
                        GammaDistribution(6.0, 0.5),
                    ],
                    [0.25, 0.40, 0.35],
                ),
                np.asarray([0.25, 0.75, 1.5, 3.0, 5.0]),
            ),
            (
                "log_gaussian",
                MixtureDistribution(
                    [
                        LogGaussianDistribution(-0.5, 0.25),
                        LogGaussianDistribution(0.7, 0.9),
                    ],
                    [0.35, 0.65],
                ),
                np.asarray([0.3, 0.8, 1.0, 2.5, 5.0]),
            ),
            (
                "geometric",
                MixtureDistribution(
                    [
                        GeometricDistribution(0.25),
                        GeometricDistribution(0.7),
                    ],
                    [0.40, 0.60],
                ),
                [1, 2, 3, 5, 8],
            ),
            (
                "negative_binomial",
                MixtureDistribution(
                    [
                        NegativeBinomialDistribution(1.5, 0.35),
                        NegativeBinomialDistribution(4.0, 0.65),
                    ],
                    [0.45, 0.55],
                ),
                [0, 1, 2, 4, 7, 10],
            ),
            (
                "binomial",
                MixtureDistribution(
                    [
                        BinomialDistribution(0.25, 5, min_val=2),
                        BinomialDistribution(0.60, 5, min_val=2),
                    ],
                    [0.50, 0.50],
                ),
                [2, 3, 4, 5, 6, 7],
            ),
            (
                "diagonal_gaussian",
                MixtureDistribution(
                    [
                        DiagonalGaussianDistribution([-1.0, 0.5], [0.6, 1.4]),
                        DiagonalGaussianDistribution([2.0, -0.5], [1.2, 0.8]),
                    ],
                    [0.40, 0.60],
                ),
                [[-1.5, 0.2], [-0.5, 1.0], [0.5, 0.0], [2.5, -1.0], [3.0, 0.5]],
            ),
            (
                "multivariate_gaussian",
                MixtureDistribution(
                    [
                        MultivariateGaussianDistribution([-1.0, 0.5], [[0.8, 0.2], [0.2, 1.4]]),
                        MultivariateGaussianDistribution([2.0, -0.5], [[1.5, -0.3], [-0.3, 0.9]]),
                    ],
                    [0.45, 0.55],
                ),
                [[-1.5, 0.2], [-0.5, 1.0], [0.5, 0.0], [2.5, -1.0], [3.0, 0.5]],
            ),
            (
                "student_t",
                MixtureDistribution(
                    [
                        StudentTDistribution(4.0, loc=-1.0, scale=0.8),
                        StudentTDistribution(8.0, loc=2.0, scale=1.5),
                    ],
                    [0.45, 0.55],
                ),
                np.asarray([-4.0, -1.0, 0.0, 2.0, 6.0]),
            ),
            (
                "logistic",
                MixtureDistribution(
                    [
                        LogisticDistribution(loc=-1.5, scale=0.7),
                        LogisticDistribution(loc=2.0, scale=1.2),
                    ],
                    [0.35, 0.65],
                ),
                np.asarray([-4.0, -1.0, 0.0, 1.0, 4.0]),
            ),
            (
                "weibull",
                MixtureDistribution(
                    [
                        WeibullDistribution(0.8, 1.0),
                        WeibullDistribution(2.0, 3.0),
                    ],
                    [0.40, 0.60],
                ),
                np.asarray([0.2, 0.8, 1.5, 3.0, 5.0]),
            ),
            (
                "rayleigh",
                MixtureDistribution(
                    [
                        RayleighDistribution(0.8),
                        RayleighDistribution(2.0),
                    ],
                    [0.30, 0.70],
                ),
                np.asarray([0.1, 0.5, 1.0, 2.0, 3.5]),
            ),
            (
                "pareto",
                MixtureDistribution(
                    [
                        ParetoDistribution(1.0, 2.5),
                        ParetoDistribution(2.0, 4.0),
                    ],
                    [0.55, 0.45],
                ),
                np.asarray([1.1, 1.8, 2.5, 4.0, 7.0]),
            ),
            (
                "uniform",
                MixtureDistribution(
                    [
                        UniformDistribution(-2.0, 2.0),
                        UniformDistribution(0.0, 5.0),
                    ],
                    [0.45, 0.55],
                ),
                np.asarray([-1.5, 0.5, 1.5, 3.0, 4.5]),
            ),
            (
                "integer_categorical",
                MixtureDistribution(
                    [
                        IntegerCategoricalDistribution(0, [0.55, 0.20, 0.15, 0.10]),
                        IntegerCategoricalDistribution(0, [0.10, 0.20, 0.30, 0.40]),
                    ],
                    [0.50, 0.50],
                ),
                [0, 1, 2, 3, 2, 1],
            ),
            (
                "integer_uniform_spike",
                MixtureDistribution(
                    [
                        IntegerUniformSpikeDistribution(k=1, num_vals=4, p=0.65, min_val=0),
                        IntegerUniformSpikeDistribution(k=3, num_vals=4, p=0.45, min_val=0),
                    ],
                    [0.40, 0.60],
                ),
                [0, 1, 2, 3, 1, 3, 2],
            ),
            (
                "integer_bernoulli_set",
                MixtureDistribution(
                    [
                        IntegerBernoulliSetDistribution(np.log([0.20, 0.55, 0.80])),
                        IntegerBernoulliSetDistribution(np.log([0.70, 0.25, 0.35])),
                    ],
                    [0.45, 0.55],
                ),
                [[], [0], [1, 2], [0, 2], [2], [0, 1]],
            ),
            (
                "dirac_length",
                MixtureDistribution(
                    [
                        DiracLengthMixtureDistribution(
                            len_dist=IntegerCategoricalDistribution(0, [0.10, 0.25, 0.40, 0.25]), p=0.65, v=0
                        ),
                        DiracLengthMixtureDistribution(
                            len_dist=IntegerCategoricalDistribution(0, [0.45, 0.20, 0.20, 0.15]), p=0.35, v=0
                        ),
                    ],
                    [0.45, 0.55],
                ),
                [0, 1, 2, 0, 3, 1, 0],
            ),
            (
                "markov_chain",
                MixtureDistribution(
                    [
                        MarkovChainDistribution(
                            {"a": 0.70, "b": 0.30},
                            {"a": {"a": 0.20, "b": 0.80}, "b": {"a": 0.60, "b": 0.40}},
                            len_dist=IntegerCategoricalDistribution(0, [0.10, 0.20, 0.40, 0.30]),
                        ),
                        MarkovChainDistribution(
                            {"a": 0.40, "b": 0.60},
                            {"a": {"a": 0.50, "b": 0.50}, "b": {"a": 0.25, "b": 0.75}},
                            len_dist=IntegerCategoricalDistribution(0, [0.25, 0.25, 0.30, 0.20]),
                        ),
                    ],
                    [0.45, 0.55],
                ),
                [[], ["a"], ["a", "b", "a"], ["b", "a"], ["b", "b", "a"]],
            ),
            (
                "von_mises_fisher",
                MixtureDistribution(
                    [
                        VonMisesFisherDistribution([1.0, 0.0, 0.0], 2.0),
                        VonMisesFisherDistribution([0.0, 1.0, 0.0], 4.0),
                    ],
                    [0.45, 0.55],
                ),
                np.asarray(
                    [
                        [1.0, 0.0, 0.0],
                        [0.0, 1.0, 0.0],
                        [0.0, 0.0, 1.0],
                        [1.0 / np.sqrt(2.0), 1.0 / np.sqrt(2.0), 0.0],
                        [0.0, 1.0 / np.sqrt(2.0), 1.0 / np.sqrt(2.0)],
                    ]
                ),
            ),
            (
                "spearman_ranking",
                MixtureDistribution(
                    [
                        SpearmanRankingDistribution([0, 1, 2], rho=0.7),
                        SpearmanRankingDistribution([2, 1, 0], rho=1.4),
                    ],
                    [0.45, 0.55],
                ),
                [[0, 1, 2], [0, 2, 1], [1, 0, 2], [2, 1, 0], [2, 0, 1]],
            ),
            (
                "integer_markov_chain",
                MixtureDistribution(
                    [
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
                        IntegerMarkovChainDistribution(
                            3,
                            [[0.30, 0.50, 0.20], [0.20, 0.25, 0.55], [0.15, 0.35, 0.50]],
                            lag=1,
                            init_dist=SequenceDistribution(
                                IntegerCategoricalDistribution(0, [0.50, 0.20, 0.30]),
                                len_dist=IntegerCategoricalDistribution(1, [1.0]),
                            ),
                            len_dist=IntegerCategoricalDistribution(0, [0.25, 0.25, 0.30, 0.20]),
                        ),
                    ],
                    [0.45, 0.55],
                ),
                [[], [0], [0, 1, 2], [2, 2, 1], [1, 0]],
            ),
            (
                "categorical",
                MixtureDistribution(
                    [
                        CategoricalDistribution({"red": 0.55, "green": 0.30, "blue": 0.15}),
                        CategoricalDistribution({"red": 0.10, "green": 0.35, "blue": 0.55}),
                    ],
                    [0.45, 0.55],
                ),
                ["red", "green", "blue", "blue", "red", "green"],
            ),
            (
                "bernoulli_set",
                MixtureDistribution(
                    [
                        BernoulliSetDistribution({"a": 0.25, "b": 0.60, "c": 0.80}, min_prob=0.0),
                        BernoulliSetDistribution({"a": 0.70, "b": 0.30, "c": 0.45}, min_prob=0.0),
                    ],
                    [0.45, 0.55],
                ),
                [[], ["a"], ["b", "c"], ["a", "c"], ["c"], ["a", "b"]],
            ),
            (
                "indian_buffet_process",
                MixtureDistribution(
                    [
                        IndianBuffetProcessDistribution(
                            4, alpha=1.2, feature_probs=[0.15, 0.45, 0.75, 0.35], data_format="sparse"
                        ),
                        IndianBuffetProcessDistribution(
                            4, alpha=2.5, feature_probs=[0.65, 0.25, 0.30, 0.80], data_format="sparse"
                        ),
                    ],
                    [0.45, 0.55],
                ),
                [[], [0, 2], [1], [0, 1, 3], [2, 3], [0, 3]],
            ),
            (
                "integer_bernoulli_edit",
                MixtureDistribution(
                    [
                        IntegerBernoulliEditDistribution(
                            np.log([[0.20, 0.75], [0.45, 0.30], [0.65, 0.55]]),
                            init_dist=IntegerBernoulliSetDistribution(np.log([0.30, 0.55, 0.75])),
                        ),
                        IntegerBernoulliEditDistribution(
                            np.log([[0.70, 0.25], [0.35, 0.80], [0.25, 0.40]]),
                            init_dist=IntegerBernoulliSetDistribution(np.log([0.60, 0.20, 0.40])),
                        ),
                    ],
                    [0.45, 0.55],
                ),
                [([0], [0, 1]), ([1, 2], [2]), ([], [0, 2]), ([0, 2], []), ([1], [0, 1])],
            ),
            (
                "integer_step_bernoulli_edit",
                MixtureDistribution(
                    [
                        IntegerStepBernoulliEditDistribution(
                            np.log([[0.20, 0.75], [0.45, 0.30], [0.65, 0.55]]),
                            init_dist=IntegerBernoulliSetDistribution(np.log([0.30, 0.55, 0.75])),
                        ),
                        IntegerStepBernoulliEditDistribution(
                            np.log([[0.70, 0.25], [0.35, 0.80], [0.25, 0.40]]),
                            init_dist=IntegerBernoulliSetDistribution(np.log([0.60, 0.20, 0.40])),
                        ),
                    ],
                    [0.45, 0.55],
                ),
                [([0], [0, 1]), ([1, 2], [2]), ([], [0, 2]), ([0, 2], []), ([1], [0, 1])],
            ),
            (
                "beta",
                MixtureDistribution(
                    [
                        BetaDistribution(1.5, 5.0),
                        BetaDistribution(6.0, 2.0),
                    ],
                    [0.40, 0.60],
                ),
                np.asarray([0.05, 0.2, 0.45, 0.7, 0.9]),
            ),
            (
                "dirichlet",
                MixtureDistribution(
                    [
                        DirichletDistribution([2.0, 3.0, 4.0]),
                        DirichletDistribution([4.0, 2.0, 3.0]),
                    ],
                    [0.40, 0.60],
                ),
                np.asarray([[0.2, 0.3, 0.5], [0.4, 0.4, 0.2], [0.1, 0.7, 0.2], [0.6, 0.2, 0.2]]),
            ),
            (
                "laplace",
                MixtureDistribution(
                    [
                        LaplaceDistribution(-2.0, 0.8),
                        LaplaceDistribution(2.0, 1.5),
                    ],
                    [0.35, 0.65],
                ),
                np.asarray([-5.0, -2.0, -0.5, 1.0, 3.5]),
            ),
            (
                "null",
                MixtureDistribution(
                    [
                        NullDistribution(),
                        NullDistribution(),
                    ],
                    [0.40, 0.60],
                ),
                [None, "anything", 3.0, {"x": 1}],
            ),
            (
                "point_mass",
                MixtureDistribution(
                    [
                        PointMassDistribution("fixed"),
                        PointMassDistribution("fixed"),
                    ],
                    [0.40, 0.60],
                ),
                ["fixed", "other", "fixed", "fixed", "miss"],
            ),
            (
                "optional_gaussian",
                MixtureDistribution(
                    [
                        OptionalDistribution(GaussianDistribution(-1.0, 0.8), p=0.20),
                        OptionalDistribution(GaussianDistribution(2.0, 1.5), p=0.45),
                    ],
                    [0.40, 0.60],
                ),
                [None, -1.5, 0.0, None, 2.0, 3.5],
            ),
            (
                "ignored_gaussian",
                MixtureDistribution(
                    [
                        IgnoredDistribution(GaussianDistribution(-1.0, 0.8)),
                        IgnoredDistribution(GaussianDistribution(2.0, 1.5)),
                    ],
                    [0.40, 0.60],
                ),
                np.asarray([-2.0, -0.5, 0.0, 1.5, 3.0]),
            ),
            (
                "weighted_gaussian",
                MixtureDistribution(
                    [
                        WeightedDistribution(GaussianDistribution(-1.0, 0.8)),
                        WeightedDistribution(GaussianDistribution(2.0, 1.5)),
                    ],
                    [0.40, 0.60],
                ),
                [(-1.5, 0.5), (-0.5, 2.0), (0.0, 1.0), (2.0, 1.5), (3.5, 0.75)],
            ),
            (
                "transform_gaussian",
                MixtureDistribution(
                    [
                        TransformDistribution(
                            GaussianDistribution(-1.0, 0.8), transform=AffineTransform(loc=1.0, scale=2.0)
                        ),
                        TransformDistribution(
                            GaussianDistribution(2.0, 1.5), transform=AffineTransform(loc=1.0, scale=2.0)
                        ),
                    ],
                    [0.35, 0.65],
                ),
                [-3.0, -1.0, 1.0, 5.0, 7.0],
            ),
            (
                "sequence_gaussian",
                MixtureDistribution(
                    [
                        SequenceDistribution(
                            GaussianDistribution(-1.0, 0.8),
                            len_dist=IntegerCategoricalDistribution(0, [0.10, 0.20, 0.35, 0.35]),
                        ),
                        SequenceDistribution(
                            GaussianDistribution(2.0, 1.5),
                            len_dist=IntegerCategoricalDistribution(0, [0.25, 0.25, 0.30, 0.20]),
                        ),
                    ],
                    [0.45, 0.55],
                ),
                [[], [-1.2], [-0.5, 1.5], [2.0, 2.5, -0.2], [1.0, -1.0]],
            ),
            (
                "multinomial_categorical",
                MixtureDistribution(
                    [
                        MultinomialDistribution(
                            CategoricalDistribution({"a": 0.55, "b": 0.30, "c": 0.15}),
                            len_dist=IntegerCategoricalDistribution(0, [0.10, 0.20, 0.35, 0.35]),
                        ),
                        MultinomialDistribution(
                            CategoricalDistribution({"a": 0.15, "b": 0.35, "c": 0.50}),
                            len_dist=IntegerCategoricalDistribution(0, [0.25, 0.25, 0.30, 0.20]),
                        ),
                    ],
                    [0.45, 0.55],
                ),
                [[], [("a", 2.0)], [("a", 1.0), ("b", 1.0)], [("c", 2.0), ("b", 1.0)], [("a", 1.0), ("c", 2.0)]],
            ),
            (
                "integer_multinomial",
                MixtureDistribution(
                    [
                        IntegerMultinomialDistribution(
                            0, [0.55, 0.30, 0.15], len_dist=IntegerCategoricalDistribution(0, [0.10, 0.20, 0.35, 0.35])
                        ),
                        IntegerMultinomialDistribution(
                            0, [0.15, 0.35, 0.50], len_dist=IntegerCategoricalDistribution(0, [0.25, 0.25, 0.30, 0.20])
                        ),
                    ],
                    [0.45, 0.55],
                ),
                [[], [(0, 2.0)], [(0, 1.0), (1, 1.0)], [(2, 2.0), (1, 1.0)], [(0, 1.0), (2, 2.0)]],
            ),
            (
                "conditional_gaussian",
                MixtureDistribution(
                    [
                        ConditionalDistribution(
                            {
                                "a": GaussianDistribution(-1.0, 0.8),
                                "b": GaussianDistribution(1.0, 0.9),
                            },
                            given_dist=CategoricalDistribution({"a": 0.55, "b": 0.45}),
                        ),
                        ConditionalDistribution(
                            {
                                "a": GaussianDistribution(-0.5, 1.2),
                                "b": GaussianDistribution(2.5, 1.4),
                            },
                            given_dist=CategoricalDistribution({"a": 0.30, "b": 0.70}),
                        ),
                    ],
                    [0.45, 0.55],
                ),
                [("a", -1.5), ("b", 1.25), ("a", -0.5), ("b", 2.0), ("a", 0.0)],
            ),
            (
                "record_gaussian_poisson",
                MixtureDistribution(
                    [
                        RecordDistribution(
                            {
                                "x": GaussianDistribution(-1.0, 0.8),
                                "count": PoissonDistribution(2.0),
                            }
                        ),
                        RecordDistribution(
                            {
                                "x": GaussianDistribution(2.0, 1.5),
                                "count": PoissonDistribution(6.0),
                            }
                        ),
                    ],
                    [0.45, 0.55],
                ),
                [
                    {"x": -1.2, "count": 1},
                    {"x": 0.0, "count": 2},
                    {"x": 1.5, "count": 5},
                    {"x": 2.5, "count": 7},
                    {"x": 3.0, "count": 4},
                ],
            ),
            (
                "select_gaussian_by_sign",
                MixtureDistribution(
                    [
                        SelectDistribution(
                            [
                                GaussianDistribution(-2.0, 0.7),
                                GaussianDistribution(1.0, 0.9),
                            ],
                            choose_sign,
                        ),
                        SelectDistribution(
                            [
                                GaussianDistribution(-0.5, 1.1),
                                GaussianDistribution(3.0, 1.4),
                            ],
                            choose_sign,
                        ),
                    ],
                    [0.45, 0.55],
                ),
                np.asarray([-3.0, -1.0, -0.25, 0.5, 2.0, 4.0]),
            ),
            (
                "composite_with_optional",
                MixtureDistribution(
                    [
                        CompositeDistribution(
                            (
                                GaussianDistribution(-1.0, 0.8),
                                PoissonDistribution(2.0),
                                OptionalDistribution(GaussianDistribution(0.0, 1.2), p=0.25),
                            )
                        ),
                        CompositeDistribution(
                            (
                                GaussianDistribution(2.0, 1.5),
                                PoissonDistribution(6.0),
                                OptionalDistribution(GaussianDistribution(3.0, 0.7), p=0.40),
                            )
                        ),
                    ],
                    [0.45, 0.55],
                ),
                [(-1.2, 1, None), (0.0, 2, 0.5), (1.5, 5, None), (2.5, 7, 2.8), (3.0, 4, 3.5)],
            ),
        ]

    def assert_suff_stat_allclose(self, actual, expected):
        if isinstance(expected, dict):
            self.assertIsInstance(actual, dict)
            self.assertEqual(set(actual.keys()), set(expected.keys()))
            for key in expected:
                self.assert_suff_stat_allclose(actual[key], expected[key])
        elif isinstance(expected, tuple):
            self.assertIsInstance(actual, tuple)
            self.assertEqual(len(actual), len(expected))
            for got, exp in zip(actual, expected):
                self.assert_suff_stat_allclose(got, exp)
        elif isinstance(expected, list):
            if isinstance(actual, list):
                self.assertEqual(len(actual), len(expected))
                for got, exp in zip(actual, expected):
                    self.assert_suff_stat_allclose(got, exp)
            else:
                np.testing.assert_allclose(actual, expected, rtol=1.0e-10, atol=1.0e-10)
        elif expected is None:
            self.assertIsNone(actual)
        else:
            np.testing.assert_allclose(actual, expected, rtol=1.0e-10, atol=1.0e-10)

    def test_numpy_backend_matches_seq_log_density_for_declared_leaves(self):
        for dist, data in self.backend_leaf_cases():
            with self.subTest(dist=type(dist).__name__):
                enc = dist.dist_to_encoder().seq_encode(data)
                np.testing.assert_allclose(
                    backend_seq_log_density(dist, enc, NUMPY_ENGINE),
                    dist.seq_log_density(enc),
                    rtol=1.0e-12,
                    atol=1.0e-12,
                )

    def test_numpy_backend_respects_precision_policy(self):
        dist = GaussianDistribution(0.5, 1.7)
        enc = dist.dist_to_encoder().seq_encode(np.asarray([-1.0, 0.0, 2.0]))
        actual = backend_seq_log_density(dist, enc, NumpyEngine(dtype="float32"))

        self.assertEqual(actual.dtype, np.dtype("float32"))
        np.testing.assert_allclose(actual, dist.seq_log_density(enc), rtol=1.0e-6, atol=1.0e-6)

    def test_generated_stacked_scores_match_legacy_paths(self):
        cases = [
            (
                [GaussianDistribution(-1.0, 0.6), GaussianDistribution(2.0, 1.5)],
                np.asarray([-2.0, -0.5, 0.0, 1.5, 3.0]),
            ),
            (
                [ExponentialDistribution(0.75), ExponentialDistribution(3.0)],
                np.asarray([0.0, 0.25, 1.0, 2.5, 4.0]),
            ),
            (
                [GammaDistribution(1.5, 0.7), GammaDistribution(4.0, 1.2)],
                np.asarray([0.2, 0.8, 1.5, 3.0, 5.0]),
            ),
            (
                [LogGaussianDistribution(-0.5, 0.25), LogGaussianDistribution(0.7, 0.9)],
                np.asarray([0.3, 0.8, 1.0, 2.5, 5.0]),
            ),
            (
                [RayleighDistribution(0.8), RayleighDistribution(2.0)],
                np.asarray([0.1, 0.5, 1.0, 2.0, 3.5]),
            ),
            (
                [BetaDistribution(1.5, 5.0), BetaDistribution(6.0, 2.0)],
                np.asarray([0.05, 0.2, 0.45, 0.7, 0.9]),
            ),
            (
                [BinomialDistribution(0.25, 5, min_val=2), BinomialDistribution(0.60, 5, min_val=2)],
                [2, 3, 4, 5, 6, 7],
            ),
            (
                [BernoulliDistribution(0.25), BernoulliDistribution(0.70)],
                [False, True, True, False],
            ),
            (
                [PoissonDistribution(1.5), PoissonDistribution(4.0)],
                [0, 1, 3, 5],
            ),
            (
                [GeometricDistribution(0.25), GeometricDistribution(0.70)],
                [1, 2, 3, 5, 8],
            ),
            (
                [NegativeBinomialDistribution(1.5, 0.35), NegativeBinomialDistribution(4.0, 0.65)],
                [0, 1, 2, 4, 7, 10],
            ),
            (
                [StudentTDistribution(4.0, loc=-1.0, scale=0.8), StudentTDistribution(8.0, loc=2.0, scale=1.5)],
                np.asarray([-4.0, -1.0, 0.0, 2.0, 6.0]),
            ),
            (
                [LogisticDistribution(loc=-1.5, scale=0.7), LogisticDistribution(loc=2.0, scale=1.2)],
                np.asarray([-4.0, -1.0, 0.0, 1.0, 4.0]),
            ),
            (
                [WeibullDistribution(0.8, 1.0), WeibullDistribution(2.0, 3.0)],
                np.asarray([0.2, 0.8, 1.5, 3.0, 5.0]),
            ),
            (
                [ParetoDistribution(1.0, 2.5), ParetoDistribution(2.0, 4.0)],
                np.asarray([1.1, 1.8, 2.5, 4.0, 7.0]),
            ),
            (
                [UniformDistribution(-2.0, 2.0), UniformDistribution(0.0, 5.0)],
                np.asarray([-1.5, 0.5, 1.5, 3.0, 4.5]),
            ),
            (
                [
                    DiagonalGaussianDistribution([-1.0, 0.5], [0.6, 1.4]),
                    DiagonalGaussianDistribution([2.0, -0.5], [1.2, 0.8]),
                ],
                [[-1.5, 0.2], [-0.5, 1.0], [0.5, 0.0], [2.5, -1.0], [3.0, 0.5]],
            ),
            (
                [LaplaceDistribution(-2.0, 0.8), LaplaceDistribution(2.0, 1.5)],
                np.asarray([-5.0, -2.0, -0.5, 1.0, 3.5]),
            ),
        ]

        for components, data in cases:
            with self.subTest(dist=type(components[0]).__name__):
                enc = components[0].dist_to_encoder().seq_encode(data)
                params = generated_stacked_params(components, NUMPY_ENGINE)
                actual = generated_stacked_log_density(enc, params, NUMPY_ENGINE)
                expected = np.column_stack([component.seq_log_density(enc) for component in components])
                np.testing.assert_allclose(actual, expected, rtol=1.0e-12, atol=1.0e-12)

    def test_generated_resident_statistics_preserve_matrix_moments(self):
        dist = MultivariateGaussianDistribution([0.0, 1.0], [[1.0, 0.2], [0.2, 2.0]])
        data = np.asarray([[-1.0, 0.0], [0.5, -1.0], [2.0, 1.5]])
        weights = np.asarray([[0.2, 0.8], [1.0, 0.5], [0.4, 1.2]])
        enc = dist.dist_to_encoder().seq_encode(data)

        sum_x, sum_xx, counts = generated_stacked_sufficient_statistics(
            enc, weights, {"__pysp_dist_type__": MultivariateGaussianDistribution}, NUMPY_ENGINE
        )

        expected_sum = weights.T.dot(data)
        expected_sum2 = np.stack(
            [
                sum(weights[i, k] * np.outer(data[i], data[i]) for i in range(len(data)))
                for k in range(weights.shape[1])
            ],
            axis=0,
        )
        np.testing.assert_allclose(sum_x, expected_sum, rtol=1.0e-12, atol=1.0e-12)
        np.testing.assert_allclose(sum_xx, expected_sum2, rtol=1.0e-12, atol=1.0e-12)
        np.testing.assert_allclose(counts, weights.sum(axis=0), rtol=1.0e-12, atol=1.0e-12)

    def test_wrapper_stacked_routes_delegate_to_generated_child_stackers(self):
        optional_components = [
            OptionalDistribution(GaussianDistribution(-1.0, 0.8), p=0.20),
            OptionalDistribution(GaussianDistribution(2.0, 1.5), p=0.45),
        ]
        optional_data = [None, -1.5, 0.0, None, 2.0, 3.5]
        optional_enc = optional_components[0].dist_to_encoder().seq_encode(optional_data)
        optional_params = OptionalDistribution.backend_stacked_params(optional_components, NUMPY_ENGINE)

        self.assertEqual(optional_params["child_route"].strategy, "generated")
        optional_actual = OptionalDistribution.backend_stacked_log_density(optional_enc, optional_params, NUMPY_ENGINE)
        optional_expected = np.column_stack(
            [component.seq_log_density(optional_enc) for component in optional_components]
        )
        np.testing.assert_allclose(optional_actual, optional_expected, rtol=1.0e-12, atol=1.0e-12)

        composite_components = [
            CompositeDistribution(
                (
                    GaussianDistribution(-1.0, 0.8),
                    DiagonalGaussianDistribution([0.0, 1.0], [0.6, 1.4]),
                )
            ),
            CompositeDistribution(
                (
                    GaussianDistribution(2.0, 1.5),
                    DiagonalGaussianDistribution([2.0, -0.5], [1.2, 0.8]),
                )
            ),
        ]
        composite_data = [
            (-1.2, [-0.5, 0.8]),
            (0.0, [0.5, 1.0]),
            (2.5, [2.5, -1.0]),
            (3.0, [3.0, 0.5]),
        ]
        composite_enc = composite_components[0].dist_to_encoder().seq_encode(composite_data)
        composite_params = CompositeDistribution.backend_stacked_params(composite_components, NUMPY_ENGINE)

        self.assertEqual(tuple(route.strategy for route in composite_params["children"]), ("generated", "generated"))
        composite_actual = CompositeDistribution.backend_stacked_log_density(
            composite_enc, composite_params, NUMPY_ENGINE
        )
        composite_expected = np.column_stack(
            [component.seq_log_density(composite_enc) for component in composite_components]
        )
        np.testing.assert_allclose(composite_actual, composite_expected, rtol=1.0e-12, atol=1.0e-12)

    def test_wrapper_stacked_routes_keep_explicit_table_children(self):
        components = [
            CompositeDistribution((CategoricalDistribution({"a": 0.6, "b": 0.3, "c": 0.1}),)),
            CompositeDistribution((CategoricalDistribution({"a": 0.2, "b": 0.5, "c": 0.3}),)),
        ]
        data = [("a",), ("b",), ("c",), ("b",)]
        enc = components[0].dist_to_encoder().seq_encode(data)
        params = CompositeDistribution.backend_stacked_params(components, NUMPY_ENGINE)

        self.assertEqual(params["children"][0].strategy, "explicit")
        actual = CompositeDistribution.backend_stacked_log_density(enc, params, NUMPY_ENGINE)
        expected = np.column_stack([component.seq_log_density(enc) for component in components])
        np.testing.assert_allclose(actual, expected, rtol=1.0e-12, atol=1.0e-12)

    def test_numpy_backend_composes_composite_and_mixture(self):
        comp0 = CompositeDistribution(
            (
                GaussianDistribution(-1.0, 0.7),
                PoissonDistribution(2.0),
            )
        )
        comp1 = CompositeDistribution(
            (
                GaussianDistribution(2.0, 1.2),
                PoissonDistribution(5.0),
            )
        )
        dist = MixtureDistribution([comp0, comp1], [0.35, 0.65])
        data = dist.sampler(seed=4).sample(size=40)
        enc = dist.dist_to_encoder().seq_encode(data)

        np.testing.assert_allclose(
            backend_seq_log_density(dist, enc, NUMPY_ENGINE), dist.seq_log_density(enc), rtol=1.0e-12, atol=1.0e-12
        )
        np.testing.assert_allclose(
            backend_seq_component_log_density(dist, enc, NUMPY_ENGINE),
            dist.seq_component_log_density(enc),
            rtol=1.0e-12,
            atol=1.0e-12,
        )

    def test_numpy_lda_backend_component_scores_match_legacy_path(self):
        dist = LDADistribution(
            [
                IntegerCategoricalDistribution(0, [0.70, 0.20, 0.10]),
                IntegerCategoricalDistribution(0, [0.10, 0.30, 0.60]),
            ],
            [0.8, 1.3],
            len_dist=IntegerCategoricalDistribution(0, [0.05, 0.15, 0.30, 0.30, 0.20]),
            gamma_threshold=1.0e-10,
        )
        data = [[(0, 2.0), (1, 1.0)], [(2, 3.0)], [(1, 1.0), (2, 1.0)]]
        enc = dist.dist_to_encoder().seq_encode(data)

        np.testing.assert_allclose(
            backend_seq_component_log_density(dist, enc, NUMPY_ENGINE),
            dist.seq_component_log_density(enc),
            rtol=1.0e-12,
            atol=1.0e-12,
        )

    def test_numpy_hmm_backend_preserves_numba_encoded_fallback(self):
        dist = HiddenMarkovModelDistribution(
            [GaussianDistribution(-1.0, 0.8), GaussianDistribution(2.0, 1.5)],
            [0.6, 0.4],
            [[0.75, 0.25], [0.20, 0.80]],
            len_dist=IntegerCategoricalDistribution(0, [0.1, 0.2, 0.4, 0.3]),
            use_numba=True,
        )
        data = [[], [-1.2], [-0.5, 1.5], [2.0, 2.5, -0.2]]
        enc = dist.dist_to_encoder().seq_encode(data)

        np.testing.assert_allclose(
            backend_seq_log_density(dist, enc, NUMPY_ENGINE), dist.seq_log_density(enc), rtol=1.0e-12, atol=1.0e-12
        )
        np.testing.assert_allclose(
            dist.kernel(engine=NUMPY_ENGINE).score(enc), dist.seq_log_density(enc), rtol=1.0e-12, atol=1.0e-12
        )

    @unittest.skipUnless(HAS_TORCH, "torch is not installed")
    def test_torch_backend_matches_seq_log_density_for_composite_mixture(self):
        dist = MixtureDistribution(
            [
                CompositeDistribution((GaussianDistribution(-2.0, 1.0), ExponentialDistribution(1.5))),
                CompositeDistribution((GaussianDistribution(2.0, 0.8), ExponentialDistribution(3.0))),
            ],
            [0.45, 0.55],
        )
        data = dist.sampler(seed=5).sample(size=50)
        enc = dist.dist_to_encoder().seq_encode(data)
        engine = TorchEngine(dtype=torch.float64)

        actual = backend_seq_log_density(dist, enc, engine)
        self.assertTrue(isinstance(actual, torch.Tensor))
        np.testing.assert_allclose(actual.detach().cpu().numpy(), dist.seq_log_density(enc), rtol=1.0e-10, atol=1.0e-10)

        kernel = dist.kernel(engine=engine)
        kernel_actual = kernel.score(enc)
        self.assertTrue(isinstance(kernel_actual, torch.Tensor))
        np.testing.assert_allclose(
            kernel_actual.detach().cpu().numpy(), dist.seq_log_density(enc), rtol=1.0e-10, atol=1.0e-10
        )

        kernel_components = kernel.component_scores(enc)
        self.assertTrue(isinstance(kernel_components, torch.Tensor))
        np.testing.assert_allclose(
            kernel_components.detach().cpu().numpy(), dist.seq_component_log_density(enc), rtol=1.0e-10, atol=1.0e-10
        )

    @unittest.skipUnless(HAS_TORCH, "torch is not installed")
    def test_torch_backend_matches_seq_log_density_for_declared_leaves(self):
        engine = TorchEngine(dtype=torch.float64)
        for dist, data in self.backend_leaf_cases():
            with self.subTest(dist=type(dist).__name__):
                enc = dist.dist_to_encoder().seq_encode(data)
                actual = backend_seq_log_density(dist, enc, engine)
                self.assertTrue(isinstance(actual, torch.Tensor))
                np.testing.assert_allclose(
                    actual.detach().cpu().numpy(), dist.seq_log_density(enc), rtol=1.0e-10, atol=1.0e-10
                )

                kernel_actual = dist.kernel(engine=engine).score(enc)
                self.assertTrue(isinstance(kernel_actual, torch.Tensor))
                np.testing.assert_allclose(
                    kernel_actual.detach().cpu().numpy(), dist.seq_log_density(enc), rtol=1.0e-10, atol=1.0e-10
                )

    @unittest.skipUnless(HAS_TORCH, "torch is not installed")
    def test_distribution_owned_gaussian_math_is_autograd_differentiable(self):
        engine = TorchEngine(dtype=torch.float64)
        x = engine.asarray(np.asarray([-1.0, 0.0, 1.5, 2.0]))
        mu = torch.tensor(0.25, dtype=torch.float64, requires_grad=True)
        log_sigma2 = torch.tensor(0.1, dtype=torch.float64, requires_grad=True)

        ll = GaussianDistribution.backend_log_density_from_params(x, mu, log_sigma2.exp(), engine).sum()
        (-ll).backward()

        self.assertIsNotNone(mu.grad)
        self.assertIsNotNone(log_sigma2.grad)
        self.assertTrue(torch.isfinite(mu.grad))
        self.assertTrue(torch.isfinite(log_sigma2.grad))

    @unittest.skipUnless(HAS_TORCH, "torch is not installed")
    def test_torch_stacked_gaussian_mixture_kernel_matches_legacy_paths(self):
        dist = MixtureDistribution(
            [
                GaussianDistribution(-2.0, 0.7),
                GaussianDistribution(0.0, 1.2),
                GaussianDistribution(2.5, 0.9),
            ],
            [0.25, 0.35, 0.40],
        )
        data = dist.sampler(seed=13).sample(size=60)
        enc = dist.dist_to_encoder().seq_encode(data)
        engine = TorchEngine(dtype=torch.float64)
        kernel = dist.kernel(engine=engine)

        self.assertIsInstance(kernel, StackedMixtureKernel)
        self.assertTrue(kernel._generated)
        np.testing.assert_allclose(
            kernel.component_scores(enc).detach().cpu().numpy(),
            dist.seq_component_log_density(enc),
            rtol=1.0e-10,
            atol=1.0e-10,
        )
        np.testing.assert_allclose(
            kernel.score(enc).detach().cpu().numpy(), dist.seq_log_density(enc), rtol=1.0e-10, atol=1.0e-10
        )

        replacement = MixtureDistribution(
            [
                GaussianDistribution(-1.5, 0.8),
                GaussianDistribution(0.5, 1.0),
                GaussianDistribution(3.0, 1.3),
            ],
            [0.2, 0.5, 0.3],
        )
        kernel.refresh(replacement)
        np.testing.assert_allclose(
            kernel.score(enc).detach().cpu().numpy(), replacement.seq_log_density(enc), rtol=1.0e-10, atol=1.0e-10
        )

    @unittest.skipUnless(HAS_TORCH, "torch is not installed")
    def test_torch_stacked_gaussian_mixture_accumulates_legacy_stats(self):
        dist = MixtureDistribution(
            [
                GaussianDistribution(-1.0, 0.6),
                GaussianDistribution(1.5, 1.1),
            ],
            [0.45, 0.55],
        )
        data = dist.sampler(seed=14).sample(size=50)
        weights = np.linspace(0.25, 1.25, len(data))
        enc = dist.dist_to_encoder().seq_encode(data)
        est = dist.estimator()
        engine = TorchEngine(dtype=torch.float64)
        kernel = dist.kernel(engine=engine, estimator=est)

        self.assertIsInstance(kernel, StackedMixtureKernel)
        actual = kernel.accumulate(enc, engine.asarray(weights))
        legacy_acc = est.accumulator_factory().make()
        legacy_acc.seq_update(enc, weights, dist)

        np.testing.assert_allclose(actual[0], legacy_acc.value()[0], rtol=1.0e-10, atol=1.0e-10)
        for got, exp in zip(actual[1], legacy_acc.value()[1]):
            np.testing.assert_allclose(got, exp, rtol=1.0e-10, atol=1.0e-10)

    @unittest.skipUnless(HAS_TORCH, "torch is not installed")
    def test_torch_stacked_gaussian_mixture_resident_stats_match_legacy_stats(self):
        dist = MixtureDistribution(
            [
                GaussianDistribution(-1.0, 0.6),
                GaussianDistribution(1.5, 1.1),
            ],
            [0.45, 0.55],
        )
        data = dist.sampler(seed=16).sample(size=50)
        weights = np.linspace(0.25, 1.25, len(data))
        enc = dist.dist_to_encoder().seq_encode(data)
        est = dist.estimator()
        engine = TorchEngine(dtype=torch.float64)
        kernel = dist.kernel(engine=engine, estimator=est)

        self.assertIsInstance(kernel, StackedMixtureKernel)
        self.assertTrue(kernel.has_resident_accumulate)
        resident = kernel.resident_accumulate(enc, engine.asarray(weights))

        self.assertIsInstance(resident, StackedMixtureResidentStats)
        self.assertTrue(isinstance(resident.component_counts, torch.Tensor))
        self.assertTrue(all(isinstance(stat, torch.Tensor) for stat in resident.component_stats))

        actual = resident.value()
        legacy_acc = est.accumulator_factory().make()
        legacy_acc.seq_update(enc, weights, dist)

        np.testing.assert_allclose(actual[0], legacy_acc.value()[0], rtol=1.0e-10, atol=1.0e-10)
        for got, exp in zip(actual[1], legacy_acc.value()[1]):
            np.testing.assert_allclose(got, exp, rtol=1.0e-10, atol=1.0e-10)

        fitted = resident.estimate(est)
        expected = est.estimate(None, legacy_acc.value())
        np.testing.assert_allclose(fitted.w, expected.w, rtol=1.0e-10, atol=1.0e-10)
        for got, exp in zip(fitted.components, expected.components):
            self.assertAlmostEqual(got.mu, exp.mu, places=10)
            self.assertAlmostEqual(got.sigma2, exp.sigma2, places=10)

    @unittest.skipUnless(HAS_TORCH, "torch is not installed")
    def test_torch_mesh_stacked_gaussian_mixture_uses_dtensor_component_shards(self):
        from torch.distributed.tensor import DTensor, Replicate, Shard

        dist = MixtureDistribution(
            [
                GaussianDistribution(-2.0, 0.7),
                GaussianDistribution(0.0, 1.2),
                GaussianDistribution(2.5, 0.9),
            ],
            [0.25, 0.35, 0.40],
        )
        data = dist.sampler(seed=17).sample(size=40)
        enc = dist.dist_to_encoder().seq_encode(data)
        weights = np.linspace(0.5, 1.5, len(data))
        est = dist.estimator()
        engine = TorchEngine(dtype=torch.float64, mesh=_single_rank_mesh(), shard="components")
        kernel = dist.kernel(engine=engine, estimator=est)

        self.assertIsInstance(kernel, StackedMixtureKernel)
        self.assertIsInstance(kernel.params["mu"], DTensor)
        self.assertIsInstance(kernel.params["mu"].placements[0], Shard)
        self.assertEqual(kernel.params["mu"].placements[0].dim, 0)
        self.assertIsInstance(kernel.log_w, DTensor)
        self.assertIsInstance(kernel.log_w.placements[0], Shard)

        component_scores = kernel.component_scores(enc)
        scores = kernel.score(enc)

        self.assertIsInstance(component_scores, DTensor)
        self.assertIsInstance(component_scores.placements[0], Shard)
        self.assertEqual(component_scores.placements[0].dim, 1)
        self.assertIsInstance(scores, DTensor)
        self.assertIsInstance(scores.placements[0], Replicate)
        np.testing.assert_allclose(
            engine.to_numpy(component_scores), dist.seq_component_log_density(enc), rtol=1.0e-10, atol=1.0e-10
        )
        np.testing.assert_allclose(engine.to_numpy(scores), dist.seq_log_density(enc), rtol=1.0e-10, atol=1.0e-10)

        resident = kernel.resident_accumulate(enc, engine.asarray(weights))
        self.assertIsInstance(resident.component_counts, DTensor)
        self.assertIsInstance(resident.component_counts.placements[0], Shard)
        self.assertEqual(resident.component_counts.placements[0].dim, 0)
        for stat in resident.component_stats:
            self.assertIsInstance(stat, DTensor)
            self.assertIsInstance(stat.placements[0], Shard)
            self.assertEqual(stat.placements[0].dim, 0)

        actual = resident.value()
        start, local = resident.local_value()
        self.assertEqual(start, 0)
        self.assert_suff_stat_allclose(local[0], actual[0])
        self.assert_suff_stat_allclose(local[1], actual[1])

        legacy_acc = est.accumulator_factory().make()
        legacy_acc.seq_update(enc, weights, dist)
        self.assert_suff_stat_allclose(actual[0], legacy_acc.value()[0])
        self.assert_suff_stat_allclose(actual[1], legacy_acc.value()[1])

        fitted = resident.estimate(est)
        expected = est.estimate(None, legacy_acc.value())
        np.testing.assert_allclose(fitted.w, expected.w, rtol=1.0e-10, atol=1.0e-10)
        np.testing.assert_allclose(
            fitted.seq_log_density(enc), expected.seq_log_density(enc), rtol=1.0e-10, atol=1.0e-10
        )
        shard = resident.estimate_component_shard(est, total_count=float(np.sum(weights)))
        self.assertIsInstance(shard, StackedMixtureShardEstimate)
        self.assertEqual((shard.component_start, shard.component_stop), (0, dist.num_components))
        np.testing.assert_allclose(shard.weights, expected.w, rtol=1.0e-10, atol=1.0e-10)
        for got, exp in zip(shard.components, expected.components):
            np.testing.assert_allclose(got.seq_log_density(enc), exp.seq_log_density(enc), rtol=1.0e-10, atol=1.0e-10)

    @unittest.skipUnless(HAS_TORCH, "torch is not installed")
    def test_torch_mesh_vector_and_matrix_resident_stats_match_legacy_stats(self):
        from torch.distributed.tensor import DTensor, Shard

        cases = {name: (dist, data) for name, dist, data in self.stacked_mixture_cases()}
        engine = TorchEngine(dtype=torch.float64, mesh=_single_rank_mesh(), shard="components")

        def assert_component_sharded(value):
            if isinstance(value, DTensor):
                if isinstance(value.placements[0], Shard):
                    self.assertEqual(value.placements[0].dim, 0)
                    return 1
                return 0
            if isinstance(value, tuple):
                return sum(assert_component_sharded(child) for child in value)
            if isinstance(value, list):
                return sum(assert_component_sharded(child) for child in value)
            if isinstance(value, dict):
                return sum(assert_component_sharded(child) for child in value.values())
            return 0

        for name in ("diagonal_gaussian", "multivariate_gaussian"):
            with self.subTest(case=name):
                dist, data = cases[name]
                enc = dist.dist_to_encoder().seq_encode(data)
                weights = np.linspace(0.25, 1.25, len(data))
                est = dist.estimator()
                kernel = dist.kernel(engine=engine, estimator=est)

                self.assertIsInstance(kernel, StackedMixtureKernel)
                self.assertTrue(kernel.has_resident_accumulate)
                resident = kernel.resident_accumulate(enc, engine.asarray(weights))

                self.assertGreater(assert_component_sharded(resident.component_counts), 0)
                self.assertGreater(assert_component_sharded(resident.component_stats), 0)

                actual = resident.value()
                start, local = resident.local_value()
                self.assertEqual(start, 0)
                self.assert_suff_stat_allclose(local[0], actual[0])
                self.assert_suff_stat_allclose(local[1], actual[1])

                legacy_acc = est.accumulator_factory().make()
                legacy_acc.seq_update(enc, weights, dist)
                self.assert_suff_stat_allclose(actual[0], legacy_acc.value()[0])
                self.assert_suff_stat_allclose(actual[1], legacy_acc.value()[1])

                fitted = resident.estimate(est)
                expected = est.estimate(None, legacy_acc.value())
                np.testing.assert_allclose(fitted.w, expected.w, rtol=1.0e-10, atol=1.0e-10)
                np.testing.assert_allclose(
                    fitted.seq_log_density(enc), expected.seq_log_density(enc), rtol=1.0e-10, atol=1.0e-10
                )
                shard = resident.estimate_component_shard(est, total_count=float(np.sum(weights)))
                self.assertIsInstance(shard, StackedMixtureShardEstimate)
                self.assertEqual((shard.component_start, shard.component_stop), (0, dist.num_components))
                np.testing.assert_allclose(shard.weights, expected.w, rtol=1.0e-10, atol=1.0e-10)
                for got, exp in zip(shard.components, expected.components):
                    np.testing.assert_allclose(
                        got.seq_log_density(enc), exp.seq_log_density(enc), rtol=1.0e-10, atol=1.0e-10
                    )

    @unittest.skipUnless(HAS_TORCH, "torch is not installed")
    def test_torch_mesh_table_and_wrapper_resident_shards_match_legacy_stats(self):
        from torch.distributed.tensor import DTensor, Shard

        cases = {name: (dist, data) for name, dist, data in self.stacked_mixture_cases()}
        engine = TorchEngine(dtype=torch.float64, mesh=_single_rank_mesh(), shard="components")

        def assert_component_sharded(value):
            if isinstance(value, DTensor):
                if isinstance(value.placements[0], Shard):
                    self.assertEqual(value.placements[0].dim, 0)
                    return 1
                return 0
            if isinstance(value, tuple):
                return sum(assert_component_sharded(child) for child in value)
            if isinstance(value, list):
                return sum(assert_component_sharded(child) for child in value)
            if isinstance(value, dict):
                return sum(assert_component_sharded(child) for child in value.values())
            return 0

        names = (
            "categorical",
            "integer_categorical",
            "bernoulli_set",
            "integer_multinomial",
            "optional_gaussian",
            "composite_with_optional",
            "sequence_gaussian",
            "conditional_gaussian",
            "record_gaussian_poisson",
            "weighted_gaussian",
            "markov_chain",
            "integer_markov_chain",
        )
        for name in names:
            with self.subTest(case=name):
                dist, data = cases[name]
                enc = dist.dist_to_encoder().seq_encode(data)
                weights = np.linspace(0.25, 1.25, len(data))
                est = dist.estimator()
                kernel = dist.kernel(engine=engine, estimator=est)

                self.assertIsInstance(kernel, StackedMixtureKernel)
                self.assertTrue(kernel.has_resident_accumulate)
                resident = kernel.resident_accumulate(enc, engine.asarray(weights))
                self.assertGreater(assert_component_sharded(resident.component_counts), 0)
                self.assertGreaterEqual(assert_component_sharded(resident.component_stats), 0)

                actual = resident.value()
                start, local = resident.local_value()
                self.assertEqual(start, 0)
                self.assert_suff_stat_allclose(local[0], actual[0])
                self.assert_suff_stat_allclose(local[1], actual[1])

                legacy_acc = est.accumulator_factory().make()
                legacy_acc.seq_update(enc, weights, dist)
                self.assert_suff_stat_allclose(actual[0], legacy_acc.value()[0])
                self.assert_suff_stat_allclose(actual[1], legacy_acc.value()[1])

                expected = est.estimate(None, legacy_acc.value())
                shard = resident.estimate_component_shard(est, total_count=float(np.sum(weights)))
                self.assertIsInstance(shard, StackedMixtureShardEstimate)
                self.assertEqual((shard.component_start, shard.component_stop), (0, dist.num_components))
                np.testing.assert_allclose(shard.weights, expected.w, rtol=1.0e-10, atol=1.0e-10)
                for got, exp in zip(shard.components, expected.components):
                    np.testing.assert_allclose(
                        got.seq_log_density(enc), exp.seq_log_density(enc), rtol=1.0e-10, atol=1.0e-10
                    )

    @unittest.skipUnless(HAS_TORCH, "torch is not installed")
    def test_torch_stacked_non_gaussian_mixture_kernels_match_legacy_paths(self):
        engine = TorchEngine(dtype=torch.float64)
        for name, dist, data in self.stacked_mixture_cases():
            with self.subTest(case=name):
                enc = dist.dist_to_encoder().seq_encode(data)
                kernel = dist.kernel(engine=engine)

                self.assertIsInstance(kernel, StackedMixtureKernel)
                if name in (
                    "bernoulli",
                    "beta",
                    "binomial",
                    "exponential",
                    "gamma",
                    "diagonal_gaussian",
                    "geometric",
                    "laplace",
                    "log_gaussian",
                    "logistic",
                    "negative_binomial",
                    "pareto",
                    "poisson",
                    "rayleigh",
                    "student_t",
                    "uniform",
                    "weibull",
                ):
                    self.assertTrue(kernel._generated)
                np.testing.assert_allclose(
                    kernel.component_scores(enc).detach().cpu().numpy(),
                    dist.seq_component_log_density(enc),
                    rtol=1.0e-10,
                    atol=1.0e-10,
                )
                np.testing.assert_allclose(
                    kernel.score(enc).detach().cpu().numpy(), dist.seq_log_density(enc), rtol=1.0e-10, atol=1.0e-10
                )

    @unittest.skipUnless(HAS_TORCH, "torch is not installed")
    def test_torch_stacked_non_gaussian_mixture_accumulates_legacy_stats(self):
        engine = TorchEngine(dtype=torch.float64)
        resident_cases = {
            "bernoulli",
            "bernoulli_set",
            "beta",
            "binomial",
            "categorical",
            "conditional_gaussian",
            "diagonal_gaussian",
            "dirac_length",
            "dirichlet",
            "exponential",
            "gamma",
            "geometric",
            "ignored_gaussian",
            "indian_buffet_process",
            "integer_bernoulli_edit",
            "integer_bernoulli_set",
            "integer_categorical",
            "integer_markov_chain",
            "integer_multinomial",
            "integer_step_bernoulli_edit",
            "integer_uniform_spike",
            "laplace",
            "log_gaussian",
            "logistic",
            "markov_chain",
            "multivariate_gaussian",
            "multinomial_categorical",
            "null",
            "optional_gaussian",
            "pareto",
            "point_mass",
            "poisson",
            "rayleigh",
            "record_gaussian_poisson",
            "select_gaussian_by_sign",
            "sequence_gaussian",
            "spearman_ranking",
            "student_t",
            "uniform",
            "weibull",
            "composite_with_optional",
            "transform_gaussian",
            "von_mises_fisher",
            "weighted_gaussian",
        }
        generated_resident_cases = {
            "beta",
            "diagonal_gaussian",
            "geometric",
            "logistic",
            "rayleigh",
            "student_t",
            "weibull",
        }
        estimator_resident_cases = {
            "binomial",
            "conditional_gaussian",
            "dirac_length",
            "integer_multinomial",
            "integer_bernoulli_edit",
            "integer_markov_chain",
            "integer_step_bernoulli_edit",
            "markov_chain",
            "multinomial_categorical",
            "record_gaussian_poisson",
            "select_gaussian_by_sign",
            "sequence_gaussian",
            "transform_gaussian",
            "weighted_gaussian",
        }
        for name, dist, data in self.stacked_mixture_cases():
            with self.subTest(case=name):
                enc = dist.dist_to_encoder().seq_encode(data)
                weights = np.linspace(0.25, 1.25, len(data))
                est = dist.estimator()
                kernel = dist.kernel(engine=engine, estimator=est)

                self.assertIsInstance(kernel, StackedMixtureKernel)
                actual = kernel.accumulate(enc, engine.asarray(weights))
                legacy_acc = est.accumulator_factory().make()
                legacy_acc.seq_update(enc, weights, dist)

                self.assert_suff_stat_allclose(actual[0], legacy_acc.value()[0])
                self.assert_suff_stat_allclose(actual[1], legacy_acc.value()[1])
                if name in resident_cases:
                    self.assertTrue(kernel.has_resident_accumulate)
                    resident = kernel.resident_accumulate(enc, engine.asarray(weights))
                    self.assertIsInstance(resident, StackedMixtureResidentStats)
                    self.assert_suff_stat_allclose(resident.value()[0], legacy_acc.value()[0])
                    self.assert_suff_stat_allclose(resident.value()[1], legacy_acc.value()[1])
                    if name in generated_resident_cases:
                        self.assertFalse(
                            callable(getattr(kernel.component_type, "backend_stacked_sufficient_statistics", None))
                        )
                        self.assertTrue(kernel._generated)
                    if name in estimator_resident_cases:
                        self.assertTrue(
                            callable(
                                getattr(
                                    kernel.component_type, "backend_stacked_sufficient_statistics_with_estimator", None
                                )
                            )
                        )

    @unittest.skipUnless(HAS_TORCH, "torch is not installed")
    def test_torch_stacked_binomial_support_mismatch_falls_back_to_generic_kernel(self):
        engine = TorchEngine(dtype=torch.float64)
        dist = MixtureDistribution(
            [
                BinomialDistribution(0.25, 4),
                BinomialDistribution(0.60, 6),
            ],
            [0.5, 0.5],
        )
        kernel = dist.kernel(engine=engine)
        self.assertNotIsInstance(kernel, StackedMixtureKernel)
        data = [0, 1, 2, 4]
        enc = dist.dist_to_encoder().seq_encode(data)
        np.testing.assert_allclose(
            kernel.score(enc).detach().cpu().numpy(), dist.seq_log_density(enc), rtol=1.0e-10, atol=1.0e-10
        )


if __name__ == "__main__":
    unittest.main()
