"""The consolidated and package ranking APIs expose the same intentional types."""

import mixle.stats as stats
import mixle.stats.rankings as rankings

EXPECTED_RANKING_API = (
    "ItemOrdering",
    "RankVector",
    "MallowsAccumulator",
    "MallowsAccumulatorFactory",
    "MatchingAccumulator",
    "MatchingAccumulatorFactory",
    "GeneralizedMallowsComputationDiagnostics",
    "LowRankPermutationComputationDiagnostics",
    "PlackettLuceAccumulator",
    "PlackettLuceAccumulatorFactory",
    "PlackettLucePartialEstimator",
    "PlackettLucePartialAccumulator",
    "PlackettLucePartialAccumulatorFactory",
    "PlackettLucePartialDataEncoder",
    "SpearmanRankingAccumulator",
    "SpearmanRankingAccumulatorFactory",
    "ThurstoneApproximationDiagnostics",
    "ThurstoneFitDiagnostics",
)


def test_ranking_types_are_available_from_both_public_namespaces():
    for name in EXPECTED_RANKING_API:
        assert getattr(stats, name) is getattr(rankings, name)
        assert name in rankings.__all__
        if not name.endswith(("Accumulator", "AccumulatorFactory", "DataEncoder")):
            assert name in stats.__all__
