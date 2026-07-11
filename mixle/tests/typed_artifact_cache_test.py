"""Reference invalidation fixtures for the typed local artifact cache."""

import json

import pytest

from mixle.experimental.typed_runtime import ArtifactKind, VersionedArtifactCache, compile_update_graph
from mixle.stats import GaussianDistribution, GaussianEstimator, MixtureDistribution, MixtureEstimator

pytestmark = [pytest.mark.experimental, pytest.mark.fast]


def _mixture_graph():
    model = MixtureDistribution(
        [GaussianDistribution(-1.0, 1.0), GaussianDistribution(1.0, 1.0)],
        [0.5, 0.5],
    )
    estimator = MixtureEstimator([GaussianEstimator(), GaussianEstimator()])
    return compile_update_graph(model, estimator)


class DependencyAwareInvalidationTest:
    def test_child_update_invalidates_child_and_parent_but_not_sibling(self):
        graph = _mixture_graph()
        cache = VersionedArtifactCache(graph)
        children = [node.node_id for node in graph.nodes if node.node_id != graph.root_node]
        left, right = children
        for node_id in (left, right, graph.root_node):
            cache.put(node_id, ArtifactKind.SCORES, "%s-scores" % node_id)

        receipt = cache.invalidate(left)

        assert receipt.invalidated_nodes == (left, graph.root_node)
        assert not cache.contains(left, ArtifactKind.SCORES)
        assert not cache.contains(graph.root_node, ArtifactKind.SCORES)
        assert cache.get(right, ArtifactKind.SCORES) == "%s-scores" % right
        assert cache.generation(left) == 1
        assert cache.generation(graph.root_node) == 1
        assert cache.generation(right) == 0
        json.dumps(receipt.as_dict(), allow_nan=False)

    def test_parent_update_does_not_invalidate_independent_children(self):
        graph = _mixture_graph()
        cache = VersionedArtifactCache(graph)
        children = [node.node_id for node in graph.nodes if node.node_id != graph.root_node]
        for node_id in children:
            cache.put(node_id, ArtifactKind.POSTERIORS, node_id)
        cache.put(graph.root_node, ArtifactKind.POSTERIORS, "root")

        receipt = cache.invalidate(graph.root_node, ArtifactKind.PARAMETERS)
        assert receipt.invalidated_nodes == (graph.root_node,)
        assert all(cache.contains(node_id, ArtifactKind.POSTERIORS) for node_id in children)
        assert not cache.contains(graph.root_node, ArtifactKind.POSTERIORS)

    def test_generation_detects_results_started_before_invalidation(self):
        graph = _mixture_graph()
        cache = VersionedArtifactCache(graph)
        child = next(node.node_id for node in graph.nodes if node.node_id != graph.root_node)
        captured_generation = cache.generation(child)
        cache.invalidate(child)

        assert captured_generation == 0
        assert cache.generation(child) == 1
        assert captured_generation != cache.generation(child)

    def test_clear_advances_versions_even_when_cache_is_empty(self):
        graph = _mixture_graph()
        cache = VersionedArtifactCache(graph)
        cache.clear()
        assert all(cache.generation(node.node_id) == 1 for node in graph.nodes)
        assert cache.as_dict()["entries"] == []
