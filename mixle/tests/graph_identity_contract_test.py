"""Graph identity and prefetch receipts reject states the graph itself cannot reach (MXR-080-0643).

``add_edge`` already refused a second edge id for one ``(source, target, kind)`` relation, because two
edges asserting one relation double-count it in every downstream tally. ``restore`` and
``replace_if_unchanged`` take caller-supplied state and rebuilt the relation index with a dict
comprehension, which kept the LAST edge and dropped the collision silently -- reinstating exactly what
``add_edge`` refuses. ``GraphPrefetchReceipt.evicted`` was not validated at all.
"""

import unittest

from mixle.experimental.typed_runtime.context_ir import (
    ContextEdge,
    ContextEdgeKind,
    ContextGraph,
    ContextNode,
)
from mixle.experimental.typed_runtime.graph_memory import GraphPrefetchReceipt


def _graph() -> ContextGraph:
    graph = ContextGraph()
    for node_id in ("n1", "n2"):
        graph.add_node(ContextNode(node_id=node_id, kind="claim", text="t", token_count=1))
    return graph


def _edge(edge_id: str) -> ContextEdge:
    return ContextEdge(edge_id=edge_id, source_node="n1", target_node="n2", kind=ContextEdgeKind.SUPPORTS)


class RelationBijectionTest(unittest.TestCase):
    def test_add_edge_still_refuses_a_restated_relation(self):
        graph = _graph()
        graph.add_edge(_edge("e1"))
        with self.assertRaisesRegex(ValueError, "restates the relation"):
            graph.add_edge(_edge("e2"))

    def test_restore_cannot_reinstate_a_restated_relation(self):
        graph = _graph()
        graph.add_edge(_edge("e1"))
        forged = (graph.version, dict(graph.nodes), {"e1": _edge("e1"), "e2": _edge("e2")})
        with self.assertRaisesRegex(ValueError, "two edges for one relation"):
            graph.restore(forged)

    def test_replace_if_unchanged_cannot_either(self):
        graph = _graph()
        graph.add_edge(_edge("e1"))
        expected = (graph.version, graph.nodes, graph.edges)
        forged = (graph.version + 1, dict(graph.nodes), {"e1": _edge("e1"), "e2": _edge("e2")})
        with self.assertRaisesRegex(ValueError, "two edges for one relation"):
            graph.replace_if_unchanged(expected, forged)

    def test_a_legitimate_snapshot_still_restores(self):
        graph = _graph()
        graph.add_edge(_edge("e1"))
        graph.restore((graph.version, dict(graph.nodes), dict(graph.edges)))
        self.assertEqual(len(graph.edges), 1)

    def test_distinct_relations_between_the_same_nodes_coexist(self):
        # The relation key includes the KIND, so "supports" and "contradicts" are different claims.
        graph = _graph()
        graph.add_edge(_edge("e1"))
        graph.add_edge(ContextEdge(edge_id="e2", source_node="n1", target_node="n2", kind=ContextEdgeKind.CONTRADICTS))
        self.assertEqual(len(graph.edges), 2)


class PrefetchReceiptTest(unittest.TestCase):
    def _receipt(self, **overrides) -> GraphPrefetchReceipt:
        fields = dict(requested=("a",), loaded=("a",), evicted=(), resident_tokens=1)
        fields.update(overrides)
        return GraphPrefetchReceipt(**fields)

    def test_the_auditor_case_is_refused(self):
        with self.assertRaisesRegex(ValueError, "names nothing"):
            self._receipt(evicted=("", "ghost", "ghost"))

    def test_a_blank_id_is_refused_in_every_field(self):
        for field in ("requested", "loaded", "evicted"):
            with self.subTest(field=field):
                with self.assertRaisesRegex(ValueError, "names nothing"):
                    self._receipt(**{field: ("   ",)} | ({"loaded": ()} if field == "requested" else {}))

    def test_a_repeated_eviction_is_refused(self):
        with self.assertRaisesRegex(ValueError, "evicted the same partition twice"):
            self._receipt(evicted=("g", "g"))

    def test_a_repeated_request_is_refused(self):
        with self.assertRaisesRegex(ValueError, "requested the same partition twice"):
            self._receipt(requested=("a", "a"))

    def test_evicting_a_partition_this_prefetch_never_requested_is_allowed(self):
        # LRU legitimately evicts something resident from an earlier prefetch. Rejecting this would
        # be the guard rejecting what the cache actually produces.
        self._receipt(requested=("a",), loaded=("a",), evicted=("older",), resident_tokens=3)

    def test_loading_and_evicting_the_same_partition_is_allowed(self):
        # Loaded early in a prefetch, then evicted by LRU later in that same prefetch.
        self._receipt(requested=("a", "b"), loaded=("a", "b"), evicted=("a",), resident_tokens=1)


if __name__ == "__main__":
    unittest.main()
