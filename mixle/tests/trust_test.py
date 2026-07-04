"""verify_lineage / audit_substrate (N1): provenance-chain integrity over the substrate."""

import unittest

from mixle.substrate import Substrate, audit_substrate, verify_lineage
from mixle.substrate.trust import LineageReport


class VerifyLineageTest(unittest.TestCase):
    def test_intact_chain_resolves(self):
        s = Substrate()
        doc = s.add(kind="text", text="source doc")
        model = s.add(kind="artifact", text="model", links=[doc])
        deploy = s.add(kind="artifact", text="deployment", links=[model])
        r = verify_lineage(s, deploy)
        self.assertIsInstance(r, LineageReport)
        self.assertTrue(r.intact)
        self.assertEqual(r.depth, 2)  # deploy -> model -> doc
        self.assertEqual(r.visited, 3)

    def test_dangling_link_is_flagged(self):
        s = Substrate()
        orphan = s.add(kind="trace", text="orphan", links=["ghost-id"])
        r = verify_lineage(s, orphan)
        self.assertFalse(r.intact)
        self.assertEqual(r.dangling, ["ghost-id"])

    def test_missing_root_is_not_intact(self):
        r = verify_lineage(Substrate(), "nope")
        self.assertFalse(r.intact)
        self.assertIn("nope", r.dangling)

    def test_no_links_is_trivially_intact(self):
        s = Substrate()
        leaf = s.add(kind="text", text="leaf")
        r = verify_lineage(s, leaf)
        self.assertTrue(r.intact)
        self.assertEqual(r.n_links, 0)

    def test_cycles_are_handled(self):
        s = Substrate()
        c1 = s.add(kind="text", text="c1")
        c2 = s.add(kind="text", text="c2", links=[c1])
        # make c1 point back at c2 -> a cycle
        c1_item = s.get(c1)
        s.put(type(c1_item)(**{**c1_item.to_json(), "links": [c2]}))
        r = verify_lineage(s, c1)
        self.assertTrue(r.intact)  # no dangling; terminates despite the cycle
        self.assertEqual(r.visited, 2)

    def test_deep_break_is_caught(self):
        s = Substrate()
        a = s.add(kind="text", text="a", links=["gone"])
        b = s.add(kind="artifact", text="b", links=[a])
        r = verify_lineage(s, b)
        self.assertFalse(r.intact)  # the break two levels down still fails the whole chain
        self.assertIn("gone", r.dangling)


class AuditTest(unittest.TestCase):
    def test_audit_counts_intact_and_broken(self):
        s = Substrate()
        doc = s.add(kind="text", text="doc")
        s.add(kind="artifact", text="model", links=[doc])  # intact
        s.add(kind="trace", text="orphan", links=["ghost"])  # broken
        report = audit_substrate(s)
        self.assertEqual(report["n_items"], 3)
        self.assertEqual(report["n_broken"], 1)
        self.assertEqual(report["n_intact"], 2)
        self.assertEqual(report["broken"][0]["dangling"], ["ghost"])

    def test_clean_store_has_no_broken(self):
        s = Substrate()
        a = s.add(kind="text", text="a")
        s.add(kind="artifact", text="b", links=[a])
        report = audit_substrate(s)
        self.assertEqual(report["n_broken"], 0)


if __name__ == "__main__":
    unittest.main()
