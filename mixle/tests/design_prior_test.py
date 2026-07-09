"""rank_design_families: the persistent what-works prior over structural design families (CARD DESIGN-a).

After feeding N accepted recipes, the prior must rank the accepted (higher-quality) family above an
un-tried one -- so the next round's structural-search proposal starts from a sharper prior.
"""

import unittest

from mixle.task.design_prior import best_family, rank_design_families, record_accepted_recipe
from mixle.task.edge import DesignModel


def _design_with_two_families():
    design = DesignModel(signature="test-design", n_constraints=1)
    for point, quality in [([1.0, 2.0], 0.92), ([1.1, 2.1], 0.88), ([0.9, 1.9], 0.90)]:
        record_accepted_recipe(design, point, quality, [0.0], family="quotient_leaf")
    for point, quality in [([0.5, 1.0], 0.60), ([0.6, 1.1], 0.65)]:
        record_accepted_recipe(design, point, quality, [0.0], family="plain_head")
    return design


class RankDesignFamiliesTest(unittest.TestCase):
    def test_accepted_family_ranks_above_an_untried_one(self):
        """The card's own acceptance: after N accepted recipes, the prior ranks the accepted family
        above an un-tried one."""
        design = _design_with_two_families()
        ranked = rank_design_families(design, candidates=["never_tried_family"])
        names = [name for name, _score in ranked]
        self.assertEqual(names, ["quotient_leaf", "plain_head", "never_tried_family"])

    def test_untried_family_gets_the_named_default_score(self):
        design = _design_with_two_families()
        ranked = dict(rank_design_families(design, candidates=["never_tried_family"]))
        self.assertEqual(ranked["never_tried_family"], float("-inf"))

    def test_ranking_is_by_mean_recorded_quality(self):
        design = _design_with_two_families()
        ranked = dict(rank_design_families(design))
        self.assertAlmostEqual(ranked["quotient_leaf"], (0.92 + 0.88 + 0.90) / 3, places=9)
        self.assertAlmostEqual(ranked["plain_head"], (0.60 + 0.65) / 2, places=9)

    def test_best_family_returns_the_top_ranked_family(self):
        design = _design_with_two_families()
        self.assertEqual(best_family(design), "quotient_leaf")

    def test_empty_design_has_no_ranking_and_no_best_family(self):
        empty = DesignModel(signature="empty", n_constraints=1)
        self.assertEqual(rank_design_families(empty), [])
        self.assertIsNone(best_family(empty))

    def test_untried_candidate_alone_with_no_history_still_reports(self):
        empty = DesignModel(signature="empty", n_constraints=1)
        ranked = rank_design_families(empty, candidates=["some_family"])
        self.assertEqual(ranked, [("some_family", float("-inf"))])

    def test_custom_tag_key(self):
        design = DesignModel(signature="custom", n_constraints=1)
        design.add([1.0], 0.8, [0.0], structure="conv_pool")
        design.add([2.0], 0.5, [0.0], structure="linear")
        ranked = rank_design_families(design, tag_key="structure")
        self.assertEqual([name for name, _ in ranked], ["conv_pool", "linear"])

    def test_rows_without_the_tag_key_are_ignored(self):
        design = DesignModel(signature="mixed", n_constraints=1)
        design.add([1.0], 0.9, [0.0], family="tracked")
        design.add([2.0], 0.1, [0.0])  # no family tag: not a candidate for the prior
        ranked = rank_design_families(design)
        self.assertEqual(ranked, [("tracked", 0.9)])


if __name__ == "__main__":
    unittest.main()
