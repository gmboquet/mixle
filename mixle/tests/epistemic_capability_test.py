"""mixle.epistemic is discoverable through the capability catalog (Card E0)."""

import unittest

import mixle.epistemic  # noqa: F401 -- import succeeds with no side effects
from mixle.capability import catalog, render_catalog_markdown


class EpistemicCapabilityCatalogTest(unittest.TestCase):
    def test_belief_trackable_is_in_the_catalog(self):
        specs = {spec.name: spec for spec in catalog()}
        self.assertIn("BeliefTrackable", specs)
        spec = specs["BeliefTrackable"]
        self.assertEqual(spec.home, "mixle.epistemic")
        self.assertTrue(spec.summary.strip())

    def test_catalog_markdown_mentions_it(self):
        text = render_catalog_markdown()
        self.assertIn("BeliefTrackable", text)


if __name__ == "__main__":
    unittest.main()
