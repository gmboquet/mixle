"""Information-gain retrieval (mixle.substrate.eig_retrieve), CARD F6-a.

Acceptance (per the card): fewer items than cosine/lexical retrieval to reach a target accuracy. The scenario
below has several textually redundant filler items (near-duplicate wording of the query, carrying almost no
evidence) and one differently-worded item that is decisive -- so a similarity-ranked retrieval spends its
budget on the redundant filler while information-gain retrieval goes straight for the decisive item.
"""

import unittest

from mixle.inference.belief import CategoricalBelief
from mixle.substrate.core import Substrate, SubstrateItem
from mixle.substrate.eig_retrieve import eig_retrieve
from mixle.substrate.retrieve import retrieve

LABELS = ["alpha", "beta", "gamma"]


def _evidence_fn(item):
    return item.payload["log_lik"]


def _build_substrate():
    substrate = Substrate()
    query = "status update about the project rollout timeline"
    # six near-duplicate filler items: high textual overlap with the query, almost no evidence
    filler_texts = [
        "status update about the project rollout timeline for stakeholders",
        "another status update about the project rollout timeline this week",
        "weekly status update covering the project rollout timeline",
        "project rollout timeline status update from the team",
        "status update: project rollout timeline unchanged since last week",
        "brief status update about the rollout timeline for the project",
    ]
    for text in filler_texts:
        substrate.put(SubstrateItem(kind="text", text=text, payload={"log_lik": [0.01, -0.01, 0.0]}))
    # one decisive item: different wording (low textual overlap with the query), strong evidence for "beta"
    substrate.put(
        SubstrateItem(
            kind="text",
            text="engineering confirms beta migration finished successfully",
            payload={"log_lik": [-8.0, 0.0, -8.0]},
        )
    )
    return substrate, query


class EigRetrieveTest(unittest.TestCase):
    def test_stays_on_the_lexical_path_for_this_small_corpus(self):
        # load-bearing precondition: with < 8 text items, Substrate.search is the deterministic lexical
        # fallback (no learned-embedder noise), so the comparison below isn't an embedding-fit accident
        substrate, _query = _build_substrate()
        self.assertLess(len(substrate.all()), 8)

    def test_eig_retrieve_reaches_correct_map_with_fewer_items_than_cosine(self):
        substrate, query = _build_substrate()

        eig_result = eig_retrieve(substrate, CategoricalBelief.uniform(LABELS), _evidence_fn, k=1)
        eig_belief = CategoricalBelief.uniform(LABELS)
        for item in eig_result.items:
            eig_belief = eig_belief.update(_evidence_fn(item))
        self.assertEqual(eig_belief.map(), "beta")
        self.assertLess(eig_belief.entropy(), 0.2)

        cosine_result = retrieve(substrate, query, k=1, diversify=False)
        cosine_belief = CategoricalBelief.uniform(LABELS)
        for item in cosine_result.items:
            cosine_belief = cosine_belief.update(_evidence_fn(item))
        # the top cosine hit is one of the redundant filler items (near-duplicate wording of the query);
        # its near-zero evidence leaves the belief essentially unmoved from uniform
        self.assertNotEqual(cosine_belief.map(), "beta")
        self.assertGreater(cosine_belief.entropy(), eig_belief.entropy())

    def test_second_pick_scores_low_once_its_evidence_is_redundant_with_the_first(self):
        # greedy re-scoring against the shrunk pool: once the decisive item is taken, every remaining
        # filler item (near-zero evidence) should score near zero on the next round
        substrate, _query = _build_substrate()
        result = eig_retrieve(substrate, CategoricalBelief.uniform(LABELS), _evidence_fn, k=2)
        self.assertEqual(len(result.items), 2)
        self.assertGreater(result.scores[0], result.scores[1])
        self.assertLess(result.scores[1], 0.05)

    def test_items_with_unusable_evidence_are_skipped_not_fatal(self):
        substrate, _query = _build_substrate()
        substrate.put(SubstrateItem(kind="text", text="unrelated item with no evidence", payload={}))
        result = eig_retrieve(substrate, CategoricalBelief.uniform(LABELS), _evidence_fn, k=7)
        # every returned item actually had usable evidence; the malformed one was skipped, not raised
        for item in result.items:
            self.assertIn("log_lik", item.payload)


if __name__ == "__main__":
    unittest.main()
