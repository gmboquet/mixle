"""Information-gain retrieval (mixle.substrate.eig_retrieve), CARD F6-a.

Acceptance (per the card): fewer items than cosine/lexical retrieval to reach a target accuracy. The scenario
below has several textually redundant filler items (near-duplicate wording of the query, carrying almost no
evidence) and one differently-worded item that is decisive -- so a similarity-ranked retrieval spends its
budget on the redundant filler while information-gain retrieval goes straight for the decisive item.

MXR-080-0250/0251 regressions (below the CARD F6-a tests): expected (not realized) gain, refusing to select
a non-positive-gain candidate, copy-on-update belief scoring, and narrow/diagnosed failure handling.
"""

import unittest

import numpy as np

from mixle.inference.belief import BeliefState, CategoricalBelief
from mixle.substrate.core import Substrate, SubstrateItem
from mixle.substrate.eig_retrieve import EvidenceOutcome, EvidenceOutcomes, eig_retrieve
from mixle.substrate.retrieve import retrieve

LABELS = ["alpha", "beta", "gamma"]


def _evidence_fn(item):
    # MXR-080-0251: evidence_fn signals "no usable evidence" the same way belief.update() signals a bad
    # update -- ValueError, and only ValueError. A plain `item.payload["log_lik"]` would let a missing key
    # raise the un-declared KeyError instead, which eig_retrieve now correctly refuses to treat as a skip.
    try:
        return item.payload["log_lik"]
    except KeyError as exc:
        raise ValueError(f"item {item.id} has no log_lik evidence") from exc


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

    def test_second_round_stops_once_remaining_evidence_no_longer_helps(self):
        # Greedy re-scoring against the shrunk pool: once the decisive item is taken, belief is confident
        # in "beta". Every remaining filler item carries the SAME fixed-direction evidence (weakly favors
        # alpha over beta) -- applied to a belief already confident in beta, that pulls mass away from the
        # MAP and genuinely INCREASES entropy (verified independently below by recomputing from scratch,
        # not by reading back eig_retrieve's own numbers). MXR-080-0250: a real, not just "near zero", gain
        # must not be forced through -- eig_retrieve must stop at 1 item, not present a harmful second pick
        # as if it were more evidence.
        substrate, _query = _build_substrate()
        result = eig_retrieve(substrate, CategoricalBelief.uniform(LABELS), _evidence_fn, k=2)
        self.assertEqual(len(result.items), 1)
        self.assertEqual(result.stop_reason, "no_admissible_positive_gain")
        self.assertEqual(result.skipped, [])  # every filler item was validly scored, just correctly rejected

        # independently confirm every remaining filler's true gain (hand-computed, not read back from
        # eig_retrieve) is indeed <= 0 against the post-decisive belief -- the property the stop is for
        after_decisive = CategoricalBelief.uniform(LABELS).update(_evidence_fn(result.items[0]))
        chosen_id = result.items[0].id
        remaining_fillers = [it for it in substrate.all() if it.id != chosen_id and "log_lik" in it.payload]
        self.assertEqual(len(remaining_fillers), 6)
        for item in remaining_fillers:
            gain = after_decisive.entropy() - after_decisive.update(_evidence_fn(item)).entropy()
            self.assertLessEqual(gain, 0.0)

    def test_items_with_unusable_evidence_are_skipped_not_fatal(self):
        substrate, _query = _build_substrate()
        substrate.put(SubstrateItem(kind="text", text="unrelated item with no evidence", payload={}))
        result = eig_retrieve(substrate, CategoricalBelief.uniform(LABELS), _evidence_fn, k=7)
        # every returned item actually had usable evidence; the malformed one was skipped, not raised
        for item in result.items:
            self.assertIn("log_lik", item.payload)
        # MXR-080-0251: the skip is reported, not just silently absorbed -- which candidate, and why
        self.assertEqual(len(result.skipped), 1)
        self.assertEqual(result.skipped[0].item.text, "unrelated item with no evidence")
        self.assertIn("no log_lik evidence", result.skipped[0].reason)


class Eig0250ExpectedGainTest(unittest.TestCase):
    """MXR-080-0250: score by expected posterior entropy over a genuine outcome distribution, and refuse
    to select when no candidate clears a strictly-positive gain."""

    def test_all_negative_gain_candidates_yield_no_selection_not_a_forced_pick(self):
        # Belief already confident in "beta" (equivalent to having assimilated the decisive item).
        # Every offered candidate carries the SAME fixed-direction filler evidence, which pulls mass away
        # from beta and therefore genuinely increases entropy -- there is no admissible positive-gain pick
        # anywhere in this pool. The old routine picked the least-bad (still negative) one regardless and
        # reported it as if it were beneficial evidence; the fix must select nothing.
        confident_belief = CategoricalBelief.uniform(LABELS).update([-8.0, 0.0, -8.0])
        substrate = Substrate()
        for i in range(5):
            substrate.put(SubstrateItem(kind="text", text=f"filler {i}", payload={"log_lik": [0.01, -0.01, 0.0]}))

        result = eig_retrieve(substrate, confident_belief, _evidence_fn, k=3)

        self.assertEqual(result.items, [])
        self.assertEqual(result.scores, [])
        self.assertEqual(result.stop_reason, "no_admissible_positive_gain")
        self.assertEqual(result.skipped, [])  # these were scoreable, just correctly not worth picking

    def test_expected_gain_averages_over_outcomes_not_a_single_realization(self):
        # A candidate whose true evidence is genuinely unknown at scoring time: 50/50 between two OPPOSITE
        # decisive outcomes (strongly-alpha vs. strongly-gamma). Whichever way it resolves, the observer
        # ends up nearly certain -- so the TRUE expected posterior entropy is tiny and the true expected
        # gain is almost the full prior entropy. A single deterministic evaluation has no way to represent
        # this at all; the closest a caller lacking EvidenceOutcomes could do is evaluate the AVERAGED
        # evidence as one point (naive), which lands on a starkly different, much smaller number -- entropy
        # is a concave functional of the belief, so "entropy of the averaged update" != "average of the
        # per-outcome entropies" (Jensen's inequality). eig_retrieve must compute the real expectation.
        prior = CategoricalBelief.uniform(LABELS)
        outcome_a = [8.0, -8.0, -8.0]  # strongly favors alpha
        outcome_b = [-8.0, -8.0, 8.0]  # strongly favors gamma

        true_expected_entropy = 0.5 * prior.update(outcome_a).entropy() + 0.5 * prior.update(outcome_b).entropy()
        true_expected_gain = prior.entropy() - true_expected_entropy

        naive_ll = ((np.asarray(outcome_a) + np.asarray(outcome_b)) / 2.0).tolist()
        naive_gain = prior.entropy() - prior.update(naive_ll).entropy()

        # the two hand-computed approximations must themselves diverge sharply, or this scenario would not
        # actually exercise the bug -- guard the test's own premise before trusting eig_retrieve's number
        self.assertGreater(true_expected_gain - naive_gain, 0.5)

        substrate = Substrate()
        substrate.put(SubstrateItem(kind="record", text="uncertain experiment", payload={}))

        def evidence_fn(item):
            return EvidenceOutcomes([EvidenceOutcome(0.5, outcome_a), EvidenceOutcome(0.5, outcome_b)])

        result = eig_retrieve(substrate, prior, evidence_fn, k=1)

        self.assertEqual(len(result.items), 1)
        self.assertAlmostEqual(result.scores[0], true_expected_gain, places=9)
        # and NOT the naive single-point value -- the divergence a broken realized-gain scorer would show
        self.assertGreater(abs(result.scores[0] - naive_gain), 0.5)

    def test_evidence_outcomes_single_outcome_matches_deterministic_evidence(self):
        # Degenerate case / backward compatibility: a single-outcome EvidenceOutcomes (probability 1.0) is
        # mathematically the same point mass as a bare deterministic evidence value, so both must score
        # (and advance the running belief) identically.
        belief = CategoricalBelief.uniform(LABELS)
        log_lik = [-8.0, 0.0, -8.0]

        plain_substrate = Substrate()
        plain_substrate.put(SubstrateItem(kind="record", text="det", payload={"log_lik": log_lik}))
        wrapped_substrate = Substrate()
        wrapped_substrate.put(SubstrateItem(kind="record", text="det", payload={"log_lik": log_lik}))

        plain_result = eig_retrieve(plain_substrate, belief, lambda it: it.payload["log_lik"], k=1)
        wrapped_result = eig_retrieve(
            wrapped_substrate, belief, lambda it: EvidenceOutcomes([EvidenceOutcome(1.0, it.payload["log_lik"])]), k=1
        )

        self.assertAlmostEqual(plain_result.scores[0], wrapped_result.scores[0], places=12)

    def test_malformed_evidence_outcomes_probabilities_are_rejected_and_skipped(self):
        # Probabilities that don't sum to 1 are a declared evidence-incompatibility (validated eagerly by
        # EvidenceOutcomes itself), not a crash and not silently-averaged-anyway: the candidate is skipped
        # with a diagnostic, exactly like a belief.update() rejection would be.
        substrate = Substrate()
        substrate.put(SubstrateItem(kind="record", text="bad probs", payload={}))
        substrate.put(SubstrateItem(kind="record", text="good", payload={"log_lik": [-8.0, 0.0, -8.0]}))

        def evidence_fn(item):
            if item.text == "bad probs":
                return EvidenceOutcomes(
                    [EvidenceOutcome(0.3, [1.0, 0.0, 0.0]), EvidenceOutcome(0.3, [0.0, 1.0, 0.0])]
                )  # sums to 0.6, not 1.0
            return item.payload["log_lik"]

        result = eig_retrieve(substrate, CategoricalBelief.uniform(LABELS), evidence_fn, k=5)

        self.assertEqual([it.text for it in result.items], ["good"])
        self.assertEqual(len(result.skipped), 1)
        self.assertEqual(result.skipped[0].item.text, "bad probs")
        self.assertIn("sum to 1.0", result.skipped[0].reason)


class Eig0251CopyOnUpdateTest(unittest.TestCase):
    """MXR-080-0251: candidate scoring must never mutate the running belief, and only a declared
    evidence-incompatibility is a skip -- everything else propagates."""

    class _MutatingBelief(BeliefState):
        """A belief whose update() mutates self in place and returns self -- real belief types
        (CategoricalBelief, GaussianBelief) are already pure; this simulates a hypothetical non-compliant
        BeliefState to prove eig_retrieve defends against one rather than merely trusting the contract."""

        def __init__(self, value):
            self.value = float(value)
            self.update_calls = 0

        def mean(self):
            return np.array([self.value])

        def entropy(self):
            return 1.0 / (1.0 + self.value)  # monotonically shrinks as `value` grows

        def sample(self, n=1, rng=None):
            return np.full(n, self.value)

        def update(self, delta):
            self.value += float(delta)  # mutates self in place
            self.update_calls += 1
            return self  # ...and returns self, not a new object -- doubly non-compliant

    def test_candidate_scoring_never_mutates_the_running_belief_or_contaminates_siblings(self):
        belief = self._MutatingBelief(0.0)
        substrate = Substrate()
        substrate.put(SubstrateItem(kind="record", text="small", payload={"delta": 0.001}))
        substrate.put(SubstrateItem(kind="record", text="big", payload={"delta": 10.0}))

        result = eig_retrieve(substrate, belief, lambda it: it.payload["delta"], k=1)

        # the caller's own belief object must be untouched by scoring, regardless of how update() behaves
        self.assertEqual(belief.value, 0.0)
        # correctness, not just non-contamination: the genuinely more-informative candidate wins. Under the
        # old code every candidate's computed gain degenerated to exactly 0.0 (current and nxt were the
        # same mutated object), so the FIRST-examined candidate won regardless of quality; "big" is listed
        # second in insertion order, so this also distinguishes the fix from that failure mode.
        self.assertEqual([it.text for it in result.items], ["big"])
        self.assertAlmostEqual(result.scores[0], 1.0 - 1.0 / 11.0, places=12)

    def test_belief_update_rejects_incompatible_evidence_are_also_skipped(self):
        # The declared-incompatibility path covers BOTH sources of ValueError: evidence_fn's own, and
        # belief.update()'s own (here, CategoricalBelief.update rejecting a wrong-length log-likelihood
        # vector) -- both must be caught uniformly and reported, not just evidence_fn's.
        substrate = Substrate()
        substrate.put(SubstrateItem(kind="record", text="wrong shape", payload={"log_lik": [1.0, 0.0]}))
        substrate.put(SubstrateItem(kind="record", text="good", payload={"log_lik": [-8.0, 0.0, -8.0]}))

        result = eig_retrieve(substrate, CategoricalBelief.uniform(LABELS), _evidence_fn, k=5)

        self.assertEqual([it.text for it in result.items], ["good"])
        self.assertEqual(len(result.skipped), 1)
        self.assertEqual(result.skipped[0].item.text, "wrong shape")

    def test_programming_error_in_evidence_fn_propagates_not_swallowed(self):
        # A genuine bug (here a TypeError unrelated to "no usable evidence") must not be indistinguishable
        # from a legitimate evidence incompatibility -- it propagates instead of being silently skipped.
        substrate = Substrate()
        substrate.put(SubstrateItem(kind="record", text="buggy", payload={}))

        def buggy_evidence_fn(item):
            return 1 + "not a number"  # TypeError, nothing to do with missing evidence

        with self.assertRaises(TypeError):
            eig_retrieve(substrate, CategoricalBelief.uniform(LABELS), buggy_evidence_fn, k=1)


if __name__ == "__main__":
    unittest.main()
