"""MXR-080-1903: exact scalar/container contracts on the core semantic + capability surfaces.

Every test here names the state that was actually reproduced before the fix, and each fixed case is
paired with a negative control proving the legitimate input the library really produces still works
-- the failure mode this codebase keeps hitting is a guard that refuses more than the defect.

Scope of this file: `mixle.semantics` records, `mixle.capability_lifecycle` loaders,
`mixle.capability.summarize`, `mixle.enumeration.rescore`, and the `mixle.relations`
shortest/best/matching, flow, MILP/ADMM, sampler and subset boundaries.
"""

from __future__ import annotations

from datetime import UTC, datetime

import numpy as np
import pytest

from mixle import capability, relations
from mixle.capability_lifecycle import (
    SCHEMA_VERSION,
    AuthorizationDecision,
    AuthorizationOutcome,
    CapabilityIdentity,
)
from mixle.enumeration.rescore import RescoredIndex
from mixle.semantics import (
    ConstraintSpec,
    DecisionArtifact,
    LikelihoodSpec,
    ObservationSpec,
    PosteriorArtifact,
    UncertaintyComponent,
    UncertaintyKind,
    ValueSpec,
    load_reference_fixture,
    semantic_digest,
    to_record,
)

# ---------------------------------------------------------------------------
# mixle.semantics -- public records retained mutable evidence under a cached identity
# ---------------------------------------------------------------------------


def _contracts():
    fixture = load_reference_fixture()
    return (
        fixture,
        ValueSpec.from_record(fixture["value"]),
        LikelihoodSpec.from_record(fixture["observation"]["likelihood"]),
        ObservationSpec.from_record(fixture["observation"]),
    )


def _posterior(**overrides):
    fixture, value, likelihood, observation = _contracts()
    uncertainty = [
        UncertaintyComponent(f"u-{kind}", UncertaintyKind(kind), "variance", value=i / 10, unit="(kg/s)^2")
        for i, kind in enumerate(fixture["inference"]["required_uncertainty_kinds"], start=1)
    ]
    fields = {
        "id": "source-rate-posterior",
        "values": [value],
        "observations": [observation],
        "likelihood": likelihood,
        "method": "mcmc",
        "random_seed": 17,
        "summary": {"mean": 8.0},
        "uncertainty": uncertainty,
    }
    fields.update(overrides)
    return PosteriorArtifact(**fields), fields


def test_posterior_does_not_retain_the_callers_evidence_list_under_a_cached_identity():
    # Reproduced before the fix: PosteriorArtifact(values=[...], uncertainty=[...]) kept the CALLER's
    # lists by reference. Appending to `uncertainty` afterwards changed what the record contained,
    # while `identity` -- the digest cached in __post_init__ -- went on reporting the old content, so
    # semantic_digest(p) != p.identity: a durable record advertising an identity that is not its own.
    posterior, fields = _posterior()
    caller_uncertainty = fields["uncertainty"]
    assert isinstance(posterior.values, tuple)
    assert isinstance(posterior.observations, tuple)
    assert isinstance(posterior.uncertainty, tuple)

    recorded = tuple(posterior.uncertainty)
    caller_uncertainty.append(UncertaintyComponent("injected", UncertaintyKind.NUMERICAL, "variance", value=99.0))

    assert posterior.uncertainty == recorded
    assert posterior.identity == semantic_digest(posterior)


def test_decision_does_not_retain_the_callers_alternatives_list():
    # Reproduced before the fix: appending to the caller's list after construction left `utility` no
    # longer closing over `alternatives` -- the exact invariant __post_init__ had just validated.
    alternatives = ["a", "b"]
    decision = DecisionArtifact(
        id="d1",
        alternatives=alternatives,
        selected="a",
        utility={"a": 1.0, "b": 2.0},
        posterior_identity="0" * 64,
        risk_measure="cvar",
    )
    digest = semantic_digest(decision)
    alternatives.append("c")

    assert decision.alternatives == ("a", "b")
    assert set(decision.utility) == set(decision.alternatives)
    assert semantic_digest(decision) == digest


@pytest.mark.parametrize(
    ("build", "label"),
    [
        (
            lambda: DecisionArtifact(
                id="d1",
                alternatives="ab",
                selected="a",
                utility={"a": 1.0, "b": 2.0},
                posterior_identity="0" * 64,
                risk_measure="cvar",
            ),
            "decision alternatives",
        ),
        (lambda: LikelihoodSpec(id="lk", family="normal", observation_ids="o1"), "likelihood observation_ids"),
        (lambda: ConstraintSpec(allowed_values="ab"), "constraint allowed_values"),
    ],
)
def test_a_bare_string_is_not_accepted_where_a_sequence_of_items_is_declared(build, label):
    # Reproduced before the fix: DecisionArtifact(alternatives="ab") satisfied the length- and
    # uniqueness checks by iterating CHARACTERS and to_record() then emitted a JSON string where the
    # schema declares an array; LikelihoodSpec(observation_ids="o1") became ("o", "1"), so a
    # posterior's "likelihood closes over exactly its observations" check compared the wrong set.
    with pytest.raises(TypeError, match=label):
        build()


def test_sequences_the_library_itself_produces_are_still_accepted():
    # Negative control for the guard above: from_record()/to_record() round-trips, tuple-valued
    # construction, and the packaged fixture must all keep working unchanged.
    _, value, likelihood, observation = _contracts()
    posterior, _ = _posterior(values=(value,), observations=(observation,), likelihood=likelihood)
    assert PosteriorArtifact.from_record(to_record(posterior)) == posterior
    assert ValueSpec.from_record(load_reference_fixture()["value"]).shape == ()
    assert ConstraintSpec(allowed_values=["a", "b"]).allowed_values == ("a", "b")
    assert LikelihoodSpec(id="lk", family="normal", observation_ids=["o1"]).observation_ids == ("o1",)


# ---------------------------------------------------------------------------
# mixle.capability_lifecycle -- weak timestamp / authorization parsing
# ---------------------------------------------------------------------------

_IDENTITY_RECORD = {
    "schema_version": SCHEMA_VERSION,
    "capability_id": "capability.mesh.solve",
    "version": "1.2.0",
    "digest": None,
}


def _decision_record(**overrides):
    record = {
        "schema_version": SCHEMA_VERSION,
        "decision_id": "d1",
        "capability": dict(_IDENTITY_RECORD),
        "outcome": "granted",
        "issued_by": "root",
        "scopes": ["read"],
        "decided_at": "2020-01-01T00:00:00Z",
    }
    record.update(overrides)
    return record


@pytest.mark.parametrize("bad", [1700000000, None, ["2020-01-01T00:00:00Z"], b"2020-01-01T00:00:00Z"])
def test_a_non_string_timestamp_is_refused_as_malformed_content_not_an_attribute_error(bad):
    # Reproduced before the fix: `value.replace("Z", ...)` raised AttributeError straight out of
    # from_dict -- these loaders parse untrusted persisted records and document ValueError, so a
    # caller guarding the documented contract did not catch an integer epoch or a JSON null.
    with pytest.raises(ValueError, match="ISO-8601 timestamp"):
        AuthorizationDecision.from_dict(_decision_record(decided_at=bad))


@pytest.mark.parametrize("bad", [None, 12, {"a": 1}, ["x"]])
def test_identity_fields_are_not_str_coerced_into_valid_looking_content(bad):
    # Reproduced before the fix: str(None) == "None" passed the non-empty check, so a malformed
    # record produced a capability identity named "None" -- and capability_id is what an
    # authorization is bound to. str({"a": 1}) became a repr, likewise accepted.
    with pytest.raises(ValueError, match="capability_id must be a string"):
        CapabilityIdentity.from_dict({**_IDENTITY_RECORD, "capability_id": bad})


@pytest.mark.parametrize("bad", ["not-a-mapping", ["a"], 5, None])
def test_a_non_mapping_record_is_refused_as_malformed_content(bad):
    # Reproduced before the fix: value.get(...) raised AttributeError for a JSON string/list/null
    # where an object belongs, escaping the loaders' documented ValueError contract.
    with pytest.raises(ValueError, match="must be a mapping"):
        CapabilityIdentity.from_dict(bad)


def test_a_bare_string_scopes_field_is_not_exploded_into_per_character_grants():
    # Reproduced before the fix: frozenset(str(s) for s in "admin") loaded FIVE scopes -- a, d, i,
    # m, n -- from a record that named one, authorizing things nobody granted; and a bare "*"
    # loaded as the wildcard.
    with pytest.raises(ValueError, match="not str"):
        AuthorizationDecision.from_dict(_decision_record(scopes="admin"))
    with pytest.raises(ValueError, match="not str"):
        AuthorizationDecision.from_dict(_decision_record(scopes="*"))
    # Direct construction had the identical hazard and is refused the same way.
    with pytest.raises(ValueError, match="not str"):
        AuthorizationDecision(
            decision_id="d1",
            capability=CapabilityIdentity.from_dict(_IDENTITY_RECORD),
            outcome=AuthorizationOutcome.GRANTED,
            issued_by="root",
            scopes="admin",
            decided_at=datetime(2020, 1, 1, tzinfo=UTC),
        )


def test_scope_containers_the_library_itself_produces_are_still_accepted():
    # Negative control: as_dict() emits `sorted(self.scopes)` (a list) and in-process callers pass a
    # frozenset or a set literal -- refusing a bare string must not refuse any of those.
    issued = AuthorizationDecision(
        decision_id="d1",
        capability=CapabilityIdentity.from_dict(_IDENTITY_RECORD),
        outcome=AuthorizationOutcome.GRANTED,
        issued_by="root",
        scopes=frozenset({"read", "write"}),
        decided_at=datetime(2020, 1, 1, tzinfo=UTC),
    )
    assert AuthorizationDecision.from_dict(issued.as_dict()).scopes == issued.scopes
    for container in ({"read"}, ["read"], ("read",), frozenset({"read"})):
        assert AuthorizationDecision.from_dict(_decision_record(scopes=container)).scopes == frozenset({"read"})


# ---------------------------------------------------------------------------
# mixle.capability.summarize -- a ragged moment broke the "never raises" contract
# ---------------------------------------------------------------------------


class _RaggedMoments:
    """A HasMoments object whose mean() has no rectangular array form (unlike-shaped components)."""

    def mean(self):
        return [[1.0, 2.0], [3.0]]

    def variance(self):
        return 4.0

    def entropy(self):
        return 0.5


def test_a_ragged_moment_is_reported_rather_than_raised_out_of_the_whole_summary():
    # Reproduced before the fix: np.asarray(value) sat OUTSIDE capture()'s try, so a ragged mean made
    # summarize() raise ValueError -- contradicting its own "never raises on a partially-featured
    # distribution" contract and discarding every other statistic already computed.
    summary = capability.summarize(_RaggedMoments())

    assert summary["_status"]["mean"]["status"] == "unmeasurable"
    assert summary["mean"] == [[1.0, 2.0], [3.0]]
    # the statistics that WERE measurable are still reported
    assert summary["variance"] == 4.0
    assert summary["_status"]["variance"]["status"] == "available"
    assert summary["_status"]["entropy"]["status"] == "available"


class _VectorMoments:
    def mean(self):
        return np.asarray([1.0, 2.0])

    def variance(self):
        return np.asarray([4.0, 9.0])


def test_rectangular_moments_are_still_measured_and_shaped():
    # Negative control: the widened except must not swallow ordinary array-valued moments.
    summary = capability.summarize(_VectorMoments())
    np.testing.assert_array_equal(summary["mean"], np.asarray([1.0, 2.0]))
    np.testing.assert_array_equal(summary["std"], np.asarray([2.0, 3.0]))
    assert summary["_status"]["mean"] == {"status": "available", "shape": [2]}


# ---------------------------------------------------------------------------
# mixle.enumeration.rescore -- int() truncation ran before the guard meant to catch it
# ---------------------------------------------------------------------------


class _DraftIndex:
    """Rank i -> ((i,), -i): the minimal draft-index surface RescoredIndex consumes."""

    def __init__(self, n: int = 50) -> None:
        self.n = n

    def unrank(self, i):
        if i >= self.n:
            raise IndexError(i)
        return (i,), -float(i)


def _target(seqs):
    return np.asarray([-1.0 / (1.0 + s[0]) for s in seqs])


@pytest.mark.parametrize("bad", [2.9, -0.5, True, np.float64(7.9), "5"])
def test_rerank_window_is_not_silently_truncated_past_its_own_guard(bad):
    # Reproduced before the fix: `rerank_window = int(rerank_window)` ran BEFORE the `< 0` guard
    # MXR-080-0233 added, so -0.5 truncated to 0 and was accepted as "no reranking window at all" --
    # the silently-degraded pulled set that guard exists to refuse. 2.9 became 2 and "5" became 5.
    with pytest.raises(ValueError, match="rerank_window"):
        RescoredIndex(_DraftIndex(), _target, rerank_window=bad)


def test_a_boolean_is_not_a_query_size():
    # Reproduced before the fix: `if k < 1` passed for True (True == 1), so top_k(True) quietly
    # answered a one-item query and slice(True, 2) quietly started at rank 1.
    index = RescoredIndex(_DraftIndex(), _target, rerank_window=4)
    with pytest.raises(ValueError, match="k must be"):
        index.top_k(True)
    with pytest.raises(ValueError, match="start must be"):
        index.slice(True, 2)


def test_integer_windows_and_ranks_including_zero_still_work():
    # Negative control: rerank_window=0 is legitimate (rescore_test.py's certification cases rely on
    # it), and numpy integers / integer-valued floats are the shapes callers actually hand in.
    assert RescoredIndex(_DraftIndex(), _target, rerank_window=0).rerank_window == 0
    assert RescoredIndex(_DraftIndex(), _target, rerank_window=np.int64(8)).rerank_window == 8
    assert RescoredIndex(_DraftIndex(), _target, rerank_window=8.0).rerank_window == 8
    index = RescoredIndex(_DraftIndex(), _target, rerank_window=4)
    assert len(index.top_k(3).items) == 3
    assert len(index.slice(0, 2).items) == 2


# ---------------------------------------------------------------------------
# mixle.relations -- fail-open / coercive boundaries on the flow, MILP, ADMM,
# best-first, sampler and subset surfaces
# ---------------------------------------------------------------------------


def test_a_nan_capacity_no_longer_reads_as_an_absent_arc():
    # Reproduced before the fix: `residual[u, v] > 1e-12` and `cap[u, v] > 0.0` are both False for NaN,
    # so max_flow returned value 0.0 on a graph with arcs and min_cut returned capacity 0.0 with an
    # EMPTY cut-edge list -- a certificate that is impossible for any graph that has arcs.
    cap = np.array([[0.0, np.nan, 0.0], [0.0, 0.0, 5.0], [0.0, 0.0, 0.0]])
    with pytest.raises(ValueError, match="must not contain NaN"):
        relations.max_flow(cap, 0, 2)
    with pytest.raises(ValueError, match="must not contain NaN"):
        relations.min_cut(cap, 0, 2)


def test_a_rectangular_matrix_is_not_silently_read_as_its_leading_square_block():
    # Reproduced before the fix: `n = matrix.shape[0]` discarded every column past n -- a 2x3 capacity
    # answered for its 2x2 corner (7.0), a 3x4 distance produced a "tour" over three of four cities.
    with pytest.raises(ValueError, match="square"):
        relations.max_flow(np.array([[0.0, 7.0, 99.0], [0.0, 0.0, 42.0]]), 0, 1)
    with pytest.raises(ValueError, match="square"):
        relations.tsp_held_karp(np.array([[0.0, 1.0, 1.0, 9.0], [1.0, 0.0, 1.0, 9.0], [1.0, 1.0, 0.0, 9.0]]))


def test_source_equal_to_sink_is_refused_instead_of_looping_forever():
    # Reproduced before the fix: `parent[sink]` was never -1, so the augmenting loop never broke and
    # `value += inf` ran forever -- the call HUNG rather than answering or refusing.
    with pytest.raises(ValueError, match="distinct nodes"):
        relations.max_flow(np.array([[0.0, 5.0], [0.0, 0.0]]), 0, 0)


def test_a_nan_supply_no_longer_passes_the_balance_check_it_exists_to_fail():
    # Reproduced before the fix: `abs(nan) >= 1e-6` is False so the sum check passed, and both
    # `supply[i] > 1e-12` and `supply[i] < -1e-12` are False so the node was wired up as balanced --
    # min_cost_flow returned a confident "min-cost feasible flow" for an instance that has none.
    cap = np.array([[0.0, 10.0, 0.0], [0.0, 0.0, 10.0], [0.0, 0.0, 0.0]])
    cost = np.array([[0.0, 1.0, 0.0], [0.0, 0.0, 1.0], [0.0, 0.0, 0.0]])
    with pytest.raises(ValueError, match="supply must be finite"):
        relations.min_cost_flow(cap, cost, np.array([5.0, np.nan, -5.0]))
    # Negative control: an infinite CAPACITY is how an uncapacitated arc is spelled and stays legal
    # (test_h1_flows.py's unbounded-negative-cycle case depends on it).
    unbounded_cap = np.array([[0.0, 10.0, 0.0], [0.0, 0.0, np.inf], [0.0, np.inf, 0.0]])
    unbounded_cost = np.array([[0.0, 1.0, 0.0], [0.0, 0.0, -1.0], [0.0, 0.0, 0.0]])
    with pytest.raises(ValueError, match="unbounded negative cycle"):
        relations.min_cost_flow(unbounded_cap, unbounded_cost, np.array([5.0, 0.0, -5.0]))


def test_milp_integrality_tolerance_cannot_certify_a_fractional_point_as_integral():
    # Reproduced before the fix: `tol` was only ever compared against, so NaN made
    # `abs(x[i] - round(x[i])) > tol` False for every coordinate. The LP relaxation was declared
    # integral, np.round snapped it, and the incumbent kept the UN-rounded LP objective -- so the call
    # returned x=[4, 0] for `x0 + x1 <= 3.5` with objective -3.5 != c @ x = -4.0.
    c, a_ub, b_ub = np.array([-1.0, -1.0]), np.array([[1.0, 1.0]]), np.array([3.5])
    for bad in (float("nan"), 0.6, 1e9, 0.0, -1e-6):
        with pytest.raises(ValueError, match="tol"):
            relations.branch_and_bound_milp(c, a_ub, b_ub, bounds=[(0, 10)] * 2, tol=bad)
    # Negative control: the default tolerance still solves the same instance correctly.
    value, x = relations.branch_and_bound_milp(c, a_ub, b_ub, bounds=[(0, 10)] * 2)
    assert value == pytest.approx(float(c @ x))
    assert float((a_ub @ x)[0]) <= 3.5


def test_milp_integer_index_list_is_validated_against_the_variable_range():
    # Reproduced before the fix: `integer=[-1]` addressed x[n-1] by negative indexing, so the variable
    # the caller named as integral came back fractional; duplicates were re-branched silently.
    c, a_ub, b_ub = np.array([-1.0, -1.0]), np.array([[1.0, 1.0]]), np.array([3.5])
    with pytest.raises(ValueError, match="integer"):
        relations.branch_and_bound_milp(c, a_ub, b_ub, integer=[-1], bounds=[(0, 10)] * 2)
    with pytest.raises(ValueError, match="more than once"):
        relations.branch_and_bound_milp(c, a_ub, b_ub, integer=[0, 0], bounds=[(0, 10)] * 2)
    with pytest.raises(ValueError, match="bounds must have one"):
        relations.branch_and_bound_milp(c, a_ub, b_ub, bounds=[(0, 10)] * 5)


def test_admm_refuses_the_inputs_that_returned_a_plausible_but_meaningless_vector():
    # Reproduced before the fix, each returning a vector with no complaint: `lower=5, upper=0` gave
    # [0, 0, 0] (np.clip applies its bounds in order, so the returned point violates lower <= x);
    # `max_iter=0` returned the all-zero initial z as "the solution"; a NaN bound propagated to every
    # coordinate; rho <= 0 broke the SPD precondition the docstring states.
    rng = np.random.RandomState(0)
    a, b = rng.normal(size=(20, 3)), rng.normal(size=20)
    with pytest.raises(ValueError, match="lower bound must be <="):
        relations.admm_bounded_least_squares(a, b, lower=5.0, upper=0.0)
    with pytest.raises(ValueError, match="must not contain NaN"):
        relations.admm_bounded_least_squares(a, b, lower=np.nan, upper=10.0)
    with pytest.raises(ValueError, match="max_iter"):
        relations.admm_bounded_least_squares(a, b, lower=-10.0, upper=10.0, max_iter=0)
    with pytest.raises(ValueError, match="rho"):
        relations.admm_bounded_least_squares(a, b, lower=-10.0, upper=10.0, rho=-0.5)
    # Negative control: the documented NNLS spelling (upper=inf) must keep working.
    x = relations.admm_bounded_least_squares(a, b, 0.0, np.inf, max_iter=8000)
    assert np.isfinite(x).all()
    assert (x >= -1e-9).all()


def test_a_result_budget_of_zero_yields_nothing_and_a_fractional_one_is_refused():
    # Reproduced before the fix: `emitted += 1` ran BEFORE `emitted >= max_results`, so a budget of 0
    # still yielded one goal and `top(0)` returned a one-element list; a float budget was effectively
    # ceiled (max_results=2.1 produced 3 results) and a negative one behaved like 1.
    graph = {0: [(1, 1.0), (2, 4.0)], 1: [(3, 1.0)], 2: [(3, 1.0)], 3: []}
    successors = graph.__getitem__

    assert list(relations.best_first_paths(0, successors, max_results=0)) == []
    assert list(relations.nearest_first(0, successors, max_results=0)) == []
    assert relations.ShortestPath(0, successors).top(0) == []
    assert relations.EditDistance("ab", "ab").top(0) == []
    for bad in (-5, 2.9):
        with pytest.raises(ValueError, match="max_results"):
            list(relations.best_first_paths(0, successors, max_results=bad))
    # Negative control: an ordinary integer budget is unchanged, and solve() still returns one.
    assert len(relations.ShortestPath(0, successors).top(2)) == 2
    assert relations.ShortestPath(0, successors).solve().objective == pytest.approx(2.0)


def test_relation_sampler_flags_are_exact_and_its_temperature_names_a_measure():
    # Reproduced before the fix: `if self.uniform or ...` meant uniform="false" produced the UNIFORM
    # stream byte-for-byte; a NaN temperature fell into the `not np.isfinite` branch and silently
    # sampled uniformly (only +inf is documented to); a NEGATIVE temperature fell into `<= 0.0` and
    # silently became a point mass on the optimum.
    relation = relations.Assignment(np.array([[1.0, 9.0], [9.0, 1.0]]))
    with pytest.raises(TypeError, match="uniform must be an actual Boolean"):
        relation.sampler(seed=0, uniform="false")
    for bad in (float("nan"), -1.0):
        with pytest.raises(ValueError, match="temperature"):
            relation.sampler(seed=0, temperature=bad)
    with pytest.raises(ValueError, match="k must be"):
        relation.sampler(seed=0, k=2.1)
    # Negative control: the documented settings all still build and draw.
    for kwargs in ({"uniform": True}, {"temperature": np.inf}, {"temperature": 0.0}, {"k": 2}):
        assert len(relation.sampler(seed=0, **kwargs).sample(3)) == 3


def test_best_subset_regression_refuses_a_search_it_cannot_rank_or_size():
    # Reproduced before the fix: a NaN response scored EVERY subset NaN, and NaN comparisons in the
    # ranking sort are all False, so solve() returned whichever subset was built first as "the
    # optimum" with objective NaN. Separately `int(max_size)` truncated 2.9 to 2, accepted the string
    # "3", read True as 1, and let max_size=-1 make solve() return None as though infeasible.
    design = np.array([[1.0, 0.0], [0.0, 1.0], [1.0, 1.0], [2.0, 1.0]])
    with pytest.raises(ValueError, match="y must be finite"):
        relations.BestSubsetRegression(design, np.array([1.0, 2.0, np.nan, 4.0]))
    rng = np.random.RandomState(0)
    x_train = rng.normal(size=(30, 4))
    y_train = x_train[:, 0] * 2.0 + rng.normal(size=30) * 0.1
    for bad in (2.9, "3", True, -1):
        with pytest.raises(ValueError, match="max_size"):
            relations.BestSubsetRegression(x_train, y_train, criterion="rss", max_size=bad)
    # Negative control: an integer cap still bounds the search exactly as before.
    fitted = relations.BestSubsetRegression(x_train, y_train, criterion="rss", max_size=2)
    assert fitted.max_size == 2
    assert max(len(item.value) for item in fitted.enumerator()) == 2


def test_adjacency_certificates_are_not_issued_for_a_graph_the_caller_did_not_describe():
    # Reproduced before the fix: graph_coloring([[0,1],[0,0]]) returned chromatic number 1 with both
    # vertices the SAME color despite a[0,1]=1; max_clique([[0,0],[1,0]]) returned [0,1], not a
    # clique; and a 0.4-weighted matrix was reported as BOTH a maximum clique and a maximum
    # independent set, because max_clique read 0.4 as truthy while max_independent_set truncated it
    # to 0 via dtype=int.
    with pytest.raises(ValueError, match="symmetric"):
        relations.graph_coloring(np.array([[0, 1], [0, 0]]))
    with pytest.raises(ValueError, match="symmetric"):
        relations.max_clique(np.array([[0, 0], [1, 0]]))
    weighted = np.array([[0.0, 0.4], [0.4, 0.0]])
    for entry in (relations.graph_coloring, relations.max_clique, relations.max_independent_set):
        with pytest.raises(ValueError, match="0/1"):
            entry(weighted)
    with pytest.raises(ValueError, match="0/1"):
        relations.graph_coloring(np.array([["", "x"], ["x", ""]], dtype=object))
    # Negative control: the symmetric 0/1 and Boolean matrices the suite actually uses still work.
    triangle = np.array([[0, 1, 1], [1, 0, 1], [1, 1, 0]])
    assert relations.graph_coloring(triangle) == (3, [0, 1, 2])
    assert relations.max_clique(triangle) == [0, 1, 2]
    assert relations.graph_coloring(np.array([[False, True], [True, False]]))[0] == 2
    assert relations.max_independent_set(np.array([[0, 1, 0], [1, 0, 1], [0, 1, 0]])) == [0, 2]


def test_a_missing_arc_and_an_unbounded_one_are_no_longer_the_same_fact():
    # Reproduced before the fix: arc presence is decided by np.isfinite, so NaN AND -inf both read as
    # "missing arc" -- tsp_held_karp answered for a smaller graph, and a -inf cost (an unboundedly
    # profitable arc) was silently dropped rather than reported.
    with pytest.raises(ValueError, match="must not contain NaN"):
        relations.tsp_held_karp(np.array([[0.0, 1.0, 10.0], [1.0, 0.0, np.nan], [10.0, 1.0, 0.0]]))
    with pytest.raises(ValueError, match="-inf"):
        relations.tsp_held_karp(np.array([[0.0, 1.0, 10.0], [1.0, 0.0, -np.inf], [10.0, 1.0, 0.0]]))
    with pytest.raises(ValueError, match="must not contain NaN"):
        relations.min_arborescence(np.array([[np.inf, 1.0, np.nan], [np.inf, np.inf, 1.0], [np.inf] * 3]))
    # Negative control: +inf is the documented spelling of an absent arc and must keep working
    # (min_arborescence_test.py and tsp_test.py both rely on it).
    assert relations.tsp_held_karp(np.array([[0.0, 1.0, np.inf], [np.inf, 0.0, 1.0], [1.0, np.inf, 0.0]])) == (
        3.0,
        [0, 1, 2],
    )
    assert relations.min_arborescence(np.array([[np.inf, 1.0, 2.0], [np.inf, np.inf, 1.0], [np.inf] * 3])) == (
        2.0,
        [-1, 0, 1],
    )


def test_a_fractional_commodity_node_index_is_refused_rather_than_truncated():
    # Reproduced before the fix: `int(demands[kk, 0])` routed a demand row of [0.9, 2.7, 4.0] from
    # node 0 to node 2 without a word. A node index is an exact identity, not a measurement.
    cap = np.array([[0.0, 10.0, 0.0], [0.0, 0.0, 10.0], [0.0, 0.0, 0.0]])
    cost = np.array([[0.0, 1.0, 0.0], [0.0, 0.0, 1.0], [0.0, 0.0, 0.0]])
    with pytest.raises(ValueError, match="node"):
        relations.multicommodity_flow(cap, cost, [[0.9, 2.7, 4.0]])
    # Negative control: integer node indices are unaffected.
    assert relations.multicommodity_flow(cap, cost, [[0, 2, 4.0]]).value == pytest.approx(8.0)


def test_cardinality_bound_counts_variables_exactly():
    # Reproduced before the fix: `float(max_nonzero)` put 2.9 into the `sum z_i <= 2.9` row, so the
    # call silently answered the max-2 problem instead.
    c, a_ub, b_ub = np.array([-3.0, -2.0, -1.0]), np.array([[1.0, 1.0, 1.0]]), np.array([10.0])
    bounds = [(0.0, 5.0)] * 3
    with pytest.raises(ValueError, match="max_nonzero"):
        relations.cardinality_constrained_milp(c, a_ub, b_ub, 2.9, bounds)
    # Negative control: the integer bound is unchanged.
    objective, x = relations.cardinality_constrained_milp(c, a_ub, b_ub, 2, bounds)
    assert objective == pytest.approx(-25.0)
    assert int((np.abs(x) > 1e-9).sum()) == 2
