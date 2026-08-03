"""MXR-080-1905: typed-runtime controls, public logs, and stage-then-commit ordering.

Every test here names the state that was reproduced before the fix. The finding is about the layer
*around* the receipts that earlier findings repaired: the arguments that decide what a receipt will
say, the lists that hold the receipts once written, and the ordering between the state change and
the record of it.
"""

import copy

import numpy as np
import pytest

from mixle.experimental.typed_runtime import (
    ArtifactKind,
    CanaryVerdict,
    CommitStatus,
    ContractEvidenceKind,
    CostEstimate,
    GainEvidence,
    GainPerCostScheduler,
    MergeLaw,
    NodeExecutionStatus,
    NodeTerminalReceipt,
    ObjectiveKind,
    ProposalPacket,
    SchedulerConfig,
    StateSemantics,
    TransactionParticipant,
    UpdateContract,
    UpdateGraph,
    UpdateKind,
    UpdateNode,
    payload_fingerprint,
)
from mixle.experimental.typed_runtime.benchmark import (
    BenchmarkPoint,
    FailureKind,
    FailureLedger,
    FailureReceipt,
    ObjectiveTarget,
    TargetDirection,
    TimeToTargetTrace,
)
from mixle.experimental.typed_runtime.cache import VersionedArtifactCache
from mixle.experimental.typed_runtime.context_ir import ContextGraph, ContextNode, ContextNodeKind
from mixle.experimental.typed_runtime.graph_memory import (
    GraphMemoryCache,
    partition_context_graph,
)
from mixle.experimental.typed_runtime.transaction import (
    RuntimeVersions,
    TransactionalCoordinator,
)

pytestmark = [pytest.mark.experimental]


# --------------------------------------------------------------------------------------------
# fixtures


def _mutable_contract():
    return UpdateContract(
        objective_kind=ObjectiveKind.MLE,
        update_kind=UpdateKind.FIRST_ORDER,
        merge_law=MergeLaw.NON_MERGEABLE,
        state_semantics=frozenset(
            {
                StateSemantics.MUTABLE_PARAMETERS,
                StateSemantics.MUTABLE_OPTIMIZER,
                StateSemantics.STOCHASTIC_RNG,
            }
        ),
        reads=frozenset({ArtifactKind.PARAMETERS, ArtifactKind.OPTIMIZER_STATE, ArtifactKind.RNG_STATE}),
        writes=frozenset({ArtifactKind.PARAMETERS, ArtifactKind.OPTIMIZER_STATE, ArtifactKind.RNG_STATE}),
        exact=False,
        declared_by="mxr-1905",
    )


def _single_node_graph():
    node = UpdateNode(
        "node",
        "root",
        "MutableFixture",
        "MutableEstimator",
        _mutable_contract(),
        CostEstimate(compute_units=1.0),
        2,
    )
    return UpdateGraph((node,), (), "node")


def _packet(proposal_id="p1"):
    return ProposalPacket(
        proposal_id=proposal_id,
        run_id="run",
        model_id="model",
        node_id="node",
        shard_id="worker-0",
        base_model_version=0,
        dependency_versions={"node": 0},
        update_kind=UpdateKind.FIRST_ORDER,
        objective_kind=ObjectiveKind.MLE,
        payload={"delta": np.array([1.0, -1.0])},
        writes=frozenset({ArtifactKind.PARAMETERS, ArtifactKind.OPTIMIZER_STATE, ArtifactKind.RNG_STATE}),
    )


def _participants(state):
    kinds = {
        StateSemantics.MUTABLE_PARAMETERS: ArtifactKind.PARAMETERS,
        StateSemantics.MUTABLE_OPTIMIZER: ArtifactKind.OPTIMIZER_STATE,
        StateSemantics.STOCHASTIC_RNG: ArtifactKind.RNG_STATE,
    }

    def one(name, semantics):
        return TransactionParticipant(
            name,
            frozenset({semantics}),
            lambda: copy.deepcopy(state[name]),
            lambda value: state.__setitem__(name, copy.deepcopy(value)),
            lambda: payload_fingerprint(state[name]),
            frozenset({kinds[semantics]}),
        )

    return (
        one("parameters", StateSemantics.MUTABLE_PARAMETERS),
        one("optimizer", StateSemantics.MUTABLE_OPTIMIZER),
        one("rng", StateSemantics.STOCHASTIC_RNG),
    )


def _coordinator(*, canary=None, **kwargs):
    state = {
        "parameters": np.array([0.0, 0.0]),
        "optimizer": {"step": 0, "moment": np.array([0.0, 0.0])},
        "rng": {"counter": 7},
    }

    def apply(proposal):
        state["parameters"] = state["parameters"] + proposal.payload["delta"]
        state["optimizer"]["step"] += 1
        state["rng"]["counter"] += 1

    coordinator = TransactionalCoordinator(
        _single_node_graph(),
        apply,
        canary or (lambda batch: CanaryVerdict(True, "ok", 0.0, 1.0)),
        run_id="run",
        model_id="model",
        participants=_participants(state),
        **kwargs,
    )
    return state, coordinator


def _schedulable_contract(update=UpdateKind.COORDINATE):
    writes = (
        frozenset() if update is UpdateKind.FROZEN else UpdateContract.__dataclass_fields__["writes"].default_factory()
    )
    return UpdateContract(
        objective_kind=ObjectiveKind.MLE,
        update_kind=update,
        merge_law=MergeLaw.REPLICATED if update is UpdateKind.FROZEN else MergeLaw.ADDITIVE,
        writes=writes,
        outer_objective_compatible=True,
        exact=True,
        declared_by="mxr-1905",
        evidence_kind=ContractEvidenceKind.EXPLICIT_DECLARATION,
        evidence_id="mxr-1905:contract-v1",
    )


def _schedulable_node(node_id, update=UpdateKind.COORDINATE):
    return UpdateNode(
        node_id=node_id,
        path="root -> %s" % node_id,
        model_type="Fixture",
        estimator_type="FixtureEstimator",
        contract=_schedulable_contract(update),
        cost=CostEstimate(compute_units=1.0),
        parameter_count=1,
    )


def _schedulable_graph():
    nodes = (
        _schedulable_node("a"),
        _schedulable_node("b"),
        _schedulable_node("root", UpdateKind.FROZEN),
    )
    return UpdateGraph(nodes, (), "root")


def _context_graph(count=4, tokens=4):
    graph = ContextGraph()
    for index in range(count):
        graph.add_node(ContextNode("n-%d" % index, ContextNodeKind.MEMORY, "body-%d" % index, tokens))
    return graph


# --------------------------------------------------------------------------------------------
# 1. public mutable logs


class PublicLedgerIntegrityTest:
    """Reproduced: recorded history could be rewritten through the attribute that reports it."""

    def test_coordinator_receipt_ledger_cannot_be_cleared_by_a_caller(self):
        # Reproduced: coordinator.receipts.clear() erased the committed history, and
        # ledger_fingerprint() -- computed from that list -- silently changed to match.
        _state, coordinator = _coordinator()
        receipt = coordinator.commit(_packet())
        assert receipt.status is CommitStatus.ACCEPTED
        fingerprint = coordinator.ledger_fingerprint()

        assert coordinator.receipts == (receipt,)
        with pytest.raises(AttributeError):
            coordinator.receipts.clear()
        with pytest.raises(AttributeError):
            coordinator.proposal_receipts.append({"forged": True})

        assert len(coordinator.receipts) == 1
        assert len(coordinator.proposal_receipts) == 1
        assert coordinator.ledger_fingerprint() == fingerprint

    def test_unverified_rollback_poison_is_not_a_settable_attribute(self):
        # Reproduced: coordinator.poisoned = False re-enabled a coordinator whose rollback could not
        # be verified, so it committed on top of state nothing confirmed was restored.
        state = {
            "parameters": np.array([0.0, 0.0]),
            "optimizer": {"step": 0, "moment": np.array([0.0, 0.0])},
            "rng": {"counter": 7},
        }
        kinds = {
            StateSemantics.MUTABLE_PARAMETERS: ArtifactKind.PARAMETERS,
            StateSemantics.MUTABLE_OPTIMIZER: ArtifactKind.OPTIMIZER_STATE,
            StateSemantics.STOCHASTIC_RNG: ArtifactKind.RNG_STATE,
        }

        def participant(name, semantics):
            return TransactionParticipant(
                name,
                frozenset({semantics}),
                lambda: copy.deepcopy(state[name]),
                lambda value: None,  # restore does nothing: rollback cannot be verified
                lambda: payload_fingerprint(state[name]),
                frozenset({kinds[semantics]}),
            )

        def apply(proposal):
            state["parameters"] = state["parameters"] + proposal.payload["delta"]

        coordinator = TransactionalCoordinator(
            _single_node_graph(),
            apply,
            lambda batch: CanaryVerdict(False, "rejected-on-purpose"),
            run_id="run",
            model_id="model",
            participants=(
                participant("parameters", StateSemantics.MUTABLE_PARAMETERS),
                participant("optimizer", StateSemantics.MUTABLE_OPTIMIZER),
                participant("rng", StateSemantics.STOCHASTIC_RNG),
            ),
        )
        first = coordinator.commit(_packet())
        assert first.status is CommitStatus.ROLLBACK_FAILED
        assert coordinator.poisoned

        with pytest.raises(AttributeError):
            coordinator.poisoned = False
        assert coordinator.poisoned
        second = coordinator.commit(_packet("p2"))
        assert second.reason == "coordinator-poisoned-by-unverified-rollback"

    def test_failure_ledger_cannot_have_a_failed_oracle_erased(self):
        # Reproduced: ledger.receipts.clear() flipped all_oracles_passed from False to True.
        ledger = FailureLedger()
        ledger.record(FailureReceipt("bench", "case", FailureKind.NUMERICAL, "oracle", True, False, "missed"))
        assert ledger.as_dict()["all_oracles_passed"] is False
        with pytest.raises(AttributeError):
            ledger.receipts.clear()
        assert ledger.as_dict()["all_oracles_passed"] is False
        assert len(ledger.failed_oracles) == 1

    def test_time_to_target_points_cannot_bypass_the_monotone_contract(self):
        # Reproduced: trace.points.append(BenchmarkPoint(0, 0.95, 0.0)) after a step-1 point built a
        # trace whose steps ran 1, 0 -- and achieved read True from the out-of-order point.
        trace = TimeToTargetTrace(
            "bench",
            "strategy",
            ObjectiveTarget("accuracy", TargetDirection.MAXIMIZE, 0.9),
        )
        trace.record(BenchmarkPoint(1, 0.5, 1.0, operation_count=10))
        with pytest.raises(AttributeError):
            trace.points.append(BenchmarkPoint(0, 0.95, 0.0))
        assert [point.step for point in trace.points] == [1]
        assert trace.achieved is False
        with pytest.raises(ValueError, match="advance in step"):
            trace.record(BenchmarkPoint(0, 0.95, 0.0))

    def test_replay_log_history_cannot_be_rewritten_in_place(self):
        # Reproduced: record() deep-copies its inputs, but log.entries[0] = other and
        # log.entries.clear() rewrote exactly the history that copying protects.
        from mixle.experimental.typed_runtime.replay import ReplayLog

        _state, coordinator = _coordinator()
        packet = _packet()
        receipt = coordinator.commit(packet)
        log = ReplayLog()
        from mixle.experimental.typed_runtime import ProposalBatch

        batch = ProposalBatch(receipt.batch_id, (packet,))
        log.record(batch, receipt)
        assert len(log.entries) == 1
        with pytest.raises(AttributeError):
            log.entries.clear()
        with pytest.raises(TypeError):
            log.entries[0] = None
        assert len(log.entries) == 1
        # indexing into the view still reaches the recorded batch, as the replay tests rely on
        assert log.entries[0].batch.batch_id == receipt.batch_id


# --------------------------------------------------------------------------------------------
# 2. exact controls


class ExactControlTest:
    """Reproduced: fractional / NumPy / string values accepted where an exact scalar was meant."""

    @pytest.mark.parametrize("bad", [0.0, 0.5, True, "0"])
    def test_runtime_versions_refuse_inexact_model_versions(self, bad):
        # Reproduced: RuntimeVersions(0.0, ...) constructed, then the coordinator built on it
        # advanced 0.0 -> 1.0 and raised out of CommitReceipt, which requires isinstance(int).
        with pytest.raises(TypeError, match="exact integer"):
            RuntimeVersions(bad, {"node": 0})

    def test_runtime_versions_canonicalize_numpy_integers(self):
        # np.int64 is a genuine integer: canonicalized, not refused. Before the fix it was stored
        # as-is and the receipt then refused to serialize it.
        versions = RuntimeVersions(np.int64(2), {"node": np.int64(3)})
        assert type(versions.model_version) is int
        assert versions.as_dict() == {"model_version": 2, "node_versions": {"node": 3}}

    def test_fractional_model_version_no_longer_advances_state_before_failing(self):
        # Reproduced: with model_version=0.0 the commit raised AFTER model_version moved to 1.0,
        # the node version to 1, and the proposal ids were marked seen -- with zero receipts written.
        with pytest.raises(TypeError, match="exact integer"):
            _coordinator(versions=RuntimeVersions(0.0, {"node": 0}))

    def test_coordinator_version_vector_is_not_aliased_to_the_caller_dict(self):
        # Reproduced: owned["node"] = 999 rewrote the coordinator's version vector from outside, and
        # the next commit was rejected with dependency-version-mismatch against a version the
        # coordinator never reached.
        owned = {"node": 0}
        _state, coordinator = _coordinator(versions=RuntimeVersions(0, owned))
        owned["node"] = 999
        assert coordinator.versions.node_versions == {"node": 0}
        assert coordinator.commit(_packet()).status is CommitStatus.ACCEPTED

    @pytest.mark.parametrize("bad", ["", "no", 0, 1])
    def test_enforce_monotone_objective_requires_an_actual_boolean(self, bad):
        # Reproduced: enforce_monotone_objective="" disabled the strict-objective gate, so a canary
        # reporting an objective REGRESSION of 5.0 -> 1.0 committed as ACCEPTED.
        with pytest.raises(TypeError, match="actual Boolean"):
            _coordinator(enforce_monotone_objective=bad)

    def test_objective_regression_is_still_refused_with_the_gate_on(self):
        # The other half of the guard: with the gate enabled (the default), the regression that the
        # string form let through is rejected. This is what the flag is protecting.
        _state, coordinator = _coordinator(canary=lambda batch: CanaryVerdict(True, "ok", 5.0, 1.0))
        receipt = coordinator.commit(_packet())
        assert receipt.status is CommitStatus.ROLLED_BACK
        assert receipt.reason == "strict-objective-regression"

    def test_explicitly_disabling_the_monotone_gate_still_works(self):
        # Guard-overreach check: False is a legitimate configuration and must keep working.
        _state, coordinator = _coordinator(
            canary=lambda batch: CanaryVerdict(True, "ok", 5.0, 1.0),
            enforce_monotone_objective=False,
        )
        assert coordinator.commit(_packet()).status is CommitStatus.ACCEPTED

    @pytest.mark.parametrize("bad", ["no", "", 1, 0])
    def test_bootstrap_unmeasured_requires_an_actual_boolean(self, bad):
        # Reproduced: SchedulerConfig(bootstrap_unmeasured="no") stored the string and was read by a
        # bare `if`, so every unmeasured node was bootstrapped by a config that said not to.
        with pytest.raises(TypeError, match="actual Boolean"):
            SchedulerConfig(bootstrap_unmeasured=bad)

    @pytest.mark.parametrize("bad", [2.5, 2.0, True, "2"])
    def test_max_skip_rounds_requires_an_exact_count(self, bad):
        # Reproduced: max_skip_rounds=2.5 constructed and sat between two reachable skip counts.
        with pytest.raises(TypeError, match="exact integer"):
            SchedulerConfig(max_skip_rounds=bad)

    @pytest.mark.parametrize("bad", [0.5, 2.0, "3"])
    def test_gain_evidence_versions_and_sample_counts_are_exact(self, bad):
        with pytest.raises(TypeError, match="exact integer"):
            GainEvidence("a", ObjectiveKind.MLE, 1.0, bad)
        with pytest.raises(TypeError, match="exact integer"):
            GainEvidence("a", ObjectiveKind.MLE, 1.0, 0, sample_count=bad)

    def test_gain_evidence_accepts_numpy_counts(self):
        # Guard-overreach check: NumPy counters are what measurement code actually produces.
        row = GainEvidence("a", ObjectiveKind.MLE, 1.0, np.int64(2), sample_count=np.int32(8))
        assert (type(row.model_version), type(row.sample_count)) == (int, int)

    def test_schedule_rejects_an_inexact_model_version_at_the_boundary(self):
        # Reproduced: model_version=np.int64(3) ran the whole scheduling pass and then failed inside
        # ScheduleReceipt, reporting a receipt defect for a caller argument. NumPy is accepted now
        # and canonicalized; a float is refused before any work happens.
        scheduler = GainPerCostScheduler()
        with pytest.raises(TypeError, match="exact integer"):
            scheduler.schedule(_schedulable_graph(), model_version=0.5)
        receipt = scheduler.schedule(_schedulable_graph(), model_version=np.int64(3))
        assert type(receipt.model_version) is int
        assert receipt.model_version == 3

    def test_schedule_refuses_a_string_candidate_set(self):
        # Reproduced: candidate_nodes="ab" scheduled the nodes a and b, because a string is a
        # Collection of its own characters and they happened to be node ids.
        scheduler = GainPerCostScheduler()
        with pytest.raises(TypeError, match="iterate as its characters"):
            scheduler.schedule(_schedulable_graph(), model_version=0, candidate_nodes="ab")

    def test_schedule_still_accepts_the_ordinary_candidate_forms(self):
        # Guard-overreach check: sets, tuples and lists of ids are what real callers pass.
        for candidates in ({"a"}, ("a",), ["a"]):
            receipt = GainPerCostScheduler().schedule(_schedulable_graph(), model_version=0, candidate_nodes=candidates)
            assert receipt.eligible_nodes == ("a",)

    @pytest.mark.parametrize("bad", [0.5, 1.0, True, "1"])
    def test_terminal_receipt_round_index_is_exact(self, bad):
        # Reproduced: NodeTerminalReceipt(0.5, ...) constructed and then compared unequal to every
        # integer round, so the outcome belonged to no schedule.
        with pytest.raises(TypeError, match="exact integer"):
            NodeTerminalReceipt(bad, "a", NodeExecutionStatus.COMMITTED, "evidence")

    @pytest.mark.parametrize("bad", [8.5, True, "8"])
    def test_graph_memory_cache_bounds_are_exact(self, bad):
        # Reproduced: maximum_tokens=8.5 and maximum_tokens=True both passed the "<= 0" test and
        # became the threshold every eviction loop compares an integer token sum against.
        with pytest.raises(TypeError, match="exact integer"):
            GraphMemoryCache(maximum_tokens=bad, maximum_partitions=2)
        with pytest.raises(TypeError, match="exact integer"):
            GraphMemoryCache(maximum_tokens=8, maximum_partitions=bad)

    def test_partition_bounds_are_exact(self):
        # Reproduced: maximum_nodes=1.5 was compared against len(selected) and admitted two nodes.
        with pytest.raises(TypeError, match="exact integer"):
            partition_context_graph(_context_graph(), maximum_tokens=8, maximum_nodes=1.5)
        with pytest.raises(TypeError, match="exact integer"):
            partition_context_graph(_context_graph(), maximum_tokens=8.5, maximum_nodes=2)

    def test_pinned_is_a_boolean_pin_decision_not_a_truthy_one(self):
        # Reproduced: put(partition, graph, pinned="no") PINNED the partition -- _evict reads
        # `if not entry.pinned` -- so the next insertion was evicted instead of it, and the cache
        # serialized "pinned": "no".
        graph = _context_graph()
        cache = GraphMemoryCache(maximum_tokens=8, maximum_partitions=1)
        plan = partition_context_graph(graph, maximum_tokens=8, maximum_nodes=2)
        with pytest.raises(TypeError, match="actual Boolean"):
            cache.put(plan.partitions[0], graph, pinned="no")
        # a real pin still works, and still protects its partition
        cache.put(plan.partitions[0], graph, pinned=True)
        assert cache.as_dict()["entries"][0]["pinned"] is True


# --------------------------------------------------------------------------------------------
# 3. stage-then-commit ordering


class StageThenCommitTest:
    """Reproduced: state advanced, then the record of it was refused."""

    def test_invalid_written_artifact_does_not_bump_a_cache_generation(self):
        # Reproduced: cache.invalidate("node", "parameters") -- the string, not the ArtifactKind --
        # advanced the node generation 0 -> 1 and DELETED the cached entry, then raised out of
        # InvalidationReceipt. The invalidation happened; nothing recorded it.
        cache = VersionedArtifactCache(_single_node_graph())
        cache.put("node", ArtifactKind.PARAMETERS, "value")
        assert cache.generation("node") == 0

        with pytest.raises(TypeError, match="ArtifactKind"):
            cache.invalidate("node", "parameters")

        assert cache.generation("node") == 0
        assert cache.contains("node", ArtifactKind.PARAMETERS)
        assert cache.get("node", ArtifactKind.PARAMETERS) == "value"

    def test_a_valid_invalidation_still_bumps_and_removes(self):
        # Guard-overreach check: the ordinary path is unchanged.
        cache = VersionedArtifactCache(_single_node_graph())
        cache.put("node", ArtifactKind.PARAMETERS, "value")
        receipt = cache.invalidate("node", ArtifactKind.PARAMETERS)
        assert receipt.invalidated_nodes == ("node",)
        assert receipt.generations == {"node": 1}
        assert cache.generation("node") == 1
        assert not cache.contains("node", ArtifactKind.PARAMETERS)

    def test_invalidate_many_refuses_a_bare_string_source(self):
        # A bare string iterates as its characters, so "node" named four one-character nodes.
        cache = VersionedArtifactCache(_single_node_graph())
        with pytest.raises(TypeError, match="iterate as its characters"):
            cache.invalidate_many("node")
        assert cache.generation("node") == 0

    def test_accepted_commit_writes_its_receipt_before_advancing_versions(self):
        # The ordering the fractional-version reproduction exposed. That input is now refused by
        # RuntimeVersions before it reaches commit, so this is not itself a reproduction -- it pins
        # the reordered path: the receipt is built from a STAGED vector, and the coordinator's own
        # version state matches what the receipt says it moved to.
        _state, coordinator = _coordinator()
        receipt = coordinator.commit(_packet())
        assert receipt.status is CommitStatus.ACCEPTED
        assert receipt.versions_before["model_version"] == 0
        assert receipt.versions_after["model_version"] == 1
        assert coordinator.versions.model_version == 1
        assert coordinator.versions.node_versions == {"node": 1}
        assert coordinator.receipts[-1] is receipt

    def test_terminal_outcomes_advance_fairness_clocks_only_with_a_receipt(self):
        # record_terminal stages the fairness clocks and publishes them together with the
        # completion receipt, so the scheduler cannot end up past a round it cannot account for.
        scheduler = GainPerCostScheduler(SchedulerConfig(budget_fraction=1.0))
        receipt = scheduler.schedule(_schedulable_graph(), model_version=0)
        completion = scheduler.record_terminal(
            receipt,
            tuple(
                NodeTerminalReceipt(receipt.round_index, node_id, NodeExecutionStatus.COMMITTED, "t:%s" % node_id)
                for node_id in receipt.selected_nodes
            ),
            terminal_id="terminal-0",
        )
        assert set(completion.states_after) == set(receipt.eligible_nodes)
        assert scheduler.states == dict(completion.states_after)

    @pytest.mark.torch
    def test_pilot_does_not_leave_the_process_global_rng_reseeded(self):
        # Reproduced: run_graph_memory_pilot called torch.manual_seed(seed) and never put the entry
        # state back, so a caller who seeded their process, ran the pilot and then drew got
        # [-0.164139, -2.214069, -1.388025] where they would otherwise have got
        # [-0.837879, 0.45636, 0.348142]. Nothing in the receipt said the stream had moved.
        torch = pytest.importorskip("torch")
        from mixle.experimental.typed_runtime import run_graph_memory_pilot

        torch.manual_seed(999)
        expected = torch.randn(3)

        torch.manual_seed(999)
        run_graph_memory_pilot(
            seed=17,
            source_nodes=32,
            train_examples=16,
            test_examples=16,
            updates=2,
            microbatch_size=8,
            accumulation_steps=1,
        )
        assert torch.equal(torch.randn(3), expected)

    def test_mismatched_outcomes_leave_the_fairness_clocks_untouched(self):
        # NOT a reproduction: this ordering already held. record_terminal's checks all ran before
        # the old in-place clock loop, so this passed before the change too. It is here to pin the
        # behaviour the staging rewrite must preserve.
        scheduler = GainPerCostScheduler(SchedulerConfig(budget_fraction=1.0))
        receipt = scheduler.schedule(_schedulable_graph(), model_version=0)
        with pytest.raises(ValueError, match="exactly the selected nodes"):
            scheduler.record_terminal(
                receipt,
                (NodeTerminalReceipt(receipt.round_index, "a", NodeExecutionStatus.COMMITTED, "t:a"),),
                terminal_id="terminal-0",
            )
        assert scheduler.states == {}
        # the schedule is still pending, so the round cannot be silently skipped
        with pytest.raises(RuntimeError, match="terminal execution receipts"):
            scheduler.schedule(_schedulable_graph(), model_version=0)
