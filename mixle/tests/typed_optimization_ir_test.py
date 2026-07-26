"""Stage 0/1 tests for the experimental statistically typed runtime."""

import json

import numpy as np
import pytest

from mixle.experimental.typed_runtime import (
    ArtifactKind,
    ConsistencyRequirement,
    ContractEvidenceKind,
    ContractRegistry,
    CurvatureKind,
    EffectiveContextMeasurement,
    IssueSeverity,
    MeasurementCatalog,
    MergeLaw,
    ObjectiveKind,
    StateSemantics,
    UpdateContract,
    UpdateGraph,
    UpdateGraphError,
    UpdateKind,
    WorkMeasurement,
    compile_update_graph,
    validate_update_graph,
)
from mixle.experimental.typed_runtime.compiler import _bind_child_estimators, _distribution_children
from mixle.experimental.typed_runtime.contracts import ConvergenceCertificate
from mixle.inference import optimize
from mixle.models.grad_leaf import GradLeaf
from mixle.stats import (
    CompositeDistribution,
    CompositeEstimator,
    GaussianDistribution,
    GaussianEstimator,
    MixtureDistribution,
    MixtureEstimator,
)
from mixle.stats.bayes.normal_gamma import NormalGammaDistribution
from mixle.stats.compute.pdist import ParameterEstimator, ProbabilityDistribution

pytestmark = [pytest.mark.experimental, pytest.mark.fast]


class _NoTouchGaussian(GaussianDistribution):
    """Compilation must not call either method."""

    def sampler(self, seed=None):
        raise AssertionError("compiler sampled the model")

    def log_density(self, x):
        raise AssertionError("compiler scored the model")


class _FakeParameter:
    def __init__(self, size=1):
        self.size = size

    def numel(self):
        return self.size


class _FakeModule:
    """Torch module protocol without importing torch."""

    def __init__(self, size=3):
        self.param = _FakeParameter(size)
        self.log_norm = 0.0

    def parameters(self):
        return [self.param]

    def state_dict(self):
        return {"param": self.param}

    def load_state_dict(self, state):
        self.param = state["param"]

    def log_density(self, x):
        return x

    def energy(self, x):
        return x


def _gaussian_prior():
    return NormalGammaDistribution(0.0, 1.0e-3, 1.0, 1.0)


class ContractInferenceTest:
    def test_unregistered_subclass_compiles_unknown_without_touching_model(self):
        model = _NoTouchGaussian(0.0, 1.0)
        graph = compile_update_graph(model, GaussianEstimator(), nobs=100)
        root = graph.node(graph.root_node)

        assert root.contract.objective_kind is ObjectiveKind.UNKNOWN
        assert root.contract.update_kind is UpdateKind.UNKNOWN
        assert root.contract.merge_law is MergeLaw.NON_MERGEABLE
        assert root.contract.curvature_kind is CurvatureKind.UNAVAILABLE
        assert root.contract.state_semantics == frozenset({StateSemantics.EXTERNAL_STATE})
        assert not root.contract.exact
        assert root.contract.convergence_certificate is ConvergenceCertificate.UNKNOWN
        assert root.cost.source == "structural_proxy"
        assert root.cost.compute_units > 0.0

        audited = compile_update_graph(GaussianDistribution(0.0, 1.0), GaussianEstimator()).node("n0000").contract
        assert audited.objective_kind is ObjectiveKind.MLE
        assert audited.update_kind is UpdateKind.EXACT_CLOSED_FORM
        assert audited.convergence_certificate is ConvergenceCertificate.UNKNOWN

    def test_prior_is_audited_but_method_named_variational_objective_is_unknown(self):
        prior = _gaussian_prior()
        map_graph = compile_update_graph(GaussianDistribution(0.0, 1.0, prior=prior), GaussianEstimator(prior=prior))
        assert map_graph.node(map_graph.root_node).contract.objective_kind is ObjectiveKind.MAP

        class VariationalGaussian(GaussianDistribution):
            def seq_local_elbo(self, enc):
                return np.zeros(len(enc))

        vb_graph = compile_update_graph(VariationalGaussian(0.0, 1.0), GaussianEstimator())
        variational_contract = vb_graph.node(vb_graph.root_node).contract
        assert variational_contract.objective_kind is ObjectiveKind.UNKNOWN
        assert variational_contract.update_kind is UpdateKind.UNKNOWN

    def test_neural_and_surrogate_semantics_are_not_guessed_from_names(self):
        module = _FakeModule()
        mle_leaf = GradLeaf(module)
        mle_graph = compile_update_graph(mle_leaf, mle_leaf.estimator())
        mle = mle_graph.node(mle_graph.root_node).contract
        assert mle.objective_kind is ObjectiveKind.UNKNOWN
        assert mle.update_kind is UpdateKind.UNKNOWN
        assert mle.exact is False
        assert mle.state_semantics == frozenset({StateSemantics.EXTERNAL_STATE})

        custom_leaf = GradLeaf(module, loss=lambda *_: 0.0)
        surrogate = compile_update_graph(custom_leaf, custom_leaf.estimator()).node("n0000").contract
        assert surrogate.objective_kind is ObjectiveKind.UNKNOWN
        assert surrogate.outer_objective_compatible is False


class DependencyGraphTest:
    def test_mixture_compiles_component_axis_and_child_invalidation(self):
        model = MixtureDistribution(
            [GaussianDistribution(-1.0, 1.0), GaussianDistribution(1.0, 1.0)],
            [0.5, 0.5],
        )
        estimator = MixtureEstimator([GaussianEstimator(), GaussianEstimator()])
        graph = compile_update_graph(model, estimator, nobs=250)

        assert len(graph.nodes) == 3
        root = graph.node(graph.root_node)
        assert root.contract.update_kind is UpdateKind.GENERALIZED_EM
        assert root.contract.decomposition_axes == ("component",)
        children = [node for node in graph.nodes if node.node_id != root.node_id]
        for child in children:
            assert graph.invalidated_by(child.node_id) == (child.node_id, root.node_id)

    def test_shared_child_is_one_node_with_every_parent_dependency(self):
        shared = GaussianDistribution(0.0, 1.0)
        model = CompositeDistribution((shared, shared))
        estimator = CompositeEstimator((GaussianEstimator(), GaussianEstimator()))
        graph = compile_update_graph(model, estimator)

        assert len(graph.nodes) == 2
        child = next(node for node in graph.nodes if node.node_id != graph.root_node)
        shared_edges = [edge for edge in graph.edges if edge.source_node == child.node_id]
        assert len(shared_edges) == 1
        assert graph.invalidated_by(child.node_id) == (child.node_id, graph.root_node)

    def test_duplicate_dependency_edges_are_rejected(self):
        graph = compile_update_graph(
            MixtureDistribution(
                [GaussianDistribution(-1.0, 1.0), GaussianDistribution(1.0, 1.0)],
                [0.5, 0.5],
            ),
            MixtureEstimator([GaussianEstimator(), GaussianEstimator()]),
        )
        edge = graph.edges[0]
        with pytest.raises(UpdateGraphError, match="edges must be unique"):
            UpdateGraph(graph.nodes, (edge, edge), graph.root_node)

    def test_graph_is_json_explainable_without_runtime_objects(self):
        graph = compile_update_graph(GaussianDistribution(0.0, 1.0), GaussianEstimator())
        payload = graph.as_dict()
        json.dumps(payload)
        assert "model" not in payload["nodes"][0]
        explanation = graph.explain()
        assert "exact_closed_form" in explanation
        assert "structural_proxy" in explanation


class DeclarationAndValidationTest:
    def test_caller_owned_registry_overrides_inference(self):
        contract = UpdateContract(
            objective_kind=ObjectiveKind.CONSTRAINT,
            update_kind=UpdateKind.PROXIMAL,
            merge_law=MergeLaw.NON_MERGEABLE,
            consistency=ConsistencyRequirement.LOCAL_ONLY,
            exact=False,
            outer_objective_compatible=False,
            declared_by="test_registry",
            evidence_kind=ContractEvidenceKind.EXPLICIT_DECLARATION,
            evidence_id="test:registry-v1",
        )
        registry = ContractRegistry()
        registry.register(GaussianDistribution, contract)
        graph = compile_update_graph(GaussianDistribution(0.0, 1.0), GaussianEstimator(), registry=registry)
        assert graph.node("n0000").contract is contract

        fresh_registry = ContractRegistry()
        fresh = compile_update_graph(GaussianDistribution(0.0, 1.0), GaussianEstimator(), registry=fresh_registry)
        assert fresh.node("n0000").contract.objective_kind is ObjectiveKind.MLE

    def test_validation_warns_for_mutable_and_surrogate_updates(self):
        module = _FakeModule()
        leaf = GradLeaf(module, loss=lambda *_: 0.0)
        contract = UpdateContract(
            objective_kind=ObjectiveKind.USER_SURROGATE,
            update_kind=UpdateKind.FIRST_ORDER,
            merge_law=MergeLaw.NON_MERGEABLE,
            state_semantics=frozenset(
                {
                    StateSemantics.MUTABLE_PARAMETERS,
                    StateSemantics.STOCHASTIC_RNG,
                }
            ),
            outer_objective_compatible=False,
            exact=False,
            declared_by="test_explicit_neural_contract",
            evidence_kind=ContractEvidenceKind.EXPLICIT_DECLARATION,
            evidence_id="test:neural-contract-v1",
        )
        graph = compile_update_graph(leaf, leaf.estimator(), contract_overrides={"root": contract})
        issues = validate_update_graph(graph, strict=True)
        assert {issue.code for issue in issues} == {"transaction-required", "surrogate-objective"}
        assert all(issue.severity is IssueSeverity.WARNING for issue in issues)

    def test_path_contract_override_wins(self):
        override = UpdateContract(
            objective_kind=ObjectiveKind.MLE,
            update_kind=UpdateKind.FROZEN,
            merge_law=MergeLaw.REPLICATED,
            writes=frozenset(),
            declared_by="path_override",
            evidence_kind=ContractEvidenceKind.EXPLICIT_DECLARATION,
            evidence_id="test:path-override-v1",
        )
        graph = compile_update_graph(
            GaussianDistribution(0.0, 1.0),
            GaussianEstimator(),
            contract_overrides={"root": override},
        )
        assert graph.node("n0000").contract is override


class MeasurementVocabularyTest:
    def test_measurement_catalog_replaces_proxy_with_median_receipt(self):
        catalog = MeasurementCatalog()
        catalog.extend(
            [
                WorkMeasurement("GaussianDistribution", UpdateKind.EXACT_CLOSED_FORM, "cpu", 0.3, 30, 10, 80),
                WorkMeasurement("GaussianDistribution", UpdateKind.EXACT_CLOSED_FORM, "cpu", 0.1, 10, 30, 120),
                WorkMeasurement("GaussianDistribution", UpdateKind.EXACT_CLOSED_FORM, "cpu", 0.2, 20, 20, 100),
            ]
        )
        graph = compile_update_graph(
            GaussianDistribution(0.0, 1.0), GaussianEstimator(), backend="cpu", measurements=catalog
        )
        cost = graph.node("n0000").cost
        assert cost.measured
        assert cost.wall_time_seconds == pytest.approx(0.2)
        assert cost.compute_units == pytest.approx(20.0)
        assert cost.communication_bytes == 20
        assert cost.peak_memory_bytes == 120

    def test_effective_context_keeps_source_active_and_generated_counts_separate(self):
        receipt = EffectiveContextMeasurement(
            source_horizon_tokens=1_000_000_000_000,
            materialized_tokens=100_000,
            attended_tokens=32_000,
            evidence_nodes=500,
            evidence_edges=900,
            context_actions=12,
            retrieval_actions=7,
            generation_actions=2,
            verification_actions=3,
            verified_claim_fraction=0.95,
            stopped_reason="expected_value_below_cost",
        )
        assert receipt.active_to_source_ratio == pytest.approx(1.0e-7)
        assert receipt.as_dict()["generation_actions"] == 2

        with pytest.raises(ValueError, match="source horizon"):
            EffectiveContextMeasurement(source_horizon_tokens=10, materialized_tokens=11)


class BehavioralParityTest:
    def test_compilation_does_not_change_an_existing_fit(self):
        data = [-2.0, -1.0, 0.0, 1.0, 2.0]
        estimator = GaussianEstimator()
        model = GaussianDistribution(0.0, 1.0)

        baseline = optimize(data, estimator, prev_estimate=model, max_its=1, out=None)
        graph = compile_update_graph(model, estimator, nobs=len(data))
        after_compile = optimize(data, estimator, prev_estimate=model, max_its=1, out=None)

        assert graph.node("n0000").contract.update_kind is UpdateKind.EXACT_CLOSED_FORM
        assert after_compile.mu == baseline.mu
        assert after_compile.sigma2 == baseline.sigma2
        assert model.mu == 0.0
        assert model.sigma2 == 1.0


def test_artifact_vocabulary_contains_context_and_graph_state():
    assert ArtifactKind.CONTEXT_SUMMARIES.value == "context_summaries"
    assert ArtifactKind.GRAPH_STATE.value == "graph_state"


class CapabilityProbeRobustnessTest:
    """``infer_update_contract``'s own docstring promises "a conservative contract without
    scoring, sampling, or mutating the model", and ``_compute_band``'s own comment says
    "eligibility probing must never break compilation" -- but the sibling probes (``_update_kind``,
    ``_merge_law``, ``_curvature_kind``, ``_decomposition`` caught only a narrow exception tuple,
    and ``_state_semantics`` caught nothing at all) let an unexpected exception from inspecting the
    model/estimator's own attributes escape uncaught. A model whose own attribute access misbehaves
    is plausible for the arbitrary, possibly-experimental model trees this compiler exists to
    introspect, so this used to crash compilation outright instead of degrading to the documented
    conservative default.
    """

    def test_a_broken_attribute_property_degrades_to_the_conservative_default_instead_of_crashing(self):
        class _BrokenAttributeGaussian(GaussianDistribution):
            @property
            def state_dict(self):
                raise RuntimeError("simulated broken property access")

            def load_state_dict(self, *a, **kw):
                pass

            def parameters(self):
                return []

        graph = compile_update_graph(_BrokenAttributeGaussian(0.0, 1.0), GaussianEstimator())
        contract = graph.node(graph.root_node).contract
        assert contract.update_kind is UpdateKind.UNKNOWN
        assert contract.state_semantics == frozenset({StateSemantics.EXTERNAL_STATE})


class FailClosedContractEvidenceTest:
    def test_arbitrary_estimator_and_latent_method_names_remain_unknown(self):
        class ArbitraryEstimator(ParameterEstimator):
            def accumulator_factory(self):
                raise AssertionError("compiler called accumulator_factory")

            def estimate(self, nobs, suff_stat):
                raise AssertionError("compiler called estimate")

        class NamedLatentGaussian(GaussianDistribution):
            def seq_posterior(self, values):
                raise AssertionError("compiler called seq_posterior")

        contract = compile_update_graph(NamedLatentGaussian(0.0, 1.0), ArbitraryEstimator()).node("n0000").contract
        assert contract.objective_kind is ObjectiveKind.UNKNOWN
        assert contract.update_kind is UpdateKind.UNKNOWN
        assert not contract.exact
        assert contract.convergence_certificate is ConvergenceCertificate.UNKNOWN

    def test_invalid_partial_enum_is_rejected_instead_of_guessed(self):
        estimator = GaussianEstimator()
        estimator.objective_kind = "not-an-objective"
        with pytest.raises(TypeError, match="invalid partial declaration"):
            compile_update_graph(GaussianDistribution(0.0, 1.0), estimator)

    def test_callable_contract_hook_and_estimator_factory_are_never_invoked(self):
        class HookGaussian(GaussianDistribution):
            hook_calls = 0
            factory_calls = 0

            def update_contract(self):
                type(self).hook_calls += 1
                raise AssertionError("contract hook executed")

            def estimator(self, pseudo_count=None):
                type(self).factory_calls += 1
                raise AssertionError("estimator factory executed")

        model = HookGaussian(0.0, 1.0)
        with pytest.raises(TypeError, match="static UpdateContract"):
            compile_update_graph(model, GaussianEstimator())
        assert HookGaussian.hook_calls == 0

        del HookGaussian.update_contract
        graph = compile_update_graph(model)
        assert graph.node("n0000").contract.update_kind is UpdateKind.FROZEN
        assert HookGaussian.factory_calls == 0

    def test_registry_rejects_factories_and_unproven_monotonicity(self):
        registry = ContractRegistry()
        with pytest.raises(TypeError, match="UpdateContract"):
            registry.register(GaussianDistribution, lambda *_: None)

        unsupported = UpdateContract(
            objective_kind=ObjectiveKind.MLE,
            update_kind=UpdateKind.EXACT_CLOSED_FORM,
            merge_law=MergeLaw.ADDITIVE,
            convergence_certificate=ConvergenceCertificate.MONOTONE_CERTIFIED,
            declared_by="test_without_acceptance_proof",
            evidence_kind=ContractEvidenceKind.EXPLICIT_DECLARATION,
            evidence_id="test:missing-proof-v1",
        )
        with pytest.raises(ValueError, match="acceptance-proof"):
            registry.register(GaussianDistribution, unsupported)

        empty_proof = UpdateContract(
            objective_kind=ObjectiveKind.MLE,
            update_kind=UpdateKind.EXACT_CLOSED_FORM,
            merge_law=MergeLaw.ADDITIVE,
            convergence_certificate=ConvergenceCertificate.MONOTONE_CERTIFIED,
            declared_by="test_empty_acceptance_proof",
            evidence_kind=ContractEvidenceKind.EXPLICIT_DECLARATION,
            evidence_id="test:empty-proof-v1",
            notes=("acceptance-proof:   ",),
        )
        with pytest.raises(ValueError, match="acceptance-proof"):
            registry.register(GaussianDistribution, empty_proof)

        exact_unknown_objective = UpdateContract(
            objective_kind=ObjectiveKind.UNKNOWN,
            update_kind=UpdateKind.EXACT_CLOSED_FORM,
            merge_law=MergeLaw.ADDITIVE,
            outer_objective_compatible=False,
            declared_by="test_unknown_objective",
            evidence_kind=ContractEvidenceKind.EXPLICIT_DECLARATION,
            evidence_id="test:unknown-objective-v1",
        )
        with pytest.raises(ValueError, match="identify the objective"):
            registry.register(GaussianDistribution, exact_unknown_objective)

    def test_unused_paths_and_invalid_binding_values_are_rejected(self):
        model = GaussianDistribution(0.0, 1.0)
        with pytest.raises(TypeError, match="binding values"):
            compile_update_graph(model, bindings={"root": object()})
        with pytest.raises(ValueError, match="unused bindings"):
            compile_update_graph(model, bindings={"root -> absent": GaussianEstimator()})
        with pytest.raises(ValueError, match="unused contract overrides"):
            compile_update_graph(
                model,
                contract_overrides={
                    "root -> absent": UpdateContract(
                        objective_kind=ObjectiveKind.MLE,
                        update_kind=UpdateKind.FROZEN,
                        merge_law=MergeLaw.REPLICATED,
                        writes=frozenset(),
                        declared_by="test_unused_path",
                        evidence_kind=ContractEvidenceKind.EXPLICIT_DECLARATION,
                        evidence_id="test:unused-path-v1",
                    )
                },
            )

    def test_mapping_keys_cannot_execute_repr_str_or_equality_during_compilation(self):
        calls = []

        class HostileKey:
            def __repr__(self):
                calls.append("repr")
                raise AssertionError("repr called")

            def __str__(self):
                calls.append("str")
                raise AssertionError("str called")

            def __eq__(self, other):
                calls.append("eq")
                raise AssertionError("equality called")

            def __hash__(self):
                return 1

        class MappingModel(ProbabilityDistribution):
            def __init__(self):
                self.children = {HostileKey(): GaussianDistribution(0.0, 1.0)}

            def log_density(self, x):
                raise NotImplementedError

            def sampler(self, seed=None):
                raise NotImplementedError

            def estimator(self, pseudo_count=None):
                raise NotImplementedError

        graph = compile_update_graph(MappingModel())
        assert len(graph.nodes) == 2
        assert "<key-0>" in graph.node("n0001").path
        assert calls == []


class _TwoLeafModel(ProbabilityDistribution):
    """A model with two children whose attribute names share no canonical form with any
    estimator attribute name -- forces both into _bind_child_estimators's unbound/positional path."""

    def __init__(self, a, b):
        self.aaa_leaf = a
        self.zzz_leaf = b

    def log_density(self, x):
        raise NotImplementedError

    def sampler(self, seed=None):
        raise NotImplementedError

    def estimator(self, pseudo_count=None):
        raise NotImplementedError


class _TwoLeafEstimator(ParameterEstimator):
    """Attribute names deliberately don't share a canonical form with _TwoLeafModel's, AND their
    alphabetical order is reversed relative to the true (constructor-argument) correspondence --
    est_for_zzz_leaf lands in the alphabetically-first attribute, est_for_aaa_leaf in the second."""

    def __init__(self, est_for_zzz_leaf, est_for_aaa_leaf):
        self.attr_a = est_for_zzz_leaf
        self.attr_z = est_for_aaa_leaf

    def accumulator_factory(self):
        raise NotImplementedError

    def estimate(self, nobs, suff_stat):
        raise NotImplementedError


class BindChildEstimatorsTestCase:
    """``_bind_child_estimators``'s positional fallback paired ANY equal-count group of unbound
    model/estimator children by their (both alphabetical, since both lists are built from
    ``sorted(vars(...).items())``) enumeration order alone -- with no semantic check that the
    pairing is actually right. Two or more unbound children on each side is a silent coin flip
    whenever the model's and estimator's naming conventions don't happen to sort into the same
    relative order: it can swap which estimator a child is bound to, producing a wrong-but-
    structurally-valid compiled contract with no error raised. A single straggler on each side is
    safe (there is only one possible pairing either way) and is still resolved.
    """

    def test_two_or_more_unbound_children_are_left_unbound_not_silently_swapped(self):
        est_for_aaa = GaussianEstimator(name="est_for_aaa")
        est_for_zzz = GaussianEstimator(name="est_for_zzz")
        model = _TwoLeafModel(GaussianDistribution(0.0, 1.0), GaussianDistribution(5.0, 2.0))
        estimator = _TwoLeafEstimator(est_for_zzz_leaf=est_for_zzz, est_for_aaa_leaf=est_for_aaa)

        model_children = _distribution_children(model)
        bound = _bind_child_estimators(model_children, estimator)

        assert bound == {}  # neither child gets a (possibly wrong) estimator rather than a guess

    def test_a_single_unbound_child_on_each_side_still_binds_positionally(self):
        class _OneLeafModel(ProbabilityDistribution):
            def __init__(self, a):
                self.solo_leaf = a

            def log_density(self, x):
                raise NotImplementedError

            def sampler(self, seed=None):
                raise NotImplementedError

            def estimator(self, pseudo_count=None):
                raise NotImplementedError

        class _OneLeafEstimator(ParameterEstimator):
            def __init__(self, a):
                self.totally_different_name = a

            def accumulator_factory(self):
                raise NotImplementedError

            def estimate(self, nobs, suff_stat):
                raise NotImplementedError

        solo_est = GaussianEstimator(name="solo_est")
        model = _OneLeafModel(GaussianDistribution(0.0, 1.0))
        bound = _bind_child_estimators(_distribution_children(model), _OneLeafEstimator(solo_est))

        assert bound[0] is solo_est
