"""Executable contracts for native and Megatron distributed-gradient backends."""

from __future__ import annotations

import copy

import pytest

torch = pytest.importorskip("torch")

from mixle.experimental.typed_runtime.contracts import (
    CostEstimate,
    MergeLaw,
    ObjectiveKind,
    StateSemantics,
    UpdateContract,
    UpdateKind,
)
from mixle.experimental.typed_runtime.distributed import plan_distributed_updates
from mixle.experimental.typed_runtime.graph import UpdateGraph, UpdateNode
from mixle.models.transformer import build_causal_lm
from mixle.utils.parallel.dcp_checkpoint import load_training_state, save_training_state
from mixle.utils.parallel.megatron_training import MegatronBridgeBackend
from mixle.utils.parallel.torch_training import TorchDistributedBackend
from mixle.utils.parallel.training_contracts import (
    CollectiveKind,
    ParallelAxis,
    ParallelPlan,
)
from mixle.utils.parallel.training_launchers import LightningFabricLauncher


def test_parallel_plan_distinguishes_physical_mesh_from_overlapping_expert_groups():
    plan = ParallelPlan(dp_replicate=8, tp=2, pp=2, cp=2, ep=4, etp=2)

    assert plan.world_size == 64
    assert plan.mesh == (
        ("dp_replicate", "tp", "pp", "cp"),
        (8, 2, 2, 2),
    )
    assert ParallelAxis.EP in plan.active_axes
    with pytest.raises(ValueError, match="divide the data-parallel"):
        ParallelPlan(dp_replicate=2, ep=4)


def test_torch_backend_rejects_pipeline_and_expert_axes_instead_of_ignoring_them():
    backend = TorchDistributedBackend()

    with pytest.raises(NotImplementedError, match="pp"):
        backend.capabilities.validate(ParallelPlan(pp=2))
    with pytest.raises(NotImplementedError, match="ep"):
        backend.capabilities.validate(ParallelPlan(dp_replicate=2, ep=2))


def test_lightning_launcher_rejects_model_axes_before_importing_lightning():
    with pytest.raises(NotImplementedError, match="only owns data parallelism"):
        LightningFabricLauncher().create(plan=ParallelPlan(tp=2))


def test_native_microbatch_and_accumulation_matches_one_full_sgd_update():
    torch.manual_seed(4)
    reference = build_causal_lm(13, d_model=16, n_layer=1, n_head=2, block=4)
    distributed = copy.deepcopy(reference)
    first_x = torch.randint(0, 13, (2, 4))
    first_y = torch.randint(0, 13, (2, 4))
    second_x = torch.randint(0, 13, (2, 4))
    second_y = torch.randint(0, 13, (2, 4))

    optimizer = torch.optim.SGD(reference.parameters(), lr=0.01, momentum=0.9)
    first_loss = torch.nn.functional.cross_entropy(
        reference(first_x, return_all_logits=True).reshape(-1, 13), first_y.reshape(-1)
    )
    second_loss = torch.nn.functional.cross_entropy(
        reference(second_x, return_all_logits=True).reshape(-1, 13), second_y.reshape(-1)
    )
    ((first_loss + second_loss) / 2.0).backward()
    optimizer.step()

    session = TorchDistributedBackend().prepare(
        distributed,
        plan=ParallelPlan(microbatches=2, gradient_accumulation_steps=2),
        device="cpu",
        optimizer="sgd",
        lr=0.01,
        max_grad_norm=None,
    )
    skipped = session.train_batch(first_x, first_y)
    committed = session.train_batch(second_x, second_y)

    assert skipped.skipped
    assert not committed.skipped
    assert committed.local_examples == 4
    assert committed.local_tokens == 16
    assert committed.global_tokens == 16
    for expected, actual in zip(reference.parameters(), distributed.parameters()):
        torch.testing.assert_close(actual, expected, rtol=1.0e-5, atol=1.0e-6)


def test_pending_gradients_cannot_be_omitted_from_a_checkpoint(tmp_path):
    module = build_causal_lm(13, d_model=16, n_layer=1, n_head=2, block=4)
    session = TorchDistributedBackend().prepare(
        module,
        plan=ParallelPlan(gradient_accumulation_steps=2),
        device="cpu",
        optimizer="sgd",
    )
    batch = torch.randint(0, 13, (2, 4))
    session.train_batch(batch, batch)

    with pytest.raises(RuntimeError, match="finish gradient accumulation"):
        session.save_checkpoint(str(tmp_path))
    session.close()


def test_complete_checkpoint_restores_model_optimizer_rng_and_clocks(tmp_path):
    torch.manual_seed(8)
    model = build_causal_lm(11, d_model=16, n_layer=1, n_head=2, block=4)
    optimizer = torch.optim.SGD(model.parameters(), lr=0.1, momentum=0.9)
    loss = model(torch.randint(0, 11, (2, 4))).sum()
    loss.backward()
    optimizer.step()
    expected = [parameter.detach().clone() for parameter in model.parameters()]
    save_training_state(
        model,
        optimizer,
        str(tmp_path),
        step=17,
        loader_state={"epoch": 3, "batch": 9},
        parallel_plan=ParallelPlan(),
        typed_scheduler_state={"clock": 5},
    )

    with torch.no_grad():
        for parameter in model.parameters():
            parameter.zero_()
    payload = load_training_state(model, optimizer, str(tmp_path))

    assert payload["step"] == 17
    assert payload["loader_state"] == {"epoch": 3, "batch": 9}
    assert payload["typed_scheduler_state"] == {"clock": 5}
    assert (tmp_path / "_SUCCESS").is_file()
    for restored, target in zip(model.parameters(), expected):
        torch.testing.assert_close(restored, target, rtol=0.0, atol=0.0)


def test_incomplete_checkpoint_is_never_loaded(tmp_path):
    model = torch.nn.Linear(2, 1)
    optimizer = torch.optim.SGD(model.parameters(), lr=0.1)

    with pytest.raises(RuntimeError, match="incomplete"):
        load_training_state(model, optimizer, str(tmp_path))


def test_megatron_provider_receives_every_parallel_dimension(monkeypatch):
    class Provider:
        def finalize(self):
            return self

        def provide_distributed_model(self, **kwargs):
            self.provide_kwargs = kwargs
            return "distributed-model"

    backend = MegatronBridgeBackend()
    monkeypatch.setattr(backend, "_require_bridge", lambda: object())
    provider = Provider()
    plan = ParallelPlan(dp_replicate=4, tp=2, pp=2, cp=2, ep=2, etp=2)

    session = backend.prepare(provider, plan=plan, wrap_with_ddp=False)

    assert session.module == "distributed-model"
    assert provider.tensor_model_parallel_size == 2
    assert provider.pipeline_model_parallel_size == 2
    assert provider.context_parallel_size == 2
    assert provider.expert_model_parallel_size == 2
    assert provider.expert_tensor_parallel_size == 2
    assert provider.sequence_parallel
    assert provider.provide_kwargs == {"wrap_with_ddp": False}


def test_typed_first_order_update_lowers_to_reduce_scatter_and_model_axis_transfers():
    contract = UpdateContract(
        objective_kind=ObjectiveKind.MLE,
        update_kind=UpdateKind.FIRST_ORDER,
        merge_law=MergeLaw.NON_MERGEABLE,
        state_semantics=frozenset({StateSemantics.MUTABLE_PARAMETERS, StateSemantics.MUTABLE_OPTIMIZER}),
        exact=False,
    )
    graph = UpdateGraph(
        nodes=(UpdateNode("n0", "root", "Torch", "Grad", contract, CostEstimate(), 10),),
        edges=(),
        root_node="n0",
    )

    updates = plan_distributed_updates(graph, ParallelPlan(dp_shard=2, tp=2))

    assert updates[0].collective is CollectiveKind.REDUCE_SCATTER
    assert updates[0].mesh_axes == (ParallelAxis.DP_SHARD,)
    assert updates[1].node_id == "n0:tp"
    assert updates[1].collective is CollectiveKind.ALL_REDUCE
