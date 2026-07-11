"""Integrated graph-memory mixture-of-experts pilot with recovery receipts."""

from __future__ import annotations

import copy
import time
from dataclasses import dataclass
from typing import Any

import numpy as np

from mixle.experimental.typed_runtime.benchmark import FailureKind, FailureReceipt
from mixle.experimental.typed_runtime.context_attention import (
    AttentionCandidate,
    ContextAttentionConfig,
    bounded_context_attention,
)
from mixle.experimental.typed_runtime.context_ir import ContextGraph, ContextNode, ContextNodeKind
from mixle.experimental.typed_runtime.contracts import MergeLaw, ObjectiveKind, UpdateContract, UpdateKind
from mixle.experimental.typed_runtime.geometry import (
    BatchSemanticsReceipt,
    GeometryRouterConfig,
    OptimizerFamily,
    describe_parameters,
    route_optimizer_geometry,
)
from mixle.experimental.typed_runtime.proposal import payload_fingerprint
from mixle.experimental.typed_runtime.torch_optimizer import build_routed_torch_optimizer


@dataclass(frozen=True)
class PilotStrategyReceipt:
    """Fixed-budget training and evaluation result for one context/optimizer strategy."""

    strategy: str
    context_mode: str
    optimizer: str
    final_loss: float
    test_accuracy: float
    time_to_target_updates: int | None
    target_accuracy: float
    elapsed_seconds: float
    batch: BatchSemanticsReceipt
    optimizer_families: tuple[str, ...]
    maximum_active_context_tokens: int
    source_horizon_tokens: int

    def as_dict(self) -> dict[str, Any]:
        """Return a JSON-compatible strategy receipt."""

        return {
            "strategy": self.strategy,
            "context_mode": self.context_mode,
            "optimizer": self.optimizer,
            "final_loss": self.final_loss,
            "test_accuracy": self.test_accuracy,
            "time_to_target_updates": self.time_to_target_updates,
            "target_accuracy": self.target_accuracy,
            "elapsed_seconds": self.elapsed_seconds,
            "batch": self.batch.as_dict(),
            "optimizer_families": list(self.optimizer_families),
            "maximum_active_context_tokens": self.maximum_active_context_tokens,
            "source_horizon_tokens": self.source_horizon_tokens,
        }


@dataclass(frozen=True)
class RecoveryDrillReceipt:
    """Checkpoint/restart comparison for model, optimizer, and RNG state."""

    checkpoint_update: int
    final_update: int
    model_bitwise_equal: bool
    optimizer_state_equal: bool
    rng_state_equal: bool
    uninterrupted_model_hash: str
    resumed_model_hash: str

    @property
    def passed(self) -> bool:
        return self.model_bitwise_equal and self.optimizer_state_equal and self.rng_state_equal

    def as_dict(self) -> dict[str, Any]:
        return {
            "checkpoint_update": self.checkpoint_update,
            "final_update": self.final_update,
            "model_bitwise_equal": self.model_bitwise_equal,
            "optimizer_state_equal": self.optimizer_state_equal,
            "rng_state_equal": self.rng_state_equal,
            "uninterrupted_model_hash": self.uninterrupted_model_hash,
            "resumed_model_hash": self.resumed_model_hash,
            "passed": self.passed,
        }


@dataclass(frozen=True)
class GraphMemoryPilotReceipt:
    """Complete context-quality, optimizer, and recovery evidence for the pilot."""

    seed: int
    source_nodes: int
    local_adamw: PilotStrategyReceipt
    graph_adamw: PilotStrategyReceipt
    graph_routed: PilotStrategyReceipt
    recovery: RecoveryDrillReceipt
    failure_receipts: tuple[FailureReceipt, ...]

    @property
    def graph_quality_gain(self) -> float:
        return self.graph_adamw.test_accuracy - self.local_adamw.test_accuracy

    @property
    def active_context_bounded(self) -> bool:
        return self.graph_adamw.maximum_active_context_tokens < self.graph_adamw.source_horizon_tokens

    def as_dict(self) -> dict[str, Any]:
        return {
            "seed": self.seed,
            "source_nodes": self.source_nodes,
            "local_adamw": self.local_adamw.as_dict(),
            "graph_adamw": self.graph_adamw.as_dict(),
            "graph_routed": self.graph_routed.as_dict(),
            "recovery": self.recovery.as_dict(),
            "failure_receipts": [receipt.as_dict() for receipt in self.failure_receipts],
            "graph_quality_gain": self.graph_quality_gain,
            "active_context_bounded": self.active_context_bounded,
        }


def _state_hash(state: dict[str, Any]) -> str:
    payload = {}
    for key, value in state.items():
        if isinstance(value, dict):
            payload[key] = _state_hash(value)
        elif hasattr(value, "detach"):
            payload[key] = value.detach().cpu().numpy()
        elif isinstance(value, list):
            payload[key] = [item.detach().cpu().numpy() if hasattr(item, "detach") else item for item in value]
        else:
            payload[key] = value
    return payload_fingerprint(payload)


def _optimizer_hash(optimizer: Any) -> str:
    state = optimizer.state_dict()
    normalized = {
        "state": {
            str(key): {
                name: value.detach().cpu().numpy() if hasattr(value, "detach") else value for name, value in row.items()
            }
            for key, row in state["state"].items()
        },
        "param_groups": [
            {key: value for key, value in group.items() if key != "params"} for group in state["param_groups"]
        ],
    }
    return payload_fingerprint(normalized)


def run_graph_memory_pilot(
    *,
    seed: int = 17,
    source_nodes: int = 256,
    train_examples: int = 192,
    test_examples: int = 96,
    updates: int = 80,
    microbatch_size: int = 32,
    accumulation_steps: int = 2,
    target_accuracy: float = 0.9,
) -> GraphMemoryPilotReceipt:
    """Run a fixed-budget graph-context MoE ablation and deterministic restart drill."""

    try:
        import torch
    except ImportError as error:
        raise ImportError("run_graph_memory_pilot requires PyTorch.") from error
    if source_nodes < 32 or train_examples < 16 or test_examples < 16 or updates < 2:
        raise ValueError("pilot sizes are too small for a meaningful fixed-budget comparison.")
    if microbatch_size < 1 or accumulation_steps < 1:
        raise ValueError("microbatch_size and accumulation_steps must be positive.")
    np_rng = np.random.default_rng(seed)
    torch.manual_seed(seed)
    dimension = 12
    keys = np_rng.normal(size=(source_nodes, dimension))
    keys /= np.linalg.norm(keys, axis=1, keepdims=True)
    bits = np_rng.integers(0, 2, size=source_nodes)
    graph = ContextGraph()
    candidates = []
    for index in range(source_nodes):
        node_id = "memory-%05d" % index
        graph.add_node(ContextNode(node_id, ContextNodeKind.MEMORY, "bit=%d" % bits[index], 1))
        candidates.append(AttentionCandidate(node_id, keys[index], np.array([float(bits[index])]), index))
    candidates = tuple(candidates)

    def dataset(size: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        target = np_rng.integers(0, source_nodes // 2, size=size)
        regimes = np_rng.integers(0, 2, size=size)
        labels = np.bitwise_xor(bits[target], regimes)
        queries = 20.0 * keys[target]
        return queries, regimes, labels

    train_queries, train_regimes, train_labels = dataset(train_examples)
    test_queries, test_regimes, test_labels = dataset(test_examples)
    local_config = ContextAttentionConfig(exact_near_tokens=8, retrieved_nodes=0, maximum_active_tokens=8)
    graph_config = ContextAttentionConfig(exact_near_tokens=8, retrieved_nodes=1, maximum_active_tokens=9)

    def features(queries: np.ndarray, config: ContextAttentionConfig) -> tuple[np.ndarray, int]:
        values = []
        max_active = 0
        for query in queries:
            result = bounded_context_attention(
                query,
                candidates,
                graph,
                config,
                source_horizon_tokens=source_nodes,
            )
            values.append(float(result.value[0]))
            max_active = max(max_active, result.receipt.active_tokens)
        return np.asarray(values, dtype=np.float32)[:, None], max_active

    local_train, local_active = features(train_queries, local_config)
    local_test, _ = features(test_queries, local_config)
    graph_train, graph_active = features(train_queries, graph_config)
    graph_test, _ = features(test_queries, graph_config)
    train_regime = np.eye(2, dtype=np.float32)[train_regimes]
    test_regime = np.eye(2, dtype=np.float32)[test_regimes]

    class RegimeMoE(torch.nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.router = torch.nn.Linear(2, 2)
            self.experts = torch.nn.ModuleList([torch.nn.Linear(1, 2) for _ in range(2)])

        def forward(self, context: Any, regime: Any) -> Any:
            gates = torch.softmax(self.router(regime), dim=-1)
            expert_logits = torch.stack([expert(context) for expert in self.experts], dim=1)
            return torch.sum(gates.unsqueeze(-1) * expert_logits, dim=1)

    initial = RegimeMoE()
    initial_state = copy.deepcopy(initial.state_dict())
    contract = UpdateContract(
        objective_kind=ObjectiveKind.MLE,
        update_kind=UpdateKind.FIRST_ORDER,
        merge_law=MergeLaw.LOW_RANK,
        exact=False,
    )

    def tensors(context: np.ndarray, regime: np.ndarray, labels: np.ndarray) -> tuple[Any, Any, Any]:
        return (
            torch.from_numpy(context),
            torch.from_numpy(regime),
            torch.from_numpy(labels.astype(np.int64)),
        )

    local_tensors = tensors(local_train, train_regime, train_labels)
    local_test_tensors = tensors(local_test, test_regime, test_labels)
    graph_tensors = tensors(graph_train, train_regime, train_labels)
    graph_test_tensors = tensors(graph_test, test_regime, test_labels)

    def make_model() -> Any:
        model = RegimeMoE()
        model.load_state_dict(copy.deepcopy(initial_state))
        return model

    def accuracy(model: Any, rows: tuple[Any, Any, Any]) -> float:
        with torch.no_grad():
            predictions = model(rows[0], rows[1]).argmax(dim=-1)
            return float((predictions == rows[2]).float().mean())

    def train_strategy(
        name: str,
        context_mode: str,
        train_rows: tuple[Any, Any, Any],
        test_rows: tuple[Any, Any, Any],
        active_tokens: int,
        *,
        routed: bool,
        recovery: bool = False,
    ) -> tuple[PilotStrategyReceipt, Any, Any, RecoveryDrillReceipt | None]:
        # Every strategy sees the same stochastic sample stream. The RNG is also
        # checkpointed below, so restart parity covers minibatch selection.
        torch.manual_seed(seed + 1)
        model = make_model()
        if routed:
            plan = route_optimizer_geometry(
                describe_parameters(model),
                contract,
                GeometryRouterConfig(matrix_min_elements=2, matrix_min_dimension=1, max_state_to_parameter_ratio=8.0),
            )
            optimizer = build_routed_torch_optimizer(model, plan, lr=0.04)
            families = tuple(sorted({route.family.value for route in plan.routes}))
            optimizer_name = "typed-routed"
        else:
            plan = None
            optimizer = torch.optim.AdamW(model.parameters(), lr=0.04)
            families = (OptimizerFamily.ADAMW.value,)
            optimizer_name = "adamw"
        checkpoint = updates // 2
        snapshot = None
        accuracies = []
        started = time.perf_counter()
        loss = None
        for step in range(1, updates + 1):
            optimizer.zero_grad()
            accumulated_loss = 0.0
            for _ in range(accumulation_steps):
                indices = torch.randint(0, train_rows[0].shape[0], (microbatch_size,))
                loss = torch.nn.functional.cross_entropy(
                    model(train_rows[0][indices], train_rows[1][indices]),
                    train_rows[2][indices],
                )
                (loss / accumulation_steps).backward()
                accumulated_loss += float(loss.detach()) / accumulation_steps
            optimizer.step()
            accuracies.append(accuracy(model, test_rows))
            if recovery and step == checkpoint:
                snapshot = (
                    copy.deepcopy(model.state_dict()),
                    copy.deepcopy(optimizer.state_dict()),
                    torch.random.get_rng_state().clone(),
                )
        elapsed = time.perf_counter() - started
        time_to_target = next((index + 1 for index, value in enumerate(accuracies) if value >= target_accuracy), None)
        batch = BatchSemanticsReceipt(
            microbatch_size,
            microbatch_size * active_tokens,
            float(microbatch_size),
            accumulation_steps,
            1,
            updates,
            "mean",
            1.0 / accumulation_steps,
            updates,
        )
        strategy = PilotStrategyReceipt(
            name,
            context_mode,
            optimizer_name,
            accumulated_loss,
            accuracies[-1],
            time_to_target,
            target_accuracy,
            elapsed,
            batch,
            families,
            active_tokens,
            source_nodes,
        )
        recovery_receipt = None
        if recovery:
            if snapshot is None:
                raise RuntimeError("recovery checkpoint was not captured.")
            uninterrupted_model_hash = _state_hash(model.state_dict())
            uninterrupted_optimizer_hash = _optimizer_hash(optimizer)
            uninterrupted_rng = torch.random.get_rng_state().clone()
            resumed = make_model()
            resumed.load_state_dict(copy.deepcopy(snapshot[0]))
            if plan is None:
                resumed_optimizer = torch.optim.AdamW(resumed.parameters(), lr=0.04)
            else:
                resumed_optimizer = build_routed_torch_optimizer(resumed, plan, lr=0.04)
            resumed_optimizer.load_state_dict(copy.deepcopy(snapshot[1]))
            torch.random.set_rng_state(snapshot[2].clone())
            for _ in range(checkpoint + 1, updates + 1):
                resumed_optimizer.zero_grad()
                for _ in range(accumulation_steps):
                    indices = torch.randint(0, train_rows[0].shape[0], (microbatch_size,))
                    resumed_loss = torch.nn.functional.cross_entropy(
                        resumed(train_rows[0][indices], train_rows[1][indices]),
                        train_rows[2][indices],
                    )
                    (resumed_loss / accumulation_steps).backward()
                resumed_optimizer.step()
            resumed_model_hash = _state_hash(resumed.state_dict())
            resumed_optimizer_hash = _optimizer_hash(resumed_optimizer)
            resumed_rng = torch.random.get_rng_state().clone()
            recovery_receipt = RecoveryDrillReceipt(
                checkpoint,
                updates,
                uninterrupted_model_hash == resumed_model_hash,
                uninterrupted_optimizer_hash == resumed_optimizer_hash,
                bool(torch.equal(uninterrupted_rng, resumed_rng)),
                uninterrupted_model_hash,
                resumed_model_hash,
            )
        return strategy, model, optimizer, recovery_receipt

    local_receipt, _, _, _ = train_strategy(
        "local-window-adamw",
        "exact-near-only",
        local_tensors,
        local_test_tensors,
        local_active,
        routed=False,
    )
    graph_adam, _, _, _ = train_strategy(
        "graph-retrieval-adamw",
        "exact-near-plus-retrieved-far",
        graph_tensors,
        graph_test_tensors,
        graph_active,
        routed=False,
    )
    graph_routed, _, _, recovery_receipt = train_strategy(
        "graph-retrieval-routed",
        "exact-near-plus-retrieved-far",
        graph_tensors,
        graph_test_tensors,
        graph_active,
        routed=True,
        recovery=True,
    )
    if recovery_receipt is None:
        raise RuntimeError("recovery drill did not produce a receipt.")
    failure_receipts = (
        FailureReceipt(
            "graph-memory-pilot",
            "local-window-negative-control",
            FailureKind.QUALITY_REGRESSION,
            "held-out-accuracy-target",
            expected_failure=True,
            detected=local_receipt.test_accuracy < target_accuracy,
            observed="accuracy=%.6f target=%.6f" % (local_receipt.test_accuracy, target_accuracy),
        ),
        FailureReceipt(
            "graph-memory-pilot",
            "graph-retrieval-quality-control",
            FailureKind.QUALITY_REGRESSION,
            "held-out-accuracy-target",
            expected_failure=False,
            detected=graph_adam.test_accuracy < target_accuracy,
            observed="accuracy=%.6f target=%.6f" % (graph_adam.test_accuracy, target_accuracy),
        ),
        FailureReceipt(
            "graph-memory-pilot",
            "checkpoint-restart-control",
            FailureKind.REPLAY_MISMATCH,
            "model-optimizer-rng-fingerprints",
            expected_failure=False,
            detected=not recovery_receipt.passed,
            observed="restart parity passed=%s" % recovery_receipt.passed,
        ),
    )
    return GraphMemoryPilotReceipt(
        seed,
        source_nodes,
        local_receipt,
        graph_adam,
        graph_routed,
        recovery_receipt,
        failure_receipts,
    )


__all__ = [
    "GraphMemoryPilotReceipt",
    "PilotStrategyReceipt",
    "RecoveryDrillReceipt",
    "run_graph_memory_pilot",
]
