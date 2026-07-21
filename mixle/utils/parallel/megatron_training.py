"""Megatron Bridge adapter for full transformer and MoE parallelism.

Megatron owns the transformer partitioner and training loop.  Mixle owns the
statistical/parallel plan and maps it into a Bridge model provider without
importing Megatron unless this backend is selected.
"""

from __future__ import annotations

import importlib
from typing import Any

from mixle.utils.parallel.training_contracts import (
    BackendCapabilities,
    ParallelAxis,
    ParallelPlan,
    StepReceipt,
)


class MegatronBridgeSession:
    """A materialized Bridge model for callers that supply a custom batch step.

    Normal frontier pretraining should call :meth:`MegatronBridgeBackend.run_pretrain`;
    Bridge then owns schedules, inter-stage transfers, distributed optimizer
    overlap, and checkpoint cadence.  ``train_batch`` exists for integration
    tests and specialized loops and never silently substitutes a local step.
    """

    def __init__(
        self,
        model: Any,
        *,
        plan: ParallelPlan,
        capabilities: BackendCapabilities,
        train_batch_fn: Any = None,
        optimizer: Any = None,
        precision: str = "bf16",
    ) -> None:
        self.module = model
        self.plan = plan
        self.capabilities = capabilities
        self.train_batch_fn = train_batch_fn
        self.optimizer = optimizer
        self.precision = precision
        self.step = 0

    def train_batch(self, inputs: Any, targets: Any) -> StepReceipt:
        if self.train_batch_fn is None:
            raise RuntimeError(
                "Megatron Bridge owns its pretraining loop. Call backend.run_pretrain(config, forward_step), "
                "or provide train_batch_fn when preparing a specialized session."
            )
        result = self.train_batch_fn(self.module, self.optimizer, inputs, targets)
        if isinstance(result, StepReceipt):
            self.step = max(self.step, result.step)
            return result
        self.step += 1
        examples = int(getattr(inputs, "shape", (0,))[0])
        tokens = int(getattr(inputs, "numel", lambda: 0)())
        return StepReceipt(
            step=self.step,
            loss=float(result),
            local_examples=examples,
            local_tokens=tokens,
            microbatches=self.plan.microbatches,
            accumulation_steps=self.plan.gradient_accumulation_steps,
            data_parallel_size=self.plan.data_parallel_size,
            optimizer="megatron_distributed_optimizer",
            precision=self.precision,
        )

    def finish_accumulation(self) -> StepReceipt | None:
        return None

    def close(self) -> None:
        return None


class MegatronBridgeBackend:
    """Map a :class:`ParallelPlan` onto NVIDIA Megatron Bridge providers."""

    capabilities = BackendCapabilities(
        name="megatron",
        axes=frozenset(ParallelAxis),
        precisions=frozenset({"fp32", "fp16", "bf16", "fp8", "fp4"}),
        distributed_optimizer=True,
        reshardable_checkpoint=True,
        elastic_restart=False,
        requirements=("megatron-bridge", "CUDA", "NCCL"),
    )

    @staticmethod
    def configure_provider(provider: Any, plan: ParallelPlan) -> Any:
        """Apply the Mixle plan to a mutable Bridge model provider."""

        values = {
            "tensor_model_parallel_size": plan.tp,
            "pipeline_model_parallel_size": plan.pp,
            "context_parallel_size": plan.cp,
            "expert_model_parallel_size": plan.ep,
            "expert_tensor_parallel_size": plan.etp,
            "sequence_parallel": plan.tp > 1,
        }
        for name, value in values.items():
            setattr(provider, name, value)
        return provider

    @staticmethod
    def configure_distributed_optimizer(config: Any, plan: ParallelPlan) -> Any:
        """Enable sharded optimizer/parameter state and overlap on a Bridge config."""

        if config is None:
            return None
        values = {
            "use_distributed_optimizer": plan.dp_shard > 1,
            "overlap_grad_reduce": True,
            "overlap_param_gather": plan.dp_shard > 1,
        }
        for name, value in values.items():
            if hasattr(config, name):
                setattr(config, name, value)
        return config

    @staticmethod
    def _require_bridge() -> Any:
        try:
            return importlib.import_module("megatron.bridge")
        except ImportError as error:  # pragma: no cover - optional cluster dependency
            raise ImportError(
                "the Megatron backend requires NVIDIA Megatron Bridge; install it in the cluster image."
            ) from error

    def prepare(self, module: Any, *, plan: ParallelPlan, **kwargs: Any) -> MegatronBridgeSession:
        """Finalize a provider and construct its distributed model."""

        precision = str(kwargs.pop("precision", "bf16")).lower()
        self.capabilities.validate(plan, precision=precision)
        self._require_bridge()
        provider = self.configure_provider(module, plan)
        ddp_config = self.configure_distributed_optimizer(kwargs.pop("ddp_config", None), plan)
        finalize = getattr(provider, "finalize", None)
        model_config = finalize() if callable(finalize) else provider
        provide = getattr(provider, "provide_distributed_model", None)
        if not callable(provide):
            provide = getattr(model_config, "provide_distributed_model", None)
        if not callable(provide):
            raise TypeError("Megatron provider must expose provide_distributed_model().")
        provide_kwargs: dict[str, Any] = {"wrap_with_ddp": bool(kwargs.pop("wrap_with_ddp", True))}
        if ddp_config is not None:
            provide_kwargs["ddp_config"] = ddp_config
        model = provide(**provide_kwargs)
        session = MegatronBridgeSession(
            model,
            plan=plan,
            capabilities=self.capabilities,
            train_batch_fn=kwargs.pop("train_batch_fn", None),
            optimizer=kwargs.pop("optimizer", None),
            precision=precision,
        )
        if kwargs:
            raise TypeError("unexpected Megatron backend options: %s" % ", ".join(sorted(kwargs)))
        return session

    def run_pretrain(self, config: Any, forward_step: Any, *, plan: ParallelPlan | None = None) -> Any:
        """Delegate the full training loop to Bridge's public ``pretrain`` entry point."""

        if plan is not None:
            self.capabilities.validate(plan)
            model = getattr(config, "model", None)
            if model is not None:
                self.configure_provider(model, plan)
            self.configure_distributed_optimizer(getattr(config, "ddp", None), plan)
        self._require_bridge()
        module = importlib.import_module("megatron.bridge.training.pretrain")
        return module.pretrain(config, forward_step)


__all__ = ["MegatronBridgeBackend", "MegatronBridgeSession"]
