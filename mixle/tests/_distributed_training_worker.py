"""Two-process worker used by distributed_training_torchrun_test.py."""

from __future__ import annotations

import json
import os
from pathlib import Path

import torch
import torch.distributed as dist

from mixle.models.transformer import build_causal_lm
from mixle.utils.parallel.torch_training import TorchDistributedBackend
from mixle.utils.parallel.training_contracts import ParallelPlan


def main() -> None:
    rank = int(os.environ["RANK"])
    torch.manual_seed(12)
    module = build_causal_lm(17, d_model=16, n_layer=1, n_head=2, block=4)
    session = TorchDistributedBackend().prepare(
        module,
        plan=ParallelPlan(dp_replicate=2, microbatches=2),
        device="cpu",
        optimizer="sgd",
        lr=0.01,
        max_grad_norm=None,
    )
    generator = torch.Generator().manual_seed(100 + rank)
    inputs = torch.randint(0, 17, (4, 4), generator=generator)
    targets = torch.randint(0, 17, (4, 4), generator=generator)
    receipt = session.train_batch(inputs, targets)
    flattened = torch.cat([parameter.detach().reshape(-1).cpu() for parameter in module.parameters()])
    gathered = [torch.empty_like(flattened) for _ in range(2)]
    dist.all_gather(gathered, flattened)
    equal = all(torch.equal(gathered[0], value) for value in gathered[1:])
    if rank == 0:
        Path(os.environ["MIXLE_RESULT_PATH"]).write_text(
            json.dumps(
                {
                    "parameters_equal": equal,
                    "step": receipt.step,
                    "global_examples": receipt.global_examples,
                    "global_tokens": receipt.global_tokens,
                }
            ),
            encoding="utf-8",
        )
    session.close()
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
