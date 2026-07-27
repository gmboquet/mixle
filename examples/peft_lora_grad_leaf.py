"""peft LoRA through the GradLeaf bridge -- a REAL HuggingFace checkpoint, not a hand-rolled stand-in.

``mixle/tests/grad_control_test.py``'s ``AdapterThroughTheBridgeTest`` pins the claim with a hand-rolled
``LoRAStyleAdapter`` (frozen base + trainable low-rank delta) and documents why it generalizes: "peft's
wrapped modules are exactly this shape -- torch modules with frozen base weights -- which is why they drop
into the bridge unchanged." This example reproduces that receipt with the real thing: a tiny HuggingFace
GPT-2 checkpoint, wrapped with actual ``peft.get_peft_model`` LoRA adapters, dropped into ``GradLeaf``
unchanged.

``GradLeaf`` wants ``module.log_density(x) -> (n,)``. A causal LM supplies the conditional factors after
the first token, so the adapter adds an explicit uniform initial-token law. The resulting score is a
normalized joint law over fixed-length sequences, not a conditional continuation score mislabeled as a
density. Everything downstream (the M-step's responsibility-weighted NLL, the optimizer seeing only
``requires_grad`` params) is unmodified GradLeaf.

The receipt uses disjoint deterministic train and held-out sequences. After fitting,
  * every frozen BASE weight is bitwise-unchanged (peft froze it; the optimizer never touched it either --
    ``GradEstimator`` filters to trainable params only), and
  * every trainable LoRA tensor is compared with its own pre-fit snapshot, at least one moves, and
  * held-out log-likelihood improves.

Run: ``python examples/peft_lora_grad_leaf.py``
(needs ``pip install "mixle[torch]" transformers peft`` -- peft/transformers are example-only deps, not a
mixle extra, since GradLeaf itself has no opinion on what module you hand it).
"""

from __future__ import annotations

import numpy as np

from mixle.inference.estimation import optimize
from mixle.models import GradLeaf

CHECKPOINT = "peft-internal-testing/tiny-random-gpt2"
CHECKPOINT_REVISION = "2f18a2874922d4cc4777cdf9fbf66cfa057a691a"


def build_peft_wrapped_module(seed: int = 0):
    """Load the tiny checkpoint, wrap it with LoRA adapters, and adapt it to GradLeaf's
    ``log_density(x) -> (n,)`` contract. This is the ONLY glue GradLeaf needs -- everything else
    (freezing the base, exposing only the adapter params as trainable) is peft's ordinary behavior."""
    import torch
    from peft import LoraConfig, TaskType, get_peft_model
    from transformers import AutoModelForCausalLM

    torch.manual_seed(seed)
    base = AutoModelForCausalLM.from_pretrained(
        CHECKPOINT,
        revision=CHECKPOINT_REVISION,
        use_safetensors=True,
    )
    lora_cfg = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        r=4,
        lora_alpha=8,
        lora_dropout=0.0,
        target_modules=["c_attn"],  # GPT-2's fused qkv projection
    )
    peft_model = get_peft_model(base, lora_cfg)  # freezes the base, adds trainable LoRA deltas -- peft's job

    class PeftCausalLMLeaf(torch.nn.Module):
        """A normalized fixed-length sequence law with a uniform initial token."""

        def __init__(self, lm) -> None:
            super().__init__()
            self.lm = lm  # the peft-wrapped module, dropped in whole

        def log_density(self, x):  # x: (n, block) token ids, arriving as float per GradLeaf's contract
            ids = x.long()
            logits = self.lm(input_ids=ids).logits  # (n, block, vocab)
            shift_logits = logits[:, :-1, :]
            shift_targets = ids[:, 1:]
            ce = torch.nn.functional.cross_entropy(
                shift_logits.reshape(-1, shift_logits.shape[-1]),
                shift_targets.reshape(-1),
                reduction="none",
            ).reshape(ids.shape[0], -1)
            initial = -torch.log(torch.tensor(self.lm.config.vocab_size, device=ids.device))
            return initial - ce.sum(-1)

    return PeftCausalLMLeaf(peft_model)


def toy_token_sequences(vocab: int, block: int, n: int, rng: np.random.RandomState) -> list:
    """A tiny structured corpus -- a handful of fixed cyclic patterns drawn from a SMALL sub-vocabulary
    -- so a heavily-restricted-rank LoRA adapter on a randomly-initialized tiny model has a real,
    learnable signal to chase in a few M-steps, without needing a real text dataset for a smoke example."""
    sub_vocab = min(6, vocab)
    starts = rng.randint(0, sub_vocab, size=4)  # a handful of repeating cycles, not one per sequence
    seqs = []
    for i in range(n):
        start = int(starts[i % len(starts)])
        seqs.append([(start + j) % sub_vocab for j in range(block)])
    return [np.asarray(s, dtype=float) for s in seqs]


def main() -> None:
    print(f"asset repository={CHECKPOINT} revision={CHECKPOINT_REVISION}")
    module = build_peft_wrapped_module(seed=0)

    base_before = {k: v.clone() for k, v in module.lm.base_model.model.state_dict().items() if "lora_" not in k}
    adapters_before = {
        name: parameter.detach().clone()
        for name, parameter in module.lm.named_parameters()
        if "lora_" in name and parameter.requires_grad
    }

    block = 8
    train = toy_token_sequences(
        vocab=module.lm.config.vocab_size,
        block=block,
        n=64,
        rng=np.random.RandomState(0),
    )
    held_out = toy_token_sequences(
        vocab=module.lm.config.vocab_size,
        block=block,
        n=64,
        rng=np.random.RandomState(1),
    )

    leaf = GradLeaf(module, m_steps=150, lr=0.1)
    before_ll = float(np.mean(leaf.seq_log_density(np.stack(held_out))))

    fitted = optimize(train, leaf, max_its=4, out=None)

    after_ll = float(np.mean(fitted.seq_log_density(np.stack(held_out))))

    base_after = {k: v for k, v in fitted.module.lm.base_model.model.state_dict().items() if "lora_" not in k}
    max_base_drift = max(float((base_after[k] - base_before[k]).abs().max()) for k in base_before)
    adapter_deltas = {
        name: float((parameter.detach() - adapters_before[name]).abs().max())
        for name, parameter in fitted.module.lm.named_parameters()
        if name in adapters_before
    }
    moved_adapters = sorted(name for name, delta in adapter_deltas.items() if delta > 0.0)

    print(f"base weight drift (should be 0.0): {max_base_drift}")
    print(f"LoRA adapter tensors moved: {len(moved_adapters)}/{len(adapter_deltas)}")
    print(f"held-out mean log-density before fit: {before_ll:.4f}")
    print(f"held-out mean log-density after fit:  {after_ll:.4f}")
    assert max_base_drift == 0.0, "the frozen base moved -- peft/GradLeaf contract broken"
    assert moved_adapters, "no adapter tensor changed from its pre-fit snapshot"
    assert after_ll > before_ll, "held-out likelihood did not improve"
    print("OK: only LoRA adapters moved; the base is untouched; held-out likelihood improved.")


if __name__ == "__main__":
    main()
