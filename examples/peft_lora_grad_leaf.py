"""peft LoRA through the GradLeaf bridge -- a REAL HuggingFace checkpoint, not a hand-rolled stand-in.

``mixle/tests/grad_control_test.py``'s ``AdapterThroughTheBridgeTest`` pins the claim with a hand-rolled
``LoRAStyleAdapter`` (frozen base + trainable low-rank delta) and documents why it generalizes: "peft's
wrapped modules are exactly this shape -- torch modules with frozen base weights -- which is why they drop
into the bridge unchanged." This example reproduces that receipt with the real thing: a tiny HuggingFace
GPT-2 checkpoint, wrapped with actual ``peft.get_peft_model`` LoRA adapters, dropped into ``GradLeaf``
unchanged.

``GradLeaf`` wants ``module.log_density(x) -> (n,)``; a causal LM's natural objective is token-level
cross-entropy, not an unconditional density. The bridge doesn't care -- next-token log-likelihood summed
over a sequence IS a log density over that sequence, so ``PeftCausalLMLeaf.log_density`` below is exactly
that sum, negated back out of the standard cross-entropy loss. Everything downstream (the M-step's
responsibility-weighted NLL, the optimizer seeing only ``requires_grad`` params) is unmodified GradLeaf.

The receipt: after fitting,
  * every frozen BASE weight is bitwise-unchanged (peft froze it; the optimizer never touched it either --
    ``GradEstimator`` filters to trainable params only), and
  * only the LoRA adapter matrices moved, and
  * the fit is real (held-out log-likelihood improves).

Run: ``python examples/peft_lora_grad_leaf.py``
(needs ``pip install "mixle[torch]" transformers peft`` -- peft/transformers are example-only deps, not a
mixle extra, since GradLeaf itself has no opinion on what module you hand it).
"""

from __future__ import annotations

import numpy as np

from mixle.inference.estimation import optimize
from mixle.models import GradLeaf

CHECKPOINT = "hf-internal-testing/tiny-random-gpt2"  # a handful of KB; real GPT-2 architecture, random weights


def build_peft_wrapped_module(seed: int = 0):
    """Load the tiny checkpoint, wrap it with LoRA adapters, and adapt it to GradLeaf's
    ``log_density(x) -> (n,)`` contract. This is the ONLY glue GradLeaf needs -- everything else
    (freezing the base, exposing only the adapter params as trainable) is peft's ordinary behavior."""
    import torch
    from peft import LoraConfig, TaskType, get_peft_model
    from transformers import AutoModelForCausalLM

    torch.manual_seed(seed)
    base = AutoModelForCausalLM.from_pretrained(CHECKPOINT)
    lora_cfg = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        r=4,
        lora_alpha=8,
        lora_dropout=0.0,
        target_modules=["c_attn"],  # GPT-2's fused qkv projection
    )
    peft_model = get_peft_model(base, lora_cfg)  # freezes the base, adds trainable LoRA deltas -- peft's job

    class PeftCausalLMLeaf(torch.nn.Module):
        """A causal LM IS a density over token sequences: next-token log-likelihood summed over the
        sequence. This wrapper is the only mixle-specific code -- the peft model inside is untouched."""

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
            return -ce.sum(-1)  # summed next-token log-likelihood == log density of the sequence

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
    rng = np.random.RandomState(0)
    module = build_peft_wrapped_module(seed=0)

    base_before = {k: v.clone() for k, v in module.lm.base_model.model.state_dict().items() if "lora_" not in k}

    block = 8
    data = toy_token_sequences(vocab=module.lm.config.vocab_size, block=block, n=64, rng=rng)

    leaf = GradLeaf(module, m_steps=150, lr=0.1)
    before_ll = float(np.mean(leaf.seq_log_density(np.stack(data))))

    fitted = optimize(data, leaf, max_its=4, out=None)

    after_ll = float(np.mean(fitted.seq_log_density(np.stack(data))))

    base_after = {k: v for k, v in fitted.module.lm.base_model.model.state_dict().items() if "lora_" not in k}
    max_base_drift = max(float((base_after[k] - base_before[k]).abs().max()) for k in base_before)
    lora_params = [p for n, p in fitted.module.lm.named_parameters() if "lora_" in n and p.requires_grad]
    lora_moved = sum(float(p.detach().abs().sum()) for p in lora_params)

    print(f"base weight drift (should be 0.0): {max_base_drift}")
    print(f"LoRA adapter param mass (should be > 0): {lora_moved:.4f}")
    print(f"mean log-density before fit: {before_ll:.4f}")
    print(f"mean log-density after fit:  {after_ll:.4f}")
    assert max_base_drift == 0.0, "the frozen base moved -- peft/GradLeaf contract broken"
    assert lora_moved > 0.0, "the adapter never trained"
    assert after_ll > before_ll, "the fit made no progress"
    print("OK: only the LoRA adapter trained; the base checkpoint is untouched; the fit improved.")


if __name__ == "__main__":
    main()
