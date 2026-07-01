"""A learned embedding, declared once and shared across models -- so a mixture of LMs ties one word embedding.

If the per-cluster experts of a mixture-of-language-models each learn their own token vectors, they duplicate a
big parameter block and can't pool what a word *means* across the mixture. ``CategoricalEmbedding`` (``mixle.ppl.Embedding``
in the PPL) declares one embedding and hands the same module to every model that references it -- the neural
analogue of the PPL's ``name=`` scalar tying. This shows the mixture the README builds, but with the word
embedding shared across its experts.

Run: ``python shared_embedding_example.py``  (needs ``pip install "mixle[torch]"``).
"""

from __future__ import annotations

from mixle.models import CategoricalEmbedding, TransformerLMEstimator
from mixle.stats import MixtureEstimator


def n_params(module) -> int:
    return sum(p.numel() for p in module.parameters())


def unique_params(modules) -> int:
    seen, total = set(), 0
    for m in modules:
        for p in m.parameters():
            if id(p) not in seen:
                seen.add(id(p))
                total += p.numel()
    return total


def main() -> None:
    vocab, dim, n_experts = 8000, 256, 3

    def expert(embedding=None):
        return TransformerLMEstimator(vocab, d_model=dim, n_layer=4, block=64, embedding=embedding)

    print(f"a mixture of {n_experts} language-model experts over a {vocab}-word vocabulary\n")

    # independent experts: each learns its own word embedding
    indep = [expert() for _ in range(n_experts)]
    print("independent embeddings:")
    print(f"   total parameters: {unique_params(e.module for e in indep):,}")

    # shared embedding: one word embedding, tied across all experts
    emb = CategoricalEmbedding(vocab, dim, name="word")
    shared = [expert(emb) for _ in range(n_experts)]
    print("\nshared word embedding (mixle.ppl.Embedding):")
    print(f"   total parameters: {unique_params(e.module for e in shared):,}")
    tied = len({id(e.module.tok.weight) for e in shared}) == 1
    print(f"   all experts tie the same token vectors: {tied}")

    emb_params = vocab * dim
    print(f"\n   => the shared embedding removes {(n_experts - 1) * emb_params:,} duplicate parameters")
    print("      and lets every expert train (and pool) one set of word vectors.")

    # assemble the mixture exactly as in the README -- each expert is a generative language-model leaf
    est = MixtureEstimator(shared)
    print(f"\n   MixtureEstimator of {len(est.estimators)} shared-embedding LM experts, ready to fit with optimize().")


if __name__ == "__main__":
    main()
