"""One embedding space for every modality -- and discrete tokens only if you want them, learned not guessed.

A single record here carries four modalities at once: a caption (text), an image, a seismic trace, and a molecule
(a structure = a set of atom features). ``HeterogeneousEncoder`` embeds all of them into ONE shared space, so:

  * downstream: pool the unified stream into a task head and train the encoders to a label (objective B);
  * generative / discrete: fit a ``VectorQuantizer`` on the shared space to get a *learned cross-modal vocabulary*,
    then model the discrete token stream with a plain mixle model -- tokenization inferred, never hardcoded (objective A).

No modality commits to a vocabulary upstream. Run: ``python heterogeneous_representation_example.py``
(needs ``pip install "mixle[torch]"``).
"""

from __future__ import annotations

import numpy as np
import torch

from mixle.represent import (
    ByteSegmenter,
    CategoricalEmbedding,
    FeatureEmbedding,
    HeterogeneousEncoder,
    PatchSegmenter,
    SetSegmenter,
    VectorQuantizer,
    WindowSegmenter,
)

DIM = 32


def build_encoder() -> HeterogeneousEncoder:
    enc = HeterogeneousEncoder(dim=DIM)
    enc.register("text", ByteSegmenter(), CategoricalEmbedding(256, DIM))  # discrete bytes -> lookup
    enc.register("image", PatchSegmenter(patch=4), FeatureEmbedding(3 * 4 * 4, DIM))  # patches -> encoder
    enc.register("seismic", WindowSegmenter(window=16, hop=16), FeatureEmbedding(16, DIM))  # windows -> encoder
    enc.register("molecule", SetSegmenter(), FeatureEmbedding(5, DIM))  # atom-feature set -> encoder
    return enc


def record(seed: int) -> dict:
    rng = np.random.RandomState(seed)
    return {
        "text": "sample-" + "abcde"[seed % 5],
        "image": rng.rand(3, 8, 8).astype(np.float32),
        "seismic": rng.randn(64).astype(np.float32),
        "molecule": rng.rand(rng.randint(4, 9), 5).astype(np.float32),
    }


def main() -> None:
    enc = build_encoder()
    stream, tags = enc.encode_numpy(record(0))
    print("one record, four modalities -> one shared space")
    print(f"   unified stream: {stream.shape[0]} units, each a {DIM}-vector; {len(set(tags))} modalities\n")

    print("objective B (downstream): train the encoders to a label by pooling the stream")
    head = torch.nn.Linear(DIM, 2)
    opt = torch.optim.Adam(enc.parameters() + list(head.parameters()), lr=1e-2)
    recs = [record(i) for i in range(16)]
    y = torch.tensor([i % 2 for i in range(16)])
    for step in range(30):
        opt.zero_grad()
        pooled = torch.stack([enc.encode(r)[0].mean(dim=0) for r in recs])
        loss = torch.nn.functional.cross_entropy(head(pooled), y)
        loss.backward()
        opt.step()
    with torch.no_grad():
        acc = (head(torch.stack([enc.encode(r)[0].mean(dim=0) for r in recs])).argmax(1) == y).float().mean()
        final_loss = float(loss.detach())
    print(f"   loss {final_loss:.3f}, train accuracy {float(acc):.2f} -- the encoders learned the task\n")

    print("objective A (generative / discrete): learn a cross-modal vocabulary, then model the token stream")
    big = np.vstack([enc.encode_numpy(r)[0] for r in recs])  # all units across all records + modalities
    vq = VectorQuantizer(num_codes=16, dim=DIM, seed=0).fit(big)
    stream99, _tags99 = enc.encode_numpy(record(99))
    tokens = vq.quantize(stream99)
    print(f"   record 99 -> discrete tokens: {tokens.tolist()}")
    print(f"   codebook reconstruction error: {vq.reconstruction_error(big):.3f}")

    # a plain mixle model over the learned token vocabulary -- a Markov chain over the cross-modal tokens
    from mixle.inference import optimize
    from mixle.stats import MarkovChainEstimator

    token_seqs = [[int(t) for t in vq.quantize(enc.encode_numpy(r)[0])] for r in recs]
    chain = optimize(token_seqs, MarkovChainEstimator(), max_its=10, out=None)
    print(f"   fit a Markov chain over the {vq.num_codes}-token cross-modal vocabulary: {type(chain).__name__}")
    print("   => tokenization is a fitted model in the shared space, not a hardcoded upstream vocabulary.")


if __name__ == "__main__":
    main()
