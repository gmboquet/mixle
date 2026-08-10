"""One embedding space for every modality -- and discrete tokens only if you want them, learned not guessed.

Classification: illustrative -- runs on small SYNTHETIC / STAND-IN data. The "image" and
"molecule" arrays are uniform random stand-ins and the "seismic" trace is planted Gaussian noise
carrying the class shift; only the plumbing and the held-out gate are the demonstration
(STAT-RR23-05).

A single record here carries four modalities at once: a caption (text), an image, a seismic trace, and a molecule
(a structure = a set of atom features). ``HeterogeneousEncoder`` embeds all of them into ONE shared space, so:

  * downstream: pool the unified stream into a task head and train the encoders to a label (objective B);
  * generative / discrete: fit a ``VectorQuantizer`` on the shared space to get a *learned cross-modal vocabulary*,
    then model the discrete token stream with a plain mixle model -- tokenization inferred, never hardcoded (objective A).

No modality commits to a vocabulary upstream. Run: ``python examples/heterogeneous_representation_example.py``
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
    # The class is REAL structure in the record, carried by two modalities at once: the text
    # names it (letters a/b vs c/d/e) and the seismic trace shifts with it. A learnable label
    # must be a function of the features -- an earlier revision labeled records by seed parity,
    # which no feature carries (STAT-RR22-04).
    rng = np.random.RandomState(seed)
    positive = seed % 5 < 2
    return {
        "text": "sample-" + "abcde"[seed % 5],
        "image": rng.rand(3, 8, 8).astype(np.float32),
        "seismic": (rng.randn(64) + (1.0 if positive else -1.0)).astype(np.float32),
        "molecule": rng.rand(rng.randint(4, 9), 5).astype(np.float32),
    }


def main() -> None:
    enc = build_encoder()
    stream, tags = enc.encode_numpy(record(0))
    print("one record, four modalities -> one shared space")
    print(f"   unified stream: {stream.shape[0]} units, each a {DIM}-vector; {len(set(tags))} modalities\n")

    print("objective B (downstream): train the encoders to a label by pooling the stream")

    # STAT-RR22-04: the label must be a FUNCTION OF THE FEATURES and the claim must be earned on
    # records the training never saw. An earlier revision labeled records by seed parity -- pure
    # RNG bookkeeping, unlearnable from any feature -- evaluated on its own 16 training rows, and
    # printed "the encoders learned the task" at 94-100% while fresh-record accuracy measured
    # 50.5-52.5% (chance). The label here is carried by the record itself (class-named text AND a
    # class-shifted seismic trace), training uses 64 records, and the printed claim is GATED on
    # 100 fresh records the optimizer never touched.
    def label(seed: int) -> int:
        return 1 if seed % 5 < 2 else 0

    head = torch.nn.Linear(DIM, 2)
    opt = torch.optim.Adam(enc.parameters() + list(head.parameters()), lr=1e-2)
    recs = [record(i) for i in range(64)]
    y = torch.tensor([label(i) for i in range(64)])
    for step in range(120):
        opt.zero_grad()
        pooled = torch.stack([enc.encode(r)[0].mean(dim=0) for r in recs])
        loss = torch.nn.functional.cross_entropy(head(pooled), y)
        loss.backward()
        opt.step()
    with torch.no_grad():
        train_acc = (head(torch.stack([enc.encode(r)[0].mean(dim=0) for r in recs])).argmax(1) == y).float().mean()
        fresh = [record(i) for i in range(100, 200)]
        fresh_y = torch.tensor([label(i) for i in range(100, 200)])
        fresh_acc = float(
            (head(torch.stack([enc.encode(r)[0].mean(dim=0) for r in fresh])).argmax(1) == fresh_y).float().mean()
        )
        final_loss = float(loss.detach())
    print(f"   loss {final_loss:.3f}, TRAINING accuracy {float(train_acc):.2f} (64 rows; fit, not evidence)")
    print(f"   HELD-OUT accuracy on 100 fresh records: {fresh_acc:.2f}")
    if fresh_acc >= 0.8:
        print("   -> the encoders learned the task: the class carried by text+seismic transfers to unseen records\n")
    else:
        raise RuntimeError(
            f"acceptance failed: held-out accuracy {fresh_acc:.2f} < 0.8 -- training fit alone is "
            "memorization, not task learning (STAT-RR22-04)"
        )

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
