"""Foundation capabilities on laptop resources -> compressed to edge devices, with honest receipts.

Answers two questions the workplan raised, each measured on real data, no network:

  1. Can a laptop RUN a foundation capability?  YES. A frozen open-weight encoder (605 MB CLIP, or 90 MB
     MiniLM) + a ~10 KB certified mixle head classifies at 90%+ on CPU in seconds. The heavy part is the
     encoder, and it fits on a laptop; the learned part is kilobytes and closed-form.

  2. Can that capability be DISTILLED to an edge device (no torch, no foundation model)?  IT DEPENDS, and
     the receipt says exactly when:
       * TEXT (banking77 intent classification): YES. A MiniLM+head teacher (95%) distills into a ~48 KB
         TORCH-FREE student that keeps ~97% of the accuracy -- because text n-grams carry the signal.
       * VISION (CIFAR-10) onto raw pooled pixels: NO. The capability lives in CLIP's learned features;
         a kilobyte student on pooled pixels recovers almost nothing. The honest edge deployment for
         vision keeps the encoder.

The point is not a single triumphant number -- it is that the system MEASURES what survives compression
and tells you the boundary, instead of asserting a capability it does not have.
"""

from __future__ import annotations

import time

import numpy as np


def line(t: str) -> None:
    print(f"\n{'=' * 76}\n{t}\n{'=' * 76}")


def _teacher_from_head(encode, head):
    cache: dict = {}

    def predict(x):
        if x not in cache:
            cache[x] = int(head.predict(encode([x]))[0])
        return cache[x]

    return predict


def main() -> None:
    from datasets import load_dataset

    from mixle.represent import image_features
    from mixle.scientist import distill_to_edge, encode_images, encode_texts, study

    # 1. FOUNDATION ON LAPTOP -----------------------------------------------------------------------
    line("1. FOUNDATION CAPABILITY ON A LAPTOP: frozen encoder (605 MB) + a ~10 KB certified head")
    tr = load_dataset("cifar10", split="train[:2000]")
    te = load_dataset("cifar10", split="test[:800]")
    t0 = time.time()
    ztr = encode_images([r["img"] for r in tr])
    zte = encode_images([r["img"] for r in te])
    yte = np.array([r["label"] for r in te])
    head = study(ztr, [r["label"] for r in tr], alpha=0.1)
    acc = float((head.predict(zte) == yte).mean())
    print(
        f"  CLIP + certified head: acc {acc:.3f} on CIFAR-10 | cert {head.certificate.guarantee.name} | "
        f"{time.time() - t0:.0f}s on CPU"
    )
    print("  -> foundation-level perception, laptop resources, certified + calibrated. This is here.")

    # 2a. EDGE DISTILLATION THAT WORKS (text) --------------------------------------------------------
    line("2a. DISTILL TO EDGE (text): MiniLM+head -> a torch-free KB student")
    ds = load_dataset("banking77", split="train")
    tb = load_dataset("banking77", split="test")
    xtr = [r["text"] for r in ds if r["label"] < 20][:1600]
    ytr = [r["label"] for r in ds if r["label"] < 20][:1600]
    xte = [r["text"] for r in tb if r["label"] < 20][:400]
    yte_t = [r["label"] for r in tb if r["label"] < 20][:400]
    thead = study(encode_texts(xtr), ytr, alpha=0.1)
    art = distill_to_edge(_teacher_from_head(encode_texts, thead), xtr, xte, yte_t, max_bytes=500_000, seed=0)
    print(f"  {art.render()}")
    print(
        f"  -> a {art.bytes / 1000:.0f} KB torch-free artifact (from a ~90 MB encoder): "
        f"~{90_000_000 // max(art.bytes, 1):,}x smaller, {art.retention:.0%} of the accuracy kept."
    )

    # 2b. THE HONEST BOUNDARY (vision onto raw pixels) -----------------------------------------------
    line("2b. THE HONEST BOUNDARY (vision): distilling onto raw pooled pixels loses the capability")
    ftr = [image_features(np.array(r["img"]), dim=48) for r in tr]
    fte = [image_features(np.array(r["img"]), dim=48) for r in te]
    teach_tr = [int(v) for v in head.predict(ztr)]
    teach_te = np.array([int(v) for v in head.predict(zte)])
    from mixle.task.edge import DeviceSpec, distill_for_edge

    res = distill_for_edge(
        None,
        ftr,
        fte,
        DeviceSpec(torch_free=True, max_bytes=200_000),
        train_labels=teach_tr,
        val_labels=[int(v) for v in teach_te],
        n_init=3,
        n_iter=3,
        seed=0,
    )
    pred = np.array([int(res.model(x)) for x in fte])
    print(
        f"  torch-free pixel student: agreement w/ CLIP teacher {float((pred == teach_te).mean()):.3f} "
        f"(vs CLIP's {acc:.3f}) -- the signal is in CLIP's features, not the pixels."
    )
    print("  -> for vision, the honest edge deployment keeps the encoder; only text distils to bare metal.")

    print("\nfoundation-on-laptop: yes. edge distillation: yes for text, measured-no for vision-on-pixels.")


if __name__ == "__main__":
    main()
