"""Explore foundation encoders and edge students with measurements from the current run.

Answers two questions the workplan raised, each measured on real data. Needs network on the FIRST run: it
downloads the CIFAR-10 / banking77 datasets and the CLIP / MiniLM open weights from Hugging Face (cached
thereafter, so later runs are offline):

The script prints raw accuracy, size, and runtime measurements. Those values are environment-dependent
example output, not 0.8.0 performance claims. Treat them as evidence only when the exact model,
dataset, dependencies, hardware, command, and output are retained together.
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
    line("1. FOUNDATION ENCODER + CERTIFIED HEAD")
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
    print("  -> current-run measurement; retain provenance before drawing a deployment conclusion.")

    # 2a. EDGE DISTILLATION THAT WORKS (text) --------------------------------------------------------
    line("2a. TEXT STUDENT: MiniLM+head -> a torch-free artifact")
    ds = load_dataset("banking77", split="train")
    tb = load_dataset("banking77", split="test")
    xtr = [r["text"] for r in ds if r["label"] < 20][:1600]
    ytr = [r["label"] for r in ds if r["label"] < 20][:1600]
    xte = [r["text"] for r in tb if r["label"] < 20][:400]
    yte_t = [r["label"] for r in tb if r["label"] < 20][:400]
    thead = study(encode_texts(xtr), ytr, alpha=0.1)
    art = distill_to_edge(_teacher_from_head(encode_texts, thead), xtr, xte, yte_t, max_bytes=500_000, seed=0)
    print(f"  {art.render()}")
    print(f"  -> artifact bytes {art.bytes}; measured accuracy retention {art.retention:.4f}.")

    # 2b. THE HONEST BOUNDARY (vision onto raw pixels) -----------------------------------------------
    line("2b. VISION STUDENT ON RAW POOLED PIXELS")
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
    print("  -> compare this measured agreement with the deployment requirement.")
    print("     vision_edge_distillation/ demonstrates a separately verifiable feature-distillation path.")

    print(
        "\nRun complete. Preserve exact inputs, environment, command, and output before citing a result."
    )


if __name__ == "__main__":
    main()
