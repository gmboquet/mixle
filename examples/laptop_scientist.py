"""The laptop scientist: cross-modal perception + certified learning + verified reasoning, assembled.

Run it: ``python examples/laptop_scientist.py``. CPU only, a couple of minutes. Needs network on the FIRST
run: it downloads the CIFAR-10 dataset and the CLIP / local-LLM open weights from Hugging Face (cached
thereafter, so later runs are offline). Three acts, each a RECEIPT -- a measured claim, next to the named
alternative it beats:

  ACT 1  PERCEIVE + LEARN   real CLIP image features -> a closed-form certified head on CIFAR-10, vs a
                            torch CNN given the SAME wall-clock budget. mixle wins on accuracy AND ships
                            a certificate, calibration, and abstention the CNN cannot.
  ACT 2  REASON + VERIFY    grounded scientific QA through the local LLM, vs the same LLM raw: the
                            scientist answers what it can cite and ABSTAINS otherwise; raw, the LLM
                            confidently hallucinates.
  ACT 3  INVERT + QUANTIFY  a physics parameter recovered with calibrated coverage -- the UQ a frontier
                            LLM cannot give you at all.

The thesis: not open-ended generative parity with giant models, but TRUSTWORTHY answers where the answer
can be checked -- perception, certified inference, and honest uncertainty, on a laptop.
"""

from __future__ import annotations

import time

import numpy as np


def line(t: str) -> None:
    print(f"\n{'=' * 76}\n{t}\n{'=' * 76}")


def act1_perceive_and_learn() -> None:
    from datasets import load_dataset

    from mixle.scientist import Scientist, encode_images

    line("ACT 1 -- PERCEIVE (real CLIP) + LEARN (certified closed-form head) on CIFAR-10")
    tr = load_dataset("cifar10", split="train[:2000]")
    te = load_dataset("cifar10", split="test[:1000]")
    yte = np.array([r["label"] for r in te])

    t0 = time.time()
    ztr = encode_images([r["img"] for r in tr])
    zte = encode_images([r["img"] for r in te])
    m = Scientist.study(ztr, [r["label"] for r in tr], alpha=0.1)
    total = time.time() - t0

    acc = float((m.predict(zte) == yte).mean())
    sets = m.prediction_sets(zte)
    cov = float(np.mean([y in s for y, s in zip(yte, sets)]))
    conf = ~m.abstains(zte)
    print(
        f"  mixle:  acc {acc:.3f} | certificate {m.certificate.guarantee.name} | "
        f"conformal coverage {cov:.3f} | abstains {float((~conf).mean()):.1%}"
    )
    print(
        f"          acc when confident {float((m.predict(zte)[conf] == yte[conf]).mean()):.3f} | "
        f"total wall-clock {total:.0f}s (encode + fit)"
    )

    # named competition: a CNN trained end-to-end under the SAME budget
    _cnn_baseline(tr, te, yte, budget=total)


def _cnn_baseline(tr, te, yte, *, budget: float) -> None:
    import torch
    import torch.nn as nn

    torch.manual_seed(0)
    X = torch.tensor(np.stack([np.array(r["img"]) for r in tr]), dtype=torch.float32).permute(0, 3, 1, 2) / 255.0
    Xt = torch.tensor(np.stack([np.array(r["img"]) for r in te]), dtype=torch.float32).permute(0, 3, 1, 2) / 255.0
    y = torch.tensor([r["label"] for r in tr])
    cnn = nn.Sequential(
        nn.Conv2d(3, 32, 3, padding=1),
        nn.ReLU(),
        nn.MaxPool2d(2),
        nn.Conv2d(32, 64, 3, padding=1),
        nn.ReLU(),
        nn.MaxPool2d(2),
        nn.Conv2d(64, 128, 3, padding=1),
        nn.ReLU(),
        nn.AdaptiveAvgPool2d(1),
        nn.Flatten(),
        nn.Linear(128, 10),
    )
    opt = torch.optim.Adam(cnn.parameters(), lr=1e-3)
    lossf = nn.CrossEntropyLoss()
    t0 = time.time()
    epochs = 0
    while time.time() - t0 < budget:
        perm = torch.randperm(len(X))
        for i in range(0, len(X), 64):
            idx = perm[i : i + 64]
            opt.zero_grad()
            lossf(cnn(X[idx]), y[idx]).backward()
            opt.step()
            if time.time() - t0 > budget:
                break
        epochs += 1
    cnn.eval()
    with torch.no_grad():
        acc = float((cnn(Xt).argmax(1).numpy() == yte).mean())
    print(
        f"  CNN (ADAM, same {budget:.0f}s): acc {acc:.3f} in {epochs} epochs | "
        f"certificate NONE (HEURISTIC) | no calibration | no abstention"
    )


def act2_reason_and_verify() -> None:
    from mixle.scientist import Scientist, generate

    line("ACT 2 -- REASON through the local LLM, VERIFIED against a knowledge base")
    corpus = [
        "Uranium-238 decays to lead-206 with a half-life of 4.468 billion years, the basis of U-Pb dating.",
        "Zircon crystals incorporate uranium but reject lead at crystallization, so measured lead is radiogenic.",
        "The Cretaceous-Paleogene (K-Pg) boundary is dated to approximately 66.0 million years ago.",
        "Carbon-14 has a half-life of 5730 years, useful for dating materials younger than 50,000 years.",
    ]
    sci = Scientist()
    sci.learn(corpus)
    supported = ["what is the half-life of uranium-238", "when is the K-Pg boundary dated to"]
    unsupported = ["what is the boiling point of tungsten", "who discovered the electron"]
    print("  scientist (grounded + verified):")
    for q in supported + unsupported:
        inv = sci.ask(q)
        if inv.abstained:
            print(f"    ABSTAIN  {q}")
        else:
            print(f"    ANSWER   {q}\n             -> {inv.answer}  [grounded {inv.factuality.grounded_fraction}]")
    print("\n  the SAME local LLM, raw (no grounding) on the unsupported questions:")
    for q in unsupported:
        print(f"    {q} -> {generate(q + '? Answer in one sentence.', max_new_tokens=32)}")


def act3_invert_and_quantify() -> None:
    line("ACT 3 -- INVERT a physics parameter with CALIBRATED uncertainty (UQ an LLM cannot give)")
    import runpy
    import sys
    from pathlib import Path

    sys.path.insert(0, str(Path(__file__).resolve().parent))
    mod = runpy.run_path(str(Path(__file__).resolve().parent / "flagship_physics_inverse.py"))
    k_true = mod["K_TRUE"]
    hits = 0
    for s in range(8):
        d, _ = mod["infer_k"](mod["observe"](s), s, draws=600)
        lo, hi = np.quantile(d, [0.05, 0.95])
        hits += int(lo <= k_true <= hi)
    print(f"  recovered decay rate k (true {k_true}); 90% credible interval bracketed the truth {hits}/8 times.")
    print("  a frontier LLM asked 'what is k' gives a number with no interval and no coverage guarantee.")


def main() -> None:
    act1_perceive_and_learn()
    act2_reason_and_verify()
    act3_invert_and_quantify()
    print("\nperception (real encoders) + certified inference + verified reasoning + honest UQ, on a laptop.")


if __name__ == "__main__":
    main()
