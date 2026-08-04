"""Verify the GPU-distilled student runs ON THE LAPTOP (CPU) and reproduces its accuracy.

The round-trip receipt: the student was feature-distilled from CLIP on a rented GPU, but it must run
here, on CPU, with no CLIP and no GPU -- carrying CLIP's vision geometry in a few MB. Loads the retrieved
weights, classifies CIFAR-10 test zero-shot through the frozen text head, and reports laptop accuracy +
latency next to the GPU-reported number.

Takeaway: the trust boundary is the interesting part. Three files arrive from a machine you do not
control, so each is authenticated by SHA-256 BEFORE it is opened, and the weights are read with
``weights_only=True`` so deserialization cannot execute code. A distilled artifact is only as good as
your ability to say which artifact you actually ran.

Prerequisites: ``student.pt``, ``student_head.pt``, and ``metrics.json`` produced by
``distill_clip_features.py``, placed next to this script, plus their expected digests obtained over the
same trusted channel as the release evidence. All three digest arguments are REQUIRED -- the script
deliberately has no "skip verification" mode.

Run: ``python examples/vision_edge_distillation/verify_on_laptop.py \\
        --student-sha256 <student.pt> --head-sha256 <student_head.pt> --metrics-sha256 <metrics.json>``
"""

import argparse
import hashlib
import json
import os
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

os.environ["HF_HUB_OFFLINE"] = "1"
HERE = os.path.dirname(os.path.abspath(__file__))
CIFAR10_ID = "uoft-cs/cifar10"
CIFAR10_REVISION = "0b2714987fa478483af9968de7c934580d0bb9a2"
CLIP_ID = "openai/clip-vit-base-patch32"
CLIP_REVISION = "3d74acf9a28c67741b2f4f2ea7635f0aaf6f0268"


def block(ci, co, stride=1):
    return nn.Sequential(nn.Conv2d(ci, co, 3, stride, 1, bias=False), nn.BatchNorm2d(co), nn.ReLU())


class Student(nn.Module):
    def __init__(self, dim=512):
        super().__init__()
        self.body = nn.Sequential(
            block(3, 64),
            block(64, 64),
            nn.MaxPool2d(2),
            block(64, 128),
            block(128, 128),
            nn.MaxPool2d(2),
            block(128, 256),
            block(256, 256),
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
        )
        self.proj = nn.Linear(256, dim)

    def forward(self, x):
        return F.normalize(self.proj(self.body(x)), dim=-1)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _authenticate(path: Path, expected: str) -> None:
    normalized = expected.lower()
    if len(normalized) != 64 or any(char not in "0123456789abcdef" for char in normalized):
        raise ValueError(f"expected SHA-256 for {path.name} must be 64 hexadecimal characters")
    actual = _sha256(path)
    if actual != normalized:
        raise ValueError(f"SHA-256 mismatch for {path.name}: expected {normalized}, received {actual}")


def _arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--student-sha256", required=True)
    parser.add_argument("--head-sha256", required=True)
    parser.add_argument("--metrics-sha256", required=True)
    return parser.parse_args()


def main():
    args = _arguments()
    student_path = Path(HERE, "student.pt")
    head_path = Path(HERE, "student_head.pt")
    metrics_path = Path(HERE, "metrics.json")
    _authenticate(student_path, args.student_sha256)
    _authenticate(head_path, args.head_sha256)
    _authenticate(metrics_path, args.metrics_sha256)

    with metrics_path.open(encoding="utf-8") as handle:
        metrics = json.load(handle)
    expected_assets = {
        "clip": {"repository": CLIP_ID, "revision": CLIP_REVISION},
        "cifar10": {
            "repository": CIFAR10_ID,
            "revision": CIFAR10_REVISION,
            "train_fingerprint": metrics.get("assets", {}).get("cifar10", {}).get("train_fingerprint"),
            "test_fingerprint": metrics.get("assets", {}).get("cifar10", {}).get("test_fingerprint"),
        },
    }
    if metrics.get("assets") != expected_assets:
        raise ValueError("metrics external-asset identities do not match the verifier's pinned contract")
    student = Student()
    student.load_state_dict(torch.load(student_path, map_location="cpu", weights_only=True))
    student.eval()
    head = torch.load(head_path, map_location="cpu", weights_only=True)
    mean, std, tfeat = head["mean"], head["std"], head["tfeat"]

    from datasets import load_dataset

    te = load_dataset(CIFAR10_ID, split="test[:2000]", revision=CIFAR10_REVISION)
    imgs = [np.array(r["img"]) for r in te]
    y = np.array([r["label"] for r in te])
    X = torch.tensor(np.stack(imgs), dtype=torch.float32).permute(0, 3, 1, 2).div(255)
    X = (X - mean) / std

    t0 = time.time()
    with torch.no_grad():
        emb = torch.cat([student(X[i : i + 256]) for i in range(0, len(X), 256)])
        pred = (emb @ tfeat.T).argmax(1).numpy()
    dt = time.time() - t0
    acc = float((pred == y).mean())
    npar = sum(p.numel() for p in student.parameters())
    print("=" * 66)
    print("ROUND-TRIP: a GPU-distilled student, verified on the laptop (CPU)")
    print("=" * 66)
    print(f"  student: {npar / 1e6:.2f}M params, {npar * 4 / 1e6:.1f} MB, torch (no CLIP, no GPU)")
    print(f"  GPU-reported CLIP zero-shot teacher : {metrics['clip_zero_shot_acc']:.4f}")
    print(f"  GPU-reported student accuracy       : {metrics['student_acc']:.4f}")
    print(f"  LAPTOP-verified student accuracy    : {acc:.4f}  ({len(y)} test imgs, {dt:.1f}s CPU)")
    print(f"  CIFAR-10 revision/fingerprint       : {CIFAR10_REVISION} / {te._fingerprint}")
    print(f"  student/teacher accuracy ratio        : {acc / metrics['clip_zero_shot_acc']:.4f}")
    print("  Local measurements above are not a 0.8.0 performance claim without a retained run receipt.")


if __name__ == "__main__":
    main()
