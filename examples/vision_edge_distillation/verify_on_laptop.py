"""Verify the GPU-distilled student runs ON THE LAPTOP (CPU) and reproduces its accuracy.

The round-trip receipt: the student was feature-distilled from CLIP on a rented GPU, but it must run
here, on CPU, with no CLIP and no GPU -- carrying CLIP's vision geometry in a few MB. Loads the retrieved
weights, classifies CIFAR-10 test zero-shot through the frozen text head, and reports laptop accuracy +
latency next to the GPU-reported number.
"""

import json
import os
import time

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

os.environ["HF_HUB_OFFLINE"] = "1"
HERE = os.path.dirname(os.path.abspath(__file__))


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


def main():
    metrics = json.load(open(os.path.join(HERE, "metrics.json")))
    student = Student()
    student.load_state_dict(torch.load(os.path.join(HERE, "student.pt"), map_location="cpu"))
    student.eval()
    head = torch.load(os.path.join(HERE, "student_head.pt"), map_location="cpu")
    mean, std, tfeat = head["mean"], head["std"], head["tfeat"]

    from datasets import load_dataset

    te = load_dataset("uoft-cs/cifar10", split="test[:2000]")
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
    print(
        f"  retention: {acc / metrics['clip_zero_shot_acc']:.0%} of CLIP's zero-shot, at "
        f"{605 / max(npar * 4 / 1e6, 0.1):.0f}x smaller than CLIP"
    )
    print("  (pooled-pixel student was 0.14; from-scratch 3k-image CNN was 0.40)")


if __name__ == "__main__":
    main()
